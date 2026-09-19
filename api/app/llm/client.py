"""Ollama client with schema-constrained JSON output.

The single most important idea in this codebase lives here.

Normally you ask a model for JSON and hope. You write "respond only with valid
JSON" in the prompt, then defensively parse whatever comes back, strip markdown
fences, and retry when it apologises instead of answering.

Ollama supports something strictly stronger: pass a JSON Schema as `format`, and
it constrains *token sampling itself*. At each step the sampler masks out every
token that could not continue a valid document under that schema. Invalid output
is not rejected after the fact - it is never generated.

Proven on this machine before any of this was written:

    schema WITHOUT enum -> model returned  "Limitation"   (invented; not a type we handle)
    schema WITH    enum -> model returned  "exclusion"    (forced into our taxonomy)

Two consequences the rest of the project leans on:

1. Every categorical field MUST carry an `enum`. An unconstrained string field is
   an invitation for the model to invent a value that then flows into scoring.
2. Citation clause IDs get an `enum` of exactly the clause IDs supplied in that
   prompt. A hallucinated citation therefore becomes *unrepresentable* rather
   than merely unlikely - a grammar-level guarantee, not a prompt-level plea.
"""

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from app.config import settings
from app.llm import cache

log = logging.getLogger(__name__)

# How many requests have actually been sent to the model in this process, as
# opposed to answered from the cache.
#
# The eval harness reads this before and after each case to learn whether that
# case's answer was GENERATED or REPLAYED. That distinction is what lets a
# prompt change be measured against only the cases it touched: a replayed
# answer is the exact stored bytes of an earlier run, so it cannot have moved,
# while a generated one is a fresh sample from a model that does not answer
# identically twice.
#
# Counted here rather than inferred from the cache's row count, because the
# row count lies in one case. A response truncated mid-JSON is retried with a
# larger `num_predict` and stored under a key built from THOSE options - a key
# the next lookup never asks for. On the next run that request misses, is
# regenerated, and overwrites its own row, so the table does not grow and a
# regenerated answer would look replayed.
model_calls = 0

# Prompt tokens Ollama reported for the most recent request it answered.
#
# Every size in this project's context budgeting is an ESTIMATE made in
# characters (`scenario.CHARS_PER_TOKEN`), and the reservation for the system
# prompt and answer was sized once, in M5, when that prompt was much shorter.
# Ollama truncates an over-long prompt silently - that was Failure 10 - so an
# estimate that has drifted would not announce itself. This is the measured
# number, recorded so an eval can report it and checked on every call.
last_prompt_tokens: int | None = None


class LlmError(RuntimeError):
    """Raised when the model cannot produce usable output after retries."""


async def complete_json(
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    *,
    model: str | None = None,
    use_cache: bool = True,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Call the model and return parsed JSON conforming to `schema`.

    Retries exist for transport faults and for the rare case where constrained
    decoding hits the token limit mid-document, producing truncated JSON.
    """
    model = model or settings.model
    options = _decode_options()

    if use_cache:
        version = await runtime_version()
        key = cache.make_key(model, messages, schema, options, settings.kv_cache_type, version)
        if (hit := cache.get(key)) is not None:
            log.debug("llm cache hit %s", key[:12])
            return hit

    global model_calls
    model_calls += 1

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = await _post_chat(messages, schema, model, options)
            parsed = json.loads(raw)
            if use_cache:
                # Keyed on the options that actually produced this answer, which
                # may not be the ones we started with (see the escalation below).
                cache.put(
                    cache.make_key(
                        model, messages, schema, options, settings.kv_cache_type, version
                    ),
                    model,
                    parsed,
                )
            return parsed

        except json.JSONDecodeError as exc:
            # Constrained decoding guarantees valid *prefixes*, but generation
            # that hits the token ceiling still ends mid-structure. That is
            # exactly what this is: an unterminated string, not a malformed one.
            #
            # RETRYING THIS UNCHANGED IS POINTLESS. temperature is 0, so the
            # same prompt yields the same tokens every time - three identical
            # attempts at an identical truncation, which is what happened
            # before this branch existed. A retry is only worth making if
            # something about the request changes.
            #
            # So the retry raises the ceiling instead of repeating the request.
            last_error = exc
            options = {**options, "num_predict": options["num_predict"] * 2}
            log.warning(
                "attempt %d: response truncated mid-JSON; retrying with "
                "num_predict=%d. If this recurs, raise `num_predict` in config - "
                "the configured ceiling is too low for this schema.",
                attempt,
                options["num_predict"],
            )

        except httpx.HTTPError as exc:
            # Transport failures ARE worth repeating unchanged: a timeout or a
            # dropped connection is not a property of the prompt.
            last_error = exc
            log.warning("attempt %d: transport error: %s", attempt, exc)

        if attempt < max_attempts:
            await asyncio.sleep(2**(attempt - 1))  # 1s, 2s

    # The exception TYPE is included deliberately. httpx.ReadTimeout stringifies
    # to an empty string, so an earlier version of this message read
    # "failed after 3 attempts: " with nothing after the colon - which said
    # nothing at all about a failure that had taken 21 minutes.
    raise LlmError(
        f"{model} failed after {max_attempts} attempts: "
        f"{type(last_error).__name__}: {last_error or '(no message)'}"
    ) from last_error


# The Ollama version, and when it was read: (time.monotonic(), version).
#
# Remembered for a minute, because both simpler choices were wrong. Asked on
# every call, it made each cache hit take 0.8s instead of milliseconds - a new
# HTTP client costs about 0.2s, and "localhost" about 0.25s more while Windows
# tries IPv6 first. Remembered for the whole process, it would go stale when
# Ollama updates itself under an app that has been running for hours. With a
# minute, the only answer that can be filed under the old version is one
# generated in the first minute after an update, by a process already running.
_VERSION_TTL_SECONDS = 60.0
_version: tuple[float, str] | None = None


async def runtime_version() -> str:
    """The version of Ollama answering requests, as the server reports it.

    Part of every cache key (see cache.make_key), because an Ollama update alone
    changes answers (M17).
    """
    global _version
    now = time.monotonic()
    if _version is not None and now - _version[0] < _VERSION_TTL_SECONDS:
        return _version[1]
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get(f"{settings.ollama_url}/api/version")
            resp.raise_for_status()
            version = resp.json()["version"]
    except httpx.HTTPError as exc:
        raise LlmError(
            f"could not read the Ollama version from {settings.ollama_url}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    _version = (now, version)
    return version


def _decode_options() -> dict[str, Any]:
    """Decoding parameters, in one place.

    Built here rather than inline so the exact dict that is sent is also the
    dict that is hashed into the cache key - if the two were constructed
    separately they could drift, and a cache entry would then be keyed on
    settings the request never actually used.
    """
    return {
        "temperature": settings.temperature,
        # Pinned, not omitted. A system whose premise is reproducibility should
        # not leave an input unspecified - but see config.seed: this was
        # measured to fix nothing on its own.
        "seed": settings.seed,
        # Without an explicit window Ollama applies its own 4,096 default, and
        # truncates anything longer silently (see _check_context).
        "num_ctx": settings.num_ctx,
        # Bounded output: an uncapped generation ran past three consecutive
        # 420s timeouts when the model began repeating itself.
        "num_predict": settings.num_predict,
    }


async def _post_chat(
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    model: str,
    options: dict[str, Any],
) -> str:
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        resp = await client.post(
            f"{settings.ollama_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                # The whole point: this is a grammar constraint, not a hint.
                "format": schema,
                "options": options,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        _check_context(data.get("prompt_eval_count"), options)
        return data["message"]["content"]


def truncated_prompt_tokens(num_ctx: int) -> int:
    """The token count Ollama reports for a prompt it had to truncate.

    Measured in M16 on Ollama 0.34: every prompt longer than the window -
    policy text, plain prose, a column of numbers, a real system-plus-user
    scenario prompt, at windows of 8,192, 16,384 and 24,576 - reported exactly
    half the window plus two. A behaviour of one runtime version, not a law,
    which is why a test re-measures it against the live server.
    """
    return num_ctx // 2 + 2


def _check_context(prompt_tokens: int | None, options: dict[str, Any]) -> None:
    """Refuse a truncated prompt; warn when a prompt leaves the answer no margin.

    Two different failures, reported as what they are.

    TRUNCATION. The M15 version of this check assumed a truncated prompt would
    report a count at the window's size. It does not: stepping one prompt
    across an 8,192-token window, 33,000 characters reported 8,115 tokens and
    34,000 reported 4,098. Ollama discards half the context on overflow, so the
    count FALLS, and the old check read the one case it existed for as a small,
    healthy prompt.

    A ratio of characters to tokens was tried next and measured wrong both
    ways: a truncated column of numbers reported 4.9 characters per token,
    inside the range of intact policy text, and an intact run of long words
    reported 7.9. The exact count is the signal that held in every case.

    It raises rather than warns. A truncated scenario prompt means an answer
    reasoned over part of the policy with nothing to say so, and retrying is
    pointless: the same prompt truncates the same way.

    MARGIN. A prompt that fits but leaves less than num_predict has lost the
    answer's room: generation past the window makes Ollama discard context
    mid-answer. That one is a warning - most answers are far shorter than the
    ceiling.
    """
    global last_prompt_tokens
    last_prompt_tokens = prompt_tokens
    if prompt_tokens is None:
        return
    if prompt_tokens == truncated_prompt_tokens(options["num_ctx"]):
        raise LlmError(
            f"Ollama reported {prompt_tokens} prompt tokens, half the {options['num_ctx']}-token "
            f"context window: the prompt was longer than the window and was truncated"
        )
    if prompt_tokens + options["num_predict"] > options["num_ctx"]:
        log.warning(
            "prompt used %d of %d context tokens, leaving %d for an answer capped at "
            "num_predict=%d",
            prompt_tokens, options["num_ctx"], options["num_ctx"] - prompt_tokens,
            options["num_predict"],
        )


async def health() -> dict[str, Any]:
    """Check Ollama is reachable and report whether our model is present."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        version = (await client.get(f"{settings.ollama_url}/api/version")).json()
        tags = (await client.get(f"{settings.ollama_url}/api/tags")).json()
    available = [m["name"] for m in tags.get("models", [])]
    return {
        "ollama_version": version.get("version"),
        "model": settings.model,
        "model_available": settings.model in available,
        "models_installed": available,
    }


async def warm() -> None:
    """Preload the model into VRAM.

    Cold start was measured at ~27s on this machine. Paying it once at API
    startup means the first user upload doesn't appear to hang.
    """
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            # An empty message list asks Ollama to load the model and stop.
            await client.post(
                f"{settings.ollama_url}/api/chat",
                json={"model": settings.model, "messages": [], "stream": False},
            )
        log.info("warmed %s", settings.model)
    except httpx.HTTPError as exc:
        # Never block startup on this - the app is still usable, just slower on
        # the first request, and /health will report the real problem.
        log.warning("could not warm model: %s", exc)
