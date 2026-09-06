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
from typing import Any

import httpx

from app.config import settings
from app.llm import cache

log = logging.getLogger(__name__)


class LlmError(RuntimeError):
    """Raised when the model cannot produce usable output after retries."""


async def complete_json(
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    prompt_version: str,
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

    key = cache.make_key(model, prompt_version, messages, schema, options)
    if use_cache:
        if (hit := cache.get(key)) is not None:
            log.debug("llm cache hit %s", key[:12])
            return hit

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = await _post_chat(messages, schema, model, options)
            parsed = json.loads(raw)
            if use_cache:
                # Keyed on the options that actually produced this answer, which
                # may not be the ones we started with (see the escalation below).
                cache.put(
                    cache.make_key(model, prompt_version, messages, schema, options),
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


def _decode_options() -> dict[str, Any]:
    """Decoding parameters, in one place.

    Built here rather than inline so the exact dict that is sent is also the
    dict that is hashed into the cache key - if the two were constructed
    separately they could drift, and a cache entry would then be keyed on
    settings the request never actually used.
    """
    return {
        "temperature": settings.temperature,
        # Without an explicit window Ollama applies its own 4,096 default, well
        # under this project's 32k assumption, and truncates silently.
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
        return resp.json()["message"]["content"]


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
