"""M0 verification gate.

These tests hit a real Ollama server rather than a mock, on purpose. The claim
being verified - "the sampler physically cannot emit a value outside our enum" -
is a claim about Ollama's behaviour. A mock would only test that we believe it.

Run without a model:  pytest -m "not llm"
"""

import time

import pytest

from app.llm import cache, client
from app.taxonomy import ClauseType

# A real clause from an Indian health policy. It is deliberately a hard case:
# it reads like an exclusion ("no claim shall be payable") but is really a
# waiting period, because coverage does begin later.
PRE_EXISTING_CLAUSE = (
    "No claim shall be payable for any treatment arising from a pre-existing "
    "disease during the first 36 months of continuous coverage."
)

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        # The enum is the load-bearing part of this whole file.
        "clause_type": {"type": "string", "enum": ClauseType.values()},
        "plain_language": {"type": "string"},
        "waiting_months": {"type": "integer"},
    },
    "required": ["clause_type", "plain_language", "waiting_months"],
}


def _messages(clause: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You classify Indian health insurance policy clauses.",
        },
        {"role": "user", "content": f"Classify this clause:\n\n{clause}"},
    ]


@pytest.mark.llm
async def test_ollama_is_reachable_with_our_model():
    health = await client.health()
    assert health["ollama_version"], "Ollama is not running"
    assert health["model_available"], (
        f"{health['model']} not installed. Run: ollama pull {health['model']}"
    )


@pytest.mark.llm
async def test_constrained_output_stays_inside_our_taxonomy():
    """The core guarantee: output is confined to values we know how to handle."""
    result = await client.complete_json(
        _messages(PRE_EXISTING_CLAUSE),
        CLASSIFY_SCHEMA,
        use_cache=False,
    )

    # Not "is it a plausible string" - is it one of OUR strings. Anything else
    # would flow into the scoring formula as an unknown key.
    assert result["clause_type"] in ClauseType.values()
    assert isinstance(result["waiting_months"], int)
    assert result["plain_language"].strip()


@pytest.mark.llm
async def test_schema_forces_a_value_even_for_an_unrelated_clause():
    """Constrained decoding cannot abstain - which is exactly why abstention has
    to be an explicit enum member wherever we need it (see Verdict).

    Here we hand the model something with no sensible clause type at all. It
    still cannot emit free text or an apology; the grammar only permits one of
    our seven values.
    """
    result = await client.complete_json(
        _messages("The quick brown fox jumps over the lazy dog."),
        CLASSIFY_SCHEMA,
        use_cache=False,
    )
    assert result["clause_type"] in ClauseType.values()


@pytest.mark.llm
async def test_cache_returns_identical_result_and_skips_the_model():
    cache.clear()

    t0 = time.perf_counter()
    first = await client.complete_json(
        _messages(PRE_EXISTING_CLAUSE), CLASSIFY_SCHEMA
    )
    cold_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    second = await client.complete_json(
        _messages(PRE_EXISTING_CLAUSE), CLASSIFY_SCHEMA
    )
    warm_seconds = time.perf_counter() - t0

    assert first == second
    assert cache.stats()["entries"] == 1
    # A cache hit is a SQLite lookup; a miss is GPU inference. If this ever
    # fails, the cache key is including something that varies between calls.
    assert warm_seconds < cold_seconds / 5, (
        f"cache appears not to be hit: cold={cold_seconds:.2f}s warm={warm_seconds:.2f}s"
    )


def test_changing_the_prompt_text_changes_the_cache_key():
    """No Ollama needed: pure key logic.

    This is the guard against the most insidious bug in an LLM pipeline -
    editing a prompt, seeing no change, and concluding the model ignored you,
    when in fact a stale cache entry was served.

    It used to be enforced by hashing PROMPT_VERSION into the key and relying on
    everyone to bump it. It is now enforced by the prompt text itself, which
    cannot be forgotten: reword a single character and the key changes.
    """
    msgs = _messages(PRE_EXISTING_CLAUSE)
    reworded = [dict(m) for m in msgs]
    reworded[-1]["content"] = reworded[-1]["content"] + " "

    k1 = cache.make_key("m", msgs, CLASSIFY_SCHEMA)
    k2 = cache.make_key("m", reworded, CLASSIFY_SCHEMA)
    k3 = cache.make_key("other-model", msgs, CLASSIFY_SCHEMA)
    k4 = cache.make_key("m", msgs, {"type": "object"})

    assert len({k1, k2, k3, k4}) == 4, "cache key ignores an input that matters"
    assert k1 == cache.make_key("m", msgs, CLASSIFY_SCHEMA), "key is unstable"


def test_the_version_label_is_not_in_the_cache_key():
    """An unchanged prompt must keep its stored answer across a version bump.

    When the version was part of the key, bumping it for a change to ONE prompt
    discarded the cached answer to EVERY prompt, so all 40 eval cases were
    regenerated and each became a fresh chance for a nondeterministic model to
    answer differently. A three-case change could not be told apart from noise.
    """
    import inspect

    assert "prompt_version" not in inspect.signature(cache.make_key).parameters
    assert "prompt_version" not in inspect.signature(client.complete_json).parameters


def test_decode_options_change_the_cache_key():
    """num_ctx and num_predict change the ANSWER, so they must change the key.

    `num_ctx` truncates the prompt when it is too small; `num_predict` cuts the
    response short. Either produces a genuinely different result, so a cached
    entry from one setting must never be served for another - the same rule
    the message text enforces for wording, applied to the parameters.

    This was found the hard way: an uncapped `num_predict` let a generation run
    past three consecutive timeouts, and capping it would have been invisible to
    a cache that keyed only on the prompt.
    """
    msgs = _messages(PRE_EXISTING_CLAUSE)
    base = {"temperature": 0.0, "num_ctx": 8192, "num_predict": 900}

    key = cache.make_key("m", msgs, CLASSIFY_SCHEMA, base)
    smaller_ctx = cache.make_key("m", msgs, CLASSIFY_SCHEMA, {**base, "num_ctx": 4096})
    shorter_out = cache.make_key("m", msgs, CLASSIFY_SCHEMA, {**base, "num_predict": 100})

    assert len({key, smaller_ctx, shorter_out}) == 3, "decode options are not in the key"
    assert key == cache.make_key("m", msgs, CLASSIFY_SCHEMA, dict(base)), "key unstable"


def test_the_options_sent_are_the_options_hashed():
    """Guards against the two drifting apart.

    If the dict sent to Ollama and the dict hashed into the key were built
    separately, a cache entry could be keyed on settings the request never used.
    """
    from app.config import settings
    from app.llm.client import _decode_options

    options = _decode_options()
    assert options["num_ctx"] == settings.num_ctx
    assert options["num_predict"] == settings.num_predict
    assert options["temperature"] == settings.temperature


async def test_truncated_json_retries_with_a_higher_ceiling(monkeypatch):
    """A deterministic failure must not be retried unchanged.

    temperature is 0, so re-sending an identical request yields identical
    tokens. When a response is truncated mid-JSON, three identical attempts is
    three identical truncations - which is exactly what happened before this
    behaviour existed, and it failed an entire eval run.

    The retry has to change something. Here it doubles the generation ceiling,
    which addresses the actual cause.
    """
    from app.llm import client as llm_client

    seen: list[int] = []

    async def fake_post(messages, schema, model, options):
        seen.append(options["num_predict"])
        # Truncate on the first call, succeed once there is room.
        if len(seen) == 1:
            return '{"clause_type": "exclusion", "plain_language": "the quote was cut'
        return '{"clause_type": "exclusion", "plain_language": "ok", "waiting_months": 3}'

    monkeypatch.setattr(llm_client, "_post_chat", fake_post)

    result = await llm_client.complete_json(
        _messages(PRE_EXISTING_CLAUSE),
        CLASSIFY_SCHEMA,
        use_cache=False,
    )

    assert result["plain_language"] == "ok"
    assert len(seen) == 2, "should have retried exactly once"
    assert seen[1] == seen[0] * 2, "the retry must raise the ceiling, not repeat the request"


async def test_transport_errors_are_retried_unchanged(monkeypatch):
    """The mirror of the test above.

    A timeout or dropped connection is not a property of the prompt, so
    repeating it verbatim is the correct response - and the ceiling must NOT
    creep upward on failures that had nothing to do with output length.
    """
    import httpx

    from app.llm import client as llm_client

    seen: list[int] = []

    async def fake_post(messages, schema, model, options):
        seen.append(options["num_predict"])
        if len(seen) == 1:
            raise httpx.ReadTimeout("timed out")
        return '{"clause_type": "exclusion", "plain_language": "ok", "waiting_months": 3}'

    monkeypatch.setattr(llm_client, "_post_chat", fake_post)

    await llm_client.complete_json(
        _messages(PRE_EXISTING_CLAUSE),
        CLASSIFY_SCHEMA,
        use_cache=False,
    )

    assert seen == [seen[0], seen[0]], "transport retry must not change the ceiling"
