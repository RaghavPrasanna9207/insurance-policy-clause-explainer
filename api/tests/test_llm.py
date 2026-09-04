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
        prompt_version="test-v1",
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
        prompt_version="test-v1",
        use_cache=False,
    )
    assert result["clause_type"] in ClauseType.values()


@pytest.mark.llm
async def test_cache_returns_identical_result_and_skips_the_model():
    cache.clear()

    t0 = time.perf_counter()
    first = await client.complete_json(
        _messages(PRE_EXISTING_CLAUSE), CLASSIFY_SCHEMA, prompt_version="test-cache"
    )
    cold_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    second = await client.complete_json(
        _messages(PRE_EXISTING_CLAUSE), CLASSIFY_SCHEMA, prompt_version="test-cache"
    )
    warm_seconds = time.perf_counter() - t0

    assert first == second
    assert cache.stats()["entries"] == 1
    # A cache hit is a SQLite lookup; a miss is GPU inference. If this ever
    # fails, the cache key is including something that varies between calls.
    assert warm_seconds < cold_seconds / 5, (
        f"cache appears not to be hit: cold={cold_seconds:.2f}s warm={warm_seconds:.2f}s"
    )


def test_prompt_version_changes_the_cache_key():
    """No Ollama needed: pure key logic.

    This is the guard against the most insidious bug in an LLM pipeline -
    editing a prompt, seeing no change, and concluding the model ignored you,
    when in fact a stale cache entry was served.
    """
    msgs = _messages(PRE_EXISTING_CLAUSE)
    k1 = cache.make_key("m", "v1", msgs, CLASSIFY_SCHEMA)
    k2 = cache.make_key("m", "v2", msgs, CLASSIFY_SCHEMA)
    k3 = cache.make_key("other-model", "v1", msgs, CLASSIFY_SCHEMA)
    k4 = cache.make_key("m", "v1", msgs, {"type": "object"})

    assert len({k1, k2, k3, k4}) == 4, "cache key ignores an input that matters"
    assert k1 == cache.make_key("m", "v1", msgs, CLASSIFY_SCHEMA), "key is unstable"
