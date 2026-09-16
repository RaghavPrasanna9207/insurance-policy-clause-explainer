"""Central configuration.

Every tunable lives here rather than scattered through the pipeline, so the eval
harness can swap models or budgets without editing logic. Values can be
overridden by environment variables or a .env file (see .env.example).
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Ollama ---
    ollama_url: str = "http://localhost:11434"
    # The model verified on this machine: fits fully in the RTX 4060's 8GB VRAM
    # and enforces JSON-schema enums, which the grounding design depends on.
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    # The scenario reasoner sends the whole policy (~4k tokens for the synthetic
    # one; ~22,600 for a real wording, about 11s to read) and generates a
    # few hundred more, on top of a possible ~27s cold model load. 180s was not
    # enough and produced httpx.ReadTimeout under real load; 420s has margin
    # without hanging a request forever.
    # With num_predict capping generation, a call is prompt evaluation plus a
    # bounded response, so 240s is ample. The earlier 420s was compensating for
    # runaway generation rather than fixing it - and a long timeout on a broken
    # call just makes the failure take longer to discover.
    request_timeout: float = 240.0

    # EXPLICIT context window. This is not a tuning knob, it is a correctness
    # fix.
    #
    # qwen2.5 supports 32,768 tokens, but this model's Ollama definition sets no
    # `num_ctx`, so Ollama applies its own default of 4,096. The scenario
    # simulator's whole design rests on "every clause that could matter fits in
    # one prompt" - and at 4,096 that stops being true the moment a policy is
    # slightly larger than the test one. Ollama would silently drop the start of
    # the prompt, and the answer would be reasoned over a policy missing its
    # first clauses, with nothing to indicate it.
    #
    # 28,672: the largest window measured to run entirely on this machine's GPU.
    #
    # It was 8,192 from M5, sized to the synthetic policy (~3,100 tokens) after
    # 16,384 had the OS kill an eval for memory pressure on Ollama 0.33. M15
    # measured real wordings at 69,000-129,000 characters, and at 8,192 the
    # scenario step saw a quarter of the smallest - the premise of having no
    # retrieval, failing silently.
    #
    # Re-measured in M16 on Ollama 0.34, with a policy-sized prompt:
    #
    #     num_ctx   GPU memory   placement         prompt speed
    #      8,192     4.9 GB      100% GPU          2,149 tok/s
    #     24,576     5.8 GB      100% GPU          2,036 tok/s
    #     28,672     6.2 GB      100% GPU
    #     32,768     6.8 GB      8% CPU / 92% GPU  generation 20% slower
    #
    # The KV cache sits in VRAM, so free system RAM did not change between the
    # sizes. 32,768 is qwen2.5's maximum, but spilling onto the CPU slows every
    # call, including the per-clause analyses that need no long context. One
    # window for every call, because Ollama reloads the model when it changes.
    #
    # Those figures are for a 16-bit KV cache. With the q8_0 cache below, 28,672
    # takes 5.0 GB and even 32,768 fits entirely on the GPU (5.2 GB).
    num_ctx: int = 28_672

    # PRECISION OF THE KV CACHE: a setting of the Ollama server, mirrored here.
    #
    # For every token of the window, each layer stores a key and a value, 56 KB
    # per token for this model in 16-bit floats (`f16`, Ollama's default). The
    # store is allocated for the whole window when the model loads: 1,568 MiB at
    # 28,672. Starting the server with OLLAMA_KV_CACHE_TYPE=q8_0 rounds those
    # numbers to 8 bits and roughly halves that. Rounding moves the logits a
    # little, so it can change an answer.
    #
    # No request carries it, so the cache cannot learn it from the decoding
    # options; it is hashed in separately (see cache.make_key). It is read from
    # the same environment variable the server reads, so the two agree when both
    # start from the same environment. Ollama only reads it at start-up: after a
    # change, restart Ollama and check the `llama_kv_cache` line in its log.
    #
    # q8_0 is the project's setting from M16 (see the README's Ollama setup), so
    # it is also the default here. A default of f16 would be wrong exactly when
    # it matters: a process started before the variable was set would store
    # q8_0 answers under f16 keys.
    kv_cache_type: Literal["f16", "q8_0", "q4_0"] = Field(
        "q8_0", validation_alias="OLLAMA_KV_CACHE_TYPE"
    )

    # HARD CAP on generated tokens. Without one, llama.cpp generates until the
    # model emits a stop token or the context fills - and a model that starts
    # repeating itself does the latter. That is what happened: one scenario ran
    # past a 420-second timeout three times in a row, burning 21 minutes before
    # failing, because nothing bounded the output.
    #
    # 1,600 is computed from what the schema can hold, not guessed. 900 was
    # guessed, and the model hit it mid-quote: the JSON came back with an
    # unterminated string and the whole eval died.
    #
    # Worst case under the reasoning schema: 4 citations x (a clause id, a
    # quoted sentence of ~200 chars, an effect, JSON punctuation) ~= 1,100
    # chars, plus ~400 chars of reasoning, plus the missing_information array.
    # Call it 1,700 chars ~= 500 tokens. 1,600 is triple that.
    #
    # The lesson is that "cap runaway generation" and "cap useful output" are
    # the same knob, so it has to be sized from the largest legitimate response,
    # never from a round number that feels safe.
    num_predict: int = 1_600
    # temperature=0 is not decoration: extraction must be reproducible, or the
    # LLM cache and the eval numbers both become meaningless.
    temperature: float = 0.0
    # temperature=0 ALONE DOES NOT GIVE REPRODUCIBILITY, which this project
    # assumed for eight milestones and then measured.
    #
    # Two full eval runs of identical code, on identical prompts, disagreed on
    # three of forty scenario cases - `senior-but-excluded` and `breach-of-law`
    # went from correct to wrong, `copay-just-under-sixty` from wrong to
    # correct. A change worth two cases cannot be measured through noise worth
    # three.
    #
    # Greedy decoding still runs through llama.cpp's sampler chain, and with no
    # seed supplied Ollama picks a random one per request, so near-ties between
    # two tokens are broken differently from run to run. A near-tie at one
    # token is all it takes: `not_covered` and `insufficient_information` begin
    # to diverge at the first word of the verdict.
    #
    # Fixed rather than left to chance. The value is arbitrary; being the same
    # value every time is the entire point.
    seed: int = 0

    # --- Pipeline ---
    # One clause per call. This is measured, not assumed: on the golden policy,
    # batch=1 scored macro-F1 1.000 in 128.5s against batch=5's 0.973 in 121.9s.
    # Batching saved ~5% wall time and misclassified a clause that is correct
    # when analysed alone - neighbouring clauses in a policy are related, so
    # sharing a generation lets the model's reading of one bleed into the next.
    # Raise it for very large documents, where the time trade shifts.
    analyze_batch_size: int = 1
    # ONE REQUEST AT A TIME, AND THE REASON IS REPRODUCIBILITY, NOT SPEED.
    #
    # This was 2 ("Ollama serves requests concurrently, but each still competes
    # for the same GPU; 2 measured better than 4 on an 8GB card") until the
    # cost of that concurrency was measured rather than assumed.
    #
    # `evals/check_determinism.py` analyses the same 12 clauses six times with
    # the cache bypassed and compares the output byte for byte:
    #
    #     concurrency=1   6/6 identical
    #     concurrency=2   3 of 6 diverge, alternating exactly
    #     concurrency=4   1 of 6 diverges
    #
    # With more than one request in flight, Ollama batches whatever arrives
    # together. Float addition is not associative, so a different batch
    # composition sums the matmul reductions in a different order and lands on
    # a different logit in the low bits - and where two tokens were nearly
    # tied, the tie breaks the other way. The same clause came back as "any
    # place that provides inpatient or day care treatment" on one run and "any
    # place that takes sick people in for treatment" on the next.
    #
    # WHAT IT COST TO NOT KNOW THIS: three runs of the 40-case scenario eval
    # over identical code scored 0.725, 0.700 and 0.750, with five cases
    # flipping between runs. Every prompt comparison this project has made was
    # being read through noise of that size.
    #
    # AND THE TRADE IS ALMOST NOTHING, which is the part worth knowing. The
    # obvious objection to serialising is that it must halve throughput. It
    # does not. Measured on the 40-clause golden policy with the cache off:
    #
    #     concurrency=1   203.0s
    #     concurrency=2   184.2s
    #
    # Nine percent, not two hundred. One request already saturates an 8GB card,
    # so the second mostly waits its turn - the same reason 2 beat 4 in the
    # original measurement, carried one step further than it was taken. The
    # concurrency was buying 19 seconds and costing reproducibility.
    analyze_concurrency: int = 1
    # The clause budget for the scenario step is derived from num_ctx, num_predict
    # and the measured system prompt: see `clause_token_budget()` in
    # app/pipeline/scenario.py. It lives there because it needs the prompt.

    # --- Storage ---
    db_path: str = "data/app.db"
    upload_dir: str = "data/uploads"


settings = Settings()
