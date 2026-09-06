"""Central configuration.

Every tunable lives here rather than scattered through the pipeline, so the eval
harness can swap models or budgets without editing logic. Values can be
overridden by environment variables or a .env file (see .env.example).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Ollama ---
    ollama_url: str = "http://localhost:11434"
    # The model verified on this machine: fits fully in the RTX 4060's 8GB VRAM
    # and enforces JSON-schema enums, which the grounding design depends on.
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    # The scenario reasoner sends the whole policy (~4k tokens) and generates a
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
    # 8,192, not the full 32,768 and not the 16,384 first tried here.
    #
    # The KV cache lives in VRAM alongside the 4.7GB of weights. At 16,384 the
    # runtime held 5.46GB and the machine was left with 2GB of 15.7GB free -
    # the scenario eval was killed by the OS for memory pressure. Over-
    # provisioning "to be safe" is not free; it was paid for in RAM that the
    # rest of the system needed.
    #
    # 8,192 is sized from the actual requirement rather than from caution:
    # a 40-clause policy is ~3,100 tokens, the system prompt ~1,200, the
    # response ~500. That is ~4,800, so this leaves comfortable headroom while
    # halving the cache.
    num_ctx: int = 8_192

    # Tokens reserved inside the context for everything that is NOT clause text:
    # the system prompt, the extracted facts, and the generated answer.
    scenario_reserved_tokens: int = 2_500

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

    # --- Pipeline ---
    # One clause per call. This is measured, not assumed: on the golden policy,
    # batch=1 scored macro-F1 1.000 in 128.5s against batch=5's 0.973 in 121.9s.
    # Batching saved ~5% wall time and misclassified a clause that is correct
    # when analysed alone - neighbouring clauses in a policy are related, so
    # sharing a generation lets the model's reading of one bleed into the next.
    # Raise it for very large documents, where the time trade shifts.
    analyze_batch_size: int = 1
    # Ollama serves requests concurrently, but each still competes for the same
    # GPU. 2 measured better than 4 on an 8GB card.
    analyze_concurrency: int = 2
    @property
    def scenario_token_budget(self) -> int:
        """How many tokens of clause text the reasoner may be given.

        DERIVED from the context window rather than set independently, because
        the two were briefly inconsistent: a 12,000-token clause budget against
        an 8,192-token window would build a prompt larger than the context and
        Ollama would silently truncate it - dropping exactly the clauses the
        shortlist had just been careful to include.

        Two numbers that must agree should not be two numbers.
        """
        return max(self.num_ctx - self.scenario_reserved_tokens, 1_000)

    # --- Storage ---
    db_path: str = "data/app.db"
    upload_dir: str = "data/uploads"


settings = Settings()
