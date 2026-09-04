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
    # Cold model load measured at ~27s, and a batch of 5 clauses generates
    # several hundred tokens at ~50 tok/s. 180s leaves headroom for both.
    request_timeout: float = 180.0
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
    # Ceiling on the clause shortlist handed to the scenario reasoner. qwen2.5
    # has a 32k window; 12k keeps generation quality high and leaves room for
    # the schema and the reasoning itself.
    scenario_token_budget: int = 12_000

    # --- Storage ---
    db_path: str = "data/app.db"
    upload_dir: str = "data/uploads"


settings = Settings()
