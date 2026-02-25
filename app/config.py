from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Database ──────────────────────────────────────────────────────────────
    # PostgreSQL (default): postgresql+asyncpg://user:pass@host/db
    # SQLite  (Windows dev): sqlite+aiosqlite:///./parallellines.db
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost/parallellines"

    DEBUG: bool = False
    SECRET_KEY: str = "dev-secret-key"
    API_V1_PREFIX: str = "/api/v1"

    # ── Data sources ──────────────────────────────────────────────────────────
    POLYMARKET_API_URL: str = "https://clob.polymarket.com"
    GAMMA_API_URL: str = "https://gamma-api.polymarket.com"
    # How often the ingestion loop polls each connector (seconds)
    INGEST_INTERVAL_SECONDS: int = 300  # 5 minutes

    # ── External LLMs (training only — NOT used in production inference) ──────
    ANTHROPIC_API_KEY: str = ""

    # ── Local fine-tuned model ────────────────────────────────────────────────
    BASE_MODEL_NAME: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ADAPTER_PATH: str = "./checkpoints/intuone-v1/final_adapter"
    LOAD_MODEL_ON_STARTUP: bool = False


settings = Settings()
