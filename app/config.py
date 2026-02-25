from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost/parallellines"
    DEBUG: bool = False
    SECRET_KEY: str = "dev-secret-key"
    API_V1_PREFIX: str = "/api/v1"

    POLYMARKET_API_URL: str = "https://clob.polymarket.com"
    GAMMA_API_URL: str = "https://gamma-api.polymarket.com"

    # Teacher model (label generation) + inference fallback
    ANTHROPIC_API_KEY: str = ""

    # Fine-tuned model — set after your first training run
    BASE_MODEL_NAME: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ADAPTER_PATH: str = "./checkpoints/intuone-v1/final_adapter"
    # Set to "true" to auto-load the model at startup
    LOAD_MODEL_ON_STARTUP: bool = False


settings = Settings()
