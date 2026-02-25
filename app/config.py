from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost/parallellines"
    DEBUG: bool = False
    SECRET_KEY: str = "dev-secret-key"
    API_V1_PREFIX: str = "/api/v1"

    POLYMARKET_API_URL: str = "https://clob.polymarket.com"
    GAMMA_API_URL: str = "https://gamma-api.polymarket.com"


settings = Settings()
