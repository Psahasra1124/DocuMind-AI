from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

from app.services.llm.cloudflare import DEFAULT_CLOUDFLARE_MODEL


DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def _postgres_url(database_url: str, driver: str) -> str:
    url = make_url(database_url)
    if url.get_backend_name() not in {"postgres", "postgresql"}:
        raise ValueError("DATABASE_URL must use a PostgreSQL URL")

    query = dict(url.query)
    if driver == "asyncpg" and "sslmode" in query:
        query.setdefault("ssl", query.pop("sslmode"))
    elif driver == "psycopg2" and "ssl" in query:
        query.setdefault("sslmode", query.pop("ssl"))

    return url.set(
        drivername=f"postgresql+{driver}",
        query=query,
    ).render_as_string(hide_password=False)


class Settings(BaseSettings):
    DATABASE_URL: str
    SYNC_DATABASE_URL: str | None = None
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    CORS_ORIGINS: str | tuple[str, ...] = DEFAULT_CORS_ORIGINS
    DOCUMENT_STORAGE_DIR: Path = Path("storage")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_STORE_DIR: Path = Path("vector_store")
    VECTOR_COLLECTION_NAME: str = "document_chunks"
    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:4b"
    OLLAMA_TIMEOUT_SECONDS: float = 300.0
    CLOUDFLARE_ACCOUNT_ID: str | None = None
    CLOUDFLARE_API_TOKEN: str | None = None
    CLOUDFLARE_MODEL: str = DEFAULT_CLOUDFLARE_MODEL
    CLOUDFLARE_TIMEOUT_SECONDS: float = 90.0
    MAX_UPLOAD_SIZE_MB: int = 25

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def async_database_url(self) -> str:
        return _postgres_url(self.DATABASE_URL, "asyncpg")

    @property
    def alembic_database_url(self) -> str:
        sync_url = self.SYNC_DATABASE_URL
        if sync_url is None or not sync_url.strip():
            sync_url = self.DATABASE_URL
        return _postgres_url(sync_url, "psycopg2")

    @field_validator("MAX_UPLOAD_SIZE_MB")
    @classmethod
    def validate_max_upload_size(cls, value: int) -> int:
        if value < 1 or value > 250:
            raise ValueError("MAX_UPLOAD_SIZE_MB must be between 1 and 250")
        return value

    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"ollama", "cloudflare"}:
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'cloudflare'")
        return normalized

    @field_validator("CLOUDFLARE_TIMEOUT_SECONDS", "OLLAMA_TIMEOUT_SECONDS")
    @classmethod
    def validate_llm_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("LLM timeout must be greater than 0")
        return value

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret_key(cls, value: str) -> str:
        if len(value) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must be at least 32 characters long"
            )
        return value

    @field_validator("CORS_ORIGINS")
    @classmethod
    def normalize_cors_origins(cls, value: str | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(value, str):
            origins = tuple(
                origin.strip()
                for origin in value.split(",")
                if origin.strip()
            )
        else:
            origins = tuple(origin.strip() for origin in value if origin.strip())

        if not origins:
            raise ValueError("CORS_ORIGINS must contain at least one origin")

        return origins

    @model_validator(mode="after")
    def validate_cloudflare_settings(self):
        if self.LLM_PROVIDER == "cloudflare":
            if (
                not self.CLOUDFLARE_ACCOUNT_ID
                or not self.CLOUDFLARE_ACCOUNT_ID.strip()
            ):
                raise ValueError(
                    "CLOUDFLARE_ACCOUNT_ID is required when "
                    "LLM_PROVIDER=cloudflare"
                )
            if (
                not self.CLOUDFLARE_API_TOKEN
                or not self.CLOUDFLARE_API_TOKEN.strip()
            ):
                raise ValueError(
                    "CLOUDFLARE_API_TOKEN is required when "
                    "LLM_PROVIDER=cloudflare"
                )
        return self

    @model_validator(mode="after")
    def validate_cors_origins(self):
        if "*" in self.CORS_ORIGINS:
            raise ValueError(
                "CORS_ORIGINS cannot contain '*' when credentials are enabled"
            )
        return self


settings = Settings()
