from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


class Settings(BaseSettings):
    DATABASE_URL: str
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
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TIMEOUT_SECONDS: float = 60.0
    MAX_UPLOAD_SIZE_MB: int = 25

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

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
        if normalized not in {"ollama", "gemini"}:
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'gemini'")
        return normalized

    @field_validator("GEMINI_TIMEOUT_SECONDS", "OLLAMA_TIMEOUT_SECONDS")
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
    def validate_cors_origins(self):
        if "*" in self.CORS_ORIGINS:
            raise ValueError(
                "CORS_ORIGINS cannot contain '*' when credentials are enabled"
            )
        return self


settings = Settings()
