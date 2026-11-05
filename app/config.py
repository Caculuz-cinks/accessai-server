from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "production"
    DEBUG: bool = False
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # SQLite Database
    DATABASE_PATH: str = "voice_intent.db"

    # LLM Provider configuration
    LLM_PROVIDER: Literal["groq", "anthropic", "mock"] = "mock"
    LLM_REGION: str = "US"  # Region of processing for NDPA 2023 disclosures
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-20241022"

    # Security & Device Auth (v1.1)
    ENFORCE_DEVICE_AUTH: bool = False  # Enforce device key on all sensitive actions

    # Confidence threshold for intent routing
    CONFIDENCE_THRESHOLD: float = 0.85

    # Token expiration in seconds
    CONFIRMATION_TOKEN_TTL_SECONDS: int = 60

    # Ed25519 Key configuration (base64 private key or path)
    ED25519_PRIVATE_KEY_B64: str = ""
    JWKS_KID: str = "voice-intent-key-1"

    @model_validator(mode="after")
    def validate_production_safety(self) -> "Settings":
        if self.APP_ENV == "production" and not self.DEBUG:
            if self.LLM_PROVIDER == "mock":
                raise ValueError(
                    "LLM_PROVIDER cannot default to 'mock' in production. "
                    "Configure 'groq' or 'anthropic' with valid credentials."
                )
            if not self.ED25519_PRIVATE_KEY_B64:
                raise ValueError(
                    "ED25519_PRIVATE_KEY_B64 must be set in production. "
                    "Refusing to generate ephemeral keys across restarts."
                )
        return self


settings = Settings()
