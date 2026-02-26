"""Application configuration using pydantic-settings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime settings — loaded from env vars / .env file."""

    model_config = {"env_prefix": "CI_", "env_file": ".env", "extra": "ignore"}

    openai_api_key: str = Field(default="", description="OpenAI API key (leave blank for simulation mode)")
    openai_model: str = "gpt-4o"
    simulation_mode: bool = Field(
        default=True,
        description="When True, agents return realistic synthetic outputs without calling LLMs",
    )
    max_reanalysis: int = 2
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
