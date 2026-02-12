"""OpenRouter model implementation for insideLLMs via OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import Optional

from insideLLMs.models.openai import OpenAIModel


class OpenRouterModel(OpenAIModel):
    """OpenRouter model wrapper using the OpenAI-compatible API.

    OpenRouter exposes an OpenAI-compatible API surface with a dedicated API key.
    This model sets the OpenRouter base URL and pulls credentials from
    ``OPENROUTER_API_KEY`` by default.

    Optional headers for OpenRouter attribution can be provided via:
        - http_referer / OPENROUTER_HTTP_REFERER
        - app_title / OPENROUTER_APP_TITLE
    """

    def __init__(
        self,
        name: str = "OpenRouterModel",
        model_name: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: Optional[str] = None,
        app_title: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
        **kwargs,
    ):
        default_headers: dict[str, str] = {}
        referer = (
            http_referer
            or os.getenv("OPENROUTER_HTTP_REFERER")
            or "https://github.com/dr-gareth-roberts/insideLLMs"
        )
        title = app_title or os.getenv("OPENROUTER_APP_TITLE") or "insideLLMs"
        default_headers["HTTP-Referer"] = referer
        default_headers["X-Title"] = title
        if extra_headers:
            default_headers.update(dict(extra_headers))

        super().__init__(
            name=name,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            api_key_env="OPENROUTER_API_KEY",
            default_headers=default_headers,
            **kwargs,
        )

    def info(self):
        base_info = super().info()
        base_info.provider = "openrouter"
        base_info.extra.update(
            {
                "description": (
                    "OpenRouter model via OpenAI-compatible API. "
                    "Requires OPENROUTER_API_KEY env variable."
                ),
                "provider": "openrouter",
                "base_url": self._base_url,
            }
        )
        return base_info
