"""Gateway-focused tests for OpenAI/OpenRouter model wrappers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("openai")


def test_openai_model_redacts_sensitive_headers_in_info() -> None:
    with patch("insideLLMs.models.openai.OpenAI"):
        from insideLLMs.models.openai import OpenAIModel

        model = OpenAIModel(
            api_key="explicit-key",
            api_key_env="CUSTOM_KEY",
            default_headers={
                "Authorization": "Bearer secret",
                "X-Api-Key": "secret-2",
                "X-Title": "insideLLMs",
            },
        )

        info = model.info()
        assert info.extra["api_key_env"] == "CUSTOM_KEY"
        assert info.extra["default_headers"]["Authorization"] == "***"
        assert info.extra["default_headers"]["X-Api-Key"] == "***"
        assert info.extra["default_headers"]["X-Title"] == "insideLLMs"


def test_openrouter_api_key_precedence_explicit_over_env() -> None:
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key"}, clear=False):
        with patch("insideLLMs.models.openai.OpenAI") as mock_openai:
            from insideLLMs.models.openrouter import OpenRouterModel

            OpenRouterModel(api_key="explicit-key")
            assert mock_openai.call_args.kwargs["api_key"] == "explicit-key"


def test_openrouter_missing_api_key_raises() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with patch("insideLLMs.models.openai.OpenAI"):
            from insideLLMs.exceptions import ModelInitializationError
            from insideLLMs.models.openrouter import OpenRouterModel

            with pytest.raises(ModelInitializationError, match="OPENROUTER_API_KEY"):
                OpenRouterModel()
