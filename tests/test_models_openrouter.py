"""Tests for insideLLMs/models/openrouter.py module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("openai")


class TestOpenRouterModelInit:
    def test_init_passes_openrouter_defaults_to_openai_model(self) -> None:
        with patch("insideLLMs.models.openrouter.OpenAIModel.__init__", return_value=None) as init:
            from insideLLMs.models.openrouter import OpenRouterModel

            OpenRouterModel(api_key="test-key", model_name="openai/gpt-4o-mini")

            kwargs = init.call_args.kwargs
            assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
            assert kwargs["api_key_env"] == "OPENROUTER_API_KEY"
            assert kwargs["api_key"] == "test-key"
            assert kwargs["model_name"] == "openai/gpt-4o-mini"

    def test_init_builds_default_headers_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_HTTP_REFERER": "https://example.com",
                "OPENROUTER_APP_TITLE": "insideLLMs",
            },
            clear=False,
        ):
            with patch(
                "insideLLMs.models.openrouter.OpenAIModel.__init__", return_value=None
            ) as init:
                from insideLLMs.models.openrouter import OpenRouterModel

                OpenRouterModel(api_key="test-key")

                headers = init.call_args.kwargs["default_headers"]
                assert headers["HTTP-Referer"] == "https://example.com"
                assert headers["X-Title"] == "insideLLMs"

    def test_info_marks_provider(self) -> None:
        with patch("insideLLMs.models.openrouter.OpenAIModel.__init__", return_value=None):
            with patch("insideLLMs.models.openrouter.OpenAIModel.info") as info:
                info.return_value = type("Info", (), {"extra": {}})()
                from insideLLMs.models.openrouter import OpenRouterModel

                model = OpenRouterModel(api_key="test-key")
                metadata = model.info()

                assert metadata.extra["provider"] == "openrouter"
                assert "OPENROUTER_API_KEY" in metadata.extra["description"]
