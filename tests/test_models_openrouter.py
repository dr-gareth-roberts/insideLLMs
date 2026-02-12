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
            assert kwargs["default_headers"]["HTTP-Referer"]
            assert kwargs["default_headers"]["X-Title"]

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

    def test_extra_headers_are_copied(self) -> None:
        with patch("insideLLMs.models.openrouter.OpenAIModel.__init__", return_value=None) as init:
            from insideLLMs.models.openrouter import OpenRouterModel

            supplied = {"X-Custom": "abc"}
            OpenRouterModel(api_key="test-key", extra_headers=supplied)
            supplied["X-Custom"] = "mutated"

            headers = init.call_args.kwargs["default_headers"]
            assert headers["X-Custom"] == "abc"

    def test_info_marks_provider(self) -> None:
        with patch("insideLLMs.models.openrouter.OpenAIModel.__init__", return_value=None):
            with patch("insideLLMs.models.openrouter.OpenAIModel.info") as info:
                info.return_value = type("Info", (), {"extra": {}, "provider": "OpenAI"})()
                from insideLLMs.models.openrouter import OpenRouterModel

                model = OpenRouterModel(api_key="test-key")
                metadata = model.info()

                assert metadata.provider == "openrouter"
                assert metadata.extra["provider"] == "openrouter"
                assert metadata.extra["base_url"] == "https://openrouter.ai/api/v1"
                assert "OPENROUTER_API_KEY" in metadata.extra["description"]
