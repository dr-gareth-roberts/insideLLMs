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

    def test_extra_headers_override_env_headers(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://env.example.com/from-env")

        with patch("insideLLMs.models.openrouter.OpenAIModel.__init__", return_value=None) as init:
            from insideLLMs.models.openrouter import OpenRouterModel

            OpenRouterModel(
                api_key="test-key",
                model_name="openai/gpt-4o-mini",
                extra_headers={"HTTP-Referer": "https://explicit.example.com/from-extra"},
            )

            kwargs = init.call_args.kwargs
            default_headers = kwargs["default_headers"]
            assert default_headers["HTTP-Referer"] == "https://explicit.example.com/from-extra"

    def test_default_headers_none_when_no_env_or_extra_headers(self, monkeypatch) -> None:
        for var in (
            "OPENROUTER_HTTP_REFERER",
            "OPENROUTER_APP_TITLE",
            "OPENROUTER_SITE_URL",
            "OPENROUTER_SITE_NAME",
        ):
            monkeypatch.delenv(var, raising=False)

        with patch("insideLLMs.models.openrouter.OpenAIModel.__init__", return_value=None) as init:
            from insideLLMs.models.openrouter import OpenRouterModel

            OpenRouterModel(api_key="test-key", model_name="openai/gpt-4o-mini")

            kwargs = init.call_args.kwargs
            assert kwargs.get("default_headers") is None

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
