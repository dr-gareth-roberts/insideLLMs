"""Regression tests for Cluster D provider-adapter fixes.

Covers:
- D1: prompt validation at public generate/stream boundaries
- D2: Retry-After extraction from SDK exception headers
- D3: Gemini generation_config packaging (stop_sequences etc.)
- D4: Gemini/Cohere chat history shape (final user turn)
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from insideLLMs.models.base import Model, ModelWrapper, _retry_after_seconds
from insideLLMs.validation import ValidationError


class _EchoModel(Model):
    """Minimal model that records generate/stream calls."""

    def __init__(self) -> None:
        super().__init__(name="echo", model_id="echo-v1")
        self.generate_calls: list[str] = []
        self.stream_calls: list[str] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.generate_calls.append(prompt)
        return f"echo:{prompt}"

    def stream(self, prompt: str, **kwargs: Any):
        self.stream_calls.append(prompt)
        yield f"chunk:{prompt}"


# ---------------------------------------------------------------------------
# D1 — prompt validation
# ---------------------------------------------------------------------------


class TestPromptValidationBoundary:
    def test_generate_rejects_empty_prompt(self) -> None:
        model = _EchoModel()
        with pytest.raises(ValidationError, match="empty"):
            model.generate("")
        assert model.generate_calls == []

    def test_stream_rejects_empty_prompt_before_iteration(self) -> None:
        model = _EchoModel()
        with pytest.raises(ValidationError, match="empty"):
            model.stream("")
        assert model.stream_calls == []

    def test_generate_with_metadata_rejects_empty(self) -> None:
        model = _EchoModel()
        with pytest.raises(ValidationError, match="empty"):
            model.generate_with_metadata("")
        assert model.generate_calls == []

    def test_disabled_validation_allows_empty(self) -> None:
        model = _EchoModel()
        model._validate_prompts = False
        assert model.generate("") == "echo:"
        assert model.generate_calls == [""]

    def test_wrapper_validates_once_before_retries(self) -> None:
        model = _EchoModel()
        wrapper = ModelWrapper(model, max_retries=3, retry_delay=0.0)
        with pytest.raises(ValidationError, match="empty"):
            wrapper.generate("")
        assert model.generate_calls == []


# ---------------------------------------------------------------------------
# D2 — Retry-After extraction
# ---------------------------------------------------------------------------


class TestRetryAfterSeconds:
    def test_direct_attribute(self) -> None:
        err = types.SimpleNamespace(retry_after=1.5)
        assert _retry_after_seconds(err) == 1.5  # type: ignore[arg-type]

    def test_header_retry_after_case_insensitive(self) -> None:
        headers = {"retry-after": "2.5"}
        response = types.SimpleNamespace(headers=headers)
        err = types.SimpleNamespace(response=response)
        assert _retry_after_seconds(err) == 2.5  # type: ignore[arg-type]

    def test_header_retry_after_canonical_case(self) -> None:
        headers = {"Retry-After": "3"}
        response = types.SimpleNamespace(headers=headers)
        err = types.SimpleNamespace(response=response)
        assert _retry_after_seconds(err) == 3.0  # type: ignore[arg-type]

    def test_attribute_preferred_over_header(self) -> None:
        headers = {"Retry-After": "99"}
        response = types.SimpleNamespace(headers=headers)
        err = types.SimpleNamespace(retry_after=4.0, response=response)
        assert _retry_after_seconds(err) == 4.0  # type: ignore[arg-type]

    def test_missing_returns_none(self) -> None:
        assert _retry_after_seconds(Exception("x")) is None

    def test_openai_rate_limit_uses_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import insideLLMs.models.openai as openai_mod
        from insideLLMs.exceptions import RateLimitError

        headers = {"Retry-After": "7"}
        response = types.SimpleNamespace(headers=headers)

        class FakeOpenAIRateLimit(Exception):
            def __init__(self) -> None:
                super().__init__("rate limited")
                self.response = response

        monkeypatch.setattr(openai_mod, "OpenAIRateLimitError", FakeOpenAIRateLimit)

        model = openai_mod.OpenAIModel.__new__(openai_mod.OpenAIModel)
        Model.__init__(model, name="gpt-4", model_id="gpt-4")
        model.model_name = "gpt-4"
        model._timeout = 30.0
        model._budget_ledger = None
        model._create_completion = MagicMock(side_effect=FakeOpenAIRateLimit())  # type: ignore[method-assign]

        with pytest.raises(RateLimitError) as exc_info:
            model.generate("hello")
        assert exc_info.value.retry_after == 7.0

    def test_anthropic_rate_limit_uses_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import insideLLMs.models.anthropic as anthropic_mod
        from insideLLMs.exceptions import RateLimitError

        headers = {"retry-after": "9"}
        response = types.SimpleNamespace(headers=headers)

        class FakeAnthropicRateLimit(Exception):
            def __init__(self) -> None:
                super().__init__("rate limited")
                self.response = response

        monkeypatch.setattr(anthropic_mod, "AnthropicRateLimitError", FakeAnthropicRateLimit)

        model = anthropic_mod.AnthropicModel.__new__(anthropic_mod.AnthropicModel)
        Model.__init__(model, name="claude", model_id="claude")
        model.model_name = "claude"
        model._timeout = 30.0
        model._budget_ledger = None
        model._create_message = MagicMock(side_effect=FakeAnthropicRateLimit())  # type: ignore[method-assign]

        with pytest.raises(RateLimitError) as exc_info:
            model.generate("hello")
        assert exc_info.value.retry_after == 9.0


# ---------------------------------------------------------------------------
# D3 — Gemini generation_config
# ---------------------------------------------------------------------------


class TestGeminiGenerationConfig:
    def test_stop_sequences_go_in_generation_config(self) -> None:
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel.__new__(GeminiModel)
        Model.__init__(model, name="gemini", model_id="gemini")
        model.default_generation_config = {}
        model._call_count = 0
        mock_model = MagicMock()
        mock_model.generate_content.return_value = types.SimpleNamespace(text="ok")
        model._model = mock_model
        model._client = object()
        model._get_client = lambda: mock_model  # type: ignore[method-assign]

        model.generate(
            "prompt",
            temperature=0.4,
            max_tokens=12,
            stop_sequences=["END"],
            top_p=0.8,
            top_k=5,
        )

        kwargs = mock_model.generate_content.call_args.kwargs
        assert "stop_sequences" not in kwargs
        cfg = kwargs["generation_config"]
        assert cfg["stop_sequences"] == ["END"]
        assert cfg["temperature"] == 0.4
        assert cfg["max_output_tokens"] == 12
        assert cfg["top_p"] == 0.8
        assert cfg["top_k"] == 5

    def test_stream_puts_stop_sequences_in_config(self) -> None:
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel.__new__(GeminiModel)
        Model.__init__(model, name="gemini", model_id="gemini")
        model.default_generation_config = {"temperature": 0.1}
        model._call_count = 0
        mock_model = MagicMock()
        mock_model.generate_content.return_value = [types.SimpleNamespace(text="a")]
        model._model = mock_model
        model._get_client = lambda: mock_model  # type: ignore[method-assign]

        list(model.stream("p", stop_sequences=["STOP"], max_tokens=3))

        kwargs = mock_model.generate_content.call_args.kwargs
        assert kwargs.get("stream") is True
        assert "stop_sequences" not in kwargs
        assert kwargs["generation_config"]["stop_sequences"] == ["STOP"]
        assert kwargs["generation_config"]["max_output_tokens"] == 3


# ---------------------------------------------------------------------------
# D4 — chat history shape
# ---------------------------------------------------------------------------


class TestChatHistoryShape:
    def test_gemini_history_excludes_final_user(self) -> None:
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel.__new__(GeminiModel)
        Model.__init__(model, name="gemini", model_id="gemini")
        model.default_generation_config = {}
        model._call_count = 0
        mock_model = MagicMock()
        chat = MagicMock()
        chat.send_message.return_value = types.SimpleNamespace(text="reply")
        mock_model.start_chat.return_value = chat
        model._get_client = lambda: mock_model  # type: ignore[method-assign]

        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        assert model.chat(messages) == "reply"

        history = mock_model.start_chat.call_args.kwargs["history"]
        assert history == [
            {"role": "user", "parts": ["u1"]},
            {"role": "model", "parts": ["a1"]},
        ]
        assert chat.send_message.call_args.args[0] == "u2"

    def test_gemini_system_prefixes_current_when_no_prior_user(self) -> None:
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel.__new__(GeminiModel)
        Model.__init__(model, name="gemini", model_id="gemini")
        model.default_generation_config = {}
        model._call_count = 0
        mock_model = MagicMock()
        chat = MagicMock()
        chat.send_message.return_value = types.SimpleNamespace(text="reply")
        mock_model.start_chat.return_value = chat
        model._get_client = lambda: mock_model  # type: ignore[method-assign]

        model.chat(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
        )
        assert mock_model.start_chat.call_args.kwargs["history"] == []
        assert chat.send_message.call_args.args[0] == "[System: be brief]\nhi"

    def test_cohere_history_excludes_final_user(self) -> None:
        from insideLLMs.models.cohere import CohereModel

        model = CohereModel.__new__(CohereModel)
        Model.__init__(model, name="cohere", model_id="cohere")
        model.model_name = "command-r"
        model.default_preamble = None
        model._call_count = 0
        client = MagicMock()
        client.chat.return_value = types.SimpleNamespace(text="ok")
        model._get_client = lambda: client  # type: ignore[method-assign]

        model.chat(
            [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
            ],
        )
        kwargs = client.chat.call_args.kwargs
        assert kwargs["message"] == "u2"
        assert kwargs["chat_history"] == [
            {"role": "USER", "message": "u1"},
            {"role": "CHATBOT", "message": "a1"},
        ]

    def test_cohere_rejects_empty_and_non_user_final(self) -> None:
        from insideLLMs.models.cohere import CohereModel

        model = CohereModel.__new__(CohereModel)
        Model.__init__(model, name="cohere", model_id="cohere")
        model.model_name = "command-r"
        model.default_preamble = None
        model._call_count = 0
        client = MagicMock()
        model._get_client = lambda: client  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="non-empty"):
            model.chat([])
        with pytest.raises(ValueError, match='role="user"'):
            model.chat([{"role": "assistant", "content": "a"}])
        client.chat.assert_not_called()
