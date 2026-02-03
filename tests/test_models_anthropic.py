"""Tests for insideLLMs/models/anthropic.py module."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests in this module if anthropic is not installed
pytest.importorskip("anthropic")


class TestAnthropicModelInit:
    """Tests for AnthropicModel initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch("insideLLMs.models.anthropic.anthropic.Anthropic"):
            from insideLLMs.models.anthropic import AnthropicModel

            model = AnthropicModel(api_key="test_key", model_name="claude-3-opus-20240229")
            assert model.api_key == "test_key"
            assert model.model_name == "claude-3-opus-20240229"

    def test_init_with_env_var(self):
        """Test initialization with ANTHROPIC_API_KEY environment variable."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env_key"}):
            with patch("insideLLMs.models.anthropic.anthropic.Anthropic"):
                from insideLLMs.models.anthropic import AnthropicModel

                model = AnthropicModel(model_name="claude-3-opus-20240229")
                assert model.api_key == "env_key"

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises error."""
        from insideLLMs.exceptions import ModelInitializationError
        from insideLLMs.models.anthropic import AnthropicModel

        env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            with pytest.raises(ModelInitializationError):
                AnthropicModel(model_name="claude-3-opus-20240229")

    def test_init_with_custom_name(self):
        """Test initialization with custom name."""
        with patch("insideLLMs.models.anthropic.anthropic.Anthropic"):
            from insideLLMs.models.anthropic import AnthropicModel

            model = AnthropicModel(name="MyAnthropicModel", api_key="test_key")
            assert model.name == "MyAnthropicModel"

    def test_default_model_name(self):
        """Test default model name."""
        with patch("insideLLMs.models.anthropic.anthropic.Anthropic"):
            from insideLLMs.models.anthropic import AnthropicModel

            model = AnthropicModel(api_key="test_key")
            assert model.model_name == "claude-3-opus-20240229"


class TestAnthropicModelInfo:
    """Tests for AnthropicModel.info method."""

    def test_info_returns_model_info(self):
        """Test that info returns correct ModelInfo."""
        with patch("insideLLMs.models.anthropic.anthropic.Anthropic"):
            from insideLLMs.models.anthropic import AnthropicModel

            model = AnthropicModel(api_key="test_key", model_name="claude-3-opus-20240229")
            info = model.info()

            assert info.name == "AnthropicModel"
            assert info.provider == "Anthropic"
            assert info.model_id == "claude-3-opus-20240229"
            assert info.supports_streaming is True
            assert info.supports_chat is True

    def test_info_extra_fields(self):
        """Test that info includes extra fields."""
        with patch("insideLLMs.models.anthropic.anthropic.Anthropic"):
            from insideLLMs.models.anthropic import AnthropicModel

            model = AnthropicModel(api_key="test_key", model_name="claude-3-opus-20240229")
            info = model.info()

            assert "model_name" in info.extra
            assert info.extra["model_name"] == "claude-3-opus-20240229"


class TestAnthropicModelGenerate:
    """Tests for AnthropicModel.generate method."""

    def test_generate_basic(self):
        """Test basic generation."""
        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Generated response"
            mock_client.messages.create.return_value = mock_response
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            result = model.generate("Test prompt")

            assert result == "Generated response"

    def test_generate_with_kwargs(self):
        """Test generation with additional kwargs."""
        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Response"
            mock_client.messages.create.return_value = mock_response
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            model.generate("Test", temperature=0.5, max_tokens=100)

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 100


class TestAnthropicModelChat:
    """Tests for AnthropicModel.chat method."""

    def test_chat_basic(self):
        """Test basic chat."""
        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Chat response"
            mock_client.messages.create.return_value = mock_response
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            messages = [{"role": "user", "content": "Hello"}]
            result = model.chat(messages)

            assert result == "Chat response"

    def test_chat_with_multiple_messages(self):
        """Test chat with multiple messages."""
        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Response"
            mock_client.messages.create.return_value = mock_response
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
            result = model.chat(messages)

            assert result == "Response"


class TestAnthropicModelStream:
    """Tests for AnthropicModel.stream method."""

    def test_stream_basic(self):
        """Test basic streaming."""
        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()

            # Create a mock context manager for stream
            mock_stream = MagicMock()
            mock_stream.text_stream = ["Hello", " world"]
            mock_stream.__enter__ = MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = MagicMock(return_value=False)

            mock_client.messages.stream.return_value = mock_stream
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")
            chunks = list(model.stream("Test prompt"))

            assert chunks == ["Hello", " world"]


class TestAnthropicModelErrorHandling:
    """Tests for AnthropicModel error handling."""

    def test_generate_rate_limit_error(self):
        """Test that rate limit errors are properly handled."""
        from anthropic import RateLimitError as AnthropicRateLimitError

        from insideLLMs.exceptions import RateLimitError

        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_client.messages.create.side_effect = AnthropicRateLimitError(
                "Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")

            with pytest.raises(RateLimitError):
                model.generate("Test prompt")

    def test_generate_timeout_error(self):
        """Test that timeout errors are properly handled."""
        from anthropic import APITimeoutError

        from insideLLMs.exceptions import TimeoutError as InsideLLMsTimeoutError

        with patch("insideLLMs.models.anthropic.anthropic.Anthropic") as MockAnthropic:
            from insideLLMs.models.anthropic import AnthropicModel

            mock_client = MagicMock()
            mock_request = MagicMock()
            mock_client.messages.create.side_effect = APITimeoutError(mock_request)
            MockAnthropic.return_value = mock_client

            model = AnthropicModel(api_key="test_key")

            with pytest.raises(InsideLLMsTimeoutError):
                model.generate("Test prompt")
