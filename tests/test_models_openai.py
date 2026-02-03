"""Tests for insideLLMs/models/openai.py module."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests in this module if openai is not installed
pytest.importorskip("openai")


class TestOpenAIModelInit:
    """Tests for OpenAIModel initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch("insideLLMs.models.openai.OpenAI"):
            from insideLLMs.models.openai import OpenAIModel

            model = OpenAIModel(api_key="test_key", model_name="gpt-4")
            assert model.api_key == "test_key"
            assert model.model_name == "gpt-4"

    def test_init_with_env_var(self):
        """Test initialization with OPENAI_API_KEY environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env_key"}):
            with patch("insideLLMs.models.openai.OpenAI"):
                from insideLLMs.models.openai import OpenAIModel

                model = OpenAIModel(model_name="gpt-4")
                assert model.api_key == "env_key"

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises error."""
        from insideLLMs.exceptions import ModelInitializationError
        from insideLLMs.models.openai import OpenAIModel

        env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            with pytest.raises(ModelInitializationError):
                OpenAIModel(model_name="gpt-4")

    def test_init_with_custom_name(self):
        """Test initialization with custom name."""
        with patch("insideLLMs.models.openai.OpenAI"):
            from insideLLMs.models.openai import OpenAIModel

            model = OpenAIModel(name="MyOpenAIModel", api_key="test_key")
            assert model.name == "MyOpenAIModel"

    def test_default_model_name(self):
        """Test default model name."""
        with patch("insideLLMs.models.openai.OpenAI"):
            from insideLLMs.models.openai import OpenAIModel

            model = OpenAIModel(api_key="test_key")
            assert model.model_name == "gpt-3.5-turbo"


class TestOpenAIModelInfo:
    """Tests for OpenAIModel.info method."""

    def test_info_returns_model_info(self):
        """Test that info returns correct ModelInfo."""
        with patch("insideLLMs.models.openai.OpenAI"):
            from insideLLMs.models.openai import OpenAIModel

            model = OpenAIModel(api_key="test_key", model_name="gpt-4")
            info = model.info()

            assert info.name == "OpenAIModel"
            assert info.provider == "OpenAI"
            assert info.model_id == "gpt-4"
            assert info.supports_streaming is True
            assert info.supports_chat is True

    def test_info_extra_fields(self):
        """Test that info includes extra fields."""
        with patch("insideLLMs.models.openai.OpenAI"):
            from insideLLMs.models.openai import OpenAIModel

            model = OpenAIModel(api_key="test_key", model_name="gpt-4")
            info = model.info()

            assert "model_name" in info.extra
            assert info.extra["model_name"] == "gpt-4"


class TestOpenAIModelGenerate:
    """Tests for OpenAIModel.generate method."""

    def test_generate_basic(self):
        """Test basic generation."""
        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Generated response"
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            result = model.generate("Test prompt")

            assert result == "Generated response"

    def test_generate_with_kwargs(self):
        """Test generation with additional kwargs."""
        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response"
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            model.generate("Test", temperature=0.5, max_tokens=100)

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 100


class TestOpenAIModelChat:
    """Tests for OpenAIModel.chat method."""

    def test_chat_basic(self):
        """Test basic chat."""
        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Chat response"
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            messages = [{"role": "user", "content": "Hello"}]
            result = model.chat(messages)

            assert result == "Chat response"

    def test_chat_with_multiple_messages(self):
        """Test chat with multiple messages."""
        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response"
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
            result = model.chat(messages)

            assert result == "Response"


class TestOpenAIModelStream:
    """Tests for OpenAIModel.stream method."""

    def test_stream_basic(self):
        """Test basic streaming."""
        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()

            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = "Hello"

            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = " world"

            mock_client.chat.completions.create.return_value = [chunk1, chunk2]
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")
            chunks = list(model.stream("Test prompt"))

            assert chunks == ["Hello", " world"]


class TestOpenAIModelErrorHandling:
    """Tests for OpenAIModel error handling."""

    def test_generate_rate_limit_error(self):
        """Test that rate limit errors are properly handled."""
        from openai import RateLimitError as OpenAIRateLimitError

        from insideLLMs.exceptions import RateLimitError

        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = OpenAIRateLimitError(
                "Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")

            with pytest.raises(RateLimitError):
                model.generate("Test prompt")

    def test_generate_timeout_error(self):
        """Test that timeout errors are properly handled."""
        from openai import APITimeoutError

        from insideLLMs.exceptions import TimeoutError as InsideLLMsTimeoutError

        with patch("insideLLMs.models.openai.OpenAI") as MockOpenAI:
            from insideLLMs.models.openai import OpenAIModel

            mock_client = MagicMock()
            mock_request = MagicMock()
            mock_client.chat.completions.create.side_effect = APITimeoutError(mock_request)
            MockOpenAI.return_value = mock_client

            model = OpenAIModel(api_key="test_key")

            with pytest.raises(InsideLLMsTimeoutError):
                model.generate("Test prompt")
