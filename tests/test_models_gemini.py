"""Tests for insideLLMs/models/gemini.py module."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestGeminiModelInit:
    """Tests for GeminiModel initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key", model_name="gemini-pro")
        assert model.api_key == "test_key"
        assert model.model_name == "gemini-pro"

    def test_init_with_env_var(self):
        """Test initialization with GOOGLE_API_KEY environment variable."""
        from insideLLMs.models.gemini import GeminiModel

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env_key"}):
            model = GeminiModel(model_name="gemini-pro")
            assert model.api_key == "env_key"

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises ValueError."""
        from insideLLMs.models.gemini import GeminiModel

        # Clear API key env var
        env_without_key = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                GeminiModel(model_name="gemini-pro")

    def test_init_with_custom_name(self):
        """Test initialization with custom name."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(name="MyGeminiModel", api_key="test_key")
        assert model.name == "MyGeminiModel"

    def test_init_with_safety_settings(self):
        """Test initialization with safety settings."""
        from insideLLMs.models.gemini import GeminiModel

        safety = [{"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"}]
        model = GeminiModel(api_key="test_key", safety_settings=safety)
        assert model.safety_settings == safety

    def test_init_with_generation_config(self):
        """Test initialization with generation config."""
        from insideLLMs.models.gemini import GeminiModel

        config = {"temperature": 0.7, "max_output_tokens": 1000}
        model = GeminiModel(api_key="test_key", generation_config=config)
        assert model.default_generation_config == config

    def test_default_model_name(self):
        """Test default model name."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        assert model.model_name == "gemini-1.5-flash"


class TestGeminiModelInfo:
    """Tests for GeminiModel.info method."""

    def test_info_returns_model_info(self):
        """Test that info returns correct ModelInfo."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key", model_name="gemini-pro")
        info = model.info()

        assert info.name == "GeminiModel"
        assert info.provider == "Google"
        assert info.model_id == "gemini-pro"
        assert info.supports_streaming is True
        assert info.supports_chat is True

    def test_info_extra_fields(self):
        """Test that info includes extra fields."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key", model_name="gemini-pro")
        info = model.info()

        assert "model_name" in info.extra
        assert info.extra["model_name"] == "gemini-pro"


class TestGeminiModelClientInit:
    """Tests for GeminiModel client initialization."""

    def test_client_is_none_initially(self):
        """Test that client is None after init."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")
        assert model._client is None
        assert model._model is None


class TestGeminiModelGenerate:
    """Tests for GeminiModel.generate method with mocked client."""

    def test_generate_basic(self):
        """Test basic generation with mocked model."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")

        # Create mock model
        mock_genai = MagicMock()
        mock_genai_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated response"
        mock_genai_model.generate_content.return_value = mock_response

        model._client = mock_genai
        model._model = mock_genai_model

        result = model.generate("Test prompt")

        assert result == "Generated response"
        mock_genai_model.generate_content.assert_called_once()
        assert model._call_count == 1

    def test_generate_with_temperature(self):
        """Test generation with temperature parameter."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")

        mock_genai_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_genai_model.generate_content.return_value = mock_response

        model._client = MagicMock()
        model._model = mock_genai_model

        model.generate("Test", temperature=0.7)

        call_kwargs = mock_genai_model.generate_content.call_args[1]
        assert call_kwargs["generation_config"]["temperature"] == 0.7

    def test_generate_with_max_tokens(self):
        """Test generation with max_tokens parameter."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")

        mock_genai_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_genai_model.generate_content.return_value = mock_response

        model._client = MagicMock()
        model._model = mock_genai_model

        model.generate("Test", max_tokens=100)

        call_kwargs = mock_genai_model.generate_content.call_args[1]
        assert call_kwargs["generation_config"]["max_output_tokens"] == 100


class TestGeminiModelChat:
    """Tests for GeminiModel.chat method."""

    def test_chat_basic(self):
        """Test basic chat."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")

        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Chat response"
        mock_chat.send_message.return_value = mock_response

        mock_genai_model = MagicMock()
        mock_genai_model.start_chat.return_value = mock_chat

        model._client = MagicMock()
        model._model = mock_genai_model

        messages = [{"role": "user", "content": "Hello"}]
        result = model.chat(messages)

        assert result == "Chat response"

    def test_chat_with_system_message(self):
        """Test chat with system message."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")

        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_chat.send_message.return_value = mock_response

        mock_genai_model = MagicMock()
        mock_genai_model.start_chat.return_value = mock_chat

        model._client = MagicMock()
        model._model = mock_genai_model

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = model.chat(messages)

        assert result == "Response"


class TestGeminiModelStream:
    """Tests for GeminiModel.stream method."""

    def test_stream_basic(self):
        """Test basic streaming."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")

        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = " world"

        mock_genai_model = MagicMock()
        mock_genai_model.generate_content.return_value = [chunk1, chunk2]

        model._client = MagicMock()
        model._model = mock_genai_model

        chunks = list(model.stream("Test prompt"))

        assert chunks == ["Hello", " world"]


class TestGeminiModelCountTokens:
    """Tests for GeminiModel.count_tokens method."""

    def test_count_tokens(self):
        """Test token counting."""
        from insideLLMs.models.gemini import GeminiModel

        model = GeminiModel(api_key="test_key")

        mock_result = MagicMock()
        mock_result.total_tokens = 42

        mock_genai_model = MagicMock()
        mock_genai_model.count_tokens.return_value = mock_result

        model._client = MagicMock()
        model._model = mock_genai_model

        result = model.count_tokens("Hello world")

        assert result == 42
