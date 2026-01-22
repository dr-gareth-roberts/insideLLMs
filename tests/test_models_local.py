"""Tests for insideLLMs/models/local.py module."""

from unittest.mock import MagicMock, patch

import pytest


class TestLlamaCppModelInit:
    """Tests for LlamaCppModel initialization."""

    def test_init_stores_params(self):
        """Test that init stores parameters correctly."""
        from insideLLMs.models.local import LlamaCppModel

        model = LlamaCppModel(
            model_path="/path/to/model.gguf",
            name="TestModel",
            n_ctx=2048,
            n_gpu_layers=10,
            seed=42,
            f16_kv=False,
            verbose=True,
        )

        assert model.model_path == "/path/to/model.gguf"
        assert model.name == "TestModel"
        assert model.n_ctx == 2048
        assert model.n_gpu_layers == 10
        assert model.seed == 42
        assert model.f16_kv is False
        assert model.verbose is True
        assert model._model is None

    def test_init_default_values(self):
        """Test default values are set correctly."""
        from insideLLMs.models.local import LlamaCppModel

        model = LlamaCppModel(model_path="/path/to/model.gguf")

        assert model.name == "LlamaCppModel"
        assert model.n_ctx == 4096
        assert model.n_gpu_layers == -1
        assert model.seed == -1
        assert model.f16_kv is True
        assert model.verbose is False

    def test_model_id_from_basename(self):
        """Test model_id is derived from path basename."""
        from insideLLMs.models.local import LlamaCppModel

        model = LlamaCppModel(model_path="/long/path/to/llama-7b.gguf")

        assert model.model_id == "llama-7b.gguf"


class TestLlamaCppModelGetModel:
    """Tests for LlamaCppModel._get_model method."""

    def test_get_model_import_error(self):
        """Test that ImportError is raised when llama-cpp not installed."""
        from insideLLMs.models.local import LlamaCppModel

        model = LlamaCppModel(model_path="/path/to/model.gguf")

        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError, match="llama-cpp-python"):
                model._get_model()

    def test_get_model_lazy_init(self):
        """Test lazy initialization of model."""
        with patch("insideLLMs.models.local.LlamaCppModel._get_model") as mock_get:
            from insideLLMs.models.local import LlamaCppModel

            mock_llama = MagicMock()
            mock_get.return_value = mock_llama

            model = LlamaCppModel(model_path="/path/to/model.gguf")

            # _model should be None initially
            assert model._model is None


class TestLlamaCppModelGenerate:
    """Tests for LlamaCppModel.generate method."""

    def test_generate_basic(self):
        """Test basic generation."""
        from insideLLMs.models.local import LlamaCppModel

        model = LlamaCppModel(model_path="/path/to/model.gguf")

        mock_llama = MagicMock()
        mock_llama.return_value = {"choices": [{"text": "Generated response"}]}

        with patch.object(model, "_get_model", return_value=mock_llama):
            result = model.generate("Test prompt")

        assert result == "Generated response"
        assert model._call_count == 1

    def test_generate_with_kwargs(self):
        """Test generation with custom kwargs."""
        from insideLLMs.models.local import LlamaCppModel

        model = LlamaCppModel(model_path="/path/to/model.gguf")

        mock_llama = MagicMock()
        mock_llama.return_value = {"choices": [{"text": "Response"}]}

        with patch.object(model, "_get_model", return_value=mock_llama):
            model.generate("Test", temperature=0.5, max_tokens=100)

        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100


class TestLlamaCppModelChat:
    """Tests for LlamaCppModel.chat method."""

    def test_chat_basic(self):
        """Test basic chat."""
        from insideLLMs.models.local import LlamaCppModel

        model = LlamaCppModel(model_path="/path/to/model.gguf")

        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Chat response"}}]
        }

        with patch.object(model, "_get_model", return_value=mock_llama):
            messages = [{"role": "user", "content": "Hello"}]
            result = model.chat(messages)

        assert result == "Chat response"
        assert model._call_count == 1


class TestLlamaCppModelStream:
    """Tests for LlamaCppModel.stream method."""

    def test_stream_basic(self):
        """Test basic streaming."""
        from insideLLMs.models.local import LlamaCppModel

        model = LlamaCppModel(model_path="/path/to/model.gguf")

        mock_llama = MagicMock()
        mock_llama.return_value = iter(
            [
                {"choices": [{"text": "Hello"}]},
                {"choices": [{"text": " world"}]},
            ]
        )

        with patch.object(model, "_get_model", return_value=mock_llama):
            chunks = list(model.stream("Test prompt"))

        assert chunks == ["Hello", " world"]


class TestLlamaCppModelInfo:
    """Tests for LlamaCppModel.info method."""

    def test_info_returns_model_info(self):
        """Test info returns correct ModelInfo."""
        from insideLLMs.models.local import LlamaCppModel

        model = LlamaCppModel(
            model_path="/path/to/llama-7b.gguf",
            name="MyLlama",
            n_ctx=2048,
            n_gpu_layers=20,
        )
        info = model.info()

        assert info.name == "MyLlama"
        assert info.provider == "llama.cpp"
        assert info.model_id == "llama-7b.gguf"
        assert info.supports_streaming is True
        assert info.supports_chat is True
        assert info.extra["n_ctx"] == 2048
        assert info.extra["n_gpu_layers"] == 20


class TestOllamaModelInit:
    """Tests for OllamaModel initialization."""

    def test_init_default_values(self):
        """Test default values are set correctly."""
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel()

        assert model.model_name == "llama3.2"
        assert model.name == "OllamaModel"
        assert model.base_url == "http://localhost:11434"
        assert model.timeout == 120
        assert model._client is None

    def test_init_custom_values(self):
        """Test custom initialization."""
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel(
            model_name="mistral",
            name="MyOllama",
            base_url="http://custom:8080",
            timeout=60,
        )

        assert model.model_name == "mistral"
        assert model.name == "MyOllama"
        assert model.base_url == "http://custom:8080"
        assert model.timeout == 60


class TestOllamaModelGetClient:
    """Tests for OllamaModel._get_client method."""

    def test_get_client_import_error(self):
        """Test that ImportError is raised when ollama not installed."""
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel()

        with patch.dict("sys.modules", {"ollama": None}):
            with pytest.raises(ImportError, match="ollama"):
                model._get_client()


class TestOllamaModelGenerate:
    """Tests for OllamaModel.generate method."""

    def test_generate_basic(self):
        """Test basic generation."""
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel()

        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "Generated response"}

        with patch.object(model, "_get_client", return_value=mock_client):
            result = model.generate("Test prompt")

        assert result == "Generated response"
        assert model._call_count == 1


class TestOllamaModelChat:
    """Tests for OllamaModel.chat method."""

    def test_chat_basic(self):
        """Test basic chat."""
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel()

        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "Chat response"}}

        with patch.object(model, "_get_client", return_value=mock_client):
            messages = [{"role": "user", "content": "Hello"}]
            result = model.chat(messages)

        assert result == "Chat response"
        assert model._call_count == 1


class TestOllamaModelStream:
    """Tests for OllamaModel.stream method."""

    def test_stream_basic(self):
        """Test basic streaming."""
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel()

        mock_client = MagicMock()
        mock_client.generate.return_value = iter(
            [
                {"response": "Hello"},
                {"response": " world"},
            ]
        )

        with patch.object(model, "_get_client", return_value=mock_client):
            chunks = list(model.stream("Test prompt"))

        assert chunks == ["Hello", " world"]


class TestOllamaModelInfo:
    """Tests for OllamaModel.info method."""

    def test_info_returns_model_info(self):
        """Test info returns correct ModelInfo."""
        from insideLLMs.models.local import OllamaModel

        model = OllamaModel(model_name="llama3.2", name="MyOllama")
        info = model.info()

        assert info.name == "MyOllama"
        assert info.provider == "Ollama"
        assert info.model_id == "llama3.2"
        assert info.supports_streaming is True
        assert info.supports_chat is True
        assert info.extra["base_url"] == "http://localhost:11434"
