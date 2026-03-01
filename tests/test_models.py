"""Tests for model implementations."""

from unittest.mock import patch

import pytest

from insideLLMs.models import (
    CohereModel,
    DummyModel,
    GeminiModel,
    LlamaCppModel,
    OllamaModel,
    VLLMModel,
)
from insideLLMs.models.base import ModelProtocol, ModelWrapper
from insideLLMs.types import ModelInfo


class TestDummyModel:
    """Tests for DummyModel."""

    def test_dummy_model_creation(self):
        """Test creating a DummyModel."""
        model = DummyModel()
        assert model.name == "DummyModel"
        assert model.model_id == "dummy-v1"

    def test_dummy_model_custom_name(self):
        """Test DummyModel with custom name."""
        model = DummyModel(name="CustomDummy")
        assert model.name == "CustomDummy"

    def test_dummy_model_generate(self):
        """Test DummyModel generate method."""
        model = DummyModel()
        response = model.generate("Hello, world!")

        assert isinstance(response, str)
        assert "Hello, world!" in response
        assert "[DummyModel]" in response

    def test_dummy_model_custom_prefix(self):
        """Test DummyModel with custom prefix."""
        model = DummyModel(response_prefix="[TEST]")
        response = model.generate("test")

        assert "[TEST]" in response

    def test_dummy_model_canned_response(self):
        """Test DummyModel with canned response."""
        model = DummyModel(canned_response="Always this response")
        response = model.generate("anything")

        assert response == "Always this response"

    def test_dummy_model_chat(self):
        """Test DummyModel chat method."""
        model = DummyModel()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        response = model.chat(messages)  # type: ignore[arg-type]
        assert "How are you?" in response

    def test_dummy_model_stream(self):
        """Test DummyModel stream method."""
        model = DummyModel()
        chunks = list(model.stream("Hello world"))

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert "Hello" in full_response

    def test_dummy_model_info(self):
        """Test DummyModel info method."""
        model = DummyModel()
        info = model.info()

        assert isinstance(info, ModelInfo)
        assert info.name == "DummyModel"
        assert info.provider == "Dummy"
        assert info.supports_streaming is True
        assert info.supports_chat is True


class TestModelBase:
    """Tests for Model base class functionality."""

    def test_model_generate_with_metadata(self):
        """Test generate_with_metadata method."""
        model = DummyModel()
        response = model.generate_with_metadata("test prompt")

        assert response.content is not None
        assert response.model == "dummy-v1"
        assert response.latency_ms is not None
        assert response.latency_ms > 0

    def test_model_repr(self):
        """Test model string representation."""
        model = DummyModel(name="TestModel")
        repr_str = repr(model)

        assert "DummyModel" in repr_str
        assert "TestModel" in repr_str


class TestModelWrapper:
    """Tests for ModelWrapper functionality."""

    def test_wrapper_basic(self):
        """Test basic wrapper functionality."""
        base_model = DummyModel()
        wrapped = ModelWrapper(base_model)

        response = wrapped.generate("test")
        assert isinstance(response, str)
        assert wrapped.name == "DummyModel"

    def test_wrapper_retry_on_success(self):
        """Test wrapper doesn't retry on success."""
        base_model = DummyModel()
        wrapped = ModelWrapper(base_model, max_retries=3)

        # Should work on first try
        response = wrapped.generate("test")
        assert "test" in response

    def test_wrapper_caching(self):
        """Test response caching."""
        call_count = 0

        class CountingModel(DummyModel):
            def generate(self, prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                return f"Response {call_count}"

        base_model = CountingModel()
        wrapped = ModelWrapper(base_model, cache_responses=True)

        # First call
        response1 = wrapped.generate("same prompt")
        # Second call with same prompt
        response2 = wrapped.generate("same prompt")

        assert response1 == response2
        assert call_count == 1  # Only called once due to caching

    def test_wrapper_no_caching(self):
        """Test wrapper without caching."""
        call_count = 0

        class CountingModel(DummyModel):
            def generate(self, prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                return f"Response {call_count}"

        base_model = CountingModel()
        wrapped = ModelWrapper(base_model, cache_responses=False)

        wrapped.generate("same prompt")
        wrapped.generate("same prompt")

        assert call_count == 2  # Called twice, no caching

    def test_wrapper_info(self):
        """Test wrapper info passes through."""
        base_model = DummyModel()
        wrapped = ModelWrapper(base_model)

        info = wrapped.info()
        assert info.name == "DummyModel"


class TestModelProtocol:
    """Tests for ModelProtocol."""

    def test_dummy_model_implements_protocol(self):
        """Test that DummyModel implements ModelProtocol."""
        model = DummyModel()
        assert isinstance(model, ModelProtocol)

    def test_protocol_has_required_methods(self):
        """Test protocol requires expected methods."""
        model = DummyModel()

        # These should exist and be callable
        assert hasattr(model, "name")
        assert callable(model.generate)
        assert callable(model.info)


class TestGeminiModel:
    """Tests for GeminiModel."""

    def test_gemini_model_requires_api_key(self):
        """Test that GeminiModel requires API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Google API key required"):
                GeminiModel()

    def test_gemini_model_with_api_key(self):
        """Test GeminiModel initialization with API key."""
        model = GeminiModel(api_key="test-key")
        assert model.name == "GeminiModel"
        assert model.model_name == "gemini-1.5-flash"
        assert model.api_key == "test-key"

    def test_gemini_model_custom_model_name(self):
        """Test GeminiModel with custom model name."""
        model = GeminiModel(
            api_key="test-key",
            model_name="gemini-pro",
            name="CustomGemini",
        )
        assert model.model_name == "gemini-pro"
        assert model.name == "CustomGemini"

    def test_gemini_model_info(self):
        """Test GeminiModel info method."""
        model = GeminiModel(api_key="test-key")
        info = model.info()

        assert isinstance(info, ModelInfo)
        assert info.provider == "Google"
        assert info.supports_streaming is True
        assert info.supports_chat is True

    def test_gemini_model_generate_without_sdk(self):
        """Test GeminiModel generate method raises ImportError without SDK."""
        model = GeminiModel(api_key="test-key")

        # If google-generativeai is not installed, should raise ImportError
        try:
            import google.generativeai

            pytest.skip("google-generativeai is installed, skipping import test")
        except ImportError:
            with pytest.raises(ImportError, match="google-generativeai"):
                model.generate("Hello")


class TestCohereModel:
    """Tests for CohereModel."""

    def test_cohere_model_requires_api_key(self):
        """Test that CohereModel requires API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Cohere API key required"):
                CohereModel()

    def test_cohere_model_with_api_key(self):
        """Test CohereModel initialization with API key."""
        model = CohereModel(api_key="test-key")
        assert model.name == "CohereModel"
        assert model.model_name == "command-r-plus"
        assert model.api_key == "test-key"

    def test_cohere_model_custom_settings(self):
        """Test CohereModel with custom settings."""
        model = CohereModel(
            api_key="test-key",
            model_name="command-r",
            name="CustomCohere",
            default_preamble="You are a helpful assistant.",
        )
        assert model.model_name == "command-r"
        assert model.default_preamble == "You are a helpful assistant."

    def test_cohere_model_info(self):
        """Test CohereModel info method."""
        model = CohereModel(api_key="test-key")
        info = model.info()

        assert isinstance(info, ModelInfo)
        assert info.provider == "Cohere"
        assert info.supports_streaming is True
        assert info.supports_chat is True


class TestLlamaCppModel:
    """Tests for LlamaCppModel."""

    def test_llama_cpp_model_init(self):
        """Test LlamaCppModel initialization."""
        model = LlamaCppModel(
            model_path="/path/to/model.gguf",
            name="TestLlama",
            n_ctx=2048,
        )
        assert model.name == "TestLlama"
        assert model.model_path == "/path/to/model.gguf"
        assert model.n_ctx == 2048

    def test_llama_cpp_model_info(self):
        """Test LlamaCppModel info method."""
        model = LlamaCppModel(model_path="/path/to/model.gguf")
        info = model.info()

        assert isinstance(info, ModelInfo)
        assert info.provider == "llama.cpp"
        assert info.supports_streaming is True
        assert info.supports_chat is True


class TestOllamaModel:
    """Tests for OllamaModel."""

    def test_ollama_model_init(self):
        """Test OllamaModel initialization."""
        model = OllamaModel(model_name="llama3.2")
        assert model.name == "OllamaModel"
        assert model.model_name == "llama3.2"
        assert model.base_url == "http://localhost:11434"

    def test_ollama_model_custom_url(self):
        """Test OllamaModel with custom base URL."""
        model = OllamaModel(
            model_name="mistral",
            base_url="http://custom-host:8080",
            timeout=60,
        )
        assert model.base_url == "http://custom-host:8080"
        assert model.timeout == 60

    def test_ollama_model_info(self):
        """Test OllamaModel info method."""
        model = OllamaModel(model_name="llama3.2")
        info = model.info()

        assert isinstance(info, ModelInfo)
        assert info.provider == "Ollama"
        assert info.supports_streaming is True
        assert info.supports_chat is True


class TestVLLMModel:
    """Tests for VLLMModel."""

    def test_vllm_model_init(self):
        """Test VLLMModel initialization."""
        model = VLLMModel(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:8000",
        )
        assert model.name == "VLLMModel"
        assert model.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert model.base_url == "http://localhost:8000"

    def test_vllm_model_info(self):
        """Test VLLMModel info method."""
        model = VLLMModel(model_name="test-model")
        info = model.info()

        assert isinstance(info, ModelInfo)
        assert info.provider == "vLLM"
        assert info.supports_streaming is True
        assert info.supports_chat is True
