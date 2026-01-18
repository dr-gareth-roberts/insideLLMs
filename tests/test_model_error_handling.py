"""Tests for model error handling."""

import pytest
from unittest.mock import MagicMock, patch

from insideLLMs.exceptions import (
    ModelInitializationError,
    ModelGenerationError,
    RateLimitError,
    TimeoutError as InsideLLMsTimeoutError,
    APIError as InsideLLMsAPIError,
)


class TestOpenAIModelErrorHandling:
    """Tests for OpenAI model error handling."""

    def test_openai_missing_api_key(self):
        """Test that missing API key raises ModelInitializationError."""
        with patch.dict("os.environ", {}, clear=True):
            from insideLLMs.models.openai import OpenAIModel
            with pytest.raises(ModelInitializationError) as exc_info:
                OpenAIModel()
            assert "OPENAI_API_KEY" in str(exc_info.value)

    @patch("insideLLMs.models.openai.OpenAI")
    def test_openai_rate_limit_error(self, mock_openai_class):
        """Test that rate limit errors are converted."""
        from openai import RateLimitError as OpenAIRateLimitError
        from insideLLMs.models.openai import OpenAIModel

        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = OpenAIRateLimitError(
            "Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )

        model = OpenAIModel(api_key="test-key")
        with pytest.raises(RateLimitError):
            model.generate("test prompt")

    @patch("insideLLMs.models.openai.OpenAI")
    def test_openai_timeout_error(self, mock_openai_class):
        """Test that timeout errors are converted."""
        from openai import APITimeoutError
        from insideLLMs.models.openai import OpenAIModel

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock()
        )

        model = OpenAIModel(api_key="test-key")
        with pytest.raises(InsideLLMsTimeoutError):
            model.generate("test prompt")

    @patch("insideLLMs.models.openai.OpenAI")
    def test_openai_api_error(self, mock_openai_class):
        """Test that API errors are converted."""
        from openai import APIError
        from insideLLMs.models.openai import OpenAIModel

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APIError(
            "API error",
            request=MagicMock(),
            body=None,
        )

        model = OpenAIModel(api_key="test-key")
        with pytest.raises(InsideLLMsAPIError):
            model.generate("test prompt")

    @patch("insideLLMs.models.openai.OpenAI")
    def test_openai_generic_error(self, mock_openai_class):
        """Test that generic errors are wrapped in ModelGenerationError."""
        from insideLLMs.models.openai import OpenAIModel

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Something went wrong")

        model = OpenAIModel(api_key="test-key")
        with pytest.raises(ModelGenerationError) as exc_info:
            model.generate("test prompt")
        assert "Something went wrong" in str(exc_info.value)


class TestAnthropicModelErrorHandling:
    """Tests for Anthropic model error handling."""

    def test_anthropic_missing_api_key(self):
        """Test that missing API key raises ModelInitializationError."""
        with patch.dict("os.environ", {}, clear=True):
            from insideLLMs.models.anthropic import AnthropicModel
            with pytest.raises(ModelInitializationError) as exc_info:
                AnthropicModel()
            assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    @patch("insideLLMs.models.anthropic.anthropic.Anthropic")
    def test_anthropic_rate_limit_error(self, mock_anthropic_class):
        """Test that rate limit errors are converted."""
        from anthropic import RateLimitError as AnthropicRateLimitError
        from insideLLMs.models.anthropic import AnthropicModel

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = AnthropicRateLimitError(
            "Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )

        model = AnthropicModel(api_key="test-key")
        with pytest.raises(RateLimitError):
            model.generate("test prompt")

    @patch("insideLLMs.models.anthropic.anthropic.Anthropic")
    def test_anthropic_generic_error(self, mock_anthropic_class):
        """Test that generic errors are wrapped."""
        from insideLLMs.models.anthropic import AnthropicModel

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("Something went wrong")

        model = AnthropicModel(api_key="test-key")
        with pytest.raises(ModelGenerationError):
            model.generate("test prompt")


class TestHuggingFaceModelErrorHandling:
    """Tests for HuggingFace model error handling."""

    @patch("insideLLMs.models.huggingface.AutoTokenizer")
    def test_huggingface_tokenizer_load_error(self, mock_tokenizer):
        """Test that tokenizer load errors raise ModelInitializationError."""
        from insideLLMs.models.huggingface import HuggingFaceModel

        mock_tokenizer.from_pretrained.side_effect = Exception("Tokenizer not found")

        with pytest.raises(ModelInitializationError) as exc_info:
            HuggingFaceModel(model_name="nonexistent-model")
        assert "tokenizer" in str(exc_info.value).lower()

    @patch("insideLLMs.models.huggingface.AutoTokenizer")
    @patch("insideLLMs.models.huggingface.AutoModelForCausalLM")
    def test_huggingface_model_load_error(self, mock_model, mock_tokenizer):
        """Test that model load errors raise ModelInitializationError."""
        from insideLLMs.models.huggingface import HuggingFaceModel

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.side_effect = Exception("Model not found")

        with pytest.raises(ModelInitializationError) as exc_info:
            HuggingFaceModel(model_name="nonexistent-model")
        assert "model" in str(exc_info.value).lower()

    @patch("insideLLMs.models.huggingface.AutoTokenizer")
    @patch("insideLLMs.models.huggingface.AutoModelForCausalLM")
    @patch("insideLLMs.models.huggingface.pipeline")
    def test_huggingface_generation_error(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test that generation errors are wrapped."""
        from insideLLMs.models.huggingface import HuggingFaceModel

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_generator = MagicMock()
        mock_pipeline.return_value = mock_generator
        mock_generator.side_effect = Exception("Generation failed")

        model = HuggingFaceModel(model_name="gpt2")
        with pytest.raises(ModelGenerationError):
            model.generate("test prompt")
