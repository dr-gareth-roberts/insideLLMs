"""Tests for insideLLMs/models/huggingface.py module."""

from unittest.mock import MagicMock, patch

import pytest


class TestHuggingFaceModelInit:
    """Tests for HuggingFaceModel initialization."""

    def test_init_loads_components(self):
        """Test that init loads all components correctly."""
        with patch(
            "insideLLMs.models.huggingface.AutoTokenizer"
        ) as MockTokenizer, patch(
            "insideLLMs.models.huggingface.AutoModelForCausalLM"
        ) as MockModel, patch(
            "insideLLMs.models.huggingface.pipeline"
        ) as mock_pipeline:
            from insideLLMs.models.huggingface import HuggingFaceModel

            MockTokenizer.from_pretrained.return_value = MagicMock()
            MockModel.from_pretrained.return_value = MagicMock()
            mock_pipeline.return_value = MagicMock()

            model = HuggingFaceModel(
                name="TestModel", model_name="gpt2", device=0
            )

            assert model.name == "TestModel"
            assert model.model_name == "gpt2"
            assert model.device == 0
            MockTokenizer.from_pretrained.assert_called_once_with("gpt2")
            MockModel.from_pretrained.assert_called_once_with("gpt2")

    def test_init_tokenizer_error(self):
        """Test that tokenizer error raises ModelInitializationError."""
        from insideLLMs.exceptions import ModelInitializationError

        with patch(
            "insideLLMs.models.huggingface.AutoTokenizer"
        ) as MockTokenizer:
            from insideLLMs.models.huggingface import HuggingFaceModel

            MockTokenizer.from_pretrained.side_effect = Exception(
                "Tokenizer error"
            )

            with pytest.raises(ModelInitializationError) as exc_info:
                HuggingFaceModel(model_name="bad-model")

            assert "Failed to load tokenizer" in str(exc_info.value)

    def test_init_model_error(self):
        """Test that model error raises ModelInitializationError."""
        from insideLLMs.exceptions import ModelInitializationError

        with patch(
            "insideLLMs.models.huggingface.AutoTokenizer"
        ) as MockTokenizer, patch(
            "insideLLMs.models.huggingface.AutoModelForCausalLM"
        ) as MockModel:
            from insideLLMs.models.huggingface import HuggingFaceModel

            MockTokenizer.from_pretrained.return_value = MagicMock()
            MockModel.from_pretrained.side_effect = Exception("Model error")

            with pytest.raises(ModelInitializationError) as exc_info:
                HuggingFaceModel(model_name="bad-model")

            assert "Failed to load model" in str(exc_info.value)

    def test_init_pipeline_error(self):
        """Test that pipeline error raises ModelInitializationError."""
        from insideLLMs.exceptions import ModelInitializationError

        with patch(
            "insideLLMs.models.huggingface.AutoTokenizer"
        ) as MockTokenizer, patch(
            "insideLLMs.models.huggingface.AutoModelForCausalLM"
        ) as MockModel, patch(
            "insideLLMs.models.huggingface.pipeline"
        ) as mock_pipeline:
            from insideLLMs.models.huggingface import HuggingFaceModel

            MockTokenizer.from_pretrained.return_value = MagicMock()
            MockModel.from_pretrained.return_value = MagicMock()
            mock_pipeline.side_effect = Exception("Pipeline error")

            with pytest.raises(ModelInitializationError) as exc_info:
                HuggingFaceModel(model_name="bad-model")

            assert "Failed to create pipeline" in str(exc_info.value)


class TestHuggingFaceModelGenerate:
    """Tests for HuggingFaceModel.generate method."""

    def test_generate_basic(self):
        """Test basic generation."""
        with patch(
            "insideLLMs.models.huggingface.AutoTokenizer"
        ) as MockTokenizer, patch(
            "insideLLMs.models.huggingface.AutoModelForCausalLM"
        ) as MockModel, patch(
            "insideLLMs.models.huggingface.pipeline"
        ) as mock_pipeline:
            from insideLLMs.models.huggingface import HuggingFaceModel

            MockTokenizer.from_pretrained.return_value = MagicMock()
            MockModel.from_pretrained.return_value = MagicMock()

            mock_generator = MagicMock()
            mock_generator.return_value = [{"generated_text": "Hello world!"}]
            mock_pipeline.return_value = mock_generator

            model = HuggingFaceModel()
            result = model.generate("Hello")

            assert result == "Hello world!"
            mock_generator.assert_called_once_with("Hello")

    def test_generate_error(self):
        """Test that generation errors are wrapped."""
        from insideLLMs.exceptions import ModelGenerationError

        with patch(
            "insideLLMs.models.huggingface.AutoTokenizer"
        ) as MockTokenizer, patch(
            "insideLLMs.models.huggingface.AutoModelForCausalLM"
        ) as MockModel, patch(
            "insideLLMs.models.huggingface.pipeline"
        ) as mock_pipeline:
            from insideLLMs.models.huggingface import HuggingFaceModel

            MockTokenizer.from_pretrained.return_value = MagicMock()
            MockModel.from_pretrained.return_value = MagicMock()

            mock_generator = MagicMock()
            mock_generator.side_effect = Exception("Generation failed")
            mock_pipeline.return_value = mock_generator

            model = HuggingFaceModel()

            with pytest.raises(ModelGenerationError):
                model.generate("Hello")


class TestHuggingFaceModelChat:
    """Tests for HuggingFaceModel.chat method."""

    def test_chat_basic(self):
        """Test basic chat."""
        with patch(
            "insideLLMs.models.huggingface.AutoTokenizer"
        ) as MockTokenizer, patch(
            "insideLLMs.models.huggingface.AutoModelForCausalLM"
        ) as MockModel, patch(
            "insideLLMs.models.huggingface.pipeline"
        ) as mock_pipeline:
            from insideLLMs.models.huggingface import HuggingFaceModel

            MockTokenizer.from_pretrained.return_value = MagicMock()
            MockModel.from_pretrained.return_value = MagicMock()

            mock_generator = MagicMock()
            mock_generator.return_value = [{"generated_text": "Chat response"}]
            mock_pipeline.return_value = mock_generator

            model = HuggingFaceModel()
            messages = [{"role": "user", "content": "Hello"}]
            result = model.chat(messages)

            assert result == "Chat response"

    def test_chat_error(self):
        """Test that chat errors are wrapped."""
        from insideLLMs.exceptions import ModelGenerationError

        with patch(
            "insideLLMs.models.huggingface.AutoTokenizer"
        ) as MockTokenizer, patch(
            "insideLLMs.models.huggingface.AutoModelForCausalLM"
        ) as MockModel, patch(
            "insideLLMs.models.huggingface.pipeline"
        ) as mock_pipeline:
            from insideLLMs.models.huggingface import HuggingFaceModel

            MockTokenizer.from_pretrained.return_value = MagicMock()
            MockModel.from_pretrained.return_value = MagicMock()

            mock_generator = MagicMock()
            mock_generator.side_effect = Exception("Chat failed")
            mock_pipeline.return_value = mock_generator

            model = HuggingFaceModel()
            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(ModelGenerationError):
                model.chat(messages)


class TestHuggingFaceModelStream:
    """Tests for HuggingFaceModel.stream method."""

    def test_stream_basic(self):
        """Test basic streaming."""
        with patch(
            "insideLLMs.models.huggingface.AutoTokenizer"
        ) as MockTokenizer, patch(
            "insideLLMs.models.huggingface.AutoModelForCausalLM"
        ) as MockModel, patch(
            "insideLLMs.models.huggingface.pipeline"
        ) as mock_pipeline:
            from insideLLMs.models.huggingface import HuggingFaceModel

            MockTokenizer.from_pretrained.return_value = MagicMock()
            MockModel.from_pretrained.return_value = MagicMock()

            mock_generator = MagicMock()
            mock_generator.return_value = [{"generated_text": "Stream output"}]
            mock_pipeline.return_value = mock_generator

            model = HuggingFaceModel()
            chunks = list(model.stream("Hello"))

            assert chunks == ["Stream output"]

    def test_stream_error(self):
        """Test that stream errors are wrapped."""
        from insideLLMs.exceptions import ModelGenerationError

        with patch(
            "insideLLMs.models.huggingface.AutoTokenizer"
        ) as MockTokenizer, patch(
            "insideLLMs.models.huggingface.AutoModelForCausalLM"
        ) as MockModel, patch(
            "insideLLMs.models.huggingface.pipeline"
        ) as mock_pipeline:
            from insideLLMs.models.huggingface import HuggingFaceModel

            MockTokenizer.from_pretrained.return_value = MagicMock()
            MockModel.from_pretrained.return_value = MagicMock()

            mock_generator = MagicMock()
            mock_generator.side_effect = Exception("Stream failed")
            mock_pipeline.return_value = mock_generator

            model = HuggingFaceModel()

            with pytest.raises(ModelGenerationError):
                list(model.stream("Hello"))


class TestHuggingFaceModelInfo:
    """Tests for HuggingFaceModel.info method."""

    def test_info_returns_model_info(self):
        """Test info returns correct ModelInfo."""
        with patch(
            "insideLLMs.models.huggingface.AutoTokenizer"
        ) as MockTokenizer, patch(
            "insideLLMs.models.huggingface.AutoModelForCausalLM"
        ) as MockModel, patch(
            "insideLLMs.models.huggingface.pipeline"
        ) as mock_pipeline:
            from insideLLMs.models.huggingface import HuggingFaceModel

            MockTokenizer.from_pretrained.return_value = MagicMock()
            MockModel.from_pretrained.return_value = MagicMock()
            mock_pipeline.return_value = MagicMock()

            model = HuggingFaceModel(name="TestModel", model_name="gpt2", device=0)
            info = model.info()

            assert info.name == "TestModel"
            assert info.extra["model_name"] == "gpt2"
            assert info.extra["device"] == 0
