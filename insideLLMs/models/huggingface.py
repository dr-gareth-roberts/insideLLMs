from collections.abc import Iterator

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from insideLLMs.exceptions import (
    ModelGenerationError,
    ModelInitializationError,
)

from .base import ChatMessage, Model


class HuggingFaceModel(Model):
    """Model implementation for HuggingFace Transformers models.

    Provides robust error handling for model loading and generation.
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        name: str = "HuggingFaceModel",
        model_name: str = "gpt2",
        device: int = -1,
    ):
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.device = device

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise ModelInitializationError(
                model_id=model_name,
                reason=f"Failed to load tokenizer: {e}",
            )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            raise ModelInitializationError(
                model_id=model_name,
                reason=f"Failed to load model: {e}",
            )

        try:
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
            )
        except Exception as e:
            raise ModelInitializationError(
                model_id=model_name,
                reason=f"Failed to create pipeline: {e}",
            )

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            outputs = self.generator(prompt, **kwargs)
            return outputs[0]["generated_text"]
        except Exception as e:
            raise ModelGenerationError(
                model_id=self.model_name,
                prompt=prompt,
                reason=str(e),
                original_error=e,
            )

    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        try:
            # Simple chat: concatenate messages
            prompt = "\n".join([m.get("content", "") for m in messages])
            outputs = self.generator(prompt, **kwargs)
            return outputs[0]["generated_text"]
        except Exception as e:
            first_msg = messages[0]["content"] if messages else ""
            raise ModelGenerationError(
                model_id=self.model_name,
                prompt=first_msg,
                reason=str(e),
                original_error=e,
            )

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        try:
            # Streaming not natively supported; yield the full output as one chunk
            outputs = self.generator(prompt, **kwargs)
            yield outputs[0]["generated_text"]
        except Exception as e:
            raise ModelGenerationError(
                model_id=self.model_name,
                prompt=prompt,
                reason=str(e),
                original_error=e,
            )

    def info(self):
        base_info = super().info()
        base_info.extra.update(
            {
                "model_name": self.model_name,
                "device": self.device,
                "description": "HuggingFace Transformers model via pipeline.",
            }
        )
        return base_info
