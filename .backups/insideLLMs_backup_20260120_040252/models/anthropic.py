import os
from collections.abc import Iterator
from typing import Optional

import anthropic
from anthropic import APIError as AnthropicAPIError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import RateLimitError as AnthropicRateLimitError

from insideLLMs.exceptions import (
    APIError as InsideLLMsAPIError,
)
from insideLLMs.exceptions import (
    ModelGenerationError,
    ModelInitializationError,
    RateLimitError,
)
from insideLLMs.exceptions import (
    TimeoutError as InsideLLMsTimeoutError,
)

from .base import ChatMessage, Model


class AnthropicModel(Model):
    """Model implementation for Anthropic's Claude models via API.

    Provides robust error handling for common API failure modes.
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        name: str = "AnthropicModel",
        model_name: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ModelInitializationError(
                model_id=model_name,
                reason="ANTHROPIC_API_KEY environment variable not set and no api_key provided.",
            )
        try:
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
        except Exception as e:
            raise ModelInitializationError(
                model_id=model_name,
                reason=f"Failed to initialize Anthropic client: {e}",
            )
        self._timeout = timeout

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except AnthropicRateLimitError as e:
            raise RateLimitError(
                model_id=self.model_name,
                retry_after=getattr(e, "retry_after", None),
            )
        except AnthropicTimeoutError:
            raise InsideLLMsTimeoutError(
                model_id=self.model_name,
                timeout_seconds=self._timeout,
            )
        except AnthropicAPIError as e:
            raise InsideLLMsAPIError(
                model_id=self.model_name,
                status_code=getattr(e, "status_code", None),
                message=str(e),
            )
        except Exception as e:
            raise ModelGenerationError(
                model_id=self.model_name,
                prompt=prompt,
                reason=str(e),
                original_error=e,
            )

    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        try:
            # Convert messages to Anthropic format if needed
            anthropic_messages = []
            for msg in messages:
                role = "assistant" if msg.get("role") == "assistant" else "user"
                anthropic_messages.append({"role": role, "content": msg.get("content", "")})

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7),
                messages=anthropic_messages,
            )
            return response.content[0].text
        except AnthropicRateLimitError as e:
            raise RateLimitError(
                model_id=self.model_name,
                retry_after=getattr(e, "retry_after", None),
            )
        except AnthropicTimeoutError:
            raise InsideLLMsTimeoutError(
                model_id=self.model_name,
                timeout_seconds=self._timeout,
            )
        except AnthropicAPIError as e:
            raise InsideLLMsAPIError(
                model_id=self.model_name,
                status_code=getattr(e, "status_code", None),
                message=str(e),
            )
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
            with self.client.messages.stream(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                yield from stream.text_stream
        except AnthropicRateLimitError as e:
            raise RateLimitError(
                model_id=self.model_name,
                retry_after=getattr(e, "retry_after", None),
            )
        except AnthropicTimeoutError:
            raise InsideLLMsTimeoutError(
                model_id=self.model_name,
                timeout_seconds=self._timeout,
            )
        except AnthropicAPIError as e:
            raise InsideLLMsAPIError(
                model_id=self.model_name,
                status_code=getattr(e, "status_code", None),
                message=str(e),
            )
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
                "description": (
                    "Anthropic Claude model via API. Requires ANTHROPIC_API_KEY env variable."
                ),
            }
        )
        return base_info
