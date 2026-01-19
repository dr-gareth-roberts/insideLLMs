import os
from collections.abc import Iterator
from typing import Optional

from openai import APIError, APITimeoutError, OpenAI
from openai import RateLimitError as OpenAIRateLimitError

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


class OpenAIModel(Model):
    """Model implementation for OpenAI's GPT models via API (openai>=1.0.0).

    Provides robust error handling for common API failure modes.
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        name: str = "OpenAIModel",
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ModelInitializationError(
                model_id=model_name,
                reason="OPENAI_API_KEY environment variable not set and no api_key provided.",
            )
        try:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
                organization=organization,
                project=project,
                timeout=timeout,
                max_retries=max_retries,
            )
        except Exception as e:
            raise ModelInitializationError(
                model_id=model_name,
                reason=f"Failed to initialize OpenAI client: {e}",
            )
        self._base_url = base_url
        self._organization = organization
        self._project = project
        self._timeout = timeout

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except OpenAIRateLimitError as e:
            raise RateLimitError(
                model_id=self.model_name,
                retry_after=getattr(e, "retry_after", None),
            )
        except APITimeoutError:
            raise InsideLLMsTimeoutError(
                model_id=self.model_name,
                timeout_seconds=self._timeout,
            )
        except APIError as e:
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
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except OpenAIRateLimitError as e:
            raise RateLimitError(
                model_id=self.model_name,
                retry_after=getattr(e, "retry_after", None),
            )
        except APITimeoutError:
            raise InsideLLMsTimeoutError(
                model_id=self.model_name,
                timeout_seconds=self._timeout,
            )
        except APIError as e:
            raise InsideLLMsAPIError(
                model_id=self.model_name,
                status_code=getattr(e, "status_code", None),
                message=str(e),
            )
        except Exception as e:
            # Get first message content for error context
            first_msg = messages[0]["content"] if messages else ""
            raise ModelGenerationError(
                model_id=self.model_name,
                prompt=first_msg,
                reason=str(e),
                original_error=e,
            )

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except OpenAIRateLimitError as e:
            raise RateLimitError(
                model_id=self.model_name,
                retry_after=getattr(e, "retry_after", None),
            )
        except APITimeoutError:
            raise InsideLLMsTimeoutError(
                model_id=self.model_name,
                timeout_seconds=self._timeout,
            )
        except APIError as e:
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
                "description": ("OpenAI GPT model via API. Requires OPENAI_API_KEY env variable."),
                "base_url": self._base_url,
                "organization": self._organization,
                "project": self._project,
            }
        )
        return base_info
