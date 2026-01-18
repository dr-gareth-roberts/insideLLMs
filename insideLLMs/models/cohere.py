"""Cohere model implementation.

This module provides a wrapper for Cohere's language models via their API.
"""

import os
from collections.abc import Iterator
from typing import Any, Optional

from insideLLMs.models.base import ChatMessage, Model
from insideLLMs.types import ModelInfo


class CohereModel(Model):
    """Model implementation for Cohere's language models via API.

    Supports text generation, chat, and streaming responses.

    Example:
        >>> model = CohereModel(model_name="command-r-plus")
        >>> response = model.generate("Explain quantum computing")
        >>> print(response)
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        name: str = "CohereModel",
        model_name: str = "command-r-plus",
        api_key: Optional[str] = None,
        default_preamble: Optional[str] = None,
    ):
        """Initialize the Cohere model.

        Args:
            name: Human-readable name for this model instance.
            model_name: The Cohere model to use (e.g., "command-r-plus", "command-r").
            api_key: Cohere API key. If not provided, uses CO_API_KEY env var.
            default_preamble: Optional system preamble for chat mode.
        """
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")
        self.default_preamble = default_preamble

        if not self.api_key:
            raise ValueError(
                "Cohere API key required. Set CO_API_KEY or COHERE_API_KEY "
                "environment variable or pass api_key parameter."
            )

        self._client = None

    def _get_client(self):
        """Lazily initialize the Cohere client."""
        if self._client is None:
            try:
                import cohere
            except ImportError:
                raise ImportError("cohere package required. Install with: pip install cohere")

            self._client = cohere.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the Cohere model.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            The model's text response.
        """
        client = self._get_client()

        # Map common parameter names
        params = {
            "model": self.model_name,
            "message": prompt,
        }

        if "temperature" in kwargs:
            params["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs.pop("max_tokens")
        if "p" in kwargs or "top_p" in kwargs:
            params["p"] = kwargs.pop("p", kwargs.pop("top_p", None))
        if "k" in kwargs or "top_k" in kwargs:
            params["k"] = kwargs.pop("k", kwargs.pop("top_k", None))

        # Add preamble if set
        if self.default_preamble:
            params["preamble"] = self.default_preamble
        if "preamble" in kwargs:
            params["preamble"] = kwargs.pop("preamble")

        params.update(kwargs)

        response = client.chat(**params)

        self._call_count += 1
        return response.text

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat conversation.

        Args:
            messages: List of chat messages with role and content.
            **kwargs: Additional arguments for generation.

        Returns:
            The assistant's response.
        """
        client = self._get_client()

        # Convert to Cohere's chat format
        chat_history = []
        preamble = self.default_preamble
        current_message = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                preamble = content
            elif role == "user":
                current_message = content
                # Don't add to history yet - only add completed turns
            elif role == "assistant" and current_message:
                # Add completed turn to history
                chat_history.append(
                    {
                        "role": "USER",
                        "message": current_message,
                    }
                )
                chat_history.append(
                    {
                        "role": "CHATBOT",
                        "message": content,
                    }
                )
                current_message = ""

        # The last user message becomes the current message
        if not current_message and messages:
            # Find last user message
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    current_message = msg.get("content", "")
                    break

        params = {
            "model": self.model_name,
            "message": current_message,
            "chat_history": chat_history if chat_history else None,
        }

        if preamble:
            params["preamble"] = preamble

        if "temperature" in kwargs:
            params["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs.pop("max_tokens")

        params.update(kwargs)

        response = client.chat(**params)

        self._call_count += 1
        return response.text

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the response from the model.

        Args:
            prompt: The input prompt.
            **kwargs: Additional arguments for generation.

        Yields:
            Chunks of the response text.
        """
        client = self._get_client()

        params = {
            "model": self.model_name,
            "message": prompt,
        }

        if "temperature" in kwargs:
            params["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs.pop("max_tokens")
        if self.default_preamble:
            params["preamble"] = self.default_preamble

        params.update(kwargs)

        response = client.chat_stream(**params)

        self._call_count += 1
        for event in response:
            if event.event_type == "text-generation":
                yield event.text

    def info(self) -> ModelInfo:
        """Return model information.

        Returns:
            ModelInfo with Cohere model details.
        """
        return ModelInfo(
            name=self.name,
            provider="Cohere",
            model_id=self.model_id,
            supports_streaming=True,
            supports_chat=True,
            extra={
                "model_name": self.model_name,
                "description": "Cohere language model via cohere SDK.",
            },
        )

    def embed(
        self,
        texts: list[str],
        input_type: str = "search_document",
        embedding_types: Optional[list[str]] = None,
    ) -> list[list[float]]:
        """Generate embeddings for the given texts.

        Args:
            texts: List of texts to embed.
            input_type: Type of input ("search_document", "search_query", etc.)
            embedding_types: Types of embeddings to return.

        Returns:
            List of embedding vectors.
        """
        client = self._get_client()

        params = {
            "texts": texts,
            "model": "embed-english-v3.0",  # Cohere's embedding model
            "input_type": input_type,
        }

        if embedding_types:
            params["embedding_types"] = embedding_types

        response = client.embed(**params)

        return response.embeddings

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: Optional[int] = None,
        model: str = "rerank-english-v3.0",
    ) -> list[dict[str, Any]]:
        """Rerank documents based on relevance to a query.

        Args:
            query: The search query.
            documents: List of documents to rerank.
            top_n: Number of top results to return.
            model: Reranking model to use.

        Returns:
            List of reranked results with scores.
        """
        client = self._get_client()

        params = {
            "query": query,
            "documents": documents,
            "model": model,
        }

        if top_n:
            params["top_n"] = top_n

        response = client.rerank(**params)

        return [
            {
                "index": r.index,
                "relevance_score": r.relevance_score,
                "document": documents[r.index],
            }
            for r in response.results
        ]
