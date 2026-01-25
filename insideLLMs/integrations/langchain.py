from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any, Optional

from insideLLMs.models.base import ChatMessage, ModelProtocol


class LangChainIntegrationError(ImportError):
    """Raised when LangChain/LangGraph integration dependencies are missing."""


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, list):
        # LangChain message content can be a list of blocks (e.g., multimodal).
        # Keep it readable and deterministic.
        parts = [_message_content_to_text(part) for part in content]
        return "\n".join([p for p in parts if p])
    try:
        return json.dumps(content, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(content)


def _lc_messages_to_insidellms(messages: Sequence[Any]) -> list[ChatMessage]:
    """Convert LangChain BaseMessage objects to insideLLMs ChatMessage dicts.

    We intentionally accept `Any` so this module can remain importable even when
    LangChain is not installed (the integration is optional).
    """
    converted: list[ChatMessage] = []
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type == "system":
            role = "system"
        elif msg_type == "human":
            role = "user"
        elif msg_type == "ai":
            role = "assistant"
        elif msg_type == "tool":
            # insideLLMs does not have a native "tool" role; treat tool outputs as assistant text.
            role = "assistant"
        else:
            # Best-effort default; many frameworks treat unknown messages as user-supplied context.
            role = "user"

        content = _message_content_to_text(getattr(msg, "content", None))
        name = getattr(msg, "name", None)
        converted.append({"role": role, "content": content, "name": name})
    return converted


def _insidellms_messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
    """Fallback prompt rendering when the underlying model lacks chat()."""
    lines: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "user").upper()
        content = msg.get("content") or ""
        if not content:
            continue
        lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _call_with_stop(
    func,
    *args: Any,
    stop: Optional[list[str]] = None,
    **kwargs: Any,
) -> Any:
    if stop is None:
        return func(*args, **kwargs)

    # LangChain uses `stop`; some insideLLMs models use `stop_sequences`.
    for stop_kw in ("stop", "stop_sequences"):
        try:
            return func(*args, **kwargs, **{stop_kw: stop})
        except TypeError:
            continue

    # Last resort: ignore stop.
    return func(*args, **kwargs)


def as_langchain_chat_model(model: ModelProtocol):
    """Wrap an insideLLMs model as a LangChain ChatModel.

    This adapter lets you use insideLLMs models inside LangChain chains and
    LangGraph graphs.

    Requirements:
      - `langchain-core` (and optionally `langchain`, `langgraph`)

    Install:
      - `pip install "insideLLMs[langchain]"`
    """
    try:
        from pydantic import Field

        try:
            # pydantic v2
            from pydantic import ConfigDict  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            ConfigDict = None  # type: ignore[assignment]

        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
    except Exception as e:  # pragma: no cover
        raise LangChainIntegrationError(
            'LangChain integration requires extra deps. Install with: pip install "insideLLMs[langchain]"'
        ) from e

    class _InsideLLMsChatModel(BaseChatModel):
        model: Any = Field(exclude=True)

        if ConfigDict is not None:
            model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore[misc]
        else:  # pragma: no cover

            class Config:
                arbitrary_types_allowed = True

        @property
        def _llm_type(self) -> str:
            return "insidellms-chat"

        def _generate(  # type: ignore[override]
            self,
            messages: list[Any],
            stop: Optional[list[str]] = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            insidellms_messages = _lc_messages_to_insidellms(messages)

            try:
                text = _call_with_stop(self.model.chat, insidellms_messages, stop=stop, **kwargs)
            except NotImplementedError:
                prompt = _insidellms_messages_to_prompt(insidellms_messages)
                text = _call_with_stop(self.model.generate, prompt, stop=stop, **kwargs)

            generation = ChatGeneration(message=AIMessage(content=str(text)))
            return ChatResult(
                generations=[generation],
                llm_output={"model_id": getattr(self.model, "model_id", None)},
            )

        def _stream(  # type: ignore[override]
            self,
            messages: list[Any],
            stop: Optional[list[str]] = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Iterator[Any]:
            # Best-effort streaming: if the underlying insideLLMs model supports stream(prompt),
            # we use the rendered prompt fallback and stream that. Chat streaming is not currently
            # supported by the insideLLMs Model interface.
            try:
                from langchain_core.messages import AIMessageChunk
                from langchain_core.outputs import ChatGenerationChunk
            except Exception:  # pragma: no cover
                # If chunk classes aren't available, fall back to non-streaming.
                result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                yield from result.generations
                return

            insidellms_messages = _lc_messages_to_insidellms(messages)

            try:
                prompt = _insidellms_messages_to_prompt(insidellms_messages)
                for chunk in self.model.stream(prompt, **kwargs):
                    if run_manager is not None:
                        try:
                            run_manager.on_llm_new_token(str(chunk))
                        except Exception:
                            pass
                    yield ChatGenerationChunk(message=AIMessageChunk(content=str(chunk)))
            except Exception:
                result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                yield from result.generations

    return _InsideLLMsChatModel(model=model)


def as_langchain_runnable(model: ModelProtocol):
    """Wrap an insideLLMs model as a LangChain Runnable.

    This is a lightweight alternative to a full ChatModel subclass. It supports:
      - `str` input -> `model.generate(str)`
      - `list[BaseMessage]` input -> `model.chat([...])` (or prompt fallback)
    """
    try:
        from langchain_core.runnables import RunnableLambda
    except Exception as e:  # pragma: no cover
        raise LangChainIntegrationError(
            'LangChain integration requires extra deps. Install with: pip install "insideLLMs[langchain]"'
        ) from e

    def _invoke(inp: Any) -> Any:
        if isinstance(inp, str):
            return model.generate(inp)
        if isinstance(inp, list):
            insidellms_messages = _lc_messages_to_insidellms(inp)
            try:
                return model.chat(insidellms_messages)
            except NotImplementedError:
                return model.generate(_insidellms_messages_to_prompt(insidellms_messages))
        return model.generate(str(inp))

    return RunnableLambda(_invoke)
