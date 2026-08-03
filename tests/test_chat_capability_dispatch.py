"""Chat dispatch must gate on capability predicates, not attribute presence.

``Model.chat`` and ``AsyncModel.achat`` are *concrete* stubs that raise
``NotImplementedError`` — only ``generate`` (and ``agenerate`` on ``AsyncModel``)
is abstract. So ``hasattr(model, "chat")`` is True for every model and reports a
capability the model does not have, which silently disables whatever fallback
the caller wrote underneath.

These callers each documented support for generate-only models and each had that
support broken by the presence check. Grouped here because they are one defect,
found together, rather than three unrelated bugs in three modules.
"""

import pytest

from insideLLMs.models.base import Model, ModelInfo, can_chat


class GenerateOnly(Model):
    """A model that implements only ``generate`` — the documented minimum."""

    def __init__(self, output: str = "generated-answer") -> None:
        super().__init__(name="gen-only")
        self._output = output

    def generate(self, prompt: str, **kwargs: object) -> str:
        return self._output

    def info(self) -> ModelInfo:
        return ModelInfo(name=self.name, provider="test")


def test_the_stub_is_present_but_is_not_an_implementation() -> None:
    """The premise the three fixes below rest on."""
    model = GenerateOnly()
    assert hasattr(model, "chat") is True
    assert can_chat(model) is False
    with pytest.raises(NotImplementedError):
        model.chat([{"role": "user", "content": "hi"}])


def test_structured_output_falls_back_to_generate() -> None:
    """``StructuredOutputGenerator`` documents "chat() or generate()" support.

    The presence check made the ``generate()`` branch unreachable, so a
    generate-only model raised ``NotImplementedError`` — once per retry attempt,
    since the dispatch sits inside the retry loop.
    """
    pydantic = pytest.importorskip("pydantic")

    from insideLLMs.structured import StructuredOutputGenerator

    class Out(pydantic.BaseModel):
        value: int

    class RealChat(GenerateOnly):
        def chat(self, messages: list, **kwargs: object) -> str:
            return '{"value": 42}'

    assert StructuredOutputGenerator(GenerateOnly('{"value": 1}'), Out).generate("go").value == 1
    # And a model that genuinely implements chat still reaches it.
    assert StructuredOutputGenerator(RealChat('{"value": 1}'), Out).generate("go").value == 42


def test_rag_chain_falls_back_to_generate() -> None:
    """``RAGChain`` documents accepting a model with only ``generate``."""
    from insideLLMs.contrib.retrieval import RAGChain

    class RealChat(GenerateOnly):
        def chat(self, messages: list, **kwargs: object) -> str:
            return "chat-answer"

    for model, expected in ((GenerateOnly(), "generated-answer"), (RealChat(), "chat-answer")):
        chain = RAGChain(model=model)
        chain.add_documents(["The sky is blue."])
        assert chain.query_with_chat("what colour is the sky?").answer == expected


async def test_receipt_middleware_falls_back_to_sync_chat() -> None:
    """``ReceiptMiddleware.aprocess_chat`` documents a run-in-executor fallback.

    ``hasattr(model, "achat")`` is True for every ``AsyncModel``, so the executor
    branch never ran for a model implementing only synchronous ``chat``. The
    subject has to be an ``AsyncModel``: plain ``Model`` has no ``achat``
    attribute at all, so it reaches the fallback either way and would pass this
    test without the fix.
    """
    from insideLLMs.models.base import AsyncModel, can_chat_async
    from insideLLMs.runtime.receipt import ReceiptMiddleware

    class SyncChatOnly(AsyncModel):
        def generate(self, prompt: str, **kwargs: object) -> str:
            return "generated-answer"

        async def agenerate(self, prompt: str, **kwargs: object) -> str:
            return "async-generated-answer"

        def chat(self, messages: list, **kwargs: object) -> str:
            return "sync-chat-answer"

        def info(self) -> ModelInfo:
            return ModelInfo(name=self.name, provider="test")

    model = SyncChatOnly(name="sync-chat-only")
    # The stub is present, which is exactly what made hasattr wrong here.
    assert hasattr(model, "achat") is True
    assert can_chat_async(model) is False

    middleware = ReceiptMiddleware()
    middleware.model = model
    response = await middleware.aprocess_chat([{"role": "user", "content": "hi"}])
    assert response == "sync-chat-answer"
