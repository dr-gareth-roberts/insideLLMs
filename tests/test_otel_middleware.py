import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from insideLLMs.runtime.observability import OTelMiddleware, TracingConfig
from insideLLMs.runtime.pipeline import AsyncModelPipeline, ModelPipeline


# A simple dummy model to test middleware logic without making actual API calls.
class DummyModel:
    def __init__(self, name="dummy-model"):
        self.name = name

    def generate(self, prompt: str, **kwargs) -> str:
        if "error" in prompt:
            raise ValueError("Simulated model error")
        return f"Response to: {prompt}"

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        if any("error" in msg.get("content", "") for msg in messages):
            raise ValueError("Simulated chat error")
        return "Chat response"

    def stream(self, prompt: str, **kwargs):
        if "setup-error" in prompt:
            raise ValueError("Simulated stream setup error")
        yield "Chunk 1 "
        yield "Chunk 2 "
        if "mid-error" in prompt:
            raise ValueError("Simulated stream mid-generation error")
        yield "Chunk 3"

    async def agenerate(self, prompt: str, **kwargs) -> str:
        if "error" in prompt:
            raise ValueError("Simulated async model error")
        return f"Async response to: {prompt}"

    async def achat(self, messages: list[dict[str, str]], **kwargs) -> str:
        if any("error" in msg.get("content", "") for msg in messages):
            raise ValueError("Simulated async chat error")
        return "Async chat response"

    async def astream(self, prompt: str, **kwargs):
        if "error" in prompt:
            raise ValueError("Simulated async stream error")
        for chunk in ["Async ", "chunk ", "stream"]:
            yield chunk


@pytest.fixture
def otel_setup():
    """Sets up an in-memory tracer for each test.

    Note (TE6): We use ``trace._TRACER_PROVIDER`` because OpenTelemetry's
    public ``set_tracer_provider()`` has a set-once guard that prevents
    resetting between tests. This is a known limitation — there is no
    public API for resetting the global tracer provider. The OTel test
    utilities (``opentelemetry.test``) also use private APIs internally.

    If a future OTel release provides a public reset mechanism, this
    fixture should be updated to use it. Track:
    https://github.com/open-telemetry/opentelemetry-python/issues/3064
    """
    original_provider = trace._TRACER_PROVIDER

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace._TRACER_PROVIDER = provider
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]

    yield exporter

    # Restore original provider
    provider.shutdown()
    trace._TRACER_PROVIDER = original_provider


def test_otel_middleware_generate(otel_setup):
    config = TracingConfig(log_prompts=True, log_responses=True)
    middleware = OTelMiddleware(config=config)

    pipeline = ModelPipeline(DummyModel(), middlewares=[middleware])

    response = pipeline.generate("Hello world")
    assert response == "Response to: Hello world"

    spans = otel_setup.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "llm.pipeline.generate"
    assert span.attributes["llm.model"] == "dummy-model"
    assert span.attributes["llm.operation"] == "generate"
    assert span.attributes["llm.prompt"] == "Hello world"
    assert span.attributes["llm.response"] == "Response to: Hello world"
    assert span.attributes["llm.success"] is True


def test_otel_middleware_generate_error(otel_setup):
    middleware = OTelMiddleware()
    pipeline = ModelPipeline(DummyModel(), middlewares=[middleware])

    with pytest.raises(ValueError):
        pipeline.generate("Trigger an error")

    spans = otel_setup.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes["llm.success"] is False
    assert "Simulated model error" in span.attributes["llm.error"]


def test_otel_middleware_chat(otel_setup):
    config = TracingConfig(log_prompts=True, log_responses=True)
    middleware = OTelMiddleware(config=config)
    pipeline = ModelPipeline(DummyModel(), middlewares=[middleware])

    messages = [{"role": "user", "content": "How are you?"}]
    response = pipeline.chat(messages)
    assert response == "Chat response"

    spans = otel_setup.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "llm.pipeline.chat"
    assert span.attributes["llm.operation"] == "chat"
    assert span.attributes["llm.success"] is True
    assert span.attributes["llm.message_count"] == 1


def test_otel_middleware_stream(otel_setup):
    config = TracingConfig(log_prompts=True, log_responses=True)
    middleware = OTelMiddleware(config=config)
    pipeline = ModelPipeline(DummyModel(), middlewares=[middleware])

    chunks = list(pipeline.stream("Tell a story"))
    assert "".join(chunks) == "Chunk 1 Chunk 2 Chunk 3"

    spans = otel_setup.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "llm.pipeline.stream"
    assert span.attributes["llm.operation"] == "stream"
    assert span.attributes["llm.success"] is True
    assert span.attributes["llm.response"] == "Chunk 1 Chunk 2 Chunk 3"


def test_otel_middleware_stream_error(otel_setup):
    config = TracingConfig()
    middleware = OTelMiddleware(config=config)
    pipeline = ModelPipeline(DummyModel(), middlewares=[middleware])

    with pytest.raises(ValueError):
        list(pipeline.stream("mid-error"))

    spans = otel_setup.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "llm.pipeline.stream"
    assert span.attributes["llm.success"] is False
    assert "Simulated stream mid-generation error" in span.attributes["llm.error"]


@pytest.mark.asyncio
async def test_otel_middleware_agenerate(otel_setup):
    config = TracingConfig(log_prompts=True, log_responses=True)
    middleware = OTelMiddleware(config=config)
    pipeline = AsyncModelPipeline(DummyModel(), middlewares=[middleware])

    response = await pipeline.agenerate("Hello async")
    assert response == "Async response to: Hello async"

    spans = otel_setup.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "llm.pipeline.agenerate"
    assert span.attributes["llm.operation"] == "agenerate"
    assert span.attributes["llm.success"] is True
    assert span.attributes["llm.prompt"] == "Hello async"


@pytest.mark.asyncio
async def test_otel_middleware_achat(otel_setup):
    config = TracingConfig(log_prompts=True, log_responses=True)
    middleware = OTelMiddleware(config=config)
    pipeline = AsyncModelPipeline(DummyModel(), middlewares=[middleware])

    messages = [{"role": "user", "content": "How are you async?"}]
    response = await pipeline.achat(messages)
    assert response == "Async chat response"

    spans = otel_setup.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "llm.pipeline.achat"
    assert span.attributes["llm.operation"] == "achat"
    assert span.attributes["llm.success"] is True


@pytest.mark.asyncio
async def test_otel_middleware_astream(otel_setup):
    config = TracingConfig(log_prompts=True, log_responses=True)
    middleware = OTelMiddleware(config=config)
    pipeline = AsyncModelPipeline(DummyModel(), middlewares=[middleware])

    chunks = []
    async for chunk in pipeline.astream("Hello async stream"):
        chunks.append(chunk)
    output = "".join(chunks)
    assert "Async" in output
    assert "stream" in output

    spans = otel_setup.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "llm.pipeline.astream"
    assert span.attributes["llm.operation"] == "astream"
    assert span.attributes["llm.success"] is True
