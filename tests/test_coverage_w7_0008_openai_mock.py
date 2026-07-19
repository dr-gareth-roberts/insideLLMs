"""W7-0008: mock OpenAI SDK so models/openai.py can leave coverage omit."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from insideLLMs.exceptions import (
    APIError as InsideLLMsAPIError,
)
from insideLLMs.exceptions import (
    ModelGenerationError,
    ModelInitializationError,
    RateLimitError,
)
from insideLLMs.exceptions import (
    ModelTimeoutError as InsideLLMsTimeoutError,
)


@pytest.fixture()
def openai_stub(monkeypatch: pytest.MonkeyPatch):
    """Install a fake openai package and force-reload insideLLMs.models.openai."""
    openai_mod = types.ModuleType("openai")

    class APIError(Exception):
        def __init__(self, message="api", status_code=500):
            super().__init__(message)
            self.status_code = status_code

    class APITimeoutError(Exception):
        pass

    class RateLimitErrorSDK(Exception):
        def __init__(self, message="rl"):
            super().__init__(message)
            self.retry_after = 1.5

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)
            self.delta = types.SimpleNamespace(content=content)

    class _Resp:
        def __init__(self, content="hi", empty=False):
            self.choices = [] if empty else [_Choice(content)]

    class OpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=MagicMock(return_value=_Resp("hi")))
            )

    openai_mod.APIError = APIError
    openai_mod.APITimeoutError = APITimeoutError
    openai_mod.RateLimitError = RateLimitErrorSDK
    openai_mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_mod)

    for key in list(sys.modules):
        if key == "insideLLMs.models.openai" or key.startswith("insideLLMs.models.openai."):
            del sys.modules[key]

    import insideLLMs.models.openai as openai_model

    try:
        yield openai_model, OpenAI, APIError, APITimeoutError, RateLimitErrorSDK, _Resp
    finally:
        # Drop poisoned module so later suites re-import against real/absent SDK.
        for key in list(sys.modules):
            if key == "insideLLMs.models.openai" or key.startswith("insideLLMs.models.openai."):
                del sys.modules[key]


def test_openai_init_generate_chat_stream_info(
    openai_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    openai_model, OpenAI, APIError, APITimeoutError, RateLimitErrorSDK, _Resp = openai_stub
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ModelInitializationError):
        openai_model.OpenAIModel(api_key=None)

    # client init failure
    def boom(**kwargs):
        raise RuntimeError("no client")

    monkeypatch.setattr(openai_model, "OpenAI", boom)
    with pytest.raises(ModelInitializationError, match="Failed to initialize"):
        openai_model.OpenAIModel(api_key="sk-x")
    monkeypatch.setattr(openai_model, "OpenAI", OpenAI)

    m = openai_model.OpenAIModel(
        model_name="gpt-4",
        api_key="sk-test",
        base_url="https://example.com",
        organization="org",
        project="proj",
        default_headers={"Authorization": "secret", "X-Custom": "1"},
    )
    assert m.generate("hello") == "hi"
    assert m.chat([{"role": "user", "content": "x"}]) == "hi"

    # empty choices
    m._client.chat.completions.create.return_value = _Resp(empty=True)
    assert m.generate("x") == ""
    assert m.chat([{"role": "user", "content": "x"}]) == ""

    # stream
    def stream_create(**kwargs):
        assert kwargs.get("stream") is True
        return iter(
            [
                types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="a"))]
                ),
                types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="b"))]
                ),
                types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=None))]
                ),
            ]
        )

    m._client.chat.completions.create = stream_create
    assert "".join(m.stream("x")) == "ab"

    info = m.info()
    assert info.extra["model_name"] == "gpt-4"
    assert info.extra["default_headers"]["Authorization"] == "***"
    assert info.extra["default_headers"]["X-Custom"] == "1"
    assert openai_model.OpenAIModel._redact_headers(None) is None


def test_openai_error_mapping(openai_stub) -> None:
    openai_model, OpenAI, APIError, APITimeoutError, RateLimitErrorSDK, _Resp = openai_stub
    m = openai_model.OpenAIModel(model_name="gpt-4", api_key="sk")

    m._client.chat.completions.create = MagicMock(side_effect=RateLimitErrorSDK())
    with pytest.raises(RateLimitError):
        m.generate("x")
    with pytest.raises(RateLimitError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(RateLimitError):
        list(m.stream("x"))

    m._client.chat.completions.create = MagicMock(side_effect=APITimeoutError())
    with pytest.raises(InsideLLMsTimeoutError):
        m.generate("x")
    with pytest.raises(InsideLLMsTimeoutError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(InsideLLMsTimeoutError):
        list(m.stream("x"))

    m._client.chat.completions.create = MagicMock(side_effect=APIError("bad", status_code=400))
    with pytest.raises(InsideLLMsAPIError):
        m.generate("x")
    with pytest.raises(InsideLLMsAPIError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(InsideLLMsAPIError):
        list(m.stream("x"))

    m._client.chat.completions.create = MagicMock(side_effect=ValueError("weird"))
    with pytest.raises(ModelGenerationError):
        m.generate("x")
    with pytest.raises(ModelGenerationError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(ModelGenerationError):
        list(m.stream("x"))
