"""W7-0008: mock Anthropic SDK so models/anthropic.py can leave coverage omit."""

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
def anthropic_stub(monkeypatch: pytest.MonkeyPatch):
    anthropic_mod = types.ModuleType("anthropic")

    class APIError(Exception):
        def __init__(self, message="api", status_code=500):
            super().__init__(message)
            self.status_code = status_code

    class APITimeoutError(Exception):
        pass

    class RateLimitErrorSDK(Exception):
        def __init__(self, message="rl"):
            super().__init__(message)
            self.retry_after = 2.0

    class _Block:
        def __init__(self, text=None, as_str=None):
            if text is not None:
                self.text = text
            self._as_str = as_str or "fallback"

        def __str__(self):
            return self._as_str

    class _Resp:
        def __init__(self, text="hi", empty=False, bare_block=False):
            if empty:
                self.content = []
            elif bare_block:
                self.content = [_Block(as_str="bare")]
            else:
                self.content = [_Block(text=text)]

    class _StreamCtx:
        def __init__(self, chunks):
            self.text_stream = iter(chunks)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Anthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = types.SimpleNamespace(
                create=MagicMock(return_value=_Resp("hi")),
                stream=MagicMock(return_value=_StreamCtx(["a", "b"])),
            )

    anthropic_mod.APIError = APIError
    anthropic_mod.APITimeoutError = APITimeoutError
    anthropic_mod.RateLimitError = RateLimitErrorSDK
    anthropic_mod.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_mod)

    for key in list(sys.modules):
        if key == "insideLLMs.models.anthropic" or key.startswith("insideLLMs.models.anthropic."):
            del sys.modules[key]

    import insideLLMs.models.anthropic as anth

    try:
        yield anth, Anthropic, APIError, APITimeoutError, RateLimitErrorSDK, _Resp, _StreamCtx
    finally:
        for key in list(sys.modules):
            if key == "insideLLMs.models.anthropic" or key.startswith(
                "insideLLMs.models.anthropic."
            ):
                del sys.modules[key]


def test_anthropic_happy_and_errors(anthropic_stub, monkeypatch: pytest.MonkeyPatch) -> None:
    anth, Anthropic, APIError, APITimeoutError, RateLimitErrorSDK, _Resp, _StreamCtx = (
        anthropic_stub
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ModelInitializationError):
        anth.AnthropicModel(api_key=None)

    def boom(**kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(anth.anthropic, "Anthropic", boom)
    with pytest.raises(ModelInitializationError, match="Failed to initialize"):
        anth.AnthropicModel(api_key="sk")
    monkeypatch.setattr(anth.anthropic, "Anthropic", Anthropic)

    m = anth.AnthropicModel(model_name="claude-3", api_key="sk")
    assert m.generate("x") == "hi"
    assert m.chat([{"role": "user", "content": "x"}]) == "hi"
    assert "".join(m.stream("x")) == "ab"
    assert m.info().extra["model_name"] == "claude-3"

    # system message + explicit system= + empty chat + bare block str()
    assert (
        m.chat(
            [
                {"role": "system", "content": "be nice"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "hi"},
            ],
            system="also",
        )
        == "hi"
    )
    m.client.messages.create.return_value = _Resp(empty=True)
    assert m.generate("x") == ""
    assert m.chat([{"role": "user", "content": "x"}]) == ""
    m.client.messages.create.return_value = _Resp(bare_block=True)
    assert m.generate("x") == "bare"

    m.client.messages.create = MagicMock(side_effect=RateLimitErrorSDK())
    with pytest.raises(RateLimitError):
        m.generate("x")
    with pytest.raises(RateLimitError):
        m.chat([{"role": "user", "content": "x"}])
    m.client.messages.stream = MagicMock(side_effect=RateLimitErrorSDK())
    with pytest.raises(RateLimitError):
        list(m.stream("x"))

    m.client.messages.create = MagicMock(side_effect=APITimeoutError())
    m.client.messages.stream = MagicMock(side_effect=APITimeoutError())
    with pytest.raises(InsideLLMsTimeoutError):
        m.generate("x")
    with pytest.raises(InsideLLMsTimeoutError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(InsideLLMsTimeoutError):
        list(m.stream("x"))

    m.client.messages.create = MagicMock(side_effect=APIError("bad", 400))
    m.client.messages.stream = MagicMock(side_effect=APIError("bad", 400))
    with pytest.raises(InsideLLMsAPIError):
        m.generate("x")
    with pytest.raises(InsideLLMsAPIError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(InsideLLMsAPIError):
        list(m.stream("x"))

    m.client.messages.create = MagicMock(side_effect=ValueError("x"))
    m.client.messages.stream = MagicMock(side_effect=ValueError("x"))
    with pytest.raises(ModelGenerationError):
        m.generate("x")
    with pytest.raises(ModelGenerationError):
        m.chat([])
    with pytest.raises(ModelGenerationError):
        list(m.stream("x"))
