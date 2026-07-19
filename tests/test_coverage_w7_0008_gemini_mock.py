"""W7-0008: mock google-generativeai so models/gemini.py can leave coverage omit."""

from __future__ import annotations

import builtins
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def gemini_mod(monkeypatch: pytest.MonkeyPatch):
    for key in list(sys.modules):
        if key == "insideLLMs.models.gemini" or key.startswith("insideLLMs.models.gemini."):
            del sys.modules[key]
    import insideLLMs.models.gemini as gemini

    try:
        yield gemini
    finally:
        for key in list(sys.modules):
            if key == "insideLLMs.models.gemini" or key.startswith("insideLLMs.models.gemini."):
                del sys.modules[key]


def _install_genai(monkeypatch: pytest.MonkeyPatch):
    google = types.ModuleType("google")
    genai = types.ModuleType("google.generativeai")

    class GenerativeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.generate_content = MagicMock(
                return_value=types.SimpleNamespace(text="hi"),
            )
            chat = MagicMock()
            chat.send_message.return_value = types.SimpleNamespace(text="chat-hi")
            self.start_chat = MagicMock(return_value=chat)
            self.count_tokens = MagicMock(
                return_value=types.SimpleNamespace(total_tokens=7),
            )

        def stream_chunks(self, *args, **kwargs):
            return [
                types.SimpleNamespace(text="a"),
                types.SimpleNamespace(text=""),
                types.SimpleNamespace(text="b"),
            ]

    def configure(**kwargs):
        genai._configured = kwargs

    def list_models():
        return [
            types.SimpleNamespace(
                name="models/gemini-1.5-flash",
                supported_generation_methods=["generateContent"],
            ),
            types.SimpleNamespace(
                name="models/embed",
                supported_generation_methods=["embedContent"],
            ),
        ]

    genai.configure = configure
    genai.GenerativeModel = GenerativeModel
    genai.list_models = list_models
    google.generativeai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.generativeai", genai)
    return genai, GenerativeModel


def test_gemini_full_paths(gemini_mod, monkeypatch: pytest.MonkeyPatch) -> None:
    gemini = gemini_mod
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key"):
        gemini.GeminiModel(api_key=None)

    genai, GenerativeModel = _install_genai(monkeypatch)
    m = gemini.GeminiModel(
        api_key="gk",
        model_name="gemini-1.5-flash",
        safety_settings=[{"category": "x", "threshold": "y"}],
        generation_config={"temperature": 0.1},
    )
    assert m.generate("x", temperature=0.5, max_tokens=10, top_p=0.9, top_k=4) == "hi"
    assert m.generate("y", max_output_tokens=5) == "hi"

    class _BadText:
        @property
        def text(self):
            raise ValueError("blocked")

    m._model.generate_content.return_value = _BadText()
    assert m.generate("z") == ""

    class _BadIndex:
        @property
        def text(self):
            raise IndexError("empty")

    m._model.generate_content.return_value = _BadIndex()
    assert m.generate("z2") == ""

    m._model.generate_content.return_value = types.SimpleNamespace(text="hi")
    assert (
        m.chat(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
            ],
            temperature=0.2,
            max_tokens=9,
        )
        == "chat-hi"
    )
    # trailing system-only
    assert m.chat([{"role": "system", "content": "only"}]) == "chat-hi"
    # second system while current_message already set is skipped
    assert (
        m.chat(
            [
                {"role": "system", "content": "first"},
                {"role": "system", "content": "second"},
                {"role": "user", "content": "u"},
            ]
        )
        == "chat-hi"
    )
    # system after history is skipped
    assert (
        m.chat(
            [
                {"role": "user", "content": "u"},
                {"role": "system", "content": "late"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "u2"},
            ]
        )
        == "chat-hi"
    )
    # empty history edge (no user) — still sends
    assert m.chat([{"role": "assistant", "content": "only-asst"}]) == "chat-hi"
    # unknown role falls through the role chain
    assert (
        m.chat([{"role": "tool", "content": "ignored"}, {"role": "user", "content": "u"}])
        == "chat-hi"
    )

    def stream_gc(*args, **kwargs):
        assert kwargs.get("stream") is True
        return [
            types.SimpleNamespace(text="a"),
            types.SimpleNamespace(text=""),
            types.SimpleNamespace(text="b"),
        ]

    m._model.generate_content = stream_gc
    assert "".join(m.stream("x", temperature=0.3, max_tokens=3)) == "ab"
    assert "".join(m.stream("plain")) == "ab"  # no temp/max_tokens kwargs
    assert m.info().provider == "Google"
    assert m.count_tokens("abc") == 7
    assert m.list_models() == ["models/gemini-1.5-flash"]
    # cached client path
    assert m._get_client() is m._model

    # ImportError on missing SDK
    m3 = gemini.GeminiModel(api_key="gk")
    m3._client = None
    m3._model = None
    for key in ("google.generativeai", "google"):
        sys.modules.pop(key, None)
    real_import = builtins.__import__

    def block(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("google", "google.generativeai") or name.startswith("google."):
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(ImportError, match="google-generativeai"):
        m3._get_client()
