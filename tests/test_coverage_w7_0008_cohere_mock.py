"""W7-0008: mock Cohere SDK so models/cohere.py can leave coverage omit."""

from __future__ import annotations

import builtins
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def cohere_stub(monkeypatch: pytest.MonkeyPatch):
    # CohereModel imports cohere lazily inside _get_client
    for key in list(sys.modules):
        if key == "insideLLMs.models.cohere" or key.startswith("insideLLMs.models.cohere."):
            del sys.modules[key]

    import insideLLMs.models.cohere as cohere_model

    try:
        yield cohere_model
    finally:
        for key in list(sys.modules):
            if key == "insideLLMs.models.cohere" or key.startswith("insideLLMs.models.cohere."):
                del sys.modules[key]


def _install_cohere(monkeypatch: pytest.MonkeyPatch, client: MagicMock | None = None):
    mod = types.ModuleType("cohere")
    if client is None:
        client = MagicMock()
        client.chat.return_value = types.SimpleNamespace(text="hi")
        client.chat_stream.return_value = [
            types.SimpleNamespace(event_type="stream-start", text=""),
            types.SimpleNamespace(event_type="text-generation", text="a"),
            types.SimpleNamespace(event_type="text-generation", text="b"),
        ]
        client.embed.return_value = types.SimpleNamespace(embeddings=[[0.1, 0.2]])
        client.rerank.return_value = types.SimpleNamespace(
            results=[types.SimpleNamespace(index=1, relevance_score=0.9)]
        )
    mod.Client = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "cohere", mod)
    return client, mod


def test_cohere_full_paths(cohere_stub, monkeypatch: pytest.MonkeyPatch) -> None:
    cohere_model = cohere_stub
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key"):
        cohere_model.CohereModel(api_key=None)

    client, _ = _install_cohere(monkeypatch)
    m = cohere_model.CohereModel(
        api_key="ck",
        model_name="command-r",
        default_preamble="be brief",
    )
    assert m.generate("x", temperature=0.1, max_tokens=10, top_p=0.9, top_k=5) == "hi"
    assert m._client is client
    # second call hits cached client; p/k aliases; preamble override
    assert m.generate("y", preamble="override", p=0.5, k=3) == "hi"

    m_plain = cohere_model.CohereModel(api_key="ck")
    # no default preamble; only top_p/top_k via alternate names already covered — bare generate
    assert m_plain.generate("bare") == "hi"

    assert (
        m.chat(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
            ],
            temperature=0.2,
            max_tokens=8,
        )
        == "hi"
    )
    # recover last user after completed turn cleared current_message (708-711)
    assert (
        m.chat([{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]) == "hi"
    )
    assert m.chat([{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}]) == "hi"
    # chat without preamble
    assert m_plain.chat([{"role": "user", "content": "q"}]) == "hi"
    # no messages -> empty current_message still calls chat
    assert m.chat([]) == "hi"
    # recovery loop finds no user role
    assert (
        m.chat([{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]) == "hi"
    )

    assert "".join(m.stream("x", temperature=0.3, max_tokens=4)) == "ab"
    # stream without default preamble
    assert "".join(m_plain.stream("z")) == "ab"

    assert m.info().provider == "Cohere"
    assert m.embed(["a"], embedding_types=["float"]) == [[0.1, 0.2]]
    assert m.embed(["b"]) == [[0.1, 0.2]]
    ranked = m.rerank("q", ["d0", "d1"], top_n=1)
    assert ranked == [{"index": 1, "relevance_score": 0.9, "document": "d1"}]
    client.rerank.return_value = types.SimpleNamespace(
        results=[types.SimpleNamespace(index=0, relevance_score=0.5)]
    )
    assert m.rerank("q", ["d0"]) == [{"index": 0, "relevance_score": 0.5, "document": "d0"}]

    # ImportError path
    m2 = cohere_model.CohereModel(api_key="ck")
    sys.modules.pop("cohere", None)
    real_import = builtins.__import__

    def block(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cohere":
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(ImportError, match="cohere package"):
        m2._get_client()
