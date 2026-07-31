"""W7-0008: mock transformers so models/huggingface.py can leave coverage omit."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from insideLLMs.exceptions import ModelGenerationError, ModelInitializationError


@pytest.fixture()
def hf_stub(monkeypatch: pytest.MonkeyPatch):
    transformers = types.ModuleType("transformers")
    state = {"tok": "ok", "mdl": "ok", "pipe": "ok"}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            if state["tok"] != "ok":
                raise RuntimeError(state["tok"])
            return MagicMock(name=f"tok:{name}")

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name):
            if state["mdl"] != "ok":
                raise RuntimeError(state["mdl"])
            return MagicMock(name=f"model:{name}")

    def pipeline(task, model=None, tokenizer=None, device=None):
        if state["pipe"] != "ok":
            raise RuntimeError(state["pipe"])
        gen = MagicMock(return_value=[{"generated_text": "out"}])
        gen.task = task
        gen.device = device
        return gen

    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    transformers.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    for key in list(sys.modules):
        if key == "insideLLMs.models.huggingface" or key.startswith(
            "insideLLMs.models.huggingface."
        ):
            del sys.modules[key]

    import insideLLMs.models.huggingface as hf

    try:
        yield hf, state
    finally:
        for key in list(sys.modules):
            if key == "insideLLMs.models.huggingface" or key.startswith(
                "insideLLMs.models.huggingface."
            ):
                del sys.modules[key]
        # Drop stub transformers if we installed it (monkeypatch restores after).
        # Ensure model module is not left bound to stub classes.


def test_huggingface_full_paths(hf_stub) -> None:
    hf, state = hf_stub

    m = hf.HuggingFaceModel(model_name="gpt2", device=-1)
    assert m.generate("hi") == "out"
    assert m.chat([{"role": "user", "content": "hi"}]) == "out"
    assert list(m.stream("hi")) == ["out"]
    info = m.info()
    assert info.extra["model_name"] == "gpt2"
    assert info.extra["device"] == -1

    m.generator.return_value = []
    assert m.generate("x") == ""
    assert m.chat([{"role": "user", "content": "x"}]) == ""
    assert list(m.stream("x")) == []

    m.generator.side_effect = RuntimeError("boom")
    with pytest.raises(ModelGenerationError):
        m.generate("x")
    with pytest.raises(ModelGenerationError):
        m.chat([{"role": "user", "content": "x"}])
    with pytest.raises(ModelGenerationError):
        list(m.stream("x"))
    with pytest.raises(ModelGenerationError):
        m.chat([])

    state["tok"] = "tok-fail"
    with pytest.raises(ModelInitializationError, match="tokenizer"):
        hf.HuggingFaceModel(model_name="bad")
    state["tok"] = "ok"

    state["mdl"] = "mdl-fail"
    with pytest.raises(ModelInitializationError, match="model"):
        hf.HuggingFaceModel(model_name="bad")
    state["mdl"] = "ok"

    state["pipe"] = "pipe-fail"
    with pytest.raises(ModelInitializationError, match="pipeline"):
        hf.HuggingFaceModel(model_name="bad")
    state["pipe"] = "ok"
