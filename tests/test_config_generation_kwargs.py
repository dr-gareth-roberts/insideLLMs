from __future__ import annotations

from typing import Any

import pytest

from insideLLMs.models import DummyModel
from insideLLMs.probes.base import Probe
from insideLLMs.probes.bias import BiasProbe
from insideLLMs.probes.factuality import FactualityProbe
from insideLLMs.registry import ensure_builtins_registered, probe_registry
from insideLLMs.runtime.runner import run_experiment_from_config, run_harness_from_config
from insideLLMs.types import ProbeCategory


class _KwargsCaptureProbe(Probe[dict[str, Any]]):
    def __init__(self, name: str = "_KwargsCaptureProbe"):
        super().__init__(name=name, category=ProbeCategory.CUSTOM)

    def run(self, model: Any, data: Any, **kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)


@pytest.fixture()
def _registered_kwargs_probe() -> str:
    ensure_builtins_registered()
    name = "_test_kwargs_capture"
    probe_registry.register(name, _KwargsCaptureProbe)
    try:
        yield name
    finally:
        probe_registry.unregister(name)


def test_run_config_generation_passes_probe_kwargs(tmp_path, _registered_kwargs_probe: str) -> None:
    (tmp_path / "dataset.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        f"""
model:
  type: dummy
  args: {{}}
probe:
  type: {_registered_kwargs_probe}
  args: {{}}
dataset:
  format: jsonl
  path: dataset.jsonl
generation:
  temperature: 0.0
  max_tokens: 123
""".lstrip(),
        encoding="utf-8",
    )

    results = run_experiment_from_config(tmp_path / "config.yaml", emit_run_artifacts=False)
    assert results[0]["output"] == {"max_tokens": 123, "temperature": 0.0}


def test_run_config_probe_generation_overrides_global(
    tmp_path, _registered_kwargs_probe: str
) -> None:
    (tmp_path / "dataset.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        f"""
model:
  type: dummy
  args: {{}}
probe:
  type: {_registered_kwargs_probe}
  args: {{}}
  generation:
    max_tokens: 999
dataset:
  format: jsonl
  path: dataset.jsonl
generation:
  temperature: 0.0
  max_tokens: 123
""".lstrip(),
        encoding="utf-8",
    )

    results = run_experiment_from_config(tmp_path / "config.yaml", emit_run_artifacts=False)
    assert results[0]["output"] == {"max_tokens": 999, "temperature": 0.0}


def test_harness_config_generation_passes_probe_kwargs(
    tmp_path, _registered_kwargs_probe: str
) -> None:
    (tmp_path / "dataset.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        f"""
models:
  - type: dummy
    args: {{}}
probes:
  - type: {_registered_kwargs_probe}
    args: {{}}
    generation:
      max_tokens: 999
dataset:
  format: jsonl
  path: dataset.jsonl
generation:
  temperature: 0.0
  max_tokens: 123
max_examples: 1
""".lstrip(),
        encoding="utf-8",
    )

    result = run_harness_from_config(tmp_path / "config.yaml")
    assert result["records"][0]["output"] == {"max_tokens": 999, "temperature": 0.0}


def test_bias_probe_accepts_dict_prompt_pair() -> None:
    model = DummyModel()
    probe = BiasProbe(bias_dimension="gender")

    results = probe.run(model, {"prompt_a": "Hello A", "prompt_b": "Hello B"})
    assert len(results) == 1
    assert results[0].prompt_a == "Hello A"
    assert results[0].prompt_b == "Hello B"


def test_factuality_probe_accepts_single_question_dict() -> None:
    model = DummyModel()
    probe = FactualityProbe()

    results = probe.run(
        model,
        {"question": "What is the capital of Japan?", "reference_answer": "Tokyo"},
    )
    assert len(results) == 1
    assert results[0]["question"] == "What is the capital of Japan?"
    assert results[0]["reference_answer"] == "Tokyo"
