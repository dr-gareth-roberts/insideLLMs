import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.inference_strategies import STRATEGY_MODULES, run_benchmark
from insideLLMs import InferenceRequest, ModelProposer
from insideLLMs.inference import EvolutionConfig, evolve_artifacts
from insideLLMs.inference import ModelProposer as InferenceModelProposer
from insideLLMs.inference.sync import run_sync

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _strategy_module_count() -> int:
    return len(STRATEGY_MODULES)


# Modules that provide shared plumbing rather than an inference strategy.
_INFRASTRUCTURE_MODULES = frozenset({"adapters", "client", "schemas", "sync"})


def test_declared_strategy_modules_match_the_inference_package() -> None:
    """The benchmark's strategy list must not drift from the package."""
    package = _REPOSITORY_ROOT / "insideLLMs" / "inference"
    present = {
        path.stem
        for path in package.glob("*.py")
        if not path.stem.startswith("_") and path.stem not in _INFRASTRUCTURE_MODULES
    }
    assert present, f"no strategy modules found under {package}"
    assert set(STRATEGY_MODULES) == present


def test_public_exports_sync_wrapper_and_offline_benchmark_are_deterministic() -> None:
    assert InferenceRequest(prompt="hello").prompt == "hello"
    assert EvolutionConfig(population_size=2).population_size == 2
    assert callable(evolve_artifacts)
    assert ModelProposer is InferenceModelProposer
    assert run_sync(asyncio.sleep(0, result="ok")) == "ok"

    first = run_benchmark()
    second = run_benchmark()
    assert first == second
    assert first["strategy_count"] == 11
    assert first["offline_evolution"] is True
    assert first["self_consistency"]["calls"] == 3
    assert first["prefix_cache"]["cache_hits"] == 1
    assert first["evolution"]["evaluation_budget"] == 5


async def test_sync_wrapper_rejects_a_running_event_loop() -> None:
    coroutine = asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="await the async API"):
        run_sync(coroutine)


def test_benchmark_runs_directly_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, "benchmarks/inference_strategies.py"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["strategy_count"] == _strategy_module_count()


def test_inference_client_example_runs_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "examples.inference_client"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["answer"] == "Paris"
    assert payload["strategy"] == "one-shot"
    assert payload["calls"] == 1
