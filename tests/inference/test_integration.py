import asyncio
import json
import subprocess
import sys

import pytest

from benchmarks.inference_strategies import run_benchmark
from insideLLMs import InferenceRequest, ModelProposer
from insideLLMs.inference import EvolutionConfig, evolve_artifacts
from insideLLMs.inference import ModelProposer as InferenceModelProposer
from insideLLMs.inference.sync import run_sync


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
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["strategy_count"] == 11


def test_inference_client_example_runs_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "examples.inference_client"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["answer"] == "Paris"
    assert payload["strategy"] == "one-shot"
    assert payload["calls"] == 1
