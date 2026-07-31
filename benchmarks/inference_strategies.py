"""Deterministic offline example for inference strategy accounting.

The latency values are simulated workload units so repeated runs remain diffable;
use a real harness run for environment-specific p50/p95 measurements.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


async def _benchmark() -> dict[str, object]:
    from insideLLMs.inference import Candidate, InferenceRequest, PromptParts
    from insideLLMs.inference.evolution import EvolutionConfig, evolve_artifacts
    from insideLLMs.inference.prefix_cache import compose_cached_prompt
    from insideLLMs.inference.self_consistency import sample_consistent

    parts = PromptParts(stable=("system",), dynamic=("question",))
    cached = compose_cached_prompt(parts, tenant_id="benchmark")
    repeated = compose_cached_prompt(parts, tenant_id="benchmark")

    async def sample(request: InferenceRequest, index: int) -> Candidate:
        return Candidate(f"sample-{index}", "4")

    consistency = await sample_consistent(
        InferenceRequest("2 + 2"),
        sample=sample,
        normalize=lambda candidate: candidate.output,
        max_samples=5,
    )
    evolution = await evolve_artifacts(
        ["prompt"],
        evaluate=lambda text: float(text.count("!")),
        mutate=lambda parent, rng: parent.artifact_text + "!",
        select_parent=lambda population, rng: population[0],
        config=EvolutionConfig(
            population_size=2,
            elite_count=1,
            max_generations=10,
            max_evaluations=5,
            seed=7,
        ),
    )
    return {
        "strategy_count": 11,
        "offline_evolution": True,
        "prefix_cache": {
            "cache_hits": int(cached.cache_key == repeated.cache_key),
            "cached_tokens": 1,
        },
        "self_consistency": {
            "answer": consistency.answer,
            "calls": consistency.spend.calls,
            "early_stop": consistency.stop_reason.value,
            "simulated_latency_ms": {"p50": 3, "p95": 3, "ttft": 1},
            "cost": 0.0,
        },
        "evolution": {
            "best": evolution.best.artifact_text,
            "evaluations": evolution.evaluations,
            "evaluation_budget": 5,
            "stop_reason": evolution.stop_reason,
        },
        "comparison": {
            "one_shot_calls": 1,
            "matched_compute_calls": consistency.spend.calls,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }


def run_benchmark() -> dict[str, object]:
    from insideLLMs.inference.sync import run_sync

    return run_sync(_benchmark())


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
