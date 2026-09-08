"""Offline smoke test for matched-compute evaluation.

This proves orchestration, request-bound output-cap evidence, and artifact generation
for this trusted offline model only. Replace
``SmokeModel`` with an ``InferenceClient.from_model_config(...)`` client and a
representative dataset before drawing quality conclusions.
"""

from __future__ import annotations

import asyncio
import json

from insideLLMs.analysis.evaluation import ExactMatchEvaluator
from insideLLMs.analysis.matched_compute import (
    one_shot_baseline,
    run_matched_compute,
    single_result_variant,
)
from insideLLMs.benchmark_datasets import DatasetExample
from insideLLMs.inference import InferenceClient, OutputLimitBinding, Verification, VerifierSpec
from insideLLMs.types import ModelResponse, TokenUsage


class SmokeModel:
    """Deterministic metadata-aware model for an offline lifecycle check."""

    name = "matched-compute-smoke"
    model_id = "offline/matched-compute-smoke-v1"

    def __init__(self) -> None:
        self.calls = 0

    async def agenerate_with_metadata(self, prompt: str, **kwargs: object) -> ModelResponse:
        answer = ("4", "four")[self.calls % 2]
        self.calls += 1
        return ModelResponse(
            content=answer,
            model=self.model_id,
            latency_ms=1.0,
            usage=TokenUsage(prompt_tokens=4, completion_tokens=1, total_tokens=5),
        )


async def main() -> None:
    client = InferenceClient(
        SmokeModel(),
        generation_kwargs={"max_tokens": 16},
        output_limit=OutputLimitBinding("max_tokens", 16),
    )
    numeric_output = VerifierSpec(
        "numeric-output",
        lambda candidate: Verification(
            "numeric-output",
            score=float(candidate.output.strip().isdigit()),
            passed=True,
        ),
    )
    report = await run_matched_compute(
        [
            DatasetExample(
                id="offline-arithmetic",
                input_text="Return only the numeral for two plus two.",
                expected_output="4",
                category="offline-smoke",
            )
        ],
        baseline=one_shot_baseline(client, samples=2, max_output_tokens_per_call=16),
        strategy=single_result_variant(
            "best-of-2-format-verifier",
            lambda request: client.best_of_n(
                request,
                n=2,
                verifiers=(numeric_output,),
            ),
            client=client,
            generated_calls=2,
            max_output_tokens_per_call=16,
        ),
        evaluator=ExactMatchEvaluator(),
        trials=2,
    )
    artifact = report.to_dict()
    assert artifact["matching"]["declared_limits_match"] is True
    assert artifact["matching"]["output_limit_assurance"] == "verified"
    assert artifact["matching"]["max_output_tokens"] is True
    limitations = artifact["limitations"]
    assert isinstance(limitations, dict)
    limitations["evidence_scope"] = "offline-smoke-only"
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
