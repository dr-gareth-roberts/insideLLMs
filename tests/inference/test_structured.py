from __future__ import annotations

import pytest

from insideLLMs.inference import Candidate, InferenceRequest, Verification
from insideLLMs.inference.structured import generate_validated


async def test_structured_output_repairs_once_after_objective_validation_failure() -> None:
    calls: list[str] = []

    async def generate(request: InferenceRequest) -> str:
        calls.append("generate")
        return '{"answer": "four"}'

    def validate(candidate: Candidate) -> Verification:
        passed = candidate.output == '{"answer": 4}'
        return Verification(
            verifier_id="json-schema",
            score=1.0 if passed else 0.0,
            passed=passed,
            evidence="integer required",
        )

    async def repair(candidate: Candidate, failure: Verification) -> str:
        calls.append(f"repair:{failure.evidence}")
        return '{"answer": 4}'

    result = await generate_validated(
        InferenceRequest(prompt="What is 2 + 2?"),
        generate=generate,
        validate=validate,
        repair=repair,
        max_repairs=1,
    )

    assert result.answer == '{"answer": 4}'
    assert result.verifications[-1].passed is True
    assert result.provenance["original_output"] == '{"answer": "four"}'
    assert result.spend.calls == 2
    assert calls == ["generate", "repair:integer required"]


async def test_structured_output_rejects_empty_validators_before_generation() -> None:
    generated = False

    def generate(request: InferenceRequest) -> str:
        nonlocal generated
        generated = True
        return "unused"

    with pytest.raises(ValueError, match="validator"):
        await generate_validated(
            InferenceRequest(prompt="invalid configuration"),
            generate=generate,
            validate=(),
        )

    assert generated is False
