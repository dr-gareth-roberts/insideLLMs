import pytest

from insideLLMs.inference import Budget, Candidate, InferenceRequest, StopReason
from insideLLMs.inference.escalation import EscalationStep, escalate_adaptively


async def test_adaptive_escalation_runs_next_step_only_below_confidence_threshold() -> None:
    calls: list[str] = []

    def cheap(request: InferenceRequest, previous: Candidate | None) -> Candidate:
        calls.append("cheap")
        return Candidate("cheap", "uncertain", metadata={"confidence": 0.4})

    async def add_verifier(request: InferenceRequest, previous: Candidate | None) -> Candidate:
        calls.append("verify")
        return Candidate("verified", "answer", metadata={"confidence": 0.92})

    result = await escalate_adaptively(
        InferenceRequest(prompt="question"),
        steps=(
            EscalationStep("cheap-model", cheap, minimum_confidence=0.8, cost=0.1),
            EscalationStep("add-verifier", add_verifier, minimum_confidence=0.9, cost=0.2),
        ),
        confidence=lambda candidate: float(candidate.metadata["confidence"]),
        budget=Budget(max_calls=2),
    )

    assert result.answer == "answer"
    assert result.stop_reason is StopReason.VERIFIED
    assert calls == ["cheap", "verify"]
    assert result.spend.calls == 2
    assert result.spend.cost == 0.3
    assert result.provenance["actions"] == ("cheap-model", "add-verifier")


async def test_adaptive_escalation_rejects_zero_call_budget_before_callback() -> None:
    called = False

    def run(request: InferenceRequest, previous: Candidate | None) -> Candidate:
        nonlocal called
        called = True
        return Candidate("never", "never")

    with pytest.raises(ValueError, match="at least one call"):
        await escalate_adaptively(
            InferenceRequest(prompt="question"),
            steps=(EscalationStep("step", run, minimum_confidence=1.0),),
            confidence=lambda candidate: 0.0,
            budget=Budget(max_calls=0),
        )

    assert called is False


async def test_adaptive_escalation_rejects_zero_time_before_callback() -> None:
    called = False

    async def run(request: InferenceRequest, previous: Candidate | None) -> Candidate:
        nonlocal called
        called = True
        return Candidate("never", "never")

    with pytest.raises(ValueError, match="positive time"):
        await escalate_adaptively(
            InferenceRequest(prompt="question"),
            steps=(EscalationStep("step", run, minimum_confidence=1.0),),
            confidence=lambda candidate: 0.0,
            budget=Budget(max_seconds=0),
        )

    assert called is False
