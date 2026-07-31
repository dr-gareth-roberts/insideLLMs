from insideLLMs.inference import Candidate, InferenceRequest, StopReason
from insideLLMs.inference.self_consistency import sample_consistent


async def test_self_consistency_stops_when_leader_is_mathematically_unbeatable() -> None:
    answers = ["4", "4", "4", "5", "5"]
    sampled: list[int] = []

    async def sample(request: InferenceRequest, index: int) -> Candidate:
        sampled.append(index)
        return Candidate(id=f"c{index}", output=answers[index])

    result = await sample_consistent(
        InferenceRequest(prompt="2 + 2"),
        sample=sample,
        normalize=lambda candidate: candidate.output.strip(),
        max_samples=5,
    )

    assert result.answer == "4"
    assert result.stop_reason is StopReason.AGREEMENT
    assert result.confidence == 1.0
    assert sampled == [0, 1, 2]
    assert result.spend.calls == 3
