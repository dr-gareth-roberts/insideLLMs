from insideLLMs.inference import Candidate, InferenceRequest, Verification
from insideLLMs.inference.repair import repair_with_evidence


async def test_grounded_repair_retains_original_and_accepts_only_improvement() -> None:
    revisions = ["better", "worse"]

    def verify(candidate: Candidate) -> Verification:
        scores = {"wrong": 0.2, "better": 0.9, "worse": 0.1}
        score = scores[candidate.output]
        return Verification("tests", score, score >= 0.8, {"observed": candidate.output})

    async def revise(candidate: Candidate, evidence: object, round_index: int) -> str:
        return revisions[round_index]

    result = await repair_with_evidence(
        InferenceRequest(prompt="task"),
        Candidate("original", "wrong"),
        verify=verify,
        revise=revise,
        max_rounds=2,
    )

    assert result.answer == "better"
    assert result.provenance["original_output"] == "wrong"
    assert result.provenance["accepted_revision_ids"] == ("repair-1",)
    assert result.provenance["wrong_to_right"] == 1
    # right_to_wrong is structurally impossible under monotonic acceptance and
    # is no longer reported as if it were a measured metric.
    assert "right_to_wrong" not in result.provenance
    assert result.spend.calls == 2


async def test_rejected_revision_cannot_supply_evidence_for_the_retained_candidate() -> None:
    evidence_seen: list[object] = []

    def verify(candidate: Candidate) -> Verification:
        scores = {"original": 0.5, "worse": 0.1, "better": 1.0}
        score = scores[candidate.output]
        return Verification("tests", score, score == 1.0, candidate.output)

    def revise(candidate: Candidate, evidence: object, round_index: int) -> str:
        evidence_seen.append(evidence)
        return "worse" if round_index == 0 else "better"

    result = await repair_with_evidence(
        InferenceRequest(prompt="task"),
        Candidate("original", "original"),
        verify=verify,
        revise=revise,
        max_rounds=2,
    )

    assert result.answer == "better"
    assert evidence_seen == ["original", "original"]


async def test_passing_repair_beats_higher_scoring_failed_original() -> None:
    def verify(candidate: Candidate) -> Verification:
        if candidate.output == "bad":
            return Verification("tests", 0.9, False, "invalid")
        return Verification("tests", 0.8, True, "valid")

    result = await repair_with_evidence(
        InferenceRequest(prompt="task"),
        Candidate("original", "bad"),
        verify=verify,
        revise=lambda candidate, evidence, round_index: "good",
    )

    assert result.answer == "good"
    assert result.stop_reason.value == "verified"
