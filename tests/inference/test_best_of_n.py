import pytest

from insideLLMs.inference import Candidate, InferenceRequest, Verification
from insideLLMs.inference.best_of_n import (
    NoVerifiedCandidateError,
    VerifierSpec,
    rank_candidates,
    select_best,
)


def test_rank_candidates_scores_once_and_breaks_ties_by_candidate_id() -> None:
    candidates = [Candidate("z", "first"), Candidate("a", "second")]
    calls: list[str] = []

    ranked = rank_candidates(
        candidates,
        score=lambda candidate: calls.append(candidate.id) or 1.0,
    )

    assert tuple(candidate.id for candidate, _ in ranked) == ("a", "z")
    assert calls == ["z", "a"]


async def test_best_of_n_applies_hard_verifiers_before_soft_scores() -> None:
    candidates = [
        Candidate(id="invalid", output="99"),
        Candidate(id="valid-low", output="4"),
        Candidate(id="valid-high", output="04"),
    ]
    calls: list[tuple[str, str]] = []

    def schema(candidate: Candidate) -> Verification:
        calls.append(("schema", candidate.id))
        passed = candidate.output in {"4", "04"}
        return Verification("schema", 1.0 if passed else 0.0, passed, "allowed answer")

    def preference(candidate: Candidate) -> Verification:
        calls.append(("preference", candidate.id))
        return Verification("preference", 0.9 if candidate.id == "valid-high" else 0.5, True)

    result = await select_best(
        InferenceRequest(prompt="2 + 2"),
        generate=lambda request, n: candidates[:n],
        n=3,
        verifiers=(
            VerifierSpec("schema", schema, hard=True),
            VerifierSpec("preference", preference),
        ),
        top_k=2,
    )

    assert result.answer == "04"
    assert ("preference", "invalid") not in calls
    assert result.provenance["pass_at_n"] is True
    assert result.provenance["oracle_best_score"] == 1.9
    assert result.provenance["top_k_ids"] == ("valid-high", "valid-low")


async def test_best_of_n_fails_closed_when_every_candidate_fails_a_hard_verifier() -> None:
    def reject(candidate: Candidate) -> Verification:
        return Verification("hidden-tests", 0.0, False, "failed")

    with pytest.raises(NoVerifiedCandidateError, match="hard verification"):
        await select_best(
            InferenceRequest(prompt="unsafe task"),
            generate=lambda request, n: [Candidate("unsafe", "rm everything")],
            n=1,
            verifiers=(VerifierSpec("hidden-tests", reject, hard=True),),
        )


async def test_best_of_n_votes_across_top_k_normalized_answers() -> None:
    candidates = [
        Candidate("minority", "no"),
        Candidate("majority-1", "YES"),
        Candidate("majority-2", "yes"),
    ]
    scores = {"minority": 1.0, "majority-1": 0.9, "majority-2": 0.8}

    result = await select_best(
        InferenceRequest(prompt="vote"),
        generate=lambda request, n: candidates,
        n=3,
        verifiers=(
            VerifierSpec(
                "score",
                lambda candidate: Verification("score", scores[candidate.id], True),
            ),
        ),
        top_k=3,
        normalize=lambda candidate: candidate.output.lower(),
    )

    assert result.answer == "YES"
    assert result.provenance["top_k_vote_counts"] == {"no": 1, "yes": 2}


async def test_best_of_n_rejects_duplicate_candidate_ids_before_verification() -> None:
    with pytest.raises(ValueError, match="candidate ids must be unique"):
        await select_best(
            InferenceRequest(prompt="duplicate ids"),
            generate=lambda request, n: [
                Candidate("duplicate", "unsafe"),
                Candidate("duplicate", "safe"),
            ],
            n=2,
            verifiers=(VerifierSpec("accept", lambda candidate: Verification("accept", 1, True)),),
        )


@pytest.mark.parametrize("scores", ([1.0], [1.0, 0.5, 0.0]))
async def test_best_of_n_rejects_malformed_judge_score_counts(scores: list[float]) -> None:
    with pytest.raises(ValueError, match="one score per candidate"):
        await select_best(
            InferenceRequest(prompt="judge"),
            generate=lambda request, n: [Candidate("a", "A"), Candidate("b", "B")],
            n=2,
            verifiers=(VerifierSpec("accept", lambda candidate: Verification("accept", 1, True)),),
            judge=lambda request, outputs: scores,
        )


async def test_best_of_n_requires_a_verifier_before_generation() -> None:
    generated = False

    def generate(request: InferenceRequest, n: int) -> list[Candidate]:
        nonlocal generated
        generated = True
        return [Candidate("a", "A")]

    with pytest.raises(ValueError, match="verifier"):
        await select_best(
            InferenceRequest(prompt="unverified"),
            generate=generate,
            n=1,
            verifiers=(),
        )

    assert generated is False


async def test_best_of_n_runs_hard_verifiers_before_soft_verifiers() -> None:
    calls: list[str] = []

    def soft(candidate: Candidate) -> Verification:
        calls.append("soft")
        return Verification("soft", 1.0, True)

    def hard(candidate: Candidate) -> Verification:
        calls.append("hard")
        return Verification("hard", 0.0, False)

    with pytest.raises(NoVerifiedCandidateError):
        await select_best(
            InferenceRequest(prompt="ordering"),
            generate=lambda request, n: [Candidate("a", "A")],
            n=1,
            verifiers=(VerifierSpec("soft", soft), VerifierSpec("hard", hard, hard=True)),
        )

    assert calls == ["hard"]
