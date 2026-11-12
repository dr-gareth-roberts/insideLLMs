"""Regression tests for Cluster C probe correctness fixes (C1–C5)."""

from unittest.mock import MagicMock

import pytest

from insideLLMs.exceptions import (
    EvaluationError,
    ModelTimeoutError,
    RateLimitError,
)
from insideLLMs.probes.attack import AttackProbe
from insideLLMs.probes.base import Probe, classify_batch_exception
from insideLLMs.probes.bias import BiasProbe
from insideLLMs.probes.code import CodeGenerationProbe
from insideLLMs.probes.instruction import (
    ConstraintComplianceProbe,
    InstructionFollowingProbe,
    MultiStepTaskProbe,
)
from insideLLMs.probes.judge import JudgeScorer
from insideLLMs.types import ProbeResult, ResultStatus


class _EchoProbe(Probe[str]):
    """Minimal probe for batch classification tests."""

    def run(self, model, data, **kwargs):
        return model.generate(data, **kwargs)


# ---------------------------------------------------------------------------
# C1 — batch failure classification
# ---------------------------------------------------------------------------


class TestClassifyBatchException:
    """classify_batch_exception uses explicit types, not substrings."""

    def test_model_timeout_error_is_timeout(self):
        status, msg = classify_batch_exception(ModelTimeoutError(model_id="m", timeout_seconds=5.0))
        assert status == ResultStatus.TIMEOUT
        assert "timed out" in msg.lower()

    def test_builtin_timeout_error_is_timeout(self):
        status, _msg = classify_batch_exception(TimeoutError("slow"))
        assert status == ResultStatus.TIMEOUT

    def test_rate_limit_error_is_rate_limited(self):
        status, _msg = classify_batch_exception(RateLimitError(model_id="m", retry_after=1.0))
        assert status == ResultStatus.RATE_LIMITED

    def test_generate_in_message_is_not_rate_limited(self):
        """'could not generate response' must not match substring 'rate'."""
        status, _msg = classify_batch_exception(RuntimeError("could not generate response"))
        assert status == ResultStatus.ERROR

    def test_plain_exception_is_error(self):
        status, msg = classify_batch_exception(ValueError("boom"))
        assert status == ResultStatus.ERROR
        assert "ValueError" in msg


class TestRunBatchClassification:
    """run_batch sequential and parallel paths share the classifier."""

    def test_sequential_model_timeout_marked_timeout(self):
        probe = _EchoProbe(name="t")
        model = MagicMock()
        model.generate = MagicMock(side_effect=ModelTimeoutError(model_id="m", timeout_seconds=1.0))
        results = probe.run_batch(model, ["a"])
        assert results[0].status == ResultStatus.TIMEOUT

    def test_sequential_generate_message_not_rate_limited(self):
        probe = _EchoProbe(name="t")
        model = MagicMock()
        model.generate = MagicMock(side_effect=RuntimeError("could not generate response"))
        results = probe.run_batch(model, ["a"])
        assert results[0].status == ResultStatus.ERROR

    def test_sequential_rate_limit_error(self):
        probe = _EchoProbe(name="t")
        model = MagicMock()
        model.generate = MagicMock(side_effect=RateLimitError(model_id="m"))
        results = probe.run_batch(model, ["a"])
        assert results[0].status == ResultStatus.RATE_LIMITED

    def test_parallel_model_timeout_marked_timeout(self):
        probe = _EchoProbe(name="t")
        model = MagicMock()
        model.generate = MagicMock(side_effect=ModelTimeoutError(model_id="m", timeout_seconds=1.0))
        results = probe.run_batch(model, ["a", "b"], max_workers=2)
        assert all(r.status == ResultStatus.TIMEOUT for r in results)


# ---------------------------------------------------------------------------
# C2 — instruction compliance
# ---------------------------------------------------------------------------


class TestInstructionConstraintRejection:
    def test_language_constraint_rejected_at_format(self):
        probe = InstructionFollowingProbe()
        with pytest.raises(ValueError, match="language"):
            probe._format_constraints({"language": "French"})

    def test_tone_constraint_rejected_at_evaluate(self):
        probe = InstructionFollowingProbe()
        with pytest.raises(ValueError, match="tone"):
            probe.evaluate_single("hi", {"constraints": {"tone": "casual"}})

    def test_unsupported_format_rejected(self):
        probe = InstructionFollowingProbe()
        with pytest.raises(ValueError, match="Unsupported format"):
            probe.evaluate_single("x", {"constraints": {"format": "xml"}})


class TestInstructionNumericLimitValidation:
    """G5 — word/item limits must be positive non-bool integers."""

    @pytest.mark.parametrize(
        "key,value",
        [
            ("min_words", -1),
            ("max_words", 0),
            ("min_items", True),
            ("max_items", False),
            ("min_words", [5]),
            ("max_words", 1.5),
            ("min_items", "3"),
        ],
    )
    def test_invalid_limits_rejected_at_evaluate(self, key, value):
        probe = InstructionFollowingProbe()
        with pytest.raises(ValueError, match="positive integer"):
            probe.evaluate_single("hello world", {"constraints": {key: value}})

    @pytest.mark.parametrize(
        "key,value",
        [
            ("min_words", -1),
            ("max_words", 0),
            ("min_items", True),
            ("max_items", [3]),
        ],
    )
    def test_invalid_limits_rejected_at_format(self, key, value):
        probe = InstructionFollowingProbe()
        with pytest.raises(ValueError, match="positive integer"):
            probe._format_constraints({key: value})

    def test_valid_positive_limits_accepted(self):
        probe = InstructionFollowingProbe()
        result = probe.evaluate_single(
            "one two three four five",
            {"constraints": {"min_words": 3, "max_words": 10}},
        )
        assert result.metadata["within_min_words"] is True
        assert result.metadata["within_max_words"] is True


class TestInstructionItemCountZero:
    def test_min_items_with_empty_output_scores_zero(self):
        probe = InstructionFollowingProbe()
        result = probe.evaluate_single(
            "",
            {"constraints": {"min_items": 2}},
        )
        assert result.metadata["item_count"] == 0
        assert result.metadata["within_min_items"] is False
        assert result.metadata["score"] == 0.0


class TestMultiStepSectionScoping:
    def test_keyword_in_wrong_step_does_not_credit(self):
        probe = MultiStepTaskProbe()
        output = "Step 1: mentions javascript early\nStep 2: only other text"
        reference = {
            "steps": ["a", "b"],
            "expected": {"step_1": ["python"], "step_2": ["javascript"]},
        }
        result = probe.evaluate_single(output, reference)
        assert result.metadata["step_1"] == 0.0
        assert result.metadata["step_2"] == 0.0

    def test_correct_sections_score_full(self):
        probe = MultiStepTaskProbe()
        output = "Step 1: python here\nStep 2: javascript there"
        reference = {
            "steps": ["a", "b"],
            "expected": {"step_1": ["python"], "step_2": ["javascript"]},
        }
        result = probe.evaluate_single(output, reference)
        assert result.metadata["step_1"] == 1.0
        assert result.metadata["step_2"] == 1.0


class TestConstraintLimitValidation:
    def test_limit_zero_rejected_at_init(self):
        with pytest.raises(ValueError, match="positive integer"):
            ConstraintComplianceProbe(constraint_type="word_limit", limit=0)

    def test_limit_negative_rejected_at_init(self):
        with pytest.raises(ValueError, match="positive integer"):
            ConstraintComplianceProbe(constraint_type="character_limit", limit=-1)

    def test_limit_bool_rejected_at_init(self):
        with pytest.raises(ValueError, match="positive integer"):
            ConstraintComplianceProbe(constraint_type="word_limit", limit=True)

    @pytest.mark.parametrize("bad_limit", [-1, 0, True, False])
    def test_invalid_reference_override_rejected(self, bad_limit):
        probe = ConstraintComplianceProbe(constraint_type="word_limit", limit=10)
        with pytest.raises(ValueError, match="positive integer"):
            probe.evaluate_single("a few words here", reference=bad_limit)

    def test_score_never_exceeds_one_when_over_limit(self):
        probe = ConstraintComplianceProbe(constraint_type="word_limit", limit=1)
        result = probe.evaluate_single("a b c d e", None)
        assert 0.0 <= result.metadata["score"] <= 1.0


# ---------------------------------------------------------------------------
# C3 — code generation correctness
# ---------------------------------------------------------------------------


class TestCodeGenerationCorrectness:
    def test_empty_python_is_not_correct(self):
        probe = CodeGenerationProbe(language="python")
        result = probe.evaluate_single("", reference=None)
        assert result["is_correct"] is False
        assert result["score"] == 0.0
        assert result.get("empty_code") is True

    def test_invalid_python_with_pattern_is_not_correct(self):
        probe = CodeGenerationProbe(language="python")
        invalid = "def broken(:\n    pass"
        result = probe.evaluate_single(
            invalid,
            reference={"patterns": ["def broken"]},
        )
        assert result["syntax_valid"] is False
        assert result["is_correct"] is False
        assert result["score"] == 0.0

    def test_plain_english_as_javascript_not_correct(self):
        probe = CodeGenerationProbe(language="javascript")
        result = probe.evaluate_single(
            "Just write a friendly hello message without any code.",
            reference=None,
        )
        assert result["is_correct"] is False
        assert result["score"] < 0.5

    def test_unexecuted_tests_metadata_does_not_grant_full_credit(self):
        """G6 — declared-but-unexecuted tests must not bypass non-Python cap."""
        probe = CodeGenerationProbe(language="javascript")
        prose = "Just write a friendly hello message without any code."
        result = probe.evaluate_single(
            prose,
            reference={"tests": ["not executed"], "patterns": ["hello"]},
        )
        assert result["is_correct"] is False
        assert result["score"] < 0.5
        assert result["score"] <= 0.49

    def test_unexecuted_test_cases_metadata_does_not_grant_full_credit(self):
        """G6 — test_cases key is likewise not executed evidence."""
        probe = CodeGenerationProbe(language="javascript")
        prose = "A short explanation in plain English only."
        result = probe.evaluate_single(
            prose,
            reference={"test_cases": [{"input": 1, "expected": 2}]},
        )
        assert result["is_correct"] is False
        assert result["score"] <= 0.49

    def test_valid_python_with_pattern_can_be_correct(self):
        probe = CodeGenerationProbe(language="python")
        code = "def add(a, b):\n    return a + b"
        result = probe.evaluate_single(code, {"patterns": ["def add"]})
        assert result["syntax_valid"] is True
        assert result["is_correct"] is True
        assert result["score"] >= 0.5


# ---------------------------------------------------------------------------
# C4 — bias/attack zero-output health metrics
# ---------------------------------------------------------------------------


class TestBiasScoreHealthMetrics:
    def test_all_errors_preserves_error_rate(self):
        probe = BiasProbe()
        results = [
            ProbeResult(input=("a", "b"), output=None, status=ResultStatus.ERROR),
            ProbeResult(input=("c", "d"), output=None, status=ResultStatus.ERROR),
        ]
        score = probe.score(results)
        assert score.error_rate == 1.0
        assert score.accuracy is None

    def test_timeouts_not_counted_as_errors(self):
        probe = BiasProbe()
        results = [
            ProbeResult(input=("a", "b"), output=None, status=ResultStatus.TIMEOUT),
        ]
        score = probe.score(results)
        assert score.error_rate == 0.0
        assert score.accuracy is None


class TestAttackScoreHealthMetrics:
    def test_timeout_only_run_not_labeled_all_errors(self):
        probe = AttackProbe()
        results = [
            ProbeResult(input="p", output=None, status=ResultStatus.TIMEOUT),
            ProbeResult(input="q", output=None, status=ResultStatus.TIMEOUT),
        ]
        score = probe.score(results)
        assert score.error_rate == 0.0
        assert score.accuracy is None

    def test_error_only_run_preserves_error_rate(self):
        probe = AttackProbe()
        results = [
            ProbeResult(input="p", output=None, status=ResultStatus.ERROR, error="x"),
        ]
        score = probe.score(results)
        assert score.error_rate == 1.0
        assert score.accuracy is None


# ---------------------------------------------------------------------------
# C5 — judge threshold and JSON shape
# ---------------------------------------------------------------------------


class TestJudgeThresholdAndJson:
    def test_threshold_negative_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 5\]"):
            JudgeScorer(judge_model=MagicMock(), threshold=-1)

    def test_threshold_above_five_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 5\]"):
            JudgeScorer(judge_model=MagicMock(), threshold=6)

    def test_threshold_zero_accepts_score_zero(self):
        scorer = JudgeScorer(judge_model=MagicMock(), threshold=0)
        result = scorer._parse_judge_response('{"reasoning": "empty", "score": 0}')
        assert result["is_correct"] is True
        assert result["score"] == 0

    def test_threshold_five_accepts_score_five(self):
        scorer = JudgeScorer(judge_model=MagicMock(), threshold=5)
        result = scorer._parse_judge_response('{"reasoning": "perfect", "score": 5}')
        assert result["is_correct"] is True

    def test_json_array_raises_evaluation_error(self):
        scorer = JudgeScorer(judge_model=MagicMock(), threshold=4)
        with pytest.raises(EvaluationError, match="object/mapping"):
            scorer._parse_judge_response("[]")

    def test_json_string_raises_evaluation_error(self):
        scorer = JudgeScorer(judge_model=MagicMock(), threshold=4)
        with pytest.raises(EvaluationError, match="object/mapping"):
            scorer._parse_judge_response('"just a string"')
