"""Tests for insideLLMs/probes/judge.py module."""

import json
from unittest.mock import MagicMock

import pytest

from insideLLMs.probes.judge import JudgeScoredProbe, JudgeScorer
from insideLLMs.types import ProbeCategory, ProbeResult, ResultStatus

# ---------------------------------------------------------------------------
# JudgeScorer tests
# ---------------------------------------------------------------------------


class TestJudgeScorerInit:
    """Tests for JudgeScorer initialization."""

    def test_init_defaults(self):
        judge = MagicMock()
        scorer = JudgeScorer(judge_model=judge)
        assert scorer.judge_model is judge
        assert scorer.threshold == 4
        assert scorer.judge_kwargs == {}
        assert "factual" in scorer.rubric.lower()

    def test_init_custom_rubric_and_threshold(self):
        judge = MagicMock()
        scorer = JudgeScorer(
            judge_model=judge,
            rubric="Custom rubric",
            threshold=3,
            judge_kwargs={"temperature": 0.0},
        )
        assert scorer.rubric == "Custom rubric"
        assert scorer.threshold == 3
        assert scorer.judge_kwargs == {"temperature": 0.0}


class TestJudgeScorerScoreOutput:
    """Tests for JudgeScorer.score_output method."""

    @pytest.fixture
    def judge_model(self):
        return MagicMock()

    @pytest.fixture
    def scorer(self, judge_model):
        return JudgeScorer(judge_model=judge_model, threshold=4)

    def test_score_output_correct_json(self, scorer, judge_model):
        """Test scoring with valid JSON response from judge."""
        judge_model.generate.return_value = json.dumps(
            {"reasoning": "The answer matches the reference.", "score": 5}
        )
        result = scorer.score_output(
            model_output="Paris",
            reference="Paris",
            input_data="What is the capital of France?",
        )
        assert result["is_correct"] is True
        assert result["score"] == 5
        assert result["reasoning"] == "The answer matches the reference."
        assert result["raw_judge_response"] == judge_model.generate.return_value

    def test_score_output_incorrect(self, scorer, judge_model):
        """Test scoring with a low score from judge."""
        judge_model.generate.return_value = json.dumps(
            {"reasoning": "The answer is wrong.", "score": 1}
        )
        result = scorer.score_output(
            model_output="London",
            reference="Paris",
            input_data="What is the capital of France?",
        )
        assert result["is_correct"] is False
        assert result["score"] == 1

    def test_score_output_threshold_boundary(self, scorer, judge_model):
        """Test that threshold boundary is inclusive."""
        judge_model.generate.return_value = json.dumps({"reasoning": "Mostly correct.", "score": 4})
        result = scorer.score_output("Paris", "Paris", "Capital?")
        assert result["is_correct"] is True

        judge_model.generate.return_value = json.dumps(
            {"reasoning": "Close but imprecise.", "score": 3}
        )
        result = scorer.score_output("Paris area", "Paris", "Capital?")
        assert result["is_correct"] is False

    def test_score_output_passes_kwargs(self, judge_model):
        """Test that judge_kwargs are passed to the judge model."""
        scorer = JudgeScorer(
            judge_model=judge_model,
            judge_kwargs={"temperature": 0.0, "max_tokens": 200},
        )
        judge_model.generate.return_value = json.dumps({"reasoning": "ok", "score": 5})
        scorer.score_output("Paris", "Paris", "Capital?")
        call_kwargs = judge_model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 200

    def test_score_output_prompt_contains_rubric(self, scorer, judge_model):
        """Test that the prompt sent to judge contains the rubric."""
        scorer.rubric = "Check for logical consistency."
        judge_model.generate.return_value = json.dumps({"reasoning": "ok", "score": 5})
        scorer.score_output("yes", "yes", "Is A > B?")
        prompt = judge_model.generate.call_args[0][0]
        assert "Check for logical consistency." in prompt

    def test_score_output_prompt_contains_all_fields(self, scorer, judge_model):
        """Test the prompt includes input, reference, and model output."""
        judge_model.generate.return_value = json.dumps({"reasoning": "ok", "score": 5})
        scorer.score_output(
            model_output="The answer is 42",
            reference="42",
            input_data="What is 6*7?",
        )
        prompt = judge_model.generate.call_args[0][0]
        assert "The answer is 42" in prompt
        assert "42" in prompt
        assert "What is 6*7?" in prompt


class TestJudgeScorerParseResponse:
    """Tests for JudgeScorer._parse_judge_response method."""

    @pytest.fixture
    def scorer(self):
        return JudgeScorer(judge_model=MagicMock(), threshold=4)

    def test_parse_valid_json(self, scorer):
        response = '{"reasoning": "Good answer.", "score": 5}'
        result = scorer._parse_judge_response(response)
        assert result["score"] == 5
        assert result["reasoning"] == "Good answer."
        assert result["is_correct"] is True

    def test_parse_json_with_markdown_fences(self, scorer):
        response = '```json\n{"reasoning": "ok", "score": 4}\n```'
        result = scorer._parse_judge_response(response)
        assert result["score"] == 4
        assert result["is_correct"] is True

    def test_parse_json_with_plain_fences(self, scorer):
        response = '```\n{"reasoning": "ok", "score": 3}\n```'
        result = scorer._parse_judge_response(response)
        assert result["score"] == 3
        assert result["is_correct"] is False

    def test_parse_fallback_regex(self, scorer):
        """Test regex fallback when JSON is malformed."""
        response = 'Here is my eval: {"reasoning": "some reasoning", "score": 4} done'
        result = scorer._parse_judge_response(response)
        # Regex should extract the score
        assert result["score"] == 4

    def test_parse_score_only_regex(self, scorer):
        """Test regex fallback extracts score even without valid reasoning."""
        response = 'The model did well. "score": 5'
        result = scorer._parse_judge_response(response)
        assert result["score"] == 5

    def test_parse_no_score_found(self, scorer):
        """Test default score when nothing parseable is found."""
        response = "I think the model did okay."
        result = scorer._parse_judge_response(response)
        assert result["score"] == 0
        assert result["is_correct"] is False

    def test_parse_score_clamped_high(self, scorer):
        """Test that scores above 5 are clamped."""
        response = '{"reasoning": "amazing", "score": 9}'
        result = scorer._parse_judge_response(response)
        assert result["score"] == 5

    def test_parse_score_clamped_low(self, scorer):
        """Test that negative scores are clamped to 0."""
        response = '{"reasoning": "terrible", "score": -1}'
        result = scorer._parse_judge_response(response)
        assert result["score"] == 0

    def test_parse_preserves_raw_response(self, scorer):
        """Test that raw_judge_response is always set."""
        response = '{"reasoning": "ok", "score": 3}'
        result = scorer._parse_judge_response(response)
        assert result["raw_judge_response"] == response


# ---------------------------------------------------------------------------
# JudgeScoredProbe tests
# ---------------------------------------------------------------------------


class TestJudgeScoredProbeInit:
    """Tests for JudgeScoredProbe initialization."""

    def test_init_default_name(self):
        judge = MagicMock()
        probe = JudgeScoredProbe(judge_model=judge)
        assert probe.name == "JudgeScoredProbe"
        assert probe.category == ProbeCategory.FACTUALITY

    def test_init_custom_name_and_category(self):
        judge = MagicMock()
        probe = JudgeScoredProbe(
            name="MyJudge",
            judge_model=judge,
            category=ProbeCategory.LOGIC,
        )
        assert probe.name == "MyJudge"
        assert probe.category == ProbeCategory.LOGIC

    def test_init_requires_judge_model(self):
        with pytest.raises(ValueError, match="requires a judge_model"):
            JudgeScoredProbe()

    def test_init_scorer_configured(self):
        judge = MagicMock()
        probe = JudgeScoredProbe(
            judge_model=judge,
            rubric="Custom rubric",
            threshold=3,
            judge_kwargs={"temperature": 0.0},
        )
        assert probe.scorer.rubric == "Custom rubric"
        assert probe.scorer.threshold == 3
        assert probe.scorer.judge_kwargs == {"temperature": 0.0}

    def test_description_set(self):
        judge = MagicMock()
        probe = JudgeScoredProbe(judge_model=judge)
        assert "LLM judge" in probe.description


class TestJudgeScoredProbeRun:
    """Tests for JudgeScoredProbe.run method."""

    @pytest.fixture
    def subject_model(self):
        model = MagicMock()
        model.generate.return_value = "Paris"
        return model

    @pytest.fixture
    def judge_model(self):
        model = MagicMock()
        model.generate.return_value = json.dumps({"reasoning": "Correct.", "score": 5})
        return model

    @pytest.fixture
    def probe(self, judge_model):
        return JudgeScoredProbe(judge_model=judge_model)

    def test_run_returns_model_output(self, probe, subject_model):
        result = probe.run(subject_model, "What is the capital of France?")
        assert result == "Paris"
        subject_model.generate.assert_called_once()

    def test_run_passes_kwargs(self, probe, subject_model):
        probe.run(subject_model, "test prompt", temperature=0.5)
        call_kwargs = subject_model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    def test_run_converts_data_to_string(self, probe, subject_model):
        probe.run(subject_model, {"question": "test"})
        prompt = subject_model.generate.call_args[0][0]
        assert isinstance(prompt, str)


class TestJudgeScoredProbeEvaluateSingle:
    """Tests for JudgeScoredProbe.evaluate_single method."""

    @pytest.fixture
    def judge_model(self):
        return MagicMock()

    @pytest.fixture
    def probe(self, judge_model):
        return JudgeScoredProbe(judge_model=judge_model)

    def test_evaluate_single_correct(self, probe, judge_model):
        judge_model.generate.return_value = json.dumps({"reasoning": "Perfect match.", "score": 5})
        result = probe.evaluate_single("Paris", "Paris", "Capital of France?")
        assert result["is_correct"] is True
        assert result["score"] == 5
        assert result["reasoning"] == "Perfect match."

    def test_evaluate_single_incorrect(self, probe, judge_model):
        judge_model.generate.return_value = json.dumps({"reasoning": "Wrong city.", "score": 1})
        result = probe.evaluate_single("London", "Paris", "Capital of France?")
        assert result["is_correct"] is False
        assert result["score"] == 1

    def test_evaluate_single_converts_to_string(self, probe, judge_model):
        judge_model.generate.return_value = json.dumps({"reasoning": "ok", "score": 5})
        result = probe.evaluate_single(42, 42, {"q": "6*7"})
        assert result["is_correct"] is True
        # Verify the judge was called with string versions
        prompt = judge_model.generate.call_args[0][0]
        assert "42" in prompt


class TestJudgeScoredProbeScore:
    """Tests for JudgeScoredProbe.score (inherited from ScoredProbe)."""

    @pytest.fixture
    def probe(self):
        return JudgeScoredProbe(judge_model=MagicMock())

    def test_score_all_correct(self, probe):
        results = [
            ProbeResult(
                input="q1",
                output="a1",
                status=ResultStatus.SUCCESS,
                metadata={"is_correct": True, "score": 5},
            ),
            ProbeResult(
                input="q2",
                output="a2",
                status=ResultStatus.SUCCESS,
                metadata={"is_correct": True, "score": 4},
            ),
        ]
        score = probe.score(results)
        assert score.accuracy == 1.0

    def test_score_mixed(self, probe):
        results = [
            ProbeResult(
                input="q1",
                output="a1",
                status=ResultStatus.SUCCESS,
                metadata={"is_correct": True, "score": 5},
            ),
            ProbeResult(
                input="q2",
                output="a2",
                status=ResultStatus.SUCCESS,
                metadata={"is_correct": False, "score": 2},
            ),
        ]
        score = probe.score(results)
        assert score.accuracy == 0.5

    def test_score_with_errors(self, probe):
        results = [
            ProbeResult(
                input="q1",
                output="a1",
                status=ResultStatus.SUCCESS,
                metadata={"is_correct": True, "score": 5},
            ),
            ProbeResult(
                input="q2",
                status=ResultStatus.ERROR,
                error="timeout",
            ),
        ]
        score = probe.score(results)
        assert score.accuracy == 1.0  # 1 correct out of 1 evaluated
        assert score.error_rate == 0.5  # 1 error out of 2 total

    def test_score_empty(self, probe):
        score = probe.score([])
        assert score.accuracy is None


class TestJudgeScoredProbeIntegration:
    """Integration tests for end-to-end JudgeScoredProbe workflows."""

    def test_full_workflow(self):
        """Test complete workflow: run -> evaluate -> score."""
        # Setup subject model
        subject = MagicMock()
        subject.generate.side_effect = ["Paris", "London", "Berlin"]

        # Setup judge model
        judge = MagicMock()
        judge.generate.side_effect = [
            json.dumps({"reasoning": "Correct.", "score": 5}),
            json.dumps({"reasoning": "Wrong — should be Tokyo.", "score": 0}),
            json.dumps({"reasoning": "Correct.", "score": 5}),
        ]

        probe = JudgeScoredProbe(
            name="integration_test",
            judge_model=judge,
            rubric="Is the answer the correct capital city?",
        )

        # Run and evaluate
        dataset = [
            {"input": "Capital of France?", "reference": "Paris"},
            {"input": "Capital of Japan?", "reference": "Tokyo"},
            {"input": "Capital of Germany?", "reference": "Berlin"},
        ]

        results = []
        for item in dataset:
            output = probe.run(subject, item["input"])
            evaluation = probe.evaluate_single(output, item["reference"], item["input"])
            results.append(
                ProbeResult(
                    input=item["input"],
                    output=output,
                    status=ResultStatus.SUCCESS,
                    metadata=evaluation,
                )
            )

        score = probe.score(results)
        assert score.accuracy == pytest.approx(2 / 3)
        assert score.error_rate == 0.0

    def test_info_method(self):
        """Test that probe info is correct."""
        judge = MagicMock()
        probe = JudgeScoredProbe(
            name="my_judge",
            judge_model=judge,
            category=ProbeCategory.FACTUALITY,
        )
        info = probe.info()
        assert info["name"] == "my_judge"
        assert info["category"] == "factuality"
        assert info["type"] == "JudgeScoredProbe"

    def test_custom_threshold(self):
        """Test that custom threshold affects is_correct."""
        judge = MagicMock()
        judge.generate.return_value = json.dumps({"reasoning": "Roughly correct.", "score": 3})

        # With threshold=4 (default), score 3 is incorrect
        probe_strict = JudgeScoredProbe(judge_model=judge, threshold=4)
        result = probe_strict.evaluate_single("Paris area", "Paris", "Capital?")
        assert result["is_correct"] is False

        # With threshold=3, score 3 is correct
        probe_lenient = JudgeScoredProbe(judge_model=judge, threshold=3)
        result = probe_lenient.evaluate_single("Paris area", "Paris", "Capital?")
        assert result["is_correct"] is True
