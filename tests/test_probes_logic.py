"""Tests for insideLLMs/probes/logic.py module."""

from unittest.mock import MagicMock

import pytest

from insideLLMs.probes.logic import LogicProbe
from insideLLMs.types import ProbeCategory, ProbeResult, ResultStatus


class TestLogicProbeInitialization:
    """Tests for LogicProbe initialization."""

    def test_default_initialization(self):
        """Test default probe initialization."""
        probe = LogicProbe()
        assert probe.name == "LogicProbe"
        assert probe.category == ProbeCategory.LOGIC
        assert probe.extract_answer is True
        assert "{problem}" in probe.prompt_template

    def test_custom_name(self):
        """Test probe with custom name."""
        probe = LogicProbe(name="MyLogicProbe")
        assert probe.name == "MyLogicProbe"

    def test_custom_template(self):
        """Test probe with custom template."""
        template = "Solve this: {problem}"
        probe = LogicProbe(prompt_template=template)
        assert probe.prompt_template == template

    def test_disable_extract_answer(self):
        """Test probe with extract_answer disabled."""
        probe = LogicProbe(extract_answer=False)
        assert probe.extract_answer is False


class TestLogicProbeRun:
    """Tests for LogicProbe.run method."""

    def test_run_with_string_problem(self):
        """Test run with string problem."""
        probe = LogicProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="A is greater than C")

        result = probe.run(mock_model, "If A > B and B > C, what is the relationship?")

        assert result == "A is greater than C"
        mock_model.generate.assert_called_once()

    def test_run_with_dict_problem(self):
        """Test run with dict problem."""
        probe = LogicProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="The answer is 42")

        problem = {"problem": "What is the answer to life?"}
        result = probe.run(mock_model, problem)

        assert result == "The answer is 42"

    def test_run_with_dict_question_key(self):
        """Test run with dict using 'question' key."""
        probe = LogicProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="Response")

        problem = {"question": "What is 2+2?"}
        result = probe.run(mock_model, problem)

        assert result == "Response"

    def test_run_formats_prompt_correctly(self):
        """Test that prompt is formatted correctly."""
        probe = LogicProbe(prompt_template="Question: {problem}")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="Answer")

        probe.run(mock_model, "What is 2+2?")

        call_args = mock_model.generate.call_args[0][0]
        assert "Question: What is 2+2?" in call_args


class TestLogicProbeEvaluateSingle:
    """Tests for LogicProbe.evaluate_single method."""

    def test_evaluate_correct_answer(self):
        """Test evaluation with correct answer."""
        probe = LogicProbe()
        result = probe.evaluate_single(
            model_output="After analyzing, the answer is 42",
            reference="42",
            input_data="What is the answer?",
        )

        assert result["is_correct"] is True
        assert result["extracted_answer"] == "42"
        assert result["reference_answer"] == "42"

    def test_evaluate_incorrect_answer(self):
        """Test evaluation with incorrect answer."""
        probe = LogicProbe()
        result = probe.evaluate_single(
            model_output="The answer is 100", reference="42", input_data="What is the answer?"
        )

        assert result["is_correct"] is False

    def test_evaluate_no_reference(self):
        """Test evaluation without reference."""
        probe = LogicProbe()
        result = probe.evaluate_single(
            model_output="Some response", reference=None, input_data="Problem"
        )

        assert result["evaluated"] is False

    def test_evaluate_contains_reasoning(self):
        """Test evaluation detects reasoning."""
        probe = LogicProbe()
        result = probe.evaluate_single(
            model_output="First, let's analyze. Then, we conclude. Therefore, the answer is yes.",
            reference="yes",
            input_data="Problem",
        )

        assert result["has_reasoning"] is True

    def test_evaluate_output_shape_and_types(self):
        """Evaluation should return a stable metric surface."""
        probe = LogicProbe()
        result = probe.evaluate_single(
            model_output="Step 1: Analyze. Therefore: YES",
            reference=" yes ",
            input_data="Problem",
        )

        expected_keys = {
            "is_correct",
            "extracted_answer",
            "reference_answer",
            "response_length",
            "has_reasoning",
        }
        assert set(result.keys()) == expected_keys
        assert isinstance(result["is_correct"], bool)
        assert isinstance(result["has_reasoning"], bool)
        assert isinstance(result["response_length"], int)
        # Answers are normalized for consistent downstream metrics.
        assert result["extracted_answer"] == "yes"
        assert result["reference_answer"] == "yes"

    def test_evaluate_without_reference_is_minimal(self):
        """No-reference evaluation should stay intentionally small."""
        probe = LogicProbe(extract_answer=False)
        result = probe.evaluate_single(
            model_output="The answer is 42",
            reference=None,
            input_data="Problem",
        )

        assert result == {"evaluated": False}


class TestLogicProbeExtractFinalAnswer:
    """Tests for LogicProbe._extract_final_answer method."""

    def test_extract_answer_is_pattern(self):
        """Test extraction with 'the answer is' pattern."""
        probe = LogicProbe()
        answer = probe._extract_final_answer("After calculation, the answer is 42.")
        assert answer == "42"

    def test_extract_therefore_pattern(self):
        """Test extraction with 'therefore' pattern."""
        probe = LogicProbe()
        answer = probe._extract_final_answer("Based on logic, therefore, A > C.")
        assert answer == "A > C"

    def test_extract_final_answer_pattern(self):
        """Test extraction with 'final answer' pattern."""
        probe = LogicProbe()
        answer = probe._extract_final_answer("Final answer: Yes")
        assert answer == "Yes"

    def test_extract_conclusion_pattern(self):
        """Test extraction with 'conclusion' pattern."""
        probe = LogicProbe()
        answer = probe._extract_final_answer("Conclusion: The statement is true")
        assert answer == "The statement is true"

    def test_extract_last_sentence(self):
        """Test extraction falls back to last sentence."""
        probe = LogicProbe()
        answer = probe._extract_final_answer("Some reasoning. More analysis. The result is B")
        # Should get "The result is B" or similar
        assert len(answer) > 0

    def test_extract_empty_last_sentence(self):
        """Test extraction with empty last sentence."""
        probe = LogicProbe()
        answer = probe._extract_final_answer("First sentence. Second sentence.")
        assert len(answer) > 0


class TestLogicProbeHasReasoning:
    """Tests for LogicProbe._has_reasoning method."""

    def test_has_step_indicator(self):
        """Test detection of 'step' indicator."""
        probe = LogicProbe()
        assert probe._has_reasoning("Step 1: Analyze the data") is True

    def test_has_first_indicator(self):
        """Test detection of 'first' indicator."""
        probe = LogicProbe()
        assert probe._has_reasoning("First, we need to understand") is True

    def test_has_therefore_indicator(self):
        """Test detection of 'therefore' indicator."""
        probe = LogicProbe()
        assert probe._has_reasoning("Therefore, the answer is yes") is True

    def test_has_because_indicator(self):
        """Test detection of 'because' indicator."""
        probe = LogicProbe()
        assert probe._has_reasoning("This is true because of X") is True

    def test_has_given_indicator(self):
        """Test detection of 'given' indicator."""
        probe = LogicProbe()
        assert probe._has_reasoning("Given that A > B") is True

    def test_no_reasoning_indicators(self):
        """Test response without reasoning indicators."""
        probe = LogicProbe()
        assert probe._has_reasoning("The answer is 42.") is False


class TestLogicProbeScore:
    """Tests for LogicProbe.score method."""

    def test_score_with_successful_results(self):
        """Test score calculation with successful results."""
        probe = LogicProbe()

        results = [
            ProbeResult(
                input="Problem 1",
                output="First, let's analyze. Therefore, the answer is yes.",
                status=ResultStatus.SUCCESS,
            ),
            ProbeResult(
                input="Problem 2",
                output="Step by step: the answer is no.",
                status=ResultStatus.SUCCESS,
            ),
        ]

        score = probe.score(results)

        assert score.custom_metrics["reasoning_rate"] == 1.0  # Both have reasoning
        assert score.custom_metrics["avg_response_length"] > 0

    def test_score_with_mixed_results(self):
        """Test score with mix of success and error."""
        probe = LogicProbe()

        results = [
            ProbeResult(
                input="Problem 1",
                output="The answer is yes.",
                status=ResultStatus.SUCCESS,
            ),
            ProbeResult(
                input="Problem 2",
                output=None,
                status=ResultStatus.ERROR,
                error="Test error",
            ),
        ]

        score = probe.score(results)

        # Only successful results are analyzed
        assert score.custom_metrics["reasoning_rate"] == 0.0  # No reasoning in first one

    def test_score_with_partial_reasoning(self):
        """Test score with some responses having reasoning."""
        probe = LogicProbe()

        results = [
            ProbeResult(
                input="Problem 1",
                output="First, let's analyze. Therefore, yes.",
                status=ResultStatus.SUCCESS,
            ),
            ProbeResult(
                input="Problem 2",
                output="Yes.",
                status=ResultStatus.SUCCESS,
            ),
        ]

        score = probe.score(results)

        assert score.custom_metrics["reasoning_rate"] == 0.5  # 1 of 2 has reasoning

    def test_score_empty_results(self):
        """Test score with empty results."""
        probe = LogicProbe()
        score = probe.score([])

        assert score.custom_metrics["reasoning_rate"] == 0
        assert score.custom_metrics["avg_response_length"] == 0


class TestLogicProbeIntegration:
    """Integration tests for LogicProbe."""

    def test_full_workflow(self):
        """Test complete logic probe workflow."""
        probe = LogicProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(
            return_value="Let me analyze step by step. Given A > B and B > C, "
            "since > is transitive, therefore, A > C. "
            "The final answer is: A is greater than C."
        )

        # Run the probe
        response = probe.run(mock_model, "If A > B and B > C, what is the relationship?")

        # Evaluate - the reference should contain part of the extracted answer
        evaluation = probe.evaluate_single(
            model_output=response,
            reference="a is greater than c",  # Case insensitive matching
            input_data="If A > B and B > C, what is the relationship?",
        )

        assert evaluation["is_correct"] is True
        assert evaluation["has_reasoning"] is True
        assert "a is greater than c" in evaluation["extracted_answer"]

    def test_batch_workflow(self):
        """Test batch processing workflow."""
        probe = LogicProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(
            side_effect=[
                "First, analyze. Therefore, yes.",
                "The answer is no.",
                "Step 1: Consider. Thus, maybe.",
            ]
        )

        problems = ["Problem 1", "Problem 2", "Problem 3"]
        results = probe.run_batch(mock_model, problems)

        assert len(results) == 3
        assert all(r.status == ResultStatus.SUCCESS for r in results)

        score = probe.score(results)
        # 2 of 3 have reasoning (first and third)
        assert score.custom_metrics["reasoning_rate"] == 2 / 3
