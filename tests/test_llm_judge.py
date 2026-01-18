"""Tests for LLM-as-a-Judge evaluation framework."""

import json
import pytest
from unittest.mock import MagicMock, patch

from insideLLMs.evaluation import (
    JudgeCriterion,
    JudgeResult,
    JudgeModel,
    JudgeEvaluator,
    create_judge,
    create_evaluator,
    HELPFULNESS_CRITERIA,
    ACCURACY_CRITERIA,
    SAFETY_CRITERIA,
    CODE_QUALITY_CRITERIA,
    DEFAULT_JUDGE_SYSTEM_PROMPT,
    DEFAULT_JUDGE_TEMPLATE,
)
from insideLLMs.models import DummyModel


class TestJudgeCriterion:
    """Tests for JudgeCriterion dataclass."""

    def test_default_values(self):
        """Test criterion with default values."""
        criterion = JudgeCriterion(
            name="test",
            description="Test criterion",
        )
        assert criterion.name == "test"
        assert criterion.description == "Test criterion"
        assert criterion.weight == 1.0
        assert criterion.scale_min == 1
        assert criterion.scale_max == 5

    def test_custom_values(self):
        """Test criterion with custom values."""
        criterion = JudgeCriterion(
            name="custom",
            description="Custom criterion",
            weight=0.5,
            scale_min=0,
            scale_max=10,
        )
        assert criterion.weight == 0.5
        assert criterion.scale_min == 0
        assert criterion.scale_max == 10


class TestJudgeResult:
    """Tests for JudgeResult dataclass."""

    def test_basic_result(self):
        """Test basic JudgeResult creation."""
        result = JudgeResult(
            overall_score=0.8,
            criteria_scores={"helpfulness": 4, "accuracy": 5},
            reasoning="Good response",
            raw_response="...",
            passed=True,
        )
        assert result.overall_score == 0.8
        assert result.criteria_scores["helpfulness"] == 4
        assert result.passed is True

    def test_to_evaluation_result(self):
        """Test conversion to EvaluationResult."""
        result = JudgeResult(
            overall_score=0.75,
            criteria_scores={"test": 4},
            reasoning="Test reasoning",
            raw_response="raw",
            passed=True,
        )
        eval_result = result.to_evaluation_result()
        assert eval_result.score == 0.75
        assert eval_result.passed is True
        assert eval_result.metric_name == "llm_judge"
        assert "criteria_scores" in eval_result.details
        assert "reasoning" in eval_result.details


class TestPredefinedCriteria:
    """Tests for predefined criteria sets."""

    def test_helpfulness_criteria(self):
        """Test helpfulness criteria are defined."""
        assert len(HELPFULNESS_CRITERIA) == 3
        names = [c.name for c in HELPFULNESS_CRITERIA]
        assert "helpfulness" in names
        assert "completeness" in names
        assert "clarity" in names

    def test_accuracy_criteria(self):
        """Test accuracy criteria are defined."""
        assert len(ACCURACY_CRITERIA) == 3
        names = [c.name for c in ACCURACY_CRITERIA]
        assert "factual_accuracy" in names
        assert "logical_consistency" in names

    def test_safety_criteria(self):
        """Test safety criteria are defined."""
        assert len(SAFETY_CRITERIA) == 3
        names = [c.name for c in SAFETY_CRITERIA]
        assert "harmlessness" in names
        assert "bias_free" in names

    def test_code_quality_criteria(self):
        """Test code quality criteria are defined."""
        assert len(CODE_QUALITY_CRITERIA) == 4
        names = [c.name for c in CODE_QUALITY_CRITERIA]
        assert "correctness" in names
        assert "efficiency" in names
        assert "readability" in names


class TestJudgeModel:
    """Tests for JudgeModel class."""

    def test_initialization_defaults(self):
        """Test JudgeModel initialization with defaults."""
        mock_model = MagicMock()
        judge = JudgeModel(judge_model=mock_model)

        assert judge.judge_model is mock_model
        assert judge.criteria == HELPFULNESS_CRITERIA
        assert judge.threshold == 0.6
        assert judge.temperature == 0.0
        assert judge.system_prompt == DEFAULT_JUDGE_SYSTEM_PROMPT

    def test_initialization_custom(self):
        """Test JudgeModel initialization with custom parameters."""
        mock_model = MagicMock()
        custom_criteria = [
            JudgeCriterion(name="custom", description="Custom test")
        ]
        judge = JudgeModel(
            judge_model=mock_model,
            criteria=custom_criteria,
            threshold=0.8,
            temperature=0.5,
        )

        assert judge.criteria == custom_criteria
        assert judge.threshold == 0.8
        assert judge.temperature == 0.5

    def test_build_criteria_section(self):
        """Test criteria section building."""
        mock_model = MagicMock()
        criteria = [
            JudgeCriterion(
                name="test1",
                description="First criterion",
                weight=1.0,
                scale_min=1,
                scale_max=5,
            ),
            JudgeCriterion(
                name="test2",
                description="Second criterion",
                weight=0.5,
                scale_min=1,
                scale_max=10,
            ),
        ]
        judge = JudgeModel(judge_model=mock_model, criteria=criteria)
        section = judge._build_criteria_section()

        assert "test1" in section
        assert "test2" in section
        assert "First criterion" in section
        assert "weight: 1.0" in section
        assert "scale: 1-5" in section
        assert "scale: 1-10" in section

    def test_parse_judge_response_with_code_block(self):
        """Test parsing JSON from markdown code block."""
        mock_model = MagicMock()
        judge = JudgeModel(judge_model=mock_model)

        response = '''Here is my evaluation:

```json
{
    "criteria_scores": {
        "helpfulness": {"score": 4, "reasoning": "Good"}
    },
    "overall_reasoning": "Overall good",
    "overall_score": 4
}
```
'''
        parsed = judge._parse_judge_response(response)
        assert "criteria_scores" in parsed
        assert parsed["criteria_scores"]["helpfulness"]["score"] == 4

    def test_parse_judge_response_raw_json(self):
        """Test parsing raw JSON without code block."""
        mock_model = MagicMock()
        judge = JudgeModel(judge_model=mock_model)

        response = '{"criteria_scores": {"test": 5}, "overall_reasoning": "Good"}'
        parsed = judge._parse_judge_response(response)
        assert parsed["criteria_scores"]["test"] == 5

    def test_parse_judge_response_invalid(self):
        """Test parsing invalid response raises error."""
        mock_model = MagicMock()
        judge = JudgeModel(judge_model=mock_model)

        with pytest.raises(ValueError, match="Could not find JSON"):
            judge._parse_judge_response("No JSON here!")

    def test_calculate_overall_score(self):
        """Test overall score calculation with normalization."""
        mock_model = MagicMock()
        criteria = [
            JudgeCriterion(name="a", description="", weight=1.0, scale_min=1, scale_max=5),
            JudgeCriterion(name="b", description="", weight=1.0, scale_min=1, scale_max=5),
        ]
        judge = JudgeModel(judge_model=mock_model, criteria=criteria)

        # Score of 5 on 1-5 scale = 1.0 normalized
        # Score of 3 on 1-5 scale = 0.5 normalized
        scores = {"a": 5, "b": 3}
        overall = judge._calculate_overall_score(scores)

        # Expected: (1.0 * 1.0 + 0.5 * 1.0) / 2.0 = 0.75
        assert abs(overall - 0.75) < 0.01

    def test_calculate_overall_score_weighted(self):
        """Test weighted score calculation."""
        mock_model = MagicMock()
        criteria = [
            JudgeCriterion(name="a", description="", weight=2.0, scale_min=1, scale_max=5),
            JudgeCriterion(name="b", description="", weight=1.0, scale_min=1, scale_max=5),
        ]
        judge = JudgeModel(judge_model=mock_model, criteria=criteria)

        # a=5 (normalized 1.0) with weight 2.0
        # b=3 (normalized 0.5) with weight 1.0
        scores = {"a": 5, "b": 3}
        overall = judge._calculate_overall_score(scores)

        # Expected: (1.0 * 2.0 + 0.5 * 1.0) / 3.0 = 2.5/3.0 = 0.833...
        assert abs(overall - 0.833) < 0.01

    def test_evaluate_with_chat_model(self):
        """Test evaluation using a model with chat support."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value=json.dumps({
            "criteria_scores": {
                "helpfulness": {"score": 4, "reasoning": "Good"},
                "completeness": {"score": 5, "reasoning": "Complete"},
                "clarity": {"score": 4, "reasoning": "Clear"},
            },
            "overall_reasoning": "Good response overall",
            "overall_score": 4.3
        }))

        judge = JudgeModel(judge_model=mock_model)
        result = judge.evaluate(
            prompt="What is 2+2?",
            response="2+2 equals 4.",
        )

        assert isinstance(result, JudgeResult)
        assert result.overall_score > 0
        assert "helpfulness" in result.criteria_scores
        mock_model.chat.assert_called_once()

    def test_evaluate_with_generate_model(self):
        """Test evaluation using a model without chat support."""
        mock_model = MagicMock(spec=['generate'])
        mock_model.generate = MagicMock(return_value=json.dumps({
            "criteria_scores": {
                "helpfulness": 4,
                "completeness": 5,
                "clarity": 4,
            },
            "overall_reasoning": "Good response",
            "overall_score": 4.3
        }))

        judge = JudgeModel(judge_model=mock_model)
        result = judge.evaluate(
            prompt="What is 2+2?",
            response="2+2 equals 4.",
        )

        assert isinstance(result, JudgeResult)
        mock_model.generate.assert_called_once()

    def test_evaluate_with_reference(self):
        """Test evaluation with reference answer."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value=json.dumps({
            "criteria_scores": {"helpfulness": 5},
            "overall_reasoning": "Matches reference",
        }))

        judge = JudgeModel(
            judge_model=mock_model,
            criteria=[JudgeCriterion(name="helpfulness", description="")]
        )
        result = judge.evaluate(
            prompt="What is 2+2?",
            response="4",
            reference="The answer is 4.",
        )

        # Verify reference was included in the prompt
        call_args = mock_model.chat.call_args
        messages = call_args[0][0]
        assert "Reference Answer" in messages[1]["content"]

    def test_evaluate_error_handling(self):
        """Test evaluation handles errors gracefully."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(side_effect=Exception("API Error"))

        judge = JudgeModel(judge_model=mock_model)
        result = judge.evaluate(
            prompt="Test",
            response="Test response",
        )

        assert result.overall_score == 0.0
        assert result.passed is False
        assert "API Error" in result.reasoning
        assert "error" in result.details

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value=json.dumps({
            "criteria_scores": {"helpfulness": 4},
            "overall_reasoning": "OK",
        }))

        judge = JudgeModel(
            judge_model=mock_model,
            criteria=[JudgeCriterion(name="helpfulness", description="")]
        )
        results = judge.evaluate_batch(
            prompts=["Q1", "Q2"],
            responses=["A1", "A2"],
        )

        assert len(results) == 2
        assert all(isinstance(r, JudgeResult) for r in results)
        assert mock_model.chat.call_count == 2

    def test_compare(self):
        """Test pairwise comparison."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value=json.dumps({
            "criteria_comparison": {
                "helpfulness": {"winner": "A", "reasoning": "More helpful"}
            },
            "overall_winner": "A",
            "overall_reasoning": "A is better",
        }))

        judge = JudgeModel(
            judge_model=mock_model,
            criteria=[JudgeCriterion(name="helpfulness", description="")]
        )
        result = judge.compare(
            prompt="What is 2+2?",
            response_a="4",
            response_b="Around 4",
        )

        assert result["winner"] == "A"
        assert "criteria_comparison" in result
        assert "reasoning" in result

    def test_compare_error_handling(self):
        """Test comparison handles errors."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(side_effect=Exception("API Error"))

        judge = JudgeModel(judge_model=mock_model)
        result = judge.compare(
            prompt="Test",
            response_a="A",
            response_b="B",
        )

        assert result["winner"] == "error"
        assert "API Error" in result["reasoning"]


class TestJudgeEvaluator:
    """Tests for JudgeEvaluator wrapper class."""

    def test_initialization(self):
        """Test JudgeEvaluator initialization."""
        mock_model = MagicMock()
        judge = JudgeModel(judge_model=mock_model, threshold=0.7)
        evaluator = JudgeEvaluator(judge)

        assert evaluator.judge is judge
        assert evaluator.use_reference is True
        assert evaluator.threshold == 0.7
        assert evaluator.name == "llm_judge"

    def test_evaluate(self):
        """Test JudgeEvaluator.evaluate()."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value=json.dumps({
            "criteria_scores": {"helpfulness": 4},
            "overall_reasoning": "Good",
        }))

        judge = JudgeModel(
            judge_model=mock_model,
            criteria=[JudgeCriterion(name="helpfulness", description="")]
        )
        evaluator = JudgeEvaluator(judge)

        result = evaluator.evaluate(
            prediction="Test response",
            reference="Expected answer",
            prompt="Test prompt",
        )

        assert result.metric_name == "llm_judge"
        assert result.score >= 0


class TestCreateJudge:
    """Tests for create_judge convenience function."""

    def test_create_judge_default(self):
        """Test create_judge with defaults."""
        mock_model = MagicMock()
        judge = create_judge(mock_model)

        assert isinstance(judge, JudgeModel)
        assert judge.criteria == HELPFULNESS_CRITERIA

    def test_create_judge_with_preset(self):
        """Test create_judge with preset criteria."""
        mock_model = MagicMock()
        judge = create_judge(mock_model, criteria_preset="accuracy")

        assert judge.criteria == ACCURACY_CRITERIA

    def test_create_judge_with_custom_criteria(self):
        """Test create_judge with custom criteria."""
        mock_model = MagicMock()
        custom = [JudgeCriterion(name="custom", description="Test")]
        judge = create_judge(mock_model, custom_criteria=custom)

        assert judge.criteria == custom

    def test_create_judge_invalid_preset(self):
        """Test create_judge with invalid preset."""
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="Unknown criteria preset"):
            create_judge(mock_model, criteria_preset="nonexistent")


class TestCreateEvaluatorWithJudge:
    """Tests for create_evaluator with llm_judge type."""

    def test_create_llm_judge_evaluator(self):
        """Test creating llm_judge evaluator."""
        mock_model = MagicMock()
        evaluator = create_evaluator("llm_judge", judge_model=mock_model)

        assert isinstance(evaluator, JudgeEvaluator)

    def test_create_llm_judge_evaluator_missing_model(self):
        """Test creating llm_judge without model raises error."""
        with pytest.raises(ValueError, match="requires 'judge_model'"):
            create_evaluator("llm_judge")


class TestIntegrationWithDummyModel:
    """Integration tests using DummyModel."""

    def test_judge_with_dummy_model(self):
        """Test JudgeModel with actual DummyModel."""
        # Create a DummyModel that returns valid JSON
        class JSONDummyModel(DummyModel):
            def generate(self, prompt, **kwargs):
                return json.dumps({
                    "criteria_scores": {
                        "helpfulness": {"score": 4, "reasoning": "Helpful"},
                        "completeness": {"score": 3, "reasoning": "Partial"},
                        "clarity": {"score": 5, "reasoning": "Clear"},
                    },
                    "overall_reasoning": "Good response",
                    "overall_score": 4.0
                })

            def chat(self, messages, **kwargs):
                return self.generate(messages[-1]["content"])

        dummy_judge = JSONDummyModel()
        judge = JudgeModel(judge_model=dummy_judge)

        result = judge.evaluate(
            prompt="What is Python?",
            response="Python is a programming language.",
        )

        assert isinstance(result, JudgeResult)
        assert result.overall_score > 0
        assert not result.details.get("error")
