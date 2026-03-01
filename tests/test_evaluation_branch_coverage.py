"""Additional branch coverage for analysis.evaluation."""

from __future__ import annotations

import pytest

from insideLLMs.evaluation import (
    CompositeEvaluator,
    EvaluationResult,
    Evaluator,
    ExactMatchEvaluator,
    JudgeCriterion,
    JudgeModel,
    MultiMetricResult,
    MultipleChoiceEvaluator,
    NumericEvaluator,
    TokenF1Evaluator,
    bleu_score,
    calculate_classification_metrics,
    cosine_similarity_bow,
    evaluate_predictions,
    extract_answer,
    extract_choice,
    get_ngrams,
    jaccard_similarity,
    levenshtein_similarity,
    rouge_l,
    token_f1,
)


class _GenerateOnlyModel:
    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt: str, **kwargs) -> str:
        _ = prompt, kwargs
        return self._response


def test_multi_metric_result_get_scores():
    r1 = EvaluationResult(score=1.0, passed=True, metric_name="m1")
    r2 = EvaluationResult(score=0.25, passed=False, metric_name="m2")
    multi = MultiMetricResult(
        results={"m1": r1, "m2": r2}, overall_score=0.625, overall_passed=False
    )
    assert multi["m1"] is r1
    assert multi.get_scores() == {"m1": 1.0, "m2": 0.25}


def test_extract_answer_and_choice_fallback_paths():
    assert extract_answer("   \n   ") == ""
    assert extract_choice("No option selected", ["A", "B", "C"]) is None


def test_similarity_metric_empty_and_edge_paths():
    assert levenshtein_similarity("", "") == 1.0
    assert levenshtein_similarity("hello", "") == 0.0
    assert jaccard_similarity("", "") == 1.0
    assert jaccard_similarity("hello", "") == 0.0
    assert cosine_similarity_bow("", "hello") == 0.0
    assert token_f1("", "") == 1.0
    assert token_f1("hello", "") == 0.0
    assert get_ngrams("one two", 3) == []


def test_bleu_and_rouge_branch_edges():
    # Brevity penalty path.
    assert 0.0 <= bleu_score("a", "a b c") < 1.0
    # Geometric mean contains -inf when one n-gram precision is zero.
    assert bleu_score("a b", "a c", smoothing=False) == 0.0
    assert rouge_l("", "a b c") == 0.0
    assert rouge_l("a b c", "x y z") == 0.0


def test_classification_metrics_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        calculate_classification_metrics(["A"], ["A", "B"])


def test_evaluator_base_methods_edge_paths():
    # Exercise abstract method body directly (pass branch).
    assert Evaluator.evaluate(object(), "prediction", "reference") is None  # type: ignore[arg-type]

    evaluator = ExactMatchEvaluator()
    assert evaluator.aggregate_results([]) == {"mean_score": 0.0, "pass_rate": 0.0}


def test_numeric_and_multiple_choice_unhappy_paths():
    numeric = NumericEvaluator(tolerance=0.1, relative=False)
    numeric_result = numeric.evaluate("95", "100")
    assert numeric_result.details["difference"] == 5.0
    assert not numeric_result.passed

    mc = MultipleChoiceEvaluator(choices=["A", "B", "C"])
    bad = mc.evaluate("I refuse to answer", "A")
    assert bad.score == 0.0
    assert "Could not extract choice" in bad.details["error"]


def test_composite_evaluator_require_all_branch():
    evaluator = CompositeEvaluator(
        evaluators=[ExactMatchEvaluator(), TokenF1Evaluator(threshold=0.8)],
        weights=[0.5, 0.5],
        require_all=True,
    )
    result = evaluator.evaluate("hello world", "hello there")
    assert result.score < 1.0
    assert not result.passed


def test_judge_model_parse_score_compare_branches():
    criterion = JudgeCriterion(name="quality", description="Quality", weight=0.0)
    judge = JudgeModel(
        judge_model=_GenerateOnlyModel('{"overall_winner":"A"}'),
        criteria=[criterion],  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="Failed to parse judge response as JSON"):
        judge._parse_judge_response("```json\n{bad json}\n```")

    assert judge._calculate_overall_score({"quality": 3.0}) == 0.0

    ok = judge.compare("p", "a", "b")
    assert ok["winner"] == "A"

    bad_judge = JudgeModel(judge_model=_GenerateOnlyModel("nonsense"), criteria=[criterion])  # type: ignore[arg-type]
    failed = bad_judge.compare("p", "a", "b")
    assert failed["winner"] == "error"
    assert "Comparison failed" in failed["reasoning"]


def test_evaluate_predictions_additional_metric_branches():
    evaluated = evaluate_predictions(
        predictions=["the cat sat", "hello there"],
        references=["the cat is here", "hello world"],
        metrics=["bleu", "rouge_l"],
    )
    assert "bleu" in evaluated["aggregated"]
    assert "rouge_l" in evaluated["aggregated"]
