"""Tests for evaluation metrics and utilities."""

import pytest

from insideLLMs.evaluation import (
    CompositeEvaluator,
    ContainsEvaluator,
    EvaluationResult,
    ExactMatchEvaluator,
    FuzzyMatchEvaluator,
    MultipleChoiceEvaluator,
    NumericEvaluator,
    SemanticSimilarityEvaluator,
    TokenF1Evaluator,
    bleu_score,
    calculate_classification_metrics,
    contains_match,
    cosine_similarity_bow,
    create_evaluator,
    evaluate_predictions,
    exact_match,
    extract_answer,
    extract_choice,
    extract_number,
    get_ngrams,
    jaccard_similarity,
    levenshtein_distance,
    levenshtein_similarity,
    normalize_text,
    rouge_l,
    token_f1,
)


class TestTextNormalization:
    """Tests for text normalization utilities."""

    def test_normalize_lowercase(self):
        """Test lowercase normalization."""
        assert normalize_text("Hello WORLD") == "hello world"

    def test_normalize_punctuation(self):
        """Test punctuation removal."""
        assert normalize_text("Hello, world!") == "hello world"

    def test_normalize_articles(self):
        """Test article removal."""
        result = normalize_text("The quick a fox an apple")
        assert "the" not in result
        assert "a" not in result.split()
        assert "an" not in result.split()

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        assert normalize_text("  hello   world  ") == "hello world"

    def test_normalize_combined(self):
        """Test combined normalization."""
        text = "  The QUICK, brown FOX!  "
        result = normalize_text(text)
        assert result == "quick brown fox"


class TestAnswerExtraction:
    """Tests for answer extraction utilities."""

    def test_extract_answer_explicit(self):
        """Test extraction with explicit answer marker."""
        text = "Let me think... The answer is 42."
        assert extract_answer(text) == "42"

    def test_extract_answer_therefore(self):
        """Test extraction with 'therefore'."""
        text = "We can see that x + y = 10. Therefore, the sum is 10."
        assert "10" in extract_answer(text)

    def test_extract_answer_equals(self):
        """Test extraction with equals sign."""
        text = "Calculating: 2 + 2 = 4"
        assert extract_answer(text) == "4"

    def test_extract_answer_fallback(self):
        """Test fallback to last line."""
        text = "First line\nSecond line\nFinal answer"
        assert extract_answer(text) == "Final answer"

    def test_extract_number_integer(self):
        """Test extracting integer."""
        assert extract_number("The result is 42") == 42.0

    def test_extract_number_decimal(self):
        """Test extracting decimal."""
        assert extract_number("Pi is approximately 3.14159") == 3.14159

    def test_extract_number_negative(self):
        """Test extracting negative number."""
        assert extract_number("Temperature: -5 degrees") == -5.0

    def test_extract_number_fraction(self):
        """Test extracting fraction."""
        assert extract_number("One half is 1/2") == 0.5

    def test_extract_number_none(self):
        """Test when no number found."""
        assert extract_number("No numbers here") is None

    def test_extract_choice_letter(self):
        """Test extracting choice letter."""
        assert extract_choice("I think the answer is B", ["A", "B", "C", "D"]) == "B"

    def test_extract_choice_with_parentheses(self):
        """Test extracting choice with parentheses."""
        assert extract_choice("The correct option is (C)", ["A", "B", "C", "D"]) == "C"

    def test_extract_choice_explicit(self):
        """Test extracting explicit choice."""
        assert extract_choice("Answer: A", ["A", "B", "C", "D"]) == "A"


class TestSimilarityMetrics:
    """Tests for similarity metrics."""

    def test_exact_match_identical(self):
        """Test exact match with identical strings."""
        assert exact_match("hello world", "hello world") == 1.0

    def test_exact_match_different(self):
        """Test exact match with different strings."""
        assert exact_match("hello", "world") == 0.0

    def test_exact_match_normalize(self):
        """Test exact match with normalization."""
        assert exact_match("Hello, World!", "hello world") == 1.0

    def test_contains_match_yes(self):
        """Test contains match when reference is in prediction."""
        assert contains_match("The answer is 42 because...", "42") == 1.0

    def test_contains_match_no(self):
        """Test contains match when reference is not in prediction."""
        assert contains_match("The answer is 42", "43") == 0.0

    def test_levenshtein_distance_identical(self):
        """Test Levenshtein distance for identical strings."""
        assert levenshtein_distance("hello", "hello") == 0

    def test_levenshtein_distance_one_edit(self):
        """Test Levenshtein distance for one edit."""
        assert levenshtein_distance("hello", "hallo") == 1

    def test_levenshtein_distance_empty(self):
        """Test Levenshtein distance with empty string."""
        assert levenshtein_distance("hello", "") == 5

    def test_levenshtein_similarity_identical(self):
        """Test Levenshtein similarity for identical strings."""
        assert levenshtein_similarity("hello", "hello") == 1.0

    def test_levenshtein_similarity_partial(self):
        """Test Levenshtein similarity for similar strings."""
        sim = levenshtein_similarity("hello", "hallo")
        assert 0.7 < sim < 1.0

    def test_jaccard_similarity_identical(self):
        """Test Jaccard similarity for identical strings."""
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_jaccard_similarity_partial(self):
        """Test Jaccard similarity for overlapping strings."""
        sim = jaccard_similarity("hello world", "hello there")
        assert 0.3 < sim < 0.7

    def test_jaccard_similarity_no_overlap(self):
        """Test Jaccard similarity for non-overlapping strings."""
        assert jaccard_similarity("hello", "world") == 0.0

    def test_cosine_similarity_identical(self):
        """Test cosine similarity for identical strings."""
        assert cosine_similarity_bow("hello world", "hello world") == 1.0

    def test_cosine_similarity_partial(self):
        """Test cosine similarity for similar strings."""
        sim = cosine_similarity_bow("hello world today", "hello world tomorrow")
        assert 0.5 < sim < 1.0

    def test_token_f1_identical(self):
        """Test token F1 for identical strings."""
        assert token_f1("hello world", "hello world") == 1.0

    def test_token_f1_partial(self):
        """Test token F1 for partial overlap."""
        f1 = token_f1("hello world foo", "hello world bar")
        assert 0.5 < f1 < 1.0

    def test_token_f1_no_overlap(self):
        """Test token F1 for no overlap."""
        assert token_f1("hello", "world") == 0.0


class TestNgramMetrics:
    """Tests for n-gram based metrics."""

    def test_get_ngrams(self):
        """Test n-gram extraction."""
        ngrams = get_ngrams("a b c d", 2)
        assert len(ngrams) == 3
        assert ("a", "b") in ngrams

    def test_bleu_score_identical(self):
        """Test BLEU for identical strings."""
        bleu = bleu_score("hello world test", "hello world test")
        assert bleu > 0.9

    def test_bleu_score_partial(self):
        """Test BLEU for partial match."""
        bleu = bleu_score("hello world foo bar", "hello world baz qux")
        assert 0.0 < bleu < 1.0

    def test_bleu_score_empty(self):
        """Test BLEU with empty prediction."""
        assert bleu_score("", "hello world") == 0.0

    def test_rouge_l_identical(self):
        """Test ROUGE-L for identical strings."""
        rouge = rouge_l("hello world test", "hello world test")
        assert rouge == 1.0

    def test_rouge_l_partial(self):
        """Test ROUGE-L for partial match."""
        rouge = rouge_l("a b c d e", "a b x d e")
        assert 0.5 < rouge < 1.0

    def test_rouge_l_no_match(self):
        """Test ROUGE-L with no common subsequence."""
        rouge = rouge_l("a b c", "x y z")
        assert rouge == 0.0


class TestClassificationMetrics:
    """Tests for classification metrics."""

    def test_perfect_classification(self):
        """Test metrics for perfect classification."""
        preds = ["A", "B", "A", "B"]
        refs = ["A", "B", "A", "B"]
        metrics = calculate_classification_metrics(preds, refs)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_imperfect_classification(self):
        """Test metrics for imperfect classification."""
        preds = ["A", "A", "B", "B"]
        refs = ["A", "B", "A", "B"]
        metrics = calculate_classification_metrics(preds, refs)

        assert metrics["accuracy"] == 0.5

    def test_empty_predictions(self):
        """Test with empty lists."""
        metrics = calculate_classification_metrics([], [])
        assert metrics["accuracy"] == 0.0


class TestEvaluators:
    """Tests for evaluator classes."""

    def test_exact_match_evaluator(self):
        """Test ExactMatchEvaluator."""
        evaluator = ExactMatchEvaluator()
        result = evaluator.evaluate("hello world", "hello world")

        assert result.score == 1.0
        assert result.passed

    def test_contains_evaluator(self):
        """Test ContainsEvaluator."""
        evaluator = ContainsEvaluator()
        result = evaluator.evaluate("The answer is 42", "42")

        assert result.score == 1.0
        assert result.passed

    def test_fuzzy_match_evaluator(self):
        """Test FuzzyMatchEvaluator."""
        evaluator = FuzzyMatchEvaluator(threshold=0.8)
        result = evaluator.evaluate("hello world", "hello warld")

        assert result.score > 0.8
        assert result.passed

    def test_token_f1_evaluator(self):
        """Test TokenF1Evaluator."""
        evaluator = TokenF1Evaluator(threshold=0.5)
        result = evaluator.evaluate("hello world foo", "hello world bar")

        assert result.score > 0.5
        assert result.passed

    def test_semantic_similarity_evaluator(self):
        """Test SemanticSimilarityEvaluator."""
        evaluator = SemanticSimilarityEvaluator(threshold=0.5)
        result = evaluator.evaluate("hello world", "hello world")

        assert result.score > 0.9
        assert result.passed
        assert "component_scores" in result.details

    def test_numeric_evaluator_exact(self):
        """Test NumericEvaluator with exact match."""
        evaluator = NumericEvaluator(tolerance=0.01)
        result = evaluator.evaluate("The answer is 42", "42")

        assert result.score == 1.0
        assert result.passed

    def test_numeric_evaluator_close(self):
        """Test NumericEvaluator with close match."""
        evaluator = NumericEvaluator(tolerance=0.1, relative=True)
        result = evaluator.evaluate("I got 10.05", "10")

        assert result.passed

    def test_numeric_evaluator_no_numbers(self):
        """Test NumericEvaluator with no numbers."""
        evaluator = NumericEvaluator()
        result = evaluator.evaluate("no numbers", "also no numbers")

        assert result.score == 0.0
        assert not result.passed

    def test_multiple_choice_evaluator(self):
        """Test MultipleChoiceEvaluator."""
        evaluator = MultipleChoiceEvaluator(choices=["A", "B", "C", "D"])
        result = evaluator.evaluate("I think B is correct", "B")

        assert result.score == 1.0
        assert result.passed

    def test_multiple_choice_evaluator_wrong(self):
        """Test MultipleChoiceEvaluator with wrong answer."""
        evaluator = MultipleChoiceEvaluator()
        result = evaluator.evaluate("The answer is A", "B")

        assert result.score == 0.0
        assert not result.passed

    def test_composite_evaluator(self):
        """Test CompositeEvaluator."""
        evaluator = CompositeEvaluator(
            evaluators=[ExactMatchEvaluator(), ContainsEvaluator()],
            weights=[0.5, 0.5],
        )
        result = evaluator.evaluate("hello world", "hello world")

        assert result.score == 1.0
        assert result.passed

    def test_evaluator_batch(self):
        """Test batch evaluation."""
        evaluator = ExactMatchEvaluator()
        results = evaluator.evaluate_batch(
            ["a", "b", "c"],
            ["a", "x", "c"],
        )

        assert len(results) == 3
        assert results[0].passed
        assert not results[1].passed
        assert results[2].passed

    def test_evaluator_aggregate(self):
        """Test result aggregation."""
        evaluator = ExactMatchEvaluator()
        results = evaluator.evaluate_batch(
            ["a", "b", "c", "d"],
            ["a", "x", "c", "y"],
        )
        aggregated = evaluator.aggregate_results(results)

        assert aggregated["mean_score"] == 0.5
        assert aggregated["pass_rate"] == 0.5


class TestFactoryAndConvenience:
    """Tests for factory and convenience functions."""

    def test_create_evaluator(self):
        """Test evaluator factory."""
        evaluator = create_evaluator("exact_match")
        assert isinstance(evaluator, ExactMatchEvaluator)

        evaluator = create_evaluator("fuzzy", threshold=0.9)
        assert isinstance(evaluator, FuzzyMatchEvaluator)

    def test_create_evaluator_unknown(self):
        """Test factory with unknown type."""
        with pytest.raises(ValueError):
            create_evaluator("unknown_type")

    def test_evaluate_predictions(self):
        """Test evaluate_predictions function."""
        result = evaluate_predictions(
            predictions=["hello world", "foo bar"],
            references=["hello world", "foo baz"],
            metrics=["exact_match", "token_f1"],
        )

        assert "results" in result
        assert "aggregated" in result
        assert result["n_samples"] == 2
        assert "exact_match" in result["aggregated"]


class TestEvaluationResultDataclass:
    """Tests for EvaluationResult dataclass."""

    def test_repr(self):
        """Test string representation."""
        result = EvaluationResult(
            score=0.85,
            passed=True,
            metric_name="test",
        )
        repr_str = repr(result)

        assert "0.85" in repr_str
        assert "PASS" in repr_str
        assert "test" in repr_str

    def test_fail_repr(self):
        """Test representation for failed result."""
        result = EvaluationResult(
            score=0.3,
            passed=False,
            metric_name="test",
        )
        repr_str = repr(result)

        assert "FAIL" in repr_str
