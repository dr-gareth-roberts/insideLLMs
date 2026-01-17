"""Evaluation metrics and utilities for LLM outputs.

This module provides comprehensive evaluation capabilities including:
- Text similarity metrics (exact match, fuzzy match, semantic similarity)
- Answer extraction and normalization
- Classification metrics (accuracy, precision, recall, F1)
- Generation quality metrics (BLEU, ROUGE approximations)
- Custom evaluator framework
"""

import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class EvaluationResult:
    """Result from an evaluation."""

    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    metric_name: str = ""

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"EvaluationResult({self.metric_name}: {self.score:.4f} [{status}])"


@dataclass
class MultiMetricResult:
    """Result from multiple evaluation metrics."""

    results: Dict[str, EvaluationResult]
    overall_score: float
    overall_passed: bool

    def __getitem__(self, key: str) -> EvaluationResult:
        return self.results[key]

    def get_scores(self) -> Dict[str, float]:
        """Get all scores as a dictionary."""
        return {name: r.score for name, r in self.results.items()}


# Text Normalization Utilities


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_articles: bool = True,
    strip_whitespace: bool = True,
) -> str:
    """Normalize text for comparison.

    Args:
        text: Input text.
        lowercase: Convert to lowercase.
        remove_punctuation: Remove punctuation marks.
        remove_articles: Remove articles (a, an, the).
        strip_whitespace: Collapse and strip whitespace.

    Returns:
        Normalized text.
    """
    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    if remove_articles:
        text = re.sub(r"\b(a|an|the)\b", " ", text)

    if strip_whitespace:
        text = " ".join(text.split())

    return text.strip()


def extract_answer(
    text: str,
    patterns: Optional[List[str]] = None,
) -> str:
    """Extract the final answer from a response.

    Args:
        text: Full response text.
        patterns: Optional custom patterns to try.

    Returns:
        Extracted answer or original text if no pattern matches.
    """
    if patterns is None:
        patterns = [
            r"(?:the answer is|answer:|final answer:?)\s*(.+?)(?:\.|$)",
            r"(?:therefore|thus|so|hence),?\s*(.+?)(?:\.|$)",
            r"= (.+?)(?:\.|$)",
            r"(?:is|are|equals?)\s+(.+?)(?:\.|$)",
        ]

    text_lower = text.lower()

    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # If no pattern matches, return the last sentence or line
    lines = text.strip().split("\n")
    last_line = lines[-1].strip()
    if last_line:
        return last_line

    return text.strip()


def extract_number(text: str) -> Optional[float]:
    """Extract a numeric value from text.

    Args:
        text: Text containing a number.

    Returns:
        Extracted number or None.
    """
    # Try fractions first (e.g., "1/2")
    fraction_match = re.search(r"[-+]?\d+/\d+", text)
    if fraction_match:
        num_str = fraction_match.group()
        try:
            num, denom = num_str.split("/")
            return float(num) / float(denom)
        except (ValueError, ZeroDivisionError):
            pass

    # Try standard decimal numbers
    decimal_match = re.search(r"[-+]?\d*\.?\d+", text)
    if decimal_match:
        num_str = decimal_match.group()
        try:
            return float(num_str)
        except ValueError:
            pass

    return None


def extract_choice(text: str, choices: List[str]) -> Optional[str]:
    """Extract a multiple-choice answer.

    Args:
        text: Response text.
        choices: List of valid choices (e.g., ["A", "B", "C", "D"]).

    Returns:
        Matched choice or None.
    """
    text_upper = text.upper()

    # Look for explicit choice patterns
    for choice in choices:
        patterns = [
            rf"\b{choice}\b",  # Just the letter
            rf"(?:answer|choice|option)(?:\s+is)?\s*[:\s]*{choice}\b",
            rf"\({choice}\)",  # (A), (B), etc.
        ]
        for pattern in patterns:
            if re.search(pattern, text_upper):
                return choice

    return None


# Similarity Metrics


def exact_match(prediction: str, reference: str, normalize: bool = True) -> float:
    """Check for exact match between prediction and reference.

    Args:
        prediction: Model prediction.
        reference: Reference answer.
        normalize: Whether to normalize before comparison.

    Returns:
        1.0 if match, 0.0 otherwise.
    """
    if normalize:
        prediction = normalize_text(prediction)
        reference = normalize_text(reference)

    return 1.0 if prediction == reference else 0.0


def contains_match(prediction: str, reference: str, normalize: bool = True) -> float:
    """Check if prediction contains the reference.

    Args:
        prediction: Model prediction.
        reference: Reference answer.
        normalize: Whether to normalize before comparison.

    Returns:
        1.0 if reference is contained in prediction, 0.0 otherwise.
    """
    if normalize:
        prediction = normalize_text(prediction)
        reference = normalize_text(reference)

    return 1.0 if reference in prediction else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Edit distance.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str, normalize: bool = True) -> float:
    """Calculate normalized Levenshtein similarity.

    Args:
        s1: First string.
        s2: Second string.
        normalize: Whether to normalize strings first.

    Returns:
        Similarity score between 0 and 1.
    """
    if normalize:
        s1 = normalize_text(s1)
        s2 = normalize_text(s2)

    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)


def jaccard_similarity(s1: str, s2: str, normalize: bool = True) -> float:
    """Calculate Jaccard similarity between word sets.

    Args:
        s1: First string.
        s2: Second string.
        normalize: Whether to normalize strings first.

    Returns:
        Jaccard similarity between 0 and 1.
    """
    if normalize:
        s1 = normalize_text(s1)
        s2 = normalize_text(s2)

    words1 = set(s1.split())
    words2 = set(s2.split())

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union


def cosine_similarity_bow(s1: str, s2: str, normalize: bool = True) -> float:
    """Calculate cosine similarity using bag-of-words.

    Args:
        s1: First string.
        s2: Second string.
        normalize: Whether to normalize strings first.

    Returns:
        Cosine similarity between 0 and 1.
    """
    if normalize:
        s1 = normalize_text(s1)
        s2 = normalize_text(s2)

    words1 = Counter(s1.split())
    words2 = Counter(s2.split())

    if not words1 or not words2:
        return 0.0

    # Get all unique words
    all_words = set(words1.keys()) | set(words2.keys())

    # Calculate dot product and magnitudes
    dot_product = sum(words1.get(w, 0) * words2.get(w, 0) for w in all_words)
    magnitude1 = sum(v ** 2 for v in words1.values()) ** 0.5
    magnitude2 = sum(v ** 2 for v in words2.values()) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    similarity = dot_product / (magnitude1 * magnitude2)
    # Round to handle floating-point precision issues
    return round(similarity, 10)


def token_f1(prediction: str, reference: str, normalize: bool = True) -> float:
    """Calculate token-level F1 score.

    Args:
        prediction: Model prediction.
        reference: Reference answer.
        normalize: Whether to normalize strings first.

    Returns:
        F1 score between 0 and 1.
    """
    if normalize:
        prediction = normalize_text(prediction)
        reference = normalize_text(reference)

    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


# N-gram Based Metrics


def get_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from text.

    Args:
        text: Input text.
        n: N-gram size.

    Returns:
        List of n-gram tuples.
    """
    words = text.split()
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def bleu_score(
    prediction: str,
    reference: str,
    max_n: int = 4,
    normalize: bool = True,
    smoothing: bool = True,
) -> float:
    """Calculate approximate BLEU score.

    Uses add-1 smoothing by default to handle partial matches.

    Args:
        prediction: Model prediction.
        reference: Reference text.
        max_n: Maximum n-gram size.
        normalize: Whether to normalize strings first.
        smoothing: Whether to apply add-1 smoothing for zero counts.

    Returns:
        BLEU score between 0 and 1.
    """
    import math

    if normalize:
        prediction = normalize_text(prediction)
        reference = normalize_text(reference)

    pred_words = prediction.split()
    ref_words = reference.split()

    if not pred_words:
        return 0.0

    # Brevity penalty
    bp = 1.0
    if len(pred_words) < len(ref_words):
        bp = math.exp(1 - len(ref_words) / len(pred_words))

    # Calculate precision for each n-gram size
    precisions = []
    for n in range(1, min(max_n + 1, len(pred_words) + 1)):
        pred_ngrams = Counter(get_ngrams(prediction, n))
        ref_ngrams = Counter(get_ngrams(reference, n))

        clipped = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = sum(pred_ngrams.values())

        if total > 0:
            precision = clipped / total
            # Apply smoothing for zero precision (add-1 smoothing)
            if precision == 0 and smoothing and n > 1:
                precision = 1 / (total + 1)
            precisions.append(precision)
        else:
            precisions.append(0.0)

    if not precisions or all(p == 0 for p in precisions):
        return 0.0

    # Geometric mean of precisions
    log_precisions = [math.log(p) if p > 0 else -float("inf") for p in precisions]
    avg_log_precision = sum(log_precisions) / len(log_precisions)

    if avg_log_precision == -float("inf"):
        return 0.0

    return bp * math.exp(avg_log_precision)


def rouge_l(prediction: str, reference: str, normalize: bool = True) -> float:
    """Calculate ROUGE-L score (longest common subsequence).

    Args:
        prediction: Model prediction.
        reference: Reference text.
        normalize: Whether to normalize strings first.

    Returns:
        ROUGE-L F1 score between 0 and 1.
    """
    if normalize:
        prediction = normalize_text(prediction)
        reference = normalize_text(reference)

    pred_words = prediction.split()
    ref_words = reference.split()

    if not pred_words or not ref_words:
        return 0.0

    # Find LCS length using dynamic programming
    m, n = len(pred_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == ref_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / m
    recall = lcs_length / n

    return 2 * precision * recall / (precision + recall)


# Classification Metrics


def calculate_classification_metrics(
    predictions: List[Any],
    references: List[Any],
    labels: Optional[List[Any]] = None,
) -> Dict[str, float]:
    """Calculate classification metrics.

    Args:
        predictions: List of predicted labels.
        references: List of true labels.
        labels: Optional list of all possible labels.

    Returns:
        Dictionary with accuracy, precision, recall, F1.
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    if not predictions:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    if labels is None:
        labels = list(set(predictions) | set(references))

    # Calculate per-class metrics
    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}

    for pred, ref in zip(predictions, references):
        if pred == ref:
            tp[pred] = tp.get(pred, 0) + 1
        else:
            fp[pred] = fp.get(pred, 0) + 1
            fn[ref] = fn.get(ref, 0) + 1

    # Calculate accuracy
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    accuracy = correct / len(predictions)

    # Macro-averaged precision, recall, F1
    precisions = []
    recalls = []
    f1s = []

    for label in labels:
        if tp[label] + fp[label] > 0:
            p = tp[label] / (tp[label] + fp[label])
        else:
            p = 0.0

        if tp[label] + fn[label] > 0:
            r = tp[label] / (tp[label] + fn[label])
        else:
            r = 0.0

        if p + r > 0:
            f = 2 * p * r / (p + r)
        else:
            f = 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "accuracy": accuracy,
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }


# Evaluator Framework


class Evaluator(ABC):
    """Base class for evaluators."""

    name: str = "base"
    threshold: float = 0.5

    @abstractmethod
    def evaluate(
        self,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a single prediction against a reference.

        Args:
            prediction: Model prediction.
            reference: Reference answer.
            **kwargs: Additional arguments.

        Returns:
            EvaluationResult with score and pass/fail.
        """
        pass

    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs: Any,
    ) -> List[EvaluationResult]:
        """Evaluate multiple predictions.

        Args:
            predictions: List of model predictions.
            references: List of reference answers.
            **kwargs: Additional arguments.

        Returns:
            List of EvaluationResults.
        """
        return [
            self.evaluate(p, r, **kwargs)
            for p, r in zip(predictions, references)
        ]

    def aggregate_results(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, float]:
        """Aggregate evaluation results.

        Args:
            results: List of evaluation results.

        Returns:
            Dictionary with aggregated metrics.
        """
        if not results:
            return {"mean_score": 0.0, "pass_rate": 0.0}

        scores = [r.score for r in results]
        passed = [r.passed for r in results]

        return {
            "mean_score": sum(scores) / len(scores),
            "pass_rate": sum(passed) / len(passed),
            "min_score": min(scores),
            "max_score": max(scores),
        }


class ExactMatchEvaluator(Evaluator):
    """Evaluator for exact match."""

    name = "exact_match"

    def __init__(self, normalize: bool = True, threshold: float = 1.0):
        self.normalize = normalize
        self.threshold = threshold

    def evaluate(
        self,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> EvaluationResult:
        score = exact_match(prediction, reference, self.normalize)
        return EvaluationResult(
            score=score,
            passed=score >= self.threshold,
            metric_name=self.name,
        )


class ContainsEvaluator(Evaluator):
    """Evaluator that checks if prediction contains reference."""

    name = "contains"

    def __init__(self, normalize: bool = True, threshold: float = 1.0):
        self.normalize = normalize
        self.threshold = threshold

    def evaluate(
        self,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> EvaluationResult:
        score = contains_match(prediction, reference, self.normalize)
        return EvaluationResult(
            score=score,
            passed=score >= self.threshold,
            metric_name=self.name,
        )


class FuzzyMatchEvaluator(Evaluator):
    """Evaluator using Levenshtein similarity."""

    name = "fuzzy_match"

    def __init__(self, normalize: bool = True, threshold: float = 0.8):
        self.normalize = normalize
        self.threshold = threshold

    def evaluate(
        self,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> EvaluationResult:
        score = levenshtein_similarity(prediction, reference, self.normalize)
        return EvaluationResult(
            score=score,
            passed=score >= self.threshold,
            metric_name=self.name,
            details={"threshold": self.threshold},
        )


class TokenF1Evaluator(Evaluator):
    """Evaluator using token-level F1."""

    name = "token_f1"

    def __init__(self, normalize: bool = True, threshold: float = 0.5):
        self.normalize = normalize
        self.threshold = threshold

    def evaluate(
        self,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> EvaluationResult:
        score = token_f1(prediction, reference, self.normalize)
        return EvaluationResult(
            score=score,
            passed=score >= self.threshold,
            metric_name=self.name,
        )


class SemanticSimilarityEvaluator(Evaluator):
    """Evaluator using multiple similarity metrics."""

    name = "semantic_similarity"

    def __init__(
        self,
        normalize: bool = True,
        threshold: float = 0.6,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.normalize = normalize
        self.threshold = threshold
        self.weights = weights or {
            "jaccard": 0.3,
            "cosine": 0.4,
            "token_f1": 0.3,
        }

    def evaluate(
        self,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> EvaluationResult:
        scores = {
            "jaccard": jaccard_similarity(prediction, reference, self.normalize),
            "cosine": cosine_similarity_bow(prediction, reference, self.normalize),
            "token_f1": token_f1(prediction, reference, self.normalize),
        }

        weighted_score = sum(
            scores.get(metric, 0) * weight
            for metric, weight in self.weights.items()
        )

        return EvaluationResult(
            score=weighted_score,
            passed=weighted_score >= self.threshold,
            metric_name=self.name,
            details={"component_scores": scores},
        )


class NumericEvaluator(Evaluator):
    """Evaluator for numeric answers."""

    name = "numeric"

    def __init__(self, tolerance: float = 0.01, relative: bool = True):
        self.tolerance = tolerance
        self.relative = relative
        self.threshold = 1.0 - tolerance

    def evaluate(
        self,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> EvaluationResult:
        pred_num = extract_number(prediction)
        ref_num = extract_number(reference)

        if pred_num is None or ref_num is None:
            return EvaluationResult(
                score=0.0,
                passed=False,
                metric_name=self.name,
                details={"error": "Could not extract numbers"},
            )

        if self.relative and ref_num != 0:
            diff = abs(pred_num - ref_num) / abs(ref_num)
        else:
            diff = abs(pred_num - ref_num)

        score = max(0.0, 1.0 - diff / self.tolerance) if diff <= self.tolerance else 0.0

        return EvaluationResult(
            score=score,
            passed=diff <= self.tolerance,
            metric_name=self.name,
            details={
                "predicted": pred_num,
                "reference": ref_num,
                "difference": diff,
            },
        )


class MultipleChoiceEvaluator(Evaluator):
    """Evaluator for multiple choice answers."""

    name = "multiple_choice"

    def __init__(self, choices: List[str] = ["A", "B", "C", "D"]):
        self.choices = choices
        self.threshold = 1.0

    def evaluate(
        self,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> EvaluationResult:
        pred_choice = extract_choice(prediction, self.choices)
        ref_choice = reference.upper().strip()

        if pred_choice is None:
            return EvaluationResult(
                score=0.0,
                passed=False,
                metric_name=self.name,
                details={"error": "Could not extract choice"},
            )

        score = 1.0 if pred_choice == ref_choice else 0.0

        return EvaluationResult(
            score=score,
            passed=score >= self.threshold,
            metric_name=self.name,
            details={"predicted": pred_choice, "reference": ref_choice},
        )


class CompositeEvaluator(Evaluator):
    """Evaluator that combines multiple evaluators."""

    name = "composite"

    def __init__(
        self,
        evaluators: List[Evaluator],
        weights: Optional[List[float]] = None,
        require_all: bool = False,
    ):
        self.evaluators = evaluators
        self.weights = weights or [1.0 / len(evaluators)] * len(evaluators)
        self.require_all = require_all
        self.threshold = 0.5

    def evaluate(
        self,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> EvaluationResult:
        results = {}
        weighted_sum = 0.0

        for evaluator, weight in zip(self.evaluators, self.weights):
            result = evaluator.evaluate(prediction, reference, **kwargs)
            results[evaluator.name] = result
            weighted_sum += result.score * weight

        if self.require_all:
            passed = all(r.passed for r in results.values())
        else:
            passed = weighted_sum >= self.threshold

        return EvaluationResult(
            score=weighted_sum,
            passed=passed,
            metric_name=self.name,
            details={"component_results": {k: v.score for k, v in results.items()}},
        )


# Convenience Functions


def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    evaluator: Optional[Evaluator] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate predictions using specified metrics.

    Args:
        predictions: List of model predictions.
        references: List of reference answers.
        evaluator: Optional custom evaluator.
        metrics: List of metric names to compute.

    Returns:
        Dictionary with evaluation results.
    """
    if evaluator is None:
        evaluator = SemanticSimilarityEvaluator()

    results = evaluator.evaluate_batch(predictions, references)
    aggregated = evaluator.aggregate_results(results)

    if metrics:
        additional = {}
        for metric in metrics:
            if metric == "exact_match":
                scores = [exact_match(p, r) for p, r in zip(predictions, references)]
                additional["exact_match"] = sum(scores) / len(scores)
            elif metric == "token_f1":
                scores = [token_f1(p, r) for p, r in zip(predictions, references)]
                additional["token_f1"] = sum(scores) / len(scores)
            elif metric == "bleu":
                scores = [bleu_score(p, r) for p, r in zip(predictions, references)]
                additional["bleu"] = sum(scores) / len(scores)
            elif metric == "rouge_l":
                scores = [rouge_l(p, r) for p, r in zip(predictions, references)]
                additional["rouge_l"] = sum(scores) / len(scores)

        aggregated.update(additional)

    return {
        "results": results,
        "aggregated": aggregated,
        "n_samples": len(predictions),
    }


def create_evaluator(
    evaluator_type: str,
    **kwargs: Any,
) -> Evaluator:
    """Factory function to create evaluators.

    Args:
        evaluator_type: Type of evaluator to create.
        **kwargs: Arguments for the evaluator.

    Returns:
        Evaluator instance.
    """
    evaluators = {
        "exact_match": ExactMatchEvaluator,
        "contains": ContainsEvaluator,
        "fuzzy": FuzzyMatchEvaluator,
        "token_f1": TokenF1Evaluator,
        "semantic": SemanticSimilarityEvaluator,
        "numeric": NumericEvaluator,
        "multiple_choice": MultipleChoiceEvaluator,
    }

    if evaluator_type not in evaluators:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")

    return evaluators[evaluator_type](**kwargs)
