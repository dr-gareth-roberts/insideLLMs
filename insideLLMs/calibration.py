"""
Model calibration and confidence estimation utilities.

Provides tools for:
- Measuring calibration (reliability diagrams, ECE, MCE)
- Confidence score analysis
- Temperature scaling
- Platt scaling
- Histogram binning calibration
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class CalibrationMethod(Enum):
    """Methods for calibration."""

    TEMPERATURE_SCALING = "temperature_scaling"
    PLATT_SCALING = "platt_scaling"
    HISTOGRAM_BINNING = "histogram_binning"
    ISOTONIC_REGRESSION = "isotonic_regression"
    BETA_CALIBRATION = "beta_calibration"


class ConfidenceSource(Enum):
    """Sources of confidence scores."""

    LOGPROBS = "logprobs"
    SELF_REPORTED = "self_reported"
    CONSISTENCY = "consistency"
    ENTROPY = "entropy"
    VERBALIZED = "verbalized"


@dataclass
class CalibrationBin:
    """A single bin in calibration analysis."""

    bin_start: float
    bin_end: float
    confidence_sum: float
    accuracy_sum: float
    count: int

    @property
    def avg_confidence(self) -> float:
        """Average confidence in this bin."""
        return self.confidence_sum / self.count if self.count > 0 else 0.0

    @property
    def avg_accuracy(self) -> float:
        """Average accuracy in this bin."""
        return self.accuracy_sum / self.count if self.count > 0 else 0.0

    @property
    def gap(self) -> float:
        """Calibration gap (|confidence - accuracy|)."""
        return abs(self.avg_confidence - self.avg_accuracy)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bin_range": [self.bin_start, self.bin_end],
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_accuracy": round(self.avg_accuracy, 4),
            "count": self.count,
            "gap": round(self.gap, 4),
        }


@dataclass
class CalibrationMetrics:
    """Calibration metrics for a model."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    ace: float  # Average Calibration Error
    overconfidence: float  # How much model overestimates
    underconfidence: float  # How much model underestimates
    brier_score: float  # Brier score (mean squared error)
    log_loss: float  # Log loss / cross-entropy
    bins: List[CalibrationBin] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "ace": round(self.ace, 4),
            "overconfidence": round(self.overconfidence, 4),
            "underconfidence": round(self.underconfidence, 4),
            "brier_score": round(self.brier_score, 4),
            "log_loss": round(self.log_loss, 4),
            "bins": [b.to_dict() for b in self.bins],
        }

    @property
    def is_well_calibrated(self) -> bool:
        """Check if model is well calibrated (ECE < 0.05)."""
        return self.ece < 0.05

    @property
    def calibration_quality(self) -> str:
        """Get calibration quality description."""
        if self.ece < 0.02:
            return "excellent"
        elif self.ece < 0.05:
            return "good"
        elif self.ece < 0.10:
            return "moderate"
        elif self.ece < 0.20:
            return "poor"
        else:
            return "very_poor"


@dataclass
class ConfidenceEstimate:
    """A confidence estimate for a prediction."""

    value: float
    source: ConfidenceSource
    raw_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": round(self.value, 4),
            "source": self.source.value,
            "raw_scores": self.raw_scores,
            "metadata": self.metadata,
        }


@dataclass
class CalibrationResult:
    """Result of calibration analysis."""

    predictions: List[float]
    confidences: List[float]
    labels: List[int]
    metrics: CalibrationMetrics
    reliability_diagram: Dict[str, List[float]]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": len(self.predictions),
            "metrics": self.metrics.to_dict(),
            "reliability_diagram": self.reliability_diagram,
            "recommendations": self.recommendations,
        }


class CalibrationAnalyzer:
    """Analyzes model calibration."""

    def __init__(self, n_bins: int = 10):
        """Initialize analyzer."""
        self.n_bins = n_bins

    def analyze(
        self,
        confidences: List[float],
        correct: List[bool],
    ) -> CalibrationResult:
        """Analyze calibration from confidence scores and correctness.

        Args:
            confidences: List of confidence scores (0-1)
            correct: List of whether predictions were correct

        Returns:
            CalibrationResult with metrics and analysis
        """
        if len(confidences) != len(correct):
            raise ValueError("Confidences and correct must have same length")

        if not confidences:
            raise ValueError("Empty input")

        # Convert to labels
        labels = [1 if c else 0 for c in correct]

        # Create bins
        bins = self._create_bins(confidences, labels)

        # Calculate metrics
        metrics = self._calculate_metrics(confidences, labels, bins)

        # Create reliability diagram data
        reliability_diagram = self._create_reliability_diagram(bins)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)

        return CalibrationResult(
            predictions=confidences,
            confidences=confidences,
            labels=labels,
            metrics=metrics,
            reliability_diagram=reliability_diagram,
            recommendations=recommendations,
        )

    def _create_bins(
        self,
        confidences: List[float],
        labels: List[int],
    ) -> List[CalibrationBin]:
        """Create calibration bins."""
        bins = []
        bin_width = 1.0 / self.n_bins

        for i in range(self.n_bins):
            bin_start = i * bin_width
            bin_end = (i + 1) * bin_width

            bins.append(
                CalibrationBin(
                    bin_start=bin_start,
                    bin_end=bin_end,
                    confidence_sum=0.0,
                    accuracy_sum=0.0,
                    count=0,
                )
            )

        # Populate bins
        for conf, label in zip(confidences, labels):
            bin_idx = min(int(conf * self.n_bins), self.n_bins - 1)
            bins[bin_idx].confidence_sum += conf
            bins[bin_idx].accuracy_sum += label
            bins[bin_idx].count += 1

        return bins

    def _calculate_metrics(
        self,
        confidences: List[float],
        labels: List[int],
        bins: List[CalibrationBin],
    ) -> CalibrationMetrics:
        """Calculate calibration metrics."""
        n = len(confidences)

        # ECE: Expected Calibration Error (weighted average gap)
        ece = sum(b.count * b.gap for b in bins) / n if n > 0 else 0.0

        # MCE: Maximum Calibration Error
        mce = max((b.gap for b in bins if b.count > 0), default=0.0)

        # ACE: Average Calibration Error (unweighted)
        non_empty_bins = [b for b in bins if b.count > 0]
        ace = sum(b.gap for b in non_empty_bins) / len(non_empty_bins) if non_empty_bins else 0.0

        # Over/under confidence
        overconfidence = 0.0
        underconfidence = 0.0
        for b in bins:
            if b.count > 0:
                diff = b.avg_confidence - b.avg_accuracy
                if diff > 0:
                    overconfidence += b.count * diff
                else:
                    underconfidence += b.count * abs(diff)
        overconfidence /= n if n > 0 else 1
        underconfidence /= n if n > 0 else 1

        # Brier score
        brier_score = sum(
            (conf - label) ** 2 for conf, label in zip(confidences, labels)
        ) / n if n > 0 else 0.0

        # Log loss (with clipping to avoid log(0))
        eps = 1e-15
        log_loss = -sum(
            label * math.log(max(conf, eps)) + (1 - label) * math.log(max(1 - conf, eps))
            for conf, label in zip(confidences, labels)
        ) / n if n > 0 else 0.0

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            ace=ace,
            overconfidence=overconfidence,
            underconfidence=underconfidence,
            brier_score=brier_score,
            log_loss=log_loss,
            bins=bins,
        )

    def _create_reliability_diagram(
        self,
        bins: List[CalibrationBin],
    ) -> Dict[str, List[float]]:
        """Create reliability diagram data."""
        return {
            "bin_centers": [(b.bin_start + b.bin_end) / 2 for b in bins],
            "accuracies": [b.avg_accuracy for b in bins],
            "confidences": [b.avg_confidence for b in bins],
            "counts": [b.count for b in bins],
            "gaps": [b.gap for b in bins],
        }

    def _generate_recommendations(
        self,
        metrics: CalibrationMetrics,
    ) -> List[str]:
        """Generate calibration recommendations."""
        recommendations = []

        if metrics.ece > 0.1:
            recommendations.append("Consider applying temperature scaling to improve calibration")

        if metrics.overconfidence > metrics.underconfidence * 2:
            recommendations.append("Model is significantly overconfident - increase temperature")

        if metrics.underconfidence > metrics.overconfidence * 2:
            recommendations.append("Model is underconfident - decrease temperature or use Platt scaling")

        if metrics.mce > 0.3:
            recommendations.append("High maximum calibration error - check for outlier confidence bins")

        if metrics.brier_score > 0.25:
            recommendations.append("High Brier score suggests poor overall probability estimates")

        if not recommendations:
            recommendations.append("Model appears reasonably well calibrated")

        return recommendations


class TemperatureScaler:
    """Applies temperature scaling for calibration."""

    def __init__(self, temperature: float = 1.0):
        """Initialize scaler."""
        self.temperature = temperature
        self._fitted = False

    def fit(
        self,
        logits: List[float],
        labels: List[int],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> float:
        """Fit optimal temperature using gradient descent on NLL.

        Args:
            logits: Raw logits or log-probabilities
            labels: True labels (0 or 1)
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for gradient descent

        Returns:
            Optimal temperature
        """
        if not logits or len(logits) != len(labels):
            raise ValueError("Invalid input")

        # Simple gradient descent on negative log likelihood
        temp = 1.0

        for _ in range(n_iterations):
            # Forward pass: softmax with temperature
            scaled_probs = [self._sigmoid(l / temp) for l in logits]

            # Compute gradient
            grad = 0.0
            for logit, prob, label in zip(logits, scaled_probs, labels):
                # Gradient of NLL w.r.t. temperature
                grad += (prob - label) * logit / (temp * temp)

            grad /= len(logits)

            # Update temperature
            temp = max(0.1, temp - learning_rate * grad)

        self.temperature = temp
        self._fitted = True
        return temp

    def scale(self, confidence: float) -> float:
        """Scale a confidence score."""
        if confidence <= 0 or confidence >= 1:
            return confidence

        # Convert to logit, scale, convert back
        logit = math.log(confidence / (1 - confidence))
        scaled_logit = logit / self.temperature
        return self._sigmoid(scaled_logit)

    def scale_batch(self, confidences: List[float]) -> List[float]:
        """Scale multiple confidence scores."""
        return [self.scale(c) for c in confidences]

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function with overflow protection."""
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)


class PlattScaler:
    """Applies Platt scaling for calibration."""

    def __init__(self):
        """Initialize scaler."""
        self.a = 1.0
        self.b = 0.0
        self._fitted = False

    def fit(
        self,
        scores: List[float],
        labels: List[int],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> Tuple[float, float]:
        """Fit Platt scaling parameters.

        Args:
            scores: Model output scores
            labels: True labels
            n_iterations: Number of iterations
            learning_rate: Learning rate

        Returns:
            Tuple of (a, b) parameters
        """
        if not scores or len(scores) != len(labels):
            raise ValueError("Invalid input")

        a, b = 1.0, 0.0

        for _ in range(n_iterations):
            # Compute probabilities
            probs = [self._sigmoid(a * s + b) for s in scores]

            # Compute gradients
            grad_a = sum((p - l) * s for p, l, s in zip(probs, labels, scores)) / len(scores)
            grad_b = sum(p - l for p, l in zip(probs, labels)) / len(scores)

            # Update parameters
            a -= learning_rate * grad_a
            b -= learning_rate * grad_b

        self.a = a
        self.b = b
        self._fitted = True
        return a, b

    def scale(self, score: float) -> float:
        """Scale a score to calibrated probability."""
        return self._sigmoid(self.a * score + self.b)

    def scale_batch(self, scores: List[float]) -> List[float]:
        """Scale multiple scores."""
        return [self.scale(s) for s in scores]

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function with overflow protection."""
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)


class HistogramBinner:
    """Applies histogram binning for calibration."""

    def __init__(self, n_bins: int = 10):
        """Initialize binner."""
        self.n_bins = n_bins
        self.bin_accuracies: List[float] = []
        self._fitted = False

    def fit(
        self,
        confidences: List[float],
        labels: List[int],
    ) -> List[float]:
        """Fit histogram binning.

        Args:
            confidences: Confidence scores
            labels: True labels

        Returns:
            List of bin accuracies
        """
        if not confidences or len(confidences) != len(labels):
            raise ValueError("Invalid input")

        bin_width = 1.0 / self.n_bins
        bin_sums = [0.0] * self.n_bins
        bin_counts = [0] * self.n_bins

        for conf, label in zip(confidences, labels):
            bin_idx = min(int(conf * self.n_bins), self.n_bins - 1)
            bin_sums[bin_idx] += label
            bin_counts[bin_idx] += 1

        self.bin_accuracies = [
            bin_sums[i] / bin_counts[i] if bin_counts[i] > 0 else (i + 0.5) / self.n_bins
            for i in range(self.n_bins)
        ]

        self._fitted = True
        return self.bin_accuracies

    def calibrate(self, confidence: float) -> float:
        """Calibrate a confidence score."""
        if not self._fitted:
            raise RuntimeError("Must fit before calibrating")

        bin_idx = min(int(confidence * self.n_bins), self.n_bins - 1)
        return self.bin_accuracies[bin_idx]

    def calibrate_batch(self, confidences: List[float]) -> List[float]:
        """Calibrate multiple confidence scores."""
        return [self.calibrate(c) for c in confidences]


class ConfidenceEstimator:
    """Estimates confidence from various sources."""

    def __init__(self):
        """Initialize estimator."""
        self.verbalized_patterns = {
            "certain": 0.95,
            "confident": 0.85,
            "likely": 0.75,
            "probably": 0.70,
            "possibly": 0.50,
            "might": 0.45,
            "maybe": 0.45,
            "not sure": 0.35,
            "uncertain": 0.35,
            "unsure": 0.30,
            "don't know": 0.15,
            "no idea": 0.10,
        }

    def from_logprobs(
        self,
        logprobs: List[float],
        aggregate: str = "mean",
    ) -> ConfidenceEstimate:
        """Estimate confidence from log probabilities.

        Args:
            logprobs: List of log probabilities
            aggregate: Aggregation method ('mean', 'min', 'product')

        Returns:
            ConfidenceEstimate
        """
        if not logprobs:
            return ConfidenceEstimate(value=0.0, source=ConfidenceSource.LOGPROBS)

        probs = [math.exp(lp) for lp in logprobs]

        if aggregate == "mean":
            value = sum(probs) / len(probs)
        elif aggregate == "min":
            value = min(probs)
        elif aggregate == "product":
            value = 1.0
            for p in probs:
                value *= p
            # Normalize by taking nth root
            value = value ** (1 / len(probs))
        else:
            value = sum(probs) / len(probs)

        return ConfidenceEstimate(
            value=value,
            source=ConfidenceSource.LOGPROBS,
            raw_scores=probs,
        )

    def from_consistency(
        self,
        responses: List[str],
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ) -> ConfidenceEstimate:
        """Estimate confidence from response consistency.

        Args:
            responses: Multiple responses to same prompt
            similarity_fn: Function to compute similarity between responses

        Returns:
            ConfidenceEstimate
        """
        if not responses:
            return ConfidenceEstimate(value=0.0, source=ConfidenceSource.CONSISTENCY)

        if len(responses) == 1:
            return ConfidenceEstimate(value=0.5, source=ConfidenceSource.CONSISTENCY)

        # Default similarity: exact match
        if similarity_fn is None:
            def similarity_fn(a: str, b: str) -> float:
                return 1.0 if a.strip().lower() == b.strip().lower() else 0.0

        # Compute pairwise similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = similarity_fn(responses[i], responses[j])
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        return ConfidenceEstimate(
            value=avg_similarity,
            source=ConfidenceSource.CONSISTENCY,
            raw_scores=similarities,
            metadata={"n_responses": len(responses)},
        )

    def from_verbalized(self, response: str) -> ConfidenceEstimate:
        """Estimate confidence from verbalized uncertainty.

        Args:
            response: Model response that may contain confidence language

        Returns:
            ConfidenceEstimate
        """
        response_lower = response.lower()

        # Find matching patterns
        matches = []
        for pattern, confidence in self.verbalized_patterns.items():
            if pattern in response_lower:
                matches.append(confidence)

        if not matches:
            # No explicit uncertainty markers - assume moderate confidence
            return ConfidenceEstimate(
                value=0.7,
                source=ConfidenceSource.VERBALIZED,
                metadata={"patterns_found": []},
            )

        # Use the lowest confidence found (most conservative)
        value = min(matches)

        return ConfidenceEstimate(
            value=value,
            source=ConfidenceSource.VERBALIZED,
            metadata={"patterns_found": [p for p in self.verbalized_patterns if p in response_lower]},
        )

    def from_entropy(
        self,
        token_probs: List[List[float]],
    ) -> ConfidenceEstimate:
        """Estimate confidence from token probability entropy.

        Args:
            token_probs: List of probability distributions for each token position

        Returns:
            ConfidenceEstimate
        """
        if not token_probs:
            return ConfidenceEstimate(value=0.0, source=ConfidenceSource.ENTROPY)

        entropies = []
        for probs in token_probs:
            if not probs or all(p <= 0 for p in probs):
                continue
            # Compute entropy
            entropy = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
            # Normalize by max entropy (log of vocab size approximation)
            max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            entropies.append(normalized_entropy)

        if not entropies:
            return ConfidenceEstimate(value=0.0, source=ConfidenceSource.ENTROPY)

        # Low entropy = high confidence
        avg_entropy = sum(entropies) / len(entropies)
        confidence = 1.0 - avg_entropy

        return ConfidenceEstimate(
            value=max(0.0, min(1.0, confidence)),
            source=ConfidenceSource.ENTROPY,
            raw_scores=entropies,
        )


class Calibrator:
    """Main calibration class combining multiple methods."""

    def __init__(self, method: CalibrationMethod = CalibrationMethod.TEMPERATURE_SCALING):
        """Initialize calibrator."""
        self.method = method
        self._scaler = None
        self._fitted = False

    def fit(
        self,
        confidences: List[float],
        labels: List[int],
    ) -> None:
        """Fit calibration model.

        Args:
            confidences: Confidence scores
            labels: True labels (0 or 1)
        """
        if self.method == CalibrationMethod.TEMPERATURE_SCALING:
            # Convert confidences to logits
            logits = [
                math.log(c / (1 - c + 1e-15) + 1e-15)
                for c in confidences
            ]
            self._scaler = TemperatureScaler()
            self._scaler.fit(logits, labels)

        elif self.method == CalibrationMethod.PLATT_SCALING:
            self._scaler = PlattScaler()
            self._scaler.fit(confidences, labels)

        elif self.method == CalibrationMethod.HISTOGRAM_BINNING:
            self._scaler = HistogramBinner()
            self._scaler.fit(confidences, labels)

        self._fitted = True

    def calibrate(self, confidence: float) -> float:
        """Calibrate a single confidence score.

        Args:
            confidence: Raw confidence score

        Returns:
            Calibrated confidence
        """
        if not self._fitted:
            raise RuntimeError("Must fit before calibrating")

        if self.method == CalibrationMethod.TEMPERATURE_SCALING:
            return self._scaler.scale(confidence)
        elif self.method == CalibrationMethod.PLATT_SCALING:
            return self._scaler.scale(confidence)
        elif self.method == CalibrationMethod.HISTOGRAM_BINNING:
            return self._scaler.calibrate(confidence)

        return confidence

    def calibrate_batch(self, confidences: List[float]) -> List[float]:
        """Calibrate multiple confidence scores.

        Args:
            confidences: List of raw confidence scores

        Returns:
            List of calibrated confidences
        """
        return [self.calibrate(c) for c in confidences]


# Convenience functions
def analyze_calibration(
    confidences: List[float],
    correct: List[bool],
    n_bins: int = 10,
) -> CalibrationResult:
    """Analyze model calibration.

    Args:
        confidences: List of confidence scores
        correct: List of whether predictions were correct
        n_bins: Number of bins for analysis

    Returns:
        CalibrationResult with metrics
    """
    analyzer = CalibrationAnalyzer(n_bins=n_bins)
    return analyzer.analyze(confidences, correct)


def calculate_ece(
    confidences: List[float],
    correct: List[bool],
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error.

    Args:
        confidences: Confidence scores
        correct: Correctness labels
        n_bins: Number of bins

    Returns:
        ECE value
    """
    result = analyze_calibration(confidences, correct, n_bins)
    return result.metrics.ece


def calculate_brier_score(
    confidences: List[float],
    correct: List[bool],
) -> float:
    """Calculate Brier score.

    Args:
        confidences: Confidence scores
        correct: Correctness labels

    Returns:
        Brier score
    """
    if not confidences or len(confidences) != len(correct):
        raise ValueError("Invalid input")

    labels = [1 if c else 0 for c in correct]
    return sum((conf - label) ** 2 for conf, label in zip(confidences, labels)) / len(confidences)


def apply_temperature_scaling(
    confidences: List[float],
    temperature: float,
) -> List[float]:
    """Apply temperature scaling to confidences.

    Args:
        confidences: Raw confidence scores
        temperature: Temperature parameter

    Returns:
        Scaled confidences
    """
    scaler = TemperatureScaler(temperature=temperature)
    return scaler.scale_batch(confidences)


def fit_temperature(
    confidences: List[float],
    labels: List[int],
) -> float:
    """Fit optimal temperature for calibration.

    Args:
        confidences: Confidence scores
        labels: True labels

    Returns:
        Optimal temperature
    """
    # Convert to logits
    logits = [
        math.log(c / (1 - c + 1e-15) + 1e-15)
        for c in confidences
    ]
    scaler = TemperatureScaler()
    return scaler.fit(logits, labels)


def estimate_confidence_from_consistency(
    responses: List[str],
) -> float:
    """Estimate confidence from response consistency.

    Args:
        responses: Multiple responses to same prompt

    Returns:
        Confidence estimate
    """
    estimator = ConfidenceEstimator()
    result = estimator.from_consistency(responses)
    return result.value


def estimate_confidence_from_verbalized(
    response: str,
) -> float:
    """Estimate confidence from verbalized uncertainty.

    Args:
        response: Model response

    Returns:
        Confidence estimate
    """
    estimator = ConfidenceEstimator()
    result = estimator.from_verbalized(response)
    return result.value


def calibrate_confidences(
    confidences: List[float],
    labels: List[int],
    method: CalibrationMethod = CalibrationMethod.TEMPERATURE_SCALING,
) -> List[float]:
    """Fit and apply calibration to confidences.

    Args:
        confidences: Raw confidence scores
        labels: True labels
        method: Calibration method

    Returns:
        Calibrated confidences
    """
    calibrator = Calibrator(method=method)
    calibrator.fit(confidences, labels)
    return calibrator.calibrate_batch(confidences)
