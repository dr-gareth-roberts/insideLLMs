"""
Model calibration and confidence estimation utilities.

This module provides comprehensive tools for evaluating and improving the calibration
of machine learning model confidence scores. A well-calibrated model produces confidence
scores that accurately reflect the true probability of being correct - for example,
predictions made with 80% confidence should be correct roughly 80% of the time.

Provides tools for:
    - Measuring calibration (reliability diagrams, ECE, MCE, Brier score)
    - Confidence score analysis from multiple sources (logprobs, consistency, verbalized)
    - Temperature scaling for post-hoc calibration
    - Platt scaling (sigmoid calibration)
    - Histogram binning calibration

Key Concepts:
    - **Expected Calibration Error (ECE)**: Weighted average of confidence-accuracy gaps
      across bins. Lower is better, with < 0.05 generally considered well-calibrated.
    - **Maximum Calibration Error (MCE)**: Worst-case calibration gap across bins.
    - **Brier Score**: Mean squared error between confidence and binary outcome.
    - **Temperature Scaling**: Divides logits by a learned temperature parameter.
    - **Platt Scaling**: Fits a sigmoid function to map scores to probabilities.

Examples:
    Basic calibration analysis:

    >>> from insideLLMs.calibration import analyze_calibration
    >>> confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    >>> correct = [True, True, True, False, True, False, False, False]
    >>> result = analyze_calibration(confidences, correct)
    >>> print(f"ECE: {result.metrics.ece:.4f}")
    ECE: 0.0750

    Temperature scaling for overconfident models:

    >>> from insideLLMs.calibration import TemperatureScaler
    >>> scaler = TemperatureScaler(temperature=1.5)
    >>> raw_confidence = 0.95
    >>> calibrated = scaler.scale(raw_confidence)  # Reduces overconfidence
    >>> print(f"Calibrated: {calibrated:.3f}")
    Calibrated: 0.864

    Using the Calibrator with different methods:

    >>> from insideLLMs.calibration import Calibrator, CalibrationMethod
    >>> calibrator = Calibrator(method=CalibrationMethod.PLATT_SCALING)
    >>> train_conf = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    >>> train_labels = [1, 1, 0, 0, 0, 0]
    >>> calibrator.fit(train_conf, train_labels)
    >>> new_conf = calibrator.calibrate(0.85)

    Estimating confidence from model responses:

    >>> from insideLLMs.calibration import ConfidenceEstimator
    >>> estimator = ConfidenceEstimator()
    >>> response = "I'm not entirely sure, but the answer might be 42."
    >>> estimate = estimator.from_verbalized(response)
    >>> print(f"Confidence: {estimate.value:.2f}")
    Confidence: 0.45

See Also:
    - :class:`CalibrationAnalyzer`: Main class for calibration analysis
    - :class:`Calibrator`: Unified interface for calibration methods
    - :func:`calculate_ece`: Quick ECE calculation
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class CalibrationMethod(Enum):
    """Enumeration of available post-hoc calibration methods.

    Post-hoc calibration methods adjust model confidence scores after training
    to improve alignment between confidence and actual accuracy. Each method
    has different characteristics and is suitable for different scenarios.

    Attributes:
        TEMPERATURE_SCALING: Single-parameter method that divides logits by a
            learned temperature. Simple and effective for neural networks.
            Best when calibration error is uniform across confidence levels.
        PLATT_SCALING: Fits a sigmoid function (a*x + b) to map scores to
            probabilities. More flexible than temperature scaling with two
            parameters. Good for SVM outputs and binary classification.
        HISTOGRAM_BINNING: Non-parametric method that bins predictions and
            replaces confidence with empirical accuracy in each bin.
            Requires sufficient data per bin for reliable estimates.
        ISOTONIC_REGRESSION: Non-parametric monotonic regression. Ensures
            calibrated probabilities maintain rank ordering of original scores.
            Good when monotonicity is important.
        BETA_CALIBRATION: Fits a beta distribution to model score distributions.
            Handles boundary effects better than Platt scaling. Good for
            scores naturally bounded in [0, 1].

    Examples:
        Selecting a calibration method:

        >>> method = CalibrationMethod.TEMPERATURE_SCALING
        >>> print(method.value)
        'temperature_scaling'

        Using with the Calibrator class:

        >>> from insideLLMs.calibration import Calibrator, CalibrationMethod
        >>> calibrator = Calibrator(method=CalibrationMethod.PLATT_SCALING)

        Iterating over available methods:

        >>> for method in CalibrationMethod:
        ...     print(f"{method.name}: {method.value}")
        TEMPERATURE_SCALING: temperature_scaling
        PLATT_SCALING: platt_scaling
        HISTOGRAM_BINNING: histogram_binning
        ISOTONIC_REGRESSION: isotonic_regression
        BETA_CALIBRATION: beta_calibration

        Checking method type:

        >>> method = CalibrationMethod.HISTOGRAM_BINNING
        >>> is_parametric = method in [
        ...     CalibrationMethod.TEMPERATURE_SCALING,
        ...     CalibrationMethod.PLATT_SCALING
        ... ]
        >>> print(f"Parametric: {is_parametric}")
        Parametric: False
    """

    TEMPERATURE_SCALING = "temperature_scaling"
    PLATT_SCALING = "platt_scaling"
    HISTOGRAM_BINNING = "histogram_binning"
    ISOTONIC_REGRESSION = "isotonic_regression"
    BETA_CALIBRATION = "beta_calibration"


class ConfidenceSource(Enum):
    """Enumeration of sources for obtaining model confidence scores.

    Different methods of extracting confidence from language models yield
    confidence scores with varying characteristics. Understanding the source
    helps interpret the confidence and choose appropriate calibration methods.

    Attributes:
        LOGPROBS: Confidence derived from token log-probabilities. Direct access
            to model's internal probability estimates. Often well-calibrated
            for single tokens but can degrade for sequences.
        SELF_REPORTED: Model explicitly states a confidence percentage in its
            response (e.g., "I am 85% confident..."). May not correlate well
            with actual accuracy; often overconfident.
        CONSISTENCY: Confidence estimated from agreement across multiple
            independent samples. High consistency suggests high confidence.
            Robust but computationally expensive (requires multiple inferences).
        ENTROPY: Confidence derived from the entropy of token probability
            distributions. Low entropy indicates high confidence (model
            concentrates probability mass on few tokens).
        VERBALIZED: Confidence inferred from hedging language in responses
            (e.g., "probably", "I think", "not sure"). Interpretable but
            may not match model's internal uncertainty.

    Examples:
        Creating a confidence estimate with source tracking:

        >>> from insideLLMs.calibration import ConfidenceEstimate, ConfidenceSource
        >>> estimate = ConfidenceEstimate(
        ...     value=0.85,
        ...     source=ConfidenceSource.LOGPROBS
        ... )
        >>> print(f"Source: {estimate.source.value}")
        Source: logprobs

        Filtering estimates by source:

        >>> estimates = [
        ...     ConfidenceEstimate(0.9, ConfidenceSource.LOGPROBS),
        ...     ConfidenceEstimate(0.7, ConfidenceSource.VERBALIZED),
        ...     ConfidenceEstimate(0.8, ConfidenceSource.CONSISTENCY),
        ... ]
        >>> logprob_estimates = [e for e in estimates if e.source == ConfidenceSource.LOGPROBS]
        >>> len(logprob_estimates)
        1

        Comparing confidence sources for reliability:

        >>> # Generally, reliability ranking (from research):
        >>> reliability_order = [
        ...     ConfidenceSource.CONSISTENCY,  # Most reliable
        ...     ConfidenceSource.ENTROPY,
        ...     ConfidenceSource.LOGPROBS,
        ...     ConfidenceSource.VERBALIZED,
        ...     ConfidenceSource.SELF_REPORTED,  # Least reliable
        ... ]

        Using in calibration workflows:

        >>> from insideLLMs.calibration import ConfidenceEstimator
        >>> estimator = ConfidenceEstimator()
        >>> # Choose method based on available data
        >>> if have_logprobs:
        ...     estimate = estimator.from_logprobs(logprobs)
        ... else:
        ...     estimate = estimator.from_verbalized(response_text)
    """

    LOGPROBS = "logprobs"
    SELF_REPORTED = "self_reported"
    CONSISTENCY = "consistency"
    ENTROPY = "entropy"
    VERBALIZED = "verbalized"


@dataclass
class CalibrationBin:
    """A single bin in calibration analysis for reliability diagrams.

    Calibration bins partition the confidence range [0, 1] into intervals and
    track the average confidence and accuracy of predictions falling within
    each interval. This enables calculation of calibration metrics and
    visualization via reliability diagrams.

    The calibration gap for a bin measures how much the average confidence
    deviates from the average accuracy - a gap of 0 indicates perfect
    calibration within that bin.

    Attributes:
        bin_start: Lower bound of the confidence interval (inclusive).
        bin_end: Upper bound of the confidence interval (exclusive, except for
            the last bin which is inclusive).
        confidence_sum: Sum of confidence scores in this bin (for computing avg).
        accuracy_sum: Sum of binary correctness labels (1 if correct, 0 otherwise).
        count: Number of predictions falling in this bin.

    Examples:
        Creating and inspecting a calibration bin:

        >>> bin = CalibrationBin(
        ...     bin_start=0.8,
        ...     bin_end=0.9,
        ...     confidence_sum=4.25,  # Sum of confidences: 0.85 + 0.82 + 0.88 + 0.85 + 0.85
        ...     accuracy_sum=4.0,     # 4 out of 5 correct
        ...     count=5
        ... )
        >>> print(f"Avg confidence: {bin.avg_confidence:.3f}")
        Avg confidence: 0.850
        >>> print(f"Avg accuracy: {bin.avg_accuracy:.3f}")
        Avg accuracy: 0.800

        Checking calibration quality of a bin:

        >>> bin = CalibrationBin(0.7, 0.8, 3.75, 3.0, 5)  # 75% conf, 60% accuracy
        >>> print(f"Calibration gap: {bin.gap:.3f}")
        Calibration gap: 0.150
        >>> if bin.gap > 0.1:
        ...     print("Bin is poorly calibrated")
        Bin is poorly calibrated

        Converting to dictionary for serialization:

        >>> bin = CalibrationBin(0.9, 1.0, 1.9, 1.0, 2)
        >>> d = bin.to_dict()
        >>> print(d["bin_range"])
        [0.9, 1.0]
        >>> print(d["gap"])
        0.45

        Handling empty bins:

        >>> empty_bin = CalibrationBin(0.0, 0.1, 0.0, 0.0, 0)
        >>> print(f"Avg confidence: {empty_bin.avg_confidence}")
        Avg confidence: 0.0
        >>> print(f"Gap: {empty_bin.gap}")
        Gap: 0.0
    """

    bin_start: float
    bin_end: float
    confidence_sum: float
    accuracy_sum: float
    count: int

    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence score in this bin.

        Returns:
            Average confidence if bin has samples, 0.0 otherwise.

        Examples:
            >>> bin = CalibrationBin(0.8, 0.9, 2.55, 2.0, 3)
            >>> print(f"{bin.avg_confidence:.3f}")
            0.850
        """
        return self.confidence_sum / self.count if self.count > 0 else 0.0

    @property
    def avg_accuracy(self) -> float:
        """Calculate average accuracy (fraction correct) in this bin.

        Returns:
            Average accuracy if bin has samples, 0.0 otherwise.

        Examples:
            >>> bin = CalibrationBin(0.8, 0.9, 2.55, 2.0, 3)
            >>> print(f"{bin.avg_accuracy:.3f}")
            0.667
        """
        return self.accuracy_sum / self.count if self.count > 0 else 0.0

    @property
    def gap(self) -> float:
        """Calculate absolute calibration gap between confidence and accuracy.

        The gap measures miscalibration: positive gaps where avg_confidence >
        avg_accuracy indicate overconfidence, while negative gaps indicate
        underconfidence. This property returns the absolute value.

        Returns:
            Absolute difference between average confidence and accuracy.

        Examples:
            >>> bin = CalibrationBin(0.8, 0.9, 4.25, 3.0, 5)  # 85% conf, 60% acc
            >>> print(f"Gap: {bin.gap:.3f}")
            Gap: 0.250
        """
        return abs(self.avg_confidence - self.avg_accuracy)

    def to_dict(self) -> dict[str, Any]:
        """Convert bin data to a dictionary for serialization.

        Returns:
            Dictionary containing bin statistics with keys:
            - bin_range: [start, end] interval
            - avg_confidence: Mean confidence (rounded to 4 decimals)
            - avg_accuracy: Mean accuracy (rounded to 4 decimals)
            - count: Number of samples
            - gap: Calibration gap (rounded to 4 decimals)

        Examples:
            >>> bin = CalibrationBin(0.7, 0.8, 3.75, 4.0, 5)
            >>> d = bin.to_dict()
            >>> d["bin_range"]
            [0.7, 0.8]
            >>> d["count"]
            5
        """
        return {
            "bin_range": [self.bin_start, self.bin_end],
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_accuracy": round(self.avg_accuracy, 4),
            "count": self.count,
            "gap": round(self.gap, 4),
        }


@dataclass
class CalibrationMetrics:
    """Comprehensive calibration metrics for evaluating model confidence quality.

    This dataclass aggregates multiple calibration metrics that measure different
    aspects of how well a model's confidence scores align with actual accuracy.
    It provides both aggregate scores and per-bin statistics for detailed analysis.

    Attributes:
        ece: Expected Calibration Error. Weighted average of per-bin gaps,
            where weights are proportional to bin sample counts. Primary metric
            for overall calibration quality. Range [0, 1], lower is better.
        mce: Maximum Calibration Error. Largest calibration gap across all
            non-empty bins. Captures worst-case miscalibration. Useful for
            detecting problematic confidence regions.
        ace: Average Calibration Error. Unweighted average of per-bin gaps.
            Treats all confidence levels equally regardless of sample frequency.
        overconfidence: Weighted average of gaps where confidence > accuracy.
            Measures tendency to be too confident. Common in neural networks.
        underconfidence: Weighted average of gaps where accuracy > confidence.
            Measures tendency to be too uncertain.
        brier_score: Mean squared error between confidences and binary outcomes.
            Combines calibration and refinement. Range [0, 1], lower is better.
            Score of 0.25 corresponds to random guessing with 50% confidence.
        log_loss: Negative log likelihood (cross-entropy loss). More sensitive
            to confident wrong predictions than Brier score. Heavily penalizes
            extreme miscalibration.
        bins: List of CalibrationBin objects containing per-bin statistics.
            Enables detailed analysis and reliability diagram visualization.

    Examples:
        Creating and inspecting calibration metrics:

        >>> metrics = CalibrationMetrics(
        ...     ece=0.08,
        ...     mce=0.15,
        ...     ace=0.07,
        ...     overconfidence=0.06,
        ...     underconfidence=0.02,
        ...     brier_score=0.18,
        ...     log_loss=0.52,
        ...     bins=[]
        ... )
        >>> print(f"ECE: {metrics.ece:.2f}, Quality: {metrics.calibration_quality}")
        ECE: 0.08, Quality: moderate

        Checking if model is well calibrated:

        >>> metrics = CalibrationMetrics(0.03, 0.08, 0.04, 0.02, 0.01, 0.15, 0.40)
        >>> if metrics.is_well_calibrated:
        ...     print("Model is well calibrated!")
        Model is well calibrated!

        Analyzing over/underconfidence patterns:

        >>> metrics = CalibrationMetrics(0.12, 0.25, 0.10, 0.10, 0.02, 0.22, 0.55)
        >>> if metrics.overconfidence > 2 * metrics.underconfidence:
        ...     print("Model is significantly overconfident - consider temperature scaling")
        Model is significantly overconfident - consider temperature scaling

        Converting to dictionary for JSON serialization:

        >>> metrics = CalibrationMetrics(0.05, 0.12, 0.06, 0.03, 0.02, 0.17, 0.45)
        >>> d = metrics.to_dict()
        >>> print(f"ECE: {d['ece']}, Brier: {d['brier_score']}")
        ECE: 0.05, Brier: 0.17

    See Also:
        CalibrationBin: Per-bin statistics container
        CalibrationAnalyzer: Computes these metrics from raw data
    """

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    ace: float  # Average Calibration Error
    overconfidence: float  # How much model overestimates
    underconfidence: float  # How much model underestimates
    brier_score: float  # Brier score (mean squared error)
    log_loss: float  # Log loss / cross-entropy
    bins: list[CalibrationBin] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert all metrics to a dictionary for serialization.

        Returns:
            Dictionary containing all metrics rounded to 4 decimal places,
            plus a 'bins' key with serialized bin data.

        Examples:
            >>> metrics = CalibrationMetrics(0.05, 0.12, 0.06, 0.03, 0.02, 0.17, 0.45)
            >>> d = metrics.to_dict()
            >>> list(d.keys())
            ['ece', 'mce', 'ace', 'overconfidence', 'underconfidence', 'brier_score', 'log_loss', 'bins']
        """
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
        """Check if model is well calibrated based on ECE threshold.

        Uses a threshold of 0.05 (5%) for ECE, which is a commonly accepted
        standard in the calibration literature.

        Returns:
            True if ECE < 0.05, False otherwise.

        Examples:
            >>> CalibrationMetrics(0.03, 0.08, 0.04, 0.02, 0.01, 0.15, 0.40).is_well_calibrated
            True
            >>> CalibrationMetrics(0.08, 0.15, 0.07, 0.06, 0.02, 0.20, 0.50).is_well_calibrated
            False
        """
        return self.ece < 0.05

    @property
    def calibration_quality(self) -> str:
        """Get human-readable calibration quality description based on ECE.

        Returns:
            Quality descriptor string:
            - 'excellent': ECE < 0.02
            - 'good': 0.02 <= ECE < 0.05
            - 'moderate': 0.05 <= ECE < 0.10
            - 'poor': 0.10 <= ECE < 0.20
            - 'very_poor': ECE >= 0.20

        Examples:
            >>> CalibrationMetrics(0.01, 0.05, 0.02, 0.01, 0.005, 0.10, 0.30).calibration_quality
            'excellent'
            >>> CalibrationMetrics(0.15, 0.30, 0.12, 0.10, 0.05, 0.25, 0.60).calibration_quality
            'poor'
        """
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
    """A confidence estimate for a prediction with source tracking and metadata.

    Encapsulates a confidence score along with information about how it was
    derived. This enables downstream analysis to appropriately weight or
    process confidence scores based on their source, as different sources
    have different reliability characteristics.

    Attributes:
        value: The confidence score in range [0, 1]. Higher values indicate
            greater model confidence in the prediction.
        source: ConfidenceSource enum indicating how the confidence was derived.
            Affects interpretation and appropriate calibration methods.
        raw_scores: Optional list of underlying scores used to compute the
            confidence value. Useful for debugging and detailed analysis.
            For logprobs, these are the token probabilities; for consistency,
            these are pairwise similarity scores.
        metadata: Optional dictionary of additional information about the
            confidence estimate. May include patterns found (for verbalized),
            number of samples (for consistency), etc.

    Examples:
        Creating a confidence estimate from logprobs:

        >>> from insideLLMs.calibration import ConfidenceEstimate, ConfidenceSource
        >>> estimate = ConfidenceEstimate(
        ...     value=0.85,
        ...     source=ConfidenceSource.LOGPROBS,
        ...     raw_scores=[0.9, 0.8, 0.85],
        ...     metadata={"tokens": ["The", "answer", "is"]}
        ... )
        >>> print(f"Confidence: {estimate.value:.2f} from {estimate.source.value}")
        Confidence: 0.85 from logprobs

        Creating from verbalized uncertainty:

        >>> estimate = ConfidenceEstimate(
        ...     value=0.45,
        ...     source=ConfidenceSource.VERBALIZED,
        ...     metadata={"patterns_found": ["might", "possibly"]}
        ... )
        >>> print(f"Found hedging: {estimate.metadata['patterns_found']}")
        Found hedging: ['might', 'possibly']

        Comparing estimates from different sources:

        >>> estimates = [
        ...     ConfidenceEstimate(0.9, ConfidenceSource.LOGPROBS),
        ...     ConfidenceEstimate(0.7, ConfidenceSource.CONSISTENCY),
        ...     ConfidenceEstimate(0.5, ConfidenceSource.VERBALIZED),
        ... ]
        >>> avg_conf = sum(e.value for e in estimates) / len(estimates)
        >>> print(f"Average confidence: {avg_conf:.2f}")
        Average confidence: 0.70

        Serializing for storage or API response:

        >>> estimate = ConfidenceEstimate(0.75, ConfidenceSource.ENTROPY)
        >>> d = estimate.to_dict()
        >>> print(d)
        {'value': 0.75, 'source': 'entropy', 'raw_scores': None, 'metadata': {}}
    """

    value: float
    source: ConfidenceSource
    raw_scores: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert confidence estimate to a dictionary for serialization.

        Returns:
            Dictionary with keys:
            - value: Confidence score (rounded to 4 decimal places)
            - source: String value of the ConfidenceSource enum
            - raw_scores: List of underlying scores or None
            - metadata: Additional metadata dictionary

        Examples:
            >>> estimate = ConfidenceEstimate(
            ...     value=0.8567,
            ...     source=ConfidenceSource.LOGPROBS,
            ...     raw_scores=[0.9, 0.8, 0.82],
            ...     metadata={"model": "gpt-4"}
            ... )
            >>> d = estimate.to_dict()
            >>> d["value"]
            0.8567
            >>> d["source"]
            'logprobs'
        """
        return {
            "value": round(self.value, 4),
            "source": self.source.value,
            "raw_scores": self.raw_scores,
            "metadata": self.metadata,
        }


@dataclass
class ConfidenceCalibrationResult:
    """Complete result of calibration analysis including metrics and recommendations.

    This dataclass contains all outputs from a calibration analysis, including
    the raw data, computed metrics, reliability diagram data for visualization,
    and actionable recommendations for improving calibration.

    Attributes:
        predictions: List of model confidence/probability predictions.
            Same as confidences for binary classification.
        confidences: List of confidence scores in range [0, 1].
        labels: List of ground truth binary labels (0 or 1).
        metrics: CalibrationMetrics object containing ECE, MCE, Brier score, etc.
        reliability_diagram: Dictionary containing data for plotting reliability
            diagrams. Keys include 'bin_centers', 'accuracies', 'confidences',
            'counts', and 'gaps'.
        recommendations: List of actionable string recommendations for improving
            model calibration based on the analysis results.

    Examples:
        Running a calibration analysis:

        >>> from insideLLMs.calibration import analyze_calibration
        >>> confidences = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25]
        >>> correct = [True, True, True, False, True, False, False, False]
        >>> result = analyze_calibration(confidences, correct)
        >>> print(f"ECE: {result.metrics.ece:.3f}")
        ECE: 0.075

        Accessing reliability diagram data for visualization:

        >>> result = analyze_calibration([0.9, 0.7, 0.5, 0.3], [True, True, False, False])
        >>> print(f"Bin centers: {result.reliability_diagram['bin_centers'][:3]}")
        Bin centers: [0.05, 0.15, 0.25]

        Getting calibration recommendations:

        >>> confidences = [0.99, 0.95, 0.90, 0.85]  # Very high confidence
        >>> correct = [True, False, True, False]   # Only 50% correct
        >>> result = analyze_calibration(confidences, correct)
        >>> for rec in result.recommendations:
        ...     print(f"- {rec}")
        - Consider applying temperature scaling to improve calibration
        - Model is significantly overconfident - increase temperature

        Converting to JSON-serializable format:

        >>> result = analyze_calibration([0.8, 0.6, 0.4], [True, True, False])
        >>> d = result.to_dict()
        >>> print(f"Samples: {d['n_samples']}, ECE: {d['metrics']['ece']}")
        Samples: 3, ECE: ...

    See Also:
        analyze_calibration: Convenience function to create this result
        CalibrationAnalyzer: Class that performs the analysis
    """

    predictions: list[float]
    confidences: list[float]
    labels: list[int]
    metrics: CalibrationMetrics
    reliability_diagram: dict[str, list[float]]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the calibration result to a dictionary for serialization.

        Note that raw predictions, confidences, and labels are not included
        to keep the output compact. Use the attributes directly if you need
        the raw data.

        Returns:
            Dictionary containing:
            - n_samples: Number of samples analyzed
            - metrics: Serialized CalibrationMetrics
            - reliability_diagram: Data for plotting
            - recommendations: List of improvement suggestions

        Examples:
            >>> from insideLLMs.calibration import analyze_calibration
            >>> result = analyze_calibration([0.8, 0.6], [True, False])
            >>> d = result.to_dict()
            >>> 'n_samples' in d and 'metrics' in d
            True
        """
        return {
            "n_samples": len(self.predictions),
            "metrics": self.metrics.to_dict(),
            "reliability_diagram": self.reliability_diagram,
            "recommendations": self.recommendations,
        }


class CalibrationAnalyzer:
    """Analyzes model calibration using binning-based metrics and reliability diagrams.

    This class computes various calibration metrics by partitioning predictions
    into bins based on confidence scores and comparing average confidence to
    average accuracy within each bin. It produces comprehensive calibration
    statistics and actionable recommendations.

    The analyzer uses equal-width binning where the [0, 1] confidence interval
    is divided into n_bins equal segments. Predictions are assigned to bins
    based on their confidence scores, and per-bin statistics are computed.

    Attributes:
        n_bins: Number of equal-width bins for calibration analysis. More bins
            provide finer granularity but require more samples for reliable
            estimates. Default is 10, which is standard in the literature.

    Examples:
        Basic calibration analysis:

        >>> analyzer = CalibrationAnalyzer(n_bins=10)
        >>> confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        >>> correct = [True, True, True, False, True, False, False, False]
        >>> result = analyzer.analyze(confidences, correct)
        >>> print(f"ECE: {result.metrics.ece:.4f}")
        ECE: 0.0750

        Using fewer bins for small datasets:

        >>> analyzer = CalibrationAnalyzer(n_bins=5)
        >>> confidences = [0.95, 0.75, 0.55, 0.35, 0.15]
        >>> correct = [True, True, False, False, False]
        >>> result = analyzer.analyze(confidences, correct)
        >>> print(f"Quality: {result.metrics.calibration_quality}")
        Quality: good

        Analyzing overconfident predictions:

        >>> analyzer = CalibrationAnalyzer()
        >>> # High confidence but mixed accuracy
        >>> confidences = [0.99, 0.95, 0.92, 0.88, 0.85]
        >>> correct = [True, False, True, False, True]  # 60% correct
        >>> result = analyzer.analyze(confidences, correct)
        >>> print(f"Overconfidence: {result.metrics.overconfidence:.3f}")
        Overconfidence: 0.318

        Extracting reliability diagram data for plotting:

        >>> result = CalibrationAnalyzer().analyze(
        ...     [0.9, 0.7, 0.5, 0.3, 0.1],
        ...     [True, True, False, False, False]
        ... )
        >>> diagram = result.reliability_diagram
        >>> # Plot with matplotlib: plt.bar(diagram['bin_centers'], diagram['accuracies'])

    See Also:
        analyze_calibration: Convenience function wrapping this class
        CalibrationMetrics: Container for computed metrics
    """

    def __init__(self, n_bins: int = 10):
        """Initialize the calibration analyzer.

        Args:
            n_bins: Number of equal-width bins for calibration analysis.
                More bins provide finer-grained analysis but require more
                samples for reliable per-bin estimates. Standard values
                range from 10 to 20. Default is 10.

        Examples:
            >>> analyzer = CalibrationAnalyzer()  # Default 10 bins
            >>> analyzer = CalibrationAnalyzer(n_bins=15)  # Finer granularity
            >>> analyzer = CalibrationAnalyzer(n_bins=5)   # For small datasets
        """
        self.n_bins = n_bins

    def analyze(
        self,
        confidences: list[float],
        correct: list[bool],
    ) -> ConfidenceCalibrationResult:
        """Analyze calibration from confidence scores and correctness labels.

        Computes comprehensive calibration metrics including ECE, MCE, Brier
        score, log loss, and over/underconfidence measures. Also generates
        reliability diagram data and actionable recommendations.

        Args:
            confidences: List of confidence scores in range [0, 1]. Each score
                represents the model's predicted probability of being correct.
            correct: List of boolean correctness labels. True indicates the
                model's prediction was correct, False indicates incorrect.
                Must have same length as confidences.

        Returns:
            ConfidenceCalibrationResult containing:
            - predictions/confidences: The input confidence scores
            - labels: Binary labels (1 for correct, 0 for incorrect)
            - metrics: CalibrationMetrics with ECE, MCE, Brier score, etc.
            - reliability_diagram: Data for plotting (bin centers, accuracies, etc.)
            - recommendations: List of suggested calibration improvements

        Raises:
            ValueError: If confidences and correct have different lengths,
                or if the input lists are empty.

        Examples:
            Standard analysis with mixed calibration:

            >>> analyzer = CalibrationAnalyzer()
            >>> result = analyzer.analyze(
            ...     confidences=[0.9, 0.8, 0.7, 0.6],
            ...     correct=[True, True, False, True]
            ... )
            >>> print(f"ECE: {result.metrics.ece:.3f}")
            ECE: 0.100

            Analyzing a well-calibrated model:

            >>> result = analyzer.analyze(
            ...     confidences=[0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
            ...     correct=[True, True, True, False, False, False]
            ... )
            >>> print(f"Well calibrated: {result.metrics.is_well_calibrated}")
            Well calibrated: True

            Getting recommendations for improvement:

            >>> result = analyzer.analyze(
            ...     confidences=[0.95] * 10,  # Always very confident
            ...     correct=[True, False] * 5  # But only 50% correct
            ... )
            >>> print(result.recommendations[0])
            Consider applying temperature scaling to improve calibration

            Error handling:

            >>> try:
            ...     analyzer.analyze([0.9, 0.8], [True])  # Mismatched lengths
            ... except ValueError as e:
            ...     print("Error:", e)
            Error: Confidences and correct must have same length
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

        return ConfidenceCalibrationResult(
            predictions=confidences,
            confidences=confidences,
            labels=labels,
            metrics=metrics,
            reliability_diagram=reliability_diagram,
            recommendations=recommendations,
        )

    def _create_bins(
        self,
        confidences: list[float],
        labels: list[int],
    ) -> list[CalibrationBin]:
        """Create equal-width calibration bins and populate with samples.

        Partitions the [0, 1] confidence interval into n_bins equal-width
        segments and assigns each (confidence, label) pair to the appropriate
        bin based on the confidence value.

        Args:
            confidences: List of confidence scores in [0, 1].
            labels: List of binary labels (0 or 1).

        Returns:
            List of CalibrationBin objects with populated statistics.

        Examples:
            >>> analyzer = CalibrationAnalyzer(n_bins=5)
            >>> bins = analyzer._create_bins([0.9, 0.1], [1, 0])
            >>> bins[0].count  # Bin [0.0, 0.2) has 1 sample
            1
            >>> bins[4].count  # Bin [0.8, 1.0] has 1 sample
            1
        """
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
        confidences: list[float],
        labels: list[int],
        bins: list[CalibrationBin],
    ) -> CalibrationMetrics:
        """Calculate all calibration metrics from binned data.

        Computes ECE, MCE, ACE, over/underconfidence, Brier score, and log loss.

        Args:
            confidences: List of confidence scores.
            labels: List of binary labels.
            bins: List of populated CalibrationBin objects.

        Returns:
            CalibrationMetrics object with all computed metrics.

        Examples:
            >>> analyzer = CalibrationAnalyzer()
            >>> bins = analyzer._create_bins([0.9, 0.8], [1, 1])
            >>> metrics = analyzer._calculate_metrics([0.9, 0.8], [1, 1], bins)
            >>> metrics.ece  # Should be low for well-calibrated
            0.15
        """
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
        brier_score = (
            sum((conf - label) ** 2 for conf, label in zip(confidences, labels)) / n
            if n > 0
            else 0.0
        )

        # Log loss (with clipping to avoid log(0))
        eps = 1e-15
        log_loss = (
            -sum(
                label * math.log(max(conf, eps)) + (1 - label) * math.log(max(1 - conf, eps))
                for conf, label in zip(confidences, labels)
            )
            / n
            if n > 0
            else 0.0
        )

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
        bins: list[CalibrationBin],
    ) -> dict[str, list[float]]:
        """Create data structure for reliability diagram visualization.

        Extracts bin statistics in a format suitable for plotting with
        matplotlib or other visualization libraries.

        Args:
            bins: List of CalibrationBin objects.

        Returns:
            Dictionary with keys:
            - bin_centers: Center point of each bin interval
            - accuracies: Average accuracy in each bin
            - confidences: Average confidence in each bin
            - counts: Number of samples in each bin
            - gaps: Calibration gap for each bin

        Examples:
            >>> analyzer = CalibrationAnalyzer(n_bins=5)
            >>> bins = analyzer._create_bins([0.9, 0.1], [1, 0])
            >>> diagram = analyzer._create_reliability_diagram(bins)
            >>> diagram['bin_centers']
            [0.1, 0.3, 0.5, 0.7, 0.9]
        """
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
    ) -> list[str]:
        """Generate actionable recommendations based on calibration metrics.

        Analyzes the computed metrics and produces specific suggestions for
        improving model calibration.

        Args:
            metrics: Computed CalibrationMetrics object.

        Returns:
            List of recommendation strings. Returns a positive message if
            model is already well calibrated.

        Examples:
            >>> metrics = CalibrationMetrics(0.15, 0.25, 0.12, 0.12, 0.03, 0.22, 0.55)
            >>> analyzer = CalibrationAnalyzer()
            >>> recs = analyzer._generate_recommendations(metrics)
            >>> 'temperature scaling' in recs[0].lower()
            True
        """
        recommendations = []

        if metrics.ece > 0.1:
            recommendations.append("Consider applying temperature scaling to improve calibration")

        if metrics.overconfidence > metrics.underconfidence * 2:
            recommendations.append("Model is significantly overconfident - increase temperature")

        if metrics.underconfidence > metrics.overconfidence * 2:
            recommendations.append(
                "Model is underconfident - decrease temperature or use Platt scaling"
            )

        if metrics.mce > 0.3:
            recommendations.append(
                "High maximum calibration error - check for outlier confidence bins"
            )

        if metrics.brier_score > 0.25:
            recommendations.append("High Brier score suggests poor overall probability estimates")

        if not recommendations:
            recommendations.append("Model appears reasonably well calibrated")

        return recommendations


class TemperatureScaler:
    """Applies temperature scaling for post-hoc calibration of neural network outputs.

    Temperature scaling is a simple but effective calibration method that divides
    the logits by a learned temperature parameter T before applying softmax.
    This single-parameter method preserves the ranking of predictions while
    adjusting confidence levels.

    For T > 1: Reduces confidence (softens probabilities) - useful for
               overconfident models (most neural networks).
    For T < 1: Increases confidence (sharpens probabilities) - useful for
               underconfident models.
    For T = 1: No change to original probabilities.

    The optimal temperature is learned by minimizing negative log-likelihood
    (NLL) on a held-out validation set using gradient descent.

    Attributes:
        temperature: The temperature parameter. Values > 1 reduce confidence,
            values < 1 increase confidence. Default is 1.0 (no scaling).

    References:
        Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)

    Examples:
        Basic usage with manual temperature:

        >>> scaler = TemperatureScaler(temperature=1.5)
        >>> raw_confidence = 0.95
        >>> calibrated = scaler.scale(raw_confidence)
        >>> print(f"Raw: {raw_confidence:.2f}, Calibrated: {calibrated:.3f}")
        Raw: 0.95, Calibrated: 0.864

        Reducing overconfidence:

        >>> scaler = TemperatureScaler(temperature=2.0)
        >>> overconfident_preds = [0.99, 0.95, 0.90, 0.85]
        >>> calibrated = scaler.scale_batch(overconfident_preds)
        >>> print([f"{c:.3f}" for c in calibrated])
        ['0.962', '0.864', '0.768', '0.672']

        Learning optimal temperature from validation data:

        >>> scaler = TemperatureScaler()
        >>> # Convert confidences to logits for fitting
        >>> import math
        >>> confidences = [0.9, 0.85, 0.75, 0.65, 0.55]
        >>> logits = [math.log(c / (1-c)) for c in confidences]
        >>> labels = [1, 1, 0, 1, 0]  # Ground truth
        >>> optimal_temp = scaler.fit(logits, labels)
        >>> print(f"Optimal temperature: {optimal_temp:.2f}")
        Optimal temperature: ...

        End-to-end calibration workflow:

        >>> # Step 1: Fit on validation set
        >>> scaler = TemperatureScaler()
        >>> val_logits = [2.0, 1.5, 0.5, -0.5, -1.5]  # Raw model outputs
        >>> val_labels = [1, 1, 1, 0, 0]
        >>> scaler.fit(val_logits, val_labels)
        ...
        >>> # Step 2: Apply to test set predictions
        >>> test_confidences = [0.88, 0.75, 0.62]
        >>> calibrated = scaler.scale_batch(test_confidences)

    See Also:
        PlattScaler: Two-parameter sigmoid calibration
        Calibrator: Unified interface for multiple calibration methods
    """

    def __init__(self, temperature: float = 1.0):
        """Initialize the temperature scaler.

        Args:
            temperature: Initial temperature value. Use > 1 for overconfident
                models, < 1 for underconfident models. Will be updated if
                fit() is called. Default is 1.0.

        Examples:
            >>> scaler = TemperatureScaler()  # Default T=1.0
            >>> scaler = TemperatureScaler(temperature=1.5)  # For overconfident models
            >>> scaler = TemperatureScaler(temperature=0.8)  # For underconfident models
        """
        self.temperature = temperature
        self._fitted = False

    def fit(
        self,
        logits: list[float],
        labels: list[int],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> float:
        """Fit optimal temperature using gradient descent on negative log-likelihood.

        Learns the temperature parameter that minimizes NLL on the provided
        validation data. The optimization uses simple gradient descent with
        a lower bound of 0.1 to prevent numerical issues.

        Args:
            logits: Raw logits or log-odds from the model. These should be
                the values before sigmoid/softmax is applied. For confidences,
                convert using: logit = log(p / (1-p)).
            labels: Ground truth binary labels (0 or 1). Must have same length
                as logits.
            n_iterations: Number of gradient descent iterations. More iterations
                may improve fit but increase computation time. Default is 100.
            learning_rate: Step size for gradient descent. Smaller values are
                more stable but converge slower. Default is 0.01.

        Returns:
            The optimal temperature value found by optimization.

        Raises:
            ValueError: If logits is empty or has different length than labels.

        Examples:
            Basic fitting:

            >>> scaler = TemperatureScaler()
            >>> logits = [2.0, 1.0, 0.5, -0.5, -1.0, -2.0]
            >>> labels = [1, 1, 1, 0, 0, 0]
            >>> temp = scaler.fit(logits, labels)
            >>> print(f"Learned temperature: {temp:.2f}")
            Learned temperature: ...

            Fitting with custom hyperparameters:

            >>> scaler = TemperatureScaler()
            >>> temp = scaler.fit(
            ...     logits=[3.0, 2.0, -2.0, -3.0],
            ...     labels=[1, 1, 0, 0],
            ...     n_iterations=200,
            ...     learning_rate=0.005
            ... )

            After fitting, use scale() or scale_batch():

            >>> scaler = TemperatureScaler()
            >>> scaler.fit([2.0, -2.0], [1, 0])
            ...
            >>> calibrated = scaler.scale(0.9)
        """
        if not logits or len(logits) != len(labels):
            raise ValueError("Invalid input")

        # Simple gradient descent on negative log likelihood
        temp = 1.0

        for _ in range(n_iterations):
            # Forward pass: softmax with temperature
            scaled_probs = [self._sigmoid(logit_value / temp) for logit_value in logits]

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
        """Scale a single confidence score using the temperature parameter.

        Converts confidence to logit, divides by temperature, and converts
        back to probability. Handles edge cases where confidence is exactly
        0 or 1 by returning the input unchanged.

        Args:
            confidence: Confidence score in range [0, 1].

        Returns:
            Temperature-scaled confidence score.

        Examples:
            Reducing high confidence (T > 1):

            >>> scaler = TemperatureScaler(temperature=2.0)
            >>> scaler.scale(0.95)
            0.8175744761936437

            Increasing low confidence (T < 1):

            >>> scaler = TemperatureScaler(temperature=0.5)
            >>> scaler.scale(0.6)
            0.7310585786300049

            Edge cases:

            >>> scaler = TemperatureScaler(temperature=2.0)
            >>> scaler.scale(0.0)  # Returns unchanged
            0.0
            >>> scaler.scale(1.0)  # Returns unchanged
            1.0
        """
        if confidence <= 0 or confidence >= 1:
            return confidence

        # Convert to logit, scale, convert back
        logit = math.log(confidence / (1 - confidence))
        scaled_logit = logit / self.temperature
        return self._sigmoid(scaled_logit)

    def scale_batch(self, confidences: list[float]) -> list[float]:
        """Scale multiple confidence scores using the temperature parameter.

        Applies temperature scaling to each confidence in the list.

        Args:
            confidences: List of confidence scores in range [0, 1].

        Returns:
            List of temperature-scaled confidence scores.

        Examples:
            >>> scaler = TemperatureScaler(temperature=1.5)
            >>> calibrated = scaler.scale_batch([0.9, 0.7, 0.5, 0.3, 0.1])
            >>> [round(c, 3) for c in calibrated]
            [0.829, 0.618, 0.5, 0.382, 0.171]
        """
        return [self.scale(c) for c in confidences]

    def _sigmoid(self, x: float) -> float:
        """Compute sigmoid function with numerical stability.

        Uses different formulations for positive and negative inputs to
        avoid overflow in exp().

        Args:
            x: Input value.

        Returns:
            Sigmoid(x) = 1 / (1 + exp(-x)).

        Examples:
            >>> scaler = TemperatureScaler()
            >>> scaler._sigmoid(0)
            0.5
            >>> round(scaler._sigmoid(2), 4)
            0.8808
        """
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)


class PlattScaler:
    """Applies Platt scaling (sigmoid calibration) for probability calibration.

    Platt scaling fits a logistic regression model on top of the classifier
    outputs to produce calibrated probabilities. It learns two parameters
    (a, b) and maps scores to probabilities via: P(y=1|s) = sigmoid(a*s + b).

    This method was originally developed for calibrating SVM outputs but works
    well for any classifier that produces real-valued scores. It is more
    flexible than temperature scaling as it can correct both miscalibration
    and systematic bias in the scores.

    Attributes:
        a: Slope parameter for the sigmoid transformation. Positive values
            preserve score ordering; negative values would reverse it.
        b: Intercept parameter for the sigmoid transformation. Shifts the
            decision boundary.

    References:
        Platt, "Probabilistic Outputs for Support Vector Machines" (1999)

    Examples:
        Basic usage with default initialization:

        >>> scaler = PlattScaler()
        >>> scores = [2.0, 1.5, 0.5, -0.5, -1.5, -2.0]
        >>> labels = [1, 1, 1, 0, 0, 0]
        >>> a, b = scaler.fit(scores, labels)
        >>> print(f"Parameters: a={a:.3f}, b={b:.3f}")
        Parameters: a=..., b=...

        Calibrating SVM-like outputs:

        >>> scaler = PlattScaler()
        >>> # SVM decision values (distance from hyperplane)
        >>> svm_outputs = [1.5, 0.8, 0.3, -0.2, -0.9, -1.4]
        >>> ground_truth = [1, 1, 1, 0, 0, 0]
        >>> scaler.fit(svm_outputs, ground_truth)
        (...)
        >>> # Convert new predictions to probabilities
        >>> new_outputs = [0.5, -0.3]
        >>> probabilities = scaler.scale_batch(new_outputs)

        Calibrating neural network outputs:

        >>> scaler = PlattScaler()
        >>> # Raw logits from neural network
        >>> logits = [3.0, 2.0, 1.0, -1.0, -2.0, -3.0]
        >>> labels = [1, 1, 0, 1, 0, 0]  # Some noise in labels
        >>> scaler.fit(logits, labels)
        (...)
        >>> calibrated = scaler.scale(2.5)
        >>> print(f"Calibrated probability: {calibrated:.3f}")
        Calibrated probability: ...

        Comparing raw and calibrated scores:

        >>> scaler = PlattScaler()
        >>> scaler.fit([2.0, 1.0, 0.0, -1.0, -2.0], [1, 1, 1, 0, 0])
        (...)
        >>> raw_score = 1.5
        >>> calibrated = scaler.scale(raw_score)
        >>> print(f"Raw: {raw_score}, Calibrated: {calibrated:.3f}")
        Raw: 1.5, Calibrated: ...

    See Also:
        TemperatureScaler: Single-parameter calibration
        HistogramBinner: Non-parametric calibration
    """

    def __init__(self):
        """Initialize the Platt scaler with default parameters.

        The parameters a=1.0 and b=0.0 correspond to the identity
        transformation (sigmoid of score). These will be updated when
        fit() is called.

        Examples:
            >>> scaler = PlattScaler()
            >>> scaler.a, scaler.b
            (1.0, 0.0)
        """
        self.a = 1.0
        self.b = 0.0
        self._fitted = False

    def fit(
        self,
        scores: list[float],
        labels: list[int],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> tuple[float, float]:
        """Fit Platt scaling parameters using gradient descent.

        Learns the (a, b) parameters by minimizing negative log-likelihood
        on the provided training data.

        Args:
            scores: Model output scores (e.g., SVM decision values, logits).
                Can be any real-valued numbers.
            labels: Ground truth binary labels (0 or 1). Must have same
                length as scores.
            n_iterations: Number of gradient descent iterations. Default is 100.
            learning_rate: Step size for gradient descent. Default is 0.01.

        Returns:
            Tuple of (a, b) parameters after fitting.

        Raises:
            ValueError: If scores is empty or has different length than labels.

        Examples:
            Standard fitting:

            >>> scaler = PlattScaler()
            >>> a, b = scaler.fit(
            ...     scores=[2.0, 1.0, 0.0, -1.0, -2.0],
            ...     labels=[1, 1, 1, 0, 0]
            ... )

            With custom hyperparameters for better convergence:

            >>> scaler = PlattScaler()
            >>> a, b = scaler.fit(
            ...     scores=[3.0, 1.5, -1.5, -3.0],
            ...     labels=[1, 1, 0, 0],
            ...     n_iterations=200,
            ...     learning_rate=0.005
            ... )

            Checking fit status:

            >>> scaler = PlattScaler()
            >>> scaler._fitted
            False
            >>> scaler.fit([1.0, -1.0], [1, 0])
            (...)
            >>> scaler._fitted
            True
        """
        if not scores or len(scores) != len(labels):
            raise ValueError("Invalid input")

        a, b = 1.0, 0.0

        for _ in range(n_iterations):
            # Compute probabilities
            probs = [self._sigmoid(a * s + b) for s in scores]

            # Compute gradients
            grad_a = sum((p - label) * s for p, label, s in zip(probs, labels, scores)) / len(
                scores
            )
            grad_b = sum(p - label for p, label in zip(probs, labels)) / len(scores)

            # Update parameters
            a -= learning_rate * grad_a
            b -= learning_rate * grad_b

        self.a = a
        self.b = b
        self._fitted = True
        return a, b

    def scale(self, score: float) -> float:
        """Scale a single score to a calibrated probability.

        Applies the learned sigmoid transformation: P = sigmoid(a*score + b).

        Args:
            score: Raw model output score (any real number).

        Returns:
            Calibrated probability in range [0, 1].

        Examples:
            >>> scaler = PlattScaler()
            >>> scaler.fit([2.0, 1.0, -1.0, -2.0], [1, 1, 0, 0])
            (...)
            >>> prob = scaler.scale(0.5)
            >>> 0 <= prob <= 1
            True

            >>> # Without fitting, uses identity sigmoid
            >>> scaler = PlattScaler()
            >>> round(scaler.scale(0), 3)
            0.5
        """
        return self._sigmoid(self.a * score + self.b)

    def scale_batch(self, scores: list[float]) -> list[float]:
        """Scale multiple scores to calibrated probabilities.

        Args:
            scores: List of raw model output scores.

        Returns:
            List of calibrated probabilities.

        Examples:
            >>> scaler = PlattScaler()
            >>> scaler.fit([2.0, -2.0], [1, 0])
            (...)
            >>> probs = scaler.scale_batch([1.5, 0.0, -1.5])
            >>> all(0 <= p <= 1 for p in probs)
            True
        """
        return [self.scale(s) for s in scores]

    def _sigmoid(self, x: float) -> float:
        """Compute sigmoid function with numerical stability.

        Uses different formulations for positive and negative inputs to
        avoid overflow in exp().

        Args:
            x: Input value.

        Returns:
            Sigmoid(x) = 1 / (1 + exp(-x)).

        Examples:
            >>> scaler = PlattScaler()
            >>> scaler._sigmoid(0)
            0.5
            >>> round(scaler._sigmoid(-10), 6)
            4.5e-05
        """
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)


class HistogramBinner:
    """Applies histogram binning for non-parametric calibration.

    Histogram binning is a simple non-parametric calibration method that
    partitions the confidence space into bins and replaces each confidence
    with the empirical accuracy of predictions in that bin. This directly
    enforces calibration within each bin.

    The method works by:
    1. Dividing [0, 1] into n_bins equal-width intervals
    2. Computing the accuracy of predictions in each bin
    3. Mapping any new confidence to the accuracy of its bin

    Advantages:
        - Simple and interpretable
        - Makes no parametric assumptions
        - Guarantees calibration within bins

    Disadvantages:
        - Produces only n_bins distinct calibrated values
        - Requires sufficient samples per bin for reliable estimates
        - May not preserve ranking of predictions

    Attributes:
        n_bins: Number of equal-width bins. Default is 10.
        bin_accuracies: Learned accuracy for each bin. Empty until fit() is called.

    References:
        Zadrozny & Elkan, "Obtaining calibrated probability estimates from
        decision trees and naive Bayesian classifiers" (ICML 2001)

    Examples:
        Basic histogram binning:

        >>> binner = HistogramBinner(n_bins=10)
        >>> confidences = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25]
        >>> labels = [1, 1, 1, 0, 1, 0, 0, 0]
        >>> bin_accs = binner.fit(confidences, labels)
        >>> calibrated = binner.calibrate(0.90)
        >>> print(f"Calibrated confidence: {calibrated}")
        Calibrated confidence: 1.0

        Using fewer bins for small datasets:

        >>> binner = HistogramBinner(n_bins=5)
        >>> confidences = [0.9, 0.7, 0.5, 0.3, 0.1]
        >>> labels = [1, 1, 0, 0, 0]
        >>> binner.fit(confidences, labels)
        [...]
        >>> # New predictions get mapped to bin accuracies
        >>> binner.calibrate(0.85)  # Maps to high-confidence bin
        1.0

        Batch calibration:

        >>> binner = HistogramBinner(n_bins=10)
        >>> binner.fit([0.9, 0.7, 0.5, 0.3, 0.1], [1, 1, 0, 0, 0])
        [...]
        >>> test_confs = [0.95, 0.75, 0.55, 0.35, 0.15]
        >>> calibrated = binner.calibrate_batch(test_confs)

        Inspecting bin accuracies:

        >>> binner = HistogramBinner(n_bins=5)
        >>> binner.fit([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], [1, 1, 0, 0, 1, 0])
        [...]
        >>> for i, acc in enumerate(binner.bin_accuracies):
        ...     print(f"Bin {i}: accuracy={acc:.2f}")
        Bin 0: accuracy=...
        Bin 1: accuracy=...
        ...

    See Also:
        TemperatureScaler: Parametric single-parameter calibration
        PlattScaler: Parametric two-parameter calibration
    """

    def __init__(self, n_bins: int = 10):
        """Initialize the histogram binner.

        Args:
            n_bins: Number of equal-width bins for calibration. More bins
                provide finer calibration but require more training samples
                for reliable per-bin accuracy estimates. Default is 10.

        Examples:
            >>> binner = HistogramBinner()  # Default 10 bins
            >>> binner = HistogramBinner(n_bins=20)  # Finer granularity
            >>> binner = HistogramBinner(n_bins=5)   # For small datasets
        """
        self.n_bins = n_bins
        self.bin_accuracies: list[float] = []
        self._fitted = False

    def fit(
        self,
        confidences: list[float],
        labels: list[int],
    ) -> list[float]:
        """Fit histogram binning by computing per-bin accuracies.

        Partitions the training data into bins based on confidence scores
        and computes the empirical accuracy in each bin. For empty bins,
        uses the bin center as a fallback (assumes calibration).

        Args:
            confidences: List of confidence scores in range [0, 1].
            labels: Ground truth binary labels (0 or 1). Must have same
                length as confidences.

        Returns:
            List of bin accuracies (one per bin).

        Raises:
            ValueError: If confidences is empty or has different length
                than labels.

        Examples:
            Standard fitting:

            >>> binner = HistogramBinner(n_bins=5)
            >>> accs = binner.fit(
            ...     confidences=[0.9, 0.7, 0.5, 0.3, 0.1],
            ...     labels=[1, 1, 0, 0, 0]
            ... )
            >>> len(accs)
            5

            Examining learned accuracies:

            >>> binner = HistogramBinner(n_bins=5)
            >>> binner.fit([0.95, 0.85, 0.15, 0.05], [1, 1, 0, 0])
            [...]
            >>> binner.bin_accuracies[4]  # High confidence bin
            1.0
            >>> binner.bin_accuracies[0]  # Low confidence bin
            0.0

            Empty bins use bin center as default:

            >>> binner = HistogramBinner(n_bins=10)
            >>> binner.fit([0.95, 0.05], [1, 0])  # Only 2 samples
            [...]
            >>> binner.bin_accuracies[5]  # Empty bin, uses center
            0.55
        """
        if not confidences or len(confidences) != len(labels):
            raise ValueError("Invalid input")

        1.0 / self.n_bins
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
        """Calibrate a single confidence score using histogram binning.

        Maps the confidence to its corresponding bin and returns the
        learned accuracy for that bin.

        Args:
            confidence: Confidence score in range [0, 1].

        Returns:
            Calibrated confidence (empirical accuracy of the bin).

        Raises:
            RuntimeError: If called before fit().

        Examples:
            >>> binner = HistogramBinner(n_bins=5)
            >>> binner.fit([0.9, 0.7, 0.3, 0.1], [1, 1, 0, 0])
            [...]
            >>> binner.calibrate(0.85)  # Maps to high-confidence bin
            1.0

            >>> binner.calibrate(0.15)  # Maps to low-confidence bin
            0.0

            Error if not fitted:

            >>> binner = HistogramBinner()
            >>> try:
            ...     binner.calibrate(0.5)
            ... except RuntimeError as e:
            ...     print("Error:", e)
            Error: Must fit before calibrating
        """
        if not self._fitted:
            raise RuntimeError("Must fit before calibrating")

        bin_idx = min(int(confidence * self.n_bins), self.n_bins - 1)
        return self.bin_accuracies[bin_idx]

    def calibrate_batch(self, confidences: list[float]) -> list[float]:
        """Calibrate multiple confidence scores using histogram binning.

        Args:
            confidences: List of confidence scores in range [0, 1].

        Returns:
            List of calibrated confidences.

        Examples:
            >>> binner = HistogramBinner(n_bins=5)
            >>> binner.fit([0.9, 0.7, 0.3, 0.1], [1, 1, 0, 0])
            [...]
            >>> calibrated = binner.calibrate_batch([0.95, 0.55, 0.15])
            >>> len(calibrated)
            3
        """
        return [self.calibrate(c) for c in confidences]


class ConfidenceEstimator:
    """Estimates model confidence from various sources including logprobs and text.

    This class provides multiple methods for extracting confidence estimates from
    language model outputs. Different sources have different characteristics:

    - **Logprobs**: Direct probability estimates from the model. Often the most
      accurate source when available, but may not be exposed by all APIs.
    - **Consistency**: Agreement across multiple samples. Robust but expensive.
    - **Verbalized**: Parsed from hedging language in responses. Interpretable
      but may not reflect true model uncertainty.
    - **Entropy**: Derived from token probability distributions. Low entropy
      indicates high model confidence.

    The estimator uses conservative defaults: when detecting verbalized uncertainty,
    it uses the lowest confidence among detected patterns.

    Attributes:
        verbalized_patterns: Dictionary mapping uncertainty phrases to confidence
            scores. Phrases like "certain" map to high confidence (0.95), while
            "don't know" maps to low confidence (0.15).

    Examples:
        Estimating confidence from log probabilities:

        >>> estimator = ConfidenceEstimator()
        >>> logprobs = [-0.1, -0.2, -0.15, -0.3]  # Log probs for tokens
        >>> estimate = estimator.from_logprobs(logprobs, aggregate="mean")
        >>> print(f"Confidence: {estimate.value:.3f}")
        Confidence: 0.835

        Detecting verbalized uncertainty in responses:

        >>> estimator = ConfidenceEstimator()
        >>> response = "I think the answer might be 42, but I'm not entirely sure."
        >>> estimate = estimator.from_verbalized(response)
        >>> print(f"Confidence: {estimate.value:.2f}")
        Confidence: 0.35
        >>> print(f"Patterns: {estimate.metadata['patterns_found']}")
        Patterns: ['might', 'not sure']

        Measuring consistency across multiple samples:

        >>> estimator = ConfidenceEstimator()
        >>> responses = ["Paris", "Paris", "Paris", "Lyon"]
        >>> estimate = estimator.from_consistency(responses)
        >>> print(f"Consistency-based confidence: {estimate.value:.2f}")
        Consistency-based confidence: 0.50

        Using entropy-based confidence:

        >>> estimator = ConfidenceEstimator()
        >>> # Token probability distributions (top-k probs for each position)
        >>> token_probs = [
        ...     [0.9, 0.05, 0.05],  # Very confident first token
        ...     [0.7, 0.2, 0.1],    # Moderately confident
        ...     [0.33, 0.33, 0.34], # Uncertain
        ... ]
        >>> estimate = estimator.from_entropy(token_probs)
        >>> print(f"Entropy-based confidence: {estimate.value:.2f}")
        Entropy-based confidence: ...

    See Also:
        ConfidenceEstimate: Container for confidence values with metadata
        ConfidenceSource: Enum of available confidence sources
    """

    def __init__(self):
        """Initialize the confidence estimator with default verbalized patterns.

        The default patterns map common uncertainty expressions to confidence
        values. These can be customized by modifying verbalized_patterns after
        initialization.

        Examples:
            >>> estimator = ConfidenceEstimator()
            >>> estimator.verbalized_patterns["certain"]
            0.95

            Customizing patterns:

            >>> estimator = ConfidenceEstimator()
            >>> estimator.verbalized_patterns["absolutely"] = 0.99
            >>> estimator.verbalized_patterns["perhaps"] = 0.4
        """
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
        logprobs: list[float],
        aggregate: str = "mean",
    ) -> ConfidenceEstimate:
        """Estimate confidence from token log probabilities.

        Converts log probabilities to probabilities and aggregates them into
        a single confidence score. Different aggregation methods are useful
        for different scenarios.

        Args:
            logprobs: List of log probabilities (negative values). Each value
                represents log(P(token)) for a token in the sequence.
            aggregate: Method to combine token probabilities:
                - 'mean': Average probability (default). Good for general use.
                - 'min': Minimum probability. Conservative; captures weakest link.
                - 'product': Geometric mean. Penalizes any low-confidence token.

        Returns:
            ConfidenceEstimate with the aggregated confidence value.

        Examples:
            Using mean aggregation:

            >>> estimator = ConfidenceEstimator()
            >>> logprobs = [-0.1, -0.2, -0.3]  # ~90%, 82%, 74% probabilities
            >>> est = estimator.from_logprobs(logprobs, aggregate="mean")
            >>> print(f"Mean confidence: {est.value:.3f}")
            Mean confidence: 0.820

            Using min for conservative estimate:

            >>> est = estimator.from_logprobs(logprobs, aggregate="min")
            >>> print(f"Min confidence: {est.value:.3f}")
            Min confidence: 0.741

            Using product (geometric mean):

            >>> est = estimator.from_logprobs(logprobs, aggregate="product")
            >>> print(f"Product confidence: {est.value:.3f}")
            Product confidence: 0.819

            Empty input returns zero confidence:

            >>> est = estimator.from_logprobs([])
            >>> est.value
            0.0
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
        responses: list[str],
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ) -> ConfidenceEstimate:
        """Estimate confidence from response consistency across multiple samples.

        Generates confidence by measuring agreement between multiple responses
        to the same prompt. High agreement suggests high confidence. This is
        based on the idea that a confident model will produce similar outputs
        when sampled multiple times.

        Args:
            responses: List of model responses to the same prompt. Should contain
                at least 2 responses for meaningful comparison.
            similarity_fn: Optional function to compute similarity between two
                responses. Takes two strings, returns float in [0, 1].
                Default is exact match (case-insensitive, whitespace-normalized).

        Returns:
            ConfidenceEstimate with average pairwise similarity as confidence.
            Returns 0.5 for single response (no comparison possible).

        Examples:
            High consistency (all same):

            >>> estimator = ConfidenceEstimator()
            >>> responses = ["42", "42", "42", "42"]
            >>> est = estimator.from_consistency(responses)
            >>> est.value
            1.0

            Mixed responses:

            >>> responses = ["Paris", "Paris", "London", "Paris"]
            >>> est = estimator.from_consistency(responses)
            >>> print(f"Consistency: {est.value:.2f}")
            Consistency: 0.50

            Using custom similarity function:

            >>> def fuzzy_match(a, b):
            ...     # Simple overlap-based similarity
            ...     a_words = set(a.lower().split())
            ...     b_words = set(b.lower().split())
            ...     if not a_words or not b_words:
            ...         return 0.0
            ...     overlap = len(a_words & b_words)
            ...     return overlap / max(len(a_words), len(b_words))
            >>> est = estimator.from_consistency(
            ...     ["The capital is Paris", "Paris is the capital"],
            ...     similarity_fn=fuzzy_match
            ... )

            Single response:

            >>> est = estimator.from_consistency(["only one"])
            >>> est.value
            0.5
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
        """Estimate confidence from hedging language in model response.

        Parses the response text for uncertainty markers like "probably",
        "maybe", "I'm not sure", etc. and maps them to confidence scores.
        Uses the lowest confidence among detected patterns (conservative).

        Args:
            response: Model response text that may contain uncertainty language.

        Returns:
            ConfidenceEstimate with confidence based on detected patterns.
            Returns 0.7 (moderate confidence) if no patterns are found.

        Examples:
            High confidence language:

            >>> estimator = ConfidenceEstimator()
            >>> est = estimator.from_verbalized("I am certain the answer is 42.")
            >>> est.value
            0.95

            Low confidence language:

            >>> est = estimator.from_verbalized("I don't know, maybe it's 42?")
            >>> print(f"Confidence: {est.value}")
            Confidence: 0.15

            Multiple patterns (uses minimum):

            >>> est = estimator.from_verbalized(
            ...     "I'm fairly confident it's probably correct, but I might be wrong."
            ... )
            >>> est.value  # Uses 'might' (0.45), the lowest
            0.45
            >>> est.metadata["patterns_found"]
            ['confident', 'probably', 'might']

            No uncertainty markers:

            >>> est = estimator.from_verbalized("The answer is 42.")
            >>> est.value  # Default moderate confidence
            0.7
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
            metadata={
                "patterns_found": [p for p in self.verbalized_patterns if p in response_lower]
            },
        )

    def from_entropy(
        self,
        token_probs: list[list[float]],
    ) -> ConfidenceEstimate:
        """Estimate confidence from token probability distribution entropy.

        Computes the entropy of probability distributions at each token position
        and converts to confidence. Low entropy (probability concentrated on
        few tokens) indicates high confidence; high entropy (spread across
        many tokens) indicates uncertainty.

        Args:
            token_probs: List of probability distributions, one per token position.
                Each inner list contains probabilities for different token choices
                (e.g., top-k candidates). Probabilities should sum to ~1.

        Returns:
            ConfidenceEstimate where confidence = 1 - normalized_entropy.
            Entropy is normalized by log(vocab_size) to give [0, 1] range.

        Examples:
            High confidence (low entropy):

            >>> estimator = ConfidenceEstimator()
            >>> token_probs = [
            ...     [0.95, 0.03, 0.02],  # Very peaked distribution
            ...     [0.90, 0.05, 0.05],
            ... ]
            >>> est = estimator.from_entropy(token_probs)
            >>> print(f"Confidence: {est.value:.2f}")
            Confidence: 0.78

            Low confidence (high entropy):

            >>> token_probs = [
            ...     [0.33, 0.33, 0.34],  # Uniform distribution
            ...     [0.33, 0.33, 0.34],
            ... ]
            >>> est = estimator.from_entropy(token_probs)
            >>> print(f"Confidence: {est.value:.2f}")
            Confidence: 0.00

            Mixed confidence:

            >>> token_probs = [
            ...     [0.9, 0.05, 0.05],   # Confident
            ...     [0.33, 0.33, 0.34],  # Uncertain
            ... ]
            >>> est = estimator.from_entropy(token_probs)
            >>> 0.3 < est.value < 0.7  # Somewhere in between
            True

            Empty input:

            >>> est = estimator.from_entropy([])
            >>> est.value
            0.0
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
    """Unified interface for applying different calibration methods.

    This class provides a consistent API for fitting and applying various
    calibration techniques. It abstracts away the differences between
    temperature scaling, Platt scaling, and histogram binning, making it
    easy to switch between methods.

    The typical workflow is:
    1. Initialize with desired method
    2. Fit on validation data (confidences + labels)
    3. Apply to new predictions using calibrate() or calibrate_batch()

    Attributes:
        method: The CalibrationMethod being used.

    Examples:
        Using temperature scaling (default):

        >>> calibrator = Calibrator()
        >>> train_conf = [0.9, 0.85, 0.75, 0.65, 0.55, 0.45]
        >>> train_labels = [1, 1, 1, 0, 0, 0]
        >>> calibrator.fit(train_conf, train_labels)
        >>> calibrated = calibrator.calibrate(0.95)
        >>> print(f"Calibrated: {calibrated:.3f}")
        Calibrated: ...

        Using Platt scaling:

        >>> from insideLLMs.calibration import Calibrator, CalibrationMethod
        >>> calibrator = Calibrator(method=CalibrationMethod.PLATT_SCALING)
        >>> calibrator.fit([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
        >>> calibrated = calibrator.calibrate_batch([0.95, 0.75, 0.25, 0.05])

        Using histogram binning:

        >>> calibrator = Calibrator(method=CalibrationMethod.HISTOGRAM_BINNING)
        >>> calibrator.fit(
        ...     confidences=[0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25],
        ...     labels=[1, 1, 1, 0, 1, 0, 0, 0]
        ... )
        >>> calibrated = calibrator.calibrate(0.90)

        Comparing calibration methods:

        >>> methods = [
        ...     CalibrationMethod.TEMPERATURE_SCALING,
        ...     CalibrationMethod.PLATT_SCALING,
        ...     CalibrationMethod.HISTOGRAM_BINNING,
        ... ]
        >>> confs = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        >>> labels = [1, 1, 0, 0, 0, 0]
        >>> for method in methods:
        ...     cal = Calibrator(method=method)
        ...     cal.fit(confs, labels)
        ...     result = cal.calibrate(0.85)
        ...     print(f"{method.name}: {result:.3f}")
        TEMPERATURE_SCALING: ...
        PLATT_SCALING: ...
        HISTOGRAM_BINNING: ...

    See Also:
        TemperatureScaler: Direct temperature scaling interface
        PlattScaler: Direct Platt scaling interface
        HistogramBinner: Direct histogram binning interface
        CalibrationMethod: Available calibration methods
    """

    def __init__(self, method: CalibrationMethod = CalibrationMethod.TEMPERATURE_SCALING):
        """Initialize the calibrator with the specified method.

        Args:
            method: Calibration method to use. Default is TEMPERATURE_SCALING.
                Currently supported: TEMPERATURE_SCALING, PLATT_SCALING,
                HISTOGRAM_BINNING.

        Examples:
            >>> calibrator = Calibrator()  # Default: temperature scaling
            >>> calibrator = Calibrator(CalibrationMethod.PLATT_SCALING)
            >>> calibrator = Calibrator(CalibrationMethod.HISTOGRAM_BINNING)
        """
        self.method = method
        self._scaler = None
        self._fitted = False

    def fit(
        self,
        confidences: list[float],
        labels: list[int],
    ) -> None:
        """Fit the calibration model on validation data.

        Learns calibration parameters from the provided confidence scores
        and ground truth labels. After fitting, use calibrate() or
        calibrate_batch() to apply calibration to new predictions.

        Args:
            confidences: List of raw confidence scores in range [0, 1].
            labels: Ground truth binary labels (0 or 1). Must have same
                length as confidences.

        Raises:
            ValueError: If confidences and labels have different lengths
                (propagated from underlying scaler).

        Examples:
            Standard fitting:

            >>> calibrator = Calibrator()
            >>> calibrator.fit(
            ...     confidences=[0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
            ...     labels=[1, 1, 0, 0, 0, 0]
            ... )

            Fitting with different method:

            >>> calibrator = Calibrator(CalibrationMethod.HISTOGRAM_BINNING)
            >>> calibrator.fit([0.95, 0.85, 0.15, 0.05], [1, 1, 0, 0])

            Checking if fitted:

            >>> calibrator = Calibrator()
            >>> calibrator._fitted
            False
            >>> calibrator.fit([0.9, 0.1], [1, 0])
            >>> calibrator._fitted
            True
        """
        if self.method == CalibrationMethod.TEMPERATURE_SCALING:
            # Convert confidences to logits
            logits = [math.log(c / (1 - c + 1e-15) + 1e-15) for c in confidences]
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

        Applies the learned calibration transformation to a raw confidence
        score. Must call fit() before using this method.

        Args:
            confidence: Raw confidence score in range [0, 1].

        Returns:
            Calibrated confidence score in range [0, 1].

        Raises:
            RuntimeError: If called before fit().

        Examples:
            >>> calibrator = Calibrator()
            >>> calibrator.fit([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
            >>> calibrated = calibrator.calibrate(0.85)
            >>> 0 <= calibrated <= 1
            True

            Error without fitting:

            >>> calibrator = Calibrator()
            >>> try:
            ...     calibrator.calibrate(0.5)
            ... except RuntimeError as e:
            ...     print("Error:", e)
            Error: Must fit before calibrating
        """
        if not self._fitted:
            raise RuntimeError("Must fit before calibrating")

        if (
            self.method == CalibrationMethod.TEMPERATURE_SCALING
            or self.method == CalibrationMethod.PLATT_SCALING
        ):
            return self._scaler.scale(confidence)
        elif self.method == CalibrationMethod.HISTOGRAM_BINNING:
            return self._scaler.calibrate(confidence)

        return confidence

    def calibrate_batch(self, confidences: list[float]) -> list[float]:
        """Calibrate multiple confidence scores.

        Convenience method to apply calibration to a list of confidences.

        Args:
            confidences: List of raw confidence scores in range [0, 1].

        Returns:
            List of calibrated confidence scores.

        Examples:
            >>> calibrator = Calibrator()
            >>> calibrator.fit([0.9, 0.7, 0.3, 0.1], [1, 1, 0, 0])
            >>> calibrated = calibrator.calibrate_batch([0.95, 0.75, 0.55, 0.25])
            >>> len(calibrated)
            4
            >>> all(0 <= c <= 1 for c in calibrated)
            True
        """
        return [self.calibrate(c) for c in confidences]


# Convenience functions


def analyze_calibration(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> ConfidenceCalibrationResult:
    """Analyze model calibration and compute comprehensive metrics.

    Convenience function that creates a CalibrationAnalyzer and runs analysis.
    Computes ECE, MCE, Brier score, log loss, and generates recommendations.

    Args:
        confidences: List of confidence scores in range [0, 1]. Each score
            represents the model's predicted probability of being correct.
        correct: List of boolean correctness labels. True indicates the
            prediction was correct, False indicates incorrect. Must have
            same length as confidences.
        n_bins: Number of bins for calibration analysis. More bins provide
            finer granularity but require more samples. Default is 10.

    Returns:
        ConfidenceCalibrationResult containing metrics, reliability diagram
        data, and recommendations.

    Raises:
        ValueError: If confidences and correct have different lengths,
            or if lists are empty.

    Examples:
        Basic analysis:

        >>> confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        >>> correct = [True, True, True, False, True, False, False, False]
        >>> result = analyze_calibration(confidences, correct)
        >>> print(f"ECE: {result.metrics.ece:.4f}")
        ECE: 0.0750

        With custom bin count:

        >>> result = analyze_calibration(confidences, correct, n_bins=5)
        >>> len(result.reliability_diagram["bin_centers"])
        5

        Checking calibration quality:

        >>> result = analyze_calibration([0.9, 0.1], [True, False])
        >>> print(f"Quality: {result.metrics.calibration_quality}")
        Quality: excellent

        Getting recommendations:

        >>> result = analyze_calibration([0.95, 0.9, 0.85], [True, False, False])
        >>> print(result.recommendations[0])
        Consider applying temperature scaling to improve calibration

    See Also:
        CalibrationAnalyzer: Class-based interface for calibration analysis
        calculate_ece: Quick ECE calculation
    """
    analyzer = CalibrationAnalyzer(n_bins=n_bins)
    return analyzer.analyze(confidences, correct)


def calculate_ece(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE).

    ECE is the primary metric for measuring calibration. It computes the
    weighted average of the absolute difference between confidence and
    accuracy across bins, where weights are proportional to bin sample counts.

    Args:
        confidences: List of confidence scores in range [0, 1].
        correct: List of boolean correctness labels.
        n_bins: Number of bins for analysis. Default is 10.

    Returns:
        ECE value in range [0, 1]. Lower is better. Values below 0.05 are
        generally considered well-calibrated.

    Examples:
        Well-calibrated model:

        >>> confidences = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        >>> correct = [True, True, True, False, False, False]
        >>> ece = calculate_ece(confidences, correct)
        >>> ece < 0.05
        True

        Poorly calibrated (overconfident):

        >>> confidences = [0.95, 0.90, 0.85, 0.80]
        >>> correct = [True, False, True, False]  # 50% accuracy
        >>> ece = calculate_ece(confidences, correct)
        >>> ece > 0.3
        True

        With different bin counts:

        >>> ece_10 = calculate_ece([0.9, 0.5, 0.1], [True, True, False], n_bins=10)
        >>> ece_5 = calculate_ece([0.9, 0.5, 0.1], [True, True, False], n_bins=5)

    See Also:
        analyze_calibration: Full calibration analysis with all metrics
    """
    result = analyze_calibration(confidences, correct, n_bins)
    return result.metrics.ece


def calculate_brier_score(
    confidences: list[float],
    correct: list[bool],
) -> float:
    """Calculate Brier score for probability predictions.

    The Brier score is the mean squared error between confidence scores and
    binary outcomes. It measures both calibration and refinement (ability
    to produce extreme probabilities for confident predictions).

    Score interpretation:
    - 0.0: Perfect predictions
    - 0.25: Equivalent to always predicting 0.5 (random guessing)
    - 1.0: Worst possible (always wrong with certainty)

    Args:
        confidences: List of confidence scores in range [0, 1].
        correct: List of boolean correctness labels.

    Returns:
        Brier score in range [0, 1]. Lower is better.

    Raises:
        ValueError: If inputs are empty or have different lengths.

    Examples:
        Perfect predictions:

        >>> confidences = [1.0, 1.0, 0.0, 0.0]
        >>> correct = [True, True, False, False]
        >>> brier = calculate_brier_score(confidences, correct)
        >>> brier
        0.0

        Random guessing baseline:

        >>> confidences = [0.5, 0.5, 0.5, 0.5]
        >>> correct = [True, False, True, False]
        >>> brier = calculate_brier_score(confidences, correct)
        >>> brier
        0.25

        Overconfident wrong predictions:

        >>> confidences = [0.9, 0.9, 0.1, 0.1]
        >>> correct = [False, False, True, True]  # All wrong!
        >>> brier = calculate_brier_score(confidences, correct)
        >>> brier > 0.5
        True

    See Also:
        analyze_calibration: Includes Brier score among other metrics
    """
    if not confidences or len(confidences) != len(correct):
        raise ValueError("Invalid input")

    labels = [1 if c else 0 for c in correct]
    return sum((conf - label) ** 2 for conf, label in zip(confidences, labels)) / len(confidences)


def apply_temperature_scaling(
    confidences: list[float],
    temperature: float,
) -> list[float]:
    """Apply temperature scaling to confidence scores.

    Transforms confidence scores by converting to logits, dividing by
    temperature, and converting back. This adjusts the sharpness of
    probability distributions.

    Args:
        confidences: List of raw confidence scores in range [0, 1].
        temperature: Temperature parameter. Values > 1 reduce confidence
            (soften), values < 1 increase confidence (sharpen).

    Returns:
        List of temperature-scaled confidence scores.

    Examples:
        Reducing overconfidence (T > 1):

        >>> confidences = [0.95, 0.90, 0.85, 0.80]
        >>> scaled = apply_temperature_scaling(confidences, temperature=2.0)
        >>> all(s < c for s, c in zip(scaled, confidences))  # All reduced
        True

        Increasing confidence (T < 1):

        >>> confidences = [0.6, 0.7, 0.8]
        >>> scaled = apply_temperature_scaling(confidences, temperature=0.5)
        >>> all(s > c for s, c in zip(scaled, confidences))  # All increased
        True

        No change with T = 1:

        >>> confidences = [0.9, 0.5, 0.1]
        >>> scaled = apply_temperature_scaling(confidences, temperature=1.0)
        >>> scaled == confidences
        True

        Typical calibration workflow:

        >>> raw_confs = [0.98, 0.95, 0.92, 0.88]  # Overconfident
        >>> calibrated = apply_temperature_scaling(raw_confs, temperature=1.5)
        >>> print([f"{c:.3f}" for c in calibrated])
        ['0.938', '0.864', '0.787', '0.707']

    See Also:
        fit_temperature: Learn optimal temperature from data
        TemperatureScaler: Class-based interface
    """
    scaler = TemperatureScaler(temperature=temperature)
    return scaler.scale_batch(confidences)


def fit_temperature(
    confidences: list[float],
    labels: list[int],
) -> float:
    """Fit optimal temperature for calibration using validation data.

    Learns the temperature parameter that minimizes negative log-likelihood
    on the provided validation set. After fitting, use apply_temperature_scaling
    to calibrate new predictions.

    Args:
        confidences: List of confidence scores in range [0, 1].
        labels: Ground truth binary labels (0 or 1).

    Returns:
        Optimal temperature value. Typically > 1 for overconfident models.

    Examples:
        Fitting on validation data:

        >>> confidences = [0.95, 0.90, 0.85, 0.60, 0.40, 0.20]
        >>> labels = [1, 1, 0, 1, 0, 0]  # Some miscalibration
        >>> optimal_temp = fit_temperature(confidences, labels)
        >>> print(f"Optimal temperature: {optimal_temp:.2f}")
        Optimal temperature: ...

        End-to-end calibration:

        >>> # Step 1: Fit on validation set
        >>> val_confs = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        >>> val_labels = [1, 1, 0, 0, 0, 0]
        >>> temp = fit_temperature(val_confs, val_labels)
        >>> # Step 2: Apply to test set
        >>> test_confs = [0.95, 0.85, 0.75]
        >>> calibrated = apply_temperature_scaling(test_confs, temp)

        Checking if temperature reduces ECE:

        >>> confs = [0.95, 0.90, 0.85, 0.15, 0.10, 0.05]
        >>> correct = [True, True, False, False, False, False]
        >>> original_ece = calculate_ece(confs, correct)
        >>> temp = fit_temperature(confs, [1, 1, 0, 0, 0, 0])
        >>> calibrated = apply_temperature_scaling(confs, temp)
        >>> new_ece = calculate_ece(calibrated, correct)

    See Also:
        apply_temperature_scaling: Apply fitted temperature
        TemperatureScaler: Class-based interface
    """
    # Convert to logits
    logits = [math.log(c / (1 - c + 1e-15) + 1e-15) for c in confidences]
    scaler = TemperatureScaler()
    return scaler.fit(logits, labels)


def estimate_confidence_from_consistency(
    responses: list[str],
) -> float:
    """Estimate confidence from agreement across multiple model responses.

    Measures confidence by computing pairwise similarity between responses.
    High consistency (similar responses) suggests high model confidence.

    Args:
        responses: List of model responses to the same prompt. Should have
            at least 2 responses for meaningful comparison.

    Returns:
        Confidence estimate in range [0, 1] based on average pairwise
        similarity. Returns 0.5 for single response.

    Examples:
        High consistency:

        >>> responses = ["Paris", "Paris", "Paris", "Paris"]
        >>> conf = estimate_confidence_from_consistency(responses)
        >>> conf
        1.0

        Low consistency:

        >>> responses = ["Paris", "London", "Berlin", "Madrid"]
        >>> conf = estimate_confidence_from_consistency(responses)
        >>> conf
        0.0

        Mixed consistency:

        >>> responses = ["Paris", "Paris", "London", "Paris"]
        >>> conf = estimate_confidence_from_consistency(responses)
        >>> print(f"Confidence: {conf:.2f}")
        Confidence: 0.50

        Single response (undefined):

        >>> conf = estimate_confidence_from_consistency(["only one"])
        >>> conf
        0.5

    See Also:
        ConfidenceEstimator: Class with more options for similarity functions
    """
    estimator = ConfidenceEstimator()
    result = estimator.from_consistency(responses)
    return result.value


def estimate_confidence_from_verbalized(
    response: str,
) -> float:
    """Estimate confidence from hedging language in model response.

    Parses the response for uncertainty markers like "probably", "maybe",
    "not sure", etc., and returns a confidence score based on the most
    uncertain language found.

    Args:
        response: Model response text that may contain uncertainty language.

    Returns:
        Confidence estimate in range [0, 1]. Uses the lowest confidence
        among detected patterns (conservative). Returns 0.7 if no
        uncertainty markers are found.

    Examples:
        High confidence language:

        >>> conf = estimate_confidence_from_verbalized("I am certain it's Paris.")
        >>> conf
        0.95

        Low confidence language:

        >>> conf = estimate_confidence_from_verbalized("I don't know, maybe Paris?")
        >>> print(f"Confidence: {conf:.2f}")
        Confidence: 0.15

        Multiple hedging words (uses minimum):

        >>> conf = estimate_confidence_from_verbalized(
        ...     "I'm fairly confident it's probably Paris, but I might be wrong."
        ... )
        >>> conf  # Uses 'might' (0.45)
        0.45

        No uncertainty markers:

        >>> conf = estimate_confidence_from_verbalized("The capital of France is Paris.")
        >>> conf
        0.7

    See Also:
        ConfidenceEstimator: Class with customizable patterns
    """
    estimator = ConfidenceEstimator()
    result = estimator.from_verbalized(response)
    return result.value


def calibrate_confidences(
    confidences: list[float],
    labels: list[int],
    method: CalibrationMethod = CalibrationMethod.TEMPERATURE_SCALING,
) -> list[float]:
    """Fit calibration model and apply to confidence scores.

    One-shot function that fits a calibration method on the provided data
    and returns calibrated versions of the same confidences. Useful for
    quick calibration without managing separate fit/transform steps.

    Note: For production use, fit on validation data and apply to test data
    separately using the Calibrator class.

    Args:
        confidences: List of raw confidence scores in range [0, 1].
        labels: Ground truth binary labels (0 or 1). Must have same
            length as confidences.
        method: Calibration method to use. Default is TEMPERATURE_SCALING.

    Returns:
        List of calibrated confidence scores.

    Examples:
        Using temperature scaling (default):

        >>> confidences = [0.95, 0.85, 0.75, 0.25, 0.15, 0.05]
        >>> labels = [1, 1, 0, 0, 0, 0]
        >>> calibrated = calibrate_confidences(confidences, labels)
        >>> len(calibrated)
        6

        Using Platt scaling:

        >>> calibrated = calibrate_confidences(
        ...     confidences=[0.9, 0.7, 0.3, 0.1],
        ...     labels=[1, 1, 0, 0],
        ...     method=CalibrationMethod.PLATT_SCALING
        ... )

        Using histogram binning:

        >>> calibrated = calibrate_confidences(
        ...     confidences=[0.95, 0.85, 0.75, 0.65, 0.35, 0.25, 0.15, 0.05],
        ...     labels=[1, 1, 1, 0, 0, 0, 0, 0],
        ...     method=CalibrationMethod.HISTOGRAM_BINNING
        ... )

        Checking if calibration improves ECE:

        >>> confs = [0.95, 0.90, 0.85, 0.15, 0.10, 0.05]
        >>> labels = [1, 1, 0, 0, 0, 0]
        >>> correct = [True, True, False, False, False, False]
        >>> original_ece = calculate_ece(confs, correct)
        >>> calibrated = calibrate_confidences(confs, labels)
        >>> new_ece = calculate_ece(calibrated, correct)

    See Also:
        Calibrator: Class-based interface for separate fit/transform
        fit_temperature: Fit temperature scaling only
    """
    calibrator = Calibrator(method=method)
    calibrator.fit(confidences, labels)
    return calibrator.calibrate_batch(confidences)
