"""Tests for calibration module."""

import math

import pytest

from insideLLMs.contrib.calibration import (
    CalibrationAnalyzer,
    CalibrationBin,
    CalibrationMethod,
    CalibrationMetrics,
    CalibrationResult,
    Calibrator,
    ConfidenceEstimate,
    ConfidenceEstimator,
    ConfidenceSource,
    HistogramBinner,
    PlattScaler,
    TemperatureScaler,
    analyze_calibration,
    apply_temperature_scaling,
    calculate_brier_score,
    calculate_ece,
    calibrate_confidences,
    estimate_confidence_from_consistency,
    estimate_confidence_from_verbalized,
    fit_temperature,
)


class TestCalibrationMethod:
    """Tests for CalibrationMethod enum."""

    def test_all_methods_exist(self):
        """Test that all methods exist."""
        assert CalibrationMethod.TEMPERATURE_SCALING.value == "temperature_scaling"
        assert CalibrationMethod.PLATT_SCALING.value == "platt_scaling"
        assert CalibrationMethod.HISTOGRAM_BINNING.value == "histogram_binning"


class TestConfidenceSource:
    """Tests for ConfidenceSource enum."""

    def test_all_sources_exist(self):
        """Test that all sources exist."""
        assert ConfidenceSource.LOGPROBS.value == "logprobs"
        assert ConfidenceSource.SELF_REPORTED.value == "self_reported"
        assert ConfidenceSource.CONSISTENCY.value == "consistency"
        assert ConfidenceSource.ENTROPY.value == "entropy"
        assert ConfidenceSource.VERBALIZED.value == "verbalized"


class TestCalibrationBin:
    """Tests for CalibrationBin class."""

    def test_avg_confidence(self):
        """Test average confidence calculation."""
        bin_ = CalibrationBin(
            bin_start=0.0,
            bin_end=0.1,
            confidence_sum=0.5,
            accuracy_sum=0.3,
            count=5,
        )
        assert abs(bin_.avg_confidence - 0.1) < 0.001

    def test_avg_accuracy(self):
        """Test average accuracy calculation."""
        bin_ = CalibrationBin(
            bin_start=0.0,
            bin_end=0.1,
            confidence_sum=0.5,
            accuracy_sum=0.4,
            count=5,
        )
        assert abs(bin_.avg_accuracy - 0.08) < 0.001

    def test_gap(self):
        """Test calibration gap calculation."""
        bin_ = CalibrationBin(
            bin_start=0.0,
            bin_end=0.1,
            confidence_sum=0.8,
            accuracy_sum=0.4,
            count=2,
        )
        # avg_conf = 0.4, avg_acc = 0.2, gap = 0.2
        assert abs(bin_.gap - 0.2) < 0.001

    def test_empty_bin(self):
        """Test empty bin behavior."""
        bin_ = CalibrationBin(
            bin_start=0.0,
            bin_end=0.1,
            confidence_sum=0.0,
            accuracy_sum=0.0,
            count=0,
        )
        assert bin_.avg_confidence == 0.0
        assert bin_.avg_accuracy == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        bin_ = CalibrationBin(
            bin_start=0.0,
            bin_end=0.1,
            confidence_sum=0.5,
            accuracy_sum=0.5,
            count=5,
        )
        d = bin_.to_dict()
        assert "bin_range" in d
        assert "avg_confidence" in d
        assert "gap" in d


class TestCalibrationMetrics:
    """Tests for CalibrationMetrics class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = CalibrationMetrics(
            ece=0.05,
            mce=0.1,
            ace=0.06,
            overconfidence=0.03,
            underconfidence=0.02,
            brier_score=0.15,
            log_loss=0.5,
        )
        d = metrics.to_dict()
        assert d["ece"] == 0.05
        assert d["mce"] == 0.1

    def test_is_well_calibrated(self):
        """Test well-calibrated check."""
        good_metrics = CalibrationMetrics(
            ece=0.03,
            mce=0.1,
            ace=0.04,
            overconfidence=0.01,
            underconfidence=0.02,
            brier_score=0.1,
            log_loss=0.3,
        )
        assert good_metrics.is_well_calibrated

        bad_metrics = CalibrationMetrics(
            ece=0.15,
            mce=0.3,
            ace=0.2,
            overconfidence=0.1,
            underconfidence=0.05,
            brier_score=0.3,
            log_loss=0.8,
        )
        assert not bad_metrics.is_well_calibrated

    def test_calibration_quality(self):
        """Test calibration quality description."""
        assert (
            CalibrationMetrics(
                ece=0.01,
                mce=0,
                ace=0,
                overconfidence=0,
                underconfidence=0,
                brier_score=0,
                log_loss=0,
            ).calibration_quality
            == "excellent"
        )
        assert (
            CalibrationMetrics(
                ece=0.03,
                mce=0,
                ace=0,
                overconfidence=0,
                underconfidence=0,
                brier_score=0,
                log_loss=0,
            ).calibration_quality
            == "good"
        )
        assert (
            CalibrationMetrics(
                ece=0.08,
                mce=0,
                ace=0,
                overconfidence=0,
                underconfidence=0,
                brier_score=0,
                log_loss=0,
            ).calibration_quality
            == "moderate"
        )
        assert (
            CalibrationMetrics(
                ece=0.15,
                mce=0,
                ace=0,
                overconfidence=0,
                underconfidence=0,
                brier_score=0,
                log_loss=0,
            ).calibration_quality
            == "poor"
        )
        assert (
            CalibrationMetrics(
                ece=0.25,
                mce=0,
                ace=0,
                overconfidence=0,
                underconfidence=0,
                brier_score=0,
                log_loss=0,
            ).calibration_quality
            == "very_poor"
        )


class TestConfidenceEstimate:
    """Tests for ConfidenceEstimate class."""

    def test_basic_creation(self):
        """Test basic creation."""
        estimate = ConfidenceEstimate(
            value=0.85,
            source=ConfidenceSource.LOGPROBS,
        )
        assert estimate.value == 0.85
        assert estimate.source == ConfidenceSource.LOGPROBS

    def test_to_dict(self):
        """Test conversion to dictionary."""
        estimate = ConfidenceEstimate(
            value=0.75,
            source=ConfidenceSource.CONSISTENCY,
            raw_scores=[0.8, 0.7, 0.75],
        )
        d = estimate.to_dict()
        assert d["value"] == 0.75
        assert d["source"] == "consistency"


class TestCalibrationResult:
    """Tests for CalibrationResult class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = CalibrationMetrics(
            ece=0.05,
            mce=0.1,
            ace=0.06,
            overconfidence=0.03,
            underconfidence=0.02,
            brier_score=0.15,
            log_loss=0.5,
        )
        result = CalibrationResult(
            predictions=[0.8, 0.9],
            confidences=[0.8, 0.9],
            labels=[1, 1],
            metrics=metrics,
            reliability_diagram={"bin_centers": [0.5]},
            recommendations=["Test recommendation"],
        )
        d = result.to_dict()
        assert d["n_samples"] == 2
        assert "metrics" in d


class TestCalibrationAnalyzer:
    """Tests for CalibrationAnalyzer class."""

    def test_analyze_perfectly_calibrated(self):
        """Test analysis of reasonably calibrated model."""
        analyzer = CalibrationAnalyzer(n_bins=10)
        # Reasonable calibration: lower confidence for wrong, higher for correct
        confidences = [0.2, 0.3, 0.3, 0.4, 0.7, 0.8, 0.8, 0.9, 0.9, 0.9]
        correct = [False, False, False, False, True, True, True, True, True, True]

        result = analyzer.analyze(confidences, correct)
        assert result.metrics.ece < 0.3  # Should be reasonably low

    def test_analyze_overconfident(self):
        """Test analysis of overconfident model."""
        analyzer = CalibrationAnalyzer(n_bins=5)
        # High confidence but many wrong
        confidences = [0.9] * 10
        correct = [True, True, False, False, False, False, False, False, False, False]

        result = analyzer.analyze(confidences, correct)
        assert result.metrics.overconfidence > 0

    def test_analyze_underconfident(self):
        """Test analysis of underconfident model."""
        analyzer = CalibrationAnalyzer(n_bins=5)
        # Low confidence but many correct
        confidences = [0.2] * 10
        correct = [True] * 10

        result = analyzer.analyze(confidences, correct)
        assert result.metrics.underconfidence > 0

    def test_analyze_empty_raises(self):
        """Test that empty input raises error."""
        analyzer = CalibrationAnalyzer()
        with pytest.raises(ValueError):
            analyzer.analyze([], [])

    def test_analyze_mismatched_raises(self):
        """Test that mismatched input raises error."""
        analyzer = CalibrationAnalyzer()
        with pytest.raises(ValueError):
            analyzer.analyze([0.5], [True, False])

    def test_reliability_diagram(self):
        """Test reliability diagram data."""
        analyzer = CalibrationAnalyzer(n_bins=5)
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        correct = [False, True, True, True, True]

        result = analyzer.analyze(confidences, correct)

        assert "bin_centers" in result.reliability_diagram
        assert "accuracies" in result.reliability_diagram
        assert len(result.reliability_diagram["bin_centers"]) == 5

    def test_recommendations_generated(self):
        """Test that recommendations are generated."""
        analyzer = CalibrationAnalyzer()
        confidences = [0.5] * 10
        correct = [True, False] * 5

        result = analyzer.analyze(confidences, correct)
        assert len(result.recommendations) > 0


class TestTemperatureScaler:
    """Tests for TemperatureScaler class."""

    def test_default_temperature(self):
        """Test default temperature of 1.0."""
        scaler = TemperatureScaler()
        assert scaler.temperature == 1.0

    def test_scale_with_temperature_1(self):
        """Test scaling with temperature 1 (no change)."""
        scaler = TemperatureScaler(temperature=1.0)
        result = scaler.scale(0.7)
        assert abs(result - 0.7) < 0.01

    def test_scale_with_high_temperature(self):
        """Test scaling with high temperature (less confident)."""
        scaler = TemperatureScaler(temperature=2.0)
        result = scaler.scale(0.9)
        # Higher temp should bring extreme values closer to 0.5
        assert result < 0.9
        assert result > 0.5

    def test_scale_with_low_temperature(self):
        """Test scaling with low temperature (more confident)."""
        scaler = TemperatureScaler(temperature=0.5)
        result = scaler.scale(0.7)
        # Lower temp should push values away from 0.5
        assert result > 0.7

    def test_scale_batch(self):
        """Test batch scaling."""
        scaler = TemperatureScaler(temperature=1.5)
        results = scaler.scale_batch([0.6, 0.7, 0.8])
        assert len(results) == 3

    def test_fit(self):
        """Test fitting temperature."""
        scaler = TemperatureScaler()
        logits = [1.0, 2.0, -1.0, 0.5]
        labels = [1, 1, 0, 1]
        temp = scaler.fit(logits, labels)
        assert temp > 0
        assert scaler._fitted


class TestPlattScaler:
    """Tests for PlattScaler class."""

    def test_default_params(self):
        """Test default parameters."""
        scaler = PlattScaler()
        assert scaler.a == 1.0
        assert scaler.b == 0.0

    def test_fit(self):
        """Test fitting Platt scaling."""
        scaler = PlattScaler()
        scores = [0.2, 0.4, 0.6, 0.8]
        labels = [0, 0, 1, 1]
        a, b = scaler.fit(scores, labels)
        assert scaler._fitted

    def test_scale(self):
        """Test scaling."""
        scaler = PlattScaler()
        scaler.a = 2.0
        scaler.b = -1.0
        scaler._fitted = True
        result = scaler.scale(0.5)
        # sigmoid(2*0.5 - 1) = sigmoid(0) = 0.5
        assert abs(result - 0.5) < 0.01

    def test_scale_batch(self):
        """Test batch scaling."""
        scaler = PlattScaler()
        scaler._fitted = True
        results = scaler.scale_batch([0.3, 0.5, 0.7])
        assert len(results) == 3


class TestHistogramBinner:
    """Tests for HistogramBinner class."""

    def test_fit(self):
        """Test fitting histogram binner."""
        binner = HistogramBinner(n_bins=5)
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]
        accuracies = binner.fit(confidences, labels)
        assert len(accuracies) == 5
        assert binner._fitted

    def test_calibrate(self):
        """Test calibration."""
        binner = HistogramBinner(n_bins=5)
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]
        binner.fit(confidences, labels)

        result = binner.calibrate(0.15)
        assert 0 <= result <= 1

    def test_calibrate_before_fit_raises(self):
        """Test that calibrating before fitting raises error."""
        binner = HistogramBinner()
        with pytest.raises(RuntimeError):
            binner.calibrate(0.5)

    def test_calibrate_batch(self):
        """Test batch calibration."""
        binner = HistogramBinner(n_bins=5)
        binner.fit([0.1, 0.3, 0.5, 0.7, 0.9], [0, 0, 1, 1, 1])
        results = binner.calibrate_batch([0.2, 0.4, 0.6])
        assert len(results) == 3


class TestConfidenceEstimator:
    """Tests for ConfidenceEstimator class."""

    def test_from_logprobs_mean(self):
        """Test confidence from log probs with mean aggregation."""
        estimator = ConfidenceEstimator()
        logprobs = [math.log(0.8), math.log(0.9), math.log(0.7)]
        result = estimator.from_logprobs(logprobs, aggregate="mean")
        assert result.source == ConfidenceSource.LOGPROBS
        assert 0 < result.value < 1

    def test_from_logprobs_min(self):
        """Test confidence from log probs with min aggregation."""
        estimator = ConfidenceEstimator()
        logprobs = [math.log(0.8), math.log(0.5), math.log(0.9)]
        result = estimator.from_logprobs(logprobs, aggregate="min")
        assert abs(result.value - 0.5) < 0.01

    def test_from_logprobs_empty(self):
        """Test confidence from empty log probs."""
        estimator = ConfidenceEstimator()
        result = estimator.from_logprobs([])
        assert result.value == 0.0

    def test_from_consistency_identical(self):
        """Test consistency with identical responses."""
        estimator = ConfidenceEstimator()
        responses = ["Paris", "Paris", "Paris"]
        result = estimator.from_consistency(responses)
        assert result.value == 1.0

    def test_from_consistency_different(self):
        """Test consistency with different responses."""
        estimator = ConfidenceEstimator()
        responses = ["Paris", "London", "Berlin"]
        result = estimator.from_consistency(responses)
        assert result.value == 0.0

    def test_from_consistency_mixed(self):
        """Test consistency with mixed responses."""
        estimator = ConfidenceEstimator()
        responses = ["Paris", "Paris", "London"]
        result = estimator.from_consistency(responses)
        # 2 out of 3 pairs match
        assert 0 < result.value < 1

    def test_from_consistency_single(self):
        """Test consistency with single response."""
        estimator = ConfidenceEstimator()
        result = estimator.from_consistency(["Paris"])
        assert result.value == 0.5

    def test_from_verbalized_certain(self):
        """Test verbalized confidence - certain."""
        estimator = ConfidenceEstimator()
        result = estimator.from_verbalized("I am certain the answer is Paris.")
        assert result.value > 0.9

    def test_from_verbalized_uncertain(self):
        """Test verbalized confidence - uncertain."""
        estimator = ConfidenceEstimator()
        result = estimator.from_verbalized("I'm not sure, but it might be Paris.")
        assert result.value < 0.5

    def test_from_verbalized_no_markers(self):
        """Test verbalized confidence - no markers."""
        estimator = ConfidenceEstimator()
        result = estimator.from_verbalized("The answer is Paris.")
        assert result.value == 0.7  # Default moderate confidence

    def test_from_entropy_low(self):
        """Test confidence from low entropy (high confidence)."""
        estimator = ConfidenceEstimator()
        # Very peaked distribution
        token_probs = [[0.95, 0.03, 0.02]]
        result = estimator.from_entropy(token_probs)
        assert result.value > 0.5

    def test_from_entropy_high(self):
        """Test confidence from high entropy (low confidence)."""
        estimator = ConfidenceEstimator()
        # Uniform distribution
        token_probs = [[0.25, 0.25, 0.25, 0.25]]
        result = estimator.from_entropy(token_probs)
        assert result.value < 0.5


class TestCalibrator:
    """Tests for Calibrator class."""

    def test_fit_temperature_scaling(self):
        """Test fitting with temperature scaling."""
        calibrator = Calibrator(method=CalibrationMethod.TEMPERATURE_SCALING)
        confidences = [0.6, 0.7, 0.8, 0.9]
        labels = [0, 1, 1, 1]
        calibrator.fit(confidences, labels)
        assert calibrator._fitted

    def test_fit_platt_scaling(self):
        """Test fitting with Platt scaling."""
        calibrator = Calibrator(method=CalibrationMethod.PLATT_SCALING)
        confidences = [0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1]
        calibrator.fit(confidences, labels)
        assert calibrator._fitted

    def test_fit_histogram_binning(self):
        """Test fitting with histogram binning."""
        calibrator = Calibrator(method=CalibrationMethod.HISTOGRAM_BINNING)
        confidences = [0.2, 0.4, 0.6, 0.8]
        labels = [0, 0, 1, 1]
        calibrator.fit(confidences, labels)
        assert calibrator._fitted

    def test_calibrate_before_fit_raises(self):
        """Test that calibrating before fitting raises error."""
        calibrator = Calibrator()
        with pytest.raises(RuntimeError):
            calibrator.calibrate(0.5)

    def test_calibrate(self):
        """Test calibration."""
        calibrator = Calibrator(method=CalibrationMethod.HISTOGRAM_BINNING)
        calibrator.fit([0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
        result = calibrator.calibrate(0.5)
        assert 0 <= result <= 1

    def test_calibrate_batch(self):
        """Test batch calibration."""
        calibrator = Calibrator(method=CalibrationMethod.HISTOGRAM_BINNING)
        calibrator.fit([0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
        results = calibrator.calibrate_batch([0.3, 0.5, 0.7])
        assert len(results) == 3


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_analyze_calibration(self):
        """Test analyze_calibration function."""
        confidences = [0.3, 0.5, 0.7, 0.9]
        correct = [False, True, True, True]
        result = analyze_calibration(confidences, correct)
        assert isinstance(result, CalibrationResult)

    def test_calculate_ece(self):
        """Test calculate_ece function."""
        confidences = [0.5] * 10
        correct = [True, False] * 5
        ece = calculate_ece(confidences, correct)
        assert 0 <= ece <= 1

    def test_calculate_brier_score(self):
        """Test calculate_brier_score function."""
        confidences = [0.8, 0.8, 0.2, 0.2]
        correct = [True, True, False, False]
        brier = calculate_brier_score(confidences, correct)
        # Perfect predictions: (0.8-1)^2 * 2 + (0.2-0)^2 * 2 = 0.08 + 0.08 = 0.16 / 4 = 0.04
        assert abs(brier - 0.04) < 0.01

    def test_apply_temperature_scaling(self):
        """Test apply_temperature_scaling function."""
        confidences = [0.6, 0.7, 0.8]
        scaled = apply_temperature_scaling(confidences, temperature=1.5)
        assert len(scaled) == 3

    def test_fit_temperature(self):
        """Test fit_temperature function."""
        confidences = [0.6, 0.7, 0.8, 0.9]
        labels = [0, 1, 1, 1]
        temp = fit_temperature(confidences, labels)
        assert temp > 0

    def test_estimate_confidence_from_consistency(self):
        """Test estimate_confidence_from_consistency function."""
        responses = ["Paris", "Paris", "London"]
        conf = estimate_confidence_from_consistency(responses)
        assert 0 <= conf <= 1

    def test_estimate_confidence_from_verbalized(self):
        """Test estimate_confidence_from_verbalized function."""
        response = "I'm fairly confident it's Paris"
        conf = estimate_confidence_from_verbalized(response)
        assert 0 <= conf <= 1

    def test_calibrate_confidences(self):
        """Test calibrate_confidences function."""
        confidences = [0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1]
        calibrated = calibrate_confidences(confidences, labels)
        assert len(calibrated) == 4


class TestEdgeCases:
    """Tests for edge cases."""

    def test_confidence_at_boundaries(self):
        """Test confidence at 0 and 1."""
        scaler = TemperatureScaler(temperature=1.5)
        # At exact boundaries, should return same value
        assert scaler.scale(0.0) == 0.0
        assert scaler.scale(1.0) == 1.0

    def test_single_sample_calibration(self):
        """Test calibration with single sample."""
        analyzer = CalibrationAnalyzer(n_bins=5)
        result = analyzer.analyze([0.8], [True])
        assert result.metrics.ece >= 0

    def test_all_correct_predictions(self):
        """Test all correct predictions."""
        analyzer = CalibrationAnalyzer()
        confidences = [0.9] * 10
        correct = [True] * 10
        result = analyzer.analyze(confidences, correct)
        # Should show underconfidence since 0.9 < 1.0 accuracy
        assert result.metrics.underconfidence > 0

    def test_all_wrong_predictions(self):
        """Test all wrong predictions."""
        analyzer = CalibrationAnalyzer()
        confidences = [0.9] * 10
        correct = [False] * 10
        result = analyzer.analyze(confidences, correct)
        # Should show severe overconfidence
        assert result.metrics.overconfidence > 0.5

    def test_very_small_temperature(self):
        """Test very small temperature."""
        scaler = TemperatureScaler(temperature=0.1)
        # Very small temp should push probabilities to extremes
        result = scaler.scale(0.6)
        assert result > 0.9  # Should be very high

    def test_very_large_temperature(self):
        """Test very large temperature."""
        scaler = TemperatureScaler(temperature=10.0)
        # Very large temp should push probabilities toward 0.5
        result = scaler.scale(0.9)
        assert 0.4 < result < 0.7  # Should be near 0.5
