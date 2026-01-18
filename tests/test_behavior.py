"""Tests for model behavior analysis utilities."""

import pytest

from insideLLMs.behavior import (
    BehaviorFingerprint,
    BehaviorPattern,
    BehaviorProfiler,
    CalibrationAssessor,
    CalibrationResult,
    ConsistencyAnalyzer,
    ConsistencyReport,
    PatternDetector,
    PatternMatch,
    SensitivityAnalyzer,
    SensitivityResult,
    analyze_consistency,
    analyze_sensitivity,
    assess_calibration,
    create_behavior_fingerprint,
    detect_patterns,
)


class TestBehaviorPattern:
    """Tests for BehaviorPattern enum."""

    def test_all_patterns_exist(self):
        """Test that all patterns are defined."""
        assert BehaviorPattern.HEDGING.value == "hedging"
        assert BehaviorPattern.VERBOSITY.value == "verbosity"
        assert BehaviorPattern.REPETITION.value == "repetition"
        assert BehaviorPattern.REFUSAL.value == "refusal"
        assert BehaviorPattern.UNCERTAINTY.value == "uncertainty"
        assert BehaviorPattern.OVERCONFIDENCE.value == "overconfidence"
        assert BehaviorPattern.SYCOPHANCY.value == "sycophancy"


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_basic_creation(self):
        """Test basic pattern match creation."""
        match = PatternMatch(
            pattern=BehaviorPattern.HEDGING,
            confidence=0.8,
            evidence=["i think", "maybe"],
        )

        assert match.pattern == BehaviorPattern.HEDGING
        assert match.confidence == 0.8
        assert len(match.evidence) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        match = PatternMatch(
            pattern=BehaviorPattern.REFUSAL,
            confidence=0.9,
            evidence=["i cannot"],
            location="beginning",
        )

        d = match.to_dict()
        assert d["pattern"] == "refusal"
        assert d["confidence"] == 0.9
        assert d["location"] == "beginning"


class TestPatternDetector:
    """Tests for PatternDetector."""

    def test_detect_hedging(self):
        """Test hedging detection."""
        detector = PatternDetector()
        response = "I think this might be correct, but perhaps it could also be something else. Maybe we should consider other options."

        patterns = detector.detect_patterns(response)

        hedging_patterns = [p for p in patterns if p.pattern == BehaviorPattern.HEDGING]
        assert len(hedging_patterns) > 0
        assert hedging_patterns[0].confidence > 0

    def test_detect_refusal(self):
        """Test refusal detection."""
        detector = PatternDetector()
        response = "I cannot provide information on that topic. As an AI, I'm unable to assist with this request."

        patterns = detector.detect_patterns(response)

        refusal_patterns = [p for p in patterns if p.pattern == BehaviorPattern.REFUSAL]
        assert len(refusal_patterns) > 0
        assert refusal_patterns[0].confidence >= 0.3

    def test_detect_uncertainty(self):
        """Test uncertainty detection."""
        detector = PatternDetector()
        response = "I'm not sure about this, and I don't know all the details. To the best of my knowledge, this is correct."

        patterns = detector.detect_patterns(response)

        uncertainty_patterns = [p for p in patterns if p.pattern == BehaviorPattern.UNCERTAINTY]
        assert len(uncertainty_patterns) > 0

    def test_detect_sycophancy(self):
        """Test sycophancy detection."""
        detector = PatternDetector()
        response = (
            "Great question! That's a fantastic point. I completely agree with your assessment."
        )

        patterns = detector.detect_patterns(response)

        sycophancy_patterns = [p for p in patterns if p.pattern == BehaviorPattern.SYCOPHANCY]
        assert len(sycophancy_patterns) > 0

    def test_detect_verbosity(self):
        """Test verbosity detection."""
        detector = PatternDetector()
        # Create a verbose response
        response = ". ".join(["This is a long sentence with many words"] * 20) + "."

        patterns = detector.detect_patterns(response)

        verbosity_patterns = [p for p in patterns if p.pattern == BehaviorPattern.VERBOSITY]
        assert len(verbosity_patterns) > 0

    def test_detect_repetition(self):
        """Test repetition detection."""
        detector = PatternDetector()
        response = "The quick brown fox jumped. The quick brown fox jumped again. The quick brown fox jumped once more. The quick brown fox is tired."

        patterns = detector.detect_patterns(response)

        repetition_patterns = [p for p in patterns if p.pattern == BehaviorPattern.REPETITION]
        assert len(repetition_patterns) > 0

    def test_no_patterns_in_neutral_response(self):
        """Test that neutral response has fewer patterns."""
        detector = PatternDetector()
        response = "Paris is the capital of France. It is located in northern Europe."

        patterns = detector.detect_patterns(response)

        # Should have few or no strong patterns
        strong_patterns = [p for p in patterns if p.confidence >= 0.5]
        assert len(strong_patterns) <= 1

    def test_detect_overconfidence(self):
        """Test overconfidence detection."""
        detector = PatternDetector()
        response = "This is definitely correct. The answer is absolutely 42%. There is no doubt about this fact."

        patterns = detector.detect_patterns(response)

        overconfidence_patterns = [
            p for p in patterns if p.pattern == BehaviorPattern.OVERCONFIDENCE
        ]
        assert len(overconfidence_patterns) > 0


class TestConsistencyAnalyzer:
    """Tests for ConsistencyAnalyzer."""

    def test_consistent_responses(self):
        """Test analysis of consistent responses."""
        analyzer = ConsistencyAnalyzer()
        prompt = "What is the capital of France?"
        responses = [
            "Paris is the capital of France.",
            "The capital of France is Paris.",
            "France's capital city is Paris.",
        ]

        report = analyzer.analyze(prompt, responses)

        assert report.consistency_score > 0.3
        assert "paris" in [e.lower() for e in report.common_elements]

    def test_inconsistent_responses(self):
        """Test analysis of inconsistent responses."""
        analyzer = ConsistencyAnalyzer()
        prompt = "What is a good programming language?"
        responses = [
            "Python is excellent for data science and machine learning.",
            "JavaScript dominates web development and frontend work.",
            "Rust provides memory safety and high performance.",
        ]

        report = analyzer.analyze(prompt, responses)

        # Should have lower consistency
        assert report.variance_score > 0.3

    def test_single_response(self):
        """Test with single response."""
        analyzer = ConsistencyAnalyzer()
        report = analyzer.analyze("test", ["Single response"])

        assert report.consistency_score == 1.0
        assert report.variance_score == 0.0

    def test_empty_responses(self):
        """Test with no responses."""
        analyzer = ConsistencyAnalyzer()
        report = analyzer.analyze("test", [])

        assert report.consistency_score == 0.0
        assert report.variance_score == 1.0

    def test_semantic_clusters(self):
        """Test semantic cluster estimation."""
        analyzer = ConsistencyAnalyzer()
        prompt = "test"
        responses = [
            "Answer A with details about topic one.",
            "Answer A with more about topic one.",
            "Completely different answer B about another subject.",
            "Answer B continued with different ideas.",
        ]

        report = analyzer.analyze(prompt, responses)

        # Should detect multiple clusters
        assert report.semantic_clusters >= 1


class TestConsistencyReport:
    """Tests for ConsistencyReport."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = ConsistencyReport(
            responses=["a", "b"],
            prompt="test",
            consistency_score=0.75,
            variance_score=0.25,
            common_elements=["element"],
            divergent_elements=["other"],
            semantic_clusters=2,
            dominant_response_type="standard",
        )

        d = report.to_dict()
        assert d["consistency_score"] == 0.75
        assert d["num_responses"] == 2
        assert d["semantic_clusters"] == 2


class TestBehaviorProfiler:
    """Tests for BehaviorProfiler."""

    def test_create_fingerprint(self):
        """Test fingerprint creation."""
        profiler = BehaviorProfiler()
        responses = [
            "This is a response with some hedging. I think it might be correct.",
            "Another response that is fairly straightforward.",
            "A third response with more detail and explanation of the topic at hand.",
        ]

        fingerprint = profiler.create_fingerprint("test-model", responses)

        assert fingerprint.model_id == "test-model"
        assert fingerprint.sample_size == 3
        assert fingerprint.avg_response_length > 0
        assert 0 <= fingerprint.formality_score <= 1
        assert 0 <= fingerprint.confidence_score <= 1

    def test_empty_responses(self):
        """Test with empty responses."""
        profiler = BehaviorProfiler()
        fingerprint = profiler.create_fingerprint("empty-model", [])

        assert fingerprint.sample_size == 0
        assert fingerprint.avg_response_length == 0

    def test_fingerprint_comparison(self):
        """Test comparing fingerprints."""
        fp1 = BehaviorFingerprint(
            model_id="model1",
            sample_size=10,
            avg_response_length=100,
            avg_sentence_count=5,
            hedging_frequency=0.3,
            refusal_rate=0.1,
            verbosity_score=0.2,
            formality_score=0.7,
            confidence_score=0.6,
        )

        fp2 = BehaviorFingerprint(
            model_id="model2",
            sample_size=10,
            avg_response_length=150,
            avg_sentence_count=8,
            hedging_frequency=0.5,
            refusal_rate=0.2,
            verbosity_score=0.4,
            formality_score=0.5,
            confidence_score=0.4,
        )

        diff = fp1.compare_to(fp2)

        assert diff["avg_response_length_diff"] == -50
        assert diff["hedging_frequency_diff"] == -0.2


class TestBehaviorFingerprint:
    """Tests for BehaviorFingerprint."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        fp = BehaviorFingerprint(
            model_id="test",
            sample_size=5,
            avg_response_length=100,
            avg_sentence_count=5,
            hedging_frequency=0.2,
            refusal_rate=0.1,
            verbosity_score=0.15,
            formality_score=0.6,
            confidence_score=0.7,
            common_phrases=[("the answer is", 3)],
            pattern_frequencies={"hedging": 0.2},
        )

        d = fp.to_dict()
        assert d["model_id"] == "test"
        assert d["sample_size"] == 5


class TestSensitivityAnalyzer:
    """Tests for SensitivityAnalyzer."""

    def test_low_sensitivity(self):
        """Test with low sensitivity (stable responses)."""
        analyzer = SensitivityAnalyzer()

        original_prompt = "What is 2+2?"
        original_response = "The answer is 4."
        variations = [
            "What is two plus two?",
            "Calculate 2+2",
        ]
        variation_responses = [
            "The answer is 4.",
            "4 is the result.",
        ]

        result = analyzer.analyze_sensitivity(
            original_prompt, original_response, variations, variation_responses
        )

        # Should be relatively stable
        assert result.sensitivity_score < 0.7

    def test_high_sensitivity(self):
        """Test with high sensitivity (varying responses)."""
        analyzer = SensitivityAnalyzer()

        original_prompt = "Tell me something interesting."
        original_response = "The ocean is vast and deep."
        variations = [
            "Share something fascinating.",
            "What's an interesting fact?",
        ]
        variation_responses = [
            "Mount Everest is the tallest mountain.",
            "Honey never spoils and can last thousands of years.",
        ]

        result = analyzer.analyze_sensitivity(
            original_prompt, original_response, variations, variation_responses
        )

        # Should have higher sensitivity
        assert result.sensitivity_score >= 0.3

    def test_mismatched_lengths(self):
        """Test error on mismatched lengths."""
        analyzer = SensitivityAnalyzer()

        with pytest.raises(ValueError):
            analyzer.analyze_sensitivity("prompt", "response", ["var1", "var2"], ["resp1"])


class TestSensitivityResult:
    """Tests for SensitivityResult."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = SensitivityResult(
            original_prompt="test",
            variations=["v1", "v2"],
            original_response="response",
            variation_responses=["r1", "r2"],
            sensitivity_score=0.5,
            stable_elements=["element1"],
            sensitive_elements=["element2"],
        )

        d = result.to_dict()
        assert d["sensitivity_score"] == 0.5
        assert d["num_variations"] == 2


class TestCalibrationAssessor:
    """Tests for CalibrationAssessor."""

    def test_well_calibrated(self):
        """Test well-calibrated model."""
        assessor = CalibrationAssessor()

        # Perfect calibration: confidence matches accuracy
        predictions = ["a", "b", "c", "d", "e"]
        ground_truths = ["a", "x", "c", "x", "e"]
        confidences = [0.9, 0.4, 0.8, 0.3, 0.9]  # Correct: 0.6 (3/5)

        result = assessor.assess(predictions, ground_truths, confidences)

        # Should have reasonable ECE
        assert result.expected_calibration_error <= 1.0
        assert 0 <= result.overconfidence_score <= 1.0
        assert 0 <= result.underconfidence_score <= 1.0

    def test_overconfident_model(self):
        """Test overconfident model detection."""
        assessor = CalibrationAssessor()

        # High confidence but low accuracy
        predictions = ["a", "b", "c", "d", "e"]
        ground_truths = ["x", "x", "x", "x", "e"]  # Only 1 correct
        confidences = [0.9, 0.9, 0.9, 0.9, 0.9]  # All high confidence

        result = assessor.assess(predictions, ground_truths, confidences)

        # Should detect overconfidence
        assert result.overconfidence_score > 0

    def test_underconfident_model(self):
        """Test underconfident model detection."""
        assessor = CalibrationAssessor()

        # Low confidence but high accuracy
        predictions = ["a", "b", "c", "d", "e"]
        ground_truths = ["a", "b", "c", "d", "e"]  # All correct
        confidences = [0.2, 0.3, 0.2, 0.3, 0.2]  # All low confidence

        result = assessor.assess(predictions, ground_truths, confidences)

        # Should detect underconfidence
        assert result.underconfidence_score > 0


class TestCalibrationResult:
    """Tests for CalibrationResult."""

    def test_is_well_calibrated(self):
        """Test calibration check."""
        good_result = CalibrationResult(
            confidence_levels=[0.5, 0.8],
            accuracy_at_levels=[0.48, 0.77],
            expected_calibration_error=0.03,
            overconfidence_score=0.01,
            underconfidence_score=0.02,
        )
        assert good_result.is_well_calibrated(threshold=0.1) is True

        bad_result = CalibrationResult(
            confidence_levels=[0.5, 0.8],
            accuracy_at_levels=[0.2, 0.3],
            expected_calibration_error=0.4,
            overconfidence_score=0.35,
            underconfidence_score=0.05,
        )
        assert bad_result.is_well_calibrated(threshold=0.1) is False

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = CalibrationResult(
            confidence_levels=[0.5],
            accuracy_at_levels=[0.5],
            expected_calibration_error=0.05,
            overconfidence_score=0.02,
            underconfidence_score=0.03,
        )

        d = result.to_dict()
        assert d["expected_calibration_error"] == 0.05
        assert d["is_well_calibrated"] is True


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_detect_patterns(self):
        """Test detect_patterns function."""
        patterns = detect_patterns("I think maybe this could be right.")

        assert isinstance(patterns, list)
        # Should detect hedging
        hedging = [p for p in patterns if p.pattern == BehaviorPattern.HEDGING]
        assert len(hedging) > 0

    def test_analyze_consistency(self):
        """Test analyze_consistency function."""
        report = analyze_consistency(
            "test prompt",
            ["Response one about X.", "Response two about X."],
        )

        assert isinstance(report, ConsistencyReport)
        assert report.prompt == "test prompt"

    def test_create_behavior_fingerprint(self):
        """Test create_behavior_fingerprint function."""
        fingerprint = create_behavior_fingerprint(
            "test-model",
            ["Response A.", "Response B."],
        )

        assert isinstance(fingerprint, BehaviorFingerprint)
        assert fingerprint.model_id == "test-model"

    def test_analyze_sensitivity(self):
        """Test analyze_sensitivity function."""
        result = analyze_sensitivity(
            "original prompt",
            "original response",
            ["variation 1"],
            ["variation response 1"],
        )

        assert isinstance(result, SensitivityResult)

    def test_assess_calibration(self):
        """Test assess_calibration function."""
        result = assess_calibration(
            predictions=["a", "b"],
            ground_truths=["a", "x"],
            confidences=[0.8, 0.3],
        )

        assert isinstance(result, CalibrationResult)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_response_pattern_detection(self):
        """Test pattern detection on empty response."""
        detector = PatternDetector()
        patterns = detector.detect_patterns("")

        assert patterns == []

    def test_very_short_response(self):
        """Test with very short response."""
        detector = PatternDetector()
        patterns = detector.detect_patterns("Yes.")

        # Should handle gracefully
        assert isinstance(patterns, list)

    def test_unicode_in_response(self):
        """Test with unicode characters."""
        detector = PatternDetector()
        patterns = detector.detect_patterns("I think this answer might be correct. 日本語テスト 🎉")

        # Should still detect hedging
        hedging = [p for p in patterns if p.pattern == BehaviorPattern.HEDGING]
        assert len(hedging) > 0

    def test_single_word_responses_consistency(self):
        """Test consistency with single word responses."""
        analyzer = ConsistencyAnalyzer()
        report = analyzer.analyze("test", ["Yes", "Yes", "No"])

        assert isinstance(report, ConsistencyReport)

    def test_all_same_responses(self):
        """Test with identical responses."""
        analyzer = ConsistencyAnalyzer()
        report = analyzer.analyze(
            "test",
            ["Exact same response.", "Exact same response.", "Exact same response."],
        )

        # Should be perfectly consistent
        assert report.consistency_score == 1.0
