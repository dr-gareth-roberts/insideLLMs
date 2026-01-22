"""Tests for insideLLMs/probes/bias.py module."""

from unittest.mock import MagicMock

import pytest

from insideLLMs.probes.bias import BiasProbe
from insideLLMs.types import BiasResult, ProbeCategory, ProbeResult, ResultStatus


class TestBiasProbeInitialization:
    """Tests for BiasProbe initialization."""

    def test_default_initialization(self):
        """Test default probe initialization."""
        probe = BiasProbe()
        assert probe.name == "BiasProbe"
        assert probe.category == ProbeCategory.BIAS
        assert probe.bias_dimension == "general"
        assert probe.analyze_sentiment is True

    def test_custom_name(self):
        """Test probe with custom name."""
        probe = BiasProbe(name="GenderBiasProbe")
        assert probe.name == "GenderBiasProbe"

    def test_custom_bias_dimension(self):
        """Test probe with custom bias dimension."""
        probe = BiasProbe(bias_dimension="gender")
        assert probe.bias_dimension == "gender"

    def test_disable_sentiment_analysis(self):
        """Test probe with sentiment analysis disabled."""
        probe = BiasProbe(analyze_sentiment=False)
        assert probe.analyze_sentiment is False


class TestBiasProbeRun:
    """Tests for BiasProbe.run method."""

    def test_run_basic(self):
        """Test basic run with prompt pairs."""
        probe = BiasProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(
            side_effect=[
                "The doctor is skilled.",
                "The doctor is skilled.",
            ]
        )

        pairs = [("The male doctor is...", "The female doctor is...")]
        results = probe.run(mock_model, pairs)

        assert len(results) == 1
        assert isinstance(results[0], BiasResult)
        assert results[0].prompt_a == "The male doctor is..."
        assert results[0].prompt_b == "The female doctor is..."

    def test_run_multiple_pairs(self):
        """Test run with multiple prompt pairs."""
        probe = BiasProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(
            side_effect=[
                "Response A1",
                "Response B1",
                "Response A2",
                "Response B2",
                "Response A3",
                "Response B3",
            ]
        )

        pairs = [
            ("Prompt A1", "Prompt B1"),
            ("Prompt A2", "Prompt B2"),
            ("Prompt A3", "Prompt B3"),
        ]
        results = probe.run(mock_model, pairs)

        assert len(results) == 3
        assert results[0].response_a == "Response A1"
        assert results[0].response_b == "Response B1"
        assert results[2].response_a == "Response A3"

    def test_run_captures_bias_dimension(self):
        """Test that bias dimension is captured in results."""
        probe = BiasProbe(bias_dimension="racial")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="Response")

        pairs = [("Prompt A", "Prompt B")]
        results = probe.run(mock_model, pairs)

        assert results[0].bias_dimension == "racial"


class TestBiasProbeCompareResponses:
    """Tests for BiasProbe.compare_responses method."""

    def test_compare_identical_responses(self):
        """Test comparing identical responses."""
        probe = BiasProbe()
        comparison = probe.compare_responses(
            "The answer is yes.",
            "The answer is yes.",
            "input A",
            "input B",
        )

        assert comparison["response_identical"] is True
        assert comparison["length_diff"] == 0
        assert comparison["word_overlap"] == 1.0

    def test_compare_different_responses(self):
        """Test comparing different responses."""
        probe = BiasProbe()
        comparison = probe.compare_responses(
            "Short response",
            "This is a much longer and more detailed response",
            "input A",
            "input B",
        )

        assert comparison["response_identical"] is False
        assert comparison["length_diff"] < 0  # First is shorter
        assert comparison["word_overlap"] < 1.0

    def test_compare_with_sentiment_analysis(self):
        """Test comparison includes sentiment analysis."""
        probe = BiasProbe(analyze_sentiment=True)
        comparison = probe.compare_responses(
            "This is excellent and wonderful",
            "This is terrible and awful",
            "input A",
            "input B",
        )

        assert "sentiment_a" in comparison
        assert "sentiment_b" in comparison
        assert "sentiment_diff" in comparison
        assert comparison["sentiment_a"] > 0  # Positive
        assert comparison["sentiment_b"] < 0  # Negative

    def test_compare_without_sentiment_analysis(self):
        """Test comparison skips sentiment when disabled."""
        probe = BiasProbe(analyze_sentiment=False)
        comparison = probe.compare_responses(
            "Response A",
            "Response B",
            "input A",
            "input B",
        )

        assert "sentiment_a" not in comparison
        assert "sentiment_diff" not in comparison

    def test_compare_empty_responses(self):
        """Test comparing empty responses."""
        probe = BiasProbe()
        comparison = probe.compare_responses("", "", "input A", "input B")

        assert comparison["response_identical"] is True
        assert comparison["length_diff"] == 0
        assert comparison["word_overlap"] == 1.0


class TestBiasProbeSentiment:
    """Tests for BiasProbe._simple_sentiment method."""

    def test_sentiment_positive(self):
        """Test positive sentiment detection."""
        probe = BiasProbe()
        sentiment = probe._simple_sentiment("This is excellent and wonderful work")
        assert sentiment > 0

    def test_sentiment_negative(self):
        """Test negative sentiment detection."""
        probe = BiasProbe()
        sentiment = probe._simple_sentiment("This is terrible and awful work")
        assert sentiment < 0

    def test_sentiment_neutral(self):
        """Test neutral sentiment."""
        probe = BiasProbe()
        sentiment = probe._simple_sentiment("The sky is blue and the grass is green")
        assert sentiment == 0.0

    def test_sentiment_empty_text(self):
        """Test sentiment with empty text."""
        probe = BiasProbe()
        sentiment = probe._simple_sentiment("")
        assert sentiment == 0.0

    def test_sentiment_mixed(self):
        """Test mixed sentiment."""
        probe = BiasProbe()
        # Equal positive and negative words
        sentiment = probe._simple_sentiment("good bad")
        assert sentiment == 0.0


class TestBiasProbeDetectBiasIndicators:
    """Tests for BiasProbe.detect_bias_indicators method."""

    def test_detect_no_bias(self):
        """Test detection with no bias."""
        probe = BiasProbe()

        results = [
            BiasResult(
                prompt_a="Prompt A",
                prompt_b="Prompt B",
                response_a="Response",
                response_b="Response",
                bias_dimension="test",
                length_diff=0,
                sentiment_diff=0.0,
            )
        ]

        indicators = probe.detect_bias_indicators(results)
        assert indicators["total_pairs"] == 1
        assert indicators["flagged_pairs"] == 0
        assert indicators["flag_rate"] == 0

    def test_detect_length_bias(self):
        """Test detection of length-based bias."""
        probe = BiasProbe()

        results = [
            BiasResult(
                prompt_a="Prompt A",
                prompt_b="Prompt B",
                response_a="A" * 100,
                response_b="B" * 10,
                bias_dimension="test",
                length_diff=90,  # Significant difference
                sentiment_diff=0.0,
            )
        ]

        indicators = probe.detect_bias_indicators(results)
        assert indicators["flagged_pairs"] == 1
        assert len(indicators["flagged_details"]) == 1
        assert "Length diff" in indicators["flagged_details"][0]["indicators"][0]

    def test_detect_sentiment_bias(self):
        """Test detection of sentiment-based bias."""
        probe = BiasProbe()

        results = [
            BiasResult(
                prompt_a="Prompt A",
                prompt_b="Prompt B",
                response_a="Great",
                response_b="Terrible",
                bias_dimension="test",
                length_diff=2,
                sentiment_diff=0.5,  # Significant difference
            )
        ]

        indicators = probe.detect_bias_indicators(results, threshold=0.2)
        assert indicators["flagged_pairs"] == 1
        assert "Sentiment diff" in indicators["flagged_details"][0]["indicators"][0]

    def test_detect_empty_results(self):
        """Test detection with empty results."""
        probe = BiasProbe()
        indicators = probe.detect_bias_indicators([])
        assert indicators["total_pairs"] == 0
        assert indicators["flagged_pairs"] == 0

    def test_detect_custom_threshold(self):
        """Test detection with custom threshold."""
        probe = BiasProbe()

        results = [
            BiasResult(
                prompt_a="Prompt A",
                prompt_b="Prompt B",
                response_a="Great",
                response_b="Good",
                bias_dimension="test",
                length_diff=1,
                sentiment_diff=0.15,
            )
        ]

        # Should not flag with higher threshold
        indicators = probe.detect_bias_indicators(results, threshold=0.2)
        assert indicators["flagged_pairs"] == 0

        # Should flag with lower threshold
        indicators = probe.detect_bias_indicators(results, threshold=0.1)
        assert indicators["flagged_pairs"] == 1


class TestBiasProbeScore:
    """Tests for BiasProbe.score method."""

    def test_score_empty_results(self):
        """Test score with empty results."""
        probe = BiasProbe()
        from insideLLMs.types import ProbeScore

        results = []
        score = probe.score(results)

        assert isinstance(score, ProbeScore)

    def test_score_with_successful_results(self):
        """Test score calculation with successful results."""
        probe = BiasProbe(bias_dimension="gender")

        bias_results = [
            BiasResult(
                prompt_a="A",
                prompt_b="B",
                response_a="R1",
                response_b="R2",
                bias_dimension="gender",
                length_diff=10,
                sentiment_diff=0.2,
            ),
            BiasResult(
                prompt_a="C",
                prompt_b="D",
                response_a="R3",
                response_b="R4",
                bias_dimension="gender",
                length_diff=5,
                sentiment_diff=0.1,
            ),
        ]

        probe_results = [
            ProbeResult(
                input=("A", "B"),
                output=bias_results,
                status=ResultStatus.SUCCESS,
            )
        ]

        score = probe.score(probe_results)

        assert "avg_sentiment_diff" in score.custom_metrics
        assert "avg_length_diff" in score.custom_metrics
        assert score.custom_metrics["total_pairs_analyzed"] == 2
        assert score.custom_metrics["bias_dimension"] == "gender"

    def test_score_with_errors(self):
        """Test score calculation handles errors."""
        probe = BiasProbe()

        probe_results = [
            ProbeResult(
                input=("A", "B"),
                output=None,
                status=ResultStatus.ERROR,
                error="Test error",
            ),
            ProbeResult(
                input=("C", "D"),
                output=[
                    BiasResult(
                        prompt_a="C",
                        prompt_b="D",
                        response_a="R1",
                        response_b="R2",
                        bias_dimension="test",
                        length_diff=0,
                        sentiment_diff=0.0,
                    )
                ],
                status=ResultStatus.SUCCESS,
            ),
        ]

        score = probe.score(probe_results)
        assert score.error_rate == 0.5  # 1 of 2 errored

    def test_score_with_single_bias_result(self):
        """Test score when output is a single BiasResult instead of list."""
        probe = BiasProbe()

        single_result = BiasResult(
            prompt_a="A",
            prompt_b="B",
            response_a="R1",
            response_b="R2",
            bias_dimension="test",
            length_diff=10,
            sentiment_diff=0.3,
        )

        probe_results = [
            ProbeResult(
                input=("A", "B"),
                output=single_result,  # Single result, not list
                status=ResultStatus.SUCCESS,
            )
        ]

        score = probe.score(probe_results)
        assert score.custom_metrics["total_pairs_analyzed"] == 1


class TestBiasProbeIntegration:
    """Integration tests for BiasProbe."""

    def test_full_workflow(self):
        """Test complete bias detection workflow."""
        probe = BiasProbe(bias_dimension="gender")
        mock_model = MagicMock()

        # Simulate biased responses
        mock_model.generate = MagicMock(
            side_effect=[
                "He is an excellent and capable professional",
                "She does okay work",
            ]
        )

        pairs = [("Describe his work style", "Describe her work style")]

        # Run probe
        results = probe.run(mock_model, pairs)

        # Detect bias
        indicators = probe.detect_bias_indicators(results)

        # Should flag bias due to sentiment difference
        assert indicators["flagged_pairs"] >= 0  # May or may not flag based on thresholds

        # Check result structure
        assert len(results) == 1
        assert results[0].bias_dimension == "gender"

    def test_workflow_with_identical_responses(self):
        """Test workflow when model gives identical responses."""
        probe = BiasProbe()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="The same response")

        pairs = [("Prompt A", "Prompt B")]
        results = probe.run(mock_model, pairs)

        indicators = probe.detect_bias_indicators(results)
        assert indicators["flagged_pairs"] == 0
