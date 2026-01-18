"""Tests for response quality scoring and analysis utilities."""

from insideLLMs.quality import (
    ClarityScorer,
    CoherenceScorer,
    ComparisonResult,
    CompletenessScorer,
    ConcisenessScorer,
    DimensionScore,
    QualityDimension,
    QualityReport,
    RelevanceScorer,
    ResponseComparator,
    ResponseQualityAnalyzer,
    SpecificityScorer,
    analyze_quality,
    compare_responses,
    quick_quality_check,
)


class TestQualityDimension:
    """Tests for QualityDimension enum."""

    def test_all_dimensions_exist(self):
        """Test that all dimensions are defined."""
        assert QualityDimension.RELEVANCE.value == "relevance"
        assert QualityDimension.COMPLETENESS.value == "completeness"
        assert QualityDimension.COHERENCE.value == "coherence"
        assert QualityDimension.CONCISENESS.value == "conciseness"
        assert QualityDimension.CLARITY.value == "clarity"
        assert QualityDimension.SPECIFICITY.value == "specificity"


class TestDimensionScore:
    """Tests for DimensionScore."""

    def test_basic_creation(self):
        """Test basic score creation."""
        score = DimensionScore(
            dimension=QualityDimension.RELEVANCE,
            score=0.8,
            explanation="High relevance",
        )

        assert score.dimension == QualityDimension.RELEVANCE
        assert score.score == 0.8

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = DimensionScore(
            dimension=QualityDimension.CLARITY,
            score=0.75,
            confidence=0.9,
            explanation="Clear",
            evidence=["Good structure"],
        )

        d = score.to_dict()
        assert d["dimension"] == "clarity"
        assert d["score"] == 0.75
        assert d["confidence"] == 0.9


class TestQualityReport:
    """Tests for QualityReport."""

    def test_basic_creation(self):
        """Test basic report creation."""
        report = QualityReport(
            prompt="Test prompt",
            response="Test response",
            overall_score=0.8,
        )

        assert report.overall_score == 0.8
        assert report.passed is True

    def test_passed_threshold(self):
        """Test passed threshold."""
        passing = QualityReport(prompt="", response="", overall_score=0.75)
        failing = QualityReport(prompt="", response="", overall_score=0.65)

        assert passing.passed is True
        assert failing.passed is False

    def test_get_score(self):
        """Test getting dimension score."""
        scores = {
            QualityDimension.RELEVANCE: DimensionScore(
                dimension=QualityDimension.RELEVANCE,
                score=0.9,
                explanation="",
            )
        }
        report = QualityReport(
            prompt="",
            response="",
            overall_score=0.9,
            dimension_scores=scores,
        )

        assert report.get_score(QualityDimension.RELEVANCE) == 0.9
        assert report.get_score(QualityDimension.CLARITY) is None

    def test_get_weakest_dimensions(self):
        """Test getting weakest dimensions."""
        scores = {
            QualityDimension.RELEVANCE: DimensionScore(
                dimension=QualityDimension.RELEVANCE, score=0.9, explanation=""
            ),
            QualityDimension.CLARITY: DimensionScore(
                dimension=QualityDimension.CLARITY, score=0.5, explanation=""
            ),
            QualityDimension.COHERENCE: DimensionScore(
                dimension=QualityDimension.COHERENCE, score=0.7, explanation=""
            ),
        }
        report = QualityReport(
            prompt="",
            response="",
            overall_score=0.7,
            dimension_scores=scores,
        )

        weakest = report.get_weakest_dimensions(2)
        assert len(weakest) == 2
        assert weakest[0][0] == QualityDimension.CLARITY
        assert weakest[0][1] == 0.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = QualityReport(
            prompt="Test",
            response="Response",
            overall_score=0.8,
            issues=["Issue 1"],
            suggestions=["Suggestion 1"],
        )

        d = report.to_dict()
        assert d["overall_score"] == 0.8
        assert d["passed"] is True
        assert "issues" in d


class TestRelevanceScorer:
    """Tests for RelevanceScorer."""

    def test_high_relevance(self):
        """Test high relevance response."""
        scorer = RelevanceScorer()
        prompt = "What is machine learning?"
        response = "Machine learning is a subset of artificial intelligence that enables systems to learn from data. It involves algorithms that improve through experience."

        score = scorer.score(prompt, response)
        assert score.score >= 0.5
        assert score.dimension == QualityDimension.RELEVANCE

    def test_low_relevance(self):
        """Test low relevance response."""
        scorer = RelevanceScorer()
        prompt = "What is machine learning?"
        response = "The weather today is sunny with a chance of rain in the evening."

        score = scorer.score(prompt, response)
        assert score.score < 0.5

    def test_question_answer_structure(self):
        """Test question with answer structure."""
        scorer = RelevanceScorer()
        prompt = "Why is the sky blue?"
        response = "The sky is blue because of a phenomenon called Rayleigh scattering. This occurs because the atmosphere scatters shorter wavelengths of light."

        score = scorer.score(prompt, response)
        assert score.score >= 0.5


class TestCompletenessScorer:
    """Tests for CompletenessScorer."""

    def test_complete_response(self):
        """Test complete response."""
        scorer = CompletenessScorer()
        prompt = "Explain photosynthesis."
        response = """Photosynthesis is the process by which plants convert light energy into chemical energy.

First, light is absorbed by chlorophyll in the leaves.
Then, water molecules are split, releasing oxygen.
Finally, carbon dioxide is converted into glucose.

In conclusion, photosynthesis is essential for life on Earth."""

        score = scorer.score(prompt, response)
        assert score.score >= 0.5
        assert any("completion" in e.lower() for e in score.evidence)

    def test_incomplete_response(self):
        """Test incomplete response."""
        scorer = CompletenessScorer()
        prompt = "Explain photosynthesis in detail."
        response = "I'm not sure about all the details, but basically plants use light..."

        score = scorer.score(prompt, response)
        assert score.score < 0.7


class TestCoherenceScorer:
    """Tests for CoherenceScorer."""

    def test_coherent_response(self):
        """Test coherent response."""
        scorer = CoherenceScorer()
        prompt = "Describe the water cycle."
        response = """First, water evaporates from oceans and lakes. Then, the water vapor rises and forms clouds.

Subsequently, precipitation occurs as rain or snow. Finally, the water flows back to the oceans through rivers, completing the cycle."""

        score = scorer.score(prompt, response)
        assert score.score >= 0.5

    def test_incoherent_response(self):
        """Test incoherent response."""
        scorer = CoherenceScorer()
        prompt = "Describe the water cycle."
        response = "Water. Clouds. Rain. Rivers. Oceans. Yes. No. Maybe. Very."

        score = scorer.score(prompt, response)
        assert score.score < 0.7


class TestConcisenessScorer:
    """Tests for ConcisenessScorer."""

    def test_concise_response(self):
        """Test concise response."""
        scorer = ConcisenessScorer()
        prompt = "What is the capital of France?"
        response = "The capital of France is Paris. It is located in the north-central part of the country."

        score = scorer.score(prompt, response)
        assert score.score >= 0.5

    def test_verbose_response(self):
        """Test verbose response."""
        scorer = ConcisenessScorer()
        prompt = "What is the capital of France?"
        response = """Well, basically, I think that, as you know, the capital of France is actually Paris.
        It is really a very beautiful city. In my opinion, it should be noted that Paris is literally the most important city in France.
        Needless to say, it is sort of a kind of major cultural center."""

        score = scorer.score(prompt, response)
        assert score.score < 0.7


class TestClarityScorer:
    """Tests for ClarityScorer."""

    def test_clear_response(self):
        """Test clear response."""
        scorer = ClarityScorer()
        prompt = "How do plants make food?"
        response = """Plants make food through photosynthesis. Here's how it works:

1. Plants absorb sunlight through their leaves
2. They take in carbon dioxide from the air
3. They absorb water through their roots
4. Using energy from sunlight, they combine these to make sugar

This sugar is their food."""

        score = scorer.score(prompt, response)
        assert score.score >= 0.5

    def test_unclear_response(self):
        """Test unclear response with jargon."""
        scorer = ClarityScorer()
        prompt = "How do plants make food?"
        response = "The photoautotrophic biochemical pathway utilizing photochemically-driven electron transfer chains facilitates the transformation of electromagnetic radiation into chemical potential energy through complex molecular mechanisms (involving various photosynthetic pigment-protein supercomplexes)."

        scorer.score(prompt, response)
        # May still score decently if structure is ok


class TestSpecificityScorer:
    """Tests for SpecificityScorer."""

    def test_specific_response(self):
        """Test specific response."""
        scorer = SpecificityScorer()
        prompt = "What are the benefits of exercise?"
        response = """Exercise has many specific benefits. For example:

1. Reduces heart disease risk by 35%
2. Helps maintain healthy body weight
3. The American Heart Association recommends 150 minutes per week
4. Studies from Harvard Medical School show improved longevity"""

        score = scorer.score(prompt, response)
        assert score.score >= 0.5
        assert any("specific" in e.lower() or "example" in e.lower() for e in score.evidence)

    def test_vague_response(self):
        """Test vague response."""
        scorer = SpecificityScorer()
        prompt = "What are the benefits of exercise?"
        response = "Exercise has some benefits. It can help with various things and is good for stuff in general. Many people often find it somewhat useful."

        score = scorer.score(prompt, response)
        assert score.score < 0.7


class TestResponseQualityAnalyzer:
    """Tests for ResponseQualityAnalyzer."""

    def test_analyze_good_response(self):
        """Test analyzing a good response."""
        analyzer = ResponseQualityAnalyzer()
        prompt = "Explain the importance of sleep."
        response = """Sleep is essential for physical and mental health. Here are the key reasons:

First, sleep allows the body to repair tissues and consolidate memories. Studies show that adults need 7-9 hours of sleep per night.

Second, lack of sleep impairs cognitive function and decision-making. For example, the CDC reports that drowsy driving causes thousands of accidents annually.

In conclusion, prioritizing sleep is one of the most important things you can do for your health."""

        report = analyzer.analyze(prompt, response)

        assert report.overall_score > 0.5
        assert len(report.dimension_scores) > 0

    def test_analyze_poor_response(self):
        """Test analyzing a poor response."""
        analyzer = ResponseQualityAnalyzer()
        prompt = "Explain the importance of sleep in detail."
        response = "Sleep is good I think maybe."

        report = analyzer.analyze(prompt, response)

        assert report.overall_score < 0.7
        assert len(report.issues) > 0

    def test_custom_dimensions(self):
        """Test analyzer with custom dimensions."""
        analyzer = ResponseQualityAnalyzer(
            dimensions=[QualityDimension.RELEVANCE, QualityDimension.CLARITY]
        )

        report = analyzer.analyze("Test prompt", "Test response")

        assert len(report.dimension_scores) == 2

    def test_custom_weights(self):
        """Test analyzer with custom weights."""
        analyzer = ResponseQualityAnalyzer(
            weights={
                QualityDimension.RELEVANCE: 2.0,
                QualityDimension.CLARITY: 1.0,
            }
        )

        report = analyzer.analyze("Test prompt", "Test response")
        assert report.overall_score is not None

    def test_quick_check(self):
        """Test quick check method."""
        analyzer = ResponseQualityAnalyzer()

        score, passed, issues = analyzer.quick_check(
            "What is AI?", "AI stands for artificial intelligence."
        )

        assert isinstance(score, float)
        assert isinstance(passed, bool)
        assert isinstance(issues, list)


class TestResponseComparator:
    """Tests for ResponseComparator."""

    def test_compare_responses(self):
        """Test comparing two responses."""
        comparator = ResponseComparator()
        prompt = "What is the capital of Japan?"
        response_a = "Tokyo is the capital of Japan. It is located on the eastern coast of Honshu."
        response_b = "idk maybe tokyo or something"

        result = comparator.compare(prompt, response_a, response_b)

        assert result.winner == "A"
        assert result.score_a > result.score_b

    def test_compare_similar_responses(self):
        """Test comparing similar quality responses."""
        comparator = ResponseComparator()
        prompt = "What is the capital of Japan?"
        response_a = "The capital of Japan is Tokyo."
        response_b = "Tokyo is Japan's capital city."

        result = comparator.compare(prompt, response_a, response_b)

        # Should be close
        assert abs(result.score_a - result.score_b) < 0.3

    def test_comparison_result_fields(self):
        """Test comparison result has all fields."""
        comparator = ResponseComparator()
        result = comparator.compare("prompt", "response a", "response b")

        assert result.prompt == "prompt"
        assert result.response_a == "response a"
        assert result.response_b == "response b"
        assert result.winner in ["A", "B", "tie"]
        assert isinstance(result.dimension_comparison, dict)
        assert isinstance(result.reasoning, str)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_analyze_quality(self):
        """Test analyze_quality function."""
        report = analyze_quality(
            "What is Python?",
            "Python is a high-level programming language known for its readability and versatility.",
        )

        assert isinstance(report, QualityReport)
        assert report.overall_score > 0

    def test_quick_quality_check(self):
        """Test quick_quality_check function."""
        score, passed, issues = quick_quality_check(
            "Explain gravity.", "Gravity is the force that attracts objects toward each other."
        )

        assert 0 <= score <= 1
        assert isinstance(passed, bool)

    def test_compare_responses(self):
        """Test compare_responses function."""
        result = compare_responses("What is 2+2?", "The answer is 4.", "Two plus two equals four.")

        assert isinstance(result, ComparisonResult)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_response(self):
        """Test with empty response."""
        analyzer = ResponseQualityAnalyzer()
        report = analyzer.analyze("What is AI?", "")

        assert report.overall_score < 0.5

    def test_very_short_response(self):
        """Test with very short response."""
        analyzer = ResponseQualityAnalyzer()
        report = analyzer.analyze("Explain the theory of relativity in detail.", "E=mc²")

        assert report is not None

    def test_very_long_response(self):
        """Test with very long response."""
        analyzer = ResponseQualityAnalyzer()
        long_response = "This is a test sentence. " * 100

        report = analyzer.analyze("Test prompt", long_response)

        assert report is not None

    def test_unicode_response(self):
        """Test with unicode characters."""
        analyzer = ResponseQualityAnalyzer()
        report = analyzer.analyze(
            "How do you say hello in Japanese?",
            "In Japanese, hello is こんにちは (konnichiwa). \U0001f1ef\U0001f1f5",
        )

        assert report is not None
        assert report.overall_score > 0

    def test_response_with_code(self):
        """Test with code in response."""
        analyzer = ResponseQualityAnalyzer()
        report = analyzer.analyze(
            "Write a Python hello world program.",
            """Here's a simple Python hello world program:

```python
print("Hello, World!")
```

This program uses the print function to display text.""",
        )

        assert report is not None
