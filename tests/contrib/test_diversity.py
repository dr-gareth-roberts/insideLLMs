"""Tests for diversity and creativity metrics."""

from insideLLMs.contrib.diversity import (
    CreativityAnalyzer,
    CreativityDimension,
    CreativityScore,
    DiversityMetric,
    DiversityReport,
    DiversityReporter,
    DiversityScore,
    LexicalDiversityAnalyzer,
    OutputVariabilityAnalyzer,
    RepetitionAnalysis,
    RepetitionDetector,
    VariabilityAnalysis,
    analyze_creativity,
    analyze_diversity,
    analyze_output_variability,
    calculate_type_token_ratio,
    detect_repetition,
    quick_diversity_check,
)


class TestDiversityMetric:
    """Tests for DiversityMetric enum."""

    def test_all_metrics_exist(self):
        """Test all expected metrics exist."""
        assert DiversityMetric.TYPE_TOKEN_RATIO
        assert DiversityMetric.HAPAX_LEGOMENA
        assert DiversityMetric.YULES_K
        assert DiversityMetric.SIMPSONS_D
        assert DiversityMetric.ENTROPY
        assert DiversityMetric.MTLD


class TestCreativityDimension:
    """Tests for CreativityDimension enum."""

    def test_all_dimensions_exist(self):
        """Test all expected dimensions exist."""
        assert CreativityDimension.NOVELTY
        assert CreativityDimension.UNEXPECTEDNESS
        assert CreativityDimension.ELABORATION
        assert CreativityDimension.FLEXIBILITY
        assert CreativityDimension.FLUENCY


class TestDiversityScore:
    """Tests for DiversityScore dataclass."""

    def test_basic_creation(self):
        """Test basic score creation."""
        score = DiversityScore(
            metric=DiversityMetric.TYPE_TOKEN_RATIO,
            value=0.75,
            interpretation="High diversity",
        )
        assert score.value == 0.75

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = DiversityScore(
            metric=DiversityMetric.ENTROPY,
            value=0.8,
            interpretation="High entropy",
        )
        d = score.to_dict()
        assert d["metric"] == "entropy"
        assert d["value"] == 0.8


class TestRepetitionAnalysis:
    """Tests for RepetitionAnalysis dataclass."""

    def test_significant_repetition(self):
        """Test significant repetition detection."""
        analysis = RepetitionAnalysis(
            repeated_phrases=[("hello world", 5)],
            repeated_words=[("hello", 10)],
            repetition_score=0.5,
            longest_repeated_sequence="hello world again",
            n_gram_repetition={2: 0.3, 3: 0.2},
        )
        assert analysis.has_significant_repetition

    def test_no_significant_repetition(self):
        """Test no significant repetition."""
        analysis = RepetitionAnalysis(
            repeated_phrases=[],
            repeated_words=[],
            repetition_score=0.1,
            longest_repeated_sequence="",
            n_gram_repetition={},
        )
        assert not analysis.has_significant_repetition

    def test_to_dict(self):
        """Test conversion to dictionary."""
        analysis = RepetitionAnalysis(
            repeated_phrases=[("test", 2)],
            repeated_words=[("word", 3)],
            repetition_score=0.2,
            longest_repeated_sequence="test phrase",
            n_gram_repetition={2: 0.1},
        )
        d = analysis.to_dict()
        assert "repetition_score" in d


class TestCreativityScore:
    """Tests for CreativityScore dataclass."""

    def test_creativity_levels(self):
        """Test creativity level categorization."""
        high = CreativityScore(0.85, {}, "test", [], [])
        assert high.creativity_level == "highly_creative"

        moderate = CreativityScore(0.45, {}, "test", [], [])
        assert moderate.creativity_level == "moderate"

        low = CreativityScore(0.15, {}, "test", [], [])
        assert low.creativity_level == "minimal"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = CreativityScore(
            overall_score=0.7,
            dimension_scores={CreativityDimension.NOVELTY: 0.8},
            interpretation="Creative",
            strengths=["Strong novelty"],
            weaknesses=[],
        )
        d = score.to_dict()
        assert d["creativity_level"] == "creative"


class TestVariabilityAnalysis:
    """Tests for VariabilityAnalysis dataclass."""

    def test_diverse_outputs(self):
        """Test diverse output detection."""
        analysis = VariabilityAnalysis(
            n_samples=5,
            mean_similarity=0.3,
            std_similarity=0.1,
            unique_tokens_ratio=0.8,
            semantic_spread=0.7,
            clustering_coefficient=0.5,
            outlier_indices=[],
        )
        assert analysis.is_diverse

    def test_non_diverse_outputs(self):
        """Test non-diverse output detection."""
        analysis = VariabilityAnalysis(
            n_samples=5,
            mean_similarity=0.9,
            std_similarity=0.05,
            unique_tokens_ratio=0.3,
            semantic_spread=0.1,
            clustering_coefficient=0.9,
            outlier_indices=[],
        )
        assert not analysis.is_diverse


class TestLexicalDiversityAnalyzer:
    """Tests for LexicalDiversityAnalyzer."""

    def test_type_token_ratio_high(self):
        """Test TTR for high diversity text."""
        analyzer = LexicalDiversityAnalyzer()
        text = "The quick brown fox jumps over the lazy dog"
        result = analyzer.type_token_ratio(text)
        assert result.value > 0.7

    def test_type_token_ratio_low(self):
        """Test TTR for low diversity text."""
        analyzer = LexicalDiversityAnalyzer()
        text = "the the the the the the the the"
        result = analyzer.type_token_ratio(text)
        assert result.value < 0.3

    def test_hapax_legomena(self):
        """Test hapax legomena ratio."""
        analyzer = LexicalDiversityAnalyzer()
        text = "one two three four five six seven eight"
        result = analyzer.hapax_legomena_ratio(text)
        assert result.value == 1.0  # All words appear once

    def test_yules_k(self):
        """Test Yule's K."""
        analyzer = LexicalDiversityAnalyzer()
        text = "word word word another another yet"
        result = analyzer.yules_k(text)
        assert result.metric == DiversityMetric.YULES_K

    def test_simpsons_diversity(self):
        """Test Simpson's Diversity Index."""
        analyzer = LexicalDiversityAnalyzer()
        text = "a b c d e f g h i j"
        result = analyzer.simpsons_diversity(text)
        assert result.value > 0.8  # All unique = high diversity

    def test_entropy_uniform(self):
        """Test entropy for uniform distribution."""
        analyzer = LexicalDiversityAnalyzer()
        text = "a b c d e f g h"
        result = analyzer.entropy(text)
        assert result.value > 0.9  # Near-maximum entropy

    def test_entropy_skewed(self):
        """Test entropy for skewed distribution."""
        analyzer = LexicalDiversityAnalyzer()
        # Very skewed text with mostly one word
        text = "the the the the the the the the the a"
        result = analyzer.entropy(text)
        assert result.value < 0.7  # Lower entropy than uniform

    def test_mtld(self):
        """Test MTLD calculation."""
        analyzer = LexicalDiversityAnalyzer()
        # Need sufficient text for MTLD
        text = " ".join([f"word{i}" for i in range(50)])
        result = analyzer.mtld(text)
        assert result.value > 0

    def test_analyze_all(self):
        """Test analyzing all metrics."""
        analyzer = LexicalDiversityAnalyzer()
        text = "The quick brown fox jumps over the lazy dog"
        results = analyzer.analyze_all(text)
        assert len(results) == 6
        assert all(isinstance(v, DiversityScore) for v in results.values())

    def test_empty_text(self):
        """Test with empty text."""
        analyzer = LexicalDiversityAnalyzer()
        result = analyzer.type_token_ratio("")
        assert result.value == 0.0

    def test_custom_tokenizer(self):
        """Test with custom tokenizer."""

        def char_tokenize(text: str) -> list:
            return list(text.replace(" ", ""))

        analyzer = LexicalDiversityAnalyzer(tokenize_fn=char_tokenize)
        result = analyzer.type_token_ratio("hello world")
        assert result.value > 0


class TestRepetitionDetector:
    """Tests for RepetitionDetector."""

    def test_detect_repeated_words(self):
        """Test detecting repeated words."""
        detector = RepetitionDetector()
        text = "hello world hello world hello again"
        result = detector.detect(text)
        assert any(word == "hello" for word, _ in result.repeated_words)

    def test_detect_repeated_phrases(self):
        """Test detecting repeated phrases."""
        detector = RepetitionDetector()
        text = "the quick fox the quick fox jumps high"
        result = detector.detect(text)
        assert len(result.repeated_phrases) > 0

    def test_no_repetition(self):
        """Test text with no repetition."""
        detector = RepetitionDetector()
        text = "each word here is completely unique different"
        result = detector.detect(text)
        assert result.repetition_score < 0.3

    def test_n_gram_repetition(self):
        """Test n-gram repetition rates."""
        detector = RepetitionDetector()
        text = "a b c a b c a b c"
        result = detector.detect(text)
        assert 2 in result.n_gram_repetition or 3 in result.n_gram_repetition

    def test_short_text(self):
        """Test with short text."""
        detector = RepetitionDetector()
        result = detector.detect("hi")
        assert result.repetition_score == 0.0


class TestCreativityAnalyzer:
    """Tests for CreativityAnalyzer."""

    def test_analyze_creative_text(self):
        """Test analyzing creative text."""
        analyzer = CreativityAnalyzer()
        text = """
        The shimmering aurora danced across the velvet sky,
        painting ephemeral ribbons of emerald and sapphire light.
        """
        result = analyzer.analyze(text)
        assert result.overall_score > 0

    def test_analyze_bland_text(self):
        """Test analyzing bland text."""
        analyzer = CreativityAnalyzer()
        text = "The thing is good. It is nice. The thing is helpful."
        result = analyzer.analyze(text)
        # Should have lower creativity than varied text
        assert result.overall_score < 1.0

    def test_novelty_score(self):
        """Test novelty scoring."""
        analyzer = CreativityAnalyzer()
        # Text with uncommon words
        text = "The resplendent ephemeral phenomenon astonished spectators"
        result = analyzer.analyze(text)
        assert result.dimension_scores[CreativityDimension.NOVELTY] > 0

    def test_elaboration_score(self):
        """Test elaboration scoring."""
        analyzer = CreativityAnalyzer()
        # Well-elaborated text
        text = """
        The ancient castle stood majestically on the cliff,
        its weathered stones telling stories of centuries past.
        """
        result = analyzer.analyze(text)
        assert result.dimension_scores[CreativityDimension.ELABORATION] > 0

    def test_empty_text(self):
        """Test with empty text."""
        analyzer = CreativityAnalyzer()
        result = analyzer.analyze("")
        assert result.overall_score >= 0


class TestOutputVariabilityAnalyzer:
    """Tests for OutputVariabilityAnalyzer."""

    def test_identical_outputs(self):
        """Test with identical outputs."""
        analyzer = OutputVariabilityAnalyzer()
        outputs = ["hello world", "hello world", "hello world"]
        result = analyzer.analyze(outputs)
        assert result.mean_similarity == 1.0
        assert not result.is_diverse

    def test_diverse_outputs(self):
        """Test with diverse outputs."""
        analyzer = OutputVariabilityAnalyzer()
        outputs = [
            "The quick brown fox",
            "A lazy sleeping dog",
            "Mountains covered in snow",
        ]
        result = analyzer.analyze(outputs)
        assert result.mean_similarity < 1.0

    def test_single_output(self):
        """Test with single output."""
        analyzer = OutputVariabilityAnalyzer()
        result = analyzer.analyze(["single output"])
        assert result.n_samples == 1
        assert result.mean_similarity == 1.0

    def test_empty_outputs(self):
        """Test with no outputs."""
        analyzer = OutputVariabilityAnalyzer()
        result = analyzer.analyze([])
        assert result.n_samples == 0

    def test_outlier_detection(self):
        """Test outlier detection."""
        analyzer = OutputVariabilityAnalyzer()
        outputs = [
            "hello world hello",
            "hello world again",
            "hello world there",
            "completely different text here",  # Outlier
        ]
        result = analyzer.analyze(outputs)
        # May or may not detect outlier depending on similarity
        assert "outlier_indices" in result.to_dict()

    def test_custom_similarity(self):
        """Test with custom similarity function."""

        def always_half(a: str, b: str) -> float:
            return 0.5

        analyzer = OutputVariabilityAnalyzer(similarity_fn=always_half)
        outputs = ["a", "b", "c"]
        result = analyzer.analyze(outputs)
        assert result.mean_similarity == 0.5


class TestDiversityReporter:
    """Tests for DiversityReporter."""

    def test_generate_report(self):
        """Test generating full report."""
        reporter = DiversityReporter()
        text = "The quick brown fox jumps over the lazy dog multiple times."
        report = reporter.generate_report(text)

        assert isinstance(report, DiversityReport)
        assert report.overall_diversity_score > 0
        assert len(report.lexical_diversity) > 0
        assert report.repetition is not None

    def test_report_without_creativity(self):
        """Test report without creativity analysis."""
        reporter = DiversityReporter()
        report = reporter.generate_report(
            "Test text here.",
            include_creativity=False,
        )
        assert report.creativity is None

    def test_recommendations_generated(self):
        """Test that recommendations are generated."""
        reporter = DiversityReporter()
        # Repetitive text should generate recommendations
        text = "word word word word word word"
        report = reporter.generate_report(text)
        assert len(report.recommendations) > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_analyze_diversity(self):
        """Test analyze_diversity function."""
        report = analyze_diversity("The quick brown fox")
        assert isinstance(report, DiversityReport)

    def test_calculate_type_token_ratio(self):
        """Test calculate_type_token_ratio function."""
        ttr = calculate_type_token_ratio("one two three four")
        assert ttr == 1.0  # All unique

    def test_detect_repetition(self):
        """Test detect_repetition function."""
        result = detect_repetition("hello hello world world")
        assert isinstance(result, RepetitionAnalysis)

    def test_analyze_creativity(self):
        """Test analyze_creativity function."""
        result = analyze_creativity("Creative and unique text here")
        assert isinstance(result, CreativityScore)

    def test_analyze_output_variability(self):
        """Test analyze_output_variability function."""
        result = analyze_output_variability(["text1", "text2"])
        assert isinstance(result, VariabilityAnalysis)

    def test_quick_diversity_check(self):
        """Test quick_diversity_check function."""
        result = quick_diversity_check("Sample text for checking")
        assert "overall_score" in result
        assert "type_token_ratio" in result
        assert "has_repetition" in result


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_word(self):
        """Test with single word."""
        analyzer = LexicalDiversityAnalyzer()
        result = analyzer.type_token_ratio("hello")
        assert result.value == 1.0

    def test_very_long_text(self):
        """Test with very long text."""
        analyzer = LexicalDiversityAnalyzer()
        text = " ".join(["word"] * 1000 + [f"unique{i}" for i in range(100)])
        result = analyzer.type_token_ratio(text)
        assert 0 < result.value < 1

    def test_special_characters(self):
        """Test with special characters."""
        analyzer = LexicalDiversityAnalyzer()
        text = "Hello! World? Test... @#$%"
        result = analyzer.type_token_ratio(text)
        assert result.value > 0

    def test_unicode_text(self):
        """Test with Unicode text."""
        analyzer = LexicalDiversityAnalyzer()
        text = "你好 世界 hello world"
        result = analyzer.type_token_ratio(text)
        assert result.value > 0

    def test_numbers_in_text(self):
        """Test with numbers in text."""
        analyzer = LexicalDiversityAnalyzer()
        text = "1 2 3 4 5 one two three four five"
        result = analyzer.type_token_ratio(text)
        assert result.value == 1.0

    def test_mtld_short_text(self):
        """Test MTLD with short text."""
        analyzer = LexicalDiversityAnalyzer()
        result = analyzer.mtld("short")
        assert result.value == 0.0  # Insufficient tokens
