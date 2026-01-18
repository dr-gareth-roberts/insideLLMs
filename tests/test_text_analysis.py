"""Tests for advanced text analysis utilities."""

from insideLLMs.nlp.text_analysis import (
    ContentAnalysis,
    ReadabilityMetrics,
    ResponseQualityScore,
    TextAnalyzer,
    TextProfile,
    ToneAnalysis,
    analyze_text,
    compare_responses,
    score_response,
)


class TestTextProfile:
    """Tests for TextProfile."""

    def test_from_text_basic(self):
        """Test basic profile creation."""
        text = "Hello world. This is a test."
        profile = TextProfile.from_text(text)

        assert profile.char_count == len(text)
        assert profile.word_count == 6
        assert profile.sentence_count == 2
        assert profile.unique_words == 6

    def test_from_text_empty(self):
        """Test empty text profile."""
        profile = TextProfile.from_text("")
        assert profile.word_count == 0
        assert profile.char_count == 0

    def test_lexical_diversity(self):
        """Test lexical diversity calculation."""
        # High diversity (all unique words)
        text1 = "apple banana cherry date elderberry"
        profile1 = TextProfile.from_text(text1)
        assert profile1.lexical_diversity == 1.0

        # Low diversity (repeated words)
        text2 = "the the the cat cat"
        profile2 = TextProfile.from_text(text2)
        assert profile2.lexical_diversity == 0.4  # 2 unique / 5 total

    def test_word_frequencies(self):
        """Test word frequency counting."""
        text = "cat dog cat bird cat"
        profile = TextProfile.from_text(text)

        assert profile.word_frequencies["cat"] == 3
        assert profile.word_frequencies["dog"] == 1
        assert profile.word_frequencies["bird"] == 1

    def test_paragraph_count(self):
        """Test paragraph counting."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        profile = TextProfile.from_text(text)
        assert profile.paragraph_count == 3


class TestReadabilityMetrics:
    """Tests for ReadabilityMetrics."""

    def test_get_reading_level(self):
        """Test reading level descriptions."""
        # Elementary level
        metrics = ReadabilityMetrics(avg_grade_level=3.0)
        assert metrics.get_reading_level() == "elementary"

        # Middle school
        metrics = ReadabilityMetrics(avg_grade_level=7.0)
        assert metrics.get_reading_level() == "middle_school"

        # High school
        metrics = ReadabilityMetrics(avg_grade_level=10.0)
        assert metrics.get_reading_level() == "high_school"

        # College
        metrics = ReadabilityMetrics(avg_grade_level=14.0)
        assert metrics.get_reading_level() == "college"

        # Graduate
        metrics = ReadabilityMetrics(avg_grade_level=18.0)
        assert metrics.get_reading_level() == "graduate"


class TestContentAnalysis:
    """Tests for ContentAnalysis defaults."""

    def test_defaults(self):
        """Test default values."""
        analysis = ContentAnalysis()
        assert not analysis.has_questions
        assert not analysis.has_lists
        assert not analysis.has_code
        assert analysis.dominant_structure == "prose"


class TestToneAnalysis:
    """Tests for ToneAnalysis."""

    def test_get_formality_level(self):
        """Test formality level descriptions."""
        # Informal
        tone = ToneAnalysis(formality_score=0.2)
        assert tone.get_formality_level() == "informal"

        # Neutral
        tone = ToneAnalysis(formality_score=0.5)
        assert tone.get_formality_level() == "neutral"

        # Formal
        tone = ToneAnalysis(formality_score=0.8)
        assert tone.get_formality_level() == "formal"


class TestTextAnalyzer:
    """Tests for TextAnalyzer."""

    def test_profile(self):
        """Test profile method."""
        analyzer = TextAnalyzer()
        profile = analyzer.profile("Hello world.")

        assert isinstance(profile, TextProfile)
        assert profile.word_count == 2

    def test_analyze_readability(self):
        """Test readability analysis."""
        analyzer = TextAnalyzer()

        # Simple text
        simple = "The cat sat. The dog ran."
        metrics = analyzer.analyze_readability(simple)

        assert isinstance(metrics, ReadabilityMetrics)
        assert metrics.flesch_reading_ease > 0
        assert metrics.avg_grade_level >= 0

    def test_analyze_readability_empty(self):
        """Test readability with empty text."""
        analyzer = TextAnalyzer()
        metrics = analyzer.analyze_readability("")

        assert metrics.flesch_reading_ease == 0.0
        assert metrics.avg_grade_level == 0.0

    def test_analyze_content_questions(self):
        """Test question detection."""
        analyzer = TextAnalyzer()
        text = "What is Python? How does it work? Why is it popular?"
        analysis = analyzer.analyze_content(text)

        assert analysis.has_questions
        assert analysis.question_count == 3

    def test_analyze_content_lists(self):
        """Test list detection."""
        analyzer = TextAnalyzer()
        text = """Here are some items:
- First item
- Second item
- Third item"""
        analysis = analyzer.analyze_content(text)

        assert analysis.has_lists
        assert analysis.list_item_count == 3

    def test_analyze_content_numbered_lists(self):
        """Test numbered list detection."""
        analyzer = TextAnalyzer()
        text = """Steps:
1. Do this
2. Then that
3. Finally this"""
        analysis = analyzer.analyze_content(text)

        assert analysis.has_lists
        assert analysis.list_item_count == 3

    def test_analyze_content_code(self):
        """Test code detection."""
        analyzer = TextAnalyzer()
        text = """Here is some code:
```python
print("hello")
```"""
        analysis = analyzer.analyze_content(text)

        assert analysis.has_code
        assert analysis.code_block_count == 1

    def test_analyze_content_urls(self):
        """Test URL detection."""
        analyzer = TextAnalyzer()
        text = "Visit https://example.com for more info."
        analysis = analyzer.analyze_content(text)

        assert analysis.has_urls
        assert analysis.url_count == 1

    def test_analyze_content_dominant_structure(self):
        """Test dominant structure detection."""
        analyzer = TextAnalyzer()

        # Code dominant
        code_text = "```python\ncode1\n```\n```python\ncode2\n```"
        assert analyzer.analyze_content(code_text).dominant_structure == "code"

        # List dominant
        list_text = "- item1\n- item2\n- item3\n- item4"
        assert analyzer.analyze_content(list_text).dominant_structure == "list"

        # Prose (default)
        prose_text = "This is a simple paragraph of text."
        assert analyzer.analyze_content(prose_text).dominant_structure == "prose"

    def test_analyze_tone_informal(self):
        """Test tone analysis for informal text."""
        analyzer = TextAnalyzer()
        text = "Hey, gonna show you some cool stuff. Yeah, it's awesome!"
        tone = analyzer.analyze_tone(text)

        assert tone.formality_score < 0.5

    def test_analyze_tone_formal(self):
        """Test tone analysis for formal text."""
        analyzer = TextAnalyzer()
        text = "Furthermore, the aforementioned analysis demonstrates that consequently the results are significant."
        tone = analyzer.analyze_tone(text)

        assert tone.formality_score > 0.5

    def test_analyze_tone_confident(self):
        """Test confidence detection."""
        analyzer = TextAnalyzer()
        text = "This will definitely work. It is absolutely certain."
        tone = analyzer.analyze_tone(text)

        assert tone.confidence_score > 0.5

    def test_analyze_tone_uncertain(self):
        """Test uncertainty detection."""
        analyzer = TextAnalyzer()
        text = "This might work. Perhaps it could be possible."
        tone = analyzer.analyze_tone(text)

        assert tone.confidence_score < 0.5

    def test_analyze_tone_empty(self):
        """Test tone analysis with empty text."""
        analyzer = TextAnalyzer()
        tone = analyzer.analyze_tone("")

        assert tone.formality_score == 0.0
        assert tone.confidence_score == 0.0

    def test_compare_texts(self):
        """Test text comparison."""
        analyzer = TextAnalyzer()
        text1 = "Short simple text."
        text2 = "This is a much longer and more elaborate piece of writing with many words."

        comparison = analyzer.compare_texts(text1, text2)

        assert "word_count" in comparison
        assert comparison["word_count"][0] < comparison["word_count"][1]
        assert comparison["word_count"][2] > 0  # Difference

    def test_vocabulary_overlap_identical(self):
        """Test vocabulary overlap with identical texts."""
        analyzer = TextAnalyzer()
        text = "the cat sat on the mat"

        jaccard, common, unique1, unique2 = analyzer.vocabulary_overlap(text, text)

        assert jaccard == 1.0
        assert len(unique1) == 0
        assert len(unique2) == 0

    def test_vocabulary_overlap_different(self):
        """Test vocabulary overlap with different texts."""
        analyzer = TextAnalyzer()
        text1 = "the cat sat"
        text2 = "the dog ran"

        jaccard, common, unique1, unique2 = analyzer.vocabulary_overlap(text1, text2)

        assert 0 < jaccard < 1
        assert "the" in common
        assert "cat" in unique1
        assert "dog" in unique2

    def test_vocabulary_overlap_no_overlap(self):
        """Test vocabulary overlap with no common words."""
        analyzer = TextAnalyzer()
        text1 = "apple banana cherry"
        text2 = "dog elephant fox"

        jaccard, common, unique1, unique2 = analyzer.vocabulary_overlap(text1, text2)

        assert jaccard == 0.0
        assert len(common) == 0

    def test_count_syllables(self):
        """Test syllable counting."""
        analyzer = TextAnalyzer()

        assert analyzer._count_syllables("cat") == 1
        assert analyzer._count_syllables("happy") == 2
        assert analyzer._count_syllables("beautiful") >= 3
        assert analyzer._count_syllables("a") == 1


class TestResponseQualityScore:
    """Tests for ResponseQualityScore."""

    def test_calculate_basic(self):
        """Test basic quality score calculation."""
        prompt = "What is Python?"
        response = "Python is a programming language. It is widely used for web development, data science, and automation."

        score = ResponseQualityScore.calculate(response, prompt)

        assert 0 <= score.relevance <= 1
        assert 0 <= score.completeness <= 1
        assert 0 <= score.coherence <= 1
        assert 0 <= score.conciseness <= 1
        assert 0 <= score.overall <= 1

    def test_calculate_with_keywords(self):
        """Test scoring with required keywords."""
        prompt = "Explain machine learning."
        response = "Machine learning is a type of artificial intelligence that enables systems to learn from data."

        score = ResponseQualityScore.calculate(
            response, prompt, required_keywords=["machine", "learning", "data"]
        )

        assert score.relevance > 0.5

    def test_calculate_empty_response(self):
        """Test scoring with empty response."""
        score = ResponseQualityScore.calculate("", "What is X?")

        assert score.completeness < 0.5
        assert score.overall < 0.5

    def test_calculate_very_long_response(self):
        """Test scoring with very long response."""
        prompt = "What is 2+2?"
        response = "The answer is four. " * 100  # Very verbose

        score = ResponseQualityScore.calculate(response, prompt)

        # Should be penalized for verbosity
        assert score.conciseness < 1.0

    def test_calculate_with_expected_length(self):
        """Test scoring with expected length."""
        prompt = "Write a sentence."
        response = "This is a simple sentence."

        score = ResponseQualityScore.calculate(response, prompt, expected_length=5)

        assert score.completeness == 1.0  # Close to expected


class TestAnalyzeText:
    """Tests for analyze_text function."""

    def test_analyze_text_complete(self):
        """Test complete text analysis."""
        text = "Hello world. This is a test. How are you?"
        result = analyze_text(text)

        assert "profile" in result
        assert "readability" in result
        assert "content" in result
        assert "tone" in result

        assert isinstance(result["profile"], TextProfile)
        assert isinstance(result["readability"], ReadabilityMetrics)
        assert isinstance(result["content"], ContentAnalysis)
        assert isinstance(result["tone"], ToneAnalysis)


class TestScoreResponse:
    """Tests for score_response function."""

    def test_score_response(self):
        """Test response scoring function."""
        score = score_response(
            response="Python is a programming language.",
            prompt="What is Python?",
        )

        assert isinstance(score, ResponseQualityScore)
        assert score.overall > 0


class TestCompareResponses:
    """Tests for compare_responses function."""

    def test_compare_responses(self):
        """Test response comparison."""
        prompt = "Explain AI."
        response1 = "AI stands for artificial intelligence. It refers to computer systems that can perform tasks normally requiring human intelligence."
        response2 = "AI is tech stuff."

        comparison = compare_responses(response1, response2, prompt)

        assert "response1_score" in comparison
        assert "response2_score" in comparison
        assert "comparison" in comparison
        assert "vocabulary_overlap" in comparison
        assert "winner" in comparison

        # response1 should be better
        assert comparison["winner"] == "response1"

    def test_compare_responses_vocabulary(self):
        """Test vocabulary overlap in comparison."""
        prompt = "What?"
        response1 = "The cat sat on the mat."
        response2 = "The cat sat on the mat."

        comparison = compare_responses(response1, response2, prompt)

        assert comparison["vocabulary_overlap"]["jaccard_similarity"] == 1.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_whitespace_only(self):
        """Test with whitespace only."""
        analyzer = TextAnalyzer()
        profile = analyzer.profile("   \n\t  ")
        assert profile.word_count == 0

    def test_unicode_text(self):
        """Test with Unicode text."""
        analyzer = TextAnalyzer()
        text = "Hello 世界! Comment ça va?"
        profile = analyzer.profile(text)
        assert profile.word_count > 0

    def test_very_short_text(self):
        """Test with very short text."""
        analyzer = TextAnalyzer()
        profile = analyzer.profile("Hi")
        assert profile.word_count == 1

    def test_single_sentence(self):
        """Test readability with single sentence."""
        analyzer = TextAnalyzer()
        metrics = analyzer.analyze_readability("This is one sentence")
        assert metrics.avg_grade_level >= 0
