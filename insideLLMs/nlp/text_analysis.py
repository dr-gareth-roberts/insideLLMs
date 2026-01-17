"""
Advanced text analysis utilities for LLM exploration.

Provides high-level analysis tools for examining text characteristics,
patterns, and quality metrics relevant to LLM evaluation.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class TextProfile:
    """Comprehensive profile of text characteristics."""

    text: str
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    unique_words: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    lexical_diversity: float = 0.0
    vocabulary: Set[str] = field(default_factory=set)
    word_frequencies: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_text(cls, text: str) -> "TextProfile":
        """Create a profile from text."""
        if not text or not text.strip():
            return cls(text=text)

        # Basic counts
        words = re.findall(r"\b\w+\b", text.lower())
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        word_count = len(words)
        unique_words = len(set(words))
        vocab = set(words)
        word_freq = Counter(words)

        # Averages
        avg_word_len = sum(len(w) for w in words) / word_count if word_count else 0
        avg_sent_len = word_count / len(sentences) if sentences else 0

        # Lexical diversity (type-token ratio)
        lex_div = unique_words / word_count if word_count else 0

        return cls(
            text=text,
            char_count=len(text),
            word_count=word_count,
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs),
            unique_words=unique_words,
            avg_word_length=avg_word_len,
            avg_sentence_length=avg_sent_len,
            lexical_diversity=lex_div,
            vocabulary=vocab,
            word_frequencies=dict(word_freq),
        )


@dataclass
class ReadabilityMetrics:
    """Collection of readability scores."""

    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    smog_index: float = 0.0
    coleman_liau: float = 0.0
    automated_readability: float = 0.0
    avg_grade_level: float = 0.0

    def get_reading_level(self) -> str:
        """Get human-readable reading level description."""
        grade = self.avg_grade_level
        if grade < 5:
            return "elementary"
        elif grade < 8:
            return "middle_school"
        elif grade < 12:
            return "high_school"
        elif grade < 16:
            return "college"
        else:
            return "graduate"


@dataclass
class ContentAnalysis:
    """Analysis of content structure and patterns."""

    has_questions: bool = False
    has_lists: bool = False
    has_code: bool = False
    has_numbers: bool = False
    has_urls: bool = False
    has_emails: bool = False
    question_count: int = 0
    list_item_count: int = 0
    code_block_count: int = 0
    url_count: int = 0
    dominant_structure: str = "prose"


@dataclass
class ToneAnalysis:
    """Analysis of text tone and style."""

    formality_score: float = 0.0  # 0=informal, 1=formal
    confidence_score: float = 0.0  # 0=uncertain, 1=confident
    subjectivity_score: float = 0.0  # 0=objective, 1=subjective
    sentiment_polarity: float = 0.0  # -1=negative, 0=neutral, 1=positive

    def get_formality_level(self) -> str:
        """Get formality level description."""
        if self.formality_score < 0.3:
            return "informal"
        elif self.formality_score < 0.7:
            return "neutral"
        else:
            return "formal"


class TextAnalyzer:
    """Comprehensive text analyzer for LLM output evaluation."""

    # Words indicating informal writing
    INFORMAL_WORDS = {
        "gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "yep", "nope",
        "ok", "okay", "hey", "hi", "bye", "lol", "haha", "wow", "cool",
        "awesome", "stuff", "things", "like", "really", "very", "just",
        "actually", "basically", "literally", "totally", "pretty",
    }

    # Words indicating formal writing
    FORMAL_WORDS = {
        "therefore", "consequently", "furthermore", "moreover", "nevertheless",
        "notwithstanding", "whereas", "hereby", "henceforth", "accordingly",
        "subsequently", "aforementioned", "regarding", "concerning", "pursuant",
        "herein", "therein", "wherein", "hereafter", "theretofore",
    }

    # Hedging words (uncertainty)
    HEDGE_WORDS = {
        "maybe", "perhaps", "possibly", "might", "could", "may", "seem",
        "appear", "suggest", "believe", "think", "assume", "suppose",
        "probably", "likely", "unlikely", "uncertain", "unclear",
    }

    # Confident words
    CONFIDENT_WORDS = {
        "certainly", "definitely", "absolutely", "clearly", "obviously",
        "undoubtedly", "surely", "precisely", "exactly", "always", "never",
        "must", "will", "shall", "proven", "established", "confirmed",
    }

    # Subjective words
    SUBJECTIVE_WORDS = {
        "beautiful", "ugly", "good", "bad", "best", "worst", "amazing",
        "terrible", "wonderful", "horrible", "love", "hate", "feel",
        "believe", "think", "opinion", "prefer", "favorite", "personally",
    }

    def __init__(self):
        """Initialize the analyzer."""
        pass

    def profile(self, text: str) -> TextProfile:
        """Create a comprehensive text profile."""
        return TextProfile.from_text(text)

    def analyze_readability(self, text: str) -> ReadabilityMetrics:
        """Calculate various readability metrics."""
        if not text or not text.strip():
            return ReadabilityMetrics()

        words = re.findall(r"\b\w+\b", text)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        word_count = len(words)
        sentence_count = len(sentences) or 1
        char_count = sum(len(w) for w in words)

        # Count syllables
        syllable_count = sum(self._count_syllables(w) for w in words)

        # Count complex words (3+ syllables)
        complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)

        # Flesch Reading Ease
        fre = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (
            syllable_count / word_count if word_count else 0
        )
        fre = max(0, min(100, fre))

        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * (word_count / sentence_count) + 11.8 * (
            syllable_count / word_count if word_count else 0
        ) - 15.59
        fk_grade = max(0, fk_grade)

        # Gunning Fog Index
        fog = 0.4 * (
            (word_count / sentence_count)
            + 100 * (complex_words / word_count if word_count else 0)
        )
        fog = max(0, fog)

        # SMOG Index (requires 30+ sentences ideally)
        smog = 1.0430 * math.sqrt(complex_words * (30 / sentence_count)) + 3.1291
        smog = max(0, smog) if sentence_count >= 3 else 0

        # Coleman-Liau Index
        L = (char_count / word_count) * 100 if word_count else 0
        S = (sentence_count / word_count) * 100 if word_count else 0
        cli = 0.0588 * L - 0.296 * S - 15.8
        cli = max(0, cli)

        # Automated Readability Index
        ari = 4.71 * (char_count / word_count if word_count else 0) + 0.5 * (
            word_count / sentence_count
        ) - 21.43
        ari = max(0, ari)

        # Average grade level
        grades = [fk_grade, fog, cli, ari]
        if smog > 0:
            grades.append(smog)
        avg_grade = sum(grades) / len(grades)

        return ReadabilityMetrics(
            flesch_reading_ease=round(fre, 2),
            flesch_kincaid_grade=round(fk_grade, 2),
            gunning_fog=round(fog, 2),
            smog_index=round(smog, 2),
            coleman_liau=round(cli, 2),
            automated_readability=round(ari, 2),
            avg_grade_level=round(avg_grade, 2),
        )

    def analyze_content(self, text: str) -> ContentAnalysis:
        """Analyze content structure and patterns."""
        if not text:
            return ContentAnalysis()

        # Detect questions
        questions = re.findall(r"[^.!?]*\?", text)
        has_questions = len(questions) > 0

        # Detect lists (bullets, numbers)
        list_patterns = [
            r"^[\-\*\•]\s+.+$",  # Bullet points
            r"^\d+[\.\)]\s+.+$",  # Numbered lists
        ]
        list_items = []
        for line in text.split("\n"):
            for pattern in list_patterns:
                if re.match(pattern, line.strip()):
                    list_items.append(line)
                    break
        has_lists = len(list_items) > 0

        # Detect code blocks
        code_blocks = re.findall(r"```[\s\S]*?```", text)
        inline_code = re.findall(r"`[^`]+`", text)
        has_code = len(code_blocks) > 0 or len(inline_code) > 0

        # Detect numbers
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        has_numbers = len(numbers) > 0

        # Detect URLs
        urls = re.findall(r"https?://\S+", text)
        has_urls = len(urls) > 0

        # Detect emails
        emails = re.findall(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)
        has_emails = len(emails) > 0

        # Determine dominant structure
        if len(code_blocks) > 0 and len(code_blocks) >= len(list_items):
            dominant = "code"
        elif has_lists and len(list_items) >= len(questions):
            dominant = "list"
        elif has_questions and len(questions) > 2:
            dominant = "q&a"
        else:
            dominant = "prose"

        return ContentAnalysis(
            has_questions=has_questions,
            has_lists=has_lists,
            has_code=has_code,
            has_numbers=has_numbers,
            has_urls=has_urls,
            has_emails=has_emails,
            question_count=len(questions),
            list_item_count=len(list_items),
            code_block_count=len(code_blocks),
            url_count=len(urls),
            dominant_structure=dominant,
        )

    def analyze_tone(self, text: str) -> ToneAnalysis:
        """Analyze the tone and style of text."""
        if not text:
            return ToneAnalysis()

        words = set(re.findall(r"\b\w+\b", text.lower()))
        word_list = re.findall(r"\b\w+\b", text.lower())
        word_count = len(word_list)

        if word_count == 0:
            return ToneAnalysis()

        # Formality score
        informal_count = len(words & self.INFORMAL_WORDS)
        formal_count = len(words & self.FORMAL_WORDS)
        total_markers = informal_count + formal_count
        if total_markers > 0:
            formality = formal_count / total_markers
        else:
            # Use heuristics: contractions, first person, etc.
            contractions = len(re.findall(r"\b\w+'\w+\b", text))
            first_person = sum(1 for w in word_list if w in {"i", "me", "my", "we", "us"})
            formality = 0.5 - (contractions + first_person) / (word_count * 2)
            formality = max(0, min(1, formality))

        # Confidence score
        hedge_count = len(words & self.HEDGE_WORDS)
        confident_count = len(words & self.CONFIDENT_WORDS)
        uncertainty_markers = hedge_count + confident_count
        if uncertainty_markers > 0:
            confidence = confident_count / uncertainty_markers
        else:
            confidence = 0.5

        # Subjectivity score
        subjective_count = len(words & self.SUBJECTIVE_WORDS)
        subjectivity = min(1.0, subjective_count / (word_count * 0.05) if word_count > 0 else 0)

        # Simple sentiment polarity
        positive_words = {"good", "great", "excellent", "amazing", "wonderful", "best", "love"}
        negative_words = {"bad", "terrible", "awful", "worst", "hate", "poor", "horrible"}
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total_sentiment = pos_count + neg_count
        if total_sentiment > 0:
            polarity = (pos_count - neg_count) / total_sentiment
        else:
            polarity = 0.0

        return ToneAnalysis(
            formality_score=round(formality, 3),
            confidence_score=round(confidence, 3),
            subjectivity_score=round(subjectivity, 3),
            sentiment_polarity=round(polarity, 3),
        )

    def compare_texts(
        self, text1: str, text2: str
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Compare two texts across multiple dimensions.

        Returns dict with metric name -> (text1_value, text2_value, difference)
        """
        profile1 = self.profile(text1)
        profile2 = self.profile(text2)

        readability1 = self.analyze_readability(text1)
        readability2 = self.analyze_readability(text2)

        tone1 = self.analyze_tone(text1)
        tone2 = self.analyze_tone(text2)

        return {
            "word_count": (
                profile1.word_count,
                profile2.word_count,
                profile2.word_count - profile1.word_count,
            ),
            "lexical_diversity": (
                profile1.lexical_diversity,
                profile2.lexical_diversity,
                profile2.lexical_diversity - profile1.lexical_diversity,
            ),
            "avg_word_length": (
                profile1.avg_word_length,
                profile2.avg_word_length,
                profile2.avg_word_length - profile1.avg_word_length,
            ),
            "grade_level": (
                readability1.avg_grade_level,
                readability2.avg_grade_level,
                readability2.avg_grade_level - readability1.avg_grade_level,
            ),
            "formality": (
                tone1.formality_score,
                tone2.formality_score,
                tone2.formality_score - tone1.formality_score,
            ),
            "confidence": (
                tone1.confidence_score,
                tone2.confidence_score,
                tone2.confidence_score - tone1.confidence_score,
            ),
        }

    def vocabulary_overlap(
        self, text1: str, text2: str
    ) -> Tuple[float, Set[str], Set[str], Set[str]]:
        """
        Calculate vocabulary overlap between two texts.

        Returns (jaccard_similarity, common_words, unique_to_text1, unique_to_text2)
        """
        words1 = set(re.findall(r"\b\w+\b", text1.lower()))
        words2 = set(re.findall(r"\b\w+\b", text2.lower()))

        common = words1 & words2
        unique1 = words1 - words2
        unique2 = words2 - words1

        union = words1 | words2
        jaccard = len(common) / len(union) if union else 0.0

        return (jaccard, common, unique1, unique2)

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        if len(word) <= 3:
            return 1

        # Remove trailing e
        if word.endswith("e"):
            word = word[:-1]

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        return max(1, count)


@dataclass
class ResponseQualityScore:
    """Quality scoring for LLM responses."""

    relevance: float = 0.0  # How relevant to the prompt
    completeness: float = 0.0  # How complete the response is
    coherence: float = 0.0  # How well-structured and logical
    conciseness: float = 0.0  # How efficient (not verbose)
    overall: float = 0.0

    @classmethod
    def calculate(
        cls,
        response: str,
        prompt: str,
        expected_length: Optional[int] = None,
        required_keywords: Optional[List[str]] = None,
    ) -> "ResponseQualityScore":
        """Calculate quality score for a response."""
        analyzer = TextAnalyzer()

        # Relevance: vocabulary overlap with prompt
        jaccard, common, _, _ = analyzer.vocabulary_overlap(prompt, response)
        relevance = min(1.0, jaccard * 3)  # Scale up since prompts are usually short

        # Check for required keywords
        if required_keywords:
            response_lower = response.lower()
            keyword_matches = sum(
                1 for kw in required_keywords if kw.lower() in response_lower
            )
            keyword_score = keyword_matches / len(required_keywords)
            relevance = (relevance + keyword_score) / 2

        # Completeness: based on response length vs expected
        response_words = len(re.findall(r"\b\w+\b", response))
        if expected_length:
            # Score based on how close to expected length
            ratio = response_words / expected_length
            if ratio < 0.5:
                completeness = ratio * 2
            elif ratio > 2.0:
                completeness = max(0, 1 - (ratio - 2) * 0.5)
            else:
                completeness = 1.0
        else:
            # Default: assume reasonable response is 50-500 words
            if response_words < 10:
                completeness = response_words / 10
            elif response_words > 1000:
                completeness = max(0.5, 1 - (response_words - 1000) / 2000)
            else:
                completeness = 1.0

        # Coherence: based on sentence structure and connectors
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Check for transition words
        connectors = {
            "however", "therefore", "furthermore", "moreover", "additionally",
            "consequently", "thus", "hence", "first", "second", "third",
            "finally", "in conclusion", "to summarize", "for example",
        }
        response_lower = response.lower()
        connector_count = sum(1 for c in connectors if c in response_lower)

        # Score based on sentence count and connectors
        if len(sentences) > 1:
            coherence = min(1.0, 0.5 + connector_count * 0.1)
        else:
            coherence = 0.5 if response_words > 10 else 0.3

        # Conciseness: penalize excessive repetition and filler
        profile = analyzer.profile(response)
        word_list = re.findall(r"\b\w+\b", response.lower())

        # Check for repetition
        word_freq = Counter(word_list)
        if word_freq:
            most_common_freq = max(word_freq.values())
            repetition_ratio = most_common_freq / len(word_list) if word_list else 0
        else:
            repetition_ratio = 0

        # Filler words
        fillers = {"very", "really", "just", "actually", "basically", "literally"}
        filler_count = sum(1 for w in word_list if w in fillers)
        filler_ratio = filler_count / len(word_list) if word_list else 0

        conciseness = 1.0 - (repetition_ratio * 0.5) - (filler_ratio * 0.5)
        conciseness = max(0, conciseness)

        # Overall score (weighted average)
        overall = (
            relevance * 0.3
            + completeness * 0.25
            + coherence * 0.25
            + conciseness * 0.2
        )

        return cls(
            relevance=round(relevance, 3),
            completeness=round(completeness, 3),
            coherence=round(coherence, 3),
            conciseness=round(conciseness, 3),
            overall=round(overall, 3),
        )


def analyze_text(text: str) -> Dict:
    """
    Perform comprehensive text analysis.

    Returns a dictionary with profile, readability, content, and tone analysis.
    """
    analyzer = TextAnalyzer()
    return {
        "profile": analyzer.profile(text),
        "readability": analyzer.analyze_readability(text),
        "content": analyzer.analyze_content(text),
        "tone": analyzer.analyze_tone(text),
    }


def score_response(
    response: str,
    prompt: str,
    expected_length: Optional[int] = None,
    required_keywords: Optional[List[str]] = None,
) -> ResponseQualityScore:
    """Score the quality of an LLM response."""
    return ResponseQualityScore.calculate(
        response, prompt, expected_length, required_keywords
    )


def compare_responses(
    response1: str, response2: str, prompt: str
) -> Dict[str, Dict]:
    """
    Compare two responses to the same prompt.

    Returns comparison metrics for both responses.
    """
    analyzer = TextAnalyzer()

    score1 = score_response(response1, prompt)
    score2 = score_response(response2, prompt)

    comparison = analyzer.compare_texts(response1, response2)
    vocab_overlap = analyzer.vocabulary_overlap(response1, response2)

    return {
        "response1_score": score1,
        "response2_score": score2,
        "comparison": comparison,
        "vocabulary_overlap": {
            "jaccard_similarity": vocab_overlap[0],
            "common_word_count": len(vocab_overlap[1]),
            "unique_to_response1": len(vocab_overlap[2]),
            "unique_to_response2": len(vocab_overlap[3]),
        },
        "winner": "response1" if score1.overall > score2.overall else "response2",
    }
