"""
Response quality scoring and analysis utilities.

Provides tools for:
- Multi-dimensional quality assessment
- Response completeness checking
- Coherence analysis
- Relevance scoring
- Quality benchmarking
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class QualityDimension(Enum):
    """Dimensions of response quality."""

    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    CONCISENESS = "conciseness"
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    CLARITY = "clarity"
    SPECIFICITY = "specificity"


@dataclass
class DimensionScore:
    """Score for a single quality dimension."""

    dimension: QualityDimension
    score: float  # 0-1
    confidence: float = 1.0
    explanation: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "evidence": self.evidence,
        }


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""

    prompt: str
    response: str
    overall_score: float
    dimension_scores: dict[QualityDimension, DimensionScore] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if response meets quality threshold."""
        return self.overall_score >= 0.7

    def get_score(self, dimension: QualityDimension) -> Optional[float]:
        """Get score for a dimension."""
        score_obj = self.dimension_scores.get(dimension)
        return score_obj.score if score_obj else None

    def get_weakest_dimensions(self, n: int = 3) -> list[tuple[QualityDimension, float]]:
        """Get n weakest dimensions."""
        sorted_dims = sorted(self.dimension_scores.items(), key=lambda x: x[1].score)
        return [(d, s.score) for d, s in sorted_dims[:n]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "passed": self.passed,
            "dimension_scores": {d.value: s.to_dict() for d, s in self.dimension_scores.items()},
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


class RelevanceScorer:
    """Score relevance of response to prompt."""

    def __init__(self):
        """Initialize scorer."""
        pass

    def score(self, prompt: str, response: str) -> DimensionScore:
        """Score relevance.

        Args:
            prompt: Original prompt.
            response: Model response.

        Returns:
            Relevance score.
        """
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        # Extract key content terms from prompt (words 3+ chars)
        prompt_words = set(re.findall(r"\b\w{3,}\b", prompt_lower))
        # Remove common stop words that don't carry semantic meaning
        stop_words = {
            "the",
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "is",
            "are",
            "was",
            "were",
            "can",
            "could",
            "would",
            "should",
            "does",
            "did",
            "has",
            "have",
            "had",
            "been",
            "being",
            "this",
            "that",
            "these",
            "those",
            "there",
            "here",
            "with",
            "from",
            "about",
            "into",
            "through",
            "during",
            "for",
            "and",
            "but",
            "not",
            "you",
            "your",
            "our",
            "their",
            "will",
            "may",
            "might",
            "must",
            "some",
            "any",
            "all",
            "most",
            "more",
            "such",
            "than",
        }
        prompt_content_words = prompt_words - stop_words

        response_words = set(re.findall(r"\b\w{3,}\b", response_lower))

        # Calculate term overlap - this is the core relevance signal
        overlap = prompt_content_words & response_words
        overlap_ratio = len(overlap) / len(prompt_content_words) if prompt_content_words else 0

        evidence = []

        # Term overlap is the dominant factor (weight: 0.8)
        # A response that shares no content words with the prompt is almost certainly irrelevant
        term_score = overlap_ratio * 0.8
        evidence.append(f"Term overlap: {len(overlap)}/{len(prompt_content_words)} key terms")

        # Length appropriateness (weight: 0.2)
        # Very short or very long responses relative to prompt complexity get penalized
        response_words_count = len(response.split())
        prompt_words_count = len(prompt.split())

        if response_words_count < 3:
            length_score = 0.0
            evidence.append("Response too short")
        elif prompt_words_count > 0:
            ratio = response_words_count / prompt_words_count
            if 1.0 <= ratio <= 15.0:
                length_score = 0.2
            elif 0.5 <= ratio < 1.0 or 15.0 < ratio <= 30.0:
                length_score = 0.1
            else:
                length_score = 0.05
        else:
            length_score = 0.1

        score = term_score + length_score

        explanation = (
            "High relevance"
            if score >= 0.7
            else "Moderate relevance"
            if score >= 0.4
            else "Low relevance"
        )

        return DimensionScore(
            dimension=QualityDimension.RELEVANCE,
            score=min(1.0, max(0.0, score)),
            explanation=explanation,
            evidence=evidence,
        )


class CompletenessScorer:
    """Score completeness of response."""

    # Indicators of incomplete responses
    INCOMPLETE_INDICATORS = [
        "i don't know",
        "i'm not sure",
        "i cannot",
        "unable to",
        "no information",
        "...",
        "etc",
        "and so on",
        "to be continued",
    ]

    # Hedging language that suggests incomplete thinking
    HEDGING_INDICATORS = [
        "i think",
        "i guess",
        "maybe",
        "perhaps",
        "probably",
        "might be",
        "not sure",
        "i believe",
    ]

    # Indicators of complete responses
    COMPLETE_INDICATORS = [
        "in conclusion",
        "to summarize",
        "in summary",
        "finally",
        "overall",
        "therefore",
    ]

    def score(self, prompt: str, response: str) -> DimensionScore:
        """Score completeness.

        Args:
            prompt: Original prompt.
            response: Model response.

        Returns:
            Completeness score.
        """
        response_lower = response.lower()
        prompt_lower = prompt.lower()
        evidence = []

        # Check for incomplete indicators
        incomplete_count = sum(1 for ind in self.INCOMPLETE_INDICATORS if ind in response_lower)

        # Check for hedging language
        hedging_count = sum(1 for ind in self.HEDGING_INDICATORS if ind in response_lower)

        # Check for complete indicators
        complete_count = sum(1 for ind in self.COMPLETE_INDICATORS if ind in response_lower)

        # Check response structure
        sentence_count = len(re.findall(r"[.!?]", response))
        has_multiple_sentences = sentence_count >= 2
        has_paragraphs = "\n\n" in response or len(response) > 200

        # Count questions in prompt
        prompt_questions = len(re.findall(r"\?", prompt))

        # Check if response addresses multiple questions
        response_points = len(re.findall(r"^[-•*\d+.]\s", response, re.MULTILINE))

        # Check for "in detail" or similar requests in prompt
        requests_detail = any(
            phrase in prompt_lower
            for phrase in ["in detail", "explain", "elaborate", "describe", "comprehensive"]
        )

        # Word counts
        response_word_count = len(response.split())
        len(prompt.split())

        # Score calculation - start based on basic response length adequacy
        if response_word_count < 10:
            score = 0.3  # Very short responses start low
            evidence.append("Very short response")
        elif response_word_count < 25:
            score = 0.4
            evidence.append("Brief response")
        else:
            score = 0.5

        # Penalize incomplete indicators
        score -= incomplete_count * 0.15
        if incomplete_count > 0:
            evidence.append(f"Found {incomplete_count} incomplete indicators")

        # Penalize hedging language (suggests uncertainty/incompleteness)
        score -= hedging_count * 0.1
        if hedging_count > 0:
            evidence.append(f"Contains hedging language ({hedging_count} instances)")

        # Reward complete indicators
        score += complete_count * 0.1
        if complete_count > 0:
            evidence.append(f"Found {complete_count} completion indicators")

        # Structure bonus
        if has_multiple_sentences:
            score += 0.15
            evidence.append("Has multiple sentences")

        if has_paragraphs:
            score += 0.1
            evidence.append("Has substantial content")

        # Multiple questions addressed
        if prompt_questions > 1 and response_points >= prompt_questions:
            score += 0.15
            evidence.append("Addresses multiple questions")

        # Penalize if prompt requests detail but response is short
        if requests_detail and response_word_count < 50:
            score -= 0.2
            evidence.append("Prompt requested detail but response is brief")

        score = max(0.0, min(1.0, score))

        explanation = (
            "Response appears complete"
            if score >= 0.7
            else "Response may be incomplete"
            if score >= 0.4
            else "Response appears incomplete"
        )

        return DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=score,
            explanation=explanation,
            evidence=evidence,
        )


class CoherenceScorer:
    """Score coherence and logical flow."""

    # Transition words indicating coherent structure
    TRANSITIONS = {
        "first",
        "second",
        "third",
        "finally",
        "next",
        "then",
        "however",
        "therefore",
        "consequently",
        "moreover",
        "additionally",
        "furthermore",
        "in addition",
        "as a result",
        "because",
        "since",
        "although",
        "despite",
        "meanwhile",
    }

    def score(self, prompt: str, response: str) -> DimensionScore:
        """Score coherence.

        Args:
            prompt: Original prompt.
            response: Model response.

        Returns:
            Coherence score.
        """
        response_lower = response.lower()
        evidence = []

        # Count transition words
        transition_count = sum(
            1 for t in self.TRANSITIONS if re.search(rf"\b{t}\b", response_lower)
        )

        # Check sentence structure
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Analyze sentence lengths
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

        # Check for very short or very long sentences
        problematic_sentences = sum(1 for l in sentence_lengths if l < 3 or l > 50)

        # Check for repeated starts (sign of incoherence)
        sentence_starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
        unique_starts = len(set(sentence_starts))
        start_diversity = unique_starts / len(sentence_starts) if sentence_starts else 0

        # Score calculation
        score = 0.5

        # Transition words (max 0.3)
        score += min(0.3, transition_count * 0.05)
        if transition_count >= 3:
            evidence.append(f"Good use of transitions ({transition_count} found)")

        # Sentence length variation (max 0.2)
        if 8 <= avg_length <= 25:
            score += 0.2
            evidence.append("Appropriate sentence lengths")
        elif 5 <= avg_length <= 35:
            score += 0.1

        # Sentence start diversity (max 0.15)
        if start_diversity >= 0.7:
            score += 0.15
            evidence.append("Good sentence variety")
        elif start_diversity >= 0.5:
            score += 0.08

        # Penalize problematic sentences
        score -= problematic_sentences * 0.05

        score = max(0.0, min(1.0, score))

        explanation = (
            "Well-structured and coherent"
            if score >= 0.7
            else "Moderately coherent"
            if score >= 0.4
            else "May lack coherence"
        )

        return DimensionScore(
            dimension=QualityDimension.COHERENCE,
            score=score,
            explanation=explanation,
            evidence=evidence,
        )


class ConcisenessScorer:
    """Score conciseness of response."""

    # Filler words and phrases
    FILLERS = [
        "basically",
        "actually",
        "literally",
        "really",
        "very",
        "kind of",
        "sort of",
        "i think",
        "i believe",
        "in my opinion",
        "as you know",
        "as mentioned",
        "it should be noted",
        "it is important to note",
        "needless to say",
    ]

    # Redundant phrases
    REDUNDANT = [
        "absolutely essential",
        "advance planning",
        "basic fundamentals",
        "completely finished",
        "end result",
        "final outcome",
        "future plans",
        "past history",
        "true fact",
    ]

    def score(self, prompt: str, response: str) -> DimensionScore:
        """Score conciseness.

        Args:
            prompt: Original prompt.
            response: Model response.

        Returns:
            Conciseness score.
        """
        response_lower = response.lower()
        evidence = []

        # Count filler words
        filler_count = sum(1 for f in self.FILLERS if f in response_lower)

        # Count redundant phrases
        redundant_count = sum(1 for r in self.REDUNDANT if r in response_lower)

        # Analyze word count vs information density
        words = response.split()
        word_count = len(words)

        # Check for repetition
        unique_words = {w.lower() for w in words if len(w) > 3}
        len(unique_words) / word_count if word_count > 0 else 0

        # Calculate information density (unique words per total words)
        info_density = len(unique_words) / word_count if word_count > 0 else 0

        # Score calculation
        score = 0.7  # Start high, penalize verbosity

        # Penalize fillers
        score -= filler_count * 0.05
        if filler_count > 3:
            evidence.append(f"Contains {filler_count} filler words/phrases")

        # Penalize redundancy
        score -= redundant_count * 0.1
        if redundant_count > 0:
            evidence.append(f"Contains {redundant_count} redundant phrases")

        # Information density
        if info_density >= 0.5:
            score += 0.15
            evidence.append("Good information density")
        elif info_density < 0.3:
            score -= 0.1
            evidence.append("Low information density")

        # Penalize excessive length relative to prompt
        if word_count > len(prompt.split()) * 10 and word_count > 200:
            score -= 0.15
            evidence.append("Response may be overly verbose")

        score = max(0.0, min(1.0, score))

        explanation = (
            "Concise and to the point"
            if score >= 0.7
            else "Somewhat verbose"
            if score >= 0.4
            else "Could be more concise"
        )

        return DimensionScore(
            dimension=QualityDimension.CONCISENESS,
            score=score,
            explanation=explanation,
            evidence=evidence,
        )


class ClarityScorer:
    """Score clarity of response."""

    # Complex/jargon indicators
    JARGON_PATTERNS = [
        r"\b\w{15,}\b",  # Very long words
        r"\([^)]{50,}\)",  # Long parentheticals
    ]

    def score(self, prompt: str, response: str) -> DimensionScore:
        """Score clarity.

        Args:
            prompt: Original prompt.
            response: Model response.

        Returns:
            Clarity score.
        """
        evidence = []

        # Calculate readability metrics
        words = response.split()
        word_count = len(words)

        if word_count == 0:
            return DimensionScore(
                dimension=QualityDimension.CLARITY,
                score=0.0,
                explanation="Empty response",
                evidence=["No content to evaluate"],
            )

        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences) or 1

        # Average words per sentence
        avg_words_per_sentence = word_count / sentence_count

        # Count complex words (3+ syllables, approximated by length)
        complex_words = sum(1 for w in words if len(w) > 8)
        complex_ratio = complex_words / word_count

        # Check for jargon
        jargon_matches = sum(len(re.findall(p, response)) for p in self.JARGON_PATTERNS)

        # Check for clear structure
        has_lists = bool(re.search(r"^[-•*\d+.]\s", response, re.MULTILINE))
        has_headings = bool(re.search(r"^#+\s|^[A-Z][^.!?]*:$", response, re.MULTILINE))

        # Score calculation
        score = 0.5

        # Sentence length (ideal 15-25 words)
        if 15 <= avg_words_per_sentence <= 25:
            score += 0.2
            evidence.append("Good sentence length")
        elif avg_words_per_sentence < 10:
            score += 0.1
            evidence.append("Short sentences (may lack detail)")
        elif avg_words_per_sentence > 35:
            score -= 0.1
            evidence.append("Long sentences may hurt readability")

        # Complex words (ideal < 15%)
        if complex_ratio < 0.15:
            score += 0.2
            evidence.append("Good vocabulary accessibility")
        elif complex_ratio > 0.3:
            score -= 0.1
            evidence.append("Many complex words")

        # Jargon
        score -= jargon_matches * 0.05

        # Structure bonus
        if has_lists:
            score += 0.1
            evidence.append("Uses lists for clarity")
        if has_headings:
            score += 0.1
            evidence.append("Uses headings for organization")

        score = max(0.0, min(1.0, score))

        explanation = (
            "Clear and easy to understand"
            if score >= 0.7
            else "Moderately clear"
            if score >= 0.4
            else "Could be clearer"
        )

        return DimensionScore(
            dimension=QualityDimension.CLARITY,
            score=score,
            explanation=explanation,
            evidence=evidence,
        )


class SpecificityScorer:
    """Score specificity of response."""

    # Vague terms
    VAGUE_TERMS = [
        "some",
        "many",
        "few",
        "often",
        "sometimes",
        "usually",
        "things",
        "stuff",
        "something",
        "somehow",
        "somewhat",
        "a lot",
        "various",
        "several",
        "certain",
        "particular",
    ]

    # Specific indicators
    SPECIFIC_PATTERNS = [
        r"\b\d+(?:\.\d+)?%\b",  # Percentages
        r"\b\d{4}\b",  # Years
        r"\$\d+",  # Dollar amounts
        r"\b\d+\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",  # Time durations
    ]

    def score(self, prompt: str, response: str) -> DimensionScore:
        """Score specificity.

        Args:
            prompt: Original prompt.
            response: Model response.

        Returns:
            Specificity score.
        """
        response_lower = response.lower()
        evidence = []

        # Count vague terms
        vague_count = sum(1 for v in self.VAGUE_TERMS if re.search(rf"\b{v}\b", response_lower))

        # Count specific patterns
        specific_count = sum(
            len(re.findall(p, response, re.IGNORECASE)) for p in self.SPECIFIC_PATTERNS
        )

        # Check for proper nouns (capitalized words not at sentence start)
        proper_nouns = len(re.findall(r"(?<![.!?]\s)[A-Z][a-z]+", response))

        # Check for examples
        has_examples = any(
            indicator in response_lower
            for indicator in ["for example", "such as", "e.g.", "for instance", "including"]
        )

        # Word count
        word_count = len(response.split())

        # Score calculation
        score = 0.5

        # Penalize vague terms
        vague_ratio = vague_count / (word_count / 50 + 1)  # Normalized by text length
        score -= min(0.2, vague_ratio * 0.05)
        if vague_count > 5:
            evidence.append(f"Contains {vague_count} vague terms")

        # Reward specific patterns
        score += min(0.2, specific_count * 0.05)
        if specific_count >= 3:
            evidence.append(f"Contains {specific_count} specific data points")

        # Reward proper nouns
        score += min(0.15, proper_nouns * 0.02)
        if proper_nouns >= 3:
            evidence.append("References specific entities")

        # Reward examples
        if has_examples:
            score += 0.15
            evidence.append("Provides examples")

        score = max(0.0, min(1.0, score))

        explanation = (
            "Specific and detailed"
            if score >= 0.7
            else "Moderately specific"
            if score >= 0.4
            else "Could be more specific"
        )

        return DimensionScore(
            dimension=QualityDimension.SPECIFICITY,
            score=score,
            explanation=explanation,
            evidence=evidence,
        )


class ResponseQualityAnalyzer:
    """Comprehensive response quality analyzer."""

    def __init__(
        self,
        dimensions: Optional[list[QualityDimension]] = None,
        weights: Optional[dict[QualityDimension, float]] = None,
    ):
        """Initialize analyzer.

        Args:
            dimensions: Dimensions to analyze (default: all).
            weights: Custom weights for each dimension.
        """
        self.dimensions = dimensions or list(QualityDimension)
        self.weights = weights or dict.fromkeys(QualityDimension, 1.0)

        # Initialize scorers
        self._scorers = {
            QualityDimension.RELEVANCE: RelevanceScorer(),
            QualityDimension.COMPLETENESS: CompletenessScorer(),
            QualityDimension.COHERENCE: CoherenceScorer(),
            QualityDimension.CONCISENESS: ConcisenessScorer(),
            QualityDimension.CLARITY: ClarityScorer(),
            QualityDimension.SPECIFICITY: SpecificityScorer(),
        }

    def analyze(self, prompt: str, response: str) -> QualityReport:
        """Analyze response quality.

        Args:
            prompt: Original prompt.
            response: Model response.

        Returns:
            Quality report.
        """
        dimension_scores: dict[QualityDimension, DimensionScore] = {}
        issues: list[str] = []
        suggestions: list[str] = []

        # Score each dimension
        for dimension in self.dimensions:
            if dimension in self._scorers:
                score = self._scorers[dimension].score(prompt, response)
                dimension_scores[dimension] = score

                # Track issues
                if score.score < 0.5:
                    issues.append(f"Low {dimension.value}: {score.explanation}")
                    suggestions.append(f"Improve {dimension.value}")

        # Calculate overall score (weighted average)
        total_weight = sum(self.weights.get(d, 1.0) for d in dimension_scores)
        weighted_sum = sum(
            dimension_scores[d].score * self.weights.get(d, 1.0) for d in dimension_scores
        )
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0

        return QualityReport(
            prompt=prompt,
            response=response,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            issues=issues,
            suggestions=suggestions,
        )

    def quick_check(self, prompt: str, response: str) -> tuple[float, bool, list[str]]:
        """Quick quality check.

        Args:
            prompt: Original prompt.
            response: Model response.

        Returns:
            Tuple of (overall_score, passed, issues).
        """
        report = self.analyze(prompt, response)
        return report.overall_score, report.passed, report.issues


@dataclass
class ComparisonResult:
    """Result of comparing two responses."""

    prompt: str
    response_a: str
    response_b: str
    winner: str  # "A", "B", or "tie"
    score_a: float
    score_b: float
    dimension_comparison: dict[str, str]  # dimension -> winner
    reasoning: str


class ResponseComparator:
    """Compare quality of two responses."""

    def __init__(self, analyzer: Optional[ResponseQualityAnalyzer] = None):
        """Initialize comparator.

        Args:
            analyzer: Quality analyzer to use.
        """
        self.analyzer = analyzer or ResponseQualityAnalyzer()

    def compare(self, prompt: str, response_a: str, response_b: str) -> ComparisonResult:
        """Compare two responses.

        Args:
            prompt: Original prompt.
            response_a: First response.
            response_b: Second response.

        Returns:
            Comparison result.
        """
        report_a = self.analyzer.analyze(prompt, response_a)
        report_b = self.analyzer.analyze(prompt, response_b)

        # Compare each dimension
        dimension_comparison = {}
        for dimension in self.analyzer.dimensions:
            score_a = report_a.get_score(dimension)
            score_b = report_b.get_score(dimension)

            if score_a is not None and score_b is not None:
                if abs(score_a - score_b) < 0.05:
                    dimension_comparison[dimension.value] = "tie"
                elif score_a > score_b:
                    dimension_comparison[dimension.value] = "A"
                else:
                    dimension_comparison[dimension.value] = "B"

        # Determine overall winner
        if abs(report_a.overall_score - report_b.overall_score) < 0.05:
            winner = "tie"
            reasoning = "Both responses are of similar quality"
        elif report_a.overall_score > report_b.overall_score:
            winner = "A"
            a_wins = [d for d, w in dimension_comparison.items() if w == "A"]
            reasoning = f"Response A wins in: {', '.join(a_wins)}"
        else:
            winner = "B"
            b_wins = [d for d, w in dimension_comparison.items() if w == "B"]
            reasoning = f"Response B wins in: {', '.join(b_wins)}"

        return ComparisonResult(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            winner=winner,
            score_a=report_a.overall_score,
            score_b=report_b.overall_score,
            dimension_comparison=dimension_comparison,
            reasoning=reasoning,
        )


# Convenience functions


def analyze_quality(prompt: str, response: str) -> QualityReport:
    """Analyze response quality.

    Args:
        prompt: Original prompt.
        response: Model response.

    Returns:
        Quality report.
    """
    analyzer = ResponseQualityAnalyzer()
    return analyzer.analyze(prompt, response)


def quick_quality_check(prompt: str, response: str) -> tuple[float, bool, list[str]]:
    """Quick quality check.

    Args:
        prompt: Original prompt.
        response: Model response.

    Returns:
        Tuple of (score, passed, issues).
    """
    analyzer = ResponseQualityAnalyzer()
    return analyzer.quick_check(prompt, response)


def compare_responses(prompt: str, response_a: str, response_b: str) -> ComparisonResult:
    """Compare two responses.

    Args:
        prompt: Original prompt.
        response_a: First response.
        response_b: Second response.

    Returns:
        Comparison result.
    """
    comparator = ResponseComparator()
    return comparator.compare(prompt, response_a, response_b)
