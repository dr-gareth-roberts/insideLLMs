"""
Model output diversity and creativity metrics.

Provides tools for:
- Measuring lexical and semantic diversity
- Creativity and novelty scoring
- Repetition detection
- Output variability analysis across samples
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class DiversityMetric(Enum):
    """Types of diversity metrics."""

    TYPE_TOKEN_RATIO = "type_token_ratio"
    HAPAX_LEGOMENA = "hapax_legomena"
    YULES_K = "yules_k"
    SIMPSONS_D = "simpsons_d"
    ENTROPY = "entropy"
    MTLD = "mtld"  # Measure of Textual Lexical Diversity


class CreativityDimension(Enum):
    """Dimensions of creativity."""

    NOVELTY = "novelty"
    UNEXPECTEDNESS = "unexpectedness"
    ELABORATION = "elaboration"
    FLEXIBILITY = "flexibility"
    FLUENCY = "fluency"


@dataclass
class DiversityScore:
    """Score for a diversity metric."""

    metric: DiversityMetric
    value: float
    interpretation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric.value,
            "value": round(self.value, 4),
            "interpretation": self.interpretation,
            "details": self.details,
        }


@dataclass
class RepetitionAnalysis:
    """Analysis of repetition in text."""

    repeated_phrases: List[Tuple[str, int]]  # (phrase, count)
    repeated_words: List[Tuple[str, int]]
    repetition_score: float  # 0-1, 0 = no repetition
    longest_repeated_sequence: str
    n_gram_repetition: Dict[int, float]  # n -> repetition rate
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_significant_repetition(self) -> bool:
        """Check if text has significant repetition."""
        return self.repetition_score > 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repeated_phrases": self.repeated_phrases[:10],
            "repeated_words": self.repeated_words[:10],
            "repetition_score": round(self.repetition_score, 4),
            "longest_repeated_sequence": self.longest_repeated_sequence[:100],
            "n_gram_repetition": {k: round(v, 4) for k, v in self.n_gram_repetition.items()},
            "has_significant_repetition": self.has_significant_repetition,
            "metadata": self.metadata,
        }


@dataclass
class CreativityScore:
    """Score for creativity assessment."""

    overall_score: float  # 0-1
    dimension_scores: Dict[CreativityDimension, float]
    interpretation: str
    strengths: List[str]
    weaknesses: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def creativity_level(self) -> str:
        """Categorize creativity level."""
        if self.overall_score >= 0.8:
            return "highly_creative"
        elif self.overall_score >= 0.6:
            return "creative"
        elif self.overall_score >= 0.4:
            return "moderate"
        elif self.overall_score >= 0.2:
            return "low"
        else:
            return "minimal"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": round(self.overall_score, 4),
            "creativity_level": self.creativity_level,
            "dimension_scores": {
                d.value: round(s, 4) for d, s in self.dimension_scores.items()
            },
            "interpretation": self.interpretation,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "metadata": self.metadata,
        }


@dataclass
class VariabilityAnalysis:
    """Analysis of variability across multiple outputs."""

    n_samples: int
    mean_similarity: float  # Average pairwise similarity
    std_similarity: float
    unique_tokens_ratio: float  # Unique tokens / total tokens
    semantic_spread: float  # How spread out responses are semantically
    clustering_coefficient: float  # How much responses cluster
    outlier_indices: List[int]  # Indices of outlier responses
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_diverse(self) -> bool:
        """Check if outputs are diverse."""
        return self.mean_similarity < 0.7 and self.semantic_spread > 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "mean_similarity": round(self.mean_similarity, 4),
            "std_similarity": round(self.std_similarity, 4),
            "unique_tokens_ratio": round(self.unique_tokens_ratio, 4),
            "semantic_spread": round(self.semantic_spread, 4),
            "clustering_coefficient": round(self.clustering_coefficient, 4),
            "outlier_indices": self.outlier_indices,
            "is_diverse": self.is_diverse,
            "metadata": self.metadata,
        }


@dataclass
class DiversityReport:
    """Comprehensive diversity analysis report."""

    text: str
    lexical_diversity: Dict[DiversityMetric, DiversityScore]
    repetition: RepetitionAnalysis
    creativity: Optional[CreativityScore]
    overall_diversity_score: float
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text_length": len(self.text),
            "lexical_diversity": {
                k.value: v.to_dict() for k, v in self.lexical_diversity.items()
            },
            "repetition": self.repetition.to_dict(),
            "creativity": self.creativity.to_dict() if self.creativity else None,
            "overall_diversity_score": round(self.overall_diversity_score, 4),
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


class LexicalDiversityAnalyzer:
    """Analyzes lexical diversity of text."""

    def __init__(self, tokenize_fn: Optional[Callable[[str], List[str]]] = None):
        """Initialize analyzer."""
        self.tokenize = tokenize_fn or self._default_tokenize

    def _default_tokenize(self, text: str) -> List[str]:
        """Default tokenization: lowercase words."""
        return re.findall(r'\b\w+\b', text.lower())

    def type_token_ratio(self, text: str) -> DiversityScore:
        """Calculate Type-Token Ratio (TTR)."""
        tokens = self.tokenize(text)
        if not tokens:
            return DiversityScore(
                metric=DiversityMetric.TYPE_TOKEN_RATIO,
                value=0.0,
                interpretation="No tokens found",
            )

        types = set(tokens)
        ttr = len(types) / len(tokens)

        if ttr >= 0.7:
            interp = "High lexical diversity"
        elif ttr >= 0.5:
            interp = "Moderate lexical diversity"
        else:
            interp = "Low lexical diversity (repetitive vocabulary)"

        return DiversityScore(
            metric=DiversityMetric.TYPE_TOKEN_RATIO,
            value=ttr,
            interpretation=interp,
            details={"n_types": len(types), "n_tokens": len(tokens)},
        )

    def hapax_legomena_ratio(self, text: str) -> DiversityScore:
        """Calculate ratio of words appearing only once."""
        tokens = self.tokenize(text)
        if not tokens:
            return DiversityScore(
                metric=DiversityMetric.HAPAX_LEGOMENA,
                value=0.0,
                interpretation="No tokens found",
            )

        freq = Counter(tokens)
        hapax = sum(1 for count in freq.values() if count == 1)
        ratio = hapax / len(tokens)

        if ratio >= 0.5:
            interp = "High proportion of unique words"
        elif ratio >= 0.3:
            interp = "Moderate proportion of unique words"
        else:
            interp = "Low proportion of unique words"

        return DiversityScore(
            metric=DiversityMetric.HAPAX_LEGOMENA,
            value=ratio,
            interpretation=interp,
            details={"hapax_count": hapax, "total_tokens": len(tokens)},
        )

    def yules_k(self, text: str) -> DiversityScore:
        """Calculate Yule's K characteristic."""
        tokens = self.tokenize(text)
        if len(tokens) < 2:
            return DiversityScore(
                metric=DiversityMetric.YULES_K,
                value=0.0,
                interpretation="Insufficient tokens",
            )

        freq = Counter(tokens)
        n = len(tokens)
        m1 = n  # Sum of frequencies
        m2 = sum(f * f for f in freq.values())

        if m1 == m2:
            k = 0.0
        else:
            k = 10000 * (m2 - m1) / (m1 * m1)

        # Lower K = more diverse
        if k < 50:
            interp = "High vocabulary diversity (low repetition)"
        elif k < 100:
            interp = "Moderate vocabulary diversity"
        else:
            interp = "Low vocabulary diversity (high repetition)"

        return DiversityScore(
            metric=DiversityMetric.YULES_K,
            value=k,
            interpretation=interp,
            details={"m1": m1, "m2": m2},
        )

    def simpsons_diversity(self, text: str) -> DiversityScore:
        """Calculate Simpson's Diversity Index."""
        tokens = self.tokenize(text)
        n = len(tokens)
        if n < 2:
            return DiversityScore(
                metric=DiversityMetric.SIMPSONS_D,
                value=0.0,
                interpretation="Insufficient tokens",
            )

        freq = Counter(tokens)
        d = sum(f * (f - 1) for f in freq.values()) / (n * (n - 1))
        diversity = 1 - d  # Invert so higher = more diverse

        if diversity >= 0.9:
            interp = "Very high diversity"
        elif diversity >= 0.7:
            interp = "High diversity"
        elif diversity >= 0.5:
            interp = "Moderate diversity"
        else:
            interp = "Low diversity"

        return DiversityScore(
            metric=DiversityMetric.SIMPSONS_D,
            value=diversity,
            interpretation=interp,
        )

    def entropy(self, text: str) -> DiversityScore:
        """Calculate Shannon entropy of word distribution."""
        tokens = self.tokenize(text)
        if not tokens:
            return DiversityScore(
                metric=DiversityMetric.ENTROPY,
                value=0.0,
                interpretation="No tokens found",
            )

        freq = Counter(tokens)
        n = len(tokens)
        probs = [f / n for f in freq.values()]

        h = -sum(p * math.log2(p) for p in probs if p > 0)

        # Normalize by max possible entropy
        max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1
        normalized = h / max_entropy if max_entropy > 0 else 0

        if normalized >= 0.9:
            interp = "Near-maximum entropy (highly uniform distribution)"
        elif normalized >= 0.7:
            interp = "High entropy (diverse word usage)"
        elif normalized >= 0.5:
            interp = "Moderate entropy"
        else:
            interp = "Low entropy (skewed word distribution)"

        return DiversityScore(
            metric=DiversityMetric.ENTROPY,
            value=normalized,
            interpretation=interp,
            details={"raw_entropy": h, "max_entropy": max_entropy},
        )

    def mtld(self, text: str, threshold: float = 0.72) -> DiversityScore:
        """Calculate Measure of Textual Lexical Diversity (MTLD)."""
        tokens = self.tokenize(text)
        if len(tokens) < 10:
            return DiversityScore(
                metric=DiversityMetric.MTLD,
                value=0.0,
                interpretation="Insufficient tokens for MTLD",
            )

        def _mtld_forward(tokens: List[str], threshold: float) -> float:
            """Calculate MTLD in forward direction."""
            factor_count = 0
            factor_length = 0
            current_ttr = 1.0
            types: Set[str] = set()

            for token in tokens:
                types.add(token)
                factor_length += 1
                current_ttr = len(types) / factor_length

                if current_ttr <= threshold:
                    factor_count += 1
                    types = set()
                    factor_length = 0
                    current_ttr = 1.0

            # Handle partial factor
            if factor_length > 0:
                factor_count += (1 - current_ttr) / (1 - threshold)

            return len(tokens) / factor_count if factor_count > 0 else len(tokens)

        # Calculate in both directions and average
        forward = _mtld_forward(tokens, threshold)
        backward = _mtld_forward(tokens[::-1], threshold)
        mtld_value = (forward + backward) / 2

        if mtld_value >= 100:
            interp = "Very high lexical diversity"
        elif mtld_value >= 70:
            interp = "High lexical diversity"
        elif mtld_value >= 50:
            interp = "Moderate lexical diversity"
        else:
            interp = "Low lexical diversity"

        return DiversityScore(
            metric=DiversityMetric.MTLD,
            value=mtld_value,
            interpretation=interp,
            details={"forward": forward, "backward": backward},
        )

    def analyze_all(self, text: str) -> Dict[DiversityMetric, DiversityScore]:
        """Run all diversity analyses."""
        return {
            DiversityMetric.TYPE_TOKEN_RATIO: self.type_token_ratio(text),
            DiversityMetric.HAPAX_LEGOMENA: self.hapax_legomena_ratio(text),
            DiversityMetric.YULES_K: self.yules_k(text),
            DiversityMetric.SIMPSONS_D: self.simpsons_diversity(text),
            DiversityMetric.ENTROPY: self.entropy(text),
            DiversityMetric.MTLD: self.mtld(text),
        }


class RepetitionDetector:
    """Detects repetition patterns in text."""

    def __init__(self, min_phrase_length: int = 3, min_occurrences: int = 2):
        """Initialize detector."""
        self.min_phrase_length = min_phrase_length
        self.min_occurrences = min_occurrences

    def _get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Get n-grams from tokens."""
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def detect(self, text: str) -> RepetitionAnalysis:
        """Detect repetition in text."""
        words = re.findall(r'\b\w+\b', text.lower())

        if len(words) < self.min_phrase_length:
            return RepetitionAnalysis(
                repeated_phrases=[],
                repeated_words=[],
                repetition_score=0.0,
                longest_repeated_sequence="",
                n_gram_repetition={},
            )

        # Find repeated words
        word_counts = Counter(words)
        repeated_words = [
            (word, count) for word, count in word_counts.most_common()
            if count >= self.min_occurrences and len(word) > 2
        ]

        # Find repeated phrases (n-grams)
        repeated_phrases = []
        n_gram_repetition = {}

        for n in range(2, min(6, len(words))):
            ngrams = self._get_ngrams(words, n)
            ngram_counts = Counter(ngrams)
            repeated = [
                (phrase, count) for phrase, count in ngram_counts.items()
                if count >= self.min_occurrences
            ]
            repeated_phrases.extend(repeated)

            # Calculate repetition rate for this n
            if ngrams:
                repeated_count = sum(c - 1 for _, c in repeated)
                n_gram_repetition[n] = repeated_count / len(ngrams)

        # Sort by count
        repeated_phrases.sort(key=lambda x: x[1], reverse=True)

        # Find longest repeated sequence
        longest = ""
        for phrase, count in repeated_phrases:
            if len(phrase) > len(longest):
                longest = phrase

        # Calculate overall repetition score
        total_repetitions = sum(count - 1 for _, count in repeated_words[:20])
        repetition_score = min(1.0, total_repetitions / max(len(words), 1))

        return RepetitionAnalysis(
            repeated_phrases=repeated_phrases[:20],
            repeated_words=repeated_words[:20],
            repetition_score=repetition_score,
            longest_repeated_sequence=longest,
            n_gram_repetition=n_gram_repetition,
        )


class CreativityAnalyzer:
    """Analyzes creativity of text."""

    def __init__(
        self,
        reference_corpus: Optional[List[str]] = None,
        common_words: Optional[Set[str]] = None,
    ):
        """Initialize analyzer."""
        self.reference_corpus = reference_corpus or []
        self.common_words = common_words or self._default_common_words()

    def _default_common_words(self) -> Set[str]:
        """Default set of common words."""
        return {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "this", "that", "these", "those",
            "it", "its", "i", "you", "he", "she", "we", "they", "what",
        }

    def _novelty_score(self, text: str) -> float:
        """Calculate novelty based on rare word usage."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        uncommon = [w for w in words if w not in self.common_words and len(w) > 3]
        return len(uncommon) / len(words)

    def _unexpectedness_score(self, text: str) -> float:
        """Calculate unexpectedness based on unusual combinations."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 2:
            return 0.0

        # Check for unusual word pairs (simplified)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

        # Score based on bigram variety
        unique_bigrams = len(set(bigrams))
        return unique_bigrams / len(bigrams) if bigrams else 0.0

    def _elaboration_score(self, text: str) -> float:
        """Calculate elaboration based on detail level."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # Average sentence length as proxy for elaboration
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Normalize: 15-25 words is optimal
        if avg_length < 5:
            return 0.2
        elif avg_length < 10:
            return 0.4
        elif avg_length < 15:
            return 0.6
        elif avg_length < 25:
            return 0.8
        else:
            return 0.7  # Very long sentences reduce clarity

    def _flexibility_score(self, text: str) -> float:
        """Calculate flexibility based on variety of structures."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5

        # Check for variety in sentence starters
        starters = [s.split()[0].lower() if s.split() else "" for s in sentences]
        unique_starters = len(set(starters))

        return min(1.0, unique_starters / len(sentences))

    def _fluency_score(self, text: str) -> float:
        """Calculate fluency based on grammatical flow."""
        # Simplified: check for proper punctuation and capitalization
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        proper_count = 0
        for sent in sentences:
            if sent and sent[0].isupper():
                proper_count += 1

        return proper_count / len(sentences)

    def analyze(self, text: str) -> CreativityScore:
        """Analyze creativity of text."""
        dimension_scores = {
            CreativityDimension.NOVELTY: self._novelty_score(text),
            CreativityDimension.UNEXPECTEDNESS: self._unexpectedness_score(text),
            CreativityDimension.ELABORATION: self._elaboration_score(text),
            CreativityDimension.FLEXIBILITY: self._flexibility_score(text),
            CreativityDimension.FLUENCY: self._fluency_score(text),
        }

        overall = sum(dimension_scores.values()) / len(dimension_scores)

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        for dim, score in dimension_scores.items():
            if score >= 0.7:
                strengths.append(f"Strong {dim.value}")
            elif score <= 0.3:
                weaknesses.append(f"Low {dim.value}")

        # Generate interpretation
        if overall >= 0.7:
            interp = "Highly creative output with varied vocabulary and structure"
        elif overall >= 0.5:
            interp = "Moderately creative with room for improvement"
        else:
            interp = "Output shows limited creativity; consider more varied expression"

        return CreativityScore(
            overall_score=overall,
            dimension_scores=dimension_scores,
            interpretation=interp,
            strengths=strengths,
            weaknesses=weaknesses,
        )


class OutputVariabilityAnalyzer:
    """Analyzes variability across multiple model outputs."""

    def __init__(
        self,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """Initialize analyzer."""
        self.similarity_fn = similarity_fn or self._default_similarity

    def _default_similarity(self, a: str, b: str) -> float:
        """Default word overlap similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0

    def analyze(self, outputs: List[str]) -> VariabilityAnalysis:
        """Analyze variability across outputs."""
        n = len(outputs)

        if n < 2:
            return VariabilityAnalysis(
                n_samples=n,
                mean_similarity=1.0 if n == 1 else 0.0,
                std_similarity=0.0,
                unique_tokens_ratio=1.0,
                semantic_spread=0.0,
                clustering_coefficient=1.0,
                outlier_indices=[],
            )

        # Calculate pairwise similarities
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.similarity_fn(outputs[i], outputs[j])
                similarities.append(sim)

        mean_sim = sum(similarities) / len(similarities)
        variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
        std_sim = variance ** 0.5

        # Calculate unique tokens ratio
        all_tokens: List[str] = []
        for output in outputs:
            all_tokens.extend(output.lower().split())
        unique_ratio = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0

        # Semantic spread (inverse of mean similarity)
        semantic_spread = 1 - mean_sim

        # Clustering coefficient (how much they cluster together)
        # Low variance = high clustering
        clustering = 1 - min(1.0, std_sim * 2)

        # Find outliers (responses very different from others)
        outliers = []
        for i in range(n):
            avg_sim_to_others = sum(
                self.similarity_fn(outputs[i], outputs[j])
                for j in range(n) if j != i
            ) / (n - 1)

            if avg_sim_to_others < mean_sim - 2 * std_sim:
                outliers.append(i)

        return VariabilityAnalysis(
            n_samples=n,
            mean_similarity=mean_sim,
            std_similarity=std_sim,
            unique_tokens_ratio=unique_ratio,
            semantic_spread=semantic_spread,
            clustering_coefficient=clustering,
            outlier_indices=outliers,
        )


class DiversityReporter:
    """Generates comprehensive diversity reports."""

    def __init__(
        self,
        lexical_analyzer: Optional[LexicalDiversityAnalyzer] = None,
        repetition_detector: Optional[RepetitionDetector] = None,
        creativity_analyzer: Optional[CreativityAnalyzer] = None,
    ):
        """Initialize reporter."""
        self.lexical = lexical_analyzer or LexicalDiversityAnalyzer()
        self.repetition = repetition_detector or RepetitionDetector()
        self.creativity = creativity_analyzer or CreativityAnalyzer()

    def generate_report(
        self,
        text: str,
        include_creativity: bool = True,
    ) -> DiversityReport:
        """Generate comprehensive diversity report."""
        # Analyze lexical diversity
        lexical_scores = self.lexical.analyze_all(text)

        # Detect repetition
        repetition = self.repetition.detect(text)

        # Analyze creativity
        creativity = self.creativity.analyze(text) if include_creativity else None

        # Calculate overall diversity score
        # Combine key metrics
        ttr = lexical_scores[DiversityMetric.TYPE_TOKEN_RATIO].value
        entropy = lexical_scores[DiversityMetric.ENTROPY].value
        simpsons = lexical_scores[DiversityMetric.SIMPSONS_D].value
        rep_penalty = 1 - repetition.repetition_score

        overall = (ttr + entropy + simpsons + rep_penalty) / 4

        # Generate recommendations
        recommendations = self._generate_recommendations(
            lexical_scores, repetition, creativity, overall
        )

        return DiversityReport(
            text=text,
            lexical_diversity=lexical_scores,
            repetition=repetition,
            creativity=creativity,
            overall_diversity_score=overall,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        lexical: Dict[DiversityMetric, DiversityScore],
        repetition: RepetitionAnalysis,
        creativity: Optional[CreativityScore],
        overall: float,
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if lexical[DiversityMetric.TYPE_TOKEN_RATIO].value < 0.5:
            recommendations.append("Consider using more varied vocabulary")

        if repetition.has_significant_repetition:
            recommendations.append("Reduce repetitive phrases and words")
            if repetition.longest_repeated_sequence:
                recommendations.append(
                    f"Consider varying: '{repetition.longest_repeated_sequence[:30]}...'"
                )

        if creativity and creativity.overall_score < 0.5:
            for weakness in creativity.weaknesses[:2]:
                recommendations.append(f"Improve {weakness.lower()}")

        if overall >= 0.8:
            recommendations.append("Output shows excellent diversity!")
        elif overall < 0.4:
            recommendations.append("Overall diversity needs significant improvement")

        return recommendations


# Convenience functions
def analyze_diversity(text: str) -> DiversityReport:
    """Analyze diversity of text."""
    reporter = DiversityReporter()
    return reporter.generate_report(text)


def calculate_type_token_ratio(text: str) -> float:
    """Calculate Type-Token Ratio."""
    analyzer = LexicalDiversityAnalyzer()
    return analyzer.type_token_ratio(text).value


def detect_repetition(text: str) -> RepetitionAnalysis:
    """Detect repetition in text."""
    detector = RepetitionDetector()
    return detector.detect(text)


def analyze_creativity(text: str) -> CreativityScore:
    """Analyze creativity of text."""
    analyzer = CreativityAnalyzer()
    return analyzer.analyze(text)


def analyze_output_variability(outputs: List[str]) -> VariabilityAnalysis:
    """Analyze variability across multiple outputs."""
    analyzer = OutputVariabilityAnalyzer()
    return analyzer.analyze(outputs)


def quick_diversity_check(text: str) -> Dict[str, Any]:
    """Quick diversity check returning summary."""
    report = analyze_diversity(text)
    return {
        "overall_score": report.overall_diversity_score,
        "type_token_ratio": report.lexical_diversity[DiversityMetric.TYPE_TOKEN_RATIO].value,
        "has_repetition": report.repetition.has_significant_repetition,
        "creativity_level": report.creativity.creativity_level if report.creativity else None,
        "recommendations": report.recommendations[:3],
    }
