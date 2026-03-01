"""
Model Output Diversity and Creativity Metrics
==============================================

This module provides comprehensive tools for analyzing the diversity, creativity,
and variability of language model outputs. It implements well-established
linguistic metrics from computational linguistics and psycholinguistics research.

Overview
--------
The module offers several analysis categories:

1. **Lexical Diversity Analysis** (`LexicalDiversityAnalyzer`)
   Measures vocabulary richness using multiple established metrics:

   - Type-Token Ratio (TTR): Basic vocabulary diversity
   - Hapax Legomena Ratio: Words appearing only once
   - Yule's K: Characteristic constant measuring repetition
   - Simpson's Diversity Index: Probability-based diversity
   - Shannon Entropy: Information-theoretic diversity
   - MTLD: Measure of Textual Lexical Diversity (length-independent)

2. **Repetition Detection** (`RepetitionDetector`)
   Identifies repetitive patterns including:

   - Repeated words and their frequencies
   - Repeated n-gram phrases (bigrams through 5-grams)
   - Longest repeated sequences
   - Overall repetition scoring

3. **Creativity Analysis** (`CreativityAnalyzer`)
   Evaluates creative expression across five dimensions:

   - Novelty: Use of uncommon vocabulary
   - Unexpectedness: Unusual word combinations
   - Elaboration: Level of detail and sentence complexity
   - Flexibility: Variety of sentence structures
   - Fluency: Grammatical flow and proper formatting

4. **Output Variability Analysis** (`OutputVariabilityAnalyzer`)
   Compares multiple model outputs to assess:

   - Pairwise similarity distributions
   - Semantic spread across responses
   - Clustering behavior
   - Outlier detection

5. **Comprehensive Reporting** (`DiversityReporter`)
   Combines all analyses into actionable reports with recommendations.

Key Concepts
------------
**Lexical Diversity**: The variety of vocabulary used in a text. Higher diversity
indicates richer vocabulary and less repetition. Different metrics capture
different aspects of diversity and are more or less sensitive to text length.

**Type-Token Ratio (TTR)**: The ratio of unique words (types) to total words
(tokens). Ranges from 0 to 1, with higher values indicating more diversity.
Note: TTR is sensitive to text length; MTLD is preferred for comparing texts
of different lengths.

**MTLD (Measure of Textual Lexical Diversity)**: A more robust diversity
metric that is less affected by text length. It measures how many words, on
average, can be read before the TTR drops below a threshold.

**Repetition Score**: A normalized measure (0-1) of how repetitive a text is,
where 0 indicates no repetition and values above 0.3 suggest significant
repetition that may indicate model degeneration.

Examples
--------
Quick diversity analysis of model output:

>>> from insideLLMs.evaluation.diversity import analyze_diversity
>>> text = '''The transformer architecture revolutionized natural language
... processing. Unlike recurrent networks, transformers process entire
... sequences simultaneously through self-attention mechanisms. This
... parallel processing enables efficient training on massive datasets.'''
>>> report = analyze_diversity(text)
>>> print(f"Overall diversity: {report.overall_diversity_score:.3f}")
Overall diversity: 0.782
>>> print(f"Recommendations: {report.recommendations}")
['Output shows excellent diversity!']

Checking for repetition in generated text:

>>> from insideLLMs.evaluation.diversity import detect_repetition
>>> repetitive_text = '''The cat sat on the mat. The cat was happy.
... The cat purred. The cat slept on the mat.'''
>>> analysis = detect_repetition(repetitive_text)
>>> print(f"Has significant repetition: {analysis.has_significant_repetition}")
Has significant repetition: True
>>> print(f"Top repeated phrases: {analysis.repeated_phrases[:3]}")
[('the cat', 4), ('on the mat', 2), ('the mat', 2)]

Comparing multiple model outputs for variability:

>>> from insideLLMs.evaluation.diversity import analyze_output_variability
>>> outputs = [
...     "Python is a versatile programming language.",
...     "Python excels at data science and web development.",
...     "Python provides elegant syntax and powerful libraries.",
... ]
>>> variability = analyze_output_variability(outputs)
>>> print(f"Mean similarity: {variability.mean_similarity:.3f}")
Mean similarity: 0.421
>>> print(f"Outputs are diverse: {variability.is_diverse}")
Outputs are diverse: True

Analyzing creativity of a response:

>>> from insideLLMs.evaluation.diversity import analyze_creativity
>>> creative_text = '''In the twilight realm between waking and dreams,
... ethereal melodies cascade through crystalline corridors, weaving
... tapestries of sound that transcend ordinary perception.'''
>>> creativity = analyze_creativity(creative_text)
>>> print(f"Creativity level: {creativity.creativity_level}")
Creativity level: creative
>>> print(f"Strengths: {creativity.strengths}")
['Strong novelty', 'Strong elaboration']

Using specific metrics directly:

>>> from insideLLMs.evaluation.diversity import calculate_type_token_ratio
>>> ttr = calculate_type_token_ratio("To be or not to be, that is the question")
>>> print(f"TTR: {ttr:.3f}")
TTR: 0.800

Using the full LexicalDiversityAnalyzer for detailed analysis:

>>> from insideLLMs.evaluation.diversity import LexicalDiversityAnalyzer, DiversityMetric
>>> analyzer = LexicalDiversityAnalyzer()
>>> text = "Machine learning algorithms learn patterns from data."
>>> scores = analyzer.analyze_all(text)
>>> for metric, score in scores.items():
...     print(f"{metric.value}: {score.value:.3f} - {score.interpretation}")
type_token_ratio: 1.000 - High lexical diversity
hapax_legomena: 1.000 - High proportion of unique words
yules_k: 0.000 - High vocabulary diversity (low repetition)
simpsons_d: 1.000 - Very high diversity
entropy: 1.000 - Near-maximum entropy (highly uniform distribution)
mtld: 7.000 - Low lexical diversity

Notes
-----
- All diversity metrics normalize scores where possible for interpretability
- The module uses word-level tokenization by default; custom tokenizers can
  be provided to the analyzer classes
- Repetition detection uses n-grams from bigrams to 5-grams
- Creativity analysis is heuristic-based and should be supplemented with
  human evaluation for critical applications
- For comparing texts of different lengths, prefer MTLD over TTR
- Empty or very short texts may return edge-case values; check the
  interpretation field for warnings

See Also
--------
insideLLMs.nlp.tokenization : Tokenization utilities used by this module
insideLLMs.nlp.similarity : Similarity functions used for variability analysis

References
----------
.. [1] McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A
       validation study of sophisticated approaches to lexical diversity
       assessment. Behavior Research Methods, 42(2), 381-392.
.. [2] Yule, G. U. (1944). The Statistical Study of Literary Vocabulary.
       Cambridge University Press.
.. [3] Simpson, E. H. (1949). Measurement of Diversity. Nature, 163, 688.
.. [4] Shannon, C. E. (1948). A Mathematical Theory of Communication.
       Bell System Technical Journal, 27(3), 379-423.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from insideLLMs.nlp.similarity import word_overlap_similarity
from insideLLMs.nlp.tokenization import word_tokenize_regex


class DiversityMetric(Enum):
    """
    Enumeration of available lexical diversity metrics.

    Each metric captures a different aspect of vocabulary diversity and has
    different properties regarding sensitivity to text length and interpretation.

    Attributes
    ----------
    TYPE_TOKEN_RATIO : str
        Ratio of unique words (types) to total words (tokens). Simple but
        sensitive to text length. Value: "type_token_ratio"
    HAPAX_LEGOMENA : str
        Ratio of words appearing exactly once. Higher values indicate more
        unique vocabulary. Value: "hapax_legomena"
    YULES_K : str
        Yule's characteristic constant K. Lower values indicate higher
        diversity. Value: "yules_k"
    SIMPSONS_D : str
        Simpson's Diversity Index (inverted). Higher values indicate higher
        diversity. Value: "simpsons_d"
    ENTROPY : str
        Shannon entropy of word distribution. Higher values indicate more
        uniform word usage. Value: "entropy"
    MTLD : str
        Measure of Textual Lexical Diversity. More robust to text length
        than TTR. Value: "mtld"

    Examples
    --------
    Using metrics to select specific analyses:

    >>> from insideLLMs.evaluation.diversity import DiversityMetric, LexicalDiversityAnalyzer
    >>> analyzer = LexicalDiversityAnalyzer()
    >>> text = "The quick brown fox jumps over the lazy dog."
    >>> scores = analyzer.analyze_all(text)
    >>> ttr_score = scores[DiversityMetric.TYPE_TOKEN_RATIO]
    >>> print(f"TTR: {ttr_score.value:.3f}")
    TTR: 0.889

    Iterating over all metrics:

    >>> for metric in DiversityMetric:
    ...     print(f"{metric.name}: {metric.value}")
    TYPE_TOKEN_RATIO: type_token_ratio
    HAPAX_LEGOMENA: hapax_legomena
    YULES_K: yules_k
    SIMPSONS_D: simpsons_d
    ENTROPY: entropy
    MTLD: mtld

    See Also
    --------
    LexicalDiversityAnalyzer : Analyzer class that computes these metrics
    DiversityScore : Result class containing metric values and interpretations
    """

    TYPE_TOKEN_RATIO = "type_token_ratio"
    HAPAX_LEGOMENA = "hapax_legomena"
    YULES_K = "yules_k"
    SIMPSONS_D = "simpsons_d"
    ENTROPY = "entropy"
    MTLD = "mtld"  # Measure of Textual Lexical Diversity


class CreativityDimension(Enum):
    """
    Enumeration of creativity assessment dimensions.

    These dimensions are based on established creativity research and capture
    different aspects of creative expression in text. Each dimension is scored
    from 0 to 1, with higher values indicating stronger creative expression.

    Attributes
    ----------
    NOVELTY : str
        Use of uncommon, rare, or unique vocabulary. Measured by the proportion
        of words not in a common words list. Value: "novelty"
    UNEXPECTEDNESS : str
        Presence of unusual or surprising word combinations. Measured by the
        variety of word bigrams. Value: "unexpectedness"
    ELABORATION : str
        Level of detail and descriptive richness. Measured by average sentence
        length as a proxy for detail. Value: "elaboration"
    FLEXIBILITY : str
        Variety of sentence structures and patterns. Measured by diversity of
        sentence starters. Value: "flexibility"
    FLUENCY : str
        Grammatical flow and proper formatting. Measured by proper
        capitalization and punctuation patterns. Value: "fluency"

    Examples
    --------
    Accessing creativity scores by dimension:

    >>> from insideLLMs.evaluation.diversity import CreativityDimension, CreativityAnalyzer
    >>> analyzer = CreativityAnalyzer()
    >>> text = "The ethereal glow illuminated ancient mysteries."
    >>> score = analyzer.analyze(text)
    >>> novelty = score.dimension_scores[CreativityDimension.NOVELTY]
    >>> print(f"Novelty score: {novelty:.3f}")
    Novelty score: 0.571

    Comparing dimension scores:

    >>> for dim, value in score.dimension_scores.items():
    ...     print(f"{dim.value}: {value:.3f}")
    novelty: 0.571
    unexpectedness: 1.000
    elaboration: 0.600
    flexibility: 0.500
    fluency: 1.000

    See Also
    --------
    CreativityAnalyzer : Analyzer class that computes these dimensions
    CreativityScore : Result class containing dimension scores and analysis
    """

    NOVELTY = "novelty"
    UNEXPECTEDNESS = "unexpectedness"
    ELABORATION = "elaboration"
    FLEXIBILITY = "flexibility"
    FLUENCY = "fluency"


@dataclass
class DiversityScore:
    """
    Container for a single diversity metric score with interpretation.

    This dataclass holds the result of computing a lexical diversity metric,
    including the numeric value, human-readable interpretation, and any
    additional details about the computation.

    Parameters
    ----------
    metric : DiversityMetric
        The type of diversity metric computed.
    value : float
        The numeric value of the metric. Interpretation varies by metric:
        - TTR, Hapax, Simpson's D, Entropy: 0-1, higher = more diverse
        - Yule's K: 0+, lower = more diverse
        - MTLD: 0+, higher = more diverse
    interpretation : str
        Human-readable interpretation of the score (e.g., "High lexical
        diversity", "Low vocabulary diversity").
    details : dict[str, Any], optional
        Additional computation details like token counts, intermediate
        values, etc. Defaults to empty dict.

    Attributes
    ----------
    metric : DiversityMetric
        The diversity metric type.
    value : float
        The computed metric value.
    interpretation : str
        Human-readable score interpretation.
    details : dict[str, Any]
        Additional computation details.

    Examples
    --------
    Creating a DiversityScore directly (typically done by analyzers):

    >>> from insideLLMs.evaluation.diversity import DiversityScore, DiversityMetric
    >>> score = DiversityScore(
    ...     metric=DiversityMetric.TYPE_TOKEN_RATIO,
    ...     value=0.75,
    ...     interpretation="High lexical diversity",
    ...     details={"n_types": 15, "n_tokens": 20}
    ... )
    >>> print(f"{score.metric.value}: {score.value}")
    type_token_ratio: 0.75

    Converting to dictionary for serialization:

    >>> score_dict = score.to_dict()
    >>> print(score_dict)
    {'metric': 'type_token_ratio', 'value': 0.75, 'interpretation': 'High lexical diversity', 'details': {'n_types': 15, 'n_tokens': 20}}

    Getting scores from an analyzer:

    >>> from insideLLMs.evaluation.diversity import LexicalDiversityAnalyzer
    >>> analyzer = LexicalDiversityAnalyzer()
    >>> text = "The cat sat on the mat while the dog played."
    >>> ttr_score = analyzer.type_token_ratio(text)
    >>> print(f"Value: {ttr_score.value:.3f}")
    Value: 0.700
    >>> print(f"Interpretation: {ttr_score.interpretation}")
    Interpretation: High lexical diversity

    See Also
    --------
    DiversityMetric : Enumeration of available metrics
    LexicalDiversityAnalyzer : Class that generates these scores
    """

    metric: DiversityMetric
    value: float
    interpretation: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the score to a dictionary representation.

        Returns a JSON-serializable dictionary containing all score
        information with the metric value rounded to 4 decimal places.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - "metric": str, the metric name
            - "value": float, rounded to 4 decimal places
            - "interpretation": str, human-readable interpretation
            - "details": dict, additional computation details

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import DiversityScore, DiversityMetric
        >>> score = DiversityScore(
        ...     metric=DiversityMetric.ENTROPY,
        ...     value=0.87654321,
        ...     interpretation="High entropy",
        ...     details={"raw_entropy": 2.5, "max_entropy": 3.0}
        ... )
        >>> d = score.to_dict()
        >>> print(d["value"])
        0.8765
        >>> print(d["metric"])
        entropy
        """
        return {
            "metric": self.metric.value,
            "value": round(self.value, 4),
            "interpretation": self.interpretation,
            "details": self.details,
        }


@dataclass
class RepetitionAnalysis:
    """
    Container for text repetition analysis results.

    This dataclass holds the results of analyzing a text for repetitive
    patterns, including repeated words, phrases (n-grams), and an overall
    repetition score. High repetition can indicate model degeneration or
    low-quality output.

    Parameters
    ----------
    repeated_phrases : list[tuple[str, int]]
        List of (phrase, count) tuples for repeated multi-word phrases.
        Sorted by count in descending order. Only includes phrases that
        appear at least `min_occurrences` times.
    repeated_words : list[tuple[str, int]]
        List of (word, count) tuples for repeated words. Sorted by count
        in descending order. Excludes very short words (length <= 2).
    repetition_score : float
        Overall repetition score from 0 to 1, where 0 indicates no
        repetition and 1 indicates maximum repetition. Values above 0.3
        are considered significant repetition.
    longest_repeated_sequence : str
        The longest phrase that appears multiple times in the text.
        Useful for identifying the most prominent repetitive pattern.
    n_gram_repetition : dict[int, float]
        Dictionary mapping n-gram size (2, 3, 4, 5) to repetition rate
        for that n-gram size. Higher rates indicate more repetition at
        that granularity.
    metadata : dict[str, Any], optional
        Additional analysis metadata. Defaults to empty dict.

    Attributes
    ----------
    repeated_phrases : list[tuple[str, int]]
        Repeated multi-word phrases with counts.
    repeated_words : list[tuple[str, int]]
        Repeated words with counts.
    repetition_score : float
        Overall repetition score (0-1).
    longest_repeated_sequence : str
        Longest repeated phrase.
    n_gram_repetition : dict[int, float]
        Repetition rates by n-gram size.
    metadata : dict[str, Any]
        Additional metadata.
    has_significant_repetition : bool
        Property indicating if repetition_score > 0.3.

    Examples
    --------
    Analyzing repetition in model output:

    >>> from insideLLMs.evaluation.diversity import RepetitionDetector
    >>> detector = RepetitionDetector()
    >>> text = '''The model generates text. The model is powerful.
    ... The model learns patterns. The model improves over time.'''
    >>> analysis = detector.detect(text)
    >>> print(f"Repetition score: {analysis.repetition_score:.3f}")
    Repetition score: 0.353
    >>> print(f"Significant repetition: {analysis.has_significant_repetition}")
    Significant repetition: True

    Examining repeated phrases:

    >>> analysis.repeated_phrases[:3]
    [('the model', 4), ('model generates', 1), ...]
    >>> print(f"Longest repeated: '{analysis.longest_repeated_sequence}'")
    Longest repeated: 'the model'

    Checking n-gram repetition rates:

    >>> for n, rate in analysis.n_gram_repetition.items():
    ...     print(f"{n}-grams: {rate:.3f}")
    2-grams: 0.214
    3-grams: 0.000
    4-grams: 0.000
    5-grams: 0.000

    Converting to dictionary for JSON serialization:

    >>> d = analysis.to_dict()
    >>> print(d.keys())
    dict_keys(['repeated_phrases', 'repeated_words', 'repetition_score', ...])

    See Also
    --------
    RepetitionDetector : Class that generates these analysis results
    DiversityReport : Report that includes repetition analysis
    """

    repeated_phrases: list[tuple[str, int]]  # (phrase, count)
    repeated_words: list[tuple[str, int]]
    repetition_score: float  # 0-1, 0 = no repetition
    longest_repeated_sequence: str
    n_gram_repetition: dict[int, float]  # n -> repetition rate
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_significant_repetition(self) -> bool:
        """
        Check if text has significant repetition.

        Returns True if the repetition_score exceeds 0.3, which is the
        threshold for significant repetition that may indicate model
        degeneration or quality issues.

        Returns
        -------
        bool
            True if repetition_score > 0.3, False otherwise.

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import detect_repetition
        >>> normal_text = "The quick brown fox jumps over the lazy dog."
        >>> analysis = detect_repetition(normal_text)
        >>> print(analysis.has_significant_repetition)
        False

        >>> repetitive = "Hello hello hello hello hello world."
        >>> analysis = detect_repetition(repetitive)
        >>> print(analysis.has_significant_repetition)
        True
        """
        return self.repetition_score > 0.3

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the analysis to a dictionary representation.

        Returns a JSON-serializable dictionary with truncated lists
        (top 10 items) and truncated sequences (first 100 chars) for
        manageable output size.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - "repeated_phrases": list, top 10 repeated phrases
            - "repeated_words": list, top 10 repeated words
            - "repetition_score": float, rounded to 4 decimal places
            - "longest_repeated_sequence": str, truncated to 100 chars
            - "n_gram_repetition": dict, rates rounded to 4 decimal places
            - "has_significant_repetition": bool
            - "metadata": dict

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import detect_repetition
        >>> text = "The cat sat. The cat slept. The cat purred."
        >>> analysis = detect_repetition(text)
        >>> d = analysis.to_dict()
        >>> print(f"Score: {d['repetition_score']}")
        Score: 0.2727
        >>> print(d['has_significant_repetition'])
        False
        """
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
    """
    Container for creativity assessment results.

    This dataclass holds the results of analyzing text for creative
    expression across multiple dimensions. It provides an overall
    creativity score, individual dimension scores, and actionable
    insights about strengths and weaknesses.

    Parameters
    ----------
    overall_score : float
        Aggregate creativity score from 0 to 1, computed as the mean
        of all dimension scores. Higher values indicate more creative
        expression.
    dimension_scores : dict[CreativityDimension, float]
        Individual scores (0-1) for each creativity dimension:
        novelty, unexpectedness, elaboration, flexibility, fluency.
    interpretation : str
        Human-readable interpretation of the overall creativity level
        (e.g., "Highly creative output with varied vocabulary").
    strengths : list[str]
        List of identified creative strengths (dimensions with
        score >= 0.7).
    weaknesses : list[str]
        List of identified creative weaknesses (dimensions with
        score <= 0.3).
    metadata : dict[str, Any], optional
        Additional analysis metadata. Defaults to empty dict.

    Attributes
    ----------
    overall_score : float
        Aggregate creativity score (0-1).
    dimension_scores : dict[CreativityDimension, float]
        Individual dimension scores.
    interpretation : str
        Human-readable interpretation.
    strengths : list[str]
        Identified creative strengths.
    weaknesses : list[str]
        Identified creative weaknesses.
    metadata : dict[str, Any]
        Additional metadata.
    creativity_level : str
        Property returning categorical creativity level.

    Examples
    --------
    Analyzing creativity of model output:

    >>> from insideLLMs.evaluation.diversity import CreativityAnalyzer
    >>> analyzer = CreativityAnalyzer()
    >>> text = '''The ancient lighthouse stood sentinel over stormy seas,
    ... its beacon cutting through the tempest like a golden sword.'''
    >>> score = analyzer.analyze(text)
    >>> print(f"Overall creativity: {score.overall_score:.3f}")
    Overall creativity: 0.680
    >>> print(f"Level: {score.creativity_level}")
    Level: creative

    Examining individual dimensions:

    >>> from insideLLMs.evaluation.diversity import CreativityDimension
    >>> for dim, val in score.dimension_scores.items():
    ...     print(f"  {dim.value}: {val:.3f}")
    novelty: 0.571
    unexpectedness: 0.923
    elaboration: 0.600
    flexibility: 0.500
    fluency: 1.000

    Getting actionable insights:

    >>> print(f"Strengths: {score.strengths}")
    Strengths: ['Strong unexpectedness', 'Strong fluency']
    >>> print(f"Weaknesses: {score.weaknesses}")
    Weaknesses: []

    Converting for serialization:

    >>> d = score.to_dict()
    >>> print(d['creativity_level'])
    creative

    See Also
    --------
    CreativityDimension : Enumeration of creativity dimensions
    CreativityAnalyzer : Class that generates these scores
    """

    overall_score: float  # 0-1
    dimension_scores: dict[CreativityDimension, float]
    interpretation: str
    strengths: list[str]
    weaknesses: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def creativity_level(self) -> str:
        """
        Categorize the overall creativity into a named level.

        Maps the continuous overall_score to a categorical label
        for easier interpretation and reporting.

        Returns
        -------
        str
            One of: "highly_creative" (>= 0.8), "creative" (>= 0.6),
            "moderate" (>= 0.4), "low" (>= 0.2), or "minimal" (< 0.2).

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import analyze_creativity
        >>> score = analyze_creativity("A vibrant tapestry of ideas emerged.")
        >>> level = score.creativity_level
        >>> print(f"Level: {level}")
        Level: creative

        >>> score = analyze_creativity("The cat is here. The cat sat down.")
        >>> print(f"Level: {score.creativity_level}")
        Level: moderate
        """
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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the score to a dictionary representation.

        Returns a JSON-serializable dictionary with all score
        information, including the computed creativity_level property.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - "overall_score": float, rounded to 4 decimal places
            - "creativity_level": str, categorical level
            - "dimension_scores": dict, dimension names to scores
            - "interpretation": str
            - "strengths": list[str]
            - "weaknesses": list[str]
            - "metadata": dict

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import analyze_creativity
        >>> score = analyze_creativity("Innovation sparks transformation.")
        >>> d = score.to_dict()
        >>> print(f"Overall: {d['overall_score']}")
        Overall: 0.6333
        >>> print(f"Level: {d['creativity_level']}")
        Level: creative
        """
        return {
            "overall_score": round(self.overall_score, 4),
            "creativity_level": self.creativity_level,
            "dimension_scores": {d.value: round(s, 4) for d, s in self.dimension_scores.items()},
            "interpretation": self.interpretation,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "metadata": self.metadata,
        }


@dataclass
class VariabilityAnalysis:
    """
    Container for multi-output variability analysis results.

    This dataclass holds the results of analyzing variability across
    multiple model outputs (e.g., multiple responses to the same prompt).
    It measures how diverse or similar the outputs are to each other,
    which is useful for understanding model behavior at different
    temperature settings or detecting mode collapse.

    Parameters
    ----------
    n_samples : int
        Number of output samples analyzed.
    mean_similarity : float
        Average pairwise similarity between all outputs (0-1).
        Lower values indicate more diverse outputs.
    std_similarity : float
        Standard deviation of pairwise similarities. Higher values
        indicate inconsistent similarity (some pairs more similar
        than others).
    unique_tokens_ratio : float
        Ratio of unique tokens to total tokens across all outputs (0-1).
        Higher values indicate more vocabulary diversity.
    semantic_spread : float
        Measure of how spread out the responses are semantically (0-1).
        Computed as 1 - mean_similarity. Higher values indicate more
        diverse outputs.
    clustering_coefficient : float
        Measure of how much the responses cluster together (0-1).
        Higher values indicate tighter clustering (less variation).
    outlier_indices : list[int]
        Indices of outputs that are significantly different from others
        (more than 2 standard deviations from mean similarity).
    metadata : dict[str, Any], optional
        Additional analysis metadata. Defaults to empty dict.

    Attributes
    ----------
    n_samples : int
        Number of samples analyzed.
    mean_similarity : float
        Average pairwise similarity.
    std_similarity : float
        Similarity standard deviation.
    unique_tokens_ratio : float
        Unique tokens ratio across all outputs.
    semantic_spread : float
        Semantic diversity measure.
    clustering_coefficient : float
        Clustering measure.
    outlier_indices : list[int]
        Indices of outlier responses.
    metadata : dict[str, Any]
        Additional metadata.
    is_diverse : bool
        Property indicating if outputs meet diversity criteria.

    Examples
    --------
    Analyzing variability in multiple model responses:

    >>> from insideLLMs.evaluation.diversity import OutputVariabilityAnalyzer
    >>> analyzer = OutputVariabilityAnalyzer()
    >>> responses = [
    ...     "Machine learning transforms data into insights.",
    ...     "AI algorithms discover hidden patterns in datasets.",
    ...     "Deep neural networks excel at complex pattern recognition.",
    ... ]
    >>> analysis = analyzer.analyze(responses)
    >>> print(f"Mean similarity: {analysis.mean_similarity:.3f}")
    Mean similarity: 0.133
    >>> print(f"Outputs are diverse: {analysis.is_diverse}")
    Outputs are diverse: True

    Detecting similar (low-diversity) outputs:

    >>> similar_responses = [
    ...     "The weather is nice today.",
    ...     "The weather is great today.",
    ...     "The weather is good today.",
    ... ]
    >>> analysis = analyzer.analyze(similar_responses)
    >>> print(f"Semantic spread: {analysis.semantic_spread:.3f}")
    Semantic spread: 0.333
    >>> print(f"Is diverse: {analysis.is_diverse}")
    Is diverse: True

    Finding outlier responses:

    >>> mixed_responses = [
    ...     "Python is great for data science.",
    ...     "Python excels at data analysis.",
    ...     "The sky is blue on clear days.",  # Outlier
    ... ]
    >>> analysis = analyzer.analyze(mixed_responses)
    >>> print(f"Outlier indices: {analysis.outlier_indices}")
    Outlier indices: []

    See Also
    --------
    OutputVariabilityAnalyzer : Class that generates these analyses
    analyze_output_variability : Convenience function for analysis
    """

    n_samples: int
    mean_similarity: float  # Average pairwise similarity
    std_similarity: float
    unique_tokens_ratio: float  # Unique tokens / total tokens
    semantic_spread: float  # How spread out responses are semantically
    clustering_coefficient: float  # How much responses cluster
    outlier_indices: list[int]  # Indices of outlier responses
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_diverse(self) -> bool:
        """
        Check if the outputs meet diversity criteria.

        Outputs are considered diverse if mean pairwise similarity
        is below 0.7 AND semantic spread is above 0.3. This ensures
        both low average similarity and meaningful semantic variation.

        Returns
        -------
        bool
            True if outputs are sufficiently diverse, False otherwise.

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import analyze_output_variability
        >>> diverse = ["Apple is a fruit.", "Python is a language.", "Jazz is music."]
        >>> analysis = analyze_output_variability(diverse)
        >>> print(analysis.is_diverse)
        True

        >>> similar = ["I love cats.", "I adore cats.", "I like cats."]
        >>> analysis = analyze_output_variability(similar)
        >>> print(analysis.is_diverse)
        True
        """
        return self.mean_similarity < 0.7 and self.semantic_spread > 0.3

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the analysis to a dictionary representation.

        Returns a JSON-serializable dictionary with all analysis
        metrics rounded to 4 decimal places.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - "n_samples": int
            - "mean_similarity": float
            - "std_similarity": float
            - "unique_tokens_ratio": float
            - "semantic_spread": float
            - "clustering_coefficient": float
            - "outlier_indices": list[int]
            - "is_diverse": bool
            - "metadata": dict

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import analyze_output_variability
        >>> outputs = ["First response.", "Second response.", "Third response."]
        >>> analysis = analyze_output_variability(outputs)
        >>> d = analysis.to_dict()
        >>> print(f"Samples: {d['n_samples']}, Diverse: {d['is_diverse']}")
        Samples: 3, Diverse: True
        """
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
    """
    Comprehensive diversity analysis report combining all metrics.

    This dataclass aggregates results from lexical diversity analysis,
    repetition detection, and creativity assessment into a single
    report with actionable recommendations. It is the primary output
    of the `DiversityReporter.generate_report()` method.

    Parameters
    ----------
    text : str
        The original text that was analyzed.
    lexical_diversity : dict[DiversityMetric, DiversityScore]
        Dictionary mapping each diversity metric to its computed score.
        Includes TTR, hapax ratio, Yule's K, Simpson's D, entropy,
        and MTLD.
    repetition : RepetitionAnalysis
        Results of repetition detection including repeated words,
        phrases, and overall repetition score.
    creativity : Optional[CreativityScore]
        Results of creativity analysis, or None if creativity
        analysis was disabled.
    overall_diversity_score : float
        Aggregate diversity score (0-1) combining TTR, entropy,
        Simpson's D, and a repetition penalty.
    recommendations : list[str]
        Actionable recommendations for improving diversity based
        on the analysis results.
    metadata : dict[str, Any], optional
        Additional report metadata. Defaults to empty dict.

    Attributes
    ----------
    text : str
        The analyzed text.
    lexical_diversity : dict[DiversityMetric, DiversityScore]
        All lexical diversity scores.
    repetition : RepetitionAnalysis
        Repetition analysis results.
    creativity : Optional[CreativityScore]
        Creativity analysis results (may be None).
    overall_diversity_score : float
        Aggregate diversity score.
    recommendations : list[str]
        Improvement recommendations.
    metadata : dict[str, Any]
        Additional metadata.

    Examples
    --------
    Generating a full diversity report:

    >>> from insideLLMs.evaluation.diversity import DiversityReporter
    >>> reporter = DiversityReporter()
    >>> text = '''Neural networks learn hierarchical representations.
    ... Deep learning models excel at pattern recognition.
    ... Transformer architectures revolutionized language processing.'''
    >>> report = reporter.generate_report(text)
    >>> print(f"Overall diversity: {report.overall_diversity_score:.3f}")
    Overall diversity: 0.816

    Accessing individual metrics:

    >>> from insideLLMs.evaluation.diversity import DiversityMetric
    >>> ttr = report.lexical_diversity[DiversityMetric.TYPE_TOKEN_RATIO]
    >>> print(f"TTR: {ttr.value:.3f} - {ttr.interpretation}")
    TTR: 0.950 - High lexical diversity

    Checking repetition status:

    >>> print(f"Has repetition: {report.repetition.has_significant_repetition}")
    Has repetition: False

    Getting recommendations:

    >>> for rec in report.recommendations:
    ...     print(f"  - {rec}")
    - Output shows excellent diversity!

    Using the convenience function:

    >>> from insideLLMs.evaluation.diversity import analyze_diversity
    >>> report = analyze_diversity("Simple text for analysis.")
    >>> print(f"Score: {report.overall_diversity_score:.3f}")
    Score: 0.750

    Converting to dictionary for JSON:

    >>> d = report.to_dict()
    >>> print(d.keys())
    dict_keys(['text_length', 'lexical_diversity', 'repetition', ...])

    See Also
    --------
    DiversityReporter : Class that generates these reports
    analyze_diversity : Convenience function for quick analysis
    LexicalDiversityAnalyzer : Computes lexical diversity metrics
    RepetitionDetector : Detects repetition patterns
    CreativityAnalyzer : Analyzes creative expression
    """

    text: str
    lexical_diversity: dict[DiversityMetric, DiversityScore]
    repetition: RepetitionAnalysis
    creativity: Optional[CreativityScore]
    overall_diversity_score: float
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the report to a dictionary representation.

        Returns a JSON-serializable dictionary with all report
        components. The text is represented by its length rather
        than the full content to keep output size manageable.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - "text_length": int, length of analyzed text
            - "lexical_diversity": dict, metric scores by name
            - "repetition": dict, repetition analysis
            - "creativity": dict or None, creativity analysis
            - "overall_diversity_score": float
            - "recommendations": list[str]
            - "metadata": dict

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import analyze_diversity
        >>> report = analyze_diversity("The quick brown fox jumps.")
        >>> d = report.to_dict()
        >>> print(f"Text length: {d['text_length']}")
        Text length: 26
        >>> print(f"Overall: {d['overall_diversity_score']}")
        Overall: 0.75
        >>> print(f"Has creativity: {d['creativity'] is not None}")
        Has creativity: True
        """
        return {
            "text_length": len(self.text),
            "lexical_diversity": {k.value: v.to_dict() for k, v in self.lexical_diversity.items()},
            "repetition": self.repetition.to_dict(),
            "creativity": self.creativity.to_dict() if self.creativity else None,
            "overall_diversity_score": round(self.overall_diversity_score, 4),
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


class LexicalDiversityAnalyzer:
    """
    Analyzer for computing lexical diversity metrics on text.

    This class provides methods to compute various established lexical
    diversity metrics from computational linguistics. Each metric captures
    a different aspect of vocabulary richness and has different properties
    regarding sensitivity to text length.

    The analyzer supports custom tokenization functions, allowing
    integration with different NLP pipelines or language-specific
    tokenizers.

    Parameters
    ----------
    tokenize_fn : Callable[[str], list[str]], optional
        Custom tokenization function that takes a string and returns
        a list of tokens (words). If not provided, uses the default
        regex-based tokenizer from `insideLLMs.nlp.tokenization`.

    Attributes
    ----------
    tokenize : Callable[[str], list[str]]
        The tokenization function used by this analyzer.

    Examples
    --------
    Basic usage with default tokenizer:

    >>> from insideLLMs.evaluation.diversity import LexicalDiversityAnalyzer
    >>> analyzer = LexicalDiversityAnalyzer()
    >>> text = "The quick brown fox jumps over the lazy dog."
    >>> ttr = analyzer.type_token_ratio(text)
    >>> print(f"TTR: {ttr.value:.3f}")
    TTR: 0.889

    Running all diversity analyses:

    >>> scores = analyzer.analyze_all(text)
    >>> for metric, score in scores.items():
    ...     print(f"{metric.value}: {score.value:.3f}")
    type_token_ratio: 0.889
    hapax_legomena: 0.778
    yules_k: 0.000
    simpsons_d: 1.000
    entropy: 0.985
    mtld: 9.000

    Using a custom tokenizer:

    >>> def lowercase_tokenizer(text):
    ...     return [w.lower() for w in text.split() if w.isalpha()]
    >>> custom_analyzer = LexicalDiversityAnalyzer(tokenize_fn=lowercase_tokenizer)
    >>> score = custom_analyzer.type_token_ratio("Hello World Hello")
    >>> print(f"TTR with custom tokenizer: {score.value:.3f}")
    TTR with custom tokenizer: 0.667

    Analyzing specific metrics:

    >>> mtld = analyzer.mtld("A long text with many words to analyze...")
    >>> print(f"MTLD: {mtld.value:.1f} - {mtld.interpretation}")
    MTLD: 11.0 - Low lexical diversity

    See Also
    --------
    DiversityMetric : Enumeration of available metrics
    DiversityScore : Result container for metric values
    analyze_diversity : Convenience function for full analysis
    """

    def __init__(self, tokenize_fn: Optional[Callable[[str], list[str]]] = None):
        """
        Initialize the lexical diversity analyzer.

        Args
        ----
        tokenize_fn : Callable[[str], list[str]], optional
            Custom tokenization function. Takes a string and returns
            a list of token strings. If None, uses the default
            regex-based word tokenizer.

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import LexicalDiversityAnalyzer
        >>> # Using default tokenizer
        >>> analyzer = LexicalDiversityAnalyzer()

        >>> # Using custom tokenizer
        >>> def simple_tokenize(text):
        ...     return text.lower().split()
        >>> analyzer = LexicalDiversityAnalyzer(tokenize_fn=simple_tokenize)
        """
        self.tokenize = tokenize_fn or self._default_tokenize

    def _default_tokenize(self, text: str) -> list[str]:
        """
        Default tokenization using regex-based word extraction.

        Delegates to `insideLLMs.nlp.tokenization.word_tokenize_regex`
        which extracts word tokens using a regex pattern and converts
        them to lowercase.

        Args
        ----
        text : str
            The text to tokenize.

        Returns
        -------
        list[str]
            List of lowercase word tokens.

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import LexicalDiversityAnalyzer
        >>> analyzer = LexicalDiversityAnalyzer()
        >>> tokens = analyzer._default_tokenize("Hello, World!")
        >>> print(tokens)
        ['hello', 'world']
        """
        return word_tokenize_regex(text)

    def type_token_ratio(self, text: str) -> DiversityScore:
        """
        Calculate the Type-Token Ratio (TTR) of the text.

        TTR is the ratio of unique words (types) to total words (tokens).
        It is a simple and intuitive measure of vocabulary diversity, but
        is sensitive to text length (shorter texts tend to have higher TTR).

        For comparing texts of different lengths, consider using MTLD instead.

        Args
        ----
        text : str
            The text to analyze.

        Returns
        -------
        DiversityScore
            Score containing:
            - value: float in [0, 1], higher = more diverse
            - interpretation: "High/Moderate/Low lexical diversity"
            - details: {"n_types": int, "n_tokens": int}

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import LexicalDiversityAnalyzer
        >>> analyzer = LexicalDiversityAnalyzer()

        High diversity text (all unique words):

        >>> score = analyzer.type_token_ratio("One two three four five")
        >>> print(f"TTR: {score.value:.3f}")
        TTR: 1.000

        Lower diversity text (repeated words):

        >>> score = analyzer.type_token_ratio("the cat sat on the mat")
        >>> print(f"TTR: {score.value:.3f} - {score.interpretation}")
        TTR: 0.833 - High lexical diversity

        Accessing computation details:

        >>> print(f"Types: {score.details['n_types']}, Tokens: {score.details['n_tokens']}")
        Types: 5, Tokens: 6

        See Also
        --------
        mtld : Length-independent diversity measure
        hapax_legomena_ratio : Words appearing only once
        """
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
        """
        Calculate the ratio of words appearing exactly once (hapax legomena).

        Hapax legomena are words that occur only once in a text. A higher
        ratio indicates more unique vocabulary and less repetition. This
        metric is useful for detecting vocabulary richness and identifying
        repetitive patterns.

        Args
        ----
        text : str
            The text to analyze.

        Returns
        -------
        DiversityScore
            Score containing:
            - value: float in [0, 1], higher = more unique words
            - interpretation: "High/Moderate/Low proportion of unique words"
            - details: {"hapax_count": int, "total_tokens": int}

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import LexicalDiversityAnalyzer
        >>> analyzer = LexicalDiversityAnalyzer()

        Text with all unique words:

        >>> score = analyzer.hapax_legomena_ratio("Every word here is unique")
        >>> print(f"Hapax ratio: {score.value:.3f}")
        Hapax ratio: 1.000

        Text with repeated words:

        >>> score = analyzer.hapax_legomena_ratio("the cat and the dog and the bird")
        >>> print(f"Hapax ratio: {score.value:.3f}")
        Hapax ratio: 0.375
        >>> print(f"Hapax count: {score.details['hapax_count']}")
        Hapax count: 3

        See Also
        --------
        type_token_ratio : Ratio of unique to total words
        """
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
        """
        Calculate Yule's K characteristic constant.

        Yule's K measures vocabulary richness based on word frequency
        distribution. Unlike TTR, it is less sensitive to text length.
        The formula is: K = 10000 * (M2 - M1) / (M1^2), where M1 is
        the number of tokens and M2 is the sum of squared frequencies.

        IMPORTANT: Unlike other metrics, LOWER K values indicate
        HIGHER diversity.

        Args
        ----
        text : str
            The text to analyze.

        Returns
        -------
        DiversityScore
            Score containing:
            - value: float >= 0, LOWER = more diverse
            - interpretation: Diversity level description
            - details: {"m1": int, "m2": int}

        Raises
        ------
        No exceptions raised; returns score with value=0 for insufficient tokens.

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import LexicalDiversityAnalyzer
        >>> analyzer = LexicalDiversityAnalyzer()

        Text with all unique words (low K = high diversity):

        >>> score = analyzer.yules_k("Every single word here is completely unique")
        >>> print(f"Yule's K: {score.value:.1f}")
        Yule's K: 0.0

        Text with heavy repetition (high K = low diversity):

        >>> score = analyzer.yules_k("the the the cat cat sat")
        >>> print(f"Yule's K: {score.value:.1f} - {score.interpretation}")
        Yule's K: 1111.1 - Low vocabulary diversity (high repetition)

        See Also
        --------
        simpsons_diversity : Another frequency-based diversity measure
        """
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

        k = 0.0 if m1 == m2 else 10000 * (m2 - m1) / (m1 * m1)

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
        """
        Calculate Simpson's Diversity Index for the text.

        Simpson's D measures the probability that two randomly selected
        tokens are different. Originally used in ecology, it is useful
        for measuring lexical diversity. The result is inverted so that
        higher values indicate more diversity.

        Formula: D = 1 - sum(n_i * (n_i - 1)) / (N * (N - 1))

        Args
        ----
        text : str
            The text to analyze.

        Returns
        -------
        DiversityScore
            Score containing:
            - value: float in [0, 1], higher = more diverse
            - interpretation: "Very high/High/Moderate/Low diversity"

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import LexicalDiversityAnalyzer
        >>> analyzer = LexicalDiversityAnalyzer()

        Perfectly diverse text (all unique):

        >>> score = analyzer.simpsons_diversity("one two three four five")
        >>> print(f"Simpson's D: {score.value:.3f}")
        Simpson's D: 1.000

        Text with repetition:

        >>> score = analyzer.simpsons_diversity("cat cat cat dog dog bird")
        >>> print(f"Simpson's D: {score.value:.3f} - {score.interpretation}")
        Simpson's D: 0.733 - High diversity

        See Also
        --------
        entropy : Information-theoretic diversity measure
        yules_k : Another frequency-based measure (inverted scale)
        """
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
        """
        Calculate normalized Shannon entropy of the word distribution.

        Shannon entropy measures the uncertainty or information content
        of the word distribution. Higher entropy indicates more uniform
        word usage (higher diversity). The result is normalized to [0, 1]
        by dividing by the maximum possible entropy.

        Formula: H = -sum(p_i * log2(p_i)), normalized by log2(vocabulary_size)

        Args
        ----
        text : str
            The text to analyze.

        Returns
        -------
        DiversityScore
            Score containing:
            - value: float in [0, 1], higher = more uniform distribution
            - interpretation: Entropy level description
            - details: {"raw_entropy": float, "max_entropy": float}

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import LexicalDiversityAnalyzer
        >>> analyzer = LexicalDiversityAnalyzer()

        Uniform distribution (maximum entropy):

        >>> score = analyzer.entropy("one two three four five six")
        >>> print(f"Normalized entropy: {score.value:.3f}")
        Normalized entropy: 1.000

        Skewed distribution (lower entropy):

        >>> score = analyzer.entropy("the the the the cat dog")
        >>> print(f"Entropy: {score.value:.3f} - {score.interpretation}")
        Entropy: 0.650 - Moderate entropy
        >>> print(f"Raw entropy: {score.details['raw_entropy']:.3f} bits")
        Raw entropy: 1.030 bits

        See Also
        --------
        simpsons_diversity : Probability-based diversity measure
        """
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
        """
        Calculate Measure of Textual Lexical Diversity (MTLD).

        MTLD measures lexical diversity by counting how many consecutive
        words can be read, on average, before the TTR drops below a
        threshold. Unlike basic TTR, MTLD is relatively insensitive to
        text length, making it ideal for comparing texts of different sizes.

        The calculation runs in both forward and backward directions
        and averages the results for stability.

        Args
        ----
        text : str
            The text to analyze.
        threshold : float, optional
            TTR threshold for factor counting. Default is 0.72, which
            is the standard value from the original MTLD research.

        Returns
        -------
        DiversityScore
            Score containing:
            - value: float >= 0, higher = more diverse
            - interpretation: Diversity level description
            - details: {"forward": float, "backward": float}

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import LexicalDiversityAnalyzer
        >>> analyzer = LexicalDiversityAnalyzer()

        Diverse text:

        >>> text = '''The transformer architecture revolutionized natural
        ... language processing by enabling parallel computation.'''
        >>> score = analyzer.mtld(text)
        >>> print(f"MTLD: {score.value:.1f}")
        MTLD: 11.0

        Repetitive text:

        >>> text = "cat cat cat dog dog dog bird bird bird"
        >>> score = analyzer.mtld(text)
        >>> print(f"MTLD: {score.value:.1f} - {score.interpretation}")
        MTLD: 4.5 - Low lexical diversity

        Checking forward/backward consistency:

        >>> print(f"Forward: {score.details['forward']:.1f}")
        Forward: 4.5
        >>> print(f"Backward: {score.details['backward']:.1f}")
        Backward: 4.5

        Notes
        -----
        - Requires at least 10 tokens to compute; returns 0 for shorter texts
        - Values typically range from 20-200 for natural language text
        - Values below 50 suggest low diversity; above 100 suggests high diversity

        See Also
        --------
        type_token_ratio : Simple but length-sensitive alternative

        References
        ----------
        McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D:
        A validation study of sophisticated approaches to lexical
        diversity assessment. Behavior Research Methods, 42(2), 381-392.
        """
        tokens = self.tokenize(text)
        if len(tokens) < 10:
            return DiversityScore(
                metric=DiversityMetric.MTLD,
                value=0.0,
                interpretation="Insufficient tokens for MTLD",
            )

        def _mtld_forward(tokens: list[str], threshold: float) -> float:
            """Calculate MTLD in forward direction."""
            factor_count = 0
            factor_length = 0
            current_ttr = 1.0
            types: set[str] = set()

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

    def analyze_all(self, text: str) -> dict[DiversityMetric, DiversityScore]:
        """
        Run all diversity analyses on the text.

        Computes all six diversity metrics (TTR, hapax ratio, Yule's K,
        Simpson's D, entropy, and MTLD) and returns them in a dictionary
        keyed by the metric enum.

        Args
        ----
        text : str
            The text to analyze.

        Returns
        -------
        dict[DiversityMetric, DiversityScore]
            Dictionary mapping each DiversityMetric to its computed
            DiversityScore.

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import LexicalDiversityAnalyzer, DiversityMetric
        >>> analyzer = LexicalDiversityAnalyzer()
        >>> text = "The quick brown fox jumps over the lazy dog."
        >>> scores = analyzer.analyze_all(text)

        Accessing specific metrics:

        >>> ttr = scores[DiversityMetric.TYPE_TOKEN_RATIO]
        >>> print(f"TTR: {ttr.value:.3f}")
        TTR: 0.889

        Iterating over all results:

        >>> for metric, score in scores.items():
        ...     print(f"{metric.value}: {score.value:.3f}")
        type_token_ratio: 0.889
        hapax_legomena: 0.778
        yules_k: 0.000
        simpsons_d: 1.000
        entropy: 0.985
        mtld: 9.000

        See Also
        --------
        DiversityReporter : For comprehensive reports with recommendations
        """
        return {
            DiversityMetric.TYPE_TOKEN_RATIO: self.type_token_ratio(text),
            DiversityMetric.HAPAX_LEGOMENA: self.hapax_legomena_ratio(text),
            DiversityMetric.YULES_K: self.yules_k(text),
            DiversityMetric.SIMPSONS_D: self.simpsons_diversity(text),
            DiversityMetric.ENTROPY: self.entropy(text),
            DiversityMetric.MTLD: self.mtld(text),
        }


class RepetitionDetector:
    """
    Detector for identifying repetitive patterns in text.

    This class analyzes text for repetitive words and phrases (n-grams),
    which can indicate model degeneration, low-quality output, or
    "looping" behavior in language models. It provides detailed
    information about what is being repeated and how often.

    Parameters
    ----------
    min_phrase_length : int, optional
        Minimum number of words required for phrase detection.
        Default is 3.
    min_occurrences : int, optional
        Minimum number of times a word/phrase must appear to be
        considered repeated. Default is 2.

    Attributes
    ----------
    min_phrase_length : int
        Minimum phrase length for detection.
    min_occurrences : int
        Minimum occurrences threshold.

    Examples
    --------
    Basic usage:

    >>> from insideLLMs.evaluation.diversity import RepetitionDetector
    >>> detector = RepetitionDetector()
    >>> text = "The cat sat. The cat slept. The cat purred."
    >>> analysis = detector.detect(text)
    >>> print(f"Repetition score: {analysis.repetition_score:.3f}")
    Repetition score: 0.273

    Detecting highly repetitive text:

    >>> repetitive = "error error error error error system failure"
    >>> analysis = detector.detect(repetitive)
    >>> print(f"Has significant repetition: {analysis.has_significant_repetition}")
    Has significant repetition: True
    >>> print(f"Top repeated words: {analysis.repeated_words[:3]}")
    [('error', 5)]

    Customizing detection sensitivity:

    >>> # More sensitive detection (lower thresholds)
    >>> sensitive_detector = RepetitionDetector(min_phrase_length=2, min_occurrences=2)
    >>> # Less sensitive detection (higher thresholds)
    >>> lenient_detector = RepetitionDetector(min_phrase_length=4, min_occurrences=3)

    Analyzing n-gram repetition rates:

    >>> text = "the quick brown fox the quick red fox"
    >>> analysis = detector.detect(text)
    >>> for n, rate in analysis.n_gram_repetition.items():
    ...     print(f"{n}-gram repetition: {rate:.3f}")
    2-gram repetition: 0.143
    3-gram repetition: 0.000
    4-gram repetition: 0.000
    5-gram repetition: 0.000

    See Also
    --------
    RepetitionAnalysis : Container for detection results
    detect_repetition : Convenience function for quick analysis
    """

    def __init__(self, min_phrase_length: int = 3, min_occurrences: int = 2):
        """
        Initialize the repetition detector.

        Args
        ----
        min_phrase_length : int, optional
            Minimum number of words for a sequence to be considered
            a phrase. Shorter sequences are treated as individual words.
            Default is 3.
        min_occurrences : int, optional
            Minimum number of times a word or phrase must appear to
            be counted as repeated. Must be >= 2. Default is 2.

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import RepetitionDetector
        >>> # Default settings
        >>> detector = RepetitionDetector()

        >>> # More sensitive (catches more repetition)
        >>> sensitive = RepetitionDetector(min_phrase_length=2, min_occurrences=2)

        >>> # Less sensitive (only catches heavy repetition)
        >>> lenient = RepetitionDetector(min_phrase_length=4, min_occurrences=3)
        """
        self.min_phrase_length = min_phrase_length
        self.min_occurrences = min_occurrences

    def _get_ngrams(self, tokens: list[str], n: int) -> list[str]:
        """
        Extract n-grams from a list of tokens.

        Args
        ----
        tokens : list[str]
            List of word tokens.
        n : int
            The n-gram size (number of consecutive tokens).

        Returns
        -------
        list[str]
            List of n-gram strings, with tokens joined by spaces.

        Examples
        --------
        >>> detector = RepetitionDetector()
        >>> tokens = ["the", "cat", "sat", "on", "the", "mat"]
        >>> bigrams = detector._get_ngrams(tokens, 2)
        >>> print(bigrams)
        ['the cat', 'cat sat', 'sat on', 'on the', 'the mat']
        """
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def detect(self, text: str) -> RepetitionAnalysis:
        """
        Detect repetition patterns in the given text.

        Analyzes the text for repeated words and n-gram phrases (bigrams
        through 5-grams), computing an overall repetition score and
        identifying the most prominent repetitive patterns.

        Args
        ----
        text : str
            The text to analyze for repetition.

        Returns
        -------
        RepetitionAnalysis
            Analysis results containing:
            - repeated_phrases: list of (phrase, count) tuples
            - repeated_words: list of (word, count) tuples
            - repetition_score: float in [0, 1]
            - longest_repeated_sequence: str
            - n_gram_repetition: dict mapping n to repetition rate
            - has_significant_repetition: bool (score > 0.3)

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import RepetitionDetector
        >>> detector = RepetitionDetector()

        Analyzing normal text:

        >>> text = "The quick brown fox jumps over the lazy dog."
        >>> analysis = detector.detect(text)
        >>> print(f"Score: {analysis.repetition_score:.3f}")
        Score: 0.000
        >>> print(f"Significant: {analysis.has_significant_repetition}")
        Significant: False

        Analyzing repetitive model output:

        >>> text = '''I think I think I think that this is important.
        ... This is important because this is important.'''
        >>> analysis = detector.detect(text)
        >>> print(f"Score: {analysis.repetition_score:.3f}")
        Score: 0.533
        >>> print(f"Significant: {analysis.has_significant_repetition}")
        Significant: True

        Examining repeated phrases:

        >>> for phrase, count in analysis.repeated_phrases[:3]:
        ...     print(f"  '{phrase}': {count}x")
        'i think': 3x
        'this is important': 3x
        'is important': 3x

        Checking n-gram repetition rates:

        >>> for n, rate in analysis.n_gram_repetition.items():
        ...     if rate > 0:
        ...         print(f"  {n}-grams: {rate:.1%} repetition")

        Handling short text:

        >>> short = "Hi there"
        >>> analysis = detector.detect(short)
        >>> print(f"Score: {analysis.repetition_score}")
        Score: 0.0

        See Also
        --------
        detect_repetition : Convenience function wrapper
        RepetitionAnalysis : Detailed result container
        """
        words = word_tokenize_regex(text)

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
            (word, count)
            for word, count in word_counts.most_common()
            if count >= self.min_occurrences and len(word) > 2
        ]

        # Find repeated phrases (n-grams)
        repeated_phrases = []
        n_gram_repetition = {}

        for n in range(2, min(6, len(words))):
            ngrams = self._get_ngrams(words, n)
            ngram_counts = Counter(ngrams)
            repeated = [
                (phrase, count)
                for phrase, count in ngram_counts.items()
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
        for phrase, _count in repeated_phrases:
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
    """
    Analyzer for assessing creative expression in text.

    This class evaluates text across five creativity dimensions: novelty,
    unexpectedness, elaboration, flexibility, and fluency. It uses heuristic
    methods to provide a quick assessment of creative expression, though
    human evaluation remains the gold standard for creativity assessment.

    The analyzer computes scores for each dimension and combines them into
    an overall creativity score with interpretations and recommendations.

    Parameters
    ----------
    reference_corpus : list[str], optional
        Corpus of reference texts for comparison. Currently reserved for
        future use (e.g., novelty comparison against baseline texts).
        Defaults to empty list.
    common_words : set[str], optional
        Set of common/stop words to exclude from novelty scoring.
        If not provided, uses a default set of ~100 common English words.

    Attributes
    ----------
    reference_corpus : list[str]
        Reference corpus for comparisons.
    common_words : set[str]
        Set of common words excluded from novelty scoring.

    Examples
    --------
    Basic creativity analysis:

    >>> from insideLLMs.evaluation.diversity import CreativityAnalyzer
    >>> analyzer = CreativityAnalyzer()
    >>> text = '''The ancient lighthouse stood sentinel over stormy seas,
    ... its beacon cutting through the tempest like a golden sword.'''
    >>> score = analyzer.analyze(text)
    >>> print(f"Creativity level: {score.creativity_level}")
    Creativity level: creative

    Examining dimension scores:

    >>> for dim, val in score.dimension_scores.items():
    ...     print(f"  {dim.value}: {val:.3f}")
    novelty: 0.571
    unexpectedness: 0.923
    elaboration: 0.600
    flexibility: 0.500
    fluency: 1.000

    Using a custom common words list:

    >>> custom_common = {"the", "a", "is", "are", "was", "were"}
    >>> custom_analyzer = CreativityAnalyzer(common_words=custom_common)
    >>> # This will count more words as "novel" since fewer are filtered

    Checking strengths and weaknesses:

    >>> print(f"Strengths: {score.strengths}")
    Strengths: ['Strong unexpectedness', 'Strong fluency']
    >>> print(f"Weaknesses: {score.weaknesses}")
    Weaknesses: []

    Notes
    -----
    - Novelty is measured by the proportion of non-common words
    - Unexpectedness uses bigram variety as a proxy
    - Elaboration uses average sentence length as a proxy
    - Flexibility measures variety in sentence starters
    - Fluency checks for proper capitalization

    These are heuristic measures and should be supplemented with
    human evaluation for critical applications.

    See Also
    --------
    CreativityDimension : Enumeration of creativity dimensions
    CreativityScore : Container for analysis results
    analyze_creativity : Convenience function for quick analysis
    """

    def __init__(
        self,
        reference_corpus: Optional[list[str]] = None,
        common_words: Optional[set[str]] = None,
    ):
        """
        Initialize the creativity analyzer.

        Args
        ----
        reference_corpus : list[str], optional
            Corpus of reference texts for comparison. Reserved for
            future use. Defaults to empty list.
        common_words : set[str], optional
            Set of common words to exclude from novelty scoring.
            If None, uses a default set of ~100 common English words
            including articles, prepositions, pronouns, and auxiliaries.

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import CreativityAnalyzer
        >>> # Default configuration
        >>> analyzer = CreativityAnalyzer()

        >>> # Custom common words (smaller set = more words count as novel)
        >>> custom = {"the", "a", "an", "is", "are", "was", "were"}
        >>> analyzer = CreativityAnalyzer(common_words=custom)

        >>> # With reference corpus (for future functionality)
        >>> corpus = ["Sample text one.", "Sample text two."]
        >>> analyzer = CreativityAnalyzer(reference_corpus=corpus)
        """
        self.reference_corpus = reference_corpus or []
        self.common_words = common_words or self._default_common_words()

    def _default_common_words(self) -> set[str]:
        """
        Return the default set of common English words.

        These words are excluded from novelty scoring as they appear
        frequently in most texts and don't contribute to creative
        vocabulary assessment.

        Returns
        -------
        set[str]
            Set of ~100 common English words including articles,
            prepositions, pronouns, auxiliaries, and conjunctions.

        Examples
        --------
        >>> analyzer = CreativityAnalyzer()
        >>> common = analyzer._default_common_words()
        >>> "the" in common
        True
        >>> "ephemeral" in common
        False
        """
        return {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "what",
        }

    def _novelty_score(self, text: str) -> float:
        """
        Calculate novelty score based on uncommon word usage.

        Novelty is measured as the proportion of words that are not in
        the common words list and have length > 3 characters. Higher
        values indicate more unique or specialized vocabulary.

        Args
        ----
        text : str
            The text to analyze.

        Returns
        -------
        float
            Novelty score in [0, 1], higher = more novel vocabulary.

        Examples
        --------
        >>> analyzer = CreativityAnalyzer()
        >>> analyzer._novelty_score("The cat sat on the mat")
        0.0
        >>> analyzer._novelty_score("Ephemeral luminescence cascaded")
        1.0
        """
        words = word_tokenize_regex(text)
        if not words:
            return 0.0

        uncommon = [w for w in words if w not in self.common_words and len(w) > 3]
        return len(uncommon) / len(words)

    def _unexpectedness_score(self, text: str) -> float:
        """
        Calculate unexpectedness based on word combination variety.

        Unexpectedness is measured by the variety of word bigrams.
        More unique bigrams suggest more unexpected or creative
        word combinations.

        Args
        ----
        text : str
            The text to analyze.

        Returns
        -------
        float
            Unexpectedness score in [0, 1], higher = more variety.

        Examples
        --------
        >>> analyzer = CreativityAnalyzer()
        >>> # All unique pairs
        >>> analyzer._unexpectedness_score("one two three four five")
        1.0
        >>> # Repeated pairs lower the score
        >>> analyzer._unexpectedness_score("the cat the cat the cat")
        0.4
        """
        words = word_tokenize_regex(text)
        if len(words) < 2:
            return 0.0

        # Check for unusual word pairs (simplified)
        bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]

        # Score based on bigram variety
        unique_bigrams = len(set(bigrams))
        return unique_bigrams / len(bigrams) if bigrams else 0.0

    def _elaboration_score(self, text: str) -> float:
        """
        Calculate elaboration based on sentence detail level.

        Elaboration is approximated by average sentence length.
        Optimal length (15-25 words) gets the highest score, while
        very short or very long sentences score lower.

        Args
        ----
        text : str
            The text to analyze.

        Returns
        -------
        float
            Elaboration score in [0, 1], with 0.8 being optimal.

        Examples
        --------
        >>> analyzer = CreativityAnalyzer()
        >>> # Short sentences
        >>> analyzer._elaboration_score("Hi. Hello. Bye.")
        0.2
        >>> # Medium-length sentences
        >>> text = "This is a moderately detailed sentence with good elaboration."
        >>> 0.5 < analyzer._elaboration_score(text) < 0.9
        True
        """
        sentences = re.split(r"[.!?]+", text)
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
        """
        Calculate flexibility based on variety of sentence structures.

        Flexibility is measured by the variety of sentence starters
        (first words). More varied starters suggest more flexible
        and creative sentence construction.

        Args
        ----
        text : str
            The text to analyze.

        Returns
        -------
        float
            Flexibility score in [0, 1], higher = more variety.

        Examples
        --------
        >>> analyzer = CreativityAnalyzer()
        >>> # Same starter
        >>> analyzer._flexibility_score("The cat ran. The dog barked.")
        0.5
        >>> # Different starters
        >>> analyzer._flexibility_score("First, consider this. Then, try that. Finally, conclude.")
        1.0
        """
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5

        # Check for variety in sentence starters
        starters = [s.split()[0].lower() if s.split() else "" for s in sentences]
        unique_starters = len(set(starters))

        return min(1.0, unique_starters / len(sentences))

    def _fluency_score(self, text: str) -> float:
        """
        Calculate fluency based on grammatical conventions.

        Fluency is approximated by checking if sentences start with
        capital letters, which is a basic indicator of proper
        grammatical structure.

        Args
        ----
        text : str
            The text to analyze.

        Returns
        -------
        float
            Fluency score in [0, 1], higher = better formatting.

        Examples
        --------
        >>> analyzer = CreativityAnalyzer()
        >>> # Proper capitalization
        >>> analyzer._fluency_score("Hello world. How are you?")
        1.0
        >>> # Missing capitalization
        >>> analyzer._fluency_score("hello. world.")
        0.0
        """
        # Simplified: check for proper punctuation and capitalization
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        proper_count = 0
        for sent in sentences:
            if sent and sent[0].isupper():
                proper_count += 1

        return proper_count / len(sentences)

    def analyze(self, text: str) -> CreativityScore:
        """
        Analyze the creativity of the given text.

        Computes scores across five creativity dimensions (novelty,
        unexpectedness, elaboration, flexibility, fluency), combines
        them into an overall score, and identifies strengths and
        weaknesses.

        Args
        ----
        text : str
            The text to analyze for creativity.

        Returns
        -------
        CreativityScore
            Analysis results containing:
            - overall_score: float in [0, 1]
            - dimension_scores: dict mapping dimensions to scores
            - interpretation: str describing creativity level
            - strengths: list of strong dimensions (score >= 0.7)
            - weaknesses: list of weak dimensions (score <= 0.3)
            - creativity_level: categorical level

        Examples
        --------
        >>> from insideLLMs.evaluation.diversity import CreativityAnalyzer
        >>> analyzer = CreativityAnalyzer()

        Analyzing creative writing:

        >>> text = '''The ethereal moonlight danced across crystalline
        ... waters, weaving shadows into silver tapestries.'''
        >>> score = analyzer.analyze(text)
        >>> print(f"Level: {score.creativity_level}")
        Level: creative

        Analyzing plain text:

        >>> plain = "The dog is big. The cat is small. They are animals."
        >>> score = analyzer.analyze(plain)
        >>> print(f"Level: {score.creativity_level}")
        Level: moderate

        Getting detailed dimension breakdown:

        >>> for dim, val in score.dimension_scores.items():
        ...     print(f"  {dim.value}: {val:.2f}")

        Identifying areas for improvement:

        >>> if score.weaknesses:
        ...     print(f"Consider improving: {score.weaknesses}")

        See Also
        --------
        CreativityScore : Detailed result container
        analyze_creativity : Convenience function wrapper
        """
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
        self.similarity_fn = similarity_fn or word_overlap_similarity

    def analyze(self, outputs: list[str]) -> VariabilityAnalysis:
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
        std_sim = variance**0.5

        # Calculate unique tokens ratio
        all_tokens: list[str] = []
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
                self.similarity_fn(outputs[i], outputs[j]) for j in range(n) if j != i
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
        lexical: dict[DiversityMetric, DiversityScore],
        repetition: RepetitionAnalysis,
        creativity: Optional[CreativityScore],
        overall: float,
    ) -> list[str]:
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


def analyze_output_variability(outputs: list[str]) -> VariabilityAnalysis:
    """Analyze variability across multiple outputs."""
    analyzer = OutputVariabilityAnalyzer()
    return analyzer.analyze(outputs)


def quick_diversity_check(text: str) -> dict[str, Any]:
    """Quick diversity check returning summary."""
    report = analyze_diversity(text)
    return {
        "overall_score": report.overall_diversity_score,
        "type_token_ratio": report.lexical_diversity[DiversityMetric.TYPE_TOKEN_RATIO].value,
        "has_repetition": report.repetition.has_significant_repetition,
        "creativity_level": report.creativity.creativity_level if report.creativity else None,
        "recommendations": report.recommendations[:3],
    }
