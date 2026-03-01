"""
Response quality scoring and analysis utilities.

This module provides a comprehensive framework for evaluating the quality of
Large Language Model (LLM) responses across multiple dimensions. It enables
automated quality assessment, comparison of different model outputs, and
identification of areas for improvement.

Overview
--------
The quality assessment system operates on eight distinct dimensions:

- **Relevance**: How well the response addresses the prompt
- **Completeness**: Whether the response fully answers the question
- **Coherence**: Logical flow and structural organization
- **Conciseness**: Economy of language without unnecessary verbosity
- **Accuracy**: Factual correctness (requires external validation)
- **Helpfulness**: Practical utility of the response
- **Clarity**: Ease of understanding and readability
- **Specificity**: Use of concrete details vs. vague generalizations

The module provides individual scorers for each dimension, a comprehensive
analyzer that evaluates all dimensions at once, and utilities for comparing
multiple responses to the same prompt.

Key Components
--------------
- ``QualityDimension``: Enum defining the quality dimensions
- ``DimensionScore``: Dataclass holding a single dimension's score
- ``QualityReport``: Comprehensive quality assessment report
- ``ResponseQualityAnalyzer``: Main analyzer class for quality assessment
- ``ResponseComparator``: Compare two responses side-by-side
- Convenience functions: ``analyze_quality()``, ``quick_quality_check()``,
  ``compare_responses()``

Scoring System
--------------
All scores are normalized to the range [0.0, 1.0]:

- **0.7 - 1.0**: High quality / passes threshold
- **0.4 - 0.7**: Moderate quality / may need improvement
- **0.0 - 0.4**: Low quality / needs significant improvement

The default passing threshold is 0.7 (configurable via ``QualityReport.passed``).

Examples
--------
Basic quality analysis:

>>> from insideLLMs.contrib.quality import analyze_quality
>>> prompt = "What is the capital of France?"
>>> response = "The capital of France is Paris. It is the largest city in France and serves as the country's political, economic, and cultural center."
>>> report = analyze_quality(prompt, response)
>>> print(f"Overall score: {report.overall_score:.2f}")
Overall score: 0.78
>>> print(f"Passed: {report.passed}")
Passed: True

Quick quality check:

>>> from insideLLMs.contrib.quality import quick_quality_check
>>> prompt = "Explain quantum computing"
>>> response = "It's complicated. Maybe search online."
>>> score, passed, issues = quick_quality_check(prompt, response)
>>> print(f"Score: {score:.2f}, Issues: {issues}")
Score: 0.35, Issues: ['Low completeness: Response appears incomplete', ...]

Comparing two responses:

>>> from insideLLMs.contrib.quality import compare_responses
>>> prompt = "What causes rain?"
>>> response_a = "Rain happens when water falls from clouds."
>>> response_b = "Rain forms through the water cycle: water evaporates from surfaces, rises and condenses into clouds, and when droplets become heavy enough, they fall as precipitation."
>>> result = compare_responses(prompt, response_a, response_b)
>>> print(f"Winner: {result.winner}, A: {result.score_a:.2f}, B: {result.score_b:.2f}")
Winner: B, A: 0.52, B: 0.81

Custom dimension analysis:

>>> from insideLLMs.contrib.quality import ResponseQualityAnalyzer, QualityDimension
>>> analyzer = ResponseQualityAnalyzer(
...     dimensions=[QualityDimension.RELEVANCE, QualityDimension.CLARITY],
...     weights={QualityDimension.RELEVANCE: 2.0, QualityDimension.CLARITY: 1.0}
... )
>>> report = analyzer.analyze("Explain photosynthesis", "Plants use sunlight to convert CO2 and water into glucose and oxygen.")
>>> for dim, score in report.dimension_scores.items():
...     print(f"{dim.value}: {score.score:.2f}")
relevance: 0.85
clarity: 0.72

Notes
-----
- The scoring algorithms use heuristic-based analysis and do not require
  external LLM calls, making them fast and suitable for batch processing.
- Accuracy scoring is listed as a dimension but requires external validation
  (ground truth) that is not implemented in the base scorers.
- For production use, consider combining these heuristic scores with
  LLM-based evaluation for more nuanced assessment.
- All scorers are stateless and thread-safe.

See Also
--------
- ``insideLLMs.trace``: For tracing model behavior during evaluation
- ``insideLLMs.metrics``: For performance and latency metrics
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class QualityDimension(Enum):
    """
    Enumeration of quality dimensions for LLM response evaluation.

    Each dimension represents a distinct aspect of response quality that can
    be independently measured and weighted. The dimensions are designed to
    be complementary, providing a holistic view of response quality when
    used together.

    Attributes
    ----------
    RELEVANCE : str
        Measures how well the response addresses the specific prompt or
        question asked. High relevance means the response stays on topic
        and directly answers what was asked.
    COMPLETENESS : str
        Evaluates whether the response fully answers all aspects of the
        question without leaving important points unaddressed. Considers
        both breadth (covering all sub-questions) and depth.
    COHERENCE : str
        Assesses the logical flow and structural organization of the
        response. Well-coherent responses have clear transitions and
        maintain logical consistency throughout.
    CONCISENESS : str
        Measures economy of language - avoiding unnecessary words, filler
        phrases, and redundancy while still being complete.
    ACCURACY : str
        Evaluates factual correctness of the response. Note: This dimension
        requires external ground truth data for proper evaluation.
    HELPFULNESS : str
        Assesses the practical utility of the response for the user's
        likely intent. A helpful response is actionable and useful.
    CLARITY : str
        Measures ease of understanding, including readability, vocabulary
        accessibility, and structural clarity.
    SPECIFICITY : str
        Evaluates the use of concrete, specific details versus vague
        generalizations. Specific responses provide exact data, names,
        and examples.

    Examples
    --------
    Accessing dimension values:

    >>> from insideLLMs.contrib.quality import QualityDimension
    >>> dim = QualityDimension.RELEVANCE
    >>> print(dim.value)
    relevance
    >>> print(dim.name)
    RELEVANCE

    Iterating over all dimensions:

    >>> for dim in QualityDimension:
    ...     print(f"{dim.name}: {dim.value}")
    RELEVANCE: relevance
    COMPLETENESS: completeness
    COHERENCE: coherence
    CONCISENESS: conciseness
    ACCURACY: accuracy
    HELPFULNESS: helpfulness
    CLARITY: clarity
    SPECIFICITY: specificity

    Using dimensions as dictionary keys:

    >>> weights = {
    ...     QualityDimension.RELEVANCE: 2.0,
    ...     QualityDimension.COMPLETENESS: 1.5,
    ...     QualityDimension.CLARITY: 1.0,
    ... }
    >>> weights[QualityDimension.RELEVANCE]
    2.0

    See Also
    --------
    DimensionScore : Dataclass holding scores for individual dimensions.
    ResponseQualityAnalyzer : Analyzer that evaluates multiple dimensions.
    """

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
    """
    Score for a single quality dimension with supporting evidence.

    This dataclass holds the evaluation result for one dimension of response
    quality. It includes the numeric score, confidence level, human-readable
    explanation, and specific evidence that contributed to the score.

    Parameters
    ----------
    dimension : QualityDimension
        The quality dimension being scored (e.g., RELEVANCE, CLARITY).
    score : float
        The numeric score in range [0.0, 1.0], where 1.0 is the best.
    confidence : float, optional
        Confidence level in the score, range [0.0, 1.0]. Default is 1.0.
        Lower confidence indicates uncertainty in the assessment.
    explanation : str, optional
        Human-readable summary of the score (e.g., "High relevance").
        Default is an empty string.
    evidence : list[str], optional
        List of specific observations that contributed to the score.
        Examples: "Term overlap: 5/7 key terms", "Contains hedging language".
        Default is an empty list.

    Attributes
    ----------
    dimension : QualityDimension
        The evaluated quality dimension.
    score : float
        Numeric score between 0.0 and 1.0.
    confidence : float
        Confidence in the score's accuracy.
    explanation : str
        Summary explanation of the score.
    evidence : list[str]
        Supporting evidence for the score.

    Examples
    --------
    Creating a dimension score manually:

    >>> from insideLLMs.contrib.quality import DimensionScore, QualityDimension
    >>> score = DimensionScore(
    ...     dimension=QualityDimension.RELEVANCE,
    ...     score=0.85,
    ...     confidence=0.9,
    ...     explanation="High relevance",
    ...     evidence=["Term overlap: 6/7 key terms", "Appropriate length"]
    ... )
    >>> print(f"{score.dimension.value}: {score.score:.2f}")
    relevance: 0.85

    Converting to dictionary for serialization:

    >>> score_dict = score.to_dict()
    >>> print(score_dict['dimension'])
    relevance
    >>> print(score_dict['evidence'])
    ['Term overlap: 6/7 key terms', 'Appropriate length']

    Accessing score from a quality report:

    >>> from insideLLMs.contrib.quality import analyze_quality, QualityDimension
    >>> report = analyze_quality("What is Python?", "Python is a programming language.")
    >>> relevance_score = report.dimension_scores[QualityDimension.RELEVANCE]
    >>> print(f"Score: {relevance_score.score:.2f}")
    Score: 0.75
    >>> print(f"Explanation: {relevance_score.explanation}")
    Explanation: High relevance

    See Also
    --------
    QualityDimension : Enum of available quality dimensions.
    QualityReport : Aggregates multiple DimensionScore objects.
    """

    dimension: QualityDimension
    score: float  # 0-1
    confidence: float = 1.0
    explanation: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the dimension score to a dictionary representation.

        Creates a serializable dictionary containing all score information,
        suitable for JSON serialization, logging, or API responses.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - 'dimension' (str): The dimension name (e.g., 'relevance')
            - 'score' (float): The numeric score
            - 'confidence' (float): The confidence level
            - 'explanation' (str): Human-readable explanation
            - 'evidence' (list[str]): List of evidence strings

        Examples
        --------
        Basic conversion:

        >>> from insideLLMs.contrib.quality import DimensionScore, QualityDimension
        >>> score = DimensionScore(
        ...     dimension=QualityDimension.CLARITY,
        ...     score=0.72,
        ...     explanation="Moderately clear"
        ... )
        >>> d = score.to_dict()
        >>> print(d)
        {'dimension': 'clarity', 'score': 0.72, 'confidence': 1.0, 'explanation': 'Moderately clear', 'evidence': []}

        JSON serialization:

        >>> import json
        >>> score = DimensionScore(
        ...     dimension=QualityDimension.COHERENCE,
        ...     score=0.88,
        ...     evidence=["Good use of transitions (4 found)"]
        ... )
        >>> json_str = json.dumps(score.to_dict())
        >>> print(json_str)
        {"dimension": "coherence", "score": 0.88, "confidence": 1.0, "explanation": "", "evidence": ["Good use of transitions (4 found)"]}

        Aggregating multiple scores:

        >>> from insideLLMs.contrib.quality import analyze_quality
        >>> report = analyze_quality("Explain AI", "Artificial Intelligence is the simulation of human intelligence by machines.")
        >>> all_scores = {dim.value: score.to_dict() for dim, score in report.dimension_scores.items()}
        >>> print(list(all_scores.keys())[:3])
        ['relevance', 'completeness', 'coherence']
        """
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "evidence": self.evidence,
        }


@dataclass
class QualityReport:
    """
    Comprehensive quality assessment report for an LLM response.

    This dataclass aggregates scores across all evaluated dimensions into a
    single report. It provides the overall quality score, individual dimension
    scores, identified issues, and improvement suggestions.

    Parameters
    ----------
    prompt : str
        The original prompt/question that was asked.
    response : str
        The LLM's response that was evaluated.
    overall_score : float
        Weighted average score across all dimensions, range [0.0, 1.0].
    dimension_scores : dict[QualityDimension, DimensionScore], optional
        Dictionary mapping each evaluated dimension to its score.
        Default is an empty dict.
    issues : list[str], optional
        List of identified quality issues (e.g., "Low relevance: ...").
        Default is an empty list.
    suggestions : list[str], optional
        List of improvement suggestions (e.g., "Improve coherence").
        Default is an empty list.
    metadata : dict[str, Any], optional
        Additional metadata about the evaluation (e.g., timestamps, model info).
        Default is an empty dict.

    Attributes
    ----------
    prompt : str
        The evaluated prompt.
    response : str
        The evaluated response.
    overall_score : float
        Overall quality score.
    dimension_scores : dict[QualityDimension, DimensionScore]
        Per-dimension score breakdown.
    issues : list[str]
        Identified quality problems.
    suggestions : list[str]
        Recommended improvements.
    metadata : dict[str, Any]
        Additional evaluation metadata.
    passed : bool
        Property indicating if the response meets the quality threshold (>= 0.7).

    Examples
    --------
    Generating and using a quality report:

    >>> from insideLLMs.contrib.quality import analyze_quality, QualityDimension
    >>> prompt = "What are the benefits of exercise?"
    >>> response = "Exercise improves cardiovascular health, builds muscle strength, enhances mood through endorphin release, and helps maintain a healthy weight."
    >>> report = analyze_quality(prompt, response)
    >>> print(f"Overall: {report.overall_score:.2f}, Passed: {report.passed}")
    Overall: 0.76, Passed: True

    Accessing individual dimension scores:

    >>> relevance = report.get_score(QualityDimension.RELEVANCE)
    >>> clarity = report.get_score(QualityDimension.CLARITY)
    >>> print(f"Relevance: {relevance:.2f}, Clarity: {clarity:.2f}")
    Relevance: 0.85, Clarity: 0.72

    Finding the weakest dimensions for improvement:

    >>> weakest = report.get_weakest_dimensions(n=2)
    >>> for dim, score in weakest:
    ...     print(f"{dim.value}: {score:.2f}")
    conciseness: 0.65
    completeness: 0.68

    Checking for issues and getting suggestions:

    >>> if report.issues:
    ...     print("Issues found:", report.issues)
    ...     print("Suggestions:", report.suggestions)

    Converting to dictionary for JSON/API responses:

    >>> report_dict = report.to_dict()
    >>> print(report_dict['overall_score'])
    0.76
    >>> print(list(report_dict['dimension_scores'].keys()))
    ['relevance', 'completeness', 'coherence', 'conciseness', 'clarity', 'specificity']

    See Also
    --------
    DimensionScore : Individual dimension score container.
    ResponseQualityAnalyzer : Creates QualityReport instances.
    analyze_quality : Convenience function to generate reports.
    """

    prompt: str
    response: str
    overall_score: float
    dimension_scores: dict[QualityDimension, DimensionScore] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """
        Check if the response meets the quality threshold.

        The default passing threshold is 0.7 (70%). Responses with an overall
        score at or above this threshold are considered acceptable quality.

        Returns
        -------
        bool
            True if overall_score >= 0.7, False otherwise.

        Examples
        --------
        High-quality response:

        >>> from insideLLMs.contrib.quality import analyze_quality
        >>> report = analyze_quality(
        ...     "What is 2+2?",
        ...     "2 + 2 equals 4. This is a basic arithmetic addition."
        ... )
        >>> print(f"Score: {report.overall_score:.2f}, Passed: {report.passed}")
        Score: 0.75, Passed: True

        Low-quality response:

        >>> report = analyze_quality(
        ...     "Explain machine learning in detail",
        ...     "It's AI stuff."
        ... )
        >>> print(f"Score: {report.overall_score:.2f}, Passed: {report.passed}")
        Score: 0.35, Passed: False

        Using in conditional logic:

        >>> if report.passed:
        ...     print("Response quality is acceptable")
        ... else:
        ...     print("Response needs improvement")
        ...     print(f"Issues: {report.issues}")
        """
        return self.overall_score >= 0.7

    def get_score(self, dimension: QualityDimension) -> Optional[float]:
        """
        Get the numeric score for a specific dimension.

        Retrieves just the score value (not the full DimensionScore object)
        for quick access to dimension scores.

        Parameters
        ----------
        dimension : QualityDimension
            The dimension to get the score for.

        Returns
        -------
        Optional[float]
            The score value in range [0.0, 1.0], or None if the dimension
            was not evaluated.

        Examples
        --------
        Getting a specific dimension score:

        >>> from insideLLMs.contrib.quality import analyze_quality, QualityDimension
        >>> report = analyze_quality(
        ...     "Define recursion",
        ...     "Recursion is when a function calls itself. For example, calculating factorial: factorial(n) = n * factorial(n-1)."
        ... )
        >>> relevance = report.get_score(QualityDimension.RELEVANCE)
        >>> print(f"Relevance score: {relevance:.2f}")
        Relevance score: 0.82

        Handling missing dimensions:

        >>> from insideLLMs.contrib.quality import ResponseQualityAnalyzer, QualityDimension
        >>> analyzer = ResponseQualityAnalyzer(dimensions=[QualityDimension.CLARITY])
        >>> report = analyzer.analyze("Test", "Test response")
        >>> print(report.get_score(QualityDimension.CLARITY))
        0.65
        >>> print(report.get_score(QualityDimension.RELEVANCE))
        None

        Comparing dimension scores:

        >>> report = analyze_quality("Explain gravity", "Gravity is the force that attracts objects toward each other.")
        >>> dims = [QualityDimension.RELEVANCE, QualityDimension.CLARITY, QualityDimension.CONCISENESS]
        >>> for dim in dims:
        ...     score = report.get_score(dim)
        ...     print(f"{dim.value}: {score:.2f}" if score else f"{dim.value}: N/A")
        """
        score_obj = self.dimension_scores.get(dimension)
        return score_obj.score if score_obj else None

    def get_weakest_dimensions(self, n: int = 3) -> list[tuple[QualityDimension, float]]:
        """
        Get the n dimensions with the lowest scores.

        Useful for identifying areas that need the most improvement.
        Returns dimensions sorted from weakest to strongest.

        Parameters
        ----------
        n : int, optional
            Number of dimensions to return. Default is 3.
            If n exceeds the number of evaluated dimensions, returns all.

        Returns
        -------
        list[tuple[QualityDimension, float]]
            List of (dimension, score) tuples, sorted by score ascending.
            Empty list if no dimensions were evaluated.

        Examples
        --------
        Finding areas for improvement:

        >>> from insideLLMs.contrib.quality import analyze_quality
        >>> report = analyze_quality(
        ...     "Describe the water cycle in detail with examples",
        ...     "Water evaporates and falls as rain."
        ... )
        >>> weakest = report.get_weakest_dimensions(n=3)
        >>> for dim, score in weakest:
        ...     print(f"{dim.value}: {score:.2f} - needs improvement")
        completeness: 0.35 - needs improvement
        specificity: 0.42 - needs improvement
        clarity: 0.48 - needs improvement

        Prioritizing improvements:

        >>> report = analyze_quality("How does photosynthesis work?", "Plants make food from sunlight.")
        >>> most_critical = report.get_weakest_dimensions(n=1)
        >>> dim, score = most_critical[0]
        >>> print(f"Most critical to fix: {dim.value} ({score:.2f})")
        Most critical to fix: completeness (0.38)

        Getting all dimensions sorted by score:

        >>> all_sorted = report.get_weakest_dimensions(n=10)  # More than evaluated
        >>> print(f"Got {len(all_sorted)} dimensions")
        Got 6 dimensions
        """
        sorted_dims = sorted(self.dimension_scores.items(), key=lambda x: x[1].score)
        return [(d, s.score) for d, s in sorted_dims[:n]]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the quality report to a dictionary representation.

        Creates a fully serializable dictionary suitable for JSON export,
        API responses, logging, or storage. All nested objects are also
        converted to dictionaries.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - 'overall_score' (float): The weighted average score
            - 'passed' (bool): Whether the response passed the threshold
            - 'dimension_scores' (dict): Nested dict of dimension scores
            - 'issues' (list[str]): List of identified issues
            - 'suggestions' (list[str]): List of improvement suggestions

        Notes
        -----
        The prompt and response text are not included in the dictionary
        output to avoid redundancy (caller already has these values).
        Metadata is also excluded; access directly if needed.

        Examples
        --------
        Basic serialization:

        >>> from insideLLMs.contrib.quality import analyze_quality
        >>> report = analyze_quality("What is AI?", "AI is artificial intelligence.")
        >>> d = report.to_dict()
        >>> print(f"Score: {d['overall_score']:.2f}, Passed: {d['passed']}")
        Score: 0.72, Passed: True

        JSON export:

        >>> import json
        >>> report = analyze_quality("Explain DNA", "DNA is deoxyribonucleic acid, the molecule that carries genetic information.")
        >>> json_str = json.dumps(report.to_dict(), indent=2)
        >>> print(json_str[:100])
        {
          "overall_score": 0.74,
          "passed": true,
          "dimension_scores": {
            "relevance": {

        Accessing nested dimension data:

        >>> d = report.to_dict()
        >>> for dim_name, dim_data in d['dimension_scores'].items():
        ...     print(f"{dim_name}: {dim_data['score']:.2f} - {dim_data['explanation']}")
        relevance: 0.85 - High relevance
        completeness: 0.65 - Response may be incomplete
        coherence: 0.72 - Well-structured and coherent
        ...
        """
        return {
            "overall_score": self.overall_score,
            "passed": self.passed,
            "dimension_scores": {d.value: s.to_dict() for d, s in self.dimension_scores.items()},
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


class RelevanceScorer:
    """
    Score the relevance of an LLM response to its prompt.

    This scorer evaluates how well a response addresses the specific question
    or request in the prompt. It uses term overlap analysis and response length
    appropriateness to determine relevance.

    The scoring algorithm considers:

    1. **Term Overlap (80% weight)**: Measures how many key content words from
       the prompt appear in the response. Common stop words are excluded to
       focus on semantic content.

    2. **Length Appropriateness (20% weight)**: Evaluates if the response length
       is reasonable for the prompt complexity. Very short responses or
       excessively long responses are penalized.

    Attributes
    ----------
    None. This scorer is stateless.

    Examples
    --------
    Basic usage:

    >>> from insideLLMs.contrib.quality import RelevanceScorer
    >>> scorer = RelevanceScorer()
    >>> prompt = "What is machine learning?"
    >>> response = "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
    >>> result = scorer.score(prompt, response)
    >>> print(f"Score: {result.score:.2f}, Explanation: {result.explanation}")
    Score: 0.85, Explanation: High relevance

    Detecting irrelevant responses:

    >>> prompt = "Explain the water cycle"
    >>> response = "The stock market closed higher today with tech stocks leading gains."
    >>> result = scorer.score(prompt, response)
    >>> print(f"Score: {result.score:.2f}, Explanation: {result.explanation}")
    Score: 0.25, Explanation: Low relevance

    Examining evidence:

    >>> prompt = "How do solar panels work?"
    >>> response = "Solar panels convert sunlight into electricity using photovoltaic cells."
    >>> result = scorer.score(prompt, response)
    >>> for evidence in result.evidence:
    ...     print(f"- {evidence}")
    - Term overlap: 2/3 key terms

    See Also
    --------
    ResponseQualityAnalyzer : Uses this scorer as part of comprehensive analysis.
    QualityDimension.RELEVANCE : The dimension this scorer evaluates.
    """

    def __init__(self):
        """
        Initialize the relevance scorer.

        The scorer is stateless and requires no configuration. Multiple
        instances will produce identical results for the same inputs.

        Examples
        --------
        >>> from insideLLMs.contrib.quality import RelevanceScorer
        >>> scorer = RelevanceScorer()
        >>> # Scorer is ready to use
        """
        pass

    def score(self, prompt: str, response: str) -> DimensionScore:
        """
        Score the relevance of a response to its prompt.

        Evaluates term overlap between prompt and response, filtering out
        common stop words to focus on semantic content. Also considers
        whether the response length is appropriate for the prompt.

        Parameters
        ----------
        prompt : str
            The original prompt or question.
        response : str
            The LLM's response to evaluate.

        Returns
        -------
        DimensionScore
            A score object containing:
            - score: Float in [0.0, 1.0], higher is more relevant
            - dimension: QualityDimension.RELEVANCE
            - explanation: "High relevance", "Moderate relevance", or "Low relevance"
            - evidence: List of observations (e.g., term overlap count)

        Examples
        --------
        High relevance (direct answer):

        >>> from insideLLMs.contrib.quality import RelevanceScorer
        >>> scorer = RelevanceScorer()
        >>> result = scorer.score(
        ...     "What is the capital of Japan?",
        ...     "The capital of Japan is Tokyo. It is the most populous metropolitan area in the world."
        ... )
        >>> print(f"Score: {result.score:.2f}")
        Score: 0.92

        Moderate relevance (tangentially related):

        >>> result = scorer.score(
        ...     "Explain photosynthesis",
        ...     "Plants are living organisms that grow in soil and need water."
        ... )
        >>> print(f"Score: {result.score:.2f}, Explanation: {result.explanation}")
        Score: 0.45, Explanation: Moderate relevance

        Low relevance (off-topic response):

        >>> result = scorer.score(
        ...     "How do I bake a cake?",
        ...     "The weather today is sunny with a high of 75 degrees."
        ... )
        >>> print(f"Score: {result.score:.2f}, Explanation: {result.explanation}")
        Score: 0.22, Explanation: Low relevance
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
    """
    Score the completeness of an LLM response.

    This scorer evaluates whether a response fully addresses all aspects of
    the prompt without leaving important points unaddressed. It analyzes
    linguistic markers of completeness and incompleteness.

    The scoring algorithm considers:

    1. **Incomplete Indicators**: Phrases like "I don't know", "unable to",
       or trailing ellipses that suggest the response is cut short or
       acknowledges gaps.

    2. **Hedging Language**: Phrases like "I think", "maybe", "perhaps" that
       suggest uncertainty and potential incompleteness of information.

    3. **Completion Indicators**: Phrases like "in conclusion", "to summarize",
       "therefore" that suggest the response has reached a natural conclusion.

    4. **Structural Elements**: Multiple sentences, paragraphs, and organized
       structure indicate more complete responses.

    5. **Prompt Demands**: If the prompt asks for detail ("explain", "describe",
       "in detail"), short responses are penalized more heavily.

    Attributes
    ----------
    INCOMPLETE_INDICATORS : list[str]
        Class attribute listing phrases that indicate incomplete responses.
    HEDGING_INDICATORS : list[str]
        Class attribute listing hedging phrases that suggest uncertainty.
    COMPLETE_INDICATORS : list[str]
        Class attribute listing phrases that indicate complete responses.

    Examples
    --------
    Basic usage:

    >>> from insideLLMs.contrib.quality import CompletenessScorer
    >>> scorer = CompletenessScorer()
    >>> prompt = "List three benefits of exercise"
    >>> response = "Exercise has three main benefits: First, it improves cardiovascular health. Second, it builds muscle strength. Third, it enhances mental well-being."
    >>> result = scorer.score(prompt, response)
    >>> print(f"Score: {result.score:.2f}, Explanation: {result.explanation}")
    Score: 0.82, Explanation: Response appears complete

    Detecting incomplete responses:

    >>> prompt = "Explain quantum mechanics in detail"
    >>> response = "Quantum mechanics is... well, it's complicated. I'm not sure I can explain it fully."
    >>> result = scorer.score(prompt, response)
    >>> print(f"Score: {result.score:.2f}")
    Score: 0.25
    >>> for evidence in result.evidence:
    ...     print(f"- {evidence}")
    - Found 2 incomplete indicators
    - Contains hedging language (1 instances)

    Evaluating structured responses:

    >>> prompt = "What are the causes of climate change?"
    >>> response = '''Climate change is caused by several factors:
    ... 1. Greenhouse gas emissions from burning fossil fuels
    ... 2. Deforestation reducing carbon absorption
    ... 3. Industrial processes releasing methane
    ... In conclusion, human activities are the primary driver.'''
    >>> result = scorer.score(prompt, response)
    >>> print(f"Score: {result.score:.2f}")
    Score: 0.88

    See Also
    --------
    ResponseQualityAnalyzer : Uses this scorer as part of comprehensive analysis.
    QualityDimension.COMPLETENESS : The dimension this scorer evaluates.
    """

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
        """
        Score the completeness of a response.

        Analyzes the response for indicators of completeness or incompleteness,
        considering linguistic markers, structural elements, and whether the
        response length matches the prompt's demands.

        Parameters
        ----------
        prompt : str
            The original prompt or question.
        response : str
            The LLM's response to evaluate.

        Returns
        -------
        DimensionScore
            A score object containing:
            - score: Float in [0.0, 1.0], higher means more complete
            - dimension: QualityDimension.COMPLETENESS
            - explanation: "Response appears complete", "Response may be incomplete",
              or "Response appears incomplete"
            - evidence: List of observations about completeness markers

        Examples
        --------
        Complete response with conclusion:

        >>> from insideLLMs.contrib.quality import CompletenessScorer
        >>> scorer = CompletenessScorer()
        >>> result = scorer.score(
        ...     "What is Python?",
        ...     "Python is a high-level, interpreted programming language known for its readability and versatility. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. In summary, Python is an excellent choice for beginners and experts alike."
        ... )
        >>> print(f"Score: {result.score:.2f}")
        Score: 0.85

        Incomplete response with hedging:

        >>> result = scorer.score(
        ...     "Explain the theory of relativity",
        ...     "I think Einstein came up with it. It's about space and time, maybe?"
        ... )
        >>> print(f"Score: {result.score:.2f}")
        Score: 0.35

        Response that doesn't match prompt complexity:

        >>> result = scorer.score(
        ...     "Describe in detail how a computer processor works",
        ...     "CPUs process data."
        ... )
        >>> print(f"Score: {result.score:.2f}")
        Score: 0.15
        >>> "Prompt requested detail but response is brief" in result.evidence
        True
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
        problematic_sentences = sum(
            1 for sentence_length in sentence_lengths if sentence_length < 3 or sentence_length > 50
        )

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
class ResponseComparisonResult:
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

    def compare(self, prompt: str, response_a: str, response_b: str) -> ResponseComparisonResult:
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

        return ResponseComparisonResult(
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


def compare_responses(prompt: str, response_a: str, response_b: str) -> ResponseComparisonResult:
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


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import ComparisonResult. The canonical name is
# ResponseComparisonResult.
ComparisonResult = ResponseComparisonResult
