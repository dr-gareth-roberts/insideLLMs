"""
Model hallucination detection utilities.

This module provides comprehensive tools for detecting potential hallucinations
in Large Language Model (LLM) outputs. Hallucinations are outputs that are
factually incorrect, fabricated, or not grounded in the provided context.

Key Features:
    - Pattern-based detection of overconfident or fabricated claims
    - Consistency checking across multiple LLM responses
    - Factual accuracy verification against knowledge bases
    - Groundedness checking to verify responses are supported by context
    - Source attribution verification

Main Classes:
    - ComprehensiveHallucinationDetector: All-in-one hallucination detection
    - PatternBasedDetector: Regex-based pattern matching for indicators
    - FactConsistencyChecker: Cross-response consistency analysis
    - FactualityChecker: Fact verification against knowledge bases
    - GroundednessChecker: Context grounding verification
    - AttributionChecker: Source attribution analysis

Data Classes:
    - HallucinationFlag: Individual detected hallucination
    - HallucinationReport: Comprehensive detection report
    - ConsistencyCheck: Cross-response consistency results
    - FactualityScore: Factual accuracy scoring

Examples:
    Basic hallucination detection:

    >>> from insideLLMs.evaluation.hallucination import detect_hallucinations
    >>> text = "According to a 2023 study, 87.3% of developers prefer Python."
    >>> report = detect_hallucinations(text)
    >>> print(f"Has hallucinations: {report.has_hallucinations}")
    Has hallucinations: True
    >>> print(f"Overall score: {report.overall_score:.2f}")
    Overall score: 0.75

    Checking consistency across multiple responses:

    >>> from insideLLMs.evaluation.hallucination import check_consistency
    >>> responses = [
    ...     "Paris is the capital of France.",
    ...     "The capital of France is Paris.",
    ...     "France's capital city is Lyon."
    ... ]
    >>> result = check_consistency(responses)
    >>> print(f"Consistent: {result.is_consistent}")
    Consistent: False

    Checking if response is grounded in context:

    >>> from insideLLMs.evaluation.hallucination import check_groundedness
    >>> context = "Python was created by Guido van Rossum in 1991."
    >>> response = "Python was invented by Guido van Rossum."
    >>> result = check_groundedness(response, context)
    >>> print(f"Grounded: {result['is_grounded']}")
    Grounded: True

    Quick hallucination check:

    >>> from insideLLMs.evaluation.hallucination import quick_hallucination_check
    >>> text = "Everyone knows that AI will definitely replace all jobs."
    >>> result = quick_hallucination_check(text)
    >>> print(f"Potential issues: {result['n_flags']}")
    Potential issues: 2
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from insideLLMs.nlp.similarity import word_overlap_similarity


class HallucinationType(Enum):
    """
    Enumeration of hallucination types that can be detected in LLM outputs.

    This enum categorizes the different kinds of hallucinations that language
    models can produce, helping to classify and prioritize detected issues.

    Attributes:
        FACTUAL_ERROR: Claims that contradict established facts.
            Example: "The Earth orbits the Moon."
        FABRICATED_ENTITY: References to non-existent people, places, or things.
            Example: "Dr. James Thompson published this in Nature" (fabricated person).
        CONFLATION: Mixing up distinct entities or facts.
            Example: "Einstein developed the theory of evolution."
        ANACHRONISM: Temporally impossible claims.
            Example: "Shakespeare used a typewriter to write his plays."
        LOGICAL_INCONSISTENCY: Contradictions within the same response.
            Example: "The object is both completely red and completely blue."
        ATTRIBUTION_ERROR: Incorrect source citations.
            Example: "As stated in the Constitution: E=mc^2."
        EXAGGERATION: Overstatements or hyperbolic claims.
            Example: "Everyone agrees that..." or "This always works."
        UNSUPPORTED_CLAIM: Assertions without verifiable evidence.
            Example: "Studies show that 73.2% of people prefer X."

    Examples:
        Classifying a detected hallucination:

        >>> flag = HallucinationFlag(
        ...     hallucination_type=HallucinationType.FACTUAL_ERROR,
        ...     severity=SeverityLevel.HIGH,
        ...     text_span="Paris is the capital of Germany",
        ...     start_pos=0, end_pos=32,
        ...     confidence=0.95,
        ...     explanation="Paris is the capital of France, not Germany"
        ... )
        >>> flag.hallucination_type.value
        'factual_error'

        Checking hallucination type:

        >>> flag.hallucination_type == HallucinationType.FACTUAL_ERROR
        True

        Getting all hallucination types:

        >>> types = [t.value for t in HallucinationType]
        >>> 'fabricated_entity' in types
        True

        Filtering flags by type:

        >>> flags = [flag1, flag2, flag3]  # List of HallucinationFlag objects
        >>> factual_errors = [f for f in flags if f.hallucination_type == HallucinationType.FACTUAL_ERROR]
    """

    FACTUAL_ERROR = "factual_error"
    FABRICATED_ENTITY = "fabricated_entity"
    CONFLATION = "conflation"  # Mixing up distinct entities/facts
    ANACHRONISM = "anachronism"  # Temporally impossible claims
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    ATTRIBUTION_ERROR = "attribution_error"  # Wrong source attribution
    EXAGGERATION = "exaggeration"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class SeverityLevel(Enum):
    """
    Severity levels for detected hallucinations.

    This enum defines the severity of detected hallucinations, helping to
    prioritize which issues need immediate attention versus those that are
    minor concerns.

    Attributes:
        LOW: Minor issues unlikely to mislead users significantly.
            Examples: hedging language, minor exaggerations.
            Penalty weight: 0.1
        MEDIUM: Moderate issues that may cause some confusion.
            Examples: unverified statistics, vague attributions.
            Penalty weight: 0.2
        HIGH: Significant issues that could mislead users.
            Examples: factual errors, fabricated citations.
            Penalty weight: 0.3
        CRITICAL: Severe issues requiring immediate attention.
            Examples: dangerous misinformation, completely fabricated entities.
            Penalty weight: 0.5

    Examples:
        Creating a flag with severity:

        >>> flag = HallucinationFlag(
        ...     hallucination_type=HallucinationType.FACTUAL_ERROR,
        ...     severity=SeverityLevel.CRITICAL,
        ...     text_span="Drinking bleach cures diseases",
        ...     start_pos=0, end_pos=30,
        ...     confidence=0.99,
        ...     explanation="Dangerous medical misinformation"
        ... )
        >>> flag.severity == SeverityLevel.CRITICAL
        True

        Comparing severity levels:

        >>> SeverityLevel.CRITICAL.value
        'critical'
        >>> SeverityLevel.HIGH.value > SeverityLevel.LOW.value  # Alphabetical comparison
        False

        Filtering critical flags:

        >>> report = detect_hallucinations(text)
        >>> critical_flags = report.critical_flags
        >>> high_priority = [f for f in report.flags if f.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]

        Using severity for scoring:

        >>> severity_weights = {
        ...     SeverityLevel.LOW: 0.1,
        ...     SeverityLevel.MEDIUM: 0.2,
        ...     SeverityLevel.HIGH: 0.3,
        ...     SeverityLevel.CRITICAL: 0.5
        ... }
        >>> penalty = severity_weights[flag.severity] * flag.confidence
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionMethod(Enum):
    """
    Enumeration of methods for detecting hallucinations.

    Different detection methods have varying strengths and are suited for
    different types of hallucinations. This enum allows specifying which
    detection approaches to use.

    Attributes:
        CONSISTENCY: Check self-consistency across multiple responses.
            Useful for detecting when the model contradicts itself.
            Implementation: Generate multiple responses and compare claims.
        FACTUAL: Verify claims against known facts or knowledge bases.
            Useful for detecting factual errors.
            Implementation: Match claims against a fact database.
        ATTRIBUTION: Verify source citations and attributions.
            Useful for detecting fabricated references.
            Implementation: Check cited sources exist and contain claimed info.
        CONFIDENCE: Use model's own confidence scores.
            Useful for identifying uncertain outputs.
            Implementation: Analyze token probabilities or stated uncertainty.
        ENTAILMENT: Check logical consistency and entailment.
            Useful for detecting logical contradictions.
            Implementation: Use NLI models to check claim relationships.

    Examples:
        Selecting detection methods:

        >>> methods = [DetectionMethod.CONSISTENCY, DetectionMethod.FACTUAL]
        >>> for method in methods:
        ...     print(f"Using: {method.value}")
        Using: consistency
        Using: factual

        Method-specific detection configuration:

        >>> config = {
        ...     DetectionMethod.CONSISTENCY: {'min_responses': 3},
        ...     DetectionMethod.FACTUAL: {'knowledge_base': kb},
        ...     DetectionMethod.ATTRIBUTION: {'verify_urls': True}
        ... }

        Checking method capabilities:

        >>> method = DetectionMethod.CONSISTENCY
        >>> method.value
        'consistency'
        >>> method == DetectionMethod.CONSISTENCY
        True

        Iterating over all methods:

        >>> all_methods = list(DetectionMethod)
        >>> len(all_methods)
        5
    """

    CONSISTENCY = "consistency"  # Check self-consistency
    FACTUAL = "factual"  # Check against known facts
    ATTRIBUTION = "attribution"  # Check source claims
    CONFIDENCE = "confidence"  # Use model confidence
    ENTAILMENT = "entailment"  # Check logical entailment


@dataclass
class HallucinationFlag:
    """
    Represents a single detected potential hallucination in text.

    This dataclass captures all relevant information about a detected
    hallucination, including its type, severity, location in the text,
    and supporting evidence for why it was flagged.

    Attributes:
        hallucination_type: The category of hallucination (from HallucinationType).
        severity: How serious the hallucination is (from SeverityLevel).
        text_span: The actual text that was flagged as a potential hallucination.
        start_pos: Character position where the flagged text starts.
        end_pos: Character position where the flagged text ends.
        confidence: Confidence score (0.0-1.0) that this is a hallucination.
            Higher values indicate more certainty.
        explanation: Human-readable explanation of why this was flagged.
        evidence: List of supporting evidence or contradicting facts.
        metadata: Additional information about the detection.

    Examples:
        Creating a basic hallucination flag:

        >>> flag = HallucinationFlag(
        ...     hallucination_type=HallucinationType.FACTUAL_ERROR,
        ...     severity=SeverityLevel.HIGH,
        ...     text_span="The Great Wall of China is visible from space",
        ...     start_pos=0,
        ...     end_pos=45,
        ...     confidence=0.92,
        ...     explanation="This is a common myth; the wall is not visible from space"
        ... )
        >>> flag.confidence
        0.92

        Creating a flag with evidence:

        >>> flag = HallucinationFlag(
        ...     hallucination_type=HallucinationType.ATTRIBUTION_ERROR,
        ...     severity=SeverityLevel.MEDIUM,
        ...     text_span="According to NASA, the moon is made of cheese",
        ...     start_pos=100,
        ...     end_pos=145,
        ...     confidence=0.85,
        ...     explanation="NASA has never made this claim",
        ...     evidence=["NASA lunar composition data shows rock and regolith"]
        ... )
        >>> len(flag.evidence)
        1

        Converting to dictionary for serialization:

        >>> flag_dict = flag.to_dict()
        >>> flag_dict['type']
        'attribution_error'
        >>> 'position' in flag_dict
        True

        Using flags in reports:

        >>> report = detect_hallucinations(text)
        >>> for flag in report.flags:
        ...     print(f"{flag.severity.value}: {flag.text_span[:50]}...")
    """

    hallucination_type: HallucinationType
    severity: SeverityLevel
    text_span: str
    start_pos: int
    end_pos: int
    confidence: float  # Confidence that this is a hallucination
    explanation: str
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the hallucination flag to a dictionary representation.

        Creates a serializable dictionary containing all flag information,
        suitable for JSON export or API responses.

        Returns:
            dict[str, Any]: Dictionary containing:
                - type: Hallucination type as string
                - severity: Severity level as string
                - text_span: The flagged text
                - position: Dict with start and end positions
                - confidence: Rounded confidence score (4 decimal places)
                - explanation: Why this was flagged
                - evidence: List of supporting evidence
                - metadata: Additional information

        Examples:
            Basic conversion:

            >>> flag = HallucinationFlag(
            ...     hallucination_type=HallucinationType.EXAGGERATION,
            ...     severity=SeverityLevel.LOW,
            ...     text_span="definitely",
            ...     start_pos=10, end_pos=20,
            ...     confidence=0.4,
            ...     explanation="Overconfident language"
            ... )
            >>> d = flag.to_dict()
            >>> d['type']
            'exaggeration'

            JSON serialization:

            >>> import json
            >>> json_str = json.dumps(flag.to_dict())
            >>> 'position' in json_str
            True

            Accessing position info:

            >>> d = flag.to_dict()
            >>> d['position']['start']
            10

            Checking confidence precision:

            >>> flag.confidence = 0.123456789
            >>> d = flag.to_dict()
            >>> d['confidence']
            0.1235
        """
        return {
            "type": self.hallucination_type.value,
            "severity": self.severity.value,
            "text_span": self.text_span,
            "position": {"start": self.start_pos, "end": self.end_pos},
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass
class ConsistencyCheck:
    """
    Result of consistency checking across multiple LLM responses.

    When an LLM is asked the same question multiple times, inconsistent
    answers often indicate hallucination. This dataclass captures the
    results of comparing multiple responses to identify consistent
    and inconsistent claims.

    Attributes:
        responses: List of response texts that were compared.
        consistent_claims: Claims that appear consistently across responses.
        inconsistent_claims: Tuples of (claim1, claim2, disagreement_score)
            where claims conflict. Higher disagreement indicates more conflict.
        overall_consistency: Score from 0.0-1.0 indicating overall consistency.
            1.0 means all claims are consistent.
        metadata: Additional information about the check.

    Examples:
        Basic consistency check:

        >>> from insideLLMs.evaluation.hallucination import check_consistency
        >>> responses = [
        ...     "Python was created by Guido van Rossum in 1991.",
        ...     "Guido van Rossum created Python in 1991.",
        ...     "Python was developed by Guido van Rossum."
        ... ]
        >>> result = check_consistency(responses)
        >>> result.is_consistent
        True
        >>> result.overall_consistency > 0.8
        True

        Detecting inconsistencies:

        >>> responses = [
        ...     "The company was founded in 2010.",
        ...     "The company was established in 2015.",
        ...     "The company started in 2010."
        ... ]
        >>> result = check_consistency(responses)
        >>> len(result.inconsistent_claims) > 0
        True

        Analyzing inconsistent claims:

        >>> for claim1, claim2, disagreement in result.inconsistent_claims:
        ...     print(f"Conflict: '{claim1[:30]}...' vs '{claim2[:30]}...'")
        ...     print(f"Disagreement: {disagreement:.2f}")

        Using with hallucination detector:

        >>> detector = ComprehensiveHallucinationDetector()
        >>> report = detector.detect(text, reference_texts=other_responses)
        >>> if report.consistency_check and not report.consistency_check.is_consistent:
        ...     print("Warning: Inconsistent responses detected")
    """

    responses: list[str]
    consistent_claims: list[str]
    inconsistent_claims: list[tuple[str, str, float]]  # (claim1, claim2, disagreement)
    overall_consistency: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_consistent(self) -> bool:
        """
        Check if responses are mostly consistent.

        A response set is considered consistent if the overall consistency
        score is at least 0.8 (80%).

        Returns:
            bool: True if overall_consistency >= 0.8, False otherwise.

        Examples:
            Checking consistency:

            >>> result = check_consistency(responses)
            >>> if result.is_consistent:
            ...     print("Responses are consistent")
            ... else:
            ...     print(f"Found {len(result.inconsistent_claims)} conflicts")

            Using in conditionals:

            >>> result.overall_consistency = 0.85
            >>> result.is_consistent
            True
            >>> result.overall_consistency = 0.75
            >>> result.is_consistent
            False

            Threshold behavior:

            >>> result.overall_consistency = 0.8
            >>> result.is_consistent  # Exactly at threshold
            True

            Combining with other checks:

            >>> if result.is_consistent and len(result.consistent_claims) > 5:
            ...     print("Strong agreement on multiple claims")
        """
        return self.overall_consistency >= 0.8

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the consistency check to a dictionary representation.

        Creates a serializable dictionary containing all consistency
        information, suitable for JSON export or API responses.

        Returns:
            dict[str, Any]: Dictionary containing:
                - n_responses: Number of responses compared
                - consistent_claims: List of consistent claim strings
                - inconsistent_claims: List of dicts with claim1, claim2, disagreement
                - overall_consistency: Rounded consistency score
                - is_consistent: Boolean consistency determination
                - metadata: Additional information

        Examples:
            Basic conversion:

            >>> result = check_consistency(responses)
            >>> d = result.to_dict()
            >>> d['n_responses']
            3

            JSON serialization:

            >>> import json
            >>> json_str = json.dumps(result.to_dict())
            >>> 'overall_consistency' in json_str
            True

            Accessing inconsistent claims:

            >>> d = result.to_dict()
            >>> for claim in d['inconsistent_claims']:
            ...     print(f"Disagreement: {claim['disagreement']}")

            Checking response count:

            >>> d['n_responses'] == len(result.responses)
            True
        """
        return {
            "n_responses": len(self.responses),
            "consistent_claims": self.consistent_claims,
            "inconsistent_claims": [
                {"claim1": c1, "claim2": c2, "disagreement": round(d, 4)}
                for c1, c2, d in self.inconsistent_claims
            ],
            "overall_consistency": round(self.overall_consistency, 4),
            "is_consistent": self.is_consistent,
            "metadata": self.metadata,
        }


@dataclass
class FactualityScore:
    """
    Score representing the factual accuracy of analyzed text.

    This dataclass captures the results of fact-checking claims extracted
    from text against a knowledge base or fact-checking function. It provides
    both aggregate scores and detailed per-claim information.

    Attributes:
        score: Overall factuality score from 0.0 to 1.0, where 1.0 indicates
            no contradicted claims were found.
        verifiable_claims: Total number of claims that could be fact-checked.
        verified_claims: Number of claims confirmed as factual.
        unverified_claims: Number of claims that couldn't be verified.
        contradicted_claims: Number of claims found to be false.
        claim_details: List of dicts with detailed info for each claim,
            including the claim text and verification status.
        metadata: Additional information about the fact-checking process.

    Examples:
        Basic factuality checking:

        >>> from insideLLMs.evaluation.hallucination import check_factuality
        >>> text = "Paris is the capital of France. Berlin is the capital of Germany."
        >>> knowledge = {
        ...     "france_capital": "Paris is the capital of France",
        ...     "germany_capital": "Berlin is the capital of Germany"
        ... }
        >>> result = check_factuality(text, knowledge_base=knowledge)
        >>> result.accuracy
        1.0

        Analyzing claim breakdown:

        >>> print(f"Verified: {result.verified_claims}/{result.verifiable_claims}")
        >>> print(f"Contradicted: {result.contradicted_claims}")
        >>> for detail in result.claim_details:
        ...     print(f"{detail['status']}: {detail['claim'][:50]}")

        Detecting factual errors:

        >>> text = "The Eiffel Tower is in London."
        >>> result = check_factuality(text, knowledge_base=kb)
        >>> if result.contradicted_claims > 0:
        ...     print("Found factual errors!")

        Using with comprehensive detector:

        >>> detector = ComprehensiveHallucinationDetector(
        ...     factuality_checker=FactualityChecker(knowledge_base=kb)
        ... )
        >>> report = detector.detect(text, check_factuality=True)
        >>> if report.factuality_score and report.factuality_score.score < 0.9:
        ...     print("Low factuality score - review claims")
    """

    score: float  # 0-1, 1 = fully factual
    verifiable_claims: int
    verified_claims: int
    unverified_claims: int
    contradicted_claims: int
    claim_details: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """
        Calculate the factual accuracy rate.

        Computes the proportion of verifiable claims that were successfully
        verified. Returns 1.0 if there are no verifiable claims (assumes
        accuracy when nothing can be checked).

        Returns:
            float: Ratio of verified_claims to verifiable_claims (0.0 to 1.0).

        Examples:
            Calculating accuracy:

            >>> score = FactualityScore(
            ...     score=0.9, verifiable_claims=10,
            ...     verified_claims=8, unverified_claims=1,
            ...     contradicted_claims=1
            ... )
            >>> score.accuracy
            0.8

            No verifiable claims:

            >>> score = FactualityScore(
            ...     score=1.0, verifiable_claims=0,
            ...     verified_claims=0, unverified_claims=0,
            ...     contradicted_claims=0
            ... )
            >>> score.accuracy  # Defaults to 1.0
            1.0

            Perfect accuracy:

            >>> score.verified_claims = score.verifiable_claims = 5
            >>> score.accuracy
            1.0

            Using for thresholds:

            >>> if score.accuracy < 0.7:
            ...     print("Warning: Low factual accuracy")
        """
        if self.verifiable_claims == 0:
            return 1.0
        return self.verified_claims / self.verifiable_claims

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the factuality score to a dictionary representation.

        Creates a serializable dictionary containing all factuality
        information, suitable for JSON export or API responses.
        Limits claim_details to first 10 entries to prevent large outputs.

        Returns:
            dict[str, Any]: Dictionary containing:
                - score: Rounded factuality score
                - verifiable_claims: Number of checkable claims
                - verified_claims: Number confirmed factual
                - unverified_claims: Number not verifiable
                - contradicted_claims: Number found false
                - accuracy: Calculated accuracy rate
                - claim_details: First 10 claim detail dicts
                - metadata: Additional information

        Examples:
            Basic conversion:

            >>> result = check_factuality(text, knowledge_base=kb)
            >>> d = result.to_dict()
            >>> d['verifiable_claims']
            5

            JSON serialization:

            >>> import json
            >>> json_str = json.dumps(result.to_dict())
            >>> 'accuracy' in json_str
            True

            Checking claim details limit:

            >>> result.claim_details = [{'claim': f'claim_{i}'} for i in range(20)]
            >>> d = result.to_dict()
            >>> len(d['claim_details'])  # Limited to 10
            10

            Accessing accuracy:

            >>> d = result.to_dict()
            >>> d['accuracy'] == round(result.accuracy, 4)
            True
        """
        return {
            "score": round(self.score, 4),
            "verifiable_claims": self.verifiable_claims,
            "verified_claims": self.verified_claims,
            "unverified_claims": self.unverified_claims,
            "contradicted_claims": self.contradicted_claims,
            "accuracy": round(self.accuracy, 4),
            "claim_details": self.claim_details[:10],  # First 10
            "metadata": self.metadata,
        }


@dataclass
class HallucinationReport:
    """
    Comprehensive report from hallucination detection analysis.

    This dataclass aggregates results from multiple detection methods
    (pattern-based, consistency checking, factuality checking) into a
    single comprehensive report with actionable recommendations.

    Attributes:
        text: The original text that was analyzed.
        flags: List of HallucinationFlag objects for each detected issue.
        overall_score: Aggregate score from 0.0-1.0, where 1.0 means no
            hallucinations were detected. Lower scores indicate more issues.
        consistency_check: Optional ConsistencyCheck results if multiple
            responses were compared.
        factuality_score: Optional FactualityScore if fact-checking was performed.
        recommendations: List of actionable recommendations based on findings.
        metadata: Additional information about the detection process.

    Examples:
        Basic hallucination detection:

        >>> from insideLLMs.evaluation.hallucination import detect_hallucinations
        >>> text = "According to a 2023 study, 99.9% of people agree."
        >>> report = detect_hallucinations(text)
        >>> print(f"Score: {report.overall_score:.2f}")
        >>> print(f"Issues found: {len(report.flags)}")

        Checking for critical issues:

        >>> if report.critical_flags:
        ...     print("Critical issues found!")
        ...     for flag in report.critical_flags:
        ...         print(f"  - {flag.text_span}")

        Getting high-confidence flags:

        >>> confident_issues = report.high_confidence_flags
        >>> for flag in confident_issues:
        ...     print(f"{flag.confidence:.0%} confident: {flag.explanation}")

        Using recommendations:

        >>> for rec in report.recommendations:
        ...     print(f"Recommendation: {rec}")

        Converting to JSON:

        >>> import json
        >>> json_report = json.dumps(report.to_dict(), indent=2)
        >>> print(json_report)
    """

    text: str
    flags: list[HallucinationFlag]
    overall_score: float  # 0-1, 1 = no hallucinations detected
    consistency_check: Optional[ConsistencyCheck] = None
    factuality_score: Optional[FactualityScore] = None
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_hallucinations(self) -> bool:
        """
        Check if any potential hallucinations were detected.

        Returns True if at least one HallucinationFlag was generated,
        regardless of severity or confidence level.

        Returns:
            bool: True if flags list is non-empty, False otherwise.

        Examples:
            Checking for issues:

            >>> report = detect_hallucinations(text)
            >>> if report.has_hallucinations:
            ...     print(f"Found {len(report.flags)} potential issues")
            ... else:
            ...     print("No issues detected")

            Using in conditionals:

            >>> clean_text = "The sky is blue."
            >>> report = detect_hallucinations(clean_text)
            >>> report.has_hallucinations
            False

            Problematic text:

            >>> bad_text = "Everyone definitely knows this is always true."
            >>> report = detect_hallucinations(bad_text)
            >>> report.has_hallucinations
            True

            Combining checks:

            >>> if report.has_hallucinations and report.overall_score < 0.5:
            ...     print("Significant hallucination concerns")
        """
        return len(self.flags) > 0

    @property
    def critical_flags(self) -> list[HallucinationFlag]:
        """
        Get only critical severity flags.

        Filters the flags list to return only those with CRITICAL severity,
        which represent the most serious issues requiring immediate attention.

        Returns:
            list[HallucinationFlag]: Flags with severity == SeverityLevel.CRITICAL.

        Examples:
            Getting critical issues:

            >>> report = detect_hallucinations(dangerous_text)
            >>> critical = report.critical_flags
            >>> print(f"Critical issues: {len(critical)}")

            Iterating critical flags:

            >>> for flag in report.critical_flags:
            ...     print(f"CRITICAL: {flag.text_span}")
            ...     print(f"  Reason: {flag.explanation}")

            Checking for critical issues:

            >>> if report.critical_flags:
            ...     raise ValueError("Critical hallucinations detected!")

            Priority handling:

            >>> if len(report.critical_flags) > 0:
            ...     # Handle critical issues first
            ...     for flag in report.critical_flags:
            ...         handle_critical(flag)
        """
        return [f for f in self.flags if f.severity == SeverityLevel.CRITICAL]

    @property
    def high_confidence_flags(self) -> list[HallucinationFlag]:
        """
        Get flags with high confidence (> 0.8).

        Filters flags to return only those where the detector is highly
        confident (>80%) that they represent actual hallucinations.

        Returns:
            list[HallucinationFlag]: Flags with confidence > 0.8.

        Examples:
            Getting confident detections:

            >>> report = detect_hallucinations(text)
            >>> confident = report.high_confidence_flags
            >>> print(f"High confidence issues: {len(confident)}")

            Focusing on reliable flags:

            >>> for flag in report.high_confidence_flags:
            ...     print(f"{flag.confidence:.0%}: {flag.text_span}")

            Threshold comparison:

            >>> report.flags[0].confidence = 0.85
            >>> len(report.high_confidence_flags) >= 1
            True
            >>> report.flags[0].confidence = 0.75
            >>> report.flags[0] in report.high_confidence_flags
            False

            Combining with severity:

            >>> serious_issues = [
            ...     f for f in report.high_confidence_flags
            ...     if f.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
            ... ]
        """
        return [f for f in self.flags if f.confidence > 0.8]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the report to a dictionary representation.

        Creates a comprehensive serializable dictionary containing all
        report information, suitable for JSON export, API responses,
        or logging.

        Returns:
            dict[str, Any]: Dictionary containing:
                - text_preview: First 200 chars of analyzed text
                - has_hallucinations: Boolean indicating if issues found
                - n_flags: Total number of flags
                - overall_score: Rounded overall score
                - flags: List of flag dictionaries
                - critical_count: Number of critical flags
                - consistency: ConsistencyCheck dict or None
                - factuality: FactualityScore dict or None
                - recommendations: List of recommendation strings
                - metadata: Additional information

        Examples:
            Basic conversion:

            >>> report = detect_hallucinations(text)
            >>> d = report.to_dict()
            >>> d['has_hallucinations']
            True

            JSON serialization:

            >>> import json
            >>> json_str = json.dumps(report.to_dict(), indent=2)
            >>> print(json_str)

            Accessing nested data:

            >>> d = report.to_dict()
            >>> for flag in d['flags']:
            ...     print(f"{flag['type']}: {flag['text_span']}")

            Checking preview truncation:

            >>> long_text = "A" * 500
            >>> report = detect_hallucinations(long_text)
            >>> d = report.to_dict()
            >>> d['text_preview'].endswith('...')
            True
            >>> len(d['text_preview'])
            203
        """
        return {
            "text_preview": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "has_hallucinations": self.has_hallucinations,
            "n_flags": len(self.flags),
            "overall_score": round(self.overall_score, 4),
            "flags": [f.to_dict() for f in self.flags],
            "critical_count": len(self.critical_flags),
            "consistency": self.consistency_check.to_dict() if self.consistency_check else None,
            "factuality": self.factuality_score.to_dict() if self.factuality_score else None,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


class PatternBasedDetector:
    """
    Detects potential hallucinations using regex pattern matching.

    This detector uses predefined patterns to identify language that
    commonly indicates potential hallucinations, such as overconfident
    assertions, fabricated-sounding citations, and specific statistics
    that may be invented.

    The detector looks for several categories of indicators:
        - Hedge patterns: Language indicating uncertainty (not flagged as hallucinations)
        - Overconfidence patterns: Absolute statements that may be unverified
        - Fabrication indicators: Specific claims that commonly appear in hallucinations
        - Contradiction patterns: Opposing statements (for internal consistency checks)

    Attributes:
        hedge_patterns: Patterns for hedging language (uncertainty indicators).
        overconfidence_patterns: Patterns for overconfident assertions.
        fabrication_indicators: Patterns for potentially fabricated specifics.
        contradiction_patterns: Tuples of opposing patterns.

    Examples:
        Basic pattern detection:

        >>> detector = PatternBasedDetector()
        >>> text = "Everyone definitely knows this is always true."
        >>> flags = detector.detect(text)
        >>> len(flags) > 0
        True
        >>> flags[0].hallucination_type
        <HallucinationType.EXAGGERATION: 'exaggeration'>

        Detecting fabrication indicators:

        >>> text = "According to a study, 87.3% of users prefer this product."
        >>> flags = detector.detect(text)
        >>> has_unsupported = any(
        ...     f.hallucination_type == HallucinationType.UNSUPPORTED_CLAIM
        ...     for f in flags
        ... )
        >>> has_unsupported
        True

        Custom usage in pipeline:

        >>> detector = PatternBasedDetector()
        >>> for response in llm_responses:
        ...     flags = detector.detect(response)
        ...     if flags:
        ...         print(f"Found {len(flags)} potential issues")

        Integrating with comprehensive detector:

        >>> custom_detector = PatternBasedDetector()
        >>> full_detector = ComprehensiveHallucinationDetector(
        ...     pattern_detector=custom_detector
        ... )
    """

    def __init__(self):
        """
        Initialize the pattern-based detector with default patterns.

        Sets up regex patterns for detecting various hallucination indicators:
            - Hedge patterns for uncertainty language
            - Overconfidence patterns for absolute statements
            - Fabrication indicators for specific claims
            - Contradiction patterns for opposing statements

        Examples:
            Default initialization:

            >>> detector = PatternBasedDetector()
            >>> len(detector.overconfidence_patterns)
            2

            Accessing patterns:

            >>> detector = PatternBasedDetector()
            >>> 'definitely' in detector.overconfidence_patterns[0]
            True

            Modifying patterns:

            >>> detector = PatternBasedDetector()
            >>> detector.fabrication_indicators.append(r"\b(sources say)\b")
            >>> len(detector.fabrication_indicators)
            6

            Pattern categories:

            >>> detector = PatternBasedDetector()
            >>> print(f"Hedge patterns: {len(detector.hedge_patterns)}")
            >>> print(f"Overconfidence patterns: {len(detector.overconfidence_patterns)}")
            >>> print(f"Fabrication indicators: {len(detector.fabrication_indicators)}")
        """
        # Patterns that might indicate hallucinations
        self.hedge_patterns = [
            r"\b(i think|i believe|i assume|probably|possibly|maybe|might)\b",
            r"\b(it seems|it appears|apparently|supposedly)\b",
            r"\b(i'm not sure|i don't know|uncertain)\b",
        ]

        self.overconfidence_patterns = [
            r"\b(definitely|certainly|absolutely|always|never)\b",
            r"\b(everyone knows|it's obvious|clearly|undoubtedly)\b",
        ]

        self.fabrication_indicators = [
            r"\b(according to a study|research shows|studies have shown)\b",
            r"\b(in \d{4}|on [A-Z][a-z]+ \d{1,2}, \d{4})\b",  # Specific dates
            r"\b(Dr\.|Professor|CEO|President)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Names with titles
            r"\bhttps?://[^\s]+\b",  # URLs
            r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b",  # Specific statistics
        ]

        self.contradiction_patterns = [
            (r"\bis\b", r"\bis not\b"),
            (r"\bwas\b", r"\bwas not\b"),
            (r"\bcan\b", r"\bcannot\b"),
            (r"\bwill\b", r"\bwill not\b"),
        ]

    def detect(self, text: str) -> list[HallucinationFlag]:
        """
        Detect potential hallucinations in text using pattern matching.

        Scans the input text for patterns that commonly indicate potential
        hallucinations, including overconfident language and fabrication
        indicators (specific statistics, citations, dates, etc.).

        Args:
            text: The text to analyze for hallucination indicators.

        Returns:
            list[HallucinationFlag]: List of flags for detected patterns.
                Each flag includes the hallucination type, severity,
                matched text span, position, and explanation.

        Examples:
            Detecting overconfident language:

            >>> detector = PatternBasedDetector()
            >>> text = "This is definitely the best solution."
            >>> flags = detector.detect(text)
            >>> len(flags)
            1
            >>> flags[0].text_span
            'definitely'

            Detecting multiple indicators:

            >>> text = "According to a study, 95% of people always agree."
            >>> flags = detector.detect(text)
            >>> types = [f.hallucination_type for f in flags]
            >>> HallucinationType.UNSUPPORTED_CLAIM in types
            True

            Clean text with no flags:

            >>> text = "The weather today is pleasant."
            >>> flags = detector.detect(text)
            >>> len(flags)
            0

            Accessing flag details:

            >>> text = "Dr. John Smith confirmed that everyone knows this."
            >>> flags = detector.detect(text)
            >>> for flag in flags:
            ...     print(f"{flag.severity.value}: {flag.text_span}")
            ...     print(f"  Position: {flag.start_pos}-{flag.end_pos}")
        """
        flags = []
        text_lower = text.lower()

        # Check for overconfidence
        for pattern in self.overconfidence_patterns:
            for match in re.finditer(pattern, text_lower):
                flags.append(
                    HallucinationFlag(
                        hallucination_type=HallucinationType.EXAGGERATION,
                        severity=SeverityLevel.LOW,
                        text_span=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.4,
                        explanation="Overconfident language may indicate unverified claims",
                    )
                )

        # Check for fabrication indicators
        for pattern in self.fabrication_indicators:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                flags.append(
                    HallucinationFlag(
                        hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
                        severity=SeverityLevel.MEDIUM,
                        text_span=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.5,
                        explanation="Specific claims may need verification",
                    )
                )

        return flags


class FactConsistencyChecker:
    """
    Checks consistency of claims across multiple LLM responses.

    This checker helps detect hallucinations by comparing multiple responses
    to the same question. Inconsistent answers across responses often indicate
    that the model is generating fabricated information rather than retrieving
    factual knowledge.

    The checker extracts claims (sentences) from each response, groups similar
    claims together, and identifies where responses contradict each other.

    Attributes:
        similarity_fn: Function to compute similarity between two strings.
            Should return float from 0.0 (different) to 1.0 (identical).

    Examples:
        Basic consistency checking:

        >>> checker = FactConsistencyChecker()
        >>> responses = [
        ...     "Python was created by Guido van Rossum.",
        ...     "Guido van Rossum created Python.",
        ...     "Python was developed by Guido van Rossum."
        ... ]
        >>> result = checker.check(responses)
        >>> result.is_consistent
        True

        Detecting inconsistencies:

        >>> responses = [
        ...     "The company was founded in 2010 by John Smith.",
        ...     "The company was founded in 2015 by Jane Doe."
        ... ]
        >>> result = checker.check(responses)
        >>> result.is_consistent
        False
        >>> len(result.inconsistent_claims) > 0
        True

        Custom similarity function:

        >>> def my_similarity(a: str, b: str) -> float:
        ...     # Custom implementation
        ...     return 0.5
        >>> checker = FactConsistencyChecker(similarity_fn=my_similarity)

        Using with comprehensive detector:

        >>> checker = FactConsistencyChecker()
        >>> detector = ComprehensiveHallucinationDetector(
        ...     consistency_checker=checker
        ... )
        >>> report = detector.detect(text, reference_texts=other_responses)
    """

    def __init__(
        self,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """
        Initialize the consistency checker.

        Args:
            similarity_fn: Optional custom function for comparing text similarity.
                Must accept two strings and return a float between 0.0 and 1.0.
                Defaults to word_overlap_similarity if not provided.

        Examples:
            Default initialization:

            >>> checker = FactConsistencyChecker()
            >>> checker.similarity_fn is not None
            True

            Custom similarity function:

            >>> def jaccard_similarity(a: str, b: str) -> float:
            ...     set_a = set(a.lower().split())
            ...     set_b = set(b.lower().split())
            ...     intersection = len(set_a & set_b)
            ...     union = len(set_a | set_b)
            ...     return intersection / union if union > 0 else 0.0
            >>> checker = FactConsistencyChecker(similarity_fn=jaccard_similarity)

            Using embedding-based similarity:

            >>> def embedding_similarity(a: str, b: str) -> float:
            ...     # Would use sentence embeddings in practice
            ...     return 0.8  # Placeholder
            >>> checker = FactConsistencyChecker(similarity_fn=embedding_similarity)

            Verifying the function is set:

            >>> checker = FactConsistencyChecker()
            >>> callable(checker.similarity_fn)
            True
        """
        self.similarity_fn = similarity_fn or word_overlap_similarity

    def _extract_claims(self, text: str) -> list[str]:
        """
        Extract claims from text using sentence-based splitting.

        Splits text into sentences and filters out questions and short
        fragments to extract meaningful claims that can be compared.

        Args:
            text: The text to extract claims from.

        Returns:
            list[str]: List of claim strings (sentences > 20 chars, non-questions).

        Examples:
            Basic extraction:

            >>> checker = FactConsistencyChecker()
            >>> text = "Python is a programming language. It was created in 1991."
            >>> claims = checker._extract_claims(text)
            >>> len(claims)
            2

            Filtering questions:

            >>> text = "What is Python? Python is a programming language."
            >>> claims = checker._extract_claims(text)
            >>> len(claims)  # Question is filtered out
            1

            Filtering short fragments:

            >>> text = "Yes. No. Python is a great programming language."
            >>> claims = checker._extract_claims(text)
            >>> len(claims)  # Short sentences filtered
            1

            Multiple sentence types:

            >>> text = "Python is great! It was made in 1991. Really?"
            >>> claims = checker._extract_claims(text)
            >>> 'Python is great' in claims[0]
            True
        """
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20 and not sent.endswith("?"):  # Filter questions and short fragments
                claims.append(sent)
        return claims

    def check(self, responses: list[str]) -> ConsistencyCheck:
        """
        Check consistency across multiple responses.

        Analyzes multiple responses to identify claims that are consistent
        (appear similarly across responses) and claims that are inconsistent
        (similar topics but contradictory information).

        Args:
            responses: List of response texts to compare.

        Returns:
            ConsistencyCheck: Results containing consistent claims, inconsistent
                claims, and an overall consistency score.

        Examples:
            Checking consistent responses:

            >>> checker = FactConsistencyChecker()
            >>> responses = [
            ...     "The Eiffel Tower is in Paris, France.",
            ...     "Paris, France is home to the Eiffel Tower.",
            ...     "The Eiffel Tower can be found in Paris."
            ... ]
            >>> result = checker.check(responses)
            >>> result.overall_consistency > 0.5
            True

            Checking inconsistent responses:

            >>> responses = [
            ...     "The movie was released in 2019.",
            ...     "The movie came out in 2021.",
            ...     "It premiered in 2019."
            ... ]
            >>> result = checker.check(responses)
            >>> len(result.inconsistent_claims) > 0
            True

            Single response handling:

            >>> result = checker.check(["Just one response."])
            >>> result.overall_consistency
            1.0

            Accessing results:

            >>> result = checker.check(responses)
            >>> print(f"Consistent: {len(result.consistent_claims)}")
            >>> print(f"Inconsistent: {len(result.inconsistent_claims)}")
            >>> print(f"Overall: {result.overall_consistency:.2f}")
        """
        if len(responses) < 2:
            return ConsistencyCheck(
                responses=responses,
                consistent_claims=[],
                inconsistent_claims=[],
                overall_consistency=1.0,
            )

        # Extract claims from each response
        all_claims = []
        for response in responses:
            claims = self._extract_claims(response)
            all_claims.extend([(response, claim) for claim in claims])

        # Find consistent and inconsistent claims
        consistent = []
        inconsistent = []

        # Compare claims across responses
        claim_groups: dict[str, list[str]] = {}
        for response, claim in all_claims:
            # Find similar claims
            found_group = False
            for key in list(claim_groups.keys()):
                if self.similarity_fn(claim, key) > 0.7:
                    claim_groups[key].append(claim)
                    found_group = True
                    break
            if not found_group:
                claim_groups[claim] = [claim]

        # Claims that appear in multiple responses are consistent
        for key, claims in claim_groups.items():
            if len(claims) >= 2:
                consistent.append(key)
            else:
                # Check for contradictions
                for other_key in claim_groups:
                    if other_key != key:
                        sim = self.similarity_fn(key, other_key)
                        if 0.3 < sim < 0.7:  # Similar but different = potential inconsistency
                            inconsistent.append((key, other_key, 1.0 - sim))

        # Calculate overall consistency
        total_claims = len(claim_groups)
        overall = 1.0 if total_claims == 0 else len(consistent) / total_claims

        return ConsistencyCheck(
            responses=responses,
            consistent_claims=consistent[:10],  # Limit to 10
            inconsistent_claims=inconsistent[:10],
            overall_consistency=overall,
        )


class FactualityChecker:
    """
    Checks factual accuracy of claims against a knowledge base or fact-checker.

    This checker extracts verifiable claims from text using pattern matching,
    then verifies them either against a provided knowledge base or using a
    custom fact-checking function.

    The checker identifies common factual claim patterns such as:
        - "[Entity] is [description]"
        - "[Entity] was founded/created in [year]"
        - "The [attribute] of [Entity] is [value]"

    Attributes:
        fact_checker: Optional callable that takes a claim string and returns
            a tuple of (is_true: bool, confidence: float).
        knowledge_base: Dict mapping fact IDs to fact strings for verification.
        factual_patterns: List of regex patterns for extracting factual claims.

    Examples:
        Basic factuality checking with knowledge base:

        >>> knowledge = {
        ...     "paris": "Paris is the capital of France",
        ...     "berlin": "Berlin is the capital of Germany"
        ... }
        >>> checker = FactualityChecker(knowledge_base=knowledge)
        >>> result = checker.check("Paris is the capital of France.")
        >>> result.verified_claims >= 1
        True

        Using a custom fact-checker function:

        >>> def my_fact_checker(claim: str) -> tuple[bool, float]:
        ...     # Custom fact verification logic
        ...     if "Paris" in claim and "France" in claim:
        ...         return (True, 0.95)
        ...     return (False, 0.5)
        >>> checker = FactualityChecker(fact_checker=my_fact_checker)
        >>> result = checker.check("Paris is in France.")

        Checking multiple claims:

        >>> text = "Python was created in 1991. Java was founded in 1995."
        >>> checker = FactualityChecker(knowledge_base=kb)
        >>> result = checker.check(text)
        >>> print(f"Verified: {result.verified_claims}")
        >>> print(f"Contradicted: {result.contradicted_claims}")

        Integrating with comprehensive detector:

        >>> checker = FactualityChecker(knowledge_base=my_kb)
        >>> detector = ComprehensiveHallucinationDetector(
        ...     factuality_checker=checker
        ... )
    """

    def __init__(
        self,
        fact_checker: Optional[Callable[[str], tuple[bool, float]]] = None,
        knowledge_base: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the factuality checker.

        Args:
            fact_checker: Optional function for verifying claims. Should accept
                a claim string and return (is_true, confidence) tuple.
            knowledge_base: Optional dictionary of known facts for verification.
                Keys are fact identifiers, values are fact strings.

        Examples:
            Default initialization:

            >>> checker = FactualityChecker()
            >>> checker.knowledge_base
            {}

            With knowledge base:

            >>> kb = {"fact1": "The sky is blue", "fact2": "Water is wet"}
            >>> checker = FactualityChecker(knowledge_base=kb)
            >>> len(checker.knowledge_base)
            2

            With custom fact checker:

            >>> def api_fact_check(claim: str) -> tuple[bool, float]:
            ...     # Call external API
            ...     return (True, 0.9)
            >>> checker = FactualityChecker(fact_checker=api_fact_check)

            Combined initialization:

            >>> checker = FactualityChecker(
            ...     fact_checker=my_checker,
            ...     knowledge_base=my_kb
            ... )
        """
        self.fact_checker = fact_checker
        self.knowledge_base = knowledge_base or {}

        # Common factual patterns
        self.factual_patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:the\s+)?([^.]+)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+(?:born|founded|created|established)\s+(?:in|on)\s+(\d{4})",
            r"The\s+([a-z]+)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+([^.]+)",
        ]

    def _extract_factual_claims(self, text: str) -> list[dict[str, Any]]:
        """
        Extract verifiable factual claims from text.

        Uses regex patterns to identify claims that follow common factual
        statement structures.

        Args:
            text: The text to extract claims from.

        Returns:
            list[dict[str, Any]]: List of claim dictionaries, each containing:
                - text: The full matched claim text
                - start: Start position in original text
                - end: End position in original text
                - groups: Regex capture groups

        Examples:
            Extracting entity claims:

            >>> checker = FactualityChecker()
            >>> text = "Paris is the capital of France."
            >>> claims = checker._extract_factual_claims(text)
            >>> len(claims) >= 1
            True

            Extracting date claims:

            >>> text = "Python was created in 1991."
            >>> claims = checker._extract_factual_claims(text)
            >>> any("1991" in c["text"] for c in claims)
            True

            No claims found:

            >>> text = "hello world"
            >>> claims = checker._extract_factual_claims(text)
            >>> len(claims)
            0

            Accessing claim details:

            >>> claims = checker._extract_factual_claims("Apple was founded in 1976.")
            >>> claim = claims[0]
            >>> 'start' in claim and 'end' in claim
            True
        """
        claims = []

        for pattern in self.factual_patterns:
            for match in re.finditer(pattern, text):
                claims.append(
                    {
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "groups": match.groups(),
                    }
                )

        return claims

    def check(self, text: str) -> FactualityScore:
        """
        Check factual accuracy of text.

        Extracts verifiable claims from the text and checks them against
        either the custom fact_checker function (if provided) or the
        knowledge_base dictionary.

        Args:
            text: The text to fact-check.

        Returns:
            FactualityScore: Results containing the overall score, claim counts,
                and detailed status for each claim.

        Examples:
            Checking factual text:

            >>> kb = {"paris": "Paris is the capital of France"}
            >>> checker = FactualityChecker(knowledge_base=kb)
            >>> result = checker.check("Paris is the capital of France.")
            >>> result.score
            1.0

            Checking text with no verifiable claims:

            >>> checker = FactualityChecker()
            >>> result = checker.check("hello world")
            >>> result.verifiable_claims
            0
            >>> result.score
            1.0

            Getting claim details:

            >>> result = checker.check("Python was created in 1991.")
            >>> for detail in result.claim_details:
            ...     print(f"{detail['status']}: {detail['claim']}")

            Using with fact-checker function:

            >>> def check_fn(claim: str) -> tuple[bool, float]:
            ...     return (False, 0.9)  # Everything is false
            >>> checker = FactualityChecker(fact_checker=check_fn)
            >>> result = checker.check("Paris is in France.")
            >>> result.contradicted_claims >= 0
            True
        """
        claims = self._extract_factual_claims(text)

        if not claims:
            return FactualityScore(
                score=1.0,
                verifiable_claims=0,
                verified_claims=0,
                unverified_claims=0,
                contradicted_claims=0,
            )

        verified = 0
        unverified = 0
        contradicted = 0
        details = []

        for claim in claims:
            claim_text = claim["text"]

            if self.fact_checker:
                is_true, confidence = self.fact_checker(claim_text)
                if is_true and confidence > 0.8:
                    verified += 1
                    status = "verified"
                elif not is_true and confidence > 0.8:
                    contradicted += 1
                    status = "contradicted"
                else:
                    unverified += 1
                    status = "unverified"
            else:
                # Check against knowledge base
                found = False
                for fact in self.knowledge_base.values():
                    if claim_text.lower() in fact.lower() or fact.lower() in claim_text.lower():
                        verified += 1
                        found = True
                        status = "verified"
                        break
                if not found:
                    unverified += 1
                    status = "unverified"

            details.append(
                {
                    "claim": claim_text,
                    "status": status,
                }
            )

        total = len(claims)
        score = 1.0 - (contradicted / total) if total > 0 else 1.0

        return FactualityScore(
            score=score,
            verifiable_claims=total,
            verified_claims=verified,
            unverified_claims=unverified,
            contradicted_claims=contradicted,
            claim_details=details,
        )


class ComprehensiveHallucinationDetector:
    """
    All-in-one hallucination detection system combining multiple methods.

    This detector integrates pattern-based detection, consistency checking,
    and factuality verification to provide comprehensive hallucination
    analysis. It generates a unified report with all findings and
    actionable recommendations.

    The detection pipeline:
        1. Pattern-based detection for overconfident/fabricated language
        2. Consistency checking across multiple responses (if provided)
        3. Factuality verification against knowledge base (if configured)
        4. Score calculation and recommendation generation

    Attributes:
        pattern_detector: PatternBasedDetector for regex-based detection.
        consistency_checker: FactConsistencyChecker for cross-response checks.
        factuality_checker: FactualityChecker for fact verification.

    Examples:
        Basic usage:

        >>> detector = ComprehensiveHallucinationDetector()
        >>> text = "According to a 2023 study, everyone definitely agrees."
        >>> report = detector.detect(text)
        >>> print(f"Score: {report.overall_score:.2f}")
        >>> print(f"Issues: {len(report.flags)}")

        With consistency checking:

        >>> detector = ComprehensiveHallucinationDetector()
        >>> main_text = "Python was created in 1991."
        >>> other_responses = [
        ...     "Python was created in 1989.",
        ...     "Python was made in 1991."
        ... ]
        >>> report = detector.detect(main_text, reference_texts=other_responses)
        >>> if report.consistency_check:
        ...     print(f"Consistent: {report.consistency_check.is_consistent}")

        With custom components:

        >>> kb = {"python": "Python was created by Guido van Rossum in 1991"}
        >>> detector = ComprehensiveHallucinationDetector(
        ...     factuality_checker=FactualityChecker(knowledge_base=kb)
        ... )
        >>> report = detector.detect("Python was created in 1991.")

        Disabling specific checks:

        >>> report = detector.detect(
        ...     text,
        ...     check_consistency=False,
        ...     check_factuality=False
        ... )
    """

    def __init__(
        self,
        pattern_detector: Optional[PatternBasedDetector] = None,
        consistency_checker: Optional[FactConsistencyChecker] = None,
        factuality_checker: Optional[FactualityChecker] = None,
    ):
        """
        Initialize the comprehensive hallucination detector.

        Args:
            pattern_detector: Optional custom PatternBasedDetector instance.
                Defaults to a new PatternBasedDetector with default patterns.
            consistency_checker: Optional custom FactConsistencyChecker instance.
                Defaults to a new FactConsistencyChecker with default similarity.
            factuality_checker: Optional custom FactualityChecker instance.
                Defaults to a new FactualityChecker with empty knowledge base.

        Examples:
            Default initialization:

            >>> detector = ComprehensiveHallucinationDetector()
            >>> detector.pattern_detector is not None
            True

            With custom pattern detector:

            >>> custom_pattern = PatternBasedDetector()
            >>> custom_pattern.overconfidence_patterns.append(r"\\b(surely)\\b")
            >>> detector = ComprehensiveHallucinationDetector(
            ...     pattern_detector=custom_pattern
            ... )

            With knowledge base:

            >>> kb = {"fact1": "The Earth is round"}
            >>> fact_checker = FactualityChecker(knowledge_base=kb)
            >>> detector = ComprehensiveHallucinationDetector(
            ...     factuality_checker=fact_checker
            ... )

            Full custom configuration:

            >>> detector = ComprehensiveHallucinationDetector(
            ...     pattern_detector=my_pattern_detector,
            ...     consistency_checker=my_consistency_checker,
            ...     factuality_checker=my_factuality_checker
            ... )
        """
        self.pattern_detector = pattern_detector or PatternBasedDetector()
        self.consistency_checker = consistency_checker or FactConsistencyChecker()
        self.factuality_checker = factuality_checker or FactualityChecker()

    def detect(
        self,
        text: str,
        reference_texts: Optional[list[str]] = None,
        check_consistency: bool = True,
        check_factuality: bool = True,
    ) -> HallucinationReport:
        """
        Detect potential hallucinations in text using all available methods.

        Runs the complete detection pipeline: pattern matching, consistency
        checking (if reference texts provided), and factuality verification.
        Aggregates all findings into a comprehensive report.

        Args:
            text: The primary text to analyze for hallucinations.
            reference_texts: Optional list of other responses to compare for
                consistency checking. If provided and check_consistency=True,
                will compare claims across all texts.
            check_consistency: Whether to perform consistency checking.
                Requires reference_texts to have any effect. Default: True.
            check_factuality: Whether to perform factuality checking.
                Default: True.

        Returns:
            HallucinationReport: Comprehensive report containing all flags,
                scores, and recommendations.

        Examples:
            Basic detection:

            >>> detector = ComprehensiveHallucinationDetector()
            >>> report = detector.detect("This is definitely always true.")
            >>> report.has_hallucinations
            True

            With reference texts:

            >>> report = detector.detect(
            ...     "The meeting is at 3pm.",
            ...     reference_texts=["The meeting is at 2pm.", "Meeting at 3pm."]
            ... )
            >>> report.consistency_check is not None
            True

            Selective checking:

            >>> report = detector.detect(
            ...     text,
            ...     check_consistency=False,
            ...     check_factuality=True
            ... )
            >>> report.consistency_check is None
            True

            Analyzing results:

            >>> report = detector.detect(text)
            >>> for flag in report.flags:
            ...     print(f"{flag.hallucination_type.value}: {flag.text_span}")
            >>> for rec in report.recommendations:
            ...     print(f"- {rec}")
        """
        flags = []

        # Pattern-based detection
        pattern_flags = self.pattern_detector.detect(text)
        flags.extend(pattern_flags)

        # Consistency checking (if multiple texts provided)
        consistency_result = None
        if check_consistency and reference_texts:
            all_texts = [text] + reference_texts
            consistency_result = self.consistency_checker.check(all_texts)

            # Add flags for inconsistencies
            for claim1, claim2, disagreement in consistency_result.inconsistent_claims:
                flags.append(
                    HallucinationFlag(
                        hallucination_type=HallucinationType.LOGICAL_INCONSISTENCY,
                        severity=SeverityLevel.MEDIUM if disagreement < 0.5 else SeverityLevel.HIGH,
                        text_span=claim1[:100],
                        start_pos=text.find(claim1) if claim1 in text else 0,
                        end_pos=text.find(claim1) + len(claim1) if claim1 in text else 0,
                        confidence=disagreement,
                        explanation=f"Inconsistent with: '{claim2[:50]}...'",
                        evidence=[claim2],
                    )
                )

        # Factuality checking
        factuality_result = None
        if check_factuality:
            factuality_result = self.factuality_checker.check(text)

            # Add flags for contradicted claims
            for detail in factuality_result.claim_details:
                if detail["status"] == "contradicted":
                    flags.append(
                        HallucinationFlag(
                            hallucination_type=HallucinationType.FACTUAL_ERROR,
                            severity=SeverityLevel.HIGH,
                            text_span=detail["claim"][:100],
                            start_pos=text.find(detail["claim"]) if detail["claim"] in text else 0,
                            end_pos=text.find(detail["claim"]) + len(detail["claim"])
                            if detail["claim"] in text
                            else 0,
                            confidence=0.8,
                            explanation="Claim contradicts known facts",
                        )
                    )

        # Calculate overall score
        if flags:
            # Weight by severity and confidence
            penalty = 0.0
            for flag in flags:
                severity_weight = {
                    SeverityLevel.LOW: 0.1,
                    SeverityLevel.MEDIUM: 0.2,
                    SeverityLevel.HIGH: 0.3,
                    SeverityLevel.CRITICAL: 0.5,
                }[flag.severity]
                penalty += severity_weight * flag.confidence
            overall_score = max(0.0, 1.0 - min(1.0, penalty))
        else:
            overall_score = 1.0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            flags, consistency_result, factuality_result
        )

        return HallucinationReport(
            text=text,
            flags=flags,
            overall_score=overall_score,
            consistency_check=consistency_result,
            factuality_score=factuality_result,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        flags: list[HallucinationFlag],
        consistency: Optional[ConsistencyCheck],
        factuality: Optional[FactualityScore],
    ) -> list[str]:
        """
        Generate actionable recommendations based on detection results.

        Analyzes the flags, consistency results, and factuality scores to
        produce a list of recommendations for addressing detected issues.

        Args:
            flags: List of HallucinationFlag objects from detection.
            consistency: Optional ConsistencyCheck results.
            factuality: Optional FactualityScore results.

        Returns:
            list[str]: List of recommendation strings.

        Examples:
            No issues found:

            >>> detector = ComprehensiveHallucinationDetector()
            >>> recs = detector._generate_recommendations([], None, None)
            >>> "No obvious hallucinations detected" in recs
            True

            With critical flags:

            >>> flags = [HallucinationFlag(
            ...     hallucination_type=HallucinationType.FACTUAL_ERROR,
            ...     severity=SeverityLevel.CRITICAL,
            ...     text_span="test", start_pos=0, end_pos=4,
            ...     confidence=0.9, explanation="test"
            ... )]
            >>> recs = detector._generate_recommendations(flags, None, None)
            >>> any("critical" in r.lower() for r in recs)
            True

            With inconsistent responses:

            >>> consistency = ConsistencyCheck(
            ...     responses=["a", "b"],
            ...     consistent_claims=[],
            ...     inconsistent_claims=[("x", "y", 0.8)],
            ...     overall_consistency=0.5
            ... )
            >>> recs = detector._generate_recommendations([], consistency, None)
            >>> any("inconsistencies" in r.lower() for r in recs)
            True

            With unverified claims:

            >>> factuality = FactualityScore(
            ...     score=0.5, verifiable_claims=10,
            ...     verified_claims=2, unverified_claims=8,
            ...     contradicted_claims=0
            ... )
            >>> recs = detector._generate_recommendations([], None, factuality)
            >>> any("verified" in r.lower() for r in recs)
            True
        """
        recommendations = []

        if not flags:
            recommendations.append("No obvious hallucinations detected")
        else:
            critical = [f for f in flags if f.severity == SeverityLevel.CRITICAL]
            high = [f for f in flags if f.severity == SeverityLevel.HIGH]

            if critical:
                recommendations.append(
                    f"Found {len(critical)} critical issues requiring immediate attention"
                )
            if high:
                recommendations.append(f"Found {len(high)} high-severity potential hallucinations")

        if consistency and not consistency.is_consistent:
            recommendations.append("Response shows inconsistencies - consider regenerating")

        if factuality:
            if factuality.contradicted_claims > 0:
                recommendations.append(
                    f"{factuality.contradicted_claims} claims contradict known facts"
                )
            if factuality.unverified_claims > factuality.verified_claims:
                recommendations.append("Many claims could not be verified")

        # Type-specific recommendations
        type_counts: dict[HallucinationType, int] = {}
        for flag in flags:
            type_counts[flag.hallucination_type] = type_counts.get(flag.hallucination_type, 0) + 1

        if type_counts.get(HallucinationType.UNSUPPORTED_CLAIM, 0) > 2:
            recommendations.append("Many unsupported claims - request sources or citations")
        if type_counts.get(HallucinationType.EXAGGERATION, 0) > 2:
            recommendations.append("Overconfident language detected - verify claims independently")

        return recommendations


class GroundednessChecker:
    """
    Checks if LLM responses are grounded in provided source context.

    Groundedness checking helps detect hallucinations by verifying that
    claims in a response have supporting evidence in the provided context
    (e.g., retrieved documents for RAG systems). Ungrounded sentences
    may indicate the model is generating information not present in the
    source material.

    The checker compares each sentence in the response against sentences
    in the context using a similarity function, flagging sentences that
    don't have sufficient support.

    Attributes:
        similarity_fn: Function to compute similarity between two strings.
            Returns float from 0.0 (different) to 1.0 (identical).
        threshold: Minimum similarity score for a sentence to be considered
            grounded. Default is 0.5.

    Examples:
        Basic groundedness checking:

        >>> checker = GroundednessChecker()
        >>> context = "Python was created by Guido van Rossum in 1991."
        >>> response = "Python was made by Guido van Rossum."
        >>> result = checker.check(response, context)
        >>> result['is_grounded']
        True

        Detecting ungrounded content:

        >>> context = "The company sells software products."
        >>> response = "The company was founded in 1999 and sells software."
        >>> result = checker.check(response, context)
        >>> result['ungrounded_sentences'] > 0
        True  # "founded in 1999" is not in context

        Custom threshold:

        >>> checker = GroundednessChecker(threshold=0.7)
        >>> result = checker.check(response, context)
        >>> result['groundedness_score']  # Higher threshold = stricter

        Using with RAG systems:

        >>> retrieved_docs = "Doc 1 content. Doc 2 content."
        >>> llm_response = "Based on the documents..."
        >>> result = checker.check(llm_response, retrieved_docs)
        >>> if not result['is_grounded']:
        ...     print("Warning: Response may contain hallucinated content")
    """

    def __init__(
        self,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the groundedness checker.

        Args:
            similarity_fn: Optional custom function for comparing text similarity.
                Must accept two strings and return a float between 0.0 and 1.0.
                Defaults to word_overlap_similarity if not provided.
            threshold: Minimum similarity score to consider a sentence grounded.
                Range: 0.0 to 1.0. Default: 0.5. Higher values are stricter.

        Examples:
            Default initialization:

            >>> checker = GroundednessChecker()
            >>> checker.threshold
            0.5

            Custom threshold:

            >>> checker = GroundednessChecker(threshold=0.7)
            >>> checker.threshold
            0.7

            Custom similarity function:

            >>> def cosine_sim(a: str, b: str) -> float:
            ...     # Would use embeddings in practice
            ...     return 0.8
            >>> checker = GroundednessChecker(similarity_fn=cosine_sim)

            Strict checking:

            >>> strict_checker = GroundednessChecker(threshold=0.8)
        """
        self.similarity_fn = similarity_fn or word_overlap_similarity
        self.threshold = threshold

    def check(
        self,
        response: str,
        context: str,
        strict: bool = False,
    ) -> dict[str, Any]:
        """
        Check if response sentences are grounded in the provided context.

        Splits both response and context into sentences, then finds the
        best matching context sentence for each response sentence. Sentences
        with similarity below the threshold are flagged as ungrounded.

        Args:
            response: The LLM response text to check for groundedness.
            context: The source context that should support the response.
            strict: Reserved for future use with stricter checking modes.

        Returns:
            dict[str, Any]: Dictionary containing:
                - groundedness_score: Float 0.0-1.0, proportion of grounded sentences
                - grounded_sentences: Count of sentences with context support
                - ungrounded_sentences: Count of sentences without support
                - is_grounded: True if groundedness_score >= 0.8
                - grounded: List of dicts with sentence, support, similarity (max 5)
                - ungrounded: List of dicts with sentence, best_match, similarity (max 5)

        Examples:
            Fully grounded response:

            >>> checker = GroundednessChecker()
            >>> context = "Paris is the capital of France. It has the Eiffel Tower."
            >>> response = "Paris is France's capital."
            >>> result = checker.check(response, context)
            >>> result['groundedness_score'] > 0.5
            True

            Partially grounded response:

            >>> context = "The product costs $50."
            >>> response = "The product costs $50 and ships in 2 days."
            >>> result = checker.check(response, context)
            >>> result['ungrounded_sentences'] >= 0
            True

            Accessing grounded details:

            >>> result = checker.check(response, context)
            >>> for g in result['grounded']:
            ...     print(f"'{g['sentence']}' supported by '{g['support']}'")
            ...     print(f"Similarity: {g['similarity']:.2f}")

            Checking ungrounded sentences:

            >>> for u in result['ungrounded']:
            ...     print(f"Ungrounded: '{u['sentence']}'")
            ...     print(f"Best match: '{u['best_match']}' ({u['similarity']:.2f})")
        """
        # Extract sentences from response
        sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]

        grounded = []
        ungrounded = []

        for sentence in sentences:
            if len(sentence) < 10:
                continue

            # Check if sentence has support in context
            max_sim = 0.0
            best_match = ""

            # Compare against context sentences
            context_sentences = [s.strip() for s in re.split(r"[.!?]+", context) if s.strip()]
            for ctx_sent in context_sentences:
                sim = self.similarity_fn(sentence, ctx_sent)
                if sim > max_sim:
                    max_sim = sim
                    best_match = ctx_sent

            if max_sim >= self.threshold:
                grounded.append(
                    {
                        "sentence": sentence,
                        "support": best_match,
                        "similarity": max_sim,
                    }
                )
            else:
                ungrounded.append(
                    {
                        "sentence": sentence,
                        "best_match": best_match,
                        "similarity": max_sim,
                    }
                )

        total = len(grounded) + len(ungrounded)
        groundedness_score = len(grounded) / total if total > 0 else 1.0

        return {
            "groundedness_score": round(groundedness_score, 4),
            "grounded_sentences": len(grounded),
            "ungrounded_sentences": len(ungrounded),
            "is_grounded": groundedness_score >= 0.8,
            "grounded": grounded[:5],  # First 5
            "ungrounded": ungrounded[:5],
        }


class AttributionChecker:
    """
    Checks if specific claims in text are properly attributed to sources.

    This checker identifies attribution phrases (e.g., "according to",
    "research shows") and verifies that specific claims like statistics
    have nearby attributions. Claims without attribution may indicate
    fabricated information.

    The checker looks for:
        - Attribution patterns: "according to X", "X said", "cited by X", etc.
        - Specific claims that need attribution: statistics, research references
        - Proximity of attributions to claims (within 100 characters)

    Attributes:
        attribution_patterns: List of regex patterns for finding attributions.

    Examples:
        Basic attribution checking:

        >>> checker = AttributionChecker()
        >>> text = "According to NASA, the moon is 238,900 miles away."
        >>> result = checker.check(text)
        >>> result['attributions_found']
        1
        >>> result['attribution_score']
        1.0

        Detecting unattributed claims:

        >>> text = "87% of users prefer this product. Studies show it works."
        >>> result = checker.check(text)
        >>> result['unattributed_claims'] > 0
        True

        Properly attributed statistics:

        >>> text = "According to Pew Research, 65% of Americans use social media."
        >>> result = checker.check(text)
        >>> result['attribution_score']
        1.0

        Analyzing attribution details:

        >>> result = checker.check(text)
        >>> for attr in result['attributions']:
        ...     print(f"Source: {attr['source']} at position {attr['position']}")
    """

    def __init__(self):
        """
        Initialize the attribution checker with default patterns.

        Sets up regex patterns for detecting various attribution phrases
        commonly used in text to cite sources.

        Examples:
            Default initialization:

            >>> checker = AttributionChecker()
            >>> len(checker.attribution_patterns)
            6

            Accessing patterns:

            >>> checker = AttributionChecker()
            >>> any("according" in p for p in checker.attribution_patterns)
            True

            Extending patterns:

            >>> checker = AttributionChecker()
            >>> checker.attribution_patterns.append(r"as stated by ([^,]+)")
            >>> len(checker.attribution_patterns)
            7

            Pattern structure:

            >>> checker = AttributionChecker()
            >>> # Each pattern captures the source name in group 1
        """
        self.attribution_patterns = [
            r"according to ([^,]+)",
            r"([^,]+) said",
            r"([^,]+) reported",
            r"([^,]+) found that",
            r"cited by ([^,]+)",
            r"source: ([^,\n]+)",
        ]

    def check(self, text: str) -> dict[str, Any]:
        """
        Check if claims in text are properly attributed to sources.

        Finds attribution phrases and specific claims that need attribution
        (statistics, research references), then calculates an attribution
        score based on whether claims have nearby attributions.

        Args:
            text: The text to check for proper attribution.

        Returns:
            dict[str, Any]: Dictionary containing:
                - attributions_found: Count of attribution phrases found
                - attributions: List of dicts with text, source, position
                - unattributed_claims: Count of claims without attribution
                - unattributed: List of dicts with text, position
                - attribution_score: Float 0.0-1.0, ratio of attributed claims

        Examples:
            Well-attributed text:

            >>> checker = AttributionChecker()
            >>> text = "According to the WHO, vaccines save millions of lives."
            >>> result = checker.check(text)
            >>> result['attributions_found']
            1
            >>> result['attributions'][0]['source']
            'the WHO'

            Detecting unattributed statistics:

            >>> text = "75% of developers prefer Python over other languages."
            >>> result = checker.check(text)
            >>> result['unattributed_claims']
            1
            >>> result['unattributed'][0]['text']
            '75% of'

            Mixed attribution:

            >>> text = "According to a survey, 60% like X. Also, 40% prefer Y."
            >>> result = checker.check(text)
            >>> # First stat is attributed, second may not be

            Using attribution score:

            >>> result = checker.check(text)
            >>> if result['attribution_score'] < 0.8:
            ...     print("Warning: Many claims lack proper attribution")
        """
        attributions = []

        for pattern in self.attribution_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                attributions.append(
                    {
                        "text": match.group(),
                        "source": match.group(1).strip(),
                        "position": match.start(),
                    }
                )

        # Check for unattributed specific claims
        claim_patterns = [
            r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?%?)\s+(?:of|people|percent)",
            r"(studies|research|data)\s+(?:shows?|indicates?|suggests?)",
        ]

        unattributed_claims = []
        for pattern in claim_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Check if this claim has an attribution nearby
                claim_pos = match.start()
                has_attribution = any(
                    abs(attr["position"] - claim_pos) < 100 for attr in attributions
                )
                if not has_attribution:
                    unattributed_claims.append(
                        {
                            "text": match.group(),
                            "position": claim_pos,
                        }
                    )

        return {
            "attributions_found": len(attributions),
            "attributions": attributions,
            "unattributed_claims": len(unattributed_claims),
            "unattributed": unattributed_claims,
            "attribution_score": 1.0
            if not unattributed_claims
            else (len(attributions) / (len(attributions) + len(unattributed_claims))),
        }


# Convenience functions
def detect_hallucinations(
    text: str,
    reference_texts: Optional[list[str]] = None,
) -> HallucinationReport:
    """
    Detect potential hallucinations in text using comprehensive analysis.

    This is a convenience function that creates a ComprehensiveHallucinationDetector
    with default settings and runs detection. For custom configurations, use
    the ComprehensiveHallucinationDetector class directly.

    Args:
        text: The text to analyze for potential hallucinations.
        reference_texts: Optional list of other responses to compare for
            consistency checking.

    Returns:
        HallucinationReport: Comprehensive report with flags, scores,
            and recommendations.

    Examples:
        Basic detection:

        >>> report = detect_hallucinations("Everyone definitely agrees with this.")
        >>> print(f"Has issues: {report.has_hallucinations}")
        >>> print(f"Score: {report.overall_score:.2f}")

        With consistency checking:

        >>> responses = ["The year is 2023.", "The year is 2024."]
        >>> report = detect_hallucinations(responses[0], reference_texts=responses[1:])
        >>> print(f"Consistent: {report.consistency_check.is_consistent if report.consistency_check else 'N/A'}")

        Analyzing results:

        >>> report = detect_hallucinations(llm_output)
        >>> for flag in report.flags:
        ...     print(f"{flag.severity.value}: {flag.text_span}")

        Quick validation:

        >>> report = detect_hallucinations(text)
        >>> if report.overall_score < 0.7:
        ...     print("Warning: Potential hallucinations detected")
    """
    detector = ComprehensiveHallucinationDetector()
    return detector.detect(text, reference_texts)


def check_consistency(responses: list[str]) -> ConsistencyCheck:
    """
    Check consistency across multiple LLM responses.

    This is a convenience function for comparing multiple responses to the
    same question. Inconsistencies often indicate hallucination.

    Args:
        responses: List of response strings to compare.

    Returns:
        ConsistencyCheck: Results with consistent/inconsistent claims and
            overall consistency score.

    Examples:
        Checking multiple responses:

        >>> responses = [
        ...     "Python was created in 1991.",
        ...     "Python was made in 1991.",
        ...     "Python was developed in 1989."
        ... ]
        >>> result = check_consistency(responses)
        >>> print(f"Consistent: {result.is_consistent}")
        >>> print(f"Score: {result.overall_consistency:.2f}")

        Analyzing inconsistencies:

        >>> for claim1, claim2, disagreement in result.inconsistent_claims:
        ...     print(f"Conflict: '{claim1}' vs '{claim2}'")

        Single response handling:

        >>> result = check_consistency(["Only one response"])
        >>> result.overall_consistency
        1.0

        Using as validation:

        >>> result = check_consistency(responses)
        >>> if not result.is_consistent:
        ...     print("Responses contradict each other")
    """
    checker = FactConsistencyChecker()
    return checker.check(responses)


def check_factuality(
    text: str,
    knowledge_base: Optional[dict[str, str]] = None,
) -> FactualityScore:
    """
    Check factual accuracy of text against a knowledge base.

    This is a convenience function for fact-checking extracted claims
    against known facts. For custom fact-checking functions, use the
    FactualityChecker class directly.

    Args:
        text: The text to fact-check.
        knowledge_base: Optional dictionary of known facts. Keys are
            identifiers, values are fact strings.

    Returns:
        FactualityScore: Results with verified/unverified/contradicted
            claim counts and overall score.

    Examples:
        Basic fact-checking:

        >>> kb = {"paris": "Paris is the capital of France"}
        >>> result = check_factuality("Paris is the capital of France.", kb)
        >>> print(f"Score: {result.score}")
        >>> print(f"Accuracy: {result.accuracy:.0%}")

        Without knowledge base:

        >>> result = check_factuality("Some claim about something.")
        >>> print(f"Verifiable claims: {result.verifiable_claims}")
        >>> print(f"Unverified: {result.unverified_claims}")

        Analyzing claim details:

        >>> result = check_factuality(text, knowledge_base=kb)
        >>> for detail in result.claim_details:
        ...     print(f"{detail['status']}: {detail['claim']}")

        Using for validation:

        >>> result = check_factuality(llm_response, kb)
        >>> if result.contradicted_claims > 0:
        ...     print("Contains factual errors!")
    """
    checker = FactualityChecker(knowledge_base=knowledge_base)
    return checker.check(text)


def check_groundedness(
    response: str,
    context: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Check if a response is grounded in the provided context.

    This is a convenience function for RAG (Retrieval-Augmented Generation)
    systems to verify that generated responses are supported by the
    retrieved context.

    Args:
        response: The LLM response to check.
        context: The source context that should support the response.
        threshold: Minimum similarity score for grounding (0.0-1.0).
            Default: 0.5.

    Returns:
        dict[str, Any]: Dictionary with groundedness score, grounded/ungrounded
            sentence counts and details, and is_grounded boolean.

    Examples:
        Basic groundedness check:

        >>> context = "Python was created by Guido van Rossum in 1991."
        >>> response = "Python was made by Guido van Rossum."
        >>> result = check_groundedness(response, context)
        >>> print(f"Grounded: {result['is_grounded']}")
        >>> print(f"Score: {result['groundedness_score']:.2f}")

        Custom threshold:

        >>> result = check_groundedness(response, context, threshold=0.7)
        >>> print(f"Stricter check grounded: {result['is_grounded']}")

        Analyzing ungrounded content:

        >>> result = check_groundedness(response, context)
        >>> for u in result['ungrounded']:
        ...     print(f"Ungrounded: '{u['sentence']}'")
        ...     print(f"  Best match: '{u['best_match']}'")

        RAG validation:

        >>> retrieved_docs = get_relevant_documents(query)
        >>> llm_response = generate_response(query, retrieved_docs)
        >>> result = check_groundedness(llm_response, retrieved_docs)
        >>> if not result['is_grounded']:
        ...     print("Warning: Response may contain hallucinated content")
    """
    checker = GroundednessChecker(threshold=threshold)
    return checker.check(response, context)


def check_attribution(text: str) -> dict[str, Any]:
    """
    Check if claims in text are properly attributed to sources.

    This is a convenience function for verifying that specific claims
    (statistics, research references) have proper attribution phrases.

    Args:
        text: The text to check for proper attribution.

    Returns:
        dict[str, Any]: Dictionary with attribution count, attribution details,
            unattributed claim count and details, and attribution score.

    Examples:
        Checking attribution:

        >>> text = "According to NASA, the moon is 238,900 miles from Earth."
        >>> result = check_attribution(text)
        >>> print(f"Attributions: {result['attributions_found']}")
        >>> print(f"Score: {result['attribution_score']:.2f}")

        Detecting unattributed claims:

        >>> text = "87% of developers prefer Python. Studies show it's efficient."
        >>> result = check_attribution(text)
        >>> print(f"Unattributed claims: {result['unattributed_claims']}")

        Analyzing sources:

        >>> result = check_attribution(text)
        >>> for attr in result['attributions']:
        ...     print(f"Source: {attr['source']}")

        Validation workflow:

        >>> result = check_attribution(llm_response)
        >>> if result['unattributed_claims'] > 0:
        ...     print("Warning: Some claims lack proper attribution")
        ...     for claim in result['unattributed']:
        ...         print(f"  - {claim['text']}")
    """
    checker = AttributionChecker()
    return checker.check(text)


def quick_hallucination_check(text: str) -> dict[str, Any]:
    """
    Perform a quick hallucination check and return a summary.

    This is a convenience function for rapid assessment of potential
    hallucinations, returning a simplified summary dictionary instead
    of the full HallucinationReport object.

    Args:
        text: The text to check for hallucinations.

    Returns:
        dict[str, Any]: Summary dictionary containing:
            - has_potential_hallucinations: Boolean indicating if issues found
            - overall_score: Float 0.0-1.0 indicating hallucination likelihood
            - n_flags: Total number of potential issues detected
            - critical_issues: Count of critical severity issues
            - recommendations: Top 3 recommendations (list of strings)

    Examples:
        Quick assessment:

        >>> result = quick_hallucination_check("Everyone definitely knows this.")
        >>> print(f"Issues: {result['n_flags']}")
        >>> print(f"Score: {result['overall_score']:.2f}")

        Conditional handling:

        >>> result = quick_hallucination_check(llm_output)
        >>> if result['has_potential_hallucinations']:
        ...     print("Potential issues detected!")
        ...     for rec in result['recommendations']:
        ...         print(f"  - {rec}")

        Critical issue check:

        >>> result = quick_hallucination_check(response)
        >>> if result['critical_issues'] > 0:
        ...     raise ValueError("Critical hallucinations found!")

        Threshold-based validation:

        >>> result = quick_hallucination_check(text)
        >>> if result['overall_score'] < 0.7:
        ...     # Request regeneration or human review
        ...     regenerate_response()
    """
    report = detect_hallucinations(text)
    return {
        "has_potential_hallucinations": report.has_hallucinations,
        "overall_score": report.overall_score,
        "n_flags": len(report.flags),
        "critical_issues": len(report.critical_flags),
        "recommendations": report.recommendations[:3],
    }


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import ConsistencyChecker. The canonical name is
# FactConsistencyChecker.
ConsistencyChecker = FactConsistencyChecker

# Older code and tests may import HallucinationDetector. The canonical name is
# ComprehensiveHallucinationDetector.
HallucinationDetector = ComprehensiveHallucinationDetector
