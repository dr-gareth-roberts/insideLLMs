"""
Knowledge probing and fact verification utilities.

This module provides comprehensive tools for testing and evaluating the knowledge
capabilities of large language models. It enables systematic probing of model
knowledge, extraction and verification of factual claims, consistency testing
across paraphrased queries, and source attribution.

Key Components:
    - **Claim Extraction**: Extract structured factual claims from text using
      pattern-based analysis.
    - **Fact Verification**: Verify claims against knowledge bases with evidence
      collection and confidence scoring.
    - **Knowledge Probing**: Test model knowledge through targeted questions with
      expected answers and difficulty levels.
    - **Consistency Testing**: Evaluate response consistency across paraphrased
      versions of the same question.
    - **Source Attribution**: Track and attribute claims to their sources with
      reliability scoring.

Provides tools for:
    - Knowledge probing and fact extraction
    - Claim verification and fact-checking
    - Knowledge consistency testing
    - Confidence calibration
    - Source attribution

Example: Basic Claim Extraction and Verification
    >>> from insideLLMs.contrib.knowledge import extract_claims, verify_claim
    >>> text = "Albert Einstein was a physicist. He developed the theory of relativity."
    >>> claims = extract_claims(text, min_confidence=0.3)
    >>> for claim in claims:
    ...     print(f"Claim: {claim.text}")
    ...     print(f"Subject: {claim.subject}, Category: {claim.category.value}")

Example: Knowledge Probing with a Model
    >>> from insideLLMs.contrib.knowledge import probe_knowledge
    >>> def mock_model(question: str) -> str:
    ...     return "Paris is the capital of France."
    >>> questions = ["What is the capital of France?", "Who wrote Hamlet?"]
    >>> expected = ["Paris", "Shakespeare"]
    >>> report = probe_knowledge(questions, mock_model, expected)
    >>> print(f"Accuracy: {report.accuracy:.2%}")

Example: Consistency Testing
    >>> from insideLLMs.contrib.knowledge import check_consistency
    >>> def model_fn(q: str) -> str:
    ...     return "The answer is 42."
    >>> result = check_consistency("What is the meaning of life?", model_fn)
    >>> print(f"Consistency score: {result.consistency_score:.2f}")

Example: Full Fact Verification Pipeline
    >>> from insideLLMs.contrib.knowledge import verify_facts
    >>> kb = {"Einstein": "physicist who developed relativity"}
    >>> text = "Einstein was a renowned physicist."
    >>> results = verify_facts(text, knowledge_base=kb)
    >>> for r in results:
    ...     print(f"Status: {r.status.value}, Confidence: {r.confidence:.2f}")
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from insideLLMs.nlp.similarity import word_overlap_similarity


class VerificationStatus(Enum):
    """Status of fact verification.

    Represents the outcome of verifying a factual claim against a knowledge base
    or other evidence sources. Used by FactVerifier to communicate verification
    results.

    Attributes:
        VERIFIED: The claim is supported by evidence with no contradictions.
        CONTRADICTED: The claim conflicts with available evidence.
        UNVERIFIABLE: Insufficient evidence to verify or contradict the claim.
        PARTIAL: Mixed evidence - some support and some contradiction.
        UNKNOWN: Verification could not be performed (e.g., system error).

    Example: Checking Verification Status
        >>> from insideLLMs.contrib.knowledge import verify_claim, Claim, VerificationStatus
        >>> claim = Claim(text="Paris is the capital of France")
        >>> kb = {"Paris": "capital of France"}
        >>> result = verify_claim(claim, knowledge_base=kb)
        >>> if result.status == VerificationStatus.VERIFIED:
        ...     print("Claim is verified!")

    Example: Handling Different Statuses
        >>> result = verify_claim(claim)
        >>> status_messages = {
        ...     VerificationStatus.VERIFIED: "Confirmed true",
        ...     VerificationStatus.CONTRADICTED: "Found to be false",
        ...     VerificationStatus.UNVERIFIABLE: "Cannot determine",
        ...     VerificationStatus.PARTIAL: "Partially supported",
        ... }
        >>> print(status_messages.get(result.status, "Unknown status"))

    Example: Filtering Results by Status
        >>> from insideLLMs.contrib.knowledge import verify_facts
        >>> results = verify_facts("Einstein was a physicist. Pluto is a planet.")
        >>> verified = [r for r in results if r.status == VerificationStatus.VERIFIED]
        >>> contradicted = [r for r in results if r.status == VerificationStatus.CONTRADICTED]
    """

    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class KnowledgeCategory(Enum):
    """Categories of knowledge for classification and analysis.

    Categorizes different types of knowledge that can be probed or verified.
    This taxonomy helps organize knowledge probes and analyze model performance
    across different knowledge domains.

    Attributes:
        FACTUAL: Concrete facts about entities, events, or properties.
            Example: "The Earth orbits the Sun."
        PROCEDURAL: Knowledge about how to perform tasks or processes.
            Example: "To bake bread, first mix flour and water."
        CONCEPTUAL: Abstract concepts, definitions, and relationships.
            Example: "Democracy is a form of government."
        TEMPORAL: Time-related knowledge including dates and sequences.
            Example: "World War II ended in 1945."
        RELATIONAL: Knowledge about relationships between entities.
            Example: "Einstein developed the theory of relativity."
        COMMONSENSE: Everyday knowledge most humans possess.
            Example: "Water is wet."

    Example: Categorizing Knowledge Probes
        >>> from insideLLMs.contrib.knowledge import KnowledgeProbe, KnowledgeCategory
        >>> factual_probe = KnowledgeProbe(
        ...     question="What is the speed of light?",
        ...     expected_answer="299,792,458 meters per second",
        ...     category=KnowledgeCategory.FACTUAL
        ... )
        >>> temporal_probe = KnowledgeProbe(
        ...     question="When did the French Revolution begin?",
        ...     expected_answer="1789",
        ...     category=KnowledgeCategory.TEMPORAL
        ... )

    Example: Analyzing Results by Category
        >>> from insideLLMs.contrib.knowledge import KnowledgeProber
        >>> prober = KnowledgeProber()
        >>> prober.add_probe("What is gravity?", category=KnowledgeCategory.CONCEPTUAL)
        >>> prober.add_probe("When was Einstein born?", category=KnowledgeCategory.TEMPORAL)
        >>> # After running probes, report shows accuracy by_category

    Example: Filtering Claims by Category
        >>> from insideLLMs.contrib.knowledge import extract_claims
        >>> claims = extract_claims("In 1969, Neil Armstrong walked on the Moon.")
        >>> temporal_claims = [c for c in claims if c.category == KnowledgeCategory.TEMPORAL]
        >>> factual_claims = [c for c in claims if c.category == KnowledgeCategory.FACTUAL]
    """

    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    TEMPORAL = "temporal"
    RELATIONAL = "relational"
    COMMONSENSE = "commonsense"


class ConfidenceLevel(Enum):
    """Confidence levels for knowledge assertions and model responses.

    Provides a discrete scale for expressing confidence in claims, predictions,
    or model responses. Useful for calibration analysis and uncertainty
    quantification in knowledge testing.

    Attributes:
        VERY_LOW: Minimal confidence (0-20%). Model is highly uncertain.
        LOW: Low confidence (20-40%). Model has significant doubt.
        MEDIUM: Moderate confidence (40-60%). Model is somewhat certain.
        HIGH: High confidence (60-80%). Model is fairly confident.
        VERY_HIGH: Very high confidence (80-100%). Model is highly certain.

    Example: Mapping Numeric Confidence to Levels
        >>> from insideLLMs.contrib.knowledge import ConfidenceLevel
        >>> def numeric_to_level(confidence: float) -> ConfidenceLevel:
        ...     if confidence < 0.2:
        ...         return ConfidenceLevel.VERY_LOW
        ...     elif confidence < 0.4:
        ...         return ConfidenceLevel.LOW
        ...     elif confidence < 0.6:
        ...         return ConfidenceLevel.MEDIUM
        ...     elif confidence < 0.8:
        ...         return ConfidenceLevel.HIGH
        ...     else:
        ...         return ConfidenceLevel.VERY_HIGH
        >>> print(numeric_to_level(0.75).value)  # 'high'

    Example: Using Confidence Levels in Analysis
        >>> levels = [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]
        >>> for level in levels:
        ...     print(f"Confidence: {level.value}")

    Example: Confidence Calibration
        >>> # Check if high confidence corresponds to high accuracy
        >>> high_conf_results = [r for r in results if r.confidence_expressed > 0.8]
        >>> high_conf_accuracy = sum(r.is_correct for r in high_conf_results) / len(high_conf_results)
        >>> # Well-calibrated if high_conf_accuracy is close to 0.8+
    """

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class Claim:
    """A factual claim extracted from text for verification.

    Represents a structured factual assertion that can be verified against
    knowledge bases or other evidence sources. Claims are extracted from
    natural language text and decomposed into subject-predicate-object
    triples when possible.

    Attributes:
        text: The full text of the claim as extracted from the source.
        subject: The entity or topic the claim is about (e.g., "Einstein").
        predicate: The relationship or property being asserted (e.g., "was").
        object: The value or target of the assertion (e.g., "a physicist").
        category: The type of knowledge this claim represents.
        confidence: Extraction confidence score from 0.0 to 1.0.

    Example: Creating a Simple Claim
        >>> from insideLLMs.contrib.knowledge import Claim, KnowledgeCategory
        >>> claim = Claim(
        ...     text="Albert Einstein was a physicist",
        ...     subject="Albert Einstein",
        ...     predicate="was",
        ...     object="a physicist",
        ...     category=KnowledgeCategory.FACTUAL,
        ...     confidence=0.85
        ... )
        >>> print(f"Subject: {claim.subject}, Confidence: {claim.confidence}")

    Example: Extracting Claims from Text
        >>> from insideLLMs.contrib.knowledge import extract_claims
        >>> text = "Marie Curie discovered radium in 1898."
        >>> claims = extract_claims(text)
        >>> for claim in claims:
        ...     print(f"Claim: {claim.text}")
        ...     print(f"Category: {claim.category.value}")

    Example: Converting to Dictionary for Serialization
        >>> claim = Claim(text="Water boils at 100 degrees Celsius")
        >>> claim_dict = claim.to_dict()
        >>> print(claim_dict["text"])
        >>> print(claim_dict["category"])  # 'factual'

    Example: Creating Temporal Claims
        >>> temporal_claim = Claim(
        ...     text="In 1969, humans landed on the Moon",
        ...     category=KnowledgeCategory.TEMPORAL,
        ...     confidence=0.9
        ... )
    """

    text: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    category: KnowledgeCategory = KnowledgeCategory.FACTUAL
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert the Claim to a dictionary representation.

        Serializes all claim attributes to a dictionary format suitable for
        JSON serialization, logging, or storage. The category enum is
        converted to its string value.

        Returns:
            dict[str, Any]: Dictionary containing all claim attributes with
                the category converted to its string value.

        Example: Basic Serialization
            >>> claim = Claim(text="The sky is blue", confidence=0.8)
            >>> d = claim.to_dict()
            >>> print(d["text"])  # "The sky is blue"
            >>> print(d["category"])  # "factual"

        Example: JSON Serialization
            >>> import json
            >>> claim = Claim(text="Python is a programming language")
            >>> json_str = json.dumps(claim.to_dict())
            >>> print(json_str)

        Example: Batch Serialization
            >>> claims = [Claim(text="Fact 1"), Claim(text="Fact 2")]
            >>> claim_dicts = [c.to_dict() for c in claims]
        """
        return {
            "text": self.text,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "category": self.category.value,
            "confidence": self.confidence,
        }


@dataclass
class VerificationResult:
    """Result of verifying a factual claim against evidence.

    Contains the outcome of fact verification including the verification status,
    confidence level, and collected evidence (both supporting and contradicting).
    Produced by FactVerifier after checking claims against knowledge bases.

    Attributes:
        claim: The original Claim that was verified.
        status: The verification outcome (VERIFIED, CONTRADICTED, etc.).
        confidence: Confidence in the verification result (0.0 to 1.0).
        supporting_evidence: List of evidence strings supporting the claim.
        contradicting_evidence: List of evidence strings contradicting the claim.
        source_quality: Quality score of the evidence sources (0.0 to 1.0).

    Example: Basic Verification
        >>> from insideLLMs.contrib.knowledge import verify_claim, Claim
        >>> claim = Claim(text="Paris is in France")
        >>> kb = {"Paris": "capital city of France"}
        >>> result = verify_claim(claim, knowledge_base=kb)
        >>> print(f"Status: {result.status.value}")
        >>> print(f"Confidence: {result.confidence:.2f}")

    Example: Examining Evidence
        >>> result = verify_claim(claim, knowledge_base=kb)
        >>> if result.supporting_evidence:
        ...     print("Supporting evidence:")
        ...     for evidence in result.supporting_evidence:
        ...         print(f"  - {evidence}")
        >>> if result.contradicting_evidence:
        ...     print("Contradicting evidence:")
        ...     for evidence in result.contradicting_evidence:
        ...         print(f"  - {evidence}")

    Example: Batch Verification Analysis
        >>> from insideLLMs.contrib.knowledge import verify_facts
        >>> results = verify_facts("Einstein was a physicist. He invented the telephone.")
        >>> verified = [r for r in results if r.status.value == "verified"]
        >>> contradicted = [r for r in results if r.status.value == "contradicted"]
        >>> print(f"Verified: {len(verified)}, Contradicted: {len(contradicted)}")

    Example: Quality-Weighted Confidence
        >>> # Combine confidence with source quality
        >>> effective_confidence = result.confidence * result.source_quality
        >>> print(f"Effective confidence: {effective_confidence:.2f}")
    """

    claim: Claim
    status: VerificationStatus
    confidence: float
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    source_quality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert the VerificationResult to a dictionary representation.

        Serializes all verification result attributes including the nested
        claim object. Useful for JSON serialization, logging, or creating
        verification reports.

        Returns:
            dict[str, Any]: Dictionary containing all result attributes with
                nested claim dictionary and status enum converted to string.

        Example: Serialization for Reporting
            >>> result = verify_claim(Claim(text="Test claim"))
            >>> result_dict = result.to_dict()
            >>> print(result_dict["status"])  # e.g., "unverifiable"
            >>> print(result_dict["claim"]["text"])  # "Test claim"

        Example: JSON Export
            >>> import json
            >>> json_str = json.dumps(result.to_dict(), indent=2)
            >>> print(json_str)

        Example: Creating Summary Reports
            >>> results = verify_facts("Multiple facts here.")
            >>> summary = [r.to_dict() for r in results]
            >>> verified_count = sum(1 for r in summary if r["status"] == "verified")
        """
        return {
            "claim": self.claim.to_dict(),
            "status": self.status.value,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "source_quality": self.source_quality,
        }


@dataclass
class KnowledgeProbe:
    """A probe for testing model knowledge through targeted questioning.

    Represents a single knowledge test consisting of a question, an optional
    expected answer for evaluation, metadata about the knowledge type and
    difficulty, and optional hints that can be provided to the model.

    Attributes:
        question: The question to ask the model.
        expected_answer: The expected correct answer for evaluation (optional).
        category: The type of knowledge being tested (factual, temporal, etc.).
        difficulty: Difficulty level from 0.0 (easy) to 1.0 (hard).
        hints: Optional list of hints that can be provided to help the model.

    Example: Creating a Factual Knowledge Probe
        >>> from insideLLMs.contrib.knowledge import KnowledgeProbe, KnowledgeCategory
        >>> probe = KnowledgeProbe(
        ...     question="What is the capital of France?",
        ...     expected_answer="Paris",
        ...     category=KnowledgeCategory.FACTUAL,
        ...     difficulty=0.2
        ... )

    Example: Creating a Difficult Temporal Probe with Hints
        >>> probe = KnowledgeProbe(
        ...     question="In what year did the Byzantine Empire fall?",
        ...     expected_answer="1453",
        ...     category=KnowledgeCategory.TEMPORAL,
        ...     difficulty=0.8,
        ...     hints=["Ottoman conquest", "Fall of Constantinople"]
        ... )

    Example: Using with KnowledgeProber
        >>> from insideLLMs.contrib.knowledge import KnowledgeProber
        >>> prober = KnowledgeProber()
        >>> probe = prober.add_probe(
        ...     question="Who wrote Romeo and Juliet?",
        ...     expected_answer="William Shakespeare",
        ...     difficulty=0.3
        ... )
        >>> print(probe.question)

    Example: Batch Probe Creation
        >>> questions = [
        ...     ("What is H2O?", "Water", 0.1),
        ...     ("What is the speed of light?", "299792458 m/s", 0.7)
        ... ]
        >>> probes = [
        ...     KnowledgeProbe(q, a, difficulty=d)
        ...     for q, a, d in questions
        ... ]
    """

    question: str
    expected_answer: Optional[str] = None
    category: KnowledgeCategory = KnowledgeCategory.FACTUAL
    difficulty: float = 0.5  # 0-1
    hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the KnowledgeProbe to a dictionary representation.

        Serializes all probe attributes for JSON export, logging, or
        storage. The category enum is converted to its string value.

        Returns:
            dict[str, Any]: Dictionary containing all probe attributes.

        Example: Basic Serialization
            >>> probe = KnowledgeProbe(
            ...     question="What is 2+2?",
            ...     expected_answer="4",
            ...     difficulty=0.1
            ... )
            >>> d = probe.to_dict()
            >>> print(d["question"])  # "What is 2+2?"
            >>> print(d["difficulty"])  # 0.1

        Example: Exporting Probe Set
            >>> probes = [probe1, probe2, probe3]
            >>> export_data = [p.to_dict() for p in probes]

        Example: Logging Probe Details
            >>> import logging
            >>> logging.info(f"Running probe: {probe.to_dict()}")
        """
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "category": self.category.value,
            "difficulty": self.difficulty,
            "hints": self.hints,
        }


@dataclass
class KnowledgeProbeResult:
    """Result of running a knowledge probe against a model.

    Contains the model's response to a knowledge probe along with evaluation
    metrics including correctness, partial credit scoring, expressed confidence,
    and whether reasoning was provided. Used by KnowledgeProber to track
    individual probe outcomes.

    Attributes:
        probe: The original KnowledgeProbe that was run.
        response: The model's response to the probe question.
        is_correct: Whether the response was correct (None if no expected answer).
        partial_score: Partial credit score from 0.0 to 1.0 based on word overlap.
        confidence_expressed: Detected confidence level in the response (0.0 to 1.0).
        reasoning_provided: Whether the model provided reasoning for its answer.

    Example: Examining a Probe Result
        >>> from insideLLMs.contrib.knowledge import KnowledgeProber
        >>> prober = KnowledgeProber()
        >>> prober.add_probe("What is 2+2?", expected_answer="4")
        >>> def model(q): return "The answer is 4 because 2 plus 2 equals 4."
        >>> results = prober.run_all_probes(model)
        >>> result = results[0]
        >>> print(f"Correct: {result.is_correct}")
        >>> print(f"Reasoning provided: {result.reasoning_provided}")

    Example: Analyzing Confidence Expression
        >>> # Model expresses high confidence
        >>> def confident_model(q): return "I am absolutely certain the answer is Paris."
        >>> # Model expresses uncertainty
        >>> def uncertain_model(q): return "I'm not sure, but maybe it's Paris?"
        >>> # Check confidence_expressed field to compare

    Example: Partial Credit Scoring
        >>> prober.add_probe(
        ...     "Name three programming languages",
        ...     expected_answer="Python Java JavaScript"
        ... )
        >>> result = prober.run_probe(probe, model)
        >>> # partial_score reflects word overlap even if not fully correct
        >>> print(f"Partial score: {result.partial_score:.2f}")

    Example: Aggregating Results
        >>> results = prober.run_all_probes(model_fn)
        >>> correct = [r for r in results if r.is_correct]
        >>> with_reasoning = [r for r in results if r.reasoning_provided]
        >>> print(f"Correct: {len(correct)}/{len(results)}")
        >>> print(f"With reasoning: {len(with_reasoning)}/{len(results)}")
    """

    probe: KnowledgeProbe
    response: str
    is_correct: Optional[bool] = None
    partial_score: float = 0.0
    confidence_expressed: float = 0.0
    reasoning_provided: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the KnowledgeProbeResult to a dictionary representation.

        Serializes all result attributes including the nested probe object
        for JSON export, logging, or detailed analysis reports.

        Returns:
            dict[str, Any]: Dictionary containing all result attributes with
                nested probe dictionary.

        Example: Result Serialization
            >>> result = prober.run_probe(probe, model_fn)
            >>> d = result.to_dict()
            >>> print(d["is_correct"])
            >>> print(d["probe"]["question"])

        Example: JSON Export for Analysis
            >>> import json
            >>> results = prober.run_all_probes(model_fn)
            >>> export = [r.to_dict() for r in results]
            >>> with open("results.json", "w") as f:
            ...     json.dump(export, f, indent=2)

        Example: Creating Evaluation Reports
            >>> report_data = {
            ...     "results": [r.to_dict() for r in results],
            ...     "summary": {"accuracy": 0.85, "total": len(results)}
            ... }
        """
        return {
            "probe": self.probe.to_dict(),
            "response": self.response,
            "is_correct": self.is_correct,
            "partial_score": self.partial_score,
            "confidence_expressed": self.confidence_expressed,
            "reasoning_provided": self.reasoning_provided,
        }


# Backward compatibility alias
ProbeResult = KnowledgeProbeResult


@dataclass
class ConsistencyResult:
    """Result of testing response consistency across question variations.

    Contains the outcome of consistency testing where the same question is
    asked in multiple paraphrased forms to check if the model gives consistent
    answers. Includes the original and paraphrased responses, a consistency
    score, detected contradictions, and semantic drift measurement.

    Attributes:
        original_response: The model's response to the original question.
        paraphrased_responses: List of responses to paraphrased versions.
        consistency_score: Measure of response consistency (0.0 to 1.0).
        contradictions: List of (original, paraphrased) response pairs that contradict.
        semantic_drift: Measure of how much meaning changed across responses.

    Example: Basic Consistency Testing
        >>> from insideLLMs.contrib.knowledge import check_consistency
        >>> def model(q): return "The capital of France is Paris."
        >>> result = check_consistency("What is the capital of France?", model)
        >>> print(f"Consistency: {result.consistency_score:.2f}")
        >>> print(f"Semantic drift: {result.semantic_drift:.2f}")

    Example: Detecting Contradictions
        >>> # A model that gives inconsistent answers
        >>> import random
        >>> def inconsistent_model(q):
        ...     return random.choice(["Paris", "London", "Berlin"])
        >>> result = check_consistency("What is the capital of France?", inconsistent_model)
        >>> if result.contradictions:
        ...     print("Found contradictions:")
        ...     for orig, para in result.contradictions:
        ...         print(f"  Original: {orig}")
        ...         print(f"  Paraphrased: {para}")

    Example: Analyzing Response Variations
        >>> result = check_consistency("Who invented the telephone?", model)
        >>> print(f"Original: {result.original_response}")
        >>> for i, resp in enumerate(result.paraphrased_responses):
        ...     print(f"Variation {i+1}: {resp}")

    Example: Quality Threshold Checking
        >>> result = check_consistency(question, model)
        >>> if result.consistency_score < 0.7:
        ...     print("Warning: Model shows inconsistent responses")
        >>> if result.semantic_drift > 0.3:
        ...     print("Warning: Significant semantic drift detected")
    """

    original_response: str
    paraphrased_responses: list[str]
    consistency_score: float  # 0-1
    contradictions: list[tuple[str, str]] = field(default_factory=list)
    semantic_drift: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert the ConsistencyResult to a dictionary representation.

        Serializes consistency test results for reporting. Note that
        paraphrased_responses are summarized as a count rather than
        included in full to keep the output concise.

        Returns:
            dict[str, Any]: Dictionary with consistency metrics and counts.

        Example: Generating Consistency Reports
            >>> result = check_consistency(question, model)
            >>> d = result.to_dict()
            >>> print(f"Tested {d['num_paraphrases']} variations")
            >>> print(f"Found {d['num_contradictions']} contradictions")

        Example: Batch Report Generation
            >>> questions = ["Q1?", "Q2?", "Q3?"]
            >>> reports = []
            >>> for q in questions:
            ...     result = check_consistency(q, model)
            ...     reports.append({"question": q, **result.to_dict()})

        Example: JSON Export
            >>> import json
            >>> consistency_data = result.to_dict()
            >>> json_str = json.dumps(consistency_data, indent=2)
        """
        return {
            "original_response": self.original_response,
            "num_paraphrases": len(self.paraphrased_responses),
            "consistency_score": self.consistency_score,
            "num_contradictions": len(self.contradictions),
            "semantic_drift": self.semantic_drift,
        }


@dataclass
class KnowledgeReport:
    """Comprehensive report on knowledge probing results.

    Aggregates results from multiple knowledge probes into a summary report
    with overall accuracy, breakdowns by category and difficulty, confidence
    calibration metrics, and identified knowledge gaps. Generated by
    KnowledgeProber.generate_report().

    Attributes:
        probes_run: Total number of probes executed.
        correct_count: Number of probes answered correctly.
        accuracy: Overall accuracy as a fraction (0.0 to 1.0).
        by_category: Accuracy broken down by knowledge category.
        by_difficulty: Accuracy broken down by difficulty level (easy/medium/hard).
        confidence_calibration: How well expressed confidence matches actual accuracy.
        knowledge_gaps: List of categories where the model performed poorly.

    Example: Generating and Using a Knowledge Report
        >>> from insideLLMs.contrib.knowledge import probe_knowledge
        >>> questions = [
        ...     "What is the capital of France?",
        ...     "Who wrote Hamlet?",
        ...     "When did WWII end?"
        ... ]
        >>> expected = ["Paris", "Shakespeare", "1945"]
        >>> def model(q): return "Paris, Shakespeare, 1945"  # simplified
        >>> report = probe_knowledge(questions, model, expected)
        >>> print(f"Accuracy: {report.accuracy:.2%}")
        >>> print(f"Probes run: {report.probes_run}")

    Example: Analyzing Performance by Category
        >>> report = prober.generate_report(results)
        >>> for category, acc in report.by_category.items():
        ...     print(f"{category}: {acc:.2%}")
        >>> # Identify weak areas
        >>> weak_categories = [c for c, a in report.by_category.items() if a < 0.5]

    Example: Difficulty Analysis
        >>> for difficulty, acc in report.by_difficulty.items():
        ...     print(f"{difficulty}: {acc:.2%}")
        >>> # Check if model struggles with hard questions
        >>> if report.by_difficulty.get("hard", 1.0) < 0.3:
        ...     print("Model struggles with difficult questions")

    Example: Confidence Calibration Assessment
        >>> report = prober.generate_report(results)
        >>> if report.confidence_calibration > 0.8:
        ...     print("Model is well-calibrated")
        >>> elif report.confidence_calibration < 0.5:
        ...     print("Model confidence poorly calibrated")
        >>> # Check knowledge gaps
        >>> if report.knowledge_gaps:
        ...     print(f"Gaps found in: {', '.join(report.knowledge_gaps)}")
    """

    probes_run: int
    correct_count: int
    accuracy: float
    by_category: dict[str, float]
    by_difficulty: dict[str, float]
    confidence_calibration: float
    knowledge_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the KnowledgeReport to a dictionary representation.

        Serializes all report metrics for JSON export, logging, or
        creating formatted reports.

        Returns:
            dict[str, Any]: Dictionary containing all report metrics.

        Example: Creating a Summary Dashboard
            >>> report = prober.generate_report(results)
            >>> d = report.to_dict()
            >>> print(f"Overall: {d['accuracy']:.2%} ({d['correct_count']}/{d['probes_run']})")

        Example: Exporting Report Data
            >>> import json
            >>> report_dict = report.to_dict()
            >>> with open("knowledge_report.json", "w") as f:
            ...     json.dump(report_dict, f, indent=2)

        Example: Comparative Analysis
            >>> reports = [prober.generate_report(r) for r in all_results]
            >>> report_dicts = [r.to_dict() for r in reports]
            >>> accuracies = [r["accuracy"] for r in report_dicts]
            >>> print(f"Average accuracy: {sum(accuracies)/len(accuracies):.2%}")
        """
        return {
            "probes_run": self.probes_run,
            "correct_count": self.correct_count,
            "accuracy": self.accuracy,
            "by_category": self.by_category,
            "by_difficulty": self.by_difficulty,
            "confidence_calibration": self.confidence_calibration,
            "knowledge_gaps": self.knowledge_gaps,
        }


class ClaimExtractor:
    """Extracts factual claims from text using pattern-based analysis.

    Uses regular expression patterns to identify and extract structured factual
    claims from natural language text. Supports extraction of factual, temporal,
    and relational claims with confidence scoring based on linguistic features.

    The extractor identifies subject-predicate-object triples and assigns
    confidence scores based on factors like proper nouns, specific numbers,
    hedging language, and subjective expressions.

    Class Attributes:
        FACTUAL_PATTERNS: Regex patterns for extracting factual statements.
        TEMPORAL_PATTERNS: Regex patterns for extracting time-related claims.
        RELATIONAL_PATTERNS: Regex patterns for extracting relationship claims.

    Example: Basic Claim Extraction
        >>> from insideLLMs.contrib.knowledge import ClaimExtractor
        >>> extractor = ClaimExtractor()
        >>> text = "Albert Einstein was a physicist. He won the Nobel Prize in 1921."
        >>> claims = extractor.extract(text)
        >>> for claim in claims:
        ...     print(f"Claim: {claim.text}")
        ...     print(f"Category: {claim.category.value}")
        ...     print(f"Confidence: {claim.confidence:.2f}")

    Example: Filtering by Confidence Threshold
        >>> # Only extract high-confidence claims
        >>> high_conf_claims = extractor.extract(text, min_confidence=0.6)
        >>> # Extract all possible claims including uncertain ones
        >>> all_claims = extractor.extract(text, min_confidence=0.1)

    Example: Using the Convenience Function
        >>> from insideLLMs.contrib.knowledge import extract_claims
        >>> claims = extract_claims("Marie Curie discovered radium.")
        >>> print(f"Found {len(claims)} claims")

    Example: Processing Multiple Documents
        >>> documents = ["Text 1...", "Text 2...", "Text 3..."]
        >>> extractor = ClaimExtractor()
        >>> all_claims = []
        >>> for doc in documents:
        ...     claims = extractor.extract(doc)
        ...     all_claims.extend(claims)
        >>> print(f"Total claims: {len(all_claims)}")
    """

    # Patterns for different claim types
    FACTUAL_PATTERNS = [
        r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|was|are|were)\s+(?P<predicate>(?:a|an|the)\s+)?(?P<object>[^.!?]+)",
        r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?P<predicate>has|have|had)\s+(?P<object>[^.!?]+)",
        r"(?:The|A|An)\s+(?P<subject>[a-z]+(?:\s+[a-z]+)*)\s+of\s+(?P<object>[^.!?]+)\s+(?:is|was)\s+(?P<predicate>[^.!?]+)",
    ]

    TEMPORAL_PATTERNS = [
        r"[Ii]n\s+(?P<time>\d{4}),?\s+(?P<claim>[^.!?]+)",
        r"(?P<claim>[^.!?]+)\s+(?:in|on|during)\s+(?P<time>\d{4}|\w+\s+\d{1,2},?\s+\d{4})",
    ]

    RELATIONAL_PATTERNS = [
        r"(?P<entity1>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:and|with)\s+(?P<entity2>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?P<relation>[^.!?]+)",
    ]

    def extract(
        self,
        text: str,
        min_confidence: float = 0.3,
    ) -> list[Claim]:
        """Extract factual claims from text with confidence filtering.

        Analyzes the input text using pattern matching to identify factual
        assertions. Each identified claim is assigned a confidence score
        based on linguistic features, and claims below the minimum confidence
        threshold are filtered out.

        Args:
            text: The input text to extract claims from.
            min_confidence: Minimum confidence threshold for including claims.
                Claims with confidence below this value are excluded.
                Default is 0.3.

        Returns:
            list[Claim]: List of extracted Claim objects, deduplicated by text,
                with confidence scores at or above the threshold.

        Example: Basic Extraction
            >>> extractor = ClaimExtractor()
            >>> text = "Paris is the capital of France."
            >>> claims = extractor.extract(text)
            >>> print(claims[0].text if claims else "No claims found")

        Example: High Confidence Only
            >>> claims = extractor.extract(text, min_confidence=0.7)
            >>> print(f"High confidence claims: {len(claims)}")

        Example: Processing Scientific Text
            >>> science_text = "In 1905, Einstein published his theory of relativity."
            >>> claims = extractor.extract(science_text)
            >>> temporal = [c for c in claims if c.category.value == "temporal"]
            >>> print(f"Found {len(temporal)} temporal claims")

        Example: Analyzing Extraction Results
            >>> claims = extractor.extract(long_text)
            >>> avg_confidence = sum(c.confidence for c in claims) / len(claims)
            >>> print(f"Average confidence: {avg_confidence:.2f}")
        """
        claims = []

        # Split into sentences
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue

            # Try to extract factual claims
            for pattern in self.FACTUAL_PATTERNS:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    groups = match.groupdict()
                    subject = groups.get("subject")
                    predicate = groups.get("predicate")
                    obj = groups.get("object")
                    claim = Claim(
                        text=sentence,
                        subject=subject.strip() if subject else None,
                        predicate=predicate.strip() if predicate else None,
                        object=obj.strip() if obj else None,
                        category=KnowledgeCategory.FACTUAL,
                        confidence=self._estimate_confidence(sentence, groups),
                    )
                    if claim.confidence >= min_confidence:
                        claims.append(claim)

            # Try temporal patterns
            for pattern in self.TEMPORAL_PATTERNS:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    groups = match.groupdict()
                    claim = Claim(
                        text=sentence,
                        category=KnowledgeCategory.TEMPORAL,
                        confidence=self._estimate_confidence(sentence, groups),
                    )
                    if claim.confidence >= min_confidence:
                        claims.append(claim)

        # Deduplicate by text
        seen = set()
        unique_claims = []
        for claim in claims:
            if claim.text not in seen:
                seen.add(claim.text)
                unique_claims.append(claim)

        return unique_claims

    def _estimate_confidence(
        self,
        sentence: str,
        groups: dict[str, str],
    ) -> float:
        """Estimate confidence in an extracted claim based on linguistic features.

        Analyzes the sentence and extracted groups to assign a confidence score.
        Confidence is boosted for specific entities and numbers, and reduced
        for hedging language and subjective expressions.

        Args:
            sentence: The full sentence containing the claim.
            groups: Dictionary of named groups from regex matching (subject,
                predicate, object, etc.).

        Returns:
            float: Confidence score between 0.0 and 1.0.

        Confidence Adjustments:
            - +0.1 for proper noun subjects (capitalized)
            - +0.1 for presence of numbers/dates
            - -0.15 for hedging words (maybe, perhaps, possibly, etc.)
            - -0.2 for subjective language (think, believe, feel, opinion)

        Example: High Confidence Claim
            >>> extractor = ClaimExtractor()
            >>> # Proper noun + number = high confidence
            >>> conf = extractor._estimate_confidence(
            ...     "Einstein won in 1921",
            ...     {"subject": "Einstein"}
            ... )
            >>> print(f"Confidence: {conf:.2f}")  # ~0.7

        Example: Low Confidence Claim
            >>> # Hedging language reduces confidence
            >>> conf = extractor._estimate_confidence(
            ...     "Maybe the answer is 42",
            ...     {"subject": "answer"}
            ... )
            >>> print(f"Confidence: {conf:.2f}")  # ~0.35

        Example: Subjective Language
            >>> conf = extractor._estimate_confidence(
            ...     "I believe Einstein was brilliant",
            ...     {"subject": "Einstein"}
            ... )
            >>> print(f"Confidence: {conf:.2f}")  # Lower due to "believe"
        """
        confidence = 0.5

        # Boost for specific entities
        if groups.get("subject") and groups["subject"][0].isupper():
            confidence += 0.1

        # Boost for numbers/dates
        if re.search(r"\d+", sentence):
            confidence += 0.1

        # Reduce for hedging language
        hedges = ["maybe", "perhaps", "possibly", "might", "could", "seems"]
        for hedge in hedges:
            if hedge in sentence.lower():
                confidence -= 0.15

        # Reduce for subjective language
        subjective = ["think", "believe", "feel", "opinion"]
        for subj in subjective:
            if subj in sentence.lower():
                confidence -= 0.2

        return max(0.0, min(1.0, confidence))


class FactVerifier:
    """Verifies factual claims against a knowledge base.

    Checks claims against a provided knowledge base to determine their
    verification status. Collects supporting and contradicting evidence,
    assesses source quality, and produces VerificationResult objects
    with confidence scores.

    The verifier uses keyword matching to find relevant knowledge base
    entries and evaluates claim consistency with any provided context.

    Attributes:
        knowledge_base: Dictionary mapping topics/entities to their known facts.

    Example: Basic Fact Verification
        >>> from insideLLMs.contrib.knowledge import FactVerifier, Claim
        >>> kb = {
        ...     "Paris": "capital of France",
        ...     "Einstein": "physicist who developed relativity"
        ... }
        >>> verifier = FactVerifier(knowledge_base=kb)
        >>> claim = Claim(text="Paris is the capital of France")
        >>> result = verifier.verify(claim)
        >>> print(f"Status: {result.status.value}")

    Example: Batch Verification
        >>> claims = [
        ...     Claim(text="Einstein was a physicist"),
        ...     Claim(text="Paris is in Germany")
        ... ]
        >>> results = verifier.verify_batch(claims)
        >>> for r in results:
        ...     print(f"{r.claim.text}: {r.status.value}")

    Example: Using the Convenience Function
        >>> from insideLLMs.contrib.knowledge import verify_claim
        >>> result = verify_claim(
        ...     Claim(text="Water boils at 100 degrees"),
        ...     knowledge_base={"Water": "boils at 100 degrees Celsius"}
        ... )

    Example: Context-Aware Verification
        >>> context = "The document discusses French geography and capitals."
        >>> result = verifier.verify(claim, context=context)
        >>> # Context helps assess claim consistency
    """

    def __init__(
        self,
        knowledge_base: Optional[dict[str, Any]] = None,
    ):
        """Initialize the FactVerifier with an optional knowledge base.

        Args:
            knowledge_base: Dictionary mapping topics/entities to their known
                facts. Keys are typically entity names or topics, values are
                descriptions or facts about them. Default is an empty dict.

        Example: Empty Verifier
            >>> verifier = FactVerifier()
            >>> # Without a knowledge base, all claims will be "unverifiable"

        Example: With Knowledge Base
            >>> kb = {"Python": "programming language created by Guido van Rossum"}
            >>> verifier = FactVerifier(knowledge_base=kb)

        Example: Comprehensive Knowledge Base
            >>> kb = {
            ...     "Einstein": "physicist, Nobel Prize 1921, relativity",
            ...     "Curie": "physicist, chemist, Nobel Prizes in physics and chemistry",
            ...     "Newton": "physicist, mathematician, laws of motion and gravity"
            ... }
            >>> verifier = FactVerifier(knowledge_base=kb)
        """
        self.knowledge_base = knowledge_base or {}

    def verify(
        self,
        claim: Claim,
        context: Optional[str] = None,
    ) -> VerificationResult:
        """Verify a single claim against the knowledge base.

        Checks the claim against the knowledge base to find supporting or
        contradicting evidence. Also evaluates consistency with any provided
        context. Returns a VerificationResult with the verification status,
        confidence, and collected evidence.

        Args:
            claim: The Claim object to verify.
            context: Optional context string for consistency checking.
                Can be surrounding text or related information.

        Returns:
            VerificationResult: Result containing verification status,
                confidence score, supporting/contradicting evidence, and
                source quality assessment.

        Verification Logic:
            - VERIFIED: Supporting evidence found, no contradictions
            - CONTRADICTED: Contradicting evidence found, no support
            - PARTIAL: Both supporting and contradicting evidence found
            - UNVERIFIABLE: No relevant evidence found

        Example: Simple Verification
            >>> verifier = FactVerifier({"Paris": "capital of France"})
            >>> claim = Claim(text="Paris is the capital of France")
            >>> result = verifier.verify(claim)
            >>> print(result.status.value)  # "verified"

        Example: With Context
            >>> context = "This article discusses European capitals."
            >>> result = verifier.verify(claim, context=context)
            >>> # Context is used for consistency checking

        Example: Examining Evidence
            >>> result = verifier.verify(claim)
            >>> if result.supporting_evidence:
            ...     print("Supported by:", result.supporting_evidence)
            >>> if result.contradicting_evidence:
            ...     print("Contradicted by:", result.contradicting_evidence)

        Example: Handling Unverifiable Claims
            >>> claim = Claim(text="Aliens exist on Mars")
            >>> result = verifier.verify(claim)
            >>> if result.status.value == "unverifiable":
            ...     print("Insufficient evidence to verify")
        """
        supporting = []
        contradicting = []

        # Check against knowledge base
        if self.knowledge_base:
            kb_result = self._check_knowledge_base(claim)
            if kb_result:
                if kb_result["supports"]:
                    supporting.extend(kb_result["evidence"])
                else:
                    contradicting.extend(kb_result["evidence"])

        # Check internal consistency
        self._check_consistency(claim, context)

        # Determine status
        if supporting and not contradicting:
            status = VerificationStatus.VERIFIED
            confidence = min(0.9, 0.5 + len(supporting) * 0.1)
        elif contradicting and not supporting:
            status = VerificationStatus.CONTRADICTED
            confidence = min(0.9, 0.5 + len(contradicting) * 0.1)
        elif supporting and contradicting:
            status = VerificationStatus.PARTIAL
            confidence = 0.5
        else:
            status = VerificationStatus.UNVERIFIABLE
            confidence = 0.3

        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            source_quality=self._assess_source_quality(supporting + contradicting),
        )

    def verify_batch(
        self,
        claims: list[Claim],
        context: Optional[str] = None,
    ) -> list[VerificationResult]:
        """Verify multiple claims in batch.

        Processes a list of claims, verifying each against the knowledge base.
        This is a convenience method that applies the same context to all claims.

        Args:
            claims: List of Claim objects to verify.
            context: Optional context string applied to all verifications.

        Returns:
            list[VerificationResult]: List of verification results in the same
                order as the input claims.

        Example: Batch Verification
            >>> claims = [
            ...     Claim(text="Paris is in France"),
            ...     Claim(text="London is in England"),
            ...     Claim(text="Berlin is in Italy")
            ... ]
            >>> results = verifier.verify_batch(claims)
            >>> for claim, result in zip(claims, results):
            ...     print(f"{claim.text}: {result.status.value}")

        Example: With Shared Context
            >>> context = "European geography facts"
            >>> results = verifier.verify_batch(claims, context=context)

        Example: Aggregating Results
            >>> results = verifier.verify_batch(claims)
            >>> verified = sum(1 for r in results if r.status.value == "verified")
            >>> print(f"Verified: {verified}/{len(results)}")

        Example: Processing Extracted Claims
            >>> text = "Einstein was a physicist. Curie won two Nobel Prizes."
            >>> claims = extract_claims(text)
            >>> results = verifier.verify_batch(claims)
        """
        return [self.verify(claim, context) for claim in claims]

    def _check_knowledge_base(
        self,
        claim: Claim,
    ) -> Optional[dict[str, Any]]:
        """Check a claim against the knowledge base using keyword matching.

        Searches the knowledge base for entries with keywords that overlap
        with the claim text. Returns evidence information if a match is found.

        Args:
            claim: The Claim to check against the knowledge base.

        Returns:
            Optional[dict[str, Any]]: Dictionary with 'supports' (bool) and
                'evidence' (list of strings) if a match is found, None otherwise.

        Example: Matching Entry
            >>> # With kb = {"Paris": "capital of France"}
            >>> result = verifier._check_knowledge_base(
            ...     Claim(text="Paris is beautiful")
            ... )
            >>> # Returns {"supports": True, "evidence": [...]}

        Example: No Match
            >>> result = verifier._check_knowledge_base(
            ...     Claim(text="Tokyo is in Japan")
            ... )
            >>> # Returns None if "Tokyo" not in knowledge base
        """
        if not self.knowledge_base:
            return None

        # Simple keyword matching
        claim_words = set(claim.text.lower().split())

        for key, value in self.knowledge_base.items():
            key_words = set(key.lower().split())
            if claim_words & key_words:
                return {
                    "supports": True,
                    "evidence": [f"Knowledge base entry: {key} -> {value}"],
                }

        return None

    def _check_consistency(
        self,
        claim: Claim,
        context: Optional[str],
    ) -> float:
        """Check internal consistency between a claim and its context.

        Calculates word overlap between the claim and context to assess
        how consistent the claim is with the surrounding information.

        Args:
            claim: The Claim to check for consistency.
            context: Context string to compare against. If None, returns 0.5.

        Returns:
            float: Consistency score from 0.0 (no overlap) to 1.0 (complete overlap).
                Returns 0.5 if no context is provided.

        Example: High Consistency
            >>> claim = Claim(text="Paris is the capital of France")
            >>> context = "France is a country with Paris as its capital city"
            >>> score = verifier._check_consistency(claim, context)
            >>> # Higher score due to word overlap

        Example: No Context
            >>> score = verifier._check_consistency(claim, None)
            >>> # Returns 0.5 (neutral)
        """
        if not context:
            return 0.5

        # Simple word overlap check
        claim_words = set(claim.text.lower().split())
        context_words = set(context.lower().split())

        overlap = len(claim_words & context_words)
        total = len(claim_words)

        return overlap / total if total > 0 else 0.5

    def _assess_source_quality(
        self,
        evidence: list[str],
    ) -> float:
        """Assess the quality of evidence sources.

        Uses a simple heuristic based on the amount of evidence available.
        More evidence generally indicates higher quality/reliability.

        Args:
            evidence: List of evidence strings (supporting or contradicting).

        Returns:
            float: Quality score from 0.0 (no evidence) to 1.0 (high quality).
                Calculated as min(1.0, len(evidence) * 0.2).

        Example: No Evidence
            >>> score = verifier._assess_source_quality([])
            >>> # Returns 0.0

        Example: Multiple Sources
            >>> evidence = ["Source 1", "Source 2", "Source 3"]
            >>> score = verifier._assess_source_quality(evidence)
            >>> # Returns 0.6 (3 * 0.2)
        """
        if not evidence:
            return 0.0

        # Simple heuristic based on evidence count
        return min(1.0, len(evidence) * 0.2)


class KnowledgeProber:
    """Probes model knowledge through targeted questioning and evaluation.

    Manages a collection of knowledge probes, executes them against a model
    function, evaluates responses for correctness, and generates comprehensive
    reports on model knowledge. Supports categorization by knowledge type and
    difficulty, confidence calibration analysis, and knowledge gap identification.

    Attributes:
        probes: List of KnowledgeProbe objects registered for testing.

    Example: Basic Knowledge Probing
        >>> from insideLLMs.contrib.knowledge import KnowledgeProber, KnowledgeCategory
        >>> prober = KnowledgeProber()
        >>> prober.add_probe(
        ...     question="What is the capital of France?",
        ...     expected_answer="Paris",
        ...     category=KnowledgeCategory.FACTUAL,
        ...     difficulty=0.2
        ... )
        >>> def model(q): return "The capital of France is Paris."
        >>> results = prober.run_all_probes(model)
        >>> print(f"Correct: {results[0].is_correct}")

    Example: Multi-Category Evaluation
        >>> prober = KnowledgeProber()
        >>> prober.add_probe("What is 2+2?", "4", KnowledgeCategory.FACTUAL, 0.1)
        >>> prober.add_probe("When was WWII?", "1939-1945", KnowledgeCategory.TEMPORAL, 0.5)
        >>> prober.add_probe("Why is the sky blue?", "light scattering", KnowledgeCategory.CONCEPTUAL, 0.7)
        >>> results = prober.run_all_probes(model_fn)
        >>> report = prober.generate_report(results)
        >>> print(report.by_category)

    Example: Using Hints
        >>> probe = prober.add_probe(
        ...     question="Who discovered penicillin?",
        ...     expected_answer="Alexander Fleming",
        ...     hints=["Scottish scientist", "1928"]
        ... )
        >>> result = prober.run_probe(probe, model_fn, include_hints=True)

    Example: Complete Evaluation Pipeline
        >>> prober = KnowledgeProber()
        >>> # Add many probes...
        >>> results = prober.run_all_probes(model_fn)
        >>> report = prober.generate_report(results)
        >>> print(f"Accuracy: {report.accuracy:.2%}")
        >>> print(f"Calibration: {report.confidence_calibration:.2f}")
        >>> print(f"Knowledge gaps: {report.knowledge_gaps}")
    """

    def __init__(self):
        """Initialize the KnowledgeProber with an empty probe list.

        Example: Creating a New Prober
            >>> prober = KnowledgeProber()
            >>> print(len(prober.probes))  # 0

        Example: Ready for Probe Registration
            >>> prober = KnowledgeProber()
            >>> prober.add_probe("What is Python?", "programming language")
            >>> print(len(prober.probes))  # 1
        """
        self.probes: list[KnowledgeProbe] = []

    def add_probe(
        self,
        question: str,
        expected_answer: Optional[str] = None,
        category: KnowledgeCategory = KnowledgeCategory.FACTUAL,
        difficulty: float = 0.5,
        hints: Optional[list[str]] = None,
    ) -> KnowledgeProbe:
        """Add a knowledge probe to the prober's collection.

        Creates a new KnowledgeProbe and registers it for later execution.
        Returns the created probe for reference or immediate use.

        Args:
            question: The question to ask the model.
            expected_answer: The expected correct answer for evaluation.
                If None, correctness cannot be assessed.
            category: The type of knowledge being tested.
                Default is KnowledgeCategory.FACTUAL.
            difficulty: Difficulty level from 0.0 (easy) to 1.0 (hard).
                Default is 0.5 (medium).
            hints: Optional list of hints that can be provided to the model.

        Returns:
            KnowledgeProbe: The created and registered probe object.

        Example: Simple Factual Probe
            >>> prober = KnowledgeProber()
            >>> probe = prober.add_probe(
            ...     question="What is the speed of light?",
            ...     expected_answer="299792458 m/s"
            ... )
            >>> print(probe.question)

        Example: Categorized Probe with Difficulty
            >>> probe = prober.add_probe(
            ...     "When did the Roman Empire fall?",
            ...     "476 AD",
            ...     KnowledgeCategory.TEMPORAL,
            ...     difficulty=0.6
            ... )

        Example: Probe with Hints
            >>> probe = prober.add_probe(
            ...     "Who painted the Mona Lisa?",
            ...     "Leonardo da Vinci",
            ...     hints=["Italian Renaissance", "Florentine artist"]
            ... )

        Example: Building a Probe Set
            >>> questions = [("Q1?", "A1"), ("Q2?", "A2"), ("Q3?", "A3")]
            >>> for q, a in questions:
            ...     prober.add_probe(q, a)
            >>> print(f"Total probes: {len(prober.probes)}")
        """
        probe = KnowledgeProbe(
            question=question,
            expected_answer=expected_answer,
            category=category,
            difficulty=difficulty,
            hints=hints or [],
        )
        self.probes.append(probe)
        return probe

    def run_probe(
        self,
        probe: KnowledgeProbe,
        model_fn: Callable[[str], str],
        include_hints: bool = False,
    ) -> ProbeResult:
        """Run a single knowledge probe against a model.

        Executes the probe by calling the model function with the question,
        optionally including hints. Evaluates the response for correctness,
        partial credit, expressed confidence, and reasoning presence.

        Args:
            probe: The KnowledgeProbe to execute.
            model_fn: A callable that takes a question string and returns
                the model's response string.
            include_hints: Whether to append hints to the question prompt.
                Default is False.

        Returns:
            ProbeResult: Result containing the response, correctness assessment,
                partial score, confidence level, and reasoning detection.

        Example: Basic Probe Execution
            >>> prober = KnowledgeProber()
            >>> probe = prober.add_probe("What is 2+2?", "4")
            >>> def model(q): return "The answer is 4."
            >>> result = prober.run_probe(probe, model)
            >>> print(f"Correct: {result.is_correct}")  # True

        Example: With Hints
            >>> probe = prober.add_probe(
            ...     "Who discovered gravity?",
            ...     "Isaac Newton",
            ...     hints=["Apple falling", "17th century"]
            ... )
            >>> result = prober.run_probe(probe, model, include_hints=True)
            >>> # Model receives: "Who discovered gravity?\\n\\nHints: Apple falling, 17th century"

        Example: Analyzing Response Quality
            >>> result = prober.run_probe(probe, model)
            >>> print(f"Partial score: {result.partial_score:.2f}")
            >>> print(f"Confidence: {result.confidence_expressed:.2f}")
            >>> print(f"Has reasoning: {result.reasoning_provided}")

        Example: Testing Different Models
            >>> def model_a(q): return "4"
            >>> def model_b(q): return "I think it's 4 because 2 plus 2 equals 4."
            >>> result_a = prober.run_probe(probe, model_a)
            >>> result_b = prober.run_probe(probe, model_b)
            >>> # model_b shows reasoning, model_a doesn't
        """
        # Build prompt
        prompt = probe.question
        if include_hints and probe.hints:
            prompt += "\n\nHints: " + ", ".join(probe.hints)

        # Get response
        response = model_fn(prompt)

        # Evaluate
        is_correct = None
        partial_score = 0.0

        if probe.expected_answer:
            is_correct = self._check_answer(response, probe.expected_answer)
            partial_score = self._calculate_partial_score(response, probe.expected_answer)

        # Check for confidence expression
        confidence_expressed = self._detect_confidence(response)

        # Check for reasoning
        reasoning_provided = self._detect_reasoning(response)

        return ProbeResult(
            probe=probe,
            response=response,
            is_correct=is_correct,
            partial_score=partial_score,
            confidence_expressed=confidence_expressed,
            reasoning_provided=reasoning_provided,
        )

    def run_all_probes(
        self,
        model_fn: Callable[[str], str],
    ) -> list[ProbeResult]:
        """Run all registered probes against a model.

        Executes every probe in the prober's collection and returns all results.
        This is a convenience method for running complete evaluation sessions.

        Args:
            model_fn: A callable that takes a question string and returns
                the model's response string.

        Returns:
            list[ProbeResult]: List of results in the same order as the probes.

        Example: Running Full Evaluation
            >>> prober = KnowledgeProber()
            >>> prober.add_probe("Q1?", "A1")
            >>> prober.add_probe("Q2?", "A2")
            >>> prober.add_probe("Q3?", "A3")
            >>> results = prober.run_all_probes(model_fn)
            >>> print(f"Ran {len(results)} probes")

        Example: With Report Generation
            >>> results = prober.run_all_probes(model_fn)
            >>> report = prober.generate_report(results)
            >>> print(f"Accuracy: {report.accuracy:.2%}")

        Example: Comparing Models
            >>> results_a = prober.run_all_probes(model_a)
            >>> results_b = prober.run_all_probes(model_b)
            >>> accuracy_a = sum(r.is_correct for r in results_a if r.is_correct is not None)
            >>> accuracy_b = sum(r.is_correct for r in results_b if r.is_correct is not None)
        """
        return [self.run_probe(probe, model_fn) for probe in self.probes]

    def generate_report(
        self,
        results: list[ProbeResult],
    ) -> KnowledgeReport:
        """Generate a comprehensive report from probe results.

        Analyzes all probe results to compute overall accuracy, category-specific
        performance, difficulty-based breakdown, confidence calibration, and
        identified knowledge gaps.

        Args:
            results: List of ProbeResult objects from running probes.

        Returns:
            KnowledgeReport: Comprehensive report with all computed metrics.

        Report Metrics:
            - accuracy: Overall fraction of correct answers
            - by_category: Accuracy per knowledge category
            - by_difficulty: Accuracy for easy/medium/hard questions
            - confidence_calibration: How well confidence matches correctness
            - knowledge_gaps: Categories with poor performance

        Example: Basic Report
            >>> results = prober.run_all_probes(model_fn)
            >>> report = prober.generate_report(results)
            >>> print(f"Accuracy: {report.accuracy:.2%}")
            >>> print(f"Correct: {report.correct_count}/{report.probes_run}")

        Example: Category Analysis
            >>> report = prober.generate_report(results)
            >>> for cat, acc in report.by_category.items():
            ...     print(f"{cat}: {acc:.2%}")
            >>> if report.knowledge_gaps:
            ...     print(f"Weak areas: {report.knowledge_gaps}")

        Example: Difficulty Analysis
            >>> report = prober.generate_report(results)
            >>> print(f"Easy: {report.by_difficulty.get('easy', 0):.2%}")
            >>> print(f"Medium: {report.by_difficulty.get('medium', 0):.2%}")
            >>> print(f"Hard: {report.by_difficulty.get('hard', 0):.2%}")

        Example: Calibration Assessment
            >>> report = prober.generate_report(results)
            >>> if report.confidence_calibration > 0.8:
            ...     print("Model is well-calibrated")
            >>> else:
            ...     print("Model confidence needs improvement")
        """
        if not results:
            return KnowledgeReport(
                probes_run=0,
                correct_count=0,
                accuracy=0.0,
                by_category={},
                by_difficulty={},
                confidence_calibration=0.0,
            )

        # Calculate accuracy
        correct = sum(1 for r in results if r.is_correct)
        total = len(results)
        accuracy = correct / total if total > 0 else 0.0

        # By category
        by_category: dict[str, list[bool]] = {}
        for result in results:
            cat = result.probe.category.value
            if cat not in by_category:
                by_category[cat] = []
            if result.is_correct is not None:
                by_category[cat].append(result.is_correct)

        category_accuracy = {
            cat: sum(scores) / len(scores) if scores else 0.0 for cat, scores in by_category.items()
        }

        # By difficulty
        difficulty_bins = {"easy": [], "medium": [], "hard": []}
        for result in results:
            diff = result.probe.difficulty
            if diff < 0.33:
                bin_name = "easy"
            elif diff < 0.67:
                bin_name = "medium"
            else:
                bin_name = "hard"
            if result.is_correct is not None:
                difficulty_bins[bin_name].append(result.is_correct)

        difficulty_accuracy = {
            bin_name: sum(scores) / len(scores) if scores else 0.0
            for bin_name, scores in difficulty_bins.items()
        }

        # Confidence calibration
        calibration = self._calculate_calibration(results)

        # Identify knowledge gaps
        gaps = []
        for result in results:
            if result.is_correct is False:
                gaps.append(result.probe.category.value)

        return KnowledgeReport(
            probes_run=total,
            correct_count=correct,
            accuracy=accuracy,
            by_category=category_accuracy,
            by_difficulty=difficulty_accuracy,
            confidence_calibration=calibration,
            knowledge_gaps=list(set(gaps)),
        )

    def _check_answer(
        self,
        response: str,
        expected: str,
    ) -> bool:
        """Check if a response contains the expected answer.

        Uses two matching strategies: exact substring match and word-level
        overlap (requiring 80% of expected words to be present).

        Args:
            response: The model's response string.
            expected: The expected answer string.

        Returns:
            bool: True if the response contains the expected answer
                (exact match or 80%+ word overlap), False otherwise.

        Example: Exact Match
            >>> prober._check_answer("The answer is Paris", "Paris")
            True

        Example: Word Overlap
            >>> prober._check_answer("Newton discovered gravity", "Isaac Newton")
            # True if "Newton" present (1/2 = 50% < 80%, but "Newton" exact match)

        Example: No Match
            >>> prober._check_answer("London is great", "Paris")
            False
        """
        response_lower = response.lower()
        expected_lower = expected.lower()

        # Exact match
        if expected_lower in response_lower:
            return True

        # Word-level match
        expected_words = set(expected_lower.split())
        response_words = set(response_lower.split())

        overlap = len(expected_words & response_words)
        return overlap >= len(expected_words) * 0.8

    def _calculate_partial_score(
        self,
        response: str,
        expected: str,
    ) -> float:
        """Calculate partial match score based on word overlap.

        Computes the fraction of expected answer words that appear in the
        response. Useful for giving partial credit when the answer is
        partially correct.

        Args:
            response: The model's response string.
            expected: The expected answer string.

        Returns:
            float: Score from 0.0 (no overlap) to 1.0 (complete overlap).

        Example: Full Match
            >>> prober._calculate_partial_score("Paris is the capital", "Paris")
            1.0  # "Paris" fully present

        Example: Partial Match
            >>> prober._calculate_partial_score("Newton was smart", "Isaac Newton")
            0.5  # "Newton" present, "Isaac" missing

        Example: No Match
            >>> prober._calculate_partial_score("The sky is blue", "Paris France")
            0.0
        """
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            return 0.0

        overlap = len(response_words & expected_words)
        return overlap / len(expected_words)

    def _detect_confidence(self, response: str) -> float:
        """Detect the confidence level expressed in a response.

        Analyzes the response for linguistic markers of confidence or
        uncertainty. Returns a score indicating the expressed confidence level.

        Args:
            response: The model's response string.

        Returns:
            float: Confidence score:
                - 0.3 for low confidence (hedging words)
                - 0.5 for neutral (no confidence markers)
                - 0.6 for medium confidence (probabilistic words)
                - 0.9 for high confidence (certainty words)

        Confidence Markers:
            - Low: "not sure", "maybe", "perhaps", "possibly", "might be", etc.
            - Medium: "probably", "likely", "think", "believe"
            - High: "certainly", "definitely", "absolutely", "I'm sure"

        Example: High Confidence
            >>> prober._detect_confidence("I am absolutely certain it's Paris")
            0.9

        Example: Low Confidence
            >>> prober._detect_confidence("I'm not sure, but maybe Paris?")
            0.3

        Example: Medium Confidence
            >>> prober._detect_confidence("I think the answer is probably Paris")
            0.6
        """
        # Check for uncertainty markers first (they override high confidence words)
        low_confidence = [
            "not sure",
            "maybe",
            "perhaps",
            "possibly",
            "might be",
            "could be",
            "uncertain",
        ]
        medium_confidence = ["probably", "likely", "think", "believe"]
        high_confidence = ["certainly", "definitely", "absolutely", "i'm sure", "i am sure"]

        response_lower = response.lower()

        # Check low confidence first (takes precedence)
        for phrase in low_confidence:
            if phrase in response_lower:
                return 0.3

        # Then check high confidence
        for word in high_confidence:
            if word in response_lower:
                return 0.9

        # Then medium confidence
        for word in medium_confidence:
            if word in response_lower:
                return 0.6

        return 0.5

    def _detect_reasoning(self, response: str) -> bool:
        """Detect if the response contains reasoning or explanation.

        Checks for common reasoning indicators like causal words and
        explanation phrases. Used to assess whether the model provides
        justification for its answers.

        Args:
            response: The model's response string.

        Returns:
            bool: True if reasoning indicators are found, False otherwise.

        Reasoning Indicators:
            - "because", "therefore", "since"
            - "as a result", "this is because", "the reason"
            - "due to", "consequently"

        Example: With Reasoning
            >>> prober._detect_reasoning("It's Paris because Paris is the capital")
            True

        Example: Without Reasoning
            >>> prober._detect_reasoning("The answer is Paris")
            False

        Example: Complex Reasoning
            >>> prober._detect_reasoning("Since France is a country, therefore Paris...")
            True
        """
        reasoning_indicators = [
            "because",
            "therefore",
            "since",
            "as a result",
            "this is because",
            "the reason",
            "due to",
            "consequently",
        ]

        response_lower = response.lower()
        return any(ind in response_lower for ind in reasoning_indicators)

    def _calculate_calibration(
        self,
        results: list[ProbeResult],
    ) -> float:
        """Calculate confidence calibration score using Expected Calibration Error.

        Measures how well the model's expressed confidence aligns with actual
        correctness. A well-calibrated model is correct about 90% of the time
        when expressing high confidence, 55% for medium, and 30% for low.

        Args:
            results: List of ProbeResult objects to analyze.

        Returns:
            float: Calibration score from 0.0 (poorly calibrated) to 1.0
                (perfectly calibrated). Computed as 1 - ECE.

        Calibration Logic:
            - Groups results into low (<0.4), medium (0.4-0.7), high (>0.7) bins
            - Compares actual accuracy in each bin to expected accuracy
            - Computes weighted Expected Calibration Error (ECE)
            - Returns 1 - ECE as final score

        Example: Well-Calibrated Model
            >>> # Model says "definitely" and is usually right
            >>> # Model says "maybe" and is often wrong
            >>> calibration = prober._calculate_calibration(results)
            >>> # High score (>0.8) indicates good calibration

        Example: Overconfident Model
            >>> # Model always says "definitely" but is often wrong
            >>> calibration = prober._calculate_calibration(results)
            >>> # Low score indicates poor calibration

        Example: Empty Results
            >>> calibration = prober._calculate_calibration([])
            >>> # Returns 0.0
        """
        if not results:
            return 0.0

        # Group by confidence bins
        bins: dict[str, list[bool]] = {
            "low": [],
            "medium": [],
            "high": [],
        }

        for result in results:
            if result.is_correct is None:
                continue

            conf = result.confidence_expressed
            if conf < 0.4:
                bins["low"].append(result.is_correct)
            elif conf < 0.7:
                bins["medium"].append(result.is_correct)
            else:
                bins["high"].append(result.is_correct)

        # Calculate ECE (Expected Calibration Error)
        ece = 0.0
        total = sum(len(b) for b in bins.values())

        if total == 0:
            return 0.5

        expected = {"low": 0.3, "medium": 0.55, "high": 0.85}

        for bin_name, scores in bins.items():
            if scores:
                actual = sum(scores) / len(scores)
                ece += len(scores) / total * abs(actual - expected[bin_name])

        # Convert to calibration score (1 - ECE)
        return max(0.0, 1.0 - ece)


class ConsistencyTester:
    """Tests consistency of model responses across question paraphrases.

    Evaluates whether a model gives consistent answers when the same question
    is asked in different ways. Generates paraphrased versions of questions,
    collects responses, and analyzes them for consistency, contradictions,
    and semantic drift.

    This is useful for detecting:
        - Hallucination (inconsistent "facts")
        - Sensitivity to prompt phrasing
        - Knowledge uncertainty

    Example: Basic Consistency Testing
        >>> from insideLLMs.contrib.knowledge import ConsistencyTester
        >>> tester = ConsistencyTester()
        >>> def model(q): return "The capital of France is Paris."
        >>> result = tester.test_consistency(
        ...     "What is the capital of France?",
        ...     model,
        ...     num_variations=3
        ... )
        >>> print(f"Consistency: {result.consistency_score:.2f}")

    Example: Using the Convenience Function
        >>> from insideLLMs.contrib.knowledge import check_consistency
        >>> result = check_consistency("What is 2+2?", model_fn, num_variations=5)
        >>> print(f"Found {len(result.contradictions)} contradictions")

    Example: Detecting Inconsistent Models
        >>> import random
        >>> def unstable_model(q):
        ...     return random.choice(["Paris", "London", "Berlin"])
        >>> result = tester.test_consistency("Capital of France?", unstable_model)
        >>> if result.consistency_score < 0.5:
        ...     print("Warning: Model is inconsistent!")

    Example: Analyzing Semantic Drift
        >>> result = tester.test_consistency(question, model)
        >>> if result.semantic_drift > 0.3:
        ...     print("Responses diverge significantly from original")
    """

    def __init__(self):
        """Initialize the ConsistencyTester.

        Example: Creating a Tester
            >>> tester = ConsistencyTester()
            >>> # Ready to test consistency
        """
        pass

    def generate_paraphrases(
        self,
        question: str,
        num_paraphrases: int = 3,
    ) -> list[str]:
        """Generate paraphrased versions of a question.

        Uses multiple paraphrasing strategies to create variations of the
        original question while preserving its meaning. Strategies include
        rephrasing question words, adding context, simplifying, and elaborating.

        Args:
            question: The original question to paraphrase.
            num_paraphrases: Number of paraphrases to generate. Default is 3.

        Returns:
            list[str]: List of paraphrased questions. May be fewer than
                requested if strategies produce identical results.

        Paraphrasing Strategies:
            - Rephrase question words (what -> explain, who -> tell me about)
            - Add context prefixes ("I'm curious about this: ...")
            - Simplify by removing filler words
            - Elaborate with suffix requests

        Example: Basic Paraphrasing
            >>> tester = ConsistencyTester()
            >>> paraphrases = tester.generate_paraphrases(
            ...     "What is the capital of France?",
            ...     num_paraphrases=3
            ... )
            >>> for p in paraphrases:
            ...     print(p)

        Example: More Variations
            >>> paraphrases = tester.generate_paraphrases(question, num_paraphrases=5)
            >>> print(f"Generated {len(paraphrases)} variations")

        Example: Question Word Rephrasing
            >>> paraphrases = tester.generate_paraphrases("What is gravity?")
            >>> # May produce: "can you explain gravity?"
        """
        paraphrases = []

        # Simple paraphrase strategies
        strategies = [
            self._rephrase_question_word,
            self._add_context,
            self._simplify,
            self._elaborate,
        ]

        for i in range(num_paraphrases):
            strategy = strategies[i % len(strategies)]
            paraphrase = strategy(question)
            if paraphrase != question:
                paraphrases.append(paraphrase)

        return paraphrases

    def test_consistency(
        self,
        question: str,
        model_fn: Callable[[str], str],
        num_variations: int = 3,
    ) -> ConsistencyResult:
        """Test response consistency across paraphrased versions of a question.

        Generates paraphrases of the question, gets model responses for each,
        and analyzes them for consistency, contradictions, and semantic drift.

        Args:
            question: The original question to test.
            model_fn: A callable that takes a question string and returns
                the model's response string.
            num_variations: Number of paraphrased variations to test.
                Default is 3.

        Returns:
            ConsistencyResult: Contains the original response, paraphrased
                responses, consistency score, contradictions, and drift.

        Example: Basic Consistency Test
            >>> tester = ConsistencyTester()
            >>> def model(q): return "Paris"
            >>> result = tester.test_consistency("Capital of France?", model)
            >>> print(f"Score: {result.consistency_score:.2f}")

        Example: More Thorough Testing
            >>> result = tester.test_consistency(
            ...     "Who wrote Hamlet?",
            ...     model_fn,
            ...     num_variations=5
            ... )
            >>> print(f"Tested {len(result.paraphrased_responses)} variations")

        Example: Detecting Problems
            >>> result = tester.test_consistency(question, model)
            >>> if result.contradictions:
            ...     print(f"Found {len(result.contradictions)} contradictions!")
            >>> if result.semantic_drift > 0.5:
            ...     print("High semantic drift detected")

        Example: Quality Assessment
            >>> result = tester.test_consistency(question, model)
            >>> is_reliable = (
            ...     result.consistency_score > 0.8 and
            ...     len(result.contradictions) == 0
            ... )
        """
        # Get original response
        original_response = model_fn(question)

        # Generate and test paraphrases
        paraphrases = self.generate_paraphrases(question, num_variations)
        paraphrased_responses = [model_fn(p) for p in paraphrases]

        # Calculate consistency
        consistency_score = self._calculate_consistency(original_response, paraphrased_responses)

        # Find contradictions
        contradictions = self._find_contradictions(original_response, paraphrased_responses)

        # Calculate semantic drift
        drift = self._calculate_drift(original_response, paraphrased_responses)

        return ConsistencyResult(
            original_response=original_response,
            paraphrased_responses=paraphrased_responses,
            consistency_score=consistency_score,
            contradictions=contradictions,
            semantic_drift=drift,
        )

    def _rephrase_question_word(self, question: str) -> str:
        """Rephrase a question by replacing common question word patterns.

        Transforms question openers like "what is" to alternatives like
        "can you explain" to test if the model responds consistently
        regardless of phrasing.

        Args:
            question: The original question to rephrase.

        Returns:
            str: Rephrased question, or original if no patterns match.

        Replacements:
            - "what is" -> "can you explain"
            - "who is" -> "tell me about"
            - "where is" -> "what is the location of"
            - "when did" -> "at what time did"
            - "why does" -> "what is the reason that"
            - "how does" -> "in what way does"

        Example: Rephrasing "What is"
            >>> tester._rephrase_question_word("What is gravity?")
            "can you explain gravity?"

        Example: No Matching Pattern
            >>> tester._rephrase_question_word("Tell me about Paris")
            "Tell me about Paris"  # Unchanged
        """
        replacements = {
            "what is": "can you explain",
            "who is": "tell me about",
            "where is": "what is the location of",
            "when did": "at what time did",
            "why does": "what is the reason that",
            "how does": "in what way does",
        }

        for old, new in replacements.items():
            if question.lower().startswith(old):
                return new + question[len(old) :]

        return question

    def _add_context(self, question: str) -> str:
        """Add contextual prefix to a question.

        Prepends conversational context to the question to test if the
        model responds consistently with different framing.

        Args:
            question: The original question.

        Returns:
            str: Question with a randomly selected context prefix.

        Prefixes:
            - "I'm curious about this: "
            - "Could you help me understand: "
            - "I'd like to know: "

        Example:
            >>> tester._add_context("What is gravity?")
            "I'm curious about this: What is gravity?"
        """
        prefixes = [
            "I'm curious about this: ",
            "Could you help me understand: ",
            "I'd like to know: ",
        ]
        import random

        return random.choice(prefixes) + question

    def _simplify(self, question: str) -> str:
        """Simplify a question by removing filler words.

        Strips common polite phrases and filler words to create a more
        direct version of the question.

        Args:
            question: The original question.

        Returns:
            str: Simplified question with filler words removed.

        Removed Fillers:
            - "please", "kindly"
            - "could you", "would you"

        Example:
            >>> tester._simplify("Could you please tell me about Paris?")
            "tell me about Paris?"
        """
        # Remove filler words
        fillers = ["please", "kindly", "could you", "would you"]
        result = question
        for filler in fillers:
            result = result.replace(filler, "").replace(filler.title(), "")
        return result.strip()

    def _elaborate(self, question: str) -> str:
        """Elaborate on a question by adding detail requests.

        Appends a suffix requesting more detailed responses to test if
        the model's core answer remains consistent.

        Args:
            question: The original question.

        Returns:
            str: Question with an elaboration suffix.

        Suffixes:
            - " Please provide details."
            - " I want a thorough answer."
            - " Explain in detail."

        Example:
            >>> tester._elaborate("What is gravity?")
            "What is gravity? Please provide details."
        """
        suffixes = [
            " Please provide details.",
            " I want a thorough answer.",
            " Explain in detail.",
        ]
        import random

        return question.rstrip("?") + "?" + random.choice(suffixes)

    def _calculate_consistency(
        self,
        original: str,
        paraphrased: list[str],
    ) -> float:
        """Calculate consistency score between original and paraphrased responses.

        Computes the average word overlap similarity between the original
        response and all paraphrased responses. Higher scores indicate
        more consistent responses.

        Args:
            original: The original response string.
            paraphrased: List of responses to paraphrased questions.

        Returns:
            float: Consistency score from 0.0 (completely different) to
                1.0 (identical responses). Returns 1.0 if no paraphrases.

        Example: High Consistency
            >>> original = "The capital of France is Paris."
            >>> paraphrased = ["Paris is the capital of France.",
            ...               "France's capital is Paris."]
            >>> score = tester._calculate_consistency(original, paraphrased)
            >>> # High score due to word overlap

        Example: Low Consistency
            >>> original = "Paris"
            >>> paraphrased = ["London", "Berlin"]
            >>> score = tester._calculate_consistency(original, paraphrased)
            >>> # Low score - different answers
        """
        if not paraphrased:
            return 1.0

        similarities = []
        for response in paraphrased:
            sim = word_overlap_similarity(original, response)
            similarities.append(sim)

        return sum(similarities) / len(similarities)

    def _find_contradictions(
        self,
        original: str,
        paraphrased: list[str],
    ) -> list[tuple[str, str]]:
        """Find potential contradictions between original and paraphrased responses.

        Detects contradictions by checking for negation differences between
        the original response and paraphrased responses. A difference in
        negation (one has "not" and the other doesn't) suggests contradiction.

        Args:
            original: The original response string.
            paraphrased: List of responses to paraphrased questions.

        Returns:
            list[tuple[str, str]]: List of (original_excerpt, paraphrased_excerpt)
                tuples for each detected contradiction. Excerpts are truncated
                to 100 characters.

        Example: Detecting Contradiction
            >>> original = "Paris is the capital"
            >>> paraphrased = ["Paris is not the capital"]
            >>> contradictions = tester._find_contradictions(original, paraphrased)
            >>> # Returns [("Paris is the capital", "Paris is not the capital")]

        Example: No Contradictions
            >>> original = "Paris is the capital"
            >>> paraphrased = ["The capital is Paris"]
            >>> contradictions = tester._find_contradictions(original, paraphrased)
            >>> # Returns []
        """
        contradictions = []

        # Simple negation detection
        for response in paraphrased:
            if self._has_negation_difference(original, response):
                contradictions.append((original[:100], response[:100]))

        return contradictions

    def _has_negation_difference(self, text1: str, text2: str) -> bool:
        """Check if two texts differ in their use of negation.

        Detects whether one text contains negation words while the other
        doesn't, which may indicate a contradiction.

        Args:
            text1: First text to check.
            text2: Second text to check.

        Returns:
            bool: True if exactly one text contains negation, False otherwise.

        Negation Words:
            - "not", "never", "no", "none", "neither", "nobody"

        Example: Different Negation
            >>> tester._has_negation_difference("It is true", "It is not true")
            True

        Example: Same Negation (both negative)
            >>> tester._has_negation_difference("It is not A", "It is not B")
            False

        Example: Same Negation (both positive)
            >>> tester._has_negation_difference("It is A", "It is B")
            False
        """
        negations = ["not", "never", "no", "none", "neither", "nobody"]

        text1_has_neg = any(neg in text1.lower() for neg in negations)
        text2_has_neg = any(neg in text2.lower() for neg in negations)

        # Different negation state might indicate contradiction
        return text1_has_neg != text2_has_neg

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute a deterministic similarity score between two responses."""
        return float(word_overlap_similarity(text1, text2))

    def _calculate_drift(
        self,
        original: str,
        paraphrased: list[str],
    ) -> float:
        """Calculate semantic drift from the original response.

        Measures how much the paraphrased responses diverge from the
        original in terms of meaning/content. Higher drift indicates
        more variation in responses.

        Args:
            original: The original response string.
            paraphrased: List of responses to paraphrased questions.

        Returns:
            float: Drift score from 0.0 (no drift, identical) to 1.0
                (maximum drift, completely different). Returns 0.0 if
                no paraphrases.

        Example: Low Drift
            >>> original = "Paris is the capital of France"
            >>> paraphrased = ["The capital of France is Paris"]
            >>> drift = tester._calculate_drift(original, paraphrased)
            >>> # Low drift - same meaning

        Example: High Drift
            >>> original = "Paris is beautiful"
            >>> paraphrased = ["London has great museums"]
            >>> drift = tester._calculate_drift(original, paraphrased)
            >>> # High drift - completely different content
        """
        if not paraphrased:
            return 0.0

        # Calculate average distance from original
        distances = []
        for response in paraphrased:
            sim = self._text_similarity(original, response)
            distances.append(1.0 - sim)

        return sum(distances) / len(distances)


class SourceAttributor:
    """Handles source attribution for claims with reliability tracking.

    Manages a registry of information sources with reliability scores and
    metadata. Attributes claims to their most likely sources based on
    topic matching and source reliability.

    This is useful for:
        - Tracking where claims originate
        - Assessing claim reliability based on source quality
        - Building provenance chains for fact-checking

    Attributes:
        sources: Dictionary mapping source names to their metadata.

    Example: Basic Source Attribution
        >>> from insideLLMs.contrib.knowledge import SourceAttributor, Claim
        >>> attributor = SourceAttributor()
        >>> attributor.add_source(
        ...     "Wikipedia",
        ...     reliability=0.7,
        ...     metadata={"topics": ["history", "science", "geography"]}
        ... )
        >>> claim = Claim(text="Paris is the capital of France")
        >>> attribution = attributor.attribute(claim)
        >>> print(attribution["best_source"])

    Example: Multiple Sources
        >>> attributor.add_source("Encyclopedia", reliability=0.9)
        >>> attributor.add_source("Blog", reliability=0.3)
        >>> attributor.add_source("Academic Paper", reliability=0.95)
        >>> attribution = attributor.attribute(claim)
        >>> for attr in attribution["attributions"]:
        ...     print(f"{attr['source']}: {attr['confidence']:.2f}")

    Example: Topic-Based Attribution
        >>> attributor.add_source(
        ...     "Science Journal",
        ...     reliability=0.9,
        ...     metadata={"topics": ["physics", "chemistry", "biology"]}
        ... )
        >>> science_claim = Claim(text="Physics explains motion")
        >>> attribution = attributor.attribute(science_claim)
        >>> # Higher confidence for Science Journal due to topic match
    """

    def __init__(self):
        """Initialize the SourceAttributor with an empty source registry.

        Example: Creating an Attributor
            >>> attributor = SourceAttributor()
            >>> print(len(attributor.sources))  # 0
        """
        self.sources: dict[str, dict[str, Any]] = {}

    def add_source(
        self,
        name: str,
        reliability: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a source to the registry.

        Registers a new source with its reliability score and optional
        metadata for use in claim attribution.

        Args:
            name: Unique name/identifier for the source.
            reliability: Reliability score from 0.0 (unreliable) to 1.0
                (highly reliable). Default is 0.5.
            metadata: Optional dictionary with additional source info.
                Can include 'topics' list for topic-based matching.

        Example: Simple Source
            >>> attributor = SourceAttributor()
            >>> attributor.add_source("Wikipedia", reliability=0.7)

        Example: Source with Topics
            >>> attributor.add_source(
            ...     "Medical Journal",
            ...     reliability=0.9,
            ...     metadata={"topics": ["medicine", "health", "disease"]}
            ... )

        Example: Building a Source Registry
            >>> sources = [
            ...     ("Encyclopedia", 0.8, ["general"]),
            ...     ("News Site", 0.5, ["current events"]),
            ...     ("Academic DB", 0.95, ["research", "science"])
            ... ]
            >>> for name, rel, topics in sources:
            ...     attributor.add_source(name, rel, {"topics": topics})
        """
        self.sources[name] = {
            "reliability": reliability,
            "metadata": metadata or {},
        }

    def attribute(
        self,
        claim: Claim,
        candidate_sources: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Attribute a claim to its most likely sources.

        Searches registered sources for matches based on topic overlap
        and reliability. Returns ranked attributions with confidence scores.

        Args:
            claim: The Claim to attribute.
            candidate_sources: Optional list of source names to consider.
                If None, all registered sources are checked.

        Returns:
            dict[str, Any]: Attribution result containing:
                - "claim": The claim as a dictionary
                - "attributions": List of matches sorted by confidence
                - "best_source": The highest-confidence attribution or None

        Example: Basic Attribution
            >>> attributor.add_source("Wikipedia", 0.7, {"topics": ["Paris"]})
            >>> claim = Claim(text="Paris is beautiful")
            >>> result = attributor.attribute(claim)
            >>> if result["best_source"]:
            ...     print(f"Best: {result['best_source']['source']}")

        Example: Filtering Candidates
            >>> result = attributor.attribute(
            ...     claim,
            ...     candidate_sources=["Wikipedia", "Encyclopedia"]
            ... )

        Example: Examining All Attributions
            >>> result = attributor.attribute(claim)
            >>> for attr in result["attributions"]:
            ...     print(f"{attr['source']}: conf={attr['confidence']:.2f}, "
            ...           f"rel={attr['reliability']:.2f}")

        Example: No Match
            >>> result = attributor.attribute(obscure_claim)
            >>> if result["best_source"] is None:
            ...     print("No source found for this claim")
        """
        sources_to_check = candidate_sources or list(self.sources.keys())

        attributions = []
        for source_name in sources_to_check:
            if source_name in self.sources:
                score = self._match_source(claim, source_name)
                if score > 0.3:
                    attributions.append(
                        {
                            "source": source_name,
                            "confidence": score,
                            "reliability": self.sources[source_name]["reliability"],
                        }
                    )

        # Sort by confidence
        attributions.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "claim": claim.to_dict(),
            "attributions": attributions,
            "best_source": attributions[0] if attributions else None,
        }

    def _match_source(
        self,
        claim: Claim,
        source_name: str,
    ) -> float:
        """Calculate match score between a claim and a source.

        Computes a confidence score for how well a claim matches a source
        based on the source's reliability and topic overlap with the claim.

        Args:
            claim: The Claim to match.
            source_name: Name of the source to check.

        Returns:
            float: Match score from 0.0 to 1.0. Computed as:
                - Base: reliability * 0.5
                - +0.3 if claim words overlap with source topics

        Example: High Match
            >>> # Source has topic "Paris", claim mentions "Paris"
            >>> score = attributor._match_source(paris_claim, "Geography DB")
            >>> # Score boosted by topic match

        Example: Reliability Only
            >>> # No topic overlap, score based only on reliability
            >>> score = attributor._match_source(random_claim, "High Rel Source")
            >>> # Score = reliability * 0.5
        """
        # Simple heuristic - in real system would check source content
        source = self.sources[source_name]

        # Base score from reliability
        base_score = source["reliability"] * 0.5

        # Boost for matching metadata
        metadata = source.get("metadata", {})
        if "topics" in metadata:
            claim_words = set(claim.text.lower().split())
            topics = {t.lower() for t in metadata["topics"]}
            if claim_words & topics:
                base_score += 0.3

        return min(1.0, base_score)


# Convenience functions


def extract_claims(
    text: str,
    min_confidence: float = 0.3,
) -> list[Claim]:
    """Extract factual claims from text.

    Convenience function that creates a ClaimExtractor and extracts claims
    from the provided text. Use this for quick, one-off extractions.

    Args:
        text: Input text to extract claims from. Can be any natural language
            text containing factual statements.
        min_confidence: Minimum confidence threshold for including claims.
            Claims with confidence below this value are filtered out.
            Default is 0.3.

    Returns:
        list[Claim]: List of extracted Claim objects with confidence scores
            at or above the threshold, deduplicated by text.

    Example: Basic Extraction
        >>> from insideLLMs.contrib.knowledge import extract_claims
        >>> text = "Albert Einstein was a physicist. He won the Nobel Prize."
        >>> claims = extract_claims(text)
        >>> for claim in claims:
        ...     print(f"Claim: {claim.text}")
        ...     print(f"Confidence: {claim.confidence:.2f}")

    Example: High Confidence Only
        >>> claims = extract_claims(text, min_confidence=0.7)
        >>> print(f"Found {len(claims)} high-confidence claims")

    Example: Processing Multiple Texts
        >>> texts = ["Text 1...", "Text 2...", "Text 3..."]
        >>> all_claims = []
        >>> for t in texts:
        ...     all_claims.extend(extract_claims(t))
        >>> print(f"Total claims: {len(all_claims)}")

    Example: With Verification Pipeline
        >>> claims = extract_claims(text)
        >>> for claim in claims:
        ...     result = verify_claim(claim, knowledge_base)
        ...     print(f"{claim.text}: {result.status.value}")
    """
    extractor = ClaimExtractor()
    return extractor.extract(text, min_confidence)


def verify_claim(
    claim: Claim,
    knowledge_base: Optional[dict[str, Any]] = None,
    context: Optional[str] = None,
) -> VerificationResult:
    """Verify a factual claim against a knowledge base.

    Convenience function that creates a FactVerifier and verifies the claim.
    Use this for quick, one-off verifications.

    Args:
        claim: The Claim object to verify.
        knowledge_base: Optional dictionary mapping topics/entities to their
            known facts. If None, the claim will be marked as unverifiable.
        context: Optional context string for consistency checking.

    Returns:
        VerificationResult: Result containing verification status, confidence,
            supporting/contradicting evidence, and source quality.

    Example: Simple Verification
        >>> from insideLLMs.contrib.knowledge import verify_claim, Claim
        >>> claim = Claim(text="Paris is the capital of France")
        >>> kb = {"Paris": "capital of France"}
        >>> result = verify_claim(claim, knowledge_base=kb)
        >>> print(f"Status: {result.status.value}")
        >>> print(f"Confidence: {result.confidence:.2f}")

    Example: Without Knowledge Base
        >>> result = verify_claim(Claim(text="Some claim"))
        >>> # Status will be "unverifiable" without a knowledge base

    Example: With Context
        >>> claim = Claim(text="The city is beautiful")
        >>> context = "Paris is known for its architecture and culture."
        >>> result = verify_claim(claim, context=context)

    Example: Batch Verification
        >>> claims = extract_claims(text)
        >>> results = [verify_claim(c, kb) for c in claims]
        >>> verified = sum(1 for r in results if r.status.value == "verified")
    """
    verifier = FactVerifier(knowledge_base)
    return verifier.verify(claim, context)


def probe_knowledge(
    questions: list[str],
    model_fn: Callable[[str], str],
    expected_answers: Optional[list[str]] = None,
) -> KnowledgeReport:
    """Probe model knowledge with a list of questions.

    Convenience function that creates a KnowledgeProber, adds all questions
    as probes, runs them against the model, and generates a report.

    Args:
        questions: List of questions to ask the model.
        model_fn: A callable that takes a question string and returns
            the model's response string.
        expected_answers: Optional list of expected answers for evaluation.
            If provided, must have the same length as questions.

    Returns:
        KnowledgeReport: Comprehensive report with accuracy, category breakdown,
            difficulty analysis, confidence calibration, and knowledge gaps.

    Example: Basic Knowledge Probing
        >>> from insideLLMs.contrib.knowledge import probe_knowledge
        >>> questions = [
        ...     "What is the capital of France?",
        ...     "Who wrote Hamlet?",
        ...     "When did WWII end?"
        ... ]
        >>> expected = ["Paris", "Shakespeare", "1945"]
        >>> def model(q): return "The answer is Paris, Shakespeare, 1945."
        >>> report = probe_knowledge(questions, model, expected)
        >>> print(f"Accuracy: {report.accuracy:.2%}")

    Example: Without Expected Answers
        >>> # Just capture responses without evaluation
        >>> report = probe_knowledge(questions, model)
        >>> print(f"Ran {report.probes_run} probes")

    Example: Analyzing Results
        >>> report = probe_knowledge(questions, model, expected)
        >>> print(f"Correct: {report.correct_count}/{report.probes_run}")
        >>> print(f"Calibration: {report.confidence_calibration:.2f}")
        >>> if report.knowledge_gaps:
        ...     print(f"Weak areas: {report.knowledge_gaps}")

    Example: Comparing Models
        >>> report_a = probe_knowledge(questions, model_a, expected)
        >>> report_b = probe_knowledge(questions, model_b, expected)
        >>> print(f"Model A: {report_a.accuracy:.2%}")
        >>> print(f"Model B: {report_b.accuracy:.2%}")
    """
    prober = KnowledgeProber()

    for i, question in enumerate(questions):
        expected = expected_answers[i] if expected_answers and i < len(expected_answers) else None
        prober.add_probe(question, expected)

    results = prober.run_all_probes(model_fn)
    return prober.generate_report(results)


def check_consistency(
    question: str,
    model_fn: Callable[[str], str],
    num_variations: int = 3,
) -> ConsistencyResult:
    """Check response consistency across question variations.

    Convenience function that creates a ConsistencyTester and tests whether
    the model gives consistent answers when the same question is asked in
    different ways.

    Args:
        question: The question to test for consistency.
        model_fn: A callable that takes a question string and returns
            the model's response string.
        num_variations: Number of paraphrased variations to test.
            Default is 3.

    Returns:
        ConsistencyResult: Result containing original response, paraphrased
            responses, consistency score, contradictions, and semantic drift.

    Example: Basic Consistency Check
        >>> from insideLLMs.contrib.knowledge import check_consistency
        >>> def model(q): return "The capital of France is Paris."
        >>> result = check_consistency("What is the capital of France?", model)
        >>> print(f"Consistency: {result.consistency_score:.2f}")
        >>> print(f"Drift: {result.semantic_drift:.2f}")

    Example: More Thorough Testing
        >>> result = check_consistency(question, model, num_variations=5)
        >>> print(f"Tested {len(result.paraphrased_responses)} variations")

    Example: Detecting Issues
        >>> result = check_consistency("Who invented the telephone?", model)
        >>> if result.consistency_score < 0.7:
        ...     print("Warning: Inconsistent responses!")
        >>> if result.contradictions:
        ...     print(f"Found {len(result.contradictions)} contradictions")

    Example: Batch Consistency Testing
        >>> questions = ["Q1?", "Q2?", "Q3?"]
        >>> results = [check_consistency(q, model) for q in questions]
        >>> avg_consistency = sum(r.consistency_score for r in results) / len(results)
        >>> print(f"Average consistency: {avg_consistency:.2f}")
    """
    tester = ConsistencyTester()
    return tester.test_consistency(question, model_fn, num_variations)


def verify_facts(
    text: str,
    knowledge_base: Optional[dict[str, Any]] = None,
) -> list[VerificationResult]:
    """Extract and verify facts from text in one step.

    Convenience function that extracts claims from text and verifies each
    one against a knowledge base. Combines extract_claims() and verify_claim()
    into a single call.

    Args:
        text: Input text to analyze for factual claims.
        knowledge_base: Optional dictionary mapping topics/entities to their
            known facts for verification.

    Returns:
        list[VerificationResult]: List of verification results, one for each
            extracted claim.

    Example: Basic Fact Verification
        >>> from insideLLMs.contrib.knowledge import verify_facts
        >>> text = "Einstein was a physicist. He won the Nobel Prize in 1921."
        >>> kb = {"Einstein": "physicist, Nobel Prize 1921"}
        >>> results = verify_facts(text, knowledge_base=kb)
        >>> for r in results:
        ...     print(f"{r.claim.text}: {r.status.value}")

    Example: Without Knowledge Base
        >>> results = verify_facts(text)
        >>> # All claims will be "unverifiable" without a knowledge base
        >>> unverifiable = sum(1 for r in results if r.status.value == "unverifiable")

    Example: Aggregating Results
        >>> results = verify_facts(document_text, kb)
        >>> verified = [r for r in results if r.status.value == "verified"]
        >>> contradicted = [r for r in results if r.status.value == "contradicted"]
        >>> print(f"Verified: {len(verified)}, Contradicted: {len(contradicted)}")

    Example: Processing Multiple Documents
        >>> documents = ["Doc 1...", "Doc 2...", "Doc 3..."]
        >>> all_results = []
        >>> for doc in documents:
        ...     results = verify_facts(doc, kb)
        ...     all_results.extend(results)
        >>> total_verified = sum(1 for r in all_results if r.status.value == "verified")
    """
    claims = extract_claims(text)
    verifier = FactVerifier(knowledge_base)
    return verifier.verify_batch(claims)
