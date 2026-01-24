"""
Knowledge probing and fact verification utilities.

Provides tools for:
- Knowledge probing and fact extraction
- Claim verification and fact-checking
- Knowledge consistency testing
- Confidence calibration
- Source attribution
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from insideLLMs.nlp.similarity import word_overlap_similarity


class VerificationStatus(Enum):
    """Status of fact verification."""

    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class KnowledgeCategory(Enum):
    """Categories of knowledge."""

    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    TEMPORAL = "temporal"
    RELATIONAL = "relational"
    COMMONSENSE = "commonsense"


class ConfidenceLevel(Enum):
    """Confidence levels."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class Claim:
    """A factual claim extracted from text."""

    text: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    category: KnowledgeCategory = KnowledgeCategory.FACTUAL
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Result of verifying a claim."""

    claim: Claim
    status: VerificationStatus
    confidence: float
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    source_quality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """A probe for testing knowledge."""

    question: str
    expected_answer: Optional[str] = None
    category: KnowledgeCategory = KnowledgeCategory.FACTUAL
    difficulty: float = 0.5  # 0-1
    hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "category": self.category.value,
            "difficulty": self.difficulty,
            "hints": self.hints,
        }


@dataclass
class KnowledgeProbeResult:
    """Result of a knowledge probe."""

    probe: KnowledgeProbe
    response: str
    is_correct: Optional[bool] = None
    partial_score: float = 0.0
    confidence_expressed: float = 0.0
    reasoning_provided: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Result of consistency testing."""

    original_response: str
    paraphrased_responses: list[str]
    consistency_score: float  # 0-1
    contradictions: list[tuple[str, str]] = field(default_factory=list)
    semantic_drift: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_response": self.original_response,
            "num_paraphrases": len(self.paraphrased_responses),
            "consistency_score": self.consistency_score,
            "num_contradictions": len(self.contradictions),
            "semantic_drift": self.semantic_drift,
        }


@dataclass
class KnowledgeReport:
    """Report on knowledge probing results."""

    probes_run: int
    correct_count: int
    accuracy: float
    by_category: dict[str, float]
    by_difficulty: dict[str, float]
    confidence_calibration: float
    knowledge_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Extracts factual claims from text."""

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
        """Extract claims from text."""
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
        """Estimate confidence in extracted claim."""
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
    """Verifies factual claims."""

    def __init__(
        self,
        knowledge_base: Optional[dict[str, Any]] = None,
    ):
        """Initialize verifier."""
        self.knowledge_base = knowledge_base or {}

    def verify(
        self,
        claim: Claim,
        context: Optional[str] = None,
    ) -> VerificationResult:
        """Verify a single claim."""
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
        """Verify multiple claims."""
        return [self.verify(claim, context) for claim in claims]

    def _check_knowledge_base(
        self,
        claim: Claim,
    ) -> Optional[dict[str, Any]]:
        """Check claim against knowledge base."""
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
        """Check internal consistency of claim."""
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
        """Assess quality of evidence sources."""
        if not evidence:
            return 0.0

        # Simple heuristic based on evidence count
        return min(1.0, len(evidence) * 0.2)


class KnowledgeProber:
    """Probes model knowledge through questioning."""

    def __init__(self):
        """Initialize prober."""
        self.probes: list[KnowledgeProbe] = []

    def add_probe(
        self,
        question: str,
        expected_answer: Optional[str] = None,
        category: KnowledgeCategory = KnowledgeCategory.FACTUAL,
        difficulty: float = 0.5,
        hints: Optional[list[str]] = None,
    ) -> KnowledgeProbe:
        """Add a knowledge probe."""
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
        """Run a single knowledge probe."""
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
        """Run all registered probes."""
        return [self.run_probe(probe, model_fn) for probe in self.probes]

    def generate_report(
        self,
        results: list[ProbeResult],
    ) -> KnowledgeReport:
        """Generate report from probe results."""
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
        """Check if response contains expected answer."""
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
        """Calculate partial match score."""
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            return 0.0

        overlap = len(response_words & expected_words)
        return overlap / len(expected_words)

    def _detect_confidence(self, response: str) -> float:
        """Detect expressed confidence level."""
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
        """Detect if reasoning is provided."""
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
        """Calculate confidence calibration score."""
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
    """Tests consistency of model responses."""

    def __init__(self):
        """Initialize tester."""
        pass

    def generate_paraphrases(
        self,
        question: str,
        num_paraphrases: int = 3,
    ) -> list[str]:
        """Generate paraphrased versions of a question."""
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
        """Test consistency across paraphrased questions."""
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
        """Rephrase using different question words."""
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
        """Add context to question."""
        prefixes = [
            "I'm curious about this: ",
            "Could you help me understand: ",
            "I'd like to know: ",
        ]
        import random

        return random.choice(prefixes) + question

    def _simplify(self, question: str) -> str:
        """Simplify the question."""
        # Remove filler words
        fillers = ["please", "kindly", "could you", "would you"]
        result = question
        for filler in fillers:
            result = result.replace(filler, "").replace(filler.title(), "")
        return result.strip()

    def _elaborate(self, question: str) -> str:
        """Elaborate on the question."""
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
        """Calculate consistency score."""
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
        """Find potential contradictions."""
        contradictions = []

        # Simple negation detection
        for response in paraphrased:
            if self._has_negation_difference(original, response):
                contradictions.append((original[:100], response[:100]))

        return contradictions

    def _has_negation_difference(self, text1: str, text2: str) -> bool:
        """Check if texts differ in negation."""
        negations = ["not", "never", "no", "none", "neither", "nobody"]

        text1_has_neg = any(neg in text1.lower() for neg in negations)
        text2_has_neg = any(neg in text2.lower() for neg in negations)

        # Different negation state might indicate contradiction
        return text1_has_neg != text2_has_neg

    def _calculate_drift(
        self,
        original: str,
        paraphrased: list[str],
    ) -> float:
        """Calculate semantic drift from original."""
        if not paraphrased:
            return 0.0

        # Calculate average distance from original
        distances = []
        for response in paraphrased:
            sim = self._text_similarity(original, response)
            distances.append(1.0 - sim)

        return sum(distances) / len(distances)


class SourceAttributor:
    """Handles source attribution for claims."""

    def __init__(self):
        """Initialize attributor."""
        self.sources: dict[str, dict[str, Any]] = {}

    def add_source(
        self,
        name: str,
        reliability: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a source."""
        self.sources[name] = {
            "reliability": reliability,
            "metadata": metadata or {},
        }

    def attribute(
        self,
        claim: Claim,
        candidate_sources: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Attribute a claim to sources."""
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
        """Calculate match score between claim and source."""
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

    Args:
        text: Input text to extract claims from
        min_confidence: Minimum confidence threshold

    Returns:
        List of extracted claims
    """
    extractor = ClaimExtractor()
    return extractor.extract(text, min_confidence)


def verify_claim(
    claim: Claim,
    knowledge_base: Optional[dict[str, Any]] = None,
    context: Optional[str] = None,
) -> VerificationResult:
    """Verify a factual claim.

    Args:
        claim: The claim to verify
        knowledge_base: Optional knowledge base to check against
        context: Optional context for verification

    Returns:
        VerificationResult with verification status
    """
    verifier = FactVerifier(knowledge_base)
    return verifier.verify(claim, context)


def probe_knowledge(
    questions: list[str],
    model_fn: Callable[[str], str],
    expected_answers: Optional[list[str]] = None,
) -> KnowledgeReport:
    """Probe model knowledge with questions.

    Args:
        questions: List of questions to ask
        model_fn: Function that takes question and returns answer
        expected_answers: Optional expected answers for evaluation

    Returns:
        KnowledgeReport with results
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

    Args:
        question: The question to test
        model_fn: Function that takes question and returns answer
        num_variations: Number of variations to test

    Returns:
        ConsistencyResult with consistency analysis
    """
    tester = ConsistencyTester()
    return tester.test_consistency(question, model_fn, num_variations)


def verify_facts(
    text: str,
    knowledge_base: Optional[dict[str, Any]] = None,
) -> list[VerificationResult]:
    """Extract and verify facts from text.

    Args:
        text: Input text to analyze
        knowledge_base: Optional knowledge base

    Returns:
        List of verification results for extracted claims
    """
    claims = extract_claims(text)
    verifier = FactVerifier(knowledge_base)
    return verifier.verify_batch(claims)
