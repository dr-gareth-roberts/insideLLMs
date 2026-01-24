"""
Model hallucination detection utilities.

Provides tools for:
- Detecting potential hallucinations in LLM outputs
- Fact consistency checking
- Source attribution verification
- Confidence-based hallucination assessment
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from insideLLMs.nlp.similarity import word_overlap_similarity


class HallucinationType(Enum):
    """Types of hallucinations."""

    FACTUAL_ERROR = "factual_error"
    FABRICATED_ENTITY = "fabricated_entity"
    CONFLATION = "conflation"  # Mixing up distinct entities/facts
    ANACHRONISM = "anachronism"  # Temporally impossible claims
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    ATTRIBUTION_ERROR = "attribution_error"  # Wrong source attribution
    EXAGGERATION = "exaggeration"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class SeverityLevel(Enum):
    """Severity levels for hallucinations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionMethod(Enum):
    """Methods for detecting hallucinations."""

    CONSISTENCY = "consistency"  # Check self-consistency
    FACTUAL = "factual"  # Check against known facts
    ATTRIBUTION = "attribution"  # Check source claims
    CONFIDENCE = "confidence"  # Use model confidence
    ENTAILMENT = "entailment"  # Check logical entailment


@dataclass
class HallucinationFlag:
    """A single detected potential hallucination."""

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
        """Convert to dictionary."""
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
    """Result of consistency checking across responses."""

    responses: list[str]
    consistent_claims: list[str]
    inconsistent_claims: list[tuple[str, str, float]]  # (claim1, claim2, disagreement)
    overall_consistency: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_consistent(self) -> bool:
        """Check if responses are mostly consistent."""
        return self.overall_consistency >= 0.8

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Score for factual accuracy."""

    score: float  # 0-1, 1 = fully factual
    verifiable_claims: int
    verified_claims: int
    unverified_claims: int
    contradicted_claims: int
    claim_details: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Factual accuracy rate."""
        if self.verifiable_claims == 0:
            return 1.0
        return self.verified_claims / self.verifiable_claims

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Comprehensive hallucination detection report."""

    text: str
    flags: list[HallucinationFlag]
    overall_score: float  # 0-1, 1 = no hallucinations detected
    consistency_check: Optional[ConsistencyCheck] = None
    factuality_score: Optional[FactualityScore] = None
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_hallucinations(self) -> bool:
        """Check if any hallucinations were detected."""
        return len(self.flags) > 0

    @property
    def critical_flags(self) -> list[HallucinationFlag]:
        """Get only critical severity flags."""
        return [f for f in self.flags if f.severity == SeverityLevel.CRITICAL]

    @property
    def high_confidence_flags(self) -> list[HallucinationFlag]:
        """Get flags with high confidence (> 0.8)."""
        return [f for f in self.flags if f.confidence > 0.8]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Detects hallucinations using pattern matching."""

    def __init__(self):
        """Initialize detector."""
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
        """Detect potential hallucinations using patterns."""
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
    """Checks consistency across multiple responses."""

    def __init__(
        self,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """Initialize checker."""
        self.similarity_fn = similarity_fn or word_overlap_similarity

    def _extract_claims(self, text: str) -> list[str]:
        """Extract claims from text (simple sentence-based)."""
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20 and not sent.endswith("?"):  # Filter questions and short fragments
                claims.append(sent)
        return claims

    def check(self, responses: list[str]) -> ConsistencyCheck:
        """Check consistency across multiple responses."""
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
    """Checks factual accuracy of claims."""

    def __init__(
        self,
        fact_checker: Optional[Callable[[str], tuple[bool, float]]] = None,
        knowledge_base: Optional[dict[str, str]] = None,
    ):
        """Initialize checker."""
        self.fact_checker = fact_checker
        self.knowledge_base = knowledge_base or {}

        # Common factual patterns
        self.factual_patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:the\s+)?([^.]+)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+(?:born|founded|created|established)\s+(?:in|on)\s+(\d{4})",
            r"The\s+([a-z]+)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+([^.]+)",
        ]

    def _extract_factual_claims(self, text: str) -> list[dict[str, Any]]:
        """Extract claims that can be fact-checked."""
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
        """Check factual accuracy of text."""
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
    """Comprehensive hallucination detection system."""

    def __init__(
        self,
        pattern_detector: Optional[PatternBasedDetector] = None,
        consistency_checker: Optional[FactConsistencyChecker] = None,
        factuality_checker: Optional[FactualityChecker] = None,
    ):
        """Initialize detector."""
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
        """Detect potential hallucinations in text."""
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
        """Generate recommendations based on detection results."""
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
    """Checks if response is grounded in provided context."""

    def __init__(
        self,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        threshold: float = 0.5,
    ):
        """Initialize checker."""
        self.similarity_fn = similarity_fn or word_overlap_similarity
        self.threshold = threshold

    def check(
        self,
        response: str,
        context: str,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Check if response is grounded in context."""
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
    """Checks if claims are properly attributed to sources."""

    def __init__(self):
        """Initialize checker."""
        self.attribution_patterns = [
            r"according to ([^,]+)",
            r"([^,]+) said",
            r"([^,]+) reported",
            r"([^,]+) found that",
            r"cited by ([^,]+)",
            r"source: ([^,\n]+)",
        ]

    def check(self, text: str) -> dict[str, Any]:
        """Check attribution in text."""
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
    """Detect potential hallucinations in text."""
    detector = ComprehensiveHallucinationDetector()
    return detector.detect(text, reference_texts)


def check_consistency(responses: list[str]) -> ConsistencyCheck:
    """Check consistency across multiple responses."""
    checker = FactConsistencyChecker()
    return checker.check(responses)


def check_factuality(
    text: str,
    knowledge_base: Optional[dict[str, str]] = None,
) -> FactualityScore:
    """Check factual accuracy of text."""
    checker = FactualityChecker(knowledge_base=knowledge_base)
    return checker.check(text)


def check_groundedness(
    response: str,
    context: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Check if response is grounded in context."""
    checker = GroundednessChecker(threshold=threshold)
    return checker.check(response, context)


def check_attribution(text: str) -> dict[str, Any]:
    """Check if claims are properly attributed."""
    checker = AttributionChecker()
    return checker.check(text)


def quick_hallucination_check(text: str) -> dict[str, Any]:
    """Quick hallucination check returning summary."""
    report = detect_hallucinations(text)
    return {
        "has_potential_hallucinations": report.has_hallucinations,
        "overall_score": report.overall_score,
        "n_flags": len(report.flags),
        "critical_issues": len(report.critical_flags),
        "recommendations": report.recommendations[:3],
    }
