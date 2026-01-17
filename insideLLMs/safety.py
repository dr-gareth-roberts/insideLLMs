"""
Safety and content analysis utilities for LLM outputs.

Provides tools for:
- Content safety classification
- Toxicity detection
- PII detection and masking
- Hallucination indicators
- Bias detection patterns
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple


class SafetyCategory(Enum):
    """Categories of safety concerns."""

    SAFE = "safe"
    TOXICITY = "toxicity"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL_CONTENT = "sexual_content"
    SELF_HARM = "self_harm"
    DANGEROUS_CONTENT = "dangerous_content"
    PII_EXPOSURE = "pii_exposure"
    MISINFORMATION = "misinformation"
    MANIPULATION = "manipulation"


class RiskLevel(Enum):
    """Risk level classifications."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyFlag:
    """A single safety flag."""

    category: SafetyCategory
    risk_level: RiskLevel
    description: str
    matched_text: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyReport:
    """Comprehensive safety analysis report."""

    text: str
    is_safe: bool
    overall_risk: RiskLevel
    flags: List[SafetyFlag] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_flags_by_category(self, category: SafetyCategory) -> List[SafetyFlag]:
        """Get flags for a specific category."""
        return [f for f in self.flags if f.category == category]

    def get_highest_risk_flag(self) -> Optional[SafetyFlag]:
        """Get the flag with highest risk level."""
        if not self.flags:
            return None
        risk_order = [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]
        for level in risk_order:
            for flag in self.flags:
                if flag.risk_level == level:
                    return flag
        return self.flags[0] if self.flags else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_safe": self.is_safe,
            "overall_risk": self.overall_risk.value,
            "flags": [
                {
                    "category": f.category.value,
                    "risk_level": f.risk_level.value,
                    "description": f.description,
                    "confidence": f.confidence,
                }
                for f in self.flags
            ],
            "scores": self.scores,
        }


@dataclass
class PIIMatch:
    """A detected PII match."""

    pii_type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class PIIReport:
    """Report of PII detection."""

    text: str
    has_pii: bool
    matches: List[PIIMatch] = field(default_factory=list)
    masked_text: Optional[str] = None

    def get_by_type(self, pii_type: str) -> List[PIIMatch]:
        """Get matches of a specific type."""
        return [m for m in self.matches if m.pii_type == pii_type]


class PIIDetector:
    """Detect personally identifiable information in text."""

    # Common PII patterns
    PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "phone_intl": re.compile(r"\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"),
        "ssn": re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"),
        "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "date_of_birth": re.compile(
            r"\b(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}\b"
        ),
        "address": re.compile(
            r"\b\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|"
            r"boulevard|blvd|way|court|ct)\b",
            re.IGNORECASE,
        ),
        "zip_code": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    }

    # Mask replacements
    MASKS = {
        "email": "[EMAIL]",
        "phone_us": "[PHONE]",
        "phone_intl": "[PHONE]",
        "ssn": "[SSN]",
        "credit_card": "[CREDIT_CARD]",
        "ip_address": "[IP_ADDRESS]",
        "date_of_birth": "[DOB]",
        "address": "[ADDRESS]",
        "zip_code": "[ZIP]",
    }

    def __init__(self, patterns: Optional[Dict[str, Pattern]] = None):
        """Initialize detector with optional custom patterns."""
        self.patterns = patterns or self.PATTERNS.copy()

    def add_pattern(self, name: str, pattern: str, mask: str = "[REDACTED]") -> None:
        """Add a custom PII pattern.

        Args:
            name: Pattern identifier.
            pattern: Regex pattern string.
            mask: Replacement mask.
        """
        self.patterns[name] = re.compile(pattern)
        self.MASKS[name] = mask

    def detect(self, text: str) -> PIIReport:
        """Detect PII in text.

        Args:
            text: Text to analyze.

        Returns:
            PIIReport with matches.
        """
        matches = []

        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                ))

        # Sort by position
        matches.sort(key=lambda m: m.start)

        return PIIReport(
            text=text,
            has_pii=len(matches) > 0,
            matches=matches,
        )

    def mask(self, text: str, types: Optional[List[str]] = None) -> str:
        """Mask PII in text.

        Args:
            text: Text to mask.
            types: Optional list of PII types to mask (None = all).

        Returns:
            Masked text.
        """
        result = text
        patterns_to_use = (
            {k: v for k, v in self.patterns.items() if k in types}
            if types
            else self.patterns
        )

        for pii_type, pattern in patterns_to_use.items():
            mask = self.MASKS.get(pii_type, "[REDACTED]")
            result = pattern.sub(mask, result)

        return result

    def detect_and_mask(self, text: str) -> PIIReport:
        """Detect and mask PII in text.

        Args:
            text: Text to analyze and mask.

        Returns:
            PIIReport with matches and masked text.
        """
        report = self.detect(text)
        report.masked_text = self.mask(text)
        return report


class ToxicityAnalyzer:
    """Analyze text for toxic content patterns."""

    # Word lists for basic detection (intentionally limited for safety)
    # In production, use ML-based classifiers
    PROFANITY_INDICATORS = {
        "damn", "hell", "crap", "suck",
    }

    THREAT_PATTERNS = [
        re.compile(r"\b(kill|murder|hurt|harm|attack)\s+(you|them|him|her)\b", re.I),
        re.compile(r"\b(going to|gonna|will)\s+(kill|hurt|harm)\b", re.I),
    ]

    HARASSMENT_PATTERNS = [
        re.compile(r"\byou[''']?(?:re| are)\s+(?:an?\s+)?(stupid|idiot|dumb|moron)\b", re.I),
        re.compile(r"\bshut\s+up\b", re.I),
    ]

    def __init__(self):
        """Initialize the analyzer."""
        self._custom_patterns: List[Tuple[str, Pattern, RiskLevel]] = []

    def add_pattern(
        self, name: str, pattern: str, risk_level: RiskLevel = RiskLevel.MEDIUM
    ) -> None:
        """Add a custom detection pattern.

        Args:
            name: Pattern name.
            pattern: Regex pattern.
            risk_level: Risk level for matches.
        """
        self._custom_patterns.append((name, re.compile(pattern, re.I), risk_level))

    def analyze(self, text: str) -> List[SafetyFlag]:
        """Analyze text for toxicity.

        Args:
            text: Text to analyze.

        Returns:
            List of safety flags.
        """
        flags = []
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        # Check profanity
        profanity_found = words & self.PROFANITY_INDICATORS
        if profanity_found:
            flags.append(SafetyFlag(
                category=SafetyCategory.TOXICITY,
                risk_level=RiskLevel.LOW,
                description="Contains mild profanity",
                matched_text=", ".join(profanity_found),
                confidence=0.7,
            ))

        # Check threat patterns
        for pattern in self.THREAT_PATTERNS:
            match = pattern.search(text)
            if match:
                flags.append(SafetyFlag(
                    category=SafetyCategory.VIOLENCE,
                    risk_level=RiskLevel.HIGH,
                    description="Contains potential threat language",
                    matched_text=match.group(),
                    confidence=0.8,
                ))
                break

        # Check harassment patterns
        for pattern in self.HARASSMENT_PATTERNS:
            match = pattern.search(text)
            if match:
                flags.append(SafetyFlag(
                    category=SafetyCategory.TOXICITY,
                    risk_level=RiskLevel.MEDIUM,
                    description="Contains potential harassment",
                    matched_text=match.group(),
                    confidence=0.7,
                ))
                break

        # Check custom patterns
        for name, pattern, risk_level in self._custom_patterns:
            match = pattern.search(text)
            if match:
                flags.append(SafetyFlag(
                    category=SafetyCategory.TOXICITY,
                    risk_level=risk_level,
                    description=f"Matched custom pattern: {name}",
                    matched_text=match.group(),
                    confidence=0.9,
                ))

        return flags


class HallucinationDetector:
    """Detect potential hallucination indicators in LLM outputs."""

    # Phrases that may indicate uncertainty or fabrication
    UNCERTAINTY_PHRASES = [
        "i think", "i believe", "i'm not sure", "i'm not certain",
        "i assume", "i suppose", "might be", "could be",
        "possibly", "perhaps", "probably", "likely",
    ]

    CONFIDENT_FALSE_INDICATORS = [
        "as everyone knows",
        "it's well known that",
        "obviously",
        "clearly",
        "definitely",
        "certainly",
        "without a doubt",
    ]

    # Patterns suggesting made-up citations
    FAKE_CITATION_PATTERNS = [
        re.compile(r"according to (?:a |the )?(?:recent )?(?:study|research|report)", re.I),
        re.compile(r"studies (?:have )?show(?:n|s)?", re.I),
        re.compile(r"research (?:has )?(?:shown|indicates|suggests)", re.I),
        re.compile(r"experts (?:say|believe|agree)", re.I),
    ]

    # Patterns for specific claims
    SPECIFIC_CLAIM_PATTERNS = [
        re.compile(r"\b\d+(?:\.\d+)?%\b"),  # Percentages
        re.compile(r"\$\d+(?:,\d{3})*(?:\.\d{2})?\b"),  # Dollar amounts
        re.compile(r"\b(?:19|20)\d{2}\b"),  # Years
    ]

    def __init__(self):
        """Initialize the detector."""
        pass

    def analyze(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Analyze text for hallucination indicators.

        Args:
            text: Text to analyze.
            context: Optional context/prompt for comparison.

        Returns:
            Analysis results.
        """
        text_lower = text.lower()

        # Count uncertainty indicators
        uncertainty_count = sum(
            1 for phrase in self.UNCERTAINTY_PHRASES
            if phrase in text_lower
        )

        # Count overconfident phrases
        overconfidence_count = sum(
            1 for phrase in self.CONFIDENT_FALSE_INDICATORS
            if phrase in text_lower
        )

        # Count vague citations
        citation_matches = []
        for pattern in self.FAKE_CITATION_PATTERNS:
            matches = pattern.findall(text)
            citation_matches.extend(matches)

        # Count specific claims
        specific_claims = []
        for pattern in self.SPECIFIC_CLAIM_PATTERNS:
            matches = pattern.findall(text)
            specific_claims.extend(matches)

        # Calculate risk score
        risk_score = 0.0
        risk_score += min(0.3, uncertainty_count * 0.05)  # Some uncertainty is OK
        risk_score += min(0.3, overconfidence_count * 0.1)  # Overconfidence is risky
        risk_score += min(0.3, len(citation_matches) * 0.15)  # Vague citations
        risk_score += min(0.2, len(specific_claims) * 0.05)  # Unverifiable specifics

        return {
            "risk_score": min(1.0, risk_score),
            "uncertainty_count": uncertainty_count,
            "overconfidence_count": overconfidence_count,
            "vague_citations": citation_matches,
            "specific_claims": specific_claims,
            "indicators": {
                "has_uncertainty": uncertainty_count > 0,
                "has_overconfidence": overconfidence_count > 0,
                "has_vague_citations": len(citation_matches) > 0,
                "has_specific_claims": len(specific_claims) > 0,
            },
        }

    def get_risk_level(self, analysis: Dict[str, Any]) -> RiskLevel:
        """Get risk level from analysis."""
        score = analysis.get("risk_score", 0)
        if score < 0.2:
            return RiskLevel.LOW
        elif score < 0.4:
            return RiskLevel.MEDIUM
        elif score < 0.7:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL


class BiasDetector:
    """Detect potential bias patterns in text."""

    # Gender-related terms for balance checking
    GENDER_TERMS = {
        "male": {"he", "him", "his", "man", "men", "boy", "boys", "male", "father", "son"},
        "female": {"she", "her", "hers", "woman", "women", "girl", "girls", "female", "mother", "daughter"},
    }

    # Stereotyping patterns
    STEREOTYPE_PATTERNS = [
        re.compile(r"\b(all|every|most)\s+(men|women|people from)\b", re.I),
        re.compile(r"\b(men|women)\s+are\s+(always|never|usually)\b", re.I),
        re.compile(r"\b(typical|stereotypical)\s+\w+\b", re.I),
    ]

    # Absolutes that may indicate bias
    ABSOLUTE_TERMS = {
        "always", "never", "all", "none", "every", "ever",
        "everyone", "nobody", "completely", "absolutely",
    }

    # Multi-word absolutes to check separately
    ABSOLUTE_PHRASES = [
        "no one",
    ]

    def __init__(self):
        """Initialize the detector."""
        pass

    def analyze_gender_balance(self, text: str) -> Dict[str, Any]:
        """Analyze gender representation balance.

        Args:
            text: Text to analyze.

        Returns:
            Gender balance analysis.
        """
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        male_count = len(words & self.GENDER_TERMS["male"])
        female_count = len(words & self.GENDER_TERMS["female"])
        total = male_count + female_count

        if total == 0:
            balance = 0.5  # No gender terms = neutral
        else:
            balance = male_count / total

        return {
            "male_terms": male_count,
            "female_terms": female_count,
            "balance_ratio": balance,  # 0.5 = balanced, 1.0 = all male, 0.0 = all female
            "is_balanced": 0.3 <= balance <= 0.7 if total > 2 else True,
        }

    def analyze_stereotypes(self, text: str) -> List[str]:
        """Detect potential stereotyping language.

        Args:
            text: Text to analyze.

        Returns:
            List of matched stereotype patterns.
        """
        matches = []
        for pattern in self.STEREOTYPE_PATTERNS:
            found = pattern.findall(text)
            if found:
                if isinstance(found[0], str):
                    matches.extend(found)
                else:
                    matches.extend([" ".join(f) for f in found])
        return matches

    def analyze_absolutes(self, text: str) -> Dict[str, Any]:
        """Analyze use of absolute language.

        Args:
            text: Text to analyze.

        Returns:
            Analysis of absolute term usage.
        """
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        word_set = set(words)

        absolutes_found = word_set & self.ABSOLUTE_TERMS
        absolute_count = sum(1 for w in words if w in self.ABSOLUTE_TERMS)

        # Also check for multi-word absolute phrases
        for phrase in self.ABSOLUTE_PHRASES:
            if phrase in text_lower:
                absolutes_found.add(phrase)
                absolute_count += text_lower.count(phrase)

        return {
            "absolutes_found": list(absolutes_found),
            "absolute_count": absolute_count,
            "word_count": len(words),
            "absolute_ratio": absolute_count / len(words) if words else 0,
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """Comprehensive bias analysis.

        Args:
            text: Text to analyze.

        Returns:
            Full bias analysis.
        """
        gender = self.analyze_gender_balance(text)
        stereotypes = self.analyze_stereotypes(text)
        absolutes = self.analyze_absolutes(text)

        # Calculate overall bias score
        bias_score = 0.0

        # Gender imbalance
        if not gender["is_balanced"]:
            bias_score += 0.3

        # Stereotypes
        bias_score += min(0.4, len(stereotypes) * 0.2)

        # Excessive absolutes
        if absolutes["absolute_ratio"] > 0.05:
            bias_score += 0.2

        return {
            "bias_score": min(1.0, bias_score),
            "gender_balance": gender,
            "stereotypes": stereotypes,
            "absolutes": absolutes,
            "has_potential_bias": bias_score > 0.3,
        }


class ContentSafetyAnalyzer:
    """Comprehensive content safety analyzer."""

    def __init__(self):
        """Initialize analyzers."""
        self.pii_detector = PIIDetector()
        self.toxicity_analyzer = ToxicityAnalyzer()
        self.hallucination_detector = HallucinationDetector()
        self.bias_detector = BiasDetector()

    def analyze(
        self,
        text: str,
        check_pii: bool = True,
        check_toxicity: bool = True,
        check_hallucination: bool = True,
        check_bias: bool = True,
    ) -> SafetyReport:
        """Perform comprehensive safety analysis.

        Args:
            text: Text to analyze.
            check_pii: Whether to check for PII.
            check_toxicity: Whether to check for toxicity.
            check_hallucination: Whether to check for hallucination indicators.
            check_bias: Whether to check for bias.

        Returns:
            Comprehensive safety report.
        """
        flags: List[SafetyFlag] = []
        scores: Dict[str, float] = {}

        # PII detection
        if check_pii:
            pii_report = self.pii_detector.detect(text)
            if pii_report.has_pii:
                flags.append(SafetyFlag(
                    category=SafetyCategory.PII_EXPOSURE,
                    risk_level=RiskLevel.HIGH,
                    description=f"Found {len(pii_report.matches)} PII items",
                    metadata={"pii_types": list(set(m.pii_type for m in pii_report.matches))},
                ))
            scores["pii_count"] = len(pii_report.matches)

        # Toxicity analysis
        if check_toxicity:
            toxicity_flags = self.toxicity_analyzer.analyze(text)
            flags.extend(toxicity_flags)
            scores["toxicity_flags"] = len(toxicity_flags)

        # Hallucination detection
        if check_hallucination:
            hallucination_analysis = self.hallucination_detector.analyze(text)
            risk_level = self.hallucination_detector.get_risk_level(hallucination_analysis)
            if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                flags.append(SafetyFlag(
                    category=SafetyCategory.MISINFORMATION,
                    risk_level=risk_level,
                    description="High hallucination risk indicators",
                    confidence=hallucination_analysis["risk_score"],
                    metadata=hallucination_analysis["indicators"],
                ))
            scores["hallucination_risk"] = hallucination_analysis["risk_score"]

        # Bias detection
        if check_bias:
            bias_analysis = self.bias_detector.analyze(text)
            if bias_analysis["has_potential_bias"]:
                flags.append(SafetyFlag(
                    category=SafetyCategory.TOXICITY,
                    risk_level=RiskLevel.MEDIUM,
                    description="Potential bias detected",
                    confidence=bias_analysis["bias_score"],
                    metadata={
                        "gender_balanced": bias_analysis["gender_balance"]["is_balanced"],
                        "stereotypes_found": len(bias_analysis["stereotypes"]) > 0,
                    },
                ))
            scores["bias_score"] = bias_analysis["bias_score"]

        # Determine overall risk
        if not flags:
            overall_risk = RiskLevel.NONE
        else:
            risk_levels = [f.risk_level for f in flags]
            if RiskLevel.CRITICAL in risk_levels:
                overall_risk = RiskLevel.CRITICAL
            elif RiskLevel.HIGH in risk_levels:
                overall_risk = RiskLevel.HIGH
            elif RiskLevel.MEDIUM in risk_levels:
                overall_risk = RiskLevel.MEDIUM
            else:
                overall_risk = RiskLevel.LOW

        return SafetyReport(
            text=text,
            is_safe=overall_risk in (RiskLevel.NONE, RiskLevel.LOW),
            overall_risk=overall_risk,
            flags=flags,
            scores=scores,
        )


def quick_safety_check(text: str) -> Tuple[bool, RiskLevel, List[str]]:
    """Quick safety check for text.

    Args:
        text: Text to check.

    Returns:
        Tuple of (is_safe, risk_level, list of issues).
    """
    analyzer = ContentSafetyAnalyzer()
    report = analyzer.analyze(text)

    issues = [f.description for f in report.flags]
    return report.is_safe, report.overall_risk, issues


def mask_pii(text: str) -> str:
    """Quick function to mask PII in text.

    Args:
        text: Text to mask.

    Returns:
        Masked text.
    """
    detector = PIIDetector()
    return detector.mask(text)


def detect_pii(text: str) -> List[Dict[str, Any]]:
    """Quick function to detect PII in text.

    Args:
        text: Text to analyze.

    Returns:
        List of PII matches as dictionaries.
    """
    detector = PIIDetector()
    report = detector.detect(text)
    return [
        {
            "type": m.pii_type,
            "value": m.value,
            "start": m.start,
            "end": m.end,
        }
        for m in report.matches
    ]
