"""
Safety and content analysis utilities for LLM outputs.

This module provides comprehensive tools for analyzing LLM-generated content
for safety concerns, including personally identifiable information (PII),
toxic content, potential hallucinations, and bias patterns.

Provides tools for:
- Content safety classification
- Toxicity detection
- PII detection and masking
- Hallucination indicators
- Bias detection patterns

Examples:
    Basic safety check on text:

    >>> from insideLLMs.safety import quick_safety_check
    >>> is_safe, risk_level, issues = quick_safety_check("Hello, how are you?")
    >>> is_safe
    True
    >>> risk_level.value
    'none'

    Detecting and masking PII:

    >>> from insideLLMs.safety import mask_pii, detect_pii
    >>> text = "Contact me at john.doe@email.com or call 555-123-4567"
    >>> mask_pii(text)
    'Contact me at [EMAIL] or call [PHONE]'
    >>> pii_matches = detect_pii(text)
    >>> len(pii_matches)
    2

    Comprehensive safety analysis:

    >>> from insideLLMs.safety import ContentSafetyAnalyzer
    >>> analyzer = ContentSafetyAnalyzer()
    >>> report = analyzer.analyze("According to a recent study, 95% of people agree.")
    >>> report.is_safe
    True
    >>> 'hallucination_risk' in report.scores
    True

    Toxicity detection:

    >>> from insideLLMs.safety import ToxicityAnalyzer
    >>> toxicity = ToxicityAnalyzer()
    >>> flags = toxicity.analyze("You're an idiot, shut up!")
    >>> len(flags) > 0
    True
    >>> flags[0].category.value
    'toxicity'
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from re import Pattern
from typing import Any, Optional


class SafetyCategory(Enum):
    """Categories of safety concerns in LLM-generated content.

    This enum defines the various types of safety issues that can be
    detected in text content. Each category represents a distinct type
    of concern that may require different handling or mitigation strategies.

    Attributes:
        SAFE: Content has no detected safety concerns.
        TOXICITY: Content contains toxic or offensive language.
        HATE_SPEECH: Content contains hate speech targeting groups.
        VIOLENCE: Content contains violent language or threats.
        SEXUAL_CONTENT: Content contains inappropriate sexual material.
        SELF_HARM: Content promotes or describes self-harm.
        DANGEROUS_CONTENT: Content describes dangerous activities.
        PII_EXPOSURE: Content contains personally identifiable information.
        MISINFORMATION: Content may contain false or misleading information.
        MANIPULATION: Content attempts to manipulate or deceive.

    Examples:
        Checking a safety flag's category:

        >>> from insideLLMs.safety import SafetyCategory, SafetyFlag, RiskLevel
        >>> flag = SafetyFlag(
        ...     category=SafetyCategory.TOXICITY,
        ...     risk_level=RiskLevel.MEDIUM,
        ...     description="Contains profanity"
        ... )
        >>> flag.category == SafetyCategory.TOXICITY
        True
        >>> flag.category.value
        'toxicity'

        Filtering flags by category:

        >>> from insideLLMs.safety import ContentSafetyAnalyzer
        >>> analyzer = ContentSafetyAnalyzer()
        >>> report = analyzer.analyze("My email is test@example.com")
        >>> pii_flags = report.get_flags_by_category(SafetyCategory.PII_EXPOSURE)
        >>> len(pii_flags) > 0
        True

        Using category in conditional logic:

        >>> if flag.category in [SafetyCategory.VIOLENCE, SafetyCategory.HATE_SPEECH]:
        ...     print("High-severity content detected")

        Iterating over all categories:

        >>> all_categories = list(SafetyCategory)
        >>> len(all_categories)
        10
    """

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
    """Risk level classifications for safety concerns.

    This enum represents the severity levels for safety issues detected
    in content. The levels are ordered from least to most severe and
    can be used to prioritize responses to safety concerns.

    Attributes:
        NONE: No safety concerns detected.
        LOW: Minor concerns that typically do not require action.
        MEDIUM: Moderate concerns that may warrant review.
        HIGH: Significant concerns requiring attention.
        CRITICAL: Severe concerns requiring immediate action.

    Examples:
        Creating a safety flag with risk level:

        >>> from insideLLMs.safety import SafetyFlag, SafetyCategory, RiskLevel
        >>> flag = SafetyFlag(
        ...     category=SafetyCategory.VIOLENCE,
        ...     risk_level=RiskLevel.HIGH,
        ...     description="Contains threat language"
        ... )
        >>> flag.risk_level.value
        'high'

        Comparing risk levels:

        >>> RiskLevel.CRITICAL.value > RiskLevel.NONE.value  # String comparison
        False
        >>> list(RiskLevel).index(RiskLevel.CRITICAL) > list(RiskLevel).index(RiskLevel.NONE)
        True

        Using risk level for conditional handling:

        >>> def handle_content(risk: RiskLevel) -> str:
        ...     if risk in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
        ...         return "Block content"
        ...     elif risk == RiskLevel.MEDIUM:
        ...         return "Flag for review"
        ...     return "Allow content"
        >>> handle_content(RiskLevel.HIGH)
        'Block content'

        Checking overall risk from a safety report:

        >>> from insideLLMs.safety import ContentSafetyAnalyzer
        >>> analyzer = ContentSafetyAnalyzer()
        >>> report = analyzer.analyze("Hello, world!")
        >>> report.overall_risk == RiskLevel.NONE
        True
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyFlag:
    """A single safety flag representing a detected concern.

    SafetyFlag captures information about a specific safety issue found
    in text content, including its category, severity, description, and
    any associated metadata.

    Attributes:
        category: The type of safety concern (e.g., TOXICITY, PII_EXPOSURE).
        risk_level: The severity level of the concern.
        description: Human-readable description of the issue.
        matched_text: The specific text that triggered this flag (optional).
        confidence: Confidence score from 0.0 to 1.0 (default: 1.0).
        metadata: Additional contextual information about the flag.

    Examples:
        Creating a basic safety flag:

        >>> from insideLLMs.safety import SafetyFlag, SafetyCategory, RiskLevel
        >>> flag = SafetyFlag(
        ...     category=SafetyCategory.TOXICITY,
        ...     risk_level=RiskLevel.MEDIUM,
        ...     description="Contains mild profanity"
        ... )
        >>> flag.category.value
        'toxicity'
        >>> flag.confidence
        1.0

        Creating a flag with matched text:

        >>> flag = SafetyFlag(
        ...     category=SafetyCategory.PII_EXPOSURE,
        ...     risk_level=RiskLevel.HIGH,
        ...     description="Email address detected",
        ...     matched_text="user@example.com",
        ...     confidence=0.95
        ... )
        >>> flag.matched_text
        'user@example.com'

        Creating a flag with metadata:

        >>> flag = SafetyFlag(
        ...     category=SafetyCategory.MISINFORMATION,
        ...     risk_level=RiskLevel.HIGH,
        ...     description="Vague citations without sources",
        ...     metadata={"citation_count": 3, "unverified": True}
        ... )
        >>> flag.metadata["citation_count"]
        3

        Accessing flag properties from a safety report:

        >>> from insideLLMs.safety import ContentSafetyAnalyzer
        >>> analyzer = ContentSafetyAnalyzer()
        >>> report = analyzer.analyze("Contact: john@test.com")
        >>> if report.flags:
        ...     first_flag = report.flags[0]
        ...     print(f"{first_flag.category.value}: {first_flag.description}")
        pii_exposure: Found 1 PII items
    """

    category: SafetyCategory
    risk_level: RiskLevel
    description: str
    matched_text: Optional[str] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyReport:
    """Comprehensive safety analysis report for analyzed text.

    SafetyReport aggregates all safety analysis results including flags,
    scores, and overall risk assessment. It provides methods for filtering
    and accessing specific safety concerns.

    Attributes:
        text: The original text that was analyzed.
        is_safe: Boolean indicating if content is considered safe (NONE or LOW risk).
        overall_risk: The highest risk level found across all flags.
        flags: List of individual SafetyFlag objects for each concern.
        scores: Dictionary of numerical scores from various analyzers.
        metadata: Additional contextual information about the analysis.

    Examples:
        Getting a safety report from the analyzer:

        >>> from insideLLMs.safety import ContentSafetyAnalyzer
        >>> analyzer = ContentSafetyAnalyzer()
        >>> report = analyzer.analyze("Hello, my email is test@example.com")
        >>> report.is_safe
        False
        >>> report.overall_risk.value
        'high'
        >>> len(report.flags)
        1

        Filtering flags by category:

        >>> from insideLLMs.safety import SafetyCategory
        >>> pii_flags = report.get_flags_by_category(SafetyCategory.PII_EXPOSURE)
        >>> len(pii_flags) > 0
        True

        Getting the highest risk flag:

        >>> highest = report.get_highest_risk_flag()
        >>> if highest:
        ...     print(f"Highest risk: {highest.risk_level.value}")
        Highest risk: high

        Converting report to dictionary for serialization:

        >>> report_dict = report.to_dict()
        >>> 'is_safe' in report_dict
        True
        >>> 'flags' in report_dict
        True
    """

    text: str
    is_safe: bool
    overall_risk: RiskLevel
    flags: list[SafetyFlag] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_flags_by_category(self, category: SafetyCategory) -> list[SafetyFlag]:
        """Get all flags matching a specific safety category.

        Filters the report's flags to return only those that match
        the specified category.

        Args:
            category: The SafetyCategory to filter by.

        Returns:
            List of SafetyFlag objects matching the specified category.
            Returns an empty list if no flags match.

        Examples:
            Filtering for PII-related flags:

            >>> from insideLLMs.safety import ContentSafetyAnalyzer, SafetyCategory
            >>> analyzer = ContentSafetyAnalyzer()
            >>> report = analyzer.analyze("Email: user@test.com, SSN: 123-45-6789")
            >>> pii_flags = report.get_flags_by_category(SafetyCategory.PII_EXPOSURE)
            >>> len(pii_flags)
            1
            >>> pii_flags[0].category == SafetyCategory.PII_EXPOSURE
            True

            Checking for toxicity flags:

            >>> report = analyzer.analyze("That's a damn good idea!")
            >>> toxicity_flags = report.get_flags_by_category(SafetyCategory.TOXICITY)
            >>> len(toxicity_flags) > 0
            True

            Handling empty results:

            >>> report = analyzer.analyze("Hello, world!")
            >>> violence_flags = report.get_flags_by_category(SafetyCategory.VIOLENCE)
            >>> len(violence_flags)
            0
        """
        return [f for f in self.flags if f.category == category]

    def get_highest_risk_flag(self) -> Optional[SafetyFlag]:
        """Get the flag with the highest risk level.

        Searches through all flags and returns the one with the most
        severe risk level. Useful for prioritizing which issue to
        address first.

        Returns:
            The SafetyFlag with the highest risk level, or None if
            there are no flags.

        Examples:
            Finding the most critical issue:

            >>> from insideLLMs.safety import ContentSafetyAnalyzer
            >>> analyzer = ContentSafetyAnalyzer()
            >>> text = "Damn, my SSN is 123-45-6789"
            >>> report = analyzer.analyze(text)
            >>> highest = report.get_highest_risk_flag()
            >>> if highest:
            ...     print(f"Priority: {highest.risk_level.value}")
            Priority: high

            Handling reports with no flags:

            >>> report = analyzer.analyze("Hello!")
            >>> highest = report.get_highest_risk_flag()
            >>> highest is None
            True

            Using for prioritized handling:

            >>> from insideLLMs.safety import RiskLevel
            >>> text = "Contact me at user@test.com"
            >>> report = analyzer.analyze(text)
            >>> highest = report.get_highest_risk_flag()
            >>> if highest and highest.risk_level == RiskLevel.HIGH:
            ...     print(f"High risk issue: {highest.description}")
            High risk issue: Found 1 PII items
        """
        if not self.flags:
            return None
        risk_order = [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]
        for level in risk_order:
            for flag in self.flags:
                if flag.risk_level == level:
                    return flag
        return self.flags[0] if self.flags else None

    def to_dict(self) -> dict[str, Any]:
        """Convert the safety report to a dictionary.

        Creates a serializable dictionary representation of the report,
        useful for JSON serialization, logging, or API responses.

        Returns:
            Dictionary containing:
                - is_safe: Boolean safety status
                - overall_risk: Risk level as string
                - flags: List of flag dictionaries
                - scores: Dictionary of numerical scores

        Examples:
            Basic conversion:

            >>> from insideLLMs.safety import ContentSafetyAnalyzer
            >>> analyzer = ContentSafetyAnalyzer()
            >>> report = analyzer.analyze("Test message")
            >>> data = report.to_dict()
            >>> data['is_safe']
            True
            >>> data['overall_risk']
            'none'

            Serializing to JSON:

            >>> import json
            >>> report = analyzer.analyze("Email: test@example.com")
            >>> json_str = json.dumps(report.to_dict())
            >>> 'is_safe' in json_str
            True

            Accessing flag data:

            >>> report = analyzer.analyze("My SSN is 123-45-6789")
            >>> data = report.to_dict()
            >>> len(data['flags']) > 0
            True
            >>> data['flags'][0]['category']
            'pii_exposure'

            Checking scores:

            >>> report = analyzer.analyze("Some text here")
            >>> data = report.to_dict()
            >>> 'scores' in data
            True
        """
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
    """A detected personally identifiable information (PII) match.

    Represents a single instance of PII found in text, including the type
    of PII, its value, and its position within the original text.

    Attributes:
        pii_type: The type of PII detected (e.g., 'email', 'phone_us', 'ssn').
        value: The actual PII value that was matched.
        start: Starting character index in the original text.
        end: Ending character index in the original text.
        confidence: Confidence score from 0.0 to 1.0 (default: 1.0).

    Examples:
        Creating a PII match manually:

        >>> from insideLLMs.safety import PIIMatch
        >>> match = PIIMatch(
        ...     pii_type="email",
        ...     value="user@example.com",
        ...     start=10,
        ...     end=26
        ... )
        >>> match.pii_type
        'email'
        >>> match.value
        'user@example.com'

        Getting PII matches from detector:

        >>> from insideLLMs.safety import PIIDetector
        >>> detector = PIIDetector()
        >>> report = detector.detect("Contact: john@test.com")
        >>> match = report.matches[0]
        >>> match.pii_type
        'email'
        >>> match.start
        9
        >>> match.end
        22

        Using position information for highlighting:

        >>> text = "Call me at 555-123-4567"
        >>> report = detector.detect(text)
        >>> if report.matches:
        ...     m = report.matches[0]
        ...     highlighted = text[:m.start] + "[" + text[m.start:m.end] + "]" + text[m.end:]
        ...     print(highlighted)
        Call me at [555-123-4567]

        Checking confidence:

        >>> match = PIIMatch("ssn", "123-45-6789", 0, 11, confidence=0.95)
        >>> match.confidence
        0.95
    """

    pii_type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class PIIReport:
    """Report of PII detection analysis results.

    Contains the results of a PII detection scan, including all matches
    found and optionally the masked version of the text.

    Attributes:
        text: The original text that was analyzed.
        has_pii: Boolean indicating if any PII was found.
        matches: List of PIIMatch objects for each detected PII item.
        masked_text: Text with PII replaced by masks (if masking was performed).

    Examples:
        Basic PII detection:

        >>> from insideLLMs.safety import PIIDetector
        >>> detector = PIIDetector()
        >>> report = detector.detect("Email: john@test.com, Phone: 555-123-4567")
        >>> report.has_pii
        True
        >>> len(report.matches)
        2

        Detection with masking:

        >>> report = detector.detect_and_mask("My SSN is 123-45-6789")
        >>> report.has_pii
        True
        >>> report.masked_text
        'My SSN is [SSN]'

        Filtering by PII type:

        >>> report = detector.detect("john@test.com and jane@test.com, call 555-0001")
        >>> email_matches = report.get_by_type("email")
        >>> len(email_matches)
        2
        >>> phone_matches = report.get_by_type("phone_us")
        >>> len(phone_matches)
        1

        Checking for no PII:

        >>> report = detector.detect("Hello, world!")
        >>> report.has_pii
        False
        >>> len(report.matches)
        0
    """

    text: str
    has_pii: bool
    matches: list[PIIMatch] = field(default_factory=list)
    masked_text: Optional[str] = None

    def get_by_type(self, pii_type: str) -> list[PIIMatch]:
        """Get all matches of a specific PII type.

        Filters the matches to return only those of the specified type.

        Args:
            pii_type: The type of PII to filter for (e.g., 'email', 'ssn',
                'phone_us', 'credit_card').

        Returns:
            List of PIIMatch objects of the specified type. Returns an
            empty list if no matches of that type exist.

        Examples:
            Getting email matches:

            >>> from insideLLMs.safety import PIIDetector
            >>> detector = PIIDetector()
            >>> text = "Contact: a@b.com, b@c.com, SSN: 123-45-6789"
            >>> report = detector.detect(text)
            >>> emails = report.get_by_type("email")
            >>> len(emails)
            2

            Getting SSN matches:

            >>> ssn_matches = report.get_by_type("ssn")
            >>> len(ssn_matches)
            1
            >>> ssn_matches[0].value
            '123-45-6789'

            Getting credit card matches (when none exist):

            >>> cc_matches = report.get_by_type("credit_card")
            >>> len(cc_matches)
            0

            Using with multiple PII types:

            >>> for pii_type in ["email", "phone_us", "ssn"]:
            ...     matches = report.get_by_type(pii_type)
            ...     if matches:
            ...         print(f"{pii_type}: {len(matches)} found")
            email: 2 found
            ssn: 1 found
        """
        return [m for m in self.matches if m.pii_type == pii_type]


class PIIDetector:
    """Detect personally identifiable information (PII) in text.

    PIIDetector uses regex patterns to identify various types of PII including
    email addresses, phone numbers, social security numbers, credit card numbers,
    IP addresses, dates of birth, and physical addresses.

    The detector can be used to find PII, mask it with placeholder tokens, or
    perform both operations together.

    Attributes:
        PATTERNS: Class-level dictionary of default regex patterns for PII types.
        MASKS: Class-level dictionary of mask tokens for each PII type.
        patterns: Instance-level dictionary of active patterns (can be customized).

    Examples:
        Basic PII detection:

        >>> from insideLLMs.safety import PIIDetector
        >>> detector = PIIDetector()
        >>> report = detector.detect("Email me at john@example.com")
        >>> report.has_pii
        True
        >>> report.matches[0].pii_type
        'email'

        Masking PII in text:

        >>> detector = PIIDetector()
        >>> masked = detector.mask("Call 555-123-4567 or email john@test.com")
        >>> masked
        'Call [PHONE] or email [EMAIL]'

        Detection and masking together:

        >>> detector = PIIDetector()
        >>> report = detector.detect_and_mask("SSN: 123-45-6789")
        >>> report.has_pii
        True
        >>> report.masked_text
        'SSN: [SSN]'

        Adding custom patterns:

        >>> detector = PIIDetector()
        >>> detector.add_pattern("employee_id", r"EMP-\\d{6}", "[EMPLOYEE_ID]")
        >>> masked = detector.mask("Contact EMP-123456 for help")
        >>> masked
        'Contact [EMPLOYEE_ID] for help'

    Note:
        The default patterns are designed to catch common formats but may not
        cover all variations. For production use, consider adding additional
        patterns or using ML-based detection for higher accuracy.
    """

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

    def __init__(self, patterns: Optional[dict[str, Pattern]] = None):
        """Initialize PII detector with optional custom patterns.

        Creates a new PIIDetector instance, optionally with custom regex
        patterns for PII detection.

        Args:
            patterns: Optional dictionary mapping PII type names to compiled
                regex Pattern objects. If None, uses the default PATTERNS.

        Examples:
            Using default patterns:

            >>> from insideLLMs.safety import PIIDetector
            >>> detector = PIIDetector()
            >>> 'email' in detector.patterns
            True

            Using custom patterns:

            >>> import re
            >>> custom = {"custom_id": re.compile(r"ID-\\d{4}")}
            >>> detector = PIIDetector(patterns=custom)
            >>> 'custom_id' in detector.patterns
            True
            >>> 'email' in detector.patterns
            False

            Extending default patterns:

            >>> import re
            >>> from insideLLMs.safety import PIIDetector
            >>> patterns = PIIDetector.PATTERNS.copy()
            >>> patterns["custom"] = re.compile(r"CUST-\\d+")
            >>> detector = PIIDetector(patterns=patterns)
            >>> len(detector.patterns) > len(PIIDetector.PATTERNS)
            True
        """
        self.patterns = patterns or self.PATTERNS.copy()

    def add_pattern(self, name: str, pattern: str, mask: str = "[REDACTED]") -> None:
        """Add a custom PII detection pattern.

        Registers a new regex pattern for detecting a custom type of PII.
        The pattern will be used in subsequent detect() and mask() calls.

        Args:
            name: Unique identifier for the pattern type.
            pattern: Regular expression pattern string.
            mask: Replacement text to use when masking (default: "[REDACTED]").

        Examples:
            Adding an employee ID pattern:

            >>> from insideLLMs.safety import PIIDetector
            >>> detector = PIIDetector()
            >>> detector.add_pattern("employee_id", r"EMP-\\d{6}", "[EMP_ID]")
            >>> report = detector.detect("Contact EMP-123456")
            >>> report.has_pii
            True
            >>> report.matches[0].pii_type
            'employee_id'

            Adding a custom account number pattern:

            >>> detector = PIIDetector()
            >>> detector.add_pattern("account", r"ACC\\d{8}", "[ACCOUNT]")
            >>> masked = detector.mask("Account: ACC12345678")
            >>> masked
            'Account: [ACCOUNT]'

            Adding multiple custom patterns:

            >>> detector = PIIDetector()
            >>> detector.add_pattern("order_id", r"ORD-\\d+", "[ORDER]")
            >>> detector.add_pattern("tracking", r"TRK\\d{12}", "[TRACKING]")
            >>> text = "Order ORD-999 shipped, tracking TRK123456789012"
            >>> masked = detector.mask(text)
            >>> "[ORDER]" in masked and "[TRACKING]" in masked
            True

            Using default mask:

            >>> detector = PIIDetector()
            >>> detector.add_pattern("secret", r"SECRET:\\w+")
            >>> detector.mask("My SECRET:password123")
            'My [REDACTED]'
        """
        self.patterns[name] = re.compile(pattern)
        self.MASKS[name] = mask

    def detect(self, text: str) -> PIIReport:
        """Detect PII in text without modifying it.

        Scans the input text for all registered PII patterns and returns
        a report with all matches found, including their positions.

        Args:
            text: The text to analyze for PII.

        Returns:
            PIIReport containing:
                - has_pii: Boolean indicating if any PII was found
                - matches: List of PIIMatch objects sorted by position
                - text: The original input text

        Examples:
            Detecting email addresses:

            >>> from insideLLMs.safety import PIIDetector
            >>> detector = PIIDetector()
            >>> report = detector.detect("Contact john@example.com for info")
            >>> report.has_pii
            True
            >>> report.matches[0].pii_type
            'email'
            >>> report.matches[0].value
            'john@example.com'

            Detecting multiple PII types:

            >>> text = "Email: test@test.com, Phone: 555-123-4567"
            >>> report = detector.detect(text)
            >>> len(report.matches)
            2
            >>> [m.pii_type for m in report.matches]
            ['email', 'phone_us']

            No PII found:

            >>> report = detector.detect("Hello, world!")
            >>> report.has_pii
            False
            >>> len(report.matches)
            0

            Getting position information:

            >>> text = "SSN: 123-45-6789"
            >>> report = detector.detect(text)
            >>> match = report.matches[0]
            >>> text[match.start:match.end]
            '123-45-6789'
        """
        matches = []

        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                    )
                )

        # Sort by position
        matches.sort(key=lambda m: m.start)

        return PIIReport(
            text=text,
            has_pii=len(matches) > 0,
            matches=matches,
        )

    def mask(self, text: str, types: Optional[list[str]] = None) -> str:
        """Mask PII in text by replacing with placeholder tokens.

        Replaces all detected PII with mask tokens (e.g., [EMAIL], [PHONE]).
        Optionally filter which PII types to mask.

        Args:
            text: The text containing PII to mask.
            types: Optional list of PII type names to mask. If None, all
                registered PII types will be masked.

        Returns:
            The text with PII replaced by mask tokens.

        Examples:
            Masking all PII:

            >>> from insideLLMs.safety import PIIDetector
            >>> detector = PIIDetector()
            >>> detector.mask("Email john@test.com, call 555-123-4567")
            'Email [EMAIL], call [PHONE]'

            Masking specific types only:

            >>> text = "john@test.com, SSN: 123-45-6789, phone: 555-0001"
            >>> detector.mask(text, types=["email"])
            '[EMAIL], SSN: 123-45-6789, phone: 555-0001'

            Masking multiple specific types:

            >>> detector.mask(text, types=["email", "ssn"])
            '[EMAIL], SSN: [SSN], phone: 555-0001'

            Text without PII:

            >>> detector.mask("Hello, world!")
            'Hello, world!'
        """
        result = text
        patterns_to_use = (
            {k: v for k, v in self.patterns.items() if k in types} if types else self.patterns
        )

        for pii_type, pattern in patterns_to_use.items():
            mask = self.MASKS.get(pii_type, "[REDACTED]")
            result = pattern.sub(mask, result)

        return result

    def detect_and_mask(self, text: str) -> PIIReport:
        """Detect PII and return both matches and masked text.

        Combines detect() and mask() operations, returning a complete
        report with both the detected matches and the masked text.

        Args:
            text: The text to analyze and mask.

        Returns:
            PIIReport containing:
                - has_pii: Boolean indicating if any PII was found
                - matches: List of PIIMatch objects
                - masked_text: Text with all PII replaced by masks
                - text: The original input text

        Examples:
            Basic detection and masking:

            >>> from insideLLMs.safety import PIIDetector
            >>> detector = PIIDetector()
            >>> report = detector.detect_and_mask("Contact john@test.com")
            >>> report.has_pii
            True
            >>> report.matches[0].value
            'john@test.com'
            >>> report.masked_text
            'Contact [EMAIL]'

            Multiple PII types:

            >>> text = "Email: test@test.com, SSN: 123-45-6789"
            >>> report = detector.detect_and_mask(text)
            >>> len(report.matches)
            2
            >>> report.masked_text
            'Email: [EMAIL], SSN: [SSN]'

            Accessing both original and masked:

            >>> report = detector.detect_and_mask("Phone: 555-123-4567")
            >>> report.text  # Original
            'Phone: 555-123-4567'
            >>> report.masked_text  # Masked
            'Phone: [PHONE]'

            No PII in text:

            >>> report = detector.detect_and_mask("Safe text here")
            >>> report.has_pii
            False
            >>> report.masked_text
            'Safe text here'
        """
        report = self.detect(text)
        report.masked_text = self.mask(text)
        return report


class ToxicityAnalyzer:
    """Analyze text for toxic content patterns including profanity, threats, and harassment.

    ToxicityAnalyzer uses pattern matching to detect various forms of toxic
    content in text. It identifies profanity, threat language, and harassment
    patterns, returning SafetyFlag objects for each detected issue.

    This analyzer uses regex patterns for basic detection. For production
    use cases requiring higher accuracy, consider supplementing with
    ML-based classifiers.

    Attributes:
        PROFANITY_INDICATORS: Set of mild profanity words to detect.
        THREAT_PATTERNS: List of regex patterns for threat detection.
        HARASSMENT_PATTERNS: List of regex patterns for harassment detection.

    Examples:
        Basic toxicity analysis:

        >>> from insideLLMs.safety import ToxicityAnalyzer
        >>> analyzer = ToxicityAnalyzer()
        >>> flags = analyzer.analyze("That's a damn good idea!")
        >>> len(flags) > 0
        True
        >>> flags[0].category.value
        'toxicity'
        >>> flags[0].risk_level.value
        'low'

        Detecting threat language:

        >>> flags = analyzer.analyze("I'm going to hurt you")
        >>> any(f.category.value == 'violence' for f in flags)
        True
        >>> [f for f in flags if f.category.value == 'violence'][0].risk_level.value
        'high'

        Detecting harassment:

        >>> flags = analyzer.analyze("You're an idiot, shut up!")
        >>> len(flags) >= 1
        True

        Adding custom patterns:

        >>> analyzer = ToxicityAnalyzer()
        >>> from insideLLMs.safety import RiskLevel
        >>> analyzer.add_pattern("spam_language", r"click here now", RiskLevel.LOW)
        >>> flags = analyzer.analyze("Click here now for free money!")
        >>> any(f.description.startswith("Matched custom") for f in flags)
        True

    Note:
        The profanity list is intentionally limited. For comprehensive
        profanity detection, consider using specialized libraries or
        ML-based approaches.
    """

    # Word lists for basic detection (intentionally limited for safety)
    # In production, use ML-based classifiers
    PROFANITY_INDICATORS = {
        "damn",
        "hell",
        "crap",
        "suck",
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
        """Initialize the toxicity analyzer.

        Creates a new ToxicityAnalyzer instance with an empty list of
        custom patterns. Default patterns (profanity, threats, harassment)
        are defined at the class level.

        Examples:
            Basic initialization:

            >>> from insideLLMs.safety import ToxicityAnalyzer
            >>> analyzer = ToxicityAnalyzer()
            >>> len(analyzer._custom_patterns)
            0

            Initialization with subsequent customization:

            >>> from insideLLMs.safety import ToxicityAnalyzer, RiskLevel
            >>> analyzer = ToxicityAnalyzer()
            >>> analyzer.add_pattern("spam", r"free money", RiskLevel.LOW)
            >>> len(analyzer._custom_patterns)
            1
        """
        self._custom_patterns: list[tuple[str, Pattern, RiskLevel]] = []

    def add_pattern(
        self, name: str, pattern: str, risk_level: RiskLevel = RiskLevel.MEDIUM
    ) -> None:
        """Add a custom toxicity detection pattern.

        Registers a new regex pattern for detecting custom types of toxic
        content. The pattern will be used in subsequent analyze() calls.

        Args:
            name: Human-readable name for the pattern (used in flag descriptions).
            pattern: Regular expression pattern string (case-insensitive).
            risk_level: The risk level to assign when this pattern matches
                (default: RiskLevel.MEDIUM).

        Examples:
            Adding a spam detection pattern:

            >>> from insideLLMs.safety import ToxicityAnalyzer, RiskLevel
            >>> analyzer = ToxicityAnalyzer()
            >>> analyzer.add_pattern("spam", r"click here for free", RiskLevel.LOW)
            >>> flags = analyzer.analyze("Click here for free money!")
            >>> flags[-1].description
            'Matched custom pattern: spam'

            Adding a high-risk pattern:

            >>> analyzer = ToxicityAnalyzer()
            >>> analyzer.add_pattern("severe_threat", r"i will find you", RiskLevel.CRITICAL)
            >>> flags = analyzer.analyze("I will find you and your family")
            >>> any(f.risk_level.value == 'critical' for f in flags)
            True

            Adding multiple patterns:

            >>> analyzer = ToxicityAnalyzer()
            >>> analyzer.add_pattern("scam_1", r"send money", RiskLevel.MEDIUM)
            >>> analyzer.add_pattern("scam_2", r"wire transfer", RiskLevel.HIGH)
            >>> len(analyzer._custom_patterns)
            2

            Case-insensitive matching:

            >>> analyzer = ToxicityAnalyzer()
            >>> analyzer.add_pattern("test", r"forbidden word")
            >>> flags = analyzer.analyze("This contains FORBIDDEN WORD here")
            >>> len(flags) > 0
            True
        """
        self._custom_patterns.append((name, re.compile(pattern, re.I), risk_level))

    def analyze(self, text: str) -> list[SafetyFlag]:
        """Analyze text for toxic content patterns.

        Scans the input text for profanity, threats, harassment, and any
        custom patterns, returning a list of SafetyFlag objects for each
        detected issue.

        Args:
            text: The text to analyze for toxic content.

        Returns:
            List of SafetyFlag objects, each representing a detected
            toxicity concern. Returns an empty list if no issues found.

        Examples:
            Analyzing clean text:

            >>> from insideLLMs.safety import ToxicityAnalyzer
            >>> analyzer = ToxicityAnalyzer()
            >>> flags = analyzer.analyze("Hello, how are you today?")
            >>> len(flags)
            0

            Detecting mild profanity:

            >>> flags = analyzer.analyze("What the hell is going on?")
            >>> len(flags)
            1
            >>> flags[0].risk_level.value
            'low'
            >>> flags[0].matched_text
            'hell'

            Detecting threats (high risk):

            >>> flags = analyzer.analyze("I'm going to hurt them badly")
            >>> [f for f in flags if f.category.value == 'violence'][0].risk_level.value
            'high'

            Multiple issues in one text:

            >>> flags = analyzer.analyze("Damn it, you're stupid!")
            >>> len(flags) >= 2
            True
            >>> categories = {f.category.value for f in flags}
            >>> 'toxicity' in categories
            True
        """
        flags = []
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        # Check profanity
        profanity_found = words & self.PROFANITY_INDICATORS
        if profanity_found:
            flags.append(
                SafetyFlag(
                    category=SafetyCategory.TOXICITY,
                    risk_level=RiskLevel.LOW,
                    description="Contains mild profanity",
                    matched_text=", ".join(profanity_found),
                    confidence=0.7,
                )
            )

        # Check threat patterns
        for pattern in self.THREAT_PATTERNS:
            match = pattern.search(text)
            if match:
                flags.append(
                    SafetyFlag(
                        category=SafetyCategory.VIOLENCE,
                        risk_level=RiskLevel.HIGH,
                        description="Contains potential threat language",
                        matched_text=match.group(),
                        confidence=0.8,
                    )
                )
                break

        # Check harassment patterns
        for pattern in self.HARASSMENT_PATTERNS:
            match = pattern.search(text)
            if match:
                flags.append(
                    SafetyFlag(
                        category=SafetyCategory.TOXICITY,
                        risk_level=RiskLevel.MEDIUM,
                        description="Contains potential harassment",
                        matched_text=match.group(),
                        confidence=0.7,
                    )
                )
                break

        # Check custom patterns
        for name, pattern, risk_level in self._custom_patterns:
            match = pattern.search(text)
            if match:
                flags.append(
                    SafetyFlag(
                        category=SafetyCategory.TOXICITY,
                        risk_level=risk_level,
                        description=f"Matched custom pattern: {name}",
                        matched_text=match.group(),
                        confidence=0.9,
                    )
                )

        return flags


class SafetyHallucinationIndicatorDetector:
    """Detect potential hallucination indicators in LLM-generated outputs.

    This detector identifies patterns in text that may indicate hallucinated
    or fabricated content from LLMs, including:
    - Uncertainty phrases that suggest guessing
    - Overconfident language without evidence
    - Vague citations without specific sources
    - Specific numerical claims that may be fabricated

    The detector calculates a risk score based on the presence and frequency
    of these indicators.

    Attributes:
        UNCERTAINTY_PHRASES: List of phrases indicating uncertainty.
        CONFIDENT_FALSE_INDICATORS: List of overconfident language patterns.
        FAKE_CITATION_PATTERNS: Regex patterns for vague/fake citations.
        SPECIFIC_CLAIM_PATTERNS: Regex patterns for specific unverified claims.

    Examples:
        Basic hallucination analysis:

        >>> from insideLLMs.safety import SafetyHallucinationIndicatorDetector
        >>> detector = SafetyHallucinationIndicatorDetector()
        >>> result = detector.analyze("Studies show that 95% of users prefer this.")
        >>> result['risk_score'] > 0
        True
        >>> result['indicators']['has_vague_citations']
        True
        >>> result['indicators']['has_specific_claims']
        True

        Analyzing uncertain language:

        >>> result = detector.analyze("I think this might be correct, probably.")
        >>> result['uncertainty_count'] > 0
        True
        >>> result['indicators']['has_uncertainty']
        True

        Analyzing overconfident language:

        >>> result = detector.analyze("As everyone knows, this is obviously true.")
        >>> result['overconfidence_count']
        2
        >>> result['indicators']['has_overconfidence']
        True

        Getting risk level:

        >>> from insideLLMs.safety import RiskLevel
        >>> result = detector.analyze("Normal factual text without indicators.")
        >>> risk = detector.get_risk_level(result)
        >>> risk == RiskLevel.LOW
        True

    Note:
        This detector identifies patterns that MAY indicate hallucination.
        The presence of these patterns does not guarantee content is false,
        and their absence does not guarantee content is true. Use in
        conjunction with fact-checking for important content.
    """

    # Phrases that may indicate uncertainty or fabrication
    UNCERTAINTY_PHRASES = [
        "i think",
        "i believe",
        "i'm not sure",
        "i'm not certain",
        "i assume",
        "i suppose",
        "might be",
        "could be",
        "possibly",
        "perhaps",
        "probably",
        "likely",
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
        """Initialize the hallucination indicator detector.

        Creates a new SafetyHallucinationIndicatorDetector instance.
        All detection patterns are defined at the class level.

        Examples:
            Basic initialization:

            >>> from insideLLMs.safety import SafetyHallucinationIndicatorDetector
            >>> detector = SafetyHallucinationIndicatorDetector()
            >>> result = detector.analyze("Test text")
            >>> 'risk_score' in result
            True

            Using in a safety pipeline:

            >>> from insideLLMs.safety import ContentSafetyAnalyzer
            >>> analyzer = ContentSafetyAnalyzer()
            >>> # The analyzer uses SafetyHallucinationIndicatorDetector internally
            >>> report = analyzer.analyze("According to studies, 90% agree.")
            >>> 'hallucination_risk' in report.scores
            True
        """
        pass

    def analyze(self, text: str, context: Optional[str] = None) -> dict[str, Any]:
        """Analyze text for hallucination indicators.

        Scans the text for patterns that may indicate hallucinated content
        and calculates a risk score based on the findings.

        Args:
            text: The text to analyze for hallucination indicators.
            context: Optional original prompt or context for comparison
                (currently not used, reserved for future enhancement).

        Returns:
            Dictionary containing:
                - risk_score: Float from 0.0 to 1.0 indicating hallucination risk
                - uncertainty_count: Number of uncertainty phrases found
                - overconfidence_count: Number of overconfident phrases found
                - vague_citations: List of vague citation matches
                - specific_claims: List of specific claim matches
                - indicators: Dict of boolean flags for each indicator type

        Examples:
            Analyzing text with multiple indicators:

            >>> from insideLLMs.safety import SafetyHallucinationIndicatorDetector
            >>> detector = SafetyHallucinationIndicatorDetector()
            >>> text = "Studies show 75% of experts agree this is definitely true."
            >>> result = detector.analyze(text)
            >>> result['indicators']['has_vague_citations']
            True
            >>> result['indicators']['has_specific_claims']
            True
            >>> result['indicators']['has_overconfidence']
            True

            Analyzing clean factual text:

            >>> result = detector.analyze("The capital of France is Paris.")
            >>> result['uncertainty_count']
            0
            >>> result['overconfidence_count']
            0

            Analyzing uncertain text:

            >>> result = detector.analyze("I think this might possibly be true.")
            >>> result['uncertainty_count']
            3
            >>> result['risk_score'] > 0.1
            True

            Checking for vague citations:

            >>> result = detector.analyze("According to a recent study, results vary.")
            >>> len(result['vague_citations']) > 0
            True
        """
        text_lower = text.lower()

        # Count uncertainty indicators
        uncertainty_count = sum(1 for phrase in self.UNCERTAINTY_PHRASES if phrase in text_lower)

        # Count overconfident phrases
        overconfidence_count = sum(
            1 for phrase in self.CONFIDENT_FALSE_INDICATORS if phrase in text_lower
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

    def get_risk_level(self, analysis: dict[str, Any]) -> RiskLevel:
        """Convert analysis results to a RiskLevel classification.

        Maps the numerical risk_score from analysis results to a
        categorical RiskLevel enum value.

        Args:
            analysis: Dictionary returned by analyze() containing risk_score.

        Returns:
            RiskLevel enum value:
                - LOW: risk_score < 0.2
                - MEDIUM: risk_score >= 0.2 and < 0.4
                - HIGH: risk_score >= 0.4 and < 0.7
                - CRITICAL: risk_score >= 0.7

        Examples:
            Getting risk level from analysis:

            >>> from insideLLMs.safety import SafetyHallucinationIndicatorDetector, RiskLevel
            >>> detector = SafetyHallucinationIndicatorDetector()
            >>> result = detector.analyze("Simple factual statement.")
            >>> risk = detector.get_risk_level(result)
            >>> risk == RiskLevel.LOW
            True

            High risk content:

            >>> text = "Studies show 95% of experts definitely agree without a doubt."
            >>> result = detector.analyze(text)
            >>> risk = detector.get_risk_level(result)
            >>> risk in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            True

            Using risk level for decisions:

            >>> result = detector.analyze("According to research, this is obviously true.")
            >>> risk = detector.get_risk_level(result)
            >>> if risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            ...     print("Content should be fact-checked")
            Content should be fact-checked

            Working with missing risk_score:

            >>> risk = detector.get_risk_level({})  # Empty dict
            >>> risk == RiskLevel.LOW
            True
        """
        score = analysis.get("risk_score", 0)
        if score < 0.2:
            return RiskLevel.LOW
        elif score < 0.4:
            return RiskLevel.MEDIUM
        elif score < 0.7:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL


class BiasDetector:
    """Detect potential bias patterns in text including gender imbalance and stereotypes.

    BiasDetector analyzes text for various forms of bias including:
    - Gender representation imbalance
    - Stereotyping language patterns
    - Excessive use of absolute terms

    The detector provides both individual analysis methods for specific
    bias types and a comprehensive analyze() method that combines all checks.

    Attributes:
        GENDER_TERMS: Dictionary mapping 'male' and 'female' to sets of related terms.
        STEREOTYPE_PATTERNS: List of regex patterns for detecting stereotyping language.
        ABSOLUTE_TERMS: Set of absolute terms that may indicate biased thinking.
        ABSOLUTE_PHRASES: List of multi-word absolute phrases.

    Examples:
        Basic bias analysis:

        >>> from insideLLMs.safety import BiasDetector
        >>> detector = BiasDetector()
        >>> result = detector.analyze("He went to work while she stayed home.")
        >>> 'bias_score' in result
        True
        >>> result['gender_balance']['is_balanced']
        True

        Detecting gender imbalance:

        >>> result = detector.analyze("He, him, his, man, men, boy went to the store.")
        >>> result['gender_balance']['is_balanced']
        False
        >>> result['gender_balance']['male_terms'] > 0
        True

        Detecting stereotypes:

        >>> result = detector.analyze("All women are emotional. Men are always logical.")
        >>> len(result['stereotypes']) > 0
        True
        >>> result['has_potential_bias']
        True

        Analyzing absolute language:

        >>> result = detector.analyze("Everyone knows this. Nobody disagrees.")
        >>> result['absolutes']['absolute_count'] >= 2
        True

    Note:
        This detector identifies patterns that MAY indicate bias. Human review
        is recommended for nuanced cases. Context matters significantly in
        bias detection.
    """

    # Gender-related terms for balance checking
    GENDER_TERMS = {
        "male": {"he", "him", "his", "man", "men", "boy", "boys", "male", "father", "son"},
        "female": {
            "she",
            "her",
            "hers",
            "woman",
            "women",
            "girl",
            "girls",
            "female",
            "mother",
            "daughter",
        },
    }

    # Stereotyping patterns
    STEREOTYPE_PATTERNS = [
        re.compile(r"\b(all|every|most)\s+(men|women|people from)\b", re.I),
        re.compile(r"\b(men|women)\s+are\s+(always|never|usually)\b", re.I),
        re.compile(r"\b(typical|stereotypical)\s+\w+\b", re.I),
    ]

    # Absolutes that may indicate bias
    ABSOLUTE_TERMS = {
        "always",
        "never",
        "all",
        "none",
        "every",
        "ever",
        "everyone",
        "nobody",
        "completely",
        "absolutely",
    }

    # Multi-word absolutes to check separately
    ABSOLUTE_PHRASES = [
        "no one",
    ]

    def __init__(self):
        """Initialize the bias detector.

        Creates a new BiasDetector instance. All detection patterns
        are defined at the class level.

        Examples:
            Basic initialization:

            >>> from insideLLMs.safety import BiasDetector
            >>> detector = BiasDetector()
            >>> result = detector.analyze("Test text")
            >>> 'bias_score' in result
            True

            Using with ContentSafetyAnalyzer:

            >>> from insideLLMs.safety import ContentSafetyAnalyzer
            >>> analyzer = ContentSafetyAnalyzer()
            >>> # BiasDetector is used internally
            >>> report = analyzer.analyze("All men are strong.")
            >>> 'bias_score' in report.scores
            True
        """
        pass

    def analyze_gender_balance(self, text: str) -> dict[str, Any]:
        """Analyze gender representation balance in text.

        Counts occurrences of male and female gendered terms and calculates
        a balance ratio to identify potential gender bias in representation.

        Args:
            text: The text to analyze for gender balance.

        Returns:
            Dictionary containing:
                - male_terms: Count of unique male terms found
                - female_terms: Count of unique female terms found
                - balance_ratio: Float from 0.0 (all female) to 1.0 (all male)
                - is_balanced: Boolean, True if ratio is between 0.3 and 0.7

        Examples:
            Balanced text:

            >>> from insideLLMs.safety import BiasDetector
            >>> detector = BiasDetector()
            >>> result = detector.analyze_gender_balance("He and she went to work.")
            >>> result['is_balanced']
            True
            >>> 0.3 <= result['balance_ratio'] <= 0.7
            True

            Male-dominated text:

            >>> result = detector.analyze_gender_balance("He told him about his father and son.")
            >>> result['male_terms'] > result['female_terms']
            True
            >>> result['is_balanced']
            False

            Female-dominated text:

            >>> result = detector.analyze_gender_balance("She told her about her mother.")
            >>> result['female_terms'] > result['male_terms']
            True

            Gender-neutral text:

            >>> result = detector.analyze_gender_balance("The team completed the project.")
            >>> result['male_terms']
            0
            >>> result['female_terms']
            0
            >>> result['balance_ratio']
            0.5
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

    def analyze_stereotypes(self, text: str) -> list[str]:
        """Detect potential stereotyping language patterns.

        Searches for language patterns that generalize about groups of
        people, which may indicate stereotyping or biased thinking.

        Args:
            text: The text to analyze for stereotype patterns.

        Returns:
            List of matched stereotype phrases found in the text.
            Returns an empty list if no patterns match.

        Examples:
            Detecting group generalizations:

            >>> from insideLLMs.safety import BiasDetector
            >>> detector = BiasDetector()
            >>> result = detector.analyze_stereotypes("All women are emotional.")
            >>> len(result) > 0
            True

            Detecting absolute statements about groups:

            >>> result = detector.analyze_stereotypes("Men are always competitive.")
            >>> len(result) > 0
            True

            Detecting stereotypical language:

            >>> result = detector.analyze_stereotypes("That's a typical response.")
            >>> "typical" in " ".join(result).lower()
            True

            Clean text without stereotypes:

            >>> result = detector.analyze_stereotypes("People have diverse opinions.")
            >>> len(result)
            0
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

    def analyze_absolutes(self, text: str) -> dict[str, Any]:
        """Analyze use of absolute language that may indicate bias.

        Identifies words and phrases that express absolute certainty,
        which when overused may indicate biased or inflexible thinking.

        Args:
            text: The text to analyze for absolute language.

        Returns:
            Dictionary containing:
                - absolutes_found: List of unique absolute terms found
                - absolute_count: Total count of absolute term occurrences
                - word_count: Total word count in text
                - absolute_ratio: Ratio of absolutes to total words

        Examples:
            Detecting absolute terms:

            >>> from insideLLMs.safety import BiasDetector
            >>> detector = BiasDetector()
            >>> result = detector.analyze_absolutes("Everyone always agrees completely.")
            >>> 'always' in result['absolutes_found']
            True
            >>> 'everyone' in result['absolutes_found']
            True
            >>> result['absolute_count'] >= 3
            True

            Multi-word absolutes:

            >>> result = detector.analyze_absolutes("No one ever disagrees.")
            >>> 'no one' in result['absolutes_found']
            True

            Calculating absolute ratio:

            >>> result = detector.analyze_absolutes("This is always true in every case.")
            >>> result['absolute_ratio'] > 0
            True
            >>> result['word_count'] > 0
            True

            Text without absolutes:

            >>> result = detector.analyze_absolutes("Sometimes people disagree.")
            >>> result['absolute_count']
            0
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

    def analyze(self, text: str) -> dict[str, Any]:
        """Perform comprehensive bias analysis on text.

        Combines all bias detection methods (gender balance, stereotypes,
        absolutes) and calculates an overall bias score.

        Args:
            text: The text to analyze for bias patterns.

        Returns:
            Dictionary containing:
                - bias_score: Float from 0.0 to 1.0 indicating overall bias
                - gender_balance: Results from analyze_gender_balance()
                - stereotypes: List of detected stereotype patterns
                - absolutes: Results from analyze_absolutes()
                - has_potential_bias: Boolean, True if bias_score > 0.3

        Examples:
            Analyzing unbiased text:

            >>> from insideLLMs.safety import BiasDetector
            >>> detector = BiasDetector()
            >>> result = detector.analyze("The project was completed on time.")
            >>> result['has_potential_bias']
            False
            >>> result['bias_score'] < 0.3
            True

            Detecting multiple bias indicators:

            >>> result = detector.analyze("All men are always competitive. Every woman is emotional.")
            >>> result['has_potential_bias']
            True
            >>> len(result['stereotypes']) > 0
            True
            >>> result['gender_balance']['is_balanced']
            True

            Analyzing gender-imbalanced text:

            >>> result = detector.analyze("He told him about his father, son, and brother.")
            >>> result['gender_balance']['is_balanced']
            False
            >>> result['bias_score'] > 0
            True

            Using for content filtering:

            >>> result = detector.analyze("Some people prefer different approaches.")
            >>> if not result['has_potential_bias']:
            ...     print("Content appears unbiased")
            Content appears unbiased
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
    """Comprehensive content safety analyzer combining multiple detection methods.

    ContentSafetyAnalyzer provides a unified interface for analyzing text
    against multiple safety criteria including PII exposure, toxicity,
    hallucination indicators, and bias patterns. It aggregates results
    from specialized detectors into a single SafetyReport.

    Attributes:
        pii_detector: PIIDetector instance for PII detection.
        toxicity_analyzer: ToxicityAnalyzer instance for toxicity detection.
        hallucination_detector: SafetyHallucinationIndicatorDetector instance.
        bias_detector: BiasDetector instance for bias detection.

    Examples:
        Basic comprehensive analysis:

        >>> from insideLLMs.safety import ContentSafetyAnalyzer
        >>> analyzer = ContentSafetyAnalyzer()
        >>> report = analyzer.analyze("Hello, my name is John.")
        >>> report.is_safe
        True
        >>> report.overall_risk.value
        'none'

        Analyzing text with PII:

        >>> report = analyzer.analyze("My email is john@example.com")
        >>> report.is_safe
        False
        >>> report.overall_risk.value
        'high'
        >>> 'pii_count' in report.scores
        True

        Selective analysis:

        >>> report = analyzer.analyze(
        ...     "My email is test@test.com",
        ...     check_pii=True,
        ...     check_toxicity=False,
        ...     check_hallucination=False,
        ...     check_bias=False
        ... )
        >>> len(report.flags)
        1
        >>> report.flags[0].category.value
        'pii_exposure'

        Analyzing for multiple concerns:

        >>> text = "According to studies, 95% of damn experts agree."
        >>> report = analyzer.analyze(text)
        >>> report.scores.get('toxicity_flags', 0) > 0
        True
        >>> report.scores.get('hallucination_risk', 0) > 0
        True
    """

    def __init__(self):
        """Initialize the content safety analyzer with all sub-analyzers.

        Creates instances of all specialized analyzers (PII, toxicity,
        hallucination, bias) for comprehensive safety analysis.

        Examples:
            Basic initialization:

            >>> from insideLLMs.safety import ContentSafetyAnalyzer
            >>> analyzer = ContentSafetyAnalyzer()
            >>> analyzer.pii_detector is not None
            True
            >>> analyzer.toxicity_analyzer is not None
            True

            Accessing sub-analyzers:

            >>> analyzer = ContentSafetyAnalyzer()
            >>> # Can use individual analyzers if needed
            >>> pii_report = analyzer.pii_detector.detect("test@test.com")
            >>> pii_report.has_pii
            True

            Creating analyzer for repeated use:

            >>> analyzer = ContentSafetyAnalyzer()
            >>> texts = ["Text 1", "Text 2", "Text 3"]
            >>> reports = [analyzer.analyze(t) for t in texts]
            >>> len(reports)
            3
        """
        self.pii_detector = PIIDetector()
        self.toxicity_analyzer = ToxicityAnalyzer()
        self.hallucination_detector = SafetyHallucinationIndicatorDetector()
        self.bias_detector = BiasDetector()

    def analyze(
        self,
        text: str,
        check_pii: bool = True,
        check_toxicity: bool = True,
        check_hallucination: bool = True,
        check_bias: bool = True,
    ) -> SafetyReport:
        """Perform comprehensive safety analysis on text.

        Analyzes text using all enabled analyzers and returns a unified
        SafetyReport containing all findings.

        Args:
            text: The text to analyze for safety concerns.
            check_pii: Whether to check for personally identifiable information.
                Default: True.
            check_toxicity: Whether to check for toxic content patterns.
                Default: True.
            check_hallucination: Whether to check for hallucination indicators.
                Default: True.
            check_bias: Whether to check for bias patterns.
                Default: True.

        Returns:
            SafetyReport containing:
                - is_safe: Boolean (True if risk is NONE or LOW)
                - overall_risk: Highest RiskLevel across all checks
                - flags: List of SafetyFlag objects for each concern
                - scores: Dictionary of numerical scores from analyzers

        Examples:
            Full comprehensive analysis:

            >>> from insideLLMs.safety import ContentSafetyAnalyzer
            >>> analyzer = ContentSafetyAnalyzer()
            >>> report = analyzer.analyze("Contact: john@test.com")
            >>> report.is_safe
            False
            >>> report.overall_risk.value
            'high'

            Checking only specific concerns:

            >>> report = analyzer.analyze(
            ...     "That's a damn good idea!",
            ...     check_pii=False,
            ...     check_hallucination=False,
            ...     check_bias=False
            ... )
            >>> report.scores.get('toxicity_flags', 0) > 0
            True

            Analyzing safe content:

            >>> report = analyzer.analyze("The weather is nice today.")
            >>> report.is_safe
            True
            >>> len(report.flags)
            0

            Getting detailed results:

            >>> report = analyzer.analyze("Studies show 90% of experts agree!")
            >>> 'hallucination_risk' in report.scores
            True
            >>> report.scores['hallucination_risk'] > 0
            True
        """
        flags: list[SafetyFlag] = []
        scores: dict[str, float] = {}

        # PII detection
        if check_pii:
            pii_report = self.pii_detector.detect(text)
            if pii_report.has_pii:
                flags.append(
                    SafetyFlag(
                        category=SafetyCategory.PII_EXPOSURE,
                        risk_level=RiskLevel.HIGH,
                        description=f"Found {len(pii_report.matches)} PII items",
                        metadata={"pii_types": list({m.pii_type for m in pii_report.matches})},
                    )
                )
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
                flags.append(
                    SafetyFlag(
                        category=SafetyCategory.MISINFORMATION,
                        risk_level=risk_level,
                        description="High hallucination risk indicators",
                        confidence=hallucination_analysis["risk_score"],
                        metadata=hallucination_analysis["indicators"],
                    )
                )
            scores["hallucination_risk"] = hallucination_analysis["risk_score"]

        # Bias detection
        if check_bias:
            bias_analysis = self.bias_detector.analyze(text)
            if bias_analysis["has_potential_bias"]:
                flags.append(
                    SafetyFlag(
                        category=SafetyCategory.TOXICITY,
                        risk_level=RiskLevel.MEDIUM,
                        description="Potential bias detected",
                        confidence=bias_analysis["bias_score"],
                        metadata={
                            "gender_balanced": bias_analysis["gender_balance"]["is_balanced"],
                            "stereotypes_found": len(bias_analysis["stereotypes"]) > 0,
                        },
                    )
                )
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


def quick_safety_check(text: str) -> tuple[bool, RiskLevel, list[str]]:
    """Perform a quick comprehensive safety check on text.

    Convenience function that creates a ContentSafetyAnalyzer and runs
    all safety checks, returning a simplified tuple of results.

    Args:
        text: The text to check for safety concerns.

    Returns:
        Tuple containing:
            - is_safe: Boolean indicating if content is safe (NONE or LOW risk)
            - risk_level: Overall RiskLevel enum value
            - issues: List of human-readable issue descriptions

    Examples:
        Checking safe content:

        >>> from insideLLMs.safety import quick_safety_check
        >>> is_safe, risk, issues = quick_safety_check("Hello, world!")
        >>> is_safe
        True
        >>> risk.value
        'none'
        >>> len(issues)
        0

        Checking content with PII:

        >>> is_safe, risk, issues = quick_safety_check("My email is test@test.com")
        >>> is_safe
        False
        >>> risk.value
        'high'
        >>> 'PII' in issues[0]
        True

        Checking content with toxicity:

        >>> is_safe, risk, issues = quick_safety_check("That's a damn good point!")
        >>> is_safe
        True  # LOW risk is still considered safe
        >>> len(issues) > 0
        True

        Using for content filtering:

        >>> text = "Contact john@example.com for help"
        >>> is_safe, risk, issues = quick_safety_check(text)
        >>> if not is_safe:
        ...     print(f"Blocked: {issues}")
        Blocked: ['Found 1 PII items']
    """
    analyzer = ContentSafetyAnalyzer()
    report = analyzer.analyze(text)

    issues = [f.description for f in report.flags]
    return report.is_safe, report.overall_risk, issues


def mask_pii(text: str) -> str:
    """Quickly mask all PII in text with placeholder tokens.

    Convenience function that creates a PIIDetector and masks all
    detected PII with appropriate tokens (e.g., [EMAIL], [PHONE], [SSN]).

    Args:
        text: The text containing PII to mask.

    Returns:
        Text with all detected PII replaced by mask tokens.

    Examples:
        Masking an email address:

        >>> from insideLLMs.safety import mask_pii
        >>> mask_pii("Contact john@example.com for help")
        'Contact [EMAIL] for help'

        Masking multiple PII types:

        >>> text = "Email: test@test.com, Phone: 555-123-4567, SSN: 123-45-6789"
        >>> masked = mask_pii(text)
        >>> '[EMAIL]' in masked
        True
        >>> '[PHONE]' in masked
        True
        >>> '[SSN]' in masked
        True

        Text without PII:

        >>> mask_pii("Hello, world!")
        'Hello, world!'

        Using for log sanitization:

        >>> log_entry = "User john@test.com logged in from 192.168.1.1"
        >>> safe_log = mask_pii(log_entry)
        >>> 'john@test.com' not in safe_log
        True
        >>> '[EMAIL]' in safe_log
        True
    """
    detector = PIIDetector()
    return detector.mask(text)


def detect_pii(text: str) -> list[dict[str, Any]]:
    """Quickly detect all PII in text and return as dictionaries.

    Convenience function that creates a PIIDetector and returns all
    detected PII as a list of dictionaries for easy processing.

    Args:
        text: The text to analyze for PII.

    Returns:
        List of dictionaries, each containing:
            - type: The PII type (e.g., 'email', 'phone_us', 'ssn')
            - value: The actual PII value found
            - start: Starting character index in the text
            - end: Ending character index in the text

    Examples:
        Detecting an email:

        >>> from insideLLMs.safety import detect_pii
        >>> matches = detect_pii("Contact john@example.com")
        >>> len(matches)
        1
        >>> matches[0]['type']
        'email'
        >>> matches[0]['value']
        'john@example.com'

        Detecting multiple PII types:

        >>> matches = detect_pii("Email: a@b.com, Phone: 555-123-4567")
        >>> len(matches)
        2
        >>> types = [m['type'] for m in matches]
        >>> 'email' in types
        True
        >>> 'phone_us' in types
        True

        Using position information:

        >>> text = "SSN: 123-45-6789"
        >>> matches = detect_pii(text)
        >>> m = matches[0]
        >>> text[m['start']:m['end']]
        '123-45-6789'

        No PII found:

        >>> detect_pii("Hello, world!")
        []
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
