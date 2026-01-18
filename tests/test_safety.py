"""Tests for safety and content analysis utilities."""

from insideLLMs.safety import (
    BiasDetector,
    ContentSafetyAnalyzer,
    HallucinationDetector,
    PIIDetector,
    RiskLevel,
    SafetyCategory,
    SafetyFlag,
    SafetyReport,
    ToxicityAnalyzer,
    detect_pii,
    mask_pii,
    quick_safety_check,
)


class TestSafetyCategory:
    """Tests for SafetyCategory enum."""

    def test_all_categories_exist(self):
        """Test that all categories are defined."""
        assert SafetyCategory.SAFE.value == "safe"
        assert SafetyCategory.TOXICITY.value == "toxicity"
        assert SafetyCategory.PII_EXPOSURE.value == "pii_exposure"
        assert SafetyCategory.MISINFORMATION.value == "misinformation"


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_all_levels_exist(self):
        """Test that all risk levels are defined."""
        assert RiskLevel.NONE.value == "none"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestSafetyFlag:
    """Tests for SafetyFlag."""

    def test_basic_creation(self):
        """Test basic flag creation."""
        flag = SafetyFlag(
            category=SafetyCategory.TOXICITY,
            risk_level=RiskLevel.MEDIUM,
            description="Test flag",
        )
        assert flag.category == SafetyCategory.TOXICITY
        assert flag.risk_level == RiskLevel.MEDIUM
        assert flag.confidence == 1.0


class TestSafetyReport:
    """Tests for SafetyReport."""

    def test_basic_creation(self):
        """Test basic report creation."""
        report = SafetyReport(
            text="Test text",
            is_safe=True,
            overall_risk=RiskLevel.NONE,
        )
        assert report.is_safe
        assert len(report.flags) == 0

    def test_get_flags_by_category(self):
        """Test filtering flags by category."""
        flags = [
            SafetyFlag(SafetyCategory.TOXICITY, RiskLevel.LOW, "T1"),
            SafetyFlag(SafetyCategory.PII_EXPOSURE, RiskLevel.HIGH, "P1"),
            SafetyFlag(SafetyCategory.TOXICITY, RiskLevel.MEDIUM, "T2"),
        ]
        report = SafetyReport(
            text="Test",
            is_safe=False,
            overall_risk=RiskLevel.HIGH,
            flags=flags,
        )

        toxicity_flags = report.get_flags_by_category(SafetyCategory.TOXICITY)
        assert len(toxicity_flags) == 2

    def test_get_highest_risk_flag(self):
        """Test getting highest risk flag."""
        flags = [
            SafetyFlag(SafetyCategory.TOXICITY, RiskLevel.LOW, "Low"),
            SafetyFlag(SafetyCategory.TOXICITY, RiskLevel.HIGH, "High"),
            SafetyFlag(SafetyCategory.TOXICITY, RiskLevel.MEDIUM, "Medium"),
        ]
        report = SafetyReport(
            text="Test",
            is_safe=False,
            overall_risk=RiskLevel.HIGH,
            flags=flags,
        )

        highest = report.get_highest_risk_flag()
        assert highest.risk_level == RiskLevel.HIGH

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = SafetyReport(
            text="Test",
            is_safe=True,
            overall_risk=RiskLevel.LOW,
            scores={"test": 0.5},
        )

        d = report.to_dict()
        assert d["is_safe"] is True
        assert d["overall_risk"] == "low"


class TestPIIDetector:
    """Tests for PIIDetector."""

    def test_detect_email(self):
        """Test email detection."""
        detector = PIIDetector()
        report = detector.detect("Contact me at john.doe@example.com please")

        assert report.has_pii
        assert len(report.matches) == 1
        assert report.matches[0].pii_type == "email"
        assert report.matches[0].value == "john.doe@example.com"

    def test_detect_phone_us(self):
        """Test US phone number detection."""
        detector = PIIDetector()
        report = detector.detect("Call me at 555-123-4567")

        assert report.has_pii
        assert any(m.pii_type == "phone_us" for m in report.matches)

    def test_detect_ssn(self):
        """Test SSN detection."""
        detector = PIIDetector()
        report = detector.detect("My SSN is 123-45-6789")

        assert report.has_pii
        assert any(m.pii_type == "ssn" for m in report.matches)

    def test_detect_credit_card(self):
        """Test credit card detection."""
        detector = PIIDetector()
        report = detector.detect("Card: 4111-1111-1111-1111")

        assert report.has_pii
        assert any(m.pii_type == "credit_card" for m in report.matches)

    def test_detect_multiple(self):
        """Test detecting multiple PII types."""
        detector = PIIDetector()
        text = "Email: test@example.com, Phone: 555-123-4567"
        report = detector.detect(text)

        assert len(report.matches) >= 2

    def test_no_pii(self):
        """Test text without PII."""
        detector = PIIDetector()
        report = detector.detect("Hello, this is a normal message.")

        assert not report.has_pii
        assert len(report.matches) == 0

    def test_mask(self):
        """Test PII masking."""
        detector = PIIDetector()
        text = "Email me at test@example.com"
        masked = detector.mask(text)

        assert "[EMAIL]" in masked
        assert "test@example.com" not in masked

    def test_mask_specific_types(self):
        """Test masking specific PII types."""
        detector = PIIDetector()
        text = "Email: test@example.com, Phone: 555-123-4567"
        masked = detector.mask(text, types=["email"])

        assert "[EMAIL]" in masked
        assert "555-123-4567" in masked  # Phone not masked

    def test_detect_and_mask(self):
        """Test combined detection and masking."""
        detector = PIIDetector()
        text = "Contact: test@example.com"
        report = detector.detect_and_mask(text)

        assert report.has_pii
        assert report.masked_text is not None
        assert "[EMAIL]" in report.masked_text

    def test_add_custom_pattern(self):
        """Test adding custom pattern."""
        detector = PIIDetector()
        detector.add_pattern("custom_id", r"ID-\d{6}", "[CUSTOM_ID]")

        report = detector.detect("My ID-123456 is here")
        assert report.has_pii

        masked = detector.mask("My ID-123456 is here")
        assert "[CUSTOM_ID]" in masked

    def test_get_by_type(self):
        """Test filtering matches by type."""
        detector = PIIDetector()
        text = "Email: a@b.com and c@d.com"
        report = detector.detect(text)

        emails = report.get_by_type("email")
        assert len(emails) == 2


class TestToxicityAnalyzer:
    """Tests for ToxicityAnalyzer."""

    def test_clean_text(self):
        """Test clean text has no flags."""
        analyzer = ToxicityAnalyzer()
        flags = analyzer.analyze("This is a friendly message about programming.")

        assert len(flags) == 0

    def test_mild_profanity(self):
        """Test mild profanity detection."""
        analyzer = ToxicityAnalyzer()
        flags = analyzer.analyze("This damn code won't work.")

        assert len(flags) > 0
        assert any(f.category == SafetyCategory.TOXICITY for f in flags)
        assert any(f.risk_level == RiskLevel.LOW for f in flags)

    def test_threat_detection(self):
        """Test threat language detection."""
        analyzer = ToxicityAnalyzer()
        flags = analyzer.analyze("I'm going to hurt you")

        assert len(flags) > 0
        assert any(f.category == SafetyCategory.VIOLENCE for f in flags)
        assert any(f.risk_level == RiskLevel.HIGH for f in flags)

    def test_harassment_detection(self):
        """Test harassment detection."""
        analyzer = ToxicityAnalyzer()
        flags = analyzer.analyze("You're an idiot for thinking that")

        assert len(flags) > 0

    def test_custom_pattern(self):
        """Test adding custom pattern."""
        analyzer = ToxicityAnalyzer()
        analyzer.add_pattern("test_pattern", r"bad phrase", RiskLevel.HIGH)

        flags = analyzer.analyze("This has a bad phrase in it")
        assert len(flags) > 0


class TestHallucinationDetector:
    """Tests for HallucinationDetector."""

    def test_clean_text(self):
        """Test text with low hallucination risk."""
        detector = HallucinationDetector()
        result = detector.analyze("The sky is blue. Water is wet.")

        assert result["risk_score"] < 0.3

    def test_uncertainty_detection(self):
        """Test uncertainty phrase detection."""
        detector = HallucinationDetector()
        result = detector.analyze("I think this might be correct, perhaps.")

        assert result["uncertainty_count"] >= 2
        assert result["indicators"]["has_uncertainty"]

    def test_overconfidence_detection(self):
        """Test overconfidence detection."""
        detector = HallucinationDetector()
        result = detector.analyze("Obviously this is true. As everyone knows, it's certain.")

        assert result["overconfidence_count"] >= 2
        assert result["indicators"]["has_overconfidence"]

    def test_vague_citation_detection(self):
        """Test vague citation detection."""
        detector = HallucinationDetector()
        result = detector.analyze("According to a recent study, experts say this is true.")

        assert len(result["vague_citations"]) >= 1
        assert result["indicators"]["has_vague_citations"]

    def test_specific_claims(self):
        """Test specific claim detection."""
        detector = HallucinationDetector()
        result = detector.analyze("In 2023, 73% of users paid $500 for this service.")

        assert len(result["specific_claims"]) >= 2
        assert result["indicators"]["has_specific_claims"]

    def test_get_risk_level(self):
        """Test risk level determination."""
        detector = HallucinationDetector()

        low_risk = {"risk_score": 0.1}
        assert detector.get_risk_level(low_risk) == RiskLevel.LOW

        high_risk = {"risk_score": 0.6}
        assert detector.get_risk_level(high_risk) == RiskLevel.HIGH


class TestBiasDetector:
    """Tests for BiasDetector."""

    def test_gender_balance_neutral(self):
        """Test gender-neutral text."""
        detector = BiasDetector()
        result = detector.analyze_gender_balance("The person went to the store.")

        assert result["is_balanced"]

    def test_gender_balance_male_heavy(self):
        """Test male-heavy text."""
        detector = BiasDetector()
        result = detector.analyze_gender_balance("He told him that his father and son were there.")

        assert result["male_terms"] > result["female_terms"]

    def test_gender_balance_calculation(self):
        """Test balance ratio calculation."""
        detector = BiasDetector()
        result = detector.analyze_gender_balance("He and she went together.")

        # Should be roughly balanced
        assert 0.4 <= result["balance_ratio"] <= 0.6

    def test_stereotype_detection(self):
        """Test stereotype pattern detection."""
        detector = BiasDetector()
        result = detector.analyze_stereotypes("All men are always like that.")

        assert len(result) > 0

    def test_absolute_detection(self):
        """Test absolute term detection."""
        detector = BiasDetector()
        result = detector.analyze_absolutes("Everyone always does this. No one ever disagrees.")

        assert result["absolute_count"] >= 3
        assert "always" in result["absolutes_found"]

    def test_comprehensive_analysis(self):
        """Test full bias analysis."""
        detector = BiasDetector()
        result = detector.analyze("All women are always emotional. Every man is logical.")

        assert result["has_potential_bias"]
        assert result["bias_score"] > 0.3


class TestContentSafetyAnalyzer:
    """Tests for ContentSafetyAnalyzer."""

    def test_safe_text(self):
        """Test analysis of safe text."""
        analyzer = ContentSafetyAnalyzer()
        report = analyzer.analyze("Hello, this is a friendly message about coding.")

        assert report.is_safe
        assert report.overall_risk in (RiskLevel.NONE, RiskLevel.LOW)

    def test_pii_detection(self):
        """Test PII detection in analysis."""
        analyzer = ContentSafetyAnalyzer()
        report = analyzer.analyze("My email is test@example.com")

        assert not report.is_safe
        assert any(f.category == SafetyCategory.PII_EXPOSURE for f in report.flags)

    def test_selective_checks(self):
        """Test selective analysis."""
        analyzer = ContentSafetyAnalyzer()

        # Only check PII
        report = analyzer.analyze(
            "Email: test@example.com",
            check_pii=True,
            check_toxicity=False,
            check_hallucination=False,
            check_bias=False,
        )

        assert "pii_count" in report.scores
        assert "toxicity_flags" not in report.scores

    def test_multiple_issues(self):
        """Test detection of multiple issues."""
        analyzer = ContentSafetyAnalyzer()
        text = "Email: bad@example.com. You're stupid. Obviously everyone knows this."

        report = analyzer.analyze(text)

        # Should have multiple flags
        assert len(report.flags) >= 2


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_quick_safety_check_safe(self):
        """Test quick check on safe text."""
        is_safe, risk, issues = quick_safety_check("Hello world")

        assert is_safe
        assert risk in (RiskLevel.NONE, RiskLevel.LOW)

    def test_quick_safety_check_unsafe(self):
        """Test quick check on unsafe text."""
        is_safe, risk, issues = quick_safety_check("Contact me at secret@email.com")

        assert not is_safe
        assert len(issues) > 0

    def test_mask_pii_function(self):
        """Test mask_pii utility function."""
        result = mask_pii("Email: test@example.com")

        assert "[EMAIL]" in result
        assert "test@example.com" not in result

    def test_detect_pii_function(self):
        """Test detect_pii utility function."""
        matches = detect_pii("Call 555-123-4567 or email test@example.com")

        assert len(matches) >= 2
        assert any(m["type"] == "email" for m in matches)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self):
        """Test analysis of empty text."""
        analyzer = ContentSafetyAnalyzer()
        report = analyzer.analyze("")

        assert report.is_safe

    def test_unicode_text(self):
        """Test analysis of unicode text."""
        analyzer = ContentSafetyAnalyzer()
        report = analyzer.analyze("Hello 世界! Bonjour le monde!")

        assert report.is_safe

    def test_very_long_text(self):
        """Test analysis of long text."""
        analyzer = ContentSafetyAnalyzer()
        long_text = "This is a test. " * 1000
        report = analyzer.analyze(long_text)

        assert report is not None

    def test_special_characters(self):
        """Test text with special characters."""
        detector = PIIDetector()
        text = "Contact: <script>alert('xss')</script>"
        report = detector.detect(text)

        # Should not crash
        assert report is not None
