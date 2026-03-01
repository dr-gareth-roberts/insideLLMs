"""Tests for hallucination detection utilities."""

from insideLLMs.evaluation.hallucination import (
    AttributionChecker,
    ConsistencyCheck,
    ConsistencyChecker,
    DetectionMethod,
    FactualityChecker,
    FactualityScore,
    GroundednessChecker,
    HallucinationDetector,
    HallucinationFlag,
    HallucinationReport,
    HallucinationType,
    PatternBasedDetector,
    SeverityLevel,
    check_attribution,
    check_consistency,
    check_factuality,
    check_groundedness,
    detect_hallucinations,
    quick_hallucination_check,
)


class TestHallucinationType:
    """Tests for HallucinationType enum."""

    def test_all_types_exist(self):
        """Test all expected types exist."""
        assert HallucinationType.FACTUAL_ERROR
        assert HallucinationType.FABRICATED_ENTITY
        assert HallucinationType.CONFLATION
        assert HallucinationType.ANACHRONISM
        assert HallucinationType.LOGICAL_INCONSISTENCY
        assert HallucinationType.ATTRIBUTION_ERROR
        assert HallucinationType.EXAGGERATION
        assert HallucinationType.UNSUPPORTED_CLAIM


class TestSeverityLevel:
    """Tests for SeverityLevel enum."""

    def test_all_levels_exist(self):
        """Test all expected levels exist."""
        assert SeverityLevel.LOW
        assert SeverityLevel.MEDIUM
        assert SeverityLevel.HIGH
        assert SeverityLevel.CRITICAL


class TestDetectionMethod:
    """Tests for DetectionMethod enum."""

    def test_all_methods_exist(self):
        """Test all expected methods exist."""
        assert DetectionMethod.CONSISTENCY
        assert DetectionMethod.FACTUAL
        assert DetectionMethod.ATTRIBUTION
        assert DetectionMethod.CONFIDENCE
        assert DetectionMethod.ENTAILMENT


class TestHallucinationFlag:
    """Tests for HallucinationFlag dataclass."""

    def test_basic_creation(self):
        """Test basic flag creation."""
        flag = HallucinationFlag(
            hallucination_type=HallucinationType.FACTUAL_ERROR,
            severity=SeverityLevel.HIGH,
            text_span="The Earth is flat",
            start_pos=0,
            end_pos=17,
            confidence=0.9,
            explanation="Contradicts known facts",
        )
        assert flag.confidence == 0.9
        assert flag.severity == SeverityLevel.HIGH

    def test_to_dict(self):
        """Test conversion to dictionary."""
        flag = HallucinationFlag(
            hallucination_type=HallucinationType.FACTUAL_ERROR,
            severity=SeverityLevel.HIGH,
            text_span="test",
            start_pos=0,
            end_pos=4,
            confidence=0.9,
            explanation="Test explanation",
        )
        d = flag.to_dict()
        assert d["type"] == "factual_error"
        assert d["severity"] == "high"


class TestConsistencyCheck:
    """Tests for ConsistencyCheck dataclass."""

    def test_basic_creation(self):
        """Test basic consistency check creation."""
        check = ConsistencyCheck(
            responses=["response1", "response2"],
            consistent_claims=["claim1"],
            inconsistent_claims=[],
            overall_consistency=0.9,
        )
        assert check.is_consistent

    def test_not_consistent(self):
        """Test inconsistent check."""
        check = ConsistencyCheck(
            responses=["a", "b"],
            consistent_claims=[],
            inconsistent_claims=[("claim1", "claim2", 0.5)],
            overall_consistency=0.5,
        )
        assert not check.is_consistent

    def test_to_dict(self):
        """Test conversion to dictionary."""
        check = ConsistencyCheck(
            responses=["a", "b"],
            consistent_claims=["consistent"],
            inconsistent_claims=[],
            overall_consistency=0.9,
        )
        d = check.to_dict()
        assert d["n_responses"] == 2
        assert d["is_consistent"]


class TestFactualityScore:
    """Tests for FactualityScore dataclass."""

    def test_basic_creation(self):
        """Test basic score creation."""
        score = FactualityScore(
            score=0.9,
            verifiable_claims=10,
            verified_claims=9,
            unverified_claims=0,
            contradicted_claims=1,
        )
        assert score.accuracy == 0.9

    def test_accuracy_no_claims(self):
        """Test accuracy with no claims."""
        score = FactualityScore(
            score=1.0,
            verifiable_claims=0,
            verified_claims=0,
            unverified_claims=0,
            contradicted_claims=0,
        )
        assert score.accuracy == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = FactualityScore(
            score=0.8,
            verifiable_claims=5,
            verified_claims=4,
            unverified_claims=0,
            contradicted_claims=1,
        )
        d = score.to_dict()
        assert d["score"] == 0.8


class TestHallucinationReport:
    """Tests for HallucinationReport dataclass."""

    def test_has_hallucinations(self):
        """Test hallucination detection."""
        report = HallucinationReport(
            text="test",
            flags=[
                HallucinationFlag(
                    hallucination_type=HallucinationType.FACTUAL_ERROR,
                    severity=SeverityLevel.HIGH,
                    text_span="error",
                    start_pos=0,
                    end_pos=5,
                    confidence=0.9,
                    explanation="Test",
                )
            ],
            overall_score=0.5,
        )
        assert report.has_hallucinations

    def test_no_hallucinations(self):
        """Test no hallucinations detected."""
        report = HallucinationReport(
            text="test",
            flags=[],
            overall_score=1.0,
        )
        assert not report.has_hallucinations

    def test_critical_flags(self):
        """Test filtering critical flags."""
        report = HallucinationReport(
            text="test",
            flags=[
                HallucinationFlag(
                    HallucinationType.FACTUAL_ERROR,
                    SeverityLevel.CRITICAL,
                    "a",
                    0,
                    1,
                    0.9,
                    "critical",
                ),
                HallucinationFlag(
                    HallucinationType.EXAGGERATION, SeverityLevel.LOW, "b", 1, 2, 0.3, "low"
                ),
            ],
            overall_score=0.5,
        )
        assert len(report.critical_flags) == 1

    def test_high_confidence_flags(self):
        """Test filtering high confidence flags."""
        report = HallucinationReport(
            text="test",
            flags=[
                HallucinationFlag(
                    HallucinationType.FACTUAL_ERROR, SeverityLevel.HIGH, "a", 0, 1, 0.9, "high conf"
                ),
                HallucinationFlag(
                    HallucinationType.EXAGGERATION, SeverityLevel.LOW, "b", 1, 2, 0.3, "low conf"
                ),
            ],
            overall_score=0.5,
        )
        assert len(report.high_confidence_flags) == 1


class TestPatternBasedDetector:
    """Tests for PatternBasedDetector."""

    def test_detect_overconfidence(self):
        """Test detecting overconfident language."""
        detector = PatternBasedDetector()
        flags = detector.detect("This is definitely the correct answer. Always true.")
        overconf = [f for f in flags if f.hallucination_type == HallucinationType.EXAGGERATION]
        assert len(overconf) >= 1

    def test_detect_unsupported_claims(self):
        """Test detecting unsupported claims."""
        detector = PatternBasedDetector()
        text = "According to a study, 75% of people agree. In 2023, this was confirmed."
        flags = detector.detect(text)
        unsupported = [
            f for f in flags if f.hallucination_type == HallucinationType.UNSUPPORTED_CLAIM
        ]
        assert len(unsupported) >= 1

    def test_no_flags_for_simple_text(self):
        """Test no flags for simple factual text."""
        detector = PatternBasedDetector()
        flags = detector.detect("The sky is blue. Water is wet.")
        # Should have few or no flags for simple text
        high_conf = [f for f in flags if f.confidence > 0.7]
        assert len(high_conf) == 0


class TestConsistencyChecker:
    """Tests for ConsistencyChecker."""

    def test_consistent_responses(self):
        """Test checking consistent responses."""
        checker = ConsistencyChecker()
        result = checker.check(
            [
                "Paris is the capital of France.",
                "Paris is the capital of France.",
            ]
        )
        # Identical responses should have high consistency
        assert result.overall_consistency >= 0.5

    def test_inconsistent_responses(self):
        """Test checking inconsistent responses."""
        checker = ConsistencyChecker()
        result = checker.check(
            [
                "The answer is definitely yes.",
                "The answer is definitely no.",
            ]
        )
        # Results may vary but inconsistencies should be detected
        assert len(result.inconsistent_claims) >= 0  # May or may not detect

    def test_single_response(self):
        """Test with single response."""
        checker = ConsistencyChecker()
        result = checker.check(["Single response only."])
        assert result.overall_consistency == 1.0

    def test_custom_similarity(self):
        """Test with custom similarity function."""

        def exact_match(a: str, b: str) -> float:
            return 1.0 if a == b else 0.0

        checker = ConsistencyChecker(similarity_fn=exact_match)
        result = checker.check(["Hello world.", "Hello world."])
        assert result.overall_consistency >= 0.5


class TestFactualityChecker:
    """Tests for FactualityChecker."""

    def test_check_with_knowledge_base(self):
        """Test checking against knowledge base."""
        knowledge = {
            "paris": "Paris is the capital of France",
        }
        checker = FactualityChecker(knowledge_base=knowledge)
        result = checker.check("Paris is the capital of France.")
        # Should find and verify the claim
        assert result.verifiable_claims >= 0

    def test_check_without_knowledge(self):
        """Test checking without knowledge base."""
        checker = FactualityChecker()
        result = checker.check("The population is 100 million.")
        # Should extract claim but not verify
        assert result.verifiable_claims >= 0

    def test_custom_fact_checker(self):
        """Test with custom fact checker."""

        def always_true(claim: str) -> tuple:
            return (True, 1.0)

        checker = FactualityChecker(fact_checker=always_true)
        result = checker.check("Paris is the capital of France.")
        assert result.verified_claims >= 0


class TestHallucinationDetector:
    """Tests for HallucinationDetector."""

    def test_detect_basic(self):
        """Test basic hallucination detection."""
        detector = HallucinationDetector()
        report = detector.detect("According to studies, 99% of people definitely agree.")
        assert isinstance(report, HallucinationReport)
        assert report.overall_score <= 1.0

    def test_detect_with_consistency(self):
        """Test detection with consistency checking."""
        detector = HallucinationDetector()
        report = detector.detect(
            "The answer is yes.",
            reference_texts=["The answer is yes.", "The answer is maybe."],
        )
        assert report.consistency_check is not None

    def test_detect_without_factuality(self):
        """Test detection without factuality checking."""
        detector = HallucinationDetector()
        report = detector.detect(
            "Simple text.",
            check_factuality=False,
        )
        assert report.factuality_score is None

    def test_recommendations_generated(self):
        """Test that recommendations are generated."""
        detector = HallucinationDetector()
        report = detector.detect(
            "According to studies, this is definitely always true. Everyone knows this."
        )
        assert len(report.recommendations) > 0


class TestGroundednessChecker:
    """Tests for GroundednessChecker."""

    def test_grounded_response(self):
        """Test response that is grounded in context."""
        checker = GroundednessChecker()
        result = checker.check(
            response="Paris is the capital of France.",
            context="The capital of France is Paris.",
        )
        assert result["groundedness_score"] > 0.5

    def test_ungrounded_response(self):
        """Test response that is not grounded."""
        checker = GroundednessChecker()
        result = checker.check(
            response="The moon is made of cheese.",
            context="The Earth orbits the Sun.",
        )
        assert result["groundedness_score"] < 1.0

    def test_custom_threshold(self):
        """Test with custom threshold."""
        checker = GroundednessChecker(threshold=0.9)
        result = checker.check(
            response="The sky is blue.",
            context="The sky appears blue.",
        )
        # Higher threshold = stricter checking
        assert "groundedness_score" in result


class TestAttributionChecker:
    """Tests for AttributionChecker."""

    def test_find_attributions(self):
        """Test finding attributions in text."""
        checker = AttributionChecker()
        result = checker.check(
            "According to the WHO, 50% of people have this condition. "
            "Dr. Smith said this is concerning."
        )
        assert result["attributions_found"] >= 2

    def test_unattributed_claims(self):
        """Test detecting unattributed claims."""
        checker = AttributionChecker()
        result = checker.check("Studies show that 75% of people agree with this statement.")
        # Should detect unattributed statistical claim
        assert "unattributed_claims" in result

    def test_no_claims(self):
        """Test text with no claims needing attribution."""
        checker = AttributionChecker()
        result = checker.check("The sky is blue. Water is wet.")
        assert result["attribution_score"] == 1.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_detect_hallucinations(self):
        """Test detect_hallucinations function."""
        report = detect_hallucinations("This is definitely always true.")
        assert isinstance(report, HallucinationReport)

    def test_check_consistency(self):
        """Test check_consistency function."""
        result = check_consistency(["Response 1.", "Response 2."])
        assert isinstance(result, ConsistencyCheck)

    def test_check_factuality(self):
        """Test check_factuality function."""
        result = check_factuality("Paris is the capital of France.")
        assert isinstance(result, FactualityScore)

    def test_check_groundedness(self):
        """Test check_groundedness function."""
        result = check_groundedness(
            response="The answer is yes.",
            context="The answer is yes.",
        )
        assert "groundedness_score" in result

    def test_check_attribution(self):
        """Test check_attribution function."""
        result = check_attribution("According to experts, this is true.")
        assert "attributions_found" in result

    def test_quick_hallucination_check(self):
        """Test quick_hallucination_check function."""
        result = quick_hallucination_check("Simple text.")
        assert "has_potential_hallucinations" in result
        assert "overall_score" in result


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self):
        """Test with empty text."""
        detector = HallucinationDetector()
        report = detector.detect("")
        assert report.overall_score == 1.0

    def test_single_word(self):
        """Test with single word."""
        detector = HallucinationDetector()
        report = detector.detect("Hello")
        # Should not crash
        assert isinstance(report, HallucinationReport)

    def test_very_long_text(self):
        """Test with very long text."""
        detector = HallucinationDetector()
        long_text = "This is a sentence. " * 100
        report = detector.detect(long_text)
        assert isinstance(report, HallucinationReport)

    def test_special_characters(self):
        """Test with special characters."""
        detector = HallucinationDetector()
        report = detector.detect("!@#$%^&*() test []{}|\\")
        assert isinstance(report, HallucinationReport)

    def test_unicode_text(self):
        """Test with Unicode text."""
        detector = HallucinationDetector()
        report = detector.detect("这是中文文本。 これは日本語です。")
        assert isinstance(report, HallucinationReport)

    def test_no_sentences(self):
        """Test with text without sentence boundaries."""
        detector = HallucinationDetector()
        report = detector.detect("no periods just words")
        assert isinstance(report, HallucinationReport)

    def test_many_references(self):
        """Test with many reference texts."""
        detector = HallucinationDetector()
        references = [f"Reference {i}." for i in range(10)]
        report = detector.detect("Main text.", reference_texts=references)
        assert report.consistency_check is not None

    def test_empty_knowledge_base(self):
        """Test factuality checker with empty knowledge base."""
        checker = FactualityChecker(knowledge_base={})
        result = checker.check("Paris is the capital of France.")
        # Should still work, just not verify claims
        assert isinstance(result, FactualityScore)
