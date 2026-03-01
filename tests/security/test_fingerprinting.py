"""Tests for model capability fingerprinting."""

from insideLLMs.security.fingerprinting import (
    CapabilityCategory,
    CapabilityLevel,
    CapabilityProfiler,
    CapabilityScore,
    FingerprintComparator,
    FingerprintComparison,
    FingerprintGenerator,
    LimitationDetector,
    LimitationReport,
    LimitationType,
    ModelFingerprint,
    ReasoningProbe,
    SkillAssessor,
    SkillProfile,
    SkillType,
    compare_fingerprints,
    create_fingerprint,
    detect_limitations,
    quick_capability_assessment,
)


class TestCapabilityCategory:
    """Tests for CapabilityCategory enum."""

    def test_all_categories_exist(self):
        """Test all expected categories exist."""
        assert CapabilityCategory.REASONING
        assert CapabilityCategory.LANGUAGE
        assert CapabilityCategory.KNOWLEDGE
        assert CapabilityCategory.CREATIVITY
        assert CapabilityCategory.CODING
        assert CapabilityCategory.MATH
        assert CapabilityCategory.ANALYSIS
        assert CapabilityCategory.INSTRUCTION_FOLLOWING


class TestCapabilityLevel:
    """Tests for CapabilityLevel enum."""

    def test_all_levels_exist(self):
        """Test all expected levels exist."""
        assert CapabilityLevel.NONE
        assert CapabilityLevel.BASIC
        assert CapabilityLevel.INTERMEDIATE
        assert CapabilityLevel.ADVANCED
        assert CapabilityLevel.EXPERT


class TestSkillType:
    """Tests for SkillType enum."""

    def test_reasoning_skills_exist(self):
        """Test reasoning skills exist."""
        assert SkillType.LOGICAL_DEDUCTION
        assert SkillType.CAUSAL_REASONING
        assert SkillType.ANALOGICAL_REASONING

    def test_coding_skills_exist(self):
        """Test coding skills exist."""
        assert SkillType.CODE_GENERATION
        assert SkillType.DEBUGGING
        assert SkillType.CODE_UNDERSTANDING


class TestLimitationType:
    """Tests for LimitationType enum."""

    def test_all_limitations_exist(self):
        """Test all expected limitations exist."""
        assert LimitationType.FACTUAL_ERROR_PRONE
        assert LimitationType.CONTEXT_LENGTH_LIMITED
        assert LimitationType.HALLUCINATION_PRONE
        assert LimitationType.REPETITIVE_OUTPUT


class TestCapabilityScore:
    """Tests for CapabilityScore dataclass."""

    def test_basic_creation(self):
        """Test basic score creation."""
        score = CapabilityScore(
            category=CapabilityCategory.REASONING,
            skill=SkillType.LOGICAL_DEDUCTION,
            score=0.85,
            level=CapabilityLevel.ADVANCED,
            confidence=0.9,
            evidence=["Good performance"],
        )
        assert score.score == 0.85
        assert score.level == CapabilityLevel.ADVANCED

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = CapabilityScore(
            category=CapabilityCategory.MATH,
            skill=SkillType.ARITHMETIC,
            score=0.9,
            level=CapabilityLevel.EXPERT,
            confidence=0.95,
        )
        d = score.to_dict()
        assert d["category"] == "math"
        assert d["skill"] == "arithmetic"
        assert d["level"] == "expert"


class TestLimitationReport:
    """Tests for LimitationReport dataclass."""

    def test_basic_creation(self):
        """Test basic report creation."""
        report = LimitationReport(
            limitation_type=LimitationType.REPETITIVE_OUTPUT,
            severity=0.7,
            frequency=0.3,
            examples=["Example 1"],
            mitigation_suggestions=["Increase temperature"],
        )
        assert report.severity == 0.7

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = LimitationReport(
            limitation_type=LimitationType.HALLUCINATION_PRONE,
            severity=0.5,
            frequency=0.2,
            examples=[],
            mitigation_suggestions=["Use RAG"],
        )
        d = report.to_dict()
        assert d["type"] == "hallucination_prone"


class TestSkillProfile:
    """Tests for SkillProfile dataclass."""

    def test_basic_creation(self):
        """Test basic profile creation."""
        profile = SkillProfile(
            category=CapabilityCategory.CODING,
            skills={},
            overall_level=CapabilityLevel.INTERMEDIATE,
            overall_score=0.6,
            strengths=[SkillType.CODE_GENERATION],
            weaknesses=[],
        )
        assert profile.overall_score == 0.6

    def test_to_dict(self):
        """Test conversion to dictionary."""
        profile = SkillProfile(
            category=CapabilityCategory.REASONING,
            skills={
                SkillType.LOGICAL_DEDUCTION: CapabilityScore(
                    CapabilityCategory.REASONING,
                    SkillType.LOGICAL_DEDUCTION,
                    0.8,
                    CapabilityLevel.ADVANCED,
                    0.9,
                )
            },
            overall_level=CapabilityLevel.ADVANCED,
            overall_score=0.8,
            strengths=[SkillType.LOGICAL_DEDUCTION],
            weaknesses=[],
        )
        d = profile.to_dict()
        assert d["category"] == "reasoning"
        assert "logical_deduction" in d["skills"]


class TestModelFingerprint:
    """Tests for ModelFingerprint dataclass."""

    def test_capability_signature(self):
        """Test capability signature generation."""
        fingerprint = ModelFingerprint(
            model_id="test-model",
            version="1.0",
            fingerprint_hash="abc123",
            category_scores={
                CapabilityCategory.REASONING: 0.9,
                CapabilityCategory.CODING: 0.8,
                CapabilityCategory.MATH: 0.7,
            },
            skill_profiles={},
            limitations=[],
            top_capabilities=[],
            main_limitations=[],
            overall_capability_score=0.8,
        )
        sig = fingerprint.capability_signature
        assert "rea:" in sig
        assert "cod:" in sig

    def test_to_dict(self):
        """Test conversion to dictionary."""
        fingerprint = ModelFingerprint(
            model_id="test-model",
            version="1.0",
            fingerprint_hash="abc123",
            category_scores={CapabilityCategory.REASONING: 0.8},
            skill_profiles={},
            limitations=[],
            top_capabilities=[(SkillType.LOGICAL_DEDUCTION, 0.9)],
            main_limitations=[],
            overall_capability_score=0.8,
            metadata={"source": "test"},
        )
        d = fingerprint.to_dict()
        assert d["model_id"] == "test-model"
        assert d["overall_capability_score"] == 0.8


class TestFingerprintComparison:
    """Tests for FingerprintComparison dataclass."""

    def test_winner_clear(self):
        """Test winner determination with clear winner."""
        comparison = FingerprintComparison(
            model_a_id="model_a",
            model_b_id="model_b",
            similarity_score=0.6,
            category_differences={},
            skill_differences={},
            model_a_advantages=[
                SkillType.CODE_GENERATION,
                SkillType.DEBUGGING,
                SkillType.CODE_UNDERSTANDING,
            ],
            model_b_advantages=[SkillType.GRAMMAR],
            shared_strengths=[],
            shared_weaknesses=[],
        )
        assert comparison.winner == "model_a"

    def test_winner_unclear(self):
        """Test winner determination when unclear."""
        comparison = FingerprintComparison(
            model_a_id="model_a",
            model_b_id="model_b",
            similarity_score=0.9,
            category_differences={},
            skill_differences={},
            model_a_advantages=[SkillType.CODE_GENERATION],
            model_b_advantages=[SkillType.GRAMMAR],
            shared_strengths=[],
            shared_weaknesses=[],
        )
        assert comparison.winner is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        comparison = FingerprintComparison(
            model_a_id="model_a",
            model_b_id="model_b",
            similarity_score=0.8,
            category_differences={CapabilityCategory.REASONING: 0.1},
            skill_differences={SkillType.LOGICAL_DEDUCTION: 0.05},
            model_a_advantages=[],
            model_b_advantages=[],
            shared_strengths=[SkillType.CODE_GENERATION],
            shared_weaknesses=[],
        )
        d = comparison.to_dict()
        assert d["similarity_score"] == 0.8


class TestReasoningProbe:
    """Tests for ReasoningProbe."""

    def test_generate_test(self):
        """Test generating reasoning test."""
        probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
        prompt, expected = probe.generate_test()
        assert "roses" in prompt.lower() or "reasoning" in prompt.lower()
        assert expected is not None

    def test_evaluate_correct(self):
        """Test evaluating correct response."""
        probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
        _, expected = probe.generate_test()
        score = probe.evaluate("No, we cannot conclude that.", expected)
        assert score > 0.5

    def test_evaluate_incorrect(self):
        """Test evaluating incorrect response."""
        probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
        _, expected = probe.generate_test()
        score = probe.evaluate("Yes, definitely.", expected)
        assert score < 0.5


class TestSkillAssessor:
    """Tests for SkillAssessor."""

    def test_assess_skill(self):
        """Test assessing a skill."""
        assessor = SkillAssessor()
        responses = [
            ("The answer is 4", "4"),
            ("I think it's 5", "5"),
            ("That would be 10", "10"),
        ]
        score = assessor.assess_skill(SkillType.ARITHMETIC, responses)
        assert score.skill == SkillType.ARITHMETIC
        assert score.score > 0

    def test_assess_skill_empty(self):
        """Test assessing with no responses."""
        assessor = SkillAssessor()
        score = assessor.assess_skill(SkillType.ARITHMETIC, [])
        assert score.score == 0.0
        assert score.level == CapabilityLevel.NONE

    def test_custom_evaluator(self):
        """Test with custom evaluator."""

        def strict_evaluator(response: str, expected: str) -> float:
            return 1.0 if response.strip() == expected.strip() else 0.0

        assessor = SkillAssessor(evaluator=strict_evaluator)
        responses = [
            ("correct", "correct"),
            ("wrong", "correct"),
        ]
        score = assessor.assess_skill(SkillType.COMMON_SENSE, responses)
        assert score.score == 0.5

    def test_register_probe(self):
        """Test registering a probe."""
        assessor = SkillAssessor()
        probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
        assessor.register_probe(probe)
        # Should not raise


class TestLimitationDetector:
    """Tests for LimitationDetector."""

    def test_detect_repetition(self):
        """Test detecting repetitive output."""
        detector = LimitationDetector()
        responses = [
            "The answer is important. The answer is important. The answer is important.",
            "This is a normal response without repetition.",
        ]
        reports = detector.detect(responses)
        repetition_reports = [
            r for r in reports if r.limitation_type == LimitationType.REPETITIVE_OUTPUT
        ]
        assert len(repetition_reports) > 0

    def test_detect_format_issues(self):
        """Test detecting format inconsistencies."""
        detector = LimitationDetector()
        responses = [
            "Here's a list:\n- Item 1\n* Item 2\n1. Item 3",  # Mixed markers
            "Normal text",
        ]
        reports = detector.detect(responses)
        format_reports = [
            r for r in reports if r.limitation_type == LimitationType.FORMAT_INCONSISTENT
        ]
        assert len(format_reports) > 0

    def test_detect_hallucination_markers(self):
        """Test detecting hallucination markers."""
        detector = LimitationDetector()
        responses = [
            "I believe it might be around approximately roughly possibly true.",
        ]
        reports = detector.detect(responses)
        hallucination_reports = [
            r for r in reports if r.limitation_type == LimitationType.HALLUCINATION_PRONE
        ]
        assert len(hallucination_reports) > 0

    def test_no_limitations(self):
        """Test with clean responses."""
        detector = LimitationDetector()
        responses = [
            "This is a clear and direct response.",
            "Here is another well-formatted response.",
        ]
        reports = detector.detect(responses)
        # May still have some reports, but should be low severity
        high_severity = [r for r in reports if r.severity > 0.8]
        assert len(high_severity) == 0


class TestCapabilityProfiler:
    """Tests for CapabilityProfiler."""

    def test_profile_category(self):
        """Test profiling a category."""
        profiler = CapabilityProfiler()
        skill_results = {
            SkillType.LOGICAL_DEDUCTION: [
                ("No, that's invalid", False),
                ("Cannot conclude", False),
            ],
            SkillType.CAUSAL_REASONING: [
                ("Check other areas", "check_other_areas"),
            ],
        }
        profile = profiler.profile_category(CapabilityCategory.REASONING, skill_results)
        assert profile.category == CapabilityCategory.REASONING
        assert len(profile.skills) > 0

    def test_profile_empty_category(self):
        """Test profiling with no matching skills."""
        profiler = CapabilityProfiler()
        profile = profiler.profile_category(
            CapabilityCategory.CODING,
            {},  # No results
        )
        assert profile.overall_score == 0.0
        assert profile.overall_level == CapabilityLevel.NONE


class TestFingerprintGenerator:
    """Tests for FingerprintGenerator."""

    def test_generate_fingerprint(self):
        """Test generating a fingerprint."""
        generator = FingerprintGenerator()
        skill_results = {
            SkillType.ARITHMETIC: [
                ("4", "4"),
                ("10", "10"),
            ],
            SkillType.CODE_GENERATION: [
                ("def hello(): print('hello')", "hello function"),
            ],
        }
        responses = ["Test response 1", "Test response 2"]

        fingerprint = generator.generate(
            model_id="test-model",
            skill_results=skill_results,
            responses=responses,
            version="1.0",
            metadata={"source": "test"},
        )

        assert fingerprint.model_id == "test-model"
        assert fingerprint.version == "1.0"
        assert len(fingerprint.fingerprint_hash) == 16

    def test_fingerprint_hash_uniqueness(self):
        """Test that different fingerprints have different hashes."""
        generator = FingerprintGenerator()

        fp1 = generator.generate(
            "model-a",
            {SkillType.ARITHMETIC: [("5", "5")]},
            [],
        )
        fp2 = generator.generate(
            "model-b",
            {SkillType.ARITHMETIC: [("5", "5")]},
            [],
        )

        assert fp1.fingerprint_hash != fp2.fingerprint_hash


class TestFingerprintComparator:
    """Tests for FingerprintComparator."""

    def _create_fingerprint(self, model_id: str, scores: dict) -> ModelFingerprint:
        """Helper to create a fingerprint."""
        skill_profiles = {}
        for cat in CapabilityCategory:
            skill_profiles[cat] = SkillProfile(
                category=cat,
                skills={},
                overall_level=CapabilityLevel.INTERMEDIATE,
                overall_score=scores.get(cat, 0.5),
                strengths=[],
                weaknesses=[],
            )

        return ModelFingerprint(
            model_id=model_id,
            version="1.0",
            fingerprint_hash="test",
            category_scores=scores,
            skill_profiles=skill_profiles,
            limitations=[],
            top_capabilities=[],
            main_limitations=[],
            overall_capability_score=sum(scores.values()) / max(len(scores), 1),
        )

    def test_compare_similar(self):
        """Test comparing similar fingerprints."""
        comparator = FingerprintComparator()

        fp1 = self._create_fingerprint(
            "model_a",
            {
                CapabilityCategory.REASONING: 0.8,
                CapabilityCategory.CODING: 0.7,
            },
        )
        fp2 = self._create_fingerprint(
            "model_b",
            {
                CapabilityCategory.REASONING: 0.78,
                CapabilityCategory.CODING: 0.72,
            },
        )

        comparison = comparator.compare(fp1, fp2)
        assert comparison.similarity_score > 0.8

    def test_compare_different(self):
        """Test comparing different fingerprints."""
        comparator = FingerprintComparator()

        # Use all categories to make difference clearer
        all_categories_high = dict.fromkeys(CapabilityCategory, 0.9)
        all_categories_low = dict.fromkeys(CapabilityCategory, 0.2)

        fp1 = self._create_fingerprint("model_a", all_categories_high)
        fp2 = self._create_fingerprint("model_b", all_categories_low)

        comparison = comparator.compare(fp1, fp2)
        # With large differences across all categories
        assert comparison.similarity_score < 0.5


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_fingerprint(self):
        """Test create_fingerprint function."""
        fingerprint = create_fingerprint(
            model_id="test-model",
            skill_results={
                SkillType.ARITHMETIC: [("4", "4")],
            },
            responses=["Test response"],
        )
        assert isinstance(fingerprint, ModelFingerprint)
        assert fingerprint.model_id == "test-model"

    def test_compare_fingerprints(self):
        """Test compare_fingerprints function."""
        fp1 = create_fingerprint("model_a", {}, [])
        fp2 = create_fingerprint("model_b", {}, [])

        comparison = compare_fingerprints(fp1, fp2)
        assert isinstance(comparison, FingerprintComparison)

    def test_quick_capability_assessment(self):
        """Test quick_capability_assessment function."""
        responses = {
            "arithmetic": ("4", "4"),
            "common_sense": ("yes", "yes"),
        }
        result = quick_capability_assessment(responses)

        assert "skills" in result
        assert "overall_score" in result
        assert result["n_skills_tested"] == 2

    def test_detect_limitations(self):
        """Test detect_limitations function."""
        responses = [
            "The answer is repeated. The answer is repeated.",
            "Normal response here.",
        ]
        limitations = detect_limitations(responses)
        assert isinstance(limitations, list)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_skill_results(self):
        """Test with empty skill results."""
        generator = FingerprintGenerator()
        fingerprint = generator.generate("test", {}, [])
        assert fingerprint.overall_capability_score == 0.0

    def test_single_response(self):
        """Test with single response."""
        detector = LimitationDetector()
        reports = detector.detect(["Single response"])
        # Should not crash
        assert isinstance(reports, list)

    def test_empty_responses(self):
        """Test with empty responses."""
        detector = LimitationDetector()
        reports = detector.detect([])
        assert len(reports) == 0

    def test_very_long_response(self):
        """Test with very long response."""
        detector = LimitationDetector()
        long_response = "word " * 1000
        reports = detector.detect([long_response])
        assert isinstance(reports, list)

    def test_special_characters_in_response(self):
        """Test with special characters."""
        assessor = SkillAssessor()
        responses = [
            ("@#$%^&*()", "@#$%^&*()"),
            ("日本語", "日本語"),
        ]
        score = assessor.assess_skill(SkillType.COMMON_SENSE, responses)
        assert score.score > 0

    def test_profiler_with_wrong_category_skills(self):
        """Test profiler with skills from wrong category."""
        profiler = CapabilityProfiler()
        # Pass coding skills when asking for reasoning
        skill_results = {
            SkillType.CODE_GENERATION: [("code", "code")],
        }
        profile = profiler.profile_category(
            CapabilityCategory.REASONING,  # Wrong category
            skill_results,
        )
        # Should return empty profile since skills don't match category
        assert len(profile.skills) == 0
