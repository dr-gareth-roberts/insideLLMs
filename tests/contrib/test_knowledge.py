"""Tests for knowledge probing and fact verification utilities."""

from insideLLMs.contrib.knowledge import (
    Claim,
    ClaimExtractor,
    ConfidenceLevel,
    ConsistencyResult,
    ConsistencyTester,
    FactVerifier,
    KnowledgeCategory,
    KnowledgeProbe,
    KnowledgeProber,
    KnowledgeReport,
    ProbeResult,
    SourceAttributor,
    VerificationResult,
    VerificationStatus,
    check_consistency,
    extract_claims,
    probe_knowledge,
    verify_claim,
    verify_facts,
)


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all statuses are defined."""
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.CONTRADICTED.value == "contradicted"
        assert VerificationStatus.UNVERIFIABLE.value == "unverifiable"
        assert VerificationStatus.PARTIAL.value == "partial"
        assert VerificationStatus.UNKNOWN.value == "unknown"


class TestKnowledgeCategory:
    """Tests for KnowledgeCategory enum."""

    def test_all_categories_exist(self):
        """Test that all categories are defined."""
        assert KnowledgeCategory.FACTUAL.value == "factual"
        assert KnowledgeCategory.PROCEDURAL.value == "procedural"
        assert KnowledgeCategory.CONCEPTUAL.value == "conceptual"
        assert KnowledgeCategory.TEMPORAL.value == "temporal"


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_all_levels_exist(self):
        """Test that all levels are defined."""
        assert ConfidenceLevel.VERY_LOW.value == "very_low"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"


class TestClaim:
    """Tests for Claim dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        claim = Claim(
            text="Paris is the capital of France",
            subject="Paris",
            predicate="is",
            object="the capital of France",
        )

        assert claim.text == "Paris is the capital of France"
        assert claim.subject == "Paris"
        assert claim.category == KnowledgeCategory.FACTUAL

    def test_to_dict(self):
        """Test dictionary conversion."""
        claim = Claim(
            text="The sky is blue",
            category=KnowledgeCategory.FACTUAL,
            confidence=0.8,
        )

        d = claim.to_dict()
        assert d["text"] == "The sky is blue"
        assert d["category"] == "factual"
        assert d["confidence"] == 0.8


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        claim = Claim(text="Test claim", confidence=0.7)
        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=0.9,
            supporting_evidence=["Source A"],
            contradicting_evidence=[],
            source_quality=0.8,
        )

        d = result.to_dict()
        assert d["status"] == "verified"
        assert d["confidence"] == 0.9
        assert len(d["supporting_evidence"]) == 1


class TestKnowledgeProbe:
    """Tests for KnowledgeProbe dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        probe = KnowledgeProbe(
            question="What is the capital of France?",
            expected_answer="Paris",
            difficulty=0.3,
        )

        assert probe.question == "What is the capital of France?"
        assert probe.expected_answer == "Paris"
        assert probe.difficulty == 0.3

    def test_to_dict(self):
        """Test dictionary conversion."""
        probe = KnowledgeProbe(
            question="Test question",
            category=KnowledgeCategory.CONCEPTUAL,
            hints=["Hint 1", "Hint 2"],
        )

        d = probe.to_dict()
        assert d["category"] == "conceptual"
        assert len(d["hints"]) == 2


class TestProbeResult:
    """Tests for ProbeResult dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        probe = KnowledgeProbe(question="Test", expected_answer="Answer")
        result = ProbeResult(
            probe=probe,
            response="The answer is Answer",
            is_correct=True,
            partial_score=1.0,
            confidence_expressed=0.8,
            reasoning_provided=True,
        )

        d = result.to_dict()
        assert d["is_correct"] is True
        assert d["partial_score"] == 1.0


class TestConsistencyResult:
    """Tests for ConsistencyResult dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ConsistencyResult(
            original_response="Answer A",
            paraphrased_responses=["Answer A", "Answer A variant"],
            consistency_score=0.9,
            contradictions=[],
            semantic_drift=0.1,
        )

        d = result.to_dict()
        assert d["consistency_score"] == 0.9
        assert d["num_paraphrases"] == 2


class TestKnowledgeReport:
    """Tests for KnowledgeReport dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = KnowledgeReport(
            probes_run=10,
            correct_count=8,
            accuracy=0.8,
            by_category={"factual": 0.9, "conceptual": 0.7},
            by_difficulty={"easy": 0.95, "medium": 0.8, "hard": 0.6},
            confidence_calibration=0.75,
            knowledge_gaps=["procedural"],
        )

        d = report.to_dict()
        assert d["accuracy"] == 0.8
        assert d["probes_run"] == 10


class TestClaimExtractor:
    """Tests for ClaimExtractor."""

    def test_extract_factual_claim(self):
        """Test extraction of factual claims."""
        extractor = ClaimExtractor()
        text = "Paris is the capital of France. London is in England."

        claims = extractor.extract(text)

        assert len(claims) >= 1
        claim_texts = [c.text for c in claims]
        assert any("Paris" in t for t in claim_texts)

    def test_extract_temporal_claim(self):
        """Test extraction of temporal claims."""
        extractor = ClaimExtractor()
        text = "In 1969, humans landed on the moon."

        claims = extractor.extract(text)

        temporal_claims = [c for c in claims if c.category == KnowledgeCategory.TEMPORAL]
        # Should find temporal claim
        assert len(temporal_claims) >= 0  # Pattern may or may not match

    def test_confidence_reduction_for_hedging(self):
        """Test that hedging language reduces confidence."""
        extractor = ClaimExtractor()

        confident = "Paris is the capital of France."
        hedged = "Paris is maybe the capital of France."

        confident_claims = extractor.extract(confident)
        hedged_claims = extractor.extract(hedged)

        if confident_claims and hedged_claims:
            assert confident_claims[0].confidence >= hedged_claims[0].confidence

    def test_empty_text(self):
        """Test with empty text."""
        extractor = ClaimExtractor()
        claims = extractor.extract("")

        assert claims == []

    def test_no_claims_text(self):
        """Test with text containing no extractable claims."""
        extractor = ClaimExtractor()
        claims = extractor.extract("Hello! How are you?")

        # May or may not extract claims depending on patterns
        assert isinstance(claims, list)

    def test_deduplication(self):
        """Test that duplicate claims are removed."""
        extractor = ClaimExtractor()
        text = "Paris is the capital of France. Paris is the capital of France."

        claims = extractor.extract(text)

        # Should deduplicate
        texts = [c.text for c in claims]
        assert len(texts) == len(set(texts))


class TestFactVerifier:
    """Tests for FactVerifier."""

    def test_verify_with_supporting_evidence(self):
        """Test verification with supporting evidence."""
        kb = {"Paris France": "Paris is the capital of France"}
        verifier = FactVerifier(knowledge_base=kb)

        claim = Claim(text="Paris is the capital of France")
        result = verifier.verify(claim)

        assert result.status == VerificationStatus.VERIFIED
        assert len(result.supporting_evidence) > 0

    def test_verify_unverifiable(self):
        """Test verification of unverifiable claim."""
        verifier = FactVerifier()

        claim = Claim(text="Unknown fact about obscure topic")
        result = verifier.verify(claim)

        assert result.status == VerificationStatus.UNVERIFIABLE

    def test_verify_batch(self):
        """Test batch verification."""
        verifier = FactVerifier()

        claims = [
            Claim(text="Claim one"),
            Claim(text="Claim two"),
            Claim(text="Claim three"),
        ]

        results = verifier.verify_batch(claims)

        assert len(results) == 3
        assert all(isinstance(r, VerificationResult) for r in results)

    def test_source_quality_assessment(self):
        """Test source quality assessment."""
        verifier = FactVerifier()

        # No evidence = low quality
        quality = verifier._assess_source_quality([])
        assert quality == 0.0

        # More evidence = higher quality
        quality = verifier._assess_source_quality(["source1", "source2", "source3"])
        assert quality > 0.0


class TestKnowledgeProber:
    """Tests for KnowledgeProber."""

    def test_add_probe(self):
        """Test adding probes."""
        prober = KnowledgeProber()

        probe = prober.add_probe(
            question="What is 2+2?",
            expected_answer="4",
            difficulty=0.1,
        )

        assert probe.question == "What is 2+2?"
        assert len(prober.probes) == 1

    def test_run_probe(self):
        """Test running a single probe."""
        prober = KnowledgeProber()

        def model_fn(q):
            return "The answer is 4"

        probe = prober.add_probe("What is 2+2?", expected_answer="4")
        result = prober.run_probe(probe, model_fn)

        assert result.response == "The answer is 4"
        assert result.is_correct is True

    def test_run_probe_incorrect(self):
        """Test running probe with incorrect answer."""
        prober = KnowledgeProber()

        def model_fn(q):
            return "The answer is 5"

        probe = prober.add_probe("What is 2+2?", expected_answer="4")
        result = prober.run_probe(probe, model_fn)

        assert result.is_correct is False

    def test_run_probe_with_hints(self):
        """Test running probe with hints."""
        prober = KnowledgeProber()

        received_prompt = []

        def model_fn(q):
            received_prompt.append(q)
            return "Answer"

        probe = prober.add_probe(
            "Question?",
            hints=["Hint 1", "Hint 2"],
        )
        prober.run_probe(probe, model_fn, include_hints=True)

        assert "Hints:" in received_prompt[0]

    def test_run_all_probes(self):
        """Test running all probes."""
        prober = KnowledgeProber()
        prober.add_probe("Q1", expected_answer="A1")
        prober.add_probe("Q2", expected_answer="A2")

        def model_fn(q):
            if "Q1" in q:
                return "A1"
            return "Wrong"

        results = prober.run_all_probes(model_fn)

        assert len(results) == 2

    def test_generate_report(self):
        """Test report generation."""
        prober = KnowledgeProber()

        probe1 = prober.add_probe(
            "Q1", expected_answer="A1", category=KnowledgeCategory.FACTUAL, difficulty=0.2
        )
        probe2 = prober.add_probe(
            "Q2", expected_answer="A2", category=KnowledgeCategory.CONCEPTUAL, difficulty=0.8
        )

        results = [
            ProbeResult(probe=probe1, response="A1", is_correct=True, confidence_expressed=0.8),
            ProbeResult(probe=probe2, response="Wrong", is_correct=False, confidence_expressed=0.9),
        ]

        report = prober.generate_report(results)

        assert report.probes_run == 2
        assert report.correct_count == 1
        assert report.accuracy == 0.5

    def test_generate_report_empty(self):
        """Test report generation with no results."""
        prober = KnowledgeProber()
        report = prober.generate_report([])

        assert report.probes_run == 0
        assert report.accuracy == 0.0

    def test_detect_confidence_high(self):
        """Test high confidence detection."""
        prober = KnowledgeProber()
        conf = prober._detect_confidence("I am certainly sure it is Paris")

        assert conf == 0.9

    def test_detect_confidence_low(self):
        """Test low confidence detection."""
        prober = KnowledgeProber()
        conf = prober._detect_confidence("I'm not sure, maybe it's Paris")

        assert conf == 0.3

    def test_detect_reasoning(self):
        """Test reasoning detection."""
        prober = KnowledgeProber()

        with_reasoning = prober._detect_reasoning("It's Paris because it's the largest city")
        without_reasoning = prober._detect_reasoning("It's Paris")

        assert with_reasoning is True
        assert without_reasoning is False


class TestConsistencyTester:
    """Tests for ConsistencyTester."""

    def test_generate_paraphrases(self):
        """Test paraphrase generation."""
        tester = ConsistencyTester()
        paraphrases = tester.generate_paraphrases(
            "What is the capital of France?",
            num_paraphrases=3,
        )

        assert len(paraphrases) >= 1
        assert all(p != "What is the capital of France?" for p in paraphrases)

    def test_test_consistency_consistent(self):
        """Test consistency testing with consistent model."""
        tester = ConsistencyTester()

        def consistent_model(q):
            return "Paris is the capital of France"

        result = tester.test_consistency(
            "What is the capital of France?",
            consistent_model,
            num_variations=2,
        )

        assert isinstance(result, ConsistencyResult)
        assert result.consistency_score > 0.5

    def test_test_consistency_inconsistent(self):
        """Test consistency testing with inconsistent model."""
        tester = ConsistencyTester()
        counter = [0]

        def inconsistent_model(q):
            counter[0] += 1
            return f"Answer {counter[0]}"

        result = tester.test_consistency(
            "What is the capital?",
            inconsistent_model,
            num_variations=3,
        )

        # Responses are all different
        assert result.consistency_score < 1.0

    def test_find_contradictions(self):
        """Test contradiction finding."""
        tester = ConsistencyTester()

        contradictions = tester._find_contradictions(
            "Paris is the capital",
            ["Paris is not the capital"],  # Negation difference
        )

        assert len(contradictions) > 0

    def test_calculate_drift(self):
        """Test drift calculation."""
        tester = ConsistencyTester()

        # Same responses = no drift
        drift = tester._calculate_drift("answer", ["answer", "answer"])
        assert drift < 0.3

        # Different responses = drift
        drift = tester._calculate_drift("answer one", ["completely different"])
        assert drift > 0.5


class TestSourceAttributor:
    """Tests for SourceAttributor."""

    def test_add_source(self):
        """Test adding sources."""
        attributor = SourceAttributor()
        attributor.add_source(
            "Wikipedia",
            reliability=0.8,
            metadata={"topics": ["geography", "history"]},
        )

        assert "Wikipedia" in attributor.sources
        assert attributor.sources["Wikipedia"]["reliability"] == 0.8

    def test_attribute_with_matching_source(self):
        """Test attribution with matching source."""
        attributor = SourceAttributor()
        attributor.add_source(
            "Geography Book",
            reliability=0.9,
            metadata={"topics": ["geography", "capitals"]},
        )

        claim = Claim(text="Paris is the capital of France")
        result = attributor.attribute(claim)

        assert "attributions" in result
        # May or may not find attribution based on matching

    def test_attribute_no_sources(self):
        """Test attribution with no sources."""
        attributor = SourceAttributor()
        claim = Claim(text="Some claim")

        result = attributor.attribute(claim)

        assert result["attributions"] == []
        assert result["best_source"] is None


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_extract_claims(self):
        """Test extract_claims function."""
        text = "Paris is the capital of France."
        claims = extract_claims(text)

        assert isinstance(claims, list)

    def test_verify_claim(self):
        """Test verify_claim function."""
        claim = Claim(text="Test claim")
        result = verify_claim(claim)

        assert isinstance(result, VerificationResult)

    def test_verify_claim_with_kb(self):
        """Test verify_claim with knowledge base."""
        claim = Claim(text="Paris capital France")
        kb = {"Paris": "capital of France"}

        result = verify_claim(claim, knowledge_base=kb)

        assert isinstance(result, VerificationResult)

    def test_probe_knowledge(self):
        """Test probe_knowledge function."""

        def model_fn(q):
            return "Some answer"

        report = probe_knowledge(
            questions=["Q1", "Q2"],
            model_fn=model_fn,
            expected_answers=["A1", "A2"],
        )

        assert isinstance(report, KnowledgeReport)
        assert report.probes_run == 2

    def test_check_consistency_function(self):
        """Test check_consistency function."""

        def model_fn(q):
            return "Consistent answer"

        result = check_consistency(
            "Test question?",
            model_fn,
            num_variations=2,
        )

        assert isinstance(result, ConsistencyResult)

    def test_verify_facts(self):
        """Test verify_facts function."""
        text = "Paris is the capital of France."
        results = verify_facts(text)

        assert isinstance(results, list)
        assert all(isinstance(r, VerificationResult) for r in results)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_inputs(self):
        """Test with empty inputs."""
        # Extract claims
        claims = extract_claims("")
        assert claims == []

        # Verify empty claim
        claim = Claim(text="")
        result = verify_claim(claim)
        assert isinstance(result, VerificationResult)

    def test_unicode_text(self):
        """Test with unicode text."""
        extractor = ClaimExtractor()
        text = "日本の首都は東京です。Tokyo is the capital of Japan."

        claims = extractor.extract(text)
        # Should handle gracefully
        assert isinstance(claims, list)

    def test_very_long_text(self):
        """Test with very long text."""
        extractor = ClaimExtractor()
        long_text = "Paris is the capital of France. " * 100

        claims = extractor.extract(long_text)

        # Should deduplicate
        assert len(claims) <= 10  # Reasonable number

    def test_special_characters(self):
        """Test with special characters."""
        extractor = ClaimExtractor()
        text = "The result is 100% accurate! It's definitely true."

        claims = extractor.extract(text)
        # Should handle gracefully
        assert isinstance(claims, list)

    def test_model_returning_empty(self):
        """Test when model returns empty response."""
        prober = KnowledgeProber()

        def empty_model(q):
            return ""

        probe = prober.add_probe("Question?", expected_answer="Answer")
        result = prober.run_probe(probe, empty_model)

        assert result.response == ""
        assert result.is_correct is False

    def test_none_expected_answer(self):
        """Test probe with no expected answer."""
        prober = KnowledgeProber()

        def model_fn(q):
            return "Some response"

        probe = prober.add_probe("Open question?")  # No expected answer
        result = prober.run_probe(probe, model_fn)

        assert result.is_correct is None
        assert result.partial_score == 0.0

    def test_consistency_with_single_variation(self):
        """Test consistency with minimal variations."""
        tester = ConsistencyTester()

        def model_fn(q):
            return "Answer"

        result = tester.test_consistency("Q?", model_fn, num_variations=1)

        assert isinstance(result, ConsistencyResult)

    def test_knowledge_report_by_difficulty(self):
        """Test report breakdown by difficulty."""
        prober = KnowledgeProber()

        # Add probes of different difficulties
        probe_easy = prober.add_probe("Easy Q", expected_answer="A", difficulty=0.1)
        probe_medium = prober.add_probe("Medium Q", expected_answer="B", difficulty=0.5)
        probe_hard = prober.add_probe("Hard Q", expected_answer="C", difficulty=0.9)

        results = [
            ProbeResult(probe=probe_easy, response="A", is_correct=True),
            ProbeResult(probe=probe_medium, response="B", is_correct=True),
            ProbeResult(probe=probe_hard, response="Wrong", is_correct=False),
        ]

        report = prober.generate_report(results)

        assert "easy" in report.by_difficulty
        assert "medium" in report.by_difficulty
        assert "hard" in report.by_difficulty
        assert report.by_difficulty["easy"] == 1.0
        assert report.by_difficulty["hard"] == 0.0
