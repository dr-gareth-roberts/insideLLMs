"""Close misses to un-omit calibration, retrieval, quality, hallucination."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_calibration_remaining_edges() -> None:
    from insideLLMs.contrib.calibration import (
        CalibrationMethod,
        Calibrator,
        ConfidenceEstimator,
        HistogramBinner,
        PlattScaler,
        TemperatureScaler,
        calculate_brier_score,
    )

    with pytest.raises(ValueError):
        TemperatureScaler().fit([], [])
    with pytest.raises(ValueError):
        TemperatureScaler().fit([1.0], [1, 0])
    with pytest.raises(ValueError):
        PlattScaler().fit([], [])
    with pytest.raises(ValueError):
        HistogramBinner().fit([], [])
    with pytest.raises(ValueError):
        calculate_brier_score([], [])

    assert 0.0 < PlattScaler()._sigmoid(-5.0) < 0.5

    est = ConfidenceEstimator()
    assert est.from_logprobs([]).value == 0.0
    assert est.from_logprobs([-0.1, -0.2], aggregate="min").value > 0
    assert est.from_logprobs([-0.1, -0.2], aggregate="product").value > 0
    assert est.from_logprobs([-0.1], aggregate="other").value > 0
    assert est.from_consistency([]).value == 0.0
    assert est.from_consistency(["only"]).value == 0.5
    assert est.from_consistency(["a", "b"], similarity_fn=lambda a, b: 0.25).value == 0.25
    assert est.from_entropy([]).value == 0.0
    assert est.from_entropy([[0.0, 0.0]]).value == 0.0
    assert est.from_entropy([[1.0]]).value >= 0.0

    cal = Calibrator(CalibrationMethod.HISTOGRAM_BINNING)
    cal.fit([0.1, 0.9, 0.5, 0.5], [0, 1, 1, 0])
    assert 0.0 <= cal.calibrate(0.5) <= 1.0

    cal2 = Calibrator(CalibrationMethod.TEMPERATURE_SCALING)
    cal2.fit([0.9, 0.1], [1, 0])
    cal2.method = "not-a-real-method"  # type: ignore[assignment]
    assert cal2.calibrate(0.42) == 0.42


def test_retrieval_chunker_edges() -> None:
    from insideLLMs.contrib.retrieval import ChunkingConfig, TextChunker

    chunker = TextChunker(ChunkingConfig(chunk_size=20))
    assert chunker._split_text("hello", []) == ["hello"]
    assert chunker._split_text("abcdefghij", [".", ""])

    cfg = ChunkingConfig(chunk_size=10, chunk_overlap=3)
    assert TextChunker(cfg).chunk("Sentence one here. Sentence two here. Sentence three here.")

    cfg3 = ChunkingConfig(chunk_size=15, chunk_overlap=0)
    assert TextChunker(cfg3).chunk("Alpha beta gamma delta epsilon zeta.")

    # empty split continue (1810) via patched splits
    cfg4 = ChunkingConfig(chunk_size=50, chunk_overlap=0)
    c4 = TextChunker(cfg4)
    with patch.object(c4, "_split_text", return_value=["Hi", "", "There"]):
        assert c4.chunk("ignored")

    # oversized first split with empty current (1827)
    cfg5 = ChunkingConfig(chunk_size=5, chunk_overlap=0)
    c5 = TextChunker(cfg5)
    c5.SEPARATORS = [". ", " "]
    assert c5.chunk("HelloWorldToken. Next")

    # empty docs on Retriever.add_documents
    from insideLLMs.contrib.retrieval import Retriever

    class FakeEmbed:
        def embed_batch(self, texts):
            return [[0.1] * 3 for _ in texts]

    class FakeStore:
        def add(self, docs, embeddings):
            return None

    try:
        r = Retriever(embedding_model=FakeEmbed(), vector_store=FakeStore())
        assert r.add_documents([]) == []
    except TypeError:
        # alternate constructor
        r = Retriever.__new__(Retriever)
        r.chunker = chunker
        r.embedding_model = FakeEmbed()
        r.vector_store = FakeStore()
        assert Retriever.add_documents(r, []) == []


def test_quality_dimension_edge_branches() -> None:
    from insideLLMs.contrib.quality import (
        ClarityScorer,
        CoherenceScorer,
        CompletenessScorer,
        ConcisenessScorer,
        RelevanceScorer,
        ResponseQualityAnalyzer,
        SpecificityScorer,
    )

    # empty prompt length else (885)
    RelevanceScorer().score("", "Some response text here about the topic.")

    # multi-question addressed (1145-1146): list markers are single-char ^[-•*\d+.]\s
    CompletenessScorer().score(
        "What is A? What is B?",
        "- A is one detailed answer with enough words here today.\n"
        "- B is two detailed answer with enough words here today.\n"
        "- Extra point with enough words for length today here.",
    )
    # coherence mid start_diversity (1253): 0.5 <= diversity < 0.7 → 3/5 = 0.6
    CoherenceScorer().score(
        "q",
        "The cat sat. The dog ran. The bird flew. Dogs swim deep. Cats hide well.",
    )
    # conciseness redundant evidence (1352)
    ConcisenessScorer().score(
        "q",
        "The absolutely essential true fact and past history show the end result.",
    )
    # clarity lists (1463)
    ClarityScorer().score("q", "- item one here\n- item two here\n- item three here")

    # clarity ideal 15-25 words (1441) + long >35 (1447)
    ClarityScorer().score(
        "q",
        "This sentence has about fifteen words so it should hit the ideal range nicely today.",
    )
    ClarityScorer().score("q", " ".join(["word"] * 40) + ".")
    # specificity specific_count >= 3 (1563)
    SpecificityScorer().score(
        "q",
        "For example, Apple and Microsoft reported 12%, 34%, and 56% in 2019, 2020, and 2021.",
    )

    ResponseQualityAnalyzer().analyze("prompt?", "A solid response with several sentences here.")


def test_hallucination_report_and_detector_edges() -> None:
    from insideLLMs.contrib.hallucination import (
        ComprehensiveHallucinationDetector,
        ConsistencyCheck,
        FactualityChecker,
        FactualityScore,
        GroundednessChecker,
        HallucinationFlag,
        HallucinationReport,
        HallucinationType,
        SeverityLevel,
    )

    # long text → preview truncation (982)
    long = "A" * 250
    report = HallucinationReport(text=long, flags=[], overall_score=0.0)
    d = report.to_dict()
    assert d["text_preview"].endswith("...")

    det = ComprehensiveHallucinationDetector()
    ref = ["The capital is Paris.", "The capital is London."]
    out = det.detect(
        "The capital is Paris. Everyone knows this for sure.",
        reference_texts=ref,
        check_consistency=True,
        check_factuality=True,
    )
    assert isinstance(out, HallucinationReport)

    # fact_checker branches need extractable patterns ("X is Y" / "X was created in YYYY")
    claim_text = "Paris is the capital of France."
    fc = FactualityChecker(fact_checker=lambda claim: (False, 0.9))
    assert fc.check(claim_text).contradicted_claims >= 1
    fc2 = FactualityChecker(fact_checker=lambda claim: (True, 0.5))  # low conf → unverified
    assert fc2.check(claim_text).unverified_claims >= 1
    fc3 = FactualityChecker(fact_checker=lambda claim: (True, 0.95))
    assert fc3.check(claim_text).verified_claims >= 1
    # knowledge_base match path (1718) when no fact_checker
    fc_kb = FactualityChecker(
        knowledge_base={"paris": "Paris is the capital of France"},
    )
    assert fc_kb.check(claim_text).verified_claims >= 1

    # consistency inconsistent flag (1934) + contradicted detail flag (1955)
    det2 = ComprehensiveHallucinationDetector()
    det2.factuality_checker = FactualityChecker(fact_checker=lambda c: (False, 0.95))
    out2 = det2.detect(
        "Python is a programming language created carefully over many years.",
        reference_texts=[
            "Python is a programming language invented quickly by many people.",
        ],
        check_consistency=True,
        check_factuality=True,
    )
    assert isinstance(out2, HallucinationReport)
    assert any(
        f.hallucination_type == HallucinationType.LOGICAL_INCONSISTENCY for f in out2.flags
    ) or any(f.hallucination_type == HallucinationType.FACTUAL_ERROR for f in out2.flags)

    flags = [
        HallucinationFlag(
            hallucination_type=t,
            severity=s,
            text_span=span,
            start_pos=0,
            end_pos=1,
            confidence=0.9,
            explanation="e",
        )
        for t, s, span in [
            (HallucinationType.UNSUPPORTED_CLAIM, SeverityLevel.CRITICAL, "x"),
            (HallucinationType.UNSUPPORTED_CLAIM, SeverityLevel.HIGH, "y"),
            (HallucinationType.UNSUPPORTED_CLAIM, SeverityLevel.MEDIUM, "z"),
            (HallucinationType.EXAGGERATION, SeverityLevel.LOW, "w"),
            (HallucinationType.EXAGGERATION, SeverityLevel.LOW, "v"),
            (HallucinationType.EXAGGERATION, SeverityLevel.LOW, "u"),
        ]
    ]
    consistency = ConsistencyCheck(
        responses=["a", "b"],
        consistent_claims=[],
        inconsistent_claims=[("a", "b", 0.8)],
        overall_consistency=0.2,
    )
    factuality = FactualityScore(
        score=0.1,
        verifiable_claims=5,
        verified_claims=0,
        unverified_claims=5,
        contradicted_claims=2,
    )
    # attach property-like attrs if helper expects them
    if not hasattr(consistency, "is_consistent"):
        consistency.is_consistent = False  # type: ignore[attr-defined]
    recs = det._generate_recommendations(flags, consistency, factuality)
    assert any("critical" in r.lower() or "high" in r.lower() for r in recs)

    gc = GroundednessChecker(threshold=0.1)
    result = gc.check(
        "Hi. This longer sentence about cats being mammals is grounded here.",
        "Cats are mammals. Dogs are animals.",
    )
    assert isinstance(result, dict)
