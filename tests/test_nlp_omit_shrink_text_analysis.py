"""Un-omit path: text_analysis remaining edges."""

from __future__ import annotations

from insideLLMs.nlp.text_analysis import (
    ResponseQualityScore,
    TextAnalyzer,
    analyze_text,
    compare_responses,
    score_response,
)


def test_text_analysis_empty_and_quality_edges() -> None:
    analyzer = TextAnalyzer()
    # empty content / tone (967, 1092)
    content = analyzer.analyze_content("")
    assert content.question_count == 0
    tone = analyzer.analyze_tone("...")  # no word tokens → early ToneAnalysis()
    assert tone.formality_score == 0.0

    result = analyze_text("What is Python? It is great. Visit https://x.com")
    assert "profile" in result

    # ResponseQualityScore length branches (1525 short, 1527 long, 1535 very long)
    short = ResponseQualityScore.calculate(
        "tiny", "Explain everything in detail", expected_length=100
    )
    assert short.completeness < 1.0

    longish = ResponseQualityScore.calculate("word " * 250, "Explain briefly", expected_length=50)
    assert longish.completeness <= 1.0

    very_long = ResponseQualityScore.calculate(
        "word " * 1200, "Say something", expected_length=None
    )
    assert very_long.completeness <= 1.0

    scored = score_response(
        "Deep learning uses neural networks.",
        "What is deep learning?",
        required_keywords=["deep", "learning", "neural"],
        expected_length=20,
    )
    assert scored.overall >= 0

    cmp = compare_responses("Answer A is solid.", "Answer B is also fine.", "Q?")
    assert "response1_score" in cmp and "winner" in cmp
