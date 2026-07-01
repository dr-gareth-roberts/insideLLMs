"""Regression tests for production-quality audit wave 7 fixes."""


# W7-0001 — the SafetyHallucinationIndicatorDetector percentage pattern must
# match normally-formatted percentages. The old pattern r"\b\d+(?:\.\d+)?%\b"
# ended in \b immediately after '%'; since '%' is a non-word char, that word
# boundary required a word char right after '%', so "95%" (followed by space/
# punctuation/end) never matched and percentages were silently ignored.
def test_percentage_counts_as_specific_claim():
    from insideLLMs.safety import SafetyHallucinationIndicatorDetector

    detector = SafetyHallucinationIndicatorDetector()
    # Text whose ONLY specific claim is a percentage — before the fix this
    # returned specific_claims == [] and has_specific_claims is False.
    result = detector.analyze("Roughly 95% of users prefer this option.")

    assert "95%" in result["specific_claims"]
    assert result["indicators"]["has_specific_claims"] is True


# W7-0001 — the raw pattern must match the common percentage forms (trailing
# space, punctuation, end-of-string) that the stray \b previously excluded.
def test_specific_claim_percentage_pattern_matches_common_forms():
    from insideLLMs.safety import SafetyHallucinationIndicatorDetector

    pattern = SafetyHallucinationIndicatorDetector.SPECIFIC_CLAIM_PATTERNS[0]
    assert pattern.findall("95% of users") == ["95%"]
    assert pattern.findall("growth of 12.5%.") == ["12.5%"]
    assert pattern.findall("hit 100%!") == ["100%"]


# W7-0001 — the module's own docstring example
# ("Studies show 75% of experts agree this is definitely true.") documents
# has_specific_claims == True; that intent must hold in behaviour.
def test_docstring_percentage_example_holds():
    from insideLLMs.safety import SafetyHallucinationIndicatorDetector

    detector = SafetyHallucinationIndicatorDetector()
    result = detector.analyze("Studies show 75% of experts agree this is definitely true.")
    assert result["indicators"]["has_specific_claims"] is True


# W7-0009 — a blank cell in a markdown table row must not shift the following
# values left. The old parser filtered empty cells before zip(headers, cells),
# so a blank cell silently moved every later value into the wrong column and
# dropped the last column.
def test_markdown_table_blank_cell_keeps_columns_aligned():
    from insideLLMs.structured_extraction import TableExtractor

    text = "| Name | Age | City |\n|------|-----|------|\n| Bob  |     | NYC  |\n"
    rows = TableExtractor().extract(text).extracted_data["rows"]
    # 'NYC' is a City value and must stay in City, not slide into Age.
    assert rows == [{"Name": "Bob", "Age": "", "City": "NYC"}]


# W7-0009 — rows without outer border pipes must still parse correctly.
def test_markdown_table_without_outer_pipes():
    from insideLLMs.structured_extraction import TableExtractor

    text = "Name | Age | City\nBob | 30 | NYC\n"
    rows = TableExtractor().extract(text).extracted_data["rows"]
    assert rows == [{"Name": "Bob", "Age": "30", "City": "NYC"}]
