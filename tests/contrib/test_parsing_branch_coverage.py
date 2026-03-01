"""Additional branch coverage for parsing helpers."""

from __future__ import annotations

from insideLLMs.contrib.parsing import (
    AnswerExtractor,
    CodeBlockParser,
    JSONParser,
    ListParser,
    OutputDetector,
    ParseResult,
)


def test_parse_result_map_exception_path_preserves_error():
    result = ParseResult(success=True, value=3, raw="3")
    mapped = result.map(lambda _: 1 / 0)
    assert not mapped.success
    assert "division by zero" in (mapped.error or "")


def test_json_parser_invalid_code_block_and_pattern_errors():
    bad_code_block = "```json\n{not valid json}\n```"
    code_block_result = JSONParser.parse(bad_code_block)
    assert not code_block_result.success
    assert "No JSON found" in (code_block_result.error or "") or "JSON parse error" in (
        code_block_result.error or ""
    )

    bad_pattern = "Result: {'a': 1, 'b': oops}"
    pattern_result = JSONParser.parse(bad_pattern)
    assert not pattern_result.success
    assert pattern_result.format_detected is not None


def test_json_parser_extract_pattern_array_and_extract_all_error_tolerance():
    assert JSONParser._extract_json_pattern("Items: [1, 2, 3]") == "[1, 2, 3]"

    text = """```json
{bad json}
```
And maybe {"ok": true}
And trailing malformed [not valid]
"""
    results = JSONParser.extract_all(text)
    assert any(r.success and isinstance(r.value, dict) for r in results)


def test_code_block_parser_extract_sql_convenience():
    text = """```sql
SELECT * FROM users;
```"""
    sql_blocks = CodeBlockParser.extract_sql(text)
    assert sql_blocks == ["SELECT * FROM users;"]


def test_list_parser_parse_nested_blank_line_branch():
    text = "1. Alpha\n\n2. Beta\n  - Child"
    parsed = ListParser.parse_nested(text)
    assert parsed and parsed[0] == "Alpha"


def test_answer_extractor_none_paths_and_number_value_error():
    assert AnswerExtractor.extract_final_answer("No explicit answer present.") is None
    assert AnswerExtractor.extract_yes_no("Unclear response") is None
    assert AnswerExtractor.extract_choice("No option selected") is None
    assert AnswerExtractor.extract_number("There are 1000 items") is None


def test_output_detector_looks_like_json_invalid_json_branch():
    assert OutputDetector._looks_like_json("{invalid: json}") is False
