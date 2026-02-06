"""Additional branch coverage for structured extraction utilities."""

from __future__ import annotations

import pytest

from insideLLMs.structured_extraction import (
    EntityExtractor,
    ExtractionFormat,
    ExtractionResult,
    ExtractionSchema,
    ExtractionStatus,
    FieldSchema,
    FieldType,
    JSONExtractor,
    KeyValueExtractor,
    ResponseParser,
    StructuredExtractor,
    TableExtractor,
    TypeCoercer,
)


def test_field_schema_type_validation_edge_branches():
    any_field = FieldSchema(name="x", field_type=FieldType.ANY)
    assert any_field.validate(object())[0]

    assert not FieldSchema(name="s", field_type=FieldType.STRING).validate(1)[0]
    assert not FieldSchema(name="f", field_type=FieldType.FLOAT).validate(True)[0]
    assert not FieldSchema(name="l", field_type=FieldType.LIST).validate("no-list")[0]
    assert not FieldSchema(name="d", field_type=FieldType.DICT).validate("no-dict")[0]


def test_json_extractor_balanced_braces_unmatched_returns_none():
    extractor = JSONExtractor()
    assert extractor._find_balanced_braces('{"a": 1', 0, "{", "}") is None


def test_key_value_extractor_skip_blank_lines_and_parse_float():
    extractor = KeyValueExtractor()
    result = extractor.extract("\n\nprice: 12.5\n")
    assert result.is_success
    assert result.extracted_data["price"] == 12.5


def test_entity_extractor_custom_patterns_without_defaults():
    extractor = EntityExtractor(patterns={"ticket": r"TKT-\d+"}, include_defaults=False)
    entities = extractor.extract("Contact ref TKT-1234 for details")
    assert len(entities) == 1
    assert entities[0].entity_type == "ticket"


def test_table_extractor_markdown_failure_paths():
    extractor = TableExtractor()

    too_short = extractor._extract_markdown_table("|col1|\n")
    assert too_short.status == ExtractionStatus.FAILED
    assert "No valid table found" in too_short.errors

    no_rows = extractor._extract_markdown_table("|name|age|\n|---|---|")
    assert no_rows.status == ExtractionStatus.FAILED
    assert "No data rows found in table" in no_rows.errors


def test_structured_extractor_non_autodetect_and_schema_validation_error():
    schema = ExtractionSchema(
        fields=[FieldSchema(name="required_name", field_type=FieldType.STRING)]
    )
    extractor = StructuredExtractor(schema=schema, auto_detect_format=False)

    result = extractor.extract('{"other":"value"}')
    assert result.status == ExtractionStatus.VALIDATION_ERROR
    assert any("required_name" in err for err in result.errors)


def test_structured_extractor_extract_format_dispatch_branches():
    extractor = StructuredExtractor()

    json_result = extractor._extract_format('{"a": 1}', ExtractionFormat.JSON)
    kv_result = extractor._extract_format("a: 1", ExtractionFormat.KEY_VALUE)
    list_result = extractor._extract_format("- a\n- b", ExtractionFormat.LIST)
    table_result = extractor._extract_format("|h|\n|---|\n|v|", ExtractionFormat.TABLE)
    fallback = extractor._extract_format("free text", ExtractionFormat.FREE_TEXT)

    assert json_result.format_detected == ExtractionFormat.JSON
    assert kv_result.format_detected == ExtractionFormat.KEY_VALUE
    assert list_result.format_detected == ExtractionFormat.LIST
    assert table_result.format_detected == ExtractionFormat.TABLE
    assert fallback.format_detected == ExtractionFormat.FREE_TEXT


def test_structured_extractor_auto_extract_fallback_when_all_fail(monkeypatch: pytest.MonkeyPatch):
    extractor = StructuredExtractor()

    def _raise(_: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(extractor.json_extractor, "extract", _raise)
    monkeypatch.setattr(
        extractor.table_extractor,
        "extract",
        lambda _: ExtractionResult("", {}, ExtractionStatus.FAILED, ExtractionFormat.TABLE, 0.0),
    )
    monkeypatch.setattr(
        extractor.kv_extractor,
        "extract",
        lambda _: ExtractionResult(
            "", {}, ExtractionStatus.FAILED, ExtractionFormat.KEY_VALUE, 0.0
        ),
    )
    monkeypatch.setattr(
        extractor.list_extractor,
        "extract",
        lambda _: ExtractionResult("", {}, ExtractionStatus.FAILED, ExtractionFormat.LIST, 0.0),
    )

    fallback = extractor.extract("nothing structured here")
    assert fallback.status == ExtractionStatus.PARTIAL
    assert fallback.format_detected == ExtractionFormat.FREE_TEXT
    assert fallback.extracted_data["text"] == "nothing structured here"


def test_extract_with_schema_and_extract_fields_fallback_branches():
    extractor = StructuredExtractor()
    required_schema = ExtractionSchema(
        fields=[FieldSchema(name="must_exist", field_type=FieldType.STRING, required=True)]
    )

    validated = extractor.extract_with_schema('{"other":"x"}', required_schema)
    assert validated.status == ExtractionStatus.VALIDATION_ERROR

    json_only = extractor.extract_fields('{"name":"Alice","age":30}', ["name"])
    assert json_only == {"name": "Alice"}

    none_found = extractor.extract_fields("plain text with nothing", ["name"])
    assert none_found == {}


def test_response_parser_exception_and_failure_paths():
    class _BoomExtractor:
        def extract(self, _: str):
            raise ValueError("explode")

    class _SuccessExtractor:
        def extract(self, _: str):
            return ExtractionResult(
                raw_text="x",
                extracted_data={"name": "Alice", "items": [1, 2], "extra": 3},
                status=ExtractionStatus.SUCCESS,
                format_detected=ExtractionFormat.JSON,
                confidence=0.8,
            )

    parser = ResponseParser(extractors=[_BoomExtractor(), _SuccessExtractor()])
    filtered = parser.parse("ignored", expected_fields=["name"])
    assert filtered.extracted_data == {"name": "Alice", "items": [1, 2]}

    failing = ResponseParser(extractors=[_BoomExtractor()])
    failed_result = failing.parse("still nothing")
    assert failed_result.status == ExtractionStatus.FAILED
    assert "All extraction strategies failed" in failed_result.errors


def test_type_coercer_branches():
    assert TypeCoercer.coerce(True, FieldType.BOOLEAN) == (True, True)
    assert TypeCoercer.coerce("yes", FieldType.BOOLEAN) == (True, True)
    assert TypeCoercer.coerce("off", FieldType.BOOLEAN) == (False, True)
    assert TypeCoercer.coerce(2, FieldType.BOOLEAN) == (True, True)
    assert TypeCoercer.coerce("a,b", FieldType.LIST) == (["a", "b"], True)
    assert TypeCoercer.coerce(7, FieldType.LIST) == ([7], True)
    assert TypeCoercer.coerce({"a": 1}, FieldType.DICT) == ({"a": 1}, True)
    assert TypeCoercer.coerce("x", FieldType.DICT) == ({"value": "x"}, True)
    assert TypeCoercer.coerce("x", FieldType.URL) == ("x", True)

    bad_int, ok = TypeCoercer.coerce("not-a-number", FieldType.INTEGER)
    assert bad_int == "not-a-number"
    assert not ok
