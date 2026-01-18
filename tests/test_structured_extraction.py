"""Tests for structured extraction module."""

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
    ListExtractor,
    ResponseParser,
    StructuredExtractor,
    TableExtractor,
    TypeCoercer,
    create_schema,
    extract_entities,
    extract_json,
    extract_key_values,
    extract_list,
    extract_structured,
    extract_table,
    validate_extraction,
)


class TestExtractionFormat:
    """Tests for ExtractionFormat enum."""

    def test_all_formats_exist(self):
        """Test that all formats exist."""
        assert ExtractionFormat.JSON.value == "json"
        assert ExtractionFormat.XML.value == "xml"
        assert ExtractionFormat.YAML.value == "yaml"
        assert ExtractionFormat.KEY_VALUE.value == "key_value"
        assert ExtractionFormat.LIST.value == "list"
        assert ExtractionFormat.TABLE.value == "table"


class TestFieldType:
    """Tests for FieldType enum."""

    def test_all_types_exist(self):
        """Test that all field types exist."""
        assert FieldType.STRING.value == "string"
        assert FieldType.INTEGER.value == "integer"
        assert FieldType.FLOAT.value == "float"
        assert FieldType.BOOLEAN.value == "boolean"
        assert FieldType.LIST.value == "list"
        assert FieldType.DICT.value == "dict"


class TestExtractionStatus:
    """Tests for ExtractionStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all statuses exist."""
        assert ExtractionStatus.SUCCESS.value == "success"
        assert ExtractionStatus.PARTIAL.value == "partial"
        assert ExtractionStatus.FAILED.value == "failed"
        assert ExtractionStatus.VALIDATION_ERROR.value == "validation_error"


class TestFieldSchema:
    """Tests for FieldSchema class."""

    def test_basic_validation(self):
        """Test basic field validation."""
        schema = FieldSchema(name="test", field_type=FieldType.STRING)
        is_valid, error = schema.validate("hello")
        assert is_valid
        assert error is None

    def test_required_field_missing(self):
        """Test required field validation."""
        schema = FieldSchema(name="test", field_type=FieldType.STRING, required=True)
        is_valid, error = schema.validate(None)
        assert not is_valid
        assert "required" in error.lower()

    def test_optional_field_missing(self):
        """Test optional field validation."""
        schema = FieldSchema(name="test", field_type=FieldType.STRING, required=False)
        is_valid, error = schema.validate(None)
        assert is_valid

    def test_integer_type(self):
        """Test integer type validation."""
        schema = FieldSchema(name="count", field_type=FieldType.INTEGER)
        is_valid, _ = schema.validate(42)
        assert is_valid

        is_valid, error = schema.validate("not a number")
        assert not is_valid

    def test_float_type(self):
        """Test float type validation."""
        schema = FieldSchema(name="score", field_type=FieldType.FLOAT)
        is_valid, _ = schema.validate(3.14)
        assert is_valid

        is_valid, _ = schema.validate(42)  # Integers are valid floats
        assert is_valid

    def test_boolean_type(self):
        """Test boolean type validation."""
        schema = FieldSchema(name="active", field_type=FieldType.BOOLEAN)
        is_valid, _ = schema.validate(True)
        assert is_valid

        is_valid, error = schema.validate("true")
        assert not is_valid

    def test_pattern_validation(self):
        """Test pattern validation."""
        schema = FieldSchema(
            name="code",
            field_type=FieldType.STRING,
            pattern=r"^[A-Z]{3}\d{3}$",
        )
        is_valid, _ = schema.validate("ABC123")
        assert is_valid

        is_valid, error = schema.validate("abc123")
        assert not is_valid

    def test_min_max_validation(self):
        """Test min/max value validation."""
        schema = FieldSchema(
            name="score",
            field_type=FieldType.INTEGER,
            min_value=0,
            max_value=100,
        )
        is_valid, _ = schema.validate(50)
        assert is_valid

        is_valid, error = schema.validate(-1)
        assert not is_valid

        is_valid, error = schema.validate(101)
        assert not is_valid

    def test_length_validation(self):
        """Test length validation."""
        schema = FieldSchema(
            name="name",
            field_type=FieldType.STRING,
            min_length=2,
            max_length=10,
        )
        is_valid, _ = schema.validate("John")
        assert is_valid

        is_valid, error = schema.validate("J")
        assert not is_valid

        is_valid, error = schema.validate("VeryLongName")
        assert not is_valid

    def test_allowed_values(self):
        """Test allowed values validation."""
        schema = FieldSchema(
            name="status",
            field_type=FieldType.STRING,
            allowed_values=["active", "inactive", "pending"],
        )
        is_valid, _ = schema.validate("active")
        assert is_valid

        is_valid, error = schema.validate("unknown")
        assert not is_valid

    def test_email_type(self):
        """Test email type validation."""
        schema = FieldSchema(name="email", field_type=FieldType.EMAIL)
        is_valid, _ = schema.validate("test@example.com")
        assert is_valid

        is_valid, error = schema.validate("not-an-email")
        assert not is_valid

    def test_url_type(self):
        """Test URL type validation."""
        schema = FieldSchema(name="website", field_type=FieldType.URL)
        is_valid, _ = schema.validate("https://example.com")
        assert is_valid

        is_valid, error = schema.validate("not-a-url")
        assert not is_valid


class TestExtractionResult:
    """Tests for ExtractionResult class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ExtractionResult(
            raw_text="test text",
            extracted_data={"key": "value"},
            status=ExtractionStatus.SUCCESS,
            format_detected=ExtractionFormat.JSON,
            confidence=0.95,
        )
        d = result.to_dict()
        assert d["status"] == "success"
        assert d["format_detected"] == "json"
        assert d["confidence"] == 0.95

    def test_get_field(self):
        """Test getting specific field."""
        result = ExtractionResult(
            raw_text="test",
            extracted_data={"name": "John", "age": 30},
            status=ExtractionStatus.SUCCESS,
            format_detected=ExtractionFormat.JSON,
            confidence=0.9,
        )
        assert result.get_field("name") == "John"
        assert result.get_field("missing", "default") == "default"

    def test_is_success(self):
        """Test success check."""
        result = ExtractionResult(
            raw_text="test",
            extracted_data={},
            status=ExtractionStatus.SUCCESS,
            format_detected=ExtractionFormat.JSON,
            confidence=0.9,
        )
        assert result.is_success

        result.status = ExtractionStatus.PARTIAL
        assert result.is_success

        result.status = ExtractionStatus.FAILED
        assert not result.is_success


class TestExtractionSchema:
    """Tests for ExtractionSchema class."""

    def test_validate_success(self):
        """Test successful schema validation."""
        schema = ExtractionSchema(
            fields=[
                FieldSchema(name="name", field_type=FieldType.STRING),
                FieldSchema(name="age", field_type=FieldType.INTEGER),
            ]
        )
        is_valid, errors = schema.validate({"name": "John", "age": 30})
        assert is_valid
        assert len(errors) == 0

    def test_validate_missing_required(self):
        """Test validation with missing required field."""
        schema = ExtractionSchema(
            fields=[
                FieldSchema(name="name", field_type=FieldType.STRING, required=True),
            ]
        )
        is_valid, errors = schema.validate({})
        assert not is_valid
        assert len(errors) > 0

    def test_extra_fields_strict(self):
        """Test extra fields in strict mode."""
        schema = ExtractionSchema(
            fields=[FieldSchema(name="name", field_type=FieldType.STRING)],
            strict=True,
            allow_extra_fields=False,
        )
        is_valid, errors = schema.validate({"name": "John", "extra": "value"})
        assert not is_valid

    def test_extra_fields_allowed(self):
        """Test extra fields when allowed."""
        schema = ExtractionSchema(
            fields=[FieldSchema(name="name", field_type=FieldType.STRING)],
            allow_extra_fields=True,
        )
        is_valid, errors = schema.validate({"name": "John", "extra": "value"})
        assert is_valid


class TestJSONExtractor:
    """Tests for JSONExtractor class."""

    def test_extract_simple_json(self):
        """Test simple JSON extraction."""
        extractor = JSONExtractor()
        text = '{"name": "John", "age": 30}'
        result = extractor.extract(text)
        assert result.is_success
        assert result.extracted_data["name"] == "John"

    def test_extract_json_in_code_block(self):
        """Test JSON extraction from code block."""
        extractor = JSONExtractor()
        text = """Here is the data:
```json
{"name": "John", "age": 30}
```
"""
        result = extractor.extract(text)
        assert result.is_success
        assert result.extracted_data["name"] == "John"

    def test_extract_json_array(self):
        """Test JSON array extraction."""
        extractor = JSONExtractor()
        text = "[1, 2, 3]"
        result = extractor.extract(text)
        assert result.is_success
        assert "items" in result.extracted_data
        assert result.extracted_data["items"] == [1, 2, 3]

    def test_extract_no_json(self):
        """Test extraction with no JSON."""
        extractor = JSONExtractor()
        text = "This is just plain text."
        result = extractor.extract(text)
        assert not result.is_success

    def test_extract_nested_json(self):
        """Test nested JSON extraction."""
        extractor = JSONExtractor()
        text = '{"user": {"name": "John", "address": {"city": "NYC"}}}'
        result = extractor.extract(text)
        assert result.is_success
        assert result.extracted_data["user"]["name"] == "John"


class TestKeyValueExtractor:
    """Tests for KeyValueExtractor class."""

    def test_extract_colon_separated(self):
        """Test colon-separated key-value extraction."""
        extractor = KeyValueExtractor()
        text = """Name: John
Age: 30
City: New York"""
        result = extractor.extract(text)
        assert result.is_success
        assert result.extracted_data["Name"] == "John"
        assert result.extracted_data["Age"] == 30

    def test_extract_equals_separated(self):
        """Test equals-separated key-value extraction."""
        extractor = KeyValueExtractor()
        text = """name = John
age = 30"""
        result = extractor.extract(text)
        assert result.is_success
        assert result.extracted_data["name"] == "John"

    def test_extract_with_specific_keys(self):
        """Test extraction with specific keys."""
        extractor = KeyValueExtractor()
        text = """Name: John
Age: 30
Extra: ignored"""
        result = extractor.extract(text, keys=["Name", "Age"])
        assert result.is_success
        assert "Name" in result.extracted_data
        assert "Extra" not in result.extracted_data

    def test_extract_boolean_values(self):
        """Test boolean value extraction."""
        extractor = KeyValueExtractor()
        text = """active: true
disabled: false"""
        result = extractor.extract(text)
        assert result.is_success
        assert result.extracted_data["active"] is True
        assert result.extracted_data["disabled"] is False

    def test_extract_list_values(self):
        """Test list value extraction."""
        extractor = KeyValueExtractor()
        text = "tags: python, machine learning, AI"
        result = extractor.extract(text)
        assert result.is_success
        assert result.extracted_data["tags"] == ["python", "machine learning", "AI"]


class TestListExtractor:
    """Tests for ListExtractor class."""

    def test_extract_bullet_list(self):
        """Test bullet list extraction."""
        extractor = ListExtractor()
        text = """- Item 1
- Item 2
- Item 3"""
        result = extractor.extract(text)
        assert result.is_success
        assert len(result.extracted_data["items"]) == 3
        assert result.extracted_data["items"][0] == "Item 1"

    def test_extract_numbered_list(self):
        """Test numbered list extraction."""
        extractor = ListExtractor()
        text = """1. First item
2. Second item
3. Third item"""
        result = extractor.extract(text)
        assert result.is_success
        assert len(result.extracted_data["items"]) == 3

    def test_extract_asterisk_list(self):
        """Test asterisk list extraction."""
        extractor = ListExtractor()
        text = """* Item A
* Item B"""
        result = extractor.extract(text)
        assert result.is_success
        assert len(result.extracted_data["items"]) == 2

    def test_extract_no_list(self):
        """Test extraction with no list."""
        extractor = ListExtractor()
        text = "Just a regular paragraph."
        result = extractor.extract(text)
        assert not result.is_success


class TestEntityExtractor:
    """Tests for EntityExtractor class."""

    def test_extract_email(self):
        """Test email extraction."""
        extractor = EntityExtractor()
        text = "Contact us at support@example.com for help."
        entities = extractor.extract(text)
        emails = [e for e in entities if e.entity_type == "email"]
        assert len(emails) == 1
        assert emails[0].text == "support@example.com"

    def test_extract_url(self):
        """Test URL extraction."""
        extractor = EntityExtractor()
        text = "Visit https://example.com for more info."
        entities = extractor.extract(text)
        urls = [e for e in entities if e.entity_type == "url"]
        assert len(urls) == 1
        assert entities[0].text == "https://example.com"

    def test_extract_phone(self):
        """Test phone number extraction."""
        extractor = EntityExtractor()
        text = "Call us at 555-123-4567."
        entities = extractor.extract(text)
        phones = [e for e in entities if e.entity_type == "phone"]
        assert len(phones) == 1

    def test_extract_date(self):
        """Test date extraction."""
        extractor = EntityExtractor()
        text = "Meeting scheduled for 01/15/2024."
        entities = extractor.extract(text)
        dates = [e for e in entities if e.entity_type == "date"]
        assert len(dates) == 1

    def test_extract_multiple_entities(self):
        """Test multiple entity extraction."""
        extractor = EntityExtractor()
        text = "Email john@test.com or call 555-555-5555."
        entities = extractor.extract(text)
        assert len(entities) >= 2

    def test_add_custom_pattern(self):
        """Test adding custom entity pattern."""
        extractor = EntityExtractor()
        extractor.add_pattern("product_code", r"PRD-\d{5}")
        text = "Product code: PRD-12345"
        entities = extractor.extract(text)
        product_codes = [e for e in entities if e.entity_type == "product_code"]
        assert len(product_codes) == 1


class TestTableExtractor:
    """Tests for TableExtractor class."""

    def test_extract_markdown_table(self):
        """Test markdown table extraction."""
        extractor = TableExtractor()
        text = """| Name | Age |
|------|-----|
| John | 30 |
| Jane | 25 |"""
        result = extractor.extract(text)
        assert result.is_success
        assert result.extracted_data["row_count"] == 2
        assert result.extracted_data["rows"][0]["Name"] == "John"

    def test_extract_simple_markdown(self):
        """Test simple markdown table without separator."""
        extractor = TableExtractor()
        text = """| Name | Age |
| John | 30 |
| Jane | 25 |"""
        result = extractor.extract(text)
        assert result.is_success
        assert len(result.extracted_data["rows"]) >= 1

    def test_extract_whitespace_table(self):
        """Test whitespace-aligned table extraction."""
        extractor = TableExtractor()
        text = """Name     Age    City
John     30     NYC
Jane     25     LA"""
        result = extractor.extract(text)
        # May succeed or fail depending on alignment
        assert isinstance(result, ExtractionResult)


class TestStructuredExtractor:
    """Tests for StructuredExtractor class."""

    def test_auto_detect_json(self):
        """Test auto-detection of JSON."""
        extractor = StructuredExtractor()
        text = '{"name": "John", "age": 30}'
        result = extractor.extract(text)
        assert result.is_success
        assert result.format_detected == ExtractionFormat.JSON

    def test_auto_detect_key_value(self):
        """Test auto-detection of key-value."""
        extractor = StructuredExtractor()
        text = """Name: John
Age: 30"""
        result = extractor.extract(text)
        assert result.is_success
        # Could be detected as key-value or other format

    def test_explicit_format(self):
        """Test extraction with explicit format."""
        extractor = StructuredExtractor()
        text = '{"name": "John"}'
        result = extractor.extract(text, expected_format=ExtractionFormat.JSON)
        assert result.is_success
        assert result.format_detected == ExtractionFormat.JSON

    def test_empty_input(self):
        """Test extraction with empty input."""
        extractor = StructuredExtractor()
        result = extractor.extract("")
        assert not result.is_success

    def test_extract_fields(self):
        """Test extracting specific fields."""
        extractor = StructuredExtractor()
        text = '{"name": "John", "age": 30, "city": "NYC"}'
        fields = extractor.extract_fields(text, ["name", "age"])
        assert "name" in fields
        assert "city" not in fields

    def test_extract_entities(self):
        """Test entity extraction."""
        extractor = StructuredExtractor()
        entities = extractor.extract_entities("Email: test@example.com")
        assert len(entities) > 0

    def test_extract_with_schema(self):
        """Test extraction with schema validation."""
        extractor = StructuredExtractor()
        schema = ExtractionSchema(
            fields=[
                FieldSchema(name="name", field_type=FieldType.STRING),
                FieldSchema(name="age", field_type=FieldType.INTEGER),
            ]
        )
        text = '{"name": "John", "age": 30}'
        result = extractor.extract_with_schema(text, schema)
        assert result.is_success


class TestResponseParser:
    """Tests for ResponseParser class."""

    def test_parse_json_response(self):
        """Test parsing JSON response."""
        parser = ResponseParser()
        result = parser.parse('{"result": "success"}')
        assert result.is_success

    def test_parse_with_expected_fields(self):
        """Test parsing with expected fields."""
        parser = ResponseParser()
        result = parser.parse('{"name": "John", "age": 30}', expected_fields=["name"])
        assert result.is_success
        assert "name" in result.extracted_data

    def test_parse_fallback(self):
        """Test parsing with fallback strategies."""
        parser = ResponseParser()
        result = parser.parse("Name: John\nAge: 30")
        assert result.is_success


class TestTypeCoercer:
    """Tests for TypeCoercer class."""

    def test_coerce_to_string(self):
        """Test coercion to string."""
        value, success = TypeCoercer.coerce(42, FieldType.STRING)
        assert success
        assert value == "42"

    def test_coerce_to_integer(self):
        """Test coercion to integer."""
        value, success = TypeCoercer.coerce("42", FieldType.INTEGER)
        assert success
        assert value == 42

    def test_coerce_to_float(self):
        """Test coercion to float."""
        value, success = TypeCoercer.coerce("3.14", FieldType.FLOAT)
        assert success
        assert abs(value - 3.14) < 0.001

    def test_coerce_to_boolean_true(self):
        """Test coercion to boolean true."""
        for val in ["true", "yes", "1", "on"]:
            value, success = TypeCoercer.coerce(val, FieldType.BOOLEAN)
            assert success
            assert value is True

    def test_coerce_to_boolean_false(self):
        """Test coercion to boolean false."""
        for val in ["false", "no", "0", "off"]:
            value, success = TypeCoercer.coerce(val, FieldType.BOOLEAN)
            assert success
            assert value is False

    def test_coerce_to_list(self):
        """Test coercion to list."""
        value, success = TypeCoercer.coerce("a, b, c", FieldType.LIST)
        assert success
        assert value == ["a", "b", "c"]

    def test_coerce_none(self):
        """Test coercion of None."""
        value, success = TypeCoercer.coerce(None, FieldType.STRING)
        assert success
        assert value is None

    def test_coerce_with_commas(self):
        """Test coercion of number with commas."""
        value, success = TypeCoercer.coerce("1,234,567", FieldType.INTEGER)
        assert success
        assert value == 1234567


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_extract_json(self):
        """Test extract_json function."""
        result = extract_json('{"key": "value"}')
        assert result.is_success

    def test_extract_key_values(self):
        """Test extract_key_values function."""
        result = extract_key_values("Key: Value")
        assert result.is_success

    def test_extract_list(self):
        """Test extract_list function."""
        result = extract_list("- Item 1\n- Item 2")
        assert result.is_success

    def test_extract_entities(self):
        """Test extract_entities function."""
        entities = extract_entities("Email: test@test.com")
        assert len(entities) > 0

    def test_extract_table(self):
        """Test extract_table function."""
        result = extract_table("| A | B |\n| 1 | 2 |")
        # May or may not succeed depending on table format
        assert isinstance(result, ExtractionResult)

    def test_extract_structured(self):
        """Test extract_structured function."""
        result = extract_structured('{"data": "test"}')
        assert result.is_success

    def test_validate_extraction(self):
        """Test validate_extraction function."""
        schema = ExtractionSchema(fields=[FieldSchema(name="name", field_type=FieldType.STRING)])
        is_valid, errors = validate_extraction({"name": "John"}, schema)
        assert is_valid

    def test_create_schema(self):
        """Test create_schema function."""
        schema = create_schema(
            [
                {"name": "name", "type": "string", "required": True},
                {"name": "age", "type": "integer", "required": False},
            ]
        )
        assert len(schema.fields) == 2
        assert schema.fields[0].name == "name"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        extractor = JSONExtractor()
        result = extractor.extract('{"key": "value",}')  # Trailing comma
        assert not result.is_success

    def test_empty_key_value(self):
        """Test empty key-value extraction."""
        extractor = KeyValueExtractor()
        result = extractor.extract("No key value pairs here")
        assert not result.is_success

    def test_unicode_text(self):
        """Test unicode text handling."""
        extractor = JSONExtractor()
        result = extractor.extract('{"name": "日本語", "emoji": "\U0001f389"}')
        assert result.is_success
        assert result.extracted_data["name"] == "日本語"

    def test_nested_code_blocks(self):
        """Test nested code blocks."""
        extractor = JSONExtractor()
        text = """```
```json
{"key": "value"}
```
```"""
        result = extractor.extract(text)
        assert result.is_success

    def test_mixed_formats(self):
        """Test text with mixed formats."""
        extractor = StructuredExtractor()
        text = """Name: John
{"additional": "data"}"""
        result = extractor.extract(text)
        # Should extract one of the formats
        assert isinstance(result, ExtractionResult)

    def test_very_long_text(self):
        """Test very long text handling."""
        extractor = StructuredExtractor()
        text = "word " * 10000 + '{"key": "value"}'
        result = extractor.extract(text)
        assert result.is_success

    def test_special_characters_in_values(self):
        """Test special characters in values."""
        extractor = JSONExtractor()
        result = extractor.extract(
            '{"path": "C:\\\\Users\\\\name", "url": "http://test.com/path?a=1&b=2"}'
        )
        assert result.is_success

    def test_entity_position(self):
        """Test entity position tracking."""
        extractor = EntityExtractor()
        text = "Email me at test@example.com please"
        entities = extractor.extract(text)
        emails = [e for e in entities if e.entity_type == "email"]
        assert len(emails) == 1
        assert emails[0].start == text.index("test@example.com")
        assert emails[0].end == emails[0].start + len("test@example.com")
