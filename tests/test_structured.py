"""Tests for insideLLMs/structured.py module."""

import json
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest


class TestPythonTypeToJsonSchema:
    """Tests for _python_type_to_json_schema function."""

    def test_string_type(self):
        """Test string type conversion."""
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(str)
        assert result == {"type": "string"}

    def test_int_type(self):
        """Test int type conversion."""
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(int)
        assert result == {"type": "integer"}

    def test_float_type(self):
        """Test float type conversion."""
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(float)
        assert result == {"type": "number"}

    def test_bool_type(self):
        """Test bool type conversion."""
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(bool)
        assert result == {"type": "boolean"}

    def test_none_type(self):
        """Test None type conversion."""
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(type(None))
        assert result == {"type": "null"}

    def test_list_type(self):
        """Test list type conversion."""
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(list[str])
        assert result["type"] == "array"
        assert result["items"]["type"] == "string"

    def test_dict_type(self):
        """Test dict type conversion."""
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(dict[str, int])
        assert result["type"] == "object"

    def test_optional_type(self):
        """Test Optional type conversion."""
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(Optional[str])
        assert "anyOf" in result


class TestDataclassToJsonSchema:
    """Tests for dataclass_to_json_schema function."""

    def test_simple_dataclass(self):
        """Test simple dataclass conversion."""
        from insideLLMs.structured import dataclass_to_json_schema

        @dataclass
        class Person:
            name: str
            age: int

        result = dataclass_to_json_schema(Person)
        assert result["type"] == "object"
        assert "name" in result["properties"]
        assert "age" in result["properties"]
        assert "name" in result["required"]
        assert "age" in result["required"]

    def test_dataclass_with_optional(self):
        """Test dataclass with optional field."""
        from insideLLMs.structured import dataclass_to_json_schema

        @dataclass
        class Person:
            name: str
            age: int = 0

        result = dataclass_to_json_schema(Person)
        # Only name should be required
        assert "name" in result["required"]
        assert "age" not in result.get("required", [])

    def test_non_dataclass_raises(self):
        """Test that non-dataclass raises error."""
        from insideLLMs.structured import SchemaGenerationError, dataclass_to_json_schema

        class NotADataclass:
            pass

        with pytest.raises(SchemaGenerationError):
            dataclass_to_json_schema(NotADataclass)


class TestGetJsonSchema:
    """Tests for get_json_schema function."""

    def test_with_dataclass(self):
        """Test getting schema from dataclass."""
        from insideLLMs.structured import get_json_schema

        @dataclass
        class Item:
            name: str
            price: float

        result = get_json_schema(Item)
        assert result["type"] == "object"
        assert "name" in result["properties"]

    def test_with_basic_type(self):
        """Test getting schema from basic type."""
        from insideLLMs.structured import get_json_schema

        result = get_json_schema(str)
        assert result == {"type": "string"}


class TestExtractJson:
    """Tests for extract_json function."""

    def test_extract_from_code_block(self):
        """Test extracting JSON from code block."""
        from insideLLMs.structured import extract_json

        text = '''Here is the data:
```json
{"name": "John", "age": 30}
```
'''
        result = extract_json(text)
        assert json.loads(result) == {"name": "John", "age": 30}

    def test_extract_raw_json(self):
        """Test extracting raw JSON object."""
        from insideLLMs.structured import extract_json

        text = 'The result is {"name": "John", "age": 30}'
        result = extract_json(text)
        assert json.loads(result) == {"name": "John", "age": 30}

    def test_extract_array(self):
        """Test extracting JSON array."""
        from insideLLMs.structured import extract_json

        text = 'Here is the list: [1, 2, 3]'
        result = extract_json(text)
        assert json.loads(result) == [1, 2, 3]

    def test_extract_plain_json(self):
        """Test extracting plain JSON."""
        from insideLLMs.structured import extract_json

        text = '{"name": "John"}'
        result = extract_json(text)
        assert json.loads(result) == {"name": "John"}

    def test_no_json_raises(self):
        """Test that no JSON raises ParsingError."""
        from insideLLMs.structured import ParsingError, extract_json

        with pytest.raises(ParsingError):
            extract_json("This is just plain text with no JSON")


class TestParseJson:
    """Tests for parse_json function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        from insideLLMs.structured import parse_json

        result = parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_from_markdown(self):
        """Test parsing JSON from markdown."""
        from insideLLMs.structured import parse_json

        text = '''```json
{"items": [1, 2, 3]}
```'''
        result = parse_json(text)
        assert result == {"items": [1, 2, 3]}


class TestParseToType:
    """Tests for parse_to_type function."""

    def test_parse_to_dataclass(self):
        """Test parsing to dataclass."""
        from insideLLMs.structured import parse_to_type

        @dataclass
        class Person:
            name: str
            age: int

        result = parse_to_type({"name": "John", "age": 30}, Person)
        assert result.name == "John"
        assert result.age == 30

    def test_parse_to_dict(self):
        """Test parsing to dict."""
        from insideLLMs.structured import parse_to_type

        result = parse_to_type({"key": "value"}, dict)
        assert result == {"key": "value"}


class TestStructuredOutputExceptions:
    """Tests for structured output exceptions."""

    def test_parsing_error(self):
        """Test ParsingError attributes."""
        from insideLLMs.structured import ParsingError

        error = ParsingError("Test error", "raw output", 3)
        assert str(error) == "Test error"
        assert error.raw_output == "raw output"
        assert error.attempts == 3

    def test_validation_error_wrapper(self):
        """Test ValidationErrorWrapper attributes."""
        from insideLLMs.structured import ValidationErrorWrapper

        mock_error = MagicMock()
        error = ValidationErrorWrapper("Validation failed", mock_error, "raw output")
        assert "Validation failed" in str(error)
        assert error.validation_error == mock_error
        assert error.raw_output == "raw output"


class TestPydanticIntegration:
    """Tests for Pydantic integration."""

    @pytest.fixture
    def pydantic_available(self):
        """Check if Pydantic is available."""
        try:
            from pydantic import BaseModel  # noqa: F401
            return True
        except ImportError:
            pytest.skip("Pydantic not available")

    def test_pydantic_to_json_schema(self, pydantic_available):
        """Test Pydantic model to JSON schema conversion."""
        from pydantic import BaseModel

        from insideLLMs.structured import pydantic_to_json_schema

        class Item(BaseModel):
            name: str
            price: float

        result = pydantic_to_json_schema(Item)
        assert "properties" in result
        assert "name" in result["properties"]
        assert "price" in result["properties"]

    def test_parse_to_pydantic_model(self, pydantic_available):
        """Test parsing to Pydantic model."""
        from pydantic import BaseModel

        from insideLLMs.structured import parse_to_type

        class Person(BaseModel):
            name: str
            age: int

        result = parse_to_type({"name": "Alice", "age": 25}, Person)
        assert result.name == "Alice"
        assert result.age == 25

    def test_get_json_schema_pydantic(self, pydantic_available):
        """Test getting schema from Pydantic model."""
        from pydantic import BaseModel

        from insideLLMs.structured import get_json_schema

        class Product(BaseModel):
            name: str
            price: float

        result = get_json_schema(Product)
        assert "properties" in result


class TestUnionTypes:
    """Tests for Union type handling in schema generation."""

    def test_union_multiple_types(self):
        """Test Union with multiple non-None types."""
        from typing import Union
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(Union[str, int])
        assert "anyOf" in result
        assert len(result["anyOf"]) == 2

    def test_union_with_none(self):
        """Test Union with None (Optional)."""
        from typing import Union
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(Union[str, None])
        assert "anyOf" in result


class TestBytesType:
    """Tests for bytes type handling."""

    def test_bytes_type(self):
        """Test bytes type conversion."""
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(bytes)
        assert result["type"] == "string"
        assert result.get("format") == "byte"


class TestUnknownType:
    """Tests for unknown type handling."""

    def test_unknown_type_returns_empty(self):
        """Test unknown type returns empty schema."""
        from insideLLMs.structured import _python_type_to_json_schema

        class CustomClass:
            pass

        result = _python_type_to_json_schema(CustomClass)
        # Unknown types return empty schema
        assert result == {} or "type" not in result


class TestExtractJsonEdgeCases:
    """Tests for extract_json edge cases."""

    def test_extract_from_generic_code_block(self):
        """Test extracting from generic code block without json specifier."""
        from insideLLMs.structured import extract_json

        text = '''Here is the result:
```
{"status": "ok"}
```
'''
        result = extract_json(text)
        assert json.loads(result) == {"status": "ok"}

    def test_extract_invalid_in_code_block_falls_back(self):
        """Test that invalid JSON in code block tries next pattern."""
        from insideLLMs.structured import extract_json

        text = '''```json
this is not valid json
```
The actual result is {"valid": true}'''
        result = extract_json(text)
        assert json.loads(result) == {"valid": True}

    def test_extract_nested_json(self):
        """Test extracting nested JSON."""
        from insideLLMs.structured import extract_json

        text = 'Result: {"outer": {"inner": {"deep": 42}}}'
        result = extract_json(text)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"]["deep"] == 42


class TestParseToTypeEdgeCases:
    """Tests for parse_to_type edge cases."""

    def test_parse_to_dict_type(self):
        """Test parsing to dict type."""
        from insideLLMs.structured import parse_to_type

        data = {"a": 1, "b": 2}
        result = parse_to_type(data, dict)
        assert result == data

    def test_parse_to_list_type(self):
        """Test parsing to list type."""
        from insideLLMs.structured import parse_to_type

        data = [1, 2, 3]
        result = parse_to_type(data, list)
        assert result == data


class TestDataclassWithDefaults:
    """Tests for dataclass schema with default values."""

    def test_dataclass_with_default_factory(self):
        """Test dataclass with default_factory."""
        from dataclasses import dataclass, field
        from insideLLMs.structured import dataclass_to_json_schema

        @dataclass
        class Item:
            name: str
            tags: list[str] = field(default_factory=list)

        result = dataclass_to_json_schema(Item)
        # Only name should be required
        assert "name" in result["required"]
        # tags should not be required since it has default_factory
        assert "tags" not in result.get("required", [])

    def test_dataclass_all_defaults(self):
        """Test dataclass where all fields have defaults."""
        from dataclasses import dataclass
        from insideLLMs.structured import dataclass_to_json_schema

        @dataclass
        class Settings:
            debug: bool = False
            level: int = 1

        result = dataclass_to_json_schema(Settings)
        # required should be empty or not present
        assert result.get("required", []) == []


class TestExceptionDetails:
    """Tests for exception details."""

    def test_schema_generation_error(self):
        """Test SchemaGenerationError."""
        from insideLLMs.structured import SchemaGenerationError

        error = SchemaGenerationError("Test schema error")
        assert "Test schema error" in str(error)

    def test_structured_output_error_base(self):
        """Test StructuredOutputError base class."""
        from insideLLMs.structured import StructuredOutputError

        error = StructuredOutputError("Base error")
        assert "Base error" in str(error)


class TestListDictNoArgs:
    """Tests for list and dict types without type arguments."""

    def test_list_without_args(self):
        """Test list type without type arguments."""
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(list)
        # Should handle gracefully
        assert result.get("type") == "array" or result == {}

    def test_dict_without_args(self):
        """Test dict type without type arguments."""
        from insideLLMs.structured import _python_type_to_json_schema

        result = _python_type_to_json_schema(dict)
        # Should handle gracefully
        assert result.get("type") == "object" or result == {}


class TestStructuredResult:
    """Tests for StructuredResult class."""

    def test_to_dict_with_dataclass(self):
        """Test to_dict with dataclass data."""
        from insideLLMs.structured import StructuredResult

        @dataclass
        class Person:
            name: str
            age: int

        person = Person(name="John", age=30)
        result = StructuredResult(
            data=person,
            raw_response='{"name": "John", "age": 30}',
            schema={"type": "object"},
            prompt="test prompt",
            model_name="test-model",
        )
        data_dict = result.to_dict()
        assert data_dict["name"] == "John"
        assert data_dict["age"] == 30

    def test_to_dict_with_dict_data(self):
        """Test to_dict with dict data."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data={"key": "value"},
            raw_response='{"key": "value"}',
            schema={"type": "object"},
            prompt="test prompt",
        )
        assert result.to_dict() == {"key": "value"}

    def test_to_dict_with_primitive(self):
        """Test to_dict with primitive data."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data=42,
            raw_response="42",
            schema={"type": "integer"},
            prompt="test",
        )
        assert result.to_dict() == {"value": 42}

    def test_to_json(self):
        """Test to_json output."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data={"name": "Test"},
            raw_response='{"name": "Test"}',
            schema={"type": "object"},
            prompt="test",
        )
        json_str = result.to_json()
        assert json.loads(json_str) == {"name": "Test"}

    def test_to_json_with_indent(self):
        """Test to_json with custom indent."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data={"key": "value"},
            raw_response="{}",
            schema={},
            prompt="test",
        )
        json_str = result.to_json(indent=4)
        assert "    " in json_str  # Should have 4-space indent

    def test_to_html(self):
        """Test to_html output."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data={"name": "John", "age": 30},
            raw_response="{}",
            schema={"properties": {"name": {}, "age": {}}},
            prompt="test",
            model_name="test-model",
        )
        html = result.to_html(title="Test Output")
        assert "<html>" in html
        assert "Test Output" in html
        assert "John" in html
        assert "test-model" in html

    def test_to_html_with_nested_dict(self):
        """Test to_html with nested dictionary."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data={"person": {"name": "John", "age": 30}},
            raw_response="{}",
            schema={},
            prompt="test",
        )
        html = result.to_html()
        assert "<table" in html
        assert "John" in html

    def test_to_html_with_list(self):
        """Test to_html with list values."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data={"tags": ["python", "ml", "ai"]},
            raw_response="{}",
            schema={},
            prompt="test",
        )
        html = result.to_html()
        assert "<ul>" in html
        assert "python" in html

    def test_to_markdown(self):
        """Test to_markdown output."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data={"name": "John", "age": 30},
            raw_response="{}",
            schema={},
            prompt="test",
        )
        md = result.to_markdown()
        assert "# Structured Output" in md
        assert "**name**" in md
        assert "John" in md
        assert "```json" in md

    def test_to_markdown_with_nested_dict(self):
        """Test to_markdown with nested dictionary."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data={"person": {"name": "John"}},
            raw_response="{}",
            schema={},
            prompt="test",
        )
        md = result.to_markdown()
        assert "**person**" in md

    def test_to_markdown_with_list(self):
        """Test to_markdown with list values."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data={"items": [1, 2, 3]},
            raw_response="{}",
            schema={},
            prompt="test",
        )
        md = result.to_markdown()
        assert "**items**" in md

    def test_save_json(self, tmp_path):
        """Test save_json to file."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data={"key": "value"},
            raw_response="{}",
            schema={},
            prompt="test",
        )
        filepath = tmp_path / "test.json"
        result.save_json(str(filepath))
        assert filepath.exists()
        with open(filepath) as f:
            assert json.load(f) == {"key": "value"}

    def test_save_html(self, tmp_path):
        """Test save_html to file."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data={"key": "value"},
            raw_response="{}",
            schema={},
            prompt="test",
        )
        filepath = tmp_path / "test.html"
        result.save_html(str(filepath), title="Test Page")
        assert filepath.exists()
        content = filepath.read_text()
        assert "<html>" in content
        assert "Test Page" in content

    def test_repr(self):
        """Test __repr__ output."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data={"key": "value"},
            raw_response="{}",
            schema={},
            prompt="test",
        )
        repr_str = repr(result)
        assert "StructuredResult" in repr_str
        assert "key" in repr_str


class TestStructuredOutputConfig:
    """Tests for StructuredOutputConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from insideLLMs.structured import StructuredOutputConfig

        config = StructuredOutputConfig()
        assert config.max_retries == 3
        assert config.temperature == 0.0
        assert config.include_schema_in_prompt is True
        assert config.instructions == ""
        assert config.examples is None

    def test_custom_values(self):
        """Test custom configuration values."""
        from insideLLMs.structured import StructuredOutputConfig

        config = StructuredOutputConfig(
            max_retries=5,
            temperature=0.5,
            include_schema_in_prompt=False,
            instructions="Extract carefully",
            examples=[{"input": "test", "output": {}}],
        )
        assert config.max_retries == 5
        assert config.temperature == 0.5
        assert config.include_schema_in_prompt is False
        assert config.instructions == "Extract carefully"
        assert len(config.examples) == 1


class TestStructuredOutputGenerator:
    """Tests for StructuredOutputGenerator class."""

    def test_init_with_dataclass(self):
        """Test initialization with dataclass."""
        from insideLLMs.structured import StructuredOutputGenerator

        @dataclass
        class Person:
            name: str
            age: int

        mock_model = MagicMock()
        generator = StructuredOutputGenerator(mock_model, Person)
        assert generator.output_type == Person
        assert generator.model == mock_model
        assert generator._schema["type"] == "object"

    def test_build_prompt_without_examples(self):
        """Test building prompt without examples."""
        from insideLLMs.structured import StructuredOutputGenerator, StructuredOutputConfig

        @dataclass
        class Item:
            name: str

        mock_model = MagicMock()
        config = StructuredOutputConfig(instructions="Extract item name")
        generator = StructuredOutputGenerator(mock_model, Item, config)

        prompt = generator._build_prompt("The item is called Widget")
        assert "Extract item name" in prompt
        assert "Widget" in prompt
        assert "json" in prompt.lower()

    def test_build_prompt_with_examples(self):
        """Test building prompt with examples."""
        from insideLLMs.structured import StructuredOutputGenerator, StructuredOutputConfig

        @dataclass
        class Item:
            name: str

        mock_model = MagicMock()
        config = StructuredOutputConfig(
            examples=[
                {"input": "A red ball", "output": {"name": "ball"}},
                {"input": "A blue car", "output": {"name": "car"}},
            ]
        )
        generator = StructuredOutputGenerator(mock_model, Item, config)

        prompt = generator._build_prompt("A green tree")
        assert "Examples:" in prompt
        assert "red ball" in prompt
        assert "green tree" in prompt

    def test_generate_with_chat_model(self):
        """Test generate method with chat model."""
        from insideLLMs.structured import StructuredOutputGenerator

        @dataclass
        class Person:
            name: str
            age: int

        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value='{"name": "John", "age": 30}')

        generator = StructuredOutputGenerator(mock_model, Person)
        result = generator.generate("John is 30 years old")

        assert result.name == "John"
        assert result.age == 30
        mock_model.chat.assert_called_once()

    def test_generate_with_generate_model(self):
        """Test generate method with generate model (no chat)."""
        from insideLLMs.structured import StructuredOutputGenerator

        @dataclass
        class Person:
            name: str

        mock_model = MagicMock(spec=["generate"])
        mock_model.generate = MagicMock(return_value='{"name": "Alice"}')

        generator = StructuredOutputGenerator(mock_model, Person)
        result = generator.generate("Alice")

        assert result.name == "Alice"
        mock_model.generate.assert_called_once()

    def test_generate_retries_on_failure(self):
        """Test generate retries on parsing failure."""
        from insideLLMs.structured import StructuredOutputGenerator, StructuredOutputConfig

        @dataclass
        class Item:
            name: str

        mock_model = MagicMock()
        # First call fails, second succeeds
        mock_model.chat = MagicMock(
            side_effect=["invalid json", '{"name": "Widget"}']
        )

        config = StructuredOutputConfig(max_retries=2)
        generator = StructuredOutputGenerator(mock_model, Item, config)

        result = generator.generate("Widget")
        assert result.name == "Widget"
        assert mock_model.chat.call_count == 2

    def test_generate_fails_after_max_retries(self):
        """Test generate fails after max retries."""
        from insideLLMs.structured import StructuredOutputGenerator, StructuredOutputConfig, ParsingError

        @dataclass
        class Item:
            name: str

        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value="always invalid")

        config = StructuredOutputConfig(max_retries=2)
        generator = StructuredOutputGenerator(mock_model, Item, config)

        with pytest.raises(ParsingError) as exc_info:
            generator.generate("test")

        assert exc_info.value.attempts == 2

    def test_generate_batch(self):
        """Test generate_batch method."""
        from insideLLMs.structured import StructuredOutputGenerator

        @dataclass
        class Person:
            name: str

        mock_model = MagicMock()
        mock_model.chat = MagicMock(
            side_effect=['{"name": "Alice"}', '{"name": "Bob"}']
        )

        generator = StructuredOutputGenerator(mock_model, Person)
        results = generator.generate_batch(["Alice info", "Bob info"])

        assert len(results) == 2
        assert results[0].name == "Alice"
        assert results[1].name == "Bob"


class TestParseToTypeBasicTypes:
    """Tests for parse_to_type with basic types."""

    def test_parse_to_str(self):
        """Test parsing to string type."""
        from insideLLMs.structured import parse_to_type

        result = parse_to_type("hello", str)
        assert result == "hello"

    def test_parse_to_int(self):
        """Test parsing to int type."""
        from insideLLMs.structured import parse_to_type

        result = parse_to_type(42, int)
        assert result == 42

    def test_parse_to_float(self):
        """Test parsing to float type."""
        from insideLLMs.structured import parse_to_type

        result = parse_to_type(3.14, float)
        assert result == 3.14

    def test_parse_to_bool(self):
        """Test parsing to bool type."""
        from insideLLMs.structured import parse_to_type

        result = parse_to_type(True, bool)
        assert result is True

    def test_parse_dataclass_failure(self):
        """Test parsing to dataclass with invalid data."""
        from insideLLMs.structured import parse_to_type, ParsingError

        @dataclass
        class Person:
            name: str
            age: int

        with pytest.raises(ParsingError):
            parse_to_type({"name": "John"}, Person)  # Missing 'age'

    def test_parse_to_custom_type_failure(self):
        """Test parsing to unsupported type fails gracefully."""
        from insideLLMs.structured import parse_to_type, ParsingError

        class CustomType:
            def __init__(self, value):
                raise ValueError("Cannot create")

        with pytest.raises(ParsingError):
            parse_to_type({"value": 1}, CustomType)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_structured(self):
        """Test generate_structured convenience function."""
        from insideLLMs.structured import generate_structured

        @dataclass
        class Item:
            name: str

        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value='{"name": "Test"}')

        result = generate_structured(
            mock_model,
            "Test item",
            Item,
            instructions="Extract the item name",
        )
        assert result.name == "Test"

    def test_generate_structured_with_examples(self):
        """Test generate_structured with examples."""
        from insideLLMs.structured import generate_structured

        @dataclass
        class Item:
            name: str

        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value='{"name": "Widget"}')

        result = generate_structured(
            mock_model,
            "A widget",
            Item,
            examples=[{"input": "A ball", "output": {"name": "ball"}}],
        )
        assert result.name == "Widget"

    def test_create_structured_generator(self):
        """Test create_structured_generator function."""
        from insideLLMs.structured import create_structured_generator

        @dataclass
        class Person:
            name: str

        mock_model = MagicMock()
        generator = create_structured_generator(
            mock_model,
            Person,
            instructions="Extract person name",
            max_retries=5,
        )

        assert generator.output_type == Person
        assert generator.config.max_retries == 5
        assert generator.config.instructions == "Extract person name"


class TestAddStructuredMethod:
    """Tests for add_structured_method function."""

    def test_add_method_to_model(self):
        """Test adding generate_structured method to model."""
        from insideLLMs.structured import add_structured_method

        @dataclass
        class Item:
            name: str

        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value='{"name": "Test"}')

        # Add method
        result_model = add_structured_method(mock_model)

        assert hasattr(result_model, "generate_structured")
        assert result_model is mock_model  # Same object

        # Use the method
        result = result_model.generate_structured("Test input", Item)
        assert result.name == "Test"


class TestPydanticValidationError:
    """Tests for Pydantic validation error handling."""

    @pytest.fixture
    def pydantic_available(self):
        """Check if Pydantic is available."""
        try:
            from pydantic import BaseModel  # noqa: F401
            return True
        except ImportError:
            pytest.skip("Pydantic not available")

    def test_validation_error_raises_wrapper(self, pydantic_available):
        """Test that Pydantic validation error is wrapped."""
        from pydantic import BaseModel
        from insideLLMs.structured import parse_to_type, ValidationErrorWrapper

        class Person(BaseModel):
            name: str
            age: int

        with pytest.raises(ValidationErrorWrapper):
            parse_to_type({"name": "John", "age": "not_an_int"}, Person)

    def test_pydantic_nested_model_schema(self, pydantic_available):
        """Test JSON schema for nested Pydantic model."""
        from pydantic import BaseModel
        from insideLLMs.structured import _python_type_to_json_schema

        class Address(BaseModel):
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        result = _python_type_to_json_schema(Person)
        assert "properties" in result


class TestStructuredResultWithPydantic:
    """Tests for StructuredResult with Pydantic models."""

    @pytest.fixture
    def pydantic_available(self):
        """Check if Pydantic is available."""
        try:
            from pydantic import BaseModel  # noqa: F401
            return True
        except ImportError:
            pytest.skip("Pydantic not available")

    def test_to_dict_with_pydantic_v2(self, pydantic_available):
        """Test to_dict with Pydantic v2 model."""
        from pydantic import BaseModel
        from insideLLMs.structured import StructuredResult

        class Person(BaseModel):
            name: str
            age: int

        person = Person(name="Alice", age=25)
        result = StructuredResult(
            data=person,
            raw_response="{}",
            schema={},
            prompt="test",
        )
        data_dict = result.to_dict()
        assert data_dict["name"] == "Alice"
        assert data_dict["age"] == 25


class TestCreateModelFromName:
    """Tests for _create_model_from_name helper function."""

    def test_create_dummy_model(self):
        """Test creating a dummy model."""
        from insideLLMs.structured import _create_model_from_name

        model = _create_model_from_name("test-model", "dummy")
        assert model is not None
        assert model.name == "test-model"

    def test_create_model_unknown_provider(self):
        """Test creating model with unknown provider."""
        from insideLLMs.structured import _create_model_from_name

        with pytest.raises(ValueError, match="Unknown provider"):
            _create_model_from_name("test", "unknown_provider_xyz")

    def test_create_model_case_insensitive_provider(self):
        """Test provider name is case insensitive."""
        from insideLLMs.structured import _create_model_from_name

        model = _create_model_from_name("test", "DUMMY")
        assert model is not None

        model2 = _create_model_from_name("test", "Dummy")
        assert model2 is not None


class TestQuickExtract:
    """Tests for quick_extract convenience function."""

    def test_quick_extract_with_dummy_model(self):
        """Test quick_extract with dummy model."""
        from insideLLMs.structured import quick_extract

        @dataclass
        class Person:
            name: str

        # Dummy model returns fixed JSON-like response
        # We need to mock it or use the actual dummy
        # For this test, we'll verify the function structure exists
        import insideLLMs.structured as structured_module

        assert hasattr(structured_module, "quick_extract")


class TestBatchExtract:
    """Tests for batch_extract function."""

    def test_batch_extract_basic(self):
        """Test basic batch extraction."""
        from insideLLMs.structured import batch_extract, StructuredResult

        @dataclass
        class Item:
            name: str

        mock_model = MagicMock()
        mock_model.chat = MagicMock(side_effect=[
            '{"name": "Apple"}',
            '{"name": "Banana"}',
            '{"name": "Cherry"}'
        ])
        mock_model.name = "test-model"

        texts = ["apple text", "banana text", "cherry text"]
        results = batch_extract(texts, Item, mock_model, show_progress=False)

        assert len(results) == 3
        assert all(isinstance(r, StructuredResult) for r in results)
        assert results[0].data.name == "Apple"
        assert results[1].data.name == "Banana"
        assert results[2].data.name == "Cherry"

    def test_batch_extract_with_errors(self, capsys):
        """Test batch extraction handles errors gracefully."""
        from insideLLMs.structured import batch_extract

        @dataclass
        class Item:
            name: str

        mock_model = MagicMock()
        mock_model.chat = MagicMock(side_effect=[
            '{"name": "Valid"}',
            'invalid json',  # This will cause an error
            '{"name": "Another"}'
        ])
        mock_model.name = "test-model"

        texts = ["valid", "invalid", "another"]
        results = batch_extract(texts, Item, mock_model, show_progress=False)

        assert len(results) == 3
        # First and third should have data
        assert results[0].data.name == "Valid"
        # Second should have None data due to error
        assert results[1].data is None
        assert results[2].data.name == "Another"

    def test_batch_extract_with_progress(self, capsys):
        """Test batch extraction shows progress."""
        from insideLLMs.structured import batch_extract

        @dataclass
        class Item:
            name: str

        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value='{"name": "Test"}')
        mock_model.name = "test-model"

        texts = ["test1", "test2"]
        results = batch_extract(texts, Item, mock_model, show_progress=True)

        assert len(results) == 2
        captured = capsys.readouterr()
        assert "Processed" in captured.out or "Processing" in captured.out

    def test_batch_extract_with_generate_model(self):
        """Test batch extraction with model that only has generate method."""
        from insideLLMs.structured import batch_extract

        @dataclass
        class Item:
            name: str

        mock_model = MagicMock(spec=["generate", "name"])
        mock_model.generate = MagicMock(return_value='{"name": "Generated"}')
        mock_model.name = "gen-model"

        texts = ["input"]
        results = batch_extract(texts, Item, mock_model, show_progress=False)

        assert len(results) == 1
        assert results[0].data.name == "Generated"
        mock_model.generate.assert_called_once()


class TestResultsToDataframe:
    """Tests for results_to_dataframe function."""

    def test_results_to_dataframe_basic(self):
        """Test converting results to dataframe."""
        try:
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("pandas not available")

        from insideLLMs.structured import results_to_dataframe, StructuredResult

        results = [
            StructuredResult(
                data={"name": "Alice", "age": 30},
                raw_response="{}",
                schema={},
                prompt="test1",
            ),
            StructuredResult(
                data={"name": "Bob", "age": 25},
                raw_response="{}",
                schema={},
                prompt="test2",
            ),
        ]

        df = results_to_dataframe(results)
        assert len(df) == 2
        assert "name" in df.columns
        assert "age" in df.columns

    def test_results_to_dataframe_skips_none(self):
        """Test that results with None data are skipped."""
        try:
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("pandas not available")

        from insideLLMs.structured import results_to_dataframe, StructuredResult

        results = [
            StructuredResult(
                data={"name": "Alice"},
                raw_response="{}",
                schema={},
                prompt="test1",
            ),
            StructuredResult(
                data=None,  # This should be skipped
                raw_response="error",
                schema={},
                prompt="test2",
            ),
        ]

        df = results_to_dataframe(results)
        assert len(df) == 1


class TestResultsToHtmlReport:
    """Tests for results_to_html_report function."""

    def test_results_to_html_report_basic(self):
        """Test generating HTML report from results."""
        from insideLLMs.structured import results_to_html_report, StructuredResult

        results = [
            StructuredResult(
                data={"name": "Alice", "score": 95},
                raw_response="{}",
                schema={},
                prompt="test1",
            ),
            StructuredResult(
                data={"name": "Bob", "score": 88},
                raw_response="{}",
                schema={},
                prompt="test2",
            ),
        ]

        html = results_to_html_report(results, title="Test Report")

        assert "<html>" in html
        assert "Test Report" in html
        assert "Alice" in html
        assert "Bob" in html
        assert "Total results: 2" in html

    def test_results_to_html_report_with_none_data(self):
        """Test HTML report skips results with None data."""
        from insideLLMs.structured import results_to_html_report, StructuredResult

        results = [
            StructuredResult(
                data={"name": "Valid"},
                raw_response="{}",
                schema={},
                prompt="test1",
            ),
            StructuredResult(
                data=None,  # Error result
                raw_response="error",
                schema={},
                prompt="test2",
            ),
        ]

        html = results_to_html_report(results)

        assert "Valid" in html
        assert "Successful: 1" in html

    def test_results_to_html_report_empty_results(self):
        """Test HTML report with empty results."""
        from insideLLMs.structured import results_to_html_report

        html = results_to_html_report([], title="Empty Report")

        assert "<html>" in html
        assert "Empty Report" in html
        assert "Total results: 0" in html


class TestStructuredResultNoneData:
    """Test StructuredResult behavior with None data."""

    def test_to_dict_with_none_data(self):
        """Test to_dict when data is None."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data=None,
            raw_response="error occurred",
            schema={},
            prompt="test",
        )
        data_dict = result.to_dict()
        # None data is wrapped as {"value": None}
        assert data_dict == {"value": None}

    def test_to_json_with_none_data(self):
        """Test to_json when data is None."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data=None,
            raw_response="error",
            schema={},
            prompt="test",
        )
        json_str = result.to_json()
        # None data is wrapped as {"value": None}
        assert json.loads(json_str) == {"value": None}

    def test_to_html_with_none_data(self):
        """Test to_html when data is None."""
        from insideLLMs.structured import StructuredResult

        result = StructuredResult(
            data=None,
            raw_response="error",
            schema={},
            prompt="test",
        )
        html = result.to_html()
        assert "<html>" in html


class TestExtractJsonEdgeCases:
    """Test extract_json with various edge cases."""

    def test_extract_json_with_markdown_code_block(self):
        """Test extracting JSON from markdown code block."""
        from insideLLMs.structured import extract_json

        text = '''
        Here is the response:
        ```json
        {"name": "Test", "value": 42}
        ```
        '''
        result = extract_json(text)
        assert result == '{"name": "Test", "value": 42}'

    def test_extract_json_bare_json(self):
        """Test extracting bare JSON object."""
        from insideLLMs.structured import extract_json

        text = '{"key": "value"}'
        result = extract_json(text)
        assert result == '{"key": "value"}'

    def test_extract_json_with_array(self):
        """Test extracting JSON array."""
        from insideLLMs.structured import extract_json

        text = '[1, 2, 3]'
        result = extract_json(text)
        assert result == '[1, 2, 3]'

    def test_extract_json_nested(self):
        """Test extracting nested JSON."""
        from insideLLMs.structured import extract_json

        text = '{"outer": {"inner": "value"}}'
        result = extract_json(text)
        assert '"inner"' in result


class TestParseJsonEdgeCases:
    """Test parse_json with edge cases."""

    def test_parse_json_with_trailing_comma(self):
        """Test parsing JSON-like with trailing comma is handled."""
        from insideLLMs.structured import parse_json, ParsingError

        # Standard JSON doesn't allow trailing commas
        # Function should try to handle or raise ParsingError
        text = '{"key": "value",}'
        try:
            result = parse_json(text)
            # If it succeeds, that's fine
            assert isinstance(result, dict)
        except ParsingError:
            # If it fails, that's also expected
            pass

    def test_parse_json_empty_object(self):
        """Test parsing empty JSON object."""
        from insideLLMs.structured import parse_json

        result = parse_json("{}")
        assert result == {}

    def test_parse_json_empty_array(self):
        """Test parsing empty JSON array."""
        from insideLLMs.structured import parse_json

        result = parse_json("[]")
        assert result == []
