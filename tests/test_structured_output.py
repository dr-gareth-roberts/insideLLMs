"""Tests for Structured Output Parsing with Pydantic integration."""

import json
import dataclasses
import pytest
from typing import List, Optional, Dict, Any, Union
from unittest.mock import MagicMock

# Import Pydantic if available
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


from insideLLMs.structured import (
    StructuredOutputError,
    SchemaGenerationError,
    ParsingError,
    ValidationErrorWrapper,
    extract_json,
    parse_json,
    parse_to_type,
    get_json_schema,
    dataclass_to_json_schema,
    StructuredOutputConfig,
    StructuredOutputGenerator,
    generate_structured,
    create_structured_generator,
    add_structured_method,
    PYDANTIC_AVAILABLE as MODULE_PYDANTIC_AVAILABLE,
)
from insideLLMs.models import DummyModel


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclasses.dataclass
class PersonDataclass:
    """Test dataclass."""
    name: str
    age: int
    email: Optional[str] = None


@dataclasses.dataclass
class AddressDataclass:
    """Test nested dataclass."""
    street: str
    city: str
    country: str = "USA"


@dataclasses.dataclass
class PersonWithAddressDataclass:
    """Test dataclass with nested type."""
    name: str
    address: AddressDataclass


if PYDANTIC_AVAILABLE:
    class PersonModel(BaseModel):
        """Test Pydantic model."""
        name: str
        age: int
        email: Optional[str] = None

    class AddressModel(BaseModel):
        """Test nested Pydantic model."""
        street: str
        city: str
        country: str = "USA"

    class PersonWithAddressModel(BaseModel):
        """Test Pydantic model with nested model."""
        name: str
        address: AddressModel

    class ItemModel(BaseModel):
        """Test model for lists."""
        id: int
        name: str
        tags: List[str] = []
        metadata: Dict[str, Any] = {}


# =============================================================================
# Test JSON Extraction
# =============================================================================


class TestExtractJson:
    """Tests for JSON extraction from text."""

    def test_extract_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        text = '''Here is the result:
```json
{"name": "John", "age": 30}
```
That's all.'''
        result = extract_json(text)
        assert json.loads(result) == {"name": "John", "age": 30}

    def test_extract_from_plain_code_block(self):
        """Test extracting JSON from plain code block."""
        text = '''```
{"name": "Jane", "age": 25}
```'''
        result = extract_json(text)
        assert json.loads(result) == {"name": "Jane", "age": 25}

    def test_extract_raw_json_object(self):
        """Test extracting raw JSON object."""
        text = 'The result is {"name": "Bob", "age": 40} as you can see.'
        result = extract_json(text)
        assert json.loads(result) == {"name": "Bob", "age": 40}

    def test_extract_raw_json_array(self):
        """Test extracting raw JSON array."""
        text = 'Here are the items: [1, 2, 3]'
        result = extract_json(text)
        assert json.loads(result) == [1, 2, 3]

    def test_extract_plain_json(self):
        """Test extracting plain JSON text."""
        text = '{"key": "value"}'
        result = extract_json(text)
        assert json.loads(result) == {"key": "value"}

    def test_extract_multiline_json(self):
        """Test extracting multiline JSON."""
        text = '''```json
{
    "name": "Test",
    "items": [1, 2, 3],
    "nested": {
        "key": "value"
    }
}
```'''
        result = extract_json(text)
        data = json.loads(result)
        assert data["name"] == "Test"
        assert data["items"] == [1, 2, 3]
        assert data["nested"]["key"] == "value"

    def test_extract_no_json_raises(self):
        """Test that missing JSON raises ParsingError."""
        text = "No JSON here, just plain text."
        with pytest.raises(ParsingError, match="Could not extract valid JSON"):
            extract_json(text)

    def test_extract_invalid_json_raises(self):
        """Test that invalid JSON raises ParsingError."""
        text = "{not: valid: json}"
        with pytest.raises(ParsingError):
            extract_json(text)


class TestParseJson:
    """Tests for parse_json function."""

    def test_parse_simple_object(self):
        """Test parsing simple JSON object."""
        text = '{"name": "Test", "value": 42}'
        result = parse_json(text)
        assert result == {"name": "Test", "value": 42}

    def test_parse_from_markdown(self):
        """Test parsing JSON from markdown."""
        text = '```json\n[1, 2, 3]\n```'
        result = parse_json(text)
        assert result == [1, 2, 3]


# =============================================================================
# Test Schema Generation
# =============================================================================


class TestDataclassSchema:
    """Tests for dataclass schema generation."""

    def test_simple_dataclass_schema(self):
        """Test schema generation for simple dataclass."""
        schema = dataclass_to_json_schema(PersonDataclass)

        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"
        assert "name" in schema["required"]
        assert "age" in schema["required"]
        assert "email" not in schema.get("required", [])

    def test_non_dataclass_raises(self):
        """Test that non-dataclass raises error."""
        with pytest.raises(SchemaGenerationError):
            dataclass_to_json_schema(str)


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not installed")
class TestPydanticSchema:
    """Tests for Pydantic schema generation."""

    def test_simple_model_schema(self):
        """Test schema generation for simple Pydantic model."""
        schema = get_json_schema(PersonModel)

        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_nested_model_schema(self):
        """Test schema generation for nested Pydantic model."""
        schema = get_json_schema(PersonWithAddressModel)

        assert "name" in schema["properties"]
        assert "address" in schema["properties"]

    def test_model_with_lists(self):
        """Test schema for model with list fields."""
        schema = get_json_schema(ItemModel)

        assert "tags" in schema["properties"]


class TestGetJsonSchema:
    """Tests for get_json_schema universal function."""

    def test_schema_for_dataclass(self):
        """Test schema generation for dataclass."""
        schema = get_json_schema(PersonDataclass)
        assert schema["type"] == "object"

    def test_schema_for_basic_types(self):
        """Test schema generation for basic types."""
        from insideLLMs.structured import _python_type_to_json_schema

        assert _python_type_to_json_schema(str) == {"type": "string"}
        assert _python_type_to_json_schema(int) == {"type": "integer"}
        assert _python_type_to_json_schema(float) == {"type": "number"}
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_schema_for_list(self):
        """Test schema for list type."""
        from insideLLMs.structured import _python_type_to_json_schema

        schema = _python_type_to_json_schema(List[str])
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_schema_for_dict(self):
        """Test schema for dict type."""
        from insideLLMs.structured import _python_type_to_json_schema

        schema = _python_type_to_json_schema(Dict[str, int])
        assert schema["type"] == "object"
        assert schema["additionalProperties"]["type"] == "integer"

    def test_schema_for_optional(self):
        """Test schema for Optional type."""
        from insideLLMs.structured import _python_type_to_json_schema

        schema = _python_type_to_json_schema(Optional[str])
        assert "anyOf" in schema


# =============================================================================
# Test Parsing to Types
# =============================================================================


class TestParseToType:
    """Tests for parse_to_type function."""

    def test_parse_to_dataclass(self):
        """Test parsing dict to dataclass."""
        data = {"name": "John", "age": 30, "email": "john@example.com"}
        result = parse_to_type(data, PersonDataclass)

        assert isinstance(result, PersonDataclass)
        assert result.name == "John"
        assert result.age == 30
        assert result.email == "john@example.com"

    def test_parse_to_dataclass_with_defaults(self):
        """Test parsing to dataclass uses defaults."""
        data = {"name": "Jane", "age": 25}
        result = parse_to_type(data, PersonDataclass)

        assert result.name == "Jane"
        assert result.email is None

    def test_parse_to_dict(self):
        """Test parsing to dict."""
        data = {"key": "value"}
        result = parse_to_type(data, dict)
        assert result == {"key": "value"}

    def test_parse_to_list(self):
        """Test parsing to list."""
        data = [1, 2, 3]
        result = parse_to_type(data, list)
        assert result == [1, 2, 3]

    def test_parse_to_basic_types(self):
        """Test parsing to basic types."""
        assert parse_to_type("test", str) == "test"
        assert parse_to_type(42, int) == 42
        assert parse_to_type(3.14, float) == 3.14
        assert parse_to_type(True, bool) is True


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not installed")
class TestParseToTypePydantic:
    """Tests for parsing to Pydantic models."""

    def test_parse_to_pydantic_model(self):
        """Test parsing dict to Pydantic model."""
        data = {"name": "John", "age": 30}
        result = parse_to_type(data, PersonModel)

        assert isinstance(result, PersonModel)
        assert result.name == "John"
        assert result.age == 30

    def test_parse_to_nested_pydantic_model(self):
        """Test parsing to nested Pydantic model."""
        data = {
            "name": "John",
            "address": {
                "street": "123 Main St",
                "city": "Boston",
            }
        }
        result = parse_to_type(data, PersonWithAddressModel)

        assert result.name == "John"
        assert result.address.street == "123 Main St"
        assert result.address.city == "Boston"
        assert result.address.country == "USA"  # default

    def test_parse_invalid_data_raises(self):
        """Test that invalid data raises ValidationErrorWrapper."""
        data = {"name": "John", "age": "not a number"}
        with pytest.raises(ValidationErrorWrapper):
            parse_to_type(data, PersonModel)


# =============================================================================
# Test Structured Output Config
# =============================================================================


class TestStructuredOutputConfig:
    """Tests for StructuredOutputConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StructuredOutputConfig()

        assert config.max_retries == 3
        assert config.temperature == 0.0
        assert config.include_schema_in_prompt is True
        assert config.instructions == ""
        assert config.examples is None

    def test_custom_config(self):
        """Test custom configuration."""
        examples = [{"input": "test", "output": {"name": "Test"}}]
        config = StructuredOutputConfig(
            max_retries=5,
            temperature=0.7,
            instructions="Extract carefully",
            examples=examples,
        )

        assert config.max_retries == 5
        assert config.temperature == 0.7
        assert config.instructions == "Extract carefully"
        assert config.examples == examples


# =============================================================================
# Test Structured Output Generator
# =============================================================================


class TestStructuredOutputGenerator:
    """Tests for StructuredOutputGenerator class."""

    def test_generator_initialization(self):
        """Test generator initialization."""
        mock_model = MagicMock()
        generator = StructuredOutputGenerator(mock_model, PersonDataclass)

        assert generator.model is mock_model
        assert generator.output_type is PersonDataclass
        assert generator._schema is not None

    def test_generator_with_config(self):
        """Test generator with custom config."""
        mock_model = MagicMock()
        config = StructuredOutputConfig(max_retries=5)
        generator = StructuredOutputGenerator(mock_model, PersonDataclass, config)

        assert generator.config.max_retries == 5

    def test_build_prompt(self):
        """Test prompt building."""
        mock_model = MagicMock()
        generator = StructuredOutputGenerator(mock_model, PersonDataclass)
        prompt = generator._build_prompt("John is 30 years old")

        assert "John is 30 years old" in prompt
        assert "name" in prompt  # Schema should be included
        assert "age" in prompt

    def test_build_prompt_with_examples(self):
        """Test prompt building with examples."""
        mock_model = MagicMock()
        config = StructuredOutputConfig(
            examples=[
                {"input": "Jane is 25", "output": {"name": "Jane", "age": 25}}
            ]
        )
        generator = StructuredOutputGenerator(mock_model, PersonDataclass, config)
        prompt = generator._build_prompt("Bob is 40")

        assert "Jane is 25" in prompt
        assert "Examples:" in prompt

    def test_generate_with_chat_model(self):
        """Test generation using chat model."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value='{"name": "John", "age": 30}')

        generator = StructuredOutputGenerator(mock_model, PersonDataclass)
        result = generator.generate("John is 30 years old")

        assert result.name == "John"
        assert result.age == 30
        mock_model.chat.assert_called_once()

    def test_generate_with_generate_model(self):
        """Test generation using generate-only model."""
        mock_model = MagicMock(spec=['generate'])
        mock_model.generate = MagicMock(return_value='{"name": "Jane", "age": 25}')

        generator = StructuredOutputGenerator(mock_model, PersonDataclass)
        result = generator.generate("Jane is 25 years old")

        assert result.name == "Jane"
        assert result.age == 25
        mock_model.generate.assert_called_once()

    def test_generate_with_code_block_response(self):
        """Test generation handles markdown code blocks."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value='''```json
{"name": "Bob", "age": 40}
```''')

        generator = StructuredOutputGenerator(mock_model, PersonDataclass)
        result = generator.generate("Bob is 40")

        assert result.name == "Bob"
        assert result.age == 40

    def test_generate_retries_on_failure(self):
        """Test that generator retries on parsing failure."""
        mock_model = MagicMock()
        # First two calls return invalid JSON, third succeeds
        mock_model.chat = MagicMock(side_effect=[
            "Invalid JSON",
            "Still invalid",
            '{"name": "Test", "age": 1}',
        ])

        config = StructuredOutputConfig(max_retries=3)
        generator = StructuredOutputGenerator(mock_model, PersonDataclass, config)
        result = generator.generate("Test is 1")

        assert result.name == "Test"
        assert mock_model.chat.call_count == 3

    def test_generate_fails_after_max_retries(self):
        """Test that generator fails after max retries."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value="Never valid JSON")

        config = StructuredOutputConfig(max_retries=2)
        generator = StructuredOutputGenerator(mock_model, PersonDataclass, config)

        with pytest.raises(ParsingError) as exc_info:
            generator.generate("Test")

        assert exc_info.value.attempts == 2

    def test_generate_batch(self):
        """Test batch generation."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(side_effect=[
            '{"name": "John", "age": 30}',
            '{"name": "Jane", "age": 25}',
        ])

        generator = StructuredOutputGenerator(mock_model, PersonDataclass)
        results = generator.generate_batch(["John is 30", "Jane is 25"])

        assert len(results) == 2
        assert results[0].name == "John"
        assert results[1].name == "Jane"


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestGenerateStructured:
    """Tests for generate_structured convenience function."""

    def test_basic_generation(self):
        """Test basic structured generation."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value='{"name": "Test", "age": 42}')

        result = generate_structured(mock_model, "Test is 42", PersonDataclass)

        assert result.name == "Test"
        assert result.age == 42

    def test_with_instructions(self):
        """Test generation with custom instructions."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value='{"name": "Test", "age": 1}')

        result = generate_structured(
            mock_model,
            "Test",
            PersonDataclass,
            instructions="Extract the person's information carefully.",
        )

        # Verify instructions were passed (via prompt)
        call_args = mock_model.chat.call_args
        messages = call_args[0][0]
        assert "Extract the person's information carefully" in messages[1]["content"]

    def test_with_examples(self):
        """Test generation with few-shot examples."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value='{"name": "New", "age": 99}')

        examples = [
            {"input": "Alice is 20", "output": {"name": "Alice", "age": 20}},
        ]

        result = generate_structured(
            mock_model,
            "New person is 99",
            PersonDataclass,
            examples=examples,
        )

        # Verify examples were included in prompt
        call_args = mock_model.chat.call_args
        messages = call_args[0][0]
        assert "Alice is 20" in messages[1]["content"]


class TestCreateStructuredGenerator:
    """Tests for create_structured_generator convenience function."""

    def test_create_generator(self):
        """Test creating a reusable generator."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value='{"name": "Test", "age": 1}')

        generator = create_structured_generator(
            mock_model,
            PersonDataclass,
            instructions="Extract person info",
            max_retries=5,
        )

        assert isinstance(generator, StructuredOutputGenerator)
        assert generator.config.max_retries == 5
        assert generator.config.instructions == "Extract person info"

        # Test that generator works
        result = generator.generate("Test is 1")
        assert result.name == "Test"


class TestAddStructuredMethod:
    """Tests for add_structured_method function."""

    def test_add_method_to_model(self):
        """Test adding structured method to model."""
        class MockModel:
            def chat(self, messages, **kwargs):
                return '{"name": "Extended", "age": 99}'

        model = MockModel()
        extended_model = add_structured_method(model)

        # Method should be added
        assert hasattr(extended_model, "generate_structured")

        # Method should work
        result = extended_model.generate_structured("Extended is 99", PersonDataclass)
        assert result.name == "Extended"
        assert result.age == 99


# =============================================================================
# Test Integration with DummyModel
# =============================================================================


class TestDummyModelIntegration:
    """Integration tests using DummyModel."""

    def test_with_json_dummy_model(self):
        """Test with a DummyModel that returns JSON."""
        class JSONDummyModel(DummyModel):
            def generate(self, prompt, **kwargs):
                return '{"name": "Dummy", "age": 100}'

            def chat(self, messages, **kwargs):
                return self.generate(messages[-1]["content"])

        model = JSONDummyModel()
        result = generate_structured(model, "Dummy is 100", PersonDataclass)

        assert result.name == "Dummy"
        assert result.age == 100


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not installed")
class TestPydanticIntegration:
    """Integration tests with Pydantic models."""

    def test_complex_pydantic_model(self):
        """Test with complex Pydantic model."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value=json.dumps({
            "id": 1,
            "name": "Test Item",
            "tags": ["tag1", "tag2"],
            "metadata": {"key": "value"}
        }))

        result = generate_structured(mock_model, "Item data", ItemModel)

        assert result.id == 1
        assert result.name == "Test Item"
        assert result.tags == ["tag1", "tag2"]
        assert result.metadata == {"key": "value"}

    def test_nested_pydantic_model(self):
        """Test with nested Pydantic model."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value=json.dumps({
            "name": "John",
            "address": {
                "street": "123 Main",
                "city": "Boston",
                "country": "USA"
            }
        }))

        result = generate_structured(mock_model, "Person data", PersonWithAddressModel)

        assert result.name == "John"
        assert result.address.street == "123 Main"
        assert result.address.city == "Boston"


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_parsing_error_contains_raw_output(self):
        """Test that ParsingError contains raw output."""
        try:
            extract_json("not json")
        except ParsingError as e:
            assert "not json" in e.raw_output

    def test_parsing_error_contains_attempts(self):
        """Test that ParsingError tracks attempts."""
        mock_model = MagicMock()
        mock_model.chat = MagicMock(return_value="invalid")

        config = StructuredOutputConfig(max_retries=2)
        generator = StructuredOutputGenerator(mock_model, PersonDataclass, config)

        try:
            generator.generate("test")
        except ParsingError as e:
            assert e.attempts == 2

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not installed")
    def test_validation_error_wrapper(self):
        """Test ValidationErrorWrapper contains original error."""
        mock_model = MagicMock()
        # Return data with wrong type for 'age'
        mock_model.chat = MagicMock(return_value='{"name": "Test", "age": "not a number"}')

        generator = StructuredOutputGenerator(mock_model, PersonModel)

        with pytest.raises(ParsingError):
            generator.generate("test")
