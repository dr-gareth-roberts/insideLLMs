"""Structured Output Parsing with Pydantic Integration.

This module provides a comprehensive framework for extracting structured data
from Large Language Model (LLM) outputs. It bridges the gap between free-form
text generation and type-safe Python objects, enabling reliable data extraction
workflows.

Overview
--------
The module offers several layers of abstraction for different use cases:

1. **Quick Extraction**: One-liner functions for simple extraction tasks
2. **Generator Pattern**: Reusable generators for repeated extractions
3. **Batch Processing**: Efficient processing of multiple inputs
4. **Result Containers**: Rich result objects with export capabilities

Key Features
------------
- **Pydantic Integration**: First-class support for Pydantic models (v1 and v2)
- **Dataclass Support**: Works with Python dataclasses as output types
- **JSON Schema Generation**: Automatic schema generation for prompting
- **Robust Parsing**: Extracts JSON from markdown, code blocks, and mixed text
- **Validation**: Automatic validation with helpful error messages
- **Retry Logic**: Configurable retry attempts for malformed outputs
- **Export Formats**: JSON, HTML, Markdown, and pandas DataFrame exports
- **Nested Models**: Full support for complex nested data structures

Architecture
------------
The extraction pipeline follows these stages:

1. Schema Generation: Convert Python type to JSON schema
2. Prompt Building: Create extraction prompt with schema and instructions
3. LLM Generation: Send prompt to model and receive response
4. JSON Extraction: Parse JSON from potentially mixed-content response
5. Validation: Validate and instantiate the target type
6. Result Wrapping: Package data with metadata and export methods

Examples
--------
Basic extraction with Pydantic model:

    >>> from pydantic import BaseModel
    >>> from insideLLMs.structured import generate_structured
    >>> from insideLLMs.models import OpenAIModel
    >>>
    >>> class Person(BaseModel):
    ...     name: str
    ...     age: int
    ...     occupation: str
    >>>
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> person = generate_structured(
    ...     model,
    ...     "Extract: John is a 30-year-old software engineer",
    ...     Person
    ... )
    >>> print(person.name, person.age)
    John 30

Quick extraction with automatic model setup:

    >>> from pydantic import BaseModel
    >>> from insideLLMs.structured import quick_extract
    >>>
    >>> class Product(BaseModel):
    ...     name: str
    ...     price: float
    ...     in_stock: bool
    >>>
    >>> result = quick_extract(
    ...     "The iPhone 15 Pro costs $999 and is currently available",
    ...     Product,
    ...     model_name="gpt-4",
    ...     provider="openai"
    ... )
    >>> print(result.data.name)  # "iPhone 15 Pro"
    >>> result.save_json("product.json")
    >>> print(result.to_html())  # Get HTML view

Reusable generator for multiple extractions:

    >>> from insideLLMs.structured import create_structured_generator
    >>>
    >>> class ContactInfo(BaseModel):
    ...     email: str
    ...     phone: Optional[str] = None
    ...     company: str
    >>>
    >>> generator = create_structured_generator(
    ...     model,
    ...     ContactInfo,
    ...     instructions="Extract contact information from business cards"
    ... )
    >>> contact1 = generator.generate("John Smith, john@acme.com, Acme Inc")
    >>> contact2 = generator.generate("Jane Doe, jane@corp.com, 555-1234, Corp LLC")

Batch processing with progress:

    >>> from insideLLMs.structured import batch_extract, results_to_dataframe
    >>>
    >>> texts = [
    ...     "Alice, 28, Engineer at TechCorp",
    ...     "Bob, 35, Manager at RetailCo",
    ...     "Carol, 42, Director at FinanceInc"
    ... ]
    >>> results = batch_extract(texts, Person, model)
    >>> df = results_to_dataframe(results)
    >>> print(df)

Using dataclasses instead of Pydantic:

    >>> from dataclasses import dataclass
    >>> from insideLLMs.structured import generate_structured
    >>>
    >>> @dataclass
    ... class Event:
    ...     title: str
    ...     date: str
    ...     location: str
    >>>
    >>> event = generate_structured(
    ...     model,
    ...     "Annual Tech Conference on March 15, 2024 at Convention Center",
    ...     Event
    ... )

Few-shot prompting with examples:

    >>> examples = [
    ...     {
    ...         "input": "The meeting is at 3pm tomorrow",
    ...         "output": {"title": "Meeting", "date": "tomorrow 3pm", "location": "TBD"}
    ...     },
    ...     {
    ...         "input": "Lunch at Joe's Diner, Friday noon",
    ...         "output": {"title": "Lunch", "date": "Friday 12:00", "location": "Joe's Diner"}
    ...     }
    ... ]
    >>> event = generate_structured(
    ...     model,
    ...     "Team standup Monday 9am in Room 101",
    ...     Event,
    ...     examples=examples
    ... )

Notes
-----
- Pydantic v2 is preferred but v1 is fully supported
- For complex nested types, ensure all nested models are also Pydantic/dataclass
- Temperature is set to 0.0 by default for deterministic extraction
- The module falls back gracefully when Pydantic is not installed
- JSON extraction handles markdown code blocks, raw JSON, and mixed content

See Also
--------
insideLLMs.structured_extraction : Lower-level extraction utilities
insideLLMs.models : Model implementations for different providers
pydantic : Data validation library (https://docs.pydantic.dev/)

References
----------
.. [1] Pydantic Documentation: https://docs.pydantic.dev/
.. [2] JSON Schema Specification: https://json-schema.org/
"""

import json
import re
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

if TYPE_CHECKING:
    from insideLLMs.models.base import Model

# Try to import Pydantic
try:
    from pydantic import BaseModel, ValidationError

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
    ValidationError = None

T = TypeVar("T")


# =============================================================================
# Exceptions
# =============================================================================


class StructuredOutputError(Exception):
    """Base exception for all structured output errors.

    This is the parent class for all exceptions raised by the structured
    output module. Catching this exception will handle any error from
    schema generation, parsing, or validation.

    Parameters
    ----------
    message : str
        Human-readable error description.

    Examples
    --------
    Catch any structured output error:

        >>> try:
        ...     result = generate_structured(model, text, MyModel)
        ... except StructuredOutputError as e:
        ...     print(f"Extraction failed: {e}")
        ...     # Handle error gracefully

    Catch specific error types:

        >>> try:
        ...     result = generate_structured(model, text, MyModel)
        ... except ParsingError as e:
        ...     print(f"JSON parsing failed: {e.raw_output}")
        ... except ValidationErrorWrapper as e:
        ...     print(f"Validation failed: {e.validation_error}")
        ... except SchemaGenerationError as e:
        ...     print(f"Schema error: {e}")

    See Also
    --------
    SchemaGenerationError : Raised when JSON schema cannot be generated.
    ParsingError : Raised when JSON cannot be extracted from output.
    ValidationErrorWrapper : Raised when parsed data fails validation.
    """

    pass


class SchemaGenerationError(StructuredOutputError):
    """Exception raised when JSON schema generation fails.

    This error occurs when the module cannot convert a Python type
    (Pydantic model, dataclass, or other type) into a valid JSON schema.
    Common causes include unsupported types or circular references.

    Parameters
    ----------
    message : str
        Description of the schema generation failure.

    Examples
    --------
    Handle schema generation errors:

        >>> from insideLLMs.structured import get_json_schema, SchemaGenerationError
        >>>
        >>> class UnsupportedType:
        ...     def __init__(self, data):
        ...         self.data = data
        >>>
        >>> try:
        ...     schema = get_json_schema(UnsupportedType)
        ... except SchemaGenerationError as e:
        ...     print(f"Cannot create schema: {e}")

    Check if a type is supported before use:

        >>> from pydantic import BaseModel
        >>> from insideLLMs.structured import pydantic_to_json_schema
        >>>
        >>> class ValidModel(BaseModel):
        ...     name: str
        >>>
        >>> try:
        ...     schema = pydantic_to_json_schema(ValidModel)
        ...     print("Schema generated successfully")
        ... except SchemaGenerationError as e:
        ...     print(f"Model not supported: {e}")

    See Also
    --------
    pydantic_to_json_schema : Generate schema from Pydantic models.
    dataclass_to_json_schema : Generate schema from dataclasses.
    get_json_schema : Universal schema generator.
    """

    pass


class ParsingError(StructuredOutputError):
    """Exception raised when JSON cannot be extracted or parsed from LLM output.

    This error indicates that the LLM response did not contain valid JSON,
    or the JSON could not be extracted from the surrounding text. The error
    preserves the raw output for debugging and logging purposes.

    Parameters
    ----------
    message : str
        Description of the parsing failure.
    raw_output : str
        The original LLM output that failed to parse.
    attempts : int, optional
        Number of generation attempts made before failing. Default is 1.

    Attributes
    ----------
    raw_output : str
        The original text that could not be parsed.
    attempts : int
        Total number of attempts made (useful when retries are enabled).

    Examples
    --------
    Handle parsing errors and access raw output:

        >>> from insideLLMs.structured import generate_structured, ParsingError
        >>>
        >>> try:
        ...     result = generate_structured(model, "ambiguous input", MyModel)
        ... except ParsingError as e:
        ...     print(f"Parsing failed after {e.attempts} attempts")
        ...     print(f"Raw output was: {e.raw_output[:200]}...")
        ...     # Log for debugging
        ...     with open("failed_output.txt", "w") as f:
        ...         f.write(e.raw_output)

    Implement custom retry logic:

        >>> def extract_with_fallback(model, text, output_type, fallback=None):
        ...     try:
        ...         return generate_structured(model, text, output_type)
        ...     except ParsingError as e:
        ...         print(f"Extraction failed: {e}")
        ...         return fallback

    Check if retry limit was reached:

        >>> try:
        ...     result = generate_structured(model, text, MyModel, max_retries=5)
        ... except ParsingError as e:
        ...     if e.attempts >= 5:
        ...         print("Maximum retries exhausted")
        ...     else:
        ...         print(f"Failed after only {e.attempts} attempts")

    See Also
    --------
    extract_json : Function that extracts JSON from mixed text.
    parse_json : Function that parses extracted JSON.
    ValidationErrorWrapper : Raised when JSON is valid but data is invalid.
    """

    def __init__(self, message: str, raw_output: str, attempts: int = 1):
        super().__init__(message)
        self.raw_output = raw_output
        self.attempts = attempts


class ValidationErrorWrapper(StructuredOutputError):
    """Wrapper exception for Pydantic validation errors.

    This exception is raised when the JSON was successfully extracted and
    parsed, but the data does not conform to the Pydantic model's constraints.
    It wraps the original Pydantic ValidationError for detailed field-level
    error information.

    Parameters
    ----------
    message : str
        Summary of the validation failure.
    validation_error : pydantic.ValidationError
        The original Pydantic validation error with detailed field errors.
    raw_output : str
        The parsed JSON data that failed validation (as string).

    Attributes
    ----------
    validation_error : pydantic.ValidationError
        Access the original Pydantic error for detailed field-level messages.
    raw_output : str
        The data that was parsed but failed validation.

    Examples
    --------
    Handle validation errors with field-level details:

        >>> from pydantic import BaseModel, Field
        >>> from insideLLMs.structured import generate_structured, ValidationErrorWrapper
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int = Field(ge=0, le=150)  # Age must be 0-150
        >>>
        >>> try:
        ...     # LLM might return invalid age
        ...     person = generate_structured(model, "John is -5 years old", Person)
        ... except ValidationErrorWrapper as e:
        ...     print(f"Validation failed: {e}")
        ...     for error in e.validation_error.errors():
        ...         print(f"  Field '{error['loc'][0]}': {error['msg']}")

    Log validation failures for analysis:

        >>> try:
        ...     result = generate_structured(model, text, StrictModel)
        ... except ValidationErrorWrapper as e:
        ...     import json
        ...     log_entry = {
        ...         "error": str(e),
        ...         "raw_data": e.raw_output,
        ...         "field_errors": [
        ...             {"field": err["loc"], "message": err["msg"]}
        ...             for err in e.validation_error.errors()
        ...         ]
        ...     }
        ...     with open("validation_errors.jsonl", "a") as f:
        ...         f.write(json.dumps(log_entry) + "\\n")

    Implement partial extraction on validation failure:

        >>> try:
        ...     result = generate_structured(model, text, StrictModel)
        ... except ValidationErrorWrapper as e:
        ...     # Try parsing with a more lenient model
        ...     import json
        ...     data = json.loads(e.raw_output)
        ...     # Extract whatever fields are valid
        ...     partial = {k: v for k, v in data.items() if k in ["name", "email"]}

    See Also
    --------
    ParsingError : Raised when JSON extraction fails (before validation).
    pydantic.ValidationError : The underlying Pydantic exception.
    parse_to_type : Function that performs validation and instantiation.
    """

    def __init__(self, message: str, validation_error: Any, raw_output: str):
        super().__init__(message)
        self.validation_error = validation_error
        self.raw_output = raw_output


# =============================================================================
# JSON Schema Generation
# =============================================================================


def _python_type_to_json_schema(python_type: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema representation.

    This internal function handles the conversion of Python's built-in types
    and generic types (List, Dict, Optional, Union) to their JSON Schema
    equivalents. It supports recursive conversion for nested types.

    Parameters
    ----------
    python_type : Any
        A Python type annotation. Supported types include:
        - Basic types: str, int, float, bool, bytes, None
        - Generic types: List[T], Dict[K, V], Optional[T], Union[T1, T2, ...]
        - Pydantic models (when Pydantic is available)
        - Any (returns empty schema allowing any value)

    Returns
    -------
    dict[str, Any]
        A JSON Schema dictionary representing the type. The schema follows
        JSON Schema Draft 7 specification.

    Examples
    --------
    Convert basic types:

        >>> _python_type_to_json_schema(str)
        {'type': 'string'}
        >>> _python_type_to_json_schema(int)
        {'type': 'integer'}
        >>> _python_type_to_json_schema(float)
        {'type': 'number'}
        >>> _python_type_to_json_schema(bool)
        {'type': 'boolean'}

    Convert collection types:

        >>> from typing import List, Dict
        >>> _python_type_to_json_schema(List[str])
        {'type': 'array', 'items': {'type': 'string'}}
        >>> _python_type_to_json_schema(Dict[str, int])
        {'type': 'object', 'additionalProperties': {'type': 'integer'}}

    Convert Optional types:

        >>> from typing import Optional
        >>> _python_type_to_json_schema(Optional[str])
        {'anyOf': [{'type': 'string'}, {'type': 'null'}]}

    Convert Union types:

        >>> from typing import Union
        >>> _python_type_to_json_schema(Union[str, int])
        {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}

    Notes
    -----
    - This is an internal function; use `get_json_schema` for public API
    - Unknown types return an empty schema `{}` (allows any value)
    - Pydantic models are delegated to `pydantic_to_json_schema`

    See Also
    --------
    get_json_schema : Public API for schema generation.
    pydantic_to_json_schema : Schema generation for Pydantic models.
    dataclass_to_json_schema : Schema generation for dataclasses.
    """
    # Handle None type
    if python_type is type(None):
        return {"type": "null"}

    # Handle basic types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        bytes: {"type": "string", "format": "byte"},
    }

    if python_type in type_mapping:
        return type_mapping[python_type]

    # Handle generic types
    origin = get_origin(python_type)
    args = get_args(python_type)

    if origin is list or origin is list:
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": _python_type_to_json_schema(item_type),
        }

    if origin is dict or origin is dict:
        value_type = args[1] if len(args) > 1 else Any
        return {
            "type": "object",
            "additionalProperties": _python_type_to_json_schema(value_type),
        }

    if origin is Union:
        # Handle Optional (Union[X, None])
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1 and type(None) in args:
            # This is Optional[X]
            schema = _python_type_to_json_schema(non_none_args[0])
            return {"anyOf": [schema, {"type": "null"}]}
        else:
            return {"anyOf": [_python_type_to_json_schema(a) for a in args]}

    # Handle Pydantic models
    if PYDANTIC_AVAILABLE and isinstance(python_type, type) and issubclass(python_type, BaseModel):
        return pydantic_to_json_schema(python_type)

    # Default to any
    return {}


def pydantic_to_json_schema(model: type["BaseModel"]) -> dict[str, Any]:
    """Convert a Pydantic model class to a JSON Schema dictionary.

    This function generates a complete JSON Schema from a Pydantic model,
    including all field types, constraints, descriptions, and nested models.
    It supports both Pydantic v1 and v2, automatically detecting the version.

    Parameters
    ----------
    model : type[BaseModel]
        A Pydantic model class (not an instance). The class should inherit
        from pydantic.BaseModel.

    Returns
    -------
    dict[str, Any]
        A complete JSON Schema dictionary representing the model structure.
        Includes properties, required fields, type constraints, and any
        field validators defined in the model.

    Raises
    ------
    ImportError
        If Pydantic is not installed.
    SchemaGenerationError
        If schema generation fails due to unsupported types or model issues.

    Examples
    --------
    Generate schema from a simple model:

        >>> from pydantic import BaseModel
        >>> from insideLLMs.structured import pydantic_to_json_schema
        >>>
        >>> class User(BaseModel):
        ...     name: str
        ...     email: str
        ...     age: int
        >>>
        >>> schema = pydantic_to_json_schema(User)
        >>> print(schema["properties"].keys())
        dict_keys(['name', 'email', 'age'])
        >>> print(schema["required"])
        ['name', 'email', 'age']

    Generate schema with nested models:

        >>> class Address(BaseModel):
        ...     street: str
        ...     city: str
        ...     country: str = "USA"
        >>>
        >>> class Company(BaseModel):
        ...     name: str
        ...     address: Address
        ...     employees: list[str]
        >>>
        >>> schema = pydantic_to_json_schema(Company)
        >>> # Nested model is included in definitions
        >>> print("Address" in str(schema))
        True

    Generate schema with field constraints:

        >>> from pydantic import Field
        >>>
        >>> class Product(BaseModel):
        ...     name: str = Field(min_length=1, max_length=100)
        ...     price: float = Field(gt=0)
        ...     quantity: int = Field(ge=0, default=0)
        >>>
        >>> schema = pydantic_to_json_schema(Product)
        >>> print(schema["properties"]["price"])
        {'exclusiveMinimum': 0, 'type': 'number'}

    Use schema for LLM prompting:

        >>> import json
        >>> schema = pydantic_to_json_schema(User)
        >>> prompt = f"Return JSON matching this schema:\\n{json.dumps(schema, indent=2)}"

    Notes
    -----
    - Pydantic v2 uses `model_json_schema()` method
    - Pydantic v1 uses `schema()` method
    - Complex validators may not be fully represented in JSON Schema
    - Circular references are handled via $ref pointers

    See Also
    --------
    dataclass_to_json_schema : Schema generation for dataclasses.
    get_json_schema : Universal schema generator (auto-detects type).
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "Pydantic is required for structured outputs. Install with: pip install pydantic"
        )

    # Use Pydantic's built-in schema generation
    try:
        # Pydantic v2
        if hasattr(model, "model_json_schema"):
            return model.model_json_schema()
        # Pydantic v1
        elif hasattr(model, "schema"):
            return model.schema()
        else:
            raise SchemaGenerationError(f"Cannot generate schema for {model}")
    except Exception as e:
        raise SchemaGenerationError(f"Failed to generate schema for {model}: {e}") from e


def dataclass_to_json_schema(dc: type) -> dict[str, Any]:
    """Convert a Python dataclass to a JSON Schema dictionary.

    This function generates a JSON Schema from a dataclass definition,
    including field types and required/optional status based on default values.

    Parameters
    ----------
    dc : type
        A Python dataclass type (decorated with @dataclass). Must be a class,
        not an instance.

    Returns
    -------
    dict[str, Any]
        A JSON Schema dictionary with the following structure:
        - "type": "object"
        - "properties": dict mapping field names to their schemas
        - "required": list of field names without defaults

    Raises
    ------
    SchemaGenerationError
        If the provided type is not a dataclass.

    Examples
    --------
    Generate schema from a simple dataclass:

        >>> from dataclasses import dataclass
        >>> from insideLLMs.structured import dataclass_to_json_schema
        >>>
        >>> @dataclass
        ... class Event:
        ...     title: str
        ...     date: str
        ...     attendees: int
        >>>
        >>> schema = dataclass_to_json_schema(Event)
        >>> print(schema)
        {
            'type': 'object',
            'properties': {
                'title': {'type': 'string'},
                'date': {'type': 'string'},
                'attendees': {'type': 'integer'}
            },
            'required': ['title', 'date', 'attendees']
        }

    Dataclass with optional fields (defaults):

        >>> from dataclasses import dataclass, field
        >>> from typing import List
        >>>
        >>> @dataclass
        ... class Meeting:
        ...     title: str
        ...     organizer: str
        ...     location: str = "Conference Room A"
        ...     participants: List[str] = field(default_factory=list)
        >>>
        >>> schema = dataclass_to_json_schema(Meeting)
        >>> print(schema["required"])
        ['title', 'organizer']  # Only fields without defaults

    Dataclass with complex nested types:

        >>> from dataclasses import dataclass
        >>> from typing import Optional, Dict
        >>>
        >>> @dataclass
        ... class Config:
        ...     name: str
        ...     settings: Dict[str, str]
        ...     description: Optional[str] = None
        >>>
        >>> schema = dataclass_to_json_schema(Config)
        >>> print(schema["properties"]["settings"])
        {'type': 'object', 'additionalProperties': {'type': 'string'}}

    Use schema for extraction:

        >>> from insideLLMs.structured import generate_structured
        >>>
        >>> @dataclass
        ... class Person:
        ...     name: str
        ...     age: int
        >>>
        >>> # Dataclass works directly with generate_structured
        >>> person = generate_structured(model, "John is 30 years old", Person)

    Notes
    -----
    - Fields without defaults are marked as required
    - Fields with default or default_factory are optional
    - Nested dataclasses are converted recursively
    - Type annotations are converted using `_python_type_to_json_schema`

    See Also
    --------
    pydantic_to_json_schema : Schema generation for Pydantic models.
    get_json_schema : Universal schema generator (auto-detects type).
    """
    import dataclasses

    if not dataclasses.is_dataclass(dc):
        raise SchemaGenerationError(f"{dc} is not a dataclass")

    properties = {}
    required = []

    for field in dataclasses.fields(dc):
        properties[field.name] = _python_type_to_json_schema(field.type)

        # Check if field has default
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            required.append(field.name)

    schema = {
        "type": "object",
        "properties": properties,
    }

    if required:
        schema["required"] = required

    return schema


def get_json_schema(output_type: type) -> dict[str, Any]:
    """Generate a JSON Schema for any supported Python type.

    This is the primary public API for schema generation. It automatically
    detects the type (Pydantic model, dataclass, or basic Python type) and
    delegates to the appropriate schema generator.

    Parameters
    ----------
    output_type : type
        Any supported Python type:
        - Pydantic BaseModel subclass
        - Python dataclass
        - Basic types (str, int, float, bool)
        - Generic types (List, Dict, Optional, Union)

    Returns
    -------
    dict[str, Any]
        A JSON Schema dictionary suitable for LLM prompting. The schema
        format follows JSON Schema specification.

    Examples
    --------
    Schema for Pydantic model:

        >>> from pydantic import BaseModel
        >>> from insideLLMs.structured import get_json_schema
        >>>
        >>> class User(BaseModel):
        ...     name: str
        ...     email: str
        >>>
        >>> schema = get_json_schema(User)
        >>> print(schema["type"])
        'object'
        >>> print(list(schema["properties"].keys()))
        ['name', 'email']

    Schema for dataclass:

        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class Point:
        ...     x: float
        ...     y: float
        >>>
        >>> schema = get_json_schema(Point)
        >>> print(schema["properties"]["x"])
        {'type': 'number'}

    Schema for basic types:

        >>> schema = get_json_schema(str)
        >>> print(schema)
        {'type': 'string'}
        >>>
        >>> from typing import List
        >>> schema = get_json_schema(List[int])
        >>> print(schema)
        {'type': 'array', 'items': {'type': 'integer'}}

    Schema for complex nested types:

        >>> from typing import Dict, Optional
        >>> from pydantic import BaseModel
        >>>
        >>> class Address(BaseModel):
        ...     street: str
        ...     city: str
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     address: Optional[Address] = None
        ...     tags: Dict[str, str] = {}
        >>>
        >>> schema = get_json_schema(Person)
        >>> # Full nested schema with definitions

    Use schema in custom prompts:

        >>> import json
        >>> schema = get_json_schema(User)
        >>> prompt = f'''
        ... Extract user information as JSON matching this schema:
        ... {json.dumps(schema, indent=2)}
        ... '''

    Notes
    -----
    - Pydantic models are preferred for complex schemas
    - Dataclasses work well for simple structures
    - Basic types return minimal schemas
    - Unknown types return empty schema `{}`

    See Also
    --------
    pydantic_to_json_schema : Direct Pydantic schema generation.
    dataclass_to_json_schema : Direct dataclass schema generation.
    """
    import dataclasses

    if PYDANTIC_AVAILABLE and isinstance(output_type, type) and issubclass(output_type, BaseModel):
        return pydantic_to_json_schema(output_type)
    elif dataclasses.is_dataclass(output_type):
        return dataclass_to_json_schema(output_type)
    else:
        return _python_type_to_json_schema(output_type)


# =============================================================================
# JSON Extraction
# =============================================================================


def extract_json(text: str) -> str:
    """Extract valid JSON from text that may contain markdown or other content.

    This function intelligently extracts JSON from LLM responses that often
    include markdown formatting, explanatory text, or code blocks. It tries
    multiple extraction strategies in order of specificity.

    Parameters
    ----------
    text : str
        Text potentially containing JSON. Can be:
        - Pure JSON
        - JSON in markdown code blocks (```json ... ``` or ``` ... ```)
        - JSON embedded in explanatory text
        - Multiple JSON objects (returns first valid one)

    Returns
    -------
    str
        The extracted JSON string, ready for parsing with json.loads().

    Raises
    ------
    ParsingError
        If no valid JSON can be found in the text.

    Examples
    --------
    Extract from markdown code block:

        >>> from insideLLMs.structured import extract_json
        >>>
        >>> text = '''
        ... Here's the extracted data:
        ... ```json
        ... {"name": "John", "age": 30}
        ... ```
        ... '''
        >>> json_str = extract_json(text)
        >>> print(json_str)
        {"name": "John", "age": 30}

    Extract from plain JSON:

        >>> text = '{"product": "Widget", "price": 29.99}'
        >>> json_str = extract_json(text)
        >>> print(json_str)
        {"product": "Widget", "price": 29.99}

    Extract from mixed content:

        >>> text = '''
        ... I found the following information:
        ... {"title": "Meeting", "date": "2024-01-15"}
        ... Let me know if you need more details.
        ... '''
        >>> json_str = extract_json(text)
        >>> print(json_str)
        {"title": "Meeting", "date": "2024-01-15"}

    Extract JSON array:

        >>> text = 'The items are: [1, 2, 3, 4, 5]'
        >>> json_str = extract_json(text)
        >>> print(json_str)
        [1, 2, 3, 4, 5]

    Handle extraction failure:

        >>> text = "No JSON here, just plain text."
        >>> try:
        ...     extract_json(text)
        ... except ParsingError as e:
        ...     print(f"Failed: {e}")
        ...     print(f"Original text: {e.raw_output}")

    Notes
    -----
    Extraction strategies (in order):

    1. JSON in ```json ... ``` code blocks
    2. JSON in ``` ... ``` code blocks (any language)
    3. JSON objects {...} anywhere in text
    4. JSON arrays [...] anywhere in text
    5. Entire text as JSON

    For each strategy, the function validates the extracted content by
    attempting to parse it before returning.

    See Also
    --------
    parse_json : Combines extraction and parsing in one call.
    insideLLMs.structured_extraction.extract_json : Version with ExtractionResult metadata.
    """
    # Try to find JSON in code blocks
    code_block_patterns = [
        r"```json\s*([\s\S]*?)```",
        r"```\s*([\s\S]*?)```",
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, text)
        if match:
            json_str = match.group(1).strip()
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                continue

    # Try to find raw JSON object or array
    json_patterns = [
        r"(\{[\s\S]*\})",  # Object
        r"(\[[\s\S]*\])",  # Array
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue

    # Try the entire text
    try:
        json.loads(text.strip())
        return text.strip()
    except json.JSONDecodeError:
        pass

    raise ParsingError("Could not extract valid JSON from output", text)


def parse_json(text: str) -> Any:
    """Extract and parse JSON from text in a single operation.

    This is a convenience function that combines `extract_json` and
    `json.loads` into a single call. Use this when you need the parsed
    Python object rather than the raw JSON string.

    Parameters
    ----------
    text : str
        Text containing JSON, potentially with markdown or other content.

    Returns
    -------
    Any
        The parsed JSON value. Can be:
        - dict for JSON objects
        - list for JSON arrays
        - str, int, float, bool, None for primitives

    Raises
    ------
    ParsingError
        If no valid JSON can be found or extracted.

    Examples
    --------
    Parse JSON object from text:

        >>> from insideLLMs.structured import parse_json
        >>>
        >>> text = '```json\\n{"name": "Alice", "age": 25}\\n```'
        >>> data = parse_json(text)
        >>> print(data["name"])
        Alice
        >>> print(type(data))
        <class 'dict'>

    Parse JSON array:

        >>> text = "The numbers are: [1, 2, 3, 4, 5]"
        >>> data = parse_json(text)
        >>> print(sum(data))
        15

    Parse nested structures:

        >>> text = '''
        ... {
        ...     "user": {"name": "Bob", "id": 123},
        ...     "items": ["apple", "banana"],
        ...     "active": true
        ... }
        ... '''
        >>> data = parse_json(text)
        >>> print(data["user"]["name"])
        Bob
        >>> print(data["items"])
        ['apple', 'banana']

    Error handling:

        >>> try:
        ...     data = parse_json("no json here")
        ... except ParsingError as e:
        ...     print("Parsing failed")

    Notes
    -----
    - Uses `extract_json` internally for robust extraction
    - Returns native Python types (dict, list, etc.)
    - For typed parsing into models, use `parse_to_type` instead

    See Also
    --------
    extract_json : Extract JSON string without parsing.
    parse_to_type : Parse into specific type (Pydantic, dataclass).
    """
    json_str = extract_json(text)
    return json.loads(json_str)


# =============================================================================
# Output Parsing
# =============================================================================


def parse_to_type(data: Any, output_type: type[T]) -> T:
    """Parse and validate data into a specific Python type.

    This function converts parsed JSON data (typically a dict) into instances
    of Pydantic models, dataclasses, or other Python types. It handles
    validation and provides detailed error messages on failure.

    Parameters
    ----------
    data : Any
        Data to parse, typically a dict from JSON parsing. Can also be
        a list, primitive value, or nested structure.
    output_type : type[T]
        The target type to convert to. Supported types:
        - Pydantic BaseModel subclasses (v1 and v2)
        - Python dataclasses
        - Basic types (dict, list, str, int, float, bool)
        - Any class with compatible __init__ signature

    Returns
    -------
    T
        An instance of output_type populated with data from input.

    Raises
    ------
    ValidationErrorWrapper
        If Pydantic validation fails (invalid data for model constraints).
    ParsingError
        If type conversion fails (incompatible data structure).

    Examples
    --------
    Parse into Pydantic model:

        >>> from pydantic import BaseModel
        >>> from insideLLMs.structured import parse_to_type
        >>>
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        ...     email: str
        >>>
        >>> data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
        >>> user = parse_to_type(data, User)
        >>> print(user.name)
        Alice
        >>> print(type(user))
        <class 'User'>

    Parse into dataclass:

        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class Point:
        ...     x: float
        ...     y: float
        ...     label: str = ""
        >>>
        >>> data = {"x": 10.5, "y": 20.3, "label": "origin"}
        >>> point = parse_to_type(data, Point)
        >>> print(point.x, point.y)
        10.5 20.3

    Parse into basic types:

        >>> # Dict passthrough
        >>> result = parse_to_type({"a": 1, "b": 2}, dict)
        >>> print(result)
        {'a': 1, 'b': 2}
        >>>
        >>> # List passthrough
        >>> result = parse_to_type([1, 2, 3], list)
        >>> print(result)
        [1, 2, 3]
        >>>
        >>> # Type conversion
        >>> result = parse_to_type("42", int)
        >>> print(result)
        42

    Handle validation errors:

        >>> from pydantic import BaseModel, Field
        >>>
        >>> class Product(BaseModel):
        ...     name: str
        ...     price: float = Field(gt=0)
        >>>
        >>> try:
        ...     product = parse_to_type({"name": "Widget", "price": -10}, Product)
        ... except ValidationErrorWrapper as e:
        ...     print(f"Validation failed: {e.validation_error}")
        ...     for err in e.validation_error.errors():
        ...         print(f"  {err['loc']}: {err['msg']}")

    Parse nested structures:

        >>> class Address(BaseModel):
        ...     street: str
        ...     city: str
        >>>
        >>> class Company(BaseModel):
        ...     name: str
        ...     address: Address
        >>>
        >>> data = {
        ...     "name": "Acme Inc",
        ...     "address": {"street": "123 Main St", "city": "Boston"}
        ... }
        >>> company = parse_to_type(data, Company)
        >>> print(company.address.city)
        Boston

    Notes
    -----
    - Pydantic v2 uses `model_validate`, v1 uses `parse_obj`
    - Dataclasses are instantiated via keyword arguments
    - Unknown types attempt direct instantiation
    - ValidationErrorWrapper preserves the original Pydantic error

    See Also
    --------
    parse_json : Extract and parse JSON from text.
    generate_structured : Full extraction pipeline with LLM.
    """
    import dataclasses

    if PYDANTIC_AVAILABLE and isinstance(output_type, type) and issubclass(output_type, BaseModel):
        # Pydantic model
        try:
            # Pydantic v2
            if hasattr(output_type, "model_validate"):
                return output_type.model_validate(data)
            # Pydantic v1
            else:
                return output_type.parse_obj(data)
        except ValidationError as e:
            raise ValidationErrorWrapper(
                f"Validation failed for {output_type.__name__}", e, str(data)
            ) from e

    elif dataclasses.is_dataclass(output_type):
        # Dataclass
        try:
            return output_type(**data)
        except TypeError as e:
            raise ParsingError(f"Failed to create {output_type.__name__}: {e}", str(data)) from e

    elif output_type in (dict, dict) or output_type in (list, list):
        return data

    elif output_type in (str, int, float, bool):
        return output_type(data)

    else:
        # Try direct instantiation
        try:
            return output_type(**data) if isinstance(data, dict) else output_type(data)
        except Exception as e:
            raise ParsingError(f"Failed to create {output_type}: {e}", str(data)) from e


# =============================================================================
# Prompt Templates
# =============================================================================


STRUCTURED_OUTPUT_SYSTEM_PROMPT = """You are a helpful assistant that extracts structured information from text.
Always respond with valid JSON that matches the specified schema exactly.
Do not include any explanation or text outside the JSON."""

STRUCTURED_OUTPUT_TEMPLATE = """Extract the requested information and return it as JSON.

{instructions}

Output Schema:
```json
{schema}
```

{input_section}

Respond with ONLY the JSON object, no other text."""

STRUCTURED_OUTPUT_TEMPLATE_WITH_EXAMPLES = """Extract the requested information and return it as JSON.

{instructions}

Output Schema:
```json
{schema}
```

Examples:
{examples}

{input_section}

Respond with ONLY the JSON object, no other text."""


# =============================================================================
# Structured Output Generator
# =============================================================================


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output generation.

    This dataclass holds all configuration options for the
    `StructuredOutputGenerator`, controlling retry behavior, temperature,
    prompting strategy, and few-shot examples.

    Parameters
    ----------
    max_retries : int, default=3
        Maximum number of retry attempts when parsing fails. Each retry
        includes error feedback in the prompt to help the LLM correct
        its output.
    temperature : float, default=0.0
        Temperature for LLM generation. Use 0.0 for deterministic outputs
        (recommended for extraction), higher values for more variety.
    include_schema_in_prompt : bool, default=True
        Whether to include the full JSON schema in the prompt. Set to
        False if providing examples that sufficiently demonstrate format.
    instructions : str, default=""
        Additional extraction instructions to include in the prompt.
        Use this to guide the LLM on specific extraction behavior.
    examples : list[dict[str, Any]] or None, default=None
        Optional list of few-shot examples. Each example should have
        "input" and "output" keys showing input text and expected JSON.

    Attributes
    ----------
    max_retries : int
        Maximum retry attempts on parsing failure.
    temperature : float
        Generation temperature (0.0 = deterministic).
    include_schema_in_prompt : bool
        Whether to include JSON schema.
    instructions : str
        Additional extraction instructions.
    examples : list[dict[str, Any]] or None
        Few-shot examples for prompting.

    Examples
    --------
    Default configuration:

        >>> from insideLLMs.structured import StructuredOutputConfig
        >>>
        >>> config = StructuredOutputConfig()
        >>> print(config.max_retries)
        3
        >>> print(config.temperature)
        0.0

    Custom retry and temperature:

        >>> config = StructuredOutputConfig(
        ...     max_retries=5,
        ...     temperature=0.1
        ... )

    Add extraction instructions:

        >>> config = StructuredOutputConfig(
        ...     instructions='''
        ...     Extract contact information from business cards.
        ...     - Use "N/A" for missing fields
        ...     - Normalize phone numbers to (XXX) XXX-XXXX format
        ...     - Extract only professional email addresses
        ...     '''
        ... )

    Few-shot prompting with examples:

        >>> config = StructuredOutputConfig(
        ...     examples=[
        ...         {
        ...             "input": "John Smith, CEO at Acme Corp, john@acme.com",
        ...             "output": {
        ...                 "name": "John Smith",
        ...                 "title": "CEO",
        ...                 "company": "Acme Corp",
        ...                 "email": "john@acme.com"
        ...             }
        ...         },
        ...         {
        ...             "input": "Jane Doe - Product Manager, jane.doe@startup.io",
        ...             "output": {
        ...                 "name": "Jane Doe",
        ...                 "title": "Product Manager",
        ...                 "company": "startup.io",
        ...                 "email": "jane.doe@startup.io"
        ...             }
        ...         }
        ...     ],
        ...     instructions="Extract contact information from text."
        ... )

    Disable schema in prompt (when examples are sufficient):

        >>> config = StructuredOutputConfig(
        ...     include_schema_in_prompt=False,
        ...     examples=[...],  # Comprehensive examples
        ... )

    Use with generator:

        >>> from insideLLMs.structured import StructuredOutputGenerator
        >>>
        >>> config = StructuredOutputConfig(
        ...     max_retries=3,
        ...     instructions="Extract person details accurately."
        ... )
        >>> generator = StructuredOutputGenerator(model, Person, config)
        >>> person = generator.generate("John is 30 years old")

    Notes
    -----
    - Temperature 0.0 is recommended for consistent extraction
    - More retries increase reliability but also latency and cost
    - Few-shot examples significantly improve extraction quality
    - Combine instructions with examples for best results

    See Also
    --------
    StructuredOutputGenerator : Uses this config for generation.
    generate_structured : Convenience function that creates config internally.
    """

    max_retries: int = 3
    temperature: float = 0.0
    include_schema_in_prompt: bool = True
    instructions: str = ""
    examples: Optional[list[dict[str, Any]]] = None


class StructuredOutputGenerator(Generic[T]):
    """Reusable generator for extracting structured data from LLM responses.

    This class encapsulates the full extraction pipeline: schema generation,
    prompt building, LLM invocation, JSON parsing, and validation. It is
    designed for scenarios where you need to extract the same type of data
    from multiple inputs.

    Parameters
    ----------
    model : Model
        An LLM model instance (e.g., OpenAIModel, AnthropicModel).
        Must support either `chat()` or `generate()` method.
    output_type : type[T]
        The target type for extraction. Can be a Pydantic model,
        dataclass, or other supported type.
    config : StructuredOutputConfig or None, default=None
        Configuration for generation. If None, uses default config.

    Attributes
    ----------
    model : Model
        The LLM model used for generation.
    output_type : type[T]
        Target type for parsed outputs.
    config : StructuredOutputConfig
        Generation configuration.

    Examples
    --------
    Basic usage with Pydantic model:

        >>> from pydantic import BaseModel
        >>> from insideLLMs.structured import StructuredOutputGenerator
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...     occupation: str
        >>>
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> generator = StructuredOutputGenerator(model, Person)
        >>>
        >>> # Extract from multiple texts
        >>> person1 = generator.generate("John is a 30-year-old engineer")
        >>> person2 = generator.generate("Jane, 25, works as a designer")
        >>> print(person1.name, person2.name)
        John Jane

    With custom configuration:

        >>> from insideLLMs.structured import StructuredOutputConfig
        >>>
        >>> config = StructuredOutputConfig(
        ...     max_retries=5,
        ...     temperature=0.1,
        ...     instructions="Extract person information. Use 'Unknown' for missing fields."
        ... )
        >>> generator = StructuredOutputGenerator(model, Person, config)
        >>> person = generator.generate("Some guy named Bob")

    With few-shot examples:

        >>> config = StructuredOutputConfig(
        ...     examples=[
        ...         {
        ...             "input": "Alice is 28 and teaches math",
        ...             "output": {"name": "Alice", "age": 28, "occupation": "Teacher"}
        ...         },
        ...         {
        ...             "input": "Bob, software dev, turning 35",
        ...             "output": {"name": "Bob", "age": 35, "occupation": "Software Developer"}
        ...         }
        ...     ]
        ... )
        >>> generator = StructuredOutputGenerator(model, Person, config)
        >>> person = generator.generate("Carol, 42, runs a bakery")
        >>> print(person.occupation)
        Baker

    Batch processing:

        >>> texts = [
        ...     "Alice, 28, Engineer",
        ...     "Bob, 35, Manager",
        ...     "Carol, 42, Director"
        ... ]
        >>> people = generator.generate_batch(texts)
        >>> for p in people:
        ...     print(f"{p.name}: {p.age}")
        Alice: 28
        Bob: 35
        Carol: 42

    Complex nested extraction:

        >>> class Address(BaseModel):
        ...     street: str
        ...     city: str
        ...     zip_code: str
        >>>
        >>> class Company(BaseModel):
        ...     name: str
        ...     address: Address
        ...     employee_count: int
        >>>
        >>> generator = StructuredOutputGenerator(model, Company)
        >>> company = generator.generate(
        ...     "Acme Inc is located at 123 Main St, Boston 02101, with 500 employees"
        ... )
        >>> print(company.address.city)
        Boston

    Notes
    -----
    - The generator caches the JSON schema for efficiency
    - Supports both chat and completion APIs
    - Retry logic includes error feedback for self-correction
    - For one-off extractions, use `generate_structured()` instead

    See Also
    --------
    generate_structured : Convenience function for one-off extraction.
    create_structured_generator : Factory function for creating generators.
    StructuredOutputConfig : Configuration options.
    """

    def __init__(
        self,
        model: "Model",
        output_type: type[T],
        config: Optional[StructuredOutputConfig] = None,
    ):
        """Initialize the structured output generator.

        Parameters
        ----------
        model : Model
            LLM model instance with `chat()` or `generate()` method.
        output_type : type[T]
            Target Pydantic model, dataclass, or type for output.
        config : StructuredOutputConfig or None, optional
            Generation configuration. Uses defaults if None.

        Examples
        --------
        Initialize with defaults:

            >>> generator = StructuredOutputGenerator(model, Person)

        Initialize with custom config:

            >>> config = StructuredOutputConfig(max_retries=5)
            >>> generator = StructuredOutputGenerator(model, Person, config)
        """
        self.model = model
        self.output_type = output_type
        self.config = config or StructuredOutputConfig()
        self._schema = get_json_schema(output_type)

    def _build_prompt(self, input_text: str) -> str:
        """Build the extraction prompt from input text.

        Constructs a complete prompt including schema, instructions, and
        optionally few-shot examples. This is an internal method used by
        `generate()`.

        Parameters
        ----------
        input_text : str
            The text to extract information from.

        Returns
        -------
        str
            Complete prompt string ready to send to the model.

        Examples
        --------
        Build prompt (internal use):

            >>> generator = StructuredOutputGenerator(model, Person)
            >>> prompt = generator._build_prompt("John is 30 years old")
            >>> print(prompt[:100])
            Extract the requested information and return it as JSON...
        """
        schema_str = json.dumps(self._schema, indent=2)

        if self.config.examples:
            examples_str = "\n\n".join(
                f"Input: {ex.get('input', '')}\nOutput: {json.dumps(ex.get('output', {}))}"
                for ex in self.config.examples
            )
            return STRUCTURED_OUTPUT_TEMPLATE_WITH_EXAMPLES.format(
                instructions=self.config.instructions or "Extract the information from the text.",
                schema=schema_str if self.config.include_schema_in_prompt else "See schema above",
                examples=examples_str,
                input_section=f"Input:\n{input_text}",
            )
        else:
            return STRUCTURED_OUTPUT_TEMPLATE.format(
                instructions=self.config.instructions or "Extract the information from the text.",
                schema=schema_str if self.config.include_schema_in_prompt else "See schema above",
                input_section=f"Input:\n{input_text}",
            )

    def generate(self, input_text: str, **model_kwargs: Any) -> T:
        """Generate structured output from input text.

        This is the primary method for extracting structured data. It builds
        the prompt, calls the LLM, parses the response, and validates the
        result. On failure, it retries with error feedback.

        Parameters
        ----------
        input_text : str
            The text to extract information from. Can be any length,
            but very long texts may hit model context limits.
        **model_kwargs : Any
            Additional keyword arguments passed to the model's
            `chat()` or `generate()` method. Common options include
            `max_tokens`, `stop`, etc.

        Returns
        -------
        T
            An instance of the output_type populated with extracted data.

        Raises
        ------
        ParsingError
            If parsing fails after all retry attempts. The exception
            includes the raw output and number of attempts made.

        Examples
        --------
        Basic extraction:

            >>> generator = StructuredOutputGenerator(model, Person)
            >>> person = generator.generate("John is a 30-year-old engineer")
            >>> print(person.name, person.age)
            John 30

        With model kwargs:

            >>> person = generator.generate(
            ...     "Extract: Alice, 25, Designer",
            ...     max_tokens=500,
            ...     stop=["\\n\\n"]
            ... )

        Handle extraction failure:

            >>> try:
            ...     result = generator.generate("ambiguous text")
            ... except ParsingError as e:
            ...     print(f"Failed after {e.attempts} attempts")
            ...     print(f"Last response: {e.raw_output[:100]}...")

        Extract from longer text:

            >>> bio = '''
            ... John Smith is a software engineer who has been working in the
            ... tech industry for over 10 years. He started his career at a
            ... small startup and is now 35 years old. He specializes in
            ... machine learning and data engineering.
            ... '''
            >>> person = generator.generate(bio)
            >>> print(person.occupation)
            Software Engineer

        Notes
        -----
        - Retry logic adds error feedback to help LLM self-correct
        - Each retry is a fresh API call (not a continuation)
        - Temperature from config is used unless overridden in kwargs
        """
        prompt = self._build_prompt(input_text)
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # Generate response
                if hasattr(self.model, "chat"):
                    messages = [
                        {"role": "system", "content": STRUCTURED_OUTPUT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                    response = self.model.chat(
                        messages, temperature=self.config.temperature, **model_kwargs
                    )
                else:
                    full_prompt = f"{STRUCTURED_OUTPUT_SYSTEM_PROMPT}\n\n{prompt}"
                    response = self.model.generate(
                        full_prompt, temperature=self.config.temperature, **model_kwargs
                    )

                # Parse response
                data = parse_json(response)
                return parse_to_type(data, self.output_type)

            except (ParsingError, ValidationErrorWrapper) as e:
                last_error = e
                # Add context for retry
                if attempt < self.config.max_retries - 1:
                    prompt = self._build_prompt(input_text)
                    prompt += f"\n\nPrevious attempt failed with error: {str(e)}\nPlease try again with valid JSON."

        raise ParsingError(
            f"Failed to generate valid structured output after {self.config.max_retries} attempts: {last_error}",
            getattr(last_error, "raw_output", ""),
            self.config.max_retries,
        )

    def generate_batch(
        self,
        inputs: list[str],
        **model_kwargs: Any,
    ) -> list[T]:
        """Generate structured outputs for multiple inputs.

        Processes a list of input texts sequentially, extracting structured
        data from each. This is a convenience method for batch processing.

        Parameters
        ----------
        inputs : list[str]
            List of text strings to extract from.
        **model_kwargs : Any
            Additional arguments passed to the model for each call.

        Returns
        -------
        list[T]
            List of parsed output instances, one per input.

        Raises
        ------
        ParsingError
            If any extraction fails after retries. The batch stops at
            the first failure.

        Examples
        --------
        Process multiple texts:

            >>> generator = StructuredOutputGenerator(model, Person)
            >>> texts = [
            ...     "Alice is 28 years old and works as a teacher",
            ...     "Bob, 35, is a software engineer",
            ...     "Carol is a 42-year-old manager"
            ... ]
            >>> people = generator.generate_batch(texts)
            >>> for person in people:
            ...     print(f"{person.name}: {person.age}, {person.occupation}")
            Alice: 28, Teacher
            Bob: 35, Software Engineer
            Carol: 42, Manager

        Batch extraction from documents:

            >>> class Article(BaseModel):
            ...     title: str
            ...     author: str
            ...     summary: str
            >>>
            >>> generator = StructuredOutputGenerator(model, Article)
            >>> articles = generator.generate_batch(document_texts)

        With custom model parameters:

            >>> people = generator.generate_batch(
            ...     texts,
            ...     max_tokens=200,
            ...     temperature=0.1
            ... )

        Notes
        -----
        - Processes inputs sequentially (not parallel)
        - Each input gets full retry logic
        - For progress tracking, use `batch_extract()` instead
        - Consider `batch_extract()` for error handling per-item

        See Also
        --------
        batch_extract : Batch processing with progress and error handling.
        generate : Single input extraction.
        """
        return [self.generate(text, **model_kwargs) for text in inputs]


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_structured(
    model: "Model",
    input_text: str,
    output_type: type[T],
    instructions: str = "",
    max_retries: int = 3,
    examples: Optional[list[dict[str, Any]]] = None,
    **model_kwargs: Any,
) -> T:
    """Generate structured output from text using an LLM.

    This is the primary convenience function for one-off structured extraction.
    It creates a generator internally, so for repeated extractions of the same
    type, use `create_structured_generator()` instead.

    Parameters
    ----------
    model : Model
        LLM model instance (e.g., OpenAIModel, AnthropicModel).
        Must support either `chat()` or `generate()` method.
    input_text : str
        Text to extract information from.
    output_type : type[T]
        Target type for the extracted data. Can be:
        - Pydantic BaseModel subclass
        - Python dataclass
        - Basic types (dict, list, str, int, float, bool)
    instructions : str, default=""
        Additional instructions to guide extraction behavior.
    max_retries : int, default=3
        Maximum number of retry attempts on parsing failure.
    examples : list[dict[str, Any]] or None, default=None
        Optional few-shot examples. Each example should be a dict
        with "input" and "output" keys.
    **model_kwargs : Any
        Additional arguments passed to the model's generation method.

    Returns
    -------
    T
        An instance of output_type populated with extracted data.

    Raises
    ------
    ParsingError
        If parsing fails after all retry attempts.
    ValidationErrorWrapper
        If parsed data fails Pydantic validation.

    Examples
    --------
    Basic extraction:

        >>> from pydantic import BaseModel
        >>> from insideLLMs.structured import generate_structured
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...     occupation: str
        >>>
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> person = generate_structured(
        ...     model,
        ...     "John is a 30-year-old software engineer",
        ...     Person
        ... )
        >>> print(person.name)
        John

    With custom instructions:

        >>> person = generate_structured(
        ...     model,
        ...     "Someone named Bob works here",
        ...     Person,
        ...     instructions="Use 'Unknown' for missing fields and 0 for unknown age"
        ... )
        >>> print(person.age)
        0

    With few-shot examples:

        >>> examples = [
        ...     {
        ...         "input": "Alice teaches math",
        ...         "output": {"name": "Alice", "age": 0, "occupation": "Teacher"}
        ...     }
        ... ]
        >>> person = generate_structured(
        ...     model,
        ...     "Bob drives trucks",
        ...     Person,
        ...     examples=examples
        ... )

    Extract complex nested data:

        >>> class Address(BaseModel):
        ...     city: str
        ...     country: str
        >>>
        >>> class Company(BaseModel):
        ...     name: str
        ...     address: Address
        >>>
        >>> company = generate_structured(
        ...     model,
        ...     "Acme Inc is headquartered in Boston, USA",
        ...     Company
        ... )
        >>> print(company.address.city)
        Boston

    Using dataclass instead of Pydantic:

        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class Event:
        ...     title: str
        ...     date: str
        >>>
        >>> event = generate_structured(
        ...     model,
        ...     "Team meeting on Friday at 2pm",
        ...     Event
        ... )

    Notes
    -----
    - Creates a new generator for each call (use `create_structured_generator`
      for repeated extractions of the same type)
    - Temperature defaults to 0.0 for deterministic outputs
    - For richer result objects with export methods, use `quick_extract()`

    See Also
    --------
    create_structured_generator : Create reusable generator.
    quick_extract : Quick extraction with StructuredResult wrapper.
    StructuredOutputGenerator : Full-featured generator class.
    """
    config = StructuredOutputConfig(
        max_retries=max_retries,
        instructions=instructions,
        examples=examples,
    )
    generator = StructuredOutputGenerator(model, output_type, config)
    return generator.generate(input_text, **model_kwargs)


def create_structured_generator(
    model: "Model",
    output_type: type[T],
    instructions: str = "",
    examples: Optional[list[dict[str, Any]]] = None,
    max_retries: int = 3,
) -> StructuredOutputGenerator[T]:
    """Create a reusable structured output generator.

    Factory function for creating a configured StructuredOutputGenerator.
    Use this when you need to extract the same type of data multiple times,
    as it avoids recreating the schema and configuration for each call.

    Parameters
    ----------
    model : Model
        LLM model instance to use for generation.
    output_type : type[T]
        Target Pydantic model, dataclass, or type for extraction.
    instructions : str, default=""
        Additional instructions for guiding extraction behavior.
    examples : list[dict[str, Any]] or None, default=None
        Optional few-shot examples with "input" and "output" keys.
    max_retries : int, default=3
        Maximum retry attempts on parsing failure.

    Returns
    -------
    StructuredOutputGenerator[T]
        A configured generator ready for repeated use.

    Examples
    --------
    Create and use a generator:

        >>> from pydantic import BaseModel
        >>> from insideLLMs.structured import create_structured_generator
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>>
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> generator = create_structured_generator(model, Person)
        >>>
        >>> # Reuse for multiple extractions
        >>> person1 = generator.generate("John is 30 years old")
        >>> person2 = generator.generate("Jane is 25 years old")
        >>> person3 = generator.generate("Bob is 40 years old")
        >>> print(f"{person1.name}, {person2.name}, {person3.name}")
        John, Jane, Bob

    With custom instructions:

        >>> generator = create_structured_generator(
        ...     model,
        ...     Person,
        ...     instructions="Extract person info. Use 0 for unknown age."
        ... )
        >>> person = generator.generate("Someone named Alice")
        >>> print(person.age)
        0

    With few-shot examples:

        >>> generator = create_structured_generator(
        ...     model,
        ...     Person,
        ...     examples=[
        ...         {"input": "Alice, 28", "output": {"name": "Alice", "age": 28}},
        ...         {"input": "Bob is 35", "output": {"name": "Bob", "age": 35}}
        ...     ]
        ... )

    Batch processing with generator:

        >>> texts = ["Alice 28", "Bob 35", "Carol 42"]
        >>> people = generator.generate_batch(texts)
        >>> for p in people:
        ...     print(p.name, p.age)

    Process a file of descriptions:

        >>> with open("people.txt") as f:
        ...     lines = f.readlines()
        >>> people = [generator.generate(line) for line in lines]

    Notes
    -----
    - More efficient than `generate_structured` for repeated extractions
    - Schema is generated once and cached
    - Same generator can be used across different inputs

    See Also
    --------
    generate_structured : One-off extraction function.
    StructuredOutputGenerator : The underlying generator class.
    """
    config = StructuredOutputConfig(
        max_retries=max_retries,
        instructions=instructions,
        examples=examples,
    )
    return StructuredOutputGenerator(model, output_type, config)


# =============================================================================
# Model Extension
# =============================================================================


def add_structured_method(model: "Model") -> "Model":
    """Add a generate_structured method to an existing model instance.

    This function extends a model instance in-place by adding a
    `generate_structured` method that wraps the `generate_structured()`
    function. This provides a more object-oriented interface for
    structured extraction.

    Parameters
    ----------
    model : Model
        The model instance to extend. Can be any model that supports
        `chat()` or `generate()` methods.

    Returns
    -------
    Model
        The same model instance with the added `generate_structured` method.
        Returns the model for method chaining.

    Examples
    --------
    Add method to OpenAI model:

        >>> from pydantic import BaseModel
        >>> from insideLLMs.models import OpenAIModel
        >>> from insideLLMs.structured import add_structured_method
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>>
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> model = add_structured_method(model)
        >>>
        >>> # Now use as a method
        >>> person = model.generate_structured("John is 30", Person)
        >>> print(person.name)
        John

    Method chaining:

        >>> model = add_structured_method(OpenAIModel(model_name="gpt-4"))
        >>> person = model.generate_structured(
        ...     "Jane is 25 and works as a designer",
        ...     Person,
        ...     instructions="Extract person information"
        ... )

    Use with any model:

        >>> from insideLLMs.models import AnthropicModel
        >>>
        >>> claude = add_structured_method(AnthropicModel(model_name="claude-3-opus"))
        >>> result = claude.generate_structured(text, MyModel)

    With custom parameters:

        >>> model = add_structured_method(OpenAIModel())
        >>> person = model.generate_structured(
        ...     "Some text about Bob",
        ...     Person,
        ...     instructions="Be thorough",
        ...     max_retries=5
        ... )

    Notes
    -----
    - Modifies the model in-place (returns same instance)
    - The added method wraps `generate_structured()` function
    - Does not affect other model methods
    - Safe to call multiple times (just overwrites the method)

    See Also
    --------
    generate_structured : The underlying function that gets wrapped.
    create_structured_generator : For repeated extractions.
    """

    def generate_structured_method(
        input_text: str,
        output_type: type[T],
        instructions: str = "",
        max_retries: int = 3,
        **kwargs: Any,
    ) -> T:
        return generate_structured(
            model,
            input_text,
            output_type,
            instructions=instructions,
            max_retries=max_retries,
            **kwargs,
        )

    model.generate_structured = generate_structured_method
    return model


# =============================================================================
# Result Viewer and Export Utilities
# =============================================================================


@dataclass
class StructuredResult(Generic[T]):
    """Container for structured extraction results with metadata and export methods.

    This dataclass wraps the extracted data along with useful metadata about
    the extraction process. It provides multiple export formats (JSON, HTML,
    Markdown) and file saving capabilities.

    Parameters
    ----------
    data : T
        The parsed and validated structured data instance.
    raw_response : str
        The original raw text response from the LLM.
    schema : dict[str, Any]
        The JSON schema that was used for extraction.
    prompt : str
        The complete prompt that was sent to the model.
    model_name : str, default="unknown"
        Name/identifier of the model used for extraction.

    Attributes
    ----------
    data : T
        The extracted structured data.
    raw_response : str
        Original LLM response text.
    schema : dict[str, Any]
        JSON schema used for extraction.
    prompt : str
        The prompt sent to the model.
    model_name : str
        Name of the model used.

    Examples
    --------
    Access extracted data:

        >>> from insideLLMs.structured import quick_extract
        >>> from pydantic import BaseModel
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>>
        >>> result = quick_extract("John is 30 years old", Person)
        >>> print(result.data.name)
        John
        >>> print(result.data.age)
        30

    Export to different formats:

        >>> # Get as dictionary
        >>> data_dict = result.to_dict()
        >>> print(data_dict)
        {'name': 'John', 'age': 30}
        >>>
        >>> # Get as JSON string
        >>> json_str = result.to_json()
        >>> print(json_str)
        {
          "name": "John",
          "age": 30
        }
        >>>
        >>> # Get as Markdown
        >>> md = result.to_markdown()
        >>> print(md)
        # Structured Output
        - **name**: John
        - **age**: 30

    Save to files:

        >>> # Save as JSON
        >>> result.save_json("person.json")
        >>>
        >>> # Save as HTML (with styling)
        >>> result.save_html("person.html", title="Person Details")

    Access metadata:

        >>> print(result.model_name)
        gpt-4
        >>> print(result.schema["properties"].keys())
        dict_keys(['name', 'age'])
        >>> print(len(result.raw_response))
        42

    Create HTML report:

        >>> html = result.to_html(title="Extracted Person")
        >>> with open("report.html", "w") as f:
        ...     f.write(html)

    Notes
    -----
    - The `data` attribute contains the actual Pydantic/dataclass instance
    - Export methods handle nested structures automatically
    - HTML export includes basic styling for readability
    - Markdown export is suitable for documentation

    See Also
    --------
    quick_extract : Creates StructuredResult with automatic model setup.
    batch_extract : Creates list of StructuredResults for batch processing.
    results_to_dataframe : Convert multiple results to pandas DataFrame.
    """

    data: T
    raw_response: str
    schema: dict[str, Any]
    prompt: str
    model_name: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert the result data to a plain dictionary.

        Handles Pydantic models (v1 and v2), dataclasses, and regular
        objects by extracting their attributes.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the extracted data.

        Examples
        --------
        Convert to dict:

            >>> result = quick_extract("John is 30", Person)
            >>> d = result.to_dict()
            >>> print(d)
            {'name': 'John', 'age': 30}
            >>> print(type(d))
            <class 'dict'>

        Use with JSON serialization:

            >>> import json
            >>> json.dumps(result.to_dict())
            '{"name": "John", "age": 30}'
        """
        if hasattr(self.data, "model_dump"):  # Pydantic v2
            return self.data.model_dump()
        elif hasattr(self.data, "dict"):  # Pydantic v1
            return self.data.dict()
        elif hasattr(self.data, "__dict__"):
            return {k: v for k, v in self.data.__dict__.items() if not k.startswith("_")}
        elif isinstance(self.data, dict):
            return self.data
        else:
            return {"value": self.data}

    def to_json(self, indent: int = 2) -> str:
        """Convert the result data to a formatted JSON string.

        Parameters
        ----------
        indent : int, default=2
            Number of spaces for JSON indentation. Use 0 or None
            for compact output.

        Returns
        -------
        str
            JSON-formatted string representation of the data.

        Examples
        --------
        Get formatted JSON:

            >>> result = quick_extract("John is 30", Person)
            >>> print(result.to_json())
            {
              "name": "John",
              "age": 30
            }

        Get compact JSON:

            >>> print(result.to_json(indent=0))
            {"name": "John", "age": 30}

        Write to file:

            >>> with open("output.json", "w") as f:
            ...     f.write(result.to_json())
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_html(self, title: str = "Structured Output") -> str:
        """Convert result to HTML for viewing.

        Args:
            title: Title for the HTML page.

        Returns:
            HTML string.
        """
        data_dict = self.to_dict()

        def dict_to_html_table(d: dict, level: int = 0) -> str:
            rows = []
            indent = "  " * level
            for key, value in d.items():
                if isinstance(value, dict):
                    nested = dict_to_html_table(value, level + 1)
                    rows.append(
                        f"{indent}<tr><td><strong>{key}</strong></td><td>{nested}</td></tr>"
                    )
                elif isinstance(value, list):
                    items = "<ul>" + "".join(f"<li>{v}</li>" for v in value) + "</ul>"
                    rows.append(f"{indent}<tr><td><strong>{key}</strong></td><td>{items}</td></tr>")
                else:
                    rows.append(f"{indent}<tr><td><strong>{key}</strong></td><td>{value}</td></tr>")
            return f"<table border='1' cellpadding='5'>\n{''.join(rows)}\n</table>"

        table = dict_to_html_table(data_dict)

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; margin: 10px 0; }}
        td {{ padding: 8px; vertical-align: top; }}
        .metadata {{ color: #666; font-size: 0.9em; margin-top: 20px; }}
        .json-view {{ background: #f5f5f5; padding: 10px; border-radius: 5px; }}
        pre {{ margin: 0; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <h2>Extracted Data</h2>
    {table}
    <h2>JSON View</h2>
    <div class="json-view">
        <pre>{self.to_json()}</pre>
    </div>
    <div class="metadata">
        <p><strong>Model:</strong> {self.model_name}</p>
        <p><strong>Schema Fields:</strong> {", ".join(self.schema.get("properties", {}).keys())}</p>
    </div>
</body>
</html>"""

    def to_markdown(self) -> str:
        """Convert result to Markdown format.

        Returns:
            Markdown string.
        """
        data_dict = self.to_dict()

        def dict_to_md(d: dict, level: int = 0) -> str:
            lines = []
            indent = "  " * level
            for key, value in d.items():
                if isinstance(value, dict):
                    lines.append(f"{indent}- **{key}**:")
                    lines.append(dict_to_md(value, level + 1))
                elif isinstance(value, list):
                    lines.append(f"{indent}- **{key}**: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"{indent}- **{key}**: {value}")
            return "\n".join(lines)

        md = f"# Structured Output\n\n{dict_to_md(data_dict)}\n\n"
        md += f"## JSON\n\n```json\n{self.to_json()}\n```\n"
        return md

    def save_json(self, filepath: str) -> None:
        """Save result as JSON file.

        Args:
            filepath: Path to save to.
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())

    def save_html(self, filepath: str, title: str = "Structured Output") -> None:
        """Save result as HTML file.

        Args:
            filepath: Path to save to.
            title: HTML page title.
        """
        with open(filepath, "w") as f:
            f.write(self.to_html(title))

    def __repr__(self) -> str:
        return f"StructuredResult({self.to_dict()})"


# =============================================================================
# Quick Setup Functions
# =============================================================================


def quick_extract(
    text: str,
    output_type: type[T],
    model_name: str = "gpt-4",
    provider: str = "openai",
    api_key: Optional[str] = None,
    instructions: str = "",
    **model_kwargs: Any,
) -> StructuredResult[T]:
    """Quick one-liner for structured extraction with automatic model setup.

    This is the easiest way to extract structured data - just provide text
    and the expected output type.

    Args:
        text: Text to extract information from.
        output_type: Pydantic model, dataclass, or type for output.
        model_name: Model name (e.g., "gpt-4", "claude-3-opus", "llama-3").
        provider: Model provider ("openai", "anthropic", "huggingface").
        api_key: Optional API key (uses env var if not provided).
        instructions: Additional extraction instructions.
        **model_kwargs: Additional model configuration.

    Returns:
        StructuredResult containing parsed data and export methods.

    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>>
        >>> result = quick_extract(
        ...     "John is a 30-year-old engineer",
        ...     Person,
        ...     model_name="gpt-4"
        ... )
        >>> print(result.data.name)  # John
        >>> result.save_html("person.html")  # Export to HTML
        >>> print(result.to_json())  # Get JSON
    """
    # Auto-detect and create model
    model = _create_model_from_name(model_name, provider, api_key, **model_kwargs)

    # Build prompt
    schema = get_json_schema(output_type)
    config = StructuredOutputConfig(instructions=instructions)
    generator = StructuredOutputGenerator(model, output_type, config)

    prompt = generator._build_prompt(text)

    # Generate
    if hasattr(model, "chat"):
        messages = [
            {"role": "system", "content": STRUCTURED_OUTPUT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw_response = model.chat(messages, temperature=0.0)
    else:
        raw_response = model.generate(
            f"{STRUCTURED_OUTPUT_SYSTEM_PROMPT}\n\n{prompt}", temperature=0.0
        )

    # Parse
    data = parse_json(raw_response)
    parsed = parse_to_type(data, output_type)

    return StructuredResult(
        data=parsed,
        raw_response=raw_response,
        schema=schema,
        prompt=prompt,
        model_name=model_name,
    )


def _create_model_from_name(
    model_name: str,
    provider: str,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> "Model":
    """Create a model instance from name and provider.

    Args:
        model_name: Name of the model.
        provider: Provider name.
        api_key: Optional API key.
        **kwargs: Additional model config.

    Returns:
        Configured model instance.
    """
    provider = provider.lower()

    if provider == "openai":
        from insideLLMs.models import OpenAIModel

        return OpenAIModel(model_name=model_name, api_key=api_key, **kwargs)

    elif provider == "anthropic":
        from insideLLMs.models import AnthropicModel

        return AnthropicModel(model_name=model_name, api_key=api_key, **kwargs)

    elif provider == "huggingface":
        from insideLLMs.models import HuggingFaceModel

        return HuggingFaceModel(model_name=model_name, **kwargs)

    elif provider == "dummy":
        from insideLLMs.models import DummyModel

        return DummyModel(name=model_name)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: openai, anthropic, huggingface, dummy"
        )


# =============================================================================
# Batch Processing with Results
# =============================================================================


def batch_extract(
    texts: list[str],
    output_type: type[T],
    model: "Model",
    instructions: str = "",
    show_progress: bool = True,
) -> list[StructuredResult[T]]:
    """Extract structured data from multiple texts.

    Args:
        texts: List of texts to process.
        output_type: Target type for extraction.
        model: Model to use.
        instructions: Extraction instructions.
        show_progress: Whether to print progress.

    Returns:
        List of StructuredResult objects.

    Example:
        >>> texts = ["John is 30", "Jane is 25", "Bob is 40"]
        >>> results = batch_extract(texts, Person, model)
        >>> for r in results:
        ...     print(r.data.name, r.data.age)
    """
    results = []
    schema = get_json_schema(output_type)
    config = StructuredOutputConfig(instructions=instructions)
    generator = StructuredOutputGenerator(model, output_type, config)

    for i, text in enumerate(texts):
        if show_progress:
            print(f"Processing {i + 1}/{len(texts)}...", end="\r")

        try:
            prompt = generator._build_prompt(text)

            if hasattr(model, "chat"):
                messages = [
                    {"role": "system", "content": STRUCTURED_OUTPUT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                raw_response = model.chat(messages, temperature=0.0)
            else:
                raw_response = model.generate(
                    f"{STRUCTURED_OUTPUT_SYSTEM_PROMPT}\n\n{prompt}", temperature=0.0
                )

            data = parse_json(raw_response)
            parsed = parse_to_type(data, output_type)

            results.append(
                StructuredResult(
                    data=parsed,
                    raw_response=raw_response,
                    schema=schema,
                    prompt=prompt,
                    model_name=getattr(model, "name", "unknown"),
                )
            )

        except Exception as e:
            # Create error result
            results.append(
                StructuredResult(
                    data=None,
                    raw_response=str(e),
                    schema=schema,
                    prompt=text,
                    model_name=getattr(model, "name", "unknown"),
                )
            )

    if show_progress:
        print(f"Processed {len(texts)} texts.       ")

    return results


def results_to_dataframe(results: list[StructuredResult]) -> Any:
    """Convert list of StructuredResults to pandas DataFrame.

    Args:
        results: List of StructuredResult objects.

    Returns:
        pandas DataFrame.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for DataFrame export. Install with: pip install pandas"
        ) from None

    rows = []
    for r in results:
        if r.data is not None:
            row = r.to_dict()
            rows.append(row)

    return pd.DataFrame(rows)


def results_to_html_report(
    results: list[StructuredResult],
    title: str = "Extraction Results",
) -> str:
    """Generate HTML report from multiple results.

    Args:
        results: List of StructuredResult objects.
        title: Report title.

    Returns:
        HTML string.
    """
    rows = []
    for i, r in enumerate(results, 1):
        if r.data is not None:
            data_dict = r.to_dict()
            cells = "".join(f"<td>{v}</td>" for v in data_dict.values())
            rows.append(f"<tr><td>{i}</td>{cells}</tr>")

    # Get headers from first valid result
    headers = ["#"]
    for r in results:
        if r.data is not None:
            headers.extend(r.to_dict().keys())
            break

    header_row = "".join(f"<th>{h}</th>" for h in headers)

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
        .summary {{ color: #666; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="summary">Total results: {len(results)} | Successful: {sum(1 for r in results if r.data is not None)}</p>
    <table>
        <tr>{header_row}</tr>
        {"".join(rows)}
    </table>
</body>
</html>"""


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Exceptions
    "StructuredOutputError",
    "SchemaGenerationError",
    "ParsingError",
    "ValidationErrorWrapper",
    # Schema generation
    "pydantic_to_json_schema",
    "dataclass_to_json_schema",
    "get_json_schema",
    # Parsing
    "extract_json",
    "parse_json",
    "parse_to_type",
    # Configuration
    "StructuredOutputConfig",
    # Generator
    "StructuredOutputGenerator",
    # Result container
    "StructuredResult",
    # Convenience functions
    "generate_structured",
    "create_structured_generator",
    "add_structured_method",
    # Quick setup
    "quick_extract",
    # Batch processing
    "batch_extract",
    "results_to_dataframe",
    "results_to_html_report",
    # Constants
    "PYDANTIC_AVAILABLE",
]
