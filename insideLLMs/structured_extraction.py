"""
Model output parsing and structured extraction utilities.

This module provides a comprehensive suite of tools for extracting structured data
from Large Language Model (LLM) responses and other unstructured text sources.
It supports multiple extraction formats, schema validation, entity recognition,
and intelligent format auto-detection.

Overview
--------
The module is organized around several key concepts:

1. **Extractors**: Specialized classes for parsing specific data formats
   - `JSONExtractor`: Extracts JSON objects from text, handling code blocks
   - `KeyValueExtractor`: Parses key-value pairs with configurable separators
   - `ListExtractor`: Identifies and extracts bulleted/numbered lists
   - `TableExtractor`: Parses markdown and whitespace-aligned tables
   - `EntityExtractor`: Recognizes named entities (emails, URLs, dates, etc.)
   - `StructuredExtractor`: Main orchestrator with auto-detection capabilities

2. **Schema Validation**: Define and validate extraction schemas
   - `FieldSchema`: Individual field definitions with type and constraint validation
   - `ExtractionSchema`: Collection of fields with strict/lenient modes

3. **Results**: Structured output with confidence scores and diagnostics
   - `ExtractionResult`: Complete extraction outcome with status and metadata
   - `EntityMatch`: Individual entity occurrence with position information

Features
--------
- Auto-detection of JSON, key-value, list, and table formats
- Schema-based validation with type coercion
- Configurable extraction parameters (separators, patterns, strictness)
- Confidence scoring for extraction quality assessment
- Entity recognition for common patterns (email, URL, phone, date, etc.)
- Support for nested JSON and balanced brace matching
- Markdown table parsing with header detection

Examples
--------
Basic JSON extraction from LLM response:

>>> from insideLLMs.structured_extraction import extract_json
>>> response = '''
... Here is the user data:
... ```json
... {"name": "Alice", "age": 30, "email": "alice@example.com"}
... ```
... '''
>>> result = extract_json(response)
>>> result.is_success
True
>>> result.extracted_data
{'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}

Schema-validated extraction:

>>> from insideLLMs.structured_extraction import (
...     create_schema, StructuredExtractor, FieldType
... )
>>> schema = create_schema([
...     {"name": "username", "type": "string", "min_length": 3},
...     {"name": "score", "type": "integer", "min_value": 0, "max_value": 100},
... ])
>>> extractor = StructuredExtractor(schema=schema)
>>> result = extractor.extract('{"username": "bob", "score": 85}')
>>> result.status.value
'success'

Auto-detecting format from mixed content:

>>> from insideLLMs.structured_extraction import extract_structured
>>> text = '''
... Name: John Doe
... Role: Engineer
... Active: yes
... '''
>>> result = extract_structured(text)
>>> result.format_detected.value
'key_value'
>>> result.extracted_data
{'Name': 'John Doe', 'Role': 'Engineer', 'Active': True}

Entity extraction from text:

>>> from insideLLMs.structured_extraction import extract_entities
>>> text = "Contact us at support@company.com or visit https://company.com"
>>> entities = extract_entities(text)
>>> [(e.entity_type, e.text) for e in entities]
[('email', 'support@company.com'), ('url', 'https://company.com')]

Notes
-----
- JSON extraction supports code blocks with or without language hints
- Key-value extraction automatically coerces types (numbers, booleans, lists)
- Entity patterns are case-insensitive by default
- Table extraction attempts markdown format first, then whitespace-aligned
- Schema validation can be strict (reject extra fields) or lenient

See Also
--------
- `insideLLMs.probing`: LLM probing utilities that may produce responses
  requiring structured extraction
- `insideLLMs.visualization`: Tools for visualizing extraction results

Module Attributes
-----------------
DEFAULT_PATTERNS : dict
    Default regex patterns for entity extraction (defined in EntityExtractor)
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from re import Pattern
from typing import Any, Optional


class ExtractionFormat(Enum):
    """Enumeration of supported extraction formats for structured data parsing.

    This enum defines the various data formats that the extraction utilities
    can recognize, parse, and convert to structured Python dictionaries.
    Used throughout the module to indicate both expected and detected formats.

    Attributes
    ----------
    JSON : str
        JavaScript Object Notation format. Supports nested objects, arrays,
        and standard JSON data types. Extracted from code blocks or raw text.
    XML : str
        Extensible Markup Language format. Supports hierarchical tag-based
        structures with attributes and text content.
    YAML : str
        YAML Ain't Markup Language format. Human-readable data serialization
        with indentation-based nesting.
    KEY_VALUE : str
        Simple key-value pair format. Lines contain keys and values separated
        by configurable delimiters (e.g., ":", "=", "->").
    LIST : str
        Bulleted or numbered list format. Supports markers like "-", "*",
        numbers with periods or parentheses, and letter prefixes.
    TABLE : str
        Tabular data format. Supports markdown tables with pipe delimiters
        and whitespace-aligned columns.
    MARKDOWN : str
        General markdown format. May contain mixed structures including
        headers, lists, code blocks, and inline formatting.
    FREE_TEXT : str
        Unstructured plain text. Used as fallback when no structured format
        is detected. The text is returned as-is in the extraction result.

    Examples
    --------
    Using format hints with StructuredExtractor:

    >>> from insideLLMs.structured_extraction import (
    ...     StructuredExtractor, ExtractionFormat
    ... )
    >>> extractor = StructuredExtractor()
    >>> text = "name: Alice\\nage: 30"
    >>> result = extractor.extract(text, expected_format=ExtractionFormat.KEY_VALUE)
    >>> result.format_detected == ExtractionFormat.KEY_VALUE
    True

    Checking detected format after extraction:

    >>> result = extractor.extract('{"key": "value"}')
    >>> result.format_detected
    <ExtractionFormat.JSON: 'json'>

    See Also
    --------
    StructuredExtractor : Main extraction class that uses these format types.
    ExtractionResult : Contains the detected format in its `format_detected` field.
    """

    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    KEY_VALUE = "key_value"
    LIST = "list"
    TABLE = "table"
    MARKDOWN = "markdown"
    FREE_TEXT = "free_text"


class FieldType(Enum):
    """Enumeration of field data types for schema validation.

    Defines the allowed data types for fields in an ExtractionSchema. Each type
    has associated validation logic in FieldSchema.validate() that checks whether
    extracted values conform to the expected type. The TypeCoercer class can
    convert values between types when possible.

    Attributes
    ----------
    STRING : str
        Text string values. Validates using isinstance(value, str).
    INTEGER : str
        Whole number values. Excludes boolean values despite Python's int subclass
        relationship. Validates that value is int but not bool.
    FLOAT : str
        Decimal number values. Accepts both int and float values, but excludes
        booleans. Useful for fields that may receive whole or fractional numbers.
    BOOLEAN : str
        True/False values. Validates using isinstance(value, bool). TypeCoercer
        can convert strings like "yes", "true", "1" to boolean.
    LIST : str
        Array/list values. Validates using isinstance(value, list). TypeCoercer
        can split comma-separated strings into lists.
    DICT : str
        Dictionary/object values. Validates using isinstance(value, dict).
    DATE : str
        Date values in standard formats (e.g., "2024-01-15", "01/15/2024").
        Currently relies on pattern validation rather than parsing.
    DATETIME : str
        Date and time values. Extends DATE with time component support.
    EMAIL : str
        Email address values. Validates against a basic email regex pattern
        matching the format "user@domain.tld".
    URL : str
        Web URL values. Validates against a pattern requiring "http://" or
        "https://" prefix followed by non-whitespace characters.
    ANY : str
        Accepts any value type. Skips type validation entirely. Useful for
        fields where the type is unknown or intentionally flexible.

    Examples
    --------
    Creating a typed field schema:

    >>> from insideLLMs.structured_extraction import FieldSchema, FieldType
    >>> email_field = FieldSchema(
    ...     name="contact_email",
    ...     field_type=FieldType.EMAIL,
    ...     required=True
    ... )
    >>> email_field.validate("user@example.com")
    (True, None)
    >>> email_field.validate("invalid-email")
    (False, "Field 'contact_email' must be a valid email")

    Using TypeCoercer for type conversion:

    >>> from insideLLMs.structured_extraction import TypeCoercer, FieldType
    >>> TypeCoercer.coerce("42", FieldType.INTEGER)
    (42, True)
    >>> TypeCoercer.coerce("yes", FieldType.BOOLEAN)
    (True, True)
    >>> TypeCoercer.coerce("a, b, c", FieldType.LIST)
    (['a', 'b', 'c'], True)

    See Also
    --------
    FieldSchema : Uses FieldType for field definitions and validation.
    TypeCoercer : Converts values between FieldType types.
    """

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    URL = "url"
    ANY = "any"


class ExtractionStatus(Enum):
    """Enumeration of extraction operation outcomes.

    Represents the result status of an extraction attempt, indicating whether
    the operation succeeded, partially succeeded, or failed. The status helps
    consumers decide how to handle the extracted data and whether to apply
    fallback strategies.

    Attributes
    ----------
    SUCCESS : str
        Extraction completed successfully with high confidence. The extracted
        data fully matches the expected format and passes all validations.
        Safe to use the extracted data directly.
    PARTIAL : str
        Extraction partially succeeded. Some data was extracted but may be
        incomplete or have lower confidence. Common when auto-detection falls
        back to FREE_TEXT format. Check `warnings` in ExtractionResult.
    FAILED : str
        Extraction failed entirely. No usable data was extracted. The `errors`
        list in ExtractionResult contains failure reasons. May occur when
        input text contains no recognizable structured format.
    VALIDATION_ERROR : str
        Extraction succeeded but schema validation failed. The data was parsed
        correctly but does not conform to the provided ExtractionSchema.
        Check `errors` for specific validation failures (missing required
        fields, type mismatches, constraint violations).

    Examples
    --------
    Checking extraction status for error handling:

    >>> from insideLLMs.structured_extraction import (
    ...     extract_json, ExtractionStatus
    ... )
    >>> result = extract_json("This has no JSON in it")
    >>> result.status == ExtractionStatus.FAILED
    True
    >>> result.errors
    ['No JSON found in text']

    Handling partial success:

    >>> from insideLLMs.structured_extraction import extract_structured
    >>> result = extract_structured("Just some plain text here")
    >>> result.status == ExtractionStatus.PARTIAL
    True
    >>> result.warnings
    ['No structured format detected, returning raw text']

    Schema validation failure:

    >>> from insideLLMs.structured_extraction import (
    ...     StructuredExtractor, create_schema
    ... )
    >>> schema = create_schema([{"name": "required_field", "required": True}])
    >>> extractor = StructuredExtractor(schema=schema)
    >>> result = extractor.extract('{"other_field": "value"}')
    >>> result.status == ExtractionStatus.VALIDATION_ERROR
    True

    See Also
    --------
    ExtractionResult : Contains status along with extracted data and diagnostics.
    ExtractionResult.is_success : Property that checks for SUCCESS or PARTIAL status.
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    VALIDATION_ERROR = "validation_error"


@dataclass
class FieldSchema:
    """Schema definition for a single field in structured data extraction.

    Defines the expected name, type, and constraints for a field that should
    be extracted from text. Used within ExtractionSchema to specify the complete
    structure of expected data. Provides validation logic for checking extracted
    values against the defined constraints.

    Parameters
    ----------
    name : str
        The field name/key as it appears in the extracted data. Case-sensitive
        matching is used during validation.
    field_type : FieldType, optional
        The expected data type for this field. Defaults to FieldType.STRING.
        Used for type validation and can guide type coercion.
    required : bool, optional
        Whether this field must be present in extracted data. Defaults to True.
        Missing required fields cause validation to fail.
    default : Any, optional
        Default value when field is missing. Only used if required=False.
        Defaults to None.
    description : str, optional
        Human-readable description of the field's purpose. Useful for
        documentation and error messages. Defaults to empty string.
    pattern : str, optional
        Regular expression pattern for string validation. Only applied to
        string values. Uses re.match() for validation. Defaults to None.
    min_value : float, optional
        Minimum allowed value for numeric fields (INTEGER, FLOAT). Values
        below this threshold fail validation. Defaults to None (no minimum).
    max_value : float, optional
        Maximum allowed value for numeric fields. Values above this
        threshold fail validation. Defaults to None (no maximum).
    min_length : int, optional
        Minimum length for strings and lists. Uses len() for validation.
        Defaults to None (no minimum length).
    max_length : int, optional
        Maximum length for strings and lists. Defaults to None (no maximum).
    allowed_values : list[Any], optional
        Explicit list of permitted values. If set, extracted value must be
        in this list. Defaults to None (any value allowed).
    nested_schema : list[FieldSchema], optional
        For DICT or LIST types, defines the schema for nested elements.
        Enables validation of complex nested structures. Defaults to None.

    Attributes
    ----------
    name : str
        The field identifier.
    field_type : FieldType
        The expected data type.
    required : bool
        Whether the field is mandatory.
    default : Any
        Default value for optional fields.
    description : str
        Field documentation.
    pattern : str or None
        Regex pattern constraint.
    min_value : float or None
        Minimum numeric value.
    max_value : float or None
        Maximum numeric value.
    min_length : int or None
        Minimum length constraint.
    max_length : int or None
        Maximum length constraint.
    allowed_values : list or None
        Enumerated allowed values.
    nested_schema : list[FieldSchema] or None
        Schema for nested structures.

    Examples
    --------
    Basic string field with length constraints:

    >>> from insideLLMs.structured_extraction import FieldSchema, FieldType
    >>> username_field = FieldSchema(
    ...     name="username",
    ...     field_type=FieldType.STRING,
    ...     required=True,
    ...     min_length=3,
    ...     max_length=20,
    ...     pattern=r"^[a-zA-Z0-9_]+$"
    ... )
    >>> username_field.validate("alice_123")
    (True, None)
    >>> username_field.validate("ab")
    (False, "Field 'username' must have length >= 3")

    Numeric field with range validation:

    >>> score_field = FieldSchema(
    ...     name="score",
    ...     field_type=FieldType.INTEGER,
    ...     min_value=0,
    ...     max_value=100,
    ...     description="User's score from 0-100"
    ... )
    >>> score_field.validate(85)
    (True, None)
    >>> score_field.validate(150)
    (False, "Field 'score' must be <= 100")

    Optional field with allowed values:

    >>> status_field = FieldSchema(
    ...     name="status",
    ...     field_type=FieldType.STRING,
    ...     required=False,
    ...     default="pending",
    ...     allowed_values=["pending", "active", "completed", "cancelled"]
    ... )
    >>> status_field.validate("active")
    (True, None)
    >>> status_field.validate("unknown")
    (False, "Field 'status' must be one of ['pending', 'active', 'completed', 'cancelled']")
    >>> status_field.validate(None)  # Optional field allows None
    (True, None)

    See Also
    --------
    ExtractionSchema : Combines multiple FieldSchema instances.
    FieldType : Enumeration of supported field types.
    create_schema : Convenience function for creating schemas from dicts.
    """

    name: str
    field_type: FieldType = FieldType.STRING
    required: bool = True
    default: Any = None
    description: str = ""
    pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[list[Any]] = None
    nested_schema: Optional[list["FieldSchema"]] = None

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against this field's schema constraints.

        Performs comprehensive validation including type checking, pattern
        matching, range validation, length constraints, and allowed value
        verification. Returns a tuple indicating success and an optional
        error message.

        Args
        ----
        value : Any
            The value to validate. Can be any type; type checking is
            performed based on field_type.

        Returns
        -------
        tuple[bool, Optional[str]]
            A tuple of (is_valid, error_message). If valid, returns
            (True, None). If invalid, returns (False, "error description").

        Examples
        --------
        Validating required vs optional fields:

        >>> field = FieldSchema(name="email", field_type=FieldType.EMAIL)
        >>> field.validate("user@example.com")
        (True, None)
        >>> field.validate(None)  # Required field
        (False, "Field 'email' is required")

        >>> optional = FieldSchema(name="nickname", required=False)
        >>> optional.validate(None)
        (True, None)

        Type validation examples:

        >>> int_field = FieldSchema(name="count", field_type=FieldType.INTEGER)
        >>> int_field.validate(42)
        (True, None)
        >>> int_field.validate("42")  # String, not int
        (False, "Field 'count' must be an integer")
        >>> int_field.validate(True)  # Booleans excluded
        (False, "Field 'count' must be an integer")

        Pattern validation:

        >>> phone = FieldSchema(
        ...     name="phone",
        ...     pattern=r"^\\d{3}-\\d{4}$"
        ... )
        >>> phone.validate("555-1234")
        (True, None)
        >>> phone.validate("5551234")
        (False, "Field 'phone' does not match pattern '^\\\\d{3}-\\\\d{4}$'")
        """
        if value is None:
            if self.required:
                return False, f"Field '{self.name}' is required"
            return True, None

        # Type validation
        type_valid, type_error = self._validate_type(value)
        if not type_valid:
            return False, type_error

        # Pattern validation
        if self.pattern and isinstance(value, str) and not re.match(self.pattern, value):
            return False, f"Field '{self.name}' does not match pattern '{self.pattern}'"

        # Range validation
        if self.min_value is not None and isinstance(value, (int, float)):
            if value < self.min_value:
                return False, f"Field '{self.name}' must be >= {self.min_value}"
        if self.max_value is not None and isinstance(value, (int, float)):
            if value > self.max_value:
                return False, f"Field '{self.name}' must be <= {self.max_value}"

        # Length validation
        if self.min_length is not None and isinstance(value, (str, list)):
            if len(value) < self.min_length:
                return False, f"Field '{self.name}' must have length >= {self.min_length}"
        if self.max_length is not None and isinstance(value, (str, list)):
            if len(value) > self.max_length:
                return False, f"Field '{self.name}' must have length <= {self.max_length}"

        # Allowed values validation
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"Field '{self.name}' must be one of {self.allowed_values}"

        return True, None

    def _validate_type(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate that a value matches the expected field type.

        Internal method that performs type-specific validation based on the
        field's configured field_type. Handles special cases like excluding
        booleans from integer/float validation and pattern matching for
        email/URL types.

        Args
        ----
        value : Any
            The value to type-check. Should not be None (handled by validate()).

        Returns
        -------
        tuple[bool, Optional[str]]
            A tuple of (is_valid, error_message). Returns (True, None) if the
            value matches the expected type, or (False, "error message") if
            type validation fails.

        Examples
        --------
        >>> field = FieldSchema(name="count", field_type=FieldType.INTEGER)
        >>> field._validate_type(42)
        (True, None)
        >>> field._validate_type(3.14)
        (False, "Field 'count' must be an integer")

        >>> url_field = FieldSchema(name="website", field_type=FieldType.URL)
        >>> url_field._validate_type("https://example.com")
        (True, None)
        >>> url_field._validate_type("not-a-url")
        (False, "Field 'website' must be a valid URL")

        Notes
        -----
        Boolean values are explicitly excluded from INTEGER and FLOAT types
        because in Python, bool is a subclass of int (True == 1, False == 0).
        This prevents accidental acceptance of booleans as numeric values.
        """
        if self.field_type == FieldType.ANY:
            return True, None
        if self.field_type == FieldType.STRING:
            if not isinstance(value, str):
                return False, f"Field '{self.name}' must be a string"
        elif self.field_type == FieldType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Field '{self.name}' must be an integer"
        elif self.field_type == FieldType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"Field '{self.name}' must be a number"
        elif self.field_type == FieldType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Field '{self.name}' must be a boolean"
        elif self.field_type == FieldType.LIST:
            if not isinstance(value, list):
                return False, f"Field '{self.name}' must be a list"
        elif self.field_type == FieldType.DICT:
            if not isinstance(value, dict):
                return False, f"Field '{self.name}' must be a dictionary"
        elif self.field_type == FieldType.EMAIL:
            if not isinstance(value, str) or not re.match(r"[^@]+@[^@]+\.[^@]+", value):
                return False, f"Field '{self.name}' must be a valid email"
        elif self.field_type == FieldType.URL:
            if not isinstance(value, str) or not re.match(r"https?://\S+", value):
                return False, f"Field '{self.name}' must be a valid URL"
        return True, None


@dataclass
class ExtractionResult:
    """Complete result of a structured data extraction operation.

    Encapsulates all information about an extraction attempt, including the
    extracted data, operation status, detected format, confidence score, and
    any errors or warnings generated during extraction. This is the primary
    return type for all extractor classes.

    Parameters
    ----------
    raw_text : str
        The original input text that was processed. Retained for debugging
        and re-processing purposes.
    extracted_data : dict[str, Any]
        The structured data extracted from the text. For successful extractions,
        contains the parsed key-value pairs. For failed extractions, typically
        empty or contains partial data.
    status : ExtractionStatus
        The outcome of the extraction operation. SUCCESS indicates complete
        extraction, PARTIAL indicates some data was recovered, FAILED indicates
        no data could be extracted, VALIDATION_ERROR indicates schema mismatch.
    format_detected : ExtractionFormat
        The data format that was identified in the input text. For auto-detection,
        this reflects what format the extractor determined was present.
    confidence : float
        A score from 0.0 to 1.0 indicating extraction confidence. Higher values
        indicate greater certainty that the extraction is correct. Affected by
        format clarity, data completeness, and extraction method used.
    errors : list[str], optional
        List of error messages if extraction failed or had issues. Empty for
        successful extractions. Defaults to empty list.
    warnings : list[str], optional
        List of warning messages for non-fatal issues encountered during
        extraction. May include format ambiguities or data transformations.
        Defaults to empty list.
    metadata : dict[str, Any], optional
        Additional metadata about the extraction process. May include timing
        information, format-specific details, or extractor configuration.
        Defaults to empty dict.

    Attributes
    ----------
    raw_text : str
        The original input text.
    extracted_data : dict[str, Any]
        The parsed structured data.
    status : ExtractionStatus
        Operation outcome status.
    format_detected : ExtractionFormat
        Identified data format.
    confidence : float
        Extraction confidence score.
    errors : list[str]
        Error messages from extraction.
    warnings : list[str]
        Warning messages from extraction.
    metadata : dict[str, Any]
        Additional extraction metadata.

    Examples
    --------
    Successful JSON extraction result:

    >>> from insideLLMs.structured_extraction import extract_json
    >>> result = extract_json('{"name": "Alice", "age": 30}')
    >>> result.is_success
    True
    >>> result.status.value
    'success'
    >>> result.extracted_data
    {'name': 'Alice', 'age': 30}
    >>> result.confidence > 0.8
    True

    Failed extraction with error information:

    >>> result = extract_json("This is plain text with no JSON")
    >>> result.is_success
    False
    >>> result.status.value
    'failed'
    >>> result.errors
    ['No JSON found in text']
    >>> result.extracted_data
    {}

    Using get_field for safe access:

    >>> result = extract_json('{"user": {"name": "Bob"}, "active": true}')
    >>> result.get_field("user")
    {'name': 'Bob'}
    >>> result.get_field("missing_field", default="N/A")
    'N/A'

    Converting to dictionary for serialization:

    >>> result = extract_json('{"key": "value"}')
    >>> d = result.to_dict()
    >>> d["status"]
    'success'
    >>> d["format_detected"]
    'json'

    See Also
    --------
    ExtractionStatus : Possible status values for extraction results.
    ExtractionFormat : Possible format values for detected formats.
    StructuredExtractor : Main class that produces ExtractionResult objects.
    """

    raw_text: str
    extracted_data: dict[str, Any]
    status: ExtractionStatus
    format_detected: ExtractionFormat
    confidence: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the extraction result to a plain dictionary.

        Serializes the ExtractionResult to a dictionary suitable for JSON
        encoding, logging, or API responses. Enum values are converted to
        their string representations. Long raw_text is truncated to 100
        characters with an ellipsis for readability.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all result fields with serializable values.
            Includes: raw_text (truncated), extracted_data, status (string),
            format_detected (string), confidence, errors, warnings, metadata.

        Examples
        --------
        Basic serialization:

        >>> from insideLLMs.structured_extraction import extract_json
        >>> result = extract_json('{"name": "Alice"}')
        >>> d = result.to_dict()
        >>> d["status"]
        'success'
        >>> d["format_detected"]
        'json'
        >>> isinstance(d["confidence"], float)
        True

        Long text truncation:

        >>> long_text = '{"data": "' + 'x' * 200 + '"}'
        >>> result = extract_json(long_text)
        >>> len(result.to_dict()["raw_text"]) <= 103  # 100 + "..."
        True

        JSON serialization:

        >>> import json
        >>> result = extract_json('{"key": "value"}')
        >>> json_str = json.dumps(result.to_dict())
        >>> "success" in json_str
        True
        """
        return {
            "raw_text": self.raw_text[:100] + "..." if len(self.raw_text) > 100 else self.raw_text,
            "extracted_data": self.extracted_data,
            "status": self.status.value,
            "format_detected": self.format_detected.value,
            "confidence": self.confidence,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def get_field(self, field_name: str, default: Any = None) -> Any:
        """Get a specific field from extracted data."""
        return self.extracted_data.get(field_name, default)

    @property
    def is_success(self) -> bool:
        """Check if extraction was successful."""
        return self.status in (ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL)


@dataclass
class EntityMatch:
    """A matched entity in text."""

    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    normalized_value: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionSchema:
    """Schema for structured extraction."""

    fields: list[FieldSchema]
    strict: bool = True
    allow_extra_fields: bool = False
    name: str = ""
    description: str = ""

    def validate(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate data against schema."""
        errors = []

        # Check required fields
        for field_schema in self.fields:
            value = data.get(field_schema.name)
            is_valid, error = field_schema.validate(value)
            if not is_valid:
                errors.append(error)

        # Check for extra fields
        if not self.allow_extra_fields:
            field_names = {f.name for f in self.fields}
            extra_fields = set(data.keys()) - field_names
            if extra_fields and self.strict:
                errors.append(f"Extra fields not allowed: {extra_fields}")

        return len(errors) == 0, errors


class JSONExtractor:
    """Extracts JSON from text responses."""

    def __init__(self, strict: bool = False):
        """Initialize extractor."""
        self.strict = strict

    def extract(self, text: str) -> ExtractionResult:
        """Extract JSON from text."""
        errors = []
        warnings = []

        # Try to find JSON in various formats
        json_candidates = self._find_json_candidates(text)

        if not json_candidates:
            return ExtractionResult(
                raw_text=text,
                extracted_data={},
                status=ExtractionStatus.FAILED,
                format_detected=ExtractionFormat.JSON,
                confidence=0.0,
                errors=["No JSON found in text"],
            )

        # Try to parse each candidate
        for candidate, confidence_boost in json_candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return ExtractionResult(
                        raw_text=text,
                        extracted_data=data,
                        status=ExtractionStatus.SUCCESS,
                        format_detected=ExtractionFormat.JSON,
                        confidence=0.9 + confidence_boost,
                        warnings=warnings,
                    )
                elif isinstance(data, list):
                    return ExtractionResult(
                        raw_text=text,
                        extracted_data={"items": data},
                        status=ExtractionStatus.SUCCESS,
                        format_detected=ExtractionFormat.JSON,
                        confidence=0.85 + confidence_boost,
                        warnings=["JSON array wrapped in 'items' key"],
                    )
            except json.JSONDecodeError as e:
                errors.append(f"JSON parse error: {e}")

        return ExtractionResult(
            raw_text=text,
            extracted_data={},
            status=ExtractionStatus.FAILED,
            format_detected=ExtractionFormat.JSON,
            confidence=0.0,
            errors=errors,
        )

    def _find_json_candidates(self, text: str) -> list[tuple[str, float]]:
        """Find potential JSON strings in text."""
        candidates = []

        # Look for code blocks
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        for match in re.finditer(code_block_pattern, text):
            content = match.group(1).strip()
            if content.startswith("{") or content.startswith("["):
                candidates.append((content, 0.05))

        # Look for balanced JSON objects/arrays using brace matching
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            for i, c in enumerate(text):
                if c == start_char:
                    balanced = self._find_balanced_braces(text, i, start_char, end_char)
                    if balanced:
                        candidates.append((balanced, 0.0))

        return candidates

    def _find_balanced_braces(
        self, text: str, start: int, open_char: str, close_char: str
    ) -> Optional[str]:
        """Find balanced brace expression starting at position."""
        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            c = text[i]

            if escape:
                escape = False
                continue

            if c == "\\":
                escape = True
                continue

            if c == '"' and not escape:
                in_string = not in_string
                continue

            if in_string:
                continue

            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None


class KeyValueExtractor:
    """Extracts key-value pairs from text."""

    def __init__(
        self,
        separators: Optional[list[str]] = None,
        case_sensitive: bool = False,
    ):
        """Initialize extractor."""
        self.separators = separators or [":", "=", "->", "-"]
        self.case_sensitive = case_sensitive

    def extract(self, text: str, keys: Optional[list[str]] = None) -> ExtractionResult:
        """Extract key-value pairs from text."""
        extracted = {}
        warnings = []

        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for sep in self.separators:
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        key = parts[0].strip().strip("*-#").strip()
                        value = parts[1].strip()

                        if keys:
                            # Check if this matches any expected key
                            for expected_key in keys:
                                key_match = key if self.case_sensitive else key.lower()
                                expected_match = (
                                    expected_key if self.case_sensitive else expected_key.lower()
                                )
                                if key_match == expected_match:
                                    extracted[expected_key] = self._parse_value(value)
                                    break
                        else:
                            extracted[key] = self._parse_value(value)
                        break

        if not extracted:
            return ExtractionResult(
                raw_text=text,
                extracted_data={},
                status=ExtractionStatus.FAILED,
                format_detected=ExtractionFormat.KEY_VALUE,
                confidence=0.0,
                errors=["No key-value pairs found"],
            )

        confidence = min(0.9, 0.5 + len(extracted) * 0.1)

        return ExtractionResult(
            raw_text=text,
            extracted_data=extracted,
            status=ExtractionStatus.SUCCESS,
            format_detected=ExtractionFormat.KEY_VALUE,
            confidence=confidence,
            warnings=warnings,
        )

    def _parse_value(self, value: str) -> Any:
        """Parse value to appropriate type."""
        value = value.strip()

        # Boolean
        if value.lower() in ("true", "yes", "on"):
            return True
        if value.lower() in ("false", "no", "off"):
            return False

        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # List (comma-separated)
        if "," in value:
            items = [item.strip() for item in value.split(",")]
            return items

        return value


class ListExtractor:
    """Extracts lists from text."""

    def __init__(self):
        """Initialize extractor."""
        self.list_patterns = [
            r"^\s*[-*•]\s*(.+)$",  # Bullet points
            r"^\s*\d+[.)]\s*(.+)$",  # Numbered lists
            r"^\s*[a-zA-Z][.)]\s*(.+)$",  # Letter lists
        ]

    def extract(self, text: str) -> ExtractionResult:
        """Extract list items from text."""
        items = []

        lines = text.split("\n")

        for line in lines:
            for pattern in self.list_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    items.append(match.group(1).strip())
                    break

        if not items:
            return ExtractionResult(
                raw_text=text,
                extracted_data={"items": []},
                status=ExtractionStatus.FAILED,
                format_detected=ExtractionFormat.LIST,
                confidence=0.0,
                errors=["No list items found"],
            )

        return ExtractionResult(
            raw_text=text,
            extracted_data={"items": items, "count": len(items)},
            status=ExtractionStatus.SUCCESS,
            format_detected=ExtractionFormat.LIST,
            confidence=min(0.95, 0.6 + len(items) * 0.05),
        )


class EntityExtractor:
    """Extracts named entities from text."""

    DEFAULT_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "url": r"https?://[^\s<>\"{}|\\^`\[\]]+",
        "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        "time": r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b",
        "number": r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
        "percentage": r"\b\d+(?:\.\d+)?%\b",
        "currency": r"\$\d+(?:,\d{3})*(?:\.\d{2})?\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    def __init__(
        self,
        patterns: Optional[dict[str, str]] = None,
        include_defaults: bool = True,
    ):
        """Initialize extractor."""
        self.patterns: dict[str, Pattern] = {}

        if include_defaults:
            for name, pattern in self.DEFAULT_PATTERNS.items():
                self.patterns[name] = re.compile(pattern, re.IGNORECASE)

        if patterns:
            for name, pattern in patterns.items():
                self.patterns[name] = re.compile(pattern, re.IGNORECASE)

    def extract(self, text: str) -> list[EntityMatch]:
        """Extract entities from text."""
        entities = []

        for entity_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entity = EntityMatch(
                    text=match.group(),
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                )
                entities.append(entity)

        # Sort by position
        entities.sort(key=lambda e: e.start)

        return entities

    def add_pattern(self, name: str, pattern: str) -> None:
        """Add a custom entity pattern."""
        self.patterns[name] = re.compile(pattern, re.IGNORECASE)


class TableExtractor:
    """Extracts tabular data from text."""

    def extract(self, text: str) -> ExtractionResult:
        """Extract table from text."""
        # Try markdown table format
        if "|" in text:
            result = self._extract_markdown_table(text)
            if result.is_success:
                return result

        # Try whitespace-aligned format
        result = self._extract_whitespace_table(text)
        return result

    def _extract_markdown_table(self, text: str) -> ExtractionResult:
        """Extract markdown-style table."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        table_lines = [line for line in lines if "|" in line]

        if len(table_lines) < 2:
            return ExtractionResult(
                raw_text=text,
                extracted_data={},
                status=ExtractionStatus.FAILED,
                format_detected=ExtractionFormat.TABLE,
                confidence=0.0,
                errors=["No valid table found"],
            )

        # Parse header
        header_line = table_lines[0]
        headers = [cell.strip() for cell in header_line.split("|") if cell.strip()]

        # Skip separator line if present
        data_start = 1
        if len(table_lines) > 1 and re.match(r"^[\s|:-]+$", table_lines[1]):
            data_start = 2

        # Parse rows
        rows = []
        for line in table_lines[data_start:]:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if cells:
                row = dict(zip(headers, cells))
                rows.append(row)

        if not rows:
            return ExtractionResult(
                raw_text=text,
                extracted_data={},
                status=ExtractionStatus.FAILED,
                format_detected=ExtractionFormat.TABLE,
                confidence=0.0,
                errors=["No data rows found in table"],
            )

        return ExtractionResult(
            raw_text=text,
            extracted_data={"headers": headers, "rows": rows, "row_count": len(rows)},
            status=ExtractionStatus.SUCCESS,
            format_detected=ExtractionFormat.TABLE,
            confidence=0.9,
        )

    def _extract_whitespace_table(self, text: str) -> ExtractionResult:
        """Extract whitespace-aligned table."""
        lines = [line for line in text.split("\n") if line.strip()]

        if len(lines) < 2:
            return ExtractionResult(
                raw_text=text,
                extracted_data={},
                status=ExtractionStatus.FAILED,
                format_detected=ExtractionFormat.TABLE,
                confidence=0.0,
                errors=["Insufficient lines for table"],
            )

        # Simple approach: split on multiple spaces
        rows = []
        headers = None

        for line in lines:
            cells = re.split(r"\s{2,}", line.strip())
            if len(cells) > 1:
                if headers is None:
                    headers = cells
                else:
                    if len(cells) == len(headers):
                        rows.append(dict(zip(headers, cells)))

        if not headers or not rows:
            return ExtractionResult(
                raw_text=text,
                extracted_data={},
                status=ExtractionStatus.FAILED,
                format_detected=ExtractionFormat.TABLE,
                confidence=0.0,
                errors=["Could not parse whitespace-aligned table"],
            )

        return ExtractionResult(
            raw_text=text,
            extracted_data={"headers": headers, "rows": rows, "row_count": len(rows)},
            status=ExtractionStatus.SUCCESS,
            format_detected=ExtractionFormat.TABLE,
            confidence=0.7,
        )


class StructuredExtractor:
    """Main class for structured data extraction."""

    def __init__(
        self,
        schema: Optional[ExtractionSchema] = None,
        auto_detect_format: bool = True,
    ):
        """Initialize extractor."""
        self.schema = schema
        self.auto_detect_format = auto_detect_format

        # Initialize sub-extractors
        self.json_extractor = JSONExtractor()
        self.kv_extractor = KeyValueExtractor()
        self.list_extractor = ListExtractor()
        self.entity_extractor = EntityExtractor()
        self.table_extractor = TableExtractor()

    def extract(
        self,
        text: str,
        expected_format: Optional[ExtractionFormat] = None,
    ) -> ExtractionResult:
        """Extract structured data from text."""
        if not text or not text.strip():
            return ExtractionResult(
                raw_text=text,
                extracted_data={},
                status=ExtractionStatus.FAILED,
                format_detected=ExtractionFormat.FREE_TEXT,
                confidence=0.0,
                errors=["Empty input text"],
            )

        # Try specific format if provided
        if expected_format:
            result = self._extract_format(text, expected_format)
        elif self.auto_detect_format:
            result = self._auto_extract(text)
        else:
            result = self.json_extractor.extract(text)

        # Apply schema validation if provided
        if self.schema and result.is_success:
            is_valid, errors = self.schema.validate(result.extracted_data)
            if not is_valid:
                result.status = ExtractionStatus.VALIDATION_ERROR
                result.errors.extend(errors)

        return result

    def _extract_format(self, text: str, format_type: ExtractionFormat) -> ExtractionResult:
        """Extract using specific format."""
        if format_type == ExtractionFormat.JSON:
            return self.json_extractor.extract(text)
        elif format_type == ExtractionFormat.KEY_VALUE:
            return self.kv_extractor.extract(text)
        elif format_type == ExtractionFormat.LIST:
            return self.list_extractor.extract(text)
        elif format_type == ExtractionFormat.TABLE:
            return self.table_extractor.extract(text)
        else:
            return self._auto_extract(text)

    def _auto_extract(self, text: str) -> ExtractionResult:
        """Auto-detect format and extract."""
        # Try each format in order of confidence
        extractors = [
            (self.json_extractor, ExtractionFormat.JSON),
            (self.table_extractor, ExtractionFormat.TABLE),
            (self.kv_extractor, ExtractionFormat.KEY_VALUE),
            (self.list_extractor, ExtractionFormat.LIST),
        ]

        best_result = None
        best_confidence = 0.0

        for extractor, _format_type in extractors:
            try:
                result = extractor.extract(text)
                if result.is_success and result.confidence > best_confidence:
                    best_result = result
                    best_confidence = result.confidence
            except Exception:
                continue

        if best_result:
            return best_result

        # Fallback: return as free text
        return ExtractionResult(
            raw_text=text,
            extracted_data={"text": text.strip()},
            status=ExtractionStatus.PARTIAL,
            format_detected=ExtractionFormat.FREE_TEXT,
            confidence=0.3,
            warnings=["No structured format detected, returning raw text"],
        )

    def extract_with_schema(
        self,
        text: str,
        schema: ExtractionSchema,
    ) -> ExtractionResult:
        """Extract data and validate against schema."""
        result = self.extract(text)

        if result.is_success:
            is_valid, errors = schema.validate(result.extracted_data)
            if not is_valid:
                result.status = ExtractionStatus.VALIDATION_ERROR
                result.errors.extend(errors)

        return result

    def extract_fields(
        self,
        text: str,
        field_names: list[str],
    ) -> dict[str, Any]:
        """Extract specific fields from text."""
        # Try key-value extraction first
        kv_result = self.kv_extractor.extract(text, keys=field_names)
        if kv_result.is_success:
            return kv_result.extracted_data

        # Try JSON extraction
        json_result = self.json_extractor.extract(text)
        if json_result.is_success:
            return {k: v for k, v in json_result.extracted_data.items() if k in field_names}

        return {}

    def extract_entities(self, text: str) -> list[EntityMatch]:
        """Extract entities from text."""
        return self.entity_extractor.extract(text)


class ResponseParser:
    """Parses structured responses with fallback strategies."""

    def __init__(self, extractors: Optional[list[Any]] = None):
        """Initialize parser."""
        self.extractors = extractors or [
            JSONExtractor(),
            KeyValueExtractor(),
            ListExtractor(),
        ]

    def parse(
        self,
        response: str,
        expected_fields: Optional[list[str]] = None,
    ) -> ExtractionResult:
        """Parse response using multiple strategies."""
        for extractor in self.extractors:
            try:
                result = extractor.extract(response)
                if result.is_success:
                    # Filter to expected fields if provided
                    if expected_fields and result.extracted_data:
                        filtered = {
                            k: v
                            for k, v in result.extracted_data.items()
                            if k in expected_fields or k == "items"
                        }
                        result.extracted_data = filtered
                    return result
            except Exception:
                continue

        return ExtractionResult(
            raw_text=response,
            extracted_data={},
            status=ExtractionStatus.FAILED,
            format_detected=ExtractionFormat.FREE_TEXT,
            confidence=0.0,
            errors=["All extraction strategies failed"],
        )


class TypeCoercer:
    """Coerces values to expected types."""

    @staticmethod
    def coerce(value: Any, target_type: FieldType) -> tuple[Any, bool]:
        """Coerce value to target type."""
        if value is None:
            return None, True

        try:
            if target_type == FieldType.STRING:
                return str(value), True
            elif target_type == FieldType.INTEGER:
                if isinstance(value, str):
                    value = value.replace(",", "")
                return int(float(value)), True
            elif target_type == FieldType.FLOAT:
                if isinstance(value, str):
                    value = value.replace(",", "")
                return float(value), True
            elif target_type == FieldType.BOOLEAN:
                if isinstance(value, bool):
                    return value, True
                if isinstance(value, str):
                    if value.lower() in ("true", "yes", "1", "on"):
                        return True, True
                    if value.lower() in ("false", "no", "0", "off"):
                        return False, True
                return bool(value), True
            elif target_type == FieldType.LIST:
                if isinstance(value, list):
                    return value, True
                if isinstance(value, str):
                    return [v.strip() for v in value.split(",")], True
                return [value], True
            elif target_type == FieldType.DICT:
                if isinstance(value, dict):
                    return value, True
                return {"value": value}, True
            else:
                return value, True
        except (ValueError, TypeError):
            return value, False


# Convenience functions
def extract_json(text: str) -> ExtractionResult:
    """Extract JSON from text.

    Args:
        text: Text potentially containing JSON

    Returns:
        ExtractionResult with extracted JSON data
    """
    extractor = JSONExtractor()
    return extractor.extract(text)


def extract_key_values(
    text: str,
    keys: Optional[list[str]] = None,
) -> ExtractionResult:
    """Extract key-value pairs from text.

    Args:
        text: Text containing key-value pairs
        keys: Optional list of expected keys

    Returns:
        ExtractionResult with extracted key-value pairs
    """
    extractor = KeyValueExtractor()
    return extractor.extract(text, keys=keys)


def extract_list(text: str) -> ExtractionResult:
    """Extract list items from text.

    Args:
        text: Text containing a list

    Returns:
        ExtractionResult with extracted list items
    """
    extractor = ListExtractor()
    return extractor.extract(text)


def extract_entities(text: str) -> list[EntityMatch]:
    """Extract named entities from text.

    Args:
        text: Text to extract entities from

    Returns:
        List of matched entities
    """
    extractor = EntityExtractor()
    return extractor.extract(text)


def extract_table(text: str) -> ExtractionResult:
    """Extract table from text.

    Args:
        text: Text containing a table

    Returns:
        ExtractionResult with extracted table data
    """
    extractor = TableExtractor()
    return extractor.extract(text)


def extract_structured(
    text: str,
    expected_format: Optional[ExtractionFormat] = None,
) -> ExtractionResult:
    """Extract structured data with auto-detection.

    Args:
        text: Text to extract from
        expected_format: Optional expected format hint

    Returns:
        ExtractionResult with extracted data
    """
    extractor = StructuredExtractor()
    return extractor.extract(text, expected_format=expected_format)


def validate_extraction(
    data: dict[str, Any],
    schema: ExtractionSchema,
) -> tuple[bool, list[str]]:
    """Validate extracted data against schema.

    Args:
        data: Extracted data dictionary
        schema: Schema to validate against

    Returns:
        Tuple of (is_valid, list of errors)
    """
    return schema.validate(data)


def create_schema(
    fields: list[dict[str, Any]],
    strict: bool = True,
) -> ExtractionSchema:
    """Create extraction schema from field definitions.

    Args:
        fields: List of field definition dicts
        strict: Whether to enforce strict validation

    Returns:
        ExtractionSchema instance
    """
    field_schemas = []
    for field_def in fields:
        field_type = FieldType(field_def.get("type", "string"))
        field_schema = FieldSchema(
            name=field_def["name"],
            field_type=field_type,
            required=field_def.get("required", True),
            default=field_def.get("default"),
            description=field_def.get("description", ""),
            pattern=field_def.get("pattern"),
            min_value=field_def.get("min_value"),
            max_value=field_def.get("max_value"),
            min_length=field_def.get("min_length"),
            max_length=field_def.get("max_length"),
            allowed_values=field_def.get("allowed_values"),
        )
        field_schemas.append(field_schema)

    return ExtractionSchema(fields=field_schemas, strict=strict)
