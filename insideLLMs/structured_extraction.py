"""
Model output parsing and structured extraction utilities.

Provides tools for:
- Extracting structured data from LLM responses
- Parsing specific response formats (JSON, XML, YAML)
- Schema validation and type coercion
- Entity and field extraction
- Multi-format response handling
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Type, Union


class ExtractionFormat(Enum):
    """Supported extraction formats."""

    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    KEY_VALUE = "key_value"
    LIST = "list"
    TABLE = "table"
    MARKDOWN = "markdown"
    FREE_TEXT = "free_text"


class FieldType(Enum):
    """Field data types for schema validation."""

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
    """Status of extraction operation."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    VALIDATION_ERROR = "validation_error"


@dataclass
class FieldSchema:
    """Schema definition for a field."""

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
    allowed_values: Optional[List[Any]] = None
    nested_schema: Optional[List["FieldSchema"]] = None

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a value against this schema."""
        if value is None:
            if self.required:
                return False, f"Field '{self.name}' is required"
            return True, None

        # Type validation
        type_valid, type_error = self._validate_type(value)
        if not type_valid:
            return False, type_error

        # Pattern validation
        if self.pattern and isinstance(value, str):
            if not re.match(self.pattern, value):
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
        if self.allowed_values is not None:
            if value not in self.allowed_values:
                return False, f"Field '{self.name}' must be one of {self.allowed_values}"

        return True, None

    def _validate_type(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate value type."""
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
    """Result of structured extraction."""

    raw_text: str
    extracted_data: Dict[str, Any]
    status: ExtractionStatus
    format_detected: ExtractionFormat
    confidence: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionSchema:
    """Schema for structured extraction."""

    fields: List[FieldSchema]
    strict: bool = True
    allow_extra_fields: bool = False
    name: str = ""
    description: str = ""

    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
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

    def _find_json_candidates(self, text: str) -> List[Tuple[str, float]]:
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

    def _is_valid_json_structure(self, text: str) -> bool:
        """Check if text looks like valid JSON structure."""
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return True
        if text.startswith("[") and text.endswith("]"):
            return True
        return False


class KeyValueExtractor:
    """Extracts key-value pairs from text."""

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        case_sensitive: bool = False,
    ):
        """Initialize extractor."""
        self.separators = separators or [":", "=", "->", "-"]
        self.case_sensitive = case_sensitive

    def extract(self, text: str, keys: Optional[List[str]] = None) -> ExtractionResult:
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
        patterns: Optional[Dict[str, str]] = None,
        include_defaults: bool = True,
    ):
        """Initialize extractor."""
        self.patterns: Dict[str, Pattern] = {}

        if include_defaults:
            for name, pattern in self.DEFAULT_PATTERNS.items():
                self.patterns[name] = re.compile(pattern, re.IGNORECASE)

        if patterns:
            for name, pattern in patterns.items():
                self.patterns[name] = re.compile(pattern, re.IGNORECASE)

    def extract(self, text: str) -> List[EntityMatch]:
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

        for extractor, format_type in extractors:
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
        field_names: List[str],
    ) -> Dict[str, Any]:
        """Extract specific fields from text."""
        # Try key-value extraction first
        kv_result = self.kv_extractor.extract(text, keys=field_names)
        if kv_result.is_success:
            return kv_result.extracted_data

        # Try JSON extraction
        json_result = self.json_extractor.extract(text)
        if json_result.is_success:
            return {
                k: v
                for k, v in json_result.extracted_data.items()
                if k in field_names
            }

        return {}

    def extract_entities(self, text: str) -> List[EntityMatch]:
        """Extract entities from text."""
        return self.entity_extractor.extract(text)


class ResponseParser:
    """Parses structured responses with fallback strategies."""

    def __init__(self, extractors: Optional[List[Any]] = None):
        """Initialize parser."""
        self.extractors = extractors or [
            JSONExtractor(),
            KeyValueExtractor(),
            ListExtractor(),
        ]

    def parse(
        self,
        response: str,
        expected_fields: Optional[List[str]] = None,
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
    def coerce(value: Any, target_type: FieldType) -> Tuple[Any, bool]:
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
    keys: Optional[List[str]] = None,
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


def extract_entities(text: str) -> List[EntityMatch]:
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
    data: Dict[str, Any],
    schema: ExtractionSchema,
) -> Tuple[bool, List[str]]:
    """Validate extracted data against schema.

    Args:
        data: Extracted data dictionary
        schema: Schema to validate against

    Returns:
        Tuple of (is_valid, list of errors)
    """
    return schema.validate(data)


def create_schema(
    fields: List[Dict[str, Any]],
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
