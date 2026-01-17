"""Structured output parsing utilities for LLM responses.

This module provides tools for extracting and validating structured data
from LLM outputs, including JSON, code blocks, lists, and custom formats.

Key features:
- JSON extraction and validation
- Code block extraction (with language detection)
- List and table parsing
- Schema validation
- Format detection and conversion
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Pattern,
    Tuple,
    Type,
    TypeVar,
    Union,
)


T = TypeVar("T")


class OutputFormat(Enum):
    """Common output formats from LLMs."""

    JSON = "json"
    MARKDOWN = "markdown"
    CODE = "code"
    LIST = "list"
    TABLE = "table"
    XML = "xml"
    YAML = "yaml"
    PLAIN = "plain"
    UNKNOWN = "unknown"


@dataclass
class ParseResult(Generic[T]):
    """Result of a parsing operation.

    Attributes:
        success: Whether parsing succeeded.
        value: The parsed value (if successful).
        raw: The original raw text.
        error: Error message (if failed).
        format_detected: The detected output format.
    """

    success: bool
    value: Optional[T]
    raw: str
    error: Optional[str] = None
    format_detected: OutputFormat = OutputFormat.UNKNOWN

    @property
    def is_ok(self) -> bool:
        """Alias for success."""
        return self.success

    def unwrap(self) -> T:
        """Get the value or raise an error.

        Returns:
            The parsed value.

        Raises:
            ValueError: If parsing failed.
        """
        if not self.success or self.value is None:
            raise ValueError(self.error or "Parsing failed")
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get the value or return a default.

        Args:
            default: Default value if parsing failed.

        Returns:
            The parsed value or default.
        """
        if self.success and self.value is not None:
            return self.value
        return default

    def map(self, func: Callable[[T], Any]) -> "ParseResult":
        """Apply a function to the parsed value.

        Args:
            func: Function to apply.

        Returns:
            New ParseResult with transformed value.
        """
        if self.success and self.value is not None:
            try:
                new_value = func(self.value)
                return ParseResult(
                    success=True,
                    value=new_value,
                    raw=self.raw,
                    format_detected=self.format_detected,
                )
            except Exception as e:
                return ParseResult(
                    success=False,
                    value=None,
                    raw=self.raw,
                    error=str(e),
                    format_detected=self.format_detected,
                )
        return self


@dataclass
class CodeBlock:
    """A code block extracted from text.

    Attributes:
        code: The code content.
        language: The programming language (if specified).
        start_line: Starting line number in original text.
        end_line: Ending line number in original text.
    """

    code: str
    language: Optional[str] = None
    start_line: int = 0
    end_line: int = 0

    def __str__(self) -> str:
        return self.code


@dataclass
class TableData:
    """Parsed table data.

    Attributes:
        headers: Column headers.
        rows: Data rows.
        raw: Original table text.
    """

    headers: List[str]
    rows: List[List[str]]
    raw: str = ""

    @property
    def num_rows(self) -> int:
        """Number of data rows."""
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        """Number of columns."""
        return len(self.headers)

    def to_dicts(self) -> List[Dict[str, str]]:
        """Convert rows to list of dictionaries."""
        return [dict(zip(self.headers, row)) for row in self.rows]

    def get_column(self, name: str) -> List[str]:
        """Get all values in a column by name."""
        if name not in self.headers:
            raise KeyError(f"Column not found: {name}")
        idx = self.headers.index(name)
        return [row[idx] if idx < len(row) else "" for row in self.rows]


class JSONParser:
    """Parser for JSON output from LLMs.

    Handles common issues like markdown code blocks, trailing commas,
    and partial JSON.
    """

    # Patterns for extracting JSON
    JSON_BLOCK_PATTERN = re.compile(
        r"```(?:json)?\s*\n?([\s\S]*?)\n?```", re.MULTILINE
    )
    JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}", re.MULTILINE)
    JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*\]", re.MULTILINE)

    @classmethod
    def parse(cls, text: str, strict: bool = False) -> ParseResult[Any]:
        """Parse JSON from LLM output.

        Args:
            text: The text to parse.
            strict: If True, don't try to fix common issues.

        Returns:
            ParseResult containing the parsed JSON or error.
        """
        # Try direct parsing first
        try:
            value = json.loads(text)
            return ParseResult(
                success=True,
                value=value,
                raw=text,
                format_detected=OutputFormat.JSON,
            )
        except json.JSONDecodeError:
            pass

        if strict:
            return ParseResult(
                success=False,
                value=None,
                raw=text,
                error="Invalid JSON (strict mode)",
                format_detected=OutputFormat.UNKNOWN,
            )

        # Try extracting from code block
        json_text = cls._extract_from_code_block(text)
        if json_text:
            try:
                value = json.loads(json_text)
                return ParseResult(
                    success=True,
                    value=value,
                    raw=text,
                    format_detected=OutputFormat.JSON,
                )
            except json.JSONDecodeError:
                pass

        # Try extracting object or array pattern
        json_text = cls._extract_json_pattern(text)
        if json_text:
            # Try to fix common issues
            fixed_text = cls._fix_common_issues(json_text)
            try:
                value = json.loads(fixed_text)
                return ParseResult(
                    success=True,
                    value=value,
                    raw=text,
                    format_detected=OutputFormat.JSON,
                )
            except json.JSONDecodeError as e:
                return ParseResult(
                    success=False,
                    value=None,
                    raw=text,
                    error=f"JSON parse error: {e}",
                    format_detected=OutputFormat.JSON,
                )

        return ParseResult(
            success=False,
            value=None,
            raw=text,
            error="No JSON found in text",
            format_detected=OutputFormat.UNKNOWN,
        )

    @classmethod
    def _extract_from_code_block(cls, text: str) -> Optional[str]:
        """Extract JSON from markdown code block."""
        match = cls.JSON_BLOCK_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    @classmethod
    def _extract_json_pattern(cls, text: str) -> Optional[str]:
        """Extract JSON object or array from text."""
        # Try object first
        match = cls.JSON_OBJECT_PATTERN.search(text)
        if match:
            return match.group(0)

        # Try array
        match = cls.JSON_ARRAY_PATTERN.search(text)
        if match:
            return match.group(0)

        return None

    @classmethod
    def _fix_common_issues(cls, json_text: str) -> str:
        """Fix common JSON issues from LLM output."""
        # Remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", json_text)

        # Fix single quotes to double quotes (simple cases)
        # This is risky but common in LLM output
        if "'" in fixed and '"' not in fixed:
            fixed = fixed.replace("'", '"')

        # Remove comments (// style)
        fixed = re.sub(r"//[^\n]*\n", "\n", fixed)

        return fixed

    @classmethod
    def extract_all(cls, text: str) -> List[ParseResult[Any]]:
        """Extract all JSON objects/arrays from text.

        Args:
            text: The text to parse.

        Returns:
            List of ParseResults for each JSON found.
        """
        results = []

        # Find all code blocks
        for match in cls.JSON_BLOCK_PATTERN.finditer(text):
            json_text = match.group(1).strip()
            try:
                value = json.loads(json_text)
                results.append(
                    ParseResult(
                        success=True,
                        value=value,
                        raw=json_text,
                        format_detected=OutputFormat.JSON,
                    )
                )
            except json.JSONDecodeError:
                pass

        # Find JSON objects not in code blocks
        # Remove code blocks first
        text_without_blocks = cls.JSON_BLOCK_PATTERN.sub("", text)

        for pattern in [cls.JSON_OBJECT_PATTERN, cls.JSON_ARRAY_PATTERN]:
            for match in pattern.finditer(text_without_blocks):
                json_text = match.group(0)
                try:
                    value = json.loads(json_text)
                    results.append(
                        ParseResult(
                            success=True,
                            value=value,
                            raw=json_text,
                            format_detected=OutputFormat.JSON,
                        )
                    )
                except json.JSONDecodeError:
                    pass

        return results


class CodeBlockParser:
    """Parser for code blocks in LLM output."""

    # Pattern for fenced code blocks
    FENCED_PATTERN = re.compile(
        r"```(\w*)\s*\n([\s\S]*?)\n```", re.MULTILINE
    )

    # Pattern for indented code blocks (4 spaces or 1 tab)
    INDENTED_PATTERN = re.compile(
        r"(?:^(?:    |\t).*$\n?)+", re.MULTILINE
    )

    # Common language aliases
    LANGUAGE_ALIASES = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "rb": "ruby",
        "sh": "bash",
        "shell": "bash",
        "yml": "yaml",
        "": None,
    }

    @classmethod
    def parse(cls, text: str) -> List[CodeBlock]:
        """Extract all code blocks from text.

        Args:
            text: The text to parse.

        Returns:
            List of CodeBlock objects.
        """
        blocks = []

        # Find fenced code blocks
        for match in cls.FENCED_PATTERN.finditer(text):
            language = match.group(1).lower()
            language = cls.LANGUAGE_ALIASES.get(language, language)
            code = match.group(2)

            # Calculate line numbers
            start = text[:match.start()].count("\n")
            end = start + code.count("\n") + 2  # +2 for fences

            blocks.append(
                CodeBlock(
                    code=code,
                    language=language,
                    start_line=start,
                    end_line=end,
                )
            )

        return blocks

    @classmethod
    def parse_first(cls, text: str) -> Optional[CodeBlock]:
        """Extract the first code block.

        Args:
            text: The text to parse.

        Returns:
            First CodeBlock or None.
        """
        blocks = cls.parse(text)
        return blocks[0] if blocks else None

    @classmethod
    def parse_by_language(cls, text: str, language: str) -> List[CodeBlock]:
        """Extract code blocks of a specific language.

        Args:
            text: The text to parse.
            language: The language to filter by.

        Returns:
            List of matching CodeBlocks.
        """
        language = language.lower()
        language = cls.LANGUAGE_ALIASES.get(language, language)

        blocks = cls.parse(text)
        return [b for b in blocks if b.language == language]

    @classmethod
    def extract_python(cls, text: str) -> List[str]:
        """Extract Python code blocks.

        Args:
            text: The text to parse.

        Returns:
            List of Python code strings.
        """
        return [b.code for b in cls.parse_by_language(text, "python")]

    @classmethod
    def extract_sql(cls, text: str) -> List[str]:
        """Extract SQL code blocks.

        Args:
            text: The text to parse.

        Returns:
            List of SQL code strings.
        """
        return [b.code for b in cls.parse_by_language(text, "sql")]


class ListParser:
    """Parser for lists in LLM output."""

    # Patterns for different list formats
    NUMBERED_PATTERN = re.compile(r"^\s*(\d+)[.)\]]\s*(.+)$", re.MULTILINE)
    BULLET_PATTERN = re.compile(r"^\s*[-*+]\s*(.+)$", re.MULTILINE)
    LETTERED_PATTERN = re.compile(r"^\s*([a-zA-Z])[.)\]]\s*(.+)$", re.MULTILINE)

    @classmethod
    def parse(cls, text: str) -> List[str]:
        """Extract list items from text.

        Handles numbered lists, bullet lists, and lettered lists.

        Args:
            text: The text to parse.

        Returns:
            List of item strings.
        """
        items = []

        # Try numbered list
        numbered = cls.NUMBERED_PATTERN.findall(text)
        if numbered:
            items = [item[1].strip() for item in numbered]
            return items

        # Try bullet list
        bullets = cls.BULLET_PATTERN.findall(text)
        if bullets:
            items = [item.strip() for item in bullets]
            return items

        # Try lettered list
        lettered = cls.LETTERED_PATTERN.findall(text)
        if lettered:
            items = [item[1].strip() for item in lettered]
            return items

        return items

    @classmethod
    def parse_nested(cls, text: str) -> List[Union[str, List]]:
        """Parse nested lists based on indentation.

        Args:
            text: The text to parse.

        Returns:
            Nested list structure.
        """
        lines = text.split("\n")
        result: List[Union[str, List]] = []
        stack: List[Tuple[int, List]] = [(0, result)]

        for line in lines:
            if not line.strip():
                continue

            # Calculate indentation
            indent = len(line) - len(line.lstrip())

            # Extract item text
            match = cls.BULLET_PATTERN.match(line) or cls.NUMBERED_PATTERN.match(line)
            if match:
                item_text = match.group(1) if isinstance(match.group(1), str) and not match.group(1).isdigit() else match.groups()[-1]
                item_text = item_text.strip()

                # Find appropriate level
                while stack and stack[-1][0] >= indent and len(stack) > 1:
                    stack.pop()

                current_list = stack[-1][1]
                current_list.append(item_text)

        return result


class TableParser:
    """Parser for tables in LLM output."""

    # Pattern for markdown table
    MD_TABLE_PATTERN = re.compile(
        r"^\|(.+)\|\s*\n\|[-:\s|]+\|\s*\n((?:\|.+\|\s*\n?)+)",
        re.MULTILINE,
    )

    @classmethod
    def parse(cls, text: str) -> Optional[TableData]:
        """Parse a markdown table from text.

        Args:
            text: The text containing a table.

        Returns:
            TableData or None if no table found.
        """
        match = cls.MD_TABLE_PATTERN.search(text)
        if not match:
            return None

        header_row = match.group(1)
        data_rows = match.group(2)

        # Parse headers
        headers = [h.strip() for h in header_row.split("|") if h.strip()]

        # Parse data rows
        rows = []
        for line in data_rows.strip().split("\n"):
            if line.strip():
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if cells:
                    rows.append(cells)

        return TableData(
            headers=headers,
            rows=rows,
            raw=match.group(0),
        )

    @classmethod
    def parse_all(cls, text: str) -> List[TableData]:
        """Parse all markdown tables from text.

        Args:
            text: The text containing tables.

        Returns:
            List of TableData objects.
        """
        tables = []
        for match in cls.MD_TABLE_PATTERN.finditer(text):
            header_row = match.group(1)
            data_rows = match.group(2)

            headers = [h.strip() for h in header_row.split("|") if h.strip()]
            rows = []
            for line in data_rows.strip().split("\n"):
                if line.strip():
                    cells = [c.strip() for c in line.split("|") if c.strip()]
                    if cells:
                        rows.append(cells)

            tables.append(
                TableData(headers=headers, rows=rows, raw=match.group(0))
            )

        return tables


class AnswerExtractor:
    """Extract specific answer formats from LLM responses."""

    # Patterns for common answer formats
    FINAL_ANSWER_PATTERNS = [
        re.compile(r"(?:final\s+)?answer\s*[:=]\s*(.+?)(?:\n|$)", re.IGNORECASE),
        re.compile(r"(?:the\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\n|$)", re.IGNORECASE),
        re.compile(r"(?:therefore|thus|hence)[,:]?\s*(.+?)(?:\n|$)", re.IGNORECASE),
        re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE),  # LaTeX boxed
    ]

    YES_NO_PATTERNS = [
        re.compile(r"^(yes|no)\b", re.IGNORECASE | re.MULTILINE),
        re.compile(r"(?:answer|response)\s*[:=]?\s*(yes|no)\b", re.IGNORECASE),
    ]

    CHOICE_PATTERNS = [
        re.compile(r"(?:the\s+)?answer\s+is\s+([A-E])\b", re.IGNORECASE),
        re.compile(r"(?:answer|choice)\s*[:=]\s*([A-E])\b", re.IGNORECASE),
        re.compile(r"\b([A-E])\s+is\s+(?:correct|the\s+answer)", re.IGNORECASE),
    ]

    NUMBER_PATTERNS = [
        re.compile(r"(?:answer|result)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"=\s*(-?\d+(?:\.\d+)?)\s*$", re.MULTILINE),
    ]

    @classmethod
    def extract_final_answer(cls, text: str) -> Optional[str]:
        """Extract the final answer from a response.

        Args:
            text: The response text.

        Returns:
            The extracted answer or None.
        """
        for pattern in cls.FINAL_ANSWER_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return None

    @classmethod
    def extract_yes_no(cls, text: str) -> Optional[bool]:
        """Extract a yes/no answer.

        Args:
            text: The response text.

        Returns:
            True for yes, False for no, None if not found.
        """
        for pattern in cls.YES_NO_PATTERNS:
            match = pattern.search(text)
            if match:
                answer = match.group(1).lower()
                return answer == "yes"
        return None

    @classmethod
    def extract_choice(cls, text: str) -> Optional[str]:
        """Extract a multiple choice answer (A-E).

        Args:
            text: The response text.

        Returns:
            The choice letter or None.
        """
        for pattern in cls.CHOICE_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).upper()
        return None

    @classmethod
    def extract_number(cls, text: str) -> Optional[float]:
        """Extract a numeric answer.

        Args:
            text: The response text.

        Returns:
            The number or None.
        """
        for pattern in cls.NUMBER_PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None

    @classmethod
    def extract_all_numbers(cls, text: str) -> List[float]:
        """Extract all numbers from text.

        Args:
            text: The response text.

        Returns:
            List of numbers found.
        """
        pattern = re.compile(r"-?\d+(?:\.\d+)?")
        matches = pattern.findall(text)
        return [float(m) for m in matches]


class OutputDetector:
    """Detect the format of LLM output."""

    @classmethod
    def detect(cls, text: str) -> OutputFormat:
        """Detect the primary format of the text.

        Args:
            text: The text to analyze.

        Returns:
            The detected OutputFormat.
        """
        text = text.strip()

        # Check for JSON
        if cls._looks_like_json(text):
            return OutputFormat.JSON

        # Check for code blocks
        if "```" in text:
            return OutputFormat.CODE

        # Check for markdown table
        if TableParser.MD_TABLE_PATTERN.search(text):
            return OutputFormat.TABLE

        # Check for XML
        if cls._looks_like_xml(text):
            return OutputFormat.XML

        # Check for YAML
        if cls._looks_like_yaml(text):
            return OutputFormat.YAML

        # Check for list
        if cls._looks_like_list(text):
            return OutputFormat.LIST

        # Check for markdown (headers, links, etc.)
        if cls._looks_like_markdown(text):
            return OutputFormat.MARKDOWN

        return OutputFormat.PLAIN

    @classmethod
    def _looks_like_json(cls, text: str) -> bool:
        """Check if text looks like JSON."""
        text = text.strip()
        if text.startswith(("{", "[")) and text.endswith(("}", "]")):
            try:
                json.loads(text)
                return True
            except json.JSONDecodeError:
                pass
        return False

    @classmethod
    def _looks_like_xml(cls, text: str) -> bool:
        """Check if text looks like XML."""
        text = text.strip()
        return (
            text.startswith("<") and
            text.endswith(">") and
            re.search(r"<\w+[^>]*>", text) is not None
        )

    @classmethod
    def _looks_like_yaml(cls, text: str) -> bool:
        """Check if text looks like YAML."""
        lines = text.split("\n")
        yaml_indicators = 0
        key_value_count = 0
        for line in lines[:10]:  # Check first 10 lines
            # Key: value pattern (YAML specific)
            if re.match(r"^\w+:\s+\S", line):
                key_value_count += 1
                yaml_indicators += 1
        # Need at least one key: value pattern to be YAML
        return yaml_indicators >= 2 and key_value_count >= 1

    @classmethod
    def _looks_like_list(cls, text: str) -> bool:
        """Check if text looks like a list."""
        lines = text.strip().split("\n")
        list_lines = 0
        for line in lines:
            if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+[.)\]]\s+", line):
                list_lines += 1
        return list_lines >= 2 and list_lines / len(lines) > 0.5

    @classmethod
    def _looks_like_markdown(cls, text: str) -> bool:
        """Check if text has markdown features."""
        md_patterns = [
            r"^#{1,6}\s+",  # Headers
            r"\[.+\]\(.+\)",  # Links
            r"\*\*.+\*\*",  # Bold
            r"`.+`",  # Inline code
        ]
        for pattern in md_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False


def parse_json(text: str, strict: bool = False) -> ParseResult[Any]:
    """Parse JSON from LLM output.

    Convenience function for JSONParser.parse().
    """
    return JSONParser.parse(text, strict=strict)


def parse_code(text: str) -> List[CodeBlock]:
    """Extract code blocks from text.

    Convenience function for CodeBlockParser.parse().
    """
    return CodeBlockParser.parse(text)


def parse_list(text: str) -> List[str]:
    """Extract list items from text.

    Convenience function for ListParser.parse().
    """
    return ListParser.parse(text)


def parse_table(text: str) -> Optional[TableData]:
    """Parse a table from text.

    Convenience function for TableParser.parse().
    """
    return TableParser.parse(text)


def detect_format(text: str) -> OutputFormat:
    """Detect the format of text.

    Convenience function for OutputDetector.detect().
    """
    return OutputDetector.detect(text)


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from text.

    Convenience function for AnswerExtractor.extract_final_answer().
    """
    return AnswerExtractor.extract_final_answer(text)
