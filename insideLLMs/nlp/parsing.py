"""Structured output parsing utilities for LLM responses.

This module provides comprehensive tools for extracting and validating structured
data from Large Language Model (LLM) outputs. It handles common output formats
including JSON, code blocks, lists, tables, and various answer formats.

Overview
--------
When working with LLMs, responses often contain structured data embedded within
natural language text. This module provides robust parsers that can extract and
validate this data, handling common issues like markdown formatting, trailing
commas in JSON, and inconsistent list formatting.

Key Components
--------------
- **JSONParser**: Extract and validate JSON from text, handling code blocks and
  common LLM-specific formatting issues.
- **CodeBlockParser**: Extract fenced code blocks with language detection.
- **ListParser**: Parse numbered, bulleted, and lettered lists.
- **TableParser**: Extract markdown tables into structured data.
- **AnswerExtractor**: Extract specific answer formats (yes/no, multiple choice,
  numeric, final answers).
- **OutputDetector**: Automatically detect the format of LLM output.

Examples
--------
Basic JSON extraction from LLM response:

>>> from insideLLMs.nlp.parsing import parse_json
>>> response = '''
... Here is the user data:
... ```json
... {"name": "Alice", "age": 30}
... ```
... '''
>>> result = parse_json(response)
>>> result.success
True
>>> result.value
{'name': 'Alice', 'age': 30}

Extracting code blocks:

>>> from insideLLMs.nlp.parsing import parse_code
>>> response = '''
... Here's a Python function:
... ```python
... def greet(name):
...     return f"Hello, {name}!"
... ```
... '''
>>> blocks = parse_code(response)
>>> blocks[0].language
'python'
>>> print(blocks[0].code)
def greet(name):
    return f"Hello, {name}!"

Parsing lists from responses:

>>> from insideLLMs.nlp.parsing import parse_list
>>> response = '''
... The top programming languages are:
... 1. Python
... 2. JavaScript
... 3. TypeScript
... '''
>>> items = parse_list(response)
>>> items
['Python', 'JavaScript', 'TypeScript']

Extracting answers:

>>> from insideLLMs.nlp.parsing import extract_answer
>>> response = "After analyzing the data, the answer is: 42"
>>> extract_answer(response)
'42'

Notes
-----
- All parsers are designed to be fault-tolerant and handle malformed input
  gracefully.
- The `ParseResult` class provides a Result-like interface (similar to Rust's
  Result type) for safe error handling.
- Format detection uses heuristics and may not be 100% accurate for ambiguous
  content.

See Also
--------
- json : Python's built-in JSON library
- re : Regular expression operations used internally

Module Attributes
-----------------
T : TypeVar
    Generic type variable used in ParseResult for type-safe parsing.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
)

T = TypeVar("T")


class OutputFormat(Enum):
    """Enumeration of common output formats from LLMs.

    This enum represents the various structured formats that can be detected
    in LLM outputs. It is used by the `OutputDetector` class and stored in
    `ParseResult` objects to indicate the detected format.

    Attributes
    ----------
    JSON : str
        JavaScript Object Notation format. Value: "json"
    MARKDOWN : str
        Markdown formatted text with headers, links, etc. Value: "markdown"
    CODE : str
        Code blocks (fenced with triple backticks). Value: "code"
    LIST : str
        Bulleted or numbered lists. Value: "list"
    TABLE : str
        Markdown tables with headers and rows. Value: "table"
    XML : str
        Extensible Markup Language format. Value: "xml"
    YAML : str
        YAML Ain't Markup Language format. Value: "yaml"
    PLAIN : str
        Plain text without detected structure. Value: "plain"
    UNKNOWN : str
        Format could not be determined. Value: "unknown"

    Examples
    --------
    Using OutputFormat with detect_format:

    >>> from insideLLMs.nlp.parsing import detect_format, OutputFormat
    >>> text = '{"key": "value"}'
    >>> fmt = detect_format(text)
    >>> fmt == OutputFormat.JSON
    True
    >>> fmt.value
    'json'

    Checking format in ParseResult:

    >>> from insideLLMs.nlp.parsing import parse_json
    >>> result = parse_json('{"name": "test"}')
    >>> result.format_detected
    <OutputFormat.JSON: 'json'>

    Using in conditional logic:

    >>> from insideLLMs.nlp.parsing import detect_format, OutputFormat
    >>> response = '''
    ... - Item 1
    ... - Item 2
    ... - Item 3
    ... '''
    >>> fmt = detect_format(response)
    >>> if fmt == OutputFormat.LIST:
    ...     print("Response is a list")
    Response is a list

    See Also
    --------
    OutputDetector : Class that uses this enum for format detection.
    ParseResult : Contains format_detected attribute of this type.
    """

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
    """Result of a parsing operation with success/failure status.

    This class implements a Result-like pattern (similar to Rust's Result type)
    for parsing operations. It encapsulates the parsing outcome, including the
    parsed value on success, error information on failure, and metadata about
    the original input and detected format.

    The generic type parameter T represents the type of the successfully parsed
    value, enabling type-safe parsing operations.

    Parameters
    ----------
    success : bool
        Whether the parsing operation succeeded.
    value : Optional[T]
        The parsed value if successful, None otherwise.
    raw : str
        The original raw text that was parsed.
    error : Optional[str], default=None
        Error message if parsing failed, None on success.
    format_detected : OutputFormat, default=OutputFormat.UNKNOWN
        The detected format of the input text.

    Attributes
    ----------
    success : bool
        Whether the parsing operation succeeded.
    value : Optional[T]
        The parsed value if successful, None otherwise.
    raw : str
        The original raw text that was parsed.
    error : Optional[str]
        Error message if parsing failed, None on success.
    format_detected : OutputFormat
        The detected format of the input text.

    Examples
    --------
    Creating a successful ParseResult:

    >>> result = ParseResult(
    ...     success=True,
    ...     value={"name": "Alice"},
    ...     raw='{"name": "Alice"}',
    ...     format_detected=OutputFormat.JSON
    ... )
    >>> result.is_ok
    True
    >>> result.unwrap()
    {'name': 'Alice'}

    Creating a failed ParseResult:

    >>> result = ParseResult(
    ...     success=False,
    ...     value=None,
    ...     raw='invalid json {',
    ...     error="JSON parse error: Expecting property name",
    ...     format_detected=OutputFormat.UNKNOWN
    ... )
    >>> result.is_ok
    False
    >>> result.error
    "JSON parse error: Expecting property name"

    Using unwrap_or for safe value extraction:

    >>> result = ParseResult(success=False, value=None, raw='bad data')
    >>> result.unwrap_or({'default': True})
    {'default': True}

    Transforming values with map:

    >>> result = ParseResult(
    ...     success=True,
    ...     value=[1, 2, 3],
    ...     raw='[1, 2, 3]'
    ... )
    >>> doubled = result.map(lambda x: [i * 2 for i in x])
    >>> doubled.value
    [2, 4, 6]

    See Also
    --------
    JSONParser.parse : Returns ParseResult[Any] for JSON parsing.
    OutputFormat : Enum used for format_detected attribute.
    """

    success: bool
    value: Optional[T]
    raw: str
    error: Optional[str] = None
    format_detected: OutputFormat = OutputFormat.UNKNOWN

    @property
    def is_ok(self) -> bool:
        """Check if the parsing operation succeeded.

        This is an alias for the `success` attribute, providing a more
        idiomatic interface similar to Rust's Result type.

        Returns
        -------
        bool
            True if parsing succeeded, False otherwise.

        Examples
        --------
        >>> result = ParseResult(success=True, value=42, raw='42')
        >>> result.is_ok
        True
        >>> if result.is_ok:
        ...     print(f"Parsed value: {result.value}")
        Parsed value: 42

        >>> failed = ParseResult(success=False, value=None, raw='bad')
        >>> failed.is_ok
        False
        """
        return self.success

    def unwrap(self) -> T:
        """Get the parsed value or raise an error if parsing failed.

        This method provides direct access to the parsed value when you are
        confident the parsing succeeded. If parsing failed, it raises a
        ValueError with the error message.

        Returns
        -------
        T
            The successfully parsed value.

        Raises
        ------
        ValueError
            If parsing failed (success is False) or value is None.
            The exception message contains the parsing error.

        Examples
        --------
        Successful unwrap:

        >>> result = ParseResult(
        ...     success=True,
        ...     value={'key': 'value'},
        ...     raw='{"key": "value"}'
        ... )
        >>> result.unwrap()
        {'key': 'value'}

        Failed unwrap raises exception:

        >>> result = ParseResult(
        ...     success=False,
        ...     value=None,
        ...     raw='invalid',
        ...     error='Parse error: invalid syntax'
        ... )
        >>> try:
        ...     result.unwrap()
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Parse error: invalid syntax

        Using in a try-except pattern:

        >>> from insideLLMs.nlp.parsing import parse_json
        >>> result = parse_json('{"valid": true}')
        >>> try:
        ...     data = result.unwrap()
        ...     print(f"Got: {data}")
        ... except ValueError as e:
        ...     print(f"Parse failed: {e}")
        Got: {'valid': True}

        See Also
        --------
        unwrap_or : Safe alternative that returns a default on failure.
        is_ok : Check success before unwrapping.
        """
        if not self.success or self.value is None:
            raise ValueError(self.error or "Parsing failed")
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get the parsed value or return a default if parsing failed.

        This method provides safe access to the parsed value with a fallback
        default value. It never raises an exception.

        Parameters
        ----------
        default : T
            The default value to return if parsing failed or value is None.

        Returns
        -------
        T
            The parsed value if successful, otherwise the default value.

        Examples
        --------
        Successful parse returns the value:

        >>> result = ParseResult(
        ...     success=True,
        ...     value={'name': 'Alice'},
        ...     raw='{"name": "Alice"}'
        ... )
        >>> result.unwrap_or({'name': 'Unknown'})
        {'name': 'Alice'}

        Failed parse returns the default:

        >>> result = ParseResult(
        ...     success=False,
        ...     value=None,
        ...     raw='invalid json',
        ...     error='Parse error'
        ... )
        >>> result.unwrap_or({'name': 'Unknown'})
        {'name': 'Unknown'}

        Common pattern with parse_json:

        >>> from insideLLMs.nlp.parsing import parse_json
        >>> config = parse_json('not json').unwrap_or({
        ...     'debug': False,
        ...     'timeout': 30
        ... })
        >>> config
        {'debug': False, 'timeout': 30}

        Using with list values:

        >>> result = ParseResult(success=False, value=None, raw='')
        >>> items = result.unwrap_or([])
        >>> len(items)
        0

        See Also
        --------
        unwrap : Raises exception on failure instead of returning default.
        """
        if self.success and self.value is not None:
            return self.value
        return default

    def map(self, func: Callable[[T], Any]) -> "ParseResult":
        """Apply a transformation function to the parsed value.

        If the ParseResult is successful, applies the given function to the
        value and returns a new ParseResult with the transformed value.
        If the ParseResult is a failure, returns self unchanged.

        If the transformation function raises an exception, returns a new
        failed ParseResult with the exception message as the error.

        Parameters
        ----------
        func : Callable[[T], Any]
            A function to apply to the parsed value. Takes the current value
            and returns a transformed value.

        Returns
        -------
        ParseResult
            A new ParseResult with the transformed value on success,
            or the original failed result, or a new failure if the
            function raised an exception.

        Examples
        --------
        Transforming a successful result:

        >>> result = ParseResult(
        ...     success=True,
        ...     value=[1, 2, 3],
        ...     raw='[1, 2, 3]',
        ...     format_detected=OutputFormat.JSON
        ... )
        >>> summed = result.map(sum)
        >>> summed.value
        6

        Chaining multiple transformations:

        >>> result = ParseResult(success=True, value='hello', raw='hello')
        >>> transformed = result.map(str.upper).map(lambda s: s + '!')
        >>> transformed.value
        'HELLO!'

        Handling transformation errors:

        >>> result = ParseResult(success=True, value='not a number', raw='')
        >>> parsed = result.map(int)
        >>> parsed.success
        False
        >>> 'invalid literal' in parsed.error
        True

        Mapping over a failed result (no-op):

        >>> failed = ParseResult(
        ...     success=False,
        ...     value=None,
        ...     raw='bad',
        ...     error='Original error'
        ... )
        >>> mapped = failed.map(lambda x: x * 2)
        >>> mapped.error
        'Original error'

        Practical example - extracting a field:

        >>> from insideLLMs.nlp.parsing import parse_json
        >>> result = parse_json('{"user": {"name": "Alice", "id": 123}}')
        >>> user_id = result.map(lambda d: d['user']['id'])
        >>> user_id.value
        123

        See Also
        --------
        unwrap : Extract value directly (raises on failure).
        unwrap_or : Extract value with default (never raises).
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
    """A code block extracted from text, typically from LLM output.

    This class represents a fenced code block (using triple backticks) that
    has been extracted from text. It includes the code content, optional
    language identifier, and line position information from the source.

    Parameters
    ----------
    code : str
        The code content inside the block (without the fence markers).
    language : Optional[str], default=None
        The programming language specified after the opening fence.
        None if no language was specified.
    start_line : int, default=0
        The starting line number in the original text (0-indexed).
    end_line : int, default=0
        The ending line number in the original text (0-indexed).

    Attributes
    ----------
    code : str
        The code content inside the block.
    language : Optional[str]
        The programming language identifier (e.g., 'python', 'javascript').
    start_line : int
        Starting line number in the original text.
    end_line : int
        Ending line number in the original text.

    Examples
    --------
    Creating a CodeBlock manually:

    >>> block = CodeBlock(
    ...     code='print("Hello, World!")',
    ...     language='python',
    ...     start_line=5,
    ...     end_line=7
    ... )
    >>> block.code
    'print("Hello, World!")'
    >>> block.language
    'python'

    Using str() to get the code:

    >>> block = CodeBlock(code='SELECT * FROM users;', language='sql')
    >>> str(block)
    'SELECT * FROM users;'

    Extracting from LLM output:

    >>> from insideLLMs.nlp.parsing import parse_code
    >>> response = '''
    ... Here's how to read a file:
    ... ```python
    ... with open('file.txt') as f:
    ...     content = f.read()
    ... ```
    ... '''
    >>> blocks = parse_code(response)
    >>> blocks[0].language
    'python'
    >>> 'open' in blocks[0].code
    True

    Checking for specific languages:

    >>> from insideLLMs.nlp.parsing import CodeBlockParser
    >>> response = '''
    ... ```javascript
    ... console.log("Hello");
    ... ```
    ... ```python
    ... print("Hello")
    ... ```
    ... '''
    >>> python_blocks = CodeBlockParser.parse_by_language(response, 'python')
    >>> len(python_blocks)
    1

    See Also
    --------
    CodeBlockParser : Parser that creates CodeBlock instances.
    parse_code : Convenience function to extract code blocks.
    """

    code: str
    language: Optional[str] = None
    start_line: int = 0
    end_line: int = 0

    def __str__(self) -> str:
        """Return the code content as a string.

        This allows using the CodeBlock directly where a string is expected.

        Returns
        -------
        str
            The code content.

        Examples
        --------
        >>> block = CodeBlock(code='x = 42', language='python')
        >>> str(block)
        'x = 42'
        >>> print(block)
        x = 42
        """
        return self.code


@dataclass
class TableData:
    """Parsed table data from markdown format.

    This class represents a markdown table that has been parsed into a
    structured format with headers and rows. It provides convenient methods
    for accessing the data in different formats.

    Parameters
    ----------
    headers : list[str]
        Column header names from the table's first row.
    rows : list[list[str]]
        Data rows as a list of cell value lists.
    raw : str, default=""
        The original markdown table text.

    Attributes
    ----------
    headers : list[str]
        Column header names.
    rows : list[list[str]]
        Data rows.
    raw : str
        Original table text.
    num_rows : int
        Number of data rows (property).
    num_cols : int
        Number of columns (property).

    Examples
    --------
    Creating TableData manually:

    >>> table = TableData(
    ...     headers=['Name', 'Age', 'City'],
    ...     rows=[
    ...         ['Alice', '30', 'New York'],
    ...         ['Bob', '25', 'Los Angeles'],
    ...         ['Carol', '35', 'Chicago']
    ...     ]
    ... )
    >>> table.num_rows
    3
    >>> table.num_cols
    3

    Converting to list of dictionaries:

    >>> table = TableData(
    ...     headers=['Product', 'Price'],
    ...     rows=[['Widget', '$10'], ['Gadget', '$25']]
    ... )
    >>> table.to_dicts()
    [{'Product': 'Widget', 'Price': '$10'}, {'Product': 'Gadget', 'Price': '$25'}]

    Extracting a single column:

    >>> table = TableData(
    ...     headers=['Name', 'Score'],
    ...     rows=[['Alice', '95'], ['Bob', '87'], ['Carol', '92']]
    ... )
    >>> table.get_column('Score')
    ['95', '87', '92']

    Parsing from LLM response:

    >>> from insideLLMs.nlp.parsing import parse_table
    >>> response = '''
    ... | Language   | Year |
    ... |------------|------|
    ... | Python     | 1991 |
    ... | JavaScript | 1995 |
    ... '''
    >>> table = parse_table(response)
    >>> table.headers
    ['Language', 'Year']
    >>> table.rows[0]
    ['Python', '1991']

    Handling missing column:

    >>> table = TableData(headers=['A', 'B'], rows=[['1', '2']])
    >>> try:
    ...     table.get_column('C')
    ... except KeyError as e:
    ...     print(f"Column not found: {e}")
    Column not found: 'Column not found: C'

    See Also
    --------
    TableParser : Parser that creates TableData instances.
    parse_table : Convenience function to parse tables.
    """

    headers: list[str]
    rows: list[list[str]]
    raw: str = ""

    @property
    def num_rows(self) -> int:
        """Get the number of data rows in the table.

        Returns
        -------
        int
            Number of data rows (excluding the header row).

        Examples
        --------
        >>> table = TableData(
        ...     headers=['A', 'B'],
        ...     rows=[['1', '2'], ['3', '4'], ['5', '6']]
        ... )
        >>> table.num_rows
        3
        """
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        """Get the number of columns in the table.

        Returns
        -------
        int
            Number of columns based on the header count.

        Examples
        --------
        >>> table = TableData(
        ...     headers=['Name', 'Age', 'City', 'Country'],
        ...     rows=[]
        ... )
        >>> table.num_cols
        4
        """
        return len(self.headers)

    def to_dicts(self) -> list[dict[str, str]]:
        """Convert table rows to a list of dictionaries.

        Each row is converted to a dictionary where keys are the column
        headers and values are the corresponding cell values.

        Returns
        -------
        list[dict[str, str]]
            List of dictionaries, one per row.

        Examples
        --------
        >>> table = TableData(
        ...     headers=['id', 'name', 'email'],
        ...     rows=[
        ...         ['1', 'Alice', 'alice@example.com'],
        ...         ['2', 'Bob', 'bob@example.com']
        ...     ]
        ... )
        >>> dicts = table.to_dicts()
        >>> dicts[0]
        {'id': '1', 'name': 'Alice', 'email': 'alice@example.com'}
        >>> dicts[1]['name']
        'Bob'

        Useful for pandas DataFrame creation:

        >>> import pandas as pd  # doctest: +SKIP
        >>> df = pd.DataFrame(table.to_dicts())  # doctest: +SKIP
        """
        return [dict(zip(self.headers, row)) for row in self.rows]

    def get_column(self, name: str) -> list[str]:
        """Get all values in a column by its header name.

        Parameters
        ----------
        name : str
            The column header name to retrieve.

        Returns
        -------
        list[str]
            List of values in the specified column.

        Raises
        ------
        KeyError
            If the column name is not found in headers.

        Examples
        --------
        >>> table = TableData(
        ...     headers=['City', 'Population'],
        ...     rows=[
        ...         ['Tokyo', '13960000'],
        ...         ['Delhi', '31181000'],
        ...         ['Shanghai', '27796000']
        ...     ]
        ... )
        >>> table.get_column('City')
        ['Tokyo', 'Delhi', 'Shanghai']
        >>> table.get_column('Population')
        ['13960000', '31181000', '27796000']

        Converting column values to numbers:

        >>> populations = [int(p) for p in table.get_column('Population')]
        >>> max(populations)
        31181000

        Handling non-existent column:

        >>> try:
        ...     table.get_column('Country')
        ... except KeyError:
        ...     print("Column not found")
        Column not found
        """
        if name not in self.headers:
            raise KeyError(f"Column not found: {name}")
        idx = self.headers.index(name)
        return [row[idx] if idx < len(row) else "" for row in self.rows]


class JSONParser:
    """Parser for JSON output from LLMs with automatic error recovery.

    This class provides robust JSON parsing that handles common issues in
    LLM output, including:

    - JSON wrapped in markdown code blocks
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - JavaScript-style comments

    The parser attempts multiple extraction strategies in order:
    1. Direct JSON parsing
    2. Extraction from markdown code blocks
    3. Pattern matching for JSON objects/arrays with error correction

    Attributes
    ----------
    JSON_BLOCK_PATTERN : re.Pattern
        Regex pattern for markdown JSON code blocks.
    JSON_OBJECT_PATTERN : re.Pattern
        Regex pattern for JSON objects ({...}).
    JSON_ARRAY_PATTERN : re.Pattern
        Regex pattern for JSON arrays ([...]).

    Examples
    --------
    Basic JSON parsing:

    >>> from insideLLMs.nlp.parsing import JSONParser
    >>> result = JSONParser.parse('{"name": "Alice", "age": 30}')
    >>> result.success
    True
    >>> result.value
    {'name': 'Alice', 'age': 30}

    Parsing JSON from markdown code block:

    >>> response = '''
    ... The configuration is:
    ... ```json
    ... {
    ...     "debug": true,
    ...     "timeout": 30
    ... }
    ... ```
    ... '''
    >>> result = JSONParser.parse(response)
    >>> result.value
    {'debug': True, 'timeout': 30}

    Handling trailing commas (common LLM mistake):

    >>> result = JSONParser.parse('{"items": [1, 2, 3,]}')
    >>> result.success
    True
    >>> result.value
    {'items': [1, 2, 3]}

    Strict mode rejects malformed JSON:

    >>> result = JSONParser.parse('{"trailing": "comma",}', strict=True)
    >>> result.success
    False
    >>> 'strict mode' in result.error
    True

    Extracting all JSON from a response:

    >>> response = '''
    ... First object: {"id": 1}
    ... Second object: {"id": 2}
    ... '''
    >>> results = JSONParser.extract_all(response)
    >>> len(results)
    2
    >>> [r.value['id'] for r in results]
    [1, 2]

    Handling parse errors gracefully:

    >>> result = JSONParser.parse('This is not JSON at all')
    >>> result.success
    False
    >>> result.error
    'No JSON found in text'

    See Also
    --------
    parse_json : Convenience function wrapping JSONParser.parse.
    ParseResult : The return type containing parsing results.
    """

    # Patterns for extracting JSON
    JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", re.MULTILINE)
    JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}", re.MULTILINE)
    JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*\]", re.MULTILINE)

    @classmethod
    def parse(cls, text: str, strict: bool = False) -> ParseResult[Any]:
        """Parse JSON from LLM output with optional error recovery.

        This method attempts to extract and parse JSON from text, handling
        common formatting issues that occur in LLM output.

        Parameters
        ----------
        text : str
            The text to parse. May contain pure JSON, JSON in markdown
            code blocks, or JSON embedded in prose.
        strict : bool, default=False
            If True, only attempt direct JSON parsing without any
            error recovery or extraction. If False, try multiple
            extraction strategies and fix common issues.

        Returns
        -------
        ParseResult[Any]
            A ParseResult containing:
            - success=True and value=parsed_data on success
            - success=False and error=message on failure
            The format_detected will be OutputFormat.JSON on success.

        Examples
        --------
        Parsing clean JSON:

        >>> result = JSONParser.parse('{"key": "value"}')
        >>> result.success
        True
        >>> result.value
        {'key': 'value'}

        Parsing from code block:

        >>> text = '''Here is the data:
        ... ```json
        ... {"users": [{"name": "Alice"}]}
        ... ```
        ... '''
        >>> result = JSONParser.parse(text)
        >>> result.value['users'][0]['name']
        'Alice'

        Handling LLM formatting issues:

        >>> # Trailing comma (common LLM mistake)
        >>> result = JSONParser.parse('{"a": 1, "b": 2,}')
        >>> result.value
        {'a': 1, 'b': 2}

        >>> # Single quotes (common in Python-influenced output)
        >>> result = JSONParser.parse("{'key': 'value'}")
        >>> result.value
        {'key': 'value'}

        Strict mode for validation:

        >>> result = JSONParser.parse('{"trailing": "comma",}', strict=True)
        >>> result.success
        False

        JSON embedded in prose:

        >>> response = "The answer is: {'result': 42} as expected."
        >>> result = JSONParser.parse(response)
        >>> result.value
        {'result': 42}

        Accessing raw text on failure:

        >>> result = JSONParser.parse('not json')
        >>> result.raw
        'not json'

        See Also
        --------
        extract_all : Extract all JSON objects/arrays from text.
        ParseResult : Container for parsing results.
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
        """Extract JSON content from a markdown code block.

        Parameters
        ----------
        text : str
            Text that may contain a markdown code block.

        Returns
        -------
        Optional[str]
            The content inside the code block if found, None otherwise.

        Examples
        --------
        >>> text = '''```json
        ... {"key": "value"}
        ... ```'''
        >>> JSONParser._extract_from_code_block(text)
        '{"key": "value"}'

        >>> JSONParser._extract_from_code_block('no code block here')
        """
        match = cls.JSON_BLOCK_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    @classmethod
    def _extract_json_pattern(cls, text: str) -> Optional[str]:
        """Extract a JSON object or array pattern from text.

        Searches for content that looks like a JSON object ({...}) or
        array ([...]) and returns the first match.

        Parameters
        ----------
        text : str
            Text that may contain JSON patterns.

        Returns
        -------
        Optional[str]
            The matched JSON-like content if found, None otherwise.

        Examples
        --------
        >>> text = 'The result is {"answer": 42} as expected'
        >>> JSONParser._extract_json_pattern(text)
        '{"answer": 42}'

        >>> text = 'Items: [1, 2, 3]'
        >>> JSONParser._extract_json_pattern(text)
        '[1, 2, 3]'
        """
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
        """Fix common JSON formatting issues from LLM output.

        Applies corrections for common mistakes LLMs make when generating
        JSON, including trailing commas, single quotes, and comments.

        Parameters
        ----------
        json_text : str
            JSON text that may have formatting issues.

        Returns
        -------
        str
            Corrected JSON text.

        Examples
        --------
        Trailing comma removal:

        >>> JSONParser._fix_common_issues('{"a": 1,}')
        '{"a": 1}'

        Single quote conversion:

        >>> JSONParser._fix_common_issues("{'key': 'value'}")
        '{"key": "value"}'

        Comment removal:

        >>> text = '''{
        ...     "key": "value" // this is a comment
        ... }'''
        >>> 'comment' in JSONParser._fix_common_issues(text)
        False
        """
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
    def extract_all(cls, text: str) -> list[ParseResult[Any]]:
        """Extract all JSON objects and arrays from text.

        This method finds and parses all JSON structures in the text,
        including those in code blocks and those embedded in prose.
        It does not apply error correction to avoid false positives.

        Parameters
        ----------
        text : str
            The text containing one or more JSON structures.

        Returns
        -------
        list[ParseResult[Any]]
            List of successful ParseResult objects for each valid JSON
            found. Failed parses are not included in the results.

        Examples
        --------
        Multiple JSON objects:

        >>> text = '''
        ... First: {"id": 1, "name": "Alice"}
        ... Second: {"id": 2, "name": "Bob"}
        ... '''
        >>> results = JSONParser.extract_all(text)
        >>> len(results)
        2
        >>> [r.value['name'] for r in results]
        ['Alice', 'Bob']

        JSON in code blocks and prose:

        >>> text = '''
        ... Here's the config:
        ... ```json
        ... {"version": 1}
        ... ```
        ... And inline: {"inline": true}
        ... '''
        >>> results = JSONParser.extract_all(text)
        >>> len(results)
        2

        No JSON found:

        >>> results = JSONParser.extract_all('No JSON here')
        >>> len(results)
        0

        Array and object extraction:

        >>> text = '{"obj": true} and [1, 2, 3]'
        >>> results = JSONParser.extract_all(text)
        >>> [r.value for r in results]
        [{'obj': True}, [1, 2, 3]]

        See Also
        --------
        parse : Parse a single JSON from text with error recovery.
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
    """Parser for extracting code blocks from LLM output.

    This class provides methods to extract fenced code blocks (using triple
    backticks) from text, with support for language detection and filtering.

    The parser handles common language aliases (e.g., 'py' -> 'python',
    'js' -> 'javascript') and provides both line-number tracking and
    language-specific extraction methods.

    Attributes
    ----------
    FENCED_PATTERN : re.Pattern
        Regex pattern for fenced code blocks with optional language.
    INDENTED_PATTERN : re.Pattern
        Regex pattern for indented code blocks (4 spaces or 1 tab).
    LANGUAGE_ALIASES : dict[str, Optional[str]]
        Mapping of common language aliases to canonical names.

    Examples
    --------
    Basic code block extraction:

    >>> from insideLLMs.nlp.parsing import CodeBlockParser
    >>> text = '''
    ... Here's the code:
    ... ```python
    ... def hello():
    ...     print("Hello!")
    ... ```
    ... '''
    >>> blocks = CodeBlockParser.parse(text)
    >>> len(blocks)
    1
    >>> blocks[0].language
    'python'

    Getting the first code block:

    >>> text = '''
    ... ```javascript
    ... console.log("Hello");
    ... ```
    ... '''
    >>> block = CodeBlockParser.parse_first(text)
    >>> block.language
    'javascript'

    Filtering by language:

    >>> text = '''
    ... ```python
    ... print("Python")
    ... ```
    ... ```javascript
    ... console.log("JS");
    ... ```
    ... '''
    >>> python_blocks = CodeBlockParser.parse_by_language(text, 'python')
    >>> len(python_blocks)
    1
    >>> python_blocks[0].code
    'print("Python")'

    Using language aliases:

    >>> text = '''
    ... ```py
    ... x = 1
    ... ```
    ... '''
    >>> blocks = CodeBlockParser.parse(text)
    >>> blocks[0].language  # 'py' normalized to 'python'
    'python'

    Convenience extraction methods:

    >>> text = '''
    ... ```sql
    ... SELECT * FROM users;
    ... ```
    ... '''
    >>> sql_code = CodeBlockParser.extract_sql(text)
    >>> sql_code
    ['SELECT * FROM users;']

    Line number tracking:

    >>> text = '''Line 0
    ... Line 1
    ... ```python
    ... code here
    ... ```
    ... Line 5
    ... '''
    >>> blocks = CodeBlockParser.parse(text)
    >>> blocks[0].start_line
    2

    See Also
    --------
    parse_code : Convenience function wrapping CodeBlockParser.parse.
    CodeBlock : The data class returned by parsing methods.
    """

    # Pattern for fenced code blocks
    FENCED_PATTERN = re.compile(r"```(\w*)\s*\n([\s\S]*?)\n```", re.MULTILINE)

    # Pattern for indented code blocks (4 spaces or 1 tab)
    INDENTED_PATTERN = re.compile(r"(?:^(?:    |\t).*$\n?)+", re.MULTILINE)

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
    def parse(cls, text: str) -> list[CodeBlock]:
        """Extract all fenced code blocks from text.

        Finds all code blocks delimited by triple backticks and returns
        them as CodeBlock objects with language detection and line
        number tracking.

        Parameters
        ----------
        text : str
            The text to parse for code blocks.

        Returns
        -------
        list[CodeBlock]
            List of CodeBlock objects in order of appearance.
            Empty list if no code blocks are found.

        Examples
        --------
        Single code block:

        >>> text = '''
        ... ```python
        ... print("Hello")
        ... ```
        ... '''
        >>> blocks = CodeBlockParser.parse(text)
        >>> len(blocks)
        1
        >>> blocks[0].code
        'print("Hello")'

        Multiple code blocks:

        >>> text = '''
        ... ```python
        ... x = 1
        ... ```
        ... Some text
        ... ```javascript
        ... let y = 2;
        ... ```
        ... '''
        >>> blocks = CodeBlockParser.parse(text)
        >>> len(blocks)
        2
        >>> blocks[0].language
        'python'
        >>> blocks[1].language
        'javascript'

        Code block without language:

        >>> text = '''
        ... ```
        ... some code
        ... ```
        ... '''
        >>> blocks = CodeBlockParser.parse(text)
        >>> blocks[0].language is None
        True

        Multi-line code:

        >>> text = '''
        ... ```python
        ... def greet(name):
        ...     return f"Hello, {name}!"
        ...
        ... print(greet("World"))
        ... ```
        ... '''
        >>> blocks = CodeBlockParser.parse(text)
        >>> len(blocks[0].code.split('\\n'))
        4

        See Also
        --------
        parse_first : Get only the first code block.
        parse_by_language : Filter code blocks by language.
        """
        blocks = []

        # Find fenced code blocks
        for match in cls.FENCED_PATTERN.finditer(text):
            language = match.group(1).lower()
            language = cls.LANGUAGE_ALIASES.get(language, language)
            code = match.group(2)

            # Calculate line numbers
            start = text[: match.start()].count("\n")
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
        """Extract the first code block from text.

        A convenience method when you only need the first code block
        or expect only one code block in the text.

        Parameters
        ----------
        text : str
            The text to parse.

        Returns
        -------
        Optional[CodeBlock]
            The first CodeBlock found, or None if no code blocks exist.

        Examples
        --------
        Getting the first block:

        >>> text = '''
        ... ```python
        ... first_block()
        ... ```
        ... ```javascript
        ... second_block();
        ... ```
        ... '''
        >>> block = CodeBlockParser.parse_first(text)
        >>> block.language
        'python'
        >>> block.code
        'first_block()'

        No code blocks found:

        >>> block = CodeBlockParser.parse_first('No code here')
        >>> block is None
        True

        Single code block:

        >>> text = '''
        ... ```sql
        ... SELECT 1;
        ... ```
        ... '''
        >>> block = CodeBlockParser.parse_first(text)
        >>> block.code
        'SELECT 1;'

        See Also
        --------
        parse : Get all code blocks.
        """
        blocks = cls.parse(text)
        return blocks[0] if blocks else None

    @classmethod
    def parse_by_language(cls, text: str, language: str) -> list[CodeBlock]:
        """Extract code blocks of a specific programming language.

        Filters code blocks to return only those matching the specified
        language. Language aliases are automatically normalized.

        Parameters
        ----------
        text : str
            The text to parse.
        language : str
            The language to filter by (e.g., 'python', 'javascript').
            Aliases like 'py' or 'js' are accepted.

        Returns
        -------
        list[CodeBlock]
            List of CodeBlock objects matching the specified language.
            Empty list if no matching blocks are found.

        Examples
        --------
        Filtering Python blocks:

        >>> text = '''
        ... ```python
        ... x = 1
        ... ```
        ... ```javascript
        ... let y = 2;
        ... ```
        ... ```python
        ... z = 3
        ... ```
        ... '''
        >>> python_blocks = CodeBlockParser.parse_by_language(text, 'python')
        >>> len(python_blocks)
        2
        >>> [b.code for b in python_blocks]
        ['x = 1', 'z = 3']

        Using language alias:

        >>> text = '''
        ... ```py
        ... print("hi")
        ... ```
        ... '''
        >>> blocks = CodeBlockParser.parse_by_language(text, 'py')
        >>> len(blocks)
        1

        No matching language:

        >>> text = '''
        ... ```rust
        ... fn main() {}
        ... ```
        ... '''
        >>> blocks = CodeBlockParser.parse_by_language(text, 'python')
        >>> len(blocks)
        0

        See Also
        --------
        parse : Get all code blocks regardless of language.
        extract_python : Convenience method for Python code.
        extract_sql : Convenience method for SQL code.
        """
        language = language.lower()
        language = cls.LANGUAGE_ALIASES.get(language, language)

        blocks = cls.parse(text)
        return [b for b in blocks if b.language == language]

    @classmethod
    def extract_python(cls, text: str) -> list[str]:
        """Extract Python code from all Python code blocks.

        A convenience method that returns just the code strings (not
        CodeBlock objects) for Python code blocks.

        Parameters
        ----------
        text : str
            The text to parse.

        Returns
        -------
        list[str]
            List of Python code strings.

        Examples
        --------
        >>> text = '''
        ... ```python
        ... def hello():
        ...     print("Hello!")
        ... ```
        ... '''
        >>> python_code = CodeBlockParser.extract_python(text)
        >>> len(python_code)
        1
        >>> 'def hello' in python_code[0]
        True

        Multiple Python blocks:

        >>> text = '''
        ... ```python
        ... x = 1
        ... ```
        ... ```python
        ... y = 2
        ... ```
        ... '''
        >>> code = CodeBlockParser.extract_python(text)
        >>> code
        ['x = 1', 'y = 2']

        See Also
        --------
        parse_by_language : More flexible language filtering.
        extract_sql : Similar method for SQL.
        """
        return [b.code for b in cls.parse_by_language(text, "python")]

    @classmethod
    def extract_sql(cls, text: str) -> list[str]:
        """Extract SQL code from all SQL code blocks.

        A convenience method that returns just the code strings (not
        CodeBlock objects) for SQL code blocks.

        Parameters
        ----------
        text : str
            The text to parse.

        Returns
        -------
        list[str]
            List of SQL code strings.

        Examples
        --------
        >>> text = '''
        ... Here's the query:
        ... ```sql
        ... SELECT * FROM users WHERE active = true;
        ... ```
        ... '''
        >>> sql_code = CodeBlockParser.extract_sql(text)
        >>> sql_code
        ['SELECT * FROM users WHERE active = true;']

        Multiple SQL blocks:

        >>> text = '''
        ... ```sql
        ... CREATE TABLE users (id INT);
        ... ```
        ... ```sql
        ... INSERT INTO users VALUES (1);
        ... ```
        ... '''
        >>> queries = CodeBlockParser.extract_sql(text)
        >>> len(queries)
        2

        See Also
        --------
        parse_by_language : More flexible language filtering.
        extract_python : Similar method for Python.
        """
        return [b.code for b in cls.parse_by_language(text, "sql")]


class ListParser:
    """Parser for extracting lists from LLM output.

    This class handles various list formats commonly produced by LLMs:
    - Numbered lists (1. item, 2) item, 3] item)
    - Bulleted lists (- item, * item, + item)
    - Lettered lists (a. item, b) item, c] item)

    The parser also supports nested list detection based on indentation.

    Attributes
    ----------
    NUMBERED_PATTERN : re.Pattern
        Regex pattern for numbered list items (1. 2) 3] etc.).
    BULLET_PATTERN : re.Pattern
        Regex pattern for bullet list items (-, *, +).
    LETTERED_PATTERN : re.Pattern
        Regex pattern for lettered list items (a. b) c] etc.).

    Examples
    --------
    Parsing a numbered list:

    >>> from insideLLMs.nlp.parsing import ListParser
    >>> text = '''
    ... 1. First item
    ... 2. Second item
    ... 3. Third item
    ... '''
    >>> items = ListParser.parse(text)
    >>> items
    ['First item', 'Second item', 'Third item']

    Parsing a bulleted list:

    >>> text = '''
    ... - Apple
    ... - Banana
    ... - Cherry
    ... '''
    >>> items = ListParser.parse(text)
    >>> items
    ['Apple', 'Banana', 'Cherry']

    Parsing a lettered list:

    >>> text = '''
    ... a. Option A
    ... b. Option B
    ... c. Option C
    ... '''
    >>> items = ListParser.parse(text)
    >>> items
    ['Option A', 'Option B', 'Option C']

    Different bullet styles:

    >>> text = '''
    ... * Item with asterisk
    ... + Item with plus
    ... - Item with dash
    ... '''
    >>> items = ListParser.parse(text)
    >>> len(items)
    3

    Nested list parsing:

    >>> text = '''
    ... - Parent item
    ...   - Child item 1
    ...   - Child item 2
    ... - Another parent
    ... '''
    >>> nested = ListParser.parse_nested(text)
    >>> len(nested) >= 2
    True

    See Also
    --------
    parse_list : Convenience function wrapping ListParser.parse.
    """

    # Patterns for different list formats
    NUMBERED_PATTERN = re.compile(r"^\s*(\d+)[.)\]]\s*(.+)$", re.MULTILINE)
    BULLET_PATTERN = re.compile(r"^\s*[-*+]\s*(.+)$", re.MULTILINE)
    LETTERED_PATTERN = re.compile(r"^\s*([a-zA-Z])[.)\]]\s*(.+)$", re.MULTILINE)

    @classmethod
    def parse(cls, text: str) -> list[str]:
        """Extract list items from text.

        Detects and parses numbered, bulleted, or lettered lists.
        Returns items in order of appearance. The first matching
        list format takes precedence.

        Parameters
        ----------
        text : str
            The text containing a list.

        Returns
        -------
        list[str]
            List of item strings with leading/trailing whitespace trimmed.
            Empty list if no list is found.

        Examples
        --------
        Numbered list:

        >>> text = '''
        ... Top 3 languages:
        ... 1. Python
        ... 2. JavaScript
        ... 3. Go
        ... '''
        >>> ListParser.parse(text)
        ['Python', 'JavaScript', 'Go']

        Bullet list with dashes:

        >>> text = '''
        ... Shopping list:
        ... - Milk
        ... - Eggs
        ... - Bread
        ... '''
        >>> ListParser.parse(text)
        ['Milk', 'Eggs', 'Bread']

        Bullet list with asterisks:

        >>> text = '''
        ... * Red
        ... * Green
        ... * Blue
        ... '''
        >>> ListParser.parse(text)
        ['Red', 'Green', 'Blue']

        Lettered list:

        >>> text = '''
        ... Answer choices:
        ... a) True
        ... b) False
        ... c) Unknown
        ... '''
        >>> ListParser.parse(text)
        ['True', 'False', 'Unknown']

        Mixed content (first format wins):

        >>> text = '''
        ... 1. First numbered
        ... 2. Second numbered
        ... - Bullet item (ignored)
        ... '''
        >>> items = ListParser.parse(text)
        >>> 'First numbered' in items
        True

        No list found:

        >>> ListParser.parse('No list here, just text.')
        []

        See Also
        --------
        parse_nested : Parse lists with indentation hierarchy.
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
    def parse_nested(cls, text: str) -> list[Union[str, list]]:
        """Parse nested lists based on indentation.

        Analyzes indentation levels to construct a hierarchical list
        structure. Items at the same indentation level are siblings,
        while items with greater indentation become children.

        Parameters
        ----------
        text : str
            The text containing a potentially nested list.

        Returns
        -------
        list[Union[str, list]]
            A nested list structure where each element is either a
            string (leaf item) or a list (nested items).

        Examples
        --------
        Simple flat list (no nesting):

        >>> text = '''
        ... - Item 1
        ... - Item 2
        ... - Item 3
        ... '''
        >>> result = ListParser.parse_nested(text)
        >>> len(result)
        3

        Nested list:

        >>> text = '''
        ... - Parent 1
        ...   - Child 1.1
        ...   - Child 1.2
        ... - Parent 2
        ... '''
        >>> result = ListParser.parse_nested(text)
        >>> len(result) >= 2
        True

        Numbered nested list:

        >>> text = '''
        ... 1. First
        ...    1. Sub-first
        ... 2. Second
        ... '''
        >>> result = ListParser.parse_nested(text)
        >>> len(result) >= 2
        True

        Note
        ----
        The nesting algorithm uses simple indentation detection and may
        not perfectly handle all edge cases. For complex nested structures,
        consider implementing custom parsing logic.

        See Also
        --------
        parse : Flat list parsing without nesting.
        """
        lines = text.split("\n")
        result: list[Union[str, list]] = []
        stack: list[tuple[int, list]] = [(0, result)]

        for line in lines:
            if not line.strip():
                continue

            # Calculate indentation
            indent = len(line) - len(line.lstrip())

            # Extract item text
            match = cls.BULLET_PATTERN.match(line) or cls.NUMBERED_PATTERN.match(line)
            if match:
                item_text = (
                    match.group(1)
                    if isinstance(match.group(1), str) and not match.group(1).isdigit()
                    else match.groups()[-1]
                )
                item_text = item_text.strip()

                # Find appropriate level
                while stack and stack[-1][0] >= indent and len(stack) > 1:
                    stack.pop()

                current_list = stack[-1][1]
                current_list.append(item_text)

        return result


class TableParser:
    """Parser for markdown tables in LLM output.

    This class extracts and parses markdown-style tables, converting them
    into structured TableData objects with headers and rows.

    Markdown tables have the format:
    ```
    | Header 1 | Header 2 |
    |----------|----------|
    | Cell 1   | Cell 2   |
    ```

    Attributes
    ----------
    MD_TABLE_PATTERN : re.Pattern
        Regex pattern for matching markdown tables.

    Examples
    --------
    Parsing a basic table:

    >>> from insideLLMs.nlp.parsing import TableParser
    >>> text = '''
    ... | Name    | Age |
    ... |---------|-----|
    ... | Alice   | 30  |
    ... | Bob     | 25  |
    ... '''
    >>> table = TableParser.parse(text)
    >>> table.headers
    ['Name', 'Age']
    >>> table.num_rows
    2

    Parsing multiple tables:

    >>> text = '''
    ... First table:
    ... | A | B |
    ... |---|---|
    ... | 1 | 2 |
    ...
    ... Second table:
    ... | X | Y |
    ... |---|---|
    ... | 3 | 4 |
    ... '''
    >>> tables = TableParser.parse_all(text)
    >>> len(tables)
    2

    Accessing table data:

    >>> text = '''
    ... | Product | Price | Stock |
    ... |---------|-------|-------|
    ... | Widget  | $10   | 100   |
    ... | Gadget  | $25   | 50    |
    ... '''
    >>> table = TableParser.parse(text)
    >>> table.get_column('Price')
    ['$10', '$25']
    >>> table.to_dicts()[0]
    {'Product': 'Widget', 'Price': '$10', 'Stock': '100'}

    No table found:

    >>> result = TableParser.parse('No table here')
    >>> result is None
    True

    See Also
    --------
    parse_table : Convenience function wrapping TableParser.parse.
    TableData : The data class returned by parsing methods.
    """

    # Pattern for markdown table
    MD_TABLE_PATTERN = re.compile(
        r"^\|(.+)\|\s*\n\|[-:\s|]+\|\s*\n((?:\|.+\|\s*\n?)+)",
        re.MULTILINE,
    )

    @classmethod
    def parse(cls, text: str) -> Optional[TableData]:
        """Parse the first markdown table from text.

        Extracts and parses the first markdown table found in the text.
        Tables must have a header row, separator row (with dashes), and
        at least one data row.

        Parameters
        ----------
        text : str
            The text containing a markdown table.

        Returns
        -------
        Optional[TableData]
            A TableData object if a table is found, None otherwise.

        Examples
        --------
        Basic table parsing:

        >>> text = '''
        ... | Language   | Year | Creator       |
        ... |------------|------|---------------|
        ... | Python     | 1991 | Guido         |
        ... | JavaScript | 1995 | Brendan Eich  |
        ... | Go         | 2009 | Google        |
        ... '''
        >>> table = TableParser.parse(text)
        >>> table.headers
        ['Language', 'Year', 'Creator']
        >>> table.rows[0]
        ['Python', '1991', 'Guido']
        >>> table.num_rows
        3

        Table with aligned columns:

        >>> text = '''
        ... |  Left  | Center | Right |
        ... |:-------|:------:|------:|
        ... | data   | data   | data  |
        ... '''
        >>> table = TableParser.parse(text)
        >>> table is not None
        True

        Converting to dictionaries:

        >>> text = '''
        ... | id | name  |
        ... |----|-------|
        ... | 1  | Alice |
        ... | 2  | Bob   |
        ... '''
        >>> table = TableParser.parse(text)
        >>> records = table.to_dicts()
        >>> records[0]
        {'id': '1', 'name': 'Alice'}

        No table returns None:

        >>> result = TableParser.parse('Just some text')
        >>> result is None
        True

        Accessing raw table text:

        >>> text = '''
        ... | A | B |
        ... |---|---|
        ... | 1 | 2 |
        ... '''
        >>> table = TableParser.parse(text)
        >>> '|---|---|' in table.raw
        True

        See Also
        --------
        parse_all : Parse all tables from text.
        TableData : The returned data structure.
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
    def parse_all(cls, text: str) -> list[TableData]:
        """Parse all markdown tables from text.

        Finds and parses all markdown tables in the text, returning
        them in order of appearance.

        Parameters
        ----------
        text : str
            The text containing one or more markdown tables.

        Returns
        -------
        list[TableData]
            List of TableData objects, one for each table found.
            Empty list if no tables are found.

        Examples
        --------
        Multiple tables:

        >>> text = '''
        ... Users table:
        ... | id | name  |
        ... |----|-------|
        ... | 1  | Alice |
        ...
        ... Products table:
        ... | id | product |
        ... |----|---------|
        ... | 1  | Widget  |
        ... '''
        >>> tables = TableParser.parse_all(text)
        >>> len(tables)
        2
        >>> tables[0].headers
        ['id', 'name']
        >>> tables[1].headers
        ['id', 'product']

        Processing all tables:

        >>> text = '''
        ... | A |
        ... |---|
        ... | 1 |
        ...
        ... | B |
        ... |---|
        ... | 2 |
        ... '''
        >>> tables = TableParser.parse_all(text)
        >>> for table in tables:
        ...     print(table.headers[0])
        A
        B

        No tables returns empty list:

        >>> tables = TableParser.parse_all('No tables here')
        >>> tables
        []

        See Also
        --------
        parse : Parse only the first table.
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

            tables.append(TableData(headers=headers, rows=rows, raw=match.group(0)))

        return tables


class AnswerExtractor:
    """Extract specific answer formats from LLM responses.

    This class provides methods to extract common answer formats from
    LLM outputs, including:

    - Final answers (e.g., "The answer is: 42")
    - Yes/No responses
    - Multiple choice answers (A-E)
    - Numeric results

    The extractor uses multiple regex patterns for each format to handle
    variations in how LLMs express answers.

    Attributes
    ----------
    FINAL_ANSWER_PATTERNS : list[re.Pattern]
        Patterns for extracting final answer statements.
    YES_NO_PATTERNS : list[re.Pattern]
        Patterns for extracting yes/no responses.
    CHOICE_PATTERNS : list[re.Pattern]
        Patterns for extracting multiple choice answers.
    NUMBER_PATTERNS : list[re.Pattern]
        Patterns for extracting numeric answers.

    Examples
    --------
    Extracting final answers:

    >>> from insideLLMs.nlp.parsing import AnswerExtractor
    >>> text = "After analysis, the answer is: 42"
    >>> AnswerExtractor.extract_final_answer(text)
    '42'

    Extracting yes/no:

    >>> text = "Based on the evidence, yes, that is correct."
    >>> AnswerExtractor.extract_yes_no(text)
    True
    >>> text = "No, that is not possible."
    >>> AnswerExtractor.extract_yes_no(text)
    False

    Extracting multiple choice:

    >>> text = "The answer is B because..."
    >>> AnswerExtractor.extract_choice(text)
    'B'

    Extracting numbers:

    >>> text = "The result = 3.14159"
    >>> AnswerExtractor.extract_number(text)
    3.14159

    Getting all numbers:

    >>> text = "The values are 1, 2, and 3.5"
    >>> AnswerExtractor.extract_all_numbers(text)
    [1.0, 2.0, 3.5]

    See Also
    --------
    extract_answer : Convenience function for final answer extraction.
    """

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

        Searches for common answer patterns like "The answer is: X",
        "Therefore, X", or LaTeX \\boxed{X} notation.

        Parameters
        ----------
        text : str
            The response text to search.

        Returns
        -------
        Optional[str]
            The extracted answer string, or None if no answer found.

        Examples
        --------
        Standard answer format:

        >>> text = "The answer is: Paris"
        >>> AnswerExtractor.extract_final_answer(text)
        'Paris'

        Final answer format:

        >>> text = "Final answer: 42"
        >>> AnswerExtractor.extract_final_answer(text)
        '42'

        Therefore format:

        >>> text = "The sum is 2+2. Therefore, 4."
        >>> AnswerExtractor.extract_final_answer(text)
        '4.'

        LaTeX boxed format:

        >>> text = "The solution is \\\\boxed{x=5}"
        >>> AnswerExtractor.extract_final_answer(text)
        'x=5'

        No answer found:

        >>> text = "I'm not sure about the answer."
        >>> result = AnswerExtractor.extract_final_answer(text)
        >>> result is None
        True

        Answer with equals sign:

        >>> text = "Answer = 100"
        >>> AnswerExtractor.extract_final_answer(text)
        '100'

        See Also
        --------
        extract_number : For numeric answers.
        extract_choice : For multiple choice answers.
        """
        for pattern in cls.FINAL_ANSWER_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return None

    @classmethod
    def extract_yes_no(cls, text: str) -> Optional[bool]:
        """Extract a yes/no answer from a response.

        Searches for explicit yes or no responses at the beginning of
        lines or in common answer formats.

        Parameters
        ----------
        text : str
            The response text to search.

        Returns
        -------
        Optional[bool]
            True for "yes", False for "no", None if not found.

        Examples
        --------
        Yes at start of line:

        >>> text = "Yes, that is correct."
        >>> AnswerExtractor.extract_yes_no(text)
        True

        No at start of line:

        >>> text = "No, that is incorrect."
        >>> AnswerExtractor.extract_yes_no(text)
        False

        Answer format with yes:

        >>> text = "The answer: yes"
        >>> AnswerExtractor.extract_yes_no(text)
        True

        Case insensitive:

        >>> text = "YES"
        >>> AnswerExtractor.extract_yes_no(text)
        True

        No clear yes/no:

        >>> text = "Maybe, it depends."
        >>> result = AnswerExtractor.extract_yes_no(text)
        >>> result is None
        True

        Yes in middle of sentence (not matched):

        >>> text = "I would say yes to that proposal"
        >>> # This would match because "yes" appears, but check patterns
        >>> result = AnswerExtractor.extract_yes_no("Well, I'm not sure")
        >>> result is None
        True

        See Also
        --------
        extract_choice : For multiple choice questions.
        """
        for pattern in cls.YES_NO_PATTERNS:
            match = pattern.search(text)
            if match:
                answer = match.group(1).lower()
                return answer == "yes"
        return None

    @classmethod
    def extract_choice(cls, text: str) -> Optional[str]:
        """Extract a multiple choice answer (A-E) from a response.

        Searches for common patterns indicating a choice selection,
        such as "The answer is B" or "Choice: C".

        Parameters
        ----------
        text : str
            The response text to search.

        Returns
        -------
        Optional[str]
            The choice letter (uppercase A-E), or None if not found.

        Examples
        --------
        Standard format:

        >>> text = "The answer is B"
        >>> AnswerExtractor.extract_choice(text)
        'B'

        Answer with colon:

        >>> text = "Answer: C"
        >>> AnswerExtractor.extract_choice(text)
        'C'

        Choice format:

        >>> text = "Choice = D"
        >>> AnswerExtractor.extract_choice(text)
        'D'

        Correct statement:

        >>> text = "A is correct"
        >>> AnswerExtractor.extract_choice(text)
        'A'

        Case insensitive:

        >>> text = "the answer is e"
        >>> AnswerExtractor.extract_choice(text)
        'E'

        No choice found:

        >>> text = "The best approach is to use Python"
        >>> result = AnswerExtractor.extract_choice(text)
        >>> result is None
        True

        See Also
        --------
        extract_yes_no : For yes/no questions.
        extract_final_answer : For general answer extraction.
        """
        for pattern in cls.CHOICE_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).upper()
        return None

    @classmethod
    def extract_number(cls, text: str) -> Optional[float]:
        """Extract a numeric answer from a response.

        Searches for common patterns indicating a numeric result,
        such as "Answer: 42" or "= 3.14".

        Parameters
        ----------
        text : str
            The response text to search.

        Returns
        -------
        Optional[float]
            The extracted number, or None if not found.

        Examples
        --------
        Integer answer:

        >>> text = "The answer is 42"
        >>> AnswerExtractor.extract_number(text)
        42.0

        Decimal answer:

        >>> text = "Result: 3.14159"
        >>> AnswerExtractor.extract_number(text)
        3.14159

        Equation result:

        >>> text = "2 + 2 = 4"
        >>> AnswerExtractor.extract_number(text)
        4.0

        Negative number:

        >>> text = "The answer: -10"
        >>> AnswerExtractor.extract_number(text)
        -10.0

        No number in answer format:

        >>> text = "The answer involves complex calculations"
        >>> result = AnswerExtractor.extract_number(text)
        >>> result is None
        True

        Note: Standalone numbers not in answer format aren't matched:

        >>> text = "There are 5 items"
        >>> result = AnswerExtractor.extract_number(text)
        >>> result is None
        True

        See Also
        --------
        extract_all_numbers : Get all numbers from text.
        extract_final_answer : For non-numeric answers.
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
    def extract_all_numbers(cls, text: str) -> list[float]:
        """Extract all numbers from text.

        Finds all integer and decimal numbers in the text, regardless
        of context. Useful for extracting multiple numeric values.

        Parameters
        ----------
        text : str
            The response text to search.

        Returns
        -------
        list[float]
            List of all numbers found (as floats).

        Examples
        --------
        Multiple integers:

        >>> text = "The values are 1, 2, 3, and 4"
        >>> AnswerExtractor.extract_all_numbers(text)
        [1.0, 2.0, 3.0, 4.0]

        Mixed integers and decimals:

        >>> text = "Price: $10.99, Quantity: 5"
        >>> AnswerExtractor.extract_all_numbers(text)
        [10.99, 5.0]

        Negative numbers:

        >>> text = "Range: -5 to 5"
        >>> AnswerExtractor.extract_all_numbers(text)
        [-5.0, 5.0]

        Decimal numbers:

        >>> text = "Pi is 3.14159, e is 2.71828"
        >>> numbers = AnswerExtractor.extract_all_numbers(text)
        >>> len(numbers)
        2
        >>> numbers[0]
        3.14159

        No numbers:

        >>> text = "No numbers here"
        >>> AnswerExtractor.extract_all_numbers(text)
        []

        Numbers in various contexts:

        >>> text = "Step 1: Add 100. Step 2: Subtract 50."
        >>> AnswerExtractor.extract_all_numbers(text)
        [1.0, 100.0, 2.0, 50.0]

        See Also
        --------
        extract_number : Extract a single answer number.
        """
        pattern = re.compile(r"-?\d+(?:\.\d+)?")
        matches = pattern.findall(text)
        return [float(m) for m in matches]


class OutputDetector:
    """Detect the format of LLM output.

    This class uses heuristics to determine the primary format of text
    output from an LLM. It can identify JSON, code blocks, tables, lists,
    markdown, XML, YAML, and plain text.

    The detection is performed in priority order, so JSON is checked
    before code blocks, which is checked before tables, etc.

    Examples
    --------
    Detecting JSON:

    >>> from insideLLMs.nlp.parsing import OutputDetector, OutputFormat
    >>> text = '{"name": "Alice", "age": 30}'
    >>> OutputDetector.detect(text)
    <OutputFormat.JSON: 'json'>

    Detecting code blocks:

    >>> text = '''
    ... ```python
    ... print("Hello")
    ... ```
    ... '''
    >>> OutputDetector.detect(text)
    <OutputFormat.CODE: 'code'>

    Detecting tables:

    >>> text = '''
    ... | A | B |
    ... |---|---|
    ... | 1 | 2 |
    ... '''
    >>> OutputDetector.detect(text)
    <OutputFormat.TABLE: 'table'>

    Detecting lists:

    >>> text = '''
    ... - Item 1
    ... - Item 2
    ... - Item 3
    ... '''
    >>> OutputDetector.detect(text)
    <OutputFormat.LIST: 'list'>

    Detecting markdown:

    >>> text = '''
    ... # Header
    ... This is **bold** and [a link](http://example.com)
    ... '''
    >>> OutputDetector.detect(text)
    <OutputFormat.MARKDOWN: 'markdown'>

    Plain text fallback:

    >>> text = "Just some regular text without any special formatting."
    >>> OutputDetector.detect(text)
    <OutputFormat.PLAIN: 'plain'>

    See Also
    --------
    detect_format : Convenience function for format detection.
    OutputFormat : Enum of possible detected formats.
    """

    @classmethod
    def detect(cls, text: str) -> OutputFormat:
        """Detect the primary format of the text.

        Analyzes the text and returns the most likely format. Detection
        is performed in priority order (JSON, code, table, XML, YAML,
        list, markdown, plain).

        Parameters
        ----------
        text : str
            The text to analyze.

        Returns
        -------
        OutputFormat
            The detected format enum value.

        Examples
        --------
        JSON detection:

        >>> text = '{"key": "value"}'
        >>> OutputDetector.detect(text)
        <OutputFormat.JSON: 'json'>

        >>> text = '[1, 2, 3]'
        >>> OutputDetector.detect(text)
        <OutputFormat.JSON: 'json'>

        Code detection:

        >>> text = '```javascript\\nconsole.log("hi");\\n```'
        >>> OutputDetector.detect(text) == OutputFormat.CODE
        True

        Table detection:

        >>> text = '| Col |\\n|-----|\\n| Val |'
        >>> OutputDetector.detect(text) == OutputFormat.TABLE
        True

        XML detection:

        >>> text = '<root><child>value</child></root>'
        >>> OutputDetector.detect(text)
        <OutputFormat.XML: 'xml'>

        YAML detection:

        >>> text = 'name: John\\nage: 30\\ncity: NYC'
        >>> OutputDetector.detect(text)
        <OutputFormat.YAML: 'yaml'>

        List detection:

        >>> text = '1. First\\n2. Second\\n3. Third'
        >>> OutputDetector.detect(text)
        <OutputFormat.LIST: 'list'>

        Markdown detection:

        >>> text = '# Title\\nSome **bold** text'
        >>> OutputDetector.detect(text)
        <OutputFormat.MARKDOWN: 'markdown'>

        Plain text:

        >>> text = 'Just some regular text'
        >>> OutputDetector.detect(text)
        <OutputFormat.PLAIN: 'plain'>

        See Also
        --------
        OutputFormat : The enum of possible formats.
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
        """Check if text looks like valid JSON.

        Parameters
        ----------
        text : str
            Text to check.

        Returns
        -------
        bool
            True if the text parses as valid JSON.

        Examples
        --------
        >>> OutputDetector._looks_like_json('{"key": "value"}')
        True
        >>> OutputDetector._looks_like_json('[1, 2, 3]')
        True
        >>> OutputDetector._looks_like_json('not json')
        False
        """
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
        """Check if text looks like XML.

        Parameters
        ----------
        text : str
            Text to check.

        Returns
        -------
        bool
            True if the text appears to be XML.

        Examples
        --------
        >>> OutputDetector._looks_like_xml('<root><child/></root>')
        True
        >>> OutputDetector._looks_like_xml('<html><body>Hi</body></html>')
        True
        >>> OutputDetector._looks_like_xml('not xml')
        False
        """
        text = text.strip()
        return (
            text.startswith("<")
            and text.endswith(">")
            and re.search(r"<\w+[^>]*>", text) is not None
        )

    @classmethod
    def _looks_like_yaml(cls, text: str) -> bool:
        """Check if text looks like YAML.

        Parameters
        ----------
        text : str
            Text to check.

        Returns
        -------
        bool
            True if the text appears to be YAML.

        Examples
        --------
        >>> OutputDetector._looks_like_yaml('name: John\\nage: 30')
        True
        >>> OutputDetector._looks_like_yaml('key: value\\nanother: item')
        True
        >>> OutputDetector._looks_like_yaml('Just text')
        False
        """
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
        """Check if text looks like a list.

        Parameters
        ----------
        text : str
            Text to check.

        Returns
        -------
        bool
            True if the text appears to be a list.

        Examples
        --------
        >>> OutputDetector._looks_like_list('- Item 1\\n- Item 2\\n- Item 3')
        True
        >>> OutputDetector._looks_like_list('1. First\\n2. Second')
        True
        >>> OutputDetector._looks_like_list('Just a paragraph.')
        False
        """
        lines = text.strip().split("\n")
        list_lines = 0
        for line in lines:
            if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+[.)\]]\s+", line):
                list_lines += 1
        return list_lines >= 2 and list_lines / len(lines) > 0.5

    @classmethod
    def _looks_like_markdown(cls, text: str) -> bool:
        """Check if text has markdown features.

        Parameters
        ----------
        text : str
            Text to check.

        Returns
        -------
        bool
            True if the text contains markdown formatting.

        Examples
        --------
        >>> OutputDetector._looks_like_markdown('# Header')
        True
        >>> OutputDetector._looks_like_markdown('[link](url)')
        True
        >>> OutputDetector._looks_like_markdown('**bold**')
        True
        >>> OutputDetector._looks_like_markdown('Plain text')
        False
        """
        md_patterns = [
            r"^#{1,6}\s+",  # Headers
            r"\[.+\]\(.+\)",  # Links
            r"\*\*.+\*\*",  # Bold
            r"`.+`",  # Inline code
        ]
        return any(re.search(pattern, text, re.MULTILINE) for pattern in md_patterns)


def parse_json(text: str, strict: bool = False) -> ParseResult[Any]:
    """Parse JSON from LLM output.

    This is a convenience function that wraps JSONParser.parse() for
    simpler access to JSON parsing functionality.

    Parameters
    ----------
    text : str
        The text to parse for JSON content.
    strict : bool, default=False
        If True, only attempt direct JSON parsing without error recovery.

    Returns
    -------
    ParseResult[Any]
        A ParseResult containing the parsed JSON or error information.

    Examples
    --------
    Basic parsing:

    >>> result = parse_json('{"key": "value"}')
    >>> result.success
    True
    >>> result.value
    {'key': 'value'}

    Parsing from markdown:

    >>> text = '''
    ... ```json
    ... {"name": "Alice"}
    ... ```
    ... '''
    >>> result = parse_json(text)
    >>> result.value['name']
    'Alice'

    Handling errors:

    >>> result = parse_json('not json')
    >>> result.success
    False
    >>> result.error
    'No JSON found in text'

    See Also
    --------
    JSONParser : The underlying parser class.
    ParseResult : The return type.
    """
    return JSONParser.parse(text, strict=strict)


def parse_code(text: str) -> list[CodeBlock]:
    """Extract code blocks from text.

    This is a convenience function that wraps CodeBlockParser.parse() for
    simpler access to code block extraction.

    Parameters
    ----------
    text : str
        The text to parse for code blocks.

    Returns
    -------
    list[CodeBlock]
        List of extracted CodeBlock objects.

    Examples
    --------
    >>> text = '''
    ... ```python
    ... print("Hello")
    ... ```
    ... '''
    >>> blocks = parse_code(text)
    >>> blocks[0].language
    'python'

    See Also
    --------
    CodeBlockParser : The underlying parser class.
    CodeBlock : The returned data type.
    """
    return CodeBlockParser.parse(text)


def parse_list(text: str) -> list[str]:
    """Extract list items from text.

    This is a convenience function that wraps ListParser.parse() for
    simpler access to list parsing.

    Parameters
    ----------
    text : str
        The text to parse for list items.

    Returns
    -------
    list[str]
        List of extracted item strings.

    Examples
    --------
    >>> text = '''
    ... 1. First
    ... 2. Second
    ... 3. Third
    ... '''
    >>> parse_list(text)
    ['First', 'Second', 'Third']

    See Also
    --------
    ListParser : The underlying parser class.
    """
    return ListParser.parse(text)


def parse_table(text: str) -> Optional[TableData]:
    """Parse a table from text.

    This is a convenience function that wraps TableParser.parse() for
    simpler access to table parsing.

    Parameters
    ----------
    text : str
        The text containing a markdown table.

    Returns
    -------
    Optional[TableData]
        Parsed TableData or None if no table found.

    Examples
    --------
    >>> text = '''
    ... | A | B |
    ... |---|---|
    ... | 1 | 2 |
    ... '''
    >>> table = parse_table(text)
    >>> table.headers
    ['A', 'B']

    See Also
    --------
    TableParser : The underlying parser class.
    TableData : The returned data type.
    """
    return TableParser.parse(text)


def detect_format(text: str) -> OutputFormat:
    """Detect the format of text.

    This is a convenience function that wraps OutputDetector.detect() for
    simpler access to format detection.

    Parameters
    ----------
    text : str
        The text to analyze.

    Returns
    -------
    OutputFormat
        The detected format.

    Examples
    --------
    >>> detect_format('{"key": "value"}')
    <OutputFormat.JSON: 'json'>

    >>> detect_format('- item 1\\n- item 2')
    <OutputFormat.LIST: 'list'>

    See Also
    --------
    OutputDetector : The underlying detector class.
    OutputFormat : The returned enum type.
    """
    return OutputDetector.detect(text)


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from text.

    This is a convenience function that wraps
    AnswerExtractor.extract_final_answer() for simpler access to
    answer extraction.

    Parameters
    ----------
    text : str
        The text to search for an answer.

    Returns
    -------
    Optional[str]
        The extracted answer or None if not found.

    Examples
    --------
    >>> extract_answer("The answer is: 42")
    '42'

    >>> extract_answer("Therefore, Paris is the capital.")
    'Paris is the capital.'

    See Also
    --------
    AnswerExtractor : The underlying extractor class.
    """
    return AnswerExtractor.extract_final_answer(text)
