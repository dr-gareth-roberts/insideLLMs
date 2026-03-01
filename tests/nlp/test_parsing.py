"""Tests for structured output parsing utilities."""

import pytest

from insideLLMs.nlp.parsing import (
    AnswerExtractor,
    CodeBlock,
    CodeBlockParser,
    JSONParser,
    ListParser,
    OutputDetector,
    OutputFormat,
    ParseResult,
    TableData,
    TableParser,
    detect_format,
    extract_answer,
    parse_code,
    parse_json,
    parse_list,
    parse_table,
)


class TestParseResult:
    """Tests for ParseResult dataclass."""

    def test_success_result(self):
        """Test successful parse result."""
        result = ParseResult(success=True, value={"key": "value"}, raw='{"key": "value"}')
        assert result.is_ok
        assert result.unwrap() == {"key": "value"}

    def test_failed_result(self):
        """Test failed parse result."""
        result = ParseResult(success=False, value=None, raw="invalid", error="Parse error")
        assert not result.is_ok
        with pytest.raises(ValueError):
            result.unwrap()

    def test_unwrap_or(self):
        """Test unwrap_or with default."""
        success = ParseResult(success=True, value=42, raw="42")
        failed = ParseResult(success=False, value=None, raw="bad", error="Error")

        assert success.unwrap_or(0) == 42
        assert failed.unwrap_or(0) == 0

    def test_map(self):
        """Test mapping over result."""
        result = ParseResult(success=True, value=5, raw="5")
        mapped = result.map(lambda x: x * 2)

        assert mapped.success
        assert mapped.unwrap() == 10

    def test_map_failure(self):
        """Test map on failed result."""
        result = ParseResult(success=False, value=None, raw="bad", error="Error")
        mapped = result.map(lambda x: x * 2)

        assert not mapped.success


class TestJSONParser:
    """Tests for JSONParser."""

    def test_parse_simple_json(self):
        """Test parsing simple JSON."""
        result = JSONParser.parse('{"key": "value"}')
        assert result.success
        assert result.unwrap() == {"key": "value"}

    def test_parse_array(self):
        """Test parsing JSON array."""
        result = JSONParser.parse("[1, 2, 3]")
        assert result.success
        assert result.unwrap() == [1, 2, 3]

    def test_parse_from_code_block(self):
        """Test parsing JSON from markdown code block."""
        text = """Here is the data:
```json
{"name": "Alice", "age": 30}
```
That's all."""

        result = JSONParser.parse(text)
        assert result.success
        assert result.unwrap() == {"name": "Alice", "age": 30}

    def test_parse_embedded_json(self):
        """Test parsing JSON embedded in text."""
        text = 'The result is {"status": "ok", "count": 5} as expected.'

        result = JSONParser.parse(text)
        assert result.success
        assert result.unwrap() == {"status": "ok", "count": 5}

    def test_parse_with_trailing_comma(self):
        """Test handling trailing commas."""
        text = '{"items": [1, 2, 3,], "name": "test",}'

        result = JSONParser.parse(text)
        assert result.success
        assert result.unwrap()["name"] == "test"

    def test_parse_strict_mode(self):
        """Test strict mode rejects invalid JSON."""
        text = "not json at all"

        result = JSONParser.parse(text, strict=True)
        assert not result.success
        assert "strict mode" in result.error

    def test_parse_no_json_found(self):
        """Test error when no JSON found."""
        result = JSONParser.parse("This is just plain text.")
        assert not result.success
        assert "No JSON found" in result.error

    def test_extract_all(self):
        """Test extracting multiple JSON objects."""
        text = """First:
```json
{"a": 1}
```
Second:
```json
{"b": 2}
```"""

        results = JSONParser.extract_all(text)
        assert len(results) >= 2  # Should find at least the objects


class TestCodeBlockParser:
    """Tests for CodeBlockParser."""

    def test_parse_python_block(self):
        """Test parsing Python code block."""
        text = """Here's some code:
```python
def hello():
    print("Hello!")
```
That's it."""

        blocks = CodeBlockParser.parse(text)
        assert len(blocks) == 1
        assert blocks[0].language == "python"
        assert "def hello" in blocks[0].code

    def test_parse_multiple_blocks(self):
        """Test parsing multiple code blocks."""
        text = """Python:
```python
x = 1
```

JavaScript:
```javascript
const x = 1;
```"""

        blocks = CodeBlockParser.parse(text)
        assert len(blocks) == 2
        assert blocks[0].language == "python"
        assert blocks[1].language == "javascript"

    def test_parse_no_language(self):
        """Test parsing block without language."""
        text = """```
some code
```"""

        blocks = CodeBlockParser.parse(text)
        assert len(blocks) == 1
        assert blocks[0].language is None

    def test_parse_first(self):
        """Test getting first code block."""
        text = """```python
first
```
```javascript
second
```"""

        block = CodeBlockParser.parse_first(text)
        assert block is not None
        assert "first" in block.code

    def test_parse_by_language(self):
        """Test filtering by language."""
        text = """```python
py1
```
```javascript
js
```
```python
py2
```"""

        blocks = CodeBlockParser.parse_by_language(text, "python")
        assert len(blocks) == 2

    def test_extract_python(self):
        """Test Python extraction convenience method."""
        text = """```python
def foo():
    pass
```"""

        code = CodeBlockParser.extract_python(text)
        assert len(code) == 1
        assert "def foo" in code[0]

    def test_language_aliases(self):
        """Test language alias resolution."""
        text = """```py
x = 1
```
```js
const y = 2;
```"""

        blocks = CodeBlockParser.parse(text)
        assert blocks[0].language == "python"
        assert blocks[1].language == "javascript"


class TestCodeBlock:
    """Tests for CodeBlock dataclass."""

    def test_str_conversion(self):
        """Test string conversion."""
        block = CodeBlock(code="print('hello')", language="python")
        assert str(block) == "print('hello')"


class TestListParser:
    """Tests for ListParser."""

    def test_parse_numbered_list(self):
        """Test parsing numbered list."""
        text = """1. First item
2. Second item
3. Third item"""

        items = ListParser.parse(text)
        assert len(items) == 3
        assert items[0] == "First item"

    def test_parse_bullet_list(self):
        """Test parsing bullet list."""
        text = """- Item one
- Item two
- Item three"""

        items = ListParser.parse(text)
        assert len(items) == 3
        assert items[0] == "Item one"

    def test_parse_asterisk_list(self):
        """Test parsing asterisk bullet list."""
        text = """* First
* Second
* Third"""

        items = ListParser.parse(text)
        assert len(items) == 3

    def test_parse_lettered_list(self):
        """Test parsing lettered list."""
        text = """a) Option A
b) Option B
c) Option C"""

        items = ListParser.parse(text)
        assert len(items) == 3
        assert items[0] == "Option A"

    def test_parse_empty_text(self):
        """Test parsing text with no list."""
        items = ListParser.parse("Just regular text.")
        assert len(items) == 0


class TestTableParser:
    """Tests for TableParser."""

    def test_parse_markdown_table(self):
        """Test parsing markdown table."""
        text = """| Name | Age | City |
| --- | --- | --- |
| Alice | 30 | NYC |
| Bob | 25 | LA |"""

        table = TableParser.parse(text)
        assert table is not None
        assert table.headers == ["Name", "Age", "City"]
        assert table.num_rows == 2
        assert table.num_cols == 3

    def test_table_to_dicts(self):
        """Test converting table to dictionaries."""
        text = """| Name | Score |
| --- | --- |
| Alice | 95 |
| Bob | 87 |"""

        table = TableParser.parse(text)
        dicts = table.to_dicts()

        assert len(dicts) == 2
        assert dicts[0]["Name"] == "Alice"
        assert dicts[0]["Score"] == "95"

    def test_get_column(self):
        """Test getting a column by name."""
        text = """| Name | Score |
| --- | --- |
| Alice | 95 |
| Bob | 87 |"""

        table = TableParser.parse(text)
        names = table.get_column("Name")

        assert names == ["Alice", "Bob"]

    def test_get_column_not_found(self):
        """Test error on missing column."""
        text = """| A | B |
| --- | --- |
| 1 | 2 |"""

        table = TableParser.parse(text)
        with pytest.raises(KeyError):
            table.get_column("C")

    def test_parse_no_table(self):
        """Test parsing text without table."""
        table = TableParser.parse("No table here.")
        assert table is None

    def test_parse_all_tables(self):
        """Test parsing multiple tables."""
        text = """First table:
| A | B |
| --- | --- |
| 1 | 2 |

Second table:
| X | Y |
| --- | --- |
| 3 | 4 |"""

        tables = TableParser.parse_all(text)
        assert len(tables) == 2


class TestTableData:
    """Tests for TableData dataclass."""

    def test_properties(self):
        """Test table properties."""
        table = TableData(
            headers=["A", "B", "C"],
            rows=[["1", "2", "3"], ["4", "5", "6"]],
        )
        assert table.num_rows == 2
        assert table.num_cols == 3


class TestAnswerExtractor:
    """Tests for AnswerExtractor."""

    def test_extract_final_answer(self):
        """Test extracting final answer."""
        text = "After calculation, the final answer is: 42"
        answer = AnswerExtractor.extract_final_answer(text)
        assert answer == "42"

    def test_extract_answer_equals(self):
        """Test extracting answer with equals."""
        text = "Therefore, answer = Paris"
        answer = AnswerExtractor.extract_final_answer(text)
        assert answer == "Paris"

    def test_extract_boxed_answer(self):
        """Test extracting LaTeX boxed answer."""
        text = r"The solution is \boxed{42}"
        answer = AnswerExtractor.extract_final_answer(text)
        assert answer == "42"

    def test_extract_yes_no_yes(self):
        """Test extracting yes answer."""
        text = "Yes, that is correct."
        result = AnswerExtractor.extract_yes_no(text)
        assert result is True

    def test_extract_yes_no_no(self):
        """Test extracting no answer."""
        text = "No, that's wrong."
        result = AnswerExtractor.extract_yes_no(text)
        assert result is False

    def test_extract_choice(self):
        """Test extracting multiple choice."""
        text = "After considering all options, the answer is B."
        choice = AnswerExtractor.extract_choice(text)
        assert choice == "B"

    def test_extract_choice_from_list(self):
        """Test extracting choice from list format."""
        text = """Looking at the options:
A) Wrong
B) Correct
C) Wrong

After analysis, B is the answer."""

        choice = AnswerExtractor.extract_choice(text)
        assert choice == "B"

    def test_extract_number(self):
        """Test extracting numeric answer."""
        text = "The result = 3.14"
        number = AnswerExtractor.extract_number(text)
        assert number == 3.14

    def test_extract_all_numbers(self):
        """Test extracting all numbers."""
        text = "Values are 1, 2.5, -3, and 42"
        numbers = AnswerExtractor.extract_all_numbers(text)
        assert 1.0 in numbers
        assert 2.5 in numbers
        assert -3.0 in numbers
        assert 42.0 in numbers


class TestOutputDetector:
    """Tests for OutputDetector."""

    def test_detect_json(self):
        """Test detecting JSON."""
        text = '{"key": "value"}'
        assert OutputDetector.detect(text) == OutputFormat.JSON

    def test_detect_code(self):
        """Test detecting code blocks."""
        text = """```python
print("hello")
```"""
        assert OutputDetector.detect(text) == OutputFormat.CODE

    def test_detect_table(self):
        """Test detecting table."""
        text = """| A | B |
| --- | --- |
| 1 | 2 |"""
        assert OutputDetector.detect(text) == OutputFormat.TABLE

    def test_detect_list(self):
        """Test detecting list."""
        text = """Here are the items:
- Item 1
- Item 2
- Item 3
- Item 4"""
        assert OutputDetector.detect(text) == OutputFormat.LIST

    def test_detect_xml(self):
        """Test detecting XML."""
        text = "<root><item>value</item></root>"
        assert OutputDetector.detect(text) == OutputFormat.XML

    def test_detect_yaml(self):
        """Test detecting YAML."""
        text = """name: test
value: 123
items:
  - one
  - two"""
        assert OutputDetector.detect(text) == OutputFormat.YAML

    def test_detect_markdown(self):
        """Test detecting markdown."""
        text = """# Header

This has **bold** and [links](http://example.com)."""
        assert OutputDetector.detect(text) == OutputFormat.MARKDOWN

    def test_detect_plain(self):
        """Test detecting plain text."""
        text = "Just regular text without any special formatting."
        assert OutputDetector.detect(text) == OutputFormat.PLAIN


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_parse_json_function(self):
        """Test parse_json convenience function."""
        result = parse_json('{"key": "value"}')
        assert result.success
        assert result.unwrap()["key"] == "value"

    def test_parse_code_function(self):
        """Test parse_code convenience function."""
        text = """```python
x = 1
```"""
        blocks = parse_code(text)
        assert len(blocks) == 1

    def test_parse_list_function(self):
        """Test parse_list convenience function."""
        text = """- A
- B
- C"""
        items = parse_list(text)
        assert len(items) == 3

    def test_parse_table_function(self):
        """Test parse_table convenience function."""
        text = """| A | B |
| --- | --- |
| 1 | 2 |"""
        table = parse_table(text)
        assert table is not None

    def test_detect_format_function(self):
        """Test detect_format convenience function."""
        assert detect_format('{"a": 1}') == OutputFormat.JSON

    def test_extract_answer_function(self):
        """Test extract_answer convenience function."""
        text = "The answer is: 42"
        answer = extract_answer(text)
        assert answer == "42"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input(self):
        """Test handling empty input."""
        assert parse_json("").success is False
        assert parse_code("") == []
        assert parse_list("") == []
        assert parse_table("") is None

    def test_whitespace_only(self):
        """Test handling whitespace-only input."""
        assert parse_json("   \n\t  ").success is False

    def test_mixed_content(self):
        """Test parsing mixed content."""
        text = """Here's some JSON:
```json
{"result": "success"}
```

And a list:
- Item 1
- Item 2

And a table:
| Col |
| --- |
| Val |"""

        # Should be able to extract each format
        json_result = parse_json(text)
        assert json_result.success

        code_blocks = parse_code(text)
        assert len(code_blocks) >= 1

        # Note: list and table detection may interfere
        # The primary format should be detected as CODE due to code block

    def test_malformed_json_recovery(self):
        """Test recovering from slightly malformed JSON."""
        # Single quotes instead of double
        text = "{'key': 'value'}"
        parse_json(text)
        # May or may not succeed depending on implementation
        # At minimum, shouldn't crash

    def test_nested_code_blocks(self):
        """Test handling nested-looking content."""
        text = """```python
# This has backticks in a string
s = "```"
print(s)
```"""

        blocks = parse_code(text)
        assert len(blocks) == 1


class TestRealWorldExamples:
    """Tests with realistic LLM output examples."""

    def test_chain_of_thought_answer(self):
        """Test extracting answer from chain-of-thought response."""
        text = """Let me think through this step by step.

1. First, we need to consider...
2. Then, we calculate...
3. Finally, we conclude...

Therefore, the answer is: 42"""

        answer = extract_answer(text)
        assert answer == "42"

    def test_code_with_explanation(self):
        """Test parsing code with surrounding explanation."""
        text = """Here's how to solve this problem:

```python
def solution(n):
    return n * 2

result = solution(21)
print(result)  # Output: 42
```

This function doubles the input value."""

        blocks = parse_code(text)
        assert len(blocks) == 1
        assert "def solution" in blocks[0].code
        assert blocks[0].language == "python"

    def test_structured_api_response(self):
        """Test parsing structured API response."""
        text = """Based on the query, here's the API response:

```json
{
  "status": "success",
  "data": {
    "users": [
      {"id": 1, "name": "Alice"},
      {"id": 2, "name": "Bob"}
    ]
  },
  "meta": {
    "total": 2,
    "page": 1
  }
}
```

The response contains 2 users."""

        result = parse_json(text)
        assert result.success
        data = result.unwrap()
        assert data["status"] == "success"
        assert len(data["data"]["users"]) == 2
