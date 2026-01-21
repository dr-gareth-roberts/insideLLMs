"""Tests for insideLLMs/nlp/char_level.py module."""

import pytest

from insideLLMs.nlp.char_level import (
    get_char_frequency,
    get_char_ngrams,
    to_camelcase,
    to_snakecase,
    to_titlecase,
    to_uppercase,
)


class TestGetCharNgrams:
    """Tests for get_char_ngrams function."""

    def test_bigrams(self):
        """Test generating character bigrams."""
        text = "hello"
        result = get_char_ngrams(text, n=2)
        assert result == ["he", "el", "ll", "lo"]

    def test_trigrams(self):
        """Test generating character trigrams."""
        text = "hello"
        result = get_char_ngrams(text, n=3)
        assert result == ["hel", "ell", "llo"]

    def test_unigrams(self):
        """Test generating character unigrams."""
        text = "abc"
        result = get_char_ngrams(text, n=1)
        assert result == ["a", "b", "c"]

    def test_empty_string(self):
        """Test with empty string."""
        result = get_char_ngrams("", n=2)
        assert result == []

    def test_short_string(self):
        """Test when string is shorter than n."""
        text = "a"
        result = get_char_ngrams(text, n=2)
        assert result == []

    def test_string_equals_n(self):
        """Test when string length equals n."""
        text = "ab"
        result = get_char_ngrams(text, n=2)
        assert result == ["ab"]

    def test_with_spaces(self):
        """Test with spaces in text."""
        text = "a b"
        result = get_char_ngrams(text, n=2)
        assert result == ["a ", " b"]


class TestGetCharFrequency:
    """Tests for get_char_frequency function."""

    def test_basic_frequency(self):
        """Test basic character frequency."""
        text = "hello"
        result = get_char_frequency(text)
        assert result["l"] == 2
        assert result["h"] == 1
        assert result["e"] == 1
        assert result["o"] == 1

    def test_empty_string(self):
        """Test with empty string."""
        result = get_char_frequency("")
        assert result == {}

    def test_single_char(self):
        """Test with single character."""
        result = get_char_frequency("a")
        assert result == {"a": 1}

    def test_repeated_char(self):
        """Test with repeated character."""
        result = get_char_frequency("aaa")
        assert result == {"a": 3}

    def test_with_spaces(self):
        """Test with spaces."""
        text = "a b"
        result = get_char_frequency(text)
        assert result["a"] == 1
        assert result[" "] == 1
        assert result["b"] == 1


class TestToUppercase:
    """Tests for to_uppercase function."""

    def test_lowercase_to_upper(self):
        """Test converting lowercase to uppercase."""
        result = to_uppercase("hello")
        assert result == "HELLO"

    def test_mixed_case(self):
        """Test converting mixed case."""
        result = to_uppercase("Hello World")
        assert result == "HELLO WORLD"

    def test_already_upper(self):
        """Test text already uppercase."""
        result = to_uppercase("HELLO")
        assert result == "HELLO"

    def test_empty_string(self):
        """Test empty string."""
        result = to_uppercase("")
        assert result == ""


class TestToTitlecase:
    """Tests for to_titlecase function."""

    def test_lowercase_to_title(self):
        """Test converting lowercase to title case."""
        result = to_titlecase("hello world")
        assert result == "Hello World"

    def test_uppercase_to_title(self):
        """Test converting uppercase to title case."""
        result = to_titlecase("HELLO WORLD")
        assert result == "Hello World"

    def test_already_title(self):
        """Test text already in title case."""
        result = to_titlecase("Hello World")
        assert result == "Hello World"

    def test_empty_string(self):
        """Test empty string."""
        result = to_titlecase("")
        assert result == ""


class TestToCamelcase:
    """Tests for to_camelcase function."""

    def test_space_separated(self):
        """Test converting space-separated words."""
        result = to_camelcase("hello world")
        assert result == "helloWorld"

    def test_underscore_separated(self):
        """Test converting underscore-separated words."""
        result = to_camelcase("hello_world")
        assert result == "helloWorld"

    def test_mixed_separators(self):
        """Test with mixed separators."""
        result = to_camelcase("hello_world test")
        assert result == "helloWorldTest"

    def test_single_word(self):
        """Test single word."""
        result = to_camelcase("hello")
        assert result == "hello"

    def test_empty_string(self):
        """Test empty string."""
        result = to_camelcase("")
        assert result == ""

    def test_uppercase_words(self):
        """Test with uppercase words."""
        result = to_camelcase("HELLO WORLD")
        assert result.lower() == "helloworld"


class TestToSnakecase:
    """Tests for to_snakecase function."""

    def test_space_separated(self):
        """Test converting space-separated words."""
        result = to_snakecase("hello world")
        assert result == "hello_world"

    def test_camel_case(self):
        """Test converting camelCase."""
        result = to_snakecase("helloWorld")
        assert result == "hello_world"

    def test_pascal_case(self):
        """Test converting PascalCase."""
        result = to_snakecase("HelloWorld")
        assert result == "hello_world"

    def test_hyphen_separated(self):
        """Test converting hyphen-separated words."""
        result = to_snakecase("hello-world")
        assert result == "hello_world"

    def test_single_word(self):
        """Test single word."""
        result = to_snakecase("hello")
        assert result == "hello"

    def test_empty_string(self):
        """Test empty string."""
        result = to_snakecase("")
        assert result == ""

    def test_multiple_underscores(self):
        """Test that multiple underscores are collapsed."""
        result = to_snakecase("hello  world")
        assert "__" not in result

    def test_special_characters_removed(self):
        """Test that special characters are removed."""
        result = to_snakecase("hello@world!")
        assert "@" not in result
        assert "!" not in result
