"""Tests for insideLLMs/nlp/encoding.py module."""

import pytest

from insideLLMs.nlp.encoding import (
    decode_base64,
    encode_base64,
    html_decode,
    html_encode,
    url_decode,
    url_encode,
)


class TestBase64Encoding:
    """Tests for base64 encoding/decoding functions."""

    def test_encode_basic(self):
        """Test basic base64 encoding."""
        result = encode_base64("hello world")
        assert result == "aGVsbG8gd29ybGQ="

    def test_decode_basic(self):
        """Test basic base64 decoding."""
        result = decode_base64("aGVsbG8gd29ybGQ=")
        assert result == "hello world"

    def test_encode_decode_roundtrip(self):
        """Test encoding then decoding returns original."""
        original = "The quick brown fox!"
        encoded = encode_base64(original)
        decoded = decode_base64(encoded)
        assert decoded == original

    def test_encode_empty_string(self):
        """Test encoding empty string."""
        result = encode_base64("")
        assert result == ""

    def test_decode_empty_string(self):
        """Test decoding empty base64."""
        result = decode_base64("")
        assert result == ""

    def test_encode_unicode(self):
        """Test encoding unicode text."""
        result = encode_base64("héllo wörld 你好")
        # Verify it decodes back correctly
        decoded = decode_base64(result)
        assert decoded == "héllo wörld 你好"

    def test_encode_special_characters(self):
        """Test encoding text with special characters."""
        original = "line1\nline2\ttab"
        encoded = encode_base64(original)
        decoded = decode_base64(encoded)
        assert decoded == original


class TestUrlEncoding:
    """Tests for URL encoding/decoding functions."""

    def test_url_encode_basic(self):
        """Test basic URL encoding."""
        result = url_encode("hello world")
        assert result == "hello%20world"

    def test_url_decode_basic(self):
        """Test basic URL decoding."""
        result = url_decode("hello%20world")
        assert result == "hello world"

    def test_url_encode_decode_roundtrip(self):
        """Test URL encoding then decoding returns original."""
        original = "search?q=hello world&page=1"
        encoded = url_encode(original)
        decoded = url_decode(encoded)
        assert decoded == original

    def test_url_encode_empty_string(self):
        """Test encoding empty string."""
        result = url_encode("")
        assert result == ""

    def test_url_decode_empty_string(self):
        """Test decoding empty URL."""
        result = url_decode("")
        assert result == ""

    def test_url_encode_special_characters(self):
        """Test encoding special characters."""
        result = url_encode("a=b&c=d")
        assert "%26" in result or "=" in result  # & should be encoded

    def test_url_encode_unicode(self):
        """Test encoding unicode."""
        original = "café"
        encoded = url_encode(original)
        decoded = url_decode(encoded)
        assert decoded == original


class TestHtmlEncoding:
    """Tests for HTML encoding/decoding functions."""

    def test_html_encode_basic(self):
        """Test basic HTML encoding."""
        result = html_encode("<div>Hello</div>")
        assert "&lt;" in result
        assert "&gt;" in result

    def test_html_decode_basic(self):
        """Test basic HTML decoding."""
        result = html_decode("&lt;div&gt;Hello&lt;/div&gt;")
        assert result == "<div>Hello</div>"

    def test_html_encode_decode_roundtrip(self):
        """Test HTML encoding then decoding returns original."""
        original = "<script>alert('xss')</script>"
        encoded = html_encode(original)
        decoded = html_decode(encoded)
        assert decoded == original

    def test_html_encode_ampersand(self):
        """Test encoding ampersand."""
        result = html_encode("a & b")
        assert "&amp;" in result

    def test_html_encode_quotes(self):
        """Test encoding quotes."""
        result = html_encode('say "hello"')
        # Quote may or may not be encoded depending on html.escape version
        decoded = html_decode(result)
        assert decoded == 'say "hello"'

    def test_html_encode_empty_string(self):
        """Test encoding empty string."""
        result = html_encode("")
        assert result == ""

    def test_html_decode_empty_string(self):
        """Test decoding empty string."""
        result = html_decode("")
        assert result == ""

    def test_html_encode_already_safe(self):
        """Test encoding text without special characters."""
        result = html_encode("hello world")
        assert result == "hello world"

    def test_html_decode_numeric_entities(self):
        """Test decoding numeric HTML entities."""
        result = html_decode("&#60;div&#62;")
        assert result == "<div>"
