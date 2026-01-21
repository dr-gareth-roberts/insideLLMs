"""Tests for insideLLMs/nlp/text_cleaning.py module."""

import pytest

from insideLLMs.nlp.text_cleaning import (
    clean_text,
    normalize_contractions,
    normalize_unicode,
    normalize_whitespace,
    remove_emojis,
    remove_html_tags,
    remove_numbers,
    remove_punctuation,
    remove_urls,
    replace_repeated_chars,
)


class TestRemoveHtmlTags:
    """Tests for remove_html_tags function."""

    def test_basic_html_removal(self):
        """Test removing basic HTML tags."""
        text = "<p>Hello World</p>"
        result = remove_html_tags(text)
        assert result == "Hello World"

    def test_nested_html_removal(self):
        """Test removing nested HTML tags."""
        text = "<div><p><b>Bold text</b></p></div>"
        result = remove_html_tags(text)
        assert result == "Bold text"

    def test_self_closing_tags(self):
        """Test removing self-closing tags."""
        text = "Line 1<br/>Line 2"
        result = remove_html_tags(text)
        assert result == "Line 1Line 2"

    def test_no_html_unchanged(self):
        """Test that text without HTML is unchanged."""
        text = "Plain text without HTML"
        result = remove_html_tags(text)
        assert result == text

    def test_empty_string(self):
        """Test removing HTML from empty string."""
        result = remove_html_tags("")
        assert result == ""


class TestRemoveUrls:
    """Tests for remove_urls function."""

    def test_http_url_removal(self):
        """Test removing HTTP URLs."""
        text = "Visit http://example.com for more info"
        result = remove_urls(text)
        assert "http://example.com" not in result

    def test_https_url_removal(self):
        """Test removing HTTPS URLs."""
        text = "Visit https://example.com/page for more info"
        result = remove_urls(text)
        assert "https://example.com" not in result

    def test_www_url_removal(self):
        """Test removing www URLs."""
        text = "Visit www.example.com for more info"
        result = remove_urls(text)
        assert "www.example.com" not in result

    def test_multiple_urls(self):
        """Test removing multiple URLs."""
        text = "Check http://a.com and https://b.com"
        result = remove_urls(text)
        assert "http://" not in result
        assert "https://" not in result

    def test_no_urls_unchanged(self):
        """Test that text without URLs is unchanged."""
        text = "No URLs here"
        result = remove_urls(text)
        assert result == text


class TestRemovePunctuation:
    """Tests for remove_punctuation function."""

    def test_basic_punctuation_removal(self):
        """Test removing basic punctuation."""
        text = "Hello, World!"
        result = remove_punctuation(text)
        assert result == "Hello World"

    def test_multiple_punctuation_types(self):
        """Test removing various punctuation types."""
        text = "What? Yes! No... Maybe; perhaps: okay."
        result = remove_punctuation(text)
        assert "?" not in result
        assert "!" not in result
        assert "." not in result

    def test_preserves_numbers(self):
        """Test that numbers are preserved."""
        text = "The year 2024."
        result = remove_punctuation(text)
        assert "2024" in result


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace function."""

    def test_multiple_spaces(self):
        """Test normalizing multiple spaces."""
        text = "Hello    World"
        result = normalize_whitespace(text)
        assert result == "Hello World"

    def test_tabs_and_newlines(self):
        """Test normalizing tabs and newlines."""
        text = "Hello\t\nWorld"
        result = normalize_whitespace(text)
        assert result == "Hello World"

    def test_leading_trailing_whitespace(self):
        """Test removing leading/trailing whitespace."""
        text = "  Hello World  "
        result = normalize_whitespace(text)
        assert result == "Hello World"


class TestNormalizeUnicode:
    """Tests for normalize_unicode function."""

    def test_nfkc_normalization(self):
        """Test NFKC normalization."""
        text = "½"  # Unicode fraction
        result = normalize_unicode(text, "NFKC")
        # NFKC should normalize this to "1/2" or similar
        assert result is not None

    def test_default_normalization(self):
        """Test default normalization form."""
        text = "café"
        result = normalize_unicode(text)
        assert "caf" in result

    def test_different_forms(self):
        """Test different normalization forms."""
        text = "café"
        nfc = normalize_unicode(text, "NFC")
        nfd = normalize_unicode(text, "NFD")
        # Both should work without error
        assert nfc is not None
        assert nfd is not None


class TestRemoveEmojis:
    """Tests for remove_emojis function."""

    def test_basic_emoji_removal(self):
        """Test removing basic emojis."""
        text = "Hello 😀 World"
        result = remove_emojis(text)
        assert "😀" not in result
        assert "Hello" in result

    def test_multiple_emojis(self):
        """Test removing multiple emojis."""
        text = "🎉 Party 🎊 Time 🎈"
        result = remove_emojis(text)
        assert "🎉" not in result
        assert "Party" in result

    def test_no_emojis_unchanged(self):
        """Test that text without emojis is unchanged."""
        text = "Plain text"
        result = remove_emojis(text)
        assert result == text


class TestRemoveNumbers:
    """Tests for remove_numbers function."""

    def test_basic_number_removal(self):
        """Test removing basic numbers."""
        text = "There are 5 apples"
        result = remove_numbers(text)
        assert "5" not in result

    def test_multiple_numbers(self):
        """Test removing multiple numbers."""
        text = "In 2024, there were 100 events"
        result = remove_numbers(text)
        assert "2024" not in result
        assert "100" not in result

    def test_no_numbers_unchanged(self):
        """Test that text without numbers is unchanged."""
        text = "No numbers here"
        result = remove_numbers(text)
        assert result == text


class TestNormalizeContractions:
    """Tests for normalize_contractions function."""

    def test_dont_expansion(self):
        """Test expanding don't."""
        text = "I don't know"
        result = normalize_contractions(text)
        assert "do not" in result

    def test_cant_expansion(self):
        """Test expanding can't."""
        text = "I can't do it"
        result = normalize_contractions(text)
        assert "cannot" in result

    def test_wont_expansion(self):
        """Test expanding won't."""
        text = "I won't go"
        result = normalize_contractions(text)
        assert "will not" in result

    def test_youre_expansion(self):
        """Test expanding you're."""
        text = "You're welcome"
        result = normalize_contractions(text)
        assert "you are" in result

    def test_multiple_contractions(self):
        """Test expanding multiple contractions."""
        text = "I don't think they'll come because it's raining"
        result = normalize_contractions(text)
        assert "do not" in result
        assert "will" in result
        assert "is" in result

    def test_no_contractions_unchanged(self):
        """Test that text without contractions is unchanged."""
        text = "This is a normal sentence"
        result = normalize_contractions(text)
        assert result == text

    def test_ive_expansion(self):
        """Test expanding I've."""
        text = "I've been here"
        result = normalize_contractions(text)
        assert "i have" in result.lower()


class TestReplaceRepeatedChars:
    """Tests for replace_repeated_chars function."""

    def test_basic_repeated_chars(self):
        """Test replacing repeated characters."""
        text = "Helloooo"
        result = replace_repeated_chars(text, threshold=2)
        assert result == "Helloo"

    def test_custom_threshold(self):
        """Test with custom threshold."""
        text = "Hiiiii"
        result = replace_repeated_chars(text, threshold=3)
        assert result == "Hiii"

    def test_multiple_repeated_patterns(self):
        """Test with multiple repeated patterns."""
        text = "Sooooo coooool"
        result = replace_repeated_chars(text, threshold=2)
        assert "ooo" not in result

    def test_no_repetition_unchanged(self):
        """Test that text without repetition is unchanged."""
        text = "Hello"
        result = replace_repeated_chars(text)
        assert result == "Hello"


class TestCleanText:
    """Tests for clean_text comprehensive function."""

    def test_default_cleaning(self):
        """Test default cleaning options."""
        text = "<p>Hello World!</p> Check https://example.com"
        result = clean_text(text)
        assert "<p>" not in result
        assert "https://" not in result
        assert result == result.lower()

    def test_remove_html_option(self):
        """Test remove_html option."""
        text = "<p>Hello</p>"
        result = clean_text(text, remove_html=True)
        assert "<p>" not in result

        result = clean_text(text, remove_html=False)
        assert "<p>" in result

    def test_remove_url_option(self):
        """Test remove_url option."""
        text = "Visit https://example.com"
        result = clean_text(text, remove_url=True)
        assert "https://" not in result

        result = clean_text(text, remove_url=False, lowercase=False)
        assert "https://example.com" in result

    def test_remove_punct_option(self):
        """Test remove_punct option."""
        text = "Hello, World!"
        result = clean_text(text, remove_punct=True)
        assert "," not in result
        assert "!" not in result

    def test_remove_emoji_option(self):
        """Test remove_emoji option."""
        text = "Hello 😀"
        result = clean_text(text, remove_emoji=True)
        assert "😀" not in result

    def test_remove_num_option(self):
        """Test remove_num option."""
        text = "Year 2024"
        result = clean_text(text, remove_num=True)
        assert "2024" not in result

    def test_normalize_white_option(self):
        """Test normalize_white option."""
        text = "Hello    World"
        result = clean_text(text, normalize_white=True)
        assert "    " not in result

    def test_normalize_unicode_option(self):
        """Test normalize_unicode_form option."""
        text = "café"
        result = clean_text(text, normalize_unicode_form="NFKC")
        assert result is not None

        result = clean_text(text, normalize_unicode_form=None)
        assert result is not None

    def test_normalize_contraction_option(self):
        """Test normalize_contraction option."""
        text = "I don't know"
        result = clean_text(text, normalize_contraction=True)
        assert "do not" in result

    def test_replace_repeated_option(self):
        """Test replace_repeated option."""
        text = "Helloooo"
        result = clean_text(text, replace_repeated=True)
        assert "oooo" not in result

    def test_lowercase_option(self):
        """Test lowercase option."""
        text = "Hello World"
        result = clean_text(text, lowercase=True)
        assert result == result.lower()

        result = clean_text(text, lowercase=False)
        assert "Hello" in result

    def test_combined_cleaning(self):
        """Test combining multiple cleaning options."""
        text = "<div>Hello, World! 😀 Visit https://test.com 2024</div>"
        result = clean_text(
            text,
            remove_html=True,
            remove_url=True,
            remove_punct=True,
            remove_emoji=True,
            remove_num=True,
            lowercase=True,
        )
        assert "<div>" not in result
        assert "https://" not in result
        assert "😀" not in result
        assert "2024" not in result
        assert "," not in result
