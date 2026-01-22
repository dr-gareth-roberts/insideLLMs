"""Tests for insideLLMs/nlp/extraction.py module."""

import pytest

from insideLLMs.nlp.extraction import (
    extract_emails,
    extract_hashtags,
    extract_ip_addresses,
    extract_mentions,
    extract_phone_numbers,
    extract_urls,
)


class TestExtractEmails:
    """Tests for extract_emails function."""

    def test_basic_email(self):
        """Test extracting basic email."""
        text = "Contact me at test@example.com"
        result = extract_emails(text)
        assert "test@example.com" in result

    def test_multiple_emails(self):
        """Test extracting multiple emails."""
        text = "Contact test@example.com or admin@test.org"
        result = extract_emails(text)
        assert len(result) == 2

    def test_complex_email(self):
        """Test extracting complex email addresses."""
        text = "Email: first.last+tag@subdomain.example.co.uk"
        result = extract_emails(text)
        assert len(result) == 1

    def test_no_emails(self):
        """Test when no emails present."""
        text = "No email addresses here"
        result = extract_emails(text)
        assert result == []


class TestExtractPhoneNumbers:
    """Tests for extract_phone_numbers function."""

    def test_us_phone_number(self):
        """Test extracting US phone number."""
        text = "Call me at 555-123-4567"
        result = extract_phone_numbers(text, country="us")
        assert len(result) >= 1

    def test_us_phone_with_area_code(self):
        """Test extracting US phone with area code."""
        text = "Call (555) 123-4567"
        result = extract_phone_numbers(text, country="us")
        assert len(result) >= 1

    def test_international_phone(self):
        """Test extracting international phone number."""
        text = "Call +1-555-123-4567"
        result = extract_phone_numbers(text, country="international")
        assert len(result) >= 1

    def test_uk_phone(self):
        """Test extracting UK phone number."""
        text = "Call +44 20 7123 4567"
        result = extract_phone_numbers(text, country="uk")
        assert len(result) >= 1

    def test_no_phone_numbers(self):
        """Test when no phone numbers present."""
        text = "No phone numbers here"
        result = extract_phone_numbers(text)
        assert result == []


class TestExtractUrls:
    """Tests for extract_urls function."""

    def test_http_url(self):
        """Test extracting HTTP URL."""
        text = "Visit http://example.com for more"
        result = extract_urls(text)
        assert len(result) >= 1

    def test_https_url(self):
        """Test extracting HTTPS URL."""
        text = "Visit https://example.com/page for more"
        result = extract_urls(text)
        assert len(result) >= 1

    def test_www_url(self):
        """Test extracting www URL."""
        text = "Visit www.example.com"
        result = extract_urls(text)
        assert len(result) >= 1

    def test_multiple_urls(self):
        """Test extracting multiple URLs."""
        text = "Check http://a.com and https://b.com"
        result = extract_urls(text)
        assert len(result) >= 2

    def test_no_urls(self):
        """Test when no URLs present."""
        text = "No URLs here"
        result = extract_urls(text)
        assert result == []


class TestExtractHashtags:
    """Tests for extract_hashtags function."""

    def test_basic_hashtag(self):
        """Test extracting basic hashtag."""
        text = "Check out #Python programming"
        result = extract_hashtags(text)
        assert "#Python" in result

    def test_multiple_hashtags(self):
        """Test extracting multiple hashtags."""
        text = "#Python #MachineLearning #AI"
        result = extract_hashtags(text)
        assert len(result) == 3

    def test_hashtag_with_numbers(self):
        """Test hashtag with numbers."""
        text = "Welcome to #Python3"
        result = extract_hashtags(text)
        assert "#Python3" in result

    def test_no_hashtags(self):
        """Test when no hashtags present."""
        text = "No hashtags here"
        result = extract_hashtags(text)
        assert result == []


class TestExtractMentions:
    """Tests for extract_mentions function."""

    def test_basic_mention(self):
        """Test extracting basic mention."""
        text = "Hello @user how are you"
        result = extract_mentions(text)
        assert "@user" in result

    def test_multiple_mentions(self):
        """Test extracting multiple mentions."""
        text = "Hello @user1 and @user2"
        result = extract_mentions(text)
        assert len(result) == 2

    def test_mention_with_underscore(self):
        """Test mention with underscore."""
        text = "Hello @user_name"
        result = extract_mentions(text)
        assert "@user_name" in result

    def test_no_mentions(self):
        """Test when no mentions present."""
        text = "No mentions here"
        result = extract_mentions(text)
        assert result == []


class TestExtractIpAddresses:
    """Tests for extract_ip_addresses function."""

    def test_basic_ipv4(self):
        """Test extracting basic IPv4 address."""
        text = "Server IP is 192.168.1.1"
        result = extract_ip_addresses(text)
        assert "192.168.1.1" in result

    def test_multiple_ips(self):
        """Test extracting multiple IPs."""
        text = "Connect to 10.0.0.1 or 172.16.0.1"
        result = extract_ip_addresses(text)
        assert len(result) == 2

    def test_edge_case_ips(self):
        """Test edge case IP addresses."""
        text = "Range: 0.0.0.0 to 255.255.255.255"
        result = extract_ip_addresses(text)
        assert len(result) == 2

    def test_no_ips(self):
        """Test when no IPs present."""
        text = "No IP addresses here"
        result = extract_ip_addresses(text)
        assert result == []

    def test_invalid_ip_not_extracted(self):
        """Test that invalid IPs are not extracted."""
        text = "Invalid: 999.999.999.999"
        result = extract_ip_addresses(text)
        # 999 is > 255, should not match
        assert len(result) == 0


class TestNamedEntityRecognition:
    """Tests for NER functions."""

    @pytest.fixture
    def spacy_available(self):
        """Check if spaCy is available."""
        try:
            import spacy

            spacy.load("en_core_web_sm")
            return True
        except (ImportError, OSError):
            pytest.skip("spaCy with en_core_web_sm not available")

    def test_extract_named_entities(self, spacy_available):
        """Test extracting named entities."""
        from insideLLMs.nlp.extraction import extract_named_entities

        text = "Apple Inc. is based in Cupertino, California."
        result = extract_named_entities(text)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_extract_entities_by_type(self, spacy_available):
        """Test extracting entities by type."""
        from insideLLMs.nlp.extraction import extract_entities_by_type

        text = "Barack Obama visited Paris in 2015."
        result = extract_entities_by_type(text, ["PERSON", "GPE", "DATE"])
        assert isinstance(result, dict)
