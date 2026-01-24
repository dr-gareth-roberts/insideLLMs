"""
Text extraction utilities for identifying and extracting structured data from unstructured text.

This module provides functions for extracting various types of structured information
from raw text, including email addresses, phone numbers, URLs, social media elements
(hashtags and mentions), IP addresses, and named entities using NLP.

The module is divided into two main categories:
1. Pattern Matching: Regex-based extraction for well-defined formats (emails, phones, URLs, etc.)
2. Named Entity Recognition: NLP-based extraction using spaCy models

Examples
--------
Basic pattern extraction:

>>> from insideLLMs.nlp.extraction import extract_emails, extract_urls
>>> text = "Contact us at support@example.com or visit https://example.com"
>>> extract_emails(text)
['support@example.com']
>>> extract_urls(text)
['https://example.com']

Social media extraction:

>>> from insideLLMs.nlp.extraction import extract_hashtags, extract_mentions
>>> tweet = "Great talk by @pythondev about #MachineLearning #NLP"
>>> extract_hashtags(tweet)
['#MachineLearning', '#NLP']
>>> extract_mentions(tweet)
['@pythondev']

Named entity recognition:

>>> from insideLLMs.nlp.extraction import extract_named_entities
>>> text = "Apple Inc. was founded by Steve Jobs in California."
>>> entities = extract_named_entities(text)  # Returns list of (entity, label) tuples

Notes
-----
- Pattern matching functions use compiled regex patterns for efficiency
- Named entity functions require spaCy and will auto-install models if needed
- Phone number extraction supports US, UK, and international formats
"""

import re
from collections import defaultdict

from insideLLMs.nlp.dependencies import ensure_spacy


# Backward compatibility alias
check_spacy = ensure_spacy


# ===== Pattern Matching and Extraction =====


def extract_emails(text: str) -> list[str]:
    """
    Extract email addresses from text using regex pattern matching.

    This function identifies and extracts all valid email addresses from the input
    text. It uses a comprehensive regex pattern that handles most common email
    formats, including addresses with dots, underscores, and plus signs in the
    local part.

    Parameters
    ----------
    text : str
        The input text to search for email addresses. Can be any length and
        may contain multiple email addresses.

    Returns
    -------
    list[str]
        A list of email addresses found in the text. Returns an empty list
        if no email addresses are found. Email addresses are returned in the
        order they appear in the text.

    Examples
    --------
    Extract a single email address:

    >>> extract_emails("Contact me at john.doe@example.com for more info")
    ['john.doe@example.com']

    Extract multiple email addresses:

    >>> text = "Send to alice@company.org or bob_smith@mail.co.uk"
    >>> extract_emails(text)
    ['alice@company.org', 'bob_smith@mail.co.uk']

    Handle email addresses with plus signs (common for email aliases):

    >>> extract_emails("Subscribe using user+newsletter@gmail.com")
    ['user+newsletter@gmail.com']

    Return empty list when no emails are found:

    >>> extract_emails("No email addresses in this text!")
    []

    Notes
    -----
    The regex pattern matches emails with:
    - Local part: alphanumeric characters, dots, underscores, percent, plus, hyphen
    - Domain: alphanumeric characters, dots, hyphens
    - TLD: alphabetic characters (minimum 2 characters)

    See Also
    --------
    extract_urls : Extract URLs from text
    extract_mentions : Extract @mentions from text
    """
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    return email_pattern.findall(text)


def extract_phone_numbers(text: str, country: str = "us") -> list[str]:
    """
    Extract phone numbers from text with support for multiple country formats.

    This function identifies and extracts phone numbers from text using
    country-specific regex patterns. It supports US, UK, and international
    phone number formats, handling various separators (spaces, hyphens) and
    optional country codes.

    Parameters
    ----------
    text : str
        The input text to search for phone numbers. Can contain multiple
        phone numbers in various formats.
    country : str, optional
        The country format to use for matching. Supported values:
        - "us" (default): US phone numbers (e.g., (555) 123-4567, +1 555-123-4567)
        - "uk": UK phone numbers (e.g., +44 20 7946 0958, 0207 946 0958)
        - "international": Generic international format with country codes
        Case-insensitive.

    Returns
    -------
    list[str]
        A list of phone numbers found in the text. Returns an empty list
        if no phone numbers matching the specified format are found.
        Numbers are returned as they appear in the text (not normalized).

    Examples
    --------
    Extract US phone numbers (default):

    >>> extract_phone_numbers("Call us at (555) 123-4567 or 555-987-6543")
    ['(555) 123-4567', '555-987-6543']

    Extract US numbers with country code:

    >>> extract_phone_numbers("International: +1-800-555-0199")
    ['+1-800-555-0199']

    Extract UK phone numbers:

    >>> extract_phone_numbers("London office: +44 20 7946 0958", country="uk")
    ['+44 20 7946 0958']

    Extract international format numbers:

    >>> text = "Germany: +49 30 12345678, Japan: +81 3 1234 5678"
    >>> extract_phone_numbers(text, country="international")
    ['+49 30 12345678', '+81 3 1234 5678']

    Notes
    -----
    - The function uses pre-defined patterns optimized for each country
    - If an unrecognized country code is provided, falls back to international format
    - Phone numbers are returned exactly as they appear (no normalization)
    - May match partial numbers in some edge cases; validate as needed

    See Also
    --------
    extract_emails : Extract email addresses from text
    extract_ip_addresses : Extract IP addresses from text
    """
    patterns = {
        "us": r"(?:\+?1[-\s]?)?(?:\(?[0-9]{3}\)?[-\s]?)?[0-9]{3}[-\s]?[0-9]{4}",
        "uk": r"(?:\+?44[-\s]?)?(?:\(?0[0-9]{1,4}\)?[-\s]?)?[0-9]{3,4}[-\s]?[0-9]{3,4}",
        "international": r"\+?[0-9]{1,3}[-\s]?[0-9]{1,4}[-\s]?[0-9]{1,4}[-\s]?[0-9]{1,9}",
    }

    pattern = patterns.get(country.lower(), patterns["international"])
    phone_pattern = re.compile(pattern)
    return phone_pattern.findall(text)


def extract_urls(text: str) -> list[str]:
    """
    Extract URLs from text, including both http(s) and www formats.

    This function identifies and extracts web URLs from the input text using
    regex pattern matching. It handles both fully-qualified URLs (with http://
    or https://) and URLs starting with www.

    Parameters
    ----------
    text : str
        The input text to search for URLs. Can contain multiple URLs
        in various formats and positions within the text.

    Returns
    -------
    list[str]
        A list of URLs found in the text. Returns an empty list if no URLs
        are found. URLs are returned in the order they appear and include
        the full URL path, query parameters, and fragments.

    Examples
    --------
    Extract a simple HTTPS URL:

    >>> extract_urls("Visit https://example.com for more info")
    ['https://example.com']

    Extract URLs with paths and parameters:

    >>> text = "Check https://api.example.com/v1/users?id=123&active=true"
    >>> extract_urls(text)
    ['https://api.example.com/v1/users?id=123&active=true']

    Extract www URLs (without protocol):

    >>> extract_urls("Go to www.example.org/docs for documentation")
    ['www.example.org/docs']

    Extract multiple URLs from text:

    >>> text = "Links: https://site1.com, http://site2.net, www.site3.org"
    >>> extract_urls(text)
    ['https://site1.com,', 'http://site2.net,', 'www.site3.org']

    Notes
    -----
    - Matches both HTTP and HTTPS protocols
    - Matches www. URLs without explicit protocol
    - Handles URL-encoded characters (%XX format)
    - URL extraction stops at whitespace; punctuation at end may be included
    - Does not validate that URLs are actually reachable

    See Also
    --------
    extract_emails : Extract email addresses from text
    extract_ip_addresses : Extract IP addresses from text
    """
    url_pattern = re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*|"
        r"www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*"
    )
    return url_pattern.findall(text)


def extract_hashtags(text: str) -> list[str]:
    """
    Extract hashtags from text (e.g., social media posts).

    This function identifies and extracts all hashtags from the input text.
    Hashtags are words or phrases prefixed with the '#' symbol, commonly used
    on social media platforms like Twitter, Instagram, and LinkedIn.

    Parameters
    ----------
    text : str
        The input text to search for hashtags. Typically social media content,
        but can be any text containing hashtag patterns.

    Returns
    -------
    list[str]
        A list of hashtags found in the text, including the '#' prefix.
        Returns an empty list if no hashtags are found. Hashtags are
        returned in the order they appear in the text.

    Examples
    --------
    Extract hashtags from a tweet:

    >>> extract_hashtags("Loving the new #Python features! #coding #programming")
    ['#Python', '#coding', '#programming']

    Extract hashtags with numbers:

    >>> extract_hashtags("Looking forward to #PyCon2024 and #AI4Good")
    ['#PyCon2024', '#AI4Good']

    Extract hashtags with underscores:

    >>> extract_hashtags("Check out #machine_learning and #deep_learning")
    ['#machine_learning', '#deep_learning']

    Return empty list when no hashtags present:

    >>> extract_hashtags("This text has no hashtags")
    []

    Notes
    -----
    - Hashtags must start with '#' followed by word characters (a-z, A-Z, 0-9, _)
    - The '#' symbol is included in the returned strings
    - Hashtags are case-sensitive in the output (preserves original case)
    - Does not handle Unicode hashtags or special characters beyond underscore

    See Also
    --------
    extract_mentions : Extract @mentions from text
    """
    hashtag_pattern = re.compile(r"#\w+")
    return hashtag_pattern.findall(text)


def extract_mentions(text: str) -> list[str]:
    """
    Extract @mentions from text (e.g., social media posts).

    This function identifies and extracts all user mentions from the input text.
    Mentions are usernames prefixed with the '@' symbol, commonly used on social
    media platforms like Twitter, Instagram, and GitHub to reference other users.

    Parameters
    ----------
    text : str
        The input text to search for mentions. Typically social media content,
        but can be any text containing mention patterns (e.g., code comments,
        documentation).

    Returns
    -------
    list[str]
        A list of mentions found in the text, including the '@' prefix.
        Returns an empty list if no mentions are found. Mentions are
        returned in the order they appear in the text.

    Examples
    --------
    Extract mentions from a tweet:

    >>> extract_mentions("Great talk by @elonmusk and @satloansek!")
    ['@elonmusk', '@satloansek']

    Extract mentions from GitHub-style text:

    >>> text = "Thanks @contributor1 and @reviewer_2 for the PR review!"
    >>> extract_mentions(text)
    ['@contributor1', '@reviewer_2']

    Extract mentions with numbers in usernames:

    >>> extract_mentions("Following @user123 and @dev2024")
    ['@user123', '@dev2024']

    Return empty list when no mentions present:

    >>> extract_mentions("This text mentions no one")
    []

    Notes
    -----
    - Mentions must start with '@' followed by word characters (a-z, A-Z, 0-9, _)
    - The '@' symbol is included in the returned strings
    - Mentions are case-sensitive in the output (preserves original case)
    - Email addresses are NOT matched (use extract_emails for that)
    - Does not validate that mentions correspond to real usernames

    See Also
    --------
    extract_hashtags : Extract #hashtags from text
    extract_emails : Extract email addresses from text
    """
    mention_pattern = re.compile(r"@\w+")
    return mention_pattern.findall(text)


def extract_ip_addresses(text: str) -> list[str]:
    """
    Extract IPv4 addresses from text.

    This function identifies and extracts all valid IPv4 addresses from the
    input text using regex pattern matching. It validates that each octet
    is within the valid range (0-255).

    Parameters
    ----------
    text : str
        The input text to search for IP addresses. Can be log files,
        configuration data, network documentation, or any text containing
        IP addresses.

    Returns
    -------
    list[str]
        A list of valid IPv4 addresses found in the text. Returns an empty
        list if no valid IP addresses are found. Addresses are returned in
        the order they appear in the text.

    Examples
    --------
    Extract IP addresses from log entries:

    >>> text = "Connection from 192.168.1.100 to 10.0.0.1 established"
    >>> extract_ip_addresses(text)
    ['192.168.1.100', '10.0.0.1']

    Extract localhost and public IP addresses:

    >>> text = "Server at 127.0.0.1, gateway 8.8.8.8"
    >>> extract_ip_addresses(text)
    ['127.0.0.1', '8.8.8.8']

    Handle edge case IP addresses:

    >>> extract_ip_addresses("Valid: 0.0.0.0, 255.255.255.255")
    ['0.0.0.0', '255.255.255.255']

    Invalid octets are not matched:

    >>> extract_ip_addresses("Invalid: 256.1.1.1 or 192.168.1.999")
    []

    Notes
    -----
    - Only matches IPv4 addresses (not IPv6)
    - Validates each octet is in range 0-255
    - Uses word boundaries to avoid partial matches
    - Does not validate that IPs are routable or reachable
    - Private, public, and special addresses (localhost, broadcast) are all matched

    See Also
    --------
    extract_urls : Extract URLs from text
    extract_emails : Extract email addresses from text
    """
    ipv4_pattern = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )
    return ipv4_pattern.findall(text)


# ===== Named Entity Recognition =====


def extract_named_entities(text: str, model_name: str = "en_core_web_sm") -> list[tuple[str, str]]:
    """
    Extract named entities from text using spaCy's NLP models.

    This function uses spaCy's named entity recognition (NER) to identify and
    classify entities in the text. It automatically loads (and installs if
    necessary) the specified spaCy model.

    Parameters
    ----------
    text : str
        The input text to analyze for named entities. Can be any length,
        though very long texts may be slower to process.
    model_name : str, optional
        The spaCy model to use for entity extraction. Default is "en_core_web_sm"
        (small English model). Other options include:
        - "en_core_web_md": Medium English model (better accuracy)
        - "en_core_web_lg": Large English model (best accuracy)
        - "en_core_web_trf": Transformer-based model (highest accuracy)
        - Language-specific models (e.g., "de_core_news_sm" for German)

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples where each tuple contains:
        - entity (str): The text of the named entity
        - label (str): The entity type/label (e.g., "PERSON", "ORG", "GPE")
        Returns an empty list if no entities are found.

    Examples
    --------
    Extract entities from a news-style sentence:

    >>> text = "Apple Inc. announced that CEO Tim Cook will visit Paris next week."
    >>> entities = extract_named_entities(text)
    >>> # Returns: [('Apple Inc.', 'ORG'), ('Tim Cook', 'PERSON'), ('Paris', 'GPE')]

    Extract entities mentioning dates and money:

    >>> text = "The contract worth $5 million was signed on January 15, 2024."
    >>> entities = extract_named_entities(text)
    >>> # Returns: [('$5 million', 'MONEY'), ('January 15, 2024', 'DATE')]

    Extract entities from historical text:

    >>> text = "Albert Einstein developed the theory of relativity in Germany."
    >>> entities = extract_named_entities(text)
    >>> # Returns: [('Albert Einstein', 'PERSON'), ('Germany', 'GPE')]

    Use a larger model for better accuracy:

    >>> text = "Microsoft acquired LinkedIn for $26.2 billion."
    >>> entities = extract_named_entities(text, model_name="en_core_web_lg")
    >>> # Returns: [('Microsoft', 'ORG'), ('LinkedIn', 'ORG'), ('$26.2 billion', 'MONEY')]

    Notes
    -----
    Common entity labels in spaCy English models:
    - PERSON: People, including fictional
    - ORG: Organizations, companies, agencies
    - GPE: Geopolitical entities (countries, cities, states)
    - LOC: Non-GPE locations (mountains, rivers)
    - DATE: Absolute or relative dates
    - TIME: Times smaller than a day
    - MONEY: Monetary values
    - PERCENT: Percentages
    - PRODUCT: Objects, vehicles, foods (not services)
    - EVENT: Named events (hurricanes, wars, sports events)
    - WORK_OF_ART: Titles of books, songs, etc.
    - LAW: Named documents made into laws
    - LANGUAGE: Named languages
    - NORP: Nationalities, religious/political groups

    The model will be automatically downloaded if not already installed.

    See Also
    --------
    extract_entities_by_type : Extract only specific entity types
    """
    nlp = ensure_spacy(model_name)
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def extract_entities_by_type(
    text: str, entity_types: list[str], model_name: str = "en_core_web_sm"
) -> dict[str, list[str]]:
    """
    Extract named entities of specific types from text.

    This function filters named entities by their type labels, returning only
    entities that match the specified types. Useful when you only need certain
    categories of entities (e.g., only people and organizations).

    Parameters
    ----------
    text : str
        The input text to analyze for named entities.
    entity_types : list[str]
        A list of entity type labels to extract. Common types include:
        - "PERSON": People, including fictional characters
        - "ORG": Organizations, companies, agencies
        - "GPE": Geopolitical entities (countries, cities, states)
        - "LOC": Non-GPE locations (mountains, rivers, regions)
        - "DATE": Absolute or relative dates
        - "MONEY": Monetary values
        - "PRODUCT": Objects, vehicles, foods
        Entity types are case-sensitive and must match spaCy's labels exactly.
    model_name : str, optional
        The spaCy model to use. Default is "en_core_web_sm".

    Returns
    -------
    dict[str, list[str]]
        A dictionary where keys are entity types and values are lists of
        entities of that type found in the text. Only includes types that
        were both requested AND found in the text.

    Examples
    --------
    Extract only person and organization names:

    >>> text = "Elon Musk founded SpaceX and Tesla in the United States."
    >>> result = extract_entities_by_type(text, ["PERSON", "ORG"])
    >>> # Returns: {'PERSON': ['Elon Musk'], 'ORG': ['SpaceX', 'Tesla']}

    Extract locations and dates from travel text:

    >>> text = "We visited Paris in June 2023 and Tokyo in December."
    >>> result = extract_entities_by_type(text, ["GPE", "DATE"])
    >>> # Returns: {'GPE': ['Paris', 'Tokyo'], 'DATE': ['June 2023', 'December']}

    Extract monetary values from financial text:

    >>> text = "The acquisition cost $2.5 billion, paid in March 2024."
    >>> result = extract_entities_by_type(text, ["MONEY", "DATE"])
    >>> # Returns: {'MONEY': ['$2.5 billion'], 'DATE': ['March 2024']}

    Handle case where requested types are not found:

    >>> text = "The quick brown fox jumps over the lazy dog."
    >>> result = extract_entities_by_type(text, ["PERSON", "ORG"])
    >>> # Returns: {} (empty dict, no matching entities)

    Notes
    -----
    - Entity types must match spaCy's label names exactly (case-sensitive)
    - If an entity type is not found in the text, it won't appear in the result
    - Multiple occurrences of the same entity are all included in the list
    - Uses the same model as extract_named_entities internally

    See Also
    --------
    extract_named_entities : Extract all named entities with their types
    """
    entities = extract_named_entities(text, model_name)
    result = defaultdict(list)

    for entity, entity_type in entities:
        if entity_type in entity_types:
            result[entity_type].append(entity)

    return dict(result)
