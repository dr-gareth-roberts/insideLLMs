import re
from collections import defaultdict

from insideLLMs.nlp.dependencies import ensure_spacy

# ===== Dependency Management =====


def check_spacy(model_name: str = "en_core_web_sm"):
    """Ensure spaCy and the requested model are available."""
    return ensure_spacy(model_name)


# ===== Pattern Matching and Extraction =====


def extract_emails(text: str) -> list[str]:
    """Extract email addresses from text."""
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    return email_pattern.findall(text)


def extract_phone_numbers(text: str, country: str = "us") -> list[str]:
    """Extract phone numbers from text."""
    patterns = {
        "us": r"(?:\+?1[-\s]?)?(?:\(?[0-9]{3}\)?[-\s]?)?[0-9]{3}[-\s]?[0-9]{4}",
        "uk": r"(?:\+?44[-\s]?)?(?:\(?0[0-9]{1,4}\)?[-\s]?)?[0-9]{3,4}[-\s]?[0-9]{3,4}",
        "international": r"\+?[0-9]{1,3}[-\s]?[0-9]{1,4}[-\s]?[0-9]{1,4}[-\s]?[0-9]{1,9}",
    }

    pattern = patterns.get(country.lower(), patterns["international"])
    phone_pattern = re.compile(pattern)
    return phone_pattern.findall(text)


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text."""
    url_pattern = re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*|"
        r"www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*"
    )
    return url_pattern.findall(text)


def extract_hashtags(text: str) -> list[str]:
    """Extract hashtags from text."""
    hashtag_pattern = re.compile(r"#\w+")
    return hashtag_pattern.findall(text)


def extract_mentions(text: str) -> list[str]:
    """Extract mentions from text."""
    mention_pattern = re.compile(r"@\w+")
    return mention_pattern.findall(text)


def extract_ip_addresses(text: str) -> list[str]:
    """Extract IP addresses from text."""
    ipv4_pattern = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )
    return ipv4_pattern.findall(text)


# ===== Named Entity Recognition =====


def extract_named_entities(text: str, model_name: str = "en_core_web_sm") -> list[tuple[str, str]]:
    """Extract named entities from text using spaCy."""
    nlp = check_spacy(model_name)
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def extract_entities_by_type(
    text: str, entity_types: list[str], model_name: str = "en_core_web_sm"
) -> dict[str, list[str]]:
    """Extract named entities of specific types from text."""
    entities = extract_named_entities(text, model_name)
    result = defaultdict(list)

    for entity, entity_type in entities:
        if entity_type in entity_types:
            result[entity_type].append(entity)

    return dict(result)
