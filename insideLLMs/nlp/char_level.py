import re
from collections import Counter

# ===== Character-Level Operations =====


def get_char_ngrams(text: str, n: int = 2) -> list[str]:
    """Generate character n-grams from text.

    Args:
        text: Input text
        n: Size of n-grams

    Returns:
        List of character n-grams
    """
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def get_char_frequency(text: str) -> dict[str, int]:
    """Get character frequencies in text.

    Args:
        text: Input text

    Returns:
        Dictionary mapping characters to their frequencies
    """
    return dict(Counter(text))


def to_uppercase(text: str) -> str:
    """Convert text to uppercase.

    Args:
        text: Input text

    Returns:
        Uppercase text
    """
    return text.upper()


def to_titlecase(text: str) -> str:
    """Convert text to title case.

    Args:
        text: Input text

    Returns:
        Title case text
    """
    return text.title()


def to_camelcase(text: str) -> str:
    """Convert text to camel case.

    Args:
        text: Input text with spaces or underscores

    Returns:
        Camel case text
    """
    # Replace underscores with spaces, then split by spaces
    words = text.replace("_", " ").split()
    if not words:
        return ""

    # First word lowercase, rest title case
    return words[0].lower() + "".join(word.title() for word in words[1:])


def to_snakecase(text: str) -> str:
    """Convert text to snake case.

    Args:
        text: Input text

    Returns:
        Snake case text
    """
    # Replace spaces and hyphens with underscores
    text = re.sub(r"[ -]", "_", text)
    # Handle camel case
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9_]", "", text).lower()
    # Replace multiple underscores with a single one
    text = re.sub(r"_+", "_", text)
    return text
