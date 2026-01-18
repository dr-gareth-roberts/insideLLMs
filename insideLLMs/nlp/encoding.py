import base64
import html
import urllib.parse

# ===== Text Encoding/Decoding =====


def encode_base64(text: str) -> str:
    """Encode text to Base64.

    Args:
        text: Input text

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def decode_base64(encoded: str) -> str:
    """Decode Base64 to text.

    Args:
        encoded: Base64 encoded string

    Returns:
        Decoded text
    """
    return base64.b64decode(encoded.encode("utf-8")).decode("utf-8")


def url_encode(text: str) -> str:
    """URL encode text.

    Args:
        text: Input text

    Returns:
        URL encoded string
    """
    return urllib.parse.quote(text)


def url_decode(encoded: str) -> str:
    """URL decode text.

    Args:
        encoded: URL encoded string

    Returns:
        Decoded text
    """
    return urllib.parse.unquote(encoded)


def html_encode(text: str) -> str:
    """HTML encode text (convert special characters to HTML entities).

    Args:
        text: Input text

    Returns:
        HTML encoded string
    """
    return html.escape(text)


def html_decode(encoded: str) -> str:
    """HTML decode text (convert HTML entities to characters).

    Args:
        encoded: HTML encoded string

    Returns:
        Decoded text
    """
    return html.unescape(encoded)
