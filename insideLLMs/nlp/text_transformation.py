import re
from typing import Dict

# ===== Text Transformation =====

def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """Truncate text to a maximum length.

    Args:
        text: Input text
        max_length: Maximum length of the output text
        add_ellipsis: Whether to add an ellipsis (...) if text is truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    if add_ellipsis and max_length > 3:
        return text[:max_length-3] + '...'
    else:
        return text[:max_length]

def pad_text(text: str, length: int, pad_char: str = ' ', align: str = 'left') -> str:
    """Pad text to a specified length.

    Args:
        text: Input text
        length: Desired length of the output text
        pad_char: Character to use for padding
        align: Alignment of the text ('left', 'right', 'center')

    Returns:
        Padded text
    """
    if len(text) >= length:
        return text

    if align == 'left':
        return text + pad_char * (length - len(text))
    elif align == 'right':
        return pad_char * (length - len(text)) + text
    elif align == 'center':
        left_pad = (length - len(text)) // 2
        right_pad = length - len(text) - left_pad
        return pad_char * left_pad + text + pad_char * right_pad
    else:
        raise ValueError("align must be 'left', 'right', or 'center'")

def mask_pii(text: str, mask_char: str = '*') -> str:
    """Mask personally identifiable information (PII) in text.

    Args:
        text: Input text
        mask_char: Character to use for masking

    Returns:
        Text with masked PII
    """
    # Mask email addresses
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    text = email_pattern.sub(lambda m: mask_char * len(m.group(0)), text)

    # Mask phone numbers (simple pattern)
    phone_pattern = re.compile(r'\+?[0-9][\s\-\(\)0-9]{6,}')
    text = phone_pattern.sub(lambda m: mask_char * len(m.group(0)), text)

    # Mask credit card numbers
    cc_pattern = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
    text = cc_pattern.sub(lambda m: mask_char * len(m.group(0)), text)

    # Mask SSN (US Social Security Numbers)
    ssn_pattern = re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b')
    text = ssn_pattern.sub(lambda m: mask_char * len(m.group(0)), text)

    return text

def replace_words(text: str, replacements: Dict[str, str], case_sensitive: bool = False) -> str:
    """Replace specific words in text.

    Args:
        text: Input text
        replacements: Dictionary mapping words to their replacements
        case_sensitive: Whether to perform case-sensitive replacement

    Returns:
        Text with replaced words
    """
    if not case_sensitive:
        # Create a regex pattern that matches any of the words to replace
        pattern = re.compile('\\b(' + '|'.join(map(re.escape, replacements.keys())) + ')\\b', re.IGNORECASE)

        # Function to get the replacement with proper case
        def replace(match):
            word = match.group(0)
            replacement = replacements.get(word.lower(), word)

            # Preserve case of the original word
            if word.islower():
                return replacement.lower()
            elif word.isupper():
                return replacement.upper()
            elif word[0].isupper():
                return replacement.capitalize()
            else:
                return replacement

        return pattern.sub(replace, text)
    else:
        # For case-sensitive replacement, use a simpler approach
        pattern = re.compile('\\b(' + '|'.join(map(re.escape, replacements.keys())) + ')\\b')
        return pattern.sub(lambda m: replacements[m.group(0)], text)
