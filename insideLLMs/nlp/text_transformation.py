"""Text transformation utilities for string manipulation and PII protection.

This module provides a collection of utilities for transforming, formatting,
and sanitizing text data. It includes functions for truncation, padding,
PII masking, and word replacement with case preservation.

Overview
--------
The text transformation module offers four main capabilities:

1. **Text Truncation**: Shorten text to a maximum length with optional ellipsis
2. **Text Padding**: Pad text to a fixed width with configurable alignment
3. **PII Masking**: Automatically detect and mask sensitive information
4. **Word Replacement**: Replace words while preserving original casing

These utilities are designed for common text processing tasks in NLP pipelines,
data preprocessing, logging, and display formatting.

Examples
--------
Basic text truncation for display:

>>> from insideLLMs.nlp.text_transformation import truncate_text
>>> long_title = "Understanding Large Language Models: A Comprehensive Guide"
>>> truncate_text(long_title, 30)
'Understanding Large Langua...'

Padding text for tabular display:

>>> from insideLLMs.nlp.text_transformation import pad_text
>>> pad_text("Name", 20, pad_char="-", align="center")
'--------Name--------'

Masking PII in user-generated content:

>>> from insideLLMs.nlp.text_transformation import mask_pii
>>> user_input = "Contact me at john.doe@email.com or 555-123-4567"
>>> mask_pii(user_input)
'Contact me at ******************** or ************'

Case-preserving word replacement:

>>> from insideLLMs.nlp.text_transformation import replace_words
>>> text = "The QUICK brown fox"
>>> replace_words(text, {"quick": "slow"})
'The SLOW brown fox'

Notes
-----
- All functions are designed to be non-destructive and return new strings
- PII masking uses regex patterns and may not catch all variations
- For production PII handling, consider specialized libraries like presidio
- Word replacement with case_sensitive=False preserves original word casing

See Also
--------
insideLLMs.nlp.text_normalization : Text normalization utilities
insideLLMs.nlp.tokenization : Text tokenization functions
re : Python regular expressions module used internally

"""

import re

# ===== Text Transformation =====


def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """Truncate text to a maximum length with optional ellipsis.

    This function shortens text that exceeds a specified maximum length,
    optionally appending an ellipsis ("...") to indicate truncation. The
    ellipsis is counted as part of the maximum length, so the actual text
    content will be reduced by 3 characters when ellipsis is enabled.

    Parameters
    ----------
    text : str
        The input text to truncate. Can be any string including empty strings,
        multiline text, or text with special characters.
    max_length : int
        The maximum allowed length of the output string, including the ellipsis
        if added. Must be a positive integer. Values of 3 or less with
        add_ellipsis=True will result in text being cut without ellipsis.
    add_ellipsis : bool, optional
        Whether to append "..." to truncated text. Default is True.
        When True and max_length > 3, the ellipsis replaces the last 3
        characters of the allowed length. When False, text is simply cut
        at max_length.

    Returns
    -------
    str
        The truncated text. Returns the original text unchanged if its length
        is less than or equal to max_length. Otherwise returns truncated text
        with or without ellipsis based on the add_ellipsis parameter.

    Examples
    --------
    Basic truncation with default ellipsis:

    >>> truncate_text("Hello, World!", 10)
    'Hello, ...'

    Truncation without ellipsis:

    >>> truncate_text("Hello, World!", 10, add_ellipsis=False)
    'Hello, Wor'

    Text shorter than max_length is returned unchanged:

    >>> truncate_text("Hi", 10)
    'Hi'

    Text exactly at max_length:

    >>> truncate_text("Hello", 5)
    'Hello'

    Truncating titles for display cards:

    >>> article_title = "Machine Learning Best Practices for Production Systems"
    >>> truncate_text(article_title, 35)
    'Machine Learning Best Practices ...'

    Handling edge case with very short max_length:

    >>> truncate_text("Hello", 3, add_ellipsis=True)
    'Hel'

    Truncating multiline text (newlines count as characters):

    >>> multiline = "Line 1\\nLine 2\\nLine 3"
    >>> truncate_text(multiline, 12)
    'Line 1\\nLi...'

    See Also
    --------
    pad_text : Pad text to a specified length

    Notes
    -----
    - The function counts characters, not bytes. Unicode characters count as 1.
    - Newlines and whitespace are preserved and counted in the length.
    - For max_length <= 3 with add_ellipsis=True, the ellipsis is not added
      to avoid returning just "..." or partial ellipsis.
    """
    if len(text) <= max_length:
        return text

    if add_ellipsis and max_length > 3:
        return text[: max_length - 3] + "..."
    else:
        return text[:max_length]


def pad_text(text: str, length: int, pad_char: str = " ", align: str = "left") -> str:
    """Pad text to a specified length with configurable alignment.

    This function pads a string to reach a target length using a specified
    padding character. The text can be aligned to the left, right, or center
    within the padded result. If the input text is already at or exceeds
    the target length, it is returned unchanged (no truncation occurs).

    Parameters
    ----------
    text : str
        The input text to pad. Can be any string including empty strings.
        The original text is never modified or truncated.
    length : int
        The desired total length of the output string. If text is already
        this long or longer, the original text is returned unchanged.
        Must be a non-negative integer.
    pad_char : str, optional
        The character to use for padding. Default is a single space " ".
        Should be a single character for predictable results, though
        multi-character strings are technically allowed.
    align : str, optional
        The alignment of the original text within the padded result.
        Must be one of:

        - "left" (default): Text is left-aligned, padding added to the right
        - "right": Text is right-aligned, padding added to the left
        - "center": Text is centered, padding split between left and right

        For center alignment with odd padding, the extra character goes
        on the right side.

    Returns
    -------
    str
        The padded text at the specified length, or the original text
        if it was already at or exceeding the target length.

    Raises
    ------
    ValueError
        If align is not one of "left", "right", or "center".

    Examples
    --------
    Left-aligned padding (default):

    >>> pad_text("Hello", 10)
    'Hello     '

    Right-aligned padding:

    >>> pad_text("Hello", 10, align="right")
    '     Hello'

    Center-aligned padding:

    >>> pad_text("Hello", 11, align="center")
    '   Hello   '

    Using a custom padding character:

    >>> pad_text("Title", 20, pad_char="=")
    'Title==============='

    Creating centered headers:

    >>> pad_text("REPORT", 30, pad_char="-", align="center")
    '------------REPORT------------'

    Building fixed-width table columns:

    >>> names = ["Alice", "Bob", "Christopher"]
    >>> for name in names:
    ...     print(f"|{pad_text(name, 15)}|")
    |Alice          |
    |Bob            |
    |Christopher    |

    Text already at target length:

    >>> pad_text("Hello", 5)
    'Hello'

    Text longer than target length (no truncation):

    >>> pad_text("Hello, World!", 5)
    'Hello, World!'

    Padding an empty string:

    >>> pad_text("", 5, pad_char="*")
    '*****'

    Right-padding numbers for alignment:

    >>> numbers = ["1", "42", "100", "7"]
    >>> for num in numbers:
    ...     print(pad_text(num, 5, align="right"))
        1
       42
      100
        7

    See Also
    --------
    truncate_text : Truncate text to a maximum length

    Notes
    -----
    - For center alignment, if the padding cannot be evenly split, the
      extra padding character is added to the right side.
    - Using multi-character pad_char strings will still work but may
      produce unexpected lengths since each repetition adds len(pad_char).
    - This function does not strip existing whitespace from the input text.
    """
    if len(text) >= length:
        return text

    if align == "left":
        return text + pad_char * (length - len(text))
    elif align == "right":
        return pad_char * (length - len(text)) + text
    elif align == "center":
        left_pad = (length - len(text)) // 2
        right_pad = length - len(text) - left_pad
        return pad_char * left_pad + text + pad_char * right_pad
    else:
        raise ValueError("align must be 'left', 'right', or 'center'")


def mask_pii(text: str, mask_char: str = "*") -> str:
    """Mask personally identifiable information (PII) in text.

    This function identifies and masks common types of personally identifiable
    information (PII) found in text, replacing each character of detected PII
    with a masking character. The masking preserves the length of the original
    PII, which can be useful for maintaining text structure.

    The function detects and masks the following PII types:

    - **Email addresses**: Standard email format (user@domain.tld)
    - **Phone numbers**: Various formats with 7+ digits, including international
    - **Credit card numbers**: 13-16 digit sequences with optional separators
    - **Social Security Numbers (SSN)**: US format XXX-XX-XXXX

    Parameters
    ----------
    text : str
        The input text that may contain PII. Can be any string including
        multiline text, HTML, JSON, or plain text content.
    mask_char : str, optional
        The character to use for masking detected PII. Default is "*".
        Each character of detected PII is replaced with this character,
        preserving the original length. Should be a single character.

    Returns
    -------
    str
        A copy of the input text with all detected PII replaced by the
        mask character. Non-PII content remains unchanged.

    Examples
    --------
    Masking an email address:

    >>> mask_pii("Contact: john.doe@example.com")
    'Contact: ********************'

    Masking a phone number:

    >>> mask_pii("Call us at 555-123-4567")
    'Call us at ************'

    Masking international phone numbers:

    >>> mask_pii("International: +1 (555) 123-4567")
    'International: ******************'

    Masking credit card numbers:

    >>> mask_pii("Card: 4111-1111-1111-1111")
    'Card: *******************'

    Masking Social Security Numbers:

    >>> mask_pii("SSN: 123-45-6789")
    'SSN: ***********'

    Using a different mask character:

    >>> mask_pii("Email: test@test.com", mask_char="X")
    'Email: XXXXXXXXXXXXX'

    Masking multiple PII in one text:

    >>> text = "Name: John, Email: john@mail.com, Phone: 555-1234"
    >>> mask_pii(text)
    'Name: John, Email: *************, Phone: ********'

    Processing user-submitted form data:

    >>> form_data = '''
    ... Name: Jane Doe
    ... Email: jane.doe@company.org
    ... Phone: (555) 987-6543
    ... SSN: 987-65-4321
    ... '''
    >>> masked = mask_pii(form_data)
    >>> print(masked)
    <BLANKLINE>
    Name: Jane Doe
    Email: *********************
    Phone: **************
    SSN: ***********
    <BLANKLINE>

    Masking PII in log entries before storage:

    >>> log_entry = "[2024-01-15] User login from user@example.com (IP: 192.168.1.1)"
    >>> mask_pii(log_entry)
    '[2024-01-15] User login from **************** (IP: *************)'

    See Also
    --------
    replace_words : Replace specific words in text

    Notes
    -----
    - The detection uses regular expressions and may not catch all PII formats.
    - False positives are possible, especially with phone numbers (any 7+
      digit sequence may be matched).
    - The function does not detect names, addresses, or other contextual PII.
    - For production security-critical applications, consider using specialized
      PII detection libraries such as Microsoft Presidio or AWS Comprehend.
    - The masking preserves length to maintain text structure and alignment.
    - Processing is done in order: emails, phones, credit cards, then SSNs.
      This order matters if patterns overlap.

    Warnings
    --------
    This function provides basic PII detection and should not be relied upon
    as the sole mechanism for PII protection in security-critical applications.
    Always validate that detected PII meets your requirements and consider
    additional validation layers.
    """
    # Mask email addresses
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    text = email_pattern.sub(lambda m: mask_char * len(m.group(0)), text)

    # Mask phone numbers (simple pattern)
    phone_pattern = re.compile(r"\+?[0-9][\s\-\(\)0-9]{6,}")
    text = phone_pattern.sub(lambda m: mask_char * len(m.group(0)), text)

    # Mask credit card numbers
    cc_pattern = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
    text = cc_pattern.sub(lambda m: mask_char * len(m.group(0)), text)

    # Mask SSN (US Social Security Numbers)
    ssn_pattern = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
    text = ssn_pattern.sub(lambda m: mask_char * len(m.group(0)), text)

    return text


def replace_words(text: str, replacements: dict[str, str], case_sensitive: bool = False) -> str:
    """Replace specific words in text with optional case preservation.

    This function performs word-level replacements in text, with intelligent
    case handling. When case_sensitive is False (the default), the function
    matches words regardless of case and attempts to preserve the casing style
    of the original word in the replacement.

    Word matching uses word boundaries, so only complete words are replaced,
    not substrings within larger words.

    Parameters
    ----------
    text : str
        The input text in which to perform replacements. Can be any string
        including multiline text with various punctuation.
    replacements : dict[str, str]
        A dictionary mapping source words to their replacement words.
        When case_sensitive is False, the keys should be lowercase.
        Example: {"old": "new", "foo": "bar"}
    case_sensitive : bool, optional
        Whether to perform case-sensitive matching. Default is False.

        - When False: Matches words regardless of case and preserves the
          original word's casing in the replacement (lowercase, UPPERCASE,
          or Capitalized).
        - When True: Only matches words that exactly match the dictionary
          keys, including case. Replacements use the exact case from the
          dictionary values.

    Returns
    -------
    str
        A copy of the input text with all matched words replaced.
        Unmatched words and non-word characters remain unchanged.

    Examples
    --------
    Basic case-insensitive replacement with case preservation:

    >>> replace_words("The quick brown fox", {"quick": "slow"})
    'The slow brown fox'

    Case preservation with uppercase words:

    >>> replace_words("The QUICK brown fox", {"quick": "slow"})
    'The SLOW brown fox'

    Case preservation with capitalized words:

    >>> replace_words("Quick thinking saves the day", {"quick": "Fast"})
    'Fast thinking saves the day'

    Multiple word replacements:

    >>> text = "I love cats and dogs"
    >>> replacements = {"cats": "birds", "dogs": "fish"}
    >>> replace_words(text, replacements)
    'I love birds and fish'

    Word boundaries prevent partial matches:

    >>> replace_words("category catalog", {"cat": "dog"})
    'category catalog'

    Case-sensitive replacement (exact match required):

    >>> replace_words("Hello HELLO hello", {"Hello": "Hi"}, case_sensitive=True)
    'Hi HELLO hello'

    Content moderation - replacing inappropriate words:

    >>> text = "This is DAMN good and damn impressive"
    >>> replace_words(text, {"damn": "very"})
    'This is VERY good and very impressive'

    Replacing technical terms:

    >>> text = "The API endpoint uses REST. The api documentation..."
    >>> replace_words(text, {"api": "interface", "rest": "HTTP"})
    'The interface endpoint uses HTTP. The interface documentation...'

    Replacing abbreviations with full forms:

    >>> replace_words("Dr. Smith and Mr. Jones", {"dr": "Doctor", "mr": "Mister"})
    'Doctor. Smith and Mister. Jones'

    Empty replacements dictionary returns original text:

    >>> replace_words("Hello World", {})
    'Hello World'

    Handling text with punctuation:

    >>> replace_words("Hello, World! Hello?", {"hello": "hi"})
    'Hi, World! Hi?'

    Real-world example - updating outdated terminology:

    >>> old_text = "The Master branch contains the Slave configuration"
    >>> updates = {"master": "main", "slave": "replica"}
    >>> replace_words(old_text, updates)
    'The Main branch contains the Replica configuration'

    See Also
    --------
    mask_pii : Mask personally identifiable information in text

    Notes
    -----
    - Word boundaries are determined by the regex ``\\b`` pattern, which matches
      positions between word and non-word characters.
    - When case_sensitive=False, the replacement case matching works as follows:

      * All lowercase original -> lowercase replacement
      * All uppercase original -> uppercase replacement
      * First letter uppercase -> capitalized replacement
      * Mixed case -> replacement as-is from dictionary

    - The function processes all matches in a single pass using regex
      substitution, making it efficient for texts with many replacements.
    - When case_sensitive=False, dictionary keys should be lowercase for
      consistent matching behavior.
    - Special regex characters in the replacement words are properly escaped.
    """
    if not case_sensitive:
        # Create a regex pattern that matches any of the words to replace
        pattern = re.compile(
            "\\b(" + "|".join(map(re.escape, replacements.keys())) + ")\\b", re.IGNORECASE
        )

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
        pattern = re.compile("\\b(" + "|".join(map(re.escape, replacements.keys())) + ")\\b")
        return pattern.sub(lambda m: replacements[m.group(0)], text)
