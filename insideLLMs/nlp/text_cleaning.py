"""Text cleaning and normalization utilities.

This module provides comprehensive text preprocessing functions for cleaning
and normalizing text data before NLP processing or model input. It handles
common noise sources like HTML, URLs, emojis, and inconsistent formatting.

Key Functions:
    - remove_html_tags: Strip HTML markup
    - remove_urls: Remove URLs/links
    - remove_punctuation: Strip punctuation marks
    - remove_emojis: Remove emoji characters
    - remove_numbers: Strip numeric characters
    - normalize_whitespace: Fix spacing issues
    - normalize_unicode: Normalize Unicode forms
    - normalize_contractions: Expand English contractions
    - replace_repeated_chars: Reduce character repetition
    - clean_text: All-in-one cleaning with options

Example - Quick Cleaning:
    >>> from insideLLMs.nlp.text_cleaning import clean_text
    >>>
    >>> dirty = "<p>Check out https://example.com!! Sooo cool 😎</p>"
    >>> clean = clean_text(dirty)
    >>> print(clean)
    'check out sooo cool'

Example - Selective Cleaning:
    >>> from insideLLMs.nlp.text_cleaning import remove_html_tags, remove_urls
    >>>
    >>> text = "<a href='url'>Visit http://example.com</a>"
    >>> text = remove_html_tags(text)
    >>> text = remove_urls(text)
    >>> print(text)
    'Visit '

Example - Full Pipeline:
    >>> from insideLLMs.nlp.text_cleaning import (
    ...     clean_text, normalize_contractions
    ... )
    >>>
    >>> text = "I can't believe it's not butter!!!"
    >>> cleaned = clean_text(
    ...     normalize_contractions(text),
    ...     remove_punct=True,
    ...     replace_repeated=True
    ... )
    >>> print(cleaned)
    'i can not believe it is not butter'

See Also:
    - insideLLMs.nlp.tokenization: Tokenization functions
    - insideLLMs.nlp.text_transformation: Additional transformations
"""

import re
import string
import unicodedata
from typing import Optional

# ===== Text Cleaning and Normalization =====


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text.

    Strips all HTML/XML tags while preserving the text content between them.
    Does not decode HTML entities (use normalize_unicode for that).

    Args:
        text: Input text with potential HTML tags.

    Returns:
        Text with HTML tags removed.

    Example - Basic Usage:
        >>> remove_html_tags("<p>Hello <b>World</b></p>")
        'Hello World'

    Example - Self-Closing Tags:
        >>> remove_html_tags("Line 1<br/>Line 2")
        'Line 1Line 2'

    Example - With Attributes:
        >>> remove_html_tags('<a href="url" class="link">Click here</a>')
        'Click here'

    Example - Nested Tags:
        >>> remove_html_tags("<div><span>Nested</span> text</div>")
        'Nested text'

    Example - Empty Tags:
        >>> remove_html_tags("<div></div>text<span></span>")
        'text'

    Note:
        This uses a simple regex pattern and may not handle malformed HTML
        or edge cases like tags spanning multiple lines. For complex HTML,
        consider using a proper HTML parser like BeautifulSoup.
    """
    html_pattern = re.compile("<.*?>")
    return html_pattern.sub("", text)


def remove_urls(text: str) -> str:
    """Remove URLs from text.

    Removes both http/https URLs and www-prefixed URLs, including any
    trailing path or query parameters.

    Args:
        text: Input text with potential URLs.

    Returns:
        Text with URLs removed.

    Example - HTTP URLs:
        >>> remove_urls("Visit https://example.com for more")
        'Visit  for more'

    Example - WWW URLs:
        >>> remove_urls("Go to www.example.com/page?id=1")
        'Go to '

    Example - Multiple URLs:
        >>> remove_urls("Link 1: http://a.com Link 2: http://b.com")
        'Link 1:  Link 2: '

    Example - Mixed Content:
        >>> remove_urls("Email me@example.com or visit http://example.com")
        'Email me@example.com or visit '

    Note:
        The pattern matches URLs starting with http://, https://, or www.
        It does not match email addresses or other URL-like patterns.
    """
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation from text.

    Removes all characters defined in string.punctuation:
    !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~

    Args:
        text: Input text with punctuation.

    Returns:
        Text with punctuation removed.

    Example - Basic Usage:
        >>> remove_punctuation("Hello, World!")
        'Hello World'

    Example - All Punctuation:
        >>> remove_punctuation("a!b@c#d$e%f")
        'abcdef'

    Example - Preserves Numbers:
        >>> remove_punctuation("Price: $19.99")
        'Price 1999'

    Example - Apostrophes:
        >>> remove_punctuation("It's John's book")
        'Its Johns book'

    Note:
        This removes ALL punctuation including apostrophes in contractions.
        For preserving contractions, use normalize_contractions first.
    """
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    Collapses multiple whitespace characters (spaces, tabs, newlines)
    into single spaces and trims leading/trailing whitespace.

    Args:
        text: Input text with irregular whitespace.

    Returns:
        Text with normalized whitespace (single spaces, trimmed).

    Example - Multiple Spaces:
        >>> normalize_whitespace("Hello    World")
        'Hello World'

    Example - Tabs and Newlines:
        >>> normalize_whitespace("Line1\\n\\nLine2\\tTab")
        'Line1 Line2 Tab'

    Example - Leading/Trailing:
        >>> normalize_whitespace("   text   ")
        'text'

    Example - Mixed Whitespace:
        >>> normalize_whitespace("  a   b\\t\\tc\\n\\nd  ")
        'a b c d'
    """
    return " ".join(text.split())


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize Unicode characters in text.

    Applies Unicode normalization to convert equivalent representations
    of characters to a canonical form. Useful for handling text from
    different sources with inconsistent encoding.

    Args:
        text: Input text with Unicode characters.
        form: Unicode normalization form. Options:
            - 'NFC': Canonical Decomposition followed by Canonical Composition
            - 'NFD': Canonical Decomposition
            - 'NFKC': Compatibility Decomposition followed by Canonical Composition (default)
            - 'NFKD': Compatibility Decomposition

    Returns:
        Text with normalized Unicode characters.

    Example - Compatibility Normalization:
        >>> # Full-width characters -> ASCII
        >>> normalize_unicode("Ｈｅｌｌｏ")  # Full-width
        'Hello'

    Example - Ligatures:
        >>> normalize_unicode("ﬁle")  # fi ligature
        'file'

    Example - Superscripts:
        >>> normalize_unicode("x²")  # Superscript 2
        'x2'

    Example - Accented Characters (NFC):
        >>> # Composed vs decomposed é
        >>> len(normalize_unicode("café", "NFC"))
        4
        >>> len(normalize_unicode("café", "NFD"))  # Decomposed
        5

    Example - Different Forms:
        >>> text = "ﬁle²"
        >>> normalize_unicode(text, "NFKC")
        'file2'
        >>> normalize_unicode(text, "NFC")  # Preserves compatibility chars
        'ﬁle²'

    Note:
        NFKC is recommended for most NLP tasks as it converts compatibility
        characters to their standard equivalents. Use NFC for preserving
        visual representation while normalizing composed characters.
    """
    return unicodedata.normalize(form, text)


def remove_emojis(text: str) -> str:
    """Remove emojis from text.

    Removes emoji characters from various Unicode blocks including emoticons,
    symbols, transport symbols, and other pictographic characters.

    Args:
        text: Input text with potential emojis.

    Returns:
        Text with emojis removed.

    Example - Basic Emojis:
        >>> remove_emojis("Hello 😀 World 🌍")
        'Hello  World '

    Example - Multiple Emojis:
        >>> remove_emojis("Great job! 👍🎉🔥")
        'Great job! '

    Example - Mixed Content:
        >>> remove_emojis("Meeting at 3pm 📅 Location: NYC 🗽")
        'Meeting at 3pm  Location: NYC '

    Example - Text Without Emojis:
        >>> remove_emojis("Plain text stays unchanged")
        'Plain text stays unchanged'

    Note:
        This pattern covers most common emojis but may not catch every
        Unicode emoji as new ones are added regularly. The pattern includes
        emoticons, symbols, transport symbols, alchemical symbols, and more.
    """
    # This pattern matches most emoji characters
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f700-\U0001f77f"  # alchemical symbols
        "\U0001f780-\U0001f7ff"  # Geometric Shapes
        "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
        "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
        "\U0001fa00-\U0001fa6f"  # Chess Symbols
        "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027b0"  # Dingbats
        "\U000024c2-\U0000257f"  # Enclosed characters
        "\U00002600-\U000026ff"  # Miscellaneous Symbols
        "\U00002700-\U000027bf"  # Dingbats
        "\U0000fe00-\U0000fe0f"  # Variation Selectors
        "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
        "\U00002b50"  # Star
        "\U00002b55"  # Circle
        "]",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def remove_numbers(text: str) -> str:
    """Remove numeric characters from text.

    Removes all sequences of digits (0-9) from the input text. This is useful
    for NLP tasks where numerical values are noise or when you want to focus
    solely on alphabetic content. Note that this removes entire number sequences,
    not individual digits within words.

    Args:
        text: Input text containing numeric characters to be removed.

    Returns:
        Text with all digit sequences removed. Surrounding text and whitespace
        are preserved, which may result in extra spaces where numbers were.

    Example - Basic Usage:
        >>> remove_numbers("I have 3 apples and 5 oranges")
        'I have  apples and  oranges'

    Example - Prices and Quantities:
        >>> remove_numbers("Price: $19.99 for 100 units")
        'Price: $. for  units'

    Example - Dates and Times:
        >>> remove_numbers("Meeting on 2024-01-15 at 10:30am")
        'Meeting on -- at :am'

    Example - Mixed Alphanumeric:
        >>> remove_numbers("Order ID: ABC123XYZ")
        'Order ID: ABCXYZ'

    Example - Text Without Numbers:
        >>> remove_numbers("No numbers here")
        'No numbers here'

    Note:
        This function removes ONLY digit characters (0-9). It does not affect:
        - Decimal points (use remove_punctuation for those)
        - Currency symbols
        - Roman numerals (these are letters)
        - Written numbers like "three" or "twenty"

        Consider using normalize_whitespace() after this function to clean up
        any double spaces left behind from removed numbers.
    """
    return re.sub(r"\d+", "", text)


def normalize_contractions(text: str) -> str:
    """Expand English contractions to their full forms.

    Converts common English contractions (e.g., "don't", "I'm", "they've")
    to their expanded equivalents (e.g., "do not", "I am", "they have").
    This is useful for text normalization before NLP tasks, improving
    consistency and potentially helping with tokenization.

    The function handles over 80 common contractions including:
    - Negative contractions (don't, can't, won't, etc.)
    - Pronoun contractions (I'm, you're, he's, etc.)
    - Have/would contractions (I've, she'd, they'll, etc.)
    - Informal contractions (y'all, 'cause, ain't, etc.)

    Args:
        text: Input text containing English contractions to be expanded.
            The matching is case-insensitive but the output preserves
            lowercase for the expanded forms.

    Returns:
        Text with all recognized contractions replaced by their expanded
        forms. Unrecognized contractions are left unchanged.

    Example - Basic Contractions:
        >>> normalize_contractions("I can't believe it's happening")
        'I cannot believe it is happening'

    Example - Negative Contractions:
        >>> normalize_contractions("They don't know we aren't coming")
        'They do not know we are not coming'

    Example - Pronoun Contractions:
        >>> normalize_contractions("She's happy and they're excited")
        'She is happy and they are excited'

    Example - Informal Speech:
        >>> normalize_contractions("Y'all shouldn't've done that")
        'You all should not have done that'

    Example - Mixed Case (case-insensitive matching):
        >>> normalize_contractions("I CAN'T and WON'T do it")
        'I cannot and will not do it'

    Example - Unrecognized Patterns:
        >>> normalize_contractions("The cat's toy is broken")
        "The cat's toy is broken"

    Note:
        - Possessive apostrophes (e.g., "John's book") are NOT expanded
          because they are not contractions.
        - Some contractions like "he's" could mean "he is" OR "he has"
          depending on context. This function always uses the more common
          interpretation (e.g., "he is").
        - The expanded forms use lowercase (e.g., "cannot" not "CANNOT")
          regardless of input case. Use str.title() or other case methods
          if you need to preserve capitalization.
        - For best results, run this BEFORE remove_punctuation() to ensure
          the apostrophes needed for contraction matching are present.
    """
    # Dictionary of common contractions
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
    }

    # Create a regular expression pattern for contractions
    pattern = re.compile(r"\b(" + "|".join(contractions.keys()) + r")\b", re.IGNORECASE)

    # Function to replace contractions
    def replace(match):
        word = match.group(0)
        return contractions.get(word.lower(), word)

    return pattern.sub(replace, text)


def replace_repeated_chars(text: str, threshold: int = 2) -> str:
    """Reduce excessive character repetition to a maximum count.

    Finds sequences where a single character is repeated more than the
    threshold number of times and reduces them to exactly the threshold
    count. This is useful for normalizing informal text, social media
    content, or text with emphatic repetition (e.g., "sooooo" -> "soo").

    The function works on any character including letters, punctuation,
    and whitespace. It processes the entire text, reducing all instances
    of excessive repetition.

    Args:
        text: Input text that may contain repeated character sequences.
        threshold: Maximum number of consecutive repetitions to allow.
            Any sequence longer than this will be reduced to exactly
            this length. Defaults to 2.

    Returns:
        Text with repeated character sequences reduced to at most
        `threshold` occurrences.

    Example - Basic Repetition:
        >>> replace_repeated_chars("Helllllo World!")
        'Helllo World!'

    Example - Emphatic Text:
        >>> replace_repeated_chars("I'm soooooo happy!!!!!!")
        "I'm soo happy!!"

    Example - Custom Threshold:
        >>> replace_repeated_chars("Hellooooo", threshold=1)
        'Helo'

    Example - Higher Threshold:
        >>> replace_repeated_chars("yeeeees", threshold=3)
        'yeeees'

    Example - Multiple Repetitions:
        >>> replace_repeated_chars("Wooooow!!! Amaaazing!!!")
        'Woow!! Amaazin!!'

    Example - No Repetition Needed:
        >>> replace_repeated_chars("Hello World")
        'Hello World'

    Example - Whitespace Repetition:
        >>> replace_repeated_chars("Hello    World", threshold=1)
        'Hello World'

    Note:
        - A threshold of 2 means sequences of 3+ identical characters are
          reduced to 2 (e.g., "aaa" -> "aa", "aaaa" -> "aa").
        - A threshold of 1 removes all repetition (e.g., "aa" -> "a").
        - This function treats each character independently, so it won't
          distinguish between intentional repetition (like "bookkeeper")
          and emphatic repetition (like "booook"). Words with legitimate
          double letters may be affected if threshold=1.
        - For social media or informal text normalization, threshold=2
          usually provides good balance between normalization and
          preserving legitimate double letters.
    """
    pattern = re.compile(r"(.)\1{" + str(threshold) + ",}")
    return pattern.sub(lambda m: m.group(1) * threshold, text)


def clean_text(
    text: str,
    remove_html: bool = True,
    remove_url: bool = True,
    remove_punct: bool = False,
    remove_emoji: bool = False,
    remove_num: bool = False,
    normalize_white: bool = True,
    normalize_unicode_form: Optional[str] = "NFKC",
    normalize_contraction: bool = False,
    replace_repeated: bool = False,
    repeated_threshold: int = 2,
    lowercase: bool = True,
) -> str:
    """Apply multiple text cleaning operations in a single pipeline.

    This is the primary convenience function for text preprocessing. It applies
    a configurable sequence of cleaning operations to normalize and sanitize
    text data. The operations are applied in a specific order designed to
    maximize effectiveness and avoid conflicts between operations.

    The order of operations is:
        1. Remove HTML tags
        2. Remove URLs
        3. Normalize Unicode
        4. Remove emojis
        5. Remove numbers
        6. Remove punctuation
        7. Normalize contractions
        8. Replace repeated characters
        9. Normalize whitespace
        10. Convert to lowercase

    Args:
        text: Input text to clean. Can be any string including empty strings.
        remove_html: If True, strips all HTML/XML tags from the text.
            Defaults to True.
        remove_url: If True, removes HTTP/HTTPS URLs and www-prefixed URLs.
            Defaults to True.
        remove_punct: If True, removes all punctuation characters defined in
            string.punctuation. Defaults to False.
        remove_emoji: If True, removes emoji characters from various Unicode
            blocks. Defaults to False.
        remove_num: If True, removes all digit sequences (0-9).
            Defaults to False.
        normalize_white: If True, collapses multiple whitespace characters
            into single spaces and trims leading/trailing whitespace.
            Defaults to True.
        normalize_unicode_form: Unicode normalization form to apply. Options
            are 'NFC', 'NFD', 'NFKC', 'NFKD', or None to skip normalization.
            Defaults to 'NFKC' (recommended for most NLP tasks).
        normalize_contraction: If True, expands English contractions to their
            full forms (e.g., "don't" -> "do not"). Defaults to False.
        replace_repeated: If True, reduces excessive character repetition
            (e.g., "sooo" -> "soo"). Defaults to False.
        repeated_threshold: Maximum number of consecutive identical characters
            to allow when replace_repeated is True. Defaults to 2.
        lowercase: If True, converts the entire text to lowercase.
            Defaults to True.

    Returns:
        Cleaned and normalized text string. If all cleaning operations result
        in empty content, returns an empty string.

    Example - Default Cleaning (HTML, URLs, whitespace, lowercase):
        >>> clean_text("<p>Visit https://example.com  for more info!</p>")
        'visit for more info!'

    Example - Social Media Text:
        >>> clean_text(
        ...     "OMG!!! 😍 Check this out: http://link.co Sooooo cool!!!",
        ...     remove_emoji=True,
        ...     replace_repeated=True
        ... )
        'omg!! check this out: soo cool!!'

    Example - Full Normalization:
        >>> clean_text(
        ...     "I can't believe it's 2024!!! 🎉",
        ...     remove_punct=True,
        ...     remove_emoji=True,
        ...     remove_num=True,
        ...     normalize_contraction=True,
        ...     replace_repeated=True
        ... )
        'i can not believe it is'

    Example - Preserve Case and Punctuation:
        >>> clean_text(
        ...     "<b>Hello World!</b> Visit http://test.com",
        ...     remove_punct=False,
        ...     lowercase=False
        ... )
        'Hello World! Visit'

    Example - Minimal Cleaning:
        >>> clean_text(
        ...     "Text with   extra   spaces",
        ...     remove_html=False,
        ...     remove_url=False,
        ...     normalize_unicode_form=None,
        ...     lowercase=False
        ... )
        'Text with extra spaces'

    Example - Processing User Comments:
        >>> clean_text(
        ...     "AMAZING product!!! 🌟🌟🌟 Can't wait to buy more @ $19.99",
        ...     remove_emoji=True,
        ...     remove_num=True,
        ...     normalize_contraction=True,
        ...     replace_repeated=True
        ... )
        'amazing product!! can not wait to buy more @ $.'

    Note:
        - The order of operations matters. For example, contractions are
          expanded AFTER punctuation removal would strip apostrophes, so
          if you need both, be aware that remove_punct=True will prevent
          contraction normalization from working properly.
        - For best contraction handling with punctuation removal, consider
          calling normalize_contractions() separately before clean_text()
          with remove_punct=True.
        - Unicode normalization (NFKC) is applied early to ensure consistent
          character representation for subsequent operations.
        - Empty input returns empty output; no operations are applied.
    """
    if remove_html:
        text = remove_html_tags(text)

    if remove_url:
        text = remove_urls(text)

    if normalize_unicode_form:
        text = normalize_unicode(text, normalize_unicode_form)

    if remove_emoji:
        text = remove_emojis(text)

    if remove_num:
        text = remove_numbers(text)

    if remove_punct:
        text = remove_punctuation(text)

    if normalize_contraction:
        text = normalize_contractions(text)

    if replace_repeated:
        text = replace_repeated_chars(text, repeated_threshold)

    if normalize_white:
        text = normalize_whitespace(text)

    if lowercase:
        text = text.lower()

    return text
