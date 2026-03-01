"""Data preprocessing and text normalization utilities for LLM applications.

This module provides a comprehensive suite of tools for cleaning, normalizing,
and transforming text data before feeding it to Large Language Models (LLMs)
or analyzing their outputs. It includes configurable text normalization,
pattern-based cleaning, intelligent text splitting, validation pipelines,
and batch processing utilities.

Key Features
------------
- **Text Normalization**: Unicode normalization, whitespace handling, case conversion
- **Data Cleaning**: URL/email/phone removal, HTML stripping, PII masking
- **Text Splitting**: Chunk-based splitting with overlap for context preservation
- **Processing Pipelines**: Configurable, chainable text transformation pipelines
- **Validation**: Rule-based data validation with batch support
- **Batch Processing**: Token-aware batching for API rate limiting

Module Structure
----------------
Classes:
    NormalizationLevel : Enum defining levels of text normalization
    TextStats : Dataclass for text statistics computation
    TextNormalizer : Configurable text normalization with multiple options
    TextCleaner : Pattern-based text cleaning utilities
    TextSplitter : Intelligent text chunking for LLM context windows
    ProcessingStep : Single step in a processing pipeline
    ProcessingPipeline : Chainable text processing pipeline
    DataValidator : Rule-based text validation

Functions:
    normalize_whitespace : Normalize whitespace in text
    normalize_unicode : Apply Unicode normalization
    remove_special_chars : Remove special characters from text
    truncate_text : Truncate text with word boundary awareness
    count_tokens_approx : Approximate token count estimation
    batch_texts : Batch texts with size and token limits
    deduplicate_texts : Remove duplicate texts preserving order
    filter_by_length : Filter texts by character/word/token count
    create_standard_pipeline : Create a standard preprocessing pipeline
    create_minimal_pipeline : Create a minimal preprocessing pipeline

Examples
--------
Basic text normalization:

>>> from insideLLMs.nlp.preprocessing import TextNormalizer
>>> normalizer = TextNormalizer(lowercase=True, collapse_whitespace=True)
>>> normalizer.normalize("  Hello   WORLD!  ")
'hello world!'

Cleaning text with PII masking:

>>> from insideLLMs.nlp.preprocessing import TextCleaner
>>> text = "Contact john@example.com or call 555-123-4567"
>>> TextCleaner.mask_pii(text)
'Contact [REDACTED] or call[REDACTED]'

Building a processing pipeline:

>>> from insideLLMs.nlp.preprocessing import ProcessingPipeline, TextNormalizer, TextCleaner
>>> pipeline = ProcessingPipeline()
>>> _ = pipeline.add_step("normalize", TextNormalizer())
>>> _ = pipeline.add_step("remove_urls", TextCleaner.remove_urls)
>>> pipeline.process("Visit https://example.com for info")
'Visit  for info'

Splitting text for LLM context windows:

>>> from insideLLMs.nlp.preprocessing import TextSplitter
>>> splitter = TextSplitter(chunk_size=100, overlap=20)
>>> long_text = "This is a long document. " * 20
>>> chunks = splitter.split(long_text)
>>> len(chunks) > 1
True

Batch processing with token limits:

>>> from insideLLMs.nlp.preprocessing import batch_texts
>>> texts = ["Short text", "Another short one", "A bit longer text here"]
>>> batches = list(batch_texts(texts, batch_size=2, max_tokens_per_batch=50))
>>> len(batches) >= 1
True

Notes
-----
- All text processing functions handle empty strings gracefully
- Unicode normalization uses NFKC form by default for maximum compatibility
- Token counting is approximate (4 chars/token) - use a proper tokenizer for accuracy
- Processing pipelines can be used as callable objects for convenience
- PII masking is best-effort and should not be relied upon for compliance

See Also
--------
insideLLMs.nlp.text_cleaning : Lower-level text cleaning utilities
insideLLMs.tokenization : Proper tokenization with model-specific tokenizers

References
----------
.. [1] Unicode Normalization Forms: https://unicode.org/reports/tr15/
.. [2] Text Chunking Strategies: https://arxiv.org/abs/2312.06648
"""

import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Optional,
    TypeVar,
)

T = TypeVar("T")


class NormalizationLevel(Enum):
    """Enumeration of text normalization intensity levels.

    This enum defines four levels of text normalization, from no processing
    to aggressive normalization that applies all available transformations.
    Use this to configure TextNormalizer behavior or to communicate
    normalization requirements across your application.

    Attributes
    ----------
    NONE : str
        No normalization applied. Text is passed through unchanged.
    MINIMAL : str
        Only whitespace normalization. Collapses multiple spaces,
        removes leading/trailing whitespace.
    STANDARD : str
        Whitespace plus Unicode normalization. Recommended for most
        LLM preprocessing tasks.
    AGGRESSIVE : str
        All normalizations applied including control character removal,
        zero-width character removal, and case normalization.

    Examples
    --------
    Using normalization levels for configuration:

    >>> from insideLLMs.nlp.preprocessing import NormalizationLevel
    >>> level = NormalizationLevel.STANDARD
    >>> level.value
    'standard'

    Checking normalization level in conditionals:

    >>> level = NormalizationLevel.AGGRESSIVE
    >>> if level == NormalizationLevel.AGGRESSIVE:
    ...     print("Applying all normalizations")
    Applying all normalizations

    Iterating over all levels:

    >>> for level in NormalizationLevel:
    ...     print(f"{level.name}: {level.value}")
    NONE: none
    MINIMAL: minimal
    STANDARD: standard
    AGGRESSIVE: aggressive

    Using with dictionary configuration:

    >>> config = {"normalization": NormalizationLevel.MINIMAL}
    >>> config["normalization"].value
    'minimal'

    See Also
    --------
    TextNormalizer : Class that uses these levels for configuration
    """

    NONE = "none"
    MINIMAL = "minimal"  # Just whitespace
    STANDARD = "standard"  # Whitespace + unicode
    AGGRESSIVE = "aggressive"  # All normalizations


@dataclass
class TextStats:
    """Statistical analysis of text content.

    A dataclass that computes and stores various statistics about a text,
    including character count, word count, sentence count, line count,
    average word length, and unique word count. Useful for analyzing
    text before processing or for quality assessment.

    Parameters
    ----------
    char_count : int, default=0
        Total number of characters in the text, including whitespace.
    word_count : int, default=0
        Number of whitespace-separated words in the text.
    sentence_count : int, default=0
        Approximate number of sentences, detected by sentence-ending
        punctuation (., !, ?).
    line_count : int, default=0
        Number of lines in the text (newline-separated).
    avg_word_length : float, default=0.0
        Average length of words in characters.
    unique_words : int, default=0
        Number of unique words (case-insensitive).

    Attributes
    ----------
    char_count : int
        Total character count.
    word_count : int
        Word count (whitespace-separated).
    sentence_count : int
        Approximate sentence count.
    line_count : int
        Number of lines.
    avg_word_length : float
        Average word length.
    unique_words : int
        Number of unique words.

    Examples
    --------
    Computing stats from text using the class method:

    >>> from insideLLMs.nlp.preprocessing import TextStats
    >>> text = "Hello world. This is a test. Hello again!"
    >>> stats = TextStats.from_text(text)
    >>> stats.word_count
    8
    >>> stats.sentence_count
    3
    >>> stats.unique_words
    7

    Creating stats manually:

    >>> stats = TextStats(char_count=100, word_count=20, sentence_count=3)
    >>> stats.char_count
    100

    Analyzing multi-line text:

    >>> multiline = "Line one.\\nLine two.\\nLine three."
    >>> stats = TextStats.from_text(multiline)
    >>> stats.line_count
    3
    >>> stats.sentence_count
    3

    Checking vocabulary richness:

    >>> text = "the the the cat sat on the mat"
    >>> stats = TextStats.from_text(text)
    >>> vocabulary_richness = stats.unique_words / stats.word_count
    >>> vocabulary_richness < 0.6  # Low richness due to repetition
    True

    Handling empty text:

    >>> stats = TextStats.from_text("")
    >>> stats.word_count
    0
    >>> stats.avg_word_length
    0.0

    See Also
    --------
    TextNormalizer : Normalize text before computing stats
    count_tokens_approx : Estimate token count for LLM context
    """

    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    line_count: int = 0
    avg_word_length: float = 0.0
    unique_words: int = 0

    @classmethod
    def from_text(cls, text: str) -> "TextStats":
        """Compute statistics from text.

        Analyzes the input text and returns a TextStats object containing
        various metrics about the text content. This is the primary way
        to create a TextStats instance from actual text.

        Parameters
        ----------
        text : str
            The text to analyze. Can be empty, single-line, or multi-line.
            Unicode text is fully supported.

        Returns
        -------
        TextStats
            A TextStats object populated with computed statistics.
            If text is empty, returns a TextStats with all zero values.

        Examples
        --------
        Basic usage with simple text:

        >>> from insideLLMs.nlp.preprocessing import TextStats
        >>> stats = TextStats.from_text("Hello world!")
        >>> stats.char_count
        12
        >>> stats.word_count
        2

        Analyzing a paragraph:

        >>> paragraph = '''The quick brown fox jumps over the lazy dog.
        ... This sentence has nine words in it.
        ... And this is the third sentence!'''
        >>> stats = TextStats.from_text(paragraph)
        >>> stats.sentence_count
        3
        >>> stats.line_count
        3

        Computing average word length:

        >>> text = "I am a test"
        >>> stats = TextStats.from_text(text)
        >>> round(stats.avg_word_length, 2)
        2.0

        Finding unique word ratio:

        >>> repetitive = "hello hello world world world"
        >>> stats = TextStats.from_text(repetitive)
        >>> stats.word_count
        5
        >>> stats.unique_words
        2

        Handling Unicode text:

        >>> unicode_text = "Cafe with resume"
        >>> stats = TextStats.from_text(unicode_text)
        >>> stats.word_count
        3

        Empty text returns zero stats:

        >>> stats = TextStats.from_text("")
        >>> stats.word_count == stats.char_count == 0
        True

        Notes
        -----
        - Sentence detection is approximate, using [.!?] as delimiters
        - Words are split on whitespace only
        - Unique words are compared case-insensitively
        - Line count is newline-based, single-line text has line_count=1

        See Also
        --------
        TextNormalizer.normalize : Normalize text before analysis
        """
        if not text:
            return cls()

        words = text.split()
        unique = {w.lower() for w in words}

        # Approximate sentence count
        sentences = re.split(r"[.!?]+", text)
        sentence_count = len([s for s in sentences if s.strip()])

        avg_len = sum(len(w) for w in words) / len(words) if words else 0

        return cls(
            char_count=len(text),
            word_count=len(words),
            sentence_count=sentence_count,
            line_count=text.count("\n") + 1,
            avg_word_length=avg_len,
            unique_words=len(unique),
        )


class TextNormalizer:
    """Configurable text normalizer with multiple normalization options.

    TextNormalizer provides a flexible way to normalize text by applying
    various transformations in a consistent order. It handles Unicode
    normalization, whitespace collapsing, control character removal,
    and more. The normalizer can be configured at instantiation time
    and used as a callable.

    Parameters
    ----------
    lowercase : bool, default=False
        Convert all text to lowercase.
    strip : bool, default=True
        Strip leading/trailing whitespace from the text and each line.
    collapse_whitespace : bool, default=True
        Collapse multiple consecutive spaces to a single space.
    remove_extra_newlines : bool, default=True
        Reduce more than 2 consecutive newlines to exactly 2.
    unicode_normalize : bool, default=True
        Apply Unicode normalization to the text.
    unicode_form : str, default="NFKC"
        Unicode normalization form. One of: "NFC", "NFD", "NFKC", "NFKD".
        NFKC is recommended for maximum compatibility.
    remove_control_chars : bool, default=True
        Remove ASCII control characters (except newline and tab).
    remove_zero_width : bool, default=True
        Remove zero-width Unicode characters (ZWSP, ZWNJ, ZWJ, BOM).

    Attributes
    ----------
    lowercase : bool
        Whether to convert to lowercase.
    strip : bool
        Whether to strip whitespace.
    collapse_whitespace : bool
        Whether to collapse multiple spaces.
    remove_extra_newlines : bool
        Whether to remove extra newlines.
    unicode_normalize : bool
        Whether to apply Unicode normalization.
    unicode_form : str
        The Unicode normalization form to use.
    remove_control_chars : bool
        Whether to remove control characters.
    remove_zero_width : bool
        Whether to remove zero-width characters.

    Examples
    --------
    Basic normalization with defaults:

    >>> from insideLLMs.nlp.preprocessing import TextNormalizer
    >>> normalizer = TextNormalizer()
    >>> normalizer.normalize("  Hello   World!  ")
    'Hello World!'

    Case-insensitive normalization:

    >>> normalizer = TextNormalizer(lowercase=True)
    >>> normalizer.normalize("Hello WORLD")
    'hello world'

    Preserving whitespace patterns:

    >>> normalizer = TextNormalizer(collapse_whitespace=False, strip=False)
    >>> normalizer.normalize("  multiple   spaces  ")
    '  multiple   spaces  '

    Handling Unicode edge cases:

    >>> normalizer = TextNormalizer(unicode_form="NFC")
    >>> # Combining character normalization
    >>> text_with_combining = "cafe\\u0301"  # cafe with combining acute
    >>> len(text_with_combining)
    5
    >>> normalized = normalizer.normalize(text_with_combining)
    >>> # NFC composes the combining character
    >>> len(normalized) == 5 or len(normalized) == 4  # Depends on form
    True

    Removing zero-width characters (often used in steganography):

    >>> normalizer = TextNormalizer(remove_zero_width=True)
    >>> text_with_zwsp = "hello\\u200bworld"  # Zero-width space
    >>> normalizer.normalize(text_with_zwsp)
    'helloworld'

    Using as a callable:

    >>> normalizer = TextNormalizer()
    >>> texts = ["  text1  ", "  text2  "]
    >>> list(map(normalizer, texts))
    ['text1', 'text2']

    Cleaning multi-line text:

    >>> normalizer = TextNormalizer(remove_extra_newlines=True)
    >>> text = "Paragraph 1\\n\\n\\n\\n\\nParagraph 2"
    >>> normalizer.normalize(text)
    'Paragraph 1\\n\\nParagraph 2'

    Full aggressive normalization:

    >>> normalizer = TextNormalizer(
    ...     lowercase=True,
    ...     strip=True,
    ...     collapse_whitespace=True,
    ...     remove_extra_newlines=True,
    ...     unicode_normalize=True,
    ...     remove_control_chars=True,
    ...     remove_zero_width=True,
    ... )
    >>> messy_text = "  HELLO\\x00world\\u200b  "
    >>> normalizer.normalize(messy_text)
    'helloworld'

    Notes
    -----
    The normalization steps are applied in the following order:
    1. Unicode normalization
    2. Zero-width character removal
    3. Control character removal
    4. Whitespace collapsing
    5. Extra newline removal
    6. Stripping (text and individual lines)
    7. Lowercase conversion

    See Also
    --------
    NormalizationLevel : Enum for normalization intensity levels
    normalize_whitespace : Standalone whitespace normalization
    normalize_unicode : Standalone Unicode normalization
    """

    def __init__(
        self,
        lowercase: bool = False,
        strip: bool = True,
        collapse_whitespace: bool = True,
        remove_extra_newlines: bool = True,
        unicode_normalize: bool = True,
        unicode_form: str = "NFKC",
        remove_control_chars: bool = True,
        remove_zero_width: bool = True,
    ):
        """Initialize normalizer with configuration options.

        Parameters
        ----------
        lowercase : bool, default=False
            Convert all text to lowercase. Useful for case-insensitive
            processing or when case is not semantically meaningful.
        strip : bool, default=True
            Strip leading/trailing whitespace from the entire text
            and from each individual line.
        collapse_whitespace : bool, default=True
            Replace multiple consecutive spaces/tabs with a single space.
            Newlines are preserved separately.
        remove_extra_newlines : bool, default=True
            Reduce sequences of more than 2 consecutive newlines to
            exactly 2 (preserving paragraph breaks).
        unicode_normalize : bool, default=True
            Apply Unicode normalization. Essential for consistent
            text comparison and processing.
        unicode_form : str, default="NFKC"
            Unicode normalization form to use. Options:
            - "NFC": Canonical Decomposition, followed by Canonical Composition
            - "NFD": Canonical Decomposition
            - "NFKC": Compatibility Decomposition, followed by Canonical Composition
            - "NFKD": Compatibility Decomposition
        remove_control_chars : bool, default=True
            Remove ASCII control characters (0x00-0x08, 0x0B, 0x0C,
            0x0E-0x1F, 0x7F). Newline and tab are preserved.
        remove_zero_width : bool, default=True
            Remove zero-width Unicode characters including:
            - U+200B Zero Width Space
            - U+200C Zero Width Non-Joiner
            - U+200D Zero Width Joiner
            - U+FEFF Byte Order Mark

        Examples
        --------
        Default configuration (recommended for most use cases):

        >>> normalizer = TextNormalizer()
        >>> normalizer.strip
        True
        >>> normalizer.unicode_form
        'NFKC'

        Configuration for case-sensitive comparison:

        >>> normalizer = TextNormalizer(lowercase=False, unicode_form="NFC")
        >>> normalizer.lowercase
        False

        Minimal normalization (preserve formatting):

        >>> normalizer = TextNormalizer(
        ...     strip=False,
        ...     collapse_whitespace=False,
        ...     remove_extra_newlines=False
        ... )
        >>> normalizer.collapse_whitespace
        False

        See Also
        --------
        normalize : Apply the configured normalization
        """
        self.lowercase = lowercase
        self.strip = strip
        self.collapse_whitespace = collapse_whitespace
        self.remove_extra_newlines = remove_extra_newlines
        self.unicode_normalize = unicode_normalize
        self.unicode_form = unicode_form
        self.remove_control_chars = remove_control_chars
        self.remove_zero_width = remove_zero_width

        # Zero-width characters
        self._zero_width = re.compile(r"[\u200b\u200c\u200d\ufeff]")
        # Control characters (except newline and tab)
        self._control = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def normalize(self, text: str) -> str:
        """Normalize text according to configured settings.

        Applies all enabled normalization steps to the input text in a
        consistent order. Empty strings are returned unchanged.

        Parameters
        ----------
        text : str
            The text to normalize. Can be empty, single-line, or multi-line.
            Unicode text is fully supported.

        Returns
        -------
        str
            The normalized text with all enabled transformations applied.
            Returns the original text if it's empty or None-like.

        Examples
        --------
        Basic whitespace normalization:

        >>> normalizer = TextNormalizer()
        >>> normalizer.normalize("  hello    world  ")
        'hello world'

        Unicode normalization with combining characters:

        >>> normalizer = TextNormalizer(unicode_normalize=True)
        >>> # e followed by combining acute accent
        >>> normalizer.normalize("caf\\u0065\\u0301")  # doctest: +SKIP
        'cafe'

        Removing hidden characters:

        >>> normalizer = TextNormalizer()
        >>> # Text with zero-width spaces (often used to bypass filters)
        >>> hidden = "pass\\u200bword"
        >>> normalizer.normalize(hidden)
        'password'

        Multi-line text cleanup:

        >>> normalizer = TextNormalizer()
        >>> messy = "  Line 1  \\n\\n\\n\\n  Line 2  "
        >>> normalizer.normalize(messy)
        'Line 1\\n\\nLine 2'

        Case normalization:

        >>> normalizer = TextNormalizer(lowercase=True)
        >>> normalizer.normalize("HeLLo WoRLD")
        'hello world'

        Preserving empty input:

        >>> normalizer = TextNormalizer()
        >>> normalizer.normalize("")
        ''

        Processing text with control characters:

        >>> normalizer = TextNormalizer(remove_control_chars=True)
        >>> text_with_null = "hello\\x00world"
        >>> normalizer.normalize(text_with_null)
        'helloworld'

        Notes
        -----
        Normalization steps are applied in order:
        1. Unicode normalization (composing/decomposing characters)
        2. Zero-width character removal
        3. Control character removal
        4. Whitespace collapsing (spaces/tabs to single space)
        5. Extra newline removal (>2 newlines to 2)
        6. Stripping (whole text and each line)
        7. Lowercase conversion

        See Also
        --------
        __call__ : Alias for normalize, allows using instance as function
        """
        if not text:
            return text

        result = text

        # Unicode normalization first
        if self.unicode_normalize:
            result = unicodedata.normalize(
                self.unicode_form,  # type: ignore[arg-type]
                result,  # type: ignore[arg-type]
            )

        # Remove zero-width characters
        if self.remove_zero_width:
            result = self._zero_width.sub("", result)

        # Remove control characters
        if self.remove_control_chars:
            result = self._control.sub("", result)

        # Collapse whitespace
        if self.collapse_whitespace:
            result = re.sub(r"[^\S\n]+", " ", result)

        # Remove extra newlines (more than 2 consecutive)
        if self.remove_extra_newlines:
            result = re.sub(r"\n{3,}", "\n\n", result)

        # Strip
        if self.strip:
            result = result.strip()
            # Also strip each line
            result = "\n".join(line.strip() for line in result.split("\n"))

        # Lowercase
        if self.lowercase:
            result = result.lower()

        return result

    def __call__(self, text: str) -> str:
        """Allow using normalizer as a callable function.

        This method enables using the TextNormalizer instance directly
        as a function, which is convenient for use with map(), filter(),
        or as a step in ProcessingPipeline.

        Parameters
        ----------
        text : str
            The text to normalize.

        Returns
        -------
        str
            The normalized text.

        Examples
        --------
        Using with map():

        >>> normalizer = TextNormalizer(lowercase=True)
        >>> texts = ["  HELLO  ", "  WORLD  "]
        >>> list(map(normalizer, texts))
        ['hello', 'world']

        Using in a list comprehension:

        >>> normalizer = TextNormalizer()
        >>> [normalizer(t) for t in ["  a  ", "  b  "]]
        ['a', 'b']

        Direct callable syntax:

        >>> normalizer = TextNormalizer()
        >>> normalizer("  test  ")
        'test'

        See Also
        --------
        normalize : The underlying normalization method
        ProcessingPipeline : Use normalizer as a pipeline step
        """
        return self.normalize(text)


class TextCleaner:
    """Utility class for cleaning text by removing unwanted patterns.

    TextCleaner provides a collection of class methods for removing or
    replacing common patterns in text such as URLs, email addresses,
    phone numbers, HTML tags, emojis, and personally identifiable
    information (PII). All methods are static/class methods, so no
    instantiation is required.

    Attributes
    ----------
    URL_PATTERN : re.Pattern
        Compiled regex for matching URLs (http://, https://, www.).
    EMAIL_PATTERN : re.Pattern
        Compiled regex for matching email addresses.
    PHONE_PATTERN : re.Pattern
        Compiled regex for matching phone numbers (various formats).
    HTML_TAG_PATTERN : re.Pattern
        Compiled regex for matching HTML tags.
    MARKDOWN_LINK_PATTERN : re.Pattern
        Compiled regex for matching Markdown-style links [text](url).
    EMOJI_PATTERN : re.Pattern
        Compiled regex for matching common emoji ranges.

    Examples
    --------
    Removing URLs from text:

    >>> from insideLLMs.nlp.preprocessing import TextCleaner
    >>> text = "Check out https://example.com and www.test.org"
    >>> TextCleaner.remove_urls(text)
    'Check out  and '

    Replacing URLs with placeholder:

    >>> TextCleaner.remove_urls(text, replacement="[URL]")
    'Check out [URL] and [URL]'

    Masking PII in user content:

    >>> sensitive = "Email me at john@example.com or call 555-123-4567"
    >>> TextCleaner.mask_pii(sensitive)
    'Email me at [REDACTED] or call [REDACTED]'

    Cleaning HTML content:

    >>> html = "<p>Hello <strong>world</strong>!</p>"
    >>> TextCleaner.remove_html_tags(html)
    'Hello world!'

    Converting Markdown links to plain text:

    >>> md = "Visit [our website](https://example.com) for more"
    >>> TextCleaner.convert_markdown_links(md)
    'Visit our website for more'

    Removing emojis from text:

    >>> text_with_emoji = "Hello world! :-)"
    >>> TextCleaner.remove_emojis(text_with_emoji)  # doctest: +SKIP
    'Hello world! :-)'

    Chaining multiple cleaners:

    >>> messy = "Contact <a href='mailto:a@b.com'>us</a> at https://example.com"
    >>> cleaned = TextCleaner.remove_html_tags(messy)
    >>> cleaned = TextCleaner.remove_urls(cleaned)
    >>> cleaned = TextCleaner.remove_emails(cleaned)
    >>> cleaned
    'Contact us at '

    Notes
    -----
    - All methods are class methods and can be called without instantiation
    - Patterns are compiled at class definition time for efficiency
    - Phone number detection may have false positives with other number sequences
    - PII masking is best-effort and should not be relied upon for compliance

    See Also
    --------
    TextNormalizer : For whitespace and Unicode normalization
    ProcessingPipeline : For chaining multiple cleaning operations
    """

    # Common patterns
    URL_PATTERN = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+")
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    PHONE_PATTERN = re.compile(r"\+?[\d\s\-().]{7,}")
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
    EMOJI_PATTERN = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002702-\U000027B0\U0001F900-\U0001F9FF]+"
    )

    @classmethod
    def remove_urls(cls, text: str, replacement: str = "") -> str:
        """Remove URLs from text.

        Matches and removes both http(s):// URLs and www. prefixed URLs.
        URLs are replaced with the specified replacement string.

        Parameters
        ----------
        text : str
            Input text containing URLs to remove.
        replacement : str, default=""
            String to replace URLs with. Use "[URL]" or similar
            placeholders to indicate removed content.

        Returns
        -------
        str
            Text with all URLs replaced by the replacement string.

        Examples
        --------
        Remove URLs completely:

        >>> from insideLLMs.nlp.preprocessing import TextCleaner
        >>> TextCleaner.remove_urls("Visit https://example.com today")
        'Visit  today'

        Replace with placeholder:

        >>> TextCleaner.remove_urls("See https://example.com", "[LINK]")
        'See [LINK]'

        Handle multiple URLs:

        >>> text = "Sites: https://a.com, https://b.org, www.c.net"
        >>> TextCleaner.remove_urls(text, "[URL]")
        'Sites: [URL], [URL], [URL]'

        URLs with query strings:

        >>> TextCleaner.remove_urls("https://example.com/page?id=123&foo=bar")
        ''

        Handle text without URLs:

        >>> TextCleaner.remove_urls("No URLs here")
        'No URLs here'

        See Also
        --------
        mask_pii : Mask URLs along with other PII
        """
        return cls.URL_PATTERN.sub(replacement, text)

    @classmethod
    def remove_emails(cls, text: str, replacement: str = "") -> str:
        """Remove email addresses from text.

        Detects and removes email addresses matching the standard format
        of local-part@domain.tld.

        Parameters
        ----------
        text : str
            Input text containing email addresses to remove.
        replacement : str, default=""
            String to replace email addresses with.

        Returns
        -------
        str
            Text with all email addresses replaced.

        Examples
        --------
        Remove email addresses:

        >>> from insideLLMs.nlp.preprocessing import TextCleaner
        >>> TextCleaner.remove_emails("Contact john@example.com for help")
        'Contact  for help'

        Replace with placeholder:

        >>> TextCleaner.remove_emails("Email: test@domain.org", "[EMAIL]")
        'Email: [EMAIL]'

        Handle multiple emails:

        >>> text = "CC: alice@a.com, bob@b.org"
        >>> TextCleaner.remove_emails(text, "[REDACTED]")
        'CC: [REDACTED], [REDACTED]'

        Complex email addresses:

        >>> TextCleaner.remove_emails("user.name+tag@sub.domain.co.uk")
        ''

        Preserve non-email @ symbols:

        >>> TextCleaner.remove_emails("@username on Twitter")
        '@username on Twitter'

        See Also
        --------
        mask_pii : Mask emails along with other PII
        """
        return cls.EMAIL_PATTERN.sub(replacement, text)

    @classmethod
    def remove_phone_numbers(cls, text: str, replacement: str = "") -> str:
        """Remove phone numbers from text.

        Detects and removes phone numbers in various formats including
        international format with country codes, parenthesized area codes,
        and various separators (spaces, dashes, dots).

        Parameters
        ----------
        text : str
            Input text containing phone numbers to remove.
        replacement : str, default=""
            String to replace phone numbers with.

        Returns
        -------
        str
            Text with phone numbers replaced.

        Examples
        --------
        Remove US-style phone numbers:

        >>> from insideLLMs.nlp.preprocessing import TextCleaner
        >>> TextCleaner.remove_phone_numbers("Call 555-123-4567")
        'Call '

        Remove international format:

        >>> TextCleaner.remove_phone_numbers("Phone: +1 (555) 123-4567")
        'Phone: '

        Replace with placeholder:

        >>> TextCleaner.remove_phone_numbers("Call 555.123.4567", "[PHONE]")
        'Call [PHONE]'

        Handle various formats:

        >>> TextCleaner.remove_phone_numbers("Tel: (555) 123 4567")
        'Tel: '

        Notes
        -----
        The phone number pattern is intentionally broad and may match
        other sequences of digits. For precise phone number detection,
        consider using a specialized library like phonenumbers.

        Warning
        -------
        This method may produce false positives with other numeric
        sequences like dates, IDs, or version numbers.

        See Also
        --------
        mask_pii : Mask phone numbers along with other PII
        """
        return cls.PHONE_PATTERN.sub(replacement, text)

    @classmethod
    def remove_html_tags(cls, text: str) -> str:
        """Remove HTML tags from text, keeping only content.

        Strips all HTML/XML tags from the text, leaving the text content
        between tags intact. Does not decode HTML entities.

        Parameters
        ----------
        text : str
            Input text containing HTML tags.

        Returns
        -------
        str
            Text with all HTML tags removed.

        Examples
        --------
        Remove simple tags:

        >>> from insideLLMs.nlp.preprocessing import TextCleaner
        >>> TextCleaner.remove_html_tags("<p>Hello</p>")
        'Hello'

        Handle nested tags:

        >>> TextCleaner.remove_html_tags("<div><p>Nested <b>content</b></p></div>")
        'Nested content'

        Self-closing tags:

        >>> TextCleaner.remove_html_tags("Line 1<br/>Line 2")
        'Line 1Line 2'

        Tags with attributes:

        >>> TextCleaner.remove_html_tags('<a href="url">Link text</a>')
        'Link text'

        Script and style tags (removes tags but keeps content):

        >>> TextCleaner.remove_html_tags("<script>code</script>text")
        'codetext'

        Notes
        -----
        This is a simple tag removal and does not:
        - Decode HTML entities (&amp;, &lt;, etc.)
        - Handle malformed HTML gracefully
        - Remove script/style content (only tags)

        For complex HTML processing, consider using BeautifulSoup
        or similar HTML parsing libraries.

        See Also
        --------
        convert_markdown_links : Convert Markdown links to plain text
        """
        return cls.HTML_TAG_PATTERN.sub("", text)

    @classmethod
    def convert_markdown_links(cls, text: str) -> str:
        """Convert Markdown links to plain text, keeping link text.

        Transforms Markdown-style links [text](url) to just the
        visible text portion, removing the URL.

        Parameters
        ----------
        text : str
            Input text containing Markdown links.

        Returns
        -------
        str
            Text with Markdown links converted to plain text.

        Examples
        --------
        Convert a simple link:

        >>> from insideLLMs.nlp.preprocessing import TextCleaner
        >>> TextCleaner.convert_markdown_links("[Click here](https://example.com)")
        'Click here'

        Multiple links in text:

        >>> text = "See [docs](https://docs.com) and [API](https://api.com)"
        >>> TextCleaner.convert_markdown_links(text)
        'See docs and API'

        Links with complex URLs:

        >>> md = "[Search](https://google.com/search?q=test&lang=en)"
        >>> TextCleaner.convert_markdown_links(md)
        'Search'

        Mixed content:

        >>> text = "Visit [our site](https://example.com) for **more info**"
        >>> TextCleaner.convert_markdown_links(text)
        'Visit our site for **more info**'

        No links to convert:

        >>> TextCleaner.convert_markdown_links("Plain text only")
        'Plain text only'

        See Also
        --------
        remove_urls : Remove URLs entirely
        remove_html_tags : Remove HTML tags
        """
        return cls.MARKDOWN_LINK_PATTERN.sub(r"\1", text)

    @classmethod
    def remove_emojis(cls, text: str, replacement: str = "") -> str:
        """Remove emoji characters from text.

        Detects and removes common emoji characters from various
        Unicode emoji blocks including emoticons, symbols, and flags.

        Parameters
        ----------
        text : str
            Input text containing emojis to remove.
        replacement : str, default=""
            String to replace emojis with.

        Returns
        -------
        str
            Text with emojis removed.

        Examples
        --------
        Remove smiley emojis:

        >>> from insideLLMs.nlp.preprocessing import TextCleaner
        >>> # Using actual emoji (may not display correctly in all terminals)
        >>> TextCleaner.remove_emojis("Hello! Great job!")  # doctest: +SKIP
        'Hello! Great job!'

        Replace with placeholder:

        >>> TextCleaner.remove_emojis("Thumbs up!", "[EMOJI]")  # doctest: +SKIP
        'Thumbs up[EMOJI]'

        Text without emojis unchanged:

        >>> TextCleaner.remove_emojis("Plain ASCII text :)")
        'Plain ASCII text :)'

        Notes
        -----
        The emoji pattern covers common emoji ranges but may not catch
        all emojis, especially newer ones or combined emoji sequences.
        Text emoticons like :) are not removed.

        See Also
        --------
        remove_special_chars : Remove non-alphanumeric characters
        """
        return cls.EMOJI_PATTERN.sub(replacement, text)

    @classmethod
    def remove_punctuation(cls, text: str, keep: str = "", replacement: str = "") -> str:
        """Remove punctuation from text.

        Removes all standard ASCII punctuation characters, optionally
        keeping specified characters.

        Parameters
        ----------
        text : str
            Input text containing punctuation to remove.
        keep : str, default=""
            Punctuation characters to preserve. For example, ".,!?"
            would keep periods, commas, exclamation and question marks.
        replacement : str, default=""
            String to replace each punctuation character with.

        Returns
        -------
        str
            Text with punctuation removed or replaced.

        Examples
        --------
        Remove all punctuation:

        >>> from insideLLMs.nlp.preprocessing import TextCleaner
        >>> TextCleaner.remove_punctuation("Hello, world! How are you?")
        'Hello world How are you'

        Keep sentence-ending punctuation:

        >>> TextCleaner.remove_punctuation("Hello, world! Yes?", keep=".!?")
        'Hello world! Yes?'

        Replace with spaces:

        >>> TextCleaner.remove_punctuation("one-two-three", replacement=" ")
        'one two three'

        Keep apostrophes for contractions:

        >>> TextCleaner.remove_punctuation("don't stop!", keep="'")
        "don't stop"

        No punctuation to remove:

        >>> TextCleaner.remove_punctuation("no punctuation here")
        'no punctuation here'

        See Also
        --------
        remove_special_chars : Remove non-alphanumeric characters
        """
        import string

        punct = set(string.punctuation) - set(keep)
        for p in punct:
            text = text.replace(p, replacement)
        return text

    @classmethod
    def remove_numbers(cls, text: str, replacement: str = "") -> str:
        """Remove numeric digits from text.

        Removes all sequences of digits (0-9) from the text.

        Parameters
        ----------
        text : str
            Input text containing numbers to remove.
        replacement : str, default=""
            String to replace numbers with.

        Returns
        -------
        str
            Text with numbers removed.

        Examples
        --------
        Remove all numbers:

        >>> from insideLLMs.nlp.preprocessing import TextCleaner
        >>> TextCleaner.remove_numbers("Room 101 on Floor 3")
        'Room  on Floor '

        Replace with placeholder:

        >>> TextCleaner.remove_numbers("Order #12345", "[NUM]")
        'Order #[NUM]'

        Handle mixed content:

        >>> TextCleaner.remove_numbers("v2.0.1 released on 2024-01-15")
        'v.. released on --'

        No numbers to remove:

        >>> TextCleaner.remove_numbers("no numbers here")
        'no numbers here'

        See Also
        --------
        remove_punctuation : Remove punctuation characters
        remove_special_chars : Remove non-alphanumeric characters
        """
        return re.sub(r"\d+", replacement, text)

    @classmethod
    def mask_pii(
        cls,
        text: str,
        mask_email: bool = True,
        mask_phone: bool = True,
        mask_url: bool = False,
        mask_char: str = "[REDACTED]",
    ) -> str:
        """Mask personally identifiable information in text.

        Replaces detected PII (emails, phone numbers, optionally URLs)
        with a mask string. Useful for anonymizing user-generated content
        before processing or logging.

        Parameters
        ----------
        text : str
            Input text containing PII to mask.
        mask_email : bool, default=True
            Whether to mask email addresses.
        mask_phone : bool, default=True
            Whether to mask phone numbers.
        mask_url : bool, default=False
            Whether to mask URLs. Disabled by default as URLs
            are often not considered PII.
        mask_char : str, default="[REDACTED]"
            String to replace PII with. Common choices:
            "[REDACTED]", "[PII]", "***", "<removed>"

        Returns
        -------
        str
            Text with PII replaced by mask string.

        Examples
        --------
        Basic PII masking:

        >>> from insideLLMs.nlp.preprocessing import TextCleaner
        >>> text = "Contact john@example.com or call 555-123-4567"
        >>> TextCleaner.mask_pii(text)
        'Contact [REDACTED] or call [REDACTED]'

        Custom mask string:

        >>> TextCleaner.mask_pii("Email: user@domain.com", mask_char="***")
        'Email: ***'

        Include URLs in masking:

        >>> text = "Visit https://example.com, email info@site.com"
        >>> TextCleaner.mask_pii(text, mask_url=True)
        'Visit [REDACTED], email [REDACTED]'

        Selective masking (email only):

        >>> text = "Email john@x.com, phone 555-1234"
        >>> TextCleaner.mask_pii(text, mask_phone=False)
        'Email [REDACTED], phone 555-1234'

        No PII detected:

        >>> TextCleaner.mask_pii("Plain text without PII")
        'Plain text without PII'

        Warning
        -------
        This is a best-effort PII detection and should NOT be relied
        upon for regulatory compliance (GDPR, HIPAA, etc.). For
        production PII handling, use specialized libraries like
        Microsoft Presidio or AWS Comprehend.

        Notes
        -----
        Detection patterns are intentionally broad, which may result
        in some false positives. Phone number detection in particular
        may match other numeric sequences.

        See Also
        --------
        remove_emails : Remove emails without replacement
        remove_phone_numbers : Remove phone numbers without replacement
        remove_urls : Remove URLs without replacement
        """
        result = text
        if mask_email:
            result = cls.remove_emails(result, mask_char)
        if mask_phone:
            result = cls.remove_phone_numbers(result, mask_char)
        if mask_url:
            result = cls.remove_urls(result, mask_char)
        return result


class TextSplitter:
    """Intelligent text splitter for chunking content.

    TextSplitter divides long text into smaller chunks suitable for
    processing by LLMs with limited context windows. It attempts to
    split at natural boundaries (paragraphs, sentences, words) and
    supports overlapping chunks to preserve context across boundaries.

    Parameters
    ----------
    chunk_size : int, default=1000
        Maximum chunk size in characters. Actual chunks may be slightly
        smaller when breaking at natural boundaries.
    overlap : int, default=100
        Number of characters to overlap between consecutive chunks.
        Helps preserve context across chunk boundaries.
    separator : str, default="\\n\\n"
        Preferred split point (e.g., paragraph break). The splitter
        will try to break at this separator when possible.
    keep_separator : bool, default=True
        Whether to include the separator in the chunk that precedes it.

    Attributes
    ----------
    chunk_size : int
        Maximum chunk size in characters.
    overlap : int
        Overlap between chunks in characters.
    separator : str
        Preferred split point.
    keep_separator : bool
        Whether to keep separator in chunks.

    Examples
    --------
    Basic text splitting:

    >>> from insideLLMs.nlp.preprocessing import TextSplitter
    >>> splitter = TextSplitter(chunk_size=50, overlap=10)
    >>> long_text = "This is a test. " * 10
    >>> chunks = splitter.split(long_text)
    >>> len(chunks) >= 2
    True
    >>> all(len(c) <= 50 for c in chunks)
    True

    Splitting with paragraph awareness:

    >>> splitter = TextSplitter(chunk_size=100, separator="\\n\\n")
    >>> text = "Paragraph 1.\\n\\nParagraph 2.\\n\\nParagraph 3."
    >>> chunks = splitter.split(text)
    >>> len(chunks) >= 1
    True

    Sentence-level splitting:

    >>> splitter = TextSplitter()
    >>> text = "First sentence. Second sentence. Third sentence."
    >>> sentences = splitter.split_by_sentences(text)
    >>> len(sentences)
    3
    >>> sentences[0]
    'First sentence.'

    Paragraph-level splitting:

    >>> text = "Para 1.\\n\\nPara 2.\\n\\nPara 3."
    >>> paragraphs = splitter.split_by_paragraphs(text)
    >>> len(paragraphs)
    3

    Processing chunks with overlap:

    >>> splitter = TextSplitter(chunk_size=30, overlap=5)
    >>> text = "This is chunk one. This is chunk two."
    >>> chunks = splitter.split(text)
    >>> # Overlap helps maintain context between chunks
    >>> len(chunks) >= 2
    True

    Short text returns single chunk:

    >>> splitter = TextSplitter(chunk_size=1000)
    >>> splitter.split("Short text")
    ['Short text']

    Notes
    -----
    The splitter tries boundaries in this priority order:
    1. Preferred separator (default: paragraph break)
    2. Sentence-ending punctuation (. ! ? followed by space)
    3. Newline
    4. Word boundary (space)
    5. Hard cut at chunk_size

    See Also
    --------
    batch_texts : Batch texts by count and token limits
    ProcessingPipeline : Chain splitting with other processing steps
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        separator: str = "\n\n",
        keep_separator: bool = True,
    ):
        """Initialize splitter with chunking configuration.

        Parameters
        ----------
        chunk_size : int, default=1000
            Maximum chunk size in characters. Choose based on your
            LLM's context window and desired processing granularity.
            Common values: 500-2000 for analysis, 2000-8000 for RAG.
        overlap : int, default=100
            Characters to overlap between chunks. Helps preserve
            context when information spans chunk boundaries.
            Set to 0 for no overlap.
        separator : str, default="\\n\\n"
            Preferred split point. The splitter will try to break
            at this pattern when possible. Use "\\n\\n" for paragraphs,
            "\\n" for lines, ". " for sentences.
        keep_separator : bool, default=True
            If True, includes the separator at the end of the chunk
            before the split. If False, separator is discarded.

        Examples
        --------
        Configure for short chunks with no overlap:

        >>> splitter = TextSplitter(chunk_size=200, overlap=0)
        >>> splitter.chunk_size
        200

        Configure for RAG with generous overlap:

        >>> splitter = TextSplitter(chunk_size=2000, overlap=200)
        >>> splitter.overlap
        200

        Split on sentences instead of paragraphs:

        >>> splitter = TextSplitter(separator=". ", keep_separator=True)
        >>> splitter.separator
        '. '

        See Also
        --------
        split : Split text into chunks
        split_by_sentences : Split specifically by sentences
        split_by_paragraphs : Split specifically by paragraphs
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator
        self.keep_separator = keep_separator

    def split(self, text: str) -> list[str]:
        """Split text into chunks respecting natural boundaries.

        Divides the input text into chunks of at most chunk_size
        characters, attempting to break at natural boundaries like
        paragraphs, sentences, and words. Adjacent chunks may overlap
        by the configured overlap amount.

        Parameters
        ----------
        text : str
            The text to split into chunks.

        Returns
        -------
        list[str]
            List of text chunks. Each chunk is at most chunk_size
            characters. Empty input returns an empty list.

        Examples
        --------
        Split a long document:

        >>> from insideLLMs.nlp.preprocessing import TextSplitter
        >>> splitter = TextSplitter(chunk_size=100, overlap=20)
        >>> doc = "This is the first paragraph.\\n\\nThis is the second paragraph."
        >>> chunks = splitter.split(doc)
        >>> isinstance(chunks, list)
        True

        Verify chunk sizes:

        >>> splitter = TextSplitter(chunk_size=50)
        >>> long_text = "word " * 50
        >>> chunks = splitter.split(long_text)
        >>> all(len(chunk) <= 50 for chunk in chunks)
        True

        Handle short text:

        >>> splitter = TextSplitter(chunk_size=1000)
        >>> splitter.split("Short")
        ['Short']

        Handle empty text:

        >>> splitter.split("")
        []

        Process with overlap:

        >>> splitter = TextSplitter(chunk_size=20, overlap=5)
        >>> text = "One two three four five six"
        >>> chunks = splitter.split(text)
        >>> len(chunks) >= 2
        True

        Notes
        -----
        The splitting algorithm:
        1. If text fits in chunk_size, return as single chunk
        2. Otherwise, find best split point before chunk_size:
           a. Look for preferred separator
           b. Look for sentence ending (. ! ? followed by space)
           c. Look for newline
           d. Look for word boundary (space)
           e. Hard cut at chunk_size if no boundary found
        3. Create chunk, advance by (end - overlap), repeat

        See Also
        --------
        split_by_sentences : Split strictly by sentences
        split_by_paragraphs : Split strictly by paragraphs
        """
        if len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If not at the end, try to find a good split point
            if end < len(text):
                # Look for separator
                sep_pos = text.rfind(self.separator, start, end)
                if sep_pos > start:
                    end = sep_pos + len(self.separator) if self.keep_separator else sep_pos

                else:
                    # Look for sentence end
                    for delim in [". ", "! ", "? ", "\n"]:
                        pos = text.rfind(delim, start, end)
                        if pos > start:
                            end = pos + len(delim)
                            break
                    else:
                        # Look for word boundary
                        space_pos = text.rfind(" ", start, end)
                        if space_pos > start:
                            end = space_pos + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            start = end - self.overlap if self.overlap > 0 else end

        return chunks

    def split_by_sentences(self, text: str) -> list[str]:
        """Split text into individual sentences.

        Uses sentence-ending punctuation (. ! ?) followed by whitespace
        as sentence boundaries. This is a simple heuristic that works
        well for most prose but may fail on abbreviations like "Dr."
        or "U.S."

        Parameters
        ----------
        text : str
            The text to split into sentences.

        Returns
        -------
        list[str]
            List of sentences, with leading/trailing whitespace stripped.
            Empty input returns an empty list.

        Examples
        --------
        Split simple sentences:

        >>> from insideLLMs.nlp.preprocessing import TextSplitter
        >>> splitter = TextSplitter()
        >>> text = "First sentence. Second sentence. Third sentence."
        >>> splitter.split_by_sentences(text)
        ['First sentence.', 'Second sentence.', 'Third sentence.']

        Handle different punctuation:

        >>> text = "Statement. Question? Exclamation!"
        >>> splitter.split_by_sentences(text)
        ['Statement.', 'Question?', 'Exclamation!']

        Multi-line text:

        >>> text = "Line one.\\nLine two. Line three."
        >>> sentences = splitter.split_by_sentences(text)
        >>> len(sentences)
        3

        Handle empty text:

        >>> splitter.split_by_sentences("")
        []

        Single sentence:

        >>> splitter.split_by_sentences("Just one sentence.")
        ['Just one sentence.']

        Notes
        -----
        This is a simple regex-based splitter. For production use with
        complex text (legal documents, scientific papers), consider
        using spaCy or NLTK's sentence tokenizers.

        Warning
        -------
        May incorrectly split on abbreviations like "Dr. Smith" or
        "U.S. Army".

        See Also
        --------
        split : Split by chunk size with boundary awareness
        split_by_paragraphs : Split by paragraph breaks
        """
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def split_by_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs.

        Splits text at blank lines (one or more empty lines between
        text blocks). Each paragraph is stripped of leading/trailing
        whitespace.

        Parameters
        ----------
        text : str
            The text to split into paragraphs.

        Returns
        -------
        list[str]
            List of paragraphs, with leading/trailing whitespace stripped.
            Empty input returns an empty list.

        Examples
        --------
        Split standard paragraphs:

        >>> from insideLLMs.nlp.preprocessing import TextSplitter
        >>> splitter = TextSplitter()
        >>> text = "First para.\\n\\nSecond para.\\n\\nThird para."
        >>> splitter.split_by_paragraphs(text)
        ['First para.', 'Second para.', 'Third para.']

        Handle multiple blank lines:

        >>> text = "Para 1.\\n\\n\\n\\nPara 2."
        >>> splitter.split_by_paragraphs(text)
        ['Para 1.', 'Para 2.']

        Single paragraph:

        >>> splitter.split_by_paragraphs("Just one paragraph.")
        ['Just one paragraph.']

        Empty input:

        >>> splitter.split_by_paragraphs("")
        []

        Paragraphs with internal newlines:

        >>> text = "Line 1\\nLine 2\\n\\nLine 3\\nLine 4"
        >>> paras = splitter.split_by_paragraphs(text)
        >>> len(paras)
        2

        See Also
        --------
        split : Split by chunk size with boundary awareness
        split_by_sentences : Split by sentence boundaries
        """
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]


@dataclass
class ProcessingStep:
    """A single step in a text processing pipeline.

    ProcessingStep wraps a text transformation function with a name
    and enabled flag, allowing steps to be toggled on/off without
    removing them from the pipeline.

    Parameters
    ----------
    name : str
        Unique identifier for this step. Used for enabling/disabling.
    func : Callable[[str], str]
        Function that takes a string and returns a transformed string.
    enabled : bool, default=True
        Whether this step should be applied during processing.

    Attributes
    ----------
    name : str
        Step name.
    func : Callable[[str], str]
        Processing function.
    enabled : bool
        Whether step is enabled.

    Examples
    --------
    Create a processing step:

    >>> from insideLLMs.nlp.preprocessing import ProcessingStep
    >>> step = ProcessingStep(
    ...     name="lowercase",
    ...     func=str.lower,
    ...     enabled=True
    ... )
    >>> step.apply("HELLO")
    'hello'

    Disabled step passes through unchanged:

    >>> step = ProcessingStep(name="upper", func=str.upper, enabled=False)
    >>> step.apply("hello")
    'hello'

    Using a custom function:

    >>> def add_prefix(text: str) -> str:
    ...     return f"PREFIX: {text}"
    >>> step = ProcessingStep(name="prefix", func=add_prefix)
    >>> step.apply("content")
    'PREFIX: content'

    Using a lambda:

    >>> step = ProcessingStep(name="strip", func=lambda x: x.strip())
    >>> step.apply("  text  ")
    'text'

    See Also
    --------
    ProcessingPipeline : Manages multiple ProcessingSteps
    """

    name: str
    func: Callable[[str], str]
    enabled: bool = True

    def apply(self, text: str) -> str:
        """Apply this processing step to text if enabled.

        If the step is enabled, applies the configured function to
        the input text. If disabled, returns the text unchanged.

        Parameters
        ----------
        text : str
            The text to process.

        Returns
        -------
        str
            Processed text if enabled, original text if disabled.

        Examples
        --------
        Apply an enabled step:

        >>> from insideLLMs.nlp.preprocessing import ProcessingStep
        >>> step = ProcessingStep(name="upper", func=str.upper, enabled=True)
        >>> step.apply("hello")
        'HELLO'

        Disabled step returns input unchanged:

        >>> step = ProcessingStep(name="upper", func=str.upper, enabled=False)
        >>> step.apply("hello")
        'hello'

        Toggle step and reapply:

        >>> step = ProcessingStep(name="lower", func=str.lower, enabled=True)
        >>> step.apply("HELLO")
        'hello'
        >>> step.enabled = False
        >>> step.apply("HELLO")
        'HELLO'

        See Also
        --------
        ProcessingPipeline.process : Apply all steps in sequence
        """
        if self.enabled:
            return self.func(text)
        return text


class ProcessingPipeline:
    """A configurable, chainable text processing pipeline.

    ProcessingPipeline allows you to define a sequence of text
    transformation steps that are applied in order. Steps can be
    added, enabled, or disabled dynamically. The pipeline can be
    used as a callable for convenience.

    This is useful for creating reusable preprocessing configurations
    that can be shared across your application.

    Attributes
    ----------
    _steps : list[ProcessingStep]
        Internal list of processing steps.

    Examples
    --------
    Build a basic pipeline:

    >>> from insideLLMs.nlp.preprocessing import ProcessingPipeline, TextNormalizer, TextCleaner
    >>> pipeline = ProcessingPipeline()
    >>> pipeline.add_step("normalize", TextNormalizer())
    <...ProcessingPipeline...>
    >>> pipeline.add_step("remove_urls", TextCleaner.remove_urls)
    <...ProcessingPipeline...>
    >>> result = pipeline.process("  Visit https://example.com  ")
    >>> result
    'Visit'

    Chain step additions:

    >>> pipeline = (
    ...     ProcessingPipeline()
    ...     .add_step("strip", str.strip)
    ...     .add_step("lower", str.lower)
    ... )
    >>> pipeline("  HELLO  ")
    'hello'

    Enable/disable steps dynamically:

    >>> pipeline = ProcessingPipeline()
    >>> pipeline.add_step("upper", str.upper)
    <...ProcessingPipeline...>
    >>> pipeline.process("hello")
    'HELLO'
    >>> pipeline.disable_step("upper")
    <...ProcessingPipeline...>
    >>> pipeline.process("hello")
    'hello'

    Batch processing:

    >>> pipeline = ProcessingPipeline().add_step("strip", str.strip)
    >>> texts = ["  a  ", "  b  ", "  c  "]
    >>> pipeline.process_batch(texts)
    ['a', 'b', 'c']

    List pipeline steps:

    >>> pipeline = ProcessingPipeline()
    >>> pipeline.add_step("step1", str.strip, enabled=True)
    <...ProcessingPipeline...>
    >>> pipeline.add_step("step2", str.lower, enabled=False)
    <...ProcessingPipeline...>
    >>> pipeline.list_steps()
    [('step1', True), ('step2', False)]

    Use as callable:

    >>> pipeline = ProcessingPipeline().add_step("upper", str.upper)
    >>> list(map(pipeline, ["a", "b", "c"]))
    ['A', 'B', 'C']

    Notes
    -----
    Steps are applied in the order they are added. The pipeline
    uses method chaining, so all configuration methods return self.

    See Also
    --------
    ProcessingStep : Individual step in the pipeline
    create_standard_pipeline : Factory for common pipeline configuration
    create_minimal_pipeline : Factory for minimal pipeline configuration
    """

    def __init__(self):
        """Initialize an empty processing pipeline.

        Creates a new pipeline with no steps. Steps can be added
        using the add_step method.

        Examples
        --------
        Create an empty pipeline:

        >>> from insideLLMs.nlp.preprocessing import ProcessingPipeline
        >>> pipeline = ProcessingPipeline()
        >>> pipeline.list_steps()
        []

        Immediately add steps:

        >>> pipeline = ProcessingPipeline()
        >>> pipeline.add_step("strip", str.strip)
        <...ProcessingPipeline...>

        See Also
        --------
        add_step : Add a processing step
        create_standard_pipeline : Create a pre-configured pipeline
        """
        self._steps: list[ProcessingStep] = []

    def add_step(
        self,
        name: str,
        func: Callable[[str], str],
        enabled: bool = True,
    ) -> "ProcessingPipeline":
        """Add a processing step to the pipeline.

        Appends a new step to the end of the pipeline. Steps are
        applied in the order they are added.

        Parameters
        ----------
        name : str
            Unique identifier for this step. Used for enable/disable.
        func : Callable[[str], str]
            Function that transforms text. Can be a regular function,
            lambda, method, or callable object (like TextNormalizer).
        enabled : bool, default=True
            Whether the step should be active. Disabled steps are
            skipped during processing.

        Returns
        -------
        ProcessingPipeline
            Self, for method chaining.

        Examples
        --------
        Add a simple function:

        >>> from insideLLMs.nlp.preprocessing import ProcessingPipeline
        >>> pipeline = ProcessingPipeline()
        >>> pipeline.add_step("strip", str.strip)
        <...ProcessingPipeline...>

        Add a lambda:

        >>> pipeline.add_step("prefix", lambda x: f">>> {x}")
        <...ProcessingPipeline...>

        Add a callable class instance:

        >>> from insideLLMs.nlp.preprocessing import TextNormalizer
        >>> pipeline.add_step("normalize", TextNormalizer(lowercase=True))
        <...ProcessingPipeline...>

        Add a disabled step:

        >>> pipeline.add_step("debug", print, enabled=False)
        <...ProcessingPipeline...>

        Chain multiple additions:

        >>> pipeline = (
        ...     ProcessingPipeline()
        ...     .add_step("step1", str.strip)
        ...     .add_step("step2", str.lower)
        ...     .add_step("step3", str.title)
        ... )
        >>> pipeline.process("  hello world  ")
        'Hello World'

        See Also
        --------
        enable_step : Enable a disabled step
        disable_step : Disable an enabled step
        list_steps : View all steps and their status
        """
        self._steps.append(ProcessingStep(name=name, func=func, enabled=enabled))
        return self

    def enable_step(self, name: str) -> "ProcessingPipeline":
        """Enable a processing step by name.

        Sets the enabled flag to True for the step with the given name.
        If no step with that name exists, this method does nothing.

        Parameters
        ----------
        name : str
            Name of the step to enable.

        Returns
        -------
        ProcessingPipeline
            Self, for method chaining.

        Examples
        --------
        Enable a disabled step:

        >>> from insideLLMs.nlp.preprocessing import ProcessingPipeline
        >>> pipeline = ProcessingPipeline()
        >>> pipeline.add_step("upper", str.upper, enabled=False)
        <...ProcessingPipeline...>
        >>> pipeline.process("hello")
        'hello'
        >>> pipeline.enable_step("upper")
        <...ProcessingPipeline...>
        >>> pipeline.process("hello")
        'HELLO'

        Enable already-enabled step (no effect):

        >>> pipeline.enable_step("upper")  # No error
        <...ProcessingPipeline...>

        Enable non-existent step (no error):

        >>> pipeline.enable_step("nonexistent")  # No error
        <...ProcessingPipeline...>

        See Also
        --------
        disable_step : Disable a step
        list_steps : View step status
        """
        for step in self._steps:
            if step.name == name:
                step.enabled = True
                break
        return self

    def disable_step(self, name: str) -> "ProcessingPipeline":
        """Disable a processing step by name.

        Sets the enabled flag to False for the step with the given name.
        Disabled steps are skipped during processing but remain in the
        pipeline and can be re-enabled later.

        Parameters
        ----------
        name : str
            Name of the step to disable.

        Returns
        -------
        ProcessingPipeline
            Self, for method chaining.

        Examples
        --------
        Disable a step:

        >>> from insideLLMs.nlp.preprocessing import ProcessingPipeline
        >>> pipeline = ProcessingPipeline()
        >>> pipeline.add_step("upper", str.upper)
        <...ProcessingPipeline...>
        >>> pipeline.process("hello")
        'HELLO'
        >>> pipeline.disable_step("upper")
        <...ProcessingPipeline...>
        >>> pipeline.process("hello")
        'hello'

        Chain with other operations:

        >>> (pipeline
        ...     .enable_step("upper")
        ...     .add_step("strip", str.strip)
        ...     .disable_step("upper"))
        <...ProcessingPipeline...>

        See Also
        --------
        enable_step : Enable a step
        list_steps : View step status
        """
        for step in self._steps:
            if step.name == name:
                step.enabled = False
                break
        return self

    def process(self, text: str) -> str:
        """Process text through all enabled steps.

        Applies each enabled step in sequence, passing the output
        of each step as input to the next.

        Parameters
        ----------
        text : str
            Input text to process.

        Returns
        -------
        str
            Processed text after all enabled steps have been applied.

        Examples
        --------
        Process through multiple steps:

        >>> from insideLLMs.nlp.preprocessing import ProcessingPipeline
        >>> pipeline = (
        ...     ProcessingPipeline()
        ...     .add_step("strip", str.strip)
        ...     .add_step("lower", str.lower)
        ... )
        >>> pipeline.process("  HELLO WORLD  ")
        'hello world'

        Empty pipeline returns input unchanged:

        >>> ProcessingPipeline().process("unchanged")
        'unchanged'

        Only enabled steps are applied:

        >>> pipeline = (
        ...     ProcessingPipeline()
        ...     .add_step("upper", str.upper, enabled=True)
        ...     .add_step("lower", str.lower, enabled=False)
        ... )
        >>> pipeline.process("Hello")
        'HELLO'

        See Also
        --------
        process_batch : Process multiple texts
        __call__ : Alias for process
        """
        result = text
        for step in self._steps:
            result = step.apply(result)
        return result

    def process_batch(self, texts: Iterable[str]) -> list[str]:
        """Process multiple texts through the pipeline.

        Applies the pipeline to each text in the input iterable.

        Parameters
        ----------
        texts : Iterable[str]
            Iterable of texts to process.

        Returns
        -------
        list[str]
            List of processed texts in the same order.

        Examples
        --------
        Process a list of texts:

        >>> from insideLLMs.nlp.preprocessing import ProcessingPipeline
        >>> pipeline = ProcessingPipeline().add_step("upper", str.upper)
        >>> pipeline.process_batch(["hello", "world"])
        ['HELLO', 'WORLD']

        Process a generator:

        >>> def gen_texts():
        ...     yield "a"
        ...     yield "b"
        >>> pipeline.process_batch(gen_texts())
        ['A', 'B']

        Empty input:

        >>> pipeline.process_batch([])
        []

        See Also
        --------
        process : Process single text
        """
        return [self.process(text) for text in texts]

    def list_steps(self) -> list[tuple[str, bool]]:
        """List all steps with their enabled status.

        Returns a list of tuples containing step names and their
        current enabled/disabled status.

        Returns
        -------
        list[tuple[str, bool]]
            List of (name, enabled) tuples for each step.

        Examples
        --------
        List steps in a pipeline:

        >>> from insideLLMs.nlp.preprocessing import ProcessingPipeline
        >>> pipeline = (
        ...     ProcessingPipeline()
        ...     .add_step("strip", str.strip, enabled=True)
        ...     .add_step("lower", str.lower, enabled=False)
        ...     .add_step("title", str.title, enabled=True)
        ... )
        >>> pipeline.list_steps()
        [('strip', True), ('lower', False), ('title', True)]

        Empty pipeline:

        >>> ProcessingPipeline().list_steps()
        []

        See Also
        --------
        enable_step : Enable a step
        disable_step : Disable a step
        """
        return [(step.name, step.enabled) for step in self._steps]

    def __call__(self, text: str) -> str:
        """Allow using pipeline as a callable function.

        This enables using the pipeline with map(), filter(), and
        other higher-order functions.

        Parameters
        ----------
        text : str
            The text to process.

        Returns
        -------
        str
            The processed text.

        Examples
        --------
        Use with map():

        >>> from insideLLMs.nlp.preprocessing import ProcessingPipeline
        >>> pipeline = ProcessingPipeline().add_step("upper", str.upper)
        >>> list(map(pipeline, ["a", "b", "c"]))
        ['A', 'B', 'C']

        Use in list comprehension:

        >>> [pipeline(t) for t in ["x", "y"]]
        ['X', 'Y']

        Direct call:

        >>> pipeline("hello")
        'HELLO'

        See Also
        --------
        process : The underlying processing method
        """
        return self.process(text)


class DataValidator:
    """Rule-based data validator for text preprocessing.

    DataValidator allows you to define validation rules that text
    must pass before processing. Rules are functions that return
    True for valid input and False otherwise. Multiple rules can
    be combined, and validation returns all failures.

    Attributes
    ----------
    _rules : dict[str, tuple[Callable[[str], bool], str]]
        Dictionary mapping rule names to (rule_func, error_message) tuples.

    Examples
    --------
    Basic validation:

    >>> from insideLLMs.nlp.preprocessing import DataValidator
    >>> validator = DataValidator()
    >>> validator.add_rule("non_empty", lambda x: len(x) > 0, "Text cannot be empty")
    <...DataValidator...>
    >>> validator.validate("hello")
    (True, [])
    >>> validator.validate("")
    (False, ['Text cannot be empty'])

    Multiple rules:

    >>> validator = DataValidator()
    >>> validator.add_rule("non_empty", lambda x: len(x) > 0)
    <...DataValidator...>
    >>> validator.add_rule("max_length", lambda x: len(x) <= 100)
    <...DataValidator...>
    >>> validator.add_rule("no_urls", lambda x: "http" not in x)
    <...DataValidator...>
    >>> is_valid, errors = validator.validate("Short text")
    >>> is_valid
    True

    Validation with custom error messages:

    >>> validator = DataValidator()
    >>> validator.add_rule(
    ...     "min_words",
    ...     lambda x: len(x.split()) >= 3,
    ...     "Text must have at least 3 words"
    ... )
    <...DataValidator...>
    >>> validator.validate("too short")
    (False, ['Text must have at least 3 words'])

    Batch validation:

    >>> validator = DataValidator()
    >>> validator.add_rule("non_empty", lambda x: bool(x.strip()))
    <...DataValidator...>
    >>> results = validator.validate_batch(["valid", "", "also valid"])
    >>> [(i, valid) for i, valid, _ in results]
    [(0, True), (1, False), (2, True)]

    Chaining rule additions:

    >>> validator = (
    ...     DataValidator()
    ...     .add_rule("r1", lambda x: True)
    ...     .add_rule("r2", lambda x: True)
    ... )
    >>> len(validator._rules)
    2

    See Also
    --------
    ProcessingPipeline : For text transformation after validation
    """

    def __init__(self):
        """Initialize validator with no rules.

        Creates a new validator with an empty rule set. Rules can
        be added using the add_rule method.

        Examples
        --------
        Create empty validator:

        >>> from insideLLMs.nlp.preprocessing import DataValidator
        >>> validator = DataValidator()
        >>> validator.validate("anything")
        (True, [])

        See Also
        --------
        add_rule : Add validation rules
        """
        self._rules: dict[str, tuple[Callable[[str], bool], str]] = {}

    def add_rule(
        self,
        name: str,
        rule: Callable[[str], bool],
        error_msg: Optional[str] = None,
    ) -> "DataValidator":
        """Add a validation rule.

        Rules are functions that take a string and return True if
        the string is valid, False otherwise. Each rule has a name
        and an optional custom error message.

        Parameters
        ----------
        name : str
            Unique identifier for this rule.
        rule : Callable[[str], bool]
            Validation function. Should return True for valid input,
            False for invalid. Should not raise exceptions for
            invalid input.
        error_msg : str, optional
            Custom error message. If not provided, defaults to
            "Failed rule: {name}".

        Returns
        -------
        DataValidator
            Self, for method chaining.

        Examples
        --------
        Add a simple rule:

        >>> from insideLLMs.nlp.preprocessing import DataValidator
        >>> validator = DataValidator()
        >>> validator.add_rule("non_empty", lambda x: len(x) > 0)
        <...DataValidator...>

        Add with custom error message:

        >>> validator.add_rule(
        ...     "max_100",
        ...     lambda x: len(x) <= 100,
        ...     "Text exceeds 100 character limit"
        ... )
        <...DataValidator...>

        Add complex validation:

        >>> import re
        >>> validator.add_rule(
        ...     "valid_email",
        ...     lambda x: bool(re.match(r"[^@]+@[^@]+\\.[^@]+", x)),
        ...     "Invalid email format"
        ... )
        <...DataValidator...>

        Chain multiple rules:

        >>> validator = (
        ...     DataValidator()
        ...     .add_rule("r1", lambda x: True)
        ...     .add_rule("r2", lambda x: True)
        ...     .add_rule("r3", lambda x: True)
        ... )

        See Also
        --------
        validate : Run all rules on text
        """
        self._rules[name] = (rule, error_msg or f"Failed rule: {name}")
        return self

    def validate(self, text: str) -> tuple[bool, list[str]]:
        """Validate text against all rules.

        Runs all registered validation rules and collects any failures.
        Returns a tuple of (is_valid, error_messages).

        Parameters
        ----------
        text : str
            Text to validate.

        Returns
        -------
        tuple[bool, list[str]]
            A tuple of (is_valid, errors) where:
            - is_valid: True if all rules passed, False otherwise
            - errors: List of error messages for failed rules

        Examples
        --------
        All rules pass:

        >>> from insideLLMs.nlp.preprocessing import DataValidator
        >>> validator = DataValidator()
        >>> validator.add_rule("non_empty", lambda x: len(x) > 0)
        <...DataValidator...>
        >>> validator.validate("valid text")
        (True, [])

        Some rules fail:

        >>> validator.add_rule("min_10", lambda x: len(x) >= 10, "Too short")
        <...DataValidator...>
        >>> is_valid, errors = validator.validate("short")
        >>> is_valid
        False
        >>> "Too short" in errors
        True

        Multiple failures:

        >>> validator = DataValidator()
        >>> validator.add_rule("r1", lambda x: False, "Error 1")
        <...DataValidator...>
        >>> validator.add_rule("r2", lambda x: False, "Error 2")
        <...DataValidator...>
        >>> is_valid, errors = validator.validate("test")
        >>> len(errors)
        2

        Rule that raises exception:

        >>> validator = DataValidator()
        >>> validator.add_rule("bad", lambda x: 1/0)  # Will raise
        <...DataValidator...>
        >>> is_valid, errors = validator.validate("test")
        >>> is_valid
        False
        >>> "error" in errors[0].lower()
        True

        See Also
        --------
        validate_batch : Validate multiple texts
        add_rule : Add validation rules
        """
        errors = []
        for name, (rule, error_msg) in self._rules.items():
            try:
                if not rule(text):
                    errors.append(error_msg)
            except Exception as e:
                errors.append(f"Rule '{name}' error: {e}")
        return len(errors) == 0, errors

    def validate_batch(self, texts: Iterable[str]) -> list[tuple[int, bool, list[str]]]:
        """Validate multiple texts and return indexed results.

        Validates each text in the input and returns results with
        their original indices for easy correlation.

        Parameters
        ----------
        texts : Iterable[str]
            Iterable of texts to validate.

        Returns
        -------
        list[tuple[int, bool, list[str]]]
            List of (index, is_valid, errors) tuples for each text.

        Examples
        --------
        Validate a batch:

        >>> from insideLLMs.nlp.preprocessing import DataValidator
        >>> validator = DataValidator()
        >>> validator.add_rule("non_empty", lambda x: bool(x.strip()))
        <...DataValidator...>
        >>> results = validator.validate_batch(["valid", "", "valid"])
        >>> [(i, valid) for i, valid, _ in results]
        [(0, True), (1, False), (2, True)]

        Filter invalid texts:

        >>> texts = ["good", "", "also good", "   "]
        >>> results = validator.validate_batch(texts)
        >>> valid_indices = [i for i, valid, _ in results if valid]
        >>> valid_indices
        [0, 2]

        Get all error messages:

        >>> validator = DataValidator()
        >>> validator.add_rule("no_x", lambda x: "x" not in x, "Contains x")
        <...DataValidator...>
        >>> results = validator.validate_batch(["abc", "xyz", "def"])
        >>> [errors for _, _, errors in results if errors]
        [['Contains x']]

        See Also
        --------
        validate : Validate single text
        """
        results = []
        for i, text in enumerate(texts):
            is_valid, errors = self.validate(text)
            results.append((i, is_valid, errors))
        return results


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text by collapsing and trimming.

    A convenience function that collapses multiple consecutive spaces
    to a single space and removes leading/trailing whitespace. This
    delegates to the implementation in insideLLMs.nlp.text_cleaning.

    Parameters
    ----------
    text : str
        Input text with potentially irregular whitespace.

    Returns
    -------
    str
        Text with normalized whitespace.

    Examples
    --------
    Collapse multiple spaces:

    >>> from insideLLMs.nlp.preprocessing import normalize_whitespace
    >>> normalize_whitespace("hello    world")
    'hello world'

    Trim leading/trailing whitespace:

    >>> normalize_whitespace("   hello world   ")
    'hello world'

    Handle tabs and mixed whitespace:

    >>> normalize_whitespace("hello\\t\\tworld")
    'hello world'

    Empty and whitespace-only strings:

    >>> normalize_whitespace("   ")
    ''
    >>> normalize_whitespace("")
    ''

    Notes
    -----
    This function delegates to insideLLMs.nlp.text_cleaning.normalize_whitespace
    for the actual implementation.

    See Also
    --------
    TextNormalizer : Full-featured text normalization
    normalize_unicode : Unicode normalization
    """
    from insideLLMs.nlp.text_cleaning import normalize_whitespace as _normalize_ws

    return _normalize_ws(text)


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize Unicode characters in text.

    Applies Unicode normalization to ensure consistent character
    representation. This is important for text comparison, searching,
    and processing.

    Parameters
    ----------
    text : str
        Input text with potentially non-normalized Unicode.
    form : str, default="NFKC"
        Unicode normalization form. One of:
        - "NFC": Canonical Decomposition, followed by Canonical Composition
        - "NFD": Canonical Decomposition
        - "NFKC": Compatibility Decomposition, followed by Canonical Composition
        - "NFKD": Compatibility Decomposition

    Returns
    -------
    str
        Unicode-normalized text.

    Examples
    --------
    Normalize combining characters:

    >>> from insideLLMs.nlp.preprocessing import normalize_unicode
    >>> # e + combining acute accent
    >>> text = "caf\\u0065\\u0301"
    >>> len(text)
    5
    >>> normalized = normalize_unicode(text, "NFC")
    >>> # May compose to single character depending on form
    >>> isinstance(normalized, str)
    True

    Normalize compatibility characters:

    >>> normalize_unicode("\\ufb01", "NFKC")  # fi ligature
    'fi'

    Different normalization forms:

    >>> text = "test"
    >>> normalize_unicode(text, "NFC") == normalize_unicode(text, "NFKC")
    True

    Notes
    -----
    - NFC/NFD: Canonical forms, preserve character semantics
    - NFKC/NFKD: Compatibility forms, normalize visual variants

    NFKC is recommended for most text processing as it handles
    the widest range of variations.

    This function delegates to insideLLMs.nlp.text_cleaning.normalize_unicode
    for the actual implementation.

    See Also
    --------
    TextNormalizer : Full-featured text normalization
    normalize_whitespace : Whitespace normalization
    """
    from insideLLMs.nlp.text_cleaning import normalize_unicode as _normalize_uc

    return _normalize_uc(text, form)


def remove_special_chars(text: str, keep: str = "") -> str:
    """Remove special characters, keeping only alphanumeric and specified chars.

    Removes all characters that are not letters, digits, or whitespace,
    except for characters explicitly specified in the keep parameter.

    Parameters
    ----------
    text : str
        Input text containing special characters to remove.
    keep : str, default=""
        Additional characters to preserve. Each character in this
        string will be kept in the output.

    Returns
    -------
    str
        Text with special characters removed.

    Examples
    --------
    Remove all special characters:

    >>> from insideLLMs.nlp.preprocessing import remove_special_chars
    >>> remove_special_chars("Hello, World! @2024")
    'Hello World 2024'

    Keep specific characters:

    >>> remove_special_chars("price: $99.99", keep="$.")
    'price $99.99'

    Keep hyphens and apostrophes for natural text:

    >>> remove_special_chars("don't stop-go!", keep="-'")
    "don't stop-go"

    Handle empty string:

    >>> remove_special_chars("")
    ''

    Preserve whitespace:

    >>> remove_special_chars("hello   world!!!")
    'hello   world'

    Multiple special characters:

    >>> remove_special_chars("a@b#c$d%e", keep="#")
    'ab#cde'

    See Also
    --------
    TextCleaner.remove_punctuation : Remove punctuation specifically
    TextNormalizer : Full text normalization
    """
    pattern = f"[^a-zA-Z0-9\\s{re.escape(keep)}]"
    return re.sub(pattern, "", text)


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "...",
    word_boundary: bool = True,
) -> str:
    """Truncate text to a maximum length with optional word boundary awareness.

    Shortens text to fit within max_length characters, optionally
    breaking at word boundaries for cleaner output. A suffix
    (default "...") is appended to indicate truncation.

    Parameters
    ----------
    text : str
        Input text to truncate.
    max_length : int
        Maximum length of the output including suffix.
    suffix : str, default="..."
        String appended when text is truncated.
    word_boundary : bool, default=True
        If True, try to break at the last word boundary before
        max_length. If False, cut exactly at max_length.

    Returns
    -------
    str
        Truncated text, or original if already within limit.

    Examples
    --------
    Basic truncation:

    >>> from insideLLMs.nlp.preprocessing import truncate_text
    >>> truncate_text("Hello World", max_length=8)
    'Hell...'

    Word boundary awareness:

    >>> truncate_text("Hello World Example", max_length=12, word_boundary=True)
    'Hello...'

    Without word boundary:

    >>> truncate_text("Hello World", max_length=8, word_boundary=False)
    'Hell...'

    Custom suffix:

    >>> truncate_text("Hello World", max_length=10, suffix="[more]")
    'Hel[more]'

    No truncation needed:

    >>> truncate_text("Short", max_length=100)
    'Short'

    Empty suffix:

    >>> truncate_text("Hello World", max_length=5, suffix="")
    'Hello'

    Single word:

    >>> truncate_text("Supercalifragilistic", max_length=10)
    'Superc...'

    See Also
    --------
    TextSplitter : For splitting long text into chunks
    count_tokens_approx : Estimate tokens for LLM limits
    """
    if len(text) <= max_length:
        return text

    cut_length = max_length - len(suffix)

    if word_boundary:
        # Find last space before cut point
        space_pos = text.rfind(" ", 0, cut_length)
        if space_pos > 0:
            cut_length = space_pos

    return text[:cut_length].rstrip() + suffix


def count_tokens_approx(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count using character-based approximation.

    Provides a quick estimate of token count without requiring
    a tokenizer. Useful for rough calculations when exact counts
    aren't critical.

    Parameters
    ----------
    text : str
        Input text to estimate tokens for.
    chars_per_token : float, default=4.0
        Average characters per token. Common values:
        - 4.0: General English text (default)
        - 3.5: Technical/code content
        - 5.0: Simple vocabulary text

    Returns
    -------
    int
        Estimated token count (always rounded down).

    Examples
    --------
    Estimate tokens for English text:

    >>> from insideLLMs.nlp.preprocessing import count_tokens_approx
    >>> count_tokens_approx("Hello, world!")
    3

    Adjust for different content types:

    >>> code = "def hello_world(): return 42"
    >>> count_tokens_approx(code, chars_per_token=3.5)
    8

    Empty text:

    >>> count_tokens_approx("")
    0

    Long text estimation:

    >>> long_text = "word " * 1000
    >>> tokens = count_tokens_approx(long_text)
    >>> 1000 <= tokens <= 1500
    True

    Notes
    -----
    This is a rough approximation. Actual token counts depend on:
    - The specific tokenizer (GPT-4, Claude, etc.)
    - Text content (code vs prose)
    - Language (English vs other languages)
    - Special characters and formatting

    For accurate counts, use the appropriate model's tokenizer.

    See Also
    --------
    batch_texts : Batch texts with token limits
    truncate_text : Truncate by character count
    """
    return int(len(text) / chars_per_token)


def batch_texts(
    texts: list[str],
    batch_size: int,
    max_tokens_per_batch: Optional[int] = None,
) -> Iterator[list[str]]:
    """Batch texts for processing with size and token limits.

    Groups texts into batches respecting both item count and
    approximate token limits. Useful for API rate limiting or
    processing in chunks that fit model context windows.

    Parameters
    ----------
    texts : list[str]
        List of texts to batch.
    batch_size : int
        Maximum number of texts per batch.
    max_tokens_per_batch : int, optional
        Maximum approximate tokens per batch. If None, only
        batch_size is considered.

    Yields
    ------
    list[str]
        Batches of texts. Each batch contains at most batch_size
        items and approximately max_tokens_per_batch tokens.

    Examples
    --------
    Batch by count only:

    >>> from insideLLMs.nlp.preprocessing import batch_texts
    >>> texts = ["a", "b", "c", "d", "e"]
    >>> list(batch_texts(texts, batch_size=2))
    [['a', 'b'], ['c', 'd'], ['e']]

    Batch with token limit:

    >>> texts = ["short", "also short", "this is a longer text"]
    >>> batches = list(batch_texts(texts, batch_size=10, max_tokens_per_batch=10))
    >>> len(batches) >= 1
    True

    Empty input:

    >>> list(batch_texts([], batch_size=5))
    []

    Single item batches:

    >>> list(batch_texts(["a", "b"], batch_size=1))
    [['a'], ['b']]

    Large texts may get their own batch:

    >>> texts = ["tiny", "x" * 1000, "tiny"]
    >>> batches = list(batch_texts(texts, batch_size=10, max_tokens_per_batch=50))
    >>> len(batches) >= 2
    True

    Notes
    -----
    Token counting uses count_tokens_approx (4 chars/token by default).
    For accurate batching, consider using a proper tokenizer.

    See Also
    --------
    count_tokens_approx : Token estimation function
    ProcessingPipeline.process_batch : Process batches through pipeline
    """
    batch: list[str] = []
    batch_tokens = 0

    for text in texts:
        text_tokens = count_tokens_approx(text)

        # Check if adding this text would exceed limits
        would_exceed_size = len(batch) >= batch_size
        would_exceed_tokens = (
            max_tokens_per_batch is not None
            and batch_tokens + text_tokens > max_tokens_per_batch
            and len(batch) > 0
        )

        if would_exceed_size or would_exceed_tokens:
            yield batch
            batch = []
            batch_tokens = 0

        batch.append(text)
        batch_tokens += text_tokens

    if batch:
        yield batch


def deduplicate_texts(
    texts: list[str],
    case_sensitive: bool = False,
) -> list[str]:
    """Remove duplicate texts while preserving order.

    Removes duplicate texts from a list, keeping the first occurrence
    of each unique text. Comparison can be case-sensitive or
    case-insensitive.

    Parameters
    ----------
    texts : list[str]
        List of texts potentially containing duplicates.
    case_sensitive : bool, default=False
        If True, "Hello" and "hello" are considered different.
        If False (default), they are considered duplicates.

    Returns
    -------
    list[str]
        List with duplicates removed, preserving original order
        and original casing of first occurrences.

    Examples
    --------
    Case-insensitive deduplication (default):

    >>> from insideLLMs.nlp.preprocessing import deduplicate_texts
    >>> deduplicate_texts(["Hello", "hello", "HELLO", "World"])
    ['Hello', 'World']

    Case-sensitive deduplication:

    >>> deduplicate_texts(["Hello", "hello", "World"], case_sensitive=True)
    ['Hello', 'hello', 'World']

    Preserve order:

    >>> deduplicate_texts(["c", "a", "b", "a", "c"])
    ['c', 'a', 'b']

    No duplicates:

    >>> deduplicate_texts(["a", "b", "c"])
    ['a', 'b', 'c']

    All duplicates:

    >>> deduplicate_texts(["same", "same", "same"])
    ['same']

    Empty input:

    >>> deduplicate_texts([])
    []

    See Also
    --------
    filter_by_length : Filter texts by length criteria
    """
    seen: set[str] = set()
    result = []

    for text in texts:
        key = text if case_sensitive else text.lower()
        if key not in seen:
            seen.add(key)
            result.append(text)

    return result


def filter_by_length(
    texts: list[str],
    min_length: int = 0,
    max_length: Optional[int] = None,
    unit: str = "chars",
) -> list[str]:
    """Filter texts by length criteria.

    Removes texts that don't meet length requirements. Length can
    be measured in characters, words, or approximate tokens.

    Parameters
    ----------
    texts : list[str]
        List of texts to filter.
    min_length : int, default=0
        Minimum length (inclusive). Texts shorter than this are removed.
    max_length : int, optional
        Maximum length (inclusive). If None, no upper limit.
        Texts longer than this are removed.
    unit : str, default="chars"
        Unit of measurement. One of:
        - "chars": Character count
        - "words": Whitespace-separated word count
        - "tokens": Approximate token count (4 chars/token)

    Returns
    -------
    list[str]
        Filtered list containing only texts meeting length criteria.

    Examples
    --------
    Filter by character count:

    >>> from insideLLMs.nlp.preprocessing import filter_by_length
    >>> texts = ["hi", "hello", "hello world"]
    >>> filter_by_length(texts, min_length=3)
    ['hello', 'hello world']

    Filter by maximum length:

    >>> filter_by_length(texts, max_length=5)
    ['hi', 'hello']

    Filter by word count:

    >>> filter_by_length(texts, min_length=2, unit="words")
    ['hello world']

    Filter by approximate tokens:

    >>> texts = ["short", "a bit longer text here"]
    >>> filter_by_length(texts, min_length=3, unit="tokens")
    ['a bit longer text here']

    Range filter:

    >>> texts = ["a", "abc", "abcde", "abcdefgh"]
    >>> filter_by_length(texts, min_length=2, max_length=5)
    ['abc', 'abcde']

    Empty result:

    >>> filter_by_length(["short"], min_length=1000)
    []

    See Also
    --------
    deduplicate_texts : Remove duplicate texts
    count_tokens_approx : Token estimation
    """

    def get_length(text: str) -> int:
        if unit == "chars":
            return len(text)
        elif unit == "words":
            return len(text.split())
        elif unit == "tokens":
            return count_tokens_approx(text)
        else:
            return len(text)

    result = []
    for text in texts:
        length = get_length(text)
        if length >= min_length and (max_length is None or length <= max_length):
            result.append(text)

    return result


def create_standard_pipeline() -> ProcessingPipeline:
    """Create a standard text preprocessing pipeline.

    Returns a pre-configured ProcessingPipeline with common
    preprocessing steps suitable for most LLM applications:
    - Text normalization (whitespace, unicode, control chars)
    - URL removal
    - HTML tag stripping
    - Markdown link conversion

    Returns
    -------
    ProcessingPipeline
        A configured pipeline ready for use.

    Examples
    --------
    Use the standard pipeline:

    >>> from insideLLMs.nlp.preprocessing import create_standard_pipeline
    >>> pipeline = create_standard_pipeline()
    >>> text = "  Visit <a href='url'>site</a> at https://example.com  "
    >>> pipeline.process(text)
    'Visit site at'

    Customize by disabling steps:

    >>> pipeline = create_standard_pipeline()
    >>> pipeline.disable_step("clean_urls")
    <...ProcessingPipeline...>
    >>> "https://example.com" in pipeline.process("Visit https://example.com")
    True

    List available steps:

    >>> pipeline = create_standard_pipeline()
    >>> [name for name, _ in pipeline.list_steps()]
    ['normalize', 'clean_urls', 'clean_html', 'convert_links']

    Process multiple documents:

    >>> pipeline = create_standard_pipeline()
    >>> docs = ["  Doc 1  ", "  Doc 2  "]
    >>> pipeline.process_batch(docs)
    ['Doc 1', 'Doc 2']

    Notes
    -----
    The pipeline includes these steps in order:
    1. "normalize": TextNormalizer with default settings
    2. "clean_urls": TextCleaner.remove_urls
    3. "clean_html": TextCleaner.remove_html_tags
    4. "convert_links": TextCleaner.convert_markdown_links

    See Also
    --------
    create_minimal_pipeline : Minimal whitespace-only pipeline
    ProcessingPipeline : Build custom pipelines
    """
    normalizer = TextNormalizer()

    return (
        ProcessingPipeline()
        .add_step("normalize", normalizer)
        .add_step("clean_urls", TextCleaner.remove_urls)
        .add_step("clean_html", TextCleaner.remove_html_tags)
        .add_step("convert_links", TextCleaner.convert_markdown_links)
    )


def create_minimal_pipeline() -> ProcessingPipeline:
    """Create a minimal preprocessing pipeline.

    Returns a lightweight pipeline that only normalizes whitespace.
    Useful when you need minimal processing to preserve most of
    the original text structure.

    Returns
    -------
    ProcessingPipeline
        A minimal pipeline with whitespace normalization only.

    Examples
    --------
    Use the minimal pipeline:

    >>> from insideLLMs.nlp.preprocessing import create_minimal_pipeline
    >>> pipeline = create_minimal_pipeline()
    >>> pipeline.process("  hello    world  ")
    'hello world'

    URLs and HTML are preserved:

    >>> pipeline = create_minimal_pipeline()
    >>> pipeline.process("Visit https://example.com")
    'Visit https://example.com'

    Add more steps if needed:

    >>> pipeline = create_minimal_pipeline()
    >>> pipeline.add_step("lower", str.lower)
    <...ProcessingPipeline...>
    >>> pipeline.process("  HELLO  ")
    'hello'

    List steps:

    >>> pipeline = create_minimal_pipeline()
    >>> pipeline.list_steps()
    [('whitespace', True)]

    Notes
    -----
    The pipeline includes only one step:
    1. "whitespace": normalize_whitespace function

    Use this as a starting point when you need to build a
    custom pipeline with specific steps.

    See Also
    --------
    create_standard_pipeline : Full-featured pipeline
    ProcessingPipeline : Build custom pipelines
    normalize_whitespace : The underlying function
    """
    return ProcessingPipeline().add_step("whitespace", normalize_whitespace)
