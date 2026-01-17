"""Data preprocessing and text normalization utilities.

This module provides tools for cleaning, normalizing, and transforming
text data before feeding to LLMs or analyzing their outputs.

Key features:
- Text normalization (unicode, whitespace, casing)
- Data cleaning pipelines
- Tokenization helpers
- Batch processing utilities
- Input validation and sanitization
"""

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Pattern,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)


T = TypeVar("T")


class NormalizationLevel(Enum):
    """Levels of text normalization."""

    NONE = "none"
    MINIMAL = "minimal"  # Just whitespace
    STANDARD = "standard"  # Whitespace + unicode
    AGGRESSIVE = "aggressive"  # All normalizations


@dataclass
class TextStats:
    """Statistics about a text.

    Attributes:
        char_count: Total character count.
        word_count: Word count (whitespace-separated).
        sentence_count: Approximate sentence count.
        line_count: Number of lines.
        avg_word_length: Average word length.
        unique_words: Number of unique words.
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

        Args:
            text: The text to analyze.

        Returns:
            TextStats object.
        """
        if not text:
            return cls()

        words = text.split()
        unique = set(w.lower() for w in words)

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
    """Normalize text with configurable options.

    Example:
        >>> normalizer = TextNormalizer()
        >>> normalizer.normalize("  Hello   World!  ")
        "Hello World!"
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
        """Initialize normalizer.

        Args:
            lowercase: Convert to lowercase.
            strip: Strip leading/trailing whitespace.
            collapse_whitespace: Collapse multiple spaces to one.
            remove_extra_newlines: Remove redundant blank lines.
            unicode_normalize: Apply Unicode normalization.
            unicode_form: Unicode normalization form (NFC, NFD, NFKC, NFKD).
            remove_control_chars: Remove control characters.
            remove_zero_width: Remove zero-width characters.
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
        """Normalize text according to settings.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text.
        """
        if not text:
            return text

        result = text

        # Unicode normalization first
        if self.unicode_normalize:
            result = unicodedata.normalize(self.unicode_form, result)

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
        """Allow using normalizer as a function."""
        return self.normalize(text)


class TextCleaner:
    """Clean text by removing unwanted patterns.

    Example:
        >>> cleaner = TextCleaner()
        >>> cleaner.remove_urls("Visit https://example.com for more")
        "Visit  for more"
    """

    # Common patterns
    URL_PATTERN = re.compile(
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
    )
    EMAIL_PATTERN = re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    )
    PHONE_PATTERN = re.compile(
        r"\+?[\d\s\-().]{7,}"
    )
    HTML_TAG_PATTERN = re.compile(
        r"<[^>]+>"
    )
    MARKDOWN_LINK_PATTERN = re.compile(
        r"\[([^\]]+)\]\([^)]+\)"
    )
    EMOJI_PATTERN = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002702-\U000027B0\U0001F900-\U0001F9FF]+"
    )

    @classmethod
    def remove_urls(cls, text: str, replacement: str = "") -> str:
        """Remove URLs from text.

        Args:
            text: Input text.
            replacement: Replacement string.

        Returns:
            Text with URLs removed.
        """
        return cls.URL_PATTERN.sub(replacement, text)

    @classmethod
    def remove_emails(cls, text: str, replacement: str = "") -> str:
        """Remove email addresses from text.

        Args:
            text: Input text.
            replacement: Replacement string.

        Returns:
            Text with emails removed.
        """
        return cls.EMAIL_PATTERN.sub(replacement, text)

    @classmethod
    def remove_phone_numbers(cls, text: str, replacement: str = "") -> str:
        """Remove phone numbers from text.

        Args:
            text: Input text.
            replacement: Replacement string.

        Returns:
            Text with phone numbers removed.
        """
        return cls.PHONE_PATTERN.sub(replacement, text)

    @classmethod
    def remove_html_tags(cls, text: str) -> str:
        """Remove HTML tags from text.

        Args:
            text: Input text.

        Returns:
            Text with HTML tags removed.
        """
        return cls.HTML_TAG_PATTERN.sub("", text)

    @classmethod
    def convert_markdown_links(cls, text: str) -> str:
        """Convert markdown links to plain text.

        Args:
            text: Input text.

        Returns:
            Text with link text only.
        """
        return cls.MARKDOWN_LINK_PATTERN.sub(r"\1", text)

    @classmethod
    def remove_emojis(cls, text: str, replacement: str = "") -> str:
        """Remove emojis from text.

        Args:
            text: Input text.
            replacement: Replacement string.

        Returns:
            Text with emojis removed.
        """
        return cls.EMOJI_PATTERN.sub(replacement, text)

    @classmethod
    def remove_punctuation(
        cls, text: str, keep: str = "", replacement: str = ""
    ) -> str:
        """Remove punctuation from text.

        Args:
            text: Input text.
            keep: Punctuation characters to keep.
            replacement: Replacement string.

        Returns:
            Text with punctuation removed.
        """
        import string
        punct = set(string.punctuation) - set(keep)
        for p in punct:
            text = text.replace(p, replacement)
        return text

    @classmethod
    def remove_numbers(cls, text: str, replacement: str = "") -> str:
        """Remove numbers from text.

        Args:
            text: Input text.
            replacement: Replacement string.

        Returns:
            Text with numbers removed.
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
        """Mask personally identifiable information.

        Args:
            text: Input text.
            mask_email: Mask email addresses.
            mask_phone: Mask phone numbers.
            mask_url: Mask URLs.
            mask_char: Replacement string.

        Returns:
            Text with PII masked.
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
    """Split text into chunks for processing.

    Example:
        >>> splitter = TextSplitter(chunk_size=100, overlap=20)
        >>> chunks = splitter.split("Long text...")
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        separator: str = "\n\n",
        keep_separator: bool = True,
    ):
        """Initialize splitter.

        Args:
            chunk_size: Maximum chunk size in characters.
            overlap: Overlap between chunks in characters.
            separator: Preferred split point.
            keep_separator: Keep separator in chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator
        self.keep_separator = keep_separator

    def split(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: Text to split.

        Returns:
            List of chunks.
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
                    if self.keep_separator:
                        end = sep_pos + len(self.separator)
                    else:
                        end = sep_pos

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

    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs.

        Args:
            text: Text to split.

        Returns:
            List of paragraphs.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]


@dataclass
class ProcessingStep:
    """A single processing step in a pipeline.

    Attributes:
        name: Step name.
        func: Processing function.
        enabled: Whether step is enabled.
    """

    name: str
    func: Callable[[str], str]
    enabled: bool = True

    def apply(self, text: str) -> str:
        """Apply this step if enabled."""
        if self.enabled:
            return self.func(text)
        return text


class ProcessingPipeline:
    """A configurable text processing pipeline.

    Example:
        >>> pipeline = ProcessingPipeline()
        >>> pipeline.add_step("normalize", TextNormalizer())
        >>> pipeline.add_step("clean_urls", TextCleaner.remove_urls)
        >>> result = pipeline.process("Some text with https://url.com")
    """

    def __init__(self):
        """Initialize empty pipeline."""
        self._steps: List[ProcessingStep] = []

    def add_step(
        self,
        name: str,
        func: Callable[[str], str],
        enabled: bool = True,
    ) -> "ProcessingPipeline":
        """Add a processing step.

        Args:
            name: Step name.
            func: Processing function.
            enabled: Whether step is enabled.

        Returns:
            Self for chaining.
        """
        self._steps.append(ProcessingStep(name=name, func=func, enabled=enabled))
        return self

    def enable_step(self, name: str) -> "ProcessingPipeline":
        """Enable a step by name.

        Args:
            name: Step name.

        Returns:
            Self for chaining.
        """
        for step in self._steps:
            if step.name == name:
                step.enabled = True
                break
        return self

    def disable_step(self, name: str) -> "ProcessingPipeline":
        """Disable a step by name.

        Args:
            name: Step name.

        Returns:
            Self for chaining.
        """
        for step in self._steps:
            if step.name == name:
                step.enabled = False
                break
        return self

    def process(self, text: str) -> str:
        """Process text through all enabled steps.

        Args:
            text: Input text.

        Returns:
            Processed text.
        """
        result = text
        for step in self._steps:
            result = step.apply(result)
        return result

    def process_batch(self, texts: Iterable[str]) -> List[str]:
        """Process multiple texts.

        Args:
            texts: Iterable of texts.

        Returns:
            List of processed texts.
        """
        return [self.process(text) for text in texts]

    def list_steps(self) -> List[Tuple[str, bool]]:
        """List all steps with their enabled status.

        Returns:
            List of (name, enabled) tuples.
        """
        return [(step.name, step.enabled) for step in self._steps]

    def __call__(self, text: str) -> str:
        """Allow using pipeline as a function."""
        return self.process(text)


class DataValidator:
    """Validate data before processing.

    Example:
        >>> validator = DataValidator()
        >>> validator.add_rule("non_empty", lambda x: len(x) > 0)
        >>> validator.validate("test")
        (True, [])
    """

    def __init__(self):
        """Initialize with no rules."""
        self._rules: Dict[str, Tuple[Callable[[str], bool], str]] = {}

    def add_rule(
        self,
        name: str,
        rule: Callable[[str], bool],
        error_msg: Optional[str] = None,
    ) -> "DataValidator":
        """Add a validation rule.

        Args:
            name: Rule name.
            rule: Validation function.
            error_msg: Custom error message.

        Returns:
            Self for chaining.
        """
        self._rules[name] = (rule, error_msg or f"Failed rule: {name}")
        return self

    def validate(self, text: str) -> Tuple[bool, List[str]]:
        """Validate text against all rules.

        Args:
            text: Text to validate.

        Returns:
            Tuple of (is_valid, list of errors).
        """
        errors = []
        for name, (rule, error_msg) in self._rules.items():
            try:
                if not rule(text):
                    errors.append(error_msg)
            except Exception as e:
                errors.append(f"Rule '{name}' error: {e}")
        return len(errors) == 0, errors

    def validate_batch(
        self, texts: Iterable[str]
    ) -> List[Tuple[int, bool, List[str]]]:
        """Validate multiple texts.

        Args:
            texts: Iterable of texts.

        Returns:
            List of (index, is_valid, errors) tuples.
        """
        results = []
        for i, text in enumerate(texts):
            is_valid, errors = self.validate(text)
            results.append((i, is_valid, errors))
        return results


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    Collapses multiple spaces and trims.

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """
    return " ".join(text.split())


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize Unicode in text.

    Args:
        text: Input text.
        form: Unicode normalization form.

    Returns:
        Normalized text.
    """
    return unicodedata.normalize(form, text)


def remove_special_chars(text: str, keep: str = "") -> str:
    """Remove special characters, keeping only alphanumeric and specified.

    Args:
        text: Input text.
        keep: Additional characters to keep.

    Returns:
        Cleaned text.
    """
    pattern = f"[^a-zA-Z0-9\\s{re.escape(keep)}]"
    return re.sub(pattern, "", text)


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "...",
    word_boundary: bool = True,
) -> str:
    """Truncate text to maximum length.

    Args:
        text: Input text.
        max_length: Maximum length.
        suffix: Suffix to add when truncated.
        word_boundary: Try to break at word boundary.

    Returns:
        Truncated text.
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
    """Approximate token count.

    Args:
        text: Input text.
        chars_per_token: Average chars per token.

    Returns:
        Estimated token count.
    """
    return int(len(text) / chars_per_token)


def batch_texts(
    texts: List[str],
    batch_size: int,
    max_tokens_per_batch: Optional[int] = None,
) -> Iterator[List[str]]:
    """Batch texts for processing.

    Args:
        texts: List of texts.
        batch_size: Maximum items per batch.
        max_tokens_per_batch: Maximum tokens per batch.

    Yields:
        Batches of texts.
    """
    batch: List[str] = []
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
    texts: List[str],
    case_sensitive: bool = False,
) -> List[str]:
    """Remove duplicate texts.

    Args:
        texts: List of texts.
        case_sensitive: Whether comparison is case-sensitive.

    Returns:
        List with duplicates removed (preserves order).
    """
    seen: Set[str] = set()
    result = []

    for text in texts:
        key = text if case_sensitive else text.lower()
        if key not in seen:
            seen.add(key)
            result.append(text)

    return result


def filter_by_length(
    texts: List[str],
    min_length: int = 0,
    max_length: Optional[int] = None,
    unit: str = "chars",
) -> List[str]:
    """Filter texts by length.

    Args:
        texts: List of texts.
        min_length: Minimum length.
        max_length: Maximum length (None for no limit).
        unit: "chars" or "words" or "tokens".

    Returns:
        Filtered list.
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
        if length >= min_length:
            if max_length is None or length <= max_length:
                result.append(text)

    return result


def create_standard_pipeline() -> ProcessingPipeline:
    """Create a standard text preprocessing pipeline.

    Returns:
        Configured ProcessingPipeline.
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

    Returns:
        Configured ProcessingPipeline with minimal steps.
    """
    return (
        ProcessingPipeline()
        .add_step("whitespace", normalize_whitespace)
    )
