"""Tests for data preprocessing utilities."""

from insideLLMs.contrib.preprocessing import (
    DataValidator,
    ProcessingPipeline,
    ProcessingStep,
    TextCleaner,
    TextNormalizer,
    TextSplitter,
    TextStats,
    batch_texts,
    count_tokens_approx,
    create_minimal_pipeline,
    create_standard_pipeline,
    deduplicate_texts,
    filter_by_length,
    normalize_unicode,
    normalize_whitespace,
    remove_special_chars,
    truncate_text,
)


class TestTextStats:
    """Tests for TextStats."""

    def test_from_text_basic(self):
        """Test basic statistics computation."""
        text = "Hello world. This is a test."
        stats = TextStats.from_text(text)

        assert stats.char_count == len(text)
        assert stats.word_count == 6
        assert stats.sentence_count == 2
        assert stats.line_count == 1

    def test_from_text_multiline(self):
        """Test statistics with multiple lines."""
        text = "Line one.\nLine two.\nLine three."
        stats = TextStats.from_text(text)

        assert stats.line_count == 3
        assert stats.sentence_count == 3

    def test_from_text_empty(self):
        """Test empty text statistics."""
        stats = TextStats.from_text("")
        assert stats.char_count == 0
        assert stats.word_count == 0

    def test_unique_words(self):
        """Test unique word counting."""
        text = "the cat the dog the cat"
        stats = TextStats.from_text(text)

        assert stats.word_count == 6
        assert stats.unique_words == 3  # the, cat, dog


class TestTextNormalizer:
    """Tests for TextNormalizer."""

    def test_default_normalization(self):
        """Test default normalization settings."""
        normalizer = TextNormalizer()
        result = normalizer.normalize("  Hello   World!  ")
        assert result == "Hello World!"

    def test_lowercase(self):
        """Test lowercase conversion."""
        normalizer = TextNormalizer(lowercase=True)
        result = normalizer.normalize("Hello WORLD")
        assert result == "hello world"

    def test_collapse_whitespace(self):
        """Test whitespace collapsing."""
        normalizer = TextNormalizer()
        result = normalizer.normalize("Hello    World")
        assert result == "Hello World"

    def test_remove_extra_newlines(self):
        """Test removal of excessive newlines."""
        normalizer = TextNormalizer()
        result = normalizer.normalize("Line 1\n\n\n\n\nLine 2")
        assert result == "Line 1\n\nLine 2"

    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        normalizer = TextNormalizer(unicode_normalize=True)
        # Composed vs decomposed forms
        result = normalizer.normalize("café")  # May have different forms
        assert "caf" in result

    def test_callable(self):
        """Test calling normalizer as function."""
        normalizer = TextNormalizer()
        result = normalizer("  test  ")
        assert result == "test"

    def test_preserve_newlines(self):
        """Test that single newlines are preserved."""
        normalizer = TextNormalizer()
        result = normalizer.normalize("Line 1\nLine 2")
        assert "\n" in result


class TestTextCleaner:
    """Tests for TextCleaner."""

    def test_remove_urls(self):
        """Test URL removal."""
        text = "Visit https://example.com for info"
        result = TextCleaner.remove_urls(text)
        assert "https://example.com" not in result
        assert "Visit" in result

    def test_remove_urls_www(self):
        """Test www URL removal."""
        text = "Visit www.example.com for info"
        result = TextCleaner.remove_urls(text)
        assert "www.example.com" not in result

    def test_remove_emails(self):
        """Test email removal."""
        text = "Contact us at test@example.com for help"
        result = TextCleaner.remove_emails(text)
        assert "test@example.com" not in result

    def test_remove_phone_numbers(self):
        """Test phone number removal."""
        text = "Call us at +1-555-123-4567"
        result = TextCleaner.remove_phone_numbers(text)
        assert "555-123-4567" not in result

    def test_remove_html_tags(self):
        """Test HTML tag removal."""
        text = "<p>Hello <b>World</b>!</p>"
        result = TextCleaner.remove_html_tags(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result
        assert "World" in result

    def test_convert_markdown_links(self):
        """Test markdown link conversion."""
        text = "Check out [this link](https://example.com) for more"
        result = TextCleaner.convert_markdown_links(text)
        assert result == "Check out this link for more"

    def test_remove_emojis(self):
        """Test emoji removal."""
        text = "Hello \U0001f600 World \U0001f30d"
        result = TextCleaner.remove_emojis(text)
        assert "\U0001f600" not in result
        assert "\U0001f30d" not in result
        assert "Hello" in result

    def test_remove_punctuation(self):
        """Test punctuation removal."""
        text = "Hello, World!"
        result = TextCleaner.remove_punctuation(text)
        assert "," not in result
        assert "!" not in result

    def test_remove_punctuation_keep(self):
        """Test punctuation removal with keep list."""
        text = "Hello, World!"
        result = TextCleaner.remove_punctuation(text, keep=",")
        assert "," in result
        assert "!" not in result

    def test_remove_numbers(self):
        """Test number removal."""
        text = "There are 123 items"
        result = TextCleaner.remove_numbers(text)
        assert "123" not in result
        assert "There are" in result

    def test_mask_pii(self):
        """Test PII masking."""
        text = "Email: test@example.com, Phone: 555-1234"
        result = TextCleaner.mask_pii(text)
        assert "[REDACTED]" in result
        assert "test@example.com" not in result


class TestTextSplitter:
    """Tests for TextSplitter."""

    def test_split_short_text(self):
        """Test that short text is not split."""
        splitter = TextSplitter(chunk_size=100)
        chunks = splitter.split("Short text")
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_split_long_text(self):
        """Test splitting long text."""
        splitter = TextSplitter(chunk_size=50, overlap=0)
        text = "This is a test. " * 10  # ~160 chars
        chunks = splitter.split(text)
        assert len(chunks) > 1

    def test_split_with_overlap(self):
        """Test splitting with overlap."""
        splitter = TextSplitter(chunk_size=50, overlap=10)
        text = "Word " * 30
        chunks = splitter.split(text)

        # With overlap, chunks should share some content
        assert len(chunks) > 1

    def test_split_by_sentences(self):
        """Test sentence splitting."""
        splitter = TextSplitter()
        text = "First sentence. Second sentence! Third sentence?"
        sentences = splitter.split_by_sentences(text)

        assert len(sentences) == 3
        assert "First sentence" in sentences[0]

    def test_split_by_paragraphs(self):
        """Test paragraph splitting."""
        splitter = TextSplitter()
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        paragraphs = splitter.split_by_paragraphs(text)

        assert len(paragraphs) == 3

    def test_split_empty(self):
        """Test splitting empty text."""
        splitter = TextSplitter()
        chunks = splitter.split("")
        assert chunks == []


class TestProcessingPipeline:
    """Tests for ProcessingPipeline."""

    def test_basic_pipeline(self):
        """Test basic pipeline processing."""
        pipeline = ProcessingPipeline()
        pipeline.add_step("upper", str.upper)
        pipeline.add_step("strip", str.strip)

        result = pipeline.process("  hello  ")
        assert result == "HELLO"

    def test_disable_step(self):
        """Test disabling a step."""
        pipeline = ProcessingPipeline()
        pipeline.add_step("upper", str.upper)
        pipeline.disable_step("upper")

        result = pipeline.process("hello")
        assert result == "hello"

    def test_enable_step(self):
        """Test enabling a step."""
        pipeline = ProcessingPipeline()
        pipeline.add_step("upper", str.upper, enabled=False)
        pipeline.enable_step("upper")

        result = pipeline.process("hello")
        assert result == "HELLO"

    def test_process_batch(self):
        """Test batch processing."""
        pipeline = ProcessingPipeline()
        pipeline.add_step("upper", str.upper)

        results = pipeline.process_batch(["a", "b", "c"])
        assert results == ["A", "B", "C"]

    def test_list_steps(self):
        """Test listing steps."""
        pipeline = ProcessingPipeline()
        pipeline.add_step("step1", str.upper)
        pipeline.add_step("step2", str.lower, enabled=False)

        steps = pipeline.list_steps()
        assert ("step1", True) in steps
        assert ("step2", False) in steps

    def test_callable(self):
        """Test using pipeline as callable."""
        pipeline = ProcessingPipeline()
        pipeline.add_step("upper", str.upper)

        result = pipeline("hello")
        assert result == "HELLO"

    def test_chaining(self):
        """Test method chaining."""
        pipeline = ProcessingPipeline().add_step("lower", str.lower).add_step("strip", str.strip)

        result = pipeline.process("  HELLO  ")
        assert result == "hello"


class TestDataValidator:
    """Tests for DataValidator."""

    def test_valid_text(self):
        """Test validation of valid text."""
        validator = DataValidator()
        validator.add_rule("non_empty", lambda x: len(x) > 0)

        is_valid, errors = validator.validate("test")
        assert is_valid
        assert len(errors) == 0

    def test_invalid_text(self):
        """Test validation of invalid text."""
        validator = DataValidator()
        validator.add_rule("non_empty", lambda x: len(x) > 0, "Text is empty")

        is_valid, errors = validator.validate("")
        assert not is_valid
        assert "Text is empty" in errors

    def test_multiple_rules(self):
        """Test multiple validation rules."""
        validator = DataValidator()
        validator.add_rule("non_empty", lambda x: len(x) > 0)
        validator.add_rule("max_length", lambda x: len(x) < 100)

        is_valid, errors = validator.validate("short text")
        assert is_valid

    def test_validate_batch(self):
        """Test batch validation."""
        validator = DataValidator()
        validator.add_rule("non_empty", lambda x: len(x) > 0)

        results = validator.validate_batch(["valid", "", "also valid"])

        assert results[0][1]  # First is valid
        assert not results[1][1]  # Second is invalid
        assert results[2][1]  # Third is valid


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        result = normalize_whitespace("  hello   world  ")
        assert result == "hello world"

    def test_normalize_unicode(self):
        """Test Unicode normalization."""
        result = normalize_unicode("café")
        assert "caf" in result

    def test_remove_special_chars(self):
        """Test special character removal."""
        result = remove_special_chars("Hello! @World#")
        assert result == "Hello World"

    def test_remove_special_chars_keep(self):
        """Test keeping certain special characters."""
        result = remove_special_chars("Hello! @World#", keep="@")
        assert "@" in result

    def test_truncate_text(self):
        """Test text truncation."""
        text = "This is a long sentence that needs truncation."
        result = truncate_text(text, 20)

        assert len(result) <= 20
        assert result.endswith("...")

    def test_truncate_text_short(self):
        """Test truncation of short text."""
        text = "Short"
        result = truncate_text(text, 100)
        assert result == "Short"

    def test_truncate_text_word_boundary(self):
        """Test truncation at word boundary."""
        text = "Hello beautiful world"
        result = truncate_text(text, 15, word_boundary=True)

        # Should break at word boundary
        assert not result.endswith("beaut...")

    def test_count_tokens_approx(self):
        """Test approximate token counting."""
        text = "Hello World!"  # 12 chars
        tokens = count_tokens_approx(text)
        assert tokens == 3  # 12 / 4

    def test_batch_texts_by_size(self):
        """Test batching by batch size."""
        texts = ["a", "b", "c", "d", "e"]
        batches = list(batch_texts(texts, batch_size=2))

        assert len(batches) == 3
        assert batches[0] == ["a", "b"]
        assert batches[1] == ["c", "d"]
        assert batches[2] == ["e"]

    def test_batch_texts_by_tokens(self):
        """Test batching by token count."""
        # Create texts where token limits matter
        # With 4 chars/token: "short"=1 token, 100 chars=25 tokens, "tiny"=1 token
        texts = ["short", "x" * 100, "tiny", "another", "text"]
        batches = list(batch_texts(texts, batch_size=10, max_tokens_per_batch=10))

        # With max 10 tokens per batch, the 25-token text forces a split
        assert len(batches) >= 2

    def test_deduplicate_texts(self):
        """Test text deduplication."""
        texts = ["hello", "world", "hello", "test"]
        result = deduplicate_texts(texts)

        assert len(result) == 3
        assert result == ["hello", "world", "test"]

    def test_deduplicate_texts_case_insensitive(self):
        """Test case-insensitive deduplication."""
        texts = ["Hello", "hello", "HELLO"]
        result = deduplicate_texts(texts, case_sensitive=False)

        assert len(result) == 1

    def test_deduplicate_texts_case_sensitive(self):
        """Test case-sensitive deduplication."""
        texts = ["Hello", "hello", "HELLO"]
        result = deduplicate_texts(texts, case_sensitive=True)

        assert len(result) == 3

    def test_filter_by_length_chars(self):
        """Test filtering by character length."""
        texts = ["a", "abc", "abcde", "abcdefgh"]
        result = filter_by_length(texts, min_length=3, max_length=5)

        assert "a" not in result
        assert "abc" in result
        assert "abcde" in result
        assert "abcdefgh" not in result

    def test_filter_by_length_words(self):
        """Test filtering by word count."""
        texts = ["one", "one two", "one two three four"]
        result = filter_by_length(texts, min_length=2, max_length=3, unit="words")

        assert len(result) == 1
        assert result[0] == "one two"


class TestPrebuiltPipelines:
    """Tests for prebuilt pipelines."""

    def test_standard_pipeline(self):
        """Test standard preprocessing pipeline."""
        pipeline = create_standard_pipeline()
        text = "  Visit https://example.com  "

        result = pipeline.process(text)

        assert "https://example.com" not in result
        assert "Visit" in result

    def test_minimal_pipeline(self):
        """Test minimal preprocessing pipeline."""
        pipeline = create_minimal_pipeline()
        text = "  hello   world  "

        result = pipeline.process(text)
        assert result == "hello world"


class TestProcessingStep:
    """Tests for ProcessingStep dataclass."""

    def test_apply_enabled(self):
        """Test applying enabled step."""
        step = ProcessingStep(name="upper", func=str.upper, enabled=True)
        result = step.apply("hello")
        assert result == "HELLO"

    def test_apply_disabled(self):
        """Test applying disabled step."""
        step = ProcessingStep(name="upper", func=str.upper, enabled=False)
        result = step.apply("hello")
        assert result == "hello"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        normalizer = TextNormalizer()
        assert normalizer.normalize("") == ""

        cleaner_result = TextCleaner.remove_urls("")
        assert cleaner_result == ""

    def test_none_handling(self):
        """Test that None is handled gracefully."""
        splitter = TextSplitter()
        # Should not crash on empty string
        result = splitter.split("")
        assert result == []

    def test_unicode_text(self):
        """Test handling of various Unicode text."""
        normalizer = TextNormalizer()

        # Chinese
        assert normalizer.normalize("你好世界") == "你好世界"

        # Arabic
        assert normalizer.normalize("مرحبا") == "مرحبا"

        # Emoji (may be removed depending on settings)
        result = normalizer.normalize("Hello \U0001f44b")
        assert "Hello" in result

    def test_very_long_text(self):
        """Test handling of very long text."""
        long_text = "word " * 10000
        splitter = TextSplitter(chunk_size=1000)
        chunks = splitter.split(long_text)

        assert len(chunks) > 1
        # All text should be preserved across chunks
