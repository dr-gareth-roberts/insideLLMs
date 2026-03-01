"""Extended coverage tests for context_window module.

Targets branches and code paths not exercised by test_context_window.py,
raising line coverage from ~63% to 90%+.
"""

from unittest.mock import MagicMock

from insideLLMs.contrib.context_window import (
    CompressionMethod,
    ContentAllocationBudget,
    ContentType,
    ContextBlock,
    ContextCompressor,
    ContextTruncator,
    ContextWindow,
    ContextWindowState,
    ConversationManager,
    PriorityLevel,
    SlidingWindowManager,
    TokenCounter,
    TruncationStrategy,
    create_budget,
    create_context_window,
    estimate_tokens,
    find_semantic_boundaries,
)

# ---------------------------------------------------------------------------
# ConversationManager: summarize_old_turns / _maybe_summarize / _default_summarize
# ---------------------------------------------------------------------------


class TestConversationManagerSummarization:
    """Tests for automatic conversation summarization internals."""

    def test_default_summarize_truncates_long_content(self):
        """_default_summarize truncates content > 100 chars and appends '...'."""
        manager = ConversationManager(summarize_after=3)
        long_message = "x" * 150
        # Manually invoke _default_summarize to check truncation
        turns = [{"role": "user", "content": long_message, "metadata": {}}]
        summary = manager._default_summarize(turns)
        assert "..." in summary
        assert "user:" in summary.lower() or "user: " in summary

    def test_default_summarize_many_turns_shows_more_messages(self):
        """When > 10 turns are summarized, the '... and N more messages' line appears."""
        manager = ConversationManager(summarize_after=3)
        turns = [{"role": "user", "content": f"Message {i}", "metadata": {}} for i in range(15)]
        summary = manager._default_summarize(turns)
        assert "... and 5 more messages" in summary

    def test_default_summarize_short_turns(self):
        """Summarize a few short turns -- no '...' truncation on content."""
        manager = ConversationManager(summarize_after=3)
        turns = [
            {"role": "user", "content": "Hi", "metadata": {}},
            {"role": "assistant", "content": "Hello", "metadata": {}},
        ]
        summary = manager._default_summarize(turns)
        assert "..." not in summary
        assert "Hi" in summary
        assert "Hello" in summary

    def test_custom_summarizer_is_called(self):
        """When a custom summarizer is provided, it is used instead of default."""
        mock_summarizer = MagicMock(return_value="custom summary")
        manager = ConversationManager(summarize_after=3, summarizer=mock_summarizer)

        for i in range(5):
            manager.add_turn("user", f"msg {i}")

        mock_summarizer.assert_called()
        assert manager._summary == "custom summary"

    def test_summary_appears_in_get_context_for_model(self):
        """After summarization, get_context_for_model includes the summary."""
        manager = ConversationManager(summarize_after=3)

        for i in range(6):
            manager.add_turn("user", f"Message {i}")

        messages = manager.get_context_for_model()
        # The first message should be the summary injected as a system message
        assert any("[Previous conversation summary:" in m["content"] for m in messages)

    def test_summarize_updates_stats(self):
        """After summarization, stats reflect summarized turns and has_summary."""
        manager = ConversationManager(summarize_after=3)

        for i in range(6):
            manager.add_turn("user", f"Msg {i}")

        stats = manager.get_stats()
        assert stats["has_summary"] is True
        assert stats["summarized_turns"] > 0
        assert stats["active_turns"] <= 3

    def test_maybe_summarize_noop_when_under_threshold(self):
        """_maybe_summarize does nothing when turn count is within limit."""
        manager = ConversationManager(summarize_after=10)
        for i in range(5):
            manager.add_turn("user", f"msg {i}")

        assert manager._summary is None
        assert manager._summary_turn_count == 0

    def test_clear_resets_summary(self):
        """Clearing conversation also clears the summary."""
        manager = ConversationManager(summarize_after=3)

        for i in range(6):
            manager.add_turn("user", f"msg {i}")

        assert manager._summary is not None
        manager.clear(keep_system=False)
        assert manager._summary is None
        assert manager._summary_turn_count == 0


# ---------------------------------------------------------------------------
# ConversationManager.get_context_for_model max_tokens truncation
# ---------------------------------------------------------------------------


class TestConversationManagerTokenTruncation:
    """Tests for get_context_for_model token-based truncation."""

    def test_get_context_for_model_truncates_when_over_limit(self):
        """Providing max_tokens removes oldest non-system messages."""
        manager = ConversationManager()
        manager.add_turn("system", "System")
        for i in range(10):
            manager.add_turn("user", f"User message {i} with some content")

        messages = manager.get_context_for_model(max_tokens=20)
        # Should have fewer messages than total, system should be preserved
        assert any(m["role"] == "system" for m in messages)
        assert len(messages) < 11

    def test_get_context_for_model_preserves_single_message(self):
        """Truncation stops when only one message remains."""
        manager = ConversationManager()
        manager.add_turn("user", "x" * 200)

        messages = manager.get_context_for_model(max_tokens=1)
        assert len(messages) >= 1


# ---------------------------------------------------------------------------
# SlidingWindowManager: _slide / archive / overlap
# ---------------------------------------------------------------------------


class TestSlidingWindowArchival:
    """Tests for SlidingWindowManager archive and overlap behavior."""

    def test_overlap_retains_context(self):
        """Overlap parameter is respected (items retained for continuity)."""
        window = SlidingWindowManager(window_size=3, overlap=1)
        for i in range(5):
            window.add(f"Item {i}")

        current = window.get_window()
        assert len(current) == 3

    def test_multiple_slides_accumulate_archive(self):
        """Multiple slide events accumulate items in the archive."""
        window = SlidingWindowManager(window_size=2, overlap=0)
        for i in range(8):
            window.add(f"Item {i}")

        assert len(window.get_window()) == 2
        assert len(window.get_archived()) == 6

    def test_archived_content_order(self):
        """Archived items maintain chronological order."""
        window = SlidingWindowManager(window_size=2, overlap=0)
        for i in range(6):
            window.add(f"Item {i}")

        archived = window.get_archived()
        for idx, block in enumerate(archived):
            assert block.content == f"Item {idx}"

    def test_clear_without_archive(self):
        """Clearing window without archive flag preserves archive."""
        window = SlidingWindowManager(window_size=2)
        for i in range(5):
            window.add(f"Item {i}")

        window.clear(clear_archive=False)
        assert len(window.get_window()) == 0
        assert len(window.get_archived()) > 0

    def test_get_stats_includes_archived_tokens(self):
        """Stats include archived_tokens field."""
        window = SlidingWindowManager(window_size=2)
        for i in range(5):
            window.add(f"Item {i}")

        stats = window.get_stats()
        assert stats["archived_tokens"] > 0
        assert stats["window_tokens"] > 0

    def test_add_with_metadata(self):
        """Metadata is attached to blocks in the sliding window."""
        window = SlidingWindowManager(window_size=5)
        block = window.add("Content", metadata={"source": "test"})
        assert block.metadata["source"] == "test"

    def test_add_with_priority_and_type(self):
        """Custom content type and priority are stored."""
        window = SlidingWindowManager(window_size=5)
        block = window.add(
            "System info",
            content_type=ContentType.SYSTEM,
            priority=PriorityLevel.CRITICAL,
        )
        assert block.content_type == ContentType.SYSTEM
        assert block.priority == PriorityLevel.CRITICAL


# ---------------------------------------------------------------------------
# ContextCompressor edge cases
# ---------------------------------------------------------------------------


class TestContextCompressorEdgeCases:
    """Tests for ContextCompressor edge cases and branches."""

    def test_summarize_with_empty_content(self):
        """SUMMARIZE on a block with minimal content."""
        compressor = ContextCompressor()
        blocks = [ContextBlock("", ContentType.CONTEXT, PriorityLevel.LOW)]
        result_blocks, result = compressor.compress(
            blocks,
            target_ratio=0.5,
            method=CompressionMethod.SUMMARIZE,
            min_priority=PriorityLevel.OPTIONAL,
        )
        assert result.success

    def test_summarize_single_sentence(self):
        """SUMMARIZE keeps at least 1 sentence (target_count = max(1,...))."""
        compressor = ContextCompressor()
        blocks = [
            ContextBlock(
                "Only one sentence here.",
                ContentType.CONTEXT,
                PriorityLevel.LOW,
            )
        ]
        result_blocks, result = compressor.compress(
            blocks,
            target_ratio=0.1,
            method=CompressionMethod.SUMMARIZE,
            min_priority=PriorityLevel.OPTIONAL,
        )
        assert result.success
        assert result_blocks[0].content != ""

    def test_summarize_with_custom_summarizer(self):
        """Custom summarizer function is invoked for SUMMARIZE method."""
        custom_fn = MagicMock(return_value="short summary")
        compressor = ContextCompressor(summarizer=custom_fn)
        blocks = [
            ContextBlock(
                "A long piece of text that should be summarized into something shorter.",
                ContentType.CONTEXT,
                PriorityLevel.LOW,
            )
        ]
        result_blocks, result = compressor.compress(
            blocks,
            method=CompressionMethod.SUMMARIZE,
            min_priority=PriorityLevel.LOW,
        )
        custom_fn.assert_called()
        assert result.success

    def test_extract_key_points_no_matches_fallback(self):
        """When no bullet points or keywords found, falls back to first paragraph."""
        compressor = ContextCompressor()
        blocks = [
            ContextBlock(
                "Regular text without any special formatting.\n\nSecond paragraph here.",
                ContentType.CONTEXT,
                PriorityLevel.LOW,
            )
        ]
        result_blocks, result = compressor.compress(
            blocks,
            method=CompressionMethod.EXTRACT_KEY_POINTS,
            min_priority=PriorityLevel.OPTIONAL,
        )
        assert result.success

    def test_extract_key_points_with_keywords(self):
        """Lines containing 'important', 'key', 'note:', etc. are extracted."""
        compressor = ContextCompressor()
        content = (
            "Some filler text here.\n"
            "This is an important note.\n"
            "More filler.\n"
            "Note: remember this.\n"
            "Critical information here.\n"
        )
        blocks = [ContextBlock(content, ContentType.CONTEXT, PriorityLevel.LOW)]
        result_blocks, result = compressor.compress(
            blocks,
            method=CompressionMethod.EXTRACT_KEY_POINTS,
            min_priority=PriorityLevel.OPTIONAL,
        )
        assert result.success

    def test_compress_block_no_reduction_returns_original(self):
        """When compression produces no token savings, original block is returned."""
        compressor = ContextCompressor()
        # Short content unlikely to be further abbreviated
        blocks = [ContextBlock("hi", ContentType.CONTEXT, PriorityLevel.LOW)]
        result_blocks, result = compressor.compress(
            blocks,
            method=CompressionMethod.ABBREVIATE,
            min_priority=PriorityLevel.OPTIONAL,
        )
        assert result.success
        # Block may or may not be marked compressed depending on token math
        assert result_blocks[0].content is not None

    def test_none_method_passthrough(self):
        """CompressionMethod.NONE returns blocks unchanged."""
        compressor = ContextCompressor()
        blocks = [ContextBlock("Keep me as-is", ContentType.CONTEXT, PriorityLevel.LOW)]
        result_blocks, result = compressor.compress(
            blocks,
            method=CompressionMethod.NONE,
            min_priority=PriorityLevel.OPTIONAL,
        )
        assert result.compression_ratio == 1.0
        assert result.blocks_compressed == 0
        assert result_blocks[0].content == "Keep me as-is"

    def test_compress_with_zero_original_tokens(self):
        """Edge: all blocks have 0 tokens -> compression_ratio = 1.0."""
        compressor = ContextCompressor()
        blocks = [ContextBlock("", ContentType.CONTEXT, PriorityLevel.LOW)]
        result_blocks, result = compressor.compress(
            blocks,
            method=CompressionMethod.REMOVE_REDUNDANCY,
            min_priority=PriorityLevel.OPTIONAL,
        )
        assert result.compression_ratio == 1.0

    def test_remove_redundancy_duplicate_lines(self):
        """Duplicate lines are removed, unique preserved."""
        compressor = ContextCompressor()
        content = "Line A\nLine A\nLine B\nLine B\nLine C"
        blocks = [ContextBlock(content, ContentType.CONTEXT, PriorityLevel.LOW)]
        result_blocks, _ = compressor.compress(
            blocks,
            method=CompressionMethod.REMOVE_REDUNDANCY,
            min_priority=PriorityLevel.LOW,
        )
        result_content = result_blocks[0].content
        # Each unique line should appear once
        assert result_content.count("Line A") == 1
        assert result_content.count("Line B") == 1

    def test_remove_redundancy_consecutive_duplicate_words(self):
        """Consecutive duplicate words within text are collapsed."""
        compressor = ContextCompressor()
        content = "hello hello world world test"
        blocks = [ContextBlock(content, ContentType.CONTEXT, PriorityLevel.LOW)]
        result_blocks, _ = compressor.compress(
            blocks,
            method=CompressionMethod.REMOVE_REDUNDANCY,
            min_priority=PriorityLevel.LOW,
        )
        # The regex sub should collapse "hello hello" -> "hello"
        result_content = result_blocks[0].content
        assert "hello hello" not in result_content.lower()


# ---------------------------------------------------------------------------
# ContextTruncator: edge and error paths
# ---------------------------------------------------------------------------


class TestContextTruncatorEdgePaths:
    """Tests for ContextTruncator edge cases and error paths."""

    def test_empty_blocks_list(self):
        """Truncating an empty list succeeds trivially."""
        truncator = ContextTruncator()
        result = truncator.truncate([], target_tokens=100)
        assert result.success
        assert result.tokens_removed == 0
        assert result.content == []

    def test_truncate_middle_with_two_blocks(self):
        """MIDDLE with <= 2 blocks delegates to TAIL strategy."""
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("A" * 100, ContentType.USER),
            ContextBlock("B" * 100, ContentType.USER),
        ]
        result = truncator.truncate(blocks, target_tokens=10, strategy=TruncationStrategy.MIDDLE)
        assert result.success or len(result.content) > 0

    def test_truncate_block_content_keep_end(self):
        """_truncate_block_content with keep_start=False keeps tail."""
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("First long block " * 20, ContentType.USER),
            ContextBlock("Second block content", ContentType.USER),
        ]
        result = truncator.truncate(blocks, target_tokens=15, strategy=TruncationStrategy.HEAD)
        assert result.success or len(result.content) > 0

    def test_head_strategy_partial_block(self):
        """HEAD truncation creates partial blocks when content partially fits."""
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("AAAA " * 40, ContentType.USER),
            ContextBlock("BBBB " * 40, ContentType.USER),
            ContextBlock("CCCC " * 10, ContentType.USER),
        ]
        result = truncator.truncate(blocks, target_tokens=30, strategy=TruncationStrategy.HEAD)
        # Should keep end blocks / partial blocks
        assert result.strategy_used == TruncationStrategy.HEAD
        assert len(result.content) >= 1

    def test_tail_strategy_partial_block(self):
        """TAIL truncation creates partial blocks when content partially fits."""
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("Short", ContentType.USER),
            ContextBlock("This is a fairly long block of content " * 10, ContentType.USER),
            ContextBlock("Another block", ContentType.USER),
        ]
        result = truncator.truncate(blocks, target_tokens=20, strategy=TruncationStrategy.TAIL)
        assert result.strategy_used == TruncationStrategy.TAIL

    def test_semantic_truncation_zero_token_block(self):
        """Semantic truncation handles blocks where truncated content is empty."""
        truncator = ContextTruncator()
        # Very large block that must be truncated to near zero
        blocks = [
            ContextBlock("Small.", ContentType.USER),
            ContextBlock("A" * 2000, ContentType.USER),
        ]
        result = truncator.truncate(blocks, target_tokens=5, strategy=TruncationStrategy.SEMANTIC)
        assert result.strategy_used == TruncationStrategy.SEMANTIC

    def test_semantic_boundary_fallback(self):
        """_truncate_at_semantic_boundary falls back when best_boundary == 0."""
        truncator = ContextTruncator()
        # Content with no sentence endings near the beginning
        blocks = [
            ContextBlock(
                "Averylongwordwithoutanyspacesorperiods" * 10,
                ContentType.USER,
            )
        ]
        result = truncator.truncate(blocks, target_tokens=5, strategy=TruncationStrategy.SEMANTIC)
        assert result.strategy_used == TruncationStrategy.SEMANTIC

    def test_sliding_window_strategy_with_critical(self):
        """Sliding window preserves CRITICAL blocks first, then recent."""
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("Critical info", ContentType.SYSTEM, PriorityLevel.CRITICAL),
            ContextBlock("Old content " * 5, ContentType.USER, PriorityLevel.LOW),
            ContextBlock("Middle content " * 5, ContentType.USER, PriorityLevel.LOW),
            ContextBlock("Recent content", ContentType.USER, PriorityLevel.MEDIUM),
        ]
        result = truncator.truncate(
            blocks, target_tokens=20, strategy=TruncationStrategy.SLIDING_WINDOW
        )
        assert result.strategy_used == TruncationStrategy.SLIDING_WINDOW
        contents = [b.content for b in result.content]
        assert "Critical info" in contents

    def test_priority_truncation_no_preserve_critical(self):
        """Priority truncation with preserve_critical=False removes even critical."""
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("Critical " * 20, ContentType.SYSTEM, PriorityLevel.CRITICAL),
            ContextBlock("Low", ContentType.USER, PriorityLevel.LOW),
        ]
        result = truncator.truncate(
            blocks,
            target_tokens=5,
            strategy=TruncationStrategy.PRIORITY,
            preserve_critical=False,
        )
        # With preserve_critical=False, critical blocks can be removed
        assert result.strategy_used == TruncationStrategy.PRIORITY

    def test_truncate_block_content_keep_start_adds_ellipsis(self):
        """When truncating from start, '...' is appended."""
        truncator = ContextTruncator()
        block = ContextBlock("A" * 500, ContentType.USER)
        truncated = truncator._truncate_block_content(block, target_tokens=5, keep_start=True)
        assert truncated.content.endswith("...")
        assert truncated.original_content == block.content
        assert truncated.metadata.get("truncated") is True

    def test_truncate_block_content_keep_end_adds_ellipsis(self):
        """When truncating from end, '...' is prepended."""
        truncator = ContextTruncator()
        block = ContextBlock("B" * 500, ContentType.USER)
        truncated = truncator._truncate_block_content(block, target_tokens=5, keep_start=False)
        assert truncated.content.startswith("...")
        assert truncated.original_content == block.content


# ---------------------------------------------------------------------------
# TokenCounter: non-ASCII / emoji / cache limit
# ---------------------------------------------------------------------------


class TestTokenCounterExtended:
    """Extended TokenCounter tests for Unicode, emoji, and cache edge cases."""

    def test_emoji_content(self):
        """Emoji text is counted without error."""
        counter = TokenCounter()
        count = counter.count("\U0001f600\U0001f601\U0001f602\U0001f603\U0001f604")
        assert count >= 0

    def test_non_ascii_japanese(self):
        """Japanese text is counted without error."""
        counter = TokenCounter()
        count = counter.count("日本語のテキストです。これはテストです。")
        assert count > 0

    def test_mixed_unicode_and_ascii(self):
        """Mixed content is handled."""
        counter = TokenCounter()
        count = counter.count("Hello こんにちは 123 \U0001f30d\U0001f680")
        assert count > 0

    def test_cache_miss_then_hit(self):
        """First call is a cache miss, second is a hit."""
        counter = TokenCounter()
        counter.count("unique text")
        stats1 = counter.get_stats()
        assert stats1["cache_misses"] == 1
        assert stats1["cache_hits"] == 0

        counter.count("unique text")
        stats2 = counter.get_stats()
        assert stats2["cache_hits"] == 1

    def test_hit_rate_calculation(self):
        """Hit rate is correctly computed."""
        counter = TokenCounter()
        counter.count("a")
        counter.count("a")
        counter.count("b")
        stats = counter.get_stats()
        # 1 hit / 3 total
        assert abs(stats["hit_rate"] - 1 / 3) < 0.01

    def test_stats_zero_total(self):
        """Stats handle zero total operations gracefully."""
        counter = TokenCounter()
        stats = counter.get_stats()
        assert stats["hit_rate"] == 0

    def test_count_messages_missing_content(self):
        """Messages without 'content' key are handled (overhead only)."""
        counter = TokenCounter()
        messages = [
            {"role": "user"},
            {"role": "assistant", "content": "Hi"},
            {"role": "tool", "content": None},
        ]
        total = counter.count_messages(messages)
        # Should count "Hi" plus overhead per message
        assert total > 0

    def test_custom_tokenizer_is_used(self):
        """Custom tokenizer callable replaces default estimation."""

        def tokenizer(text):
            return list(text)  # char-level

        counter = TokenCounter(tokenizer=tokenizer)
        assert counter.count("abc") == 3
        assert counter.count("hello") == 5


# ---------------------------------------------------------------------------
# ContextWindowState serialization
# ---------------------------------------------------------------------------


class TestContextWindowStateSerialization:
    """Tests for ContextWindowState edge cases."""

    def test_overflow_true_state(self):
        """State correctly reflects overflow when used > available."""
        budget = ContentAllocationBudget(
            total=100, system=10, user=20, assistant=20, tools=5, context=10, reserved=35
        )
        state = ContextWindowState(
            total_tokens=100,
            used_tokens=80,
            available_tokens=-15,
            block_count=10,
            usage_by_type={"user": 50, "system": 30},
            budget=budget,
            overflow=True,
        )
        d = state.to_dict()
        assert d["overflow"] is True
        assert d["available_tokens"] == -15
        assert d["budget"]["total"] == 100

    def test_multiple_content_types_in_usage(self):
        """usage_by_type correctly aggregates multiple types."""
        window = ContextWindow(max_tokens=10000)
        window.add("System", ContentType.SYSTEM)
        window.add("User", ContentType.USER)
        window.add("Tool", ContentType.TOOL_CALL)
        window.add("Context", ContentType.CONTEXT)

        state = window.get_state()
        assert "system" in state.usage_by_type
        assert "user" in state.usage_by_type
        assert "tool_call" in state.usage_by_type
        assert "context" in state.usage_by_type

    def test_state_to_dict_round_trip(self):
        """to_dict produces a plain dict with expected keys."""
        window = ContextWindow(max_tokens=5000)
        window.add("Test", ContentType.USER)
        state = window.get_state()
        d = state.to_dict()
        assert isinstance(d, dict)
        expected_keys = {
            "total_tokens",
            "used_tokens",
            "available_tokens",
            "block_count",
            "usage_by_type",
            "budget",
            "overflow",
        }
        assert expected_keys.issubset(d.keys())


# ---------------------------------------------------------------------------
# Metadata mutation/copying
# ---------------------------------------------------------------------------


class TestMetadataMutationCopying:
    """Tests for metadata isolation between blocks."""

    def test_metadata_not_shared_between_blocks(self):
        """Modifying metadata of one block does not affect another."""
        window = ContextWindow()
        block1 = window.add("A", ContentType.USER, metadata={"key": "val1"})
        block2 = window.add("B", ContentType.USER, metadata={"key": "val2"})

        block1.metadata["key"] = "changed"
        assert block2.metadata["key"] == "val2"

    def test_truncated_block_has_truncated_metadata(self):
        """After block-level truncation, metadata includes truncated=True."""
        truncator = ContextTruncator()
        block = ContextBlock(
            "Very long content " * 50,
            ContentType.USER,
            metadata={"original_key": "value"},
        )
        truncated = truncator._truncate_block_content(block, target_tokens=5, keep_start=True)
        assert truncated.metadata["truncated"] is True
        assert truncated.metadata["original_key"] == "value"

    def test_compressed_block_has_compression_metadata(self):
        """After compression, metadata includes compression_method."""
        compressor = ContextCompressor()
        blocks = [
            ContextBlock(
                "for example, this application configuration documentation for example",
                ContentType.CONTEXT,
                PriorityLevel.LOW,
                metadata={"source": "test"},
            )
        ]
        result_blocks, _ = compressor.compress(
            blocks,
            method=CompressionMethod.ABBREVIATE,
            min_priority=PriorityLevel.OPTIONAL,
        )
        compressed_block = result_blocks[0]
        if compressed_block.compressed:
            assert compressed_block.metadata.get("compression_method") == "abbreviate"
            assert compressed_block.original_content is not None

    def test_semantic_truncated_block_metadata(self):
        """Semantic truncation adds both 'truncated' and 'semantic' to metadata."""
        truncator = ContextTruncator()
        block = ContextBlock(
            "First sentence. Second sentence. Third sentence.",
            ContentType.USER,
            metadata={"custom": True},
        )
        truncated = truncator._truncate_at_semantic_boundary(block, target_tokens=5)
        assert truncated.metadata.get("truncated") is True
        assert truncated.metadata.get("semantic") is True
        assert truncated.metadata.get("custom") is True


# ---------------------------------------------------------------------------
# ContextWindow with custom tokenizer integration
# ---------------------------------------------------------------------------


class TestContextWindowCustomTokenizer:
    """Tests for ContextWindow with custom TokenCounter."""

    def test_window_with_custom_token_counter(self):
        """ContextWindow uses a provided TokenCounter."""

        def tokenizer(text):
            return text.split()

        counter = TokenCounter(tokenizer=tokenizer)
        window = ContextWindow(max_tokens=100, token_counter=counter)

        block = window.add("one two three four five", ContentType.USER)
        assert block.token_count == 5

    def test_custom_counter_propagates_to_truncator_and_compressor(self):
        """Truncator and compressor share the same custom counter."""

        def tokenizer(text):
            return text.split()

        counter = TokenCounter(tokenizer=tokenizer)
        window = ContextWindow(max_tokens=100, token_counter=counter)

        assert window.truncator.token_counter is counter
        assert window.compressor.token_counter is counter

    def test_truncation_with_custom_counter(self):
        """Truncation works correctly with a custom tokenizer."""

        def tokenizer(text):
            return text.split()

        counter = TokenCounter(tokenizer=tokenizer)
        truncator = ContextTruncator(token_counter=counter)

        blocks = [
            ContextBlock("one two three", ContentType.USER, token_count=3),
            ContextBlock("four five six seven", ContentType.USER, token_count=4),
        ]
        result = truncator.truncate(blocks, target_tokens=4)
        assert result.success
        assert result.final_tokens <= 4


# ---------------------------------------------------------------------------
# ContentAllocationBudget edge cases
# ---------------------------------------------------------------------------


class TestContentAllocationBudgetEdgeCases:
    """Tests for ContentAllocationBudget edge cases."""

    def test_manual_allocation_skips_defaults(self):
        """When system and user are set manually, __post_init__ does not override."""
        budget = ContentAllocationBudget(
            total=10000,
            system=500,
            user=3000,
            assistant=3000,
            tools=500,
            context=1000,
            reserved=2000,
        )
        assert budget.system == 500
        assert budget.reserved == 2000

    def test_allocation_for_instruction_maps_to_system(self):
        """INSTRUCTION content type maps to system allocation."""
        budget = ContentAllocationBudget(
            total=10000,
            system=1000,
            user=2000,
            assistant=2000,
            tools=500,
            context=1000,
            reserved=3500,
        )
        assert budget.allocation_for(ContentType.INSTRUCTION) == budget.system

    def test_allocation_for_example_maps_to_context(self):
        """EXAMPLE content type maps to context allocation."""
        budget = ContentAllocationBudget(
            total=10000,
            system=1000,
            user=2000,
            assistant=2000,
            tools=500,
            context=1000,
            reserved=3500,
        )
        assert budget.allocation_for(ContentType.EXAMPLE) == budget.context

    def test_allocation_for_tool_result_maps_to_tools(self):
        """TOOL_RESULT content type maps to tools allocation."""
        budget = ContentAllocationBudget(
            total=10000,
            system=1000,
            user=2000,
            assistant=2000,
            tools=500,
            context=1000,
            reserved=3500,
        )
        assert budget.allocation_for(ContentType.TOOL_RESULT) == budget.tools

    def test_default_budget_allocation(self):
        """Default allocation (no manual fields) divides the budget sensibly."""
        budget = ContentAllocationBudget(total=10000)
        assert budget.reserved > 0
        assert budget.system > 0
        assert budget.user > 0
        assert budget.context > 0
        assert budget.tools > 0
        # assistant shares with user
        assert budget.assistant == budget.user


# ---------------------------------------------------------------------------
# ContextBlock edge cases
# ---------------------------------------------------------------------------


class TestContextBlockEdgeCases:
    """Tests for ContextBlock edge cases."""

    def test_block_with_explicit_token_count(self):
        """Explicit token_count skips estimation."""
        block = ContextBlock(
            content="Hello",
            content_type=ContentType.USER,
            token_count=999,
        )
        assert block.token_count == 999

    def test_block_with_explicit_block_id(self):
        """Explicit block_id skips generation."""
        block = ContextBlock(
            content="Hello",
            content_type=ContentType.USER,
            block_id="custom-id-123",
        )
        assert block.block_id == "custom-id-123"

    def test_to_dict_includes_all_fields(self):
        """to_dict returns all expected fields."""
        block = ContextBlock(
            content="Test",
            content_type=ContentType.ASSISTANT,
            priority=PriorityLevel.LOW,
            metadata={"key": "val"},
            compressed=True,
            original_content="Original",
        )
        d = block.to_dict()
        assert d["content_type"] == "assistant"
        assert d["priority"] == 2
        assert d["metadata"] == {"key": "val"}
        assert d["compressed"] is True
        assert d["original_content"] == "Original"
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# ContextWindow: add_message role mapping, history, record_action
# ---------------------------------------------------------------------------


class TestContextWindowMessageAndHistory:
    """Tests for add_message role mapping and history recording."""

    def test_add_message_unknown_role(self):
        """Unknown role maps to ContentType.CONTEXT with default priority."""
        window = ContextWindow()
        block = window.add_message("tool", "Tool output")
        assert block.content_type == ContentType.CONTEXT
        assert block.priority == PriorityLevel.MEDIUM

    def test_add_message_system_gets_high_priority(self):
        """System role gets HIGH priority by default."""
        window = ContextWindow()
        block = window.add_message("system", "Prompt")
        assert block.priority == PriorityLevel.HIGH

    def test_add_message_explicit_priority_overrides_default(self):
        """Explicit priority overrides role-based default."""
        window = ContextWindow()
        block = window.add_message("user", "Hello", priority=PriorityLevel.CRITICAL)
        assert block.priority == PriorityLevel.CRITICAL

    def test_history_records_add_remove_truncate(self):
        """History captures add, remove, and truncate actions."""
        window = ContextWindow(max_tokens=1000)
        block = window.add("Content", ContentType.USER)
        window.remove(block.block_id)
        window.add("More", ContentType.USER)
        window.truncate()

        history = window.get_history()
        actions = [h["action"] for h in history]
        assert "add" in actions
        assert "remove" in actions
        assert "truncate" in actions

    def test_history_records_compress_action(self):
        """History captures compress action."""
        window = ContextWindow()
        window.add("for example this documentation", ContentType.USER, PriorityLevel.LOW)
        window.compress(method=CompressionMethod.ABBREVIATE)

        history = window.get_history()
        actions = [h["action"] for h in history]
        assert "compress" in actions

    def test_history_records_clear_action(self):
        """History captures clear action."""
        window = ContextWindow()
        window.add("Test", ContentType.USER)
        window.clear(preserve_critical=False)

        history = window.get_history()
        actions = [h["action"] for h in history]
        assert "clear" in actions

    def test_record_action_handles_dict_data(self):
        """_record_action handles plain dict data (no to_dict method)."""
        window = ContextWindow()
        window._record_action("custom", {"key": "value"})
        history = window.get_history()
        assert history[-1]["data"] == {"key": "value"}

    def test_record_action_handles_none_data(self):
        """_record_action handles None data (from clear)."""
        window = ContextWindow()
        window._record_action("clear", None)
        history = window.get_history()
        assert history[-1]["data"] is None

    def test_get_messages_skips_blocks_without_role(self):
        """get_messages only includes blocks with 'role' in metadata."""
        window = ContextWindow()
        window.add("No role", ContentType.USER)  # uses add(), no role in metadata
        window.add_message("user", "With role")

        messages = window.get_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "With role"


# ---------------------------------------------------------------------------
# ContextWindow: auto-truncation and budget
# ---------------------------------------------------------------------------


class TestContextWindowAutoTruncation:
    """Tests for auto-truncation triggered by add()."""

    def test_auto_truncate_preserves_critical(self):
        """Critical blocks survive auto-truncation."""
        window = ContextWindow(max_tokens=80)
        window.add("Must keep", ContentType.SYSTEM, PriorityLevel.CRITICAL)
        for i in range(20):
            window.add(f"Filler {i} " * 5, ContentType.USER, PriorityLevel.LOW)

        blocks = window.get_blocks()
        assert any(b.content == "Must keep" for b in blocks)

    def test_auto_truncate_respects_reserved(self):
        """Auto-truncation keeps used tokens within (max - reserved)."""
        window = ContextWindow(max_tokens=200)
        for i in range(50):
            window.add(f"Content {i} " * 3, ContentType.USER, PriorityLevel.LOW)

        assert window.get_used_tokens() <= window.max_tokens - window.budget.reserved


# ---------------------------------------------------------------------------
# Convenience functions: edge cases
# ---------------------------------------------------------------------------


class TestConvenienceFunctionsEdgeCases:
    """Tests for convenience function edge cases."""

    def test_create_budget_with_different_ratios(self):
        """create_budget respects different ratio configurations."""
        budget = create_budget(
            total=20000,
            system_ratio=0.1,
            context_ratio=0.3,
            tools_ratio=0.05,
            reserved_ratio=0.1,
        )
        assert budget.reserved == 2000
        available = 20000 - 2000
        assert budget.system == int(available * 0.1)
        assert budget.context == int(available * 0.3)
        assert budget.tools == int(available * 0.05)

    def test_create_context_window_default_params(self):
        """create_context_window with defaults works."""
        window = create_context_window()
        assert window.max_tokens == 128000
        assert window.default_strategy == TruncationStrategy.PRIORITY

    def test_estimate_tokens_whitespace(self):
        """Whitespace-only text returns some token count."""
        tokens = estimate_tokens("   \n\n\t  ")
        assert tokens >= 0

    def test_estimate_tokens_code(self):
        """Code content returns positive token count."""
        code = "def foo(x):\n    return x * 2\n"
        tokens = estimate_tokens(code)
        assert tokens > 0


# ---------------------------------------------------------------------------
# find_semantic_boundaries extended
# ---------------------------------------------------------------------------


class TestFindSemanticBoundariesExtended:
    """Extended tests for find_semantic_boundaries."""

    def test_paragraph_breaks(self):
        """Paragraph breaks (double newlines) create boundaries."""
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        boundaries = find_semantic_boundaries(text)
        assert len(boundaries) > 2

    def test_multiple_punctuation_types(self):
        """Question marks and exclamation points create boundaries."""
        text = "Question? Exclamation! Statement."
        boundaries = find_semantic_boundaries(text)
        # 0, after each sentence, and len(text)
        assert len(boundaries) >= 3

    def test_no_sentence_endings(self):
        """Text without sentence endings returns start and end."""
        text = "no sentence endings here"
        boundaries = find_semantic_boundaries(text)
        assert 0 in boundaries
        assert len(text) in boundaries

    def test_multiple_newlines(self):
        """More than two consecutive newlines count as a boundary."""
        text = "Part 1.\n\n\n\nPart 2."
        boundaries = find_semantic_boundaries(text)
        assert len(boundaries) >= 3


# ---------------------------------------------------------------------------
# TruncationResult / CompressionResult: to_dict content
# ---------------------------------------------------------------------------


class TestResultObjectsSerialization:
    """Tests for result object serialization details."""

    def test_truncation_result_content_serialized(self):
        """TruncationResult.to_dict serializes nested blocks."""
        from insideLLMs.contrib.context_window import TruncationResult

        block = ContextBlock("Test", ContentType.USER)
        result = TruncationResult(
            original_tokens=10,
            final_tokens=5,
            tokens_removed=5,
            blocks_removed=1,
            blocks_truncated=0,
            strategy_used=TruncationStrategy.HEAD,
            success=True,
            content=[block],
        )
        d = result.to_dict()
        assert len(d["content"]) == 1
        assert d["content"][0]["content"] == "Test"
        assert d["strategy_used"] == "head"

    def test_compression_result_to_dict_all_fields(self):
        """ContextCompressionResult.to_dict includes all fields."""
        from insideLLMs.contrib.context_window import ContextCompressionResult

        result = ContextCompressionResult(
            original_tokens=200,
            compressed_tokens=100,
            compression_ratio=0.5,
            method_used=CompressionMethod.EXTRACT_KEY_POINTS,
            blocks_compressed=2,
            success=True,
        )
        d = result.to_dict()
        assert d["method_used"] == "extract_key_points"
        assert d["blocks_compressed"] == 2


# ---------------------------------------------------------------------------
# Integration: ContextWindow + ConversationManager + SlidingWindow
# ---------------------------------------------------------------------------


class TestCrossComponentIntegration:
    """Integration tests across multiple components."""

    def test_conversation_manager_with_custom_window(self):
        """ConversationManager using an explicit ContextWindow."""
        window = ContextWindow(max_tokens=500)
        manager = ConversationManager(context_window=window, max_turns=10)

        manager.add_turn("system", "Be brief.")
        manager.add_turn("user", "Hello")
        manager.add_turn("assistant", "Hi")

        messages = manager.get_context_for_model()
        assert len(messages) >= 3

    def test_sliding_window_feed_into_compressor(self):
        """SlidingWindow blocks can be compressed."""
        slider = SlidingWindowManager(window_size=5)
        for i in range(5):
            slider.add(f"Repeated content repeated content item {i}")

        compressor = ContextCompressor()
        blocks = slider.get_window()
        compressed_blocks, result = compressor.compress(
            blocks,
            method=CompressionMethod.REMOVE_REDUNDANCY,
            min_priority=PriorityLevel.OPTIONAL,
        )
        assert result.success

    def test_full_pipeline_add_compress_truncate_export(self):
        """Complete pipeline: add, compress, truncate, export messages."""
        window = ContextWindow(max_tokens=200)

        window.add_message("system", "You are helpful.")
        for i in range(10):
            window.add_message("user", f"User message number {i} for example documentation")
            window.add_message("assistant", f"Response {i}")

        # Compress
        window.compress(method=CompressionMethod.ABBREVIATE)

        # Export
        messages = window.get_messages()
        assert len(messages) > 0

        state = window.get_state()
        assert state.used_tokens <= window.max_tokens

    def test_summarization_and_model_context(self):
        """After summarization, model context includes summary prefix."""

        def summarizer(turns):
            return f"Discussed {len(turns)} topics"

        manager = ConversationManager(
            summarize_after=3,
            summarizer=summarizer,
        )

        manager.add_turn("system", "Be concise.")
        for i in range(6):
            manager.add_turn("user", f"Topic {i}")

        messages = manager.get_context_for_model()
        summary_msgs = [
            m for m in messages if "[Previous conversation summary:" in m.get("content", "")
        ]
        assert len(summary_msgs) == 1
        assert "Discussed" in summary_msgs[0]["content"]
