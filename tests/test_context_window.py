"""Tests for context window management module."""

from insideLLMs.context_window import (
    CompressionMethod,
    CompressionResult,
    ContentType,
    ContextBlock,
    ContextCompressor,
    ContextTruncator,
    ContextWindow,
    ConversationManager,
    PriorityLevel,
    SlidingWindowManager,
    # Dataclasses
    TokenBudget,
    TokenCounter,
    TruncationResult,
    # Enums
    TruncationStrategy,
    compress_context,
    create_budget,
    create_context_window,
    create_conversation_manager,
    create_sliding_window,
    estimate_context_tokens,
    # Functions
    estimate_tokens,
    find_semantic_boundaries,
    truncate_context,
)


class TestEnums:
    """Tests for enum types."""

    def test_truncation_strategy_values(self):
        assert TruncationStrategy.TAIL.value == "tail"
        assert TruncationStrategy.HEAD.value == "head"
        assert TruncationStrategy.MIDDLE.value == "middle"
        assert TruncationStrategy.SEMANTIC.value == "semantic"
        assert TruncationStrategy.PRIORITY.value == "priority"
        assert TruncationStrategy.SLIDING_WINDOW.value == "sliding_window"

    def test_content_type_values(self):
        assert ContentType.SYSTEM.value == "system"
        assert ContentType.USER.value == "user"
        assert ContentType.ASSISTANT.value == "assistant"
        assert ContentType.TOOL_CALL.value == "tool_call"
        assert ContentType.TOOL_RESULT.value == "tool_result"

    def test_compression_method_values(self):
        assert CompressionMethod.NONE.value == "none"
        assert CompressionMethod.SUMMARIZE.value == "summarize"
        assert CompressionMethod.REMOVE_REDUNDANCY.value == "remove_redundancy"

    def test_priority_level_ordering(self):
        assert PriorityLevel.CRITICAL.value > PriorityLevel.HIGH.value
        assert PriorityLevel.HIGH.value > PriorityLevel.MEDIUM.value
        assert PriorityLevel.MEDIUM.value > PriorityLevel.LOW.value
        assert PriorityLevel.LOW.value > PriorityLevel.OPTIONAL.value


class TestTokenBudget:
    """Tests for TokenBudget."""

    def test_token_budget_creation(self):
        budget = TokenBudget(total=10000)
        assert budget.total == 10000
        assert budget.reserved > 0
        assert budget.system > 0

    def test_token_budget_custom_allocation(self):
        budget = TokenBudget(
            total=10000,
            system=1000,
            user=3000,
            assistant=3000,
            tools=500,
            context=1500,
            reserved=1000,
        )
        assert budget.system == 1000
        assert budget.user == 3000
        assert budget.reserved == 1000

    def test_token_budget_remaining(self):
        budget = TokenBudget(
            total=10000,
            system=1000,
            user=2000,
            assistant=2000,
            tools=500,
            context=1000,
            reserved=2000,
        )
        current_usage = {"system": 500, "user": 1000}
        remaining = budget.remaining(current_usage)
        assert remaining == 10000 - 2000 - 1500  # total - reserved - used

    def test_token_budget_allocation_for(self):
        budget = TokenBudget(
            total=10000,
            system=1000,
            user=2000,
            assistant=2000,
            tools=500,
            context=1000,
            reserved=1000,
        )
        assert budget.allocation_for(ContentType.SYSTEM) == 1000
        assert budget.allocation_for(ContentType.USER) == 2000
        assert budget.allocation_for(ContentType.TOOL_CALL) == 500

    def test_token_budget_to_dict(self):
        budget = TokenBudget(total=10000)
        d = budget.to_dict()
        assert "total" in d
        assert "system" in d
        assert "reserved" in d
        assert d["total"] == 10000


class TestContextBlock:
    """Tests for ContextBlock."""

    def test_context_block_creation(self):
        block = ContextBlock(
            content="Hello world",
            content_type=ContentType.USER,
        )
        assert block.content == "Hello world"
        assert block.content_type == ContentType.USER
        assert block.token_count > 0
        assert block.block_id != ""

    def test_context_block_with_priority(self):
        block = ContextBlock(
            content="Important message",
            content_type=ContentType.SYSTEM,
            priority=PriorityLevel.CRITICAL,
        )
        assert block.priority == PriorityLevel.CRITICAL

    def test_context_block_with_metadata(self):
        block = ContextBlock(
            content="Test",
            content_type=ContentType.CONTEXT,
            metadata={"source": "file", "line": 42},
        )
        assert block.metadata["source"] == "file"
        assert block.metadata["line"] == 42

    def test_context_block_to_dict(self):
        block = ContextBlock(
            content="Test content",
            content_type=ContentType.USER,
            priority=PriorityLevel.HIGH,
        )
        d = block.to_dict()
        assert d["content"] == "Test content"
        assert d["content_type"] == "user"
        assert d["priority"] == 4
        assert "block_id" in d

    def test_context_block_compressed_tracking(self):
        block = ContextBlock(
            content="Compressed",
            content_type=ContentType.CONTEXT,
            compressed=True,
            original_content="This was the original longer content",
        )
        assert block.compressed is True
        assert block.original_content is not None


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_simple(self):
        tokens = estimate_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10

    def test_estimate_tokens_longer_text(self):
        text = "This is a longer piece of text that should have more tokens."
        tokens = estimate_tokens(text)
        assert tokens > 5

    def test_find_semantic_boundaries(self):
        text = "First sentence. Second sentence!\n\nNew paragraph."
        boundaries = find_semantic_boundaries(text)
        assert 0 in boundaries
        assert len(text) in boundaries
        assert len(boundaries) > 2

    def test_find_semantic_boundaries_empty(self):
        boundaries = find_semantic_boundaries("")
        assert 0 in boundaries


class TestTokenCounter:
    """Tests for TokenCounter."""

    def test_token_counter_count(self):
        counter = TokenCounter()
        count = counter.count("Hello world")
        assert count > 0

    def test_token_counter_caching(self):
        counter = TokenCounter()
        text = "Test text for caching"

        count1 = counter.count(text)
        count2 = counter.count(text)

        assert count1 == count2
        stats = counter.get_stats()
        assert stats["cache_hits"] >= 1

    def test_token_counter_count_messages(self):
        counter = TokenCounter()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        count = counter.count_messages(messages)
        assert count > 0

    def test_token_counter_empty(self):
        counter = TokenCounter()
        assert counter.count("") == 0

    def test_token_counter_clear_cache(self):
        counter = TokenCounter()
        counter.count("Test")
        counter.clear_cache()
        stats = counter.get_stats()
        assert stats["cache_size"] == 0

    def test_token_counter_custom_tokenizer(self):
        def custom_tokenizer(text):
            return text.split()

        counter = TokenCounter(tokenizer=custom_tokenizer)
        count = counter.count("one two three")
        assert count == 3


class TestContextTruncator:
    """Tests for ContextTruncator."""

    def test_truncator_no_truncation_needed(self):
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("Short", ContentType.USER),
        ]
        result = truncator.truncate(blocks, 1000)
        assert result.success
        assert result.tokens_removed == 0
        assert result.blocks_removed == 0

    def test_truncator_priority_strategy(self):
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("Low priority", ContentType.USER, PriorityLevel.LOW),
            ContextBlock("High priority", ContentType.USER, PriorityLevel.HIGH),
            ContextBlock("Critical", ContentType.SYSTEM, PriorityLevel.CRITICAL),
        ]

        result = truncator.truncate(blocks, 20, TruncationStrategy.PRIORITY)
        assert result.success
        # Critical should be preserved
        assert any(b.content == "Critical" for b in result.content)

    def test_truncator_tail_strategy(self):
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("First block with content", ContentType.USER),
            ContextBlock("Second block with content", ContentType.USER),
            ContextBlock("Third block with content", ContentType.USER),
        ]

        result = truncator.truncate(blocks, 30, TruncationStrategy.TAIL)
        # Should keep beginning
        assert len(result.content) > 0

    def test_truncator_head_strategy(self):
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("First block with content", ContentType.USER),
            ContextBlock("Second block with content", ContentType.USER),
            ContextBlock("Third block with content", ContentType.USER),
        ]

        result = truncator.truncate(blocks, 30, TruncationStrategy.HEAD)
        # Should keep end
        assert len(result.content) > 0

    def test_truncator_middle_strategy(self):
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("First", ContentType.USER),
            ContextBlock("Second", ContentType.USER),
            ContextBlock("Third", ContentType.USER),
            ContextBlock("Fourth", ContentType.USER),
            ContextBlock("Fifth", ContentType.USER),
        ]

        result = truncator.truncate(blocks, 15, TruncationStrategy.MIDDLE)
        assert result.success

    def test_truncator_semantic_strategy(self):
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("First sentence. Second sentence. Third sentence.", ContentType.USER),
        ]

        result = truncator.truncate(blocks, 15, TruncationStrategy.SEMANTIC)
        assert result.success

    def test_truncator_sliding_window(self):
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("Old content", ContentType.USER),
            ContextBlock("Middle content", ContentType.USER),
            ContextBlock("Recent content", ContentType.USER),
        ]

        result = truncator.truncate(blocks, 15, TruncationStrategy.SLIDING_WINDOW)
        assert result.success

    def test_truncator_preserve_critical(self):
        truncator = ContextTruncator()
        blocks = [
            ContextBlock("Critical content", ContentType.SYSTEM, PriorityLevel.CRITICAL),
            ContextBlock("Normal content " * 10, ContentType.USER, PriorityLevel.LOW),
        ]

        result = truncator.truncate(blocks, 20, preserve_critical=True)
        # Critical should always be preserved
        assert any(b.content == "Critical content" for b in result.content)


class TestContextCompressor:
    """Tests for ContextCompressor."""

    def test_compressor_no_compression(self):
        compressor = ContextCompressor()
        blocks = [ContextBlock("Test content", ContentType.USER)]

        result_blocks, result = compressor.compress(blocks, method=CompressionMethod.NONE)

        assert result.compression_ratio == 1.0
        assert result.blocks_compressed == 0

    def test_compressor_remove_redundancy(self):
        compressor = ContextCompressor()
        blocks = [
            ContextBlock(
                "This line appears\nThis line appears\nUnique line",
                ContentType.USER,
                priority=PriorityLevel.LOW,
            )
        ]

        result_blocks, result = compressor.compress(
            blocks,
            method=CompressionMethod.REMOVE_REDUNDANCY,
            min_priority=PriorityLevel.OPTIONAL,
        )

        assert result.success

    def test_compressor_abbreviate(self):
        compressor = ContextCompressor()
        blocks = [
            ContextBlock(
                "for example this is documentation for the application",
                ContentType.USER,
                priority=PriorityLevel.LOW,
            )
        ]

        result_blocks, result = compressor.compress(
            blocks,
            method=CompressionMethod.ABBREVIATE,
            min_priority=PriorityLevel.OPTIONAL,
        )

        assert result.success

    def test_compressor_summarize(self):
        compressor = ContextCompressor()
        blocks = [
            ContextBlock(
                "First sentence. Second sentence. Third sentence. Fourth sentence.",
                ContentType.USER,
                priority=PriorityLevel.LOW,
            )
        ]

        result_blocks, result = compressor.compress(
            blocks,
            target_ratio=0.5,
            method=CompressionMethod.SUMMARIZE,
            min_priority=PriorityLevel.OPTIONAL,
        )

        assert result.success

    def test_compressor_extract_key_points(self):
        compressor = ContextCompressor()
        blocks = [
            ContextBlock(
                "Regular text\n- Important point one\n- Important point two\nMore text",
                ContentType.USER,
                priority=PriorityLevel.LOW,
            )
        ]

        result_blocks, result = compressor.compress(
            blocks,
            method=CompressionMethod.EXTRACT_KEY_POINTS,
            min_priority=PriorityLevel.OPTIONAL,
        )

        assert result.success

    def test_compressor_respects_priority(self):
        compressor = ContextCompressor()
        blocks = [
            ContextBlock("High priority", ContentType.SYSTEM, PriorityLevel.HIGH),
            ContextBlock("Low priority content to compress", ContentType.USER, PriorityLevel.LOW),
        ]

        result_blocks, result = compressor.compress(
            blocks,
            method=CompressionMethod.ABBREVIATE,
            min_priority=PriorityLevel.MEDIUM,
        )

        # High priority block should not be compressed
        high_block = next(b for b in result_blocks if "High priority" in b.content)
        assert not high_block.compressed


class TestContextWindow:
    """Tests for ContextWindow."""

    def test_context_window_creation(self):
        window = ContextWindow(max_tokens=10000)
        assert window.max_tokens == 10000
        assert len(window.get_blocks()) == 0

    def test_context_window_add(self):
        window = ContextWindow()
        block = window.add("Test content", ContentType.USER)

        assert block.content == "Test content"
        assert len(window.get_blocks()) == 1

    def test_context_window_add_message(self):
        window = ContextWindow()
        block = window.add_message("user", "Hello")

        assert block.content == "Hello"
        assert block.content_type == ContentType.USER

    def test_context_window_remove(self):
        window = ContextWindow()
        block = window.add("Test", ContentType.USER)
        block_id = block.block_id

        assert window.remove(block_id)
        assert len(window.get_blocks()) == 0

    def test_context_window_remove_nonexistent(self):
        window = ContextWindow()
        assert not window.remove("nonexistent")

    def test_context_window_clear(self):
        window = ContextWindow()
        window.add("Test 1", ContentType.USER)
        window.add("Test 2", ContentType.USER)

        window.clear(preserve_critical=False)
        assert len(window.get_blocks()) == 0

    def test_context_window_clear_preserve_critical(self):
        window = ContextWindow()
        window.add("Critical", ContentType.SYSTEM, PriorityLevel.CRITICAL)
        window.add("Normal", ContentType.USER)

        window.clear(preserve_critical=True)

        blocks = window.get_blocks()
        assert len(blocks) == 1
        assert blocks[0].content == "Critical"

    def test_context_window_truncate(self):
        window = ContextWindow(max_tokens=100)

        # Add content that exceeds limit
        for i in range(10):
            window.add(f"Content block {i} " * 5, ContentType.USER)

        result = window.truncate(target_tokens=50)
        assert result.success
        assert window.get_used_tokens() <= 50

    def test_context_window_compress(self):
        window = ContextWindow()
        window.add("for example this is documentation", ContentType.USER, PriorityLevel.LOW)

        result = window.compress(method=CompressionMethod.ABBREVIATE)
        assert result.success

    def test_context_window_get_blocks_filtered(self):
        window = ContextWindow()
        window.add("System", ContentType.SYSTEM)
        window.add("User", ContentType.USER)
        window.add("Assistant", ContentType.ASSISTANT)

        user_blocks = window.get_blocks(content_type=ContentType.USER)
        assert len(user_blocks) == 1
        assert user_blocks[0].content == "User"

    def test_context_window_get_blocks_by_priority(self):
        window = ContextWindow()
        window.add("Low", ContentType.USER, PriorityLevel.LOW)
        window.add("High", ContentType.USER, PriorityLevel.HIGH)
        window.add("Critical", ContentType.SYSTEM, PriorityLevel.CRITICAL)

        high_blocks = window.get_blocks(min_priority=PriorityLevel.HIGH)
        assert len(high_blocks) == 2

    def test_context_window_get_content(self):
        window = ContextWindow()
        window.add("First", ContentType.USER)
        window.add("Second", ContentType.USER)

        content = window.get_content(separator="\n")
        assert "First" in content
        assert "Second" in content

    def test_context_window_get_messages(self):
        window = ContextWindow()
        window.add_message("user", "Hello")
        window.add_message("assistant", "Hi")

        messages = window.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_context_window_get_used_tokens(self):
        window = ContextWindow()
        window.add("Test content", ContentType.USER)

        used = window.get_used_tokens()
        assert used > 0

    def test_context_window_get_available_tokens(self):
        window = ContextWindow(max_tokens=10000)
        initial = window.get_available_tokens()

        window.add("Test content", ContentType.USER)

        assert window.get_available_tokens() < initial

    def test_context_window_get_state(self):
        window = ContextWindow(max_tokens=10000)
        window.add("Test", ContentType.USER)

        state = window.get_state()

        assert state.total_tokens == 10000
        assert state.used_tokens > 0
        assert state.block_count == 1
        assert "user" in state.usage_by_type

    def test_context_window_state_to_dict(self):
        window = ContextWindow()
        window.add("Test", ContentType.USER)

        state = window.get_state()
        d = state.to_dict()

        assert "total_tokens" in d
        assert "used_tokens" in d
        assert "budget" in d

    def test_context_window_history(self):
        window = ContextWindow()
        window.add("Test", ContentType.USER)

        history = window.get_history()
        assert len(history) > 0
        assert history[0]["action"] == "add"

    def test_context_window_auto_truncate(self):
        window = ContextWindow(max_tokens=100)

        # Add enough content to trigger auto-truncate
        for i in range(20):
            window.add(f"Content {i} " * 10, ContentType.USER, PriorityLevel.LOW)

        # Should have auto-truncated
        assert window.get_used_tokens() <= 100


class TestConversationManager:
    """Tests for ConversationManager."""

    def test_conversation_manager_creation(self):
        manager = ConversationManager()
        assert len(manager.get_turns()) == 0

    def test_conversation_manager_add_turn(self):
        manager = ConversationManager()
        turn = manager.add_turn("user", "Hello")

        assert turn["role"] == "user"
        assert turn["content"] == "Hello"
        assert turn["turn_number"] == 1

    def test_conversation_manager_multiple_turns(self):
        manager = ConversationManager()
        manager.add_turn("user", "Hello")
        manager.add_turn("assistant", "Hi there")
        manager.add_turn("user", "How are you?")

        turns = manager.get_turns()
        assert len(turns) == 3

    def test_conversation_manager_get_turns_limit(self):
        manager = ConversationManager()
        for i in range(10):
            manager.add_turn("user", f"Message {i}")

        turns = manager.get_turns(limit=3)
        assert len(turns) == 3

    def test_conversation_manager_get_context_for_model(self):
        manager = ConversationManager()
        manager.add_turn("system", "You are helpful")
        manager.add_turn("user", "Hello")
        manager.add_turn("assistant", "Hi")

        messages = manager.get_context_for_model()
        assert len(messages) == 3
        assert messages[0]["role"] == "system"

    def test_conversation_manager_clear(self):
        manager = ConversationManager()
        manager.add_turn("user", "Hello")
        manager.add_turn("assistant", "Hi")

        manager.clear(keep_system=False)
        assert len(manager.get_turns()) == 0

    def test_conversation_manager_clear_keep_system(self):
        manager = ConversationManager()
        manager.add_turn("system", "System message")
        manager.add_turn("user", "Hello")

        manager.clear(keep_system=True)

        turns = manager.get_turns()
        assert len(turns) == 1
        assert turns[0]["role"] == "system"

    def test_conversation_manager_summarization(self):
        manager = ConversationManager(summarize_after=5)

        # Add more turns than summarize_after
        for i in range(10):
            manager.add_turn("user", f"Message {i}")

        # Should have summarized older messages
        stats = manager.get_stats()
        assert stats["active_turns"] <= 5

    def test_conversation_manager_get_stats(self):
        manager = ConversationManager()
        manager.add_turn("user", "Hello")

        stats = manager.get_stats()

        assert "total_turns" in stats
        assert "active_turns" in stats
        assert stats["total_turns"] == 1

    def test_conversation_manager_with_metadata(self):
        manager = ConversationManager()
        turn = manager.add_turn("user", "Hello", metadata={"source": "web"})

        assert turn["metadata"]["source"] == "web"


class TestSlidingWindowManager:
    """Tests for SlidingWindowManager."""

    def test_sliding_window_creation(self):
        window = SlidingWindowManager(window_size=5)
        assert window.window_size == 5

    def test_sliding_window_add(self):
        window = SlidingWindowManager(window_size=5)
        block = window.add("Test content")

        assert block.content == "Test content"
        assert len(window.get_window()) == 1

    def test_sliding_window_slides(self):
        window = SlidingWindowManager(window_size=3)

        for i in range(5):
            window.add(f"Content {i}")

        # Should only have last 3
        current = window.get_window()
        assert len(current) == 3

        # Should have archived 2
        archived = window.get_archived()
        assert len(archived) == 2

    def test_sliding_window_get_content(self):
        window = SlidingWindowManager(window_size=3)
        window.add("First")
        window.add("Second")

        content = window.get_content()
        assert "First" in content
        assert "Second" in content

    def test_sliding_window_get_content_with_archived(self):
        window = SlidingWindowManager(window_size=2)

        for i in range(4):
            window.add(f"Content {i}")

        content = window.get_content(include_archived=True)
        assert "Content 0" in content
        assert "Content 3" in content

    def test_sliding_window_get_tokens(self):
        window = SlidingWindowManager()
        window.add("Test content")

        tokens = window.get_tokens()
        assert tokens > 0

    def test_sliding_window_clear(self):
        window = SlidingWindowManager()
        window.add("Test")

        window.clear()
        assert len(window.get_window()) == 0

    def test_sliding_window_clear_with_archive(self):
        window = SlidingWindowManager(window_size=2)

        for i in range(5):
            window.add(f"Content {i}")

        window.clear(clear_archive=True)

        assert len(window.get_window()) == 0
        assert len(window.get_archived()) == 0

    def test_sliding_window_get_stats(self):
        window = SlidingWindowManager(window_size=3)

        for i in range(5):
            window.add(f"Content {i}")

        stats = window.get_stats()

        assert stats["window_size"] == 3
        assert stats["current_items"] == 3
        assert stats["archived_items"] == 2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_context_window(self):
        window = create_context_window(max_tokens=50000)
        assert window.max_tokens == 50000

    def test_create_context_window_with_strategy(self):
        window = create_context_window(strategy=TruncationStrategy.SEMANTIC)
        assert window.default_strategy == TruncationStrategy.SEMANTIC

    def test_estimate_context_tokens(self):
        tokens = estimate_context_tokens("Hello world test")
        assert tokens > 0

    def test_truncate_context(self):
        blocks = [
            ContextBlock("First block content", ContentType.USER),
            ContextBlock("Second block content", ContentType.USER),
        ]

        result = truncate_context(blocks, 10)
        assert result.success

    def test_compress_context(self):
        blocks = [
            ContextBlock(
                "for example this documentation",
                ContentType.USER,
                PriorityLevel.LOW,
            ),
        ]

        result_blocks, result = compress_context(
            blocks,
            method=CompressionMethod.ABBREVIATE,
        )

        assert result.success

    def test_create_budget(self):
        budget = create_budget(
            total=10000,
            system_ratio=0.2,
            reserved_ratio=0.25,
        )

        assert budget.total == 10000
        assert budget.reserved == 2500

    def test_create_conversation_manager(self):
        manager = create_conversation_manager(
            max_tokens=50000,
            max_turns=100,
        )

        assert manager.max_turns == 100

    def test_create_sliding_window(self):
        window = create_sliding_window(
            window_size=10,
            overlap=2,
        )

        assert window.window_size == 10
        assert window.overlap == 2


class TestTruncationResult:
    """Tests for TruncationResult."""

    def test_truncation_result_to_dict(self):
        result = TruncationResult(
            original_tokens=100,
            final_tokens=50,
            tokens_removed=50,
            blocks_removed=2,
            blocks_truncated=1,
            strategy_used=TruncationStrategy.PRIORITY,
            success=True,
            content=[],
        )

        d = result.to_dict()

        assert d["original_tokens"] == 100
        assert d["tokens_removed"] == 50
        assert d["strategy_used"] == "priority"


class TestCompressionResult:
    """Tests for CompressionResult."""

    def test_compression_result_to_dict(self):
        result = CompressionResult(
            original_tokens=100,
            compressed_tokens=60,
            compression_ratio=0.6,
            method_used=CompressionMethod.SUMMARIZE,
            blocks_compressed=3,
            success=True,
        )

        d = result.to_dict()

        assert d["compression_ratio"] == 0.6
        assert d["method_used"] == "summarize"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_context_window(self):
        window = ContextWindow()
        assert window.get_used_tokens() == 0
        assert window.get_content() == ""
        assert len(window.get_messages()) == 0

    def test_single_block_truncation(self):
        truncator = ContextTruncator()
        blocks = [ContextBlock("Test", ContentType.USER)]

        result = truncator.truncate(blocks, 1000)
        assert result.success
        assert len(result.content) == 1

    def test_empty_block_content(self):
        block = ContextBlock(
            content="",
            content_type=ContentType.USER,
        )
        assert block.token_count == 0

    def test_very_long_content(self):
        long_content = "word " * 10000
        block = ContextBlock(
            content=long_content,
            content_type=ContentType.USER,
        )
        assert block.token_count > 1000

    def test_unicode_content(self):
        window = ContextWindow()
        block = window.add("こんにちは 世界 🌍", ContentType.USER)
        assert block.token_count > 0

    def test_mixed_priorities_truncation(self):
        window = ContextWindow(max_tokens=50)

        window.add("Critical", ContentType.SYSTEM, PriorityLevel.CRITICAL)
        window.add("High", ContentType.USER, PriorityLevel.HIGH)
        window.add("Medium " * 20, ContentType.USER, PriorityLevel.MEDIUM)
        window.add("Low " * 20, ContentType.USER, PriorityLevel.LOW)

        window.truncate()

        blocks = window.get_blocks()
        # Critical should be preserved
        assert any(b.content == "Critical" for b in blocks)


class TestIntegration:
    """Integration tests."""

    def test_full_conversation_flow(self):
        manager = create_conversation_manager(max_tokens=10000)

        # Add system message
        manager.add_turn("system", "You are a helpful assistant.")

        # Simulate conversation
        manager.add_turn("user", "Hello!")
        manager.add_turn("assistant", "Hi there! How can I help?")
        manager.add_turn("user", "Tell me about Python.")
        manager.add_turn("assistant", "Python is a programming language...")

        # Get messages for API
        messages = manager.get_context_for_model()
        assert len(messages) >= 3
        assert messages[0]["role"] == "system"

    def test_context_window_lifecycle(self):
        window = create_context_window(max_tokens=1000)

        # Add content
        window.add_message("system", "System prompt")
        window.add_message("user", "User question")
        window.add_message("assistant", "Assistant response")

        # Check state
        state = window.get_state()
        assert state.block_count == 3
        assert state.used_tokens > 0

        # Compress
        window.compress(method=CompressionMethod.REMOVE_REDUNDANCY)

        # Truncate if needed
        window.truncate()

        # Clear
        window.clear(preserve_critical=False)
        assert len(window.get_blocks()) == 0

    def test_sliding_window_with_context_window(self):
        sliding = create_sliding_window(window_size=5)
        context = create_context_window(max_tokens=500)

        # Populate sliding window
        for i in range(10):
            sliding.add(f"Message {i}", ContentType.USER)

        # Transfer to context window
        for block in sliding.get_window():
            context.add(
                block.content,
                block.content_type,
                block.priority,
            )

        assert len(context.get_blocks()) == 5
