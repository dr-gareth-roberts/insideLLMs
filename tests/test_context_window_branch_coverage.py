"""Additional branch coverage for context_window internals."""

from __future__ import annotations

from insideLLMs.context_window import (
    CompressionMethod,
    ContentType,
    ContextBlock,
    ContextCompressor,
    ContextTruncator,
    ContextWindow,
    ConversationManager,
    PriorityLevel,
    TruncationStrategy,
)


def _block(
    content: str, tokens: int, priority: PriorityLevel = PriorityLevel.MEDIUM
) -> ContextBlock:
    return ContextBlock(
        content=content,
        content_type=ContentType.CONTEXT,
        priority=priority,
        token_count=tokens,
    )


def test_truncate_unknown_strategy_defaults_to_priority():
    truncator = ContextTruncator()
    blocks = [_block("A", 30), _block("B", 30), _block("C", 30, PriorityLevel.CRITICAL)]
    result = truncator.truncate(
        blocks,
        target_tokens=40,
        strategy="not-a-strategy",  # type: ignore[arg-type]
        preserve_critical=True,
    )
    assert result.strategy_used == TruncationStrategy.PRIORITY
    assert any(b.priority == PriorityLevel.CRITICAL for b in result.content)


def test_truncate_middle_head_tail_paths_and_skip_head_branch():
    truncator = ContextTruncator()
    blocks = [
        _block("head", 10),
        _block("middle-1", 12),
        _block("critical", 25, PriorityLevel.CRITICAL),
        _block("tail", 10),
    ]
    result = truncator._truncate_middle(blocks, target_tokens=30, preserve_critical=True)
    assert result.strategy_used == TruncationStrategy.MIDDLE
    assert result.final_tokens >= 0

    # Ensure the "block in head_blocks" skip branch is exercised.
    blocks2 = [_block("h1", 10), _block("h2", 10), _block("t1", 10)]
    result2 = truncator._truncate_middle(blocks2, target_tokens=25, preserve_critical=False)
    assert len(result2.content) <= len(blocks2)


def test_truncate_semantic_removed_and_over_target_branches(monkeypatch):
    truncator = ContextTruncator()

    # Force semantic truncation to return empty content => removed branch.
    monkeypatch.setattr(
        truncator,
        "_truncate_at_semantic_boundary",
        lambda block, remaining: ContextBlock(
            content="",
            content_type=block.content_type,
            priority=block.priority,
            token_count=0,
        ),
    )
    removed_result = truncator._truncate_semantic(
        [_block("too long", 10), _block("also long", 8)],
        target_tokens=5,
        preserve_critical=False,
    )
    assert removed_result.blocks_removed >= 1

    # current_tokens >= target_tokens branch.
    over_target_result = truncator._truncate_semantic(
        [_block("fit", 5), _block("overflow", 5)],
        target_tokens=5,
        preserve_critical=False,
    )
    assert over_target_result.blocks_removed >= 1


def test_truncate_sliding_window_and_block_content_end_truncation():
    truncator = ContextTruncator()
    sw_result = truncator._truncate_sliding_window(
        [_block("old", 15), _block("new", 15), _block("latest", 15)],
        target_tokens=20,
        preserve_critical=False,
    )
    assert sw_result.strategy_used == TruncationStrategy.SLIDING_WINDOW

    content_block = _block("x" * 40, 10)
    tail_truncated = truncator._truncate_block_content(
        content_block, target_tokens=3, keep_start=False
    )
    assert tail_truncated.content.startswith("...")
    assert tail_truncated.metadata["truncated"] is True
    assert tail_truncated.original_content == content_block.content


def test_context_compressor_none_summarize_and_key_point_fallback_paths():
    compressor = ContextCompressor()
    block = _block("Sentence one. Sentence two? Sentence three!", 50)

    # NONE branch returns original block unchanged.
    same_block = compressor._compress_block(block, target_ratio=0.5, method=CompressionMethod.NONE)
    assert same_block is block

    summary = compressor._summarize("A. B. C.", target_ratio=0.34)
    assert summary.startswith("A.")

    key_points = compressor._extract_key_points("- Important item\n2) Another key point\n")
    assert "Important item" in key_points

    fallback = compressor._extract_key_points("No bullets here\n\nSecond paragraph")
    assert fallback == "No bullets here"

    # Force lower token count to exercise compressed-block construction branch.
    compressor.token_counter.count = lambda text: len(text)  # type: ignore[assignment]
    verbose = _block("- point one\n- point two\n", tokens=100)
    compressed = compressor._compress_block(
        verbose,
        target_ratio=0.5,
        method=CompressionMethod.EXTRACT_KEY_POINTS,
    )
    assert compressed.compressed is True
    assert compressed.metadata["compression_method"] == CompressionMethod.EXTRACT_KEY_POINTS.value


def test_context_window_remove_and_conversation_maybe_summarize_noop():
    window = ContextWindow(max_tokens=100)
    added = window.add("hello", ContentType.USER)
    assert window.remove("missing-id") is False
    assert window.remove(added.block_id) is True

    manager = ConversationManager(summarize_after=5)
    manager.add_turn("user", "short")
    manager.add_turn("assistant", "reply")
    manager._maybe_summarize()
    assert manager._summary is None
