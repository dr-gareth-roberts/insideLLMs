"""
Context Window Management Module

Smart context management for LLM applications including:
- Token budget allocation and tracking
- Context truncation strategies
- Content priority scoring
- Context compression techniques
- Sliding window management
- Multi-turn conversation context handling
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from insideLLMs.tokens import estimate_tokens as _canonical_estimate_tokens


class TruncationStrategy(Enum):
    """Strategy for truncating content when exceeding limits."""

    TAIL = "tail"  # Keep beginning, remove end
    HEAD = "head"  # Keep end, remove beginning
    MIDDLE = "middle"  # Keep beginning and end, remove middle
    SEMANTIC = "semantic"  # Use semantic boundaries (sentences, paragraphs)
    PRIORITY = "priority"  # Remove lowest priority content first
    SLIDING_WINDOW = "sliding_window"  # Keep most recent content


class ContentType(Enum):
    """Types of content in context."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONTEXT = "context"
    EXAMPLE = "example"
    INSTRUCTION = "instruction"


class CompressionMethod(Enum):
    """Methods for compressing context content."""

    NONE = "none"
    SUMMARIZE = "summarize"
    EXTRACT_KEY_POINTS = "extract_key_points"
    REMOVE_REDUNDANCY = "remove_redundancy"
    ABBREVIATE = "abbreviate"


class PriorityLevel(Enum):
    """Priority levels for context content."""

    CRITICAL = 5  # Must never be removed
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    OPTIONAL = 1  # Can be removed first


@dataclass
class ContentAllocationBudget:
    """Token budget allocation for context."""

    total: int
    system: int = 0
    user: int = 0
    assistant: int = 0
    tools: int = 0
    context: int = 0
    reserved: int = 0  # Reserved for response

    def __post_init__(self):
        if self.system == 0 and self.user == 0:
            # Default allocation if not specified
            self.reserved = min(self.total // 4, 4096)
            available = self.total - self.reserved
            self.system = available // 5
            self.context = available // 5
            self.tools = available // 10
            self.user = available - self.system - self.context - self.tools
            self.assistant = self.user  # Share with user

    def remaining(self, current_usage: dict[str, int]) -> int:
        """Calculate remaining tokens."""
        used = sum(current_usage.values())
        return self.total - self.reserved - used

    def allocation_for(self, content_type: ContentType) -> int:
        """Get allocation for content type."""
        mapping = {
            ContentType.SYSTEM: self.system,
            ContentType.USER: self.user,
            ContentType.ASSISTANT: self.assistant,
            ContentType.TOOL_CALL: self.tools,
            ContentType.TOOL_RESULT: self.tools,
            ContentType.CONTEXT: self.context,
            ContentType.EXAMPLE: self.context,
            ContentType.INSTRUCTION: self.system,
        }
        return mapping.get(content_type, self.context)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "system": self.system,
            "user": self.user,
            "assistant": self.assistant,
            "tools": self.tools,
            "context": self.context,
            "reserved": self.reserved,
        }


@dataclass
class ContextBlock:
    """A block of content in the context."""

    content: str
    content_type: ContentType
    priority: PriorityLevel = PriorityLevel.MEDIUM
    token_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    original_content: Optional[str] = None
    block_id: str = ""

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)
        if not self.block_id:
            self.block_id = hashlib.md5(
                f"{self.content[:100]}{self.timestamp}".encode()
            ).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "content_type": self.content_type.value,
            "priority": self.priority.value,
            "token_count": self.token_count,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "compressed": self.compressed,
            "original_content": self.original_content,
            "block_id": self.block_id,
        }


@dataclass
class TruncationResult:
    """Result of a truncation operation."""

    original_tokens: int
    final_tokens: int
    tokens_removed: int
    blocks_removed: int
    blocks_truncated: int
    strategy_used: TruncationStrategy
    success: bool
    content: list["ContextBlock"]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
            "tokens_removed": self.tokens_removed,
            "blocks_removed": self.blocks_removed,
            "blocks_truncated": self.blocks_truncated,
            "strategy_used": self.strategy_used.value,
            "success": self.success,
            "content": [b.to_dict() for b in self.content],
        }


@dataclass
class ContextCompressionResult:
    """Result of a compression operation."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method_used: CompressionMethod
    blocks_compressed: int
    success: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "method_used": self.method_used.value,
            "blocks_compressed": self.blocks_compressed,
            "success": self.success,
        }


@dataclass
class ContextWindowState:
    """Current state of the context window."""

    total_tokens: int
    used_tokens: int
    available_tokens: int
    block_count: int
    usage_by_type: dict[str, int]
    budget: ContentAllocationBudget
    overflow: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "used_tokens": self.used_tokens,
            "available_tokens": self.available_tokens,
            "block_count": self.block_count,
            "usage_by_type": self.usage_by_type,
            "budget": self.budget.to_dict(),
            "overflow": self.overflow,
        }


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Uses approximation of ~4 characters per token for English.
    """
    return _canonical_estimate_tokens(text)


def find_semantic_boundaries(text: str) -> list[int]:
    """Find sentence and paragraph boundaries in text."""
    boundaries = [0]

    # Find paragraph breaks
    for match in re.finditer(r"\n\n+", text):
        boundaries.append(match.end())

    # Find sentence endings
    for match in re.finditer(r"[.!?]+\s+", text):
        boundaries.append(match.end())

    boundaries.append(len(text))
    return sorted(set(boundaries))


class TokenCounter:
    """Token counting utility with caching."""

    def __init__(self, tokenizer: Optional[Callable[[str], list]] = None):
        """
        Initialize token counter.

        Args:
            tokenizer: Optional custom tokenizer function
        """
        self.tokenizer = tokenizer
        self._cache: dict[str, int] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0

        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1

        # Count tokens
        count = len(self.tokenizer(text)) if self.tokenizer else estimate_tokens(text)

        # Cache result (limit cache size)
        if len(self._cache) < 10000:
            self._cache[cache_key] = count

        return count

    def count_messages(self, messages: list[dict]) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for msg in messages:
            if isinstance(msg.get("content"), str):
                total += self.count(msg["content"])
            # Add overhead for message structure
            total += 4  # Approximate overhead per message
        return total

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0,
            "cache_size": len(self._cache),
        }

    def clear_cache(self):
        """Clear the token cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class ContextTruncator:
    """Truncates context using various strategies."""

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        """
        Initialize truncator.

        Args:
            token_counter: Token counter to use
        """
        self.token_counter = token_counter or TokenCounter()

    def truncate(
        self,
        blocks: list[ContextBlock],
        target_tokens: int,
        strategy: TruncationStrategy = TruncationStrategy.PRIORITY,
        preserve_critical: bool = True,
    ) -> TruncationResult:
        """
        Truncate blocks to fit within token limit.

        Args:
            blocks: Context blocks to truncate
            target_tokens: Target token count
            strategy: Truncation strategy to use
            preserve_critical: Whether to preserve critical priority blocks
        """
        original_tokens = sum(b.token_count for b in blocks)

        if original_tokens <= target_tokens:
            return TruncationResult(
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                tokens_removed=0,
                blocks_removed=0,
                blocks_truncated=0,
                strategy_used=strategy,
                success=True,
                content=blocks.copy(),
            )

        if strategy == TruncationStrategy.PRIORITY:
            return self._truncate_by_priority(blocks, target_tokens, preserve_critical)
        elif strategy == TruncationStrategy.TAIL:
            return self._truncate_tail(blocks, target_tokens, preserve_critical)
        elif strategy == TruncationStrategy.HEAD:
            return self._truncate_head(blocks, target_tokens, preserve_critical)
        elif strategy == TruncationStrategy.MIDDLE:
            return self._truncate_middle(blocks, target_tokens, preserve_critical)
        elif strategy == TruncationStrategy.SEMANTIC:
            return self._truncate_semantic(blocks, target_tokens, preserve_critical)
        elif strategy == TruncationStrategy.SLIDING_WINDOW:
            return self._truncate_sliding_window(blocks, target_tokens, preserve_critical)
        else:
            # Default to priority
            return self._truncate_by_priority(blocks, target_tokens, preserve_critical)

    def _truncate_by_priority(
        self,
        blocks: list[ContextBlock],
        target_tokens: int,
        preserve_critical: bool,
    ) -> TruncationResult:
        """Truncate by removing lowest priority blocks first."""
        # Sort by priority (lowest first for removal)
        sorted_blocks = sorted(blocks, key=lambda b: b.priority.value)

        result_blocks = []
        current_tokens = 0
        blocks_removed = 0
        original_tokens = sum(b.token_count for b in blocks)

        # Add blocks from highest priority first
        for block in reversed(sorted_blocks):
            if (
                preserve_critical
                and block.priority == PriorityLevel.CRITICAL
                or current_tokens + block.token_count <= target_tokens
            ):
                result_blocks.append(block)
                current_tokens += block.token_count
            else:
                blocks_removed += 1

        # Restore original order
        result_blocks.sort(key=lambda b: blocks.index(b) if b in blocks else 0)

        return TruncationResult(
            original_tokens=original_tokens,
            final_tokens=current_tokens,
            tokens_removed=original_tokens - current_tokens,
            blocks_removed=blocks_removed,
            blocks_truncated=0,
            strategy_used=TruncationStrategy.PRIORITY,
            success=current_tokens <= target_tokens,
            content=result_blocks,
        )

    def _truncate_tail(
        self,
        blocks: list[ContextBlock],
        target_tokens: int,
        preserve_critical: bool,
    ) -> TruncationResult:
        """Keep beginning, remove end."""
        result_blocks = []
        current_tokens = 0
        blocks_removed = 0
        blocks_truncated = 0
        original_tokens = sum(b.token_count for b in blocks)

        for block in blocks:
            if (
                preserve_critical
                and block.priority == PriorityLevel.CRITICAL
                or current_tokens + block.token_count <= target_tokens
            ):
                result_blocks.append(block)
                current_tokens += block.token_count
            elif current_tokens < target_tokens:
                # Partial block
                remaining = target_tokens - current_tokens
                truncated = self._truncate_block_content(block, remaining, keep_start=True)
                result_blocks.append(truncated)
                current_tokens += truncated.token_count
                blocks_truncated += 1
            else:
                blocks_removed += 1

        return TruncationResult(
            original_tokens=original_tokens,
            final_tokens=current_tokens,
            tokens_removed=original_tokens - current_tokens,
            blocks_removed=blocks_removed,
            blocks_truncated=blocks_truncated,
            strategy_used=TruncationStrategy.TAIL,
            success=current_tokens <= target_tokens,
            content=result_blocks,
        )

    def _truncate_head(
        self,
        blocks: list[ContextBlock],
        target_tokens: int,
        preserve_critical: bool,
    ) -> TruncationResult:
        """Keep end, remove beginning."""
        result_blocks = []
        current_tokens = 0
        blocks_removed = 0
        blocks_truncated = 0
        original_tokens = sum(b.token_count for b in blocks)

        # Process in reverse
        for block in reversed(blocks):
            if (
                preserve_critical
                and block.priority == PriorityLevel.CRITICAL
                or current_tokens + block.token_count <= target_tokens
            ):
                result_blocks.insert(0, block)
                current_tokens += block.token_count
            elif current_tokens < target_tokens:
                # Partial block
                remaining = target_tokens - current_tokens
                truncated = self._truncate_block_content(block, remaining, keep_start=False)
                result_blocks.insert(0, truncated)
                current_tokens += truncated.token_count
                blocks_truncated += 1
            else:
                blocks_removed += 1

        return TruncationResult(
            original_tokens=original_tokens,
            final_tokens=current_tokens,
            tokens_removed=original_tokens - current_tokens,
            blocks_removed=blocks_removed,
            blocks_truncated=blocks_truncated,
            strategy_used=TruncationStrategy.HEAD,
            success=current_tokens <= target_tokens,
            content=result_blocks,
        )

    def _truncate_middle(
        self,
        blocks: list[ContextBlock],
        target_tokens: int,
        preserve_critical: bool,
    ) -> TruncationResult:
        """Keep beginning and end, remove middle."""
        original_tokens = sum(b.token_count for b in blocks)

        if len(blocks) <= 2:
            return self._truncate_tail(blocks, target_tokens, preserve_critical)

        # Calculate how much to keep from each end
        half_target = target_tokens // 2

        # Get head blocks
        head_blocks = []
        head_tokens = 0
        for block in blocks:
            if (
                head_tokens + block.token_count <= half_target
                or preserve_critical
                and block.priority == PriorityLevel.CRITICAL
            ):
                head_blocks.append(block)
                head_tokens += block.token_count
            else:
                break

        # Get tail blocks
        tail_blocks = []
        tail_tokens = 0
        remaining_target = target_tokens - head_tokens
        for block in reversed(blocks):
            if block in head_blocks:
                continue
            if (
                tail_tokens + block.token_count <= remaining_target
                or preserve_critical
                and block.priority == PriorityLevel.CRITICAL
            ):
                tail_blocks.insert(0, block)
                tail_tokens += block.token_count

        result_blocks = head_blocks + tail_blocks
        current_tokens = head_tokens + tail_tokens
        blocks_removed = len(blocks) - len(result_blocks)

        return TruncationResult(
            original_tokens=original_tokens,
            final_tokens=current_tokens,
            tokens_removed=original_tokens - current_tokens,
            blocks_removed=blocks_removed,
            blocks_truncated=0,
            strategy_used=TruncationStrategy.MIDDLE,
            success=current_tokens <= target_tokens,
            content=result_blocks,
        )

    def _truncate_semantic(
        self,
        blocks: list[ContextBlock],
        target_tokens: int,
        preserve_critical: bool,
    ) -> TruncationResult:
        """Truncate at semantic boundaries."""
        result_blocks = []
        current_tokens = 0
        blocks_removed = 0
        blocks_truncated = 0
        original_tokens = sum(b.token_count for b in blocks)

        for block in blocks:
            if (
                preserve_critical
                and block.priority == PriorityLevel.CRITICAL
                or current_tokens + block.token_count <= target_tokens
            ):
                result_blocks.append(block)
                current_tokens += block.token_count
            elif current_tokens < target_tokens:
                # Truncate at semantic boundary
                remaining = target_tokens - current_tokens
                truncated = self._truncate_at_semantic_boundary(block, remaining)
                if truncated.token_count > 0:
                    result_blocks.append(truncated)
                    current_tokens += truncated.token_count
                    blocks_truncated += 1
                else:
                    blocks_removed += 1
            else:
                blocks_removed += 1

        return TruncationResult(
            original_tokens=original_tokens,
            final_tokens=current_tokens,
            tokens_removed=original_tokens - current_tokens,
            blocks_removed=blocks_removed,
            blocks_truncated=blocks_truncated,
            strategy_used=TruncationStrategy.SEMANTIC,
            success=current_tokens <= target_tokens,
            content=result_blocks,
        )

    def _truncate_sliding_window(
        self,
        blocks: list[ContextBlock],
        target_tokens: int,
        preserve_critical: bool,
    ) -> TruncationResult:
        """Keep most recent content (sliding window)."""
        # Similar to HEAD but preserves order
        result_blocks = []
        current_tokens = 0
        blocks_removed = 0
        original_tokens = sum(b.token_count for b in blocks)

        # First pass: collect critical blocks
        critical_tokens = 0
        if preserve_critical:
            for block in blocks:
                if block.priority == PriorityLevel.CRITICAL:
                    result_blocks.append(block)
                    critical_tokens += block.token_count

        # Second pass: add recent blocks
        remaining_target = target_tokens - critical_tokens
        recent_blocks = []
        recent_tokens = 0

        for block in reversed(blocks):
            if block.priority == PriorityLevel.CRITICAL:
                continue
            if recent_tokens + block.token_count <= remaining_target:
                recent_blocks.insert(0, block)
                recent_tokens += block.token_count
            else:
                blocks_removed += 1

        # Merge maintaining order
        all_blocks = []
        for block in blocks:
            if block in result_blocks or block in recent_blocks:
                all_blocks.append(block)

        current_tokens = critical_tokens + recent_tokens

        return TruncationResult(
            original_tokens=original_tokens,
            final_tokens=current_tokens,
            tokens_removed=original_tokens - current_tokens,
            blocks_removed=blocks_removed,
            blocks_truncated=0,
            strategy_used=TruncationStrategy.SLIDING_WINDOW,
            success=current_tokens <= target_tokens,
            content=all_blocks,
        )

    def _truncate_block_content(
        self,
        block: ContextBlock,
        target_tokens: int,
        keep_start: bool = True,
    ) -> ContextBlock:
        """Truncate a single block's content."""
        content = block.content
        # Estimate chars to keep
        chars_to_keep = target_tokens * 4

        if keep_start:
            truncated_content = content[:chars_to_keep]
            if len(content) > chars_to_keep:
                truncated_content += "..."
        else:
            truncated_content = content[-chars_to_keep:]
            if len(content) > chars_to_keep:
                truncated_content = "..." + truncated_content

        return ContextBlock(
            content=truncated_content,
            content_type=block.content_type,
            priority=block.priority,
            token_count=self.token_counter.count(truncated_content),
            timestamp=block.timestamp,
            metadata={**block.metadata, "truncated": True},
            original_content=block.content,
        )

    def _truncate_at_semantic_boundary(
        self,
        block: ContextBlock,
        target_tokens: int,
    ) -> ContextBlock:
        """Truncate block at semantic boundary."""
        content = block.content
        boundaries = find_semantic_boundaries(content)

        # Find best boundary that fits
        chars_target = target_tokens * 4
        best_boundary = 0

        for boundary in boundaries:
            if boundary <= chars_target:
                best_boundary = boundary
            else:
                break

        if best_boundary == 0:
            best_boundary = min(chars_target, len(content))

        truncated_content = content[:best_boundary].rstrip()

        return ContextBlock(
            content=truncated_content,
            content_type=block.content_type,
            priority=block.priority,
            token_count=self.token_counter.count(truncated_content),
            timestamp=block.timestamp,
            metadata={**block.metadata, "truncated": True, "semantic": True},
            original_content=block.content,
        )


class ContextCompressor:
    """Compresses context content to save tokens."""

    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
        summarizer: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize compressor.

        Args:
            token_counter: Token counter to use
            summarizer: Optional function to summarize text
        """
        self.token_counter = token_counter or TokenCounter()
        self.summarizer = summarizer

    def compress(
        self,
        blocks: list[ContextBlock],
        target_ratio: float = 0.5,
        method: CompressionMethod = CompressionMethod.REMOVE_REDUNDANCY,
        min_priority: PriorityLevel = PriorityLevel.LOW,
    ) -> tuple[list[ContextBlock], ContextCompressionResult]:
        """
        Compress context blocks.

        Args:
            blocks: Blocks to compress
            target_ratio: Target compression ratio (0.5 = 50% of original)
            method: Compression method to use
            min_priority: Minimum priority level to compress
        """
        original_tokens = sum(b.token_count for b in blocks)
        result_blocks = []
        blocks_compressed = 0

        for block in blocks:
            if block.priority.value > min_priority.value:
                # Don't compress high priority
                result_blocks.append(block)
            else:
                compressed = self._compress_block(block, target_ratio, method)
                result_blocks.append(compressed)
                if compressed.compressed:
                    blocks_compressed += 1

        compressed_tokens = sum(b.token_count for b in result_blocks)

        result = ContextCompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            method_used=method,
            blocks_compressed=blocks_compressed,
            success=True,
        )

        return result_blocks, result

    def _compress_block(
        self,
        block: ContextBlock,
        target_ratio: float,
        method: CompressionMethod,
    ) -> ContextBlock:
        """Compress a single block."""
        if method == CompressionMethod.NONE:
            return block

        content = block.content
        original_tokens = block.token_count

        if method == CompressionMethod.SUMMARIZE:
            compressed = self._summarize(content, target_ratio)
        elif method == CompressionMethod.EXTRACT_KEY_POINTS:
            compressed = self._extract_key_points(content)
        elif method == CompressionMethod.REMOVE_REDUNDANCY:
            compressed = self._remove_redundancy(content)
        elif method == CompressionMethod.ABBREVIATE:
            compressed = self._abbreviate(content)
        else:
            compressed = content

        new_tokens = self.token_counter.count(compressed)

        if new_tokens < original_tokens:
            return ContextBlock(
                content=compressed,
                content_type=block.content_type,
                priority=block.priority,
                token_count=new_tokens,
                timestamp=block.timestamp,
                metadata={**block.metadata, "compression_method": method.value},
                compressed=True,
                original_content=block.content,
            )

        return block

    def _summarize(self, content: str, target_ratio: float) -> str:
        """Summarize content."""
        if self.summarizer:
            return self.summarizer(content)

        # Simple extractive summary: keep first sentences
        sentences = re.split(r"(?<=[.!?])\s+", content)
        target_count = max(1, int(len(sentences) * target_ratio))
        return " ".join(sentences[:target_count])

    def _extract_key_points(self, content: str) -> str:
        """Extract key points from content."""
        # Simple heuristic: look for bullet points, numbered lists, and important sentences
        lines = content.split("\n")
        key_points = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Keep bullet points and numbered items
            if (
                re.match(r"^[-•*]\s+", line)
                or re.match(r"^\d+[.)]\s+", line)
                or any(
                    kw in line.lower() for kw in ["important", "key", "note:", "critical", "must"]
                )
            ):
                key_points.append(line)

        if key_points:
            return "\n".join(key_points)

        # Fallback: return first paragraph
        paragraphs = content.split("\n\n")
        return paragraphs[0] if paragraphs else content

    def _remove_redundancy(self, content: str) -> str:
        """Remove redundant content."""
        lines = content.split("\n")
        seen_content: set[str] = set()
        unique_lines = []

        for line in lines:
            # Normalize for comparison
            normalized = " ".join(line.lower().split())
            if normalized and normalized not in seen_content:
                seen_content.add(normalized)
                unique_lines.append(line)
            elif not normalized:  # Keep empty lines for formatting
                unique_lines.append(line)

        # Also remove repeated phrases within text
        result = "\n".join(unique_lines)

        # Remove consecutive duplicate words
        result = re.sub(r"\b(\w+)\s+\1\b", r"\1", result, flags=re.IGNORECASE)

        return result

    def _abbreviate(self, content: str) -> str:
        """Abbreviate common phrases."""
        abbreviations = {
            "for example": "e.g.",
            "that is": "i.e.",
            "and so on": "etc.",
            "in other words": "i.e.",
            "with respect to": "w.r.t.",
            "as soon as possible": "ASAP",
            "information": "info",
            "configuration": "config",
            "documentation": "docs",
            "application": "app",
            "implementation": "impl",
            "approximately": "approx.",
            "function": "func",
            "parameter": "param",
            "argument": "arg",
        }

        result = content
        for phrase, abbrev in abbreviations.items():
            result = re.sub(rf"\b{re.escape(phrase)}\b", abbrev, result, flags=re.IGNORECASE)

        return result


class ContextWindow:
    """
    Main context window manager.

    Manages context blocks with budget allocation, truncation, and compression.
    """

    def __init__(
        self,
        max_tokens: int = 128000,
        budget: Optional[ContentAllocationBudget] = None,
        token_counter: Optional[TokenCounter] = None,
        default_strategy: TruncationStrategy = TruncationStrategy.PRIORITY,
    ):
        """
        Initialize context window.

        Args:
            max_tokens: Maximum tokens for the context window
            budget: Token budget allocation
            token_counter: Token counter to use
            default_strategy: Default truncation strategy
        """
        self.max_tokens = max_tokens
        self.budget = budget or ContentAllocationBudget(total=max_tokens)
        self.token_counter = token_counter or TokenCounter()
        self.default_strategy = default_strategy

        self.truncator = ContextTruncator(self.token_counter)
        self.compressor = ContextCompressor(self.token_counter)

        self._blocks: list[ContextBlock] = []
        self._history: list[dict[str, Any]] = []

    def add(
        self,
        content: str,
        content_type: ContentType = ContentType.CONTEXT,
        priority: PriorityLevel = PriorityLevel.MEDIUM,
        metadata: Optional[dict] = None,
    ) -> ContextBlock:
        """
        Add content to the context window.

        Args:
            content: Content to add
            content_type: Type of content
            priority: Priority level
            metadata: Optional metadata
        """
        block = ContextBlock(
            content=content,
            content_type=content_type,
            priority=priority,
            token_count=self.token_counter.count(content),
            metadata=metadata or {},
        )

        self._blocks.append(block)
        self._record_action("add", block)

        # Auto-truncate if over budget
        if self.get_used_tokens() > self.max_tokens - self.budget.reserved:
            self.truncate()

        return block

    def add_message(
        self,
        role: str,
        content: str,
        priority: Optional[PriorityLevel] = None,
        metadata: Optional[dict] = None,
    ) -> ContextBlock:
        """
        Add a chat message to context.

        Args:
            role: Message role (system, user, assistant)
            content: Message content
            priority: Priority level (auto-determined if not specified)
            metadata: Optional metadata
        """
        # Map role to content type
        role_mapping = {
            "system": ContentType.SYSTEM,
            "user": ContentType.USER,
            "assistant": ContentType.ASSISTANT,
        }
        content_type = role_mapping.get(role, ContentType.CONTEXT)

        # Default priorities by role
        if priority is None:
            priority_mapping = {
                "system": PriorityLevel.HIGH,
                "user": PriorityLevel.MEDIUM,
                "assistant": PriorityLevel.MEDIUM,
            }
            priority = priority_mapping.get(role, PriorityLevel.MEDIUM)

        return self.add(
            content=content,
            content_type=content_type,
            priority=priority,
            metadata={**(metadata or {}), "role": role},
        )

    def remove(self, block_id: str) -> bool:
        """Remove a block by ID."""
        for i, block in enumerate(self._blocks):
            if block.block_id == block_id:
                removed = self._blocks.pop(i)
                self._record_action("remove", removed)
                return True
        return False

    def clear(self, preserve_critical: bool = True):
        """
        Clear context window.

        Args:
            preserve_critical: Whether to keep critical priority blocks
        """
        if preserve_critical:
            self._blocks = [b for b in self._blocks if b.priority == PriorityLevel.CRITICAL]
        else:
            self._blocks = []

        self._record_action("clear", None)

    def truncate(
        self,
        target_tokens: Optional[int] = None,
        strategy: Optional[TruncationStrategy] = None,
    ) -> TruncationResult:
        """
        Truncate context to fit within limits.

        Args:
            target_tokens: Target token count
            strategy: Truncation strategy to use
        """
        target = target_tokens or (self.max_tokens - self.budget.reserved)
        strat = strategy or self.default_strategy

        result = self.truncator.truncate(
            self._blocks,
            target,
            strat,
            preserve_critical=True,
        )

        self._blocks = result.content
        self._record_action("truncate", result.to_dict())

        return result

    def compress(
        self,
        target_ratio: float = 0.5,
        method: CompressionMethod = CompressionMethod.REMOVE_REDUNDANCY,
    ) -> ContextCompressionResult:
        """
        Compress context content.

        Args:
            target_ratio: Target compression ratio
            method: Compression method to use
        """
        self._blocks, result = self.compressor.compress(
            self._blocks,
            target_ratio,
            method,
        )

        self._record_action("compress", result.to_dict())

        return result

    def get_blocks(
        self,
        content_type: Optional[ContentType] = None,
        min_priority: Optional[PriorityLevel] = None,
    ) -> list[ContextBlock]:
        """
        Get blocks, optionally filtered.

        Args:
            content_type: Filter by content type
            min_priority: Filter by minimum priority
        """
        blocks = self._blocks

        if content_type:
            blocks = [b for b in blocks if b.content_type == content_type]

        if min_priority:
            blocks = [b for b in blocks if b.priority.value >= min_priority.value]

        return blocks

    def get_content(self, separator: str = "\n\n") -> str:
        """Get all content as a single string."""
        return separator.join(b.content for b in self._blocks)

    def get_messages(self) -> list[dict[str, str]]:
        """Get blocks as chat messages."""
        messages = []
        for block in self._blocks:
            role = block.metadata.get("role")
            if role:
                messages.append({"role": role, "content": block.content})
        return messages

    def get_used_tokens(self) -> int:
        """Get total tokens currently used."""
        return sum(b.token_count for b in self._blocks)

    def get_available_tokens(self) -> int:
        """Get available tokens."""
        return self.max_tokens - self.budget.reserved - self.get_used_tokens()

    def get_state(self) -> ContextWindowState:
        """Get current context window state."""
        usage_by_type: dict[str, int] = {}
        for block in self._blocks:
            type_name = block.content_type.value
            usage_by_type[type_name] = usage_by_type.get(type_name, 0) + block.token_count

        used = self.get_used_tokens()

        return ContextWindowState(
            total_tokens=self.max_tokens,
            used_tokens=used,
            available_tokens=self.get_available_tokens(),
            block_count=len(self._blocks),
            usage_by_type=usage_by_type,
            budget=self.budget,
            overflow=used > self.max_tokens - self.budget.reserved,
        )

    def _record_action(self, action: str, data: Any):
        """Record action in history."""
        self._history.append(
            {
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "data": data.to_dict() if hasattr(data, "to_dict") else data,
                "state": {
                    "used_tokens": self.get_used_tokens(),
                    "block_count": len(self._blocks),
                },
            }
        )

    def get_history(self) -> list[dict[str, Any]]:
        """Get action history."""
        return self._history.copy()


class ConversationManager:
    """
    Manages multi-turn conversation context.

    Provides specialized handling for conversation history with
    automatic summarization of older messages.
    """

    def __init__(
        self,
        context_window: Optional[ContextWindow] = None,
        max_turns: int = 50,
        summarize_after: int = 20,
        summarizer: Optional[Callable[[list[dict]], str]] = None,
    ):
        """
        Initialize conversation manager.

        Args:
            context_window: Context window to use
            max_turns: Maximum conversation turns to keep
            summarize_after: Summarize messages after this many turns
            summarizer: Optional function to summarize messages
        """
        self.context_window = context_window or ContextWindow()
        self.max_turns = max_turns
        self.summarize_after = summarize_after
        self.summarizer = summarizer

        self._turns: list[dict[str, Any]] = []
        self._summary: Optional[str] = None
        self._summary_turn_count = 0

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Add a conversation turn.

        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
        """
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "turn_number": len(self._turns) + 1,
            "metadata": metadata or {},
        }

        self._turns.append(turn)

        # Check if we need to summarize
        if len(self._turns) > self.summarize_after:
            self._maybe_summarize()

        # Add to context window
        priority = PriorityLevel.HIGH if role == "system" else PriorityLevel.MEDIUM
        self.context_window.add_message(role, content, priority, metadata)

        return turn

    def get_turns(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Get conversation turns."""
        if limit:
            return self._turns[-limit:]
        return self._turns.copy()

    def get_context_for_model(self, max_tokens: Optional[int] = None) -> list[dict[str, str]]:
        """
        Get conversation context formatted for model.

        Args:
            max_tokens: Optional token limit
        """
        messages = []

        # Add summary if available
        if self._summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"[Previous conversation summary: {self._summary}]",
                }
            )

        # Add recent turns
        for turn in self._turns:
            messages.append(
                {
                    "role": turn["role"],
                    "content": turn["content"],
                }
            )

        # Truncate if needed
        if max_tokens:
            token_counter = self.context_window.token_counter
            total_tokens = sum(token_counter.count(m["content"]) for m in messages)

            while total_tokens > max_tokens and len(messages) > 1:
                # Remove oldest non-system message
                for i, msg in enumerate(messages):
                    if msg["role"] != "system":
                        removed = messages.pop(i)
                        total_tokens -= token_counter.count(removed["content"])
                        break

        return messages

    def clear(self, keep_system: bool = True):
        """
        Clear conversation history.

        Args:
            keep_system: Whether to keep system messages
        """
        if keep_system:
            self._turns = [t for t in self._turns if t["role"] == "system"]
        else:
            self._turns = []

        self._summary = None
        self._summary_turn_count = 0
        self.context_window.clear(preserve_critical=keep_system)

    def _maybe_summarize(self):
        """Summarize older turns if needed."""
        if len(self._turns) <= self.summarize_after:
            return

        # Turns to summarize
        turns_to_summarize = self._turns[: -self.summarize_after]

        if self.summarizer:
            self._summary = self.summarizer(turns_to_summarize)
        else:
            # Simple default summary
            self._summary = self._default_summarize(turns_to_summarize)

        self._summary_turn_count = len(turns_to_summarize)

        # Keep only recent turns
        self._turns = self._turns[-self.summarize_after :]

    def _default_summarize(self, turns: list[dict[str, Any]]) -> str:
        """Default summarization method."""
        summary_parts = []

        for turn in turns[:10]:  # Limit summary length
            role = turn["role"]
            content = turn["content"][:100]  # Truncate content
            if len(turn["content"]) > 100:
                content += "..."
            summary_parts.append(f"{role}: {content}")

        if len(turns) > 10:
            summary_parts.append(f"... and {len(turns) - 10} more messages")

        return "\n".join(summary_parts)

    def get_stats(self) -> dict[str, Any]:
        """Get conversation statistics."""
        return {
            "total_turns": len(self._turns) + self._summary_turn_count,
            "active_turns": len(self._turns),
            "summarized_turns": self._summary_turn_count,
            "has_summary": self._summary is not None,
            "context_tokens": self.context_window.get_used_tokens(),
            "available_tokens": self.context_window.get_available_tokens(),
        }


class SlidingWindowManager:
    """
    Sliding window context management.

    Maintains a fixed-size window of recent content.
    """

    def __init__(
        self,
        window_size: int = 10,
        overlap: int = 2,
        token_counter: Optional[TokenCounter] = None,
    ):
        """
        Initialize sliding window.

        Args:
            window_size: Number of items in window
            overlap: Overlap between windows
            token_counter: Token counter to use
        """
        self.window_size = window_size
        self.overlap = overlap
        self.token_counter = token_counter or TokenCounter()

        self._items: list[ContextBlock] = []
        self._archived: list[ContextBlock] = []

    def add(
        self,
        content: str,
        content_type: ContentType = ContentType.CONTEXT,
        priority: PriorityLevel = PriorityLevel.MEDIUM,
        metadata: Optional[dict] = None,
    ) -> ContextBlock:
        """Add item to sliding window."""
        block = ContextBlock(
            content=content,
            content_type=content_type,
            priority=priority,
            token_count=self.token_counter.count(content),
            metadata=metadata or {},
        )

        self._items.append(block)

        # Slide window if needed
        if len(self._items) > self.window_size:
            self._slide()

        return block

    def _slide(self):
        """Slide the window forward."""
        # Archive items outside window
        items_to_archive = len(self._items) - self.window_size

        for i in range(items_to_archive):
            self._archived.append(self._items[i])

        # Keep overlap
        self._items = self._items[items_to_archive:]

    def get_window(self) -> list[ContextBlock]:
        """Get current window contents."""
        return self._items.copy()

    def get_archived(self) -> list[ContextBlock]:
        """Get archived items."""
        return self._archived.copy()

    def get_content(self, include_archived: bool = False) -> str:
        """Get content as string."""
        items = self._items
        if include_archived:
            items = self._archived + items
        return "\n\n".join(b.content for b in items)

    def get_tokens(self) -> int:
        """Get total tokens in current window."""
        return sum(b.token_count for b in self._items)

    def clear(self, clear_archive: bool = False):
        """Clear window."""
        self._items = []
        if clear_archive:
            self._archived = []

    def get_stats(self) -> dict[str, Any]:
        """Get window statistics."""
        return {
            "window_size": self.window_size,
            "current_items": len(self._items),
            "archived_items": len(self._archived),
            "window_tokens": self.get_tokens(),
            "archived_tokens": sum(b.token_count for b in self._archived),
        }


# Convenience functions


def create_context_window(
    max_tokens: int = 128000,
    strategy: TruncationStrategy = TruncationStrategy.PRIORITY,
) -> ContextWindow:
    """Create a context window with default settings."""
    return ContextWindow(
        max_tokens=max_tokens,
        default_strategy=strategy,
    )


def estimate_context_tokens(content: str) -> int:
    """Estimate tokens for content."""
    return estimate_tokens(content)


def truncate_context(
    blocks: list[ContextBlock],
    target_tokens: int,
    strategy: TruncationStrategy = TruncationStrategy.PRIORITY,
) -> TruncationResult:
    """Truncate context blocks to target token count."""
    truncator = ContextTruncator()
    return truncator.truncate(blocks, target_tokens, strategy)


def compress_context(
    blocks: list[ContextBlock],
    target_ratio: float = 0.5,
    method: CompressionMethod = CompressionMethod.REMOVE_REDUNDANCY,
) -> tuple[list[ContextBlock], ContextCompressionResult]:
    """Compress context blocks."""
    compressor = ContextCompressor()
    return compressor.compress(blocks, target_ratio, method)


def create_budget(
    total: int,
    system_ratio: float = 0.2,
    context_ratio: float = 0.2,
    tools_ratio: float = 0.1,
    reserved_ratio: float = 0.25,
) -> ContentAllocationBudget:
    """
    Create a token budget with specified ratios.

    Args:
        total: Total tokens
        system_ratio: Ratio for system content
        context_ratio: Ratio for context content
        tools_ratio: Ratio for tool calls/results
        reserved_ratio: Ratio reserved for response
    """
    reserved = int(total * reserved_ratio)
    available = total - reserved
    system = int(available * system_ratio)
    context = int(available * context_ratio)
    tools = int(available * tools_ratio)
    user = available - system - context - tools

    return ContentAllocationBudget(
        total=total,
        system=system,
        user=user,
        assistant=user,
        tools=tools,
        context=context,
        reserved=reserved,
    )


def create_conversation_manager(
    max_tokens: int = 128000,
    max_turns: int = 50,
) -> ConversationManager:
    """Create a conversation manager."""
    context_window = ContextWindow(max_tokens=max_tokens)
    return ConversationManager(
        context_window=context_window,
        max_turns=max_turns,
    )


def create_sliding_window(
    window_size: int = 10,
    overlap: int = 2,
) -> SlidingWindowManager:
    """Create a sliding window manager."""
    return SlidingWindowManager(
        window_size=window_size,
        overlap=overlap,
    )
