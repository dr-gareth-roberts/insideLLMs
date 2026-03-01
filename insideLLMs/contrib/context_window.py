"""
Context Window Management Module
================================

Smart context management for LLM applications including:
- Token budget allocation and tracking
- Context truncation strategies
- Content priority scoring
- Context compression techniques
- Sliding window management
- Multi-turn conversation context handling

This module provides a comprehensive set of tools for managing context windows
in Large Language Model (LLM) applications. It handles the complex task of
fitting conversation history, system prompts, tool calls, and other content
within the token limits of various LLMs.

Key Components
--------------
- **ContextWindow**: Main context window manager with budget allocation
- **ConversationManager**: Multi-turn conversation context handling
- **SlidingWindowManager**: Fixed-size sliding window for recent content
- **ContextTruncator**: Truncation strategies (priority, semantic, etc.)
- **ContextCompressor**: Content compression techniques
- **TokenCounter**: Cached token counting utility

Examples
--------
Basic context window usage:

>>> from insideLLMs.contrib.context_window import ContextWindow, ContentType, PriorityLevel
>>> window = ContextWindow(max_tokens=8000)
>>> window.add("You are a helpful assistant.", ContentType.SYSTEM, PriorityLevel.CRITICAL)
>>> window.add("What is Python?", ContentType.USER)
>>> window.add("Python is a programming language.", ContentType.ASSISTANT)
>>> print(window.get_used_tokens())
25

Using conversation manager for multi-turn conversations:

>>> from insideLLMs.contrib.context_window import ConversationManager
>>> manager = ConversationManager(max_turns=50)
>>> manager.add_turn("system", "You are a helpful assistant.")
>>> manager.add_turn("user", "Hello!")
>>> manager.add_turn("assistant", "Hi there! How can I help?")
>>> messages = manager.get_context_for_model()

Creating a sliding window for streaming content:

>>> from insideLLMs.contrib.context_window import SlidingWindowManager
>>> slider = SlidingWindowManager(window_size=5, overlap=1)
>>> for chunk in ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5", "chunk6"]:
...     slider.add(chunk)
>>> len(slider.get_window())  # Only keeps 5 most recent
5

Using token budgets for allocation:

>>> from insideLLMs.contrib.context_window import create_budget
>>> budget = create_budget(total=32000, system_ratio=0.15, reserved_ratio=0.2)
>>> print(f"System allocation: {budget.system} tokens")
System allocation: 3840 tokens
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from insideLLMs.tokens import estimate_tokens as _canonical_estimate_tokens


class TruncationStrategy(Enum):
    """
    Strategy for truncating content when exceeding token limits.

    Different strategies are appropriate for different use cases:
    - PRIORITY: Best for general use when content has varying importance
    - TAIL: Best when beginning content (like system prompts) is most important
    - HEAD: Best for chat where recent messages matter most
    - MIDDLE: Best when both beginning and end are important
    - SEMANTIC: Best for preserving readable, coherent content
    - SLIDING_WINDOW: Best for streaming or real-time content

    Attributes
    ----------
    TAIL : str
        Keep beginning, remove end. Preserves initial context like system prompts.
    HEAD : str
        Keep end, remove beginning. Preserves recent conversation history.
    MIDDLE : str
        Keep beginning and end, remove middle. Good for long documents.
    SEMANTIC : str
        Use semantic boundaries (sentences, paragraphs) for cleaner cuts.
    PRIORITY : str
        Remove lowest priority content first. Most flexible strategy.
    SLIDING_WINDOW : str
        Keep most recent content with overlap. Good for streaming.

    Examples
    --------
    Using different strategies for truncation:

    >>> from insideLLMs.contrib.context_window import (
    ...     ContextTruncator, ContextBlock, ContentType, PriorityLevel, TruncationStrategy
    ... )
    >>> truncator = ContextTruncator()
    >>> blocks = [
    ...     ContextBlock("Important system prompt", ContentType.SYSTEM, PriorityLevel.HIGH),
    ...     ContextBlock("User message 1", ContentType.USER),
    ...     ContextBlock("User message 2", ContentType.USER),
    ... ]

    Priority-based truncation (removes lowest priority first):

    >>> result = truncator.truncate(blocks, target_tokens=50, strategy=TruncationStrategy.PRIORITY)
    >>> result.strategy_used
    <TruncationStrategy.PRIORITY: 'priority'>

    Tail truncation (keeps beginning):

    >>> result = truncator.truncate(blocks, target_tokens=30, strategy=TruncationStrategy.TAIL)
    >>> len(result.content) <= len(blocks)
    True

    Semantic truncation (respects sentence boundaries):

    >>> result = truncator.truncate(blocks, target_tokens=40, strategy=TruncationStrategy.SEMANTIC)
    >>> result.success
    True
    """

    TAIL = "tail"  # Keep beginning, remove end
    HEAD = "head"  # Keep end, remove beginning
    MIDDLE = "middle"  # Keep beginning and end, remove middle
    SEMANTIC = "semantic"  # Use semantic boundaries (sentences, paragraphs)
    PRIORITY = "priority"  # Remove lowest priority content first
    SLIDING_WINDOW = "sliding_window"  # Keep most recent content


class ContentType(Enum):
    """
    Types of content that can be stored in the context window.

    Content types help the context manager understand the role and importance
    of different pieces of content. This is used for budget allocation and
    intelligent truncation decisions.

    Attributes
    ----------
    SYSTEM : str
        System prompts and instructions. Usually highest priority.
    USER : str
        User messages in the conversation.
    ASSISTANT : str
        Assistant/model responses.
    TOOL_CALL : str
        Tool/function call requests from the model.
    TOOL_RESULT : str
        Results returned from tool/function executions.
    CONTEXT : str
        Additional context like documents or retrieved information.
    EXAMPLE : str
        Few-shot examples for prompting.
    INSTRUCTION : str
        Specific instructions (similar to system but for specific tasks).

    Examples
    --------
    Categorizing different content types:

    >>> from insideLLMs.contrib.context_window import ContextBlock, ContentType, PriorityLevel
    >>> system_block = ContextBlock(
    ...     content="You are a helpful assistant.",
    ...     content_type=ContentType.SYSTEM,
    ...     priority=PriorityLevel.CRITICAL
    ... )
    >>> system_block.content_type
    <ContentType.SYSTEM: 'system'>

    Adding user and assistant messages:

    >>> user_msg = ContextBlock("What is Python?", ContentType.USER)
    >>> assistant_msg = ContextBlock("Python is a programming language.", ContentType.ASSISTANT)
    >>> user_msg.content_type.value
    'user'

    Working with tool calls and results:

    >>> tool_call = ContextBlock('{"function": "search", "args": {"query": "python"}}', ContentType.TOOL_CALL)
    >>> tool_result = ContextBlock('{"results": ["Python docs", "Python tutorial"]}', ContentType.TOOL_RESULT)
    >>> tool_call.content_type == ContentType.TOOL_CALL
    True

    Adding context documents:

    >>> doc_context = ContextBlock(
    ...     "Python was created by Guido van Rossum in 1991.",
    ...     ContentType.CONTEXT,
    ...     metadata={"source": "wikipedia"}
    ... )
    >>> doc_context.content_type.value
    'context'
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONTEXT = "context"
    EXAMPLE = "example"
    INSTRUCTION = "instruction"


class CompressionMethod(Enum):
    """
    Methods for compressing context content to reduce token usage.

    Compression methods provide different approaches to reducing content size
    while attempting to preserve important information. The choice of method
    depends on the content type and acceptable information loss.

    Attributes
    ----------
    NONE : str
        No compression applied. Content remains unchanged.
    SUMMARIZE : str
        Create a summary of the content. Good for long documents.
    EXTRACT_KEY_POINTS : str
        Extract bullet points and key information. Good for structured content.
    REMOVE_REDUNDANCY : str
        Remove duplicate and redundant content. Safe, minimal information loss.
    ABBREVIATE : str
        Replace common phrases with abbreviations. Lightweight compression.

    Examples
    --------
    Using different compression methods:

    >>> from insideLLMs.contrib.context_window import (
    ...     ContextCompressor, ContextBlock, ContentType, CompressionMethod
    ... )
    >>> compressor = ContextCompressor()
    >>> blocks = [ContextBlock("This is a long document. This is a long document.", ContentType.CONTEXT)]

    Remove redundancy (safest method):

    >>> compressed, result = compressor.compress(blocks, method=CompressionMethod.REMOVE_REDUNDANCY)
    >>> result.compression_ratio < 1.0 or len(blocks[0].content) == len(compressed[0].content)
    True

    Summarize content:

    >>> long_block = ContextBlock(
    ...     "First sentence here. Second sentence here. Third sentence here. Fourth sentence.",
    ...     ContentType.CONTEXT
    ... )
    >>> compressed, result = compressor.compress([long_block], target_ratio=0.5, method=CompressionMethod.SUMMARIZE)
    >>> result.method_used
    <CompressionMethod.SUMMARIZE: 'summarize'>

    Abbreviate common phrases:

    >>> text_block = ContextBlock("For example, the application configuration is important.", ContentType.CONTEXT)
    >>> compressed, result = compressor.compress([text_block], method=CompressionMethod.ABBREVIATE)
    >>> # "For example" -> "e.g.", "application" -> "app", "configuration" -> "config"
    >>> result.success
    True

    No compression (passthrough):

    >>> compressed, result = compressor.compress(blocks, method=CompressionMethod.NONE)
    >>> result.compression_ratio
    1.0
    """

    NONE = "none"
    SUMMARIZE = "summarize"
    EXTRACT_KEY_POINTS = "extract_key_points"
    REMOVE_REDUNDANCY = "remove_redundancy"
    ABBREVIATE = "abbreviate"


class PriorityLevel(Enum):
    """
    Priority levels for context content.

    Priority levels determine which content is preserved during truncation.
    Higher priority content is kept while lower priority content is removed
    first when the context window exceeds its token limit.

    The numeric values (1-5) allow for comparison and sorting. Higher values
    indicate higher priority.

    Attributes
    ----------
    CRITICAL : int
        Value 5. Must never be removed. Use for essential system prompts.
    HIGH : int
        Value 4. Important content. Use for recent user messages.
    MEDIUM : int
        Value 3. Standard priority. Default for most content.
    LOW : int
        Value 2. Less important. Can be removed if needed.
    OPTIONAL : int
        Value 1. Can be removed first. Use for supplementary information.

    Examples
    --------
    Assigning priorities to context blocks:

    >>> from insideLLMs.contrib.context_window import ContextBlock, ContentType, PriorityLevel
    >>> critical = ContextBlock("System prompt", ContentType.SYSTEM, PriorityLevel.CRITICAL)
    >>> critical.priority.value
    5

    Comparing priorities:

    >>> high = PriorityLevel.HIGH
    >>> low = PriorityLevel.LOW
    >>> high.value > low.value
    True

    Using priorities with truncation:

    >>> from insideLLMs.contrib.context_window import ContextWindow, TruncationStrategy
    >>> window = ContextWindow(max_tokens=100)
    >>> window.add("Must keep!", ContentType.SYSTEM, PriorityLevel.CRITICAL)
    >>> window.add("Optional info", ContentType.CONTEXT, PriorityLevel.OPTIONAL)
    >>> window.add("More optional", ContentType.CONTEXT, PriorityLevel.OPTIONAL)
    >>> # When truncation happens, OPTIONAL content is removed first
    >>> result = window.truncate(target_tokens=50, strategy=TruncationStrategy.PRIORITY)
    >>> any(b.priority == PriorityLevel.CRITICAL for b in result.content)
    True

    Priority-based filtering:

    >>> blocks = window.get_blocks(min_priority=PriorityLevel.HIGH)
    >>> all(b.priority.value >= PriorityLevel.HIGH.value for b in blocks)
    True
    """

    CRITICAL = 5  # Must never be removed
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    OPTIONAL = 1  # Can be removed first


@dataclass
class ContentAllocationBudget:
    """
    Token budget allocation for different content types in the context window.

    This class manages how the total token budget is distributed among different
    types of content (system prompts, user messages, assistant responses, etc.).
    It ensures that each content type has a dedicated allocation while reserving
    space for the model's response.

    If no specific allocations are provided, sensible defaults are calculated
    automatically based on the total token budget.

    Attributes
    ----------
    total : int
        Total token budget for the context window.
    system : int
        Tokens allocated for system prompts and instructions.
    user : int
        Tokens allocated for user messages.
    assistant : int
        Tokens allocated for assistant responses in history.
    tools : int
        Tokens allocated for tool calls and results.
    context : int
        Tokens allocated for additional context (documents, RAG content).
    reserved : int
        Tokens reserved for the model's response output.

    Examples
    --------
    Creating a budget with automatic allocation:

    >>> from insideLLMs.contrib.context_window import ContentAllocationBudget
    >>> budget = ContentAllocationBudget(total=32000)
    >>> budget.reserved > 0  # Response space is automatically reserved
    True
    >>> budget.system + budget.user + budget.context + budget.tools + budget.reserved <= budget.total
    True

    Creating a custom budget:

    >>> custom_budget = ContentAllocationBudget(
    ...     total=16000,
    ...     system=2000,
    ...     user=4000,
    ...     assistant=4000,
    ...     tools=1000,
    ...     context=2000,
    ...     reserved=3000
    ... )
    >>> custom_budget.system
    2000

    Checking remaining tokens:

    >>> budget = ContentAllocationBudget(total=8000)
    >>> usage = {"system": 500, "user": 1000, "assistant": 800}
    >>> remaining = budget.remaining(usage)
    >>> remaining == budget.total - budget.reserved - 2300
    True

    Getting allocation for a content type:

    >>> from insideLLMs.contrib.context_window import ContentType
    >>> budget = ContentAllocationBudget(total=32000)
    >>> system_allocation = budget.allocation_for(ContentType.SYSTEM)
    >>> system_allocation == budget.system
    True
    """

    total: int
    system: int = 0
    user: int = 0
    assistant: int = 0
    tools: int = 0
    context: int = 0
    reserved: int = 0  # Reserved for response

    def __post_init__(self):
        """Initialize default allocations if not specified."""
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
        """
        Calculate remaining tokens based on current usage.

        Args
        ----
        current_usage : dict[str, int]
            Dictionary mapping content type names to token counts used.

        Returns
        -------
        int
            Number of tokens remaining (can be negative if over budget).

        Examples
        --------
        >>> budget = ContentAllocationBudget(total=8000)
        >>> usage = {"system": 500, "user": 1000}
        >>> budget.remaining(usage)  # doctest: +SKIP
        4500
        """
        used = sum(current_usage.values())
        return self.total - self.reserved - used

    def allocation_for(self, content_type: ContentType) -> int:
        """
        Get the token allocation for a specific content type.

        Args
        ----
        content_type : ContentType
            The type of content to get allocation for.

        Returns
        -------
        int
            Token allocation for the specified content type.

        Examples
        --------
        >>> budget = ContentAllocationBudget(total=32000)
        >>> budget.allocation_for(ContentType.SYSTEM) == budget.system
        True
        >>> budget.allocation_for(ContentType.TOOL_CALL) == budget.tools
        True
        """
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
        """
        Convert the budget to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all budget allocations.

        Examples
        --------
        >>> budget = ContentAllocationBudget(total=8000, system=1000, user=2000,
        ...                                   assistant=2000, tools=500, context=1000, reserved=1500)
        >>> d = budget.to_dict()
        >>> d["total"]
        8000
        >>> "reserved" in d
        True
        """
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
    """
    A block of content in the context window.

    ContextBlock is the fundamental unit of content storage in the context window.
    Each block contains the actual content along with metadata about its type,
    priority, token count, and other attributes used for context management.

    Attributes
    ----------
    content : str
        The actual text content of the block.
    content_type : ContentType
        The type of content (SYSTEM, USER, ASSISTANT, etc.).
    priority : PriorityLevel
        Priority level for truncation decisions. Default is MEDIUM.
    token_count : int
        Estimated token count. Auto-calculated if not provided.
    timestamp : datetime
        When the block was created. Auto-set to current time.
    metadata : dict[str, Any]
        Additional metadata (source, role, tags, etc.).
    compressed : bool
        Whether this block has been compressed.
    original_content : Optional[str]
        Original content before compression/truncation.
    block_id : str
        Unique identifier for the block. Auto-generated if not provided.

    Examples
    --------
    Creating a basic context block:

    >>> from insideLLMs.contrib.context_window import ContextBlock, ContentType, PriorityLevel
    >>> block = ContextBlock(
    ...     content="You are a helpful assistant.",
    ...     content_type=ContentType.SYSTEM
    ... )
    >>> block.priority
    <PriorityLevel.MEDIUM: 3>
    >>> block.token_count > 0
    True

    Creating a high-priority user message:

    >>> user_block = ContextBlock(
    ...     content="What is machine learning?",
    ...     content_type=ContentType.USER,
    ...     priority=PriorityLevel.HIGH,
    ...     metadata={"turn_number": 1, "user_id": "user123"}
    ... )
    >>> user_block.metadata["turn_number"]
    1

    Creating a critical system prompt:

    >>> system_block = ContextBlock(
    ...     content="IMPORTANT: Never reveal confidential information.",
    ...     content_type=ContentType.SYSTEM,
    ...     priority=PriorityLevel.CRITICAL
    ... )
    >>> system_block.priority == PriorityLevel.CRITICAL
    True

    Converting to dictionary for serialization:

    >>> block = ContextBlock("Hello", ContentType.USER)
    >>> d = block.to_dict()
    >>> d["content"]
    'Hello'
    >>> d["content_type"]
    'user'
    >>> "block_id" in d
    True
    """

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
        """Initialize token count and block ID if not provided."""
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)
        if not self.block_id:
            self.block_id = hashlib.md5(
                f"{self.content[:100]}{self.timestamp}".encode()
            ).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the context block to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all block attributes with enum values
            converted to their string/int representations.

        Examples
        --------
        >>> block = ContextBlock("Test content", ContentType.USER, PriorityLevel.HIGH)
        >>> d = block.to_dict()
        >>> d["content"]
        'Test content'
        >>> d["content_type"]
        'user'
        >>> d["priority"]
        4
        """
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
    """
    Result of a truncation operation on context blocks.

    Contains detailed information about what happened during truncation,
    including how many tokens and blocks were removed, the strategy used,
    and the resulting content.

    Attributes
    ----------
    original_tokens : int
        Total token count before truncation.
    final_tokens : int
        Total token count after truncation.
    tokens_removed : int
        Number of tokens removed during truncation.
    blocks_removed : int
        Number of complete blocks removed.
    blocks_truncated : int
        Number of blocks that were partially truncated.
    strategy_used : TruncationStrategy
        The truncation strategy that was applied.
    success : bool
        Whether truncation achieved the target token count.
    content : list[ContextBlock]
        The resulting context blocks after truncation.

    Examples
    --------
    Examining truncation results:

    >>> from insideLLMs.contrib.context_window import (
    ...     ContextTruncator, ContextBlock, ContentType, TruncationStrategy, PriorityLevel
    ... )
    >>> truncator = ContextTruncator()
    >>> blocks = [
    ...     ContextBlock("First block", ContentType.CONTEXT, PriorityLevel.HIGH),
    ...     ContextBlock("Second block", ContentType.CONTEXT, PriorityLevel.LOW),
    ...     ContextBlock("Third block", ContentType.CONTEXT, PriorityLevel.LOW),
    ... ]
    >>> result = truncator.truncate(blocks, target_tokens=20, strategy=TruncationStrategy.PRIORITY)
    >>> result.success
    True
    >>> result.tokens_removed >= 0
    True

    Checking if truncation was needed:

    >>> result = truncator.truncate(blocks, target_tokens=1000)
    >>> result.tokens_removed
    0
    >>> result.original_tokens == result.final_tokens
    True

    Analyzing truncation statistics:

    >>> result = truncator.truncate(blocks, target_tokens=15, strategy=TruncationStrategy.TAIL)
    >>> result.strategy_used
    <TruncationStrategy.TAIL: 'tail'>

    Converting to dictionary for logging:

    >>> result_dict = result.to_dict()
    >>> "original_tokens" in result_dict
    True
    >>> "strategy_used" in result_dict
    True
    """

    original_tokens: int
    final_tokens: int
    tokens_removed: int
    blocks_removed: int
    blocks_truncated: int
    strategy_used: TruncationStrategy
    success: bool
    content: list["ContextBlock"]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the truncation result to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all result attributes, with nested
            ContextBlocks also converted to dictionaries.

        Examples
        --------
        >>> from insideLLMs.contrib.context_window import TruncationResult, TruncationStrategy
        >>> result = TruncationResult(
        ...     original_tokens=100, final_tokens=50, tokens_removed=50,
        ...     blocks_removed=2, blocks_truncated=0, strategy_used=TruncationStrategy.PRIORITY,
        ...     success=True, content=[]
        ... )
        >>> d = result.to_dict()
        >>> d["tokens_removed"]
        50
        >>> d["strategy_used"]
        'priority'
        """
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
    """
    Result of a compression operation on context blocks.

    Contains detailed information about the compression operation including
    the compression ratio achieved and how many blocks were affected.

    Attributes
    ----------
    original_tokens : int
        Total token count before compression.
    compressed_tokens : int
        Total token count after compression.
    compression_ratio : float
        Ratio of compressed to original tokens (0.5 = 50% of original size).
    method_used : CompressionMethod
        The compression method that was applied.
    blocks_compressed : int
        Number of blocks that were actually compressed.
    success : bool
        Whether compression completed successfully.

    Examples
    --------
    Examining compression results:

    >>> from insideLLMs.contrib.context_window import (
    ...     ContextCompressor, ContextBlock, ContentType, CompressionMethod
    ... )
    >>> compressor = ContextCompressor()
    >>> blocks = [ContextBlock("This is repeated. This is repeated.", ContentType.CONTEXT)]
    >>> compressed_blocks, result = compressor.compress(blocks, method=CompressionMethod.REMOVE_REDUNDANCY)
    >>> result.success
    True
    >>> 0 < result.compression_ratio <= 1.0
    True

    Checking compression effectiveness:

    >>> blocks = [ContextBlock("Short text", ContentType.CONTEXT)]
    >>> _, result = compressor.compress(blocks, method=CompressionMethod.ABBREVIATE)
    >>> result.original_tokens >= result.compressed_tokens
    True

    No compression (passthrough):

    >>> _, result = compressor.compress(blocks, method=CompressionMethod.NONE)
    >>> result.compression_ratio
    1.0
    >>> result.blocks_compressed
    0

    Analyzing compression statistics:

    >>> long_text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    >>> blocks = [ContextBlock(long_text, ContentType.CONTEXT)]
    >>> _, result = compressor.compress(blocks, target_ratio=0.5, method=CompressionMethod.SUMMARIZE)
    >>> result.method_used
    <CompressionMethod.SUMMARIZE: 'summarize'>
    """

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method_used: CompressionMethod
    blocks_compressed: int
    success: bool

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the compression result to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all result attributes.

        Examples
        --------
        >>> from insideLLMs.contrib.context_window import ContextCompressionResult, CompressionMethod
        >>> result = ContextCompressionResult(
        ...     original_tokens=100, compressed_tokens=60, compression_ratio=0.6,
        ...     method_used=CompressionMethod.SUMMARIZE, blocks_compressed=3, success=True
        ... )
        >>> d = result.to_dict()
        >>> d["compression_ratio"]
        0.6
        >>> d["method_used"]
        'summarize'
        """
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
    """
    Current state snapshot of the context window.

    Provides a comprehensive view of the context window's current state,
    including token usage, available capacity, and overflow status.
    Useful for monitoring and debugging context management.

    Attributes
    ----------
    total_tokens : int
        Maximum tokens the context window can hold.
    used_tokens : int
        Number of tokens currently used.
    available_tokens : int
        Number of tokens available for new content.
    block_count : int
        Number of context blocks currently stored.
    usage_by_type : dict[str, int]
        Token usage breakdown by content type.
    budget : ContentAllocationBudget
        The token budget configuration.
    overflow : bool
        True if used tokens exceed available capacity.

    Examples
    --------
    Getting context window state:

    >>> from insideLLMs.contrib.context_window import ContextWindow, ContentType, PriorityLevel
    >>> window = ContextWindow(max_tokens=1000)
    >>> window.add("System prompt", ContentType.SYSTEM, PriorityLevel.HIGH)
    >>> window.add("User message", ContentType.USER)
    >>> state = window.get_state()
    >>> state.block_count
    2
    >>> state.used_tokens > 0
    True
    >>> state.overflow
    False

    Checking token usage by type:

    >>> state = window.get_state()
    >>> "system" in state.usage_by_type
    True
    >>> "user" in state.usage_by_type
    True

    Monitoring for overflow:

    >>> large_window = ContextWindow(max_tokens=100)
    >>> for i in range(20):
    ...     large_window.add(f"Message {i} with some content", ContentType.USER)
    >>> state = large_window.get_state()
    >>> # Window auto-truncates, so overflow should be False after truncation
    >>> isinstance(state.overflow, bool)
    True

    Converting to dictionary for logging:

    >>> state = window.get_state()
    >>> d = state.to_dict()
    >>> "total_tokens" in d
    True
    >>> "budget" in d
    True
    """

    total_tokens: int
    used_tokens: int
    available_tokens: int
    block_count: int
    usage_by_type: dict[str, int]
    budget: ContentAllocationBudget
    overflow: bool

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the context window state to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all state attributes, with nested
            budget also converted to a dictionary.

        Examples
        --------
        >>> from insideLLMs.contrib.context_window import ContextWindowState, ContentAllocationBudget
        >>> budget = ContentAllocationBudget(total=1000)
        >>> state = ContextWindowState(
        ...     total_tokens=1000, used_tokens=500, available_tokens=250,
        ...     block_count=5, usage_by_type={"user": 300, "system": 200},
        ...     budget=budget, overflow=False
        ... )
        >>> d = state.to_dict()
        >>> d["used_tokens"]
        500
        >>> d["overflow"]
        False
        """
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

    Uses an approximation of ~4 characters per token for English text.
    This is a fast estimation suitable for context window management
    but should not be used for precise token counting in production.

    For accurate token counting, use a proper tokenizer from the
    model provider (e.g., tiktoken for OpenAI models).

    Args
    ----
    text : str
        The text to estimate tokens for.

    Returns
    -------
    int
        Estimated number of tokens.

    Examples
    --------
    Estimating tokens for short text:

    >>> from insideLLMs.contrib.context_window import estimate_tokens
    >>> estimate_tokens("Hello, world!")
    4

    Estimating tokens for longer text:

    >>> tokens = estimate_tokens("This is a longer piece of text that spans multiple words.")
    >>> tokens > 10
    True

    Empty string returns zero:

    >>> estimate_tokens("")
    0

    Estimating for code content:

    >>> code = "def hello():\\n    print('Hello, World!')"
    >>> tokens = estimate_tokens(code)
    >>> tokens > 0
    True
    """
    return _canonical_estimate_tokens(text)


def find_semantic_boundaries(text: str) -> list[int]:
    """
    Find sentence and paragraph boundaries in text.

    Identifies natural breaking points in text that can be used for
    semantic truncation. This ensures truncation happens at sentence
    or paragraph boundaries rather than mid-word or mid-sentence.

    Args
    ----
    text : str
        The text to analyze for boundaries.

    Returns
    -------
    list[int]
        Sorted list of character positions representing boundaries.
        Always includes 0 (start) and len(text) (end).

    Examples
    --------
    Finding boundaries in simple text:

    >>> from insideLLMs.contrib.context_window import find_semantic_boundaries
    >>> text = "First sentence. Second sentence."
    >>> boundaries = find_semantic_boundaries(text)
    >>> 0 in boundaries
    True
    >>> len(text) in boundaries
    True

    Finding paragraph boundaries:

    >>> text = "First paragraph.\\n\\nSecond paragraph."
    >>> boundaries = find_semantic_boundaries(text)
    >>> len(boundaries) >= 2
    True

    Single sentence:

    >>> text = "Just one sentence here."
    >>> boundaries = find_semantic_boundaries(text)
    >>> boundaries[0]
    0
    >>> boundaries[-1] == len(text)
    True

    Text with multiple sentence endings:

    >>> text = "Question? Exclamation! Statement."
    >>> boundaries = find_semantic_boundaries(text)
    >>> len(boundaries) >= 3
    True
    """
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
    """
    Token counting utility with caching for improved performance.

    TokenCounter provides efficient token counting by caching results
    for previously counted text. This is particularly useful when the
    same content is counted multiple times during context management.

    Supports custom tokenizers (e.g., tiktoken) or falls back to
    estimation-based counting.

    Attributes
    ----------
    tokenizer : Optional[Callable[[str], list]]
        Custom tokenizer function that returns a list of tokens.
    _cache : dict[str, int]
        Internal cache mapping content hashes to token counts.
    _cache_hits : int
        Number of cache hits for statistics.
    _cache_misses : int
        Number of cache misses for statistics.

    Examples
    --------
    Basic token counting with default estimator:

    >>> from insideLLMs.contrib.context_window import TokenCounter
    >>> counter = TokenCounter()
    >>> tokens = counter.count("Hello, world!")
    >>> tokens > 0
    True

    Counting with caching:

    >>> counter = TokenCounter()
    >>> text = "This text will be counted multiple times."
    >>> count1 = counter.count(text)
    >>> count2 = counter.count(text)  # Uses cache
    >>> count1 == count2
    True
    >>> counter.get_stats()["cache_hits"] >= 1
    True

    Counting messages in a conversation:

    >>> counter = TokenCounter()
    >>> messages = [
    ...     {"role": "user", "content": "Hello!"},
    ...     {"role": "assistant", "content": "Hi there!"}
    ... ]
    >>> total = counter.count_messages(messages)
    >>> total > 0
    True

    Using a custom tokenizer:

    >>> def simple_tokenizer(text):
    ...     return text.split()  # Simple word-based tokenizer
    >>> counter = TokenCounter(tokenizer=simple_tokenizer)
    >>> counter.count("Hello world")
    2
    """

    def __init__(self, tokenizer: Optional[Callable[[str], list]] = None):
        """
        Initialize token counter.

        Args
        ----
        tokenizer : Optional[Callable[[str], list]]
            Optional custom tokenizer function that takes a string and
            returns a list of tokens. If not provided, uses estimation.

        Examples
        --------
        >>> counter = TokenCounter()  # Use default estimation
        >>> counter.count("test") > 0
        True
        """
        self.tokenizer = tokenizer
        self._cache: dict[str, int] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def count(self, text: str) -> int:
        """
        Count tokens in text with caching.

        Args
        ----
        text : str
            The text to count tokens for.

        Returns
        -------
        int
            Number of tokens in the text.

        Examples
        --------
        >>> counter = TokenCounter()
        >>> counter.count("Hello, how are you?")
        5
        >>> counter.count("")
        0
        """
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
        """
        Count tokens in a list of chat messages.

        Includes overhead for message structure (approximately 4 tokens
        per message for role and formatting).

        Args
        ----
        messages : list[dict]
            List of message dictionaries with 'content' key.

        Returns
        -------
        int
            Total token count including message overhead.

        Examples
        --------
        >>> counter = TokenCounter()
        >>> messages = [{"role": "user", "content": "Hi"}]
        >>> counter.count_messages(messages) >= 1
        True
        """
        total = 0
        for msg in messages:
            if isinstance(msg.get("content"), str):
                total += self.count(msg["content"])
            # Add overhead for message structure
            total += 4  # Approximate overhead per message
        return total

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict[str, Any]
            Dictionary with cache_hits, cache_misses, hit_rate, and cache_size.

        Examples
        --------
        >>> counter = TokenCounter()
        >>> counter.count("test")
        1
        >>> stats = counter.get_stats()
        >>> "cache_hits" in stats
        True
        >>> stats["cache_misses"] >= 1
        True
        """
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0,
            "cache_size": len(self._cache),
        }

    def clear_cache(self):
        """
        Clear the token cache and reset statistics.

        Examples
        --------
        >>> counter = TokenCounter()
        >>> counter.count("test")
        1
        >>> counter.clear_cache()
        >>> counter.get_stats()["cache_size"]
        0
        """
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class ContextTruncator:
    """
    Truncates context blocks using various strategies to fit token limits.

    ContextTruncator provides multiple strategies for reducing context size
    while attempting to preserve the most important information. Different
    strategies are suitable for different use cases.

    Attributes
    ----------
    token_counter : TokenCounter
        Token counter used for measuring content size.

    Examples
    --------
    Basic truncation with priority strategy:

    >>> from insideLLMs.contrib.context_window import (
    ...     ContextTruncator, ContextBlock, ContentType, PriorityLevel, TruncationStrategy
    ... )
    >>> truncator = ContextTruncator()
    >>> blocks = [
    ...     ContextBlock("Important", ContentType.SYSTEM, PriorityLevel.HIGH),
    ...     ContextBlock("Less important", ContentType.CONTEXT, PriorityLevel.LOW),
    ... ]
    >>> result = truncator.truncate(blocks, target_tokens=20)
    >>> result.success
    True

    Using tail truncation (keep beginning):

    >>> truncator = ContextTruncator()
    >>> blocks = [ContextBlock(f"Block {i}", ContentType.USER) for i in range(5)]
    >>> result = truncator.truncate(blocks, target_tokens=15, strategy=TruncationStrategy.TAIL)
    >>> result.strategy_used
    <TruncationStrategy.TAIL: 'tail'>

    Preserving critical content:

    >>> truncator = ContextTruncator()
    >>> blocks = [
    ...     ContextBlock("CRITICAL", ContentType.SYSTEM, PriorityLevel.CRITICAL),
    ...     ContextBlock("Optional", ContentType.CONTEXT, PriorityLevel.OPTIONAL),
    ... ]
    >>> result = truncator.truncate(blocks, target_tokens=10, preserve_critical=True)
    >>> any(b.priority == PriorityLevel.CRITICAL for b in result.content)
    True

    Semantic truncation at sentence boundaries:

    >>> truncator = ContextTruncator()
    >>> blocks = [ContextBlock("First. Second. Third.", ContentType.CONTEXT)]
    >>> result = truncator.truncate(blocks, target_tokens=8, strategy=TruncationStrategy.SEMANTIC)
    >>> result.strategy_used
    <TruncationStrategy.SEMANTIC: 'semantic'>
    """

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        """
        Initialize truncator.

        Args
        ----
        token_counter : Optional[TokenCounter]
            Token counter to use. Creates a default one if not provided.

        Examples
        --------
        >>> truncator = ContextTruncator()
        >>> truncator.token_counter is not None
        True
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

        Args
        ----
        blocks : list[ContextBlock]
            Context blocks to truncate.
        target_tokens : int
            Maximum token count for the result.
        strategy : TruncationStrategy
            Strategy to use for truncation. Default is PRIORITY.
        preserve_critical : bool
            Whether to preserve CRITICAL priority blocks regardless of
            token limit. Default is True.

        Returns
        -------
        TruncationResult
            Result containing truncated blocks and statistics.

        Examples
        --------
        >>> truncator = ContextTruncator()
        >>> blocks = [ContextBlock("Test", ContentType.USER)]
        >>> result = truncator.truncate(blocks, target_tokens=100)
        >>> result.success
        True
        >>> result.tokens_removed
        0
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
    """
    Compresses context content to reduce token usage.

    ContextCompressor provides multiple compression methods for reducing
    the size of context content while attempting to preserve important
    information. Different methods are suitable for different content types.

    Attributes
    ----------
    token_counter : TokenCounter
        Token counter used for measuring compression effectiveness.
    summarizer : Optional[Callable[[str], str]]
        Custom summarization function for SUMMARIZE method.

    Examples
    --------
    Basic compression with redundancy removal:

    >>> from insideLLMs.contrib.context_window import (
    ...     ContextCompressor, ContextBlock, ContentType, CompressionMethod
    ... )
    >>> compressor = ContextCompressor()
    >>> blocks = [ContextBlock("Hello hello world world", ContentType.CONTEXT)]
    >>> compressed, result = compressor.compress(blocks, method=CompressionMethod.REMOVE_REDUNDANCY)
    >>> result.success
    True

    Summarization compression:

    >>> compressor = ContextCompressor()
    >>> long_text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    >>> blocks = [ContextBlock(long_text, ContentType.CONTEXT)]
    >>> compressed, result = compressor.compress(blocks, target_ratio=0.5, method=CompressionMethod.SUMMARIZE)
    >>> result.method_used
    <CompressionMethod.SUMMARIZE: 'summarize'>

    Using abbreviation compression:

    >>> compressor = ContextCompressor()
    >>> blocks = [ContextBlock("For example, the application configuration is here.", ContentType.CONTEXT)]
    >>> compressed, result = compressor.compress(blocks, method=CompressionMethod.ABBREVIATE)
    >>> result.success
    True

    Custom summarizer function:

    >>> def my_summarizer(text):
    ...     return text[:50] + "..." if len(text) > 50 else text
    >>> compressor = ContextCompressor(summarizer=my_summarizer)
    >>> blocks = [ContextBlock("A" * 100, ContentType.CONTEXT)]
    >>> compressed, result = compressor.compress(blocks, method=CompressionMethod.SUMMARIZE)
    >>> len(compressed[0].content) < 100
    True
    """

    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
        summarizer: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize compressor.

        Args
        ----
        token_counter : Optional[TokenCounter]
            Token counter to use. Creates a default one if not provided.
        summarizer : Optional[Callable[[str], str]]
            Custom function to summarize text. Used when compression
            method is SUMMARIZE.

        Examples
        --------
        >>> compressor = ContextCompressor()
        >>> compressor.token_counter is not None
        True
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
        Compress context blocks using the specified method.

        Compresses blocks with priority at or below min_priority. Higher
        priority blocks are preserved unchanged.

        Args
        ----
        blocks : list[ContextBlock]
            Blocks to compress.
        target_ratio : float
            Target compression ratio (0.5 = 50% of original size).
            Only used by SUMMARIZE method.
        method : CompressionMethod
            Compression method to use. Default is REMOVE_REDUNDANCY.
        min_priority : PriorityLevel
            Only compress blocks at or below this priority level.
            Default is LOW.

        Returns
        -------
        tuple[list[ContextBlock], ContextCompressionResult]
            Tuple of (compressed blocks, compression statistics).

        Examples
        --------
        >>> compressor = ContextCompressor()
        >>> blocks = [ContextBlock("test test", ContentType.CONTEXT)]
        >>> compressed, result = compressor.compress(blocks)
        >>> result.success
        True
        >>> result.compression_ratio <= 1.0
        True
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
    Main context window manager for LLM applications.

    ContextWindow is the primary interface for managing context in LLM
    applications. It handles content storage, budget allocation, automatic
    truncation, and compression to ensure content fits within model limits.

    Features:
    - Automatic token budget management
    - Priority-based content management
    - Multiple truncation strategies
    - Content compression
    - Action history tracking

    Attributes
    ----------
    max_tokens : int
        Maximum tokens the context window can hold.
    budget : ContentAllocationBudget
        Token budget allocation for different content types.
    token_counter : TokenCounter
        Token counter used for measurements.
    default_strategy : TruncationStrategy
        Default strategy for automatic truncation.
    truncator : ContextTruncator
        Truncator instance for truncation operations.
    compressor : ContextCompressor
        Compressor instance for compression operations.

    Examples
    --------
    Basic context window usage:

    >>> from insideLLMs.contrib.context_window import ContextWindow, ContentType, PriorityLevel
    >>> window = ContextWindow(max_tokens=8000)
    >>> window.add("You are helpful.", ContentType.SYSTEM, PriorityLevel.CRITICAL)
    >>> window.add("Hello!", ContentType.USER)
    >>> window.get_used_tokens() > 0
    True

    Adding chat messages:

    >>> window = ContextWindow(max_tokens=4000)
    >>> window.add_message("system", "Be helpful.")
    >>> window.add_message("user", "Hi!")
    >>> window.add_message("assistant", "Hello!")
    >>> messages = window.get_messages()
    >>> len(messages)
    3

    Automatic truncation when over budget:

    >>> window = ContextWindow(max_tokens=100)
    >>> for i in range(20):
    ...     window.add(f"Message {i} content here", ContentType.USER)
    >>> window.get_used_tokens() <= 100 - window.budget.reserved
    True

    Manual truncation with specific strategy:

    >>> from insideLLMs.contrib.context_window import TruncationStrategy
    >>> window = ContextWindow(max_tokens=1000)
    >>> for i in range(10):
    ...     window.add(f"Content {i}", ContentType.CONTEXT)
    >>> result = window.truncate(target_tokens=50, strategy=TruncationStrategy.TAIL)
    >>> result.success
    True

    Content compression:

    >>> from insideLLMs.contrib.context_window import CompressionMethod
    >>> window = ContextWindow(max_tokens=1000)
    >>> window.add("Repeated word word content content here here", ContentType.CONTEXT)
    >>> result = window.compress(method=CompressionMethod.REMOVE_REDUNDANCY)
    >>> result.success
    True

    Getting context state:

    >>> window = ContextWindow(max_tokens=8000)
    >>> window.add("Test", ContentType.USER)
    >>> state = window.get_state()
    >>> state.block_count
    1
    >>> state.overflow
    False
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

        Args
        ----
        max_tokens : int
            Maximum tokens for the context window. Default is 128000.
        budget : Optional[ContentAllocationBudget]
            Token budget allocation. Creates default if not provided.
        token_counter : Optional[TokenCounter]
            Token counter to use. Creates default if not provided.
        default_strategy : TruncationStrategy
            Default truncation strategy. Default is PRIORITY.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=4000)
        >>> window.max_tokens
        4000
        >>> window.budget is not None
        True
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

        Creates a new ContextBlock and adds it to the window. If the total
        tokens exceed the budget (max_tokens - reserved), automatic truncation
        is triggered.

        Args
        ----
        content : str
            The text content to add.
        content_type : ContentType
            Type of content being added. Default is CONTEXT.
        priority : PriorityLevel
            Priority level for truncation decisions. Default is MEDIUM.
        metadata : Optional[dict]
            Additional metadata to store with the block.

        Returns
        -------
        ContextBlock
            The created context block.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=1000)
        >>> block = window.add("Hello, world!", ContentType.USER)
        >>> block.content
        'Hello, world!'

        >>> block = window.add("System prompt", ContentType.SYSTEM, PriorityLevel.CRITICAL)
        >>> block.priority
        <PriorityLevel.CRITICAL: 5>

        >>> block = window.add("Data", ContentType.CONTEXT, metadata={"source": "api"})
        >>> block.metadata["source"]
        'api'
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

        Convenience method for adding chat-style messages. Automatically maps
        roles to content types and assigns default priorities based on role.

        Args
        ----
        role : str
            Message role: "system", "user", or "assistant".
        content : str
            Message content.
        priority : Optional[PriorityLevel]
            Priority level. If not specified, defaults to HIGH for system
            messages and MEDIUM for user/assistant messages.
        metadata : Optional[dict]
            Additional metadata. Role is automatically added.

        Returns
        -------
        ContextBlock
            The created context block with role in metadata.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=4000)
        >>> block = window.add_message("system", "You are helpful.")
        >>> block.content_type
        <ContentType.SYSTEM: 'system'>
        >>> block.metadata["role"]
        'system'

        >>> block = window.add_message("user", "Hello!")
        >>> block.priority
        <PriorityLevel.MEDIUM: 3>

        >>> block = window.add_message("assistant", "Hi!", priority=PriorityLevel.HIGH)
        >>> block.priority
        <PriorityLevel.HIGH: 4>

        >>> messages = window.get_messages()
        >>> len(messages)
        3
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
        """
        Remove a block by its ID.

        Args
        ----
        block_id : str
            The unique identifier of the block to remove.

        Returns
        -------
        bool
            True if a block was found and removed, False otherwise.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=1000)
        >>> block = window.add("Test content", ContentType.USER)
        >>> block_id = block.block_id
        >>> window.remove(block_id)
        True
        >>> window.remove("nonexistent_id")
        False
        """
        for i, block in enumerate(self._blocks):
            if block.block_id == block_id:
                removed = self._blocks.pop(i)
                self._record_action("remove", removed)
                return True
        return False

    def clear(self, preserve_critical: bool = True):
        """
        Clear context window.

        Removes all blocks from the context window, optionally preserving
        blocks with CRITICAL priority.

        Args
        ----
        preserve_critical : bool
            If True, keeps blocks with CRITICAL priority. Default is True.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=1000)
        >>> window.add("Keep this", ContentType.SYSTEM, PriorityLevel.CRITICAL)
        >>> window.add("Remove this", ContentType.USER)
        >>> window.clear(preserve_critical=True)
        >>> len(window.get_blocks())
        1

        >>> window.clear(preserve_critical=False)
        >>> len(window.get_blocks())
        0
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
        Truncate context to fit within token limits.

        Removes or trims content blocks to meet the target token count.
        Always preserves blocks with CRITICAL priority.

        Args
        ----
        target_tokens : Optional[int]
            Target token count. Defaults to max_tokens minus reserved.
        strategy : Optional[TruncationStrategy]
            Truncation strategy to use. Uses default_strategy if not specified.

        Returns
        -------
        TruncationResult
            Result containing truncated blocks and statistics.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=1000)
        >>> for i in range(10):
        ...     window.add(f"Message {i}", ContentType.USER)
        >>> result = window.truncate(target_tokens=50)
        >>> result.success
        True
        >>> result.final_tokens <= 50
        True

        >>> result = window.truncate(strategy=TruncationStrategy.TAIL)
        >>> result.strategy_used
        <TruncationStrategy.TAIL: 'tail'>
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
        Compress context content to reduce token usage.

        Applies compression to eligible blocks (those with priority at or
        below LOW). High priority content is preserved unchanged.

        Args
        ----
        target_ratio : float
            Target compression ratio (0.5 = 50% of original). Only used
            by SUMMARIZE method.
        method : CompressionMethod
            Compression method to use. Default is REMOVE_REDUNDANCY.

        Returns
        -------
        ContextCompressionResult
            Result containing compression statistics.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=1000)
        >>> window.add("Same same content content here here", ContentType.CONTEXT)
        >>> result = window.compress(method=CompressionMethod.REMOVE_REDUNDANCY)
        >>> result.success
        True
        >>> result.compression_ratio <= 1.0
        True

        >>> window.add("For example, the application configuration", ContentType.CONTEXT)
        >>> result = window.compress(method=CompressionMethod.ABBREVIATE)
        >>> result.method_used
        <CompressionMethod.ABBREVIATE: 'abbreviate'>
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
        Get blocks, optionally filtered by type and/or priority.

        Args
        ----
        content_type : Optional[ContentType]
            If provided, only return blocks of this type.
        min_priority : Optional[PriorityLevel]
            If provided, only return blocks at or above this priority.

        Returns
        -------
        list[ContextBlock]
            List of matching context blocks.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=1000)
        >>> window.add("System", ContentType.SYSTEM, PriorityLevel.HIGH)
        >>> window.add("User", ContentType.USER, PriorityLevel.MEDIUM)
        >>> len(window.get_blocks())
        2

        >>> len(window.get_blocks(content_type=ContentType.USER))
        1

        >>> len(window.get_blocks(min_priority=PriorityLevel.HIGH))
        1
        """
        blocks = self._blocks

        if content_type:
            blocks = [b for b in blocks if b.content_type == content_type]

        if min_priority:
            blocks = [b for b in blocks if b.priority.value >= min_priority.value]

        return blocks

    def get_content(self, separator: str = "\n\n") -> str:
        """
        Get all content as a single concatenated string.

        Args
        ----
        separator : str
            String to use between blocks. Default is double newline.

        Returns
        -------
        str
            All block content joined by the separator.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=1000)
        >>> window.add("First", ContentType.USER)
        >>> window.add("Second", ContentType.USER)
        >>> window.get_content()
        'First\\n\\nSecond'
        >>> window.get_content(separator=" | ")
        'First | Second'
        """
        return separator.join(b.content for b in self._blocks)

    def get_messages(self) -> list[dict[str, str]]:
        """
        Get blocks as chat messages format.

        Only includes blocks that have a 'role' in their metadata
        (typically blocks added via add_message).

        Returns
        -------
        list[dict[str, str]]
            List of message dictionaries with 'role' and 'content' keys.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=4000)
        >>> window.add_message("system", "Be helpful.")
        >>> window.add_message("user", "Hi!")
        >>> messages = window.get_messages()
        >>> len(messages)
        2
        >>> messages[0]["role"]
        'system'
        >>> messages[1]["content"]
        'Hi!'
        """
        messages = []
        for block in self._blocks:
            role = block.metadata.get("role")
            if role:
                messages.append({"role": role, "content": block.content})
        return messages

    def get_used_tokens(self) -> int:
        """
        Get total tokens currently used in the context window.

        Returns
        -------
        int
            Sum of token counts from all blocks.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=1000)
        >>> window.add("Hello world", ContentType.USER)
        >>> window.get_used_tokens() > 0
        True
        """
        return sum(b.token_count for b in self._blocks)

    def get_available_tokens(self) -> int:
        """
        Get number of tokens available for new content.

        Calculated as max_tokens minus reserved tokens minus used tokens.

        Returns
        -------
        int
            Available token count.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=1000)
        >>> initial_available = window.get_available_tokens()
        >>> window.add("Some content", ContentType.USER)
        >>> window.get_available_tokens() < initial_available
        True
        """
        return self.max_tokens - self.budget.reserved - self.get_used_tokens()

    def get_state(self) -> ContextWindowState:
        """
        Get current context window state snapshot.

        Returns a comprehensive view of the context window's current state
        including token usage, available capacity, and usage breakdown.

        Returns
        -------
        ContextWindowState
            State object with usage information.

        Examples
        --------
        >>> window = ContextWindow(max_tokens=8000)
        >>> window.add("Test", ContentType.USER)
        >>> state = window.get_state()
        >>> state.block_count
        1
        >>> state.total_tokens
        8000
        >>> state.overflow
        False
        """
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
    Manages multi-turn conversation context with automatic summarization.

    ConversationManager provides specialized handling for conversation history,
    automatically summarizing older messages when the conversation exceeds
    a configurable threshold to manage context window usage.

    Attributes
    ----------
    context_window : ContextWindow
        Underlying context window for storage.
    max_turns : int
        Maximum conversation turns to keep.
    summarize_after : int
        Summarize messages after this many turns.
    summarizer : Optional[Callable[[list[dict]], str]]
        Custom function to summarize turns.

    Examples
    --------
    Basic conversation management:

    >>> from insideLLMs.contrib.context_window import ConversationManager
    >>> manager = ConversationManager(max_turns=50)
    >>> manager.add_turn("system", "You are helpful.")
    >>> manager.add_turn("user", "Hello!")
    >>> manager.add_turn("assistant", "Hi! How can I help?")
    >>> len(manager.get_turns())
    3

    Getting formatted messages for model:

    >>> messages = manager.get_context_for_model()
    >>> len(messages)
    3
    >>> messages[0]["role"]
    'system'

    Automatic summarization (with custom summarizer):

    >>> def my_summarizer(turns):
    ...     return f"Summary of {len(turns)} turns"
    >>> manager = ConversationManager(summarize_after=5, summarizer=my_summarizer)
    >>> for i in range(10):
    ...     manager.add_turn("user", f"Message {i}")
    >>> manager.get_stats()["has_summary"]
    True

    Getting conversation statistics:

    >>> manager = ConversationManager()
    >>> manager.add_turn("user", "Test")
    >>> stats = manager.get_stats()
    >>> stats["total_turns"]
    1
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

        Args
        ----
        context_window : Optional[ContextWindow]
            Context window to use. Creates default if not provided.
        max_turns : int
            Maximum conversation turns to keep. Default is 50.
        summarize_after : int
            Summarize messages after this many turns. Default is 20.
        summarizer : Optional[Callable[[list[dict]], str]]
            Custom function to summarize messages. If not provided,
            uses a simple default summarizer.

        Examples
        --------
        >>> manager = ConversationManager(max_turns=100, summarize_after=30)
        >>> manager.max_turns
        100
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

        Adds a new turn to the conversation history. Automatically triggers
        summarization if the turn count exceeds summarize_after threshold.

        Args
        ----
        role : str
            Message role: "system", "user", or "assistant".
        content : str
            Message content.
        metadata : Optional[dict]
            Additional metadata for this turn.

        Returns
        -------
        dict[str, Any]
            The created turn dictionary with role, content, timestamp,
            turn_number, and metadata.

        Examples
        --------
        >>> manager = ConversationManager()
        >>> turn = manager.add_turn("user", "Hello!")
        >>> turn["role"]
        'user'
        >>> turn["turn_number"]
        1
        >>> "timestamp" in turn
        True
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
        """
        Get conversation turns.

        Args
        ----
        limit : Optional[int]
            If provided, return only the most recent N turns.

        Returns
        -------
        list[dict[str, Any]]
            List of turn dictionaries.

        Examples
        --------
        >>> manager = ConversationManager()
        >>> manager.add_turn("user", "First")
        >>> manager.add_turn("assistant", "Second")
        >>> manager.add_turn("user", "Third")
        >>> len(manager.get_turns())
        3
        >>> len(manager.get_turns(limit=2))
        2
        """
        if limit:
            return self._turns[-limit:]
        return self._turns.copy()

    def get_context_for_model(self, max_tokens: Optional[int] = None) -> list[dict[str, str]]:
        """
        Get conversation context formatted for model API.

        Returns a list of messages suitable for sending to an LLM API.
        Includes any accumulated summary as a system message prefix.

        Args
        ----
        max_tokens : Optional[int]
            If provided, truncates to fit within this token limit
            by removing oldest non-system messages.

        Returns
        -------
        list[dict[str, str]]
            List of message dictionaries with 'role' and 'content' keys.

        Examples
        --------
        >>> manager = ConversationManager()
        >>> manager.add_turn("system", "Be helpful.")
        >>> manager.add_turn("user", "Hello!")
        >>> messages = manager.get_context_for_model()
        >>> messages[0]["role"]
        'system'
        >>> messages[1]["content"]
        'Hello!'

        >>> messages = manager.get_context_for_model(max_tokens=10)
        >>> len(messages) >= 1
        True
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

        Removes all turns from history, optionally preserving system messages.
        Also clears any accumulated summary.

        Args
        ----
        keep_system : bool
            If True, keeps turns with role "system". Default is True.

        Examples
        --------
        >>> manager = ConversationManager()
        >>> manager.add_turn("system", "Be helpful.")
        >>> manager.add_turn("user", "Hello!")
        >>> manager.clear(keep_system=True)
        >>> len(manager.get_turns())
        1
        >>> manager.get_turns()[0]["role"]
        'system'

        >>> manager.clear(keep_system=False)
        >>> len(manager.get_turns())
        0
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
        """
        Get conversation statistics.

        Returns
        -------
        dict[str, Any]
            Dictionary with conversation metrics including total_turns,
            active_turns, summarized_turns, has_summary, context_tokens,
            and available_tokens.

        Examples
        --------
        >>> manager = ConversationManager()
        >>> manager.add_turn("user", "Hello")
        >>> stats = manager.get_stats()
        >>> stats["total_turns"]
        1
        >>> stats["has_summary"]
        False
        """
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

    Maintains a fixed-size window of recent content, automatically archiving
    older items when the window exceeds its capacity. Useful for streaming
    content or maintaining a rolling context.

    Attributes
    ----------
    window_size : int
        Maximum number of items in the active window.
    overlap : int
        Number of items to keep when sliding (for context continuity).
    token_counter : TokenCounter
        Token counter for measurements.

    Examples
    --------
    Basic sliding window usage:

    >>> from insideLLMs.contrib.context_window import SlidingWindowManager
    >>> slider = SlidingWindowManager(window_size=5, overlap=1)
    >>> for i in range(7):
    ...     slider.add(f"Item {i}")
    >>> len(slider.get_window())
    5
    >>> len(slider.get_archived())
    2

    Getting window content:

    >>> slider = SlidingWindowManager(window_size=3)
    >>> slider.add("First")
    >>> slider.add("Second")
    >>> slider.add("Third")
    >>> "First" in slider.get_content()
    True

    Window statistics:

    >>> slider = SlidingWindowManager(window_size=5)
    >>> slider.add("Test content")
    >>> stats = slider.get_stats()
    >>> stats["current_items"]
    1
    >>> stats["window_tokens"] > 0
    True

    Including archived content:

    >>> slider = SlidingWindowManager(window_size=2)
    >>> for i in range(5):
    ...     slider.add(f"Item {i}")
    >>> len(slider.get_content(include_archived=True)) > len(slider.get_content())
    True
    """

    def __init__(
        self,
        window_size: int = 10,
        overlap: int = 2,
        token_counter: Optional[TokenCounter] = None,
    ):
        """
        Initialize sliding window.

        Args
        ----
        window_size : int
            Maximum number of items in the active window. Default is 10.
        overlap : int
            Items to retain for context continuity when sliding. Default is 2.
        token_counter : Optional[TokenCounter]
            Token counter to use. Creates default if not provided.

        Examples
        --------
        >>> slider = SlidingWindowManager(window_size=20, overlap=3)
        >>> slider.window_size
        20
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
        """
        Add item to sliding window.

        If adding this item exceeds window_size, older items are automatically
        archived to make room.

        Args
        ----
        content : str
            Content to add.
        content_type : ContentType
            Type of content. Default is CONTEXT.
        priority : PriorityLevel
            Priority level. Default is MEDIUM.
        metadata : Optional[dict]
            Additional metadata.

        Returns
        -------
        ContextBlock
            The created context block.

        Examples
        --------
        >>> slider = SlidingWindowManager(window_size=3)
        >>> block = slider.add("Test content")
        >>> block.content
        'Test content'
        """
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


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import CompressionResult. The canonical name is
# ContextCompressionResult.
CompressionResult = ContextCompressionResult

# Older code and tests may import TokenBudget. The canonical name is
# ContentAllocationBudget.
TokenBudget = ContentAllocationBudget
