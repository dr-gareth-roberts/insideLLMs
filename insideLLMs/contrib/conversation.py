"""Multi-turn conversation analysis for LLM evaluation.

This module provides comprehensive tools for analyzing multi-turn conversations
with Large Language Models (LLMs). It supports turn-level quality assessment,
topic tracking, consistency checking, and engagement analysis.

Key Features:
    - Turn-level analysis (quality, relevance, coherence per turn)
    - Conversation flow tracking (topic drift, context maintenance)
    - Memory and consistency evaluation across turns
    - Engagement and interaction patterns
    - Conversation summarization and metadata extraction

Main Classes:
    - Conversation: Container for multi-turn conversation messages
    - ConversationAnalyzer: Comprehensive analyzer for full conversations
    - TurnAnalyzer: Analyzes individual conversation turns
    - TopicTracker: Tracks topics and transitions across turns
    - ConversationConsistencyChecker: Checks consistency across responses
    - EngagementAnalyzer: Analyzes user engagement patterns

Data Classes:
    - ConversationMessage: Single message in a conversation
    - ConversationTurn: A user-assistant exchange pair
    - MemoryReference: Reference to earlier conversation content
    - TopicAnalysis: Analysis results for topic tracking
    - ConsistencyAnalysis: Analysis results for consistency checking
    - EngagementMetrics: Metrics for engagement analysis
    - ConversationReport: Comprehensive analysis report

Enums:
    - MessageRole: USER, ASSISTANT, SYSTEM
    - TurnQuality: EXCELLENT, GOOD, ACCEPTABLE, POOR, FAILED
    - ConversationState: STARTING, FLOWING, CLARIFYING, REDIRECTING, STALLED, CONCLUDING
    - TopicTransition: CONTINUATION, NATURAL_SHIFT, EXPLICIT_CHANGE, ABRUPT_CHANGE, RETURN_TO_PREVIOUS

Examples:
    Basic conversation creation and analysis:

    >>> from insideLLMs.contrib.conversation import Conversation, ConversationAnalyzer
    >>> conv = Conversation()
    >>> conv.add_user_message("What is machine learning?")
    >>> conv.add_assistant_message("Machine learning is a subset of AI...")
    >>> conv.add_user_message("How does it differ from deep learning?")
    >>> conv.add_assistant_message("Deep learning is a subset of ML...")
    >>> analyzer = ConversationAnalyzer()
    >>> report = analyzer.analyze(conv)
    >>> print(f"Quality: {report.quality_level}")
    Quality: good

    Quick analysis using convenience functions:

    >>> from insideLLMs.contrib.conversation import analyze_messages
    >>> messages = [
    ...     {"role": "user", "content": "Hello, how are you?"},
    ...     {"role": "assistant", "content": "I'm doing well, thank you!"},
    ... ]
    >>> report = analyze_messages(messages)
    >>> print(f"Turns: {report.n_turns}, Score: {report.overall_quality_score:.2f}")
    Turns: 1, Score: 0.65

    Tracking topics across a conversation:

    >>> from insideLLMs.contrib.conversation import TopicTracker
    >>> tracker = TopicTracker()
    >>> tracker.add_turn(1, "Tell me about Python", "Python is a programming language...")
    'python'
    >>> tracker.add_turn(2, "What about its libraries?", "Python has many libraries...")
    'libraries'
    >>> analysis = tracker.analyze()
    >>> print(f"Topics: {analysis.main_topics}")
    Topics: ['python', 'libraries']

    Checking engagement patterns:

    >>> from insideLLMs.contrib.conversation import Conversation, EngagementAnalyzer
    >>> conv = Conversation([
    ...     {"role": "user", "content": "Can you help me with Python?"},
    ...     {"role": "assistant", "content": "Of course! What do you need?"},
    ...     {"role": "user", "content": "Thanks! That's very helpful."},
    ...     {"role": "assistant", "content": "You're welcome!"},
    ... ])
    >>> analyzer = EngagementAnalyzer()
    >>> metrics = analyzer.analyze(conv.get_turns())
    >>> print(f"Satisfaction indicators: {metrics.user_satisfaction_indicators}")
    Satisfaction indicators: 1
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class MessageRole(Enum):
    """Role of a message in a conversation.

    This enum defines the three standard roles in LLM conversations:
    user messages, assistant responses, and system prompts.

    Attributes:
        USER: A message from the human user.
        ASSISTANT: A response from the AI assistant.
        SYSTEM: A system-level instruction or prompt.

    Examples:
        Creating messages with different roles:

        >>> role = MessageRole.USER
        >>> print(role.value)
        'user'

        Using in conditionals:

        >>> msg_role = MessageRole.ASSISTANT
        >>> if msg_role == MessageRole.ASSISTANT:
        ...     print("This is an AI response")
        This is an AI response

        Iterating over all roles:

        >>> for role in MessageRole:
        ...     print(role.value)
        user
        assistant
        system

        Converting from string:

        >>> MessageRole("user")
        <MessageRole.USER: 'user'>
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TurnQuality(Enum):
    """Quality rating for a conversation turn.

    This enum represents the quality levels assigned to individual
    conversation turns based on relevance and coherence scores.

    Quality levels are determined by the average of relevance and
    coherence scores:
        - EXCELLENT: score >= 0.8
        - GOOD: score >= 0.6
        - ACCEPTABLE: score >= 0.4
        - POOR: score >= 0.2
        - FAILED: score < 0.2

    Attributes:
        EXCELLENT: Outstanding turn with high relevance and coherence.
        GOOD: Above average quality with minor issues.
        ACCEPTABLE: Meets basic quality standards.
        POOR: Below average quality with notable issues.
        FAILED: Unacceptable quality, response not useful.

    Examples:
        Checking turn quality:

        >>> quality = TurnQuality.EXCELLENT
        >>> print(quality.value)
        'excellent'

        Comparing quality levels:

        >>> TurnQuality.GOOD.value == "good"
        True

        Using in quality checks:

        >>> quality = TurnQuality.POOR
        >>> if quality in [TurnQuality.POOR, TurnQuality.FAILED]:
        ...     print("Quality needs improvement")
        Quality needs improvement

        Quality level ordering (manual comparison):

        >>> quality_order = [TurnQuality.FAILED, TurnQuality.POOR,
        ...                  TurnQuality.ACCEPTABLE, TurnQuality.GOOD,
        ...                  TurnQuality.EXCELLENT]
        >>> quality_order.index(TurnQuality.GOOD) > quality_order.index(TurnQuality.POOR)
        True
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


class ConversationState(Enum):
    """State of conversation flow.

    This enum represents the current state of a conversation's flow,
    helping to identify patterns and potential issues in the dialogue.

    State transitions are determined by analyzing turn quality, topic
    drift, and user intent signals like gratitude or frustration.

    Attributes:
        STARTING: Conversation has just begun (1 turn or less).
        FLOWING: Normal conversation progression with good quality.
        CLARIFYING: User or assistant seeking clarification.
        REDIRECTING: Topic drift detected, conversation changing direction.
        STALLED: Low quality turns, conversation not progressing well.
        CONCLUDING: End signals detected (thanks, goodbye, etc.).

    Examples:
        Checking conversation state:

        >>> state = ConversationState.FLOWING
        >>> print(state.value)
        'flowing'

        State-based logic:

        >>> state = ConversationState.STALLED
        >>> if state == ConversationState.STALLED:
        ...     print("Consider intervention or topic change")
        Consider intervention or topic change

        Detecting conversation end:

        >>> state = ConversationState.CONCLUDING
        >>> if state in [ConversationState.CONCLUDING, ConversationState.STALLED]:
        ...     print("Conversation may be ending")
        Conversation may be ending

        All possible states:

        >>> [s.value for s in ConversationState]
        ['starting', 'flowing', 'clarifying', 'redirecting', 'stalled', 'concluding']
    """

    STARTING = "starting"
    FLOWING = "flowing"
    CLARIFYING = "clarifying"
    REDIRECTING = "redirecting"
    STALLED = "stalled"
    CONCLUDING = "concluding"


class TopicTransition(Enum):
    """Type of topic transition between conversation turns.

    This enum categorizes how topics change between consecutive turns
    in a conversation. It helps identify topic drift and conversation
    coherence patterns.

    Attributes:
        CONTINUATION: Same topic continues from previous turn.
        NATURAL_SHIFT: Related topic emerges naturally from discussion.
        EXPLICIT_CHANGE: User explicitly changes to a new topic.
        ABRUPT_CHANGE: Sudden, unrelated topic change (may indicate issues).
        RETURN_TO_PREVIOUS: Conversation returns to an earlier topic.

    Examples:
        Checking transition type:

        >>> transition = TopicTransition.CONTINUATION
        >>> print(transition.value)
        'continuation'

        Detecting problematic transitions:

        >>> transition = TopicTransition.ABRUPT_CHANGE
        >>> if transition == TopicTransition.ABRUPT_CHANGE:
        ...     print("Topic drift detected - may indicate confusion")
        Topic drift detected - may indicate confusion

        Categorizing transitions:

        >>> smooth_transitions = [TopicTransition.CONTINUATION,
        ...                       TopicTransition.NATURAL_SHIFT]
        >>> transition = TopicTransition.NATURAL_SHIFT
        >>> transition in smooth_transitions
        True

        Checking for topic return:

        >>> transition = TopicTransition.RETURN_TO_PREVIOUS
        >>> if transition == TopicTransition.RETURN_TO_PREVIOUS:
        ...     print("User returned to an earlier topic")
        User returned to an earlier topic
    """

    CONTINUATION = "continuation"
    NATURAL_SHIFT = "natural_shift"
    EXPLICIT_CHANGE = "explicit_change"
    ABRUPT_CHANGE = "abrupt_change"
    RETURN_TO_PREVIOUS = "return_to_previous"


@dataclass
class ConversationMessage:
    """A single message in a conversation.

    This dataclass represents an individual message within a conversation,
    containing the role (user/assistant/system), content, and optional
    metadata like timestamps.

    Attributes:
        role: The role of the message sender (MessageRole enum).
        content: The text content of the message.
        timestamp: Optional Unix timestamp when message was created.
        metadata: Optional dictionary for additional message properties.

    Properties:
        word_count: Number of words in the message content.
        char_count: Number of characters in the message content.

    Examples:
        Creating a user message:

        >>> msg = ConversationMessage(
        ...     role=MessageRole.USER,
        ...     content="Hello, how are you?"
        ... )
        >>> print(msg.word_count)
        4

        Creating a message with timestamp:

        >>> import time
        >>> msg = ConversationMessage(
        ...     role=MessageRole.ASSISTANT,
        ...     content="I am doing well, thank you for asking!",
        ...     timestamp=time.time()
        ... )
        >>> msg.char_count
        39

        Adding metadata to a message:

        >>> msg = ConversationMessage(
        ...     role=MessageRole.USER,
        ...     content="What is Python?",
        ...     metadata={"source": "web_chat", "user_id": "12345"}
        ... )
        >>> msg.metadata["source"]
        'web_chat'

        Converting to dictionary:

        >>> msg = ConversationMessage(
        ...     role=MessageRole.USER,
        ...     content="Hello"
        ... )
        >>> d = msg.to_dict()
        >>> d["role"]
        'user'
        >>> d["word_count"]
        1
    """

    role: MessageRole
    content: str
    timestamp: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Get word count of message.

        Returns:
            int: Number of space-separated words in the content.

        Examples:
            >>> msg = ConversationMessage(MessageRole.USER, "Hello world")
            >>> msg.word_count
            2
        """
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        """Get character count of message.

        Returns:
            int: Total number of characters including spaces.

        Examples:
            >>> msg = ConversationMessage(MessageRole.USER, "Hello")
            >>> msg.char_count
            5
        """
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary representation.

        Returns:
            dict: Dictionary containing all message fields including
                computed properties (word_count, char_count).

        Examples:
            >>> msg = ConversationMessage(
            ...     role=MessageRole.ASSISTANT,
            ...     content="Hello there!"
            ... )
            >>> d = msg.to_dict()
            >>> d["role"]
            'assistant'
            >>> d["word_count"]
            2
        """
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "metadata": self.metadata,
        }


@dataclass
class ConversationTurn:
    """A conversation turn consisting of a user message and assistant response.

    This dataclass represents a complete exchange in a conversation: one user
    message paired with its corresponding assistant response. It includes
    quality metrics and topic information for analysis purposes.

    Attributes:
        turn_number: The sequential number of this turn (1-indexed).
        user_message: The ConversationMessage from the user.
        assistant_response: The ConversationMessage from the assistant.
        topic: Detected topic for this turn (set by TopicTracker).
        quality: Quality rating for the turn (TurnQuality enum).
        relevance_score: How relevant the response is to the query (0-1).
        coherence_score: How coherent the response is (0-1).
        metadata: Optional dictionary for additional turn properties.

    Properties:
        response_ratio: Ratio of assistant words to user words.
        turn_score: Average of relevance and coherence scores.

    Examples:
        Creating a basic turn:

        >>> user_msg = ConversationMessage(MessageRole.USER, "What is Python?")
        >>> asst_msg = ConversationMessage(
        ...     MessageRole.ASSISTANT,
        ...     "Python is a high-level programming language."
        ... )
        >>> turn = ConversationTurn(
        ...     turn_number=1,
        ...     user_message=user_msg,
        ...     assistant_response=asst_msg
        ... )
        >>> turn.turn_number
        1

        Checking response ratio:

        >>> user_msg = ConversationMessage(MessageRole.USER, "Hello")
        >>> asst_msg = ConversationMessage(
        ...     MessageRole.ASSISTANT,
        ...     "Hello! How can I help you today?"
        ... )
        >>> turn = ConversationTurn(1, user_msg, asst_msg)
        >>> turn.response_ratio  # 7 words / 1 word
        7.0

        Setting quality metrics:

        >>> turn = ConversationTurn(
        ...     turn_number=1,
        ...     user_message=user_msg,
        ...     assistant_response=asst_msg,
        ...     quality=TurnQuality.GOOD,
        ...     relevance_score=0.8,
        ...     coherence_score=0.9
        ... )
        >>> turn.turn_score
        0.85

        Converting to dictionary for serialization:

        >>> d = turn.to_dict()
        >>> d["turn_number"]
        1
        >>> d["quality"]
        'good'
    """

    turn_number: int
    user_message: ConversationMessage
    assistant_response: ConversationMessage
    topic: Optional[str] = None
    quality: TurnQuality = TurnQuality.ACCEPTABLE
    relevance_score: float = 0.5
    coherence_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def response_ratio(self) -> float:
        """Calculate response length ratio (assistant words / user words).

        Returns:
            float: Ratio of assistant response word count to user message
                word count. Returns 0.0 if user message is empty.

        Examples:
            >>> user = ConversationMessage(MessageRole.USER, "Hi")
            >>> asst = ConversationMessage(MessageRole.ASSISTANT, "Hello there friend")
            >>> turn = ConversationTurn(1, user, asst)
            >>> turn.response_ratio
            3.0
        """
        if self.user_message.word_count == 0:
            return 0.0
        return self.assistant_response.word_count / self.user_message.word_count

    @property
    def turn_score(self) -> float:
        """Calculate overall turn score as average of relevance and coherence.

        Returns:
            float: Score between 0 and 1 representing overall turn quality.

        Examples:
            >>> user = ConversationMessage(MessageRole.USER, "Hello")
            >>> asst = ConversationMessage(MessageRole.ASSISTANT, "Hi!")
            >>> turn = ConversationTurn(1, user, asst, relevance_score=0.8, coherence_score=0.6)
            >>> turn.turn_score
            0.7
        """
        return (self.relevance_score + self.coherence_score) / 2

    def to_dict(self) -> dict[str, Any]:
        """Convert turn to dictionary representation.

        Returns:
            dict: Dictionary containing all turn fields, nested message
                dictionaries, and computed properties.

        Examples:
            >>> user = ConversationMessage(MessageRole.USER, "Test")
            >>> asst = ConversationMessage(MessageRole.ASSISTANT, "Response")
            >>> turn = ConversationTurn(1, user, asst, topic="testing")
            >>> d = turn.to_dict()
            >>> d["topic"]
            'testing'
            >>> d["user_message"]["content"]
            'Test'
        """
        return {
            "turn_number": self.turn_number,
            "user_message": self.user_message.to_dict(),
            "assistant_response": self.assistant_response.to_dict(),
            "topic": self.topic,
            "quality": self.quality.value,
            "relevance_score": self.relevance_score,
            "coherence_score": self.coherence_score,
            "response_ratio": self.response_ratio,
            "turn_score": self.turn_score,
            "metadata": self.metadata,
        }


@dataclass
class MemoryReference:
    """A reference to information from earlier in the conversation.

    This dataclass tracks when a later turn in a conversation references
    or recalls information from an earlier turn. It's used for evaluating
    how well the LLM maintains context and memory across a conversation.

    Attributes:
        source_turn: The turn number where the original information appeared.
        target_turn: The turn number where the reference is made.
        reference_type: Category of reference (e.g., "entity", "fact",
            "instruction", "content_overlap").
        content: Description or content of what is being referenced.
        is_accurate: Whether the reference accurately reflects the source.
        confidence: Confidence score for the detected reference (0-1).

    Examples:
        Creating a memory reference:

        >>> ref = MemoryReference(
        ...     source_turn=1,
        ...     target_turn=3,
        ...     reference_type="entity",
        ...     content="User's name: John",
        ...     is_accurate=True,
        ...     confidence=0.95
        ... )
        >>> ref.source_turn
        1

        Tracking factual references:

        >>> ref = MemoryReference(
        ...     source_turn=2,
        ...     target_turn=5,
        ...     reference_type="fact",
        ...     content="Python was created by Guido van Rossum",
        ...     is_accurate=True,
        ...     confidence=0.8
        ... )
        >>> ref.is_accurate
        True

        Detecting inaccurate references:

        >>> ref = MemoryReference(
        ...     source_turn=1,
        ...     target_turn=4,
        ...     reference_type="instruction",
        ...     content="User requested JSON format",
        ...     is_accurate=False,  # Response used XML instead
        ...     confidence=0.7
        ... )
        >>> if not ref.is_accurate:
        ...     print("Inconsistency detected")
        Inconsistency detected

        Converting for analysis:

        >>> d = ref.to_dict()
        >>> d["reference_type"]
        'instruction'
    """

    source_turn: int
    target_turn: int
    reference_type: str  # e.g., "entity", "fact", "instruction"
    content: str
    is_accurate: bool = True
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert memory reference to dictionary representation.

        Returns:
            dict: Dictionary containing all reference fields for
                serialization or further analysis.

        Examples:
            >>> ref = MemoryReference(1, 3, "fact", "Important detail")
            >>> d = ref.to_dict()
            >>> d["source_turn"]
            1
            >>> d["confidence"]
            0.5
        """
        return {
            "source_turn": self.source_turn,
            "target_turn": self.target_turn,
            "reference_type": self.reference_type,
            "content": self.content,
            "is_accurate": self.is_accurate,
            "confidence": self.confidence,
        }


@dataclass
class TopicAnalysis:
    """Analysis of topics discussed throughout a conversation.

    This dataclass contains comprehensive topic tracking results including
    what topics were discussed, in what order, how they transitioned, and
    how much attention each received.

    Attributes:
        main_topics: List of unique topics in order of first appearance.
        topic_sequence: List of (turn_number, topic) tuples showing topic
            progression through the conversation.
        topic_transitions: List of (turn_number, TopicTransition) tuples
            describing how topics changed between turns.
        topic_coverage: Dictionary mapping topic to number of turns spent on it.
        topic_depth: Dictionary mapping topic to depth score (0-1), indicating
            how thoroughly the topic was explored.

    Properties:
        n_topics: Number of distinct topics discussed.
        avg_topic_duration: Average number of turns spent per topic.
        has_topic_drift: True if conversation has >1 abrupt topic changes.

    Examples:
        Examining topic analysis results:

        >>> analysis = TopicAnalysis(
        ...     main_topics=["python", "machine_learning"],
        ...     topic_sequence=[(1, "python"), (2, "python"), (3, "machine_learning")],
        ...     topic_transitions=[(2, TopicTransition.CONTINUATION),
        ...                        (3, TopicTransition.NATURAL_SHIFT)],
        ...     topic_coverage={"python": 2, "machine_learning": 1},
        ...     topic_depth={"python": 0.67, "machine_learning": 0.33}
        ... )
        >>> analysis.n_topics
        2

        Checking for topic drift:

        >>> analysis.has_topic_drift
        False

        Getting average topic duration:

        >>> analysis.avg_topic_duration
        1.5

        Analyzing topic coverage:

        >>> analysis.topic_coverage["python"]
        2

        Converting for reporting:

        >>> d = analysis.to_dict()
        >>> len(d["main_topics"])
        2
        >>> d["topic_sequence"][0]["topic"]
        'python'
    """

    main_topics: list[str]
    topic_sequence: list[tuple[int, str]]  # (turn_number, topic)
    topic_transitions: list[tuple[int, TopicTransition]]
    topic_coverage: dict[str, int]  # topic -> turn count
    topic_depth: dict[str, float]  # topic -> depth score

    @property
    def n_topics(self) -> int:
        """Get number of distinct topics discussed.

        Returns:
            int: Count of unique topics in the conversation.

        Examples:
            >>> analysis = TopicAnalysis(["a", "b", "c"], [], [], {}, {})
            >>> analysis.n_topics
            3
        """
        return len(self.main_topics)

    @property
    def avg_topic_duration(self) -> float:
        """Calculate average number of turns spent per topic.

        Returns:
            float: Average turns per topic. Returns 0.0 if no topics.

        Examples:
            >>> analysis = TopicAnalysis(
            ...     ["a", "b"], [], [],
            ...     {"a": 3, "b": 1}, {}
            ... )
            >>> analysis.avg_topic_duration
            2.0
        """
        if not self.main_topics:
            return 0.0
        return sum(self.topic_coverage.values()) / len(self.main_topics)

    @property
    def has_topic_drift(self) -> bool:
        """Check if conversation has significant topic drift.

        Topic drift is detected when there are more than one abrupt
        topic changes in the conversation.

        Returns:
            bool: True if >1 abrupt topic changes detected.

        Examples:
            >>> analysis = TopicAnalysis(
            ...     [], [], [(1, TopicTransition.ABRUPT_CHANGE),
            ...              (2, TopicTransition.ABRUPT_CHANGE)],
            ...     {}, {}
            ... )
            >>> analysis.has_topic_drift
            True
        """
        abrupt_changes = sum(
            1 for _, t in self.topic_transitions if t == TopicTransition.ABRUPT_CHANGE
        )
        return abrupt_changes > 1

    def to_dict(self) -> dict[str, Any]:
        """Convert topic analysis to dictionary representation.

        Returns:
            dict: Dictionary containing all analysis fields with
                transitions converted to human-readable format.

        Examples:
            >>> analysis = TopicAnalysis(
            ...     ["test"], [(1, "test")],
            ...     [(1, TopicTransition.CONTINUATION)],
            ...     {"test": 1}, {"test": 1.0}
            ... )
            >>> d = analysis.to_dict()
            >>> d["n_topics"]
            1
            >>> d["topic_transitions"][0]["type"]
            'continuation'
        """
        return {
            "main_topics": self.main_topics,
            "topic_sequence": [{"turn": t, "topic": topic} for t, topic in self.topic_sequence],
            "topic_transitions": [
                {"turn": t, "type": trans.value} for t, trans in self.topic_transitions
            ],
            "topic_coverage": self.topic_coverage,
            "topic_depth": self.topic_depth,
            "n_topics": self.n_topics,
            "avg_topic_duration": self.avg_topic_duration,
            "has_topic_drift": self.has_topic_drift,
        }


@dataclass
class ConsistencyAnalysis:
    """Analysis of consistency across conversation turns.

    This dataclass contains results from checking whether the LLM maintains
    consistency throughout a conversation in terms of facts, style, and
    context. It tracks memory references and detected inconsistencies.

    Attributes:
        memory_references: List of MemoryReference objects tracking cross-turn
            references and their accuracy.
        factual_consistency_score: Score (0-1) for factual consistency across
            turns (are facts stated consistently?).
        stylistic_consistency_score: Score (0-1) for style consistency
            (response length variance, tone, etc.).
        contextual_consistency_score: Score (0-1) for contextual consistency
            (does the LLM remember conversation context?).
        inconsistencies: List of dictionaries describing detected inconsistencies
            with details about what was inconsistent.

    Properties:
        overall_consistency: Average of all three consistency scores.
        is_consistent: True if overall_consistency >= 0.7.
        n_inconsistencies: Number of detected inconsistencies.

    Examples:
        Examining consistency results:

        >>> analysis = ConsistencyAnalysis(
        ...     memory_references=[],
        ...     factual_consistency_score=0.9,
        ...     stylistic_consistency_score=0.8,
        ...     contextual_consistency_score=0.85,
        ...     inconsistencies=[]
        ... )
        >>> analysis.is_consistent
        True

        Checking overall consistency:

        >>> round(analysis.overall_consistency, 2)
        0.85

        Analyzing with inconsistencies:

        >>> analysis = ConsistencyAnalysis(
        ...     memory_references=[],
        ...     factual_consistency_score=0.5,
        ...     stylistic_consistency_score=0.6,
        ...     contextual_consistency_score=0.5,
        ...     inconsistencies=[{"type": "factual", "description": "Contradicted earlier statement"}]
        ... )
        >>> analysis.is_consistent
        False
        >>> analysis.n_inconsistencies
        1

        Converting for reporting:

        >>> d = analysis.to_dict()
        >>> d["is_consistent"]
        False
        >>> len(d["inconsistencies"])
        1
    """

    memory_references: list[MemoryReference]
    factual_consistency_score: float
    stylistic_consistency_score: float
    contextual_consistency_score: float
    inconsistencies: list[dict[str, Any]]

    @property
    def overall_consistency(self) -> float:
        """Calculate overall consistency score as average of all scores.

        Returns:
            float: Average of factual, stylistic, and contextual scores.

        Examples:
            >>> analysis = ConsistencyAnalysis([], 0.9, 0.6, 0.9, [])
            >>> analysis.overall_consistency
            0.8
        """
        return (
            self.factual_consistency_score
            + self.stylistic_consistency_score
            + self.contextual_consistency_score
        ) / 3

    @property
    def is_consistent(self) -> bool:
        """Check if conversation is considered consistent.

        A conversation is consistent if overall_consistency >= 0.7.

        Returns:
            bool: True if overall consistency score is at least 0.7.

        Examples:
            >>> analysis = ConsistencyAnalysis([], 0.8, 0.8, 0.8, [])
            >>> analysis.is_consistent
            True
            >>> analysis2 = ConsistencyAnalysis([], 0.5, 0.5, 0.5, [])
            >>> analysis2.is_consistent
            False
        """
        return self.overall_consistency >= 0.7

    @property
    def n_inconsistencies(self) -> int:
        """Get number of detected inconsistencies.

        Returns:
            int: Count of inconsistencies found in the conversation.

        Examples:
            >>> analysis = ConsistencyAnalysis([], 1.0, 1.0, 1.0,
            ...     [{"type": "factual"}, {"type": "stylistic"}])
            >>> analysis.n_inconsistencies
            2
        """
        return len(self.inconsistencies)

    def to_dict(self) -> dict[str, Any]:
        """Convert consistency analysis to dictionary representation.

        Returns:
            dict: Dictionary containing all analysis fields including
                computed properties and nested memory references.

        Examples:
            >>> analysis = ConsistencyAnalysis([], 0.9, 0.85, 0.8, [])
            >>> d = analysis.to_dict()
            >>> d["is_consistent"]
            True
            >>> "overall_consistency" in d
            True
        """
        return {
            "memory_references": [r.to_dict() for r in self.memory_references],
            "factual_consistency_score": self.factual_consistency_score,
            "stylistic_consistency_score": self.stylistic_consistency_score,
            "contextual_consistency_score": self.contextual_consistency_score,
            "overall_consistency": self.overall_consistency,
            "is_consistent": self.is_consistent,
            "n_inconsistencies": self.n_inconsistencies,
            "inconsistencies": self.inconsistencies,
        }


@dataclass
class EngagementMetrics:
    """Metrics for measuring user engagement in a conversation.

    This dataclass contains quantitative measures of how engaged the user
    appears to be throughout the conversation, including response patterns,
    question frequency, and sentiment indicators.

    Attributes:
        avg_response_length: Average word count of assistant responses.
        response_length_variance: Variance in response lengths (lower = more consistent).
        question_ratio: Proportion of user turns containing questions (0-1).
        clarification_ratio: Proportion of turns seeking clarification (0-1).
        user_satisfaction_indicators: Count of positive signals (thanks, great, etc.).
        user_frustration_indicators: Count of negative signals (wrong, stop, etc.).
        conversation_momentum: Score (0-1) for how well the conversation flows
            (based on turn quality scores).

    Properties:
        engagement_score: Overall engagement score combining satisfaction and momentum.

    Examples:
        Creating engagement metrics:

        >>> metrics = EngagementMetrics(
        ...     avg_response_length=50.0,
        ...     response_length_variance=100.0,
        ...     question_ratio=0.8,
        ...     clarification_ratio=0.1,
        ...     user_satisfaction_indicators=3,
        ...     user_frustration_indicators=0,
        ...     conversation_momentum=0.75
        ... )
        >>> metrics.question_ratio
        0.8

        Calculating engagement score:

        >>> metrics.engagement_score  # (3/3 + 0.75) / 2
        0.875

        Analyzing mixed signals:

        >>> metrics = EngagementMetrics(
        ...     avg_response_length=30.0,
        ...     response_length_variance=50.0,
        ...     question_ratio=0.5,
        ...     clarification_ratio=0.3,
        ...     user_satisfaction_indicators=1,
        ...     user_frustration_indicators=2,
        ...     conversation_momentum=0.5
        ... )
        >>> round(metrics.engagement_score, 2)  # Lower due to frustration
        0.42

        Converting for analysis:

        >>> d = metrics.to_dict()
        >>> d["clarification_ratio"]
        0.3
        >>> "engagement_score" in d
        True
    """

    avg_response_length: float
    response_length_variance: float
    question_ratio: float  # % of turns with questions
    clarification_ratio: float  # % of turns seeking clarification
    user_satisfaction_indicators: int
    user_frustration_indicators: int
    conversation_momentum: float  # 0-1, how well conversation flows

    @property
    def engagement_score(self) -> float:
        """Calculate overall engagement score.

        The score is computed as the average of:
        - Satisfaction ratio: satisfaction / (satisfaction + frustration)
        - Conversation momentum: from turn quality scores

        Returns:
            float: Engagement score between 0 and 1.

        Examples:
            >>> metrics = EngagementMetrics(50.0, 10.0, 0.5, 0.1, 5, 0, 0.8)
            >>> metrics.engagement_score  # (5/5 + 0.8) / 2
            0.9

            >>> metrics2 = EngagementMetrics(50.0, 10.0, 0.5, 0.1, 0, 0, 0.6)
            >>> metrics2.engagement_score  # (0/1 + 0.6) / 2
            0.3
        """
        satisfaction = self.user_satisfaction_indicators / max(
            1, self.user_satisfaction_indicators + self.user_frustration_indicators
        )
        return (satisfaction + self.conversation_momentum) / 2

    def to_dict(self) -> dict[str, Any]:
        """Convert engagement metrics to dictionary representation.

        Returns:
            dict: Dictionary containing all metrics including the
                computed engagement_score.

        Examples:
            >>> metrics = EngagementMetrics(40.0, 20.0, 0.6, 0.2, 2, 1, 0.7)
            >>> d = metrics.to_dict()
            >>> d["question_ratio"]
            0.6
            >>> "engagement_score" in d
            True
        """
        return {
            "avg_response_length": self.avg_response_length,
            "response_length_variance": self.response_length_variance,
            "question_ratio": self.question_ratio,
            "clarification_ratio": self.clarification_ratio,
            "user_satisfaction_indicators": self.user_satisfaction_indicators,
            "user_frustration_indicators": self.user_frustration_indicators,
            "conversation_momentum": self.conversation_momentum,
            "engagement_score": self.engagement_score,
        }


@dataclass
class ConversationReport:
    """Comprehensive conversation analysis report.

    This dataclass is the main output of ConversationAnalyzer.analyze(),
    containing all analysis results: turn-by-turn analysis, topic tracking,
    consistency checking, engagement metrics, and actionable insights.

    Attributes:
        n_turns: Total number of conversation turns analyzed.
        turns: List of analyzed ConversationTurn objects with quality scores.
        topic_analysis: TopicAnalysis object with topic tracking results.
        consistency_analysis: ConsistencyAnalysis with consistency results.
        engagement_metrics: EngagementMetrics with engagement analysis.
        conversation_state: Current ConversationState enum value.
        overall_quality_score: Weighted quality score (0-1) combining
            turn scores (50%), consistency (30%), and engagement (20%).
        strengths: List of identified conversation strengths.
        weaknesses: List of identified conversation weaknesses.
        recommendations: List of actionable improvement recommendations.

    Properties:
        quality_level: Human-readable quality classification
            (excellent/good/acceptable/poor/failed).

    Examples:
        Examining a conversation report:

        >>> # After running analyzer.analyze(conversation)
        >>> report.n_turns
        5
        >>> report.quality_level
        'good'

        Checking conversation state:

        >>> report.conversation_state
        <ConversationState.FLOWING: 'flowing'>

        Reviewing insights:

        >>> for strength in report.strengths:
        ...     print(f"+ {strength}")
        + High quality responses throughout conversation
        + Good consistency maintained across turns

        >>> for rec in report.recommendations:
        ...     print(f"- {rec}")
        - Continue current conversation approach

        Accessing nested analysis:

        >>> report.topic_analysis.n_topics
        2
        >>> report.consistency_analysis.is_consistent
        True
        >>> report.engagement_metrics.engagement_score
        0.75

        Converting for JSON export:

        >>> d = report.to_dict()
        >>> d["quality_level"]
        'good'
        >>> len(d["turns"])
        5
    """

    n_turns: int
    turns: list[ConversationTurn]
    topic_analysis: TopicAnalysis
    consistency_analysis: ConsistencyAnalysis
    engagement_metrics: EngagementMetrics
    conversation_state: ConversationState
    overall_quality_score: float
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]

    @property
    def quality_level(self) -> str:
        """Get human-readable quality level classification.

        Quality levels based on overall_quality_score:
            - excellent: >= 0.85
            - good: >= 0.70
            - acceptable: >= 0.50
            - poor: >= 0.30
            - failed: < 0.30

        Returns:
            str: Quality level as a lowercase string.

        Examples:
            >>> # With overall_quality_score = 0.90
            >>> report.quality_level
            'excellent'

            >>> # With overall_quality_score = 0.55
            >>> report.quality_level
            'acceptable'
        """
        if self.overall_quality_score >= 0.85:
            return "excellent"
        elif self.overall_quality_score >= 0.70:
            return "good"
        elif self.overall_quality_score >= 0.50:
            return "acceptable"
        elif self.overall_quality_score >= 0.30:
            return "poor"
        return "failed"

    def to_dict(self) -> dict[str, Any]:
        """Convert complete report to dictionary representation.

        Creates a fully serializable dictionary containing all analysis
        results, suitable for JSON export or further processing.

        Returns:
            dict: Nested dictionary with all report data including
                all nested dataclass conversions.

        Examples:
            >>> d = report.to_dict()
            >>> d["n_turns"]
            5
            >>> d["conversation_state"]
            'flowing'
            >>> d["topic_analysis"]["n_topics"]
            2
        """
        return {
            "n_turns": self.n_turns,
            "turns": [t.to_dict() for t in self.turns],
            "topic_analysis": self.topic_analysis.to_dict(),
            "consistency_analysis": self.consistency_analysis.to_dict(),
            "engagement_metrics": self.engagement_metrics.to_dict(),
            "conversation_state": self.conversation_state.value,
            "overall_quality_score": self.overall_quality_score,
            "quality_level": self.quality_level,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
        }


class Conversation:
    """Represents a multi-turn conversation between a user and an assistant.

    This class is the primary container for managing conversation messages.
    It supports adding messages, extracting conversation turns, and
    converting to various formats for analysis or storage.

    Attributes:
        messages (property): List of all ConversationMessage objects.
        n_messages (property): Total count of messages.
        n_turns (property): Count of user messages (conversation turns).

    Examples:
        Creating an empty conversation and adding messages:

        >>> conv = Conversation()
        >>> conv.add_user_message("What is Python?")
        ConversationMessage(role=<MessageRole.USER: 'user'>, content='What is Python?', ...)
        >>> conv.add_assistant_message("Python is a programming language.")
        ConversationMessage(role=<MessageRole.ASSISTANT: 'assistant'>, ...)
        >>> conv.n_turns
        1

        Creating a conversation from message dictionaries:

        >>> messages = [
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ...     {"role": "user", "content": "How are you?"},
        ...     {"role": "assistant", "content": "I'm doing well!"},
        ... ]
        >>> conv = Conversation(messages)
        >>> conv.n_messages
        4
        >>> conv.n_turns
        2

        Using a system prompt:

        >>> conv = Conversation(
        ...     messages=[{"role": "user", "content": "Hi"}],
        ...     system_prompt="You are a helpful assistant."
        ... )
        >>> text = conv.to_text(include_system=True)
        >>> print(text)
        System: You are a helpful assistant.
        <BLANKLINE>
        User: Hi
        <BLANKLINE>

        Extracting turns for analysis:

        >>> conv = Conversation([
        ...     {"role": "user", "content": "Question 1"},
        ...     {"role": "assistant", "content": "Answer 1"},
        ...     {"role": "user", "content": "Question 2"},
        ...     {"role": "assistant", "content": "Answer 2"},
        ... ])
        >>> turns = conv.get_turns()
        >>> len(turns)
        2
        >>> turns[0].turn_number
        1
    """

    def __init__(
        self,
        messages: Optional[list[dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize a conversation.

        Args:
            messages: Optional list of message dictionaries. Each dictionary
                should have 'role' (user/assistant/system) and 'content' keys.
                Defaults to None for an empty conversation.
            system_prompt: Optional system prompt/instruction for the conversation.
                This is stored separately from messages and can be included
                in text output via to_text(include_system=True).

        Examples:
            Empty conversation:

            >>> conv = Conversation()
            >>> conv.n_messages
            0

            With initial messages:

            >>> conv = Conversation([
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi!"}
            ... ])
            >>> conv.n_messages
            2

            With system prompt:

            >>> conv = Conversation(system_prompt="Be concise.")
            >>> conv.to_text(include_system=True)
            'System: Be concise.\\n\\n'
        """
        self._messages: list[ConversationMessage] = []
        self._system_prompt = system_prompt

        if messages:
            for msg in messages:
                self.add_message(
                    role=MessageRole(msg.get("role", "user")),
                    content=msg.get("content", ""),
                )

    def add_message(
        self,
        role: MessageRole,
        content: str,
        timestamp: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationMessage:
        """Add a message to the conversation.

        Args:
            role: The MessageRole (USER, ASSISTANT, or SYSTEM).
            content: The text content of the message.
            timestamp: Optional Unix timestamp for when the message was created.
            metadata: Optional dictionary of additional properties.

        Returns:
            ConversationMessage: The created and added message object.

        Examples:
            Adding a user message:

            >>> conv = Conversation()
            >>> msg = conv.add_message(MessageRole.USER, "Hello!")
            >>> msg.content
            'Hello!'

            Adding with timestamp:

            >>> import time
            >>> msg = conv.add_message(
            ...     MessageRole.ASSISTANT,
            ...     "Hi there!",
            ...     timestamp=time.time()
            ... )
            >>> msg.timestamp is not None
            True

            Adding with metadata:

            >>> msg = conv.add_message(
            ...     MessageRole.USER,
            ...     "Help me",
            ...     metadata={"source": "api", "session_id": "abc123"}
            ... )
            >>> msg.metadata["source"]
            'api'
        """
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=timestamp,
            metadata=metadata or {},
        )
        self._messages.append(message)
        return message

    def add_user_message(self, content: str, **kwargs) -> ConversationMessage:
        """Add a user message to the conversation.

        Convenience method that wraps add_message with MessageRole.USER.

        Args:
            content: The text content of the user message.
            **kwargs: Additional arguments passed to add_message
                (timestamp, metadata).

        Returns:
            ConversationMessage: The created user message.

        Examples:
            >>> conv = Conversation()
            >>> msg = conv.add_user_message("What is AI?")
            >>> msg.role
            <MessageRole.USER: 'user'>

            >>> msg = conv.add_user_message("Hello", metadata={"intent": "greeting"})
            >>> msg.metadata["intent"]
            'greeting'
        """
        return self.add_message(MessageRole.USER, content, **kwargs)

    def add_assistant_message(self, content: str, **kwargs) -> ConversationMessage:
        """Add an assistant message to the conversation.

        Convenience method that wraps add_message with MessageRole.ASSISTANT.

        Args:
            content: The text content of the assistant response.
            **kwargs: Additional arguments passed to add_message
                (timestamp, metadata).

        Returns:
            ConversationMessage: The created assistant message.

        Examples:
            >>> conv = Conversation()
            >>> msg = conv.add_assistant_message("I can help with that!")
            >>> msg.role
            <MessageRole.ASSISTANT: 'assistant'>

            >>> msg = conv.add_assistant_message(
            ...     "Here's your answer",
            ...     metadata={"model": "gpt-4", "tokens": 150}
            ... )
            >>> msg.metadata["model"]
            'gpt-4'
        """
        return self.add_message(MessageRole.ASSISTANT, content, **kwargs)

    @property
    def messages(self) -> list[ConversationMessage]:
        """Get a copy of all messages in the conversation.

        Returns:
            list[ConversationMessage]: Copy of the messages list to prevent
                external modification.

        Examples:
            >>> conv = Conversation([{"role": "user", "content": "Hi"}])
            >>> msgs = conv.messages
            >>> len(msgs)
            1
            >>> msgs[0].content
            'Hi'
        """
        return self._messages.copy()

    @property
    def n_messages(self) -> int:
        """Get the total number of messages.

        Returns:
            int: Count of all messages (user, assistant, and system).

        Examples:
            >>> conv = Conversation()
            >>> conv.n_messages
            0
            >>> conv.add_user_message("Hello")
            ConversationMessage(...)
            >>> conv.n_messages
            1
        """
        return len(self._messages)

    @property
    def n_turns(self) -> int:
        """Get the number of conversation turns.

        A turn is counted as one user message. This gives the number
        of user queries/requests in the conversation.

        Returns:
            int: Count of user messages.

        Examples:
            >>> conv = Conversation([
            ...     {"role": "user", "content": "Q1"},
            ...     {"role": "assistant", "content": "A1"},
            ...     {"role": "user", "content": "Q2"},
            ...     {"role": "assistant", "content": "A2"},
            ... ])
            >>> conv.n_turns
            2
        """
        user_messages = [m for m in self._messages if m.role == MessageRole.USER]
        return len(user_messages)

    def get_turns(self) -> list[ConversationTurn]:
        """Extract conversation turns as user-assistant pairs.

        Pairs consecutive user and assistant messages into ConversationTurn
        objects. Unpaired user messages (without following assistant response)
        are skipped.

        Returns:
            list[ConversationTurn]: List of turn objects, each containing
                one user message and one assistant response.

        Examples:
            Basic turn extraction:

            >>> conv = Conversation([
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi!"},
            ... ])
            >>> turns = conv.get_turns()
            >>> len(turns)
            1
            >>> turns[0].user_message.content
            'Hello'
            >>> turns[0].assistant_response.content
            'Hi!'

            Multiple turns:

            >>> conv = Conversation([
            ...     {"role": "user", "content": "Q1"},
            ...     {"role": "assistant", "content": "A1"},
            ...     {"role": "user", "content": "Q2"},
            ...     {"role": "assistant", "content": "A2"},
            ... ])
            >>> turns = conv.get_turns()
            >>> [t.turn_number for t in turns]
            [1, 2]

            Unpaired messages are skipped:

            >>> conv = Conversation([
            ...     {"role": "user", "content": "Question without answer"},
            ... ])
            >>> len(conv.get_turns())
            0
        """
        turns = []
        turn_num = 0
        i = 0

        while i < len(self._messages):
            if self._messages[i].role == MessageRole.USER:
                user_msg = self._messages[i]
                # Look for next assistant message
                if (
                    i + 1 < len(self._messages)
                    and self._messages[i + 1].role == MessageRole.ASSISTANT
                ):
                    assistant_msg = self._messages[i + 1]
                    turn_num += 1
                    turns.append(
                        ConversationTurn(
                            turn_number=turn_num,
                            user_message=user_msg,
                            assistant_response=assistant_msg,
                        )
                    )
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        return turns

    def to_text(self, include_system: bool = False) -> str:
        """Convert conversation to plain text format.

        Creates a human-readable text representation with role labels.

        Args:
            include_system: If True and a system prompt exists, include it
                at the start of the output. Defaults to False.

        Returns:
            str: Text representation with "Role: content" format,
                each message separated by blank lines.

        Examples:
            Basic conversion:

            >>> conv = Conversation([
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi there!"},
            ... ])
            >>> print(conv.to_text())
            User: Hello
            <BLANKLINE>
            Assistant: Hi there!
            <BLANKLINE>

            Including system prompt:

            >>> conv = Conversation(
            ...     messages=[{"role": "user", "content": "Hi"}],
            ...     system_prompt="Be friendly."
            ... )
            >>> print(conv.to_text(include_system=True))
            System: Be friendly.
            <BLANKLINE>
            User: Hi
            <BLANKLINE>
        """
        lines = []
        if include_system and self._system_prompt:
            lines.append(f"System: {self._system_prompt}")
            lines.append("")

        for msg in self._messages:
            role_label = msg.role.value.capitalize()
            lines.append(f"{role_label}: {msg.content}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert conversation to dictionary representation.

        Creates a serializable dictionary suitable for JSON export or storage.

        Returns:
            dict: Dictionary containing system_prompt, messages list,
                n_messages count, and n_turns count.

        Examples:
            >>> conv = Conversation(
            ...     messages=[{"role": "user", "content": "Hi"}],
            ...     system_prompt="Be helpful."
            ... )
            >>> d = conv.to_dict()
            >>> d["system_prompt"]
            'Be helpful.'
            >>> d["n_messages"]
            1
            >>> d["messages"][0]["role"]
            'user'
        """
        return {
            "system_prompt": self._system_prompt,
            "messages": [m.to_dict() for m in self._messages],
            "n_messages": self.n_messages,
            "n_turns": self.n_turns,
        }


class TurnAnalyzer:
    """Analyze individual conversation turns for quality metrics.

    This class evaluates user-assistant message pairs to assess response
    relevance and coherence. It supports custom scoring functions or uses
    built-in heuristic-based defaults.

    The analyzer produces relevance scores (how well the response addresses
    the query) and coherence scores (how well-structured and sensible the
    response is), which are combined into an overall quality rating.

    Attributes:
        _relevance_fn: Function for computing relevance scores.
        _coherence_fn: Function for computing coherence scores.

    Examples:
        Basic turn analysis:

        >>> analyzer = TurnAnalyzer()
        >>> result = analyzer.analyze_turn(
        ...     "What is Python?",
        ...     "Python is a programming language used for web development."
        ... )
        >>> result["quality"]
        <TurnQuality.GOOD: 'good'>
        >>> 0 <= result["relevance_score"] <= 1
        True

        Analyzing a poor response:

        >>> result = analyzer.analyze_turn(
        ...     "What is the capital of France?",
        ...     "I like pizza."
        ... )
        >>> result["quality"] in [TurnQuality.POOR, TurnQuality.ACCEPTABLE]
        True

        Using custom scoring functions:

        >>> def custom_relevance(query, response):
        ...     # Custom logic here
        ...     return 0.8 if "python" in response.lower() else 0.3
        >>> analyzer = TurnAnalyzer(relevance_fn=custom_relevance)
        >>> result = analyzer.analyze_turn("Tell me about Python", "Python is great!")
        >>> result["relevance_score"]
        0.8

        Analyzing with context:

        >>> result = analyzer.analyze_turn(
        ...     "What else can you tell me?",
        ...     "It also supports object-oriented programming.",
        ...     context="We were discussing Python features."
        ... )
        >>> "relevance_score" in result
        True
    """

    def __init__(
        self,
        relevance_fn: Optional[Callable[[str, str], float]] = None,
        coherence_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """Initialize the turn analyzer.

        Args:
            relevance_fn: Optional custom function to compute relevance score.
                Signature: (query: str, response: str) -> float (0-1).
                If None, uses keyword overlap heuristic.
            coherence_fn: Optional custom function to compute coherence score.
                Signature: (query: str, response: str) -> float (0-1).
                If None, uses question/answer alignment heuristic.

        Examples:
            Default analyzer:

            >>> analyzer = TurnAnalyzer()

            With custom relevance function:

            >>> def my_relevance(q, r):
            ...     return 1.0 if len(r) > 50 else 0.5
            >>> analyzer = TurnAnalyzer(relevance_fn=my_relevance)

            With both custom functions:

            >>> def my_coherence(q, r):
            ...     return 0.9 if r.endswith(".") else 0.6
            >>> analyzer = TurnAnalyzer(
            ...     relevance_fn=my_relevance,
            ...     coherence_fn=my_coherence
            ... )
        """
        self._relevance_fn = relevance_fn or self._default_relevance
        self._coherence_fn = coherence_fn or self._default_coherence

    @staticmethod
    def _default_relevance(query: str, response: str) -> float:
        """Default relevance scoring using keyword overlap.

        Computes relevance as the proportion of query keywords (excluding
        stopwords) that appear in the response.

        Args:
            query: The user's query/message.
            response: The assistant's response.

        Returns:
            float: Relevance score between 0 and 1.

        Examples:
            >>> TurnAnalyzer._default_relevance(
            ...     "What is Python programming?",
            ...     "Python is a programming language."
            ... )  # High overlap
            1.0

            >>> TurnAnalyzer._default_relevance(
            ...     "Tell me about cats",
            ...     "Dogs are great pets."
            ... )  # Low overlap
            0.0
        """
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or"}
        query_words -= stopwords
        response_words -= stopwords

        if not query_words:
            return 0.5

        overlap = len(query_words & response_words)
        return min(1.0, overlap / len(query_words))

    @staticmethod
    def _default_coherence(query: str, response: str) -> float:
        """Default coherence scoring based on structure heuristics.

        Evaluates coherence based on question/answer alignment and
        response structure (length, completeness).

        Args:
            query: The user's query/message.
            response: The assistant's response.

        Returns:
            float: Coherence score between 0 and 1.

        Examples:
            >>> TurnAnalyzer._default_coherence(
            ...     "What is the meaning of life?",
            ...     "The meaning of life is a philosophical question with many answers."
            ... )  # Question with substantive answer
            0.9

            >>> TurnAnalyzer._default_coherence(
            ...     "What time is it?",
            ...     "Yes"
            ... )  # Question with inadequate answer
            0.3
        """
        # Check for question/answer alignment
        has_question = "?" in query
        has_answer = len(response) > 10

        if has_question and has_answer:
            coherence = 0.7
        elif not has_question:
            coherence = 0.5
        else:
            coherence = 0.3

        # Check response structure
        if response.strip() and len(response.split()) >= 3:
            coherence += 0.2

        return min(1.0, coherence)

    def analyze_turn(
        self,
        user_message: str,
        assistant_response: str,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze a single conversation turn for quality.

        Computes relevance and coherence scores, then determines an
        overall quality rating based on their average.

        Args:
            user_message: The user's message/query.
            assistant_response: The assistant's response.
            context: Optional string containing previous conversation
                context. Currently reserved for future use with
                context-aware scoring.

        Returns:
            dict: Analysis results containing:
                - relevance_score (float): How relevant the response is (0-1)
                - coherence_score (float): How coherent the response is (0-1)
                - quality (TurnQuality): Overall quality rating enum
                - avg_score (float): Average of relevance and coherence

        Examples:
            Analyzing an excellent response:

            >>> analyzer = TurnAnalyzer()
            >>> result = analyzer.analyze_turn(
            ...     "Explain machine learning",
            ...     "Machine learning is a branch of AI that enables computers to learn from data."
            ... )
            >>> result["quality"] in [TurnQuality.GOOD, TurnQuality.EXCELLENT]
            True

            Analyzing a poor response:

            >>> result = analyzer.analyze_turn(
            ...     "What is 2+2?",
            ...     ""
            ... )
            >>> result["quality"]
            <TurnQuality.FAILED: 'failed'>

            Using context (future feature):

            >>> result = analyzer.analyze_turn(
            ...     "What about the other one?",
            ...     "The other approach uses neural networks.",
            ...     context="We discussed two ML approaches."
            ... )
            >>> "relevance_score" in result
            True
        """
        relevance = self._relevance_fn(user_message, assistant_response)
        coherence = self._coherence_fn(user_message, assistant_response)

        # Determine quality rating
        avg_score = (relevance + coherence) / 2
        if avg_score >= 0.8:
            quality = TurnQuality.EXCELLENT
        elif avg_score >= 0.6:
            quality = TurnQuality.GOOD
        elif avg_score >= 0.4:
            quality = TurnQuality.ACCEPTABLE
        elif avg_score >= 0.2:
            quality = TurnQuality.POOR
        else:
            quality = TurnQuality.FAILED

        return {
            "relevance_score": relevance,
            "coherence_score": coherence,
            "quality": quality,
            "avg_score": avg_score,
        }


class TopicTracker:
    """Track topics and their transitions across conversation turns.

    This class monitors the topics discussed throughout a conversation,
    detecting when topics change and how they transition (naturally,
    abruptly, or returning to previous topics).

    The tracker maintains a history of topics and can generate a
    comprehensive TopicAnalysis report with coverage and depth metrics.

    Attributes:
        _topic_extractor: Function for extracting topics from text.
        _topic_sequence: List of (turn_number, topic) pairs.
        _topic_history: Ordered list of topics as they appeared.

    Examples:
        Basic topic tracking:

        >>> tracker = TopicTracker()
        >>> tracker.add_turn(1, "What is Python?", "Python is a programming language.")
        'python'
        >>> tracker.add_turn(2, "Tell me about its libraries", "Python has numpy, pandas...")
        'libraries'
        >>> analysis = tracker.analyze()
        >>> analysis.n_topics
        2

        Detecting topic transitions:

        >>> tracker = TopicTracker()
        >>> tracker.add_turn(1, "Discuss Python", "Python is great for data science.")
        'python'
        >>> tracker.add_turn(2, "More about Python", "Python also supports web development.")
        'python'
        >>> analysis = tracker.analyze()
        >>> analysis.topic_transitions[0][1]
        <TopicTransition.CONTINUATION: 'continuation'>

        Using custom topic extractor:

        >>> def my_extractor(text):
        ...     if "python" in text.lower():
        ...         return ["programming"]
        ...     return ["general"]
        >>> tracker = TopicTracker(topic_extractor=my_extractor)
        >>> tracker.add_turn(1, "Python question", "Python answer")
        'programming'

        Clearing and reusing tracker:

        >>> tracker = TopicTracker()
        >>> tracker.add_turn(1, "Topic A", "Response A")
        'topic'
        >>> tracker.clear()
        >>> len(tracker._topic_history)
        0
    """

    def __init__(
        self,
        topic_extractor: Optional[Callable[[str], list[str]]] = None,
    ):
        """Initialize the topic tracker.

        Args:
            topic_extractor: Optional custom function to extract topics from text.
                Signature: (text: str) -> list[str].
                Should return a list of topic keywords, ordered by importance.
                If None, uses a simple keyword extraction heuristic.

        Examples:
            Default tracker:

            >>> tracker = TopicTracker()

            With custom extractor:

            >>> def extract_topics(text):
            ...     keywords = []
            ...     if "machine learning" in text.lower():
            ...         keywords.append("ml")
            ...     if "python" in text.lower():
            ...         keywords.append("python")
            ...     return keywords or ["general"]
            >>> tracker = TopicTracker(topic_extractor=extract_topics)
        """
        self._topic_extractor = topic_extractor or self._default_topic_extractor
        self._topic_sequence: list[tuple[int, str]] = []
        self._topic_history: list[str] = []

    @staticmethod
    def _default_topic_extractor(text: str) -> list[str]:
        """Simple keyword-based topic extraction.

        Extracts potential topic keywords by filtering out stopwords
        and short words, then ranking by frequency.

        Args:
            text: The text to extract topics from.

        Returns:
            list[str]: Up to 3 topic keywords, ordered by frequency.

        Examples:
            >>> TopicTracker._default_topic_extractor("Python programming language")
            ['python', 'programming', 'language']

            >>> TopicTracker._default_topic_extractor("The quick brown fox")
            ['quick', 'brown', 'fox']
        """
        # Extract potential topics (nouns and important words)
        words = text.lower().split()
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "and",
            "or",
            "it",
            "this",
            "that",
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "can",
            "could",
            "would",
            "should",
            "will",
            "do",
            "does",
            "did",
            "have",
            "has",
            "had",
            "be",
            "been",
            "being",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "me",
            "him",
            "us",
            "them",
            "please",
            "thanks",
            "thank",
        }

        topics = []
        for word in words:
            # Clean word
            word = re.sub(r"[^\w]", "", word)
            if word and word not in stopwords and len(word) > 2:
                topics.append(word)

        # Return most frequent/important topics
        topic_counts = defaultdict(int)
        for t in topics:
            topic_counts[t] += 1

        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_topics[:3]]

    def add_turn(self, turn_number: int, user_message: str, response: str) -> str:
        """Add a conversation turn and extract its topic.

        Combines the user message and response, extracts topics using
        the configured extractor, and records the primary topic.

        Args:
            turn_number: The sequential turn number (1-indexed).
            user_message: The user's message text.
            response: The assistant's response text.

        Returns:
            str: The detected primary topic for this turn.

        Examples:
            >>> tracker = TopicTracker()
            >>> tracker.add_turn(1, "What is Python?", "Python is a language.")
            'python'

            >>> tracker.add_turn(2, "And JavaScript?", "JavaScript is for web.")
            'javascript'

            >>> len(tracker._topic_history)
            2
        """
        combined_text = f"{user_message} {response}"
        topics = self._topic_extractor(combined_text)
        main_topic = topics[0] if topics else "general"

        self._topic_sequence.append((turn_number, main_topic))
        self._topic_history.append(main_topic)

        return main_topic

    def get_transition_type(self, prev_topic: str, curr_topic: str) -> TopicTransition:
        """Determine the type of transition between two topics.

        Classifies topic transitions as continuation (same topic),
        return to previous (revisiting earlier topic), or natural shift
        (moving to related/new topic).

        Args:
            prev_topic: The topic from the previous turn.
            curr_topic: The topic from the current turn.

        Returns:
            TopicTransition: The type of transition that occurred.

        Examples:
            Same topic (continuation):

            >>> tracker = TopicTracker()
            >>> tracker.get_transition_type("python", "python")
            <TopicTransition.CONTINUATION: 'continuation'>

            Different topics (natural shift):

            >>> tracker.get_transition_type("python", "javascript")
            <TopicTransition.NATURAL_SHIFT: 'natural_shift'>

            Returning to earlier topic:

            >>> tracker._topic_history = ["python", "javascript"]
            >>> tracker.get_transition_type("javascript", "python")
            <TopicTransition.RETURN_TO_PREVIOUS: 'return_to_previous'>
        """
        if prev_topic == curr_topic:
            return TopicTransition.CONTINUATION

        # Check if returning to earlier topic
        if curr_topic in self._topic_history[:-1]:
            return TopicTransition.RETURN_TO_PREVIOUS

        # Simple heuristic for transition type
        # In practice, this would use more sophisticated NLP
        if prev_topic.split()[0] == curr_topic.split()[0] if " " in prev_topic else False:
            return TopicTransition.NATURAL_SHIFT

        return TopicTransition.NATURAL_SHIFT  # Default to natural shift

    def analyze(self) -> TopicAnalysis:
        """Generate a comprehensive topic analysis report.

        Compiles all tracked topic data into a TopicAnalysis object
        including topic coverage, transitions, and depth metrics.

        Returns:
            TopicAnalysis: Complete analysis with topics, sequences,
                transitions, coverage, and depth information.

        Examples:
            >>> tracker = TopicTracker()
            >>> tracker.add_turn(1, "Python basics", "Python intro")
            'python'
            >>> tracker.add_turn(2, "Python advanced", "More Python")
            'python'
            >>> analysis = tracker.analyze()
            >>> analysis.n_topics
            1
            >>> analysis.topic_coverage["python"]
            2

            >>> tracker.clear()
            >>> tracker.add_turn(1, "Topic A", "Response A")
            'topic'
            >>> tracker.add_turn(2, "Topic B", "Response B")
            'topic'
            >>> analysis = tracker.analyze()
            >>> len(analysis.topic_transitions)
            1
        """
        main_topics = list(dict.fromkeys(self._topic_history))

        # Calculate topic coverage
        topic_coverage = defaultdict(int)
        for topic in self._topic_history:
            topic_coverage[topic] += 1

        # Calculate transitions
        transitions = []
        for i in range(1, len(self._topic_sequence)):
            prev_topic = self._topic_sequence[i - 1][1]
            curr_topic = self._topic_sequence[i][1]
            turn_num = self._topic_sequence[i][0]
            transition = self.get_transition_type(prev_topic, curr_topic)
            transitions.append((turn_num, transition))

        # Calculate topic depth (simple heuristic based on coverage)
        topic_depth = {
            topic: count / max(len(self._topic_history), 1)
            for topic, count in topic_coverage.items()
        }

        return TopicAnalysis(
            main_topics=main_topics,
            topic_sequence=self._topic_sequence.copy(),
            topic_transitions=transitions,
            topic_coverage=dict(topic_coverage),
            topic_depth=topic_depth,
        )

    def clear(self) -> None:
        """Clear all tracking state for reuse.

        Resets the topic sequence and history to empty lists,
        allowing the tracker to be reused for a new conversation.

        Examples:
            >>> tracker = TopicTracker()
            >>> tracker.add_turn(1, "Topic", "Response")
            'topic'
            >>> tracker.clear()
            >>> len(tracker._topic_history)
            0
            >>> len(tracker._topic_sequence)
            0
        """
        self._topic_sequence.clear()
        self._topic_history.clear()


class ConversationConsistencyChecker:
    """Check consistency across conversation turns.

    This class analyzes whether an LLM maintains factual, stylistic,
    and contextual consistency throughout a conversation. It tracks
    responses and identifies cross-turn references and inconsistencies.

    Attributes:
        _similarity_fn: Function for computing text similarity.
        _facts: Dictionary mapping turn numbers to extracted facts.

    Examples:
        Basic consistency checking:

        >>> checker = ConversationConsistencyChecker()
        >>> checker.add_turn(1, "Python was created by Guido van Rossum.")
        >>> checker.add_turn(2, "As mentioned, Guido created Python.")
        >>> # Create mock turns for checking
        >>> from insideLLMs.contrib.conversation import ConversationMessage, ConversationTurn
        >>> turns = [
        ...     ConversationTurn(1,
        ...         ConversationMessage(MessageRole.USER, "Who created Python?"),
        ...         ConversationMessage(MessageRole.ASSISTANT, "Python was created by Guido.")
        ...     ),
        ...     ConversationTurn(2,
        ...         ConversationMessage(MessageRole.USER, "Tell me more"),
        ...         ConversationMessage(MessageRole.ASSISTANT, "Guido created Python in 1991.")
        ...     )
        ... ]
        >>> analysis = checker.check(turns)
        >>> analysis.is_consistent
        True

        Using custom similarity function:

        >>> def jaccard_sim(t1, t2):
        ...     w1, w2 = set(t1.split()), set(t2.split())
        ...     return len(w1 & w2) / len(w1 | w2) if w1 | w2 else 0
        >>> checker = ConversationConsistencyChecker(similarity_fn=jaccard_sim)

        Clearing for reuse:

        >>> checker.clear()
        >>> len(checker._facts)
        0
    """

    def __init__(
        self,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """Initialize the consistency checker.

        Args:
            similarity_fn: Optional custom function to compute text similarity.
                Signature: (text1: str, text2: str) -> float (0-1).
                If None, uses Dice coefficient based on word overlap.

        Examples:
            Default checker:

            >>> checker = ConversationConsistencyChecker()

            With custom similarity:

            >>> def cosine_sim(t1, t2):
            ...     # Custom implementation
            ...     return 0.8
            >>> checker = ConversationConsistencyChecker(similarity_fn=cosine_sim)
        """
        self._similarity_fn = similarity_fn or self._default_similarity
        self._facts: dict[int, list[str]] = {}  # turn -> facts

    @staticmethod
    def _default_similarity(text1: str, text2: str) -> float:
        """Compute text similarity using Dice coefficient.

        Uses word overlap to compute similarity as:
        2 * |intersection| / (|set1| + |set2|)

        Args:
            text1: First text to compare.
            text2: Second text to compare.

        Returns:
            float: Similarity score between 0 and 1.

        Examples:
            >>> ConversationConsistencyChecker._default_similarity(
            ...     "Python is great",
            ...     "Python is awesome"
            ... )  # 2 common words out of 5 unique
            0.5

            >>> ConversationConsistencyChecker._default_similarity(
            ...     "hello world",
            ...     "hello world"
            ... )  # Identical
            1.0
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        overlap = len(words1 & words2)
        return 2 * overlap / (len(words1) + len(words2))

    def add_turn(self, turn_number: int, response: str) -> None:
        """Add a turn's response for consistency tracking.

        Extracts declarative statements (potential facts) from the
        response and stores them for later consistency analysis.

        Args:
            turn_number: The turn number (1-indexed).
            response: The assistant's response text.

        Examples:
            >>> checker = ConversationConsistencyChecker()
            >>> checker.add_turn(1, "Python is a programming language. It was created in 1991.")
            >>> 1 in checker._facts
            True
            >>> len(checker._facts[1])  # Two sentences extracted
            2
        """
        # Extract potential facts (sentences with declarative statements)
        sentences = re.split(r"[.!?]", response)
        facts = [s.strip() for s in sentences if len(s.strip()) > 10]
        self._facts[turn_number] = facts

    def check(self, turns: list[ConversationTurn]) -> ConsistencyAnalysis:
        """Analyze consistency across all conversation turns.

        Compares responses across turns to detect memory references,
        compute consistency scores, and identify any inconsistencies.

        Args:
            turns: List of ConversationTurn objects to analyze.

        Returns:
            ConsistencyAnalysis: Analysis containing memory references,
                consistency scores, and any detected inconsistencies.

        Examples:
            >>> checker = ConversationConsistencyChecker()
            >>> # Analysis with consistent turns
            >>> user1 = ConversationMessage(MessageRole.USER, "Q1")
            >>> asst1 = ConversationMessage(MessageRole.ASSISTANT, "Python is great.")
            >>> user2 = ConversationMessage(MessageRole.USER, "Q2")
            >>> asst2 = ConversationMessage(MessageRole.ASSISTANT, "Python is indeed great.")
            >>> turns = [
            ...     ConversationTurn(1, user1, asst1),
            ...     ConversationTurn(2, user2, asst2)
            ... ]
            >>> analysis = checker.check(turns)
            >>> analysis.is_consistent
            True

            >>> # Empty turns
            >>> analysis = checker.check([])
            >>> analysis.overall_consistency
            1.0
        """
        memory_references = []
        inconsistencies = []

        # Compare responses for consistency
        for i, turn in enumerate(turns):
            if i == 0:
                continue

            # Look for references to earlier content
            current_response = turn.assistant_response.content.lower()
            for j in range(i):
                prev_response = turns[j].assistant_response.content.lower()
                similarity = self._similarity_fn(current_response, prev_response)

                if similarity > 0.3:  # Some overlap detected
                    memory_references.append(
                        MemoryReference(
                            source_turn=j + 1,
                            target_turn=i + 1,
                            reference_type="content_overlap",
                            content=f"Similarity with turn {j + 1}",
                            is_accurate=True,
                            confidence=similarity,
                        )
                    )

        # Calculate consistency scores
        if turns:
            # Factual consistency (based on memory reference accuracy)
            accurate_refs = sum(1 for r in memory_references if r.is_accurate)
            factual_score = accurate_refs / len(memory_references) if memory_references else 1.0

            # Stylistic consistency (based on response length variance)
            lengths = [t.assistant_response.word_count for t in turns]
            if len(lengths) > 1:
                mean_len = sum(lengths) / len(lengths)
                variance = sum((length - mean_len) ** 2 for length in lengths) / len(lengths)
                # Lower variance = higher consistency
                stylistic_score = max(0, 1 - (variance / max(mean_len**2, 1)))
            else:
                stylistic_score = 1.0

            # Contextual consistency
            contextual_score = 0.8 if not inconsistencies else 0.5
        else:
            factual_score = 1.0
            stylistic_score = 1.0
            contextual_score = 1.0

        return ConsistencyAnalysis(
            memory_references=memory_references,
            factual_consistency_score=factual_score,
            stylistic_consistency_score=stylistic_score,
            contextual_consistency_score=contextual_score,
            inconsistencies=inconsistencies,
        )

    def clear(self) -> None:
        """Clear all tracked facts for reuse.

        Resets the internal facts dictionary, allowing the checker
        to be reused for analyzing a new conversation.

        Examples:
            >>> checker = ConversationConsistencyChecker()
            >>> checker.add_turn(1, "Some facts here.")
            >>> checker.clear()
            >>> len(checker._facts)
            0
        """
        self._facts.clear()


class EngagementAnalyzer:
    """Analyze user engagement patterns in conversations.

    This class evaluates how engaged users appear to be throughout
    a conversation by analyzing response patterns, question frequency,
    and detecting satisfaction/frustration signals.

    Attributes:
        _satisfaction_patterns: Regex patterns indicating user satisfaction.
        _frustration_patterns: Regex patterns indicating user frustration.

    Examples:
        Basic engagement analysis:

        >>> analyzer = EngagementAnalyzer()
        >>> user1 = ConversationMessage(MessageRole.USER, "What is Python?")
        >>> asst1 = ConversationMessage(MessageRole.ASSISTANT, "Python is a language.")
        >>> user2 = ConversationMessage(MessageRole.USER, "Thanks, that's helpful!")
        >>> asst2 = ConversationMessage(MessageRole.ASSISTANT, "You're welcome!")
        >>> turns = [
        ...     ConversationTurn(1, user1, asst1),
        ...     ConversationTurn(2, user2, asst2)
        ... ]
        >>> metrics = analyzer.analyze(turns)
        >>> metrics.user_satisfaction_indicators
        1

        Detecting frustration:

        >>> user1 = ConversationMessage(MessageRole.USER, "Help me")
        >>> asst1 = ConversationMessage(MessageRole.ASSISTANT, "Sure")
        >>> user2 = ConversationMessage(MessageRole.USER, "That's wrong, not helpful")
        >>> asst2 = ConversationMessage(MessageRole.ASSISTANT, "I apologize")
        >>> turns = [
        ...     ConversationTurn(1, user1, asst1),
        ...     ConversationTurn(2, user2, asst2)
        ... ]
        >>> metrics = analyzer.analyze(turns)
        >>> metrics.user_frustration_indicators > 0
        True

        Empty conversation:

        >>> metrics = analyzer.analyze([])
        >>> metrics.engagement_score
        0
    """

    def __init__(self):
        """Initialize the engagement analyzer with default patterns.

        Sets up regex patterns for detecting satisfaction signals
        (thanks, great, helpful, etc.) and frustration signals
        (wrong, not helpful, stop, etc.).

        Examples:
            >>> analyzer = EngagementAnalyzer()
            >>> len(analyzer._satisfaction_patterns) > 0
            True
        """
        self._satisfaction_patterns = [
            r"\bthanks?\b",
            r"\bthank you\b",
            r"\bgreat\b",
            r"\bperfect\b",
            r"\bexcellent\b",
            r"\bhelpful\b",
            r"\bawesome\b",
            r"\bgood\b",
        ]
        self._frustration_patterns = [
            r"\bnot helpful\b",
            r"\bwrong\b",
            r"\bincorrect\b",
            r"\bno\b",
            r"\bstop\b",
            r"\bi said\b",
            r"\bthat\'s not\b",
            r"\byou\'re not\b",
        ]

    def analyze(self, turns: list[ConversationTurn]) -> EngagementMetrics:
        """Analyze engagement patterns across all conversation turns.

        Computes comprehensive engagement metrics including response
        length statistics, question/clarification ratios, satisfaction
        and frustration indicators, and conversation momentum.

        Args:
            turns: List of ConversationTurn objects to analyze.

        Returns:
            EngagementMetrics: Complete engagement analysis with all
                computed metrics and the overall engagement score.

        Examples:
            Analyzing engaged conversation:

            >>> analyzer = EngagementAnalyzer()
            >>> user = ConversationMessage(MessageRole.USER, "What is Python?")
            >>> asst = ConversationMessage(MessageRole.ASSISTANT, "Python is great!")
            >>> turns = [ConversationTurn(1, user, asst)]
            >>> metrics = analyzer.analyze(turns)
            >>> metrics.question_ratio
            1.0

            Empty conversation returns zeros:

            >>> metrics = analyzer.analyze([])
            >>> metrics.avg_response_length
            0
            >>> metrics.engagement_score
            0

            Multiple turns with satisfaction:

            >>> user1 = ConversationMessage(MessageRole.USER, "Help me")
            >>> asst1 = ConversationMessage(MessageRole.ASSISTANT, "Sure, here's help.")
            >>> user2 = ConversationMessage(MessageRole.USER, "Thanks!")
            >>> asst2 = ConversationMessage(MessageRole.ASSISTANT, "Welcome!")
            >>> turns = [
            ...     ConversationTurn(1, user1, asst1),
            ...     ConversationTurn(2, user2, asst2)
            ... ]
            >>> metrics = analyzer.analyze(turns)
            >>> metrics.user_satisfaction_indicators
            1
        """
        if not turns:
            return EngagementMetrics(
                avg_response_length=0,
                response_length_variance=0,
                question_ratio=0,
                clarification_ratio=0,
                user_satisfaction_indicators=0,
                user_frustration_indicators=0,
                conversation_momentum=0,
            )

        # Response length analysis
        response_lengths = [t.assistant_response.word_count for t in turns]
        avg_length = sum(response_lengths) / len(response_lengths)
        if len(response_lengths) > 1:
            variance = sum((length - avg_length) ** 2 for length in response_lengths) / len(
                response_lengths
            )
        else:
            variance = 0

        # Question analysis
        questions = sum(1 for t in turns if "?" in t.user_message.content)
        question_ratio = questions / len(turns)

        # Clarification analysis
        clarification_patterns = [
            r"\bwhat do you mean\b",
            r"\bcan you explain\b",
            r"\bi don\'t understand\b",
        ]
        clarifications = 0
        for turn in turns:
            for pattern in clarification_patterns:
                if re.search(pattern, turn.user_message.content.lower()):
                    clarifications += 1
                    break
        clarification_ratio = clarifications / len(turns)

        # Satisfaction/frustration indicators
        satisfaction = 0
        frustration = 0
        for turn in turns:
            user_text = turn.user_message.content.lower()
            for pattern in self._satisfaction_patterns:
                if re.search(pattern, user_text):
                    satisfaction += 1
                    break
            for pattern in self._frustration_patterns:
                if re.search(pattern, user_text):
                    frustration += 1
                    break

        # Conversation momentum (based on turn quality scores)
        quality_scores = [t.turn_score for t in turns]
        momentum = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        return EngagementMetrics(
            avg_response_length=avg_length,
            response_length_variance=variance,
            question_ratio=question_ratio,
            clarification_ratio=clarification_ratio,
            user_satisfaction_indicators=satisfaction,
            user_frustration_indicators=frustration,
            conversation_momentum=momentum,
        )


class ConversationAnalyzer:
    """Comprehensive multi-turn conversation analyzer.

    This is the main analyzer class that orchestrates turn-level analysis,
    topic tracking, consistency checking, and engagement analysis to produce
    a complete ConversationReport.

    The analyzer combines multiple sub-analyzers:
    - TurnAnalyzer: Evaluates individual turn quality
    - TopicTracker: Tracks topics and transitions
    - ConversationConsistencyChecker: Checks for inconsistencies
    - EngagementAnalyzer: Measures user engagement

    Attributes:
        _turn_analyzer: Analyzer for individual turns.
        _topic_tracker: Tracker for topic progression.
        _consistency_checker: Checker for cross-turn consistency.
        _engagement_analyzer: Analyzer for engagement patterns.

    Examples:
        Basic conversation analysis:

        >>> analyzer = ConversationAnalyzer()
        >>> conv = Conversation([
        ...     {"role": "user", "content": "What is Python?"},
        ...     {"role": "assistant", "content": "Python is a programming language."},
        ...     {"role": "user", "content": "Thanks!"},
        ...     {"role": "assistant", "content": "You're welcome!"},
        ... ])
        >>> report = analyzer.analyze(conv)
        >>> report.n_turns
        2
        >>> report.quality_level in ["excellent", "good", "acceptable"]
        True

        Using custom sub-analyzers:

        >>> custom_turn = TurnAnalyzer()
        >>> custom_topic = TopicTracker()
        >>> analyzer = ConversationAnalyzer(
        ...     turn_analyzer=custom_turn,
        ...     topic_tracker=custom_topic
        ... )

        Analyzing empty conversation:

        >>> conv = Conversation()
        >>> report = analyzer.analyze(conv)
        >>> report.n_turns
        0
        >>> report.conversation_state
        <ConversationState.STARTING: 'starting'>

        Accessing full report details:

        >>> conv = Conversation([
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ... ])
        >>> report = analyzer.analyze(conv)
        >>> len(report.strengths) > 0 or len(report.recommendations) > 0
        True
    """

    def __init__(
        self,
        turn_analyzer: Optional[TurnAnalyzer] = None,
        topic_tracker: Optional[TopicTracker] = None,
        consistency_checker: Optional[ConversationConsistencyChecker] = None,
        engagement_analyzer: Optional[EngagementAnalyzer] = None,
    ):
        """Initialize the conversation analyzer with optional custom components.

        Args:
            turn_analyzer: Optional custom TurnAnalyzer instance. If None,
                creates a default TurnAnalyzer.
            topic_tracker: Optional custom TopicTracker instance. If None,
                creates a default TopicTracker.
            consistency_checker: Optional custom ConversationConsistencyChecker.
                If None, creates a default checker.
            engagement_analyzer: Optional custom EngagementAnalyzer. If None,
                creates a default analyzer.

        Examples:
            Default analyzer:

            >>> analyzer = ConversationAnalyzer()

            With custom turn analyzer:

            >>> def custom_relevance(q, r):
            ...     return 0.9  # Always high relevance
            >>> custom_turn = TurnAnalyzer(relevance_fn=custom_relevance)
            >>> analyzer = ConversationAnalyzer(turn_analyzer=custom_turn)

            Fully customized:

            >>> analyzer = ConversationAnalyzer(
            ...     turn_analyzer=TurnAnalyzer(),
            ...     topic_tracker=TopicTracker(),
            ...     consistency_checker=ConversationConsistencyChecker(),
            ...     engagement_analyzer=EngagementAnalyzer()
            ... )
        """
        self._turn_analyzer = turn_analyzer or TurnAnalyzer()
        self._topic_tracker = topic_tracker or TopicTracker()
        self._consistency_checker = consistency_checker or ConversationConsistencyChecker()
        self._engagement_analyzer = engagement_analyzer or EngagementAnalyzer()

    def analyze(
        self,
        conversation: Conversation,
    ) -> ConversationReport:
        """Analyze a complete conversation and generate a comprehensive report.

        Processes all turns in the conversation, computing quality scores,
        tracking topics, checking consistency, and measuring engagement.
        Produces a ConversationReport with insights and recommendations.

        The overall quality score is computed as:
        - 50% from average turn scores
        - 30% from consistency analysis
        - 20% from engagement metrics

        Args:
            conversation: The Conversation object to analyze.

        Returns:
            ConversationReport: Complete analysis report including turn
                details, topic analysis, consistency analysis, engagement
                metrics, state assessment, and recommendations.

        Examples:
            Standard analysis:

            >>> analyzer = ConversationAnalyzer()
            >>> conv = Conversation([
            ...     {"role": "user", "content": "Hello!"},
            ...     {"role": "assistant", "content": "Hi, how can I help?"},
            ... ])
            >>> report = analyzer.analyze(conv)
            >>> report.n_turns
            1
            >>> 0 <= report.overall_quality_score <= 1
            True

            Accessing analysis components:

            >>> report.topic_analysis.n_topics >= 0
            True
            >>> report.consistency_analysis.is_consistent
            True
            >>> report.engagement_metrics.question_ratio >= 0
            True

            Empty conversation handling:

            >>> report = analyzer.analyze(Conversation())
            >>> report.n_turns
            0
            >>> "Add messages" in report.recommendations[0]
            True
        """
        turns = conversation.get_turns()

        if not turns:
            return self._empty_report()

        # Analyze each turn
        analyzed_turns = []
        context = ""
        for turn in turns:
            analysis = self._turn_analyzer.analyze_turn(
                turn.user_message.content,
                turn.assistant_response.content,
                context,
            )
            turn.relevance_score = analysis["relevance_score"]
            turn.coherence_score = analysis["coherence_score"]
            turn.quality = analysis["quality"]

            # Track topic
            topic = self._topic_tracker.add_turn(
                turn.turn_number,
                turn.user_message.content,
                turn.assistant_response.content,
            )
            turn.topic = topic

            # Update context
            context += (
                f"\nUser: {turn.user_message.content}\nAssistant: {turn.assistant_response.content}"
            )

            # Track for consistency
            self._consistency_checker.add_turn(
                turn.turn_number,
                turn.assistant_response.content,
            )

            analyzed_turns.append(turn)

        # Generate analyses
        topic_analysis = self._topic_tracker.analyze()
        consistency_analysis = self._consistency_checker.check(analyzed_turns)
        engagement_metrics = self._engagement_analyzer.analyze(analyzed_turns)

        # Determine conversation state
        state = self._determine_state(analyzed_turns, topic_analysis)

        # Calculate overall quality
        turn_scores = [t.turn_score for t in analyzed_turns]
        base_quality = sum(turn_scores) / len(turn_scores) if turn_scores else 0.5
        consistency_factor = consistency_analysis.overall_consistency
        engagement_factor = engagement_metrics.engagement_score

        overall_quality = base_quality * 0.5 + consistency_factor * 0.3 + engagement_factor * 0.2

        # Generate insights
        strengths, weaknesses, recommendations = self._generate_insights(
            analyzed_turns, topic_analysis, consistency_analysis, engagement_metrics
        )

        # Clear trackers for next analysis
        self._topic_tracker.clear()
        self._consistency_checker.clear()

        return ConversationReport(
            n_turns=len(analyzed_turns),
            turns=analyzed_turns,
            topic_analysis=topic_analysis,
            consistency_analysis=consistency_analysis,
            engagement_metrics=engagement_metrics,
            conversation_state=state,
            overall_quality_score=overall_quality,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    def _empty_report(self) -> ConversationReport:
        """Generate empty report for empty conversation."""
        return ConversationReport(
            n_turns=0,
            turns=[],
            topic_analysis=TopicAnalysis(
                main_topics=[],
                topic_sequence=[],
                topic_transitions=[],
                topic_coverage={},
                topic_depth={},
            ),
            consistency_analysis=ConsistencyAnalysis(
                memory_references=[],
                factual_consistency_score=1.0,
                stylistic_consistency_score=1.0,
                contextual_consistency_score=1.0,
                inconsistencies=[],
            ),
            engagement_metrics=EngagementMetrics(
                avg_response_length=0,
                response_length_variance=0,
                question_ratio=0,
                clarification_ratio=0,
                user_satisfaction_indicators=0,
                user_frustration_indicators=0,
                conversation_momentum=0,
            ),
            conversation_state=ConversationState.STARTING,
            overall_quality_score=0.5,
            strengths=[],
            weaknesses=[],
            recommendations=["Add messages to the conversation to begin analysis"],
        )

    def _determine_state(
        self,
        turns: list[ConversationTurn],
        topic_analysis: TopicAnalysis,
    ) -> ConversationState:
        """Determine the current conversation state."""
        if not turns:
            return ConversationState.STARTING

        if len(turns) == 1:
            return ConversationState.STARTING

        # Check for stalling (many low-quality turns)
        recent_turns = turns[-3:] if len(turns) >= 3 else turns
        recent_quality = sum(t.turn_score for t in recent_turns) / len(recent_turns)

        if recent_quality < 0.3:
            return ConversationState.STALLED

        # Check for topic drift/redirect
        if topic_analysis.has_topic_drift:
            return ConversationState.REDIRECTING

        # Check for conclusion patterns
        last_user = turns[-1].user_message.content.lower()
        conclusion_patterns = ["thanks", "thank you", "bye", "goodbye", "that's all"]
        if any(p in last_user for p in conclusion_patterns):
            return ConversationState.CONCLUDING

        return ConversationState.FLOWING

    def _generate_insights(
        self,
        turns: list[ConversationTurn],
        topic_analysis: TopicAnalysis,
        consistency_analysis: ConsistencyAnalysis,
        engagement_metrics: EngagementMetrics,
    ) -> tuple[list[str], list[str], list[str]]:
        """Generate strengths, weaknesses, and recommendations."""
        strengths = []
        weaknesses = []
        recommendations = []

        # Analyze turn quality
        high_quality_turns = sum(
            1 for t in turns if t.quality in [TurnQuality.EXCELLENT, TurnQuality.GOOD]
        )
        if high_quality_turns > len(turns) * 0.7:
            strengths.append("High quality responses throughout conversation")
        elif high_quality_turns < len(turns) * 0.3:
            weaknesses.append("Many low-quality responses detected")
            recommendations.append("Focus on improving relevance and coherence")

        # Analyze consistency
        if consistency_analysis.is_consistent:
            strengths.append("Good consistency maintained across turns")
        else:
            weaknesses.append("Inconsistencies detected in responses")
            recommendations.append("Improve context tracking to maintain consistency")

        # Analyze engagement
        if engagement_metrics.engagement_score > 0.7:
            strengths.append("Good user engagement indicators")
        elif engagement_metrics.user_frustration_indicators > 0:
            weaknesses.append("User frustration signals detected")
            recommendations.append("Address user concerns more directly")

        # Analyze topic handling
        if not topic_analysis.has_topic_drift:
            strengths.append("Focused topic handling")
        else:
            weaknesses.append("Topic drift detected in conversation")
            recommendations.append("Maintain better topic focus")

        if not strengths:
            strengths.append("Conversation completed without critical failures")

        if not recommendations:
            recommendations.append("Continue current conversation approach")

        return strengths, weaknesses, recommendations


# Convenience functions


def create_conversation(
    messages: Optional[list[dict[str, str]]] = None,
) -> Conversation:
    """Create a new conversation.

    Args:
        messages: Optional list of message dicts

    Returns:
        Conversation object
    """
    return Conversation(messages)


def analyze_conversation(
    conversation: Conversation,
) -> ConversationReport:
    """Analyze a conversation.

    Args:
        conversation: Conversation to analyze

    Returns:
        ConversationReport object
    """
    analyzer = ConversationAnalyzer()
    return analyzer.analyze(conversation)


def analyze_messages(
    messages: list[dict[str, str]],
) -> ConversationReport:
    """Analyze a list of messages.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        ConversationReport object
    """
    conversation = Conversation(messages)
    return analyze_conversation(conversation)


def quick_conversation_check(
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Quick check of conversation quality.

    Args:
        messages: List of message dicts

    Returns:
        Dictionary with quick analysis results
    """
    conversation = Conversation(messages)
    turns = conversation.get_turns()

    if not turns:
        return {
            "n_turns": 0,
            "overall_quality": "unknown",
            "is_consistent": True,
            "has_topic_drift": False,
        }

    # Quick turn analysis
    turn_analyzer = TurnAnalyzer()
    turn_scores = []
    for turn in turns:
        analysis = turn_analyzer.analyze_turn(
            turn.user_message.content,
            turn.assistant_response.content,
        )
        turn_scores.append(analysis["avg_score"])

    avg_score = sum(turn_scores) / len(turn_scores)

    if avg_score >= 0.7:
        quality = "good"
    elif avg_score >= 0.4:
        quality = "acceptable"
    else:
        quality = "poor"

    return {
        "n_turns": len(turns),
        "overall_quality": quality,
        "avg_turn_score": avg_score,
        "is_consistent": True,  # Simplified
        "has_topic_drift": False,  # Simplified
    }


def get_conversation_summary(
    conversation: Conversation,
    max_turns: int = 3,
) -> str:
    """Get a brief summary of the conversation.

    Args:
        conversation: Conversation to summarize
        max_turns: Maximum turns to include in summary

    Returns:
        Text summary
    """
    turns = conversation.get_turns()

    if not turns:
        return "Empty conversation"

    summary_parts = []
    for i, turn in enumerate(turns[:max_turns]):
        user_preview = turn.user_message.content[:50]
        if len(turn.user_message.content) > 50:
            user_preview += "..."
        summary_parts.append(f"Turn {i + 1}: User asked about '{user_preview}'")

    if len(turns) > max_turns:
        summary_parts.append(f"... and {len(turns) - max_turns} more turns")

    return "\n".join(summary_parts)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import ConsistencyChecker. The canonical name is
# ConversationConsistencyChecker.
ConsistencyChecker = ConversationConsistencyChecker
