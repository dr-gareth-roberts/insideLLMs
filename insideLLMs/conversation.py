"""Multi-turn conversation analysis for LLM evaluation.

This module provides tools for analyzing multi-turn conversations with LLMs:

- Turn-level analysis (quality, relevance, coherence per turn)
- Conversation flow tracking (topic drift, context maintenance)
- Memory and consistency evaluation across turns
- Engagement and interaction patterns
- Conversation summarization and metadata extraction
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class MessageRole(Enum):
    """Role of a message in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TurnQuality(Enum):
    """Quality rating for a conversation turn."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


class ConversationState(Enum):
    """State of conversation flow."""

    STARTING = "starting"
    FLOWING = "flowing"
    CLARIFYING = "clarifying"
    REDIRECTING = "redirecting"
    STALLED = "stalled"
    CONCLUDING = "concluding"


class TopicTransition(Enum):
    """Type of topic transition."""

    CONTINUATION = "continuation"
    NATURAL_SHIFT = "natural_shift"
    EXPLICIT_CHANGE = "explicit_change"
    ABRUPT_CHANGE = "abrupt_change"
    RETURN_TO_PREVIOUS = "return_to_previous"


@dataclass
class ConversationMessage:
    """A single message in a conversation."""

    role: MessageRole
    content: str
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Get word count of message."""
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        """Get character count of message."""
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """A conversation turn (user message + assistant response)."""

    turn_number: int
    user_message: ConversationMessage
    assistant_response: ConversationMessage
    topic: str | None = None
    quality: TurnQuality = TurnQuality.ACCEPTABLE
    relevance_score: float = 0.5
    coherence_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def response_ratio(self) -> float:
        """Calculate response length ratio (assistant/user)."""
        if self.user_message.word_count == 0:
            return 0.0
        return self.assistant_response.word_count / self.user_message.word_count

    @property
    def turn_score(self) -> float:
        """Calculate overall turn score."""
        return (self.relevance_score + self.coherence_score) / 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """A reference to information from earlier in the conversation."""

    source_turn: int
    target_turn: int
    reference_type: str  # e.g., "entity", "fact", "instruction"
    content: str
    is_accurate: bool = True
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Analysis of topics in a conversation."""

    main_topics: list[str]
    topic_sequence: list[tuple[int, str]]  # (turn_number, topic)
    topic_transitions: list[tuple[int, TopicTransition]]
    topic_coverage: dict[str, int]  # topic -> turn count
    topic_depth: dict[str, float]  # topic -> depth score

    @property
    def n_topics(self) -> int:
        """Get number of distinct topics."""
        return len(self.main_topics)

    @property
    def avg_topic_duration(self) -> float:
        """Calculate average number of turns per topic."""
        if not self.main_topics:
            return 0.0
        return sum(self.topic_coverage.values()) / len(self.main_topics)

    @property
    def has_topic_drift(self) -> bool:
        """Check if conversation has significant topic drift."""
        abrupt_changes = sum(
            1 for _, t in self.topic_transitions if t == TopicTransition.ABRUPT_CHANGE
        )
        return abrupt_changes > 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Analysis of consistency across conversation turns."""

    memory_references: list[MemoryReference]
    factual_consistency_score: float
    stylistic_consistency_score: float
    contextual_consistency_score: float
    inconsistencies: list[dict[str, Any]]

    @property
    def overall_consistency(self) -> float:
        """Calculate overall consistency score."""
        return (
            self.factual_consistency_score
            + self.stylistic_consistency_score
            + self.contextual_consistency_score
        ) / 3

    @property
    def is_consistent(self) -> bool:
        """Check if conversation is consistent."""
        return self.overall_consistency >= 0.7

    @property
    def n_inconsistencies(self) -> int:
        """Get number of detected inconsistencies."""
        return len(self.inconsistencies)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Metrics for conversation engagement."""

    avg_response_length: float
    response_length_variance: float
    question_ratio: float  # % of turns with questions
    clarification_ratio: float  # % of turns seeking clarification
    user_satisfaction_indicators: int
    user_frustration_indicators: int
    conversation_momentum: float  # 0-1, how well conversation flows

    @property
    def engagement_score(self) -> float:
        """Calculate overall engagement score."""
        satisfaction = self.user_satisfaction_indicators / max(
            1, self.user_satisfaction_indicators + self.user_frustration_indicators
        )
        return (satisfaction + self.conversation_momentum) / 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Comprehensive conversation analysis report."""

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
        """Get quality level classification."""
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
        """Convert to dictionary."""
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
    """Represents a multi-turn conversation."""

    def __init__(
        self,
        messages: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ):
        """Initialize conversation.

        Args:
            messages: Optional list of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
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
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        """Add a message to the conversation.

        Args:
            role: Message role
            content: Message content
            timestamp: Optional timestamp
            metadata: Optional metadata

        Returns:
            The added message
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
        """Add a user message."""
        return self.add_message(MessageRole.USER, content, **kwargs)

    def add_assistant_message(self, content: str, **kwargs) -> ConversationMessage:
        """Add an assistant message."""
        return self.add_message(MessageRole.ASSISTANT, content, **kwargs)

    @property
    def messages(self) -> list[ConversationMessage]:
        """Get all messages."""
        return self._messages.copy()

    @property
    def n_messages(self) -> int:
        """Get number of messages."""
        return len(self._messages)

    @property
    def n_turns(self) -> int:
        """Get number of conversation turns."""
        user_messages = [m for m in self._messages if m.role == MessageRole.USER]
        return len(user_messages)

    def get_turns(self) -> list[ConversationTurn]:
        """Get conversation as list of turns.

        Returns:
            List of ConversationTurn objects
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
        """Convert conversation to plain text.

        Args:
            include_system: Whether to include system prompt

        Returns:
            Text representation
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
        """Convert to dictionary."""
        return {
            "system_prompt": self._system_prompt,
            "messages": [m.to_dict() for m in self._messages],
            "n_messages": self.n_messages,
            "n_turns": self.n_turns,
        }


class TurnAnalyzer:
    """Analyze individual conversation turns."""

    def __init__(
        self,
        relevance_fn: Callable[[str, str], float] | None = None,
        coherence_fn: Callable[[str, str], float] | None = None,
    ):
        """Initialize analyzer.

        Args:
            relevance_fn: Function to compute relevance score
            coherence_fn: Function to compute coherence score
        """
        self._relevance_fn = relevance_fn or self._default_relevance
        self._coherence_fn = coherence_fn or self._default_coherence

    @staticmethod
    def _default_relevance(query: str, response: str) -> float:
        """Default relevance scoring using keyword overlap."""
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
        """Default coherence scoring."""
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
        context: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a single conversation turn.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            context: Optional previous conversation context

        Returns:
            Dictionary with turn analysis
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
    """Track topics across conversation turns."""

    def __init__(
        self,
        topic_extractor: Callable[[str], list[str]] | None = None,
    ):
        """Initialize tracker.

        Args:
            topic_extractor: Function to extract topics from text
        """
        self._topic_extractor = topic_extractor or self._default_topic_extractor
        self._topic_sequence: list[tuple[int, str]] = []
        self._topic_history: list[str] = []

    @staticmethod
    def _default_topic_extractor(text: str) -> list[str]:
        """Simple keyword-based topic extraction."""
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
        """Add a turn and track topic.

        Args:
            turn_number: Turn number
            user_message: User's message
            response: Assistant's response

        Returns:
            Detected topic for this turn
        """
        combined_text = f"{user_message} {response}"
        topics = self._topic_extractor(combined_text)
        main_topic = topics[0] if topics else "general"

        self._topic_sequence.append((turn_number, main_topic))
        self._topic_history.append(main_topic)

        return main_topic

    def get_transition_type(self, prev_topic: str, curr_topic: str) -> TopicTransition:
        """Determine type of topic transition.

        Args:
            prev_topic: Previous topic
            curr_topic: Current topic

        Returns:
            TopicTransition type
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
        """Generate topic analysis.

        Returns:
            TopicAnalysis object
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
        """Clear tracking state."""
        self._topic_sequence.clear()
        self._topic_history.clear()


class ConversationConsistencyChecker:
    """Check consistency across conversation turns."""

    def __init__(
        self,
        similarity_fn: Callable[[str, str], float] | None = None,
    ):
        """Initialize checker.

        Args:
            similarity_fn: Function to compute text similarity
        """
        self._similarity_fn = similarity_fn or self._default_similarity
        self._facts: dict[int, list[str]] = {}  # turn -> facts

    @staticmethod
    def _default_similarity(text1: str, text2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        overlap = len(words1 & words2)
        return 2 * overlap / (len(words1) + len(words2))

    def add_turn(self, turn_number: int, response: str) -> None:
        """Add a turn's response for consistency tracking.

        Args:
            turn_number: Turn number
            response: Assistant's response
        """
        # Extract potential facts (sentences with declarative statements)
        sentences = re.split(r"[.!?]", response)
        facts = [s.strip() for s in sentences if len(s.strip()) > 10]
        self._facts[turn_number] = facts

    def check(self, turns: list[ConversationTurn]) -> ConsistencyAnalysis:
        """Check consistency across turns.

        Args:
            turns: List of conversation turns

        Returns:
            ConsistencyAnalysis object
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
        """Clear tracking state."""
        self._facts.clear()


class EngagementAnalyzer:
    """Analyze engagement patterns in conversations."""

    def __init__(self):
        """Initialize analyzer."""
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
        """Analyze engagement across turns.

        Args:
            turns: List of conversation turns

        Returns:
            EngagementMetrics object
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
    """Comprehensive conversation analyzer."""

    def __init__(
        self,
        turn_analyzer: TurnAnalyzer | None = None,
        topic_tracker: TopicTracker | None = None,
        consistency_checker: ConversationConsistencyChecker | None = None,
        engagement_analyzer: EngagementAnalyzer | None = None,
    ):
        """Initialize analyzer.

        Args:
            turn_analyzer: Optional custom turn analyzer
            topic_tracker: Optional custom topic tracker
            consistency_checker: Optional custom consistency checker
            engagement_analyzer: Optional custom engagement analyzer
        """
        self._turn_analyzer = turn_analyzer or TurnAnalyzer()
        self._topic_tracker = topic_tracker or TopicTracker()
        self._consistency_checker = consistency_checker or ConversationConsistencyChecker()
        self._engagement_analyzer = engagement_analyzer or EngagementAnalyzer()

    def analyze(
        self,
        conversation: Conversation,
    ) -> ConversationReport:
        """Analyze a complete conversation.

        Args:
            conversation: Conversation to analyze

        Returns:
            ConversationReport object
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
    messages: list[dict[str, str]] | None = None,
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
