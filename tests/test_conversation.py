"""Tests for multi-turn conversation analysis."""

import pytest

from insideLLMs.conversation import (
    ConsistencyAnalysis,
    ConsistencyChecker,
    Conversation,
    ConversationAnalyzer,
    ConversationMessage,
    ConversationReport,
    ConversationState,
    ConversationTurn,
    EngagementAnalyzer,
    EngagementMetrics,
    MemoryReference,
    MessageRole,
    TopicAnalysis,
    TopicTracker,
    TopicTransition,
    TurnAnalyzer,
    TurnQuality,
    analyze_conversation,
    analyze_messages,
    create_conversation,
    get_conversation_summary,
    quick_conversation_check,
)


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_all_roles_exist(self):
        """Test all expected roles exist."""
        assert MessageRole.USER
        assert MessageRole.ASSISTANT
        assert MessageRole.SYSTEM


class TestTurnQuality:
    """Tests for TurnQuality enum."""

    def test_all_qualities_exist(self):
        """Test all expected quality levels exist."""
        assert TurnQuality.EXCELLENT
        assert TurnQuality.GOOD
        assert TurnQuality.ACCEPTABLE
        assert TurnQuality.POOR
        assert TurnQuality.FAILED


class TestConversationState:
    """Tests for ConversationState enum."""

    def test_all_states_exist(self):
        """Test all expected states exist."""
        assert ConversationState.STARTING
        assert ConversationState.FLOWING
        assert ConversationState.CLARIFYING
        assert ConversationState.REDIRECTING
        assert ConversationState.STALLED
        assert ConversationState.CONCLUDING


class TestConversationMessage:
    """Tests for ConversationMessage dataclass."""

    def test_basic_creation(self):
        """Test basic message creation."""
        msg = ConversationMessage(
            role=MessageRole.USER,
            content="Hello, how are you?",
        )
        assert msg.role == MessageRole.USER
        assert msg.word_count == 4
        assert msg.char_count == 19

    def test_to_dict(self):
        """Test conversion to dictionary."""
        msg = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content="I'm doing well, thank you!",
            metadata={"model": "test"},
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["word_count"] == 5
        assert d["metadata"]["model"] == "test"


class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""

    def test_response_ratio(self):
        """Test response ratio calculation."""
        turn = ConversationTurn(
            turn_number=1,
            user_message=ConversationMessage(
                role=MessageRole.USER,
                content="Hello world",
            ),
            assistant_response=ConversationMessage(
                role=MessageRole.ASSISTANT,
                content="Hello world back to you today",
            ),
        )
        assert turn.response_ratio == 3.0  # 6/2

    def test_turn_score(self):
        """Test turn score calculation."""
        turn = ConversationTurn(
            turn_number=1,
            user_message=ConversationMessage(MessageRole.USER, "Hi"),
            assistant_response=ConversationMessage(MessageRole.ASSISTANT, "Hello"),
            relevance_score=0.8,
            coherence_score=0.6,
        )
        assert turn.turn_score == 0.7

    def test_to_dict(self):
        """Test conversion to dictionary."""
        turn = ConversationTurn(
            turn_number=1,
            user_message=ConversationMessage(MessageRole.USER, "Test"),
            assistant_response=ConversationMessage(MessageRole.ASSISTANT, "Response"),
            topic="greeting",
            quality=TurnQuality.GOOD,
        )
        d = turn.to_dict()
        assert d["turn_number"] == 1
        assert d["topic"] == "greeting"
        assert d["quality"] == "good"


class TestMemoryReference:
    """Tests for MemoryReference dataclass."""

    def test_basic_creation(self):
        """Test basic reference creation."""
        ref = MemoryReference(
            source_turn=1,
            target_turn=3,
            reference_type="entity",
            content="Python programming",
            is_accurate=True,
            confidence=0.9,
        )
        assert ref.source_turn == 1
        assert ref.target_turn == 3
        assert ref.is_accurate

    def test_to_dict(self):
        """Test conversion to dictionary."""
        ref = MemoryReference(
            source_turn=1,
            target_turn=2,
            reference_type="fact",
            content="Test",
        )
        d = ref.to_dict()
        assert d["reference_type"] == "fact"


class TestTopicAnalysis:
    """Tests for TopicAnalysis dataclass."""

    def test_n_topics(self):
        """Test topic count."""
        analysis = TopicAnalysis(
            main_topics=["python", "coding", "testing"],
            topic_sequence=[(1, "python"), (2, "coding")],
            topic_transitions=[],
            topic_coverage={"python": 3, "coding": 2, "testing": 1},
            topic_depth={"python": 0.5, "coding": 0.33, "testing": 0.17},
        )
        assert analysis.n_topics == 3

    def test_avg_topic_duration(self):
        """Test average topic duration."""
        analysis = TopicAnalysis(
            main_topics=["python", "coding"],
            topic_sequence=[],
            topic_transitions=[],
            topic_coverage={"python": 3, "coding": 3},
            topic_depth={},
        )
        assert analysis.avg_topic_duration == 3.0

    def test_has_topic_drift(self):
        """Test topic drift detection."""
        no_drift = TopicAnalysis(
            main_topics=["python"],
            topic_sequence=[(1, "python"), (2, "python")],
            topic_transitions=[(2, TopicTransition.CONTINUATION)],
            topic_coverage={"python": 2},
            topic_depth={"python": 1.0},
        )
        assert not no_drift.has_topic_drift

        has_drift = TopicAnalysis(
            main_topics=["python", "cooking", "music"],
            topic_sequence=[],
            topic_transitions=[
                (2, TopicTransition.ABRUPT_CHANGE),
                (3, TopicTransition.ABRUPT_CHANGE),
            ],
            topic_coverage={},
            topic_depth={},
        )
        assert has_drift.has_topic_drift


class TestConsistencyAnalysis:
    """Tests for ConsistencyAnalysis dataclass."""

    def test_overall_consistency(self):
        """Test overall consistency calculation."""
        analysis = ConsistencyAnalysis(
            memory_references=[],
            factual_consistency_score=0.9,
            stylistic_consistency_score=0.8,
            contextual_consistency_score=0.7,
            inconsistencies=[],
        )
        assert analysis.overall_consistency == pytest.approx(0.8)

    def test_is_consistent(self):
        """Test consistency check."""
        consistent = ConsistencyAnalysis(
            memory_references=[],
            factual_consistency_score=0.9,
            stylistic_consistency_score=0.8,
            contextual_consistency_score=0.7,
            inconsistencies=[],
        )
        assert consistent.is_consistent

        inconsistent = ConsistencyAnalysis(
            memory_references=[],
            factual_consistency_score=0.5,
            stylistic_consistency_score=0.5,
            contextual_consistency_score=0.5,
            inconsistencies=[{"issue": "test"}],
        )
        assert not inconsistent.is_consistent


class TestEngagementMetrics:
    """Tests for EngagementMetrics dataclass."""

    def test_engagement_score(self):
        """Test engagement score calculation."""
        metrics = EngagementMetrics(
            avg_response_length=100,
            response_length_variance=10,
            question_ratio=0.5,
            clarification_ratio=0.1,
            user_satisfaction_indicators=5,
            user_frustration_indicators=1,
            conversation_momentum=0.8,
        )
        # satisfaction = 5/6 ≈ 0.83, engagement = (0.83 + 0.8) / 2 ≈ 0.82
        assert metrics.engagement_score > 0.8

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = EngagementMetrics(
            avg_response_length=100,
            response_length_variance=10,
            question_ratio=0.5,
            clarification_ratio=0.1,
            user_satisfaction_indicators=3,
            user_frustration_indicators=0,
            conversation_momentum=0.7,
        )
        d = metrics.to_dict()
        assert "engagement_score" in d


class TestConversation:
    """Tests for Conversation class."""

    def test_create_empty_conversation(self):
        """Test creating empty conversation."""
        conv = Conversation()
        assert conv.n_messages == 0
        assert conv.n_turns == 0

    def test_create_with_messages(self):
        """Test creating conversation with messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        conv = Conversation(messages)
        assert conv.n_messages == 2
        assert conv.n_turns == 1

    def test_add_messages(self):
        """Test adding messages."""
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")
        assert conv.n_messages == 2
        assert conv.n_turns == 1

    def test_get_turns(self):
        """Test getting turns."""
        conv = Conversation()
        conv.add_user_message("Question 1?")
        conv.add_assistant_message("Answer 1")
        conv.add_user_message("Question 2?")
        conv.add_assistant_message("Answer 2")

        turns = conv.get_turns()
        assert len(turns) == 2
        assert turns[0].turn_number == 1
        assert turns[1].turn_number == 2

    def test_to_text(self):
        """Test conversion to text."""
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")

        text = conv.to_text()
        assert "User: Hello" in text
        assert "Assistant: Hi!" in text

    def test_to_dict(self):
        """Test conversion to dictionary."""
        conv = Conversation()
        conv.add_user_message("Test")
        d = conv.to_dict()
        assert d["n_messages"] == 1


class TestTurnAnalyzer:
    """Tests for TurnAnalyzer."""

    def test_analyze_relevant_turn(self):
        """Test analyzing a relevant turn."""
        analyzer = TurnAnalyzer()
        result = analyzer.analyze_turn(
            user_message="What is Python programming?",
            assistant_response="Python is a popular programming language used for web development.",
        )
        assert result["relevance_score"] > 0  # Should have some overlap
        assert "quality" in result

    def test_analyze_irrelevant_turn(self):
        """Test analyzing an irrelevant turn."""
        analyzer = TurnAnalyzer()
        result = analyzer.analyze_turn(
            user_message="What is Python?",
            assistant_response="The weather today is sunny.",
        )
        # Low relevance expected
        assert result["relevance_score"] < 0.5

    def test_custom_relevance_function(self):
        """Test with custom relevance function."""

        def always_high(query: str, response: str) -> float:
            return 0.95

        analyzer = TurnAnalyzer(relevance_fn=always_high)
        result = analyzer.analyze_turn("Any question", "Any response")
        assert result["relevance_score"] == 0.95


class TestTopicTracker:
    """Tests for TopicTracker."""

    def test_add_turn(self):
        """Test adding a turn."""
        tracker = TopicTracker()
        topic = tracker.add_turn(1, "Tell me about Python", "Python is...")
        assert topic is not None
        assert len(topic) > 0

    def test_analyze(self):
        """Test analyzing topics."""
        tracker = TopicTracker()
        tracker.add_turn(1, "Tell me about Python", "Python is a language")
        tracker.add_turn(2, "What about Python syntax?", "The syntax is clean")
        tracker.add_turn(3, "Explain Python functions", "Functions in Python...")

        analysis = tracker.analyze()
        assert len(analysis.main_topics) > 0
        assert len(analysis.topic_sequence) == 3

    def test_topic_transitions(self):
        """Test topic transition detection."""
        tracker = TopicTracker()
        tracker.add_turn(1, "Talk about cats", "Cats are cute")
        tracker.add_turn(2, "What do cats eat?", "Cats eat fish")
        tracker.add_turn(3, "Tell me about dogs", "Dogs are loyal")

        analysis = tracker.analyze()
        assert len(analysis.topic_transitions) == 2

    def test_clear(self):
        """Test clearing tracker."""
        tracker = TopicTracker()
        tracker.add_turn(1, "Test", "Response")
        tracker.clear()
        analysis = tracker.analyze()
        assert len(analysis.topic_sequence) == 0


class TestConsistencyChecker:
    """Tests for ConsistencyChecker."""

    def test_check_consistent(self):
        """Test checking consistent conversation."""
        checker = ConsistencyChecker()

        turns = [
            ConversationTurn(
                turn_number=1,
                user_message=ConversationMessage(MessageRole.USER, "What is Python?"),
                assistant_response=ConversationMessage(
                    MessageRole.ASSISTANT, "Python is a programming language."
                ),
            ),
            ConversationTurn(
                turn_number=2,
                user_message=ConversationMessage(MessageRole.USER, "Tell me more about Python."),
                assistant_response=ConversationMessage(
                    MessageRole.ASSISTANT, "Python is used for web development and data science."
                ),
            ),
        ]

        checker.add_turn(1, turns[0].assistant_response.content)
        checker.add_turn(2, turns[1].assistant_response.content)

        result = checker.check(turns)
        assert isinstance(result, ConsistencyAnalysis)

    def test_clear(self):
        """Test clearing checker."""
        checker = ConsistencyChecker()
        checker.add_turn(1, "Test response")
        checker.clear()
        # Should not raise errors


class TestEngagementAnalyzer:
    """Tests for EngagementAnalyzer."""

    def test_analyze_engaged_conversation(self):
        """Test analyzing engaged conversation."""
        analyzer = EngagementAnalyzer()

        turns = [
            ConversationTurn(
                turn_number=1,
                user_message=ConversationMessage(MessageRole.USER, "Help me with Python?"),
                assistant_response=ConversationMessage(
                    MessageRole.ASSISTANT, "Sure! Python is a great language for beginners."
                ),
                relevance_score=0.8,
                coherence_score=0.8,
            ),
            ConversationTurn(
                turn_number=2,
                user_message=ConversationMessage(MessageRole.USER, "Thanks, that's helpful!"),
                assistant_response=ConversationMessage(
                    MessageRole.ASSISTANT, "You're welcome! Let me know if you have more questions."
                ),
                relevance_score=0.9,
                coherence_score=0.9,
            ),
        ]

        metrics = analyzer.analyze(turns)
        assert metrics.user_satisfaction_indicators > 0
        assert metrics.engagement_score > 0

    def test_analyze_empty(self):
        """Test analyzing empty conversation."""
        analyzer = EngagementAnalyzer()
        metrics = analyzer.analyze([])
        assert metrics.avg_response_length == 0


class TestConversationAnalyzer:
    """Tests for ConversationAnalyzer."""

    def test_analyze_simple_conversation(self):
        """Test analyzing a simple conversation."""
        conv = Conversation()
        conv.add_user_message("What is Python?")
        conv.add_assistant_message("Python is a programming language.")
        conv.add_user_message("What can I do with Python?")
        conv.add_assistant_message("You can build web apps, analyze data, and more.")

        analyzer = ConversationAnalyzer()
        report = analyzer.analyze(conv)

        assert isinstance(report, ConversationReport)
        assert report.n_turns == 2
        assert len(report.turns) == 2

    def test_analyze_empty_conversation(self):
        """Test analyzing empty conversation."""
        conv = Conversation()
        analyzer = ConversationAnalyzer()
        report = analyzer.analyze(conv)

        assert report.n_turns == 0
        assert report.conversation_state == ConversationState.STARTING

    def test_report_quality_level(self):
        """Test report quality level classification."""
        conv = Conversation()
        conv.add_user_message("Test question?")
        conv.add_assistant_message("A relevant test response about the question.")

        analyzer = ConversationAnalyzer()
        report = analyzer.analyze(conv)

        assert report.quality_level in ["excellent", "good", "acceptable", "poor", "failed"]

    def test_strengths_and_weaknesses(self):
        """Test that strengths and weaknesses are generated."""
        conv = Conversation()
        conv.add_user_message("Question 1?")
        conv.add_assistant_message("Answer 1 with details.")
        conv.add_user_message("Question 2?")
        conv.add_assistant_message("Answer 2 with more details.")

        analyzer = ConversationAnalyzer()
        report = analyzer.analyze(conv)

        assert len(report.strengths) > 0 or len(report.weaknesses) > 0
        assert len(report.recommendations) > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_conversation(self):
        """Test create_conversation function."""
        conv = create_conversation(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        )
        assert isinstance(conv, Conversation)
        assert conv.n_turns == 1

    def test_analyze_conversation(self):
        """Test analyze_conversation function."""
        conv = Conversation()
        conv.add_user_message("Question?")
        conv.add_assistant_message("Answer.")

        report = analyze_conversation(conv)
        assert isinstance(report, ConversationReport)

    def test_analyze_messages(self):
        """Test analyze_messages function."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        report = analyze_messages(messages)
        assert isinstance(report, ConversationReport)
        assert report.n_turns == 1

    def test_quick_conversation_check(self):
        """Test quick_conversation_check function."""
        messages = [
            {"role": "user", "content": "What is Python programming?"},
            {"role": "assistant", "content": "Python is a popular programming language."},
            {"role": "user", "content": "Thanks, that's helpful!"},
            {"role": "assistant", "content": "You're welcome!"},
        ]
        result = quick_conversation_check(messages)

        assert result["n_turns"] == 2
        assert result["overall_quality"] in ["good", "acceptable", "poor"]
        assert "avg_turn_score" in result

    def test_quick_conversation_check_empty(self):
        """Test quick_conversation_check with empty messages."""
        result = quick_conversation_check([])
        assert result["n_turns"] == 0
        assert result["overall_quality"] == "unknown"

    def test_get_conversation_summary(self):
        """Test get_conversation_summary function."""
        conv = Conversation()
        conv.add_user_message("What is Python?")
        conv.add_assistant_message("Python is a programming language.")
        conv.add_user_message("How do I learn Python?")
        conv.add_assistant_message("Start with tutorials.")

        summary = get_conversation_summary(conv)
        assert "Turn 1" in summary
        assert "Turn 2" in summary

    def test_get_conversation_summary_empty(self):
        """Test get_conversation_summary with empty conversation."""
        conv = Conversation()
        summary = get_conversation_summary(conv)
        assert summary == "Empty conversation"


class TestConversationReport:
    """Tests for ConversationReport."""

    def test_quality_level_excellent(self):
        """Test excellent quality level."""
        report = ConversationReport(
            n_turns=2,
            turns=[],
            topic_analysis=TopicAnalysis([], [], [], {}, {}),
            consistency_analysis=ConsistencyAnalysis([], 1, 1, 1, []),
            engagement_metrics=EngagementMetrics(100, 10, 0.5, 0.1, 5, 0, 0.8),
            conversation_state=ConversationState.FLOWING,
            overall_quality_score=0.9,
            strengths=["Great"],
            weaknesses=[],
            recommendations=[],
        )
        assert report.quality_level == "excellent"

    def test_quality_level_poor(self):
        """Test poor quality level."""
        report = ConversationReport(
            n_turns=2,
            turns=[],
            topic_analysis=TopicAnalysis([], [], [], {}, {}),
            consistency_analysis=ConsistencyAnalysis([], 0.5, 0.5, 0.5, []),
            engagement_metrics=EngagementMetrics(100, 10, 0.5, 0.1, 0, 5, 0.3),
            conversation_state=ConversationState.STALLED,
            overall_quality_score=0.35,
            strengths=[],
            weaknesses=["Many issues"],
            recommendations=["Improve"],
        )
        assert report.quality_level == "poor"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = ConversationReport(
            n_turns=1,
            turns=[],
            topic_analysis=TopicAnalysis(["test"], [], [], {"test": 1}, {"test": 1.0}),
            consistency_analysis=ConsistencyAnalysis([], 1, 1, 1, []),
            engagement_metrics=EngagementMetrics(50, 5, 0.5, 0.1, 2, 0, 0.7),
            conversation_state=ConversationState.FLOWING,
            overall_quality_score=0.75,
            strengths=["Good"],
            weaknesses=[],
            recommendations=["Continue"],
        )
        d = report.to_dict()
        assert d["n_turns"] == 1
        assert d["quality_level"] == "good"
        assert "topic_analysis" in d


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_user_message(self):
        """Test conversation with only user message."""
        conv = Conversation()
        conv.add_user_message("Hello?")
        # No assistant response
        turns = conv.get_turns()
        assert len(turns) == 0  # No complete turn

    def test_single_assistant_message(self):
        """Test conversation with only assistant message."""
        conv = Conversation()
        conv.add_assistant_message("Welcome!")
        turns = conv.get_turns()
        assert len(turns) == 0  # No complete turn

    def test_very_long_message(self):
        """Test with very long message."""
        conv = Conversation()
        long_text = "word " * 1000
        conv.add_user_message(long_text)
        conv.add_assistant_message("Response")

        turns = conv.get_turns()
        assert turns[0].user_message.word_count == 1000

    def test_empty_message_content(self):
        """Test with empty message content."""
        msg = ConversationMessage(
            role=MessageRole.USER,
            content="",
        )
        assert msg.word_count == 0
        assert msg.char_count == 0

    def test_special_characters_in_message(self):
        """Test with special characters."""
        conv = Conversation()
        conv.add_user_message("Hello! @#$% How are you? 你好")
        conv.add_assistant_message("I'm fine, thanks! \U0001f600")

        turns = conv.get_turns()
        assert len(turns) == 1

    def test_multiple_questions_in_turn(self):
        """Test turn with multiple questions."""
        analyzer = TurnAnalyzer()
        result = analyzer.analyze_turn(
            user_message="What is Python? Is it hard to learn? Can I use it for web dev?",
            assistant_response="Python is a programming language that's easy to learn and can be used for web development.",
        )
        assert "relevance_score" in result

    def test_conversation_with_system_message(self):
        """Test conversation with system prompt."""
        conv = Conversation(system_prompt="You are a helpful assistant.")
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")

        text = conv.to_text(include_system=True)
        assert "System: You are a helpful assistant." in text
