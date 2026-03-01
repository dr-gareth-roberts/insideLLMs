"""Tests for Human-in-the-Loop (HITL) module."""

import threading
import time
from datetime import datetime
from unittest.mock import Mock

from insideLLMs.agents.hitl import (
    Annotation,
    AnnotationCollector,
    AnnotationWorkflow,
    # Workflows
    ApprovalWorkflow,
    # Input handlers
    CallbackInputHandler,
    ConsensusValidator,
    Feedback,
    # Collectors
    FeedbackCollector,
    # Core types
    FeedbackType,
    # Config
    HITLConfig,
    # Sessions
    HITLSession,
    # Validators
    HumanValidator,
    InteractiveSession,
    Priority,
    PriorityReviewQueue,
    ReviewItem,
    # Queues
    ReviewQueue,
    ReviewStatus,
    ReviewWorkflow,
    collect_feedback,
    create_hitl_session,
    quick_review,
)

# =============================================================================
# Test FeedbackType Enum
# =============================================================================


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_all_types_exist(self):
        """Test all feedback types are defined."""
        assert FeedbackType.APPROVE == "approve"
        assert FeedbackType.REJECT == "reject"
        assert FeedbackType.EDIT == "edit"
        assert FeedbackType.FLAG == "flag"
        assert FeedbackType.SKIP == "skip"
        assert FeedbackType.RATING == "rating"
        assert FeedbackType.COMMENT == "comment"
        assert FeedbackType.CORRECTION == "correction"


class TestReviewStatus:
    """Tests for ReviewStatus enum."""

    def test_all_statuses_exist(self):
        """Test all review statuses are defined."""
        assert ReviewStatus.PENDING == "pending"
        assert ReviewStatus.IN_PROGRESS == "in_progress"
        assert ReviewStatus.APPROVED == "approved"
        assert ReviewStatus.REJECTED == "rejected"
        assert ReviewStatus.EDITED == "edited"
        assert ReviewStatus.FLAGGED == "flagged"
        assert ReviewStatus.SKIPPED == "skipped"
        assert ReviewStatus.EXPIRED == "expired"


class TestPriority:
    """Tests for Priority enum."""

    def test_priority_ordering(self):
        """Test priority values are ordered correctly."""
        assert Priority.CRITICAL.value < Priority.HIGH.value
        assert Priority.HIGH.value < Priority.MEDIUM.value
        assert Priority.MEDIUM.value < Priority.LOW.value
        assert Priority.LOW.value < Priority.BACKGROUND.value


# =============================================================================
# Test Feedback Dataclass
# =============================================================================


class TestFeedback:
    """Tests for Feedback dataclass."""

    def test_default_values(self):
        """Test default feedback values."""
        feedback = Feedback()
        assert feedback.feedback_type == FeedbackType.COMMENT
        assert feedback.content == ""
        assert feedback.rating is None
        assert feedback.edited_content is None
        assert feedback.reviewer_id is None
        assert isinstance(feedback.feedback_id, str)
        assert isinstance(feedback.timestamp, datetime)

    def test_custom_values(self):
        """Test custom feedback values."""
        feedback = Feedback(
            feedback_type=FeedbackType.RATING,
            content="Great response",
            rating=0.9,
            reviewer_id="user123",
        )
        assert feedback.feedback_type == FeedbackType.RATING
        assert feedback.content == "Great response"
        assert feedback.rating == 0.9
        assert feedback.reviewer_id == "user123"

    def test_to_dict(self):
        """Test feedback serialization."""
        feedback = Feedback(
            feedback_type=FeedbackType.EDIT,
            content="Needs improvement",
            edited_content="Improved response",
        )
        data = feedback.to_dict()
        assert data["feedback_type"] == "edit"
        assert data["content"] == "Needs improvement"
        assert data["edited_content"] == "Improved response"
        assert "timestamp" in data

    def test_from_dict(self):
        """Test feedback deserialization."""
        data = {
            "feedback_id": "fb123",
            "feedback_type": "rating",
            "content": "Good",
            "rating": 0.8,
            "timestamp": "2024-01-15T10:30:00",
        }
        feedback = Feedback.from_dict(data)
        assert feedback.feedback_id == "fb123"
        assert feedback.feedback_type == FeedbackType.RATING
        assert feedback.rating == 0.8


# =============================================================================
# Test Annotation Dataclass
# =============================================================================


class TestAnnotation:
    """Tests for Annotation dataclass."""

    def test_default_values(self):
        """Test default annotation values."""
        annotation = Annotation()
        assert annotation.text == ""
        assert annotation.label == ""
        assert annotation.confidence == 1.0
        assert isinstance(annotation.annotation_id, str)

    def test_span_annotation(self):
        """Test span annotation with offsets."""
        annotation = Annotation(
            text="important text",
            label="highlight",
            start_offset=10,
            end_offset=24,
            annotator_id="ann1",
        )
        assert annotation.start_offset == 10
        assert annotation.end_offset == 24
        assert annotation.annotator_id == "ann1"

    def test_to_dict(self):
        """Test annotation serialization."""
        annotation = Annotation(
            text="test",
            label="positive",
            confidence=0.95,
        )
        data = annotation.to_dict()
        assert data["text"] == "test"
        assert data["label"] == "positive"
        assert data["confidence"] == 0.95


# =============================================================================
# Test ReviewItem Dataclass
# =============================================================================


class TestReviewItem:
    """Tests for ReviewItem dataclass."""

    def test_default_values(self):
        """Test default review item values."""
        item = ReviewItem()
        assert item.prompt == ""
        assert item.response == ""
        assert item.status == ReviewStatus.PENDING
        assert item.priority == Priority.MEDIUM
        assert item.feedback == []
        assert item.annotations == []

    def test_add_feedback(self):
        """Test adding feedback to item."""
        item = ReviewItem(prompt="test", response="response")
        feedback = Feedback(content="Good")
        initial_time = item.updated_at

        time.sleep(0.01)
        item.add_feedback(feedback)

        assert len(item.feedback) == 1
        assert item.feedback[0].content == "Good"
        assert item.updated_at > initial_time

    def test_add_annotation(self):
        """Test adding annotation to item."""
        item = ReviewItem(prompt="test", response="response")
        annotation = Annotation(label="positive")
        item.add_annotation(annotation)

        assert len(item.annotations) == 1
        assert item.annotations[0].label == "positive"

    def test_comparison_by_priority(self):
        """Test item comparison for queue ordering."""
        high = ReviewItem(priority=Priority.HIGH)
        low = ReviewItem(priority=Priority.LOW)
        assert high < low

    def test_to_dict(self):
        """Test item serialization."""
        item = ReviewItem(
            prompt="test prompt",
            response="test response",
            status=ReviewStatus.APPROVED,
        )
        data = item.to_dict()
        assert data["prompt"] == "test prompt"
        assert data["response"] == "test response"
        assert data["status"] == "approved"


# =============================================================================
# Test HITLConfig
# =============================================================================


class TestHITLConfig:
    """Tests for HITLConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = HITLConfig()
        assert config.auto_approve_threshold == 0.9
        assert config.require_consensus is False
        assert config.min_reviewers == 1
        assert config.allow_skip is True
        assert config.allow_edit is True

    def test_custom_values(self):
        """Test custom config values."""
        config = HITLConfig(
            auto_approve_threshold=0.7,
            require_consensus=True,
            min_reviewers=3,
        )
        assert config.auto_approve_threshold == 0.7
        assert config.require_consensus is True
        assert config.min_reviewers == 3

    def test_to_dict(self):
        """Test config serialization."""
        config = HITLConfig(timeout_seconds=60)
        data = config.to_dict()
        assert data["timeout_seconds"] == 60


# =============================================================================
# Test ReviewQueue
# =============================================================================


class TestReviewQueue:
    """Tests for ReviewQueue."""

    def test_add_and_get(self):
        """Test adding and getting items."""
        queue = ReviewQueue()
        item = ReviewItem(prompt="test", response="response")

        assert queue.add(item)
        assert len(queue) == 1

        retrieved = queue.get_next()
        assert retrieved is not None
        assert retrieved.item_id == item.item_id
        assert retrieved.status == ReviewStatus.IN_PROGRESS

    def test_max_size(self):
        """Test queue max size."""
        queue = ReviewQueue(max_size=2)

        assert queue.add(ReviewItem(prompt="1", response="1"))
        assert queue.add(ReviewItem(prompt="2", response="2"))
        assert not queue.add(ReviewItem(prompt="3", response="3"))
        assert len(queue) == 2

    def test_get_by_id(self):
        """Test getting item by ID."""
        queue = ReviewQueue()
        item = ReviewItem(prompt="test", response="response")
        queue.add(item)

        retrieved = queue.get_by_id(item.item_id)
        assert retrieved is not None
        assert retrieved.prompt == "test"

    def test_remove(self):
        """Test removing items."""
        queue = ReviewQueue()
        item = ReviewItem(prompt="test", response="response")
        queue.add(item)

        removed = queue.remove(item.item_id)
        assert removed is not None
        assert len(queue) == 0

    def test_get_pending(self):
        """Test getting pending items."""
        queue = ReviewQueue()
        queue.add(ReviewItem(prompt="1", response="1"))
        queue.add(ReviewItem(prompt="2", response="2"))

        pending = queue.get_pending()
        assert len(pending) == 2

    def test_get_by_status(self):
        """Test getting items by status."""
        queue = ReviewQueue()
        item1 = ReviewItem(prompt="1", response="1")
        item2 = ReviewItem(prompt="2", response="2", status=ReviewStatus.APPROVED)

        queue.add(item1)
        queue.add(item2)

        approved = queue.get_by_status(ReviewStatus.APPROVED)
        assert len(approved) == 1

    def test_stats(self):
        """Test queue statistics."""
        queue = ReviewQueue()
        queue.add(ReviewItem(status=ReviewStatus.PENDING))
        queue.add(ReviewItem(status=ReviewStatus.APPROVED))
        queue.add(ReviewItem(status=ReviewStatus.APPROVED))

        stats = queue.stats()
        assert stats["pending"] == 1
        assert stats["approved"] == 2


class TestPriorityReviewQueue:
    """Tests for PriorityReviewQueue."""

    def test_priority_ordering(self):
        """Test items are returned in priority order."""
        queue = PriorityReviewQueue()

        low = ReviewItem(prompt="low", response="low", priority=Priority.LOW)
        high = ReviewItem(prompt="high", response="high", priority=Priority.HIGH)
        critical = ReviewItem(prompt="critical", response="critical", priority=Priority.CRITICAL)

        queue.add(low)
        queue.add(critical)
        queue.add(high)

        first = queue.get_next()
        assert first.priority == Priority.CRITICAL

        second = queue.get_next()
        assert second.priority == Priority.HIGH

        third = queue.get_next()
        assert third.priority == Priority.LOW


# =============================================================================
# Test CallbackInputHandler
# =============================================================================


class TestCallbackInputHandler:
    """Tests for CallbackInputHandler."""

    def test_approval_callback(self):
        """Test approval via callback."""
        handler = CallbackInputHandler(approval_callback=lambda item: (True, "Looks good"))
        item = ReviewItem(prompt="test", response="response")
        approved, comment = handler.get_approval(item)
        assert approved is True
        assert comment == "Looks good"

    def test_default_approval(self):
        """Test default approval when no callback."""
        handler = CallbackInputHandler()
        item = ReviewItem(prompt="test", response="response")
        approved, comment = handler.get_approval(item)
        assert approved is True  # Auto-approve
        assert comment is None

    def test_feedback_callback(self):
        """Test feedback via callback."""
        handler = CallbackInputHandler(feedback_callback=lambda item: Feedback(rating=0.8))
        item = ReviewItem(prompt="test", response="response")
        feedback = handler.get_feedback(item)
        assert feedback.rating == 0.8

    def test_edit_callback(self):
        """Test edit via callback."""
        handler = CallbackInputHandler(edit_callback=lambda item: "edited content")
        item = ReviewItem(prompt="test", response="original")
        edited = handler.get_edit(item)
        assert edited == "edited content"

    def test_default_edit(self):
        """Test default edit keeps original."""
        handler = CallbackInputHandler()
        item = ReviewItem(prompt="test", response="original")
        edited = handler.get_edit(item)
        assert edited == "original"


# =============================================================================
# Test HITLSession
# =============================================================================


class TestHITLSession:
    """Tests for HITLSession."""

    def test_initialization(self):
        """Test session initialization."""
        model = Mock()
        session = HITLSession(model)
        assert session.model == model
        assert isinstance(session.session_id, str)
        assert session.history == []

    def test_generate_and_review_approved(self):
        """Test generate and review with approval."""
        model = Mock()
        model.generate.return_value = "test response"

        handler = CallbackInputHandler(approval_callback=lambda item: (True, "Good"))
        session = HITLSession(model, input_handler=handler)

        response, item = session.generate_and_review("test prompt")

        assert response == "test response"
        assert item.status == ReviewStatus.APPROVED
        assert len(item.feedback) == 1
        assert item.feedback[0].content == "Good"

    def test_generate_and_review_rejected(self):
        """Test generate and review with rejection."""
        model = Mock()
        model.generate.return_value = "test response"

        handler = CallbackInputHandler(approval_callback=lambda item: (False, "Needs work"))
        session = HITLSession(model, input_handler=handler)

        response, item = session.generate_and_review("test prompt")

        assert item.status == ReviewStatus.REJECTED
        assert len(item.feedback) == 1

    def test_generate_without_approval(self):
        """Test generate without requiring approval."""
        model = Mock()
        model.generate.return_value = "test response"
        session = HITLSession(model)

        response, item = session.generate_and_review("test", require_approval=False)

        assert response == "test response"
        assert item.status == ReviewStatus.APPROVED

    def test_collect_feedback(self):
        """Test feedback collection."""
        model = Mock()
        model.generate.return_value = "response"

        handler = CallbackInputHandler(
            feedback_callback=lambda item: Feedback(rating=0.9, content="Great")
        )
        session = HITLSession(model, input_handler=handler)

        response, feedback = session.collect_feedback("prompt")

        assert response == "response"
        assert feedback.rating == 0.9
        assert feedback.content == "Great"

    def test_edit_response(self):
        """Test response editing."""
        model = Mock()
        model.generate.return_value = "original"

        handler = CallbackInputHandler(edit_callback=lambda item: "edited")
        session = HITLSession(model, input_handler=handler)

        original, edited = session.edit_response("prompt")

        assert original == "original"
        assert edited == "edited"
        assert session.history[0].status == ReviewStatus.EDITED

    def test_edit_response_unchanged(self):
        """Test edit that keeps original."""
        model = Mock()
        model.generate.return_value = "original"

        handler = CallbackInputHandler(edit_callback=lambda item: "original")
        session = HITLSession(model, input_handler=handler)

        original, edited = session.edit_response("prompt")

        assert original == edited
        assert session.history[0].status == ReviewStatus.APPROVED

    def test_history(self):
        """Test session history tracking."""
        model = Mock()
        model.generate.return_value = "response"
        session = HITLSession(model)

        session.generate_and_review("prompt1", require_approval=False)
        session.generate_and_review("prompt2", require_approval=False)

        assert len(session.history) == 2

    def test_statistics(self):
        """Test session statistics."""
        model = Mock()
        model.generate.return_value = "response"

        # Create handler that approves first, rejects second
        call_count = [0]

        def approval_callback(item):
            call_count[0] += 1
            return (call_count[0] == 1, None)

        handler = CallbackInputHandler(approval_callback=approval_callback)
        session = HITLSession(model, input_handler=handler)

        session.generate_and_review("prompt1")
        session.generate_and_review("prompt2")

        stats = session.get_statistics()
        assert stats["total"] == 2
        assert stats["approved"] == 1
        assert stats["rejected"] == 1
        assert stats["approval_rate"] == 0.5

    def test_export_history(self):
        """Test history export."""
        model = Mock()
        model.generate.return_value = "response"
        session = HITLSession(model)

        session.generate_and_review("prompt", require_approval=False)

        export = session.export_history()
        assert len(export) == 1
        assert export[0]["prompt"] == "prompt"


# =============================================================================
# Test InteractiveSession
# =============================================================================


class TestInteractiveSession:
    """Tests for InteractiveSession."""

    def test_event_callbacks(self):
        """Test event callback registration and emission."""
        model = Mock()
        model.generate.return_value = "response"

        session = InteractiveSession(model)

        events = []
        session.on("on_generate", lambda item: events.append("generate"))
        session.on("on_approve", lambda item: events.append("approve"))

        handler = CallbackInputHandler(approval_callback=lambda item: (True, None))
        session.input_handler = handler

        session.generate_and_review("prompt")

        assert "generate" in events
        assert "approve" in events

    def test_reject_callback(self):
        """Test reject event callback."""
        model = Mock()
        model.generate.return_value = "response"

        session = InteractiveSession(model)

        events = []
        session.on("on_reject", lambda item: events.append("reject"))

        handler = CallbackInputHandler(approval_callback=lambda item: (False, None))
        session.input_handler = handler

        session.generate_and_review("prompt")

        assert "reject" in events


# =============================================================================
# Test ApprovalWorkflow
# =============================================================================


class TestApprovalWorkflow:
    """Tests for ApprovalWorkflow."""

    def test_auto_approval_high_confidence(self):
        """Test auto-approval when confidence is high."""
        model = Mock()
        model.generate.return_value = "response"

        workflow = ApprovalWorkflow(
            model,
            auto_approve_threshold=0.8,
            confidence_func=lambda p, r: 0.95,
        )

        response, approved, confidence = workflow.generate_with_approval("prompt")

        assert approved is True
        assert confidence == 0.95
        assert workflow.stats["auto_approved"] == 1

    def test_manual_approval_low_confidence(self):
        """Test manual approval when confidence is low."""
        model = Mock()
        model.generate.return_value = "response"

        handler = CallbackInputHandler(approval_callback=lambda item: (True, None))

        workflow = ApprovalWorkflow(
            model,
            auto_approve_threshold=0.8,
            confidence_func=lambda p, r: 0.5,
            input_handler=handler,
        )

        response, approved, confidence = workflow.generate_with_approval("prompt")

        assert approved is True
        assert confidence == 0.5
        assert workflow.stats["manual_approved"] == 1

    def test_rejection(self):
        """Test rejection in workflow."""
        model = Mock()
        model.generate.return_value = "response"

        handler = CallbackInputHandler(approval_callback=lambda item: (False, None))

        workflow = ApprovalWorkflow(
            model,
            auto_approve_threshold=0.9,
            confidence_func=lambda p, r: 0.5,
            input_handler=handler,
        )

        response, approved, confidence = workflow.generate_with_approval("prompt")

        assert approved is False
        assert workflow.stats["rejected"] == 1

    def test_default_confidence(self):
        """Test default confidence when no function provided."""
        model = Mock()
        model.generate.return_value = "response"

        workflow = ApprovalWorkflow(model, auto_approve_threshold=0.4)

        response, approved, confidence = workflow.generate_with_approval("prompt")

        assert confidence == 0.5  # Default
        assert approved is True  # 0.5 >= 0.4


# =============================================================================
# Test ReviewWorkflow
# =============================================================================


class TestReviewWorkflow:
    """Tests for ReviewWorkflow."""

    def test_add_for_review(self):
        """Test adding items for review."""
        workflow = ReviewWorkflow()

        item = workflow.add_for_review("prompt", "response", Priority.HIGH)

        assert item.prompt == "prompt"
        assert item.response == "response"
        assert item.priority == Priority.HIGH
        assert workflow.queue.pending_count == 1

    def test_get_batch(self):
        """Test getting batch of items."""
        workflow = ReviewWorkflow(batch_size=2)

        workflow.add_for_review("p1", "r1")
        workflow.add_for_review("p2", "r2")
        workflow.add_for_review("p3", "r3")

        batch = workflow.get_batch()
        assert len(batch) == 2

    def test_submit_reviews(self):
        """Test submitting reviews."""
        workflow = ReviewWorkflow()

        item1 = workflow.add_for_review("p1", "r1")
        item2 = workflow.add_for_review("p2", "r2")

        reviews = [
            (item1.item_id, ReviewStatus.APPROVED, Feedback(content="Good")),
            (item2.item_id, ReviewStatus.REJECTED, None),
        ]

        processed = workflow.submit_reviews(reviews)

        assert processed == 2
        assert workflow.queue.get_by_id(item1.item_id).status == ReviewStatus.APPROVED
        assert workflow.queue.get_by_id(item2.item_id).status == ReviewStatus.REJECTED


# =============================================================================
# Test AnnotationWorkflow
# =============================================================================


class TestAnnotationWorkflow:
    """Tests for AnnotationWorkflow."""

    def test_add_for_annotation(self):
        """Test adding text for annotation."""
        workflow = AnnotationWorkflow(labels=["positive", "negative", "neutral"])

        item_id = workflow.add_for_annotation("Test text")
        assert isinstance(item_id, str)

    def test_annotate(self):
        """Test adding annotation."""
        workflow = AnnotationWorkflow(labels=["positive", "negative"])

        item_id = workflow.add_for_annotation("Great product!")
        annotation = workflow.annotate(item_id, "positive", annotator_id="ann1")

        assert annotation is not None
        assert annotation.label == "positive"

    def test_invalid_label(self):
        """Test annotation with invalid label."""
        workflow = AnnotationWorkflow(labels=["positive", "negative"])

        item_id = workflow.add_for_annotation("Test")
        annotation = workflow.annotate(item_id, "invalid")

        assert annotation is None

    def test_span_annotation(self):
        """Test span annotation with offsets."""
        workflow = AnnotationWorkflow(labels=["highlight"])

        item_id = workflow.add_for_annotation("This is important text here")
        annotation = workflow.annotate(item_id, "highlight", start_offset=8, end_offset=17)

        assert annotation is not None
        assert annotation.text == "important"

    def test_multi_label_disabled(self):
        """Test multi-label constraint."""
        workflow = AnnotationWorkflow(labels=["a", "b"], multi_label=False)

        item_id = workflow.add_for_annotation("Test")
        workflow.annotate(item_id, "a")
        duplicate = workflow.annotate(item_id, "a")

        assert duplicate is None  # Can't add same label twice

    def test_multi_label_enabled(self):
        """Test multi-label mode."""
        workflow = AnnotationWorkflow(labels=["a", "b"], multi_label=True)

        item_id = workflow.add_for_annotation("Test")
        workflow.annotate(item_id, "a")
        workflow.annotate(item_id, "b")

        annotations = workflow.get_annotations(item_id)
        assert len(annotations) == 2

    def test_export(self):
        """Test export annotations."""
        workflow = AnnotationWorkflow(labels=["pos", "neg"])

        item_id = workflow.add_for_annotation("Text")
        workflow.annotate(item_id, "pos")

        export = workflow.export()
        assert len(export) == 1
        assert export[0]["item_id"] == item_id


# =============================================================================
# Test HumanValidator
# =============================================================================


class TestHumanValidator:
    """Tests for HumanValidator."""

    def test_validate_with_function(self):
        """Test validation with custom function."""
        validator = HumanValidator(validation_func=lambda p, r: len(r) > 5)

        is_valid, feedback = validator.validate("prompt", "long response")
        assert is_valid is True

        is_valid, feedback = validator.validate("prompt", "short")
        assert is_valid is False

    def test_validate_with_handler(self):
        """Test validation with input handler."""
        handler = CallbackInputHandler(approval_callback=lambda item: (True, "Valid"))
        validator = HumanValidator(input_handler=handler)

        is_valid, feedback = validator.validate("prompt", "response")
        assert is_valid is True
        assert feedback == "Valid"

    def test_validation_history(self):
        """Test validation history tracking."""
        validator = HumanValidator(validation_func=lambda p, r: True)

        validator.validate("p1", "r1")
        validator.validate("p2", "r2")

        assert len(validator.validation_history) == 2

    def test_accuracy(self):
        """Test accuracy calculation."""
        validator = HumanValidator(validation_func=lambda p, r: r == "valid")

        validator.validate("p", "valid")
        validator.validate("p", "invalid")

        assert validator.accuracy() == 0.5


# =============================================================================
# Test ConsensusValidator
# =============================================================================


class TestConsensusValidator:
    """Tests for ConsensusValidator."""

    def test_create_task(self):
        """Test creating validation task."""
        validator = ConsensusValidator(min_reviewers=3)

        task_id = validator.create_validation_task("prompt", "response")
        assert isinstance(task_id, str)
        assert task_id in validator.get_pending_tasks()

    def test_submit_vote(self):
        """Test submitting votes."""
        validator = ConsensusValidator(min_reviewers=2)

        task_id = validator.create_validation_task("prompt", "response")

        result1 = validator.submit_vote(task_id, True, "reviewer1")
        assert result1 is None  # Not enough votes yet

        result2 = validator.submit_vote(task_id, True, "reviewer2")
        assert result2 is not None
        assert result2["consensus"] is True
        assert result2["is_valid"] is True

    def test_reject_consensus(self):
        """Test reaching rejection consensus."""
        validator = ConsensusValidator(min_reviewers=2, consensus_threshold=0.66)

        task_id = validator.create_validation_task("prompt", "response")

        validator.submit_vote(task_id, False, "reviewer1")
        result = validator.submit_vote(task_id, False, "reviewer2")

        assert result is not None
        assert result["is_valid"] is False

    def test_duplicate_vote(self):
        """Test duplicate vote prevention."""
        validator = ConsensusValidator(min_reviewers=2)

        task_id = validator.create_validation_task("prompt", "response")

        validator.submit_vote(task_id, True, "reviewer1")
        result = validator.submit_vote(task_id, True, "reviewer1")  # Duplicate

        assert result is None  # Should be ignored


# =============================================================================
# Test FeedbackCollector
# =============================================================================


class TestFeedbackCollector:
    """Tests for FeedbackCollector."""

    def test_add_and_get_feedback(self):
        """Test adding and getting feedback."""
        collector = FeedbackCollector()

        feedback = Feedback(content="Good", rating=0.8)
        collector.add_feedback("item1", feedback)

        retrieved = collector.get_feedback("item1")
        assert len(retrieved) == 1
        assert retrieved[0].rating == 0.8

    def test_aggregate_ratings(self):
        """Test rating aggregation."""
        collector = FeedbackCollector()

        collector.add_feedback("item1", Feedback(rating=0.8))
        collector.add_feedback("item1", Feedback(rating=0.6))

        avg = collector.aggregate_ratings("item1")
        assert avg == 0.7

    def test_aggregate_ratings_empty(self):
        """Test aggregation with no ratings."""
        collector = FeedbackCollector()
        assert collector.aggregate_ratings("nonexistent") is None

    def test_consensus_type(self):
        """Test getting consensus feedback type."""
        collector = FeedbackCollector()

        collector.add_feedback("item1", Feedback(feedback_type=FeedbackType.APPROVE))
        collector.add_feedback("item1", Feedback(feedback_type=FeedbackType.APPROVE))
        collector.add_feedback("item1", Feedback(feedback_type=FeedbackType.REJECT))

        consensus = collector.get_consensus_type("item1")
        assert consensus == FeedbackType.APPROVE

    def test_export(self):
        """Test feedback export."""
        collector = FeedbackCollector()
        collector.add_feedback("item1", Feedback(content="test"))

        export = collector.export()
        assert "item1" in export
        assert len(export["item1"]) == 1


# =============================================================================
# Test AnnotationCollector
# =============================================================================


class TestAnnotationCollector:
    """Tests for AnnotationCollector."""

    def test_add_annotation(self):
        """Test adding annotation."""
        collector = AnnotationCollector(labels=["a", "b"])

        annotation = Annotation(label="a")
        assert collector.add_annotation("item1", annotation) is True

        retrieved = collector.get_annotations("item1")
        assert len(retrieved) == 1

    def test_invalid_label(self):
        """Test adding annotation with invalid label."""
        collector = AnnotationCollector(labels=["a", "b"])

        annotation = Annotation(label="invalid")
        assert collector.add_annotation("item1", annotation) is False

    def test_agreement_identical(self):
        """Test agreement with identical labels."""
        collector = AnnotationCollector(labels=["a", "b"])

        collector.add_annotation("item1", Annotation(label="a"))
        collector.add_annotation("item1", Annotation(label="a"))
        collector.add_annotation("item1", Annotation(label="a"))

        agreement = collector.compute_agreement("item1")
        assert agreement == 1.0

    def test_agreement_mixed(self):
        """Test agreement with mixed labels."""
        collector = AnnotationCollector(labels=["a", "b"])

        collector.add_annotation("item1", Annotation(label="a"))
        collector.add_annotation("item1", Annotation(label="b"))

        agreement = collector.compute_agreement("item1")
        assert agreement == 0.0  # No matching pairs

    def test_majority_label(self):
        """Test getting majority label."""
        collector = AnnotationCollector(labels=["a", "b"])

        collector.add_annotation("item1", Annotation(label="a"))
        collector.add_annotation("item1", Annotation(label="a"))
        collector.add_annotation("item1", Annotation(label="b"))

        majority = collector.get_majority_label("item1")
        assert majority == "a"

    def test_export(self):
        """Test export with agreement."""
        collector = AnnotationCollector(labels=["a"])

        collector.add_annotation("item1", Annotation(label="a"))
        collector.add_annotation("item1", Annotation(label="a"))

        export = collector.export()
        assert "item1" in export
        assert export["item1"]["agreement"] == 1.0


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_hitl_session(self):
        """Test session creation."""
        model = Mock()
        session = create_hitl_session(model, auto_approve_threshold=0.85)

        assert session.model == model
        assert session.config.auto_approve_threshold == 0.85

    def test_quick_review(self):
        """Test quick review function."""
        approved, comment = quick_review("prompt", "response", lambda item: (True, "Good"))

        assert approved is True
        assert comment == "Good"

    def test_collect_feedback(self):
        """Test feedback collection function."""
        items = [
            ("prompt1", "response1"),
            ("prompt2", "response2"),
        ]

        feedback_list = collect_feedback(items, lambda item: Feedback(rating=0.8))

        assert len(feedback_list) == 2
        assert all(f.rating == 0.8 for f in feedback_list)


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_queue_concurrent_access(self):
        """Test concurrent queue access."""
        queue = ReviewQueue()
        results = []

        def add_items():
            for i in range(50):
                queue.add(ReviewItem(prompt=f"p{i}", response=f"r{i}"))

        def get_items():
            for _ in range(50):
                item = queue.get_next()
                if item:
                    results.append(item)

        threads = [
            threading.Thread(target=add_items),
            threading.Thread(target=add_items),
            threading.Thread(target=get_items),
            threading.Thread(target=get_items),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have processed some items without errors
        assert len(results) > 0

    def test_session_concurrent_generate(self):
        """Test concurrent session usage."""
        model = Mock()
        model.generate.return_value = "response"

        session = HITLSession(model)
        results = []

        def generate():
            for i in range(20):
                response, item = session.generate_and_review(f"prompt{i}", require_approval=False)
                results.append(item)

        threads = [
            threading.Thread(target=generate),
            threading.Thread(target=generate),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(session.history) == 40

    def test_feedback_collector_concurrent(self):
        """Test concurrent feedback collection."""
        collector = FeedbackCollector()

        def add_feedback():
            for _i in range(50):
                collector.add_feedback("item1", Feedback(rating=0.5))

        threads = [
            threading.Thread(target=add_feedback),
            threading.Thread(target=add_feedback),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        feedback = collector.get_feedback("item1")
        assert len(feedback) == 100


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_queue_get_next(self):
        """Test getting from empty queue."""
        queue = ReviewQueue()
        assert queue.get_next() is None

    def test_get_nonexistent_item(self):
        """Test getting nonexistent item."""
        queue = ReviewQueue()
        assert queue.get_by_id("nonexistent") is None

    def test_empty_session_statistics(self):
        """Test statistics on empty session."""
        model = Mock()
        session = HITLSession(model)

        stats = session.get_statistics()
        assert stats["total"] == 0

    def test_validation_empty_history(self):
        """Test validator with no history."""
        validator = HumanValidator()
        assert validator.accuracy() == 0.0

    def test_annotation_no_annotations(self):
        """Test agreement with insufficient annotations."""
        collector = AnnotationCollector(labels=["a"])
        collector.add_annotation("item1", Annotation(label="a"))

        agreement = collector.compute_agreement("item1")
        assert agreement is None  # Need at least 2

    def test_consensus_with_comments(self):
        """Test consensus validator preserves comments."""
        validator = ConsensusValidator(min_reviewers=2)

        task_id = validator.create_validation_task("prompt", "response")

        validator.submit_vote(task_id, True, "reviewer1", "Great response")
        result = validator.submit_vote(task_id, True, "reviewer2", "Agreed")

        assert result is not None
        assert result["votes"][0]["comment"] == "Great response"
        assert result["votes"][1]["comment"] == "Agreed"

    def test_skipped_status(self):
        """Test skipped review status."""
        model = Mock()
        model.generate.return_value = "response"

        # Return None for approval (skip)
        handler = CallbackInputHandler(
            approval_callback=lambda item: (None, "Skipping")  # type: ignore
        )
        session = HITLSession(model, input_handler=handler)

        response, item = session.generate_and_review("prompt")
        assert item.status == ReviewStatus.SKIPPED
