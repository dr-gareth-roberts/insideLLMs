"""Human-in-the-Loop (HITL) module for interactive model evaluation and feedback.

This module provides comprehensive tools for incorporating human feedback into LLM
workflows, enabling quality control, validation, and iterative improvement of model
outputs through human oversight.

Key Features:
    - Interactive approval workflows for gated model outputs
    - Human validation and correction capabilities
    - Feedback collection and aggregation from multiple reviewers
    - Priority-based review queues for efficient workflow management
    - Annotation interfaces for labeling and span marking
    - Consensus validation with multiple reviewer support

Core Components:
    Sessions:
        - HITLSession: Basic interactive session for model evaluation
        - InteractiveSession: Extended session with event callbacks

    Workflows:
        - ApprovalWorkflow: Approval-based interactions with auto-approve thresholds
        - ReviewWorkflow: Batched review of model outputs
        - AnnotationWorkflow: Structured annotation collection

    Queues:
        - ReviewQueue: FIFO queue for review items
        - PriorityReviewQueue: Priority-ordered review queue

    Validators:
        - HumanValidator: Single-reviewer validation
        - ConsensusValidator: Multi-reviewer consensus validation

    Collectors:
        - FeedbackCollector: Aggregates feedback from multiple sources
        - AnnotationCollector: Tracks annotations with inter-annotator agreement

Example: Basic HITL Session
    >>> from insideLLMs.agents.hitl import HITLSession, CallbackInputHandler
    >>> from insideLLMs import DummyModel
    >>>
    >>> # Create a model and auto-approve handler
    >>> model = DummyModel()
    >>> handler = CallbackInputHandler(
    ...     approval_callback=lambda item: (True, "Looks good")
    ... )
    >>> session = HITLSession(model, input_handler=handler)
    >>>
    >>> # Generate and review
    >>> response, item = session.generate_and_review("Summarize this text")
    >>> print(f"Status: {item.status}")  # ReviewStatus.APPROVED

Example: Approval Workflow with Confidence Threshold
    >>> from insideLLMs.agents.hitl import ApprovalWorkflow
    >>>
    >>> # Define a confidence function
    >>> def confidence_scorer(prompt, response):
    ...     return 0.95 if len(response) > 50 else 0.5
    >>>
    >>> workflow = ApprovalWorkflow(
    ...     model=model,
    ...     auto_approve_threshold=0.9,
    ...     confidence_func=confidence_scorer
    ... )
    >>> response, approved, confidence = workflow.generate_with_approval("Write a story")
    >>> # High confidence responses are auto-approved

Example: Priority Review Queue
    >>> from insideLLMs.agents.hitl import PriorityReviewQueue, ReviewItem, Priority
    >>>
    >>> queue = PriorityReviewQueue()
    >>> queue.add(ReviewItem(prompt="Low priority", response="...", priority=Priority.LOW))
    >>> queue.add(ReviewItem(prompt="Critical!", response="...", priority=Priority.CRITICAL))
    >>>
    >>> # Critical items are retrieved first
    >>> next_item = queue.get_next()
    >>> print(next_item.priority)  # Priority.CRITICAL

Example: Consensus Validation
    >>> from insideLLMs.agents.hitl import ConsensusValidator
    >>>
    >>> validator = ConsensusValidator(min_reviewers=3, consensus_threshold=0.66)
    >>> task_id = validator.create_validation_task("prompt", "response")
    >>>
    >>> # Multiple reviewers submit votes
    >>> validator.submit_vote(task_id, is_valid=True, reviewer_id="reviewer1")
    >>> validator.submit_vote(task_id, is_valid=True, reviewer_id="reviewer2")
    >>> result = validator.submit_vote(task_id, is_valid=True, reviewer_id="reviewer3")
    >>> # Consensus reached with 3/3 valid votes
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Empty, PriorityQueue
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
)

__all__ = [
    # Core types
    "FeedbackType",
    "ReviewStatus",
    "Priority",
    "Feedback",
    "ReviewItem",
    "Annotation",
    # Sessions
    "HITLSession",
    "InteractiveSession",
    # Workflows
    "ApprovalWorkflow",
    "ReviewWorkflow",
    "AnnotationWorkflow",
    # Queues
    "ReviewQueue",
    "PriorityReviewQueue",
    # Validators
    "HumanValidator",
    "ConsensusValidator",
    # Collectors
    "FeedbackCollector",
    "AnnotationCollector",
    # Config
    "HITLConfig",
    # Convenience
    "create_hitl_session",
    "quick_review",
    "collect_feedback",
]


class FeedbackType(str, Enum):
    """Types of human feedback that can be provided on model outputs.

    This enumeration defines the various ways humans can interact with and
    provide feedback on LLM-generated content in HITL workflows.

    Attributes:
        APPROVE: Indicates the output meets quality standards and is accepted.
        REJECT: Indicates the output does not meet standards and is declined.
        EDIT: Indicates the output was modified by the reviewer.
        FLAG: Marks the output for special attention or further review.
        SKIP: The reviewer chose not to evaluate this output.
        RATING: A numerical quality rating was provided.
        COMMENT: A textual comment or note was added.
        CORRECTION: A specific correction was made to fix an error.

    Examples:
        Basic usage with Feedback:
            >>> from insideLLMs.agents.hitl import FeedbackType, Feedback
            >>> feedback = Feedback(
            ...     feedback_type=FeedbackType.APPROVE,
            ...     content="This response is accurate and helpful"
            ... )
            >>> print(feedback.feedback_type.value)
            'approve'

        Using feedback type for conditional logic:
            >>> feedback_type = FeedbackType.REJECT
            >>> if feedback_type == FeedbackType.REJECT:
            ...     print("Output needs improvement")
            Output needs improvement

        Checking feedback type from string:
            >>> FeedbackType("edit") == FeedbackType.EDIT
            True

        Iterating over feedback types:
            >>> approval_types = [FeedbackType.APPROVE, FeedbackType.EDIT]
            >>> FeedbackType.APPROVE in approval_types
            True
    """

    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    FLAG = "flag"
    SKIP = "skip"
    RATING = "rating"
    COMMENT = "comment"
    CORRECTION = "correction"


class ReviewStatus(str, Enum):
    """Status of items in the human review workflow.

    This enumeration tracks the lifecycle state of review items as they
    progress through the HITL review process.

    Attributes:
        PENDING: Item is waiting in queue for review.
        IN_PROGRESS: Item has been picked up and is currently being reviewed.
        APPROVED: Item has been reviewed and accepted.
        REJECTED: Item has been reviewed and declined.
        EDITED: Item was modified during review.
        FLAGGED: Item was marked for special attention.
        SKIPPED: Reviewer chose to skip this item.
        EXPIRED: Item's review window has passed without action.

    Examples:
        Creating a review item with status:
            >>> from insideLLMs.agents.hitl import ReviewItem, ReviewStatus
            >>> item = ReviewItem(
            ...     prompt="Translate to French",
            ...     response="Bonjour",
            ...     status=ReviewStatus.PENDING
            ... )
            >>> print(item.status)
            ReviewStatus.PENDING

        Checking item status:
            >>> item = ReviewItem(prompt="test", response="result")
            >>> if item.status == ReviewStatus.PENDING:
            ...     print("Item awaiting review")
            Item awaiting review

        Filtering items by status:
            >>> queue = ReviewQueue()
            >>> approved_items = queue.get_by_status(ReviewStatus.APPROVED)
            >>> print(f"Found {len(approved_items)} approved items")

        Status transitions in workflow:
            >>> item = ReviewItem(prompt="test", response="result")
            >>> item.status = ReviewStatus.IN_PROGRESS  # Picked up for review
            >>> item.status = ReviewStatus.APPROVED     # Review complete
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
    FLAGGED = "flagged"
    SKIPPED = "skipped"
    EXPIRED = "expired"


class Priority(int, Enum):
    """Priority levels for ordering review items in queues.

    This enumeration defines priority levels for review items, used by
    PriorityReviewQueue to determine which items should be reviewed first.
    Lower numeric values indicate higher priority.

    Attributes:
        CRITICAL: Highest priority (1). For urgent items requiring immediate review.
        HIGH: High priority (2). Important items that should be reviewed soon.
        MEDIUM: Default priority (3). Standard review items.
        LOW: Lower priority (4). Can wait until higher priority items are done.
        BACKGROUND: Lowest priority (5). Review when queue is otherwise empty.

    Examples:
        Setting priority on review items:
            >>> from insideLLMs.agents.hitl import ReviewItem, Priority
            >>> urgent_item = ReviewItem(
            ...     prompt="Safety check",
            ...     response="...",
            ...     priority=Priority.CRITICAL
            ... )
            >>> normal_item = ReviewItem(
            ...     prompt="General query",
            ...     response="...",
            ...     priority=Priority.MEDIUM
            ... )

        Using with PriorityReviewQueue:
            >>> from insideLLMs.agents.hitl import PriorityReviewQueue, ReviewItem, Priority
            >>> queue = PriorityReviewQueue()
            >>> queue.add(ReviewItem(prompt="p1", response="r1", priority=Priority.LOW))
            >>> queue.add(ReviewItem(prompt="p2", response="r2", priority=Priority.CRITICAL))
            >>> next_item = queue.get_next()
            >>> print(next_item.priority)  # CRITICAL comes first
            Priority.CRITICAL

        Comparing priorities:
            >>> Priority.CRITICAL < Priority.LOW  # Lower value = higher priority
            True
            >>> Priority.HIGH.value < Priority.MEDIUM.value
            True

        Dynamic priority assignment:
            >>> def assign_priority(response_length):
            ...     if response_length > 1000:
            ...         return Priority.HIGH  # Long responses need careful review
            ...     return Priority.MEDIUM
            >>> priority = assign_priority(1500)
            >>> print(priority)
            Priority.HIGH
    """

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class Feedback:
    """Represents human feedback on a model output.

    This dataclass captures all aspects of human feedback including the type of
    feedback, any textual content, numerical ratings, and edited versions of
    the original output. It supports serialization for storage and transmission.

    Attributes:
        feedback_id: Unique identifier for this feedback (auto-generated UUID).
        feedback_type: The type of feedback (approve, reject, edit, etc.).
        content: Textual content of the feedback (comments, notes).
        rating: Optional numerical rating from 0.0 to 1.0.
        edited_content: If feedback_type is EDIT, contains the modified output.
        metadata: Additional key-value pairs for custom data.
        timestamp: When the feedback was created.
        reviewer_id: Identifier of the person who provided the feedback.

    Examples:
        Creating approval feedback:
            >>> from insideLLMs.agents.hitl import Feedback, FeedbackType
            >>> feedback = Feedback(
            ...     feedback_type=FeedbackType.APPROVE,
            ...     content="Response is accurate and well-formatted",
            ...     reviewer_id="reviewer_001"
            ... )
            >>> print(feedback.feedback_type)
            FeedbackType.APPROVE

        Creating feedback with rating:
            >>> feedback = Feedback(
            ...     feedback_type=FeedbackType.RATING,
            ...     content="Good but could be more concise",
            ...     rating=0.8  # 80% quality score
            ... )
            >>> print(f"Rating: {feedback.rating * 100}%")
            Rating: 80.0%

        Creating edit feedback:
            >>> feedback = Feedback(
            ...     feedback_type=FeedbackType.EDIT,
            ...     content="Fixed grammatical errors",
            ...     edited_content="The corrected response text here..."
            ... )
            >>> print(feedback.edited_content is not None)
            True

        Serializing and deserializing feedback:
            >>> feedback = Feedback(
            ...     feedback_type=FeedbackType.COMMENT,
            ...     content="Needs more detail",
            ...     metadata={"category": "completeness"}
            ... )
            >>> data = feedback.to_dict()
            >>> restored = Feedback.from_dict(data)
            >>> print(restored.content)
            Needs more detail
    """

    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    feedback_type: FeedbackType = FeedbackType.COMMENT
    content: str = ""
    rating: Optional[float] = None  # 0.0 to 1.0
    edited_content: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    reviewer_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the Feedback instance to a dictionary for serialization.

        Converts all fields to JSON-serializable types, including converting
        the timestamp to ISO format and the feedback_type to its string value.

        Returns:
            A dictionary containing all feedback data in serializable format.

        Examples:
            Basic serialization:
                >>> feedback = Feedback(
                ...     feedback_type=FeedbackType.APPROVE,
                ...     content="Looks good"
                ... )
                >>> data = feedback.to_dict()
                >>> print(data["feedback_type"])
                'approve'

            Serializing for JSON storage:
                >>> import json
                >>> feedback = Feedback(content="Test feedback", rating=0.9)
                >>> json_str = json.dumps(feedback.to_dict())
                >>> print("timestamp" in json_str)
                True

            Preserving metadata:
                >>> feedback = Feedback(metadata={"source": "api", "version": 2})
                >>> data = feedback.to_dict()
                >>> print(data["metadata"]["source"])
                'api'
        """
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "content": self.content,
            "rating": self.rating,
            "edited_content": self.edited_content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "reviewer_id": self.reviewer_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Feedback":
        """Create a Feedback instance from a dictionary.

        Reconstructs a Feedback object from a dictionary, typically one
        created by to_dict(). Handles missing fields with sensible defaults.

        Args:
            data: Dictionary containing feedback data. Expected keys match
                the Feedback attributes. Missing keys use default values.

        Returns:
            A new Feedback instance populated with the dictionary data.

        Examples:
            Basic deserialization:
                >>> data = {
                ...     "feedback_type": "approve",
                ...     "content": "Well done",
                ...     "rating": 0.95
                ... }
                >>> feedback = Feedback.from_dict(data)
                >>> print(feedback.content)
                Well done

            Round-trip serialization:
                >>> original = Feedback(
                ...     feedback_type=FeedbackType.EDIT,
                ...     content="Fixed typos",
                ...     edited_content="Corrected text"
                ... )
                >>> restored = Feedback.from_dict(original.to_dict())
                >>> print(restored.edited_content)
                Corrected text

            Handling partial data:
                >>> data = {"content": "Minimal feedback"}
                >>> feedback = Feedback.from_dict(data)
                >>> print(feedback.feedback_type)  # Uses default
                FeedbackType.COMMENT

            With timestamp parsing:
                >>> data = {
                ...     "content": "Time-stamped feedback",
                ...     "timestamp": "2024-01-15T10:30:00"
                ... }
                >>> feedback = Feedback.from_dict(data)
                >>> print(feedback.timestamp.year)
                2024
        """
        return cls(
            feedback_id=data.get("feedback_id", str(uuid.uuid4())),
            feedback_type=FeedbackType(data.get("feedback_type", "comment")),
            content=data.get("content", ""),
            rating=data.get("rating"),
            edited_content=data.get("edited_content"),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.now(),
            reviewer_id=data.get("reviewer_id"),
        )


@dataclass
class Annotation:
    """Represents a human annotation on text for labeling and span marking.

    This dataclass supports both document-level annotations (labeling entire texts)
    and span annotations (marking specific portions of text). It's commonly used
    in annotation workflows for training data creation and quality assessment.

    Attributes:
        annotation_id: Unique identifier for this annotation (auto-generated UUID).
        text: The annotated text content (either full text or extracted span).
        label: The label or category assigned to this annotation.
        start_offset: For span annotations, the starting character position.
        end_offset: For span annotations, the ending character position.
        confidence: Annotator's confidence in the annotation (0.0 to 1.0).
        metadata: Additional key-value pairs for custom annotation data.
        annotator_id: Identifier of the person who created the annotation.
        timestamp: When the annotation was created.

    Examples:
        Document-level annotation (sentiment labeling):
            >>> from insideLLMs.agents.hitl import Annotation
            >>> annotation = Annotation(
            ...     text="This product is amazing!",
            ...     label="positive",
            ...     confidence=0.95,
            ...     annotator_id="annotator_001"
            ... )
            >>> print(f"Label: {annotation.label}, Confidence: {annotation.confidence}")
            Label: positive, Confidence: 0.95

        Span annotation (named entity recognition):
            >>> text = "John Smith works at OpenAI in San Francisco."
            >>> annotation = Annotation(
            ...     text="John Smith",
            ...     label="PERSON",
            ...     start_offset=0,
            ...     end_offset=10
            ... )
            >>> print(f"Entity: '{annotation.text}' is a {annotation.label}")
            Entity: 'John Smith' is a PERSON

        Multiple span annotations on same text:
            >>> text = "Apple released the iPhone in California."
            >>> annotations = [
            ...     Annotation(text="Apple", label="ORG", start_offset=0, end_offset=5),
            ...     Annotation(text="iPhone", label="PRODUCT", start_offset=18, end_offset=24),
            ...     Annotation(text="California", label="LOC", start_offset=28, end_offset=38),
            ... ]
            >>> for ann in annotations:
            ...     print(f"{ann.label}: {ann.text}")
            ORG: Apple
            PRODUCT: iPhone
            LOC: California

        Annotation with metadata:
            >>> annotation = Annotation(
            ...     text="The model performed well",
            ...     label="quality_assessment",
            ...     metadata={
            ...         "criteria": "accuracy",
            ...         "task_type": "summarization",
            ...         "review_round": 2
            ...     }
            ... )
            >>> print(annotation.metadata["criteria"])
            accuracy
    """

    annotation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    label: str = ""
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    annotator_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert the Annotation instance to a dictionary for serialization.

        Converts all fields to JSON-serializable types, including converting
        the timestamp to ISO format.

        Returns:
            A dictionary containing all annotation data in serializable format.

        Examples:
            Basic serialization:
                >>> annotation = Annotation(text="Hello", label="greeting")
                >>> data = annotation.to_dict()
                >>> print(data["label"])
                'greeting'

            Serializing span annotation:
                >>> annotation = Annotation(
                ...     text="New York",
                ...     label="LOCATION",
                ...     start_offset=10,
                ...     end_offset=18
                ... )
                >>> data = annotation.to_dict()
                >>> print(f"Span: {data['start_offset']}-{data['end_offset']}")
                Span: 10-18

            For JSON storage:
                >>> import json
                >>> annotation = Annotation(text="test", label="example")
                >>> json_str = json.dumps(annotation.to_dict())
                >>> print("annotation_id" in json_str)
                True
        """
        return {
            "annotation_id": self.annotation_id,
            "text": self.text,
            "label": self.label,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "annotator_id": self.annotator_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReviewItem:
    """An item in the human review queue awaiting evaluation.

    ReviewItem is the central data structure for HITL workflows, containing the
    model's prompt and response along with all associated feedback, annotations,
    and metadata collected during the review process.

    Attributes:
        item_id: Unique identifier for this review item (auto-generated UUID).
        prompt: The original prompt sent to the model.
        response: The model's generated response to be reviewed.
        model_id: Optional identifier of the model that generated the response.
        status: Current review status (pending, approved, rejected, etc.).
        priority: Priority level for queue ordering.
        feedback: List of Feedback objects collected for this item.
        annotations: List of Annotation objects applied to this item.
        metadata: Additional key-value pairs for custom data.
        created_at: When this item was created.
        updated_at: When this item was last modified.
        assigned_to: Reviewer ID if this item is assigned to a specific person.
        expires_at: Optional deadline after which the item expires.

    Examples:
        Creating a review item:
            >>> from insideLLMs.agents.hitl import ReviewItem, Priority, ReviewStatus
            >>> item = ReviewItem(
            ...     prompt="Summarize this article about AI",
            ...     response="AI is transforming industries...",
            ...     model_id="gpt-4",
            ...     priority=Priority.HIGH
            ... )
            >>> print(item.status)
            ReviewStatus.PENDING

        Adding feedback to an item:
            >>> from insideLLMs.agents.hitl import ReviewItem, Feedback, FeedbackType
            >>> item = ReviewItem(prompt="test", response="result")
            >>> item.add_feedback(Feedback(
            ...     feedback_type=FeedbackType.APPROVE,
            ...     content="Accurate summary",
            ...     rating=0.9
            ... ))
            >>> print(len(item.feedback))
            1

        Adding annotations:
            >>> from insideLLMs.agents.hitl import ReviewItem, Annotation
            >>> item = ReviewItem(
            ...     prompt="Extract entities",
            ...     response="John works at Google in NYC"
            ... )
            >>> item.add_annotation(Annotation(
            ...     text="John", label="PERSON", start_offset=0, end_offset=4
            ... ))
            >>> item.add_annotation(Annotation(
            ...     text="Google", label="ORG", start_offset=14, end_offset=20
            ... ))
            >>> print(len(item.annotations))
            2

        Using in a priority queue:
            >>> from insideLLMs.agents.hitl import PriorityReviewQueue, ReviewItem, Priority
            >>> queue = PriorityReviewQueue()
            >>> queue.add(ReviewItem(prompt="p1", response="r1", priority=Priority.LOW))
            >>> queue.add(ReviewItem(prompt="p2", response="r2", priority=Priority.CRITICAL))
            >>> next_item = queue.get_next()
            >>> print(next_item.priority)  # Higher priority first
            Priority.CRITICAL
    """

    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    response: str = ""
    model_id: Optional[str] = None
    status: ReviewStatus = ReviewStatus.PENDING
    priority: Priority = Priority.MEDIUM
    feedback: list[Feedback] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    expires_at: Optional[datetime] = None

    def __lt__(self, other: "ReviewItem") -> bool:
        """Compare review items by priority for queue ordering.

        Items with lower priority values (higher urgency) come first.
        If priorities are equal, older items come first (FIFO within priority).

        Args:
            other: Another ReviewItem to compare against.

        Returns:
            True if this item should come before the other in the queue.

        Examples:
            Priority comparison:
                >>> item1 = ReviewItem(prompt="p1", response="r1", priority=Priority.LOW)
                >>> item2 = ReviewItem(prompt="p2", response="r2", priority=Priority.HIGH)
                >>> item2 < item1  # HIGH priority comes before LOW
                True

            Same priority uses creation time:
                >>> import time
                >>> item1 = ReviewItem(prompt="p1", response="r1", priority=Priority.MEDIUM)
                >>> time.sleep(0.01)
                >>> item2 = ReviewItem(prompt="p2", response="r2", priority=Priority.MEDIUM)
                >>> item1 < item2  # Older item comes first
                True
        """
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at

    def add_feedback(self, feedback: Feedback) -> None:
        """Add feedback to this review item.

        Appends the feedback to the item's feedback list and updates
        the updated_at timestamp.

        Args:
            feedback: The Feedback object to add.

        Examples:
            Adding approval feedback:
                >>> item = ReviewItem(prompt="test", response="result")
                >>> item.add_feedback(Feedback(
                ...     feedback_type=FeedbackType.APPROVE,
                ...     content="Good response"
                ... ))
                >>> print(item.feedback[-1].content)
                Good response

            Multiple reviewers adding feedback:
                >>> item = ReviewItem(prompt="test", response="result")
                >>> item.add_feedback(Feedback(content="Reviewer 1", reviewer_id="r1"))
                >>> item.add_feedback(Feedback(content="Reviewer 2", reviewer_id="r2"))
                >>> print(len(item.feedback))
                2

            Feedback with rating:
                >>> item = ReviewItem(prompt="test", response="result")
                >>> item.add_feedback(Feedback(
                ...     feedback_type=FeedbackType.RATING,
                ...     rating=0.85
                ... ))
                >>> print(item.feedback[-1].rating)
                0.85
        """
        self.feedback.append(feedback)
        self.updated_at = datetime.now()

    def add_annotation(self, annotation: Annotation) -> None:
        """Add an annotation to this review item.

        Appends the annotation to the item's annotations list and updates
        the updated_at timestamp.

        Args:
            annotation: The Annotation object to add.

        Examples:
            Adding a label annotation:
                >>> item = ReviewItem(prompt="classify", response="positive text")
                >>> item.add_annotation(Annotation(
                ...     text="positive text",
                ...     label="sentiment_positive"
                ... ))
                >>> print(item.annotations[-1].label)
                sentiment_positive

            Adding span annotations:
                >>> item = ReviewItem(prompt="NER", response="John lives in Paris")
                >>> item.add_annotation(Annotation(
                ...     text="John", label="PERSON", start_offset=0, end_offset=4
                ... ))
                >>> item.add_annotation(Annotation(
                ...     text="Paris", label="LOCATION", start_offset=14, end_offset=19
                ... ))
                >>> print(len(item.annotations))
                2

            Multi-annotator scenario:
                >>> item = ReviewItem(prompt="classify", response="text")
                >>> item.add_annotation(Annotation(label="A", annotator_id="ann1"))
                >>> item.add_annotation(Annotation(label="A", annotator_id="ann2"))
                >>> print(len(item.annotations))
                2
        """
        self.annotations.append(annotation)
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert the ReviewItem to a dictionary for serialization.

        Converts all fields to JSON-serializable types, including nested
        Feedback and Annotation objects.

        Returns:
            A dictionary containing all review item data in serializable format.

        Examples:
            Basic serialization:
                >>> item = ReviewItem(prompt="test", response="result")
                >>> data = item.to_dict()
                >>> print(data["status"])
                'pending'

            With feedback and annotations:
                >>> item = ReviewItem(prompt="test", response="result")
                >>> item.add_feedback(Feedback(content="good"))
                >>> item.add_annotation(Annotation(label="positive"))
                >>> data = item.to_dict()
                >>> print(len(data["feedback"]))
                1

            For JSON storage:
                >>> import json
                >>> item = ReviewItem(
                ...     prompt="test",
                ...     response="result",
                ...     priority=Priority.HIGH
                ... )
                >>> json_str = json.dumps(item.to_dict())
                >>> print("priority" in json_str)
                True
        """
        return {
            "item_id": self.item_id,
            "prompt": self.prompt,
            "response": self.response,
            "model_id": self.model_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "feedback": [f.to_dict() for f in self.feedback],
            "annotations": [a.to_dict() for a in self.annotations],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "assigned_to": self.assigned_to,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class HITLConfig:
    """Configuration settings for Human-in-the-Loop sessions.

    This dataclass defines the behavior and constraints of HITL sessions,
    controlling aspects like auto-approval thresholds, consensus requirements,
    and reviewer permissions.

    Attributes:
        auto_approve_threshold: Confidence threshold (0.0-1.0) above which
            responses are automatically approved without human review.
        require_consensus: If True, requires multiple reviewers to agree.
        min_reviewers: Minimum number of reviewers required per item.
        timeout_seconds: Maximum time allowed for review (None for no limit).
        allow_skip: Whether reviewers can skip items without reviewing.
        allow_edit: Whether reviewers can edit responses.
        require_comment: Whether feedback must include a comment.
        track_time: Whether to track time spent on reviews.

    Examples:
        Default configuration:
            >>> from insideLLMs.agents.hitl import HITLConfig
            >>> config = HITLConfig()
            >>> print(f"Auto-approve at {config.auto_approve_threshold}")
            Auto-approve at 0.9

        Strict review configuration:
            >>> config = HITLConfig(
            ...     auto_approve_threshold=0.99,  # Rarely auto-approve
            ...     require_consensus=True,
            ...     min_reviewers=3,
            ...     allow_skip=False,
            ...     require_comment=True
            ... )
            >>> print(f"Need {config.min_reviewers} reviewers")
            Need 3 reviewers

        Lenient configuration for low-risk content:
            >>> config = HITLConfig(
            ...     auto_approve_threshold=0.7,
            ...     allow_skip=True,
            ...     allow_edit=True,
            ...     require_comment=False
            ... )
            >>> print(config.require_comment)
            False

        Time-limited review sessions:
            >>> config = HITLConfig(
            ...     timeout_seconds=300.0,  # 5 minutes per item
            ...     track_time=True
            ... )
            >>> print(f"Timeout: {config.timeout_seconds} seconds")
            Timeout: 300.0 seconds
    """

    auto_approve_threshold: float = 0.9
    require_consensus: bool = False
    min_reviewers: int = 1
    timeout_seconds: Optional[float] = None
    allow_skip: bool = True
    allow_edit: bool = True
    require_comment: bool = False
    track_time: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert the HITLConfig to a dictionary for serialization.

        Returns:
            A dictionary containing all configuration settings.

        Examples:
            Basic serialization:
                >>> config = HITLConfig(auto_approve_threshold=0.85)
                >>> data = config.to_dict()
                >>> print(data["auto_approve_threshold"])
                0.85

            For JSON storage:
                >>> import json
                >>> config = HITLConfig(min_reviewers=3, require_consensus=True)
                >>> json_str = json.dumps(config.to_dict())
                >>> print("require_consensus" in json_str)
                True

            Preserving all settings:
                >>> config = HITLConfig(
                ...     timeout_seconds=120.0,
                ...     allow_edit=False
                ... )
                >>> data = config.to_dict()
                >>> print(data["timeout_seconds"], data["allow_edit"])
                120.0 False
        """
        return {
            "auto_approve_threshold": self.auto_approve_threshold,
            "require_consensus": self.require_consensus,
            "min_reviewers": self.min_reviewers,
            "timeout_seconds": self.timeout_seconds,
            "allow_skip": self.allow_skip,
            "allow_edit": self.allow_edit,
            "require_comment": self.require_comment,
            "track_time": self.track_time,
        }


class ReviewQueue:
    """Thread-safe FIFO queue for managing items awaiting human review.

    ReviewQueue provides a simple first-in-first-out queue for review items
    with thread-safe operations. Use PriorityReviewQueue if you need
    priority-based ordering.

    Attributes:
        _items: Internal dictionary mapping item IDs to ReviewItems.
        _pending: List of item IDs awaiting review.
        _lock: Threading lock for thread-safe operations.
        _max_size: Maximum queue capacity (None for unlimited).

    Examples:
        Basic queue operations:
            >>> from insideLLMs.agents.hitl import ReviewQueue, ReviewItem
            >>> queue = ReviewQueue()
            >>> item = ReviewItem(prompt="Test prompt", response="Test response")
            >>> queue.add(item)
            True
            >>> print(len(queue))
            1

        Processing items from queue:
            >>> queue = ReviewQueue()
            >>> queue.add(ReviewItem(prompt="p1", response="r1"))
            >>> queue.add(ReviewItem(prompt="p2", response="r2"))
            >>> item = queue.get_next()
            >>> print(item.prompt)  # First item added
            p1

        Queue with size limit:
            >>> queue = ReviewQueue(max_size=2)
            >>> queue.add(ReviewItem(prompt="p1", response="r1"))
            True
            >>> queue.add(ReviewItem(prompt="p2", response="r2"))
            True
            >>> queue.add(ReviewItem(prompt="p3", response="r3"))  # Queue full
            False

        Getting queue statistics:
            >>> queue = ReviewQueue()
            >>> queue.add(ReviewItem(prompt="p1", response="r1"))
            >>> stats = queue.stats()
            >>> print(stats["pending"])
            1
    """

    def __init__(self, max_size: Optional[int] = None):
        """Initialize review queue.

        Args:
            max_size: Maximum queue size. None for unlimited capacity.
                When the queue is full, add() returns False.

        Examples:
            Unlimited queue:
                >>> queue = ReviewQueue()
                >>> queue._max_size is None
                True

            Limited queue:
                >>> queue = ReviewQueue(max_size=100)
                >>> queue._max_size
                100

            For high-volume processing:
                >>> queue = ReviewQueue(max_size=1000)
                >>> # Prevents memory issues with large backlogs
        """
        self._items: dict[str, ReviewItem] = {}
        self._pending: list[str] = []
        self._lock = threading.Lock()
        self._max_size = max_size

    def add(self, item: ReviewItem) -> bool:
        """Add item to the review queue.

        Thread-safe method to add a review item. If the item's status is
        PENDING, it will be added to the pending queue for retrieval.

        Args:
            item: Review item to add to the queue.

        Returns:
            True if the item was added successfully.
            False if the queue is at max capacity.

        Examples:
            Adding a pending item:
                >>> queue = ReviewQueue()
                >>> item = ReviewItem(prompt="test", response="result")
                >>> queue.add(item)
                True
                >>> queue.pending_count
                1

            Adding a non-pending item:
                >>> queue = ReviewQueue()
                >>> item = ReviewItem(
                ...     prompt="test",
                ...     response="result",
                ...     status=ReviewStatus.APPROVED
                ... )
                >>> queue.add(item)
                True
                >>> queue.pending_count  # Not in pending queue
                0

            Queue at capacity:
                >>> queue = ReviewQueue(max_size=1)
                >>> queue.add(ReviewItem(prompt="p1", response="r1"))
                True
                >>> queue.add(ReviewItem(prompt="p2", response="r2"))
                False
        """
        with self._lock:
            if self._max_size and len(self._items) >= self._max_size:
                return False
            self._items[item.item_id] = item
            if item.status == ReviewStatus.PENDING:
                self._pending.append(item.item_id)
            return True

    def get_next(self) -> Optional[ReviewItem]:
        """Get the next pending item from the queue.

        Retrieves the oldest pending item (FIFO order) and changes its
        status to IN_PROGRESS. Thread-safe operation.

        Returns:
            The next ReviewItem to be processed, or None if no pending items.

        Examples:
            Getting next item:
                >>> queue = ReviewQueue()
                >>> queue.add(ReviewItem(prompt="first", response="r1"))
                >>> queue.add(ReviewItem(prompt="second", response="r2"))
                >>> item = queue.get_next()
                >>> print(item.prompt)
                first
                >>> print(item.status)
                ReviewStatus.IN_PROGRESS

            Empty queue:
                >>> queue = ReviewQueue()
                >>> queue.get_next() is None
                True

            Processing all items:
                >>> queue = ReviewQueue()
                >>> queue.add(ReviewItem(prompt="p1", response="r1"))
                >>> queue.add(ReviewItem(prompt="p2", response="r2"))
                >>> while (item := queue.get_next()) is not None:
                ...     print(f"Processing: {item.prompt}")
                Processing: p1
                Processing: p2
        """
        with self._lock:
            while self._pending:
                item_id = self._pending.pop(0)
                if item_id in self._items:
                    item = self._items[item_id]
                    if item.status == ReviewStatus.PENDING:
                        item.status = ReviewStatus.IN_PROGRESS
                        return item
            return None

    def get_by_id(self, item_id: str) -> Optional[ReviewItem]:
        """Get a specific item by its ID.

        Args:
            item_id: The unique identifier of the item to retrieve.

        Returns:
            The ReviewItem if found, None otherwise.

        Examples:
            Finding an item:
                >>> queue = ReviewQueue()
                >>> item = ReviewItem(prompt="test", response="result")
                >>> queue.add(item)
                >>> found = queue.get_by_id(item.item_id)
                >>> print(found.prompt)
                test

            Item not found:
                >>> queue = ReviewQueue()
                >>> queue.get_by_id("nonexistent-id") is None
                True

            Updating item after retrieval:
                >>> queue = ReviewQueue()
                >>> item = ReviewItem(prompt="test", response="result")
                >>> queue.add(item)
                >>> found = queue.get_by_id(item.item_id)
                >>> found.status = ReviewStatus.APPROVED
                >>> queue.update(found)
        """
        return self._items.get(item_id)

    def update(self, item: ReviewItem) -> None:
        """Update an existing item in the queue.

        Replaces the stored item with the updated version. Thread-safe.

        Args:
            item: The ReviewItem with updated data.

        Examples:
            Updating status:
                >>> queue = ReviewQueue()
                >>> item = ReviewItem(prompt="test", response="result")
                >>> queue.add(item)
                >>> item.status = ReviewStatus.APPROVED
                >>> queue.update(item)
                >>> updated = queue.get_by_id(item.item_id)
                >>> print(updated.status)
                ReviewStatus.APPROVED

            Adding feedback and updating:
                >>> queue = ReviewQueue()
                >>> item = ReviewItem(prompt="test", response="result")
                >>> queue.add(item)
                >>> item.add_feedback(Feedback(content="Good"))
                >>> queue.update(item)

            Bulk update pattern:
                >>> queue = ReviewQueue()
                >>> for i in range(3):
                ...     queue.add(ReviewItem(prompt=f"p{i}", response=f"r{i}"))
                >>> for item in queue.get_pending():
                ...     item.status = ReviewStatus.APPROVED
                ...     queue.update(item)
        """
        with self._lock:
            self._items[item.item_id] = item

    def remove(self, item_id: str) -> Optional[ReviewItem]:
        """Remove an item from the queue.

        Removes the item completely from the queue, including from the
        pending list if present. Thread-safe.

        Args:
            item_id: The unique identifier of the item to remove.

        Returns:
            The removed ReviewItem if found, None otherwise.

        Examples:
            Removing an item:
                >>> queue = ReviewQueue()
                >>> item = ReviewItem(prompt="test", response="result")
                >>> queue.add(item)
                >>> removed = queue.remove(item.item_id)
                >>> print(len(queue))
                0

            Item not in queue:
                >>> queue = ReviewQueue()
                >>> queue.remove("nonexistent-id") is None
                True

            Cleanup pattern:
                >>> queue = ReviewQueue()
                >>> for i in range(5):
                ...     queue.add(ReviewItem(prompt=f"p{i}", response=f"r{i}"))
                >>> for item in queue.get_by_status(ReviewStatus.PENDING):
                ...     queue.remove(item.item_id)
        """
        with self._lock:
            item = self._items.pop(item_id, None)
            if item_id in self._pending:
                self._pending.remove(item_id)
            return item

    def get_pending(self) -> list[ReviewItem]:
        """Get all items with pending status.

        Returns a list of all items currently awaiting review. Thread-safe.

        Returns:
            List of ReviewItems with status PENDING.

        Examples:
            Getting pending items:
                >>> queue = ReviewQueue()
                >>> queue.add(ReviewItem(prompt="p1", response="r1"))
                >>> queue.add(ReviewItem(prompt="p2", response="r2"))
                >>> pending = queue.get_pending()
                >>> print(len(pending))
                2

            After processing some items:
                >>> queue = ReviewQueue()
                >>> queue.add(ReviewItem(prompt="p1", response="r1"))
                >>> queue.add(ReviewItem(prompt="p2", response="r2"))
                >>> _ = queue.get_next()  # Process one
                >>> pending = queue.get_pending()
                >>> print(len(pending))
                1

            Batch assignment:
                >>> queue = ReviewQueue()
                >>> for i in range(10):
                ...     queue.add(ReviewItem(prompt=f"p{i}", response=f"r{i}"))
                >>> for item in queue.get_pending()[:5]:
                ...     item.assigned_to = "reviewer_001"
                ...     queue.update(item)
        """
        with self._lock:
            return [self._items[item_id] for item_id in self._pending if item_id in self._items]

    def get_by_status(self, status: ReviewStatus) -> list[ReviewItem]:
        """Get all items with a specific status.

        Args:
            status: The ReviewStatus to filter by.

        Returns:
            List of ReviewItems matching the specified status.

        Examples:
            Getting approved items:
                >>> queue = ReviewQueue()
                >>> item = ReviewItem(prompt="test", response="result")
                >>> item.status = ReviewStatus.APPROVED
                >>> queue.add(item)
                >>> approved = queue.get_by_status(ReviewStatus.APPROVED)
                >>> print(len(approved))
                1

            Getting rejected items for re-review:
                >>> queue = ReviewQueue()
                >>> item = ReviewItem(prompt="test", response="bad result")
                >>> item.status = ReviewStatus.REJECTED
                >>> queue.add(item)
                >>> rejected = queue.get_by_status(ReviewStatus.REJECTED)
                >>> for item in rejected:
                ...     item.status = ReviewStatus.PENDING
                ...     queue.update(item)

            Generating reports:
                >>> queue = ReviewQueue()
                >>> # ... add items and process ...
                >>> for status in ReviewStatus:
                ...     count = len(queue.get_by_status(status))
                ...     print(f"{status.value}: {count}")
        """
        return [item for item in self._items.values() if item.status == status]

    def __len__(self) -> int:
        """Get total number of items in the queue.

        Returns:
            Total count of all items (all statuses).

        Examples:
            >>> queue = ReviewQueue()
            >>> queue.add(ReviewItem(prompt="p1", response="r1"))
            >>> queue.add(ReviewItem(prompt="p2", response="r2"))
            >>> print(len(queue))
            2
        """
        return len(self._items)

    @property
    def pending_count(self) -> int:
        """Get count of pending items.

        Returns:
            Number of items awaiting review.

        Examples:
            >>> queue = ReviewQueue()
            >>> queue.add(ReviewItem(prompt="p1", response="r1"))
            >>> print(queue.pending_count)
            1
        """
        return len(self._pending)

    def stats(self) -> dict[str, int]:
        """Get queue statistics by status.

        Returns a dictionary with counts for each ReviewStatus.

        Returns:
            Dictionary mapping status names to item counts.

        Examples:
            Basic statistics:
                >>> queue = ReviewQueue()
                >>> queue.add(ReviewItem(prompt="p1", response="r1"))
                >>> stats = queue.stats()
                >>> print(stats["pending"])
                1

            Comprehensive reporting:
                >>> queue = ReviewQueue()
                >>> for i in range(5):
                ...     queue.add(ReviewItem(prompt=f"p{i}", response=f"r{i}"))
                >>> _ = queue.get_next()  # One in progress
                >>> stats = queue.stats()
                >>> print(f"Pending: {stats['pending']}, In Progress: {stats['in_progress']}")
                Pending: 4, In Progress: 1

            Dashboard display:
                >>> stats = queue.stats()
                >>> total = sum(stats.values())
                >>> for status, count in stats.items():
                ...     if count > 0:
                ...         pct = count / total * 100
                ...         print(f"{status}: {count} ({pct:.1f}%)")
        """
        stats = {status.value: 0 for status in ReviewStatus}
        for item in self._items.values():
            stats[item.status.value] += 1
        return stats


class PriorityReviewQueue(ReviewQueue):
    """Review queue that orders items by priority level.

    Extends ReviewQueue to provide priority-based ordering. Items with higher
    priority (lower Priority enum values) are retrieved first. Within the same
    priority level, items are ordered by creation time (FIFO).

    This is useful for workflows where certain items need immediate attention
    (e.g., safety reviews, high-value customers) while routine items can wait.

    Inherits from:
        ReviewQueue: Provides base queue functionality and thread-safety.

    Examples:
        Priority ordering:
            >>> from insideLLMs.agents.hitl import PriorityReviewQueue, ReviewItem, Priority
            >>> queue = PriorityReviewQueue()
            >>> queue.add(ReviewItem(prompt="routine", response="r1", priority=Priority.LOW))
            >>> queue.add(ReviewItem(prompt="urgent", response="r2", priority=Priority.CRITICAL))
            >>> queue.add(ReviewItem(prompt="normal", response="r3", priority=Priority.MEDIUM))
            >>> item = queue.get_next()
            >>> print(item.prompt)  # CRITICAL comes first
            urgent

        Mixed priority workflow:
            >>> queue = PriorityReviewQueue()
            >>> # Add safety-critical content
            >>> queue.add(ReviewItem(
            ...     prompt="Check for harmful content",
            ...     response="...",
            ...     priority=Priority.CRITICAL
            ... ))
            >>> # Add routine content
            >>> queue.add(ReviewItem(
            ...     prompt="Summarize article",
            ...     response="...",
            ...     priority=Priority.BACKGROUND
            ... ))
            >>> # Critical items reviewed first
            >>> next_item = queue.get_next()
            >>> print(next_item.priority)
            Priority.CRITICAL

        Same priority uses FIFO:
            >>> queue = PriorityReviewQueue()
            >>> queue.add(ReviewItem(prompt="first", response="r1", priority=Priority.HIGH))
            >>> queue.add(ReviewItem(prompt="second", response="r2", priority=Priority.HIGH))
            >>> item = queue.get_next()
            >>> print(item.prompt)  # First HIGH priority item
            first

        Dynamic priority assignment:
            >>> def prioritize_by_content(response):
            ...     if "error" in response.lower():
            ...         return Priority.HIGH
            ...     return Priority.MEDIUM
            >>> queue = PriorityReviewQueue()
            >>> response = "An error occurred..."
            >>> priority = prioritize_by_content(response)
            >>> queue.add(ReviewItem(prompt="check", response=response, priority=priority))
    """

    def __init__(self, max_size: Optional[int] = None):
        """Initialize priority review queue.

        Args:
            max_size: Maximum queue size. None for unlimited capacity.

        Examples:
            Basic initialization:
                >>> queue = PriorityReviewQueue()
                >>> print(queue.pending_count)
                0

            With size limit:
                >>> queue = PriorityReviewQueue(max_size=500)
                >>> # Queue will reject items when full

            For production use:
                >>> queue = PriorityReviewQueue(max_size=10000)
                >>> # Limits memory usage for high-volume systems
        """
        super().__init__(max_size)
        self._priority_queue: PriorityQueue = PriorityQueue()

    def add(self, item: ReviewItem) -> bool:
        """Add item to queue with priority-based ordering.

        Items are stored and will be retrieved in priority order.
        Higher priority items (lower Priority enum values) come first.

        Args:
            item: Review item to add. Its priority attribute determines
                retrieval order.

        Returns:
            True if added successfully, False if queue is at capacity.

        Examples:
            Adding with priority:
                >>> queue = PriorityReviewQueue()
                >>> queue.add(ReviewItem(
                ...     prompt="urgent",
                ...     response="result",
                ...     priority=Priority.CRITICAL
                ... ))
                True

            Multiple priorities:
                >>> queue = PriorityReviewQueue()
                >>> queue.add(ReviewItem(prompt="p1", response="r1", priority=Priority.LOW))
                >>> queue.add(ReviewItem(prompt="p2", response="r2", priority=Priority.HIGH))
                >>> queue.pending_count
                2

            Capacity handling:
                >>> queue = PriorityReviewQueue(max_size=1)
                >>> queue.add(ReviewItem(prompt="p1", response="r1", priority=Priority.HIGH))
                True
                >>> queue.add(ReviewItem(prompt="p2", response="r2", priority=Priority.CRITICAL))
                False  # Queue full, even for high priority
        """
        if not super().add(item):
            return False
        with self._lock:
            if item.status == ReviewStatus.PENDING:
                self._priority_queue.put((item.priority.value, item.created_at, item.item_id))
        return True

    def get_next(self) -> Optional[ReviewItem]:
        """Get the highest priority pending item.

        Retrieves items in priority order (CRITICAL > HIGH > MEDIUM > LOW >
        BACKGROUND). Within the same priority level, older items are returned
        first (FIFO). Changes item status to IN_PROGRESS.

        Returns:
            The highest priority pending ReviewItem, or None if empty.

        Examples:
            Priority ordering:
                >>> queue = PriorityReviewQueue()
                >>> queue.add(ReviewItem(prompt="low", response="r1", priority=Priority.LOW))
                >>> queue.add(ReviewItem(prompt="high", response="r2", priority=Priority.HIGH))
                >>> item = queue.get_next()
                >>> print(item.prompt)
                high

            Emptying queue by priority:
                >>> queue = PriorityReviewQueue()
                >>> queue.add(ReviewItem(prompt="p1", response="r1", priority=Priority.MEDIUM))
                >>> queue.add(ReviewItem(prompt="p2", response="r2", priority=Priority.CRITICAL))
                >>> queue.add(ReviewItem(prompt="p3", response="r3", priority=Priority.LOW))
                >>> priorities = []
                >>> while (item := queue.get_next()) is not None:
                ...     priorities.append(item.priority.name)
                >>> print(priorities)
                ['CRITICAL', 'MEDIUM', 'LOW']

            Status transition:
                >>> queue = PriorityReviewQueue()
                >>> queue.add(ReviewItem(prompt="test", response="result"))
                >>> item = queue.get_next()
                >>> print(item.status)
                ReviewStatus.IN_PROGRESS
        """
        with self._lock:
            while not self._priority_queue.empty():
                try:
                    _, _, item_id = self._priority_queue.get_nowait()
                    if item_id in self._items:
                        item = self._items[item_id]
                        if item.status == ReviewStatus.PENDING:
                            item.status = ReviewStatus.IN_PROGRESS
                            if item_id in self._pending:
                                self._pending.remove(item_id)
                            return item
                except Empty:
                    break
            return None


# Protocol for input handlers
class InputHandler(Protocol):
    """Protocol defining the interface for handling human input in HITL workflows.

    This Protocol defines the required methods that any input handler must
    implement to integrate with HITL sessions. Implementations can be console-based,
    GUI-based, API-based, or callback-based depending on the application.

    Methods that must be implemented:
        get_approval: Request approval decision from a human reviewer.
        get_feedback: Collect feedback on a review item.
        get_edit: Allow human to edit the response content.

    Examples:
        Implementing a custom input handler:
            >>> class MyInputHandler:
            ...     def get_approval(self, item: ReviewItem) -> tuple[bool, Optional[str]]:
            ...         # Custom approval logic
            ...         return True, "Auto-approved by policy"
            ...
            ...     def get_feedback(self, item: ReviewItem) -> Feedback:
            ...         return Feedback(feedback_type=FeedbackType.APPROVE)
            ...
            ...     def get_edit(self, item: ReviewItem) -> str:
            ...         return item.response  # No edits

        Using with HITLSession:
            >>> handler = CallbackInputHandler(
            ...     approval_callback=lambda item: (True, None)
            ... )
            >>> session = HITLSession(model, input_handler=handler)

        Type checking:
            >>> def process_with_handler(handler: InputHandler):
            ...     item = ReviewItem(prompt="test", response="result")
            ...     approved, comment = handler.get_approval(item)
            ...     return approved
    """

    def get_approval(self, item: ReviewItem) -> tuple[bool, Optional[str]]:
        """Get approval decision for a review item.

        Args:
            item: The ReviewItem to be approved or rejected.

        Returns:
            A tuple of (approved, comment) where:
            - approved: True if approved, False if rejected, None if skipped
            - comment: Optional comment from the reviewer

        Examples:
            Approval with comment:
                >>> approved, comment = handler.get_approval(item)
                >>> if approved:
                ...     print(f"Approved: {comment}")

            Handling skip:
                >>> approved, comment = handler.get_approval(item)
                >>> if approved is None:
                ...     print("Item skipped")
        """
        ...

    def get_feedback(self, item: ReviewItem) -> Feedback:
        """Collect feedback for a review item.

        Args:
            item: The ReviewItem to collect feedback on.

        Returns:
            A Feedback object containing the human's feedback.

        Examples:
            Collecting feedback:
                >>> feedback = handler.get_feedback(item)
                >>> print(f"Type: {feedback.feedback_type}")
                >>> print(f"Rating: {feedback.rating}")
        """
        ...

    def get_edit(self, item: ReviewItem) -> str:
        """Allow human to edit the response content.

        Args:
            item: The ReviewItem whose response may be edited.

        Returns:
            The edited response text (may be unchanged from original).

        Examples:
            Getting edited content:
                >>> edited = handler.get_edit(item)
                >>> if edited != item.response:
                ...     print("Response was modified")
        """
        ...


class ConsoleInputHandler:
    """Console-based input handler for interactive HITL sessions in terminals.

    Provides a text-based interface for human reviewers to approve, provide
    feedback, and edit model responses. Useful for local development, testing,
    and simple review workflows.

    Attributes:
        timeout: Optional timeout in seconds for input operations.

    Examples:
        Basic console interaction:
            >>> handler = ConsoleInputHandler()
            >>> # In an interactive session, this prompts for input:
            >>> # Review Item: abc-123
            >>> # Prompt: Summarize the document
            >>> # Response: The document discusses...
            >>> # Approve? (y/n/skip): y
            >>> # Comment (optional): Good summary

        Using with HITLSession:
            >>> from insideLLMs.agents.hitl import HITLSession, ConsoleInputHandler
            >>> handler = ConsoleInputHandler()
            >>> session = HITLSession(model, input_handler=handler)
            >>> # Interactive prompts appear in console
            >>> response, item = session.generate_and_review("Test prompt")

        With timeout for timed reviews:
            >>> handler = ConsoleInputHandler(timeout=60.0)  # 60 second limit
            >>> # Note: timeout is stored but not enforced by default input()

        For development and testing:
            >>> handler = ConsoleInputHandler()
            >>> item = ReviewItem(prompt="test", response="result")
            >>> # This will prompt interactively:
            >>> # approved, comment = handler.get_approval(item)
    """

    def __init__(self, timeout: Optional[float] = None):
        """Initialize console input handler.

        Args:
            timeout: Timeout in seconds for input operations. None means no
                timeout. Note that standard input() does not support timeout
                directly; this is stored for potential use with custom
                implementations.

        Examples:
            Default handler:
                >>> handler = ConsoleInputHandler()
                >>> handler.timeout is None
                True

            With timeout (for reference):
                >>> handler = ConsoleInputHandler(timeout=30.0)
                >>> handler.timeout
                30.0

            For production review sessions:
                >>> handler = ConsoleInputHandler(timeout=120.0)
                >>> # 2 minutes per item review time
        """
        self.timeout = timeout

    def get_approval(self, item: ReviewItem) -> tuple[bool, Optional[str]]:
        """Get approval decision via console input.

        Displays the review item details and prompts for approval decision
        and optional comment.

        Args:
            item: The ReviewItem to display and get approval for.

        Returns:
            Tuple of (approved, comment):
            - approved: True if 'y', False if 'n', None if 'skip'
            - comment: Optional text comment from reviewer

        Examples:
            User approves (interactive):
                >>> # Console shows:
                >>> # ============================================================
                >>> # Review Item: abc-123
                >>> # Prompt: Summarize this text
                >>> # Response: This is a summary...
                >>> # ============================================================
                >>> # Approve? (y/n/skip): y
                >>> # Comment (optional): Looks good
                >>> # Returns: (True, "Looks good")

            User rejects (interactive):
                >>> # Approve? (y/n/skip): n
                >>> # Comment (optional): Missing key details
                >>> # Returns: (False, "Missing key details")

            User skips (interactive):
                >>> # Approve? (y/n/skip): skip
                >>> # Comment (optional):
                >>> # Returns: (None, None)
        """
        print(f"\n{'=' * 60}")
        print(f"Review Item: {item.item_id}")
        print(f"Prompt: {item.prompt}")
        print(f"Response: {item.response}")
        print(f"{'=' * 60}")

        response = input("Approve? (y/n/skip): ").strip().lower()
        comment = input("Comment (optional): ").strip() or None

        if response == "y":
            return True, comment
        elif response == "skip":
            return None, comment  # type: ignore[return-value]  # Skip returns None for approval status
        return False, comment

    def get_feedback(self, item: ReviewItem) -> Feedback:
        """Collect feedback via console input.

        Displays the response and prompts for a rating (0-10) and comment.

        Args:
            item: The ReviewItem to collect feedback on.

        Returns:
            Feedback object with rating and/or comment.

        Examples:
            With rating (interactive):
                >>> # Console shows:
                >>> # ============================================================
                >>> # Provide feedback for: abc-123
                >>> # Response: This is the response...
                >>> # ============================================================
                >>> # Rating (0-10, or Enter to skip): 8
                >>> # Comment: Well structured response
                >>> # Returns: Feedback(feedback_type=RATING, rating=0.8, content="...")

            Comment only (interactive):
                >>> # Rating (0-10, or Enter to skip):
                >>> # Comment: Needs more detail
                >>> # Returns: Feedback(feedback_type=COMMENT, content="Needs more detail")

            Both rating and comment:
                >>> # Rating (0-10, or Enter to skip): 7
                >>> # Comment: Good but could improve
                >>> # Returns: Feedback with both rating=0.7 and content
        """
        print(f"\n{'=' * 60}")
        print(f"Provide feedback for: {item.item_id}")
        print(f"Response: {item.response}")
        print(f"{'=' * 60}")

        rating_str = input("Rating (0-10, or Enter to skip): ").strip()
        rating = float(rating_str) / 10 if rating_str else None

        comment = input("Comment: ").strip()

        return Feedback(
            feedback_type=FeedbackType.RATING if rating else FeedbackType.COMMENT,
            content=comment,
            rating=rating,
        )

    def get_edit(self, item: ReviewItem) -> str:
        """Get edited response via console input.

        Displays the current response and prompts for an edited version.

        Args:
            item: The ReviewItem whose response may be edited.

        Returns:
            The edited response text, or original if Enter pressed.

        Examples:
            User edits response (interactive):
                >>> # Console shows:
                >>> # ============================================================
                >>> # Edit response for: abc-123
                >>> # Current: Original response text
                >>> # ============================================================
                >>> # New response (or Enter to keep): Corrected response text
                >>> # Returns: "Corrected response text"

            User keeps original (interactive):
                >>> # New response (or Enter to keep):
                >>> # Returns: item.response (original)

            For correction workflows:
                >>> # Current: The captial of France is Berlin
                >>> # New response (or Enter to keep): The capital of France is Paris
                >>> # Returns: "The capital of France is Paris"
        """
        print(f"\n{'=' * 60}")
        print(f"Edit response for: {item.item_id}")
        print(f"Current: {item.response}")
        print(f"{'=' * 60}")

        edited = input("New response (or Enter to keep): ").strip()
        return edited if edited else item.response


class CallbackInputHandler:
    """Callback-based input handler for programmatic HITL interactions.

    Enables programmatic control over HITL decisions via callback functions.
    This is ideal for automated testing, rule-based approval systems, and
    integration with external systems (APIs, UIs, databases).

    Attributes:
        _approval_callback: Function called for approval decisions.
        _feedback_callback: Function called for feedback collection.
        _edit_callback: Function called for edit operations.

    Examples:
        Auto-approve all items:
            >>> handler = CallbackInputHandler(
            ...     approval_callback=lambda item: (True, None)
            ... )
            >>> item = ReviewItem(prompt="test", response="result")
            >>> approved, comment = handler.get_approval(item)
            >>> print(approved)
            True

        Rule-based approval:
            >>> def approve_if_long(item):
            ...     approved = len(item.response) > 100
            ...     comment = "Auto-approved (sufficient length)" if approved else "Too short"
            ...     return approved, comment
            >>> handler = CallbackInputHandler(approval_callback=approve_if_long)

        Custom feedback collection:
            >>> def rate_by_length(item):
            ...     rating = min(len(item.response) / 500, 1.0)
            ...     return Feedback(feedback_type=FeedbackType.RATING, rating=rating)
            >>> handler = CallbackInputHandler(feedback_callback=rate_by_length)

        Integration with external API:
            >>> def external_approval(item):
            ...     # Call external moderation API
            ...     # result = moderation_api.check(item.response)
            ...     # return result.approved, result.reason
            ...     return True, "API approved"
            >>> handler = CallbackInputHandler(approval_callback=external_approval)
    """

    def __init__(
        self,
        approval_callback: Optional[Callable[[ReviewItem], tuple[bool, Optional[str]]]] = None,
        feedback_callback: Optional[Callable[[ReviewItem], Feedback]] = None,
        edit_callback: Optional[Callable[[ReviewItem], str]] = None,
    ):
        """Initialize callback-based input handler.

        Args:
            approval_callback: Function that receives a ReviewItem and returns
                (approved: bool, comment: Optional[str]). If None, all items
                are auto-approved.
            feedback_callback: Function that receives a ReviewItem and returns
                a Feedback object. If None, returns approval feedback.
            edit_callback: Function that receives a ReviewItem and returns
                the edited response string. If None, returns original response.

        Examples:
            Simple auto-approve handler:
                >>> handler = CallbackInputHandler(
                ...     approval_callback=lambda item: (True, "Auto-approved")
                ... )

            Complete handler with all callbacks:
                >>> handler = CallbackInputHandler(
                ...     approval_callback=lambda item: (True, None),
                ...     feedback_callback=lambda item: Feedback(
                ...         feedback_type=FeedbackType.RATING,
                ...         rating=0.9
                ...     ),
                ...     edit_callback=lambda item: item.response.upper()
                ... )

            Conditional approval:
                >>> def conditional_approve(item):
                ...     if "error" in item.response.lower():
                ...         return False, "Contains error mention"
                ...     return True, None
                >>> handler = CallbackInputHandler(approval_callback=conditional_approve)

            With external state:
                >>> approved_count = {"count": 0}
                >>> def counting_approval(item):
                ...     approved_count["count"] += 1
                ...     return True, f"Approved #{approved_count['count']}"
                >>> handler = CallbackInputHandler(approval_callback=counting_approval)
        """
        self._approval_callback = approval_callback
        self._feedback_callback = feedback_callback
        self._edit_callback = edit_callback

    def get_approval(self, item: ReviewItem) -> tuple[bool, Optional[str]]:
        """Get approval decision via callback.

        Calls the approval callback if set, otherwise auto-approves.

        Args:
            item: The ReviewItem to get approval for.

        Returns:
            Tuple of (approved, comment). If no callback is set, returns
            (True, None) to auto-approve all items.

        Examples:
            With callback:
                >>> handler = CallbackInputHandler(
                ...     approval_callback=lambda item: (
                ...         len(item.response) > 50,
                ...         "Length check"
                ...     )
                ... )
                >>> item = ReviewItem(prompt="test", response="Short")
                >>> approved, comment = handler.get_approval(item)
                >>> print(approved)
                False

            Without callback (auto-approve):
                >>> handler = CallbackInputHandler()
                >>> item = ReviewItem(prompt="test", response="result")
                >>> approved, comment = handler.get_approval(item)
                >>> print(approved, comment)
                True None

            Complex approval logic:
                >>> def complex_approval(item):
                ...     checks = [
                ...         len(item.response) > 10,
                ...         "error" not in item.response.lower(),
                ...         item.priority != Priority.CRITICAL
                ...     ]
                ...     return all(checks), f"Passed {sum(checks)}/3 checks"
                >>> handler = CallbackInputHandler(approval_callback=complex_approval)
        """
        if self._approval_callback:
            return self._approval_callback(item)
        return True, None  # Auto-approve if no callback

    def get_feedback(self, item: ReviewItem) -> Feedback:
        """Get feedback via callback.

        Calls the feedback callback if set, otherwise returns approval feedback.

        Args:
            item: The ReviewItem to get feedback for.

        Returns:
            Feedback object from callback, or default approval feedback.

        Examples:
            With callback:
                >>> handler = CallbackInputHandler(
                ...     feedback_callback=lambda item: Feedback(
                ...         feedback_type=FeedbackType.RATING,
                ...         rating=0.85,
                ...         content="Good quality"
                ...     )
                ... )
                >>> item = ReviewItem(prompt="test", response="result")
                >>> feedback = handler.get_feedback(item)
                >>> print(feedback.rating)
                0.85

            Without callback (default):
                >>> handler = CallbackInputHandler()
                >>> item = ReviewItem(prompt="test", response="result")
                >>> feedback = handler.get_feedback(item)
                >>> print(feedback.feedback_type)
                FeedbackType.APPROVE

            Dynamic rating based on content:
                >>> def rate_response(item):
                ...     # Rate based on response length
                ...     rating = min(len(item.response) / 200, 1.0)
                ...     return Feedback(
                ...         feedback_type=FeedbackType.RATING,
                ...         rating=rating,
                ...         content=f"Rated {rating:.2f} based on length"
                ...     )
                >>> handler = CallbackInputHandler(feedback_callback=rate_response)
        """
        if self._feedback_callback:
            return self._feedback_callback(item)
        return Feedback(feedback_type=FeedbackType.APPROVE)

    def get_edit(self, item: ReviewItem) -> str:
        """Get edited response via callback.

        Calls the edit callback if set, otherwise returns original response.

        Args:
            item: The ReviewItem to potentially edit.

        Returns:
            Edited response from callback, or original response.

        Examples:
            With callback:
                >>> handler = CallbackInputHandler(
                ...     edit_callback=lambda item: item.response.strip()
                ... )
                >>> item = ReviewItem(prompt="test", response="  result  ")
                >>> edited = handler.get_edit(item)
                >>> print(repr(edited))
                'result'

            Without callback (keep original):
                >>> handler = CallbackInputHandler()
                >>> item = ReviewItem(prompt="test", response="original")
                >>> edited = handler.get_edit(item)
                >>> print(edited)
                original

            Auto-correction callback:
                >>> import re
                >>> def fix_common_typos(item):
                ...     text = item.response
                ...     text = re.sub(r'\bteh\b', 'the', text)
                ...     text = re.sub(r'\brecieve\b', 'receive', text)
                ...     return text
                >>> handler = CallbackInputHandler(edit_callback=fix_common_typos)

            Transformation callback:
                >>> handler = CallbackInputHandler(
                ...     edit_callback=lambda item: item.response.title()
                ... )
                >>> item = ReviewItem(prompt="test", response="hello world")
                >>> print(handler.get_edit(item))
                Hello World
        """
        if self._edit_callback:
            return self._edit_callback(item)
        return item.response  # Keep original if no callback


class HITLSession:
    """Interactive human-in-the-loop session for model evaluation and feedback.

    HITLSession provides a managed environment for generating model outputs
    and collecting human feedback, approvals, and edits. It maintains a history
    of all interactions and provides statistics for quality monitoring.

    Attributes:
        model: The language model being evaluated.
        config: Session configuration settings.
        input_handler: Handler for collecting human input.
        session_id: Unique identifier for this session.

    Examples:
        Basic approval workflow:
            >>> from insideLLMs.agents.hitl import HITLSession, CallbackInputHandler
            >>> # Auto-approve handler for testing
            >>> handler = CallbackInputHandler(
            ...     approval_callback=lambda item: (True, "Looks good")
            ... )
            >>> session = HITLSession(model, input_handler=handler)
            >>> response, item = session.generate_and_review("Summarize this text")
            >>> print(item.status)
            ReviewStatus.APPROVED

        Collecting feedback on responses:
            >>> handler = CallbackInputHandler(
            ...     feedback_callback=lambda item: Feedback(
            ...         feedback_type=FeedbackType.RATING,
            ...         rating=0.85,
            ...         content="Good quality"
            ...     )
            ... )
            >>> session = HITLSession(model, input_handler=handler)
            >>> response, feedback = session.collect_feedback("Generate a poem")
            >>> print(f"Rating: {feedback.rating}")
            Rating: 0.85

        Human editing workflow:
            >>> handler = CallbackInputHandler(
            ...     edit_callback=lambda item: item.response.replace("error", "success")
            ... )
            >>> session = HITLSession(model, input_handler=handler)
            >>> original, edited = session.edit_response("Write code")
            >>> if original != edited:
            ...     print("Response was corrected")

        Getting session statistics:
            >>> session = HITLSession(model, input_handler=handler)
            >>> # ... perform multiple reviews ...
            >>> stats = session.get_statistics()
            >>> print(f"Approval rate: {stats['approval_rate']:.1%}")
    """

    def __init__(
        self,
        model: Any,
        config: Optional[HITLConfig] = None,
        input_handler: Optional[InputHandler] = None,
    ):
        """Initialize a Human-in-the-Loop session.

        Args:
            model: The model to evaluate. Must have a generate(prompt, **kwargs)
                method that returns a string response.
            config: Session configuration. If None, uses default HITLConfig.
            input_handler: Handler for user input. If None, uses CallbackInputHandler
                which auto-approves all items.

        Examples:
            With default settings:
                >>> session = HITLSession(model)
                >>> # Uses default config and auto-approve handler

            With custom configuration:
                >>> config = HITLConfig(
                ...     auto_approve_threshold=0.95,
                ...     require_comment=True
                ... )
                >>> session = HITLSession(model, config=config)

            With custom input handler:
                >>> handler = CallbackInputHandler(
                ...     approval_callback=lambda item: (True, None)
                ... )
                >>> session = HITLSession(model, input_handler=handler)

            Full configuration:
                >>> config = HITLConfig(timeout_seconds=60.0)
                >>> handler = ConsoleInputHandler()
                >>> session = HITLSession(model, config=config, input_handler=handler)
        """
        self.model = model
        self.config = config or HITLConfig()
        self.input_handler = input_handler or CallbackInputHandler()
        self.session_id = str(uuid.uuid4())
        self._history: list[ReviewItem] = []
        self._lock = threading.Lock()
        self._start_time = datetime.now()

    def generate_and_review(
        self,
        prompt: str,
        require_approval: bool = True,
        **kwargs: Any,
    ) -> tuple[str, ReviewItem]:
        """Generate a model response and optionally get human review.

        Calls the model to generate a response, then requests human approval
        if required. The response and all feedback are recorded in session history.

        Args:
            prompt: The input prompt to send to the model.
            require_approval: If True, requests human approval. If False,
                auto-approves the response.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            A tuple of (response, review_item) where:
            - response: The model's generated text response.
            - review_item: ReviewItem containing the prompt, response, status,
              and any feedback collected.

        Examples:
            Basic review:
                >>> response, item = session.generate_and_review("Explain gravity")
                >>> print(f"Status: {item.status}")
                >>> print(f"Response: {response[:50]}...")

            Without approval requirement:
                >>> response, item = session.generate_and_review(
                ...     "Low risk prompt",
                ...     require_approval=False
                ... )
                >>> print(item.status)  # Auto-approved
                ReviewStatus.APPROVED

            With model parameters:
                >>> response, item = session.generate_and_review(
                ...     "Be creative",
                ...     temperature=0.9,
                ...     max_tokens=500
                ... )

            Processing multiple prompts:
                >>> prompts = ["Question 1", "Question 2", "Question 3"]
                >>> for prompt in prompts:
                ...     response, item = session.generate_and_review(prompt)
                ...     if item.status == ReviewStatus.REJECTED:
                ...         print(f"Rejected: {prompt}")
        """
        # Generate response
        response = self.model.generate(prompt, **kwargs)

        # Create review item
        item = ReviewItem(
            prompt=prompt,
            response=response,
            model_id=getattr(self.model, "model_id", None),
            metadata=kwargs,
        )

        if require_approval:
            approved, comment = self.input_handler.get_approval(item)

            if approved is None:
                item.status = ReviewStatus.SKIPPED
            elif approved:
                item.status = ReviewStatus.APPROVED
            else:
                item.status = ReviewStatus.REJECTED

            if comment:
                item.add_feedback(
                    Feedback(
                        feedback_type=FeedbackType.COMMENT,
                        content=comment,
                    )
                )
        else:
            item.status = ReviewStatus.APPROVED

        with self._lock:
            self._history.append(item)

        return response, item

    def collect_feedback(self, prompt: str, **kwargs: Any) -> tuple[str, Feedback]:
        """Generate a response and collect detailed feedback.

        Unlike generate_and_review which focuses on approval, this method
        collects structured feedback including ratings and comments.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            A tuple of (response, feedback) where:
            - response: The model's generated text response.
            - feedback: Feedback object containing ratings, comments, etc.

        Examples:
            Collecting ratings:
                >>> response, feedback = session.collect_feedback("Write a summary")
                >>> if feedback.rating:
                ...     print(f"Quality: {feedback.rating * 100:.0f}%")

            With model parameters:
                >>> response, feedback = session.collect_feedback(
                ...     "Generate code",
                ...     temperature=0.2
                ... )

            Batch feedback collection:
                >>> prompts = ["Task 1", "Task 2", "Task 3"]
                >>> results = []
                >>> for prompt in prompts:
                ...     response, feedback = session.collect_feedback(prompt)
                ...     results.append({
                ...         "prompt": prompt,
                ...         "response": response,
                ...         "rating": feedback.rating
                ...     })

            Analyzing feedback:
                >>> response, feedback = session.collect_feedback("Test prompt")
                >>> print(f"Type: {feedback.feedback_type}")
                >>> print(f"Comment: {feedback.content}")
        """
        response = self.model.generate(prompt, **kwargs)

        item = ReviewItem(
            prompt=prompt,
            response=response,
            model_id=getattr(self.model, "model_id", None),
        )

        feedback = self.input_handler.get_feedback(item)
        item.add_feedback(feedback)
        item.status = ReviewStatus.APPROVED

        with self._lock:
            self._history.append(item)

        return response, feedback

    def edit_response(self, prompt: str, **kwargs: Any) -> tuple[str, str]:
        """Generate a response and allow human editing.

        Useful for correction workflows where humans can improve model outputs.
        Tracks both original and edited versions.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            A tuple of (original, edited) where:
            - original: The model's original generated response.
            - edited: The human-edited version (may be same as original).

        Examples:
            Basic editing:
                >>> original, edited = session.edit_response("Write instructions")
                >>> if original != edited:
                ...     print("Human made corrections")
                ... else:
                ...     print("Response accepted as-is")

            Correction workflow:
                >>> original, edited = session.edit_response("Translate to French")
                >>> # Use edited version for downstream tasks
                >>> final_output = edited

            Tracking edits:
                >>> prompts = ["Task 1", "Task 2", "Task 3"]
                >>> edits = []
                >>> for prompt in prompts:
                ...     original, edited = session.edit_response(prompt)
                ...     if original != edited:
                ...         edits.append({"prompt": prompt, "original": original, "edited": edited})
                >>> print(f"Made {len(edits)} corrections")

            Building training data:
                >>> corrections = []
                >>> for prompt in eval_prompts:
                ...     original, edited = session.edit_response(prompt)
                ...     if original != edited:
                ...         corrections.append({
                ...             "input": prompt,
                ...             "bad_output": original,
                ...             "good_output": edited
                ...         })
        """
        original = self.model.generate(prompt, **kwargs)

        item = ReviewItem(
            prompt=prompt,
            response=original,
            model_id=getattr(self.model, "model_id", None),
        )

        edited = self.input_handler.get_edit(item)

        if edited != original:
            item.status = ReviewStatus.EDITED
            item.add_feedback(
                Feedback(
                    feedback_type=FeedbackType.EDIT,
                    edited_content=edited,
                )
            )
        else:
            item.status = ReviewStatus.APPROVED

        with self._lock:
            self._history.append(item)

        return original, edited

    @property
    def history(self) -> list[ReviewItem]:
        """Get a copy of the session history.

        Returns:
            List of all ReviewItems processed in this session.

        Examples:
            Accessing history:
                >>> for item in session.history:
                ...     print(f"{item.status}: {item.prompt[:30]}...")

            Finding rejected items:
                >>> rejected = [item for item in session.history
                ...             if item.status == ReviewStatus.REJECTED]
                >>> print(f"Rejected {len(rejected)} items")

            Exporting for analysis:
                >>> history_data = [item.to_dict() for item in session.history]
        """
        return list(self._history)

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics for the session.

        Returns:
            Dictionary containing:
            - session_id: Unique session identifier
            - total: Total items processed
            - approved/rejected/edited/skipped: Counts by status
            - approval_rate: Fraction of items approved
            - edit_rate: Fraction of items edited
            - average_rating: Mean of all ratings (if any)
            - duration_seconds: Session duration

        Examples:
            Basic statistics:
                >>> stats = session.get_statistics()
                >>> print(f"Processed {stats['total']} items")
                >>> print(f"Approval rate: {stats['approval_rate']:.1%}")

            Quality monitoring:
                >>> stats = session.get_statistics()
                >>> if stats['average_rating'] and stats['average_rating'] < 0.7:
                ...     print("Warning: Low average quality rating")

            Session summary:
                >>> stats = session.get_statistics()
                >>> print(f"Session {stats['session_id']}")
                >>> print(f"Duration: {stats['duration_seconds']:.0f} seconds")
                >>> print(f"Approved: {stats['approved']}/{stats['total']}")
                >>> print(f"Rejected: {stats['rejected']}/{stats['total']}")

            Comparing sessions:
                >>> stats1 = session1.get_statistics()
                >>> stats2 = session2.get_statistics()
                >>> print(f"Model A approval: {stats1['approval_rate']:.1%}")
                >>> print(f"Model B approval: {stats2['approval_rate']:.1%}")
        """
        with self._lock:
            total = len(self._history)
            if total == 0:
                return {"total": 0, "duration_seconds": 0}

            approved = sum(1 for item in self._history if item.status == ReviewStatus.APPROVED)
            rejected = sum(1 for item in self._history if item.status == ReviewStatus.REJECTED)
            edited = sum(1 for item in self._history if item.status == ReviewStatus.EDITED)
            skipped = sum(1 for item in self._history if item.status == ReviewStatus.SKIPPED)

            ratings = [
                f.rating for item in self._history for f in item.feedback if f.rating is not None
            ]

            return {
                "session_id": self.session_id,
                "total": total,
                "approved": approved,
                "rejected": rejected,
                "edited": edited,
                "skipped": skipped,
                "approval_rate": approved / total if total > 0 else 0,
                "edit_rate": edited / total if total > 0 else 0,
                "average_rating": sum(ratings) / len(ratings) if ratings else None,
                "duration_seconds": (datetime.now() - self._start_time).total_seconds(),
            }

    def export_history(self) -> list[dict[str, Any]]:
        """Export session history as a list of dictionaries.

        Useful for saving to JSON, sending to APIs, or database storage.

        Returns:
            List of dictionaries, one per ReviewItem.

        Examples:
            Save to JSON file:
                >>> import json
                >>> history = session.export_history()
                >>> with open("session_history.json", "w") as f:
                ...     json.dump(history, f, indent=2)

            Send to logging system:
                >>> for item_dict in session.export_history():
                ...     logger.info("Review completed", extra=item_dict)

            Database insertion:
                >>> for item_dict in session.export_history():
                ...     db.insert("reviews", item_dict)

            Analysis pipeline:
                >>> import pandas as pd
                >>> history = session.export_history()
                >>> df = pd.DataFrame(history)
                >>> print(df["status"].value_counts())
        """
        return [item.to_dict() for item in self._history]


class InteractiveSession(HITLSession):
    """Extended HITL session with event-driven callbacks for real-time integration.

    InteractiveSession extends HITLSession with an event system that notifies
    registered callbacks when specific actions occur (generation, approval,
    rejection, editing). This is useful for logging, metrics, notifications,
    and integrating with external systems.

    Supported events:
        - on_generate: Fired after every generation
        - on_approve: Fired when an item is approved
        - on_reject: Fired when an item is rejected
        - on_edit: Fired when an item is edited
        - on_feedback: Reserved for feedback collection events

    Inherits from:
        HITLSession: Provides base HITL functionality.

    Examples:
        Basic event handling:
            >>> session = InteractiveSession(model)
            >>> session.on("on_approve", lambda item: print(f"Approved: {item.item_id}"))
            >>> session.on("on_reject", lambda item: print(f"Rejected: {item.item_id}"))
            >>> response, item = session.generate_and_review("Test prompt")

        Logging integration:
            >>> import logging
            >>> logger = logging.getLogger(__name__)
            >>> session = InteractiveSession(model)
            >>> session.on("on_generate", lambda item: logger.info(f"Generated: {item.item_id}"))
            >>> session.on("on_reject", lambda item: logger.warning(f"Rejected: {item.prompt}"))

        Metrics collection:
            >>> metrics = {"approvals": 0, "rejections": 0}
            >>> session = InteractiveSession(model)
            >>> session.on("on_approve", lambda item: metrics.__setitem__("approvals", metrics["approvals"] + 1))
            >>> session.on("on_reject", lambda item: metrics.__setitem__("rejections", metrics["rejections"] + 1))

        External notification:
            >>> def notify_on_reject(item):
            ...     # Send Slack/email notification for rejected items
            ...     print(f"ALERT: Response rejected - {item.prompt[:50]}...")
            >>> session = InteractiveSession(model)
            >>> session.on("on_reject", notify_on_reject)
    """

    def __init__(
        self,
        model: Any,
        config: Optional[HITLConfig] = None,
        input_handler: Optional[InputHandler] = None,
    ):
        """Initialize an interactive session with event support.

        Args:
            model: The model to evaluate.
            config: Session configuration. If None, uses defaults.
            input_handler: Handler for user input. If None, auto-approves.

        Examples:
            Basic initialization:
                >>> session = InteractiveSession(model)

            With configuration:
                >>> config = HITLConfig(require_comment=True)
                >>> session = InteractiveSession(model, config=config)

            With callbacks pre-registered:
                >>> session = InteractiveSession(model)
                >>> session.on("on_generate", lambda item: print("Generated"))
        """
        super().__init__(model, config, input_handler)
        self._callbacks: dict[str, list[Callable]] = {
            "on_generate": [],
            "on_approve": [],
            "on_reject": [],
            "on_edit": [],
            "on_feedback": [],
        }

    def on(self, event: str, callback: Callable) -> None:
        """Register a callback for a specific event.

        Callbacks are called with the ReviewItem as the argument when the
        event occurs. Multiple callbacks can be registered for the same event.

        Args:
            event: Event name. One of: "on_generate", "on_approve",
                "on_reject", "on_edit", "on_feedback".
            callback: Function to call when event occurs. Receives a
                ReviewItem as its argument.

        Examples:
            Simple logging:
                >>> session.on("on_approve", lambda item: print(f"Approved: {item.item_id}"))

            Multiple callbacks:
                >>> session.on("on_generate", log_to_file)
                >>> session.on("on_generate", send_to_metrics)
                >>> session.on("on_generate", update_dashboard)

            Status-specific handling:
                >>> session.on("on_reject", escalate_to_supervisor)
                >>> session.on("on_edit", track_correction)

            With closure for context:
                >>> def make_counter(name):
                ...     count = [0]
                ...     def counter(item):
                ...         count[0] += 1
                ...         print(f"{name}: {count[0]}")
                ...     return counter
                >>> session.on("on_approve", make_counter("approvals"))
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Emit an event to all registered callbacks.

        Internal method that dispatches events to callbacks. Errors in
        callbacks are silently ignored to prevent disrupting the workflow.

        Args:
            event: The event name to emit.
            *args: Positional arguments to pass to callbacks.
            **kwargs: Keyword arguments to pass to callbacks.
        """
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                pass  # Ignore callback errors

    def generate_and_review(
        self,
        prompt: str,
        require_approval: bool = True,
        **kwargs: Any,
    ) -> tuple[str, ReviewItem]:
        """Generate a response with review and emit appropriate events.

        Extends the parent method to emit events after the review is complete.
        Always emits "on_generate", then emits a status-specific event.

        Args:
            prompt: The input prompt to send to the model.
            require_approval: If True, requests human approval.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            A tuple of (response, review_item).

        Examples:
            With event logging:
                >>> session.on("on_generate", lambda item: print("Generated"))
                >>> session.on("on_approve", lambda item: print("Approved"))
                >>> response, item = session.generate_and_review("Test")
                Generated
                Approved

            Tracking approvals vs rejections:
                >>> results = []
                >>> session.on("on_approve", lambda item: results.append(("approved", item)))
                >>> session.on("on_reject", lambda item: results.append(("rejected", item)))
                >>> response, item = session.generate_and_review("Test")

            Real-time dashboard update:
                >>> session.on("on_generate", update_dashboard)
                >>> for prompt in prompts:
                ...     response, item = session.generate_and_review(prompt)
        """
        response, item = super().generate_and_review(prompt, require_approval, **kwargs)

        self._emit("on_generate", item)

        if item.status == ReviewStatus.APPROVED:
            self._emit("on_approve", item)
        elif item.status == ReviewStatus.REJECTED:
            self._emit("on_reject", item)
        elif item.status == ReviewStatus.EDITED:
            self._emit("on_edit", item)

        return response, item


class ApprovalWorkflow:
    """Workflow for confidence-based approval of model outputs.

    ApprovalWorkflow implements a gated approval system where responses are
    either auto-approved (if confidence exceeds threshold) or sent for manual
    human review. This reduces reviewer burden while maintaining quality control.

    Attributes:
        model: The language model to use for generation.
        auto_approve_threshold: Confidence level required for auto-approval.
        confidence_func: Function that computes confidence scores.
        input_handler: Handler for manual approval requests.

    Examples:
        Basic usage with auto-approval:
            >>> from insideLLMs.agents.hitl import ApprovalWorkflow
            >>> workflow = ApprovalWorkflow(model, auto_approve_threshold=0.9)
            >>> response, approved, confidence = workflow.generate_with_approval("Test")
            >>> print(f"Approved: {approved}, Confidence: {confidence}")

        With custom confidence function:
            >>> def confidence_scorer(prompt, response):
            ...     # Higher confidence for longer, well-structured responses
            ...     if len(response) > 100 and "." in response:
            ...         return 0.95
            ...     return 0.5
            >>> workflow = ApprovalWorkflow(
            ...     model,
            ...     confidence_func=confidence_scorer,
            ...     auto_approve_threshold=0.9
            ... )

        Tracking approval statistics:
            >>> workflow = ApprovalWorkflow(model)
            >>> for prompt in prompts:
            ...     response, approved, conf = workflow.generate_with_approval(prompt)
            >>> stats = workflow.stats
            >>> print(f"Auto-approved: {stats['auto_approved']}")
            >>> print(f"Manual approved: {stats['manual_approved']}")
            >>> print(f"Rejected: {stats['rejected']}")

        Integration with moderation API:
            >>> def moderation_confidence(prompt, response):
            ...     # Call external moderation API
            ...     # score = moderation_api.score(response)
            ...     # return score
            ...     return 0.8
            >>> workflow = ApprovalWorkflow(
            ...     model,
            ...     confidence_func=moderation_confidence,
            ...     auto_approve_threshold=0.95
            ... )
    """

    def __init__(
        self,
        model: Any,
        auto_approve_threshold: float = 0.9,
        confidence_func: Optional[Callable[[str, str], float]] = None,
        input_handler: Optional[InputHandler] = None,
    ):
        """Initialize an approval workflow.

        Args:
            model: Model to use for generation. Must have generate() method.
            auto_approve_threshold: Confidence threshold (0.0-1.0) above which
                responses are automatically approved without human review.
            confidence_func: Optional function that takes (prompt, response) and
                returns a confidence score (0.0-1.0). If None, uses default 0.5.
            input_handler: Handler for manual approval when confidence is below
                threshold. If None, uses CallbackInputHandler (auto-approves).

        Examples:
            Default configuration:
                >>> workflow = ApprovalWorkflow(model)
                >>> # Uses 0.9 threshold, default 0.5 confidence

            High-confidence threshold:
                >>> workflow = ApprovalWorkflow(
                ...     model,
                ...     auto_approve_threshold=0.99
                ... )
                >>> # Almost everything goes to manual review

            With confidence function:
                >>> def scorer(prompt, response):
                ...     return 0.9 if len(response) > 50 else 0.3
                >>> workflow = ApprovalWorkflow(model, confidence_func=scorer)

            With custom handler for rejections:
                >>> handler = CallbackInputHandler(
                ...     approval_callback=lambda item: (False, "Needs revision")
                ... )
                >>> workflow = ApprovalWorkflow(model, input_handler=handler)
        """
        self.model = model
        self.auto_approve_threshold = auto_approve_threshold
        self.confidence_func = confidence_func
        self.input_handler = input_handler or CallbackInputHandler()
        self._stats = {"auto_approved": 0, "manual_approved": 0, "rejected": 0}

    def generate_with_approval(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> tuple[str, bool, float]:
        """Generate a response with confidence-based approval.

        Generates a response, computes confidence, and either auto-approves
        (if confidence >= threshold) or requests manual approval.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            A tuple of (response, approved, confidence) where:
            - response: The model's generated text.
            - approved: True if approved (auto or manual), False if rejected.
            - confidence: The computed confidence score (0.0-1.0).

        Examples:
            Basic generation:
                >>> response, approved, conf = workflow.generate_with_approval("Question")
                >>> if approved:
                ...     print(f"Using response (confidence: {conf:.2f})")

            Handling rejections:
                >>> response, approved, conf = workflow.generate_with_approval("Risky prompt")
                >>> if not approved:
                ...     print(f"Rejected at confidence {conf:.2f}")
                ...     # Fall back to alternative approach

            Batch processing:
                >>> results = []
                >>> for prompt in prompts:
                ...     response, approved, conf = workflow.generate_with_approval(prompt)
                ...     results.append({
                ...         "prompt": prompt,
                ...         "response": response if approved else None,
                ...         "confidence": conf
                ...     })

            Distinguishing auto vs manual approval:
                >>> workflow = ApprovalWorkflow(model, auto_approve_threshold=0.9)
                >>> response, approved, conf = workflow.generate_with_approval("Test")
                >>> if conf >= 0.9:
                ...     print("Auto-approved")
                ... elif approved:
                ...     print("Manually approved")
                ... else:
                ...     print("Rejected")
        """
        response = self.model.generate(prompt, **kwargs)

        # Compute confidence
        if self.confidence_func:
            confidence = self.confidence_func(prompt, response)
        else:
            confidence = 0.5  # Default confidence

        # Check for auto-approval
        if confidence >= self.auto_approve_threshold:
            self._stats["auto_approved"] += 1
            return response, True, confidence

        # Manual approval needed
        item = ReviewItem(prompt=prompt, response=response)
        approved, _ = self.input_handler.get_approval(item)

        if approved:
            self._stats["manual_approved"] += 1
        else:
            self._stats["rejected"] += 1

        return response, bool(approved), confidence

    @property
    def stats(self) -> dict[str, int]:
        """Get workflow statistics.

        Returns:
            Dictionary with counts for auto_approved, manual_approved, rejected.

        Examples:
            After processing:
                >>> stats = workflow.stats
                >>> print(f"Auto: {stats['auto_approved']}, Manual: {stats['manual_approved']}")

            Computing rates:
                >>> stats = workflow.stats
                >>> total = sum(stats.values())
                >>> auto_rate = stats['auto_approved'] / total if total else 0
                >>> print(f"Auto-approval rate: {auto_rate:.1%}")

            Quality metrics:
                >>> stats = workflow.stats
                >>> rejection_rate = stats['rejected'] / sum(stats.values())
                >>> if rejection_rate > 0.3:
                ...     print("Warning: High rejection rate")
        """
        return dict(self._stats)


class ReviewWorkflow:
    """Workflow for batched human review of model outputs.

    ReviewWorkflow manages the process of collecting model outputs, queuing
    them for review, and processing reviews in batches. This is ideal for
    scenarios where reviews happen asynchronously or in bulk.

    Attributes:
        queue: The review queue holding items awaiting review.
        batch_size: Number of items to retrieve per batch.

    Examples:
        Basic batch review workflow:
            >>> from insideLLMs.agents.hitl import ReviewWorkflow, Priority
            >>> workflow = ReviewWorkflow(batch_size=5)
            >>>
            >>> # Add items for review
            >>> for prompt, response in model_outputs:
            ...     workflow.add_for_review(prompt, response)
            >>>
            >>> # Get a batch for review
            >>> batch = workflow.get_batch()
            >>> print(f"Reviewing {len(batch)} items")

        Priority-based review:
            >>> from insideLLMs.agents.hitl import ReviewWorkflow, PriorityReviewQueue, Priority
            >>> queue = PriorityReviewQueue()
            >>> workflow = ReviewWorkflow(queue=queue)
            >>>
            >>> workflow.add_for_review("Critical task", "response", priority=Priority.CRITICAL)
            >>> workflow.add_for_review("Routine task", "response", priority=Priority.LOW)
            >>>
            >>> # Critical items come first
            >>> batch = workflow.get_batch()

        Submitting batch reviews:
            >>> workflow = ReviewWorkflow()
            >>> # ... add items ...
            >>> batch = workflow.get_batch()
            >>> reviews = [
            ...     (item.item_id, ReviewStatus.APPROVED, Feedback(content="OK"))
            ...     for item in batch
            ... ]
            >>> processed = workflow.submit_reviews(reviews)
            >>> print(f"Processed {processed} reviews")

        Async review pattern:
            >>> # Producer: add items as they're generated
            >>> def producer(model, prompts):
            ...     for prompt in prompts:
            ...         response = model.generate(prompt)
            ...         workflow.add_for_review(prompt, response)
            >>>
            >>> # Consumer: review in batches
            >>> def consumer():
            ...     while batch := workflow.get_batch():
            ...         reviews = process_batch(batch)
            ...         workflow.submit_reviews(reviews)
    """

    def __init__(
        self,
        queue: Optional[ReviewQueue] = None,
        batch_size: int = 10,
    ):
        """Initialize a batched review workflow.

        Args:
            queue: Review queue to use. If None, creates a standard ReviewQueue.
                Pass a PriorityReviewQueue for priority-based ordering.
            batch_size: Number of items to retrieve per get_batch() call.

        Examples:
            Default configuration:
                >>> workflow = ReviewWorkflow()
                >>> # Uses standard queue, batch size 10

            Custom batch size:
                >>> workflow = ReviewWorkflow(batch_size=50)
                >>> # For high-volume review sessions

            With priority queue:
                >>> queue = PriorityReviewQueue()
                >>> workflow = ReviewWorkflow(queue=queue, batch_size=20)

            With size-limited queue:
                >>> queue = ReviewQueue(max_size=1000)
                >>> workflow = ReviewWorkflow(queue=queue)
        """
        self.queue = queue or ReviewQueue()
        self.batch_size = batch_size

    def add_for_review(
        self,
        prompt: str,
        response: str,
        priority: Priority = Priority.MEDIUM,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ReviewItem:
        """Add a model output to the review queue.

        Args:
            prompt: The input prompt that generated the response.
            response: The model's generated response to be reviewed.
            priority: Priority level for review ordering.
            metadata: Additional metadata to attach to the review item.

        Returns:
            The created ReviewItem (also added to queue).

        Examples:
            Basic addition:
                >>> item = workflow.add_for_review(
                ...     prompt="Summarize this text",
                ...     response="This is a summary..."
                ... )
                >>> print(item.item_id)

            With priority:
                >>> item = workflow.add_for_review(
                ...     prompt="Safety-critical question",
                ...     response="...",
                ...     priority=Priority.CRITICAL
                ... )

            With metadata:
                >>> item = workflow.add_for_review(
                ...     prompt="Question",
                ...     response="Answer",
                ...     metadata={
                ...         "model": "gpt-4",
                ...         "temperature": 0.7,
                ...         "user_id": "user_123"
                ...     }
                ... )

            Bulk addition:
                >>> for prompt, response in outputs:
                ...     workflow.add_for_review(prompt, response)
                >>> print(f"Queued {len(workflow.queue)} items")
        """
        item = ReviewItem(
            prompt=prompt,
            response=response,
            priority=priority,
            metadata=metadata or {},
        )
        self.queue.add(item)
        return item

    def get_batch(self) -> list[ReviewItem]:
        """Get a batch of items for review.

        Retrieves up to batch_size items from the queue. Items are marked
        as IN_PROGRESS when retrieved.

        Returns:
            List of ReviewItems ready for review. May be fewer than batch_size
            if queue has fewer pending items.

        Examples:
            Processing a batch:
                >>> batch = workflow.get_batch()
                >>> for item in batch:
                ...     # Review each item
                ...     decision = review(item)

            Checking batch size:
                >>> batch = workflow.get_batch()
                >>> print(f"Got {len(batch)} items (requested {workflow.batch_size})")

            Draining queue:
                >>> while batch := workflow.get_batch():
                ...     print(f"Processing {len(batch)} items")
                ...     # ... process batch ...

            With progress tracking:
                >>> total = len(workflow.queue)
                >>> processed = 0
                >>> while batch := workflow.get_batch():
                ...     processed += len(batch)
                ...     print(f"Progress: {processed}/{total}")
        """
        items = []
        for _ in range(self.batch_size):
            item = self.queue.get_next()
            if item is None:
                break
            items.append(item)
        return items

    def submit_reviews(
        self,
        reviews: list[tuple[str, ReviewStatus, Optional[Feedback]]],
    ) -> int:
        """Submit a batch of completed reviews.

        Updates items in the queue with their review status and feedback.

        Args:
            reviews: List of tuples, each containing:
                - item_id: The ID of the reviewed item
                - status: The review decision (APPROVED, REJECTED, etc.)
                - feedback: Optional Feedback object with details

        Returns:
            Number of reviews successfully processed.

        Examples:
            Basic submission:
                >>> batch = workflow.get_batch()
                >>> reviews = []
                >>> for item in batch:
                ...     reviews.append((item.item_id, ReviewStatus.APPROVED, None))
                >>> processed = workflow.submit_reviews(reviews)

            With feedback:
                >>> reviews = [
                ...     (item.item_id, ReviewStatus.APPROVED, Feedback(
                ...         feedback_type=FeedbackType.RATING,
                ...         rating=0.9,
                ...         content="Good response"
                ...     ))
                ...     for item in batch
                ... ]
                >>> workflow.submit_reviews(reviews)

            Mixed decisions:
                >>> reviews = [
                ...     (batch[0].item_id, ReviewStatus.APPROVED, None),
                ...     (batch[1].item_id, ReviewStatus.REJECTED, Feedback(content="Inaccurate")),
                ...     (batch[2].item_id, ReviewStatus.EDITED, Feedback(
                ...         feedback_type=FeedbackType.EDIT,
                ...         edited_content="Corrected text"
                ...     )),
                ... ]
                >>> workflow.submit_reviews(reviews)

            Error handling:
                >>> processed = workflow.submit_reviews(reviews)
                >>> if processed < len(reviews):
                ...     print(f"Warning: Only {processed}/{len(reviews)} processed")
        """
        processed = 0
        for item_id, status, feedback in reviews:
            item = self.queue.get_by_id(item_id)
            if item:
                item.status = status
                if feedback:
                    item.add_feedback(feedback)
                self.queue.update(item)
                processed += 1
        return processed


class AnnotationWorkflow:
    """Workflow for collecting structured annotations on text data.

    AnnotationWorkflow provides a managed environment for collecting human
    annotations on text, supporting both document-level labels and span
    annotations. Useful for creating training data, quality assessment,
    and content classification.

    Attributes:
        labels: List of valid annotation labels.
        multi_label: Whether multiple labels can be applied per item.

    Examples:
        Sentiment annotation:
            >>> from insideLLMs.agents.hitl import AnnotationWorkflow
            >>> workflow = AnnotationWorkflow(labels=["positive", "negative", "neutral"])
            >>> item_id = workflow.add_for_annotation("Great product, highly recommend!")
            >>> workflow.annotate(item_id, "positive", annotator_id="ann1")
            >>> annotations = workflow.get_annotations(item_id)

        Named entity recognition:
            >>> workflow = AnnotationWorkflow(labels=["PERSON", "ORG", "LOC"])
            >>> text = "John works at Google in NYC"
            >>> item_id = workflow.add_for_annotation(text)
            >>> workflow.annotate(item_id, "PERSON", start_offset=0, end_offset=4)
            >>> workflow.annotate(item_id, "ORG", start_offset=14, end_offset=20)
            >>> workflow.annotate(item_id, "LOC", start_offset=24, end_offset=27)

        Multi-label classification:
            >>> workflow = AnnotationWorkflow(
            ...     labels=["informative", "well_written", "accurate"],
            ...     multi_label=True
            ... )
            >>> item_id = workflow.add_for_annotation("Model response text...")
            >>> workflow.annotate(item_id, "informative")
            >>> workflow.annotate(item_id, "well_written")

        Export for training:
            >>> data = workflow.export()
            >>> # Save as JSON for ML training
            >>> import json
            >>> with open("annotations.json", "w") as f:
            ...     json.dump(data, f)
    """

    def __init__(
        self,
        labels: list[str],
        multi_label: bool = False,
    ):
        """Initialize an annotation workflow.

        Args:
            labels: List of valid annotation labels. Only these labels can
                be used when annotating items.
            multi_label: If True, allows multiple labels per item. If False,
                each unique label can only be applied once per item.

        Examples:
            Binary classification:
                >>> workflow = AnnotationWorkflow(labels=["good", "bad"])

            Multi-class classification:
                >>> workflow = AnnotationWorkflow(
                ...     labels=["happy", "sad", "angry", "neutral"]
                ... )

            Multi-label tagging:
                >>> workflow = AnnotationWorkflow(
                ...     labels=["funny", "informative", "helpful", "creative"],
                ...     multi_label=True
                ... )

            NER task:
                >>> workflow = AnnotationWorkflow(
                ...     labels=["PERSON", "ORG", "LOC", "DATE", "MONEY"]
                ... )
        """
        self.labels = labels
        self.multi_label = multi_label
        self._items: dict[str, ReviewItem] = {}

    def add_for_annotation(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add text for annotation.

        Args:
            text: The text to be annotated.
            metadata: Optional metadata to associate with the text.

        Returns:
            The item ID that can be used for subsequent annotate() calls.

        Examples:
            Basic addition:
                >>> item_id = workflow.add_for_annotation("Text to annotate")
                >>> print(item_id)  # UUID string

            With metadata:
                >>> item_id = workflow.add_for_annotation(
                ...     "Model response text",
                ...     metadata={
                ...         "model": "gpt-4",
                ...         "prompt": "Original prompt",
                ...         "task": "summarization"
                ...     }
                ... )

            Batch addition:
                >>> item_ids = []
                >>> for text in texts:
                ...     item_id = workflow.add_for_annotation(text)
                ...     item_ids.append(item_id)
        """
        item = ReviewItem(
            response=text,
            metadata=metadata or {},
        )
        self._items[item.item_id] = item
        return item.item_id

    def annotate(
        self,
        item_id: str,
        label: str,
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
        annotator_id: Optional[str] = None,
    ) -> Optional[Annotation]:
        """Add an annotation to an item.

        Args:
            item_id: The ID of the item to annotate.
            label: The annotation label (must be in self.labels).
            start_offset: For span annotations, the start character position.
            end_offset: For span annotations, the end character position.
            annotator_id: Optional ID of the person creating this annotation.

        Returns:
            The created Annotation, or None if invalid (bad label, item not
            found, or multi-label constraint violated).

        Examples:
            Document-level annotation:
                >>> ann = workflow.annotate(item_id, "positive")
                >>> print(ann.label)
                positive

            Span annotation:
                >>> ann = workflow.annotate(
                ...     item_id,
                ...     "PERSON",
                ...     start_offset=0,
                ...     end_offset=10,
                ...     annotator_id="annotator_001"
                ... )
                >>> print(ann.text)  # Extracted span text

            Multiple annotators:
                >>> workflow.annotate(item_id, "positive", annotator_id="ann1")
                >>> workflow.annotate(item_id, "positive", annotator_id="ann2")
                >>> # With multi_label=False, same label blocked

            Invalid label handling:
                >>> result = workflow.annotate(item_id, "invalid_label")
                >>> print(result)  # None
        """
        if label not in self.labels:
            return None

        item = self._items.get(item_id)
        if not item:
            return None

        # Check multi-label constraint
        if not self.multi_label and item.annotations:
            existing_labels = {a.label for a in item.annotations}
            if label in existing_labels:
                return None

        annotation = Annotation(
            text=item.response[start_offset:end_offset]
            if start_offset and end_offset
            else item.response,
            label=label,
            start_offset=start_offset,
            end_offset=end_offset,
            annotator_id=annotator_id,
        )
        item.add_annotation(annotation)
        return annotation

    def get_annotations(self, item_id: str) -> list[Annotation]:
        """Get all annotations for an item.

        Args:
            item_id: The ID of the item.

        Returns:
            List of Annotations for the item, or empty list if not found.

        Examples:
            Retrieving annotations:
                >>> annotations = workflow.get_annotations(item_id)
                >>> for ann in annotations:
                ...     print(f"{ann.label}: {ann.text[:20]}...")

            Checking annotation count:
                >>> if len(workflow.get_annotations(item_id)) >= 3:
                ...     print("Enough annotations for consensus")
        """
        item = self._items.get(item_id)
        return item.annotations if item else []

    def export(self) -> list[dict[str, Any]]:
        """Export all annotations for analysis or training.

        Returns:
            List of dictionaries, each containing item_id, text, annotations,
            and metadata.

        Examples:
            Export to JSON:
                >>> data = workflow.export()
                >>> import json
                >>> with open("annotations.json", "w") as f:
                ...     json.dump(data, f, indent=2)

            Convert to DataFrame:
                >>> import pandas as pd
                >>> data = workflow.export()
                >>> df = pd.DataFrame(data)

            Filter by annotation count:
                >>> data = workflow.export()
                >>> annotated = [d for d in data if d["annotations"]]
                >>> print(f"{len(annotated)} items have annotations")
        """
        return [
            {
                "item_id": item.item_id,
                "text": item.response,
                "annotations": [a.to_dict() for a in item.annotations],
                "metadata": item.metadata,
            }
            for item in self._items.values()
        ]


class HumanValidator:
    """Validates model outputs with human feedback."""

    def __init__(
        self,
        validation_func: Optional[Callable[[str, str], bool]] = None,
        input_handler: Optional[InputHandler] = None,
    ):
        """Initialize human validator.

        Args:
            validation_func: Optional custom validation function
            input_handler: Handler for human validation
        """
        self.validation_func = validation_func
        self.input_handler = input_handler or CallbackInputHandler()
        self._validations: list[dict[str, Any]] = []

    def validate(
        self,
        prompt: str,
        response: str,
        criteria: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Validate a response.

        Args:
            prompt: Original prompt
            response: Model response
            criteria: Validation criteria

        Returns:
            Tuple of (is_valid, feedback)
        """
        item = ReviewItem(
            prompt=prompt,
            response=response,
            metadata={"criteria": criteria} if criteria else {},
        )

        # Use custom function if provided
        if self.validation_func:
            is_valid = self.validation_func(prompt, response)
            feedback = None
        else:
            is_valid, feedback = self.input_handler.get_approval(item)

        self._validations.append(
            {
                "prompt": prompt,
                "response": response,
                "is_valid": is_valid,
                "feedback": feedback,
                "criteria": criteria,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return bool(is_valid), feedback

    @property
    def validation_history(self) -> list[dict[str, Any]]:
        """Get validation history."""
        return list(self._validations)

    def accuracy(self) -> float:
        """Get validation accuracy (valid / total)."""
        if not self._validations:
            return 0.0
        valid = sum(1 for v in self._validations if v["is_valid"])
        return valid / len(self._validations)


class ConsensusValidator:
    """Validates using consensus from multiple reviewers."""

    def __init__(
        self,
        min_reviewers: int = 3,
        consensus_threshold: float = 0.66,
    ):
        """Initialize consensus validator.

        Args:
            min_reviewers: Minimum number of reviewers required
            consensus_threshold: Proportion needed for consensus
        """
        self.min_reviewers = min_reviewers
        self.consensus_threshold = consensus_threshold
        self._pending: dict[str, dict[str, Any]] = {}

    def create_validation_task(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a validation task for multiple reviewers.

        Args:
            prompt: Original prompt
            response: Model response
            metadata: Additional metadata

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        self._pending[task_id] = {
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
            "votes": [],
            "created_at": datetime.now(),
        }
        return task_id

    def submit_vote(
        self,
        task_id: str,
        is_valid: bool,
        reviewer_id: str,
        comment: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Submit a vote for a validation task.

        Args:
            task_id: Task to vote on
            is_valid: Validation vote
            reviewer_id: ID of reviewer
            comment: Optional comment

        Returns:
            Consensus result if reached, None otherwise
        """
        if task_id not in self._pending:
            return None

        task = self._pending[task_id]

        # Check for duplicate vote
        if any(v["reviewer_id"] == reviewer_id for v in task["votes"]):
            return None

        task["votes"].append(
            {
                "is_valid": is_valid,
                "reviewer_id": reviewer_id,
                "comment": comment,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Check for consensus
        if len(task["votes"]) >= self.min_reviewers:
            valid_votes = sum(1 for v in task["votes"] if v["is_valid"])
            total_votes = len(task["votes"])
            consensus_ratio = valid_votes / total_votes

            if consensus_ratio >= self.consensus_threshold:
                result = {
                    "task_id": task_id,
                    "consensus": True,
                    "is_valid": True,
                    "ratio": consensus_ratio,
                    "votes": task["votes"],
                }
                del self._pending[task_id]
                return result
            elif (1 - consensus_ratio) >= self.consensus_threshold:
                result = {
                    "task_id": task_id,
                    "consensus": True,
                    "is_valid": False,
                    "ratio": consensus_ratio,
                    "votes": task["votes"],
                }
                del self._pending[task_id]
                return result

        return None

    def get_pending_tasks(self) -> list[str]:
        """Get list of pending task IDs."""
        return list(self._pending.keys())


class FeedbackCollector:
    """Collects and aggregates feedback from multiple sources."""

    def __init__(self):
        """Initialize feedback collector."""
        self._feedback: dict[str, list[Feedback]] = {}
        self._lock = threading.Lock()

    def add_feedback(
        self,
        item_id: str,
        feedback: Feedback,
    ) -> None:
        """Add feedback for an item.

        Args:
            item_id: Item identifier
            feedback: Feedback to add
        """
        with self._lock:
            if item_id not in self._feedback:
                self._feedback[item_id] = []
            self._feedback[item_id].append(feedback)

    def get_feedback(self, item_id: str) -> list[Feedback]:
        """Get all feedback for an item."""
        return self._feedback.get(item_id, [])

    def aggregate_ratings(self, item_id: str) -> Optional[float]:
        """Get average rating for an item."""
        feedback_list = self._feedback.get(item_id, [])
        ratings = [f.rating for f in feedback_list if f.rating is not None]
        return sum(ratings) / len(ratings) if ratings else None

    def get_consensus_type(self, item_id: str) -> Optional[FeedbackType]:
        """Get most common feedback type for an item."""
        feedback_list = self._feedback.get(item_id, [])
        if not feedback_list:
            return None

        type_counts: dict[FeedbackType, int] = {}
        for f in feedback_list:
            type_counts[f.feedback_type] = type_counts.get(f.feedback_type, 0) + 1

        return max(type_counts.keys(), key=lambda k: type_counts[k])

    def export(self) -> dict[str, list[dict[str, Any]]]:
        """Export all collected feedback."""
        return {
            item_id: [f.to_dict() for f in feedback_list]
            for item_id, feedback_list in self._feedback.items()
        }


class AnnotationCollector:
    """Collects annotations with inter-annotator agreement tracking."""

    def __init__(self, labels: list[str]):
        """Initialize annotation collector.

        Args:
            labels: Valid annotation labels
        """
        self.labels = labels
        self._annotations: dict[str, list[Annotation]] = {}

    def add_annotation(
        self,
        item_id: str,
        annotation: Annotation,
    ) -> bool:
        """Add annotation for an item.

        Args:
            item_id: Item identifier
            annotation: Annotation to add

        Returns:
            True if added, False if invalid label
        """
        if annotation.label not in self.labels:
            return False

        if item_id not in self._annotations:
            self._annotations[item_id] = []
        self._annotations[item_id].append(annotation)
        return True

    def get_annotations(self, item_id: str) -> list[Annotation]:
        """Get all annotations for an item."""
        return self._annotations.get(item_id, [])

    def compute_agreement(self, item_id: str) -> Optional[float]:
        """Compute inter-annotator agreement for an item.

        Uses simple agreement ratio (matching labels / total pairs).
        """
        annotations = self._annotations.get(item_id, [])
        if len(annotations) < 2:
            return None

        labels = [a.label for a in annotations]
        total_pairs = len(labels) * (len(labels) - 1) / 2
        matching_pairs = 0

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    matching_pairs += 1

        return matching_pairs / total_pairs if total_pairs > 0 else 0.0

    def get_majority_label(self, item_id: str) -> Optional[str]:
        """Get majority label for an item."""
        annotations = self._annotations.get(item_id, [])
        if not annotations:
            return None

        label_counts: dict[str, int] = {}
        for a in annotations:
            label_counts[a.label] = label_counts.get(a.label, 0) + 1

        return max(label_counts.keys(), key=lambda k: label_counts[k])

    def export(self) -> dict[str, Any]:
        """Export all annotations with agreement scores."""
        return {
            item_id: {
                "annotations": [a.to_dict() for a in annotations],
                "agreement": self.compute_agreement(item_id),
                "majority_label": self.get_majority_label(item_id),
            }
            for item_id, annotations in self._annotations.items()
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_hitl_session(
    model: Any,
    auto_approve_threshold: float = 0.9,
    input_handler: Optional[InputHandler] = None,
) -> HITLSession:
    """Create a HITL session with default settings.

    Args:
        model: Model to evaluate
        auto_approve_threshold: Threshold for auto-approval
        input_handler: Handler for user input

    Returns:
        Configured HITL session
    """
    config = HITLConfig(auto_approve_threshold=auto_approve_threshold)
    return HITLSession(model, config, input_handler)


def quick_review(
    prompt: str,
    response: str,
    approval_callback: Callable[[ReviewItem], tuple[bool, Optional[str]]],
) -> tuple[bool, Optional[str]]:
    """Quick review of a single response.

    Args:
        prompt: Original prompt
        response: Model response
        approval_callback: Callback for approval decision

    Returns:
        Tuple of (approved, comment)
    """
    item = ReviewItem(prompt=prompt, response=response)
    return approval_callback(item)


def collect_feedback(
    items: list[tuple[str, str]],
    feedback_callback: Callable[[ReviewItem], Feedback],
) -> list[Feedback]:
    """Collect feedback for multiple items.

    Args:
        items: List of (prompt, response) tuples
        feedback_callback: Callback for feedback collection

    Returns:
        List of collected feedback
    """
    feedback_list = []
    for prompt, response in items:
        item = ReviewItem(prompt=prompt, response=response)
        feedback = feedback_callback(item)
        feedback_list.append(feedback)
    return feedback_list
