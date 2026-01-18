"""Human-in-the-Loop (HITL) module for interactive model evaluation and feedback.

This module provides tools for incorporating human feedback into LLM workflows:
- Interactive approval workflows
- Human validation and correction
- Feedback collection and aggregation
- Review queues with priority handling
- Annotation interfaces

Example:
    >>> from insideLLMs.hitl import HITLSession, ApprovalWorkflow
    >>> from insideLLMs import DummyModel
    >>>
    >>> model = DummyModel()
    >>> session = HITLSession(model)
    >>>
    >>> # With approval workflow
    >>> workflow = ApprovalWorkflow(model, auto_approve_threshold=0.9)
    >>> result = workflow.generate_with_approval("Write a summary")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)
import json
import threading
import time
import uuid
from queue import PriorityQueue, Empty

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
    """Types of human feedback."""

    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    FLAG = "flag"
    SKIP = "skip"
    RATING = "rating"
    COMMENT = "comment"
    CORRECTION = "correction"


class ReviewStatus(str, Enum):
    """Status of items in review."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
    FLAGGED = "flagged"
    SKIPPED = "skipped"
    EXPIRED = "expired"


class Priority(int, Enum):
    """Priority levels for review items."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class Feedback:
    """Represents human feedback on a model output."""

    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    feedback_type: FeedbackType = FeedbackType.COMMENT
    content: str = ""
    rating: Optional[float] = None  # 0.0 to 1.0
    edited_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    reviewer_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    def from_dict(cls, data: Dict[str, Any]) -> "Feedback":
        """Create from dictionary."""
        return cls(
            feedback_id=data.get("feedback_id", str(uuid.uuid4())),
            feedback_type=FeedbackType(data.get("feedback_type", "comment")),
            content=data.get("content", ""),
            rating=data.get("rating"),
            edited_content=data.get("edited_content"),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            reviewer_id=data.get("reviewer_id"),
        )


@dataclass
class Annotation:
    """Represents a human annotation on text."""

    annotation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    label: str = ""
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotator_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    """An item in the review queue."""

    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    response: str = ""
    model_id: Optional[str] = None
    status: ReviewStatus = ReviewStatus.PENDING
    priority: Priority = Priority.MEDIUM
    feedback: List[Feedback] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    expires_at: Optional[datetime] = None

    def __lt__(self, other: "ReviewItem") -> bool:
        """Compare by priority for queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at

    def add_feedback(self, feedback: Feedback) -> None:
        """Add feedback to this item."""
        self.feedback.append(feedback)
        self.updated_at = datetime.now()

    def add_annotation(self, annotation: Annotation) -> None:
        """Add annotation to this item."""
        self.annotations.append(annotation)
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    """Configuration for HITL sessions."""

    auto_approve_threshold: float = 0.9
    require_consensus: bool = False
    min_reviewers: int = 1
    timeout_seconds: Optional[float] = None
    allow_skip: bool = True
    allow_edit: bool = True
    require_comment: bool = False
    track_time: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    """Queue for managing items awaiting human review."""

    def __init__(self, max_size: Optional[int] = None):
        """Initialize review queue.

        Args:
            max_size: Maximum queue size (None for unlimited)
        """
        self._items: Dict[str, ReviewItem] = {}
        self._pending: List[str] = []
        self._lock = threading.Lock()
        self._max_size = max_size

    def add(self, item: ReviewItem) -> bool:
        """Add item to queue.

        Args:
            item: Review item to add

        Returns:
            True if added, False if queue full
        """
        with self._lock:
            if self._max_size and len(self._items) >= self._max_size:
                return False
            self._items[item.item_id] = item
            if item.status == ReviewStatus.PENDING:
                self._pending.append(item.item_id)
            return True

    def get_next(self) -> Optional[ReviewItem]:
        """Get next pending item from queue."""
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
        """Get item by ID."""
        return self._items.get(item_id)

    def update(self, item: ReviewItem) -> None:
        """Update item in queue."""
        with self._lock:
            self._items[item.item_id] = item

    def remove(self, item_id: str) -> Optional[ReviewItem]:
        """Remove item from queue."""
        with self._lock:
            item = self._items.pop(item_id, None)
            if item_id in self._pending:
                self._pending.remove(item_id)
            return item

    def get_pending(self) -> List[ReviewItem]:
        """Get all pending items."""
        with self._lock:
            return [
                self._items[item_id]
                for item_id in self._pending
                if item_id in self._items
            ]

    def get_by_status(self, status: ReviewStatus) -> List[ReviewItem]:
        """Get items by status."""
        return [item for item in self._items.values() if item.status == status]

    def __len__(self) -> int:
        """Get queue size."""
        return len(self._items)

    @property
    def pending_count(self) -> int:
        """Get count of pending items."""
        return len(self._pending)

    def stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        stats = {status.value: 0 for status in ReviewStatus}
        for item in self._items.values():
            stats[item.status.value] += 1
        return stats


class PriorityReviewQueue(ReviewQueue):
    """Review queue with priority ordering."""

    def __init__(self, max_size: Optional[int] = None):
        """Initialize priority queue."""
        super().__init__(max_size)
        self._priority_queue: PriorityQueue = PriorityQueue()

    def add(self, item: ReviewItem) -> bool:
        """Add item to queue with priority."""
        if not super().add(item):
            return False
        with self._lock:
            if item.status == ReviewStatus.PENDING:
                self._priority_queue.put((item.priority.value, item.created_at, item.item_id))
        return True

    def get_next(self) -> Optional[ReviewItem]:
        """Get highest priority pending item."""
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
    """Protocol for handling user input."""

    def get_approval(self, item: ReviewItem) -> Tuple[bool, Optional[str]]:
        """Get approval for item. Returns (approved, comment)."""
        ...

    def get_feedback(self, item: ReviewItem) -> Feedback:
        """Get feedback for item."""
        ...

    def get_edit(self, item: ReviewItem) -> str:
        """Get edited content for item."""
        ...


class ConsoleInputHandler:
    """Console-based input handler for HITL interactions."""

    def __init__(self, timeout: Optional[float] = None):
        """Initialize console handler.

        Args:
            timeout: Timeout in seconds for input (None for no timeout)
        """
        self.timeout = timeout

    def get_approval(self, item: ReviewItem) -> Tuple[bool, Optional[str]]:
        """Get approval from console input."""
        print(f"\n{'='*60}")
        print(f"Review Item: {item.item_id}")
        print(f"Prompt: {item.prompt}")
        print(f"Response: {item.response}")
        print(f"{'='*60}")

        response = input("Approve? (y/n/skip): ").strip().lower()
        comment = input("Comment (optional): ").strip() or None

        if response == 'y':
            return True, comment
        elif response == 'skip':
            return None, comment  # type: ignore
        return False, comment

    def get_feedback(self, item: ReviewItem) -> Feedback:
        """Get feedback from console input."""
        print(f"\n{'='*60}")
        print(f"Provide feedback for: {item.item_id}")
        print(f"Response: {item.response}")
        print(f"{'='*60}")

        rating_str = input("Rating (0-10, or Enter to skip): ").strip()
        rating = float(rating_str) / 10 if rating_str else None

        comment = input("Comment: ").strip()

        return Feedback(
            feedback_type=FeedbackType.RATING if rating else FeedbackType.COMMENT,
            content=comment,
            rating=rating,
        )

    def get_edit(self, item: ReviewItem) -> str:
        """Get edited content from console input."""
        print(f"\n{'='*60}")
        print(f"Edit response for: {item.item_id}")
        print(f"Current: {item.response}")
        print(f"{'='*60}")

        edited = input("New response (or Enter to keep): ").strip()
        return edited if edited else item.response


class CallbackInputHandler:
    """Callback-based input handler for programmatic HITL interactions."""

    def __init__(
        self,
        approval_callback: Optional[Callable[[ReviewItem], Tuple[bool, Optional[str]]]] = None,
        feedback_callback: Optional[Callable[[ReviewItem], Feedback]] = None,
        edit_callback: Optional[Callable[[ReviewItem], str]] = None,
    ):
        """Initialize callback handler.

        Args:
            approval_callback: Callback for approval decisions
            feedback_callback: Callback for feedback collection
            edit_callback: Callback for edit operations
        """
        self._approval_callback = approval_callback
        self._feedback_callback = feedback_callback
        self._edit_callback = edit_callback

    def get_approval(self, item: ReviewItem) -> Tuple[bool, Optional[str]]:
        """Get approval via callback."""
        if self._approval_callback:
            return self._approval_callback(item)
        return True, None  # Auto-approve if no callback

    def get_feedback(self, item: ReviewItem) -> Feedback:
        """Get feedback via callback."""
        if self._feedback_callback:
            return self._feedback_callback(item)
        return Feedback(feedback_type=FeedbackType.APPROVE)

    def get_edit(self, item: ReviewItem) -> str:
        """Get edit via callback."""
        if self._edit_callback:
            return self._edit_callback(item)
        return item.response  # Keep original if no callback


class HITLSession:
    """Interactive human-in-the-loop session for model evaluation."""

    def __init__(
        self,
        model: Any,
        config: Optional[HITLConfig] = None,
        input_handler: Optional[InputHandler] = None,
    ):
        """Initialize HITL session.

        Args:
            model: Model to evaluate
            config: Session configuration
            input_handler: Handler for user input
        """
        self.model = model
        self.config = config or HITLConfig()
        self.input_handler = input_handler or CallbackInputHandler()
        self.session_id = str(uuid.uuid4())
        self._history: List[ReviewItem] = []
        self._lock = threading.Lock()
        self._start_time = datetime.now()

    def generate_and_review(
        self,
        prompt: str,
        require_approval: bool = True,
        **kwargs: Any,
    ) -> Tuple[str, ReviewItem]:
        """Generate response and optionally get human review.

        Args:
            prompt: Input prompt
            require_approval: Whether to require human approval
            **kwargs: Additional arguments for model

        Returns:
            Tuple of (final_response, review_item)
        """
        # Generate response
        response = self.model.generate(prompt, **kwargs)

        # Create review item
        item = ReviewItem(
            prompt=prompt,
            response=response,
            model_id=getattr(self.model, 'model_id', None),
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
                item.add_feedback(Feedback(
                    feedback_type=FeedbackType.COMMENT,
                    content=comment,
                ))
        else:
            item.status = ReviewStatus.APPROVED

        with self._lock:
            self._history.append(item)

        return response, item

    def collect_feedback(self, prompt: str, **kwargs: Any) -> Tuple[str, Feedback]:
        """Generate response and collect feedback.

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments for model

        Returns:
            Tuple of (response, feedback)
        """
        response = self.model.generate(prompt, **kwargs)

        item = ReviewItem(
            prompt=prompt,
            response=response,
            model_id=getattr(self.model, 'model_id', None),
        )

        feedback = self.input_handler.get_feedback(item)
        item.add_feedback(feedback)
        item.status = ReviewStatus.APPROVED

        with self._lock:
            self._history.append(item)

        return response, feedback

    def edit_response(self, prompt: str, **kwargs: Any) -> Tuple[str, str]:
        """Generate response and allow human edit.

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments for model

        Returns:
            Tuple of (original_response, edited_response)
        """
        original = self.model.generate(prompt, **kwargs)

        item = ReviewItem(
            prompt=prompt,
            response=original,
            model_id=getattr(self.model, 'model_id', None),
        )

        edited = self.input_handler.get_edit(item)

        if edited != original:
            item.status = ReviewStatus.EDITED
            item.add_feedback(Feedback(
                feedback_type=FeedbackType.EDIT,
                edited_content=edited,
            ))
        else:
            item.status = ReviewStatus.APPROVED

        with self._lock:
            self._history.append(item)

        return original, edited

    @property
    def history(self) -> List[ReviewItem]:
        """Get session history."""
        return list(self._history)

    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        with self._lock:
            total = len(self._history)
            if total == 0:
                return {"total": 0, "duration_seconds": 0}

            approved = sum(1 for item in self._history if item.status == ReviewStatus.APPROVED)
            rejected = sum(1 for item in self._history if item.status == ReviewStatus.REJECTED)
            edited = sum(1 for item in self._history if item.status == ReviewStatus.EDITED)
            skipped = sum(1 for item in self._history if item.status == ReviewStatus.SKIPPED)

            ratings = [
                f.rating for item in self._history
                for f in item.feedback
                if f.rating is not None
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

    def export_history(self) -> List[Dict[str, Any]]:
        """Export session history as list of dictionaries."""
        return [item.to_dict() for item in self._history]


class InteractiveSession(HITLSession):
    """Extended HITL session with interactive features."""

    def __init__(
        self,
        model: Any,
        config: Optional[HITLConfig] = None,
        input_handler: Optional[InputHandler] = None,
    ):
        """Initialize interactive session."""
        super().__init__(model, config, input_handler)
        self._callbacks: Dict[str, List[Callable]] = {
            "on_generate": [],
            "on_approve": [],
            "on_reject": [],
            "on_edit": [],
            "on_feedback": [],
        }

    def on(self, event: str, callback: Callable) -> None:
        """Register event callback.

        Args:
            event: Event name (on_generate, on_approve, on_reject, on_edit, on_feedback)
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Emit event to callbacks."""
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
    ) -> Tuple[str, ReviewItem]:
        """Generate with event emission."""
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
    """Workflow for approval-based model interactions."""

    def __init__(
        self,
        model: Any,
        auto_approve_threshold: float = 0.9,
        confidence_func: Optional[Callable[[str, str], float]] = None,
        input_handler: Optional[InputHandler] = None,
    ):
        """Initialize approval workflow.

        Args:
            model: Model to use
            auto_approve_threshold: Confidence threshold for auto-approval
            confidence_func: Function to compute confidence (prompt, response) -> float
            input_handler: Handler for user input when manual approval needed
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
    ) -> Tuple[str, bool, float]:
        """Generate response with approval workflow.

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments for model

        Returns:
            Tuple of (response, approved, confidence)
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
    def stats(self) -> Dict[str, int]:
        """Get workflow statistics."""
        return dict(self._stats)


class ReviewWorkflow:
    """Workflow for batched review of model outputs."""

    def __init__(
        self,
        queue: Optional[ReviewQueue] = None,
        batch_size: int = 10,
    ):
        """Initialize review workflow.

        Args:
            queue: Review queue to use
            batch_size: Number of items per batch
        """
        self.queue = queue or ReviewQueue()
        self.batch_size = batch_size

    def add_for_review(
        self,
        prompt: str,
        response: str,
        priority: Priority = Priority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReviewItem:
        """Add item for review.

        Args:
            prompt: Input prompt
            response: Model response
            priority: Review priority
            metadata: Additional metadata

        Returns:
            Created review item
        """
        item = ReviewItem(
            prompt=prompt,
            response=response,
            priority=priority,
            metadata=metadata or {},
        )
        self.queue.add(item)
        return item

    def get_batch(self) -> List[ReviewItem]:
        """Get batch of items for review."""
        items = []
        for _ in range(self.batch_size):
            item = self.queue.get_next()
            if item is None:
                break
            items.append(item)
        return items

    def submit_reviews(
        self,
        reviews: List[Tuple[str, ReviewStatus, Optional[Feedback]]],
    ) -> int:
        """Submit batch of reviews.

        Args:
            reviews: List of (item_id, status, feedback) tuples

        Returns:
            Number of reviews processed
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
    """Workflow for collecting annotations on model outputs."""

    def __init__(
        self,
        labels: List[str],
        multi_label: bool = False,
    ):
        """Initialize annotation workflow.

        Args:
            labels: Available annotation labels
            multi_label: Whether multiple labels can be applied
        """
        self.labels = labels
        self.multi_label = multi_label
        self._items: Dict[str, ReviewItem] = {}

    def add_for_annotation(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add text for annotation.

        Args:
            text: Text to annotate
            metadata: Additional metadata

        Returns:
            Item ID
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
        """Add annotation to item.

        Args:
            item_id: Item to annotate
            label: Annotation label
            start_offset: Start position for span annotation
            end_offset: End position for span annotation
            annotator_id: ID of annotator

        Returns:
            Created annotation or None if invalid
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
            text=item.response[start_offset:end_offset] if start_offset and end_offset else item.response,
            label=label,
            start_offset=start_offset,
            end_offset=end_offset,
            annotator_id=annotator_id,
        )
        item.add_annotation(annotation)
        return annotation

    def get_annotations(self, item_id: str) -> List[Annotation]:
        """Get annotations for item."""
        item = self._items.get(item_id)
        return item.annotations if item else []

    def export(self) -> List[Dict[str, Any]]:
        """Export all annotations."""
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
        self._validations: List[Dict[str, Any]] = []

    def validate(
        self,
        prompt: str,
        response: str,
        criteria: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
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

        self._validations.append({
            "prompt": prompt,
            "response": response,
            "is_valid": is_valid,
            "feedback": feedback,
            "criteria": criteria,
            "timestamp": datetime.now().isoformat(),
        })

        return bool(is_valid), feedback

    @property
    def validation_history(self) -> List[Dict[str, Any]]:
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
        self._pending: Dict[str, Dict[str, Any]] = {}

    def create_validation_task(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
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
    ) -> Optional[Dict[str, Any]]:
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

        task["votes"].append({
            "is_valid": is_valid,
            "reviewer_id": reviewer_id,
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
        })

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

    def get_pending_tasks(self) -> List[str]:
        """Get list of pending task IDs."""
        return list(self._pending.keys())


class FeedbackCollector:
    """Collects and aggregates feedback from multiple sources."""

    def __init__(self):
        """Initialize feedback collector."""
        self._feedback: Dict[str, List[Feedback]] = {}
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

    def get_feedback(self, item_id: str) -> List[Feedback]:
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

        type_counts: Dict[FeedbackType, int] = {}
        for f in feedback_list:
            type_counts[f.feedback_type] = type_counts.get(f.feedback_type, 0) + 1

        return max(type_counts, key=type_counts.get)  # type: ignore

    def export(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export all collected feedback."""
        return {
            item_id: [f.to_dict() for f in feedback_list]
            for item_id, feedback_list in self._feedback.items()
        }


class AnnotationCollector:
    """Collects annotations with inter-annotator agreement tracking."""

    def __init__(self, labels: List[str]):
        """Initialize annotation collector.

        Args:
            labels: Valid annotation labels
        """
        self.labels = labels
        self._annotations: Dict[str, List[Annotation]] = {}

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

    def get_annotations(self, item_id: str) -> List[Annotation]:
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

        label_counts: Dict[str, int] = {}
        for a in annotations:
            label_counts[a.label] = label_counts.get(a.label, 0) + 1

        return max(label_counts, key=label_counts.get)  # type: ignore

    def export(self) -> Dict[str, Any]:
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
    approval_callback: Callable[[ReviewItem], Tuple[bool, Optional[str]]],
) -> Tuple[bool, Optional[str]]:
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
    items: List[Tuple[str, str]],
    feedback_callback: Callable[[ReviewItem], Feedback],
) -> List[Feedback]:
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
