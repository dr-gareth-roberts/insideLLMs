"""Additional branch coverage for HITL edge paths."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from insideLLMs.hitl import (
    Annotation,
    AnnotationCollector,
    AnnotationWorkflow,
    CallbackInputHandler,
    ConsensusValidator,
    ConsoleInputHandler,
    Feedback,
    FeedbackCollector,
    FeedbackType,
    HITLSession,
    InteractiveSession,
    Priority,
    PriorityReviewQueue,
    ReviewItem,
    ReviewStatus,
    ReviewWorkflow,
)


def test_review_item_lt_uses_created_time_for_equal_priority():
    earlier = ReviewItem(prompt="p1", response="r1", priority=Priority.MEDIUM)
    later = ReviewItem(prompt="p2", response="r2", priority=Priority.MEDIUM)
    earlier.created_at = datetime.now() - timedelta(seconds=1)
    later.created_at = datetime.now()
    assert earlier < later


def test_priority_queue_add_capacity_failure_and_empty_exception_branch(monkeypatch):
    full_queue = PriorityReviewQueue(max_size=1)
    assert full_queue.add(ReviewItem(prompt="p1", response="r1")) is True
    assert full_queue.add(ReviewItem(prompt="p2", response="r2")) is False

    queue = PriorityReviewQueue()
    monkeypatch.setattr(queue._priority_queue, "empty", lambda: False)

    def _raise_empty():
        raise Exception("wrapped empty")

    # Raise stdlib queue.Empty path by delegating to get_nowait on an actually empty queue.
    monkeypatch.setattr(queue._priority_queue, "get_nowait", lambda: (_ for _ in ()).throw(__import__("queue").Empty()))
    assert queue.get_next() is None


def test_console_input_handler_approval_feedback_and_edit_paths(monkeypatch):
    item = ReviewItem(prompt="prompt", response="original")
    handler = ConsoleInputHandler(timeout=10.0)
    assert handler.timeout == 10.0

    approvals = iter(["y", "looks good", "skip", "", "n", "needs work"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(approvals))
    assert handler.get_approval(item) == (True, "looks good")
    assert handler.get_approval(item) == (None, None)
    assert handler.get_approval(item) == (False, "needs work")

    feedback_inputs = iter(["8", "nice", "", "comment only"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(feedback_inputs))
    rated = handler.get_feedback(item)
    assert rated.feedback_type == FeedbackType.RATING
    assert rated.rating == 0.8

    comment_only = handler.get_feedback(item)
    assert comment_only.feedback_type == FeedbackType.COMMENT

    edit_inputs = iter(["updated", ""])
    monkeypatch.setattr("builtins.input", lambda _p="": next(edit_inputs))
    assert handler.get_edit(item) == "updated"
    assert handler.get_edit(item) == "original"


def test_callback_input_handler_default_feedback_path():
    handler = CallbackInputHandler()
    item = ReviewItem(prompt="p", response="r")
    feedback = handler.get_feedback(item)
    assert feedback.feedback_type == FeedbackType.APPROVE


def test_interactive_session_emit_ignores_callback_errors():
    class DummyModel:
        def generate(self, _prompt: str, **_kwargs):
            return "response"

    session = InteractiveSession(DummyModel())
    session.on("on_generate", lambda _item: (_ for _ in ()).throw(RuntimeError("boom")))
    session._emit("on_generate", ReviewItem(prompt="p", response="r"))


def test_interactive_session_rejected_and_edited_event_branches(monkeypatch):
    class DummyModel:
        def generate(self, _prompt: str, **_kwargs):
            return "response"

    session = InteractiveSession(DummyModel())
    events: list[str] = []
    session.on("on_reject", lambda _item: events.append("reject"))
    session.on("on_edit", lambda _item: events.append("edit"))

    def fake_rejected(self, prompt: str, require_approval: bool = True, **kwargs):
        item = ReviewItem(prompt=prompt, response="r", status=ReviewStatus.REJECTED)
        return "r", item

    def fake_edited(self, prompt: str, require_approval: bool = True, **kwargs):
        item = ReviewItem(prompt=prompt, response="r", status=ReviewStatus.EDITED)
        return "r", item

    monkeypatch.setattr(HITLSession, "generate_and_review", fake_rejected)
    session.generate_and_review("p")
    monkeypatch.setattr(HITLSession, "generate_and_review", fake_edited)
    session.generate_and_review("p")
    assert events == ["reject", "edit"]


def test_review_workflow_batch_break_and_submit_missing_item_branch():
    workflow = ReviewWorkflow(batch_size=3)
    workflow.add_for_review("p1", "r1")
    batch = workflow.get_batch()
    assert len(batch) == 1

    processed = workflow.submit_reviews(
        [
            (batch[0].item_id, ReviewStatus.APPROVED, None),
            ("missing-id", ReviewStatus.REJECTED, None),
        ]
    )
    assert processed == 1


def test_annotation_workflow_missing_item_nonduplicate_and_get_annotations_missing():
    workflow = AnnotationWorkflow(labels=["a", "b"], multi_label=False)
    assert workflow.annotate("missing", "a") is None

    item_id = workflow.add_for_annotation("text")
    assert workflow.annotate(item_id, "a") is not None
    second_label = workflow.annotate(item_id, "b")
    assert second_label is not None
    assert workflow.get_annotations("unknown-id") == []


def test_consensus_validator_missing_task_and_no_consensus_path():
    validator = ConsensusValidator(min_reviewers=3, consensus_threshold=0.8)
    assert validator.submit_vote("missing", True, "r1") is None

    task_id = validator.create_validation_task("prompt", "response")
    assert validator.submit_vote(task_id, True, "r1") is None
    assert validator.submit_vote(task_id, False, "r2") is None
    # Reaches min_reviewers but does not meet either consensus threshold.
    assert validator.submit_vote(task_id, True, "r3") is None


def test_feedback_and_annotation_collectors_empty_consensus_paths():
    feedback_collector = FeedbackCollector()
    assert feedback_collector.get_consensus_type("missing") is None

    annotation_collector = AnnotationCollector(labels=["x", "y"])
    assert annotation_collector.get_majority_label("missing") is None
    assert annotation_collector.compute_agreement("missing") is None

    annotation_collector.add_annotation("item", Annotation(label="x"))
    annotation_collector.add_annotation("item", Annotation(label="y"))
    assert annotation_collector.compute_agreement("item") == 0.0
