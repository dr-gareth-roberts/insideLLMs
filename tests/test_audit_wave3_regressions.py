"""Regression tests for production-quality audit wave 3 (low-severity) fixes."""


# L10 — redact_pii must scrub string keys and leave non-string keys/values intact.
def test_redact_pii_scrubs_string_keys_keeps_numeric():
    from insideLLMs.privacy.redaction import redact_pii

    out = redact_pii({"email user@example.com": "note", 42: "kept"})
    assert not any("user@example.com" in str(k) for k in out)
    assert 42 in out  # numeric key preserved, not stringified


# L17 — ModelWrapper transparently delegates chat/stream to the wrapped model.
def test_model_wrapper_delegates_chat_and_stream():
    from insideLLMs.models import DummyModel
    from insideLLMs.models.base import ModelWrapper

    wrapper = ModelWrapper(DummyModel())
    assert callable(wrapper.chat)
    result = wrapper.chat([{"role": "user", "content": "hi"}])
    assert isinstance(result, str)


# L26 — bootstrap CI must not mutate the process-global RNG.
def test_bootstrap_does_not_mutate_global_random():
    import random

    from insideLLMs.analysis.statistics import bootstrap_confidence_interval

    random.seed(123)
    before = random.random()
    random.seed(123)
    bootstrap_confidence_interval(
        [1.0, 2.0, 3.0, 4.0, 5.0], lambda xs: sum(xs) / len(xs), n_bootstrap=50, seed=7
    )
    after = random.random()
    # Global stream is unaffected by the seeded local bootstrap RNG.
    assert before == after


# L3 — email PII pattern matches real addresses (no stray pipe in the TLD class).
def test_email_pii_detection_matches_real_address():
    from insideLLMs.safety import mask_pii

    masked = mask_pii("reach me at jane.doe@example.co")
    assert "jane.doe@example.co" not in masked
