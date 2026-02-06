import pytest

import insideLLMs.template_versioning as tv
from insideLLMs.template_versioning import (
    ABTest,
    ABTestRunner,
    ABTestStatus,
    ABVariant,
    AllocationStrategy,
    TemplateVersion,
    TemplateVersionManager,
    VersionStatus,
)


def _template(version_id: str, number: str = "1.0.0") -> TemplateVersion:
    return TemplateVersion(
        version_id=version_id,
        template_name="tmpl",
        content="Hello {name}",
        version_number=number,
        status=VersionStatus.ACTIVE,
        created_at="2024-01-01T00:00:00",
    )


def _variant(variant_id: str, name: str) -> ABVariant:
    return ABVariant(
        variant_id=variant_id, name=name, template_version=_template(f"t-{variant_id}")
    )


def _abtest(
    *,
    strategy: AllocationStrategy = AllocationStrategy.RANDOM,
    status: ABTestStatus = ABTestStatus.PENDING,
    confidence_threshold: float = 0.95,
) -> ABTest:
    return ABTest(
        test_id="test-1",
        name="Experiment",
        variants=[_variant("a", "A"), _variant("b", "B")],
        strategy=strategy,
        min_samples=1,
        confidence_threshold=confidence_threshold,
        status=status,
    )


def test_abvariant_score_properties_and_to_dict_empty_score_branch():
    variant = _variant("v1", "Variant A")
    assert variant.score_variance == 0.0
    assert variant.score_std == 0.0
    payload = variant.to_dict()
    assert payload["variant_id"] == "v1"
    assert payload["avg_score"] == 0.0


def test_manager_create_version_missing_template_and_latest_fallback_and_lookups():
    manager = TemplateVersionManager()

    # Missing template in create_version delegates to create_template.
    created = manager.create_version("new-template", "content")
    assert created.version_number == "1.0.0"

    # Force active id to point to missing entry so latest version fallback path is used.
    manager.create_template("tmpl", "v1")
    v2 = manager.create_version("tmpl", "v2")
    manager.active_versions["tmpl"] = "missing-active-id"
    v3 = manager.create_version("tmpl", "v3")
    assert v3.parent_version == v2.version_id

    with pytest.raises(ValueError, match="not found"):
        manager.activate_version("tmpl", "missing-version")

    assert manager.get_version("tmpl", v2.version_id) is v2

    assert manager.get_version_by_number("missing", "1.0.0") is None
    assert manager.get_version_by_number("tmpl", "1.0.1") is v2
    assert manager.get_version_by_number("tmpl", "9.9.9") is None

    assert manager.list_versions("missing") == []
    assert sorted(manager.list_templates()) == ["new-template", "tmpl"]


def test_manager_rollback_archive_diff_export_import_error_paths():
    manager = TemplateVersionManager()

    with pytest.raises(ValueError, match="Template 'missing' not found"):
        manager.rollback("missing")

    v1 = manager.create_template("tmpl", "v1")
    v2 = manager.create_version("tmpl", "v2")
    manager.activate_version("tmpl", v2.version_id)

    with pytest.raises(ValueError, match="Version missing-version not found"):
        manager.rollback("tmpl", to_version_id="missing-version")

    # No active version branch.
    del manager.active_versions["tmpl"]
    with pytest.raises(ValueError, match="No active version"):
        manager.rollback("tmpl")

    with pytest.raises(ValueError, match="not found"):
        manager.archive_version("tmpl", "missing")

    with pytest.raises(ValueError, match="Template 'missing' not found"):
        manager.diff_versions("missing", "a", "b")

    with pytest.raises(ValueError, match="One or both versions not found"):
        manager.diff_versions("tmpl", v1.version_id, "missing")

    with pytest.raises(ValueError, match="Template 'missing' not found"):
        manager.export_template("missing")

    with pytest.raises(ValueError, match="Version missing not found"):
        manager.export_template("tmpl", version_id="missing")

    # Import existing template without overwrite should fail.
    exported = manager.export_template("tmpl")
    with pytest.raises(ValueError, match="already exists"):
        manager.import_template(exported, overwrite=False)

    # Overwrite branch clears existing template and active version entries.
    manager.import_template(exported, overwrite=True)
    assert "tmpl" in manager.templates


def test_abtestrunner_get_test_and_status_filtered_listing():
    runner = ABTestRunner()
    t = _template("v1")
    created = runner.create_test("My test", [("A", t, 1.0)], description="d")

    assert runner.get_test(created.test_id) is created
    assert runner.get_test("missing") is None

    assert runner.list_tests(status=ABTestStatus.PENDING) == [created]
    created.start()
    assert runner.list_tests(status=ABTestStatus.RUNNING) == [created]


def test_abtest_invalid_state_transitions_selection_and_recording_errors():
    test = _abtest()

    test.status = ABTestStatus.RUNNING
    with pytest.raises(ValueError, match="Cannot start"):
        test.start()

    test.status = ABTestStatus.PENDING
    with pytest.raises(ValueError, match="Cannot pause"):
        test.pause()

    test.status = ABTestStatus.RUNNING
    with pytest.raises(ValueError, match="Cannot resume"):
        test.resume()

    test.status = ABTestStatus.PENDING
    with pytest.raises(ValueError, match="Cannot stop"):
        test.stop()

    test.status = ABTestStatus.PAUSED
    with pytest.raises(ValueError, match="Cannot select variant"):
        test.select_variant()

    test.status = ABTestStatus.RUNNING
    with pytest.raises(ValueError, match="Variant missing not found"):
        test.record_result("missing", score=0.5)

    # Unknown strategy falls back to random choice branch.
    test.strategy = "unknown"  # type: ignore[assignment]
    selected = test.select_variant()
    assert selected.variant_id in {"a", "b"}


def test_abtest_z_score_edge_cases_recommendations_and_to_dict():
    test = _abtest(status=ABTestStatus.RUNNING, confidence_threshold=0.8)

    # n=0 branch
    assert test._calculate_z_score(0.5, 0.5, 0, 10) == 0.0
    # pooled proportion 0 or 1 branch
    assert test._calculate_z_score(0.0, 0.0, 10, 10) == 0.0
    assert test._calculate_z_score(1.0, 1.0, 10, 10) == 0.0
    # se == 0 branch via floating-point underflow in the standard error term.
    huge_n = 10**307
    assert test._calculate_z_score(1e-307, 1e-307, huge_n, huge_n) == 0.0

    # Record enough data for recommendation generation.
    for _ in range(5):
        test.record_result("a", score=0.9, converted=True)
        test.record_result("b", score=0.4, converted=False)

    # Winner recommendation branch (>= confidence threshold)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(test, "_z_to_confidence", lambda _z: 0.95)
        result = test.get_results()
    assert any("winner" in rec.lower() for rec in result.recommendations)
    assert any("promoting" in rec.lower() for rec in result.recommendations)

    # Leading-but-inconclusive branch (>=0.9 but < threshold)
    high_threshold_test = _abtest(status=ABTestStatus.RUNNING, confidence_threshold=0.99)
    for _ in range(5):
        high_threshold_test.record_result("a", score=0.9, converted=True)
        high_threshold_test.record_result("b", score=0.4, converted=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(high_threshold_test, "_z_to_confidence", lambda _z: 0.92)
        high_result = high_threshold_test.get_results()
    assert any("leading" in rec.lower() for rec in high_result.recommendations)

    payload = high_threshold_test.to_dict()
    assert payload["test_id"] == "test-1"
    assert payload["strategy"] in {
        AllocationStrategy.RANDOM.value,
        AllocationStrategy.ROUND_ROBIN.value,
        AllocationStrategy.WEIGHTED.value,
        AllocationStrategy.MULTI_ARMED_BANDIT.value,
    }


def test_get_default_manager_initialization_branch():
    original = tv._default_manager
    try:
        tv._default_manager = None
        manager = tv.get_default_manager()
        assert isinstance(manager, TemplateVersionManager)
    finally:
        tv._default_manager = original
