"""Tests for template versioning and A/B testing utilities."""

import pytest

from insideLLMs.prompts.template_versioning import (
    ABTest,
    ABTestResult,
    ABTestRunner,
    ABTestStatus,
    ABVariant,
    AllocationStrategy,
    TemplateChange,
    TemplateExperiment,
    TemplateVersion,
    TemplateVersionManager,
    VersionStatus,
    activate_version,
    create_ab_test,
    create_template,
    create_version,
    diff_template_versions,
    get_active_template,
    list_template_versions,
    rollback_template,
    run_template_comparison,
    set_default_manager,
)


class TestVersionStatus:
    """Tests for VersionStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        assert VersionStatus.DRAFT
        assert VersionStatus.ACTIVE
        assert VersionStatus.DEPRECATED
        assert VersionStatus.ARCHIVED


class TestABTestStatus:
    """Tests for ABTestStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        assert ABTestStatus.PENDING
        assert ABTestStatus.RUNNING
        assert ABTestStatus.PAUSED
        assert ABTestStatus.COMPLETED
        assert ABTestStatus.CANCELLED


class TestAllocationStrategy:
    """Tests for AllocationStrategy enum."""

    def test_all_strategies_exist(self):
        """Test all expected strategies exist."""
        assert AllocationStrategy.RANDOM
        assert AllocationStrategy.ROUND_ROBIN
        assert AllocationStrategy.WEIGHTED
        assert AllocationStrategy.MULTI_ARMED_BANDIT


class TestTemplateChange:
    """Tests for TemplateChange dataclass."""

    def test_basic_creation(self):
        """Test basic change creation."""
        change = TemplateChange(
            timestamp="2024-01-01T00:00:00",
            change_type="create",
            description="Initial creation",
        )
        assert change.change_type == "create"
        assert change.description == "Initial creation"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        change = TemplateChange(
            timestamp="2024-01-01T00:00:00",
            change_type="update",
            description="Updated prompt",
            author="test_user",
        )
        d = change.to_dict()
        assert d["change_type"] == "update"
        assert d["author"] == "test_user"


class TestTemplateVersion:
    """Tests for TemplateVersion dataclass."""

    def test_basic_creation(self):
        """Test basic version creation."""
        version = TemplateVersion(
            version_id="abc123",
            template_name="test_template",
            content="Hello {name}!",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )
        assert version.version_id == "abc123"
        assert version.version_number == "1.0.0"
        assert version.is_active

    def test_content_hash(self):
        """Test content hash generation."""
        version = TemplateVersion(
            version_id="abc123",
            template_name="test",
            content="Hello world",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )
        assert len(version.content_hash) == 12

        # Same content should produce same hash
        version2 = TemplateVersion(
            version_id="def456",
            template_name="test",
            content="Hello world",
            version_number="1.0.1",
            status=VersionStatus.DRAFT,
            created_at="2024-01-02T00:00:00",
        )
        assert version.content_hash == version2.content_hash

    def test_to_dict(self):
        """Test conversion to dictionary."""
        version = TemplateVersion(
            version_id="abc123",
            template_name="test",
            content="Hello {name}!",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
            variables=["name"],
            description="Test template",
        )
        d = version.to_dict()
        assert d["version_id"] == "abc123"
        assert d["status"] == "active"
        assert "name" in d["variables"]


class TestABVariant:
    """Tests for ABVariant dataclass."""

    def test_basic_creation(self):
        """Test basic variant creation."""
        template = TemplateVersion(
            version_id="abc123",
            template_name="test",
            content="Hello!",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )
        variant = ABVariant(
            variant_id="var1",
            name="Control",
            template_version=template,
        )
        assert variant.name == "Control"
        assert variant.impressions == 0

    def test_record_impression(self):
        """Test recording impressions."""
        template = TemplateVersion(
            version_id="abc123",
            template_name="test",
            content="Hello!",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )
        variant = ABVariant(
            variant_id="var1",
            name="Control",
            template_version=template,
        )
        variant.record_impression(0.8, converted=True)
        variant.record_impression(0.6, converted=False)

        assert variant.impressions == 2
        assert variant.conversions == 1
        assert variant.conversion_rate == 0.5
        assert variant.avg_score == 0.7

    def test_score_variance(self):
        """Test score variance calculation."""
        template = TemplateVersion(
            version_id="abc123",
            template_name="test",
            content="Hello!",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )
        variant = ABVariant(
            variant_id="var1",
            name="Control",
            template_version=template,
        )
        variant.record_impression(0.5)
        variant.record_impression(0.5)
        assert variant.score_variance == 0.0

        variant.record_impression(0.8)
        assert variant.score_variance > 0


class TestTemplateVersionManager:
    """Tests for TemplateVersionManager."""

    def test_create_template(self):
        """Test creating a new template."""
        manager = TemplateVersionManager()
        version = manager.create_template(
            name="greeting",
            content="Hello {name}!",
            description="A greeting template",
        )
        assert version.template_name == "greeting"
        assert version.version_number == "1.0.0"
        assert version.is_active
        assert "name" in version.variables

    def test_create_template_duplicate_raises(self):
        """Test that creating duplicate template raises error."""
        manager = TemplateVersionManager()
        manager.create_template("greeting", "Hello!")
        with pytest.raises(ValueError, match="already exists"):
            manager.create_template("greeting", "Hi!")

    def test_create_version(self):
        """Test creating new version of template."""
        manager = TemplateVersionManager()
        v1 = manager.create_template("greeting", "Hello {name}!")
        v2 = manager.create_version(
            "greeting",
            "Hi {name}, welcome!",
            description="More friendly",
        )
        assert v2.version_number == "1.0.1"
        assert v2.parent_version == v1.version_id
        assert v2.status == VersionStatus.DRAFT

    def test_create_version_with_level(self):
        """Test version increment levels."""
        manager = TemplateVersionManager()
        manager.create_template("test", "v1")

        v2 = manager.create_version("test", "v2", version_level="patch")
        assert v2.version_number == "1.0.1"

        v3 = manager.create_version("test", "v3", version_level="minor")
        assert v3.version_number == "1.1.0"

        v4 = manager.create_version("test", "v4", version_level="major")
        assert v4.version_number == "2.0.0"

    def test_activate_version(self):
        """Test activating a version."""
        manager = TemplateVersionManager()
        v1 = manager.create_template("greeting", "Hello!")
        v2 = manager.create_version("greeting", "Hi there!")

        assert v2.status == VersionStatus.DRAFT
        manager.activate_version("greeting", v2.version_id)
        assert v2.status == VersionStatus.ACTIVE
        assert v1.status == VersionStatus.DEPRECATED

    def test_get_active_version(self):
        """Test getting active version."""
        manager = TemplateVersionManager()
        v1 = manager.create_template("greeting", "Hello!")
        active = manager.get_active_version("greeting")
        assert active.version_id == v1.version_id

    def test_rollback(self):
        """Test rolling back to previous version."""
        manager = TemplateVersionManager()
        v1 = manager.create_template("greeting", "Hello!")
        v2 = manager.create_version("greeting", "Hi!")
        manager.activate_version("greeting", v2.version_id)

        rolled_back = manager.rollback("greeting")
        assert rolled_back.version_id == v1.version_id
        assert rolled_back.is_active

    def test_rollback_to_specific_version(self):
        """Test rolling back to specific version."""
        manager = TemplateVersionManager()
        v1 = manager.create_template("greeting", "Hello!")
        v2 = manager.create_version("greeting", "Hi!")
        v3 = manager.create_version("greeting", "Hey!")
        manager.activate_version("greeting", v2.version_id)
        manager.activate_version("greeting", v3.version_id)

        rolled_back = manager.rollback("greeting", v1.version_id)
        assert rolled_back.version_id == v1.version_id

    def test_list_versions(self):
        """Test listing versions."""
        manager = TemplateVersionManager()
        manager.create_template("greeting", "Hello!")
        v2 = manager.create_version("greeting", "Hi!")
        # Activate v2 so next version increments from it
        manager.activate_version("greeting", v2.version_id)
        manager.create_version("greeting", "Hey!")

        versions = manager.list_versions("greeting")
        assert len(versions) == 3
        # Should be sorted by version number descending
        assert versions[0].version_number == "1.0.2"

    def test_archive_version(self):
        """Test archiving a version."""
        manager = TemplateVersionManager()
        v1 = manager.create_template("greeting", "Hello!")
        v2 = manager.create_version("greeting", "Hi!")
        manager.activate_version("greeting", v2.version_id)

        archived = manager.archive_version("greeting", v1.version_id)
        assert archived.status == VersionStatus.ARCHIVED

    def test_archive_active_raises(self):
        """Test that archiving active version raises error."""
        manager = TemplateVersionManager()
        v1 = manager.create_template("greeting", "Hello!")
        with pytest.raises(ValueError, match="Cannot archive active"):
            manager.archive_version("greeting", v1.version_id)

    def test_diff_versions(self):
        """Test diffing two versions."""
        manager = TemplateVersionManager()
        v1 = manager.create_template("greeting", "Hello {name}!")
        v2 = manager.create_version("greeting", "Hi {name} and {friend}!")

        diff = manager.diff_versions("greeting", v1.version_id, v2.version_id)
        assert diff["content_changed"] is True
        assert "friend" in diff["variables_added"]
        assert len(diff["variables_removed"]) == 0

    def test_export_import_template(self):
        """Test exporting and importing templates."""
        manager1 = TemplateVersionManager()
        manager1.create_template("greeting", "Hello!")
        manager1.create_version("greeting", "Hi!")

        exported = manager1.export_template("greeting")
        assert exported["template_name"] == "greeting"
        assert len(exported["versions"]) == 2

        manager2 = TemplateVersionManager()
        imported = manager2.import_template(exported)
        assert len(imported) == 2
        assert manager2.get_active_version("greeting") is not None


class TestABTestRunner:
    """Tests for ABTestRunner."""

    def test_create_test(self):
        """Test creating an A/B test."""
        runner = ABTestRunner()
        t1 = TemplateVersion(
            version_id="v1",
            template_name="greeting",
            content="Hello!",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )
        t2 = TemplateVersion(
            version_id="v2",
            template_name="greeting",
            content="Hi there!",
            version_number="2.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-02T00:00:00",
        )

        test = runner.create_test(
            name="Greeting Test",
            variants=[
                ("Control", t1, 1.0),
                ("Treatment", t2, 1.0),
            ],
        )
        assert test.name == "Greeting Test"
        assert len(test.variants) == 2

    def test_list_tests(self):
        """Test listing tests."""
        runner = ABTestRunner()
        t1 = TemplateVersion(
            version_id="v1",
            template_name="greeting",
            content="Hello!",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )

        runner.create_test("Test 1", [("A", t1, 1.0)])
        runner.create_test("Test 2", [("B", t1, 1.0)])

        assert len(runner.list_tests()) == 2


class TestABTest:
    """Tests for ABTest."""

    def _create_test(self, strategy: AllocationStrategy = AllocationStrategy.RANDOM) -> ABTest:
        """Helper to create a test."""
        t1 = TemplateVersion(
            version_id="v1",
            template_name="greeting",
            content="Hello!",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )
        t2 = TemplateVersion(
            version_id="v2",
            template_name="greeting",
            content="Hi!",
            version_number="2.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-02T00:00:00",
        )
        runner = ABTestRunner(strategy=strategy)
        return runner.create_test(
            name="Test",
            variants=[
                ("Control", t1, 1.0),
                ("Treatment", t2, 2.0),
            ],
        )

    def test_start_stop(self):
        """Test starting and stopping test."""
        test = self._create_test()
        assert test.status == ABTestStatus.PENDING

        test.start()
        assert test.status == ABTestStatus.RUNNING
        assert test.started_at is not None

        result = test.stop()
        assert test.status == ABTestStatus.COMPLETED
        assert isinstance(result, ABTestResult)

    def test_pause_resume(self):
        """Test pausing and resuming test."""
        test = self._create_test()
        test.start()

        test.pause()
        assert test.status == ABTestStatus.PAUSED

        test.resume()
        assert test.status == ABTestStatus.RUNNING

    def test_cancel(self):
        """Test cancelling test."""
        test = self._create_test()
        test.start()
        test.cancel()
        assert test.status == ABTestStatus.CANCELLED

    def test_select_variant_random(self):
        """Test random variant selection."""
        test = self._create_test(AllocationStrategy.RANDOM)
        test.start()

        # Run many selections and check both variants get selected
        selected = set()
        for _ in range(100):
            variant = test.select_variant()
            selected.add(variant.name)

        assert len(selected) == 2

    def test_select_variant_round_robin(self):
        """Test round-robin variant selection."""
        test = self._create_test(AllocationStrategy.ROUND_ROBIN)
        test.start()

        v1 = test.select_variant()
        v2 = test.select_variant()
        v3 = test.select_variant()

        assert v1.name != v2.name
        assert v3.name == v1.name

    def test_select_variant_weighted(self):
        """Test weighted variant selection."""
        test = self._create_test(AllocationStrategy.WEIGHTED)
        test.start()

        # Treatment has weight 2.0, Control has 1.0
        # Treatment should be selected about 2x more often
        counts = {"Control": 0, "Treatment": 0}
        for _ in range(1000):
            variant = test.select_variant()
            counts[variant.name] += 1

        # Treatment should be roughly 2x Control
        ratio = counts["Treatment"] / counts["Control"]
        assert 1.5 < ratio < 2.5

    def test_record_result(self):
        """Test recording results."""
        test = self._create_test()
        test.start()

        variant = test.variants[0]
        test.record_result(variant.variant_id, 0.8, converted=True)
        test.record_result(variant.variant_id, 0.6, converted=False)

        assert variant.impressions == 2
        assert variant.conversions == 1

    def test_get_results(self):
        """Test getting results."""
        test = self._create_test()
        test.start()

        # Record some results
        for variant in test.variants:
            for _ in range(10):
                test.record_result(variant.variant_id, 0.7, converted=True)

        result = test.get_results()
        assert result.total_impressions == 20
        assert len(result.variants) == 2

    def test_is_ready_for_analysis(self):
        """Test readiness check."""
        runner = ABTestRunner(min_samples_per_variant=10)
        t1 = TemplateVersion(
            version_id="v1",
            template_name="greeting",
            content="Hello!",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )
        test = runner.create_test("Test", [("A", t1, 1.0), ("B", t1, 1.0)])
        test.start()

        assert not test.is_ready_for_analysis

        for variant in test.variants:
            for _ in range(10):
                test.record_result(variant.variant_id, 0.5)

        assert test.is_ready_for_analysis


class TestABTestResult:
    """Tests for ABTestResult."""

    def test_is_significant(self):
        """Test significance check."""
        result = ABTestResult(
            test_id="test1",
            test_name="Test",
            status=ABTestStatus.COMPLETED,
            variants=[],
            winner="var1",
            confidence=0.96,
            total_impressions=1000,
            duration_seconds=3600,
            started_at="2024-01-01T00:00:00",
            ended_at="2024-01-01T01:00:00",
        )
        assert result.is_significant

        result.confidence = 0.90
        assert not result.is_significant

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ABTestResult(
            test_id="test1",
            test_name="Test",
            status=ABTestStatus.COMPLETED,
            variants=[],
            winner="var1",
            confidence=0.96,
            total_impressions=1000,
            duration_seconds=3600,
            started_at="2024-01-01T00:00:00",
            ended_at="2024-01-01T01:00:00",
        )
        d = result.to_dict()
        assert d["test_id"] == "test1"
        assert d["is_significant"] is True


class TestTemplateExperiment:
    """Tests for TemplateExperiment."""

    def test_run_comparison(self):
        """Test running a comparison experiment."""

        def scorer(response: str, expected: str) -> float:
            return 1.0 if expected.lower() in response.lower() else 0.0

        def converter(response: str, expected: str) -> bool:
            return expected.lower() in response.lower()

        experiment = TemplateExperiment(scorer, converter)

        t1 = TemplateVersion(
            version_id="v1",
            template_name="qa",
            content="Question: {question}\nAnswer:",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )
        t2 = TemplateVersion(
            version_id="v2",
            template_name="qa",
            content="Q: {question}\nA:",
            version_number="2.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-02T00:00:00",
        )

        def render_fn(content: str, variables: dict) -> str:
            return content.format(**variables)

        def model_fn(prompt: str) -> str:
            return "The answer is Paris"

        test_cases = [
            {"variables": {"question": "Capital of France?"}, "expected": "Paris"},
        ]

        results = experiment.run_comparison([t1, t2], test_cases, render_fn, model_fn, n_runs=1)

        assert "results" in results
        assert "winner" in results
        assert results["results"]["v1"]["avg_score"] == 1.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_and_get_template(self):
        """Test creating and getting templates."""
        # Reset manager
        set_default_manager(TemplateVersionManager())

        version = create_template("test_greeting", "Hello {name}!")
        assert version.template_name == "test_greeting"

        active = get_active_template("test_greeting")
        assert active.version_id == version.version_id

    def test_create_version_convenience(self):
        """Test creating version with convenience function."""
        set_default_manager(TemplateVersionManager())

        create_template("test_greeting", "Hello!")
        v2 = create_version("test_greeting", "Hi there!")
        assert v2.version_number == "1.0.1"

    def test_activate_and_rollback(self):
        """Test activate and rollback convenience functions."""
        set_default_manager(TemplateVersionManager())

        v1 = create_template("test_greeting", "Hello!")
        v2 = create_version("test_greeting", "Hi!")
        activate_version("test_greeting", v2.version_id)

        active = get_active_template("test_greeting")
        assert active.version_id == v2.version_id

        rolled_back = rollback_template("test_greeting")
        assert rolled_back.version_id == v1.version_id

    def test_list_versions_convenience(self):
        """Test listing versions."""
        set_default_manager(TemplateVersionManager())

        create_template("test_greeting", "Hello!")
        create_version("test_greeting", "Hi!")
        create_version("test_greeting", "Hey!")

        versions = list_template_versions("test_greeting")
        assert len(versions) == 3

    def test_diff_versions_convenience(self):
        """Test diff convenience function."""
        set_default_manager(TemplateVersionManager())

        v1 = create_template("test_greeting", "Hello {name}!")
        v2 = create_version("test_greeting", "Hi {name}!")

        diff = diff_template_versions("test_greeting", v1.version_id, v2.version_id)
        assert diff["content_changed"] is True

    def test_create_ab_test_convenience(self):
        """Test creating A/B test."""
        t1 = TemplateVersion(
            version_id="v1",
            template_name="greeting",
            content="Hello!",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )
        test = create_ab_test("Test", [("Control", t1, 1.0)])
        assert test.name == "Test"

    def test_run_template_comparison_convenience(self):
        """Test template comparison convenience function."""
        t1 = TemplateVersion(
            version_id="v1",
            template_name="qa",
            content="Q: {q}",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )

        def scorer(r: str, e: str) -> float:
            return 0.8

        def render_fn(c: str, v: dict) -> str:
            return c.format(**v)

        def model_fn(p: str) -> str:
            return "Answer"

        results = run_template_comparison(
            [t1],
            [{"variables": {"q": "test"}, "expected": ""}],
            render_fn,
            model_fn,
            scorer,
        )
        assert results["results"]["v1"]["avg_score"] == 0.8


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_template_content(self):
        """Test template with empty content."""
        manager = TemplateVersionManager()
        version = manager.create_template("empty", "")
        assert version.content == ""
        assert len(version.variables) == 0

    def test_template_with_many_variables(self):
        """Test template with many variables."""
        manager = TemplateVersionManager()
        content = "Hello {name}, you have {count} messages from {sender}"
        version = manager.create_template("complex", content)
        assert len(version.variables) == 3
        assert "name" in version.variables
        assert "count" in version.variables
        assert "sender" in version.variables

    def test_template_with_double_braces(self):
        """Test template with double braces (escaped)."""
        manager = TemplateVersionManager()
        content = "Hello {{name}}"
        version = manager.create_template("escaped", content)
        assert "name" in version.variables

    def test_version_number_parsing(self):
        """Test various version number formats."""
        manager = TemplateVersionManager()
        manager.create_template("test", "v1", initial_version="1.0.0")
        v2 = manager.create_version("test", "v2", explicit_version="2.0.0")
        v3 = manager.create_version("test", "v3", explicit_version="2.1.5")

        assert v2.version_number == "2.0.0"
        assert v3.version_number == "2.1.5"

    def test_ab_test_single_variant(self):
        """Test A/B test with single variant."""
        t1 = TemplateVersion(
            version_id="v1",
            template_name="greeting",
            content="Hello!",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )
        runner = ABTestRunner()
        test = runner.create_test("Single", [("Only", t1, 1.0)])
        test.start()

        variant = test.select_variant()
        assert variant.name == "Only"

    def test_multi_armed_bandit_learns(self):
        """Test that MAB strategy learns from results."""
        t1 = TemplateVersion(
            version_id="v1",
            template_name="greeting",
            content="Hello!",
            version_number="1.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-01T00:00:00",
        )
        t2 = TemplateVersion(
            version_id="v2",
            template_name="greeting",
            content="Hi!",
            version_number="2.0.0",
            status=VersionStatus.ACTIVE,
            created_at="2024-01-02T00:00:00",
        )
        runner = ABTestRunner(strategy=AllocationStrategy.MULTI_ARMED_BANDIT)
        test = runner.create_test("MAB", [("A", t1, 1.0), ("B", t2, 1.0)])
        test.start()

        # Simulate A being much better than B
        for _ in range(50):
            test.record_result(test.variants[0].variant_id, 0.9, converted=True)
            test.record_result(test.variants[1].variant_id, 0.1, converted=False)

        # After learning, A should be selected more often
        a_count = 0
        for _ in range(100):
            variant = test.select_variant()
            if variant.name == "A":
                a_count += 1

        # A should be selected most of the time
        assert a_count > 70

    def test_get_nonexistent_template(self):
        """Test getting nonexistent template."""
        manager = TemplateVersionManager()
        result = manager.get_active_version("nonexistent")
        assert result is None

    def test_rollback_no_parent(self):
        """Test rollback when there's no parent."""
        manager = TemplateVersionManager()
        manager.create_template("test", "Hello!")
        with pytest.raises(ValueError, match="No parent version"):
            manager.rollback("test")
