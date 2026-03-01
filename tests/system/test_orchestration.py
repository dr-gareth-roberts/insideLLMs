"""Tests for experiment orchestration utilities."""

import pytest

from insideLLMs.system.orchestration import (
    AsyncBatchRunner,
    BatchResult,
    BatchRunner,
    ComparisonExperiment,
    ExperimentConfig,
    ExperimentGrid,
    ExperimentOrchestrator,
    ExperimentQueue,
    ExperimentRun,
    ExperimentStatus,
    create_experiment_configs,
    run_quick_batch,
)


class TestExperimentStatus:
    """Tests for ExperimentStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all statuses are defined."""
        assert ExperimentStatus.PENDING.value == "pending"
        assert ExperimentStatus.RUNNING.value == "running"
        assert ExperimentStatus.COMPLETED.value == "completed"
        assert ExperimentStatus.FAILED.value == "failed"
        assert ExperimentStatus.CANCELLED.value == "cancelled"


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_basic_creation(self):
        """Test basic config creation."""
        config = ExperimentConfig(
            id="exp1",
            name="Test Experiment",
            prompt="What is 2+2?",
            model_id="gpt-4",
        )
        assert config.id == "exp1"
        assert config.name == "Test Experiment"
        assert config.model_id == "gpt-4"
        assert config.priority == 0

    def test_with_parameters(self):
        """Test config with parameters."""
        config = ExperimentConfig(
            id="exp1",
            name="Test",
            prompt="Hello",
            model_id="model",
            parameters={"temperature": 0.7},
            tags=["test", "example"],
        )
        assert config.parameters["temperature"] == 0.7
        assert "test" in config.tags


class TestExperimentRun:
    """Tests for ExperimentRun."""

    def test_basic_creation(self):
        """Test basic run creation."""
        config = ExperimentConfig(id="exp1", name="Test", prompt="Hello", model_id="model")
        run = ExperimentRun(config=config)

        assert run.status == ExperimentStatus.PENDING
        assert run.response is None
        assert not run.is_complete

    def test_completed_run(self):
        """Test completed run."""
        config = ExperimentConfig(id="exp1", name="Test", prompt="Hello", model_id="model")
        run = ExperimentRun(
            config=config,
            status=ExperimentStatus.COMPLETED,
            response="World",
            latency_ms=150.0,
        )

        assert run.is_complete
        assert run.response == "World"

    def test_duration_calculation(self):
        """Test duration from latency."""
        config = ExperimentConfig(id="exp1", name="Test", prompt="Hello", model_id="model")
        run = ExperimentRun(config=config, latency_ms=200.0)

        assert run.duration_ms == 200.0


class TestBatchResult:
    """Tests for BatchResult."""

    def _create_runs(self) -> list:
        """Helper to create test runs."""
        configs = [
            ExperimentConfig(id=f"exp{i}", name=f"Test{i}", prompt="P", model_id="m")
            for i in range(5)
        ]
        runs = [
            ExperimentRun(
                config=configs[0],
                status=ExperimentStatus.COMPLETED,
                latency_ms=100,
                metrics={"score": 0.9},
            ),
            ExperimentRun(
                config=configs[1],
                status=ExperimentStatus.COMPLETED,
                latency_ms=150,
                metrics={"score": 0.8},
            ),
            ExperimentRun(
                config=configs[2],
                status=ExperimentStatus.FAILED,
                error="Test error",
            ),
            ExperimentRun(
                config=configs[3],
                status=ExperimentStatus.COMPLETED,
                latency_ms=200,
                metrics={"score": 0.95},
            ),
            ExperimentRun(
                config=configs[4],
                status=ExperimentStatus.PENDING,
            ),
        ]
        # Add tags to some configs
        configs[0].tags = ["tag1"]
        configs[1].tags = ["tag1", "tag2"]
        return runs

    def test_counts(self):
        """Test count properties."""
        runs = self._create_runs()
        result = BatchResult(batch_id="test", runs=runs)

        assert result.total == 5
        assert result.completed == 3
        assert result.failed == 1

    def test_success_rate(self):
        """Test success rate calculation."""
        runs = self._create_runs()
        result = BatchResult(batch_id="test", runs=runs)

        # 3 completed, 1 failed = 75%
        assert result.success_rate == 0.75

    def test_avg_latency(self):
        """Test average latency calculation."""
        runs = self._create_runs()
        result = BatchResult(batch_id="test", runs=runs)

        # (100 + 150 + 200) / 3 = 150
        assert result.avg_latency_ms == 150.0

    def test_get_by_status(self):
        """Test filtering by status."""
        runs = self._create_runs()
        result = BatchResult(batch_id="test", runs=runs)

        completed = result.get_by_status(ExperimentStatus.COMPLETED)
        assert len(completed) == 3

        failed = result.get_by_status(ExperimentStatus.FAILED)
        assert len(failed) == 1

    def test_get_by_tag(self):
        """Test filtering by tag."""
        runs = self._create_runs()
        result = BatchResult(batch_id="test", runs=runs)

        tag1_runs = result.get_by_tag("tag1")
        assert len(tag1_runs) == 2

    def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        runs = self._create_runs()
        result = BatchResult(batch_id="test", runs=runs)

        agg = result.aggregate_metrics()

        assert "score" in agg
        assert agg["score"]["count"] == 3
        assert abs(agg["score"]["mean"] - 0.883) < 0.01
        assert agg["score"]["min"] == 0.8
        assert agg["score"]["max"] == 0.95


class TestExperimentQueue:
    """Tests for ExperimentQueue."""

    def test_add_and_next(self):
        """Test adding and getting from queue."""
        queue = ExperimentQueue()
        config = ExperimentConfig(id="exp1", name="Test", prompt="Hello", model_id="model")

        queue.add(config)
        assert queue.pending_count == 1

        next_config = queue.next()
        assert next_config == config
        assert queue.pending_count == 0
        assert queue.running_count == 1

    def test_priority_ordering(self):
        """Test priority-based ordering."""
        queue = ExperimentQueue()

        low = ExperimentConfig(id="low", name="Low", prompt="P", model_id="m", priority=0)
        high = ExperimentConfig(id="high", name="High", prompt="P", model_id="m", priority=10)

        queue.add(low)
        queue.add(high)

        # High priority should come first
        first = queue.next()
        assert first.id == "high"

    def test_complete(self):
        """Test completing an experiment."""
        queue = ExperimentQueue()
        config = ExperimentConfig(id="exp1", name="Test", prompt="Hello", model_id="model")

        queue.add(config)
        queue.next()

        run = ExperimentRun(config=config, status=ExperimentStatus.COMPLETED)
        queue.complete(run)

        assert queue.running_count == 0
        assert queue.completed_count == 1

    def test_is_empty(self):
        """Test empty check."""
        queue = ExperimentQueue()
        assert queue.is_empty()

        config = ExperimentConfig(id="exp1", name="Test", prompt="Hello", model_id="model")
        queue.add(config)
        assert not queue.is_empty()


class TestBatchRunner:
    """Tests for BatchRunner."""

    def test_run_sequential(self):
        """Test sequential execution."""

        def mock_executor(prompt: str, **kwargs) -> str:
            return f"Response to: {prompt}"

        runner = BatchRunner(executor=mock_executor)
        configs = [
            ExperimentConfig(id=f"exp{i}", name=f"Test{i}", prompt=f"P{i}", model_id="m")
            for i in range(3)
        ]

        result = runner.run_sequential(configs)

        assert result.total == 3
        assert result.completed == 3
        assert all(r.response is not None for r in result.runs)

    def test_run_with_metrics(self):
        """Test running with metrics calculator."""

        def mock_executor(prompt: str, **kwargs) -> str:
            return "Response"

        def mock_metrics(prompt: str, response: str, config) -> dict:
            return {"length": len(response)}

        runner = BatchRunner(
            executor=mock_executor,
            metrics_calculator=mock_metrics,
        )
        configs = [ExperimentConfig(id="exp1", name="Test", prompt="Hello", model_id="m")]

        result = runner.run(configs)

        assert result.runs[0].metrics["length"] == 8

    def test_run_with_error(self):
        """Test handling errors."""

        def failing_executor(prompt: str, **kwargs) -> str:
            raise ValueError("Test error")

        runner = BatchRunner(executor=failing_executor)
        configs = [ExperimentConfig(id="exp1", name="Test", prompt="Hello", model_id="m")]

        result = runner.run(configs)

        assert result.failed == 1
        assert result.runs[0].error == "Test error"

    def test_progress_callback(self):
        """Test progress callback."""
        progress_calls = []

        def mock_executor(prompt: str, **kwargs) -> str:
            return "Response"

        def on_progress(completed, total, current):
            progress_calls.append((completed, total))

        runner = BatchRunner(executor=mock_executor)
        runner.on_progress(on_progress)

        configs = [
            ExperimentConfig(id=f"exp{i}", name=f"Test{i}", prompt=f"P{i}", model_id="m")
            for i in range(3)
        ]

        runner.run(configs)

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)


class TestExperimentGrid:
    """Tests for ExperimentGrid."""

    def test_basic_generation(self):
        """Test basic grid generation."""
        grid = ExperimentGrid(
            base_prompt="Test prompt",
            model_ids=["model1", "model2"],
        )

        configs = grid.generate()
        assert len(configs) == 2

    def test_with_parameters(self):
        """Test grid with parameters."""
        grid = ExperimentGrid(
            base_prompt="Test prompt",
            model_ids=["model1"],
            parameters={
                "temperature": [0.0, 0.5, 1.0],
                "top_p": [0.9, 1.0],
            },
        )

        configs = grid.generate()
        # 1 model * 3 temps * 2 top_p = 6
        assert len(configs) == 6

    def test_length(self):
        """Test __len__ method."""
        grid = ExperimentGrid(
            base_prompt="Test",
            model_ids=["m1", "m2"],
            parameters={"temp": [0, 1]},
        )

        assert len(grid) == 4


class TestComparisonExperiment:
    """Tests for ComparisonExperiment."""

    def test_generation(self):
        """Test comparison generation."""
        comparison = ComparisonExperiment(
            prompts={
                "simple": "What is {input}?",
                "detailed": "Please explain {input} in detail.",
            },
            model_id="model1",
            inputs=["Python", "AI"],
        )

        configs = comparison.generate()

        # 2 prompts * 2 inputs = 4
        assert len(configs) == 4

    def test_tags(self):
        """Test that prompt names are added as tags."""
        comparison = ComparisonExperiment(
            prompts={"simple": "What is {input}?"},
            model_id="model1",
            inputs=["test"],
        )

        configs = comparison.generate()
        assert "simple" in configs[0].tags


class TestExperimentOrchestrator:
    """Tests for ExperimentOrchestrator."""

    def test_run_grid(self):
        """Test running a grid experiment."""

        def mock_executor(prompt: str, **kwargs) -> str:
            return "Response"

        orchestrator = ExperimentOrchestrator(executor=mock_executor)
        grid = ExperimentGrid(
            base_prompt="Test",
            model_ids=["m1", "m2"],
        )

        result = orchestrator.run_grid(grid)

        assert result.completed == 2
        assert len(orchestrator.list_batches()) == 1

    def test_run_comparison(self):
        """Test running a comparison experiment."""

        def mock_executor(prompt: str, **kwargs) -> str:
            return "Response"

        orchestrator = ExperimentOrchestrator(executor=mock_executor)
        comparison = ComparisonExperiment(
            prompts={"a": "Prompt A: {input}", "b": "Prompt B: {input}"},
            model_id="model1",
            inputs=["test"],
        )

        result = orchestrator.run_comparison(comparison)

        assert result.completed == 2

    def test_generate_report(self):
        """Test report generation."""

        def mock_executor(prompt: str, **kwargs) -> str:
            return "Response"

        orchestrator = ExperimentOrchestrator(executor=mock_executor)
        grid = ExperimentGrid(base_prompt="Test", model_ids=["m1"])
        orchestrator.run_grid(grid)

        report = orchestrator.generate_report()

        assert "Experiment Orchestration Report" in report
        assert "Total Batches" in report


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_experiment_configs(self):
        """Test creating configs from prompts."""
        prompts = ["P1", "P2", "P3"]
        configs = create_experiment_configs(prompts, "model1", prefix="test")

        assert len(configs) == 3
        assert configs[0].id == "test_0"
        assert configs[0].prompt == "P1"

    def test_run_quick_batch(self):
        """Test quick batch execution."""

        def mock_executor(prompt: str, **kwargs) -> str:
            return f"Response to {prompt}"

        prompts = ["Hello", "World"]
        result = run_quick_batch(prompts, mock_executor)

        assert result.completed == 2


class TestAsyncBatchRunner:
    """Tests for AsyncBatchRunner."""

    @pytest.mark.asyncio
    async def test_async_run(self):
        """Test async execution."""

        async def mock_executor(prompt: str, **kwargs) -> str:
            return f"Response to: {prompt}"

        runner = AsyncBatchRunner(executor=mock_executor)
        configs = [
            ExperimentConfig(id=f"exp{i}", name=f"Test{i}", prompt=f"P{i}", model_id="m")
            for i in range(3)
        ]

        result = await runner.run(configs)

        assert result.completed == 3

    @pytest.mark.asyncio
    async def test_async_concurrency_limit(self):
        """Test concurrency limiting."""
        call_count = 0

        async def mock_executor(prompt: str, **kwargs) -> str:
            nonlocal call_count
            call_count += 1
            return "Response"

        runner = AsyncBatchRunner(executor=mock_executor, max_concurrent=2)
        configs = [
            ExperimentConfig(id=f"exp{i}", name=f"Test{i}", prompt=f"P{i}", model_id="m")
            for i in range(5)
        ]

        result = await runner.run(configs)

        assert result.completed == 5
        assert call_count == 5
