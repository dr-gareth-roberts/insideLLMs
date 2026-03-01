"""Tests for the distributed execution module."""

import pickle
import tempfile
import time

import pytest


class TestTask:
    """Test the Task class."""

    def test_task_creation(self):
        """Test creating a Task."""
        from insideLLMs.contrib.distributed import Task

        task = Task(id="task_1", payload="test data")

        assert task.id == "task_1"
        assert task.payload == "test data"
        assert task.priority == 0
        assert task.retries == 0

    def test_task_with_priority(self):
        """Test Task with priority."""
        from insideLLMs.contrib.distributed import Task

        task = Task(id="task_1", payload="data", priority=10)

        assert task.priority == 10

    def test_task_to_dict(self):
        """Test Task serialization."""
        from insideLLMs.contrib.distributed import Task

        task = Task(id="task_1", payload="data", metadata={"key": "value"})
        d = task.to_dict()

        assert d["id"] == "task_1"
        assert d["payload"] == "data"
        assert d["metadata"]["key"] == "value"

    def test_task_from_dict(self):
        """Test Task deserialization."""
        from insideLLMs.contrib.distributed import Task

        data = {
            "id": "task_1",
            "payload": "data",
            "priority": 5,
            "retries": 1,
            "max_retries": 3,
            "created_at": 1234567890.0,
            "metadata": {},
        }
        task = Task.from_dict(data)

        assert task.id == "task_1"
        assert task.priority == 5
        assert task.retries == 1


class TestTaskResult:
    """Test the TaskResult class."""

    def test_result_creation(self):
        """Test creating a TaskResult."""
        from insideLLMs.contrib.distributed import TaskResult

        result = TaskResult(
            task_id="task_1",
            success=True,
            result="output",
            latency_ms=100.0,
        )

        assert result.task_id == "task_1"
        assert result.success is True
        assert result.result == "output"
        assert result.latency_ms == 100.0

    def test_failed_result(self):
        """Test a failed TaskResult."""
        from insideLLMs.contrib.distributed import TaskResult

        result = TaskResult(
            task_id="task_1",
            success=False,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_result_to_dict(self):
        """Test TaskResult serialization."""
        from insideLLMs.contrib.distributed import TaskResult

        result = TaskResult(
            task_id="task_1",
            success=True,
            result="output",
            latency_ms=100.0,
            worker_id="worker_0",
        )
        d = result.to_dict()

        assert d["task_id"] == "task_1"
        assert d["success"] is True
        assert d["worker_id"] == "worker_0"


class TestWorkQueue:
    """Test the WorkQueue class."""

    def test_queue_creation(self):
        """Test creating a WorkQueue."""
        from insideLLMs.contrib.distributed import WorkQueue

        queue = WorkQueue()

        assert queue.size == 0
        assert queue.is_empty() is True

    def test_queue_put_get(self):
        """Test putting and getting tasks."""
        from insideLLMs.contrib.distributed import Task, WorkQueue

        queue = WorkQueue()
        task = Task(id="task_1", payload="data")

        queue.put(task)
        assert queue.size == 1

        retrieved = queue.get(timeout=1.0)
        assert retrieved is not None
        assert retrieved.id == "task_1"

    def test_queue_priority(self):
        """Test priority ordering."""
        from insideLLMs.contrib.distributed import Task, WorkQueue

        queue = WorkQueue()

        queue.put(Task(id="low", payload="low", priority=1))
        queue.put(Task(id="high", payload="high", priority=10))
        queue.put(Task(id="medium", payload="medium", priority=5))

        # Higher priority should come first
        first = queue.get(timeout=1.0)
        assert first.id == "high"

        second = queue.get(timeout=1.0)
        assert second.id == "medium"

        third = queue.get(timeout=1.0)
        assert third.id == "low"

    def test_queue_timeout(self):
        """Test queue timeout."""
        from insideLLMs.contrib.distributed import WorkQueue

        queue = WorkQueue()

        start = time.time()
        result = queue.get(timeout=0.1)
        elapsed = time.time() - start

        assert result is None
        assert elapsed >= 0.1


class TestResultCollector:
    """Test the ResultCollector class."""

    def test_collector_creation(self):
        """Test creating a ResultCollector."""
        from insideLLMs.contrib.distributed import ResultCollector

        collector = ResultCollector()

        assert collector.count == 0

    def test_collector_add_get(self):
        """Test adding and getting results."""
        from insideLLMs.contrib.distributed import ResultCollector, TaskResult

        collector = ResultCollector()
        result = TaskResult(task_id="task_1", success=True, result="output")

        collector.add(result)
        assert collector.count == 1

        retrieved = collector.get("task_1")
        assert retrieved is not None
        assert retrieved.result == "output"

    def test_collector_success_count(self):
        """Test counting successful results."""
        from insideLLMs.contrib.distributed import ResultCollector, TaskResult

        collector = ResultCollector()

        collector.add(TaskResult(task_id="1", success=True))
        collector.add(TaskResult(task_id="2", success=True))
        collector.add(TaskResult(task_id="3", success=False, error="fail"))

        assert collector.success_count == 2
        assert collector.failure_count == 1

    def test_collector_callback(self):
        """Test result callback."""
        from insideLLMs.contrib.distributed import ResultCollector, TaskResult

        collector = ResultCollector()
        received = []

        collector.on_result(lambda r: received.append(r))
        collector.add(TaskResult(task_id="task_1", success=True))

        assert len(received) == 1
        assert received[0].task_id == "task_1"


class TestFunctionExecutor:
    """Test the FunctionExecutor class."""

    def test_executor_creation(self):
        """Test creating a FunctionExecutor."""
        from insideLLMs.contrib.distributed import FunctionExecutor

        executor = FunctionExecutor(lambda x: x * 2)

        assert executor is not None

    def test_executor_execute(self):
        """Test executing with FunctionExecutor."""
        from insideLLMs.contrib.distributed import FunctionExecutor

        executor = FunctionExecutor(lambda x: x.upper())
        result = executor.execute("hello")

        assert result == "HELLO"


class TestLocalDistributedExecutor:
    """Test the LocalDistributedExecutor class."""

    def test_executor_creation(self):
        """Test creating a LocalDistributedExecutor."""
        from insideLLMs.contrib.distributed import FunctionExecutor, LocalDistributedExecutor

        executor = LocalDistributedExecutor(
            executor=FunctionExecutor(lambda x: x),
            num_workers=2,
        )

        assert executor is not None
        assert executor.num_workers == 2

    def test_executor_basic_run(self):
        """Test basic execution."""
        from insideLLMs.contrib.distributed import (
            FunctionExecutor,
            LocalDistributedExecutor,
            Task,
        )

        executor = LocalDistributedExecutor(
            executor=FunctionExecutor(lambda x: x * 2),
            num_workers=2,
        )

        with executor:
            executor.submit(Task(id="task_1", payload=5))
            results = executor.wait_for_completion(timeout=5.0)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].result == 10

    def test_executor_batch(self):
        """Test batch execution."""
        from insideLLMs.contrib.distributed import (
            FunctionExecutor,
            LocalDistributedExecutor,
            Task,
        )

        executor = LocalDistributedExecutor(
            executor=FunctionExecutor(lambda x: x**2),
            num_workers=4,
        )

        tasks = [Task(id=f"task_{i}", payload=i) for i in range(10)]

        with executor:
            executor.submit_batch(tasks)
            results = executor.wait_for_completion(timeout=10.0)

        assert len(results) == 10
        results_by_id = {r.task_id: r.result for r in results}
        for i in range(10):
            assert results_by_id[f"task_{i}"] == i**2


class TestCheckpointManager:
    """Test the CheckpointManager class."""

    def test_checkpoint_creation(self):
        """Test creating a CheckpointManager."""
        from insideLLMs.contrib.distributed import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            assert manager is not None

    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoints."""
        from insideLLMs.contrib.distributed import CheckpointManager, Task, TaskResult

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            pending = [Task(id="task_1", payload="data")]
            completed = [TaskResult(task_id="task_0", success=True, result="done")]
            metadata = {"experiment": "test"}

            manager.save("checkpoint_1", pending, completed, metadata)

            loaded_pending, loaded_completed, loaded_meta = manager.load("checkpoint_1")

            assert len(loaded_pending) == 1
            assert loaded_pending[0].id == "task_1"
            assert len(loaded_completed) == 1
            assert loaded_completed[0].task_id == "task_0"
            assert loaded_meta["experiment"] == "test"

    def test_checkpoint_list(self):
        """Test listing checkpoints."""
        from insideLLMs.contrib.distributed import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            manager.save("checkpoint_1", [], [])
            manager.save("checkpoint_2", [], [])

            checkpoints = manager.list_checkpoints()
            assert "checkpoint_1" in checkpoints
            assert "checkpoint_2" in checkpoints

    def test_checkpoint_delete(self):
        """Test deleting checkpoints."""
        from insideLLMs.contrib.distributed import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            manager.save("checkpoint_1", [], [])
            assert "checkpoint_1" in manager.list_checkpoints()

            manager.delete("checkpoint_1")
            assert "checkpoint_1" not in manager.list_checkpoints()

    def test_checkpoint_rejects_path_traversal_id(self):
        """Checkpoint IDs with path traversal are rejected."""
        from insideLLMs.contrib.distributed import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            with pytest.raises(ValueError, match="Invalid checkpoint_id"):
                manager.save("../escape", [], [])

    def test_checkpoint_rejects_path_separator(self):
        """Checkpoint IDs with separators are rejected."""
        from insideLLMs.contrib.distributed import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            with pytest.raises(ValueError, match="path separators"):
                manager.save("bad/name", [], [])

    def test_checkpoint_legacy_pickle_blocked_by_default(self):
        """Legacy pickle checkpoints are always rejected for security."""
        from insideLLMs.contrib.distributed import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            path = manager._get_checkpoint_path("legacy_pickle")
            with open(path, "wb") as f:
                pickle.dump({"pending_tasks": [], "completed_results": [], "metadata": {}}, f)

            with pytest.raises(ValueError, match="no longer supported"):
                manager.load("legacy_pickle")

    def test_checkpoint_legacy_pickle_rejected_even_with_flag(self):
        """Even with allow_unsafe_pickle flag, pickle loading is rejected (removed)."""
        from insideLLMs.contrib.distributed import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, allow_unsafe_pickle=True)
            path = manager._get_checkpoint_path("legacy_pickle")
            with open(path, "wb") as f:
                pickle.dump({"pending_tasks": [], "completed_results": [], "metadata": {}}, f)

            with pytest.raises(ValueError, match="no longer supported"):
                manager.load("legacy_pickle")


class TestMapReduceExecutor:
    """Test the MapReduceExecutor class."""

    def test_mapreduce_basic(self):
        """Test basic map-reduce."""
        from insideLLMs.contrib.distributed import MapReduceExecutor

        executor = MapReduceExecutor(
            mapper=lambda x: x * 2,
            reducer=lambda xs: sum(xs),
            num_workers=2,
        )

        result = executor.execute([1, 2, 3, 4, 5])

        # Map: [2, 4, 6, 8, 10], Reduce: 30
        assert result == 30

    def test_mapreduce_strings(self):
        """Test map-reduce with strings."""
        from insideLLMs.contrib.distributed import MapReduceExecutor

        executor = MapReduceExecutor(
            mapper=str.upper,
            reducer=lambda xs: " ".join(xs),
            num_workers=2,
        )

        result = executor.execute(["hello", "world"])

        assert result == "HELLO WORLD"

    def test_mapreduce_with_partitions(self):
        """Test partitioned map-reduce."""
        from insideLLMs.contrib.distributed import MapReduceExecutor

        executor = MapReduceExecutor(
            mapper=lambda x: x,
            reducer=lambda xs: sum(xs) if isinstance(xs[0], int) else sum(xs),
            num_workers=2,
        )

        result = executor.execute_with_partitions(
            list(range(100)),
            partition_size=10,
        )

        assert result == sum(range(100))


class TestDistributedExperimentRunner:
    """Test the DistributedExperimentRunner class."""

    def test_runner_creation(self):
        """Test creating a DistributedExperimentRunner."""
        from insideLLMs.contrib.distributed import DistributedExperimentRunner

        runner = DistributedExperimentRunner(
            model_func=lambda x: f"Response: {x}",
            num_workers=2,
        )

        assert runner is not None

    def test_runner_run_prompts(self):
        """Test running prompts."""
        from insideLLMs.contrib.distributed import DistributedExperimentRunner

        runner = DistributedExperimentRunner(
            model_func=lambda x: f"Response to: {x}",
            num_workers=2,
        )

        results = runner.run_prompts(["Hello", "World"])

        assert len(results) == 2
        assert results[0]["success"] is True
        assert "Hello" in results[0]["response"]
        assert results[1]["success"] is True
        assert "World" in results[1]["response"]

    def test_runner_run_experiments(self):
        """Test running experiments from dictionaries."""
        from insideLLMs.contrib.distributed import DistributedExperimentRunner

        runner = DistributedExperimentRunner(
            model_func=lambda x: f"Answer: {x}",
            num_workers=2,
        )

        experiments = [
            {"prompt": "Question 1?", "id": 1},
            {"prompt": "Question 2?", "id": 2},
        ]

        results = runner.run_experiments(experiments)

        assert len(results) == 2
        assert results[0]["success"] is True
        assert "response" in results[0]
        assert results[0]["id"] == 1


class TestParallelMap:
    """Test the parallel_map function."""

    def test_parallel_map_basic(self):
        """Test basic parallel map."""
        from insideLLMs.contrib.distributed import parallel_map

        results = parallel_map(
            func=lambda x: x**2,
            items=[1, 2, 3, 4, 5],
            num_workers=2,
        )

        assert results == [1, 4, 9, 16, 25]

    def test_parallel_map_strings(self):
        """Test parallel map with strings."""
        from insideLLMs.contrib.distributed import parallel_map

        results = parallel_map(
            func=str.upper,
            items=["a", "b", "c"],
            num_workers=2,
        )

        assert results == ["A", "B", "C"]


class TestBatchProcess:
    """Test the batch_process function."""

    def test_batch_process_basic(self):
        """Test basic batch processing."""
        from insideLLMs.contrib.distributed import batch_process

        results = batch_process(
            func=lambda batch: [x * 2 for x in batch],
            items=list(range(10)),
            batch_size=3,
            num_workers=2,
        )

        assert results == [x * 2 for x in range(10)]

    def test_batch_process_strings(self):
        """Test batch processing with strings."""
        from insideLLMs.contrib.distributed import batch_process

        results = batch_process(
            func=lambda batch: [s.upper() for s in batch],
            items=["a", "b", "c", "d", "e"],
            batch_size=2,
            num_workers=2,
        )

        assert results == ["A", "B", "C", "D", "E"]


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_task_batch(self):
        """Test processing empty batch."""
        from insideLLMs.contrib.distributed import (
            FunctionExecutor,
            LocalDistributedExecutor,
        )

        executor = LocalDistributedExecutor(
            executor=FunctionExecutor(lambda x: x),
            num_workers=2,
        )

        with executor:
            executor.submit_batch([])
            results = executor.wait_for_completion(timeout=1.0)

        assert len(results) == 0

    def test_executor_error_handling(self):
        """Test error handling in executor."""
        from insideLLMs.contrib.distributed import (
            FunctionExecutor,
            LocalDistributedExecutor,
            Task,
        )

        def failing_func(x):
            raise ValueError("Test error")

        executor = LocalDistributedExecutor(
            executor=FunctionExecutor(failing_func),
            num_workers=1,
        )

        with executor:
            executor.submit(Task(id="task_1", payload="data"))
            results = executor.wait_for_completion(timeout=5.0)

        assert len(results) == 1
        assert results[0].success is False
        assert "Test error" in results[0].error

    def test_checkpoint_not_found(self):
        """Test loading non-existent checkpoint."""
        from insideLLMs.contrib.distributed import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            with pytest.raises(FileNotFoundError):
                manager.load("nonexistent")
