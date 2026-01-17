"""
Distributed execution capabilities for running experiments across multiple workers.

Provides tools for:
- Multi-process execution using Python's multiprocessing
- Ray-based distributed execution for cluster environments
- Work distribution and result aggregation
- Fault tolerance and checkpointing
"""

import hashlib
import json
import os
import pickle
import queue
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

T = TypeVar("T")
R = TypeVar("R")


class WorkerStatus(Enum):
    """Status of a distributed worker."""

    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


@dataclass(order=False)
class Task:
    """A single task to be executed."""

    id: str
    payload: Any
    priority: int = 0
    retries: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "Task") -> bool:
        """Compare tasks for priority queue ordering."""
        # Higher priority comes first (negative priority in queue)
        # If same priority, earlier creation time comes first
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "payload": self.payload,
            "priority": self.priority,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TaskResult:
    """Result from executing a task."""

    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    worker_id: Optional[str] = None
    completed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "worker_id": self.worker_id,
            "completed_at": self.completed_at,
        }


@dataclass
class WorkerInfo:
    """Information about a worker."""

    id: str
    status: WorkerStatus = WorkerStatus.IDLE
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_task: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)


class TaskExecutor(ABC):
    """Abstract base class for task executors."""

    @abstractmethod
    def execute(self, payload: Any) -> Any:
        """Execute a task payload and return result."""
        pass


class FunctionExecutor(TaskExecutor):
    """Execute tasks using a provided function."""

    def __init__(self, func: Callable[[Any], Any]):
        """Initialize with a callable."""
        self.func = func

    def execute(self, payload: Any) -> Any:
        """Execute the function with the payload."""
        return self.func(payload)


class WorkQueue:
    """Thread-safe work queue with priority support."""

    def __init__(self, maxsize: int = 0):
        """Initialize the queue."""
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize)
        self._pending: Dict[str, Task] = {}
        self._lock = threading.Lock()

    def put(self, task: Task) -> None:
        """Add a task to the queue."""
        with self._lock:
            self._pending[task.id] = task
            # Use negative priority so higher priority comes first
            self._queue.put((-task.priority, task.created_at, task))

    def get(self, timeout: Optional[float] = None) -> Optional[Task]:
        """Get the next task from the queue."""
        try:
            _, _, task = self._queue.get(timeout=timeout)
            with self._lock:
                if task.id in self._pending:
                    del self._pending[task.id]
            return task
        except queue.Empty:
            return None

    def task_done(self) -> None:
        """Mark a task as done."""
        self._queue.task_done()

    @property
    def size(self) -> int:
        """Get queue size."""
        return self._queue.qsize()

    @property
    def pending_count(self) -> int:
        """Get pending task count."""
        with self._lock:
            return len(self._pending)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()


class ResultCollector:
    """Collect and aggregate results from workers."""

    def __init__(self):
        """Initialize the collector."""
        self._results: Dict[str, TaskResult] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[TaskResult], None]] = []

    def add(self, result: TaskResult) -> None:
        """Add a result."""
        with self._lock:
            self._results[result.task_id] = result
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception:
                pass

    def get(self, task_id: str) -> Optional[TaskResult]:
        """Get a result by task ID."""
        with self._lock:
            return self._results.get(task_id)

    def get_all(self) -> List[TaskResult]:
        """Get all results."""
        with self._lock:
            return list(self._results.values())

    def on_result(self, callback: Callable[[TaskResult], None]) -> None:
        """Register a callback for new results."""
        self._callbacks.append(callback)

    @property
    def count(self) -> int:
        """Get result count."""
        with self._lock:
            return len(self._results)

    @property
    def success_count(self) -> int:
        """Get successful result count."""
        with self._lock:
            return sum(1 for r in self._results.values() if r.success)

    @property
    def failure_count(self) -> int:
        """Get failed result count."""
        with self._lock:
            return sum(1 for r in self._results.values() if not r.success)


class LocalWorker:
    """A local worker that processes tasks in a thread."""

    def __init__(
        self,
        worker_id: str,
        executor: TaskExecutor,
        work_queue: WorkQueue,
        result_collector: ResultCollector,
    ):
        """Initialize the worker."""
        self.worker_id = worker_id
        self.executor = executor
        self.work_queue = work_queue
        self.result_collector = result_collector
        self.info = WorkerInfo(id=worker_id)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the worker."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self.info.status = WorkerStatus.SHUTDOWN

    def _run(self) -> None:
        """Main worker loop."""
        while self._running:
            task = self.work_queue.get(timeout=1.0)
            if task is None:
                continue

            self.info.status = WorkerStatus.BUSY
            self.info.current_task = task.id
            self.info.last_heartbeat = time.time()

            start_time = time.time()
            try:
                result_data = self.executor.execute(task.payload)
                latency_ms = (time.time() - start_time) * 1000
                result = TaskResult(
                    task_id=task.id,
                    success=True,
                    result=result_data,
                    latency_ms=latency_ms,
                    worker_id=self.worker_id,
                )
                self.info.tasks_completed += 1
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                result = TaskResult(
                    task_id=task.id,
                    success=False,
                    error=str(e),
                    latency_ms=latency_ms,
                    worker_id=self.worker_id,
                )
                self.info.tasks_failed += 1

            self.result_collector.add(result)
            self.work_queue.task_done()
            self.info.status = WorkerStatus.IDLE
            self.info.current_task = None


class LocalDistributedExecutor:
    """Execute tasks across multiple local threads/processes."""

    def __init__(
        self,
        executor: TaskExecutor,
        num_workers: int = 4,
        use_processes: bool = False,
    ):
        """Initialize the distributed executor.

        Args:
            executor: Task executor to use.
            num_workers: Number of worker threads/processes.
            use_processes: Use processes instead of threads.
        """
        self.executor = executor
        self.num_workers = num_workers
        self.use_processes = use_processes
        self.work_queue = WorkQueue()
        self.result_collector = ResultCollector()
        self._workers: List[LocalWorker] = []
        self._started = False

    def start(self) -> None:
        """Start all workers."""
        if self._started:
            return

        for i in range(self.num_workers):
            worker = LocalWorker(
                worker_id=f"worker_{i}",
                executor=self.executor,
                work_queue=self.work_queue,
                result_collector=self.result_collector,
            )
            worker.start()
            self._workers.append(worker)

        self._started = True

    def stop(self) -> None:
        """Stop all workers."""
        for worker in self._workers:
            worker.stop()
        self._workers.clear()
        self._started = False

    def submit(self, task: Task) -> None:
        """Submit a task for execution."""
        if not self._started:
            self.start()
        self.work_queue.put(task)

    def submit_batch(self, tasks: List[Task]) -> None:
        """Submit multiple tasks."""
        for task in tasks:
            self.submit(task)

    def wait_for_completion(
        self,
        timeout: Optional[float] = None,
        poll_interval: float = 0.1,
    ) -> List[TaskResult]:
        """Wait for all submitted tasks to complete."""
        start_time = time.time()
        expected_count = self.work_queue.size + len(self._workers)  # Rough estimate

        while self.work_queue.size > 0 or any(
            w.info.status == WorkerStatus.BUSY for w in self._workers
        ):
            if timeout and (time.time() - start_time) > timeout:
                break
            time.sleep(poll_interval)

        return self.result_collector.get_all()

    def get_results(self) -> List[TaskResult]:
        """Get all collected results."""
        return self.result_collector.get_all()

    def get_worker_info(self) -> List[WorkerInfo]:
        """Get information about all workers."""
        return [w.info for w in self._workers]

    def __enter__(self) -> "LocalDistributedExecutor":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class CheckpointManager:
    """Manage checkpoints for fault tolerance."""

    def __init__(self, checkpoint_dir: Optional[str] = None):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(
            checkpoint_dir or tempfile.gettempdir()
        ) / "insidellms_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get path for a checkpoint."""
        return self.checkpoint_dir / f"{checkpoint_id}.checkpoint"

    def save(
        self,
        checkpoint_id: str,
        pending_tasks: List[Task],
        completed_results: List[TaskResult],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a checkpoint.

        Args:
            checkpoint_id: Unique identifier for the checkpoint.
            pending_tasks: Tasks that haven't been processed.
            completed_results: Results that have been collected.
            metadata: Optional metadata.

        Returns:
            Path to the checkpoint file.
        """
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "timestamp": time.time(),
            "pending_tasks": [t.to_dict() for t in pending_tasks],
            "completed_results": [r.to_dict() for r in completed_results],
            "metadata": metadata or {},
        }

        path = self._get_checkpoint_path(checkpoint_id)
        with open(path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        return str(path)

    def load(
        self, checkpoint_id: str
    ) -> Tuple[List[Task], List[TaskResult], Dict[str, Any]]:
        """Load a checkpoint.

        Args:
            checkpoint_id: Identifier of the checkpoint to load.

        Returns:
            Tuple of (pending_tasks, completed_results, metadata).
        """
        path = self._get_checkpoint_path(checkpoint_id)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        pending_tasks = [Task.from_dict(t) for t in data["pending_tasks"]]
        completed_results = [
            TaskResult(**r) for r in data["completed_results"]
        ]

        return pending_tasks, completed_results, data.get("metadata", {})

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        return [
            p.stem for p in self.checkpoint_dir.glob("*.checkpoint")
        ]

    def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        path = self._get_checkpoint_path(checkpoint_id)
        if path.exists():
            path.unlink()


class ProcessPoolDistributedExecutor:
    """Execute tasks across multiple processes."""

    def __init__(
        self,
        func: Callable[[Any], Any],
        num_workers: int = 4,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        """Initialize the process pool executor.

        Args:
            func: Function to execute tasks (must be pickleable).
            num_workers: Number of worker processes.
            checkpoint_manager: Optional checkpoint manager for fault tolerance.
        """
        self.func = func
        self.num_workers = num_workers
        self.checkpoint_manager = checkpoint_manager
        self._pending_tasks: List[Task] = []
        self._results: List[TaskResult] = []

    def submit(self, task: Task) -> None:
        """Submit a task."""
        self._pending_tasks.append(task)

    def submit_batch(self, tasks: List[Task]) -> None:
        """Submit multiple tasks."""
        self._pending_tasks.extend(tasks)

    def run(
        self,
        checkpoint_id: Optional[str] = None,
        checkpoint_interval: int = 100,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[TaskResult]:
        """Run all submitted tasks.

        Args:
            checkpoint_id: ID for checkpointing (enables fault tolerance).
            checkpoint_interval: Save checkpoint every N tasks.
            progress_callback: Callback(completed, total) for progress updates.

        Returns:
            List of task results.
        """
        # Load from checkpoint if available
        if checkpoint_id and self.checkpoint_manager:
            try:
                pending, completed, _ = self.checkpoint_manager.load(checkpoint_id)
                self._pending_tasks = pending
                self._results = completed
            except FileNotFoundError:
                pass

        total = len(self._pending_tasks) + len(self._results)
        completed_count = len(self._results)

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for task in self._pending_tasks:
                future = executor.submit(self._execute_task, task)
                futures[future] = task

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    self._results.append(result)
                except Exception as e:
                    self._results.append(
                        TaskResult(
                            task_id=task.id,
                            success=False,
                            error=str(e),
                        )
                    )

                completed_count += 1

                if progress_callback:
                    progress_callback(completed_count, total)

                # Save checkpoint periodically
                if (
                    checkpoint_id
                    and self.checkpoint_manager
                    and completed_count % checkpoint_interval == 0
                ):
                    remaining = [
                        t for t in self._pending_tasks
                        if t.id not in {r.task_id for r in self._results}
                    ]
                    self.checkpoint_manager.save(
                        checkpoint_id, remaining, self._results
                    )

        # Clear pending tasks
        self._pending_tasks.clear()

        # Delete checkpoint if completed successfully
        if checkpoint_id and self.checkpoint_manager:
            self.checkpoint_manager.delete(checkpoint_id)

        return self._results

    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        start_time = time.time()
        try:
            result = self.func(task.payload)
            return TaskResult(
                task_id=task.id,
                success=True,
                result=result,
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )


class MapReduceExecutor(Generic[T, R]):
    """Execute map-reduce style distributed computations."""

    def __init__(
        self,
        mapper: Callable[[T], R],
        reducer: Callable[[List[R]], Any],
        num_workers: int = 4,
    ):
        """Initialize the map-reduce executor.

        Args:
            mapper: Function to map each item.
            reducer: Function to reduce mapped results.
            num_workers: Number of parallel workers.
        """
        self.mapper = mapper
        self.reducer = reducer
        self.num_workers = num_workers

    def execute(self, items: List[T]) -> Any:
        """Execute map-reduce on items.

        Args:
            items: Items to process.

        Returns:
            Reduced result.
        """
        # Map phase (parallel)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            mapped_results = list(executor.map(self.mapper, items))

        # Reduce phase
        return self.reducer(mapped_results)

    def execute_with_partitions(
        self,
        items: List[T],
        partition_size: int = 100,
    ) -> Any:
        """Execute with partitioned reduce.

        Args:
            items: Items to process.
            partition_size: Size of each partition for intermediate reduce.

        Returns:
            Final reduced result.
        """
        # Map phase
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            mapped_results = list(executor.map(self.mapper, items))

        # Partitioned reduce
        partitions = [
            mapped_results[i : i + partition_size]
            for i in range(0, len(mapped_results), partition_size)
        ]

        # Reduce each partition
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            partition_results = list(executor.map(self.reducer, partitions))

        # Final reduce
        return self.reducer(partition_results)


class DistributedExperimentRunner:
    """Run LLM experiments in a distributed manner."""

    def __init__(
        self,
        model_func: Callable[[str], str],
        num_workers: int = 4,
        use_processes: bool = False,
        checkpoint_dir: Optional[str] = None,
    ):
        """Initialize the distributed experiment runner.

        Args:
            model_func: Function that takes a prompt and returns a response.
            num_workers: Number of parallel workers.
            use_processes: Use processes instead of threads.
            checkpoint_dir: Directory for checkpoints (enables fault tolerance).
        """
        self.model_func = model_func
        self.num_workers = num_workers
        self.use_processes = use_processes
        self.checkpoint_manager = (
            CheckpointManager(checkpoint_dir) if checkpoint_dir else None
        )

    def run_prompts(
        self,
        prompts: List[str],
        checkpoint_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Run a list of prompts in parallel.

        Args:
            prompts: List of prompts to run.
            checkpoint_id: ID for checkpointing.
            progress_callback: Callback for progress updates.

        Returns:
            List of results with 'prompt', 'response', 'success', 'error'.
        """
        tasks = [
            Task(
                id=hashlib.md5(prompt.encode()).hexdigest()[:12],
                payload=prompt,
            )
            for prompt in prompts
        ]

        if self.use_processes:
            executor = ProcessPoolDistributedExecutor(
                func=self._execute_prompt,
                num_workers=self.num_workers,
                checkpoint_manager=self.checkpoint_manager,
            )
            executor.submit_batch(tasks)
            results = executor.run(
                checkpoint_id=checkpoint_id,
                progress_callback=progress_callback,
            )
        else:
            executor = LocalDistributedExecutor(
                executor=FunctionExecutor(self._execute_prompt),
                num_workers=self.num_workers,
            )
            with executor:
                executor.submit_batch(tasks)
                results = executor.wait_for_completion()

        # Map results back to prompts
        result_map = {r.task_id: r for r in results}
        output = []
        for task, prompt in zip(tasks, prompts):
            result = result_map.get(task.id)
            if result:
                output.append({
                    "prompt": prompt,
                    "response": result.result if result.success else None,
                    "success": result.success,
                    "error": result.error,
                    "latency_ms": result.latency_ms,
                })
            else:
                output.append({
                    "prompt": prompt,
                    "response": None,
                    "success": False,
                    "error": "Task not found",
                    "latency_ms": 0,
                })

        return output

    def _execute_prompt(self, prompt: str) -> str:
        """Execute a single prompt."""
        return self.model_func(prompt)

    def run_experiments(
        self,
        experiments: List[Dict[str, Any]],
        prompt_key: str = "prompt",
        checkpoint_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run experiments from a list of dictionaries.

        Args:
            experiments: List of experiment dictionaries.
            prompt_key: Key containing the prompt.
            checkpoint_id: ID for checkpointing.

        Returns:
            Experiments with added 'response' and 'success' fields.
        """
        prompts = [exp[prompt_key] for exp in experiments]
        results = self.run_prompts(prompts, checkpoint_id=checkpoint_id)

        # Merge results back
        for exp, result in zip(experiments, results):
            exp["response"] = result["response"]
            exp["success"] = result["success"]
            exp["error"] = result.get("error")
            exp["latency_ms"] = result["latency_ms"]

        return experiments


def parallel_map(
    func: Callable[[T], R],
    items: List[T],
    num_workers: int = 4,
    use_processes: bool = False,
) -> List[R]:
    """Simple parallel map function.

    Args:
        func: Function to apply to each item.
        items: Items to process.
        num_workers: Number of parallel workers.
        use_processes: Use processes instead of threads.

    Returns:
        List of results.
    """
    ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    with ExecutorClass(max_workers=num_workers) as executor:
        return list(executor.map(func, items))


def batch_process(
    func: Callable[[List[T]], List[R]],
    items: List[T],
    batch_size: int = 10,
    num_workers: int = 4,
) -> List[R]:
    """Process items in batches with parallel batch execution.

    Args:
        func: Function that processes a batch of items.
        items: Items to process.
        batch_size: Size of each batch.
        num_workers: Number of parallel workers.

    Returns:
        Flattened list of results.
    """
    # Create batches
    batches = [
        items[i : i + batch_size]
        for i in range(0, len(items), batch_size)
    ]

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_results = list(executor.map(func, batches))

    # Flatten results
    return [item for batch in batch_results for item in batch]
