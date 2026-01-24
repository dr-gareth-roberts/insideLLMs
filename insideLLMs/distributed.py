"""
Distributed execution capabilities for running experiments across multiple workers.

This module provides a comprehensive toolkit for executing LLM experiments and
general computational tasks in a distributed manner. It supports both thread-based
and process-based parallelism, with optional fault tolerance through checkpointing.

Key Components
--------------
Task Management
    - `Task`: Represents a unit of work with priority and retry support
    - `TaskResult`: Captures execution results including success status and latency
    - `WorkQueue`: Thread-safe priority queue for task distribution

Workers and Executors
    - `TaskExecutor`: Abstract base class for defining task execution logic
    - `FunctionExecutor`: Wraps a callable as a TaskExecutor
    - `LocalWorker`: Worker thread that processes tasks from a queue
    - `LocalDistributedExecutor`: Coordinates multiple LocalWorkers
    - `ProcessPoolDistributedExecutor`: Process-based parallel execution

Higher-Level Abstractions
    - `MapReduceExecutor`: Implements the map-reduce paradigm
    - `DistributedExperimentRunner`: Purpose-built for LLM experiments

Utility Functions
    - `parallel_map`: Simple parallel map over items
    - `batch_process`: Parallel batch processing with result flattening

Fault Tolerance
    - `DistributedCheckpointManager`: Saves and restores execution state

Examples
--------
Basic parallel execution with threads:

>>> from insideLLMs.distributed import parallel_map
>>> def process_item(x):
...     return x ** 2
>>> results = parallel_map(process_item, [1, 2, 3, 4, 5], num_workers=4)
>>> print(results)
[1, 4, 9, 16, 25]

Running LLM experiments in parallel:

>>> from insideLLMs.distributed import DistributedExperimentRunner
>>> def mock_llm(prompt):
...     return f"Response to: {prompt}"
>>> runner = DistributedExperimentRunner(mock_llm, num_workers=4)
>>> prompts = ["What is AI?", "Explain ML", "Define NLP"]
>>> results = runner.run_prompts(prompts)
>>> for r in results:
...     print(f"Success: {r['success']}, Response: {r['response'][:30]}...")

Using map-reduce for aggregation:

>>> from insideLLMs.distributed import MapReduceExecutor
>>> mapper = lambda x: len(x.split())
>>> reducer = lambda counts: sum(counts)
>>> executor = MapReduceExecutor(mapper, reducer, num_workers=2)
>>> sentences = ["Hello world", "This is a test", "Map reduce example"]
>>> total_words = executor.execute(sentences)
>>> print(f"Total words: {total_words}")
Total words: 9

Task-based execution with priority:

>>> from insideLLMs.distributed import (
...     Task, LocalDistributedExecutor, FunctionExecutor
... )
>>> executor = FunctionExecutor(lambda x: x * 2)
>>> with LocalDistributedExecutor(executor, num_workers=2) as dist:
...     dist.submit(Task(id="low", payload=5, priority=1))
...     dist.submit(Task(id="high", payload=10, priority=10))
...     results = dist.wait_for_completion()
>>> for r in sorted(results, key=lambda x: x.task_id):
...     print(f"{r.task_id}: {r.result}")

Checkpointed execution for fault tolerance:

>>> from insideLLMs.distributed import (
...     ProcessPoolDistributedExecutor, DistributedCheckpointManager, Task
... )
>>> checkpoint_mgr = DistributedCheckpointManager("/tmp/checkpoints")
>>> executor = ProcessPoolDistributedExecutor(
...     func=lambda x: x ** 2,
...     num_workers=2,
...     checkpoint_manager=checkpoint_mgr
... )
>>> tasks = [Task(id=f"task_{i}", payload=i) for i in range(100)]
>>> executor.submit_batch(tasks)
>>> results = executor.run(checkpoint_id="my_experiment", checkpoint_interval=10)

Notes
-----
Thread vs Process Execution:
    - Use threads (`use_processes=False`) for I/O-bound tasks like API calls
    - Use processes (`use_processes=True`) for CPU-bound tasks
    - Process-based execution requires pickleable functions and payloads

Fault Tolerance:
    - Enable checkpointing by providing a `checkpoint_manager`
    - Checkpoints are automatically deleted on successful completion
    - Resume from checkpoint by using the same `checkpoint_id`

Performance Considerations:
    - Higher `num_workers` increases parallelism but also memory usage
    - For API rate limits, consider adding delays in your executor function
    - Batch processing can reduce overhead for many small tasks

See Also
--------
concurrent.futures : Standard library for parallel execution
multiprocessing : Low-level process-based parallelism
threading : Low-level thread-based parallelism
"""

import contextlib
import hashlib
import pickle
import queue
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class WorkerStatus(Enum):
    """
    Enumeration of possible worker states in the distributed execution system.

    Workers transition through these states during their lifecycle, from IDLE
    when waiting for work, to BUSY when processing, and potentially to FAILED
    or SHUTDOWN when issues occur or the worker is stopped.

    Attributes
    ----------
    IDLE : str
        Worker is available and waiting for tasks. This is the initial state
        and the state returned to after completing a task.
    BUSY : str
        Worker is actively processing a task. The worker's `current_task`
        attribute will contain the task ID.
    FAILED : str
        Worker encountered an unrecoverable error. This state indicates the
        worker thread/process has crashed and needs to be restarted.
    SHUTDOWN : str
        Worker has been gracefully stopped. This is the terminal state after
        calling `stop()` on a worker.

    Examples
    --------
    Checking worker state:

    >>> from insideLLMs.distributed import WorkerStatus, LocalWorker
    >>> # After creating a worker
    >>> worker.info.status == WorkerStatus.IDLE
    True

    Using status in conditionals:

    >>> from insideLLMs.distributed import WorkerStatus
    >>> status = WorkerStatus.BUSY
    >>> if status == WorkerStatus.BUSY:
    ...     print("Worker is processing a task")
    Worker is processing a task

    Iterating over all statuses:

    >>> from insideLLMs.distributed import WorkerStatus
    >>> for status in WorkerStatus:
    ...     print(f"{status.name}: {status.value}")
    IDLE: idle
    BUSY: busy
    FAILED: failed
    SHUTDOWN: shutdown

    See Also
    --------
    WorkerInfo : Contains the status along with other worker metadata
    LocalWorker : Worker implementation that uses these statuses
    """

    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


@dataclass(order=False)
class Task:
    """
    A single unit of work to be executed by a distributed worker.

    Tasks are the fundamental unit of work in the distributed execution system.
    Each task has a unique identifier, a payload containing the work to be done,
    and optional priority and retry settings for fault tolerance.

    Parameters
    ----------
    id : str
        Unique identifier for the task. Used for tracking, checkpointing,
        and result correlation. Should be unique across all tasks in a batch.
    payload : Any
        The data to be processed. Can be any pickleable Python object.
        For LLM experiments, this is typically a prompt string or dict.
    priority : int, optional
        Task priority for queue ordering. Higher values are processed first.
        Default is 0. Tasks with equal priority are processed in FIFO order.
    retries : int, optional
        Current retry count. Starts at 0 and is incremented on failure.
        Default is 0.
    max_retries : int, optional
        Maximum number of retry attempts before marking as failed.
        Default is 3.
    created_at : float, optional
        Unix timestamp of task creation. Auto-populated if not provided.
        Used for FIFO ordering within same priority level.
    metadata : dict[str, Any], optional
        Additional metadata for tracking or debugging. Can contain any
        serializable data. Default is empty dict.

    Attributes
    ----------
    id : str
        The task's unique identifier.
    payload : Any
        The task's data payload.
    priority : int
        The task's priority level.
    retries : int
        Number of times this task has been retried.
    max_retries : int
        Maximum retries allowed.
    created_at : float
        Creation timestamp.
    metadata : dict[str, Any]
        Additional task metadata.

    Examples
    --------
    Creating a simple task:

    >>> from insideLLMs.distributed import Task
    >>> task = Task(id="task_001", payload="What is machine learning?")
    >>> print(f"Task {task.id}: {task.payload}")
    Task task_001: What is machine learning?

    Creating a high-priority task:

    >>> from insideLLMs.distributed import Task
    >>> urgent_task = Task(
    ...     id="urgent_001",
    ...     payload={"prompt": "Critical query", "temperature": 0.1},
    ...     priority=100,
    ...     metadata={"source": "emergency_queue"}
    ... )
    >>> print(f"Priority: {urgent_task.priority}")
    Priority: 100

    Creating tasks with retry configuration:

    >>> from insideLLMs.distributed import Task
    >>> flaky_task = Task(
    ...     id="flaky_001",
    ...     payload="Retry-prone operation",
    ...     max_retries=5
    ... )
    >>> print(f"Max retries: {flaky_task.max_retries}")
    Max retries: 5

    Comparing tasks (for priority queue):

    >>> from insideLLMs.distributed import Task
    >>> import time
    >>> task1 = Task(id="t1", payload="first", priority=1)
    >>> time.sleep(0.01)  # Ensure different timestamps
    >>> task2 = Task(id="t2", payload="second", priority=10)
    >>> task3 = Task(id="t3", payload="third", priority=1)
    >>> # Higher priority comes first
    >>> task2 < task1  # task2 has higher priority
    True
    >>> # Same priority: earlier creation time wins
    >>> task1 < task3  # task1 was created first
    True

    Serializing and deserializing:

    >>> from insideLLMs.distributed import Task
    >>> import json
    >>> task = Task(id="ser_001", payload={"key": "value"}, priority=5)
    >>> task_dict = task.to_dict()
    >>> json_str = json.dumps(task_dict)
    >>> restored = Task.from_dict(json.loads(json_str))
    >>> restored.id == task.id and restored.priority == task.priority
    True

    See Also
    --------
    TaskResult : The result object returned after task execution
    WorkQueue : Priority queue that uses Task comparison for ordering
    LocalDistributedExecutor : Executor that processes tasks
    """

    id: str
    payload: Any
    priority: int = 0
    retries: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "Task") -> bool:
        """
        Compare tasks for priority queue ordering.

        Implements comparison such that higher priority tasks come first.
        For tasks with equal priority, earlier creation time takes precedence.

        Parameters
        ----------
        other : Task
            Another task to compare against.

        Returns
        -------
        bool
            True if this task should be processed before `other`.

        Examples
        --------
        >>> from insideLLMs.distributed import Task
        >>> import time
        >>> high = Task(id="high", payload="x", priority=10)
        >>> low = Task(id="low", payload="y", priority=1)
        >>> high < low  # high priority comes first
        True
        >>> early = Task(id="early", payload="x", priority=5)
        >>> time.sleep(0.01)
        >>> late = Task(id="late", payload="y", priority=5)
        >>> early < late  # same priority, earlier wins
        True
        """
        # Higher priority comes first (negative priority in queue)
        # If same priority, earlier creation time comes first
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the task to a dictionary for serialization.

        Creates a dictionary representation suitable for JSON serialization,
        pickle, or storage in a database. All fields are included.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all task fields with their current values.

        Examples
        --------
        >>> from insideLLMs.distributed import Task
        >>> task = Task(id="dict_001", payload={"prompt": "test"}, priority=5)
        >>> d = task.to_dict()
        >>> print(d["id"], d["priority"])
        dict_001 5
        >>> import json
        >>> json_str = json.dumps(d)  # Safe to serialize
        >>> len(json_str) > 0
        True

        See Also
        --------
        from_dict : Recreate a Task from a dictionary
        """
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
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """
        Create a Task instance from a dictionary.

        Factory method that reconstructs a Task from its dictionary
        representation, typically created by `to_dict()`.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing task fields. Must include at least 'id'
            and 'payload'. Other fields use defaults if not present.

        Returns
        -------
        Task
            A new Task instance with the provided data.

        Raises
        ------
        TypeError
            If required fields ('id', 'payload') are missing.

        Examples
        --------
        >>> from insideLLMs.distributed import Task
        >>> data = {
        ...     "id": "restored_001",
        ...     "payload": "test payload",
        ...     "priority": 5,
        ...     "retries": 0,
        ...     "max_retries": 3,
        ...     "created_at": 1234567890.0,
        ...     "metadata": {"key": "value"}
        ... }
        >>> task = Task.from_dict(data)
        >>> task.id
        'restored_001'
        >>> task.priority
        5

        Minimal dictionary (other fields use defaults):

        >>> from insideLLMs.distributed import Task
        >>> minimal = {"id": "min_001", "payload": "data"}
        >>> task = Task.from_dict(minimal)
        >>> task.max_retries  # Uses default
        3

        See Also
        --------
        to_dict : Convert a Task to a dictionary
        """
        return cls(**data)


@dataclass
class TaskResult:
    """
    The result of executing a task, including success status and timing.

    TaskResult captures the outcome of task execution, including the actual
    result data (on success), error information (on failure), latency metrics,
    and worker identification for debugging distributed systems.

    Parameters
    ----------
    task_id : str
        Identifier of the task that produced this result. Matches the `id`
        field of the corresponding Task object.
    success : bool
        Whether the task completed successfully. True indicates the executor
        returned without raising an exception.
    result : Any, optional
        The return value from the task executor. Only populated when
        `success=True`. Default is None.
    error : str, optional
        Error message if the task failed. Contains the string representation
        of the exception. Default is None.
    latency_ms : float, optional
        Time taken to execute the task in milliseconds. Measured from
        when the worker started processing to completion. Default is 0.0.
    worker_id : str, optional
        Identifier of the worker that processed this task. Useful for
        debugging worker-specific issues. Default is None.
    completed_at : float, optional
        Unix timestamp when the task completed. Auto-populated if not
        provided. Default is current time.

    Attributes
    ----------
    task_id : str
        The originating task's identifier.
    success : bool
        Success status of the execution.
    result : Any
        The task's return value (if successful).
    error : str or None
        Error message (if failed).
    latency_ms : float
        Execution time in milliseconds.
    worker_id : str or None
        ID of the processing worker.
    completed_at : float
        Completion timestamp.

    Examples
    --------
    Creating a successful result:

    >>> from insideLLMs.distributed import TaskResult
    >>> result = TaskResult(
    ...     task_id="task_001",
    ...     success=True,
    ...     result={"answer": "Machine learning is..."},
    ...     latency_ms=150.5,
    ...     worker_id="worker_0"
    ... )
    >>> print(f"Task {result.task_id} took {result.latency_ms}ms")
    Task task_001 took 150.5ms

    Creating a failure result:

    >>> from insideLLMs.distributed import TaskResult
    >>> error_result = TaskResult(
    ...     task_id="task_002",
    ...     success=False,
    ...     error="API rate limit exceeded",
    ...     latency_ms=50.0
    ... )
    >>> if not error_result.success:
    ...     print(f"Task failed: {error_result.error}")
    Task failed: API rate limit exceeded

    Processing results from a batch:

    >>> from insideLLMs.distributed import TaskResult
    >>> results = [
    ...     TaskResult(task_id="t1", success=True, result=10),
    ...     TaskResult(task_id="t2", success=False, error="timeout"),
    ...     TaskResult(task_id="t3", success=True, result=20),
    ... ]
    >>> successful = [r for r in results if r.success]
    >>> print(f"Success rate: {len(successful)}/{len(results)}")
    Success rate: 2/3

    Calculating aggregate statistics:

    >>> from insideLLMs.distributed import TaskResult
    >>> results = [
    ...     TaskResult(task_id="t1", success=True, latency_ms=100),
    ...     TaskResult(task_id="t2", success=True, latency_ms=200),
    ...     TaskResult(task_id="t3", success=True, latency_ms=150),
    ... ]
    >>> avg_latency = sum(r.latency_ms for r in results) / len(results)
    >>> print(f"Average latency: {avg_latency}ms")
    Average latency: 150.0ms

    Serializing for storage:

    >>> from insideLLMs.distributed import TaskResult
    >>> import json
    >>> result = TaskResult(task_id="ser_001", success=True, result="data")
    >>> json_str = json.dumps(result.to_dict())
    >>> len(json_str) > 0
    True

    See Also
    --------
    Task : The input task that produces a TaskResult
    ResultCollector : Aggregates TaskResults from multiple workers
    """

    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    worker_id: Optional[str] = None
    completed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the result to a dictionary for serialization.

        Creates a dictionary representation suitable for JSON serialization,
        checkpoint storage, or database persistence.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all result fields with their current values.

        Examples
        --------
        >>> from insideLLMs.distributed import TaskResult
        >>> result = TaskResult(
        ...     task_id="dict_001",
        ...     success=True,
        ...     result=42,
        ...     latency_ms=100.5
        ... )
        >>> d = result.to_dict()
        >>> print(d["task_id"], d["success"], d["result"])
        dict_001 True 42

        Serializing to JSON:

        >>> from insideLLMs.distributed import TaskResult
        >>> import json
        >>> result = TaskResult(task_id="json_001", success=True, result="test")
        >>> json_data = json.dumps(result.to_dict())
        >>> restored = json.loads(json_data)
        >>> restored["task_id"]
        'json_001'
        """
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
    """
    Runtime information and statistics about a worker.

    WorkerInfo provides real-time visibility into a worker's state, including
    its current status, task counts, and heartbeat for health monitoring.
    This information is useful for debugging, load balancing, and monitoring
    distributed execution.

    Parameters
    ----------
    id : str
        Unique identifier for the worker. Typically assigned by the executor
        (e.g., "worker_0", "worker_1").
    status : WorkerStatus, optional
        Current state of the worker. Default is WorkerStatus.IDLE.
    tasks_completed : int, optional
        Number of tasks successfully completed by this worker. Default is 0.
    tasks_failed : int, optional
        Number of tasks that failed on this worker. Default is 0.
    current_task : str, optional
        ID of the task currently being processed. None when idle.
        Default is None.
    last_heartbeat : float, optional
        Unix timestamp of the worker's last activity. Used for health
        monitoring and detecting stalled workers. Auto-populated.

    Attributes
    ----------
    id : str
        The worker's identifier.
    status : WorkerStatus
        Current worker state.
    tasks_completed : int
        Successful task count.
    tasks_failed : int
        Failed task count.
    current_task : str or None
        Currently processing task ID.
    last_heartbeat : float
        Last activity timestamp.

    Examples
    --------
    Creating worker info:

    >>> from insideLLMs.distributed import WorkerInfo, WorkerStatus
    >>> info = WorkerInfo(id="worker_0")
    >>> print(f"Worker {info.id} is {info.status.value}")
    Worker worker_0 is idle

    Checking worker health:

    >>> from insideLLMs.distributed import WorkerInfo, WorkerStatus
    >>> import time
    >>> info = WorkerInfo(id="worker_0", status=WorkerStatus.BUSY)
    >>> info.last_heartbeat = time.time() - 300  # 5 minutes ago
    >>> stale_threshold = 60  # 1 minute
    >>> is_stale = (time.time() - info.last_heartbeat) > stale_threshold
    >>> if is_stale:
    ...     print(f"Worker {info.id} may be stuck!")
    Worker worker_0 may be stuck!

    Monitoring worker statistics:

    >>> from insideLLMs.distributed import WorkerInfo
    >>> info = WorkerInfo(
    ...     id="worker_0",
    ...     tasks_completed=95,
    ...     tasks_failed=5
    ... )
    >>> total = info.tasks_completed + info.tasks_failed
    >>> success_rate = info.tasks_completed / total if total > 0 else 0
    >>> print(f"Success rate: {success_rate:.1%}")
    Success rate: 95.0%

    Checking if worker is available:

    >>> from insideLLMs.distributed import WorkerInfo, WorkerStatus
    >>> workers = [
    ...     WorkerInfo(id="w0", status=WorkerStatus.IDLE),
    ...     WorkerInfo(id="w1", status=WorkerStatus.BUSY),
    ...     WorkerInfo(id="w2", status=WorkerStatus.IDLE),
    ... ]
    >>> available = [w for w in workers if w.status == WorkerStatus.IDLE]
    >>> print(f"Available workers: {[w.id for w in available]}")
    Available workers: ['w0', 'w2']

    See Also
    --------
    WorkerStatus : Enumeration of worker states
    LocalWorker : Worker implementation that maintains WorkerInfo
    LocalDistributedExecutor : Provides get_worker_info() for monitoring
    """

    id: str
    status: WorkerStatus = WorkerStatus.IDLE
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_task: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)


class TaskExecutor(ABC):
    """
    Abstract base class defining the interface for task execution.

    TaskExecutor provides a standardized interface for executing task payloads.
    Subclasses must implement the `execute` method to define how payloads
    are processed. This abstraction allows different execution strategies
    (function calls, API requests, subprocess execution, etc.) to be used
    interchangeably with the distributed execution infrastructure.

    This is an abstract base class and cannot be instantiated directly.
    Use `FunctionExecutor` for wrapping simple callables, or create a
    custom subclass for more complex execution logic.

    Methods
    -------
    execute(payload)
        Abstract method that processes a payload and returns a result.

    Examples
    --------
    Creating a custom executor:

    >>> from insideLLMs.distributed import TaskExecutor
    >>> class SquareExecutor(TaskExecutor):
    ...     def execute(self, payload):
    ...         return payload ** 2
    >>> executor = SquareExecutor()
    >>> executor.execute(5)
    25

    Executor with external API call:

    >>> from insideLLMs.distributed import TaskExecutor
    >>> class LLMExecutor(TaskExecutor):
    ...     def __init__(self, api_client):
    ...         self.client = api_client
    ...     def execute(self, payload):
    ...         prompt = payload.get("prompt", str(payload))
    ...         return self.client.complete(prompt)

    Executor with retry logic:

    >>> from insideLLMs.distributed import TaskExecutor
    >>> import time
    >>> class RetryingExecutor(TaskExecutor):
    ...     def __init__(self, inner_executor, max_retries=3):
    ...         self.inner = inner_executor
    ...         self.max_retries = max_retries
    ...     def execute(self, payload):
    ...         for attempt in range(self.max_retries):
    ...             try:
    ...                 return self.inner.execute(payload)
    ...             except Exception:
    ...                 if attempt == self.max_retries - 1:
    ...                     raise
    ...                 time.sleep(2 ** attempt)  # Exponential backoff

    Using with LocalDistributedExecutor:

    >>> from insideLLMs.distributed import TaskExecutor, LocalDistributedExecutor
    >>> class MultiplyExecutor(TaskExecutor):
    ...     def __init__(self, factor):
    ...         self.factor = factor
    ...     def execute(self, payload):
    ...         return payload * self.factor
    >>> executor = MultiplyExecutor(10)
    >>> with LocalDistributedExecutor(executor, num_workers=2) as dist:
    ...     pass  # Submit tasks here

    See Also
    --------
    FunctionExecutor : Concrete implementation wrapping a callable
    LocalDistributedExecutor : Uses TaskExecutor for task processing
    LocalWorker : Worker that invokes TaskExecutor.execute()
    """

    @abstractmethod
    def execute(self, payload: Any) -> Any:
        """
        Execute a task payload and return the result.

        This method must be implemented by all subclasses. It receives the
        payload from a Task and should return the processed result. Any
        exception raised will be caught and recorded as a task failure.

        Parameters
        ----------
        payload : Any
            The data to process. The type and structure depend on the
            specific executor implementation and the tasks being processed.

        Returns
        -------
        Any
            The result of processing the payload. Will be stored in the
            TaskResult's `result` field.

        Raises
        ------
        Exception
            Any exception raised will cause the task to be marked as failed,
            with the error message stored in TaskResult's `error` field.

        Examples
        --------
        >>> from insideLLMs.distributed import TaskExecutor
        >>> class UppercaseExecutor(TaskExecutor):
        ...     def execute(self, payload):
        ...         return payload.upper()
        >>> executor = UppercaseExecutor()
        >>> executor.execute("hello")
        'HELLO'

        Handling structured payloads:

        >>> from insideLLMs.distributed import TaskExecutor
        >>> class DictExecutor(TaskExecutor):
        ...     def execute(self, payload):
        ...         if isinstance(payload, dict):
        ...             return payload.get("value", 0) * 2
        ...         return payload * 2
        >>> executor = DictExecutor()
        >>> executor.execute({"value": 21})
        42
        """
        pass


class FunctionExecutor(TaskExecutor):
    """
    TaskExecutor implementation that wraps a callable function.

    FunctionExecutor provides the simplest way to create a TaskExecutor by
    wrapping any callable. The wrapped function is invoked with the task
    payload as its single argument.

    Parameters
    ----------
    func : Callable[[Any], Any]
        The function to wrap. Should accept a single argument (the payload)
        and return the result. For process-based execution, the function
        must be pickleable (module-level functions work, lambdas don't).

    Attributes
    ----------
    func : Callable[[Any], Any]
        The wrapped function.

    Examples
    --------
    Basic usage with a simple function:

    >>> from insideLLMs.distributed import FunctionExecutor
    >>> def square(x):
    ...     return x ** 2
    >>> executor = FunctionExecutor(square)
    >>> executor.execute(5)
    25

    Using with a lambda (thread-based only):

    >>> from insideLLMs.distributed import FunctionExecutor
    >>> executor = FunctionExecutor(lambda x: x.upper())
    >>> executor.execute("hello")
    'HELLO'

    Processing structured data:

    >>> from insideLLMs.distributed import FunctionExecutor
    >>> def process_dict(data):
    ...     return {
    ...         "input": data,
    ...         "result": data.get("value", 0) * 2
    ...     }
    >>> executor = FunctionExecutor(process_dict)
    >>> executor.execute({"value": 21})
    {'input': {'value': 21}, 'result': 42}

    Combining with LocalDistributedExecutor:

    >>> from insideLLMs.distributed import FunctionExecutor, LocalDistributedExecutor, Task
    >>> def mock_llm(prompt):
    ...     return f"Response to: {prompt}"
    >>> executor = FunctionExecutor(mock_llm)
    >>> with LocalDistributedExecutor(executor, num_workers=2) as dist:
    ...     dist.submit(Task(id="t1", payload="What is AI?"))
    ...     dist.submit(Task(id="t2", payload="Explain ML"))
    ...     results = dist.wait_for_completion()
    >>> len(results)
    2

    Error handling - exceptions propagate:

    >>> from insideLLMs.distributed import FunctionExecutor
    >>> def risky_func(x):
    ...     if x < 0:
    ...         raise ValueError("Negative not allowed")
    ...     return x * 2
    >>> executor = FunctionExecutor(risky_func)
    >>> executor.execute(5)
    10
    >>> try:
    ...     executor.execute(-1)
    ... except ValueError as e:
    ...     print(f"Caught: {e}")
    Caught: Negative not allowed

    See Also
    --------
    TaskExecutor : Abstract base class
    LocalDistributedExecutor : Uses FunctionExecutor for task processing
    ProcessPoolDistributedExecutor : Requires pickleable functions
    """

    def __init__(self, func: Callable[[Any], Any]):
        """
        Initialize the FunctionExecutor with a callable.

        Parameters
        ----------
        func : Callable[[Any], Any]
            The function to execute for each task payload. Must accept
            exactly one argument and return a result.

        Examples
        --------
        >>> from insideLLMs.distributed import FunctionExecutor
        >>> executor = FunctionExecutor(lambda x: x * 2)
        >>> executor.func(5)
        10
        """
        self.func = func

    def execute(self, payload: Any) -> Any:
        """
        Execute the wrapped function with the given payload.

        Parameters
        ----------
        payload : Any
            The data to pass to the wrapped function.

        Returns
        -------
        Any
            The return value of the wrapped function.

        Raises
        ------
        Exception
            Any exception raised by the wrapped function propagates up.

        Examples
        --------
        >>> from insideLLMs.distributed import FunctionExecutor
        >>> executor = FunctionExecutor(len)
        >>> executor.execute([1, 2, 3, 4, 5])
        5
        >>> executor.execute("hello")
        5
        """
        return self.func(payload)


class WorkQueue:
    """
    Thread-safe priority queue for distributing work to workers.

    WorkQueue provides a thread-safe mechanism for enqueueing and dequeueing
    tasks with priority ordering. Higher priority tasks are dequeued first,
    with FIFO ordering for tasks of equal priority. The queue tracks pending
    tasks to support introspection and monitoring.

    Parameters
    ----------
    maxsize : int, optional
        Maximum number of items in the queue. If 0 (default), the queue
        is unbounded. If positive, `put()` blocks when the queue is full.

    Attributes
    ----------
    _queue : queue.PriorityQueue
        Internal priority queue for task storage.
    _pending : dict[str, Task]
        Dictionary tracking tasks that have been added but not yet retrieved.
    _lock : threading.Lock
        Lock for thread-safe access to pending dict.

    Examples
    --------
    Basic queue operations:

    >>> from insideLLMs.distributed import WorkQueue, Task
    >>> q = WorkQueue()
    >>> q.put(Task(id="task1", payload="data1"))
    >>> q.put(Task(id="task2", payload="data2"))
    >>> print(f"Queue size: {q.size}")
    Queue size: 2
    >>> task = q.get()
    >>> print(f"Got task: {task.id}")
    Got task: task1
    >>> q.task_done()

    Priority ordering:

    >>> from insideLLMs.distributed import WorkQueue, Task
    >>> q = WorkQueue()
    >>> q.put(Task(id="low", payload="x", priority=1))
    >>> q.put(Task(id="high", payload="y", priority=10))
    >>> q.put(Task(id="medium", payload="z", priority=5))
    >>> # Higher priority comes first
    >>> print(q.get().id)
    high
    >>> q.task_done()
    >>> print(q.get().id)
    medium
    >>> q.task_done()

    Non-blocking get with timeout:

    >>> from insideLLMs.distributed import WorkQueue, Task
    >>> q = WorkQueue()
    >>> result = q.get(timeout=0.1)  # Returns None if empty
    >>> print(result is None)
    True
    >>> q.put(Task(id="t1", payload="data"))
    >>> result = q.get(timeout=0.1)
    >>> print(result.id)
    t1

    Checking queue state:

    >>> from insideLLMs.distributed import WorkQueue, Task
    >>> q = WorkQueue()
    >>> print(f"Empty: {q.is_empty()}")
    Empty: True
    >>> q.put(Task(id="t1", payload="data"))
    >>> print(f"Empty: {q.is_empty()}, Size: {q.size}")
    Empty: False, Size: 1

    See Also
    --------
    Task : The items stored in the queue
    LocalWorker : Consumer of tasks from WorkQueue
    LocalDistributedExecutor : Manages WorkQueue and workers
    """

    def __init__(self, maxsize: int = 0):
        """
        Initialize the work queue.

        Parameters
        ----------
        maxsize : int, optional
            Maximum queue size. 0 means unlimited. Default is 0.

        Examples
        --------
        >>> from insideLLMs.distributed import WorkQueue
        >>> unbounded = WorkQueue()
        >>> bounded = WorkQueue(maxsize=100)
        """
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize)
        self._pending: dict[str, Task] = {}
        self._lock = threading.Lock()

    def put(self, task: Task) -> None:
        """
        Add a task to the queue.

        Tasks are ordered by priority (higher first), then by creation time
        (earlier first). This method is thread-safe and may block if the
        queue has a maxsize and is full.

        Parameters
        ----------
        task : Task
            The task to add to the queue.

        Examples
        --------
        >>> from insideLLMs.distributed import WorkQueue, Task
        >>> q = WorkQueue()
        >>> q.put(Task(id="t1", payload="first"))
        >>> q.put(Task(id="t2", payload="second", priority=10))
        >>> print(q.size)
        2
        """
        with self._lock:
            self._pending[task.id] = task
            # Use negative priority so higher priority comes first
            self._queue.put((-task.priority, task.created_at, task))

    def get(self, timeout: Optional[float] = None) -> Optional[Task]:
        """
        Get the highest priority task from the queue.

        Retrieves and removes the task with the highest priority. If multiple
        tasks have the same priority, returns the one added first (FIFO).

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds. If None, blocks indefinitely
            until a task is available. If 0, returns immediately.

        Returns
        -------
        Task or None
            The next task to process, or None if the queue is empty and
            the timeout expired.

        Examples
        --------
        >>> from insideLLMs.distributed import WorkQueue, Task
        >>> q = WorkQueue()
        >>> q.put(Task(id="t1", payload="data"))
        >>> task = q.get(timeout=1.0)
        >>> print(task.id)
        t1
        >>> empty_result = q.get(timeout=0.1)
        >>> print(empty_result is None)
        True
        """
        try:
            _, _, task = self._queue.get(timeout=timeout)
            with self._lock:
                if task.id in self._pending:
                    del self._pending[task.id]
            return task
        except queue.Empty:
            return None

    def task_done(self) -> None:
        """
        Signal that a previously retrieved task is complete.

        Should be called after finishing processing a task retrieved via
        `get()`. This is used internally by the queue to track completion.

        Examples
        --------
        >>> from insideLLMs.distributed import WorkQueue, Task
        >>> q = WorkQueue()
        >>> q.put(Task(id="t1", payload="data"))
        >>> task = q.get()
        >>> # Process task...
        >>> q.task_done()  # Signal completion
        """
        self._queue.task_done()

    @property
    def size(self) -> int:
        """
        Get the current number of tasks in the queue.

        Returns
        -------
        int
            Approximate number of tasks in the queue. Note that this may
            be slightly inaccurate in highly concurrent scenarios.

        Examples
        --------
        >>> from insideLLMs.distributed import WorkQueue, Task
        >>> q = WorkQueue()
        >>> print(q.size)
        0
        >>> q.put(Task(id="t1", payload="data"))
        >>> print(q.size)
        1
        """
        return self._queue.qsize()

    @property
    def pending_count(self) -> int:
        """
        Get the count of tasks that were added but not yet retrieved.

        Returns
        -------
        int
            Number of pending tasks.

        Examples
        --------
        >>> from insideLLMs.distributed import WorkQueue, Task
        >>> q = WorkQueue()
        >>> q.put(Task(id="t1", payload="d1"))
        >>> q.put(Task(id="t2", payload="d2"))
        >>> print(q.pending_count)
        2
        >>> _ = q.get()
        >>> print(q.pending_count)
        1
        """
        with self._lock:
            return len(self._pending)

    def is_empty(self) -> bool:
        """
        Check if the queue is empty.

        Returns
        -------
        bool
            True if the queue contains no tasks, False otherwise.

        Examples
        --------
        >>> from insideLLMs.distributed import WorkQueue, Task
        >>> q = WorkQueue()
        >>> print(q.is_empty())
        True
        >>> q.put(Task(id="t1", payload="data"))
        >>> print(q.is_empty())
        False
        """
        return self._queue.empty()


class ResultCollector:
    """
    Thread-safe collector for aggregating task results from multiple workers.

    ResultCollector provides a centralized storage for TaskResults produced
    by distributed workers. It supports callback registration for real-time
    result processing and provides aggregate statistics about execution
    success/failure rates.

    The collector is thread-safe, allowing multiple workers to add results
    concurrently without explicit synchronization.

    Attributes
    ----------
    _results : dict[str, TaskResult]
        Internal storage mapping task IDs to their results.
    _lock : threading.Lock
        Lock for thread-safe access to results.
    _callbacks : list[Callable[[TaskResult], None]]
        List of callbacks to invoke when results are added.

    Examples
    --------
    Basic result collection:

    >>> from insideLLMs.distributed import ResultCollector, TaskResult
    >>> collector = ResultCollector()
    >>> collector.add(TaskResult(task_id="t1", success=True, result=42))
    >>> collector.add(TaskResult(task_id="t2", success=True, result=84))
    >>> print(f"Collected {collector.count} results")
    Collected 2 results

    Retrieving results:

    >>> from insideLLMs.distributed import ResultCollector, TaskResult
    >>> collector = ResultCollector()
    >>> collector.add(TaskResult(task_id="t1", success=True, result="done"))
    >>> result = collector.get("t1")
    >>> print(f"Result: {result.result}")
    Result: done
    >>> missing = collector.get("nonexistent")
    >>> print(missing is None)
    True

    Registering callbacks for real-time processing:

    >>> from insideLLMs.distributed import ResultCollector, TaskResult
    >>> collector = ResultCollector()
    >>> processed = []
    >>> collector.on_result(lambda r: processed.append(r.task_id))
    >>> collector.add(TaskResult(task_id="t1", success=True))
    >>> collector.add(TaskResult(task_id="t2", success=True))
    >>> print(processed)
    ['t1', 't2']

    Tracking success/failure rates:

    >>> from insideLLMs.distributed import ResultCollector, TaskResult
    >>> collector = ResultCollector()
    >>> collector.add(TaskResult(task_id="t1", success=True))
    >>> collector.add(TaskResult(task_id="t2", success=False, error="timeout"))
    >>> collector.add(TaskResult(task_id="t3", success=True))
    >>> print(f"Success: {collector.success_count}, Failed: {collector.failure_count}")
    Success: 2, Failed: 1
    >>> rate = collector.success_count / collector.count
    >>> print(f"Success rate: {rate:.1%}")
    Success rate: 66.7%

    Getting all results for processing:

    >>> from insideLLMs.distributed import ResultCollector, TaskResult
    >>> collector = ResultCollector()
    >>> collector.add(TaskResult(task_id="t1", success=True, result=10))
    >>> collector.add(TaskResult(task_id="t2", success=True, result=20))
    >>> all_results = collector.get_all()
    >>> total = sum(r.result for r in all_results if r.success)
    >>> print(f"Total: {total}")
    Total: 30

    See Also
    --------
    TaskResult : The result type stored in the collector
    LocalWorker : Adds results to the collector
    LocalDistributedExecutor : Owns and provides access to ResultCollector
    """

    def __init__(self):
        """
        Initialize the result collector.

        Creates an empty result storage with no registered callbacks.

        Examples
        --------
        >>> from insideLLMs.distributed import ResultCollector
        >>> collector = ResultCollector()
        >>> print(collector.count)
        0
        """
        self._results: dict[str, TaskResult] = {}
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[TaskResult], None]] = []

    def add(self, result: TaskResult) -> None:
        """
        Add a task result to the collector.

        Stores the result and invokes all registered callbacks. If a result
        for the same task_id already exists, it will be overwritten.

        Callbacks are invoked outside the lock and any exceptions they raise
        are suppressed to prevent affecting other callbacks or the worker.

        Parameters
        ----------
        result : TaskResult
            The result to add.

        Examples
        --------
        >>> from insideLLMs.distributed import ResultCollector, TaskResult
        >>> collector = ResultCollector()
        >>> collector.add(TaskResult(task_id="t1", success=True, result="done"))
        >>> print(collector.count)
        1

        With callback:

        >>> from insideLLMs.distributed import ResultCollector, TaskResult
        >>> collector = ResultCollector()
        >>> collector.on_result(lambda r: print(f"Got result: {r.task_id}"))
        >>> collector.add(TaskResult(task_id="t1", success=True))
        Got result: t1
        """
        with self._lock:
            self._results[result.task_id] = result
        for callback in self._callbacks:
            with contextlib.suppress(Exception):
                callback(result)

    def get(self, task_id: str) -> Optional[TaskResult]:
        """
        Get a specific result by task ID.

        Parameters
        ----------
        task_id : str
            The ID of the task whose result to retrieve.

        Returns
        -------
        TaskResult or None
            The result for the task, or None if not found.

        Examples
        --------
        >>> from insideLLMs.distributed import ResultCollector, TaskResult
        >>> collector = ResultCollector()
        >>> collector.add(TaskResult(task_id="t1", success=True, result=42))
        >>> result = collector.get("t1")
        >>> print(result.result)
        42
        >>> print(collector.get("missing") is None)
        True
        """
        with self._lock:
            return self._results.get(task_id)

    def get_all(self) -> list[TaskResult]:
        """
        Get all collected results.

        Returns
        -------
        list[TaskResult]
            List of all results in the collector. The order is not guaranteed.

        Examples
        --------
        >>> from insideLLMs.distributed import ResultCollector, TaskResult
        >>> collector = ResultCollector()
        >>> collector.add(TaskResult(task_id="t1", success=True, result=1))
        >>> collector.add(TaskResult(task_id="t2", success=True, result=2))
        >>> results = collector.get_all()
        >>> len(results)
        2
        >>> sum(r.result for r in results)
        3
        """
        with self._lock:
            return list(self._results.values())

    def on_result(self, callback: Callable[[TaskResult], None]) -> None:
        """
        Register a callback to be invoked when results are added.

        Callbacks are invoked synchronously after each result is added.
        Multiple callbacks can be registered and are called in order.
        Exceptions in callbacks are suppressed.

        Parameters
        ----------
        callback : Callable[[TaskResult], None]
            Function to call with each new result. Receives the TaskResult
            as its only argument.

        Examples
        --------
        >>> from insideLLMs.distributed import ResultCollector, TaskResult
        >>> collector = ResultCollector()
        >>> successes = []
        >>> failures = []
        >>> collector.on_result(
        ...     lambda r: successes.append(r) if r.success else failures.append(r)
        ... )
        >>> collector.add(TaskResult(task_id="t1", success=True))
        >>> collector.add(TaskResult(task_id="t2", success=False, error="err"))
        >>> print(len(successes), len(failures))
        1 1
        """
        self._callbacks.append(callback)

    @property
    def count(self) -> int:
        """
        Get the total number of results collected.

        Returns
        -------
        int
            Total count of results (both successful and failed).

        Examples
        --------
        >>> from insideLLMs.distributed import ResultCollector, TaskResult
        >>> collector = ResultCollector()
        >>> collector.add(TaskResult(task_id="t1", success=True))
        >>> collector.add(TaskResult(task_id="t2", success=False, error="err"))
        >>> print(collector.count)
        2
        """
        with self._lock:
            return len(self._results)

    @property
    def success_count(self) -> int:
        """
        Get the count of successful results.

        Returns
        -------
        int
            Number of results where success=True.

        Examples
        --------
        >>> from insideLLMs.distributed import ResultCollector, TaskResult
        >>> collector = ResultCollector()
        >>> collector.add(TaskResult(task_id="t1", success=True))
        >>> collector.add(TaskResult(task_id="t2", success=True))
        >>> collector.add(TaskResult(task_id="t3", success=False, error="err"))
        >>> print(collector.success_count)
        2
        """
        with self._lock:
            return sum(1 for r in self._results.values() if r.success)

    @property
    def failure_count(self) -> int:
        """
        Get the count of failed results.

        Returns
        -------
        int
            Number of results where success=False.

        Examples
        --------
        >>> from insideLLMs.distributed import ResultCollector, TaskResult
        >>> collector = ResultCollector()
        >>> collector.add(TaskResult(task_id="t1", success=True))
        >>> collector.add(TaskResult(task_id="t2", success=False, error="timeout"))
        >>> collector.add(TaskResult(task_id="t3", success=False, error="api_error"))
        >>> print(collector.failure_count)
        2
        """
        with self._lock:
            return sum(1 for r in self._results.values() if not r.success)


class LocalWorker:
    """
    A worker that processes tasks from a queue in a dedicated thread.

    LocalWorker runs in a daemon thread, continuously polling a WorkQueue for
    tasks. Each task is processed using the provided TaskExecutor, and results
    are added to the ResultCollector. The worker maintains WorkerInfo for
    status monitoring.

    Workers are typically managed by LocalDistributedExecutor rather than
    being used directly. However, they can be instantiated manually for
    custom distributed processing setups.

    Parameters
    ----------
    worker_id : str
        Unique identifier for this worker. Used in logging and result tracking.
    executor : TaskExecutor
        The executor to use for processing task payloads.
    work_queue : WorkQueue
        Queue to poll for tasks.
    result_collector : ResultCollector
        Collector to which results are added.

    Attributes
    ----------
    worker_id : str
        The worker's identifier.
    executor : TaskExecutor
        The task executor instance.
    work_queue : WorkQueue
        The queue being polled.
    result_collector : ResultCollector
        The result collector.
    info : WorkerInfo
        Real-time status information about this worker.

    Examples
    --------
    Creating and starting a worker manually:

    >>> from insideLLMs.distributed import (
    ...     LocalWorker, FunctionExecutor, WorkQueue, ResultCollector, Task
    ... )
    >>> queue = WorkQueue()
    >>> collector = ResultCollector()
    >>> executor = FunctionExecutor(lambda x: x * 2)
    >>> worker = LocalWorker("w0", executor, queue, collector)
    >>> worker.start()
    >>> queue.put(Task(id="t1", payload=5))
    >>> import time; time.sleep(0.5)  # Wait for processing
    >>> result = collector.get("t1")
    >>> print(f"Result: {result.result}")
    Result: 10
    >>> worker.stop()

    Monitoring worker status:

    >>> from insideLLMs.distributed import (
    ...     LocalWorker, FunctionExecutor, WorkQueue, ResultCollector, WorkerStatus
    ... )
    >>> queue = WorkQueue()
    >>> collector = ResultCollector()
    >>> executor = FunctionExecutor(lambda x: x)
    >>> worker = LocalWorker("w0", executor, queue, collector)
    >>> print(worker.info.status == WorkerStatus.IDLE)
    True
    >>> worker.start()
    >>> # Worker is now running
    >>> print(worker.info.status == WorkerStatus.IDLE)
    True
    >>> worker.stop()
    >>> print(worker.info.status == WorkerStatus.SHUTDOWN)
    True

    Multiple workers sharing a queue:

    >>> from insideLLMs.distributed import (
    ...     LocalWorker, FunctionExecutor, WorkQueue, ResultCollector, Task
    ... )
    >>> import time
    >>> queue = WorkQueue()
    >>> collector = ResultCollector()
    >>> executor = FunctionExecutor(lambda x: x ** 2)
    >>> workers = [
    ...     LocalWorker(f"w{i}", executor, queue, collector)
    ...     for i in range(3)
    ... ]
    >>> for w in workers:
    ...     w.start()
    >>> for i in range(10):
    ...     queue.put(Task(id=f"t{i}", payload=i))
    >>> time.sleep(1.0)  # Wait for processing
    >>> print(f"Completed: {collector.count}")
    Completed: 10
    >>> for w in workers:
    ...     w.stop()

    See Also
    --------
    LocalDistributedExecutor : High-level executor that manages LocalWorkers
    WorkQueue : The queue from which workers retrieve tasks
    ResultCollector : Where workers store their results
    WorkerInfo : Status information maintained by each worker
    """

    def __init__(
        self,
        worker_id: str,
        executor: TaskExecutor,
        work_queue: WorkQueue,
        result_collector: ResultCollector,
    ):
        """
        Initialize a local worker.

        Parameters
        ----------
        worker_id : str
            Unique identifier for this worker.
        executor : TaskExecutor
            Executor for processing task payloads.
        work_queue : WorkQueue
            Queue to retrieve tasks from.
        result_collector : ResultCollector
            Collector to store results in.

        Examples
        --------
        >>> from insideLLMs.distributed import (
        ...     LocalWorker, FunctionExecutor, WorkQueue, ResultCollector
        ... )
        >>> worker = LocalWorker(
        ...     worker_id="worker_0",
        ...     executor=FunctionExecutor(str.upper),
        ...     work_queue=WorkQueue(),
        ...     result_collector=ResultCollector()
        ... )
        >>> print(worker.worker_id)
        worker_0
        """
        self.worker_id = worker_id
        self.executor = executor
        self.work_queue = work_queue
        self.result_collector = result_collector
        self.info = WorkerInfo(id=worker_id)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """
        Start the worker thread.

        Launches a daemon thread that polls the work queue for tasks.
        The worker will process tasks until stop() is called.

        Examples
        --------
        >>> from insideLLMs.distributed import (
        ...     LocalWorker, FunctionExecutor, WorkQueue, ResultCollector
        ... )
        >>> worker = LocalWorker(
        ...     "w0", FunctionExecutor(lambda x: x),
        ...     WorkQueue(), ResultCollector()
        ... )
        >>> worker.start()
        >>> # Worker is now running in background
        >>> worker.stop()
        """
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop the worker thread.

        Signals the worker to stop and waits up to 5 seconds for the thread
        to finish. After stopping, the worker's status is set to SHUTDOWN.

        Examples
        --------
        >>> from insideLLMs.distributed import (
        ...     LocalWorker, FunctionExecutor, WorkQueue, ResultCollector, WorkerStatus
        ... )
        >>> worker = LocalWorker(
        ...     "w0", FunctionExecutor(lambda x: x),
        ...     WorkQueue(), ResultCollector()
        ... )
        >>> worker.start()
        >>> worker.stop()
        >>> print(worker.info.status == WorkerStatus.SHUTDOWN)
        True
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self.info.status = WorkerStatus.SHUTDOWN

    def _run(self) -> None:
        """
        Main worker loop (internal).

        Continuously polls the work queue for tasks and processes them.
        Updates worker status and heartbeat during execution. Results
        (success or failure) are added to the result collector.

        This method runs in a separate thread and should not be called
        directly. Use start() and stop() to control the worker.
        """
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
    """
    Coordinate multiple local workers for parallel task execution.

    LocalDistributedExecutor manages a pool of LocalWorker threads that process
    tasks from a shared WorkQueue. It provides a simple interface for submitting
    tasks, waiting for completion, and retrieving results. The executor can be
    used as a context manager for automatic cleanup.

    This is the recommended way to run parallel tasks on a single machine using
    threads. For CPU-bound tasks that need to bypass the GIL, use
    ProcessPoolDistributedExecutor instead.

    Parameters
    ----------
    executor : TaskExecutor
        The task executor that workers will use to process payloads.
    num_workers : int, optional
        Number of worker threads to create. Default is 4.
    use_processes : bool, optional
        Reserved for future use. Currently has no effect as workers
        always use threads. Default is False.

    Attributes
    ----------
    executor : TaskExecutor
        The task executor instance.
    num_workers : int
        Number of workers.
    use_processes : bool
        Whether to use processes (reserved).
    work_queue : WorkQueue
        The shared work queue.
    result_collector : ResultCollector
        The shared result collector.

    Examples
    --------
    Basic usage with context manager:

    >>> from insideLLMs.distributed import (
    ...     LocalDistributedExecutor, FunctionExecutor, Task
    ... )
    >>> executor = FunctionExecutor(lambda x: x ** 2)
    >>> with LocalDistributedExecutor(executor, num_workers=2) as dist:
    ...     dist.submit(Task(id="t1", payload=5))
    ...     dist.submit(Task(id="t2", payload=10))
    ...     results = dist.wait_for_completion()
    >>> for r in sorted(results, key=lambda x: x.task_id):
    ...     print(f"{r.task_id}: {r.result}")
    t1: 25
    t2: 100

    Submitting a batch of tasks:

    >>> from insideLLMs.distributed import (
    ...     LocalDistributedExecutor, FunctionExecutor, Task
    ... )
    >>> executor = FunctionExecutor(str.upper)
    >>> tasks = [Task(id=f"t{i}", payload=f"hello{i}") for i in range(5)]
    >>> with LocalDistributedExecutor(executor, num_workers=3) as dist:
    ...     dist.submit_batch(tasks)
    ...     results = dist.wait_for_completion()
    >>> print(len(results))
    5

    LLM experiment simulation:

    >>> from insideLLMs.distributed import (
    ...     LocalDistributedExecutor, FunctionExecutor, Task
    ... )
    >>> import time
    >>> def mock_llm(prompt):
    ...     time.sleep(0.01)  # Simulate API latency
    ...     return f"Response to: {prompt}"
    >>> executor = FunctionExecutor(mock_llm)
    >>> prompts = ["What is AI?", "Explain ML", "Define NLP"]
    >>> tasks = [Task(id=f"p{i}", payload=p) for i, p in enumerate(prompts)]
    >>> with LocalDistributedExecutor(executor, num_workers=3) as dist:
    ...     dist.submit_batch(tasks)
    ...     results = dist.wait_for_completion()
    >>> all(r.success for r in results)
    True

    Manual start/stop without context manager:

    >>> from insideLLMs.distributed import (
    ...     LocalDistributedExecutor, FunctionExecutor, Task
    ... )
    >>> executor = FunctionExecutor(lambda x: x * 2)
    >>> dist = LocalDistributedExecutor(executor, num_workers=2)
    >>> dist.start()
    >>> dist.submit(Task(id="t1", payload=5))
    >>> import time; time.sleep(0.5)
    >>> results = dist.get_results()
    >>> print(results[0].result)
    10
    >>> dist.stop()

    Monitoring workers:

    >>> from insideLLMs.distributed import (
    ...     LocalDistributedExecutor, FunctionExecutor, Task
    ... )
    >>> executor = FunctionExecutor(lambda x: x)
    >>> with LocalDistributedExecutor(executor, num_workers=3) as dist:
    ...     info = dist.get_worker_info()
    ...     print(f"Workers: {len(info)}")
    ...     print(f"IDs: {[w.id for w in info]}")
    Workers: 3
    IDs: ['worker_0', 'worker_1', 'worker_2']

    Handling failures:

    >>> from insideLLMs.distributed import (
    ...     LocalDistributedExecutor, FunctionExecutor, Task
    ... )
    >>> def risky_func(x):
    ...     if x < 0:
    ...         raise ValueError("Negative!")
    ...     return x * 2
    >>> executor = FunctionExecutor(risky_func)
    >>> with LocalDistributedExecutor(executor, num_workers=2) as dist:
    ...     dist.submit(Task(id="good", payload=5))
    ...     dist.submit(Task(id="bad", payload=-1))
    ...     results = dist.wait_for_completion()
    >>> success = [r for r in results if r.success]
    >>> failed = [r for r in results if not r.success]
    >>> print(f"Success: {len(success)}, Failed: {len(failed)}")
    Success: 1, Failed: 1

    See Also
    --------
    ProcessPoolDistributedExecutor : Process-based execution for CPU-bound tasks
    DistributedExperimentRunner : High-level interface for LLM experiments
    LocalWorker : Individual worker instances managed by this executor
    """

    def __init__(
        self,
        executor: TaskExecutor,
        num_workers: int = 4,
        use_processes: bool = False,
    ):
        """
        Initialize the local distributed executor.

        Parameters
        ----------
        executor : TaskExecutor
            Task executor to use for processing payloads.
        num_workers : int, optional
            Number of worker threads. Default is 4.
        use_processes : bool, optional
            Reserved for future use. Default is False.

        Examples
        --------
        >>> from insideLLMs.distributed import LocalDistributedExecutor, FunctionExecutor
        >>> executor = FunctionExecutor(lambda x: x * 2)
        >>> dist = LocalDistributedExecutor(executor, num_workers=8)
        >>> print(dist.num_workers)
        8
        """
        self.executor = executor
        self.num_workers = num_workers
        self.use_processes = use_processes
        self.work_queue = WorkQueue()
        self.result_collector = ResultCollector()
        self._workers: list[LocalWorker] = []
        self._started = False

    def start(self) -> None:
        """
        Start all worker threads.

        Creates and starts the configured number of LocalWorker instances.
        This method is idempotent - calling it multiple times has no effect
        after the first call.

        Examples
        --------
        >>> from insideLLMs.distributed import LocalDistributedExecutor, FunctionExecutor
        >>> executor = FunctionExecutor(lambda x: x)
        >>> dist = LocalDistributedExecutor(executor, num_workers=2)
        >>> dist.start()
        >>> print(len(dist.get_worker_info()))
        2
        >>> dist.stop()
        """
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
        """
        Stop all worker threads.

        Signals each worker to stop and waits for them to finish.
        Clears the worker list and resets the started flag.

        Examples
        --------
        >>> from insideLLMs.distributed import LocalDistributedExecutor, FunctionExecutor
        >>> executor = FunctionExecutor(lambda x: x)
        >>> dist = LocalDistributedExecutor(executor, num_workers=2)
        >>> dist.start()
        >>> dist.stop()
        >>> print(len(dist._workers))
        0
        """
        for worker in self._workers:
            worker.stop()
        self._workers.clear()
        self._started = False

    def submit(self, task: Task) -> None:
        """
        Submit a single task for execution.

        If the executor hasn't been started yet, it will be started
        automatically. The task is added to the work queue where it
        will be picked up by an available worker.

        Parameters
        ----------
        task : Task
            The task to execute.

        Examples
        --------
        >>> from insideLLMs.distributed import (
        ...     LocalDistributedExecutor, FunctionExecutor, Task
        ... )
        >>> executor = FunctionExecutor(lambda x: x * 2)
        >>> with LocalDistributedExecutor(executor, num_workers=2) as dist:
        ...     dist.submit(Task(id="t1", payload=5))
        ...     results = dist.wait_for_completion()
        >>> print(results[0].result)
        10
        """
        if not self._started:
            self.start()
        self.work_queue.put(task)

    def submit_batch(self, tasks: list[Task]) -> None:
        """
        Submit multiple tasks for execution.

        Convenience method that calls submit() for each task.

        Parameters
        ----------
        tasks : list[Task]
            List of tasks to execute.

        Examples
        --------
        >>> from insideLLMs.distributed import (
        ...     LocalDistributedExecutor, FunctionExecutor, Task
        ... )
        >>> executor = FunctionExecutor(lambda x: x ** 2)
        >>> tasks = [Task(id=f"t{i}", payload=i) for i in range(5)]
        >>> with LocalDistributedExecutor(executor, num_workers=3) as dist:
        ...     dist.submit_batch(tasks)
        ...     results = dist.wait_for_completion()
        >>> len(results)
        5
        """
        for task in tasks:
            self.submit(task)

    def wait_for_completion(
        self,
        timeout: Optional[float] = None,
        poll_interval: float = 0.1,
    ) -> list[TaskResult]:
        """
        Wait for all submitted tasks to complete.

        Blocks until the work queue is empty and all workers are idle,
        or until the timeout is reached.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds. If None, waits indefinitely.
        poll_interval : float, optional
            How often to check completion status. Default is 0.1 seconds.

        Returns
        -------
        list[TaskResult]
            All results collected so far.

        Examples
        --------
        >>> from insideLLMs.distributed import (
        ...     LocalDistributedExecutor, FunctionExecutor, Task
        ... )
        >>> executor = FunctionExecutor(lambda x: x * 2)
        >>> with LocalDistributedExecutor(executor, num_workers=2) as dist:
        ...     dist.submit_batch([Task(id=f"t{i}", payload=i) for i in range(10)])
        ...     results = dist.wait_for_completion(timeout=5.0)
        >>> len(results)
        10

        With timeout that may expire:

        >>> from insideLLMs.distributed import (
        ...     LocalDistributedExecutor, FunctionExecutor, Task
        ... )
        >>> import time
        >>> def slow_func(x):
        ...     time.sleep(0.5)
        ...     return x
        >>> executor = FunctionExecutor(slow_func)
        >>> with LocalDistributedExecutor(executor, num_workers=1) as dist:
        ...     dist.submit_batch([Task(id=f"t{i}", payload=i) for i in range(10)])
        ...     results = dist.wait_for_completion(timeout=1.0)
        >>> len(results) < 10  # Timeout hit before all complete
        True
        """
        start_time = time.time()
        self.work_queue.size + len(self._workers)  # Rough estimate

        while self.work_queue.size > 0 or any(
            w.info.status == WorkerStatus.BUSY for w in self._workers
        ):
            if timeout and (time.time() - start_time) > timeout:
                break
            time.sleep(poll_interval)

        return self.result_collector.get_all()

    def get_results(self) -> list[TaskResult]:
        """
        Get all collected results without waiting.

        Returns
        -------
        list[TaskResult]
            All results collected so far, including incomplete batches.

        Examples
        --------
        >>> from insideLLMs.distributed import (
        ...     LocalDistributedExecutor, FunctionExecutor, Task
        ... )
        >>> executor = FunctionExecutor(lambda x: x)
        >>> with LocalDistributedExecutor(executor, num_workers=2) as dist:
        ...     dist.submit(Task(id="t1", payload="data"))
        ...     import time; time.sleep(0.2)
        ...     results = dist.get_results()
        >>> len(results) >= 0  # May or may not have completed
        True
        """
        return self.result_collector.get_all()

    def get_worker_info(self) -> list[WorkerInfo]:
        """
        Get status information for all workers.

        Returns
        -------
        list[WorkerInfo]
            List of WorkerInfo objects, one per worker.

        Examples
        --------
        >>> from insideLLMs.distributed import (
        ...     LocalDistributedExecutor, FunctionExecutor
        ... )
        >>> executor = FunctionExecutor(lambda x: x)
        >>> with LocalDistributedExecutor(executor, num_workers=4) as dist:
        ...     info = dist.get_worker_info()
        ...     for w in info:
        ...         print(f"{w.id}: {w.status.value}")
        worker_0: idle
        worker_1: idle
        worker_2: idle
        worker_3: idle
        """
        return [w.info for w in self._workers]

    def __enter__(self) -> "LocalDistributedExecutor":
        """
        Enter context manager.

        Starts all workers and returns self.

        Returns
        -------
        LocalDistributedExecutor
            This executor instance.

        Examples
        --------
        >>> from insideLLMs.distributed import LocalDistributedExecutor, FunctionExecutor
        >>> executor = FunctionExecutor(lambda x: x)
        >>> with LocalDistributedExecutor(executor) as dist:
        ...     pass  # Workers are running
        >>> # Workers stopped automatically
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit context manager.

        Stops all workers.

        Parameters
        ----------
        exc_type : type
            Exception type if an exception was raised.
        exc_val : Exception
            Exception instance if raised.
        exc_tb : traceback
            Traceback if an exception was raised.
        """
        self.stop()


class DistributedCheckpointManager:
    """
    Manage checkpoints for fault-tolerant distributed execution.

    DistributedCheckpointManager provides persistence for distributed execution
    state, enabling recovery from failures. It saves pending tasks and completed
    results to disk, allowing execution to resume from where it left off after
    a crash or interruption.

    Checkpoints are stored as pickle files in a configurable directory. Each
    checkpoint includes a unique ID, timestamp, pending tasks, completed results,
    and optional metadata.

    Parameters
    ----------
    checkpoint_dir : str, optional
        Directory to store checkpoint files. If None, uses a subdirectory
        named 'insidellms_checkpoints' in the system temp directory.
        The directory is created if it doesn't exist.

    Attributes
    ----------
    checkpoint_dir : Path
        Path object for the checkpoint directory.

    Examples
    --------
    Basic checkpoint operations:

    >>> from insideLLMs.distributed import (
    ...     DistributedCheckpointManager, Task, TaskResult
    ... )
    >>> import tempfile, os
    >>> manager = DistributedCheckpointManager(tempfile.mkdtemp())
    >>> tasks = [Task(id=f"t{i}", payload=i) for i in range(5)]
    >>> results = [TaskResult(task_id="t0", success=True, result=0)]
    >>> path = manager.save("exp_001", tasks[1:], results)
    >>> print(os.path.exists(path))
    True
    >>> manager.delete("exp_001")

    Resuming from checkpoint:

    >>> from insideLLMs.distributed import (
    ...     DistributedCheckpointManager, Task, TaskResult
    ... )
    >>> import tempfile
    >>> manager = DistributedCheckpointManager(tempfile.mkdtemp())
    >>> # Save initial state
    >>> tasks = [Task(id=f"t{i}", payload=i) for i in range(5)]
    >>> results = [TaskResult(task_id="t0", success=True, result=0)]
    >>> manager.save("exp_001", tasks[1:], results)
    >>> # Later: resume from checkpoint
    >>> pending, completed, meta = manager.load("exp_001")
    >>> print(f"Pending: {len(pending)}, Completed: {len(completed)}")
    Pending: 4, Completed: 1
    >>> manager.delete("exp_001")

    Listing available checkpoints:

    >>> from insideLLMs.distributed import (
    ...     DistributedCheckpointManager, Task, TaskResult
    ... )
    >>> import tempfile
    >>> manager = DistributedCheckpointManager(tempfile.mkdtemp())
    >>> manager.save("exp_001", [], [])
    >>> manager.save("exp_002", [], [])
    >>> checkpoints = manager.list_checkpoints()
    >>> "exp_001" in checkpoints and "exp_002" in checkpoints
    True
    >>> manager.delete("exp_001")
    >>> manager.delete("exp_002")

    With metadata:

    >>> from insideLLMs.distributed import (
    ...     DistributedCheckpointManager, Task, TaskResult
    ... )
    >>> import tempfile
    >>> manager = DistributedCheckpointManager(tempfile.mkdtemp())
    >>> metadata = {"model": "gpt-4", "batch_size": 10}
    >>> manager.save("exp_meta", [], [], metadata=metadata)
    >>> _, _, loaded_meta = manager.load("exp_meta")
    >>> print(loaded_meta["model"])
    gpt-4
    >>> manager.delete("exp_meta")

    See Also
    --------
    ProcessPoolDistributedExecutor : Uses checkpoints for fault tolerance
    DistributedExperimentRunner : High-level experiment runner with checkpointing
    """

    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        Initialize the checkpoint manager.

        Parameters
        ----------
        checkpoint_dir : str, optional
            Directory for storing checkpoint files. Creates the directory
            if it doesn't exist. Defaults to system temp directory.

        Examples
        --------
        >>> from insideLLMs.distributed import DistributedCheckpointManager
        >>> import tempfile
        >>> manager = DistributedCheckpointManager(tempfile.mkdtemp())
        >>> print(manager.checkpoint_dir.exists())
        True

        Using default temp directory:

        >>> from insideLLMs.distributed import DistributedCheckpointManager
        >>> manager = DistributedCheckpointManager()
        >>> "insidellms_checkpoints" in str(manager.checkpoint_dir)
        True
        """
        self.checkpoint_dir = (
            Path(checkpoint_dir or tempfile.gettempdir()) / "insidellms_checkpoints"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """
        Get the file path for a checkpoint.

        Parameters
        ----------
        checkpoint_id : str
            The checkpoint identifier.

        Returns
        -------
        Path
            Path object for the checkpoint file.
        """
        return self.checkpoint_dir / f"{checkpoint_id}.checkpoint"

    def save(
        self,
        checkpoint_id: str,
        pending_tasks: list[Task],
        completed_results: list[TaskResult],
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Save execution state to a checkpoint file.

        Creates or overwrites a checkpoint with the current execution state.
        The checkpoint can later be loaded to resume execution.

        Parameters
        ----------
        checkpoint_id : str
            Unique identifier for this checkpoint. Used as the filename
            (with .checkpoint extension).
        pending_tasks : list[Task]
            Tasks that haven't been processed yet.
        completed_results : list[TaskResult]
            Results that have been collected so far.
        metadata : dict[str, Any], optional
            Additional metadata to store (e.g., model name, parameters).

        Returns
        -------
        str
            Absolute path to the saved checkpoint file.

        Examples
        --------
        >>> from insideLLMs.distributed import (
        ...     DistributedCheckpointManager, Task, TaskResult
        ... )
        >>> import tempfile, os
        >>> manager = DistributedCheckpointManager(tempfile.mkdtemp())
        >>> pending = [Task(id="t1", payload="data1")]
        >>> completed = [TaskResult(task_id="t0", success=True, result="done")]
        >>> path = manager.save("my_checkpoint", pending, completed)
        >>> os.path.basename(path)
        'my_checkpoint.checkpoint'
        >>> manager.delete("my_checkpoint")

        With metadata:

        >>> from insideLLMs.distributed import (
        ...     DistributedCheckpointManager, Task, TaskResult
        ... )
        >>> import tempfile
        >>> manager = DistributedCheckpointManager(tempfile.mkdtemp())
        >>> meta = {"experiment": "test", "iteration": 5}
        >>> path = manager.save("exp_with_meta", [], [], metadata=meta)
        >>> manager.delete("exp_with_meta")
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

    def load(self, checkpoint_id: str) -> tuple[list[Task], list[TaskResult], dict[str, Any]]:
        """
        Load execution state from a checkpoint file.

        Reconstructs the execution state from a previously saved checkpoint,
        allowing execution to resume from where it left off.

        Parameters
        ----------
        checkpoint_id : str
            Identifier of the checkpoint to load.

        Returns
        -------
        tuple[list[Task], list[TaskResult], dict[str, Any]]
            A tuple containing:
            - pending_tasks: Tasks that still need to be processed
            - completed_results: Results that were already collected
            - metadata: Any metadata that was saved with the checkpoint

        Raises
        ------
        FileNotFoundError
            If no checkpoint with the given ID exists.

        Examples
        --------
        >>> from insideLLMs.distributed import (
        ...     DistributedCheckpointManager, Task, TaskResult
        ... )
        >>> import tempfile
        >>> manager = DistributedCheckpointManager(tempfile.mkdtemp())
        >>> # Save a checkpoint
        >>> pending = [Task(id="t1", payload="data")]
        >>> completed = [TaskResult(task_id="t0", success=True, result=42)]
        >>> manager.save("test_cp", pending, completed, {"key": "value"})
        >>> # Load it back
        >>> loaded_pending, loaded_completed, meta = manager.load("test_cp")
        >>> print(loaded_pending[0].id)
        t1
        >>> print(loaded_completed[0].result)
        42
        >>> print(meta["key"])
        value
        >>> manager.delete("test_cp")

        Handling missing checkpoint:

        >>> from insideLLMs.distributed import DistributedCheckpointManager
        >>> import tempfile
        >>> manager = DistributedCheckpointManager(tempfile.mkdtemp())
        >>> try:
        ...     manager.load("nonexistent")
        ... except FileNotFoundError as e:
        ...     print("Checkpoint not found")
        Checkpoint not found
        """
        path = self._get_checkpoint_path(checkpoint_id)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        pending_tasks = [Task.from_dict(t) for t in data["pending_tasks"]]
        completed_results = [TaskResult(**r) for r in data["completed_results"]]

        return pending_tasks, completed_results, data.get("metadata", {})

    def list_checkpoints(self) -> list[str]:
        """
        List all available checkpoint IDs.

        Returns
        -------
        list[str]
            List of checkpoint identifiers (without file extension).

        Examples
        --------
        >>> from insideLLMs.distributed import DistributedCheckpointManager
        >>> import tempfile
        >>> manager = DistributedCheckpointManager(tempfile.mkdtemp())
        >>> manager.save("cp_a", [], [])
        >>> manager.save("cp_b", [], [])
        >>> sorted(manager.list_checkpoints())
        ['cp_a', 'cp_b']
        >>> manager.delete("cp_a")
        >>> manager.delete("cp_b")
        """
        return [p.stem for p in self.checkpoint_dir.glob("*.checkpoint")]

    def delete(self, checkpoint_id: str) -> None:
        """
        Delete a checkpoint file.

        Removes the checkpoint file if it exists. No error is raised if
        the checkpoint doesn't exist.

        Parameters
        ----------
        checkpoint_id : str
            Identifier of the checkpoint to delete.

        Examples
        --------
        >>> from insideLLMs.distributed import DistributedCheckpointManager
        >>> import tempfile
        >>> manager = DistributedCheckpointManager(tempfile.mkdtemp())
        >>> manager.save("to_delete", [], [])
        >>> "to_delete" in manager.list_checkpoints()
        True
        >>> manager.delete("to_delete")
        >>> "to_delete" in manager.list_checkpoints()
        False
        >>> manager.delete("already_deleted")  # No error
        """
        path = self._get_checkpoint_path(checkpoint_id)
        if path.exists():
            path.unlink()


class ProcessPoolDistributedExecutor:
    """
    Execute tasks across multiple processes with optional checkpointing.

    ProcessPoolDistributedExecutor uses Python's ProcessPoolExecutor for true
    parallel execution, bypassing the GIL. This is ideal for CPU-bound tasks.
    It also supports fault tolerance through periodic checkpointing.

    Unlike LocalDistributedExecutor which uses threads, this executor runs
    tasks in separate processes, allowing full utilization of multiple CPU cores.
    However, the function and payloads must be pickleable.

    Parameters
    ----------
    func : Callable[[Any], Any]
        Function to execute for each task payload. Must be pickleable
        (typically a module-level function, not a lambda or closure).
    num_workers : int, optional
        Number of worker processes. Default is 4.
    checkpoint_manager : DistributedCheckpointManager, optional
        Checkpoint manager for fault tolerance. If provided, execution
        state can be saved periodically and restored after failures.

    Attributes
    ----------
    func : Callable[[Any], Any]
        The execution function.
    num_workers : int
        Number of worker processes.
    checkpoint_manager : DistributedCheckpointManager or None
        The checkpoint manager instance.

    Examples
    --------
    Basic CPU-bound processing:

    >>> from insideLLMs.distributed import ProcessPoolDistributedExecutor, Task
    >>> def cpu_intensive(x):
    ...     return sum(i*i for i in range(x))
    >>> executor = ProcessPoolDistributedExecutor(cpu_intensive, num_workers=4)
    >>> tasks = [Task(id=f"t{i}", payload=1000*(i+1)) for i in range(4)]
    >>> executor.submit_batch(tasks)
    >>> results = executor.run()
    >>> all(r.success for r in results)
    True

    With progress callback:

    >>> from insideLLMs.distributed import ProcessPoolDistributedExecutor, Task
    >>> def square(x):
    ...     return x ** 2
    >>> completed_tasks = []
    >>> def on_progress(done, total):
    ...     completed_tasks.append(done)
    >>> executor = ProcessPoolDistributedExecutor(square, num_workers=2)
    >>> executor.submit_batch([Task(id=f"t{i}", payload=i) for i in range(5)])
    >>> results = executor.run(progress_callback=on_progress)
    >>> len(completed_tasks) == 5
    True

    With fault-tolerant checkpointing:

    >>> from insideLLMs.distributed import (
    ...     ProcessPoolDistributedExecutor, DistributedCheckpointManager, Task
    ... )
    >>> import tempfile
    >>> def process(x):
    ...     return x * 2
    >>> checkpoint_mgr = DistributedCheckpointManager(tempfile.mkdtemp())
    >>> executor = ProcessPoolDistributedExecutor(
    ...     process,
    ...     num_workers=2,
    ...     checkpoint_manager=checkpoint_mgr
    ... )
    >>> tasks = [Task(id=f"t{i}", payload=i) for i in range(10)]
    >>> executor.submit_batch(tasks)
    >>> results = executor.run(
    ...     checkpoint_id="my_experiment",
    ...     checkpoint_interval=5
    ... )
    >>> len(results)
    10

    Single task submission:

    >>> from insideLLMs.distributed import ProcessPoolDistributedExecutor, Task
    >>> executor = ProcessPoolDistributedExecutor(lambda x: x.upper(), num_workers=1)
    >>> executor.submit(Task(id="t1", payload="hello"))
    >>> executor.submit(Task(id="t2", payload="world"))
    >>> results = executor.run()
    >>> sorted([r.result for r in results])
    ['HELLO', 'WORLD']

    Notes
    -----
    - The function must be defined at module level (not lambda or closure)
    - All task payloads must be pickleable
    - Process creation overhead makes this less efficient for many small tasks
    - Use LocalDistributedExecutor for I/O-bound tasks (API calls, etc.)

    See Also
    --------
    LocalDistributedExecutor : Thread-based execution for I/O-bound tasks
    DistributedCheckpointManager : Manages checkpoint storage
    DistributedExperimentRunner : High-level interface using this executor
    """

    def __init__(
        self,
        func: Callable[[Any], Any],
        num_workers: int = 4,
        checkpoint_manager: Optional[DistributedCheckpointManager] = None,
    ):
        """
        Initialize the process pool executor.

        Parameters
        ----------
        func : Callable[[Any], Any]
            Function to execute for each task. Must be pickleable.
        num_workers : int, optional
            Number of worker processes. Default is 4.
        checkpoint_manager : DistributedCheckpointManager, optional
            Checkpoint manager for fault tolerance.

        Examples
        --------
        >>> from insideLLMs.distributed import ProcessPoolDistributedExecutor
        >>> def my_func(x):
        ...     return x * 2
        >>> executor = ProcessPoolDistributedExecutor(my_func, num_workers=8)
        >>> print(executor.num_workers)
        8
        """
        self.func = func
        self.num_workers = num_workers
        self.checkpoint_manager = checkpoint_manager
        self._pending_tasks: list[Task] = []
        self._results: list[TaskResult] = []

    def submit(self, task: Task) -> None:
        """
        Submit a single task for later execution.

        Tasks are queued and executed when run() is called.

        Parameters
        ----------
        task : Task
            The task to submit.

        Examples
        --------
        >>> from insideLLMs.distributed import ProcessPoolDistributedExecutor, Task
        >>> executor = ProcessPoolDistributedExecutor(lambda x: x, num_workers=1)
        >>> executor.submit(Task(id="t1", payload="data"))
        >>> len(executor._pending_tasks)
        1
        """
        self._pending_tasks.append(task)

    def submit_batch(self, tasks: list[Task]) -> None:
        """
        Submit multiple tasks for later execution.

        Parameters
        ----------
        tasks : list[Task]
            List of tasks to submit.

        Examples
        --------
        >>> from insideLLMs.distributed import ProcessPoolDistributedExecutor, Task
        >>> executor = ProcessPoolDistributedExecutor(lambda x: x, num_workers=2)
        >>> tasks = [Task(id=f"t{i}", payload=i) for i in range(5)]
        >>> executor.submit_batch(tasks)
        >>> len(executor._pending_tasks)
        5
        """
        self._pending_tasks.extend(tasks)

    def run(
        self,
        checkpoint_id: Optional[str] = None,
        checkpoint_interval: int = 100,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[TaskResult]:
        """
        Execute all submitted tasks using the process pool.

        This method blocks until all tasks are complete. If a checkpoint_id
        is provided and a checkpoint exists, execution resumes from the
        checkpoint. Checkpoints are saved periodically and deleted on
        successful completion.

        Parameters
        ----------
        checkpoint_id : str, optional
            Identifier for checkpointing. If provided with a checkpoint_manager,
            enables fault tolerance. On resume, loads state from checkpoint.
        checkpoint_interval : int, optional
            Save checkpoint every N completed tasks. Default is 100.
        progress_callback : Callable[[int, int], None], optional
            Function called with (completed_count, total_count) after each
            task completes.

        Returns
        -------
        list[TaskResult]
            List of results for all tasks, in completion order.

        Examples
        --------
        Basic execution:

        >>> from insideLLMs.distributed import ProcessPoolDistributedExecutor, Task
        >>> def square(x):
        ...     return x ** 2
        >>> executor = ProcessPoolDistributedExecutor(square, num_workers=2)
        >>> executor.submit_batch([Task(id=f"t{i}", payload=i) for i in range(5)])
        >>> results = executor.run()
        >>> sorted([r.result for r in results])
        [0, 1, 4, 9, 16]

        With progress tracking:

        >>> from insideLLMs.distributed import ProcessPoolDistributedExecutor, Task
        >>> progress_log = []
        >>> def track_progress(done, total):
        ...     progress_log.append(f"{done}/{total}")
        >>> executor = ProcessPoolDistributedExecutor(lambda x: x, num_workers=2)
        >>> executor.submit_batch([Task(id=f"t{i}", payload=i) for i in range(3)])
        >>> _ = executor.run(progress_callback=track_progress)
        >>> "3/3" in progress_log
        True

        With checkpointing:

        >>> from insideLLMs.distributed import (
        ...     ProcessPoolDistributedExecutor, DistributedCheckpointManager, Task
        ... )
        >>> import tempfile
        >>> mgr = DistributedCheckpointManager(tempfile.mkdtemp())
        >>> executor = ProcessPoolDistributedExecutor(
        ...     lambda x: x * 2, num_workers=2, checkpoint_manager=mgr
        ... )
        >>> executor.submit_batch([Task(id=f"t{i}", payload=i) for i in range(10)])
        >>> results = executor.run(checkpoint_id="test", checkpoint_interval=3)
        >>> len(results)
        10
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
                        t
                        for t in self._pending_tasks
                        if t.id not in {r.task_id for r in self._results}
                    ]
                    self.checkpoint_manager.save(checkpoint_id, remaining, self._results)

        # Clear pending tasks
        self._pending_tasks.clear()

        # Delete checkpoint if completed successfully
        if checkpoint_id and self.checkpoint_manager:
            self.checkpoint_manager.delete(checkpoint_id)

        return self._results

    def _execute_task(self, task: Task) -> TaskResult:
        """
        Execute a single task (internal).

        This method is called in a worker process.

        Parameters
        ----------
        task : Task
            The task to execute.

        Returns
        -------
        TaskResult
            The execution result.
        """
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
        reducer: Callable[[list[R]], Any],
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

    def execute(self, items: list[T]) -> Any:
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
        items: list[T],
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
        self.checkpoint_manager = DistributedCheckpointManager(checkpoint_dir) if checkpoint_dir else None

    def run_prompts(
        self,
        prompts: list[str],
        checkpoint_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[dict[str, Any]]:
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
                output.append(
                    {
                        "prompt": prompt,
                        "response": result.result if result.success else None,
                        "success": result.success,
                        "error": result.error,
                        "latency_ms": result.latency_ms,
                    }
                )
            else:
                output.append(
                    {
                        "prompt": prompt,
                        "response": None,
                        "success": False,
                        "error": "Task not found",
                        "latency_ms": 0,
                    }
                )

        return output

    def _execute_prompt(self, prompt: str) -> str:
        """Execute a single prompt."""
        return self.model_func(prompt)

    def run_experiments(
        self,
        experiments: list[dict[str, Any]],
        prompt_key: str = "prompt",
        checkpoint_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
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
    items: list[T],
    num_workers: int = 4,
    use_processes: bool = False,
) -> list[R]:
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
    func: Callable[[list[T]], list[R]],
    items: list[T],
    batch_size: int = 10,
    num_workers: int = 4,
) -> list[R]:
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
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_results = list(executor.map(func, batches))

    # Flatten results
    return [item for batch in batch_results for item in batch]
