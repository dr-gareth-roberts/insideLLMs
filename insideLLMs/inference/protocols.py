"""Composable protocols used by inference strategies."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from .schemas import Candidate, Decision, InferenceRequest, Verification

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class Proposer(Protocol):
    async def sample(self, request: InferenceRequest, n: int) -> list[Candidate]: ...


class Normalizer(Protocol):
    def key(self, candidate: Candidate) -> str | None: ...


class Verifier(Protocol):
    async def verify(self, request: InferenceRequest, candidate: Candidate) -> Verification: ...


class Reranker(Protocol):
    async def score(self, query: str, document: Any) -> float: ...


class ToolEnvironment(Protocol):
    async def execute(self, action: Any, limits: Any) -> Any: ...


class BudgetPolicy(Protocol):
    def decide(self, trace: tuple[Any, ...], uncertainty: float, budget: Any) -> Decision: ...


class SearchProblem(Protocol[StateT]):
    async def propose(self, state: StateT) -> list[StateT]: ...

    async def value(self, state: StateT) -> float: ...

    def key(self, state: StateT) -> str: ...
