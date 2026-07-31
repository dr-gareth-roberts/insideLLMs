"""Offline population search over reusable textual harness artifacts."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Awaitable

from ._callbacks import invoke_with_timeout, is_async_callable


class EvolutionBudgetExceeded(RuntimeError):
    """Raised when no initial candidate completes inside the time budget."""


@dataclass(frozen=True)
class Fitness:
    """Held-out optimization fitness with an optional separate validation score."""

    fitness: float
    validation_fitness: float | None = None


@dataclass(frozen=True)
class EvolutionCandidate:
    """A textual artifact plus complete evolutionary provenance."""

    id: str
    artifact_text: str
    fitness: float
    validation_fitness: float | None = None
    parent_ids: tuple[str, ...] = ()
    lineage: tuple[str, ...] = ()
    generation: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    candidate_ids: tuple[str, ...]
    best_id: str
    best_fitness: float


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int
    elite_count: int = 1
    max_generations: int = 10
    max_evaluations: int = 100
    max_seconds: float | None = None
    max_validation_evaluations: int = 1
    seed: int = 0


@dataclass(frozen=True)
class EvolutionResult:
    best: EvolutionCandidate
    population: tuple[EvolutionCandidate, ...]
    history: tuple[GenerationRecord, ...]
    evaluations: int
    validation_evaluations: int
    generations: int
    stop_reason: str
    seed: int
    provenance: dict[str, object]


ParentSelector = Callable[
    [tuple[EvolutionCandidate, ...], random.Random],
    EvolutionCandidate | Awaitable[EvolutionCandidate],
]
Mutator = Callable[
    [EvolutionCandidate, random.Random],
    str | EvolutionCandidate | Awaitable[str | EvolutionCandidate],
]
Evaluator = Callable[[str], float | Fitness | Awaitable[float | Fitness]]


def _candidate_id(text: str, generation: int, parent_ids: tuple[str, ...]) -> str:
    payload = f"{generation}\0{','.join(parent_ids)}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _rank(population: Sequence[EvolutionCandidate]) -> list[EvolutionCandidate]:
    return sorted(population, key=lambda item: (-item.fitness, item.artifact_text, item.id))


def _default_select(
    population: tuple[EvolutionCandidate, ...], rng: random.Random
) -> EvolutionCandidate:
    """Seeded tournament selection avoids assumptions about fitness scale."""

    contenders = rng.sample(population, k=min(3, len(population)))
    return _rank(contenders)[0]


async def evolve_artifacts(
    initial_artifacts: Sequence[str],
    *,
    evaluate: Evaluator,
    mutate: Mutator,
    config: EvolutionConfig,
    select_parent: ParentSelector | None = None,
    validation_evaluate: Evaluator | None = None,
    final_validation: (Callable[[EvolutionCandidate], float | Awaitable[float]] | None) = None,
) -> EvolutionResult:
    """Optimize artifacts offline; validation scores never influence selection."""

    if config.population_size < 1:
        raise ValueError("population_size must be positive")
    if not 0 <= config.elite_count <= config.population_size:
        raise ValueError("elite_count must be between zero and population_size")
    if (
        config.max_generations < 0
        or config.max_evaluations < 1
        or (config.max_seconds is not None and config.max_seconds <= 0)
    ):
        raise ValueError("evolution budgets must be positive")
    if not initial_artifacts:
        raise ValueError("initial_artifacts must not be empty")
    if config.max_validation_evaluations < 0:
        raise ValueError("max_validation_evaluations must be non-negative")
    if validation_evaluate is not None and final_validation is not None:
        raise ValueError("use validation_evaluate or final_validation, not both")

    started = time.monotonic()
    rng = random.Random(config.seed)
    selector = select_parent or _default_select
    timed_callbacks = (evaluate, mutate, selector, validation_evaluate, final_validation)
    if config.max_seconds is not None and any(
        callback is not None and not is_async_callable(callback) for callback in timed_callbacks
    ):
        raise ValueError("evolution time budgets require asynchronous callbacks")
    seen_texts: set[str] = set()
    duplicates_removed = 0
    evaluations = 0
    validation_evaluations = 0
    population: list[EvolutionCandidate] = []

    def remaining_seconds() -> float | None:
        if config.max_seconds is None:
            return None
        return config.max_seconds - (time.monotonic() - started)

    async def call(callback: Callable, *args: object):
        remaining = remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise EvolutionBudgetExceeded("evolution time budget exhausted")
        try:
            return await invoke_with_timeout(callback, *args, timeout=remaining)
        except TimeoutError as error:
            if remaining is None:
                # No time budget was armed: this is the callback's own error.
                raise
            raise EvolutionBudgetExceeded("evolution time budget exhausted") from error

    async def score(text: str) -> Fitness:
        nonlocal evaluations
        measured = await call(evaluate, text)
        evaluations += 1
        fitness = measured if isinstance(measured, Fitness) else Fitness(float(measured))
        return fitness

    for text in initial_artifacts:
        if text in seen_texts:
            duplicates_removed += 1
            continue
        if evaluations >= config.max_evaluations or len(population) >= config.population_size:
            break
        seen_texts.add(text)
        try:
            measured = await score(text)
        except EvolutionBudgetExceeded:
            if not population:
                raise
            break
        population.append(
            EvolutionCandidate(
                id=_candidate_id(text, 0, ()),
                artifact_text=text,
                fitness=measured.fitness,
                validation_fitness=measured.validation_fitness,
            )
        )
    if not population:
        raise ValueError("evaluation budget did not admit an initial candidate")

    population = _rank(population)
    best_seen = population[0]
    history = [
        GenerationRecord(
            generation=0,
            candidate_ids=tuple(item.id for item in population),
            best_id=population[0].id,
            best_fitness=population[0].fitness,
        )
    ]
    generation = 0
    stop_reason = "generation_budget"

    while generation < config.max_generations:
        if evaluations >= config.max_evaluations:
            stop_reason = "evaluation_budget"
            break
        if config.max_seconds is not None and time.monotonic() - started >= config.max_seconds:
            stop_reason = "time_budget"
            break
        generation += 1
        next_population = _rank(population)[: config.elite_count]
        attempts = 0
        max_attempts = max(20, config.population_size * 20)
        budget_hit: str | None = None

        while len(next_population) < config.population_size and attempts < max_attempts:
            if evaluations >= config.max_evaluations:
                budget_hit = "evaluation_budget"
                break
            if config.max_seconds is not None and time.monotonic() - started >= config.max_seconds:
                budget_hit = "time_budget"
                break
            attempts += 1
            parent_pool = tuple(_rank((*population, *next_population)))
            canonical_by_id = {candidate.id: candidate for candidate in parent_pool}
            selection_view = tuple(
                replace(candidate, validation_fitness=None) for candidate in parent_pool
            )
            try:
                selected_view = await call(selector, selection_view, rng)
            except EvolutionBudgetExceeded:
                budget_hit = "time_budget"
                break
            if selected_view.id not in canonical_by_id:
                raise ValueError("select_parent returned a candidate outside the population")
            parent = canonical_by_id[selected_view.id]
            mutation_parent = replace(parent, validation_fitness=None)
            try:
                mutation = await call(mutate, mutation_parent, rng)
            except EvolutionBudgetExceeded:
                budget_hit = "time_budget"
                break
            text = mutation.artifact_text if isinstance(mutation, EvolutionCandidate) else mutation
            if text in seen_texts:
                duplicates_removed += 1
                continue
            seen_texts.add(text)
            try:
                measured = await score(text)
            except EvolutionBudgetExceeded:
                budget_hit = "time_budget"
                break
            parent_ids = (parent.id,)
            metadata = dict(mutation.metadata) if isinstance(mutation, EvolutionCandidate) else {}
            next_population.append(
                EvolutionCandidate(
                    id=_candidate_id(text, generation, parent_ids),
                    artifact_text=text,
                    fitness=measured.fitness,
                    validation_fitness=measured.validation_fitness,
                    parent_ids=parent_ids,
                    lineage=(*parent.lineage, parent.id),
                    generation=generation,
                    metadata=metadata,
                )
            )

        if next_population:
            population = _rank(next_population)
            best_seen = _rank((best_seen, population[0]))[0]
            history.append(
                GenerationRecord(
                    generation=generation,
                    candidate_ids=tuple(item.id for item in population),
                    best_id=population[0].id,
                    best_fitness=population[0].fitness,
                )
            )
        if budget_hit is not None:
            stop_reason = budget_hit
            break
        if len(next_population) < config.population_size:
            stop_reason = "deduplication_exhausted"
            break
    else:
        stop_reason = "generation_budget"

    best = best_seen
    validator = final_validation or validation_evaluate
    if validator is not None and validation_evaluations < config.max_validation_evaluations:
        validation_input = best if final_validation is not None else best.artifact_text
        try:
            validation = await call(validator, validation_input)
        except EvolutionBudgetExceeded:
            stop_reason = "time_budget"
        else:
            validation_evaluations += 1
            if isinstance(validation, Fitness):
                validation_score = (
                    validation.validation_fitness
                    if validation.validation_fitness is not None
                    else validation.fitness
                )
            else:
                validation_score = float(validation)
            best = replace(
                best,
                validation_fitness=validation_score,
                metadata={**best.metadata, "final_validation": True},
            )
            population = [best if item.id == best.id else item for item in population]

    return EvolutionResult(
        best=best,
        population=tuple(population),
        history=tuple(history),
        evaluations=evaluations,
        validation_evaluations=validation_evaluations,
        generations=generation,
        stop_reason=stop_reason,
        seed=config.seed,
        provenance={
            "duplicates_removed": duplicates_removed,
            "selection_fitness": "fitness",
            "validation_used_for_selection": False,
            "initial_artifact_count": len(initial_artifacts),
            "time_budget_enforced": True,
        },
    )
