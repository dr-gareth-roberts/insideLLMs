import pytest

from insideLLMs.inference.evolution import EvolutionConfig, Fitness, evolve_artifacts


async def test_evolution_is_seeded_budgeted_deduplicated_and_validation_aware() -> None:
    def evaluate(artifact: str) -> float:
        return float(artifact.count("!"))

    def select_best(population, rng):
        return min(
            population,
            key=lambda candidate: (-float(candidate.fitness), candidate.artifact_text),
        )

    def mutate(parent, rng) -> str:
        return parent.artifact_text + "!"

    config = EvolutionConfig(
        population_size=3,
        elite_count=1,
        max_generations=10,
        max_evaluations=6,
        seed=13,
    )
    result = await evolve_artifacts(
        ["a", "a", "b"],
        evaluate=evaluate,
        mutate=mutate,
        select_parent=select_best,
        final_validation=lambda candidate: float(len(candidate.artifact_text)),
        config=config,
    )

    assert result.best.artifact_text == "a!!!!"
    assert result.best.fitness == 4.0
    assert result.best.validation_fitness == 5.0
    assert result.best.generation == 2
    assert len(result.best.lineage) == 4
    assert result.evaluations == 6
    assert result.stop_reason == "evaluation_budget"
    assert result.provenance["duplicates_removed"] >= 1
    assert [record.generation for record in result.history] == [0, 1, 2]

    repeated = await evolve_artifacts(
        ["a", "a", "b"],
        evaluate=evaluate,
        mutate=mutate,
        select_parent=select_best,
        final_validation=lambda candidate: float(len(candidate.artifact_text)),
        config=config,
    )
    assert repeated == result


async def test_validation_fitness_is_hidden_from_optimization_callbacks() -> None:
    observed_validation: list[float | None] = []

    def evaluate(artifact: str) -> Fitness:
        return Fitness(
            fitness=1.0 if artifact.startswith("train-best") else 0.0,
            validation_fitness=100.0 if artifact.startswith("validation-best") else 0.0,
        )

    def select(population, rng):
        observed_validation.extend(candidate.validation_fitness for candidate in population)
        return population[0]

    def mutate(parent, rng) -> str:
        observed_validation.append(parent.validation_fitness)
        return parent.artifact_text + "!"

    result = await evolve_artifacts(
        ["train-best", "validation-best"],
        evaluate=evaluate,
        mutate=mutate,
        select_parent=select,
        config=EvolutionConfig(
            population_size=2,
            elite_count=1,
            max_generations=1,
            max_evaluations=3,
            seed=1,
        ),
    )

    assert observed_validation and set(observed_validation) == {None}
    assert result.best.artifact_text.startswith("train-best")
    assert result.provenance["validation_used_for_selection"] is False


async def test_evolution_preserves_global_best_when_elitism_is_disabled() -> None:
    scores = {"best": 10.0, "other": 0.0, "worse": -1.0}

    result = await evolve_artifacts(
        ["best", "other"],
        evaluate=lambda artifact: scores[artifact],
        mutate=lambda parent, rng: "worse",
        select_parent=lambda population, rng: population[0],
        config=EvolutionConfig(
            population_size=2,
            elite_count=0,
            max_generations=1,
            max_evaluations=3,
            seed=1,
        ),
    )

    assert result.best.artifact_text == "best"
    assert result.best.fitness == 10.0


async def test_evolution_global_best_uses_deterministic_tie_break() -> None:
    result = await evolve_artifacts(
        ["z"],
        evaluate=lambda artifact: 1.0,
        mutate=lambda parent, rng: "a",
        select_parent=lambda population, rng: population[0],
        config=EvolutionConfig(
            population_size=1,
            elite_count=0,
            max_generations=1,
            max_evaluations=2,
            seed=1,
        ),
    )

    assert result.best.artifact_text == "a"


async def test_timed_evolution_rejects_sync_callbacks_before_evaluation() -> None:
    evaluated = False

    def evaluate(artifact: str) -> float:
        nonlocal evaluated
        evaluated = True
        return 1.0

    with pytest.raises(ValueError, match="asynchronous"):
        await evolve_artifacts(
            ["seed"],
            evaluate=evaluate,
            mutate=lambda parent, rng: parent.artifact_text + "!",
            config=EvolutionConfig(
                population_size=1,
                max_generations=1,
                max_evaluations=2,
                max_seconds=0.1,
            ),
        )

    assert evaluated is False


async def test_timed_evolution_runs_with_async_callbacks_and_default_selector() -> None:
    """The default parent selector must not trip the async-callback guard."""

    async def evaluate(artifact: str) -> float:
        return float(artifact.count("!"))

    async def mutate(parent: object, rng: object) -> str:
        return parent.artifact_text + "!"

    result = await evolve_artifacts(
        ["seed"],
        evaluate=evaluate,
        mutate=mutate,
        config=EvolutionConfig(
            population_size=2,
            max_generations=2,
            max_evaluations=4,
            max_seconds=30.0,
            seed=3,
        ),
    )

    assert result.best.artifact_text.startswith("seed")
    assert result.evaluations >= 1
