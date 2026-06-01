"""Regression tests for production-quality audit wave 6 (remaining) fixes."""


# L24 — logic metrics use a consistent denominator (a SUCCESS result with empty
# output must not skew reasoning_rate / avg_response_length).
def test_logic_metrics_consistent_denominator_with_empty_output():
    from insideLLMs.probes.logic import LogicProbe
    from insideLLMs.types import ProbeResult, ResultStatus

    results = [
        ProbeResult(
            input="q1",
            output="Because 2 plus 2 equals 4, therefore the answer is 4",
            status=ResultStatus.SUCCESS,
        ),
        ProbeResult(input="q2", output=None, status=ResultStatus.SUCCESS),  # SUCCESS, no output
    ]
    score = LogicProbe().score(results)
    # The output=None result is excluded from BOTH numerator and denominator, so
    # the rate is 1/1 = 1.0, not the old skewed 1/2 (None inflated the denominator).
    assert score.custom_metrics["reasoning_rate"] == 1.0


# M20 — PromptVariator must not seed the process-global random module.
def test_prompt_variator_does_not_mutate_global_random():
    import random

    from insideLLMs.contrib.synthesis import PromptVariator, SynthesisConfig

    random.seed(999)
    before = random.random()
    random.seed(999)
    PromptVariator(model=None, config=SynthesisConfig(seed=12345))
    after = random.random()
    assert before == after


# L13/L14 — public __all__ no longer leaks imported stdlib/typing symbols.
def test_public_all_excludes_leaked_imports():
    import insideLLMs.dataset_utils as du
    import insideLLMs.registry as reg
    import insideLLMs.types as ty

    for mod, leaked in [
        (reg, {"os", "warnings", "Any", "Optional", "TypeVar"}),
        (ty, {"Any", "Optional", "dataclass", "datetime", "Enum"}),
        (du, {"csv", "json", "importlib", "Any", "load_dataset"}),
    ]:
        assert leaked.isdisjoint(set(mod.__all__)), (mod.__name__, leaked & set(mod.__all__))
    # Genuine public symbols remain exported.
    assert "ConfigDict" in ty.__all__
    assert "model_registry" in reg.__all__


# L30 — a bare ExportMetadata() no longer embeds a wall-clock timestamp.
def test_export_metadata_default_has_no_walltime():
    from insideLLMs.analysis.export import ExportMetadata

    assert ExportMetadata().export_time is None
