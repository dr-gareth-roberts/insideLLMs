import asyncio

from insideLLMs.inference import (
    InferenceClient,
    InferenceRequest,
    StopReason,
    Verification,
    VerifierSpec,
)
from insideLLMs.types import ModelResponse, TokenUsage


class MetadataModel:
    name = "metadata-model"
    model_id = "provider/model-v1"

    async def agenerate_with_metadata(self, prompt: str, **kwargs: object) -> ModelResponse:
        return ModelResponse(
            content=f"answer:{prompt}",
            model=self.model_id,
            latency_ms=20.0,
            usage=TokenUsage(prompt_tokens=4, completion_tokens=3, total_tokens=7),
        )


async def test_inference_client_one_shot_returns_common_auditable_result() -> None:
    client = InferenceClient(MetadataModel(), generation_kwargs={"temperature": 0})

    result = await client.generate(InferenceRequest(prompt="question"))

    assert result.answer == "answer:question"
    assert result.confidence == 1.0
    assert result.stop_reason is StopReason.COMPLETED
    assert result.spend.calls == 1
    assert result.spend.input_tokens == 4
    assert result.spend.output_tokens == 3
    assert result.spend.elapsed_seconds == 0.02
    assert result.provenance == {
        "strategy": "one-shot",
        "model": "provider/model-v1",
    }
    assert result.trace[0].kind == "model-generation"
    assert result.trace[0].calls == 1


async def test_inference_client_accepts_a_prompt_string() -> None:
    result = await InferenceClient(MetadataModel()).generate("hello")

    assert result.answer == "answer:hello"


async def test_inference_client_generates_independent_matched_baseline_samples() -> None:
    results = await InferenceClient(MetadataModel()).generate_many("hello", n=3)

    assert [result.answer for result in results] == ["answer:hello"] * 3
    assert [result.spend.calls for result in results] == [1, 1, 1]
    assert [result.spend.input_tokens for result in results] == [4, 4, 4]
    assert [result.candidates[0].metadata["sample_index"] for result in results] == [0, 1, 2]
    assert len({result.candidates[0].id for result in results}) == 3


class SequenceModel:
    name = "sequence-model"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls = 0

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        self.calls += 1
        return next(self.outputs)


async def test_inference_client_self_consistency_uses_safe_early_stop() -> None:
    model = SequenceModel(["  FOUR ", "four", "wrong"])
    client = InferenceClient(model)

    result = await client.self_consistency("2 + 2", max_samples=3)

    assert result.answer == "four"
    assert result.confidence == 1.0
    assert result.stop_reason is StopReason.AGREEMENT
    assert result.spend.calls == 2
    assert len({candidate.id for candidate in result.candidates}) == 2
    assert result.provenance["strategy"] == "self-consistency"
    assert result.provenance["model"] == "sequence-model"
    assert model.calls == 2


async def test_inference_client_best_of_n_uses_model_adapter_and_hard_verifier() -> None:
    model = SequenceModel(["bad", "good"])
    client = InferenceClient(model)

    result = await client.best_of_n(
        "choose",
        n=2,
        verifiers=(
            VerifierSpec(
                "must-be-good",
                lambda candidate: Verification(
                    "must-be-good",
                    score=1.0 if candidate.output == "good" else 0.0,
                    passed=candidate.output == "good",
                ),
                hard=True,
            ),
        ),
    )

    assert result.answer == "good"
    assert result.stop_reason is StopReason.VERIFIED
    assert result.spend.calls == 2
    assert len(result.candidates) == 2
    assert result.provenance["strategy"] == "best-of-n"
    assert result.provenance["model"] == "sequence-model"
    assert model.calls == 2


async def test_best_of_n_elapsed_time_includes_verifier_work() -> None:
    client = InferenceClient(SequenceModel(["answer"]))

    async def delayed_verifier(candidate: object) -> Verification:
        await asyncio.sleep(0.01)
        return Verification("delayed", score=1.0, passed=True)

    result = await client.best_of_n(
        "choose",
        n=1,
        verifiers=(VerifierSpec("delayed", delayed_verifier),),
    )

    assert result.spend.elapsed_seconds >= 0.01


async def test_inference_client_uses_existing_model_configuration_path() -> None:
    client = InferenceClient.from_model_config(
        {
            "type": "dummy",
            "args": {"canned_response": "configured answer"},
            "pipeline": {
                "middlewares": [{"type": "passthrough"}],
            },
        }
    )

    result = await client.generate("configured question")

    assert result.answer == "configured answer"
    assert result.provenance["model"] == "dummy-v1"


class MetadataSequenceModel:
    name = "metadata-sequence"
    model_id = "provider/sequence-v1"

    def __init__(self) -> None:
        self.index = 0

    async def agenerate_with_metadata(self, prompt: str, **kwargs: object) -> ModelResponse:
        index = self.index
        self.index += 1
        return ModelResponse(
            content=("bad", "good")[index],
            model=self.model_id,
            latency_ms=(10.0, 20.0)[index],
            usage=TokenUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )


async def test_best_of_n_aggregates_real_model_token_and_latency_metadata() -> None:
    client = InferenceClient(MetadataSequenceModel())

    result = await client.best_of_n(
        "choose",
        n=2,
        verifiers=(
            VerifierSpec(
                "score",
                lambda candidate: Verification(
                    "score",
                    score=float(candidate.output == "good"),
                    passed=True,
                ),
            ),
        ),
    )

    assert result.spend.calls == 2
    assert result.spend.input_tokens == 4
    assert result.spend.output_tokens == 2
    assert result.spend.elapsed_seconds == 0.02
    assert result.trace[0].calls == 2
    assert result.trace[0].input_tokens == 4
    assert result.trace[0].output_tokens == 2
    assert result.trace[0].latency_seconds == 0.02
