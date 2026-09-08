"""Budget admission tests use fake transports; no provider requests are made."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from insideLLMs.runtime.budget import BudgetExceededError, BudgetLedger, BudgetPolicy


def policy(allowance: str = "0.30") -> BudgetPolicy:
    return BudgetPolicy(currency="USD", allowance=allowance, prices=[])


def test_pending_and_uncertain_calls_consume_the_shared_allowance():
    ledger = BudgetLedger(policy())
    first = ledger.reserve(Decimal("0.20"))
    ledger.settle(first, actual_cost=None)
    second = ledger.reserve(Decimal("0.10"))

    with pytest.raises(BudgetExceededError):
        ledger.reserve(Decimal("0.01"))

    assert ledger.snapshot()["uncertain"] == "0.20"
    assert ledger.snapshot()["reserved"] == "0.10"
    ledger.settle(second, actual_cost=Decimal("0.03"))
    assert ledger.snapshot()["settled"] == "0.03"


def paid_policy(allowance: str = "0.12", provider: str = "openai") -> BudgetPolicy:
    return BudgetPolicy(
        currency="USD",
        allowance=allowance,
        prices=[
            {
                "provider": provider,
                "model": "budget-test",
                "endpoint": (
                    "https://api.openai.com/v1"
                    if provider == "openai"
                    else "https://api.anthropic.com"
                ),
                "pricing_id": "test-fixture-not-real-pricing",
                "input_cost_per_million": "100",
                "output_cost_per_million": "200",
                "maximum_input_tokens": 1000,
                "maximum_output_tokens": 100,
                "output_token_parameter": "max_tokens",
            }
        ],
    )


@pytest.fixture
def fake_openai(monkeypatch):
    pytest.importorskip("openai")
    import insideLLMs.models.openai as adapter

    class FakeClient:
        instances = []
        error = None
        hook = None
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)

        def __init__(self, **kwargs):
            self.max_retries = kwargs.get("max_retries", 2)
            self.base_url = kwargs.get("base_url") or "https://api.openai.com/v1/"
            self.requests = []
            self.chat = SimpleNamespace(completions=self)
            self.instances.append(self)

        def create(self, **kwargs):
            self.requests.append(kwargs)
            if type(self).hook is not None:
                type(self).hook()
            if type(self).error is not None:
                raise type(self).error
            return SimpleNamespace(
                model="budget-test",
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=type(self).usage,
            )

    monkeypatch.setattr(adapter, "OpenAI", FakeClient)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    return FakeClient


def test_provider_dispatch_obeys_allowance_and_enforces_the_configured_output_cap(
    fake_openai,
):
    from insideLLMs.runtime._config_loader import _create_model_from_config

    ledger = BudgetLedger(paid_policy("0.119"))
    model = _create_model_from_config(
        {"type": "openai", "args": {"model_name": "budget-test"}},
        budget_ledger=ledger,
    )
    with pytest.raises(BudgetExceededError):
        model.generate("hello")
    assert fake_openai.instances[-1].requests == []

    funded = BudgetLedger(paid_policy())
    model = _create_model_from_config(
        {"type": "openai", "args": {"model_name": "budget-test"}},
        budget_ledger=funded,
    )
    assert model.generate("hello") == "ok"
    assert fake_openai.instances[-1].max_retries == 0
    assert fake_openai.instances[-1].requests[0]["max_tokens"] == 100
    assert Decimal(funded.snapshot()["settled"]) == Decimal("0.002")
    assert Decimal(funded.snapshot()["reserved"]) == 0


def test_subject_and_judge_share_one_ledger_and_unknown_judges_fail_preflight(
    fake_openai,
):
    from insideLLMs.runtime._config_loader import (
        _create_model_from_config,
        _create_probe_from_config,
    )
    from insideLLMs.runtime.budget import (
        BudgetUnsupportedError,
        preflight_budget_config,
    )

    config = {
        "model": {"type": "openai", "args": {"model_name": "budget-test"}},
        "probe": {
            "type": "judge",
            "args": {"judge_model": {"type": "openai", "args": {"model_name": "budget-test"}}},
        },
        "budget": paid_policy("0.12").model_dump(mode="json"),
        "dataset": {"format": "inline", "data": []},
    }
    preflight_budget_config(config)
    ledger = BudgetLedger(paid_policy())
    subject = _create_model_from_config(config["model"], budget_ledger=ledger)
    judge = _create_probe_from_config(config["probe"], budget_ledger=ledger)
    assert subject.generate("subject") == "ok"
    with pytest.raises(BudgetExceededError):
        judge.evaluate_single("ok", "reference", "input")
    assert sum(len(client.requests) for client in fake_openai.instances) == 1

    config["probe"]["args"]["judge_model"] = {"type": "unknown"}
    with pytest.raises(BudgetUnsupportedError):
        preflight_budget_config(config)


def test_anthropic_text_chat_settles_usage_and_refuses_streaming(monkeypatch):
    pytest.importorskip("anthropic")
    import insideLLMs.models.anthropic as adapter
    from insideLLMs.runtime._config_loader import _create_model_from_config
    from insideLLMs.runtime.budget import BudgetUnsupportedError

    requests = []

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.max_retries = kwargs["max_retries"]
            self.base_url = "https://api.anthropic.com"
            self.messages = self

        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                model="budget-test",
                content=[SimpleNamespace(text="hello")],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            )

        def stream(self, **kwargs):
            requests.append(kwargs)
            raise AssertionError("Streaming must not dispatch")

    monkeypatch.setattr(adapter.anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test-key")
    ledger = BudgetLedger(paid_policy(provider="anthropic"))
    model = _create_model_from_config(
        {"type": "anthropic", "args": {"model_name": "budget-test"}},
        budget_ledger=ledger,
    )
    assert (
        model.chat(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
            ]
        )
        == "hello"
    )
    assert requests[0]["max_tokens"] == 100
    assert requests[0]["system"] == "be brief"
    assert Decimal(ledger.snapshot()["settled"]) == Decimal("0.002")
    with pytest.raises(BudgetUnsupportedError):
        list(model.stream("hello"))
    assert len(requests) == 1


def test_budget_denial_is_not_retried_by_model_wrapper():
    from insideLLMs.models.base import ModelWrapper

    calls = []

    class Rejected:
        def generate(self, prompt, **kwargs):
            calls.append(prompt)
            raise BudgetExceededError()

    with pytest.raises(BudgetExceededError):
        ModelWrapper(Rejected(), max_retries=3, retry_delay=0).generate("hello")
    assert calls == ["hello"]


@pytest.mark.parametrize(
    "operation,kwargs",
    [
        ("generate", {"n": 2}),
        ("generate", {"max_tokens": 101}),
        ("generate", {"max_tokens": None}),
        ("generate", {"tools": []}),
        ("generate", {"stream": True}),
        ("generate", {"extra_body": {}}),
        ("generate", {"max_completion_tokens": 50}),
        ("stream", {}),
    ],
)
def test_unsupported_request_shapes_do_not_dispatch(fake_openai, operation, kwargs):
    from insideLLMs.runtime._config_loader import _create_model_from_config
    from insideLLMs.runtime.budget import BudgetUnsupportedError

    model = _create_model_from_config(
        {"type": "openai", "args": {"model_name": "budget-test"}},
        budget_ledger=BudgetLedger(paid_policy()),
    )
    with pytest.raises(BudgetUnsupportedError):
        result = getattr(model, operation)("hello", **kwargs)
        if operation == "stream":
            list(result)
    assert fake_openai.instances[-1].requests == []


@pytest.mark.parametrize(
    "args",
    [
        {"model_name": "unknown"},
        {"model_name": "budget-test", "max_retries": 2},
        {"model_name": "budget-test", "base_url": "https://example.invalid/v1"},
    ],
)
def test_unknown_prices_endpoints_and_hidden_retries_deny_before_constructor(fake_openai, args):
    from insideLLMs.runtime._config_loader import _create_model_from_config
    from insideLLMs.runtime.budget import BudgetUnsupportedError

    with pytest.raises(BudgetUnsupportedError):
        _create_model_from_config(
            {"type": "openai", "args": args}, budget_ledger=BudgetLedger(paid_policy())
        )
    assert fake_openai.instances == []


def test_thread_contention_cannot_spend_the_last_reservation_twice():
    ledger = BudgetLedger(policy("0.10"))
    barrier = Barrier(12)

    def attempt():
        barrier.wait(timeout=5)
        try:
            return ledger.reserve(Decimal("0.10"))
        except BudgetExceededError:
            return None

    with ThreadPoolExecutor(max_workers=12) as executor:
        admitted = list(executor.map(lambda _: attempt(), range(12)))
    assert sum(item is not None for item in admitted) == 1
    assert ledger.snapshot()["reserved"] == "0.10"


def test_duplicate_settlement_and_underquote_do_not_hide_liability():
    from insideLLMs.runtime.budget import BudgetBreachError

    ledger = BudgetLedger(policy())
    first = ledger.reserve(Decimal("0.10"))
    ledger.settle(first, actual_cost=Decimal("0.04"))
    ledger.settle(first, actual_cost=Decimal("0.04"))
    assert ledger.snapshot()["settled"] == "0.04"
    second = ledger.reserve(Decimal("0.10"))
    with pytest.raises(BudgetBreachError):
        ledger.settle(second, actual_cost=Decimal("0.40"))
    assert ledger.snapshot()["settled"] == "0.44"
    with pytest.raises(BudgetExceededError):
        ledger.reserve(Decimal(0))


def test_retry_attempt_retains_failed_call_liability(fake_openai):
    from insideLLMs.runtime._config_loader import _create_model_from_config
    from insideLLMs.runtime.pipeline import ModelPipeline, RetryMiddleware

    ledger = BudgetLedger(paid_policy("0.239"))
    model = _create_model_from_config(
        {"type": "openai", "args": {"model_name": "budget-test"}}, budget_ledger=ledger
    )
    fake_openai.error = ConnectionError("fake connection lost after dispatch")
    pipeline = ModelPipeline(model, [RetryMiddleware(max_retries=2, initial_delay=0)])
    with pytest.raises(BudgetExceededError):
        pipeline.generate("hello")
    assert len(fake_openai.instances[-1].requests) == 1
    assert Decimal(ledger.snapshot()["uncertain"]) == Decimal("0.12")


@pytest.mark.asyncio
async def test_cancellation_does_not_refund_a_running_sync_provider_call(fake_openai):
    from insideLLMs.runtime._config_loader import _create_model_from_config
    from insideLLMs.runtime.pipeline import ModelPipeline

    started, finish = Event(), Event()

    def hold():
        started.set()
        assert finish.wait(timeout=5)

    fake_openai.hook = hold
    fake_openai.usage = None
    ledger = BudgetLedger(paid_policy())
    model = _create_model_from_config(
        {"type": "openai", "args": {"model_name": "budget-test"}}, budget_ledger=ledger
    )
    task = asyncio.create_task(ModelPipeline(model).agenerate("hello"))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert Decimal(ledger.snapshot()["reserved"]) == Decimal("0.12")
        with pytest.raises(BudgetExceededError):
            model.generate("another call")
        assert len(fake_openai.instances[-1].requests) == 1
    finally:
        finish.set()
        # Drain the worker so no test leaves a provider call running.
        await asyncio.get_running_loop().shutdown_default_executor()
    assert Decimal(ledger.snapshot()["uncertain"]) == Decimal("0.12")


def test_reported_input_beyond_declared_bound_aborts_even_with_cheaper_output(
    fake_openai,
):
    from insideLLMs.runtime._config_loader import _create_model_from_config
    from insideLLMs.runtime.budget import BudgetBreachError

    fake_openai.usage = SimpleNamespace(prompt_tokens=1100, completion_tokens=0)
    ledger = BudgetLedger(paid_policy())
    model = _create_model_from_config(
        {"type": "openai", "args": {"model_name": "budget-test"}}, budget_ledger=ledger
    )
    with pytest.raises(BudgetBreachError):
        model.generate("hello")
    assert ledger.snapshot()["aborted"] is True
    assert Decimal(ledger.snapshot()["settled"]) == Decimal("0.11")


def test_cache_hit_makes_no_second_reservation(fake_openai):
    from insideLLMs.runtime._config_loader import _create_model_from_config

    ledger = BudgetLedger(paid_policy())
    model = _create_model_from_config(
        {
            "type": "openai",
            "args": {"model_name": "budget-test"},
            "pipeline": {"middlewares": ["cache"]},
        },
        budget_ledger=ledger,
    )
    assert model.generate("hello") == model.generate("hello") == "ok"
    assert len(fake_openai.instances[-1].requests) == 1
    assert Decimal(ledger.snapshot()["settled"]) == Decimal("0.002")


@pytest.mark.asyncio
async def test_async_batch_contention_admits_only_one_provider_call(fake_openai):
    from insideLLMs.runtime._config_loader import _create_model_from_config
    from insideLLMs.runtime.pipeline import ModelPipeline

    fake_openai.usage = None
    ledger = BudgetLedger(paid_policy())
    model = _create_model_from_config(
        {"type": "openai", "args": {"model_name": "budget-test"}}, budget_ledger=ledger
    )
    outputs = await ModelPipeline(model).abatch_generate(["a", "b", "c"], return_exceptions=True)
    assert outputs.count("ok") == 1
    assert len(fake_openai.instances[-1].requests) == 1
    assert Decimal(ledger.snapshot()["uncertain"]) == Decimal("0.12")


def test_policies_reject_unknown_or_absent_bounds_and_nonfinite_money():
    from pydantic import ValidationError

    for amount in ("NaN", "Infinity", "-1"):
        with pytest.raises(ValidationError):
            policy(amount)
    raw = paid_policy().model_dump(mode="json")
    for field in ("maximum_input_tokens", "maximum_output_tokens", "pricing_id"):
        incomplete = {
            **raw,
            "prices": [{k: v for k, v in raw["prices"][0].items() if k != field}],
        }
        with pytest.raises(ValidationError):
            BudgetPolicy.model_validate(incomplete)


def test_budgeted_resume_and_callback_objects_fail_without_calls(fake_openai):
    from insideLLMs.runtime.budget import (
        BudgetUnsupportedError,
        create_budget_ledger,
        preflight_budget_config,
    )

    config = {
        "budget": paid_policy().model_dump(mode="json"),
        "model": {"type": "openai", "args": {"model_name": "budget-test"}},
        "probe": {
            "type": "constraint_compliance",
            "args": {"validator": lambda x: True},
        },
    }
    with pytest.raises(BudgetUnsupportedError):
        create_budget_ledger(config, resume=True)
    with pytest.raises(BudgetUnsupportedError):
        preflight_budget_config(config)
    assert fake_openai.instances == []


def test_decimal_allowance_does_not_round_away_a_small_pending_charge():
    ledger = BudgetLedger(policy("999999999999.000000000001"))
    ledger.reserve(Decimal("999999999999"))
    ledger.reserve(Decimal("0.000000000001"))
    with pytest.raises(BudgetExceededError):
        ledger.reserve(Decimal("0.000000000000000001"))


def test_preflight_rejects_unbounded_judge_parameters_before_subject_call(fake_openai):
    from insideLLMs.runtime.budget import (
        BudgetUnsupportedError,
        preflight_budget_config,
    )

    config = {
        "budget": paid_policy().model_dump(mode="json"),
        "model": {"type": "openai", "args": {"model_name": "budget-test"}},
        "probe": {
            "type": "judge",
            "args": {
                "judge_model": {
                    "type": "openai",
                    "args": {"model_name": "budget-test"},
                },
                "judge_kwargs": {"max_tokens": 10000},
            },
        },
    }
    with pytest.raises(BudgetUnsupportedError):
        preflight_budget_config(config)
    assert fake_openai.instances == []


def test_budget_configuration_roundtrips_as_json_strings():
    import json

    import yaml

    from insideLLMs.config_schema import normalize_runtime_config

    config = {
        "budget": paid_policy().model_dump(mode="json"),
        "model": {"type": "dummy"},
        "probe": {"type": "logic"},
        "dataset": {"format": "inline", "data": []},
    }
    normalized = normalize_runtime_config(config)
    assert yaml.safe_load(yaml.safe_dump(normalized))["budget"]["allowance"] == "0.12"
    assert json.loads(json.dumps(normalized))["budget"]["allowance"] == "0.12"


def test_runner_validation_rejects_unbound_models_and_custom_probes_without_calls(
    fake_openai,
):
    from insideLLMs.probes import LogicProbe
    from insideLLMs.runtime._config_loader import _create_model_from_config
    from insideLLMs.runtime.budget import BudgetUnsupportedError, validate_budget_runner

    ledger = BudgetLedger(paid_policy())
    calls = []

    class Unchecked:
        def generate(self, prompt):
            calls.append(prompt)

    with pytest.raises(BudgetUnsupportedError):
        validate_budget_runner(Unchecked(), LogicProbe(), ledger)

    model = _create_model_from_config(
        {"type": "openai", "args": {"model_name": "budget-test"}}, budget_ledger=ledger
    )
    validate_budget_runner(model, LogicProbe(), ledger)
    with pytest.raises(BudgetUnsupportedError):
        validate_budget_runner(model, LogicProbe(), BudgetLedger(paid_policy()))

    class CustomProbe(LogicProbe):
        def run(self, model, data, **kwargs):
            calls.append(data)

    with pytest.raises(BudgetUnsupportedError):
        validate_budget_runner(model, CustomProbe(), ledger)
    assert calls == []
    assert fake_openai.instances[-1].requests == []


@pytest.mark.parametrize("inputs,outputs,cost", [(1100, 0, "0.11"), (0, 101, "0.0202")])
def test_bound_violation_blocks_competing_dispatch_before_settlement_returns(
    fake_openai, inputs, outputs, cost
):
    from insideLLMs.runtime._config_loader import _create_model_from_config
    from insideLLMs.runtime.budget import BudgetBreachError

    settled, finish = Event(), Event()

    class PausingLedger(BudgetLedger):
        def settle(self, reservation, **kwargs):
            try:
                super().settle(reservation, **kwargs)
            finally:
                if reservation.identifier == 0:
                    settled.set()
                    assert finish.wait(timeout=5)

    ledger = PausingLedger(paid_policy("0.24"))
    fake_openai.usage = SimpleNamespace(prompt_tokens=inputs, completion_tokens=outputs)
    model = _create_model_from_config(
        {"type": "openai", "args": {"model_name": "budget-test"}}, budget_ledger=ledger
    )
    rejected = False
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(model.generate, "first")
        try:
            assert settled.wait(timeout=5)
            fake_openai.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
            try:
                model.generate("competing")
            except BudgetExceededError:
                rejected = True
        finally:
            finish.set()
        with pytest.raises(BudgetBreachError):
            pending.result(timeout=5)
    assert len(fake_openai.instances[-1].requests) == 1
    assert rejected
    assert Decimal(ledger.snapshot()["settled"]) == Decimal(cost)
