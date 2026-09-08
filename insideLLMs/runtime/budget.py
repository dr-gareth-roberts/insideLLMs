"""Pre-dispatch admission under explicit, fixed pricing and request bounds.

The policy is an application allowance, not an assertion about a provider's
invoice. Input bounds must cover the provider's maximum billable context.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, localcontext
from threading import Lock
from typing import Any

from insideLLMs.config_schema import BudgetPolicy, BudgetRequestPolicy
from insideLLMs.exceptions import InsideLLMsError


class BudgetError(InsideLLMsError):
    """A fatal invocation stop, never eligible for a model retry."""

    retryable = False

    def __init__(self, message: str, *, code: str = "budget_unsupported") -> None:
        self.abort_reason = {"code": code, "message": message}
        super().__init__(message, details={"abort_reason": self.abort_reason})


class BudgetExceededError(BudgetError):
    def __init__(self, message: str = "The invocation budget cannot admit another request") -> None:
        super().__init__(message, code="budget_exceeded")


class BudgetUnsupportedError(BudgetError):
    pass


class BudgetBreachError(BudgetError):
    def __init__(self, message: str = "Reported usage exceeded its reserved upper bound") -> None:
        super().__init__(message, code="budget_breach")


@dataclass(frozen=True)
class Reservation:
    identifier: int
    amount: Decimal


def _money(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError("Costs must be finite nonnegative Decimal values")
    if value.adjusted() > 30 or int(value.as_tuple().exponent) < -18:
        raise ValueError("Costs exceed the supported monetary precision/range")
    return value


class BudgetLedger:
    """One lock protects allowance shared by threads, tasks, models and retries."""

    def __init__(self, policy: BudgetPolicy) -> None:
        self.policy = policy
        self._lock = Lock()
        self._entries: dict[int, tuple[Reservation, str]] = {}
        self._settled = Decimal(0)
        self._uncertain = Decimal(0)
        self._reserved = Decimal(0)
        self.abort_reason: dict[str, str] | None = None

    def reserve(self, amount: Decimal) -> Reservation:
        amount = _money(amount)
        with self._lock, localcontext() as context:
            context.prec = 80
            liability = self._settled + self._uncertain + self._reserved + amount
            if self.abort_reason or liability > self.policy.allowance:
                error = BudgetExceededError()
                self.abort_reason = self.abort_reason or error.abort_reason
                raise error
            reservation = Reservation(len(self._entries), amount)
            self._entries[reservation.identifier] = (reservation, "reserved")
            self._reserved += amount
            return reservation

    def settle(
        self,
        reservation: Reservation,
        *,
        actual_cost: Decimal | None,
        violation_reason: str | None = None,
    ) -> None:
        """None retains full liability; repeat settlements never change totals."""
        if actual_cost is not None:
            _money(actual_cost)
        with self._lock, localcontext() as context:
            context.prec = 80
            existing, state = self._entries[reservation.identifier]
            if existing is not reservation:
                raise ValueError("Reservation belongs to a different ledger")
            if state != "reserved":
                return
            self._reserved -= reservation.amount
            self._entries[reservation.identifier] = (reservation, "settled")
            if actual_cost is None:
                self._uncertain += reservation.amount
            else:
                self._settled += actual_cost
            if violation_reason or (actual_cost is not None and actual_cost > reservation.amount):
                error = BudgetBreachError(
                    violation_reason or "Reported usage exceeded its reserved upper bound"
                )
                # Releasing a reservation and stopping on invalid bounds must
                # be one transaction: another worker must not reuse this money.
                self.abort_reason = error.abort_reason
                raise error

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "currency": self.policy.currency,
                "scope": self.policy.scope,
                "allowance": str(self.policy.allowance),
                "settled": str(self._settled),
                "uncertain": str(self._uncertain),
                "reserved": str(self._reserved),
                "aborted": self.abort_reason is not None,
                "abort_reason": dict(self.abort_reason) if self.abort_reason else None,
            }


def request_policy(ledger: BudgetLedger, provider: str, model: str) -> BudgetRequestPolicy:
    for price in ledger.policy.prices:
        if (price.provider, price.model) == (provider, model):
            return price
    raise BudgetUnsupportedError(f"No explicit budget price/bounds for {provider}/{model}")


def validate_budget_model_config(config: dict[str, Any], ledger: BudgetLedger) -> None:
    """Reject unknown factories BEFORE invoking their possibly effectful constructors."""
    from insideLLMs.registry import get_builtin_model_factory, model_registry

    _require_plain_data(config)
    if set(config) - {"type", "args", "pipeline"}:
        raise BudgetUnsupportedError("Budget model config supports only type, args and pipeline")
    provider = config.get("type")
    canonical = get_builtin_model_factory(provider)
    if (
        provider not in {"dummy", "openai", "anthropic"}
        or canonical is None
        or model_registry.get_factory(provider) is not canonical
        or model_registry.info(provider).get("default_kwargs")
    ):
        raise BudgetUnsupportedError("Budget mode requires an unchanged supported builtin model")
    if provider == "dummy":
        return
    args = config.get("args", {})
    price = request_policy(ledger, provider, args.get("model_name", ""))
    if args.get("max_retries", 0) != 0:
        raise BudgetUnsupportedError("Budget mode requires SDK max_retries=0")
    if args.get("base_url", price.endpoint) != price.endpoint:
        raise BudgetUnsupportedError("Budget mode requires the exact priced endpoint")
    if args.get("default_headers"):
        raise BudgetUnsupportedError("Custom provider headers are unsupported in budget mode")


def _require_plain_data(value: Any) -> None:
    """Do not accept callback objects hiding behind otherwise builtin configs."""
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) in {list, tuple}:
        for item in value:
            _require_plain_data(item)
        return
    if type(value) is dict and all(isinstance(key, str) for key in value):
        for item in value.values():
            _require_plain_data(item)
        return
    raise BudgetUnsupportedError("Budget mode rejects arbitrary callback/model objects in config")


def validate_budget_probe_config(config: dict[str, Any], ledger: BudgetLedger) -> None:
    from insideLLMs.registry import get_builtin_probe_factory, probe_registry

    _require_plain_data(config)
    name = config.get("type")
    canonical = get_builtin_probe_factory(name)
    if (
        canonical is None
        or probe_registry.get_factory(name) is not canonical
        or probe_registry.info(name).get("default_kwargs")
    ):
        raise BudgetUnsupportedError("Budget mode requires an unchanged builtin probe")
    if name == "judge":
        judge = config.get("args", {}).get("judge_model")
        if not isinstance(judge, dict):
            raise BudgetUnsupportedError("A budgeted judge requires a nested model config")
        validate_budget_model_config(judge, ledger)
        _validate_generation_policy(judge, config.get("args", {}).get("judge_kwargs", {}), ledger)


def _validate_generation_policy(
    model: dict[str, Any], kwargs: dict[str, Any], ledger: BudgetLedger
) -> None:
    if model.get("type") != "dummy":
        price = request_policy(ledger, model["type"], model.get("args", {}).get("model_name", ""))
        _bounded_request(price, [{"role": "user", "content": ""}], kwargs)


def create_budget_ledger(config: dict[str, Any], *, resume: bool = False) -> BudgetLedger | None:
    """Create exactly once at invocation entry; callers share the returned ledger."""
    raw = config.get("budget")
    if raw is None:
        return None
    if resume:
        raise BudgetUnsupportedError(
            "Budgeted resume requires durable reservations and is unsupported"
        )
    return BudgetLedger(BudgetPolicy.model_validate(raw))


def preflight_budget_config(config: dict[str, Any], *, resume: bool = False) -> None:
    """Check every subject and judge before ANY model constructor or call runs."""
    from insideLLMs.registry import ensure_builtins_registered

    ledger = create_budget_ledger(config, resume=resume)
    if ledger is None:
        return
    ensure_builtins_registered(load_plugins=False)
    models = config.get("models") or [config.get("model", {})]
    probes = config.get("probes") or [config.get("probe", {})]
    for model in models:
        _require_plain_data(model)
        validate_budget_model_config(model, ledger)
    for probe in probes:
        validate_budget_probe_config(probe, ledger)
        kwargs: dict[str, Any] = {}
        for source in (config, probe):
            for key in ("generation", "probe_kwargs", "run_kwargs"):
                kwargs.update(source.get(key) or {})
        for model in models:
            _validate_generation_policy(model, kwargs, ledger)
    for key in ("generation", "probe_kwargs", "run_kwargs"):
        _require_plain_data(config.get(key))


def _validate_bound_model(model: Any, ledger: BudgetLedger) -> None:
    from importlib import import_module

    from insideLLMs.runtime.pipeline import (
        AsyncModelPipeline,
        CacheMiddleware,
        CostTrackingMiddleware,
        ModelPipeline,
        PassthroughMiddleware,
        RateLimitMiddleware,
        RetryMiddleware,
        TraceMiddleware,
    )

    allowed_middleware = {
        CacheMiddleware,
        CostTrackingMiddleware,
        PassthroughMiddleware,
        RateLimitMiddleware,
        RetryMiddleware,
        TraceMiddleware,
    }
    seen: set[int] = set()
    while type(model) in {ModelPipeline, AsyncModelPipeline}:
        if id(model) in seen:
            raise BudgetUnsupportedError("Cyclic budget model pipeline")
        seen.add(id(model))
        for index, middleware in enumerate(model.middlewares):
            following = model.middlewares[index + 1] if index + 1 < len(model.middlewares) else None
            if (
                type(middleware) not in allowed_middleware
                or middleware.model is not model.base_model
                or middleware.next_middleware is not following
                or any(callable(value) for value in vars(middleware).values())
            ):
                raise BudgetUnsupportedError("Budget mode rejects custom middleware callouts")
        model = model.base_model
    identity = (type(model).__module__, type(model).__name__)
    allowed_models = {
        ("insideLLMs.models", "DummyModel"),
        ("insideLLMs.models.openai", "OpenAIModel"),
        ("insideLLMs.models.anthropic", "AnthropicModel"),
    }
    if (
        identity not in allowed_models
        or type(model) is not getattr(import_module(identity[0]), identity[1])
        or getattr(model, "_budget_ledger", None) is not ledger
        or any(callable(value) for value in vars(model).values())
    ):
        raise BudgetUnsupportedError("Model must be a builtin bound to this invocation ledger")


def validate_budget_runner(model: Any, probe: Any, ledger: BudgetLedger) -> None:
    """Validate explicit runner plumbing; never implicitly trust or bind objects."""
    from insideLLMs.probes.judge import JudgeScoredProbe, JudgeScorer
    from insideLLMs.registry import get_builtin_probe_factory, probe_registry

    _validate_bound_model(model, ledger)
    canonical = any(
        type(probe) is get_builtin_probe_factory(name)
        and probe_registry.get_factory(name) is get_builtin_probe_factory(name)
        and not probe_registry.info(name).get("default_kwargs")
        for name in probe_registry.list()
    )
    if not canonical or any(callable(value) for value in vars(probe).values()):
        raise BudgetUnsupportedError("Budget mode rejects custom probes and callback validators")
    if type(probe) is JudgeScoredProbe:
        if type(probe.scorer) is not JudgeScorer:
            raise BudgetUnsupportedError("Budget mode requires the builtin judge scorer")
        _validate_bound_model(probe.scorer.judge_model, ledger)


def _bounded_request(
    price: BudgetRequestPolicy, messages: list[dict[str, Any]], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    allowed = {
        "temperature",
        "top_p",
        "stop",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        price.output_token_parameter,
    }
    if price.provider == "anthropic":
        allowed = {
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "system",
            "max_tokens",
        }
    if set(kwargs) - allowed:
        raise BudgetUnsupportedError("Budget mode supports only bounded single text completions")
    if not messages or any(
        not isinstance(message, dict)
        or set(message) - {"role", "content"}
        or message.get("role") not in {"system", "user", "assistant", "developer"}
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise BudgetUnsupportedError("Budget mode requires plain text messages")
    if "system" in kwargs and not isinstance(kwargs["system"], str):
        raise BudgetUnsupportedError("Budget mode requires plain text system content")
    cap = kwargs.get(price.output_token_parameter, price.maximum_output_tokens)
    if type(cap) is not int or not 0 < cap <= price.maximum_output_tokens:
        raise BudgetUnsupportedError("Requested output exceeds the explicit budget token cap")
    return {**kwargs, price.output_token_parameter: cap}, cap


def _token_cost(price: BudgetRequestPolicy, inputs: int, outputs: int) -> Decimal:
    with localcontext() as context:
        context.prec = 80
        return (
            inputs * price.input_cost_per_million + outputs * price.output_cost_per_million
        ) / Decimal(1000000)


def _usage_cost(response: Any, price: BudgetRequestPolicy, cap: int) -> tuple[Decimal | None, bool]:
    if getattr(response, "model", None) != price.model:
        return None, False
    usage = getattr(response, "usage", None)
    input_name, output_name = (
        ("prompt_tokens", "completion_tokens")
        if price.provider == "openai"
        else ("input_tokens", "output_tokens")
    )
    inputs, outputs = getattr(usage, input_name, None), getattr(usage, output_name, None)
    if type(inputs) is not int or type(outputs) is not int or min(inputs, outputs) < 0:
        return None, False
    if price.provider == "anthropic" and any(
        getattr(usage, name, 0) not in {None, 0}
        for name in ("cache_creation_input_tokens", "cache_read_input_tokens")
    ):
        return None, False
    return (
        _token_cost(price, inputs, outputs),
        inputs > price.maximum_input_tokens or outputs > cap,
    )


def dispatch_budgeted_call(
    ledger: BudgetLedger,
    *,
    provider: str,
    model: str,
    client: Any,
    messages: list[dict[str, Any]],
    kwargs: dict[str, Any],
    dispatch: Callable[..., Any],
) -> Any:
    """Reserve the full declared input context plus the enforced output cap."""
    price = request_policy(ledger, provider, model)
    if str(getattr(client, "base_url", "")).rstrip("/") != price.endpoint:
        raise BudgetUnsupportedError("SDK endpoint does not match the budget price")
    if getattr(client, "max_retries", None) != 0:
        raise BudgetUnsupportedError("SDK retry behavior is unknown or not disabled")
    bounded, cap = _bounded_request(price, messages, kwargs)
    quote = _token_cost(price, price.maximum_input_tokens, cap)
    reservation = ledger.reserve(quote)
    try:
        response = dispatch(model=model, messages=messages, **bounded)
    except BaseException:
        # A failed/cancelled transport may already have incurred provider charges.
        ledger.settle(reservation, actual_cost=None)
        raise
    try:
        actual_cost, bound_violated = _usage_cost(response, price, cap)
        ledger.settle(
            reservation,
            actual_cost=actual_cost,
            violation_reason=(
                "Reported token usage exceeds the supplied request bound"
                if bound_violated
                else None
            ),
        )
    except BudgetError:
        raise
    except Exception:
        ledger.settle(
            reservation,
            actual_cost=None,
            violation_reason="Provider usage could not be safely reconciled",
        )
    return response
