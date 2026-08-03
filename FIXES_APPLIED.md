# Post-#106 review fixes

A ten-angle review of the merged PR #106 diff produced 56 raw findings (~38
distinct after deduplication). This document records what was changed, where,
how each change was verified, and — equally important — what was **not** changed
and why.

PR #106 is merged, so this is a fresh follow-up branch off the new `main`
(`1eac507`), not a continuation of that PR.

**Verification standard.** Most fixes were verified *causally*: revert only the
source change, confirm the new regression test fails, restore, confirm it
passes. Where that was impractical the reason is stated inline. Baseline for
comparison is merged `main` at `1eac507`: 7106 tests passing, mypy clean.

**Final state:** 7177 passed, 0 failed, 331 skipped · `ruff check` clean ·
`ruff format` clean · `mypy` clean (235 files).

Fixes 18–22 were found in review *of this branch* — including one regression this
branch introduced (18) and three places where an earlier fix here was incomplete
(19, 20, 22). They are recorded in the same detail as the rest.

---

## 1. Policy verdicts passed runs with missing transparency receipts

**File:** `insideLLMs/policy/engine.py`

`run_policy` handled *receipt + attestation* and *receipt without attestation*,
but left **attestation without receipt** unhandled. No `scitt_*` check was
recorded and the verdict stayed `passed=True`, so a run with an incomplete
transparency record read as compliant. This is precisely the dishonest-trust
signal P0.2/P0.3 existed to eliminate, in a file #106 edited.

Every asymmetry now fails closed and records a check, so no combination can pass
silently.

**Verified causally** — pre-fix the new test failed with `assert True is False`.
**Tests:** `tests/test_policy_engine.py` (+5, incl. the happy path).

## 2. Async chat dispatch could never reach its documented fallback

**Files:** `insideLLMs/models/base.py`, `insideLLMs/runtime/pipeline.py`

PR #106 converted streaming dispatch to `can_stream`/`can_stream_async` *because*
`hasattr()` sees inherited raising stubs — then left the peer chat paths on
`hasattr(model, "achat")`. `Model.chat` and `AsyncModel.achat` are both concrete
raising stubs, so `ModelPipeline.achat`, `Middleware.aprocess_chat` and
`TraceMiddleware.aprocess_chat` raised `NotImplementedError` instead of reaching
the run-in-executor fallback they document.

Added `can_chat`, `can_chat_async`, `can_generate_async` to complete the
predicate family; routed all chat dispatch through them.

**Verified causally.** **Tests:** `tests/inference/test_review_regressions.py` (+2).

> `can_generate_async` is a **consistency hardening, not a bug fix**:
> `AsyncModel.agenerate` is abstract, so that path was already safe.

## 3. Two defects in generation dispatch

**File:** `insideLLMs/inference/adapters.py`

- A model whose `generate()` is `async def` was handed to `asyncio.to_thread`
  without resolving the result, so the **coroutine object became the answer
  text**, plus a never-awaited `RuntimeWarning`.
- `agenerate` (metadata-less) was preferred over `generate_with_metadata`,
  **zeroing token/latency `Spend`** for any model offering both — notably the
  pipelines built by `InferenceClient.from_model_config`, which made
  matched-compute reports compare zeros.

Dispatch now prefers metadata-bearing sources and resolves awaitables from
either path. `_has_async_generation` derives from the same selection `_generate`
performs, so the two can no longer disagree. Dropped the duplicated local
`_resolve` in favour of `_callbacks.resolve`.

**Verified causally.** **Tests:** +3.

> **Follow-up: this fix initially caused a concurrency regression**, caught in
> review of this PR. `ModelPipeline` inherits a *synchronous*
> `Model.generate_with_metadata` and defines a metadata-less `agenerate`, so
> preferring metadata selected the sync path and `_has_async_generation()`
> became `False` — every `InferenceClient.from_model_config` client sampled
> **sequentially**. Measured: 5 samples × 50 ms took 0.25 s instead of 0.05 s.
>
> Resolved by adding `ModelPipeline.agenerate_with_metadata` (an async peer of
> `Model.generate_with_metadata`) so one method carries both properties, rather
> than choosing between accounting and concurrency. Verified: selection returns
> `agenerate_with_metadata`, `_has_async_generation()` is `True`, the same 5
> samples take 0.05 s, and latency metadata is still recorded.

## 4. Provider timeouts reported as budget exhaustion

**Files:** `inference/{dag,search,evolution,escalation}.py`, `inference/_callbacks.py`

Every budgeted strategy caught `TimeoutError` and asked only *"was a budget
armed?"* before relabelling it `*BudgetExceeded`. With a budget armed, a
callback raising its own timeout was reported as exhaustion **even when the
entire budget remained**. `escalate_adaptively` was worse: it *swallowed* the
error and returned the previous cheaper candidate with `stop_reason=BUDGET`,
silently downgrading the answer and hiding a provider outage.

Added `_callbacks.budget_elapsed`, which recomputes the deadline and is exact —
`invoke_with_timeout` only raises once the remaining budget has passed, so no
tolerance is needed. This also retires the discriminator that was copy-pasted
across four modules.

**Verified causally** — pre-fix, a DAG run reported `DagBudgetExceeded` with
**299 of 300 seconds remaining**, and escalation returned `'cheap-answer'` with
`StopReason.BUDGET`. **Tests:** +3.

> The #106 CHANGELOG entry claiming callback timeouts were "no longer
> mislabelled" was **corrected** — it only covered the no-budget-armed case.

## 5. Prefix-cache keys collided across models — BREAKING

**File:** `insideLLMs/inference/prefix_cache.py`

`compose_cached_prompt`'s `model_id` was optional with a `""` default, so two
different models sharing a stable prefix and tenant produced **byte-identical
cache keys**. A shared KV/response cache would serve one model's completion for
another model's request.

`model_id` is now **required** and validated.

> I declined this during the #106 review on breaking-change grounds. That
> reasoning does not survive the reproduction: the API is one commit old with
> zero production callers, and no default value can avoid being silently wrong
> when the failure mode is serving the wrong model's output. Recorded in
> CHANGELOG under **Changed (breaking)**.

**Tests:** `tests/inference/test_prefix_cache.py` (+1); callers updated.

## 6. Self-consistency returned the wrong answer

**File:** `insideLLMs/inference/client.py`

`_default_vote_normalizer` had two defects:

- `normalized_answer or output` conflated a deliberate `""` with unset.
- `normalize_text` strips punctuation and articles, so `"!!!"`, `"..."`, `"?!"`
  all normalize to `""`. Returning that as a vote key made unrelated junk a
  single bloc: `["4","!!!","...","4","?!"]` tallied `{'4': 2, '': 3}` and
  returned **`'!!!'`** over the genuine modal answer.

Empty normalizations now abstain; `normalized_answer` is checked against `None`.

Also collapsed the token/latency extraction copy-pasted at **three** sites into
one `_usage()` helper, so `Spend` totals cannot desynchronize from trace numbers.

**Verified causally.** **Tests:** +2.

## 7. `best_of_n` under-reported model calls

**File:** `insideLLMs/inference/best_of_n.py`

- `Spend.calls` came from the *requested* `n`, not the candidates the generator
  actually produced — an over-sampling or retrying generator under-reported real
  model calls and slipped past the matched-compute ceiling (5 generations
  reported as 2).
- The judge was always charged 2 calls, raising a spurious
  `ComputeMismatchError` for a purely **local** judge. Added `judge_model_calls`
  (default 2, preserving behaviour) so a local judge can declare 0.
- `provenance["judge_order_debiased"]` reported `True` whenever a judge was
  supplied, including runs where it is never invoked.

**Verified causally.** **Tests:** +1.

## 8. Tool retry caught too much; policy error fired after side effects

**File:** `insideLLMs/inference/tools.py`

- `except (ConnectionError, OSError, TimeoutError)` — `ConnectionError` is
  already an `OSError`, so the real effect was retrying **every** `OSError`
  (`FileNotFoundError`, `PermissionError` — deterministic, can never succeed)
  **plus our own per-attempt deadline**, multiplying the declared timeout by
  `max_transport_attempts`.
- Exceeding `max_output_characters` raised bare `ToolPolicyError`, whose
  docstring promises rejection *before* execution — so a caller could retry a
  non-idempotent action and duplicate its side effect.

Retries are now limited to `ConnectionError` and transport timeouts that are not
our deadline. Added `ToolOutputTooLarge` (a **subclass**, so existing handlers
keep working) documenting that the tool already ran.

**Verified after fix:** a 0.5 s timeout takes 0.50 s (not ~1.5 s);
`FileNotFoundError` fails in 0.00 s. **Tests:** +4.

## 9. Evolution elites were selected twice as often as intended

**File:** `insideLLMs/inference/evolution.py`

`parent_pool` concatenated `population` with `next_population`, which is seeded
from population's elites — so every elite was listed twice, doubling its draw
probability in the default tournament and handing custom `select_parent`
callbacks duplicate candidate ids.

**Verified causally** — pools of 5 entries with 3 unique ids. **Tests:** +1.

## 10. Async confidence callback crashed mid-run

**File:** `insideLLMs/inference/escalation.py`

`confidence()` was invoked bare while every sibling callback accepts an
awaitable, so an async callback returned a coroutine that crashed at
`1.0 - score` — *after* the step's model call had been paid for.

**Verified causally** — reproduced `TypeError: unsupported operand type(s) for
-: 'float' and 'coroutine'`. **Tests:** +1.

## 11. Beam search burned calls after its budget was hit

**File:** `insideLLMs/inference/search.py`

The budget branch used `continue`, so every remaining action still invoked the
model-backed `transition` even though monotonic counters guaranteed no child
could be admitted. **40 → 4** transition calls in a representative case.

**Verified causally.** **Tests:** +1.

## 12. Failed `gather` left sibling model calls running

**Files:** `inference/{best_of_n,retrieval}.py`, `inference/_callbacks.py`

Three bare `asyncio.gather` calls propagated the first exception while siblings
kept executing, so provider calls continued after the caller raised and a second
failure surfaced only as *"Task exception was never retrieved"* at GC.

Added `_callbacks.gather_cancelling`, factoring out the cancel-and-drain idiom
already used in `adapters.sample` and `dag`.

**Verified:** 0 orphans where siblings previously completed post-raise. **Tests:** +1.

## 13. `Budget.max_tokens` meant three different things

**Files:** `inference/{search,dag,retrieval}.py`

Three ad-hoc counters shipped in one commit: `search` counted **characters**,
`dag` and `retrieval` counted **whitespace words** — none matching the shared
`insideLLMs.tokens.estimate_tokens`. A 200-character state was 200 tokens to
search and 50 to the shared estimator. All three defaults now use it.

## 14. Non-float scorers regressed to `TypeError`

**File:** `insideLLMs/contrib/ensemble.py`

Routing `_best_of_n` through `rank_candidates` called `math.isnan` on every
score, so scorers returning **str or tuple** — valid with the previous
`max(key=scorer)` — raised `TypeError: must be real number, not str`. The
rewrite also fabricated a `Candidate` per output to encode a list index into its
id and parse it back, and ran an O(n²) selection sort for an argmax.

Restored to a plain `max()`, **keeping** #106's NaN-ranks-last improvement.

**Verified causally.** **Tests:** `tests/contrib/test_ensemble.py` (+1, plus a
rewritten NaN test — see below).

## 15. A test asserted the opposite of its name

**File:** `tests/contrib/test_ensemble.py`

`test_best_of_n_preserves_legacy_max_semantics_for_nan_scores` only covered NaN
in a **middle** position, where legacy and current behaviour agree. The actual
divergence — NaN **first**, which legacy `max` let win — was uncovered.

Renamed to `test_best_of_n_ranks_nan_scores_last` and extended to cover
NaN-first and all-NaN, with the deliberate divergence documented.

## 16. Matched-compute executor identity was wrong in both directions

**File:** `insideLLMs/analysis/matched_compute.py`

`_executor_identity` unwrapped one `.model` level from *any* executor, resolving
to inconsistent depths: a model reached directly yielded its own inner attribute
(`HuggingFaceModel.model` is the transformers module) while the same model
reached through an `InferenceClient` yielded the model itself.

**Verified causally, both failure directions:** an identical setup raised
`ComputeMismatchError`, **and** two genuinely different wrappers sharing an inner
module **matched falsely** — a hole in the gate whose job is keeping comparisons
honest. Only the known client wrapper is unwrapped now.

Reports also drop the live `executor` reference once the identity gate has run,
so a retained report cannot pin a loaded model's weights for the process
lifetime. **Tests:** `tests/test_matched_compute.py` (+1).

## 17. Dead code and a self-contradicting comment

- **`inference/dag.py`** — the in-loop cycle `raise` is unreachable (the upfront
  Kahn validation rejects every cyclic plan regardless of checkpoint state).
  Converted to an assertion so cycles have one authority, not two that can drift.
- **`benchmarks/inference_strategies.py`** — `STRATEGY_MODULES` was a hardcoded
  literal whose comment claimed it was *"derived rather than hardcoded so adding
  or removing a strategy cannot leave the reported count stale"* — exactly the
  drift it promised to prevent. Now genuinely derived via `pkgutil`; verified it
  produces the same 11 entries.

---

## 18. This PR cost the main production path its concurrency

`ModelPipeline` inherits a synchronous `Model.generate_with_metadata` and defined
a metadata-less `agenerate`, so fix 3's metadata-preferring dispatch selected the
sync method and `ModelProposer._has_async_generation()` became `False`. Every
client from `InferenceClient.from_model_config` then sampled **sequentially**:
five 50 ms samples took 0.25 s instead of 0.05 s. Fixing the zeroed-`Spend`
defect had quietly traded away concurrency on the main path.

Added `ModelPipeline.agenerate_with_metadata` so one method carries both
properties rather than forcing a choice. *Verified:* selection returns
`agenerate_with_metadata`, `_has_async_generation()` is `True`, the same five
samples complete in 0.05 s, and `spend.elapsed_seconds > 0`.

## 19. Sync chat dispatch had the same stub-blindness as its async peer

Fix 2 converted the `achat` paths but left their synchronous counterparts on
`hasattr(model, "chat")`. Only `generate` is abstract on `Model`; `chat` is a
concrete raising stub, so the check was **always** `True` and the
`ModelError("No chat implementation available")` guard in
`Middleware.process_chat`, `TraceMiddleware.process_chat` and
`ModelPipeline.chat` was dead code. A caller wrapping the pipeline in
`except ModelError` received a bare `NotImplementedError` it never caught.

All three now gate on `can_chat`; the two middleware docstring templates that
taught the `hasattr` pattern were corrected too. *Verified causally:* reverting
the source makes the new test fail with the raising stub.

## 20. `max_seconds` did not bound the confidence callback

Fix 10 routed escalation's `confidence` through `invoke` so an async scorer was
awaited, but left it *outside* the deadline. `confidence` is documented as
usually cheap yet is explicitly allowed to be model-backed — the code after the
loop reasons about exactly that case — so one hanging scorer could overrun the
budget without bound. **Measured: 0.40 s against a 0.05 s budget.**

Scoring now runs under the recomputed remaining budget. Three properties held
deliberately:

- **Sync callbacks stay supported.** `invoke_with_timeout` runs them via
  `asyncio.to_thread`, so the deadline fires without banning the form. An earlier
  attempt at this fix *rejected* sync confidence under a budget; that broke
  supported callers and was reverted, and is not reintroduced here.
- **A scorer's own `TimeoutError` still propagates** rather than being relabelled
  `StopReason.BUDGET` — the same `budget_elapsed` discrimination as fix 4.
- **An unscored candidate is dropped, not admitted with a fabricated score.**
  Exhaustion during scoring returns the previously scored answer with
  `StopReason.BUDGET`; it does not invent a confidence.

*Verified causally,* including the "earlier work survives" path (returns
`cheap-answer` at confidence 0.1, one candidate, `BUDGET`).

## 21. Tool timeout provenance was inferred, not recorded

`execute_tool` decided whether a `TimeoutError` was its own per-attempt deadline
or the tool's transport timeout by measuring elapsed time *in the handler*. That
is not exact: if the tool stalls the event loop — a blocking segment, a GC pause
— our deadline callback cannot fire, yet the elapsed measurement reads at-or-over
the limit, so a genuine retryable fault is denied its retry. Reproduced
deterministically: a tool that blocks 0.06 s against a 0.05 s limit and then
raises `TimeoutError("upstream read timeout")` was **not retried**, despite the
message proving it was the tool's own error.

The provenance is now recorded at the raise site by a per-attempt flag, so no
timing inference remains. *Verified:* the transport timeout is retried
(`attempts=2`), our own deadline is still never retried and the declared 0.05 s
timeout is not multiplied by `max_transport_attempts`, and `ConnectionError`
retry is unchanged.

## 22. Judge calls were charged to `Spend` but missing from the trace

Fix 7 added `judge_model_calls` to `Spend.calls` and left a comment stating that
"a consumer summing `TraceEvent.calls` must not derive a different total from
spend" — while emitting no judge event, so that consumer under-counted by exactly
`judge_model_calls`. **Measured: `spend.calls=5` against a trace total of 3.**

Adds a `judge-order-debias` event, parented on the eligible candidates'
verification events and emitted only when the judge actually ran. *Verified*
across all four cases: model judge (5/5), local judge with `judge_model_calls=0`
(3/3), single eligible candidate so the judge is never invoked (1/1), and no
judge (3/3).

---

## Deliberately not changed

Recorded rather than silently skipped.

| Finding | Decision |
|---|---|
| `ModelProposer.sample` serializes sync models (efficiency) | **Kept.** Carries an explicit comment that worker-thread cancellation cannot stop a running sync call. A real trade-off, not an oversight; parallelising would trade correctness for latency. |
| `sync.run_sync` "duplicates" `async_utils.run_async` | **Declined.** Semantics differ on purpose: `run_async` applies `nest_asyncio` to re-enter a running loop; `run_sync` refuses. Merging would force an optional monkey-patching dependency into inference. Documented the contrast instead. |
| `tools.py` retry "duplicates" `retry.py` | **Declined.** That engine owns its own attempt loop and exception classification, while this retry must interleave with a per-attempt deadline and feed `Observation.transport_attempts`. Coupling would cost more than the duplication. |
| `prefix_cache` should use `caching.generate_cache_key` | **Declined.** Different scoping semantics; would change the key format for no safety gain. The required `model_id` is the actual fix. |
| `ComputeProfile.executor_id` `id()` aliasing | **Kept.** Caveat already documented; public factories always populate `executor`, so the aliasing fallback is unreachable through them. Changing a required field of a public dataclass is not warranted by the residual risk. |
| `client.from_model_config` imports a private runtime helper | **Kept, documented.** Avoids duplicating registry/middleware assembly. The coupling is fenced by `tests/inference/test_architecture.py`, which permits a runtime import in that module and nowhere else. Comment says to promote the loader before adding a second consumer. |
| Evolution re-sorts the parent pool per attempt (efficiency) | **Not addressed.** The correctness fix (deduplication) landed; the O(P² log P) re-sort is negligible for realistic population sizes and restructuring the loop risks correctness for little gain. |
| `rank_candidates(tie_break="input_order")` now has no caller | **Kept.** Public, exported and tested API shipped in #106; removing it is a breaking change with no safety benefit. |
| Remaining `isinstance(model, AsyncModelProtocol)` sites in `pipeline.py` | **Kept — verified not a defect.** Flagged as sharing fix 2's stub-blindness; it does not. `AsyncModel.agenerate` is `@abstractmethod` (`AsyncModel.__abstractmethods__ == {'agenerate', 'generate'}`), so no concrete subclass can carry the stub, and plain `Model` has no `agenerate` at all — checked directly, `isinstance(DummyModel(), AsyncModelProtocol)` is `False` and `ModelWrapper` delegation does not fabricate one. The executor fallback is reachable, so switching the seven sites would be churn with no behavioural change. `Model.chat` *is* a concrete stub, which is why fix 19 is real and this is not. |

---

## Files changed

**Source**

| File | Change |
|---|---|
| `insideLLMs/policy/engine.py` | SCITT receipt asymmetries fail closed |
| `insideLLMs/models/base.py` | `can_chat`, `can_chat_async`, `can_generate_async` |
| `insideLLMs/runtime/pipeline.py` | sync + async chat and `agenerate` dispatch via capability predicates; `ModelPipeline.agenerate_with_metadata` |
| `insideLLMs/inference/adapters.py` | metadata-preferring dispatch; awaitable resolution; shared `resolve` |
| `insideLLMs/inference/_callbacks.py` | `budget_elapsed`, `gather_cancelling` |
| `insideLLMs/inference/dag.py` | deadline discrimination; shared token estimator; dead cycle check → assertion |
| `insideLLMs/inference/search.py` | deadline discrimination; budget `break`; shared token estimator |
| `insideLLMs/inference/evolution.py` | deadline discrimination; elite deduplication |
| `insideLLMs/inference/escalation.py` | deadline discrimination; async confidence via `invoke`; confidence bounded by the remaining budget |
| `insideLLMs/inference/prefix_cache.py` | `model_id` required and validated |
| `insideLLMs/inference/client.py` | vote normalizer; `_usage` helper; documented runtime coupling |
| `insideLLMs/inference/best_of_n.py` | actual call counting; `judge_model_calls`; `judge_ran` provenance; cancelling gather; judge trace event |
| `insideLLMs/inference/retrieval.py` | cancelling gather; shared token estimator |
| `insideLLMs/inference/tools.py` | narrowed retry scope; `ToolOutputTooLarge`; `_backoff_seconds`; timeout provenance recorded at the raise site |
| `insideLLMs/inference/sync.py` | documented divergence from `run_async` |
| `insideLLMs/inference/__init__.py` | export `ToolOutputTooLarge` |
| `insideLLMs/analysis/matched_compute.py` | wrapper-aware identity; executor dropped from reports |
| `insideLLMs/contrib/ensemble.py` | `max()` restored; NaN-last kept |
| `benchmarks/inference_strategies.py` | derived `STRATEGY_MODULES`; `model_id` supplied |
| `CHANGELOG.md` | Fixed + Changed (breaking); corrected the #106 timeout claim |
| `FIXES_APPLIED.md` | This document |

**Tests** — `tests/test_policy_engine.py` (+5), `tests/inference/test_review_regressions.py` (+27),
`tests/inference/test_prefix_cache.py` (+1), `tests/test_matched_compute.py` (+1),
`tests/contrib/test_ensemble.py` (+1 new, 1 rewritten).

## Commits

| SHA | Subject |
|---|---|
| `386aced` | Fix policy fail-open, stub-blind chat dispatch, and adapter generation defects |
| `d6cf4e5` | Fix timeout-vs-budget misattribution and prefix-cache key collision |
| `0cf9c34` | Fix self-consistency vote clustering, call accounting, and tool retry scope |
| `95dbcfb` | Fix elite double-weighting, async confidence, budget overrun, orphaned tasks |
| `8e47ab2` | Unify token counting, restore scorer support, fix executor identity |
| `c563ef3` | Remove dead code, derive strategy list, document declined findings |
| `a417277` | Restore pipeline sampling concurrency and address review findings |
| _this_ | Bound the confidence deadline, record timeout provenance, complete chat dispatch |
