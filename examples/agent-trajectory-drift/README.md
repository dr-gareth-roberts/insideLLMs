# Agent trajectory drift

An agent that returns the **same answer** while taking a **different route** to
produce it. Output-level evaluation reports no change. This is what the
trajectory gate is for.

`support_agent.py` defines one probe in two versions. v1 looks the customer up
and answers. v2 looks them up, *also* checks billing, and returns the identical
sentence.

```console
$ pip install insidellms
$ python run_demo.py
wrote v1/records.jsonl
wrote v2/records.jsonl
```

The user-visible answer did not change, and the gate agrees:

```console
$ insidellms diff v1 v2 --fail-on-changes \
    --output-fingerprint-ignore trace_events,trace_fingerprint,tool_calls
$ echo $?
0
```

The route did change, and a different gate catches it:

```console
$ insidellms diff v1 v2 --fail-on-trajectory-drift
  Trajectory drifts: 2
── Trajectory Drifts ─────────────────────────────────
  dummy-v1 | support-agent | example 2cb44b1f797c98a1: trajectory steps 4 -> 6; tool calls 1 -> 2
  dummy-v1 | support-agent | example 55f3a13c24ed0228: trajectory steps 4 -> 6; tool calls 1 -> 2
$ echo $?
5
```

An extra tool call per request is real money, real latency, and a real
reliability surface. Nothing about the answer reveals it.

## What the gate actually compares

Not the tool-call count — a fingerprint of each step, which includes the tool
name, an `arguments_fingerprint` and a `result_fingerprint`
(`insideLLMs/runtime/diffing.py:301-334`). Two consequences:

**It catches route changes that keep the counts identical.** Swap one tool for
another at the same position and the gate still fires. A count comparison would
miss that.

**It also fires when only a tool's result changed.** Same route, same answer,
`{"plan": "pro"}` -> `{"plan": "enterprise"}`:

```console
$ insidellms diff a b --fail-on-trajectory-drift
  Trajectory drifts: 1
  dummy-v1 | support-agent | example 2cb44b1f: trajectory steps 4 -> 4; tool calls 1 -> 1
$ echo $?
5
```

So this is a gate for a **deterministic replay harness** — a fixed model and
stubbed tools, where every trajectory difference is a change you made. Point it
at live backends and it will fire on every run, because the data coming back
genuinely did change. Note also that the summary line is unhelpful in exactly
that case: `steps 4 -> 4; tool calls 1 -> 1` tells you nothing about what moved.

## Two more things worth knowing

**`--output-fingerprint-ignore` is doing real work here.** `AgentProbe` records
its trace into the structured output, so `trace_events`, `trace_fingerprint`
and `tool_calls` are part of the output fingerprint by default. The flag is how
you declare which parts of a structured output are user-visible and which are
internal. Without it, step 1 exits 2 — correctly, because *something* in the
output did change.

**Records are matched on model + probe + example.** Both versions register as
`support-agent` on purpose. Rename the probe between runs and the two runs share
no comparable records at all (`Common keys: 0`), because a renamed probe is a
different probe. The gate compares the same probe across two points in time.
