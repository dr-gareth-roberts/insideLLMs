---
title: IDE Extension Workflow
parent: Guides
nav_order: 7
---

# IDE Extension Workflow (VS Code / Cursor)

Use the extension scaffold to run probe workflows directly from your editor via
CodeLens.

## What the Extension Does

- Adds `Run insideLLMs probes` CodeLens actions in Python files.
- Executes:

```bash
insidellms harness <config> --run-dir <run_dir> --overwrite --skip-report
```

- Lets you set harness path and run directory in workspace settings.

## Install and Build (Scaffold)

```bash
cd extensions/vscode-insidellms
npm install
npm run build
```

## Settings

- `insidellms.harnessConfigPath` (default: `ci/harness.yaml`)
- `insidellms.runDir` (default: `.tmp/runs/ide`)

## Recommended Inner Loop

1. Edit prompt/probe code in your Python file.
2. Trigger `Run insideLLMs probes` CodeLens.
3. Inspect `records.jsonl`, `summary.json`, and `report.html` in the configured run dir.
4. If behavior changed intentionally, validate with `insidellms diff --interactive`.

## Roadmap (Current Direction)

- Inline diff summaries next to changed prompt regions.
- One-click `baseline vs candidate` local diff view.
- Optional judge triage (`--judge`) surfaced in editor diagnostics.
- Jump-to-record links from CodeLens to run artifacts.
