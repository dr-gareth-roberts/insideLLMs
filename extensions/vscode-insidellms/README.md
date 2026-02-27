# insideLLMs VS Code / Cursor Extension (Scaffold)

This extension adds Python `CodeLens` actions that run:

```bash
insidellms harness <config> --run-dir <run_dir> --overwrite --skip-report
```

## Features

- `Run insideLLMs probes` lens at the top of every Python file.
- Additional `Run insideLLMs probes` lenses near `prompt` assignments.
- Configurable harness path and IDE run directory.

## Settings

- `insidellms.harnessConfigPath` (default: `ci/harness.yaml`)
- `insidellms.runDir` (default: `.tmp/runs/ide`)

## Build Locally

```bash
cd extensions/vscode-insidellms
npm install
npm run build
```
