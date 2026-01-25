# FAQ

## Do I need API keys?

Only for hosted providers. You can use `DummyModel` for offline tests.

## Why can’t it find my dataset file?

Relative dataset paths are resolved relative to the config file’s directory (not the current
working directory). See [Configuration](Configuration).

## Why does `--overwrite` refuse to overwrite my run directory?

As a safety guard, insideLLMs only overwrites non-empty directories if they contain the
`.insidellms_run` marker file.

## Can I run local models?

Yes. insideLLMs supports local runners like Ollama and llama.cpp.

## How do I reduce cost?

Use `max_examples`, enable caching, and avoid large prompt sets in early iterations.

## Where can I find datasets?

Start with `data/` and `benchmarks/`, or use the built-in datasets in
`insideLLMs.benchmark_datasets`.

## How do I turn off coloured output?

Set `NO_COLOR=1` in your environment.

## How do I keep outputs out of `~/.insidellms`?

Use `--run-dir` / `--run-root`, or set `INSIDELLMS_RUN_ROOT`:

```bash
INSIDELLMS_RUN_ROOT=.tmp/insidellms_runs insidellms run examples/experiment.yaml
```
