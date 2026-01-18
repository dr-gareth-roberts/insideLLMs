Runner
======

The ProbeRunner is the main execution engine for running probes on models.


ProbeRunner
-----------

Basic usage:

.. code-block:: python

    from insideLLMs import DummyModel, LogicProbe, ProbeRunner

    model = DummyModel()
    probe = LogicProbe()
    runner = ProbeRunner(model, probe)

    # Run on single input
    result = runner.run(["What is 2+2?"])

    # Run on multiple inputs
    results = runner.run([
        "What comes next: 1, 2, 3, ?",
        "If A implies B, and A is true, what is B?",
        "Complete: red, blue, ?, yellow",
    ])


AsyncProbeRunner
----------------

For asynchronous execution:

.. code-block:: python

    import asyncio
    from insideLLMs import AsyncProbeRunner

    runner = AsyncProbeRunner(model, probe)

    async def main():
        results = await runner.run_async(["Question 1", "Question 2"])
        return results

    results = asyncio.run(main())


run_probe Function
------------------

A convenience function for quick probing:

.. code-block:: python

    from insideLLMs import run_probe

    result = run_probe(model, probe, "What is the capital of France?")


Creating Experiment Results
---------------------------

Create structured experiment results:

.. code-block:: python

    from insideLLMs import create_experiment_result

    result = create_experiment_result(
        model_name="gpt-4",
        probe_name="logic",
        inputs=["Q1", "Q2"],
        outputs=["A1", "A2"],
        metrics={"accuracy": 0.95},
    )


Batch Processing
----------------

Process large datasets efficiently:

.. code-block:: python

    from insideLLMs import ProbeRunner

    runner = ProbeRunner(model, probe)

    # Process in batches
    large_dataset = ["Q" + str(i) for i in range(1000)]
    results = runner.run(large_dataset)


Configuration-based Running
---------------------------

Run experiments from YAML configuration:

.. code-block:: python

    from insideLLMs.runner import run_experiment_from_config

    # config.yaml:
    # model:
    #   type: openai
    #   args:
    #     model_name: gpt-4o
    # probe:
    #   type: logic
    #   args: {}
    # dataset:
    #   format: jsonl
    #   path: test_data.jsonl

    results = run_experiment_from_config("config.yaml")


Harness Runs
------------

Run a cross-model behavioural harness from a single config:

.. code-block:: python

    from insideLLMs.runner import run_harness_from_config

    # harness.yaml:
    # models:
    #   - type: openai
    #     args:
    #       model_name: gpt-4o
    # probes:
    #   - type: logic
    #     args: {}
    #   - type: bias
    #     args: {}
    # dataset:
    #   format: jsonl
    #   path: data/questions.jsonl
    # max_examples: 50

    result = run_harness_from_config("harness.yaml")

    # result["records"] holds per-example rows
    # result["summary"] includes aggregates + confidence intervals

.. code-block:: bash

    insidellms harness harness.yaml



Progress Tracking
-----------------

Monitor progress during long runs:

.. code-block:: python

    from insideLLMs import ProbeRunner

    def on_progress(completed, total):
        print(f"Progress: {completed}/{total}")

    runner = ProbeRunner(model, probe, progress_callback=on_progress)
    results = runner.run(large_dataset)


Error Handling
--------------

Handle errors during execution:

.. code-block:: python

    from insideLLMs import ProbeRunner
    from insideLLMs.exceptions import ProbeExecutionError

    try:
        results = runner.run(data)
    except ProbeExecutionError as e:
        print(f"Probe failed: {e.message}")
        print(f"Failed at sample: {e.details.get('sample_index')}")
