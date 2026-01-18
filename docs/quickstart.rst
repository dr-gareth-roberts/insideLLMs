Quick Start Guide
=================

This guide will help you get started with insideLLMs in just a few minutes.


Installation
------------

Install insideLLMs using pip:

.. code-block:: bash

    pip install insideLLMs

For NLP utilities:

.. code-block:: bash

    pip install insideLLMs[nlp]


Your First Experiment
---------------------

Let's create a simple experiment using the DummyModel (no API key required):

.. code-block:: python

    from insideLLMs import DummyModel, LogicProbe, ProbeRunner

    # 1. Create a model
    model = DummyModel()

    # 2. Create a probe
    probe = LogicProbe()

    # 3. Create a runner
    runner = ProbeRunner(model, probe)

    # 4. Run on test data
    test_data = [
        "What comes next: 2, 4, 6, ?",
        "If A is true and B is false, what is A AND B?",
    ]

    results = runner.run(test_data)
    print(results)


Using Real Models
-----------------

To use OpenAI models:

.. code-block:: python

    import os
    from insideLLMs import model_registry, LogicProbe, ProbeRunner

    # Set your API key
    os.environ["OPENAI_API_KEY"] = "your-key-here"

    # Get OpenAI model from registry
    model = model_registry.get("openai", model_name="gpt-4")

    # Run a probe
    probe = LogicProbe()
    runner = ProbeRunner(model, probe)
    results = runner.run(["Solve: 2x + 5 = 15"])


Available Probes
----------------

insideLLMs includes several built-in probes:

* **LogicProbe**: Tests logical reasoning
* **BiasProbe**: Detects bias in responses
* **AttackProbe**: Tests vulnerability to adversarial inputs
* **FactualityProbe**: Verifies factual accuracy

.. code-block:: python

    from insideLLMs import (
        LogicProbe,
        BiasProbe,
        AttackProbe,
        FactualityProbe,
    )


Using the Registry
------------------

The registry system allows you to discover and instantiate components by name:

.. code-block:: python

    from insideLLMs import model_registry, probe_registry

    # List available items
    print(model_registry.list())  # ['dummy', 'openai', 'anthropic', ...]
    print(probe_registry.list())  # ['logic', 'bias', 'attack', ...]

    # Get by name
    model = model_registry.get("dummy")
    probe = probe_registry.get("logic")


Saving Results
--------------

Save and load experiment results:

.. code-block:: python

    from insideLLMs import save_results_json, load_results_json

    # Save
    save_results_json(results, "my_experiment.json")

    # Load
    loaded = load_results_json("my_experiment.json")


Next Steps
----------

* Explore the :doc:`models` documentation for available model implementations
* Learn about :doc:`probes` for testing different aspects of LLMs
* Check out the :doc:`registry` for the plugin system
* See the API reference for detailed documentation
