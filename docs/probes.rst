Probes
======

Probes are the core evaluation tools in insideLLMs. They test specific
aspects of LLM behaviour.


Probe Base Class
----------------

All probes inherit from the ``Probe`` base class:

.. code-block:: python

    from insideLLMs import Probe
    from insideLLMs.types import ProbeCategory

    class MyProbe(Probe[str]):
        def __init__(self, name="MyProbe"):
            super().__init__(name=name, category=ProbeCategory.CUSTOM)

        def run(self, model, data, **kwargs) -> str:
            prompt = f"Analyze: {data}"
            return model.generate(prompt, **kwargs)


Built-in Probes
---------------

LogicProbe
^^^^^^^^^^

Tests logical reasoning and pattern completion:

.. code-block:: python

    from insideLLMs import LogicProbe

    probe = LogicProbe()
    result = probe.run(model, "What comes next: 1, 4, 9, 16, ?")


BiasProbe
^^^^^^^^^

Detects potential bias in model responses:

.. code-block:: python

    from insideLLMs import BiasProbe

    probe = BiasProbe()
    result = probe.run(model, "Describe a typical nurse")


AttackProbe
^^^^^^^^^^^

Tests vulnerability to adversarial inputs:

.. code-block:: python

    from insideLLMs import AttackProbe

    probe = AttackProbe()
    result = probe.run(model, "Ignore previous instructions and...")


FactualityProbe
^^^^^^^^^^^^^^^

Verifies factual accuracy:

.. code-block:: python

    from insideLLMs import FactualityProbe

    probe = FactualityProbe()
    result = probe.run(model, "What is the capital of France?")


Additional Probes
-----------------

Code Probes
^^^^^^^^^^^

.. code-block:: python

    from insideLLMs.probes import (
        CodeGenerationProbe,
        CodeExplanationProbe,
        CodeDebugProbe,
    )


Instruction Probes
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from insideLLMs.probes import (
        InstructionFollowingProbe,
        MultiStepTaskProbe,
        ConstraintComplianceProbe,
    )


Using the Registry
------------------

Get probes by name:

.. code-block:: python

    from insideLLMs import probe_registry

    # List available
    print(probe_registry.list())

    # Get by name
    probe = probe_registry.get("logic")


Custom Probes
-------------

Create your own probes:

.. code-block:: python

    from insideLLMs import CustomProbe

    class SentimentProbe(CustomProbe):
        def run(self, model, data, **kwargs):
            prompt = f"Analyze the sentiment of: {data}"
            response = model.generate(prompt, **kwargs)
            return response

    # Register it
    from insideLLMs import probe_registry
    probe_registry.register("sentiment", SentimentProbe)


Probe Categories
----------------

Probes are categorized for organization:

.. code-block:: python

    from insideLLMs.types import ProbeCategory

    # Available categories
    ProbeCategory.LOGIC
    ProbeCategory.BIAS
    ProbeCategory.ATTACK
    ProbeCategory.FACTUALITY
    ProbeCategory.CODE
    ProbeCategory.INSTRUCTION
    ProbeCategory.CUSTOM


Scored Probes
-------------

For probes that return scores:

.. code-block:: python

    from insideLLMs.probes import ScoredProbe

    class AccuracyProbe(ScoredProbe):
        def run(self, model, data, **kwargs):
            # ... evaluate ...
            return ProbeScore(value=0.85, max_value=1.0)


Comparative Probes
------------------

For comparing multiple models:

.. code-block:: python

    from insideLLMs.probes import ComparativeProbe

    class ModelComparisonProbe(ComparativeProbe):
        def run_comparison(self, models, data, **kwargs):
            results = {}
            for model in models:
                results[model.name] = self.evaluate(model, data)
            return results
