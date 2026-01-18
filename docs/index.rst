insideLLMs Documentation
========================

**insideLLMs** is a world-class Python library for probing the inner workings
of large language models. It provides comprehensive tools for evaluating,
testing, and understanding LLM behavior.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   installation

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   models
   probes
   runner
   registry

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules


Quick Links
-----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Features
--------

* **Multiple LLM Providers**: Support for OpenAI, Anthropic, HuggingFace, and more
* **Comprehensive Probes**: Logic, Bias, Attack, Factuality testing
* **Registry System**: Extensible plugin architecture
* **Async Support**: Asynchronous model execution
* **Streaming**: Real-time response streaming
* **Caching**: Built-in caching for repeated queries


Installation
------------

Install via pip:

.. code-block:: bash

    pip install insideLLMs

Or with optional dependencies:

.. code-block:: bash

    pip install insideLLMs[nlp,visualization]


Basic Usage
-----------

.. code-block:: python

    from insideLLMs import DummyModel, LogicProbe, ProbeRunner

    # Create model and probe
    model = DummyModel()
    probe = LogicProbe()

    # Run the probe
    runner = ProbeRunner(model, probe)
    results = runner.run(["What comes next: 1, 2, 3, ?"])
