insideLLMs Documentation
========================

**insideLLMs** is a cross-model behavioural probe harness and toolkit for
evaluating and comparing large language models. It provides tools for
testing, analysis, and reporting.

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

* **Cross-model Harness**: Run the same probes across models with shared datasets
* **Multiple Providers**: OpenAI, Anthropic, HuggingFace, and more
* **Probes**: Logic, bias, attack, and factuality tests
* **Registry System**: Extensible plugin architecture
* **Async Support**: Asynchronous model execution
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
