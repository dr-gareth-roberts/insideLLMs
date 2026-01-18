Registry System
===============

insideLLMs uses a registry pattern for extensible plugin architecture.


Overview
--------

The registry system allows:

* Discovering available models and probes
* Getting instances by name
* Registering custom implementations
* Building plugin architectures


Global Registries
-----------------

Three global registries are available:

.. code-block:: python

    from insideLLMs import model_registry, probe_registry
    from insideLLMs.registry import dataset_registry

    # List registered items
    print(model_registry.list())   # ['dummy', 'openai', 'anthropic', ...]
    print(probe_registry.list())   # ['logic', 'bias', 'attack', ...]
    print(dataset_registry.list()) # ['csv', 'jsonl', 'hf', ...]


Getting Instances
-----------------

Get instances by name:

.. code-block:: python

    # Basic usage
    model = model_registry.get("dummy")

    # With parameters
    model = model_registry.get("openai", model_name="gpt-4", temperature=0.7)

    # Override defaults
    probe = probe_registry.get("logic")


Registering Custom Items
------------------------

Register your own implementations:

.. code-block:: python

    from insideLLMs import Model, model_registry

    class MyCustomModel(Model):
        def generate(self, prompt, **kwargs):
            return f"Custom: {prompt}"

        def chat(self, messages, **kwargs):
            return "Custom chat"

        def stream(self, prompt, **kwargs):
            yield "Custom stream"

    # Register with defaults
    model_registry.register("my_custom", MyCustomModel)

    # Register with default arguments
    model_registry.register(
        "my_custom_v2",
        MyCustomModel,
        name="CustomV2",
    )


Decorator Registration
----------------------

Use decorators for cleaner registration:

.. code-block:: python

    @model_registry.register_decorator("decorated_model")
    class DecoratedModel(Model):
        ...

    # With defaults
    @probe_registry.register_decorator("decorated_probe", some_param="default")
    class DecoratedProbe(Probe):
        ...


Registry Operations
-------------------

Check registration:

.. code-block:: python

    # Check if registered
    if model_registry.is_registered("openai"):
        model = model_registry.get("openai")

    # Using 'in' operator
    if "logic" in probe_registry:
        probe = probe_registry.get("logic")


Get factory without instantiating:

.. code-block:: python

    # Get the class/factory
    ModelClass = model_registry.get_factory("openai")

    # Create instance manually
    model = ModelClass(model_name="gpt-4")


Get detailed info:

.. code-block:: python

    info = model_registry.info("openai")
    print(info)
    # {
    #     'name': 'openai',
    #     'factory': 'OpenAIModel',
    #     'default_kwargs': {},
    #     'doc': 'Model implementation for OpenAI...'
    # }


Unregister items:

.. code-block:: python

    # Remove a registration
    model_registry.unregister("my_custom")

    # Clear all (careful!)
    model_registry.clear()


Creating Custom Registries
--------------------------

Create your own registries:

.. code-block:: python

    from insideLLMs import Registry

    # Type-safe registry
    evaluator_registry = Registry[Evaluator]("evaluators")

    # Register items
    evaluator_registry.register("accuracy", AccuracyEvaluator)
    evaluator_registry.register("bleu", BLEUEvaluator)


Resetting to Defaults
---------------------

Reset registries to built-in items:

.. code-block:: python

    from insideLLMs import ensure_builtins_registered

    # Re-register all built-in items
    ensure_builtins_registered()


Best Practices
--------------

1. **Use descriptive names**: ``"sentiment_v2"`` not ``"sv2"``

2. **Provide good defaults**: Register with sensible default arguments

3. **Document registrations**: Add docstrings to registered classes

4. **Handle missing items**: Check registration before getting

.. code-block:: python

    from insideLLMs.registry import NotFoundError

    try:
        model = model_registry.get("nonexistent")
    except NotFoundError as e:
        print(f"Available: {model_registry.list()}")
