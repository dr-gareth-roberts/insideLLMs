Installation
============

Requirements
------------

* Python 3.10 or higher
* pip or uv package manager


Basic Installation
------------------

Install from PyPI:

.. code-block:: bash

    pip install insideLLMs


With Optional Dependencies
--------------------------

NLP utilities (nltk, spacy, etc.):

.. code-block:: bash

    pip install insideLLMs[nlp]

Visualization tools (matplotlib, pandas, seaborn):

.. code-block:: bash

    pip install insideLLMs[visualization]

Development tools:

.. code-block:: bash

    pip install insideLLMs[dev]

Everything:

.. code-block:: bash

    pip install insideLLMs[all]


Using uv
--------

For faster installation with uv:

.. code-block:: bash

    uv pip install insideLLMs


Development Installation
------------------------

Clone and install in development mode:

.. code-block:: bash

    git clone https://github.com/dr-gareth-roberts/insideLLMs
    cd insideLLMs
    pip install -e .[dev]


Environment Setup
-----------------

Set up API keys for the models you want to use:

.. code-block:: bash

    # OpenAI
    export OPENAI_API_KEY="your-openai-key"

    # Anthropic
    export ANTHROPIC_API_KEY="your-anthropic-key"

    # HuggingFace Hub (optional, for private models)
    export HUGGINGFACEHUB_API_TOKEN="your-hf-token"


Verifying Installation
----------------------

Verify the installation:

.. code-block:: python

    import insideLLMs
    print(insideLLMs.__version__)

    # Test with DummyModel (no API key needed)
    from insideLLMs import DummyModel
    model = DummyModel()
    print(model.generate("Hello!"))
