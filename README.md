# insideLLMs

Python library for probing the inner workings of large language models. Systematically test LLMs' zero-shot ability at unseen logic problems, propensity for bias, vulnerabilities to particular attacks, factual accuracy, and more.

## Features

- **Multiple Model Support**: OpenAI, HuggingFace, Anthropic Claude, and more
- **Diverse Probes**: Test LLMs on logic, bias, attack vulnerabilities, and factual accuracy
- **Visualization Tools**: Visualize probe results with matplotlib
- **Benchmarking**: Compare multiple models on the same tasks
- **Configurable**: Run experiments with YAML/JSON configuration
- **NLP Utilities**: Text processing, tokenization, feature extraction, NER, and more

## Installation

To install `insideLLMs` from a local clone of this repository, navigate to the root directory of the clone and run:

```bash
# Install the base package
pip install .

# To include NLP utilities (requires NLTK, spaCy, scikit-learn, gensim):
pip install .[nlp]

# To include visualization tools (requires matplotlib, pandas, seaborn):
pip install .[visualization]

# To include all optional dependencies:
pip install .[nlp,visualization]
```
See `pyproject.toml` for the list of dependencies. For development, see the "Development" section below.

## Quick Start

```python
from insideLLMs.models import OpenAIModel
from insideLLMs.probes import LogicProbe

# Initialize a model
model = OpenAIModel(model_name="gpt-3.5-turbo")

# Create a probe
probe = LogicProbe()

# Run the probe
result = probe.run(model, "If all A are B, and all B are C, then are all A also C?")
print(result)
```

## Available Models

- `OpenAIModel`: For OpenAI's GPT models
- `HuggingFaceModel`: For HuggingFace Transformers models
- `AnthropicModel`: For Anthropic's Claude models
- `DummyModel`: For testing purposes

## Available Probes

- `LogicProbe`: Tests LLMs' ability to solve logic problems
- `BiasProbe`: Tests LLMs' propensity for bias
- `AttackProbe`: Tests LLMs' vulnerabilities to attacks
- `FactualityProbe`: Tests LLMs' factual accuracy

## Visualization and Benchmarking

```python
from insideLLMs.models import OpenAIModel, AnthropicModel
from insideLLMs.probes import FactualityProbe
from insideLLMs.benchmark import ModelBenchmark
from insideLLMs.visualization import plot_factuality_results

# Create models and probe
models = [OpenAIModel(), AnthropicModel()]
probe = FactualityProbe()

# Run benchmark
benchmark = ModelBenchmark(models, probe)
results = benchmark.run(factual_questions)

# Compare models
comparison = benchmark.compare_models()
print(f"Fastest model: {comparison['rankings']['total_time'][0]}")

# Visualize results
plot_factuality_results(results["models"][0]["results"])
```

## NLP Utilities

The library includes a comprehensive set of NLP utilities for text processing and analysis:

### Text Cleaning and Normalization
```python
from insideLLMs.nlp_utils import clean_text, remove_html_tags, remove_urls

# Clean text with multiple options
clean_text = clean_text(text, remove_html=True, remove_url=True, lowercase=True)
```

### Tokenization and Segmentation
```python
from insideLLMs.nlp_utils import simple_tokenize, nltk_tokenize, segment_sentences

# Get tokens and sentences
tokens = simple_tokenize(text)  # No dependencies
tokens = nltk_tokenize(text)    # Requires NLTK
sentences = segment_sentences(text)
```

### Feature Extraction
```python
from insideLLMs.nlp_utils import create_bow, create_tfidf, extract_pos_tags

# Create document representations
bow_matrix, feature_names = create_bow(texts)  # Bag of Words
tfidf_matrix, feature_names = create_tfidf(texts)  # TF-IDF
```

### Text Statistics and Metrics
```python
from insideLLMs.nlp_utils import count_words, calculate_lexical_diversity

# Get text statistics
word_count = count_words(text)
diversity = calculate_lexical_diversity(text)  # Type-token ratio
```

### Named Entity Recognition
```python
from insideLLMs.nlp_utils import extract_named_entities

# Extract entities
entities = extract_named_entities(text)  # Requires spaCy
```

### Keyword Extraction
```python
from insideLLMs.nlp_utils import extract_keywords_tfidf

# Extract keywords
keywords = extract_keywords_tfidf(text, num_keywords=5)
```

### Text Similarity
```python
from insideLLMs.nlp_utils import cosine_similarity_texts, jaccard_similarity

# Calculate similarity between texts
similarity = cosine_similarity_texts(text1, text2)  # Requires scikit-learn
similarity = jaccard_similarity(text1, text2)  # No dependencies
```

See `examples/example_nlp.py` for a complete demonstration of all NLP utilities.

## Development

To set up a development environment, clone the repository and install the development dependencies. Replace the URL with the actual repository URL.

```bash
git clone https://github.com/example/insideLLMs
cd insideLLMs
pip install -r requirements-dev.txt
```

### Running Tests

Tests are located in the `tests/` directory and can be run using pytest:

```bash
pytest
```