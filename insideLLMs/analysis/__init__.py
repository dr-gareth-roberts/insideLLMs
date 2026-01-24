"""
Analysis, reporting, and export utilities for LLM experiments.

The ``insideLLMs.analysis`` package provides a comprehensive suite of tools
for analyzing, comparing, evaluating, and exporting results from LLM probing
experiments. It integrates statistical analysis, model comparison, evaluation
metrics, data export, and visualization capabilities into a unified interface.

Package Structure
-----------------
This package aggregates functionality from five specialized submodules:

**comparison**
    Model comparison utilities including multi-dimensional performance
    comparison, cost analysis, quality metrics, and comparison report
    generation. See :mod:`insideLLMs.analysis.comparison`.

**evaluation**
    Evaluation metrics and evaluators for LLM outputs including text
    similarity metrics, answer extraction, classification metrics,
    generation quality (BLEU, ROUGE), and LLM-as-a-Judge framework.
    See :mod:`insideLLMs.analysis.evaluation`.

**export**
    Data export and serialization utilities supporting JSON, JSONL, CSV,
    TSV, Markdown, YAML, Parquet, and Excel formats with streaming,
    compression, and schema validation. See :mod:`insideLLMs.analysis.export`.

**statistics**
    Statistical analysis tools including descriptive statistics, confidence
    intervals, significance testing (t-test, Mann-Whitney U), effect size
    calculations, bootstrap resampling, and power analysis.
    See :mod:`insideLLMs.analysis.statistics`.

**visualization**
    Visualization tools ranging from text-based ASCII charts to matplotlib
    static plots, Plotly interactive charts, HTML reports, and Jupyter
    widget explorers. See :mod:`insideLLMs.analysis.visualization`.


Available Classes
-----------------
Comparison Module
~~~~~~~~~~~~~~~~~
- ``ComparisonMetric`` : Enumeration of standard comparison metrics (ACCURACY,
    LATENCY, THROUGHPUT, COST, ERROR_RATE, TOKEN_USAGE, QUALITY_SCORE)
- ``MetricValue`` : Single metric measurement with timestamp and metadata
- ``MetricSummary`` : Summary statistics (mean, std, percentiles) for a metric
- ``ModelProfile`` : Complete performance profile for an LLM model
- ``ModelComparisonResult`` : Result of comparing multiple models
- ``ModelComparator`` : Main comparison engine for ranking and comparing models
- ``LatencyProfile`` : Detailed latency breakdown (first token, throughput)
- ``CostEstimate`` : Cost calculation for token usage
- ``ModelCostComparator`` : Compare costs across models with configurable pricing
- ``QualityMetrics`` : Quality scores (coherence, relevance, accuracy, etc.)
- ``PerformanceTracker`` : Track metrics during experiment execution

Evaluation Module
~~~~~~~~~~~~~~~~~
- ``EvaluationResult`` : Result from an evaluation metric (score, passed, details)
- ``MultiMetricResult`` : Aggregated results from multiple evaluation metrics
- ``Evaluator`` : Abstract base class for all evaluators
- ``ExactMatchEvaluator`` : Binary exact match evaluation
- ``ContainsEvaluator`` : Substring containment evaluation
- ``FuzzyMatchEvaluator`` : Levenshtein similarity evaluation
- ``TokenF1Evaluator`` : Token-level F1 score evaluation
- ``SemanticSimilarityEvaluator`` : Multi-metric semantic similarity
- ``NumericEvaluator`` : Numeric answer evaluation with tolerance
- ``MultipleChoiceEvaluator`` : Multiple choice answer evaluation
- ``CompositeEvaluator`` : Combine multiple evaluators with weights
- ``JudgeCriterion`` : Evaluation criterion for LLM-as-a-Judge
- ``JudgeResult`` : Result from LLM-as-a-Judge evaluation
- ``JudgeModel`` : LLM-as-a-Judge evaluator using a model
- ``JudgeEvaluator`` : Evaluator wrapper for JudgeModel

Export Module
~~~~~~~~~~~~~
- ``ExportFormat`` : Enumeration of supported export formats (JSON, JSONL, CSV,
    TSV, YAML, PARQUET, EXCEL, MARKDOWN)
- ``CompressionType`` : Enumeration of compression types (NONE, GZIP, ZIP, BZIP2)
- ``ExportConfig`` : Configuration for data export (format, compression, encoding)
- ``ExportMetadata`` : Metadata for exported data bundles
- ``Exporter`` : Abstract base class for exporters
- ``JSONExporter`` : Export to JSON format
- ``JSONLExporter`` : Export to JSON Lines format (streaming)
- ``CSVExporter`` : Export to CSV format
- ``MarkdownExporter`` : Export to Markdown tables
- ``DataArchiver`` : Create compressed archives
- ``SchemaField`` : Schema field definition
- ``DataSchema`` : Schema definition for validation
- ``ExportPipeline`` : Fluent pipeline for data transformation and export

Statistics Module
~~~~~~~~~~~~~~~~~
- ``ConfidenceInterval`` : Confidence interval with bounds and methods
- ``HypothesisTestResult`` : Result of statistical hypothesis testing
- ``DescriptiveStats`` : Comprehensive descriptive statistics
- ``AggregatedResults`` : Aggregated results with statistics and CI
- ``StatisticalComparisonResult`` : Full comparison of two groups

Visualization Module
~~~~~~~~~~~~~~~~~~~~
- ``ExperimentExplorer`` : Interactive Jupyter widget explorer (requires ipywidgets)


Available Functions
-------------------
Comparison Functions
~~~~~~~~~~~~~~~~~~~~
- ``create_comparison_table`` : Create markdown comparison table from profiles
- ``rank_models`` : Rank models by a specific metric

Evaluation Functions
~~~~~~~~~~~~~~~~~~~~
**Text Normalization and Extraction:**

- ``normalize_text`` : Normalize text (lowercase, punctuation, articles)
- ``extract_answer`` : Extract final answer from LLM response
- ``extract_number`` : Extract numeric value from text
- ``extract_choice`` : Extract multiple-choice answer

**Similarity Metrics:**

- ``exact_match`` : Binary exact match (0 or 1)
- ``contains_match`` : Check if reference contained in prediction
- ``levenshtein_distance`` : Calculate edit distance
- ``levenshtein_similarity`` : Normalized edit distance similarity (0-1)
- ``jaccard_similarity`` : Word set overlap (intersection/union)
- ``cosine_similarity_bow`` : Bag-of-words cosine similarity
- ``token_f1`` : Token-level F1 score (SQuAD-style)

**Generation Metrics:**

- ``get_ngrams`` : Extract n-grams from text
- ``bleu_score`` : BLEU score with brevity penalty
- ``rouge_l`` : ROUGE-L (longest common subsequence) F1 score

**Classification Metrics:**

- ``calculate_classification_metrics`` : Accuracy, precision, recall, F1

**Factory Functions:**

- ``evaluate_predictions`` : Batch evaluation with multiple metrics
- ``create_evaluator`` : Factory function for evaluators
- ``create_judge`` : Create JudgeModel with preset criteria

**Criterion Presets:**

- ``HELPFULNESS_CRITERIA`` : Criteria for helpfulness evaluation
- ``ACCURACY_CRITERIA`` : Criteria for accuracy evaluation
- ``SAFETY_CRITERIA`` : Criteria for safety evaluation
- ``CODE_QUALITY_CRITERIA`` : Criteria for code quality evaluation

Export Functions
~~~~~~~~~~~~~~~~
**Quick Export:**

- ``export_to_json`` : Quick export to JSON
- ``export_to_jsonl`` : Quick export to JSON Lines
- ``export_to_csv`` : Quick export to CSV
- ``export_to_markdown`` : Quick export to Markdown table

**Advanced Export:**

- ``serialize_value`` : Serialize a single value
- ``serialize_record`` : Serialize a record to dictionary
- ``get_exporter`` : Get exporter instance for format
- ``stream_export`` : Memory-efficient streaming export
- ``create_export_bundle`` : Create multi-format export bundle with metadata

Statistics Functions
~~~~~~~~~~~~~~~~~~~~
**Basic Statistics:**

- ``calculate_mean`` : Arithmetic mean
- ``calculate_variance`` : Sample variance
- ``calculate_std`` : Sample standard deviation
- ``calculate_median`` : Median value
- ``calculate_percentile`` : Calculate specific percentile
- ``calculate_skewness`` : Fisher-Pearson skewness
- ``calculate_kurtosis`` : Excess kurtosis
- ``descriptive_statistics`` : Comprehensive descriptive statistics

**Confidence Intervals:**

- ``confidence_interval`` : CI for the mean (t, z, or bootstrap)
- ``bootstrap_confidence_interval`` : Non-parametric bootstrap CI

**Hypothesis Testing:**

- ``welchs_t_test`` : Welch's t-test for independent samples
- ``paired_t_test`` : Paired samples t-test
- ``mann_whitney_u`` : Non-parametric rank-based test

**Effect Size:**

- ``cohens_d`` : Cohen's d effect size
- ``interpret_cohens_d`` : Interpret effect size magnitude

**Experiment Analysis:**

- ``extract_metric_from_results`` : Extract specific metric values
- ``extract_latencies`` : Extract all latency measurements
- ``extract_success_rates`` : Extract success rates
- ``aggregate_experiment_results`` : Aggregate with statistics and CI
- ``compare_experiments`` : Compare two experiment sets
- ``multiple_comparison_correction`` : Bonferroni, Holm, FDR correction
- ``generate_summary_report`` : Generate comprehensive summary

**Power Analysis:**

- ``power_analysis`` : Calculate statistical power
- ``required_sample_size`` : Compute sample size for desired power

Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~
**Text-Based (No Dependencies):**

- ``text_bar_chart`` : ASCII horizontal bar chart
- ``text_histogram`` : ASCII histogram
- ``text_comparison_table`` : ASCII comparison table
- ``text_summary_stats`` : Formatted statistics summary
- ``experiment_summary_text`` : Text summary of experiment result

**Matplotlib-Based (requires matplotlib, pandas):**

- ``plot_accuracy_comparison`` : Bar chart of model accuracies
- ``plot_latency_distribution`` : Histogram of latencies
- ``plot_metric_comparison`` : Multi-metric grouped bar chart
- ``plot_confusion_matrix`` : Confusion matrix heatmap
- ``plot_metric_over_time`` : Time series line plot
- ``plot_success_failure_pie`` : Pie chart of success/failure
- ``plot_model_radar`` : Radar/spider chart for multi-metric comparison
- ``plot_latency_boxplot`` : Box plot of latency distributions

**Plotly Interactive (requires plotly, pandas):**

- ``interactive_accuracy_comparison`` : Interactive bar chart
- ``interactive_latency_distribution`` : Interactive histogram
- ``interactive_metric_scatter`` : Interactive scatter plot
- ``interactive_model_comparison_table`` : Interactive data table
- ``interactive_timeline`` : Interactive time series
- ``interactive_heatmap`` : Interactive heatmap
- ``interactive_parallel_coordinates`` : Parallel coordinates plot
- ``interactive_sunburst`` : Hierarchical sunburst chart

**HTML Reports:**

- ``create_interactive_html_report`` : Self-contained HTML report

**Dependency Checks:**

- ``check_visualization_deps`` : Check matplotlib availability
- ``check_plotly_deps`` : Check Plotly availability
- ``check_ipywidgets_deps`` : Check ipywidgets availability

**Constants:**

- ``MATPLOTLIB_AVAILABLE`` : True if matplotlib/pandas installed
- ``SEABORN_AVAILABLE`` : True if seaborn installed
- ``PLOTLY_AVAILABLE`` : True if plotly installed
- ``IPYWIDGETS_AVAILABLE`` : True if ipywidgets/IPython installed


Examples
--------
Model Comparison Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~
Compare multiple LLM models across various metrics:

>>> from insideLLMs.analysis import (
...     ModelComparator, ModelProfile, create_comparison_table
... )
>>>
>>> # Create profiles for models
>>> gpt4 = ModelProfile(model_name="gpt-4", model_id="gpt-4-turbo")
>>> gpt4.add_metric("accuracy", [0.92, 0.94, 0.91, 0.93], unit="%")
>>> gpt4.add_metric("latency", [150.0, 145.0, 160.0, 155.0], unit="ms")
>>>
>>> claude = ModelProfile(model_name="claude-3", model_id="claude-3-opus")
>>> claude.add_metric("accuracy", [0.95, 0.93, 0.94, 0.96], unit="%")
>>> claude.add_metric("latency", [120.0, 115.0, 125.0, 118.0], unit="ms")
>>>
>>> # Compare models
>>> comparator = ModelComparator()
>>> comparator.add_profile(gpt4).add_profile(claude)  # doctest: +ELLIPSIS
<insideLLMs.analysis.comparison.ModelComparator object at ...>
>>> result = comparator.compare()
>>> print(f"Winner: {result.winner}")
Winner: claude-3
>>>
>>> # Generate comparison table
>>> table = create_comparison_table([gpt4, claude])
>>> print(table)  # doctest: +NORMALIZE_WHITESPACE
| Metric | gpt-4 | claude-3 |
| --- | ---: | ---: |
| accuracy | 0.925 | 0.945 |
| latency | 152.500 | 119.500 |

Text Evaluation
~~~~~~~~~~~~~~~
Evaluate model outputs against references:

>>> from insideLLMs.analysis import (
...     exact_match, token_f1, bleu_score, rouge_l,
...     ExactMatchEvaluator, TokenF1Evaluator
... )
>>>
>>> # Quick metric calculations
>>> prediction = "The capital of France is Paris"
>>> reference = "The capital of France is Paris"
>>> print(f"Exact: {exact_match(prediction, reference)}")
Exact: 1.0
>>>
>>> prediction = "Paris is the capital of France"
>>> reference = "The capital of France is Paris"
>>> print(f"Token F1: {token_f1(prediction, reference):.3f}")
Token F1: 1.000
>>>
>>> # Using evaluators for structured evaluation
>>> evaluator = ExactMatchEvaluator(normalize=True)
>>> result = evaluator.evaluate("Hello World!", "hello world")
>>> print(f"Score: {result.score}, Passed: {result.passed}")
Score: 1.0, Passed: True

LLM-as-a-Judge Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~
Use an LLM to evaluate open-ended responses:

>>> from insideLLMs.analysis import (
...     create_judge, ACCURACY_CRITERIA, JudgeCriterion
... )
>>> from insideLLMs.models import OpenAIModel  # doctest: +SKIP
>>>
>>> # Create judge with preset criteria
>>> judge = create_judge(
...     OpenAIModel(model_name="gpt-4"),
...     criteria_preset="accuracy"
... )  # doctest: +SKIP
>>>
>>> # Evaluate a response
>>> result = judge.evaluate(
...     prompt="Explain photosynthesis",
...     response="Photosynthesis is the process by which plants convert sunlight..."
... )  # doctest: +SKIP
>>> print(f"Score: {result.overall_score:.2f}")  # doctest: +SKIP

Data Export
~~~~~~~~~~~
Export experiment results to various formats:

>>> from insideLLMs.analysis import (
...     export_to_json, export_to_csv, export_to_jsonl,
...     ExportPipeline, ExportFormat, create_export_bundle
... )
>>>
>>> # Sample data
>>> data = [
...     {"model": "gpt-4", "accuracy": 0.92, "latency_ms": 150},
...     {"model": "claude-3", "accuracy": 0.95, "latency_ms": 120},
... ]
>>>
>>> # Quick exports
>>> export_to_json(data, "/tmp/results.json")  # doctest: +SKIP
>>> export_to_csv(data, "/tmp/results.csv")  # doctest: +SKIP
>>>
>>> # Pipeline with transformations
>>> pipeline = (
...     ExportPipeline()
...     .filter(lambda x: x.get("accuracy", 0) > 0.90)
...     .select(["model", "accuracy"])
...     .sort("accuracy", reverse=True)
... )
>>> processed = pipeline.execute(data)
>>> len(processed)
2
>>>
>>> # Create export bundle (multiple formats + metadata)
>>> bundle = create_export_bundle(
...     data,
...     output_dir="/tmp/exports",
...     name="experiment_results",
...     formats=[ExportFormat.JSON, ExportFormat.CSV],
...     compress=True
... )  # doctest: +SKIP

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~
Perform statistical analysis on experiment results:

>>> from insideLLMs.analysis import (
...     descriptive_statistics, confidence_interval,
...     welchs_t_test, cohens_d, interpret_cohens_d
... )
>>>
>>> # Descriptive statistics
>>> values = [0.85, 0.88, 0.92, 0.87, 0.90, 0.89]
>>> stats = descriptive_statistics(values)
>>> print(f"Mean: {stats.mean:.3f}, Std: {stats.std:.3f}")
Mean: 0.885, Std: 0.024
>>>
>>> # Confidence interval
>>> ci = confidence_interval(values, confidence_level=0.95)
>>> print(f"95% CI: [{ci.lower:.3f}, {ci.upper:.3f}]")
95% CI: [0.860, 0.910]
>>>
>>> # Compare two groups
>>> group_a = [0.90, 0.92, 0.88, 0.91, 0.89]
>>> group_b = [0.85, 0.83, 0.86, 0.84, 0.82]
>>> result = welchs_t_test(group_a, group_b)
>>> print(f"Significant: {result.significant}, p={result.p_value:.4f}")
Significant: True, p=0.0002
>>>
>>> # Effect size
>>> d = cohens_d(group_a, group_b)
>>> print(f"Cohen's d: {d:.2f} ({interpret_cohens_d(d)})")
Cohen's d: 3.53 (large)

Visualization
~~~~~~~~~~~~~
Create visualizations for analysis results:

>>> from insideLLMs.analysis import (
...     text_bar_chart, text_histogram, experiment_summary_text
... )
>>>
>>> # Text-based bar chart (no dependencies)
>>> labels = ["GPT-4", "Claude-3", "Gemini"]
>>> values = [0.92, 0.89, 0.87]
>>> print(text_bar_chart(labels, values, title="Model Accuracy"))
Model Accuracy
==============
<BLANKLINE>
GPT-4    | ████████████████████████████████████████████████ 0.92
Claude-3 | ████████████████████████████████████████████ 0.89
Gemini   | ██████████████████████████████████████████ 0.87
>>>
>>> # Text histogram for latency distribution
>>> latencies = [100, 120, 95, 150, 110, 130, 105, 115, 140, 125]
>>> print(text_histogram(latencies, bins=5, title="Latency Distribution"))
Latency Distribution
====================
<BLANKLINE>
[ 95.00 - 106.00] | ████████████████████████████████████████ (3)
[106.00 - 117.00] | █████████████████████████████████████████████████████ (4)
[117.00 - 128.00] | █████████████████████████ (2)
[128.00 - 139.00] | █████████████ (1)
[139.00 - 150.00] | █████████████ (1)

For matplotlib-based and interactive Plotly visualizations:

>>> from insideLLMs.analysis import (
...     plot_accuracy_comparison,
...     interactive_accuracy_comparison,
...     create_interactive_html_report
... )  # doctest: +SKIP
>>>
>>> # Matplotlib static plot
>>> plot_accuracy_comparison(
...     experiments,
...     title="Model Comparison",
...     save_path="comparison.png"
... )  # doctest: +SKIP
>>>
>>> # Plotly interactive chart
>>> fig = interactive_accuracy_comparison(experiments)  # doctest: +SKIP
>>> fig.write_html("interactive_chart.html")  # doctest: +SKIP
>>>
>>> # Full HTML report
>>> create_interactive_html_report(
...     experiments,
...     title="Evaluation Report",
...     save_path="report.html"
... )  # doctest: +SKIP


Cost Comparison
~~~~~~~~~~~~~~~
Compare API costs across models:

>>> from insideLLMs.analysis import ModelCostComparator
>>>
>>> calc = ModelCostComparator()
>>> # Default pricing for common models is included
>>> cost = calc.estimate("gpt-4", input_tokens=1000, output_tokens=500)
>>> print(f"GPT-4 cost: ${cost.total_cost:.4f}")
GPT-4 cost: $0.0600
>>>
>>> # Add custom model pricing
>>> calc.set_pricing("custom-model", input_per_1k=0.001, output_per_1k=0.002)
... # doctest: +ELLIPSIS
<insideLLMs.analysis.comparison.ModelCostComparator object at ...>
>>>
>>> # Find cheapest model
>>> model, cost = calc.cheapest_model(1000, 500)
>>> print(f"Cheapest: {model}")  # doctest: +SKIP


Notes
-----
Dependencies
~~~~~~~~~~~~
The analysis package has tiered dependencies:

- **Core functionality** (comparison, evaluation, statistics, export): No
  external dependencies beyond the standard library.

- **Matplotlib visualizations**: Requires ``matplotlib`` and ``pandas``.
  Install with: ``pip install matplotlib pandas``

- **Seaborn styling**: Requires ``seaborn`` for enhanced plot aesthetics.
  Install with: ``pip install seaborn``

- **Interactive Plotly charts**: Requires ``plotly``.
  Install with: ``pip install plotly``

- **Jupyter widgets**: Requires ``ipywidgets`` and ``IPython``.
  Install with: ``pip install ipywidgets``

- **Optional export formats**: Parquet requires ``pyarrow``, Excel requires
  ``openpyxl``, YAML requires ``PyYAML``.

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~
- For large datasets (>1M records), use ``stream_export`` with JSONL format
  to avoid memory issues.

- Statistical functions work with Python lists but convert to numpy arrays
  internally when scipy is available for better performance.

- Visualization functions cache computed data where possible to improve
  rendering performance for interactive charts.


See Also
--------
insideLLMs.types : Core data types including ExperimentResult
insideLLMs.probes : Probe definitions and execution
insideLLMs.models : Model interfaces and wrappers
insideLLMs.schemas : Schema definitions and validation
"""

from insideLLMs.analysis.comparison import *  # noqa: F401,F403
from insideLLMs.analysis.evaluation import *  # noqa: F401,F403
from insideLLMs.analysis.export import *  # noqa: F401,F403
from insideLLMs.analysis.statistics import *  # noqa: F401,F403
from insideLLMs.analysis.visualization import *  # noqa: F401,F403
