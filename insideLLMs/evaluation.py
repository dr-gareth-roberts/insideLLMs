"""Compatibility shim for insideLLMs.analysis.evaluation.

This module provides a backwards-compatible import path for evaluation utilities.
All functionality is implemented in :mod:`insideLLMs.analysis.evaluation` and
re-exported here for convenience.

Module Overview
---------------
The evaluation module is designed for assessing the quality of LLM-generated
outputs against reference answers or using LLM-based judgments. It provides
both simple metrics (exact match, contains) and sophisticated measures
(semantic similarity, BLEU, ROUGE-L), as well as an LLM-as-a-Judge framework.

This shim ensures that existing code using ``from insideLLMs.evaluation import ...``
continues to work after the module reorganization. New code should prefer
importing from :mod:`insideLLMs.analysis.evaluation` directly.

Key Components
--------------
**Data Classes**

- :class:`EvaluationResult` : Standard result container for evaluation metrics
- :class:`MultiMetricResult` : Aggregated results from multiple metrics
- :class:`JudgeCriterion` : Configuration for LLM-as-a-Judge criteria
- :class:`JudgeResult` : Result from LLM-as-a-Judge evaluation

**Text Utilities**

- :func:`normalize_text` : Normalize text (lowercase, remove punctuation, etc.)
- :func:`extract_answer` : Extract final answer from LLM response
- :func:`extract_number` : Extract numeric value from text
- :func:`extract_choice` : Extract multiple-choice answer (A, B, C, D)

**Similarity Metrics**

- :func:`exact_match` : Binary exact match (0.0 or 1.0)
- :func:`contains_match` : Check if reference is substring of prediction
- :func:`levenshtein_distance` : Edit distance between strings
- :func:`levenshtein_similarity` : Normalized edit-distance similarity (0-1)
- :func:`jaccard_similarity` : Word-set overlap (intersection/union)
- :func:`cosine_similarity_bow` : Bag-of-words cosine similarity
- :func:`token_f1` : Token-level F1 score (SQuAD-style)

**Generation Metrics**

- :func:`bleu_score` : Approximate BLEU score with smoothing
- :func:`rouge_l` : ROUGE-L (longest common subsequence F1)
- :func:`get_ngrams` : Extract n-grams from text

**Classification Metrics**

- :func:`calculate_classification_metrics` : Accuracy, precision, recall, F1

**Evaluator Classes**

- :class:`Evaluator` : Abstract base class for evaluators
- :class:`ExactMatchEvaluator` : Exact string match evaluator
- :class:`ContainsEvaluator` : Substring containment evaluator
- :class:`FuzzyMatchEvaluator` : Levenshtein similarity evaluator
- :class:`TokenF1Evaluator` : Token-level F1 evaluator
- :class:`SemanticSimilarityEvaluator` : Weighted multi-metric evaluator
- :class:`NumericEvaluator` : Numeric answer evaluator with tolerance
- :class:`MultipleChoiceEvaluator` : Multiple choice answer evaluator
- :class:`CompositeEvaluator` : Combines multiple evaluators

**LLM-as-a-Judge**

- :class:`JudgeModel` : Main LLM-as-a-Judge evaluator
- :class:`JudgeEvaluator` : Evaluator wrapper for JudgeModel
- :func:`create_judge` : Factory function for JudgeModel with presets

**Criterion Presets**

- ``HELPFULNESS_CRITERIA`` : Helpfulness, completeness, clarity
- ``ACCURACY_CRITERIA`` : Factual accuracy, logical consistency, source alignment
- ``SAFETY_CRITERIA`` : Harmlessness, bias-free, appropriate refusal
- ``CODE_QUALITY_CRITERIA`` : Correctness, efficiency, readability, best practices

**Convenience Functions**

- :func:`evaluate_predictions` : Batch evaluate with multiple metrics
- :func:`create_evaluator` : Factory function for evaluator classes

Quick Start Examples
--------------------
Basic text comparison with similarity metrics:

    >>> from insideLLMs.evaluation import exact_match, token_f1, jaccard_similarity
    >>> # Check if prediction matches reference exactly (normalized)
    >>> exact_match("The Answer Is 42!", "the answer is 42")
    1.0

    >>> # Token-level F1 score for partial matches
    >>> token_f1("The quick brown fox", "The fast brown fox")
    0.75

    >>> # Word-set overlap similarity
    >>> jaccard_similarity("the cat sat on mat", "the dog sat on mat")
    0.6666666666666666

Using evaluator classes for structured evaluation:

    >>> from insideLLMs.evaluation import ExactMatchEvaluator, TokenF1Evaluator
    >>> # Create evaluator with normalization
    >>> evaluator = ExactMatchEvaluator(normalize=True)
    >>> result = evaluator.evaluate("Hello World!", "hello world")
    >>> print(f"Score: {result.score}, Passed: {result.passed}")
    Score: 1.0, Passed: True

    >>> # Token F1 with custom threshold
    >>> f1_eval = TokenF1Evaluator(threshold=0.6)
    >>> result = f1_eval.evaluate("the quick brown fox", "a quick red fox")
    >>> print(f"Score: {result.score:.2f}, Passed: {result.passed}")
    Score: 0.50, Passed: False

Evaluating numeric answers with tolerance:

    >>> from insideLLMs.evaluation import NumericEvaluator
    >>> # 5% relative tolerance
    >>> evaluator = NumericEvaluator(tolerance=0.05, relative=True)
    >>> result = evaluator.evaluate("The result is 3.14", "3.1416")
    >>> print(f"Passed: {result.passed}, Diff: {result.details['difference']:.4f}")
    Passed: True, Diff: 0.0005

Multiple choice evaluation:

    >>> from insideLLMs.evaluation import MultipleChoiceEvaluator
    >>> evaluator = MultipleChoiceEvaluator(choices=["A", "B", "C", "D"])
    >>> result = evaluator.evaluate("I think the answer is B", "B")
    >>> print(f"Score: {result.score}, Predicted: {result.details['predicted']}")
    Score: 1.0, Predicted: B

Combining multiple evaluators:

    >>> from insideLLMs.evaluation import (
    ...     CompositeEvaluator, ExactMatchEvaluator, TokenF1Evaluator
    ... )
    >>> composite = CompositeEvaluator(
    ...     evaluators=[ExactMatchEvaluator(), TokenF1Evaluator()],
    ...     weights=[0.3, 0.7],
    ...     require_all=False
    ... )
    >>> result = composite.evaluate("the quick fox", "the fast fox")
    >>> print(f"Score: {result.score:.2f}")
    Score: 0.52

Computing BLEU and ROUGE-L for generation tasks:

    >>> from insideLLMs.evaluation import bleu_score, rouge_l
    >>> prediction = "The cat sat on the mat"
    >>> reference = "The cat is sitting on the mat"
    >>> print(f"BLEU: {bleu_score(prediction, reference):.4f}")
    BLEU: 0.4353
    >>> print(f"ROUGE-L: {rouge_l(prediction, reference):.4f}")
    ROUGE-L: 0.7692

Extracting answers from verbose LLM responses:

    >>> from insideLLMs.evaluation import extract_answer, extract_number
    >>> response = "Let me think step by step... The answer is 42."
    >>> extract_answer(response)
    '42'

    >>> response = "After calculation, the result is approximately 3.14159"
    >>> extract_number(response)
    3.14159

Batch evaluation with aggregated metrics:

    >>> from insideLLMs.evaluation import evaluate_predictions
    >>> predictions = ["paris", "berlin", "madrid"]
    >>> references = ["Paris", "Berlin", "Rome"]
    >>> results = evaluate_predictions(
    ...     predictions, references,
    ...     metrics=["exact_match", "token_f1"]
    ... )
    >>> print(f"Exact match: {results['aggregated']['exact_match']:.2f}")
    Exact match: 0.67
    >>> print(f"Token F1: {results['aggregated']['token_f1']:.2f}")
    Token F1: 0.67

Using the evaluator factory:

    >>> from insideLLMs.evaluation import create_evaluator
    >>> # Create a fuzzy match evaluator with 0.7 threshold
    >>> evaluator = create_evaluator("fuzzy", threshold=0.7)
    >>> result = evaluator.evaluate("hello world", "hallo world")
    >>> print(f"Score: {result.score:.2f}, Passed: {result.passed}")
    Score: 0.91, Passed: True

Using LLM-as-a-Judge for open-ended evaluation:

    >>> from insideLLMs.evaluation import create_judge, ACCURACY_CRITERIA
    >>> from insideLLMs.models import OpenAIModel  # doctest: +SKIP
    >>> # Create judge with accuracy criteria preset
    >>> judge = create_judge(
    ...     OpenAIModel(model_name="gpt-4"),
    ...     criteria_preset="accuracy",
    ...     threshold=0.7
    ... )  # doctest: +SKIP
    >>> result = judge.evaluate(
    ...     prompt="What is the capital of France?",
    ...     response="The capital of France is Paris.",
    ...     reference="Paris"
    ... )  # doctest: +SKIP
    >>> print(f"Score: {result.overall_score:.2f}")  # doctest: +SKIP
    Score: 0.95

Pairwise comparison of responses:

    >>> from insideLLMs.evaluation import create_judge
    >>> from insideLLMs.models import OpenAIModel  # doctest: +SKIP
    >>> judge = create_judge(OpenAIModel(model_name="gpt-4"))  # doctest: +SKIP
    >>> comparison = judge.compare(
    ...     prompt="Explain photosynthesis briefly",
    ...     response_a="Plants use sunlight to make food.",
    ...     response_b="Photosynthesis is a complex biochemical process..."
    ... )  # doctest: +SKIP
    >>> print(f"Winner: {comparison['winner']}")  # doctest: +SKIP
    Winner: B

Custom criteria for LLM-as-a-Judge:

    >>> from insideLLMs.evaluation import JudgeCriterion, JudgeModel
    >>> from insideLLMs.models import OpenAIModel  # doctest: +SKIP
    >>> custom_criteria = [
    ...     JudgeCriterion(
    ...         name="technical_accuracy",
    ...         description="Are the technical details correct?",
    ...         weight=1.0,
    ...         scale_min=1,
    ...         scale_max=5
    ...     ),
    ...     JudgeCriterion(
    ...         name="code_quality",
    ...         description="Is the code clean and well-structured?",
    ...         weight=0.8,
    ...         scale_min=1,
    ...         scale_max=5
    ...     ),
    ... ]
    >>> judge = JudgeModel(
    ...     judge_model=OpenAIModel(model_name="gpt-4"),
    ...     criteria=custom_criteria,
    ...     threshold=0.7
    ... )  # doctest: +SKIP

Semantic similarity with custom weights:

    >>> from insideLLMs.evaluation import SemanticSimilarityEvaluator
    >>> evaluator = SemanticSimilarityEvaluator(
    ...     threshold=0.5,
    ...     weights={"jaccard": 0.2, "cosine": 0.5, "token_f1": 0.3}
    ... )
    >>> result = evaluator.evaluate(
    ...     "machine learning is powerful",
    ...     "deep learning is effective"
    ... )
    >>> print(f"Score: {result.score:.2f}")
    Score: 0.23
    >>> print(f"Component scores: {result.details['component_scores']}")
    Component scores: {'jaccard': 0.14285714285714285, 'cosine': 0.25, 'token_f1': 0.25}

Classification metrics for multi-class problems:

    >>> from insideLLMs.evaluation import calculate_classification_metrics
    >>> predictions = ["cat", "dog", "cat", "bird", "dog"]
    >>> references = ["cat", "dog", "dog", "bird", "cat"]
    >>> metrics = calculate_classification_metrics(predictions, references)
    >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
    Accuracy: 0.60
    >>> print(f"Macro F1: {metrics['f1']:.2f}")
    Macro F1: 0.53

Notes
-----
**Migration Guide**

If you are migrating to the new module structure, update your imports:

Old (still works via this shim):
    ``from insideLLMs.evaluation import exact_match``

New (preferred):
    ``from insideLLMs.analysis.evaluation import exact_match``

**Performance Considerations**

- Levenshtein distance has O(m*n) complexity; avoid on very long strings
- LLM-as-a-Judge requires API calls and may have latency/cost implications
- For batch evaluation, consider using ``evaluate_batch()`` methods
- BLEU and ROUGE-L use dynamic programming and scale with text length

**Metric Selection Guide**

- **Exact Match**: Use for factoid QA, single-word answers
- **Token F1**: Use for extractive QA (SQuAD-style)
- **Contains**: Use when answer may be embedded in explanation
- **Fuzzy Match**: Use when typos/variations are expected
- **BLEU/ROUGE-L**: Use for generation tasks (summarization, translation)
- **Semantic Similarity**: Use for paraphrase detection, semantic equivalence
- **LLM-as-a-Judge**: Use for open-ended evaluation, subjective quality

**Threshold Guidelines**

Default thresholds are set conservatively:

- Exact match: 1.0 (only exact matches pass)
- Token F1: 0.5 (reasonable for partial matches)
- Fuzzy match: 0.8 (high similarity required)
- Semantic similarity: 0.6 (moderate similarity)
- Numeric: tolerance-based (1.0 - tolerance)

Adjust thresholds based on your task requirements and quality bar.

See Also
--------
insideLLMs.analysis.evaluation : Primary implementation module
insideLLMs.nlp.similarity : Lower-level similarity functions
insideLLMs.nlp.tokenization : Tokenization utilities

References
----------
.. [1] Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine
       Translation", ACL 2002
.. [2] Lin, "ROUGE: A Package for Automatic Evaluation of Summaries",
       Text Summarization Branches Out, 2004
.. [3] Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension
       of Text", EMNLP 2016 (for Token F1)
.. [4] Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
       NeurIPS 2023
"""

from insideLLMs.analysis.evaluation import *  # noqa: F401,F403
