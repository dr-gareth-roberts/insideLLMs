"""
Prompt testing and experimentation utilities.

Provides tools for systematic prompt engineering:
- A/B testing of prompts
- Prompt variation generation
- Evaluation and scoring
- Experiment tracking
"""

import itertools
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Optional,
)

from insideLLMs.nlp.tokenization import word_tokenize_regex


class PromptStrategy(Enum):
    """Common prompt engineering strategies."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STEP_BY_STEP = "step_by_step"
    ROLE_PLAY = "role_play"
    SOCRATIC = "socratic"
    TREE_OF_THOUGHT = "tree_of_thought"


@dataclass
class PromptVariant:
    """A single prompt variant for testing."""

    id: str
    content: str
    strategy: Optional[PromptStrategy] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash(self.id)


@dataclass
class PromptTestResult:
    """Result from testing a single prompt variant."""

    variant_id: str
    response: str
    score: float
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PromptExperimentResult:
    """Results from a prompt testing experiment."""

    experiment_id: str
    variants: list[PromptVariant]
    results: list[PromptTestResult]
    best_variant_id: Optional[str] = None
    summary: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def get_variant_scores(self) -> dict[str, list[float]]:
        """Get scores grouped by variant ID."""
        scores: dict[str, list[float]] = {}
        for result in self.results:
            if result.variant_id not in scores:
                scores[result.variant_id] = []
            scores[result.variant_id].append(result.score)
        return scores

    def get_average_scores(self) -> dict[str, float]:
        """Get average score per variant."""
        scores = self.get_variant_scores()
        return {vid: sum(s) / len(s) if s else 0.0 for vid, s in scores.items()}

    def get_best_variant(self) -> Optional[str]:
        """Get the variant with the highest average score."""
        averages = self.get_average_scores()
        if not averages:
            return None
        return max(averages, key=averages.get)


class PromptVariationGenerator:
    """Generate variations of a prompt for testing."""

    # Common system role prefixes
    ROLE_PREFIXES = [
        "You are an expert",
        "You are a helpful assistant",
        "You are a knowledgeable",
        "Act as an expert",
        "As a professional",
    ]

    # Instruction modifiers
    INSTRUCTION_MODIFIERS = [
        "Please",
        "Kindly",
        "Could you",
        "I need you to",
        "",  # No modifier
    ]

    # Output format instructions
    OUTPUT_FORMATS = [
        "Provide a clear and concise answer.",
        "Be thorough in your response.",
        "Keep your response brief.",
        "Explain step by step.",
        "Give a direct answer.",
    ]

    def __init__(self, base_prompt: str):
        """Initialize with a base prompt.

        Args:
            base_prompt: The original prompt to generate variations from.
        """
        self.base_prompt = base_prompt
        self._variations: list[PromptVariant] = []

    def add_prefix_variations(
        self, prefixes: Optional[list[str]] = None
    ) -> "PromptVariationGenerator":
        """Add variations with different prefixes.

        Args:
            prefixes: Custom prefixes, or use defaults.

        Returns:
            Self for chaining.
        """
        prefixes = prefixes or self.ROLE_PREFIXES
        for i, prefix in enumerate(prefixes):
            variant = PromptVariant(
                id=f"prefix_{i}",
                content=f"{prefix}\n\n{self.base_prompt}",
                metadata={"variation_type": "prefix", "prefix": prefix},
            )
            self._variations.append(variant)
        return self

    def add_suffix_variations(
        self, suffixes: Optional[list[str]] = None
    ) -> "PromptVariationGenerator":
        """Add variations with different suffixes/output formats.

        Args:
            suffixes: Custom suffixes, or use defaults.

        Returns:
            Self for chaining.
        """
        suffixes = suffixes or self.OUTPUT_FORMATS
        for i, suffix in enumerate(suffixes):
            variant = PromptVariant(
                id=f"suffix_{i}",
                content=f"{self.base_prompt}\n\n{suffix}",
                metadata={"variation_type": "suffix", "suffix": suffix},
            )
            self._variations.append(variant)
        return self

    def add_instruction_variations(
        self, modifiers: Optional[list[str]] = None
    ) -> "PromptVariationGenerator":
        """Add variations with different instruction modifiers.

        Args:
            modifiers: Custom modifiers, or use defaults.

        Returns:
            Self for chaining.
        """
        modifiers = modifiers or self.INSTRUCTION_MODIFIERS
        for i, modifier in enumerate(modifiers):
            content = f"{modifier} {self.base_prompt}" if modifier else self.base_prompt
            variant = PromptVariant(
                id=f"modifier_{i}",
                content=content,
                metadata={"variation_type": "modifier", "modifier": modifier},
            )
            self._variations.append(variant)
        return self

    def add_strategy_variations(self) -> "PromptVariationGenerator":
        """Add variations using different prompting strategies.

        Returns:
            Self for chaining.
        """
        # Zero-shot (base)
        self._variations.append(
            PromptVariant(
                id="strategy_zero_shot",
                content=self.base_prompt,
                strategy=PromptStrategy.ZERO_SHOT,
                metadata={"variation_type": "strategy"},
            )
        )

        # Chain of thought
        cot_prompt = f"{self.base_prompt}\n\nLet's think through this step by step:"
        self._variations.append(
            PromptVariant(
                id="strategy_cot",
                content=cot_prompt,
                strategy=PromptStrategy.CHAIN_OF_THOUGHT,
                metadata={"variation_type": "strategy"},
            )
        )

        # Step by step
        step_prompt = f"{self.base_prompt}\n\nPlease solve this step by step, showing your work."
        self._variations.append(
            PromptVariant(
                id="strategy_step",
                content=step_prompt,
                strategy=PromptStrategy.STEP_BY_STEP,
                metadata={"variation_type": "strategy"},
            )
        )

        return self

    def add_custom_variation(
        self,
        variant_id: str,
        content: str,
        strategy: Optional[PromptStrategy] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "PromptVariationGenerator":
        """Add a custom variation.

        Args:
            variant_id: Unique identifier for this variant.
            content: The prompt content.
            strategy: Optional prompting strategy.
            metadata: Optional metadata.

        Returns:
            Self for chaining.
        """
        self._variations.append(
            PromptVariant(
                id=variant_id,
                content=content,
                strategy=strategy,
                metadata=metadata or {},
            )
        )
        return self

    def add_temperature_hint_variations(self) -> "PromptVariationGenerator":
        """Add variations suggesting different 'temperatures' through wording.

        Returns:
            Self for chaining.
        """
        # Creative/high temperature hint
        creative = f"{self.base_prompt}\n\nBe creative and explore different possibilities."
        self._variations.append(
            PromptVariant(
                id="temp_creative",
                content=creative,
                metadata={"variation_type": "temperature_hint", "hint": "creative"},
            )
        )

        # Precise/low temperature hint
        precise = f"{self.base_prompt}\n\nBe precise and give the most accurate answer."
        self._variations.append(
            PromptVariant(
                id="temp_precise",
                content=precise,
                metadata={"variation_type": "temperature_hint", "hint": "precise"},
            )
        )

        return self

    def generate(self) -> list[PromptVariant]:
        """Generate all variations.

        Returns:
            List of prompt variants.
        """
        # Always include base prompt if no variations
        if not self._variations:
            return [PromptVariant(id="base", content=self.base_prompt)]
        return self._variations

    def generate_combinations(
        self,
        prefixes: Optional[list[str]] = None,
        suffixes: Optional[list[str]] = None,
        max_combinations: int = 20,
    ) -> list[PromptVariant]:
        """Generate combinations of prefix and suffix variations.

        Args:
            prefixes: Prefix options.
            suffixes: Suffix options.
            max_combinations: Maximum number of combinations.

        Returns:
            List of combined variants.
        """
        prefixes = prefixes or ["", "You are an expert."]
        suffixes = suffixes or ["", "Be concise."]

        combinations = list(itertools.product(prefixes, suffixes))
        if len(combinations) > max_combinations:
            combinations = random.sample(combinations, max_combinations)

        variants = []
        for i, (prefix, suffix) in enumerate(combinations):
            parts = []
            if prefix:
                parts.append(prefix)
            parts.append(self.base_prompt)
            if suffix:
                parts.append(suffix)

            variants.append(
                PromptVariant(
                    id=f"combo_{i}",
                    content="\n\n".join(parts),
                    metadata={
                        "variation_type": "combination",
                        "prefix": prefix,
                        "suffix": suffix,
                    },
                )
            )

        return variants


@dataclass
class ScoringCriteria:
    """Criteria for scoring prompt responses."""

    name: str
    weight: float = 1.0
    scorer: Optional[Callable[[str, str, str], float]] = None
    description: str = ""


class PromptScorer:
    """Score prompt responses based on configurable criteria."""

    def __init__(self):
        """Initialize with default scoring criteria."""
        self._criteria: list[ScoringCriteria] = []

    def add_criteria(
        self,
        name: str,
        scorer: Callable[[str, str, str], float],
        weight: float = 1.0,
        description: str = "",
    ) -> "PromptScorer":
        """Add a scoring criterion.

        The scorer function receives (prompt, response, expected) and returns a score 0-1.

        Args:
            name: Name of the criterion.
            scorer: Scoring function.
            weight: Weight for this criterion.
            description: Optional description.

        Returns:
            Self for chaining.
        """
        self._criteria.append(
            ScoringCriteria(
                name=name,
                weight=weight,
                scorer=scorer,
                description=description,
            )
        )
        return self

    def add_length_criteria(
        self,
        min_length: int = 10,
        max_length: int = 1000,
        weight: float = 1.0,
    ) -> "PromptScorer":
        """Add length-based scoring.

        Args:
            min_length: Minimum acceptable length.
            max_length: Maximum acceptable length.
            weight: Weight for this criterion.

        Returns:
            Self for chaining.
        """

        def length_scorer(prompt: str, response: str, expected: str) -> float:
            length = len(response)
            if length < min_length:
                return length / min_length
            elif length > max_length:
                return max(0, 1 - (length - max_length) / max_length)
            return 1.0

        return self.add_criteria(
            "length",
            length_scorer,
            weight,
            f"Response length between {min_length} and {max_length}",
        )

    def add_keyword_criteria(
        self,
        required_keywords: list[str],
        weight: float = 1.0,
    ) -> "PromptScorer":
        """Add keyword presence scoring.

        Args:
            required_keywords: Keywords that should appear in response.
            weight: Weight for this criterion.

        Returns:
            Self for chaining.
        """

        def keyword_scorer(prompt: str, response: str, expected: str) -> float:
            response_lower = response.lower()
            matches = sum(1 for kw in required_keywords if kw.lower() in response_lower)
            return matches / len(required_keywords) if required_keywords else 1.0

        return self.add_criteria(
            "keywords",
            keyword_scorer,
            weight,
            f"Contains keywords: {', '.join(required_keywords)}",
        )

    def add_similarity_criteria(
        self,
        weight: float = 1.0,
    ) -> "PromptScorer":
        """Add similarity to expected response scoring.

        Args:
            weight: Weight for this criterion.

        Returns:
            Self for chaining.
        """

        def similarity_scorer(prompt: str, response: str, expected: str) -> float:
            if not expected:
                return 1.0  # No expected response to compare

            # Simple word overlap similarity
            response_words = set(word_tokenize_regex(response))
            expected_words = set(word_tokenize_regex(expected))

            if not expected_words:
                return 1.0

            overlap = len(response_words & expected_words)
            return overlap / len(expected_words)

        return self.add_criteria(
            "similarity",
            similarity_scorer,
            weight,
            "Similarity to expected response",
        )

    def add_format_criteria(
        self,
        expected_format: str,
        weight: float = 1.0,
    ) -> "PromptScorer":
        """Add format compliance scoring.

        Args:
            expected_format: Expected format (json, markdown, code, bullets).
            weight: Weight for this criterion.

        Returns:
            Self for chaining.
        """

        def format_scorer(prompt: str, response: str, expected: str) -> float:
            if expected_format == "json":
                try:
                    import json

                    json.loads(response)
                    return 1.0
                except json.JSONDecodeError:
                    # Check for JSON-like structure
                    if "{" in response and "}" in response:
                        return 0.5
                    return 0.0
            elif expected_format == "markdown":
                # Check for markdown indicators
                indicators = ["#", "```", "**", "*", "-", "1."]
                score = sum(1 for ind in indicators if ind in response)
                return min(1.0, score / 2)
            elif expected_format == "code":
                if "```" in response or response.strip().startswith("def "):
                    return 1.0
                return 0.3
            elif expected_format == "bullets":
                if re.search(r"^[\-\*\•]\s", response, re.MULTILINE):
                    return 1.0
                if re.search(r"^\d+\.\s", response, re.MULTILINE):
                    return 0.8
                return 0.0
            return 1.0

        return self.add_criteria(
            "format",
            format_scorer,
            weight,
            f"Response format: {expected_format}",
        )

    def score(
        self,
        prompt: str,
        response: str,
        expected: str = "",
    ) -> tuple[float, dict[str, float]]:
        """Score a response.

        Args:
            prompt: The prompt that was used.
            response: The response to score.
            expected: Optional expected response for comparison.

        Returns:
            Tuple of (overall_score, individual_scores).
        """
        if not self._criteria:
            return 1.0, {}

        scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for criterion in self._criteria:
            if criterion.scorer:
                score = criterion.scorer(prompt, response, expected)
                scores[criterion.name] = score
                weighted_sum += score * criterion.weight
                total_weight += criterion.weight

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        return overall, scores


class PromptABTestRunner:
    """Run A/B tests on prompt variants."""

    def __init__(
        self,
        scorer: Optional[PromptScorer] = None,
    ):
        """Initialize the A/B test runner.

        Args:
            scorer: Optional custom scorer.
        """
        self.scorer = scorer or PromptScorer()
        self._results: list[PromptTestResult] = []

    def test_variant(
        self,
        variant: PromptVariant,
        model_fn: Callable[[str], str],
        expected: str = "",
        num_runs: int = 1,
    ) -> list[PromptTestResult]:
        """Test a single variant.

        Args:
            variant: The prompt variant to test.
            model_fn: Function that takes prompt and returns response.
            expected: Optional expected response.
            num_runs: Number of times to run.

        Returns:
            List of test results.
        """
        results = []
        for _ in range(num_runs):
            start_time = datetime.now()
            error = None

            try:
                response = model_fn(variant.content)
            except Exception as e:
                response = ""
                error = str(e)

            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000

            if error:
                score = 0.0
                score_details = {}
            else:
                score, score_details = self.scorer.score(variant.content, response, expected)

            result = PromptTestResult(
                variant_id=variant.id,
                response=response,
                score=score,
                latency_ms=latency_ms,
                metadata={
                    "score_details": score_details,
                    "expected": expected,
                },
                error=error,
            )
            results.append(result)
            self._results.append(result)

        return results

    def run_experiment(
        self,
        variants: list[PromptVariant],
        model_fn: Callable[[str], str],
        expected: str = "",
        runs_per_variant: int = 1,
        experiment_id: Optional[str] = None,
    ) -> PromptExperimentResult:
        """Run a full experiment with multiple variants.

        Args:
            variants: List of variants to test.
            model_fn: Function that takes prompt and returns response.
            expected: Optional expected response.
            runs_per_variant: Number of runs per variant.
            experiment_id: Optional experiment identifier.

        Returns:
            Complete experiment results.
        """
        experiment_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_results = []

        for variant in variants:
            results = self.test_variant(variant, model_fn, expected, runs_per_variant)
            all_results.extend(results)

        experiment = PromptExperimentResult(
            experiment_id=experiment_id,
            variants=variants,
            results=all_results,
            completed_at=datetime.now(),
        )

        # Compute summary
        avg_scores = experiment.get_average_scores()
        experiment.best_variant_id = experiment.get_best_variant()
        experiment.summary = {
            "average_scores": avg_scores,
            "best_variant": experiment.best_variant_id,
            "total_runs": len(all_results),
            "variants_tested": len(variants),
        }

        return experiment


@dataclass
class ExpandablePromptTemplate:
    """Template for generating prompts with variables."""

    template: str
    variables: dict[str, list[Any]] = field(default_factory=dict)

    def expand(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Expand template with all variable combinations.

        Yields:
            Tuples of (rendered_prompt, variables_used).
        """
        if not self.variables:
            yield self.template, {}
            return

        keys = list(self.variables.keys())
        values = [self.variables[k] for k in keys]

        for combo in itertools.product(*values):
            var_dict = dict(zip(keys, combo))
            try:
                rendered = self.template.format(**var_dict)
                yield rendered, var_dict
            except KeyError:
                continue


class PromptExperiment:
    """High-level experiment orchestration for prompt testing."""

    def __init__(self, name: str):
        """Initialize an experiment.

        Args:
            name: Name for this experiment.
        """
        self.name = name
        self.variants: list[PromptVariant] = []
        self.scorer = PromptScorer()
        self.results: Optional[PromptExperimentResult] = None

    def add_variant(
        self,
        content: str,
        variant_id: Optional[str] = None,
        strategy: Optional[PromptStrategy] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "PromptExperiment":
        """Add a prompt variant.

        Args:
            content: Prompt content.
            variant_id: Optional ID (auto-generated if not provided).
            strategy: Optional prompting strategy.
            metadata: Optional metadata.

        Returns:
            Self for chaining.
        """
        variant_id = variant_id or f"variant_{len(self.variants)}"
        self.variants.append(
            PromptVariant(
                id=variant_id,
                content=content,
                strategy=strategy,
                metadata=metadata or {},
            )
        )
        return self

    def add_variants_from_generator(
        self, generator: PromptVariationGenerator
    ) -> "PromptExperiment":
        """Add variants from a generator.

        Args:
            generator: PromptVariationGenerator instance.

        Returns:
            Self for chaining.
        """
        self.variants.extend(generator.generate())
        return self

    def configure_scorer(
        self,
        length_range: Optional[tuple[int, int]] = None,
        required_keywords: Optional[list[str]] = None,
        expected_format: Optional[str] = None,
        check_similarity: bool = False,
    ) -> "PromptExperiment":
        """Configure the scorer with common criteria.

        Args:
            length_range: Optional (min, max) length range.
            required_keywords: Optional keywords to check.
            expected_format: Optional format requirement.
            check_similarity: Whether to check similarity to expected.

        Returns:
            Self for chaining.
        """
        if length_range:
            self.scorer.add_length_criteria(length_range[0], length_range[1])
        if required_keywords:
            self.scorer.add_keyword_criteria(required_keywords)
        if expected_format:
            self.scorer.add_format_criteria(expected_format)
        if check_similarity:
            self.scorer.add_similarity_criteria()
        return self

    def run(
        self,
        model_fn: Callable[[str], str],
        expected: str = "",
        runs_per_variant: int = 1,
    ) -> PromptExperimentResult:
        """Run the experiment.

        Args:
            model_fn: Function that takes prompt and returns response.
            expected: Optional expected response.
            runs_per_variant: Number of runs per variant.

        Returns:
            Experiment results.
        """
        runner = PromptABTestRunner(scorer=self.scorer)
        self.results = runner.run_experiment(
            variants=self.variants,
            model_fn=model_fn,
            expected=expected,
            runs_per_variant=runs_per_variant,
            experiment_id=self.name,
        )
        return self.results

    def get_best_prompt(self) -> Optional[str]:
        """Get the best performing prompt.

        Returns:
            The best prompt content, or None if not run.
        """
        if not self.results or not self.results.best_variant_id:
            return None

        for variant in self.variants:
            if variant.id == self.results.best_variant_id:
                return variant.content
        return None

    def generate_report(self) -> str:
        """Generate a text report of results.

        Returns:
            Formatted report string.
        """
        if not self.results:
            return "Experiment has not been run yet."

        lines = [
            f"# Prompt Experiment: {self.name}",
            "",
            f"**Total Variants:** {len(self.variants)}",
            f"**Total Runs:** {len(self.results.results)}",
            "",
            "## Results by Variant",
            "",
        ]

        avg_scores = self.results.get_average_scores()
        sorted_variants = sorted(
            avg_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for rank, (variant_id, avg_score) in enumerate(sorted_variants, 1):
            indicator = " ⭐" if variant_id == self.results.best_variant_id else ""
            lines.append(f"{rank}. **{variant_id}**: {avg_score:.3f}{indicator}")

        lines.extend(
            [
                "",
                "## Best Prompt",
                "",
                "```",
                self.get_best_prompt() or "N/A",
                "```",
            ]
        )

        return "\n".join(lines)


def create_few_shot_variants(
    task: str,
    examples: list[dict[str, str]],
    query: str,
    num_shots: list[int] = None,
) -> list[PromptVariant]:
    """Create variants with different numbers of few-shot examples.

    Args:
        task: The task description.
        examples: Pool of examples to draw from.
        query: The actual query.
        num_shots: List of example counts to try.

    Returns:
        List of prompt variants.
    """
    if num_shots is None:
        num_shots = [0, 1, 3, 5]
    variants = []

    for n in num_shots:
        if n == 0:
            # Zero-shot
            content = f"{task}\n\nQuery: {query}"
            strategy = PromptStrategy.ZERO_SHOT
        else:
            # Few-shot
            selected_examples = examples[:n]
            example_text = "\n\n".join(
                f"Input: {ex.get('input', ex.get('question', ''))}\n"
                f"Output: {ex.get('output', ex.get('answer', ''))}"
                for ex in selected_examples
            )
            content = f"{task}\n\nExamples:\n{example_text}\n\nQuery: {query}"
            strategy = PromptStrategy.FEW_SHOT

        variants.append(
            PromptVariant(
                id=f"{n}_shot",
                content=content,
                strategy=strategy,
                metadata={"num_examples": n},
            )
        )

    return variants


def create_cot_variants(
    prompt: str,
) -> list[PromptVariant]:
    """Create chain-of-thought prompt variants.

    Args:
        prompt: The base prompt.

    Returns:
        List of CoT variants.
    """
    return [
        PromptVariant(
            id="no_cot",
            content=prompt,
            strategy=PromptStrategy.ZERO_SHOT,
        ),
        PromptVariant(
            id="cot_basic",
            content=f"{prompt}\n\nLet's think step by step.",
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        ),
        PromptVariant(
            id="cot_detailed",
            content=f"{prompt}\n\nLet's work through this step by step:\n1.",
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        ),
        PromptVariant(
            id="cot_reasoning",
            content=f"{prompt}\n\nBefore answering, let me reason through this carefully:",
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        ),
    ]


# Backwards-compatible alias (deprecated)
TestResult = PromptTestResult
