"""
Prompt optimization and tuning utilities.

Provides tools for:
- Prompt compression and reduction
- Instruction optimization
- Few-shot example selection
- Prompt ablation studies
- Token budget optimization
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class OptimizationStrategy(Enum):
    """Strategies for prompt optimization."""

    COMPRESSION = "compression"  # Reduce token count
    CLARITY = "clarity"  # Improve clarity
    SPECIFICITY = "specificity"  # Add specificity
    STRUCTURE = "structure"  # Improve structure
    EXAMPLE_SELECTION = "example_selection"  # Optimize few-shot examples


@dataclass
class CompressionResult:
    """Result of prompt compression."""

    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    removed_elements: list[str] = field(default_factory=list)
    preserved_elements: list[str] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved."""
        return self.original_tokens - self.compressed_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "tokens_saved": self.tokens_saved,
            "removed_elements": self.removed_elements,
            "preserved_elements": self.preserved_elements,
        }


@dataclass
class AblationResult:
    """Result of prompt ablation study."""

    original_prompt: str
    components: list[str]
    component_scores: dict[str, float]
    essential_components: list[str]
    removable_components: list[str]
    minimal_prompt: str
    importance_ranking: list[tuple[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_components": len(self.components),
            "component_scores": self.component_scores,
            "essential_components": self.essential_components,
            "removable_components": self.removable_components,
            "importance_ranking": self.importance_ranking,
        }


@dataclass
class ExampleScore:
    """Score for a few-shot example."""

    example: dict[str, str]
    relevance_score: float
    diversity_score: float
    quality_score: float
    overall_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "relevance_score": self.relevance_score,
            "diversity_score": self.diversity_score,
            "quality_score": self.quality_score,
            "overall_score": self.overall_score,
        }


@dataclass
class ExampleSelectionResult:
    """Result of few-shot example selection."""

    query: str
    selected_examples: list[dict[str, str]]
    example_scores: list[ExampleScore]
    coverage_score: float
    diversity_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_selected": len(self.selected_examples),
            "coverage_score": self.coverage_score,
            "diversity_score": self.diversity_score,
            "example_scores": [e.to_dict() for e in self.example_scores],
        }


@dataclass
class OptimizationReport:
    """Comprehensive optimization report."""

    original_prompt: str
    optimized_prompt: str
    strategies_applied: list[OptimizationStrategy]
    improvements: dict[str, float]
    suggestions: list[str]
    token_reduction: int
    estimated_quality_change: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategies_applied": [s.value for s in self.strategies_applied],
            "improvements": self.improvements,
            "suggestions": self.suggestions,
            "token_reduction": self.token_reduction,
            "estimated_quality_change": self.estimated_quality_change,
        }


class PromptCompressor:
    """Compress prompts while preserving meaning."""

    # Filler phrases that can usually be removed
    FILLER_PHRASES = [
        "please note that",
        "it is important to",
        "keep in mind that",
        "as mentioned earlier",
        "in other words",
        "basically",
        "essentially",
        "actually",
        "literally",
        "very",
        "really",
        "quite",
        "somewhat",
        "rather",
        "just",
        "simply",
    ]

    # Verbose constructions that can be simplified
    VERBOSE_PATTERNS = [
        (r"in order to", "to"),
        (r"due to the fact that", "because"),
        (r"at this point in time", "now"),
        (r"in the event that", "if"),
        (r"for the purpose of", "for"),
        (r"with regard to", "about"),
        (r"in terms of", "regarding"),
        (r"on a daily basis", "daily"),
        (r"at the present time", "currently"),
        (r"in the near future", "soon"),
        (r"a large number of", "many"),
        (r"a small number of", "few"),
        (r"the majority of", "most"),
        (r"in spite of the fact that", "although"),
        (r"whether or not", "whether"),
    ]

    def __init__(self, preserve_structure: bool = True):
        """Initialize compressor.

        Args:
            preserve_structure: Whether to preserve structural elements.
        """
        self.preserve_structure = preserve_structure

    def compress(
        self,
        prompt: str,
        target_reduction: float = 0.2,
        preserve_keywords: Optional[set[str]] = None,
    ) -> CompressionResult:
        """Compress a prompt.

        Args:
            prompt: Original prompt.
            target_reduction: Target token reduction (0-1).
            preserve_keywords: Keywords that must be preserved.

        Returns:
            Compression result.
        """
        preserve_keywords = preserve_keywords or set()
        original_tokens = self._estimate_tokens(prompt)
        removed = []
        preserved = []

        compressed = prompt

        # Remove filler phrases
        for filler in self.FILLER_PHRASES:
            if filler in compressed.lower():
                # Check if any preserve keyword is in the filler
                if not any(kw.lower() in filler for kw in preserve_keywords):
                    pattern = re.compile(re.escape(filler), re.IGNORECASE)
                    if pattern.search(compressed):
                        compressed = pattern.sub("", compressed)
                        removed.append(f"filler: {filler}")

        # Apply verbose pattern substitutions
        for pattern, replacement in self.VERBOSE_PATTERNS:
            if re.search(pattern, compressed, re.IGNORECASE):
                compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)
                removed.append(f"verbose: {pattern} -> {replacement}")

        # Remove redundant whitespace
        compressed = re.sub(r"\s+", " ", compressed)
        compressed = compressed.strip()

        # Remove empty parentheses and brackets
        compressed = re.sub(r"\(\s*\)", "", compressed)
        compressed = re.sub(r"\[\s*\]", "", compressed)

        # Track preserved elements
        for keyword in preserve_keywords:
            if keyword.lower() in compressed.lower():
                preserved.append(keyword)

        compressed_tokens = self._estimate_tokens(compressed)
        ratio = 1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0

        return CompressionResult(
            original=prompt,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            removed_elements=removed,
            preserved_elements=preserved,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Rough approximation: ~4 characters per token
        return len(text) // 4 + 1


class InstructionOptimizer:
    """Optimize instruction clarity and effectiveness."""

    # Weak instruction verbs
    WEAK_VERBS = ["try to", "attempt to", "consider", "think about", "maybe"]

    # Strong instruction verbs
    STRONG_VERBS = {
        "try to": "ensure",
        "attempt to": "make sure to",
        "consider": "evaluate",
        "think about": "analyze",
    }

    # Ambiguous terms
    AMBIGUOUS_TERMS = [
        "good",
        "bad",
        "nice",
        "proper",
        "appropriate",
        "reasonable",
        "suitable",
        "adequate",
    ]

    def optimize(self, instruction: str) -> tuple[str, list[str]]:
        """Optimize an instruction.

        Args:
            instruction: Original instruction.

        Returns:
            Tuple of (optimized instruction, list of changes).
        """
        changes = []
        optimized = instruction

        # Replace weak verbs
        for weak, strong in self.STRONG_VERBS.items():
            if weak in optimized.lower():
                pattern = re.compile(re.escape(weak), re.IGNORECASE)
                optimized = pattern.sub(strong, optimized)
                changes.append(f"Strengthened verb: '{weak}' -> '{strong}'")

        # Flag ambiguous terms
        for term in self.AMBIGUOUS_TERMS:
            if re.search(rf"\b{term}\b", optimized, re.IGNORECASE):
                changes.append(f"Consider clarifying ambiguous term: '{term}'")

        # Ensure instruction ends with clear action
        if not optimized.strip().endswith((".", ":", "?")):
            optimized = optimized.strip() + "."
            changes.append("Added ending punctuation")

        # Check for missing context
        if len(optimized.split()) < 5:
            changes.append("Instruction may be too brief - consider adding context")

        return optimized, changes

    def analyze_clarity(self, instruction: str) -> dict[str, Any]:
        """Analyze instruction clarity.

        Args:
            instruction: Instruction to analyze.

        Returns:
            Clarity analysis.
        """
        issues = []
        score = 1.0

        # Check for weak verbs
        weak_count = sum(1 for v in self.WEAK_VERBS if v in instruction.lower())
        if weak_count > 0:
            issues.append(f"Contains {weak_count} weak verbs")
            score -= weak_count * 0.1

        # Check for ambiguous terms
        ambiguous_count = sum(
            1 for t in self.AMBIGUOUS_TERMS if re.search(rf"\b{t}\b", instruction, re.IGNORECASE)
        )
        if ambiguous_count > 0:
            issues.append(f"Contains {ambiguous_count} ambiguous terms")
            score -= ambiguous_count * 0.05

        # Check length
        word_count = len(instruction.split())
        if word_count < 5:
            issues.append("Too brief")
            score -= 0.2
        elif word_count > 100:
            issues.append("May be too long")
            score -= 0.1

        # Check for action verb at start
        first_word = instruction.split()[0].lower() if instruction.split() else ""
        action_verbs = [
            "write",
            "create",
            "generate",
            "analyze",
            "explain",
            "describe",
            "list",
            "provide",
            "summarize",
            "identify",
            "compare",
            "evaluate",
        ]
        if first_word not in action_verbs:
            issues.append("Consider starting with an action verb")
            score -= 0.05

        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "word_count": word_count,
            "has_action_verb": first_word in action_verbs,
        }


class FewShotSelector:
    """Select optimal few-shot examples."""

    def __init__(self, diversity_weight: float = 0.3):
        """Initialize selector.

        Args:
            diversity_weight: Weight for diversity in scoring (0-1).
        """
        self.diversity_weight = diversity_weight

    def select(
        self,
        query: str,
        examples: list[dict[str, str]],
        n: int = 3,
        input_key: str = "input",
        output_key: str = "output",
    ) -> ExampleSelectionResult:
        """Select optimal examples for a query.

        Args:
            query: The query/task to select examples for.
            examples: Pool of available examples.
            n: Number of examples to select.
            input_key: Key for input in example dicts.
            output_key: Key for output in example dicts.

        Returns:
            Selection result.
        """
        if not examples:
            return ExampleSelectionResult(
                query=query,
                selected_examples=[],
                example_scores=[],
                coverage_score=0.0,
                diversity_score=0.0,
            )

        # Score each example
        scored_examples = []
        for example in examples:
            relevance = self._calculate_relevance(query, example.get(input_key, ""))
            quality = self._calculate_quality(
                example.get(input_key, ""),
                example.get(output_key, ""),
            )

            scored_examples.append(
                {
                    "example": example,
                    "relevance": relevance,
                    "quality": quality,
                }
            )

        # Sort by relevance initially
        scored_examples.sort(key=lambda x: x["relevance"], reverse=True)

        # Select with diversity
        selected = []
        selected_scores = []

        for i in range(min(n, len(scored_examples))):
            if i == 0:
                # First example: pick most relevant
                best = scored_examples[0]
            else:
                # Subsequent: balance relevance and diversity
                best = None
                best_score = -1

                for candidate in scored_examples:
                    if candidate["example"] in [s["example"] for s in selected]:
                        continue

                    diversity = self._calculate_diversity(
                        candidate["example"].get(input_key, ""),
                        [s["example"].get(input_key, "") for s in selected],
                    )

                    combined = (1 - self.diversity_weight) * candidate[
                        "relevance"
                    ] + self.diversity_weight * diversity

                    if combined > best_score:
                        best_score = combined
                        best = candidate
                        best["diversity"] = diversity

                if best is None:
                    break

            selected.append(best)

            diversity = best.get("diversity", 1.0) if i > 0 else 1.0
            overall = 0.4 * best["relevance"] + 0.3 * diversity + 0.3 * best["quality"]

            selected_scores.append(
                ExampleScore(
                    example=best["example"],
                    relevance_score=best["relevance"],
                    diversity_score=diversity,
                    quality_score=best["quality"],
                    overall_score=overall,
                )
            )

        # Calculate overall metrics
        coverage = self._calculate_coverage(query, [s["example"] for s in selected], input_key)
        avg_diversity = (
            sum(s.diversity_score for s in selected_scores) / len(selected_scores)
            if selected_scores
            else 0.0
        )

        return ExampleSelectionResult(
            query=query,
            selected_examples=[s["example"] for s in selected],
            example_scores=selected_scores,
            coverage_score=coverage,
            diversity_score=avg_diversity,
        )

    def _calculate_relevance(self, query: str, example_input: str) -> float:
        """Calculate relevance of example to query."""
        query_words = set(query.lower().split())
        example_words = set(example_input.lower().split())

        # Remove stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "to",
            "of",
            "and",
            "in",
            "that",
            "it",
            "for",
        }
        query_words -= stop_words
        example_words -= stop_words

        if not query_words:
            return 0.5

        overlap = len(query_words & example_words)
        return overlap / len(query_words)

    def _calculate_diversity(self, candidate: str, selected: list[str]) -> float:
        """Calculate diversity from already selected examples."""
        if not selected:
            return 1.0

        candidate_words = set(candidate.lower().split())

        similarities = []
        for sel in selected:
            sel_words = set(sel.lower().split())
            if candidate_words | sel_words:
                sim = len(candidate_words & sel_words) / len(candidate_words | sel_words)
                similarities.append(sim)

        if not similarities:
            return 1.0

        # Diversity is inverse of max similarity
        return 1.0 - max(similarities)

    def _calculate_quality(self, input_text: str, output_text: str) -> float:
        """Calculate quality of an example."""
        score = 0.5

        # Check output length is reasonable relative to input
        input_len = len(input_text.split())
        output_len = len(output_text.split())

        if output_len >= 3:
            score += 0.2

        # Check output is not too short
        if output_len >= input_len * 0.5:
            score += 0.15

        # Check output is complete (ends properly)
        if output_text.strip().endswith((".", "!", "?", "```")):
            score += 0.15

        return min(1.0, score)

    def _calculate_coverage(
        self,
        query: str,
        examples: list[dict[str, str]],
        input_key: str,
    ) -> float:
        """Calculate how well examples cover query concepts."""
        query_words = set(query.lower().split())
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "to",
            "of",
            "and",
            "in",
            "that",
            "it",
            "for",
        }
        query_words -= stop_words

        if not query_words:
            return 1.0

        covered = set()
        for example in examples:
            example_words = set(example.get(input_key, "").lower().split())
            covered |= query_words & example_words

        return len(covered) / len(query_words)


class PromptAblator:
    """Perform ablation studies on prompts."""

    def __init__(self, scorer: Optional[Callable[[str], float]] = None):
        """Initialize ablator.

        Args:
            scorer: Function to score prompt quality (prompt -> score).
        """
        self.scorer = scorer or self._default_scorer

    def ablate(
        self,
        prompt: str,
        component_delimiter: str = "\n\n",
    ) -> AblationResult:
        """Perform ablation study.

        Args:
            prompt: Prompt to ablate.
            component_delimiter: Delimiter between components.

        Returns:
            Ablation result.
        """
        # Split into components
        components = [c.strip() for c in prompt.split(component_delimiter) if c.strip()]

        if len(components) <= 1:
            return AblationResult(
                original_prompt=prompt,
                components=components,
                component_scores={components[0] if components else "": 1.0},
                essential_components=components,
                removable_components=[],
                minimal_prompt=prompt,
                importance_ranking=[(components[0] if components else "", 1.0)],
            )

        # Score full prompt
        full_score = self.scorer(prompt)

        # Score with each component removed
        component_scores = {}
        for i, component in enumerate(components):
            # Create prompt without this component
            remaining = [c for j, c in enumerate(components) if j != i]
            ablated_prompt = component_delimiter.join(remaining)

            ablated_score = self.scorer(ablated_prompt)

            # Importance = drop in score when removed
            importance = full_score - ablated_score
            component_scores[component[:50] + "..."] = importance

        # Rank by importance
        importance_ranking = sorted(
            component_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Identify essential vs removable
        essential = []
        removable = []
        threshold = 0.1  # Components causing < 10% drop are removable

        for component, importance in importance_ranking:
            if importance > threshold * full_score:
                essential.append(component)
            else:
                removable.append(component)

        # Create minimal prompt from essential components only
        essential_full = [c for c in components if c[:50] + "..." in essential]
        minimal_prompt = component_delimiter.join(essential_full)

        return AblationResult(
            original_prompt=prompt,
            components=components,
            component_scores=component_scores,
            essential_components=essential,
            removable_components=removable,
            minimal_prompt=minimal_prompt,
            importance_ranking=importance_ranking,
        )

    def _default_scorer(self, prompt: str) -> float:
        """Default scorer based on heuristics."""
        score = 0.5

        # Length score
        word_count = len(prompt.split())
        if 10 <= word_count <= 200:
            score += 0.2
        elif word_count > 200:
            score += 0.1

        # Structure score
        if "\n" in prompt:
            score += 0.1
        if any(c in prompt for c in ["1.", "2.", "-", "•"]):
            score += 0.1

        # Clarity score
        if prompt.strip().endswith((".", "?", ":")):
            score += 0.1

        return min(1.0, score)


class TokenBudgetOptimizer:
    """Optimize prompts within token budgets."""

    def __init__(self, max_tokens: int = 4096):
        """Initialize optimizer.

        Args:
            max_tokens: Maximum token budget.
        """
        self.max_tokens = max_tokens
        self.compressor = PromptCompressor()

    def optimize(
        self,
        prompt: str,
        examples: Optional[list[dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        reserve_for_response: int = 500,
    ) -> dict[str, Any]:
        """Optimize prompt to fit within token budget.

        Args:
            prompt: Main prompt.
            examples: Few-shot examples.
            system_prompt: System prompt.
            reserve_for_response: Tokens to reserve for response.

        Returns:
            Optimization result.
        """
        available_tokens = self.max_tokens - reserve_for_response

        # Estimate current usage
        prompt_tokens = self._estimate_tokens(prompt)
        system_tokens = self._estimate_tokens(system_prompt) if system_prompt else 0
        example_tokens = sum(self._estimate_tokens(str(e)) for e in (examples or []))

        total_tokens = prompt_tokens + system_tokens + example_tokens
        over_budget = total_tokens > available_tokens

        result = {
            "original_tokens": total_tokens,
            "available_tokens": available_tokens,
            "over_budget": over_budget,
            "actions_taken": [],
        }

        if not over_budget:
            result["final_prompt"] = prompt
            result["final_examples"] = examples
            result["final_system"] = system_prompt
            result["final_tokens"] = total_tokens
            return result

        # Strategy 1: Compress prompt
        compressed = self.compressor.compress(prompt, target_reduction=0.3)
        prompt = compressed.compressed
        prompt_tokens = compressed.compressed_tokens
        result["actions_taken"].append(f"Compressed prompt: saved {compressed.tokens_saved} tokens")

        # Strategy 2: Reduce examples if still over
        total_tokens = prompt_tokens + system_tokens + example_tokens
        if total_tokens > available_tokens and examples:
            # Remove examples one by one
            while examples and total_tokens > available_tokens:
                examples = examples[:-1]
                example_tokens = sum(self._estimate_tokens(str(e)) for e in examples)
                total_tokens = prompt_tokens + system_tokens + example_tokens
            result["actions_taken"].append(f"Reduced to {len(examples)} examples")

        # Strategy 3: Truncate prompt if still over
        if total_tokens > available_tokens:
            excess = total_tokens - available_tokens
            chars_to_remove = excess * 4  # Rough conversion
            prompt = prompt[:-chars_to_remove] + "..."
            prompt_tokens = self._estimate_tokens(prompt)
            result["actions_taken"].append("Truncated prompt")

        result["final_prompt"] = prompt
        result["final_examples"] = examples
        result["final_system"] = system_prompt
        result["final_tokens"] = prompt_tokens + system_tokens + example_tokens

        return result

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        if not text:
            return 0
        return len(text) // 4 + 1


class PromptOptimizer:
    """Comprehensive prompt optimizer."""

    def __init__(self):
        """Initialize optimizer."""
        self.compressor = PromptCompressor()
        self.instruction_optimizer = InstructionOptimizer()

    def optimize(
        self,
        prompt: str,
        strategies: Optional[list[OptimizationStrategy]] = None,
    ) -> OptimizationReport:
        """Optimize a prompt using specified strategies.

        Args:
            prompt: Prompt to optimize.
            strategies: Strategies to apply (default: all).

        Returns:
            Optimization report.
        """
        strategies = strategies or [
            OptimizationStrategy.COMPRESSION,
            OptimizationStrategy.CLARITY,
            OptimizationStrategy.STRUCTURE,
        ]

        optimized = prompt
        improvements = {}
        suggestions = []
        applied_strategies = []

        # Apply compression
        if OptimizationStrategy.COMPRESSION in strategies:
            compression = self.compressor.compress(optimized)
            if compression.compression_ratio > 0.05:
                optimized = compression.compressed
                improvements["compression"] = compression.compression_ratio
                applied_strategies.append(OptimizationStrategy.COMPRESSION)

        # Apply clarity optimization
        if OptimizationStrategy.CLARITY in strategies:
            clarity_result = self.instruction_optimizer.analyze_clarity(optimized)
            if clarity_result["issues"]:
                suggestions.extend(clarity_result["issues"])
                optimized_text, changes = self.instruction_optimizer.optimize(optimized)
                if changes:
                    optimized = optimized_text
                    improvements["clarity"] = clarity_result["score"]
                    applied_strategies.append(OptimizationStrategy.CLARITY)

        # Apply structure optimization
        if OptimizationStrategy.STRUCTURE in strategies:
            structure_improved, structure_changes = self._optimize_structure(optimized)
            if structure_changes:
                optimized = structure_improved
                improvements["structure"] = 0.1 * len(structure_changes)
                suggestions.extend(structure_changes)
                applied_strategies.append(OptimizationStrategy.STRUCTURE)

        original_tokens = len(prompt) // 4 + 1
        optimized_tokens = len(optimized) // 4 + 1
        token_reduction = original_tokens - optimized_tokens

        # Estimate quality change based on improvements
        quality_change = sum(improvements.values()) / len(improvements) if improvements else 0

        return OptimizationReport(
            original_prompt=prompt,
            optimized_prompt=optimized,
            strategies_applied=applied_strategies,
            improvements=improvements,
            suggestions=suggestions,
            token_reduction=token_reduction,
            estimated_quality_change=quality_change,
        )

    def _optimize_structure(self, prompt: str) -> tuple[str, list[str]]:
        """Optimize prompt structure."""
        changes = []
        optimized = prompt

        # Add newlines before lists if missing
        if re.search(r"[^\n][-•*\d+.]\s", optimized):
            optimized = re.sub(r"([^\n])([-•*\d+.])\s", r"\1\n\2 ", optimized)
            changes.append("Added line breaks before list items")

        # Ensure consistent list formatting
        if re.search(r"^\d+\)", optimized, re.MULTILINE):
            optimized = re.sub(r"^(\d+)\)", r"\1.", optimized, flags=re.MULTILINE)
            changes.append("Standardized list numbering")

        return optimized, changes


# Convenience functions


def compress_prompt(
    prompt: str,
    target_reduction: float = 0.2,
    preserve_keywords: Optional[set[str]] = None,
) -> CompressionResult:
    """Compress a prompt.

    Args:
        prompt: Original prompt.
        target_reduction: Target reduction (0-1).
        preserve_keywords: Keywords to preserve.

    Returns:
        Compression result.
    """
    compressor = PromptCompressor()
    return compressor.compress(prompt, target_reduction, preserve_keywords)


def optimize_instruction(instruction: str) -> tuple[str, list[str]]:
    """Optimize an instruction.

    Args:
        instruction: Original instruction.

    Returns:
        Tuple of (optimized, changes).
    """
    optimizer = InstructionOptimizer()
    return optimizer.optimize(instruction)


def select_examples(
    query: str,
    examples: list[dict[str, str]],
    n: int = 3,
    input_key: str = "input",
    output_key: str = "output",
) -> ExampleSelectionResult:
    """Select optimal few-shot examples.

    Args:
        query: Query to select for.
        examples: Pool of examples.
        n: Number to select.
        input_key: Key for input.
        output_key: Key for output.

    Returns:
        Selection result.
    """
    selector = FewShotSelector()
    return selector.select(query, examples, n, input_key, output_key)


def ablate_prompt(
    prompt: str,
    component_delimiter: str = "\n\n",
    scorer: Optional[Callable[[str], float]] = None,
) -> AblationResult:
    """Perform prompt ablation study.

    Args:
        prompt: Prompt to ablate.
        component_delimiter: Delimiter between components.
        scorer: Custom scorer function.

    Returns:
        Ablation result.
    """
    ablator = PromptAblator(scorer)
    return ablator.ablate(prompt, component_delimiter)


def optimize_prompt(
    prompt: str,
    strategies: Optional[list[OptimizationStrategy]] = None,
) -> OptimizationReport:
    """Optimize a prompt.

    Args:
        prompt: Prompt to optimize.
        strategies: Strategies to apply.

    Returns:
        Optimization report.
    """
    optimizer = PromptOptimizer()
    return optimizer.optimize(prompt, strategies)


def optimize_for_budget(
    prompt: str,
    max_tokens: int = 4096,
    examples: Optional[list[dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    reserve_for_response: int = 500,
) -> dict[str, Any]:
    """Optimize prompt for token budget.

    Args:
        prompt: Main prompt.
        max_tokens: Token budget.
        examples: Few-shot examples.
        system_prompt: System prompt.
        reserve_for_response: Reserve for response.

    Returns:
        Optimization result.
    """
    optimizer = TokenBudgetOptimizer(max_tokens)
    return optimizer.optimize(prompt, examples, system_prompt, reserve_for_response)
