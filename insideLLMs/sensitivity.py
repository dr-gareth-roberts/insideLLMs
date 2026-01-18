"""Prompt sensitivity analysis for LLM evaluation.

This module provides tools for systematically analyzing how small
changes to prompts affect model outputs:

- Perturbation-based sensitivity testing
- Semantic equivalence testing
- Format sensitivity analysis
- Instruction sensitivity profiling
- Comparative sensitivity across models
"""

from __future__ import annotations

import random
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class PerturbationType(Enum):
    """Types of prompt perturbations."""

    CASE_CHANGE = "case_change"
    WHITESPACE = "whitespace"
    PUNCTUATION = "punctuation"
    SYNONYM = "synonym"
    PARAPHRASE = "paraphrase"
    WORD_ORDER = "word_order"
    TYPO = "typo"
    FORMATTING = "formatting"
    INSTRUCTION_STYLE = "instruction_style"


class SensitivityLevel(Enum):
    """Levels of sensitivity."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class OutputChangeType(Enum):
    """Types of output changes."""

    NO_CHANGE = "no_change"
    MINOR_VARIATION = "minor_variation"
    SEMANTIC_EQUIVALENT = "semantic_equivalent"
    DIFFERENT_FORMAT = "different_format"
    DIFFERENT_CONTENT = "different_content"
    CONTRADICTORY = "contradictory"
    FAILURE = "failure"


@dataclass
class Perturbation:
    """A single prompt perturbation."""

    original: str
    perturbed: str
    perturbation_type: PerturbationType
    change_description: str
    change_magnitude: float  # 0-1, how much changed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": self.original,
            "perturbed": self.perturbed,
            "type": self.perturbation_type.value,
            "description": self.change_description,
            "magnitude": self.change_magnitude,
        }


@dataclass
class OutputComparison:
    """Comparison between two outputs."""

    original_output: str
    perturbed_output: str
    change_type: OutputChangeType
    similarity_score: float  # 0-1
    semantic_similarity: float  # 0-1
    length_ratio: float
    key_differences: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_output": self.original_output[:200],
            "perturbed_output": self.perturbed_output[:200],
            "change_type": self.change_type.value,
            "similarity_score": self.similarity_score,
            "semantic_similarity": self.semantic_similarity,
            "length_ratio": self.length_ratio,
            "key_differences": self.key_differences,
        }


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for one perturbation."""

    perturbation: Perturbation
    output_comparison: OutputComparison
    sensitivity_score: float  # 0-1, higher = more sensitive
    is_robust: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "perturbation": self.perturbation.to_dict(),
            "output_comparison": self.output_comparison.to_dict(),
            "sensitivity_score": self.sensitivity_score,
            "is_robust": self.is_robust,
            "notes": self.notes,
        }


@dataclass
class SensitivityProfile:
    """Complete sensitivity profile for a prompt."""

    prompt: str
    results: list[SensitivityResult]
    overall_sensitivity: SensitivityLevel
    overall_score: float
    by_perturbation_type: dict[PerturbationType, float]
    most_sensitive_to: list[PerturbationType]
    most_robust_to: list[PerturbationType]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt[:200],
            "n_tests": len(self.results),
            "overall_sensitivity": self.overall_sensitivity.value,
            "overall_score": self.overall_score,
            "by_perturbation_type": {k.value: v for k, v in self.by_perturbation_type.items()},
            "most_sensitive_to": [p.value for p in self.most_sensitive_to],
            "most_robust_to": [p.value for p in self.most_robust_to],
            "recommendations": self.recommendations,
        }


@dataclass
class ComparativeSensitivity:
    """Compare sensitivity across multiple prompts or models."""

    profiles: list[SensitivityProfile]
    ranking: list[tuple[str, float]]  # (identifier, sensitivity_score)
    most_robust: str
    most_sensitive: str
    common_sensitivities: list[PerturbationType]
    divergent_sensitivities: list[PerturbationType]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_profiles": len(self.profiles),
            "ranking": self.ranking,
            "most_robust": self.most_robust,
            "most_sensitive": self.most_sensitive,
            "common_sensitivities": [p.value for p in self.common_sensitivities],
            "divergent_sensitivities": [p.value for p in self.divergent_sensitivities],
        }


class PromptPerturbator:
    """Generate perturbations of prompts."""

    def __init__(self, seed: int | None = None):
        """Initialize perturbator.

        Args:
            seed: Random seed for reproducibility
        """
        self._rng = random.Random(seed)
        self._synonyms = {
            "explain": ["describe", "elaborate", "clarify"],
            "write": ["compose", "create", "generate"],
            "list": ["enumerate", "name", "identify"],
            "analyze": ["examine", "evaluate", "assess"],
            "summarize": ["condense", "recap", "outline"],
            "compare": ["contrast", "differentiate", "distinguish"],
            "define": ["describe", "explain", "specify"],
            "good": ["excellent", "great", "quality"],
            "bad": ["poor", "negative", "low-quality"],
            "important": ["significant", "crucial", "key"],
            "simple": ["basic", "straightforward", "easy"],
            "complex": ["complicated", "intricate", "sophisticated"],
        }

    def perturb(
        self,
        prompt: str,
        perturbation_types: list[PerturbationType] | None = None,
        n_variations: int = 1,
    ) -> list[Perturbation]:
        """Generate perturbations of a prompt.

        Args:
            prompt: Original prompt
            perturbation_types: Types of perturbations to apply
            n_variations: Number of variations per type

        Returns:
            List of Perturbation objects
        """
        if perturbation_types is None:
            perturbation_types = list(PerturbationType)

        perturbations = []
        for ptype in perturbation_types:
            for _ in range(n_variations):
                perturbed = self._apply_perturbation(prompt, ptype)
                if perturbed and perturbed != prompt:
                    magnitude = self._calculate_magnitude(prompt, perturbed)
                    perturbations.append(
                        Perturbation(
                            original=prompt,
                            perturbed=perturbed,
                            perturbation_type=ptype,
                            change_description=self._describe_change(ptype),
                            change_magnitude=magnitude,
                        )
                    )

        return perturbations

    def _apply_perturbation(
        self,
        prompt: str,
        ptype: PerturbationType,
    ) -> str | None:
        """Apply a specific perturbation type."""
        methods = {
            PerturbationType.CASE_CHANGE: self._perturb_case,
            PerturbationType.WHITESPACE: self._perturb_whitespace,
            PerturbationType.PUNCTUATION: self._perturb_punctuation,
            PerturbationType.SYNONYM: self._perturb_synonym,
            PerturbationType.PARAPHRASE: self._perturb_paraphrase,
            PerturbationType.WORD_ORDER: self._perturb_word_order,
            PerturbationType.TYPO: self._perturb_typo,
            PerturbationType.FORMATTING: self._perturb_formatting,
            PerturbationType.INSTRUCTION_STYLE: self._perturb_instruction_style,
        }
        method = methods.get(ptype)
        if method:
            return method(prompt)
        return None

    def _perturb_case(self, prompt: str) -> str:
        """Change case of prompt."""
        options = [
            prompt.lower(),
            prompt.upper(),
            prompt.capitalize(),
            prompt.title(),
        ]
        return self._rng.choice([o for o in options if o != prompt] or [prompt])

    def _perturb_whitespace(self, prompt: str) -> str:
        """Modify whitespace in prompt."""
        options = [
            prompt.strip(),
            "  " + prompt + "  ",
            prompt.replace("  ", " "),
            re.sub(r"\s+", " ", prompt),
            prompt + "\n",
        ]
        return self._rng.choice([o for o in options if o != prompt] or [prompt])

    def _perturb_punctuation(self, prompt: str) -> str:
        """Modify punctuation in prompt."""
        options = []

        # Remove trailing punctuation
        if prompt and prompt[-1] in ".!?":
            options.append(prompt[:-1])

        # Add question mark
        if prompt and prompt[-1] not in "?":
            options.append(prompt.rstrip(".!") + "?")

        # Add period
        if prompt and prompt[-1] not in ".":
            options.append(prompt.rstrip("!?") + ".")

        # Double punctuation
        if prompt:
            options.append(prompt + "!")

        return self._rng.choice(options) if options else prompt

    def _perturb_synonym(self, prompt: str) -> str:
        """Replace words with synonyms."""
        words = prompt.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip(string.punctuation)
            if word_lower in self._synonyms:
                synonyms = self._synonyms[word_lower]
                replacement = self._rng.choice(synonyms)
                # Preserve original case
                if word[0].isupper():
                    replacement = replacement.capitalize()
                # Preserve punctuation
                if word and word[-1] in string.punctuation:
                    replacement += word[-1]
                words[i] = replacement
                break  # Only replace one word
        return " ".join(words)

    def _perturb_paraphrase(self, prompt: str) -> str:
        """Simple paraphrasing through restructuring."""
        if not prompt:
            return prompt

        # Add "Please" if not present
        if not prompt.lower().startswith("please"):
            return "Please " + prompt[0].lower() + prompt[1:]

        # Remove "Please" if present
        if prompt.lower().startswith("please ") and len(prompt) > 7:
            rest = prompt[7:]
            if rest:
                return rest[0].upper() + rest[1:]

        return prompt

    def _perturb_word_order(self, prompt: str) -> str:
        """Reorder words in prompt."""
        # Move last word to beginning for questions
        words = prompt.split()
        if len(words) > 3:
            # Swap adjacent words
            idx = self._rng.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
            return " ".join(words)
        return prompt

    def _perturb_typo(self, prompt: str) -> str:
        """Introduce a typo."""
        if len(prompt) < 5:
            return prompt

        words = prompt.split()
        if not words:
            return prompt

        # Find a word to modify
        word_idx = self._rng.randint(0, len(words) - 1)
        word = words[word_idx]

        if len(word) < 3:
            return prompt

        typo_types = ["swap", "delete", "double"]
        typo_type = self._rng.choice(typo_types)

        if typo_type == "swap" and len(word) > 2:
            # Swap two adjacent characters
            idx = self._rng.randint(1, len(word) - 2)
            word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :]
        elif typo_type == "delete" and len(word) > 3:
            # Delete a character
            idx = self._rng.randint(1, len(word) - 2)
            word = word[:idx] + word[idx + 1 :]
        elif typo_type == "double":
            # Double a character
            idx = self._rng.randint(1, len(word) - 1)
            word = word[:idx] + word[idx] + word[idx:]

        words[word_idx] = word
        return " ".join(words)

    def _perturb_formatting(self, prompt: str) -> str:
        """Change formatting of prompt."""
        options = [
            f"'{prompt}'",  # Add quotes
            f"**{prompt}**",  # Bold markers
            f"- {prompt}",  # List item
            f"1. {prompt}",  # Numbered
            f"[INSTRUCTION] {prompt}",  # Tagged
        ]
        return self._rng.choice(options)

    def _perturb_instruction_style(self, prompt: str) -> str:
        """Change instruction style."""
        # Convert imperative to question
        if prompt and not prompt.endswith("?"):
            words = prompt.lower().split()
            if words and words[0] in ["explain", "describe", "write", "list"]:
                verb = words[0]
                rest = " ".join(words[1:])
                return f"Can you {verb} {rest}?"

        # Convert question to imperative
        if prompt.lower().startswith("can you "):
            rest = prompt[8:].rstrip("?")
            return rest[0].upper() + rest[1:] + "."

        return prompt

    def _calculate_magnitude(self, original: str, perturbed: str) -> float:
        """Calculate magnitude of change between strings."""
        if original == perturbed:
            return 0.0

        # Character-level difference
        max_len = max(len(original), len(perturbed))
        if max_len == 0:
            return 0.0

        # Levenshtein-like distance approximation
        common = sum(1 for a, b in zip(original, perturbed) if a == b)
        similarity = common / max_len

        return 1.0 - similarity

    @staticmethod
    def _describe_change(ptype: PerturbationType) -> str:
        """Get description of perturbation type."""
        descriptions = {
            PerturbationType.CASE_CHANGE: "Changed letter casing",
            PerturbationType.WHITESPACE: "Modified whitespace",
            PerturbationType.PUNCTUATION: "Changed punctuation",
            PerturbationType.SYNONYM: "Replaced word with synonym",
            PerturbationType.PARAPHRASE: "Paraphrased instruction",
            PerturbationType.WORD_ORDER: "Reordered words",
            PerturbationType.TYPO: "Introduced typo",
            PerturbationType.FORMATTING: "Changed formatting",
            PerturbationType.INSTRUCTION_STYLE: "Changed instruction style",
        }
        return descriptions.get(ptype, "Unknown perturbation")


class OutputComparator:
    """Compare model outputs."""

    def __init__(
        self,
        similarity_fn: Callable[[str, str], float] | None = None,
    ):
        """Initialize comparator.

        Args:
            similarity_fn: Custom similarity function
        """
        self._similarity_fn = similarity_fn or self._default_similarity

    @staticmethod
    def _default_similarity(text1: str, text2: str) -> float:
        """Calculate default text similarity."""
        if not text1 or not text2:
            return 0.0 if text1 != text2 else 1.0

        # Word-level Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def compare(
        self,
        original: str,
        perturbed: str,
    ) -> OutputComparison:
        """Compare two outputs.

        Args:
            original: Original output
            perturbed: Output from perturbed prompt

        Returns:
            OutputComparison object
        """
        similarity = self._similarity_fn(original, perturbed)
        semantic_sim = self._estimate_semantic_similarity(original, perturbed)

        # Calculate length ratio
        len_orig = len(original)
        len_pert = len(perturbed)
        length_ratio = (
            min(len_orig, len_pert) / max(len_orig, len_pert)
            if max(len_orig, len_pert) > 0
            else 1.0
        )

        # Determine change type
        change_type = self._classify_change(similarity, semantic_sim, length_ratio)

        # Find key differences
        differences = self._find_differences(original, perturbed)

        return OutputComparison(
            original_output=original,
            perturbed_output=perturbed,
            change_type=change_type,
            similarity_score=similarity,
            semantic_similarity=semantic_sim,
            length_ratio=length_ratio,
            key_differences=differences,
        )

    def _estimate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Estimate semantic similarity (simplified)."""

        # Use n-gram overlap as proxy
        def get_ngrams(text: str, n: int) -> set[str]:
            words = text.lower().split()
            return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

        bigrams1 = get_ngrams(text1, 2)
        bigrams2 = get_ngrams(text2, 2)

        if not bigrams1 and not bigrams2:
            return 1.0 if text1 == text2 else 0.5

        if not bigrams1 or not bigrams2:
            return 0.3

        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _classify_change(
        similarity: float,
        semantic_sim: float,
        length_ratio: float,
    ) -> OutputChangeType:
        """Classify type of output change."""
        if similarity > 0.95:
            return OutputChangeType.NO_CHANGE

        if similarity > 0.8 and length_ratio > 0.8:
            return OutputChangeType.MINOR_VARIATION

        if semantic_sim > 0.7:
            return OutputChangeType.SEMANTIC_EQUIVALENT

        if length_ratio < 0.5 or length_ratio > 2.0:
            if semantic_sim > 0.4:
                return OutputChangeType.DIFFERENT_FORMAT
            return OutputChangeType.DIFFERENT_CONTENT

        if semantic_sim < 0.2:
            return OutputChangeType.CONTRADICTORY

        return OutputChangeType.DIFFERENT_CONTENT

    @staticmethod
    def _find_differences(text1: str, text2: str) -> list[str]:
        """Find key differences between texts."""
        differences = []

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        only_in_1 = words1 - words2
        only_in_2 = words2 - words1

        if only_in_1:
            differences.append(f"Original contains: {', '.join(list(only_in_1)[:5])}")
        if only_in_2:
            differences.append(f"Perturbed contains: {', '.join(list(only_in_2)[:5])}")

        len_diff = len(text2) - len(text1)
        if abs(len_diff) > 50:
            differences.append(f"Length difference: {len_diff:+d} characters")

        return differences[:5]


class SensitivityAnalyzer:
    """Analyze prompt sensitivity."""

    def __init__(
        self,
        perturbator: PromptPerturbator | None = None,
        comparator: OutputComparator | None = None,
        robustness_threshold: float = 0.7,
    ):
        """Initialize analyzer.

        Args:
            perturbator: Prompt perturbator
            comparator: Output comparator
            robustness_threshold: Similarity threshold for robustness
        """
        self._perturbator = perturbator or PromptPerturbator()
        self._comparator = comparator or OutputComparator()
        self._robustness_threshold = robustness_threshold

    def analyze(
        self,
        prompt: str,
        get_response: Callable[[str], str],
        perturbation_types: list[PerturbationType] | None = None,
        n_variations: int = 2,
    ) -> SensitivityProfile:
        """Analyze sensitivity of a prompt.

        Args:
            prompt: Original prompt
            get_response: Function to get model response
            perturbation_types: Types of perturbations to test
            n_variations: Number of variations per type

        Returns:
            SensitivityProfile object
        """
        # Get original response
        original_response = get_response(prompt)

        # Generate perturbations
        perturbations = self._perturbator.perturb(prompt, perturbation_types, n_variations)

        # Test each perturbation
        results = []
        type_scores: dict[PerturbationType, list[float]] = defaultdict(list)

        for perturbation in perturbations:
            perturbed_response = get_response(perturbation.perturbed)
            comparison = self._comparator.compare(original_response, perturbed_response)

            # Calculate sensitivity score
            sensitivity = 1.0 - comparison.similarity_score
            is_robust = comparison.similarity_score >= self._robustness_threshold

            result = SensitivityResult(
                perturbation=perturbation,
                output_comparison=comparison,
                sensitivity_score=sensitivity,
                is_robust=is_robust,
                notes=self._generate_notes(perturbation, comparison),
            )
            results.append(result)
            type_scores[perturbation.perturbation_type].append(sensitivity)

        # Aggregate by perturbation type
        by_type = {
            ptype: sum(scores) / len(scores) if scores else 0.0
            for ptype, scores in type_scores.items()
        }

        # Calculate overall
        all_scores = [r.sensitivity_score for r in results]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        overall_level = self._score_to_level(overall_score)

        # Find most/least sensitive
        sorted_types = sorted(by_type.items(), key=lambda x: x[1], reverse=True)
        most_sensitive = [t for t, s in sorted_types if s > 0.3][:3]
        most_robust = [t for t, s in sorted_types if s < 0.2][:3]

        return SensitivityProfile(
            prompt=prompt,
            results=results,
            overall_sensitivity=overall_level,
            overall_score=overall_score,
            by_perturbation_type=by_type,
            most_sensitive_to=most_sensitive,
            most_robust_to=most_robust,
            recommendations=self._generate_recommendations(by_type, overall_score),
        )

    @staticmethod
    def _score_to_level(score: float) -> SensitivityLevel:
        """Convert score to sensitivity level."""
        if score >= 0.8:
            return SensitivityLevel.VERY_HIGH
        elif score >= 0.6:
            return SensitivityLevel.HIGH
        elif score >= 0.4:
            return SensitivityLevel.MODERATE
        elif score >= 0.2:
            return SensitivityLevel.LOW
        return SensitivityLevel.VERY_LOW

    @staticmethod
    def _generate_notes(
        perturbation: Perturbation,
        comparison: OutputComparison,
    ) -> list[str]:
        """Generate notes for a result."""
        notes = []

        if comparison.change_type == OutputChangeType.CONTRADICTORY:
            notes.append("Warning: Outputs are contradictory")

        if comparison.length_ratio < 0.5:
            notes.append("Significant length reduction in output")
        elif comparison.length_ratio > 2.0:
            notes.append("Significant length increase in output")

        if perturbation.change_magnitude < 0.1 and comparison.similarity_score < 0.5:
            notes.append("Small input change caused large output change")

        return notes

    @staticmethod
    def _generate_recommendations(
        by_type: dict[PerturbationType, float],
        overall: float,
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if overall > 0.5:
            recommendations.append("Consider making prompts more specific to reduce sensitivity")

        if by_type.get(PerturbationType.CASE_CHANGE, 0) > 0.3:
            recommendations.append("Model is case-sensitive; ensure consistent casing in prompts")

        if by_type.get(PerturbationType.TYPO, 0) > 0.3:
            recommendations.append("Model is typo-sensitive; consider input validation")

        if by_type.get(PerturbationType.INSTRUCTION_STYLE, 0) > 0.3:
            recommendations.append(
                "Model responds differently to instruction styles; standardize format"
            )

        if by_type.get(PerturbationType.SYNONYM, 0) > 0.3:
            recommendations.append("Model is sensitive to word choice; use precise terminology")

        if not recommendations:
            recommendations.append("Prompt appears robust to common perturbations")

        return recommendations


class ComparativeSensitivityAnalyzer:
    """Compare sensitivity across prompts or models."""

    def __init__(self, analyzer: SensitivityAnalyzer | None = None):
        """Initialize comparative analyzer.

        Args:
            analyzer: Base sensitivity analyzer
        """
        self._analyzer = analyzer or SensitivityAnalyzer()

    def compare_prompts(
        self,
        prompts: list[str],
        get_response: Callable[[str], str],
        perturbation_types: list[PerturbationType] | None = None,
    ) -> ComparativeSensitivity:
        """Compare sensitivity across multiple prompts.

        Args:
            prompts: List of prompts to compare
            get_response: Function to get model response
            perturbation_types: Types of perturbations to test

        Returns:
            ComparativeSensitivity object
        """
        profiles = []
        for prompt in prompts:
            profile = self._analyzer.analyze(prompt, get_response, perturbation_types)
            profiles.append(profile)

        return self._build_comparison(profiles, [p[:50] for p in prompts])

    def compare_models(
        self,
        prompt: str,
        model_responses: dict[str, Callable[[str], str]],
        perturbation_types: list[PerturbationType] | None = None,
    ) -> ComparativeSensitivity:
        """Compare sensitivity across multiple models.

        Args:
            prompt: Prompt to test
            model_responses: Dict of model_id -> response function
            perturbation_types: Types of perturbations to test

        Returns:
            ComparativeSensitivity object
        """
        profiles = []
        identifiers = []

        for model_id, get_response in model_responses.items():
            profile = self._analyzer.analyze(prompt, get_response, perturbation_types)
            profiles.append(profile)
            identifiers.append(model_id)

        return self._build_comparison(profiles, identifiers)

    def _build_comparison(
        self,
        profiles: list[SensitivityProfile],
        identifiers: list[str],
    ) -> ComparativeSensitivity:
        """Build comparison from profiles."""
        # Rank by sensitivity
        ranking = sorted(
            zip(identifiers, [p.overall_score for p in profiles]),
            key=lambda x: x[1],
        )

        # Find common sensitivities
        all_sensitive: list[set[PerturbationType]] = []
        for profile in profiles:
            sensitive = {
                ptype for ptype, score in profile.by_perturbation_type.items() if score > 0.3
            }
            all_sensitive.append(sensitive)

        if all_sensitive:
            common = all_sensitive[0]
            for s in all_sensitive[1:]:
                common = common & s
        else:
            common = set()

        # Find divergent sensitivities
        all_types = set()
        for profile in profiles:
            all_types.update(profile.by_perturbation_type.keys())

        divergent = []
        for ptype in all_types:
            scores = [p.by_perturbation_type.get(ptype, 0) for p in profiles]
            if max(scores) - min(scores) > 0.4:
                divergent.append(ptype)

        return ComparativeSensitivity(
            profiles=profiles,
            ranking=ranking,
            most_robust=ranking[0][0] if ranking else "",
            most_sensitive=ranking[-1][0] if ranking else "",
            common_sensitivities=list(common),
            divergent_sensitivities=divergent,
        )


class FormatSensitivityTester:
    """Test sensitivity to output format instructions."""

    def __init__(self):
        """Initialize tester."""
        self._format_variations = [
            ("json", "Respond in JSON format"),
            ("markdown", "Respond in Markdown format"),
            ("bullet", "Respond with bullet points"),
            ("numbered", "Respond with a numbered list"),
            ("paragraph", "Respond in paragraph form"),
            ("brief", "Respond briefly"),
            ("detailed", "Respond in detail"),
        ]

    def test_format_sensitivity(
        self,
        base_prompt: str,
        get_response: Callable[[str], str],
    ) -> dict[str, Any]:
        """Test how format instructions affect output.

        Args:
            base_prompt: Base prompt without format instruction
            get_response: Function to get model response

        Returns:
            Dictionary with format sensitivity results
        """
        baseline = get_response(base_prompt)
        results = {"baseline": baseline, "variations": {}}

        comparator = OutputComparator()

        for format_name, format_instruction in self._format_variations:
            formatted_prompt = f"{base_prompt}\n\n{format_instruction}"
            response = get_response(formatted_prompt)

            comparison = comparator.compare(baseline, response)
            results["variations"][format_name] = {
                "response": response[:500],
                "similarity_to_baseline": comparison.similarity_score,
                "length_ratio": comparison.length_ratio,
                "format_followed": self._check_format_followed(response, format_name),
            }

        # Calculate format adherence
        adherence_scores = [v["format_followed"] for v in results["variations"].values()]
        results["format_adherence_rate"] = (
            sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0
        )

        return results

    @staticmethod
    def _check_format_followed(response: str, format_name: str) -> bool:
        """Check if format instruction was followed."""
        if format_name == "json":
            return "{" in response and "}" in response
        elif format_name == "markdown":
            return "#" in response or "**" in response or "*" in response
        elif format_name == "bullet":
            return "- " in response or "• " in response
        elif format_name == "numbered":
            return any(f"{i}." in response for i in range(1, 10))
        elif format_name == "brief":
            return len(response) < 500
        elif format_name == "detailed":
            return len(response) > 200
        return True  # Paragraph is default


# Convenience functions


def analyze_prompt_sensitivity(
    prompt: str,
    get_response: Callable[[str], str],
    perturbation_types: list[PerturbationType] | None = None,
) -> SensitivityProfile:
    """Analyze sensitivity of a prompt.

    Args:
        prompt: Prompt to analyze
        get_response: Function to get model response
        perturbation_types: Types of perturbations to test

    Returns:
        SensitivityProfile object
    """
    analyzer = SensitivityAnalyzer()
    return analyzer.analyze(prompt, get_response, perturbation_types)


def compare_prompt_sensitivity(
    prompts: list[str],
    get_response: Callable[[str], str],
) -> ComparativeSensitivity:
    """Compare sensitivity across prompts.

    Args:
        prompts: List of prompts to compare
        get_response: Function to get model response

    Returns:
        ComparativeSensitivity object
    """
    analyzer = ComparativeSensitivityAnalyzer()
    return analyzer.compare_prompts(prompts, get_response)


def generate_perturbations(
    prompt: str,
    perturbation_types: list[PerturbationType] | None = None,
    n_variations: int = 2,
) -> list[Perturbation]:
    """Generate perturbations of a prompt.

    Args:
        prompt: Original prompt
        perturbation_types: Types of perturbations to apply
        n_variations: Number of variations per type

    Returns:
        List of Perturbation objects
    """
    perturbator = PromptPerturbator()
    return perturbator.perturb(prompt, perturbation_types, n_variations)


def quick_sensitivity_check(
    prompt: str,
    get_response: Callable[[str], str],
) -> dict[str, Any]:
    """Quick sensitivity check with basic perturbations.

    Args:
        prompt: Prompt to check
        get_response: Function to get model response

    Returns:
        Dictionary with quick check results
    """
    basic_types = [
        PerturbationType.CASE_CHANGE,
        PerturbationType.TYPO,
        PerturbationType.SYNONYM,
    ]

    analyzer = SensitivityAnalyzer()
    profile = analyzer.analyze(prompt, get_response, basic_types, n_variations=1)

    return {
        "overall_sensitivity": profile.overall_sensitivity.value,
        "overall_score": profile.overall_score,
        "is_robust": profile.overall_score < 0.3,
        "n_tests": len(profile.results),
        "recommendations": profile.recommendations[:2],
    }


def check_format_sensitivity(
    prompt: str,
    get_response: Callable[[str], str],
) -> dict[str, Any]:
    """Check sensitivity to format instructions.

    Args:
        prompt: Base prompt
        get_response: Function to get model response

    Returns:
        Dictionary with format sensitivity results
    """
    tester = FormatSensitivityTester()
    return tester.test_format_sensitivity(prompt, get_response)
