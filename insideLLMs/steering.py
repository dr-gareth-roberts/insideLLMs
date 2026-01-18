"""Model steering and activation analysis for understanding LLM internal representations.

This module provides tools for analyzing how different prompting strategies
affect model behavior and internal representations, including:
- Steering vector analysis and manipulation
- Activation pattern extraction and comparison
- Prompt-based steering experiments
- Representation space analysis
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union


class SteeringMethod(Enum):
    """Methods for steering model behavior."""

    PROMPT_PREFIX = "prompt_prefix"
    PROMPT_SUFFIX = "prompt_suffix"
    SYSTEM_MESSAGE = "system_message"
    FEW_SHOT = "few_shot"
    CONTRAST_PAIR = "contrast_pair"
    ACTIVATION_ADDITION = "activation_addition"
    SOFT_PROMPT = "soft_prompt"


class ActivationLayer(Enum):
    """Types of activation layers to analyze."""

    INPUT_EMBEDDING = "input_embedding"
    ATTENTION = "attention"
    MLP = "mlp"
    RESIDUAL = "residual"
    OUTPUT = "output"
    ALL = "all"


class RepresentationSpace(Enum):
    """Types of representation spaces."""

    TOKEN = "token"
    SEQUENCE = "sequence"
    SEMANTIC = "semantic"
    TASK = "task"


class SteeringStrength(Enum):
    """Strength levels for steering interventions."""

    MINIMAL = "minimal"
    LIGHT = "light"
    MODERATE = "moderate"
    STRONG = "strong"
    MAXIMUM = "maximum"


@dataclass
class SteeringVector:
    """A vector that can be used to steer model behavior."""

    name: str
    direction: list[float]
    magnitude: float
    source: str  # How the vector was derived
    target_behavior: str
    layer: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "source": self.source,
            "target_behavior": self.target_behavior,
            "layer": self.layer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SteeringVector":
        return cls(
            name=data["name"],
            direction=data["direction"],
            magnitude=data["magnitude"],
            source=data["source"],
            target_behavior=data["target_behavior"],
            layer=data.get("layer"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ActivationPattern:
    """Pattern of activations extracted from model inference."""

    prompt: str
    layer: str
    activations: list[float]
    token_positions: list[int]
    attention_weights: Optional[list[list[float]]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "layer": self.layer,
            "activations": self.activations,
            "token_positions": self.token_positions,
            "attention_weights": self.attention_weights,
            "metadata": self.metadata,
        }

    @property
    def mean_activation(self) -> float:
        """Calculate mean activation value."""
        if not self.activations:
            return 0.0
        return sum(self.activations) / len(self.activations)

    @property
    def activation_variance(self) -> float:
        """Calculate activation variance."""
        if len(self.activations) < 2:
            return 0.0
        mean = self.mean_activation
        return sum((x - mean) ** 2 for x in self.activations) / len(self.activations)


@dataclass
class SteeringExperiment:
    """Results from a steering experiment."""

    original_prompt: str
    steering_method: SteeringMethod
    steering_config: dict[str, Any]
    original_output: str
    steered_output: str
    behavioral_shift: float  # 0-1 measure of how much behavior changed
    direction_alignment: float  # How well the shift aligned with intended direction
    side_effects: list[str]  # Unintended behavioral changes
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "steering_method": self.steering_method.value,
            "steering_config": self.steering_config,
            "original_output": self.original_output,
            "steered_output": self.steered_output,
            "behavioral_shift": self.behavioral_shift,
            "direction_alignment": self.direction_alignment,
            "side_effects": self.side_effects,
            "metadata": self.metadata,
        }


@dataclass
class ContrastPair:
    """A pair of prompts designed to elicit contrasting behaviors."""

    positive_prompt: str
    negative_prompt: str
    target_dimension: str  # e.g., "formality", "helpfulness", "verbosity"
    expected_difference: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "target_dimension": self.target_dimension,
            "expected_difference": self.expected_difference,
        }


@dataclass
class SteeringReport:
    """Comprehensive report on steering analysis."""

    experiments: list[SteeringExperiment]
    vectors_extracted: list[SteeringVector]
    effective_methods: list[tuple[SteeringMethod, float]]  # Method and effectiveness
    behavioral_dimensions: dict[str, float]  # Dimension name -> controllability
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiments": [e.to_dict() for e in self.experiments],
            "vectors_extracted": [v.to_dict() for v in self.vectors_extracted],
            "effective_methods": [
                {"method": m.value, "effectiveness": e} for m, e in self.effective_methods
            ],
            "behavioral_dimensions": self.behavioral_dimensions,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }

    @property
    def overall_steerability(self) -> float:
        """Calculate overall model steerability score."""
        if not self.experiments:
            return 0.0
        return sum(e.behavioral_shift for e in self.experiments) / len(self.experiments)


class SteeringVectorExtractor:
    """Extract steering vectors from contrast pairs or examples."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.extracted_vectors: list[SteeringVector] = []

    def extract_from_contrast_pair(
        self,
        positive_activations: list[float],
        negative_activations: list[float],
        name: str,
        target_behavior: str,
        layer: Optional[str] = None,
    ) -> SteeringVector:
        """Extract a steering vector from contrasting activations."""
        if len(positive_activations) != len(negative_activations):
            raise ValueError("Activation lists must have the same length")

        # Compute difference vector
        direction = [p - n for p, n in zip(positive_activations, negative_activations)]

        # Calculate magnitude
        magnitude = math.sqrt(sum(x**2 for x in direction))

        # Normalize if requested
        if self.normalize and magnitude > 0:
            direction = [x / magnitude for x in direction]

        vector = SteeringVector(
            name=name,
            direction=direction,
            magnitude=magnitude,
            source="contrast_pair",
            target_behavior=target_behavior,
            layer=layer,
        )

        self.extracted_vectors.append(vector)
        return vector

    def extract_from_examples(
        self,
        positive_examples: list[list[float]],
        negative_examples: list[list[float]],
        name: str,
        target_behavior: str,
        layer: Optional[str] = None,
    ) -> SteeringVector:
        """Extract a steering vector from multiple example pairs."""
        if not positive_examples or not negative_examples:
            raise ValueError("Need at least one example of each type")

        # Average positive activations
        dim = len(positive_examples[0])
        pos_mean = [
            sum(ex[i] for ex in positive_examples) / len(positive_examples) for i in range(dim)
        ]

        # Average negative activations
        neg_mean = [
            sum(ex[i] for ex in negative_examples) / len(negative_examples) for i in range(dim)
        ]

        return self.extract_from_contrast_pair(pos_mean, neg_mean, name, target_behavior, layer)

    def combine_vectors(
        self,
        vectors: list[SteeringVector],
        weights: Optional[list[float]] = None,
        name: str = "combined",
    ) -> SteeringVector:
        """Combine multiple steering vectors with optional weights."""
        if not vectors:
            raise ValueError("Need at least one vector to combine")

        if weights is None:
            weights = [1.0 / len(vectors)] * len(vectors)

        if len(weights) != len(vectors):
            raise ValueError("Weights must match number of vectors")

        # Ensure all vectors have same dimension
        dim = len(vectors[0].direction)
        for v in vectors:
            if len(v.direction) != dim:
                raise ValueError("All vectors must have the same dimension")

        # Weighted combination
        combined = [0.0] * dim
        for v, w in zip(vectors, weights):
            for i in range(dim):
                combined[i] += w * v.direction[i]

        # Calculate new magnitude
        magnitude = math.sqrt(sum(x**2 for x in combined))

        # Normalize if needed
        if self.normalize and magnitude > 0:
            combined = [x / magnitude for x in combined]

        return SteeringVector(
            name=name,
            direction=combined,
            magnitude=magnitude,
            source="combination",
            target_behavior=f"Combined: {', '.join(v.target_behavior for v in vectors)}",
            metadata={"component_vectors": [v.name for v in vectors], "weights": weights},
        )


class PromptSteerer:
    """Apply steering through prompt manipulation."""

    def __init__(self):
        self.steering_templates: dict[str, str] = {
            "formal": "Please respond in a formal, professional tone.",
            "casual": "Feel free to be casual and friendly in your response.",
            "concise": "Please be brief and to the point.",
            "detailed": "Please provide a comprehensive, detailed response.",
            "creative": "Think creatively and outside the box.",
            "analytical": "Approach this analytically and logically.",
            "helpful": "Be as helpful as possible.",
            "cautious": "Be careful and consider potential issues.",
        }

    def steer_with_prefix(
        self,
        prompt: str,
        steering_instruction: str,
    ) -> str:
        """Add a steering instruction as a prefix."""
        return f"{steering_instruction}\n\n{prompt}"

    def steer_with_suffix(
        self,
        prompt: str,
        steering_instruction: str,
    ) -> str:
        """Add a steering instruction as a suffix."""
        return f"{prompt}\n\n{steering_instruction}"

    def steer_with_system_message(
        self,
        prompt: str,
        system_message: str,
    ) -> tuple[str, str]:
        """Return system message and user prompt separately."""
        return system_message, prompt

    def steer_with_few_shot(
        self,
        prompt: str,
        examples: list[tuple[str, str]],  # List of (input, output) pairs
    ) -> str:
        """Steer using few-shot examples."""
        few_shot_text = ""
        for i, (inp, out) in enumerate(examples, 1):
            few_shot_text += f"Example {i}:\nInput: {inp}\nOutput: {out}\n\n"

        return f"{few_shot_text}Now, please respond to:\nInput: {prompt}\nOutput:"

    def apply_steering(
        self,
        prompt: str,
        method: SteeringMethod,
        config: dict[str, Any],
    ) -> Union[str, tuple[str, str]]:
        """Apply steering based on method and configuration."""
        if method == SteeringMethod.PROMPT_PREFIX:
            instruction = config.get("instruction", "")
            return self.steer_with_prefix(prompt, instruction)

        elif method == SteeringMethod.PROMPT_SUFFIX:
            instruction = config.get("instruction", "")
            return self.steer_with_suffix(prompt, instruction)

        elif method == SteeringMethod.SYSTEM_MESSAGE:
            system_msg = config.get("system_message", "")
            return self.steer_with_system_message(prompt, system_msg)

        elif method == SteeringMethod.FEW_SHOT:
            examples = config.get("examples", [])
            return self.steer_with_few_shot(prompt, examples)

        elif method == SteeringMethod.CONTRAST_PAIR:
            # For contrast pair, we modify the prompt to encourage one side
            positive_framing = config.get("positive_framing", "")
            return f"{positive_framing}\n\n{prompt}"

        else:
            # For methods requiring model internals, return prompt unchanged
            return prompt

    def get_preset_steering(self, style: str) -> str:
        """Get a preset steering instruction by style name."""
        return self.steering_templates.get(style, "")

    def create_contrast_pair(
        self,
        base_prompt: str,
        dimension: str,
        positive_style: Optional[str] = None,
        negative_style: Optional[str] = None,
    ) -> ContrastPair:
        """Create a contrast pair for a given dimension."""
        style_pairs = {
            "formality": ("formal", "casual"),
            "length": ("detailed", "concise"),
            "creativity": ("creative", "analytical"),
            "helpfulness": ("helpful", "cautious"),
        }

        if dimension in style_pairs and not positive_style:
            positive_style, negative_style = style_pairs[dimension]

        positive_instruction = self.get_preset_steering(positive_style or "helpful")
        negative_instruction = self.get_preset_steering(negative_style or "cautious")

        return ContrastPair(
            positive_prompt=self.steer_with_prefix(base_prompt, positive_instruction),
            negative_prompt=self.steer_with_prefix(base_prompt, negative_instruction),
            target_dimension=dimension,
            expected_difference=f"More {positive_style} vs more {negative_style}",
        )


class ActivationAnalyzer:
    """Analyze activation patterns from model inference."""

    def __init__(self):
        self.patterns: list[ActivationPattern] = []

    def record_pattern(
        self,
        prompt: str,
        layer: str,
        activations: list[float],
        token_positions: Optional[list[int]] = None,
        attention_weights: Optional[list[list[float]]] = None,
    ) -> ActivationPattern:
        """Record an activation pattern."""
        if token_positions is None:
            token_positions = list(range(len(activations)))

        pattern = ActivationPattern(
            prompt=prompt,
            layer=layer,
            activations=activations,
            token_positions=token_positions,
            attention_weights=attention_weights,
        )

        self.patterns.append(pattern)
        return pattern

    def compare_patterns(
        self,
        pattern1: ActivationPattern,
        pattern2: ActivationPattern,
    ) -> dict[str, float]:
        """Compare two activation patterns."""
        # Ensure same dimension
        min_len = min(len(pattern1.activations), len(pattern2.activations))
        act1 = pattern1.activations[:min_len]
        act2 = pattern2.activations[:min_len]

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(act1, act2))
        norm1 = math.sqrt(sum(a**2 for a in act1))
        norm2 = math.sqrt(sum(b**2 for b in act2))

        cosine_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

        # L2 distance
        l2_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(act1, act2)))

        # Mean absolute difference
        mad = sum(abs(a - b) for a, b in zip(act1, act2)) / len(act1) if act1 else 0.0

        return {
            "cosine_similarity": cosine_sim,
            "l2_distance": l2_dist,
            "mean_absolute_diff": mad,
            "correlation": self._pearson_correlation(act1, act2),
        }

    def _pearson_correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) < 2:
            return 0.0

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        denominator = math.sqrt(var_x * var_y)

        return numerator / denominator if denominator > 0 else 0.0

    def find_salient_positions(
        self,
        pattern: ActivationPattern,
        threshold: float = 2.0,  # Standard deviations above mean
    ) -> list[int]:
        """Find positions with unusually high activations."""
        if not pattern.activations:
            return []

        mean = pattern.mean_activation
        std = math.sqrt(pattern.activation_variance)

        if std == 0:
            return []

        salient = []
        for i, act in enumerate(pattern.activations):
            if abs(act - mean) > threshold * std:
                salient.append(i)

        return salient

    def cluster_patterns(
        self,
        patterns: list[ActivationPattern],
        n_clusters: int = 3,
    ) -> dict[int, list[ActivationPattern]]:
        """Simple k-means-like clustering of patterns."""
        if not patterns or n_clusters <= 0:
            return {}

        n_clusters = min(n_clusters, len(patterns))

        # Initialize centroids from first n patterns
        centroids = [p.activations[:] for p in patterns[:n_clusters]]

        # Pad or truncate to same length
        max_len = max(len(c) for c in centroids)
        for c in centroids:
            while len(c) < max_len:
                c.append(0.0)

        clusters: dict[int, list[ActivationPattern]] = defaultdict(list)

        # Assign patterns to nearest centroid
        for pattern in patterns:
            act = pattern.activations[:]
            while len(act) < max_len:
                act.append(0.0)
            act = act[:max_len]

            min_dist = float("inf")
            best_cluster = 0

            for i, centroid in enumerate(centroids):
                dist = sum((a - c) ** 2 for a, c in zip(act, centroid))
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = i

            clusters[best_cluster].append(pattern)

        return dict(clusters)


class BehavioralShiftMeasurer:
    """Measure behavioral shifts from steering interventions."""

    def __init__(self):
        self.dimension_keywords: dict[str, dict[str, list[str]]] = {
            "formality": {
                "formal": ["therefore", "thus", "consequently", "moreover", "furthermore"],
                "informal": ["yeah", "gonna", "kinda", "stuff", "like"],
            },
            "length": {
                "verbose": ["additionally", "furthermore", "in addition", "also"],
                "concise": [],  # Measured by actual length
            },
            "sentiment": {
                "positive": ["great", "excellent", "wonderful", "amazing", "fantastic"],
                "negative": ["unfortunately", "however", "but", "problem", "issue"],
            },
            "confidence": {
                "confident": ["certainly", "definitely", "clearly", "obviously", "undoubtedly"],
                "uncertain": ["perhaps", "maybe", "possibly", "might", "could"],
            },
        }

    def measure_shift(
        self,
        original_output: str,
        steered_output: str,
        dimension: str,
    ) -> float:
        """Measure the behavioral shift along a dimension."""
        if dimension == "length":
            return self._measure_length_shift(original_output, steered_output)

        elif dimension in self.dimension_keywords:
            return self._measure_keyword_shift(original_output, steered_output, dimension)

        else:
            # Generic similarity-based measurement
            return self._measure_generic_shift(original_output, steered_output)

    def _measure_length_shift(
        self,
        original: str,
        steered: str,
    ) -> float:
        """Measure shift in response length."""
        orig_len = len(original.split())
        steered_len = len(steered.split())

        if orig_len == 0:
            return 1.0 if steered_len > 0 else 0.0

        # Normalize to 0-1 range
        ratio = steered_len / orig_len
        # Convert ratio to shift measure (1.0 = doubled, -1.0 = halved)
        return min(1.0, max(-1.0, (ratio - 1.0)))

    def _measure_keyword_shift(
        self,
        original: str,
        steered: str,
        dimension: str,
    ) -> float:
        """Measure shift using keyword presence."""
        keywords = self.dimension_keywords.get(dimension, {})

        orig_lower = original.lower()
        steered_lower = steered.lower()

        # Count positive keywords
        positive_keys = keywords.get(list(keywords.keys())[0], []) if keywords else []
        negative_keys = keywords.get(list(keywords.keys())[1], []) if len(keywords) > 1 else []

        orig_pos = sum(1 for k in positive_keys if k in orig_lower)
        orig_neg = sum(1 for k in negative_keys if k in orig_lower)
        steered_pos = sum(1 for k in positive_keys if k in steered_lower)
        steered_neg = sum(1 for k in negative_keys if k in steered_lower)

        # Calculate shift
        orig_score = (orig_pos - orig_neg) / max(1, len(positive_keys) + len(negative_keys))
        steered_score = (steered_pos - steered_neg) / max(
            1, len(positive_keys) + len(negative_keys)
        )

        return steered_score - orig_score

    def _measure_generic_shift(
        self,
        original: str,
        steered: str,
    ) -> float:
        """Measure generic textual shift using basic metrics."""
        # Word overlap
        orig_words = set(original.lower().split())
        steered_words = set(steered.lower().split())

        if not orig_words and not steered_words:
            return 0.0

        intersection = len(orig_words & steered_words)
        union = len(orig_words | steered_words)

        jaccard = intersection / union if union > 0 else 0.0

        # Shift is inverse of similarity (more different = more shift)
        return 1.0 - jaccard

    def detect_side_effects(
        self,
        original_output: str,
        steered_output: str,
        target_dimension: str,
    ) -> list[str]:
        """Detect unintended side effects of steering."""
        side_effects = []

        # Check all dimensions except target
        for dimension in self.dimension_keywords:
            if dimension == target_dimension:
                continue

            shift = abs(self.measure_shift(original_output, steered_output, dimension))
            if shift > 0.3:  # Threshold for significant side effect
                side_effects.append(f"Unintended shift in {dimension}: {shift:.2f}")

        # Check for dramatic length changes
        length_shift = self._measure_length_shift(original_output, steered_output)
        if target_dimension != "length" and abs(length_shift) > 0.5:
            side_effects.append(f"Significant length change: {length_shift:.2f}")

        return side_effects


class SteeringExperimenter:
    """Run steering experiments to understand model controllability."""

    def __init__(
        self,
        model_fn: Optional[Callable[[str], str]] = None,
    ):
        self.model_fn = model_fn
        self.steerer = PromptSteerer()
        self.measurer = BehavioralShiftMeasurer()
        self.experiments: list[SteeringExperiment] = []

    def set_model(self, model_fn: Callable[[str], str]) -> None:
        """Set the model function for experiments."""
        self.model_fn = model_fn

    def run_experiment(
        self,
        prompt: str,
        method: SteeringMethod,
        config: dict[str, Any],
        target_dimension: str = "generic",
    ) -> SteeringExperiment:
        """Run a single steering experiment."""
        if self.model_fn is None:
            raise ValueError("Model function not set. Call set_model first.")

        # Get original output
        original_output = self.model_fn(prompt)

        # Apply steering
        steered_prompt = self.steerer.apply_steering(prompt, method, config)

        # Handle system message case
        if isinstance(steered_prompt, tuple):
            # For system message, we'd need a different model interface
            # For now, concatenate them
            steered_prompt = f"{steered_prompt[0]}\n\n{steered_prompt[1]}"

        # Get steered output
        steered_output = self.model_fn(steered_prompt)

        # Measure behavioral shift
        behavioral_shift = self.measurer.measure_shift(
            original_output, steered_output, target_dimension
        )

        # Estimate direction alignment (how well the shift matched intent)
        direction_alignment = min(1.0, abs(behavioral_shift))

        # Detect side effects
        side_effects = self.measurer.detect_side_effects(
            original_output, steered_output, target_dimension
        )

        experiment = SteeringExperiment(
            original_prompt=prompt,
            steering_method=method,
            steering_config=config,
            original_output=original_output,
            steered_output=steered_output,
            behavioral_shift=abs(behavioral_shift),
            direction_alignment=direction_alignment,
            side_effects=side_effects,
        )

        self.experiments.append(experiment)
        return experiment

    def run_method_comparison(
        self,
        prompt: str,
        target_dimension: str,
        methods: Optional[list[SteeringMethod]] = None,
    ) -> list[SteeringExperiment]:
        """Compare different steering methods on the same prompt."""
        if methods is None:
            methods = [
                SteeringMethod.PROMPT_PREFIX,
                SteeringMethod.PROMPT_SUFFIX,
                SteeringMethod.FEW_SHOT,
            ]

        # Get appropriate config for each method
        instruction = self.steerer.get_preset_steering(target_dimension)
        if not instruction:
            instruction = f"Please be more {target_dimension} in your response."

        results = []
        for method in methods:
            if method == SteeringMethod.FEW_SHOT:
                config = {
                    "examples": [
                        ("Hello", f"Hello! I'll be {target_dimension} today."),
                        ("Help me", f"Of course! In a {target_dimension} manner..."),
                    ]
                }
            elif method == SteeringMethod.SYSTEM_MESSAGE:
                config = {"system_message": instruction}
            else:
                config = {"instruction": instruction}

            experiment = self.run_experiment(prompt, method, config, target_dimension)
            results.append(experiment)

        return results

    def analyze_steerability(
        self,
        prompts: list[str],
        dimensions: Optional[list[str]] = None,
    ) -> SteeringReport:
        """Analyze overall model steerability."""
        if dimensions is None:
            dimensions = ["formality", "length", "sentiment", "confidence"]

        all_experiments = []
        method_scores: dict[SteeringMethod, list[float]] = defaultdict(list)
        dimension_scores: dict[str, list[float]] = defaultdict(list)

        for prompt in prompts:
            for dimension in dimensions:
                experiments = self.run_method_comparison(prompt, dimension)
                all_experiments.extend(experiments)

                for exp in experiments:
                    method_scores[exp.steering_method].append(exp.behavioral_shift)
                    dimension_scores[dimension].append(exp.behavioral_shift)

        # Calculate effective methods
        effective_methods = [
            (method, sum(scores) / len(scores) if scores else 0.0)
            for method, scores in method_scores.items()
        ]
        effective_methods.sort(key=lambda x: x[1], reverse=True)

        # Calculate dimensional controllability
        behavioral_dimensions = {
            dim: sum(scores) / len(scores) if scores else 0.0
            for dim, scores in dimension_scores.items()
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            effective_methods, behavioral_dimensions, all_experiments
        )

        return SteeringReport(
            experiments=all_experiments,
            vectors_extracted=[],  # Would need activation access
            effective_methods=effective_methods,
            behavioral_dimensions=behavioral_dimensions,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        effective_methods: list[tuple[SteeringMethod, float]],
        behavioral_dimensions: dict[str, float],
        experiments: list[SteeringExperiment],
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Best method
        if effective_methods:
            best_method, best_score = effective_methods[0]
            recommendations.append(
                f"Most effective steering method: {best_method.value} (avg shift: {best_score:.2f})"
            )

        # Most/least controllable dimensions
        if behavioral_dimensions:
            sorted_dims = sorted(behavioral_dimensions.items(), key=lambda x: x[1], reverse=True)
            most_controllable = sorted_dims[0]
            least_controllable = sorted_dims[-1]

            recommendations.append(
                f"Most controllable dimension: {most_controllable[0]} ({most_controllable[1]:.2f})"
            )
            recommendations.append(
                f"Least controllable dimension: {least_controllable[0]} "
                f"({least_controllable[1]:.2f})"
            )

        # Check for common side effects
        all_side_effects = []
        for exp in experiments:
            all_side_effects.extend(exp.side_effects)

        if all_side_effects:
            recommendations.append(
                f"Watch for side effects: {len(all_side_effects)} detected across experiments"
            )

        return recommendations


class RepresentationAnalyzer:
    """Analyze model representation spaces."""

    def __init__(self):
        self.representations: dict[str, list[float]] = {}

    def store_representation(
        self,
        key: str,
        representation: list[float],
    ) -> None:
        """Store a representation for later analysis."""
        self.representations[key] = representation

    def compute_similarity_matrix(
        self,
        keys: Optional[list[str]] = None,
    ) -> list[list[float]]:
        """Compute pairwise similarity matrix."""
        if keys is None:
            keys = list(self.representations.keys())

        n = len(keys)
        matrix = [[0.0] * n for _ in range(n)]

        for i, key1 in enumerate(keys):
            for j, key2 in enumerate(keys):
                if i == j:
                    matrix[i][j] = 1.0
                elif i < j:
                    sim = self._cosine_similarity(
                        self.representations.get(key1, []),
                        self.representations.get(key2, []),
                    )
                    matrix[i][j] = sim
                    matrix[j][i] = sim

        return matrix

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0

        min_len = min(len(vec1), len(vec2))
        v1 = vec1[:min_len]
        v2 = vec2[:min_len]

        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a**2 for a in v1))
        norm2 = math.sqrt(sum(b**2 for b in v2))

        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    def find_similar_representations(
        self,
        query_key: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find most similar representations to a query."""
        if query_key not in self.representations:
            return []

        query = self.representations[query_key]
        similarities = []

        for key, rep in self.representations.items():
            if key == query_key:
                continue
            sim = self._cosine_similarity(query, rep)
            similarities.append((key, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def project_to_2d(
        self,
        keys: Optional[list[str]] = None,
    ) -> list[tuple[str, float, float]]:
        """Simple 2D projection using first two principal directions."""
        if keys is None:
            keys = list(self.representations.keys())

        if not keys:
            return []

        # Get representations
        reps = [self.representations.get(k, []) for k in keys]

        if not reps or not reps[0]:
            return [(k, 0.0, 0.0) for k in keys]

        # Simple projection: use first two dimensions (or mean of groups)
        projections = []
        for key, rep in zip(keys, reps):
            x = rep[0] if len(rep) > 0 else 0.0
            y = rep[1] if len(rep) > 1 else 0.0
            projections.append((key, x, y))

        return projections


# Convenience functions


def extract_steering_vector(
    positive_activations: list[float],
    negative_activations: list[float],
    name: str,
    target_behavior: str,
    normalize: bool = True,
) -> SteeringVector:
    """Extract a steering vector from contrasting activations."""
    extractor = SteeringVectorExtractor(normalize=normalize)
    return extractor.extract_from_contrast_pair(
        positive_activations, negative_activations, name, target_behavior
    )


def create_contrast_pair(
    base_prompt: str,
    dimension: str,
    positive_style: Optional[str] = None,
    negative_style: Optional[str] = None,
) -> ContrastPair:
    """Create a contrast pair for steering analysis."""
    steerer = PromptSteerer()
    return steerer.create_contrast_pair(base_prompt, dimension, positive_style, negative_style)


def apply_prompt_steering(
    prompt: str,
    method: SteeringMethod,
    config: dict[str, Any],
) -> Union[str, tuple[str, str]]:
    """Apply prompt-based steering."""
    steerer = PromptSteerer()
    return steerer.apply_steering(prompt, method, config)


def measure_behavioral_shift(
    original_output: str,
    steered_output: str,
    dimension: str = "generic",
) -> float:
    """Measure behavioral shift between outputs."""
    measurer = BehavioralShiftMeasurer()
    return measurer.measure_shift(original_output, steered_output, dimension)


def analyze_activation_patterns(
    patterns: list[ActivationPattern],
) -> dict[str, Any]:
    """Analyze a list of activation patterns."""
    analyzer = ActivationAnalyzer()

    if not patterns:
        return {"error": "No patterns provided"}

    # Basic statistics
    mean_activations = [p.mean_activation for p in patterns]
    variances = [p.activation_variance for p in patterns]

    # Cluster patterns
    clusters = analyzer.cluster_patterns(patterns)

    # Find salient positions across patterns
    all_salient = []
    for p in patterns:
        salient = analyzer.find_salient_positions(p)
        all_salient.extend(salient)

    return {
        "num_patterns": len(patterns),
        "mean_activation_avg": sum(mean_activations) / len(mean_activations),
        "mean_activation_std": math.sqrt(
            sum((m - sum(mean_activations) / len(mean_activations)) ** 2 for m in mean_activations)
            / len(mean_activations)
        )
        if len(mean_activations) > 1
        else 0.0,
        "variance_avg": sum(variances) / len(variances),
        "num_clusters": len(clusters),
        "salient_positions": list(set(all_salient)),
    }


def quick_steering_analysis(
    model_fn: Callable[[str], str],
    test_prompts: list[str],
    dimensions: Optional[list[str]] = None,
) -> SteeringReport:
    """Quick analysis of model steerability."""
    experimenter = SteeringExperimenter(model_fn=model_fn)
    return experimenter.analyze_steerability(test_prompts, dimensions)
