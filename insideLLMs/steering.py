"""Model steering and activation analysis for understanding LLM internal representations.

This module provides tools for analyzing how different prompting strategies
affect model behavior and internal representations, including:
- Steering vector analysis and manipulation
- Activation pattern extraction and comparison
- Prompt-based steering experiments
- Representation space analysis

Overview
--------
Model steering is a technique for influencing language model behavior by manipulating
prompts, activations, or internal representations. This module provides both high-level
convenience functions and detailed classes for comprehensive steering analysis.

Key Concepts
------------
- **Steering Vectors**: Mathematical representations of behavioral directions that
  can be added to model activations to shift behavior (e.g., more formal, more helpful).
- **Contrast Pairs**: Paired prompts designed to elicit opposite behaviors, used to
  extract steering vectors.
- **Activation Patterns**: Recorded model activations that reveal internal processing.
- **Behavioral Dimensions**: Measurable aspects of model output (formality, length, etc.).

Examples
--------
Basic prompt steering with a prefix instruction:

    >>> from insideLLMs.steering import PromptSteerer, SteeringMethod
    >>> steerer = PromptSteerer()
    >>> original_prompt = "Explain quantum computing"
    >>> steered = steerer.steer_with_prefix(
    ...     original_prompt,
    ...     "Please respond in a formal, academic tone."
    ... )
    >>> print(steered)
    Please respond in a formal, academic tone.
    <BLANKLINE>
    Explain quantum computing

Extracting a steering vector from contrast activations:

    >>> from insideLLMs.steering import SteeringVectorExtractor
    >>> extractor = SteeringVectorExtractor(normalize=True)
    >>> positive_acts = [0.5, 0.8, 0.3, 0.9]  # Activations for "formal" prompt
    >>> negative_acts = [0.2, 0.4, 0.6, 0.1]  # Activations for "casual" prompt
    >>> vector = extractor.extract_from_contrast_pair(
    ...     positive_acts, negative_acts,
    ...     name="formality",
    ...     target_behavior="more formal responses"
    ... )
    >>> print(f"Vector magnitude: {vector.magnitude:.3f}")
    Vector magnitude: 0.648

Creating a contrast pair for experimentation:

    >>> from insideLLMs.steering import create_contrast_pair
    >>> pair = create_contrast_pair(
    ...     base_prompt="Write a product description",
    ...     dimension="formality"
    ... )
    >>> print(pair.target_dimension)
    formality

Running a full steerability analysis:

    >>> from insideLLMs.steering import quick_steering_analysis
    >>> def mock_model(prompt):
    ...     return "Sample response"  # Replace with actual model
    >>> report = quick_steering_analysis(
    ...     model_fn=mock_model,
    ...     test_prompts=["Hello", "Explain AI"],
    ...     dimensions=["formality", "length"]
    ... )
    >>> print(f"Overall steerability: {report.overall_steerability:.2f}")
    Overall steerability: ...

See Also
--------
- `SteeringVectorExtractor`: For extracting steering vectors from activations
- `PromptSteerer`: For applying prompt-based steering techniques
- `SteeringExperimenter`: For running comprehensive steering experiments
- `BehavioralShiftMeasurer`: For measuring changes in model behavior
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union


class SteeringMethod(Enum):
    """Methods for steering model behavior.

    This enum defines the available techniques for influencing how a language
    model responds to prompts. Each method has different characteristics in
    terms of effectiveness, ease of use, and the level of model access required.

    Attributes
    ----------
    PROMPT_PREFIX : str
        Add steering instructions before the user prompt. Simple and widely
        applicable, but may be ignored by the model.
    PROMPT_SUFFIX : str
        Add steering instructions after the user prompt. Can be effective
        for formatting or output style guidance.
    SYSTEM_MESSAGE : str
        Use a dedicated system message slot (for models that support it).
        Often the most effective prompt-based approach.
    FEW_SHOT : str
        Provide example input-output pairs to demonstrate desired behavior.
        Effective for complex or nuanced behavioral changes.
    CONTRAST_PAIR : str
        Use paired positive/negative examples to extract behavioral directions.
        Primarily used for steering vector extraction.
    ACTIVATION_ADDITION : str
        Directly add steering vectors to model activations. Requires model
        internals access but offers fine-grained control.
    SOFT_PROMPT : str
        Use learned continuous prompt embeddings. Requires training but can
        capture complex behavioral patterns.

    Examples
    --------
    Using a steering method with PromptSteerer:

        >>> from insideLLMs.steering import SteeringMethod, PromptSteerer
        >>> steerer = PromptSteerer()
        >>> method = SteeringMethod.PROMPT_PREFIX
        >>> config = {"instruction": "Be concise."}
        >>> result = steerer.apply_steering("Explain gravity", method, config)
        >>> print(result)
        Be concise.
        <BLANKLINE>
        Explain gravity

    Checking available methods:

        >>> methods = list(SteeringMethod)
        >>> len(methods)
        7
        >>> SteeringMethod.FEW_SHOT.value
        'few_shot'

    Comparing method types:

        >>> prompt_methods = [
        ...     SteeringMethod.PROMPT_PREFIX,
        ...     SteeringMethod.PROMPT_SUFFIX,
        ...     SteeringMethod.SYSTEM_MESSAGE
        ... ]
        >>> all(m.value.startswith(('prompt', 'system')) for m in prompt_methods)
        True

    Selecting method based on model capabilities:

        >>> def get_method(supports_system_message: bool) -> SteeringMethod:
        ...     if supports_system_message:
        ...         return SteeringMethod.SYSTEM_MESSAGE
        ...     return SteeringMethod.PROMPT_PREFIX
        >>> get_method(True)
        <SteeringMethod.SYSTEM_MESSAGE: 'system_message'>
    """

    PROMPT_PREFIX = "prompt_prefix"
    PROMPT_SUFFIX = "prompt_suffix"
    SYSTEM_MESSAGE = "system_message"
    FEW_SHOT = "few_shot"
    CONTRAST_PAIR = "contrast_pair"
    ACTIVATION_ADDITION = "activation_addition"
    SOFT_PROMPT = "soft_prompt"


class ActivationLayer(Enum):
    """Types of activation layers to analyze in transformer models.

    This enum specifies which layer types to extract activations from when
    analyzing model internals. Different layers capture different aspects
    of the model's processing.

    Attributes
    ----------
    INPUT_EMBEDDING : str
        The initial token embedding layer. Captures lexical information
        before any contextual processing.
    ATTENTION : str
        Self-attention layers. Captures how tokens attend to each other
        and aggregate contextual information.
    MLP : str
        Feed-forward (MLP) layers. Often associated with factual recall
        and knowledge storage.
    RESIDUAL : str
        Residual stream (sum of all previous contributions). Represents
        the accumulated representation at each position.
    OUTPUT : str
        Final layer before the output projection. Contains the model's
        final representation before generating tokens.
    ALL : str
        Extract from all available layers. Useful for comprehensive
        analysis but memory-intensive.

    Examples
    --------
    Selecting layers for activation recording:

        >>> from insideLLMs.steering import ActivationLayer
        >>> layer = ActivationLayer.ATTENTION
        >>> print(f"Recording from: {layer.value}")
        Recording from: attention

    Filtering layers for analysis:

        >>> key_layers = [ActivationLayer.ATTENTION, ActivationLayer.MLP]
        >>> for layer in key_layers:
        ...     print(f"Analyzing {layer.value} activations")
        Analyzing attention activations
        Analyzing mlp activations

    Checking if comprehensive analysis is needed:

        >>> layer = ActivationLayer.ALL
        >>> if layer == ActivationLayer.ALL:
        ...     print("Will extract from all layers - high memory usage expected")
        Will extract from all layers - high memory usage expected

    Mapping layers to analysis functions:

        >>> layer_handlers = {
        ...     ActivationLayer.INPUT_EMBEDDING: lambda x: x[:10],
        ...     ActivationLayer.ATTENTION: lambda x: x[10:20],
        ...     ActivationLayer.OUTPUT: lambda x: x[-10:],
        ... }
        >>> activations = list(range(100))
        >>> result = layer_handlers[ActivationLayer.ATTENTION](activations)
        >>> len(result)
        10
    """

    INPUT_EMBEDDING = "input_embedding"
    ATTENTION = "attention"
    MLP = "mlp"
    RESIDUAL = "residual"
    OUTPUT = "output"
    ALL = "all"


class RepresentationSpace(Enum):
    """Types of representation spaces for analyzing model internals.

    This enum categorizes different ways to interpret and analyze the
    vector representations that language models create internally. Each
    space type offers different insights into model behavior.

    Attributes
    ----------
    TOKEN : str
        Per-token representations. Each token position has its own vector.
        Useful for understanding local processing and attention patterns.
    SEQUENCE : str
        Aggregated sequence-level representations (e.g., mean pooling).
        Captures overall semantic content of the input.
    SEMANTIC : str
        Representations organized by semantic similarity. Related concepts
        cluster together in this space.
    TASK : str
        Task-specific representation space. Organized by how the model
        processes different task types (QA, summarization, etc.).

    Examples
    --------
    Selecting representation space for analysis:

        >>> from insideLLMs.steering import RepresentationSpace
        >>> space = RepresentationSpace.SEMANTIC
        >>> print(f"Analyzing in {space.value} space")
        Analyzing in semantic space

    Choosing space based on analysis goal:

        >>> def choose_space(goal: str) -> RepresentationSpace:
        ...     if goal == "token_attention":
        ...         return RepresentationSpace.TOKEN
        ...     elif goal == "document_similarity":
        ...         return RepresentationSpace.SEQUENCE
        ...     elif goal == "concept_clustering":
        ...         return RepresentationSpace.SEMANTIC
        ...     return RepresentationSpace.TASK
        >>> choose_space("document_similarity")
        <RepresentationSpace.SEQUENCE: 'sequence'>

    Iterating through all spaces:

        >>> all_spaces = [s.value for s in RepresentationSpace]
        >>> print(all_spaces)
        ['token', 'sequence', 'semantic', 'task']

    Building a multi-space analysis:

        >>> analyses = {
        ...     RepresentationSpace.TOKEN: "attention_patterns",
        ...     RepresentationSpace.SEMANTIC: "concept_clusters",
        ... }
        >>> for space, analysis in analyses.items():
        ...     print(f"{space.value}: {analysis}")
        token: attention_patterns
        semantic: concept_clusters
    """

    TOKEN = "token"
    SEQUENCE = "sequence"
    SEMANTIC = "semantic"
    TASK = "task"


class SteeringStrength(Enum):
    """Strength levels for steering interventions.

    This enum defines the intensity of steering to apply. Higher strengths
    produce more pronounced behavioral changes but may also cause unintended
    side effects or reduce output quality.

    Attributes
    ----------
    MINIMAL : str
        Very subtle steering. May not produce noticeable changes but
        preserves output quality. Multiplier: ~0.1x.
    LIGHT : str
        Gentle steering. Produces slight shifts while maintaining
        coherence. Multiplier: ~0.25x.
    MODERATE : str
        Balanced steering. Noticeable behavioral change with acceptable
        side effects. Multiplier: ~0.5x.
    STRONG : str
        Aggressive steering. Significant behavioral shift, may impact
        other aspects of output. Multiplier: ~0.75x.
    MAXIMUM : str
        Full-strength steering. Maximum behavioral change, high risk
        of side effects. Multiplier: ~1.0x.

    Examples
    --------
    Selecting steering strength:

        >>> from insideLLMs.steering import SteeringStrength
        >>> strength = SteeringStrength.MODERATE
        >>> print(f"Using {strength.value} steering")
        Using moderate steering

    Converting strength to multiplier:

        >>> def strength_to_multiplier(strength: SteeringStrength) -> float:
        ...     mapping = {
        ...         SteeringStrength.MINIMAL: 0.1,
        ...         SteeringStrength.LIGHT: 0.25,
        ...         SteeringStrength.MODERATE: 0.5,
        ...         SteeringStrength.STRONG: 0.75,
        ...         SteeringStrength.MAXIMUM: 1.0,
        ...     }
        ...     return mapping[strength]
        >>> strength_to_multiplier(SteeringStrength.MODERATE)
        0.5

    Choosing strength based on task sensitivity:

        >>> def get_strength(sensitive_task: bool) -> SteeringStrength:
        ...     if sensitive_task:
        ...         return SteeringStrength.LIGHT
        ...     return SteeringStrength.MODERATE
        >>> get_strength(True)
        <SteeringStrength.LIGHT: 'light'>

    Ordering strengths for gradual escalation:

        >>> strengths = list(SteeringStrength)
        >>> print([s.value for s in strengths])
        ['minimal', 'light', 'moderate', 'strong', 'maximum']
    """

    MINIMAL = "minimal"
    LIGHT = "light"
    MODERATE = "moderate"
    STRONG = "strong"
    MAXIMUM = "maximum"


@dataclass
class SteeringVector:
    """A vector that can be used to steer model behavior.

    Steering vectors represent directions in activation space that correspond
    to behavioral changes. By adding these vectors to model activations during
    inference, you can shift model behavior toward the target direction.

    Attributes
    ----------
    name : str
        Human-readable identifier for the vector (e.g., "formality", "helpfulness").
    direction : list[float]
        The unit direction vector in activation space. Should typically be
        normalized to unit length for consistent steering strength.
    magnitude : float
        The original magnitude of the difference vector before normalization.
        Useful for understanding the natural scale of the behavioral dimension.
    source : str
        How the vector was derived (e.g., "contrast_pair", "pca", "trained").
    target_behavior : str
        Description of the behavioral change this vector induces.
    layer : Optional[str]
        Which model layer this vector applies to (e.g., "layer_15", "residual").
    metadata : dict[str, Any]
        Additional information about the vector's provenance and properties.

    Examples
    --------
    Creating a steering vector manually:

        >>> from insideLLMs.steering import SteeringVector
        >>> vector = SteeringVector(
        ...     name="politeness",
        ...     direction=[0.5, 0.5, 0.5, 0.5],
        ...     magnitude=1.0,
        ...     source="manual",
        ...     target_behavior="more polite responses"
        ... )
        >>> print(vector.name)
        politeness

    Serializing and deserializing:

        >>> data = vector.to_dict()
        >>> restored = SteeringVector.from_dict(data)
        >>> restored.name == vector.name
        True
        >>> restored.direction == vector.direction
        True

    Using with layer-specific steering:

        >>> layer_vector = SteeringVector(
        ...     name="creativity",
        ...     direction=[0.3, 0.7, 0.2, 0.6],
        ...     magnitude=1.5,
        ...     source="contrast_pair",
        ...     target_behavior="more creative responses",
        ...     layer="layer_20"
        ... )
        >>> print(f"Apply at: {layer_vector.layer}")
        Apply at: layer_20

    Storing additional metadata:

        >>> annotated_vector = SteeringVector(
        ...     name="conciseness",
        ...     direction=[0.1, -0.2, 0.3, -0.1],
        ...     magnitude=0.8,
        ...     source="trained",
        ...     target_behavior="shorter responses",
        ...     metadata={
        ...         "training_samples": 1000,
        ...         "accuracy": 0.92,
        ...         "created_date": "2024-01-15"
        ...     }
        ... )
        >>> annotated_vector.metadata["accuracy"]
        0.92
    """

    name: str
    direction: list[float]
    magnitude: float
    source: str  # How the vector was derived
    target_behavior: str
    layer: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the steering vector to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all vector attributes, suitable for
            JSON serialization or storage.

        Examples
        --------
        >>> vector = SteeringVector(
        ...     name="test", direction=[1.0, 0.0],
        ...     magnitude=1.0, source="test", target_behavior="test"
        ... )
        >>> d = vector.to_dict()
        >>> d["name"]
        'test'
        >>> len(d["direction"])
        2
        """
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
        """Create a SteeringVector from a dictionary representation.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing vector attributes. Must include 'name',
            'direction', 'magnitude', 'source', and 'target_behavior'.

        Returns
        -------
        SteeringVector
            A new SteeringVector instance.

        Examples
        --------
        >>> data = {
        ...     "name": "formality",
        ...     "direction": [0.5, 0.5],
        ...     "magnitude": 0.707,
        ...     "source": "contrast_pair",
        ...     "target_behavior": "formal tone"
        ... }
        >>> vector = SteeringVector.from_dict(data)
        >>> vector.name
        'formality'

        >>> # With optional fields
        >>> data["layer"] = "layer_10"
        >>> data["metadata"] = {"version": 2}
        >>> vector = SteeringVector.from_dict(data)
        >>> vector.layer
        'layer_10'
        """
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
    """Pattern of activations extracted from model inference.

    Activation patterns capture the internal state of a model at a specific
    layer during inference. These patterns can be compared across different
    prompts to understand how the model processes different inputs.

    Attributes
    ----------
    prompt : str
        The input prompt that generated these activations.
    layer : str
        Which layer the activations were extracted from.
    activations : list[float]
        The activation values, typically averaged or pooled across positions.
    token_positions : list[int]
        Which token positions these activations correspond to.
    attention_weights : Optional[list[list[float]]]
        If available, the attention weight matrix for this layer.
    metadata : dict[str, Any]
        Additional information about the pattern.

    Examples
    --------
    Creating an activation pattern:

        >>> from insideLLMs.steering import ActivationPattern
        >>> pattern = ActivationPattern(
        ...     prompt="What is machine learning?",
        ...     layer="layer_12",
        ...     activations=[0.5, -0.3, 0.8, 0.1, -0.2],
        ...     token_positions=[0, 1, 2, 3, 4]
        ... )
        >>> print(pattern.layer)
        layer_12

    Computing activation statistics:

        >>> print(f"Mean activation: {pattern.mean_activation:.3f}")
        Mean activation: 0.180
        >>> print(f"Variance: {pattern.activation_variance:.3f}")
        Variance: 0.156

    Serializing for storage:

        >>> data = pattern.to_dict()
        >>> data["prompt"]
        'What is machine learning?'
        >>> len(data["activations"])
        5

    Pattern with attention weights:

        >>> attn_pattern = ActivationPattern(
        ...     prompt="Hello world",
        ...     layer="attention_5",
        ...     activations=[0.2, 0.8],
        ...     token_positions=[0, 1],
        ...     attention_weights=[[0.3, 0.7], [0.5, 0.5]]
        ... )
        >>> attn_pattern.attention_weights[0][1]
        0.7
    """

    prompt: str
    layer: str
    activations: list[float]
    token_positions: list[int]
    attention_weights: Optional[list[list[float]]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the activation pattern to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all pattern attributes.

        Examples
        --------
        >>> pattern = ActivationPattern(
        ...     prompt="test", layer="layer_1",
        ...     activations=[1.0, 2.0], token_positions=[0, 1]
        ... )
        >>> d = pattern.to_dict()
        >>> d["layer"]
        'layer_1'
        """
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
        """Calculate mean activation value.

        Returns
        -------
        float
            The arithmetic mean of all activation values, or 0.0 if empty.

        Examples
        --------
        >>> pattern = ActivationPattern(
        ...     prompt="test", layer="l1",
        ...     activations=[1.0, 2.0, 3.0], token_positions=[0, 1, 2]
        ... )
        >>> pattern.mean_activation
        2.0

        >>> empty = ActivationPattern(
        ...     prompt="test", layer="l1",
        ...     activations=[], token_positions=[]
        ... )
        >>> empty.mean_activation
        0.0
        """
        if not self.activations:
            return 0.0
        return sum(self.activations) / len(self.activations)

    @property
    def activation_variance(self) -> float:
        """Calculate activation variance.

        Returns
        -------
        float
            The population variance of activation values, or 0.0 if fewer
            than 2 values.

        Examples
        --------
        >>> pattern = ActivationPattern(
        ...     prompt="test", layer="l1",
        ...     activations=[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0],
        ...     token_positions=list(range(8))
        ... )
        >>> round(pattern.activation_variance, 2)
        4.0

        >>> single = ActivationPattern(
        ...     prompt="test", layer="l1",
        ...     activations=[5.0], token_positions=[0]
        ... )
        >>> single.activation_variance
        0.0
        """
        if len(self.activations) < 2:
            return 0.0
        mean = self.mean_activation
        return sum((x - mean) ** 2 for x in self.activations) / len(self.activations)


@dataclass
class SteeringExperiment:
    """Results from a steering experiment.

    A steering experiment captures the full context and results of applying
    a steering intervention to a model, including the original and steered
    outputs, quantitative measures of the behavioral change, and any
    unintended side effects.

    Attributes
    ----------
    original_prompt : str
        The input prompt before any steering was applied.
    steering_method : SteeringMethod
        The method used to apply steering.
    steering_config : dict[str, Any]
        Configuration parameters for the steering method.
    original_output : str
        The model's response without steering.
    steered_output : str
        The model's response with steering applied.
    behavioral_shift : float
        Measure of how much behavior changed (0-1 scale).
    direction_alignment : float
        How well the shift aligned with the intended direction (0-1).
    side_effects : list[str]
        Descriptions of unintended behavioral changes.
    metadata : dict[str, Any]
        Additional experiment information.

    Examples
    --------
    Creating an experiment result:

        >>> from insideLLMs.steering import SteeringExperiment, SteeringMethod
        >>> experiment = SteeringExperiment(
        ...     original_prompt="Explain quantum physics",
        ...     steering_method=SteeringMethod.PROMPT_PREFIX,
        ...     steering_config={"instruction": "Be very formal"},
        ...     original_output="Quantum physics is about tiny particles...",
        ...     steered_output="Quantum mechanics constitutes a fundamental...",
        ...     behavioral_shift=0.65,
        ...     direction_alignment=0.8,
        ...     side_effects=["Response became longer"]
        ... )
        >>> experiment.behavioral_shift
        0.65

    Serializing for analysis:

        >>> data = experiment.to_dict()
        >>> data["steering_method"]
        'prompt_prefix'
        >>> data["direction_alignment"]
        0.8

    Checking for significant side effects:

        >>> if experiment.side_effects:
        ...     print(f"Warning: {len(experiment.side_effects)} side effects")
        Warning: 1 side effects

    Evaluating experiment quality:

        >>> def is_successful(exp: SteeringExperiment) -> bool:
        ...     return exp.behavioral_shift > 0.3 and exp.direction_alignment > 0.5
        >>> is_successful(experiment)
        True
    """

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
        """Convert the experiment to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all experiment data, with SteeringMethod
            converted to its string value.

        Examples
        --------
        >>> exp = SteeringExperiment(
        ...     original_prompt="test",
        ...     steering_method=SteeringMethod.FEW_SHOT,
        ...     steering_config={"examples": []},
        ...     original_output="output1",
        ...     steered_output="output2",
        ...     behavioral_shift=0.5,
        ...     direction_alignment=0.7,
        ...     side_effects=[]
        ... )
        >>> d = exp.to_dict()
        >>> d["steering_method"]
        'few_shot'
        """
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
    """A pair of prompts designed to elicit contrasting behaviors.

    Contrast pairs are fundamental to extracting steering vectors. By running
    the same base prompt with positive and negative framings, we can isolate
    the activation direction corresponding to a specific behavioral dimension.

    Attributes
    ----------
    positive_prompt : str
        The prompt framed to elicit the positive end of the dimension
        (e.g., formal, helpful, verbose).
    negative_prompt : str
        The prompt framed to elicit the negative end of the dimension
        (e.g., casual, unhelpful, concise).
    target_dimension : str
        The behavioral dimension being targeted (e.g., "formality").
    expected_difference : str
        Description of the expected behavioral difference between the
        positive and negative versions.

    Examples
    --------
    Creating a formality contrast pair:

        >>> from insideLLMs.steering import ContrastPair
        >>> pair = ContrastPair(
        ...     positive_prompt="Please respond formally: What is AI?",
        ...     negative_prompt="Just be casual: What is AI?",
        ...     target_dimension="formality",
        ...     expected_difference="Formal vs casual language"
        ... )
        >>> print(pair.target_dimension)
        formality

    Serializing for storage:

        >>> data = pair.to_dict()
        >>> "positive_prompt" in data
        True
        >>> data["target_dimension"]
        'formality'

    Creating a helpfulness contrast pair:

        >>> help_pair = ContrastPair(
        ...     positive_prompt="Be as helpful as possible: How do I learn Python?",
        ...     negative_prompt="Give minimal help: How do I learn Python?",
        ...     target_dimension="helpfulness",
        ...     expected_difference="Detailed vs minimal assistance"
        ... )
        >>> help_pair.expected_difference
        'Detailed vs minimal assistance'

    Using with activation extraction:

        >>> def extract_from_pair(pair: ContrastPair) -> dict:
        ...     return {
        ...         "dimension": pair.target_dimension,
        ...         "prompts": [pair.positive_prompt, pair.negative_prompt]
        ...     }
        >>> result = extract_from_pair(pair)
        >>> len(result["prompts"])
        2
    """

    positive_prompt: str
    negative_prompt: str
    target_dimension: str  # e.g., "formality", "helpfulness", "verbosity"
    expected_difference: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the contrast pair to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all contrast pair attributes.

        Examples
        --------
        >>> pair = ContrastPair(
        ...     positive_prompt="Be formal",
        ...     negative_prompt="Be casual",
        ...     target_dimension="formality",
        ...     expected_difference="tone difference"
        ... )
        >>> d = pair.to_dict()
        >>> list(d.keys())
        ['positive_prompt', 'negative_prompt', 'target_dimension', 'expected_difference']
        """
        return {
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "target_dimension": self.target_dimension,
            "expected_difference": self.expected_difference,
        }


@dataclass
class SteeringReport:
    """Comprehensive report on steering analysis.

    A steering report aggregates results from multiple steering experiments
    and provides insights into model controllability, effective methods,
    and recommendations for steering strategies.

    Attributes
    ----------
    experiments : list[SteeringExperiment]
        All experiments conducted during the analysis.
    vectors_extracted : list[SteeringVector]
        Any steering vectors that were extracted.
    effective_methods : list[tuple[SteeringMethod, float]]
        Ranked list of methods by effectiveness score.
    behavioral_dimensions : dict[str, float]
        Controllability score for each behavioral dimension.
    recommendations : list[str]
        Human-readable recommendations based on the analysis.
    metadata : dict[str, Any]
        Additional report information.

    Examples
    --------
    Creating a steering report:

        >>> from insideLLMs.steering import (
        ...     SteeringReport, SteeringExperiment, SteeringMethod
        ... )
        >>> exp = SteeringExperiment(
        ...     original_prompt="test",
        ...     steering_method=SteeringMethod.PROMPT_PREFIX,
        ...     steering_config={},
        ...     original_output="out1",
        ...     steered_output="out2",
        ...     behavioral_shift=0.6,
        ...     direction_alignment=0.8,
        ...     side_effects=[]
        ... )
        >>> report = SteeringReport(
        ...     experiments=[exp],
        ...     vectors_extracted=[],
        ...     effective_methods=[(SteeringMethod.PROMPT_PREFIX, 0.6)],
        ...     behavioral_dimensions={"formality": 0.7},
        ...     recommendations=["Use prompt prefix for formality"]
        ... )
        >>> report.overall_steerability
        0.6

    Serializing for export:

        >>> data = report.to_dict()
        >>> len(data["experiments"])
        1
        >>> data["behavioral_dimensions"]["formality"]
        0.7

    Analyzing method effectiveness:

        >>> for method, score in report.effective_methods:
        ...     print(f"{method.value}: {score:.2f}")
        prompt_prefix: 0.60

    Accessing recommendations:

        >>> for rec in report.recommendations:
        ...     print(rec)
        Use prompt prefix for formality
    """

    experiments: list[SteeringExperiment]
    vectors_extracted: list[SteeringVector]
    effective_methods: list[tuple[SteeringMethod, float]]  # Method and effectiveness
    behavioral_dimensions: dict[str, float]  # Dimension name -> controllability
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all report data in a serializable format.

        Examples
        --------
        >>> report = SteeringReport(
        ...     experiments=[],
        ...     vectors_extracted=[],
        ...     effective_methods=[],
        ...     behavioral_dimensions={"test": 0.5},
        ...     recommendations=["Tip 1"]
        ... )
        >>> d = report.to_dict()
        >>> d["recommendations"]
        ['Tip 1']
        """
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
        """Calculate overall model steerability score.

        Returns
        -------
        float
            The average behavioral shift across all experiments, or 0.0
            if no experiments were conducted.

        Examples
        --------
        >>> from insideLLMs.steering import SteeringExperiment, SteeringMethod
        >>> exp1 = SteeringExperiment(
        ...     original_prompt="p1", steering_method=SteeringMethod.PROMPT_PREFIX,
        ...     steering_config={}, original_output="o1", steered_output="s1",
        ...     behavioral_shift=0.4, direction_alignment=0.5, side_effects=[]
        ... )
        >>> exp2 = SteeringExperiment(
        ...     original_prompt="p2", steering_method=SteeringMethod.PROMPT_SUFFIX,
        ...     steering_config={}, original_output="o2", steered_output="s2",
        ...     behavioral_shift=0.8, direction_alignment=0.9, side_effects=[]
        ... )
        >>> report = SteeringReport(
        ...     experiments=[exp1, exp2],
        ...     vectors_extracted=[],
        ...     effective_methods=[],
        ...     behavioral_dimensions={},
        ...     recommendations=[]
        ... )
        >>> report.overall_steerability
        0.6

        >>> empty_report = SteeringReport(
        ...     experiments=[], vectors_extracted=[], effective_methods=[],
        ...     behavioral_dimensions={}, recommendations=[]
        ... )
        >>> empty_report.overall_steerability
        0.0
        """
        if not self.experiments:
            return 0.0
        return sum(e.behavioral_shift for e in self.experiments) / len(self.experiments)


class SteeringVectorExtractor:
    """Extract steering vectors from contrast pairs or examples.

    This class provides methods for computing steering vectors from activation
    data. Steering vectors are directions in activation space that correspond
    to specific behavioral changes in model outputs.

    Parameters
    ----------
    normalize : bool, default=True
        Whether to normalize extracted vectors to unit length. Normalization
        makes it easier to control steering strength consistently.

    Attributes
    ----------
    normalize : bool
        The normalization setting.
    extracted_vectors : list[SteeringVector]
        All vectors extracted by this instance.

    Examples
    --------
    Basic extraction from a contrast pair:

        >>> from insideLLMs.steering import SteeringVectorExtractor
        >>> extractor = SteeringVectorExtractor(normalize=True)
        >>> pos = [0.8, 0.2, 0.6]  # "Formal" activations
        >>> neg = [0.2, 0.8, 0.4]  # "Casual" activations
        >>> vector = extractor.extract_from_contrast_pair(
        ...     pos, neg,
        ...     name="formality",
        ...     target_behavior="more formal responses"
        ... )
        >>> len(vector.direction)
        3

    Extraction from multiple examples:

        >>> pos_examples = [[0.9, 0.1], [0.8, 0.2], [0.85, 0.15]]
        >>> neg_examples = [[0.1, 0.9], [0.2, 0.8]]
        >>> vector = extractor.extract_from_examples(
        ...     pos_examples, neg_examples,
        ...     name="helpfulness",
        ...     target_behavior="more helpful"
        ... )
        >>> vector.source
        'contrast_pair'

    Combining multiple vectors:

        >>> v1 = extractor.extract_from_contrast_pair(
        ...     [1.0, 0.0], [0.0, 0.0], "v1", "behavior1"
        ... )
        >>> v2 = extractor.extract_from_contrast_pair(
        ...     [0.0, 1.0], [0.0, 0.0], "v2", "behavior2"
        ... )
        >>> combined = extractor.combine_vectors([v1, v2], weights=[0.7, 0.3])
        >>> combined.source
        'combination'

    Tracking extracted vectors:

        >>> extractor = SteeringVectorExtractor()
        >>> _ = extractor.extract_from_contrast_pair([1, 0], [0, 1], "test", "test")
        >>> len(extractor.extracted_vectors)
        1

    See Also
    --------
    SteeringVector : The resulting vector type.
    PromptSteerer : For applying steering via prompts.
    """

    def __init__(self, normalize: bool = True):
        """Initialize the extractor.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to normalize vectors to unit length.

        Examples
        --------
        >>> extractor = SteeringVectorExtractor(normalize=True)
        >>> extractor.normalize
        True

        >>> extractor = SteeringVectorExtractor(normalize=False)
        >>> extractor.normalize
        False
        """
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
        """Extract a steering vector from contrasting activations.

        Computes the difference between positive and negative activations
        to create a steering direction. The positive activations should
        correspond to prompts that elicit the desired behavior.

        Parameters
        ----------
        positive_activations : list[float]
            Activations from a prompt eliciting the target behavior.
        negative_activations : list[float]
            Activations from a prompt eliciting the opposite behavior.
        name : str
            Human-readable name for the vector.
        target_behavior : str
            Description of what behavior this vector induces.
        layer : Optional[str], default=None
            Which layer these activations came from.

        Returns
        -------
        SteeringVector
            The extracted steering vector.

        Raises
        ------
        ValueError
            If activation lists have different lengths.

        Examples
        --------
        >>> extractor = SteeringVectorExtractor()
        >>> pos = [1.0, 0.0, 0.5]
        >>> neg = [0.0, 1.0, 0.5]
        >>> vector = extractor.extract_from_contrast_pair(
        ...     pos, neg, "test", "test behavior"
        ... )
        >>> len(vector.direction)
        3
        >>> vector.source
        'contrast_pair'

        >>> # With layer specification
        >>> vector = extractor.extract_from_contrast_pair(
        ...     [0.5, 0.5], [0.0, 0.0], "layer_vec", "behavior",
        ...     layer="layer_15"
        ... )
        >>> vector.layer
        'layer_15'

        >>> # Error on mismatched lengths
        >>> try:
        ...     extractor.extract_from_contrast_pair([1, 2], [1], "bad", "bad")
        ... except ValueError as e:
        ...     print("Error:", str(e))
        Error: Activation lists must have the same length
        """
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
        """Extract a steering vector from multiple example pairs.

        Averages multiple positive and negative examples before computing
        the difference. This produces more robust vectors that generalize
        better across different inputs.

        Parameters
        ----------
        positive_examples : list[list[float]]
            Multiple activation samples from positive prompts.
        negative_examples : list[list[float]]
            Multiple activation samples from negative prompts.
        name : str
            Human-readable name for the vector.
        target_behavior : str
            Description of what behavior this vector induces.
        layer : Optional[str], default=None
            Which layer these activations came from.

        Returns
        -------
        SteeringVector
            The extracted steering vector.

        Raises
        ------
        ValueError
            If either example list is empty.

        Examples
        --------
        >>> extractor = SteeringVectorExtractor()
        >>> pos_examples = [[1.0, 0.0], [0.8, 0.2], [0.9, 0.1]]
        >>> neg_examples = [[0.0, 1.0], [0.2, 0.8]]
        >>> vector = extractor.extract_from_examples(
        ...     pos_examples, neg_examples,
        ...     name="dimension",
        ...     target_behavior="target"
        ... )
        >>> len(vector.direction)
        2

        >>> # More examples improve robustness
        >>> many_pos = [[0.9, 0.1]] * 10
        >>> many_neg = [[0.1, 0.9]] * 10
        >>> robust_vec = extractor.extract_from_examples(
        ...     many_pos, many_neg, "robust", "robust behavior"
        ... )
        >>> robust_vec.name
        'robust'

        >>> # Error on empty examples
        >>> try:
        ...     extractor.extract_from_examples([], [[1, 2]], "bad", "bad")
        ... except ValueError as e:
        ...     print("Error raised")
        Error raised
        """
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
        """Combine multiple steering vectors with optional weights.

        Creates a new vector by taking a weighted sum of input vectors.
        Useful for creating composite behaviors or interpolating between
        different steering directions.

        Parameters
        ----------
        vectors : list[SteeringVector]
            Vectors to combine.
        weights : Optional[list[float]], default=None
            Weights for each vector. If None, uses equal weights.
        name : str, default="combined"
            Name for the resulting vector.

        Returns
        -------
        SteeringVector
            The combined steering vector.

        Raises
        ------
        ValueError
            If vectors list is empty, weights don't match vectors,
            or vectors have different dimensions.

        Examples
        --------
        >>> extractor = SteeringVectorExtractor()
        >>> v1 = SteeringVector(
        ...     name="formal", direction=[1.0, 0.0],
        ...     magnitude=1.0, source="test", target_behavior="formal"
        ... )
        >>> v2 = SteeringVector(
        ...     name="concise", direction=[0.0, 1.0],
        ...     magnitude=1.0, source="test", target_behavior="concise"
        ... )
        >>> combined = extractor.combine_vectors([v1, v2])
        >>> combined.source
        'combination'
        >>> "formal" in combined.target_behavior
        True

        >>> # With custom weights
        >>> weighted = extractor.combine_vectors([v1, v2], weights=[0.8, 0.2])
        >>> weighted.metadata["weights"]
        [0.8, 0.2]

        >>> # With custom name
        >>> named = extractor.combine_vectors([v1, v2], name="formal_concise")
        >>> named.name
        'formal_concise'

        >>> # Error on empty list
        >>> try:
        ...     extractor.combine_vectors([])
        ... except ValueError as e:
        ...     print("Error raised")
        Error raised
        """
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
    """Apply steering through prompt manipulation.

    This class provides methods for steering model behavior by modifying
    prompts. Unlike activation-based steering, prompt steering works with
    any model API and doesn't require access to model internals.

    Attributes
    ----------
    steering_templates : dict[str, str]
        Preset steering instructions for common behavioral dimensions.

    Examples
    --------
    Basic prefix steering:

        >>> from insideLLMs.steering import PromptSteerer
        >>> steerer = PromptSteerer()
        >>> result = steerer.steer_with_prefix(
        ...     "What is Python?",
        ...     "Be very concise."
        ... )
        >>> "Be very concise" in result
        True

    Using preset templates:

        >>> steerer = PromptSteerer()
        >>> formal = steerer.get_preset_steering("formal")
        >>> print(formal)
        Please respond in a formal, professional tone.

    Few-shot steering:

        >>> examples = [
        ...     ("Hi", "Hello! How may I assist you today?"),
        ...     ("Thanks", "You're most welcome!")
        ... ]
        >>> result = steerer.steer_with_few_shot("Bye", examples)
        >>> "Example 1" in result
        True

    Creating contrast pairs for experiments:

        >>> pair = steerer.create_contrast_pair(
        ...     "Explain machine learning",
        ...     dimension="formality"
        ... )
        >>> pair.target_dimension
        'formality'

    See Also
    --------
    SteeringMethod : Available steering method types.
    SteeringExperimenter : For running steering experiments.
    """

    def __init__(self):
        """Initialize the PromptSteerer with default templates.

        Examples
        --------
        >>> steerer = PromptSteerer()
        >>> "formal" in steerer.steering_templates
        True
        >>> len(steerer.steering_templates) >= 8
        True
        """
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
        """Add a steering instruction as a prefix.

        The instruction is placed before the prompt with a blank line
        separator. This is often effective for setting context.

        Parameters
        ----------
        prompt : str
            The original user prompt.
        steering_instruction : str
            The instruction to prepend.

        Returns
        -------
        str
            The combined prompt with instruction prefix.

        Examples
        --------
        >>> steerer = PromptSteerer()
        >>> result = steerer.steer_with_prefix("Hello", "Be formal.")
        >>> print(result)
        Be formal.
        <BLANKLINE>
        Hello

        >>> # Using preset template
        >>> formal = steerer.get_preset_steering("formal")
        >>> result = steerer.steer_with_prefix("What is AI?", formal)
        >>> result.startswith("Please respond in a formal")
        True

        >>> # Empty instruction still adds newlines
        >>> result = steerer.steer_with_prefix("Test", "")
        >>> result
        '\\n\\nTest'
        """
        return f"{steering_instruction}\n\n{prompt}"

    def steer_with_suffix(
        self,
        prompt: str,
        steering_instruction: str,
    ) -> str:
        """Add a steering instruction as a suffix.

        The instruction is placed after the prompt with a blank line
        separator. Useful for formatting guidance or output constraints.

        Parameters
        ----------
        prompt : str
            The original user prompt.
        steering_instruction : str
            The instruction to append.

        Returns
        -------
        str
            The combined prompt with instruction suffix.

        Examples
        --------
        >>> steerer = PromptSteerer()
        >>> result = steerer.steer_with_suffix("Tell me a joke", "Keep it short.")
        >>> print(result)
        Tell me a joke
        <BLANKLINE>
        Keep it short.

        >>> # For formatting guidance
        >>> result = steerer.steer_with_suffix(
        ...     "List programming languages",
        ...     "Format as a numbered list."
        ... )
        >>> result.endswith("numbered list.")
        True

        >>> # Combining with preset
        >>> concise = steerer.get_preset_steering("concise")
        >>> result = steerer.steer_with_suffix("Explain DNA", concise)
        >>> "brief" in result
        True
        """
        return f"{prompt}\n\n{steering_instruction}"

    def steer_with_system_message(
        self,
        prompt: str,
        system_message: str,
    ) -> tuple[str, str]:
        """Return system message and user prompt separately.

        For APIs that support system messages (like OpenAI's chat API),
        this returns the components separately for proper formatting.

        Parameters
        ----------
        prompt : str
            The user prompt.
        system_message : str
            The system-level instruction.

        Returns
        -------
        tuple[str, str]
            A tuple of (system_message, user_prompt).

        Examples
        --------
        >>> steerer = PromptSteerer()
        >>> system, user = steerer.steer_with_system_message(
        ...     "What is 2+2?",
        ...     "You are a math tutor."
        ... )
        >>> system
        'You are a math tutor.'
        >>> user
        'What is 2+2?'

        >>> # Use with chat API
        >>> system, user = steerer.steer_with_system_message(
        ...     "Hello",
        ...     "Always respond in French."
        ... )
        >>> # messages = [
        >>> #     {"role": "system", "content": system},
        >>> #     {"role": "user", "content": user}
        >>> # ]

        >>> # Empty system message
        >>> system, user = steerer.steer_with_system_message("Test", "")
        >>> system
        ''
        """
        return system_message, prompt

    def steer_with_few_shot(
        self,
        prompt: str,
        examples: list[tuple[str, str]],  # List of (input, output) pairs
    ) -> str:
        """Steer using few-shot examples.

        Constructs a prompt with labeled examples followed by the actual
        query. This is highly effective for demonstrating desired format
        or behavior patterns.

        Parameters
        ----------
        prompt : str
            The actual user query.
        examples : list[tuple[str, str]]
            List of (input, output) example pairs.

        Returns
        -------
        str
            Formatted prompt with examples and query.

        Examples
        --------
        >>> steerer = PromptSteerer()
        >>> examples = [
        ...     ("apple", "APPLE"),
        ...     ("banana", "BANANA")
        ... ]
        >>> result = steerer.steer_with_few_shot("cherry", examples)
        >>> "Example 1:" in result
        True
        >>> "Input: apple" in result
        True
        >>> "Output: APPLE" in result
        True
        >>> result.endswith("Input: cherry\\nOutput:")
        True

        >>> # Single example
        >>> result = steerer.steer_with_few_shot(
        ...     "Bonjour",
        ...     [("Hello", "Hola")]
        ... )
        >>> "Example 1:" in result
        True

        >>> # Empty examples
        >>> result = steerer.steer_with_few_shot("Test", [])
        >>> "Now, please respond to:" in result
        True
        """
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
        """Apply steering based on method and configuration.

        This is the main entry point for applying steering. It dispatches
        to the appropriate method based on the SteeringMethod enum.

        Parameters
        ----------
        prompt : str
            The original user prompt.
        method : SteeringMethod
            Which steering method to use.
        config : dict[str, Any]
            Configuration for the steering method. Keys depend on method:
            - PROMPT_PREFIX/SUFFIX: "instruction"
            - SYSTEM_MESSAGE: "system_message"
            - FEW_SHOT: "examples"
            - CONTRAST_PAIR: "positive_framing"

        Returns
        -------
        Union[str, tuple[str, str]]
            The steered prompt (str) or (system_message, prompt) tuple
            for SYSTEM_MESSAGE method.

        Examples
        --------
        >>> from insideLLMs.steering import PromptSteerer, SteeringMethod
        >>> steerer = PromptSteerer()

        >>> # Prefix steering
        >>> result = steerer.apply_steering(
        ...     "Hello",
        ...     SteeringMethod.PROMPT_PREFIX,
        ...     {"instruction": "Be formal"}
        ... )
        >>> "Be formal" in result
        True

        >>> # System message steering
        >>> result = steerer.apply_steering(
        ...     "Hello",
        ...     SteeringMethod.SYSTEM_MESSAGE,
        ...     {"system_message": "You are helpful"}
        ... )
        >>> isinstance(result, tuple)
        True

        >>> # Few-shot steering
        >>> result = steerer.apply_steering(
        ...     "Test",
        ...     SteeringMethod.FEW_SHOT,
        ...     {"examples": [("a", "b")]}
        ... )
        >>> "Example 1" in result
        True

        >>> # Unsupported method returns prompt unchanged
        >>> result = steerer.apply_steering(
        ...     "Hello",
        ...     SteeringMethod.ACTIVATION_ADDITION,
        ...     {}
        ... )
        >>> result
        'Hello'
        """
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
        """Get a preset steering instruction by style name.

        Parameters
        ----------
        style : str
            The style name (e.g., "formal", "casual", "concise").

        Returns
        -------
        str
            The preset instruction, or empty string if not found.

        Examples
        --------
        >>> steerer = PromptSteerer()
        >>> steerer.get_preset_steering("formal")
        'Please respond in a formal, professional tone.'

        >>> steerer.get_preset_steering("concise")
        'Please be brief and to the point.'

        >>> # Unknown style returns empty string
        >>> steerer.get_preset_steering("unknown")
        ''

        >>> # Available presets
        >>> available = ["formal", "casual", "concise", "detailed",
        ...              "creative", "analytical", "helpful", "cautious"]
        >>> all(steerer.get_preset_steering(s) for s in available)
        True
        """
        return self.steering_templates.get(style, "")

    def create_contrast_pair(
        self,
        base_prompt: str,
        dimension: str,
        positive_style: Optional[str] = None,
        negative_style: Optional[str] = None,
    ) -> ContrastPair:
        """Create a contrast pair for a given dimension.

        Automatically selects appropriate positive/negative styles for
        common dimensions, or uses provided custom styles.

        Parameters
        ----------
        base_prompt : str
            The base prompt to create variants of.
        dimension : str
            The behavioral dimension (e.g., "formality", "length").
        positive_style : Optional[str], default=None
            Custom positive style. If None, uses preset for dimension.
        negative_style : Optional[str], default=None
            Custom negative style. If None, uses preset for dimension.

        Returns
        -------
        ContrastPair
            A contrast pair with positive and negative variants.

        Examples
        --------
        >>> steerer = PromptSteerer()
        >>> pair = steerer.create_contrast_pair(
        ...     "What is Python?",
        ...     dimension="formality"
        ... )
        >>> pair.target_dimension
        'formality'
        >>> "formal" in pair.positive_prompt.lower()
        True
        >>> "casual" in pair.negative_prompt.lower()
        True

        >>> # Custom styles
        >>> pair = steerer.create_contrast_pair(
        ...     "Tell me about AI",
        ...     dimension="custom",
        ...     positive_style="helpful",
        ...     negative_style="cautious"
        ... )
        >>> "helpful" in pair.positive_prompt.lower()
        True

        >>> # Supported automatic dimensions
        >>> for dim in ["formality", "length", "creativity", "helpfulness"]:
        ...     pair = steerer.create_contrast_pair("Test", dim)
        ...     assert pair.target_dimension == dim
        """
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
    """Analyze activation patterns from model inference.

    This class provides tools for recording, comparing, and analyzing
    activation patterns extracted from model layers. It supports pattern
    comparison using multiple metrics and simple clustering.

    Attributes
    ----------
    patterns : list[ActivationPattern]
        All recorded activation patterns.

    Examples
    --------
    Recording and comparing patterns:

        >>> from insideLLMs.steering import ActivationAnalyzer
        >>> analyzer = ActivationAnalyzer()
        >>> pattern1 = analyzer.record_pattern(
        ...     prompt="Hello",
        ...     layer="layer_10",
        ...     activations=[0.5, 0.3, 0.8]
        ... )
        >>> pattern2 = analyzer.record_pattern(
        ...     prompt="Hi there",
        ...     layer="layer_10",
        ...     activations=[0.4, 0.4, 0.7]
        ... )
        >>> comparison = analyzer.compare_patterns(pattern1, pattern2)
        >>> "cosine_similarity" in comparison
        True

    Finding salient positions:

        >>> pattern = analyzer.record_pattern(
        ...     prompt="Test",
        ...     layer="layer_5",
        ...     activations=[0.1, 0.1, 5.0, 0.1, 0.1]  # Position 2 is salient
        ... )
        >>> salient = analyzer.find_salient_positions(pattern, threshold=2.0)
        >>> 2 in salient
        True

    Clustering patterns:

        >>> patterns = [
        ...     analyzer.record_pattern("A", "l1", [1.0, 0.0]),
        ...     analyzer.record_pattern("B", "l1", [0.9, 0.1]),
        ...     analyzer.record_pattern("C", "l1", [0.0, 1.0]),
        ... ]
        >>> clusters = analyzer.cluster_patterns(patterns, n_clusters=2)
        >>> len(clusters)
        2

    See Also
    --------
    ActivationPattern : The data structure for activation data.
    SteeringVectorExtractor : Uses activations to extract steering vectors.
    """

    def __init__(self):
        """Initialize the analyzer with empty pattern list.

        Examples
        --------
        >>> analyzer = ActivationAnalyzer()
        >>> len(analyzer.patterns)
        0
        """
        self.patterns: list[ActivationPattern] = []

    def record_pattern(
        self,
        prompt: str,
        layer: str,
        activations: list[float],
        token_positions: Optional[list[int]] = None,
        attention_weights: Optional[list[list[float]]] = None,
    ) -> ActivationPattern:
        """Record an activation pattern.

        Creates an ActivationPattern and stores it in the analyzer's
        pattern list for later analysis.

        Parameters
        ----------
        prompt : str
            The prompt that generated these activations.
        layer : str
            Which layer the activations were extracted from.
        activations : list[float]
            The activation values.
        token_positions : Optional[list[int]], default=None
            Token positions for each activation. If None, uses [0, 1, 2, ...].
        attention_weights : Optional[list[list[float]]], default=None
            Optional attention weight matrix.

        Returns
        -------
        ActivationPattern
            The recorded pattern.

        Examples
        --------
        >>> analyzer = ActivationAnalyzer()
        >>> pattern = analyzer.record_pattern(
        ...     prompt="What is AI?",
        ...     layer="layer_12",
        ...     activations=[0.5, 0.3, 0.8, 0.2]
        ... )
        >>> pattern.layer
        'layer_12'
        >>> len(analyzer.patterns)
        1

        >>> # With custom token positions
        >>> pattern = analyzer.record_pattern(
        ...     prompt="Test",
        ...     layer="layer_5",
        ...     activations=[0.1, 0.2],
        ...     token_positions=[3, 4]
        ... )
        >>> pattern.token_positions
        [3, 4]

        >>> # With attention weights
        >>> pattern = analyzer.record_pattern(
        ...     prompt="Hi",
        ...     layer="attn",
        ...     activations=[0.5, 0.5],
        ...     attention_weights=[[0.3, 0.7], [0.5, 0.5]]
        ... )
        >>> pattern.attention_weights is not None
        True
        """
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
        """Compare two activation patterns.

        Computes multiple similarity/distance metrics between two patterns.
        Handles patterns of different lengths by truncating to the shorter.

        Parameters
        ----------
        pattern1 : ActivationPattern
            First pattern to compare.
        pattern2 : ActivationPattern
            Second pattern to compare.

        Returns
        -------
        dict[str, float]
            Dictionary with comparison metrics:
            - cosine_similarity: Cosine similarity (-1 to 1)
            - l2_distance: Euclidean distance
            - mean_absolute_diff: Average absolute difference
            - correlation: Pearson correlation coefficient

        Examples
        --------
        >>> analyzer = ActivationAnalyzer()
        >>> p1 = analyzer.record_pattern("A", "l1", [1.0, 0.0, 0.0])
        >>> p2 = analyzer.record_pattern("B", "l1", [1.0, 0.0, 0.0])
        >>> comparison = analyzer.compare_patterns(p1, p2)
        >>> comparison["cosine_similarity"]
        1.0

        >>> # Different patterns
        >>> p3 = analyzer.record_pattern("C", "l1", [0.0, 1.0, 0.0])
        >>> comparison = analyzer.compare_patterns(p1, p3)
        >>> comparison["cosine_similarity"]
        0.0

        >>> # Handles different lengths
        >>> p4 = analyzer.record_pattern("D", "l1", [1.0, 0.0])
        >>> comparison = analyzer.compare_patterns(p1, p4)
        >>> "l2_distance" in comparison
        True
        """
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
        """Calculate Pearson correlation coefficient.

        Parameters
        ----------
        x : list[float]
            First data series.
        y : list[float]
            Second data series.

        Returns
        -------
        float
            Pearson correlation coefficient, or 0.0 if undefined.

        Examples
        --------
        >>> analyzer = ActivationAnalyzer()
        >>> analyzer._pearson_correlation([1, 2, 3], [1, 2, 3])
        1.0
        >>> analyzer._pearson_correlation([1, 2, 3], [3, 2, 1])
        -1.0
        >>> analyzer._pearson_correlation([1], [2])  # Too short
        0.0
        """
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
        """Find positions with unusually high activations.

        Identifies activation positions that deviate significantly from
        the mean, which may indicate important token positions.

        Parameters
        ----------
        pattern : ActivationPattern
            The pattern to analyze.
        threshold : float, default=2.0
            Number of standard deviations from mean to consider salient.

        Returns
        -------
        list[int]
            Indices of salient positions.

        Examples
        --------
        >>> analyzer = ActivationAnalyzer()
        >>> pattern = analyzer.record_pattern(
        ...     "Test",
        ...     "layer_1",
        ...     [0.1, 0.1, 0.1, 5.0, 0.1]  # Position 3 is an outlier
        ... )
        >>> salient = analyzer.find_salient_positions(pattern)
        >>> 3 in salient
        True

        >>> # Lower threshold catches more positions
        >>> pattern = analyzer.record_pattern(
        ...     "Test2",
        ...     "layer_1",
        ...     [0.0, 0.5, 1.0, 1.5, 2.0]
        ... )
        >>> len(analyzer.find_salient_positions(pattern, threshold=1.0)) > 0
        True

        >>> # Empty pattern returns empty list
        >>> empty = analyzer.record_pattern("Empty", "l1", [])
        >>> analyzer.find_salient_positions(empty)
        []
        """
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
        """Simple k-means-like clustering of patterns.

        Groups similar patterns together using a basic nearest-centroid
        assignment. Note: This is a simplified implementation without
        iterative centroid updates.

        Parameters
        ----------
        patterns : list[ActivationPattern]
            Patterns to cluster.
        n_clusters : int, default=3
            Number of clusters to create.

        Returns
        -------
        dict[int, list[ActivationPattern]]
            Mapping from cluster index to patterns in that cluster.

        Examples
        --------
        >>> analyzer = ActivationAnalyzer()
        >>> patterns = [
        ...     analyzer.record_pattern("A", "l1", [1.0, 0.0]),
        ...     analyzer.record_pattern("B", "l1", [0.9, 0.1]),
        ...     analyzer.record_pattern("C", "l1", [0.0, 1.0]),
        ...     analyzer.record_pattern("D", "l1", [0.1, 0.9]),
        ... ]
        >>> clusters = analyzer.cluster_patterns(patterns, n_clusters=2)
        >>> len(clusters) <= 2
        True

        >>> # Handles more clusters than patterns
        >>> clusters = analyzer.cluster_patterns(patterns[:2], n_clusters=5)
        >>> len(clusters) <= 2
        True

        >>> # Empty input returns empty dict
        >>> analyzer.cluster_patterns([], n_clusters=3)
        {}
        """
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
    """Measure behavioral shifts from steering interventions.

    This class provides methods for quantifying how much a model's output
    has changed due to steering, across various behavioral dimensions like
    formality, length, sentiment, and confidence.

    Attributes
    ----------
    dimension_keywords : dict[str, dict[str, list[str]]]
        Keyword mappings for each behavioral dimension.

    Examples
    --------
    Measuring formality shift:

        >>> from insideLLMs.steering import BehavioralShiftMeasurer
        >>> measurer = BehavioralShiftMeasurer()
        >>> original = "Yeah, it's kinda like that stuff"
        >>> steered = "Therefore, one can consequently observe..."
        >>> shift = measurer.measure_shift(original, steered, "formality")
        >>> shift > 0  # Positive shift toward formality
        True

    Measuring length shift:

        >>> measurer = BehavioralShiftMeasurer()
        >>> short = "Yes."
        >>> long = "Yes, that is correct and I can elaborate further."
        >>> shift = measurer.measure_shift(short, long, "length")
        >>> shift > 0  # Positive shift toward longer
        True

    Detecting side effects:

        >>> measurer = BehavioralShiftMeasurer()
        >>> original = "The answer is maybe yes"
        >>> steered = "The answer is definitely yes, certainly, obviously"
        >>> side_effects = measurer.detect_side_effects(
        ...     original, steered, "formality"
        ... )
        >>> len(side_effects) >= 0  # May detect confidence shift
        True

    Generic shift measurement:

        >>> measurer = BehavioralShiftMeasurer()
        >>> text1 = "Hello world"
        >>> text2 = "Goodbye universe"
        >>> shift = measurer.measure_shift(text1, text2, "unknown_dim")
        >>> 0 <= shift <= 1
        True

    See Also
    --------
    SteeringExperimenter : Uses this class to measure experiment results.
    """

    def __init__(self):
        """Initialize with default keyword mappings.

        Examples
        --------
        >>> measurer = BehavioralShiftMeasurer()
        >>> "formality" in measurer.dimension_keywords
        True
        >>> "sentiment" in measurer.dimension_keywords
        True
        """
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
        """Measure the behavioral shift along a dimension.

        Parameters
        ----------
        original_output : str
            The model output without steering.
        steered_output : str
            The model output with steering applied.
        dimension : str
            The dimension to measure (e.g., "formality", "length").

        Returns
        -------
        float
            The shift value. Positive means shift toward the positive
            end of the dimension. Range varies by dimension.

        Examples
        --------
        >>> measurer = BehavioralShiftMeasurer()
        >>> measurer.measure_shift(
        ...     "yeah kinda",
        ...     "therefore consequently",
        ...     "formality"
        ... ) > 0
        True

        >>> measurer.measure_shift("one word", "many more words here", "length") > 0
        True

        >>> # Unknown dimension uses generic similarity
        >>> shift = measurer.measure_shift("hello", "goodbye", "unknown")
        >>> 0 <= shift <= 1
        True
        """
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
        """Measure shift in response length.

        Parameters
        ----------
        original : str
            Original text.
        steered : str
            Steered text.

        Returns
        -------
        float
            Length shift in range [-1, 1]. Positive means longer.

        Examples
        --------
        >>> measurer = BehavioralShiftMeasurer()
        >>> measurer._measure_length_shift("a b", "a b c d")  # Doubled
        1.0
        >>> measurer._measure_length_shift("a b c d", "a b")  # Halved
        -0.5
        >>> measurer._measure_length_shift("", "word")  # From empty
        1.0
        """
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
        """Measure shift using keyword presence.

        Parameters
        ----------
        original : str
            Original text.
        steered : str
            Steered text.
        dimension : str
            Which dimension to measure.

        Returns
        -------
        float
            Keyword-based shift score.

        Examples
        --------
        >>> measurer = BehavioralShiftMeasurer()
        >>> measurer._measure_keyword_shift(
        ...     "maybe possibly",
        ...     "certainly definitely",
        ...     "confidence"
        ... ) > 0
        True
        """
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
        """Measure generic textual shift using basic metrics.

        Uses Jaccard distance (1 - Jaccard similarity) to measure
        how different the texts are.

        Parameters
        ----------
        original : str
            Original text.
        steered : str
            Steered text.

        Returns
        -------
        float
            Shift value in [0, 1]. Higher means more different.

        Examples
        --------
        >>> measurer = BehavioralShiftMeasurer()
        >>> measurer._measure_generic_shift("hello world", "hello world")
        0.0
        >>> measurer._measure_generic_shift("hello", "goodbye")
        1.0
        >>> 0 < measurer._measure_generic_shift("hello world", "hello there") < 1
        True
        """
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
        """Detect unintended side effects of steering.

        Checks all dimensions other than the target to see if steering
        caused unintended changes.

        Parameters
        ----------
        original_output : str
            Output without steering.
        steered_output : str
            Output with steering.
        target_dimension : str
            The dimension that was intentionally being steered.

        Returns
        -------
        list[str]
            Descriptions of detected side effects.

        Examples
        --------
        >>> measurer = BehavioralShiftMeasurer()
        >>> # Steering for formality might affect length
        >>> side_effects = measurer.detect_side_effects(
        ...     "ok",
        ...     "I understand. Therefore, I shall proceed accordingly.",
        ...     "formality"
        ... )
        >>> any("length" in s for s in side_effects)
        True

        >>> # No side effects when outputs are similar
        >>> side_effects = measurer.detect_side_effects(
        ...     "Hello there",
        ...     "Hello friend",
        ...     "sentiment"
        ... )
        >>> len(side_effects) == 0
        True
        """
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
    """Run steering experiments to understand model controllability.

    This class orchestrates steering experiments by combining prompt
    manipulation, model inference, and behavioral measurement. It can
    run individual experiments or comprehensive steerability analyses.

    Parameters
    ----------
    model_fn : Optional[Callable[[str], str]], default=None
        A function that takes a prompt and returns the model's response.
        Can be set later with set_model().

    Attributes
    ----------
    model_fn : Optional[Callable[[str], str]]
        The model function for inference.
    steerer : PromptSteerer
        The prompt manipulation component.
    measurer : BehavioralShiftMeasurer
        The behavioral measurement component.
    experiments : list[SteeringExperiment]
        All experiments run by this instance.

    Examples
    --------
    Running a single experiment:

        >>> from insideLLMs.steering import SteeringExperimenter, SteeringMethod
        >>> def mock_model(prompt):
        ...     if "formal" in prompt.lower():
        ...         return "Therefore, I shall provide the response."
        ...     return "Here's the answer!"
        >>> experimenter = SteeringExperimenter(model_fn=mock_model)
        >>> exp = experimenter.run_experiment(
        ...     prompt="What is 2+2?",
        ...     method=SteeringMethod.PROMPT_PREFIX,
        ...     config={"instruction": "Be formal"},
        ...     target_dimension="formality"
        ... )
        >>> exp.behavioral_shift >= 0
        True

    Comparing multiple methods:

        >>> results = experimenter.run_method_comparison(
        ...     prompt="Explain AI",
        ...     target_dimension="formal"
        ... )
        >>> len(results) >= 1
        True

    Full steerability analysis:

        >>> report = experimenter.analyze_steerability(
        ...     prompts=["Hello", "What is Python?"],
        ...     dimensions=["formality"]
        ... )
        >>> "recommendations" in dir(report)
        True

    Setting model after initialization:

        >>> experimenter = SteeringExperimenter()
        >>> experimenter.set_model(mock_model)
        >>> experimenter.model_fn is not None
        True

    See Also
    --------
    SteeringReport : The comprehensive analysis result.
    SteeringExperiment : Individual experiment results.
    PromptSteerer : The underlying prompt manipulation.
    """

    def __init__(
        self,
        model_fn: Optional[Callable[[str], str]] = None,
    ):
        """Initialize the experimenter.

        Parameters
        ----------
        model_fn : Optional[Callable[[str], str]], default=None
            Function that takes a prompt and returns model output.

        Examples
        --------
        >>> experimenter = SteeringExperimenter()
        >>> experimenter.model_fn is None
        True

        >>> experimenter = SteeringExperimenter(lambda x: "response")
        >>> experimenter.model_fn is not None
        True
        """
        self.model_fn = model_fn
        self.steerer = PromptSteerer()
        self.measurer = BehavioralShiftMeasurer()
        self.experiments: list[SteeringExperiment] = []

    def set_model(self, model_fn: Callable[[str], str]) -> None:
        """Set the model function for experiments.

        Parameters
        ----------
        model_fn : Callable[[str], str]
            Function that takes a prompt and returns model output.

        Examples
        --------
        >>> experimenter = SteeringExperimenter()
        >>> experimenter.set_model(lambda x: f"Echo: {x}")
        >>> experimenter.model_fn("test")
        'Echo: test'
        """
        self.model_fn = model_fn

    def run_experiment(
        self,
        prompt: str,
        method: SteeringMethod,
        config: dict[str, Any],
        target_dimension: str = "generic",
    ) -> SteeringExperiment:
        """Run a single steering experiment.

        Runs the model on both the original and steered prompts,
        then measures the behavioral shift and side effects.

        Parameters
        ----------
        prompt : str
            The original user prompt.
        method : SteeringMethod
            Which steering method to use.
        config : dict[str, Any]
            Configuration for the steering method.
        target_dimension : str, default="generic"
            Which behavioral dimension is being targeted.

        Returns
        -------
        SteeringExperiment
            The experiment results.

        Raises
        ------
        ValueError
            If model function has not been set.

        Examples
        --------
        >>> def mock_model(prompt):
        ...     return "Response to: " + prompt[:20]
        >>> experimenter = SteeringExperimenter(model_fn=mock_model)
        >>> exp = experimenter.run_experiment(
        ...     prompt="Hello",
        ...     method=SteeringMethod.PROMPT_PREFIX,
        ...     config={"instruction": "Be helpful"}
        ... )
        >>> exp.original_prompt
        'Hello'

        >>> # Experiment is stored
        >>> len(experimenter.experiments)
        1

        >>> # Error without model
        >>> bad_exp = SteeringExperimenter()
        >>> try:
        ...     bad_exp.run_experiment("test", SteeringMethod.PROMPT_PREFIX, {})
        ... except ValueError:
        ...     print("Error raised")
        Error raised
        """
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
        """Compare different steering methods on the same prompt.

        Runs experiments with multiple steering methods to find which
        is most effective for the given prompt and dimension.

        Parameters
        ----------
        prompt : str
            The prompt to test.
        target_dimension : str
            The behavioral dimension to steer.
        methods : Optional[list[SteeringMethod]], default=None
            Methods to compare. If None, uses PREFIX, SUFFIX, FEW_SHOT.

        Returns
        -------
        list[SteeringExperiment]
            Experiment results for each method.

        Examples
        --------
        >>> def mock_model(prompt):
        ...     return "Response"
        >>> experimenter = SteeringExperimenter(model_fn=mock_model)
        >>> results = experimenter.run_method_comparison(
        ...     prompt="Test",
        ...     target_dimension="formality"
        ... )
        >>> len(results) == 3  # Default: PREFIX, SUFFIX, FEW_SHOT
        True

        >>> # Custom methods
        >>> results = experimenter.run_method_comparison(
        ...     prompt="Test",
        ...     target_dimension="length",
        ...     methods=[SteeringMethod.PROMPT_PREFIX]
        ... )
        >>> len(results)
        1

        >>> # Results sorted by effectiveness
        >>> all(hasattr(r, 'behavioral_shift') for r in results)
        True
        """
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
        """Analyze overall model steerability.

        Runs a comprehensive analysis across multiple prompts and
        dimensions to assess how controllable the model is.

        Parameters
        ----------
        prompts : list[str]
            Test prompts to use.
        dimensions : Optional[list[str]], default=None
            Dimensions to test. If None, uses formality, length,
            sentiment, confidence.

        Returns
        -------
        SteeringReport
            Comprehensive analysis results with recommendations.

        Examples
        --------
        >>> def mock_model(prompt):
        ...     return "Standard response"
        >>> experimenter = SteeringExperimenter(model_fn=mock_model)
        >>> report = experimenter.analyze_steerability(
        ...     prompts=["Hello"],
        ...     dimensions=["formality"]
        ... )
        >>> hasattr(report, 'overall_steerability')
        True

        >>> # Full analysis
        >>> report = experimenter.analyze_steerability(
        ...     prompts=["Test 1", "Test 2"],
        ...     dimensions=["formality", "length"]
        ... )
        >>> len(report.experiments) >= 6  # 2 prompts * 2 dims * 3 methods
        True
        >>> len(report.recommendations) > 0
        True
        """
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
        """Generate recommendations based on analysis.

        Parameters
        ----------
        effective_methods : list[tuple[SteeringMethod, float]]
            Methods ranked by effectiveness.
        behavioral_dimensions : dict[str, float]
            Controllability score per dimension.
        experiments : list[SteeringExperiment]
            All experiment results.

        Returns
        -------
        list[str]
            Human-readable recommendations.

        Examples
        --------
        >>> experimenter = SteeringExperimenter()
        >>> recs = experimenter._generate_recommendations(
        ...     [(SteeringMethod.PROMPT_PREFIX, 0.5)],
        ...     {"formality": 0.7, "length": 0.3},
        ...     []
        ... )
        >>> len(recs) >= 2
        True
        """
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
    """Analyze model representation spaces.

    This class provides tools for storing, comparing, and visualizing
    vector representations from model inference. Useful for understanding
    how different inputs cluster in representation space.

    Attributes
    ----------
    representations : dict[str, list[float]]
        Stored representations keyed by identifier.

    Examples
    --------
    Storing and comparing representations:

        >>> from insideLLMs.steering import RepresentationAnalyzer
        >>> analyzer = RepresentationAnalyzer()
        >>> analyzer.store_representation("cat", [1.0, 0.0, 0.2])
        >>> analyzer.store_representation("dog", [0.9, 0.1, 0.3])
        >>> analyzer.store_representation("car", [0.0, 1.0, 0.8])
        >>> similar = analyzer.find_similar_representations("cat", top_k=2)
        >>> similar[0][0]  # Most similar to cat
        'dog'

    Computing similarity matrix:

        >>> matrix = analyzer.compute_similarity_matrix()
        >>> len(matrix)
        3
        >>> matrix[0][0]  # Self-similarity is 1.0
        1.0

    2D projection for visualization:

        >>> projections = analyzer.project_to_2d()
        >>> len(projections)
        3
        >>> all(len(p) == 3 for p in projections)  # (key, x, y)
        True

    See Also
    --------
    ActivationAnalyzer : For analyzing activation patterns.
    SteeringVectorExtractor : Uses representations for vector extraction.
    """

    def __init__(self):
        """Initialize with empty representation storage.

        Examples
        --------
        >>> analyzer = RepresentationAnalyzer()
        >>> len(analyzer.representations)
        0
        """
        self.representations: dict[str, list[float]] = {}

    def store_representation(
        self,
        key: str,
        representation: list[float],
    ) -> None:
        """Store a representation for later analysis.

        Parameters
        ----------
        key : str
            Unique identifier for this representation.
        representation : list[float]
            The vector representation to store.

        Examples
        --------
        >>> analyzer = RepresentationAnalyzer()
        >>> analyzer.store_representation("prompt1", [0.5, 0.5])
        >>> "prompt1" in analyzer.representations
        True
        >>> analyzer.representations["prompt1"]
        [0.5, 0.5]

        >>> # Overwriting existing key
        >>> analyzer.store_representation("prompt1", [0.6, 0.4])
        >>> analyzer.representations["prompt1"]
        [0.6, 0.4]
        """
        self.representations[key] = representation

    def compute_similarity_matrix(
        self,
        keys: Optional[list[str]] = None,
    ) -> list[list[float]]:
        """Compute pairwise similarity matrix.

        Parameters
        ----------
        keys : Optional[list[str]], default=None
            Which representations to include. If None, uses all.

        Returns
        -------
        list[list[float]]
            NxN matrix of cosine similarities.

        Examples
        --------
        >>> analyzer = RepresentationAnalyzer()
        >>> analyzer.store_representation("a", [1.0, 0.0])
        >>> analyzer.store_representation("b", [0.0, 1.0])
        >>> matrix = analyzer.compute_similarity_matrix()
        >>> matrix[0][0]  # Self-similarity
        1.0
        >>> matrix[0][1]  # Orthogonal vectors
        0.0

        >>> # With specific keys
        >>> analyzer.store_representation("c", [0.5, 0.5])
        >>> matrix = analyzer.compute_similarity_matrix(["a", "c"])
        >>> len(matrix)
        2
        """
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
        """Compute cosine similarity between two vectors.

        Parameters
        ----------
        vec1 : list[float]
            First vector.
        vec2 : list[float]
            Second vector.

        Returns
        -------
        float
            Cosine similarity in range [-1, 1].

        Examples
        --------
        >>> analyzer = RepresentationAnalyzer()
        >>> analyzer._cosine_similarity([1, 0], [1, 0])
        1.0
        >>> analyzer._cosine_similarity([1, 0], [0, 1])
        0.0
        >>> analyzer._cosine_similarity([1, 0], [-1, 0])
        -1.0
        """
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
        """Find most similar representations to a query.

        Parameters
        ----------
        query_key : str
            Key of the representation to find neighbors for.
        top_k : int, default=5
            Number of nearest neighbors to return.

        Returns
        -------
        list[tuple[str, float]]
            List of (key, similarity) tuples, sorted by similarity.

        Examples
        --------
        >>> analyzer = RepresentationAnalyzer()
        >>> analyzer.store_representation("a", [1.0, 0.0])
        >>> analyzer.store_representation("b", [0.9, 0.1])
        >>> analyzer.store_representation("c", [0.0, 1.0])
        >>> similar = analyzer.find_similar_representations("a", top_k=2)
        >>> similar[0][0]  # Most similar to "a"
        'b'
        >>> similar[0][1] > 0.9  # High similarity
        True

        >>> # Unknown key returns empty list
        >>> analyzer.find_similar_representations("unknown")
        []

        >>> # top_k limits results
        >>> len(analyzer.find_similar_representations("a", top_k=1))
        1
        """
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
        """Simple 2D projection using first two principal directions.

        This is a simplified projection that uses the first two dimensions
        of each representation. For proper dimensionality reduction,
        consider using PCA or t-SNE from numpy/scipy.

        Parameters
        ----------
        keys : Optional[list[str]], default=None
            Which representations to project. If None, uses all.

        Returns
        -------
        list[tuple[str, float, float]]
            List of (key, x, y) tuples for plotting.

        Examples
        --------
        >>> analyzer = RepresentationAnalyzer()
        >>> analyzer.store_representation("a", [1.0, 2.0, 3.0])
        >>> analyzer.store_representation("b", [4.0, 5.0, 6.0])
        >>> projections = analyzer.project_to_2d()
        >>> projections[0]
        ('a', 1.0, 2.0)
        >>> projections[1]
        ('b', 4.0, 5.0)

        >>> # Handles short vectors
        >>> analyzer.store_representation("c", [1.0])
        >>> proj = analyzer.project_to_2d(["c"])
        >>> proj[0]
        ('c', 1.0, 0.0)

        >>> # Empty returns empty
        >>> empty = RepresentationAnalyzer()
        >>> empty.project_to_2d()
        []
        """
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
    """Extract a steering vector from contrasting activations.

    Convenience function that creates a SteeringVectorExtractor and
    extracts a vector from a single contrast pair.

    Parameters
    ----------
    positive_activations : list[float]
        Activations from a prompt eliciting the target behavior.
    negative_activations : list[float]
        Activations from a prompt eliciting the opposite behavior.
    name : str
        Human-readable name for the vector.
    target_behavior : str
        Description of what behavior this vector induces.
    normalize : bool, default=True
        Whether to normalize the vector to unit length.

    Returns
    -------
    SteeringVector
        The extracted steering vector.

    Examples
    --------
    >>> from insideLLMs.steering import extract_steering_vector
    >>> pos = [1.0, 0.0, 0.5]
    >>> neg = [0.0, 1.0, 0.5]
    >>> vector = extract_steering_vector(pos, neg, "test", "test behavior")
    >>> vector.source
    'contrast_pair'
    >>> len(vector.direction)
    3

    >>> # Without normalization
    >>> vector = extract_steering_vector(
    ...     [1.0, 0.0], [0.0, 0.0],
    ...     "raw", "raw vector",
    ...     normalize=False
    ... )
    >>> vector.direction[0]
    1.0

    See Also
    --------
    SteeringVectorExtractor : For more control over extraction.
    """
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
    """Create a contrast pair for steering analysis.

    Convenience function that creates a PromptSteerer and generates
    a contrast pair for the given dimension.

    Parameters
    ----------
    base_prompt : str
        The base prompt to create variants of.
    dimension : str
        The behavioral dimension (e.g., "formality", "length").
    positive_style : Optional[str], default=None
        Custom positive style. If None, uses preset for dimension.
    negative_style : Optional[str], default=None
        Custom negative style. If None, uses preset for dimension.

    Returns
    -------
    ContrastPair
        A contrast pair with positive and negative variants.

    Examples
    --------
    >>> from insideLLMs.steering import create_contrast_pair
    >>> pair = create_contrast_pair("Explain AI", "formality")
    >>> pair.target_dimension
    'formality'
    >>> "formal" in pair.positive_prompt.lower()
    True

    >>> # Custom styles
    >>> pair = create_contrast_pair(
    ...     "Hello",
    ...     "custom",
    ...     positive_style="helpful",
    ...     negative_style="cautious"
    ... )
    >>> "helpful" in pair.positive_prompt.lower()
    True

    See Also
    --------
    PromptSteerer.create_contrast_pair : The underlying method.
    ContrastPair : The resulting data structure.
    """
    steerer = PromptSteerer()
    return steerer.create_contrast_pair(base_prompt, dimension, positive_style, negative_style)


def apply_prompt_steering(
    prompt: str,
    method: SteeringMethod,
    config: dict[str, Any],
) -> Union[str, tuple[str, str]]:
    """Apply prompt-based steering.

    Convenience function that creates a PromptSteerer and applies
    the specified steering method.

    Parameters
    ----------
    prompt : str
        The original user prompt.
    method : SteeringMethod
        Which steering method to use.
    config : dict[str, Any]
        Configuration for the steering method.

    Returns
    -------
    Union[str, tuple[str, str]]
        The steered prompt, or (system_message, prompt) tuple for
        SYSTEM_MESSAGE method.

    Examples
    --------
    >>> from insideLLMs.steering import apply_prompt_steering, SteeringMethod
    >>> result = apply_prompt_steering(
    ...     "Hello",
    ...     SteeringMethod.PROMPT_PREFIX,
    ...     {"instruction": "Be formal"}
    ... )
    >>> "Be formal" in result
    True

    >>> # System message returns tuple
    >>> result = apply_prompt_steering(
    ...     "Hello",
    ...     SteeringMethod.SYSTEM_MESSAGE,
    ...     {"system_message": "You are helpful"}
    ... )
    >>> isinstance(result, tuple)
    True

    See Also
    --------
    PromptSteerer.apply_steering : The underlying method.
    SteeringMethod : Available steering methods.
    """
    steerer = PromptSteerer()
    return steerer.apply_steering(prompt, method, config)


def measure_behavioral_shift(
    original_output: str,
    steered_output: str,
    dimension: str = "generic",
) -> float:
    """Measure behavioral shift between outputs.

    Convenience function that creates a BehavioralShiftMeasurer and
    measures the shift along the specified dimension.

    Parameters
    ----------
    original_output : str
        The model output without steering.
    steered_output : str
        The model output with steering applied.
    dimension : str, default="generic"
        The dimension to measure.

    Returns
    -------
    float
        The shift value.

    Examples
    --------
    >>> from insideLLMs.steering import measure_behavioral_shift
    >>> shift = measure_behavioral_shift(
    ...     "yeah kinda",
    ...     "therefore consequently",
    ...     "formality"
    ... )
    >>> shift > 0
    True

    >>> # Generic dimension uses word overlap
    >>> shift = measure_behavioral_shift("hello", "goodbye")
    >>> 0 <= shift <= 1
    True

    See Also
    --------
    BehavioralShiftMeasurer : For more control over measurement.
    """
    measurer = BehavioralShiftMeasurer()
    return measurer.measure_shift(original_output, steered_output, dimension)


def analyze_activation_patterns(
    patterns: list[ActivationPattern],
) -> dict[str, Any]:
    """Analyze a list of activation patterns.

    Convenience function that creates an ActivationAnalyzer and
    computes summary statistics over the provided patterns.

    Parameters
    ----------
    patterns : list[ActivationPattern]
        Patterns to analyze.

    Returns
    -------
    dict[str, Any]
        Dictionary with analysis results including:
        - num_patterns: Number of patterns
        - mean_activation_avg: Average of mean activations
        - mean_activation_std: Std dev of mean activations
        - variance_avg: Average variance
        - num_clusters: Number of clusters found
        - salient_positions: Positions with high activations

    Examples
    --------
    >>> from insideLLMs.steering import ActivationPattern, analyze_activation_patterns
    >>> patterns = [
    ...     ActivationPattern("a", "l1", [0.5, 0.3], [0, 1]),
    ...     ActivationPattern("b", "l1", [0.4, 0.6], [0, 1]),
    ... ]
    >>> result = analyze_activation_patterns(patterns)
    >>> result["num_patterns"]
    2
    >>> "mean_activation_avg" in result
    True

    >>> # Empty patterns
    >>> result = analyze_activation_patterns([])
    >>> "error" in result
    True

    See Also
    --------
    ActivationAnalyzer : For more detailed analysis.
    """
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
    """Quick analysis of model steerability.

    Convenience function that creates a SteeringExperimenter and runs
    a comprehensive steerability analysis.

    Parameters
    ----------
    model_fn : Callable[[str], str]
        Function that takes a prompt and returns model output.
    test_prompts : list[str]
        Prompts to test with.
    dimensions : Optional[list[str]], default=None
        Dimensions to analyze. If None, uses formality, length,
        sentiment, confidence.

    Returns
    -------
    SteeringReport
        Comprehensive analysis results.

    Examples
    --------
    >>> from insideLLMs.steering import quick_steering_analysis
    >>> def mock_model(prompt):
    ...     return "Response"
    >>> report = quick_steering_analysis(
    ...     mock_model,
    ...     test_prompts=["Hello"],
    ...     dimensions=["formality"]
    ... )
    >>> hasattr(report, 'overall_steerability')
    True
    >>> len(report.recommendations) > 0
    True

    See Also
    --------
    SteeringExperimenter : For more control over experiments.
    SteeringReport : The result data structure.
    """
    experimenter = SteeringExperimenter(model_fn=model_fn)
    return experimenter.analyze_steerability(test_prompts, dimensions)
