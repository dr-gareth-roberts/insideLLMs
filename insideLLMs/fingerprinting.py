"""Model capability fingerprinting for LLM evaluation.

This module provides tools for systematically assessing and categorizing
the capabilities of language models:

- Capability detection and assessment
- Skill profiling across domains
- Limitation identification
- Comparative capability analysis
- Fingerprint generation and matching
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class CapabilityCategory(Enum):
    """Categories of model capabilities."""

    REASONING = "reasoning"
    LANGUAGE = "language"
    KNOWLEDGE = "knowledge"
    CREATIVITY = "creativity"
    CODING = "coding"
    MATH = "math"
    ANALYSIS = "analysis"
    INSTRUCTION_FOLLOWING = "instruction_following"


class CapabilityLevel(Enum):
    """Proficiency level for a capability."""

    NONE = "none"
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SkillType(Enum):
    """Types of specific skills."""

    # Reasoning
    LOGICAL_DEDUCTION = "logical_deduction"
    CAUSAL_REASONING = "causal_reasoning"
    ANALOGICAL_REASONING = "analogical_reasoning"
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"

    # Language
    GRAMMAR = "grammar"
    VOCABULARY = "vocabulary"
    STYLE_ADAPTATION = "style_adaptation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"

    # Knowledge
    FACTUAL_RECALL = "factual_recall"
    TEMPORAL_KNOWLEDGE = "temporal_knowledge"
    DOMAIN_EXPERTISE = "domain_expertise"
    COMMON_SENSE = "common_sense"

    # Creativity
    CREATIVE_WRITING = "creative_writing"
    BRAINSTORMING = "brainstorming"
    HUMOR = "humor"
    STORYTELLING = "storytelling"

    # Coding
    CODE_GENERATION = "code_generation"
    CODE_UNDERSTANDING = "code_understanding"
    DEBUGGING = "debugging"
    CODE_TRANSLATION = "code_translation"

    # Math
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    WORD_PROBLEMS = "word_problems"
    SYMBOLIC_MANIPULATION = "symbolic_manipulation"

    # Analysis
    DATA_ANALYSIS = "data_analysis"
    TEXT_ANALYSIS = "text_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    CRITICAL_EVALUATION = "critical_evaluation"

    # Instruction following
    TASK_DECOMPOSITION = "task_decomposition"
    FORMAT_ADHERENCE = "format_adherence"
    CONSTRAINT_HANDLING = "constraint_handling"
    MULTI_STEP_EXECUTION = "multi_step_execution"


# =============================================================================
# Helper Functions
# =============================================================================


def _score_to_level(score: float) -> CapabilityLevel:
    """Convert a 0-1 score to a capability level.

    Args:
        score: A score between 0 and 1.

    Returns:
        The corresponding capability level.
    """
    if score >= 0.9:
        return CapabilityLevel.EXPERT
    elif score >= 0.75:
        return CapabilityLevel.ADVANCED
    elif score >= 0.5:
        return CapabilityLevel.INTERMEDIATE
    elif score >= 0.25:
        return CapabilityLevel.BASIC
    return CapabilityLevel.NONE


# =============================================================================
# Enums (continued)
# =============================================================================


class LimitationType(Enum):
    """Types of model limitations."""

    FACTUAL_ERROR_PRONE = "factual_error_prone"
    CONTEXT_LENGTH_LIMITED = "context_length_limited"
    CALCULATION_ERRORS = "calculation_errors"
    INSTRUCTION_CONFUSION = "instruction_confusion"
    REPETITIVE_OUTPUT = "repetitive_output"
    KNOWLEDGE_CUTOFF = "knowledge_cutoff"
    HALLUCINATION_PRONE = "hallucination_prone"
    FORMAT_INCONSISTENT = "format_inconsistent"


@dataclass
class CapabilityScore:
    """Score for a specific capability."""

    category: CapabilityCategory
    skill: SkillType
    score: float  # 0-1
    level: CapabilityLevel
    confidence: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "skill": self.skill.value,
            "score": self.score,
            "level": self.level.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class LimitationReport:
    """Report on a detected limitation."""

    limitation_type: LimitationType
    severity: float  # 0-1
    frequency: float  # How often observed
    examples: list[str]
    mitigation_suggestions: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.limitation_type.value,
            "severity": self.severity,
            "frequency": self.frequency,
            "examples": self.examples,
            "mitigation_suggestions": self.mitigation_suggestions,
        }


@dataclass
class SkillProfile:
    """Profile of skills in a category."""

    category: CapabilityCategory
    skills: dict[SkillType, CapabilityScore]
    overall_level: CapabilityLevel
    overall_score: float
    strengths: list[SkillType]
    weaknesses: list[SkillType]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "skills": {k.value: v.to_dict() for k, v in self.skills.items()},
            "overall_level": self.overall_level.value,
            "overall_score": self.overall_score,
            "strengths": [s.value for s in self.strengths],
            "weaknesses": [w.value for w in self.weaknesses],
        }


@dataclass
class ModelFingerprint:
    """Complete fingerprint of a model's capabilities."""

    model_id: str
    version: str
    fingerprint_hash: str
    category_scores: dict[CapabilityCategory, float]
    skill_profiles: dict[CapabilityCategory, SkillProfile]
    limitations: list[LimitationReport]
    top_capabilities: list[tuple[SkillType, float]]
    main_limitations: list[LimitationType]
    overall_capability_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def capability_signature(self) -> str:
        """Get a compact capability signature."""
        scores = sorted(self.category_scores.items(), key=lambda x: x[1], reverse=True)
        parts = [f"{cat.value[:3]}:{score:.2f}" for cat, score in scores[:5]]
        return "|".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "fingerprint_hash": self.fingerprint_hash,
            "category_scores": {k.value: v for k, v in self.category_scores.items()},
            "skill_profiles": {k.value: v.to_dict() for k, v in self.skill_profiles.items()},
            "limitations": [limitation.to_dict() for limitation in self.limitations],
            "top_capabilities": [
                {"skill": s.value, "score": sc} for s, sc in self.top_capabilities
            ],
            "main_limitations": [limitation.value for limitation in self.main_limitations],
            "overall_capability_score": self.overall_capability_score,
            "capability_signature": self.capability_signature,
            "metadata": self.metadata,
        }


@dataclass
class FingerprintComparison:
    """Comparison between two model fingerprints."""

    model_a_id: str
    model_b_id: str
    similarity_score: float
    category_differences: dict[CapabilityCategory, float]
    skill_differences: dict[SkillType, float]
    model_a_advantages: list[SkillType]
    model_b_advantages: list[SkillType]
    shared_strengths: list[SkillType]
    shared_weaknesses: list[SkillType]

    @property
    def winner(self) -> str | None:
        """Get overall better model, if clear."""
        a_wins = len(self.model_a_advantages)
        b_wins = len(self.model_b_advantages)
        if a_wins > b_wins * 1.5:
            return self.model_a_id
        elif b_wins > a_wins * 1.5:
            return self.model_b_id
        return None  # Too close to call

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_a_id": self.model_a_id,
            "model_b_id": self.model_b_id,
            "similarity_score": self.similarity_score,
            "category_differences": {k.value: v for k, v in self.category_differences.items()},
            "skill_differences": {k.value: v for k, v in self.skill_differences.items()},
            "model_a_advantages": [s.value for s in self.model_a_advantages],
            "model_b_advantages": [s.value for s in self.model_b_advantages],
            "shared_strengths": [s.value for s in self.shared_strengths],
            "shared_weaknesses": [s.value for s in self.shared_weaknesses],
            "winner": self.winner,
        }


class CapabilityProbe:
    """Base class for capability probes."""

    def __init__(
        self,
        category: CapabilityCategory,
        skill: SkillType,
        name: str,
        description: str,
    ):
        """Initialize probe.

        Args:
            category: Capability category
            skill: Skill type being tested
            name: Probe name
            description: Probe description
        """
        self.category = category
        self.skill = skill
        self.name = name
        self.description = description

    def generate_test(self) -> tuple[str, Any]:
        """Generate a test prompt and expected answer.

        Returns:
            Tuple of (prompt, expected_answer)
        """
        raise NotImplementedError

    def evaluate(self, response: str, expected: Any) -> float:
        """Evaluate response against expected.

        Args:
            response: Model response
            expected: Expected answer

        Returns:
            Score between 0 and 1
        """
        raise NotImplementedError


class ReasoningProbe(CapabilityProbe):
    """Probe for reasoning capabilities."""

    def __init__(self, skill: SkillType):
        """Initialize reasoning probe."""
        super().__init__(
            category=CapabilityCategory.REASONING,
            skill=skill,
            name=f"reasoning_{skill.value}",
            description=f"Tests {skill.value} capability",
        )

    def generate_test(self) -> tuple[str, Any]:
        """Generate reasoning test."""
        if self.skill == SkillType.LOGICAL_DEDUCTION:
            return (
                "If all roses are flowers, and some flowers fade quickly, "
                "can we conclude that some roses fade quickly?",
                False,  # No, this is an invalid syllogism
            )
        elif self.skill == SkillType.CAUSAL_REASONING:
            return (
                "The street is wet. It might have rained, or a sprinkler was on. "
                "What additional observation would help determine the cause?",
                "check_other_areas",  # Check if other areas are also wet
            )
        return ("Test reasoning", "expected")

    def evaluate(self, response: str, expected: Any) -> float:
        """Evaluate reasoning response."""
        response_lower = response.lower()

        if self.skill == SkillType.LOGICAL_DEDUCTION:
            # Check for correct reasoning about syllogism
            if expected is False:
                if "no" in response_lower or "cannot" in response_lower:
                    return 1.0
                elif "yes" in response_lower:
                    return 0.0
            return 0.5

        return 0.5  # Default moderate score


class SkillAssessor:
    """Assess individual skills."""

    def __init__(
        self,
        evaluator: Callable[[str, str], float] | None = None,
    ):
        """Initialize assessor.

        Args:
            evaluator: Optional custom evaluation function
        """
        self._evaluator = evaluator or self._default_evaluator
        self._probes: dict[SkillType, list[CapabilityProbe]] = {}

    @staticmethod
    def _default_evaluator(response: str, expected: str) -> float:
        """Default evaluation using keyword matching."""
        response_lower = response.lower()
        expected_lower = str(expected).lower()

        # Check for exact match
        if expected_lower in response_lower:
            return 1.0

        # Check for partial match
        expected_words = set(expected_lower.split())
        response_words = set(response_lower.split())
        overlap = len(expected_words & response_words)
        if expected_words:
            return overlap / len(expected_words)
        return 0.5

    def register_probe(self, probe: CapabilityProbe) -> None:
        """Register a capability probe.

        Args:
            probe: Probe to register
        """
        if probe.skill not in self._probes:
            self._probes[probe.skill] = []
        self._probes[probe.skill].append(probe)

    def assess_skill(
        self,
        skill: SkillType,
        responses: list[tuple[str, Any]],
    ) -> CapabilityScore:
        """Assess a skill based on responses.

        Args:
            skill: Skill to assess
            responses: List of (response, expected) tuples

        Returns:
            CapabilityScore for the skill
        """
        if not responses:
            return CapabilityScore(
                category=self._get_category(skill),
                skill=skill,
                score=0.0,
                level=CapabilityLevel.NONE,
                confidence=0.0,
                evidence=[],
            )

        scores = []
        evidence = []
        for response, expected in responses:
            score = self._evaluator(response, str(expected))
            scores.append(score)
            if score >= 0.8:
                evidence.append(f"Correct response for {skill.value}")
            elif score <= 0.2:
                evidence.append(f"Incorrect response for {skill.value}")

        avg_score = sum(scores) / len(scores)
        level = _score_to_level(avg_score)
        confidence = min(1.0, len(responses) / 5)  # More samples = higher confidence

        return CapabilityScore(
            category=self._get_category(skill),
            skill=skill,
            score=avg_score,
            level=level,
            confidence=confidence,
            evidence=evidence[:3],  # Keep top 3
        )

    @staticmethod
    def _get_category(skill: SkillType) -> CapabilityCategory:
        """Get category for a skill."""
        skill_categories = {
            SkillType.LOGICAL_DEDUCTION: CapabilityCategory.REASONING,
            SkillType.CAUSAL_REASONING: CapabilityCategory.REASONING,
            SkillType.ANALOGICAL_REASONING: CapabilityCategory.REASONING,
            SkillType.COUNTERFACTUAL_REASONING: CapabilityCategory.REASONING,
            SkillType.GRAMMAR: CapabilityCategory.LANGUAGE,
            SkillType.VOCABULARY: CapabilityCategory.LANGUAGE,
            SkillType.STYLE_ADAPTATION: CapabilityCategory.LANGUAGE,
            SkillType.TRANSLATION: CapabilityCategory.LANGUAGE,
            SkillType.SUMMARIZATION: CapabilityCategory.LANGUAGE,
            SkillType.FACTUAL_RECALL: CapabilityCategory.KNOWLEDGE,
            SkillType.TEMPORAL_KNOWLEDGE: CapabilityCategory.KNOWLEDGE,
            SkillType.DOMAIN_EXPERTISE: CapabilityCategory.KNOWLEDGE,
            SkillType.COMMON_SENSE: CapabilityCategory.KNOWLEDGE,
            SkillType.CREATIVE_WRITING: CapabilityCategory.CREATIVITY,
            SkillType.BRAINSTORMING: CapabilityCategory.CREATIVITY,
            SkillType.HUMOR: CapabilityCategory.CREATIVITY,
            SkillType.STORYTELLING: CapabilityCategory.CREATIVITY,
            SkillType.CODE_GENERATION: CapabilityCategory.CODING,
            SkillType.CODE_UNDERSTANDING: CapabilityCategory.CODING,
            SkillType.DEBUGGING: CapabilityCategory.CODING,
            SkillType.CODE_TRANSLATION: CapabilityCategory.CODING,
            SkillType.ARITHMETIC: CapabilityCategory.MATH,
            SkillType.ALGEBRA: CapabilityCategory.MATH,
            SkillType.WORD_PROBLEMS: CapabilityCategory.MATH,
            SkillType.SYMBOLIC_MANIPULATION: CapabilityCategory.MATH,
            SkillType.DATA_ANALYSIS: CapabilityCategory.ANALYSIS,
            SkillType.TEXT_ANALYSIS: CapabilityCategory.ANALYSIS,
            SkillType.PATTERN_RECOGNITION: CapabilityCategory.ANALYSIS,
            SkillType.CRITICAL_EVALUATION: CapabilityCategory.ANALYSIS,
            SkillType.TASK_DECOMPOSITION: CapabilityCategory.INSTRUCTION_FOLLOWING,
            SkillType.FORMAT_ADHERENCE: CapabilityCategory.INSTRUCTION_FOLLOWING,
            SkillType.CONSTRAINT_HANDLING: CapabilityCategory.INSTRUCTION_FOLLOWING,
            SkillType.MULTI_STEP_EXECUTION: CapabilityCategory.INSTRUCTION_FOLLOWING,
        }
        return skill_categories.get(skill, CapabilityCategory.REASONING)


class LimitationDetector:
    """Detect model limitations."""

    def __init__(self):
        """Initialize detector."""
        self._patterns: dict[LimitationType, list[Callable[[str], bool]]] = {
            LimitationType.REPETITIVE_OUTPUT: [self._check_repetition],
            LimitationType.FORMAT_INCONSISTENT: [self._check_format],
            LimitationType.HALLUCINATION_PRONE: [self._check_hallucination_markers],
        }

    @staticmethod
    def _check_repetition(response: str) -> bool:
        """Check for repetitive patterns."""
        words = response.lower().split()
        if len(words) < 10:
            return False
        # Check for repeated phrases
        for i in range(len(words) - 3):
            phrase = " ".join(words[i : i + 3])
            remaining = " ".join(words[i + 3 :])
            if phrase in remaining:
                return True
        return False

    @staticmethod
    def _check_format(response: str) -> bool:
        """Check for format inconsistencies."""
        # Simple heuristic: inconsistent list markers
        lines = response.split("\n")
        markers_found: set[str] = set()
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                markers_found.add("dash")
            elif line.startswith("* "):
                markers_found.add("asterisk")
            elif line and line[0].isdigit() and "." in line[:3]:
                markers_found.add("number")
        return len(markers_found) > 1  # Multiple markers = inconsistent

    @staticmethod
    def _check_hallucination_markers(response: str) -> bool:
        """Check for hallucination marker phrases."""
        hallucination_markers = [
            "i believe",
            "i think",
            "approximately",
            "around",
            "roughly",
            "possibly",
            "might be",
        ]
        response_lower = response.lower()
        count = sum(1 for m in hallucination_markers if m in response_lower)
        return count >= 3  # Multiple hedging phrases

    def detect(
        self,
        responses: list[str],
    ) -> list[LimitationReport]:
        """Detect limitations from responses.

        Args:
            responses: List of model responses

        Returns:
            List of LimitationReport objects
        """
        reports = []

        for limitation_type, checkers in self._patterns.items():
            detected_count = 0
            examples = []

            for response in responses:
                for checker in checkers:
                    if checker(response):
                        detected_count += 1
                        if len(examples) < 3:
                            examples.append(response[:100] + "...")
                        break

            if detected_count > 0:
                frequency = detected_count / len(responses)
                severity = min(1.0, frequency * 2)  # Higher frequency = higher severity

                reports.append(
                    LimitationReport(
                        limitation_type=limitation_type,
                        severity=severity,
                        frequency=frequency,
                        examples=examples,
                        mitigation_suggestions=self._get_mitigations(limitation_type),
                    )
                )

        return reports

    @staticmethod
    def _get_mitigations(limitation: LimitationType) -> list[str]:
        """Get mitigation suggestions for a limitation."""
        mitigations = {
            LimitationType.REPETITIVE_OUTPUT: [
                "Increase temperature parameter",
                "Add diversity penalties",
                "Use different sampling methods",
            ],
            LimitationType.FORMAT_INCONSISTENT: [
                "Provide explicit format examples",
                "Use few-shot prompting",
                "Add format constraints to prompt",
            ],
            LimitationType.HALLUCINATION_PRONE: [
                "Request citations or sources",
                "Use retrieval augmentation",
                "Ask for confidence levels",
            ],
            LimitationType.FACTUAL_ERROR_PRONE: [
                "Use fact-checking prompts",
                "Request step-by-step reasoning",
                "Validate with external sources",
            ],
            LimitationType.CALCULATION_ERRORS: [
                "Request step-by-step calculations",
                "Use external calculator tools",
                "Ask for verification of results",
            ],
            LimitationType.KNOWLEDGE_CUTOFF: [
                "Provide current context",
                "Use retrieval augmentation",
                "Explicitly mention timeframe",
            ],
        }
        return mitigations.get(limitation, ["Review model documentation"])


class CapabilityProfiler:
    """Build skill profiles for capability categories."""

    def __init__(
        self,
        skill_assessor: SkillAssessor | None = None,
    ):
        """Initialize profiler.

        Args:
            skill_assessor: Optional custom skill assessor
        """
        self._assessor = skill_assessor or SkillAssessor()

    def profile_category(
        self,
        category: CapabilityCategory,
        skill_results: dict[SkillType, list[tuple[str, Any]]],
    ) -> SkillProfile:
        """Build profile for a category.

        Args:
            category: Category to profile
            skill_results: Results for each skill (response, expected) pairs

        Returns:
            SkillProfile for the category
        """
        skill_scores = {}
        for skill, results in skill_results.items():
            if self._is_skill_in_category(skill, category):
                skill_scores[skill] = self._assessor.assess_skill(skill, results)

        if not skill_scores:
            return SkillProfile(
                category=category,
                skills={},
                overall_level=CapabilityLevel.NONE,
                overall_score=0.0,
                strengths=[],
                weaknesses=[],
            )

        # Calculate overall score
        scores = [s.score for s in skill_scores.values()]
        overall_score = sum(scores) / len(scores)
        overall_level = _score_to_level(overall_score)

        # Identify strengths and weaknesses
        sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1].score, reverse=True)
        strengths = [s for s, score in sorted_skills if score.score >= 0.7][:3]
        weaknesses = [s for s, score in sorted_skills if score.score < 0.4][:3]

        return SkillProfile(
            category=category,
            skills=skill_scores,
            overall_level=overall_level,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
        )

    @staticmethod
    def _is_skill_in_category(skill: SkillType, category: CapabilityCategory) -> bool:
        """Check if skill belongs to category."""
        category_skills = {
            CapabilityCategory.REASONING: {
                SkillType.LOGICAL_DEDUCTION,
                SkillType.CAUSAL_REASONING,
                SkillType.ANALOGICAL_REASONING,
                SkillType.COUNTERFACTUAL_REASONING,
            },
            CapabilityCategory.LANGUAGE: {
                SkillType.GRAMMAR,
                SkillType.VOCABULARY,
                SkillType.STYLE_ADAPTATION,
                SkillType.TRANSLATION,
                SkillType.SUMMARIZATION,
            },
            CapabilityCategory.KNOWLEDGE: {
                SkillType.FACTUAL_RECALL,
                SkillType.TEMPORAL_KNOWLEDGE,
                SkillType.DOMAIN_EXPERTISE,
                SkillType.COMMON_SENSE,
            },
            CapabilityCategory.CREATIVITY: {
                SkillType.CREATIVE_WRITING,
                SkillType.BRAINSTORMING,
                SkillType.HUMOR,
                SkillType.STORYTELLING,
            },
            CapabilityCategory.CODING: {
                SkillType.CODE_GENERATION,
                SkillType.CODE_UNDERSTANDING,
                SkillType.DEBUGGING,
                SkillType.CODE_TRANSLATION,
            },
            CapabilityCategory.MATH: {
                SkillType.ARITHMETIC,
                SkillType.ALGEBRA,
                SkillType.WORD_PROBLEMS,
                SkillType.SYMBOLIC_MANIPULATION,
            },
            CapabilityCategory.ANALYSIS: {
                SkillType.DATA_ANALYSIS,
                SkillType.TEXT_ANALYSIS,
                SkillType.PATTERN_RECOGNITION,
                SkillType.CRITICAL_EVALUATION,
            },
            CapabilityCategory.INSTRUCTION_FOLLOWING: {
                SkillType.TASK_DECOMPOSITION,
                SkillType.FORMAT_ADHERENCE,
                SkillType.CONSTRAINT_HANDLING,
                SkillType.MULTI_STEP_EXECUTION,
            },
        }
        return skill in category_skills.get(category, set())


class FingerprintGenerator:
    """Generate model fingerprints."""

    def __init__(
        self,
        profiler: CapabilityProfiler | None = None,
        limitation_detector: LimitationDetector | None = None,
    ):
        """Initialize generator.

        Args:
            profiler: Optional custom capability profiler
            limitation_detector: Optional custom limitation detector
        """
        self._profiler = profiler or CapabilityProfiler()
        self._detector = limitation_detector or LimitationDetector()

    def generate(
        self,
        model_id: str,
        skill_results: dict[SkillType, list[tuple[str, Any]]],
        responses: list[str],
        version: str = "1.0",
        metadata: dict[str, Any] | None = None,
    ) -> ModelFingerprint:
        """Generate fingerprint for a model.

        Args:
            model_id: Model identifier
            skill_results: Test results by skill
            responses: Raw responses for limitation detection
            version: Fingerprint version
            metadata: Optional metadata

        Returns:
            ModelFingerprint object
        """
        # Build skill profiles for each category
        skill_profiles = {}
        category_scores = {}

        for category in CapabilityCategory:
            profile = self._profiler.profile_category(category, skill_results)
            skill_profiles[category] = profile
            category_scores[category] = profile.overall_score

        # Detect limitations
        limitations = self._detector.detect(responses)

        # Identify top capabilities
        all_skills = []
        for profile in skill_profiles.values():
            for skill, score in profile.skills.items():
                all_skills.append((skill, score.score))
        top_capabilities = sorted(all_skills, key=lambda x: x[1], reverse=True)[:5]

        # Identify main limitations
        main_limitations = [
            limitation.limitation_type for limitation in limitations if limitation.severity > 0.5
        ]

        # Calculate overall score
        if category_scores:
            overall_score = sum(category_scores.values()) / len(category_scores)
        else:
            overall_score = 0.0

        # Generate fingerprint hash
        fingerprint_hash = self._generate_hash(model_id, category_scores, top_capabilities)

        return ModelFingerprint(
            model_id=model_id,
            version=version,
            fingerprint_hash=fingerprint_hash,
            category_scores=category_scores,
            skill_profiles=skill_profiles,
            limitations=limitations,
            top_capabilities=top_capabilities,
            main_limitations=main_limitations,
            overall_capability_score=overall_score,
            metadata=metadata or {},
        )

    @staticmethod
    def _generate_hash(
        model_id: str,
        category_scores: dict[CapabilityCategory, float],
        top_capabilities: list[tuple[SkillType, float]],
    ) -> str:
        """Generate unique hash for fingerprint."""
        data = {
            "model_id": model_id,
            "scores": {k.value: round(v, 2) for k, v in category_scores.items()},
            "top": [(s.value, round(sc, 2)) for s, sc in top_capabilities],
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]


class FingerprintComparator:
    """Compare model fingerprints."""

    def compare(
        self,
        fingerprint_a: ModelFingerprint,
        fingerprint_b: ModelFingerprint,
    ) -> FingerprintComparison:
        """Compare two fingerprints.

        Args:
            fingerprint_a: First fingerprint
            fingerprint_b: Second fingerprint

        Returns:
            FingerprintComparison object
        """
        # Calculate category differences
        category_diffs = {}
        for category in CapabilityCategory:
            score_a = fingerprint_a.category_scores.get(category, 0)
            score_b = fingerprint_b.category_scores.get(category, 0)
            category_diffs[category] = score_a - score_b

        # Calculate skill differences
        skill_diffs = {}
        all_skills: set[SkillType] = set()
        for profile in fingerprint_a.skill_profiles.values():
            all_skills.update(profile.skills.keys())
        for profile in fingerprint_b.skill_profiles.values():
            all_skills.update(profile.skills.keys())

        for skill in all_skills:
            score_a = self._get_skill_score(fingerprint_a, skill)
            score_b = self._get_skill_score(fingerprint_b, skill)
            skill_diffs[skill] = score_a - score_b

        # Identify advantages
        threshold = 0.1
        model_a_advantages = [s for s, d in skill_diffs.items() if d > threshold]
        model_b_advantages = [s for s, d in skill_diffs.items() if d < -threshold]

        # Identify shared traits
        shared_strengths = []
        shared_weaknesses = []
        for skill, diff in skill_diffs.items():
            if abs(diff) < threshold:
                score_a = self._get_skill_score(fingerprint_a, skill)
                if score_a >= 0.7:
                    shared_strengths.append(skill)
                elif score_a < 0.4:
                    shared_weaknesses.append(skill)

        # Calculate overall similarity
        if category_diffs:
            total_diff = sum(abs(d) for d in category_diffs.values())
            similarity = max(0, 1 - total_diff / len(category_diffs))
        else:
            similarity = 1.0

        return FingerprintComparison(
            model_a_id=fingerprint_a.model_id,
            model_b_id=fingerprint_b.model_id,
            similarity_score=similarity,
            category_differences=category_diffs,
            skill_differences=skill_diffs,
            model_a_advantages=model_a_advantages,
            model_b_advantages=model_b_advantages,
            shared_strengths=shared_strengths,
            shared_weaknesses=shared_weaknesses,
        )

    @staticmethod
    def _get_skill_score(fingerprint: ModelFingerprint, skill: SkillType) -> float:
        """Get score for a skill from fingerprint."""
        for profile in fingerprint.skill_profiles.values():
            if skill in profile.skills:
                return profile.skills[skill].score
        return 0.0


# Convenience functions


def create_fingerprint(
    model_id: str,
    skill_results: dict[SkillType, list[tuple[str, Any]]],
    responses: list[str],
    version: str = "1.0",
) -> ModelFingerprint:
    """Create a model fingerprint.

    Args:
        model_id: Model identifier
        skill_results: Test results by skill
        responses: Raw responses for limitation detection
        version: Fingerprint version

    Returns:
        ModelFingerprint object
    """
    generator = FingerprintGenerator()
    return generator.generate(model_id, skill_results, responses, version)


def compare_fingerprints(
    fingerprint_a: ModelFingerprint,
    fingerprint_b: ModelFingerprint,
) -> FingerprintComparison:
    """Compare two model fingerprints.

    Args:
        fingerprint_a: First fingerprint
        fingerprint_b: Second fingerprint

    Returns:
        FingerprintComparison object
    """
    comparator = FingerprintComparator()
    return comparator.compare(fingerprint_a, fingerprint_b)


def quick_capability_assessment(
    responses: dict[str, tuple[str, Any]],
) -> dict[str, Any]:
    """Quick capability assessment from responses.

    Args:
        responses: Dictionary of skill_name -> (response, expected)

    Returns:
        Dictionary with quick assessment results
    """
    assessor = SkillAssessor()
    results = {}

    for skill_name, (response, expected) in responses.items():
        # Try to match skill name to SkillType
        try:
            skill = SkillType(skill_name)
        except ValueError:
            # Use first matching skill
            skill = SkillType.COMMON_SENSE  # Default

        score = assessor.assess_skill(skill, [(response, expected)])
        results[skill_name] = {
            "score": score.score,
            "level": score.level.value,
        }

    # Calculate overall
    overall = sum(r["score"] for r in results.values()) / len(results) if results else 0.0

    return {
        "skills": results,
        "overall_score": overall,
        "n_skills_tested": len(results),
    }


def detect_limitations(
    responses: list[str],
) -> list[dict[str, Any]]:
    """Detect limitations from responses.

    Args:
        responses: List of model responses

    Returns:
        List of limitation dictionaries
    """
    detector = LimitationDetector()
    reports = detector.detect(responses)
    return [r.to_dict() for r in reports]
