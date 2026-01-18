"""
Reasoning chain analysis and Chain-of-Thought evaluation utilities.

Provides tools for:
- Reasoning chain extraction and parsing
- Step-by-step logic validation
- Reasoning quality assessment
- CoT prompt generation
- Reasoning pattern detection
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ReasoningType(Enum):
    """Types of reasoning patterns."""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    MATHEMATICAL = "mathematical"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class StepType(Enum):
    """Types of reasoning steps."""

    PREMISE = "premise"
    INFERENCE = "inference"
    CALCULATION = "calculation"
    COMPARISON = "comparison"
    CONCLUSION = "conclusion"
    ASSUMPTION = "assumption"
    EVIDENCE = "evidence"
    EXAMPLE = "example"


class ReasoningQuality(Enum):
    """Quality levels of reasoning."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""

    content: str
    step_number: int
    step_type: StepType = StepType.INFERENCE
    confidence: float = 0.5
    supports_conclusion: bool = True
    depends_on: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "confidence": self.confidence,
            "supports_conclusion": self.supports_conclusion,
            "depends_on": self.depends_on,
        }


@dataclass
class ReasoningChain:
    """A complete chain of reasoning."""

    steps: list[ReasoningStep]
    conclusion: Optional[str] = None
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    is_valid: bool = True
    completeness: float = 0.0

    def get_step(self, step_number: int) -> Optional[ReasoningStep]:
        """Get step by number."""
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None

    def get_premises(self) -> list[ReasoningStep]:
        """Get all premise steps."""
        return [s for s in self.steps if s.step_type == StepType.PREMISE]

    def get_inferences(self) -> list[ReasoningStep]:
        """Get all inference steps."""
        return [s for s in self.steps if s.step_type == StepType.INFERENCE]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
            "reasoning_type": self.reasoning_type.value,
            "is_valid": self.is_valid,
            "completeness": self.completeness,
        }


@dataclass
class ChainAnalysis:
    """Analysis of a reasoning chain."""

    chain: ReasoningChain
    logical_validity: float
    coherence_score: float
    completeness_score: float
    step_quality_scores: list[float]
    identified_fallacies: list[str]
    missing_steps: list[str]
    overall_quality: ReasoningQuality

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain": self.chain.to_dict(),
            "logical_validity": self.logical_validity,
            "coherence_score": self.coherence_score,
            "completeness_score": self.completeness_score,
            "avg_step_quality": sum(self.step_quality_scores) / len(self.step_quality_scores)
            if self.step_quality_scores
            else 0,
            "num_fallacies": len(self.identified_fallacies),
            "identified_fallacies": self.identified_fallacies,
            "missing_steps": self.missing_steps,
            "overall_quality": self.overall_quality.value,
        }


@dataclass
class CoTEvaluation:
    """Evaluation of Chain-of-Thought prompting."""

    prompt: str
    response: str
    chain: ReasoningChain
    answer_correct: Optional[bool]
    reasoning_score: float
    step_accuracy: float
    explanation_quality: float
    improvements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "answer_correct": self.answer_correct,
            "reasoning_score": self.reasoning_score,
            "step_accuracy": self.step_accuracy,
            "explanation_quality": self.explanation_quality,
            "num_steps": len(self.chain.steps),
            "improvements": self.improvements,
        }


@dataclass
class ReasoningReport:
    """Report on reasoning capabilities."""

    total_evaluations: int
    avg_reasoning_score: float
    avg_step_accuracy: float
    reasoning_type_breakdown: dict[str, float]
    common_fallacies: list[tuple[str, int]]
    quality_distribution: dict[str, int]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_evaluations": self.total_evaluations,
            "avg_reasoning_score": self.avg_reasoning_score,
            "avg_step_accuracy": self.avg_step_accuracy,
            "reasoning_type_breakdown": self.reasoning_type_breakdown,
            "common_fallacies": self.common_fallacies,
            "quality_distribution": self.quality_distribution,
            "recommendations": self.recommendations,
        }


class ReasoningExtractor:
    """Extracts reasoning chains from text."""

    # Patterns for step detection
    STEP_PATTERNS = [
        r"(?:step|stage)\s*(\d+)[:\.]?\s*(.+?)(?=(?:step|stage)\s*\d+|$)",
        r"(\d+)[.)]\s*(.+?)(?=\d+[.)]|$)",
        r"(?:first|second|third|fourth|fifth|next|then|finally)[,:]?\s*(.+?)(?=(?:first|second|third|fourth|fifth|next|then|finally)|$)",
    ]

    # Markers for different step types
    PREMISE_MARKERS = ["given", "assume", "let", "suppose", "we know"]
    INFERENCE_MARKERS = ["therefore", "thus", "hence", "so", "this means"]
    CONCLUSION_MARKERS = ["therefore", "in conclusion", "finally", "the answer is"]
    CALCULATION_MARKERS = ["=", "calculate", "compute", "+", "-", "*", "/"]

    def extract(self, text: str) -> ReasoningChain:
        """Extract reasoning chain from text."""
        steps = []
        text.lower()

        # Try numbered step patterns
        for pattern in self.STEP_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches and len(matches) >= 2:
                for i, match in enumerate(matches):
                    content = match[-1].strip() if isinstance(match, tuple) else match.strip()

                    if content and len(content) > 5:
                        step_type = self._classify_step(content)
                        step = ReasoningStep(
                            content=content,
                            step_number=i + 1,
                            step_type=step_type,
                            confidence=self._estimate_confidence(content),
                        )
                        steps.append(step)
                break

        # If no numbered steps, try to split by sentences
        if not steps:
            steps = self._extract_from_sentences(text)

        # Find conclusion
        conclusion = self._extract_conclusion(text)

        # Determine reasoning type
        reasoning_type = self._classify_reasoning_type(text)

        # Calculate completeness
        completeness = self._calculate_completeness(steps, conclusion)

        return ReasoningChain(
            steps=steps,
            conclusion=conclusion,
            reasoning_type=reasoning_type,
            is_valid=len(steps) > 0,
            completeness=completeness,
        )

    def _extract_from_sentences(self, text: str) -> list[ReasoningStep]:
        """Extract steps from sentences."""
        sentences = re.split(r"[.!?]+", text)
        steps = []

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 10:
                step_type = self._classify_step(sentence)
                step = ReasoningStep(
                    content=sentence,
                    step_number=i + 1,
                    step_type=step_type,
                    confidence=self._estimate_confidence(sentence),
                )
                steps.append(step)

        return steps[:10]  # Limit to 10 steps

    def _classify_step(self, content: str) -> StepType:
        """Classify the type of reasoning step."""
        content_lower = content.lower()

        for marker in self.PREMISE_MARKERS:
            if marker in content_lower:
                return StepType.PREMISE

        for marker in self.CONCLUSION_MARKERS:
            if marker in content_lower:
                return StepType.CONCLUSION

        for marker in self.CALCULATION_MARKERS:
            if marker in content:
                return StepType.CALCULATION

        for marker in self.INFERENCE_MARKERS:
            if marker in content_lower:
                return StepType.INFERENCE

        return StepType.INFERENCE

    def _estimate_confidence(self, content: str) -> float:
        """Estimate confidence of a step."""
        confidence = 0.5
        content_lower = content.lower()

        # High confidence markers
        if any(m in content_lower for m in ["clearly", "obviously", "certainly"]):
            confidence += 0.2

        # Low confidence markers
        if any(m in content_lower for m in ["maybe", "perhaps", "possibly", "might"]):
            confidence -= 0.2

        # Evidence markers boost confidence
        if any(m in content_lower for m in ["because", "since", "as shown"]):
            confidence += 0.1

        return max(0.1, min(0.95, confidence))

    def _extract_conclusion(self, text: str) -> Optional[str]:
        """Extract the conclusion from text."""
        text_lower = text.lower()

        # Look for explicit conclusion markers
        patterns = [
            r"(?:therefore|thus|hence|so|in conclusion|finally)[,:]?\s*(.+?)(?:\.|$)",
            r"(?:the answer is|the result is)[:\s]*(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()

        # Return last sentence if no explicit conclusion
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if sentences:
            return sentences[-1]

        return None

    def _classify_reasoning_type(self, text: str) -> ReasoningType:
        """Classify the type of reasoning used."""
        text_lower = text.lower()

        # Mathematical reasoning
        if any(op in text for op in ["+", "-", "*", "/", "="]) or any(
            word in text_lower for word in ["calculate", "compute", "sum", "multiply"]
        ):
            return ReasoningType.MATHEMATICAL

        # Causal reasoning
        if any(
            word in text_lower for word in ["because", "cause", "effect", "leads to", "results in"]
        ):
            return ReasoningType.CAUSAL

        # Temporal reasoning
        if any(word in text_lower for word in ["before", "after", "then", "when", "during"]):
            return ReasoningType.TEMPORAL

        # Analogical reasoning
        if any(word in text_lower for word in ["like", "similar to", "just as", "analogous"]):
            return ReasoningType.ANALOGICAL

        # Inductive reasoning
        if any(word in text_lower for word in ["generally", "usually", "most", "pattern"]):
            return ReasoningType.INDUCTIVE

        # Default to deductive
        return ReasoningType.DEDUCTIVE

    def _calculate_completeness(
        self,
        steps: list[ReasoningStep],
        conclusion: Optional[str],
    ) -> float:
        """Calculate completeness of reasoning chain."""
        if not steps:
            return 0.0

        score = 0.0

        # Has multiple steps
        if len(steps) >= 2:
            score += 0.3

        # Has premise
        if any(s.step_type == StepType.PREMISE for s in steps):
            score += 0.2

        # Has inference
        if any(s.step_type == StepType.INFERENCE for s in steps):
            score += 0.2

        # Has conclusion
        if conclusion or any(s.step_type == StepType.CONCLUSION for s in steps):
            score += 0.3

        return min(1.0, score)


class ReasoningAnalyzer:
    """Analyzes reasoning chains for quality and validity."""

    # Common logical fallacies
    FALLACY_PATTERNS = {
        "circular_reasoning": ["because it is", "since it's true", "proves itself"],
        "hasty_generalization": ["all", "always", "never", "everyone", "nobody"],
        "false_dichotomy": ["either", "only two", "must be one or"],
        "appeal_to_authority": ["expert says", "scientists say", "studies show"],
        "ad_hominem": ["stupid", "idiot", "fool", "ignorant"],
        "straw_man": ["they think", "opponents believe", "critics say"],
        "slippery_slope": ["will lead to", "eventually", "if we allow"],
    }

    def analyze(self, chain: ReasoningChain) -> ChainAnalysis:
        """Analyze a reasoning chain."""
        # Calculate logical validity
        validity = self._check_logical_validity(chain)

        # Calculate coherence
        coherence = self._calculate_coherence(chain)

        # Calculate completeness
        completeness = chain.completeness

        # Score each step
        step_scores = [self._score_step(step) for step in chain.steps]

        # Identify fallacies
        fallacies = self._identify_fallacies(chain)

        # Identify missing steps
        missing = self._identify_missing_steps(chain)

        # Determine overall quality
        avg_score = sum([validity, coherence, completeness]) / 3
        if avg_score >= 0.8:
            quality = ReasoningQuality.EXCELLENT
        elif avg_score >= 0.6:
            quality = ReasoningQuality.GOOD
        elif avg_score >= 0.4:
            quality = ReasoningQuality.ADEQUATE
        elif avg_score >= 0.2:
            quality = ReasoningQuality.POOR
        else:
            quality = ReasoningQuality.INVALID

        return ChainAnalysis(
            chain=chain,
            logical_validity=validity,
            coherence_score=coherence,
            completeness_score=completeness,
            step_quality_scores=step_scores,
            identified_fallacies=fallacies,
            missing_steps=missing,
            overall_quality=quality,
        )

    def _check_logical_validity(self, chain: ReasoningChain) -> float:
        """Check logical validity of chain."""
        if not chain.steps:
            return 0.0

        validity = 0.5

        # Has clear flow
        if len(chain.steps) >= 2:
            validity += 0.2

        # Steps build on each other
        inference_count = sum(1 for s in chain.steps if s.step_type == StepType.INFERENCE)
        if inference_count > 0:
            validity += 0.15

        # Has evidence/premises
        premise_count = sum(1 for s in chain.steps if s.step_type == StepType.PREMISE)
        if premise_count > 0:
            validity += 0.15

        return min(1.0, validity)

    def _calculate_coherence(self, chain: ReasoningChain) -> float:
        """Calculate coherence between steps."""
        if len(chain.steps) < 2:
            return 1.0 if chain.steps else 0.0

        coherence = 0.0
        step_pairs = 0

        for i in range(len(chain.steps) - 1):
            current = chain.steps[i].content.lower().split()
            next_step = chain.steps[i + 1].content.lower().split()

            # Calculate word overlap
            overlap = len(set(current) & set(next_step))
            total = len(set(current) | set(next_step))

            if total > 0:
                coherence += overlap / total
                step_pairs += 1

        return coherence / step_pairs if step_pairs > 0 else 0.0

    def _score_step(self, step: ReasoningStep) -> float:
        """Score a single reasoning step."""
        score = 0.5

        # Content length
        words = len(step.content.split())
        if 5 <= words <= 50:
            score += 0.2
        elif words < 5:
            score -= 0.1

        # Has reasoning markers
        content_lower = step.content.lower()
        if any(m in content_lower for m in ["because", "therefore", "since", "thus"]):
            score += 0.2

        # Step confidence
        score += step.confidence * 0.1

        return min(1.0, max(0.0, score))

    def _identify_fallacies(self, chain: ReasoningChain) -> list[str]:
        """Identify logical fallacies in the chain."""
        fallacies = []
        full_text = " ".join(s.content.lower() for s in chain.steps)

        for fallacy_name, patterns in self.FALLACY_PATTERNS.items():
            for pattern in patterns:
                if pattern in full_text:
                    fallacies.append(fallacy_name)
                    break

        return list(set(fallacies))

    def _identify_missing_steps(self, chain: ReasoningChain) -> list[str]:
        """Identify potentially missing steps."""
        missing = []

        step_types = [s.step_type for s in chain.steps]

        # Missing premise
        if StepType.PREMISE not in step_types and len(chain.steps) > 1:
            missing.append("No clear premise or starting point")

        # Missing conclusion
        if StepType.CONCLUSION not in step_types and chain.conclusion is None:
            missing.append("No explicit conclusion")

        # Jumps in reasoning
        if len(chain.steps) >= 2:
            for i in range(len(chain.steps) - 1):
                coherence = self._step_coherence(chain.steps[i], chain.steps[i + 1])
                if coherence < 0.1:
                    missing.append(f"Gap between step {i + 1} and {i + 2}")

        return missing

    def _step_coherence(self, step1: ReasoningStep, step2: ReasoningStep) -> float:
        """Calculate coherence between two steps."""
        words1 = set(step1.content.lower().split())
        words2 = set(step2.content.lower().split())

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        return overlap / union if union > 0 else 0.0


class CoTEvaluator:
    """Evaluates Chain-of-Thought responses."""

    def __init__(self):
        """Initialize evaluator."""
        self.extractor = ReasoningExtractor()
        self.analyzer = ReasoningAnalyzer()

    def evaluate(
        self,
        prompt: str,
        response: str,
        expected_answer: Optional[str] = None,
    ) -> CoTEvaluation:
        """Evaluate a CoT response."""
        # Extract reasoning chain
        chain = self.extractor.extract(response)

        # Analyze chain
        analysis = self.analyzer.analyze(chain)

        # Check answer correctness
        answer_correct = None
        if expected_answer:
            answer_correct = self._check_answer(response, expected_answer)

        # Calculate reasoning score
        reasoning_score = (
            analysis.logical_validity * 0.4
            + analysis.coherence_score * 0.3
            + analysis.completeness_score * 0.3
        )

        # Calculate step accuracy
        step_accuracy = (
            sum(analysis.step_quality_scores) / len(analysis.step_quality_scores)
            if analysis.step_quality_scores
            else 0.0
        )

        # Calculate explanation quality
        explanation_quality = self._assess_explanation_quality(response, chain)

        # Generate improvements
        improvements = self._suggest_improvements(analysis)

        return CoTEvaluation(
            prompt=prompt,
            response=response,
            chain=chain,
            answer_correct=answer_correct,
            reasoning_score=reasoning_score,
            step_accuracy=step_accuracy,
            explanation_quality=explanation_quality,
            improvements=improvements,
        )

    def evaluate_batch(
        self,
        prompts: list[str],
        responses: list[str],
        expected_answers: Optional[list[str]] = None,
    ) -> list[CoTEvaluation]:
        """Evaluate multiple CoT responses."""
        results = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            expected = (
                expected_answers[i] if expected_answers and i < len(expected_answers) else None
            )
            result = self.evaluate(prompt, response, expected)
            results.append(result)
        return results

    def generate_report(
        self,
        evaluations: list[CoTEvaluation],
    ) -> ReasoningReport:
        """Generate report from evaluations."""
        if not evaluations:
            return ReasoningReport(
                total_evaluations=0,
                avg_reasoning_score=0.0,
                avg_step_accuracy=0.0,
                reasoning_type_breakdown={},
                common_fallacies=[],
                quality_distribution={},
                recommendations=[],
            )

        # Calculate averages
        avg_reasoning = sum(e.reasoning_score for e in evaluations) / len(evaluations)
        avg_step_acc = sum(e.step_accuracy for e in evaluations) / len(evaluations)

        # Reasoning type breakdown
        type_counts: dict[str, int] = {}
        for e in evaluations:
            rt = e.chain.reasoning_type.value
            type_counts[rt] = type_counts.get(rt, 0) + 1

        type_breakdown = {k: v / len(evaluations) for k, v in type_counts.items()}

        # Common fallacies
        fallacy_counts: dict[str, int] = {}
        for e in evaluations:
            analysis = self.analyzer.analyze(e.chain)
            for f in analysis.identified_fallacies:
                fallacy_counts[f] = fallacy_counts.get(f, 0) + 1

        common_fallacies = sorted(
            fallacy_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Quality distribution
        quality_dist: dict[str, int] = {}
        for e in evaluations:
            analysis = self.analyzer.analyze(e.chain)
            q = analysis.overall_quality.value
            quality_dist[q] = quality_dist.get(q, 0) + 1

        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_reasoning, avg_step_acc, common_fallacies
        )

        return ReasoningReport(
            total_evaluations=len(evaluations),
            avg_reasoning_score=avg_reasoning,
            avg_step_accuracy=avg_step_acc,
            reasoning_type_breakdown=type_breakdown,
            common_fallacies=common_fallacies,
            quality_distribution=quality_dist,
            recommendations=recommendations,
        )

    def _check_answer(self, response: str, expected: str) -> bool:
        """Check if response contains expected answer."""
        response_lower = response.lower()
        expected_lower = expected.lower()

        return expected_lower in response_lower

    def _assess_explanation_quality(
        self,
        response: str,
        chain: ReasoningChain,
    ) -> float:
        """Assess quality of explanation."""
        quality = 0.0

        # Length appropriateness
        words = len(response.split())
        if 50 <= words <= 500:
            quality += 0.3
        elif 20 <= words <= 50:
            quality += 0.2

        # Has clear structure
        if len(chain.steps) >= 3:
            quality += 0.3

        # Uses reasoning words
        reasoning_words = ["because", "therefore", "since", "thus", "so", "hence"]
        response_lower = response.lower()
        if any(w in response_lower for w in reasoning_words):
            quality += 0.2

        # Has conclusion
        if chain.conclusion:
            quality += 0.2

        return min(1.0, quality)

    def _suggest_improvements(self, analysis: ChainAnalysis) -> list[str]:
        """Suggest improvements based on analysis."""
        improvements = []

        if analysis.logical_validity < 0.5:
            improvements.append("Strengthen logical connections between steps")

        if analysis.coherence_score < 0.5:
            improvements.append("Improve coherence between reasoning steps")

        if analysis.completeness_score < 0.5:
            improvements.append("Add missing steps to complete the reasoning chain")

        if analysis.identified_fallacies:
            improvements.append(
                f"Address logical fallacies: {', '.join(analysis.identified_fallacies)}"
            )

        if analysis.missing_steps:
            improvements.append("Fill in identified gaps in reasoning")

        return improvements

    def _generate_recommendations(
        self,
        avg_reasoning: float,
        avg_step_acc: float,
        fallacies: list[tuple[str, int]],
    ) -> list[str]:
        """Generate recommendations for improvement."""
        recommendations = []

        if avg_reasoning < 0.5:
            recommendations.append("Focus on improving overall reasoning quality")

        if avg_step_acc < 0.5:
            recommendations.append("Work on clarity and validity of individual steps")

        if fallacies:
            top_fallacy = fallacies[0][0]
            recommendations.append(f"Address common fallacy: {top_fallacy}")

        if avg_reasoning < 0.7:
            recommendations.append("Use more explicit reasoning markers (therefore, because, etc.)")

        return recommendations


class CoTPromptGenerator:
    """Generates Chain-of-Thought prompts."""

    TEMPLATES = {
        "standard": "Let's think step by step.\n\n{question}",
        "structured": "Please solve this problem step by step:\n\n{question}\n\nStep 1:",
        "detailed": "I need to solve this problem. Let me break it down:\n\n{question}\n\nFirst, I'll identify what we know:\n",
        "math": "Let's solve this mathematical problem step by step:\n\n{question}\n\nGiven:\n",
        "logical": "Let me reason through this logically:\n\n{question}\n\nPremise 1:",
    }

    def generate(
        self,
        question: str,
        style: str = "standard",
        custom_template: Optional[str] = None,
    ) -> str:
        """Generate a CoT prompt."""
        template = custom_template or self.TEMPLATES.get(style, self.TEMPLATES["standard"])

        return template.format(question=question)

    def generate_variations(
        self,
        question: str,
        num_variations: int = 3,
    ) -> list[str]:
        """Generate multiple CoT prompt variations."""
        variations = []
        styles = list(self.TEMPLATES.keys())

        for i in range(min(num_variations, len(styles))):
            prompt = self.generate(question, style=styles[i])
            variations.append(prompt)

        return variations

    def add_template(self, name: str, template: str) -> None:
        """Add a custom template."""
        self.TEMPLATES[name] = template


# Convenience functions


def extract_reasoning(text: str) -> ReasoningChain:
    """Extract reasoning chain from text.

    Args:
        text: Input text containing reasoning

    Returns:
        ReasoningChain with extracted steps
    """
    extractor = ReasoningExtractor()
    return extractor.extract(text)


def analyze_reasoning(chain: ReasoningChain) -> ChainAnalysis:
    """Analyze a reasoning chain.

    Args:
        chain: Reasoning chain to analyze

    Returns:
        ChainAnalysis with quality metrics
    """
    analyzer = ReasoningAnalyzer()
    return analyzer.analyze(chain)


def evaluate_cot(
    prompt: str,
    response: str,
    expected_answer: Optional[str] = None,
) -> CoTEvaluation:
    """Evaluate a Chain-of-Thought response.

    Args:
        prompt: Original prompt
        response: Model response
        expected_answer: Expected answer (optional)

    Returns:
        CoTEvaluation with metrics
    """
    evaluator = CoTEvaluator()
    return evaluator.evaluate(prompt, response, expected_answer)


def generate_cot_prompt(
    question: str,
    style: str = "standard",
) -> str:
    """Generate a CoT prompt.

    Args:
        question: Question to wrap in CoT prompt
        style: Prompt style (standard, structured, detailed, math, logical)

    Returns:
        CoT-formatted prompt
    """
    generator = CoTPromptGenerator()
    return generator.generate(question, style)


def assess_reasoning_quality(text: str) -> ReasoningQuality:
    """Quickly assess reasoning quality.

    Args:
        text: Text containing reasoning

    Returns:
        ReasoningQuality level
    """
    extractor = ReasoningExtractor()
    analyzer = ReasoningAnalyzer()

    chain = extractor.extract(text)
    analysis = analyzer.analyze(chain)

    return analysis.overall_quality
