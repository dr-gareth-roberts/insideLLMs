"""
Model behavior analysis utilities.

Provides tools for:
- Response consistency analysis
- Output pattern detection
- Behavioral fingerprinting
- Sensitivity analysis
- Calibration assessment
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class BehaviorPattern(Enum):
    """Common behavior patterns in LLM outputs."""

    HEDGING = "hedging"  # Excessive qualification
    VERBOSITY = "verbosity"  # Overly long responses
    REPETITION = "repetition"  # Repeated phrases/concepts
    REFUSAL = "refusal"  # Declining to answer
    HALLUCINATION_RISK = "hallucination_risk"  # Confident but likely wrong
    UNCERTAINTY = "uncertainty"  # Explicit uncertainty
    OVERCONFIDENCE = "overconfidence"  # Too confident without evidence
    SYCOPHANCY = "sycophancy"  # Excessive agreement
    LITERAL = "literal"  # Overly literal interpretation
    CREATIVE = "creative"  # Creative/divergent response


@dataclass
class PatternMatch:
    """A detected behavior pattern match."""

    pattern: BehaviorPattern
    confidence: float  # 0-1
    evidence: list[str] = field(default_factory=list)
    location: Optional[str] = None  # Where in response

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern": self.pattern.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "location": self.location,
        }


@dataclass
class ConsistencyReport:
    """Report on response consistency across multiple outputs."""

    responses: list[str]
    prompt: str
    consistency_score: float  # 0-1, higher = more consistent
    variance_score: float  # 0-1, higher = more variance
    common_elements: list[str] = field(default_factory=list)
    divergent_elements: list[str] = field(default_factory=list)
    semantic_clusters: int = 1
    dominant_response_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "num_responses": len(self.responses),
            "consistency_score": self.consistency_score,
            "variance_score": self.variance_score,
            "common_elements": self.common_elements,
            "divergent_elements": self.divergent_elements,
            "semantic_clusters": self.semantic_clusters,
            "dominant_response_type": self.dominant_response_type,
        }


@dataclass
class BehaviorFingerprint:
    """Behavioral fingerprint of a model based on responses."""

    model_id: str
    sample_size: int
    avg_response_length: float
    avg_sentence_count: float
    hedging_frequency: float
    refusal_rate: float
    verbosity_score: float
    formality_score: float
    confidence_score: float
    common_phrases: list[tuple[str, int]] = field(default_factory=list)
    pattern_frequencies: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "sample_size": self.sample_size,
            "avg_response_length": self.avg_response_length,
            "avg_sentence_count": self.avg_sentence_count,
            "hedging_frequency": self.hedging_frequency,
            "refusal_rate": self.refusal_rate,
            "verbosity_score": self.verbosity_score,
            "formality_score": self.formality_score,
            "confidence_score": self.confidence_score,
            "common_phrases": self.common_phrases,
            "pattern_frequencies": self.pattern_frequencies,
        }

    def compare_to(self, other: "BehaviorFingerprint") -> dict[str, float]:
        """Compare fingerprints and return differences."""
        return {
            "avg_response_length_diff": self.avg_response_length - other.avg_response_length,
            "hedging_frequency_diff": self.hedging_frequency - other.hedging_frequency,
            "refusal_rate_diff": self.refusal_rate - other.refusal_rate,
            "verbosity_score_diff": self.verbosity_score - other.verbosity_score,
            "formality_score_diff": self.formality_score - other.formality_score,
            "confidence_score_diff": self.confidence_score - other.confidence_score,
        }


@dataclass
class PromptSensitivityResult:
    """Result of sensitivity analysis on prompt variations."""

    original_prompt: str
    variations: list[str]
    original_response: str
    variation_responses: list[str]
    sensitivity_score: float  # 0-1, higher = more sensitive
    stable_elements: list[str] = field(default_factory=list)
    sensitive_elements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_prompt": self.original_prompt,
            "num_variations": len(self.variations),
            "sensitivity_score": self.sensitivity_score,
            "stable_elements": self.stable_elements,
            "sensitive_elements": self.sensitive_elements,
        }


class PatternDetector:
    """Detect behavior patterns in LLM responses."""

    # Hedging language
    HEDGING_PHRASES = [
        "i think",
        "i believe",
        "it seems",
        "perhaps",
        "maybe",
        "might",
        "could be",
        "possibly",
        "it's possible",
        "generally",
        "typically",
        "usually",
        "in some cases",
        "it depends",
        "to some extent",
        "arguably",
        "likely",
        "probably",
    ]

    # Refusal indicators
    REFUSAL_PHRASES = [
        "i cannot",
        "i can't",
        "i'm unable",
        "i am unable",
        "i won't",
        "i will not",
        "i'm not able",
        "i am not able",
        "i don't have the ability",
        "it would not be appropriate",
        "i'm not comfortable",
        "i cannot provide",
        "i can't help with",
        "as an ai",
        "as a language model",
    ]

    # Uncertainty indicators
    UNCERTAINTY_PHRASES = [
        "i'm not sure",
        "i don't know",
        "i'm uncertain",
        "i lack information",
        "i cannot be certain",
        "to the best of my knowledge",
        "as far as i know",
        "i may be wrong",
    ]

    # Sycophantic phrases
    SYCOPHANTIC_PHRASES = [
        "great question",
        "excellent question",
        "wonderful question",
        "you're absolutely right",
        "that's a fantastic",
        "i completely agree",
        "what a thoughtful",
    ]

    def detect_patterns(self, response: str) -> list[PatternMatch]:
        """Detect behavior patterns in a response.

        Args:
            response: Model response text.

        Returns:
            List of detected patterns.
        """
        patterns = []
        response_lower = response.lower()

        # Detect hedging
        hedging = self._detect_hedging(response_lower)
        if hedging:
            patterns.append(hedging)

        # Detect refusal
        refusal = self._detect_refusal(response_lower)
        if refusal:
            patterns.append(refusal)

        # Detect uncertainty
        uncertainty = self._detect_uncertainty(response_lower)
        if uncertainty:
            patterns.append(uncertainty)

        # Detect verbosity
        verbosity = self._detect_verbosity(response)
        if verbosity:
            patterns.append(verbosity)

        # Detect repetition
        repetition = self._detect_repetition(response)
        if repetition:
            patterns.append(repetition)

        # Detect sycophancy
        sycophancy = self._detect_sycophancy(response_lower)
        if sycophancy:
            patterns.append(sycophancy)

        # Detect overconfidence
        overconfidence = self._detect_overconfidence(response_lower, response)
        if overconfidence:
            patterns.append(overconfidence)

        return patterns

    def _detect_hedging(self, response_lower: str) -> Optional[PatternMatch]:
        """Detect hedging language."""
        found = [p for p in self.HEDGING_PHRASES if p in response_lower]
        word_count = len(response_lower.split())

        if not found or word_count == 0:
            return None

        # Calculate density of hedging language
        hedging_density = len(found) / (word_count / 50)  # Per 50 words
        confidence = min(1.0, hedging_density * 0.3)

        if confidence >= 0.2:
            return PatternMatch(
                pattern=BehaviorPattern.HEDGING,
                confidence=confidence,
                evidence=found[:5],  # Top 5 examples
            )
        return None

    def _detect_refusal(self, response_lower: str) -> Optional[PatternMatch]:
        """Detect refusal to answer."""
        found = [p for p in self.REFUSAL_PHRASES if p in response_lower]

        if not found:
            return None

        # Stronger refusal indicators get higher confidence
        confidence = min(1.0, len(found) * 0.3)

        return PatternMatch(
            pattern=BehaviorPattern.REFUSAL,
            confidence=confidence,
            evidence=found[:3],
        )

    def _detect_uncertainty(self, response_lower: str) -> Optional[PatternMatch]:
        """Detect explicit uncertainty."""
        found = [p for p in self.UNCERTAINTY_PHRASES if p in response_lower]

        if not found:
            return None

        confidence = min(1.0, len(found) * 0.35)

        return PatternMatch(
            pattern=BehaviorPattern.UNCERTAINTY,
            confidence=confidence,
            evidence=found[:3],
        )

    def _detect_verbosity(self, response: str) -> Optional[PatternMatch]:
        """Detect overly verbose responses."""
        word_count = len(response.split())
        sentence_count = len(re.findall(r"[.!?]+", response))

        if sentence_count == 0:
            return None

        words_per_sentence = word_count / sentence_count
        evidence = []

        # Very long average sentences
        if words_per_sentence > 35:
            evidence.append(f"Average {words_per_sentence:.1f} words per sentence")

        # Many sentences
        if sentence_count > 15:
            evidence.append(f"Contains {sentence_count} sentences")

        # Long overall
        if word_count > 500:
            evidence.append(f"Total {word_count} words")

        if not evidence:
            return None

        confidence = min(1.0, len(evidence) * 0.35)

        return PatternMatch(
            pattern=BehaviorPattern.VERBOSITY,
            confidence=confidence,
            evidence=evidence,
        )

    def _detect_repetition(self, response: str) -> Optional[PatternMatch]:
        """Detect repeated phrases or concepts."""
        words = response.lower().split()
        word_count = len(words)

        if word_count < 20:
            return None

        # Find repeated n-grams (3-5 words)
        ngrams: dict[str, int] = {}
        for n in range(3, 6):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i : i + n])
                ngrams[ngram] = ngrams.get(ngram, 0) + 1

        # Find ngrams that appear more than once
        repeated = [(ng, count) for ng, count in ngrams.items() if count > 1]
        repeated.sort(key=lambda x: x[1], reverse=True)

        if not repeated:
            return None

        total_repetitions = sum(count - 1 for _, count in repeated)
        repetition_ratio = total_repetitions / (word_count / 20)

        confidence = min(1.0, repetition_ratio * 0.2)

        if confidence >= 0.2:
            return PatternMatch(
                pattern=BehaviorPattern.REPETITION,
                confidence=confidence,
                evidence=[f"'{ng}' repeated {c}x" for ng, c in repeated[:3]],
            )
        return None

    def _detect_sycophancy(self, response_lower: str) -> Optional[PatternMatch]:
        """Detect sycophantic responses."""
        found = [p for p in self.SYCOPHANTIC_PHRASES if p in response_lower]

        if not found:
            return None

        confidence = min(1.0, len(found) * 0.4)

        return PatternMatch(
            pattern=BehaviorPattern.SYCOPHANCY,
            confidence=confidence,
            evidence=found[:3],
        )

    def _detect_overconfidence(self, response_lower: str, response: str) -> Optional[PatternMatch]:
        """Detect potentially overconfident responses."""
        evidence = []

        # Strong assertions without hedging
        strong_assertions = [
            "definitely",
            "certainly",
            "absolutely",
            "without a doubt",
            "undoubtedly",
            "clearly",
            "obviously",
            "of course",
            "always",
            "never",
            "100%",
            "guaranteed",
        ]

        found_assertions = [a for a in strong_assertions if a in response_lower]

        # Check for lack of hedging alongside strong assertions
        has_hedging = any(h in response_lower for h in self.HEDGING_PHRASES)

        if found_assertions and not has_hedging:
            evidence.extend(found_assertions[:3])

        # Check for lack of citations/sources with factual claims
        has_numbers = bool(re.search(r"\d+%|\d+\.\d+", response))
        has_citations = any(
            c in response_lower for c in ["according to", "research shows", "study", "source"]
        )

        if has_numbers and not has_citations:
            evidence.append("Contains statistics without attribution")

        if not evidence:
            return None

        confidence = min(1.0, len(evidence) * 0.3)

        return PatternMatch(
            pattern=BehaviorPattern.OVERCONFIDENCE,
            confidence=confidence,
            evidence=evidence,
        )


class ConsistencyAnalyzer:
    """Analyze consistency across multiple responses."""

    def analyze(self, prompt: str, responses: list[str]) -> ConsistencyReport:
        """Analyze consistency across responses.

        Args:
            prompt: Original prompt.
            responses: Multiple responses to the same prompt.

        Returns:
            Consistency report.
        """
        if not responses:
            return ConsistencyReport(
                responses=[],
                prompt=prompt,
                consistency_score=0.0,
                variance_score=1.0,
            )

        if len(responses) == 1:
            return ConsistencyReport(
                responses=responses,
                prompt=prompt,
                consistency_score=1.0,
                variance_score=0.0,
            )

        # Extract key elements from each response
        all_elements: list[set[str]] = []
        for response in responses:
            elements = self._extract_key_elements(response)
            all_elements.append(elements)

        # Find common elements (appear in majority)
        threshold = len(responses) // 2 + 1
        element_counts: dict[str, int] = {}
        for elements in all_elements:
            for elem in elements:
                element_counts[elem] = element_counts.get(elem, 0) + 1

        common_elements = [e for e, c in element_counts.items() if c >= threshold]
        divergent_elements = [e for e, c in element_counts.items() if c < threshold and c > 0]

        # Calculate Jaccard similarity between all pairs
        similarities = []
        for i in range(len(all_elements)):
            for j in range(i + 1, len(all_elements)):
                sim = self._jaccard_similarity(all_elements[i], all_elements[j])
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        # Calculate length variance
        lengths = [len(r.split()) for r in responses]
        avg_length = sum(lengths) / len(lengths)
        length_variance = (
            sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
            if len(lengths) > 1
            else 0
        )

        # Determine dominant response type
        dominant_type = self._classify_response_type(responses[0])

        return ConsistencyReport(
            responses=responses,
            prompt=prompt,
            consistency_score=avg_similarity,
            variance_score=1 - avg_similarity,
            common_elements=common_elements[:10],
            divergent_elements=divergent_elements[:10],
            semantic_clusters=self._estimate_clusters(similarities, len(responses)),
            dominant_response_type=dominant_type,
        )

    def _extract_key_elements(self, response: str) -> set[str]:
        """Extract key elements from a response."""
        elements = set()

        # Extract sentences
        sentences = re.split(r"[.!?]+", response)
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if len(sentence) > 10:
                # Add key phrases (first and last meaningful words)
                words = [w for w in sentence.split() if len(w) > 3]
                if words:
                    elements.add(words[0])
                    if len(words) > 1:
                        elements.add(words[-1])

        # Extract numbers
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", response)
        elements.update(numbers)

        # Extract capitalized terms (proper nouns)
        proper_nouns = re.findall(r"\b[A-Z][a-z]+\b", response)
        elements.update(w.lower() for w in proper_nouns if len(w) > 2)

        return elements

    def _jaccard_similarity(self, set1: set[str], set2: set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0

    def _estimate_clusters(self, similarities: list[float], num_responses: int) -> int:
        """Estimate number of semantic clusters."""
        if num_responses <= 2:
            return 1

        avg_sim = sum(similarities) / len(similarities) if similarities else 0

        # Simple heuristic: low similarity suggests multiple clusters
        if avg_sim > 0.7:
            return 1
        elif avg_sim > 0.4:
            return 2
        elif avg_sim > 0.2:
            return 3
        else:
            return min(4, num_responses)

    def _classify_response_type(self, response: str) -> str:
        """Classify the response type."""
        response_lower = response.lower()

        if any(r in response_lower for r in ["i cannot", "i can't", "unable to", "i won't"]):
            return "refusal"

        if any(r in response_lower for r in ["i don't know", "not sure", "uncertain"]):
            return "uncertain"

        if "```" in response:
            return "code"

        if re.search(r"^\s*[-*\d+.]\s", response, re.MULTILINE):
            return "list"

        if len(response.split()) > 200:
            return "verbose"

        return "standard"


class BehaviorProfiler:
    """Build behavioral profiles from response samples."""

    def __init__(self):
        """Initialize profiler."""
        self.pattern_detector = PatternDetector()

    def create_fingerprint(self, model_id: str, responses: list[str]) -> BehaviorFingerprint:
        """Create a behavioral fingerprint from responses.

        Args:
            model_id: Identifier for the model.
            responses: List of model responses.

        Returns:
            Behavioral fingerprint.
        """
        if not responses:
            return BehaviorFingerprint(
                model_id=model_id,
                sample_size=0,
                avg_response_length=0,
                avg_sentence_count=0,
                hedging_frequency=0,
                refusal_rate=0,
                verbosity_score=0,
                formality_score=0,
                confidence_score=0.5,
            )

        # Calculate basic statistics
        lengths = [len(r.split()) for r in responses]
        sentence_counts = [len(re.findall(r"[.!?]+", r)) for r in responses]

        avg_length = sum(lengths) / len(lengths)
        avg_sentences = sum(sentence_counts) / len(sentence_counts)

        # Detect patterns across all responses
        pattern_counts: dict[str, int] = {}
        total_patterns = 0

        for response in responses:
            patterns = self.pattern_detector.detect_patterns(response)
            for pattern in patterns:
                pattern_counts[pattern.pattern.value] = (
                    pattern_counts.get(pattern.pattern.value, 0) + 1
                )
                total_patterns += 1

        # Calculate frequencies
        sample_size = len(responses)
        hedging_freq = pattern_counts.get("hedging", 0) / sample_size
        refusal_rate = pattern_counts.get("refusal", 0) / sample_size
        verbosity_score = pattern_counts.get("verbosity", 0) / sample_size

        # Calculate confidence score (inverse of uncertainty + hedging)
        uncertainty_freq = pattern_counts.get("uncertainty", 0) / sample_size
        overconfidence_freq = pattern_counts.get("overconfidence", 0) / sample_size
        confidence_score = (
            0.5 + (overconfidence_freq * 0.3) - ((hedging_freq + uncertainty_freq) * 0.2)
        )
        confidence_score = max(0.0, min(1.0, confidence_score))

        # Calculate formality score
        formality_score = self._calculate_formality(responses)

        # Find common phrases
        common_phrases = self._find_common_phrases(responses)

        # Pattern frequencies
        pattern_freqs = {p: count / sample_size for p, count in pattern_counts.items()}

        return BehaviorFingerprint(
            model_id=model_id,
            sample_size=sample_size,
            avg_response_length=avg_length,
            avg_sentence_count=avg_sentences,
            hedging_frequency=hedging_freq,
            refusal_rate=refusal_rate,
            verbosity_score=verbosity_score,
            formality_score=formality_score,
            confidence_score=confidence_score,
            common_phrases=common_phrases,
            pattern_frequencies=pattern_freqs,
        )

    def _calculate_formality(self, responses: list[str]) -> float:
        """Calculate average formality of responses."""
        informal_markers = [
            "gonna",
            "wanna",
            "gotta",
            "kinda",
            "sorta",
            "yeah",
            "yep",
            "nope",
            "ok",
            "okay",
            "cool",
            "awesome",
            "!",
            "...",
        ]

        formal_markers = [
            "therefore",
            "however",
            "furthermore",
            "consequently",
            "additionally",
            "nevertheless",
            "whereas",
            "hereby",
            "thus",
        ]

        informal_count = 0
        formal_count = 0

        for response in responses:
            response_lower = response.lower()
            informal_count += sum(1 for m in informal_markers if m in response_lower)
            formal_count += sum(1 for m in formal_markers if m in response_lower)

        total = informal_count + formal_count
        if total == 0:
            return 0.5  # Neutral

        return formal_count / total

    def _find_common_phrases(
        self, responses: list[str], min_count: int = 2, top_n: int = 10
    ) -> list[tuple[str, int]]:
        """Find commonly repeated phrases across responses."""
        phrase_counts: dict[str, int] = {}

        for response in responses:
            words = response.lower().split()
            # Extract 2-4 word phrases
            for n in range(2, 5):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i : i + n])
                    # Filter out very common phrases
                    if not any(stop in phrase for stop in ["the ", " a ", " an ", " is ", " are "]):
                        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        # Filter and sort
        common = [(phrase, count) for phrase, count in phrase_counts.items() if count >= min_count]
        common.sort(key=lambda x: x[1], reverse=True)

        return common[:top_n]


class PromptSensitivityAnalyzer:
    """Analyze model sensitivity to prompt variations."""

    def __init__(self):
        """Initialize analyzer."""
        self.consistency_analyzer = ConsistencyAnalyzer()

    def analyze_sensitivity(
        self,
        original_prompt: str,
        original_response: str,
        variations: list[str],
        variation_responses: list[str],
    ) -> PromptSensitivityResult:
        """Analyze sensitivity to prompt variations.

        Args:
            original_prompt: Original prompt.
            original_response: Response to original prompt.
            variations: List of prompt variations.
            variation_responses: Responses to variations.

        Returns:
            Sensitivity analysis result.
        """
        if len(variations) != len(variation_responses):
            raise ValueError("Number of variations must match number of variation responses")

        all_responses = [original_response] + variation_responses

        # Use consistency analyzer
        consistency = self.consistency_analyzer.analyze(original_prompt, all_responses)

        # Sensitivity is inverse of consistency
        sensitivity_score = consistency.variance_score

        # Find elements that are stable vs sensitive
        original_elements = self._extract_answer_elements(original_response)

        stable_elements = []
        sensitive_elements = []

        for elem in original_elements:
            in_variations = sum(1 for r in variation_responses if elem.lower() in r.lower())
            ratio = in_variations / len(variation_responses) if variation_responses else 0

            if ratio >= 0.7:
                stable_elements.append(elem)
            elif ratio <= 0.3:
                sensitive_elements.append(elem)

        return PromptSensitivityResult(
            original_prompt=original_prompt,
            variations=variations,
            original_response=original_response,
            variation_responses=variation_responses,
            sensitivity_score=sensitivity_score,
            stable_elements=stable_elements[:10],
            sensitive_elements=sensitive_elements[:10],
        )

    def _extract_answer_elements(self, response: str) -> list[str]:
        """Extract key answer elements from response."""
        elements = []

        # Extract sentences
        sentences = re.split(r"[.!?]+", response)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:
                elements.append(sentence[:50])

        # Extract numbers
        numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", response)
        elements.extend(numbers)

        # Extract capitalized terms
        proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", response)
        elements.extend(proper_nouns)

        return elements


@dataclass
class BehaviorCalibrationResult:
    """Result of calibration assessment."""

    confidence_levels: list[float]
    accuracy_at_levels: list[float]
    expected_calibration_error: float
    overconfidence_score: float
    underconfidence_score: float

    def is_well_calibrated(self, threshold: float = 0.1) -> bool:
        """Check if model is well calibrated."""
        return self.expected_calibration_error < threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "confidence_levels": self.confidence_levels,
            "accuracy_at_levels": self.accuracy_at_levels,
            "expected_calibration_error": self.expected_calibration_error,
            "overconfidence_score": self.overconfidence_score,
            "underconfidence_score": self.underconfidence_score,
            "is_well_calibrated": self.is_well_calibrated(),
        }


class CalibrationAssessor:
    """Assess model calibration (confidence vs accuracy)."""

    def assess(
        self,
        predictions: list[str],
        ground_truths: list[str],
        confidences: list[float],
        num_bins: int = 10,
    ) -> BehaviorCalibrationResult:
        """Assess calibration from predictions and confidences.

        Args:
            predictions: Model predictions.
            ground_truths: Correct answers.
            confidences: Confidence scores (0-1) for each prediction.
            num_bins: Number of bins for calibration curve.

        Returns:
            Calibration assessment result.
        """
        if len(predictions) != len(ground_truths) != len(confidences):
            raise ValueError("All lists must have the same length")

        # Determine correctness
        correct = [self._is_correct(pred, truth) for pred, truth in zip(predictions, ground_truths)]

        # Bin by confidence and calculate ECE in one pass
        bin_edges = [i / num_bins for i in range(num_bins + 1)]
        confidence_levels = []
        accuracy_at_levels = []
        bin_weights = []

        total_samples = len(predictions)

        for i in range(num_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            bin_mask = [low <= c < high for c in confidences]

            if any(bin_mask):
                bin_confidences = [c for c, m in zip(confidences, bin_mask) if m]
                bin_correct = [c for c, m in zip(correct, bin_mask) if m]
                bin_size = len(bin_confidences)

                avg_conf = sum(bin_confidences) / bin_size
                accuracy = sum(bin_correct) / bin_size
                weight = bin_size / total_samples

                confidence_levels.append(avg_conf)
                accuracy_at_levels.append(accuracy)
                bin_weights.append(weight)

        # Calculate Expected Calibration Error from collected bins
        ece = 0.0
        overconfidence = 0.0
        underconfidence = 0.0

        for conf, acc, weight in zip(confidence_levels, accuracy_at_levels, bin_weights):
            diff = conf - acc
            ece += weight * abs(diff)

            if diff > 0:
                overconfidence += weight * diff
            else:
                underconfidence += weight * abs(diff)

        return BehaviorCalibrationResult(
            confidence_levels=confidence_levels,
            accuracy_at_levels=accuracy_at_levels,
            expected_calibration_error=ece,
            overconfidence_score=overconfidence,
            underconfidence_score=underconfidence,
        )

    def _is_correct(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction is correct."""
        pred_normalized = prediction.lower().strip()
        truth_normalized = ground_truth.lower().strip()

        # Exact match
        if pred_normalized == truth_normalized:
            return True

        # Contains match
        return truth_normalized in pred_normalized


# Convenience functions


def detect_patterns(response: str) -> list[PatternMatch]:
    """Detect behavior patterns in a response.

    Args:
        response: Model response text.

    Returns:
        List of detected patterns.
    """
    detector = PatternDetector()
    return detector.detect_patterns(response)


def analyze_consistency(prompt: str, responses: list[str]) -> ConsistencyReport:
    """Analyze consistency across multiple responses.

    Args:
        prompt: Original prompt.
        responses: List of responses.

    Returns:
        Consistency report.
    """
    analyzer = ConsistencyAnalyzer()
    return analyzer.analyze(prompt, responses)


def create_behavior_fingerprint(model_id: str, responses: list[str]) -> BehaviorFingerprint:
    """Create behavioral fingerprint from responses.

    Args:
        model_id: Model identifier.
        responses: List of model responses.

    Returns:
        Behavioral fingerprint.
    """
    profiler = BehaviorProfiler()
    return profiler.create_fingerprint(model_id, responses)


def analyze_sensitivity(
    original_prompt: str,
    original_response: str,
    variations: list[str],
    variation_responses: list[str],
) -> PromptSensitivityResult:
    """Analyze sensitivity to prompt variations.

    Args:
        original_prompt: Original prompt.
        original_response: Response to original.
        variations: Prompt variations.
        variation_responses: Responses to variations.

    Returns:
        Sensitivity analysis result.
    """
    analyzer = PromptSensitivityAnalyzer()
    return analyzer.analyze_sensitivity(
        original_prompt, original_response, variations, variation_responses
    )


def assess_calibration(
    predictions: list[str],
    ground_truths: list[str],
    confidences: list[float],
) -> BehaviorCalibrationResult:
    """Assess model calibration.

    Args:
        predictions: Model predictions.
        ground_truths: Correct answers.
        confidences: Confidence scores.

    Returns:
        Calibration result.
    """
    assessor = CalibrationAssessor()
    return assessor.assess(predictions, ground_truths, confidences)
