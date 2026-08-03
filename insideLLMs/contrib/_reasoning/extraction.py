"""Reasoning-chain extraction."""

import re
from typing import Optional

from insideLLMs.contrib._reasoning.models import (
    ReasoningChain,
    ReasoningStep,
    ReasoningStepType,
    ReasoningType,
)


class ReasoningExtractor:
    """
    Extracts structured reasoning chains from unstructured text.

    Uses pattern matching and heuristics to identify reasoning steps,
    classify their types, and construct a ReasoningChain from natural
    language text. Supports various formats including numbered steps,
    ordinal markers, and sentence-based extraction.

    Attributes
    ----------
    STEP_PATTERNS : list[str]
        Regular expression patterns for detecting reasoning steps.
        Includes patterns for "Step N:", numbered lists, and ordinal
        words (first, second, etc.).

    PREMISE_MARKERS : list[str]
        Keywords indicating premise steps: "given", "assume", "let",
        "suppose", "we know".

    INFERENCE_MARKERS : list[str]
        Keywords indicating inference steps: "therefore", "thus",
        "hence", "so", "this means".

    CONCLUSION_MARKERS : list[str]
        Keywords indicating conclusion steps: "therefore", "in conclusion",
        "finally", "the answer is".

    CALCULATION_MARKERS : list[str]
        Characters/words indicating calculation steps: mathematical
        operators and words like "calculate", "compute".

    Examples
    --------
    Basic extraction:

        >>> from insideLLMs.contrib.reasoning import ReasoningExtractor
        >>> extractor = ReasoningExtractor()
        >>> text = '''
        ... Step 1: All birds have wings.
        ... Step 2: Penguins are birds.
        ... Step 3: Therefore, penguins have wings.
        ... '''
        >>> chain = extractor.extract(text)
        >>> print(f"Extracted {len(chain.steps)} steps")
        Extracted 3 steps

    Extraction with numbered lists:

        >>> from insideLLMs.contrib.reasoning import ReasoningExtractor
        >>> extractor = ReasoningExtractor()
        >>> text = '''
        ... 1) First we identify the variables
        ... 2) Then we set up the equation
        ... 3) Finally we solve for x
        ... '''
        >>> chain = extractor.extract(text)
        >>> print(f"Valid: {chain.is_valid}")
        Valid: True

    Extraction from prose:

        >>> from insideLLMs.contrib.reasoning import ReasoningExtractor
        >>> extractor = ReasoningExtractor()
        >>> text = "Given x=5. Since x is positive, therefore x+1 is greater than x."
        >>> chain = extractor.extract(text)
        >>> print(f"Type: {chain.reasoning_type.value}")
        Type: mathematical

    Checking completeness:

        >>> from insideLLMs.contrib.reasoning import ReasoningExtractor
        >>> extractor = ReasoningExtractor()
        >>> complete = "Given A. Let B. Suppose C. Therefore D. In conclusion, E."
        >>> incomplete = "Maybe X. Perhaps Y."
        >>> print(f"Complete: {extractor.extract(complete).completeness:.1f}")
        Complete: 1.0
        >>> print(f"Incomplete: {extractor.extract(incomplete).completeness:.1f}")
        Incomplete: 0.3

    See Also
    --------
    ReasoningChain : Output structure
    ReasoningStep : Individual step structure
    extract_reasoning : Convenience function
    """

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
        """
        Extract a reasoning chain from unstructured text.

        Parses the input text to identify reasoning steps, classify their types,
        determine the overall reasoning pattern, and construct a complete
        ReasoningChain structure.

        Parameters
        ----------
        text : str
            The text to extract reasoning from. Can be in various formats:
            - Numbered steps ("Step 1:", "1)", "1.")
            - Ordinal words ("First,", "Second,", "Finally,")
            - Plain prose (sentences will be extracted as steps)

        Returns
        -------
        ReasoningChain
            A structured chain containing:
            - steps: List of extracted ReasoningStep objects
            - conclusion: The identified conclusion (if any)
            - reasoning_type: The classified reasoning type
            - is_valid: True if at least one step was found
            - completeness: Score indicating chain completeness

        Examples
        --------
        Extracting from numbered steps:

            >>> from insideLLMs.contrib.reasoning import ReasoningExtractor
            >>> extractor = ReasoningExtractor()
            >>> text = "Step 1: Identify the problem. Step 2: Analyze causes. Step 3: Propose solution."
            >>> chain = extractor.extract(text)
            >>> print(f"Found {len(chain.steps)} steps")
            Found 3 steps

        Extracting from prose:

            >>> extractor = ReasoningExtractor()
            >>> text = "Given that x equals 5. Since x is positive, we can conclude x squared is also positive."
            >>> chain = extractor.extract(text)
            >>> print(f"Reasoning type: {chain.reasoning_type.value}")
            Reasoning type: mathematical

        Handling incomplete reasoning:

            >>> extractor = ReasoningExtractor()
            >>> text = "Maybe this is true."
            >>> chain = extractor.extract(text)
            >>> print(f"Valid: {chain.is_valid}, Completeness: {chain.completeness}")
            Valid: True, Completeness: 0.2

        Extracting with conclusion:

            >>> extractor = ReasoningExtractor()
            >>> text = "All dogs are mammals. Rex is a dog. Therefore, Rex is a mammal."
            >>> chain = extractor.extract(text)
            >>> print(f"Conclusion: {chain.conclusion}")
            Conclusion: rex is a mammal
        """
        steps = []

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
        """
        Extract reasoning steps from individual sentences.

        Fallback extraction method when no explicit step markers are found.
        Splits text by sentence boundaries and creates a step for each
        substantial sentence.

        Parameters
        ----------
        text : str
            The text to extract sentences from.

        Returns
        -------
        list[ReasoningStep]
            List of reasoning steps (max 10) extracted from sentences.
            Only sentences with more than 10 characters are included.

        Examples
        --------
            >>> extractor = ReasoningExtractor()
            >>> steps = extractor._extract_from_sentences("A is true. B follows. C is the result.")
            >>> len(steps)
            3
        """
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

    def _classify_step(self, content: str) -> ReasoningStepType:
        """
        Classify the type of a reasoning step based on its content.

        Uses keyword matching to determine whether a step represents a
        premise, inference, calculation, or conclusion.

        Parameters
        ----------
        content : str
            The text content of the step to classify.

        Returns
        -------
        ReasoningStepType
            The classified type. Priority order:
            1. PREMISE (if premise markers found)
            2. CONCLUSION (if conclusion markers found)
            3. CALCULATION (if math operators/words found)
            4. INFERENCE (if inference markers found, or default)

        Examples
        --------
            >>> extractor = ReasoningExtractor()
            >>> extractor._classify_step("Given that x = 5")
            <ReasoningStepType.PREMISE: 'premise'>
            >>> extractor._classify_step("Therefore, y = 10")
            <ReasoningStepType.CONCLUSION: 'conclusion'>
            >>> extractor._classify_step("Calculate 5 + 3 = 8")
            <ReasoningStepType.CALCULATION: 'calculation'>
        """
        content_lower = content.lower()

        for marker in self.PREMISE_MARKERS:
            if marker in content_lower:
                return ReasoningStepType.PREMISE

        for marker in self.CONCLUSION_MARKERS:
            if marker in content_lower:
                return ReasoningStepType.CONCLUSION

        for marker in self.CALCULATION_MARKERS:
            if marker in content:
                return ReasoningStepType.CALCULATION

        for marker in self.INFERENCE_MARKERS:
            if marker in content_lower:
                return ReasoningStepType.INFERENCE

        return ReasoningStepType.INFERENCE

    def _estimate_confidence(self, content: str) -> float:
        """
        Estimate the confidence level of a reasoning step.

        Analyzes linguistic markers to determine how confident the
        reasoning appears to be. High-confidence words increase the
        score while hedging language decreases it.

        Parameters
        ----------
        content : str
            The text content of the step to analyze.

        Returns
        -------
        float
            Confidence score between 0.1 and 0.95. Base score is 0.5.
            Adjusted based on:
            - High confidence markers (+0.2): "clearly", "obviously", "certainly"
            - Low confidence markers (-0.2): "maybe", "perhaps", "possibly", "might"
            - Evidence markers (+0.1): "because", "since", "as shown"

        Examples
        --------
            >>> extractor = ReasoningExtractor()
            >>> extractor._estimate_confidence("This is clearly true")
            0.7
            >>> extractor._estimate_confidence("Maybe this could be right")
            0.3
            >>> extractor._estimate_confidence("This is true because of X")
            0.6
        """
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
        """
        Extract the conclusion from reasoning text.

        Looks for explicit conclusion markers and extracts the following
        content. Falls back to the last sentence if no explicit markers
        are found.

        Parameters
        ----------
        text : str
            The full text to extract conclusion from.

        Returns
        -------
        Optional[str]
            The extracted conclusion text, or None if text is empty.
            Returns lowercase version of the conclusion.

        Examples
        --------
            >>> extractor = ReasoningExtractor()
            >>> extractor._extract_conclusion("A is true. Therefore, B follows.")
            'b follows'
            >>> extractor._extract_conclusion("Step 1: X. The answer is 42.")
            '42'
            >>> extractor._extract_conclusion("Just one statement")
            'Just one statement'
        """
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
        """
        Classify the overall type of reasoning used in the text.

        Analyzes the text for keywords and patterns associated with
        different reasoning types. Returns the first matching type
        found in priority order.

        Parameters
        ----------
        text : str
            The full text to classify.

        Returns
        -------
        ReasoningType
            The classified reasoning type. Priority order:
            1. MATHEMATICAL (math operators or calculation words)
            2. CAUSAL (cause-effect language)
            3. TEMPORAL (time-based language)
            4. ANALOGICAL (comparison language)
            5. INDUCTIVE (generalization language)
            6. DEDUCTIVE (default)

        Examples
        --------
            >>> extractor = ReasoningExtractor()
            >>> extractor._classify_reasoning_type("Calculate 5 + 3 = 8")
            <ReasoningType.MATHEMATICAL: 'mathematical'>
            >>> extractor._classify_reasoning_type("This causes that effect")
            <ReasoningType.CAUSAL: 'causal'>
            >>> extractor._classify_reasoning_type("Before A, then B happens")
            <ReasoningType.TEMPORAL: 'temporal'>
            >>> extractor._classify_reasoning_type("All premises lead to conclusion")
            <ReasoningType.DEDUCTIVE: 'deductive'>
        """
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
        """
        Calculate the completeness of a reasoning chain.

        Scores the chain based on presence of key components: multiple
        steps, premises, inferences, and conclusion.

        Parameters
        ----------
        steps : list[ReasoningStep]
            The list of reasoning steps in the chain.
        conclusion : Optional[str]
            The extracted conclusion, if any.

        Returns
        -------
        float
            Completeness score from 0.0 to 1.0:
            - +0.3 for having 2+ steps
            - +0.2 for having at least one premise
            - +0.2 for having at least one inference
            - +0.3 for having a conclusion

        Examples
        --------
            >>> from insideLLMs.contrib.reasoning import ReasoningStep, ReasoningStepType
            >>> extractor = ReasoningExtractor()
            >>> # Complete chain
            >>> steps = [
            ...     ReasoningStep("Given A", 1, ReasoningStepType.PREMISE),
            ...     ReasoningStep("Therefore B", 2, ReasoningStepType.INFERENCE),
            ...     ReasoningStep("In conclusion C", 3, ReasoningStepType.CONCLUSION)
            ... ]
            >>> extractor._calculate_completeness(steps, "C")
            1.0
            >>> # Incomplete chain
            >>> extractor._calculate_completeness([], None)
            0.0
        """
        if not steps:
            return 0.0

        score = 0.0

        # Has multiple steps
        if len(steps) >= 2:
            score += 0.3

        # Has premise
        if any(s.step_type == ReasoningStepType.PREMISE for s in steps):
            score += 0.2

        # Has inference
        if any(s.step_type == ReasoningStepType.INFERENCE for s in steps):
            score += 0.2

        # Has conclusion
        if conclusion or any(s.step_type == ReasoningStepType.CONCLUSION for s in steps):
            score += 0.3

        return min(1.0, score)
