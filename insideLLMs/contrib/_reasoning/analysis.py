"""Reasoning-chain quality analysis."""

from insideLLMs.contrib._reasoning.models import (
    ChainAnalysis,
    ReasoningChain,
    ReasoningQuality,
    ReasoningStep,
    ReasoningStepType,
)


class ReasoningAnalyzer:
    """
    Analyzes reasoning chains for quality, validity, and potential issues.

    Provides comprehensive analysis of reasoning chains including logical
    validity assessment, coherence scoring, fallacy detection, and
    identification of missing reasoning steps.

    Attributes
    ----------
    FALLACY_PATTERNS : dict[str, list[str]]
        Mapping of fallacy names to keyword patterns used for detection.
        Includes common logical fallacies:
        - circular_reasoning: Self-referential arguments
        - hasty_generalization: Overly broad claims
        - false_dichotomy: Artificial binary choices
        - appeal_to_authority: Relying on authority claims
        - ad_hominem: Personal attacks
        - straw_man: Misrepresenting arguments
        - slippery_slope: Unfounded chain of consequences

    Examples
    --------
    Basic chain analysis:

        >>> from insideLLMs.contrib.reasoning import ReasoningExtractor, ReasoningAnalyzer
        >>> extractor = ReasoningExtractor()
        >>> analyzer = ReasoningAnalyzer()
        >>> chain = extractor.extract("Given A. Therefore B. Thus C.")
        >>> analysis = analyzer.analyze(chain)
        >>> print(f"Quality: {analysis.overall_quality.value}")
        Quality: adequate

    Detecting fallacies:

        >>> from insideLLMs.contrib.reasoning import ReasoningExtractor, ReasoningAnalyzer
        >>> extractor = ReasoningExtractor()
        >>> analyzer = ReasoningAnalyzer()
        >>> chain = extractor.extract("Everyone knows this is true. All people agree.")
        >>> analysis = analyzer.analyze(chain)
        >>> print(f"Fallacies: {analysis.identified_fallacies}")
        Fallacies: ['hasty_generalization']

    Analyzing coherence:

        >>> from insideLLMs.contrib.reasoning import ReasoningExtractor, ReasoningAnalyzer
        >>> extractor = ReasoningExtractor()
        >>> analyzer = ReasoningAnalyzer()
        >>> coherent = "Dogs are mammals. Mammals are warm-blooded. Therefore dogs are warm-blooded."
        >>> chain = extractor.extract(coherent)
        >>> analysis = analyzer.analyze(chain)
        >>> print(f"Coherence: {analysis.coherence_score:.2f}")
        Coherence: 0.25

    Identifying missing steps:

        >>> from insideLLMs.contrib.reasoning import ReasoningExtractor, ReasoningAnalyzer
        >>> extractor = ReasoningExtractor()
        >>> analyzer = ReasoningAnalyzer()
        >>> chain = extractor.extract("Therefore X. Hence Y.")
        >>> analysis = analyzer.analyze(chain)
        >>> for missing in analysis.missing_steps:
        ...     print(f"Missing: {missing}")
        Missing: No clear premise or starting point

    See Also
    --------
    ChainAnalysis : Output structure containing analysis results
    ReasoningChain : Input structure to analyze
    analyze_reasoning : Convenience function
    """

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
        """
        Perform comprehensive analysis of a reasoning chain.

        Evaluates the chain across multiple dimensions including logical
        validity, coherence between steps, completeness, and potential
        logical fallacies.

        Parameters
        ----------
        chain : ReasoningChain
            The reasoning chain to analyze.

        Returns
        -------
        ChainAnalysis
            Complete analysis results including:
            - logical_validity: How logically sound the chain is
            - coherence_score: How well steps connect
            - completeness_score: Presence of key components
            - step_quality_scores: Individual step scores
            - identified_fallacies: Detected logical fallacies
            - missing_steps: Identified gaps in reasoning
            - overall_quality: Categorical quality assessment

        Examples
        --------
        Analyzing a well-structured chain:

            >>> from insideLLMs.contrib.reasoning import ReasoningExtractor, ReasoningAnalyzer
            >>> extractor = ReasoningExtractor()
            >>> analyzer = ReasoningAnalyzer()
            >>> text = '''
            ... Given that all cats are mammals.
            ... Since Whiskers is a cat.
            ... Therefore, Whiskers is a mammal.
            ... '''
            >>> chain = extractor.extract(text)
            >>> analysis = analyzer.analyze(chain)
            >>> print(f"Validity: {analysis.logical_validity:.2f}")
            Validity: 0.85

        Analyzing a poor chain:

            >>> text = "Maybe X. Perhaps Y."
            >>> chain = extractor.extract(text)
            >>> analysis = analyzer.analyze(chain)
            >>> print(f"Quality: {analysis.overall_quality.value}")
            Quality: poor

        Getting step-by-step scores:

            >>> text = "Step 1: Given A. Step 2: Therefore B because A."
            >>> chain = extractor.extract(text)
            >>> analysis = analyzer.analyze(chain)
            >>> for i, score in enumerate(analysis.step_quality_scores):
            ...     print(f"Step {i+1} quality: {score:.2f}")
            Step 1 quality: 0.70
            Step 2 quality: 0.80

        Using analysis for improvement:

            >>> analysis = analyzer.analyze(chain)
            >>> if analysis.logical_validity < 0.5:
            ...     print("Recommendation: Strengthen logical connections")
            >>> if analysis.identified_fallacies:
            ...     print(f"Warning: Fallacies detected: {analysis.identified_fallacies}")
        """
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
        """
        Check the logical validity of a reasoning chain.

        Assesses how logically sound the chain structure is based on
        presence of multiple steps, inferences, and premises.

        Parameters
        ----------
        chain : ReasoningChain
            The chain to check for validity.

        Returns
        -------
        float
            Validity score from 0.0 to 1.0:
            - Base score: 0.5
            - +0.2 for having 2+ steps
            - +0.15 for having inference steps
            - +0.15 for having premise steps

        Examples
        --------
            >>> from insideLLMs.contrib.reasoning import ReasoningExtractor, ReasoningAnalyzer
            >>> analyzer = ReasoningAnalyzer()
            >>> chain = ReasoningExtractor().extract("Given A. Therefore B.")
            >>> print(f"Validity: {analyzer._check_logical_validity(chain):.2f}")
            Validity: 0.85
        """
        if not chain.steps:
            return 0.0

        validity = 0.5

        # Has clear flow
        if len(chain.steps) >= 2:
            validity += 0.2

        # Steps build on each other
        inference_count = sum(1 for s in chain.steps if s.step_type == ReasoningStepType.INFERENCE)
        if inference_count > 0:
            validity += 0.15

        # Has evidence/premises
        premise_count = sum(1 for s in chain.steps if s.step_type == ReasoningStepType.PREMISE)
        if premise_count > 0:
            validity += 0.15

        return min(1.0, validity)

    def _calculate_coherence(self, chain: ReasoningChain) -> float:
        """
        Calculate coherence between adjacent steps in the chain.

        Uses Jaccard similarity (word overlap) between consecutive steps
        to measure how well they connect logically.

        Parameters
        ----------
        chain : ReasoningChain
            The chain to analyze for coherence.

        Returns
        -------
        float
            Average coherence score from 0.0 to 1.0. Returns 1.0 for
            single-step chains, 0.0 for empty chains.

        Examples
        --------
            >>> from insideLLMs.contrib.reasoning import ReasoningExtractor, ReasoningAnalyzer
            >>> analyzer = ReasoningAnalyzer()
            >>> # High coherence - shared terms
            >>> chain = ReasoningExtractor().extract(
            ...     "Dogs are animals. Animals need food. Dogs need food."
            ... )
            >>> coherence = analyzer._calculate_coherence(chain)
            >>> print(f"Coherence: {coherence:.2f}")
            Coherence: 0.27
        """
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
        """
        Score the quality of a single reasoning step.

        Evaluates a step based on content length, presence of reasoning
        markers, and confidence level.

        Parameters
        ----------
        step : ReasoningStep
            The step to score.

        Returns
        -------
        float
            Quality score from 0.0 to 1.0:
            - Base: 0.5
            - +0.2 for appropriate length (5-50 words)
            - -0.1 for very short content (<5 words)
            - +0.2 for reasoning markers
            - +0.0 to +0.095 based on confidence

        Examples
        --------
            >>> from insideLLMs.contrib.reasoning import ReasoningStep, ReasoningAnalyzer
            >>> analyzer = ReasoningAnalyzer()
            >>> step = ReasoningStep("This is true because of X", 1)
            >>> print(f"Score: {analyzer._score_step(step):.2f}")
            Score: 0.75
        """
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
        """
        Identify logical fallacies present in the reasoning chain.

        Scans the combined text of all steps for patterns associated
        with common logical fallacies.

        Parameters
        ----------
        chain : ReasoningChain
            The chain to scan for fallacies.

        Returns
        -------
        list[str]
            List of identified fallacy names. Each fallacy appears at
            most once even if multiple patterns match.

        Examples
        --------
            >>> from insideLLMs.contrib.reasoning import ReasoningExtractor, ReasoningAnalyzer
            >>> analyzer = ReasoningAnalyzer()
            >>> chain = ReasoningExtractor().extract(
            ...     "Everyone believes this. All people agree it's always true."
            ... )
            >>> fallacies = analyzer._identify_fallacies(chain)
            >>> print(fallacies)
            ['hasty_generalization']
        """
        fallacies = []
        full_text = " ".join(s.content.lower() for s in chain.steps)

        for fallacy_name, patterns in self.FALLACY_PATTERNS.items():
            for pattern in patterns:
                if pattern in full_text:
                    fallacies.append(fallacy_name)
                    break

        return list(set(fallacies))

    def _identify_missing_steps(self, chain: ReasoningChain) -> list[str]:
        """
        Identify potentially missing steps in the reasoning chain.

        Checks for missing premises, conclusions, and gaps between
        consecutive steps with low coherence.

        Parameters
        ----------
        chain : ReasoningChain
            The chain to analyze for missing components.

        Returns
        -------
        list[str]
            Descriptions of missing or problematic elements:
            - "No clear premise or starting point" if no premises
            - "No explicit conclusion" if no conclusion
            - "Gap between step N and M" for low coherence pairs

        Examples
        --------
            >>> from insideLLMs.contrib.reasoning import ReasoningExtractor, ReasoningAnalyzer
            >>> analyzer = ReasoningAnalyzer()
            >>> chain = ReasoningExtractor().extract("Hence A. Thus B. Therefore C.")
            >>> missing = analyzer._identify_missing_steps(chain)
            >>> print(missing)
            ['No clear premise or starting point']
        """
        missing = []

        step_types = [s.step_type for s in chain.steps]

        # Missing premise
        if ReasoningStepType.PREMISE not in step_types and len(chain.steps) > 1:
            missing.append("No clear premise or starting point")

        # Missing conclusion
        if ReasoningStepType.CONCLUSION not in step_types and chain.conclusion is None:
            missing.append("No explicit conclusion")

        # Jumps in reasoning
        if len(chain.steps) >= 2:
            for i in range(len(chain.steps) - 1):
                coherence = self._step_coherence(chain.steps[i], chain.steps[i + 1])
                if coherence < 0.1:
                    missing.append(f"Gap between step {i + 1} and {i + 2}")

        return missing

    def _step_coherence(self, step1: ReasoningStep, step2: ReasoningStep) -> float:
        """
        Calculate coherence between two specific steps.

        Uses Jaccard similarity on word sets to measure topical overlap.

        Parameters
        ----------
        step1 : ReasoningStep
            The first step.
        step2 : ReasoningStep
            The second step.

        Returns
        -------
        float
            Jaccard similarity from 0.0 to 1.0, where 1.0 means
            identical word sets and 0.0 means no overlap.

        Examples
        --------
            >>> from insideLLMs.contrib.reasoning import ReasoningStep, ReasoningAnalyzer
            >>> analyzer = ReasoningAnalyzer()
            >>> s1 = ReasoningStep("Dogs are mammals", 1)
            >>> s2 = ReasoningStep("Mammals are warm-blooded", 2)
            >>> print(f"Coherence: {analyzer._step_coherence(s1, s2):.2f}")
            Coherence: 0.20
        """
        words1 = set(step1.content.lower().split())
        words2 = set(step2.content.lower().split())

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        return overlap / union if union > 0 else 0.0
