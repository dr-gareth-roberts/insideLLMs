"""
Advanced text analysis utilities for LLM exploration.

Provides high-level analysis tools for examining text characteristics,
patterns, and quality metrics relevant to LLM evaluation. This module
includes comprehensive profiling, readability scoring, content structure
analysis, tone detection, and response quality assessment.

Key Features:
    - TextProfile: Comprehensive text statistics (word counts, lexical diversity)
    - ReadabilityMetrics: Multiple readability formulas (Flesch, Gunning Fog, etc.)
    - ContentAnalysis: Detection of lists, code blocks, questions, URLs
    - ToneAnalysis: Formality, confidence, subjectivity, and sentiment scoring
    - ResponseQualityScore: Multi-dimensional LLM response quality assessment

Examples:
    Basic text analysis:

    >>> from insideLLMs.nlp.text_analysis import analyze_text
    >>> result = analyze_text("The quick brown fox jumps over the lazy dog.")
    >>> result["profile"].word_count
    9
    >>> result["readability"].flesch_reading_ease > 80  # Very readable
    True

    Comparing LLM responses:

    >>> from insideLLMs.nlp.text_analysis import compare_responses
    >>> prompt = "Explain what Python is."
    >>> response1 = "Python is a programming language."
    >>> response2 = "Python is a high-level, interpreted programming language."
    >>> comparison = compare_responses(response1, response2, prompt)
    >>> comparison["winner"]
    'response2'

    Scoring response quality:

    >>> from insideLLMs.nlp.text_analysis import score_response
    >>> score = score_response(
    ...     response="Machine learning is a subset of AI.",
    ...     prompt="What is machine learning?",
    ...     required_keywords=["learning", "AI"]
    ... )
    >>> score.overall > 0.5
    True

    Using the TextAnalyzer class directly:

    >>> from insideLLMs.nlp.text_analysis import TextAnalyzer
    >>> analyzer = TextAnalyzer()
    >>> tone = analyzer.analyze_tone("I absolutely love this amazing product!")
    >>> tone.sentiment_polarity > 0
    True
    >>> tone.subjectivity_score > 0
    True
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from insideLLMs.nlp.tokenization import word_tokenize_regex


@dataclass
class TextProfile:
    """
    Comprehensive profile of text characteristics.

    A dataclass that captures detailed statistics about a text, including
    word and character counts, sentence structure, vocabulary analysis,
    and lexical diversity metrics. Use the `from_text` class method to
    create a profile from raw text.

    Attributes:
        text: The original input text.
        char_count: Total number of characters in the text.
        word_count: Total number of words (tokens).
        sentence_count: Number of sentences detected.
        paragraph_count: Number of paragraphs (separated by blank lines).
        unique_words: Count of distinct words (vocabulary size).
        avg_word_length: Average length of words in characters.
        avg_sentence_length: Average number of words per sentence.
        lexical_diversity: Type-token ratio (unique_words / word_count).
        vocabulary: Set of all unique words in the text.
        word_frequencies: Dictionary mapping words to their occurrence counts.

    Examples:
        Creating a profile from text:

        >>> profile = TextProfile.from_text("Hello world. Hello again.")
        >>> profile.word_count
        4
        >>> profile.unique_words
        3
        >>> profile.lexical_diversity
        0.75

        Analyzing a longer passage:

        >>> text = '''Python is a programming language.
        ... It is widely used for data science.
        ...
        ... Python is also great for web development.'''
        >>> profile = TextProfile.from_text(text)
        >>> profile.sentence_count
        3
        >>> profile.paragraph_count
        2

        Examining word frequencies:

        >>> profile = TextProfile.from_text("the cat and the dog")
        >>> profile.word_frequencies["the"]
        2
        >>> "cat" in profile.vocabulary
        True

        Handling empty text:

        >>> empty_profile = TextProfile.from_text("")
        >>> empty_profile.word_count
        0
        >>> empty_profile.lexical_diversity
        0.0
    """

    text: str
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    unique_words: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    lexical_diversity: float = 0.0
    vocabulary: set[str] = field(default_factory=set)
    word_frequencies: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_text(cls, text: str) -> "TextProfile":
        """
        Create a comprehensive text profile from raw text.

        Analyzes the input text to compute various statistics including
        character and word counts, sentence and paragraph counts, vocabulary
        analysis, and lexical diversity metrics.

        Args:
            text: The input text to analyze. Can be empty or contain
                whitespace-only content.

        Returns:
            TextProfile: A populated TextProfile instance with all computed
                statistics. Returns a profile with zero values for empty text.

        Examples:
            Basic usage:

            >>> profile = TextProfile.from_text("Hello world!")
            >>> profile.word_count
            2
            >>> profile.char_count
            12

            Computing lexical diversity:

            >>> profile = TextProfile.from_text("a a a b b c")
            >>> profile.unique_words
            3
            >>> profile.word_count
            6
            >>> round(profile.lexical_diversity, 2)
            0.5

            Multi-sentence analysis:

            >>> text = "First sentence. Second sentence. Third sentence."
            >>> profile = TextProfile.from_text(text)
            >>> profile.sentence_count
            3
            >>> round(profile.avg_sentence_length, 1)
            2.0

            Handling edge cases:

            >>> TextProfile.from_text("   ").word_count
            0
            >>> TextProfile.from_text(None.__str__() if False else "").word_count
            0
        """
        if not text or not text.strip():
            return cls(text=text)

        # Basic counts
        words = word_tokenize_regex(text)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        word_count = len(words)
        unique_words = len(set(words))
        vocab = set(words)
        word_freq = Counter(words)

        # Averages
        avg_word_len = sum(len(w) for w in words) / word_count if word_count else 0
        avg_sent_len = word_count / len(sentences) if sentences else 0

        # Lexical diversity (type-token ratio)
        lex_div = unique_words / word_count if word_count else 0

        return cls(
            text=text,
            char_count=len(text),
            word_count=word_count,
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs),
            unique_words=unique_words,
            avg_word_length=avg_word_len,
            avg_sentence_length=avg_sent_len,
            lexical_diversity=lex_div,
            vocabulary=vocab,
            word_frequencies=dict(word_freq),
        )


@dataclass
class ReadabilityMetrics:
    """
    Collection of readability scores computed using standard formulas.

    This dataclass holds multiple readability metrics that estimate the
    difficulty level of text. Each metric uses a different formula based
    on factors like syllable count, word length, and sentence length.

    Attributes:
        flesch_reading_ease: Score from 0-100, higher = easier to read.
            90-100: Very easy (5th grade)
            60-70: Standard (8th-9th grade)
            0-30: Very difficult (college graduate)
        flesch_kincaid_grade: US grade level needed to understand the text.
        gunning_fog: Estimates years of formal education needed.
        smog_index: Simple Measure of Gobbledygook, grade level estimate.
        coleman_liau: Grade level based on characters per word and sentence.
        automated_readability: Grade level based on characters and words.
        avg_grade_level: Average of all applicable grade level metrics.

    Examples:
        Analyzing simple text (low grade level):

        >>> from insideLLMs.nlp.text_analysis import TextAnalyzer
        >>> analyzer = TextAnalyzer()
        >>> metrics = analyzer.analyze_readability("The cat sat on the mat.")
        >>> metrics.flesch_reading_ease > 80
        True
        >>> metrics.get_reading_level()
        'elementary'

        Analyzing complex text (high grade level):

        >>> complex_text = '''The epistemological implications of quantum
        ... mechanics fundamentally challenge our preconceived notions of
        ... deterministic causality in physical systems.'''
        >>> metrics = analyzer.analyze_readability(complex_text)
        >>> metrics.avg_grade_level > 12
        True
        >>> metrics.get_reading_level()
        'graduate'

        Comparing two texts:

        >>> simple = analyzer.analyze_readability("I like dogs. Dogs are fun.")
        >>> complex = analyzer.analyze_readability("Canines exhibit loyalty.")
        >>> simple.flesch_reading_ease > complex.flesch_reading_ease
        True

        Understanding the grade level interpretation:

        >>> metrics = analyzer.analyze_readability("Hello world.")
        >>> level = metrics.get_reading_level()
        >>> level in ['elementary', 'middle_school', 'high_school', 'college', 'graduate']
        True
    """

    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    smog_index: float = 0.0
    coleman_liau: float = 0.0
    automated_readability: float = 0.0
    avg_grade_level: float = 0.0

    def get_reading_level(self) -> str:
        """
        Get a human-readable description of the reading level.

        Converts the average grade level into a categorical description
        that indicates the education level typically required to
        understand the text.

        Returns:
            str: One of the following reading level categories:
                - "elementary": Grade level < 5
                - "middle_school": Grade level 5-7
                - "high_school": Grade level 8-11
                - "college": Grade level 12-15
                - "graduate": Grade level 16+

        Examples:
            >>> from insideLLMs.nlp.text_analysis import ReadabilityMetrics
            >>> metrics = ReadabilityMetrics(avg_grade_level=3.5)
            >>> metrics.get_reading_level()
            'elementary'

            >>> metrics = ReadabilityMetrics(avg_grade_level=6.0)
            >>> metrics.get_reading_level()
            'middle_school'

            >>> metrics = ReadabilityMetrics(avg_grade_level=14.0)
            >>> metrics.get_reading_level()
            'college'

            >>> metrics = ReadabilityMetrics(avg_grade_level=18.0)
            >>> metrics.get_reading_level()
            'graduate'
        """
        grade = self.avg_grade_level
        if grade < 5:
            return "elementary"
        elif grade < 8:
            return "middle_school"
        elif grade < 12:
            return "high_school"
        elif grade < 16:
            return "college"
        else:
            return "graduate"


@dataclass
class ContentAnalysis:
    """
    Analysis of content structure and patterns.

    Detects various structural elements commonly found in text, including
    questions, lists, code blocks, numbers, URLs, and email addresses.
    Also determines the dominant structure type of the content.

    Attributes:
        has_questions: True if the text contains question marks.
        has_lists: True if bullet points or numbered lists are detected.
        has_code: True if code blocks (```) or inline code (`) are found.
        has_numbers: True if numeric values are present.
        has_urls: True if HTTP/HTTPS URLs are detected.
        has_emails: True if email addresses are detected.
        question_count: Number of questions found.
        list_item_count: Number of list items detected.
        code_block_count: Number of fenced code blocks (```).
        url_count: Number of URLs found.
        dominant_structure: Primary content type: "prose", "list", "code", or "q&a".

    Examples:
        Analyzing prose content:

        >>> from insideLLMs.nlp.text_analysis import TextAnalyzer
        >>> analyzer = TextAnalyzer()
        >>> content = analyzer.analyze_content("This is a paragraph of text.")
        >>> content.dominant_structure
        'prose'
        >>> content.has_lists
        False

        Detecting questions:

        >>> content = analyzer.analyze_content("What is Python? How does it work?")
        >>> content.has_questions
        True
        >>> content.question_count
        2

        Analyzing list content:

        >>> text = '''Shopping list:
        ... - Apples
        ... - Bananas
        ... - Oranges'''
        >>> content = analyzer.analyze_content(text)
        >>> content.has_lists
        True
        >>> content.list_item_count
        3
        >>> content.dominant_structure
        'list'

        Detecting code blocks:

        >>> code_text = '''Here is some code:
        ... ```python
        ... print("Hello")
        ... ```'''
        >>> content = analyzer.analyze_content(code_text)
        >>> content.has_code
        True
        >>> content.code_block_count
        1
    """

    has_questions: bool = False
    has_lists: bool = False
    has_code: bool = False
    has_numbers: bool = False
    has_urls: bool = False
    has_emails: bool = False
    question_count: int = 0
    list_item_count: int = 0
    code_block_count: int = 0
    url_count: int = 0
    dominant_structure: str = "prose"


@dataclass
class ToneAnalysis:
    """
    Analysis of text tone and style.

    Provides scores for various stylistic dimensions of text, including
    formality, confidence, subjectivity, and sentiment. These scores are
    computed using lexical analysis based on word lists.

    Attributes:
        formality_score: Scale from 0.0 (informal) to 1.0 (formal).
            Based on presence of formal vs. informal vocabulary and
            linguistic markers like contractions.
        confidence_score: Scale from 0.0 (uncertain) to 1.0 (confident).
            Based on hedging words vs. confident language.
        subjectivity_score: Scale from 0.0 (objective) to 1.0 (subjective).
            Based on presence of opinion-laden and evaluative words.
        sentiment_polarity: Scale from -1.0 (negative) to 1.0 (positive).
            Based on positive vs. negative vocabulary.

    Examples:
        Analyzing formal text:

        >>> from insideLLMs.nlp.text_analysis import TextAnalyzer
        >>> analyzer = TextAnalyzer()
        >>> tone = analyzer.analyze_tone("Therefore, we conclude that the hypothesis is valid.")
        >>> tone.formality_score > 0.5
        True
        >>> tone.get_formality_level()
        'formal'

        Analyzing informal text:

        >>> tone = analyzer.analyze_tone("Hey, this is really cool stuff!")
        >>> tone.formality_score < 0.5
        True
        >>> tone.get_formality_level()
        'informal'

        Detecting confident language:

        >>> tone = analyzer.analyze_tone("This will definitely work. It is absolutely correct.")
        >>> tone.confidence_score > 0.5
        True

        Detecting hedging/uncertainty:

        >>> tone = analyzer.analyze_tone("This might possibly work, but I'm uncertain.")
        >>> tone.confidence_score < 0.5
        True

        Sentiment analysis:

        >>> positive = analyzer.analyze_tone("This is amazing and wonderful!")
        >>> positive.sentiment_polarity > 0
        True
        >>> negative = analyzer.analyze_tone("This is terrible and horrible.")
        >>> negative.sentiment_polarity < 0
        True
    """

    formality_score: float = 0.0  # 0=informal, 1=formal
    confidence_score: float = 0.0  # 0=uncertain, 1=confident
    subjectivity_score: float = 0.0  # 0=objective, 1=subjective
    sentiment_polarity: float = 0.0  # -1=negative, 0=neutral, 1=positive

    def get_formality_level(self) -> str:
        """
        Get a categorical description of the formality level.

        Converts the formality score into a human-readable category
        that describes the overall tone of the text.

        Returns:
            str: One of the following formality categories:
                - "informal": Formality score < 0.3
                - "neutral": Formality score 0.3-0.7
                - "formal": Formality score > 0.7

        Examples:
            >>> from insideLLMs.nlp.text_analysis import ToneAnalysis
            >>> ToneAnalysis(formality_score=0.1).get_formality_level()
            'informal'

            >>> ToneAnalysis(formality_score=0.5).get_formality_level()
            'neutral'

            >>> ToneAnalysis(formality_score=0.9).get_formality_level()
            'formal'

            >>> ToneAnalysis(formality_score=0.3).get_formality_level()
            'neutral'
        """
        if self.formality_score < 0.3:
            return "informal"
        elif self.formality_score < 0.7:
            return "neutral"
        else:
            return "formal"


class TextAnalyzer:
    """
    Comprehensive text analyzer for LLM output evaluation.

    This class provides a suite of analysis methods for examining text
    from multiple perspectives: statistical profiling, readability assessment,
    content structure detection, tone analysis, and text comparison.

    The analyzer uses lexicon-based approaches with predefined word lists
    for informal/formal vocabulary, hedging words, confident words, and
    subjective terms.

    Class Attributes:
        INFORMAL_WORDS: Set of words indicating casual/informal writing.
        FORMAL_WORDS: Set of words indicating formal/academic writing.
        HEDGE_WORDS: Set of words indicating uncertainty or hedging.
        CONFIDENT_WORDS: Set of words indicating certainty or confidence.
        SUBJECTIVE_WORDS: Set of words indicating opinions or subjectivity.

    Examples:
        Basic text profiling:

        >>> analyzer = TextAnalyzer()
        >>> profile = analyzer.profile("The quick brown fox jumps.")
        >>> profile.word_count
        5
        >>> profile.sentence_count
        1

        Readability analysis:

        >>> readability = analyzer.analyze_readability("Simple words. Short sentences.")
        >>> readability.flesch_reading_ease > 70  # Easy to read
        True

        Content structure analysis:

        >>> content = analyzer.analyze_content("What is Python? Why use it?")
        >>> content.has_questions
        True
        >>> content.question_count
        2

        Tone analysis:

        >>> tone = analyzer.analyze_tone("This is absolutely wonderful!")
        >>> tone.sentiment_polarity > 0
        True

        Comparing two texts:

        >>> comparison = analyzer.compare_texts(
        ...     "Python is simple.",
        ...     "Python is a high-level programming language."
        ... )
        >>> comparison["word_count"][1] > comparison["word_count"][0]
        True

        Vocabulary overlap analysis:

        >>> similarity, common, unique1, unique2 = analyzer.vocabulary_overlap(
        ...     "the cat sat on the mat",
        ...     "the dog sat on the rug"
        ... )
        >>> "sat" in common
        True
        >>> "cat" in unique1
        True
        >>> "dog" in unique2
        True
    """

    # Words indicating informal writing
    INFORMAL_WORDS = {
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
        "hey",
        "hi",
        "bye",
        "lol",
        "haha",
        "wow",
        "cool",
        "awesome",
        "stuff",
        "things",
        "like",
        "really",
        "very",
        "just",
        "actually",
        "basically",
        "literally",
        "totally",
        "pretty",
    }

    # Words indicating formal writing
    FORMAL_WORDS = {
        "therefore",
        "consequently",
        "furthermore",
        "moreover",
        "nevertheless",
        "notwithstanding",
        "whereas",
        "hereby",
        "henceforth",
        "accordingly",
        "subsequently",
        "aforementioned",
        "regarding",
        "concerning",
        "pursuant",
        "herein",
        "therein",
        "wherein",
        "hereafter",
        "theretofore",
    }

    # Hedging words (uncertainty)
    HEDGE_WORDS = {
        "maybe",
        "perhaps",
        "possibly",
        "might",
        "could",
        "may",
        "seem",
        "appear",
        "suggest",
        "believe",
        "think",
        "assume",
        "suppose",
        "probably",
        "likely",
        "unlikely",
        "uncertain",
        "unclear",
    }

    # Confident words
    CONFIDENT_WORDS = {
        "certainly",
        "definitely",
        "absolutely",
        "clearly",
        "obviously",
        "undoubtedly",
        "surely",
        "precisely",
        "exactly",
        "always",
        "never",
        "must",
        "will",
        "shall",
        "proven",
        "established",
        "confirmed",
    }

    # Subjective words
    SUBJECTIVE_WORDS = {
        "beautiful",
        "ugly",
        "good",
        "bad",
        "best",
        "worst",
        "amazing",
        "terrible",
        "wonderful",
        "horrible",
        "love",
        "hate",
        "feel",
        "believe",
        "think",
        "opinion",
        "prefer",
        "favorite",
        "personally",
    }

    def __init__(self):
        """
        Initialize the text analyzer.

        Creates a new TextAnalyzer instance ready to perform various
        text analyses. No configuration is required.

        Examples:
            >>> analyzer = TextAnalyzer()
            >>> profile = analyzer.profile("Hello, world!")
            >>> profile.word_count
            2
        """
        pass

    def profile(self, text: str) -> TextProfile:
        """
        Create a comprehensive text profile with statistics.

        Analyzes the input text and returns a TextProfile containing
        word counts, character counts, sentence statistics, vocabulary
        information, and lexical diversity metrics.

        Args:
            text: The input text to analyze.

        Returns:
            TextProfile: A dataclass containing comprehensive text statistics.

        Examples:
            Basic profiling:

            >>> analyzer = TextAnalyzer()
            >>> profile = analyzer.profile("Hello world. How are you?")
            >>> profile.word_count
            5
            >>> profile.sentence_count
            2

            Analyzing vocabulary:

            >>> profile = analyzer.profile("the cat and the dog")
            >>> profile.unique_words
            4
            >>> profile.word_frequencies["the"]
            2

            Lexical diversity (type-token ratio):

            >>> profile = analyzer.profile("a b c d e")
            >>> profile.lexical_diversity
            1.0
            >>> profile = analyzer.profile("a a a a a")
            >>> profile.lexical_diversity
            0.2
        """
        return TextProfile.from_text(text)

    def analyze_readability(self, text: str) -> ReadabilityMetrics:
        """
        Calculate various readability metrics for the text.

        Computes multiple industry-standard readability formulas including
        Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog Index,
        SMOG Index, Coleman-Liau Index, and Automated Readability Index.

        Args:
            text: The input text to analyze.

        Returns:
            ReadabilityMetrics: A dataclass containing all computed readability
                scores and an average grade level.

        Note:
            - SMOG index requires at least 3 sentences for accuracy
            - Flesch Reading Ease is scaled 0-100 (clamped)
            - Grade levels are clamped to be non-negative

        Examples:
            Analyzing simple text:

            >>> analyzer = TextAnalyzer()
            >>> metrics = analyzer.analyze_readability("I like cats. Cats are nice.")
            >>> metrics.flesch_reading_ease > 80  # Very easy to read
            True
            >>> metrics.avg_grade_level < 5  # Elementary level
            True

            Analyzing academic text:

            >>> text = '''The epistemological paradigm shift necessitates
            ... a fundamental reconceptualization of our methodological
            ... approaches to phenomenological investigation.'''
            >>> metrics = analyzer.analyze_readability(text)
            >>> metrics.gunning_fog > 12  # Difficult
            True

            Comparing readability levels:

            >>> simple = analyzer.analyze_readability("Run. Jump. Play.")
            >>> complex = analyzer.analyze_readability("Furthermore, the analysis reveals significant implications.")
            >>> simple.flesch_reading_ease > complex.flesch_reading_ease
            True

            Empty text handling:

            >>> metrics = analyzer.analyze_readability("")
            >>> metrics.flesch_reading_ease
            0.0
        """
        if not text or not text.strip():
            return ReadabilityMetrics()

        words = re.findall(r"\b\w+\b", text)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        word_count = len(words)
        sentence_count = len(sentences) or 1
        char_count = sum(len(w) for w in words)

        # Count syllables
        syllable_count = sum(self._count_syllables(w) for w in words)

        # Count complex words (3+ syllables)
        complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)

        # Flesch Reading Ease
        fre = (
            206.835
            - 1.015 * (word_count / sentence_count)
            - 84.6 * (syllable_count / word_count if word_count else 0)
        )
        fre = max(0, min(100, fre))

        # Flesch-Kincaid Grade Level
        fk_grade = (
            0.39 * (word_count / sentence_count)
            + 11.8 * (syllable_count / word_count if word_count else 0)
            - 15.59
        )
        fk_grade = max(0, fk_grade)

        # Gunning Fog Index
        fog = 0.4 * (
            (word_count / sentence_count) + 100 * (complex_words / word_count if word_count else 0)
        )
        fog = max(0, fog)

        # SMOG Index (requires 30+ sentences ideally)
        smog = 1.0430 * math.sqrt(complex_words * (30 / sentence_count)) + 3.1291
        smog = max(0, smog) if sentence_count >= 3 else 0

        # Coleman-Liau Index
        L = (char_count / word_count) * 100 if word_count else 0
        S = (sentence_count / word_count) * 100 if word_count else 0
        cli = 0.0588 * L - 0.296 * S - 15.8
        cli = max(0, cli)

        # Automated Readability Index
        ari = (
            4.71 * (char_count / word_count if word_count else 0)
            + 0.5 * (word_count / sentence_count)
            - 21.43
        )
        ari = max(0, ari)

        # Average grade level
        grades = [fk_grade, fog, cli, ari]
        if smog > 0:
            grades.append(smog)
        avg_grade = sum(grades) / len(grades)

        return ReadabilityMetrics(
            flesch_reading_ease=round(fre, 2),
            flesch_kincaid_grade=round(fk_grade, 2),
            gunning_fog=round(fog, 2),
            smog_index=round(smog, 2),
            coleman_liau=round(cli, 2),
            automated_readability=round(ari, 2),
            avg_grade_level=round(avg_grade, 2),
        )

    def analyze_content(self, text: str) -> ContentAnalysis:
        """
        Analyze content structure and detect patterns in the text.

        Identifies various structural elements commonly found in text,
        including questions, lists (bulleted and numbered), code blocks,
        numeric values, URLs, and email addresses. Also determines the
        dominant structure type.

        Args:
            text: The input text to analyze.

        Returns:
            ContentAnalysis: A dataclass containing boolean flags for
                detected elements, counts, and the dominant structure type.

        Note:
            - List items are detected using common markers: -, *, bullet, 1), 1.
            - Code blocks must be fenced with triple backticks (```)
            - URLs must start with http:// or https://
            - Dominant structure is determined by relative counts

        Examples:
            Analyzing prose:

            >>> analyzer = TextAnalyzer()
            >>> content = analyzer.analyze_content("This is a simple paragraph.")
            >>> content.dominant_structure
            'prose'
            >>> content.has_lists
            False

            Detecting questions:

            >>> content = analyzer.analyze_content("What is Python? How does it work? Is it free?")
            >>> content.has_questions
            True
            >>> content.question_count
            3
            >>> content.dominant_structure
            'q&a'

            Analyzing list content:

            >>> list_text = '''Tasks:
            ... - Buy groceries
            ... - Walk the dog
            ... - Write code'''
            >>> content = analyzer.analyze_content(list_text)
            >>> content.has_lists
            True
            >>> content.list_item_count
            3

            Detecting URLs and emails:

            >>> text = "Visit https://example.com or email test@example.com"
            >>> content = analyzer.analyze_content(text)
            >>> content.has_urls
            True
            >>> content.has_emails
            True
        """
        if not text:
            return ContentAnalysis()

        # Detect questions
        questions = re.findall(r"[^.!?]*\?", text)
        has_questions = len(questions) > 0

        # Detect lists (bullets, numbers)
        list_patterns = [
            r"^[\-\*\•]\s+.+$",  # Bullet points
            r"^\d+[\.\)]\s+.+$",  # Numbered lists
        ]
        list_items = []
        for line in text.split("\n"):
            for pattern in list_patterns:
                if re.match(pattern, line.strip()):
                    list_items.append(line)
                    break
        has_lists = len(list_items) > 0

        # Detect code blocks
        code_blocks = re.findall(r"```[\s\S]*?```", text)
        inline_code = re.findall(r"`[^`]+`", text)
        has_code = len(code_blocks) > 0 or len(inline_code) > 0

        # Detect numbers
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        has_numbers = len(numbers) > 0

        # Detect URLs
        urls = re.findall(r"https?://\S+", text)
        has_urls = len(urls) > 0

        # Detect emails
        emails = re.findall(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)
        has_emails = len(emails) > 0

        # Determine dominant structure
        if len(code_blocks) > 0 and len(code_blocks) >= len(list_items):
            dominant = "code"
        elif has_lists and len(list_items) >= len(questions):
            dominant = "list"
        elif has_questions and len(questions) > 2:
            dominant = "q&a"
        else:
            dominant = "prose"

        return ContentAnalysis(
            has_questions=has_questions,
            has_lists=has_lists,
            has_code=has_code,
            has_numbers=has_numbers,
            has_urls=has_urls,
            has_emails=has_emails,
            question_count=len(questions),
            list_item_count=len(list_items),
            code_block_count=len(code_blocks),
            url_count=len(urls),
            dominant_structure=dominant,
        )

    def analyze_tone(self, text: str) -> ToneAnalysis:
        """
        Analyze the tone and style of text.

        Evaluates multiple dimensions of writing style including formality,
        confidence, subjectivity, and sentiment polarity. Uses lexicon-based
        analysis with predefined word lists.

        Args:
            text: The input text to analyze.

        Returns:
            ToneAnalysis: A dataclass containing scores for formality (0-1),
                confidence (0-1), subjectivity (0-1), and sentiment (-1 to 1).

        Note:
            - Formality considers contractions and first-person pronouns
            - Confidence is based on hedge words vs. confident language
            - Subjectivity detects opinion-laden vocabulary
            - Sentiment uses simple positive/negative word counts

        Examples:
            Analyzing formal writing:

            >>> analyzer = TextAnalyzer()
            >>> tone = analyzer.analyze_tone("Furthermore, the analysis demonstrates conclusive evidence.")
            >>> tone.formality_score > 0.5
            True

            Detecting informal tone:

            >>> tone = analyzer.analyze_tone("Hey, this is really cool stuff, ya know?")
            >>> tone.get_formality_level()
            'informal'

            Confidence analysis:

            >>> confident = analyzer.analyze_tone("This is definitely correct. It will absolutely work.")
            >>> uncertain = analyzer.analyze_tone("This might possibly work, perhaps.")
            >>> confident.confidence_score > uncertain.confidence_score
            True

            Sentiment detection:

            >>> positive = analyzer.analyze_tone("This is wonderful and amazing!")
            >>> negative = analyzer.analyze_tone("This is terrible and awful.")
            >>> positive.sentiment_polarity > 0
            True
            >>> negative.sentiment_polarity < 0
            True

            Subjectivity analysis:

            >>> subjective = analyzer.analyze_tone("I believe this is the best option.")
            >>> subjective.subjectivity_score > 0
            True
        """
        if not text:
            return ToneAnalysis()

        word_list = word_tokenize_regex(text)
        words = set(word_list)
        word_count = len(word_list)

        if word_count == 0:
            return ToneAnalysis()

        # Formality score
        informal_count = len(words & self.INFORMAL_WORDS)
        formal_count = len(words & self.FORMAL_WORDS)
        total_markers = informal_count + formal_count
        if total_markers > 0:
            formality = formal_count / total_markers
        else:
            # Use heuristics: contractions, first person, etc.
            contractions = len(re.findall(r"\b\w+'\w+\b", text))
            first_person = sum(1 for w in word_list if w in {"i", "me", "my", "we", "us"})
            formality = 0.5 - (contractions + first_person) / (word_count * 2)
            formality = max(0, min(1, formality))

        # Confidence score
        hedge_count = len(words & self.HEDGE_WORDS)
        confident_count = len(words & self.CONFIDENT_WORDS)
        uncertainty_markers = hedge_count + confident_count
        confidence = confident_count / uncertainty_markers if uncertainty_markers > 0 else 0.5

        # Subjectivity score
        subjective_count = len(words & self.SUBJECTIVE_WORDS)
        subjectivity = min(1.0, subjective_count / (word_count * 0.05) if word_count > 0 else 0)

        # Simple sentiment polarity
        positive_words = {"good", "great", "excellent", "amazing", "wonderful", "best", "love"}
        negative_words = {"bad", "terrible", "awful", "worst", "hate", "poor", "horrible"}
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total_sentiment = pos_count + neg_count
        polarity = (pos_count - neg_count) / total_sentiment if total_sentiment > 0 else 0.0

        return ToneAnalysis(
            formality_score=round(formality, 3),
            confidence_score=round(confidence, 3),
            subjectivity_score=round(subjectivity, 3),
            sentiment_polarity=round(polarity, 3),
        )

    def compare_texts(self, text1: str, text2: str) -> dict[str, tuple[float, float, float]]:
        """
        Compare two texts across multiple dimensions.

        Analyzes both texts for various metrics and returns a comparison
        showing values for each text and the difference between them.
        Useful for comparing LLM responses or evaluating text variations.

        Args:
            text1: The first text to compare.
            text2: The second text to compare.

        Returns:
            dict[str, tuple[float, float, float]]: A dictionary mapping metric
                names to tuples of (text1_value, text2_value, difference).
                Difference is calculated as text2_value - text1_value.

                Included metrics:
                - word_count: Total number of words
                - lexical_diversity: Type-token ratio
                - avg_word_length: Average word length in characters
                - grade_level: Average readability grade level
                - formality: Formality score (0-1)
                - confidence: Confidence score (0-1)

        Examples:
            Comparing response lengths:

            >>> analyzer = TextAnalyzer()
            >>> comparison = analyzer.compare_texts(
            ...     "Python is good.",
            ...     "Python is an excellent programming language for beginners."
            ... )
            >>> comparison["word_count"][0] < comparison["word_count"][1]
            True
            >>> comparison["word_count"][2] > 0  # Positive difference
            True

            Comparing formality:

            >>> formal = "Furthermore, we must consider the implications."
            >>> informal = "Hey, we gotta think about this stuff."
            >>> comparison = analyzer.compare_texts(informal, formal)
            >>> comparison["formality"][1] > comparison["formality"][0]
            True

            Comparing lexical diversity:

            >>> repetitive = "the cat and the cat and the cat"
            >>> diverse = "the cat and the dog and the bird"
            >>> comparison = analyzer.compare_texts(repetitive, diverse)
            >>> comparison["lexical_diversity"][1] > comparison["lexical_diversity"][0]
            True

            Analyzing grade level differences:

            >>> simple = "I like dogs. Dogs are fun."
            >>> complex = "Canines demonstrate remarkable behavioral adaptability."
            >>> comparison = analyzer.compare_texts(simple, complex)
            >>> comparison["grade_level"][1] > comparison["grade_level"][0]
            True
        """
        profile1 = self.profile(text1)
        profile2 = self.profile(text2)

        readability1 = self.analyze_readability(text1)
        readability2 = self.analyze_readability(text2)

        tone1 = self.analyze_tone(text1)
        tone2 = self.analyze_tone(text2)

        return {
            "word_count": (
                profile1.word_count,
                profile2.word_count,
                profile2.word_count - profile1.word_count,
            ),
            "lexical_diversity": (
                profile1.lexical_diversity,
                profile2.lexical_diversity,
                profile2.lexical_diversity - profile1.lexical_diversity,
            ),
            "avg_word_length": (
                profile1.avg_word_length,
                profile2.avg_word_length,
                profile2.avg_word_length - profile1.avg_word_length,
            ),
            "grade_level": (
                readability1.avg_grade_level,
                readability2.avg_grade_level,
                readability2.avg_grade_level - readability1.avg_grade_level,
            ),
            "formality": (
                tone1.formality_score,
                tone2.formality_score,
                tone2.formality_score - tone1.formality_score,
            ),
            "confidence": (
                tone1.confidence_score,
                tone2.confidence_score,
                tone2.confidence_score - tone1.confidence_score,
            ),
        }

    def vocabulary_overlap(
        self, text1: str, text2: str
    ) -> tuple[float, set[str], set[str], set[str]]:
        """
        Calculate vocabulary overlap between two texts.

        Computes the Jaccard similarity coefficient and identifies common
        and unique vocabulary between two texts. Useful for measuring
        semantic similarity and comparing LLM responses.

        Args:
            text1: The first text to compare.
            text2: The second text to compare.

        Returns:
            tuple[float, set[str], set[str], set[str]]: A tuple containing:
                - jaccard_similarity: Overlap coefficient (0-1), where 1 means
                  identical vocabulary and 0 means no shared words.
                - common_words: Set of words appearing in both texts.
                - unique_to_text1: Set of words only in text1.
                - unique_to_text2: Set of words only in text2.

        Examples:
            Identical texts have similarity of 1.0:

            >>> analyzer = TextAnalyzer()
            >>> similarity, common, u1, u2 = analyzer.vocabulary_overlap(
            ...     "hello world",
            ...     "hello world"
            ... )
            >>> similarity
            1.0
            >>> len(u1) == len(u2) == 0
            True

            Completely different texts have similarity of 0.0:

            >>> similarity, common, u1, u2 = analyzer.vocabulary_overlap(
            ...     "cat dog bird",
            ...     "apple banana orange"
            ... )
            >>> similarity
            0.0
            >>> len(common)
            0

            Partial overlap:

            >>> similarity, common, u1, u2 = analyzer.vocabulary_overlap(
            ...     "the cat sat on the mat",
            ...     "the dog sat on the rug"
            ... )
            >>> "sat" in common
            True
            >>> "the" in common
            True
            >>> "cat" in u1
            True
            >>> "dog" in u2
            True
            >>> 0 < similarity < 1
            True

            Using overlap to compare LLM responses:

            >>> response1 = "Python is a programming language."
            >>> response2 = "Python is a versatile programming language."
            >>> similarity, _, _, _ = analyzer.vocabulary_overlap(response1, response2)
            >>> similarity > 0.5  # High overlap
            True
        """
        words1 = set(word_tokenize_regex(text1))
        words2 = set(word_tokenize_regex(text2))

        common = words1 & words2
        unique1 = words1 - words2
        unique2 = words2 - words1

        union = words1 | words2
        jaccard = len(common) / len(union) if union else 0.0

        return (jaccard, common, unique1, unique2)

    def _count_syllables(self, word: str) -> int:
        """
        Count the number of syllables in a word.

        Uses a heuristic approach based on vowel group counting.
        Handles trailing silent 'e' and ensures minimum of 1 syllable.

        Args:
            word: The word to analyze.

        Returns:
            int: Estimated number of syllables (minimum 1).

        Examples:
            >>> analyzer = TextAnalyzer()
            >>> analyzer._count_syllables("cat")
            1
            >>> analyzer._count_syllables("hello")
            2
            >>> analyzer._count_syllables("beautiful")
            3
            >>> analyzer._count_syllables("a")
            1
        """
        word = word.lower()
        if len(word) <= 3:
            return 1

        # Remove trailing e
        if word.endswith("e"):
            word = word[:-1]

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        return max(1, count)


@dataclass
class ResponseQualityScore:
    """
    Quality scoring for LLM responses.

    Provides a multi-dimensional quality assessment of LLM responses
    based on relevance to the prompt, completeness, coherence, and
    conciseness. Scores are normalized to 0-1 range.

    Attributes:
        relevance: Score from 0-1 measuring how relevant the response is
            to the original prompt. Based on vocabulary overlap and
            required keyword presence.
        completeness: Score from 0-1 measuring how complete the response
            is. Based on response length relative to expected length.
        coherence: Score from 0-1 measuring logical structure. Based on
            sentence count and presence of transition words.
        conciseness: Score from 0-1 measuring efficiency. Penalizes
            repetition and filler words.
        overall: Weighted average of all scores (relevance: 30%,
            completeness: 25%, coherence: 25%, conciseness: 20%).

    Examples:
        Basic quality scoring:

        >>> from insideLLMs.nlp.text_analysis import ResponseQualityScore
        >>> score = ResponseQualityScore.calculate(
        ...     response="Python is a programming language used for various tasks.",
        ...     prompt="What is Python?"
        ... )
        >>> score.overall > 0
        True

        Scoring with expected length:

        >>> score = ResponseQualityScore.calculate(
        ...     response="Python is a language.",
        ...     prompt="Explain Python in detail.",
        ...     expected_length=100
        ... )
        >>> score.completeness < 0.5  # Too short
        True

        Scoring with required keywords:

        >>> score = ResponseQualityScore.calculate(
        ...     response="Machine learning uses algorithms to learn patterns.",
        ...     prompt="What is machine learning?",
        ...     required_keywords=["learning", "algorithms", "patterns"]
        ... )
        >>> score.relevance > 0.5  # Contains required keywords
        True

        Comparing two responses:

        >>> concise = ResponseQualityScore.calculate(
        ...     response="Python is versatile and beginner-friendly.",
        ...     prompt="Why use Python?"
        ... )
        >>> verbose = ResponseQualityScore.calculate(
        ...     response="Python is very really basically just actually a language.",
        ...     prompt="Why use Python?"
        ... )
        >>> concise.conciseness > verbose.conciseness  # Less filler words
        True
    """

    relevance: float = 0.0  # How relevant to the prompt
    completeness: float = 0.0  # How complete the response is
    coherence: float = 0.0  # How well-structured and logical
    conciseness: float = 0.0  # How efficient (not verbose)
    overall: float = 0.0

    @classmethod
    def calculate(
        cls,
        response: str,
        prompt: str,
        expected_length: Optional[int] = None,
        required_keywords: Optional[list[str]] = None,
    ) -> "ResponseQualityScore":
        """
        Calculate a comprehensive quality score for an LLM response.

        Evaluates the response across four dimensions: relevance to the
        prompt, completeness (length appropriateness), coherence (structure
        and flow), and conciseness (lack of filler and repetition).

        Args:
            response: The LLM-generated response to evaluate.
            prompt: The original prompt that generated the response.
            expected_length: Optional expected word count. If provided,
                completeness is scored based on how close the response
                is to this length. Default assumes 50-500 words is reasonable.
            required_keywords: Optional list of keywords that should appear
                in the response. Presence of these keywords contributes to
                the relevance score.

        Returns:
            ResponseQualityScore: A dataclass containing individual dimension
                scores and an overall weighted score.

        Examples:
            Basic scoring:

            >>> score = ResponseQualityScore.calculate(
            ...     response="Python is a versatile programming language.",
            ...     prompt="What is Python?"
            ... )
            >>> 0 <= score.overall <= 1
            True

            With expected length constraint:

            >>> short_score = ResponseQualityScore.calculate(
            ...     response="Yes.",
            ...     prompt="Explain machine learning in detail.",
            ...     expected_length=200
            ... )
            >>> short_score.completeness < 0.3  # Way too short
            True

            With required keywords:

            >>> score = ResponseQualityScore.calculate(
            ...     response="Neural networks process data through layers.",
            ...     prompt="How do neural networks work?",
            ...     required_keywords=["neural", "networks", "layers", "data"]
            ... )
            >>> score.relevance > 0.5
            True

            Evaluating coherence with transitions:

            >>> coherent = ResponseQualityScore.calculate(
            ...     response="First, we define the problem. Then, we gather data. Finally, we train the model.",
            ...     prompt="How to build ML models?"
            ... )
            >>> coherent.coherence > 0.5
            True
        """
        analyzer = TextAnalyzer()

        # Relevance: vocabulary overlap with prompt
        jaccard, common, _, _ = analyzer.vocabulary_overlap(prompt, response)
        relevance = min(1.0, jaccard * 3)  # Scale up since prompts are usually short

        # Check for required keywords
        if required_keywords:
            response_lower = response.lower()
            keyword_matches = sum(1 for kw in required_keywords if kw.lower() in response_lower)
            keyword_score = keyword_matches / len(required_keywords)
            relevance = (relevance + keyword_score) / 2

        # Completeness: based on response length vs expected
        response_words = len(re.findall(r"\b\w+\b", response))
        if expected_length:
            # Score based on how close to expected length
            ratio = response_words / expected_length
            if ratio < 0.5:
                completeness = ratio * 2
            elif ratio > 2.0:
                completeness = max(0, 1 - (ratio - 2) * 0.5)
            else:
                completeness = 1.0
        else:
            # Default: assume reasonable response is 50-500 words
            if response_words < 10:
                completeness = response_words / 10
            elif response_words > 1000:
                completeness = max(0.5, 1 - (response_words - 1000) / 2000)
            else:
                completeness = 1.0

        # Coherence: based on sentence structure and connectors
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Check for transition words
        connectors = {
            "however",
            "therefore",
            "furthermore",
            "moreover",
            "additionally",
            "consequently",
            "thus",
            "hence",
            "first",
            "second",
            "third",
            "finally",
            "in conclusion",
            "to summarize",
            "for example",
        }
        response_lower = response.lower()
        connector_count = sum(1 for c in connectors if c in response_lower)

        # Score based on sentence count and connectors
        if len(sentences) > 1:
            coherence = min(1.0, 0.5 + connector_count * 0.1)
        else:
            coherence = 0.5 if response_words > 10 else 0.3

        # Conciseness: penalize excessive repetition and filler
        analyzer.profile(response)
        word_list = word_tokenize_regex(response)

        # Check for repetition
        word_freq = Counter(word_list)
        if word_freq:
            most_common_freq = max(word_freq.values())
            repetition_ratio = most_common_freq / len(word_list) if word_list else 0
        else:
            repetition_ratio = 0

        # Filler words
        fillers = {"very", "really", "just", "actually", "basically", "literally"}
        filler_count = sum(1 for w in word_list if w in fillers)
        filler_ratio = filler_count / len(word_list) if word_list else 0

        conciseness = 1.0 - (repetition_ratio * 0.5) - (filler_ratio * 0.5)
        conciseness = max(0, conciseness)

        # Overall score (weighted average)
        overall = relevance * 0.3 + completeness * 0.25 + coherence * 0.25 + conciseness * 0.2

        return cls(
            relevance=round(relevance, 3),
            completeness=round(completeness, 3),
            coherence=round(coherence, 3),
            conciseness=round(conciseness, 3),
            overall=round(overall, 3),
        )


def analyze_text(text: str) -> dict:
    """
    Perform comprehensive text analysis.

    A convenience function that runs all available analyses on the input
    text and returns a dictionary containing profile statistics, readability
    metrics, content structure analysis, and tone analysis.

    Args:
        text: The input text to analyze.

    Returns:
        dict: A dictionary with four keys:
            - "profile": TextProfile with word counts, vocabulary, etc.
            - "readability": ReadabilityMetrics with grade levels, etc.
            - "content": ContentAnalysis with structural elements.
            - "tone": ToneAnalysis with formality, sentiment, etc.

    Examples:
        Basic analysis:

        >>> result = analyze_text("The quick brown fox jumps over the lazy dog.")
        >>> result["profile"].word_count
        9
        >>> result["readability"].flesch_reading_ease > 70
        True

        Accessing all analysis dimensions:

        >>> result = analyze_text("What is Python? It is a programming language.")
        >>> result["profile"].sentence_count
        2
        >>> result["content"].has_questions
        True
        >>> result["tone"].formality_score >= 0
        True

        Analyzing formal academic text:

        >>> academic = '''Furthermore, the empirical evidence suggests
        ... that the hypothesis is corroborated by the experimental data.'''
        >>> result = analyze_text(academic)
        >>> result["readability"].avg_grade_level > 10
        True
        >>> result["tone"].formality_score > 0.3
        True

        Analyzing informal text:

        >>> casual = "Hey, this is really cool stuff! I love it!"
        >>> result = analyze_text(casual)
        >>> result["tone"].sentiment_polarity > 0
        True
        >>> result["content"].has_questions
        False
    """
    analyzer = TextAnalyzer()
    return {
        "profile": analyzer.profile(text),
        "readability": analyzer.analyze_readability(text),
        "content": analyzer.analyze_content(text),
        "tone": analyzer.analyze_tone(text),
    }


def score_response(
    response: str,
    prompt: str,
    expected_length: Optional[int] = None,
    required_keywords: Optional[list[str]] = None,
) -> ResponseQualityScore:
    """
    Score the quality of an LLM response.

    A convenience function that creates a ResponseQualityScore for evaluating
    how well an LLM response addresses a given prompt. Measures relevance,
    completeness, coherence, and conciseness.

    Args:
        response: The LLM-generated response to evaluate.
        prompt: The original prompt that generated the response.
        expected_length: Optional expected word count for the response.
            If provided, completeness scoring considers this target.
        required_keywords: Optional list of keywords that should appear
            in the response for it to be considered relevant.

    Returns:
        ResponseQualityScore: A dataclass with scores for relevance,
            completeness, coherence, conciseness, and overall quality.

    Examples:
        Basic response scoring:

        >>> score = score_response(
        ...     response="Python is a high-level programming language.",
        ...     prompt="What is Python?"
        ... )
        >>> score.overall > 0
        True

        Scoring with length expectations:

        >>> score = score_response(
        ...     response="Yes.",
        ...     prompt="Explain Python comprehensively.",
        ...     expected_length=500
        ... )
        >>> score.completeness < 0.2  # Response is too short
        True

        Scoring with required keywords:

        >>> score = score_response(
        ...     response="Deep learning uses neural networks for pattern recognition.",
        ...     prompt="What is deep learning?",
        ...     required_keywords=["deep", "learning", "neural"]
        ... )
        >>> score.relevance > 0.5
        True

        Comparing response quality:

        >>> good = score_response(
        ...     response="Python is versatile. Furthermore, it has excellent libraries.",
        ...     prompt="Why Python?"
        ... )
        >>> poor = score_response(
        ...     response="Uh, Python is like, really really basically just stuff.",
        ...     prompt="Why Python?"
        ... )
        >>> good.overall > poor.overall
        True
    """
    return ResponseQualityScore.calculate(response, prompt, expected_length, required_keywords)


def compare_responses(response1: str, response2: str, prompt: str) -> dict[str, dict]:
    """
    Compare two LLM responses to the same prompt.

    Evaluates both responses independently for quality scoring, then
    compares them across multiple dimensions including vocabulary overlap.
    Useful for A/B testing LLM outputs or comparing different models.

    Args:
        response1: The first response to compare.
        response2: The second response to compare.
        prompt: The original prompt that generated both responses.

    Returns:
        dict[str, dict]: A dictionary containing:
            - "response1_score": ResponseQualityScore for the first response.
            - "response2_score": ResponseQualityScore for the second response.
            - "comparison": Dict of metrics with (val1, val2, diff) tuples.
            - "vocabulary_overlap": Dict with Jaccard similarity and word counts.
            - "winner": "response1" or "response2" based on overall score.

    Examples:
        Basic comparison:

        >>> result = compare_responses(
        ...     response1="Python is a language.",
        ...     response2="Python is a versatile, high-level programming language.",
        ...     prompt="What is Python?"
        ... )
        >>> result["winner"]
        'response2'

        Examining individual scores:

        >>> result = compare_responses(
        ...     response1="ML uses data.",
        ...     response2="Machine learning is a subset of AI that enables systems to learn.",
        ...     prompt="Explain machine learning."
        ... )
        >>> result["response2_score"].overall > result["response1_score"].overall
        True

        Analyzing vocabulary overlap:

        >>> result = compare_responses(
        ...     response1="Python is great for data science.",
        ...     response2="Python excels in data analysis and machine learning.",
        ...     prompt="What is Python used for?"
        ... )
        >>> result["vocabulary_overlap"]["jaccard_similarity"] > 0
        True
        >>> result["vocabulary_overlap"]["common_word_count"] > 0
        True

        Comparing similar responses:

        >>> result = compare_responses(
        ...     response1="Python is popular for web development.",
        ...     response2="Python is widely used for web applications.",
        ...     prompt="What is Python used for?"
        ... )
        >>> result["comparison"]["word_count"]  # (val1, val2, diff)
        (6, 7, 1)
    """
    analyzer = TextAnalyzer()

    score1 = score_response(response1, prompt)
    score2 = score_response(response2, prompt)

    comparison = analyzer.compare_texts(response1, response2)
    vocab_overlap = analyzer.vocabulary_overlap(response1, response2)

    return {
        "response1_score": score1,
        "response2_score": score2,
        "comparison": comparison,
        "vocabulary_overlap": {
            "jaccard_similarity": vocab_overlap[0],
            "common_word_count": len(vocab_overlap[1]),
            "unique_to_response1": len(vocab_overlap[2]),
            "unique_to_response2": len(vocab_overlap[3]),
        },
        "winner": "response1" if score1.overall > score2.overall else "response2",
    }
