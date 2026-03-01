"""
Prompt template versioning and A/B testing utilities.

This module provides a comprehensive framework for managing versioned prompt
templates with full lifecycle support including version control, changelog
tracking, rollback capabilities, A/B testing, and statistical analysis.

Overview
--------
The module is organized around three main components:

1. **Version Management** (``TemplateVersionManager``): Handles the complete
   lifecycle of prompt templates with semantic versioning, status tracking,
   and version history.

2. **A/B Testing** (``TemplateABTestRunner``, ``ABTest``): Provides a framework
   for running controlled experiments comparing different template variants
   with multiple allocation strategies.

3. **Experimentation** (``TemplateExperiment``): Enables offline comparison
   of templates using custom scoring functions and test cases.

Key Features
------------
- **Semantic Versioning**: Templates follow semver (major.minor.patch) with
  automatic version number generation.
- **Status Lifecycle**: Templates progress through DRAFT -> ACTIVE -> DEPRECATED
  -> ARCHIVED states.
- **Changelog Tracking**: Every change is recorded with timestamps, authors,
  and content diffs.
- **Rollback Support**: Easily revert to any previous version.
- **A/B Testing**: Multiple allocation strategies including random, round-robin,
  weighted, and multi-armed bandit (Thompson Sampling).
- **Statistical Analysis**: Built-in z-test for comparing conversion rates with
  confidence intervals.
- **Import/Export**: Serialize templates with full history to JSON for backup
  or migration.

Examples
--------
Basic template versioning workflow:

>>> from insideLLMs.contrib.template_versioning import (
...     TemplateVersionManager, VersionStatus
... )
>>> manager = TemplateVersionManager()
>>> # Create initial template
>>> v1 = manager.create_template(
...     name="greeting",
...     content="Hello {name}, welcome to {service}!",
...     description="Basic greeting template",
...     author="alice@example.com"
... )
>>> print(v1.version_number, v1.status)
1.0.0 VersionStatus.ACTIVE

>>> # Create a new version with improvements
>>> v2 = manager.create_version(
...     name="greeting",
...     content="Hi {name}! Welcome to {service}. How can we help?",
...     description="More friendly greeting",
...     version_level="minor"
... )
>>> print(v2.version_number, v2.status)
1.1.0 VersionStatus.DRAFT

>>> # Activate the new version
>>> manager.activate_version("greeting", v2.version_id)
>>> active = manager.get_active_version("greeting")
>>> print(active.version_number)
1.1.0

>>> # Rollback if needed
>>> manager.rollback("greeting")  # Returns to v1.0.0

A/B testing workflow:

>>> from insideLLMs.contrib.template_versioning import (
...     TemplateABTestRunner, AllocationStrategy
... )
>>> runner = TemplateABTestRunner(
...     strategy=AllocationStrategy.MULTI_ARMED_BANDIT,
...     min_samples_per_variant=100,
...     confidence_threshold=0.95
... )
>>> test = runner.create_test(
...     name="greeting_optimization",
...     variants=[
...         ("control", v1, 1.0),
...         ("treatment", v2, 1.0)
...     ]
... )
>>> test.start()
>>> # During production use
>>> variant = test.select_variant()
>>> # ... use variant.template_version.content to render prompt ...
>>> # ... get model response and evaluate ...
>>> test.record_result(variant.variant_id, score=0.85, converted=True)
>>> # After sufficient data
>>> results = test.get_results()
>>> if results.is_significant:
...     print(f"Winner: {results.winner} with {results.confidence:.1%} confidence")

Using convenience functions:

>>> from insideLLMs.contrib.template_versioning import (
...     create_template, create_version, get_active_template,
...     rollback_template, diff_template_versions
... )
>>> template = create_template(
...     name="summarizer",
...     content="Summarize the following text in {style} style: {text}",
...     author="bob@example.com"
... )
>>> updated = create_version(
...     name="summarizer",
...     content="Please provide a {style} summary of: {text}",
...     description="Improved clarity",
...     version_level="patch"
... )
>>> diff = diff_template_versions(
...     "summarizer",
...     template.version_id,
...     updated.version_id
... )
>>> print(diff["content_changed"])
True

Notes
-----
- All version IDs are SHA-256 hashes (truncated to 16 characters) combining
  template name, version number, and timestamp for uniqueness.
- The module uses dataclasses for immutable data structures with automatic
  ``to_dict()`` methods for serialization.
- Multi-armed bandit uses Thompson Sampling with Beta distributions, which
  automatically balances exploration vs exploitation.
- Statistical significance is calculated using a two-proportion z-test with
  a normal approximation for the cumulative distribution function.
- Variable extraction supports both ``{var}`` and ``{{var}}`` patterns.

See Also
--------
insideLLMs.prompt_templates : Core prompt templating functionality
insideLLMs.model_comparison : Model comparison utilities

References
----------
.. [1] Thompson, W.R. (1933). "On the Likelihood that One Unknown Probability
       Exceeds Another in View of the Evidence of Two Samples".
.. [2] Semantic Versioning 2.0.0: https://semver.org/
"""

import hashlib
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


class VersionStatus(Enum):
    """Status of a template version in its lifecycle.

    Template versions progress through a defined lifecycle:
    DRAFT -> ACTIVE -> DEPRECATED -> ARCHIVED

    Parameters
    ----------
    value : str
        The string representation of the status.

    Attributes
    ----------
    DRAFT : str
        Template is being prepared but not yet in use. New versions start here.
    ACTIVE : str
        Template is the current production version. Only one version per
        template name can be active at a time.
    DEPRECATED : str
        Template was previously active but has been superseded. Retained
        for reference and potential rollback.
    ARCHIVED : str
        Template is soft-deleted and hidden from normal listings. Cannot
        be activated without first un-archiving.

    Examples
    --------
    >>> status = VersionStatus.ACTIVE
    >>> print(status.value)
    active
    >>> status == VersionStatus.ACTIVE
    True

    >>> # Check if a version is usable
    >>> def is_usable(status: VersionStatus) -> bool:
    ...     return status in (VersionStatus.DRAFT, VersionStatus.ACTIVE)
    >>> is_usable(VersionStatus.ACTIVE)
    True
    >>> is_usable(VersionStatus.ARCHIVED)
    False

    See Also
    --------
    TemplateVersion : Uses this enum to track version status.
    TemplateVersionManager.activate_version : Changes status to ACTIVE.
    TemplateVersionManager.archive_version : Changes status to ARCHIVED.
    """

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ABTestStatus(Enum):
    """Status of an A/B test in its lifecycle.

    A/B tests progress through a defined lifecycle with state transitions:
    PENDING -> RUNNING <-> PAUSED -> COMPLETED
                    \\-> CANCELLED

    Parameters
    ----------
    value : str
        The string representation of the status.

    Attributes
    ----------
    PENDING : str
        Test has been created but not yet started. Variants can still be
        configured at this stage.
    RUNNING : str
        Test is actively collecting data. Variants are being allocated
        to users and results are being recorded.
    PAUSED : str
        Test data collection is temporarily halted. Can be resumed to
        continue from where it left off.
    COMPLETED : str
        Test has finished successfully with results available. A winner
        may have been declared if statistical significance was reached.
    CANCELLED : str
        Test was terminated early without completing. Results may be
        partial or inconclusive.

    Examples
    --------
    >>> status = ABTestStatus.RUNNING
    >>> print(status.value)
    running

    >>> # Check if test is accepting new data
    >>> def can_record_data(status: ABTestStatus) -> bool:
    ...     return status == ABTestStatus.RUNNING
    >>> can_record_data(ABTestStatus.RUNNING)
    True
    >>> can_record_data(ABTestStatus.PAUSED)
    False

    >>> # Check if test has final results
    >>> def has_final_results(status: ABTestStatus) -> bool:
    ...     return status in (ABTestStatus.COMPLETED, ABTestStatus.CANCELLED)
    >>> has_final_results(ABTestStatus.COMPLETED)
    True

    See Also
    --------
    ABTest : Uses this enum to track test status.
    ABTest.start : Transitions from PENDING to RUNNING.
    ABTest.stop : Transitions to COMPLETED and returns results.
    """

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AllocationStrategy(Enum):
    """Traffic allocation strategy for A/B tests.

    Defines how users/requests are distributed across test variants. The
    choice of strategy affects both the fairness of comparison and the
    efficiency of finding the optimal variant.

    Parameters
    ----------
    value : str
        The string representation of the strategy.

    Attributes
    ----------
    RANDOM : str
        Pure random selection with equal probability for each variant.
        Provides unbiased estimates but may take longer to find a winner.
    ROUND_ROBIN : str
        Cycles through variants in order, ensuring exactly equal
        distribution. Useful for deterministic testing scenarios.
    WEIGHTED : str
        Random selection weighted by each variant's ``weight`` attribute.
        Useful when you want to limit exposure to experimental variants.
    MULTI_ARMED_BANDIT : str
        Uses Thompson Sampling to adaptively allocate more traffic to
        better-performing variants while maintaining exploration. Optimal
        for minimizing regret during the test.

    Examples
    --------
    >>> strategy = AllocationStrategy.RANDOM
    >>> print(strategy.value)
    random

    >>> # Configure runner with specific strategy
    >>> from insideLLMs.contrib.template_versioning import TemplateABTestRunner
    >>> runner = TemplateABTestRunner(
    ...     strategy=AllocationStrategy.WEIGHTED,
    ...     min_samples_per_variant=50
    ... )

    >>> # Multi-armed bandit for production optimization
    >>> production_runner = TemplateABTestRunner(
    ...     strategy=AllocationStrategy.MULTI_ARMED_BANDIT,
    ...     confidence_threshold=0.95
    ... )

    Notes
    -----
    **Choosing a Strategy:**

    - Use ``RANDOM`` for standard A/B tests where unbiased comparison is
      the primary goal.
    - Use ``ROUND_ROBIN`` for testing environments where you need
      deterministic and reproducible allocation.
    - Use ``WEIGHTED`` when rolling out a new variant gradually (e.g.,
      10% to treatment, 90% to control).
    - Use ``MULTI_ARMED_BANDIT`` in production when you want to minimize
      the cost of serving suboptimal variants during the test.

    The multi-armed bandit strategy implements Thompson Sampling, which
    maintains Beta distributions for each variant and samples from them
    to balance exploration and exploitation.

    See Also
    --------
    ABTest.select_variant : Uses the strategy to pick variants.
    TemplateABTestRunner : Configures default strategy for tests.

    References
    ----------
    .. [1] Thompson, W.R. (1933). "On the Likelihood that One Unknown
           Probability Exceeds Another".
    """

    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"


@dataclass
class TemplateChange:
    """A single change in template version history.

    Records an immutable changelog entry for template modifications. Each
    change captures who made the modification, when, what type of change
    it was, and optionally the before/after content for diff tracking.

    Parameters
    ----------
    timestamp : str
        ISO 8601 formatted timestamp when the change occurred.
    change_type : str
        Type of change: "create" (initial creation), "update" (content
        modification), or "status_change" (lifecycle transition).
    description : str
        Human-readable description of what changed and why.
    author : str, optional
        Email or identifier of who made the change. Default is None.
    previous_content : str, optional
        Template content before the change (for updates). Default is None.
    new_content : str, optional
        Template content after the change (for creates/updates). Default is None.
    metadata : dict[str, Any], optional
        Additional context like ticket IDs, review links, etc. Default is {}.

    Attributes
    ----------
    timestamp : str
        When the change was made.
    change_type : str
        Category of the change.
    description : str
        What and why of the change.
    author : str or None
        Who made the change.
    previous_content : str or None
        Content before modification.
    new_content : str or None
        Content after modification.
    metadata : dict[str, Any]
        Extra contextual information.

    Examples
    --------
    >>> from datetime import datetime
    >>> change = TemplateChange(
    ...     timestamp=datetime.now().isoformat(),
    ...     change_type="create",
    ...     description="Initial template creation",
    ...     author="alice@example.com",
    ...     new_content="Hello {name}, welcome!"
    ... )
    >>> print(change.change_type)
    create

    >>> # Recording an update with before/after
    >>> update = TemplateChange(
    ...     timestamp=datetime.now().isoformat(),
    ...     change_type="update",
    ...     description="Made greeting more friendly",
    ...     author="bob@example.com",
    ...     previous_content="Hello {name}.",
    ...     new_content="Hi {name}! Great to see you!",
    ...     metadata={"ticket": "PROMPT-123", "reviewed_by": "carol@example.com"}
    ... )
    >>> print(update.metadata["ticket"])
    PROMPT-123

    >>> # Convert to dictionary for serialization
    >>> data = change.to_dict()
    >>> print(data["change_type"])
    create

    See Also
    --------
    TemplateVersion : Contains a list of TemplateChange entries.
    TemplateVersionManager : Creates changes automatically during operations.
    """

    timestamp: str
    change_type: str  # "create", "update", "status_change"
    description: str
    author: Optional[str] = None
    previous_content: Optional[str] = None
    new_content: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the change record to a dictionary.

        Serializes all fields to a dictionary suitable for JSON export
        or storage in a document database.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: timestamp, change_type, description,
            author, previous_content, new_content, metadata.

        Examples
        --------
        >>> from datetime import datetime
        >>> change = TemplateChange(
        ...     timestamp="2024-01-15T10:30:00",
        ...     change_type="update",
        ...     description="Fixed typo in greeting",
        ...     author="dev@example.com"
        ... )
        >>> data = change.to_dict()
        >>> print(data["change_type"])
        update
        >>> print(data["timestamp"])
        2024-01-15T10:30:00

        >>> # Serialize to JSON
        >>> import json
        >>> json_str = json.dumps(change.to_dict())
        >>> print("update" in json_str)
        True
        """
        return {
            "timestamp": self.timestamp,
            "change_type": self.change_type,
            "description": self.description,
            "author": self.author,
            "previous_content": self.previous_content,
            "new_content": self.new_content,
            "metadata": self.metadata,
        }


@dataclass
class TemplateVersion:
    """A versioned prompt template with full metadata and history.

    Represents a single version of a prompt template, including its content,
    semantic version number, lifecycle status, authorship, and changelog.
    Versions are immutable once created; modifications create new versions.

    Parameters
    ----------
    version_id : str
        Unique identifier for this version (SHA-256 hash, 16 chars).
    template_name : str
        Name of the template this version belongs to.
    content : str
        The actual prompt template text with variable placeholders.
    version_number : str
        Semantic version string (e.g., "1.2.3" for major.minor.patch).
    status : VersionStatus
        Current lifecycle status (DRAFT, ACTIVE, DEPRECATED, ARCHIVED).
    created_at : str
        ISO 8601 timestamp when this version was created.
    variables : list[str], optional
        List of variable names extracted from the template content.
        Default is [].
    description : str, optional
        Human-readable description of this version. Default is "".
    author : str, optional
        Email or identifier of who created this version. Default is None.
    changelog : list[TemplateChange], optional
        List of changes that led to or affected this version. Default is [].
    parent_version : str, optional
        Version ID of the previous version this was derived from.
        Default is None (for initial versions).
    metadata : dict[str, Any], optional
        Additional custom metadata. Default is {}.

    Attributes
    ----------
    version_id : str
        Unique identifier for this version.
    template_name : str
        Name of the parent template.
    content : str
        The template text with placeholders.
    version_number : str
        Semantic version string.
    status : VersionStatus
        Lifecycle status.
    created_at : str
        Creation timestamp.
    variables : list[str]
        Extracted variable names.
    description : str
        Version description.
    author : str or None
        Creator's identifier.
    changelog : list[TemplateChange]
        History of changes.
    parent_version : str or None
        ID of parent version.
    metadata : dict[str, Any]
        Custom metadata.
    content_hash : str
        Property: SHA-256 hash of content (12 chars).
    is_active : bool
        Property: True if status is ACTIVE.

    Examples
    --------
    >>> from datetime import datetime
    >>> version = TemplateVersion(
    ...     version_id="abc123def456",
    ...     template_name="greeting",
    ...     content="Hello {name}, welcome to {service}!",
    ...     version_number="1.0.0",
    ...     status=VersionStatus.ACTIVE,
    ...     created_at=datetime.now().isoformat(),
    ...     variables=["name", "service"],
    ...     description="Initial greeting template",
    ...     author="alice@example.com"
    ... )
    >>> print(version.template_name)
    greeting
    >>> print(version.is_active)
    True

    >>> # Check content hash for integrity verification
    >>> print(len(version.content_hash))
    12

    >>> # Access extracted variables
    >>> print(version.variables)
    ['name', 'service']

    >>> # Serialize for storage
    >>> data = version.to_dict()
    >>> print(data["version_number"])
    1.0.0

    >>> # Create a derived version (typically done via TemplateVersionManager)
    >>> v2 = TemplateVersion(
    ...     version_id="xyz789abc012",
    ...     template_name="greeting",
    ...     content="Hi {name}! Welcome to {service}. How can we help?",
    ...     version_number="1.1.0",
    ...     status=VersionStatus.DRAFT,
    ...     created_at=datetime.now().isoformat(),
    ...     variables=["name", "service"],
    ...     parent_version="abc123def456"
    ... )
    >>> print(v2.parent_version == version.version_id)
    True

    Notes
    -----
    - The ``content_hash`` property provides a quick way to verify content
      integrity or detect changes without comparing full content.
    - Variables are automatically extracted when using ``TemplateVersionManager``
      but must be provided manually when creating instances directly.
    - The semantic version format follows https://semver.org/ conventions.

    See Also
    --------
    TemplateVersionManager : Creates and manages TemplateVersion instances.
    TemplateChange : Records changes to versions.
    VersionStatus : Lifecycle status values.
    """

    version_id: str
    template_name: str
    content: str
    version_number: str  # Semantic versioning: "1.0.0"
    status: VersionStatus
    created_at: str
    variables: list[str] = field(default_factory=list)
    description: str = ""
    author: Optional[str] = None
    changelog: list[TemplateChange] = field(default_factory=list)
    parent_version: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the template content.

        Computes a truncated SHA-256 hash of the template content for
        quick integrity verification and change detection.

        Returns
        -------
        str
            First 12 characters of the SHA-256 hex digest.

        Examples
        --------
        >>> version = TemplateVersion(
        ...     version_id="test123",
        ...     template_name="test",
        ...     content="Hello {name}!",
        ...     version_number="1.0.0",
        ...     status=VersionStatus.DRAFT,
        ...     created_at="2024-01-15T10:00:00"
        ... )
        >>> hash1 = version.content_hash
        >>> print(len(hash1))
        12

        >>> # Same content produces same hash
        >>> version2 = TemplateVersion(
        ...     version_id="test456",
        ...     template_name="test",
        ...     content="Hello {name}!",
        ...     version_number="1.0.1",
        ...     status=VersionStatus.DRAFT,
        ...     created_at="2024-01-16T10:00:00"
        ... )
        >>> print(version.content_hash == version2.content_hash)
        True

        >>> # Different content produces different hash
        >>> version3 = TemplateVersion(
        ...     version_id="test789",
        ...     template_name="test",
        ...     content="Hi {name}!",
        ...     version_number="1.1.0",
        ...     status=VersionStatus.DRAFT,
        ...     created_at="2024-01-17T10:00:00"
        ... )
        >>> print(version.content_hash == version3.content_hash)
        False
        """
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]

    @property
    def is_active(self) -> bool:
        """Check if this version is the active production version.

        Returns
        -------
        bool
            True if status is VersionStatus.ACTIVE, False otherwise.

        Examples
        --------
        >>> active_version = TemplateVersion(
        ...     version_id="v1",
        ...     template_name="test",
        ...     content="Hello!",
        ...     version_number="1.0.0",
        ...     status=VersionStatus.ACTIVE,
        ...     created_at="2024-01-15T10:00:00"
        ... )
        >>> print(active_version.is_active)
        True

        >>> draft_version = TemplateVersion(
        ...     version_id="v2",
        ...     template_name="test",
        ...     content="Hello!",
        ...     version_number="1.0.1",
        ...     status=VersionStatus.DRAFT,
        ...     created_at="2024-01-16T10:00:00"
        ... )
        >>> print(draft_version.is_active)
        False
        """
        return self.status == VersionStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        """Convert the version to a dictionary for serialization.

        Creates a complete dictionary representation including computed
        properties like content_hash. Suitable for JSON serialization,
        database storage, or API responses.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all version fields plus:
            - status as string value (not enum)
            - changelog as list of dicts
            - content_hash computed property

        Examples
        --------
        >>> from datetime import datetime
        >>> version = TemplateVersion(
        ...     version_id="abc123",
        ...     template_name="greeting",
        ...     content="Hello {name}!",
        ...     version_number="1.0.0",
        ...     status=VersionStatus.ACTIVE,
        ...     created_at="2024-01-15T10:00:00",
        ...     variables=["name"],
        ...     author="dev@example.com"
        ... )
        >>> data = version.to_dict()
        >>> print(data["template_name"])
        greeting
        >>> print(data["status"])
        active
        >>> print("content_hash" in data)
        True

        >>> # Serialize to JSON
        >>> import json
        >>> json_str = json.dumps(version.to_dict())
        >>> restored = json.loads(json_str)
        >>> print(restored["version_number"])
        1.0.0

        >>> # With changelog
        >>> version.changelog.append(TemplateChange(
        ...     timestamp="2024-01-15T10:00:00",
        ...     change_type="create",
        ...     description="Initial creation"
        ... ))
        >>> data = version.to_dict()
        >>> print(len(data["changelog"]))
        1
        >>> print(data["changelog"][0]["change_type"])
        create
        """
        return {
            "version_id": self.version_id,
            "template_name": self.template_name,
            "content": self.content,
            "version_number": self.version_number,
            "status": self.status.value,
            "created_at": self.created_at,
            "variables": self.variables,
            "description": self.description,
            "author": self.author,
            "changelog": [c.to_dict() for c in self.changelog],
            "parent_version": self.parent_version,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


@dataclass
class ABVariant:
    """A variant in an A/B test with performance tracking.

    Represents one arm of an A/B test, wrapping a template version and
    tracking its performance metrics including impressions, conversions,
    and continuous scores.

    Parameters
    ----------
    variant_id : str
        Unique identifier for this variant (SHA-256 hash, 12 chars).
    name : str
        Human-readable name for the variant (e.g., "control", "treatment_a").
    template_version : TemplateVersion
        The template version being tested.
    weight : float, optional
        Allocation weight for weighted random selection. Higher weights
        get more traffic. Default is 1.0.
    impressions : int, optional
        Number of times this variant was shown. Default is 0.
    conversions : int, optional
        Number of successful conversions. Default is 0.
    total_score : float, optional
        Sum of all scores for average calculation. Default is 0.0.
    scores : list[float], optional
        Individual scores for variance calculation. Default is [].
    metadata : dict[str, Any], optional
        Additional custom metadata. Default is {}.

    Attributes
    ----------
    variant_id : str
        Unique identifier.
    name : str
        Display name.
    template_version : TemplateVersion
        Associated template.
    weight : float
        Allocation weight.
    impressions : int
        Total impressions.
    conversions : int
        Total conversions.
    total_score : float
        Cumulative score.
    scores : list[float]
        Individual score history.
    metadata : dict[str, Any]
        Custom metadata.
    conversion_rate : float
        Property: conversions / impressions.
    avg_score : float
        Property: total_score / impressions.
    score_variance : float
        Property: Sample variance of scores.
    score_std : float
        Property: Standard deviation of scores.

    Examples
    --------
    >>> template = TemplateVersion(
    ...     version_id="tmpl123",
    ...     template_name="prompt",
    ...     content="Summarize: {text}",
    ...     version_number="1.0.0",
    ...     status=VersionStatus.ACTIVE,
    ...     created_at="2024-01-15T10:00:00"
    ... )
    >>> variant = ABVariant(
    ...     variant_id="var123",
    ...     name="control",
    ...     template_version=template,
    ...     weight=1.0
    ... )
    >>> print(variant.name)
    control
    >>> print(variant.conversion_rate)
    0.0

    >>> # Record test results
    >>> variant.record_impression(score=0.85, converted=True)
    >>> variant.record_impression(score=0.72, converted=False)
    >>> variant.record_impression(score=0.91, converted=True)
    >>> print(variant.impressions)
    3
    >>> print(variant.conversions)
    2
    >>> print(f"{variant.conversion_rate:.2f}")
    0.67
    >>> print(f"{variant.avg_score:.2f}")
    0.83

    >>> # Statistical metrics
    >>> print(f"{variant.score_std:.2f}")
    0.10

    >>> # Serialize for reporting
    >>> data = variant.to_dict()
    >>> print(data["name"])
    control

    See Also
    --------
    ABTest : Contains and manages variants.
    ABTestResult : Reports variant performance.
    TemplateVersion : The template being tested.
    """

    variant_id: str
    name: str
    template_version: TemplateVersion
    weight: float = 1.0  # For weighted allocation
    impressions: int = 0
    conversions: int = 0
    total_score: float = 0.0
    scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def conversion_rate(self) -> float:
        """Calculate the conversion rate for this variant.

        Computes the ratio of conversions to impressions. Returns 0.0
        if no impressions have been recorded yet.

        Returns
        -------
        float
            Conversion rate as a decimal (0.0 to 1.0).

        Examples
        --------
        >>> variant = ABVariant(
        ...     variant_id="v1",
        ...     name="test",
        ...     template_version=template
        ... )
        >>> print(variant.conversion_rate)
        0.0

        >>> variant.impressions = 100
        >>> variant.conversions = 25
        >>> print(variant.conversion_rate)
        0.25

        >>> # As percentage
        >>> print(f"{variant.conversion_rate:.1%}")
        25.0%
        """
        return self.conversions / self.impressions if self.impressions > 0 else 0.0

    @property
    def avg_score(self) -> float:
        """Calculate the average score across all impressions.

        Computes the mean score from cumulative totals. Returns 0.0
        if no impressions have been recorded.

        Returns
        -------
        float
            Mean score value.

        Examples
        --------
        >>> variant = ABVariant(
        ...     variant_id="v1",
        ...     name="test",
        ...     template_version=template
        ... )
        >>> variant.record_impression(0.80)
        >>> variant.record_impression(0.90)
        >>> variant.record_impression(0.85)
        >>> print(variant.avg_score)
        0.85
        """
        return self.total_score / self.impressions if self.impressions > 0 else 0.0

    @property
    def score_variance(self) -> float:
        """Calculate the sample variance of scores.

        Uses Bessel's correction (n-1 denominator) for unbiased
        estimation. Returns 0.0 if fewer than 2 scores recorded.

        Returns
        -------
        float
            Sample variance of scores.

        Examples
        --------
        >>> variant = ABVariant(
        ...     variant_id="v1",
        ...     name="test",
        ...     template_version=template
        ... )
        >>> variant.record_impression(0.80)
        >>> print(variant.score_variance)  # Need at least 2 samples
        0.0

        >>> variant.record_impression(0.90)
        >>> variant.record_impression(0.70)
        >>> print(f"{variant.score_variance:.4f}")
        0.0100
        """
        if len(self.scores) < 2:
            return 0.0
        mean = self.avg_score
        return sum((s - mean) ** 2 for s in self.scores) / (len(self.scores) - 1)

    @property
    def score_std(self) -> float:
        """Calculate the standard deviation of scores.

        Square root of the sample variance. Returns 0.0 if fewer
        than 2 scores recorded.

        Returns
        -------
        float
            Standard deviation of scores.

        Examples
        --------
        >>> variant = ABVariant(
        ...     variant_id="v1",
        ...     name="test",
        ...     template_version=template
        ... )
        >>> variant.record_impression(0.80)
        >>> variant.record_impression(0.90)
        >>> variant.record_impression(0.70)
        >>> print(f"{variant.score_std:.4f}")
        0.1000

        >>> # Confidence interval approximation (95%)
        >>> margin = 1.96 * variant.score_std / (variant.impressions ** 0.5)
        >>> print(f"{variant.avg_score:.2f} +/- {margin:.2f}")
        0.80 +/- 0.11
        """
        return self.score_variance**0.5

    def record_impression(self, score: float, converted: bool = False) -> None:
        """Record an impression with its score and conversion status.

        Updates all running statistics for this variant. Should be called
        after each time the variant's template is used and evaluated.

        Parameters
        ----------
        score : float
            Quality score for this impression (typically 0.0 to 1.0).
        converted : bool, optional
            Whether this impression resulted in a conversion.
            Default is False.

        Examples
        --------
        >>> variant = ABVariant(
        ...     variant_id="v1",
        ...     name="test",
        ...     template_version=template
        ... )
        >>> # Record a successful interaction
        >>> variant.record_impression(score=0.92, converted=True)
        >>> print(variant.impressions)
        1
        >>> print(variant.conversions)
        1

        >>> # Record an unsuccessful interaction
        >>> variant.record_impression(score=0.45, converted=False)
        >>> print(variant.impressions)
        2
        >>> print(variant.conversions)
        1

        >>> # Batch recording
        >>> scores = [0.78, 0.85, 0.91, 0.67, 0.88]
        >>> conversions = [True, True, True, False, True]
        >>> for s, c in zip(scores, conversions):
        ...     variant.record_impression(s, c)
        >>> print(variant.impressions)
        7
        """
        self.impressions += 1
        self.total_score += score
        self.scores.append(score)
        if converted:
            self.conversions += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert variant to dictionary for serialization.

        Creates a dictionary representation with computed statistics
        rounded to 4 decimal places for readability.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - variant_id, name, template_version_id, weight
            - impressions, conversions
            - conversion_rate, avg_score, score_std (rounded)
            - metadata

        Examples
        --------
        >>> variant = ABVariant(
        ...     variant_id="v123",
        ...     name="treatment",
        ...     template_version=template,
        ...     weight=0.5
        ... )
        >>> variant.record_impression(0.85, True)
        >>> variant.record_impression(0.75, False)
        >>> data = variant.to_dict()
        >>> print(data["name"])
        treatment
        >>> print(data["conversion_rate"])
        0.5
        >>> print(data["avg_score"])
        0.8

        >>> # Serialize to JSON
        >>> import json
        >>> json_str = json.dumps(variant.to_dict())
        >>> print("treatment" in json_str)
        True
        """
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "template_version_id": self.template_version.version_id,
            "weight": self.weight,
            "impressions": self.impressions,
            "conversions": self.conversions,
            "conversion_rate": round(self.conversion_rate, 4),
            "avg_score": round(self.avg_score, 4),
            "score_std": round(self.score_std, 4),
            "metadata": self.metadata,
        }


@dataclass
class ABTestResult:
    """Results and analysis of an A/B test.

    Encapsulates the complete outcome of an A/B test including variant
    performance, statistical significance, and actionable recommendations.
    Generated by ``ABTest.get_results()`` or ``ABTest.stop()``.

    Parameters
    ----------
    test_id : str
        Unique identifier of the test.
    test_name : str
        Human-readable name of the test.
    status : ABTestStatus
        Current status of the test when results were generated.
    variants : list[ABVariant]
        List of all variants with their performance metrics.
    winner : str or None
        Variant ID of the winning variant if statistically significant,
        None otherwise.
    confidence : float
        Statistical confidence level (0.0 to 1.0) for the winner.
    total_impressions : int
        Total impressions across all variants.
    duration_seconds : float
        Total test duration from start to end (or current time).
    started_at : str
        ISO 8601 timestamp when the test started.
    ended_at : str or None
        ISO 8601 timestamp when the test ended, None if ongoing.
    recommendations : list[str], optional
        Actionable recommendations based on results. Default is [].
    metadata : dict[str, Any], optional
        Additional custom metadata. Default is {}.

    Attributes
    ----------
    test_id : str
        Test identifier.
    test_name : str
        Test display name.
    status : ABTestStatus
        Test status.
    variants : list[ABVariant]
        Variant performance data.
    winner : str or None
        Winning variant ID or None.
    confidence : float
        Statistical confidence.
    total_impressions : int
        Total sample size.
    duration_seconds : float
        Test duration.
    started_at : str
        Start timestamp.
    ended_at : str or None
        End timestamp.
    recommendations : list[str]
        Suggested actions.
    metadata : dict[str, Any]
        Custom metadata.
    is_significant : bool
        Property: True if confidence >= 0.95.

    Examples
    --------
    >>> # Results are typically generated by ABTest.get_results()
    >>> result = ABTestResult(
    ...     test_id="test123",
    ...     test_name="Greeting Optimization",
    ...     status=ABTestStatus.COMPLETED,
    ...     variants=variants,  # List of ABVariant objects
    ...     winner="variant_a_id",
    ...     confidence=0.97,
    ...     total_impressions=500,
    ...     duration_seconds=86400.0,
    ...     started_at="2024-01-15T10:00:00",
    ...     ended_at="2024-01-16T10:00:00",
    ...     recommendations=[
    ...         "'Treatment A' is the winner with 97.0% confidence",
    ...         "Consider promoting 'Treatment A' to production"
    ...     ]
    ... )

    >>> # Check if results are actionable
    >>> if result.is_significant:
    ...     print(f"Winner: {result.winner}")
    ...     for rec in result.recommendations:
    ...         print(f"  - {rec}")
    Winner: variant_a_id
      - 'Treatment A' is the winner with 97.0% confidence
      - Consider promoting 'Treatment A' to production

    >>> # Results that need more data
    >>> inconclusive = ABTestResult(
    ...     test_id="test456",
    ...     test_name="Ongoing Test",
    ...     status=ABTestStatus.RUNNING,
    ...     variants=variants,
    ...     winner=None,
    ...     confidence=0.78,
    ...     total_impressions=50,
    ...     duration_seconds=3600.0,
    ...     started_at="2024-01-15T10:00:00",
    ...     ended_at=None,
    ...     recommendations=["Need at least 50 more samples for statistical significance"]
    ... )
    >>> print(inconclusive.is_significant)
    False

    >>> # Serialize for API response or dashboard
    >>> data = result.to_dict()
    >>> print(data["is_significant"])
    True
    >>> print(data["confidence"])
    0.97

    See Also
    --------
    ABTest.get_results : Generates ABTestResult instances.
    ABTest.stop : Stops test and returns final results.
    ABVariant : Individual variant performance data.
    """

    test_id: str
    test_name: str
    status: ABTestStatus
    variants: list[ABVariant]
    winner: Optional[str]  # variant_id of winner
    confidence: float  # Statistical confidence level
    total_impressions: int
    duration_seconds: float
    started_at: str
    ended_at: Optional[str]
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        """Check if results are statistically significant.

        Returns True if confidence level meets or exceeds the standard
        threshold of 95%, indicating the results are unlikely to be
        due to random chance.

        Returns
        -------
        bool
            True if confidence >= 0.95, False otherwise.

        Examples
        --------
        >>> result = ABTestResult(
        ...     test_id="t1",
        ...     test_name="Test",
        ...     status=ABTestStatus.COMPLETED,
        ...     variants=[],
        ...     winner="v1",
        ...     confidence=0.97,
        ...     total_impressions=200,
        ...     duration_seconds=3600,
        ...     started_at="2024-01-15T10:00:00",
        ...     ended_at="2024-01-15T11:00:00"
        ... )
        >>> print(result.is_significant)
        True

        >>> result.confidence = 0.89
        >>> print(result.is_significant)
        False

        >>> # Use in decision making
        >>> if result.is_significant and result.winner:
        ...     print("Ready to promote winner to production")
        ... else:
        ...     print("Continue collecting data")
        Continue collecting data
        """
        return self.confidence >= 0.95

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary for serialization.

        Creates a complete dictionary representation suitable for JSON
        serialization, API responses, or dashboard display.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all result fields with:
            - status as string value
            - variants as list of dicts
            - confidence and duration_seconds rounded
            - is_significant computed property

        Examples
        --------
        >>> result = ABTestResult(
        ...     test_id="t1",
        ...     test_name="Optimization Test",
        ...     status=ABTestStatus.COMPLETED,
        ...     variants=[],
        ...     winner="v1",
        ...     confidence=0.9567,
        ...     total_impressions=500,
        ...     duration_seconds=86400.567,
        ...     started_at="2024-01-15T10:00:00",
        ...     ended_at="2024-01-16T10:00:00",
        ...     recommendations=["Promote winner"]
        ... )
        >>> data = result.to_dict()
        >>> print(data["test_name"])
        Optimization Test
        >>> print(data["confidence"])
        0.9567
        >>> print(data["duration_seconds"])
        86400.57
        >>> print(data["is_significant"])
        True

        >>> # Serialize to JSON for API
        >>> import json
        >>> json_str = json.dumps(result.to_dict())
        >>> print("COMPLETED" in json_str or "completed" in json_str)
        True
        """
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "status": self.status.value,
            "variants": [v.to_dict() for v in self.variants],
            "winner": self.winner,
            "confidence": round(self.confidence, 4),
            "is_significant": self.is_significant,
            "total_impressions": self.total_impressions,
            "duration_seconds": round(self.duration_seconds, 2),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


class TemplateVersionManager:
    """Manages versioned prompt templates with full lifecycle support.

    Provides complete version control for prompt templates including creation,
    updates, activation, deprecation, archival, and rollback. All changes are
    tracked in a changelog with timestamps and authorship.

    Parameters
    ----------
    None

    Attributes
    ----------
    templates : dict[str, dict[str, TemplateVersion]]
        Nested dictionary mapping template name -> version_id -> version.
    active_versions : dict[str, str]
        Dictionary mapping template name -> active version_id.

    Examples
    --------
    Basic usage:

    >>> manager = TemplateVersionManager()
    >>> # Create initial template (automatically becomes active)
    >>> v1 = manager.create_template(
    ...     name="greeting",
    ...     content="Hello {name}!",
    ...     description="Basic greeting",
    ...     author="alice@example.com"
    ... )
    >>> print(v1.version_number)
    1.0.0
    >>> print(v1.is_active)
    True

    >>> # Create a new version (starts as draft)
    >>> v2 = manager.create_version(
    ...     name="greeting",
    ...     content="Hi {name}! Welcome!",
    ...     description="More friendly greeting",
    ...     version_level="minor"
    ... )
    >>> print(v2.version_number)
    1.1.0
    >>> print(v2.status)
    VersionStatus.DRAFT

    >>> # Activate the new version
    >>> manager.activate_version("greeting", v2.version_id)
    >>> active = manager.get_active_version("greeting")
    >>> print(active.version_number)
    1.1.0

    >>> # Check that old version was deprecated
    >>> old = manager.get_version("greeting", v1.version_id)
    >>> print(old.status)
    VersionStatus.DEPRECATED

    Rollback example:

    >>> # Rollback to previous version
    >>> rolled_back = manager.rollback("greeting")
    >>> print(rolled_back.version_number)
    1.0.0
    >>> print(rolled_back.is_active)
    True

    >>> # Or rollback to specific version
    >>> manager.rollback("greeting", to_version_id=v2.version_id)

    Version comparison:

    >>> diff = manager.diff_versions("greeting", v1.version_id, v2.version_id)
    >>> print(diff["content_changed"])
    True
    >>> print(diff["character_diff"])  # Positive = v2 is longer
    10

    Export and import:

    >>> exported = manager.export_template("greeting")
    >>> import json
    >>> json_str = json.dumps(exported)
    >>> # ... save to file or database ...
    >>> # Later, on another system:
    >>> new_manager = TemplateVersionManager()
    >>> imported = new_manager.import_template(json.loads(json_str))
    >>> print(len(imported))  # Number of versions imported
    2

    Notes
    -----
    - Version IDs are SHA-256 hashes (16 chars) combining name, version, and
      timestamp, ensuring uniqueness even for identical content.
    - Only one version per template can be ACTIVE at a time.
    - Variables are automatically extracted from template content using the
      pattern ``{variable}`` or ``{{variable}}``.
    - The manager is in-memory by default; use export/import for persistence.

    See Also
    --------
    TemplateVersion : The version objects managed by this class.
    VersionStatus : Lifecycle status values.
    TemplateChange : Changelog entries.
    """

    def __init__(self):
        """Initialize an empty version manager.

        Creates a new manager with no templates. Templates are added using
        ``create_template()`` or ``import_template()``.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> print(len(manager.templates))
        0
        >>> print(manager.list_templates())
        []
        """
        self.templates: dict[str, dict[str, TemplateVersion]] = {}  # name -> version_id -> version
        self.active_versions: dict[str, str] = {}  # name -> active version_id

    def _generate_version_id(self, name: str, version: str) -> str:
        """Generate a unique version ID.

        Creates a deterministic but unique ID by hashing the template name,
        version number, and current timestamp.

        Parameters
        ----------
        name : str
            Template name.
        version : str
            Version number string.

        Returns
        -------
        str
            16-character hexadecimal ID.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> id1 = manager._generate_version_id("test", "1.0.0")
        >>> print(len(id1))
        16
        >>> # IDs are unique even for same name/version (due to timestamp)
        >>> import time; time.sleep(0.001)
        >>> id2 = manager._generate_version_id("test", "1.0.0")
        >>> print(id1 != id2)
        True
        """
        timestamp = datetime.now().isoformat()
        content = f"{name}:{version}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _extract_variables(self, content: str) -> list[str]:
        """Extract variable names from template content.

        Finds all variable placeholders in the template using regex pattern
        matching. Supports both single-brace ``{var}`` and double-brace
        ``{{var}}`` syntax.

        Parameters
        ----------
        content : str
            Template content to parse.

        Returns
        -------
        list[str]
            Unique variable names found in the template.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> vars = manager._extract_variables("Hello {name}, welcome to {service}!")
        >>> print(sorted(vars))
        ['name', 'service']

        >>> # Double braces also work
        >>> vars = manager._extract_variables("{{name}} at {{company}}")
        >>> print(sorted(vars))
        ['company', 'name']

        >>> # No duplicates in output
        >>> vars = manager._extract_variables("{x} and {x} again")
        >>> print(vars)
        ['x']
        """
        import re

        # Match {variable} and {{variable}} patterns
        pattern = r"\{+(\w+)\}+"
        matches = re.findall(pattern, content)
        return list(set(matches))

    def _parse_version(self, version: str) -> tuple[int, int, int]:
        """Parse a semantic version string into components.

        Extracts major, minor, and patch numbers from a version string.
        Missing components default to 0.

        Parameters
        ----------
        version : str
            Version string like "1.2.3" or "2.0".

        Returns
        -------
        tuple[int, int, int]
            Tuple of (major, minor, patch) version numbers.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> manager._parse_version("1.2.3")
        (1, 2, 3)
        >>> manager._parse_version("2.0")
        (2, 0, 0)
        >>> manager._parse_version("3")
        (3, 0, 0)
        """
        parts = version.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch)

    def _increment_version(self, current: str, level: str = "patch") -> str:
        """Increment a version number at the specified level.

        Follows semantic versioning rules: incrementing major resets minor
        and patch to 0; incrementing minor resets patch to 0.

        Parameters
        ----------
        current : str
            Current version string.
        level : str, optional
            Which component to increment: "major", "minor", or "patch".
            Default is "patch".

        Returns
        -------
        str
            New version string.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> manager._increment_version("1.2.3", "patch")
        '1.2.4'
        >>> manager._increment_version("1.2.3", "minor")
        '1.3.0'
        >>> manager._increment_version("1.2.3", "major")
        '2.0.0'
        """
        major, minor, patch = self._parse_version(current)
        if level == "major":
            return f"{major + 1}.0.0"
        elif level == "minor":
            return f"{major}.{minor + 1}.0"
        else:
            return f"{major}.{minor}.{patch + 1}"

    def create_template(
        self,
        name: str,
        content: str,
        description: str = "",
        author: Optional[str] = None,
        initial_version: str = "1.0.0",
        metadata: Optional[dict[str, Any]] = None,
    ) -> TemplateVersion:
        """Create a new template with an initial version.

        Creates a brand new template that doesn't exist yet. The initial
        version is automatically set to ACTIVE status.

        Parameters
        ----------
        name : str
            Unique name for the template.
        content : str
            Template content with variable placeholders.
        description : str, optional
            Human-readable description. Default is "".
        author : str, optional
            Email or identifier of creator. Default is None.
        initial_version : str, optional
            Starting version number. Default is "1.0.0".
        metadata : dict[str, Any], optional
            Additional custom metadata. Default is None.

        Returns
        -------
        TemplateVersion
            The created version with ACTIVE status.

        Raises
        ------
        ValueError
            If a template with this name already exists.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> version = manager.create_template(
        ...     name="summarizer",
        ...     content="Summarize the following in {style} style:\\n{text}",
        ...     description="Text summarization prompt",
        ...     author="alice@example.com"
        ... )
        >>> print(version.template_name)
        summarizer
        >>> print(version.version_number)
        1.0.0
        >>> print(version.is_active)
        True
        >>> print(version.variables)
        ['style', 'text']

        >>> # Custom initial version
        >>> v2 = manager.create_template(
        ...     name="translator",
        ...     content="Translate to {language}: {text}",
        ...     initial_version="0.1.0"  # Pre-release version
        ... )
        >>> print(v2.version_number)
        0.1.0

        >>> # With metadata
        >>> v3 = manager.create_template(
        ...     name="classifier",
        ...     content="Classify: {text}",
        ...     metadata={"model": "gpt-4", "category": "nlp"}
        ... )
        >>> print(v3.metadata["category"])
        nlp

        >>> # Duplicate name raises error
        >>> try:
        ...     manager.create_template(name="summarizer", content="...")
        ... except ValueError as e:
        ...     print("Error:", "already exists" in str(e))
        Error: True
        """
        if name in self.templates and len(self.templates[name]) > 0:
            raise ValueError(f"Template '{name}' already exists. Use create_version instead.")

        version_id = self._generate_version_id(name, initial_version)
        timestamp = datetime.now().isoformat()

        change = TemplateChange(
            timestamp=timestamp,
            change_type="create",
            description="Initial template creation",
            author=author,
            new_content=content,
        )

        version = TemplateVersion(
            version_id=version_id,
            template_name=name,
            content=content,
            version_number=initial_version,
            status=VersionStatus.ACTIVE,
            created_at=timestamp,
            variables=self._extract_variables(content),
            description=description,
            author=author,
            changelog=[change],
            metadata=metadata or {},
        )

        if name not in self.templates:
            self.templates[name] = {}
        self.templates[name][version_id] = version
        self.active_versions[name] = version_id

        return version

    def create_version(
        self,
        name: str,
        content: str,
        description: str = "",
        author: Optional[str] = None,
        version_level: str = "patch",  # "major", "minor", "patch"
        explicit_version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TemplateVersion:
        """Create a new version of an existing template.

        Creates a new version derived from the current active version. The new
        version starts in DRAFT status and must be explicitly activated. If the
        template doesn't exist, creates it as a new template.

        Parameters
        ----------
        name : str
            Name of the existing template.
        content : str
            New template content with variable placeholders.
        description : str, optional
            Description of changes in this version. Default is "".
        author : str, optional
            Email or identifier of the author. Default is None.
        version_level : str, optional
            How to increment version: "major", "minor", or "patch".
            Default is "patch".
        explicit_version : str, optional
            Explicit version number to use instead of auto-incrementing.
            Default is None.
        metadata : dict[str, Any], optional
            Additional custom metadata. Default is None.

        Returns
        -------
        TemplateVersion
            The new version with DRAFT status (or ACTIVE if template is new).

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> v1 = manager.create_template("prompt", "Hello {name}!")

        >>> # Patch version (bug fix, minor change)
        >>> v1_1 = manager.create_version(
        ...     name="prompt",
        ...     content="Hello {name}.",
        ...     description="Changed exclamation to period",
        ...     version_level="patch"
        ... )
        >>> print(v1_1.version_number)
        1.0.1

        >>> # Minor version (new feature, backwards compatible)
        >>> v1_2 = manager.create_version(
        ...     name="prompt",
        ...     content="Hello {name}! Welcome to {service}.",
        ...     description="Added service variable",
        ...     version_level="minor"
        ... )
        >>> print(v1_2.version_number)
        1.1.0

        >>> # Major version (breaking change)
        >>> v2 = manager.create_version(
        ...     name="prompt",
        ...     content="Greet the user {user} for {service}",
        ...     description="Complete redesign",
        ...     version_level="major"
        ... )
        >>> print(v2.version_number)
        2.0.0

        >>> # Explicit version number
        >>> v3 = manager.create_version(
        ...     name="prompt",
        ...     content="...",
        ...     explicit_version="3.0.0-beta"
        ... )
        >>> print(v3.version_number)
        3.0.0-beta

        >>> # New versions start as drafts
        >>> print(v2.status)
        VersionStatus.DRAFT

        >>> # Parent version is tracked
        >>> print(v2.parent_version == v1.version_id or v2.parent_version is not None)
        True
        """
        if name not in self.templates or len(self.templates[name]) == 0:
            return self.create_template(name, content, description, author, metadata=metadata)

        # Get current active version
        active_id = self.active_versions.get(name)
        if active_id and active_id in self.templates[name]:
            current = self.templates[name][active_id]
        else:
            # Get latest version
            versions = list(self.templates[name].values())
            current = max(versions, key=lambda v: self._parse_version(v.version_number))

        # Determine new version number
        if explicit_version:
            new_version = explicit_version
        else:
            new_version = self._increment_version(current.version_number, version_level)

        version_id = self._generate_version_id(name, new_version)
        timestamp = datetime.now().isoformat()

        change = TemplateChange(
            timestamp=timestamp,
            change_type="update",
            description=description or f"Updated to version {new_version}",
            author=author,
            previous_content=current.content,
            new_content=content,
        )

        version = TemplateVersion(
            version_id=version_id,
            template_name=name,
            content=content,
            version_number=new_version,
            status=VersionStatus.DRAFT,
            created_at=timestamp,
            variables=self._extract_variables(content),
            description=description,
            author=author,
            changelog=[change],
            parent_version=current.version_id,
            metadata=metadata or {},
        )

        self.templates[name][version_id] = version
        return version

    def activate_version(
        self, name: str, version_id: str, author: Optional[str] = None
    ) -> TemplateVersion:
        """Activate a specific version of a template.

        Makes the specified version the active production version. The
        previously active version is automatically deprecated.

        Parameters
        ----------
        name : str
            Name of the template.
        version_id : str
            ID of the version to activate.
        author : str, optional
            Who is performing the activation. Default is None.

        Returns
        -------
        TemplateVersion
            The activated version with ACTIVE status.

        Raises
        ------
        ValueError
            If the template or version doesn't exist.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> v1 = manager.create_template("greeting", "Hello!")
        >>> v2 = manager.create_version("greeting", "Hi there!")

        >>> # v1 is initially active
        >>> print(v1.is_active)
        True

        >>> # Activate v2
        >>> activated = manager.activate_version("greeting", v2.version_id)
        >>> print(activated.is_active)
        True

        >>> # v1 is now deprecated
        >>> print(manager.get_version("greeting", v1.version_id).status)
        VersionStatus.DEPRECATED

        >>> # Changelog is updated
        >>> print(activated.changelog[-1].change_type)
        status_change
        """
        if name not in self.templates or version_id not in self.templates[name]:
            raise ValueError(f"Version {version_id} not found for template '{name}'")

        version = self.templates[name][version_id]

        # Deprecate currently active version
        if name in self.active_versions:
            old_id = self.active_versions[name]
            if old_id in self.templates[name]:
                old_version = self.templates[name][old_id]
                old_version.status = VersionStatus.DEPRECATED
                old_version.changelog.append(
                    TemplateChange(
                        timestamp=datetime.now().isoformat(),
                        change_type="status_change",
                        description=f"Deprecated in favor of version {version.version_number}",
                        author=author,
                    )
                )

        # Activate new version
        version.status = VersionStatus.ACTIVE
        version.changelog.append(
            TemplateChange(
                timestamp=datetime.now().isoformat(),
                change_type="status_change",
                description="Activated as current version",
                author=author,
            )
        )
        self.active_versions[name] = version_id

        return version

    def get_active_version(self, name: str) -> Optional[TemplateVersion]:
        """Get the active version of a template.

        Returns the currently active (production) version of the named
        template, or None if no active version exists.

        Parameters
        ----------
        name : str
            Name of the template.

        Returns
        -------
        TemplateVersion or None
            The active version, or None if not found.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> manager.create_template("greeting", "Hello!")
        >>> active = manager.get_active_version("greeting")
        >>> print(active.version_number)
        1.0.0

        >>> # Non-existent template returns None
        >>> result = manager.get_active_version("nonexistent")
        >>> print(result is None)
        True
        """
        if name not in self.active_versions:
            return None
        version_id = self.active_versions[name]
        return self.templates.get(name, {}).get(version_id)

    def get_version(self, name: str, version_id: str) -> Optional[TemplateVersion]:
        """Get a specific version of a template by ID.

        Parameters
        ----------
        name : str
            Name of the template.
        version_id : str
            Unique ID of the version.

        Returns
        -------
        TemplateVersion or None
            The requested version, or None if not found.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> v1 = manager.create_template("test", "Hello!")
        >>> retrieved = manager.get_version("test", v1.version_id)
        >>> print(retrieved.content)
        Hello!

        >>> # Invalid ID returns None
        >>> print(manager.get_version("test", "invalid") is None)
        True
        """
        return self.templates.get(name, {}).get(version_id)

    def get_version_by_number(self, name: str, version_number: str) -> Optional[TemplateVersion]:
        """Get a version by its semantic version number.

        Looks up a version using the human-readable version number instead
        of the internal version ID.

        Parameters
        ----------
        name : str
            Name of the template.
        version_number : str
            Semantic version string like "1.2.3".

        Returns
        -------
        TemplateVersion or None
            The version with that number, or None if not found.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> manager.create_template("test", "v1 content")
        >>> manager.create_version("test", "v2 content", version_level="minor")
        >>> v = manager.get_version_by_number("test", "1.1.0")
        >>> print(v.content)
        v2 content

        >>> # Non-existent version returns None
        >>> print(manager.get_version_by_number("test", "9.9.9") is None)
        True
        """
        if name not in self.templates:
            return None
        for version in self.templates[name].values():
            if version.version_number == version_number:
                return version
        return None

    def list_versions(self, name: str, include_archived: bool = False) -> list[TemplateVersion]:
        """List all versions of a template.

        Returns versions sorted by version number in descending order
        (newest first). Archived versions are excluded by default.

        Parameters
        ----------
        name : str
            Name of the template.
        include_archived : bool, optional
            Whether to include archived versions. Default is False.

        Returns
        -------
        list[TemplateVersion]
            List of versions, newest first. Empty list if template not found.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> manager.create_template("test", "v1")
        >>> manager.create_version("test", "v2", version_level="minor")
        >>> manager.create_version("test", "v3", version_level="major")
        >>> versions = manager.list_versions("test")
        >>> [v.version_number for v in versions]
        ['2.0.0', '1.1.0', '1.0.0']

        >>> # Empty list for non-existent template
        >>> print(manager.list_versions("nonexistent"))
        []

        >>> # Archive a version and check filtering
        >>> v1 = manager.get_version_by_number("test", "1.0.0")
        >>> manager.archive_version("test", v1.version_id)
        >>> len(manager.list_versions("test"))  # Excludes archived
        2
        >>> len(manager.list_versions("test", include_archived=True))
        3
        """
        if name not in self.templates:
            return []
        versions = list(self.templates[name].values())
        if not include_archived:
            versions = [v for v in versions if v.status != VersionStatus.ARCHIVED]
        return sorted(versions, key=lambda v: self._parse_version(v.version_number), reverse=True)

    def list_templates(self) -> list[str]:
        """List all template names.

        Returns
        -------
        list[str]
            Names of all templates in the manager.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> manager.create_template("greeting", "Hello!")
        >>> manager.create_template("farewell", "Goodbye!")
        >>> sorted(manager.list_templates())
        ['farewell', 'greeting']
        """
        return list(self.templates.keys())

    def rollback(
        self, name: str, to_version_id: Optional[str] = None, author: Optional[str] = None
    ) -> TemplateVersion:
        """Rollback to a previous version.

        Activates either the parent of the current active version (default)
        or a specific version by ID. The current active version becomes
        deprecated.

        Parameters
        ----------
        name : str
            Name of the template.
        to_version_id : str, optional
            Specific version ID to rollback to. If None, rolls back to
            the parent version. Default is None.
        author : str, optional
            Who is performing the rollback. Default is None.

        Returns
        -------
        TemplateVersion
            The newly activated version.

        Raises
        ------
        ValueError
            If template not found, version not found, or no parent version
            exists for automatic rollback.

        Examples
        --------
        >>> manager = TemplateVersionManager()
        >>> v1 = manager.create_template("test", "Original")
        >>> v2 = manager.create_version("test", "Updated")
        >>> manager.activate_version("test", v2.version_id)

        >>> # Rollback to parent (v1)
        >>> rolled = manager.rollback("test")
        >>> print(rolled.content)
        Original

        >>> # Now v2 is deprecated
        >>> print(manager.get_version("test", v2.version_id).status)
        VersionStatus.DEPRECATED

        >>> # Rollback to specific version
        >>> manager.activate_version("test", v2.version_id)
        >>> rolled = manager.rollback("test", to_version_id=v1.version_id)
        >>> print(rolled.content)
        Original

        >>> # Error if no parent
        >>> try:
        ...     only = manager.create_template("single", "Only version")
        ...     manager.rollback("single")  # v1 has no parent
        ... except ValueError as e:
        ...     print("No parent" in str(e))
        True
        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")

        if to_version_id:
            # Rollback to specific version
            if to_version_id not in self.templates[name]:
                raise ValueError(f"Version {to_version_id} not found")
            return self.activate_version(name, to_version_id, author)
        else:
            # Rollback to previous version
            current_id = self.active_versions.get(name)
            if not current_id:
                raise ValueError(f"No active version for template '{name}'")

            current = self.templates[name][current_id]
            if not current.parent_version:
                raise ValueError("No parent version to rollback to")

            return self.activate_version(name, current.parent_version, author)

    def archive_version(
        self, name: str, version_id: str, author: Optional[str] = None
    ) -> TemplateVersion:
        """Archive a version (soft delete)."""
        if name not in self.templates or version_id not in self.templates[name]:
            raise ValueError(f"Version {version_id} not found for template '{name}'")

        version = self.templates[name][version_id]

        if version.is_active:
            raise ValueError("Cannot archive active version. Activate another version first.")

        version.status = VersionStatus.ARCHIVED
        version.changelog.append(
            TemplateChange(
                timestamp=datetime.now().isoformat(),
                change_type="status_change",
                description="Archived",
                author=author,
            )
        )

        return version

    def diff_versions(self, name: str, version_id_a: str, version_id_b: str) -> dict[str, Any]:
        """Compare two versions of a template."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")

        version_a = self.templates[name].get(version_id_a)
        version_b = self.templates[name].get(version_id_b)

        if not version_a or not version_b:
            raise ValueError("One or both versions not found")

        # Simple diff: character-level changes
        content_changed = version_a.content != version_b.content
        variables_added = set(version_b.variables) - set(version_a.variables)
        variables_removed = set(version_a.variables) - set(version_b.variables)

        return {
            "version_a": {
                "id": version_a.version_id,
                "number": version_a.version_number,
                "content_hash": version_a.content_hash,
            },
            "version_b": {
                "id": version_b.version_id,
                "number": version_b.version_number,
                "content_hash": version_b.content_hash,
            },
            "content_changed": content_changed,
            "variables_added": list(variables_added),
            "variables_removed": list(variables_removed),
            "character_diff": len(version_b.content) - len(version_a.content),
        }

    def export_template(self, name: str, version_id: Optional[str] = None) -> dict[str, Any]:
        """Export template and its history as JSON."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")

        if version_id:
            versions = [self.templates[name].get(version_id)]
            if not versions[0]:
                raise ValueError(f"Version {version_id} not found")
        else:
            versions = self.list_versions(name, include_archived=True)

        return {
            "template_name": name,
            "active_version_id": self.active_versions.get(name),
            "versions": [v.to_dict() for v in versions],
            "exported_at": datetime.now().isoformat(),
        }

    def import_template(
        self, data: dict[str, Any], overwrite: bool = False
    ) -> list[TemplateVersion]:
        """Import template from exported JSON."""
        name = data["template_name"]

        if name in self.templates and not overwrite:
            raise ValueError(f"Template '{name}' already exists. Use overwrite=True to replace.")

        if overwrite and name in self.templates:
            del self.templates[name]
            if name in self.active_versions:
                del self.active_versions[name]

        self.templates[name] = {}
        imported = []

        for v_data in data["versions"]:
            version = TemplateVersion(
                version_id=v_data["version_id"],
                template_name=name,
                content=v_data["content"],
                version_number=v_data["version_number"],
                status=VersionStatus(v_data["status"]),
                created_at=v_data["created_at"],
                variables=v_data.get("variables", []),
                description=v_data.get("description", ""),
                author=v_data.get("author"),
                changelog=[TemplateChange(**c) for c in v_data.get("changelog", [])],
                parent_version=v_data.get("parent_version"),
                metadata=v_data.get("metadata", {}),
            )
            self.templates[name][version.version_id] = version
            imported.append(version)

        if data.get("active_version_id"):
            self.active_versions[name] = data["active_version_id"]

        return imported


class TemplateABTestRunner:
    """Runs A/B tests on prompt templates."""

    def __init__(
        self,
        strategy: AllocationStrategy = AllocationStrategy.RANDOM,
        min_samples_per_variant: int = 100,
        confidence_threshold: float = 0.95,
    ):
        """Initialize A/B test runner."""
        self.strategy = strategy
        self.min_samples_per_variant = min_samples_per_variant
        self.confidence_threshold = confidence_threshold
        self.tests: dict[str, ABTest] = {}

    def create_test(
        self,
        name: str,
        variants: list[tuple[str, TemplateVersion, float]],  # (name, version, weight)
        description: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ABTest":
        """Create a new A/B test."""
        test_id = hashlib.sha256(f"{name}:{time.time()}".encode()).hexdigest()[:16]

        ab_variants = []
        for variant_name, template_version, weight in variants:
            variant_id = hashlib.sha256(f"{test_id}:{variant_name}".encode()).hexdigest()[:12]
            ab_variants.append(
                ABVariant(
                    variant_id=variant_id,
                    name=variant_name,
                    template_version=template_version,
                    weight=weight,
                )
            )

        test = ABTest(
            test_id=test_id,
            name=name,
            variants=ab_variants,
            strategy=self.strategy,
            min_samples=self.min_samples_per_variant,
            confidence_threshold=self.confidence_threshold,
            description=description,
            metadata=metadata or {},
        )

        self.tests[test_id] = test
        return test

    def get_test(self, test_id: str) -> Optional["ABTest"]:
        """Get a test by ID."""
        return self.tests.get(test_id)

    def list_tests(self, status: Optional[ABTestStatus] = None) -> list["ABTest"]:
        """List all tests, optionally filtered by status."""
        tests = list(self.tests.values())
        if status:
            tests = [t for t in tests if t.status == status]
        return tests


@dataclass
class ABTest:
    """An A/B test for prompt templates."""

    test_id: str
    name: str
    variants: list[ABVariant]
    strategy: AllocationStrategy
    min_samples: int
    confidence_threshold: float
    description: str = ""
    status: ABTestStatus = ABTestStatus.PENDING
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _round_robin_index: int = field(default=0, repr=False)
    _bandit_alpha: list[float] = field(default_factory=list, repr=False)
    _bandit_beta: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Initialize bandit parameters."""
        if not self._bandit_alpha:
            self._bandit_alpha = [1.0] * len(self.variants)
        if not self._bandit_beta:
            self._bandit_beta = [1.0] * len(self.variants)

    def start(self) -> None:
        """Start the test."""
        if self.status != ABTestStatus.PENDING:
            raise ValueError(f"Cannot start test in {self.status.value} status")
        self.status = ABTestStatus.RUNNING
        self.started_at = datetime.now().isoformat()

    def pause(self) -> None:
        """Pause the test."""
        if self.status != ABTestStatus.RUNNING:
            raise ValueError(f"Cannot pause test in {self.status.value} status")
        self.status = ABTestStatus.PAUSED

    def resume(self) -> None:
        """Resume a paused test."""
        if self.status != ABTestStatus.PAUSED:
            raise ValueError(f"Cannot resume test in {self.status.value} status")
        self.status = ABTestStatus.RUNNING

    def stop(self) -> ABTestResult:
        """Stop the test and return results."""
        if self.status not in [ABTestStatus.RUNNING, ABTestStatus.PAUSED]:
            raise ValueError(f"Cannot stop test in {self.status.value} status")
        self.status = ABTestStatus.COMPLETED
        self.ended_at = datetime.now().isoformat()
        return self.get_results()

    def cancel(self) -> None:
        """Cancel the test."""
        self.status = ABTestStatus.CANCELLED
        self.ended_at = datetime.now().isoformat()

    def select_variant(self) -> ABVariant:
        """Select a variant based on allocation strategy."""
        if self.status != ABTestStatus.RUNNING:
            raise ValueError(f"Cannot select variant for test in {self.status.value} status")

        if self.strategy == AllocationStrategy.RANDOM:
            return random.choice(self.variants)

        elif self.strategy == AllocationStrategy.ROUND_ROBIN:
            variant = self.variants[self._round_robin_index % len(self.variants)]
            self._round_robin_index += 1
            return variant

        elif self.strategy == AllocationStrategy.WEIGHTED:
            weights = [v.weight for v in self.variants]
            return random.choices(self.variants, weights=weights)[0]

        elif self.strategy == AllocationStrategy.MULTI_ARMED_BANDIT:
            # Thompson Sampling
            samples = []
            for i in range(len(self.variants)):
                sample = random.betavariate(self._bandit_alpha[i], self._bandit_beta[i])
                samples.append(sample)
            best_idx = samples.index(max(samples))
            return self.variants[best_idx]

        return random.choice(self.variants)

    def record_result(
        self,
        variant_id: str,
        score: float,
        converted: bool = False,
    ) -> None:
        """Record a result for a variant."""
        variant = None
        variant_idx = None
        for idx, v in enumerate(self.variants):
            if v.variant_id == variant_id:
                variant = v
                variant_idx = idx
                break

        if not variant:
            raise ValueError(f"Variant {variant_id} not found")

        variant.record_impression(score, converted)

        # Update bandit parameters
        if self.strategy == AllocationStrategy.MULTI_ARMED_BANDIT:
            if converted:
                self._bandit_alpha[variant_idx] += 1
            else:
                self._bandit_beta[variant_idx] += 1

    @property
    def total_impressions(self) -> int:
        """Total impressions across all variants."""
        return sum(v.impressions for v in self.variants)

    @property
    def is_ready_for_analysis(self) -> bool:
        """Check if test has enough samples for analysis."""
        return all(v.impressions >= self.min_samples for v in self.variants)

    def _calculate_z_score(self, p1: float, p2: float, n1: int, n2: int) -> float:
        """Calculate z-score for two proportions."""
        if n1 == 0 or n2 == 0:
            return 0.0

        # Pooled proportion
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)

        if p_pool == 0 or p_pool == 1:
            return 0.0

        # Standard error
        se = (p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) ** 0.5

        if se == 0:
            return 0.0

        return (p1 - p2) / se

    def _z_to_confidence(self, z: float) -> float:
        """Convert z-score to confidence level (approximate)."""
        # Using approximation for normal CDF
        import math

        z = abs(z)
        # Taylor series approximation
        t = 1.0 / (1.0 + 0.2316419 * z)
        d = 0.3989423 * math.exp(-z * z / 2)
        p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
        return 1 - 2 * p

    def get_results(self) -> ABTestResult:
        """Get current test results with statistical analysis."""
        # Find best variant by conversion rate
        best_variant = max(self.variants, key=lambda v: v.conversion_rate)
        max_confidence = 0.0

        # Compare best against others
        for variant in self.variants:
            if variant.variant_id != best_variant.variant_id:
                # Z-test for conversion rates
                z = self._calculate_z_score(
                    best_variant.conversion_rate,
                    variant.conversion_rate,
                    best_variant.impressions,
                    variant.impressions,
                )
                conf = self._z_to_confidence(z)
                if conf > max_confidence:
                    max_confidence = conf

        # Generate recommendations
        recommendations = []

        if not self.is_ready_for_analysis:
            min_needed = self.min_samples - min(v.impressions for v in self.variants)
            recommendations.append(
                f"Need at least {min_needed} more samples for statistical significance"
            )

        if max_confidence >= self.confidence_threshold:
            recommendations.append(
                f"'{best_variant.name}' is the winner with {max_confidence:.1%} confidence"
            )
            recommendations.append(f"Consider promoting '{best_variant.name}' to production")
        elif max_confidence >= 0.9:
            recommendations.append(
                f"'{best_variant.name}' is leading but needs more data for conclusive results"
            )
        else:
            recommendations.append("No clear winner yet - continue collecting data")

        # Calculate duration
        duration = 0.0
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.ended_at) if self.ended_at else datetime.now()
            duration = (end - start).total_seconds()

        return ABTestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=self.status,
            variants=self.variants,
            winner=best_variant.variant_id if max_confidence >= self.confidence_threshold else None,
            confidence=max_confidence,
            total_impressions=self.total_impressions,
            duration_seconds=duration,
            started_at=self.started_at or "",
            ended_at=self.ended_at,
            recommendations=recommendations,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "status": self.status.value,
            "strategy": self.strategy.value,
            "variants": [v.to_dict() for v in self.variants],
            "total_impressions": self.total_impressions,
            "is_ready_for_analysis": self.is_ready_for_analysis,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "description": self.description,
            "metadata": self.metadata,
        }


class TemplateExperiment:
    """Run experiments comparing template performance."""

    def __init__(
        self,
        scorer: Callable[[str, str], float],  # (response, expected) -> score
        converter: Optional[Callable[[str, str], bool]] = None,  # (response, expected) -> converted
    ):
        """Initialize experiment."""
        self.scorer = scorer
        self.converter = converter or (lambda r, e: False)

    def run_comparison(
        self,
        templates: list[TemplateVersion],
        test_cases: list[dict[str, Any]],  # [{"variables": {...}, "expected": "..."}]
        render_fn: Callable[[str, dict[str, Any]], str],  # (template_content, variables) -> prompt
        model_fn: Callable[[str], str],  # (prompt) -> response
        n_runs: int = 1,
    ) -> dict[str, Any]:
        """Run a comparison experiment across templates."""
        results = {}

        for template in templates:
            template_scores = []
            template_conversions = 0

            for _ in range(n_runs):
                for case in test_cases:
                    prompt = render_fn(template.content, case["variables"])
                    response = model_fn(prompt)
                    score = self.scorer(response, case.get("expected", ""))
                    converted = self.converter(response, case.get("expected", ""))

                    template_scores.append(score)
                    if converted:
                        template_conversions += 1

            results[template.version_id] = {
                "template_name": template.template_name,
                "version_number": template.version_number,
                "n_runs": n_runs * len(test_cases),
                "avg_score": sum(template_scores) / len(template_scores) if template_scores else 0,
                "min_score": min(template_scores) if template_scores else 0,
                "max_score": max(template_scores) if template_scores else 0,
                "std_score": (
                    (
                        sum(
                            (s - sum(template_scores) / len(template_scores)) ** 2
                            for s in template_scores
                        )
                        / len(template_scores)
                    )
                    ** 0.5
                    if template_scores
                    else 0
                ),
                "conversion_rate": template_conversions / (n_runs * len(test_cases)),
            }

        # Determine winner
        best_id = max(results.keys(), key=lambda k: results[k]["avg_score"])

        return {
            "results": results,
            "winner": best_id,
            "winner_version": templates[
                [t.version_id for t in templates].index(best_id)
            ].version_number,
            "total_test_cases": len(test_cases),
            "total_runs": n_runs,
        }


# Convenience functions
_default_manager: Optional[TemplateVersionManager] = None


def get_default_manager() -> TemplateVersionManager:
    """Get or create default template version manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = TemplateVersionManager()
    return _default_manager


def set_default_manager(manager: TemplateVersionManager) -> None:
    """Set the default template version manager."""
    global _default_manager
    _default_manager = manager


def create_template(
    name: str,
    content: str,
    description: str = "",
    author: Optional[str] = None,
) -> TemplateVersion:
    """Create a new template with initial version."""
    return get_default_manager().create_template(name, content, description, author)


def create_version(
    name: str,
    content: str,
    description: str = "",
    version_level: str = "patch",
) -> TemplateVersion:
    """Create a new version of an existing template."""
    return get_default_manager().create_version(
        name, content, description, version_level=version_level
    )


def get_active_template(name: str) -> Optional[TemplateVersion]:
    """Get the active version of a template."""
    return get_default_manager().get_active_version(name)


def activate_version(name: str, version_id: str) -> TemplateVersion:
    """Activate a specific version."""
    return get_default_manager().activate_version(name, version_id)


def rollback_template(name: str, to_version_id: Optional[str] = None) -> TemplateVersion:
    """Rollback to a previous version."""
    return get_default_manager().rollback(name, to_version_id)


def list_template_versions(name: str) -> list[TemplateVersion]:
    """List all versions of a template."""
    return get_default_manager().list_versions(name)


def diff_template_versions(name: str, version_a: str, version_b: str) -> dict[str, Any]:
    """Compare two versions."""
    return get_default_manager().diff_versions(name, version_a, version_b)


def create_ab_test(
    name: str,
    variants: list[tuple[str, TemplateVersion, float]],
    strategy: AllocationStrategy = AllocationStrategy.RANDOM,
) -> ABTest:
    """Create an A/B test."""
    runner = TemplateABTestRunner(strategy=strategy)
    return runner.create_test(name, variants)


def run_template_comparison(
    templates: list[TemplateVersion],
    test_cases: list[dict[str, Any]],
    render_fn: Callable[[str, dict[str, Any]], str],
    model_fn: Callable[[str], str],
    scorer: Callable[[str, str], float],
) -> dict[str, Any]:
    """Run a quick comparison of templates."""
    experiment = TemplateExperiment(scorer)
    return experiment.run_comparison(templates, test_cases, render_fn, model_fn)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import ABTestRunner. The canonical name is
# TemplateABTestRunner.
ABTestRunner = TemplateABTestRunner
