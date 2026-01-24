"""
Prompt template versioning and A/B testing utilities.

Provides tools for:
- Version management for prompt templates
- Changelog tracking and rollback
- A/B testing framework for prompt variants
- Statistical significance testing for comparisons
"""

import hashlib
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


class VersionStatus(Enum):
    """Status of a template version."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ABTestStatus(Enum):
    """Status of an A/B test."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AllocationStrategy(Enum):
    """Traffic allocation strategy for A/B tests."""

    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"


@dataclass
class TemplateChange:
    """A single change in template version history."""

    timestamp: str
    change_type: str  # "create", "update", "status_change"
    description: str
    author: Optional[str] = None
    previous_content: Optional[str] = None
    new_content: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """A versioned prompt template."""

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
        """SHA-256 hash of template content."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]

    @property
    def is_active(self) -> bool:
        """Check if version is active."""
        return self.status == VersionStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """A variant in an A/B test."""

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
        """Calculate conversion rate."""
        return self.conversions / self.impressions if self.impressions > 0 else 0.0

    @property
    def avg_score(self) -> float:
        """Calculate average score."""
        return self.total_score / self.impressions if self.impressions > 0 else 0.0

    @property
    def score_variance(self) -> float:
        """Calculate score variance."""
        if len(self.scores) < 2:
            return 0.0
        mean = self.avg_score
        return sum((s - mean) ** 2 for s in self.scores) / (len(self.scores) - 1)

    @property
    def score_std(self) -> float:
        """Calculate score standard deviation."""
        return self.score_variance**0.5

    def record_impression(self, score: float, converted: bool = False) -> None:
        """Record an impression with score."""
        self.impressions += 1
        self.total_score += score
        self.scores.append(score)
        if converted:
            self.conversions += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Results of an A/B test."""

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
        """Check if results are statistically significant."""
        return self.confidence >= 0.95

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Manages versioned prompt templates."""

    def __init__(self):
        """Initialize version manager."""
        self.templates: dict[str, dict[str, TemplateVersion]] = {}  # name -> version_id -> version
        self.active_versions: dict[str, str] = {}  # name -> active version_id

    def _generate_version_id(self, name: str, version: str) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().isoformat()
        content = f"{name}:{version}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _extract_variables(self, content: str) -> list[str]:
        """Extract variable names from template content."""
        import re

        # Match {variable} and {{variable}} patterns
        pattern = r"\{+(\w+)\}+"
        matches = re.findall(pattern, content)
        return list(set(matches))

    def _parse_version(self, version: str) -> tuple[int, int, int]:
        """Parse semantic version string."""
        parts = version.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch)

    def _increment_version(self, current: str, level: str = "patch") -> str:
        """Increment version number."""
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
        """Create a new template with initial version."""
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
        """Create a new version of an existing template."""
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
        """Activate a specific version of a template."""
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
        """Get the active version of a template."""
        if name not in self.active_versions:
            return None
        version_id = self.active_versions[name]
        return self.templates.get(name, {}).get(version_id)

    def get_version(self, name: str, version_id: str) -> Optional[TemplateVersion]:
        """Get a specific version of a template."""
        return self.templates.get(name, {}).get(version_id)

    def get_version_by_number(self, name: str, version_number: str) -> Optional[TemplateVersion]:
        """Get a version by its version number."""
        if name not in self.templates:
            return None
        for version in self.templates[name].values():
            if version.version_number == version_number:
                return version
        return None

    def list_versions(self, name: str, include_archived: bool = False) -> list[TemplateVersion]:
        """List all versions of a template."""
        if name not in self.templates:
            return []
        versions = list(self.templates[name].values())
        if not include_archived:
            versions = [v for v in versions if v.status != VersionStatus.ARCHIVED]
        return sorted(versions, key=lambda v: self._parse_version(v.version_number), reverse=True)

    def list_templates(self) -> list[str]:
        """List all template names."""
        return list(self.templates.keys())

    def rollback(
        self, name: str, to_version_id: Optional[str] = None, author: Optional[str] = None
    ) -> TemplateVersion:
        """Rollback to a previous version."""
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
