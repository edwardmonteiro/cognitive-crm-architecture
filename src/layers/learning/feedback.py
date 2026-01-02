"""
Human Feedback Collection and Analysis

From Section 4.4: "Human feedback: overrides, edits, rejection reasons;
used for policy tuning and prompt/tool refinement"

Human feedback is essential for:
- Improving recommendation quality
- Calibrating confidence thresholds
- Identifying edge cases
- Maintaining trust through responsiveness
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4
from collections import defaultdict


class FeedbackType(Enum):
    """Types of human feedback."""
    # Approval-related
    APPROVAL = "approval"
    REJECTION = "rejection"
    MODIFICATION = "modification"

    # Content feedback
    TONE_CORRECTION = "tone_correction"
    FACTUAL_CORRECTION = "factual_correction"
    CONTEXT_MISSING = "context_missing"
    IRRELEVANT = "irrelevant"

    # Quality feedback
    TOO_VERBOSE = "too_verbose"
    TOO_BRIEF = "too_brief"
    UNCLEAR = "unclear"

    # Override
    OVERRIDE = "override"
    BETTER_ALTERNATIVE = "better_alternative"

    # Positive feedback
    HELPFUL = "helpful"
    ACCURATE = "accurate"
    TIMELY = "timely"


class FeedbackSeverity(Enum):
    """Severity of feedback for prioritization."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    POSITIVE = "positive"


@dataclass
class FeedbackEntry:
    """
    A piece of human feedback on system output.

    Captures both the feedback and context for learning.
    """
    id: UUID = field(default_factory=uuid4)
    feedback_type: FeedbackType = FeedbackType.MODIFICATION
    severity: FeedbackSeverity = FeedbackSeverity.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)

    # What was feedback on
    target_type: str = ""  # recommendation, summary, email_draft, etc.
    target_id: UUID = field(default_factory=uuid4)
    playbook_name: str = ""
    step_name: str = ""

    # Context
    opportunity_id: Optional[UUID] = None
    user_id: str = ""
    user_role: str = ""

    # Feedback content
    original_content: Any = None
    corrected_content: Any = None
    feedback_text: str = ""
    tags: list = field(default_factory=list)

    # Learning signals
    should_retrain: bool = False
    pattern_identified: str = ""

    # Metrics
    time_to_feedback_seconds: int = 0  # Time from presentation to feedback


class FeedbackCollector:
    """
    Collects and organizes human feedback.

    Provides:
    - Feedback ingestion
    - Pattern tracking
    - Aggregation for learning
    """

    def __init__(self):
        self._feedback: list[FeedbackEntry] = []
        self._by_type: dict[FeedbackType, list[int]] = defaultdict(list)
        self._by_playbook: dict[str, list[int]] = defaultdict(list)
        self._by_user: dict[str, list[int]] = defaultdict(list)
        self._patterns: dict[str, int] = defaultdict(int)

    def record(self, entry: FeedbackEntry) -> None:
        """Record a feedback entry."""
        idx = len(self._feedback)
        self._feedback.append(entry)

        # Update indices
        self._by_type[entry.feedback_type].append(idx)
        if entry.playbook_name:
            self._by_playbook[entry.playbook_name].append(idx)
        if entry.user_id:
            self._by_user[entry.user_id].append(idx)

        # Track patterns
        for tag in entry.tags:
            self._patterns[tag] += 1
        if entry.pattern_identified:
            self._patterns[entry.pattern_identified] += 1

    def get_by_type(
        self,
        feedback_type: FeedbackType,
        days: int = None
    ) -> list[FeedbackEntry]:
        """Get feedback of a specific type."""
        indices = self._by_type.get(feedback_type, [])
        entries = [self._feedback[i] for i in indices]

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            entries = [e for e in entries if e.timestamp >= cutoff]

        return entries

    def get_by_playbook(self, playbook_name: str) -> list[FeedbackEntry]:
        """Get feedback for a specific playbook."""
        indices = self._by_playbook.get(playbook_name, [])
        return [self._feedback[i] for i in indices]

    def get_by_user(self, user_id: str) -> list[FeedbackEntry]:
        """Get feedback from a specific user."""
        indices = self._by_user.get(user_id, [])
        return [self._feedback[i] for i in indices]

    def get_common_patterns(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Get most common feedback patterns."""
        sorted_patterns = sorted(
            self._patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_patterns[:top_n]

    def get_recent(self, hours: int = 24) -> list[FeedbackEntry]:
        """Get recent feedback."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [e for e in self._feedback if e.timestamp >= cutoff]


class FeedbackAnalyzer:
    """
    Analyzes feedback patterns for learning insights.

    Provides:
    - Pattern detection
    - Trend analysis
    - Improvement recommendations
    """

    def __init__(self, collector: FeedbackCollector):
        self._collector = collector

    def calculate_approval_rate(self, days: int = 30) -> dict:
        """Calculate approval rate over time."""
        cutoff = datetime.now() - timedelta(days=days)

        all_feedback = [e for e in self._collector._feedback if e.timestamp >= cutoff]
        approvals = [e for e in all_feedback if e.feedback_type == FeedbackType.APPROVAL]
        rejections = [e for e in all_feedback if e.feedback_type == FeedbackType.REJECTION]
        modifications = [e for e in all_feedback if e.feedback_type == FeedbackType.MODIFICATION]

        total = len(approvals) + len(rejections) + len(modifications)

        return {
            "total": total,
            "approval_rate": len(approvals) / total if total > 0 else 0,
            "rejection_rate": len(rejections) / total if total > 0 else 0,
            "modification_rate": len(modifications) / total if total > 0 else 0,
            "period_days": days
        }

    def identify_problem_areas(self, threshold: int = 5) -> list[dict]:
        """Identify areas with repeated negative feedback."""
        problem_areas = []

        # By playbook
        for playbook_name in self._collector._by_playbook:
            feedback = self._collector.get_by_playbook(playbook_name)
            negative = [f for f in feedback if f.severity in [
                FeedbackSeverity.CRITICAL, FeedbackSeverity.HIGH
            ]]

            if len(negative) >= threshold:
                problem_areas.append({
                    "type": "playbook",
                    "name": playbook_name,
                    "negative_count": len(negative),
                    "common_issues": self._get_common_issues(negative)
                })

        # By feedback type
        for feedback_type in [FeedbackType.FACTUAL_CORRECTION, FeedbackType.CONTEXT_MISSING]:
            feedback = self._collector.get_by_type(feedback_type, days=30)
            if len(feedback) >= threshold:
                problem_areas.append({
                    "type": "feedback_type",
                    "name": feedback_type.value,
                    "count": len(feedback),
                    "common_issues": self._get_common_issues(feedback)
                })

        return problem_areas

    def _get_common_issues(self, feedback_list: list[FeedbackEntry]) -> list[str]:
        """Extract common issues from feedback."""
        all_tags = []
        for f in feedback_list:
            all_tags.extend(f.tags)

        tag_counts = defaultdict(int)
        for tag in all_tags:
            tag_counts[tag] += 1

        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, _ in sorted_tags[:5]]

    def calculate_correction_patterns(self) -> dict:
        """Analyze what types of corrections are most common."""
        modifications = self._collector.get_by_type(FeedbackType.MODIFICATION, days=30)

        patterns = {
            "tone": 0,
            "factual": 0,
            "length": 0,
            "context": 0,
            "other": 0
        }

        for mod in modifications:
            if FeedbackType.TONE_CORRECTION.value in mod.tags:
                patterns["tone"] += 1
            elif FeedbackType.FACTUAL_CORRECTION.value in mod.tags:
                patterns["factual"] += 1
            elif any(t in mod.tags for t in ["too_verbose", "too_brief"]):
                patterns["length"] += 1
            elif FeedbackType.CONTEXT_MISSING.value in mod.tags:
                patterns["context"] += 1
            else:
                patterns["other"] += 1

        total = sum(patterns.values())
        if total > 0:
            patterns = {k: {"count": v, "percentage": v/total} for k, v in patterns.items()}

        return patterns

    def calculate_response_time_metrics(self) -> dict:
        """Analyze how quickly humans provide feedback."""
        all_feedback = self._collector._feedback

        if not all_feedback:
            return {"average_seconds": 0, "median_seconds": 0}

        times = [f.time_to_feedback_seconds for f in all_feedback if f.time_to_feedback_seconds > 0]

        if not times:
            return {"average_seconds": 0, "median_seconds": 0}

        times.sort()
        median_idx = len(times) // 2

        return {
            "average_seconds": sum(times) / len(times),
            "median_seconds": times[median_idx],
            "min_seconds": min(times),
            "max_seconds": max(times),
            "sample_size": len(times)
        }

    def generate_improvement_recommendations(self) -> list[dict]:
        """Generate recommendations for system improvement."""
        recommendations = []

        # Check approval rate
        approval_stats = self.calculate_approval_rate(days=30)
        if approval_stats["rejection_rate"] > 0.2:
            recommendations.append({
                "priority": "high",
                "area": "recommendation_quality",
                "issue": f"High rejection rate ({approval_stats['rejection_rate']:.1%})",
                "suggestion": "Review rejected recommendations for common patterns"
            })

        if approval_stats["modification_rate"] > 0.3:
            recommendations.append({
                "priority": "medium",
                "area": "content_generation",
                "issue": f"High modification rate ({approval_stats['modification_rate']:.1%})",
                "suggestion": "Analyze modifications to improve initial output quality"
            })

        # Check problem areas
        problem_areas = self.identify_problem_areas(threshold=3)
        for area in problem_areas:
            recommendations.append({
                "priority": "high",
                "area": area["name"],
                "issue": f"{area['negative_count']} negative feedback items",
                "suggestion": f"Address issues: {', '.join(area['common_issues'][:3])}"
            })

        # Check correction patterns
        patterns = self.calculate_correction_patterns()
        for pattern_type, data in patterns.items():
            if isinstance(data, dict) and data.get("percentage", 0) > 0.25:
                recommendations.append({
                    "priority": "medium",
                    "area": f"{pattern_type}_corrections",
                    "issue": f"{pattern_type.title()} corrections are {data['percentage']:.1%} of all modifications",
                    "suggestion": f"Focus on improving {pattern_type} in generated content"
                })

        return recommendations

    def generate_learning_dataset(self) -> list[dict]:
        """Generate dataset for learning from feedback."""
        dataset = []

        for entry in self._collector._feedback:
            if entry.original_content and entry.corrected_content:
                record = {
                    "target_type": entry.target_type,
                    "feedback_type": entry.feedback_type.value,
                    "original": entry.original_content,
                    "corrected": entry.corrected_content,
                    "tags": entry.tags,
                    "playbook": entry.playbook_name,
                    "step": entry.step_name,
                    "feedback_text": entry.feedback_text,
                    "timestamp": entry.timestamp.isoformat()
                }
                dataset.append(record)

        return dataset
