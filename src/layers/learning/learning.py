"""
Learning Engine - Policy and Model Tuning

From Section 4.4: "Safety: model updates gated; learning may initially
be limited to retrieval ranking and playbook policies"

The learning engine implements safe, incremental learning:
- Retrieval ranking optimization
- Playbook policy tuning
- Confidence calibration
- Pattern-based improvements

Full model fine-tuning is gated and requires explicit approval.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

from .signals import SignalCollector, OutcomeTracker, OutcomeType, SignalPolarity
from .feedback import FeedbackCollector, FeedbackAnalyzer, FeedbackType


class LearningMode(Enum):
    """Learning modes with different safety levels."""
    OBSERVE_ONLY = "observe_only"         # Just collect data
    RETRIEVAL_TUNING = "retrieval_tuning" # Adjust retrieval rankings
    POLICY_TUNING = "policy_tuning"       # Adjust playbook policies
    PROMPT_TUNING = "prompt_tuning"       # Adjust prompts (gated)
    MODEL_TUNING = "model_tuning"         # Fine-tune models (heavily gated)


@dataclass
class LearningConfig:
    """Configuration for the learning engine."""
    mode: LearningMode = LearningMode.RETRIEVAL_TUNING
    min_samples_for_learning: int = 50
    confidence_threshold: float = 0.8
    auto_apply_changes: bool = False  # Require approval for changes
    learning_rate: float = 0.1
    max_change_per_iteration: float = 0.05


@dataclass
class LearningUpdate:
    """A proposed or applied learning update."""
    id: UUID = field(default_factory=uuid4)
    update_type: str = ""  # retrieval_weight, policy_threshold, etc.
    target: str = ""       # What is being updated
    current_value: Any = None
    proposed_value: Any = None

    # Evidence
    samples_used: int = 0
    confidence: float = 0.0
    expected_improvement: float = 0.0

    # Status
    status: str = "proposed"  # proposed, approved, applied, rejected
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    approved_by: Optional[str] = None

    # Validation
    validation_results: dict = field(default_factory=dict)


class RetrievalLearner:
    """
    Learns to improve retrieval ranking.

    Uses feedback on retrieved content to:
    - Adjust source weighting
    - Improve relevance scoring
    - Identify useful content patterns
    """

    def __init__(self, config: LearningConfig = None):
        self._config = config or LearningConfig()
        self._source_weights: dict[str, float] = {
            "interaction_history": 1.0,
            "product_docs": 1.0,
            "sales_playbook": 1.0,
            "policy": 1.0,
            "competitive_intel": 1.0
        }
        self._relevance_feedback: list[dict] = []

    def record_relevance_feedback(
        self,
        source_type: str,
        document_id: str,
        was_helpful: bool,
        context: dict = None
    ) -> None:
        """Record whether retrieved content was helpful."""
        self._relevance_feedback.append({
            "source_type": source_type,
            "document_id": document_id,
            "was_helpful": was_helpful,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        })

    def calculate_source_effectiveness(self) -> dict[str, dict]:
        """Calculate how effective each source type is."""
        source_stats = {}

        for source_type in self._source_weights:
            feedback = [f for f in self._relevance_feedback if f["source_type"] == source_type]

            if not feedback:
                source_stats[source_type] = {
                    "samples": 0,
                    "helpful_rate": 0.5,
                    "current_weight": self._source_weights[source_type]
                }
                continue

            helpful = sum(1 for f in feedback if f["was_helpful"])
            helpful_rate = helpful / len(feedback)

            source_stats[source_type] = {
                "samples": len(feedback),
                "helpful_rate": helpful_rate,
                "current_weight": self._source_weights[source_type]
            }

        return source_stats

    def propose_weight_updates(self) -> list[LearningUpdate]:
        """Propose updates to source weights based on effectiveness."""
        updates = []
        stats = self.calculate_source_effectiveness()

        for source_type, data in stats.items():
            if data["samples"] < self._config.min_samples_for_learning:
                continue

            helpful_rate = data["helpful_rate"]
            current_weight = data["current_weight"]

            # Calculate proposed adjustment
            if helpful_rate > 0.7:
                adjustment = self._config.learning_rate * (helpful_rate - 0.5)
            elif helpful_rate < 0.3:
                adjustment = -self._config.learning_rate * (0.5 - helpful_rate)
            else:
                continue  # No change needed

            # Cap the change
            adjustment = max(-self._config.max_change_per_iteration,
                           min(adjustment, self._config.max_change_per_iteration))

            proposed_weight = max(0.1, min(2.0, current_weight + adjustment))

            if abs(proposed_weight - current_weight) > 0.01:
                updates.append(LearningUpdate(
                    update_type="retrieval_weight",
                    target=source_type,
                    current_value=current_weight,
                    proposed_value=proposed_weight,
                    samples_used=data["samples"],
                    confidence=min(1.0, data["samples"] / 100),
                    expected_improvement=abs(adjustment)
                ))

        return updates

    def apply_update(self, update: LearningUpdate) -> bool:
        """Apply an approved update."""
        if update.update_type != "retrieval_weight":
            return False

        if update.target in self._source_weights:
            self._source_weights[update.target] = update.proposed_value
            update.status = "applied"
            update.applied_at = datetime.now()
            return True

        return False


class PolicyLearner:
    """
    Learns to improve playbook policies.

    Adjusts:
    - Confidence thresholds
    - Approval requirements
    - Escalation rules
    - Step conditions
    """

    def __init__(
        self,
        config: LearningConfig = None,
        outcome_tracker: OutcomeTracker = None,
        feedback_analyzer: FeedbackAnalyzer = None
    ):
        self._config = config or LearningConfig()
        self._outcomes = outcome_tracker
        self._feedback = feedback_analyzer

        self._policies: dict[str, dict] = {
            "approval_threshold": {"default": 0.8},
            "risk_alert_threshold": {"default": 0.6},
            "auto_update_threshold": {"default": 0.95},
            "escalation_threshold": {"default": 0.5}
        }

    def analyze_threshold_effectiveness(self, policy_name: str) -> dict:
        """Analyze if a threshold is set appropriately."""
        if not self._outcomes or not self._feedback:
            return {"error": "Missing outcome tracker or feedback analyzer"}

        current_threshold = self._policies.get(policy_name, {}).get("default", 0.5)

        # Get approval feedback
        approval_stats = self._feedback.calculate_approval_rate(days=30)

        # Analyze outcomes
        nba_stats = self._outcomes.calculate_nba_effectiveness()

        return {
            "policy": policy_name,
            "current_threshold": current_threshold,
            "approval_rate": approval_stats.get("approval_rate", 0),
            "rejection_rate": approval_stats.get("rejection_rate", 0),
            "nba_acceptance": nba_stats.get("acceptance_rate", 0),
            "sample_size": approval_stats.get("total", 0) + nba_stats.get("total", 0)
        }

    def propose_threshold_updates(self) -> list[LearningUpdate]:
        """Propose threshold adjustments."""
        updates = []

        for policy_name in self._policies:
            analysis = self.analyze_threshold_effectiveness(policy_name)

            if analysis.get("sample_size", 0) < self._config.min_samples_for_learning:
                continue

            current = analysis["current_threshold"]
            approval_rate = analysis.get("approval_rate", 0.5)
            rejection_rate = analysis.get("rejection_rate", 0.5)

            # If rejection rate is high, lower threshold (require more approval)
            if rejection_rate > 0.3:
                proposed = max(0.5, current - 0.05)
            # If approval rate is very high, could raise threshold
            elif approval_rate > 0.95 and rejection_rate < 0.02:
                proposed = min(0.95, current + 0.02)
            else:
                continue

            if abs(proposed - current) > 0.01:
                updates.append(LearningUpdate(
                    update_type="policy_threshold",
                    target=policy_name,
                    current_value=current,
                    proposed_value=proposed,
                    samples_used=analysis["sample_size"],
                    confidence=min(1.0, analysis["sample_size"] / 100),
                    expected_improvement=abs(proposed - current)
                ))

        return updates

    def apply_update(self, update: LearningUpdate) -> bool:
        """Apply an approved policy update."""
        if update.update_type != "policy_threshold":
            return False

        if update.target in self._policies:
            self._policies[update.target]["default"] = update.proposed_value
            update.status = "applied"
            update.applied_at = datetime.now()
            return True

        return False


class PlaybookOptimizer:
    """
    Optimizes playbook configurations based on outcomes.

    Learns:
    - Which steps are most effective
    - When to skip optional steps
    - Optimal step ordering
    - Condition thresholds
    """

    def __init__(
        self,
        config: LearningConfig = None,
        outcome_tracker: OutcomeTracker = None
    ):
        self._config = config or LearningConfig()
        self._outcomes = outcome_tracker
        self._step_outcomes: dict[str, list[dict]] = {}

    def record_step_outcome(
        self,
        playbook_name: str,
        step_name: str,
        outcome: str,  # success, failure, skipped
        context: dict = None
    ) -> None:
        """Record outcome for a playbook step."""
        key = f"{playbook_name}:{step_name}"
        if key not in self._step_outcomes:
            self._step_outcomes[key] = []

        self._step_outcomes[key].append({
            "outcome": outcome,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        })

    def analyze_step_effectiveness(self, playbook_name: str) -> dict:
        """Analyze effectiveness of each step in a playbook."""
        step_stats = {}

        for key, outcomes in self._step_outcomes.items():
            if key.startswith(f"{playbook_name}:"):
                step_name = key.split(":")[1]

                success = sum(1 for o in outcomes if o["outcome"] == "success")
                failure = sum(1 for o in outcomes if o["outcome"] == "failure")
                skipped = sum(1 for o in outcomes if o["outcome"] == "skipped")

                total = success + failure + skipped

                step_stats[step_name] = {
                    "total": total,
                    "success_rate": success / total if total > 0 else 0,
                    "failure_rate": failure / total if total > 0 else 0,
                    "skip_rate": skipped / total if total > 0 else 0
                }

        return step_stats

    def propose_playbook_updates(self, playbook_name: str) -> list[LearningUpdate]:
        """Propose updates to playbook configuration."""
        updates = []
        stats = self.analyze_step_effectiveness(playbook_name)

        for step_name, data in stats.items():
            if data["total"] < self._config.min_samples_for_learning:
                continue

            # High failure rate suggests step needs adjustment
            if data["failure_rate"] > 0.3:
                updates.append(LearningUpdate(
                    update_type="playbook_step",
                    target=f"{playbook_name}:{step_name}",
                    current_value={"failure_rate": data["failure_rate"]},
                    proposed_value={"action": "review_and_adjust"},
                    samples_used=data["total"],
                    confidence=min(1.0, data["total"] / 50),
                    expected_improvement=data["failure_rate"]
                ))

            # High skip rate might mean step is often unnecessary
            if data["skip_rate"] > 0.7:
                updates.append(LearningUpdate(
                    update_type="playbook_step",
                    target=f"{playbook_name}:{step_name}",
                    current_value={"skip_rate": data["skip_rate"]},
                    proposed_value={"action": "consider_making_optional"},
                    samples_used=data["total"],
                    confidence=min(1.0, data["total"] / 50),
                    expected_improvement=0.1
                ))

        return updates


class LearningEngine:
    """
    Main learning engine coordinating all learning components.

    Provides:
    - Unified learning interface
    - Safety controls
    - Update approval workflow
    - Metrics and monitoring
    """

    def __init__(
        self,
        config: LearningConfig = None,
        signal_collector: SignalCollector = None,
        feedback_collector: FeedbackCollector = None
    ):
        self._config = config or LearningConfig()

        # Initialize collectors if not provided
        self._signals = signal_collector or SignalCollector()
        self._feedback = feedback_collector or FeedbackCollector()

        # Initialize trackers and analyzers
        self._outcome_tracker = OutcomeTracker(self._signals)
        self._feedback_analyzer = FeedbackAnalyzer(self._feedback)

        # Initialize learners
        self._retrieval_learner = RetrievalLearner(self._config)
        self._policy_learner = PolicyLearner(
            self._config,
            self._outcome_tracker,
            self._feedback_analyzer
        )
        self._playbook_optimizer = PlaybookOptimizer(
            self._config,
            self._outcome_tracker
        )

        # Track updates
        self._pending_updates: list[LearningUpdate] = []
        self._applied_updates: list[LearningUpdate] = []

    def run_learning_cycle(self) -> dict:
        """
        Run a complete learning cycle.

        Returns summary of proposed updates.
        """
        if self._config.mode == LearningMode.OBSERVE_ONLY:
            return {"mode": "observe_only", "updates": []}

        updates = []

        # Retrieval learning
        if self._config.mode in [LearningMode.RETRIEVAL_TUNING, LearningMode.POLICY_TUNING]:
            retrieval_updates = self._retrieval_learner.propose_weight_updates()
            updates.extend(retrieval_updates)

        # Policy learning
        if self._config.mode in [LearningMode.POLICY_TUNING, LearningMode.PROMPT_TUNING]:
            policy_updates = self._policy_learner.propose_threshold_updates()
            updates.extend(policy_updates)

        self._pending_updates.extend(updates)

        # Auto-apply if configured
        if self._config.auto_apply_changes:
            for update in updates:
                if update.confidence >= self._config.confidence_threshold:
                    self._apply_update(update)

        return {
            "mode": self._config.mode.value,
            "updates_proposed": len(updates),
            "updates": [
                {
                    "type": u.update_type,
                    "target": u.target,
                    "current": u.current_value,
                    "proposed": u.proposed_value,
                    "confidence": u.confidence
                }
                for u in updates
            ]
        }

    def _apply_update(self, update: LearningUpdate) -> bool:
        """Apply an update through the appropriate learner."""
        success = False

        if update.update_type == "retrieval_weight":
            success = self._retrieval_learner.apply_update(update)
        elif update.update_type == "policy_threshold":
            success = self._policy_learner.apply_update(update)

        if success:
            self._applied_updates.append(update)

        return success

    def approve_update(self, update_id: UUID, approver: str) -> bool:
        """Approve a pending update."""
        for update in self._pending_updates:
            if update.id == update_id:
                update.status = "approved"
                update.approved_by = approver
                return self._apply_update(update)

        return False

    def reject_update(self, update_id: UUID, reason: str) -> bool:
        """Reject a pending update."""
        for update in self._pending_updates:
            if update.id == update_id:
                update.status = "rejected"
                update.validation_results["rejection_reason"] = reason
                return True

        return False

    def get_learning_metrics(self) -> dict:
        """Get metrics about the learning system."""
        return {
            "config": {
                "mode": self._config.mode.value,
                "auto_apply": self._config.auto_apply_changes
            },
            "outcomes": {
                "total_signals": len(self._signals._signals),
                "win_rate": self._outcome_tracker.calculate_win_rate(),
                "nba_effectiveness": self._outcome_tracker.calculate_nba_effectiveness()
            },
            "feedback": {
                "total_entries": len(self._feedback._feedback),
                "approval_rate": self._feedback_analyzer.calculate_approval_rate()
            },
            "updates": {
                "pending": len(self._pending_updates),
                "applied": len(self._applied_updates)
            },
            "retrieval_weights": self._retrieval_learner._source_weights
        }

    def generate_learning_report(self) -> dict:
        """Generate a comprehensive learning report."""
        return {
            "generated_at": datetime.now().isoformat(),
            "metrics": self.get_learning_metrics(),
            "improvement_recommendations": self._feedback_analyzer.generate_improvement_recommendations(),
            "pending_updates": [
                {
                    "id": str(u.id),
                    "type": u.update_type,
                    "target": u.target,
                    "proposed": u.proposed_value,
                    "confidence": u.confidence
                }
                for u in self._pending_updates
            ],
            "applied_updates": [
                {
                    "id": str(u.id),
                    "type": u.update_type,
                    "target": u.target,
                    "value": u.proposed_value,
                    "applied_at": u.applied_at.isoformat() if u.applied_at else None
                }
                for u in self._applied_updates[-10:]  # Last 10
            ]
        }
