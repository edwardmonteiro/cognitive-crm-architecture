"""
Outcome Signals - Tracking Results for Learning

From Section 4.4: "Outcome signals: stage changes, replies,
meetings booked, win/loss, slippage"

Outcome signals provide ground truth for:
- Evaluating recommendation quality
- Calibrating confidence scores
- Identifying successful patterns
- Detecting failure modes
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4
from collections import defaultdict


class OutcomeType(Enum):
    """Types of outcome signals."""
    # Pipeline outcomes
    STAGE_ADVANCE = "stage_advance"
    STAGE_REGRESS = "stage_regress"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"
    SLIPPAGE = "slippage"  # Close date pushed

    # Engagement outcomes
    EMAIL_REPLY = "email_reply"
    MEETING_BOOKED = "meeting_booked"
    CALL_COMPLETED = "call_completed"
    PROPOSAL_REQUESTED = "proposal_requested"

    # Negative signals
    NO_RESPONSE = "no_response"
    MEETING_CANCELLED = "meeting_cancelled"
    CONTACT_CHURNED = "contact_churned"

    # NBA specific
    NBA_ACCEPTED = "nba_accepted"
    NBA_REJECTED = "nba_rejected"
    NBA_MODIFIED = "nba_modified"


class SignalPolarity(Enum):
    """Whether signal is positive or negative for deal health."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class OutcomeSignal:
    """
    An outcome signal from the CRM or communication systems.

    Captures what happened after a system action or recommendation,
    enabling closed-loop learning.
    """
    id: UUID = field(default_factory=uuid4)
    outcome_type: OutcomeType = OutcomeType.STAGE_ADVANCE
    timestamp: datetime = field(default_factory=datetime.now)

    # Context
    opportunity_id: UUID = field(default_factory=uuid4)
    account_id: Optional[UUID] = None
    contact_id: Optional[UUID] = None

    # What triggered this outcome (if known)
    triggering_action_id: Optional[UUID] = None
    triggering_recommendation_id: Optional[UUID] = None
    playbook_execution_id: Optional[UUID] = None

    # Details
    details: dict = field(default_factory=dict)
    value_change: float = 0.0  # Change in deal value if any

    # Attribution
    attribution_confidence: float = 0.5  # How confident we are about cause
    time_since_action_hours: Optional[float] = None

    @property
    def polarity(self) -> SignalPolarity:
        """Determine if this is a positive or negative signal."""
        positive = {
            OutcomeType.STAGE_ADVANCE,
            OutcomeType.CLOSED_WON,
            OutcomeType.EMAIL_REPLY,
            OutcomeType.MEETING_BOOKED,
            OutcomeType.CALL_COMPLETED,
            OutcomeType.PROPOSAL_REQUESTED,
            OutcomeType.NBA_ACCEPTED
        }
        negative = {
            OutcomeType.STAGE_REGRESS,
            OutcomeType.CLOSED_LOST,
            OutcomeType.SLIPPAGE,
            OutcomeType.NO_RESPONSE,
            OutcomeType.MEETING_CANCELLED,
            OutcomeType.CONTACT_CHURNED,
            OutcomeType.NBA_REJECTED
        }

        if self.outcome_type in positive:
            return SignalPolarity.POSITIVE
        elif self.outcome_type in negative:
            return SignalPolarity.NEGATIVE
        return SignalPolarity.NEUTRAL


class SignalCollector:
    """
    Collects and stores outcome signals.

    Provides:
    - Signal ingestion
    - Signal querying
    - Attribution linking
    """

    def __init__(self):
        self._signals: list[OutcomeSignal] = []
        self._by_opportunity: dict[str, list[int]] = defaultdict(list)
        self._by_action: dict[str, list[int]] = defaultdict(list)
        self._by_type: dict[OutcomeType, list[int]] = defaultdict(list)

    def record(self, signal: OutcomeSignal) -> None:
        """Record an outcome signal."""
        idx = len(self._signals)
        self._signals.append(signal)

        # Update indices
        self._by_opportunity[str(signal.opportunity_id)].append(idx)

        if signal.triggering_action_id:
            self._by_action[str(signal.triggering_action_id)].append(idx)

        self._by_type[signal.outcome_type].append(idx)

    def get_for_opportunity(self, opportunity_id: UUID) -> list[OutcomeSignal]:
        """Get all signals for an opportunity."""
        indices = self._by_opportunity.get(str(opportunity_id), [])
        return [self._signals[i] for i in indices]

    def get_for_action(self, action_id: UUID) -> list[OutcomeSignal]:
        """Get outcome signals for a specific action."""
        indices = self._by_action.get(str(action_id), [])
        return [self._signals[i] for i in indices]

    def get_by_type(
        self,
        outcome_type: OutcomeType,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> list[OutcomeSignal]:
        """Get signals of a specific type."""
        indices = self._by_type.get(outcome_type, [])
        signals = [self._signals[i] for i in indices]

        if start_date:
            signals = [s for s in signals if s.timestamp >= start_date]
        if end_date:
            signals = [s for s in signals if s.timestamp <= end_date]

        return signals

    def get_recent(self, hours: int = 24) -> list[OutcomeSignal]:
        """Get recent signals."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [s for s in self._signals if s.timestamp >= cutoff]


class OutcomeTracker:
    """
    Tracks outcomes and calculates metrics.

    Provides:
    - Win/loss analysis
    - Conversion tracking
    - NBA effectiveness metrics
    - Attribution analysis
    """

    def __init__(self, signal_collector: SignalCollector):
        self._collector = signal_collector

    def calculate_win_rate(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> dict:
        """Calculate win rate for a period."""
        wins = self._collector.get_by_type(OutcomeType.CLOSED_WON, start_date, end_date)
        losses = self._collector.get_by_type(OutcomeType.CLOSED_LOST, start_date, end_date)

        total = len(wins) + len(losses)
        win_rate = len(wins) / total if total > 0 else 0

        return {
            "wins": len(wins),
            "losses": len(losses),
            "total": total,
            "win_rate": win_rate,
            "period_start": start_date.isoformat() if start_date else None,
            "period_end": end_date.isoformat() if end_date else None
        }

    def calculate_nba_effectiveness(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> dict:
        """Calculate NBA (Next-Best-Action) effectiveness."""
        accepted = self._collector.get_by_type(OutcomeType.NBA_ACCEPTED, start_date, end_date)
        rejected = self._collector.get_by_type(OutcomeType.NBA_REJECTED, start_date, end_date)
        modified = self._collector.get_by_type(OutcomeType.NBA_MODIFIED, start_date, end_date)

        total = len(accepted) + len(rejected) + len(modified)
        if total == 0:
            return {
                "total": 0,
                "acceptance_rate": 0,
                "rejection_rate": 0,
                "modification_rate": 0
            }

        return {
            "total": total,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "modified": len(modified),
            "acceptance_rate": len(accepted) / total,
            "rejection_rate": len(rejected) / total,
            "modification_rate": len(modified) / total,
            "period_start": start_date.isoformat() if start_date else None,
            "period_end": end_date.isoformat() if end_date else None
        }

    def calculate_conversion_rates(self) -> dict:
        """Calculate stage-to-stage conversion rates."""
        advances = self._collector.get_by_type(OutcomeType.STAGE_ADVANCE)
        regresses = self._collector.get_by_type(OutcomeType.STAGE_REGRESS)

        # Group by from_stage -> to_stage
        transitions = defaultdict(lambda: {"advances": 0, "regresses": 0})

        for signal in advances:
            from_stage = signal.details.get("from_stage", "unknown")
            to_stage = signal.details.get("to_stage", "unknown")
            transitions[f"{from_stage}->{to_stage}"]["advances"] += 1

        for signal in regresses:
            from_stage = signal.details.get("from_stage", "unknown")
            to_stage = signal.details.get("to_stage", "unknown")
            transitions[f"{from_stage}->{to_stage}"]["regresses"] += 1

        return dict(transitions)

    def calculate_response_rates(self) -> dict:
        """Calculate response rates for outbound actions."""
        replies = len(self._collector.get_by_type(OutcomeType.EMAIL_REPLY))
        no_response = len(self._collector.get_by_type(OutcomeType.NO_RESPONSE))
        meetings = len(self._collector.get_by_type(OutcomeType.MEETING_BOOKED))

        total_outbound = replies + no_response

        return {
            "email_reply_rate": replies / total_outbound if total_outbound > 0 else 0,
            "meeting_booking_rate": meetings / (replies + meetings) if (replies + meetings) > 0 else 0,
            "total_replies": replies,
            "total_no_response": no_response,
            "total_meetings": meetings
        }

    def get_action_outcomes(
        self,
        action_type: str,
        limit: int = 100
    ) -> list[dict]:
        """
        Get outcomes for a specific action type.

        Used for learning what actions lead to positive outcomes.
        """
        outcomes = []

        for signal in self._collector._signals[-limit:]:
            if signal.triggering_action_id:
                outcomes.append({
                    "action_id": str(signal.triggering_action_id),
                    "outcome_type": signal.outcome_type.value,
                    "polarity": signal.polarity.value,
                    "time_since_action_hours": signal.time_since_action_hours,
                    "attribution_confidence": signal.attribution_confidence
                })

        return outcomes

    def calculate_slippage_rate(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> dict:
        """Calculate forecast slippage rate."""
        slippage_signals = self._collector.get_by_type(
            OutcomeType.SLIPPAGE, start_date, end_date
        )

        # Calculate average slip days
        total_slip_days = 0
        for signal in slippage_signals:
            total_slip_days += signal.details.get("slip_days", 0)

        avg_slip = total_slip_days / len(slippage_signals) if slippage_signals else 0

        return {
            "slippage_count": len(slippage_signals),
            "total_slip_days": total_slip_days,
            "average_slip_days": avg_slip
        }

    def generate_learning_dataset(self) -> list[dict]:
        """
        Generate dataset for learning algorithms.

        Creates feature-outcome pairs for training.
        """
        dataset = []

        for signal in self._collector._signals:
            if signal.triggering_action_id:
                record = {
                    "opportunity_id": str(signal.opportunity_id),
                    "action_id": str(signal.triggering_action_id),
                    "recommendation_id": str(signal.triggering_recommendation_id) if signal.triggering_recommendation_id else None,
                    "outcome_type": signal.outcome_type.value,
                    "polarity": signal.polarity.value,
                    "attribution_confidence": signal.attribution_confidence,
                    "time_to_outcome_hours": signal.time_since_action_hours,
                    "value_change": signal.value_change,
                    "timestamp": signal.timestamp.isoformat(),
                    "details": signal.details
                }
                dataset.append(record)

        return dataset
