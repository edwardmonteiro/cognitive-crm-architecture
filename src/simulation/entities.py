"""
Simulation Entities

Defines the simulated entities used in feasibility demonstrations:
- Sellers with varying skill levels
- Opportunities with stage progression
- Interactions across channels
- NBA recommendations with outcomes
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
import random


class SellerSkillLevel(Enum):
    """Seller skill distribution."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SimulatedSeller:
    """
    A simulated seller in the system.

    Sellers have:
    - Skill levels affecting baseline performance
    - Time allocation patterns
    - NBA adoption tendencies
    """
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    skill_level: SellerSkillLevel = SellerSkillLevel.MEDIUM

    # Performance characteristics
    base_conversion_rate: float = 0.315
    base_cycle_time_days: float = 94.0
    admin_time_hours_per_week: float = 14.8
    avg_touchpoint_interval_hours: float = 52.4

    # NBA adoption
    nba_adoption_rate: float = 0.7  # Likelihood to act on NBA
    nba_modification_rate: float = 0.2  # Likelihood to modify NBA

    # Active opportunities
    active_opportunity_ids: list = field(default_factory=list)

    # Simulation state
    total_deals_closed: int = 0
    total_deals_lost: int = 0
    cumulative_admin_hours: float = 0.0

    @classmethod
    def generate(cls, skill_distribution: dict = None) -> "SimulatedSeller":
        """Generate a seller with random characteristics."""
        skill_distribution = skill_distribution or {"high": 0.2, "medium": 0.6, "low": 0.2}

        r = random.random()
        if r < skill_distribution["high"]:
            skill = SellerSkillLevel.HIGH
            conversion_mod = 1.15
            cycle_mod = 0.85
        elif r < skill_distribution["high"] + skill_distribution["medium"]:
            skill = SellerSkillLevel.MEDIUM
            conversion_mod = 1.0
            cycle_mod = 1.0
        else:
            skill = SellerSkillLevel.LOW
            conversion_mod = 0.85
            cycle_mod = 1.15

        return cls(
            name=f"Seller_{uuid4().hex[:6]}",
            skill_level=skill,
            base_conversion_rate=0.315 * conversion_mod * random.uniform(0.9, 1.1),
            base_cycle_time_days=94.0 * cycle_mod * random.uniform(0.9, 1.1),
            admin_time_hours_per_week=14.8 * random.uniform(0.85, 1.15),
            avg_touchpoint_interval_hours=52.4 * random.uniform(0.8, 1.2),
            nba_adoption_rate=random.uniform(0.5, 0.9),
            nba_modification_rate=random.uniform(0.1, 0.3)
        )


class OpportunityStage(Enum):
    """Opportunity stages with progression."""
    PROSPECTING = 0
    QUALIFICATION = 1
    NEEDS_ANALYSIS = 2
    VALUE_PROPOSITION = 3
    NEGOTIATION = 4
    CLOSED_WON = 5
    CLOSED_LOST = 6


@dataclass
class SimulatedOpportunity:
    """
    A simulated sales opportunity.

    Tracks:
    - Stage progression
    - Timing metrics
    - Interaction history
    - NBA recommendations and outcomes
    """
    id: UUID = field(default_factory=uuid4)
    seller_id: UUID = field(default_factory=uuid4)
    account_name: str = ""

    # Current state
    stage: OpportunityStage = OpportunityStage.PROSPECTING
    amount: float = 50000.0
    probability: float = 0.1

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    expected_close_date: datetime = None
    actual_close_date: Optional[datetime] = None

    # Stage tracking
    stage_entered_at: datetime = field(default_factory=datetime.now)
    days_in_current_stage: int = 0
    stage_history: list = field(default_factory=list)

    # Interaction tracking
    last_interaction_at: Optional[datetime] = None
    interaction_count: int = 0
    hours_since_last_interaction: float = 0.0

    # NBA tracking
    nbas_received: int = 0
    nbas_accepted: int = 0
    nbas_modified: int = 0
    nbas_rejected: int = 0

    # Outcome
    is_closed: bool = False
    is_won: bool = False
    slippage_days: int = 0  # Days pushed from original close date

    def __post_init__(self):
        if self.expected_close_date is None:
            self.expected_close_date = self.created_at + timedelta(days=94)

    def advance_stage(self, new_stage: OpportunityStage, timestamp: datetime = None) -> None:
        """Advance to a new stage."""
        timestamp = timestamp or datetime.now()

        self.stage_history.append({
            "from_stage": self.stage.name,
            "to_stage": new_stage.name,
            "timestamp": timestamp.isoformat(),
            "days_in_stage": self.days_in_current_stage
        })

        self.stage = new_stage
        self.stage_entered_at = timestamp
        self.days_in_current_stage = 0

        # Update probability based on stage
        stage_probabilities = {
            OpportunityStage.PROSPECTING: 0.1,
            OpportunityStage.QUALIFICATION: 0.2,
            OpportunityStage.NEEDS_ANALYSIS: 0.4,
            OpportunityStage.VALUE_PROPOSITION: 0.6,
            OpportunityStage.NEGOTIATION: 0.8,
            OpportunityStage.CLOSED_WON: 1.0,
            OpportunityStage.CLOSED_LOST: 0.0
        }
        self.probability = stage_probabilities.get(new_stage, self.probability)

        if new_stage in [OpportunityStage.CLOSED_WON, OpportunityStage.CLOSED_LOST]:
            self.is_closed = True
            self.is_won = new_stage == OpportunityStage.CLOSED_WON
            self.actual_close_date = timestamp

    @classmethod
    def generate(
        cls,
        seller_id: UUID,
        start_date: datetime,
        amount_range: tuple = (10000, 200000)
    ) -> "SimulatedOpportunity":
        """Generate a random opportunity."""
        amount = random.uniform(*amount_range)

        return cls(
            seller_id=seller_id,
            account_name=f"Account_{uuid4().hex[:6]}",
            amount=amount,
            created_at=start_date,
            expected_close_date=start_date + timedelta(days=random.randint(60, 120)),
            last_interaction_at=start_date
        )


class InteractionType(Enum):
    """Types of interactions."""
    EMAIL = "email"
    CALL = "call"
    MEETING = "meeting"


@dataclass
class SimulatedInteraction:
    """
    A simulated customer interaction.

    Used to track:
    - Time between touchpoints
    - Channel distribution
    - Interaction outcomes
    """
    id: UUID = field(default_factory=uuid4)
    opportunity_id: UUID = field(default_factory=uuid4)
    seller_id: UUID = field(default_factory=uuid4)

    interaction_type: InteractionType = InteractionType.EMAIL
    timestamp: datetime = field(default_factory=datetime.now)
    duration_minutes: int = 0

    # Content (for simulation purposes)
    sentiment: float = 0.5  # -1 to 1
    engagement_score: float = 0.5  # 0 to 1

    # Processing
    was_processed_by_ai: bool = False
    processing_time_seconds: float = 0.0

    # Follow-up
    triggered_nba: bool = False
    admin_time_saved_minutes: float = 0.0

    @classmethod
    def generate(
        cls,
        opportunity_id: UUID,
        seller_id: UUID,
        timestamp: datetime,
        ai_enabled: bool = False
    ) -> "SimulatedInteraction":
        """Generate a random interaction."""
        interaction_type = random.choice(list(InteractionType))

        duration_by_type = {
            InteractionType.EMAIL: random.randint(5, 15),
            InteractionType.CALL: random.randint(15, 45),
            InteractionType.MEETING: random.randint(30, 60)
        }

        interaction = cls(
            opportunity_id=opportunity_id,
            seller_id=seller_id,
            interaction_type=interaction_type,
            timestamp=timestamp,
            duration_minutes=duration_by_type[interaction_type],
            sentiment=random.uniform(-0.3, 0.8),
            engagement_score=random.uniform(0.3, 0.9),
            was_processed_by_ai=ai_enabled
        )

        if ai_enabled:
            interaction.processing_time_seconds = random.uniform(2, 10)
            interaction.triggered_nba = random.random() > 0.3
            interaction.admin_time_saved_minutes = random.uniform(10, 30)

        return interaction


class NBAOutcome(Enum):
    """Outcome of NBA recommendation."""
    ACCEPTED = "accepted"
    MODIFIED = "modified"
    REJECTED = "rejected"
    IGNORED = "ignored"


@dataclass
class SimulatedNBA:
    """
    A simulated Next-Best-Action recommendation.

    Tracks:
    - Recommendation content
    - Seller response
    - Downstream outcome
    """
    id: UUID = field(default_factory=uuid4)
    opportunity_id: UUID = field(default_factory=uuid4)
    seller_id: UUID = field(default_factory=uuid4)

    # Recommendation
    action_type: str = "follow_up"
    confidence_score: float = 0.8
    timestamp: datetime = field(default_factory=datetime.now)

    # Evidence
    evidence_sources: list = field(default_factory=list)
    reasoning: str = ""

    # Outcome
    outcome: NBAOutcome = NBAOutcome.ACCEPTED
    time_to_decision_minutes: int = 0
    modification_notes: str = ""

    # Impact
    led_to_stage_advance: bool = False
    led_to_response: bool = False

    @classmethod
    def generate(
        cls,
        opportunity_id: UUID,
        seller_id: UUID,
        timestamp: datetime,
        seller_adoption_rate: float = 0.7
    ) -> "SimulatedNBA":
        """Generate a simulated NBA."""
        action_types = ["follow_up", "schedule_demo", "send_proposal", "address_objection", "escalate"]
        action_type = random.choice(action_types)

        # Determine outcome based on seller adoption
        r = random.random()
        if r < seller_adoption_rate:
            outcome = NBAOutcome.ACCEPTED
        elif r < seller_adoption_rate + 0.15:
            outcome = NBAOutcome.MODIFIED
        elif r < seller_adoption_rate + 0.25:
            outcome = NBAOutcome.REJECTED
        else:
            outcome = NBAOutcome.IGNORED

        return cls(
            opportunity_id=opportunity_id,
            seller_id=seller_id,
            action_type=action_type,
            confidence_score=random.uniform(0.65, 0.95),
            timestamp=timestamp,
            outcome=outcome,
            time_to_decision_minutes=random.randint(5, 60),
            led_to_stage_advance=outcome == NBAOutcome.ACCEPTED and random.random() > 0.7,
            led_to_response=outcome in [NBAOutcome.ACCEPTED, NBAOutcome.MODIFIED] and random.random() > 0.4
        )
