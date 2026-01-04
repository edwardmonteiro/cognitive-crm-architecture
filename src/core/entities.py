"""
Core CRM Entities - System of Record

This module defines the fundamental entities that form the governed
system of record. Following the architecture, CRM remains the source
of truth while the intelligence layer operates on representations
of these entities.

Entities:
- Account: Company/organization being sold to
- Contact: Individual stakeholder within an account
- Opportunity: Sales opportunity with stage progression
- Interaction: Any touchpoint (call, email, meeting, etc.)
- Activity: Actions taken by sellers or the system
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class OpportunityStage(Enum):
    """
    Sales opportunity stages representing pipeline progression.
    Stage-to-stage conversion is a key metric in the architecture.
    """
    PROSPECTING = "prospecting"
    QUALIFICATION = "qualification"
    NEEDS_ANALYSIS = "needs_analysis"
    VALUE_PROPOSITION = "value_proposition"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class InteractionChannel(Enum):
    """
    Multi-channel interaction sources.
    The architecture emphasizes cognitive integration across
    fragmented customer signals from various channels.
    """
    EMAIL = "email"
    CALL = "call"
    MEETING = "meeting"
    CHAT = "chat"
    VIDEO_CONFERENCE = "video_conference"
    IN_PERSON = "in_person"
    DOCUMENT = "document"
    SOCIAL = "social"


class StakeholderRole(Enum):
    """
    Stakeholder roles within buying committees.
    Multi-stakeholder mapping is a key intelligence function.
    """
    DECISION_MAKER = "decision_maker"
    INFLUENCER = "influencer"
    CHAMPION = "champion"
    BLOCKER = "blocker"
    END_USER = "end_user"
    ECONOMIC_BUYER = "economic_buyer"
    TECHNICAL_BUYER = "technical_buyer"


@dataclass
class Account:
    """
    Represents a company or organization in the CRM.

    Accounts aggregate opportunities and contacts, enabling
    organizational-level intelligence and pattern recognition.
    """
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    industry: str = ""
    size: str = ""  # SMB, MidMarket, Enterprise
    annual_revenue: Optional[float] = None
    employee_count: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Metadata for identity resolution
    external_ids: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.id, str):
            self.id = UUID(self.id)


@dataclass
class Contact:
    """
    Represents an individual stakeholder within an account.

    The architecture emphasizes stakeholder mapping to understand
    buying committee dynamics and influence patterns.
    """
    id: UUID = field(default_factory=uuid4)
    account_id: UUID = field(default_factory=uuid4)
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    title: str = ""
    department: str = ""
    role: Optional[StakeholderRole] = None
    influence_score: float = 0.0  # 0.0 to 1.0
    engagement_score: float = 0.0  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


@dataclass
class Opportunity:
    """
    Represents a sales opportunity with stage progression.

    Key metrics tracked:
    - Stage-to-stage conversion rates
    - Sales cycle duration
    - Forecast accuracy
    - Time between touchpoints (execution latency)
    """
    id: UUID = field(default_factory=uuid4)
    account_id: UUID = field(default_factory=uuid4)
    name: str = ""
    stage: OpportunityStage = OpportunityStage.PROSPECTING
    amount: float = 0.0
    probability: float = 0.0  # 0.0 to 1.0
    close_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Stage progression tracking
    stage_history: list = field(default_factory=list)

    # Risk and health indicators
    risk_score: float = 0.0  # 0.0 to 1.0
    health_score: float = 0.5  # 0.0 to 1.0

    # Execution metrics
    last_activity_date: Optional[datetime] = None
    days_in_stage: int = 0

    def advance_stage(self, new_stage: OpportunityStage) -> None:
        """Record stage transition for conversion analysis."""
        self.stage_history.append({
            "from_stage": self.stage.value,
            "to_stage": new_stage.value,
            "timestamp": datetime.now().isoformat(),
            "days_in_previous_stage": self.days_in_stage
        })
        self.stage = new_stage
        self.days_in_stage = 0
        self.updated_at = datetime.now()


@dataclass
class Interaction:
    """
    Represents any customer touchpoint across channels.

    Interactions are the primary input for the intelligence layer,
    containing unstructured signals that require sensemaking.

    The architecture measures "time between touchpoints" as a key
    execution latency metric.
    """
    id: UUID = field(default_factory=uuid4)
    opportunity_id: UUID = field(default_factory=uuid4)
    contact_id: Optional[UUID] = None
    channel: InteractionChannel = InteractionChannel.EMAIL
    direction: str = "outbound"  # inbound, outbound
    timestamp: datetime = field(default_factory=datetime.now)
    duration_minutes: Optional[int] = None

    # Content for intelligence layer processing
    subject: str = ""
    content: str = ""  # Raw content (transcript, email body, notes)

    # Intelligence layer outputs (populated by processing)
    summary: Optional[str] = None
    sentiment: Optional[float] = None  # -1.0 to 1.0
    key_topics: list = field(default_factory=list)
    objections_raised: list = field(default_factory=list)
    action_items: list = field(default_factory=list)

    # Embedding for semantic retrieval (RAG)
    embedding: Optional[list] = None

    # Processing status
    processed: bool = False
    processed_at: Optional[datetime] = None


@dataclass
class Activity:
    """
    Represents actions taken by sellers or the system.

    Activities can be:
    - Human-initiated: Manual actions by sellers
    - System-recommended: NBA suggestions
    - System-executed: Automated playbook actions (with approval)

    This supports the closed-loop learning mechanism.
    """
    id: UUID = field(default_factory=uuid4)
    opportunity_id: UUID = field(default_factory=uuid4)
    activity_type: str = ""  # follow_up, proposal, demo, etc.
    description: str = ""

    # Source tracking
    source: str = "human"  # human, system_recommended, system_executed
    recommendation_id: Optional[UUID] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Outcome tracking for learning
    status: str = "pending"  # pending, completed, cancelled
    outcome: Optional[str] = None
    outcome_signal: Optional[str] = None  # positive, negative, neutral

    # Human feedback for learning loop
    human_modified: bool = False
    modification_reason: Optional[str] = None


@dataclass
class StageTransition:
    """
    Explicit record of opportunity stage changes.

    Used for:
    - Conversion rate analysis
    - Outcome signals for learning
    - Forecast model calibration
    """
    id: UUID = field(default_factory=uuid4)
    opportunity_id: UUID = field(default_factory=uuid4)
    from_stage: OpportunityStage = OpportunityStage.PROSPECTING
    to_stage: OpportunityStage = OpportunityStage.QUALIFICATION
    timestamp: datetime = field(default_factory=datetime.now)
    days_in_previous_stage: int = 0
    triggered_by: str = "human"  # human, system

    @property
    def is_advancement(self) -> bool:
        """Check if this represents forward pipeline progression."""
        stage_order = list(OpportunityStage)
        return stage_order.index(self.to_stage) > stage_order.index(self.from_stage)
