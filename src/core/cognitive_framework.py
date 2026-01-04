"""
Cognitive Framework - Theoretical Foundations

This module implements the theoretical framework from Section 3 of the paper,
operationalizing organizational cognition through:

1. Sensemaking (Weick, 1995): Constructing meaning from ambiguous cues
2. Distributed Cognition (Hutchins, 1995): Cognition across people and artifacts
3. Absorptive Capacity (Cohen & Levinthal, 1990): Recognizing and applying information

The cognitive CRM architecture operationalizes:
(i)   Sensing and representation
(ii)  Inference and sensemaking
(iii) Action orchestration
(iv)  Feedback-driven learning
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4


class CognitiveFunction(Enum):
    """
    The four cognitive functions that the architecture operationalizes.
    These map to the layers in the system architecture.
    """
    SENSING = "sensing"           # Data ingestion & representation
    SENSEMAKING = "sensemaking"   # Intelligence layer inference
    ORCHESTRATION = "orchestration"  # Action coordination
    LEARNING = "learning"         # Feedback-driven adaptation


class MemoryType(Enum):
    """
    Memory design from Section 4.2.

    The architecture distinguishes between:
    - Working memory: Short-term, per-opportunity context
    - Episodic memory: Long-term interaction history
    - Policy memory: Constraints and governance rules
    """
    WORKING = "working"      # Short-term, opportunity-specific
    EPISODIC = "episodic"    # Long-term, interaction history
    POLICY = "policy"        # Governance constraints


@dataclass
class CognitiveSignal:
    """
    Represents a signal in the cognitive processing pipeline.

    Signals flow through the architecture:
    Sensing → Sensemaking → Orchestration → Action → Learning

    This is the fundamental unit of information flow.
    """
    id: UUID = field(default_factory=uuid4)
    source: str = ""                  # Origin system/channel
    signal_type: str = ""             # Classification of signal
    content: Any = None               # Raw signal content
    timestamp: datetime = field(default_factory=datetime.now)

    # Context for sensemaking
    context: dict = field(default_factory=dict)

    # Processing metadata
    confidence: float = 1.0           # 0.0 to 1.0
    requires_attention: bool = False
    priority: int = 0                 # Higher = more urgent

    # Traceability
    processed_by: list = field(default_factory=list)


@dataclass
class SensemakingResult:
    """
    Output of the sensemaking process.

    Following Weick (1995), sensemaking involves:
    - Noticing: What signals are relevant
    - Interpreting: What do they mean
    - Acting: What should be done

    This structure captures the interpretation phase.
    """
    id: UUID = field(default_factory=uuid4)
    signal_ids: list = field(default_factory=list)  # Input signals
    timestamp: datetime = field(default_factory=datetime.now)

    # Interpretation
    summary: str = ""
    key_insights: list = field(default_factory=list)
    detected_patterns: list = field(default_factory=list)

    # Risk and opportunity assessment
    risks_identified: list = field(default_factory=list)
    opportunities_identified: list = field(default_factory=list)

    # Recommended actions
    suggested_actions: list = field(default_factory=list)

    # Confidence and evidence
    confidence: float = 0.0
    evidence_sources: list = field(default_factory=list)

    # Grounding in governed sources (RAG)
    grounding_documents: list = field(default_factory=list)


@dataclass
class DistributedCognitionState:
    """
    Represents the distributed cognitive state across the system.

    Following Hutchins (1995), cognition is distributed across:
    - Individuals (sellers, managers)
    - Artifacts (CRM, documents, tools)
    - Representations (embeddings, summaries, dashboards)

    This structure tracks the cognitive load distribution.
    """
    id: UUID = field(default_factory=uuid4)
    opportunity_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)

    # Cognitive artifacts in use
    active_artifacts: list = field(default_factory=list)

    # Information distribution
    human_held_knowledge: dict = field(default_factory=dict)
    system_held_knowledge: dict = field(default_factory=dict)

    # Coordination state
    pending_decisions: list = field(default_factory=list)
    delegated_to_system: list = field(default_factory=list)
    requiring_human_input: list = field(default_factory=list)

    # Load metrics (proxy for cognitive load theory, Sweller 1988)
    estimated_cognitive_load: float = 0.0  # 0.0 to 1.0


@dataclass
class AbsorptiveCapacityMetrics:
    """
    Metrics for organizational absorptive capacity.

    Following Cohen & Levinthal (1990), absorptive capacity is the
    ability to recognize, assimilate, and apply new information.

    The cognitive CRM aims to enhance this by:
    - Capturing successful patterns
    - Making knowledge reusable
    - Reducing individual variance
    """
    # Recognition: Ability to identify valuable new information
    signal_recognition_rate: float = 0.0
    false_positive_rate: float = 0.0

    # Assimilation: Ability to integrate new information
    pattern_reuse_rate: float = 0.0
    knowledge_sharing_index: float = 0.0

    # Application: Ability to apply knowledge to actions
    recommendation_acceptance_rate: float = 0.0
    outcome_improvement_rate: float = 0.0

    # Variance reduction (key architecture goal)
    cross_seller_variance: float = 0.0
    execution_consistency: float = 0.0


class CognitiveProcessor(ABC):
    """
    Abstract base class for cognitive processing components.

    All layers in the architecture implement this interface:
    - Data Ingestion Layer
    - Intelligence Layer
    - Orchestration Layer
    - Learning Layer
    """

    @property
    @abstractmethod
    def cognitive_function(self) -> CognitiveFunction:
        """Which cognitive function this processor implements."""
        pass

    @abstractmethod
    def process(self, signal: CognitiveSignal) -> CognitiveSignal:
        """Process a signal through this cognitive layer."""
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """Return current processing state for observability."""
        pass


class MemoryStore(ABC):
    """
    Abstract interface for memory systems.

    The architecture defines three memory types:
    - Working memory: Per-opportunity context
    - Episodic memory: Historical interactions
    - Policy memory: Governance constraints
    """

    @property
    @abstractmethod
    def memory_type(self) -> MemoryType:
        """Type of memory this store implements."""
        pass

    @abstractmethod
    def store(self, key: str, value: Any, metadata: dict = None) -> None:
        """Store information in memory."""
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve information from memory."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list:
        """Semantic search over memory contents."""
        pass


@dataclass
class GovernanceContext:
    """
    Context for governance and human oversight.

    From Section 4.3 and 4.5:
    - Human approval gates
    - Audit logs
    - Role-based permissions
    - Contestability

    This enables trust and accountability.
    """
    user_id: UUID = field(default_factory=uuid4)
    role: str = "seller"
    permissions: list = field(default_factory=list)

    # Approval context
    approval_required: bool = False
    approval_threshold: float = 0.8  # Confidence below this requires approval

    # Audit
    audit_enabled: bool = True
    session_id: UUID = field(default_factory=uuid4)

    # Contestability
    can_override: bool = True
    override_requires_reason: bool = True


@dataclass
class ExecutionLatencyMetric:
    """
    Key metric: Time between sensing and acting.

    From Section 3: "Execution latency (e.g., time between touchpoints)
    operationalises delays between sensemaking and action."

    This is the fundamental measure of cognitive efficiency.
    """
    opportunity_id: UUID = field(default_factory=uuid4)

    # Timing measurements
    signal_received_at: datetime = field(default_factory=datetime.now)
    sensemaking_completed_at: Optional[datetime] = None
    action_recommended_at: Optional[datetime] = None
    action_executed_at: Optional[datetime] = None

    # Computed latencies (in hours)
    @property
    def sensing_to_sensemaking_hours(self) -> Optional[float]:
        if self.sensemaking_completed_at:
            delta = self.sensemaking_completed_at - self.signal_received_at
            return delta.total_seconds() / 3600
        return None

    @property
    def sensemaking_to_action_hours(self) -> Optional[float]:
        if self.action_recommended_at and self.sensemaking_completed_at:
            delta = self.action_recommended_at - self.sensemaking_completed_at
            return delta.total_seconds() / 3600
        return None

    @property
    def total_latency_hours(self) -> Optional[float]:
        if self.action_executed_at:
            delta = self.action_executed_at - self.signal_received_at
            return delta.total_seconds() / 3600
        return None
