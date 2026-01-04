"""
Playbooks - Automated Workflow Sequences

From Section 4.3: "Playbooks: post-call update, follow-up drafting,
NBA recommendations, deal-risk alerts"

Playbooks are composable workflow templates that:
- Define sequences of actions
- Integrate with reasoning and memory
- Respect governance constraints
- Support customization per deal stage
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from uuid import UUID, uuid4


class PlaybookStatus(Enum):
    """Status of playbook execution."""
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionType(Enum):
    """Types of actions in playbooks."""
    ANALYZE = "analyze"           # Run reasoning task
    RETRIEVE = "retrieve"         # RAG retrieval
    GENERATE = "generate"         # Generate content
    UPDATE_CRM = "update_crm"     # Update CRM record
    SEND_NOTIFICATION = "send_notification"
    CREATE_TASK = "create_task"
    SEND_EMAIL = "send_email"
    SCHEDULE_MEETING = "schedule_meeting"
    ESCALATE = "escalate"


@dataclass
class PlaybookStep:
    """A single step in a playbook."""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    action_type: ActionType = ActionType.ANALYZE
    parameters: dict = field(default_factory=dict)

    # Conditions
    condition: Optional[Callable] = None
    skip_on_failure: bool = False

    # Approval
    requires_approval: bool = False
    approval_reason: str = ""

    # Output handling
    output_key: str = ""  # Key to store output in context


@dataclass
class PlaybookContext:
    """Context passed through playbook execution."""
    playbook_id: UUID = field(default_factory=uuid4)
    opportunity_id: UUID = field(default_factory=uuid4)
    account_id: Optional[UUID] = None

    # Input data
    trigger_data: dict = field(default_factory=dict)

    # Accumulated outputs
    outputs: dict = field(default_factory=dict)

    # Execution state
    current_step: int = 0
    status: PlaybookStatus = PlaybookStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error handling
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # Governance
    user_id: Optional[str] = None
    approval_pending: Optional[UUID] = None


@dataclass
class PlaybookResult:
    """Result of playbook execution."""
    id: UUID = field(default_factory=uuid4)
    playbook_name: str = ""
    status: PlaybookStatus = PlaybookStatus.COMPLETED
    context: PlaybookContext = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Outputs
    outputs: dict = field(default_factory=dict)
    actions_taken: list = field(default_factory=list)

    # Governance
    approvals_requested: int = 0
    approvals_received: int = 0


class Playbook(ABC):
    """
    Abstract base class for playbooks.

    Playbooks are declarative workflow definitions that:
    - Specify steps and conditions
    - Handle errors gracefully
    - Support human-in-the-loop
    """

    def __init__(self):
        self._steps: list[PlaybookStep] = []
        self._setup_steps()

    @property
    @abstractmethod
    def name(self) -> str:
        """Playbook name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Playbook description."""
        pass

    @property
    def trigger_events(self) -> list[str]:
        """Events that trigger this playbook."""
        return []

    @abstractmethod
    def _setup_steps(self) -> None:
        """Define playbook steps."""
        pass

    def add_step(
        self,
        name: str,
        action_type: ActionType,
        parameters: dict = None,
        requires_approval: bool = False,
        condition: Callable = None,
        output_key: str = None
    ) -> None:
        """Add a step to the playbook."""
        step = PlaybookStep(
            name=name,
            action_type=action_type,
            parameters=parameters or {},
            requires_approval=requires_approval,
            condition=condition,
            output_key=output_key or name.lower().replace(" ", "_")
        )
        self._steps.append(step)

    @property
    def steps(self) -> list[PlaybookStep]:
        """Get all steps."""
        return self._steps

    def should_run(self, context: PlaybookContext) -> bool:
        """Check if playbook should run given context."""
        return True


class PostCallPlaybook(Playbook):
    """
    Post-call intelligence and CRM update playbook.

    From Section 5.1: "Use Case 1: Post-call intelligence and
    CRM update (human-reviewed)"

    Steps:
    1. Summarize call transcript
    2. Extract objections
    3. Map stakeholders
    4. Detect risks
    5. Generate CRM update
    6. Request approval
    7. Update CRM
    """

    @property
    def name(self) -> str:
        return "post_call_intelligence"

    @property
    def description(self) -> str:
        return "Analyze call transcript and update CRM with insights"

    @property
    def trigger_events(self) -> list[str]:
        return ["call_ended", "transcript_ready"]

    def _setup_steps(self) -> None:
        # Step 1: Summarize
        self.add_step(
            name="Summarize Call",
            action_type=ActionType.ANALYZE,
            parameters={
                "task_type": "summarization",
                "input_key": "transcript"
            },
            output_key="summary"
        )

        # Step 2: Extract objections
        self.add_step(
            name="Extract Objections",
            action_type=ActionType.ANALYZE,
            parameters={
                "task_type": "objection_extraction",
                "input_key": "transcript"
            },
            output_key="objections"
        )

        # Step 3: Map stakeholders
        self.add_step(
            name="Map Stakeholders",
            action_type=ActionType.ANALYZE,
            parameters={
                "task_type": "stakeholder_mapping",
                "input_key": "transcript"
            },
            output_key="stakeholders"
        )

        # Step 4: Detect risks
        self.add_step(
            name="Detect Risks",
            action_type=ActionType.ANALYZE,
            parameters={
                "task_type": "risk_detection",
                "input_key": "transcript"
            },
            output_key="risks"
        )

        # Step 5: Generate CRM update
        self.add_step(
            name="Generate CRM Update",
            action_type=ActionType.GENERATE,
            parameters={
                "template": "crm_update",
                "inputs": ["summary", "objections", "stakeholders", "risks"]
            },
            output_key="crm_update"
        )

        # Step 6: Request approval
        self.add_step(
            name="Request Approval",
            action_type=ActionType.UPDATE_CRM,
            parameters={
                "update_type": "opportunity_notes",
                "data_key": "crm_update"
            },
            requires_approval=True,
            output_key="crm_update_result"
        )


class FollowUpPlaybook(Playbook):
    """
    Follow-up drafting playbook.

    Generates contextual follow-up communications
    based on recent interactions and deal state.
    """

    @property
    def name(self) -> str:
        return "follow_up_drafting"

    @property
    def description(self) -> str:
        return "Draft personalized follow-up communications"

    @property
    def trigger_events(self) -> list[str]:
        return ["meeting_ended", "email_received", "days_since_contact"]

    def _setup_steps(self) -> None:
        # Step 1: Retrieve context
        self.add_step(
            name="Retrieve Context",
            action_type=ActionType.RETRIEVE,
            parameters={
                "sources": ["interaction_history", "sales_playbook"],
                "query_key": "trigger_summary"
            },
            output_key="context"
        )

        # Step 2: Plan next steps
        self.add_step(
            name="Plan Next Steps",
            action_type=ActionType.ANALYZE,
            parameters={
                "task_type": "next_step_planning",
                "context_key": "context"
            },
            output_key="next_steps"
        )

        # Step 3: Generate email draft
        self.add_step(
            name="Generate Follow-up Email",
            action_type=ActionType.GENERATE,
            parameters={
                "template": "follow_up_email",
                "inputs": ["context", "next_steps"]
            },
            output_key="email_draft"
        )

        # Step 4: Create task with draft
        self.add_step(
            name="Create Follow-up Task",
            action_type=ActionType.CREATE_TASK,
            parameters={
                "task_type": "send_follow_up",
                "include_draft": True
            },
            requires_approval=True,
            output_key="task_created"
        )


class RiskAlertPlaybook(Playbook):
    """
    Deal risk alert playbook.

    Monitors for risk signals and escalates
    to appropriate stakeholders.
    """

    @property
    def name(self) -> str:
        return "risk_alert"

    @property
    def description(self) -> str:
        return "Detect and alert on deal risks"

    @property
    def trigger_events(self) -> list[str]:
        return ["daily_pipeline_sweep", "significant_event"]

    def _setup_steps(self) -> None:
        # Step 1: Analyze risks
        self.add_step(
            name="Analyze Deal Risks",
            action_type=ActionType.ANALYZE,
            parameters={
                "task_type": "risk_detection",
                "include_history": True
            },
            output_key="risk_analysis"
        )

        # Step 2: Check severity
        self.add_step(
            name="Evaluate Severity",
            action_type=ActionType.ANALYZE,
            parameters={
                "task_type": "severity_evaluation",
                "input_key": "risk_analysis"
            },
            condition=lambda ctx: len(ctx.outputs.get("risk_analysis", {}).get("risks", [])) > 0,
            output_key="severity"
        )

        # Step 3: Generate alert
        self.add_step(
            name="Generate Alert",
            action_type=ActionType.GENERATE,
            parameters={
                "template": "risk_alert",
                "inputs": ["risk_analysis", "severity"]
            },
            condition=lambda ctx: ctx.outputs.get("severity", {}).get("level") in ["high", "critical"],
            output_key="alert"
        )

        # Step 4: Send notification
        self.add_step(
            name="Send Alert Notification",
            action_type=ActionType.SEND_NOTIFICATION,
            parameters={
                "channel": "slack",
                "recipients_from": "escalation_list",
                "content_key": "alert"
            },
            condition=lambda ctx: ctx.outputs.get("alert") is not None,
            output_key="notification_sent"
        )

        # Step 5: Escalate if critical
        self.add_step(
            name="Escalate to Manager",
            action_type=ActionType.ESCALATE,
            parameters={
                "escalation_type": "manager_review",
                "context_keys": ["risk_analysis", "alert"]
            },
            condition=lambda ctx: ctx.outputs.get("severity", {}).get("level") == "critical",
            output_key="escalation"
        )


class NBAPlaybook(Playbook):
    """
    Next-Best-Action recommendation playbook.

    From Section 5.1: "Use Case 2: Evidence-based next-best-action
    (NBA) recommendations with justification"
    """

    @property
    def name(self) -> str:
        return "next_best_action"

    @property
    def description(self) -> str:
        return "Generate evidence-based next-best-action recommendations"

    @property
    def trigger_events(self) -> list[str]:
        return ["interaction_completed", "stage_change", "scheduled_review"]

    def _setup_steps(self) -> None:
        # Step 1: Gather context
        self.add_step(
            name="Gather Opportunity Context",
            action_type=ActionType.RETRIEVE,
            parameters={
                "sources": ["interaction_history", "working_memory"],
                "include_stakeholders": True,
                "include_risks": True
            },
            output_key="full_context"
        )

        # Step 2: Retrieve relevant playbook guidance
        self.add_step(
            name="Retrieve Playbook Guidance",
            action_type=ActionType.RETRIEVE,
            parameters={
                "sources": ["sales_playbook"],
                "filter_by_stage": True
            },
            output_key="playbook_guidance"
        )

        # Step 3: Generate recommendations
        self.add_step(
            name="Generate NBA Recommendations",
            action_type=ActionType.ANALYZE,
            parameters={
                "task_type": "next_step_planning",
                "context_keys": ["full_context", "playbook_guidance"]
            },
            output_key="recommendations"
        )

        # Step 4: Add evidence justification
        self.add_step(
            name="Add Evidence Justification",
            action_type=ActionType.GENERATE,
            parameters={
                "template": "nba_justification",
                "inputs": ["recommendations", "full_context"]
            },
            output_key="justified_recommendations"
        )

        # Step 5: Present to seller
        self.add_step(
            name="Present Recommendations",
            action_type=ActionType.SEND_NOTIFICATION,
            parameters={
                "channel": "in_app",
                "content_key": "justified_recommendations",
                "action_required": True
            },
            output_key="presentation"
        )


class PlaybookRegistry:
    """
    Registry of available playbooks.

    Provides:
    - Playbook registration and lookup
    - Event-based playbook matching
    - Playbook versioning
    """

    def __init__(self):
        self._playbooks: dict[str, Playbook] = {}
        self._event_map: dict[str, list[str]] = {}

    def register(self, playbook: Playbook) -> None:
        """Register a playbook."""
        self._playbooks[playbook.name] = playbook

        for event in playbook.trigger_events:
            if event not in self._event_map:
                self._event_map[event] = []
            self._event_map[event].append(playbook.name)

    def get(self, name: str) -> Optional[Playbook]:
        """Get playbook by name."""
        return self._playbooks.get(name)

    def get_for_event(self, event: str) -> list[Playbook]:
        """Get playbooks triggered by an event."""
        names = self._event_map.get(event, [])
        return [self._playbooks[n] for n in names if n in self._playbooks]

    def list_all(self) -> list[str]:
        """List all registered playbook names."""
        return list(self._playbooks.keys())

    @classmethod
    def create_default(cls) -> "PlaybookRegistry":
        """Create registry with default playbooks."""
        registry = cls()
        registry.register(PostCallPlaybook())
        registry.register(FollowUpPlaybook())
        registry.register(RiskAlertPlaybook())
        registry.register(NBAPlaybook())
        return registry
