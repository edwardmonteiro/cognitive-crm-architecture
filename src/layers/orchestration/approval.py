"""
Approval Gates - Human-in-the-Loop Governance

From Section 4.3: "Human approval gates: approve/edit/reject actions;
capture rationales for governance and learning"

Approval gates ensure:
- Human oversight for high-impact actions
- Audit trail for compliance
- Learning from human corrections
- Trust through transparency
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from uuid import UUID, uuid4


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


class ApprovalPriority(Enum):
    """Priority of approval request."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ApprovalCategory(Enum):
    """Category of approval request."""
    CRM_UPDATE = "crm_update"
    EMAIL_SEND = "email_send"
    PROPOSAL = "proposal"
    DISCOUNT = "discount"
    ESCALATION = "escalation"
    RISK_ALERT = "risk_alert"
    CONTRACT = "contract"


@dataclass
class ApprovalRequest:
    """
    A request for human approval.

    Captures all context needed for the approver
    to make an informed decision.
    """
    id: UUID = field(default_factory=uuid4)
    category: ApprovalCategory = ApprovalCategory.CRM_UPDATE
    priority: ApprovalPriority = ApprovalPriority.NORMAL
    status: ApprovalStatus = ApprovalStatus.PENDING

    # What needs approval
    action_type: str = ""
    action_description: str = ""
    proposed_content: Any = None

    # Context
    opportunity_id: Optional[UUID] = None
    account_id: Optional[UUID] = None
    playbook_name: str = ""
    step_name: str = ""

    # Evidence and reasoning
    reasoning: str = ""
    evidence: list = field(default_factory=list)
    confidence_score: float = 0.0

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None

    # Assignee
    assigned_to: Optional[str] = None
    created_by: str = "system"

    # For audit
    source_event_id: Optional[UUID] = None
    correlation_id: Optional[UUID] = None

    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class ApprovalDecision:
    """
    Record of an approval decision.

    Captures the human decision and any modifications,
    used for governance and learning.
    """
    id: UUID = field(default_factory=uuid4)
    request_id: UUID = field(default_factory=uuid4)
    status: ApprovalStatus = ApprovalStatus.APPROVED

    # Decision details
    decided_by: str = ""
    decided_at: datetime = field(default_factory=datetime.now)

    # Modifications
    modified_content: Optional[Any] = None
    modification_notes: str = ""

    # Rationale (critical for learning)
    rationale: str = ""
    rejection_reason: str = ""

    # Feedback for learning loop
    feedback_tags: list = field(default_factory=list)  # e.g., ["tone_adjustment", "factual_correction"]
    improvement_suggestions: str = ""

    # Time spent (for admin effort metrics)
    review_duration_seconds: int = 0


@dataclass
class ApprovalGate:
    """
    Configuration for an approval gate.

    Defines when approval is required and
    who can provide it.
    """
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    category: ApprovalCategory = ApprovalCategory.CRM_UPDATE

    # When approval is required
    always_required: bool = False
    confidence_threshold: float = 0.8  # Require approval if confidence below
    value_threshold: Optional[float] = None  # Require approval if value above
    custom_condition: Optional[Callable[[dict], bool]] = None

    # Who can approve
    required_role: str = "seller"
    allow_self_approval: bool = True
    escalation_role: str = "manager"

    # Timing
    expiration_hours: int = 24
    auto_approve_on_expiry: bool = False

    # Notifications
    notify_on_create: bool = True
    notify_on_expire: bool = True
    notification_channel: str = "email"

    def requires_approval(self, context: dict) -> tuple[bool, str]:
        """
        Check if approval is required given context.

        Returns (requires_approval, reason).
        """
        if self.always_required:
            return True, "Always requires approval"

        if "confidence" in context:
            if context["confidence"] < self.confidence_threshold:
                return True, f"Confidence {context['confidence']:.2f} below threshold {self.confidence_threshold}"

        if self.value_threshold and "value" in context:
            if context["value"] > self.value_threshold:
                return True, f"Value ${context['value']} exceeds threshold ${self.value_threshold}"

        if self.custom_condition:
            if self.custom_condition(context):
                return True, "Custom condition triggered"

        return False, "No approval required"


class ApprovalEngine:
    """
    Engine for managing approval workflows.

    Responsibilities:
    - Create approval requests
    - Route to appropriate approvers
    - Track decisions
    - Aggregate feedback for learning
    """

    def __init__(self):
        self._gates: dict[str, ApprovalGate] = {}
        self._pending: dict[str, ApprovalRequest] = {}
        self._decisions: list[ApprovalDecision] = []
        self._callbacks: dict[str, Callable] = {}

    def register_gate(self, gate: ApprovalGate) -> None:
        """Register an approval gate."""
        self._gates[gate.name] = gate

    def check_approval_required(
        self,
        gate_name: str,
        context: dict
    ) -> tuple[bool, str]:
        """Check if approval is required."""
        gate = self._gates.get(gate_name)
        if not gate:
            return False, f"Gate {gate_name} not found"

        return gate.requires_approval(context)

    def create_request(
        self,
        gate_name: str,
        action_type: str,
        action_description: str,
        proposed_content: Any,
        context: dict,
        opportunity_id: UUID = None,
        playbook_name: str = "",
        step_name: str = ""
    ) -> ApprovalRequest:
        """Create an approval request."""
        gate = self._gates.get(gate_name)
        if not gate:
            raise ValueError(f"Gate {gate_name} not found")

        # Calculate expiration
        expires_at = datetime.now() + timedelta(hours=gate.expiration_hours)

        request = ApprovalRequest(
            category=gate.category,
            priority=self._determine_priority(context),
            action_type=action_type,
            action_description=action_description,
            proposed_content=proposed_content,
            opportunity_id=opportunity_id,
            playbook_name=playbook_name,
            step_name=step_name,
            reasoning=context.get("reasoning", ""),
            evidence=context.get("evidence", []),
            confidence_score=context.get("confidence", 0.0),
            expires_at=expires_at
        )

        self._pending[str(request.id)] = request

        # Notify if configured
        if gate.notify_on_create:
            self._notify_approver(request, gate)

        return request

    def _determine_priority(self, context: dict) -> ApprovalPriority:
        """Determine priority based on context."""
        if context.get("is_critical"):
            return ApprovalPriority.URGENT
        if context.get("value", 0) > 100000:
            return ApprovalPriority.HIGH
        if context.get("confidence", 1.0) < 0.5:
            return ApprovalPriority.HIGH
        return ApprovalPriority.NORMAL

    def _notify_approver(self, request: ApprovalRequest, gate: ApprovalGate) -> None:
        """Send notification to approver."""
        # In production, would integrate with notification system
        pass

    def approve(
        self,
        request_id: UUID,
        approved_by: str,
        rationale: str = "",
        review_duration_seconds: int = 0
    ) -> ApprovalDecision:
        """Approve a request."""
        request = self._pending.get(str(request_id))
        if not request:
            raise ValueError(f"Request {request_id} not found")

        decision = ApprovalDecision(
            request_id=request_id,
            status=ApprovalStatus.APPROVED,
            decided_by=approved_by,
            rationale=rationale,
            review_duration_seconds=review_duration_seconds
        )

        request.status = ApprovalStatus.APPROVED
        request.reviewed_at = datetime.now()

        self._decisions.append(decision)
        del self._pending[str(request_id)]

        # Execute callback if registered
        self._execute_callback(request_id, decision)

        return decision

    def reject(
        self,
        request_id: UUID,
        rejected_by: str,
        rejection_reason: str,
        feedback_tags: list = None,
        improvement_suggestions: str = "",
        review_duration_seconds: int = 0
    ) -> ApprovalDecision:
        """Reject a request."""
        request = self._pending.get(str(request_id))
        if not request:
            raise ValueError(f"Request {request_id} not found")

        decision = ApprovalDecision(
            request_id=request_id,
            status=ApprovalStatus.REJECTED,
            decided_by=rejected_by,
            rejection_reason=rejection_reason,
            feedback_tags=feedback_tags or [],
            improvement_suggestions=improvement_suggestions,
            review_duration_seconds=review_duration_seconds
        )

        request.status = ApprovalStatus.REJECTED
        request.reviewed_at = datetime.now()

        self._decisions.append(decision)
        del self._pending[str(request_id)]

        return decision

    def modify_and_approve(
        self,
        request_id: UUID,
        approved_by: str,
        modified_content: Any,
        modification_notes: str = "",
        feedback_tags: list = None,
        review_duration_seconds: int = 0
    ) -> ApprovalDecision:
        """Approve with modifications."""
        request = self._pending.get(str(request_id))
        if not request:
            raise ValueError(f"Request {request_id} not found")

        decision = ApprovalDecision(
            request_id=request_id,
            status=ApprovalStatus.MODIFIED,
            decided_by=approved_by,
            modified_content=modified_content,
            modification_notes=modification_notes,
            feedback_tags=feedback_tags or [],
            review_duration_seconds=review_duration_seconds
        )

        request.status = ApprovalStatus.MODIFIED
        request.reviewed_at = datetime.now()

        self._decisions.append(decision)
        del self._pending[str(request_id)]

        # Execute callback with modified content
        self._execute_callback(request_id, decision)

        return decision

    def register_callback(
        self,
        request_id: UUID,
        callback: Callable[[ApprovalDecision], None]
    ) -> None:
        """Register callback for when decision is made."""
        self._callbacks[str(request_id)] = callback

    def _execute_callback(self, request_id: UUID, decision: ApprovalDecision) -> None:
        """Execute registered callback."""
        callback = self._callbacks.get(str(request_id))
        if callback:
            callback(decision)
            del self._callbacks[str(request_id)]

    def get_pending(
        self,
        user_id: str = None,
        category: ApprovalCategory = None
    ) -> list[ApprovalRequest]:
        """Get pending approval requests."""
        requests = list(self._pending.values())

        if user_id:
            requests = [r for r in requests if r.assigned_to == user_id]

        if category:
            requests = [r for r in requests if r.category == category]

        # Sort by priority and age
        priority_order = {
            ApprovalPriority.URGENT: 0,
            ApprovalPriority.HIGH: 1,
            ApprovalPriority.NORMAL: 2,
            ApprovalPriority.LOW: 3
        }
        requests.sort(key=lambda r: (priority_order[r.priority], r.created_at))

        return requests

    def check_expirations(self) -> list[ApprovalRequest]:
        """Check for expired requests and handle them."""
        expired = []
        now = datetime.now()

        for request_id, request in list(self._pending.items()):
            if request.is_expired():
                expired.append(request)

                gate = self._gates.get(request.category.value)
                if gate and gate.auto_approve_on_expiry:
                    self.approve(
                        request.id,
                        approved_by="system_auto",
                        rationale="Auto-approved on expiration"
                    )
                else:
                    request.status = ApprovalStatus.EXPIRED
                    del self._pending[request_id]

        return expired

    def get_feedback_summary(self, days: int = 30) -> dict:
        """
        Get summary of approval feedback for learning.

        This data feeds into the closed-loop learning system.
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_decisions = [d for d in self._decisions if d.decided_at >= cutoff]

        summary = {
            "total_decisions": len(recent_decisions),
            "approved": sum(1 for d in recent_decisions if d.status == ApprovalStatus.APPROVED),
            "rejected": sum(1 for d in recent_decisions if d.status == ApprovalStatus.REJECTED),
            "modified": sum(1 for d in recent_decisions if d.status == ApprovalStatus.MODIFIED),
            "average_review_time_seconds": 0,
            "common_feedback_tags": {},
            "rejection_reasons": []
        }

        if recent_decisions:
            summary["average_review_time_seconds"] = sum(
                d.review_duration_seconds for d in recent_decisions
            ) / len(recent_decisions)

            for d in recent_decisions:
                for tag in d.feedback_tags:
                    summary["common_feedback_tags"][tag] = \
                        summary["common_feedback_tags"].get(tag, 0) + 1

                if d.rejection_reason:
                    summary["rejection_reasons"].append(d.rejection_reason)

        return summary

    @classmethod
    def create_default(cls) -> "ApprovalEngine":
        """Create engine with default gates."""
        engine = cls()

        # CRM update gate
        engine.register_gate(ApprovalGate(
            name="crm_update",
            description="Approval for CRM field updates",
            category=ApprovalCategory.CRM_UPDATE,
            confidence_threshold=0.8,
            expiration_hours=24
        ))

        # Email send gate
        engine.register_gate(ApprovalGate(
            name="email_send",
            description="Approval for sending emails",
            category=ApprovalCategory.EMAIL_SEND,
            always_required=True,  # Always require approval for external comms
            expiration_hours=12
        ))

        # Discount gate
        engine.register_gate(ApprovalGate(
            name="discount",
            description="Approval for discounts",
            category=ApprovalCategory.DISCOUNT,
            value_threshold=10.0,  # Require approval for discounts > 10%
            required_role="manager",
            allow_self_approval=False
        ))

        # Risk escalation gate
        engine.register_gate(ApprovalGate(
            name="risk_escalation",
            description="Approval for risk escalations",
            category=ApprovalCategory.ESCALATION,
            confidence_threshold=0.6,
            expiration_hours=4,
            notify_on_expire=True
        ))

        return engine
