"""
Layer 3: Agent Orchestration and Policy

From Section 4.3 of the paper:

Playbooks:
- Post-call update
- Follow-up drafting
- NBA recommendations
- Deal-risk alerts

Triggering:
- Event-driven (meeting ended, email received)
- Schedule-driven (daily pipeline sweep)

Escalation:
- Confidence thresholds
- Policy violations
- High-value opportunities
- Compliance-sensitive topics

Human Approval Gates:
- Approve/edit/reject actions
- Capture rationales for governance and learning

LangGraph Integration:
- State machine-based workflow execution
- Conditional branching and routing
- Human-in-the-loop approval gates
- Checkpointing for state persistence
"""

# Original implementations
from .playbooks import (
    Playbook,
    PlaybookRegistry,
    PostCallPlaybook,
    FollowUpPlaybook,
    RiskAlertPlaybook
)
from .triggers import (
    Trigger,
    EventTrigger,
    ScheduleTrigger,
    TriggerEngine
)
from .approval import (
    ApprovalGate,
    ApprovalRequest,
    ApprovalDecision,
    ApprovalEngine
)
from .agents import (
    Agent,
    AgentOrchestrator,
    ActionResult
)

# LangGraph implementations
from .agents_langgraph import (
    AgentOrchestratorLangGraph,
    CRMWorkflowBuilder,
    WorkflowNodes,
    WorkflowStatus,
    AgentState,
    create_agent_orchestrator
)

__all__ = [
    # Original implementations
    "Playbook",
    "PlaybookRegistry",
    "PostCallPlaybook",
    "FollowUpPlaybook",
    "RiskAlertPlaybook",
    "Trigger",
    "EventTrigger",
    "ScheduleTrigger",
    "TriggerEngine",
    "ApprovalGate",
    "ApprovalRequest",
    "ApprovalDecision",
    "ApprovalEngine",
    "Agent",
    "AgentOrchestrator",
    "ActionResult",
    # LangGraph implementations
    "AgentOrchestratorLangGraph",
    "CRMWorkflowBuilder",
    "WorkflowNodes",
    "WorkflowStatus",
    "AgentState",
    "create_agent_orchestrator"
]
