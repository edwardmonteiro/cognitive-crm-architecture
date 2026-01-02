"""
Agent Orchestration - Coordinating Actions

This module implements the agent orchestration system that:
- Executes playbooks
- Coordinates between intelligence and action layers
- Manages approval workflows
- Tracks action outcomes

Based on the multi-agent architecture approach suggested in
the related work (Wooldridge, 2020).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

from .playbooks import (
    Playbook, PlaybookContext, PlaybookResult, PlaybookStatus,
    PlaybookStep, ActionType, PlaybookRegistry
)
from .triggers import TriggerEngine, TriggerEvent, TriggerResult
from .approval import ApprovalEngine, ApprovalRequest, ApprovalDecision, ApprovalStatus


class ActionStatus(Enum):
    """Status of an action."""
    PENDING = "pending"
    EXECUTING = "executing"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ActionResult:
    """Result of executing an action."""
    id: UUID = field(default_factory=uuid4)
    action_type: ActionType = ActionType.ANALYZE
    status: ActionStatus = ActionStatus.COMPLETED

    # Output
    output: Any = None
    error: Optional[str] = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Approval tracking
    approval_request_id: Optional[UUID] = None
    was_modified: bool = False
    modification_notes: str = ""


@dataclass
class AgentState:
    """State of an agent during execution."""
    id: UUID = field(default_factory=uuid4)
    status: str = "idle"  # idle, busy, waiting_approval

    # Current work
    current_playbook: Optional[str] = None
    current_step: int = 0

    # Metrics
    actions_executed: int = 0
    approvals_requested: int = 0
    errors_encountered: int = 0

    # Timestamps
    started_at: Optional[datetime] = None
    last_action_at: Optional[datetime] = None


class ActionExecutor:
    """
    Executes individual actions within playbooks.

    Maps action types to concrete implementations.
    """

    def __init__(
        self,
        reasoning_engine=None,
        rag_engine=None,
        memory_manager=None,
        crm_client=None,
        notification_client=None
    ):
        self._reasoning = reasoning_engine
        self._rag = rag_engine
        self._memory = memory_manager
        self._crm = crm_client
        self._notifications = notification_client

        self._executors = {
            ActionType.ANALYZE: self._execute_analyze,
            ActionType.RETRIEVE: self._execute_retrieve,
            ActionType.GENERATE: self._execute_generate,
            ActionType.UPDATE_CRM: self._execute_update_crm,
            ActionType.SEND_NOTIFICATION: self._execute_notification,
            ActionType.CREATE_TASK: self._execute_create_task,
            ActionType.SEND_EMAIL: self._execute_send_email,
            ActionType.ESCALATE: self._execute_escalate,
        }

    def execute(
        self,
        step: PlaybookStep,
        context: PlaybookContext
    ) -> ActionResult:
        """Execute a playbook step."""
        executor = self._executors.get(step.action_type)
        if not executor:
            return ActionResult(
                action_type=step.action_type,
                status=ActionStatus.FAILED,
                error=f"No executor for action type: {step.action_type}"
            )

        try:
            result = executor(step, context)
            return result
        except Exception as e:
            return ActionResult(
                action_type=step.action_type,
                status=ActionStatus.FAILED,
                error=str(e)
            )

    def _execute_analyze(self, step: PlaybookStep, context: PlaybookContext) -> ActionResult:
        """Execute an analysis action."""
        task_type = step.parameters.get("task_type", "summarization")
        input_key = step.parameters.get("input_key", "primary_content")

        # Get input from context
        input_data = context.trigger_data.get(input_key) or context.outputs.get(input_key)

        if not input_data:
            return ActionResult(
                action_type=step.action_type,
                status=ActionStatus.FAILED,
                error=f"Input not found: {input_key}"
            )

        # Execute reasoning (mock if no engine)
        if self._reasoning:
            from ..intelligence.reasoning import ReasoningContext, TaskType
            reasoning_ctx = ReasoningContext(
                opportunity_id=context.opportunity_id,
                primary_content=input_data if isinstance(input_data, str) else str(input_data),
                working_memory=context.outputs
            )
            result = self._reasoning.execute(TaskType(task_type), reasoning_ctx)
            output = result.structured_output
        else:
            output = {"mock_analysis": f"Analysis of {task_type}", "input": input_key}

        return ActionResult(
            action_type=step.action_type,
            status=ActionStatus.COMPLETED,
            output=output
        )

    def _execute_retrieve(self, step: PlaybookStep, context: PlaybookContext) -> ActionResult:
        """Execute a retrieval action."""
        sources = step.parameters.get("sources", [])
        query_key = step.parameters.get("query_key")

        query = context.trigger_data.get(query_key) or context.outputs.get(query_key, "")

        if self._rag:
            from ..intelligence.rag import SourceType
            source_types = [SourceType(s) for s in sources if hasattr(SourceType, s.upper())]
            result = self._rag.retrieve(query, source_types=source_types)
            output = {
                "retrieved_chunks": [
                    {"content": c.content, "source": c.source_type.value, "score": c.relevance_score}
                    for c in result.chunks
                ],
                "total_results": len(result.chunks)
            }
        else:
            output = {"mock_retrieval": sources, "query": query}

        return ActionResult(
            action_type=step.action_type,
            status=ActionStatus.COMPLETED,
            output=output
        )

    def _execute_generate(self, step: PlaybookStep, context: PlaybookContext) -> ActionResult:
        """Execute a content generation action."""
        template = step.parameters.get("template", "generic")
        input_keys = step.parameters.get("inputs", [])

        # Gather inputs
        inputs = {}
        for key in input_keys:
            inputs[key] = context.outputs.get(key)

        # Generate content (mock implementation)
        if template == "crm_update":
            output = {
                "notes": f"Summary: {inputs.get('summary', {}).get('executive_summary', '')}",
                "risks": inputs.get("risks", {}).get("risks", []),
                "next_steps": inputs.get("summary", {}).get("next_steps", [])
            }
        elif template == "follow_up_email":
            output = {
                "subject": "Following up on our conversation",
                "body": f"Based on our discussion, here are the next steps..."
            }
        elif template == "nba_justification":
            recommendations = inputs.get("recommendations", {}).get("actions", [])
            output = {
                "recommendations": [
                    {**r, "justification": f"Based on context analysis..."}
                    for r in recommendations
                ]
            }
        else:
            output = {"generated": template, "inputs": input_keys}

        return ActionResult(
            action_type=step.action_type,
            status=ActionStatus.COMPLETED,
            output=output
        )

    def _execute_update_crm(self, step: PlaybookStep, context: PlaybookContext) -> ActionResult:
        """Execute a CRM update action."""
        update_type = step.parameters.get("update_type")
        data_key = step.parameters.get("data_key")

        data = context.outputs.get(data_key)

        if self._crm:
            result = self._crm.update(
                opportunity_id=context.opportunity_id,
                update_type=update_type,
                data=data
            )
            output = result
        else:
            output = {
                "mock_update": update_type,
                "opportunity_id": str(context.opportunity_id),
                "data": data,
                "status": "success"
            }

        return ActionResult(
            action_type=step.action_type,
            status=ActionStatus.COMPLETED,
            output=output
        )

    def _execute_notification(self, step: PlaybookStep, context: PlaybookContext) -> ActionResult:
        """Execute a notification action."""
        channel = step.parameters.get("channel", "email")
        content_key = step.parameters.get("content_key")

        content = context.outputs.get(content_key, "")

        if self._notifications:
            result = self._notifications.send(
                channel=channel,
                content=content,
                context={"opportunity_id": str(context.opportunity_id)}
            )
            output = result
        else:
            output = {
                "mock_notification": channel,
                "content": content,
                "status": "sent"
            }

        return ActionResult(
            action_type=step.action_type,
            status=ActionStatus.COMPLETED,
            output=output
        )

    def _execute_create_task(self, step: PlaybookStep, context: PlaybookContext) -> ActionResult:
        """Execute task creation."""
        task_type = step.parameters.get("task_type", "follow_up")
        include_draft = step.parameters.get("include_draft", False)

        task = {
            "task_type": task_type,
            "opportunity_id": str(context.opportunity_id),
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }

        if include_draft:
            task["draft"] = context.outputs.get("email_draft", {})

        return ActionResult(
            action_type=step.action_type,
            status=ActionStatus.COMPLETED,
            output=task
        )

    def _execute_send_email(self, step: PlaybookStep, context: PlaybookContext) -> ActionResult:
        """Execute email sending."""
        content_key = step.parameters.get("content_key", "email_draft")
        content = context.outputs.get(content_key, {})

        output = {
            "email_sent": True,
            "subject": content.get("subject", ""),
            "to": content.get("to", []),
            "sent_at": datetime.now().isoformat()
        }

        return ActionResult(
            action_type=step.action_type,
            status=ActionStatus.COMPLETED,
            output=output
        )

    def _execute_escalate(self, step: PlaybookStep, context: PlaybookContext) -> ActionResult:
        """Execute escalation."""
        escalation_type = step.parameters.get("escalation_type", "manager_review")
        context_keys = step.parameters.get("context_keys", [])

        escalation_context = {key: context.outputs.get(key) for key in context_keys}

        output = {
            "escalation_type": escalation_type,
            "escalated_at": datetime.now().isoformat(),
            "context": escalation_context,
            "status": "escalated"
        }

        return ActionResult(
            action_type=step.action_type,
            status=ActionStatus.COMPLETED,
            output=output
        )


class Agent:
    """
    Agent that executes playbooks.

    An agent:
    - Receives trigger events
    - Executes playbook steps
    - Requests approval when needed
    - Reports outcomes
    """

    def __init__(
        self,
        executor: ActionExecutor,
        approval_engine: ApprovalEngine = None
    ):
        self.id = uuid4()
        self.state = AgentState()
        self._executor = executor
        self._approval = approval_engine

    def execute_playbook(
        self,
        playbook: Playbook,
        trigger_data: dict,
        opportunity_id: UUID,
        account_id: UUID = None,
        user_id: str = None
    ) -> PlaybookResult:
        """Execute a complete playbook."""
        # Initialize context
        context = PlaybookContext(
            opportunity_id=opportunity_id,
            account_id=account_id,
            trigger_data=trigger_data,
            status=PlaybookStatus.RUNNING,
            started_at=datetime.now(),
            user_id=user_id
        )

        # Update agent state
        self.state.status = "busy"
        self.state.current_playbook = playbook.name
        self.state.started_at = datetime.now()

        actions_taken = []
        approvals_requested = 0

        try:
            for i, step in enumerate(playbook.steps):
                context.current_step = i

                # Check condition
                if step.condition and not step.condition(context):
                    continue

                # Execute step
                result = self._executor.execute(step, context)
                self.state.actions_executed += 1
                self.state.last_action_at = datetime.now()

                if result.status == ActionStatus.FAILED:
                    if step.skip_on_failure:
                        context.warnings.append(f"Step {step.name} failed: {result.error}")
                        continue
                    else:
                        context.errors.append(f"Step {step.name} failed: {result.error}")
                        context.status = PlaybookStatus.FAILED
                        break

                # Handle approval if required
                if step.requires_approval and self._approval:
                    approval_request = self._create_approval_request(
                        step, result, context, playbook.name
                    )
                    approvals_requested += 1
                    self.state.approvals_requested += 1

                    # In async mode, would wait for approval
                    # For simulation, auto-approve
                    decision = self._approval.approve(
                        approval_request.id,
                        approved_by=user_id or "simulation",
                        rationale="Auto-approved in simulation"
                    )

                    if decision.status == ApprovalStatus.MODIFIED:
                        result.output = decision.modified_content
                        result.was_modified = True

                    elif decision.status == ApprovalStatus.REJECTED:
                        context.status = PlaybookStatus.FAILED
                        context.errors.append(f"Step {step.name} rejected: {decision.rejection_reason}")
                        break

                # Store output in context
                if step.output_key:
                    context.outputs[step.output_key] = result.output

                actions_taken.append({
                    "step": step.name,
                    "action_type": step.action_type.value,
                    "status": result.status.value,
                    "output_key": step.output_key
                })

            # Mark completed if no errors
            if context.status == PlaybookStatus.RUNNING:
                context.status = PlaybookStatus.COMPLETED

            context.completed_at = datetime.now()

        except Exception as e:
            context.status = PlaybookStatus.FAILED
            context.errors.append(str(e))
            self.state.errors_encountered += 1

        finally:
            self.state.status = "idle"
            self.state.current_playbook = None

        # Build result
        return PlaybookResult(
            playbook_name=playbook.name,
            status=context.status,
            context=context,
            started_at=context.started_at,
            completed_at=context.completed_at,
            duration_seconds=(context.completed_at - context.started_at).total_seconds()
                if context.completed_at else 0,
            outputs=context.outputs,
            actions_taken=actions_taken,
            approvals_requested=approvals_requested
        )

    def _create_approval_request(
        self,
        step: PlaybookStep,
        result: ActionResult,
        context: PlaybookContext,
        playbook_name: str
    ) -> ApprovalRequest:
        """Create approval request for a step."""
        return self._approval.create_request(
            gate_name=step.action_type.value,
            action_type=step.action_type.value,
            action_description=step.name,
            proposed_content=result.output,
            context={
                "playbook": playbook_name,
                "step": step.name,
                "confidence": 0.8  # Would come from reasoning result
            },
            opportunity_id=context.opportunity_id,
            playbook_name=playbook_name,
            step_name=step.name
        )


class AgentOrchestrator:
    """
    Orchestrates multiple agents and coordinates workflows.

    Responsibilities:
    - Route events to appropriate playbooks
    - Manage agent pool
    - Track overall execution metrics
    - Handle failures and retries
    """

    def __init__(
        self,
        playbook_registry: PlaybookRegistry = None,
        trigger_engine: TriggerEngine = None,
        approval_engine: ApprovalEngine = None,
        executor: ActionExecutor = None
    ):
        self._playbooks = playbook_registry or PlaybookRegistry.create_default()
        self._triggers = trigger_engine or TriggerEngine.create_default()
        self._approval = approval_engine or ApprovalEngine.create_default()
        self._executor = executor or ActionExecutor()

        self._agents: list[Agent] = []
        self._execution_history: list[PlaybookResult] = []

        # Create default agent pool
        self._initialize_agents(pool_size=3)

    def _initialize_agents(self, pool_size: int) -> None:
        """Initialize agent pool."""
        for _ in range(pool_size):
            agent = Agent(self._executor, self._approval)
            self._agents.append(agent)

    def _get_available_agent(self) -> Optional[Agent]:
        """Get an available agent from the pool."""
        for agent in self._agents:
            if agent.state.status == "idle":
                return agent
        return None

    def process_event(self, event: TriggerEvent) -> list[PlaybookResult]:
        """Process an event through the orchestration layer."""
        results = []

        # Get triggered playbooks
        trigger_results = self._triggers.process_event(event)

        for trigger_result in trigger_results:
            for playbook_name in trigger_result.playbooks_to_run:
                playbook = self._playbooks.get(playbook_name)
                if not playbook:
                    continue

                # Get agent
                agent = self._get_available_agent()
                if not agent:
                    # Queue for later (simplified - just skip)
                    continue

                # Execute playbook
                result = agent.execute_playbook(
                    playbook=playbook,
                    trigger_data=trigger_result.context,
                    opportunity_id=event.opportunity_id or uuid4(),
                    account_id=event.account_id,
                    user_id=event.user_id
                )

                results.append(result)
                self._execution_history.append(result)

        return results

    def run_scheduled_triggers(self) -> list[PlaybookResult]:
        """Run all due scheduled triggers."""
        results = []

        trigger_results = self._triggers.check_schedules()

        for trigger_result in trigger_results:
            for playbook_name in trigger_result.playbooks_to_run:
                playbook = self._playbooks.get(playbook_name)
                if not playbook:
                    continue

                agent = self._get_available_agent()
                if not agent:
                    continue

                result = agent.execute_playbook(
                    playbook=playbook,
                    trigger_data=trigger_result.context,
                    opportunity_id=trigger_result.event.opportunity_id or uuid4()
                )

                results.append(result)
                self._execution_history.append(result)

        return results

    def execute_playbook_directly(
        self,
        playbook_name: str,
        opportunity_id: UUID,
        context: dict,
        user_id: str = None
    ) -> PlaybookResult:
        """Execute a playbook directly without event trigger."""
        playbook = self._playbooks.get(playbook_name)
        if not playbook:
            raise ValueError(f"Playbook not found: {playbook_name}")

        agent = self._get_available_agent()
        if not agent:
            raise RuntimeError("No agents available")

        result = agent.execute_playbook(
            playbook=playbook,
            trigger_data=context,
            opportunity_id=opportunity_id,
            user_id=user_id
        )

        self._execution_history.append(result)
        return result

    def get_execution_metrics(self) -> dict:
        """Get metrics about playbook execution."""
        if not self._execution_history:
            return {"total_executions": 0}

        completed = [r for r in self._execution_history if r.status == PlaybookStatus.COMPLETED]
        failed = [r for r in self._execution_history if r.status == PlaybookStatus.FAILED]

        return {
            "total_executions": len(self._execution_history),
            "completed": len(completed),
            "failed": len(failed),
            "success_rate": len(completed) / len(self._execution_history) if self._execution_history else 0,
            "average_duration_seconds": sum(r.duration_seconds for r in self._execution_history) / len(self._execution_history),
            "total_approvals_requested": sum(r.approvals_requested for r in self._execution_history),
            "by_playbook": self._group_by_playbook()
        }

    def _group_by_playbook(self) -> dict:
        """Group execution history by playbook."""
        by_playbook = {}

        for result in self._execution_history:
            name = result.playbook_name
            if name not in by_playbook:
                by_playbook[name] = {"count": 0, "success": 0, "failed": 0}

            by_playbook[name]["count"] += 1
            if result.status == PlaybookStatus.COMPLETED:
                by_playbook[name]["success"] += 1
            elif result.status == PlaybookStatus.FAILED:
                by_playbook[name]["failed"] += 1

        return by_playbook
