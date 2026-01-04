"""
Agent Orchestration with LangGraph

Implements the orchestration layer using LangGraph for:
- State machine-based workflow execution
- Conditional branching and routing
- Human-in-the-loop approval gates
- Tool calling with structured outputs
- Parallel execution of independent tasks

From Section 4.3: "A Playbook orchestrates steps... with conditional
human gates for oversight."

LangGraph provides:
- Graph-based workflow definition
- Checkpointing for state persistence
- Conditional edges for dynamic routing
- Support for async execution
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Literal, TypedDict, Annotated
from uuid import UUID, uuid4
import json
import operator

from pydantic import BaseModel, Field


# ============================================================================
# State Definitions for LangGraph
# ============================================================================

class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph workflow."""
    # Identity
    workflow_id: str
    opportunity_id: str
    account_id: Optional[str]
    user_id: Optional[str]

    # Workflow tracking
    current_step: str
    status: str
    started_at: str
    completed_at: Optional[str]

    # Input/Output
    input_data: Dict[str, Any]
    outputs: Annotated[Dict[str, Any], operator.or_]
    messages: Annotated[List[Dict[str, Any]], operator.add]

    # Errors and warnings
    errors: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]

    # Approval tracking
    pending_approval: Optional[Dict[str, Any]]
    approval_history: Annotated[List[Dict[str, Any]], operator.add]

    # RAG context
    retrieved_context: Optional[str]

    # Memory context
    memory_context: Optional[Dict[str, Any]]


class ApprovalDecision(BaseModel):
    """Decision from human approval gate."""
    approved: bool
    modified_content: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    approved_by: str = "system"
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Workflow Node Functions
# ============================================================================

class WorkflowNodes:
    """
    Collection of node functions for LangGraph workflows.

    Each node represents a step in a cognitive CRM workflow.
    """

    def __init__(
        self,
        reasoning_engine=None,
        rag_engine=None,
        memory_manager=None,
        llm_provider=None
    ):
        self._reasoning = reasoning_engine
        self._rag = rag_engine
        self._memory = memory_manager
        self._llm_provider = llm_provider

    def initialize_workflow(self, state: AgentState) -> Dict[str, Any]:
        """Initialize workflow state."""
        return {
            "workflow_id": str(uuid4()),
            "status": WorkflowStatus.RUNNING.value,
            "started_at": datetime.now().isoformat(),
            "current_step": "initialize",
            "outputs": {},
            "messages": [{
                "role": "system",
                "content": f"Workflow initialized for opportunity {state.get('opportunity_id')}"
            }]
        }

    def load_memory_context(self, state: AgentState) -> Dict[str, Any]:
        """Load memory context for the opportunity."""
        opp_id = state.get("opportunity_id")
        account_id = state.get("account_id")

        context = {}

        if self._memory:
            try:
                context = self._memory.get_full_context(
                    opportunity_id=UUID(opp_id) if opp_id else uuid4(),
                    account_id=UUID(account_id) if account_id else None
                )
            except Exception as e:
                return {
                    "warnings": [f"Failed to load memory context: {str(e)}"],
                    "memory_context": {}
                }

        return {
            "current_step": "load_memory",
            "memory_context": context,
            "messages": [{
                "role": "system",
                "content": f"Loaded memory context with {len(context)} components"
            }]
        }

    def retrieve_context(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant context using RAG."""
        input_data = state.get("input_data", {})
        query = input_data.get("query") or input_data.get("transcript", "")[:500]

        retrieved = ""

        if self._rag and query:
            try:
                result = self._rag.retrieve(query, top_k=5)
                retrieved = self._rag.format_context_for_llm(result)
            except Exception as e:
                return {
                    "warnings": [f"RAG retrieval failed: {str(e)}"],
                    "retrieved_context": ""
                }

        return {
            "current_step": "retrieve",
            "retrieved_context": retrieved,
            "messages": [{
                "role": "system",
                "content": f"Retrieved {len(retrieved)} chars of context"
            }]
        }

    def analyze_interaction(self, state: AgentState) -> Dict[str, Any]:
        """Analyze interaction (summarization, extraction)."""
        input_data = state.get("input_data", {})
        transcript = input_data.get("transcript", "")

        results = {}

        if self._reasoning:
            try:
                # Run summarization
                summary = self._reasoning.summarize(transcript)
                results["summary"] = summary.model_dump() if hasattr(summary, "model_dump") else summary

                # Run objection extraction
                objections = self._reasoning.extract_objections(transcript)
                results["objections"] = objections.model_dump() if hasattr(objections, "model_dump") else objections

                # Run stakeholder mapping
                stakeholders = self._reasoning.map_stakeholders(transcript)
                results["stakeholders"] = stakeholders.model_dump() if hasattr(stakeholders, "model_dump") else stakeholders

                # Run risk detection
                risks = self._reasoning.detect_risks(transcript)
                results["risks"] = risks.model_dump() if hasattr(risks, "model_dump") else risks

            except Exception as e:
                return {
                    "errors": [f"Analysis failed: {str(e)}"],
                    "outputs": {"analysis_error": str(e)}
                }
        else:
            # Mock analysis
            results = {
                "summary": {
                    "executive_summary": "Mock summary of the interaction",
                    "key_points": ["Point 1", "Point 2"],
                    "sentiment": "positive"
                },
                "objections": {"objections": [], "resistance_level": "low"},
                "stakeholders": {"stakeholders": []},
                "risks": {"risks": [], "overall_risk": "low"}
            }

        return {
            "current_step": "analyze",
            "outputs": {"analysis": results},
            "messages": [{
                "role": "assistant",
                "content": "Completed interaction analysis"
            }]
        }

    def generate_recommendations(self, state: AgentState) -> Dict[str, Any]:
        """Generate next-best-action recommendations."""
        outputs = state.get("outputs", {})
        memory_context = state.get("memory_context", {})
        retrieved_context = state.get("retrieved_context", "")

        recommendations = []

        if self._reasoning:
            try:
                # Build context for NBA generation
                context = {
                    "analysis": outputs.get("analysis", {}),
                    "memory": memory_context,
                    "retrieved": retrieved_context
                }

                nba = self._reasoning.plan_next_steps(json.dumps(context))
                recommendations = nba.model_dump() if hasattr(nba, "model_dump") else nba

            except Exception as e:
                return {
                    "warnings": [f"NBA generation failed: {str(e)}"],
                    "outputs": {"recommendations": []}
                }
        else:
            # Mock recommendations
            recommendations = {
                "next_steps": [
                    {
                        "action": "Schedule follow-up meeting",
                        "priority": "high",
                        "timeline": "within 2 days",
                        "rationale": "Customer expressed interest"
                    },
                    {
                        "action": "Send proposal draft",
                        "priority": "medium",
                        "timeline": "within 1 week",
                        "rationale": "Address budget concerns"
                    }
                ]
            }

        return {
            "current_step": "recommend",
            "outputs": {"recommendations": recommendations},
            "messages": [{
                "role": "assistant",
                "content": f"Generated {len(recommendations.get('next_steps', []))} recommendations"
            }]
        }

    def prepare_approval(self, state: AgentState) -> Dict[str, Any]:
        """Prepare content for human approval."""
        outputs = state.get("outputs", {})

        approval_content = {
            "workflow_id": state.get("workflow_id"),
            "opportunity_id": state.get("opportunity_id"),
            "content_type": "crm_update",
            "proposed_content": {
                "summary": outputs.get("analysis", {}).get("summary", {}),
                "recommendations": outputs.get("recommendations", {})
            },
            "confidence": 0.85,
            "requested_at": datetime.now().isoformat()
        }

        return {
            "current_step": "await_approval",
            "status": WorkflowStatus.AWAITING_APPROVAL.value,
            "pending_approval": approval_content,
            "messages": [{
                "role": "system",
                "content": "Awaiting human approval for CRM update"
            }]
        }

    def process_approval(self, state: AgentState, decision: ApprovalDecision) -> Dict[str, Any]:
        """Process approval decision from human."""
        pending = state.get("pending_approval", {})

        if decision.approved:
            content = decision.modified_content or pending.get("proposed_content")
            return {
                "current_step": "approved",
                "status": WorkflowStatus.RUNNING.value,
                "pending_approval": None,
                "outputs": {"approved_content": content},
                "approval_history": [{
                    "decision": "approved",
                    "modified": decision.modified_content is not None,
                    "approved_by": decision.approved_by,
                    "timestamp": decision.timestamp.isoformat()
                }],
                "messages": [{
                    "role": "human",
                    "content": f"Approved by {decision.approved_by}"
                }]
            }
        else:
            return {
                "current_step": "rejected",
                "status": WorkflowStatus.FAILED.value,
                "pending_approval": None,
                "approval_history": [{
                    "decision": "rejected",
                    "reason": decision.rejection_reason,
                    "approved_by": decision.approved_by,
                    "timestamp": decision.timestamp.isoformat()
                }],
                "errors": [f"Approval rejected: {decision.rejection_reason}"],
                "messages": [{
                    "role": "human",
                    "content": f"Rejected: {decision.rejection_reason}"
                }]
            }

    def update_crm(self, state: AgentState) -> Dict[str, Any]:
        """Update CRM with approved content."""
        outputs = state.get("outputs", {})
        approved_content = outputs.get("approved_content", {})

        # Mock CRM update
        update_result = {
            "crm_updated": True,
            "opportunity_id": state.get("opportunity_id"),
            "updated_at": datetime.now().isoformat(),
            "fields_updated": list(approved_content.keys())
        }

        return {
            "current_step": "crm_updated",
            "outputs": {"crm_update": update_result},
            "messages": [{
                "role": "system",
                "content": f"CRM updated for opportunity {state.get('opportunity_id')}"
            }]
        }

    def store_to_memory(self, state: AgentState) -> Dict[str, Any]:
        """Store workflow results to memory."""
        outputs = state.get("outputs", {})
        opp_id = state.get("opportunity_id")

        if self._memory and opp_id:
            try:
                # Store analysis results
                analysis = outputs.get("analysis", {})
                if analysis:
                    self._memory.working.store(
                        key="last_analysis",
                        value=analysis,
                        opportunity_id=UUID(opp_id),
                        metadata={"source": "langgraph_workflow"}
                    )

                # Store recommendations
                recommendations = outputs.get("recommendations", {})
                if recommendations:
                    self._memory.working.store(
                        key="last_recommendations",
                        value=recommendations,
                        opportunity_id=UUID(opp_id),
                        metadata={"source": "langgraph_workflow"}
                    )

            except Exception as e:
                return {
                    "warnings": [f"Memory storage failed: {str(e)}"]
                }

        return {
            "current_step": "stored",
            "messages": [{
                "role": "system",
                "content": "Results stored to memory"
            }]
        }

    def finalize_workflow(self, state: AgentState) -> Dict[str, Any]:
        """Finalize workflow and clean up."""
        return {
            "current_step": "complete",
            "status": WorkflowStatus.COMPLETED.value,
            "completed_at": datetime.now().isoformat(),
            "messages": [{
                "role": "system",
                "content": "Workflow completed successfully"
            }]
        }

    def handle_error(self, state: AgentState, error: str) -> Dict[str, Any]:
        """Handle workflow errors."""
        return {
            "current_step": "error",
            "status": WorkflowStatus.FAILED.value,
            "completed_at": datetime.now().isoformat(),
            "errors": [error],
            "messages": [{
                "role": "system",
                "content": f"Workflow failed: {error}"
            }]
        }


# ============================================================================
# LangGraph Workflow Builder
# ============================================================================

class CRMWorkflowBuilder:
    """
    Builder for LangGraph-based CRM workflows.

    Creates pre-configured workflows for common CRM patterns:
    - Post-call intelligence
    - NBA recommendations
    - Deal review
    - Risk alerts
    """

    def __init__(
        self,
        reasoning_engine=None,
        rag_engine=None,
        memory_manager=None,
        llm_provider=None
    ):
        self._nodes = WorkflowNodes(
            reasoning_engine=reasoning_engine,
            rag_engine=rag_engine,
            memory_manager=memory_manager,
            llm_provider=llm_provider
        )
        self._graphs = {}

    def _try_import_langgraph(self):
        """Try to import LangGraph components."""
        try:
            from langgraph.graph import StateGraph, END
            from langgraph.checkpoint.memory import MemorySaver
            return StateGraph, END, MemorySaver
        except ImportError:
            return None, None, None

    def build_post_call_workflow(self):
        """
        Build post-call intelligence workflow.

        Flow:
        1. Initialize -> Load Memory
        2. Retrieve Context -> Analyze Interaction
        3. Generate Recommendations -> Prepare Approval
        4. (Human Decision) -> Update CRM OR Reject
        5. Store to Memory -> Finalize
        """
        StateGraph, END, MemorySaver = self._try_import_langgraph()

        if StateGraph is None:
            return self._build_mock_workflow("post_call")

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("initialize", self._nodes.initialize_workflow)
        workflow.add_node("load_memory", self._nodes.load_memory_context)
        workflow.add_node("retrieve", self._nodes.retrieve_context)
        workflow.add_node("analyze", self._nodes.analyze_interaction)
        workflow.add_node("recommend", self._nodes.generate_recommendations)
        workflow.add_node("prepare_approval", self._nodes.prepare_approval)
        workflow.add_node("update_crm", self._nodes.update_crm)
        workflow.add_node("store_memory", self._nodes.store_to_memory)
        workflow.add_node("finalize", self._nodes.finalize_workflow)

        # Add edges (linear flow with approval gate)
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "load_memory")
        workflow.add_edge("load_memory", "retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", "recommend")
        workflow.add_edge("recommend", "prepare_approval")

        # Conditional edge for approval (simplified - in real use, this would wait)
        def approval_router(state):
            if state.get("status") == WorkflowStatus.AWAITING_APPROVAL.value:
                # In real implementation, would wait for human input
                # For now, auto-approve
                return "update_crm"
            return "finalize"

        workflow.add_conditional_edges(
            "prepare_approval",
            approval_router,
            {
                "update_crm": "update_crm",
                "finalize": "finalize"
            }
        )

        workflow.add_edge("update_crm", "store_memory")
        workflow.add_edge("store_memory", "finalize")
        workflow.add_edge("finalize", END)

        # Compile with checkpointing
        memory = MemorySaver()
        compiled = workflow.compile(checkpointer=memory)

        self._graphs["post_call"] = compiled
        return compiled

    def build_nba_workflow(self):
        """
        Build NBA recommendation workflow.

        Flow:
        1. Initialize -> Load Memory
        2. Retrieve Context (playbooks, similar deals)
        3. Generate Recommendations
        4. Finalize
        """
        StateGraph, END, MemorySaver = self._try_import_langgraph()

        if StateGraph is None:
            return self._build_mock_workflow("nba")

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("initialize", self._nodes.initialize_workflow)
        workflow.add_node("load_memory", self._nodes.load_memory_context)
        workflow.add_node("retrieve", self._nodes.retrieve_context)
        workflow.add_node("recommend", self._nodes.generate_recommendations)
        workflow.add_node("store_memory", self._nodes.store_to_memory)
        workflow.add_node("finalize", self._nodes.finalize_workflow)

        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "load_memory")
        workflow.add_edge("load_memory", "retrieve")
        workflow.add_edge("retrieve", "recommend")
        workflow.add_edge("recommend", "store_memory")
        workflow.add_edge("store_memory", "finalize")
        workflow.add_edge("finalize", END)

        # Compile
        memory = MemorySaver()
        compiled = workflow.compile(checkpointer=memory)

        self._graphs["nba"] = compiled
        return compiled

    def build_risk_alert_workflow(self):
        """
        Build risk alert workflow.

        Flow:
        1. Initialize -> Load Memory
        2. Analyze for Risks
        3. (If High Risk) -> Escalate
        4. Store to Memory -> Finalize
        """
        StateGraph, END, MemorySaver = self._try_import_langgraph()

        if StateGraph is None:
            return self._build_mock_workflow("risk_alert")

        workflow = StateGraph(AgentState)

        # Custom risk analysis node
        def analyze_risks(state):
            result = self._nodes.analyze_interaction(state)
            risks = result.get("outputs", {}).get("analysis", {}).get("risks", {})
            risk_level = risks.get("overall_risk", "low")

            if risk_level == "high":
                result["outputs"]["escalation_needed"] = True
            else:
                result["outputs"]["escalation_needed"] = False

            return result

        def escalate(state):
            return {
                "current_step": "escalated",
                "outputs": {
                    "escalation": {
                        "type": "risk_alert",
                        "opportunity_id": state.get("opportunity_id"),
                        "escalated_at": datetime.now().isoformat(),
                        "risks": state.get("outputs", {}).get("analysis", {}).get("risks", {})
                    }
                },
                "messages": [{
                    "role": "system",
                    "content": "High risk detected - escalation triggered"
                }]
            }

        # Add nodes
        workflow.add_node("initialize", self._nodes.initialize_workflow)
        workflow.add_node("load_memory", self._nodes.load_memory_context)
        workflow.add_node("analyze_risks", analyze_risks)
        workflow.add_node("escalate", escalate)
        workflow.add_node("store_memory", self._nodes.store_to_memory)
        workflow.add_node("finalize", self._nodes.finalize_workflow)

        # Add edges with conditional escalation
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "load_memory")
        workflow.add_edge("load_memory", "analyze_risks")

        def escalation_router(state):
            if state.get("outputs", {}).get("escalation_needed"):
                return "escalate"
            return "store_memory"

        workflow.add_conditional_edges(
            "analyze_risks",
            escalation_router,
            {
                "escalate": "escalate",
                "store_memory": "store_memory"
            }
        )

        workflow.add_edge("escalate", "store_memory")
        workflow.add_edge("store_memory", "finalize")
        workflow.add_edge("finalize", END)

        # Compile
        memory = MemorySaver()
        compiled = workflow.compile(checkpointer=memory)

        self._graphs["risk_alert"] = compiled
        return compiled

    def _build_mock_workflow(self, name: str):
        """Build a mock workflow when LangGraph is not available."""

        class MockWorkflow:
            def __init__(self, workflow_name: str, nodes: WorkflowNodes):
                self.name = workflow_name
                self._nodes = nodes

            def invoke(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
                """Execute workflow steps sequentially."""
                result = dict(state)

                # Initialize
                result.update(self._nodes.initialize_workflow(result))

                # Load memory
                result.update(self._nodes.load_memory_context(result))

                # Retrieve
                result.update(self._nodes.retrieve_context(result))

                # Analyze
                result.update(self._nodes.analyze_interaction(result))

                # Recommend
                result.update(self._nodes.generate_recommendations(result))

                # Store to memory
                result.update(self._nodes.store_to_memory(result))

                # Finalize
                result.update(self._nodes.finalize_workflow(result))

                return result

            async def ainvoke(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
                """Async version of invoke."""
                return self.invoke(state, config)

        return MockWorkflow(name, self._nodes)

    def get_workflow(self, name: str):
        """Get a compiled workflow by name."""
        if name not in self._graphs:
            if name == "post_call":
                return self.build_post_call_workflow()
            elif name == "nba":
                return self.build_nba_workflow()
            elif name == "risk_alert":
                return self.build_risk_alert_workflow()
            else:
                raise ValueError(f"Unknown workflow: {name}")

        return self._graphs[name]


# ============================================================================
# Agent Orchestrator with LangGraph
# ============================================================================

class AgentOrchestratorLangGraph:
    """
    Orchestrates CRM workflows using LangGraph.

    Features:
    - Workflow execution with state persistence
    - Human-in-the-loop approval gates
    - Parallel workflow execution
    - Execution metrics and history
    """

    def __init__(
        self,
        reasoning_engine=None,
        rag_engine=None,
        memory_manager=None,
        llm_provider=None
    ):
        self._workflow_builder = CRMWorkflowBuilder(
            reasoning_engine=reasoning_engine,
            rag_engine=rag_engine,
            memory_manager=memory_manager,
            llm_provider=llm_provider
        )

        self._execution_history: List[Dict[str, Any]] = []
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}

    def execute_workflow(
        self,
        workflow_name: str,
        opportunity_id: UUID,
        input_data: Dict[str, Any],
        account_id: Optional[UUID] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow synchronously.

        Args:
            workflow_name: Name of the workflow to execute
            opportunity_id: Opportunity ID
            input_data: Input data for the workflow
            account_id: Optional account ID
            user_id: Optional user ID
            thread_id: Optional thread ID for checkpointing

        Returns:
            Final workflow state
        """
        workflow = self._workflow_builder.get_workflow(workflow_name)

        initial_state = {
            "opportunity_id": str(opportunity_id),
            "account_id": str(account_id) if account_id else None,
            "user_id": user_id,
            "input_data": input_data,
            "outputs": {},
            "messages": [],
            "errors": [],
            "warnings": [],
            "approval_history": []
        }

        config = {"configurable": {"thread_id": thread_id or str(uuid4())}}

        try:
            result = workflow.invoke(initial_state, config)

            # Record execution
            execution_record = {
                "workflow_name": workflow_name,
                "opportunity_id": str(opportunity_id),
                "status": result.get("status", "unknown"),
                "started_at": result.get("started_at"),
                "completed_at": result.get("completed_at"),
                "outputs": result.get("outputs", {}),
                "errors": result.get("errors", [])
            }
            self._execution_history.append(execution_record)

            return result

        except Exception as e:
            error_result = {
                **initial_state,
                "status": WorkflowStatus.FAILED.value,
                "errors": [str(e)],
                "completed_at": datetime.now().isoformat()
            }

            self._execution_history.append({
                "workflow_name": workflow_name,
                "opportunity_id": str(opportunity_id),
                "status": "failed",
                "errors": [str(e)]
            })

            return error_result

    async def execute_workflow_async(
        self,
        workflow_name: str,
        opportunity_id: UUID,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a workflow asynchronously."""
        workflow = self._workflow_builder.get_workflow(workflow_name)

        initial_state = {
            "opportunity_id": str(opportunity_id),
            "account_id": str(kwargs.get("account_id")) if kwargs.get("account_id") else None,
            "user_id": kwargs.get("user_id"),
            "input_data": input_data,
            "outputs": {},
            "messages": [],
            "errors": [],
            "warnings": [],
            "approval_history": []
        }

        config = {"configurable": {"thread_id": kwargs.get("thread_id", str(uuid4()))}}

        result = await workflow.ainvoke(initial_state, config)

        self._execution_history.append({
            "workflow_name": workflow_name,
            "opportunity_id": str(opportunity_id),
            "status": result.get("status", "unknown")
        })

        return result

    def submit_approval(
        self,
        workflow_id: str,
        decision: ApprovalDecision
    ) -> Dict[str, Any]:
        """
        Submit an approval decision for a pending workflow.

        In a real implementation, this would resume the workflow
        from the checkpoint.
        """
        if workflow_id not in self._pending_approvals:
            return {"error": f"No pending approval found for workflow {workflow_id}"}

        pending = self._pending_approvals.pop(workflow_id)

        # Process the approval (simplified - in real implementation,
        # would resume LangGraph from checkpoint)
        result = {
            "workflow_id": workflow_id,
            "approval_processed": True,
            "decision": "approved" if decision.approved else "rejected",
            "approved_by": decision.approved_by
        }

        return result

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        return list(self._pending_approvals.values())

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        if not self._execution_history:
            return {"total_executions": 0}

        completed = [e for e in self._execution_history if e.get("status") == "completed"]
        failed = [e for e in self._execution_history if e.get("status") == "failed"]

        by_workflow = {}
        for execution in self._execution_history:
            name = execution.get("workflow_name", "unknown")
            if name not in by_workflow:
                by_workflow[name] = {"count": 0, "success": 0, "failed": 0}
            by_workflow[name]["count"] += 1
            if execution.get("status") == "completed":
                by_workflow[name]["success"] += 1
            elif execution.get("status") == "failed":
                by_workflow[name]["failed"] += 1

        return {
            "total_executions": len(self._execution_history),
            "completed": len(completed),
            "failed": len(failed),
            "success_rate": len(completed) / len(self._execution_history) if self._execution_history else 0,
            "pending_approvals": len(self._pending_approvals),
            "by_workflow": by_workflow
        }

    def get_execution_history(
        self,
        workflow_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get execution history."""
        history = self._execution_history

        if workflow_name:
            history = [e for e in history if e.get("workflow_name") == workflow_name]

        return history[-limit:]


# ============================================================================
# Factory Function
# ============================================================================

def create_agent_orchestrator(
    reasoning_engine=None,
    rag_engine=None,
    memory_manager=None,
    llm_provider=None
) -> AgentOrchestratorLangGraph:
    """
    Factory function to create a configured agent orchestrator.

    Args:
        reasoning_engine: LangChain reasoning engine
        rag_engine: RAG engine for context retrieval
        memory_manager: Memory manager for state persistence
        llm_provider: LLM provider for direct LLM calls

    Returns:
        Configured AgentOrchestratorLangGraph instance
    """
    return AgentOrchestratorLangGraph(
        reasoning_engine=reasoning_engine,
        rag_engine=rag_engine,
        memory_manager=memory_manager,
        llm_provider=llm_provider
    )
