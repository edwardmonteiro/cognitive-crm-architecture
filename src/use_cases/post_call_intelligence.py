"""
Use Case 1: Post-call Intelligence and CRM Update

From Section 5.1:
"Post-call intelligence and CRM update (human-reviewed)"

This use case demonstrates:
1. Call transcript ingestion
2. Multi-task reasoning (summarization, objection extraction, stakeholder mapping, risk detection)
3. CRM update generation
4. Human approval workflow
5. Learning from feedback

It operationalizes the full pipeline from sensing to action with governance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from ..core.entities import Opportunity, Interaction, InteractionChannel
from ..core.cognitive_framework import CognitiveSignal, SensemakingResult, GovernanceContext
from ..layers.data_ingestion.ingestors import CallTranscriptIngestor
from ..layers.data_ingestion.event_schema import EventBuilder, EventStore, EventCategory
from ..layers.intelligence.reasoning import (
    ReasoningEngine, ReasoningContext, TaskType
)
from ..layers.intelligence.memory import MemoryManager
from ..layers.orchestration.playbooks import PostCallPlaybook, PlaybookContext, PlaybookResult
from ..layers.orchestration.approval import ApprovalEngine, ApprovalRequest, ApprovalDecision
from ..layers.learning.signals import SignalCollector, OutcomeSignal, OutcomeType
from ..layers.learning.feedback import FeedbackCollector, FeedbackEntry, FeedbackType


@dataclass
class PostCallResult:
    """Result of post-call intelligence processing."""
    id: UUID = field(default_factory=uuid4)
    opportunity_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)

    # Analysis outputs
    summary: dict = field(default_factory=dict)
    objections: list = field(default_factory=list)
    stakeholders: list = field(default_factory=list)
    risks: list = field(default_factory=list)
    next_steps: list = field(default_factory=list)

    # CRM update
    crm_update: dict = field(default_factory=dict)

    # Approval
    approval_request_id: Optional[UUID] = None
    approval_status: str = "pending"
    was_modified: bool = False
    modifications: dict = field(default_factory=dict)

    # Metrics
    processing_time_seconds: float = 0.0
    admin_time_saved_minutes: float = 0.0


class PostCallIntelligenceUseCase:
    """
    Implements the post-call intelligence use case.

    Flow:
    1. Receive call transcript
    2. Run reasoning pipeline
    3. Generate CRM update
    4. Request human approval
    5. Apply update (if approved)
    6. Record outcome for learning
    """

    def __init__(
        self,
        reasoning_engine: ReasoningEngine = None,
        memory_manager: MemoryManager = None,
        approval_engine: ApprovalEngine = None,
        event_store: EventStore = None,
        signal_collector: SignalCollector = None,
        feedback_collector: FeedbackCollector = None
    ):
        self._reasoning = reasoning_engine or ReasoningEngine()
        self._memory = memory_manager or MemoryManager()
        self._approval = approval_engine or ApprovalEngine.create_default()
        self._events = event_store or EventStore()
        self._signals = signal_collector or SignalCollector()
        self._feedback = feedback_collector or FeedbackCollector()

        self._ingestor = CallTranscriptIngestor()

    def process_call(
        self,
        transcript: str,
        opportunity_id: UUID,
        participants: list,
        duration_seconds: int,
        call_metadata: dict = None,
        user_id: str = None
    ) -> PostCallResult:
        """
        Process a call transcript end-to-end.

        This is the main entry point for the use case.
        """
        start_time = datetime.now()
        result = PostCallResult(opportunity_id=opportunity_id)

        # Step 1: Ingest the call
        signal = self._ingest_call(
            transcript, participants, duration_seconds, call_metadata
        )

        # Log event
        self._events.append(
            EventBuilder()
            .signal("call_ingested")
            .for_opportunity(opportunity_id)
            .with_payload({"duration_seconds": duration_seconds})
            .by_system("post_call_use_case")
            .build()
        )

        # Step 2: Build reasoning context
        context = self._build_context(opportunity_id, transcript)

        # Step 3: Run reasoning pipeline
        reasoning_results = self._run_reasoning_pipeline(context)

        result.summary = reasoning_results.get("summary", {})
        result.objections = reasoning_results.get("objections", [])
        result.stakeholders = reasoning_results.get("stakeholders", [])
        result.risks = reasoning_results.get("risks", [])
        result.next_steps = reasoning_results.get("next_steps", [])

        # Log inference event
        self._events.append(
            EventBuilder()
            .inference("call_analyzed")
            .for_opportunity(opportunity_id)
            .with_payload({
                "objections_found": len(result.objections),
                "risks_found": len(result.risks),
                "stakeholders_mapped": len(result.stakeholders)
            })
            .by_system("reasoning_engine")
            .build()
        )

        # Step 4: Generate CRM update
        result.crm_update = self._generate_crm_update(result)

        # Step 5: Update working memory
        self._update_memory(opportunity_id, result)

        # Step 6: Request approval
        approval_request = self._request_approval(result, user_id)
        result.approval_request_id = approval_request.id

        # Calculate metrics
        end_time = datetime.now()
        result.processing_time_seconds = (end_time - start_time).total_seconds()
        result.admin_time_saved_minutes = self._estimate_time_saved(result)

        return result

    def _ingest_call(
        self,
        transcript: str,
        participants: list,
        duration_seconds: int,
        metadata: dict = None
    ) -> CognitiveSignal:
        """Ingest call data into a cognitive signal."""
        raw_data = {
            "transcript": transcript,
            "participants": participants,
            "start_time": datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            **(metadata or {})
        }

        return self._ingestor.ingest(raw_data)

    def _build_context(
        self,
        opportunity_id: UUID,
        transcript: str
    ) -> ReasoningContext:
        """Build context for reasoning."""
        # Get working memory
        working_memory = self._memory.working.get_context(opportunity_id)

        # Get recent history from episodic memory
        history_entries = self._memory.episodic.get_by_opportunity(opportunity_id)
        relevant_history = [
            f"{e.key}: {e.value}" for e in history_entries[-5:]
        ]

        return ReasoningContext(
            opportunity_id=opportunity_id,
            primary_content=transcript,
            working_memory=working_memory,
            relevant_history=relevant_history
        )

    def _run_reasoning_pipeline(self, context: ReasoningContext) -> dict:
        """Run the full reasoning pipeline."""
        results = {}

        # Summarization
        summary_result = self._reasoning.execute(TaskType.SUMMARIZATION, context)
        results["summary"] = summary_result.structured_output

        # Objection extraction
        objection_result = self._reasoning.execute(TaskType.OBJECTION_EXTRACTION, context)
        results["objections"] = objection_result.structured_output.get("objections", [])

        # Stakeholder mapping
        stakeholder_result = self._reasoning.execute(TaskType.STAKEHOLDER_MAPPING, context)
        results["stakeholders"] = stakeholder_result.structured_output.get("stakeholders", [])

        # Risk detection
        risk_result = self._reasoning.execute(TaskType.RISK_DETECTION, context)
        results["risks"] = risk_result.structured_output.get("risks", [])

        # Next step planning
        next_step_result = self._reasoning.execute(TaskType.NEXT_STEP_PLANNING, context)
        results["next_steps"] = next_step_result.structured_output.get("actions", [])

        return results

    def _generate_crm_update(self, result: PostCallResult) -> dict:
        """Generate structured CRM update."""
        return {
            "opportunity_id": str(result.opportunity_id),
            "notes": result.summary.get("executive_summary", ""),
            "key_discussion_points": result.summary.get("key_points", []),
            "objections": [
                {
                    "category": o.get("category"),
                    "description": o.get("quote"),
                    "severity": o.get("severity")
                }
                for o in result.objections
            ],
            "stakeholder_updates": [
                {
                    "name": s.get("name"),
                    "role": s.get("role"),
                    "sentiment": s.get("sentiment")
                }
                for s in result.stakeholders
            ],
            "risks": [
                {
                    "category": r.get("category"),
                    "description": r.get("description"),
                    "severity": r.get("severity")
                }
                for r in result.risks
            ],
            "next_steps": [
                {
                    "action": ns.get("action"),
                    "priority": ns.get("priority"),
                    "target": ns.get("target")
                }
                for ns in result.next_steps
            ],
            "generated_at": datetime.now().isoformat()
        }

    def _update_memory(self, opportunity_id: UUID, result: PostCallResult) -> None:
        """Update working memory with analysis results."""
        self._memory.working.store(
            "last_call_summary",
            result.summary.get("executive_summary", ""),
            opportunity_id=opportunity_id,
            importance=0.8
        )

        if result.objections:
            self._memory.working.store(
                "active_objections",
                result.objections,
                opportunity_id=opportunity_id,
                importance=0.9
            )

        if result.risks:
            self._memory.working.store(
                "identified_risks",
                result.risks,
                opportunity_id=opportunity_id,
                importance=0.9
            )

    def _request_approval(
        self,
        result: PostCallResult,
        user_id: str = None
    ) -> ApprovalRequest:
        """Request human approval for CRM update."""
        return self._approval.create_request(
            gate_name="crm_update",
            action_type="update_crm",
            action_description="Update opportunity with post-call intelligence",
            proposed_content=result.crm_update,
            context={
                "confidence": 0.85,
                "objections_count": len(result.objections),
                "risks_count": len(result.risks)
            },
            opportunity_id=result.opportunity_id,
            playbook_name="post_call_intelligence",
            step_name="crm_update"
        )

    def _estimate_time_saved(self, result: PostCallResult) -> float:
        """Estimate admin time saved by automation."""
        # Baseline estimates per task (minutes)
        time_estimates = {
            "summarization": 10,
            "objection_extraction": 5,
            "stakeholder_mapping": 8,
            "risk_detection": 5,
            "crm_update": 15
        }

        return sum(time_estimates.values())

    def handle_approval(
        self,
        approval_request_id: UUID,
        decision: str,  # "approve", "reject", "modify"
        user_id: str,
        modified_content: dict = None,
        feedback_text: str = None,
        feedback_tags: list = None
    ) -> dict:
        """
        Handle the human approval decision.

        This completes the governance loop and captures feedback for learning.
        """
        result = {"status": decision, "request_id": str(approval_request_id)}

        if decision == "approve":
            approval = self._approval.approve(
                approval_request_id,
                approved_by=user_id,
                rationale=feedback_text or ""
            )
            result["applied"] = True

            # Record positive outcome signal
            self._signals.record(OutcomeSignal(
                outcome_type=OutcomeType.NBA_ACCEPTED,
                triggering_action_id=approval_request_id
            ))

        elif decision == "modify":
            approval = self._approval.modify_and_approve(
                approval_request_id,
                approved_by=user_id,
                modified_content=modified_content,
                modification_notes=feedback_text or "",
                feedback_tags=feedback_tags or []
            )
            result["applied"] = True
            result["modified"] = True

            # Record modification feedback
            self._feedback.record(FeedbackEntry(
                feedback_type=FeedbackType.MODIFICATION,
                target_type="crm_update",
                target_id=approval_request_id,
                user_id=user_id,
                original_content=approval.modified_content,  # Would need to store original
                corrected_content=modified_content,
                feedback_text=feedback_text or "",
                tags=feedback_tags or []
            ))

            self._signals.record(OutcomeSignal(
                outcome_type=OutcomeType.NBA_MODIFIED,
                triggering_action_id=approval_request_id
            ))

        elif decision == "reject":
            self._approval.reject(
                approval_request_id,
                rejected_by=user_id,
                rejection_reason=feedback_text or "",
                feedback_tags=feedback_tags or []
            )
            result["applied"] = False

            # Record rejection feedback
            self._feedback.record(FeedbackEntry(
                feedback_type=FeedbackType.REJECTION,
                target_type="crm_update",
                target_id=approval_request_id,
                user_id=user_id,
                feedback_text=feedback_text or "",
                tags=feedback_tags or []
            ))

            self._signals.record(OutcomeSignal(
                outcome_type=OutcomeType.NBA_REJECTED,
                triggering_action_id=approval_request_id
            ))

        # Log governance event
        self._events.append(
            EventBuilder()
            .governance(f"crm_update_{decision}")
            .with_payload(result)
            .by_user(user_id)
            .build()
        )

        return result
