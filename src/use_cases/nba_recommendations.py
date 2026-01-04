"""
Use Case 2: Evidence-based Next-Best-Action Recommendations

From Section 5.1:
"Evidence-based next-best-action (NBA) recommendations with justification"

This use case demonstrates:
1. Context gathering from multiple sources
2. RAG-grounded reasoning
3. Evidence-based recommendations
4. Justification generation
5. Seller presentation and feedback
6. Outcome tracking for learning

Key metric: NBA acceptance rate (target: 68.2% from Table 1)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from ..core.entities import Opportunity, OpportunityStage
from ..layers.intelligence.reasoning import ReasoningEngine, ReasoningContext, TaskType
from ..layers.intelligence.rag import RAGEngine, SourceType, RetrievedContext
from ..layers.intelligence.memory import MemoryManager
from ..layers.data_ingestion.event_schema import EventBuilder, EventStore
from ..layers.learning.signals import SignalCollector, OutcomeSignal, OutcomeType
from ..layers.learning.feedback import FeedbackCollector, FeedbackEntry, FeedbackType


@dataclass
class NBARecommendation:
    """A single NBA recommendation with evidence."""
    id: UUID = field(default_factory=uuid4)
    action: str = ""
    action_type: str = ""  # follow_up, schedule_demo, send_proposal, etc.
    priority: str = "normal"  # immediate, this_week, this_month
    target_stakeholder: str = ""

    # Expected outcome
    expected_outcome: str = ""
    success_criteria: str = ""

    # Evidence and justification
    justification: str = ""
    evidence_sources: list = field(default_factory=list)
    confidence_score: float = 0.0

    # Presentation
    rationale_summary: str = ""


@dataclass
class NBAResult:
    """Result of NBA recommendation generation."""
    id: UUID = field(default_factory=uuid4)
    opportunity_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)

    # Recommendations
    recommendations: list = field(default_factory=list)  # List of NBARecommendation

    # Context used
    context_summary: dict = field(default_factory=dict)
    evidence_retrieved: int = 0

    # Seller interaction
    presented_at: Optional[datetime] = None
    decision_at: Optional[datetime] = None
    decision: str = ""  # accepted, modified, rejected, ignored
    selected_recommendation_id: Optional[UUID] = None

    # Metrics
    generation_time_seconds: float = 0.0


class NBARecommendationsUseCase:
    """
    Implements the NBA recommendations use case.

    Flow:
    1. Gather opportunity context
    2. Retrieve relevant evidence (RAG)
    3. Generate recommendations
    4. Add evidence-based justifications
    5. Present to seller
    6. Track decision and outcome
    """

    def __init__(
        self,
        reasoning_engine: ReasoningEngine = None,
        rag_engine: RAGEngine = None,
        memory_manager: MemoryManager = None,
        event_store: EventStore = None,
        signal_collector: SignalCollector = None,
        feedback_collector: FeedbackCollector = None
    ):
        self._reasoning = reasoning_engine or ReasoningEngine()
        self._rag = rag_engine or RAGEngine()
        self._memory = memory_manager or MemoryManager()
        self._events = event_store or EventStore()
        self._signals = signal_collector or SignalCollector()
        self._feedback = feedback_collector or FeedbackCollector()

    def generate_recommendations(
        self,
        opportunity_id: UUID,
        opportunity_data: dict = None,
        max_recommendations: int = 5
    ) -> NBAResult:
        """
        Generate NBA recommendations for an opportunity.

        This is the main entry point for the use case.
        """
        start_time = datetime.now()
        result = NBAResult(opportunity_id=opportunity_id)

        # Step 1: Gather context
        context = self._gather_context(opportunity_id, opportunity_data)
        result.context_summary = self._summarize_context(context)

        # Step 2: Retrieve relevant evidence
        evidence = self._retrieve_evidence(context)
        result.evidence_retrieved = len(evidence)

        # Update context with evidence
        context.retrieved_evidence = [
            {"content": e.content, "source": e.source_type.value, "score": e.relevance_score}
            for e in evidence
        ]

        # Step 3: Generate recommendations
        raw_recommendations = self._generate_raw_recommendations(context)

        # Step 4: Add justifications
        recommendations = self._add_justifications(raw_recommendations, evidence)

        # Step 5: Rank and filter
        result.recommendations = self._rank_recommendations(
            recommendations,
            max_recommendations
        )

        # Log event
        self._events.append(
            EventBuilder()
            .inference("nba_generated")
            .for_opportunity(opportunity_id)
            .with_payload({
                "num_recommendations": len(result.recommendations),
                "evidence_sources": result.evidence_retrieved
            })
            .by_system("nba_use_case")
            .build()
        )

        end_time = datetime.now()
        result.generation_time_seconds = (end_time - start_time).total_seconds()

        return result

    def _gather_context(
        self,
        opportunity_id: UUID,
        opportunity_data: dict = None
    ) -> ReasoningContext:
        """Gather all relevant context for the opportunity."""
        # Get working memory
        working_memory = self._memory.working.get_context(opportunity_id)

        # Get opportunity data
        opp_data = opportunity_data or {}
        working_memory.update({
            "stage": opp_data.get("stage", "unknown"),
            "amount": opp_data.get("amount"),
            "close_date": opp_data.get("close_date"),
            "days_in_stage": opp_data.get("days_in_stage"),
            "last_activity": opp_data.get("last_activity")
        })

        # Get episodic memory
        history_entries = self._memory.episodic.get_by_opportunity(opportunity_id)
        relevant_history = [e.value for e in history_entries[-5:]]

        # Build primary content from context
        primary_content = self._format_context_for_reasoning(working_memory, relevant_history)

        return ReasoningContext(
            opportunity_id=opportunity_id,
            primary_content=primary_content,
            working_memory=working_memory,
            relevant_history=[str(h) for h in relevant_history]
        )

    def _format_context_for_reasoning(
        self,
        working_memory: dict,
        history: list
    ) -> str:
        """Format context as text for reasoning."""
        lines = ["Current Opportunity Status:"]

        for key, value in working_memory.items():
            if value is not None:
                lines.append(f"- {key}: {value}")

        if history:
            lines.append("\nRecent Activity:")
            for item in history[-3:]:
                lines.append(f"- {item}")

        return "\n".join(lines)

    def _summarize_context(self, context: ReasoningContext) -> dict:
        """Create a summary of the context used."""
        return {
            "opportunity_id": str(context.opportunity_id),
            "working_memory_keys": list(context.working_memory.keys()),
            "history_items": len(context.relevant_history),
            "retrieved_evidence": len(context.retrieved_evidence)
        }

    def _retrieve_evidence(self, context: ReasoningContext) -> list[RetrievedContext]:
        """Retrieve relevant evidence from knowledge bases."""
        # Build query from context
        query = context.primary_content

        # Retrieve from relevant sources
        retrieval_result = self._rag.retrieve(
            query=query,
            source_types=[
                SourceType.SALES_PLAYBOOK,
                SourceType.INTERACTION_HISTORY,
                SourceType.COMPETITIVE_INTEL
            ],
            top_k=10
        )

        return retrieval_result.chunks

    def _generate_raw_recommendations(self, context: ReasoningContext) -> list[dict]:
        """Generate raw recommendations using reasoning engine."""
        result = self._reasoning.execute(TaskType.NEXT_STEP_PLANNING, context)
        return result.structured_output.get("actions", [])

    def _add_justifications(
        self,
        recommendations: list[dict],
        evidence: list[RetrievedContext]
    ) -> list[NBARecommendation]:
        """Add evidence-based justifications to recommendations."""
        justified = []

        for rec in recommendations:
            # Find relevant evidence for this recommendation
            relevant_evidence = self._find_relevant_evidence(
                rec.get("action", ""),
                evidence
            )

            # Build justification
            justification = self._build_justification(rec, relevant_evidence)

            nba = NBARecommendation(
                action=rec.get("action", ""),
                action_type=self._classify_action_type(rec.get("action", "")),
                priority=rec.get("priority", "normal"),
                target_stakeholder=rec.get("target", ""),
                expected_outcome=rec.get("expected_outcome", ""),
                success_criteria=rec.get("success_criteria", ""),
                justification=justification,
                evidence_sources=[
                    {
                        "title": e.title,
                        "source_type": e.source_type.value,
                        "relevance": e.relevance_score
                    }
                    for e in relevant_evidence
                ],
                confidence_score=self._calculate_confidence(rec, relevant_evidence),
                rationale_summary=self._generate_rationale_summary(rec, relevant_evidence)
            )

            justified.append(nba)

        return justified

    def _find_relevant_evidence(
        self,
        action: str,
        evidence: list[RetrievedContext]
    ) -> list[RetrievedContext]:
        """Find evidence relevant to a specific action."""
        # Simple keyword matching (would be more sophisticated in production)
        action_lower = action.lower()
        relevant = []

        for e in evidence:
            content_lower = e.content.lower()
            # Check for keyword overlap
            action_words = set(action_lower.split())
            content_words = set(content_lower.split())
            overlap = action_words & content_words

            if len(overlap) >= 2 or e.relevance_score > 0.8:
                relevant.append(e)

        return relevant[:3]  # Top 3 most relevant

    def _build_justification(
        self,
        recommendation: dict,
        evidence: list[RetrievedContext]
    ) -> str:
        """Build a justification string from evidence."""
        if not evidence:
            return "Based on current opportunity context and best practices."

        justification_parts = ["This recommendation is supported by:"]

        for e in evidence[:2]:
            justification_parts.append(
                f"- {e.source_type.value}: {e.content[:100]}..."
            )

        return "\n".join(justification_parts)

    def _classify_action_type(self, action: str) -> str:
        """Classify the action into a category."""
        action_lower = action.lower()

        if any(word in action_lower for word in ["email", "follow", "reach"]):
            return "follow_up"
        elif any(word in action_lower for word in ["demo", "show", "present"]):
            return "schedule_demo"
        elif any(word in action_lower for word in ["proposal", "quote", "pricing"]):
            return "send_proposal"
        elif any(word in action_lower for word in ["objection", "concern", "address"]):
            return "address_objection"
        elif any(word in action_lower for word in ["escalate", "manager", "executive"]):
            return "escalate"
        else:
            return "general"

    def _calculate_confidence(
        self,
        recommendation: dict,
        evidence: list[RetrievedContext]
    ) -> float:
        """Calculate confidence score for a recommendation."""
        base_confidence = 0.5

        # Boost for evidence
        if evidence:
            evidence_boost = min(0.3, len(evidence) * 0.1)
            base_confidence += evidence_boost

        # Boost for high relevance scores
        if evidence:
            avg_relevance = sum(e.relevance_score for e in evidence) / len(evidence)
            base_confidence += avg_relevance * 0.2

        return min(0.95, base_confidence)

    def _generate_rationale_summary(
        self,
        recommendation: dict,
        evidence: list[RetrievedContext]
    ) -> str:
        """Generate a human-readable rationale summary."""
        action = recommendation.get("action", "Take action")
        priority = recommendation.get("priority", "normal")

        if evidence:
            source_types = set(e.source_type.value for e in evidence)
            sources_str = ", ".join(source_types)
            return f"{action} (Priority: {priority}). Supported by: {sources_str}"
        else:
            return f"{action} (Priority: {priority}). Based on opportunity analysis."

    def _rank_recommendations(
        self,
        recommendations: list[NBARecommendation],
        max_count: int
    ) -> list[NBARecommendation]:
        """Rank and filter recommendations."""
        # Sort by priority and confidence
        priority_order = {"immediate": 0, "this_week": 1, "this_month": 2, "normal": 3}

        sorted_recs = sorted(
            recommendations,
            key=lambda r: (
                priority_order.get(r.priority, 3),
                -r.confidence_score
            )
        )

        return sorted_recs[:max_count]

    def record_decision(
        self,
        result_id: UUID,
        decision: str,  # accepted, modified, rejected, ignored
        selected_recommendation_id: UUID = None,
        user_id: str = None,
        feedback_text: str = None,
        modified_action: str = None
    ) -> dict:
        """
        Record the seller's decision on recommendations.

        This completes the feedback loop for learning.
        """
        # Record outcome signal
        outcome_type_map = {
            "accepted": OutcomeType.NBA_ACCEPTED,
            "modified": OutcomeType.NBA_MODIFIED,
            "rejected": OutcomeType.NBA_REJECTED
        }

        if decision in outcome_type_map:
            self._signals.record(OutcomeSignal(
                outcome_type=outcome_type_map[decision],
                triggering_recommendation_id=selected_recommendation_id
            ))

        # Record feedback for learning
        if decision in ["modified", "rejected"]:
            feedback_type = (
                FeedbackType.MODIFICATION if decision == "modified"
                else FeedbackType.REJECTION
            )

            self._feedback.record(FeedbackEntry(
                feedback_type=feedback_type,
                target_type="nba_recommendation",
                target_id=selected_recommendation_id or result_id,
                user_id=user_id or "",
                corrected_content=modified_action if modified_action else None,
                feedback_text=feedback_text or ""
            ))

        # Log event
        self._events.append(
            EventBuilder()
            .governance(f"nba_{decision}")
            .with_payload({
                "result_id": str(result_id),
                "recommendation_id": str(selected_recommendation_id) if selected_recommendation_id else None,
                "decision": decision
            })
            .by_user(user_id or "unknown")
            .build()
        )

        return {
            "result_id": str(result_id),
            "decision": decision,
            "recorded_at": datetime.now().isoformat()
        }

    def get_acceptance_metrics(self, days: int = 30) -> dict:
        """
        Get NBA acceptance metrics for analysis.

        Target from paper: 68.2% acceptance rate
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)

        accepted = len([
            s for s in self._signals._signals
            if s.outcome_type == OutcomeType.NBA_ACCEPTED and s.timestamp >= cutoff
        ])
        modified = len([
            s for s in self._signals._signals
            if s.outcome_type == OutcomeType.NBA_MODIFIED and s.timestamp >= cutoff
        ])
        rejected = len([
            s for s in self._signals._signals
            if s.outcome_type == OutcomeType.NBA_REJECTED and s.timestamp >= cutoff
        ])

        total = accepted + modified + rejected

        return {
            "period_days": days,
            "total_decisions": total,
            "accepted": accepted,
            "modified": modified,
            "rejected": rejected,
            "acceptance_rate": accepted / total if total > 0 else 0,
            "acceptance_with_modification_rate": (accepted + modified) / total if total > 0 else 0
        }
