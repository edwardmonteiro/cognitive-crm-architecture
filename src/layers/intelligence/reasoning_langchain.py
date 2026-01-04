"""
LangChain-based Reasoning Engine

Implements the Intelligence Layer reasoning capabilities using:
- LangChain for LLM orchestration
- Pydantic for structured outputs
- LCEL (LangChain Expression Language) for chains

From Section 4.2: "LLM reasoning: summarisation, objection extraction,
stakeholder mapping, risk detection, next-step planning"
"""

from typing import Optional, Type, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel

from .schemas import (
    InteractionSummary,
    ObjectionExtractionResult,
    StakeholderMappingResult,
    RiskDetectionResult,
    NextBestActionResult,
    FullInteractionAnalysis
)


class TaskType(str, Enum):
    """Types of reasoning tasks."""
    SUMMARIZATION = "summarization"
    OBJECTION_EXTRACTION = "objection_extraction"
    STAKEHOLDER_MAPPING = "stakeholder_mapping"
    RISK_DETECTION = "risk_detection"
    NEXT_STEP_PLANNING = "next_step_planning"
    FULL_ANALYSIS = "full_analysis"


# =============================================================================
# System Prompts for Each Task
# =============================================================================

SUMMARIZATION_SYSTEM_PROMPT = """You are an expert sales analyst specializing in customer interaction analysis.
Your task is to provide a structured summary of sales conversations that helps sellers quickly understand key points.

Focus on:
1. The main purpose and outcome of the interaction
2. Key topics discussed
3. Customer sentiment and engagement level
4. Any commitments or action items mentioned

Be factual and objective. Do not make assumptions beyond what is explicitly stated."""


OBJECTION_EXTRACTION_SYSTEM_PROMPT = """You are an expert sales objection analyst.
Your task is to identify and categorize customer objections from sales interactions.

Objection categories:
- PRICE: Cost, budget, ROI concerns
- TIMING: Timeline, urgency, prioritization issues
- AUTHORITY: Decision-making power, stakeholder involvement
- NEED: Problem fit, requirements mismatch
- TRUST: Credibility concerns, need for references
- COMPETITION: Alternative solutions mentioned
- TECHNICAL: Technical requirements or concerns
- OTHER: Objections that don't fit other categories

For each objection, assess its severity and suggest how to address it.
Only identify genuine objections, not general questions or comments."""


STAKEHOLDER_MAPPING_SYSTEM_PROMPT = """You are an expert sales strategist analyzing buying committee dynamics.
Your task is to map stakeholders and their roles in the buying process.

Stakeholder roles:
- DECISION_MAKER: Has final authority to approve the purchase
- ECONOMIC_BUYER: Controls budget and financial approval
- TECHNICAL_BUYER: Evaluates technical fit and requirements
- INFLUENCER: Shapes opinions but doesn't decide
- CHAMPION: Internal advocate supporting the deal
- BLOCKER: Actively opposing or creating obstacles
- END_USER: Will use the solution daily
- UNKNOWN: Role not yet clear

Assess engagement level and sentiment for each stakeholder.
Note relationship dynamics and power structures when apparent."""


RISK_DETECTION_SYSTEM_PROMPT = """You are an expert deal risk analyst for enterprise sales.
Your task is to identify risks that could prevent or delay deal closure.

Risk categories:
- COMPETITIVE: Competitor involvement or preference
- BUDGET: Funding issues, approval delays
- TIMELINE: Delayed decisions, shifted priorities
- CHAMPION: Loss or weakness of internal advocate
- ENGAGEMENT: Reduced responsiveness, ghosting
- REQUIREMENTS: Scope creep, changing needs
- DECISION_PROCESS: Unclear process, committee changes
- TECHNICAL: Integration or technical concerns
- ORGANIZATIONAL: Restructuring, leadership changes

Be thorough but avoid false positives. Flag potential risks early with clear warning signals.
Suggest specific mitigation actions for each identified risk."""


NEXT_STEP_PLANNING_SYSTEM_PROMPT = """You are an expert sales execution advisor.
Your task is to recommend specific next-best-actions to advance deals.

Consider:
- Current deal stage and momentum
- Recent interactions and signals
- Stakeholder dynamics and engagement
- Identified risks and objections
- Time sensitivity and urgency

For each recommendation:
1. Be specific and actionable
2. Explain the rationale
3. Define success criteria
4. Estimate confidence level

Prioritize actions by impact and urgency.
Limit to 3-5 actionable recommendations."""


FULL_ANALYSIS_SYSTEM_PROMPT = """You are a comprehensive sales intelligence system.
Analyze the provided customer interaction and provide a complete structured analysis including:

1. Summary: Key takeaways and sentiment
2. Objections: Customer concerns and pushback
3. Stakeholders: People involved and their roles
4. Risks: Threats to deal progression
5. Next Actions: Recommended steps to advance the deal

Be thorough, objective, and actionable in your analysis."""


# =============================================================================
# Task-Specific Prompt Templates
# =============================================================================

SUMMARIZATION_USER_TEMPLATE = """Analyze the following customer interaction and provide a structured summary:

## Interaction Content:
{content}

## Additional Context:
{context}

Provide a comprehensive but concise summary focusing on actionable insights."""


OBJECTION_EXTRACTION_USER_TEMPLATE = """Extract and analyze customer objections from this interaction:

## Interaction Content:
{content}

## Known Objections from Past Interactions:
{past_objections}

## Additional Context:
{context}

Identify all objections, categorize them, and suggest responses."""


STAKEHOLDER_MAPPING_USER_TEMPLATE = """Map the stakeholders mentioned in this interaction:

## Interaction Content:
{content}

## Known Contacts:
{known_contacts}

## Additional Context:
{context}

Identify all stakeholders, their roles, and relationship dynamics."""


RISK_DETECTION_USER_TEMPLATE = """Analyze this interaction for deal risks:

## Interaction Content:
{content}

## Opportunity Context:
- Stage: {stage}
- Days in Stage: {days_in_stage}
- Amount: {amount}
- Expected Close: {close_date}

## Recent History:
{history}

Identify all risks with severity ratings and mitigation suggestions."""


NEXT_STEP_PLANNING_USER_TEMPLATE = """Recommend next-best-actions based on this analysis:

## Recent Interaction:
{content}

## Opportunity Context:
- Stage: {stage}
- Amount: {amount}
- Days in Stage: {days_in_stage}

## Identified Risks:
{risks}

## Open Objections:
{objections}

## Relevant Playbook Guidance:
{playbook_guidance}

Provide 3-5 prioritized, actionable recommendations."""


# =============================================================================
# Reasoning Context
# =============================================================================

@dataclass
class ReasoningContext:
    """Context provided to reasoning tasks."""
    opportunity_id: UUID = field(default_factory=uuid4)
    account_id: Optional[UUID] = None

    # Input content
    primary_content: str = ""
    additional_context: dict = field(default_factory=dict)

    # Retrieved context (from RAG)
    retrieved_evidence: list = field(default_factory=list)

    # Memory context
    working_memory: dict = field(default_factory=dict)
    relevant_history: list = field(default_factory=list)

    # Policy constraints
    policies: list = field(default_factory=list)


@dataclass
class ReasoningResult:
    """Result of a reasoning task."""
    id: UUID = field(default_factory=uuid4)
    task_type: TaskType = TaskType.SUMMARIZATION
    timestamp: datetime = field(default_factory=datetime.now)

    # Structured output
    output: Optional[BaseModel] = None
    raw_output: Optional[str] = None

    # Quality metrics
    confidence: float = 0.0
    latency_ms: float = 0.0

    # Governance
    requires_review: bool = False
    review_reason: Optional[str] = None

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0


# =============================================================================
# LangChain Reasoning Engine
# =============================================================================

class LangChainReasoningEngine:
    """
    Reasoning engine using LangChain for structured LLM outputs.

    Features:
    - Multiple LLM provider support
    - Structured outputs with Pydantic validation
    - Automatic retries and error handling
    - Token usage tracking
    """

    # Mapping of task types to their configurations
    TASK_CONFIG: Dict[TaskType, Dict[str, Any]] = {
        TaskType.SUMMARIZATION: {
            "schema": InteractionSummary,
            "system_prompt": SUMMARIZATION_SYSTEM_PROMPT,
            "user_template": SUMMARIZATION_USER_TEMPLATE
        },
        TaskType.OBJECTION_EXTRACTION: {
            "schema": ObjectionExtractionResult,
            "system_prompt": OBJECTION_EXTRACTION_SYSTEM_PROMPT,
            "user_template": OBJECTION_EXTRACTION_USER_TEMPLATE
        },
        TaskType.STAKEHOLDER_MAPPING: {
            "schema": StakeholderMappingResult,
            "system_prompt": STAKEHOLDER_MAPPING_SYSTEM_PROMPT,
            "user_template": STAKEHOLDER_MAPPING_USER_TEMPLATE
        },
        TaskType.RISK_DETECTION: {
            "schema": RiskDetectionResult,
            "system_prompt": RISK_DETECTION_SYSTEM_PROMPT,
            "user_template": RISK_DETECTION_USER_TEMPLATE
        },
        TaskType.NEXT_STEP_PLANNING: {
            "schema": NextBestActionResult,
            "system_prompt": NEXT_STEP_PLANNING_SYSTEM_PROMPT,
            "user_template": NEXT_STEP_PLANNING_USER_TEMPLATE
        },
        TaskType.FULL_ANALYSIS: {
            "schema": FullInteractionAnalysis,
            "system_prompt": FULL_ANALYSIS_SYSTEM_PROMPT,
            "user_template": "{content}\n\nContext: {context}"
        }
    }

    def __init__(self, llm_provider=None, settings=None):
        """
        Initialize the reasoning engine.

        Args:
            llm_provider: LLMProvider instance (optional, will create default)
            settings: Settings instance (optional)
        """
        self._provider = llm_provider
        self._settings = settings
        self._chains: Dict[TaskType, Any] = {}

    def _get_provider(self):
        """Lazy load LLM provider."""
        if self._provider is None:
            from ...config.providers import LLMProvider
            self._provider = LLMProvider()
        return self._provider

    def _get_chain(self, task_type: TaskType):
        """Get or create a chain for the task type."""
        if task_type not in self._chains:
            self._chains[task_type] = self._create_chain(task_type)
        return self._chains[task_type]

    def _create_chain(self, task_type: TaskType):
        """Create a LangChain chain for structured output."""
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.messages import SystemMessage, HumanMessage

            config = self.TASK_CONFIG[task_type]
            schema = config["schema"]

            # Get chat model with structured output
            provider = self._get_provider()
            structured_llm = provider.with_structured_output(schema)

            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", config["system_prompt"]),
                ("human", config["user_template"])
            ])

            # Create chain using LCEL
            chain = prompt | structured_llm

            return chain

        except ImportError as e:
            # Return None if LangChain not installed
            return None

    def execute(
        self,
        task_type: TaskType,
        context: ReasoningContext
    ) -> ReasoningResult:
        """
        Execute a reasoning task.

        Args:
            task_type: Type of reasoning task
            context: Context for the task

        Returns:
            ReasoningResult with structured output
        """
        start_time = datetime.now()
        result = ReasoningResult(task_type=task_type)

        try:
            chain = self._get_chain(task_type)

            if chain is None:
                # Fallback to mock if LangChain not available
                result.output = self._mock_execute(task_type, context)
                result.confidence = 0.5
                result.requires_review = True
                result.review_reason = "LangChain not installed, using mock output"
            else:
                # Prepare input variables
                input_vars = self._prepare_input(task_type, context)

                # Execute chain
                output = chain.invoke(input_vars)
                result.output = output
                result.confidence = 0.85

                # Check if review needed
                if hasattr(output, 'requires_human_review') and output.requires_human_review:
                    result.requires_review = True
                    result.review_reason = getattr(output, 'review_reason', 'Model flagged for review')

        except Exception as e:
            result.raw_output = str(e)
            result.requires_review = True
            result.review_reason = f"Execution error: {str(e)}"
            result.output = self._mock_execute(task_type, context)

        # Calculate latency
        end_time = datetime.now()
        result.latency_ms = (end_time - start_time).total_seconds() * 1000

        return result

    def _prepare_input(self, task_type: TaskType, context: ReasoningContext) -> dict:
        """Prepare input variables for the chain."""
        base_vars = {
            "content": context.primary_content,
            "context": str(context.additional_context)
        }

        # Add task-specific variables
        if task_type == TaskType.OBJECTION_EXTRACTION:
            base_vars["past_objections"] = str(context.working_memory.get("objections", []))

        elif task_type == TaskType.STAKEHOLDER_MAPPING:
            base_vars["known_contacts"] = str(context.working_memory.get("known_contacts", []))

        elif task_type == TaskType.RISK_DETECTION:
            base_vars.update({
                "stage": context.working_memory.get("stage", "unknown"),
                "days_in_stage": context.working_memory.get("days_in_stage", 0),
                "amount": context.working_memory.get("amount", "unknown"),
                "close_date": context.working_memory.get("close_date", "unknown"),
                "history": "\n".join(context.relevant_history[-5:]) if context.relevant_history else "No recent history"
            })

        elif task_type == TaskType.NEXT_STEP_PLANNING:
            base_vars.update({
                "stage": context.working_memory.get("stage", "unknown"),
                "amount": context.working_memory.get("amount", "unknown"),
                "days_in_stage": context.working_memory.get("days_in_stage", 0),
                "risks": str(context.working_memory.get("risks", [])),
                "objections": str(context.working_memory.get("objections", [])),
                "playbook_guidance": "\n".join([
                    e.get("content", "")[:200] for e in context.retrieved_evidence[:3]
                ]) if context.retrieved_evidence else "No specific guidance available"
            })

        return base_vars

    def _mock_execute(self, task_type: TaskType, context: ReasoningContext) -> BaseModel:
        """Generate mock output when LLM is not available."""
        if task_type == TaskType.SUMMARIZATION:
            return InteractionSummary(
                executive_summary="Customer discussed their CRM needs and expressed interest in the solution.",
                key_discussion_points=["Current system challenges", "Budget considerations", "Timeline requirements"],
                customer_sentiment="positive",
                commitments_made=["Follow-up meeting scheduled"],
                next_steps_mentioned=["Send proposal", "Demo with VP"],
                follow_up_required=True
            )

        elif task_type == TaskType.OBJECTION_EXTRACTION:
            from .schemas import Objection, ObjectionCategory
            return ObjectionExtractionResult(
                objections=[
                    Objection(
                        quote="The price seems higher than competitors",
                        category=ObjectionCategory.PRICE,
                        severity="medium",
                        is_addressed=False,
                        suggested_response="Emphasize ROI and total cost of ownership"
                    )
                ],
                overall_resistance_level="medium",
                primary_concern="Budget constraints"
            )

        elif task_type == TaskType.STAKEHOLDER_MAPPING:
            from .schemas import Stakeholder, StakeholderRole
            return StakeholderMappingResult(
                stakeholders=[
                    Stakeholder(
                        name="Primary Contact",
                        role=StakeholderRole.CHAMPION,
                        engagement_level="high",
                        sentiment="positive",
                        key_concerns=["Implementation timeline"],
                        influence_score=0.7
                    )
                ],
                buying_committee_complete=False,
                missing_roles=[StakeholderRole.ECONOMIC_BUYER],
                power_dynamics_summary="Champion engaged, need to identify decision maker"
            )

        elif task_type == TaskType.RISK_DETECTION:
            from .schemas import Risk, RiskCategory
            return RiskDetectionResult(
                risks=[
                    Risk(
                        description="Competitor mentioned in discussion",
                        category=RiskCategory.COMPETITIVE,
                        severity="medium",
                        warning_signals=["Customer comparing features"],
                        mitigation_suggestion="Provide competitive battle card",
                        probability=0.4
                    )
                ],
                overall_risk_level="medium",
                immediate_actions_needed=["Gather competitive intelligence"],
                deal_health_score=0.7
            )

        elif task_type == TaskType.NEXT_STEP_PLANNING:
            from .schemas import RecommendedAction, ActionPriority
            return NextBestActionResult(
                recommendations=[
                    RecommendedAction(
                        action="Send follow-up email with pricing proposal",
                        action_type="follow_up",
                        priority=ActionPriority.IMMEDIATE,
                        target_stakeholder="Primary Contact",
                        expected_outcome="Budget discussion scheduled",
                        success_criteria="Meeting confirmed within 1 week",
                        confidence=0.8,
                        rationale="Customer expressed interest but needs pricing details"
                    )
                ],
                strategic_focus="Secure budget approval",
                key_milestone="VP demo scheduled"
            )

        else:
            # Return a basic summary for unknown task types
            return InteractionSummary(
                executive_summary="Analysis completed",
                key_discussion_points=[],
                customer_sentiment="neutral",
                commitments_made=[],
                next_steps_mentioned=[],
                follow_up_required=True
            )

    async def execute_async(
        self,
        task_type: TaskType,
        context: ReasoningContext
    ) -> ReasoningResult:
        """
        Execute a reasoning task asynchronously.

        Args:
            task_type: Type of reasoning task
            context: Context for the task

        Returns:
            ReasoningResult with structured output
        """
        start_time = datetime.now()
        result = ReasoningResult(task_type=task_type)

        try:
            chain = self._get_chain(task_type)

            if chain is None:
                result.output = self._mock_execute(task_type, context)
                result.confidence = 0.5
            else:
                input_vars = self._prepare_input(task_type, context)
                output = await chain.ainvoke(input_vars)
                result.output = output
                result.confidence = 0.85

        except Exception as e:
            result.raw_output = str(e)
            result.requires_review = True
            result.review_reason = f"Execution error: {str(e)}"
            result.output = self._mock_execute(task_type, context)

        end_time = datetime.now()
        result.latency_ms = (end_time - start_time).total_seconds() * 1000

        return result

    def analyze_interaction(self, context: ReasoningContext) -> Dict[str, ReasoningResult]:
        """
        Run full analysis pipeline on an interaction.

        Executes all reasoning tasks and returns combined results.
        """
        results = {}

        for task_type in [
            TaskType.SUMMARIZATION,
            TaskType.OBJECTION_EXTRACTION,
            TaskType.STAKEHOLDER_MAPPING,
            TaskType.RISK_DETECTION,
            TaskType.NEXT_STEP_PLANNING
        ]:
            results[task_type.value] = self.execute(task_type, context)

        return results
