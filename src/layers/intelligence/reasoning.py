"""
LLM Reasoning Engine

From Section 4.2: "LLM reasoning: summarisation, objection extraction,
stakeholder mapping, risk detection, next-step planning"

This module implements the core reasoning capabilities using
Transformer-based language models. Each reasoning task is designed
to support the sensemaking process described in Section 3.

Key design principles:
- Structured outputs for downstream processing
- Confidence scoring for governance
- Evidence linking for transparency
- Grounding in retrieved context (RAG)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4


class TaskType(Enum):
    """Types of reasoning tasks."""
    SUMMARIZATION = "summarization"
    OBJECTION_EXTRACTION = "objection_extraction"
    STAKEHOLDER_MAPPING = "stakeholder_mapping"
    RISK_DETECTION = "risk_detection"
    NEXT_STEP_PLANNING = "next_step_planning"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    INTENT_CLASSIFICATION = "intent_classification"


@dataclass
class ReasoningContext:
    """Context provided to reasoning tasks."""
    opportunity_id: UUID = field(default_factory=uuid4)
    account_id: Optional[UUID] = None

    # Input content
    primary_content: str = ""
    additional_context: list = field(default_factory=list)

    # Retrieved context (from RAG)
    retrieved_evidence: list = field(default_factory=list)

    # Memory context
    working_memory: dict = field(default_factory=dict)
    relevant_history: list = field(default_factory=list)

    # Policy constraints
    policies: list = field(default_factory=list)


@dataclass
class ReasoningResult:
    """Output of a reasoning task."""
    id: UUID = field(default_factory=uuid4)
    task_type: TaskType = TaskType.SUMMARIZATION
    timestamp: datetime = field(default_factory=datetime.now)

    # Core output
    output: Any = None
    structured_output: dict = field(default_factory=dict)

    # Quality metrics
    confidence: float = 0.0
    uncertainty_reasons: list = field(default_factory=list)

    # Evidence and transparency
    evidence_used: list = field(default_factory=list)
    reasoning_trace: list = field(default_factory=list)

    # Governance
    requires_review: bool = False
    review_reason: Optional[str] = None


class ReasoningTask(ABC):
    """
    Abstract base class for reasoning tasks.

    Each task implements a specific cognitive function
    in the intelligence layer.
    """

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """The type of reasoning this task performs."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for the LLM."""
        pass

    @abstractmethod
    def format_input(self, context: ReasoningContext) -> str:
        """Format the input for the LLM."""
        pass

    @abstractmethod
    def parse_output(self, raw_output: str, context: ReasoningContext) -> ReasoningResult:
        """Parse LLM output into structured result."""
        pass

    def should_require_review(self, result: ReasoningResult) -> tuple[bool, str]:
        """Determine if result requires human review."""
        if result.confidence < 0.7:
            return True, "Low confidence output"
        return False, ""


class SummarizationTask(ReasoningTask):
    """
    Summarization of customer interactions.

    Transforms raw interaction content (transcripts, emails)
    into concise, actionable summaries.
    """

    @property
    def task_type(self) -> TaskType:
        return TaskType.SUMMARIZATION

    @property
    def system_prompt(self) -> str:
        return """You are a sales intelligence analyst. Summarize customer interactions
to help sellers understand key points quickly.

For each interaction, provide:
1. A brief executive summary (2-3 sentences)
2. Key discussion points
3. Customer sentiment indicators
4. Any commitments or next steps mentioned

Be factual and objective. Flag any uncertainty."""

    def format_input(self, context: ReasoningContext) -> str:
        prompt = f"""Summarize the following customer interaction:

INTERACTION:
{context.primary_content}

"""
        if context.relevant_history:
            prompt += "RECENT CONTEXT:\n"
            for item in context.relevant_history[-3:]:
                prompt += f"- {item}\n"

        prompt += "\nProvide a structured summary."
        return prompt

    def parse_output(self, raw_output: str, context: ReasoningContext) -> ReasoningResult:
        # Parse the summary output
        structured = {
            "executive_summary": "",
            "key_points": [],
            "sentiment": "neutral",
            "next_steps": [],
            "raw_output": raw_output
        }

        # Simple parsing logic (would be more sophisticated in production)
        lines = raw_output.strip().split('\n')
        current_section = "executive_summary"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            lower_line = line.lower()
            if 'key point' in lower_line or 'discussion' in lower_line:
                current_section = "key_points"
            elif 'sentiment' in lower_line:
                current_section = "sentiment"
            elif 'next step' in lower_line or 'commitment' in lower_line:
                current_section = "next_steps"
            elif current_section == "executive_summary":
                structured["executive_summary"] += line + " "
            elif current_section == "key_points" and line.startswith(('-', '*', '•')):
                structured["key_points"].append(line.lstrip('-*• '))
            elif current_section == "next_steps" and line.startswith(('-', '*', '•')):
                structured["next_steps"].append(line.lstrip('-*• '))

        structured["executive_summary"] = structured["executive_summary"].strip()

        return ReasoningResult(
            task_type=self.task_type,
            output=structured["executive_summary"],
            structured_output=structured,
            confidence=0.85,  # Would be computed from model
            evidence_used=[{"source": "primary_content", "used": True}]
        )


class ObjectionExtractionTask(ReasoningTask):
    """
    Extract customer objections from interactions.

    Identifies:
    - Explicit objections
    - Implied concerns
    - Competitive mentions
    - Blockers and risks
    """

    @property
    def task_type(self) -> TaskType:
        return TaskType.OBJECTION_EXTRACTION

    @property
    def system_prompt(self) -> str:
        return """You are a sales objection analyst. Extract and categorize
customer objections from sales interactions.

Categories:
- PRICE: Cost, budget, ROI concerns
- TIMING: Timeline, urgency, prioritization
- AUTHORITY: Decision-making, stakeholders
- NEED: Problem fit, requirements mismatch
- TRUST: Credibility, references, proof points
- COMPETITION: Alternative solutions mentioned

For each objection:
1. Quote the exact statement
2. Assign a category
3. Rate severity (high/medium/low)
4. Suggest a response approach"""

    def format_input(self, context: ReasoningContext) -> str:
        prompt = f"""Extract objections from this customer interaction:

INTERACTION:
{context.primary_content}

"""
        if context.retrieved_evidence:
            prompt += "RELEVANT CONTEXT (past objections, product info):\n"
            for evidence in context.retrieved_evidence[:3]:
                prompt += f"- {evidence.get('content', '')[:200]}\n"

        prompt += "\nList all objections found, or state 'No objections detected' if none."
        return prompt

    def parse_output(self, raw_output: str, context: ReasoningContext) -> ReasoningResult:
        objections = []

        # Parse objection output
        lines = raw_output.strip().split('\n')

        if 'no objection' in raw_output.lower():
            return ReasoningResult(
                task_type=self.task_type,
                output=[],
                structured_output={"objections": [], "count": 0},
                confidence=0.9
            )

        current_objection = {}
        for line in lines:
            line = line.strip()
            if not line:
                if current_objection:
                    objections.append(current_objection)
                    current_objection = {}
                continue

            # Simple parsing
            if line.startswith('"') or line.startswith("'"):
                current_objection["quote"] = line.strip('"\'')
            elif ':' in line:
                key, value = line.split(':', 1)
                key = key.lower().strip()
                value = value.strip()
                if 'categor' in key:
                    current_objection["category"] = value.upper()
                elif 'sever' in key:
                    current_objection["severity"] = value.lower()
                elif 'response' in key or 'suggest' in key:
                    current_objection["suggested_response"] = value

        if current_objection:
            objections.append(current_objection)

        return ReasoningResult(
            task_type=self.task_type,
            output=objections,
            structured_output={
                "objections": objections,
                "count": len(objections),
                "categories_found": list(set(o.get("category", "UNKNOWN") for o in objections))
            },
            confidence=0.8,
            requires_review=len(objections) > 3
        )


class StakeholderMappingTask(ReasoningTask):
    """
    Map stakeholders and their roles in the buying process.

    Identifies:
    - Decision makers
    - Influencers
    - Champions
    - Blockers
    - Relationship dynamics
    """

    @property
    def task_type(self) -> TaskType:
        return TaskType.STAKEHOLDER_MAPPING

    @property
    def system_prompt(self) -> str:
        return """You are a sales strategist analyzing buying committee dynamics.
Map stakeholders from customer interactions.

For each person mentioned:
1. Name and title (if available)
2. Role in buying process:
   - DECISION_MAKER: Final authority
   - ECONOMIC_BUYER: Controls budget
   - TECHNICAL_BUYER: Evaluates solution
   - INFLUENCER: Shapes opinions
   - CHAMPION: Internal advocate
   - BLOCKER: Opposing the deal
   - END_USER: Will use the product
3. Engagement level (high/medium/low)
4. Sentiment toward solution (positive/neutral/negative)
5. Key concerns or priorities

Also note relationship dynamics between stakeholders."""

    def format_input(self, context: ReasoningContext) -> str:
        prompt = f"""Map stakeholders from this interaction:

INTERACTION:
{context.primary_content}

"""
        if context.working_memory.get("known_contacts"):
            prompt += "KNOWN CONTACTS:\n"
            for contact in context.working_memory["known_contacts"]:
                prompt += f"- {contact.get('name')}: {contact.get('title')}\n"

        prompt += "\nProvide stakeholder analysis."
        return prompt

    def parse_output(self, raw_output: str, context: ReasoningContext) -> ReasoningResult:
        stakeholders = []

        # Simplified parsing
        lines = raw_output.strip().split('\n')

        current_stakeholder = {}
        for line in lines:
            line = line.strip()
            if not line:
                if current_stakeholder.get("name"):
                    stakeholders.append(current_stakeholder)
                    current_stakeholder = {}
                continue

            if ':' in line:
                key, value = line.split(':', 1)
                key = key.lower().strip()
                value = value.strip()

                if 'name' in key:
                    current_stakeholder["name"] = value
                elif 'title' in key:
                    current_stakeholder["title"] = value
                elif 'role' in key:
                    current_stakeholder["role"] = value.upper().replace(' ', '_')
                elif 'engagement' in key:
                    current_stakeholder["engagement"] = value.lower()
                elif 'sentiment' in key:
                    current_stakeholder["sentiment"] = value.lower()
                elif 'concern' in key or 'priorit' in key:
                    current_stakeholder["concerns"] = value

        if current_stakeholder.get("name"):
            stakeholders.append(current_stakeholder)

        return ReasoningResult(
            task_type=self.task_type,
            output=stakeholders,
            structured_output={
                "stakeholders": stakeholders,
                "total_count": len(stakeholders),
                "has_decision_maker": any(
                    s.get("role") == "DECISION_MAKER" for s in stakeholders
                ),
                "has_champion": any(
                    s.get("role") == "CHAMPION" for s in stakeholders
                ),
                "blockers_identified": [
                    s for s in stakeholders if s.get("role") == "BLOCKER"
                ]
            },
            confidence=0.75
        )


class RiskDetectionTask(ReasoningTask):
    """
    Detect risks to deal progression.

    Identifies:
    - Competitive threats
    - Budget constraints
    - Timeline risks
    - Champion changes
    - Engagement decline
    - Deal stagnation
    """

    @property
    def task_type(self) -> TaskType:
        return TaskType.RISK_DETECTION

    @property
    def system_prompt(self) -> str:
        return """You are a deal risk analyst. Identify risks to deal success.

Risk categories:
- COMPETITIVE: Competitor involvement or preference
- BUDGET: Funding issues, price sensitivity
- TIMELINE: Delayed decisions, shifted priorities
- CHAMPION: Loss of internal advocate
- ENGAGEMENT: Reduced responsiveness
- REQUIREMENTS: Scope creep, changing needs
- DECISION: Unclear process, committee changes

For each risk:
1. Description of the risk
2. Category
3. Severity (critical/high/medium/low)
4. Warning signals observed
5. Recommended mitigation

Be conservative - flag potential risks early."""

    def format_input(self, context: ReasoningContext) -> str:
        prompt = f"""Analyze deal risks from this interaction:

INTERACTION:
{context.primary_content}

OPPORTUNITY CONTEXT:
Stage: {context.working_memory.get('stage', 'Unknown')}
Days in stage: {context.working_memory.get('days_in_stage', 'Unknown')}
Last activity: {context.working_memory.get('last_activity', 'Unknown')}

"""
        if context.relevant_history:
            prompt += "RECENT HISTORY:\n"
            for item in context.relevant_history[-5:]:
                prompt += f"- {item}\n"

        prompt += "\nIdentify all risks and provide analysis."
        return prompt

    def parse_output(self, raw_output: str, context: ReasoningContext) -> ReasoningResult:
        risks = []

        if 'no risk' in raw_output.lower() or 'no significant' in raw_output.lower():
            return ReasoningResult(
                task_type=self.task_type,
                output=[],
                structured_output={
                    "risks": [],
                    "overall_risk_level": "low",
                    "critical_count": 0
                },
                confidence=0.85
            )

        # Parse risks
        lines = raw_output.strip().split('\n')
        current_risk = {}

        for line in lines:
            line = line.strip()
            if not line:
                if current_risk.get("description"):
                    risks.append(current_risk)
                    current_risk = {}
                continue

            if ':' in line:
                key, value = line.split(':', 1)
                key = key.lower().strip()
                value = value.strip()

                if 'description' in key or 'risk' in key:
                    current_risk["description"] = value
                elif 'category' in key:
                    current_risk["category"] = value.upper()
                elif 'severity' in key:
                    current_risk["severity"] = value.lower()
                elif 'signal' in key or 'warning' in key:
                    current_risk["signals"] = value
                elif 'mitigation' in key or 'recommend' in key:
                    current_risk["mitigation"] = value

        if current_risk.get("description"):
            risks.append(current_risk)

        # Compute overall risk level
        critical_count = sum(1 for r in risks if r.get("severity") == "critical")
        high_count = sum(1 for r in risks if r.get("severity") == "high")

        if critical_count > 0:
            overall = "critical"
        elif high_count > 1:
            overall = "high"
        elif high_count == 1 or len(risks) > 2:
            overall = "medium"
        else:
            overall = "low"

        return ReasoningResult(
            task_type=self.task_type,
            output=risks,
            structured_output={
                "risks": risks,
                "overall_risk_level": overall,
                "critical_count": critical_count,
                "high_count": high_count,
                "categories_found": list(set(r.get("category", "UNKNOWN") for r in risks))
            },
            confidence=0.8,
            requires_review=critical_count > 0
        )


class NextStepPlanningTask(ReasoningTask):
    """
    Generate next-best-action recommendations.

    Plans:
    - Immediate follow-ups
    - Strategic actions
    - Stakeholder engagement
    - Risk mitigation steps
    - Deal acceleration moves
    """

    @property
    def task_type(self) -> TaskType:
        return TaskType.NEXT_STEP_PLANNING

    @property
    def system_prompt(self) -> str:
        return """You are a sales execution advisor. Recommend next-best-actions
to advance deals and mitigate risks.

Consider:
- Deal stage and momentum
- Recent interactions and signals
- Stakeholder dynamics
- Identified risks and objections
- Time since last meaningful contact

For each recommendation:
1. Action description
2. Priority (immediate/this_week/this_month)
3. Target stakeholder(s)
4. Expected outcome
5. Success criteria

Provide 3-5 actionable recommendations, ranked by impact."""

    def format_input(self, context: ReasoningContext) -> str:
        prompt = f"""Recommend next steps based on this context:

RECENT INTERACTION:
{context.primary_content}

OPPORTUNITY:
Stage: {context.working_memory.get('stage', 'Unknown')}
Amount: {context.working_memory.get('amount', 'Unknown')}
Close Date: {context.working_memory.get('close_date', 'Unknown')}
Days in Stage: {context.working_memory.get('days_in_stage', 'Unknown')}

"""
        if context.working_memory.get("risks"):
            prompt += "IDENTIFIED RISKS:\n"
            for risk in context.working_memory["risks"]:
                prompt += f"- {risk}\n"

        if context.working_memory.get("objections"):
            prompt += "OPEN OBJECTIONS:\n"
            for objection in context.working_memory["objections"]:
                prompt += f"- {objection}\n"

        if context.retrieved_evidence:
            prompt += "RELEVANT PLAYBOOK GUIDANCE:\n"
            for evidence in context.retrieved_evidence[:2]:
                prompt += f"- {evidence.get('content', '')[:200]}\n"

        prompt += "\nProvide prioritized next-best-actions."
        return prompt

    def parse_output(self, raw_output: str, context: ReasoningContext) -> ReasoningResult:
        actions = []

        lines = raw_output.strip().split('\n')
        current_action = {}

        for line in lines:
            line = line.strip()
            if not line:
                if current_action.get("action"):
                    actions.append(current_action)
                    current_action = {}
                continue

            if ':' in line:
                key, value = line.split(':', 1)
                key = key.lower().strip()
                value = value.strip()

                if 'action' in key or 'description' in key:
                    current_action["action"] = value
                elif 'priority' in key:
                    current_action["priority"] = value.lower()
                elif 'target' in key or 'stakeholder' in key:
                    current_action["target"] = value
                elif 'outcome' in key or 'expected' in key:
                    current_action["expected_outcome"] = value
                elif 'success' in key or 'criteria' in key:
                    current_action["success_criteria"] = value

        if current_action.get("action"):
            actions.append(current_action)

        return ReasoningResult(
            task_type=self.task_type,
            output=actions,
            structured_output={
                "actions": actions,
                "total_count": len(actions),
                "immediate_actions": [
                    a for a in actions if a.get("priority") == "immediate"
                ]
            },
            confidence=0.75,
            requires_review=len(actions) == 0
        )


class ReasoningEngine:
    """
    Main reasoning engine that orchestrates LLM-based tasks.

    Provides:
    - Task execution with context
    - Confidence thresholding
    - Result caching
    - Error handling
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client
        self._tasks: dict[TaskType, ReasoningTask] = {
            TaskType.SUMMARIZATION: SummarizationTask(),
            TaskType.OBJECTION_EXTRACTION: ObjectionExtractionTask(),
            TaskType.STAKEHOLDER_MAPPING: StakeholderMappingTask(),
            TaskType.RISK_DETECTION: RiskDetectionTask(),
            TaskType.NEXT_STEP_PLANNING: NextStepPlanningTask()
        }

    def execute(self, task_type: TaskType, context: ReasoningContext) -> ReasoningResult:
        """Execute a reasoning task."""
        task = self._tasks.get(task_type)
        if not task:
            raise ValueError(f"Unknown task type: {task_type}")

        # Format input
        formatted_input = task.format_input(context)

        # Call LLM (or mock for simulation)
        if self._llm:
            raw_output = self._llm.generate(
                system_prompt=task.system_prompt,
                user_prompt=formatted_input
            )
        else:
            raw_output = self._mock_generate(task_type, context)

        # Parse output
        result = task.parse_output(raw_output, context)

        # Check if review required
        requires_review, reason = task.should_require_review(result)
        if requires_review:
            result.requires_review = True
            result.review_reason = reason

        return result

    def _mock_generate(self, task_type: TaskType, context: ReasoningContext) -> str:
        """Mock LLM generation for testing/simulation."""
        if task_type == TaskType.SUMMARIZATION:
            return f"""Executive Summary: The customer discussed their requirements
and expressed interest in moving forward.

Key Points:
- Budget was mentioned as a consideration
- Timeline is Q2 next year
- Technical requirements were reviewed

Next Steps:
- Schedule follow-up demo
- Provide pricing proposal"""

        elif task_type == TaskType.OBJECTION_EXTRACTION:
            return """Objection 1:
Quote: "The price seems higher than competitors"
Category: PRICE
Severity: medium
Suggested response: Emphasize ROI and total cost of ownership"""

        elif task_type == TaskType.RISK_DETECTION:
            return """Risk 1:
Description: Competitor mentioned in discussion
Category: COMPETITIVE
Severity: medium
Signals: Customer comparing features
Mitigation: Provide competitive battle card"""

        elif task_type == TaskType.NEXT_STEP_PLANNING:
            return """Action 1:
Action: Send follow-up email with pricing proposal
Priority: immediate
Target: Economic Buyer
Expected Outcome: Budget discussion scheduled
Success Criteria: Meeting confirmed within 1 week"""

        return "No analysis available"

    def analyze_interaction(self, context: ReasoningContext) -> dict:
        """Run full analysis pipeline on an interaction."""
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
