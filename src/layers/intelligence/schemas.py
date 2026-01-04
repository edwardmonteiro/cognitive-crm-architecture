"""
Pydantic Schemas for Structured LLM Outputs

These schemas define the expected output structure for each reasoning task,
enabling type-safe, validated LLM responses using LangChain's structured output.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# Summarization Schemas
# =============================================================================

class InteractionSummary(BaseModel):
    """Structured output for interaction summarization."""
    executive_summary: str = Field(
        description="Brief 2-3 sentence summary of the interaction"
    )
    key_discussion_points: List[str] = Field(
        description="Main topics discussed during the interaction",
        default_factory=list
    )
    customer_sentiment: Literal["positive", "neutral", "negative", "mixed"] = Field(
        description="Overall sentiment of the customer during the interaction"
    )
    commitments_made: List[str] = Field(
        description="Any commitments or promises made by either party",
        default_factory=list
    )
    next_steps_mentioned: List[str] = Field(
        description="Next steps explicitly mentioned in the conversation",
        default_factory=list
    )
    follow_up_required: bool = Field(
        description="Whether follow-up action is needed",
        default=True
    )


# =============================================================================
# Objection Extraction Schemas
# =============================================================================

class ObjectionCategory(str, Enum):
    """Categories of sales objections."""
    PRICE = "PRICE"
    TIMING = "TIMING"
    AUTHORITY = "AUTHORITY"
    NEED = "NEED"
    TRUST = "TRUST"
    COMPETITION = "COMPETITION"
    TECHNICAL = "TECHNICAL"
    OTHER = "OTHER"


class Objection(BaseModel):
    """A single customer objection."""
    quote: str = Field(
        description="The exact or paraphrased statement from the customer"
    )
    category: ObjectionCategory = Field(
        description="Category of the objection"
    )
    severity: Literal["high", "medium", "low"] = Field(
        description="How critical this objection is to the deal"
    )
    is_addressed: bool = Field(
        description="Whether this objection was addressed in the conversation",
        default=False
    )
    suggested_response: Optional[str] = Field(
        description="Suggested approach to handle this objection",
        default=None
    )


class ObjectionExtractionResult(BaseModel):
    """Structured output for objection extraction."""
    objections: List[Objection] = Field(
        description="List of objections identified",
        default_factory=list
    )
    overall_resistance_level: Literal["high", "medium", "low", "none"] = Field(
        description="Overall level of customer resistance"
    )
    primary_concern: Optional[str] = Field(
        description="The main underlying concern driving objections",
        default=None
    )


# =============================================================================
# Stakeholder Mapping Schemas
# =============================================================================

class StakeholderRole(str, Enum):
    """Roles in the buying process."""
    DECISION_MAKER = "DECISION_MAKER"
    ECONOMIC_BUYER = "ECONOMIC_BUYER"
    TECHNICAL_BUYER = "TECHNICAL_BUYER"
    INFLUENCER = "INFLUENCER"
    CHAMPION = "CHAMPION"
    BLOCKER = "BLOCKER"
    END_USER = "END_USER"
    UNKNOWN = "UNKNOWN"


class Stakeholder(BaseModel):
    """A stakeholder in the buying process."""
    name: Optional[str] = Field(
        description="Name of the stakeholder if mentioned",
        default=None
    )
    title: Optional[str] = Field(
        description="Job title if mentioned",
        default=None
    )
    role: StakeholderRole = Field(
        description="Role in the buying process"
    )
    engagement_level: Literal["high", "medium", "low"] = Field(
        description="Level of engagement in the conversation"
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Sentiment toward the solution"
    )
    key_concerns: List[str] = Field(
        description="Primary concerns or priorities",
        default_factory=list
    )
    influence_score: float = Field(
        description="Estimated influence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        default=0.5
    )


class StakeholderMappingResult(BaseModel):
    """Structured output for stakeholder mapping."""
    stakeholders: List[Stakeholder] = Field(
        description="List of identified stakeholders",
        default_factory=list
    )
    buying_committee_complete: bool = Field(
        description="Whether the full buying committee is identified",
        default=False
    )
    missing_roles: List[StakeholderRole] = Field(
        description="Critical roles not yet identified",
        default_factory=list
    )
    power_dynamics_summary: Optional[str] = Field(
        description="Brief summary of relationship dynamics",
        default=None
    )


# =============================================================================
# Risk Detection Schemas
# =============================================================================

class RiskCategory(str, Enum):
    """Categories of deal risks."""
    COMPETITIVE = "COMPETITIVE"
    BUDGET = "BUDGET"
    TIMELINE = "TIMELINE"
    CHAMPION = "CHAMPION"
    ENGAGEMENT = "ENGAGEMENT"
    REQUIREMENTS = "REQUIREMENTS"
    DECISION_PROCESS = "DECISION_PROCESS"
    TECHNICAL = "TECHNICAL"
    ORGANIZATIONAL = "ORGANIZATIONAL"


class Risk(BaseModel):
    """A risk to deal progression."""
    description: str = Field(
        description="Description of the risk"
    )
    category: RiskCategory = Field(
        description="Category of the risk"
    )
    severity: Literal["critical", "high", "medium", "low"] = Field(
        description="Severity of the risk"
    )
    warning_signals: List[str] = Field(
        description="Observable signals indicating this risk",
        default_factory=list
    )
    mitigation_suggestion: Optional[str] = Field(
        description="Suggested action to mitigate the risk",
        default=None
    )
    probability: float = Field(
        description="Estimated probability of risk materializing (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        default=0.5
    )


class RiskDetectionResult(BaseModel):
    """Structured output for risk detection."""
    risks: List[Risk] = Field(
        description="List of identified risks",
        default_factory=list
    )
    overall_risk_level: Literal["critical", "high", "medium", "low"] = Field(
        description="Overall risk assessment for the deal"
    )
    immediate_actions_needed: List[str] = Field(
        description="Actions that should be taken immediately",
        default_factory=list
    )
    deal_health_score: float = Field(
        description="Overall deal health score (0.0 to 1.0, higher is better)",
        ge=0.0,
        le=1.0,
        default=0.5
    )


# =============================================================================
# Next-Best-Action Schemas
# =============================================================================

class ActionPriority(str, Enum):
    """Priority levels for recommended actions."""
    IMMEDIATE = "immediate"
    THIS_WEEK = "this_week"
    THIS_MONTH = "this_month"


class RecommendedAction(BaseModel):
    """A recommended next-best-action."""
    action: str = Field(
        description="Description of the recommended action"
    )
    action_type: str = Field(
        description="Type of action (follow_up, schedule_demo, send_proposal, etc.)"
    )
    priority: ActionPriority = Field(
        description="When this action should be taken"
    )
    target_stakeholder: Optional[str] = Field(
        description="Who this action is directed at",
        default=None
    )
    expected_outcome: str = Field(
        description="What this action is expected to achieve"
    )
    success_criteria: str = Field(
        description="How to measure if the action was successful"
    )
    confidence: float = Field(
        description="Confidence in this recommendation (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        default=0.7
    )
    rationale: str = Field(
        description="Reasoning behind this recommendation"
    )


class NextBestActionResult(BaseModel):
    """Structured output for next-best-action recommendations."""
    recommendations: List[RecommendedAction] = Field(
        description="List of recommended actions, ordered by priority",
        default_factory=list
    )
    strategic_focus: str = Field(
        description="Overall strategic focus for the next phase"
    )
    key_milestone: Optional[str] = Field(
        description="Next key milestone to aim for",
        default=None
    )


# =============================================================================
# Combined Analysis Schema
# =============================================================================

class FullInteractionAnalysis(BaseModel):
    """Complete analysis of an interaction combining all reasoning tasks."""
    summary: InteractionSummary
    objections: ObjectionExtractionResult
    stakeholders: StakeholderMappingResult
    risks: RiskDetectionResult
    next_actions: NextBestActionResult
    analysis_confidence: float = Field(
        description="Overall confidence in the analysis (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        default=0.8
    )
    requires_human_review: bool = Field(
        description="Whether this analysis should be reviewed by a human",
        default=False
    )
    review_reason: Optional[str] = Field(
        description="Reason why human review is recommended",
        default=None
    )
