"""
Layer 4: Closed-Loop Learning Signals

From Section 4.4 of the paper:

Outcome Signals:
- Stage changes
- Replies
- Meetings booked
- Win/loss
- Slippage

Human Feedback:
- Overrides
- Edits
- Rejection reasons
- Used for policy tuning and prompt/tool refinement

Safety:
- Model updates gated
- Learning may initially be limited to retrieval ranking and playbook policies
"""

from .signals import (
    OutcomeSignal,
    OutcomeType,
    SignalCollector,
    OutcomeTracker
)
from .feedback import (
    FeedbackEntry,
    FeedbackType,
    FeedbackCollector,
    FeedbackAnalyzer
)
from .learning import (
    LearningEngine,
    PolicyLearner,
    RetrievalLearner,
    PlaybookOptimizer
)

__all__ = [
    "OutcomeSignal",
    "OutcomeType",
    "SignalCollector",
    "OutcomeTracker",
    "FeedbackEntry",
    "FeedbackType",
    "FeedbackCollector",
    "FeedbackAnalyzer",
    "LearningEngine",
    "PolicyLearner",
    "RetrievalLearner",
    "PlaybookOptimizer"
]
