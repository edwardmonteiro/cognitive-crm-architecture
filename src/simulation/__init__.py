"""
Simulation Framework for Feasibility Demonstrations

From Appendix B of the paper:

Purpose: The simulation evaluates whether the architecture's mechanisms
produce internally coherent, directionally plausible effects across
coupled execution metrics. It does not estimate real-world outcomes.

B.1 Entities and Timeframe:
- Entities: sellers, opportunities, interactions, recommended actions (NBA), outcomes
- Timeframe: eight-week simulated window with seller-level aggregation

B.2 Baseline Parameterisation:
- Baseline cycle time mean ≈ 94 days (control)
- Baseline conversion ≈ 31.5% (control)
- Baseline forecast MAE ≈ 18% (control)
- Admin time ≈ 14.8 hrs/week per seller (control)

B.3 Mechanism Assumptions:
- Post-call automation reduces documentation time
- RAG-grounded NBA reduces decision latency
- Human approval gates introduce bounded delay
- Override signals tune playbooks and retrieval ranking
"""

from .entities import (
    SimulatedSeller,
    SimulatedOpportunity,
    SimulatedInteraction,
    SimulatedNBA
)
from .simulator import (
    Simulator,
    SimulationConfig,
    SimulationResult
)
from .metrics import (
    MetricsCalculator,
    ComparisonResult
)

__all__ = [
    "SimulatedSeller",
    "SimulatedOpportunity",
    "SimulatedInteraction",
    "SimulatedNBA",
    "Simulator",
    "SimulationConfig",
    "SimulationResult",
    "MetricsCalculator",
    "ComparisonResult"
]
