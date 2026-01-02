"""
Use Case Implementations

From Section 5.1 of the paper:

Use Case 1: Post-call intelligence and CRM update (human-reviewed)
Use Case 2: Evidence-based next-best-action (NBA) recommendations with justification

These use cases demonstrate the end-to-end flow of the cognitive CRM architecture,
from signal ingestion through reasoning, orchestration, and learning.
"""

from .post_call_intelligence import PostCallIntelligenceUseCase
from .nba_recommendations import NBARecommendationsUseCase

__all__ = [
    "PostCallIntelligenceUseCase",
    "NBARecommendationsUseCase"
]
