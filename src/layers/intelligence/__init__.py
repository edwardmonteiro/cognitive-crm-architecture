"""
Layer 2: Intelligence Layer - Transformers and Grounding

From Section 4.2 of the paper:

LLM Reasoning:
- Summarisation
- Objection extraction
- Stakeholder mapping
- Risk detection
- Next-step planning

Grounding via RAG:
- Retrieve governed evidence
- Reduce hallucination risk

Memory Design:
- Short-term working memory per opportunity
- Long-term episodic memory across interactions
- Policy memory for constraints
"""

from .reasoning import (
    ReasoningEngine,
    SummarizationTask,
    ObjectionExtractionTask,
    StakeholderMappingTask,
    RiskDetectionTask,
    NextStepPlanningTask
)
from .rag import RAGEngine, RetrievalResult
from .memory import (
    MemoryManager,
    WorkingMemory,
    EpisodicMemory,
    PolicyMemory
)

__all__ = [
    "ReasoningEngine",
    "SummarizationTask",
    "ObjectionExtractionTask",
    "StakeholderMappingTask",
    "RiskDetectionTask",
    "NextStepPlanningTask",
    "RAGEngine",
    "RetrievalResult",
    "MemoryManager",
    "WorkingMemory",
    "EpisodicMemory",
    "PolicyMemory"
]
