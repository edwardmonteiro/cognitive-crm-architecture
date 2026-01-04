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

LangChain Integration:
- Structured outputs with Pydantic
- ChromaDB/FAISS for vector search
- Redis for persistent working memory
- LangGraph for agent workflows
"""

# Original implementations (baseline)
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

# LangChain-based implementations
from .schemas import (
    InteractionSummary,
    Objection,
    ObjectionExtractionResult,
    Stakeholder,
    StakeholderMap,
    Risk,
    RiskAssessment,
    NextStep,
    NextStepPlan,
    NBARecommendation,
    NBARecommendationSet
)
from .reasoning_langchain import LangChainReasoningEngine
from .memory_langchain import (
    MemoryManagerLangChain,
    WorkingMemoryLangChain,
    EpisodicMemoryLangChain,
    PolicyMemoryLangChain,
    MemoryEntry,
    create_memory_manager
)
from .rag_langchain import (
    RAGEngineLangChain,
    KnowledgeBaseBuilderLangChain,
    RetrievedDocument,
    create_rag_engine,
    SourceType
)

__all__ = [
    # Original implementations
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
    "PolicyMemory",
    # Pydantic schemas
    "InteractionSummary",
    "Objection",
    "ObjectionExtractionResult",
    "Stakeholder",
    "StakeholderMap",
    "Risk",
    "RiskAssessment",
    "NextStep",
    "NextStepPlan",
    "NBARecommendation",
    "NBARecommendationSet",
    # LangChain implementations
    "LangChainReasoningEngine",
    "MemoryManagerLangChain",
    "WorkingMemoryLangChain",
    "EpisodicMemoryLangChain",
    "PolicyMemoryLangChain",
    "MemoryEntry",
    "create_memory_manager",
    # RAG LangChain
    "RAGEngineLangChain",
    "KnowledgeBaseBuilderLangChain",
    "RetrievedDocument",
    "create_rag_engine",
    "SourceType"
]
