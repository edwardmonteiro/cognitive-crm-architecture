"""
Configuration Management

Centralized configuration for:
- LLM providers (OpenAI, Anthropic, Ollama)
- Vector stores (ChromaDB, FAISS)
- Memory backends (Redis, In-memory)
- Observability (LangSmith, OpenTelemetry)
"""

from .settings import (
    Settings,
    LLMConfig,
    VectorStoreConfig,
    MemoryConfig,
    get_settings
)
from .providers import (
    LLMProvider,
    get_llm,
    get_embeddings,
    get_chat_model
)

__all__ = [
    "Settings",
    "LLMConfig",
    "VectorStoreConfig",
    "MemoryConfig",
    "get_settings",
    "LLMProvider",
    "get_llm",
    "get_embeddings",
    "get_chat_model"
]
