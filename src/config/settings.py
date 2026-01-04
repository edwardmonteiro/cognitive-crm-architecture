"""
Settings Management with Pydantic

Provides type-safe configuration management with:
- Environment variable support
- Validation
- Multiple provider configurations
"""

from enum import Enum
from typing import Optional, Dict, Any
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderType(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"


class VectorStoreType(str, Enum):
    """Supported vector store backends."""
    CHROMA = "chroma"
    FAISS = "faiss"
    IN_MEMORY = "in_memory"


class MemoryBackendType(str, Enum):
    """Supported memory backends."""
    REDIS = "redis"
    IN_MEMORY = "in_memory"


class EmbeddingProviderType(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE = "huggingface"


class LLMConfig(BaseSettings):
    """LLM provider configuration."""
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        extra="ignore"
    )

    provider: LLMProviderType = LLMProviderType.OPENAI
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 60

    # API Keys (loaded from environment)
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[SecretStr] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"

    # Azure OpenAI settings
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2024-02-15-preview"
    azure_deployment_name: Optional[str] = None


class EmbeddingConfig(BaseSettings):
    """Embedding configuration."""
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        extra="ignore"
    )

    provider: EmbeddingProviderType = EmbeddingProviderType.OPENAI
    model_name: str = "text-embedding-3-small"
    dimensions: int = 1536

    # For sentence-transformers
    sentence_transformer_model: str = "all-MiniLM-L6-v2"


class VectorStoreConfig(BaseSettings):
    """Vector store configuration."""
    model_config = SettingsConfigDict(
        env_prefix="VECTORSTORE_",
        extra="ignore"
    )

    backend: VectorStoreType = VectorStoreType.CHROMA
    collection_name: str = "cognitive_crm"

    # ChromaDB settings
    chroma_persist_directory: str = "./data/chroma"
    chroma_host: Optional[str] = None
    chroma_port: int = 8000

    # FAISS settings
    faiss_index_path: str = "./data/faiss"


class MemoryConfig(BaseSettings):
    """Memory backend configuration."""
    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        extra="ignore"
    )

    backend: MemoryBackendType = MemoryBackendType.IN_MEMORY

    # Redis settings
    redis_url: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 86400  # 24 hours

    # Working memory settings
    working_memory_capacity: int = 100
    working_memory_ttl_hours: int = 168  # 1 week

    # Episodic memory settings
    episodic_memory_max_entries: int = 10000


class ObservabilityConfig(BaseSettings):
    """Observability configuration."""
    model_config = SettingsConfigDict(
        env_prefix="OBSERVABILITY_",
        extra="ignore"
    )

    enabled: bool = False

    # LangSmith
    langsmith_enabled: bool = False
    langsmith_api_key: Optional[SecretStr] = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = "cognitive-crm"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # OpenTelemetry
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "cognitive-crm"


class Settings(BaseSettings):
    """Main application settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application
    app_name: str = "Cognitive CRM"
    debug: bool = False
    log_level: str = "INFO"

    # Sub-configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Cognitive CRM specific
    approval_confidence_threshold: float = 0.8
    nba_max_recommendations: int = 5
    risk_alert_threshold: float = 0.6

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            llm=LLMConfig(),
            embedding=EmbeddingConfig(),
            vectorstore=VectorStoreConfig(),
            memory=MemoryConfig(),
            observability=ObservabilityConfig()
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_env()
