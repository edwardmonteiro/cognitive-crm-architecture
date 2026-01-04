"""
LLM and Embedding Provider Factory

Provides unified interface for:
- Multiple LLM providers (OpenAI, Anthropic, Ollama)
- Embedding models
- Chat models with structured outputs
"""

from typing import Optional, Type, Any
from enum import Enum

from pydantic import BaseModel

from .settings import (
    Settings,
    LLMConfig,
    EmbeddingConfig,
    LLMProviderType,
    EmbeddingProviderType,
    get_settings
)


class LLMProvider:
    """
    Factory for LLM providers using LangChain.

    Supports:
    - OpenAI (GPT-4, GPT-4o, GPT-3.5)
    - Anthropic (Claude 3.5, Claude 3)
    - Ollama (Llama, Mistral, etc.)
    - Azure OpenAI
    """

    def __init__(self, config: LLMConfig = None):
        self.config = config or get_settings().llm
        self._llm = None
        self._chat_model = None

    def get_llm(self):
        """Get LLM instance (lazy initialization)."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def get_chat_model(self):
        """Get chat model instance."""
        if self._chat_model is None:
            self._chat_model = self._create_chat_model()
        return self._chat_model

    def _create_llm(self):
        """Create LLM based on provider configuration."""
        provider = self.config.provider

        if provider == LLMProviderType.OPENAI:
            return self._create_openai_llm()
        elif provider == LLMProviderType.ANTHROPIC:
            return self._create_anthropic_llm()
        elif provider == LLMProviderType.OLLAMA:
            return self._create_ollama_llm()
        elif provider == LLMProviderType.AZURE_OPENAI:
            return self._create_azure_openai_llm()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _create_chat_model(self):
        """Create chat model based on provider configuration."""
        provider = self.config.provider

        if provider == LLMProviderType.OPENAI:
            return self._create_openai_chat()
        elif provider == LLMProviderType.ANTHROPIC:
            return self._create_anthropic_chat()
        elif provider == LLMProviderType.OLLAMA:
            return self._create_ollama_chat()
        elif provider == LLMProviderType.AZURE_OPENAI:
            return self._create_azure_openai_chat()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _create_openai_llm(self):
        """Create OpenAI LLM."""
        try:
            from langchain_openai import OpenAI

            api_key = self.config.openai_api_key
            if api_key:
                api_key = api_key.get_secret_value()

            return OpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=api_key,
                timeout=self.config.timeout
            )
        except ImportError:
            raise ImportError("Install langchain-openai: pip install langchain-openai")

    def _create_openai_chat(self):
        """Create OpenAI Chat model."""
        try:
            from langchain_openai import ChatOpenAI

            api_key = self.config.openai_api_key
            if api_key:
                api_key = api_key.get_secret_value()

            return ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=api_key,
                timeout=self.config.timeout
            )
        except ImportError:
            raise ImportError("Install langchain-openai: pip install langchain-openai")

    def _create_anthropic_llm(self):
        """Create Anthropic LLM."""
        try:
            from langchain_anthropic import ChatAnthropic

            api_key = self.config.anthropic_api_key
            if api_key:
                api_key = api_key.get_secret_value()

            return ChatAnthropic(
                model=self.config.model_name or "claude-3-5-sonnet-20241022",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=api_key,
                timeout=self.config.timeout
            )
        except ImportError:
            raise ImportError("Install langchain-anthropic: pip install langchain-anthropic")

    def _create_anthropic_chat(self):
        """Create Anthropic Chat model (same as LLM for Anthropic)."""
        return self._create_anthropic_llm()

    def _create_ollama_llm(self):
        """Create Ollama LLM for local models."""
        try:
            from langchain_community.llms import Ollama

            return Ollama(
                model=self.config.model_name or "llama3.2",
                base_url=self.config.ollama_base_url,
                temperature=self.config.temperature
            )
        except ImportError:
            raise ImportError("Install langchain-community: pip install langchain-community")

    def _create_ollama_chat(self):
        """Create Ollama Chat model."""
        try:
            from langchain_community.chat_models import ChatOllama

            return ChatOllama(
                model=self.config.model_name or "llama3.2",
                base_url=self.config.ollama_base_url,
                temperature=self.config.temperature
            )
        except ImportError:
            raise ImportError("Install langchain-community: pip install langchain-community")

    def _create_azure_openai_llm(self):
        """Create Azure OpenAI LLM."""
        try:
            from langchain_openai import AzureOpenAI

            api_key = self.config.openai_api_key
            if api_key:
                api_key = api_key.get_secret_value()

            return AzureOpenAI(
                azure_endpoint=self.config.azure_endpoint,
                azure_deployment=self.config.azure_deployment_name,
                api_version=self.config.azure_api_version,
                api_key=api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        except ImportError:
            raise ImportError("Install langchain-openai: pip install langchain-openai")

    def _create_azure_openai_chat(self):
        """Create Azure OpenAI Chat model."""
        try:
            from langchain_openai import AzureChatOpenAI

            api_key = self.config.openai_api_key
            if api_key:
                api_key = api_key.get_secret_value()

            return AzureChatOpenAI(
                azure_endpoint=self.config.azure_endpoint,
                azure_deployment=self.config.azure_deployment_name,
                api_version=self.config.azure_api_version,
                api_key=api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        except ImportError:
            raise ImportError("Install langchain-openai: pip install langchain-openai")

    def with_structured_output(self, schema: Type[BaseModel]):
        """
        Get chat model with structured output support.

        Uses LangChain's with_structured_output for Pydantic model outputs.
        """
        chat = self.get_chat_model()

        # Check if model supports structured output
        if hasattr(chat, 'with_structured_output'):
            return chat.with_structured_output(schema)
        else:
            # Fallback for models that don't support it natively
            from langchain_core.output_parsers import PydanticOutputParser
            parser = PydanticOutputParser(pydantic_object=schema)
            return chat | parser


class EmbeddingProvider:
    """
    Factory for embedding models.

    Supports:
    - OpenAI embeddings
    - Sentence Transformers (local)
    - HuggingFace embeddings
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or get_settings().embedding
        self._embeddings = None

    def get_embeddings(self):
        """Get embedding model instance."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings

    def _create_embeddings(self):
        """Create embedding model based on configuration."""
        provider = self.config.provider

        if provider == EmbeddingProviderType.OPENAI:
            return self._create_openai_embeddings()
        elif provider == EmbeddingProviderType.SENTENCE_TRANSFORMERS:
            return self._create_sentence_transformer_embeddings()
        elif provider == EmbeddingProviderType.HUGGINGFACE:
            return self._create_huggingface_embeddings()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _create_openai_embeddings(self):
        """Create OpenAI embeddings."""
        try:
            from langchain_openai import OpenAIEmbeddings

            settings = get_settings()
            api_key = settings.llm.openai_api_key
            if api_key:
                api_key = api_key.get_secret_value()

            return OpenAIEmbeddings(
                model=self.config.model_name,
                dimensions=self.config.dimensions,
                api_key=api_key
            )
        except ImportError:
            raise ImportError("Install langchain-openai: pip install langchain-openai")

    def _create_sentence_transformer_embeddings(self):
        """Create Sentence Transformer embeddings (local)."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=self.config.sentence_transformer_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except ImportError:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            )

    def _create_huggingface_embeddings(self):
        """Create HuggingFace embeddings."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=self.config.sentence_transformer_model
            )
        except ImportError:
            raise ImportError(
                "Install langchain-community: pip install langchain-community"
            )


# Convenience functions
def get_llm(config: LLMConfig = None):
    """Get LLM instance."""
    return LLMProvider(config).get_llm()


def get_chat_model(config: LLMConfig = None):
    """Get chat model instance."""
    return LLMProvider(config).get_chat_model()


def get_embeddings(config: EmbeddingConfig = None):
    """Get embedding model instance."""
    return EmbeddingProvider(config).get_embeddings()
