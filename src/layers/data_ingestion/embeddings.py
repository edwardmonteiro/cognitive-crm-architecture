"""
Embedding Generation for Semantic Retrieval

From Section 4.1: "Representation: embeddings for semantic retrieval"

This module provides embedding generation for:
- RAG-based grounding of LLM outputs
- Semantic search over interaction history
- Similarity-based pattern matching

The embedding strategy supports the memory design from Section 4.2,
enabling efficient retrieval from episodic memory.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
import hashlib


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    max_tokens: int = 8191
    normalize: bool = True


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    id: UUID = field(default_factory=uuid4)
    text_hash: str = ""  # For deduplication
    embedding: list = field(default_factory=list)
    dimensions: int = 0
    model: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # Metadata for retrieval
    source_type: str = ""  # interaction, document, policy
    source_id: Optional[UUID] = None
    chunk_index: int = 0


class EmbeddingGenerator(ABC):
    """
    Abstract base class for embedding generators.

    Supports multiple embedding providers while maintaining
    a consistent interface for the intelligence layer.
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()

    @abstractmethod
    def generate(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def generate_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        pass

    def _compute_hash(self, text: str) -> str:
        """Compute hash for deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class MockEmbeddingGenerator(EmbeddingGenerator):
    """
    Mock embedding generator for testing and simulation.

    Generates deterministic pseudo-embeddings based on text content.
    This is useful for:
    - Unit testing
    - Simulation runs (Appendix B)
    - Development without API costs
    """

    def generate(self, text: str) -> EmbeddingResult:
        # Generate deterministic pseudo-embedding from text
        text_hash = self._compute_hash(text)

        # Create pseudo-embedding from hash (not semantically meaningful)
        pseudo_embedding = self._text_to_pseudo_embedding(text)

        return EmbeddingResult(
            text_hash=text_hash,
            embedding=pseudo_embedding,
            dimensions=self.config.dimensions,
            model="mock-embedding"
        )

    def generate_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.generate(text) for text in texts]

    def _text_to_pseudo_embedding(self, text: str) -> list[float]:
        """Generate pseudo-embedding for testing."""
        import random

        # Use text hash as seed for reproducibility
        seed = int(self._compute_hash(text), 16) % (2**32)
        random.seed(seed)

        # Generate normalized pseudo-embedding
        embedding = [random.gauss(0, 1) for _ in range(self.config.dimensions)]

        if self.config.normalize:
            magnitude = sum(x**2 for x in embedding) ** 0.5
            embedding = [x / magnitude for x in embedding]

        return embedding


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """
    OpenAI embedding generator.

    Uses the OpenAI embeddings API for production-quality embeddings.
    Supports text-embedding-3-small and text-embedding-3-large.
    """

    def __init__(self, config: EmbeddingConfig = None, api_key: str = None):
        super().__init__(config)
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client

    def generate(self, text: str) -> EmbeddingResult:
        client = self._get_client()

        response = client.embeddings.create(
            model=self.config.model_name,
            input=text,
            dimensions=self.config.dimensions
        )

        embedding = response.data[0].embedding

        return EmbeddingResult(
            text_hash=self._compute_hash(text),
            embedding=embedding,
            dimensions=len(embedding),
            model=self.config.model_name
        )

    def generate_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        client = self._get_client()

        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            response = client.embeddings.create(
                model=self.config.model_name,
                input=batch,
                dimensions=self.config.dimensions
            )

            for j, data in enumerate(response.data):
                results.append(EmbeddingResult(
                    text_hash=self._compute_hash(batch[j]),
                    embedding=data.embedding,
                    dimensions=len(data.embedding),
                    model=self.config.model_name
                ))

        return results


@dataclass
class TextChunker:
    """
    Chunking strategy for long texts.

    Long documents (transcripts, emails threads) need to be chunked
    for effective embedding and retrieval.
    """
    chunk_size: int = 500
    chunk_overlap: int = 50
    separator: str = "\n"

    def chunk(self, text: str) -> list[dict]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [{"text": text, "index": 0, "start": 0, "end": len(text)}]

        chunks = []
        sentences = text.split(self.separator)
        current_chunk = []
        current_length = 0
        start_pos = 0

        for sentence in sentences:
            sentence_len = len(sentence) + len(self.separator)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = self.separator.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "index": len(chunks),
                    "start": start_pos,
                    "end": start_pos + len(chunk_text)
                })

                # Keep overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s) + len(self.separator)
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length
                start_pos = start_pos + len(chunk_text) - overlap_length

            current_chunk.append(sentence)
            current_length += sentence_len

        # Add final chunk
        if current_chunk:
            chunk_text = self.separator.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "index": len(chunks),
                "start": start_pos,
                "end": start_pos + len(chunk_text)
            })

        return chunks


class EmbeddingIndex:
    """
    In-memory embedding index for semantic search.

    Provides:
    - Fast similarity search
    - Filtering by metadata
    - Persistence support

    In production, this would be replaced with a vector database
    (Pinecone, Weaviate, Chroma, etc.)
    """

    def __init__(self):
        self._embeddings: dict[str, EmbeddingResult] = {}
        self._metadata: dict[str, dict] = {}

    def add(self, result: EmbeddingResult, metadata: dict = None) -> None:
        """Add embedding to index."""
        key = str(result.id)
        self._embeddings[key] = result
        self._metadata[key] = metadata or {}

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_fn: callable = None
    ) -> list[tuple[EmbeddingResult, float, dict]]:
        """
        Search for similar embeddings.

        Returns list of (result, similarity_score, metadata) tuples.
        """
        scores = []

        for key, result in self._embeddings.items():
            if filter_fn and not filter_fn(self._metadata.get(key, {})):
                continue

            similarity = self._cosine_similarity(query_embedding, result.embedding)
            scores.append((result, similarity, self._metadata.get(key, {})))

        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            raise ValueError("Vectors must have same dimensions")

        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x**2 for x in a) ** 0.5
        magnitude_b = sum(x**2 for x in b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def __len__(self) -> int:
        return len(self._embeddings)
