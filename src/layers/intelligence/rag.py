"""
Retrieval-Augmented Generation (RAG) Engine

From Section 4.2: "Grounding via RAG: retrieve governed evidence
(past interactions, product docs, policy) to reduce hallucination risk."

This module implements RAG-based grounding for LLM outputs:
- Semantic retrieval from knowledge bases
- Context ranking and selection
- Evidence attribution
- Hallucination risk reduction

Key governed sources:
- Past interactions (episodic memory)
- Product documentation
- Sales playbooks and policies
- Competitive intelligence
- Case studies and references
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from ..data_ingestion.embeddings import EmbeddingGenerator, EmbeddingIndex, EmbeddingResult


class SourceType(Enum):
    """Types of governed knowledge sources."""
    INTERACTION_HISTORY = "interaction_history"
    PRODUCT_DOCS = "product_docs"
    SALES_PLAYBOOK = "sales_playbook"
    POLICY = "policy"
    COMPETITIVE_INTEL = "competitive_intel"
    CASE_STUDY = "case_study"
    FAQ = "faq"
    CONTRACT_TEMPLATE = "contract_template"


@dataclass
class Document:
    """A document in the knowledge base."""
    id: UUID = field(default_factory=uuid4)
    source_type: SourceType = SourceType.PRODUCT_DOCS
    title: str = ""
    content: str = ""

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    author: Optional[str] = None
    tags: list = field(default_factory=list)

    # Access control
    access_level: str = "public"  # public, internal, restricted
    allowed_roles: list = field(default_factory=list)

    # Chunking
    chunks: list = field(default_factory=list)
    embeddings: list = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    id: UUID = field(default_factory=uuid4)
    query: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Retrieved documents
    documents: list = field(default_factory=list)
    chunks: list = field(default_factory=list)

    # Scores and ranking
    relevance_scores: list = field(default_factory=list)

    # Filtering applied
    source_filter: Optional[list] = None
    date_filter: Optional[dict] = None

    # For audit
    total_searched: int = 0
    retrieval_time_ms: float = 0.0


@dataclass
class RetrievedContext:
    """A piece of retrieved context for grounding."""
    document_id: UUID = field(default_factory=uuid4)
    source_type: SourceType = SourceType.PRODUCT_DOCS
    content: str = ""
    relevance_score: float = 0.0

    # Location within document
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0

    # For citation
    title: str = ""
    url: Optional[str] = None

    def to_citation(self) -> str:
        """Format as citation for LLM context."""
        return f"[Source: {self.title} ({self.source_type.value})]"


class RAGEngine:
    """
    RAG engine for grounding LLM outputs in governed sources.

    Design principles:
    - Retrieve before generate
    - Attribute all claims to sources
    - Prioritize recency for interactions
    - Respect access control
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator = None,
        default_top_k: int = 5,
        relevance_threshold: float = 0.7
    ):
        self._embedding_generator = embedding_generator
        self._default_top_k = default_top_k
        self._relevance_threshold = relevance_threshold

        # Knowledge bases by source type
        self._indices: dict[SourceType, EmbeddingIndex] = {}
        self._documents: dict[str, Document] = {}

    def add_document(self, document: Document) -> None:
        """Index a document for retrieval."""
        source_type = document.source_type

        if source_type not in self._indices:
            self._indices[source_type] = EmbeddingIndex()

        # Store document
        self._documents[str(document.id)] = document

        # Generate embeddings for chunks (or full content)
        if document.chunks:
            for i, chunk in enumerate(document.chunks):
                if self._embedding_generator:
                    embedding = self._embedding_generator.generate(chunk["text"])
                    self._indices[source_type].add(
                        embedding,
                        metadata={
                            "document_id": str(document.id),
                            "chunk_index": i,
                            "source_type": source_type.value,
                            "title": document.title,
                            "content": chunk["text"]
                        }
                    )
        else:
            # Embed full content
            if self._embedding_generator:
                embedding = self._embedding_generator.generate(document.content)
                self._indices[source_type].add(
                    embedding,
                    metadata={
                        "document_id": str(document.id),
                        "chunk_index": 0,
                        "source_type": source_type.value,
                        "title": document.title,
                        "content": document.content
                    }
                )

    def retrieve(
        self,
        query: str,
        source_types: list[SourceType] = None,
        top_k: int = None,
        min_relevance: float = None,
        date_range: dict = None,
        access_context: dict = None
    ) -> RetrievalResult:
        """
        Retrieve relevant context for a query.

        Args:
            query: The search query
            source_types: Filter by source types (default: all)
            top_k: Number of results to return
            min_relevance: Minimum relevance score
            date_range: Filter by date {"start": datetime, "end": datetime}
            access_context: User access context for filtering
        """
        start_time = datetime.now()

        top_k = top_k or self._default_top_k
        min_relevance = min_relevance if min_relevance is not None else self._relevance_threshold
        source_types = source_types or list(SourceType)

        # Generate query embedding
        if self._embedding_generator:
            query_embedding = self._embedding_generator.generate(query)
            query_vector = query_embedding.embedding
        else:
            # Mock embedding for testing
            query_vector = [0.0] * 1536

        # Search across requested source types
        all_results = []
        total_searched = 0

        for source_type in source_types:
            index = self._indices.get(source_type)
            if not index:
                continue

            total_searched += len(index)

            # Define filter function
            def create_filter(date_range, access_context):
                def filter_fn(metadata):
                    # Date filter
                    if date_range:
                        doc = self._documents.get(metadata.get("document_id"))
                        if doc:
                            if date_range.get("start") and doc.created_at < date_range["start"]:
                                return False
                            if date_range.get("end") and doc.created_at > date_range["end"]:
                                return False

                    # Access filter
                    if access_context:
                        doc = self._documents.get(metadata.get("document_id"))
                        if doc and doc.access_level == "restricted":
                            user_roles = access_context.get("roles", [])
                            if not any(r in doc.allowed_roles for r in user_roles):
                                return False

                    return True
                return filter_fn

            filter_fn = create_filter(date_range, access_context)

            # Search index
            results = index.search(
                query_vector,
                top_k=top_k,
                filter_fn=filter_fn
            )

            for embedding_result, score, metadata in results:
                all_results.append({
                    "embedding": embedding_result,
                    "score": score,
                    "metadata": metadata,
                    "source_type": source_type
                })

        # Sort by relevance and filter
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = [r for r in all_results if r["score"] >= min_relevance]
        all_results = all_results[:top_k]

        # Build retrieved contexts
        contexts = []
        for result in all_results:
            metadata = result["metadata"]
            doc_id = metadata.get("document_id")
            doc = self._documents.get(doc_id)

            contexts.append(RetrievedContext(
                document_id=UUID(doc_id) if doc_id else uuid4(),
                source_type=result["source_type"],
                content=metadata.get("content", ""),
                relevance_score=result["score"],
                chunk_index=metadata.get("chunk_index", 0),
                title=metadata.get("title", ""),
                url=doc.tags[0] if doc and doc.tags else None
            ))

        end_time = datetime.now()
        retrieval_time = (end_time - start_time).total_seconds() * 1000

        return RetrievalResult(
            query=query,
            chunks=contexts,
            relevance_scores=[c.relevance_score for c in contexts],
            source_filter=source_types,
            date_filter=date_range,
            total_searched=total_searched,
            retrieval_time_ms=retrieval_time
        )

    def format_context_for_llm(
        self,
        retrieval_result: RetrievalResult,
        max_tokens: int = 3000
    ) -> str:
        """
        Format retrieved context for LLM consumption.

        Includes:
        - Relevance-ranked content
        - Source citations
        - Token budget management
        """
        context_parts = []
        estimated_tokens = 0

        for ctx in retrieval_result.chunks:
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            content_tokens = len(ctx.content) // 4
            citation_tokens = len(ctx.to_citation()) // 4

            if estimated_tokens + content_tokens + citation_tokens > max_tokens:
                # Truncate content to fit
                available_chars = (max_tokens - estimated_tokens) * 4 - len(ctx.to_citation())
                if available_chars > 100:
                    truncated = ctx.content[:available_chars] + "..."
                    context_parts.append(f"{ctx.to_citation()}\n{truncated}")
                break

            context_parts.append(f"{ctx.to_citation()}\n{ctx.content}")
            estimated_tokens += content_tokens + citation_tokens

        return "\n\n---\n\n".join(context_parts)

    def get_evidence_for_claim(
        self,
        claim: str,
        top_k: int = 3
    ) -> list[RetrievedContext]:
        """
        Find evidence to support or refute a claim.

        Used for fact-checking LLM outputs.
        """
        result = self.retrieve(claim, top_k=top_k)
        return result.chunks


class KnowledgeBaseBuilder:
    """
    Helper for building and populating knowledge bases.

    Supports:
    - Bulk document loading
    - Chunking strategies
    - Metadata extraction
    """

    def __init__(self, rag_engine: RAGEngine):
        self._engine = rag_engine

    def add_playbook(
        self,
        title: str,
        content: str,
        stage: str = None,
        tags: list = None
    ) -> Document:
        """Add a sales playbook document."""
        doc = Document(
            source_type=SourceType.SALES_PLAYBOOK,
            title=title,
            content=content,
            tags=tags or []
        )
        if stage:
            doc.tags.append(f"stage:{stage}")

        self._engine.add_document(doc)
        return doc

    def add_policy(
        self,
        title: str,
        content: str,
        policy_type: str = "general",
        access_level: str = "internal"
    ) -> Document:
        """Add a policy document."""
        doc = Document(
            source_type=SourceType.POLICY,
            title=title,
            content=content,
            access_level=access_level,
            tags=[f"policy_type:{policy_type}"]
        )

        self._engine.add_document(doc)
        return doc

    def add_product_doc(
        self,
        title: str,
        content: str,
        product: str = None,
        category: str = None
    ) -> Document:
        """Add product documentation."""
        doc = Document(
            source_type=SourceType.PRODUCT_DOCS,
            title=title,
            content=content,
            tags=[]
        )
        if product:
            doc.tags.append(f"product:{product}")
        if category:
            doc.tags.append(f"category:{category}")

        self._engine.add_document(doc)
        return doc

    def add_competitive_intel(
        self,
        competitor: str,
        title: str,
        content: str
    ) -> Document:
        """Add competitive intelligence."""
        doc = Document(
            source_type=SourceType.COMPETITIVE_INTEL,
            title=title,
            content=content,
            tags=[f"competitor:{competitor}"]
        )

        self._engine.add_document(doc)
        return doc

    def index_interaction(
        self,
        interaction_id: UUID,
        opportunity_id: UUID,
        summary: str,
        key_points: list,
        timestamp: datetime
    ) -> Document:
        """Index an interaction for retrieval."""
        content = f"Summary: {summary}\n\nKey Points:\n"
        for point in key_points:
            content += f"- {point}\n"

        doc = Document(
            id=interaction_id,
            source_type=SourceType.INTERACTION_HISTORY,
            title=f"Interaction {timestamp.strftime('%Y-%m-%d')}",
            content=content,
            created_at=timestamp,
            tags=[f"opportunity:{opportunity_id}"]
        )

        self._engine.add_document(doc)
        return doc
