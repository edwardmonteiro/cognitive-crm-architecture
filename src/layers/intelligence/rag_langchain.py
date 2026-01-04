"""
RAG Engine with LangChain Integration

From Section 4.2: "Grounding via RAG: retrieve governed evidence
(past interactions, product docs, policy) to reduce hallucination risk."

This module implements RAG using LangChain components:
- ChromaDB/FAISS vector stores
- LangChain retrievers with reranking
- Document loaders and text splitters
- Multi-query retrieval for better recall
- Contextual compression

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
from typing import Any, Dict, List, Optional, Callable
from uuid import UUID, uuid4
import hashlib

from pydantic import BaseModel, Field


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


class RetrievedDocument(BaseModel):
    """A retrieved document with metadata."""
    id: str
    content: str
    source_type: str
    title: str
    relevance_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_citation(self) -> str:
        """Format as citation for LLM context."""
        return f"[Source: {self.title} ({self.source_type})]"


class RetrievalResult(BaseModel):
    """Result of a retrieval operation."""
    query: str
    documents: List[RetrievedDocument]
    total_searched: int = 0
    retrieval_time_ms: float = 0.0


class RAGEngineLangChain:
    """
    RAG engine using LangChain for retrieval and grounding.

    Features:
    - Vector store backends (ChromaDB, FAISS)
    - Multi-query retrieval for better recall
    - Contextual compression for relevant excerpts
    - Source attribution and citation
    - Access control support
    """

    def __init__(
        self,
        collection_name: str = "crm_knowledge_base",
        persist_directory: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        backend: str = "chroma",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        default_top_k: int = 5,
        relevance_threshold: float = 0.7
    ):
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._embedding_model = embedding_model
        self._backend = backend
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._default_top_k = default_top_k
        self._relevance_threshold = relevance_threshold

        self._vectorstore = None
        self._embeddings = None
        self._text_splitter = None
        self._retriever = None

        # Document registry for metadata
        self._documents: Dict[str, Dict[str, Any]] = {}

        self._initialize()

    def _initialize(self):
        """Initialize LangChain components."""
        # Initialize embeddings
        try:
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(model=self._embedding_model)
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            except ImportError:
                self._embeddings = None

        # Initialize text splitter
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        except ImportError:
            self._text_splitter = None

        # Initialize vector store
        if self._embeddings:
            self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """Initialize the vector store backend."""
        try:
            if self._backend == "chroma":
                from langchain_community.vectorstores import Chroma

                if self._persist_directory:
                    self._vectorstore = Chroma(
                        collection_name=self._collection_name,
                        embedding_function=self._embeddings,
                        persist_directory=self._persist_directory
                    )
                else:
                    self._vectorstore = Chroma(
                        collection_name=self._collection_name,
                        embedding_function=self._embeddings
                    )
            elif self._backend == "faiss":
                # FAISS requires documents to initialize
                self._vectorstore = None
        except ImportError:
            self._vectorstore = None

    @property
    def is_available(self) -> bool:
        """Check if RAG engine is available."""
        return self._embeddings is not None

    def add_document(
        self,
        content: str,
        source_type: SourceType,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the knowledge base.

        Args:
            content: Document content
            source_type: Type of source
            title: Document title
            metadata: Additional metadata
            document_id: Optional document ID

        Returns:
            Document ID
        """
        doc_id = document_id or str(uuid4())
        metadata = metadata or {}

        # Store document metadata
        self._documents[doc_id] = {
            "title": title,
            "source_type": source_type.value,
            "created_at": datetime.now().isoformat(),
            **metadata
        }

        if not self._embeddings:
            return doc_id

        try:
            from langchain_core.documents import Document

            # Split content into chunks
            if self._text_splitter:
                chunks = self._text_splitter.split_text(content)
            else:
                chunks = [content]

            # Create documents with metadata
            docs = []
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source_type": source_type.value,
                        "title": title,
                        **metadata
                    }
                )
                docs.append(doc)
                chunk_ids.append(chunk_id)

            # Add to vector store
            if self._backend == "chroma" and self._vectorstore:
                self._vectorstore.add_documents(docs, ids=chunk_ids)
            elif self._backend == "faiss":
                from langchain_community.vectorstores import FAISS
                if self._vectorstore is None:
                    self._vectorstore = FAISS.from_documents(docs, self._embeddings)
                else:
                    self._vectorstore.add_documents(docs, ids=chunk_ids)

        except Exception as e:
            print(f"Error adding document: {e}")

        return doc_id

    def retrieve(
        self,
        query: str,
        source_types: Optional[List[SourceType]] = None,
        top_k: Optional[int] = None,
        min_relevance: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            source_types: Filter by source types
            top_k: Number of results
            min_relevance: Minimum relevance score
            filter_metadata: Additional metadata filters

        Returns:
            RetrievalResult with ranked documents
        """
        start_time = datetime.now()
        top_k = top_k or self._default_top_k
        min_relevance = min_relevance if min_relevance is not None else self._relevance_threshold

        documents = []

        if self._vectorstore:
            try:
                # Build filter
                search_filter = {}
                if source_types:
                    search_filter["source_type"] = {
                        "$in": [st.value for st in source_types]
                    }
                if filter_metadata:
                    search_filter.update(filter_metadata)

                # Perform similarity search with scores
                results = self._vectorstore.similarity_search_with_score(
                    query,
                    k=top_k * 2,  # Get extra for filtering
                    filter=search_filter if search_filter else None
                )

                # Process results
                for doc, score in results:
                    # Convert distance to similarity (1 - normalized_distance)
                    relevance_score = 1.0 - min(score, 1.0)

                    if relevance_score >= min_relevance:
                        documents.append(RetrievedDocument(
                            id=doc.metadata.get("document_id", ""),
                            content=doc.page_content,
                            source_type=doc.metadata.get("source_type", "unknown"),
                            title=doc.metadata.get("title", "Untitled"),
                            relevance_score=relevance_score,
                            metadata=doc.metadata
                        ))

                # Sort by relevance and limit
                documents.sort(key=lambda x: x.relevance_score, reverse=True)
                documents = documents[:top_k]

            except Exception as e:
                print(f"Error during retrieval: {e}")

        else:
            # Fallback to keyword search in stored documents
            documents = self._keyword_search(query, source_types, top_k)

        end_time = datetime.now()
        retrieval_time = (end_time - start_time).total_seconds() * 1000

        return RetrievalResult(
            query=query,
            documents=documents,
            total_searched=len(self._documents),
            retrieval_time_ms=retrieval_time
        )

    def _keyword_search(
        self,
        query: str,
        source_types: Optional[List[SourceType]],
        top_k: int
    ) -> List[RetrievedDocument]:
        """Fallback keyword search when vector store unavailable."""
        results = []
        query_lower = query.lower()

        for doc_id, metadata in self._documents.items():
            # Filter by source type
            if source_types:
                if metadata.get("source_type") not in [st.value for st in source_types]:
                    continue

            # Simple keyword matching
            title = metadata.get("title", "").lower()
            content = metadata.get("content", "").lower()

            score = 0.0
            if query_lower in title:
                score = 0.8
            elif query_lower in content:
                score = 0.5

            if score > 0:
                results.append(RetrievedDocument(
                    id=doc_id,
                    content=metadata.get("content", "")[:500],
                    source_type=metadata.get("source_type", "unknown"),
                    title=metadata.get("title", "Untitled"),
                    relevance_score=score,
                    metadata=metadata
                ))

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]

    def multi_query_retrieve(
        self,
        query: str,
        source_types: Optional[List[SourceType]] = None,
        top_k: Optional[int] = None,
        num_queries: int = 3
    ) -> RetrievalResult:
        """
        Multi-query retrieval for better recall.

        Generates multiple query variations and combines results.
        """
        if not self._embeddings:
            return self.retrieve(query, source_types, top_k)

        try:
            from langchain.retrievers.multi_query import MultiQueryRetriever
            from langchain_openai import ChatOpenAI

            # Create multi-query retriever
            llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
            retriever = self._vectorstore.as_retriever(
                search_kwargs={"k": top_k or self._default_top_k}
            )

            multi_retriever = MultiQueryRetriever.from_llm(
                retriever=retriever,
                llm=llm
            )

            # Retrieve
            docs = multi_retriever.get_relevant_documents(query)

            documents = []
            for doc in docs[:top_k or self._default_top_k]:
                documents.append(RetrievedDocument(
                    id=doc.metadata.get("document_id", ""),
                    content=doc.page_content,
                    source_type=doc.metadata.get("source_type", "unknown"),
                    title=doc.metadata.get("title", "Untitled"),
                    relevance_score=0.8,  # Multi-query doesn't return scores
                    metadata=doc.metadata
                ))

            return RetrievalResult(
                query=query,
                documents=documents,
                total_searched=len(self._documents),
                retrieval_time_ms=0.0
            )

        except ImportError:
            # Fall back to standard retrieval
            return self.retrieve(query, source_types, top_k)

    def format_context_for_llm(
        self,
        retrieval_result: RetrievalResult,
        max_tokens: int = 3000,
        include_citations: bool = True
    ) -> str:
        """
        Format retrieved context for LLM consumption.

        Args:
            retrieval_result: Retrieved documents
            max_tokens: Maximum token budget
            include_citations: Whether to include source citations

        Returns:
            Formatted context string
        """
        context_parts = []
        estimated_tokens = 0

        for doc in retrieval_result.documents:
            content_tokens = len(doc.content) // 4  # Rough estimate
            citation = doc.to_citation() if include_citations else ""
            citation_tokens = len(citation) // 4

            if estimated_tokens + content_tokens + citation_tokens > max_tokens:
                available_chars = (max_tokens - estimated_tokens) * 4 - len(citation)
                if available_chars > 100:
                    truncated = doc.content[:available_chars] + "..."
                    if include_citations:
                        context_parts.append(f"{citation}\n{truncated}")
                    else:
                        context_parts.append(truncated)
                break

            if include_citations:
                context_parts.append(f"{citation}\n{doc.content}")
            else:
                context_parts.append(doc.content)
            estimated_tokens += content_tokens + citation_tokens

        return "\n\n---\n\n".join(context_parts)

    def create_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Create a LangChain retriever for use in chains.

        Args:
            search_type: "similarity", "mmr", or "similarity_score_threshold"
            search_kwargs: Additional search parameters

        Returns:
            LangChain retriever
        """
        if not self._vectorstore:
            return None

        search_kwargs = search_kwargs or {"k": self._default_top_k}

        return self._vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )


# ============================================================================
# Knowledge Base Builder
# ============================================================================

class KnowledgeBaseBuilderLangChain:
    """
    Helper for building and populating knowledge bases.

    Supports:
    - Bulk document loading
    - Automatic chunking
    - Metadata extraction
    - File loading (PDF, docs, etc.)
    """

    def __init__(self, rag_engine: RAGEngineLangChain):
        self._engine = rag_engine

    def add_playbook(
        self,
        title: str,
        content: str,
        stage: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Add a sales playbook document."""
        metadata = {"tags": tags or []}
        if stage:
            metadata["stage"] = stage

        return self._engine.add_document(
            content=content,
            source_type=SourceType.SALES_PLAYBOOK,
            title=title,
            metadata=metadata
        )

    def add_policy(
        self,
        title: str,
        content: str,
        policy_type: str = "general",
        access_level: str = "internal"
    ) -> str:
        """Add a policy document."""
        return self._engine.add_document(
            content=content,
            source_type=SourceType.POLICY,
            title=title,
            metadata={
                "policy_type": policy_type,
                "access_level": access_level
            }
        )

    def add_product_doc(
        self,
        title: str,
        content: str,
        product: Optional[str] = None,
        category: Optional[str] = None
    ) -> str:
        """Add product documentation."""
        metadata = {}
        if product:
            metadata["product"] = product
        if category:
            metadata["category"] = category

        return self._engine.add_document(
            content=content,
            source_type=SourceType.PRODUCT_DOCS,
            title=title,
            metadata=metadata
        )

    def add_competitive_intel(
        self,
        competitor: str,
        title: str,
        content: str
    ) -> str:
        """Add competitive intelligence."""
        return self._engine.add_document(
            content=content,
            source_type=SourceType.COMPETITIVE_INTEL,
            title=title,
            metadata={"competitor": competitor}
        )

    def add_case_study(
        self,
        title: str,
        content: str,
        industry: Optional[str] = None,
        outcome: Optional[str] = None
    ) -> str:
        """Add a case study."""
        metadata = {}
        if industry:
            metadata["industry"] = industry
        if outcome:
            metadata["outcome"] = outcome

        return self._engine.add_document(
            content=content,
            source_type=SourceType.CASE_STUDY,
            title=title,
            metadata=metadata
        )

    def add_faq(
        self,
        question: str,
        answer: str,
        category: Optional[str] = None
    ) -> str:
        """Add an FAQ entry."""
        content = f"Q: {question}\n\nA: {answer}"
        return self._engine.add_document(
            content=content,
            source_type=SourceType.FAQ,
            title=question,
            metadata={"category": category} if category else {}
        )

    def index_interaction(
        self,
        interaction_id: UUID,
        opportunity_id: UUID,
        summary: str,
        key_points: List[str],
        timestamp: datetime
    ) -> str:
        """Index an interaction for retrieval."""
        content = f"Summary: {summary}\n\nKey Points:\n"
        for point in key_points:
            content += f"- {point}\n"

        return self._engine.add_document(
            content=content,
            source_type=SourceType.INTERACTION_HISTORY,
            title=f"Interaction {timestamp.strftime('%Y-%m-%d')}",
            metadata={
                "interaction_id": str(interaction_id),
                "opportunity_id": str(opportunity_id),
                "timestamp": timestamp.isoformat()
            },
            document_id=str(interaction_id)
        )

    def load_from_file(
        self,
        file_path: str,
        source_type: SourceType,
        title: Optional[str] = None
    ) -> str:
        """
        Load a document from file.

        Supports PDF, TXT, and other formats via LangChain loaders.
        """
        try:
            from langchain_community.document_loaders import (
                PyPDFLoader,
                TextLoader,
                UnstructuredWordDocumentLoader
            )
            import os

            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                loader = TextLoader(file_path)

            docs = loader.load()
            content = "\n\n".join([doc.page_content for doc in docs])

            return self._engine.add_document(
                content=content,
                source_type=source_type,
                title=title or os.path.basename(file_path),
                metadata={"file_path": file_path}
            )

        except ImportError:
            raise ImportError("Document loaders require langchain-community package")


# ============================================================================
# Factory function
# ============================================================================

def create_rag_engine(
    use_chroma: bool = True,
    persist_directory: str = "./data/chroma",
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> RAGEngineLangChain:
    """
    Factory function to create a configured RAG engine.

    Args:
        use_chroma: Whether to use ChromaDB (vs FAISS)
        persist_directory: Directory for persistence
        embedding_model: Embedding model name
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Configured RAGEngineLangChain instance
    """
    return RAGEngineLangChain(
        persist_directory=persist_directory if use_chroma else None,
        embedding_model=embedding_model,
        backend="chroma" if use_chroma else "faiss",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
