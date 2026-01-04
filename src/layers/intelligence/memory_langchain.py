"""
Memory System with LangChain and Vector Stores

Implements the three-tier memory architecture using:
- ChromaDB/FAISS for episodic memory (semantic search)
- Redis for working memory (persistent, distributed)
- In-memory for policy memory (fast access)

Memory Architecture (Section 4.2):
- Working Memory: Short-term, per-opportunity context
- Episodic Memory: Long-term, semantic similarity search
- Policy Memory: Governance rules and constraints
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type, Union
from uuid import UUID, uuid4
from collections import OrderedDict
import json
import hashlib

from pydantic import BaseModel, Field

from ...core.cognitive_framework import MemoryType


# ============================================================================
# Memory Entry Models
# ============================================================================

@dataclass
class MemoryEntry:
    """A single entry in memory."""
    id: UUID = field(default_factory=uuid4)
    key: str = ""
    value: Any = None
    memory_type: MemoryType = MemoryType.WORKING

    # Temporal metadata
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # Access patterns
    access_count: int = 0
    importance: float = 0.5

    # Context
    opportunity_id: Optional[UUID] = None
    account_id: Optional[UUID] = None
    source: str = ""
    tags: List[str] = field(default_factory=list)

    # Embedding for semantic search
    embedding: Optional[List[float]] = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "key": self.key,
            "value": self.value if isinstance(self.value, (str, int, float, bool, list, dict))
                     else str(self.value),
            "memory_type": self.memory_type.value,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "importance": self.importance,
            "opportunity_id": str(self.opportunity_id) if self.opportunity_id else None,
            "account_id": str(self.account_id) if self.account_id else None,
            "source": self.source,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]),
            key=data["key"],
            value=data["value"],
            memory_type=MemoryType(data["memory_type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
            opportunity_id=UUID(data["opportunity_id"]) if data.get("opportunity_id") else None,
            account_id=UUID(data["account_id"]) if data.get("account_id") else None,
            source=data.get("source", ""),
            tags=data.get("tags", [])
        )


class SemanticSearchResult(BaseModel):
    """Result from semantic search."""
    entry: Dict[str, Any]
    score: float
    content: str


# ============================================================================
# Working Memory with Optional Redis Backend
# ============================================================================

class WorkingMemoryLangChain:
    """
    Short-term working memory per opportunity.

    Uses Redis for persistence when available, falls back to in-memory.
    Maintains current context for ongoing deals:
    - Recent interaction summaries
    - Current stakeholder state
    - Active risks and objections
    - Pending actions
    """

    def __init__(
        self,
        capacity: int = 100,
        ttl_hours: int = 168,
        redis_url: Optional[str] = None
    ):
        self._capacity = capacity
        self._ttl_hours = ttl_hours
        self._redis_client = None
        self._in_memory_store: Dict[str, OrderedDict[str, MemoryEntry]] = {}

        # Try to initialize Redis
        if redis_url:
            try:
                import redis
                self._redis_client = redis.from_url(redis_url)
                self._redis_client.ping()  # Test connection
            except Exception:
                self._redis_client = None

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.WORKING

    @property
    def is_persistent(self) -> bool:
        """Check if using persistent backend."""
        return self._redis_client is not None

    def _get_redis_key(self, opportunity_id: UUID, key: str) -> str:
        """Generate Redis key."""
        return f"crm:working:{opportunity_id}:{key}"

    def _get_opp_store(self, opportunity_id: UUID) -> OrderedDict:
        """Get or create in-memory store for an opportunity."""
        key = str(opportunity_id)
        if key not in self._in_memory_store:
            self._in_memory_store[key] = OrderedDict()
        return self._in_memory_store[key]

    def store(
        self,
        key: str,
        value: Any,
        opportunity_id: UUID,
        importance: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> None:
        """Store value in working memory."""
        entry = MemoryEntry(
            key=key,
            value=value,
            memory_type=MemoryType.WORKING,
            expires_at=datetime.now() + timedelta(hours=self._ttl_hours),
            importance=importance,
            opportunity_id=opportunity_id,
            tags=metadata.get("tags", []) if metadata else []
        )

        if self._redis_client:
            # Store in Redis
            redis_key = self._get_redis_key(opportunity_id, key)
            self._redis_client.setex(
                redis_key,
                timedelta(hours=self._ttl_hours),
                json.dumps(entry.to_dict())
            )
        else:
            # Store in memory with LRU eviction
            opp_store = self._get_opp_store(opportunity_id)

            while len(opp_store) >= self._capacity:
                opp_store.popitem(last=False)

            opp_store[key] = entry
            opp_store.move_to_end(key)

    def retrieve(self, key: str, opportunity_id: UUID) -> Optional[Any]:
        """Retrieve value from working memory."""
        if self._redis_client:
            redis_key = self._get_redis_key(opportunity_id, key)
            data = self._redis_client.get(redis_key)
            if data:
                entry_dict = json.loads(data)
                entry = MemoryEntry.from_dict(entry_dict)
                # Update access
                entry.accessed_at = datetime.now()
                entry.access_count += 1
                self._redis_client.setex(
                    redis_key,
                    timedelta(hours=self._ttl_hours),
                    json.dumps(entry.to_dict())
                )
                return entry.value
            return None
        else:
            opp_store = self._get_opp_store(opportunity_id)
            if key not in opp_store:
                return None

            entry = opp_store[key]
            if entry.is_expired():
                del opp_store[key]
                return None

            entry.accessed_at = datetime.now()
            entry.access_count += 1
            opp_store.move_to_end(key)
            return entry.value

    def get_context(self, opportunity_id: UUID) -> Dict[str, Any]:
        """Get full working memory context for an opportunity."""
        context = {}

        if self._redis_client:
            pattern = f"crm:working:{opportunity_id}:*"
            keys = self._redis_client.keys(pattern)
            for redis_key in keys:
                data = self._redis_client.get(redis_key)
                if data:
                    entry_dict = json.loads(data)
                    context[entry_dict["key"]] = entry_dict["value"]
        else:
            opp_store = self._get_opp_store(opportunity_id)
            for key, entry in opp_store.items():
                if not entry.is_expired():
                    context[key] = entry.value

        return context

    def search(self, query: str, opportunity_id: UUID, top_k: int = 5) -> List[MemoryEntry]:
        """Search working memory by key prefix or tags."""
        results = []

        if self._redis_client:
            pattern = f"crm:working:{opportunity_id}:*"
            keys = self._redis_client.keys(pattern)
            for redis_key in keys:
                data = self._redis_client.get(redis_key)
                if data:
                    entry_dict = json.loads(data)
                    entry = MemoryEntry.from_dict(entry_dict)
                    score = 0
                    if query.lower() in entry.key.lower():
                        score = 0.8
                    elif any(query.lower() in tag.lower() for tag in entry.tags):
                        score = 0.6
                    if score > 0:
                        results.append((entry, score))
        else:
            opp_store = self._get_opp_store(opportunity_id)
            for key, entry in opp_store.items():
                if entry.is_expired():
                    continue
                score = 0
                if query.lower() in key.lower():
                    score = 0.8
                elif any(query.lower() in tag.lower() for tag in entry.tags):
                    score = 0.6
                if score > 0:
                    results.append((entry, score))

        results.sort(key=lambda x: (x[1], x[0].importance), reverse=True)
        return [r[0] for r in results[:top_k]]

    def clear_opportunity(self, opportunity_id: UUID) -> None:
        """Clear working memory for an opportunity."""
        if self._redis_client:
            pattern = f"crm:working:{opportunity_id}:*"
            keys = self._redis_client.keys(pattern)
            if keys:
                self._redis_client.delete(*keys)
        else:
            key = str(opportunity_id)
            if key in self._in_memory_store:
                del self._in_memory_store[key]


# ============================================================================
# Episodic Memory with ChromaDB/FAISS Vector Store
# ============================================================================

class EpisodicMemoryLangChain:
    """
    Long-term episodic memory with semantic search.

    Uses ChromaDB or FAISS for vector storage and retrieval:
    - Successful deal patterns
    - Objection handling outcomes
    - Stakeholder relationship evolution
    - Win/loss factors
    """

    def __init__(
        self,
        collection_name: str = "crm_episodic_memory",
        persist_directory: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        backend: str = "chroma"  # "chroma" or "faiss"
    ):
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._embedding_model = embedding_model
        self._backend = backend

        self._vectorstore = None
        self._embeddings = None
        self._entries: Dict[str, MemoryEntry] = {}

        # Indices for efficient lookup
        self._by_opportunity: Dict[str, List[str]] = {}
        self._by_account: Dict[str, List[str]] = {}

        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """Initialize the vector store backend."""
        try:
            # Try LangChain OpenAI embeddings first
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(model=self._embedding_model)
        except ImportError:
            try:
                # Fall back to HuggingFace embeddings
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            except ImportError:
                self._embeddings = None

        if self._embeddings is None:
            return

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
                from langchain_community.vectorstores import FAISS
                # FAISS needs initial documents, we'll create lazily
                self._vectorstore = None  # Will be created on first store
        except ImportError:
            self._vectorstore = None

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.EPISODIC

    @property
    def has_vectorstore(self) -> bool:
        """Check if vector store is available."""
        return self._vectorstore is not None or self._embeddings is not None

    def _generate_content(self, entry: MemoryEntry) -> str:
        """Generate searchable content from entry."""
        parts = [entry.key]

        if isinstance(entry.value, str):
            parts.append(entry.value)
        elif isinstance(entry.value, dict):
            for k, v in entry.value.items():
                if isinstance(v, str):
                    parts.append(f"{k}: {v}")
                elif isinstance(v, list):
                    parts.append(f"{k}: {', '.join(str(x) for x in v)}")

        if entry.tags:
            parts.append(f"Tags: {', '.join(entry.tags)}")

        return "\n".join(parts)

    def store(
        self,
        key: str,
        value: Any,
        opportunity_id: Optional[UUID] = None,
        account_id: Optional[UUID] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store episodic memory with optional vector embedding."""
        entry = MemoryEntry(
            key=key,
            value=value,
            memory_type=MemoryType.EPISODIC,
            opportunity_id=opportunity_id,
            account_id=account_id,
            tags=metadata.get("tags", []) if metadata else [],
            source=metadata.get("source", "") if metadata else ""
        )

        entry_id = str(entry.id)
        self._entries[entry_id] = entry

        # Update indices
        if opportunity_id:
            opp_key = str(opportunity_id)
            if opp_key not in self._by_opportunity:
                self._by_opportunity[opp_key] = []
            self._by_opportunity[opp_key].append(entry_id)

        if account_id:
            acc_key = str(account_id)
            if acc_key not in self._by_account:
                self._by_account[acc_key] = []
            self._by_account[acc_key].append(entry_id)

        # Add to vector store if available
        if self._embeddings:
            content = self._generate_content(entry)
            doc_metadata = {
                "entry_id": entry_id,
                "key": key,
                "opportunity_id": str(opportunity_id) if opportunity_id else "",
                "account_id": str(account_id) if account_id else "",
                "tags": ",".join(entry.tags),
                "created_at": entry.created_at.isoformat()
            }

            try:
                if self._backend == "chroma" and self._vectorstore:
                    from langchain_core.documents import Document
                    doc = Document(page_content=content, metadata=doc_metadata)
                    self._vectorstore.add_documents([doc], ids=[entry_id])
                elif self._backend == "faiss":
                    from langchain_community.vectorstores import FAISS
                    from langchain_core.documents import Document
                    doc = Document(page_content=content, metadata=doc_metadata)
                    if self._vectorstore is None:
                        self._vectorstore = FAISS.from_documents([doc], self._embeddings)
                    else:
                        self._vectorstore.add_documents([doc], ids=[entry_id])
            except Exception:
                pass  # Continue without vector store

        return entry_id

    def retrieve(self, entry_id: str) -> Optional[Any]:
        """Retrieve by entry ID."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.accessed_at = datetime.now()
            entry.access_count += 1
            return entry.value
        return None

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filter_opportunity_id: Optional[UUID] = None,
        filter_tags: Optional[List[str]] = None
    ) -> List[SemanticSearchResult]:
        """Perform semantic search over episodic memory."""
        results = []

        if self._vectorstore:
            try:
                # Build filter
                search_filter = {}
                if filter_opportunity_id:
                    search_filter["opportunity_id"] = str(filter_opportunity_id)

                # Perform similarity search
                docs_with_scores = self._vectorstore.similarity_search_with_score(
                    query,
                    k=top_k,
                    filter=search_filter if search_filter else None
                )

                for doc, score in docs_with_scores:
                    entry_id = doc.metadata.get("entry_id")
                    entry = self._entries.get(entry_id)

                    if entry:
                        # Apply tag filter if specified
                        if filter_tags and not any(t in entry.tags for t in filter_tags):
                            continue

                        results.append(SemanticSearchResult(
                            entry=entry.to_dict(),
                            score=float(1 - score),  # Convert distance to similarity
                            content=doc.page_content
                        ))
            except Exception:
                # Fall back to keyword search
                results = self._keyword_search(query, top_k, filter_tags)
        else:
            results = self._keyword_search(query, top_k, filter_tags)

        return results

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        filter_tags: Optional[List[str]] = None
    ) -> List[SemanticSearchResult]:
        """Fallback keyword search."""
        results = []
        query_lower = query.lower()

        for entry_id, entry in self._entries.items():
            if filter_tags and not any(t in entry.tags for t in filter_tags):
                continue

            score = 0
            content = self._generate_content(entry)

            if query_lower in entry.key.lower():
                score = 0.7
            elif any(query_lower in tag.lower() for tag in entry.tags):
                score = 0.6
            elif query_lower in content.lower():
                score = 0.5

            if score > 0:
                results.append(SemanticSearchResult(
                    entry=entry.to_dict(),
                    score=score,
                    content=content
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Search episodic memory (backwards compatible)."""
        semantic_results = self.semantic_search(query, top_k)
        return [MemoryEntry.from_dict(r.entry) for r in semantic_results]

    def get_by_opportunity(self, opportunity_id: UUID) -> List[MemoryEntry]:
        """Get all memories for an opportunity."""
        opp_key = str(opportunity_id)
        entry_ids = self._by_opportunity.get(opp_key, [])
        return [self._entries[eid] for eid in entry_ids if eid in self._entries]

    def get_by_account(self, account_id: UUID) -> List[MemoryEntry]:
        """Get all memories for an account."""
        acc_key = str(account_id)
        entry_ids = self._by_account.get(acc_key, [])
        return [self._entries[eid] for eid in entry_ids if eid in self._entries]

    def get_similar_patterns(
        self,
        pattern_description: str,
        outcome: Optional[str] = None,
        top_k: int = 5
    ) -> List[MemoryEntry]:
        """Find similar historical patterns."""
        filter_tags = [outcome] if outcome else None
        results = self.semantic_search(
            pattern_description,
            top_k=top_k,
            filter_tags=filter_tags
        )
        return [MemoryEntry.from_dict(r.entry) for r in results]


# ============================================================================
# Policy Memory
# ============================================================================

class PolicyMemoryLangChain:
    """
    Policy memory for governance constraints.

    Stores and retrieves:
    - Approval requirements
    - Discount thresholds
    - Compliance rules
    - Escalation policies
    - Role permissions
    """

    def __init__(self):
        self._policies: Dict[str, Dict[str, Any]] = {}
        self._active_version: str = "1.0"

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.POLICY

    def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict] = None
    ) -> None:
        """Store a policy."""
        version = metadata.get("version", self._active_version) if metadata else self._active_version

        if version not in self._policies:
            self._policies[version] = {}

        self._policies[version][key] = {
            "value": value,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }

    def retrieve(self, key: str, version: Optional[str] = None) -> Optional[Any]:
        """Retrieve a policy."""
        version = version or self._active_version

        if version in self._policies and key in self._policies[version]:
            return self._policies[version][key]["value"]
        return None

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search policies."""
        results = []
        policies = self._policies.get(self._active_version, {})

        for key, data in policies.items():
            if query.lower() in key.lower():
                results.append({"key": key, **data})

        return results[:top_k]

    def check_policy(
        self,
        action_type: str,
        context: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Check if an action is allowed by policy.

        Returns (is_allowed, violation_reasons)
        """
        violations = []

        # Check discount policy
        if action_type == "apply_discount":
            discount_limit = self.retrieve("max_discount_percent")
            if discount_limit and context.get("discount_percent", 0) > discount_limit:
                violations.append(f"Discount exceeds limit of {discount_limit}%")

        # Check approval requirements
        if action_type == "create_proposal":
            amount = context.get("amount", 0)
            approval_threshold = self.retrieve("approval_threshold_amount")
            if approval_threshold and amount > approval_threshold:
                violations.append(
                    f"Amount ${amount:,.2f} requires approval "
                    f"(threshold: ${approval_threshold:,.2f})"
                )

        # Check role permissions
        if action_type in ["modify_contract", "apply_discount", "extend_payment_terms"]:
            user_role = context.get("user_role")
            allowed_roles = self.retrieve(f"{action_type}_allowed_roles")
            if allowed_roles and user_role not in allowed_roles:
                violations.append(f"Role '{user_role}' not authorized for {action_type}")

        return (len(violations) == 0, violations)

    def set_active_version(self, version: str) -> None:
        """Set the active policy version."""
        self._active_version = version

    def get_all_policies(self, version: Optional[str] = None) -> Dict:
        """Get all policies for a version."""
        version = version or self._active_version
        return self._policies.get(version, {})


# ============================================================================
# Unified Memory Manager with LangChain
# ============================================================================

class MemoryManagerLangChain:
    """
    Central manager for all memory types with LangChain integration.

    Coordinates:
    - Memory lifecycle
    - Cross-memory queries
    - Memory consolidation
    - Conversation memory for LLM context
    """

    def __init__(
        self,
        working_memory_capacity: int = 100,
        working_memory_ttl_hours: int = 168,
        redis_url: Optional[str] = None,
        chroma_persist_directory: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        vectorstore_backend: str = "chroma"
    ):
        self.working = WorkingMemoryLangChain(
            capacity=working_memory_capacity,
            ttl_hours=working_memory_ttl_hours,
            redis_url=redis_url
        )

        self.episodic = EpisodicMemoryLangChain(
            collection_name="crm_episodic_memory",
            persist_directory=chroma_persist_directory,
            embedding_model=embedding_model,
            backend=vectorstore_backend
        )

        self.policy = PolicyMemoryLangChain()

        # LangChain conversation memory per opportunity
        self._conversation_memories: Dict[str, Any] = {}

    def get_conversation_memory(self, opportunity_id: UUID):
        """Get or create LangChain conversation memory for an opportunity."""
        opp_key = str(opportunity_id)

        if opp_key not in self._conversation_memories:
            try:
                from langchain.memory import ConversationBufferWindowMemory
                self._conversation_memories[opp_key] = ConversationBufferWindowMemory(
                    k=10,  # Keep last 10 exchanges
                    memory_key="chat_history",
                    return_messages=True
                )
            except ImportError:
                self._conversation_memories[opp_key] = None

        return self._conversation_memories[opp_key]

    def get_full_context(
        self,
        opportunity_id: UUID,
        account_id: Optional[UUID] = None,
        include_similar_patterns: bool = True
    ) -> Dict[str, Any]:
        """
        Build complete context from all memory types.

        Used to provide rich context for reasoning.
        """
        context = {
            "working_memory": self.working.get_context(opportunity_id),
            "episodic_memory": [],
            "similar_patterns": [],
            "policies": {},
            "conversation_history": []
        }

        # Get opportunity episodic memories
        opp_memories = self.episodic.get_by_opportunity(opportunity_id)
        context["episodic_memory"] = [
            {"key": m.key, "value": m.value, "created_at": m.created_at.isoformat()}
            for m in opp_memories[-10:]
        ]

        # Get account-level memories
        if account_id:
            acc_memories = self.episodic.get_by_account(account_id)
            context["account_history"] = [
                {"key": m.key, "value": m.value}
                for m in acc_memories[-5:]
            ]

        # Find similar patterns for current context
        if include_similar_patterns and context["working_memory"]:
            working_summary = json.dumps(context["working_memory"], default=str)[:500]
            similar = self.episodic.get_similar_patterns(
                working_summary,
                outcome="won",
                top_k=3
            )
            context["similar_patterns"] = [
                {"key": m.key, "value": m.value}
                for m in similar
            ]

        # Get relevant policies
        context["policies"] = self.policy.get_all_policies()

        # Get conversation history
        conv_memory = self.get_conversation_memory(opportunity_id)
        if conv_memory:
            try:
                messages = conv_memory.load_memory_variables({})
                context["conversation_history"] = messages.get("chat_history", [])
            except Exception:
                pass

        return context

    def consolidate_to_episodic(
        self,
        opportunity_id: UUID,
        event_type: str,
        summary: Dict[str, Any],
        account_id: Optional[UUID] = None
    ) -> str:
        """
        Consolidate working memory to episodic for long-term storage.

        Called when significant events occur (stage change, close, etc.)
        """
        working_context = self.working.get_context(opportunity_id)

        # Create episodic entry
        entry_id = self.episodic.store(
            key=f"{event_type}_{opportunity_id}",
            value={
                "event_type": event_type,
                "summary": summary,
                "working_memory_snapshot": working_context
            },
            opportunity_id=opportunity_id,
            account_id=account_id,
            metadata={
                "tags": [event_type, "consolidated"],
                "source": "memory_manager"
            }
        )

        return entry_id

    def record_outcome(
        self,
        opportunity_id: UUID,
        outcome: str,
        details: Dict[str, Any],
        account_id: Optional[UUID] = None
    ) -> str:
        """
        Record opportunity outcome for learning.

        This enables pattern matching for similar future opportunities.
        """
        working_context = self.working.get_context(opportunity_id)

        entry_id = self.episodic.store(
            key=f"outcome_{opportunity_id}",
            value={
                "outcome": outcome,
                "details": details,
                "context_at_close": working_context
            },
            opportunity_id=opportunity_id,
            account_id=account_id,
            metadata={
                "tags": [outcome, "outcome", "learning"],
                "source": "outcome_recording"
            }
        )

        return entry_id

    def clear_closed_opportunity(self, opportunity_id: UUID) -> None:
        """Clean up working memory when opportunity closes."""
        # Consolidate first
        self.consolidate_to_episodic(
            opportunity_id,
            "opportunity_closed",
            {"final_state": self.working.get_context(opportunity_id)}
        )

        # Clear working memory
        self.working.clear_opportunity(opportunity_id)

        # Clear conversation memory
        opp_key = str(opportunity_id)
        if opp_key in self._conversation_memories:
            del self._conversation_memories[opp_key]

    def add_conversation_exchange(
        self,
        opportunity_id: UUID,
        human_input: str,
        ai_output: str
    ) -> None:
        """Add a conversation exchange to memory."""
        conv_memory = self.get_conversation_memory(opportunity_id)
        if conv_memory:
            try:
                conv_memory.save_context(
                    {"input": human_input},
                    {"output": ai_output}
                )
            except Exception:
                pass


# ============================================================================
# Factory function for easy initialization
# ============================================================================

def create_memory_manager(
    use_redis: bool = False,
    use_chroma: bool = True,
    redis_url: str = "redis://localhost:6379",
    chroma_persist_directory: str = "./data/chroma",
    embedding_model: str = "text-embedding-3-small"
) -> MemoryManagerLangChain:
    """
    Factory function to create a configured memory manager.

    Args:
        use_redis: Whether to use Redis for working memory
        use_chroma: Whether to use ChromaDB for episodic memory
        redis_url: Redis connection URL
        chroma_persist_directory: ChromaDB persistence directory
        embedding_model: Embedding model for semantic search

    Returns:
        Configured MemoryManagerLangChain instance
    """
    return MemoryManagerLangChain(
        redis_url=redis_url if use_redis else None,
        chroma_persist_directory=chroma_persist_directory if use_chroma else None,
        embedding_model=embedding_model
    )
