"""
Memory Design - Working, Episodic, and Policy Memory

From Section 4.2:
- Short-term working memory per opportunity
- Long-term episodic memory across interactions
- Policy memory for constraints

This module implements the memory architecture that enables
the cognitive CRM to maintain context and learn from experience.

Memory types map to different cognitive functions:
- Working: Current opportunity context (sensemaking)
- Episodic: Historical patterns (learning)
- Policy: Governance constraints (action control)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID, uuid4
from collections import OrderedDict

from ...core.cognitive_framework import MemoryType, MemoryStore


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
    importance: float = 0.5  # 0.0 to 1.0

    # Context
    opportunity_id: Optional[UUID] = None
    source: str = ""
    tags: list = field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class WorkingMemory(MemoryStore):
    """
    Short-term working memory per opportunity.

    Maintains current context for ongoing deals:
    - Recent interaction summaries
    - Current stakeholder state
    - Active risks and objections
    - Pending actions

    Characteristics:
    - Limited capacity (LRU eviction)
    - Fast access
    - Opportunity-scoped
    """

    def __init__(self, capacity: int = 100, ttl_hours: int = 168):  # 1 week default TTL
        self._capacity = capacity
        self._ttl_hours = ttl_hours
        self._store: dict[str, OrderedDict[str, MemoryEntry]] = {}

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.WORKING

    def _get_opp_store(self, opportunity_id: UUID) -> OrderedDict:
        """Get or create store for an opportunity."""
        key = str(opportunity_id)
        if key not in self._store:
            self._store[key] = OrderedDict()
        return self._store[key]

    def store(
        self,
        key: str,
        value: Any,
        opportunity_id: UUID,
        importance: float = 0.5,
        metadata: dict = None
    ) -> None:
        """Store value in working memory."""
        opp_store = self._get_opp_store(opportunity_id)

        # Check capacity and evict if needed
        while len(opp_store) >= self._capacity:
            # Remove least recently used (first item)
            opp_store.popitem(last=False)

        # Create entry
        entry = MemoryEntry(
            key=key,
            value=value,
            memory_type=MemoryType.WORKING,
            expires_at=datetime.now() + timedelta(hours=self._ttl_hours),
            importance=importance,
            opportunity_id=opportunity_id,
            tags=metadata.get("tags", []) if metadata else []
        )

        opp_store[key] = entry

        # Move to end (most recently used)
        opp_store.move_to_end(key)

    def retrieve(self, key: str, opportunity_id: UUID) -> Optional[Any]:
        """Retrieve value from working memory."""
        opp_store = self._get_opp_store(opportunity_id)

        if key not in opp_store:
            return None

        entry = opp_store[key]

        # Check expiration
        if entry.is_expired():
            del opp_store[key]
            return None

        # Update access
        entry.accessed_at = datetime.now()
        entry.access_count += 1
        opp_store.move_to_end(key)

        return entry.value

    def search(self, query: str, opportunity_id: UUID, top_k: int = 5) -> list:
        """Search working memory by key prefix or tags."""
        opp_store = self._get_opp_store(opportunity_id)
        results = []

        for key, entry in opp_store.items():
            if entry.is_expired():
                continue

            # Simple matching on key and tags
            score = 0
            if query.lower() in key.lower():
                score = 0.8
            elif any(query.lower() in tag.lower() for tag in entry.tags):
                score = 0.6

            if score > 0:
                results.append((entry, score))

        # Sort by score and importance
        results.sort(key=lambda x: (x[1], x[0].importance), reverse=True)
        return [r[0] for r in results[:top_k]]

    def get_context(self, opportunity_id: UUID) -> dict:
        """Get full working memory context for an opportunity."""
        opp_store = self._get_opp_store(opportunity_id)
        context = {}

        for key, entry in opp_store.items():
            if not entry.is_expired():
                context[key] = entry.value

        return context

    def clear_opportunity(self, opportunity_id: UUID) -> None:
        """Clear working memory for an opportunity."""
        key = str(opportunity_id)
        if key in self._store:
            del self._store[key]


class EpisodicMemory(MemoryStore):
    """
    Long-term episodic memory across interactions.

    Stores historical interaction patterns:
    - Successful deal patterns
    - Objection handling outcomes
    - Stakeholder relationship evolution
    - Win/loss factors

    Characteristics:
    - Large capacity
    - Semantic search support
    - Cross-opportunity learning
    """

    def __init__(self, embedding_index=None):
        self._entries: dict[str, MemoryEntry] = {}
        self._embedding_index = embedding_index

        # Indices for efficient lookup
        self._by_opportunity: dict[str, list[str]] = {}
        self._by_account: dict[str, list[str]] = {}
        self._by_tag: dict[str, list[str]] = {}

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.EPISODIC

    def store(
        self,
        key: str,
        value: Any,
        opportunity_id: UUID = None,
        account_id: UUID = None,
        metadata: dict = None
    ) -> None:
        """Store episodic memory."""
        entry = MemoryEntry(
            key=key,
            value=value,
            memory_type=MemoryType.EPISODIC,
            opportunity_id=opportunity_id,
            tags=metadata.get("tags", []) if metadata else [],
            source=metadata.get("source", "") if metadata else ""
        )

        entry_key = str(entry.id)
        self._entries[entry_key] = entry

        # Update indices
        if opportunity_id:
            opp_key = str(opportunity_id)
            if opp_key not in self._by_opportunity:
                self._by_opportunity[opp_key] = []
            self._by_opportunity[opp_key].append(entry_key)

        if account_id:
            acc_key = str(account_id)
            if acc_key not in self._by_account:
                self._by_account[acc_key] = []
            self._by_account[acc_key].append(entry_key)

        for tag in entry.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(entry_key)

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve by entry ID."""
        entry = self._entries.get(key)
        if entry:
            entry.accessed_at = datetime.now()
            entry.access_count += 1
            return entry.value
        return None

    def search(self, query: str, top_k: int = 5) -> list:
        """Search episodic memory."""
        results = []

        for entry_key, entry in self._entries.items():
            score = 0

            # Key matching
            if query.lower() in entry.key.lower():
                score = 0.7

            # Tag matching
            for tag in entry.tags:
                if query.lower() in tag.lower():
                    score = max(score, 0.6)

            # Value content matching (if string)
            if isinstance(entry.value, str) and query.lower() in entry.value.lower():
                score = max(score, 0.5)
            elif isinstance(entry.value, dict):
                for v in entry.value.values():
                    if isinstance(v, str) and query.lower() in v.lower():
                        score = max(score, 0.5)
                        break

            if score > 0:
                results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:top_k]]

    def get_by_opportunity(self, opportunity_id: UUID) -> list[MemoryEntry]:
        """Get all memories for an opportunity."""
        opp_key = str(opportunity_id)
        entry_keys = self._by_opportunity.get(opp_key, [])
        return [self._entries[k] for k in entry_keys if k in self._entries]

    def get_by_account(self, account_id: UUID) -> list[MemoryEntry]:
        """Get all memories for an account."""
        acc_key = str(account_id)
        entry_keys = self._by_account.get(acc_key, [])
        return [self._entries[k] for k in entry_keys if k in self._entries]

    def get_similar_patterns(
        self,
        pattern_description: str,
        outcome: str = None,
        top_k: int = 5
    ) -> list[MemoryEntry]:
        """Find similar historical patterns."""
        results = self.search(pattern_description, top_k=top_k * 2)

        if outcome:
            # Filter by outcome
            results = [r for r in results if r.tags and outcome in r.tags]

        return results[:top_k]


class PolicyMemory(MemoryStore):
    """
    Policy memory for governance constraints.

    Stores and retrieves:
    - Approval requirements
    - Discount thresholds
    - Compliance rules
    - Escalation policies
    - Role permissions

    Characteristics:
    - Hierarchical structure
    - Version controlled
    - Always available (no TTL)
    """

    def __init__(self):
        self._policies: dict[str, dict] = {}
        self._active_version: str = "1.0"

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.POLICY

    def store(self, key: str, value: Any, metadata: dict = None) -> None:
        """Store a policy."""
        version = metadata.get("version", self._active_version) if metadata else self._active_version

        if version not in self._policies:
            self._policies[version] = {}

        self._policies[version][key] = {
            "value": value,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }

    def retrieve(self, key: str, version: str = None) -> Optional[Any]:
        """Retrieve a policy."""
        version = version or self._active_version

        if version in self._policies and key in self._policies[version]:
            return self._policies[version][key]["value"]
        return None

    def search(self, query: str, top_k: int = 5) -> list:
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
        context: dict
    ) -> tuple[bool, list[str]]:
        """
        Check if an action is allowed by policy.

        Returns (is_allowed, reasons)
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
                violations.append(f"Amount ${amount} requires approval (threshold: ${approval_threshold})")

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

    def get_all_policies(self, version: str = None) -> dict:
        """Get all policies for a version."""
        version = version or self._active_version
        return self._policies.get(version, {})


class MemoryManager:
    """
    Central manager for all memory types.

    Coordinates:
    - Memory lifecycle
    - Cross-memory queries
    - Memory consolidation
    - Garbage collection
    """

    def __init__(self):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.policy = PolicyMemory()

    def get_full_context(self, opportunity_id: UUID, account_id: UUID = None) -> dict:
        """
        Build complete context from all memory types.

        Used to provide rich context for reasoning.
        """
        context = {
            "working_memory": self.working.get_context(opportunity_id),
            "episodic_memory": [],
            "policies": {}
        }

        # Get relevant episodic memories
        opp_memories = self.episodic.get_by_opportunity(opportunity_id)
        context["episodic_memory"] = [
            {"key": m.key, "value": m.value, "created_at": m.created_at.isoformat()}
            for m in opp_memories[-10:]  # Last 10
        ]

        # Add account-level memories
        if account_id:
            acc_memories = self.episodic.get_by_account(account_id)
            context["account_history"] = [
                {"key": m.key, "value": m.value}
                for m in acc_memories[-5:]
            ]

        # Get relevant policies
        context["policies"] = self.policy.get_all_policies()

        return context

    def consolidate_to_episodic(
        self,
        opportunity_id: UUID,
        event_type: str,
        summary: dict
    ) -> None:
        """
        Consolidate working memory to episodic for long-term storage.

        Called when significant events occur (stage change, close, etc.)
        """
        working_context = self.working.get_context(opportunity_id)

        # Create episodic entry
        self.episodic.store(
            key=f"{event_type}_{opportunity_id}",
            value={
                "event_type": event_type,
                "summary": summary,
                "working_memory_snapshot": working_context
            },
            opportunity_id=opportunity_id,
            metadata={
                "tags": [event_type, "consolidated"],
                "source": "memory_manager"
            }
        )

    def clear_closed_opportunity(self, opportunity_id: UUID) -> None:
        """Clean up working memory when opportunity closes."""
        # Consolidate first
        self.consolidate_to_episodic(
            opportunity_id,
            "opportunity_closed",
            {"final_state": self.working.get_context(opportunity_id)}
        )

        # Then clear working memory
        self.working.clear_opportunity(opportunity_id)
