"""
Canonical Event Schema for Audit Logs

From Section 4.1: "canonical event schema for audit logs"

This module defines:
- Standardized event format across all sources
- Audit trail for governance
- Event sourcing foundation

The event schema supports:
- Transparency and contestability
- Replay for debugging and learning
- Compliance requirements
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4
import json


class EventCategory(Enum):
    """High-level event categories."""
    SIGNAL = "signal"           # Incoming customer signal
    INFERENCE = "inference"     # Intelligence layer output
    ACTION = "action"           # Orchestrated action
    OUTCOME = "outcome"         # Result/feedback
    GOVERNANCE = "governance"   # Approval, override, audit


class EventSeverity(Enum):
    """Event severity for filtering and alerting."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CanonicalEvent:
    """
    Standardized event format for all system events.

    Design principles:
    - Immutable after creation
    - Self-contained with full context
    - Traceable via correlation IDs
    - Auditable with actor information
    """
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)

    # Classification
    category: EventCategory = EventCategory.SIGNAL
    event_type: str = ""
    severity: EventSeverity = EventSeverity.INFO

    # Content
    payload: dict = field(default_factory=dict)

    # Context and tracing
    correlation_id: Optional[UUID] = None  # Links related events
    causation_id: Optional[UUID] = None    # Event that caused this one
    session_id: Optional[UUID] = None

    # Entity references
    account_id: Optional[UUID] = None
    contact_id: Optional[UUID] = None
    opportunity_id: Optional[UUID] = None

    # Actor information (for audit)
    actor_type: str = "system"  # system, user, automation
    actor_id: Optional[str] = None

    # Source
    source_system: str = ""
    source_version: str = ""

    # Metadata
    tags: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "event_type": self.event_type,
            "severity": self.severity.value,
            "payload": self.payload,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "causation_id": str(self.causation_id) if self.causation_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "account_id": str(self.account_id) if self.account_id else None,
            "contact_id": str(self.contact_id) if self.contact_id else None,
            "opportunity_id": str(self.opportunity_id) if self.opportunity_id else None,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "source_system": self.source_system,
            "source_version": self.source_version,
            "tags": self.tags,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalEvent":
        """Create event from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else uuid4(),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            category=EventCategory(data.get("category", "signal")),
            event_type=data.get("event_type", ""),
            severity=EventSeverity(data.get("severity", "info")),
            payload=data.get("payload", {}),
            correlation_id=UUID(data["correlation_id"]) if data.get("correlation_id") else None,
            causation_id=UUID(data["causation_id"]) if data.get("causation_id") else None,
            session_id=UUID(data["session_id"]) if data.get("session_id") else None,
            account_id=UUID(data["account_id"]) if data.get("account_id") else None,
            contact_id=UUID(data["contact_id"]) if data.get("contact_id") else None,
            opportunity_id=UUID(data["opportunity_id"]) if data.get("opportunity_id") else None,
            actor_type=data.get("actor_type", "system"),
            actor_id=data.get("actor_id"),
            source_system=data.get("source_system", ""),
            source_version=data.get("source_version", ""),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )


class EventStore:
    """
    Event store for persisting and querying events.

    Provides:
    - Append-only event storage
    - Query by various dimensions
    - Correlation chain traversal
    - Export for analysis

    In production, this would be backed by a proper event store
    (EventStoreDB, Kafka, etc.)
    """

    def __init__(self):
        self._events: list[CanonicalEvent] = []
        self._index_by_correlation: dict[str, list[int]] = {}
        self._index_by_opportunity: dict[str, list[int]] = {}
        self._index_by_type: dict[str, list[int]] = {}

    def append(self, event: CanonicalEvent) -> None:
        """Append event to store (immutable)."""
        idx = len(self._events)
        self._events.append(event)

        # Update indices
        if event.correlation_id:
            key = str(event.correlation_id)
            if key not in self._index_by_correlation:
                self._index_by_correlation[key] = []
            self._index_by_correlation[key].append(idx)

        if event.opportunity_id:
            key = str(event.opportunity_id)
            if key not in self._index_by_opportunity:
                self._index_by_opportunity[key] = []
            self._index_by_opportunity[key].append(idx)

        if event.event_type:
            if event.event_type not in self._index_by_type:
                self._index_by_type[event.event_type] = []
            self._index_by_type[event.event_type].append(idx)

    def get_by_correlation(self, correlation_id: UUID) -> list[CanonicalEvent]:
        """Get all events in a correlation chain."""
        key = str(correlation_id)
        indices = self._index_by_correlation.get(key, [])
        return [self._events[i] for i in indices]

    def get_by_opportunity(self, opportunity_id: UUID) -> list[CanonicalEvent]:
        """Get all events for an opportunity."""
        key = str(opportunity_id)
        indices = self._index_by_opportunity.get(key, [])
        return [self._events[i] for i in indices]

    def get_by_type(self, event_type: str) -> list[CanonicalEvent]:
        """Get all events of a specific type."""
        indices = self._index_by_type.get(event_type, [])
        return [self._events[i] for i in indices]

    def query(
        self,
        category: EventCategory = None,
        event_type: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        opportunity_id: UUID = None,
        actor_type: str = None,
        limit: int = 100
    ) -> list[CanonicalEvent]:
        """Query events with filters."""
        results = []

        for event in reversed(self._events):  # Most recent first
            if len(results) >= limit:
                break

            if category and event.category != category:
                continue
            if event_type and event.event_type != event_type:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if opportunity_id and event.opportunity_id != opportunity_id:
                continue
            if actor_type and event.actor_type != actor_type:
                continue

            results.append(event)

        return results

    def get_causation_chain(self, event_id: UUID) -> list[CanonicalEvent]:
        """Trace back through causation chain."""
        chain = []
        current_id = event_id

        while current_id:
            event = next((e for e in self._events if e.id == current_id), None)
            if event:
                chain.append(event)
                current_id = event.causation_id
            else:
                break

        return list(reversed(chain))

    def export_jsonl(self, filepath: str) -> int:
        """Export events to JSON Lines format."""
        count = 0
        with open(filepath, 'w') as f:
            for event in self._events:
                f.write(json.dumps(event.to_dict()) + '\n')
                count += 1
        return count

    def __len__(self) -> int:
        return len(self._events)


class EventBuilder:
    """
    Fluent builder for creating canonical events.

    Simplifies event creation with sensible defaults
    and validation.
    """

    def __init__(self):
        self._event = CanonicalEvent()

    def signal(self, event_type: str) -> "EventBuilder":
        """Create a signal event."""
        self._event.category = EventCategory.SIGNAL
        self._event.event_type = event_type
        return self

    def inference(self, event_type: str) -> "EventBuilder":
        """Create an inference event."""
        self._event.category = EventCategory.INFERENCE
        self._event.event_type = event_type
        return self

    def action(self, event_type: str) -> "EventBuilder":
        """Create an action event."""
        self._event.category = EventCategory.ACTION
        self._event.event_type = event_type
        return self

    def outcome(self, event_type: str) -> "EventBuilder":
        """Create an outcome event."""
        self._event.category = EventCategory.OUTCOME
        self._event.event_type = event_type
        return self

    def governance(self, event_type: str) -> "EventBuilder":
        """Create a governance event."""
        self._event.category = EventCategory.GOVERNANCE
        self._event.event_type = event_type
        return self

    def with_payload(self, payload: dict) -> "EventBuilder":
        """Set event payload."""
        self._event.payload = payload
        return self

    def with_correlation(self, correlation_id: UUID) -> "EventBuilder":
        """Set correlation ID."""
        self._event.correlation_id = correlation_id
        return self

    def caused_by(self, causation_id: UUID) -> "EventBuilder":
        """Set causation ID."""
        self._event.causation_id = causation_id
        return self

    def for_opportunity(self, opportunity_id: UUID) -> "EventBuilder":
        """Set opportunity context."""
        self._event.opportunity_id = opportunity_id
        return self

    def for_account(self, account_id: UUID) -> "EventBuilder":
        """Set account context."""
        self._event.account_id = account_id
        return self

    def by_user(self, user_id: str) -> "EventBuilder":
        """Set user actor."""
        self._event.actor_type = "user"
        self._event.actor_id = user_id
        return self

    def by_system(self, component: str) -> "EventBuilder":
        """Set system actor."""
        self._event.actor_type = "system"
        self._event.actor_id = component
        return self

    def from_source(self, system: str, version: str = "") -> "EventBuilder":
        """Set source information."""
        self._event.source_system = system
        self._event.source_version = version
        return self

    def with_tags(self, *tags: str) -> "EventBuilder":
        """Add tags."""
        self._event.tags.extend(tags)
        return self

    def with_severity(self, severity: EventSeverity) -> "EventBuilder":
        """Set severity."""
        self._event.severity = severity
        return self

    def build(self) -> CanonicalEvent:
        """Build and return the event."""
        return self._event
