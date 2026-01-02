"""
Layer 1: Data Ingestion and Representation

From Section 4.1 of the paper:

Structured sources:
- CRM objects (accounts, contacts, opportunities)
- CPQ, contracts, service tickets

Unstructured sources:
- Call transcripts, emails, chat logs
- Meeting notes, documents

Representation:
- Embeddings for semantic retrieval
- Canonical event schema for audit logs

Identity resolution:
- Entity linking across systems
"""

from .ingestors import (
    SignalIngestor,
    EmailIngestor,
    CallTranscriptIngestor,
    MeetingIngestor,
    CRMEventIngestor
)
from .embeddings import EmbeddingGenerator
from .identity_resolution import IdentityResolver
from .event_schema import CanonicalEvent, EventStore

__all__ = [
    "SignalIngestor",
    "EmailIngestor",
    "CallTranscriptIngestor",
    "MeetingIngestor",
    "CRMEventIngestor",
    "EmbeddingGenerator",
    "IdentityResolver",
    "CanonicalEvent",
    "EventStore"
]
