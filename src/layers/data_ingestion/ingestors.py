"""
Signal Ingestors - Multi-channel data ingestion

This module implements ingestors for various customer signal sources.
Each ingestor transforms raw channel data into CognitiveSignals for
processing by the intelligence layer.

Key design principles:
- Channel-agnostic signal representation
- Metadata preservation for audit
- Real-time and batch processing support
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from ...core.entities import Interaction, InteractionChannel
from ...core.cognitive_framework import CognitiveSignal


class SignalIngestor(ABC):
    """
    Abstract base class for signal ingestors.

    Each channel (email, call, meeting, etc.) implements this interface
    to transform raw data into standardized CognitiveSignals.
    """

    @property
    @abstractmethod
    def channel(self) -> InteractionChannel:
        """The channel this ingestor handles."""
        pass

    @abstractmethod
    def ingest(self, raw_data: dict) -> CognitiveSignal:
        """Transform raw channel data into a CognitiveSignal."""
        pass

    @abstractmethod
    def validate(self, raw_data: dict) -> tuple[bool, list[str]]:
        """Validate raw data before ingestion. Returns (is_valid, errors)."""
        pass

    def create_signal(
        self,
        content: Any,
        signal_type: str,
        context: dict = None,
        confidence: float = 1.0
    ) -> CognitiveSignal:
        """Helper to create a standardized CognitiveSignal."""
        return CognitiveSignal(
            source=self.channel.value,
            signal_type=signal_type,
            content=content,
            context=context or {},
            confidence=confidence
        )


class EmailIngestor(SignalIngestor):
    """
    Ingestor for email interactions.

    Extracts:
    - Sender/recipient information
    - Subject and body content
    - Thread context
    - Attachments metadata
    """

    @property
    def channel(self) -> InteractionChannel:
        return InteractionChannel.EMAIL

    def validate(self, raw_data: dict) -> tuple[bool, list[str]]:
        errors = []
        required_fields = ["from", "to", "subject", "body", "timestamp"]

        for field in required_fields:
            if field not in raw_data:
                errors.append(f"Missing required field: {field}")

        return (len(errors) == 0, errors)

    def ingest(self, raw_data: dict) -> CognitiveSignal:
        is_valid, errors = self.validate(raw_data)
        if not is_valid:
            raise ValueError(f"Invalid email data: {errors}")

        # Determine direction
        direction = raw_data.get("direction", "inbound")

        # Extract threading context
        thread_context = {
            "message_id": raw_data.get("message_id"),
            "in_reply_to": raw_data.get("in_reply_to"),
            "thread_id": raw_data.get("thread_id"),
            "references": raw_data.get("references", [])
        }

        content = {
            "from": raw_data["from"],
            "to": raw_data["to"],
            "cc": raw_data.get("cc", []),
            "subject": raw_data["subject"],
            "body": raw_data["body"],
            "attachments": raw_data.get("attachments", []),
            "direction": direction
        }

        context = {
            "thread": thread_context,
            "timestamp": raw_data["timestamp"],
            "opportunity_id": raw_data.get("opportunity_id"),
            "account_id": raw_data.get("account_id"),
            "contact_id": raw_data.get("contact_id")
        }

        return self.create_signal(
            content=content,
            signal_type="email_interaction",
            context=context
        )


class CallTranscriptIngestor(SignalIngestor):
    """
    Ingestor for call transcripts.

    Handles:
    - Full transcript text
    - Speaker diarization
    - Call metadata (duration, participants)
    - Silence/talk ratio

    Call transcripts are a primary source for:
    - Objection extraction
    - Sentiment analysis
    - Stakeholder mapping
    """

    @property
    def channel(self) -> InteractionChannel:
        return InteractionChannel.CALL

    def validate(self, raw_data: dict) -> tuple[bool, list[str]]:
        errors = []
        required_fields = ["transcript", "participants", "start_time", "duration_seconds"]

        for field in required_fields:
            if field not in raw_data:
                errors.append(f"Missing required field: {field}")

        return (len(errors) == 0, errors)

    def ingest(self, raw_data: dict) -> CognitiveSignal:
        is_valid, errors = self.validate(raw_data)
        if not is_valid:
            raise ValueError(f"Invalid call data: {errors}")

        # Process transcript segments if available
        segments = raw_data.get("segments", [])
        speakers = self._extract_speakers(segments, raw_data["participants"])

        content = {
            "transcript": raw_data["transcript"],
            "segments": segments,
            "speakers": speakers,
            "duration_seconds": raw_data["duration_seconds"],
            "participants": raw_data["participants"]
        }

        context = {
            "start_time": raw_data["start_time"],
            "end_time": raw_data.get("end_time"),
            "opportunity_id": raw_data.get("opportunity_id"),
            "account_id": raw_data.get("account_id"),
            "call_type": raw_data.get("call_type", "sales_call"),
            "recording_quality": raw_data.get("quality_score", 1.0)
        }

        # Adjust confidence based on transcript quality
        quality = raw_data.get("quality_score", 1.0)
        confidence = min(1.0, quality)

        return self.create_signal(
            content=content,
            signal_type="call_transcript",
            context=context,
            confidence=confidence
        )

    def _extract_speakers(self, segments: list, participants: list) -> dict:
        """Extract speaker statistics from segments."""
        speaker_stats = {}

        for segment in segments:
            speaker = segment.get("speaker", "unknown")
            duration = segment.get("end", 0) - segment.get("start", 0)

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "total_duration": 0,
                    "segment_count": 0,
                    "participant_info": next(
                        (p for p in participants if p.get("id") == speaker),
                        {}
                    )
                }

            speaker_stats[speaker]["total_duration"] += duration
            speaker_stats[speaker]["segment_count"] += 1

        return speaker_stats


class MeetingIngestor(SignalIngestor):
    """
    Ingestor for meeting data (video conferences, in-person).

    Combines:
    - Transcript (if available)
    - Attendee list and roles
    - Agenda and notes
    - Screen share / presentation content
    """

    @property
    def channel(self) -> InteractionChannel:
        return InteractionChannel.MEETING

    def validate(self, raw_data: dict) -> tuple[bool, list[str]]:
        errors = []
        required_fields = ["title", "attendees", "start_time"]

        for field in required_fields:
            if field not in raw_data:
                errors.append(f"Missing required field: {field}")

        return (len(errors) == 0, errors)

    def ingest(self, raw_data: dict) -> CognitiveSignal:
        is_valid, errors = self.validate(raw_data)
        if not is_valid:
            raise ValueError(f"Invalid meeting data: {errors}")

        content = {
            "title": raw_data["title"],
            "attendees": raw_data["attendees"],
            "agenda": raw_data.get("agenda", ""),
            "notes": raw_data.get("notes", ""),
            "transcript": raw_data.get("transcript"),
            "action_items": raw_data.get("action_items", []),
            "duration_minutes": raw_data.get("duration_minutes")
        }

        context = {
            "start_time": raw_data["start_time"],
            "end_time": raw_data.get("end_time"),
            "opportunity_id": raw_data.get("opportunity_id"),
            "account_id": raw_data.get("account_id"),
            "meeting_type": raw_data.get("meeting_type", "general"),
            "location": raw_data.get("location"),
            "is_recurring": raw_data.get("is_recurring", False)
        }

        # Confidence based on data completeness
        has_transcript = raw_data.get("transcript") is not None
        has_notes = bool(raw_data.get("notes"))
        confidence = 0.5 + (0.3 if has_transcript else 0) + (0.2 if has_notes else 0)

        return self.create_signal(
            content=content,
            signal_type="meeting_interaction",
            context=context,
            confidence=confidence
        )


class CRMEventIngestor(SignalIngestor):
    """
    Ingestor for CRM system events.

    Captures:
    - Stage changes
    - Field updates
    - Record creation/deletion
    - User actions

    These events are crucial for:
    - Outcome signals (closed won/lost)
    - Stage conversion tracking
    - Activity pattern analysis
    """

    @property
    def channel(self) -> InteractionChannel:
        # CRM events don't map to a customer channel directly
        # Using EMAIL as placeholder; in production would extend enum
        return InteractionChannel.EMAIL

    def validate(self, raw_data: dict) -> tuple[bool, list[str]]:
        errors = []
        required_fields = ["event_type", "entity_type", "entity_id", "timestamp"]

        for field in required_fields:
            if field not in raw_data:
                errors.append(f"Missing required field: {field}")

        return (len(errors) == 0, errors)

    def ingest(self, raw_data: dict) -> CognitiveSignal:
        is_valid, errors = self.validate(raw_data)
        if not is_valid:
            raise ValueError(f"Invalid CRM event data: {errors}")

        content = {
            "event_type": raw_data["event_type"],
            "entity_type": raw_data["entity_type"],
            "entity_id": raw_data["entity_id"],
            "changes": raw_data.get("changes", {}),
            "previous_values": raw_data.get("previous_values", {}),
            "new_values": raw_data.get("new_values", {})
        }

        context = {
            "timestamp": raw_data["timestamp"],
            "user_id": raw_data.get("user_id"),
            "opportunity_id": raw_data.get("opportunity_id"),
            "account_id": raw_data.get("account_id"),
            "source_system": raw_data.get("source_system", "crm")
        }

        # Determine if this is a significant signal
        significant_events = ["stage_change", "close", "amount_change", "owner_change"]
        is_significant = raw_data["event_type"] in significant_events

        signal = self.create_signal(
            content=content,
            signal_type=f"crm_{raw_data['event_type']}",
            context=context
        )
        signal.requires_attention = is_significant

        return signal


@dataclass
class IngestorRegistry:
    """
    Registry of available ingestors.

    Provides a central point for:
    - Registering new ingestors
    - Routing signals to appropriate ingestors
    - Managing ingestor lifecycle
    """
    _ingestors: dict = field(default_factory=dict)

    def register(self, ingestor: SignalIngestor) -> None:
        """Register an ingestor for a channel."""
        self._ingestors[ingestor.channel] = ingestor

    def get(self, channel: InteractionChannel) -> Optional[SignalIngestor]:
        """Get ingestor for a channel."""
        return self._ingestors.get(channel)

    def ingest(self, channel: InteractionChannel, raw_data: dict) -> CognitiveSignal:
        """Route data to appropriate ingestor."""
        ingestor = self.get(channel)
        if not ingestor:
            raise ValueError(f"No ingestor registered for channel: {channel}")
        return ingestor.ingest(raw_data)

    @classmethod
    def create_default(cls) -> "IngestorRegistry":
        """Create registry with all standard ingestors."""
        registry = cls()
        registry.register(EmailIngestor())
        registry.register(CallTranscriptIngestor())
        registry.register(MeetingIngestor())
        return registry
