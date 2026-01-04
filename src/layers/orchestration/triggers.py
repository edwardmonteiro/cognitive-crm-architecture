"""
Trigger Engine - Event and Schedule-driven Automation

From Section 4.3:
- Event-driven (meeting ended, email received)
- Schedule-driven (daily pipeline sweep)

Triggers initiate playbook execution based on:
- CRM events (stage changes, updates)
- Communication events (calls, emails, meetings)
- Time-based schedules (daily reviews, SLA checks)
- Threshold conditions (days inactive, risk scores)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from uuid import UUID, uuid4
import re


class TriggerType(Enum):
    """Types of triggers."""
    EVENT = "event"
    SCHEDULE = "schedule"
    CONDITION = "condition"
    COMPOSITE = "composite"


class TriggerStatus(Enum):
    """Status of a trigger."""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    FIRED = "fired"


@dataclass
class TriggerEvent:
    """An event that can trigger playbooks."""
    id: UUID = field(default_factory=uuid4)
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Event data
    payload: dict = field(default_factory=dict)

    # Context
    opportunity_id: Optional[UUID] = None
    account_id: Optional[UUID] = None
    contact_id: Optional[UUID] = None
    user_id: Optional[str] = None

    # Source
    source_system: str = ""
    correlation_id: Optional[UUID] = None


@dataclass
class TriggerResult:
    """Result of trigger evaluation."""
    triggered: bool = False
    trigger_id: UUID = field(default_factory=uuid4)
    event: Optional[TriggerEvent] = None
    playbooks_to_run: list = field(default_factory=list)
    context: dict = field(default_factory=dict)
    reason: str = ""


class Trigger(ABC):
    """
    Abstract base class for triggers.

    Triggers evaluate conditions and initiate playbook execution.
    """

    def __init__(self, name: str, playbook_names: list[str] = None):
        self.id = uuid4()
        self.name = name
        self.playbook_names = playbook_names or []
        self.status = TriggerStatus.ACTIVE
        self.created_at = datetime.now()
        self.last_fired_at: Optional[datetime] = None
        self.fire_count = 0

    @property
    @abstractmethod
    def trigger_type(self) -> TriggerType:
        """Type of this trigger."""
        pass

    @abstractmethod
    def evaluate(self, event: TriggerEvent) -> TriggerResult:
        """Evaluate if trigger should fire."""
        pass

    def is_active(self) -> bool:
        """Check if trigger is active."""
        return self.status == TriggerStatus.ACTIVE

    def fire(self, event: TriggerEvent) -> TriggerResult:
        """Fire the trigger and update state."""
        if not self.is_active():
            return TriggerResult(
                triggered=False,
                reason="Trigger is not active"
            )

        result = self.evaluate(event)

        if result.triggered:
            self.last_fired_at = datetime.now()
            self.fire_count += 1
            result.playbooks_to_run = self.playbook_names

        return result


class EventTrigger(Trigger):
    """
    Event-driven trigger.

    Fires when specific events occur:
    - call_ended
    - email_received
    - meeting_ended
    - stage_change
    - amount_change
    """

    def __init__(
        self,
        name: str,
        event_types: list[str],
        playbook_names: list[str] = None,
        filter_fn: Callable[[TriggerEvent], bool] = None
    ):
        super().__init__(name, playbook_names)
        self.event_types = event_types
        self.filter_fn = filter_fn

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.EVENT

    def evaluate(self, event: TriggerEvent) -> TriggerResult:
        # Check event type match
        if event.event_type not in self.event_types:
            return TriggerResult(
                triggered=False,
                reason=f"Event type {event.event_type} not in {self.event_types}"
            )

        # Apply custom filter if defined
        if self.filter_fn and not self.filter_fn(event):
            return TriggerResult(
                triggered=False,
                reason="Custom filter returned False"
            )

        return TriggerResult(
            triggered=True,
            trigger_id=self.id,
            event=event,
            context={
                "trigger_name": self.name,
                "event_type": event.event_type,
                **event.payload
            },
            reason=f"Event {event.event_type} matched trigger {self.name}"
        )


class ScheduleTrigger(Trigger):
    """
    Schedule-driven trigger.

    Fires on a schedule:
    - Daily pipeline sweep
    - Weekly forecast review
    - Hourly SLA checks
    """

    def __init__(
        self,
        name: str,
        schedule: str,  # cron-like or simple: "daily", "hourly", "weekly"
        playbook_names: list[str] = None,
        context_builder: Callable[[], dict] = None
    ):
        super().__init__(name, playbook_names)
        self.schedule = schedule
        self.context_builder = context_builder
        self._next_run: Optional[datetime] = None
        self._calculate_next_run()

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.SCHEDULE

    def _calculate_next_run(self) -> None:
        """Calculate next run time based on schedule."""
        now = datetime.now()

        if self.schedule == "hourly":
            self._next_run = now.replace(minute=0, second=0) + timedelta(hours=1)
        elif self.schedule == "daily":
            self._next_run = now.replace(hour=8, minute=0, second=0)
            if self._next_run <= now:
                self._next_run += timedelta(days=1)
        elif self.schedule == "weekly":
            days_ahead = 0 - now.weekday()  # Monday
            if days_ahead <= 0:
                days_ahead += 7
            self._next_run = now.replace(hour=8, minute=0, second=0) + timedelta(days=days_ahead)
        else:
            # Default to daily
            self._next_run = now + timedelta(days=1)

    def evaluate(self, event: TriggerEvent) -> TriggerResult:
        now = datetime.now()

        if self._next_run and now >= self._next_run:
            # Build context
            context = {}
            if self.context_builder:
                context = self.context_builder()

            # Calculate next run
            self._calculate_next_run()

            return TriggerResult(
                triggered=True,
                trigger_id=self.id,
                event=event,
                context={
                    "trigger_name": self.name,
                    "schedule": self.schedule,
                    **context
                },
                reason=f"Schedule {self.schedule} reached"
            )

        return TriggerResult(
            triggered=False,
            reason=f"Next run at {self._next_run}"
        )

    def is_due(self) -> bool:
        """Check if trigger is due to fire."""
        return self._next_run and datetime.now() >= self._next_run


class ConditionTrigger(Trigger):
    """
    Condition-based trigger.

    Fires when conditions are met:
    - Days since last contact > threshold
    - Risk score > threshold
    - Deal value > threshold
    """

    def __init__(
        self,
        name: str,
        condition: Callable[[dict], bool],
        playbook_names: list[str] = None,
        description: str = ""
    ):
        super().__init__(name, playbook_names)
        self.condition = condition
        self.description = description

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.CONDITION

    def evaluate(self, event: TriggerEvent) -> TriggerResult:
        context = event.payload

        if self.condition(context):
            return TriggerResult(
                triggered=True,
                trigger_id=self.id,
                event=event,
                context={
                    "trigger_name": self.name,
                    "condition": self.description,
                    **context
                },
                reason=f"Condition met: {self.description}"
            )

        return TriggerResult(
            triggered=False,
            reason=f"Condition not met: {self.description}"
        )


class CompositeTrigger(Trigger):
    """
    Composite trigger combining multiple conditions.

    Supports AND/OR logic for complex triggering rules.
    """

    def __init__(
        self,
        name: str,
        triggers: list[Trigger],
        logic: str = "AND",  # AND, OR
        playbook_names: list[str] = None
    ):
        super().__init__(name, playbook_names)
        self.triggers = triggers
        self.logic = logic.upper()

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.COMPOSITE

    def evaluate(self, event: TriggerEvent) -> TriggerResult:
        results = [t.evaluate(event) for t in self.triggers]

        if self.logic == "AND":
            all_triggered = all(r.triggered for r in results)
            if all_triggered:
                return TriggerResult(
                    triggered=True,
                    trigger_id=self.id,
                    event=event,
                    context={
                        "trigger_name": self.name,
                        "logic": "AND",
                        "sub_results": [r.reason for r in results]
                    },
                    reason=f"All {len(self.triggers)} conditions met"
                )
            return TriggerResult(
                triggered=False,
                reason=f"Not all conditions met: {[r.reason for r in results if not r.triggered]}"
            )

        elif self.logic == "OR":
            any_triggered = any(r.triggered for r in results)
            if any_triggered:
                triggered_results = [r for r in results if r.triggered]
                return TriggerResult(
                    triggered=True,
                    trigger_id=self.id,
                    event=event,
                    context={
                        "trigger_name": self.name,
                        "logic": "OR",
                        "triggered_by": [r.reason for r in triggered_results]
                    },
                    reason=f"{len(triggered_results)} of {len(self.triggers)} conditions met"
                )
            return TriggerResult(
                triggered=False,
                reason="No conditions met"
            )

        return TriggerResult(triggered=False, reason=f"Unknown logic: {self.logic}")


class TriggerEngine:
    """
    Engine for managing and evaluating triggers.

    Responsibilities:
    - Register triggers
    - Process incoming events
    - Match events to triggers
    - Fire playbooks
    """

    def __init__(self):
        self._triggers: dict[str, Trigger] = {}
        self._event_triggers: dict[str, list[str]] = {}  # event_type -> trigger_names
        self._schedule_triggers: list[str] = []

    def register(self, trigger: Trigger) -> None:
        """Register a trigger."""
        self._triggers[trigger.name] = trigger

        if trigger.trigger_type == TriggerType.EVENT and isinstance(trigger, EventTrigger):
            for event_type in trigger.event_types:
                if event_type not in self._event_triggers:
                    self._event_triggers[event_type] = []
                self._event_triggers[event_type].append(trigger.name)

        elif trigger.trigger_type == TriggerType.SCHEDULE:
            self._schedule_triggers.append(trigger.name)

    def process_event(self, event: TriggerEvent) -> list[TriggerResult]:
        """Process an event through all applicable triggers."""
        results = []

        # Get triggers for this event type
        trigger_names = self._event_triggers.get(event.event_type, [])

        for name in trigger_names:
            trigger = self._triggers.get(name)
            if trigger and trigger.is_active():
                result = trigger.fire(event)
                if result.triggered:
                    results.append(result)

        return results

    def check_schedules(self) -> list[TriggerResult]:
        """Check all schedule triggers."""
        results = []

        # Create a dummy event for schedule evaluation
        event = TriggerEvent(
            event_type="scheduled_check",
            timestamp=datetime.now()
        )

        for name in self._schedule_triggers:
            trigger = self._triggers.get(name)
            if trigger and trigger.is_active():
                if isinstance(trigger, ScheduleTrigger) and trigger.is_due():
                    result = trigger.fire(event)
                    if result.triggered:
                        results.append(result)

        return results

    def get_trigger(self, name: str) -> Optional[Trigger]:
        """Get trigger by name."""
        return self._triggers.get(name)

    def pause_trigger(self, name: str) -> bool:
        """Pause a trigger."""
        trigger = self._triggers.get(name)
        if trigger:
            trigger.status = TriggerStatus.PAUSED
            return True
        return False

    def resume_trigger(self, name: str) -> bool:
        """Resume a paused trigger."""
        trigger = self._triggers.get(name)
        if trigger:
            trigger.status = TriggerStatus.ACTIVE
            return True
        return False

    @classmethod
    def create_default(cls) -> "TriggerEngine":
        """Create engine with default triggers."""
        engine = cls()

        # Post-call trigger
        engine.register(EventTrigger(
            name="post_call",
            event_types=["call_ended", "transcript_ready"],
            playbook_names=["post_call_intelligence"]
        ))

        # Email received trigger
        engine.register(EventTrigger(
            name="email_received",
            event_types=["email_received"],
            playbook_names=["follow_up_drafting"],
            filter_fn=lambda e: e.payload.get("direction") == "inbound"
        ))

        # Meeting ended trigger
        engine.register(EventTrigger(
            name="meeting_ended",
            event_types=["meeting_ended"],
            playbook_names=["post_call_intelligence", "follow_up_drafting"]
        ))

        # Daily pipeline sweep
        engine.register(ScheduleTrigger(
            name="daily_pipeline_sweep",
            schedule="daily",
            playbook_names=["risk_alert", "next_best_action"]
        ))

        # Stale deal trigger
        engine.register(ConditionTrigger(
            name="stale_deal",
            condition=lambda ctx: ctx.get("days_since_activity", 0) > 14,
            playbook_names=["risk_alert"],
            description="No activity for 14+ days"
        ))

        return engine
