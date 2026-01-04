"""
Alert Engine

Monitors metrics and generates alerts when thresholds are breached.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from uuid import UUID, uuid4


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    metric_name: str = ""
    description: str = ""

    # Threshold
    condition: str = "gt"  # gt, lt, gte, lte, eq
    threshold: float = 0.0

    # Severity and notification
    severity: AlertSeverity = AlertSeverity.WARNING
    notify_channels: list = field(default_factory=list)

    # Cooldown
    cooldown_minutes: int = 60
    last_triggered: Optional[datetime] = None

    # Active
    enabled: bool = True


@dataclass
class Alert:
    """An active alert."""
    id: UUID = field(default_factory=uuid4)
    rule_id: UUID = field(default_factory=uuid4)
    rule_name: str = ""
    metric_name: str = ""

    # Details
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""
    current_value: float = 0.0
    threshold: float = 0.0

    # Status
    status: AlertStatus = AlertStatus.ACTIVE
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    # Context
    context: dict = field(default_factory=dict)


class AlertEngine:
    """
    Engine for monitoring metrics and generating alerts.

    Monitors key metrics and alerts when:
    - Execution latency exceeds threshold
    - Approval rates drop
    - Error rates spike
    """

    def __init__(self):
        self._rules: dict[str, AlertRule] = {}
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []

        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Create default alert rules."""
        default_rules = [
            AlertRule(
                name="High Execution Latency",
                metric_name="execution_latency",
                description="Time between touchpoints exceeds threshold",
                condition="gt",
                threshold=60.0,  # hours
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                name="Low NBA Acceptance",
                metric_name="nba_acceptance",
                description="NBA acceptance rate below target",
                condition="lt",
                threshold=50.0,  # percent
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                name="High Admin Effort",
                metric_name="admin_effort",
                description="Administrative time exceeds threshold",
                condition="gt",
                threshold=16.0,  # hours/week
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                name="Low Conversion Rate",
                metric_name="conversion_rate",
                description="Stage conversion rate critically low",
                condition="lt",
                threshold=25.0,  # percent
                severity=AlertSeverity.CRITICAL
            ),
            AlertRule(
                name="High Forecast Error",
                metric_name="forecast_mae",
                description="Forecast accuracy degraded",
                condition="gt",
                threshold=25.0,  # percent MAE
                severity=AlertSeverity.WARNING
            )
        ]

        for rule in default_rules:
            self.add_rule(rule)

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._rules[str(rule.id)] = rule

    def remove_rule(self, rule_id: UUID) -> bool:
        """Remove an alert rule."""
        key = str(rule_id)
        if key in self._rules:
            del self._rules[key]
            return True
        return False

    def evaluate(self, metrics: dict) -> list[Alert]:
        """Evaluate all rules against current metrics."""
        new_alerts = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            metric_value = metrics.get(rule.metric_name)
            if metric_value is None:
                continue

            # Get numeric value
            if hasattr(metric_value, 'value'):
                value = metric_value.value
            else:
                value = metric_value

            # Check condition
            triggered = self._check_condition(value, rule.condition, rule.threshold)

            if triggered:
                # Check cooldown
                if rule.last_triggered:
                    cooldown_delta = datetime.now() - rule.last_triggered
                    if cooldown_delta.total_seconds() < rule.cooldown_minutes * 60:
                        continue

                # Create alert
                alert = self._create_alert(rule, value)
                new_alerts.append(alert)

                rule.last_triggered = datetime.now()
                self._active_alerts[str(alert.id)] = alert

        return new_alerts

    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if condition is met."""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return abs(value - threshold) < 0.001
        return False

    def _create_alert(self, rule: AlertRule, current_value: float) -> Alert:
        """Create an alert from a triggered rule."""
        condition_text = {
            "gt": "exceeded",
            "lt": "fell below",
            "gte": "reached or exceeded",
            "lte": "reached or fell below",
            "eq": "equals"
        }

        message = (
            f"{rule.metric_name} {condition_text.get(rule.condition, 'breached')} "
            f"threshold: {current_value:.2f} (threshold: {rule.threshold:.2f})"
        )

        return Alert(
            rule_id=rule.id,
            rule_name=rule.name,
            metric_name=rule.metric_name,
            severity=rule.severity,
            message=message,
            current_value=current_value,
            threshold=rule.threshold
        )

    def acknowledge(self, alert_id: UUID, acknowledged_by: str = None) -> bool:
        """Acknowledge an alert."""
        key = str(alert_id)
        if key in self._active_alerts:
            alert = self._active_alerts[key]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.context["acknowledged_by"] = acknowledged_by
            return True
        return False

    def resolve(self, alert_id: UUID, resolution_notes: str = None) -> bool:
        """Resolve an alert."""
        key = str(alert_id)
        if key in self._active_alerts:
            alert = self._active_alerts[key]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            if resolution_notes:
                alert.context["resolution_notes"] = resolution_notes

            # Move to history
            self._alert_history.append(alert)
            del self._active_alerts[key]
            return True
        return False

    def get_active_alerts(
        self,
        severity: AlertSeverity = None
    ) -> list[Alert]:
        """Get all active alerts."""
        alerts = list(self._active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        # Sort by severity (critical first) then by time
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.INFO: 2
        }

        alerts.sort(key=lambda a: (severity_order[a.severity], a.triggered_at))

        return alerts

    def get_alert_summary(self) -> dict:
        """Get summary of alert status."""
        active = list(self._active_alerts.values())

        return {
            "total_active": len(active),
            "by_severity": {
                "critical": sum(1 for a in active if a.severity == AlertSeverity.CRITICAL),
                "warning": sum(1 for a in active if a.severity == AlertSeverity.WARNING),
                "info": sum(1 for a in active if a.severity == AlertSeverity.INFO)
            },
            "by_metric": {
                metric: sum(1 for a in active if a.metric_name == metric)
                for metric in set(a.metric_name for a in active)
            },
            "oldest_unacknowledged": min(
                (a.triggered_at for a in active if a.status == AlertStatus.ACTIVE),
                default=None
            )
        }
