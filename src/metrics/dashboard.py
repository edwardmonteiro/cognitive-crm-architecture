"""
Dashboard Data Generation

Generates data for visualization and monitoring dashboards.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

from .calculator import MetricsCalculator, MetricValue


@dataclass
class MetricTrend:
    """Trend data for a metric over time."""
    metric_name: str
    values: list = field(default_factory=list)  # List of (timestamp, value) tuples
    trend_direction: str = "stable"  # improving, declining, stable
    trend_strength: float = 0.0  # -1 to 1


@dataclass
class DashboardData:
    """Complete dashboard data snapshot."""
    generated_at: datetime = field(default_factory=datetime.now)

    # Current metrics
    current_metrics: dict = field(default_factory=dict)

    # Trends
    trends: dict = field(default_factory=dict)

    # Comparisons
    vs_baseline: dict = field(default_factory=dict)
    vs_target: dict = field(default_factory=dict)

    # Alerts
    active_alerts: list = field(default_factory=list)

    # Summary
    overall_health: str = "good"  # good, warning, critical
    improvement_areas: list = field(default_factory=list)


class DashboardGenerator:
    """Generates dashboard data from metrics."""

    def __init__(self, calculator: MetricsCalculator = None):
        self._calculator = calculator or MetricsCalculator()
        self._history: list[dict] = []  # Historical metric snapshots

    def record_snapshot(self, metrics: dict[str, MetricValue]) -> None:
        """Record a metrics snapshot for trend analysis."""
        self._history.append({
            "timestamp": datetime.now(),
            "metrics": {k: v.value for k, v in metrics.items()}
        })

        # Keep last 90 days
        cutoff = datetime.now() - timedelta(days=90)
        self._history = [h for h in self._history if h["timestamp"] >= cutoff]

    def generate_dashboard(
        self,
        current_metrics: dict[str, MetricValue]
    ) -> DashboardData:
        """Generate complete dashboard data."""
        dashboard = DashboardData()
        dashboard.generated_at = datetime.now()

        # Current metrics
        for name, metric in current_metrics.items():
            dashboard.current_metrics[name] = {
                "value": metric.value,
                "unit": metric.unit,
                "sample_size": metric.sample_size
            }

            # Compare to baseline
            comparison = self._calculator.compare_to_baseline(metric)
            dashboard.vs_baseline[name] = comparison

        # Calculate trends
        for name in current_metrics:
            trend = self._calculate_trend(name)
            dashboard.trends[name] = trend

        # Determine overall health
        dashboard.overall_health = self._calculate_health(dashboard.vs_baseline)

        # Identify improvement areas
        dashboard.improvement_areas = self._identify_improvements(dashboard.vs_baseline)

        return dashboard

    def _calculate_trend(self, metric_name: str) -> MetricTrend:
        """Calculate trend for a metric."""
        values = []

        for snapshot in self._history[-30:]:  # Last 30 snapshots
            if metric_name in snapshot["metrics"]:
                values.append((
                    snapshot["timestamp"],
                    snapshot["metrics"][metric_name]
                ))

        if len(values) < 2:
            return MetricTrend(
                metric_name=metric_name,
                values=values,
                trend_direction="stable",
                trend_strength=0.0
            )

        # Simple linear regression for trend
        n = len(values)
        x_mean = n / 2
        y_mean = sum(v[1] for v in values) / n

        numerator = sum((i - x_mean) * (v[1] - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Normalize slope
        if y_mean != 0:
            normalized_slope = slope / y_mean
        else:
            normalized_slope = 0

        # Determine direction
        if normalized_slope > 0.05:
            direction = "improving" if self._is_higher_better(metric_name) else "declining"
        elif normalized_slope < -0.05:
            direction = "declining" if self._is_higher_better(metric_name) else "improving"
        else:
            direction = "stable"

        return MetricTrend(
            metric_name=metric_name,
            values=values,
            trend_direction=direction,
            trend_strength=min(1.0, abs(normalized_slope) * 10)
        )

    def _is_higher_better(self, metric_name: str) -> bool:
        """Check if higher values are better for this metric."""
        definition = self._calculator.get_metric_definition(metric_name)
        return definition and definition.target_direction == "higher"

    def _calculate_health(self, comparisons: dict) -> str:
        """Calculate overall health status."""
        improvements = sum(1 for c in comparisons.values() if c.get("is_improvement"))
        total = len(comparisons)

        if total == 0:
            return "unknown"

        ratio = improvements / total

        if ratio >= 0.8:
            return "excellent"
        elif ratio >= 0.6:
            return "good"
        elif ratio >= 0.4:
            return "warning"
        else:
            return "critical"

    def _identify_improvements(self, comparisons: dict) -> list:
        """Identify areas needing improvement."""
        improvements = []

        for name, comparison in comparisons.items():
            if not comparison.get("is_improvement") and not comparison.get("target_met"):
                improvements.append({
                    "metric": name,
                    "current": comparison.get("current_value"),
                    "target": comparison.get("target_value"),
                    "gap": abs(comparison.get("delta", 0))
                })

        # Sort by gap size
        improvements.sort(key=lambda x: x["gap"], reverse=True)

        return improvements

    def format_summary(self, dashboard: DashboardData) -> str:
        """Format dashboard as text summary."""
        lines = [
            f"Dashboard Summary ({dashboard.generated_at.strftime('%Y-%m-%d %H:%M')})",
            f"Overall Health: {dashboard.overall_health.upper()}",
            "",
            "Current Metrics:"
        ]

        for name, data in dashboard.current_metrics.items():
            comparison = dashboard.vs_baseline.get(name, {})
            delta = comparison.get("delta_percent", 0)
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"

            lines.append(
                f"  {name}: {data['value']:.1f} {data['unit']} "
                f"({direction} {abs(delta):.1f}% vs baseline)"
            )

        if dashboard.improvement_areas:
            lines.extend(["", "Areas for Improvement:"])
            for area in dashboard.improvement_areas[:3]:
                lines.append(
                    f"  - {area['metric']}: {area['current']:.1f} → {area['target']:.1f}"
                )

        return "\n".join(lines)
