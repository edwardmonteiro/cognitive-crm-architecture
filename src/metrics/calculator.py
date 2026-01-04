"""
Metrics Calculator

Calculates the key metrics defined in Section 5.2:
- Execution latency
- Administrative effort
- Conversion consistency
- Forecast variance
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import UUID
import statistics


class MetricCategory(Enum):
    """Categories of metrics."""
    EXECUTION = "execution"
    EFFICIENCY = "efficiency"
    EFFECTIVENESS = "effectiveness"
    GOVERNANCE = "governance"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    category: MetricCategory
    description: str
    unit: str
    target_direction: str = "lower"  # lower, higher
    baseline_value: float = 0.0
    target_value: float = 0.0


@dataclass
class MetricValue:
    """A calculated metric value."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    sample_size: int = 0
    breakdown: dict = field(default_factory=dict)


class MetricsCalculator:
    """
    Calculates key performance metrics for the cognitive CRM.

    Implements the metrics defined in Section 5.2.
    """

    def __init__(self):
        self._metrics = self._define_metrics()

    def _define_metrics(self) -> dict[str, MetricDefinition]:
        """Define all metrics with their baselines and targets."""
        return {
            "execution_latency": MetricDefinition(
                name="Time Between Touchpoints",
                category=MetricCategory.EXECUTION,
                description="Average hours between customer interactions",
                unit="hours",
                target_direction="lower",
                baseline_value=52.4,
                target_value=39.6
            ),
            "admin_effort": MetricDefinition(
                name="Administrative Time",
                category=MetricCategory.EFFICIENCY,
                description="Hours per week spent on CRM updates",
                unit="hours/week",
                target_direction="lower",
                baseline_value=14.8,
                target_value=10.1
            ),
            "sales_cycle": MetricDefinition(
                name="Average Sales Cycle",
                category=MetricCategory.EFFECTIVENESS,
                description="Days from opportunity creation to close",
                unit="days",
                target_direction="lower",
                baseline_value=94.2,
                target_value=74.1
            ),
            "conversion_rate": MetricDefinition(
                name="Stage-to-Stage Conversion",
                category=MetricCategory.EFFECTIVENESS,
                description="Percentage of deals advancing between stages",
                unit="%",
                target_direction="higher",
                baseline_value=31.5,
                target_value=37.9
            ),
            "forecast_mae": MetricDefinition(
                name="Forecast Error",
                category=MetricCategory.EFFECTIVENESS,
                description="Mean absolute error of revenue forecasts",
                unit="%",
                target_direction="lower",
                baseline_value=18.0,
                target_value=11.0
            ),
            "nba_acceptance": MetricDefinition(
                name="NBA Acceptance Rate",
                category=MetricCategory.GOVERNANCE,
                description="Percentage of recommendations accepted by sellers",
                unit="%",
                target_direction="higher",
                baseline_value=0,
                target_value=68.2
            ),
            "approval_rate": MetricDefinition(
                name="Approval Rate",
                category=MetricCategory.GOVERNANCE,
                description="Percentage of actions approved without modification",
                unit="%",
                target_direction="higher",
                baseline_value=0,
                target_value=80.0
            )
        }

    def get_metric_definition(self, metric_name: str) -> Optional[MetricDefinition]:
        """Get definition for a metric."""
        return self._metrics.get(metric_name)

    def calculate_execution_latency(
        self,
        interactions: list,
        opportunity_ids: list = None
    ) -> MetricValue:
        """
        Calculate average time between touchpoints.

        From paper: "time between touchpoints (sense→act delay)"
        """
        intervals = []

        # Group interactions by opportunity
        by_opportunity = {}
        for interaction in interactions:
            opp_id = str(interaction.get("opportunity_id", "unknown"))
            if opportunity_ids and opp_id not in [str(o) for o in opportunity_ids]:
                continue

            if opp_id not in by_opportunity:
                by_opportunity[opp_id] = []
            by_opportunity[opp_id].append(interaction)

        # Calculate intervals
        for opp_id, opp_interactions in by_opportunity.items():
            sorted_interactions = sorted(
                opp_interactions,
                key=lambda x: x.get("timestamp", datetime.min)
            )

            for i in range(1, len(sorted_interactions)):
                t1 = sorted_interactions[i-1].get("timestamp")
                t2 = sorted_interactions[i].get("timestamp")

                if t1 and t2:
                    if isinstance(t1, str):
                        t1 = datetime.fromisoformat(t1)
                    if isinstance(t2, str):
                        t2 = datetime.fromisoformat(t2)

                    delta_hours = (t2 - t1).total_seconds() / 3600
                    if 0 < delta_hours < 720:  # Filter outliers (< 30 days)
                        intervals.append(delta_hours)

        avg_interval = statistics.mean(intervals) if intervals else 52.4

        return MetricValue(
            metric_name="execution_latency",
            value=avg_interval,
            unit="hours",
            sample_size=len(intervals),
            breakdown={
                "min": min(intervals) if intervals else 0,
                "max": max(intervals) if intervals else 0,
                "median": statistics.median(intervals) if intervals else 0
            }
        )

    def calculate_admin_effort(
        self,
        seller_activities: list,
        period_weeks: int = 1
    ) -> MetricValue:
        """
        Calculate administrative hours per seller per week.

        From paper: "hours/week spent updating CRM (coordination and cognitive load proxy)"
        """
        seller_hours = {}

        for activity in seller_activities:
            seller_id = str(activity.get("seller_id", "unknown"))
            duration_minutes = activity.get("duration_minutes", 0)

            if seller_id not in seller_hours:
                seller_hours[seller_id] = 0

            seller_hours[seller_id] += duration_minutes / 60

        if not seller_hours:
            return MetricValue(
                metric_name="admin_effort",
                value=14.8,
                unit="hours/week",
                sample_size=0
            )

        # Average per seller per week
        avg_hours = sum(seller_hours.values()) / len(seller_hours) / period_weeks

        return MetricValue(
            metric_name="admin_effort",
            value=avg_hours,
            unit="hours/week",
            sample_size=len(seller_hours),
            breakdown={
                "by_seller": seller_hours,
                "total_hours": sum(seller_hours.values())
            }
        )

    def calculate_conversion_rate(
        self,
        stage_transitions: list,
        from_stage: str = None,
        to_stage: str = None
    ) -> MetricValue:
        """
        Calculate stage-to-stage conversion rate.

        From paper: "stage-to-stage conversion and variance across sellers"
        """
        advances = 0
        total = 0

        for transition in stage_transitions:
            if from_stage and transition.get("from_stage") != from_stage:
                continue
            if to_stage and transition.get("to_stage") != to_stage:
                continue

            total += 1
            if transition.get("is_advance", True):
                advances += 1

        rate = (advances / total * 100) if total > 0 else 31.5

        return MetricValue(
            metric_name="conversion_rate",
            value=rate,
            unit="%",
            sample_size=total,
            breakdown={
                "advances": advances,
                "total": total
            }
        )

    def calculate_forecast_mae(
        self,
        forecasts: list,
        actuals: list
    ) -> MetricValue:
        """
        Calculate forecast mean absolute error.

        From paper: "MAE% as a predictability proxy"
        """
        if not forecasts or not actuals or len(forecasts) != len(actuals):
            return MetricValue(
                metric_name="forecast_mae",
                value=18.0,
                unit="%",
                sample_size=0
            )

        errors = []
        for forecast, actual in zip(forecasts, actuals):
            if actual > 0:
                error = abs(forecast - actual) / actual * 100
                errors.append(error)

        mae = statistics.mean(errors) if errors else 18.0

        return MetricValue(
            metric_name="forecast_mae",
            value=mae,
            unit="%",
            sample_size=len(errors),
            breakdown={
                "min_error": min(errors) if errors else 0,
                "max_error": max(errors) if errors else 0,
                "median_error": statistics.median(errors) if errors else 0
            }
        )

    def calculate_nba_acceptance(
        self,
        nba_decisions: list
    ) -> MetricValue:
        """
        Calculate NBA acceptance rate.

        From Table 1: Target 68.2%
        """
        accepted = sum(1 for d in nba_decisions if d.get("decision") == "accepted")
        total = len(nba_decisions)

        rate = (accepted / total * 100) if total > 0 else 0

        return MetricValue(
            metric_name="nba_acceptance",
            value=rate,
            unit="%",
            sample_size=total,
            breakdown={
                "accepted": accepted,
                "modified": sum(1 for d in nba_decisions if d.get("decision") == "modified"),
                "rejected": sum(1 for d in nba_decisions if d.get("decision") == "rejected")
            }
        )

    def calculate_all_metrics(self, data: dict) -> dict[str, MetricValue]:
        """Calculate all metrics from provided data."""
        results = {}

        if "interactions" in data:
            results["execution_latency"] = self.calculate_execution_latency(
                data["interactions"]
            )

        if "seller_activities" in data:
            results["admin_effort"] = self.calculate_admin_effort(
                data["seller_activities"],
                data.get("period_weeks", 1)
            )

        if "stage_transitions" in data:
            results["conversion_rate"] = self.calculate_conversion_rate(
                data["stage_transitions"]
            )

        if "forecasts" in data and "actuals" in data:
            results["forecast_mae"] = self.calculate_forecast_mae(
                data["forecasts"],
                data["actuals"]
            )

        if "nba_decisions" in data:
            results["nba_acceptance"] = self.calculate_nba_acceptance(
                data["nba_decisions"]
            )

        return results

    def compare_to_baseline(self, metric_value: MetricValue) -> dict:
        """Compare a metric value to its baseline."""
        definition = self._metrics.get(metric_value.metric_name)
        if not definition:
            return {"error": "Unknown metric"}

        delta = metric_value.value - definition.baseline_value
        delta_percent = (delta / definition.baseline_value * 100) if definition.baseline_value else 0

        if definition.target_direction == "lower":
            is_improvement = delta < 0
            target_met = metric_value.value <= definition.target_value
        else:
            is_improvement = delta > 0
            target_met = metric_value.value >= definition.target_value

        return {
            "metric": metric_value.metric_name,
            "current_value": metric_value.value,
            "baseline_value": definition.baseline_value,
            "target_value": definition.target_value,
            "delta": delta,
            "delta_percent": delta_percent,
            "is_improvement": is_improvement,
            "target_met": target_met,
            "target_direction": definition.target_direction
        }
