"""
Metrics and Evaluation System

From Section 5.2 of the paper:

Metrics Rationale:
- Execution latency: time between touchpoints (sense→act delay)
- Administrative effort: hours/week spent updating CRM (cognitive load proxy)
- Conversion consistency: stage-to-stage conversion and variance across sellers
- Forecast variance: MAE% as a predictability proxy

This module provides:
- Metric calculation
- Dashboard data generation
- Trend analysis
- Alerting thresholds
"""

from .calculator import MetricsCalculator, MetricDefinition
from .dashboard import DashboardData, MetricTrend
from .alerts import AlertRule, AlertEngine

__all__ = [
    "MetricsCalculator",
    "MetricDefinition",
    "DashboardData",
    "MetricTrend",
    "AlertRule",
    "AlertEngine"
]
