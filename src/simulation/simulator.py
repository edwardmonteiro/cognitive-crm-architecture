"""
Simulation Engine

From Appendix B:
"Purpose: The simulation evaluates whether the architecture's mechanisms
produce internally coherent, directionally plausible effects across
coupled execution metrics."

This module implements the simulation engine that:
- Generates synthetic sales scenarios
- Runs control (baseline CRM) and treatment (cognitive CRM) simulations
- Collects metrics for comparison
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4
import random
import statistics

from .entities import (
    SimulatedSeller,
    SimulatedOpportunity,
    SimulatedInteraction,
    SimulatedNBA,
    OpportunityStage,
    NBAOutcome
)


@dataclass
class SimulationConfig:
    """
    Configuration for simulation runs.

    From Appendix B.2:
    - Baseline cycle time mean ≈ 94 days
    - Baseline conversion ≈ 31.5%
    - Baseline forecast MAE ≈ 18%
    - Admin time ≈ 14.8 hrs/week per seller
    """
    # Simulation parameters
    num_sellers: int = 20
    opportunities_per_seller: int = 10
    simulation_weeks: int = 8

    # Baseline parameters (Control)
    baseline_cycle_time_days: float = 94.0
    baseline_conversion_rate: float = 0.315
    baseline_forecast_mae_percent: float = 18.0
    baseline_admin_hours_per_week: float = 14.8
    baseline_touchpoint_interval_hours: float = 52.4

    # Treatment effect assumptions (from B.3)
    ai_admin_time_reduction: float = 0.318   # ~31.8% reduction
    ai_touchpoint_reduction: float = 0.244   # ~24.4% reduction
    ai_conversion_uplift: float = 0.064      # +6.4% absolute
    ai_cycle_time_reduction: float = 0.213   # ~21.3% reduction
    ai_forecast_mae_reduction: float = 0.389  # ~38.9% reduction

    # NBA parameters
    nba_acceptance_rate: float = 0.682

    # Randomization
    random_seed: Optional[int] = None


@dataclass
class SimulationResult:
    """
    Results of a simulation run.

    Contains all metrics needed for Table 1 comparison.
    """
    id: UUID = field(default_factory=uuid4)
    config: SimulationConfig = None
    is_treatment: bool = False

    # Core metrics
    avg_sales_cycle_days: float = 0.0
    stage_conversion_rate: float = 0.0
    forecast_mae_percent: float = 0.0
    admin_hours_per_week: float = 0.0
    touchpoint_interval_hours: float = 0.0

    # NBA metrics (treatment only)
    nba_acceptance_rate: float = 0.0
    nbas_generated: int = 0
    nbas_accepted: int = 0

    # Sample sizes
    total_opportunities: int = 0
    closed_won: int = 0
    closed_lost: int = 0

    # Raw data for analysis
    seller_level_metrics: list = field(default_factory=list)
    opportunity_outcomes: list = field(default_factory=list)

    # Timing
    simulation_start: datetime = field(default_factory=datetime.now)
    simulation_end: Optional[datetime] = None


class Simulator:
    """
    Main simulation engine.

    Runs control and treatment simulations to demonstrate
    feasibility of the cognitive CRM architecture.
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        if self.config.random_seed:
            random.seed(self.config.random_seed)

        # Simulation state
        self._sellers: list[SimulatedSeller] = []
        self._opportunities: dict[str, SimulatedOpportunity] = {}
        self._interactions: list[SimulatedInteraction] = []
        self._nbas: list[SimulatedNBA] = []

    def run_control(self) -> SimulationResult:
        """Run control simulation (baseline CRM workflow)."""
        return self._run_simulation(is_treatment=False)

    def run_treatment(self) -> SimulationResult:
        """Run treatment simulation (cognitive CRM workflow)."""
        return self._run_simulation(is_treatment=True)

    def _run_simulation(self, is_treatment: bool) -> SimulationResult:
        """Run a complete simulation."""
        self._reset_state()

        result = SimulationResult(
            config=self.config,
            is_treatment=is_treatment,
            simulation_start=datetime.now()
        )

        # Initialize sellers
        self._initialize_sellers()

        # Initialize opportunities
        start_date = datetime.now() - timedelta(weeks=self.config.simulation_weeks)
        self._initialize_opportunities(start_date)

        # Run simulation week by week
        for week in range(self.config.simulation_weeks):
            week_start = start_date + timedelta(weeks=week)
            self._simulate_week(week_start, is_treatment)

        # Calculate metrics
        self._calculate_metrics(result, is_treatment)

        result.simulation_end = datetime.now()
        return result

    def _reset_state(self) -> None:
        """Reset simulation state."""
        self._sellers = []
        self._opportunities = {}
        self._interactions = []
        self._nbas = []

    def _initialize_sellers(self) -> None:
        """Initialize seller population."""
        for _ in range(self.config.num_sellers):
            seller = SimulatedSeller.generate()
            self._sellers.append(seller)

    def _initialize_opportunities(self, start_date: datetime) -> None:
        """Initialize opportunities for each seller."""
        for seller in self._sellers:
            for i in range(self.config.opportunities_per_seller):
                # Stagger opportunity creation
                opp_start = start_date + timedelta(
                    days=random.randint(0, self.config.simulation_weeks * 7 // 2)
                )
                opp = SimulatedOpportunity.generate(seller.id, opp_start)
                self._opportunities[str(opp.id)] = opp
                seller.active_opportunity_ids.append(opp.id)

    def _simulate_week(self, week_start: datetime, is_treatment: bool) -> None:
        """Simulate one week of sales activity."""
        for seller in self._sellers:
            self._simulate_seller_week(seller, week_start, is_treatment)

    def _simulate_seller_week(
        self,
        seller: SimulatedSeller,
        week_start: datetime,
        is_treatment: bool
    ) -> None:
        """Simulate one seller's activities for a week."""
        admin_hours_this_week = 0.0

        for opp_id in seller.active_opportunity_ids[:]:  # Copy to allow modification
            opp = self._opportunities.get(str(opp_id))
            if not opp or opp.is_closed:
                continue

            # Simulate interactions for this opportunity
            interactions_this_week = self._simulate_opportunity_week(
                opp, seller, week_start, is_treatment
            )

            # Track admin time
            if is_treatment:
                # AI reduces admin time
                admin_per_interaction = (
                    seller.admin_time_hours_per_week / 10 *
                    (1 - self.config.ai_admin_time_reduction)
                )
            else:
                admin_per_interaction = seller.admin_time_hours_per_week / 10

            admin_hours_this_week += admin_per_interaction * len(interactions_this_week)

            # Update opportunity state
            opp.days_in_current_stage += 7

            # Check for stage progression
            self._check_stage_progression(opp, seller, is_treatment)

        seller.cumulative_admin_hours += admin_hours_this_week

    def _simulate_opportunity_week(
        self,
        opp: SimulatedOpportunity,
        seller: SimulatedSeller,
        week_start: datetime,
        is_treatment: bool
    ) -> list[SimulatedInteraction]:
        """Simulate interactions for one opportunity in one week."""
        interactions = []

        # Determine number of interactions based on touchpoint interval
        if is_treatment:
            avg_interval = seller.avg_touchpoint_interval_hours * (1 - self.config.ai_touchpoint_reduction)
        else:
            avg_interval = seller.avg_touchpoint_interval_hours

        expected_interactions = 168 / avg_interval  # Hours in a week

        # Add some randomness
        num_interactions = max(1, int(expected_interactions * random.uniform(0.7, 1.3)))

        for i in range(num_interactions):
            timestamp = week_start + timedelta(hours=random.uniform(0, 168))

            interaction = SimulatedInteraction.generate(
                opportunity_id=opp.id,
                seller_id=seller.id,
                timestamp=timestamp,
                ai_enabled=is_treatment
            )
            interactions.append(interaction)
            self._interactions.append(interaction)

            # Update opportunity
            opp.interaction_count += 1
            opp.last_interaction_at = timestamp

            # Generate NBA in treatment group
            if is_treatment and interaction.triggered_nba:
                nba = SimulatedNBA.generate(
                    opportunity_id=opp.id,
                    seller_id=seller.id,
                    timestamp=timestamp,
                    seller_adoption_rate=seller.nba_adoption_rate
                )
                self._nbas.append(nba)
                opp.nbas_received += 1

                if nba.outcome == NBAOutcome.ACCEPTED:
                    opp.nbas_accepted += 1
                elif nba.outcome == NBAOutcome.MODIFIED:
                    opp.nbas_modified += 1
                elif nba.outcome == NBAOutcome.REJECTED:
                    opp.nbas_rejected += 1

        return interactions

    def _check_stage_progression(
        self,
        opp: SimulatedOpportunity,
        seller: SimulatedSeller,
        is_treatment: bool
    ) -> None:
        """Check if opportunity should progress to next stage."""
        if opp.is_closed:
            return

        # Calculate progression probability
        base_conversion = seller.base_conversion_rate
        if is_treatment:
            base_conversion += self.config.ai_conversion_uplift

        # Probability increases with time in stage (up to a point)
        time_factor = min(1.0, opp.days_in_current_stage / 30)

        # Adjust for stage
        stage_factors = {
            OpportunityStage.PROSPECTING: 0.4,
            OpportunityStage.QUALIFICATION: 0.5,
            OpportunityStage.NEEDS_ANALYSIS: 0.6,
            OpportunityStage.VALUE_PROPOSITION: 0.7,
            OpportunityStage.NEGOTIATION: 0.8
        }
        stage_factor = stage_factors.get(opp.stage, 0.5)

        # Weekly probability of stage change
        progress_prob = base_conversion * time_factor * stage_factor * 0.3

        if random.random() < progress_prob:
            # Determine if advance or regress
            if random.random() < 0.85:  # 85% chance of advancing
                self._advance_opportunity(opp, seller)
            else:
                self._regress_opportunity(opp, seller)

    def _advance_opportunity(self, opp: SimulatedOpportunity, seller: SimulatedSeller) -> None:
        """Advance opportunity to next stage."""
        stage_order = [
            OpportunityStage.PROSPECTING,
            OpportunityStage.QUALIFICATION,
            OpportunityStage.NEEDS_ANALYSIS,
            OpportunityStage.VALUE_PROPOSITION,
            OpportunityStage.NEGOTIATION,
            OpportunityStage.CLOSED_WON
        ]

        current_idx = stage_order.index(opp.stage)
        if current_idx < len(stage_order) - 1:
            opp.advance_stage(stage_order[current_idx + 1])

            if opp.is_won:
                seller.total_deals_closed += 1

    def _regress_opportunity(self, opp: SimulatedOpportunity, seller: SimulatedSeller) -> None:
        """Lose the opportunity."""
        opp.advance_stage(OpportunityStage.CLOSED_LOST)
        seller.total_deals_lost += 1

    def _calculate_metrics(self, result: SimulationResult, is_treatment: bool) -> None:
        """Calculate final metrics from simulation."""
        # Collect closed opportunities
        closed_opps = [o for o in self._opportunities.values() if o.is_closed]
        won_opps = [o for o in closed_opps if o.is_won]
        lost_opps = [o for o in closed_opps if not o.is_won]

        result.total_opportunities = len(self._opportunities)
        result.closed_won = len(won_opps)
        result.closed_lost = len(lost_opps)

        # Sales cycle (for won deals)
        if won_opps:
            cycles = [
                (o.actual_close_date - o.created_at).days
                for o in won_opps
            ]
            result.avg_sales_cycle_days = statistics.mean(cycles)
        else:
            result.avg_sales_cycle_days = self.config.baseline_cycle_time_days

        # Conversion rate
        if closed_opps:
            result.stage_conversion_rate = len(won_opps) / len(closed_opps)
        else:
            result.stage_conversion_rate = self.config.baseline_conversion_rate

        # Admin time per seller per week
        total_admin = sum(s.cumulative_admin_hours for s in self._sellers)
        result.admin_hours_per_week = total_admin / (
            self.config.num_sellers * self.config.simulation_weeks
        )

        # Touchpoint interval
        intervals = []
        for opp in self._opportunities.values():
            opp_interactions = sorted([
                i for i in self._interactions if i.opportunity_id == opp.id
            ], key=lambda x: x.timestamp)

            for i in range(1, len(opp_interactions)):
                delta = opp_interactions[i].timestamp - opp_interactions[i-1].timestamp
                intervals.append(delta.total_seconds() / 3600)

        if intervals:
            result.touchpoint_interval_hours = statistics.mean(intervals)
        else:
            result.touchpoint_interval_hours = self.config.baseline_touchpoint_interval_hours

        # Forecast MAE (simulated based on assumptions)
        if is_treatment:
            result.forecast_mae_percent = self.config.baseline_forecast_mae_percent * (
                1 - self.config.ai_forecast_mae_reduction
            )
        else:
            result.forecast_mae_percent = self.config.baseline_forecast_mae_percent

        # NBA metrics (treatment only)
        if is_treatment:
            result.nbas_generated = len(self._nbas)
            result.nbas_accepted = sum(
                1 for n in self._nbas if n.outcome == NBAOutcome.ACCEPTED
            )
            if result.nbas_generated > 0:
                result.nba_acceptance_rate = result.nbas_accepted / result.nbas_generated

        # Seller-level metrics for bootstrap analysis
        for seller in self._sellers:
            seller_opps = [
                o for o in self._opportunities.values()
                if o.seller_id == seller.id
            ]
            seller_closed = [o for o in seller_opps if o.is_closed]
            seller_won = [o for o in seller_closed if o.is_won]

            seller_cycles = [
                (o.actual_close_date - o.created_at).days
                for o in seller_won
            ] if seller_won else [self.config.baseline_cycle_time_days]

            result.seller_level_metrics.append({
                "seller_id": str(seller.id),
                "skill_level": seller.skill_level.value,
                "total_opps": len(seller_opps),
                "closed": len(seller_closed),
                "won": len(seller_won),
                "conversion_rate": len(seller_won) / len(seller_closed) if seller_closed else 0,
                "avg_cycle_days": statistics.mean(seller_cycles),
                "admin_hours": seller.cumulative_admin_hours / self.config.simulation_weeks
            })


def run_comparison(
    config: SimulationConfig = None,
    num_runs: int = 1
) -> dict:
    """
    Run control and treatment simulations for comparison.

    Returns results suitable for Table 1 in the paper.
    """
    config = config or SimulationConfig()

    control_results = []
    treatment_results = []

    for run in range(num_runs):
        # Set different seed for each run
        run_config = SimulationConfig(**vars(config))
        run_config.random_seed = (config.random_seed or 42) + run

        simulator = Simulator(run_config)

        control = simulator.run_control()
        control_results.append(control)

        treatment = simulator.run_treatment()
        treatment_results.append(treatment)

    # Aggregate results
    def aggregate(results: list[SimulationResult], metric: str) -> dict:
        values = [getattr(r, metric) for r in results]
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values)
        }

    metrics = [
        "avg_sales_cycle_days",
        "stage_conversion_rate",
        "forecast_mae_percent",
        "admin_hours_per_week",
        "touchpoint_interval_hours"
    ]

    comparison = {
        "num_runs": num_runs,
        "control": {m: aggregate(control_results, m) for m in metrics},
        "treatment": {m: aggregate(treatment_results, m) for m in metrics},
        "deltas": {}
    }

    # Calculate deltas
    for metric in metrics:
        control_mean = comparison["control"][metric]["mean"]
        treatment_mean = comparison["treatment"][metric]["mean"]

        if control_mean != 0:
            delta_percent = (treatment_mean - control_mean) / control_mean * 100
        else:
            delta_percent = 0

        comparison["deltas"][metric] = {
            "absolute": treatment_mean - control_mean,
            "percent": delta_percent
        }

    # Add NBA-specific metrics for treatment
    nba_acceptance_rates = [r.nba_acceptance_rate for r in treatment_results]
    comparison["treatment"]["nba_acceptance_rate"] = {
        "mean": statistics.mean(nba_acceptance_rates),
        "std": statistics.stdev(nba_acceptance_rates) if len(nba_acceptance_rates) > 1 else 0
    }

    return comparison
