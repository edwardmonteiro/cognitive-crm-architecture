#!/usr/bin/env python3
"""
Cognitive CRM Architecture - Main Demo

This script demonstrates the complete architecture described in the paper:
"Toward Cognitive CRM: An Architecture for AI-Driven Customer Intelligence"

It runs:
1. Control simulation (baseline CRM)
2. Treatment simulation (cognitive CRM)
3. Generates Table 1 comparison
4. Demonstrates use cases
"""

from datetime import datetime
from uuid import uuid4

from src.simulation.simulator import Simulator, SimulationConfig, run_comparison
from src.simulation.metrics import MetricsCalculator, SensitivityAnalyzer
from src.use_cases.post_call_intelligence import PostCallIntelligenceUseCase
from src.use_cases.nba_recommendations import NBARecommendationsUseCase


def run_simulation_demo():
    """Run the simulation demonstration from Appendix B."""
    print("=" * 60)
    print("COGNITIVE CRM ARCHITECTURE - SIMULATION DEMO")
    print("Based on: 'Toward Cognitive CRM: An Architecture for")
    print("          AI-Driven Customer Intelligence'")
    print("=" * 60)
    print()

    # Configuration from Appendix B
    config = SimulationConfig(
        num_sellers=20,
        opportunities_per_seller=10,
        simulation_weeks=8,
        random_seed=42
    )

    print("Simulation Configuration:")
    print(f"  - Sellers: {config.num_sellers}")
    print(f"  - Opportunities per seller: {config.opportunities_per_seller}")
    print(f"  - Simulation period: {config.simulation_weeks} weeks")
    print()

    # Run comparison
    print("Running simulations...")
    print("  [1/2] Control group (baseline CRM)...")
    simulator = Simulator(config)
    control = simulator.run_control()
    print("        Done.")

    print("  [2/2] Treatment group (cognitive CRM)...")
    treatment = simulator.run_treatment()
    print("        Done.")
    print()

    # Generate Table 1
    print("=" * 60)
    print("TABLE 1: Feasibility Projections")
    print("(Simulated data - see paper Section 6)")
    print("=" * 60)
    print()
    print(f"{'Metric':<35} {'Control':<12} {'Treatment':<12} {'Δ (%)':<10}")
    print("-" * 60)

    metrics = [
        ("Average sales cycle (days)", control.avg_sales_cycle_days, treatment.avg_sales_cycle_days),
        ("Stage-to-stage conversion (%)", control.stage_conversion_rate * 100, treatment.stage_conversion_rate * 100),
        ("Forecast error (MAE %)", control.forecast_mae_percent, treatment.forecast_mae_percent),
        ("Admin time per seller (hrs/week)", control.admin_hours_per_week, treatment.admin_hours_per_week),
        ("Time between touchpoints (hrs)", control.touchpoint_interval_hours, treatment.touchpoint_interval_hours),
    ]

    for name, ctrl, treat in metrics:
        if ctrl != 0:
            delta = (treat - ctrl) / ctrl * 100
        else:
            delta = 0
        print(f"{name:<35} {ctrl:<12.1f} {treat:<12.1f} {delta:+.1f}%")

    print(f"{'NBA acceptance rate (%)':<35} {'–':<12} {treatment.nba_acceptance_rate * 100:<12.1f} {'n/a':<10}")
    print()

    # Summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print()
    print(f"Total opportunities simulated: {control.total_opportunities}")
    print(f"Control - Closed Won: {control.closed_won}, Lost: {control.closed_lost}")
    print(f"Treatment - Closed Won: {treatment.closed_won}, Lost: {treatment.closed_lost}")
    print(f"NBAs generated (treatment): {treatment.nbas_generated}")
    print(f"NBAs accepted (treatment): {treatment.nbas_accepted}")
    print()

    return control, treatment


def run_use_case_demo():
    """Demonstrate the use cases from Section 5.1."""
    print("=" * 60)
    print("USE CASE DEMONSTRATIONS")
    print("=" * 60)
    print()

    # Use Case 1: Post-Call Intelligence
    print("Use Case 1: Post-Call Intelligence")
    print("-" * 40)

    post_call = PostCallIntelligenceUseCase()

    sample_transcript = """
    Sales Rep: Thanks for taking the time to speak with us today about your CRM needs.
    Customer: Of course. We've been struggling with our current system.
    Sales Rep: What specific challenges are you facing?
    Customer: The main issue is that our sales team spends too much time on data entry.
    They estimate about 15 hours a week just updating the CRM.
    Sales Rep: That's significant. Our solution automates a lot of that.
    Customer: That sounds promising. What about the cost? Our budget is limited this quarter.
    Sales Rep: I understand budget constraints. Let me prepare a proposal with ROI analysis.
    Customer: That would be helpful. Can you have it ready by next week?
    Sales Rep: Absolutely. I'll also include some case studies from similar companies.
    Customer: Great. My VP will need to approve any purchase over $50K.
    Sales Rep: Understood. Should we schedule a demo with your VP as well?
    Customer: Yes, that would be a good next step.
    """

    result = post_call.process_call(
        transcript=sample_transcript,
        opportunity_id=uuid4(),
        participants=[
            {"id": "rep1", "name": "Sales Rep", "role": "seller"},
            {"id": "cust1", "name": "Customer", "role": "customer"}
        ],
        duration_seconds=1800,
        user_id="demo_user"
    )

    print(f"Processing time: {result.processing_time_seconds:.2f} seconds")
    print(f"Admin time saved: {result.admin_time_saved_minutes:.0f} minutes")
    print()
    print("Summary:", result.summary.get("executive_summary", "N/A")[:200] + "...")
    print()
    print(f"Objections found: {len(result.objections)}")
    for obj in result.objections[:2]:
        print(f"  - {obj.get('category', 'Unknown')}: {obj.get('quote', 'N/A')[:50]}...")
    print()
    print(f"Risks identified: {len(result.risks)}")
    for risk in result.risks[:2]:
        print(f"  - {risk.get('category', 'Unknown')}: {risk.get('severity', 'N/A')}")
    print()
    print(f"Next steps recommended: {len(result.next_steps)}")
    for step in result.next_steps[:2]:
        print(f"  - [{step.get('priority', 'normal')}] {step.get('action', 'N/A')[:50]}...")
    print()

    # Use Case 2: NBA Recommendations
    print()
    print("Use Case 2: NBA Recommendations")
    print("-" * 40)

    nba = NBARecommendationsUseCase()

    nba_result = nba.generate_recommendations(
        opportunity_id=uuid4(),
        opportunity_data={
            "stage": "value_proposition",
            "amount": 75000,
            "days_in_stage": 14,
            "last_activity": "3 days ago"
        },
        max_recommendations=3
    )

    print(f"Generation time: {nba_result.generation_time_seconds:.2f} seconds")
    print(f"Evidence retrieved: {nba_result.evidence_retrieved} sources")
    print()
    print("Recommendations:")
    for i, rec in enumerate(nba_result.recommendations, 1):
        print(f"  {i}. [{rec.priority.upper()}] {rec.action}")
        print(f"     Type: {rec.action_type}")
        print(f"     Confidence: {rec.confidence_score:.0%}")
        print(f"     Target: {rec.target_stakeholder or 'N/A'}")
        print()


def main():
    """Main entry point."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       COGNITIVE CRM ARCHITECTURE DEMONSTRATION           ║")
    print("║                                                          ║")
    print("║  A Design Science Research Implementation                ║")
    print("║  Author: Edward Monteiro                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Run simulation
    control, treatment = run_simulation_demo()

    # Run use case demos
    run_use_case_demo()

    print()
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("This demonstration shows the feasibility of the cognitive CRM")
    print("architecture as described in the paper. Key findings:")
    print()

    cycle_reduction = (1 - treatment.avg_sales_cycle_days / control.avg_sales_cycle_days) * 100
    admin_reduction = (1 - treatment.admin_hours_per_week / control.admin_hours_per_week) * 100

    print(f"  ✓ Sales cycle reduced by {cycle_reduction:.1f}%")
    print(f"  ✓ Admin time reduced by {admin_reduction:.1f}%")
    print(f"  ✓ NBA acceptance rate: {treatment.nba_acceptance_rate * 100:.1f}%")
    print()
    print("See the paper for full methodology and limitations.")
    print()


if __name__ == "__main__":
    main()
