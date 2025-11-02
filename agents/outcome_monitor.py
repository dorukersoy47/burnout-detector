"""
Outcome Monitor Agent - Tracks intervention results and provides RLHF feedback
Mostly workflow (15% agentic) - primarily monitoring and statistical analysis.

Agentic Score: 15% - Deterministic tracking with automated learning updates
"""

import json
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class MonitoringDataPoint:
    """Single data point in monitoring timeline"""
    date: str
    productivity_score: float
    pr_review_time: float
    pr_merge_time: float
    blocked_prs: int
    team_morale: str  # "low", "neutral", "high"


class OutcomeMonitor:
    """
    Monitors intervention outcomes and learns from results.

    Key characteristics:
    - Tracks metrics on fixed schedule
    - Compares to predictions
    - Detects side effects
    - Feeds learning back to system
    """

    def __init__(self):
        self.monitoring_id = None
        self.intervention_details = None
        self.baseline_metrics = None
        self.monitoring_data: List[MonitoringDataPoint] = []

    def start_monitoring(self, intervention_id: str, intervention_details: Dict,
                         baseline_metrics: Dict) -> str:
        """Start monitoring an intervention"""
        print("\n" + "=" * 70)
        print("📊 OUTCOME MONITOR - Starting Monitoring")
        print("=" * 70)

        self.monitoring_id = f"mon_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.intervention_details = intervention_details
        self.baseline_metrics = baseline_metrics

        print(f"\n📍 Monitoring Started")
        print(f"   Intervention: {intervention_details.get('recommendation', 'Unknown')}")
        print(f"   Monitoring ID: {self.monitoring_id}")
        print(f"   Start Time: {datetime.utcnow().isoformat()}")

        # Store baseline
        self._record_baseline_metrics()

        return self.monitoring_id

    def _record_baseline_metrics(self):
        """Record baseline metrics before intervention"""
        baseline = MonitoringDataPoint(
            date=datetime.utcnow().isoformat(),
            productivity_score=self.baseline_metrics.get("productivity_score", 6.0),
            pr_review_time=self.baseline_metrics.get("pr_review_time_hours", 48),
            pr_merge_time=self.baseline_metrics.get("pr_merge_time_hours", 72),
            blocked_prs=self.baseline_metrics.get("blocked_prs", 12),
            team_morale="low"
        )

        self.monitoring_data.append(baseline)

        print(f"\n📌 Baseline Recorded:")
        print(f"   Productivity: {baseline.productivity_score}/10")
        print(f"   PR Review Time: {baseline.pr_review_time}h")
        print(f"   Blocked PRs: {baseline.blocked_prs}")
        print(f"   Team Morale: {baseline.team_morale}")

    def get_dummy_monitoring_data(self) -> List[MonitoringDataPoint]:
        """Generate dummy monitoring data for days 1-7 after intervention"""
        data = []

        base_date = datetime.utcnow()

        # Day 1 (after async conversion)
        data.append(MonitoringDataPoint(
            date=(base_date + timedelta(days=1)).isoformat(),
            productivity_score=6.5,
            pr_review_time=42,
            pr_merge_time=65,
            blocked_prs=10,
            team_morale="neutral"
        ))

        # Day 3 (momentum building)
        data.append(MonitoringDataPoint(
            date=(base_date + timedelta(days=3)).isoformat(),
            productivity_score=7.8,
            pr_review_time=18,
            pr_merge_time=30,
            blocked_prs=3,
            team_morale="neutral"
        ))

        # Day 5 (strong recovery)
        data.append(MonitoringDataPoint(
            date=(base_date + timedelta(days=5)).isoformat(),
            productivity_score=8.2,
            pr_review_time=10,
            pr_merge_time=20,
            blocked_prs=1,
            team_morale="high"
        ))

        # Day 7 (sustained improvement)
        data.append(MonitoringDataPoint(
            date=(base_date + timedelta(days=7)).isoformat(),
            productivity_score=8.5,
            pr_review_time=9,
            pr_merge_time=18,
            blocked_prs=0,
            team_morale="high"
        ))

        return data

    def monitor_daily(self) -> Dict:
        """Simulate daily monitoring check"""
        print("\n" + "=" * 70)
        print("📈 OUTCOME MONITOR - Daily Check")
        print("=" * 70)

        # Get new data
        new_data = self.get_dummy_monitoring_data()
        self.monitoring_data.extend(new_data)

        # Analyze latest point
        latest = self.monitoring_data[-1]
        baseline = self.monitoring_data[0]

        # Calculate improvements
        productivity_improvement = ((latest.productivity_score - baseline.productivity_score)
                                    / baseline.productivity_score)
        review_time_improvement = ((baseline.pr_review_time - latest.pr_review_time)
                                   / baseline.pr_review_time)

        print(f"\n📊 Latest Metrics ({len(self.monitoring_data) - 1} days in):")
        print(f"   Productivity: {baseline.productivity_score} → {latest.productivity_score} "
              f"({productivity_improvement:+.1%})")
        print(f"   PR Review Time: {baseline.pr_review_time}h → {latest.pr_review_time}h "
              f"({-review_time_improvement:+.1%})")
        print(f"   Blocked PRs: {baseline.blocked_prs} → {latest.blocked_prs}")
        print(f"   Team Morale: {latest.team_morale}")

        return {
            "days_elapsed": len(self.monitoring_data) - 1,
            "latest_metrics": {
                "productivity": latest.productivity_score,
                "pr_review_time": latest.pr_review_time,
                "pr_merge_time": latest.pr_merge_time,
                "blocked_prs": latest.blocked_prs
            },
            "improvements": {
                "productivity": productivity_improvement,
                "review_time": -review_time_improvement,
            }
        }

    def detect_side_effects(self) -> List[str]:
        """Check for negative side effects"""
        print("\n" + "=" * 70)
        print("⚠️ OUTCOME MONITOR - Side Effect Detection")
        print("=" * 70)

        side_effects = []

        # Analyze trends
        if len(self.monitoring_data) > 1:
            latest = self.monitoring_data[-1]
            baseline = self.monitoring_data[0]

            # Check for concerning patterns
            if latest.team_morale == "low" and baseline.team_morale == "high":
                side_effects.append("Team morale decreased")

            if latest.blocked_prs > baseline.blocked_prs * 1.5:
                side_effects.append("Blocked PRs increased unexpectedly")

            # Check for regression
            if latest.productivity_score < baseline.productivity_score * 0.95:
                side_effects.append("Productivity regressed")

        if side_effects:
            print(f"\n⚠️ Side Effects Detected:")
            for effect in side_effects:
                print(f"   - {effect}")
        else:
            print(f"\n✓ No concerning side effects detected")

        return side_effects

    def generate_rlhf_feedback(self, intervention: str, outcome: str) -> Dict:
        """Generate RLHF feedback to improve future recommendations"""
        print("\n" + "=" * 70)
        print("🧠 OUTCOME MONITOR - RLHF Learning")
        print("=" * 70)

        feedback = {
            "intervention": intervention,
            "outcome": outcome,
            "success": outcome == "success",
            "improvement_magnitude": 0.37 if outcome == "success" else 0.10,
            "time_to_effect_days": 3 if outcome == "success" else 7,
            "team_satisfaction": "high" if outcome == "success" else "medium",
            "timestamp": datetime.utcnow().isoformat(),
            "learning": {
                "success_rate_update": 0.79 if outcome == "success" else 0.65,
                "estimated_future_success": 0.82,
                "confidence_boost": 0.05
            }
        }

        print(f"\n📚 RLHF Feedback Generated:")
        print(f"   Intervention: {intervention}")
        print(f"   Outcome: {outcome}")
        print(f"   Success Rate: {feedback['learning']['success_rate_update']:.0%}")
        print(f"   Updated Model Confidence: +{feedback['learning']['confidence_boost']:.0%}")

        return feedback

    def generate_report(self) -> Dict:
        """Generate final monitoring report"""
        print("\n" + "=" * 70)
        print("📄 OUTCOME MONITOR - Final Report")
        print("=" * 70)

        if len(self.monitoring_data) < 2:
            return {"status": "insufficient_data"}

        baseline = self.monitoring_data[0]
        latest = self.monitoring_data[-1]

        # Determine success
        productivity_improved = latest.productivity_score > baseline.productivity_score
        review_time_improved = latest.pr_review_time < baseline.pr_review_time
        success = productivity_improved and review_time_improved

        print(f"\n✅ Intervention Results:")
        print(f"   {'✓' if productivity_improved else '✗'} Productivity improved")
        print(f"   {'✓' if review_time_improved else '✗'} Review time decreased")
        print(f"   {'✓' if success else '✗'} Overall: {'SUCCESS' if success else 'NEEDS ADJUSTMENT'}")

        # Generate RLHF feedback
        outcome = "success" if success else "partial"
        rlhf_feedback = self.generate_rlhf_feedback(
            self.intervention_details.get("recommendation", "Unknown"),
            outcome
        )

        # Detect side effects
        side_effects = self.detect_side_effects()

        report = {
            "monitoring_id": self.monitoring_id,
            "status": "complete",
            "duration_days": len(self.monitoring_data) - 1,
            "baseline": {
                "productivity": baseline.productivity_score,
                "review_time": baseline.pr_review_time,
                "blocked_prs": baseline.blocked_prs
            },
            "final": {
                "productivity": latest.productivity_score,
                "review_time": latest.pr_review_time,
                "blocked_prs": latest.blocked_prs
            },
            "improvements": {
                "productivity_pct": ((latest.productivity_score - baseline.productivity_score)
                                     / baseline.productivity_score),
                "review_time_pct": ((baseline.pr_review_time - latest.pr_review_time)
                                    / baseline.pr_review_time),
            },
            "success": success,
            "side_effects": side_effects,
            "rlhf_feedback": rlhf_feedback
        }

        print(f"\n📊 Summary:")
        print(f"   Productivity: {report['improvements']['productivity_pct']:+.0%}")
        print(f"   Review Time: {report['improvements']['review_time_pct']:+.0%}")
        print(f"   Final Status: {report['success']}")

        return report


# For testing
if __name__ == "__main__":
    monitor = OutcomeMonitor()

    baseline_metrics = {
        "productivity_score": 6.0,
        "pr_review_time_hours": 48,
        "pr_merge_time_hours": 72,
        "blocked_prs": 12
    }

    intervention = {
        "recommendation": "Convert daily meeting to async updates"
    }

    monitoring_id = monitor.start_monitoring("inv_001", intervention, baseline_metrics)

    # Simulate daily checks
    for day in range(1, 8, 2):
        if day > 1:
            daily_check = monitor.monitor_daily()
            print(f"\n📋 Day {daily_check['days_elapsed']} Status:")
            print(json.dumps(daily_check, indent=2))

    # Generate final report
    report = monitor.generate_report()

    print(f"\n📋 Final Report:")
    print(json.dumps(report, indent=2))