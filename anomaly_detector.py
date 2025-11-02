"""
Anomaly Detector - Monitors productivity metrics and triggers investigations
This is a pure workflow (not agentic) that runs on a fixed schedule.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import statistics


@dataclass
class ProductivityMetrics:
    """Holds productivity metrics"""
    timestamp: str
    productivity_score: float  # 0-10 scale
    pr_review_time_hours: float
    pr_merge_time_hours: float
    blocked_prs: int
    team_members_affected: list
    meeting_hours_per_day: float


class AnomalyDetector:
    """
    Detects when productivity drops significantly.
    Runs on a fixed daily schedule.

    Agentic Score: 0% - Pure workflow with deterministic logic
    """

    def __init__(self, threshold_z_score: float = 1.5):
        self.threshold_z_score = threshold_z_score
        self.baseline_window_days = 30
        self.metrics_history: list[ProductivityMetrics] = []
        self.last_alert_time: Optional[datetime] = None

    def get_dummy_metrics(self, days_ago: int = 0) -> ProductivityMetrics:
        """
        Generates dummy productivity metrics.
        In production, this would query real data sources.
        """
        base_date = datetime.utcnow() - timedelta(days=days_ago)
        timestamp = base_date.isoformat()

        # Simulate normal baseline (8.5/10) for past 30 days
        if days_ago > 7:
            productivity_score = 8.5 + (2 * (abs(days_ago - 15) / 15))  # Normal with slight variation
            pr_review_time = 8 + (2 * (abs(days_ago - 15) / 15))
            pr_merge_time = 16 + (4 * (abs(days_ago - 15) / 15))
            blocked_prs = 1
            meeting_hours = 3
            team_affected = []
        else:
            # Last 7 days: ANOMALY - productivity dropped
            productivity_score = 6.0  # Dropped from 8.5 to 6.0
            pr_review_time = 48  # 8 hours → 48 hours
            pr_merge_time = 72  # 16 hours → 72 hours
            blocked_prs = 12
            meeting_hours = 7  # Meeting load increased
            team_affected = ["alice", "bob"]  # Senior engineers

        return ProductivityMetrics(
            timestamp=timestamp,
            productivity_score=productivity_score,
            pr_review_time_hours=pr_review_time,
            pr_merge_time_hours=pr_merge_time,
            blocked_prs=blocked_prs,
            team_members_affected=team_affected,
            meeting_hours_per_day=meeting_hours
        )

    def load_baseline(self) -> Dict[str, float]:
        """
        Load 30-day baseline metrics.
        Returns mean and std dev for each metric.
        """
        baseline_metrics = []

        # Load last 30 days of dummy data
        for days_ago in range(self.baseline_window_days, 0, -1):
            if days_ago > 7:  # Only use stable periods
                metrics = self.get_dummy_metrics(days_ago)
                baseline_metrics.append(metrics)

        # Calculate statistics
        scores = [m.productivity_score for m in baseline_metrics]
        review_times = [m.pr_review_time_hours for m in baseline_metrics]

        return {
            "productivity_mean": statistics.mean(scores),
            "productivity_stdev": statistics.stdev(scores) if len(scores) > 1 else 0.5,
            "review_time_mean": statistics.mean(review_times),
            "review_time_stdev": statistics.stdev(review_times) if len(review_times) > 1 else 2.0,
        }

    def calculate_z_score(self, value: float, mean: float, stdev: float) -> float:
        """Calculate z-score for anomaly detection"""
        if stdev == 0:
            return 0
        return abs((value - mean) / stdev)

    def check_for_anomalies(self) -> Optional[Dict]:
        """
        Daily check (deterministic workflow).
        Returns alert if anomaly detected, None otherwise.
        """
        print("\n" + "=" * 70)
        print("🕐 ANOMALY DETECTOR - Daily Check")
        print("=" * 70)

        # Get today's metrics
        today_metrics = self.get_dummy_metrics(days_ago=0)

        # Load baseline
        baseline = self.load_baseline()

        # Calculate z-scores
        productivity_z = self.calculate_z_score(
            today_metrics.productivity_score,
            baseline["productivity_mean"],
            baseline["productivity_stdev"]
        )

        review_time_z = self.calculate_z_score(
            today_metrics.pr_review_time_hours,
            baseline["review_time_mean"],
            baseline["review_time_stdev"]
        )

        print(f"\n📊 Current Metrics (Today):")
        print(f"   Productivity Score: {today_metrics.productivity_score}/10")
        print(
            f"   PR Review Time: {today_metrics.pr_review_time_hours}h (baseline: {baseline['review_time_mean']:.1f}h)")
        print(f"   PR Merge Time: {today_metrics.pr_merge_time_hours}h")
        print(f"   Blocked PRs: {today_metrics.blocked_prs}")
        print(f"   Meeting Hours: {today_metrics.meeting_hours_per_day}h")

        print(f"\n📈 Statistical Analysis:")
        print(f"   Productivity Z-Score: {productivity_z:.2f} (threshold: {self.threshold_z_score})")
        print(f"   Review Time Z-Score: {review_time_z:.2f}")

        # Trigger if anomaly detected
        if productivity_z > self.threshold_z_score or review_time_z > self.threshold_z_score:
            print(f"\n🚨 ANOMALY DETECTED! Z-score exceeds threshold")

            alert = {
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "high" if productivity_z > 2.5 else "medium",
                "metrics": asdict(today_metrics),
                "baseline": baseline,
                "z_scores": {
                    "productivity": productivity_z,
                    "review_time": review_time_z
                },
                "description": self._generate_alert_message(today_metrics, baseline),
                "affected_areas": self._identify_affected_areas(today_metrics)
            }

            self.last_alert_time = datetime.utcnow()
            return alert
        else:
            print(f"\n✅ No anomalies detected. System healthy.")
            return None

    def _generate_alert_message(self, metrics: ProductivityMetrics, baseline: Dict) -> str:
        """Generate human-readable alert message"""
        changes = []

        productivity_change = ((baseline["productivity_mean"] - metrics.productivity_score)
                               / baseline["productivity_mean"] * 100)
        review_change = ((metrics.pr_review_time_hours - baseline["review_time_mean"])
                         / baseline["review_time_mean"] * 100)

        changes.append(f"Productivity dropped {productivity_change:.0f}%")
        changes.append(f"PR review time increased {review_change:.0f}%")

        if metrics.blocked_prs > 5:
            changes.append(f"{metrics.blocked_prs} PRs blocked")

        if metrics.team_members_affected:
            changes.append(f"{', '.join(metrics.team_members_affected)} affected")

        return " | ".join(changes)

    def _identify_affected_areas(self, metrics: ProductivityMetrics) -> list[str]:
        """Identify which areas are affected"""
        areas = []

        if metrics.pr_review_time_hours > 20:
            areas.append("code_review")
        if metrics.pr_merge_time_hours > 40:
            areas.append("deployment")
        if metrics.meeting_hours_per_day > 6:
            areas.append("meeting_load")
        if metrics.blocked_prs > 5:
            areas.append("blocking_issues")

        return areas


# For testing
if __name__ == "__main__":
    detector = AnomalyDetector()
    alert = detector.check_for_anomalies()

    if alert:
        print(f"\n📋 Alert JSON:")
        print(json.dumps(alert, indent=2))