"""
Mock data provider for testing and development
Reads from CSV files in ./mock_data directory
Simulates team-level metrics without API calls
"""

import os
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from statistics import median
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)

MOCK_DATA_DIR = Path(__file__).parent / "mock_data"


class MockDataProvider:
    """
    Provides mock data from CSV files
    Format: date,repo,commits,after_hours_commits,prs_opened,prs_merged,
             median_pr_lead_time_hours,median_time_to_first_review_hours,reviews_count
    """

    def __init__(self, config: Config):
        self.config = config

        # Create mock_data directory if it doesn't exist
        MOCK_DATA_DIR.mkdir(exist_ok=True)
        logger.info(f"📁 Mock data directory: {MOCK_DATA_DIR}")

    def get_latest_snapshot(self) -> Dict[str, Any]:
        """
        Get the latest aggregated snapshot from mock data
        Returns today's data aggregated across all repos
        """
        metrics = self.get_latest_metrics()

        now = datetime.now()

        return {
            "ts": now.isoformat(),
            "team_id": self.config.team_id,
            "agent": "github",
            "window": "24h",
            "metrics": metrics,
            "ewma": {
                f"{key}_ewma": round(value * 0.9, 2)  # Mock EWMA
                for key, value in metrics.items()
            },
            "meta": {
                "repos_considered": len(self.config.github_repos),
                "source": "mock",
                "timezone": "Europe/London",
                "notes": "team aggregate, no PII"
            }
        }

    def get_latest_metrics(self) -> Dict[str, float]:
        """
        Aggregate latest metrics from all repos
        Returns team-level daily totals
        """
        all_data = self._read_all_csv_data()

        if not all_data:
            logger.warning("⚠️  No mock data found, returning zeros")
            return {
                "commits_total": 0.0,
                "after_hours_commits": 0.0,
                "prs_opened": 0.0,
                "prs_merged": 0.0,
                "median_pr_lead_time_hours": 0.0,
                "median_time_to_first_review_hours": 0.0,
                "reviews_count": 0.0
            }

        # Aggregate across all repos
        commits_total = sum(row["commits"] for row in all_data)
        after_hours = sum(row["after_hours_commits"] for row in all_data)
        prs_opened = sum(row["prs_opened"] for row in all_data)
        prs_merged = sum(row["prs_merged"] for row in all_data)
        median_pr_lead = median(
            [row["median_pr_lead_time_hours"] for row in all_data if row["median_pr_lead_time_hours"] > 0]) if any(
            row["median_pr_lead_time_hours"] > 0 for row in all_data) else 0.0
        median_review_time = median([row["median_time_to_first_review_hours"] for row in all_data if
                                     row["median_time_to_first_review_hours"] > 0]) if any(
            row["median_time_to_first_review_hours"] > 0 for row in all_data) else 0.0
        reviews = sum(row["reviews_count"] for row in all_data)

        return {
            "commits_total": float(commits_total),
            "after_hours_commits": float(after_hours),
            "prs_opened": float(prs_opened),
            "prs_merged": float(prs_merged),
            "median_pr_lead_time_hours": float(median_pr_lead),
            "median_time_to_first_review_hours": float(median_review_time),
            "reviews_count": float(reviews)
        }

    def _read_all_csv_data(self) -> List[Dict[str, Any]]:
        """
        Read and aggregate data from all CSV files in mock_data
        """
        all_data = []

        if not MOCK_DATA_DIR.exists():
            logger.warning(f"Mock data directory does not exist: {MOCK_DATA_DIR}")
            return all_data

        csv_files = list(MOCK_DATA_DIR.glob("*.csv"))
        logger.info(f"📊 Found {len(csv_files)} CSV files in mock_data")

        for csv_file in csv_files:
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_data.append({
                            "date": row.get("date", ""),
                            "repo": row.get("repo", ""),
                            "commits": int(row.get("commits", 0)),
                            "after_hours_commits": int(row.get("after_hours_commits", 0)),
                            "prs_opened": int(row.get("prs_opened", 0)),
                            "prs_merged": int(row.get("prs_merged", 0)),
                            "median_pr_lead_time_hours": float(row.get("median_pr_lead_time_hours", 0)),
                            "median_time_to_first_review_hours": float(row.get("median_time_to_first_review_hours", 0)),
                            "reviews_count": int(row.get("reviews_count", 0))
                        })
            except Exception as e:
                logger.error(f"❌ Error reading {csv_file}: {str(e)}")

        return all_data

