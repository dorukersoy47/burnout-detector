"""
GitHub feature extraction for team-level metrics
Handles API calls, pagination, rate limiting
All personally identifiable information is excluded
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo
from statistics import median
import aiohttp

from config import Config

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


class GitHubFeatureExtractor:
    """
    Extracts team-level metrics from GitHub API
    Strictly aggregates data - no per-user information retained
    """

    def __init__(self, config: Config):
        self.config = config
        self.tz = ZoneInfo("Europe/London")
        self.headers = {
            "Authorization": f"token {config.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

    async def extract_last_24h(self, session: aiohttp.ClientSession) -> Dict[str, float]:
        """
        Extract team metrics for the last 24 hours

        Returns:
        - commits_total: Total commits in 24h
        - after_hours_commits: Commits outside 09:00-18:00
        - prs_opened: PRs opened in 24h
        - prs_merged: PRs merged in 24h
        - median_pr_lead_time_hours: Median time from creation to merge
        - median_time_to_first_review_hours: Median time to first review
        - reviews_count: Total review comments in 24h
        """
        logger.info(f"🔄 Extracting metrics for repos: {self.config.github_repos}")

        now = datetime.now(self.tz)
        since = now - timedelta(hours=24)

        commits_total = 0
        after_hours_commits = 0
        prs_opened = 0
        prs_merged = 0
        pr_lead_times = []
        review_times = []
        reviews_count = 0

        for repo in self.config.github_repos:
            try:
                # Fetch commits
                commits = await self._fetch_commits(session, repo, since)
                commits_total += len(commits)
                after_hours_commits += self._count_after_hours(commits)

                # Fetch PRs
                prs = await self._fetch_prs(session, repo, since)
                prs_opened += len([pr for pr in prs if pr["created_at"] >= since.isoformat()])
                prs_merged += len([pr for pr in prs if pr.get("merged_at")])

                # Calculate PR lead times
                for pr in prs:
                    if pr.get("merged_at"):
                        created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00")).astimezone(self.tz)
                        merged = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00")).astimezone(self.tz)
                        hours = (merged - created).total_seconds() / 3600
                        pr_lead_times.append(hours)

                # Fetch reviews and calculate first review time
                for pr in prs:
                    reviews = await self._fetch_pr_reviews(session, repo, pr["number"], since)
                    reviews_count += len(reviews)

                    if reviews and pr.get("created_at"):
                        created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00")).astimezone(self.tz)
                        first_review = datetime.fromisoformat(
                            reviews[0]["submitted_at"].replace("Z", "+00:00")
                        ).astimezone(self.tz)
                        hours = (first_review - created).total_seconds() / 3600
                        review_times.append(hours)

            except Exception as e:
                logger.error(f"❌ Error extracting from {repo}: {str(e)}")
                continue

        return {
            "commits_total": float(commits_total),
            "after_hours_commits": float(after_hours_commits),
            "prs_opened": float(prs_opened),
            "prs_merged": float(prs_merged),
            "median_pr_lead_time_hours": float(median(pr_lead_times)) if pr_lead_times else 0.0,
            "median_time_to_first_review_hours": float(median(review_times)) if review_times else 0.0,
            "reviews_count": float(reviews_count)
        }

    async def _fetch_commits(
            self,
            session: aiohttp.ClientSession,
            repo: str,
            since: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch commits from a repo (author info excluded for privacy)"""
        url = f"{GITHUB_API_BASE}/repos/{self.config.github_org}/{repo}/commits"
        params = {
            "since": since.isoformat(),
            "per_page": 100
        }

        all_commits = []
        page = 1

        while page <= 10:  # Max 10 pages to prevent rate limiting
            try:
                async with session.get(url, headers=self.headers, params={**params, "page": page}) as resp:
                    if resp.status == 404:
                        logger.warning(f"⚠️  Repo not found: {repo}")
                        break
                    if resp.status == 403:
                        logger.warning(f"⚠️  Rate limited or no access to {repo}")
                        break

                    data = await resp.json()
                    if not data:
                        break

                    all_commits.extend(data)
                    page += 1
            except Exception as e:
                logger.error(f"Error fetching commits from {repo}: {str(e)}")
                break

        return all_commits

    async def _fetch_prs(
            self,
            session: aiohttp.ClientSession,
            repo: str,
            since: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch PRs from a repo"""
        url = f"{GITHUB_API_BASE}/repos/{self.config.github_org}/{repo}/pulls"
        params = {
            "state": "all",
            "sort": "updated",
            "direction": "desc",
            "per_page": 100
        }

        all_prs = []
        page = 1

        while page <= 10:
            try:
                async with session.get(url, headers=self.headers, params={**params, "page": page}) as resp:
                    if resp.status == 404:
                        break
                    if resp.status == 403:
                        logger.warning(f"⚠️  Rate limited on {repo}")
                        break

                    data = await resp.json()
                    if not data:
                        break

                    # Only include PRs from the last 24h
                    for pr in data:
                        created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
                        if created >= since:
                            all_prs.append(pr)

                    page += 1
            except Exception as e:
                logger.error(f"Error fetching PRs from {repo}: {str(e)}")
                break

        return all_prs

    async def _fetch_pr_reviews(
            self,
            session: aiohttp.ClientSession,
            repo: str,
            pr_number: int,
            since: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch reviews for a PR"""
        url = f"{GITHUB_API_BASE}/repos/{self.config.github_org}/{repo}/pulls/{pr_number}/reviews"
        params = {"per_page": 100}

        all_reviews = []
        page = 1

        while page <= 5:
            try:
                async with session.get(url, headers=self.headers, params={**params, "page": page}) as resp:
                    if resp.status in (404, 403):
                        break

                    data = await resp.json()
                    if not data:
                        break

                    for review in data:
                        if review.get("submitted_at"):
                            submitted = datetime.fromisoformat(
                                review["submitted_at"].replace("Z", "+00:00")
                            )
                            if submitted >= since:
                                all_reviews.append(review)

                    page += 1
            except Exception as e:
                logger.error(f"Error fetching reviews for PR {pr_number}: {str(e)}")
                break

        return all_reviews

    def _count_after_hours(self, commits: List[Dict[str, Any]]) -> int:
        """Count commits outside work hours (09:00-18:00)"""
        after_hours = 0

        for commit in commits:
            if not commit.get("commit", {}).get("author", {}).get("date"):
                continue

            try:
                commit_dt = datetime.fromisoformat(
                    commit["commit"]["author"]["date"].replace("Z", "+00:00")
                ).astimezone(self.tz)

                hour = commit_dt.hour
                weekday = commit_dt.weekday()

                # After hours = outside 09:00-18:00 OR weekends
                if hour < 9 or hour >= 18 or weekday >= 5:
                    after_hours += 1
            except Exception:
                continue

        return after_hours