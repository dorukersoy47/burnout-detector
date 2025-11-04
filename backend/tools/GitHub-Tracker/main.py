"""
GitHub Team Burnout Dashboard - FastAPI Application
Privacy-first team-level productivity monitoring
No per-user PII stored or returned
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
import uvicorn
import aiohttp
from pydantic import BaseModel, Field

from config import Config
from features import GitHubFeatureExtractor
from ewma import EWMACalculator
from mock_data import MockDataProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GitHub Team Burnout Dashboard",
    description="Privacy-first team-level productivity metrics",
    version="1.0.0"
)

# Global state
config: Optional[Config] = None
feature_extractor: Optional[GitHubFeatureExtractor] = None
ewma_calculator: Optional[EWMACalculator] = None
mock_provider: Optional[MockDataProvider] = None
latest_snapshot: Optional[Dict[str, Any]] = None


class PollResponse(BaseModel):
    """Response model for poll endpoint"""
    status: str = Field(..., description="Status of the poll operation")
    timestamp: str = Field(..., description="When the poll was executed")
    metrics_updated: bool = Field(..., description="Whether metrics were updated")
    team_id: str = Field(..., description="Team identifier")


class LatestSnapshot(BaseModel):
    """Response model for latest endpoint"""
    ts: str = Field(..., description="ISO 8601 timestamp")
    team_id: str = Field(..., description="Team identifier")
    agent: str = Field(default="github", description="Agent type")
    window: str = Field(default="24h", description="Time window")
    metrics: Dict[str, float] = Field(..., description="Team-aggregated metrics")
    ewma: Dict[str, float] = Field(..., description="EWMA smoothed metrics")
    meta: Dict[str, Any] = Field(..., description="Metadata")


@app.on_event("startup")
async def startup_event():
    """Initialize configuration and components on startup"""
    global config, feature_extractor, ewma_calculator, mock_provider, latest_snapshot

    logger.info("🚀 Starting GitHub Team Burnout Dashboard...")

    config = Config.from_env()
    logger.info(f"Config loaded: team_id={config.team_id}, mock={config.mock}, org={config.github_org}")

    ewma_calculator = EWMACalculator(alpha=0.3)

    if config.mock:
        logger.info("📊 Running in MOCK mode - using CSV data")
        mock_provider = MockDataProvider(config)
        # Load initial snapshot
        latest_snapshot = mock_provider.get_latest_snapshot()
    else:
        logger.info("🔗 Running in LIVE mode - will fetch from GitHub API")
        feature_extractor = GitHubFeatureExtractor(config)

    logger.info("✅ Dashboard initialized successfully")


@app.get("/")
async def root():
    """Redirect root to API documentation"""
    return RedirectResponse(url="/docs")


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    """
    Health check endpoint
    Returns 200 OK if the service is running
    """
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "timestamp": datetime.now(ZoneInfo("Europe/London")).isoformat()}
    )


@app.post("/poll")
async def poll() -> PollResponse:
    """
    Poll for last 24h data and update snapshot

    In MOCK mode: loads fresh CSV data
    In LIVE mode: fetches from GitHub API with rate limit handling

    Returns team-aggregated metrics (no PII)
    """
    global latest_snapshot, config, feature_extractor, mock_provider, ewma_calculator

    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not initialized")

    poll_time = datetime.now(ZoneInfo("UTC"))

    try:
        if config.mock:
            logger.info("📥 Poll: Loading mock data...")
            metrics = mock_provider.get_latest_metrics()
        else:
            logger.info("📥 Poll: Fetching live GitHub data...")
            async with aiohttp.ClientSession() as session:
                metrics = await feature_extractor.extract_last_24h(session)

        # Calculate EWMA for each metric
        ewma_values = {}
        if latest_snapshot and "ewma" in latest_snapshot:
            previous_ewma = latest_snapshot["ewma"]
        else:
            previous_ewma = {}

        for metric_key, metric_value in metrics.items():
            ewma_key = f"{metric_key}_ewma"
            previous = previous_ewma.get(ewma_key)
            ewma_values[ewma_key] = ewma_calculator.compute(metric_value, previous)

        # Build snapshot
        latest_snapshot = {
            "ts": poll_time.isoformat(),
            "team_id": config.team_id,
            "agent": "github",
            "window": "24h",
            "metrics": metrics,
            "ewma": ewma_values,
            "meta": {
                "repos_considered": len(config.github_repos),
                "source": "mock" if config.mock else "api",
                "timezone": "Europe/London",
                "notes": "team aggregate, no PII"
            }
        }

        logger.info(f"✅ Poll completed. Snapshot updated at {poll_time.isoformat()}")

        return PollResponse(
            status="success",
            timestamp=poll_time.isoformat(),
            metrics_updated=True,
            team_id=config.team_id
        )

    except Exception as e:
        logger.error(f"❌ Poll failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Poll failed: {str(e)}")


@app.get("/latest")
async def latest() -> LatestSnapshot:
    """
    Return the latest team-level daily snapshot

    No per-user PII included - only team aggregates
    Includes EWMA smoothed metrics for trend analysis
    """
    global latest_snapshot

    if latest_snapshot is None:
        raise HTTPException(
            status_code=404,
            detail="No snapshot available. Call POST /poll first."
        )

    return LatestSnapshot(**latest_snapshot)


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """
    Return raw metrics details (for debugging)
    """
    if latest_snapshot is None:
        raise HTTPException(status_code=404, detail="No snapshot available")

    return {
        "timestamp": latest_snapshot.get("ts"),
        "team_id": latest_snapshot.get("team_id"),
        "metrics": latest_snapshot.get("metrics"),
        "ewma_smoothed": latest_snapshot.get("ewma"),
        "cache_age_seconds": (
                datetime.fromisoformat(latest_snapshot["ts"]).timestamp() -
                datetime.now(ZoneInfo("UTC")).timestamp()
        )
    }


if __name__ == "__main__":
    # Load config to get host and port
    cfg = Config.from_env()
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        log_level="info"
    )