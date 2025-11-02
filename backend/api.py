"""FastAPI surface for Lighthouse.ai agents."""

from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Lighthouse.ai API", version="0.1.0")


@app.get("/api/investigation/latest/cot")
async def get_latest_investigation_cot():
    """Return the most recent chain-of-thought capture for visualization."""

    cot_file = Path(__file__).resolve().parent / "latest_investigation_cot.json"

    if not cot_file.exists():
        raise HTTPException(
            status_code=404,
            detail="No investigation data found. Run an investigation first."
        )

    try:
        with cot_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Investigation data is corrupted: {exc}. Run a new investigation."
        ) from exc
    except Exception as exc:  # pragma: no cover - generic safeguard
        raise HTTPException(
            status_code=500,
            detail=f"Error reading investigation data: {exc}"
        ) from exc

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

