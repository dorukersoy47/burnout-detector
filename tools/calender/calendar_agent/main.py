from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json, datetime
from pathlib import Path
from typing import List, Dict

app = FastAPI(title="Calendar Agent")

OVERLOAD_HOURS = 6
BACK_TO_BACK_GAP_MIN = 15

# Be explicit about the path
DATA_FILE = (Path(__file__).resolve().parent / "sample_schedule.json")

def _ensure_meeting_fields(m: Dict) -> Dict:
    # coerce to ints and fill missing fields robustly
    for k in ("start", "end", "duration"):
        if k in m and m[k] is not None:
            m[k] = int(m[k])
    if "duration" not in m:
        if "start" in m and "end" in m:
            m["duration"] = int(m["end"]) - int(m["start"])
        else:
            raise ValueError(f"Meeting missing duration or (start/end): {m}")
    if "end" not in m:
        if "start" in m:
            m["end"] = int(m["start"]) + int(m["duration"])
        else:
            raise ValueError(f"Meeting missing end and cannot compute: {m}")
    if "start" not in m:
        raise ValueError(f"Meeting missing start: {m}")
    return m

def analyze_schedule(meetings: List[Dict], target_date: str | None = None) -> Dict:
    target_date = target_date or datetime.date.today().isoformat()
    todays = [m for m in meetings if m.get("date") == target_date]

    if not todays:
        return {
            "agent": "calendar",
            "date": target_date,
            "meetings_count": 0,
            "meetings_hours": 0.0,
            "back_to_back": False,
            "largest_gap_min": 24 * 60,
            "meeting_overload": False
        }

    clean = [_ensure_meeting_fields(m.copy()) for m in todays]
    clean.sort(key=lambda m: m["start"])

    total_min = sum(m["duration"] for m in clean)
    gaps = [clean[i+1]["start"] - clean[i]["end"] for i in range(len(clean) - 1)]
    back_to_back = any(gap < BACK_TO_BACK_GAP_MIN for gap in gaps)
    largest_gap = max(gaps) if gaps else 0

    hours = round(total_min / 60.0, 1)
    overload = (hours > OVERLOAD_HOURS) or back_to_back

    return {
        "agent": "calendar",
        "date": target_date,
        "meetings_count": len(clean),
        "meetings_hours": hours,
        "back_to_back": back_to_back,
        "largest_gap_min": largest_gap,
        "meeting_overload": overload
    }

@app.get("/")
def home():
    return {"message": "Calendar agent is running", "data_file": str(DATA_FILE)}

@app.get("/calendar/today")
def get_calendar_today(date: str | None = None):
    try:
        if not DATA_FILE.exists():
            return JSONResponse(status_code=500, content={"error": f"sample_schedule.json not found at {DATA_FILE}"})
        with open(DATA_FILE) as f:
            meetings = json.load(f)
        if not isinstance(meetings, list):
            return JSONResponse(status_code=500, content={"error": "sample_schedule.json must be a JSON list"})
        return analyze_schedule(meetings, target_date=date)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=500, content={"error": f"Invalid JSON: {e}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
