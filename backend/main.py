from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import io
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from engine import classify_log
from whitelist.manager import add_ip, remove_ip, add_country, remove_country, get_whitelist_stats
from blacklist.manager import add_blacklist_ip, remove_blacklist_ip, get_blacklist_stats
from retrain import retrain_model
from utils.logger import log_feedback
from utils.metadata import load_metadata
from utils.metadata import get_hit_counters
import json
from typing import Optional
from pathlib import Path

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Models ====
class LogEntry(BaseModel):
    timestamp: Optional[str] = "01-01-2025 12:00"
    src_ip: str
    dst_ip: str
    src_port: Optional[int] = 0
    dst_port: Optional[int] = 0
    protocol: Optional[int] = 0
    bytes_sent: Optional[int] = 0
    bytes_received: Optional[int] = 0
    flags: Optional[int] = 0
    duration: Optional[float] = 0.0

class WhitelistInput(BaseModel):
    item: str

class FeedbackEntry(BaseModel):
    log: LogEntry
    action: int  # 0 = allow, 1 = flagged

@app.get("/")
def root():
    return {"message": "Welcome to Network Security AI Engine API"}

@app.post("/classify")
def classify(entry: LogEntry):
    try:
        result = classify_log(entry.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Whitelist Routes
@app.post("/whitelist/ip/add")
def add_ip_whitelist(data: WhitelistInput):
    if add_ip(data.item):
        return {"message": f"IP '{data.item}' added to whitelist"}
    raise HTTPException(status_code=400, detail="Failed to add IP")

@app.post("/whitelist/ip/remove")
def remove_ip_whitelist(data: WhitelistInput):
    if remove_ip(data.item):
        return {"message": f"IP '{data.item}' removed from whitelist"}
    raise HTTPException(status_code=400, detail="Failed to remove IP")

@app.post("/whitelist/country/add")
def add_country_whitelist(data: WhitelistInput):
    if add_country(data.item):
        return {"message": f"Country '{data.item}' added to whitelist"}
    raise HTTPException(status_code=400, detail="Failed to add country")

@app.post("/whitelist/country/remove")
def remove_country_whitelist(data: WhitelistInput):
    if remove_country(data.item):
        return {"message": f"Country '{data.item}' removed from whitelist"}
    raise HTTPException(status_code=400, detail="Failed to remove country")

# Blacklist Routes
@app.post("/blacklist/ip/add")
def add_blacklist(data: WhitelistInput):
    if add_blacklist_ip(data.item):
        return {"message": f"IP '{data.item}' added to blacklist"}
    raise HTTPException(status_code=400, detail="Failed to add IP")

@app.post("/blacklist/ip/remove")
def remove_blacklist(data: WhitelistInput):
    if remove_blacklist_ip(data.item):
        return {"message": f"IP '{data.item}' removed from blacklist"}
    raise HTTPException(status_code=400, detail="Failed to remove IP")

# Retrain model with feedback
@app.post("/retrain")
def retrain():
    try:
        retrain_model()
        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def submit_feedback(entry: FeedbackEntry):
    try:
        log_feedback(entry.log.dict(), entry.action)
        retrain_model()
        return {"message": "Feedback logged and model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/classify-csv")
async def classify_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        required_cols = [
            "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
            "protocol", "bytes_sent", "bytes_received", "flags", "duration"
        ]

        for col in required_cols:
            if col not in df.columns:
                if col == "timestamp":
                    df[col] = "01-01-2025 12:00"
                elif col in ["src_ip", "dst_ip"]:
                    df[col] = "0.0.0.0"
                elif col == "duration":
                    df[col] = 0.0
                else:
                    df[col] = 0

        results = []
        for _, row in df.iterrows():
            log = row[required_cols].to_dict()
            result = classify_log(log)
            results.append(result["decision"])

        df["action_predicted"] = results
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(output, media_type="text/csv", headers={
            "Content-Disposition": "attachment; filename=classified_logs.csv"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {e}")

@app.get("/status")
def get_system_status():
    try:
        metadata = load_metadata()
        whitelist_stats = get_whitelist_stats()
        blacklist_stats = get_blacklist_stats()
        hit_counters = get_hit_counters()

        feedback_count = 0
        try:
            with open("logs/feedback_log.json", "r") as f:
                feedback_count = len(json.load(f))
        except:
            pass

        return {
            "model_info": metadata,
            "feedback_count": feedback_count,
            "whitelist_stats": {
                **whitelist_stats,
                "hits": hit_counters.get("whitelist", 0)
            },
            "blacklist_stats": {
                **blacklist_stats,
                "hits": hit_counters.get("blacklist", 0)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
def get_traffic_logs():
    try:
        path = Path("logs/traffic_log.json")
        if not path.exists():
            return {"logs": []}
        with open(path, "r") as f:
            data = json.load(f)
        return {"logs": list(reversed(data))}  # newest first
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/logs")
def clear_traffic_logs():
    try:
        path = Path("logs/traffic_log.json")
        if path.exists():
            with open(path, "w") as f:
                json.dump([], f)
        return {"message": "Logs cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))