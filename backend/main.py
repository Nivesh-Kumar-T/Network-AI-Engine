from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from engine import classify_log
from whitelist.manager import add_ip, remove_ip, add_country, remove_country
from blacklist.manager import add_blacklist_ip, remove_blacklist_ip
from retrain import retrain_model
from utils.logger import log_feedback

app = FastAPI()

# ==== Models ====
class LogEntry(BaseModel):
    timestamp: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    bytes_sent: int
    bytes_received: int
    flags: int
    duration: float

class WhitelistInput(BaseModel):
    item: str

class FeedbackEntry(BaseModel):
    log: LogEntry
    action: int  # 0 = normal, 1 = anomalous

# ==== Routes ====
@app.get("/")
def root():
    return {"message": "Welcome to Network AI Engine API"}

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