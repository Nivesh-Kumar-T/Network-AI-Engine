import json
from pathlib import Path
from datetime import datetime

FEEDBACK_PATH = Path("logs/feedback_log.json")
TRAFFIC_LOG_PATH = Path("logs/traffic_log.json")

def log_feedback(log_entry: dict, action: int):
    feedback = {"log": log_entry, "action": action}
    data = []
    if FEEDBACK_PATH.exists():
        with open(FEEDBACK_PATH, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    data.append(feedback)
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(data, f, indent=2)


def log_traffic(log_entry: dict, decision: str, reason: str, confidence: float = None, explanation: dict = None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "log_timestamp": log_entry.get("timestamp", ""),
        "src_ip": log_entry.get("src_ip", ""),
        "dst_ip": log_entry.get("dst_ip", ""),
        "src_port": log_entry.get("src_port", 0),
        "dst_port": log_entry.get("dst_port", 0),
        "protocol": log_entry.get("protocol", 0),
        "bytes_sent": log_entry.get("bytes_sent", 0),
        "bytes_received": log_entry.get("bytes_received", 0),
        "flags": log_entry.get("flags", 0),
        "duration": log_entry.get("duration", 0),
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "explanation": explanation,   # ← ADD THIS
    }

    data = []
    if TRAFFIC_LOG_PATH.exists():
        with open(TRAFFIC_LOG_PATH, "r") as f:
            try:
                data = json.load(f)
            except:
                data = []

    data.append(entry)
    data = data[-200:]

    with open(TRAFFIC_LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)