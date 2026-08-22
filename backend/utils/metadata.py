import json
from pathlib import Path
from datetime import datetime

METADATA_PATH = Path("logs/metadata.json")

def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {}

def save_metadata(data):
    with open(METADATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

def append_retrain_entry(entry):
    data = load_metadata()
    if "retraining_history" not in data:
        data["retraining_history"] = []
    data["retraining_history"].append(entry)
    save_metadata(data)

def increment_hit_counter(category: str):
    data = load_metadata()
    if "hit_counters" not in data:
        data["hit_counters"] = {"whitelist": 0, "blacklist": 0}
    if category in data["hit_counters"]:
        data["hit_counters"][category] += 1
    else:
        data["hit_counters"][category] = 1
    save_metadata(data)

def get_hit_counters():
    data = load_metadata()
    return data.get("hit_counters", {"whitelist": 0, "blacklist": 0})