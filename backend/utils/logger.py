import json
from pathlib import Path

FEEDBACK_PATH = Path("logs/feedback_log.json")

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
