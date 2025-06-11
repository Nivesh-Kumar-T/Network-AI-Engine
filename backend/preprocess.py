import pandas as pd
from datetime import datetime

def preprocess_log(log: dict) -> pd.DataFrame:
    """
    Convert a full network log dictionary into a DataFrame with expected model features.
    """

    # Set defaults for missing keys
    defaults = {
        "timestamp": "01-01-2025 12:00",
        "src_port": 0,
        "dst_port": 0,
        "protocol": 0,
        "bytes_sent": 0,
        "bytes_received": 0,
        "flags": 0,
        "duration": 0.0
    }
    for key, default in defaults.items():
        log.setdefault(key, default)

    # Extract hour from timestamp in format DD-MM-YYYY HH:MM
    try:
        timestamp = pd.to_datetime(log["timestamp"], format="%d-%m-%Y %H:%M")
        hour = timestamp.hour
    except Exception:
        hour = 12  # fallback if timestamp is invalid

    # Prepare DataFrame with expected model features
    model_features = {
        "src_port": log["src_port"],
        "dst_port": log["dst_port"],
        "protocol": log["protocol"],
        "bytes_sent": log["bytes_sent"],
        "bytes_received": log["bytes_received"],
        "flags": log["flags"],
        "duration": log["duration"],
        "hour": hour
    }

    return pd.DataFrame([model_features])
