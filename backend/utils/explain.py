import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from preprocess import preprocess_log
import shap

# Define model architecture
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load model and scaler
MODEL_PATH = Path("model/model_with_ewc.pt")
SCALER_PATH = Path("model/scaler.pkl")

scaler = joblib.load(SCALER_PATH)
input_dim = len(scaler.feature_names_in_)
model = MLP(input_dim=input_dim)
checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# SHAP Explainer Setup
explainer = shap.GradientExplainer((model, model.fc1), torch.tensor(np.zeros((1, input_dim)), dtype=torch.float32))

FEATURE_DESCRIPTIONS = {
    "src_port": "Source Port",
    "dst_port": "Destination Port",
    "protocol": "Protocol (6=TCP, 17=UDP, etc.)",
    "bytes_sent": "Bytes Sent",
    "bytes_received": "Bytes Received",
    "flags": "TCP Flags (control bits)",
    "duration": "Connection Duration (seconds)",
    "hour": "Hour of Day (0-23)"
}

PORT_EXPLANATIONS = {
    22: "SSH (Secure Shell)",
    23: "Telnet",
    80: "HTTP",
    443: "HTTPS",
    3389: "RDP (Remote Desktop)",
    5900: "VNC (Remote Access)"
}

def get_port_explanation(port: int) -> str:
    if port in PORT_EXPLANATIONS:
        return PORT_EXPLANATIONS[port]
    if port < 1024:
        return "Well-known system port"
    if 1024 <= port <= 49151:
        return "Registered user port"
    return "Dynamic/private port"

def get_flag_explanation(flags: int) -> str:
    flag_meanings = {
        0x01: ("FIN", "Connection termination"),
        0x02: ("SYN", "Connection initiation/synchronization"),
        0x04: ("RST", "Reset connection (abrupt termination)"),
        0x08: ("PSH", "Push data immediately (don't buffer)"),
        0x10: ("ACK", "Acknowledgment of received data"),
        0x20: ("URG", "Urgent data present (out-of-band)")
    }
    
    common_combinations = {
        0x12: "SYN-ACK (Connection establishment response)",
        0x10: "ACK (Normal data transmission)",
        0x14: "RST-ACK (Reset with acknowledgment)",
        0x11: "FIN-ACK (Graceful connection termination)",
        0x18: "PSH-ACK (Data push with acknowledgment)",
        0x04: "RST (Abrupt connection reset)",
        0x02: "SYN (Connection request)",
        0x01: "FIN (Connection termination request)"
    }
    
    if flags in common_combinations:
        return common_combinations[flags]
    
    active_flags = []
    descriptions = []
    
    for bit, (name, desc) in flag_meanings.items():
        if flags & bit:
            active_flags.append(name)
            descriptions.append(f"{name}: {desc}")
    
    if not active_flags:
        return "No flags set (unusual for TCP)"
    
    flag_combo = "-".join(active_flags)
    
    if len(descriptions) == 1:
        return f"{flag_combo}: {descriptions[0].split(':')[1].strip()}"
    
    detailed_desc = f"{flag_combo} flags indicate: " + "; ".join(descriptions)
    
    if flags == 0x20:
        return "URG: Urgent data present (rare in modern networks)"
    elif flags == 0x24:
        return "RST-URG: Reset with urgent data (very unusual)"
    
    return detailed_desc

def format_bytes(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def explain_decision_nn(log: dict) -> dict:
    try:
        features_df = preprocess_log(log)
        X_scaled = scaler.transform(features_df)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = model(X_tensor)
        confidence = output.item()
        predicted_class = int(confidence > 0.5)

        # SHAP values
        shap_output = explainer.shap_values(X_tensor)

        if isinstance(shap_output, list) and isinstance(shap_output[0], np.ndarray):
            shap_values = shap_output[0][0]
        elif isinstance(shap_output, np.ndarray):
            shap_values = shap_output[0]
        else:
            raise ValueError(f"Unexpected SHAP output format: {type(shap_output)}")

        explanations = []
        for i, col in enumerate(features_df.columns):
            val = float(features_df[col].iloc[0])
            shap_val = shap_values[i]
            explanation = {
                "feature": FEATURE_DESCRIPTIONS.get(col, col),
                "value": val,
                "description": "",
                "importance": round(float(abs(shap_val)), 4)
            }

            if col == "dst_port":
                explanation["description"] = get_port_explanation(int(val))
            elif col == "flags":
                explanation["description"] = get_flag_explanation(int(val))
            elif col == "bytes_sent":
                explanation["description"] = f"Data volume: {format_bytes(val)}"
            elif col == "duration":
                explanation["description"] = f"{val:.2f} seconds"
            elif col == "hour":
                explanation["description"] = f"Hour {int(val)}"

            explanations.append(explanation)

        top_features = sorted(explanations, key=lambda x: x["importance"], reverse=True)[:3]
        summary = f"Classified as {'anomalous' if predicted_class else 'normal'}"
        if top_features:
            summary += " based on: " + "; ".join([f"{f['feature']} (importance={f['importance']})" for f in top_features])

        return {
            "decision": "REJECT" if predicted_class else "ALLOW",
            "confidence": round(confidence if predicted_class else 1 - confidence, 4),
            "summary": summary,
            "key_factors": top_features,
            "all_features": explanations
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "message": "SHAP explanation failed",
            "traceback": traceback.format_exc()
        }
