import torch
import pandas as pd
from pathlib import Path
import geoip2.database
from preprocess import preprocess_log
from blacklist.manager import is_blacklisted
from utils.explain import explain_decision_nn
import joblib
import torch.nn as nn

MODEL_PATH = Path("model/model_with_ewc.pt")
SCALER_PATH = Path("model/scaler.pkl")
GEOIP_DB_PATH = Path("GeoLite2-Country.mmdb")
IP_WHITELIST = Path("whitelist/ip_list.txt")
COUNTRY_WHITELIST = Path("whitelist/country_list.txt")

# === Load Scaler ===
scaler = joblib.load(SCALER_PATH)

# === Define MLP class ===
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

# === Load Model ===
input_dim = 8
model = MLP(input_dim)
checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# === Whitelist ===
def load_whitelist(path):
    if path.exists():
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def get_country_from_ip(ip: str) -> str:
    try:
        with geoip2.database.Reader(GEOIP_DB_PATH) as reader:
            response = reader.country(ip)
            return response.country.iso_code
    except Exception:
        return "UNKNOWN"

def is_whitelisted(ip: str) -> bool:
    ip_whitelist = load_whitelist(IP_WHITELIST)
    country_whitelist = load_whitelist(COUNTRY_WHITELIST)
    return ip in ip_whitelist or get_country_from_ip(ip) in country_whitelist

# === Main Inference ===
def classify_log(log: dict) -> dict:
    src_ip = log.get("src_ip", "")
    if is_blacklisted(src_ip):
        return {"decision": "REJECT", "reason": "Blacklisted"}
    if is_whitelisted(src_ip):
        return {"decision": "ALLOW", "reason": "Whitelisted"}

    features_df = preprocess_log(log)
    scaled = scaler.transform(features_df)
    tensor_input = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        output_tensor = model(tensor_input)
        confidence = output_tensor.item()
        predicted = int(confidence > 0.5)

    explanation = explain_decision_nn(log)

    decision = "REJECT" if predicted else "ALLOW"
    reason = f"{'Flagged' if predicted else 'Allowed'} with confidence {round(confidence if predicted else 1 - confidence, 4)}"

    return {
        "decision": decision,
        "reason": reason,
        "explanation": explanation
    }