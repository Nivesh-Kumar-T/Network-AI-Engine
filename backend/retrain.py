import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
import json
from pathlib import Path
from preprocess import preprocess_log

MODEL_PATH = Path("model/model_with_ewc.pt")
SCALER_PATH = Path("model/scaler.pkl")
FEEDBACK_PATH = Path("logs/feedback_log.json")

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

def load_feedback():
    if not FEEDBACK_PATH.exists():
        return [], []
    with open(FEEDBACK_PATH, "r") as f:
        entries = json.load(f)
    X, y = [], []
    for entry in entries:
        df = preprocess_log(entry["log"])
        X.append(df.iloc[0])
        y.append(entry["action"])
    return pd.DataFrame(X), pd.Series(y)

def ewc_penalty(model, fisher, opt_params, lamda=1000):
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher:
            loss += (fisher[name] * (param - opt_params[name])**2).sum()
    return lamda * loss

def retrain_model():
    input_dim = 8
    checkpoint = torch.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    model = MLP(input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    fisher = checkpoint["fisher"]
    opt_params = checkpoint["opt_params"]

    model.train()
    X_new, y_new = load_feedback()
    if X_new.empty:
        print("[INFO] No feedback to retrain.")
        return

    X_scaled = scaler.transform(X_new)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_new.values, dtype=torch.float32).unsqueeze(1)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    bce_loss = nn.BCELoss()

    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = bce_loss(output, y_tensor) + ewc_penalty(model, fisher, opt_params)
        loss.backward()
        optimizer.step()

    torch.save({
        'model_state_dict': model.state_dict(),
        'fisher': fisher,
        'opt_params': opt_params
    }, MODEL_PATH)

    print("[INFO] Retraining complete.")