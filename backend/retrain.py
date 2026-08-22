import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
import json
from pathlib import Path
from preprocess import preprocess_log
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from utils.metadata import append_retrain_entry

MODEL_PATH = Path("model/model_with_ewc.pt")
SCALER_PATH = Path("model/scaler.pkl")
FEEDBACK_PATH = Path("logs/feedback_log.json")
TEST_PATH = Path("logs/test_set.csv")

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
            loss += (fisher[name] * (param - opt_params[name]) ** 2).sum()
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

    accuracy = precision = recall = f1 = 0.0
    if TEST_PATH.exists():
        test_df = pd.read_csv(TEST_PATH)

        if 'hour' not in test_df.columns:
            test_df["hour"] = pd.to_datetime(test_df["timestamp"]).dt.hour

        feature_cols = ["src_port", "dst_port", "protocol", "bytes_sent",
                        "bytes_received", "flags", "duration", "hour"]
        X_test = test_df[feature_cols]
        y_test = test_df["action"].map({"allow": 0, "flagged": 1})

        valid_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
        dropped_rows = (~valid_mask).sum()
        X_test = X_test[valid_mask]
        y_test = y_test[valid_mask]

        if not X_test.empty:
            X_test_scaled = scaler.transform(X_test)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

            model.eval()
            with torch.no_grad():
                preds = model(X_test_tensor)
                preds_binary = (preds > 0.5).float()

            accuracy = accuracy_score(y_test_tensor, preds_binary)
            precision = precision_score(y_test_tensor, preds_binary, zero_division=0)
            recall = recall_score(y_test_tensor, preds_binary, zero_division=0)
            f1 = f1_score(y_test_tensor, preds_binary, zero_division=0)

            print(f"[INFO] Evaluated on test set. Dropped {dropped_rows} invalid rows.")
        else:
            print("[WARNING] All rows in test set have missing data after filtering.")

    retrain_entry = {
        "timestamp": datetime.now().isoformat(),
        "feedback_samples": len(X_new),
        "epochs": 10,
        "metrics": {
            "loss": round(loss.item(), 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }
    }

    append_retrain_entry(retrain_entry)
    print("[INFO] Retraining complete and metrics logged.")