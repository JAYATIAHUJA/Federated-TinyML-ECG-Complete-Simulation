"""
Script 02: Baseline Federated Learning — NO Differential Privacy
================================================================
Trains the 1D-CNN ECG classifier using FedAvg across 10 clients.
No DP noise. This establishes the accuracy ceiling.

Output: results/baseline.json
"""

import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import flwr as fl
import multiprocessing as mp

os.makedirs("./results", exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_CLIENTS    = 10
N_ROUNDS     = 40
LOCAL_EPOCHS = 5
BATCH_SIZE   = 16
LR           = 0.001
RANDOM_SEED  = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── LOAD CLIENT DATA ──────────────────────────────────────────────────────────
print("Loading client data...")
data = np.load("./data/clients.npz")
clients = []
for i in range(N_CLIENTS):
    X = data[f"client_{i}_X"]   # (N, 1000, 12)
    y = data[f"client_{i}_y"]   # (N,)
    clients.append({"X": X, "y": y})
print(f"Loaded {N_CLIENTS} clients.")

# ── MODEL DEFINITION ──────────────────────────────────────────────────────────
class ECGCNN(nn.Module):
    """
    Lightweight 1D-CNN for binary ECG classification.
    ~48,000 parameters. Fits in INT8 within 256 KB SRAM.
    Input:  (batch, 12, 1000)   — 12 leads, 1000 time steps
    Output: (batch,)            — logit for ABNORM class
    """
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool1d(2),                             # -> (32, 500)

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool1d(2),                             # -> (64, 250)

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                     # -> (64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x)).squeeze(1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())

model_check = ECGCNN()
n_params = count_params(model_check)
print(f"Model params: {n_params:,} | FP32 size: {n_params*4/1024:.1f} KB")

# ── DATA HELPER ───────────────────────────────────────────────────────────────
def make_loaders(client_data, seed=42):
    X = client_data["X"]   # (N, 1000, 12)
    y = client_data["y"]

    # Train/test split
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2,
                                      random_state=seed, stratify=y)

    def to_loader(indices, shuffle):
        Xt = torch.tensor(X[indices]).permute(0, 2, 1)  # -> (N, 12, 1000)
        yt = torch.tensor(y[indices], dtype=torch.float32)
        ds = TensorDataset(Xt, yt)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return to_loader(tr_idx, True), to_loader(te_idx, False)


# ── FLOWER CLIENT ─────────────────────────────────────────────────────────────
class ECGClient(fl.client.NumPyClient):

    def __init__(self, client_data, client_id):
        self.client_id  = client_id
        self.model      = ECGCNN().to(DEVICE)
        pos_weight = torch.tensor([0.6]).to(DEVICE)
        self.criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.train_loader, self.test_loader = make_loaders(client_data)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, params):
        for p, v in zip(self.model.parameters(), params):
            p.data = torch.tensor(v, dtype=p.dtype, device=DEVICE)
        self.model.to(DEVICE)

    def fit(self, params, config):
        self.set_parameters(params)
        self.model.train()
        total_loss = 0.0
        for _ in range(LOCAL_EPOCHS):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        self.scheduler.step()
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, params, config):
        self.set_parameters(params)
        self.model.eval()
        preds, labels = [], []
        loss = 0.0
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = self.model(xb)
                loss += self.criterion(logits, yb).item() * len(yb)
                pred   = (torch.sigmoid(logits) > 0.5).int()
                preds  += pred.cpu().tolist()
                labels += yb.int().cpu().tolist()

        f1   = f1_score(labels, preds, zero_division=0)
        sens = recall_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)
        spec = recall_score(labels, preds, pos_label=0, zero_division=0)

        return float(loss / max(1, len(labels))), len(labels), {
            "f1": float(f1),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "precision": float(prec),
        }


def start_client(cid):
    import time
    time.sleep(cid * 0.5)  # Stagger client GPU initialization
    max_retries = 5
    for attempt in range(max_retries):
        try:
            fl.client.start_numpy_client(
                server_address="127.0.0.1:8080",
                client=ECGClient(clients[cid], cid)
            )
            break  # Exit loop if client finishes successfully
        except Exception as e:
            time.sleep(3)

def run_server(strategy, n_rounds):
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
    )
    
    metrics = history.metrics_distributed
    
    f1_rounds   = [v for _, v in metrics.get("f1", [])]
    sens_rounds = [v for _, v in metrics.get("sensitivity", [])]
    spec_rounds = [v for _, v in metrics.get("specificity", [])]

    final_f1   = f1_rounds[-1]   if f1_rounds   else 0.0
    final_sens = sens_rounds[-1] if sens_rounds else 0.0
    final_spec = spec_rounds[-1] if spec_rounds else 0.0
    
    with open("temp_metrics_baseline.json", "w") as f:
        json.dump({
            "final_f1": final_f1,
            "final_sensitivity": final_sens,
            "final_specificity": final_spec,
            "f1_per_round": f1_rounds,
            "sensitivity_per_round": sens_rounds,
        }, f)

def on_evaluate_config(server_round):
    return {"round": server_round}

def weighted_average(metrics):
    if not metrics:
        return {"f1": 0.0, "sensitivity": 0.0, "specificity": 0.0}
    f1 = sum(num_examples * m["f1"] for num_examples, m in metrics) / sum(num_examples for num_examples, m in metrics)
    sens = sum(num_examples * m["sensitivity"] for num_examples, m in metrics) / sum(num_examples for num_examples, m in metrics)
    spec = sum(num_examples * m["specificity"] for num_examples, m in metrics) / sum(num_examples for num_examples, m in metrics)
    return {"f1": f1, "sensitivity": sens, "specificity": spec}

if __name__ == "__main__":
    mp.freeze_support()
    print(f"\nStarting Federated Learning — {N_ROUNDS} rounds, {N_CLIENTS} clients")
    print("=" * 60)

    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=N_CLIENTS,
        min_evaluate_clients=N_CLIENTS,
        min_available_clients=N_CLIENTS,
        on_evaluate_config_fn=on_evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        proximal_mu=0.1,
    )

    start = time.time()

    server_process = None
    client_processes = []

    try:
        server_process = mp.Process(target=run_server, args=(strategy, N_ROUNDS))
        server_process.start()

        time.sleep(3)

        for i in range(N_CLIENTS):
            p = mp.Process(target=start_client, args=(i,))
            p.start()
            client_processes.append(p)

        server_process.join()
        for p in client_processes:
            p.join()
            
    except KeyboardInterrupt:
        pass
    finally:
        if server_process and server_process.is_alive():
            server_process.terminate()
            server_process.join()
        for p in client_processes:
            if p.is_alive():
                p.terminate()
                p.join()

    elapsed = time.time() - start

    # Load metrics from temp file
    temp_file = "temp_metrics_baseline.json"
    
    final_f1, final_sens, final_spec = 0.0, 0.0, 0.0
    f1_rounds, sens_rounds = [], []
    if os.path.exists(temp_file):
        with open(temp_file, "r") as f:
            data = json.load(f)
            final_f1 = data.get("final_f1", 0.0)
            final_sens = data.get("final_sensitivity", 0.0)
            final_spec = data.get("final_specificity", 0.0)
            f1_rounds = data.get("f1_per_round", [])
            sens_rounds = data.get("sensitivity_per_round", [])
        os.remove(temp_file)

    results_data = {
        "final_f1": final_f1,
        "final_sensitivity": final_sens,
        "final_specificity": final_spec,
        "elapsed_minutes": elapsed / 60,
        "f1_per_round": f1_rounds,
        "sensitivity_per_round": sens_rounds,
    }

    with open("./results/baseline.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"BASELINE RESULTS (No DP, No Dropout)")
    print(f"  F1:          {final_f1:.4f}")
    print(f"  Sensitivity: {final_sens:.4f}")
    print(f"  Specificity: {final_spec:.4f}")
    print(f"  Time:        {elapsed/60:.1f} minutes")
    print(f"\nSaved to ./results/baseline.json")
    print("Run 03_privacy_sweep.py next.")
