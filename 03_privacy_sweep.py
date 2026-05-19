"""
Script 03: Differential Privacy Sweep — epsilon 0.5, 1.0, 2.0, 5.0, inf
=========================================================================
Trains the FL model with Opacus DP at 5 different epsilon values.
Produces the Privacy-Accuracy trade-off curve (Figure 1).

Output: results/privacy_sweep.json
Runtime: ~2-3 hours total (runs 5 separate FL simulations)
"""

import os, json, time, copy
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from opacus import PrivacyEngine
import flwr as fl

os.makedirs("./results", exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_CLIENTS    = 10
N_ROUNDS     = 40
LOCAL_EPOCHS = 5
BATCH_SIZE   = 16
LR           = 0.001
DELTA        = 1e-5
MAX_GRAD_NORM = 1.0
RANDOM_SEED  = 42

# The 5 epsilon values to sweep
# inf = no DP (baseline reference), 0.5 = strongest privacy
EPSILONS = [float("inf"), 5.0, 2.0, 1.0, 0.5]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── LOAD CLIENT DATA ──────────────────────────────────────────────────────────
print("Loading client data...")
data = np.load("./data/clients.npz")
clients = []
for i in range(N_CLIENTS):
    clients.append({
        "X": data[f"client_{i}_X"],
        "y": data[f"client_{i}_y"]
    })
print(f"Loaded {N_CLIENTS} clients.\n")

# ── MODEL ─────────────────────────────────────────────────────────────────────
class ECGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.classifier(self.conv_block(x)).squeeze(1)

# ── DATA HELPER ───────────────────────────────────────────────────────────────
def make_loaders(client_data):
    X = client_data["X"]
    y = client_data["y"]
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2,
                                      random_state=RANDOM_SEED, stratify=y)
    def to_loader(indices, shuffle):
        Xt = torch.tensor(X[indices]).permute(0, 2, 1)
        yt = torch.tensor(y[indices], dtype=torch.float32)
        return DataLoader(TensorDataset(Xt, yt),
                          batch_size=BATCH_SIZE, shuffle=shuffle)
    return to_loader(tr_idx, True), to_loader(te_idx, False)

# ── FLOWER CLIENT (with optional DP) ─────────────────────────────────────────
class ECGClientDP(fl.client.NumPyClient):

    def __init__(self, client_data, epsilon):
        self.model     = ECGCNN()
        pos_weight     = torch.tensor([0.6])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.epsilon   = epsilon
        self.train_loader_raw, self.test_loader = make_loaders(client_data)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        # Attach Opacus DP engine only when epsilon is finite
        if epsilon != float("inf"):
            privacy_engine = PrivacyEngine()
            (self.model,
             self.optimizer,
             self.train_loader) = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader_raw,
                epochs=LOCAL_EPOCHS * N_ROUNDS,
                target_epsilon=epsilon,
                target_delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
            )
        else:
            self.train_loader = self.train_loader_raw

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, params):
        for p, v in zip(self.model.parameters(), params):
            p.data = torch.tensor(v)

    def fit(self, params, config):
        self.set_parameters(params)
        self.model.train()
        for _ in range(LOCAL_EPOCHS):
            for xb, yb in self.train_loader:
                self.optimizer.zero_grad()
                self.criterion(self.model(xb), yb).backward()
                self.optimizer.step()
        self.scheduler.step()
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, params, config):
        self.set_parameters(params)
        self.model.eval()
        preds, labels = [], []
        loss = 0.0
        with torch.no_grad():
            for xb, yb in self.test_loader:
                logits = self.model(xb)
                loss += self.criterion(logits, yb).item() * len(yb)
                pred = (torch.sigmoid(logits) > 0.5).int()
                preds  += pred.tolist()
                labels += yb.int().tolist()
        f1   = f1_score(labels, preds, zero_division=0)
        sens = recall_score(labels, preds, zero_division=0)
        spec = recall_score(labels, preds, pos_label=0, zero_division=0)
        return float(loss / max(1, len(labels))), len(labels), {
            "f1": float(f1),
            "sensitivity": float(sens),
            "specificity": float(spec),
        }

# ── RUN SWEEP ─────────────────────────────────────────────────────────────────
def start_client(cid, eps):
    import time
    time.sleep(cid * 0.5)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            fl.client.start_numpy_client(
                server_address="127.0.0.1:8080",
                client=ECGClientDP(clients[cid], eps)
            )
            break
        except Exception as e:
            time.sleep(3)

def run_server(strategy, n_rounds, eps_label):
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
    )
    
    # Save the history to a temporary file
    metrics = history.metrics_distributed
    
    f1_rounds   = [v for _, v in metrics.get("f1", [])]
    sens_rounds = [v for _, v in metrics.get("sensitivity", [])]
    spec_rounds = [v for _, v in metrics.get("specificity", [])]

    final_f1   = f1_rounds[-1]   if f1_rounds   else 0.0
    final_sens = sens_rounds[-1] if sens_rounds else 0.0
    final_spec = spec_rounds[-1] if spec_rounds else 0.0
    
    with open(f"temp_metrics_{eps_label}.json", "w") as f:
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
    all_results = {}
    if os.path.exists("./results/privacy_sweep.json"):
        with open("./results/privacy_sweep.json", "r") as f:
            try:
                all_results = json.load(f)
            except:
                pass

    for eps in EPSILONS:
        eps_label = "inf" if eps == float("inf") else str(eps)
        print(f"\n{'='*60}")
        print(f"Running FL with epsilon = {eps_label}")
        print(f"{'='*60}")

        start = time.time()

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

        server_process = None
        client_processes = []

        try:
            server_process = mp.Process(target=run_server, args=(strategy, N_ROUNDS, eps_label))
            server_process.start()

            time.sleep(3)

            for i in range(N_CLIENTS):
                p = mp.Process(target=start_client, args=(i, eps))
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
        temp_file = f"temp_metrics_{eps_label}.json"
        
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

        print(f"  epsilon={eps_label}: F1={final_f1:.4f} | Sens={final_sens:.4f} | Spec={final_spec:.4f} | {elapsed/60:.1f} min")

        all_results[eps_label] = {
            "epsilon": eps_label,
            "final_f1": final_f1,
            "final_sensitivity": final_sens,
            "final_specificity": final_spec,
            "elapsed_minutes": elapsed / 60,
            "f1_per_round": f1_rounds,
            "sensitivity_per_round": sens_rounds,
        }

    # ── SAVE ──────────────────────────────────────────────────────────────────────
    with open("./results/privacy_sweep.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("PRIVACY SWEEP COMPLETE - Summary:")
    print(f"{'Epsilon':>8} | {'F1':>6} | {'Sensitivity':>11} | {'Specificity':>11}")
    print("-" * 45)
    for eps_label, r in all_results.items():
        print(f"{'epsilon='+eps_label:>8} | {r['final_f1']:>6.4f} | {r['final_sensitivity']:>11.4f} | {r['final_specificity']:>11.4f}")

    print(f"\nSaved to ./results/privacy_sweep.json")
    print("Run 04_lmic_dropout.py next.")
