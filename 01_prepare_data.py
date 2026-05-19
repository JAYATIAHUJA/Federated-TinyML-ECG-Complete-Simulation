"""
Script 01: Load PTB-XL and prepare federated client splits
==========================================================
Run this FIRST before any other script.
Output: data/clients.npz  (10 client datasets)
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH   = "./ptb-xl-1.0.3/"   # <-- folder where you extracted PTB-XL
OUTPUT_DIR  = "./data/"
N_CLIENTS   = 10
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── STEP 1: Load metadata CSV ─────────────────────────────────────────────────
print("Loading PTB-XL metadata...")
Y = pd.read_csv(DATA_PATH + "ptbxl_database.csv", index_col="ecg_id")
Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

# ── STEP 2: Load ECG waveforms (100Hz) ───────────────────────────────────────
print(f"Loading {len(Y)} ECG waveforms at 100Hz... (this takes 10-20 min)")
X = []
skipped = 0
for i, (ecg_id, row) in enumerate(Y.iterrows()):
    if i % 1000 == 0:
        print(f"  {i}/{len(Y)} loaded...")
    try:
        signal, _ = wfdb.rdsamp(DATA_PATH + row.filename_lr)
        X.append(signal)   # shape: (1000, 12)
    except Exception as e:
        X.append(np.zeros((1000, 12)))
        skipped += 1

X = np.array(X, dtype=np.float32)   # (21837, 1000, 12)
print(f"Loaded {X.shape[0]} ECGs. Skipped: {skipped}")

# ── STEP 3: Build binary labels (NORM=0, ABNORM=1) ───────────────────────────
print("Building binary labels...")
agg_df = pd.read_csv(DATA_PATH + "scp_statements.csv", index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def get_label(scp_dict):
    diag_keys = [k for k in scp_dict.keys() if k in agg_df.index]
    return 0 if ("NORM" in diag_keys) else 1

y = np.array(Y.scp_codes.apply(get_label).values, dtype=np.int64)
patient_ids = Y.patient_id.values

print(f"Labels — NORM: {(y==0).sum()} | ABNORM: {(y==1).sum()}")

# ── STEP 4: Normalise ECG signals ────────────────────────────────────────────
print("Normalising signals...")
mean = X.mean(axis=(0, 1), keepdims=True)
std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
X    = (X - mean) / std

# ── STEP 5: Split into N federated clients by patient ─────────────────────────
print(f"Splitting into {N_CLIENTS} federated clients (non-IID by patient)...")
np.random.seed(RANDOM_SEED)
unique_patients = np.unique(patient_ids)
np.random.shuffle(unique_patients)
patient_groups = np.array_split(unique_patients, N_CLIENTS)

clients_X = []
clients_y = []

for i, group in enumerate(patient_groups):
    mask = np.isin(patient_ids, group)
    cx = X[mask]
    cy = y[mask]
    clients_X.append(cx)
    clients_y.append(cy)
    norm_count   = (cy == 0).sum()
    abnorm_count = (cy == 1).sum()
    print(f"  Client {i+1:2d}: {len(cy):5d} records | NORM={norm_count} | ABNORM={abnorm_count}")

# ── STEP 6: Save ─────────────────────────────────────────────────────────────
print("Saving client data...")
save_dict = {}
for i in range(N_CLIENTS):
    save_dict[f"client_{i}_X"] = clients_X[i]
    save_dict[f"client_{i}_y"] = clients_y[i]

np.savez_compressed(OUTPUT_DIR + "clients.npz", **save_dict)
print(f"\nDone! Saved to {OUTPUT_DIR}clients.npz")
print("Run 02_baseline_fl.py next.")
