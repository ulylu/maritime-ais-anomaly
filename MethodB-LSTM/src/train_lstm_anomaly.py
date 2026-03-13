#!/usr/bin/env python3
"""
train_lstm_anomaly.py

Train an LSTM prediction model on NORMAL AIS trajectories only.
The model learns to predict the next observation given a sequence of past
observations. At test time (separate script), high prediction error indicates
anomalous vessel behavior.

Approach:
  1. Read labeled CSV, keep only is_anomaly == 0 rows.
  2. Sample a subset of vessels (for CPU-tractable training).
  3. Per vessel: compute 8 features per time step, build sliding-window sequences.
  4. Normalize features, split train / val.
  5. Train LSTM predictor: input seq[0..T-1] -> predict features at step T.
  6. Save model, scaler, and error thresholds for later inference.

Usage:
  python src/train_lstm_anomaly.py --input data/ais_2024_last7days_labeled.csv
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def log(msg: str):
    print(msg, flush=True)

# ── Feature definition ──────────────────────────────────────────
FEATURE_NAMES = [
    "SOG",
    "sin_COG",
    "cos_COG",
    "sin_HDG",
    "cos_HDG",
    "delta_lat",
    "delta_lon",
    "log_delta_t",
]
N_FEATURES = len(FEATURE_NAMES)

RAW_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "is_anomaly"]
KEEP_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading"]

MAX_GAP_SEC = 3600  # 1 hour — break sequences at gaps larger than this


# ── Dataset ─────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ── Model ───────────────────────────────────────────────────────
class LSTMPredictor(nn.Module):
    def __init__(self, n_features, hidden, n_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_features),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ── Feature engineering ─────────────────────────────────────────
def vessel_features(df: pd.DataFrame) -> np.ndarray:
    """Build (n_points, 8) feature matrix for one vessel (already sorted by time)."""
    lat = df["LAT"].values
    lon = df["LON"].values
    sog = np.clip(df["SOG"].values, 0, 40)
    cog = np.nan_to_num(df["COG"].values, nan=0.0)
    hdg = df["Heading"].values.copy()
    hdg[(hdg == 511) | np.isnan(hdg)] = 0.0

    ts = pd.to_datetime(df["BaseDateTime"], format="%Y-%m-%dT%H:%M:%S").values.astype("int64") / 1e9

    n = len(df)
    feat = np.zeros((n, N_FEATURES), dtype=np.float64)
    feat[:, 0] = sog
    feat[:, 1] = np.sin(np.deg2rad(cog))
    feat[:, 2] = np.cos(np.deg2rad(cog))
    feat[:, 3] = np.sin(np.deg2rad(hdg))
    feat[:, 4] = np.cos(np.deg2rad(hdg))
    feat[1:, 5] = np.diff(lat)
    feat[1:, 6] = np.diff(lon)
    dt = np.diff(ts)
    dt = np.clip(dt, 0, None)
    feat[1:, 7] = np.log1p(dt)
    return feat, dt


def build_sequences(feat: np.ndarray, dt: np.ndarray, seq_len: int):
    """Sliding window; skip windows that span a gap > MAX_GAP_SEC."""
    total = seq_len + 1
    n = len(feat)
    if n < total:
        return []

    seqs = []
    for start in range(n - total + 1):
        window_dt = dt[start : start + total - 1]  # dt between consecutive points
        if window_dt.max() > MAX_GAP_SEC:
            continue
        seqs.append(feat[start : start + total])
    return seqs


# ── Main ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Train LSTM on normal AIS trajectories.")
    ap.add_argument("--input", default="data/ais_2024_last7days_labeled.csv")
    ap.add_argument("--output-dir", default="output/lstm")
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-vessels", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    ap.add_argument("--patience", type=int, default=5, help="Early-stop patience")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[info] device={device}  seq_len={args.seq_len}  hidden={args.hidden}")

    # ── 1. Discover vessels from first chunk ────────────────────
    log("[1/8] scanning for vessel IDs ...")
    first = pd.read_csv(
        args.input, usecols=["MMSI", "is_anomaly"], nrows=5_000_000
    )
    normal_mmsi = first.loc[first["is_anomaly"] == 0, "MMSI"].unique()
    log(f"  found {len(normal_mmsi):,} vessels with normal data (first 5M rows)")

    rng = np.random.default_rng(42)
    if len(normal_mmsi) > args.max_vessels:
        sampled_mmsi = set(rng.choice(normal_mmsi, size=args.max_vessels, replace=False))
    else:
        sampled_mmsi = set(normal_mmsi)
    log(f"  sampled {len(sampled_mmsi):,} vessels for training")
    del first

    # ── 2. Read full file, keep sampled normal rows ─────────────
    log("[2/8] reading normal rows for sampled vessels ...")
    t0 = time.time()
    parts = []
    total_read = 0
    for chunk in pd.read_csv(args.input, usecols=RAW_COLS, chunksize=args.chunk_size):
        total_read += len(chunk)
        keep = chunk[(chunk["is_anomaly"] == 0) & (chunk["MMSI"].isin(sampled_mmsi))]
        if len(keep):
            parts.append(keep[KEEP_COLS])
        if total_read % 5_000_000 < args.chunk_size:
            log(f"  {total_read:,} rows scanned ...")

    df = pd.concat(parts, ignore_index=True)
    del parts
    df.sort_values(["MMSI", "BaseDateTime"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_vessels = df["MMSI"].nunique()
    log(f"  kept {len(df):,} rows from {n_vessels:,} vessels  ({time.time()-t0:.0f}s)")

    # ── 3. Build per-vessel feature sequences ───────────────────
    log(f"[3/8] building sequences (seq_len={args.seq_len}) ...")
    t0 = time.time()
    all_seqs = []
    grouped = df.groupby("MMSI", sort=False)
    done = 0
    for mmsi, vdf in grouped:
        if len(vdf) < args.seq_len + 2:
            continue
        feat, dt = vessel_features(vdf)
        seqs = build_sequences(feat, dt, args.seq_len)
        all_seqs.extend(seqs)
        done += 1
        if done % 500 == 0:
            log(f"  {done}/{n_vessels} vessels -> {len(all_seqs):,} seqs")

    del df, grouped
    log(f"  total sequences: {len(all_seqs):,}  ({time.time()-t0:.0f}s)")

    if not all_seqs:
        log("[error] no sequences produced -- try reducing --seq-len")
        return

    all_seqs = np.array(all_seqs, dtype=np.float32)  # (N, seq_len+1, 8)

    # ── 4. Normalize ────────────────────────────────────────────
    log("[4/8] normalizing ...")
    flat = all_seqs.reshape(-1, N_FEATURES)
    feat_mean = flat.mean(axis=0)
    feat_std = flat.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    all_seqs = (all_seqs - feat_mean) / feat_std

    scaler = {"mean": feat_mean.tolist(), "std": feat_std.tolist(), "features": FEATURE_NAMES}
    scaler_path = os.path.join(args.output_dir, "scaler.json")
    with open(scaler_path, "w") as f:
        json.dump(scaler, f, indent=2)
    log(f"  scaler -> {scaler_path}")

    # ── 5. Train / val split ────────────────────────────────────
    n_total = len(all_seqs)
    idx = rng.permutation(n_total)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val

    x_all = all_seqs[:, :-1, :]  # (N, seq_len, 8)
    y_all = all_seqs[:, -1, :]  # (N, 8)
    del all_seqs

    train_ds = SeqDataset(x_all[idx[:n_train]], y_all[idx[:n_train]])
    val_ds = SeqDataset(x_all[idx[n_train:]], y_all[idx[n_train:]])
    del x_all, y_all

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    log(f"[5/8] train={n_train:,}  val={n_val:,}")

    # ── 6. Train ────────────────────────────────────────────────
    model = LSTMPredictor(N_FEATURES, args.hidden, args.layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    n_params = sum(p.numel() for p in model.parameters())
    log(f"[6/8] model params: {n_params:,}")

    model_path = os.path.join(args.output_dir, "lstm_model.pt")
    best_val = float("inf")
    best_ep = 0
    wait = 0

    for ep in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        t_loss = 0.0
        nb = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            nb += 1
        t_loss /= max(nb, 1)

        model.eval()
        v_loss = 0.0
        nb_v = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += criterion(model(xb), yb).item()
                nb_v += 1
        v_loss /= max(nb_v, 1)

        scheduler.step(v_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_ep
        log(
            f"  epoch {ep:3d}/{args.epochs}  "
            f"train={t_loss:.6f}  val={v_loss:.6f}  lr={lr:.1e}  ({elapsed:.1f}s)"
        )

        if v_loss < best_val:
            best_val = v_loss
            best_ep = ep
            torch.save(model.state_dict(), model_path)
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                log(f"  early stop at epoch {ep} (no improvement for {args.patience} epochs)")
                break

    log(f"\n  best val_loss={best_val:.6f} at epoch {best_ep}  -> {model_path}")

    # ── 7. Error threshold from validation ──────────────────────
    log("[7/8] computing anomaly thresholds on validation set ...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    errors = []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            mse = ((pred - yb) ** 2).mean(dim=1)
            errors.append(mse.cpu().numpy())
    errors = np.concatenate(errors)

    thresholds = {
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
        "mean": float(errors.mean()),
        "std": float(errors.std()),
        "min": float(errors.min()),
        "max": float(errors.max()),
    }
    thresh_path = os.path.join(args.output_dir, "thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    log("  validation error distribution:")
    for k, v in thresholds.items():
        log(f"    {k:>5s}: {v:.6f}")

    # ── 8. Save config ──────────────────────────────────────────
    config = {
        "seq_len": args.seq_len,
        "hidden": args.hidden,
        "layers": args.layers,
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "best_epoch": best_ep,
        "best_val_loss": float(best_val),
        "n_train": n_train,
        "n_val": n_val,
        "max_vessels": args.max_vessels,
        "epochs_ran": min(ep, args.epochs),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "max_gap_sec": MAX_GAP_SEC,
    }
    cfg_path = os.path.join(args.output_dir, "train_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    log(f"\n[done] outputs in {args.output_dir}/")
    log(f"  lstm_model.pt      -- trained model weights")
    log(f"  scaler.json        -- feature normalization params")
    log(f"  thresholds.json    -- reconstruction error thresholds")
    log(f"  train_config.json  -- training hyperparameters")


if __name__ == "__main__":
    main()
