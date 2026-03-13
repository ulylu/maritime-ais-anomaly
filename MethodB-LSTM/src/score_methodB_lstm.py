#!/usr/bin/env python3
"""
score_methodB_lstm.py

Apply trained LSTM model to score AIS data for anomalies (Method B).
For each vessel's trajectory, build sliding-window sequences, predict
the next step, and flag sequences with high prediction error.

Output structure mirrors Method A:
  output/methodB_lstm/
    methodB_top1pct.csv       — top 1% most anomalous sequences
    methodB_explain.csv       — same + per-feature error breakdown
    methodB_summary.txt       — run statistics
    error_stats.json          — error distribution percentiles

Usage:
  python src/score_methodB_lstm.py --input data/ais_2024_last7days.csv
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def log(msg: str):
    print(msg, flush=True)


# ── Feature definition (must match training) ──────────────────────────────────
FEATURE_NAMES = [
    "SOG", "sin_COG", "cos_COG", "sin_HDG", "cos_HDG",
    "delta_lat", "delta_lon", "log_delta_t",
]
N_FEATURES = len(FEATURE_NAMES)
MAX_GAP_SEC = 3600
KEEP_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading"]

ERR_GROUPS = {
    "speed":         [0],
    "course":        [1, 2],
    "heading":       [3, 4],
    "position_jump": [5, 6],
    "time_gap":      [7],
}
ERR_LABELS = list(ERR_GROUPS.keys())
ERR_REASONS = {
    "speed":         "unexpected speed change",
    "course":        "unexpected course change",
    "heading":       "unexpected heading change",
    "position_jump": "unexpected position jump",
    "time_gap":      "unexpected time gap pattern",
}


# ── Model (must match training) ──────────────────────────────────────────────
class LSTMPredictor(nn.Module):
    def __init__(self, n_features, hidden, n_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
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


# ── Feature engineering (must match training) ────────────────────────────────
def vessel_features(df: pd.DataFrame):
    lat = df["LAT"].values
    lon = df["LON"].values
    sog = np.clip(df["SOG"].values, 0, 40)
    cog = np.nan_to_num(df["COG"].values, nan=0.0)
    hdg = df["Heading"].values.copy()
    hdg[(hdg == 511) | np.isnan(hdg)] = 0.0

    ts = pd.to_datetime(
        df["BaseDateTime"], format="%Y-%m-%dT%H:%M:%S"
    ).values.astype("int64") / 1e9

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


# ── Per-vessel scoring ────────────────────────────────────────────────────────
def score_vessel(mmsi, vdf, seq_len, model, scaler_mean, scaler_std, device,
                 batch_size=2048):
    """Score all valid sequences for one vessel, return list of record dicts."""
    feat, dt = vessel_features(vdf)
    times = vdf["BaseDateTime"].values
    lats = vdf["LAT"].values
    lons = vdf["LON"].values
    sogs = vdf["SOG"].values

    total = seq_len + 1
    n = len(feat)
    if n < total:
        return []

    valid_starts = []
    for s in range(n - total + 1):
        if dt[s: s + total - 1].max() <= MAX_GAP_SEC:
            valid_starts.append(s)
    if not valid_starts:
        return []

    xs = np.array(
        [(feat[s: s + seq_len] - scaler_mean) / scaler_std for s in valid_starts],
        dtype=np.float32,
    )
    ys = np.array(
        [(feat[s + seq_len] - scaler_mean) / scaler_std for s in valid_starts],
        dtype=np.float32,
    )

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            xb = torch.as_tensor(xs[i: i + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds)

    per_feat_err = (preds - ys) ** 2
    total_mse = per_feat_err.mean(axis=1)

    group_err = {}
    for name, cols in ERR_GROUPS.items():
        group_err[name] = per_feat_err[:, cols].sum(axis=1)

    records = []
    for i, s in enumerate(valid_starts):
        ti = s + seq_len
        rec = {
            "MMSI": int(mmsi),
            "t_start": str(times[s]),
            "t_prev": str(times[ti - 1]),
            "t_curr": str(times[ti]),
            "LAT": float(lats[ti]),
            "LON": float(lons[ti]),
            "SOG": float(sogs[ti]),
            "anomaly_score": float(total_mse[i]),
        }
        for name in ERR_LABELS:
            rec[f"err_{name}"] = float(group_err[name][i])
        records.append(rec)
    return records


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Score AIS data with LSTM (Method B).")
    ap.add_argument("--input", default="data/ais_2024_last7days.csv")
    ap.add_argument("--model-dir", default="output/lstm")
    ap.add_argument("--output-dir", default="output/methodB_lstm")
    ap.add_argument("--max-vessels", type=int, default=5000)
    ap.add_argument("--top-frac", type=float, default=0.01)
    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    cfg_path = os.path.join(args.model_dir, "train_config.json")
    scaler_path = os.path.join(args.model_dir, "scaler.json")
    model_path = os.path.join(args.model_dir, "lstm_model.pt")
    thresh_path = os.path.join(args.model_dir, "thresholds.json")

    with open(cfg_path) as f:
        config = json.load(f)
    with open(scaler_path) as f:
        scaler = json.load(f)
    with open(thresh_path) as f:
        train_thresholds = json.load(f)

    seq_len = config["seq_len"]
    scaler_mean = np.array(scaler["mean"])
    scaler_std = np.array(scaler["std"])

    model = LSTMPredictor(
        n_features=config["n_features"],
        hidden=config["hidden"],
        n_layers=config["layers"],
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    log(f"[info] model loaded  seq_len={seq_len}  device={device}")
    log(f"  training thresholds: p95={train_thresholds['p95']:.4f}  "
        f"p99={train_thresholds['p99']:.4f}")

    # ── 1. Discover vessels ───────────────────────────────────────────────────
    log("[1/5] scanning for vessel IDs ...")
    first = pd.read_csv(args.input, usecols=["MMSI"], nrows=5_000_000)
    all_mmsi = first["MMSI"].unique()
    log(f"  found {len(all_mmsi):,} vessels (first 5M rows)")

    rng = np.random.default_rng(123)
    if len(all_mmsi) > args.max_vessels:
        sampled = set(rng.choice(all_mmsi, size=args.max_vessels, replace=False))
    else:
        sampled = set(all_mmsi)
    log(f"  sampled {len(sampled):,} vessels for scoring")
    del first

    # ── 2. Read data ──────────────────────────────────────────────────────────
    log("[2/5] reading data for sampled vessels ...")
    t0 = time.time()
    parts = []
    total_read = 0
    for chunk in pd.read_csv(args.input, usecols=KEEP_COLS,
                             chunksize=args.chunk_size):
        total_read += len(chunk)
        keep = chunk[chunk["MMSI"].isin(sampled)]
        if len(keep):
            parts.append(keep)
        if total_read % 5_000_000 < args.chunk_size:
            log(f"  {total_read:,} rows scanned ...")

    df = pd.concat(parts, ignore_index=True)
    del parts
    df.sort_values(["MMSI", "BaseDateTime"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_vessels = df["MMSI"].nunique()
    log(f"  kept {len(df):,} rows from {n_vessels:,} vessels  ({time.time()-t0:.0f}s)")

    # ── 3. Score each vessel ──────────────────────────────────────────────────
    log("[3/5] scoring sequences ...")
    t0 = time.time()
    all_records = []
    grouped = df.groupby("MMSI", sort=False)
    done = 0
    for mmsi, vdf in grouped:
        if len(vdf) < seq_len + 2:
            continue
        recs = score_vessel(mmsi, vdf, seq_len, model,
                            scaler_mean, scaler_std, device,
                            batch_size=args.batch_size)
        all_records.extend(recs)
        done += 1
        if done % 500 == 0:
            log(f"  {done}/{n_vessels} vessels -> {len(all_records):,} scored seqs")

    del df, grouped
    log(f"  total scored: {len(all_records):,}  ({time.time()-t0:.0f}s)")

    if not all_records:
        log("[error] no sequences scored")
        return

    # ── 4. Build result tables ────────────────────────────────────────────────
    log("[4/5] building result tables ...")
    scores_df = pd.DataFrame(all_records)
    scores_df.sort_values("anomaly_score", ascending=False, inplace=True)
    scores_df.reset_index(drop=True, inplace=True)

    error_vals = scores_df["anomaly_score"].values
    stats = {}
    for p in [0, 25, 50, 75, 90, 95, 99, 100]:
        key = "min" if p == 0 else ("max" if p == 100 else f"p{p}")
        stats[key] = float(np.percentile(error_vals, p))
    stats["mean"] = float(error_vals.mean())
    stats["std"] = float(error_vals.std())

    n_top = max(1, int(len(scores_df) * args.top_frac))
    top_df = scores_df.head(n_top).copy()
    top_df.insert(0, "rank", range(1, n_top + 1))
    top_df["percentile_rank"] = np.round(
        100.0 * (1.0 - np.arange(n_top) / len(scores_df)), 4
    )

    err_cols = [f"err_{e}" for e in ERR_LABELS]
    dominant_idx = top_df[err_cols].values.argmax(axis=1)
    top_df["dominant_error"] = [ERR_LABELS[i] for i in dominant_idx]
    top_df["reason_text"] = top_df["dominant_error"].map(ERR_REASONS)

    # ── 5. Save outputs ──────────────────────────────────────────────────────
    log("[5/5] saving outputs ...")

    top_path = os.path.join(args.output_dir, "methodB_top1pct.csv")
    top_df.to_csv(top_path, index=False)
    log(f"  {top_path}  ({n_top:,} rows)")

    explain_path = os.path.join(args.output_dir, "methodB_explain.csv")
    top_df.to_csv(explain_path, index=False)
    log(f"  {explain_path}")

    stats_path = os.path.join(args.output_dir, "error_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log(f"  {stats_path}")

    dom_counts = top_df["dominant_error"].value_counts()
    summary = [
        "Method B — LSTM Anomaly Scoring Summary",
        "=" * 50,
        f"Run timestamp    : {datetime.now().isoformat()}",
        f"Input data       : {args.input}",
        f"Model directory  : {args.model_dir}",
        f"Output directory : {args.output_dir}",
        f"Vessels scored   : {done:,} / {len(sampled):,} sampled",
        f"Total sequences  : {len(scores_df):,}",
        f"Top fraction     : {args.top_frac} ({args.top_frac*100:.1f}%)",
        f"Top count        : {n_top:,}",
        "",
        "Scoring method",
        "-" * 30,
        "LSTM next-step prediction model (trained on normal data only).",
        f"Sequence length  : {seq_len}",
        f"Features         : {', '.join(FEATURE_NAMES)}",
        "anomaly_score    = MSE between predicted and actual next step.",
        "",
        "Prediction error distribution (all scored sequences)",
        "-" * 30,
    ]
    for k, v in stats.items():
        summary.append(f"  {k:>5s}: {v:.6f}")

    summary.extend([
        "",
        f"Training-set thresholds (for reference)",
        "-" * 30,
        f"  p95 : {train_thresholds['p95']:.6f}",
        f"  p99 : {train_thresholds['p99']:.6f}",
        "",
        f"Dominant error types in top {args.top_frac*100:.0f}%",
        "-" * 30,
    ])
    for label in ERR_LABELS:
        cnt = int(dom_counts.get(label, 0))
        pct = cnt / len(top_df) * 100
        summary.append(f"  {label:>15s}: {cnt:>8,} ({pct:5.1f}%)")

    summary.extend([
        "",
        "Generated files",
        "-" * 30,
        f"  {top_path}",
        f"  {explain_path}",
        f"  {stats_path}",
    ])

    summary_path = os.path.join(args.output_dir, "methodB_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary) + "\n")
    log(f"  {summary_path}")

    log(f"\n[done] outputs in {args.output_dir}/")
    log("  error distribution:")
    for k, v in stats.items():
        log(f"    {k:>5s}: {v:.6f}")


if __name__ == "__main__":
    main()
