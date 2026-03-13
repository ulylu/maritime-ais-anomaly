#!/usr/bin/env python3
"""
plot_methodB_lstm.py

Visualize top Method B (LSTM) anomalies as trajectory plots + error
breakdown bar charts.  Mirrors the output structure of Method A plots.

For each anomaly event:
  Left panel  — trajectory map (lat/lon) with anomaly point highlighted
  Right panel — per-feature prediction error bar chart

Output:
  output/figs/methodB_lstm/
    methodB_top_001_mmsi_XXXXX.png
    methodB_random_001_mmsi_XXXXX.png
    manifest.csv

Usage:
  python src/plot_methodB_lstm.py
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def log(msg: str):
    print(msg, flush=True)


# ── Utilities ────────────────────────────────────────────────────────────────

def parse_iso(s):
    s = str(s).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ── Event loading ────────────────────────────────────────────────────────────

def load_events(csv_path, n_top, n_random, seed=42):
    """
    Load top and random anomaly events from methodB_top1pct.csv.
    Deduplicates by MMSI for top selection (one vessel per slot).
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    df = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    dedup = df.drop_duplicates(subset=["MMSI"], keep="first").reset_index(drop=True)
    top_events = dedup.head(n_top).to_dict("records")
    for e in top_events:
        e["sample_kind"] = "top"

    top_mmsi = {str(e["MMSI"]) for e in top_events}
    remainder = df[~df["MMSI"].astype(str).isin(top_mmsi)].reset_index(drop=True)
    n_rand = min(n_random, len(remainder))
    rand_events = []
    if n_rand > 0:
        rand_rows = remainder.sample(n=n_rand, random_state=seed)
        rand_events = rand_rows.to_dict("records")
        for e in rand_events:
            e["sample_kind"] = "random"

    return top_events + rand_events


# ── AIS point streaming ─────────────────────────────────────────────────────

def stream_vessel_points(points_csv, mmsi_set):
    """Read AIS CSV once, collect all points for the requested MMSIs."""
    data = defaultdict(list)
    n_read = 0
    with open(points_csv, "r", newline="", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_read += 1
            mmsi = row.get("MMSI", "").strip()
            if mmsi not in mmsi_set:
                continue
            t = parse_iso(row.get("BaseDateTime", ""))
            if t is None:
                continue
            try:
                lat = float(row["LAT"])
                lon = float(row["LON"])
                sog = float(row.get("SOG") or 0)
            except (ValueError, KeyError):
                continue
            data[mmsi].append((t, lat, lon, sog))
            if n_read % 5_000_000 == 0:
                log(f"  {n_read:,} rows scanned ...")
    for mmsi in data:
        data[mmsi].sort(key=lambda x: x[0])
    log(f"  scan done: {n_read:,} rows, data for {len(data)} vessels")
    return data


# ── Plot function ────────────────────────────────────────────────────────────

ERR_LABELS = ["speed", "course", "heading", "position_jump", "time_gap"]
ERR_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]


def plot_event(event, vessel_points, out_path, context_s=3600):
    """
    Two-panel figure:
      Left  — trajectory (lat/lon), anomaly point in red
      Right — per-feature error bar chart
    """
    t_curr = parse_iso(event.get("t_curr"))
    t_prev = parse_iso(event.get("t_prev"))
    if t_curr is None:
        return False

    t_ref = t_curr
    lo = t_ref - timedelta(seconds=context_s)
    hi = t_ref + timedelta(seconds=context_s)

    ctx = [p for p in vessel_points if lo <= p[0] <= hi]
    if len(ctx) < 3:
        return False

    before = [p for p in ctx if p[0] < t_ref]
    at_or_after = [p for p in ctx if p[0] >= t_ref]

    fig, (ax_map, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]}
    )

    # ── Left: trajectory map ─────────────────────────────────────────────────
    if before:
        ax_map.plot(
            [p[2] for p in before], [p[1] for p in before],
            "o-", color="steelblue", lw=1.2, ms=3, alpha=0.7, label="context",
        )
    if at_or_after:
        ax_map.plot(
            [p[2] for p in at_or_after], [p[1] for p in at_or_after],
            "o-", color="red", lw=2.5, ms=5, alpha=0.9, label="anomaly",
        )
    if ctx:
        ax_map.scatter(
            ctx[0][2], ctx[0][1], s=60, marker="o", color="green",
            zorder=5, label="start",
        )
        ax_map.scatter(
            ctx[-1][2], ctx[-1][1], s=60, marker="x", color="black",
            zorder=5, label="end",
        )

    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title("Trajectory (context ±1h)")
    ax_map.legend(fontsize=8, loc="best")
    ax_map.grid(True, alpha=0.3)

    # ── Right: per-feature error breakdown ────────────────────────────────────
    err_vals = [float(event.get(f"err_{e}", 0) or 0) for e in ERR_LABELS]
    bars = ax_bar.barh(ERR_LABELS, err_vals, color=ERR_COLORS, alpha=0.85)
    ax_bar.set_xlabel("Squared Error")
    ax_bar.set_title("Error Breakdown by Feature")
    for bar, val in zip(bars, err_vals):
        if val > 0:
            ax_bar.text(
                bar.get_width() + max(err_vals) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8,
            )

    score = float(event.get("anomaly_score", 0))
    pct = float(event.get("percentile_rank", 0) or 0)
    sog = float(event.get("SOG", 0) or 0)
    reason = event.get("reason_text", "")

    fig.suptitle(
        f"Method B (LSTM) Anomaly — MMSI {event['MMSI']}  "
        f"[score={score:.4f}, p{pct:.2f}]\n"
        f"SOG={sog:.1f} kts | {reason} | {event.get('t_curr', '')}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Plot top Method B (LSTM) anomalies."
    )
    ap.add_argument("--top-csv", default="output/methodB_lstm/methodB_top1pct.csv")
    ap.add_argument("--points", default="data/ais_2024_last7days.csv")
    ap.add_argument("--out-dir", default="output/figs/methodB_lstm")
    ap.add_argument("--n-top", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--context-s", type=int, default=3600)
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.top_csv):
        log(f"ERROR: {args.top_csv} not found. Run score_methodB_lstm.py first.")
        sys.exit(1)
    if not os.path.exists(args.points):
        log(f"ERROR: {args.points} not found.")
        sys.exit(1)

    ensure_dir(args.out_dir)

    # Load events
    log("[1/3] loading anomaly events ...")
    events = load_events(args.top_csv, args.n_top, args.n_random, args.random_seed)
    log(f"  selected {len(events)} events "
        f"({args.n_top} top + up to {args.n_random} random)")

    if not events:
        log("No events to plot.")
        return

    # Collect vessel points
    mmsi_set = {str(e["MMSI"]) for e in events}
    log(f"[2/3] streaming points for {len(mmsi_set)} vessels from {args.points} ...")
    vessel_data = stream_vessel_points(args.points, mmsi_set)

    # Generate plots
    log("[3/3] generating plots ...")
    top_counter = 0
    rand_counter = 0
    manifest_rows = []

    for event in events:
        mmsi = str(event["MMSI"])
        kind = event.get("sample_kind", "top")
        vpts = vessel_data.get(mmsi, [])

        if kind == "top":
            top_counter += 1
            fname = f"methodB_top_{top_counter:03d}_mmsi_{mmsi}.png"
        else:
            rand_counter += 1
            fname = f"methodB_random_{rand_counter:03d}_mmsi_{mmsi}.png"

        out_path = os.path.join(args.out_dir, fname)
        ok = plot_event(event, vpts, out_path, context_s=args.context_s)
        status = "saved" if ok else "skipped (not enough context)"
        log(f"  {fname}: {status}")

        manifest_rows.append({
            "kind": kind,
            "mmsi": mmsi,
            "t_prev": event.get("t_prev"),
            "t_curr": event.get("t_curr"),
            "anomaly_score": event.get("anomaly_score"),
            "percentile_rank": event.get("percentile_rank"),
            "dominant_error": event.get("dominant_error"),
            "status": status,
            "file": out_path if ok else "",
        })

    mfst_path = os.path.join(args.out_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(mfst_path, index=False)
    log(f"\n[done] manifest -> {mfst_path}")

    n_saved = sum(1 for r in manifest_rows if r["status"] == "saved")
    log(f"  figures saved: {n_saved}/{len(events)} in {args.out_dir}")


if __name__ == "__main__":
    main()
