#!/usr/bin/env python3
"""
plot_methodA_anomalies.py

Goal:
- Visualize top Method A anomalies using cleaned AIS points
- Plot anomaly segment and a same-vessel control segment (before anomaly)

Notes:
- This script scans ais_points_clean.csv in streaming mode.
- Default is to plot only a small number of anomalies (e.g., 20).
"""

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


TIME_FMT_HINT = "ISO format like 2024-12-31T18:05:10"


def parse_iso_time(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(s))
    except Exception:
        return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class Event:
    idx: int
    mmsi: str
    t_prev: datetime
    t_curr: datetime
    anomaly_score: float
    context_before_min: int
    context_after_min: int

    @property
    def t0(self) -> datetime:
        return self.t_prev - timedelta(minutes=self.context_before_min)

    @property
    def t1(self) -> datetime:
        return self.t_curr + timedelta(minutes=self.context_after_min)

    @property
    def duration_s(self) -> float:
        return (self.t_curr - self.t_prev).total_seconds()

    @property
    def control_prev(self) -> datetime:
        # same duration window immediately before anomaly
        return self.t_prev - (self.t_curr - self.t_prev)

    @property
    def control_curr(self) -> datetime:
        return self.t_prev


def load_events(
    top_csv: str,
    n_plots: int,
    context_before_min: int,
    context_after_min: int,
    min_score: Optional[float],
) -> List[Event]:
    print(f"[info] reading top anomalies: {top_csv}")
    df = pd.read_csv(top_csv)

    required = ["MMSI", "t_prev", "t_curr", "anomaly_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Top anomaly CSV missing columns: {missing}")

    if min_score is not None:
        df = df[pd.to_numeric(df["anomaly_score"], errors="coerce") >= min_score].copy()

    # Already sorted in previous step, but keep this safe.
    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce")
    df = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    events: List[Event] = []
    for i, row in df.iterrows():
        if len(events) >= n_plots:
            break
        t_prev = parse_iso_time(row["t_prev"])
        t_curr = parse_iso_time(row["t_curr"])
        if t_prev is None or t_curr is None or t_curr <= t_prev:
            continue
        events.append(
            Event(
                idx=len(events) + 1,
                mmsi=str(row["MMSI"]).strip(),
                t_prev=t_prev,
                t_curr=t_curr,
                anomaly_score=float(row["anomaly_score"]),
                context_before_min=context_before_min,
                context_after_min=context_after_min,
            )
        )

    if not events:
        raise ValueError("No valid anomaly events selected for plotting.")

    print(f"[info] selected {len(events)} events for plotting")
    return events


def collect_points_for_events(points_csv: str, events: List[Event]) -> Dict[int, List[dict]]:
    by_mmsi: Dict[str, List[Event]] = {}
    for e in events:
        by_mmsi.setdefault(e.mmsi, []).append(e)

    rows_by_event: Dict[int, List[dict]] = {e.idx: [] for e in events}
    n_read = 0
    n_kept = 0

    print(f"[info] streaming points: {points_csv}")
    with open(points_csv, "r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_read += 1
            mmsi = str(row.get("MMSI", "")).strip()
            if mmsi not in by_mmsi:
                continue

            t = parse_iso_time(row.get("BaseDateTime", ""))
            if t is None:
                continue

            lat = row.get("LAT", "")
            lon = row.get("LON", "")
            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except Exception:
                continue

            for e in by_mmsi[mmsi]:
                if e.t0 <= t <= e.t1:
                    rows_by_event[e.idx].append(
                        {
                            "BaseDateTime": t,
                            "LAT": lat_f,
                            "LON": lon_f,
                            "SOG": pd.to_numeric(row.get("SOG", ""), errors="coerce"),
                            "COG": pd.to_numeric(row.get("COG", ""), errors="coerce"),
                            "Heading": pd.to_numeric(row.get("Heading", ""), errors="coerce"),
                        }
                    )
                    n_kept += 1

            if n_read % 1_000_000 == 0:
                print(f"[progress] read={n_read:,} kept={n_kept:,}")

    print(f"[info] point scan done: read={n_read:,} kept={n_kept:,}")
    return rows_by_event


def _plot_track(ax, df: pd.DataFrame, title: str, color: str, alpha: float = 1.0) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No points", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    ax.plot(df["LON"], df["LAT"], "-", color=color, linewidth=2, alpha=alpha)
    ax.scatter(df["LON"], df["LAT"], s=14, color=color, alpha=alpha)
    ax.scatter(df["LON"].iloc[0], df["LAT"].iloc[0], s=40, marker="o", color="green", label="start")
    ax.scatter(df["LON"].iloc[-1], df["LAT"].iloc[-1], s=40, marker="x", color="black", label="end")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)


def plot_event(event: Event, rows: List[dict], out_dir: str) -> Optional[str]:
    if not rows:
        print(f"[warn] event {event.idx}: no points found in context window")
        return None

    df = pd.DataFrame(rows).sort_values("BaseDateTime").reset_index(drop=True)

    context_df = df.copy()
    anomaly_df = df[(df["BaseDateTime"] >= event.t_prev) & (df["BaseDateTime"] <= event.t_curr)].copy()
    control_df = df[(df["BaseDateTime"] >= event.control_prev) & (df["BaseDateTime"] <= event.control_curr)].copy()

    if len(anomaly_df) < 2:
        print(f"[warn] event {event.idx}: anomaly segment has <2 points (skip)")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Left: local context + anomaly highlight
    ax = axes[0]
    _plot_track(ax, context_df, "Local Context (same vessel)", color="#B0B7C3", alpha=0.9)
    ax.plot(anomaly_df["LON"], anomaly_df["LAT"], "-", color="#D7263D", linewidth=3, label="anomaly")
    ax.scatter(anomaly_df["LON"], anomaly_df["LAT"], s=18, color="#D7263D")
    ax.legend(loc="best", fontsize=8)

    # Right: control segment (before anomaly) vs anomaly
    ax2 = axes[1]
    _plot_track(ax2, anomaly_df, "Anomaly Segment", color="#D7263D", alpha=0.95)
    if len(control_df) >= 2:
        ax2.plot(control_df["LON"], control_df["LAT"], "-", color="#1F77B4", linewidth=2.5, label="control(before)")
        ax2.scatter(control_df["LON"], control_df["LAT"], s=16, color="#1F77B4")
        ax2.legend(loc="best", fontsize=8)
    else:
        ax2.text(0.5, 0.08, "No control segment points in pre-window", ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle(
        f"Method A Anomaly #{event.idx} | MMSI={event.mmsi} | "
        f"score={event.anomaly_score:.4f} | {event.t_prev.isoformat()} -> {event.t_curr.isoformat()}",
        fontsize=10,
    )

    fname = (
        f"{event.idx:03d}_mmsi_{event.mmsi}_"
        f"{event.t_prev.strftime('%Y%m%dT%H%M%S')}_to_{event.t_curr.strftime('%H%M%S')}.png"
    )
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def write_manifest(manifest_path: str, records: List[dict]) -> None:
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    print(f"[done] saved manifest: {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot top Method A AIS anomalies using cleaned trajectories.")
    ap.add_argument("--top", default="output/ais_methodA_top1pct.csv", help="Top anomaly CSV")
    ap.add_argument("--points", default="output/ais_points_clean.csv", help="Clean AIS points CSV")
    ap.add_argument(
        "--out-dir",
        default="output/figs/methodA_top_tracks",
        help="Output folder for figures",
    )
    ap.add_argument("--n-plots", type=int, default=20, help="How many top anomalies to plot")
    ap.add_argument("--context-before-min", type=int, default=60, help="Minutes before anomaly for context")
    ap.add_argument("--context-after-min", type=int, default=60, help="Minutes after anomaly for context")
    ap.add_argument("--min-score", type=float, default=None, help="Optional score threshold before selecting top N")
    args = ap.parse_args()

    if args.n_plots <= 0:
        raise ValueError("--n-plots must be > 0")

    ensure_dir(args.out_dir)

    events = load_events(
        top_csv=args.top,
        n_plots=args.n_plots,
        context_before_min=args.context_before_min,
        context_after_min=args.context_after_min,
        min_score=args.min_score,
    )
    rows_by_event = collect_points_for_events(args.points, events)

    manifest_records: List[dict] = []
    n_saved = 0
    for e in events:
        out_path = plot_event(e, rows_by_event.get(e.idx, []), args.out_dir)
        manifest_records.append(
            {
                "event_idx": e.idx,
                "MMSI": e.mmsi,
                "t_prev": e.t_prev.isoformat(),
                "t_curr": e.t_curr.isoformat(),
                "anomaly_score": e.anomaly_score,
                "duration_s": e.duration_s,
                "plot_path": out_path or "",
            }
        )
        if out_path:
            n_saved += 1

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    write_manifest(manifest_path, manifest_records)
    print(f"[done] figures saved: {n_saved}/{len(events)} in {args.out_dir}")


if __name__ == "__main__":
    main()
