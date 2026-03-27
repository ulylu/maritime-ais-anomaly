"""
plot_if_categories.py

Category-specific anomaly figures for the IF-classified anomaly pipeline.

Replicates the visual standard of plot_methodA_by_type.py exactly:
  speed    -> SOG vs time  (anomaly window red, fleet p99/p95 reference)
  turn     -> |course change| vs time
  timegap  -> ping timeline + bar comparison  (two-panel)
  distance -> segment distance vs time + lat/lon jump view  (two-panel)

Reads per-category CSVs produced by classify_if_anomalies.py.
Run after classify_if_anomalies.py.
"""

import os
import sys
import csv
import json
import argparse
import random
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch


# ── utilities (identical to plot_methodA_by_type.py) ──────────────────────────

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_iso(s):
    s = str(s).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def wrap_delta(a, b):
    return (b - a + 180) % 360 - 180


def fmt_hm(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.figure.autofmt_xdate()


def add_ref_line(ax, value, label, color="orange", linestyle="--"):
    if value is None:
        return
    ax.axhline(value, color=color, linestyle=linestyle, lw=1.2, alpha=0.8,
               label=label)


# ── event loading ─────────────────────────────────────────────────────────────

def load_events(csv_path, n_top, n_random, seed=42):
    """
    Load top and random anomaly events from a category-classified CSV.

    Sorts by category_score (falling back to anomaly_score).
    Deduplicates by MMSI for top selection to ensure vessel diversity.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    sort_col = "category_score" if "category_score" in df.columns else "anomaly_score"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    dedup = df.drop_duplicates(subset=["MMSI"], keep="first").reset_index(drop=True)
    top_events = dedup.head(n_top).to_dict("records")
    for e in top_events:
        e["sample_kind"] = "top"

    top_mmsi = {str(e["MMSI"]) for e in top_events}
    remainder = df[~df["MMSI"].astype(str).isin(top_mmsi)].reset_index(drop=True)
    n_rand = min(n_random, len(remainder))
    if n_rand > 0:
        rand_rows = remainder.sample(n=n_rand, random_state=seed)
        rand_events = rand_rows.to_dict("records")
        for e in rand_events:
            e["sample_kind"] = "random"
    else:
        rand_events = []

    return top_events + rand_events


# ── AIS point streaming ──────────────────────────────────────────────────────

def stream_vessel_points(points_csv, mmsi_set):
    data = defaultdict(list)
    with open(points_csv, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
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
                cog_raw = row.get("COG", "") or ""
                cog = float(cog_raw) if cog_raw.strip() else 0.0
            except (ValueError, KeyError):
                continue
            data[mmsi].append((t, lat, lon, sog, cog))
    for mmsi in data:
        data[mmsi].sort(key=lambda x: x[0])
    return data


# ── window helpers ────────────────────────────────────────────────────────────

def get_context(points, t_prev, t_curr, before_s, after_s):
    lo = t_prev - timedelta(seconds=before_s)
    hi = t_curr + timedelta(seconds=after_s)
    return [p for p in points if lo <= p[0] <= hi]


def split_by_window(points, t_prev, t_curr):
    before = [p for p in points if p[0] <  t_prev]
    inside = [p for p in points if t_prev <= p[0] <= t_curr]
    after  = [p for p in points if p[0] >  t_curr]
    return before, inside, after


# ── plot functions (replicated from plot_methodA_by_type.py) ──────────────────

def plot_speed(event, vessel_points, out_path, ref):
    """SOG vs time.  Anomaly window in red.  Fleet p99/p95 reference lines."""
    t_prev = parse_iso(event["t_prev"])
    t_curr = parse_iso(event["t_curr"])
    if t_prev is None or t_curr is None:
        return False

    ctx = get_context(vessel_points, t_prev, t_curr, before_s=3600, after_s=3600)
    if len(ctx) < 2:
        return False

    before, inside, after = split_by_window(ctx, t_prev, t_curr)

    fig, ax = plt.subplots(figsize=(11, 4))

    for seg, color, lw, label in [
        (before, "steelblue", 1.0, "context (before)"),
        (after,  "steelblue", 1.0, "_nolegend_"),
        (inside, "red",       2.5, "anomaly window"),
    ]:
        if seg:
            ax.plot([p[0] for p in seg], [p[3] for p in seg],
                    "o-", color=color, lw=lw, ms=3, label=label)

    if t_prev != t_curr:
        ax.axvspan(t_prev, t_curr, color="red", alpha=0.10)

    p99_sog = ref.get("sog_curr", {}).get("p99")
    p95_sog = ref.get("sog_curr", {}).get("p95")
    add_ref_line(ax, p99_sog, f"fleet p99 SOG ({p99_sog:.1f} kts)")
    add_ref_line(ax, p95_sog, f"fleet p95 SOG ({p95_sog:.1f} kts)",
                 color="gold", linestyle=":")

    sog_val   = float(event.get("sog_curr", 0) or 0)
    dsog_val  = float(event.get("abs_dsog", 0) or 0)
    score_val = float(event.get("anomaly_score", 0))
    pct_val   = float(event.get("percentile_rank", 0) or 0)

    if inside:
        peak_t   = max(inside, key=lambda p: p[3])
        peak_sog = peak_t[3]
        ax.annotate(
            f"{peak_sog:.1f} kts",
            xy=(peak_t[0], peak_sog),
            xytext=(10, 10), textcoords="offset points",
            fontsize=9, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        )

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Speed Over Ground (knots)")
    ax.set_title(
        f"Speed Anomaly \u2014 MMSI {event['MMSI']}  "
        f"[score={score_val:.2f}, p{pct_val:.1f}]\n"
        f"SOG = {sog_val:.1f} kts  |  |\u0394SOG| = {dsog_val:.1f} kts  "
        f"|  fleet p99 = {p99_sog:.1f} kts",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fmt_hm(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def plot_turn(event, vessel_points, out_path, ref):
    """|Course change| vs time.  Anomaly window in red."""
    t_prev = parse_iso(event["t_prev"])
    t_curr = parse_iso(event["t_curr"])
    if t_prev is None or t_curr is None:
        return False

    ctx = get_context(vessel_points, t_prev, t_curr, before_s=3600, after_s=3600)
    if len(ctx) < 3:
        return False

    pair_t    = []
    pair_dcog = []
    for i in range(1, len(ctx)):
        _, _, _, _, cog_a = ctx[i - 1]
        t_b, _, _, _, cog_b = ctx[i]
        pair_t.append(t_b)
        pair_dcog.append(abs(wrap_delta(cog_a, cog_b)))

    ctx_t  = [t for t in pair_t if t < t_prev or t > t_curr]
    ctx_v  = [pair_dcog[i] for i, t in enumerate(pair_t) if t < t_prev or t > t_curr]
    anom_t = [t for t in pair_t if t_prev <= t <= t_curr]
    anom_v = [pair_dcog[i] for i, t in enumerate(pair_t) if t_prev <= t <= t_curr]

    fig, ax = plt.subplots(figsize=(11, 4))

    if ctx_t:
        ax.plot(ctx_t, ctx_v, "o-", color="steelblue", lw=1, ms=3,
                label="context", zorder=1)
    if anom_t:
        ax.plot(anom_t, anom_v, "o-", color="red", lw=2.5, ms=6,
                label="anomaly window", zorder=2)
        ax.axvspan(t_prev, t_curr, color="red", alpha=0.10)

    p99_dcog = ref.get("abs_dcog", {}).get("p99")
    add_ref_line(ax, p99_dcog, f"fleet p99 |\u0394cog| ({p99_dcog:.1f}\u00b0)")

    dcog_val  = float(event.get("abs_dcog", 0) or 0)
    score_val = float(event.get("anomaly_score", 0))
    pct_val   = float(event.get("percentile_rank", 0) or 0)

    if anom_t and anom_v:
        peak_idx = int(np.argmax(anom_v))
        ax.annotate(
            f"{anom_v[peak_idx]:.0f}\u00b0",
            xy=(anom_t[peak_idx], anom_v[peak_idx]),
            xytext=(10, 10), textcoords="offset points",
            fontsize=9, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        )

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("|Course Change| (degrees)")
    ax.set_title(
        f"Turn Anomaly \u2014 MMSI {event['MMSI']}  "
        f"[score={score_val:.2f}, p{pct_val:.1f}]\n"
        f"|\u0394cog| = {dcog_val:.1f}\u00b0  |  fleet p99 = {p99_dcog:.1f}\u00b0",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fmt_hm(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def plot_timegap(event, vessel_points, out_path, ref):
    """Two-panel: ping timeline (left) + bar comparison (right)."""
    t_prev = parse_iso(event["t_prev"])
    t_curr = parse_iso(event["t_curr"])
    if t_prev is None or t_curr is None:
        return False

    dt_s   = float(event.get("delta_t_s", 0) or 0)
    dt_min = dt_s / 60.0
    score_val = float(event.get("anomaly_score", 0))
    pct_val   = float(event.get("percentile_rank", 0) or 0)

    ctx = get_context(vessel_points, t_prev, t_curr, before_s=3600, after_s=3600)

    fig, (ax_tl, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 4),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # Left: ping timeline
    before = [p for p in ctx if p[0] <= t_prev]
    after  = [p for p in ctx if p[0] >= t_curr]

    for grp, color in [(before, "steelblue"), (after, "steelblue")]:
        if grp:
            ax_tl.vlines([p[0] for p in grp], -0.4, 0.4,
                         colors=color, lw=1.2, alpha=0.7)

    ax_tl.axvspan(t_prev, t_curr, color="red", alpha=0.18, label="silent period")

    mid = t_prev + (t_curr - t_prev) / 2
    ax_tl.annotate(
        "", xy=(t_curr, 0.78), xytext=(t_prev, 0.78),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="<->", color="red", lw=2),
    )
    ax_tl.text(
        mid, 0.84,
        f"{dt_min:.0f} min silence",
        transform=ax_tl.get_xaxis_transform(),
        ha="center", color="red", fontsize=10, fontweight="bold",
    )

    ax_tl.set_ylim(-1, 1)
    ax_tl.set_yticks([])
    ax_tl.set_xlabel("Time (UTC)")
    ax_tl.set_title(
        f"Ping timeline \u2014 MMSI {event['MMSI']}  "
        f"[score={score_val:.2f}, p{pct_val:.1f}]\n"
        f"Each tick = one AIS ping. Red = silent period ({dt_min:.0f} min).",
        fontsize=9,
    )
    ax_tl.legend(fontsize=8)
    fmt_hm(ax_tl)

    # Right: bar comparison
    p50_s = ref.get("delta_t_s", {}).get("p50",  60)
    p99_s = ref.get("delta_t_s", {}).get("p99", 300)

    labels = ["Fleet\nmedian", "Fleet\np99", "This\ngap"]
    values = [p50_s / 60.0, p99_s / 60.0, dt_min]
    colors = ["steelblue", "orange", "red"]

    bars = ax_bar.bar(labels, values, color=colors, alpha=0.85, width=0.5)
    ax_bar.set_ylabel("Gap duration (minutes)")
    ax_bar.set_title("Comparison", fontsize=9)

    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.0f} min",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def plot_distance(event, vessel_points, out_path, ref):
    """Two-panel: segment distance vs time (left) + lat/lon jump (right)."""
    t_prev = parse_iso(event["t_prev"])
    t_curr = parse_iso(event["t_curr"])
    if t_prev is None or t_curr is None:
        return False

    ctx = get_context(vessel_points, t_prev, t_curr, before_s=3600, after_s=3600)
    if len(ctx) < 3:
        return False

    pair_t    = []
    pair_dist = []
    for i in range(1, len(ctx)):
        _, lat1, lon1, _, _ = ctx[i - 1]
        t2, lat2, lon2, _, _ = ctx[i]
        pair_t.append(t2)
        pair_dist.append(haversine_m(lat1, lon1, lat2, lon2))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 4))

    for t, dist in zip(pair_t, pair_dist):
        in_anom = t_prev <= t <= t_curr
        ax_left.vlines(t, 0, dist,
                       colors="red" if in_anom else "steelblue",
                       lw=4 if in_anom else 1,
                       alpha=0.9 if in_anom else 0.5)

    p99_dist = ref.get("dist_m", {}).get("p99")
    add_ref_line(ax_left, p99_dist, f"fleet p99 dist ({p99_dist/1000:.1f} km)")

    ax_left.set_xlabel("Time (UTC)")
    ax_left.set_ylabel("Segment distance (m)")
    ax_left.set_title("Distance per segment")
    ax_left.legend(
        handles=[Patch(color="red", label="anomaly jump"),
                 Patch(color="steelblue", label="normal"),
                 Patch(color="orange",
                       label=f"fleet p99 ({p99_dist/1000:.1f} km)")],
        fontsize=8,
    )
    fmt_hm(ax_left)

    before, inside, after = split_by_window(ctx, t_prev, t_curr)
    for seg, color, ms, label in [
        (before, "steelblue", 3, "before"),
        (after,  "lightblue", 3, "after"),
        (inside, "red",       5, "anomaly"),
    ]:
        if seg:
            ax_right.plot([p[2] for p in seg], [p[1] for p in seg],
                          "o-", color=color, ms=ms, lw=1, label=label)

    if before and inside:
        x0, y0 = before[-1][2], before[-1][1]
        x1, y1 = inside[0][2],  inside[0][1]
        ax_right.annotate("", xy=(x1, y1), xytext=(x0, y0),
                          arrowprops=dict(arrowstyle="->", color="red", lw=2))

    ax_right.set_xlabel("Longitude")
    ax_right.set_ylabel("Latitude")
    ax_right.set_title("Lat/Lon view (jump highlighted)")
    ax_right.legend(fontsize=8)

    dist_m    = float(event.get("dist_m", 0) or 0)
    impl_kts  = float(event.get("implied_kts", 0) or 0)
    score_val = float(event.get("anomaly_score", 0))
    pct_val   = float(event.get("percentile_rank", 0) or 0)

    fig.suptitle(
        f"Distance/Jump Anomaly \u2014 MMSI {event['MMSI']}  "
        f"[score={score_val:.2f}, p{pct_val:.1f}]\n"
        f"Jump = {dist_m/1000:.1f} km  |  implied speed = {impl_kts:.1f} kts  "
        f"|  fleet p99 = {p99_dist/1000:.1f} km",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# ── dispatch ──────────────────────────────────────────────────────────────────

PLOT_FN = {
    "speed":    plot_speed,
    "turn":     plot_turn,
    "timegap":  plot_timegap,
    "distance": plot_distance,
}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Category-specific anomaly plots for IF-classified pipeline"
    )
    parser.add_argument("--classified-dir", default="test/output")
    parser.add_argument("--points",
                        default="preprocessing/output/ais_points_clean.csv")
    parser.add_argument("--out-dir", default="test/output/figs")
    parser.add_argument("--n-top", type=int, default=5)
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.points):
        print(f"ERROR: AIS points file not found: {args.points}", file=sys.stderr)
        sys.exit(1)

    ref_path = os.path.join(args.classified_dir, "reference_stats.json")
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref_stats = json.load(f)
        print(f"Loaded reference stats from {ref_path}")
    else:
        print(f"WARNING: reference_stats.json not found at {ref_path}. "
              f"Reference lines will be omitted.")
        ref_stats = {}

    all_events_by_type = {}
    all_mmsi = set()
    for type_name in PLOT_FN:
        csv_path = os.path.join(args.classified_dir,
                                f"{type_name}_classified.csv")
        if not os.path.exists(csv_path):
            print(f"[{type_name}] classified CSV not found, skipping: {csv_path}")
            continue
        events = load_events(csv_path, args.n_top, args.n_random,
                             args.random_seed)
        all_events_by_type[type_name] = events
        for e in events:
            all_mmsi.add(str(e["MMSI"]))
        print(f"[{type_name}] {len(events)} events "
              f"({args.n_top} top + up to {args.n_random} random)")

    if not all_mmsi:
        print("No events found.  Run classify_if_anomalies.py first.")
        return

    print(f"\nStreaming points for {len(all_mmsi)} vessels "
          f"from {args.points} ...")
    vessel_data = stream_vessel_points(args.points, all_mmsi)
    print(f"  Collected data for {len(vessel_data)} vessels.")

    for type_name, events in all_events_by_type.items():
        type_dir = os.path.join(args.out_dir, type_name)
        ensure_dir(type_dir)
        plot_fn = PLOT_FN[type_name]

        top_counter  = 0
        rand_counter = 0
        manifest_rows = []

        for event in events:
            mmsi = str(event["MMSI"])
            kind = event.get("sample_kind", "top")
            vpts = vessel_data.get(mmsi, [])

            if kind == "top":
                top_counter += 1
                fname = f"{type_name}_top_{top_counter:03d}_mmsi_{mmsi}.png"
            else:
                rand_counter += 1
                fname = f"{type_name}_random_{rand_counter:03d}_mmsi_{mmsi}.png"

            out_path = os.path.join(type_dir, fname)
            ok = plot_fn(event, vpts, out_path, ref_stats)
            note = "saved" if ok else "skipped (not enough context points)"
            print(f"  [{type_name}] {fname}: {note}")

            manifest_rows.append({
                "type":            type_name,
                "kind":            kind,
                "mmsi":            mmsi,
                "t_prev":          event.get("t_prev"),
                "t_curr":          event.get("t_curr"),
                "anomaly_score":   event.get("anomaly_score"),
                "category_score":  event.get("category_score"),
                "percentile_rank": event.get("percentile_rank"),
                "status":          note,
                "file":            out_path if ok else "",
            })

        mfst_path = os.path.join(type_dir, "manifest.csv")
        pd.DataFrame(manifest_rows).to_csv(mfst_path, index=False)
        print(f"  [{type_name}] manifest -> {mfst_path}")

    print("\nPlotting done.")


if __name__ == "__main__":
    main()
