"""
classify_if_anomalies.py

Post-hoc classification of Isolation Forest anomalies into interpretable
categories.  Takes existing IF top-1 pct anomalies and assigns each to one
or more categories based on which features exceed fleet-wide thresholds.

Categories:  speed, turn, timegap, distance
Assignment:  multi-label (an IF anomaly can belong to multiple categories)

Inputs (defaults):
  method_A/output/ais_methodA_top1pct_explain.csv
  method_A/output/methodA_by_type/reference_stats.json

Outputs (in test/output/):
  all_classified.csv            — every IF anomaly with category labels
  {speed,turn,timegap,distance}_classified.csv — per-category subsets
  unclassified.csv              — IF anomalies not assigned to any category
  classification_summary.csv / .txt
  reference_stats.json          — copied for self-containment
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime

import numpy as np
import pandas as pd


CATEGORIES = ["speed", "turn", "timegap", "distance"]

# Feature → threshold-percentile key for each category.  Logic is OR within
# each category: if ANY listed feature exceeds the threshold the category is
# assigned.
RULES = {
    "speed":    {"sog_curr": "p95", "abs_dsog": "p95"},
    "turn":     {"abs_dcog": "p95"},
    "timegap":  {"delta_t_s": "p95"},
    "distance": {"dist_m": "p95", "implied_kts": "p95"},
}

DESCRIPTIONS = {
    "speed":    "high reported SOG or sudden speed change (sog_curr or |dsog| > fleet {pct})",
    "turn":     "large course change (|dcog| > fleet {pct})",
    "timegap":  "long gap between pings (delta_t_s > fleet {pct})",
    "distance": "large position jump (dist_m or implied_kts > fleet {pct})",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _thr(ref, feat, pct):
    """Look up a fleet-wide threshold from reference_stats.json."""
    return ref.get(feat, {}).get(pct, float("inf"))


def run(explain_csv, ref_json, out_dir, threshold_pct="p95"):
    ensure_dir(out_dir)

    print(f"Loading explain table: {explain_csv}")
    df = pd.read_csv(explain_csv)
    print(f"  {len(df):,} IF anomaly rows")

    print(f"Loading reference stats: {ref_json}")
    with open(ref_json) as f:
        ref = json.load(f)

    ref_copy = os.path.join(out_dir, "reference_stats.json")
    shutil.copy2(ref_json, ref_copy)
    print(f"  Copied reference_stats.json -> {ref_copy}")

    # Ensure implied_kts exists
    if "implied_kts" not in df.columns:
        dt = pd.to_numeric(df.get("delta_t_s", 0), errors="coerce").replace(0, np.nan)
        dist = pd.to_numeric(df.get("dist_m", 0), errors="coerce")
        df["implied_kts"] = (dist / dt * 1.94384).fillna(0)

    # Override threshold level in rules if requested
    rules = {}
    for cat, feat_map in RULES.items():
        rules[cat] = {feat: threshold_pct for feat in feat_map}

    # ── classify each row ─────────────────────────────────────────────────
    for cat in CATEGORIES:
        mask = pd.Series(False, index=df.index)
        for feat, pct in rules[cat].items():
            thr = _thr(ref, feat, pct)
            vals = pd.to_numeric(df.get(feat, 0), errors="coerce").fillna(0)
            mask = mask | (vals > thr)
        df[f"cat_{cat}"] = mask

    df["categories"] = df.apply(
        lambda r: ",".join(c for c in CATEGORIES if r[f"cat_{c}"]) or "unclassified",
        axis=1,
    )
    df["n_categories"] = df[[f"cat_{c}" for c in CATEGORIES]].sum(axis=1).astype(int)

    # ── category scores (how far above threshold, normalised) ─────────────
    for cat in CATEGORIES:
        scores = pd.Series(0.0, index=df.index)
        for feat in rules[cat]:
            thr = _thr(ref, feat, threshold_pct)
            thr = max(thr, 1e-9)
            vals = pd.to_numeric(df.get(feat, 0), errors="coerce").fillna(0)
            scores = np.maximum(scores, vals / thr)
        df[f"cat_score_{cat}"] = scores.round(4)

    # ── save combined CSV ─────────────────────────────────────────────────
    all_path = os.path.join(out_dir, "all_classified.csv")
    df.to_csv(all_path, index=False)
    print(f"\nAll classified -> {all_path}  ({len(df):,} rows)")

    # ── save per-category CSVs ────────────────────────────────────────────
    summary_rows = []
    for cat in CATEGORIES:
        sub = df[df[f"cat_{cat}"]].copy()
        sub["category_score"] = sub[f"cat_score_{cat}"]
        sub["percentile_rank"] = (
            sub["category_score"].rank(pct=True).mul(100).round(2)
        )
        sub = sub.sort_values("category_score", ascending=False).reset_index(drop=True)

        cat_path = os.path.join(out_dir, f"{cat}_classified.csv")
        sub.to_csv(cat_path, index=False)
        print(f"[{cat:10s}]  {len(sub):>8,} anomalies -> {cat_path}")

        thresholds_str = ", ".join(
            f"{feat} > {_thr(ref, feat, threshold_pct):.2f}"
            for feat in rules[cat]
        )
        summary_rows.append({
            "category":             cat,
            "n_anomalies":          len(sub),
            "pct_of_if_total":      round(100 * len(sub) / max(len(df), 1), 2),
            "threshold_level":      threshold_pct,
            "thresholds":           thresholds_str,
            "rule_description":     DESCRIPTIONS[cat].format(pct=threshold_pct),
            "assignment":           "multi-label (OR)",
        })

    # Unclassified
    unclass = df[df["n_categories"] == 0]
    if len(unclass) > 0:
        unclass_path = os.path.join(out_dir, "unclassified.csv")
        unclass.to_csv(unclass_path, index=False)
        print(f"[{'unclass':10s}]  {len(unclass):>8,} anomalies -> {unclass_path}")
    summary_rows.append({
        "category":             "unclassified",
        "n_anomalies":          len(unclass),
        "pct_of_if_total":      round(100 * len(unclass) / max(len(df), 1), 2),
        "threshold_level":      threshold_pct,
        "thresholds":           "-",
        "rule_description":     "no single feature exceeds fleet threshold",
        "assignment":           "-",
    })

    # ── summary CSV ───────────────────────────────────────────────────────
    sum_csv = os.path.join(out_dir, "classification_summary.csv")
    pd.DataFrame(summary_rows).to_csv(sum_csv, index=False)
    print(f"\nSummary CSV -> {sum_csv}")

    # ── summary TXT ───────────────────────────────────────────────────────
    sum_txt = os.path.join(out_dir, "classification_summary.txt")
    n_multi = int((df["n_categories"] > 1).sum())
    with open(sum_txt, "w", encoding="utf-8") as f:
        f.write("IF Anomaly Post-Hoc Classification Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp           : {datetime.now().isoformat()}\n")
        f.write(f"Input explain table : {explain_csv}\n")
        f.write(f"Reference stats     : {ref_json}\n")
        f.write(f"Total IF anomalies  : {len(df):,}\n")
        f.write(f"Threshold level     : {threshold_pct}\n")
        f.write(f"Assignment mode     : multi-label "
                f"(each anomaly can belong to 0+ categories)\n\n")

        f.write("Classification rules\n")
        f.write("-" * 50 + "\n")
        for cat in CATEGORIES:
            parts = []
            for feat in rules[cat]:
                thr = _thr(ref, feat, threshold_pct)
                parts.append(f"{feat} > {thr:.2f}")
            f.write(f"  {cat:10s} : {' OR '.join(parts)}\n")

        f.write(f"\nResults\n")
        f.write("-" * 50 + "\n")
        for s in summary_rows:
            f.write(f"  {s['category']:15s} : {s['n_anomalies']:>8,} anomalies "
                    f"({s['pct_of_if_total']:5.1f}%)\n")
        f.write(f"\n  Multi-category   : {n_multi:>8,} anomalies "
                f"({100 * n_multi / max(len(df), 1):.1f}%) "
                f"belong to 2+ categories\n")

        f.write(f"\nOutput files\n")
        f.write("-" * 50 + "\n")
        f.write(f"  {all_path}\n")
        for cat in CATEGORIES:
            f.write(f"  {os.path.join(out_dir, f'{cat}_classified.csv')}\n")
        if len(unclass) > 0:
            f.write(f"  {os.path.join(out_dir, 'unclassified.csv')}\n")
        f.write(f"  {sum_csv}\n")
        f.write(f"  {sum_txt}\n")
        f.write(f"  {ref_copy}\n")

    print(f"Summary TXT -> {sum_txt}")
    print("\nClassification done.")


def main():
    p = argparse.ArgumentParser(
        description="Classify IF anomalies into interpretable categories"
    )
    p.add_argument(
        "--explain",
        default="method_A/output/ais_methodA_top1pct_explain.csv",
    )
    p.add_argument(
        "--ref-stats",
        default="method_A/output/methodA_by_type/reference_stats.json",
    )
    p.add_argument("--out-dir", default="test/output")
    p.add_argument(
        "--threshold", default="p95",
        choices=["p50", "p95", "p99", "p999"],
        help="Fleet percentile to use as category threshold (default: p95)",
    )
    args = p.parse_args()

    for path in [args.explain, args.ref_stats]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)

    run(args.explain, args.ref_stats, args.out_dir, args.threshold)


if __name__ == "__main__":
    main()
