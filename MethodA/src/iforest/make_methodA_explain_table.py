#!/usr/bin/env python3
"""
make_methodA_explain_table.py

Goal:
- Join Method A top anomaly scores back to original pair-wise features
- Export a readable table for analysis / report writing

Default inputs:
- output/ais_methodA_top1pct.csv
- output/ais_features.csv

Default output:
- output/ais_methodA_top1pct_explain.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


KEY_COLS = ["MMSI", "t_prev", "t_curr"]


def ensure_parent(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Absolute turn/change magnitudes are easier to read in analysis.
    for c in ["dcog", "dheading", "dsog"]:
        if c in out.columns:
            out[f"abs_{c}"] = pd.to_numeric(out[c], errors="coerce").abs()

    if "dist_m" in out.columns and "delta_t_s" in out.columns:
        dist = pd.to_numeric(out["dist_m"], errors="coerce")
        dt = pd.to_numeric(out["delta_t_s"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            implied_mps = dist / dt
        out["implied_mps"] = implied_mps.replace([np.inf, -np.inf], np.nan)
        out["implied_kts"] = out["implied_mps"] * 1.943844

    if "anomaly_score" in out.columns:
        out["anomaly_rank"] = np.arange(1, len(out) + 1, dtype=np.int64)
        if len(out) > 1:
            out["anomaly_rank_pct"] = out["anomaly_rank"] / len(out)
        else:
            out["anomaly_rank_pct"] = 1.0

    return out


def build_explain_table(
    top_csv: str,
    features_csv: str,
    output_csv: str,
    keep_top_n: int | None = None,
) -> None:
    print(f"[info] reading top scores: {top_csv}")
    top = pd.read_csv(top_csv)

    if keep_top_n is not None:
        top = top.head(keep_top_n).copy()
        print(f"[info] keep top N = {len(top):,}")

    missing_key_top = [c for c in KEY_COLS if c not in top.columns]
    if missing_key_top:
        raise ValueError(f"Top score file missing keys: {missing_key_top}")

    print(f"[info] reading original features: {features_csv}")
    feat = pd.read_csv(features_csv)

    missing_key_feat = [c for c in KEY_COLS if c not in feat.columns]
    if missing_key_feat:
        raise ValueError(f"Feature file missing keys: {missing_key_feat}")

    print("[info] joining top anomalies with original feature rows...")
    merged = top.merge(
        feat,
        on=KEY_COLS,
        how="left",
        suffixes=("", "_feat"),
        indicator=True,
    )

    n_missing = int((merged["_merge"] != "both").sum())
    if n_missing > 0:
        print(f"[warn] {n_missing:,} rows in top file did not match features by keys.")
    merged = merged.drop(columns=["_merge"])

    merged = add_derived_columns(merged)

    # Put important columns first.
    preferred_front = [
        "MMSI",
        "t_prev",
        "t_curr",
        "anomaly_score",
        "decision_function",
        "anomaly_rank",
        "anomaly_rank_pct",
        "delta_t_s",
        "dist_m",
        "implied_mps",
        "implied_kts",
        "dsog",
        "abs_dsog",
        "dcog",
        "abs_dcog",
        "dheading",
        "abs_dheading",
        "vessel_type",
    ]
    front = [c for c in preferred_front if c in merged.columns]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]

    ensure_parent(output_csv)
    merged.to_csv(output_csv, index=False)
    print(f"[done] saved explain table: {output_csv} (rows={len(merged):,}, cols={len(merged.columns)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Join Method A top anomalies back to original features.")
    ap.add_argument("--top", default="method_A/output/ais_methodA_top1pct.csv", help="Top anomaly CSV")
    ap.add_argument("--features", default="preprocessing/output/ais_features.csv", help="Original pair-wise features CSV")
    ap.add_argument(
        "--out",
        default="method_A/output/ais_methodA_top1pct_explain.csv",
        help="Output explain table CSV",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Optional: keep only first N rows from top anomaly file before join",
    )
    args = ap.parse_args()

    build_explain_table(
        top_csv=args.top,
        features_csv=args.features,
        output_csv=args.out,
        keep_top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
