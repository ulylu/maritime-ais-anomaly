#!/usr/bin/env python3
"""
prepare_features_for_iforest.py

Goal:
- Read output/ais_features.csv (pair-wise features)
- Clean it and make it ready for Method A (Isolation Forest)
- Save results to output/ as:
  1) ais_features_methodA.csv      (meta + scaled features)
  2) ais_features_methodA_X.csv    (only scaled numeric features)
  3) ais_features_methodA_meta.csv (only meta columns)

Notes:
- Keep comments simple (IELTS ~5.0).
- This script does not train the model. It only prepares the data.
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


# These columns are NOT model features.
# We keep them only for tracking and later lookup.
META_COLS: List[str] = ["MMSI", "t_prev", "t_curr"]

# We drop vessel_type for the baseline Method A (simple and safe).
DROP_COLS: List[str] = ["vessel_type"]


def ensure_dir(path: str) -> None:
    """Create folder if it does not exist."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def drop_existing_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Drop columns only if they exist."""
    keep = [c for c in cols if c in df.columns]
    if keep:
        return df.drop(columns=keep)
    return df


def prepare_features(input_csv: str, output_dir: str, log_dist: bool = True) -> None:
    # Output file base name
    out_base = os.path.join(output_dir, "ais_features_methodA")

    ensure_dir(out_base)

    print(f"[info] reading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[info] rows={len(df):,} cols={len(df.columns)}")

    # ----------------------------
    # 1) Build meta table
    # ----------------------------
    # If some meta columns are missing, create them as empty.
    meta = pd.DataFrame()
    for c in META_COLS:
        if c in df.columns:
            meta[c] = df[c]
        else:
            meta[c] = ""

    # ----------------------------
    # 2) Build feature table
    # ----------------------------
    feat = df.copy()
    feat = drop_existing_cols(feat, META_COLS)
    feat = drop_existing_cols(feat, DROP_COLS)

    # Convert all feature columns to numeric.
    # Bad values become NaN.
    feat = feat.apply(pd.to_numeric, errors="coerce")

    # Optional: log transform distance to reduce huge scale.
    if log_dist and "dist_m" in feat.columns:
        feat["dist_m"] = np.log1p(feat["dist_m"])

    # Drop columns that never change (constant).
    nunique = feat.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"[info] drop constant cols: {const_cols}")
        feat = feat.drop(columns=const_cols)

    # ----------------------------
    # 3) Handle missing values
    # ----------------------------
    # Use median for each column.
    med = feat.median(numeric_only=True)
    feat = feat.fillna(med)

    # If a column is still NaN (rare case), fill with 0.
    feat = feat.fillna(0.0)

    # ----------------------------
    # 4) Standardize (z-score)
    # ----------------------------
    means = feat.mean()
    stds = feat.std()

    # Avoid divide by zero.
    stds = stds.replace(0, 1.0)

    feat_scaled = (feat - means) / stds

    print(f"[info] final features: rows={len(feat_scaled):,} cols={len(feat_scaled.columns)}")

    # ----------------------------
    # 5) Save outputs
    # ----------------------------
    out_full = out_base + ".csv"
    out_x = out_base + "_X.csv"
    out_meta = out_base + "_meta.csv"

    # Full table: meta + scaled features
    pd.concat([meta, feat_scaled], axis=1).to_csv(out_full, index=False)

    # Only numeric matrix
    feat_scaled.to_csv(out_x, index=False)

    # Only meta
    meta.to_csv(out_meta, index=False)

    print("[done] saved:")
    print(f"  {out_full}")
    print(f"  {out_x}")
    print(f"  {out_meta}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare ais_features.csv for Isolation Forest (Method A).")
    ap.add_argument(
        "--input",
        default="output/ais_features.csv",
        help="Input CSV path (default: output/ais_features.csv).",
    )
    ap.add_argument(
        "--output-dir",
        default="output",
        help="Output folder (default: output).",
    )
    ap.add_argument(
        "--no-log-dist",
        action="store_true",
        help="Do not use log1p on dist_m.",
    )
    args = ap.parse_args()

    prepare_features(
        input_csv=args.input,
        output_dir=args.output_dir,
        log_dist=not args.no_log_dist,
    )


if __name__ == "__main__":
    main()