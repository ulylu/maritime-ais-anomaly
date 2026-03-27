#!/usr/bin/env python3
"""
train_iforest.py

Goal:
- Train Isolation Forest for Method A using prepared matrix
- Read:
  1) output/ais_features_methodA_X.csv
  2) output/ais_features_methodA_meta.csv
- Save:
  1) model file
  2) output/ais_methodA_scores.csv
  3) output/ais_methodA_top1pct.csv
  4) output/methodA_summary.txt

Notes:
- anomaly_score is defined as -score_samples(X)
  (larger value = more anomalous)
- Keep comments simple.
"""

import argparse
import os
import pickle
import time
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    import joblib  # type: ignore
except Exception:
    joblib = None


DEFAULT_X = "method_A/output/ais_features_methodA_X.csv"
DEFAULT_META = "method_A/output/ais_features_methodA_meta.csv"
DEFAULT_MODEL = "method_A/models/iforest_methodA.joblib"
DEFAULT_SCORES = "method_A/output/ais_methodA_scores.csv"
DEFAULT_TOP = "method_A/output/ais_methodA_top1pct.csv"
DEFAULT_SUMMARY = "method_A/output/methodA_summary.txt"

META_COLS = ["MMSI", "t_prev", "t_curr"]


def ensure_parent(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def parse_contamination(value: str) -> Union[str, float]:
    s = str(value).strip().lower()
    if s == "auto":
        return "auto"
    x = float(s)
    if not (0.0 < x <= 0.5):
        raise ValueError("contamination must be 'auto' or a float in (0, 0.5].")
    return x


def load_inputs(x_csv: str, meta_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[info] reading X: {x_csv}")
    X = pd.read_csv(x_csv)
    print(f"[info] reading meta: {meta_csv}")
    meta = pd.read_csv(meta_csv)

    if len(X) != len(meta):
        raise ValueError(
            f"Row mismatch: X has {len(X):,} rows but meta has {len(meta):,} rows."
        )

    # Keep only expected meta columns if present; create empty if missing.
    meta_fixed = pd.DataFrame(index=meta.index)
    for c in META_COLS:
        meta_fixed[c] = meta[c] if c in meta.columns else ""

    # Defensive numeric conversion (X should already be numeric matrix).
    X = X.apply(pd.to_numeric, errors="coerce")
    n_missing_before = int(X.isna().sum().sum())
    if n_missing_before > 0:
        print(f"[warn] X has {n_missing_before:,} missing values; filling with 0.0")
        X = X.fillna(0.0)

    # Drop constant columns if any (rare defensive step).
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"[warn] dropping constant feature cols at train time: {const_cols}")
        X = X.drop(columns=const_cols)

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns found in X.")

    print(f"[info] X shape: {X.shape[0]:,} rows x {X.shape[1]:,} cols")
    return X, meta_fixed


def save_model(model: IsolationForest, model_path: str, metadata: Dict[str, Any]) -> str:
    ensure_parent(model_path)

    payload = {
        "model": model,
        "metadata": metadata,
    }

    # Prefer joblib for sklearn models; fallback to pickle.
    if joblib is not None and model_path.lower().endswith((".joblib", ".pkl", ".pickle")):
        joblib.dump(payload, model_path)
    else:
        with open(model_path, "wb") as f:
            pickle.dump(payload, f)

    return model_path


def summarize_scores(scores: pd.Series) -> Dict[str, float]:
    q = scores.quantile([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "min": float(q.loc[0.0]),
        "p01": float(q.loc[0.01]),
        "p05": float(q.loc[0.05]),
        "p50": float(q.loc[0.5]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
        "max": float(q.loc[1.0]),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
    }


def train_iforest(
    x_csv: str,
    meta_csv: str,
    model_path: str,
    scores_csv: str,
    top_csv: str,
    summary_txt: str,
    n_estimators: int = 200,
    max_samples: Union[str, int, float] = "auto",
    contamination: Union[str, float] = "auto",
    random_state: int = 42,
    top_frac: float = 0.01,
    n_jobs: int = -1,
) -> None:
    if not (0.0 < top_frac < 1.0):
        raise ValueError("top_frac must be in (0, 1).")

    X, meta = load_inputs(x_csv, meta_csv)

    print("[info] training IsolationForest...")
    t0 = time.perf_counter()

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X)

    train_seconds = time.perf_counter() - t0
    print(f"[info] training done in {train_seconds:.3f} s")

    # sklearn score_samples: higher = more normal; lower = more abnormal
    raw_score = model.score_samples(X)
    anomaly_score = -raw_score

    scores_df = meta.copy()
    scores_df["anomaly_score"] = anomaly_score

    # Optional diagnostics (can help debugging / analysis later)
    try:
        decision = model.decision_function(X)
        scores_df["decision_function"] = decision
    except Exception:
        pass

    # Rank by anomaly score descending (larger = more anomalous)
    scores_df = scores_df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    n = len(scores_df)
    k = max(1, int(np.ceil(n * top_frac)))
    top_df = scores_df.head(k).copy()

    # Save outputs
    for p in [scores_csv, top_csv, summary_txt]:
        ensure_parent(p)
    saved_model_path = save_model(
        model,
        model_path,
        metadata={
            "feature_columns": X.columns.tolist(),
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_estimators": int(n_estimators),
            "max_samples": max_samples,
            "contamination": contamination,
            "random_state": int(random_state),
            "top_frac": float(top_frac),
            "train_seconds": float(train_seconds),
        },
    )

    # Keep required columns first in score files.
    required_first = [c for c in META_COLS if c in scores_df.columns] + ["anomaly_score"]
    rest_scores = [c for c in scores_df.columns if c not in required_first]
    scores_df = scores_df[required_first + rest_scores]

    required_first_top = [c for c in META_COLS if c in top_df.columns] + ["anomaly_score"]
    rest_top = [c for c in top_df.columns if c not in required_first_top]
    top_df = top_df[required_first_top + rest_top]

    scores_df.to_csv(scores_csv, index=False)
    top_df.to_csv(top_csv, index=False)

    stats = summarize_scores(scores_df["anomaly_score"])
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Method A (Isolation Forest) Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"x_csv: {x_csv}\n")
        f.write(f"meta_csv: {meta_csv}\n")
        f.write(f"model_path: {saved_model_path}\n")
        f.write(f"scores_csv: {scores_csv}\n")
        f.write(f"top_csv: {top_csv}\n")
        f.write("\n")
        f.write(f"n_samples: {X.shape[0]}\n")
        f.write(f"n_features: {X.shape[1]}\n")
        f.write(f"train_time_sec: {train_seconds:.6f}\n")
        f.write(f"top_fraction: {top_frac:.6f}\n")
        f.write(f"top_count: {k}\n")
        f.write("\n")
        f.write("IsolationForest params\n")
        f.write(f"- n_estimators: {n_estimators}\n")
        f.write(f"- max_samples: {max_samples}\n")
        f.write(f"- contamination: {contamination}\n")
        f.write(f"- random_state: {random_state}\n")
        f.write(f"- n_jobs: {n_jobs}\n")
        f.write("\n")
        f.write("anomaly_score stats (larger = more anomalous)\n")
        for k_stat in ["min", "p01", "p05", "p50", "p95", "p99", "max", "mean", "std"]:
            f.write(f"- {k_stat}: {stats[k_stat]:.8f}\n")

    print("[done] saved:")
    print(f"  model: {saved_model_path}")
    print(f"  scores: {scores_csv}")
    print(f"  top: {top_csv} (top {k:,} rows = {top_frac:.2%})")
    print(f"  summary: {summary_txt}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Isolation Forest for Method A AIS anomaly detection.")
    ap.add_argument("--x", default=DEFAULT_X, help=f"Feature matrix CSV (default: {DEFAULT_X})")
    ap.add_argument("--meta", default=DEFAULT_META, help=f"Meta CSV (default: {DEFAULT_META})")
    ap.add_argument("--model-out", default=DEFAULT_MODEL, help=f"Saved model path (default: {DEFAULT_MODEL})")
    ap.add_argument("--scores-out", default=DEFAULT_SCORES, help=f"Scores CSV path (default: {DEFAULT_SCORES})")
    ap.add_argument("--top-out", default=DEFAULT_TOP, help=f"Top anomalies CSV path (default: {DEFAULT_TOP})")
    ap.add_argument("--summary-out", default=DEFAULT_SUMMARY, help=f"Summary txt path (default: {DEFAULT_SUMMARY})")
    ap.add_argument("--n-estimators", type=int, default=200, help="Number of trees (default: 200)")
    ap.add_argument(
        "--max-samples",
        default="auto",
        help="IsolationForest max_samples (int/float/auto). Default: auto",
    )
    ap.add_argument(
        "--contamination",
        default="auto",
        help="IsolationForest contamination (auto or float in (0,0.5]). Default: auto",
    )
    ap.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--top-frac", type=float, default=0.01, help="Top fraction to export (default: 0.01)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for sklearn (default: -1)")
    args = ap.parse_args()

    # Parse max_samples to int/float/auto if possible.
    max_samples: Union[str, int, float]
    ms = str(args.max_samples).strip().lower()
    if ms == "auto":
        max_samples = "auto"
    else:
        # Try int first, then float.
        try:
            max_samples = int(ms)
        except ValueError:
            max_samples = float(ms)

    contamination = parse_contamination(args.contamination)

    train_iforest(
        x_csv=args.x,
        meta_csv=args.meta,
        model_path=args.model_out,
        scores_csv=args.scores_out,
        top_csv=args.top_out,
        summary_txt=args.summary_out,
        n_estimators=args.n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=args.random_state,
        top_frac=args.top_frac,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
