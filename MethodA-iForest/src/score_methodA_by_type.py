"""
score_methodA_by_type.py

Method A by-type anomaly scoring pipeline.

Each anomaly type uses ONE-SIDED scoring: only flags values that are
unusually HIGH (large gap, large turn, large speed, large jump).
Two-sided scoring caused short 1-second gaps to score as high as 6-hour
gaps, which is misleading.

Anomaly types:
  speed    — unusually high reported SOG or unusually large speed change
  turn     — unusually large course change (abs_dcog)
  timegap  — unusually LONG gap between pings (not short gaps)
  distance — unusually large position jump between pings

Scoring: one-sided robust z-score.
  score = max(0, x - median) / (1.4826 * MAD)
  Larger score = more anomalous. Only values above the median score > 0.

Memory note: use usecols + float32 to keep RAM around 2-3 GB for 34 M rows.
Full per-row score files are NOT saved by default (they are 4-5 GB each).
Use --save-scores to opt in.

Run order:
  preprocess_ais.py -> extract_features.py -> score_methodA_by_type.py
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def one_sided_zscore(series):
    """
    One-sided robust z-score: score = max(0, x - median) / (1.4826 * MAD).

    Only values ABOVE the median receive a positive score.
    Values at or below the median score 0 (not flagged).
    This prevents short time gaps from scoring equally to long ones,
    and prevents low speeds from scoring equally to high ones.
    """
    arr = series.to_numpy(dtype=np.float64)
    median = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - median))
    if mad == 0:
        std = np.nanstd(arr)
        mad = std if std > 0 else 1.0
    score = np.maximum(0.0, arr - median) / (1.4826 * mad)
    return pd.Series(score, index=series.index)


def percentile_rank(series):
    """Percentile rank 0-100. Higher = more extreme."""
    return series.rank(pct=True) * 100


def select_top_fraction(df, score_col, frac=0.01):
    """Return the top `frac` rows by score, sorted highest first."""
    threshold = df[score_col].quantile(1.0 - frac)
    top = df[df[score_col] >= threshold].copy()
    top = top.sort_values(score_col, ascending=False).reset_index(drop=True)
    return top, float(threshold)


def implied_speed_kts(df):
    """Implied speed (knots) from distance and time gap."""
    safe_dt = df['delta_t_s'].astype(float).replace(0.0, np.nan)
    mps = df['dist_m'].astype(float) / safe_dt
    return (mps * 1.94384).fillna(0.0)


# ── per-type scorers ──────────────────────────────────────────────────────────

def score_speed(df):
    """
    Speed anomaly: unusually high reported speed (sog_curr) or unusually
    large speed change (abs_dsog).

    AIS sentinel values: SOG >= 102.2 means 'speed not available' in the
    AIS protocol. These are NOT real speeds and must be excluded before
    scoring, otherwise data-quality gaps dominate the top anomaly list.

    Similarly, if sog_prev was the sentinel, the computed abs_dsog is
    meaningless (e.g., 5 kts → 102.3 kts gives dsog = 97.3, which is
    an artifact, not a real speed change). We zero out dsog in those cases.

    One-sided scoring: only flags HIGH speed / HIGH speed change.
    """
    # Any SOG above this is physically implausible for commercial vessels and
    # is treated as a bad data point.  The AIS sentinel is 102.2 kts, but
    # values like 95-101 kts also appear and are clearly erroneous.
    # 40 knots covers fast ferries, patrol boats, and high-speed craft.
    SOG_MAX_PLAUSIBLE = 40.0

    sog = df['sog_curr'].fillna(0).astype(float)
    sog_clean = sog.where(sog <= SOG_MAX_PLAUSIBLE, other=0.0)

    dsog = df['abs_dsog'].astype(float)
    # Zero out abs_dsog wherever either endpoint had an implausible speed,
    # because the computed change is meaningless (e.g. 101 kts → 1 kt).
    dsog_clean = dsog.where(sog <= SOG_MAX_PLAUSIBLE, other=0.0)
    if 'sog_prev' in df.columns:
        sog_prev = df['sog_prev'].fillna(0).astype(float)
        dsog_clean = dsog_clean.where(sog_prev <= SOG_MAX_PLAUSIBLE, other=0.0)

    z_sog  = one_sided_zscore(sog_clean)
    z_dsog = one_sided_zscore(dsog_clean)
    score  = np.maximum(z_sog.values, z_dsog.values)

    out = df[['MMSI', 't_prev', 't_curr']].copy()
    out['anomaly_type']    = 'speed'
    out['anomaly_score']   = score
    # Store cleaned values (sentinel removed) so the explain table is readable
    out['sog_curr']        = sog_clean.round(2)
    out['abs_dsog']        = dsog_clean.round(2)
    out['sog_curr_raw']    = df['sog_curr'].round(2)   # raw AIS value for reference
    out['percentile_rank'] = percentile_rank(out['anomaly_score']).round(2)
    out['reason_text']     = 'unusually high speed or sudden speed change'
    return out


def score_turn(df):
    """
    Turn anomaly: unusually large course change (abs_dcog in degrees).

    abs_dcog is already >= 0 so the one-sided scorer naturally only flags
    large turns, not small ones.
    """
    score = one_sided_zscore(df['abs_dcog'])

    out = df[['MMSI', 't_prev', 't_curr']].copy()
    out['anomaly_type']    = 'turn'
    out['anomaly_score']   = score
    out['abs_dcog']        = df['abs_dcog'].round(2)
    out['cog_prev']        = df['cog_prev'].round(2)
    out['cog_curr']        = df['cog_curr'].round(2)
    out['percentile_rank'] = percentile_rank(out['anomaly_score']).round(2)
    out['reason_text']     = 'unusually large course change'
    return out


def score_timegap(df):
    """
    Time-gap anomaly: unusually LONG gap between consecutive pings.

    Previously used two-sided scoring which flagged short 1-second gaps
    equally to long 6-hour gaps. Now one-sided: only values ABOVE the
    median (= longer than typical) receive a score > 0.

    Log scale compresses the very wide range (1 second to 6 hours).
    """
    log_dt = np.log1p(df['delta_t_s'].astype(float))
    score  = one_sided_zscore(log_dt)

    out = df[['MMSI', 't_prev', 't_curr']].copy()
    out['anomaly_type']    = 'timegap'
    out['anomaly_score']   = score
    out['delta_t_s']       = df['delta_t_s'].round(1)
    out['delta_t_min']     = (df['delta_t_s'].astype(float) / 60.0).round(2)
    out['percentile_rank'] = percentile_rank(out['anomaly_score']).round(2)
    out['reason_text']     = 'unusually long time gap between pings'
    return out


def score_distance(df):
    """
    Distance / jump anomaly: unusually large position jump between pings,
    also captured through implied speed (dist / time).

    This also absorbs cases where implied_kts is very high due to a position
    jump — previously those were bleeding into speed anomaly scores.
    """
    impl_kts = implied_speed_kts(df)
    z_dist    = one_sided_zscore(df['dist_m'].astype(float))
    z_implied = one_sided_zscore(impl_kts)
    score     = np.maximum(z_dist.values, z_implied.values)

    out = df[['MMSI', 't_prev', 't_curr']].copy()
    out['anomaly_type']    = 'distance'
    out['anomaly_score']   = score
    out['dist_m']          = df['dist_m'].round(1)
    out['implied_kts']     = impl_kts.round(2)
    out['percentile_rank'] = percentile_rank(out['anomaly_score']).round(2)
    out['reason_text']     = 'unusually large position jump between pings'
    return out


# ── reference stats (saved for use by the plot script) ────────────────────────

def compute_reference_stats(df):
    """
    Compute fleet-wide percentiles for each feature used in scoring.
    These are saved to a JSON file and used by the plot script to draw
    reference lines showing 'what is normal for the whole dataset'.
    """
    impl_kts = implied_speed_kts(df)
    stats = {}

    sog_clean_ref = df['sog_curr'].fillna(0).astype(float)
    sog_clean_ref = sog_clean_ref.where(sog_clean_ref < 102.0, other=0.0)

    for name, series in [
        ('sog_curr',    sog_clean_ref),
        ('abs_dsog',    df['abs_dsog']),
        ('abs_dcog',    df['abs_dcog']),
        ('delta_t_s',   df['delta_t_s'].astype(float)),
        ('delta_t_min', df['delta_t_s'].astype(float) / 60.0),
        ('dist_m',      df['dist_m'].astype(float)),
        ('implied_kts', impl_kts),
    ]:
        arr = series.dropna().to_numpy()
        stats[name] = {
            'p50':  float(np.percentile(arr, 50)),
            'p95':  float(np.percentile(arr, 95)),
            'p99':  float(np.percentile(arr, 99)),
            'p999': float(np.percentile(arr, 99.9)),
            'max':  float(np.max(arr)),
        }

    return stats


# ── dispatch ──────────────────────────────────────────────────────────────────

SCORERS = {
    'speed':    score_speed,
    'turn':     score_turn,
    'timegap':  score_timegap,
    'distance': score_distance,
}

NEEDED_COLS = ['MMSI', 't_prev', 't_curr',
               'delta_t_s', 'dist_m',
               'sog_prev', 'sog_curr', 'dsog',
               'dcog', 'cog_prev', 'cog_curr']

FLOAT32_COLS = {
    'delta_t_s': 'float32',
    'dist_m':    'float32',
    'sog_prev':  'float32',
    'sog_curr':  'float32',
    'dsog':      'float32',
    'dcog':      'float32',
    'cog_prev':  'float32',
    'cog_curr':  'float32',
}


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(features_csv, out_dir, top_frac, types_to_run, save_scores):
    ensure_dir(out_dir)
    t_start = datetime.now()

    print(f'Loading features from: {features_csv}')
    df = pd.read_csv(
        features_csv,
        usecols=NEEDED_COLS,
        dtype=FLOAT32_COLS,
        low_memory=False,
    )
    print(f'  Loaded {len(df):,} rows | '
          f'RAM: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB')

    df['abs_dcog'] = df['dcog'].abs()
    df['abs_dsog'] = df['dsog'].abs()

    before = len(df)
    df = df.dropna(subset=['delta_t_s', 'dist_m'])
    if len(df) < before:
        print(f'  Dropped {before - len(df):,} rows missing delta_t_s/dist_m')

    for col in ['sog_curr', 'dsog', 'abs_dsog', 'dcog', 'abs_dcog',
                'cog_prev', 'cog_curr']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Save fleet-wide reference stats (used by plot script for reference lines)
    print('Computing reference stats for plot reference lines ...')
    ref_stats = compute_reference_stats(df)
    ref_path = os.path.join(out_dir, 'reference_stats.json')
    with open(ref_path, 'w') as f:
        json.dump(ref_stats, f, indent=2)
    print(f'  Reference stats → {ref_path}')

    summaries = []

    for type_name, scorer_fn in SCORERS.items():
        if type_name not in types_to_run:
            print(f'\n[{type_name}] skipped')
            continue

        print(f'\n--- [{type_name}] ---')
        scored = scorer_fn(df)

        n_nonzero = (scored['anomaly_score'] > 0).sum()
        print(f'  Rows with score > 0 (above median): {n_nonzero:,} '
              f'({100*n_nonzero/len(scored):.1f}%)')

        if save_scores:
            scores_path = os.path.join(out_dir, f'{type_name}_scores.csv')
            scored.to_csv(scores_path, index=False)
            print(f'  Scores ({len(scored):,} rows) → {scores_path}')
        else:
            scores_path = '(not saved; use --save-scores to enable)'

        top, threshold = select_top_fraction(scored, 'anomaly_score', frac=top_frac)
        top.insert(0, 'rank', range(1, len(top) + 1))

        top_path = os.path.join(out_dir, f'{type_name}_top1pct.csv')
        top.to_csv(top_path, index=False)
        print(f'  Top {top_frac*100:.0f}% ({len(top):,} rows, '
              f'score >= {threshold:.3f}) → {top_path}')

        explain_path = os.path.join(out_dir, f'{type_name}_explain.csv')
        top.to_csv(explain_path, index=False)
        print(f'  Explain → {explain_path}')

        summaries.append({
            'anomaly_type':    type_name,
            'n_scored':        len(scored),
            'n_above_median':  int(n_nonzero),
            'top1pct_count':   len(top),
            'score_threshold': round(threshold, 6),
            'scores_csv':      scores_path,
            'top1pct_csv':     top_path,
            'explain_csv':     explain_path,
        })

        del scored, top

    summary_csv = os.path.join(out_dir, 'all_anomaly_summary.csv')
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    print(f'\nAll-type summary → {summary_csv}')

    elapsed = (datetime.now() - t_start).total_seconds()
    txt_path = os.path.join(out_dir, 'methodA_by_type_summary.txt')
    _write_summary(txt_path, features_csv, out_dir, top_frac,
                   len(df), elapsed, summaries, types_to_run)
    print(f'Summary text    → {txt_path}')
    print(f'\nDone in {elapsed:.1f}s.')


def _write_summary(path, features_csv, out_dir, top_frac,
                   n_rows, elapsed_s, summaries, types_run):
    feature_description = {
        'speed':    'sog_curr, abs_dsog  (one-sided: flags HIGH values only)',
        'turn':     'abs_dcog  (one-sided: flags LARGE turns only)',
        'timegap':  'log(delta_t_s + 1)  (one-sided: flags LONG gaps only)',
        'distance': 'dist_m, implied_kts  (one-sided: flags LARGE jumps only)',
    }
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Method A — By-Type Anomaly Scoring Summary\n')
        f.write('=' * 50 + '\n')
        f.write(f'Run timestamp    : {datetime.now().isoformat()}\n')
        f.write(f'Input features   : {features_csv}\n')
        f.write(f'Output directory : {out_dir}\n')
        f.write(f'Top fraction     : {top_frac} ({top_frac*100:.1f}%)\n')
        f.write(f'Types processed  : {", ".join(types_run)}\n')
        f.write(f'Total rows scored: {n_rows:,}\n')
        f.write(f'Elapsed seconds  : {elapsed_s:.1f}\n\n')
        f.write('Scoring method\n')
        f.write('--------------\n')
        f.write('One-sided robust z-score:\n')
        f.write('  score = max(0, x - median) / (1.4826 * MAD)\n')
        f.write('Only values ABOVE the fleet median score > 0.\n')
        f.write('This prevents short gaps from being flagged alongside long ones.\n\n')
        f.write('Results per anomaly type\n')
        f.write('------------------------\n')
        for s in summaries:
            tn = s['anomaly_type']
            f.write(f"\n[{tn}]\n")
            f.write(f"  Features       : {feature_description.get(tn, '—')}\n")
            f.write(f"  Rows scored    : {s['n_scored']:,}\n")
            f.write(f"  Above median   : {s['n_above_median']:,} "
                    f"({100*s['n_above_median']/s['n_scored']:.1f}%)\n")
            f.write(f"  Top 1% count   : {s['top1pct_count']:,}\n")
            f.write(f"  Score threshold: {s['score_threshold']:.6f}\n")
            f.write(f"  Top-1% CSV     : {s['top1pct_csv']}\n")
            f.write(f"  Explain CSV    : {s['explain_csv']}\n")
        f.write('\nGenerated files\n')
        f.write('---------------\n')
        for s in summaries:
            f.write(f"  {s['top1pct_csv']}\n")
            f.write(f"  {s['explain_csv']}\n")
        f.write(f'  {os.path.join(out_dir, "all_anomaly_summary.csv")}\n')
        f.write(f'  {os.path.join(out_dir, "reference_stats.json")}\n')
        f.write(f'  {path}\n')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    all_types = list(SCORERS.keys())
    parser = argparse.ArgumentParser(
        description='Method A by-type anomaly scoring (one-sided, explainable)'
    )
    parser.add_argument('--features', default='output/ais_features.csv')
    parser.add_argument('--out-dir', default='output/methodA_by_type')
    parser.add_argument('--top-frac', type=float, default=0.01)
    parser.add_argument('--types', nargs='+', default=all_types,
                        choices=all_types)
    parser.add_argument('--save-scores', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.features):
        print(f'ERROR: features file not found: {args.features}', file=sys.stderr)
        sys.exit(1)

    run_pipeline(args.features, args.out_dir, args.top_frac,
                 args.types, args.save_scores)


if __name__ == '__main__':
    main()
