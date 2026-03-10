# Instructions — maritime-ais-anomaly

> COMP 4905 Honours Project  
> Machine Learning-Based Detection of Anomalous Vessel Behavior in AIS Data

---

## 1. Project Overview

This project detects anomalous vessel behavior from AIS (Automatic Identification System) trajectory data. AIS is a maritime tracking system where vessels broadcast their position, speed, and course every few seconds to minutes.

**Dataset:** NOAA Marine Cadastre AIS Vessel Tracks — the last 7 days of 2024 (Dec 25–31). 34,615,902 consecutive-ping-pair records after preprocessing, covering thousands of vessels.

**Core question:** Given a week of positional broadcasts from thousands of vessels, which segments of which vessels exhibit behavior that is statistically unusual compared to the fleet?

**Two detection approaches are implemented:**

| Approach | Method | Granularity |
|---|---|---|
| Method A — Original | Isolation Forest (unsupervised ML) | One combined anomaly score per segment |
| Method A — By-Type | Robust z-score (statistical) | Separate score per anomaly category |

---

## 2. Repository Structure

```
maritime-ais-anomaly/
│
├── README.md                  ← Project overview and weekly progress
├── instructions.md            ← This file — full technical documentation
├── .gitignore                 ← Excludes CSVs, data/, and large files from git
│
├── data/
│   └── ais_2024_last7days.csv        ← Raw AIS data (5 GB, git-ignored)
│
├── outputs/
│   └── sample_mmsi_367776660.csv     ← Small sample for illustration
│
├── src/                              ← All source code (8 Python scripts)
│   ├── preprocess_ais.py
│   ├── extract_features.py
│   ├── prepare_features_for_iforest.py
│   ├── train_iforest.py
│   ├── make_methodA_explain_table.py
│   ├── plot_methodA_anomalies.py
│   ├── score_methodA_by_type.py
│   └── plot_methodA_by_type.py
│
└── output/                           ← All pipeline outputs (git-ignored)
    ├── ais_points_clean.csv
    ├── ais_features.csv
    ├── ais_features_methodA.csv
    ├── ais_features_methodA_X.csv
    ├── ais_features_methodA_meta.csv
    ├── ais_methodA_scores.csv
    ├── ais_methodA_top1pct.csv
    ├── ais_methodA_top1pct_explain.csv
    ├── methodA_summary.txt
    ├── models/
    │   └── iforest_methodA.joblib
    ├── methodA_by_type/
    │   ├── {speed,turn,timegap,distance}_scores.csv
    │   ├── {speed,turn,timegap,distance}_top1pct.csv
    │   ├── {speed,turn,timegap,distance}_explain.csv
    │   ├── all_anomaly_summary.csv
    │   ├── reference_stats.json
    │   └── methodA_by_type_summary.txt
    └── figs/
        ├── methodA_top_tracks/       ← 20 trajectory PNGs + manifest.csv
        └── methodA_by_type/
            ├── speed/                ← 10 PNGs + manifest.csv
            ├── turn/                 ← 10 PNGs + manifest.csv
            ├── timegap/              ← 10 PNGs + manifest.csv
            └── distance/             ← 10 PNGs + manifest.csv
```

---

## 3. Data Flow — Full Pipeline

```
 ┌───────────────────────────────────────────────────┐
 │ data/ais_2024_last7days.csv   (raw NOAA AIS data) │
 └──────────────────────┬────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │ preprocess_ais.py  │  Step 1: Clean & normalize
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │ extract_features.py│  Step 2: Pair-wise features
              └─────────┬──────────┘
                        │
          ┌─────────────┼─────────────────┐
          │                               │
 ┌────────▼─────────────┐     ┌───────────▼────────────┐
 │ Method A — Original  │     │ Method A — By-Type     │
 │                       │     │                        │
 │ prepare_features_     │     │ score_methodA_         │
 │   for_iforest.py      │     │   by_type.py           │
 │         │             │     │        │               │
 │ train_iforest.py      │     │ plot_methodA_          │
 │         │             │     │   by_type.py           │
 │ make_methodA_         │     └────────────────────────┘
 │   explain_table.py    │
 │         │             │
 │ plot_methodA_         │
 │   anomalies.py        │
 └───────────────────────┘
```

---

## 4. Script-by-Script Documentation

### 4.1 `preprocess_ais.py` — Data Cleaning

**Purpose:** Read the raw 5 GB AIS CSV and produce a clean, canonical version.

**Input:** `data/ais_2024_last7days.csv`  
**Output:** `output/ais_points_clean.csv`

**What it does:**
1. Streams the input row-by-row (memory-safe for large files)
2. Validates required fields: MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading
3. Drops rows with missing MMSI, timestamp, or lat/lon, or out-of-range coordinates
4. Cleans AIS "not available" conventions:
   - Heading = 511 → set to null
   - COG = 360 → set to null
5. Normalizes angles to [0, 360)
6. Optionally carries identity columns (VesselName, IMO, etc.)

**Command:**
```bash
python src/preprocess_ais.py \
    --input data/ais_2024_last7days.csv \
    --output output/ais_points_clean.csv
```

**Output columns:** `MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselType`

---

### 4.2 `extract_features.py` — Feature Engineering

**Purpose:** Compute movement features for every consecutive ping-pair per vessel.

**Input:** `output/ais_points_clean.csv`  
**Output:** `output/ais_features.csv`

**What it does:**
1. Maintains a dictionary of the last-seen point per MMSI
2. For each new point from the same vessel, computes:
   - `delta_t_s` — time gap in seconds
   - `dist_m` — haversine great-circle distance in meters
   - `dsog` — change in speed over ground (knots)
   - `dcog` — smallest signed course change in degrees (handles 0°/360° wrapping)
   - `dheading` — heading change (same wrapping logic)
3. Filters: skips pairs with Δt < 1 s or > 6 hours, or implied speed > 40 m/s

**Key algorithms:**
- **Haversine formula** for distance: `R × 2 × atan2(√a, √(1−a))` where `a = sin²(Δφ/2) + cos(φ₁)cos(φ₂)sin²(Δλ/2)`
- **Angle wrapping** for course/heading deltas: `d = (a2 − a1) mod 360; if d > 180: d −= 360` — returns values in [-180, 180]

**Command:**
```bash
python src/extract_features.py \
    --input output/ais_points_clean.csv \
    --output output/ais_features.csv
```

**Output columns:** `MMSI, t_prev, t_curr, delta_t_s, dist_m, sog_prev, sog_curr, dsog, cog_prev, cog_curr, dcog, heading_prev, heading_curr, dheading, vessel_type`

---

### 4.3 `prepare_features_for_iforest.py` — Feature Matrix Preparation

**Purpose:** Transform `ais_features.csv` into a standardized numeric matrix suitable for Isolation Forest.

**Input:** `output/ais_features.csv`  
**Output:**
- `output/ais_features_methodA.csv` — meta + scaled features combined
- `output/ais_features_methodA_X.csv` — numeric feature matrix only
- `output/ais_features_methodA_meta.csv` — meta columns only (MMSI, t_prev, t_curr)

**What it does:**
1. Separates meta columns (MMSI, t_prev, t_curr) from feature columns
2. Drops non-numeric columns (vessel_type)
3. Applies `log1p` transform to `dist_m` to compress the large range
4. Drops constant columns (columns with ≤ 1 unique value)
5. Fills missing values with column median, then any remaining with 0
6. Standardizes all features: `(x − mean) / std` (z-score normalization)

**Command:**
```bash
python src/prepare_features_for_iforest.py \
    --input output/ais_features.csv \
    --output-dir output
```

**Final feature count:** 11 features after dropping constants

---

### 4.4 `train_iforest.py` — Isolation Forest Training

**Purpose:** Train an Isolation Forest model and produce anomaly scores.

**Input:**
- `output/ais_features_methodA_X.csv`
- `output/ais_features_methodA_meta.csv`

**Output:**
- `output/models/iforest_methodA.joblib` — trained model
- `output/ais_methodA_scores.csv` — all 34.6M rows with anomaly scores
- `output/ais_methodA_top1pct.csv` — top 1% most anomalous (346,160 rows)
- `output/methodA_summary.txt` — run statistics

**How Isolation Forest works:**
1. Builds an ensemble of 200 random decision trees
2. Each tree randomly selects a feature and a split point, recursively partitioning the data
3. Anomalies are isolated in fewer splits (shorter path length) because they are rare and distinct
4. `anomaly_score = -score_samples(X)` — higher means more anomalous
5. The top 1% by score are exported as candidates

**Model parameters:**
| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | Enough trees for stable scores on 34M rows |
| `max_samples` | auto (256) | sklearn default; subsampling makes it scalable |
| `contamination` | auto | Let the model decide the threshold |
| `random_state` | 42 | Reproducibility |

**Score distribution (from actual run):**
| Stat | Value |
|---|---|
| min | 0.3269 |
| p50 (median) | 0.3941 |
| p99 | 0.6139 |
| max | 0.7561 |
| Training time | 43.6 seconds |

**Command:**
```bash
python src/train_iforest.py \
    --x output/ais_features_methodA_X.csv \
    --meta output/ais_features_methodA_meta.csv
```

---

### 4.5 `make_methodA_explain_table.py` — Explanation Table

**Purpose:** Join the top anomaly scores back to original features to create a human-readable analysis table.

**Input:**
- `output/ais_methodA_top1pct.csv`
- `output/ais_features.csv`

**Output:** `output/ais_methodA_top1pct_explain.csv`

**What it does:**
1. Joins top-1% anomaly rows to the original (unscaled) features by (MMSI, t_prev, t_curr)
2. Adds derived columns:
   - `abs_dsog`, `abs_dcog`, `abs_dheading` — absolute values of changes
   - `implied_mps`, `implied_kts` — speed implied by distance ÷ time
   - `anomaly_rank`, `anomaly_rank_pct` — ordinal rank and percentile

**Command:**
```bash
python src/make_methodA_explain_table.py \
    --top output/ais_methodA_top1pct.csv \
    --features output/ais_features.csv
```

---

### 4.6 `plot_methodA_anomalies.py` — Original Trajectory Plots

**Purpose:** Visualize the top Method A anomalies as lat/lon trajectory maps.

**Input:**
- `output/ais_methodA_top1pct.csv`
- `output/ais_points_clean.csv`

**Output:** `output/figs/methodA_top_tracks/` — 20 PNG plots + `manifest.csv`

**What it does:**
1. Selects the top 20 anomalous events from the top-1% list
2. Streams through `ais_points_clean.csv` to collect AIS points within a ±60 minute context window around each event
3. For each event, produces a 2-panel figure:
   - **Left panel:** Full context track (gray) with anomaly segment highlighted in red
   - **Right panel:** Anomaly segment (red) vs control segment before it (blue)

**Plot filename pattern:** `{rank:03d}_mmsi_{MMSI}_{start}_to_{end}.png`

**Command:**
```bash
python src/plot_methodA_anomalies.py \
    --top output/ais_methodA_top1pct.csv \
    --points output/ais_points_clean.csv \
    --n-plots 20
```

---

### 4.7 `score_methodA_by_type.py` — By-Type Anomaly Scoring

**Purpose:** Score each ping-pair for four specific anomaly categories using robust statistical methods.

**Input:** `output/ais_features.csv`  
**Output:** `output/methodA_by_type/` — per-type CSVs + summary

**Anomaly types and scoring:**

| Type | What it detects | Features used | Scoring |
|---|---|---|---|
| **speed** | Unusually high SOG or abrupt speed change | `sog_curr`, `abs_dsog` | max(z_sog, z_dsog) |
| **turn** | Unusually large course change | `abs_dcog` | z_dcog |
| **timegap** | Unusually long gap between pings | `log(delta_t_s + 1)` | z_log_dt |
| **distance** | Unusually large position jump | `dist_m`, `implied_kts` | max(z_dist, z_implied) |

**Scoring formula — one-sided robust z-score:**

```
score = max(0, x − median) / (1.4826 × MAD)
```

where MAD = median absolute deviation = `median(|xᵢ − median(x)|)`.

**Why one-sided:** Two-sided scoring (`|x − median|`) gave equal scores to 1-second gaps and 6-hour gaps, because both are equidistant from the median. One-sided scoring only flags values above the median, which is what we want: flag long gaps, not short ones.

**Why robust z-score instead of standard z-score:** Standard z-score uses mean and standard deviation, which are both pulled toward the very outliers we are trying to detect. The median and MAD are resistant to outliers, giving a more stable baseline.

**The 1.4826 constant:** For normally distributed data, `1.4826 × MAD` equals the standard deviation. This makes the score interpretable: a score of 3 means "3 standard deviations above the median" under a Gaussian assumption.

**Data quality filters in speed scoring:**
- SOG > 40 knots → treated as implausible (AIS sentinel value 102.2 means "not available"; values up to 101 kts appear in the data). These are zeroed before scoring.
- If either `sog_prev` or `sog_curr` exceeds 40 kts, `abs_dsog` is also zeroed (because the computed speed change is an artifact, not a real maneuver).

**Why `implied_kts` is in distance, not speed:** A large implied speed usually means the vessel's reported position jumped (GPS glitch or data-quality issue), not that the vessel actually traveled fast. Grouping it with distance prevents it from contaminating the speed category.

**Fleet-wide reference stats** are saved to `reference_stats.json` for use as reference lines in plots. Example values from the actual dataset:

| Metric | p50 | p95 | p99 | max |
|---|---|---|---|---|
| SOG (kts) | 0.0 | 10.8 | 17.4 | 101.0 |
| \|ΔSOG\| (kts) | 0.0 | 1.2 | 7.4 | 102.3 |
| \|Δcog\| (°) | 0.3 | 129.8 | 171.8 | 180.0 |
| Δt (sec) | 280 | 3,719 | 8,280 | 21,600 |
| Distance (m) | 2.7 | 1,905 | 9,179 | 706,894 |
| Implied speed (kts) | 0.02 | 10.6 | 16.9 | 77.5 |

**Command:**
```bash
python src/score_methodA_by_type.py \
    --features output/ais_features.csv \
    --out-dir output/methodA_by_type
```

Optional flags:
- `--save-scores` — save full per-row score CSVs (4–5 GB each); off by default
- `--types speed turn` — run only selected types
- `--top-frac 0.01` — fraction to mark as anomalous (default 1%)

---

### 4.8 `plot_methodA_by_type.py` — Type-Specific Visualizations

**Purpose:** Generate anomaly-type-matched plots that show exactly what triggered each flag.

**Input:**
- `output/methodA_by_type/{type}_top1pct.csv`
- `output/ais_points_clean.csv`
- `output/methodA_by_type/reference_stats.json`

**Output:** `output/figs/methodA_by_type/{speed,turn,timegap,distance}/` — 10 PNGs per type + `manifest.csv`

**Selection logic:**
- **5 "top" plots:** Deduplicated by MMSI — keep only the highest-scoring segment per vessel, then take the top 5. This prevents one vessel from occupying all slots.
- **5 "random" plots:** Sampled from the remaining top-1% rows (excluding MMSIs already used in top). Provides a representative view of flagged events.

**Plot designs per type:**

| Type | Left panel | Right panel |
|---|---|---|
| **speed** | SOG vs time; anomaly window in red; orange dashed line = fleet p99 SOG (17.4 kts); red annotation = peak SOG value | — (single panel) |
| **turn** | \|Δcog\| vs time; anomaly window in red; orange dashed line = fleet p99 \|Δcog\|; red annotation = peak turn degrees | — (single panel) |
| **timegap** | Ping timeline — each ping is a vertical tick; silent period is a red shaded gap with a double-headed arrow labeling duration | Bar chart — 3 bars comparing fleet median gap, fleet p99 gap, and this gap |
| **distance** | Segment distance vs time — each segment is a vertical bar; anomalous jump in red; orange line = fleet p99 distance | Lat/lon view — trajectory with red arrow showing the position jump |

**Plot filename pattern:** `{type}_{top|random}_{NNN}_mmsi_{MMSI}.png`

**Command:**
```bash
python src/plot_methodA_by_type.py \
    --by-type-dir output/methodA_by_type \
    --points output/ais_points_clean.csv \
    --out-dir output/figs/methodA_by_type \
    --n-top 5 \
    --n-random 5
```

---

## 5. Output File Reference

### 5.1 Intermediate Data Files

| File | Size (approx.) | Description |
|---|---|---|
| `ais_points_clean.csv` | ~4 GB | Cleaned AIS points with valid coordinates and timestamps |
| `ais_features.csv` | ~4 GB | One row per consecutive ping-pair; 14 feature columns |
| `ais_features_methodA.csv` | ~4 GB | Meta + z-score standardized features |
| `ais_features_methodA_X.csv` | ~3.5 GB | Numeric feature matrix only (11 columns) |
| `ais_features_methodA_meta.csv` | ~1 GB | MMSI, t_prev, t_curr for index alignment |

### 5.2 Method A — Original Outputs

| File | Description |
|---|---|
| `models/iforest_methodA.joblib` | Serialized Isolation Forest model (sklearn + joblib) |
| `ais_methodA_scores.csv` | All 34.6M rows with `anomaly_score` column, sorted descending |
| `ais_methodA_top1pct.csv` | Top 346,160 rows (1%) — anomaly candidates |
| `ais_methodA_top1pct_explain.csv` | Top 1% joined with original features + derived columns |
| `methodA_summary.txt` | Training params, score distribution, file paths |

### 5.3 Method A — By-Type Outputs

| File | Description |
|---|---|
| `{type}_scores.csv` | All 34.6M rows with per-type anomaly score (optional, large) |
| `{type}_top1pct.csv` | Top 1% for this type, with rank, score, percentile_rank, reason_text |
| `{type}_explain.csv` | Same as top1pct (alias for presentation convenience) |
| `all_anomaly_summary.csv` | One row per type: counts, thresholds, file paths |
| `reference_stats.json` | Fleet-wide p50/p95/p99/p999/max for all scored features |
| `methodA_by_type_summary.txt` | Human-readable run log |

### 5.4 Figures

| Directory | Contents |
|---|---|
| `figs/methodA_top_tracks/` | 20 trajectory PNGs (original Method A top-20) + `manifest.csv` |
| `figs/methodA_by_type/speed/` | 10 SOG-vs-time PNGs (5 top + 5 random) + `manifest.csv` |
| `figs/methodA_by_type/turn/` | 10 course-change-vs-time PNGs + `manifest.csv` |
| `figs/methodA_by_type/timegap/` | 10 ping-timeline + bar-comparison PNGs + `manifest.csv` |
| `figs/methodA_by_type/distance/` | 10 distance-vs-time + jump-map PNGs + `manifest.csv` |

### 5.5 Other Files

| File | Description |
|---|---|
| `outputs/sample_mmsi_367776660.csv` | Small sample (4,433 rows) of one vessel with identity columns, for illustration |

---

## 6. Dependencies

All scripts use only Python standard library plus these packages:

| Package | Used by | Purpose |
|---|---|---|
| `numpy` | All scoring/plotting scripts | Numeric computation |
| `pandas` | All scripts except preprocess/extract | DataFrame operations, CSV I/O |
| `scikit-learn` | `train_iforest.py` | IsolationForest implementation |
| `matplotlib` | All plot scripts | Figure generation |
| `joblib` | `train_iforest.py` | Model serialization (optional; falls back to pickle) |

Install:
```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

---

## 7. How to Run — Full Deployment

### 7.1 Prerequisites
- Python 3.11+
- ~16 GB RAM recommended (34M-row DataFrames in pandas)
- ~30 GB disk space for all intermediate/output CSVs
- Raw data file `data/ais_2024_last7days.csv` in place

### 7.2 Run the shared preprocessing (both methods need this)

```bash
# Step 1: Clean raw AIS data (~15 min)
python src/preprocess_ais.py \
    --input data/ais_2024_last7days.csv \
    --output output/ais_points_clean.csv

# Step 2: Extract pair-wise features (~20 min)
python src/extract_features.py \
    --input output/ais_points_clean.csv \
    --output output/ais_features.csv
```

### 7.3 Run Method A — Original (Isolation Forest)

```bash
# Step 3: Prepare feature matrix (~5 min)
python src/prepare_features_for_iforest.py \
    --input output/ais_features.csv \
    --output-dir output

# Step 4: Train Isolation Forest (~1 min)
python src/train_iforest.py

# Step 5: Build explanation table (~5 min)
python src/make_methodA_explain_table.py

# Step 6: Plot top anomalies (~20 min — streams 4 GB CSV)
python src/plot_methodA_anomalies.py \
    --n-plots 20
```

### 7.4 Run Method A — By-Type (Robust Z-Score)

```bash
# Step 3b: Score by type (~3 min)
python src/score_methodA_by_type.py \
    --features output/ais_features.csv \
    --out-dir output/methodA_by_type

# Step 4b: Generate type-specific plots (~20 min — streams 4 GB CSV)
python src/plot_methodA_by_type.py \
    --by-type-dir output/methodA_by_type \
    --points output/ais_points_clean.csv \
    --out-dir output/figs/methodA_by_type \
    --n-top 5 \
    --n-random 5
```

### 7.5 Total runtime estimate
| Step | Time |
|---|---|
| Preprocessing | ~15 min |
| Feature extraction | ~20 min |
| Method A Original (steps 3–6) | ~30 min |
| Method A By-Type (steps 3b–4b) | ~25 min |
| **Total** | **~90 min** |

---

## 8. Method Comparison

### 8.1 Method A — Original (Isolation Forest)

**Approach:** Unsupervised machine learning. An ensemble of 200 random trees learns to isolate unusual data points in an 11-dimensional feature space.

**Strengths:**
- Captures multi-feature interactions (e.g., a vessel that is simultaneously fast, turning sharply, and jumping position)
- No threshold tuning needed — the model learns what is "normal" from the data
- Well-established algorithm (Liu et al., 2008)

**Limitations:**
- Produces a single opaque score — hard to explain *why* a segment is flagged
- Different anomaly types compete for the same score; a speed anomaly and a time-gap anomaly may have similar scores
- Sensitive to feature scaling and log transforms

### 8.2 Method A — By-Type (Robust Z-Score)

**Approach:** Classical statistics. Each anomaly type is scored independently using a robust z-score that measures how many "standard deviations" above the fleet median a value is.

**Strengths:**
- Fully explainable: each score directly corresponds to one observable quantity ("SOG was 35 kts, which is 8.2 robust z-scores above the fleet median")
- Separate rankings per type — a speed anomaly doesn't compete with a time-gap anomaly
- Fast (no model training; pure arithmetic on 34M rows in ~3 min)
- Robust to outliers (median/MAD vs mean/std)

**Limitations:**
- Only flags single-feature extremes; cannot detect multi-feature interactions
- Assumes each feature can be scored independently (no correlation modeling)
- Threshold (top 1%) is arbitrary

### 8.3 Design Decisions

| Decision | Rationale |
|---|---|
| One-sided scoring (not two-sided) | Two-sided `\|x − median\|` gave 1-second gaps the same score as 6-hour gaps. Only high-side extremes are meaningful anomalies. |
| SOG > 40 kts filtered | AIS uses 102.2 kts as "not available" sentinel. Values 40–102 kts are physically implausible for most vessels and polluted speed rankings. |
| `implied_kts` in distance, not speed | High implied speed from position jumps is a location anomaly (GPS error), not a speed anomaly. |
| MMSI deduplication in plots | Without dedup, MMSI 366688000 occupied all 5 distance top slots (same vessel, multiple jumps). Each plot now shows a different vessel. |
| `log1p(delta_t_s)` for timegap scoring | Raw gap ranges from 1 s to 21,600 s (6 h). Log scale compresses this so the z-score is more discriminating in the middle range. |
| median/MAD instead of mean/std | The top 0.1% of values inflate mean and std, making standard z-scores unreliable for extreme outlier detection. |

---

## 9. Key Data Characteristics

From `reference_stats.json` (computed on all 34,615,902 ping-pairs):

- **50% of vessels are nearly stationary** (median SOG = 0.0 kts) — many are anchored or moored
- **Typical inter-ping gap is ~5 minutes** (median Δt = 280 s); the 99th percentile is 2.3 hours
- **Typical displacement is ~3 meters** (median dist = 2.7 m); the 99th percentile is 9.2 km
- **Course changes are mostly tiny** (median |Δcog| = 0.3°); but the p95 is 130° — many vessels do make large turns
- **Maximum position jump is 707 km** — likely a data-quality issue or a vessel that changed AIS transponders

---

## 10. Git Configuration

The `.gitignore` excludes:
- All CSV, Parquet, Feather, and HDF5 files (data too large for git)
- `data/`, `raw/`, `processed/` directories
- Python bytecode, virtual environments, IDE settings
- Log files and temp files

**What IS tracked in git:**
- All 8 Python scripts in `src/`
- `README.md` and `instructions.md`
- `.gitignore`
- Small sample file `outputs/sample_mmsi_367776660.csv` (if not caught by `*.csv` rule — verify)

---

## 11. Known Issues and Limitations

1. **Speed top is dominated by SOG = 40 kts vessels.** MMSI 366778450 (likely a fast patrol boat) hits exactly 40 kts multiple times daily. This is technically anomalous but not necessarily "interesting." Per-vessel-type stratification would help distinguish expected fast vessels from genuinely unusual ones.

2. **Distance top-1% is monopolized by one MMSI (366688000)** with a 707 km jump. Plot-level dedup solves the visualization issue, but the `distance_top1pct.csv` still has this vessel dominating the first ~50 rows.

3. **`methodA_by_type_summary.txt` only records speed** from a partial run. A full re-run with all 4 types would populate the complete summary.

4. **No trajectory-shape anomaly type** (e.g., loitering, zigzag patterns). This would require sequence-level features beyond pair-wise comparisons.

5. **No per-vessel normalization.** All scoring is relative to the global fleet. A tugboat doing 12 kts is normal for a tugboat but would not be flagged because 12 kts is below the fleet p99 of 17.4 kts. Per-vessel baselines would make scoring more nuanced.
