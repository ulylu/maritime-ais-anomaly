# maritime-ais-anomaly
# Machine Learning-Based Detection of Anomalous Vessel Behavior in AIS Data

This repository contains the implementation and experimental artifacts for a COMP 4905 Honours Project
focused on detecting anomalous vessel behavior from AIS trajectory data using machine learning methods.

## Current Status
- AIS data sourced from NOAA Marine Cadastre (AIS Vessel Tracks)
- Finalized dataset consisting of the last seven days of 2024 AIS records
- Full-scale preprocessing and feature extraction completed locally on the complete dataset
- Repository now includes the preprocessing, feature engineering, and Method A anomaly detection pipeline
- Isolation Forest (Method A) has been trained and scored on the prepared feature matrix
- Top 1% anomaly candidates, explanation tables, and trajectory visualizations have been generated
- **New:** Method A by-type pipeline added — produces separate anomaly scores and visualizations for
  speed, turn, time-gap, and distance anomaly types

---

## Method A — Original Overall Pipeline

The original Method A pipeline trains a single Isolation Forest on all movement features combined
and produces one overall anomaly score per AIS segment.

**Run order:**
```
preprocess_ais.py → extract_features.py → prepare_features_for_iforest.py
→ train_iforest.py → make_methodA_explain_table.py → plot_methodA_anomalies.py
```

| Script | What it does |
|---|---|
| `preprocess_ais.py` | Cleans and normalizes raw AIS records |
| `extract_features.py` | Computes pairwise movement features per consecutive ping pair |
| `prepare_features_for_iforest.py` | Builds and standardizes the feature matrix for Isolation Forest |
| `train_iforest.py` | Trains Isolation Forest; exports anomaly scores and top-1% candidates |
| `make_methodA_explain_table.py` | Joins top candidates back to raw features; adds readable derived columns |
| `plot_methodA_anomalies.py` | Creates lat/lon trajectory plots for the top-ranked anomalous events |

**Key outputs** (under `output/`):
- `ais_methodA_scores.csv` — all rows with a single combined anomaly score
- `ais_methodA_top1pct.csv` — top 1% anomaly candidates (346,160 rows from 34.6 M)
- `ais_methodA_top1pct_explain.csv` — explain table with derived readable columns
- `figs/methodA_top_tracks/` — trajectory figures and manifest

---

## Method A — By-Type Pipeline (Improved)

### Why by-type scoring?

The original single-score approach is good for ranking overall anomalousness, but it mixes
together very different kinds of anomalies — a vessel that stops transmitting for hours looks
the same in the score table as a vessel that suddenly jumps position.
Separate per-type scores make it easy to say in a meeting:
*"This vessel is flagged for a speed anomaly because its implied speed was 45 knots,
which is in the 99.8th percentile for this dataset."*

By-type scoring also makes visualizations more informative: each plot shows exactly the
feature responsible for the flag, not a generic lat/lon map.

### Anomaly types

| Type | What is flagged | Key features | Scoring method |
|---|---|---|---|
| **speed** | Unusually high speed or abrupt speed change | `sog_curr`, `abs_dsog` | Max of robust z-scores |
| **turn** | Unusually large course change | `abs_dcog` | Robust z-score |
| **timegap** | Unusually long gap between consecutive pings | `log(delta_t_s + 1)` | Robust z-score |
| **distance** | Unusually large position jump | `dist_m`, `implied_kts` | Max of robust z-scores |

Robust z-score formula (one-sided): `score = max(0, x − median) / (1.4826 × MAD)`  
MAD = median absolute deviation. One-sided scoring ensures only values *above* the fleet
median are flagged — this prevents short time gaps from scoring equally to long ones.
The median/MAD approach is preferred over mean/std because it is not inflated by
the very outliers we are trying to detect.

### Run order

```
preprocess_ais.py → extract_features.py → score_methodA_by_type.py → plot_methodA_by_type.py
```

The first two steps are shared with the original pipeline.

**Scoring:**
```
python src/score_methodA_by_type.py \
    --features output/ais_features.csv \
    --out-dir  output/methodA_by_type \
    --top-frac 0.01
```

**Plotting:**
```
python src/plot_methodA_by_type.py \
    --by-type-dir output/methodA_by_type \
    --points      output/ais_points_clean.csv \
    --out-dir     output/figs/methodA_by_type \
    --n-top    5 \
    --n-random 5
```

### By-type outputs

**CSVs** under `output/methodA_by_type/`:

| File | Contents |
|---|---|
| `speed_scores.csv` | All rows with speed anomaly score + raw feature values |
| `turn_scores.csv` | All rows with turn anomaly score |
| `timegap_scores.csv` | All rows with time-gap anomaly score |
| `distance_scores.csv` | All rows with distance anomaly score |
| `speed_top1pct.csv` | Top 1% speed anomalies (ranked, with reason_text) |
| `turn_top1pct.csv` | Top 1% turn anomalies |
| `timegap_top1pct.csv` | Top 1% time-gap anomalies |
| `distance_top1pct.csv` | Top 1% distance anomalies |
| `{type}_explain.csv` | Alias of top-1% with rank column; ready for presentation |
| `all_anomaly_summary.csv` | One row per type: counts, threshold, file paths |
| `methodA_by_type_summary.txt` | Human-readable run summary |

Each score CSV includes: `MMSI`, `t_prev`, `t_curr`, `anomaly_type`, `anomaly_score`,
`percentile_rank`, relevant raw feature values, and `reason_text`.

**Figures** under `output/figs/methodA_by_type/{type}/`:

Each type gets its own folder with plots suited to that anomaly:

| Type | Plot content |
|---|---|
| `speed/` | SOG vs time; anomaly window highlighted in red |
| `turn/` | \|Δcog\| vs time; anomaly window highlighted in red |
| `timegap/` | Inter-ping gap (s) vs time; anomalous gap in red |
| `distance/` | Segment distance vs time + lat/lon jump view with arrow |

Filenames follow the pattern `{type}_{top|random}_{NNN}_mmsi_{MMSI}.png`.
Each type folder also contains a `manifest.csv`.

For each type, plots include both the highest-scoring events (`top`) and a random
sample from within the top 1% (`random`). This avoids showing only extreme edge
cases and gives a more representative view of the flagged population.

---

## Weekly Progress

### 2026-02-20 to 2026-02-26
- Added Method A (Isolation Forest) pipeline scripts:
  `prepare_features_for_iforest.py`, `train_iforest.py`,
  `make_methodA_explain_table.py`, and `plot_methodA_anomalies.py`
- Trained an Isolation Forest baseline (`n_estimators=200`, `random_state=42`) on `34,615,902` samples with `11` features
- Generated anomaly scores and exported the top `1%` anomaly candidates (`346,160` rows) for review
- Produced reproducible analysis artifacts, including the trained model, score tables, explanation CSVs, and trajectory plots
- Added output figures and a manifest for inspecting top-ranked anomalous vessel tracks

### 2026-03-07
- Added `score_methodA_by_type.py`: by-type anomaly scoring pipeline using robust z-scores
- Added `plot_methodA_by_type.py`: type-specific visualizations (speed/turn/timegap/distance)
- By-type pipeline produces separate top-1% lists and explanation tables per anomaly category
- Plots are matched to the anomaly type (e.g. SOG vs time for speed; gap vs time for timegap)

### 2026-03-13
- Method B (LSTM) deliverables finalized and organized under `MethodB-LSTM/`
- Added Method B source scripts, model artifacts, summary output, and figures
- Repository timeline updated to include Method B completion milestone

---

## Project Timeline
- Weeks 1-2: Data familiarization and feasibility assessment
- Weeks 3-4: Data preprocessing and feature engineering
- Weeks 5-8: Anomaly detection method implementation (Method A baseline completed)
- Weeks 9-10: Method A by-type pipeline; evaluation and comparison

---

*Note: Due to data size constraints, full raw and processed datasets are generated locally and are not included in this repository. Only small samples are provided for illustration.*
