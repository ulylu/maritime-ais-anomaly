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

## Weekly Progress (2026-02-20 to 2026-02-26)
- Added Method A (Isolation Forest) pipeline scripts:
  `prepare_features_for_iforest.py`, `train_iforest.py`,
  `make_methodA_explain_table.py`, and `plot_methodA_anomalies.py`
- Trained an Isolation Forest baseline (`n_estimators=200`, `random_state=42`) on `34,615,902` samples with `11` features
- Generated anomaly scores and exported the top `1%` anomaly candidates (`346,160` rows) for review
- Produced reproducible analysis artifacts, including the trained model, score tables, explanation CSVs, and trajectory plots
- Added output figures and a manifest for inspecting top-ranked anomalous vessel tracks

## Data Processing Pipeline
- `preprocess_ais.py`: Cleans and normalizes raw AIS records into a consistent, trajectory-ready format
- `extract_features.py`: Computes basic movement features from consecutive AIS points for anomaly detection
- `prepare_features_for_iforest.py`: Builds the feature matrix and metadata tables for Method A
- `train_iforest.py`: Trains Isolation Forest and exports anomaly scores / top candidates
- `make_methodA_explain_table.py`: Generates interpretable summary columns for flagged anomalies
- `plot_methodA_anomalies.py`: Creates trajectory visualizations for top-ranked anomalous tracks

*Note: Due to data size constraints, full raw and processed datasets are generated locally and are not included in this repository. Only small samples are provided for illustration.*

## Project Timeline
- Weeks 1-2: Data familiarization and feasibility assessment
- Weeks 3-4: Data preprocessing and feature engineering
- Weeks 5-8: Anomaly detection method implementation (Method A baseline completed)
- Weeks 9-10: Evaluation and comparison
