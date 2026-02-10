# maritime-ais-anomaly
# Machine Learning-Based Detection of Anomalous Vessel Behavior in AIS Data

This repository contains the implementation and experimental artifacts for a COMP 4905 Honours Project
focused on detecting anomalous vessel behavior from AIS trajectory data using machine learning methods.

## Current Status
- AIS data sourced from NOAA Marine Cadastre (AIS Vessel Tracks)
- Finalized dataset consisting of the last seven days of 2024 AIS records
- Full-scale preprocessing and feature extraction completed locally on the complete dataset
- Repository now includes the data preprocessing and feature extraction pipeline, along with representative samples

## Data Processing Pipeline
- `preprocess_ais.py`: Cleans and normalizes raw AIS records into a consistent, trajectory-ready format
- `extract_features.py`: Computes basic movement features from consecutive AIS points for anomaly detection

*Note: Due to data size constraints, full raw and processed datasets are generated locally and are not included in this repository. Only small samples are provided for illustration.*

## Project Timeline
- Weeks 1–2: Data familiarization and feasibility assessment
- Weeks 3–4: Data preprocessing and feature engineering (We are here!)
- Weeks 5–8: Anomaly detection method implementation
- Weeks 9–10: Evaluation and comparison
