# maritime-ais-anomaly
# Machine Learning-Based Detection of Anomalous Vessel Behavior in AIS Data

This repository contains the implementation and experimental artifacts for a COMP 4905 Honours Project
focused on detecting anomalous vessel behavior from AIS trajectory data using machine learning methods.

## Project Overview

Automatic Identification System (AIS) data records vessel movement information such as position, speed,
course, and heading. Because large volumes of AIS data are generated continuously, manual inspection is
not practical for finding unusual vessel behavior. This project studies how machine learning methods can
be used to detect suspicious movement patterns from real AIS records.

The project uses AIS data from NOAA Marine Cadastre and focuses on the last seven days of 2024.
Two different anomaly detection methods are implemented and compared:

- **Method A:** an Isolation Forest based pipeline for anomaly scoring and organized visualization
- **Method B:** an LSTM-based sequence method for anomaly detection on time-ordered vessel behavior

The goal of this project is not only to detect anomalous vessel behavior, but also to compare how
different machine learning approaches describe suspicious movement from different angles.

---

## Current Progress

The current project stage includes:

- AIS data collection completed
- Data preprocessing completed
- Feature engineering completed
- **Method A completed**
- **Method B completed**
- Result visualization completed
- Method comparison completed
- **Final report draft in progress**

---

## Data Source

The AIS data used in this project comes from:

- **NOAA Marine Cadastre – AIS Vessel Tracks**

This project uses records from the **last seven days of 2024** as the main experimental dataset.

---

## Repository Structure

```text
maritime-ais-anomaly/
├── README.md
├── MethodA-iForest/
│   ├── src/
│   ├── output/
│   └── models/
├── MethodB-LSTM/
│   ├── src/
│   ├── output/
│   └── models/
└── report/
