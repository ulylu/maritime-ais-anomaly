# Machine Learning-Based Detection of Anomalous Vessel Behavior in AIS Data

This repository contains the code, selected outputs, and supporting files for a COMP 4905 Honours Project on anomaly detection in AIS vessel data.

## Project Overview

Automatic Identification System (AIS) data records vessel movement information such as position, speed, course, and heading. Because AIS data is generated continuously and at large scale, manual inspection is not practical for identifying unusual vessel behavior. This project studies how machine learning methods can be used to support anomaly detection in real AIS data. :contentReference[oaicite:2]{index=2}

The project uses AIS data from NOAA Marine Cadastre and focuses on the last seven days of 2024 as the main study period. Two different anomaly detection methods are implemented and compared: :contentReference[oaicite:3]{index=3}

- **Method A:** an Isolation Forest based method that performs anomaly scoring on transition-level vessel movement features, followed by script-based result organization and visualization
- **Method B:** an LSTM-based sequence method that detects anomalous behavior from short time-ordered vessel sequences through prediction error

These two methods analyze suspicious vessel behavior from different perspectives. Method A is more suitable for local transition-level anomalies, while Method B is more suitable for short continuous behavior patterns. :contentReference[oaicite:4]{index=4}

## Data Source

The AIS data used in this project comes from:

- **NOAA Marine Cadastre – AIS Data for 2024**

This project focuses on records from **December 25 to December 31, 2024**. The full raw AIS dataset is large, so this submission package includes code, selected outputs, and project files rather than the complete original raw data. :contentReference[oaicite:5]{index=5}

## Repository Structure

```text
maritime-ais-anomaly/
├── README.md
├── MethodA/
│   ├── src/
│   ├── output/
│   └── models/
└── MethodB-LSTM/
    ├── src/
    ├── output/
    └── model/
