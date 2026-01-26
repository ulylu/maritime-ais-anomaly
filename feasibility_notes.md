## Data Feasibility Assessment

The AIS dataset under inspection is stored as a GeoPackage (.gpkg), which uses SQLite as its underlying format.

- File size: ~1.2 GB
- File header confirms valid SQLite format ("SQLite format 3")
- SQLite reports internal corruption ("database disk image is malformed")

Next steps:
- Attempt database recovery using SQLite recovery tools
- If recovery fails, re-download and re-validate the dataset
- Once accessible, inspect table structure, fields, and temporal resolution
