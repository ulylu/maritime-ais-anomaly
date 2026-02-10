import argparse
import csv
import os
import math
from datetime import datetime
from typing import Optional

# ----------------------------
# Helpers
# ----------------------------

def parse_time_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def safe_float(x: str) -> Optional[float]:
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None

def safe_int_str(x: str) -> Optional[str]:
    """Return MMSI as string of digits, or None."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    # Some files store MMSI as float-like "367082130.0"
    if s.endswith(".0"):
        s = s[:-2]
    if not s.isdigit():
        return None
    return s

def in_range(lat: float, lon: float) -> bool:
    return (-90.0 <= lat <= 90.0) and (-180.0 <= lon <= 180.0)

def norm_angle_deg(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    # normalize into [0, 360)
    x = x % 360.0
    return x

# ----------------------------
# Core
# ----------------------------

def preprocess_stream(
    input_csv: str,
    output_csv: str,
    progress_every: int = 200_000,
    keep_extra_identity: bool = False,
    drop_missing_sog_cog_heading: bool = False,
) -> None:
    """
    Stream read input and write canonical points.
    """

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    required = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading"]
    optional = ["VesselType"]

    # If you want to carry identity columns through (not necessary for Method A input)
    identity_cols = ["VesselName", "IMO", "CallSign", "Status", "Length", "Width", "Draft", "Cargo", "TransceiverClass"]

    with open(input_csv, "r", newline="", encoding="utf-8", errors="ignore") as f_in:
        reader = csv.DictReader(f_in)
        header = reader.fieldnames or []

        # Check which optional columns exist
        has_vessel_type = "VesselType" in header
        keep_cols = required + (optional if has_vessel_type else [])
        if keep_extra_identity:
            keep_cols += [c for c in identity_cols if c in header]

        with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=keep_cols)
            writer.writeheader()

            n_in = 0
            n_out = 0
            n_bad = 0

            for row in reader:
                n_in += 1

                mmsi = safe_int_str(row.get("MMSI"))
                t = parse_time_iso((row.get("BaseDateTime") or "").strip())
                lat = safe_float(row.get("LAT"))
                lon = safe_float(row.get("LON"))

                sog = safe_float(row.get("SOG"))
                cog = safe_float(row.get("COG"))
                heading = safe_float(row.get("Heading"))

                # Basic required checks
                if mmsi is None or t is None or lat is None or lon is None:
                    n_bad += 1
                    continue
                if not in_range(lat, lon):
                    n_bad += 1
                    continue

                # Clean known "not available" conventions
                # Heading: 511 often means N/A
                if heading is not None and math.isclose(heading, 511.0):
                    heading = None

                # COG: 360 sometimes means N/A
                if cog is not None and math.isclose(cog, 360.0):
                    cog = None

                # Normalize angles
                cog = norm_angle_deg(cog) if cog is not None else None
                heading = norm_angle_deg(heading) if heading is not None else None

                # Optional stricter policy: require SOG/COG/Heading all present
                if drop_missing_sog_cog_heading:
                    if sog is None or cog is None or heading is None:
                        n_bad += 1
                        continue

                out_row = {
                    "MMSI": mmsi,
                    "BaseDateTime": t.isoformat(),
                    "LAT": f"{lat:.6f}",
                    "LON": f"{lon:.6f}",
                    "SOG": "" if sog is None else sog,
                    "COG": "" if cog is None else cog,
                    "Heading": "" if heading is None else heading,
                }

                if has_vessel_type:
                    vtype = row.get("VesselType")
                    out_row["VesselType"] = "" if vtype is None else str(vtype).strip()

                if keep_extra_identity:
                    for c in identity_cols:
                        if c in keep_cols:
                            out_row[c] = "" if row.get(c) is None else str(row.get(c)).strip()

                writer.writerow(out_row)
                n_out += 1

                if n_in % progress_every == 0:
                    print(f"[progress] read={n_in:,} wrote={n_out:,} dropped={n_bad:,}")

            print(f"[done] read={n_in:,} wrote={n_out:,} dropped={n_bad:,} output={output_csv}")


def warn_sorting():
    print(
        "\nNOTE about sorting:\n"
        "- Feature extraction is best when data is globally sorted by (MMSI, BaseDateTime).\n"
        "- This preprocessing script does NOT sort by default.\n"
        "- If your merged 7-day CSV is already sorted, you're good.\n"
        "- If not, ask me for an external-sort script that handles 5GB safely.\n"
    )


def main():
    ap = argparse.ArgumentParser(description="Stream preprocess a large AIS CSV into canonical points.")
    ap.add_argument("--input", required=True, help="Path to the merged 7-day AIS CSV (5GB+ ok).")
    ap.add_argument("--output", default="output/ais_points_clean.csv", help="Output canonical points CSV path.")
    ap.add_argument("--progress-every", type=int, default=200_000, help="Print progress every N rows.")
    ap.add_argument("--keep-identity", action="store_true", help="Carry extra identity columns through.")
    ap.add_argument("--strict", action="store_true", help="Drop rows missing SOG/COG/Heading after cleaning.")
    args = ap.parse_args()

    preprocess_stream(
        input_csv=args.input,
        output_csv=args.output,
        progress_every=args.progress_every,
        keep_extra_identity=args.keep_identity,
        drop_missing_sog_cog_heading=args.strict,
    )
    warn_sorting()


if __name__ == "__main__":
    main()
