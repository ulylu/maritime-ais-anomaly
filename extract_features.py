import argparse
import csv
import math
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

# ----------------------------
# Helpers
# ----------------------------

def parse_time_iso(s: str) -> Optional[datetime]:
    """Parse ISO time like 2024-12-25T00:00:05"""
    if not s:
        return None
    try:
        # Python 3.11+ supports fromisoformat with 'T'
        return datetime.fromisoformat(s)
    except Exception:
        return None

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def wrap_angle_delta(a2: float, a1: float) -> float:
    """
    Smallest signed difference between headings/courses in degrees.
    Returns value in [-180, 180].
    """
    d = (a2 - a1) % 360.0
    if d > 180.0:
        d -= 360.0
    return d

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

# ----------------------------
# Core extraction
# ----------------------------

PrevState = Tuple[datetime, float, float, Optional[float], Optional[float], Optional[float], Optional[str]]

def extract_features(
    input_csv: str,
    output_csv: str,
    chunksize_rows: int = 200_000,
    min_dt_s: float = 1.0,
    max_dt_s: float = 6 * 3600.0,
    max_speed_mps: float = 40.0,
    require_monotonic_time: bool = True,
) -> None:
    """
    Streaming extraction: keep last seen point for each MMSI.
    Write one feature-row per consecutive pair within dt bounds.
    """

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Keep previous record per MMSI
    prev: Dict[str, PrevState] = {}

    # Prepare output
    out_fields = [
        "MMSI",
        "t_prev",
        "t_curr",
        "delta_t_s",
        "dist_m",
        "sog_prev",
        "sog_curr",
        "dsog",
        "cog_prev",
        "cog_curr",
        "dcog",
        "heading_prev",
        "heading_curr",
        "dheading",
        "vessel_type",
    ]

    n_in = 0
    n_out = 0
    n_skipped = 0

    with open(input_csv, "r", newline="", encoding="utf-8", errors="ignore") as f_in, \
         open(output_csv, "w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            n_in += 1

            mmsi = (row.get("MMSI") or "").strip()
            t_str = (row.get("BaseDateTime") or "").strip()
            lat = safe_float(row.get("LAT"))
            lon = safe_float(row.get("LON"))
            sog = safe_float(row.get("SOG"))       # knots in many AIS datasets; we treat as "as-is"
            cog = safe_float(row.get("COG"))       # degrees
            heading = safe_float(row.get("Heading"))
            vtype = row.get("VesselType")
            vtype = str(vtype).strip() if vtype is not None else None

            if not mmsi or lat is None or lon is None:
                n_skipped += 1
                continue

            t = parse_time_iso(t_str)
            if t is None:
                n_skipped += 1
                continue

            # If we have previous point for this MMSI, compute pair-features
            if mmsi in prev:
                t_prev, lat_prev, lon_prev, sog_prev, cog_prev, heading_prev, vtype_prev = prev[mmsi]

                # Optionally enforce monotonic time
                if require_monotonic_time and t <= t_prev:
                    # Update prev to the newer (or keep?) — keep the latest time encountered
                    prev[mmsi] = (t, lat, lon, sog, cog, heading, vtype or vtype_prev)
                    n_skipped += 1
                    continue

                dt = (t - t_prev).total_seconds()

                # Filter unreasonable dt
                if dt < min_dt_s or dt > max_dt_s:
                    prev[mmsi] = (t, lat, lon, sog, cog, heading, vtype or vtype_prev)
                    n_skipped += 1
                    continue

                dist_m = haversine_m(lat_prev, lon_prev, lat, lon)

                # crude physical plausibility check using implied speed (m/s)
                implied_mps = dist_m / dt if dt > 0 else float("inf")
                if implied_mps > max_speed_mps:
                    prev[mmsi] = (t, lat, lon, sog, cog, heading, vtype or vtype_prev)
                    n_skipped += 1
                    continue

                dsog = None
                if sog_prev is not None and sog is not None:
                    dsog = sog - sog_prev

                dcog = None
                if cog_prev is not None and cog is not None:
                    dcog = wrap_angle_delta(cog, cog_prev)

                dheading = None
                if heading_prev is not None and heading is not None:
                    # AIS sometimes uses 511 for "not available" - treat as missing
                    if heading_prev == 511.0 or heading == 511.0:
                        dheading = None
                    else:
                        dheading = wrap_angle_delta(heading, heading_prev)

                writer.writerow({
                    "MMSI": mmsi,
                    "t_prev": t_prev.isoformat(),
                    "t_curr": t.isoformat(),
                    "delta_t_s": f"{dt:.3f}",
                    "dist_m": f"{dist_m:.3f}",
                    "sog_prev": "" if sog_prev is None else sog_prev,
                    "sog_curr": "" if sog is None else sog,
                    "dsog": "" if dsog is None else dsog,
                    "cog_prev": "" if cog_prev is None else cog_prev,
                    "cog_curr": "" if cog is None else cog,
                    "dcog": "" if dcog is None else dcog,
                    "heading_prev": "" if heading_prev is None else heading_prev,
                    "heading_curr": "" if heading is None else heading,
                    "dheading": "" if dheading is None else dheading,
                    "vessel_type": vtype or vtype_prev or "",
                })
                n_out += 1

                # Update prev to current point
                prev[mmsi] = (t, lat, lon, sog, cog, heading, vtype or vtype_prev)

            else:
                prev[mmsi] = (t, lat, lon, sog, cog, heading, vtype)

            # lightweight progress print
            if n_in % chunksize_rows == 0:
                print(f"[progress] read={n_in:,} wrote={n_out:,} skipped={n_skipped:,} unique_mmsi={len(prev):,}")

    print(f"[done] read={n_in:,} wrote={n_out:,} skipped={n_skipped:,} output={output_csv}")


def main():
    ap = argparse.ArgumentParser(description="Extract minimal AIS trajectory features from a large CSV.")
    ap.add_argument("--input", required=True, help="Path to ais_2024_last7days.csv (5GB+ ok).")
    ap.add_argument("--output", default="output/ais_features.csv", help="Output features csv path.")
    ap.add_argument("--progress-every", type=int, default=200_000, help="Print progress every N rows.")
    ap.add_argument("--min-dt", type=float, default=1.0, help="Minimum delta-t in seconds.")
    ap.add_argument("--max-dt", type=float, default=21600.0, help="Maximum delta-t in seconds (default 6h).")
    ap.add_argument("--max-speed-mps", type=float, default=40.0, help="Max implied speed (m/s) to keep.")
    ap.add_argument("--no-monotonic", action="store_true", help="Disable monotonic time enforcement per MMSI.")
    args = ap.parse_args()

    extract_features(
        input_csv=args.input,
        output_csv=args.output,
        chunksize_rows=args.progress_every,
        min_dt_s=args.min_dt,
        max_dt_s=args.max_dt,
        max_speed_mps=args.max_speed_mps,
        require_monotonic_time=not args.no_monotonic,
    )


if __name__ == "__main__":
    main()
