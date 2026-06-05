#!/usr/bin/env python3
"""
Phase 3.5 - Preprocess: GOES MAG Interim Parquet to Preprocessed Parquet
-------------------------------------------------------------------------
This is the fifth Phase 3 preprocessing script. It reads monthly
interim GOES MAG Parquet files produced by Phase 2.2
(src/clean/interim/goes_mag.py), enforces a gapless 1-minute time
index, normalizes orbital position column naming variants, harmonizes
generation-specific quality column names, applies physical bounds
validation to orbital position values, appends a generation label
and a composite validity flag, standardizes column types, and writes
preprocessed monthly Parquet files to data/preprocessed/goes_mag/.
This script has no dependency on any other Phase 3 script and can
run in parallel with Phase 3.1 through 3.4.

Purpose
-------
GOES MAG interim data carries structural differences between the
legacy goes15 format and the GOES-R format (goes16/17/18) that must
be resolved before fusion. This script resolves three categories of
cross-generation schema inconsistency:

1. Orbit column naming: GOES-R files carry orbit position under
   "orbit_{lon/lat/radius}" names. Legacy files use the longer
   "orbit_llr_geo_{lon/lat/radius}" variant. pick_orbit_columns()
   promotes the llr_geo names to the canonical short names when the
   canonical name is absent, ensuring a consistent orbit column set
   regardless of generation.

2. Quality column naming: GOES-R files use "b_quality" or "dqf";
   legacy files use "quality". harmonize_quality() maps whichever
   is present to a single "quality_summary" column typed as Int64,
   or assigns pd.NA if neither is found.

3. Orbital position bounds: sanitize_orbit() applies conservative
   physical range checks and nulls out-of-range values: radius must
   fall within the GEO sanity band [4.0e7, 4.3e7] meters; longitude
   within [-180, 180] degrees; latitude within [-90, 90] degrees.
   Values outside these bounds are set to NaN rather than raising.

Like Phase 3.4 (goes_xrs.py), this script applies minute_reindex()
to construct a gapless 1-minute pd.date_range across each file and
forward/back fills sat_id over the introduced NaN rows. The mag_valid
boolean column is True for any row where at least one of the six core
GSM/VDH vector components or b_total_nT is non-null. Provenance
columns source_file and source_type are dropped before output. No
--force flag is present; existing output files are unconditionally
overwritten.

WORK FLOW
---------
Step 1: Parse CLI arguments - input root, output root, and optional
        satellite filter list.
Step 2: Recursively discover all Parquet files under --input-root,
        optionally filtered by satellite name inferred from the
        two-levels-up directory name.
Step 3: For each file, apply ensure_sat_id(): assign sat_id from the
        directory name if the column is absent.
Step 4: Drop interim-only provenance columns (source_file,
        source_type) and parse the timestamp column.
Step 5: Apply minute_reindex(): construct a complete 1-minute
        pd.date_range and reindex, introducing NaN rows for missing
        minutes. Forward/back fill sat_id over introduced NaN rows.
Step 6: Apply pick_orbit_columns(): promote orbit_llr_geo_* column
        names to the canonical orbit_* names if the canonical name
        is absent.
Step 7: Apply harmonize_quality(): map b_quality (GOES-R) or quality
        (legacy) to a unified quality_summary column.
Step 8: Assign the mag_source label ("goesr" or "legacy") via
        build_mag_source() based on presence of b_quality or dqf.
Step 9: Retain CORE_VECTOR_COLS and present OPTIONAL_COLS; fill any
        absent CORE_VECTOR_COLS with np.nan.
Step 10: Apply sanitize_orbit(): null orbit radius, longitude, and
         latitude values that fall outside conservative physical bounds.
Step 11: Compute mag_valid boolean: True if any core vector component
         or b_total_nT is non-null for the row.
Step 12: Enforce ordered output column list, drop entirely-null non-
         protected columns, apply standardize_dtypes() (field vectors
         and orbit -> float32, quality/count columns -> Int64,
         mag_valid -> boolean), and write to
         data/preprocessed/goes_mag/{sat_id}/{YYYY}/{sat_id}_{YYYYMM}.parquet.

INPUT DATA
----------
- data/interim/goes_mag/{sat_id}/{YYYY}/*.parquet
  Monthly interim GOES MAG Parquet files produced by Phase 2.2.
  Expected CORE_VECTOR_COLS: b_gsm_x, b_gsm_y, b_gsm_z,
  b_vdh_x, b_vdh_y, b_vdh_z.
  Optional: b_total_nT, num_points, b_quality or quality, method,
  dqf, orbit position columns in either naming variant.
- CLI optional: --input-root  (default: data/interim/goes_mag)
               --output-root (default: data/preprocessed/goes_mag)
               --satellites  (default: all found; e.g. goes15 goes16
                              goes17 goes18)

OUTPUT DATA
-----------
- data/preprocessed/goes_mag/{sat_id}/{YYYY}/{sat_id}_{YYYYMM}.parquet
  One Parquet file per satellite per month with a gapless 1-minute
  time index. Output column order: timestamp, sat_id, mag_source,
  b_total_nT (float32), b_gsm_x/y/z (float32), b_vdh_x/y/z (float32),
  quality_summary (Int64), num_points (Int64), method (Int64),
  dqf (Int64), orbit_longitude_deg (float32),
  orbit_latitude_deg (float32), orbit_radius_m (float32),
  mag_valid (boolean). Entirely-null non-protected columns are dropped.

NEXT PIPELINE PHASE
-------------------
- Phase 3.5 depends only on Phase 2.2 (src/clean/interim/goes_mag.py)
  completing first. It has no dependency on Phase 3.1 through 3.4
  and can run in parallel with other Phase 3.x scripts.
- After this script completes, GOES MAG preprocessed data is ready
  for Phase 4 (src/fusion/data_fusion.py).
- All Phase 3.x preprocessing scripts must complete before Phase 4
  can begin, since the fusion stage requires all sources to be present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_ROOT = Path("data/interim/goes_mag")
DEFAULT_OUTPUT_ROOT = Path("data/preprocessed/goes_mag")


CORE_VECTOR_COLS = [
    "b_gsm_x", "b_gsm_y", "b_gsm_z",
    "b_vdh_x", "b_vdh_y", "b_vdh_z",
]

OPTIONAL_COLS = [
    "b_total_nT",
    "num_points",
    "b_quality",
    "quality",
    "method",
    "dqf",
    "orbit_radius_m",
    "orbit_longitude_deg",
    "orbit_latitude_deg",
    "orbit_llr_geo_radius_m",
    "orbit_llr_geo_longitude_deg",
    "orbit_llr_geo_latitude_deg",
]

DROP_COLUMNS_IF_PRESENT = [
    "source_file",
    "source_type",
]


def collect_files(root: Path, satellites: list[str] | None) -> list[Path]:
    files = sorted(root.rglob("*.parquet"))
    if satellites:
        sat_set = set(satellites)
        files = [p for p in files if p.parent.parent.name in sat_set]
    return files


def ensure_sat_id(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "sat_id" not in df.columns:
        df["sat_id"] = path.parent.parent.name
    return df


def minute_reindex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])
    full_index = pd.date_range(ts.min().floor("min"), ts.max().floor("min"), freq="1min")

    df = df.set_index("timestamp")
    df = df.reindex(full_index)
    df.index.name = "timestamp"
    df = df.reset_index()

    if "sat_id" in df.columns:
        df["sat_id"] = df["sat_id"].ffill().bfill()

    return df


def pick_orbit_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "orbit_longitude_deg" not in df.columns and "orbit_llr_geo_longitude_deg" in df.columns:
        df["orbit_longitude_deg"] = df["orbit_llr_geo_longitude_deg"]

    if "orbit_latitude_deg" not in df.columns and "orbit_llr_geo_latitude_deg" in df.columns:
        df["orbit_latitude_deg"] = df["orbit_llr_geo_latitude_deg"]

    if "orbit_radius_m" not in df.columns and "orbit_llr_geo_radius_m" in df.columns:
        df["orbit_radius_m"] = df["orbit_llr_geo_radius_m"]

    return df


def build_mag_source(df: pd.DataFrame) -> pd.Series:
    label = "goesr" if "b_quality" in df.columns or "dqf" in df.columns else "legacy"
    return pd.Series(label, index=df.index, dtype="object")


def harmonize_quality(df: pd.DataFrame) -> pd.DataFrame:
    if "quality_summary" not in df.columns:
        if "b_quality" in df.columns:
            df["quality_summary"] = pd.to_numeric(df["b_quality"], errors="coerce")
        elif "quality" in df.columns:
            df["quality_summary"] = pd.to_numeric(df["quality"], errors="coerce")
        else:
            df["quality_summary"] = pd.NA
    return df


def sanitize_orbit(df: pd.DataFrame) -> pd.DataFrame:
    if "orbit_radius_m" in df.columns:
        r = pd.to_numeric(df["orbit_radius_m"], errors="coerce")
        # Conservative GEO sanity band: about 4.0e7 to 4.3e7 m
        df["orbit_radius_m"] = r.where((r >= 4.0e7) & (r <= 4.3e7), np.nan)

    if "orbit_longitude_deg" in df.columns:
        lon = pd.to_numeric(df["orbit_longitude_deg"], errors="coerce")
        df["orbit_longitude_deg"] = lon.where((lon >= -180) & (lon <= 180), np.nan)

    if "orbit_latitude_deg" in df.columns:
        lat = pd.to_numeric(df["orbit_latitude_deg"], errors="coerce")
        df["orbit_latitude_deg"] = lat.where((lat >= -90) & (lat <= 90), np.nan)

    return df


def drop_all_null_columns(df: pd.DataFrame, protected: list[str]) -> pd.DataFrame:
    removable = [c for c in df.columns if c not in protected]
    all_null = [c for c in removable if df[c].isna().all()]
    if all_null:
        df = df.drop(columns=all_null)
    return df


def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = [
        "b_total_nT",
        "b_gsm_x", "b_gsm_y", "b_gsm_z",
        "b_vdh_x", "b_vdh_y", "b_vdh_z",
        "orbit_longitude_deg",
        "orbit_latitude_deg",
        "orbit_radius_m",
    ]
    int_cols = [
        "quality_summary",
        "num_points",
        "method",
        "dqf",
    ]
    bool_cols = [
        "mag_valid",
    ]

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.array(pd.to_numeric(df[col], errors="coerce"), dtype="Int64")

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    return df


def preprocess_file(path: Path, output_root: Path) -> Path:
    df = pd.read_parquet(path)
    df = ensure_sat_id(df, path)

    for col in DROP_COLUMNS_IF_PRESENT:
        if col in df.columns:
            df = df.drop(columns=col)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = minute_reindex(df)
    df = pick_orbit_columns(df)
    df = harmonize_quality(df)
    df["mag_source"] = build_mag_source(df)

    keep_candidates = ["timestamp", "sat_id", "mag_source"] + CORE_VECTOR_COLS + OPTIONAL_COLS + ["quality_summary"]
    keep_cols = [c for c in keep_candidates if c in df.columns]
    df = df[keep_cols]

    for col in CORE_VECTOR_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = sanitize_orbit(df)

    vec_cols = [c for c in CORE_VECTOR_COLS if c in df.columns]
    if "b_total_nT" in df.columns:
        df["mag_valid"] = df[vec_cols + ["b_total_nT"]].notna().any(axis=1)
    else:
        df["mag_valid"] = df[vec_cols].notna().any(axis=1)

    ordered_cols = [
        "timestamp",
        "sat_id",
        "mag_source",
        "b_total_nT",
        "b_gsm_x", "b_gsm_y", "b_gsm_z",
        "b_vdh_x", "b_vdh_y", "b_vdh_z",
        "quality_summary",
        "num_points",
        "method",
        "dqf",
        "orbit_longitude_deg",
        "orbit_latitude_deg",
        "orbit_radius_m",
        "mag_valid",
    ]
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    df = df[ordered_cols]

    protected = ["timestamp", "sat_id", "mag_source", "mag_valid"]
    df = drop_all_null_columns(df, protected)
    df = standardize_dtypes(df)

    sat_id = str(df["sat_id"].dropna().iloc[0])
    month_start = pd.to_datetime(df["timestamp"].dropna().iloc[0])
    year = f"{month_start.year:04d}"
    yyyymm = f"{month_start.year:04d}{month_start.month:02d}"

    outdir = output_root / sat_id / year
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{sat_id}_{yyyymm}.parquet"
    df.to_parquet(outpath, index=False)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess interim GOES MAG parquet files.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--satellites", nargs="*", default=None, help="Optional sat filters like goes15 goes16")
    args = parser.parse_args()

    files = collect_files(args.input_root, args.satellites)
    if not files:
        print("No interim GOES MAG parquet files found.")
        return

    outputs = []
    for path in files:
        try:
            outpath = preprocess_file(path, args.output_root)
            outputs.append(outpath)
            print(f"wrote {outpath}")
        except Exception as e:
            print(f"failed preprocessing {path}: {e}")

    print(f"\nDone. Wrote {len(outputs)} files.")


if __name__ == "__main__":
    main()
