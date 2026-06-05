#!/usr/bin/env python3
"""
Phase 3.3 - Preprocess: OMNI Interim Parquet to Preprocessed Parquet
---------------------------------------------------------------------
This is the third Phase 3 preprocessing script. It reads monthly
OMNI Parquet files produced by Phase 2.3 (src/clean/interim/omni.py),
replaces OMNI-specific fill sentinel values with NaN, appends boolean
data validity mask columns, standardizes column types and ordering,
and writes cleaned monthly Parquet files to data/preprocessed/omni/.
This script has no dependency on Phase 3.1 or 3.2 and can run in
parallel with the Swarm preprocessing scripts.

Purpose
-------
OMNI 1-minute solar wind data uses sentinel fill values to indicate
missing measurements rather than NaN or null. For example, IMF total
field Bt uses 999.9 and 9999.99, proton temperature uses 9999999.0,
and spacecraft IDs use 99 and 999. These sentinels are defined in the
OMNIWeb format specification and must be replaced with NaN before the
data can be used in any numeric computation or joined with other
sources. This is the primary purpose of this script and the key
transformation not performed in Phase 2.3.

After fill replacement, five composite boolean validity mask columns
are added by add_validity_masks():
- imf_valid: all four IMF components (Bt, Bx, By, Bz) are non-null
- plasma_valid: solar wind speed, proton density, and flow pressure
  are all non-null
- velocity_vector_valid: all three GSE velocity components non-null
- timeshift_valid: propagation timeshift_sec is non-null
- low_interpolation: percent_interpolation is at or below 20 percent

These masks allow downstream scripts to filter on measurement quality
without repeating the fill-value logic. All mask columns are typed as
pandas nullable boolean.

The interim "datetime" column (reconstructed from year/doy/hour/minute
in Phase 2.3) is renamed to "timestamp" and converted to UTC-aware
pandas Timestamp. The raw time-index columns (year, doy, hour, minute,
yyyymm) are dropped since they are redundant with timestamp. Columns
that are entirely null in a given file and are not in PROTECTED_COLUMNS
are removed by drop_all_null_nonprotected() to guard against schema
drift across OMNI annual files.

WORK FLOW
---------
Step 1: Parse CLI arguments - input root, output root, optional date
        range, and optional --force flag.
Step 2: Discover all omni_*.parquet files one directory level below
        --input-root (i.e. data/interim/omni/{YYYY}/omni_*.parquet).
        Raise FileNotFoundError if none are found.
Step 3: For each file, append source_file and source_year_dir
        provenance columns, then apply standardize_columns():
        rename "datetime" -> "timestamp" via RENAME_MAP, convert to
        UTC-aware Timestamp, drop rows with unparseable timestamps,
        drop raw time-index columns, ensure "date" string column exists.
Step 4: Apply optional date range filter (UTC-aware comparison).
        Skip the file if the result is empty.
Step 5: Apply replace_fill_values(): for each physical column defined
        in OMNI_FILL_VALUES, coerce to numeric and replace each known
        sentinel value with np.nan.
Step 6: Apply add_validity_masks(): compute five composite boolean
        validity columns (imf_valid, plasma_valid,
        velocity_vector_valid, timeshift_valid, low_interpolation).
Step 7: Apply standardize_dtypes(): cast spacecraft ID and averaging
        count columns to nullable Int64; cast physical measurement
        columns to float64; cast validity mask columns to pandas
        nullable boolean.
Step 8: Apply drop_all_null_nonprotected(): drop any non-protected
        column that is entirely null in this file.
Step 9: Sort by timestamp, deduplicate on timestamp keeping the last
        occurrence, reorder columns with PROTECTED_COLUMNS first, and
        write to data/preprocessed/omni/{YYYY}/omni_{YYYYMM}.parquet.
        Skip existing files unless --force is set.

INPUT DATA
----------
- data/interim/omni/{YYYY}/omni_{YYYYMM}.parquet
  Monthly OMNI Parquet files produced by Phase 2.3.
  Contains raw OMNI columns including year, doy, hour, minute,
  physical measurements, and a "datetime" column.
- CLI optional: --input-root  (default: data/interim/omni)
               --output-root (default: data/preprocessed/omni)
               --start-date  (default: no lower bound; YYYY-MM-DD)
               --end-date    (default: no upper bound; YYYY-MM-DD)
               --force       (overwrite existing output files)

OUTPUT DATA
-----------
- data/preprocessed/omni/{YYYY}/omni_{YYYYMM}.parquet
  One Parquet file per month. Schema leads with PROTECTED_COLUMNS
  (timestamp, date, source_file, source_year_dir, bt_nT,
  bx_gse_gsm_nT, by_gsm_nT, bz_gsm_nT, speed_km_s, vx_gse_km_s,
  vy_gse_km_s, vz_gse_km_s, proton_density_n_cc,
  proton_temperature_K, flow_pressure_nPa, percent_interpolation,
  timeshift_sec, time_between_obs_sec, n_points_imf_avg,
  n_points_plasma_avg, imf_spacecraft_id, plasma_spacecraft_id),
  followed by the five boolean validity mask columns, then any
  remaining non-null columns. Fill sentinels replaced with NaN.
  Physical columns typed as float64; ID/count columns as Int64;
  validity masks as pandas nullable boolean.

NEXT PIPELINE PHASE
-------------------
- Phase 3.3 depends only on Phase 2.3 (src/clean/interim/omni.py)
  completing first. It has no dependency on Phase 3.1 or 3.2 and
  can run in parallel with the Swarm preprocessing scripts.
- After this script completes, OMNI preprocessed data is ready for
  Phase 4 (src/fusion/data_fusion.py).
- All Phase 3.x preprocessing scripts must complete before Phase 4
  can begin, since the fusion stage requires all sources to be present.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd


OMNI_FILL_VALUES = {
    "bt_nT": [999.9, 9999.99],
    "bx_gse_gsm_nT": [999.9, 9999.99],
    "by_gsm_nT": [999.9, 9999.99],
    "bz_gsm_nT": [999.9, 9999.99],
    "speed_km_s": [9999.0, 99999.9],
    "vx_gse_km_s": [99999.9],
    "vy_gse_km_s": [99999.9],
    "vz_gse_km_s": [99999.9],
    "proton_density_n_cc": [999.9, 999.99],
    "proton_temperature_K": [9999999.0],
    "flow_pressure_nPa": [99.99, 999.99],
    "percent_interpolation": [999, 999.0],
    "timeshift_sec": [999999],
    "time_between_obs_sec": [999999],
    "n_points_imf_avg": [999, 999.0],
    "n_points_plasma_avg": [999, 999.0],
    "imf_spacecraft_id": [99, 999],
    "plasma_spacecraft_id": [99, 999],
}

RENAME_MAP = {
    "datetime": "timestamp",
}

PROTECTED_COLUMNS = [
    "timestamp",
    "date",
    "source_file",
    "source_year_dir",
    "bt_nT",
    "bx_gse_gsm_nT",
    "by_gsm_nT",
    "bz_gsm_nT",
    "speed_km_s",
    "vx_gse_km_s",
    "vy_gse_km_s",
    "vz_gse_km_s",
    "proton_density_n_cc",
    "proton_temperature_K",
    "flow_pressure_nPa",
    "percent_interpolation",
    "timeshift_sec",
    "time_between_obs_sec",
    "n_points_imf_avg",
    "n_points_plasma_avg",
    "imf_spacecraft_id",
    "plasma_spacecraft_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess interim OMNI parquet files into standardized monthly source tables."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/interim/omni"),
        help="Root directory containing interim OMNI parquet files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/preprocessed/omni"),
        help="Root directory for preprocessed OMNI parquet output.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive start date YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional inclusive end date YYYY-MM-DD.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def find_input_files(input_root: Path) -> list[Path]:
    files = sorted(input_root.glob("*/omni_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No OMNI parquet files found under {input_root}")
    return files


def parse_yyyymm_from_path(path: Path) -> str | None:
    match = re.search(r"(\d{6})", path.stem)
    return match.group(1) if match else None


def replace_fill_values(df: pd.DataFrame) -> pd.DataFrame:
    for col, fills in OMNI_FILL_VALUES.items():
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        for fill in fills:
            df.loc[df[col] == fill, col] = np.nan
    return df


def add_validity_masks(df: pd.DataFrame) -> pd.DataFrame:
    df["imf_valid"] = df[
        ["bt_nT", "bx_gse_gsm_nT", "by_gsm_nT", "bz_gsm_nT"]
    ].notna().all(axis=1)

    df["plasma_valid"] = df[
        ["speed_km_s", "proton_density_n_cc", "flow_pressure_nPa"]
    ].notna().all(axis=1)

    df["velocity_vector_valid"] = df[
        ["vx_gse_km_s", "vy_gse_km_s", "vz_gse_km_s"]
    ].notna().all(axis=1)

    df["timeshift_valid"] = (
        df["timeshift_sec"].notna() if "timeshift_sec" in df.columns else False
    )

    df["low_interpolation"] = (
        df["percent_interpolation"].le(20)
        if "percent_interpolation" in df.columns
        else False
    )

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME_MAP).copy()

    if "timestamp" not in df.columns:
        raise ValueError("Expected interim OMNI input to contain a datetime column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    drop_cols = [c for c in ["year", "doy", "hour", "minute", "yyyymm"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if "date" not in df.columns:
        df["date"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M")

    return df


def apply_date_filter(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        df = df[df["timestamp"] >= start_ts]

    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        df = df[df["timestamp"] <= end_ts]

    return df


def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    int_like_cols = [
        "imf_spacecraft_id",
        "plasma_spacecraft_id",
        "n_points_imf_avg",
        "n_points_plasma_avg",
    ]
    for col in int_like_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    float_cols = [
        "percent_interpolation",
        "timeshift_sec",
        "time_between_obs_sec",
        "bt_nT",
        "bx_gse_gsm_nT",
        "by_gsm_nT",
        "bz_gsm_nT",
        "speed_km_s",
        "vx_gse_km_s",
        "vy_gse_km_s",
        "vz_gse_km_s",
        "proton_density_n_cc",
        "proton_temperature_K",
        "flow_pressure_nPa",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = [
        "imf_valid",
        "plasma_valid",
        "velocity_vector_valid",
        "timeshift_valid",
        "low_interpolation",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    return df


def drop_all_null_nonprotected(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for col in df.columns:
        if col in PROTECTED_COLUMNS or col.endswith("_valid") or col == "low_interpolation":
            keep.append(col)
            continue
        if not df[col].isna().all():
            keep.append(col)
    return df[keep].copy()


def preprocess_file(
    path: Path,
    output_root: Path,
    start_date: str | None,
    end_date: str | None,
    force: bool,
) -> None:
    yyyymm = parse_yyyymm_from_path(path)
    if yyyymm is None:
        print(f"skip (cannot infer yyyymm): {path}")
        return

    year = yyyymm[:4]
    out_dir = output_root / year
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"omni_{yyyymm}.parquet"

    if out_path.exists() and not force:
        print(f"skip (exists): {out_path}")
        return

    df = pd.read_parquet(path)
    df["source_file"] = path.name
    df["source_year_dir"] = path.parent.name

    df = standardize_columns(df)
    df = apply_date_filter(df, start_date, end_date)
    if df.empty:
        print(f"skip (empty after date filter): {path}")
        return

    df = replace_fill_values(df)
    df = add_validity_masks(df)
    df = standardize_dtypes(df)
    df = drop_all_null_nonprotected(df)

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    expected_order = [c for c in PROTECTED_COLUMNS if c in df.columns]
    remaining = [c for c in df.columns if c not in expected_order]
    df = df[expected_order + remaining]

    df.to_parquet(out_path, index=False)
    print(
        f"wrote {out_path} | rows={len(df)} | "
        f"start={df['timestamp'].min()} | end={df['timestamp'].max()}"
    )


def main() -> None:
    args = parse_args()
    files = find_input_files(args.input_root)

    for path in files:
        preprocess_file(
            path=path,
            output_root=args.output_root,
            start_date=args.start_date,
            end_date=args.end_date,
            force=args.force,
        )


if __name__ == "__main__":
    main()
