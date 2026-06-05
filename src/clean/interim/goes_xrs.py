#!/usr/bin/env python3
"""
Phase 2.1 - Clean Interim: GOES XRS Raw NetCDF to Monthly Parquet
------------------------------------------------------------------
This is the first Phase 2 cleaning script. It reads raw GOES XRS
NetCDF files produced by Phase 1.1 (src/download/goes_xrs.py),
normalizes their structure into a consistent tabular schema, resolves
duplicate timestamps across overlapping source file types, and writes
cleaned monthly Parquet files partitioned by satellite and year. The
output of this script is required before GOES XRS preprocessing in
Phase 3.x can begin.

Purpose
-------
Raw GOES XRS data is distributed in two distinct file type formats:
"yearly" files (goes15, legacy) and "mission_span" files (goes16,
goes17, goes18, GOES-R series). These formats carry different column
schemas - GOES-R files include extended dual-channel fields, AU
correction factors, and attitude flags that are absent from goes15
files. Because satellite coverage can overlap across file types for
the same time period, this script implements a priority-based
deduplication strategy: for any duplicate (timestamp, satellite)
pair, "yearly" records take priority over "mission_span" records,
which take priority over all other file types.

The script does not interpolate, impute, or gap-fill any values.
Its role is purely structural: unify the NetCDF schema into a flat
DataFrame, cast flag and count columns to nullable Int64 to preserve
the distinction between a missing value and a zero, strip any
timezone information from timestamps, and partition the result into
monthly Parquet files for efficient downstream access.

WORK FLOW
---------
Step 1: Parse CLI arguments - raw input directory, output directory,
        optional satellite filter list, and optional date range.
Step 2: Recursively discover all .nc files under --raw-dir, optionally
        filtered to the requested satellite subset.
Step 3: For each .nc file, open it with xarray (decode_cf=True,
        mask_and_scale=True, decode_times=True), extract the time
        coordinate as a UTC-naive timestamp Series, infer satellite
        name and file type from the filename, and extract all present
        COMMON_COLUMNS and OPTIONAL_COLUMNS variables into a flat
        DataFrame. Cast flag and count columns to nullable Int64.
Step 4: Apply the optional date range filter to each per-file
        DataFrame before accumulating into a per-satellite frame list.
Step 5: For each satellite, concatenate all per-file frames and
        resolve duplicate (timestamp, satellite) pairs by preferring
        higher-priority source file types (yearly > mission_span >
        other) via merge_prefer_higher_priority().
Step 6: Partition the deduplicated, sorted DataFrame by satellite,
        year, and month, and write one Parquet file per partition to
        the output directory.

INPUT DATA
----------
- data/raw/goes_xrs/**/*.nc
  Raw GOES XRS NetCDF files produced by Phase 1.1.
  Expected to be organized in per-satellite subdirectories but
  discovered recursively regardless of depth.
- CLI optional: --raw-dir    (default: data/raw/goes_xrs)
               --output-dir (default: data/interim/goes_xrs)
               --satellites (default: all found; choices: goes15
                             goes16 goes17 goes18)
               --start-date (default: no lower bound; YYYY-MM-DD)
               --end-date   (default: no upper bound; YYYY-MM-DD)

OUTPUT DATA
-----------
- data/interim/goes_xrs/{satellite}/{YYYY}/{satellite}_{YYYY}{MM}.parquet
  One Parquet file per satellite per month. Schema includes
  COMMON_COLUMNS (timestamp, satellite, source_file,
  source_file_type, xrsa_flux, xrsb_flux, flux flags, and counts)
  plus whichever OPTIONAL_COLUMNS were present in the source files
  (extended channel fields, au_factor, roll_angle, attitude flags).
  Flag and count columns are typed as nullable Int64.

NEXT PIPELINE PHASE
-------------------
- Phase 2.1 has no dependency on any other Phase 2 script and can
  run independently once Phase 1.1 (src/download/goes_xrs.py) 
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


MISSION_SPAN_RE = re.compile(r"_s(\d{8})_e(\d{8})_")
YEARLY_RE = re.compile(r"_y(\d{4})_")
SAT_RE = re.compile(r"_(g\d{2})_")


COMMON_COLUMNS = [
    "timestamp",
    "satellite",
    "source_file",
    "source_file_type",
    "xrsa_flux",
    "xrsb_flux",
    "xrsa_flag",
    "xrsb_flag",
    "xrsa_num",
    "xrsb_num",
    "xrsa_flag_excluded",
    "xrsb_flag_excluded",
]

OPTIONAL_COLUMNS = [
    "xrsa_flux_observed",
    "xrsa_flux_electrons",
    "xrsb_flux_observed",
    "xrsb_flux_electrons",
    "au_factor",
    "roll_angle",
    "yaw_flip_flag",
    "electron_correction_flag",
    "xrs_primary_chan",
    "xrsa1_flux",
    "xrsa1_flux_observed",
    "xrsa1_flux_electrons",
    "xrsa2_flux",
    "xrsa2_flux_observed",
    "xrsa2_flux_electrons",
    "xrsb1_flux",
    "xrsb1_flux_observed",
    "xrsb1_flux_electrons",
    "xrsb2_flux",
    "xrsb2_flux_observed",
    "xrsb2_flux_electrons",
    "xrsa1_flag",
    "xrsa2_flag",
    "xrsb1_flag",
    "xrsb2_flag",
    "xrsa1_num",
    "xrsa2_num",
    "xrsb1_num",
    "xrsb2_num",
    "xrsa1_flag_excluded",
    "xrsa2_flag_excluded",
    "xrsb1_flag_excluded",
    "xrsb2_flag_excluded",
]

INT_LIKE_COLUMNS = [
    "xrsa_flag",
    "xrsb_flag",
    "xrsa_num",
    "xrsb_num",
    "xrsa_flag_excluded",
    "xrsb_flag_excluded",
    "yaw_flip_flag",
    "electron_correction_flag",
    "xrs_primary_chan",
    "xrsa1_flag",
    "xrsa2_flag",
    "xrsb1_flag",
    "xrsb2_flag",
    "xrsa1_num",
    "xrsa2_num",
    "xrsb1_num",
    "xrsb2_num",
    "xrsa1_flag_excluded",
    "xrsa2_flag_excluded",
    "xrsb1_flag_excluded",
    "xrsb2_flag_excluded",
]


def detect_file_type(path: Path) -> str:
    name = path.name
    if YEARLY_RE.search(name):
        return "yearly"
    if MISSION_SPAN_RE.search(name):
        return "mission_span"
    return "other"


def detect_satellite(path: Path) -> str:
    m = SAT_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not infer satellite from filename: {path.name}")
    return m.group(1).replace("g", "goes")


def file_priority(path: Path) -> int:
    t = detect_file_type(path)
    if t == "yearly":
        return 2
    if t == "mission_span":
        return 1
    return 0


def normalize_timestamp(ds: xr.Dataset) -> pd.Series:
    time_values = ds["time"].values
    ts = pd.to_datetime(time_values)
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_convert(None)
    return pd.Series(ts, name="timestamp")


def cast_nullable_int(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.array(df[col], dtype="Int64")
    return df


def open_xrs_file(path: Path) -> pd.DataFrame:
    with xr.open_dataset(
        path,
        decode_cf=True,
        mask_and_scale=True,
        decode_times=True,
    ) as ds:
        data = {}
        data["timestamp"] = normalize_timestamp(ds)
        data["satellite"] = detect_satellite(path)
        data["source_file"] = path.name
        data["source_file_type"] = detect_file_type(path)

        keep_cols = [c for c in COMMON_COLUMNS[3:] + OPTIONAL_COLUMNS if c in ds.variables]
        for col in keep_cols:
            arr = ds[col].values
            if getattr(arr, "ndim", 1) == 1:
                data[col] = arr

        df = pd.DataFrame(data)

    df = cast_nullable_int(df, INT_LIKE_COLUMNS)
    return df


def collect_raw_files(raw_dir: Path, satellites: list[str] | None = None) -> list[Path]:
    files = sorted(raw_dir.rglob("*.nc"))
    if satellites:
        sat_set = set(satellites)
        files = [p for p in files if detect_satellite(p) in sat_set]
    return files


def merge_prefer_higher_priority(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["_priority"] = combined["source_file_type"].map(
        {"yearly": 2, "mission_span": 1}
    ).fillna(0)

    combined = combined.sort_values(
        by=["timestamp", "_priority", "source_file"],
        ascending=[True, False, True],
    )

    combined = combined.drop_duplicates(subset=["timestamp", "satellite"], keep="first")
    combined = combined.drop(columns="_priority")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def write_monthly_parquets(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    for (satellite, year, month), chunk in df.groupby(["satellite", "year", "month"], sort=True):
        out_dir = output_dir / satellite / f"{year:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{satellite}_{year:04d}{month:02d}.parquet"

        chunk = chunk.drop(columns=["year", "month"]).sort_values("timestamp").reset_index(drop=True)
        chunk.to_parquet(out_path, index=False)
        print(f"wrote {out_path} rows={len(chunk)}")


def filter_date_range(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if df.empty:
        return df

    if start:
        start_ts = pd.Timestamp(start)
        df = df[df["timestamp"] >= start_ts]
    if end:
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[df["timestamp"] <= end_ts]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw GOES XRS NetCDF files into interim monthly parquet files.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/goes_xrs"),
        help="Root directory containing raw GOES XRS .nc files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/goes_xrs"),
        help="Root directory for cleaned interim parquet output.",
    )
    parser.add_argument(
        "--satellites",
        nargs="*",
        default=None,
        help="Optional satellite filters like goes15 goes16 goes17 goes18",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional inclusive end date YYYY-MM-DD",
    )
    args = parser.parse_args()

    raw_files = collect_raw_files(args.raw_dir, args.satellites)
    if not raw_files:
        print("no raw xrs files found")
        return

    per_sat_frames: dict[str, list[pd.DataFrame]] = {}

    for path in raw_files:
        try:
            df = open_xrs_file(path)
            df = filter_date_range(df, args.start_date, args.end_date)
            if df.empty:
                continue

            sat = str(df["satellite"].iloc[0])
            per_sat_frames.setdefault(sat, []).append(df)
            print(f"loaded {path} rows={len(df)}")
        except Exception as e:
            print(f"failed {path}: {e}")

    for sat, frames in per_sat_frames.items():
        merged = merge_prefer_higher_priority(frames)
        write_monthly_parquets(merged, args.output_dir)


if __name__ == "__main__":
    main()
