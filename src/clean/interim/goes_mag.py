#!/usr/bin/env python3
"""
Phase 2.2 - Clean Interim: GOES MAG Raw NetCDF to Monthly Parquet
------------------------------------------------------------------
This is the second Phase 2 cleaning script. It reads raw GOES MAG
NetCDF files produced by Phase 1.2 (src/download/goes_mag.py),
normalizes their structure into a consistent 1-minute tabular schema,
decomposes multi-dimensional vector field variables into scalar
columns, and writes cleaned monthly Parquet files partitioned by
satellite and year. The output of this script is required before
GOES MAG preprocessing in Phase 3.x can begin.

Purpose
-------
GOES MAG data is distributed in two incompatible formats depending
on satellite generation. GOES-R satellites (goes16, goes17, goes18)
provide "magn-l2-avg1m" files that are already at 1-minute cadence.
goes15 (legacy) provides "magn-l2-hires" files at a sub-minute
high-resolution cadence with inner-boom and outer-boom sensor columns
and a separate orbital position time coordinate (time_orbit) that is
not aligned to the main measurement time axis.

This script handles both formats through separate code paths:
- GOES-R files are opened directly and their vector variables
  (b_eci, b_epn, b_gse, b_gsm, b_brf, b_vdh, orbit_llr_geo) are
  decomposed into x/y/z or lat/lon/radius scalar columns.
- Legacy hires files are opened into a main measurement DataFrame
  and a separate orbit DataFrame, each resampled independently to
  1-minute means via resample_legacy_to_1min(), then joined. Flag
  and method columns are rounded to the nearest integer before
  being cast to nullable Int64, since resampled means of integer
  flags are floating-point.

The net result is a unified 1-minute schema across all satellites
and generations, with no gap-filling or interpolation applied beyond
the resampling of legacy hires data.

WORK FLOW
---------
Step 1: Parse CLI arguments - raw input directory, output directory,
        optional satellite filter list, and optional date range.
Step 2: Recursively discover all .nc files under --raw-dir, optionally
        filtered to the requested satellite subset.
Step 3: For each .nc file, detect source type from the filename
        (goesr_1min or legacy_hires). Skip files with unknown type.
Step 4a: If goesr_1min, open with xarray and extract b_total_nT,
         quality flags, DQF, and vector variables into scalar columns
         using add_vector_columns(). Cast flag columns to Int64.
Step 4b: If legacy_hires, open with xarray, extract main measurement
         and orbit DataFrames separately, resample both to 1-minute
         means, join on the 1-minute timestamp index, and round-cast
         flag and method columns to nullable Int64.
Step 5: Apply the optional date range filter to each per-file
         DataFrame before accumulating into a per-satellite frame list.
Step 6: For each satellite, concatenate all per-file frames and
         drop duplicate (timestamp, sat_id) pairs keeping the first
         occurrence after sorting by (timestamp, source_file).
Step 7: Partition the merged, sorted DataFrame by sat_id, year, and
         month, and write one Parquet file per partition to the output
         directory.

INPUT DATA
----------
- data/raw/goes_mag/**/*.nc
  Raw GOES MAG NetCDF files produced by Phase 1.2.
  Expected to be organized in per-satellite/year/month subdirectories
  but discovered recursively regardless of depth.
  Two file types are supported:
    - magn-l2-avg1m: GOES-R 1-minute averaged MAG (goes16/17/18)
    - magn-l2-hires: goes15 legacy high-resolution MAG
- CLI optional: --raw-dir    (default: data/raw/goes_mag)
               --output-dir (default: data/interim/goes_mag)
               --satellites (default: all found; e.g. goes15 goes16
                             goes17 goes18)
               --start-date (default: no lower bound; YYYY-MM-DD)
               --end-date   (default: no upper bound; YYYY-MM-DD)

OUTPUT DATA
-----------
- data/interim/goes_mag/{sat_id}/{YYYY}/{sat_id}_{YYYY}{MM}.parquet
  One Parquet file per satellite per month at 1-minute cadence.
  Schema includes timestamp, sat_id, source_file, source_type,
  b_total_nT, quality flags, and decomposed vector field scalar
  columns (e.g. b_epn_x, b_epn_y, b_epn_z, orbit_llr_geo_latitude_deg,
  etc.). Flag and method columns are typed as nullable Int64.

NEXT PIPELINE PHASE
-------------------
- Phase 2.2 has no dependency on any other Phase 2 script and can
  run independently once Phase 1.2 (src/download/goes_mag.py) has
  completed.
- After this script completes, the GOES MAG interim data is ready
  for Phase 3.x preprocessing (src/preprocess/).
- All other Phase 2.x interim cleaning scripts should also complete
  before moving on to Phase 3.x, since Phase 4 (fusion) requires
  all preprocessed sources to be present.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


SAT_RE = re.compile(r"_(g\d{2})_")
GOESR_RE = re.compile(r"magn-l2-avg1m")
LEGACY_RE = re.compile(r"magn-l2-hires")


GOESR_VECTOR_VARS = ["b_eci", "b_epn", "b_gse", "b_gsm", "b_brf", "b_vdh", "orbit_llr_geo"]
LEGACY_VECTOR_VARS = [
    "b_epn", "b_eci", "b_gse", "b_gsm", "b_vdh", "b_brf",
    "b_ib_sensor", "b_ib_epn", "b_ib_eci", "b_ib_gse", "b_ib_gsm", "b_ib_vdh", "b_ib_brf",
    "b_ob_sensor", "b_ob_epn", "b_ob_eci", "b_ob_gse", "b_ob_gsm", "b_ob_vdh", "b_ob_brf",
]

INT_LIKE_COLUMNS = [
    "num_points",
    "b_quality",
    "dqf",
    "quality",
    "quality_ib",
    "quality_ob",
    "method",
]


def detect_sat_id(path: Path) -> str:
    m = SAT_RE.search(path.name.lower())
    if not m:
        raise ValueError(f"Could not infer sat_id from filename: {path.name}")
    return "goes" + m.group(1)[1:]


def detect_source_type(path: Path) -> str:
    name = path.name.lower()
    if GOESR_RE.search(name):
        return "goesr_1min"
    if LEGACY_RE.search(name):
        return "legacy_hires"
    return "unknown"


def normalize_time(values) -> pd.Series:
    ts = pd.to_datetime(values)
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_convert(None)
    return pd.Series(ts, name="timestamp")


def add_vector_columns(data: dict, name: str, arr: np.ndarray, prefix: str | None = None) -> None:
    base = f"{prefix}_{name}" if prefix else name

    if name == "orbit_llr_geo":
        labels = ["latitude_deg", "longitude_deg", "radius_m"]
    else:
        labels = ["x", "y", "z"]

    for i, label in enumerate(labels):
        data[f"{base}_{label}"] = arr[:, i]


def cast_nullable_int(df: pd.DataFrame) -> pd.DataFrame:
    for col in INT_LIKE_COLUMNS:
        if col in df.columns:
            df[col] = pd.array(df[col], dtype="Int64")
    return df


def open_goesr_file(path: Path) -> pd.DataFrame:
    with xr.open_dataset(
        path,
        decode_cf=True,
        mask_and_scale=True,
        decode_times=True,
    ) as ds:
        data: dict[str, object] = {}
        data["timestamp"] = normalize_time(ds["time"].values)
        data["sat_id"] = detect_sat_id(path)
        data["source_file"] = path.name
        data["source_type"] = "goesr_1min"

        if "b_total" in ds:
            data["b_total_nT"] = ds["b_total"].values
        if "num_points" in ds:
            data["num_points"] = ds["num_points"].values
        if "b_quality" in ds:
            data["b_quality"] = ds["b_quality"].values
        if "DQF" in ds:
            data["dqf"] = ds["DQF"].values

        for var in GOESR_VECTOR_VARS:
            if var in ds:
                add_vector_columns(data, var, ds[var].values)

        if "b_epn_ib_sa_corrected" in ds:
            add_vector_columns(data, "b_epn_ib_sa_corrected", ds["b_epn_ib_sa_corrected"].values)

        df = pd.DataFrame(data)

    df = cast_nullable_int(df)
    return df


def open_legacy_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    with xr.open_dataset(
        path,
        decode_cf=True,
        mask_and_scale=True,
        decode_times=True,
    ) as ds:
        data: dict[str, object] = {}
        data["timestamp"] = normalize_time(ds["time"].values)
        data["sat_id"] = detect_sat_id(path)
        data["source_file"] = path.name
        data["source_type"] = "legacy_hires"

        if "b_total" in ds:
            data["b_total_nT"] = ds["b_total"].values
        if "quality" in ds:
            data["quality"] = ds["quality"].values
        if "quality_ib" in ds:
            data["quality_ib"] = ds["quality_ib"].values
        if "quality_ob" in ds:
            data["quality_ob"] = ds["quality_ob"].values
        if "method" in ds:
            data["method"] = ds["method"].values

        for var in LEGACY_VECTOR_VARS:
            if var in ds:
                add_vector_columns(data, var, ds[var].values)

        hires_df = pd.DataFrame(data)

        orbit_df = None
        if "time_orbit" in ds and "orbit_llr_geo" in ds and "time_orbit" in ds["orbit_llr_geo"].dims:
            orbit_data = {
                "timestamp": normalize_time(ds["time_orbit"].values),
                "orbit_latitude_deg": ds["orbit_llr_geo"].values[:, 0],
                "orbit_longitude_deg": ds["orbit_llr_geo"].values[:, 1],
                "orbit_radius_m": ds["orbit_llr_geo"].values[:, 2],
            }
            orbit_df = pd.DataFrame(orbit_data)

    hires_df = cast_nullable_int(hires_df)
    return hires_df, orbit_df


def resample_legacy_to_1min(df: pd.DataFrame, orbit_df: pd.DataFrame | None) -> pd.DataFrame:
    if df.empty:
        return df

    meta_cols = ["sat_id", "source_file", "source_type"]
    numeric_cols = [c for c in df.columns if c not in ["timestamp"] + meta_cols]

    work = df.copy()
    work = work.sort_values("timestamp").set_index("timestamp")

    agg = {}
    for col in numeric_cols:
        if pd.api.types.is_numeric_dtype(work[col]):
            agg[col] = "mean"

    out = work.resample("1min").agg(agg)

    for col in meta_cols:
        out[col] = work[col].resample("1min").first()

    if orbit_df is not None and not orbit_df.empty:
        orbit_work = orbit_df.sort_values("timestamp").set_index("timestamp")
        orbit_1min = orbit_work.resample("1min").mean()
        out = out.join(orbit_1min, how="left")

    out = out.reset_index()

    for col in ["quality", "quality_ib", "quality_ob", "method"]:
        if col in out.columns:
            out[col] = pd.array(np.rint(out[col]), dtype="Int64")

    return out


def collect_raw_files(raw_dir: Path, satellites: list[str] | None = None) -> list[Path]:
    files = sorted(raw_dir.rglob("*.nc"))
    if satellites:
        sat_set = set(satellites)
        files = [p for p in files if detect_sat_id(p) in sat_set]
    return files


def filter_date_range(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if df.empty:
        return df

    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[df["timestamp"] <= end_ts]
    return df


def merge_monthly_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["timestamp", "source_file"]).drop_duplicates(
        subset=["timestamp", "sat_id"],
        keep="first",
    )
    return combined.reset_index(drop=True)


def write_monthly_parquets(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return

    df = df.copy()
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    for (sat_id, year, month), chunk in df.groupby(["sat_id", "year", "month"], sort=True):
        out_dir = output_dir / sat_id / f"{year:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sat_id}_{year:04d}{month:02d}.parquet"

        chunk = chunk.drop(columns=["year", "month"]).sort_values("timestamp").reset_index(drop=True)
        chunk.to_parquet(out_path, index=False)
        print(f"wrote {out_path} rows={len(chunk)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw GOES MAG NetCDF files into standardized interim monthly parquet files.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/goes_mag"),
        help="Root directory containing raw GOES MAG .nc files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/goes_mag"),
        help="Root directory for cleaned interim parquet output.",
    )
    parser.add_argument(
        "--satellites",
        nargs="*",
        default=None,
        help="Optional filters like goes15 goes16 goes17 goes18",
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
        print("no raw mag files found")
        return

    per_sat_frames: dict[str, list[pd.DataFrame]] = {}

    for path in raw_files:
        try:
            source_type = detect_source_type(path)

            if source_type == "goesr_1min":
                df = open_goesr_file(path)
            elif source_type == "legacy_hires":
                hires_df, orbit_df = open_legacy_file(path)
                df = resample_legacy_to_1min(hires_df, orbit_df)
            else:
                print(f"skip unknown MAG type: {path}")
                continue

            df = filter_date_range(df, args.start_date, args.end_date)
            if df.empty:
                continue

            sat_id = str(df["sat_id"].iloc[0])
            per_sat_frames.setdefault(sat_id, []).append(df)
            print(f"loaded {path} rows={len(df)}")

        except Exception as e:
            print(f"failed {path}: {e}")

    for sat_id, frames in per_sat_frames.items():
        merged = merge_monthly_frames(frames)
        write_monthly_parquets(merged, args.output_dir)


if __name__ == "__main__":
    main()
