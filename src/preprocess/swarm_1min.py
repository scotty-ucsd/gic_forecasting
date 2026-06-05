#!/usr/bin/env python3
"""
Phase 3.1 - Preprocess: Swarm Interim to 1-Minute Aggregated Parquet
---------------------------------------------------------------------
This is the first Phase 3 preprocessing script and must run before
Phase 3.2 (swarm_chaos.py). It reads monthly cleaned Swarm Parquet
files produced by Phase 2.4 (src/clean/interim/swarm.py), aggregates
the 1 Hz measurements to 1-minute bins, and writes per-month 1-minute
Parquet files to data/preprocessed/swarm_1min/. These files are the
required input for swarm_chaos.py (Phase 3.2).

Purpose
-------
The Swarm MAGx_LR_1B product is nominally sampled at 1 Hz. The
downstream fusion stage (Phase 4) operates on a common 1-minute
time grid shared across all data sources. This script performs the
temporal aggregation from 1 Hz to 1-minute cadence and produces
per-bin diagnostic columns (valid counts, missing fractions, flag
maxima) so that data quality is traceable through the pipeline.

The aggregation uses dt.floor("min") to bin timestamps, then groupby
aggregation. Longitude is aggregated using a circular mean (sin/cos
decomposition followed by arctan2 reconstruction) to correctly handle
the 0/360 degree wraparound that an arithmetic mean would corrupt.
Flag columns (flags_f, flags_b) are reduced by max within each bin,
preserving the worst quality indicator present. All three B-component
valid counts are tracked independently, and a combined indicator
n_valid_1min_points counts bins where all three B components are
simultaneously non-null.

This script does not apply CHAOS model corrections. Its output is
intentionally labeled "pre-CHAOS" and is consumed exclusively by
swarm_chaos.py (Phase 3.2), which applies the CHAOS field model
subtraction to isolate the external field perturbation.

WORK FLOW
---------
Step 1: Parse CLI arguments - input directory, output directory,
        and optional --force flag.
Step 2: Recursively discover monthly Swarm Parquet files under
        --input-dir using is_monthly_file() to filter out any daily
        files (requires parent dir to be a 4-digit year and filename
        to contain exactly 2 underscores).
        Raise FileNotFoundError if no matching files are found.
Step 3: For each monthly file, apply aggregate_month():
        - Parse and floor timestamps to 1-minute bins.
        - Coerce all numeric columns to float64.
        - Decompose longitude into sin/cos components for circular
          mean aggregation.
        - Compute per-bin validity indicators: b_all_valid (all three
          B components non-null), f_is_missing, b_any_missing.
        - Group by 1-minute bin and aggregate: mean for position and
          field values, count for valid-point tallies, sum for
          n_valid_1min_points, max for flag columns, first for
          satellite and source_collection, size for n_rows_1min_total,
          mean for missing fraction diagnostics.
        - Reconstruct longitude_deg_mean from arctan2(sin, cos).
        - Cast count columns to int16 and flag columns to Int64.
        - Enforce OUTPUT_COLUMNS column order.
Step 4: Write the aggregated monthly DataFrame to
        data/preprocessed/swarm_1min/{sat}/{YYYY}/{original_name}_1min.parquet.
        Skip existing files unless --force is set.

INPUT DATA
----------
- data/interim/swarm/{sat}/{YYYY}/swarm_{sat}_{YYYYMM}.parquet
  Monthly cleaned Swarm Parquet files produced by Phase 2.4.
  Expected columns: timestamp, satellite, latitude_deg,
  longitude_deg, radius_m, f_nT, flags_f, flags_b, b_north_nT,
  b_east_nT, b_center_nT, source_collection.
- CLI optional: --input-dir  (default: data/interim/swarm)
               --output-dir (default: data/preprocessed/swarm_1min)
               --force      (overwrite existing output files)

OUTPUT DATA
-----------
- data/preprocessed/swarm_1min/{sat}/{YYYY}/swarm_{sat}_{YYYYMM}_1min.parquet
  One Parquet file per satellite per month at 1-minute cadence.
  Output schema (OUTPUT_COLUMNS, 20 columns): timestamp, satellite,
  latitude_deg_mean, longitude_deg_mean (circular mean), radius_m_mean,
  f_nT_mean, b_north_nT_mean, b_east_nT_mean, b_center_nT_mean,
  n_valid_f_1min, n_valid_b_north_1min, n_valid_b_east_1min,
  n_valid_b_center_1min, n_valid_1min_points, flags_f_max,
  flags_b_max, source_collection, n_rows_1min_total,
  f_missing_frac_1min, b_missing_frac_1min.

NEXT PIPELINE PHASE
-------------------
- Phase 3.1 depends on Phase 2.4 (src/clean/interim/swarm.py)
  completing with --clean-type monthly before it can run.
- Phase 3.2 (src/preprocess/swarm_chaos.py) depends on Phase 3.1
  completing and expects files under data/preprocessed/swarm_1min/.
  Do not run swarm_chaos.py until this script has finished.
- After Phase 3.2 completes, both Swarm preprocessing steps are done
  and Swarm data is ready for Phase 4 (src/fusion/).
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "timestamp",
    "satellite",
    "latitude_deg_mean",
    "longitude_deg_mean",
    "radius_m_mean",
    "f_nT_mean",
    "b_north_nT_mean",
    "b_east_nT_mean",
    "b_center_nT_mean",
    "n_valid_f_1min",
    "n_valid_b_north_1min",
    "n_valid_b_east_1min",
    "n_valid_b_center_1min",
    "n_valid_1min_points",
    "flags_f_max",
    "flags_b_max",
    "source_collection",
    "n_rows_1min_total",
    "f_missing_frac_1min",
    "b_missing_frac_1min",
]


def is_monthly_file(path: Path) -> bool:
    return path.parent.name.isdigit() and len(path.parent.name) == 4 and path.name.count("_") == 2


def output_path(infile: Path, output_dir: Path) -> Path:
    sat = infile.parent.parent.name
    year = infile.parent.name
    out_dir = output_dir / sat / year
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = infile.name.replace(".parquet", "_1min.parquet")
    return out_dir / out_name


def aggregate_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["timestamp_1min"] = df["timestamp"].dt.floor("min")

    num_cols = [
        "latitude_deg",
        "longitude_deg",
        "radius_m",
        "f_nT",
        "b_north_nT",
        "b_east_nT",
        "b_center_nT",
        "flags_f",
        "flags_b",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    lon_rad = np.deg2rad(df["longitude_deg"])
    df["lon_sin"] = np.sin(lon_rad)
    df["lon_cos"] = np.cos(lon_rad)

    df["b_all_valid"] = df[["b_north_nT", "b_east_nT", "b_center_nT"]].notna().all(axis=1).astype("int16")
    df["f_is_missing"] = df["f_nT"].isna().astype("int16")
    df["b_any_missing"] = df[["b_north_nT", "b_east_nT", "b_center_nT"]].isna().any(axis=1).astype("int16")

    grouped = (
        df.groupby("timestamp_1min", sort=True)
        .agg(
            satellite=("satellite", "first"),
            latitude_deg_mean=("latitude_deg", "mean"),
            lon_sin_mean=("lon_sin", "mean"),
            lon_cos_mean=("lon_cos", "mean"),
            radius_m_mean=("radius_m", "mean"),
            f_nT_mean=("f_nT", "mean"),
            b_north_nT_mean=("b_north_nT", "mean"),
            b_east_nT_mean=("b_east_nT", "mean"),
            b_center_nT_mean=("b_center_nT", "mean"),
            n_valid_f_1min=("f_nT", "count"),
            n_valid_b_north_1min=("b_north_nT", "count"),
            n_valid_b_east_1min=("b_east_nT", "count"),
            n_valid_b_center_1min=("b_center_nT", "count"),
            n_valid_1min_points=("b_all_valid", "sum"),
            flags_f_max=("flags_f", "max"),
            flags_b_max=("flags_b", "max"),
            source_collection=("source_collection", "first"),
            n_rows_1min_total=("timestamp", "size"),
            f_missing_frac_1min=("f_is_missing", "mean"),
            b_missing_frac_1min=("b_any_missing", "mean"),
        )
        .reset_index()
        .rename(columns={"timestamp_1min": "timestamp"})
    )

    grouped["longitude_deg_mean"] = np.rad2deg(
        np.arctan2(grouped["lon_sin_mean"], grouped["lon_cos_mean"])
    )
    grouped = grouped.drop(columns=["lon_sin_mean", "lon_cos_mean"])

    grouped["flags_f_max"] = grouped["flags_f_max"].astype("Int64")
    grouped["flags_b_max"] = grouped["flags_b_max"].astype("Int64")
    grouped["n_valid_f_1min"] = grouped["n_valid_f_1min"].astype("int16")
    grouped["n_valid_b_north_1min"] = grouped["n_valid_b_north_1min"].astype("int16")
    grouped["n_valid_b_east_1min"] = grouped["n_valid_b_east_1min"].astype("int16")
    grouped["n_valid_b_center_1min"] = grouped["n_valid_b_center_1min"].astype("int16")
    grouped["n_valid_1min_points"] = grouped["n_valid_1min_points"].astype("int16")
    grouped["n_rows_1min_total"] = grouped["n_rows_1min_total"].astype("int16")

    grouped = grouped.sort_values("timestamp").reset_index(drop=True)
    return grouped[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate monthly Swarm interim parquet files to vectorized 1-minute cadence.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/interim/swarm"),
        help="Directory containing monthly cleaned Swarm parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/preprocessed/swarm_1min"),
        help="Directory for 1-minute pre-CHAOS Swarm parquet files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    files = sorted([p for p in args.input_dir.rglob("swarm_*.parquet") if is_monthly_file(p)])
    if not files:
        raise FileNotFoundError(f"No monthly Swarm parquet files found under {args.input_dir}")

    for infile in files:
        try:
            outfile = output_path(infile, args.output_dir)
            if outfile.exists() and not args.force:
                print(f"skip existing {outfile}")
                continue

            df = pd.read_parquet(infile)
            out = aggregate_month(df)
            out.to_parquet(outfile, index=False)
            print(f"wrote {outfile} rows={len(out)}")
        except Exception as e:
            print(f"failed {infile} error={e}")


if __name__ == "__main__":
    main()
