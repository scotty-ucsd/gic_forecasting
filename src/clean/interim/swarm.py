#!/usr/bin/env python3
"""
Phase 2.4 - Clean Interim: Swarm Raw Daily Parquet to Interim Parquet
----------------------------------------------------------------------
This is the fourth Phase 2 cleaning script. It reads raw daily Swarm
Parquet files produced by Phase 1.3 (src/download/swarm.py), renames
columns from VirES API PascalCase to lowercase snake_case with units,
normalizes timestamps to UTC, deduplicates on timestamp, enforces a
fixed output column schema, and writes either cleaned daily files or
concatenated monthly Parquet files depending on the --clean-type
argument. The output of this script is required before Swarm
preprocessing in Phase 3.x can begin.

Purpose
-------
The raw Swarm Parquet files produced by Phase 1.3 use the VirES API
column naming convention (e.g. Timestamp, Latitude, B_NEC decomposed
into B_N, B_E, B_C). This script standardizes column names to the
pipeline convention (e.g. timestamp, latitude_deg, b_north_nT) and
enforces a fixed COLUMN_ORDER schema so that all downstream consumers
receive a predictable schema regardless of satellite.

The script offers two output modes:
- "monthly" (default): groups daily files by (satellite, year, month),
  concatenates them, deduplicates on timestamp, and writes one Parquet
  file per month. This is the mode expected by Phase 3.x.
- "daily": cleans each raw file in place, preserving the per-satellite
  year/month directory structure, without concatenation. This mode is
  useful for incremental or partial re-runs.

No resampling, interpolation, or gap-filling is applied. Per-file
cleaning is limited to column renaming, UTC timestamp normalization,
dropping rows with unparseable timestamps, and deduplication.

WORK FLOW
---------
Step 1: Parse CLI arguments - input directory, output directory,
        --clean-type (monthly or daily), and optional --force flag.
Step 2: Recursively discover all swarm_*.parquet files under
        --input-dir. Raise FileNotFoundError if none are found.
Step 3 (both modes): For each raw daily file, apply clean_one_file():
        rename columns via COLUMN_RENAME_MAP, normalize timestamp to
        UTC (utc=True to prevent tz-naive/tz-aware comparison errors),
        drop rows with unparseable timestamps, cast satellite to str,
        deduplicate on timestamp, and enforce COLUMN_ORDER.
Step 4a (daily mode): Write each cleaned file to the same relative
        path under --output-dir, mirroring the raw directory structure.
        Skip existing files unless --force is set.
Step 4b (monthly mode): Group cleaned daily files by (sat, year, month)
        parsed from the filename via FILENAME_RE. Concatenate all daily
        frames for each month, apply a second deduplication on timestamp
        across the combined frame, enforce COLUMN_ORDER, and write one
        Parquet file per month. Skip existing files unless --force is set.

INPUT DATA
----------
- data/raw/swarm/**/swarm_*.parquet
  Raw daily Swarm Parquet files produced by Phase 1.3.
  Expected filename pattern: swarm_{A|B|C}_{YYYYMMDD}.parquet
  Expected columns (VirES API names): Timestamp, Spacecraft,
  Latitude, Longitude, Radius, F, Flags_F, Flags_B, B_N, B_E, B_C,
  source_collection.
- CLI optional: --input-dir   (default: data/raw/swarm)
               --output-dir  (default: data/interim/swarm)
               --clean-type  (default: monthly; choices: monthly, daily)
               --force       (overwrite existing output files)

OUTPUT DATA
-----------
- Monthly mode (default):
  data/interim/swarm/{sat}/{YYYY}/swarm_{sat}_{YYYYMM}.parquet
  One Parquet file per satellite per month.
- Daily mode:
  data/interim/swarm/{sat}/{YYYY}/{MM}/swarm_{sat}_{YYYYMMDD}.parquet
  Mirrors raw input directory structure under --output-dir.
- Fixed output schema (COLUMN_ORDER): timestamp, satellite,
  latitude_deg, longitude_deg, radius_m, f_nT, flags_f, flags_b,
  b_north_nT, b_east_nT, b_center_nT, source_collection.

NEXT PIPELINE PHASE
-------------------
- Phase 2.4 has no dependency on any other Phase 2 script and can
  run independently once Phase 1.3 (src/download/swarm.py) has
  completed.
- Run with --clean-type monthly (the default) to produce the output
  expected by Phase 3.x preprocessing.
- After this script completes, the Swarm interim data is ready for
  Phase 3.x preprocessing (src/preprocess/).
- Note the critical ordering constraint in Phase 3.x: the Swarm
  preprocessing script swarm_1min.py must complete before
  swarm_chaos.py can run.
- All other Phase 2.x interim cleaning scripts should also complete
  before moving on to Phase 3.x, since Phase 4 (fusion) requires
  all preprocessed sources to be present.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import pandas as pd


COLUMN_RENAME_MAP = {
    "Timestamp": "timestamp",
    "Spacecraft": "satellite",
    "Latitude": "latitude_deg",
    "Longitude": "longitude_deg",
    "Radius": "radius_m",
    "F": "f_nT",
    "Flags_F": "flags_f",
    "Flags_B": "flags_b",
    "B_N": "b_north_nT",
    "B_E": "b_east_nT",
    "B_C": "b_center_nT",
    "source_collection": "source_collection",
}

COLUMN_ORDER = [
    "timestamp",
    "satellite",
    "latitude_deg",
    "longitude_deg",
    "radius_m",
    "f_nT",
    "flags_f",
    "flags_b",
    "b_north_nT",
    "b_east_nT",
    "b_center_nT",
    "source_collection",
]

FILENAME_RE = re.compile(r"swarm_([ABC])_(\d{8})\.parquet$")


def clean_one_file(infile: Path) -> pd.DataFrame:
    df = pd.read_parquet(infile).copy()
    df = df.rename(columns=COLUMN_RENAME_MAP)

    if "timestamp" not in df.columns:
        raise ValueError(f"missing Timestamp column in {infile}")

    # Normalize to UTC consistently to avoid tz-naive/tz-aware comparison errors.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if "satellite" in df.columns:
        df["satellite"] = df["satellite"].astype(str)

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")

    missing_cols = [col for col in COLUMN_ORDER if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing expected columns in {infile}: {missing_cols}")

    return df[COLUMN_ORDER].reset_index(drop=True)


def parse_file_info(infile: Path) -> tuple[str, str, str]:
    m = FILENAME_RE.match(infile.name)
    if not m:
        raise ValueError(f"unexpected filename format: {infile.name}")
    sat = m.group(1)
    yyyymmdd = m.group(2)
    return sat, yyyymmdd[:4], yyyymmdd[4:6]


def daily_output_path(infile: Path, input_dir: Path, output_dir: Path) -> Path:
    rel = infile.relative_to(input_dir)
    return output_dir / rel


def monthly_output_path(output_dir: Path, sat: str, year: str, month: str) -> Path:
    out_dir = output_dir / sat / year
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"swarm_{sat}_{year}{month}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean raw Swarm parquet files into interim daily or monthly parquet files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/swarm"),
        help="Root directory containing raw daily Swarm parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/swarm"),
        help="Root directory for cleaned interim Swarm parquet files",
    )
    parser.add_argument(
        "--clean-type",
        default="monthly",
        choices=["monthly", "daily"],
        help="Output cleaned daily files or monthly concatenated files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, overwrite existing cleaned daily/monthly output files.",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("swarm_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {args.input_dir}")

    if args.clean_type == "daily":
        for infile in files:
            try:
                outfile = daily_output_path(infile, args.input_dir, args.output_dir)
                if outfile.exists() and not args.force:
                    print(f"skip existing {outfile}")
                    continue

                df = clean_one_file(infile)
                outfile.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(outfile, index=False)
                print(f"cleaned daily {infile} -> {outfile} rows={len(df)}")
            except Exception as e:
                print(f"failed {infile} error={e}")

    elif args.clean_type == "monthly":
        grouped: dict[tuple[str, str, str], list[Path]] = {}

        for infile in files:
            try:
                sat, year, month = parse_file_info(infile)
                grouped.setdefault((sat, year, month), []).append(infile)
            except Exception as e:
                print(f"failed {infile} error={e}")

        for (sat, year, month), month_files in grouped.items():
            try:
                outfile = monthly_output_path(args.output_dir, sat, year, month)
                if outfile.exists() and not args.force:
                    print(f"skip existing {outfile}")
                    continue

                dfs: list[pd.DataFrame] = []
                for path in sorted(month_files):
                    try:
                        cleaned = clean_one_file(path)
                        if not cleaned.empty:
                            dfs.append(cleaned)
                    except Exception as file_error:
                        print(f"failed file {path} error={file_error}")
                if not dfs:
                    print(f"skip monthly sat={sat} year={year} month={month} (all files empty)")
                    continue
                out = pd.concat(dfs, ignore_index=True)
                out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")
                out = out[COLUMN_ORDER].reset_index(drop=True)

                outfile.parent.mkdir(parents=True, exist_ok=True)
                out.to_parquet(outfile, index=False)
                print(f"wrote monthly {outfile} rows={len(out)}")
            except Exception as e:
                print(f"failed monthly sat={sat} year={year} month={month} error={e}")


if __name__ == "__main__":
    main()
