#!/usr/bin/env python3
"""
Phase 6.1 - Targets: Build Frozen Per-Station Thresholds from Training Data
----------------------------------------------------------------------------
This is the first Phase 6 script and must run before Phase 6.2
(src/targets/attach_station_targets.py). It reads cleaned fused
station-time Parquet files produced by Phase 5.1 for training years
only, computes per-station p95 and p99 percentiles of the GIC proxy
column supermag_abs_dbdt_nt_per_min using Polars lazy evaluation, and
writes the frozen threshold table to
data/fused/metadata/station_thresholds.csv. Phase 6.2 consumes this
CSV to create six binary target columns without any data leakage.

Purpose
-------
Classification targets for geomagnetic disturbance prediction must be
defined relative to station-specific activity thresholds, not global
thresholds, because ground magnetic field variability differs
substantially by latitude, local geology, and proximity to the auroral
zone. A station at high geomagnetic latitude routinely experiences
perturbations that would be extreme at a mid-latitude station.

The freeze-then-apply design enforces a strict leakage boundary:
thresholds are estimated from training data only (default 2015-2022
inclusive) and then applied unchanged to validation and test splits
in Phase 6.2. If thresholds were recomputed on the full dataset,
the classification targets for training rows would depend on the
distributional properties of future data, constituting label leakage.

The target source column is supermag_abs_dbdt_nt_per_min, which is
the absolute 1-minute time derivative of the horizontal magnetic
field perturbation magnitude per station. This column is computed
from the SuperMAG base data in Phase 4.1 and is preserved through
Phase 5.1. Null rows in this column are excluded from the quantile
computation.

The six binary targets that Phase 6.2 will create from these
thresholds are:
- target_geq_p95_30m:  any minute in [t+1, t+30]  exceeds p95
- target_geq_p99_30m:  any minute in [t+1, t+30]  exceeds p99
- target_geq_p95_60m:  any minute in [t+1, t+60]  exceeds p95
- target_geq_p99_60m:  any minute in [t+1, t+60]  exceeds p99
- target_geq_p95_120m: any minute in [t+1, t+120] exceeds p95
- target_geq_p99_120m: any minute in [t+1, t+120] exceeds p99

This script uses Polars (not pandas) for all data operations.
The training dataset is scanned lazily via pl.scan_parquet() and
aggregated via .collect(engine="streaming") to avoid loading all
training months into memory simultaneously.

WORK FLOW
---------
Step 1: Parse CLI arguments - input directory, output CSV path,
        training year range, optional --months filter, and --force flag.
        Raise ValueError if --train-end-year < --train-start-year.
Step 2: Discover all station_time_reduced_plus_target_*.parquet files
        under --input-dir whose YYYYMM year falls within
        [--train-start-year, --train-end-year] inclusive, optionally
        filtered further by --months. Raise FileNotFoundError if none.
Step 3: For each selected file, validate schema via pl.scan_parquet()
        to confirm station, timestamp, and supermag_abs_dbdt_nt_per_min
        are present. Raise ValueError immediately on any missing column.
        Select and cast only the three required columns, then filter
        to non-null rows in all three.
Step 4: Concatenate all per-file LazyFrames via pl.concat(...,
        how="vertical_relaxed") and sort by (station, timestamp).
Step 5: Group by station and aggregate via .collect(engine="streaming"):
        n_samples (len), min_value, median_value, max_value, p95
        (quantile 0.95), p99 (quantile 0.99). Sort output by station.
Step 6: Validate the result:
        - Raise ValueError if any station has null p95 or p99.
        - Raise ValueError if any station has p99 < p95.
Step 7: Write to --output-csv via Polars write_csv(). Raise
        FileExistsError if output exists and --force is not set.

INPUT DATA
----------
- data/fused/interim/{YYYY}/station_time_reduced_plus_target_{YYYYMM}.parquet
  Phase 5.1 output. Only files within the training year range are read.
  Required columns: station, timestamp, supermag_abs_dbdt_nt_per_min.
- CLI optional: --input-dir        (default: data/fused/interim)
               --output-csv       (default: data/fused/metadata/station_thresholds.csv)
               --train-start-year (default: 2015)
               --train-end-year   (default: 2022)
               --months           (default: all within year range)
               --force            (overwrite existing output CSV;
                                   raises FileExistsError if not set)

OUTPUT DATA
-----------
- data/fused/metadata/station_thresholds.csv
  One row per station. Columns: station, n_samples, min_value,
  median_value, max_value, p95, p99. All threshold values are in
  units of nT/min (same units as supermag_abs_dbdt_nt_per_min).
  This is the only non-Parquet output file in the pipeline.
  This file is the required input for Phase 6.2.

NEXT PIPELINE PHASE
-------------------
- Phase 6.1 depends on Phase 5.1 (src/clean/fused/
  clean_fused_station_times.py) completing first for all training
  years.
- Phase 6.2 (src/targets/attach_station_targets.py) must not run
  until this script has written station_thresholds.csv. Phase 6.2
  reads this CSV to apply the frozen thresholds to all splits
  (train, val, and test) and attach the six binary target columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

TARGET_SOURCE_COL = "supermag_abs_dbdt_nt_per_min"
STATION_COL = "station"
TIMESTAMP_COL = "timestamp"

DEFAULT_INPUT_DIR = Path("data/fused/interim")
DEFAULT_OUTPUT_CSV = Path("data/fused/metadata/station_thresholds.csv")

DEFAULT_TRAIN_START_YEAR = 2015
DEFAULT_TRAIN_END_YEAR = 2022


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build frozen station-specific p95/p99 thresholds from training-only "
            "cleaned fused station-time parquet files using Polars."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Root of cleaned fused monthly parquet files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path for station thresholds.",
    )
    parser.add_argument(
        "--train-start-year",
        type=int,
        default=DEFAULT_TRAIN_START_YEAR,
        help="First training year to include.",
    )
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=DEFAULT_TRAIN_END_YEAR,
        help="Last training year to include.",
    )
    parser.add_argument(
        "--months",
        nargs="*",
        default=None,
        help="Optional YYYYMM filters.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output CSV.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_train_files(
    input_root: Path,
    train_start_year: int,
    train_end_year: int,
    months: list[str] | None,
) -> list[Path]:
    files = sorted(input_root.rglob("station_time_reduced_plus_target_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No station_time_reduced_plus_target_*.parquet files found under {input_root}"
        )

    keep_months = set(months) if months else None
    selected: list[Path] = []

    for path in files:
        yyyymm = path.stem.split("_")[-1]
        year = int(yyyymm[:4])

        if keep_months is not None and yyyymm not in keep_months:
            continue
        if train_start_year <= year <= train_end_year:
            selected.append(path)

    if not selected:
        raise FileNotFoundError(
            f"No cleaned fused parquet files matched training years "
            f"{train_start_year}-{train_end_year}."
        )

    return selected


def scan_training_data(files: list[Path]) -> pl.LazyFrame:
    lfs: list[pl.LazyFrame] = []

    for path in files:
        lf = pl.scan_parquet(str(path))
        schema_names = set(lf.collect_schema().names())

        required = {STATION_COL, TIMESTAMP_COL, TARGET_SOURCE_COL}
        missing = required - schema_names
        if missing:
            raise ValueError(
                f"{path.name}: missing required columns: {sorted(missing)}"
            )

        lfs.append(
            lf.select(
                [
                    pl.col(STATION_COL).cast(pl.Utf8, strict=False).alias(STATION_COL),
                    pl.col(TIMESTAMP_COL)
                    .cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
                    .alias(TIMESTAMP_COL),
                    pl.col(TARGET_SOURCE_COL)
                    .cast(pl.Float64, strict=False)
                    .alias(TARGET_SOURCE_COL),
                ]
            ).filter(
                pl.col(STATION_COL).is_not_null()
                & pl.col(TIMESTAMP_COL).is_not_null()
                & pl.col(TARGET_SOURCE_COL).is_not_null()
            )
        )

    if not lfs:
        raise ValueError("No usable training parquet files found.")

    return pl.concat(lfs, how="vertical_relaxed").sort([STATION_COL, TIMESTAMP_COL])


def compute_station_thresholds(lf: pl.LazyFrame) -> pl.DataFrame:
    thresholds = (
        lf.group_by(STATION_COL)
        .agg(
            [
                pl.len().alias("n_samples"),
                pl.col(TARGET_SOURCE_COL).min().alias("min_value"),
                pl.col(TARGET_SOURCE_COL).median().alias("median_value"),
                pl.col(TARGET_SOURCE_COL).max().alias("max_value"),
                pl.col(TARGET_SOURCE_COL).quantile(0.95).alias("p95"),
                pl.col(TARGET_SOURCE_COL).quantile(0.99).alias("p99"),
            ]
        )
        .sort(STATION_COL)
        .collect(engine="streaming")
    )

    if thresholds.height == 0:
        raise ValueError("No threshold rows were produced.")

    bad_null = thresholds.filter(pl.col("p95").is_null() | pl.col("p99").is_null())
    if bad_null.height > 0:
        bad_stations = bad_null.get_column(STATION_COL).to_list()
        raise ValueError(f"Failed to compute thresholds for stations: {bad_stations}")

    bad_order = thresholds.filter(pl.col("p99") < pl.col("p95"))
    if bad_order.height > 0:
        bad_stations = bad_order.get_column(STATION_COL).to_list()
        raise ValueError(
            f"Found stations with p99 < p95, which should not happen: {bad_stations}"
        )

    return thresholds


def write_thresholds_csv(df: pl.DataFrame, output_csv: Path, force: bool) -> None:
    if output_csv.exists() and not force:
        raise FileExistsError(
            f"Output exists and --force was not provided: {output_csv}"
        )

    ensure_dir(output_csv.parent)
    df.write_csv(output_csv)


def main() -> None:
    args = parse_args()

    if args.train_end_year < args.train_start_year:
        raise ValueError("--train-end-year must be >= --train-start-year")

    train_files = discover_train_files(
        input_root=args.input_dir,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        months=args.months,
    )

    train_lf = scan_training_data(train_files)
    thresholds_df = compute_station_thresholds(train_lf)
    write_thresholds_csv(thresholds_df, args.output_csv, args.force)

    row_count = (
        train_lf.select(pl.len().alias("rows"))
        .collect(engine="streaming")
        .item()
    )

    print(f"Training years used: {args.train_start_year}-{args.train_end_year}")
    print(f"Training files used: {len(train_files)}")
    print(f"Stations written: {thresholds_df.select(pl.col(STATION_COL).n_unique()).item()}")
    print(f"Rows used: {int(row_count)}")
    print(f"Output written to: {args.output_csv}")


if __name__ == "__main__":
    main()


