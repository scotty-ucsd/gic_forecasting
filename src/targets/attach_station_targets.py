#!/usr/bin/env python3
"""
Phase 6.2 - Targets: Attach Binary Classification Targets to Labeled Parquet
-----------------------------------------------------------------------------
This is the second and final Phase 6 script. It must not run until
Phase 6.1 (src/targets/build_station_thresholds.py) has completed and
written data/fused/metadata/station_thresholds.csv. It reads the
Phase 5.1 cleaned fused monthly Parquet files and the Phase 6.1 frozen
threshold CSV, computes six binary classification targets using strict
forward-window maxima per station, and writes labeled monthly Parquet
files to data/fused/labeled/. The output of this script is the final
modeling-ready dataset for the entire pipeline.

Purpose
-------
For each station-minute row at issue time t, this script evaluates
whether the station's GIC proxy column (supermag_abs_dbdt_nt_per_min)
exceeds its frozen per-station p95 or p99 threshold at any point
within three forward time horizons:

  (t+1, t+30]  -> target_geq_p95_30m,  target_geq_p99_30m
  (t+1, t+60]  -> target_geq_p95_60m,  target_geq_p99_60m
  (t+1, t+120] -> target_geq_p95_120m, target_geq_p99_120m

All six targets are Int8 (0 or 1) when the full forward window is
available. Any row for which the forward window is incomplete (any
of the required future values is null) receives a null target value
and is dropped from the final labeled output by finalize_labeled_month().
This means the final MAX_HORIZON (120) minutes of the last available
month will have no labeled rows if no next-month file exists.

Leakage safety
--------------
The frozen thresholds loaded from station_thresholds.csv were computed
on training data only in Phase 6.1. This script applies those thresholds
unchanged to all months regardless of split, so no threshold information
from validation or test periods can influence the training target
definition. Forward window computation uses Polars shift(-i).over(station)
expressions which are strictly future-looking from the perspective of
each issue time t.

Forward window implementation
------------------------------
add_forward_targets() uses Polars window expressions rather than
explicit groupby loops. For each horizon H in HORIZONS (30, 60, 120):
- H shifted series are generated as source.shift(-i).over(station)
  for i in 1..H
- pl.max_horizontal(future_vals) computes the forward window maximum
- pl.all_horizontal(valid_vals) checks that all H shifted values are
  non-null, producing a boolean completeness flag per row
- Targets are cast to Int8; incomplete-window rows receive null

The script loads a bounded forward buffer of MAX_HORIZON (120) minutes
from the start of the next month's file via load_with_buffer() to
ensure that rows near the end of the current month have a complete
forward window. Buffer rows are tagged _is_buffer=True and are
clipped off by finalize_labeled_month() before writing.

WORK FLOW
---------
Step 1: Parse CLI arguments - input directory, output directory,
        thresholds CSV path, optional --months filter, --force flag.
Step 2: Load and validate the thresholds CSV via load_thresholds():
        require station, p95, p99 columns; raise FileNotFoundError if
        absent; raise ValueError if null thresholds found.
Step 3: Discover all station_time_reduced_plus_target_*.parquet files
        under --input-dir, optionally filtered by --months.
Step 4: For each monthly file, call process_month():
  Step 4a: Load the current month via scan_month(): validate required
           columns, cast station/timestamp/target-source, filter
           null station and timestamp rows.
  Step 4b: Load forward buffer via load_with_buffer(): append up to
           MAX_HORIZON minutes of the next month tagged _is_buffer=True.
           If no next-month file exists, tag all rows _is_buffer=False.
  Step 4c: Sort by (station, timestamp). Left-join thresholds_df on
           station. Raise ValueError if any station has null thresholds.
  Step 4d: For each horizon H in (30, 60, 120), compute per-station
           forward max and completeness flag via shift().over() and
           horizontal expressions. Compute p95 and p99 target columns.
  Step 4e: Finalize: clip to current-month timestamps, filter to rows
           where all six TARGET_COLS are non-null, drop intermediate
           columns (_is_buffer, threshold columns, _valid_forward_*,
           _target_dbdt_max_*), sort by (station, timestamp).
  Step 4f: Collect summary statistics (row count, station count,
           per-target positive rates) via streaming collect, then
           write via lf.sink_parquet() to
           data/fused/labeled/{YYYY}/station_time_labeled_{YYYYMM}.parquet.
           Skip existing files silently unless --force is set.

INPUT DATA
----------
- data/fused/interim/{YYYY}/station_time_reduced_plus_target_{YYYYMM}.parquet
  Phase 5.1 output. Required columns: station, timestamp,
  supermag_abs_dbdt_nt_per_min.
- data/fused/metadata/station_thresholds.csv
  Phase 6.1 output. Required columns: station, p95, p99.
  Must exist before this script runs.
- CLI optional: --input-dir       (default: data/fused/interim)
               --output-dir      (default: data/fused/labeled)
               --thresholds-csv  (default: data/fused/metadata/
                                  station_thresholds.csv)
               --months          (default: all found)
               --force           (overwrite existing output files)

OUTPUT DATA
-----------
- data/fused/labeled/{YYYY}/station_time_labeled_{YYYYMM}.parquet
  One Parquet file per month. Contains all Phase 5.1 columns plus
  six binary target columns (Int8): target_geq_p95_30m,
  target_geq_p99_30m, target_geq_p95_60m, target_geq_p99_60m,
  target_geq_p95_120m, target_geq_p99_120m. Rows with incomplete
  forward windows (final 120 minutes of any month without a
  next-month buffer) are excluded. Written via sink_parquet()
  (Polars streaming write). Intermediate columns dropped.
  This is the final modeling-ready output of the full pipeline.

NEXT PIPELINE PHASE
-------------------
- Phase 6.2 depends on Phase 6.1 completing and writing
  station_thresholds.csv before it can run.
- Phase 6.2 also depends on Phase 5.1 output being present for all
  months to be labeled.
- This is the final Phase 6 script and the final data preparation
  step in the entire pipeline. The labeled Parquet files in
  data/fused/labeled/ are the primary input for all downstream
  model training, evaluation, and analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

TARGET_SOURCE_COL = "supermag_abs_dbdt_nt_per_min"
STATION_COL = "station"
TIMESTAMP_COL = "timestamp"

THRESHOLD_STATION_COL = "station"
THRESHOLD_P95_COL = "p95"
THRESHOLD_P99_COL = "p99"

HORIZONS = (30, 60, 120)
MAX_HORIZON = max(HORIZONS)

TARGET_COLS = [
    "target_geq_p95_30m",
    "target_geq_p99_30m",
    "target_geq_p95_60m",
    "target_geq_p99_60m",
    "target_geq_p95_120m",
    "target_geq_p99_120m",
]

DEFAULT_INPUT_DIR = Path("data/fused/interim")
DEFAULT_OUTPUT_DIR = Path("data/fused/labeled")
DEFAULT_THRESHOLDS_CSV = Path("data/fused/metadata/station_thresholds.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach frozen station-specific p95/p99 classification targets to cleaned "
            "fused monthly parquet files using strict forward windows."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Root of cleaned fused monthly parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root for labeled monthly parquet outputs.",
    )
    parser.add_argument(
        "--thresholds-csv",
        type=Path,
        default=DEFAULT_THRESHOLDS_CSV,
        help="CSV of frozen station thresholds with columns station,p95,p99.",
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
        help="Overwrite existing output parquet files.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_input_files(input_root: Path, months: list[str] | None) -> list[Path]:
    files = sorted(input_root.rglob("station_time_reduced_plus_target_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No station_time_reduced_plus_target_*.parquet files found under {input_root}"
        )

    if months:
        keep = set(months)
        files = [p for p in files if p.stem.split("_")[-1] in keep]

    if not files:
        raise FileNotFoundError("No input parquet files matched the requested months.")

    return files


def load_thresholds(thresholds_csv: Path) -> pl.DataFrame:
    if not thresholds_csv.exists():
        raise FileNotFoundError(f"Thresholds CSV not found: {thresholds_csv}")

    df = pl.read_csv(thresholds_csv)
    required = {THRESHOLD_STATION_COL, THRESHOLD_P95_COL, THRESHOLD_P99_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Thresholds CSV missing required columns: {sorted(missing)}. "
            "Expected columns: station,p95,p99"
        )

    df = df.select(
        [
            pl.col(THRESHOLD_STATION_COL).cast(pl.Utf8, strict=False).alias(STATION_COL),
            pl.col(THRESHOLD_P95_COL).cast(pl.Float64, strict=False).alias(THRESHOLD_P95_COL),
            pl.col(THRESHOLD_P99_COL).cast(pl.Float64, strict=False).alias(THRESHOLD_P99_COL),
        ]
    ).unique(subset=[STATION_COL], keep="last")

    if df.height == 0:
        raise ValueError("Thresholds CSV contains zero usable rows.")

    bad = df.filter(pl.col(THRESHOLD_P95_COL).is_null() | pl.col(THRESHOLD_P99_COL).is_null())
    if bad.height > 0:
        raise ValueError("Thresholds CSV contains null p95 or p99 values.")

    return df.sort(STATION_COL)


def parse_yyyymm_from_path(parquet_path: Path) -> tuple[int, int]:
    yyyymm = parquet_path.stem.split("_")[-1]
    return int(yyyymm[:4]), int(yyyymm[4:])


def next_month_path(current_path: Path) -> Path | None:
    year, month = parse_yyyymm_from_path(current_path)
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1

    next_yyyymm = f"{next_year}{next_month:02d}"
    next_path = (
        current_path.parent.parent
        / str(next_year)
        / f"station_time_reduced_plus_target_{next_yyyymm}.parquet"
    )
    return next_path if next_path.exists() else None


def scan_month(path: Path) -> pl.LazyFrame:
    lf = pl.scan_parquet(str(path))
    schema = set(lf.collect_schema().names())

    required = {STATION_COL, TIMESTAMP_COL, TARGET_SOURCE_COL}
    missing = required - schema
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {sorted(missing)}")

    return lf.with_columns(
        [
            pl.col(STATION_COL).cast(pl.Utf8, strict=False).alias(STATION_COL),
            pl.col(TIMESTAMP_COL)
            .cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
            .alias(TIMESTAMP_COL),
            pl.col(TARGET_SOURCE_COL).cast(pl.Float64, strict=False).alias(TARGET_SOURCE_COL),
        ]
    ).filter(
        pl.col(STATION_COL).is_not_null() &
        pl.col(TIMESTAMP_COL).is_not_null()
    )


def load_with_buffer(current_path: Path) -> tuple[pl.LazyFrame, pl.Expr | None]:
    current_lf = scan_month(current_path)

    next_path = next_month_path(current_path)
    if next_path is None:
        combined = current_lf.with_columns(pl.lit(False).alias("_is_buffer"))
        return combined, None

    current_max_ts = current_lf.select(pl.col(TIMESTAMP_COL).max()).collect().item()
    next_lf = scan_month(next_path)

    next_buffer = (
        next_lf
        .filter(
            pl.col(TIMESTAMP_COL)
            <= pl.lit(current_max_ts) + pl.duration(minutes=MAX_HORIZON)
        )
        .with_columns(pl.lit(True).alias("_is_buffer"))
    )

    current_tagged = current_lf.with_columns(pl.lit(False).alias("_is_buffer"))
    combined = pl.concat([current_tagged, next_buffer], how="diagonal_relaxed")

    buffer_cutoff_expr = pl.lit(current_max_ts)
    return combined, buffer_cutoff_expr


def add_forward_targets(lf: pl.LazyFrame, thresholds_df: pl.DataFrame) -> pl.LazyFrame:
    lf = lf.sort([STATION_COL, TIMESTAMP_COL]).join(
        thresholds_df.lazy(),
        on=STATION_COL,
        how="left",
    )

    threshold_missing = (
        lf.select(
            [
                pl.col(STATION_COL),
                pl.col(THRESHOLD_P95_COL),
                pl.col(THRESHOLD_P99_COL),
            ]
        )
        .filter(pl.col(THRESHOLD_P95_COL).is_null() | pl.col(THRESHOLD_P99_COL).is_null())
        .limit(5)
        .collect(engine="streaming")
    )
    if threshold_missing.height > 0:
        stations = threshold_missing.get_column(STATION_COL).to_list()
        raise ValueError(f"Missing frozen thresholds for one or more stations: {stations}")

    source = pl.col(TARGET_SOURCE_COL)

    for horizon in HORIZONS:
        future_vals = [source.shift(-i).over(STATION_COL) for i in range(1, horizon + 1)]
        valid_vals = [expr.is_not_null() for expr in future_vals]

        max_expr = pl.max_horizontal(future_vals).alias(f"_target_dbdt_max_{horizon}min")
        valid_expr = pl.all_horizontal(valid_vals).alias(f"_valid_forward_{horizon}m")

        lf = lf.with_columns([max_expr, valid_expr])

        lf = lf.with_columns(
            [
                pl.when(pl.col(f"_valid_forward_{horizon}m"))
                .then((pl.col(f"_target_dbdt_max_{horizon}min") >= pl.col(THRESHOLD_P95_COL)).cast(pl.Int8))
                .otherwise(None)
                .alias(f"target_geq_p95_{horizon}m"),
                pl.when(pl.col(f"_valid_forward_{horizon}m"))
                .then((pl.col(f"_target_dbdt_max_{horizon}min") >= pl.col(THRESHOLD_P99_COL)).cast(pl.Int8))
                .otherwise(None)
                .alias(f"target_geq_p99_{horizon}m"),
            ]
        )

    return lf


def finalize_labeled_month(
    lf: pl.LazyFrame,
    buffer_cutoff_expr: pl.Expr | None,
) -> pl.LazyFrame:
    if buffer_cutoff_expr is not None:
        lf = lf.filter(pl.col(TIMESTAMP_COL) <= buffer_cutoff_expr)

    valid_cols = [f"_valid_forward_{h}m" for h in HORIZONS]
    max_cols = [f"_target_dbdt_max_{h}min" for h in HORIZONS]

    lf = lf.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in TARGET_COLS]))

    drop_cols = ["_is_buffer", THRESHOLD_P95_COL, THRESHOLD_P99_COL] + valid_cols + max_cols
    existing = [c for c in drop_cols if c in lf.collect_schema().names()]
    if existing:
        lf = lf.drop(existing)

    return lf.sort([STATION_COL, TIMESTAMP_COL])


def write_month(
    lf: pl.LazyFrame,
    out_path: Path,
    force: bool,
) -> dict[str, int | float]:
    if out_path.exists() and not force:
        return {
            "skipped": 1,
            "rows": -1,
            "stations": -1,
            "target_geq_p95_30m_rate": float("nan"),
            "target_geq_p99_30m_rate": float("nan"),
            "target_geq_p95_60m_rate": float("nan"),
            "target_geq_p99_60m_rate": float("nan"),
            "target_geq_p95_120m_rate": float("nan"),
            "target_geq_p99_120m_rate": float("nan"),
        }

    ensure_dir(out_path.parent)

    summary_exprs = [
        pl.len().alias("rows"),
        pl.col(STATION_COL).n_unique().alias("stations"),
    ]
    for col in TARGET_COLS:
        summary_exprs.append(pl.col(col).cast(pl.Float64).mean().alias(f"{col}_rate"))

    summary = (
        lf.select(summary_exprs)
        .collect(engine="streaming")
        .to_dicts()[0]
    )

    lf.sink_parquet(str(out_path))
    summary["skipped"] = 0
    return summary


def process_month(
    current_path: Path,
    output_root: Path,
    thresholds_df: pl.DataFrame,
    force: bool,
) -> None:
    yyyymm = current_path.stem.split("_")[-1]
    yyyy = yyyymm[:4]
    out_path = output_root / yyyy / f"station_time_labeled_{yyyymm}.parquet"

    lf, buffer_cutoff_expr = load_with_buffer(current_path)
    lf = add_forward_targets(lf, thresholds_df)
    lf = finalize_labeled_month(lf, buffer_cutoff_expr)

    summary = write_month(lf, out_path, force=force)

    if summary["skipped"] == 1:
        print(f"[skip]  {out_path.name}")
        return

    print(
        f"[ok]    {out_path.name} | "
        f"rows={int(summary['rows']):>8,} | "
        f"stations={int(summary['stations'])} | "
        f"p95_30m={float(summary['target_geq_p95_30m_rate']):.4f} | "
        f"p99_30m={float(summary['target_geq_p99_30m_rate']):.4f} | "
        f"p95_60m={float(summary['target_geq_p95_60m_rate']):.4f} | "
        f"p99_60m={float(summary['target_geq_p99_60m_rate']):.4f} | "
        f"p95_120m={float(summary['target_geq_p95_120m_rate']):.4f} | "
        f"p99_120m={float(summary['target_geq_p99_120m_rate']):.4f}"
    )


def main() -> None:
    args = parse_args()

    thresholds_df = load_thresholds(args.thresholds_csv)
    input_files = discover_input_files(args.input_dir, args.months)

    for path in input_files:
        process_month(
            current_path=path,
            output_root=args.output_dir,
            thresholds_df=thresholds_df,
            force=args.force,
        )


if __name__ == "__main__":
    main()


