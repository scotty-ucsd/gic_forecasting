#!/usr/bin/env python3
"""
Phase 8.1 - Dataset: Build ML-Ready Train/Val/Test Splits per Target
---------------------------------------------------------------------
This is the sole Phase 8 script and the only script in src/dataset/.
It reads the standardized monthly feature Parquet files produced by
Phase 7.1 (src/features/build_feature_regimes.py), assigns each row
to a train/val/test split, validates split chronological ordering,
selects numeric feature columns while excluding leakage-prone columns,
and writes one train/val/test Parquet triplet per target to
data/ml/standardized/. Seven metadata artifacts are written after all
targets complete. This script uses Polars throughout.

Purpose
-------
Phase 7.1 produces a unified monthly feature table combining all
source families and engineered features with six binary classification
targets. This script packages that table into the final model-ready
dataset structure that all downstream training, evaluation, and
feature importance scripts consume directly.

Classification only: regression targets have been removed from the
pipeline. The six canonical binary targets are:
- target_geq_p95_30m, target_geq_p99_30m   (30-minute horizon)
- target_geq_p95_60m, target_geq_p99_60m   (60-minute horizon)
- target_geq_p95_120m, target_geq_p99_120m (120-minute horizon)

Per-target output structure: for each of the six targets, a dedicated
subdirectory is created under --output-root containing three files:
train.parquet, val.parquet, test.parquet. The feature columns are
identical across all six targets; only the target column and
(when --drop-missing-targets is set) the set of retained rows differ.

Split strategies
----------------
Two strategies are available via --split-by (default: year):

1. year (default): derive_year_split_map() reads the unique years
   present in the data. If data/figures/features/candidate_split_summary.csv
   exists and contains "year" and "suggested_split" columns, those
   assignments are used directly. Otherwise an automatic fallback rule
   applies: the last two years become val and test respectively; all
   earlier years become train. One- and two-year datasets are handled
   gracefully.

2. date: parse_date_split_config() reads a JSON file at
   --split-config-json. Requires "train", "val", and "test" keys each
   with "start" and "end" date subkeys. Rows with timestamps falling
   outside all three ranges receive a null split and are excluded.

Split boundary validation
--------------------------
Before any output files are written, validate_split_boundaries()
checks: (1) each split is non-empty; (2) train max_time < val min_time;
(3) val max_time < test min_time. If any check status is "fail",
partial metadata (including split_validation_report.csv) is written
and a ValueError is raised immediately. No target Parquet files are
written when validation fails.

Feature column selection
-------------------------
choose_feature_columns() scans column names and dtypes and keeps a
column as a feature only if all four conditions hold:
- dtype is numeric (Int8/16/32/64, UInt8/16/32/64, Float32/64, Boolean)
- not in NON_FEATURE_COLS (timestamp, station, year, month, yyyymm,
  all six target columns, split)
- not in EXCLUDED_FEATURE_COLS (supermag_dbn_nt, supermag_dbe_nt,
  supermag_dbz_nt -- raw local SuperMAG disturbance components, which
  would constitute direct label leakage for GIC proxy targets)
- lowercased name does not contain "target", "future", "lead", or "t_plus"

WORK FLOW
---------
Step 1: Parse CLI arguments - input root, output root, optional
        --months filter, optional --split-summary-csv path,
        --split-by strategy, optional --split-config-json path,
        --drop-missing-targets flag (default True), --force flag.
Step 2: Discover all station_time_features_*.parquet files under
        --input-root, optionally filtered by --months. Scan all files
        via pl.scan_parquet() with diagonal_relaxed concat. Add year/
        month/yyyymm columns if absent. Sort by (station, timestamp).
Step 3: Assign split column via assign_splits() using the selected
        strategy. Filter to rows where split is in (train, val, test).
        Raise ValueError if zero rows are assigned.
Step 4: Select feature columns via choose_feature_columns(). Identify
        metadata columns present in schema.
Step 5: Run validate_split_boundaries(). Raise ValueError with partial
        metadata write if any check fails.
Step 6: For each of the six TARGET_COLS:
        - Build per-target LazyFrame with build_task_lazyframe():
          select metadata + feature + target + split columns; if
          --drop-missing-targets, filter null target rows.
        - Write train/val/test Parquet files via write_split_files()
          to data/ml/standardized/{target_col}/{split}.parquet using
          sink_parquet(). Skip existing files unless --force is set.
        - Collect per-split summary statistics (rows, stations,
          min/max time, event_rate, target_mean, target_p95).
Step 7: Concatenate all per-target summaries and write 7 metadata
        artifacts to data/ml/standardized/metadata/:
        feature_columns.json, dropped_non_feature_columns.json,
        target_columns.json, excluded_feature_columns.json,
        split_config.json, dataset_summary.csv,
        split_validation_report.csv.

INPUT DATA
----------
- data/preprocessed/fused/standardized/{YYYY}/station_time_features_{YYYYMM}.parquet
  Phase 7.1 output. Required columns: timestamp, station, all six
  target_geq_* columns.
- data/figures/features/candidate_split_summary.csv (optional)
  If present and contains "year" and "suggested_split" columns, used
  to assign year-based splits instead of the automatic fallback rule.
- --split-config-json (optional, required only for --split-by date)
  JSON with train/val/test start and end date strings.
- CLI optional: --input-root           (default: data/preprocessed/fused/standardized)
               --output-root          (default: data/ml/standardized)
               --months               (default: all found)
               --split-summary-csv    (default: data/figures/features/
                                       candidate_split_summary.csv)
               --split-by             (default: year; choices: year, date)
               --split-config-json    (default: None)
               --drop-missing-targets (default: True; use
                                       --no-drop-missing-targets to disable)
               --force                (overwrite existing output files)

OUTPUT DATA
-----------
- data/ml/standardized/{target_col}/{split}.parquet
  18 Parquet files total (6 targets x 3 splits). Each file contains
  metadata columns (timestamp, station, year, month, yyyymm), all
  selected numeric feature columns, the target column (Int8), and the
  split column. Written via sink_parquet() (Polars streaming write).
- data/ml/standardized/metadata/feature_columns.json
- data/ml/standardized/metadata/dropped_non_feature_columns.json
- data/ml/standardized/metadata/target_columns.json
- data/ml/standardized/metadata/excluded_feature_columns.json
- data/ml/standardized/metadata/split_config.json
- data/ml/standardized/metadata/dataset_summary.csv
  Per-target per-split: rows, stations, min_time, max_time,
  event_rate, target_mean, target_p95.
- data/ml/standardized/metadata/split_validation_report.csv
  Per-check: check name, status (pass/warn/fail), details.

NEXT PIPELINE PHASE
-------------------
- Phase 8.1 depends on Phase 7.1 (src/features/build_feature_regimes.py)
  completing first for all months to be included in the dataset.
- This is the final Phase 8 script and the last data preparation step
  before model training.
- The output at data/ml/standardized/ is the direct input for all
  downstream model training scripts (Phase 9 and beyond) and for
  src/features/evaluate_feature_importance.py, which runs after
  Phase 9 model outputs are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

DEFAULT_INPUT_ROOT = Path("data/preprocessed/fused/standardized")
DEFAULT_OUTPUT_ROOT = Path("data/ml/standardized")
DEFAULT_SPLIT_SUMMARY = Path("data/figures/features/candidate_split_summary.csv")

TARGET_COLS = [
    "target_geq_p95_30m",
    "target_geq_p99_30m",
    "target_geq_p95_60m",
    "target_geq_p99_60m",
    "target_geq_p95_120m",
    "target_geq_p99_120m",
]

METADATA_COLS = ["timestamp", "station", "year", "month", "yyyymm"]
NON_FEATURE_COLS = set(METADATA_COLS + TARGET_COLS + ["split"])

EXCLUDED_FEATURE_COLS = {
    "supermag_dbn_nt",
    "supermag_dbe_nt",
    "supermag_dbz_nt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ML-ready classification datasets from standardized fused feature parquet files using Polars."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--months", nargs="*", default=None, help="Optional YYYYMM filters for feature files.")
    parser.add_argument(
        "--split-summary-csv",
        type=Path,
        default=DEFAULT_SPLIT_SUMMARY,
        help="Optional split summary from earlier features inspection.",
    )
    parser.add_argument(
        "--split-by",
        choices=["year", "date"],
        default="year",
        help="Split strategy: year-based or explicit date ranges from --split-config-json.",
    )
    parser.add_argument(
        "--split-config-json",
        type=Path,
        default=None,
        help="Optional JSON containing date ranges for train/val/test when --split-by date.",
    )
    parser.add_argument(
        "--drop-missing-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows missing a task target before writing each split.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_feature_files(root: Path, months: list[str] | None) -> list[Path]:
    files = sorted(root.rglob("station_time_features_*.parquet"))
    if months:
        keep = set(months)
        files = [p for p in files if p.stem.split("_")[-1] in keep]
    if not files:
        raise FileNotFoundError(f"No feature parquet files found under {root}")
    return files


def scan_features(files: list[Path]) -> pl.LazyFrame:
    lfs: list[pl.LazyFrame] = []

    for path in files:
        lf = pl.scan_parquet(str(path))
        schema = set(lf.collect_schema().names())

        required = {"timestamp", "station", *TARGET_COLS}
        missing = required - schema
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")

        lfs.append(
            lf.with_columns(
                [
                    pl.col("timestamp")
                    .cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
                    .alias("timestamp"),
                    pl.col("station").cast(pl.Utf8, strict=False).str.to_uppercase().alias("station"),
                ]
            ).filter(
                pl.col("timestamp").is_not_null()
                & pl.col("station").is_not_null()
            )
        )

    lf = pl.concat(lfs, how="diagonal_relaxed")

    schema = set(lf.collect_schema().names())
    exprs = []
    if "year" not in schema:
        exprs.append(pl.col("timestamp").dt.year().cast(pl.Int16).alias("year"))
    if "month" not in schema:
        exprs.append(pl.col("timestamp").dt.strftime("%Y-%m").alias("month"))
    if "yyyymm" not in schema:
        exprs.append(pl.col("timestamp").dt.strftime("%Y%m").alias("yyyymm"))

    if exprs:
        lf = lf.with_columns(exprs)

    return lf.sort(["station", "timestamp"])


def derive_year_split_map(lf: pl.LazyFrame, split_summary_csv: Path) -> tuple[dict[int, str], dict]:
    years = (
        lf.select(pl.col("year").drop_nulls().unique().sort())
        .collect(engine="streaming")
        .get_column("year")
        .to_list()
    )
    years = [int(y) for y in years]

    split_map: dict[int, str] = {}

    if split_summary_csv.exists():
        split_df = pl.read_csv(split_summary_csv)
        if {"year", "suggested_split"}.issubset(set(split_df.columns)):
            valid_rows = split_df.filter(pl.col("suggested_split").is_in(["train", "val", "test"]))
            for row in valid_rows.iter_rows(named=True):
                split_map[int(row["year"])] = str(row["suggested_split"])

    if not split_map:
        if len(years) >= 3:
            split_map = {y: "train" for y in years[:-2]}
            split_map[years[-2]] = "val"
            split_map[years[-1]] = "test"
        elif len(years) == 2:
            split_map = {years[0]: "train", years[1]: "test"}
        elif len(years) == 1:
            split_map = {years[0]: "train"}

    config = {"strategy": "year", "year_assignments": split_map}
    return split_map, config


def parse_date_split_config(config_path: Path) -> dict:
    if config_path is None or not config_path.exists():
        raise FileNotFoundError("--split-config-json is required for --split-by date")

    data = json.loads(config_path.read_text())
    for split in ["train", "val", "test"]:
        if split not in data:
            raise ValueError(f"Missing '{split}' in split config.")
        if "start" not in data[split] or "end" not in data[split]:
            raise ValueError(f"Missing start/end for split '{split}'.")
    return data


def assign_splits(lf: pl.LazyFrame, args: argparse.Namespace) -> tuple[pl.LazyFrame, dict]:
    if args.split_by == "year":
        split_map, config = derive_year_split_map(lf, args.split_summary_csv)
        split_expr = (
            pl.col("year")
            .replace_strict(split_map, default=None)
            .cast(pl.Utf8)
            .alias("split")
        )
        return lf.with_columns(split_expr), config

    config = parse_date_split_config(args.split_config_json)
    train_start = pl.lit(config["train"]["start"]).str.strptime(pl.Datetime, strict=False).dt.replace_time_zone("UTC")
    train_end = pl.lit(config["train"]["end"]).str.strptime(pl.Datetime, strict=False).dt.replace_time_zone("UTC")
    val_start = pl.lit(config["val"]["start"]).str.strptime(pl.Datetime, strict=False).dt.replace_time_zone("UTC")
    val_end = pl.lit(config["val"]["end"]).str.strptime(pl.Datetime, strict=False).dt.replace_time_zone("UTC")
    test_start = pl.lit(config["test"]["start"]).str.strptime(pl.Datetime, strict=False).dt.replace_time_zone("UTC")
    test_end = pl.lit(config["test"]["end"]).str.strptime(pl.Datetime, strict=False).dt.replace_time_zone("UTC")

    split_expr = (
        pl.when((pl.col("timestamp") >= train_start) & (pl.col("timestamp") <= train_end)).then(pl.lit("train"))
        .when((pl.col("timestamp") >= val_start) & (pl.col("timestamp") <= val_end)).then(pl.lit("val"))
        .when((pl.col("timestamp") >= test_start) & (pl.col("timestamp") <= test_end)).then(pl.lit("test"))
        .otherwise(None)
        .alias("split")
    )
    return lf.with_columns(split_expr), {"strategy": "date", "ranges": config}


def choose_feature_columns(lf: pl.LazyFrame) -> tuple[list[str], list[str]]:
    schema = lf.collect_schema()
    feature_cols: list[str] = []
    dropped_cols: list[str] = []

    numeric_like = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64, pl.Boolean,
    }

    for c, dtype in schema.items():
        low = c.lower()

        if c in NON_FEATURE_COLS or c == "split":
            dropped_cols.append(c)
            continue

        if c in EXCLUDED_FEATURE_COLS:
            dropped_cols.append(c)
            continue

        if any(tok in low for tok in ["target", "future", "lead", "t_plus"]):
            dropped_cols.append(c)
            continue

        if dtype in numeric_like:
            feature_cols.append(c)
        else:
            dropped_cols.append(c)

    return sorted(feature_cols), sorted(set(dropped_cols))


def validate_split_boundaries(lf: pl.LazyFrame) -> pl.DataFrame:
    ranges = (
        lf.filter(pl.col("split").is_in(["train", "val", "test"]))
        .group_by("split")
        .agg(
            pl.len().alias("rows"),
            pl.col("timestamp").min().alias("min_time"),
            pl.col("timestamp").max().alias("max_time"),
        )
        .collect(engine="streaming")
    )

    split_map = {row["split"]: row for row in ranges.iter_rows(named=True)}
    reports: list[dict] = []

    for split in ["train", "val", "test"]:
        row = split_map.get(split)
        if row is None:
            reports.append({"check": f"split_nonempty_{split}", "status": "warn", "details": "split has zero rows"})
        else:
            reports.append(
                {
                    "check": f"split_nonempty_{split}",
                    "status": "pass",
                    "details": f"rows={row['rows']}, min={row['min_time']}, max={row['max_time']}",
                }
            )

    def add_pair_check(left: str, right: str, label: str) -> None:
        lrow = split_map.get(left)
        rrow = split_map.get(right)
        if lrow is None or rrow is None:
            return
        status = "pass" if lrow["max_time"] < rrow["min_time"] else "fail"
        reports.append(
            {
                "check": label,
                "status": status,
                "details": f"{left}_max={lrow['max_time']}, {right}_min={rrow['min_time']}",
            }
        )

    add_pair_check("train", "val", "train_before_val")
    add_pair_check("val", "test", "val_before_test")
    if "val" not in split_map:
        add_pair_check("train", "test", "train_before_test")

    return pl.DataFrame(reports)


def build_task_lazyframe(
    lf: pl.LazyFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    metadata_cols: list[str],
    drop_missing_target: bool,
) -> pl.LazyFrame:
    keep_cols = [c for c in metadata_cols + feature_cols + [target_col, "split"] if c in lf.collect_schema().names()]
    out = lf.select(keep_cols)

    if drop_missing_target:
        out = out.filter(pl.col(target_col).is_not_null())

    return out.sort(["station", "timestamp"])


def write_split_files(task_lf: pl.LazyFrame, task_dir: Path, force: bool) -> dict[str, int]:
    ensure_dir(task_dir)

    counts_df = (
        task_lf.group_by("split")
        .agg(pl.len().alias("rows"))
        .collect(engine="streaming")
    )
    count_map = {row["split"]: int(row["rows"]) for row in counts_df.iter_rows(named=True)}
    counts = {split: count_map.get(split, 0) for split in ["train", "val", "test"]}

    for split in ["train", "val", "test"]:
        out_path = task_dir / f"{split}.parquet"
        if out_path.exists() and not force:
            print(f"skip (exists): {out_path}")
            continue

        (
            task_lf
            .filter(pl.col("split") == split)
            .sink_parquet(str(out_path))
        )
        print(f"wrote {out_path} rows={counts[split]}")

    return counts


def summarize_task(task_lf: pl.LazyFrame, target_col: str) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []

    for split in ["train", "val", "test"]:
        split_lf = task_lf.filter(pl.col("split") == split)

        count = split_lf.select(pl.len().alias("rows")).collect(engine="streaming").item()
        if int(count) == 0:
            frames.append(
                pl.DataFrame(
                    {
                        "target": [target_col],
                        "split": [split],
                        "rows": [0],
                        "stations": [0],
                        "min_time": [None],
                        "max_time": [None],
                        "event_rate": [None],
                        "target_mean": [None],
                        "target_p95": [None],
                    }
                )
            )
            continue

        summary = (
            split_lf.select(
                pl.lit(target_col).alias("target"),
                pl.lit(split).alias("split"),
                pl.len().alias("rows"),
                pl.col("station").n_unique().alias("stations"),
                pl.col("timestamp").min().alias("min_time"),
                pl.col("timestamp").max().alias("max_time"),
                pl.col(target_col).cast(pl.Float64, strict=False).mean().alias("event_rate"),
                pl.col(target_col).cast(pl.Float64, strict=False).mean().alias("target_mean"),
                pl.col(target_col).cast(pl.Float64, strict=False).quantile(0.95).alias("target_p95"),
            )
            .collect(engine="streaming")
        )
        frames.append(summary)

    return pl.concat(frames, how="vertical")


def write_metadata(
    output_root: Path,
    *,
    feature_cols: list[str],
    dropped_non_feature_cols: list[str],
    target_cols: list[str],
    split_config: dict,
    summary_df: pl.DataFrame,
    split_validation_df: pl.DataFrame,
) -> None:
    meta_dir = output_root / "metadata"
    ensure_dir(meta_dir)

    (meta_dir / "feature_columns.json").write_text(
        json.dumps({"feature_columns": feature_cols}, indent=2, sort_keys=True)
    )
    (meta_dir / "dropped_non_feature_columns.json").write_text(
        json.dumps({"dropped_non_feature_columns": dropped_non_feature_cols}, indent=2, sort_keys=True)
    )
    (meta_dir / "target_columns.json").write_text(
        json.dumps({"classification_targets": target_cols}, indent=2, sort_keys=True)
    )
    (meta_dir / "excluded_feature_columns.json").write_text(
        json.dumps({"excluded_feature_columns": sorted(EXCLUDED_FEATURE_COLS)}, indent=2, sort_keys=True)
    )
    (meta_dir / "split_config.json").write_text(json.dumps(split_config, indent=2, sort_keys=True))
    summary_df.write_csv(meta_dir / "dataset_summary.csv")
    split_validation_df.write_csv(meta_dir / "split_validation_report.csv")


def main() -> None:
    args = parse_args()

    files = discover_feature_files(args.input_root, args.months)
    lf = scan_features(files)
    lf, split_config = assign_splits(lf, args)
    lf = lf.filter(pl.col("split").is_in(["train", "val", "test"]))

    assigned_rows = lf.select(pl.len().alias("rows")).collect(engine="streaming").item()
    if int(assigned_rows) == 0:
        raise ValueError("No rows assigned to train/val/test splits. Check split configuration and data coverage.")

    feature_cols, dropped_cols = choose_feature_columns(lf)
    schema_names = set(lf.collect_schema().names())
    metadata_cols = [c for c in METADATA_COLS if c in schema_names]

    split_validation = validate_split_boundaries(lf)
    if "status" in split_validation.columns and split_validation.filter(pl.col("status") == "fail").height > 0:
        write_metadata(
            args.output_root,
            feature_cols=feature_cols,
            dropped_non_feature_cols=dropped_cols,
            target_cols=TARGET_COLS,
            split_config=split_config,
            summary_df=pl.DataFrame(),
            split_validation_df=split_validation,
        )
        raise ValueError("Split boundary validation failed. See split_validation_report.csv for details.")

    all_summaries: list[pl.DataFrame] = []
    row_count_report: dict[str, dict[str, int]] = {}

    for target_col in TARGET_COLS:
        task_lf = build_task_lazyframe(
            lf,
            target_col=target_col,
            feature_cols=feature_cols,
            metadata_cols=metadata_cols,
            drop_missing_target=args.drop_missing_targets,
        )

        counts = write_split_files(
            task_lf,
            args.output_root / target_col,
            args.force,
        )
        row_count_report[target_col] = counts
        all_summaries.append(summarize_task(task_lf, target_col))

    summary = pl.concat(all_summaries, how="vertical")

    write_metadata(
        args.output_root,
        feature_cols=feature_cols,
        dropped_non_feature_cols=dropped_cols,
        target_cols=TARGET_COLS,
        split_config=split_config,
        summary_df=summary,
        split_validation_df=split_validation,
    )

    print("\nDone.")
    print(f"Targets written: {len(TARGET_COLS)}")
    print(f"Feature columns kept: {len(feature_cols)}")
    print(f"Excluded raw SuperMAG disturbance features: {sorted(EXCLUDED_FEATURE_COLS)}")
    print(f"Metadata written to: {args.output_root / 'metadata'}")
    print("Rows by split:")
    for target_col, counts in row_count_report.items():
        print(f"  - {target_col}: {counts}")


if __name__ == "__main__":
    main()


