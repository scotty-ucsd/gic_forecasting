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

