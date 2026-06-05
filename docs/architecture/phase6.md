src/targets/build_station_thresholds.py
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

---

src/targets/attach_station_targets.py
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

---

