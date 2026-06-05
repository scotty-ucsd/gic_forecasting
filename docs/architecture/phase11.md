src/evaluation/evaluate_model_target_station.py
"""
Phase 11.1 - Evaluation: Station-Level Model Evaluation
--------------------------------------------------------
This is the first of two Phase 11 evaluation scripts and must be run
before Phase 11.2 (plot_target_station_results.py). It reads saved
prediction Parquet files from all Phase 10 model scripts and computes
14 classification metrics for every model x target x station x split
combination. Results are written as a long-format CSV, a wide-format
pivot CSV, and a run configuration JSON. Phase 11.2 reads the long CSV
to produce visual summaries.

This script uses Polars for Parquet I/O and pandas/numpy/sklearn for
all metric computation. No model weights or training artifacts are
loaded; only the prediction Parquet files written by Phase 10 are
consumed.

Purpose
-------
Pooled model metrics (written per-split by each Phase 10 script) do not
reveal station-level variation. This evaluator groups predictions by
station for each model/target/split combination, enabling comparison of
forecast skill across the monitoring network and identification of which
targets are forecastable at which stations.

The six benchmark models evaluated are:
- climatology          (Phase 10.1; val and test splits only)
- persistence          (Phase 10.2; train, val, test)
- logistic_regression  (Phase 10.3; train, val, test)
- lightgbm_main        (Phase 10.4.1; train, val, test)
- lightgbm_tuned_simple (Phase 10.4.2; train, val, test)
- lstm                 (Phase 10.5; train, val, test)

Note: climatology is registered with splits=[val, test] only, reflecting
that the climatology model does not produce a train split prediction file.

MODEL_SPECS registry
---------------------
MODEL_SPECS is the authoritative model registry for this evaluator.
Each entry defines pred_root, splits, and prediction_pattern. All
models use the same pattern:
  {target}/predictions/{split}_predictions.parquet
relative to pred_root. To add a new model, add an entry to MODEL_SPECS.

Cross-model schema normalization
----------------------------------
Prediction Parquet schemas differ across Phase 10 scripts:
- Some store the true label as "y_true" (persistence, climatology).
- Some store it under the target column name (logistic_regression,
  LightGBM, LSTM).
normalize_prediction_columns() handles this: if "y_true" is absent but
the target column name is present, it is renamed to "y_true". If
neither is present, a ValueError is raised with the file path.

Threshold inference
--------------------
The applied threshold is not stored as a row-level column in all
prediction files. infer_threshold() reconstructs it from y_prob and
y_pred: it finds the minimum probability among predicted-positive rows
(lower bound) and maximum probability among predicted-negative rows
(upper bound). If the bounds don't overlap, the midpoint is used.
Edge cases (all positive, all negative, fully separated) are handled
explicitly and clipped to [0.0, 1.0].

Metrics computed per group (14 total)
---------------------------------------
rows, n_positive, event_rate, selected_threshold,
precision, recall, f2, pr_auc (NaN if only one class present),
brier, hss (Heidke Skill Score), tn, fp, fn, tp.

Note: f1, roc_auc, log_loss, and balanced_accuracy are not computed at
the station level. Station groups often have only one class, making
roc_auc undefined; the set is intentionally conservative.

Skip behavior
--------------
Soft informational skip: if --output-root exists and --force is not set,
"[info] output exists" is printed but ensure_dir() is called and all
outputs are written/overwritten. Missing prediction files print a
[warn] message and are recorded in run_config.json under
missing_prediction_files; evaluation continues with remaining files.

Pivot format
-------------
build_pivot() produces a wide-format CSV with columns named:
  {metric}__{model}__{split}
(double underscore delimiter). With 14 metrics x 6 models x up to
3 splits this can produce up to 252 value columns plus target and
station index columns.

WORK FLOW
---------
Step 1: Parse CLI args. Resolve model, target, and split lists.
        Create output directory.
Step 2: For each model in models:
  For each target in targets:
    For each split in model's registered splits (intersected with
    --splits if provided):
      - Resolve prediction Parquet path from MODEL_SPECS.
      - If path missing: warn, record in missing_files, continue.
      - Read Parquet via pl.read_parquet().
      - normalize_prediction_columns(): alias target col -> y_true.
      - evaluate_prediction_file(): groupby station, apply
        --min-rows-per-group and --min-positive-events filters,
        compute 14 metrics per station via compute_group_metrics().
      - Extend all_rows list.
Step 3: Build long DataFrame from all_rows. Sort by
        [model, target, split, station].
Step 4: Build pivot via build_pivot().
Step 5: Write target_station_metrics_long.csv,
        target_station_metrics_pivot.csv, run_config.json.

INPUT DATA
----------
- data/reports/models/{model}/
    {target}/predictions/{split}_predictions.parquet
  Written by Phase 10 scripts (10.1 through 10.5). Each file must
  contain: station, y_prob, y_pred, and either y_true or the target
  column name.
- CLI optional: --output-root          (default: data/reports/evaluation/
                                         target_station)
               --models                (default: all 6 models)
               --targets               (default: all 6 targets)
               --splits                (default: per-model registered list)
               --min-rows-per-group    (default: 1)
               --min-positive-events   (default: 0)
               --force                 (soft skip if not set)

OUTPUT DATA
-----------
- data/reports/evaluation/target_station/target_station_metrics_long.csv
  Long format: one row per model x target x split x station.
  Columns: model, target, split, station, rows, n_positive, event_rate,
  selected_threshold, precision, recall, f2, pr_auc, brier, hss,
  tn, fp, fn, tp.
- data/reports/evaluation/target_station/target_station_metrics_pivot.csv
  Wide format: one row per target x station; columns are
  {metric}__{model}__{split}.
- data/reports/evaluation/target_station/run_config.json
  Records evaluated models, targets, requested splits, filter
  thresholds, missing files, and full MODEL_SPECS snapshot.

NEXT PIPELINE PHASE
-------------------
- Phase 11.1 depends on all Phase 10 model scripts completing first.
- Phase 11.2 (plot_target_station_results.py) reads the long CSV
  written here and must be run after this script.
"""

---

src/evaluation/plot_target_station_results.py
"""
Phase 11.2 - Evaluation: Station-Level Results Visualization
-------------------------------------------------------------
This is the second of two Phase 11 evaluation scripts and must be run
after Phase 11.1 (evaluate_model_target_station.py). It reads the
long-format station-level evaluation CSV produced by Phase 11.1 and
generates PNG heatmaps, difference heatmaps, and a grouped bar chart
for the selected split and metric. All plots are written to a plots/
subdirectory under the Phase 11.1 output root. This script uses only
pandas and Plotly; no model or training artifacts are loaded.

Purpose
-------
The Phase 11.1 long CSV is complete but not suited for rapid pattern
recognition. This script makes it easy to spot which targets are
forecastable at which stations, where learned models beat climatology,
whether gains are broadly distributed or concentrated at a few stations,
and how persistence compares to learned models.

The six benchmark models (in display order) are:
- climatology, persistence, logistic_regression,
  lightgbm_main, lightgbm_tuned_simple, lstm

Supported metrics: f2, precision, recall, pr_auc, brier, hss.

Plot outputs (default args: 6 models, metric=f2, split=test)
--------------------------------------------------------------
Per-model heatmaps (one per model):
  Target x station grid colored by --metric.
  Color scale: Viridis for f2/precision/recall/pr_auc/brier;
  RdBu (diverging, centered at 0) for hss.
  Color limits set from p5-p95 of actual values (not full 0-1 range)
  to improve visual contrast. Written as:
    {model}_{split}_{metric}_heatmap.png

Difference heatmaps (produced conditionally):
  LightGBM minus Climatology: produced when both lightgbm_main and
  climatology are in --models. RdBu scale, symmetric at 0,
  ±p95(abs(diff)). Written as:
    lightgbm_minus_climatology_{split}_{metric}_heatmap.png

  LightGBM Tuned minus LightGBM: produced when both
  lightgbm_tuned_simple and lightgbm_main are in --models. Same
  color logic. Written as:
    lightgbm_tuned_minus_main_{split}_{metric}_heatmap.png

Mean station-level bar chart (one per run):
  Grouped bar chart: x=target, color=model, y=mean(metric) across
  stations. Written as:
    mean_station_{metric}_by_target_{split}.png

All PNGs are rendered at width=1400, height=700, scale=2 (2800x1400
effective resolution) using Plotly's kaleido backend.

CSV outputs (one per plot, written alongside PNGs):
  filtered_metrics_{split}.csv            (post-filter long metrics)
  {model}_{split}_{metric}_heatmap_source.csv  (per-model subset)
  lightgbm_minus_climatology_{split}_{metric}.csv
  lightgbm_tuned_minus_main_{split}_{metric}.csv
  mean_station_{metric}_by_target_{split}.csv

Skip behavior
--------------
Soft informational only: if plots/ directory exists and --force is not
set, "[info] output exists" is printed, but ensure_dir() is called and
all plots are regenerated. There is no true skip in this script.

WORK FLOW
---------
Step 1: Parse CLI args. Resolve output directory.
Step 2: Read target_station_metrics_long.csv via read_long_metrics().
Step 3: normalize_order(): apply Categorical ordering for target
        (TARGET_ORDER), station (alphabetical), and model (MODEL_ORDER
        with remaining models appended sorted).
Step 4: Filter to --split and --models. Raise ValueError if empty.
        Write filtered_metrics_{split}.csv.
Step 5: For each model in --models:
        - Filter to model rows.
        - Write heatmap source CSV.
        - skill_color_limits(): compute zmin/zmax from p5/p95 with
          metric-specific logic (brier: no clamping; hss: symmetric;
          others: clamped [0,1]).
        - make_heatmap(): pivot to target x station, render Plotly
          imshow, save PNG.
Step 6: If lightgbm_main and climatology both present:
        build_difference_df() -> make_heatmap() with RdBu scale.
        Write difference CSV and PNG.
Step 7: If lightgbm_tuned_simple and lightgbm_main both present:
        Same as Step 6 for tuned vs untuned difference.
Step 8: make_mean_bar(): group by [target, model], mean metric,
        render grouped bar chart. Write summary CSV and PNG.

INPUT DATA
----------
- data/reports/evaluation/target_station/target_station_metrics_long.csv
  Written by Phase 11.1. Required. The pivoted CSV and run_config.json
  in the same directory are not read by this script.
- CLI optional: --eval-root  (default: data/reports/evaluation/
                               target_station)
               --split       (default: test)
               --models      (default: all 6 models)
               --metric      (default: f2; choices: f2, precision,
                              recall, pr_auc, brier, hss)
               --force       (soft skip if not set)

OUTPUT DATA
-----------
All written to data/reports/evaluation/target_station/plots/.

Per-model (2 files each x 6 models = 12):
  {model}_{split}_{metric}_heatmap.png
  {model}_{split}_{metric}_heatmap_source.csv

Difference heatmaps (2 files each x up to 2 pairs = up to 8):
  lightgbm_minus_climatology_{split}_{metric}_heatmap.png
  lightgbm_minus_climatology_{split}_{metric}.csv
  lightgbm_tuned_minus_main_{split}_{metric}_heatmap.png
  lightgbm_tuned_minus_main_{split}_{metric}.csv

Summary (3 files):
  mean_station_{metric}_by_target_{split}.png
  mean_station_{metric}_by_target_{split}.csv
  filtered_metrics_{split}.csv

Total default output: 19 files (12 + 4 + 3).

TYPICAL USAGE
-------------
Default run (test split, f2, all models):
  uv run src/evaluation/plot_target_station_results.py --force

Restrict to test split and subset of models:
  uv run src/evaluation/plot_target_station_results.py \
    --split test \
    --models climatology persistence lightgbm_main \
             lightgbm_tuned_simple lstm \
    --force

NEXT PIPELINE PHASE
-------------------
- Phase 11.2 depends on Phase 11.1 completing first.
- Phase 11.2 is the final script in the documented pipeline.
- PNG and CSV outputs from plots/ are used directly for paper figures
  and supplementary reporting.
"""

---

