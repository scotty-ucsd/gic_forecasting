src/models/baselines/persistence.py
"""
Phase 10.2 - Models/Baselines: Strict Observation-Persistence Baseline
-----------------------------------------------------------------------
This is the second Phase 10 baseline script in src/models/baselines/.
Unlike Phase 10.1 (climatology.py), it reads from the Phase 7.1
standardized feature Parquet files rather than the Phase 8.1 ML split
files, because it requires the raw supermag_dbn_nt and supermag_dbe_nt
columns that were intentionally excluded from the ML dataset as
leakage-adjacent. It re-applies the year-based split assignment
internally using data/ml/standardized/metadata/split_config.json.
This script uses pandas throughout; no Polars dependency.

Purpose
-------
The persistence baseline tests whether learned models provide skill
beyond the simplest causal assumption in geomagnetism: the current
local ground magnetic state is predictive of the near-future state.
If the horizontal magnetic perturbation at issue time t is already
above the event threshold, the model predicts an event will occur
within the forecast horizon.

Scientific definition
---------------------
Current observed disturbance proxy:
  B_H(t) = sqrt(supermag_dbn_nt(t)^2 + supermag_dbe_nt(t)^2)

Per-target current-event threshold (train-derived only):
  tau_train = quantile(B_H on train split, p)
  where p = 0.95 for p95 targets, p = 0.99 for p99 targets
  (parsed from the target column name by parse_target_spec())

Strict persistence rule:
  y_pred(t) = 1 if B_H(t) >= tau_train else 0

For evaluator compatibility, y_prob is stored as a binary 0/1 float
equal to y_pred. The fixed prediction threshold is 0.5. PR-AUC and
ROC-AUC are computed but are not meaningful as probability-calibration
metrics since y_prob carries no calibrated probability.

Important properties
--------------------
- No fitted classifier is trained.
- No probability calibration is applied.
- No validation threshold tuning is applied.
- tau_train is fixed from training data only and applied unchanged
  to train, val, and test.
- supermag_dbn_nt and supermag_dbe_nt are read directly from the
  Phase 7.1 standardized files; they are not present in the Phase 8.1
  ML split files (excluded as leakage-adjacent for learned models).
- evaluate_target() raises FileExistsError if the output directory
  already exists and --force is not set.

WORK FLOW
---------
Step 1: Parse CLI arguments - standardized root, split config path,
        output root, optional --glob-pattern, optional --targets,
        optional --station filter, optional --years filter, --force.
Step 2: Load and validate split_config.json via load_split_config():
        must have strategy="year" and non-empty year_assignments.
Step 3: Load all monthly station_time_features_*.parquet files via
        load_standardized_frames(): read each file with
        pd.read_parquet(), apply optional station and year filters,
        assign split column from year map, drop unassigned rows,
        sort by (timestamp, station).
Step 4: For each target in the selected target list:
  Step 4a: Raise FileExistsError if output dir exists and not --force.
  Step 4b: Split the loaded DataFrame into train, val, test subsets.
           Raise ValueError if any split is empty.
  Step 4c: Compute B_H(t) = sqrt(dbn^2 + dbe^2) for all three splits
           via build_horizontal_perturbation_proxy().
  Step 4d: Parse percentile and horizon_minutes from target column
           name via parse_target_spec().
  Step 4e: Compute tau_train = np.quantile(finite_train_BH, percentile).
  Step 4f: Compute binary persistence outputs (0/1 float) for all
           three splits via build_strict_persistence_outputs().
  Step 4g: Compute 18 classification metrics on train, val, test at
           fixed_threshold=0.5.
  Step 4h: Write train/val/test_predictions.parquet (includes B_H
           source values and current_event_threshold column).
  Step 4i: Write fixed_threshold_summary.csv, metrics_summary.csv,
           station_split_summary.csv (per-station per-split
           aggregation), and model_config.json.
Step 5: Concatenate per-target result dicts into all_targets_summary.csv.
        Write run_config.json.

Metrics reported (18 per split)
---------------------------------
threshold, rows, event_rate, pred_rate, precision, recall, f1, f2,
balanced_accuracy, pr_auc, roc_auc, brier, log_loss, hss
(Heidke Skill Score), tn, fp, fn, tp.

INPUT DATA
----------
- data/preprocessed/fused/standardized/{YYYY}/station_time_features_{YYYYMM}.parquet
  Phase 7.1 output. Required columns: timestamp, station, year,
  supermag_dbn_nt, supermag_dbe_nt, and all six target_geq_* columns.
- data/ml/standardized/metadata/split_config.json
  Phase 8.1 output. Must have strategy="year" and year_assignments.
- CLI optional: --standardized-root  (default: data/preprocessed/fused/standardized)
               --split-config       (default: data/ml/standardized/
                                     metadata/split_config.json)
               --output-root        (default: data/reports/models/persistence)
               --glob-pattern       (default: */station_time_features_*.parquet)
               --targets            (default: all six targets)
               --station            (default: None, all stations)
               --years              (default: None, all years)
               --force              (overwrite; otherwise raises FileExistsError)

OUTPUT DATA
-----------
Per-target (7 files each, 42 total):
- predictions/train_predictions.parquet
- predictions/val_predictions.parquet
- predictions/test_predictions.parquet
  (all include: ID cols, target col, supermag_horizontal_perturbation_proxy,
  current_event_threshold, y_prob, y_pred)
- metrics/fixed_threshold_summary.csv
- metrics/metrics_summary.csv
- metrics/station_split_summary.csv
- model_config.json
All written under data/reports/models/persistence/{target_col}/.

Run-level (2 files):
- data/reports/models/persistence/all_targets_summary.csv
- data/reports/models/persistence/run_config.json

NEXT PIPELINE PHASE
-------------------
- Phase 10.2 depends on Phase 7.1 (standardized feature files) and
  Phase 8.1 (split_config.json) completing first.
- Phase 10.2 runs independently of Phase 10.1 (climatology.py).
- Both Phase 10.1 and 10.2 outputs feed into Phase 11 (src/evaluate/)
  for comparison against learned model results.
"""

---

src/models/baselines/climatology.py
"""
Phase 10.1 - Models/Baselines: Climatology Constant-Rate Baseline
------------------------------------------------------------------
This is the first Phase 10 baseline script and the simpler of the
two baselines in src/models/baselines/. It must run after Phase 8.1
(src/dataset/build_ml_dataset.py) has written the per-target
train/val/test split Parquet files. No feature columns are used; only
the target column itself is read. This script uses Polars for file
I/O and pandas/numpy/sklearn for metric computation.

Purpose
-------
The climatology baseline is the simplest conceivable forecasting
strategy: assign every future row the same constant probability equal
to the observed event rate in the training split. This establishes
the theoretical lower bound of skill that any learned model must
exceed to demonstrate genuine predictive ability beyond the marginal
class frequency alone.

Scientific definition
---------------------
For each target, the climatological probability is:

  p_clim = mean(target_col on train split)

This scalar is broadcast to every row in the val and test splits:

  y_prob[i] = p_clim  for all i

A decision threshold is selected by sweeping THRESHOLD_GRID
(0.05, 0.10, ..., 0.95 in steps of 0.05; 19 values) evaluated on
the val split only, maximizing F2 as the primary metric (then PR-AUC,
then recall as tiebreakers). The selected threshold is applied
unchanged to test. Since y_prob is constant, all threshold choices
that lie below p_clim predict all positives; all choices above
predict all negatives. The F2-optimizing threshold therefore reduces
to a single meaningful operating point.

Important properties
--------------------
- No feature columns are read or used.
- No fitted classifier is trained.
- No per-station, per-month, or temporal variation in predicted
  probability.
- Train predictions are not written; only val and test prediction
  Parquet files are produced per target.
- Since y_prob is a constant scalar, ROC-AUC degrades to ~0.5 and
  is not a meaningful discriminability metric for this baseline.
- The --force flag is accepted but evaluate_target() always
  overwrites existing outputs without a FileExistsError guard.

WORK FLOW
---------
Step 1: Parse CLI arguments - data root, output root, optional
        --targets subset, --force flag.
Step 2: For each target in the selected target list:
  Step 2a: Read train/val/test.parquet from
           data/ml/standardized/{target_col}/ via pl.read_parquet()
           then .to_pandas().
  Step 2b: Compute train_prob = mean(train[target_col]).
  Step 2c: Broadcast train_prob as np.full(len(val/test), train_prob)
           to produce val_prob and test_prob arrays.
  Step 2d: Select best threshold via select_threshold() sweeping
           THRESHOLD_GRID on val, maximizing F2 then PR-AUC then
           recall.
  Step 2e: Compute 18 classification metrics on val and test at the
           selected threshold.
  Step 2f: Write val_predictions.parquet and test_predictions.parquet
           to data/reports/models/climatology/{target_col}/predictions/.
  Step 2g: Write validation_threshold_grid.csv (all 19 threshold
           rows), metrics_summary.csv (val + test), and
           model_config.json to
           data/reports/models/climatology/{target_col}/metrics/.
Step 3: Concatenate per-target result dicts into all_targets_summary.csv.
        Write run_config.json (model name, data/output roots, targets,
        threshold grid values).

Metrics reported (18 per split)
---------------------------------
threshold, rows, event_rate, pred_rate, precision, recall, f1, f2,
balanced_accuracy, pr_auc, roc_auc, brier, log_loss, hss
(Heidke Skill Score), tn, fp, fn, tp.

INPUT DATA
----------
- data/ml/standardized/{target_col}/train.parquet  (event rate source)
- data/ml/standardized/{target_col}/val.parquet
- data/ml/standardized/{target_col}/test.parquet
  Phase 8.1 output. Only the target column is used.
- CLI optional: --data-root    (default: data/ml/standardized)
               --output-root  (default: data/reports/models/climatology)
               --targets      (default: all six targets)
               --force        (accepted; outputs always overwritten)

OUTPUT DATA
-----------
Per-target (5 files each, 30 total):
- data/reports/models/climatology/{target_col}/predictions/val_predictions.parquet
- data/reports/models/climatology/{target_col}/predictions/test_predictions.parquet
- data/reports/models/climatology/{target_col}/metrics/validation_threshold_grid.csv
- data/reports/models/climatology/{target_col}/metrics/metrics_summary.csv
- data/reports/models/climatology/{target_col}/model_config.json

Run-level (2 files):
- data/reports/models/climatology/all_targets_summary.csv
- data/reports/models/climatology/run_config.json

NEXT PIPELINE PHASE
-------------------
- Phase 10.1 depends only on Phase 8.1 completing first.
- Phase 10.2 (src/models/baselines/persistence.py) is the second
  baseline and runs independently of Phase 10.1.
- Both baseline outputs feed into Phase 11 (src/evaluate/) for
  comparison against learned model results.
"""

---

src/models/classification/logistic_regression.py
"""
Phase 10.3 - Models/Classification: Logistic Regression Classifier
-------------------------------------------------------------------
This is the first classification model script in src/models/classification/
and the third Phase 10 script overall. It reads the Phase 8.1 per-target
train/val/test Parquet split files from data/ml/standardized/, trains a
binary logistic regression classifier for each of the six canonical targets,
selects the best inverse regularization strength C via validation performance,
selects a probability threshold on the validation set, and writes metrics,
predictions, coefficients, hyperparameter trials, and a pickled sklearn
Pipeline per target. This script uses Polars for file I/O and
pandas/numpy/sklearn for all model fitting and metric computation.

Purpose
-------
Logistic regression is the first learned model in the pipeline and the
primary skill check above the climatology baseline. Because it is linear,
interpretable through coefficients, and fast to train across a C grid, it
establishes whether the cleaned standardized feature stack carries genuine
predictive signal. Its coefficient output also serves as a complementary
signal alongside the Phase 9.1 importance scores from LightGBM.

The six binary classification targets are:
- target_geq_p95_30m, target_geq_p99_30m   (30-minute horizon)
- target_geq_p95_60m, target_geq_p99_60m   (60-minute horizon)
- target_geq_p95_120m, target_geq_p99_120m (120-minute horizon)

sklearn Pipeline
-----------------
A single sklearn Pipeline is built per (target, C) trial:
  SimpleImputer(strategy="median")
  -> StandardScaler()
  -> LogisticRegression(C=C, solver=solver, class_weight=class_weight,
                        max_iter=1000, random_state=42, penalty="l2")

The imputer and scaler are fitted on train only and reused for val and
test inference, ensuring no leakage from val or test into preprocessing.
The entire fitted Pipeline is serialized as a single pickle artifact.

C grid search and threshold selection
---------------------------------------
fit_model_grid() iterates over --c-grid (default: [0.01, 0.1, 1.0, 3.0]).
For each C, the pipeline is fitted on train, probabilities are produced
on val, and a threshold is selected by sweeping THRESHOLD_GRID
(0.05 to 0.95 in steps of 0.05; 19 values) to maximize --threshold-metric
(default F2), with PR-AUC then recall as tiebreakers.

Best C is selected by a 3-tuple score: (threshold_metric_val, pr_auc_val,
hss_val), with NaN replaced by -inf. The best fitted pipeline is retained;
all other C trial pipelines are discarded. The threshold selected during
best-C val evaluation is applied unchanged to train, val, and test for
all final reported metrics.

Feature column sources (mutually exclusive, checked in order)
--------------------------------------------------------------
1. --reduced-features-json: reads "feature_columns" or "features" key
   (or bare list). --top-k-features cap applied if set.
2. --reduced-features-csv: reads "feature" column or first column.
   --top-k-features cap applied if set.
3. Neither supplied (default): uses full feature_columns.json from
   data/ml/standardized/metadata/.

Deduplication preserves order. resolve_feature_columns() then filters
to columns actually present in the training split schema.

class_weight handling
----------------------
--class-weight default is "balanced" (sklearn scales loss by inverse
class frequency). Passing "none", "None", or empty string resolves to
sklearn's None (no weighting).

Skip behavior
--------------
If the per-target output directory already exists and --force is not
set, a "[info] output exists" message is printed but processing
continues and outputs are overwritten. This is a soft informational
warning, not a hard stop.

WORK FLOW
---------
Step 1: Parse CLI arguments. Load full feature_columns.json and
        optionally a reduced feature list. Resolve target list.
Step 2: For each target in the selected target list:
  Step 2a: Read train/val/test.parquet via pl.read_parquet(), optionally
           capped by --max-train-rows, --max-val-rows, --max-test-rows
           (head-based, preserving parquet row order).
  Step 2b: Resolve feature columns to those present in train schema.
  Step 2c: Convert to pandas X/y/meta via to_pandas_xy(): select ID +
           target + feature cols, coerce target to int, fill null -> 0.
  Step 2d: Validate binary labels (0/1 only) on all three splits.
  Step 2e: Run C grid search via fit_model_grid(): for each C, fit
           pipeline on train, score on val, select best threshold, record
           trial row. Select best C by 3-tuple score.
  Step 2f: Produce train/val/test probabilities from the best pipeline.
  Step 2g: Compute 18 classification metrics on all three splits.
  Step 2h: Write train/val/test_predictions.parquet (ID cols, y_true,
           y_prob, y_pred).
  Step 2i: Write coefficients.csv (sorted by abs_coefficient desc),
           hyperparameter_trials.csv, validation_threshold_grid.csv,
           metrics_summary.csv.
  Step 2j: Pickle the best pipeline to artifacts/model.pkl.
           Write artifacts/model_config.json (includes features list,
           selected C, selected threshold, all split metrics, top 25
           coefficients).
Step 3: Concatenate per-target summary rows into summary_metrics.csv.
        Write run_config.json.

Metrics reported (18 per split)
---------------------------------
threshold, rows, event_rate, pred_rate, precision, recall, f1, f2,
balanced_accuracy, pr_auc, roc_auc, brier, log_loss, hss
(Heidke Skill Score), tn, fp, fn, tp.

INPUT DATA
----------
- data/ml/standardized/{target_col}/train.parquet
- data/ml/standardized/{target_col}/val.parquet
- data/ml/standardized/{target_col}/test.parquet
  Phase 8.1 output. One triplet per target.
- data/ml/standardized/metadata/feature_columns.json
  Phase 8.1 output. Used when no reduced feature source is supplied.
- CLI optional: --data-root             (default: data/ml/standardized)
               --output-root           (default: data/reports/models/
                                        logistic_regression)
               --targets               (default: all six targets)
               --feature-columns-json  (default: metadata/feature_columns.json)
               --reduced-features-json (default: None)
               --reduced-features-csv  (default: None)
               --top-k-features        (default: None)
               --max-train-rows        (default: 2,000,000)
               --max-val-rows          (default: None)
               --max-test-rows         (default: None)
               --c-grid                (default: [0.01, 0.1, 1.0, 3.0])
               --threshold-metric      (default: f2; choices: f1, f2,
                                        balanced_accuracy)
               --solver                (default: lbfgs)
               --class-weight          (default: balanced)
               --force                 (soft skip warning if not set)

OUTPUT DATA
-----------
Per-target (9 files each, 54 total):
- artifacts/model.pkl              (pickled full sklearn Pipeline)
- artifacts/model_config.json      (config, features, metrics, top-25 coefs)
- metrics/coefficients.csv         (feature, coefficient, abs_coefficient,
                                    direction; sorted by abs desc)
- metrics/hyperparameter_trials.csv (one row per C trial)
- metrics/validation_threshold_grid.csv (19 threshold rows from best-C val)
- metrics/metrics_summary.csv      (train + val + test rows, 18 metrics each)
- predictions/train_predictions.parquet
- predictions/val_predictions.parquet
- predictions/test_predictions.parquet
All written under data/reports/models/logistic_regression/{target_col}/.

Run-level (2 files):
- data/reports/models/logistic_regression/summary_metrics.csv
- data/reports/models/logistic_regression/run_config.json

NEXT PIPELINE PHASE
-------------------
- Phase 10.3 depends on Phase 8.1 (src/dataset/build_ml_dataset.py)
  completing first.
- Phase 9.1 (src/features/evaluate_feature_importance.py) output
  (reduced_feature_candidates.csv) can optionally be supplied via
  --reduced-features-csv to train on the importance-selected feature
  subset.
- Additional classification model scripts in src/models/classification/
  follow as Phase 10.4 and beyond.
- Phase 11 (src/evaluate/) consumes the prediction Parquet files and
  metrics CSVs produced by all Phase 10 model scripts.
"""

---

src/models/classification/lightgbm_main.py
"""
Phase 10.4.2 - Models/Classification: LightGBM Random Search with Early Stopping
----------------------------------------------------------------------------------
This is the third classification model script in src/models/classification/
and the second of two LightGBM scripts in Phase 10.4. It extends Phase 10.4.1
(lightgbm_main.py) with random hyperparameter search and LightGBM early
stopping rather than a fixed grid. It infers feature columns autonomously
from the data schema without requiring a feature_columns.json dependency,
uses NaN-fill-with-zero preprocessing instead of median imputation, and
does not serialize the fitted model to disk. This script uses pandas
throughout; no Polars dependency.

Purpose
-------
Phase 10.4.1 evaluates three fixed LightGBM configurations. Phase 10.4.2
performs exploratory random search over a broader hyperparameter space with
up to 4000 estimators per trial and early stopping to prevent over-training.
The combination of random search and early stopping allows each trial to
find its own optimal depth without requiring a fixed n_estimators budget.
The best trial per target is selected by (val_f2, val_pr_auc, -val_brier).

The six binary classification targets are:
- target_geq_p95_30m, target_geq_p99_30m   (30-minute horizon)
- target_geq_p95_60m, target_geq_p99_60m   (60-minute horizon)
- target_geq_p95_120m, target_geq_p99_120m (120-minute horizon)

Random search space (9 hyperparameters)
-----------------------------------------
  learning_rate:       [0.01, 0.02, 0.03, 0.05, 0.07]
  num_leaves:          [15, 31, 63, 127]
  min_child_samples:   [20, 50, 100, 200]
  subsample:           [0.7, 0.85, 1.0]
  subsample_freq:      [0, 1]
  colsample_bytree:    [0.7, 0.85, 1.0]
  reg_alpha:           [0.0, 0.01, 0.1, 1.0]
  reg_lambda:          [0.0, 0.01, 0.1, 1.0]
  max_depth:           [-1, 6, 10, 14]

Each target uses a deterministic but target-specific seed:
  rng = np.random.default_rng(random_state + 1000 * target_index)
Default --trials = 20 per target.

Early stopping and model fitting
----------------------------------
Each trial fits LGBMClassifier with n_estimators=4000, objective="binary",
boosting_type="gbdt", n_jobs=-1, and the sampled hyperparameters. No
class_weight is applied (class_weight="balanced" is commented out).
LightGBM callbacks:
- lgb.early_stopping(stopping_rounds=100): halts training when val
  binary_logloss does not improve for 100 consecutive rounds.
- lgb.log_evaluation(period=50): prints val metrics every 50 trees.
best_iteration_ is recorded from the model after fitting.

Preprocessing
--------------
No median imputation. prepare_xy() coerces all feature columns to numeric
via pd.to_numeric(errors="coerce") then fills NaN with 0.0. LightGBM
handles the resulting numeric matrix directly.

Feature column inference
-------------------------
infer_feature_columns() scans the combined train+val+test DataFrame schema.
Keeps all numeric and boolean columns that are not in EXCLUDE_COLS
(timestamp, station, split, year, month, yyyymm) and not in TARGET_COLS.
No external feature_columns.json is required; feature selection is fully
data-driven. Columns are sorted alphabetically.

Filename resolution
--------------------
resolve_split_path() tries multiple candidate filenames per split from
DEFAULT_SPLIT_FILES before raising FileNotFoundError. This allows the
script to work with alternative naming conventions:
  train: [train.parquet, train_data.parquet, X_train.parquet]
  val:   [val.parquet, validation.parquet, val_data.parquet, X_val.parquet]
  test:  [test.parquet, test_data.parquet, X_test.parquet]

Model serialization
--------------------
The best model is NOT pickled. Only best_params.json is written per
target (no artifacts/ subdirectory, no model.pkl or imputer.pkl). This
is intentional for an exploratory tuning script; re-running with the
best parameters from best_params.json in Phase 10.4.1 produces a
serialized artifact.

Skip behavior: if --output-root exists and --force is not set, a message
is printed but all targets are still processed and outputs overwritten.

WORK FLOW
---------
Step 1: Parse CLI arguments - ml root, output root, optional --targets,
        --trials (default 20), --random-state (default 42), --force.
Step 2: For each target in the selected target list:
  Step 2a: Load train/val/test Parquet files via pd.read_parquet() using
           multi-candidate filename resolution.
  Step 2b: Infer feature columns from combined schema. Prepare X/y arrays
           with numeric coercion and zero-fill.
  Step 2c: Run random search loop for n_trials iterations:
    - Sample hyperparameters via sample_params(rng).
    - Fit LGBMClassifier with early stopping on val binary_logloss.
    - Compute val probabilities, select threshold (F2 primary, then
      PR-AUC, then recall), compute val metrics.
    - Record trial row. Update best if (val_f2, val_pr_auc, -val_brier)
      improves.
  Step 2d: Write tuning_trials.csv (sorted by val_f2 desc).
  Step 2e: Generate train/val/test probabilities from best model.
           Compute 18 classification metrics on all three splits.
  Step 2f: Write metrics_summary.csv, validation_threshold_grid.csv.
  Step 2g: Write train/val/test_predictions.parquet (includes
           selected_threshold column).
  Step 2h: Write best_params.json (best params, best_iteration,
           threshold, all split metrics).
Step 3: Concatenate per-target result dicts into all_targets_summary.csv.
        Write run_config.json (includes full search_space and
        feature_count_by_target).

Metrics reported (18 per split)
---------------------------------
threshold, rows, event_rate, pred_rate, precision, recall, f1, f2,
balanced_accuracy, pr_auc, roc_auc, brier, log_loss, hss
(Heidke Skill Score), tn, fp, fn, tp.

INPUT DATA
----------
- data/ml/standardized/{target_col}/train.parquet (or alternative names)
- data/ml/standardized/{target_col}/val.parquet
- data/ml/standardized/{target_col}/test.parquet
  Phase 8.1 output. Feature columns inferred from schema; no
  feature_columns.json required.
- CLI optional: --ml-root      (default: data/ml/standardized)
               --output-root  (default: data/reports/models/
                               lightgbm_tuned_simple)
               --targets      (default: all six targets)
               --trials       (default: 20)
               --random-state (default: 42)
               --force        (soft skip at root level if not set)

OUTPUT DATA
-----------
Per-target (7 files each, 42 total):
- {target_col}/best_params.json      (best params, iteration, threshold,
                                      all split metrics)
- {target_col}/tuning_trials.csv     (one row per trial, sorted by val_f2)
- {target_col}/metrics/metrics_summary.csv
- {target_col}/metrics/validation_threshold_grid.csv
- {target_col}/predictions/train_predictions.parquet
- {target_col}/predictions/val_predictions.parquet
- {target_col}/predictions/test_predictions.parquet
  (prediction files include selected_threshold column)
All written under data/reports/models/lightgbm_tuned_simple/.
No model.pkl or artifacts/ directory is produced.

Run-level (2 files):
- data/reports/models/lightgbm_tuned_simple/all_targets_summary.csv
- data/reports/models/lightgbm_tuned_simple/run_config.json

NEXT PIPELINE PHASE
-------------------
- Phase 10.4.2 depends on Phase 8.1 completing first.
- best_params.json outputs from this script can be used to configure
  a production training run in Phase 10.4.1 with a serialized artifact.
- Phase 11 (src/evaluate/) consumes prediction Parquet files and metrics
  CSVs from all Phase 10 model scripts.
"""

---

src/models/classification/lightgbm_tuned_simple.py
"""
Phase 10.4.2 - Models/Classification: LightGBM Random Search with Early Stopping
----------------------------------------------------------------------------------
This is the third classification model script in src/models/classification/
and the second of two LightGBM scripts in Phase 10.4. It extends Phase 10.4.1
(lightgbm_main.py) with random hyperparameter search and LightGBM early
stopping rather than a fixed grid. It infers feature columns autonomously
from the data schema without requiring a feature_columns.json dependency,
uses NaN-fill-with-zero preprocessing instead of median imputation, and
does not serialize the fitted model to disk. This script uses pandas
throughout; no Polars dependency.

Purpose
-------
Phase 10.4.1 evaluates three fixed LightGBM configurations. Phase 10.4.2
performs exploratory random search over a broader hyperparameter space with
up to 4000 estimators per trial and early stopping to prevent over-training.
The combination of random search and early stopping allows each trial to
find its own optimal depth without requiring a fixed n_estimators budget.
The best trial per target is selected by (val_f2, val_pr_auc, -val_brier).

The six binary classification targets are:
- target_geq_p95_30m, target_geq_p99_30m   (30-minute horizon)
- target_geq_p95_60m, target_geq_p99_60m   (60-minute horizon)
- target_geq_p95_120m, target_geq_p99_120m (120-minute horizon)

Random search space (9 hyperparameters)
-----------------------------------------
  learning_rate:       [0.01, 0.02, 0.03, 0.05, 0.07]
  num_leaves:          [15, 31, 63, 127]
  min_child_samples:   [20, 50, 100, 200]
  subsample:           [0.7, 0.85, 1.0]
  subsample_freq:      [0, 1]
  colsample_bytree:    [0.7, 0.85, 1.0]
  reg_alpha:           [0.0, 0.01, 0.1, 1.0]
  reg_lambda:          [0.0, 0.01, 0.1, 1.0]
  max_depth:           [-1, 6, 10, 14]

Each target uses a deterministic but target-specific seed:
  rng = np.random.default_rng(random_state + 1000 * target_index)
Default --trials = 20 per target.

Early stopping and model fitting
----------------------------------
Each trial fits LGBMClassifier with n_estimators=4000, objective="binary",
boosting_type="gbdt", n_jobs=-1, and the sampled hyperparameters. No
class_weight is applied (class_weight="balanced" is commented out).
LightGBM callbacks:
- lgb.early_stopping(stopping_rounds=100): halts training when val
  binary_logloss does not improve for 100 consecutive rounds.
- lgb.log_evaluation(period=50): prints val metrics every 50 trees.
best_iteration_ is recorded from the model after fitting.

Preprocessing
--------------
No median imputation. prepare_xy() coerces all feature columns to numeric
via pd.to_numeric(errors="coerce") then fills NaN with 0.0. LightGBM
handles the resulting numeric matrix directly.

Feature column inference
-------------------------
infer_feature_columns() scans the combined train+val+test DataFrame schema.
Keeps all numeric and boolean columns that are not in EXCLUDE_COLS
(timestamp, station, split, year, month, yyyymm) and not in TARGET_COLS.
No external feature_columns.json is required; feature selection is fully
data-driven. Columns are sorted alphabetically.

Filename resolution
--------------------
resolve_split_path() tries multiple candidate filenames per split from
DEFAULT_SPLIT_FILES before raising FileNotFoundError. This allows the
script to work with alternative naming conventions:
  train: [train.parquet, train_data.parquet, X_train.parquet]
  val:   [val.parquet, validation.parquet, val_data.parquet, X_val.parquet]
  test:  [test.parquet, test_data.parquet, X_test.parquet]

Model serialization
--------------------
The best model is NOT pickled. Only best_params.json is written per
target (no artifacts/ subdirectory, no model.pkl or imputer.pkl). This
is intentional for an exploratory tuning script; re-running with the
best parameters from best_params.json in Phase 10.4.1 produces a
serialized artifact.

Skip behavior: if --output-root exists and --force is not set, a message
is printed but all targets are still processed and outputs overwritten.

WORK FLOW
---------
Step 1: Parse CLI arguments - ml root, output root, optional --targets,
        --trials (default 20), --random-state (default 42), --force.
Step 2: For each target in the selected target list:
  Step 2a: Load train/val/test Parquet files via pd.read_parquet() using
           multi-candidate filename resolution.
  Step 2b: Infer feature columns from combined schema. Prepare X/y arrays
           with numeric coercion and zero-fill.
  Step 2c: Run random search loop for n_trials iterations:
    - Sample hyperparameters via sample_params(rng).
    - Fit LGBMClassifier with early stopping on val binary_logloss.
    - Compute val probabilities, select threshold (F2 primary, then
      PR-AUC, then recall), compute val metrics.
    - Record trial row. Update best if (val_f2, val_pr_auc, -val_brier)
      improves.
  Step 2d: Write tuning_trials.csv (sorted by val_f2 desc).
  Step 2e: Generate train/val/test probabilities from best model.
           Compute 18 classification metrics on all three splits.
  Step 2f: Write metrics_summary.csv, validation_threshold_grid.csv.
  Step 2g: Write train/val/test_predictions.parquet (includes
           selected_threshold column).
  Step 2h: Write best_params.json (best params, best_iteration,
           threshold, all split metrics).
Step 3: Concatenate per-target result dicts into all_targets_summary.csv.
        Write run_config.json (includes full search_space and
        feature_count_by_target).

Metrics reported (18 per split)
---------------------------------
threshold, rows, event_rate, pred_rate, precision, recall, f1, f2,
balanced_accuracy, pr_auc, roc_auc, brier, log_loss, hss
(Heidke Skill Score), tn, fp, fn, tp.

INPUT DATA
----------
- data/ml/standardized/{target_col}/train.parquet (or alternative names)
- data/ml/standardized/{target_col}/val.parquet
- data/ml/standardized/{target_col}/test.parquet
  Phase 8.1 output. Feature columns inferred from schema; no
  feature_columns.json required.
- CLI optional: --ml-root      (default: data/ml/standardized)
               --output-root  (default: data/reports/models/
                               lightgbm_tuned_simple)
               --targets      (default: all six targets)
               --trials       (default: 20)
               --random-state (default: 42)
               --force        (soft skip at root level if not set)

OUTPUT DATA
-----------
Per-target (7 files each, 42 total):
- {target_col}/best_params.json      (best params, iteration, threshold,
                                      all split metrics)
- {target_col}/tuning_trials.csv     (one row per trial, sorted by val_f2)
- {target_col}/metrics/metrics_summary.csv
- {target_col}/metrics/validation_threshold_grid.csv
- {target_col}/predictions/train_predictions.parquet
- {target_col}/predictions/val_predictions.parquet
- {target_col}/predictions/test_predictions.parquet
  (prediction files include selected_threshold column)
All written under data/reports/models/lightgbm_tuned_simple/.
No model.pkl or artifacts/ directory is produced.

Run-level (2 files):
- data/reports/models/lightgbm_tuned_simple/all_targets_summary.csv
- data/reports/models/lightgbm_tuned_simple/run_config.json

NEXT PIPELINE PHASE
-------------------
- Phase 10.4.2 depends on Phase 8.1 completing first.
- best_params.json outputs from this script can be used to configure
  a production training run in Phase 10.4.1 with a serialized artifact.
- Phase 11 (src/evaluate/) consumes prediction Parquet files and metrics
  CSVs from all Phase 10 model scripts.
"""

---

src/models/classification/lstm_classifier.py
"""
Phase 10.5 - Models/Classification: Compact PyTorch LSTM Classifier
--------------------------------------------------------------------
This is the fourth and final classification model script in
src/models/classification/ and the last script in Phase 10 overall.
It trains binary LSTM classifiers for the six canonical station-level
geomagnetic disturbance targets using a memory-efficient lazy-window
sequence construction strategy. It reads Phase 8.1 per-target
train/val/test Parquet split files, performs pure-numpy imputation and
standardization fit on train only, constructs per-station sequence
endpoint indices, and trains one LSTM per target with early stopping and
automatic mixed precision. This script uses Polars for file I/O, pandas
for metadata, numpy for arrays, PyTorch for model training, and sklearn
for metrics. No sklearn imputer or scaler objects are serialized.

Purpose
-------
The LSTM is the only sequence model in the Phase 10 benchmark. Unlike
the tabular models (Phases 10.3, 10.4.1, 10.4.2) which treat each row
as an independent observation, the LSTM receives a window of
lookback_steps consecutive timesteps sorted by (station, timestamp) and
predicts the event label at the final timestep. The design is intended
as a compact, stable first benchmark rather than a tuned production
model; see Future Work for planned extensions.

The six binary classification targets are:
- target_geq_p95_30m, target_geq_p99_30m   (30-minute horizon)
- target_geq_p95_60m, target_geq_p99_60m   (60-minute horizon)
- target_geq_p95_120m, target_geq_p99_120m (120-minute horizon)

Lazy-window dataset design
---------------------------
Instead of materializing all overlapping windows into a dense tensor
cube (which multiplies memory by lookback_steps for heavily overlapping
data), this script:
1. Sorts rows by (station.upper(), timestamp) per split.
2. Computes train-derived median imputation and standardization
   parameters in-place via fit_and_transform_numpy().
3. Builds vectorized start/end endpoint index arrays per station block
   via build_endpoint_arrays_vectorized(): for each contiguous station
   block of length L, generates (L - lookback_steps + 1) valid endpoint
   pairs. Blocks shorter than lookback_steps produce no sequences.
4. LazyStationSequenceDataset.__getitem__(idx) slices X_rows[start:end+1]
   on demand at DataLoader fetch time. The label is y_rows[end]
   (endpoint row label). No sequence crosses a station boundary or
   a split boundary.
5. get_metadata_frame() returns meta_rows.iloc[end_indices]: prediction
   metadata corresponds to the endpoint (label) row of each window.

Default --top-k-features = 20. LSTMs are sensitive to input
dimensionality and overfitting on tabular features; this default
constrains input size without a CLI override.

Physics-informed architecture (LSTMClassifier)
------------------------------------------------
LSTMClassifier supports two modes controlled by --layer-sizes:

MODE 1 -- Heterogeneous layer sizes (--layer-sizes set):
  Stacks individual single-layer nn.LSTM modules in a nn.ModuleList,
  each with its own hidden size. Dropout applied between layers (not
  after the last). Motivated by the Sun-to-ground causal propagation
  chain: each stage compresses the upstream signal, analogous to the
  progressive filtering from solar wind through magnetosphere,
  ionosphere, and to the ground GIC response.

  Example: --layer-sizes 24,16,8,4
    Layer 0: input_size -> 24   (L1/solar wind proxy)
    Layer 1: 24 -> 16           (magnetosphere/GEO proxy)
    Layer 2: 16 -> 8            (ionosphere/LEO proxy)
    Layer 3: 8 -> 4             (near-ground proxy)
    Head:    4 -> max(2,1) -> 1 (GIC logit)

  Theoretical grounding:
  - Information bottleneck (Tishby et al., 2000): narrowing enforces
    structural compression, discarding irrelevant variance at each stage.
  - Hierarchical representation learning (Bengio et al., 2013): later
    layers learn increasingly abstract summaries of the input.
  - Physics-informed neural networks (Raissi et al., 2019): encoding
    known causal structure into architecture rather than learning it
    from data alone.

  Important caveats: the specific sizes are design choices, not
  analytically derived from physics. Each input row contains all feature
  families simultaneously (L1, GEO, LEO, SuperMAG), not in causal order,
  so the per-layer physical mapping is approximate. Ablation against
  uniform-width architectures is required before claiming the narrowing
  adds measurable value (see Future Work).

MODE 2 -- Uniform hidden sizes (--layer-sizes not set):
  Falls back to standard nn.LSTM(hidden_size, num_layers) behavior with
  uniform hidden size across all layers.

Head (both modes):
  nn.Linear(final_hidden, max(final_hidden // 2, 1))
  -> nn.GELU()
  -> nn.Dropout(dropout)
  -> nn.Linear(max(final_hidden // 2, 1), 1)
  -> squeeze(-1) -> raw scalar logits

forward() returns logits. BCEWithLogitsLoss applied during training.
predict_probabilities() applies torch.sigmoid() explicitly.

Default hyperparameters: hidden_size=64, num_layers=2, dropout=0.3,
learning_rate=3e-4, weight_decay=1e-5, epochs=15, patience=3,
batch_size=1024, lookback_steps=60.

Loss and class imbalance
-------------------------
pos_weight = sqrt(neg_count / pos_count) per target on the training
sequence labels. Dampened weighting (square root rather than full
ratio) applied via BCEWithLogitsLoss(pos_weight=pos_weight).

Training loop and early stopping
----------------------------------
Patience-based early stopping monitors val_score at threshold=0.5 each
epoch (not the optimized threshold). Best model state saved as a
CPU-resident state dict clone when val_score improves. After training,
best state is loaded back, then select_threshold() runs on full val
probabilities to choose the final threshold by --threshold-metric
(default F2). AMP: torch.autocast for CUDA and MPS; GradScaler
initialized once before the target loop (CUDA only).

Model serialization
--------------------
torch.save(model.state_dict(), artifacts/model.pt) - state dict only.
To reconstruct: instantiate LSTMClassifier with the original
input_size, hidden_size/layer_sizes, num_layers, dropout, then load
the state dict. No preprocessing parameters (medians, means, stds) are
persisted to disk.

Note: model_config.json and run_config.json are NOT written by the
current implementation. The run-level summary_metrics.csv contains
only 5 columns: target, selected_threshold, train_f2, val_f2, test_f2.

Skip behavior
--------------
HARD SKIP: if the per-target output directory exists and --force is not
set, the target is skipped via continue. This is the only Phase 10
classification script with a true skip rather than a soft warning.

Device and thread controls
---------------------------
--device "auto": CUDA -> MPS (Apple Silicon) -> CPU.
--cpu-threads (default 8): torch.set_num_threads().
--interop-threads (default 1): torch.set_num_interop_threads()
  (silently ignored if called after torch init).
Memory cleanup per target: del model, optimizer, criterion +
torch.cuda.empty_cache().

BENCHMARK RUN (Phase 10 reported results)
------------------------------------------
The Phase 10 reported results were produced with the following command:

  uv run src/models/classification/lstm_classifier.py \
    --batch-size 1024 \
    --lookback-steps 30 \
    --epochs 4 \
    --cpu-threads 8 \
    --learning-rate 1e-4 \
    --top-k-features 20 \
    --reduced-features-csv data/reports/features/importance/feature_importance_all_targets_long.csv \
    --num-workers 4 \
    --weight-decay 1e-4 \
    --layer-sizes 24,16,8,4 \
    --force

All six targets completed successfully (confirmed via metrics_summary.csv
presence for all targets).

Known limitations of this benchmark run
-----------------------------------------
1. The LSTM was not hyperparameter-optimized for this study. Results
   should be interpreted as a proof-of-concept sequence baseline rather
   than a tuned model.

2. --lookback-steps 30 was held constant across all six targets. This is
   appropriate for 30-minute horizon targets but is likely too short for
   60-minute and 120-minute horizon targets, where a longer lookback
   (e.g. 60-120 steps) would better match the prediction horizon. This
   is a known limitation of the Phase 10 reported results and should be
   addressed in any follow-up study.

3. --layer-sizes 24,16,8,4 and --epochs 4 were chosen to minimize
   runtime, not to maximize performance. The architecture has not been
   ablated against uniform-width configurations (see Future Work).

WORK FLOW
---------
Step 1: Parse CLI args. Set seed (random, numpy, torch, cuda). Configure
        torch threads. Resolve device (auto -> CUDA -> MPS -> CPU).
        Load feature_columns.json and optional reduced feature list.
        Initialize GradScaler if CUDA.
Step 2: For each target in the selected target list:
  Step 2a: HARD SKIP if output dir exists and not --force.
  Step 2b: Read train/val/test.parquet via pl.read_parquet(), optionally
           capped by --max-train/val/test-rows (head-based).
  Step 2c: Resolve feature columns to those present in train schema.
  Step 2d: to_optimized_xy(): sort by (station.upper(), timestamp),
           extract X (float32), y (int8), meta (pandas), stations (str).
  Step 2e: fit_and_transform_numpy(): train medians fill NaN in-place;
           train means/stds standardize all three splits in-place.
  Step 2f: build_endpoint_arrays_vectorized(): start/end index arrays
           for all valid sequences per station block.
  Step 2g: Construct LazyStationSequenceDataset for train/val/test.
           Build DataLoaders (train: shuffle=True; val/test: False).
  Step 2h: Instantiate LSTMClassifier with parse_layer_sizes() result.
           pos_weight = sqrt(neg/pos). BCEWithLogitsLoss, Adam optimizer.
  Step 2i: Training loop (--epochs, --patience early stopping):
           run_epoch train, run_epoch val, predict val at 0.5, save
           best state if val score improves.
  Step 2j: Load best state. Generate final train/val/test probabilities.
  Step 2k: select_threshold() on val; compute 18 metrics all splits.
  Step 2l: save_predictions() for train/val/test (endpoint metadata,
           target col, y_prob, y_pred).
  Step 2m: Write training_history.csv, validation_threshold_grid.csv,
           metrics_summary.csv. torch.save(state_dict, model.pt).
  Step 2n: Append sparse summary row. Cleanup memory.
Step 3: Write summary_metrics.csv.

Metrics reported (18 per split)
---------------------------------
threshold, rows, event_rate, pred_rate, precision, recall, f1, f2,
balanced_accuracy, pr_auc, roc_auc, brier, log_loss, hss
(Heidke Skill Score), tn, fp, fn, tp.

INPUT DATA
----------
- data/ml/standardized/{target_col}/train.parquet
- data/ml/standardized/{target_col}/val.parquet
- data/ml/standardized/{target_col}/test.parquet
  Phase 8.1 output.
- data/ml/standardized/metadata/feature_columns.json
  Phase 8.1 output. Used when no reduced feature source is supplied.
- CLI optional: --data-root             (default: data/ml/standardized)
               --output-root           (default: data/reports/models/
                                        lstm_classifier)
               --targets               (default: all six targets)
               --feature-columns-json  (default: metadata/feature_columns.json)
               --reduced-features-json (default: None)
               --reduced-features-csv  (default: None)
               --top-k-features        (default: 20)
               --lookback-steps        (default: 60)
               --batch-size            (default: 1024)
               --hidden-size           (default: 64; ignored if --layer-sizes set)
               --num-layers            (default: 2; ignored if --layer-sizes set)
               --layer-sizes           (default: None; e.g. "24,16,8,4";
                                        overrides hidden-size and num-layers)
               --dropout               (default: 0.3)
               --learning-rate         (default: 3e-4)
               --weight-decay          (default: 1e-5)
               --epochs                (default: 15)
               --patience              (default: 3)
               --threshold-metric      (default: f2; choices: f1, f2,
                                        balanced_accuracy)
               --device                (default: auto)
               --num-workers           (default: 4)
               --cpu-threads           (default: 8)
               --interop-threads       (default: 1)
               --max-train-rows        (default: None)
               --max-val-rows          (default: None)
               --max-test-rows         (default: None)
               --seed                  (default: 42)
               --force                 (hard skip if not set)

OUTPUT DATA
-----------
Per-target (6 files each, 36 total):
- artifacts/model.pt                      (model state dict only)
- metrics/training_history.csv            (epoch, train_loss, val_loss,
                                           val_score per epoch)
- metrics/validation_threshold_grid.csv
- metrics/metrics_summary.csv             (train + val + test rows)
- predictions/train_predictions.parquet
- predictions/val_predictions.parquet
- predictions/test_predictions.parquet
All written under data/reports/models/lstm_classifier/{target_col}/.

Note: model_config.json and run_config.json are NOT written by the
current implementation.

Run-level (1 file):
- data/reports/models/lstm_classifier/summary_metrics.csv
  (5 columns: target, selected_threshold, train_f2, val_f2, test_f2)

FUTURE WORK
-----------
1. Ablation: --layer-sizes architecture search
   Before claiming the narrowing architecture adds value over a uniform
   baseline, run the minimum ablation set:
     A: --layer-sizes 24,16,8,4  (current benchmark)
     B: --hidden-size 16 --num-layers 4  (uniform, ~same depth/params)
     C: --hidden-size 64 --num-layers 1  (shallow uniform)
     D: --layer-sizes 32,16  (2-layer moderate narrowing)
   Compare A vs B on val F2 with identical other args. If A > B, the
   narrowing is functionally justified. If A approximately equals B,
   report as an architecture choice with information bottleneck
   motivation but no demonstrated performance advantage.

2. Per-target lookback tuning
   --lookback-steps 30 is likely suboptimal for 60- and 120-minute
   horizon targets. A reasonable starting point:
     target_*_30m:  --lookback-steps 30
     target_*_60m:  --lookback-steps 60
     target_*_120m: --lookback-steps 120

3. Event-focused training for rare p99 targets
   Build training windows centered on published substorm onset times
   from Chen (2025), keeping val and test splits unchanged.
   Candidate onset timestamps (24 events):
     "2011-08-05 18:02", "2015-02-16 19:24", "2015-03-17 04:07",
     "2015-04-09 21:52", "2015-04-14 12:55", "2015-05-12 18:05",
     "2015-05-18 10:12", "2015-06-07 10:30", "2015-06-22 05:00",
     "2015-07-04 13:06", "2015-07-10 22:21", "2015-07-23 01:51",
     "2015-08-15 08:04", "2015-08-26 05:45", "2015-09-07 13:13",
     "2015-09-08 21:45", "2015-09-20 05:46", "2015-10-04 00:30",
     "2015-10-07 01:41", "2015-11-03 05:31", "2015-11-06 18:09",
     "2015-11-30 06:09", "2015-12-19 16:13", "2024-05-10 18:00"
   Must be treated as a separate ablation under the same leakage-safe
   protocol; do not mix into the main benchmark results.

NEXT PIPELINE PHASE
-------------------
- Phase 10.5 depends on Phase 8.1 completing first.
- Phase 10.5 is the final Phase 10 script. All Phase 10 prediction
  Parquet files and metrics CSVs feed into Phase 11 (src/evaluate/)
  for cross-model comparison.
"""
---
