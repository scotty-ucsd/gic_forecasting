"""
Phase 9.1 - Feature Importance: Evaluate and Rank Features for Classification
------------------------------------------------------------------------------
This is the second script in src/features/ and the sole Phase 9 script.
It depends on Phase 8.1 (src/dataset/build_ml_dataset.py) having
written both the per-target train.parquet files and the
feature_columns.json metadata file before it can run. It reads the
training split for each of the six binary classification targets,
scores every feature using four complementary methods, computes a
weighted composite importance score, aggregates rankings globally and
by source family, and writes a reduced feature recommendation
shortlist. This script uses Polars for all I/O and Polars-to-pandas
conversion for sklearn and LightGBM computations.

Purpose
-------
The standardized feature table from Phase 7.1 contains up to 41
columns (21 base + 20 derived). Not all features contribute equally
to each forecasting horizon or threshold. This script provides
evidence-based screening to identify a compact, high-signal feature
subset before model training, reducing both training cost and the
risk of noise-driven overfitting.

The four scoring methods are complementary by design:
1. Univariate mutual information (sklearn.mutual_info_classif):
   captures nonlinear marginal associations; discrete_features=False.
2. Absolute Spearman correlation (scipy.stats.spearmanr):
   rank-order linear association; robust to outliers; computed
   per-column with a minimum of 10 finite observations required.
3. LightGBM split importance: number of times a feature is used to
   split a leaf node across all 300 trees.
4. LightGBM gain importance: total information gain attributable to
   a feature across all 300 trees.

All four raw scores are normalized to [0, 1] per target via
normalize_series() (divide by max; returns zeros if max is zero or
non-finite). The composite importance_score is then computed as:

  importance_score = 0.20 * mutual_info_norm
                   + 0.15 * abs_spearman_corr_norm
                   + 0.20 * lgb_split_importance_norm
                   + 0.45 * lgb_gain_importance_norm

LightGBM gain importance carries the largest weight (45%) because it
most directly measures predictive contribution. Results are produced
per target and then aggregated globally across all six targets.

LightGBM training configuration
---------------------------------
objective="binary", n_estimators=300, learning_rate=0.05,
num_leaves=63, subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
verbose=-1. scale_pos_weight = neg/pos row count ratio to correct
for class imbalance. Fitted on the median-imputed training matrix.

Preprocessing before scoring
------------------------------
prepare_matrix() applies three steps before passing data to
sklearn and LightGBM: (1) boolean columns are cast to Int8;
(2) any remaining non-numeric columns are coerced via
pd.to_numeric(errors="coerce"); (3) sklearn SimpleImputer with
strategy="median" fills all remaining NaN values. The imputed
numpy array (X_imp) is passed to mutual information and LightGBM;
the pre-imputation numeric DataFrame (X_num) is passed to Spearman.

Reduced feature candidate selection
-------------------------------------
build_reduced_candidates() applies a two-level constraint:
- Per-family cap: take at most --top-k-per-family (default 6)
  features by importance_score_mean within each of the six families
  (sun, l1, geo, leo, supermag_context, other).
- Global cap: take at most --top-k-global (default 24) features
  by global_rank across all families combined.
The resulting shortlist is written to reduced_feature_candidates.csv
and is the intended input for configuring Phase 10 model training
with a reduced feature set.

WORK FLOW
---------
Step 1: Parse CLI arguments - input root, output root, feature
        metadata JSON path, optional --targets subset, --max-rows
        (default 300,000), --random-seed (default 42),
        --top-k-global (default 24), --top-k-per-family (default 6).
Step 2: Load canonical feature columns from feature_columns.json via
        load_feature_columns(). Strip EXCLUDED_FEATURE_COLS
        (supermag_dbn_nt, supermag_dbe_nt, supermag_dbz_nt).
        Raise FileNotFoundError if JSON absent; ValueError if empty.
Step 3: For each target in the selected target list:
  Step 3a: Load train.parquet for the target via load_train_frame():
           scan with Polars, select available feature + target
           columns, filter null target rows, optionally subsample
           to --max-rows via Polars sample(), then convert to pandas.
  Step 3b: Prepare matrix via prepare_matrix(): cast booleans to
           Int8, coerce non-numeric, apply median imputation.
  Step 3c: Compute mutual information via compute_mutual_information().
  Step 3d: Compute per-column absolute Spearman correlation via
           compute_spearman() (skipping columns with < 10 finite
           observations or zero std).
  Step 3e: Fit LightGBM classifier via fit_lightgbm() and extract
           split and gain importances.
  Step 3f: Normalize all four metrics, compute composite
           importance_score, assign family via infer_family(), add
           target name and train diagnostics.
  Step 3g: Write per-target feature_importance_{target_col}.csv.
Step 4: Concatenate all per-target results and call
        aggregate_global_rankings(): group by (feature, family),
        aggregate importance stats across targets, sort by
        importance_score_mean desc then lgb_gain_importance_mean desc,
        assign global_rank. Build family_summary grouped by family.
Step 5: Call build_reduced_candidates() to apply two-level
        per-family and global cap selection.
Step 6: Write 5 aggregate output artifacts and run_config.json.

INPUT DATA
----------
- data/ml/standardized/{target_col}/train.parquet
  Phase 8.1 output. One file per target. Required columns: all
  feature columns present in feature_columns.json and the target
  column itself.
- data/ml/standardized/metadata/feature_columns.json
  Phase 8.1 output. Must contain "feature_columns" or
  "selected_columns" key with the list of numeric feature names.
- CLI optional: --input-root           (default: data/ml/standardized)
               --output-root          (default: data/reports/features/importance)
               --feature-metadata-json (default: data/ml/standardized/
                                        metadata/feature_columns.json)
               --targets              (default: all six targets)
               --max-rows             (default: 300,000)
               --random-seed          (default: 42)
               --top-k-global         (default: 24)
               --top-k-per-family     (default: 6)

OUTPUT DATA
-----------
Per-target (6 files):
- data/reports/features/importance/feature_importance_{target_col}.csv
  Columns: feature, family, target, mutual_info, abs_spearman_corr,
  lgb_split_importance, lgb_gain_importance, *_norm variants,
  importance_score, train_rows_used, event_rate. Sorted by
  importance_score descending.

Aggregate (5 files):
- feature_importance_all_targets_long.csv
  Vertical concat of all six per-target frames.
- feature_importance_global.csv
  One row per feature. Aggregated stats across targets plus
  global_rank (ascending from 1).
- feature_importance_family_summary.csv
  One row per source family. n_features, mean/best importance
  score, best_rank.
- reduced_feature_candidates.csv
  Feature shortlist after two-level per-family and global cap.
  Columns: global_rank, family_rank, feature, family, importance
  stats. Intended input for Phase 10 model training.
- run_config.json
  Serialized CLI arguments, n_feature_columns, excluded columns.

NEXT PIPELINE PHASE
-------------------
- Phase 9.1 depends on Phase 8.1 (src/dataset/build_ml_dataset.py)
  completing first and writing both train.parquet files and
  feature_columns.json.
- The reduced_feature_candidates.csv output is the primary input
  for configuring Phase 10 (src/models/) model training with a
  compact feature set.
- Phase 10 (src/models/) and Phase 11 (src/evaluate/) follow.
"""
