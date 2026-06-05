#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

DEFAULT_INPUT_ROOT = Path("data/ml/standardized")
DEFAULT_OUTPUT_ROOT = Path("data/reports/features/importance")
DEFAULT_FEATURE_METADATA = Path("data/ml/standardized/metadata/feature_columns.json")

TARGET_COLS = [
    "target_geq_p95_30m",
    "target_geq_p99_30m",
    "target_geq_p95_60m",
    "target_geq_p99_60m",
    "target_geq_p95_120m",
    "target_geq_p99_120m",
]

EXCLUDED_FEATURE_COLS = {
    "supermag_dbn_nt",
    "supermag_dbe_nt",
    "supermag_dbz_nt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate feature importance for standardized classification datasets."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root containing one directory per target with train/val/test parquet files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where feature-importance reports will be written.",
    )
    parser.add_argument(
        "--feature-metadata-json",
        type=Path,
        default=DEFAULT_FEATURE_METADATA,
        help="JSON file containing canonical feature_columns list.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional subset of target column names to evaluate.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=300_000,
        help="Maximum number of train rows to sample per target for importance evaluation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and model fitting.",
    )
    parser.add_argument(
        "--top-k-global",
        type=int,
        default=24,
        help="Maximum number of globally recommended features.",
    )
    parser.add_argument(
        "--top-k-per-family",
        type=int,
        default=6,
        help="Maximum number of recommended features per source family.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_feature_columns(feature_metadata_json: Path) -> list[str]:
    if not feature_metadata_json.exists():
        raise FileNotFoundError(f"Feature metadata JSON not found: {feature_metadata_json}")

    payload = json.loads(feature_metadata_json.read_text())
    cols = payload.get("feature_columns") or payload.get("selected_columns")
    if not cols:
        raise ValueError(f"No feature_columns found in {feature_metadata_json}")

    filtered = [c for c in cols if c not in EXCLUDED_FEATURE_COLS]
    if not filtered:
        raise ValueError("All feature columns were excluded; check exclusion rules.")

    return filtered


def choose_targets(requested: list[str] | None) -> list[str]:
    if not requested:
        return TARGET_COLS

    bad = sorted(set(requested) - set(TARGET_COLS))
    if bad:
        raise ValueError(f"Unknown targets requested: {bad}")
    return requested


def infer_family(col: str) -> str:
    for prefix in ["sun_", "l1_", "geo_", "leo_"]:
        if col.startswith(prefix):
            return prefix[:-1]
    if col.startswith("supermag_"):
        return "supermag_context"
    return "other"


def load_train_frame(
    train_path: Path,
    feature_cols: list[str],
    target_col: str,
    max_rows: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if not train_path.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_path}")

    scan = pl.scan_parquet(str(train_path))
    schema = set(scan.collect_schema().names())

    required = {target_col}
    missing = required - schema
    if missing:
        raise ValueError(f"{train_path}: missing target column(s): {sorted(missing)}")

    available_features = [c for c in feature_cols if c in schema]
    if not available_features:
        raise ValueError(f"{train_path}: none of the expected feature columns were found")

    df = (
        scan
        .select(available_features + [target_col])
        .filter(pl.col(target_col).is_not_null())
        .collect(engine="streaming")
    )

    if df.height == 0:
        raise ValueError(f"{train_path}: zero rows after dropping null targets")

    if max_rows is not None and df.height > max_rows:
        frac = max_rows / df.height
        df = df.sample(fraction=frac, with_replacement=False, shuffle=True, seed=random_seed)

    pdf = df.to_pandas()
    y = pdf[target_col].astype(np.int8)
    X = pdf.drop(columns=[target_col])

    return X, y


def prepare_matrix(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X_num = X.copy()

    bool_cols = X_num.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X_num[bool_cols] = X_num[bool_cols].astype(np.int8)

    for col in X_num.columns:
        if not pd.api.types.is_numeric_dtype(X_num[col]):
            X_num[col] = pd.to_numeric(X_num[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_num)
    return X_num, X_imp


def compute_mutual_information(
    X_imp: np.ndarray,
    y: pd.Series,
    columns: list[str],
    random_seed: int,
) -> pd.Series:
    mi = mutual_info_classif(
        X_imp,
        y.to_numpy(),
        discrete_features=False,
        random_state=random_seed,
    )
    return pd.Series(mi, index=columns, name="mutual_info")


def compute_spearman(X_num: pd.DataFrame, y: pd.Series) -> pd.Series:
    y_arr = y.to_numpy(dtype=float)
    vals: list[float] = []

    for col in X_num.columns:
        x = pd.to_numeric(X_num[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y_arr)
        if mask.sum() < 10:
            vals.append(np.nan)
            continue

        x_valid = x[mask]
        y_valid = y_arr[mask]

        if np.nanstd(x_valid) == 0 or np.nanstd(y_valid) == 0:
            vals.append(np.nan)
            continue

        corr = spearmanr(x_valid, y_valid, nan_policy="omit").statistic
        vals.append(corr if corr is not None else np.nan)

    return pd.Series(vals, index=X_num.columns, name="spearman_corr")


def fit_lightgbm(
    X_imp: np.ndarray,
    y: pd.Series,
    columns: list[str],
    random_seed: int,
) -> pd.DataFrame:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )
    model.fit(X_imp, y.to_numpy())

    return pd.DataFrame(
        {
            "feature": columns,
            "lgb_split_importance": model.booster_.feature_importance(importance_type="split"),
            "lgb_gain_importance": model.booster_.feature_importance(importance_type="gain"),
        }
    )


def normalize_series(s: pd.Series) -> pd.Series:
    s = s.fillna(0.0).astype(float)
    mx = s.max()
    if not np.isfinite(mx) or mx <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return s / mx


def score_target(
    target_col: str,
    input_root: Path,
    feature_cols: list[str],
    max_rows: int,
    random_seed: int,
) -> pl.DataFrame:
    train_path = input_root / target_col / "train.parquet"

    X, y = load_train_frame(
        train_path=train_path,
        feature_cols=feature_cols,
        target_col=target_col,
        max_rows=max_rows,
        random_seed=random_seed,
    )
    X_num, X_imp = prepare_matrix(X)

    mi = compute_mutual_information(X_imp, y, list(X_num.columns), random_seed)
    spearman = compute_spearman(X_num, y).abs().rename("abs_spearman_corr")
    lgb_df = fit_lightgbm(X_imp, y, list(X_num.columns), random_seed)

    df = pd.DataFrame({"feature": list(X_num.columns)})
    df = df.merge(
        mi.rename("mutual_info").reset_index().rename(columns={"index": "feature"}),
        on="feature",
        how="left",
    )
    df = df.merge(
        spearman.reset_index().rename(columns={"index": "feature"}),
        on="feature",
        how="left",
    )
    df = df.merge(lgb_df, on="feature", how="left")

    df["mutual_info_norm"] = normalize_series(df["mutual_info"])
    df["abs_spearman_corr_norm"] = normalize_series(df["abs_spearman_corr"])
    df["lgb_split_importance_norm"] = normalize_series(df["lgb_split_importance"])
    df["lgb_gain_importance_norm"] = normalize_series(df["lgb_gain_importance"])

    df["importance_score"] = (
        0.20 * df["mutual_info_norm"]
        + 0.15 * df["abs_spearman_corr_norm"]
        + 0.20 * df["lgb_split_importance_norm"]
        + 0.45 * df["lgb_gain_importance_norm"]
    )

    df["family"] = df["feature"].map(infer_family)
    df["target"] = target_col
    df["train_rows_used"] = int(len(X_num))
    df["event_rate"] = float(y.mean())
    df = df.sort_values(
        ["importance_score", "lgb_gain_importance", "mutual_info"],
        ascending=False,
    ).reset_index(drop=True)

    return pl.from_pandas(df)


def aggregate_global_rankings(all_results: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    global_df = (
        all_results
        .group_by(["feature", "family"])
        .agg(
            pl.len().alias("n_targets"),
            pl.col("importance_score").mean().alias("importance_score_mean"),
            pl.col("importance_score").median().alias("importance_score_median"),
            pl.col("importance_score").max().alias("importance_score_max"),
            pl.col("mutual_info").mean().alias("mutual_info_mean"),
            pl.col("abs_spearman_corr").mean().alias("abs_spearman_corr_mean"),
            pl.col("lgb_split_importance").mean().alias("lgb_split_importance_mean"),
            pl.col("lgb_gain_importance").mean().alias("lgb_gain_importance_mean"),
        )
        .sort(["importance_score_mean", "lgb_gain_importance_mean"], descending=True)
        .with_row_index(name="global_rank", offset=1)
    )

    family_summary = (
        global_df
        .group_by("family")
        .agg(
            pl.len().alias("n_features"),
            pl.col("importance_score_mean").mean().alias("importance_score_mean"),
            pl.col("importance_score_mean").max().alias("importance_score_best"),
            pl.col("global_rank").min().alias("best_rank"),
        )
        .sort("best_rank")
    )

    return global_df, family_summary


def build_reduced_candidates(
    global_df: pl.DataFrame,
    *,
    top_k_global: int,
    top_k_per_family: int,
) -> pl.DataFrame:
    family_tables: list[pl.DataFrame] = []

    for family in ["sun", "l1", "geo", "leo", "supermag_context", "other"]:
        fam = (
            global_df
            .filter(pl.col("family") == family)
            .sort(["importance_score_mean", "lgb_gain_importance_mean"], descending=True)
            .head(top_k_per_family)
            .with_row_index(name="family_rank", offset=1)
        )
        if fam.height > 0:
            family_tables.append(fam)

    if not family_tables:
        return pl.DataFrame()

    reduced = (
        pl.concat(family_tables, how="vertical")
        .sort("global_rank")
        .head(top_k_global)
        .select(
            [
                "global_rank",
                "family_rank",
                "feature",
                "family",
                "importance_score_mean",
                "importance_score_median",
                "lgb_gain_importance_mean",
                "mutual_info_mean",
                "abs_spearman_corr_mean",
            ]
        )
    )
    return reduced


def write_run_config(
    args: argparse.Namespace,
    output_root: Path,
    feature_cols: list[str],
    targets: list[str],
) -> None:
    payload = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "feature_metadata_json": str(args.feature_metadata_json),
        "targets": targets,
        "n_feature_columns": len(feature_cols),
        "excluded_feature_columns": sorted(EXCLUDED_FEATURE_COLS),
        "max_rows": args.max_rows,
        "random_seed": args.random_seed,
        "top_k_global": args.top_k_global,
        "top_k_per_family": args.top_k_per_family,
    }
    (output_root / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)

    feature_cols = load_feature_columns(args.feature_metadata_json)
    targets = choose_targets(args.targets)

    per_target_frames: list[pl.DataFrame] = []

    for target_col in targets:
        result_df = score_target(
            target_col=target_col,
            input_root=args.input_root,
            feature_cols=feature_cols,
            max_rows=args.max_rows,
            random_seed=args.random_seed,
        )

        result_df.write_csv(args.output_root / f"feature_importance_{target_col}.csv")
        per_target_frames.append(result_df)

        top10 = result_df.select(["feature", "family","importance_score"]).head(20).to_dicts()
        print(f"[ok] {target_col} | top features:")
        for row in top10:
            print(f"  - {row['feature']} ({row['family']}) score={float(row['importance_score']):.4f}")

    all_results = pl.concat(per_target_frames, how="vertical")
    global_df, family_summary = aggregate_global_rankings(all_results)
    reduced_df = build_reduced_candidates(
        global_df,
        top_k_global=args.top_k_global,
        top_k_per_family=args.top_k_per_family,
    )

    all_results.write_csv(args.output_root / "feature_importance_all_targets_long.csv")
    global_df.write_csv(args.output_root / "feature_importance_global.csv")
    family_summary.write_csv(args.output_root / "feature_importance_family_summary.csv")
    reduced_df.write_csv(args.output_root / "reduced_feature_candidates.csv")
    write_run_config(args, args.output_root, feature_cols, targets)

    print()
    print("Done.")
    print(f"Targets evaluated: {len(targets)}")
    print(f"Excluded near-target SuperMAG features: {sorted(EXCLUDED_FEATURE_COLS)}")
    print(f"Global ranking: {args.output_root / 'feature_importance_global.csv'}")
    print(f"Reduced candidates: {args.output_root / 'reduced_feature_candidates.csv'}")


if __name__ == "__main__":
    main()


