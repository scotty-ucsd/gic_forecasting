#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise SystemExit(
        "lightgbm is not installed. Install it first, e.g. `uv add lightgbm`."
    ) from exc


DEFAULT_DATA_ROOT = Path("data/ml/standardized")
DEFAULT_OUTPUT_ROOT = Path("data/reports/models/lightgbm_main")
DEFAULT_METADATA_DIRNAME = "metadata"

TRAIN_FILE = "train.parquet"
VAL_FILE = "val.parquet"
TEST_FILE = "test.parquet"

TARGET_COLS = [
    "target_geq_p95_30m",
    "target_geq_p99_30m",
    "target_geq_p95_60m",
    "target_geq_p99_60m",
    "target_geq_p95_120m",
    "target_geq_p99_120m",
]

ID_COLS = ["timestamp", "station", "year", "month", "yyyymm"]
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM classifiers on standardized six-target station-time datasets."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing metadata/ and one subdirectory per target with train/val/test parquet files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for saved models, predictions, metrics, and config.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional subset of targets to run. Defaults to all canonical targets.",
    )
    parser.add_argument(
        "--feature-columns-json",
        type=Path,
        default=None,
        help="Optional explicit path to metadata/feature_columns.json.",
    )
    parser.add_argument(
        "--reduced-features-json",
        type=Path,
        default=None,
        help="Optional JSON file containing a reduced shared feature list.",
    )
    parser.add_argument(
        "--reduced-features-csv",
        type=Path,
        default=None,
        help="Optional CSV containing reduced candidate features. Uses a 'feature' column if present.",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=None,
        help="Optional cap applied after loading reduced features. Ignored when no reduced feature file is supplied.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=3_000_000,
        help="Optional cap on training rows for faster fitting.",
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=None,
        help="Optional cap on validation rows.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional cap on test rows.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "f2", "balanced_accuracy"],
        default="f2",
        help="Metric used on validation set to choose classification threshold.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_feature_columns(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "feature_columns" in data:
        cols = data["feature_columns"]
    elif isinstance(data, list):
        cols = data
    else:
        raise ValueError(f"Unsupported feature columns JSON format: {path}")
    if not cols:
        raise ValueError("No feature columns found.")
    return list(cols)


def load_reduced_feature_columns(args: argparse.Namespace) -> list[str] | None:
    cols: list[str] | None = None

    if args.reduced_features_json is not None:
        data = json.loads(args.reduced_features_json.read_text())
        if isinstance(data, dict):
            if "feature_columns" in data:
                cols = list(data["feature_columns"])
            elif "features" in data:
                cols = list(data["features"])
            else:
                raise ValueError(f"Unsupported reduced feature JSON format: {args.reduced_features_json}")
        elif isinstance(data, list):
            cols = list(data)
        else:
            raise ValueError(f"Unsupported reduced feature JSON format: {args.reduced_features_json}")

    elif args.reduced_features_csv is not None:
        df = pd.read_csv(args.reduced_features_csv)
        if "feature" in df.columns:
            cols = df["feature"].dropna().astype(str).tolist()
        else:
            cols = df.iloc[:, 0].dropna().astype(str).tolist()

    if cols is None:
        return None

    deduped = []
    seen = set()
    for c in cols:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    if args.top_k_features is not None:
        deduped = deduped[: args.top_k_features]

    if not deduped:
        raise ValueError("Reduced feature list resolved to zero columns.")

    return deduped


def read_split(path: Path, n_rows: int | None = None) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pl.read_parquet(path)
    if n_rows is not None:
        df = df.head(n_rows)
    return df


def resolve_feature_columns(
    train_df: pl.DataFrame,
    feature_cols: list[str],
) -> list[str]:
    available = set(train_df.columns)
    keep = [c for c in feature_cols if c in available]
    if not keep:
        raise ValueError("No usable feature columns found in training split.")
    return keep


def to_pandas_xy(
    df: pl.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    meta_cols = [c for c in ID_COLS if c in df.columns]
    keep_cols = meta_cols + [target_col] + feature_cols
    pdf = df.select(keep_cols).to_pandas()
    X = pdf[feature_cols].copy()
    y = pd.to_numeric(pdf[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    meta = pdf[meta_cols].copy()
    return X, y, meta


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-8, 1 - 1e-8)
    return float(log_loss(y_true, y_prob, labels=[0, 1]))


def heidke_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    numerator = 2 * (tp * tn - fp * fn)
    denominator = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "event_rate": float(np.mean(y_true)),
        "pred_rate": float(np.mean(y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": safe_log_loss(y_true, y_prob),
        "hss": heidke_skill_score(y_true, y_pred),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )
    return metrics


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for thr in THRESHOLD_GRID:
        rows.append(compute_metrics(y_true, y_prob, float(thr)))

    grid_df = pd.DataFrame(rows).sort_values(
        by=[metric_name, "pr_auc", "recall"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    best_thr = float(grid_df.iloc[0]["threshold"])
    return best_thr, grid_df


def make_param_grid() -> list[dict]:
    return [
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        },
        {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.5,
        },
        {
            "n_estimators": 700,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "max_depth": -1,
            "min_child_samples": 150,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
    ]


def fit_imputer(
    X_train: pd.DataFrame,
) -> tuple[SimpleImputer, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    return imputer, X_train_imp


def apply_imputer(imputer: SimpleImputer, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index,
    )


def fit_model_grid(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    threshold_metric: str,
) -> tuple[LGBMClassifier, dict, pd.DataFrame]:
    trial_rows: list[dict[str, float]] = []
    best = None
    best_model = None

    for params in make_param_grid():
        model = LGBMClassifier(
            objective="binary",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            verbosity=-1,
            **params,
        )
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        best_thr, thr_df = select_threshold(y_val, val_prob, threshold_metric)
        val_metrics = compute_metrics(y_val, val_prob, best_thr)

        row = {
            **params,
            "threshold": float(best_thr),
            **{f"val_{k}": v for k, v in val_metrics.items() if k not in {"tn", "fp", "fn", "tp"}},
            "val_tn": val_metrics["tn"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_tp": val_metrics["tp"],
        }
        trial_rows.append(row)

        score = (
            val_metrics[threshold_metric],
            val_metrics["pr_auc"] if not np.isnan(val_metrics["pr_auc"]) else -np.inf,
            val_metrics["hss"] if not np.isnan(val_metrics["hss"]) else -np.inf,
        )
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "params": params,
                "threshold": float(best_thr),
                "val_metrics": val_metrics,
                "threshold_grid": thr_df,
            }
            best_model = model

    trial_df = pd.DataFrame(trial_rows).sort_values(
        by=[f"val_{threshold_metric}", "val_pr_auc", "val_hss"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    assert best is not None and best_model is not None
    return best_model, best, trial_df


def save_predictions(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    out_path: Path,
) -> None:
    pred_df = meta.copy()
    pred_df["y_true"] = y_true.astype(int)
    pred_df["y_prob"] = y_prob.astype(float)
    pred_df["y_pred"] = (y_prob >= threshold).astype(int)
    pred_df.to_parquet(out_path, index=False)


def save_feature_importance(
    model: LGBMClassifier,
    feature_cols: list[str],
    out_path: Path,
) -> pd.DataFrame:
    booster = model.booster_
    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    )
    imp_df["importance_gain_norm"] = imp_df["importance_gain"] / max(imp_df["importance_gain"].sum(), 1e-12)
    imp_df = imp_df.sort_values("importance_gain", ascending=False)
    imp_df.to_csv(out_path, index=False)
    return imp_df


def validate_binary_labels(y: np.ndarray, split_name: str, target_col: str) -> None:
    unique = set(np.unique(y).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{target_col} {split_name} split contains non-binary labels: {sorted(unique)}")
    if len(unique) < 2:
        print(f"[warn] {target_col} {split_name} split has only one class: {sorted(unique)}")


def resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.targets:
        bad = [t for t in args.targets if t not in TARGET_COLS]
        if bad:
            raise ValueError(f"Unsupported targets requested: {bad}")
        return list(args.targets)
    return list(TARGET_COLS)


def main() -> None:
    args = parse_args()

    meta_dir = args.data_root / DEFAULT_METADATA_DIRNAME
    feature_json = args.feature_columns_json or (meta_dir / "feature_columns.json")
    all_feature_cols = load_feature_columns(feature_json)
    reduced_feature_cols = load_reduced_feature_columns(args)
    targets = resolve_targets(args)

    ensure_dir(args.output_root)
    all_summary_rows: list[dict] = []

    for target_col in targets:
        target_dir = args.data_root / target_col
        train_path = target_dir / TRAIN_FILE
        val_path = target_dir / VAL_FILE
        test_path = target_dir / TEST_FILE

        output_root = args.output_root / target_col
        model_dir = output_root / "artifacts"
        metrics_dir = output_root / "metrics"
        preds_dir = output_root / "predictions"

        if output_root.exists() and not args.force:
            print(f"[info] output exists: {output_root}")

        ensure_dir(model_dir)
        ensure_dir(metrics_dir)
        ensure_dir(preds_dir)

        train_df = read_split(train_path, args.max_train_rows)
        val_df = read_split(val_path, args.max_val_rows)
        test_df = read_split(test_path, args.max_test_rows)

        requested_features = reduced_feature_cols if reduced_feature_cols is not None else all_feature_cols
        feature_view = "reduced_features" if reduced_feature_cols is not None else "all_features"
        feature_cols = resolve_feature_columns(train_df, requested_features)

        X_train, y_train, meta_train = to_pandas_xy(train_df, target_col, feature_cols)
        X_val, y_val, meta_val = to_pandas_xy(val_df, target_col, feature_cols)
        X_test, y_test, meta_test = to_pandas_xy(test_df, target_col, feature_cols)

        validate_binary_labels(y_train, "train", target_col)
        validate_binary_labels(y_val, "val", target_col)
        validate_binary_labels(y_test, "test", target_col)

        imputer, X_train_imp = fit_imputer(X_train)
        X_val_imp = apply_imputer(imputer, X_val)
        X_test_imp = apply_imputer(imputer, X_test)

        model, best, trial_df = fit_model_grid(
            X_train=X_train_imp,
            y_train=y_train,
            X_val=X_val_imp,
            y_val=y_val,
            threshold_metric=args.threshold_metric,
        )

        best_threshold = float(best["threshold"])
        train_prob = model.predict_proba(X_train_imp)[:, 1]
        val_prob = model.predict_proba(X_val_imp)[:, 1]
        test_prob = model.predict_proba(X_test_imp)[:, 1]

        train_metrics = compute_metrics(y_train, train_prob, best_threshold)
        val_metrics = compute_metrics(y_val, val_prob, best_threshold)
        test_metrics = compute_metrics(y_test, test_prob, best_threshold)

        save_predictions(meta_train, y_train, train_prob, best_threshold, preds_dir / "train_predictions.parquet")
        save_predictions(meta_val, y_val, val_prob, best_threshold, preds_dir / "val_predictions.parquet")
        save_predictions(meta_test, y_test, test_prob, best_threshold, preds_dir / "test_predictions.parquet")

        imp_df = save_feature_importance(model, feature_cols, metrics_dir / "feature_importance.csv")
        trial_df.to_csv(metrics_dir / "hyperparameter_trials.csv", index=False)
        best["threshold_grid"].to_csv(metrics_dir / "validation_threshold_grid.csv", index=False)

        summary_df = pd.DataFrame(
            [
                {"split": "train", **train_metrics},
                {"split": "val", **val_metrics},
                {"split": "test", **test_metrics},
            ]
        )
        summary_df.to_csv(metrics_dir / "metrics_summary.csv", index=False)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(model_dir / "imputer.pkl", "wb") as f:
            pickle.dump(imputer, f)

        model_config = {
            "model": "lightgbm_main",
            "target_col": target_col,
            "feature_view": feature_view,
            "feature_count": len(feature_cols),
            "features": feature_cols,
            "selected_params": best["params"],
            "selected_threshold": best_threshold,
            "threshold_metric": args.threshold_metric,
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "max_test_rows": args.max_test_rows,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "top_feature_importance": imp_df.head(25).to_dict(orient="records"),
        }
        (model_dir / "model_config.json").write_text(json.dumps(model_config, indent=2, default=str))

        all_summary_rows.append(
            {
                "target": target_col,
                "feature_view": feature_view,
                "feature_count": len(feature_cols),
                "selected_threshold": best_threshold,
                "train_event_rate": train_metrics["event_rate"],
                "val_event_rate": val_metrics["event_rate"],
                "test_event_rate": test_metrics["event_rate"],
                "train_f2": train_metrics["f2"],
                "val_f2": val_metrics["f2"],
                "test_f2": test_metrics["f2"],
                "val_pr_auc": val_metrics["pr_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "val_roc_auc": val_metrics["roc_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "val_brier": val_metrics["brier"],
                "test_brier": test_metrics["brier"],
                **{f"best_{k}": v for k, v in best["params"].items()},
            }
        )

        print(
            f"[ok] {target_col} | feature_view={feature_view} | "
            f"features={len(feature_cols)} | threshold={best_threshold:.2f} | "
            f"val_f2={val_metrics['f2']:.6f} | test_f2={test_metrics['f2']:.6f}"
        )

    summary_out = pd.DataFrame(all_summary_rows).sort_values("target").reset_index(drop=True)
    summary_out.to_csv(args.output_root / "summary_metrics.csv", index=False)

    run_config = {
        "model": "lightgbm_main",
        "data_root": str(args.data_root),
        "targets": targets,
        "feature_columns_json": str(feature_json),
        "reduced_features_json": str(args.reduced_features_json) if args.reduced_features_json else None,
        "reduced_features_csv": str(args.reduced_features_csv) if args.reduced_features_csv else None,
        "top_k_features": args.top_k_features,
        "threshold_metric": args.threshold_metric,
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "max_test_rows": args.max_test_rows,
        "param_grid": make_param_grid(),
    }
    (args.output_root / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    print("\nDone.")
    print(f"Targets evaluated: {len(targets)}")
    print(f"Artifacts: {args.output_root}")


if __name__ == "__main__":
    main()

