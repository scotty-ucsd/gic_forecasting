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
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
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

DEFAULT_ML_ROOT = Path("data/ml/standardized")
DEFAULT_OUTPUT_ROOT = Path("data/reports/models/lightgbm_tuned_simple")

DEFAULT_SPLIT_FILES = {
    "train": ["train.parquet", "train_data.parquet", "X_train.parquet"],
    "val": ["val.parquet", "validation.parquet", "val_data.parquet", "X_val.parquet"],
    "test": ["test.parquet", "test_data.parquet", "X_test.parquet"],
}

TARGET_COLS = [
    "target_geq_p95_30m",
    "target_geq_p99_30m",
    "target_geq_p95_60m",
    "target_geq_p99_60m",
    "target_geq_p95_120m",
    "target_geq_p99_120m",
]

EXCLUDE_COLS = {
    "timestamp",
    "station",
    "split",
    "year",
    "month",
    "yyyymm",
}

ID_CANDIDATES = ["timestamp", "station", "year", "month", "yyyymm", "split"]

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
DEFAULT_RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple LightGBM hyperparameter tuning using train/val/test splits."
    )
    parser.add_argument("--ml-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=20, help="Random search trials per target.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_targets(requested: list[str] | None) -> list[str]:
    if requested is None or len(requested) == 0:
        return list(TARGET_COLS)
    bad = sorted(set(requested) - set(TARGET_COLS))
    if bad:
        raise ValueError(f"Unknown targets requested: {bad}")
    return list(requested)


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")

def resolve_split_path(target_root: Path, split_name: str) -> Path:
    candidates = DEFAULT_SPLIT_FILES[split_name]
    for name in candidates:
        path = target_root / name
        if path.exists():
            return path
    tried = [str(target_root / name) for name in candidates]
    raise FileNotFoundError(
        f"Could not find {split_name} parquet under {target_root}. Tried: {tried}"
    )


def load_split_frame(target_root: Path, split_name: str) -> pd.DataFrame:
    path = resolve_split_path(target_root, split_name)
    df = pd.read_parquet(path)
    df["split"] = split_name
    return df


def load_all_splits_for_target(ml_root: Path, target_col: str) -> dict[str, pd.DataFrame]:
    target_root = ml_root / target_col
    if not target_root.exists():
        raise FileNotFoundError(target_root)
    return {split: load_split_frame(target_root, split) for split in ["train", "val", "test"]}


def infer_feature_columns(df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    feature_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if col in target_cols:
            continue
        if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    if not feature_cols:
        raise ValueError("No numeric feature columns were inferred.")
    return sorted(feature_cols)


def heidke_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    numerator = 2 * (tp * tn - fp * fn)
    denominator = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-9, 1.0 - 1e-9)
    return float(log_loss(y_true, y_prob, labels=[0, 1]))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = [compute_metrics(y_true, y_prob, float(t)) for t in THRESHOLD_GRID]
    grid = pd.DataFrame(rows).sort_values(
        by=["f2", "pr_auc", "recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return float(grid.iloc[0]["threshold"]), grid


def sample_params(rng: np.random.Generator) -> dict[str, float | int]:
    return {
        "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.07])),
        "num_leaves": int(rng.choice([15, 31, 63, 127])),
        "min_child_samples": int(rng.choice([20, 50, 100, 200])),
        "subsample": float(rng.choice([0.7, 0.85, 1.0])),
        "subsample_freq": int(rng.choice([0, 1])),
        "colsample_bytree": float(rng.choice([0.7, 0.85, 1.0])),
        "reg_alpha": float(rng.choice([0.0, 0.01, 0.1, 1.0])),
        "reg_lambda": float(rng.choice([0.0, 0.01, 0.1, 1.0])),
        "max_depth": int(rng.choice([-1, 6, 10, 14])),
    }


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    require_columns(df, feature_cols + [target_col], context="split dataframe")
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    return X, y


def fit_candidate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, float | int],
    random_state: int,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=4000,
        n_jobs=-1,
        random_state=random_state,
        #class_weight="balanced",
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            # Stops training early if validation loss stalls for 100 rounds
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            # Prints validation metrics every 50 trees so you can watch progress
            lgb.log_evaluation(period=50),
        ],
    )
    return model


def save_predictions(
    df_ids: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    out_path: Path,
    target_col: str,
) -> None:
    out = df_ids.copy()
    out[target_col] = y_true.astype(int)
    out["y_prob"] = np.asarray(y_prob, dtype=float)
    out["y_pred"] = (out["y_prob"].to_numpy() >= threshold).astype(int)
    out["selected_threshold"] = float(threshold)
    out.to_parquet(out_path, index=False)


def tune_target(
    target_col: str,
    ml_root: Path,
    output_root: Path,
    n_trials: int,
    random_state: int,
) -> dict[str, object]:
    split_frames = load_all_splits_for_target(ml_root, target_col)

    train_df = split_frames["train"].copy()
    val_df = split_frames["val"].copy()
    test_df = split_frames["test"].copy()

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    feature_cols = infer_feature_columns(combined, TARGET_COLS)
    X_train, y_train = prepare_xy(train_df, feature_cols, target_col)
    X_val, y_val = prepare_xy(val_df, feature_cols, target_col)
    X_test, y_test = prepare_xy(test_df, feature_cols, target_col)


    target_root = output_root / target_col
    ensure_dir(target_root / "metrics")
    ensure_dir(target_root / "predictions")

    rng = np.random.default_rng(random_state)
    trial_rows: list[dict[str, object]] = []
    best = None

    for trial_idx in range(1, n_trials + 1):
        params = sample_params(rng)
        model = fit_candidate(X_train, y_train, X_val, y_val, params, random_state + trial_idx)

        val_prob = model.predict_proba(X_val)[:, 1]
        threshold, threshold_grid = select_threshold(y_val, val_prob)
        val_metrics = compute_metrics(y_val, val_prob, threshold)

        row = {
            "trial": trial_idx,
            **params,
            "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators)),
            "selected_threshold": threshold,
            "val_f2": val_metrics["f2"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_brier": val_metrics["brier"],
            "val_hss": val_metrics["hss"],
        }
        trial_rows.append(row)

        if best is None:
            best = {"row": row, "params": params, "model": model, "threshold_grid": threshold_grid}
        else:
            cur = (row["val_f2"], row["val_pr_auc"], -row["val_brier"])
            prev = (best["row"]["val_f2"], best["row"]["val_pr_auc"], -best["row"]["val_brier"])
            if cur > prev:
                best = {"row": row, "params": params, "model": model, "threshold_grid": threshold_grid}

    trials_df = pd.DataFrame(trial_rows).sort_values(
        by=["val_f2", "val_pr_auc", "val_hss"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    trials_df.to_csv(target_root / "tuning_trials.csv", index=False)

    best_model = best["model"]
    best_threshold = float(best["row"]["selected_threshold"])
    best_iteration = int(best["row"]["best_iteration"])

    train_prob = best_model.predict_proba(X_train)[:, 1]
    val_prob = best_model.predict_proba(X_val)[:, 1]
    test_prob = best_model.predict_proba(X_test)[:, 1]

    train_metrics = compute_metrics(y_train, train_prob, best_threshold)
    val_metrics = compute_metrics(y_val, val_prob, best_threshold)
    test_metrics = compute_metrics(y_test, test_prob, best_threshold)

    metrics_summary = pd.DataFrame(
        [
            {"target": target_col, "split": "train", **train_metrics},
            {"target": target_col, "split": "val", **val_metrics},
            {"target": target_col, "split": "test", **test_metrics},
        ]
    )
    metrics_summary.to_csv(target_root / "metrics" / "metrics_summary.csv", index=False)
    best["threshold_grid"].to_csv(target_root / "metrics" / "validation_threshold_grid.csv", index=False)

    id_cols = [c for c in ID_CANDIDATES if c in train_df.columns]

    save_predictions(
        train_df[id_cols],
        y_train,
        train_prob,
        best_threshold,
        target_root / "predictions" / "train_predictions.parquet",
        target_col,
    )
    save_predictions(
        val_df[id_cols],
        y_val,
        val_prob,
        best_threshold,
        target_root / "predictions" / "val_predictions.parquet",
        target_col,
    )
    save_predictions(
        test_df[id_cols],
        y_test,
        test_prob,
        best_threshold,
        target_root / "predictions" / "test_predictions.parquet",
        target_col,
    )

    best_params_payload = {
        "target": target_col,
        "best_params": best["params"],
        "best_iteration": best_iteration,
        "selected_threshold": best_threshold,
        "selection_metric": "validation_f2",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (target_root / "best_params.json").write_text(json.dumps(best_params_payload, indent=2, default=str))

    return {
        "target": target_col,
        "feature_count": len(feature_cols),
        "selected_threshold": best_threshold,
        "best_iteration": best_iteration,
        "train_f2": train_metrics["f2"],
        "val_f2": val_metrics["f2"],
        "test_f2": test_metrics["f2"],
        "train_pr_auc": train_metrics["pr_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "train_brier": train_metrics["brier"],
        "val_brier": val_metrics["brier"],
        "test_brier": test_metrics["brier"],
        "train_hss": train_metrics["hss"],
        "val_hss": val_metrics["hss"],
        "test_hss": test_metrics["hss"],
    }

def main() -> None:
    args = parse_args()

    if args.output_root.exists() and not args.force:
        print(f"[info] output exists: {args.output_root}")
    ensure_dir(args.output_root)

    targets = choose_targets(args.targets)

    all_rows: list[dict[str, object]] = []
    feature_count_by_target: dict[str, int] = {}

    for i, target_col in enumerate(targets):
        result = tune_target(
            target_col=target_col,
            ml_root=args.ml_root,
            output_root=args.output_root,
            n_trials=args.trials,
            random_state=args.random_state + 1000 * i,
        )
        feature_count_by_target[target_col] = int(result["feature_count"])
        all_rows.append(result)

        print(
            f"[ok] {target_col} | "
            f"features={result['feature_count']} | "
            f"best_iteration={result['best_iteration']} | "
            f"threshold={result['selected_threshold']:.2f} | "
            f"val_f2={result['val_f2']:.6f} | "
            f"test_f2={result['test_f2']:.6f}"
        )

    if not all_rows:
        raise ValueError("No targets were evaluated.")

    summary_df = pd.DataFrame(all_rows).sort_values("target").reset_index(drop=True)
    summary_df.to_csv(args.output_root / "all_targets_summary.csv", index=False)

    run_config = {
        "model": "lightgbm_tuned_simple",
        "ml_root": str(args.ml_root),
        "output_root": str(args.output_root),
        "targets": targets,
        "trials": args.trials,
        "random_state": args.random_state,
        "feature_count_by_target": feature_count_by_target,
        "split_files": DEFAULT_SPLIT_FILES,
        "search_space": {
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07],
            "num_leaves": [15, 31, 63, 127],
            "min_child_samples": [20, 50, 100, 200],
            "subsample": [0.7, 0.85, 1.0],
            "subsample_freq": [0, 1],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.0, 0.01, 0.1, 1.0],
            "max_depth": [-1, 6, 10, 14],
        },
    }
    (args.output_root / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    print("\nDone.")
    print(f"Targets evaluated: {len(targets)}")
    print(f"Artifacts: {args.output_root}")


if __name__ == "__main__":
    main()


