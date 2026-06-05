#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA_ROOT = Path("data/ml/standardized")
DEFAULT_OUTPUT_ROOT = Path("data/reports/models/logistic_regression")
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
        description="Train logistic regression classifiers on standardized six-target station-time datasets."
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
        help="Directory for saved models, predictions, metrics, and configs.",
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
        help="Optional CSV containing reduced candidate features. The script will use a 'feature' column if present.",
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
        default=2_000_000,
        help="Optional cap on training rows for faster fitting. Uses current parquet ordering.",
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
        "--c-grid",
        nargs="*",
        type=float,
        default=[0.01, 0.1, 1.0, 3.0],
        help="Grid of inverse regularization strengths C to search.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "f2", "balanced_accuracy"],
        default="f2",
        help="Metric used on validation set to choose classification threshold.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        help="scikit-learn LogisticRegression solver.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        help="Class weight strategy, e.g. 'balanced' or 'none'.",
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

def build_pipeline(C: float, solver: str, class_weight: str | None) -> Pipeline:
    cw = None if class_weight in ("none", "None", "", None) else class_weight
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression( # using l2 regularization default
                    C=C,
                    solver=solver,
                    class_weight=cw,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )



def fit_model_grid(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    c_grid: list[float],
    solver: str,
    class_weight: str,
    threshold_metric: str,
) -> tuple[Pipeline, dict, pd.DataFrame]:
    trial_rows: list[dict[str, float]] = []
    best = None
    best_pipe = None

    for C in c_grid:
        pipe = build_pipeline(C=C, solver=solver, class_weight=class_weight)
        pipe.fit(X_train, y_train)

        val_prob = pipe.predict_proba(X_val)[:, 1]
        best_thr, thr_df = select_threshold(y_val, val_prob, threshold_metric)
        val_metrics = compute_metrics(y_val, val_prob, best_thr)

        row = {
            "C": float(C),
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
                "C": float(C),
                "threshold": float(best_thr),
                "val_metrics": val_metrics,
                "threshold_grid": thr_df,
            }
            best_pipe = pipe

    trial_df = pd.DataFrame(trial_rows).sort_values(
        by=[f"val_{threshold_metric}", "val_pr_auc", "val_hss"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    assert best is not None and best_pipe is not None
    return best_pipe, best, trial_df


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


def save_coefficients(
    pipe: Pipeline,
    feature_cols: list[str],
    out_path: Path,
) -> pd.DataFrame:
    model: LogisticRegression = pipe.named_steps["model"]
    coef = model.coef_.ravel()
    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
            "direction": np.where(coef >= 0, "positive", "negative"),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(out_path, index=False)
    return coef_df


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

        model, best, trial_df = fit_model_grid(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            c_grid=args.c_grid,
            solver=args.solver,
            class_weight=args.class_weight,
            threshold_metric=args.threshold_metric,
        )

        best_threshold = float(best["threshold"])
        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        train_metrics = compute_metrics(y_train, train_prob, best_threshold)
        val_metrics = compute_metrics(y_val, val_prob, best_threshold)
        test_metrics = compute_metrics(y_test, test_prob, best_threshold)

        save_predictions(meta_train, y_train, train_prob, best_threshold, preds_dir / "train_predictions.parquet")
        save_predictions(meta_val, y_val, val_prob, best_threshold, preds_dir / "val_predictions.parquet")
        save_predictions(meta_test, y_test, test_prob, best_threshold, preds_dir / "test_predictions.parquet")

        coef_df = save_coefficients(model, feature_cols, metrics_dir / "coefficients.csv")
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

        model_config = {
            "model": "logistic_regression",
            "target_col": target_col,
            "feature_view": feature_view,
            "feature_count": len(feature_cols),
            "features": feature_cols,
            "selected_C": float(best["C"]),
            "selected_threshold": best_threshold,
            "threshold_metric": args.threshold_metric,
            "solver": args.solver,
            "class_weight": args.class_weight,
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "max_test_rows": args.max_test_rows,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "top_coefficients": coef_df.head(25).to_dict(orient="records"),
        }
        (model_dir / "model_config.json").write_text(json.dumps(model_config, indent=2, default=str))

        all_summary_rows.append(
            {
                "target": target_col,
                "feature_view": feature_view,
                "feature_count": len(feature_cols),
                "selected_C": float(best["C"]),
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
            }
        )

        print(
            f"[ok] {target_col} | feature_view={feature_view} | "
            f"features={len(feature_cols)} | C={best['C']} | threshold={best_threshold:.2f} | "
            f"val_f2={val_metrics['f2']:.6f} | test_f2={test_metrics['f2']:.6f}"
        )

        del meta_train  # quiet unused-variable concern in some editors

    summary_out = pd.DataFrame(all_summary_rows).sort_values("target").reset_index(drop=True)
    summary_out.to_csv(args.output_root / "summary_metrics.csv", index=False)

    run_config = {
        "model": "logistic_regression",
        "data_root": str(args.data_root),
        "targets": targets,
        "feature_columns_json": str(feature_json),
        "reduced_features_json": str(args.reduced_features_json) if args.reduced_features_json else None,
        "reduced_features_csv": str(args.reduced_features_csv) if args.reduced_features_csv else None,
        "top_k_features": args.top_k_features,
        "c_grid": args.c_grid,
        "threshold_metric": args.threshold_metric,
        "solver": args.solver,
        "class_weight": args.class_weight,
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "max_test_rows": args.max_test_rows,
    }
    (args.output_root / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    print("\nDone.")
    print(f"Targets evaluated: {len(targets)}")
    print(f"Artifacts: {args.output_root}")


if __name__ == "__main__":
    main()


