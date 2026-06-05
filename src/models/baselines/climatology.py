#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
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

DEFAULT_DATA_ROOT = Path("data/ml/standardized")
DEFAULT_OUTPUT_ROOT = Path("data/reports/models/climatology")

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
        description="Climatology baseline for standardized binary classification targets."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional subset of targets to evaluate.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_split(task_dir: Path, split: str) -> pl.DataFrame:
    path = task_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def choose_targets(requested: list[str] | None) -> list[str]:
    if not requested:
        return TARGET_COLS
    bad = sorted(set(requested) - set(TARGET_COLS))
    if bad:
        raise ValueError(f"Unknown targets requested: {bad}")
    return requested


def heidke_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    numerator = 2 * (tp * tn - fp * fn)
    denominator = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-9, 1.0 - 1e-9)
    return float(log_loss(y_true, y_prob, labels=[0, 1]))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
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
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": safe_log_loss(y_true, y_prob),
        "hss": float(heidke_skill_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric_name: str = "f2") -> tuple[float, pd.DataFrame]:
    rows = []
    for thr in THRESHOLD_GRID:
        rows.append(compute_classification_metrics(y_true, y_prob, float(thr)))
    grid_df = pd.DataFrame(rows).sort_values(
        by=[metric_name, "pr_auc", "recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return float(grid_df.iloc[0]["threshold"]), grid_df


def save_classification_predictions(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    out_path: Path,
    target_col: str,
) -> None:
    out = df.copy()
    out[target_col] = y_true.astype(int)
    out["y_prob"] = y_prob.astype(float)
    out["y_pred"] = (y_prob >= threshold).astype(int)
    out.to_parquet(out_path, index=False)


def evaluate_target(
    data_root: Path,
    output_root: Path,
    target_col: str,
) -> dict:
    target_dir = data_root / target_col
    out_root = output_root / target_col
    metrics_dir = out_root / "metrics"
    preds_dir = out_root / "predictions"
    ensure_dir(metrics_dir)
    ensure_dir(preds_dir)

    train_df = read_split(target_dir, "train").to_pandas()
    val_df = read_split(target_dir, "val").to_pandas()
    test_df = read_split(target_dir, "test").to_pandas()

    train_prob = float(pd.to_numeric(train_df[target_col], errors="coerce").mean())

    y_val = pd.to_numeric(val_df[target_col], errors="coerce").astype(int).to_numpy()
    y_test = pd.to_numeric(test_df[target_col], errors="coerce").astype(int).to_numpy()

    val_prob = np.full(len(val_df), train_prob, dtype=float)
    test_prob = np.full(len(test_df), train_prob, dtype=float)

    best_threshold, threshold_grid = select_threshold(y_val, val_prob, metric_name="f2")
    val_metrics = compute_classification_metrics(y_val, val_prob, best_threshold)
    test_metrics = compute_classification_metrics(y_test, test_prob, best_threshold)

    save_classification_predictions(
        val_df[[c for c in ID_COLS if c in val_df.columns]],
        y_val,
        val_prob,
        best_threshold,
        preds_dir / "val_predictions.parquet",
        target_col,
    )
    save_classification_predictions(
        test_df[[c for c in ID_COLS if c in test_df.columns]],
        y_test,
        test_prob,
        best_threshold,
        preds_dir / "test_predictions.parquet",
        target_col,
    )

    threshold_grid.to_csv(metrics_dir / "validation_threshold_grid.csv", index=False)

    summary_df = pd.concat(
        [
            pd.DataFrame([{"target": target_col, "split": "val", **val_metrics}]),
            pd.DataFrame([{"target": target_col, "split": "test", **test_metrics}]),
        ],
        ignore_index=True,
    )
    summary_df.to_csv(metrics_dir / "metrics_summary.csv", index=False)

    config = {
        "model": "climatology",
        "target": target_col,
        "train_event_rate": train_prob,
        "selected_threshold": best_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (out_root / "model_config.json").write_text(json.dumps(config, indent=2, default=str))

    return {
        "target": target_col,
        "train_event_rate": train_prob,
        "selected_threshold": best_threshold,
        "val_f2": val_metrics["f2"],
        "test_f2": test_metrics["f2"],
        "val_pr_auc": val_metrics["pr_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "val_brier": val_metrics["brier"],
        "test_brier": test_metrics["brier"],
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)

    targets = choose_targets(args.targets)
    rows = []

    for target_col in targets:
        result = evaluate_target(
            data_root=args.data_root,
            output_root=args.output_root,
            target_col=target_col,
        )
        rows.append(result)
        print(
            f"[ok] {target_col} | "
            f"train_event_rate={result['train_event_rate']:.6f} | "
            f"threshold={result['selected_threshold']:.2f} | "
            f"val_f2={result['val_f2']:.6f} | "
            f"test_f2={result['test_f2']:.6f}"
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.output_root / "all_targets_summary.csv", index=False)

    run_config = {
        "model": "climatology",
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "targets": targets,
        "threshold_grid": THRESHOLD_GRID.tolist(),
    }
    (args.output_root / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    print("\nDone.")
    print(f"Targets evaluated: {len(targets)}")
    print(f"Artifacts: {args.output_root}")


if __name__ == "__main__":
    main()


