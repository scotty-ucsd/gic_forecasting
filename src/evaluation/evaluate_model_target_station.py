#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)

DEFAULT_OUTPUT_ROOT = Path("data/reports/evaluation/target_station")

TARGET_COLS = [
    "target_geq_p95_30m",
    "target_geq_p99_30m",
    "target_geq_p95_60m",
    "target_geq_p99_60m",
    "target_geq_p95_120m",
    "target_geq_p99_120m",
]

MODEL_SPECS = {
    "climatology": {
        "pred_root": Path("data/reports/models/climatology"),
        "splits": ["val", "test"],
        "prediction_pattern": "{target}/predictions/{split}_predictions.parquet",
    },
    "persistence": {
        "pred_root": Path("data/reports/models/persistence"),
        "splits": ["train", "val", "test"],
        "prediction_pattern": "{target}/predictions/{split}_predictions.parquet",
    },
    "logistic_regression": {
        "pred_root": Path("data/reports/models/logistic_regression"),
        "splits": ["train", "val", "test"],
        "prediction_pattern": "{target}/predictions/{split}_predictions.parquet",
    },
    "lightgbm_main": {
        "pred_root": Path("data/reports/models/lightgbm_main"),
        "splits": ["train", "val", "test"],
        "prediction_pattern": "{target}/predictions/{split}_predictions.parquet",
    },
    "lightgbm_tuned_simple": {
        "pred_root": Path("data/reports/models/lightgbm_tuned_simple"),
        "splits": ["train", "val", "test"],
        "prediction_pattern": "{target}/predictions/{split}_predictions.parquet",
    },
    "lstm": {
        "pred_root": Path("data/reports/models/lstm_classifier"),
        "splits": ["train", "val", "test"],
        "prediction_pattern": "{target}/predictions/{split}_predictions.parquet",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved predictions at the target x station x split level."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for long-format and pivoted evaluation summaries.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=f"Optional subset of models to evaluate. Choices: {sorted(MODEL_SPECS.keys())}",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional subset of canonical targets to evaluate.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional subset of splits to evaluate, e.g. train val test.",
    )
    parser.add_argument(
        "--min-rows-per-group",
        type=int,
        default=1,
        help="Minimum number of rows required to emit a station-level metric row.",
    )
    parser.add_argument(
        "--min-positive-events",
        type=int,
        default=0,
        help="Minimum number of positive events required to emit a station-level metric row.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_models(args: argparse.Namespace) -> list[str]:
    if args.models is None:
        return sorted(MODEL_SPECS.keys())
    bad = [m for m in args.models if m not in MODEL_SPECS]
    if bad:
        raise ValueError(f"Unsupported models requested: {bad}")
    return list(args.models)


def resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.targets is None:
        return list(TARGET_COLS)
    bad = [t for t in args.targets if t not in TARGET_COLS]
    if bad:
        raise ValueError(f"Unsupported targets requested: {bad}")
    return list(args.targets)


def resolve_splits(args: argparse.Namespace, model_name: str) -> list[str]:
    model_splits = MODEL_SPECS[model_name]["splits"]
    if args.splits is None:
        return list(model_splits)
    return [s for s in args.splits if s in model_splits]


def prediction_path(model_name: str, target: str, split: str) -> Path:
    spec = MODEL_SPECS[model_name]
    rel = spec["prediction_pattern"].format(target=target, split=split)
    return spec["pred_root"] / rel


def read_predictions(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def normalize_prediction_columns(
    df: pl.DataFrame,
    *,
    target: str,
    path: Path,
) -> pl.DataFrame:
    cols = set(df.columns)

    if "y_true" not in cols:
        if target in cols:
            df = df.rename({target: "y_true"})
        else:
            raise ValueError(
                f"{path}: missing true-label column. Expected either 'y_true' or target column '{target}'."
            )

    required = {"station", "y_true", "y_prob", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    return df


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def infer_threshold(y_prob: np.ndarray, y_pred: np.ndarray) -> float:
    pos_probs = y_prob[y_pred == 1]
    neg_probs = y_prob[y_pred == 0]

    lower = float(np.min(pos_probs)) if pos_probs.size > 0 else 1.0
    upper = float(np.max(neg_probs)) if neg_probs.size > 0 else 0.0

    if pos_probs.size == 0:
        thr = 1.0
    elif neg_probs.size == 0:
        thr = lower
    elif lower >= upper:
        thr = lower
    else:
        thr = (lower + upper) / 2.0

    return float(np.clip(thr, 0.0, 1.0))


def heidke_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    numerator = 2 * (tp * tn - fp * fn)
    denominator = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def compute_group_metrics(
    df: pd.DataFrame,
    *,
    model_name: str,
    target: str,
    split: str,
    station: str,
) -> dict:
    y_true = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_prob = pd.to_numeric(df["y_prob"], errors="coerce").fillna(0).astype(float).to_numpy()
    y_prob = np.clip(y_prob, 0.0, 1.0)
    y_pred = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0).astype(int).to_numpy()

    rows = int(len(df))
    positives = int(np.sum(y_true))
    threshold = infer_threshold(y_prob, y_pred)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    pr_auc = safe_pr_auc(y_true, y_prob)
    brier = float(brier_score_loss(y_true, y_prob))
    hss = heidke_skill_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "model": model_name,
        "target": target,
        "split": split,
        "station": station,
        "rows": rows,
        "n_positive": positives,
        "event_rate": float(np.mean(y_true)) if rows > 0 else float("nan"),
        "selected_threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "pr_auc": pr_auc,
        "brier": brier,
        "hss": hss,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_prediction_file(
    df: pl.DataFrame,
    *,
    model_name: str,
    target: str,
    split: str,
    min_rows_per_group: int,
    min_positive_events: int,
) -> list[dict]:
    pdf = df.to_pandas()
    pdf["station"] = pdf["station"].astype(str).str.upper()

    rows: list[dict] = []
    for station, group in pdf.groupby("station", sort=True):
        if len(group) < min_rows_per_group:
            continue
        positives = int(pd.to_numeric(group["y_true"], errors="coerce").fillna(0).astype(int).sum())
        if positives < min_positive_events:
            continue
        rows.append(
            compute_group_metrics(
                group,
                model_name=model_name,
                target=target,
                split=split,
                station=station,
            )
        )
    return rows


def build_pivot(long_df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "rows",
        "n_positive",
        "event_rate",
        "selected_threshold",
        "precision",
        "recall",
        "f2",
        "pr_auc",
        "brier",
        "hss",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    pivot = long_df.pivot_table(
        index=["target", "station"],
        columns=["model", "split"],
        values=value_cols,
        aggfunc="first",
    )

    pivot.columns = [
        f"{metric}__{model}__{split}"
        for metric, model, split in pivot.columns.to_flat_index()
    ]
    pivot = pivot.reset_index().sort_values(["target", "station"]).reset_index(drop=True)
    return pivot


def main() -> None:
    args = parse_args()

    if args.output_root.exists() and not args.force:
        print(f"[info] output exists: {args.output_root}")
    ensure_dir(args.output_root)

    models = resolve_models(args)
    targets = resolve_targets(args)

    all_rows: list[dict] = []
    missing_files: list[str] = []

    for model_name in models:
        splits = resolve_splits(args, model_name)

        for target in targets:
            for split in splits:
                path = prediction_path(model_name, target, split)
                if not path.exists():
                    missing_files.append(str(path))
                    print(f"[warn] missing predictions: {path}")
                    continue

                df = read_predictions(path)
                df = normalize_prediction_columns(df, target=target, path=path)
                rows = evaluate_prediction_file(
                    df,
                    model_name=model_name,
                    target=target,
                    split=split,
                    min_rows_per_group=args.min_rows_per_group,
                    min_positive_events=args.min_positive_events,
                )

                all_rows.extend(rows)

                print(
                    f"[ok] model={model_name} | target={target} | split={split} | "
                    f"stations_evaluated={len(rows)}"
                )

    if not all_rows:
        raise ValueError("No evaluation rows were produced. Check prediction paths and filtering thresholds.")

    long_df = pd.DataFrame(all_rows).sort_values(
        ["model", "target", "split", "station"]
    ).reset_index(drop=True)

    pivot_df = build_pivot(long_df)

    long_path = args.output_root / "target_station_metrics_long.csv"
    pivot_path = args.output_root / "target_station_metrics_pivot.csv"
    run_config_path = args.output_root / "run_config.json"

    long_df.to_csv(long_path, index=False)
    pivot_df.to_csv(pivot_path, index=False)

    run_config = {
        "models": models,
        "targets": targets,
        "requested_splits": args.splits,
        "min_rows_per_group": args.min_rows_per_group,
        "min_positive_events": args.min_positive_events,
        "missing_prediction_files": missing_files,
        "model_specs": {
            k: {
                "pred_root": str(v["pred_root"]),
                "splits": v["splits"],
                "prediction_pattern": v["prediction_pattern"],
            }
            for k, v in MODEL_SPECS.items()
        },
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, default=str))

    print("\nDone.")
    print(f"Rows written (long): {len(long_df)}")
    print(f"Artifacts: {args.output_root}")


if __name__ == "__main__":
    main()



