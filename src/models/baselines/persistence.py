#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

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

DEFAULT_STANDARDIZED_ROOT = Path("data/preprocessed/fused/standardized")
DEFAULT_SPLIT_CONFIG = Path("data/ml/standardized/metadata/split_config.json")
DEFAULT_OUTPUT_ROOT = Path("data/reports/models/persistence")

TARGET_COLS = [
    "target_geq_p95_30m",
    "target_geq_p99_30m",
    "target_geq_p95_60m",
    "target_geq_p99_60m",
    "target_geq_p95_120m",
    "target_geq_p99_120m",
]

DEFAULT_GLOB = "*/station_time_features_*.parquet"
ID_COLS = ["timestamp", "station", "year", "month", "yyyymm", "split"]

DBN_COL = "supermag_dbn_nt"
DBE_COL = "supermag_dbe_nt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict observation-persistence baseline using standardized fused feature parquet files and split_config.json."
    )
    parser.add_argument("--standardized-root", type=Path, default=DEFAULT_STANDARDIZED_ROOT)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default=DEFAULT_GLOB,
        help="Glob under standardized-root used to discover monthly parquet files.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional subset of targets to evaluate.",
    )
    parser.add_argument(
        "--station",
        type=str,
        default=None,
        help="Optional single-station filter.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of years to include before split assignment.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_targets(requested: list[str] | None) -> list[str]:
    if not requested:
        return list(TARGET_COLS)
    bad = sorted(set(requested) - set(TARGET_COLS))
    if bad:
        raise ValueError(f"Unknown targets requested: {bad}")
    return list(requested)


def heidke_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    numerator = 2 * (tp * tn - fp * fn)
    denominator = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


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
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
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


def parse_target_spec(target_col: str) -> tuple[float, int]:
    parts = target_col.split("_")
    percentile_token = next((p for p in parts if p.startswith("p")), None)
    horizon_token = next((p for p in parts if p.endswith("m")), None)
    if percentile_token is None or horizon_token is None:
        raise ValueError(f"Could not parse target specification from '{target_col}'")
    percentile = float(percentile_token[1:]) / 100.0
    horizon_minutes = int(horizon_token[:-1])
    return percentile, horizon_minutes


def discover_files(root: Path, glob_pattern: str) -> list[Path]:
    paths = sorted(root.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root} with pattern '{glob_pattern}'")
    return paths


def load_split_config(path: Path) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    strategy = payload.get("strategy")
    if strategy != "year":
        raise ValueError(f"Unsupported split strategy '{strategy}'. Expected 'year'.")
    year_assignments = payload.get("year_assignments", {})
    if not year_assignments:
        raise ValueError("split_config.json is missing year_assignments.")
    return {int(k): str(v) for k, v in year_assignments.items()}


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def load_standardized_frames(
    standardized_root: Path,
    glob_pattern: str,
    station: str | None,
    years: list[int] | None,
    split_map: dict[int, str],
) -> pd.DataFrame:
    paths = discover_files(standardized_root, glob_pattern)
    frames: list[pd.DataFrame] = []

    for path in paths:
        df = pd.read_parquet(path)

        require_columns(
            df,
            ["timestamp", "station", "year", DBN_COL, DBE_COL] + TARGET_COLS,
            context=str(path),
        )

        if station is not None:
            df = df.loc[df["station"] == station].copy()
        if years is not None:
            df = df.loc[df["year"].isin(years)].copy()

        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df.loc[df["year"].notna()].copy()
        df["year"] = df["year"].astype(int)

        df["split"] = df["year"].map(split_map)
        df = df.loc[df["split"].notna()].copy()

        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("No rows available after applying station/year filters and split assignment.")

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False, errors="coerce")
    out = out.sort_values(["timestamp", "station"], kind="stable").reset_index(drop=True)
    return out


def build_horizontal_perturbation_proxy(df: pd.DataFrame) -> np.ndarray:
    dbn = pd.to_numeric(df[DBN_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dbe = pd.to_numeric(df[DBE_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.sqrt(dbn**2 + dbe**2)


def build_strict_persistence_outputs(
    source_values: np.ndarray,
    current_event_threshold: float,
) -> np.ndarray:
    src = np.nan_to_num(np.asarray(source_values, dtype=float), nan=0.0)
    return (src >= float(current_event_threshold)).astype(float)


def save_predictions(
    df_ids: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    source_values: np.ndarray,
    out_path: Path,
    target_col: str,
    source_name: str,
    current_event_threshold: float,
) -> None:
    out = df_ids.copy()
    out[target_col] = np.asarray(y_true, dtype=int)
    out[source_name] = np.asarray(source_values, dtype=float)
    out["current_event_threshold"] = float(current_event_threshold)
    out["y_prob"] = np.asarray(y_prob, dtype=float)
    out["y_pred"] = (out["y_prob"].to_numpy() >= threshold).astype(int)
    out.to_parquet(out_path, index=False)


def evaluate_target(
    standardized_df: pd.DataFrame,
    output_root: Path,
    target_col: str,
    force: bool,
) -> dict[str, object]:
    out_root = output_root / target_col
    if out_root.exists() and not force:
        raise FileExistsError(f"Output path already exists: {out_root}. Use --force to overwrite.")

    metrics_dir = out_root / "metrics"
    preds_dir = out_root / "predictions"
    ensure_dir(metrics_dir)
    ensure_dir(preds_dir)

    require_columns(
        standardized_df,
        ["split", target_col, DBN_COL, DBE_COL, "timestamp", "station", "year"],
        context=f"standardized dataframe for {target_col}",
    )

    train_df = standardized_df.loc[standardized_df["split"] == "train"].copy()
    val_df = standardized_df.loc[standardized_df["split"] == "val"].copy()
    test_df = standardized_df.loc[standardized_df["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Target {target_col} requires non-empty train/val/test splits after year assignment."
        )

    y_train = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y_val = pd.to_numeric(val_df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y_test = pd.to_numeric(test_df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()

    src_train = build_horizontal_perturbation_proxy(train_df)
    src_val = build_horizontal_perturbation_proxy(val_df)
    src_test = build_horizontal_perturbation_proxy(test_df)

    percentile, horizon_minutes = parse_target_spec(target_col)
    finite_train = src_train[np.isfinite(src_train)]
    if finite_train.size == 0:
        raise ValueError(f"No finite train values available for persistence source on {target_col}.")

    current_event_threshold = float(np.quantile(finite_train, percentile))

    train_prob = build_strict_persistence_outputs(src_train, current_event_threshold)
    val_prob = build_strict_persistence_outputs(src_val, current_event_threshold)
    test_prob = build_strict_persistence_outputs(src_test, current_event_threshold)

    fixed_threshold = 0.5
    train_metrics = compute_classification_metrics(y_train, train_prob, fixed_threshold)
    val_metrics = compute_classification_metrics(y_val, val_prob, fixed_threshold)
    test_metrics = compute_classification_metrics(y_test, test_prob, fixed_threshold)

    id_keep = [c for c in ID_COLS if c in standardized_df.columns]
    source_name = "supermag_horizontal_perturbation_proxy"

    save_predictions(
        train_df[id_keep],
        y_train,
        train_prob,
        fixed_threshold,
        src_train,
        preds_dir / "train_predictions.parquet",
        target_col,
        source_name,
        current_event_threshold,
    )
    save_predictions(
        val_df[id_keep],
        y_val,
        val_prob,
        fixed_threshold,
        src_val,
        preds_dir / "val_predictions.parquet",
        target_col,
        source_name,
        current_event_threshold,
    )
    save_predictions(
        test_df[id_keep],
        y_test,
        test_prob,
        fixed_threshold,
        src_test,
        preds_dir / "test_predictions.parquet",
        target_col,
        source_name,
        current_event_threshold,
    )

    threshold_summary = pd.DataFrame(
        [
            {
                "target": target_col,
                "horizon_minutes": horizon_minutes,
                "percentile": percentile,
                "current_event_threshold": current_event_threshold,
                "prediction_threshold": fixed_threshold,
                "source_definition": "sqrt(supermag_dbn_nt^2 + supermag_dbe_nt^2)",
                "prediction_rule": "strict persistence",
            }
        ]
    )
    threshold_summary.to_csv(metrics_dir / "fixed_threshold_summary.csv", index=False)

    summary_df = pd.concat(
        [
            pd.DataFrame([{"target": target_col, "split": "train", **train_metrics}]),
            pd.DataFrame([{"target": target_col, "split": "val", **val_metrics}]),
            pd.DataFrame([{"target": target_col, "split": "test", **test_metrics}]),
        ],
        ignore_index=True,
    )
    summary_df.to_csv(metrics_dir / "metrics_summary.csv", index=False)

    station_split_summary = (
        pd.concat(
            [
                train_df.assign(_split="train"),
                val_df.assign(_split="val"),
                test_df.assign(_split="test"),
            ],
            ignore_index=True,
        )
        .assign(supermag_horizontal_perturbation_proxy=lambda d: build_horizontal_perturbation_proxy(d))
        .groupby(["station", "_split"], dropna=False)
        .agg(
            rows=(target_col, "size"),
            event_rate=(target_col, "mean"),
            source_mean=("supermag_horizontal_perturbation_proxy", "mean"),
            source_std=("supermag_horizontal_perturbation_proxy", "std"),
        )
        .reset_index()
        .rename(columns={"_split": "split"})
    )
    station_split_summary.to_csv(metrics_dir / "station_split_summary.csv", index=False)

    config = {
        "model": "persistence",
        "target": target_col,
        "source_definition": "sqrt(supermag_dbn_nt^2 + supermag_dbe_nt^2)",
        "prediction_rule": "strict persistence: y_pred = 1 if source >= current_event_threshold else 0",
        "current_event_threshold": current_event_threshold,
        "prediction_threshold": fixed_threshold,
        "target_percentile": percentile,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (out_root / "model_config.json").write_text(json.dumps(config, indent=2, default=str))

    return {
        "target": target_col,
        "source_definition": "supermag_horizontal_perturbation_proxy",
        "prediction_rule": "strict persistence",
        "current_event_threshold": current_event_threshold,
        "selected_threshold": fixed_threshold,
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
    ensure_dir(args.output_root)

    split_map = load_split_config(args.split_config)
    targets = choose_targets(args.targets)

    standardized_df = load_standardized_frames(
        standardized_root=args.standardized_root,
        glob_pattern=args.glob_pattern,
        station=args.station,
        years=args.years,
        split_map=split_map,
    )

    rows: list[dict[str, object]] = []
    for target_col in targets:
        result = evaluate_target(
            standardized_df=standardized_df,
            output_root=args.output_root,
            target_col=target_col,
            force=args.force,
        )
        rows.append(result)
        print(
            f"[ok] {target_col} | "
            f"source={result['source_definition']} | "
            f"rule={result['prediction_rule']} | "
            f"curr_thr={result['current_event_threshold']:.6f} | "
            f"pred_thr={result['selected_threshold']:.2f} | "
            f"val_f2={result['val_f2']:.6f} | "
            f"test_f2={result['test_f2']:.6f}"
        )

    summary_df = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)
    summary_df.to_csv(args.output_root / "all_targets_summary.csv", index=False)

    run_config = {
        "model": "persistence",
        "standardized_root": str(args.standardized_root),
        "split_config": str(args.split_config),
        "glob_pattern": args.glob_pattern,
        "output_root": str(args.output_root),
        "targets": targets,
        "station": args.station,
        "years": args.years,
        "source_definition": "sqrt(supermag_dbn_nt^2 + supermag_dbe_nt^2)",
        "prediction_rule": "strict persistence: y_pred = 1 if source >= current_event_threshold else 0",
        "prediction_threshold": 0.5,
    }
    (args.output_root / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    print("\nDone.")
    print(f"Targets evaluated: {len(targets)}")
    print(f"Artifacts: {args.output_root}")


if __name__ == "__main__":
    main()


