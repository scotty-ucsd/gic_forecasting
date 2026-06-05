#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATA_ROOT = Path("data/ml/standardized")
DEFAULT_OUTPUT_ROOT = Path("data/reports/models/lstm_classifier")
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
        description="Train compact PyTorch LSTM classifiers with optimized lazy sequence construction."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--feature-columns-json", type=Path, default=None)
    parser.add_argument("--reduced-features-json", type=Path, default=None)
    parser.add_argument("--reduced-features-csv", type=Path, default=None)
    parser.add_argument("--top-k-features", type=int, default=20)
    parser.add_argument("--lookback-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--layer-sizes", type=str, default=None,
        help="Comma-separated hidden sizes per LSTM layer, e.g. '32,24,16,8'. "
             "Overrides --hidden-size and --num-layers when set. "
             "Encodes physics chain: Sun(input) -> L1 -> GEO -> LEO -> Ground(output)."
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--threshold-metric", choices=["f1", "f2", "balanced_accuracy"], default="f2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_threads(cpu_threads: int, interop_threads: int) -> None:
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.targets:
        bad = [t for t in args.targets if t not in TARGET_COLS]
        if bad:
            raise ValueError(f"Unsupported targets requested: {bad}")
        return list(args.targets)
    return list(TARGET_COLS)


def load_feature_columns(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "feature_columns" in data:
        cols = data["feature_columns"]
    elif isinstance(data, list):
        cols = data
    else:
        raise ValueError(f"Unsupported feature columns JSON format: {path}")
    return list(cols)


def load_reduced_feature_columns(args: argparse.Namespace) -> list[str] | None:
    cols: list[str] | None = None
    if args.reduced_features_json is not None:
        data = json.loads(args.reduced_features_json.read_text())
        if isinstance(data, dict):
            cols = list(data.get("feature_columns", data.get("features", [])))
        elif isinstance(data, list):
            cols = list(data)
    elif args.reduced_features_csv is not None and args.reduced_features_csv.exists():
        df = pd.read_csv(args.reduced_features_csv)
        cols = df["feature"].dropna().astype(str).tolist() if "feature" in df.columns else df.iloc[:, 0].dropna().astype(str).tolist()

    if cols is None:
        return None

    seen = set()
    deduped = [c for c in cols if not (c in seen or seen.add(c))]
    return deduped[:args.top_k_features] if args.top_k_features else deduped


def read_split(path: Path, n_rows: int | None = None) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path).head(n_rows) if n_rows is not None else pl.read_parquet(path)


def to_optimized_xy(
    df: pl.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    df_sorted = (
        df.with_columns(pl.col("station").str.to_uppercase())
        .sort(["station", "timestamp"])
    )

    X = df_sorted.select(feature_cols).to_numpy().astype(np.float32)
    y = df_sorted.select(target_col).fill_null(0).cast(pl.Int8).to_numpy().squeeze()

    meta_cols = [c for c in ID_COLS if c in df_sorted.columns]
    meta_pd = df_sorted.select(meta_cols).to_pandas()
    stations_array = df_sorted.select("station").to_numpy().squeeze().astype(str)

    return X, y, meta_pd, stations_array


def fit_and_transform_numpy(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Vectorized computation of medians
    medians = np.nanmedian(X_train, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)

    # Memory-efficient column-wise imputation
    for col_idx in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, col_idx]), col_idx] = medians[col_idx]
        X_val[np.isnan(X_val[:, col_idx]), col_idx] = medians[col_idx]
        X_test[np.isnan(X_test[:, col_idx]), col_idx] = medians[col_idx]

    # Fully vectorized, in-place standardization
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0) + 1e-8

    X_train -= means
    X_train /= stds

    X_val -= means
    X_val /= stds

    X_test -= means
    X_test /= stds

    return X_train, X_val, X_test


def build_endpoint_arrays_vectorized(stations: np.ndarray, lookback_steps: int) -> tuple[np.ndarray, np.ndarray]:
    if len(stations) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    change_mask = stations[1:] != stations[:-1]
    split_indices = np.where(change_mask)[0] + 1
    station_blocks = np.split(np.arange(len(stations)), split_indices)

    starts_list = []
    ends_list = []

    for block in station_blocks:
        length = len(block)
        if length < lookback_steps:
            continue
        valid_sequences = length - lookback_steps + 1
        range_arr = np.arange(valid_sequences)

        starts_list.append(block[0] + range_arr)
        ends_list.append(block[0] + range_arr + lookback_steps - 1)

    if not starts_list:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    return np.concatenate(starts_list), np.concatenate(ends_list)


class LazyStationSequenceDataset(Dataset):
    def __init__(self, X_rows: np.ndarray, y_rows: np.ndarray, meta_rows: pd.DataFrame, starts: np.ndarray, ends: np.ndarray) -> None:
        self.X_rows = X_rows
        self.y_rows = y_rows.astype(np.float32)
        self.meta_rows = meta_rows
        self.start_indices = starts
        self.end_indices = ends

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.start_indices[idx]
        end = self.end_indices[idx]
        return torch.from_numpy(self.X_rows[start : end + 1]), torch.tensor(self.y_rows[end], dtype=torch.float32)

    def get_metadata_frame(self) -> pd.DataFrame:
        return self.meta_rows.iloc[self.end_indices].reset_index(drop=True) if len(self.end_indices) > 0 else self.meta_rows.iloc[0:0].copy()

    def get_labels(self) -> np.ndarray:
        return self.y_rows[self.end_indices].astype(np.int8) if len(self.end_indices) > 0 else np.empty((0,), dtype=np.int8)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float,
                 layer_sizes: list[int] | None = None) -> None:
        """
        Physics-informed LSTM classifier.

        When layer_sizes is provided, each LSTM layer has its own hidden size,
        encoding the causal propagation chain:
            Sun (input) -> L1 -> GEO -> LEO -> Ground (output/logit)

        Example: layer_sizes=[32, 24, 16, 8]
            Layer 0: input_size -> 32   (L1 proxy)
            Layer 1: 32 -> 24           (GEO proxy)
            Layer 2: 24 -> 16           (LEO proxy)
            Layer 3: 16 -> 8            (near-Ground)
            Head:    8 -> 1             (GIC logit)

        When layer_sizes is None, falls back to uniform hidden_size x num_layers.
        """
        super().__init__()
        self.dropout_p = dropout

        # Resolve layer size list
        sizes: list[int] = layer_sizes if layer_sizes is not None else [hidden_size] * num_layers

        # Stack individual single-layer LSTMs so each can have its own hidden size
        self.lstm_layers = nn.ModuleList()
        in_size = input_size
        for h in sizes:
            self.lstm_layers.append(
                nn.LSTM(input_size=in_size, hidden_size=h, num_layers=1, batch_first=True)
            )
            in_size = h

        self.dropout = nn.Dropout(dropout)
        final_hidden = sizes[-1]
        self.head = nn.Sequential(
            nn.Linear(final_hidden, max(final_hidden // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(final_hidden // 2, 1), 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(out)
            # Apply dropout between layers but not after the last one
            if i < len(self.lstm_layers) - 1:
                out = self.dropout(out)
        last_hidden = self.dropout(out[:, -1, :])
        return self.head(last_hidden).squeeze(-1)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    denominator = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    hss = float(2 * (tp * tn - fp * fn) / denominator) if denominator != 0 else float("nan")

    has_two_classes = len(np.unique(y_true)) >= 2
    y_prob_clipped = np.clip(y_prob, 1e-8, 1 - 1e-8)

    return {
        "threshold": float(threshold), "rows": int(len(y_true)),
        "event_rate": float(np.mean(y_true)), "pred_rate": float(np.mean(y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_two_classes else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob_clipped, labels=[0, 1])),
        "hss": hss, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric_name: str) -> tuple[float, pd.DataFrame]:
    rows = [compute_metrics(y_true, y_prob, float(thr)) for thr in THRESHOLD_GRID]
    grid_df = pd.DataFrame(rows).sort_values(by=[metric_name, "pr_auc", "recall"], ascending=[False, False, False]).reset_index(drop=True)
    return float(grid_df.iloc[0]["threshold"]), grid_df


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_count = 0.0, 0

    # Determine execution context for mixed precision
    amp_context = torch.autocast(device_type=device.type) if device.type in ["cuda", "mps"] else contextlib.nullcontext()

    # Use inference mode when evaluating
    grad_context = contextlib.nullcontext() if is_train else torch.inference_mode()

    with grad_context:
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with amp_context:
                logits = model(xb)
                loss = criterion(logits, yb)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            n = yb.shape[0]
            total_loss += float(loss.item()) * n
            total_count += n

    return total_loss / max(total_count, 1)


@torch.inference_mode()
def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    amp_context = torch.autocast(device_type=device.type) if device.type in ["cuda", "mps"] else contextlib.nullcontext()

    probs = []
    with amp_context:
        for xb, _ in loader:
            logits = model(xb.to(device, non_blocking=True))
            probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probs).astype(np.float32)


def save_predictions(meta: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, threshold: float, out_path: Path, target_col: str) -> None:
    pred_df = meta.copy()
    pred_df[target_col] = y_true.astype(int)
    pred_df["y_prob"] = y_prob.astype(float)
    pred_df["y_pred"] = (y_prob >= threshold).astype(int)
    pred_df.to_parquet(out_path, index=False)


def parse_layer_sizes(layer_sizes_str: str | None) -> list[int] | None:
    """Parse '--layer-sizes 32,24,16,8' into [32, 24, 16, 8], or None if not set."""
    if layer_sizes_str is None:
        return None
    try:
        sizes = [int(s.strip()) for s in layer_sizes_str.split(",")]
    except ValueError:
        raise ValueError(f"--layer-sizes must be comma-separated integers, got: {layer_sizes_str!r}")
    if any(s < 1 for s in sizes):
        raise ValueError(f"All layer sizes must be >= 1, got: {sizes}")
    return sizes


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_torch_threads(args.cpu_threads, args.interop_threads)
    device = resolve_device(args.device)

    all_feature_cols = load_feature_columns(args.feature_columns_json or (args.data_root / DEFAULT_METADATA_DIRNAME / "feature_columns.json"))
    reduced_feature_cols = load_reduced_feature_columns(args)
    targets = resolve_targets(args)
    ensure_dir(args.output_root)
    all_summary_rows = []

    # Initialize Automatic Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for target_col in targets:
        output_root = args.output_root / target_col
        model_dir, metrics_dir, preds_dir = output_root / "artifacts", output_root / "metrics", output_root / "predictions"
        if output_root.exists() and not args.force:
            print(f"[info] Skipping existing output: {output_root}")
            continue

        ensure_dir(model_dir); ensure_dir(metrics_dir); ensure_dir(preds_dir)

        train_df = read_split(args.data_root / target_col / TRAIN_FILE, args.max_train_rows)
        val_df = read_split(args.data_root / target_col / VAL_FILE, args.max_val_rows)
        test_df = read_split(args.data_root / target_col / TEST_FILE, args.max_test_rows)

        feature_cols = [c for c in (reduced_feature_cols if reduced_feature_cols is not None else all_feature_cols) if c in train_df.columns]

        X_tr, y_tr, m_tr, s_tr = to_optimized_xy(train_df, target_col, feature_cols)
        X_va, y_va, m_va, s_va = to_optimized_xy(val_df, target_col, feature_cols)
        X_te, y_te, m_te, s_te = to_optimized_xy(test_df, target_col, feature_cols)
        del train_df, val_df, test_df

        X_tr, X_va, X_te = fit_and_transform_numpy(X_tr, X_va, X_te)

        starts_tr, ends_tr = build_endpoint_arrays_vectorized(s_tr, args.lookback_steps)
        starts_va, ends_va = build_endpoint_arrays_vectorized(s_va, args.lookback_steps)
        starts_te, ends_te = build_endpoint_arrays_vectorized(s_te, args.lookback_steps)

        train_ds = LazyStationSequenceDataset(X_tr, y_tr, m_tr, starts_tr, ends_tr)
        val_ds = LazyStationSequenceDataset(X_va, y_va, m_va, starts_va, ends_va)
        test_ds = LazyStationSequenceDataset(X_te, y_te, m_te, starts_te, ends_te)

        y_train_seq, y_val_seq, y_test_seq = train_ds.get_labels(), val_ds.get_labels(), test_ds.get_labels()

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=False)

        model = LSTMClassifier(
            input_size=X_tr.shape[1],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            layer_sizes=parse_layer_sizes(args.layer_sizes),
        ).to(device)

        pos_ratio = float(np.sum(y_train_seq == 0)) / float(max(np.sum(y_train_seq == 1), 1))
        pos_weight = torch.tensor([math.sqrt(pos_ratio)], dtype=torch.float32, device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        history_rows = []
        best_state = None
        best_val_metric = -math.inf
        patience_left = args.patience

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
            val_loss = run_epoch(model, val_loader, criterion, None, device, scaler=None)

            val_prob = predict_probabilities(model, val_loader, device)
            current_val_metrics = compute_metrics(y_val_seq.astype(int), val_prob, 0.5)
            val_score = current_val_metrics[args.threshold_metric]

            history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_score": val_score})
            print(f"[epoch] target={target_col} | epoch={epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_{args.threshold_metric}={val_score:.4f} | seconds={time.perf_counter() - epoch_start:.1f}")

            if val_score > best_val_metric:
                best_val_metric = val_score
                patience_left = args.patience
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        assert best_state is not None
        model.load_state_dict(best_state)

        train_prob = predict_probabilities(model, train_loader, device)
        val_prob = predict_probabilities(model, val_loader, device)
        test_prob = predict_probabilities(model, test_loader, device)

        best_threshold, threshold_grid = select_threshold(y_val_seq.astype(int), val_prob, args.threshold_metric)

        train_metrics = compute_metrics(y_train_seq.astype(int), train_prob, best_threshold)
        val_metrics = compute_metrics(y_val_seq.astype(int), val_prob, best_threshold)
        test_metrics = compute_metrics(y_test_seq.astype(int), test_prob, best_threshold)

        save_predictions(train_ds.get_metadata_frame(), y_train_seq.astype(int), train_prob, best_threshold, preds_dir / "train_predictions.parquet", target_col)
        save_predictions(val_ds.get_metadata_frame(), y_val_seq.astype(int), val_prob, best_threshold, preds_dir / "val_predictions.parquet", target_col)
        save_predictions(test_ds.get_metadata_frame(), y_test_seq.astype(int), test_prob, best_threshold, preds_dir / "test_predictions.parquet", target_col)

        pd.DataFrame(history_rows).to_csv(metrics_dir / "training_history.csv", index=False)
        threshold_grid.to_csv(metrics_dir / "validation_threshold_grid.csv", index=False)
        pd.DataFrame([{"split": "train", **train_metrics}, {"split": "val", **val_metrics}, {"split": "test", **test_metrics}]).to_csv(metrics_dir / "metrics_summary.csv", index=False)

        torch.save(model.state_dict(), model_dir / "model.pt")
        all_summary_rows.append({"target": target_col, "selected_threshold": best_threshold, "train_f2": train_metrics["f2"], "val_f2": val_metrics["f2"], "test_f2": test_metrics["f2"]})
        print(f"[ok] completed target: {target_col} | best_threshold={best_threshold:.2f} | val_f2={val_metrics['f2']:.4f} | test_f2={test_metrics['f2']:.4f}")

        del model, optimizer, criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(all_summary_rows).to_csv(args.output_root / "summary_metrics.csv", index=False)
    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
