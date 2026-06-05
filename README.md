# Project Pipeline and Reproducibility Guide

* NOTE: check all args for bash commands

This repository contains a modular machine learning pipeline for station-level geomagnetic disturbance forecasting using fused space weather observations from GOES XRS, GOES magnetometer, OMNI, Swarm, and SuperMAG.

Because the full historical workflow depends on external data providers, multi-year downloads, and non-redistributable SuperMAG data, the repository supports two reproducibility paths:

1. A full-data workflow for reconstructing the complete project pipeline.
2. A smoke-test workflow that validates the core pipeline on compact test data under `data/test/`.

---

## Environment setup

This project uses `uv` for dependency management.

1. Install `uv` by following the official guide:
   [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

2. From the repository root, create and sync the environment:

```bash
uv sync
```

3. Run Python modules through `uv`:

```bash
uv run python -m <module_name>
```

---

## External data access

The full historical workflow requires credentials for upstream data providers.

### SuperMAG

SuperMAG data cannot be redistributed with this repository. To run the full
download pipeline, register for an academic account and record your username:

[https://supermag.jhuapl.edu/info/?page=faq](https://supermag.jhuapl.edu/info/?page=faq)

### SuperMAG file manifests

Because SuperMAG data cannot be redistributed, this repository includes file
manifests documenting the exact files used in the full historical pipeline.
These allow authorized users to verify their downloads match the expected
archive before running the pipeline.

Manifests are located in `docs/manifests/`:

```text
docs/manifests/supermag_ABK_files.txt
docs/manifests/supermag_MEA_files.txt
docs/manifests/supermag_SOD_files.txt
docs/manifests/supermag_YKC_files.txt
```

Each file lists one filename per line corresponding to the daily Parquet files
downloaded by `src/download/supermag.py` for that station. After downloading,
you can verify completeness with:

```bash
# Example for station ABK
comm -23 \
  <(sort docs/manifests/supermag_ABK_files.txt) \
  <(ls data/raw/supermag/ABK/**/**/*.parquet | xargs -n1 basename | sort)
```

Any filenames appearing only in the left column are missing from your local
archive.

### NASA Earthdata

GOES and OMNI access may require Earthdata authentication through a
`~/.netrc` file:

```text
machine urs.earthdata.nasa.gov
   login YOUR_USERNAME
   password YOUR_PASSWORD
```

### ESA VirES

Swarm access uses `viresclient`. Initialize your access token with:

```bash
uv run viresclient set_token
```

---

## Pipeline overview

The project follows a 13-stage modular pipeline:

1. `src/download/`
   - Download GOES XRS, GOES MAG, OMNI, Swarm, and SuperMAG data.

2. `src/clean/interim/`
   - Clean and standardize each source dataset.

3. `src/preprocess/`
   - Prepare cleaned source data for fusion.
   - Run `swarm_1min.py` before `swarm_chaos.py`.

4. `src/fusion/`
   - Join the preprocessed data into a unified station-time dataset.

5. `src/clean/fused/`
   - Clean the fused station-time dataset.

6. `src/targets/`
   - Build station thresholds with `build_station_thresholds.py`.
   - Attach labels with `attach_station_targets.py`.

7. `src/features/build_feature_regimes.py`
   - Add standardized and regime-based features.

8. `src/dataset/build_ml_dataset.py`
   - Build train, validation, and test splits.

9. `src/features/evaluate_feature_importance.py`
   - Evaluate source-specific feature groups.

10. `src/models/`
    - Train baselines and classifiers.

11. `src/evaluation/evaluate_model_target_station.py`
    - Evaluate model outputs at the station level.

12. `src/evaluation/plot_target_station_results.py`
    - Generate station-level evaluation plots.

13. `src/analysis/latex_plots_and_tables.py`
    - Generate IEEE-formatted figures and LaTeX tables for the report.

---

## Step-by-step pipeline

The commands below reproduce the full project pipeline on the smoke-test
dataset. Replace `data/test` with `data` and adjust date ranges for full
historical reconstruction.

### Step 1 — Generate synthetic SuperMAG smoke-test data

```bash
python3 -m src.tests.generate_synthetic_smoke_test_data \
  --start-date 2015-01-01 \
  --end-date 2015-04-30 \
  --output-root data/test/raw/supermag \
  --force
```

⏱ ~25 min

### Step 2 — Generate subset smoke-test data (OMNI, GOES, Swarm)

```bash
python3 -m src.tests.generate_subset_smoke_test_data \
  --start-date 2015-01-01 \
  --end-date 2015-04-30
```

⏱ ~10 min

### Step 3 — Clean interim sources

```bash
python3 -m src.clean.interim.supermag \
  --input-dir data/test/raw/supermag \
  --output-dir data/test/interim/supermag

python3 -m src.clean.interim.omni \
  --input-dir data/test/raw/omni \
  --output-dir data/test/interim/omni

python3 -m src.clean.interim.goes_xrs \
  --raw-dir data/test/raw/goes_xrs \
  --output-dir data/test/interim/goes_xrs \
  --start-date 2015-01-01 \
  --end-date 2015-04-30

python3 -m src.clean.interim.goes_mag \
  --input-dir data/test/raw/goes_mag \
  --output-dir data/test/interim/goes_mag \
  --start-date 2015-01-01 \
  --end-date 2015-04-30

python3 -m src.clean.interim.swarm \
  --input-dir data/test/raw/swarm \
  --output-dir data/test/interim/swarm
```

### Step 4 — Preprocess sources

> **Note:** `swarm_1min.py` must run before `swarm_chaos.py`.

```bash
python3 src/preprocess/omni.py \
  --input-dir data/test/interim/omni/ \
  --output-dir data/test/preprocessed/omni \
  --start-date 2015-01-01 \
  --end-date 2015-04-30

python3 src/preprocess/goes_xrs.py \
  --input-root data/test/interim/goes_xrs/ \
  --output-root data/test/preprocessed/goes_xrs

python3 src/preprocess/goes_mag.py \
  --input-dir data/test/interim/goes_mag/ \
  --output-dir data/test/preprocessed/goes_mag

python3 src/preprocess/supermag.py \
  --input-dir data/test/interim/supermag/ \
  --output-dir data/test/preprocessed/supermag \
  --start-date 2015-01-01 \
  --end-date 2015-04-30

python3 src/preprocess/swarm_1min.py \
  --input-dir data/test/interim/swarm/ \
  --output-dir data/test/preprocessed/swarm_1min

python3 src/preprocess/swarm_chaos.py \
  --input-dir data/test/preprocessed/swarm_1min/ \
  --output-dir data/test/preprocessed/swarm_chaos
```

### Step 5 — Fuse sources

```bash
python3 src/fusion/data_fusion.py \
  --supermag-root data/test/preprocessed/supermag \
  --omni-root data/test/preprocessed/omni \
  --goes-xrs-root data/test/preprocessed/goes_xrs \
  --goes-mag-root data/test/preprocessed/goes_mag \
  --swarm-root data/test/preprocessed/swarm_chaos \
  --output-root data/test/fused/station_time_master
```

### Step 6 — Clean fused station times

```bash
python3 src/clean/fused/clean_fused_station_times.py \
  --input-dir data/test/fused/station_time_master \
  --output-dir data/test/fused/interim
```

### Step 7 — Build targets

```bash
python3 src/targets/build_station_thresholds.py \
  --input-dir data/test/fused/interim \
  --output-csv data/test/fused/metadata/station_thresholds.csv \
  --months 201501 201502 \
  --train-start-year 2015 \
  --train-end-year 2015 \
  --force

python3 src/targets/attach_station_targets.py \
  --input-dir data/test/fused/interim/ \
  --output-dir data/test/fused/labeled \
  --thresholds-csv data/test/fused/metadata/station_thresholds.csv
```

### Step 8 — Build feature regimes

```bash
python3 src/features/build_feature_regimes.py \
  --input-root data/test/fused/labeled \
  --output-root data/test/preprocessed/fused/standardized
```

### Step 9 — Build ML dataset and evaluate feature importance

```bash
python3 src/dataset/build_ml_dataset.py \
  --input-root data/test/preprocessed/fused/standardized \
  --output-root data/test/ml/standardized \
  --split-by date \
  --split-config-json data/test/fused/metadata/smoke_test_split_config.json
```

> **Note:** To test different date ranges, update the train/val/test boundaries
> in `data/test/fused/metadata/smoke_test_split_config.json` before running
> this step.

```bash
python3 src/features/evaluate_feature_importance.py \
  --input-root data/test/ml/standardized \
  --output-root data/test/reports/features/importance \
  --feature-metadata-json data/test/ml/standardized/metadata/feature_columns.json
```

⏱ ~20 min

### Step 10 — Train models

```bash
python3 src/models/classification/lightgbm_main.py \
  --data-root data/test/ml/standardized \
  --output-root data/test/reports/models/lightgbm_main
```

⏱ ~30 min

```bash
python3 src/models/classification/lstm_classifier.py \
  --data-root data/test/ml/standardized \
  --output-root data/test/reports/models/lstm_classifier \
  --batch-size 1024 \
  --lookback-steps 30 \
  --epochs 4 \
  --cpu-threads 8 \
  --learning-rate 1e-4 \
  --top-k-features 20 \
  --reduced-features-csv data/test/reports/features/importance/feature_importance_all_targets_long.csv \
  --num-workers 4 \
  --weight-decay 1e-4 \
  --layer-sizes 24,16,8,4 \
  --force
```

> **Note:** Baseline models (climatology, persistence, logistic regression) and
> tuned LightGBM are run automatically by `run_smoke_test.py`. For standalone
> execution see [Running downstream models on smoke-test data](#running-downstream-models-on-smoke-test-data).

### Step 11 — Evaluate models

```bash
python3 src/evaluation/evaluate_model_target_station.py --force
```

Output: `data/reports/evaluation/target_station/`

### Step 12 — Generate evaluation plots

```bash
python3 src/evaluation/plot_target_station_results.py --force
```

Output: `data/reports/evaluation/target_station/plots/`

### Step 13 — Generate report figures and tables

```bash
python3 src/analysis/latex_plots_and_tables.py --force
```

Output: `data/reports/analysis/latex_plots/`

⏱ < 1 min

---

## Repository layout

```text
src/
├── analysis/
├── clean/
├── dataset/
├── download/
├── evaluation/
├── features/
├── fusion/
├── models/
├── preprocess/
├── targets/
└── tests/

data/
├── raw/
│   ├── goes_xrs/
│   ├── goes_mag/
│   ├── omni/
│   ├── swarm/
│   └── supermag/
└── test/
    ├── raw/
    │   ├── goes_xrs/
    │   ├── goes_mag/
    │   ├── omni/
    │   ├── swarm/
    │   └── supermag/
    ├── clean/
    │   ├── interim/
    │   └── fused/
    ├── preprocess/
    ├── fusion/
    ├── targets/
    ├── features/
    ├── dataset/
    ├── models/
    └── evaluation/

docs/
└── manifests/
    ├── supermag_ABK_files.txt
    ├── supermag_MEA_files.txt
    ├── supermag_SOD_files.txt
    └── supermag_YKC_files.txt
```

The `data/raw/` tree is used for the full historical workflow. The `data/test/`
tree is reserved for smoke-test inputs and outputs and mirrors the real pipeline
layout so that all smoke-test artifacts remain isolated from production-scale
data.

---

## Smoke-test workflow

The smoke test validates the core processing path from pipeline stages 2–13
using compact test inputs under `data/test/`. It is intentionally narrow: the
goal is to verify build stability, pathing, schema compatibility, and downstream
artifact creation without requiring a full historical data rebuild.

### Smoke-test design rules

- `data/test/` is the single smoke-test root.
- Raw smoke-test files mirror the real source-specific directory layout.
- All sources use the same timestamp window so that fusion logic is exercised
  consistently.
- All derived smoke-test artifacts remain inside `data/test/`.
- The smoke test asserts the presence of key downstream outputs rather than
  only printing a success message.
- The smoke test validates the core workflow only and does not attempt
  exhaustive benchmarking.

### Smoke-test scripts

The smoke-test utilities live in `src/tests/`:

- `generate_synthetic_smoke_test_data.py`
  — Generates synthetic raw SuperMAG files for smoke-test use.

- `generate_subset_smoke_test_data.py`
  — Creates compact matching subsets for OMNI, GOES XRS, GOES MAG, and Swarm
  over the same test window.

- `run_smoke_test.py`
  — Runs the end-to-end smoke-test workflow using `data/test/` as the data root
  and verifies required downstream outputs.

### Smoke-test raw data layout

```text
data/test/raw/
├── goes_xrs/
├── goes_mag/
├── omni/
├── swarm/
└── supermag/
    └── {station}/
        └── {YYYY}/
            └── {MM}/
                └── supermag_{station}_{YYYYMMDD}.parquet
```

### SuperMAG smoke-test schema

Raw SuperMAG smoke-test files must match the filename pattern and column schema
expected by `src/clean/interim/supermag.py`.

Expected filename format:

```text
supermag_{STATION}_{YYYYMMDD}.parquet
```

Expected raw columns:

```text
tval, ext, iaga, glon, glat, mlt, mcolat, decl, sza, N, E, Z,
source_station, source_url
```

Notes:

- `tval` should be parseable as Unix seconds.
- `iaga` should identify the station.
- `N`, `E`, and `Z` should preserve the structure expected by the cleaner,
  including `geo` and `nez` values.
- `source_station` should match the station directory and station code in the
  filename.
- `source_url` should point to the SuperMAG service URL string used by the real
  pipeline metadata.

### Smoke-test outputs

All smoke-test outputs remain inside `data/test/`. Expected output directories:

```text
data/test/clean/interim/
data/test/preprocess/
data/test/fusion/
data/test/clean/fused/
data/test/targets/
data/test/features/
data/test/dataset/
data/test/models/
data/test/evaluation/
```

The smoke-test runner verifies required downstream artifacts before reporting
success. Typical checked outputs include monthly cleaned and preprocessed files,
fused station-time tables, labeled targets, feature tables, per-target
train/val/test datasets, model summaries, and evaluation CSV outputs.

### Smoke-test restart levels

`run_smoke_test.py` supports staged restarts via `--start-level`, which is
useful when debugging a later stage without rerunning the entire workflow.

| Level | Restart point |
|---|---|
| `raw` | Start from `src/clean/interim/` and run all downstream stages |
| `fusion` | Start from `src/fusion/`, assume preprocessed inputs exist |
| `cleanfused` | Start from `src/clean/fused/`, assume fused tables exist |
| `targets` | Start from `src/targets/`, assume cleaned fused tables exist |
| `features` | Start from `src/features/`, assume labeled target tables exist |

### Observed smoke-test runtimes

| Command | Approx. runtime |
|---|---|
| `generate_synthetic_smoke_test_data` (4-month window) | ~25 min |
| `generate_subset_smoke_test_data` (4-month window) | ~10 min |
| `run_smoke_test --start-level raw` (through logistic regression) | ~80 min |
| `evaluate_feature_importance` | ~20 min |
| `lightgbm_main` (full smoke-test dataset) | ~30 min |

The `raw` smoke-test runtime above reflects running through
`climatology.py` and `logistic_regression.py`. For rapid iteration after
earlier artifacts exist, use a later `--start-level` value.

### Smoke-test model scope

The smoke-test runner is best suited for lightweight baseline models such as
climatology and logistic regression. LightGBM and the LSTM classifier are
better treated as targeted downstream tests on `data/test/dataset/` rather than
part of every full `--start-level raw` run.

---

## Running downstream models on smoke-test data

Once `data/test/dataset/` has been built, downstream classifiers can be run
directly against the smoke-test dataset.

### LightGBM

```bash
uv run python -m src.models.classification.lightgbm_main \
  --data-root data/test/dataset \
  --output-root data/test/models/lightgbm_main \
  --targets target_geq_p95_30m target_geq_p99_30m \
            target_geq_p95_60m target_geq_p99_60m \
            target_geq_p95_120m target_geq_p99_120m \
  --threshold-metric f1 \
  --force
```

A faster debug version with row caps:

```bash
uv run python -m src.models.classification.lightgbm_main \
  --data-root data/test/dataset \
  --output-root data/test/models/lightgbm_main \
  --targets target_geq_p95_30m target_geq_p99_30m \
            target_geq_p95_60m target_geq_p99_60m \
            target_geq_p95_120m target_geq_p99_120m \
  --max-train-rows 200000 \
  --max-val-rows 100000 \
  --max-test-rows 100000 \
  --threshold-metric f1 \
  --force
```

### Tuned LightGBM

```bash
uv run python -m src.models.classification.lightgbm_tuned_simple \
  --ml-root data/test/dataset \
  --output-root data/test/models/lightgbm_tuned_simple \
  --targets target_geq_p95_30m target_geq_p99_30m \
            target_geq_p95_60m target_geq_p99_60m \
            target_geq_p95_120m target_geq_p99_120m \
  --trials 25 \
  --random-state 42 \
  --force
```

### LSTM classifier

```bash
uv run python -m src.models.classification.lstm_classifier \
  --data-root data/test/dataset \
  --output-root data/test/models/lstm_classifier \
  --targets target_geq_p95_30m target_geq_p99_30m \
            target_geq_p95_60m target_geq_p99_60m \
            target_geq_p95_120m target_geq_p99_120m \
  --lookback-steps 60 \
  --batch-size 256 \
  --hidden-size 64 \
  --num-layers 2 \
  --dropout 0.1 \
  --learning-rate 1e-3 \
  --weight-decay 1e-5 \
  --epochs 20 \
  --patience 5 \
  --threshold-metric f1 \
  --device cpu \
  --num-workers 0 \
  --seed 42 \
  --force
```

With physics-chain layer sizes:

```bash
uv run python -m src.models.classification.lstm_classifier \
  --data-root data/test/dataset \
  --output-root data/test/models/lstm_classifier \
  --targets target_geq_p95_30m target_geq_p99_30m \
            target_geq_p95_60m target_geq_p99_60m \
            target_geq_p95_120m target_geq_p99_120m \
  --layer-sizes 32,24,16,8 \
  --lookback-steps 60 \
  --batch-size 256 \
  --dropout 0.1 \
  --learning-rate 1e-3 \
  --weight-decay 1e-5 \
  --epochs 20 \
  --patience 5 \
  --threshold-metric f1 \
  --device cpu \
  --num-workers 0 \
  --seed 42 \
  --force
```

A lighter debug run with row caps and fewer epochs:

```bash
uv run python -m src.models.classification.lstm_classifier \
  --data-root data/test/dataset \
  --output-root data/test/models/lstm_classifier \
  --targets target_geq_p95_30m target_geq_p99_30m \
            target_geq_p95_60m target_geq_p99_60m \
            target_geq_p95_120m target_geq_p99_120m \
  --lookback-steps 60 \
  --batch-size 256 \
  --epochs 10 \
  --patience 3 \
  --max-train-rows 200000 \
  --max-val-rows 100000 \
  --max-test-rows 100000 \
  --threshold-metric f1 \
  --device cpu \
  --num-workers 0 \
  --seed 42 \
  --force
```

---

## Full-data reconstruction

The full-data workflow reconstructs the complete multi-year archive from
external providers.

Example SuperMAG download command:

```bash
uv run python -m src.download.supermag \
  --start-date 2015-01-01 \
  --end-date 2024-12-31 \
  --userid YOUR_SUPERMAG_USERNAME \
  --output-dir data/raw/supermag/
```

Notes:

- If `--stations` is omitted, station defaults follow the implementation in
  `src/download/supermag.py`.
- Re-running download scripts is safe if the scripts are idempotent.
- Use `--force` only when you explicitly want to replace existing files.
- Cross-reference your downloaded files against the manifests in
  `docs/manifests/` to verify archive completeness before running the pipeline.

---

## Reproducibility notes

This repository is designed for reproducible verification, but not all upstream
data can be redistributed directly. SuperMAG access is governed by provider
policies and requires user-specific credentials. Full historical reconstruction
therefore depends on the user obtaining access independently.

The recommended verification path for course staff is the smoke test under
`data/test/`, which exercises the complete pipeline on synthetic and subsetted
data without requiring SuperMAG credentials. The full-data path is provided for
complete reconstruction by authorized users who have obtained SuperMAG access.

All model prediction files, metric CSVs, evaluation summaries, and report
figures are safe to distribute and are included in the repository under
`data/reports/`. These can be used to verify the reported results directly
without re-running the pipeline.
