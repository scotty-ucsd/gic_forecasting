## Step 1

`python3 -m src.tests.generate_synthetic_smoke_test_data --start-date 2015-01-01 --end-date 2015-04-30 --output-root data/test/raw/supermag --force`

## Step 2

`python3 -m src.tests.generate_subset_smoke_test_data --start-date 2015-01-01 --end-date 2015-04-30`

## Step 3

`python3 -m src.clean.interim.supermag --input-dir data/test/raw/supermag --output-dir data/test/interim/supermag`

`python3 -m src.clean.interim.omni --input-dir data/test/raw/omni --output-dir data/test/interim/omni`

`python3 -m src.clean.interim.goes_xrs --raw-dir data/test/raw/goes_xrs --output-dir data/test/interim/goes_xrs --start-date 2015-01-01 --end-date 2015-04-30`

`python3 -m src.clean.interim.goes_mag --input-dir data/test/raw/goes_mag --output-dir data/test/interim/goes_mag --start-date 2015-01-01 --end-date 2015-04-30`

`python3 -m src.clean.interim.swarm --input-dir data/test/raw/swarm --output-dir data/test/interim/swarm`

## Step 4
`python3 src/preprocess/omni.py --input-dir data/test/interim/omni/ --output-dir data/test/preprocessed/omni --start-date 2015-01-01 --end-date 2015-04-30`

`python3 src/preprocess/goes_xrs.py --input-root data/test/interim/goes_xrs/ --output-root data/test/preprocessed/goes_xrs`

`python3 src/preprocess/goes_mag.py --input-dir data/test/interim/goes_mag/ --output-dir data/test/preprocessed/goes_mag`

`python3 src/preprocess/supermag.py --input-dir data/test/interim/supermag/ --output-dir data/test/preprocessed/supermag --start-date 2015-01-01 --end-date 2015-04-30`

* note: order important here

`python3 src/preprocess/swarm_1min.py --input-dir data/test/interim/swarm/ --output-dir data/test/preprocessed/swarm_1min`

`python3 src/preprocess/swarm_chaos.py --input-dir data/test/preprocessed/swarm_1min/ --output-dir data/test/preprocessed/swarm_chaos`

## Step 5 

`python3 src/fusion/data_fusion.py --supermag-root data/test/preprocessed/supermag --omni-root data/test/preprocessed/omni --goes-xrs-root data/test/preprocessed/goes_xrs --goes-mag-root data/test/preprocessed/goes_mag --swarm-root data/test/preprocessed/swarm_chaos --output-root data/test/fused/station_time_master`

## Step 6

`python3 src/clean/fused/clean_fused_station_times.py --input-dir data/test/fused/station_time_master --output-dir data/test/fused/interim`

## Step 7

`python3 src/targets/build_station_thresholds.py --input-dir data/test/fused/interim --output-csv data/test/fused/metadata/station_thresholds.csv --months 201501 201502 --train-start-year 2015 --train-end-year 2015 --force`

`python3 src/targets/attach_station_targets.py --input-dir data/test/fused/interim/ --output-dir data/test/fused/labeled --thresholds-csv data/test/fused/metadata/station_thresholds.csv`

## Step 8

`python3 src/features/build_feature_regimes.py --input-root data/test/fused/labeled --output-root data/test/preprocessed/fused/standardized`

## Step 9

`python3 src/dataset/build_ml_dataset.py --input-root data/test/preprocessed/fused/standardized --output-root data/test/ml/standardized --split-by date --split-config-json data/test/fused/metadata/smoke_test_split_config.json`

* note if testing other dates make sure to adjust the train/val/test split in `data/test/fused/metadata/smoke_test_split_config.json`

`python3 src/features/evaluate_feature_importance.py --input-root data/test/ml/standardized --output-root data/test/reports/features/importance --feature-metadata-json data/test/ml/standardized/metadata/feature_columns.json`

## Step 10

`python3 src/models/classification/lightgbm_main.py --data-root data/test/ml/standardized --output-root data/test/reports/models/lightgbm_main`

* note: predictions are generated in `data/reports/models/{climatology,persistence,logistic_regression,lightgbm_main,lightgbm_tuned_simple}`

## Step 11

`python3 src/evaluation/evaluate_model_target_station.py --force`

* output in `data/reports/evaluation/target_station`

## Step 12

`python3 src/evaluation/plot_target_station_results.py --force`

* output in `data/reports/evaluation/target_station`

