# Project Minimal and Modernized `src/` Structure

```
./src
    ├── clean
    │   ├── fused
    │   │   └── clean_fused_station_times.py
    │   └── interim
    │       ├── cleaning_utils.py
    │       ├── goes_mag.py
    │       ├── goes_xrs.py
    │       ├── omni.py
    │       ├── supermag.py
    │       └── swarm.py
    ├── dataset
    │   └── build_ml_dataset.py
    ├── download
    │   ├── goes_config.py
    │   ├── goes_mag.py
    │   ├── goes_xray.py
    │   ├── omni.md
    │   ├── supermag.py
    │   └── swarm.py
    ├── evaluation
    │   ├── evaluate_model_target_station.py
    │   └── plot_target_station_results.py
    ├── features
    │   ├── build_feature_regimes.py
    │   └── evaluate_feature_importance.py
    ├── fusion
    │   └── data_fusion.py
    ├── models
    │   ├── baselines
    │   │   ├── climatology.py
    │   │   └── persistence.py
    │   └── classification
    │       ├── lightgbm_main.py
    │       ├── lightgbm_tuned_simple.py
    │       ├── logistic_regression.py
    │       └── lstm_classifier.py
    ├── preprocess
    │   ├── common.py
    │   ├── goes_mag.py
    │   ├── goes_xrs.py
    │   ├── omni.py
    │   ├── supermag.py
    │   ├── swarm_1min.py
    │   └── swarm_chaos.py
    └── targets
        ├── attach_station_targets.py
        └── build_station_thresholds.py
```

# Project Pipeline Flow

## 1. `./src/download`
* download scripts for: GOES_XRS, OMNI, GOES_MAG, Swarm, SuperMag

## 2. `./src/clean/interim`
* clean download data for: GOES_XRS, OMNI, GOES_MAG, Swarm, SuperMag

## 3. `./src/preprocess`
* preprocess cleaned download data for fusion script
* **note:** order matters for swarm: `swarm_1min.py` -> `swarm_chaos.py` since chaos script uses output from 1min script

## 4. `./src/fusion`
* join preprocessed data for: GOES_XRS, OMNI, GOES_MAG, Swarm, SuperMag

## 5. `./src/clean/fused`
* clean joined data for: GOES_XRS, OMNI, GOES_MAG, Swarm, SuperMag

## 6. `./src/targets
* calc threshold for 95th and 99th percentile for $\Delta$t $\in$ `[30min, 60min, 120min]`
* **note:** order matters: `build_station_thresholds.py` -> `attach_station_targets.py` since attach script uses `csv` that the threshold script uses

## 7. `./src/features/build_feature_regimes.py`
* add standardized features

## 8. `./src/dataset/build_ml_dataset`
* create train/val/test splitsvfor models
	* train: 2015 to 2022
	* val: 2023
	* test: 2024
* **note:** 2024 very active compared to train/val *Gannon Event*

## 9. `./src/features/evaluate_feature_importance.py`
* see Sun, L1, GEO, LEO, Supermag physics features and location features have value

## 10. `./src/models`
* contain baseline and classification model scripts
* baseline models: `climatology.py`, `persistence.py`
* classifier models: `lightgbm_main.py`, `logistic_regression.py`, `lightgbm_tuned_simple.py` `lstm_classifier.py`

## 11. `./src/evaluation`
* evaluate and visualize model performance


