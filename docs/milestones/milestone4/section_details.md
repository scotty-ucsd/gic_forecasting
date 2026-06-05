## Abstract
*One paragraph, ~120 words. Hit all three rubric requirements: motivation, methodology, results.*

Geomagnetic disturbances pose significant risks to power grids, satellite operations, and navigation systems, yet station-level forecasting of extreme perturbation events remains an open challenge. We present a multi-source machine learning pipeline that fuses solar wind (L1), magnetospheric (GEO), low-Earth orbit (LEO), and ground-station (SuperMAG) observations to forecast binary exceedance events at four polar monitoring stations across six target definitions (p95/p99 thresholds at 30, 60, and 120-minute horizons). We compare six model families — climatology, persistence, logistic regression, LightGBM, tuned LightGBM, and LSTM — evaluated on a held-out 2024 test year. Tuned LightGBM achieves the best test F2 across all targets (0.627–0.794), while the LSTM severely overfits despite strong validation scores. Solar wind speed and dynamic pressure are the dominant predictive features across all horizons.

---

## Introduction
*No more than 1 page. Four beats: problem → stakes → approach → input/output statement.*

* **Problem:** Geomagnetic storms driven by solar activity induce ground-level magnetic field perturbations. Predicting when a station's horizontal perturbation will exceed a climatological extreme threshold is a binary classification problem at the intersection of space weather and operational risk management.
* **Stakes:** Extreme dB/dt events can induce geomagnetically induced currents (GICs) that damage high-voltage transformers. Short-lead-time warnings (30–120 minutes) enable grid operators to take protective action.
* **Approach:** A physics-informed feature stack spanning the Sun-to-ground causal chain (solar flares → solar wind → magnetosphere → ionosphere → ground) is constructed and fed to a suite of models. Station-level evaluation — not just pooled metrics — is used to assess whether skill is spatially robust.
* **Input/output statement** *(required by rubric)*: 
> "The input to our system is a 1-minute cadence multi-source feature vector spanning solar X-ray flux, L1 solar wind state, GEO magnetospheric field, LEO residual field, and local SuperMAG station geometry. The output is a binary prediction of whether the station's horizontal magnetic perturbation will exceed its historical p95 (or p99) threshold within the next 30, 60, or 120 minutes."

---

## Related Work
*No more than 1 page. Group by approach — at least 5 references.*

* **Group 1 — Physics-based and empirical storm prediction:** Cite foundational work on Dst/Kp index prediction. Note these are global indices, not station-level — a key distinction from your work.
* **Group 2 — ML for geomagnetic disturbance forecasting:** Cite papers using neural networks or gradient boosting for dB/dt or GIC prediction. Note most prior work uses pooled or single-station evaluation; your multi-station approach is a differentiator.
* **Group 3 — Solar wind coupling functions:** Cite Newell et al. for the Newell coupling function — your `l1_newell_like` feature is directly derived from this. This is the reference for one of your top-5 features.
* **Group 4 — Sequence models for space weather:** Cite LSTM-based approaches (e.g., Gruet et al. or similar). Note the known overfitting risk in sparse-event time series — which your results confirm.
* **Group 5 — Class imbalance in event prediction:** Cite work on F2/HSS for rare event skill assessment in meteorology/space weather. Justifies your metric choices.
* **State of the art:** Current operational systems (NOAA SWPC) use empirical thresholds, not learned models. Note that ML approaches have shown promise but station-level generalisation across a held-out year remains a gap.

---

## Dataset and Features

### Dataset Description
* **Sources (5):** GOES XRS (solar flux), GOES MAG (GEO field), OMNI/ACE (L1 solar wind), Swarm LEO (residual field), SuperMAG (ground stations). *Cite each dataset paper.*
* **Coverage:** 2015–2024, 1-minute cadence, 4 stations.
* **Splits:** Chronological — train 2015–2022 (12.8M rows), val 2023 (1.7M rows), test 2024 (1.8M rows). No data leakage possible.
* **Event rates:** p95 ~22–36% (varies by target horizon), p99 ~7–14%. The p99 rate is consistent with the 99th percentile definition — ~1% of minutes in the train period exceed the threshold per station-year.
* *(Table 1 goes here)*

### Preprocessing Pipeline
* **Fusion:** 1-minute cadence; forward-fill gaps up to defined staleness limits per source.
* **Thresholding:** Station-specific p95/p99 thresholds computed on train split only (no leakage).
* **Targeting:** Binary targets attached for 6 horizon/percentile combinations.
* **Standardisation:** Feature standardisation (zero mean, unit variance) fit on train, applied to val/test.
* **Feature Reduction:** 51 raw features (from `feature_columns.json`) reduced to 20 reduced features for LightGBM fixed-grid and 44 features for LGBM Tuned (includes regime features from Phase 9).
* *(Figure 1 goes here: feature importance — motivates the feature engineering choices)*

---

## Methods
*~2 pages. One short paragraph per model + key equations.*

### Creating Targets
Station-specific percentile thresholds are fit on the train split's horizontal perturbation magnitude. A binary label is 1 if the maximum perturbation in the next $H$ minutes ($H \in \{30, 60, 120\}$) exceeds the p95 or p99 threshold for that station. This produces 6 targets per station-minute row.

### Addressing Class Imbalance
Three strategies are used depending on model family:
* **Logistic Regression / LightGBM fixed:** `class_weight='balanced'` rescales the loss.
* **LGBM Tuned:** No explicit reweighting — threshold tuning on val F2 compensates.
* **LSTM:** Positive class weight $$w^+ = \frac{N^-}{N^+}$$ applied via `BCEWithLogitsLoss`.

All models use the $F_\beta$ score ($\beta=2$) as the primary metric, weighting recall twice over precision:
$$F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{(\beta^2 \cdot \text{precision}) + \text{recall}}$$

### Model Configs
* **Climatology:** Predicts the train-split event rate as a constant probability. Threshold fixed at 0.05. No learning.
* **Persistence:** Predicts positive if the current $B_H$ measurement exceeds the train-split p95/p99 threshold. A strong rule-based baseline for short horizons.
* **Logistic Regression:** L2-regularised logistic regression with $C$ selected from $\{0.01, 0.1, 1, 10\}$ via val F2. 51 standardised features.
* **LightGBM (fixed grid):** Three fixed hyperparameter configurations (`n_estimators=300`, `lr=0.05`, `num_leaves=31`), best selected by val F2. 20 reduced features.
* **LightGBM Tuned:** Random search over 20 trials across 9 hyperparameters. 44 features including regime indicators. Best iteration selected by val F2 with early stopping.
* **LSTM:** Single-layer narrowing architecture reflecting the Sun-to-ground causal chain. Lookback window of 30 minutes. Trained for 4 epochs with positive class weighting.

### Evaluation Metrics
$$F_2 = \frac{5 \cdot \text{precision} \cdot \text{recall}}{(4 \cdot \text{precision}) + \text{recall}}$$
$$HSS = \frac{2(TP \cdot TN - FP \cdot FN)}{(TP + FN)(FN + TN) + (TP + FP)(FP + TN)}$$
* **Also report:** PR-AUC, precision, recall. 
* **Note:** Accuracy is omitted because it is misleading with a ~10% event rate for p99 targets.

---

## Results
*~2 pages.*

* *(Table 3 goes here: primary results - test F2 all models × all targets).*
* *(Figure 3 goes here: grouped bar — visual companion to Table 3).*
* *(Figure 2 goes here: Phase 11.2 LightGBM-minus-climatology heatmap — station × target skill map). Discuss which stations/targets show the largest gains.*
* *(Table 4 goes here: LGBM Tuned full metrics - precision, recall, F2, PR-AUC, HSS).*
* *(Figure 4 or 5 goes here: pick one — generalisation plot or precision/recall breakdown).*

### Discussion Beats
* **LGBM Tuned wins all 6 targets:** The random search tuning step is worth the compute cost.
* **Baseline comparisons:** Climatology beats Persistence on all targets and beats Logistic Regression on p95 targets — a calibrated constant-rate baseline is surprisingly hard to beat with a linear model when event rates are high.
* **LSTM collapse:** Validation F2 of 0.80–0.85 vs test F2 of 0.20–0.27 for p95 targets. This is the largest val-test gap in the project. Likely causes include only 4 training epochs, no regularisation tuning, and potential solar-cycle distribution shift between the 2022 train and 2024 test (Solar Cycle 25 peak).
* **Heidke Skill Score (HSS):** Confirms skill is genuine — values of 0.37–0.46 on test indicate the model is not exploiting base rates.
* **Target Difficulty:** Harder targets follow a consistent pattern: p99 < p95, 30m < 60m < 120m. Longer horizons are easier because the slowly-varying solar wind state provides more lead-time signal.

---

## Conclusions
*~1 paragraph + gap analysis paragraph.*

**Summary:** Tuned LightGBM achieves the best test F2 (0.627–0.794) across all six targets, driven primarily by L1 solar wind features. Climatology is a stronger baseline than expected. The LSTM severely overfits despite strong validation performance.

**Gap Analysis** *(required by rubric)*: Original goals included training an LSTM with full hyperparameter tuning across all targets. Computational constraints limited LSTM training to 4 epochs without dropout or architecture search, explaining the overfit. Future work should explore longer LSTM training, Monte Carlo dropout for uncertainty quantification, and incorporation of the Swarm CHAOS internal field model residuals which showed high feature importance but limited model integration.

---

## Contributions
*One paragraph per team member. State specifically what each person built — which phases, which scripts, which analyses. The rubric notes non-participation can be penalised, so be precise.*

---

## Reflections
*~half page. Hit all four rubric prompts:*

* **Would do differently:** LSTM training budget — 4 epochs is insufficient. Would allocate compute time before the project deadline, not during.
* **Different goals:** Station-level regression (predict the actual perturbation magnitude) rather than binary classification would have provided a richer evaluation surface.
* **Different dataset:** Adding more stations (SuperMAG has hundreds) would strengthen the spatial generalisation claim. Currently limited to 4 stations.
* **Lessons learned:** Chronological splits are non-negotiable for time series. Early experiments with random splits produced artificially inflated metrics that masked the true temporal generalisation gap.

---

## Reply to Review
*Max 1 page. Action-oriented, specific. Wait to write this until you have the peer reviews. Structure each response as: "Reviewer noted X → We addressed this by Y." Keep each response to 2–3 sentences.*

