# DSC 288 Capstone: Physics-Informed Station-Level GIC Forecasting

## 1. Introduction & Motivation
* **Objective:** Develop station-level geomagnetic disturbance forecasting models to predict whether local magnetic variability will exceed high-impact thresholds (95th and 99th percentiles) over 30, 60, and 120-minute lead times.
* **Impact:** Severe geomagnetic disturbances threaten power-grid stability, disrupt navigation, and endanger satellite infrastructure. Precise, localized forecasting directly meets heliophysics decadal and SWAG 2024 user needs.
* **Performance Summary:** Across the benchmark set, LightGBM emerges as the strongest overall model family. The tuned variant outperforms baseline methods, particularly when structural class imbalance techniques are applied to match the percentile-based targets.

## 2. Data Pipeline & Methodology
* **Data Sources & Fusion:** Ingestion of ~100GB of multi-format data (OMNI, GOES, Swarm, SuperMAG) to construct a synchronized 1-minute cadence ML-ready dataset representing the Sun-to-Ground causal chain.
* **Missingness Handling:** Addressing systemic L1 (ACE/DSCOVR) missingness (~20%) via forward-filling and explicit indicator flags, aligning with known operational constraints (Author, 2024 AGU Fall Meeting).
* **Target Engineering:** Dynamic, localized calculation of the horizontal magnetic field derivative ($dB/dt_{H}$) to establish station-specific 95th and 99th percentile thresholds, shifting the paradigm from global classification to highly localized rare-event prediction.

## 3. Evaluation Framework
* **Metrics:** Justification for abandoning standard Accuracy in favor of metrics suited for heavy-tailed, rare-event distributions: Precision-Recall AUC (PR-AUC), ROC-AUC, Brier Score, and the Heidke Skill Score (HSS).

## 4. Baseline Models
* **Persistence:** Assumes the current local $dB/dt$ state continues unchanged; serves as the primary short-horizon benchmark.
* **Climatology:** Predicts based on historical baseline averages; serves as the primary long-horizon benchmark.

## 5. Classifiers
### 5.1 Logistic Regression
* Serves as the interpretable, linear baseline to evaluate the necessity of non-linear feature interactions.

### 5.2 LightGBM
* **Implementation:** The primary gradient-boosted tree architecture.
* **Findings:** An expansive hyperparameter search revealed a highly stable configuration (`num_leaves=31`, `learning_rate=0.05`) across all forecast horizons, suggesting the engineered features capture underlying physics robustly without horizon-specific memorization.

### 5.3 LSTM (Physics-Informed Architecture)
* **Design Philosophy:** Motivated by the information bottleneck principle (Tishby, 1999) and Physics-Informed Neural Networks (Raissi, 2019), we propose a hierarchically narrowing LSTM architecture. 
* **Causal Compression:** Deep networks naturally learn increasingly abstract representations (Bengio, 2013). By narrowing the hidden layers, the architecture structurally enforces signal compression along the Sun $\rightarrow$ L1 $\rightarrow$ GEO $\rightarrow$ LEO $\rightarrow$ Ground causal chain, filtering high-variance raw solar wind into a sparse, localized ground response.
* **Architectural Limitations:** While the narrowing dimensions (e.g., 24 $\rightarrow$ 16 $\rightarrow$ 8 $\rightarrow$ 4) align conceptually with physics, the specific node counts are heuristically defined. Future work requires targeted ablation studies to compare this explicit structural constraint against the implicit compression learned by a standard uniform LSTM.

#

## Why Study this
* This project develops station-level geomagnetic disturbance forecasting models to predict whether local magnetic variability will exceed high-impact thresholds over 30, 60, and 120 minute lead times using fused space weather and geophysical inputs from satellites, solar wind measurements, and ground magnetometer observations. Improving these forecasts matters because severe geomagnetic disturbances can threaten power-grid stability, disrupt navigation and communications systems, and endanger costly satellite infrastructure, while also helping us better understand the near-Earth space environment. Across the current benchmark set, LightGBM is the strongest overall model family, and the tuned LightGBM variant currently performs best, outperforming the untuned version after removing class_weight='balanced', which appears to better match the project’s percentile-based targets despite their natural class imbalance.

## Baseline Models

### Persistance

### Climatology

## Classifiers

### LogReg

### LightGBM

### LSTM
* **TLDR:** Motivated by the information bottleneck principle and physics-informed modeling, we propose a hierarchically narrowing LSTM architecture where each layer's hidden dimensionality reflects the progressive signal compression along the Sun→L1→GEO→LEO→Ground causal chain. Rather than uniform hidden size across layers, we constrain each stage to a reduced representation consistent with the decreasing degrees of freedom in the physical system — from the high-variance raw solar wind to the sparse, localized ground GIC response.
* Raissi (2019); *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*
    * The core PINN idea is encoding known physical structure into the network architecture rather than learning it from scratch. You're not using the full PINN framework (no differential equations in the loss), but you're applying the same principle: the Sun→L1→GEO→LEO→Ground causal chain is known physics, so you embed it structurally rather than hoping the data alone teaches it.
* Bengio (2013); *Hierarchical feature learning in deep network*
    * Deep networks naturally learn increasingly abstract representations at each layer. Your narrowing matches the physics: L1 features (solar wind speed, Bz) are raw measurements; by the time you reach ground GIC the relevant signal is a highly compressed summary of the upstream chain.
* Tishby (1999); *The information bottleneck method*
    * The best neural networks learn to compress input information down to only what's relevant for the target. A narrowing architecture enforces this compression structurally rather than hoping the optimizer discovers it.

* Weakness: `24,16,8,4` not derived from physics and would need albation study to show it
* Weakness: Standard nn.LSTM with num_layers=4 already learns inter-layer compression implicitly. Your architecture constrains it better if the constraint matches reality






