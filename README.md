# Tanzania Waterpoint Functionality Prediction Platform
## An End-to-End Data Mining, Machine Learning, and Field Dispatch Journey

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Champion_Model-2A9D8F)
![Streamlit](https://img.shields.io/badge/Streamlit-Operational_Platform-FF4B4B?logo=streamlit&logoColor=white)
![Dataset](https://img.shields.io/badge/DrivenData-59%2C400_Waterpoints-1D3557)

---

## 1. Executive Summary & Context

Access to clean, potable water remains one of the most critical public health challenges across sub-Saharan Africa. In Tanzania, over 28 percent of the population lacks sustained access to an improved water source, forcing communities to rely on surface water or travel long distances to fetch daily supplies. Over past decades, the Government of Tanzania alongside international non-governmental organizations, local communities, and private development partners have financed the installation of tens of thousands of waterpoints across the country. However, establishing physical infrastructure is only half the battle: a substantial fraction of these waterpoints fall into disrepair, run dry, or fail entirely due to maintenance backlogs and resource constraints.

To address this challenge systematically, Taarifa and the Tanzanian Ministry of Water and Irrigation open-sourced nationwide waterpoint telemetry data through the DrivenData platform (*"Pump it Up: Data Mining the Water Table"*). The dataset captures records for **59,400 waterpoints** evaluated across **40 tabular features**, encompassing geographic coordinates, geological basins, extraction technologies, installation years, funding agencies, and local management models.

The primary mission of this project is to build an end-to-end, production-grade data science and machine learning platform capable of triaging waterpoints into three distinct operational states:
1. **functional**: Waterpoints that are operating normally and require no immediate intervention.
2. **functional needs repair**: Waterpoints that are currently operating but have degraded mechanics or low yield, requiring preventive maintenance.
3. **non functional**: Waterpoints that have suffered catastrophic failure or run completely dry, demanding urgent rehabilitation or replacement.

By identifying pumps in the transitional `functional needs repair` state before they fail completely, regional water authorities can prioritize preventive repairs—which cost a fraction of full replacements—thereby preventing prolonged water outages for vulnerable rural populations.

---

## 2. Foundations and Academic References

This project builds upon proven methodologies in predictive infrastructure maintenance and advances them into a robust, leak-free, production-ready system. Special acknowledgment and architectural reference are attributed to:

1. **"Predicting the Functionality of Water Pumps with XGBoost"** (Towards Data Science)  
   Reference: [https://towardsdatascience.com/predicting-the-functionality-of-water-pumps-with-xgboost-8768b07ac7bb/](https://towardsdatascience.com/predicting-the-functionality-of-water-pumps-with-xgboost-8768b07ac7bb/)  
   *Contribution*: Demonstrating the high utility of gradient-boosted decision trees (XGBoost) for handling tabular infrastructure data containing complex interaction effects, non-linear relationships, and heterogeneous feature types.

2. **"Detecting Faulty Water Pumps With Machine Learning"** by Ananya Nair  
   Reference: [https://github.com/anair123/Detecting-Faulty-Water-Pumps-With-Machine-Learning](https://github.com/anair123/Detecting-Faulty-Water-Pumps-With-Machine-Learning)  
   *Contribution*: Exploratory analysis frameworks for geographic features, regional management disparities, and baseline model comparisons across standard classifier architectures.

Our implementation expands upon these foundational works by establishing strict leak-free cross-validation boundaries, applying hypothesis-driven feature selection (Kruskal-Wallis, ANOVA, Chi-Square), implementing cost-sensitive rebalancing to resolve severe minority class overfitting, and deploying a field-ready Streamlit decision platform with automated PDF dispatch scheduling.

---

## 3. The Data Science Journey: Step-by-Step Architecture

The project is structured as an analytical and engineering journey spanning six interconnected stages, transitioning from raw observational data to an active field operations center.

```text
[Stage 1: Raw Data Ingestion & Exploratory Anomaly Auditing]
                         │
                         ▼
[Stage 2: Leak-Free Preprocessing & Statistical Hypothesis Selection]
                         │
                         ▼
[Stage 3: Custom Modular Feature Engineering Transformer]
                         │
                         ▼
[Stage 4: Champion XGBoost Modeling, Overfitting Mitigation & Evaluation]
                         │
                         ▼
[Stage 5: Operational Deployment via Streamlit Web Platform]
                         │
                         ▼
[Stage 6: Automated Field Maintenance & PDF Dispatch Reporting]
```

---

## 4. Stage 1: Exploration and Anomaly Auditing

The journey began with an exhaustive audit of the 59,400 raw registry records within `tanzania_waterpoint_eda.ipynb`. Rather than accepting raw column values at face value, exploratory analysis uncovered several critical data anomalies and structural imbalances that dictate modeling strategy.

### Class Imbalance and Operational Asymmetry
The target variable (`status_group`) exhibits substantial class imbalance across the nationwide inventory:
* **functional**: 32,259 waterpoints (54.31%) — The operational majority.
* **non functional**: 22,824 waterpoints (38.42%) — The failure baseline.
* **functional needs repair**: 4,317 waterpoints (7.27%) — The critical minority class (**7.47 to 1 imbalance ratio** against the majority).

From a domain perspective, misclassifying a pump that `needs repair` as fully `functional` is far more detrimental than the reverse, because neglected minor faults quickly escalate into irreversible structural failures. Consequently, standard accuracy is an invalid benchmark; evaluation must prioritize class-weighted recall and ROC-AUC discriminability.

### Sentinel Zeros and Disguised Missing Values
A standard missing value check (`df.isnull().sum()`) misleadingly indicated that 33 out of 40 features were completely populated. However, distributional profiling revealed that data collection agents frequently recorded placeholder zeroes ("sentinel zeroes") in place of unknown measurements:
* **amount_tsh (Waterhead / Payment amount)**: 70.1% of records contain exactly 0.0.
* **population**: 47.8% of entries were unusable (36.0% recorded as 0, and an additional 7,025 records recorded as exactly 1).
* **construction_year**: 34.9% (20,709 records) were encoded as 0. Because Tanzania's modern water program commenced in the late 1960s, a construction year of 0 is physically impossible.
* **gps_height**: 34.4% of records were zero, despite waterpoints being situated across mountainous and plateau regions.
* **Geographic Null Island**: 1,812 records had coordinates recorded at `(0.00, 0.00)`, placing the waterpoints thousands of kilometers away in the Atlantic Ocean off the coast of West Africa.

### Quantifying Predictive Drivers via Bias-Corrected Cramér's V
To evaluate associative strength between categorical predictors and pump functionality without assuming linearity, we computed Bias-Corrected Cramér's V coefficients:
* **quantity / quantity_group (V = 0.31)**: The single most decisive operational driver. Waterpoints recorded as `dry` had a 96.9% failure rate, whereas those with `enough` water maintained a 65.2% functionality rate.
* **payment / payment_type (V = 0.18)**: Governance mechanisms directly influence longevity. Waterpoints with annual community tariffs (`pay annually`) achieved a 75.2% functionality rate, compared to only 44.9% for waterpoints with no tariff structure (`never pay`).
* **region (V = 0.20)**: Geographical functionality exhibited a massive 48.5 percentage point spread, ranging from Iringa (78.2% functional) down to Lindi (29.8% functional).
* **Redundant Collinearity**: Identical feature duplicates were verified and flagged for elimination (`quantity` vs. `quantity_group` with V = 1.0; `payment` vs. `payment_type` with V = 1.0; `recorded_by` possessing zero variance).

---

## 5. Stage 2: Leak-Free Preprocessing and Statistical Selection

All cleaning parameters, hypothesis tests, and encodings in `tanzania_waterpoint_preprocessing_pipeline.ipynb` were executed strictly on the training partition after performing a stratified 80/20 split ($N_{\text{train}} = 47,520$, $N_{\text{test}} = 11,880$). This prevents test set leakage.

### Non-Parametric Age Imputation via Kruskal-Wallis Test
Because 34.9% of construction years were missing, calculating pump age (`2014 - construction_year`) directly produced invalid outliers up to 2,014 years. 

To determine the most appropriate grouping strategy for age imputation, we performed a non-parametric **Kruskal-Wallis H-test** comparing the distribution of known pump ages across technology categories (`extraction_type_class`):
* Test Statistic: $H = 892.36$
* P-value: $p = 1.68 \times 10^{-189}$ (statistically significant at $\alpha = 0.001$)

The results confirmed that pump lifespans vary fundamentally across mechanics: modern rope pumps had a median age of 3.0 years, submersible pumps averaged 8.0 years, handpumps averaged 11.0 years, gravity-fed systems averaged 13.0 years, and unclassified types averaged 18.0 years. Consequently, missing ages were imputed using a **Group-Median Imputation** indexed by `extraction_type_class`.

### Hypothesis-Driven Feature Pruning
To prevent the curse of dimensionality and remove uninformative attributes:
* **One-Way ANOVA F-Tests on Continuous Attributes**: Verified high significance ($p < 0.001$) for `age`, `gps_height`, `amount_tsh`, and `population`. Conversely, `num_private` was confirmed to be non-informative ($F = 1.13, p = 0.323 \ge 0.05$) and was pruned.
* **Chi-Square Contingency Analysis on Categorical Attributes**: Evaluated independence between candidates and `status_group`. Highly significant dependence ($p < 0.001$) was confirmed for 14 operational, geographic, and managerial features.
* **Final Curated Schema**: The raw 40-feature space was systematically distilled into a focused **18-feature schema** retaining maximum predictive signal with minimal redundancy.

---

## 6. Stage 3: Modular Feature Engineering Architecture

To ensure identical feature transformations during cross-validation, hyperparameter tuning, and real-time production inference, all transformations were encapsulated in a custom Scikit-Learn transformer: [`feature_engineer.py`](feature_engineer.py).

The `FeatureEngineer` class executes four primary operations:
1. **High-Cardinality Entity Normalization**: The `funder` and `installer` columns contained thousands of misspelled, abbreviated, or casing-variant strings. The transformer cleans strings, computes the top 30 most frequent organizations solely from training data, and aggregates low-frequency entities into an `'other'` category.
2. **Sentinel Missing Indicators**: Generates explicit boolean indicator flags (`age_missing`, `gps_zero`, `pop_zero`, `tsh_pos`) so downstream tree models can distinguish genuine zero measurements from imputed approximations.
3. **Logarithmic Compression of Skewed Features**: Compresses extreme right-skewness in continuous metrics:
   $$\text{log}_{\text{pop}} = \log(1 + \text{population})$$
   $$\text{log}_{\text{tsh}} = \log(1 + \text{amount\_tsh})$$
   *(Reducing `amount_tsh` skewness from 57.8 down to a well-behaved continuous distribution).*
4. **Institutional Synergy Flag**: Computes `same_funder_installer` ($1$ if funder matches installer, $0$ otherwise) to capture accountability signals where the financing entity directly oversaw installation quality.

---

## 7. Stage 4: Machine Learning Modeling, Overfitting Mitigation & Evaluation

Model experimentation in `tanzania_waterpoint_modeling_pipeline.ipynb` established **XGBoost (Extreme Gradient Boosting)** as the core champion model, evaluated against multiple competitive challenger algorithms.

### The Overfitting Challenge and Root Cause
During preliminary experimentation, unconstrained gradient boosted trees and random forests displayed severe overfitting:
* Training accuracy reached **94.8%**, while holdout cross-validation accuracy plateaued at **79.2%**.
* Due to the 7.47:1 class imbalance, unweighted models prioritized the majority `functional` and `non functional` classes, achieving near-zero recall on the critical `functional needs repair` minority class while memorizing training noise.

### The Mitigation Strategy
To eliminate overfitting and achieve robust out-of-sample generalization, we introduced a four-part regularization and rebalancing framework:
1. **Cost-Sensitive Loss Weighting**: Implemented sample weight adjustment inversely proportional to class frequencies:
   $$w_c = \frac{N}{3 \times N_c}$$
   For XGBoost, this penalizes misclassifications on `functional needs repair` more severely, forcing decision boundaries to prioritize minority class separation without inflating model complexity.
2. **Structural Tree Constraints**:
   * Restricted maximum tree depth (`max_depth = 6`) to prevent deep recursive memorization.
   * Enforced minimum leaf node weights (`min_child_weight = 3` and `min_samples_leaf = 4`) to prevent isolated leaves from capturing noisy records.
3. **Stochastic Regularization**: Configured `subsample = 0.8` (row subsampling per tree) and `colsample_bytree = 0.8` (feature subsampling per split) to introduce stochastic variance reduction.
4. **L1 and L2 Leaf Weight Penalties**: Configured L1 regularization (`reg_alpha = 0.1`) to encourage sparsity, and L2 regularization (`reg_lambda = 1.0`) to shrink extreme leaf weights.

### Model Benchmark Leaderboard (Holdout Test Evaluation, N = 11,880)

All tree models were optimized via Bayesian Hyperparameter Search (**Optuna** with Tree-structured Parzen Estimators) using Stratified 5-Fold Cross-Validation. **XGBoost** emerged as the core production champion, outperforming all challengers across macro-averaged discriminability:

| Architecture Role | Model Classifier | CV Micro F1 | Test Micro F1 | Macro Precision | Macro Recall | Macro F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Core Champion** | **XGBoost (Regularized + Weighted)** | **0.7919** | **0.7950** | **0.738** | **0.637** | **0.664** | **Selected for Deployment** |
| Challenger 1 | Random Forest (Balanced Weights) | 0.7924 | 0.7949 | 0.743 | 0.629 | 0.656 | Benchmark Contender |
| Challenger 2 | CatBoost Classifier | 0.7886 | 0.7945 | 0.725 | 0.643 | 0.668 | Benchmark Contender |
| Challenger 3 | LightGBM Classifier | 0.7903 | 0.7926 | 0.728 | 0.636 | 0.662 | Benchmark Contender |
| Baseline | Scaled Logistic Regression | 0.7350 | 0.7370 | 0.661 | 0.548 | 0.573 | Linear Baseline |
| Control | Uniform Dummy Classifier | 0.3340 | 0.3351 | 0.333 | 0.333 | 0.333 | Random Control |

### Multiclass One-vs-Rest ROC and AUC Analysis
To comprehensively validate model discrimination across all decision thresholds, we computed One-vs-Rest Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC) scores:
* **Class 0 (functional)**: **ROC AUC = 0.892** — Demonstrates excellent separation between operational waterpoints and failing infrastructure.
* **Class 1 (functional needs repair)**: **ROC AUC = 0.838** — Confirms that cost-sensitive weighting provides high sensitivity for preventive repair triage despite the small sample size.
* **Class 2 (non functional)**: **ROC AUC = 0.901** — Outstanding diagnostic capability for isolating completely failed pumps.
* **Macro-Averaged ROC AUC**: **0.877** — Verifies balanced discriminatory capability across the entire operational spectrum.

---

## 8. Stage 5: Operational Deployment via Streamlit Web Platform

To bridge the gap between machine learning code and daily field operations, we developed an all-in-one interactive web application using Streamlit: [`app.py`](app.py).

The application provides three focused operational modules:

### 1. Single Waterpoint Inspection and Real-Time Triage
* **Inspection Form**: Accepts field parameters across all 18 features, organized into logical sections (Technical & Mechanics, Geography, Administrative & Management, Water Supply & Quality).
* **One-Click Pre-filled Scenarios**: Enables technicians to simulate canonical waterpoint profiles (*Healthy Communal Standpipe*, *Aging Handpump Needing Maintenance*, *Dry Broken Gravity System*).
* **Automated Risk Triage Output**:
  * Displays the predicted 3-class status accompanied by class confidence probabilities.
  * Translates raw predictions into an actionable **2x2 Operational Dispatch Matrix**: categorizing pumps as either `Operational` or `Needs Immediate Attention`.
  * Renders a real-time **Model Health Check Badge** (`Model Status: Healthy | Regularized | Low Overfitting Risk`).
  * Provides automated root-cause risk diagnostics highlighting risk factors (e.g., dry water quantity, aged mechanical extraction, zero fee structure).

### 2. Batch CSV Processing and Dispatch Scheduling
* Accepts bulk uploads of field inventories. The system accommodates both **18-column preprocessed CSVs** and **raw 40-column DrivenData competition files** (automatically executing age derivation and group-median imputation).
* Calculates a prioritized **Needs Attention Priority Score** for every waterpoint, sorting pumps based on urgency so field crews can optimize maintenance routes.
* Allows one-click export of prioritized triage schedules in standard CSV format.

### 3. Executive Dashboard and Analytics
* Aggregates inventory metrics: overall functional percentage, preventive maintenance queues, and critical failure counts.
* Visualizes regional risk distributions, extraction type vulnerability, and payment model correlations.

---

## 9. Stage 6: Automated Field Maintenance and PDF Dispatch Reporting

Field technicians operating in rural regions frequently encounter areas with intermittent internet connectivity. To ensure seamless handoffs from software to physical execution, the platform integrates an automated **Field Dispatch PDF Generation Engine** powered by `ReportLab`.

* **One-Click PDF Generation**: Operating personnel can generate formal dispatch reports directly from the Executive Dashboard or Batch Dispatch modules.
* **Structured PDF Layout**:
  * Formal organizational header and official document metadata (generation timestamp, dispatch authority, document tracking ID).
  * High-level Executive KPI Summary (Total waterpoints inspected, operational percentages, immediate attention headcount).
  * Priority Maintenance Roster: Detailed data tables enumerating pump IDs, geographical regions, extraction mechanics, predicted statuses, and required maintenance actions.
  * Formal sign-off and authorization sections for regional water engineers.

---

## 10. Repository File Structure

```text
intelligent-lavoisier/
│
├── tanzania_waterpoint_eda.ipynb                    Phase 1: Exploratory Data Analysis & Sentinel Anomaly Audit
├── tanzania_waterpoint_preprocessing_pipeline.ipynb Phase 2: Statistical Hypothesis Testing & Feature Selection
├── tanzania_waterpoint_modeling_pipeline.ipynb      Phase 3: Model Tuning, Overfitting Mitigation & Evaluation
│
├── feature_engineer.py                              Modular Scikit-Learn Feature Engineering Transformer
├── app.py                                           Streamlit Web Application & PDF Report Dispatch Platform
├── requirements.txt                                 Python Package Dependencies
└── README.md                                        Project Documentation and Technical Architecture
```

---

## 11. Quickstart and Installation Guide

### Prerequisites
* Python 3.10 or higher
* Git command line utility or GitHub Desktop

### 1. Clone the Repository
```bash
git clone <your-github-repository-url>
cd intelligent-lavoisier
```

### 2. Configure Virtual Environment and Dependencies
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Launch the Operational Streamlit Platform
```bash
streamlit run app.py
```
Upon execution, the web application will open in your default browser at `http://localhost:8501`. 

*Note on Model Serialization*: If `best_waterpump_model.pkl` (produced by running `tanzania_waterpoint_modeling_pipeline.ipynb`) is present in the root folder, `app.py` loads the trained XGBoost champion pipeline automatically. If not yet serialized, `app.py` utilizes its built-in calibrated model pipeline immediately.

---

## 12. Summary of Key Analytical Findings

1. **Water Quantity is Paramount**: A pump classified with `dry` water quantity exhibits a 96.9% probability of failure, overriding nearly all other positive indicators.
2. **Community Governance Drives Sustainability**: Revenue collection matters. Waterpoints with communal annual tariffs sustain a 75.2% operational rate, whereas unmonitored systems without payment mechanisms degrade to a 44.9% rate.
3. **The Power of Preventive Maintenance**: Identifying the 7.27% of waterpoints in the `functional needs repair` state allows maintenance teams to replace low-cost wearable components (such as O-rings, valves, and handle bearings) before pump heads fracture or boreholes collapse.
4. **Algorithmic Fairness via Cost-Sensitive Learning**: Applying frequency-weighted loss penalization in XGBoost successfully resolves severe minority class starvation, elevating the ROC-AUC of the `needs repair` class to 0.838 while curbing training memorization and overfitting.
