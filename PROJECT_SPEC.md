# PROJECT_SPEC.md — Possession Value Optimization System (NBA)

## 0) Project Identity
**Goal:** Build a product-style decision system that estimates **shot conversion probability** from context, converts it to **expected possession value (EV)**, then simulates **strategy shifts** (5%, 10%, 15% reallocation) to quantify projected scoring gains.

**Core outputs:**
1) Calibrated probability model: P(make | context + player profile)
2) Expected Value engine: EV = P(make) * PTS_TYPE
3) Strategy simulator: reallocate lowest-EV 2pt shots → higher-EV alternative
4) Interpretability: SHAP + feature importance
5) Streamlit internal tool: input context → P(make), EV, recommendation

---

## 1) Dataset & Schema
Dataset contains the following columns (exact names):
- GAME_ID
- MATCHUP (not used)
- LOCATION (not used)
- W (NOT used; leakage)
- FINAL_MARGIN (NOT used; leakage-ish)
- SHOT_NUMBER
- PERIOD
- GAME_CLOCK (string clock like "MM:SS" or similar)
- SHOT_CLOCK (numeric; can be null)
- DRIBBLES (numeric; can be null)
- TOUCH_TIME (numeric; can be null)
- SHOT_DIST (numeric)
- PTS_TYPE (2 or 3)
- SHOT_RESULT (string made/miss or equivalent)
- CLOSEST_DEFENDER (not used)
- CLOSEST_DEFENDER_PLAYER_ID (not used)
- CLOSE_DEF_DIST (numeric)
- FGM (0/1)
- PTS (0/2/3)
- player_name (optional, not required)
- player_id (key)

### Target
Binary classification target:
- y = FGM (1 if made else 0)

---

## 2) Repository Structure (Hybrid)
possession_value_optimization/
├── data/
│   ├── raw.csv
│   └── processed.parquet (preferred) or processed.csv
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── prep.py
│   ├── features.py
│   ├── split.py
│   ├── train.py
│   ├── evaluate.py
│   ├── simulate.py
│   └── interpret.py
├── app/
│   └── streamlit_app.py
├── notebooks/
│   ├── 00_quick_sanity_checks.ipynb
│   ├── 01_model_report_plots.ipynb
│   └── 02_shap_viz.ipynb
├── models/
│   ├── baseline.json (optional)
│   ├── logreg.pkl
│   ├── rf.pkl
│   ├── xgb.pkl
│   ├── xgb_calibrated.pkl
│   └── feature_list.json
├── outputs/
│   ├── metrics.csv
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── calibration_curve.png
│   ├── ev_distribution.png
│   ├── ev_by_context.png
│   ├── simulation_results.csv
│   ├── simulation_plot.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── shap_dependence_*.png
├── requirements.txt
└── README.md

---

## 3) Core Design Rules
### 3.1 Leakage prevention
- Do NOT use W or FINAL_MARGIN as model features.
- Do NOT compute player aggregates on the full dataset.
- Compute player aggregates using TRAIN only.
- When merging aggregates into TEST, players not present in TRAIN must be filled with global TRAIN averages.

### 3.2 Time-aware split
Split by GAME_ID:
1) Sort unique GAME_ID ascending.
2) Train games = first 80% of unique GAME_ID
3) Test games = last 20% of unique GAME_ID

---

## 4) Feature Engineering

### 4.1 Base shot-level features (inputs)
Use as numeric:
- SHOT_DIST
- CLOSE_DEF_DIST
- SHOT_CLOCK
- DRIBBLES
- TOUCH_TIME
- PERIOD
- time_remaining_sec (engineered from GAME_CLOCK)
- is_late_clock (engineered)
- defender_tight (engineered)
- shot_dist_bin (engineered; can be ordinal or one-hot)

Note:
- PTS_TYPE is NOT a model feature (optional).
  - Primary plan: exclude from model to avoid the model learning 2pt vs 3pt directly.
  - PTS_TYPE will be used for EV computation.
  - (If including PTS_TYPE later, document the reason; default = exclude.)

### 4.2 Engineered features (E1)
Create:
- time_remaining_sec:
  - Parse GAME_CLOCK (e.g., "MM:SS") -> seconds = 60*MM + SS
- is_late_clock = 1 if SHOT_CLOCK < 4 else 0 (handle null -> 0 and add indicator)
- defender_tight = 1 if CLOSE_DEF_DIST < 3 else 0
- shot_dist_bin:
  - bins: [0,3), [3,10), [10,16), [16,22), [22,100)
  - store as categorical (one-hot) OR ordinal integer

### 4.3 Player-level aggregates (P2 bundle) computed on TRAIN only
Group by player_id on TRAIN and compute:
- player_fgm_rate = mean(FGM)
- player_3pt_rate = mean(FGM where PTS_TYPE==3) (if player has no 3pt in train -> fill later)
- player_avg_shot_dist = mean(SHOT_DIST)
- player_avg_close_def_dist = mean(CLOSE_DEF_DIST)
- player_avg_shot_clock = mean(SHOT_CLOCK)
- player_shot_volume = count(rows)
- player_avg_dribbles = mean(DRIBBLES)
- player_avg_touch_time = mean(TOUCH_TIME)
- player_pct_3pt = mean(PTS_TYPE==3)

Cold-start fill values (global TRAIN):
- global_fgm_rate
- global_3pt_rate
- global_avg_* for all avg fields
- global_pct_3pt
- For volume, use global median volume or 0, and also consider log1p transform.

### 4.4 Missing data handling
- For SHOT_CLOCK, DRIBBLES, TOUCH_TIME:
  - Create missing indicators: is_shot_clock_missing, is_dribbles_missing, is_touch_time_missing
  - Impute missing with TRAIN median.
- For CLOSE_DEF_DIST, SHOT_DIST:
  - Drop rows if missing (these are core).
- Standardize numeric features ONLY for Logistic Regression.

---

## 5) Modeling

### 5.1 Baseline
Baseline probability p0 = mean(FGM) on TRAIN.
Predict p_hat = p0 for all TEST.
Evaluate: log-loss, Brier, ROC-AUC (AUC may be ~0.5 baseline).

### 5.2 Logistic Regression
- Pipeline: impute -> standardize -> LogisticRegression(C=1.0, penalty="l2", max_iter=2000)
- Evaluate on TEST.

### 5.3 Random Forest
- RandomForestClassifier
- RandomizedSearch (small):
  - n_estimators: [200, 400]
  - max_depth: [6, 10, None]
  - min_samples_leaf: [50, 200]
  - max_features: ["sqrt", 0.5]
- Evaluate on TEST.

### 5.4 XGBoost (primary)
- XGBClassifier (binary:logistic)
- Use train/valid split inside TRAIN (time-aware within train if easy; else random stratified).
- Early stopping.
- RandomizedSearch (small) OR fixed good defaults:
  - n_estimators: 2000 (with early stopping)
  - learning_rate: 0.03–0.1
  - max_depth: 4–8
  - subsample: 0.7–1.0
  - colsample_bytree: 0.6–1.0
  - min_child_weight: 1–10
- Primary metric: logloss or auc.

### 5.5 Calibration (required)
Calibrate XGBoost probabilities using TRAIN only:
- Preferred: IsotonicRegression via CalibratedClassifierCV(method="isotonic", cv=3)
- Alternative: Platt scaling (sigmoid) if isotonic overfits.

Use calibrated model for EV + simulation.

---

## 6) Evaluation & Reporting

Compute on TEST for each model:
- ROC-AUC
- PR-AUC
- Log-loss
- Brier score
- (Optional) accuracy at 0.5 threshold, but not emphasized.

Generate plots:
- ROC curve (best model + baseline)
- PR curve
- Calibration curve (uncalibrated vs calibrated)
- Confusion matrix at chosen threshold (optional)

Save:
- outputs/metrics.csv with rows = models, cols = metrics
- outputs/*.png

---

## 7) Expected Value (EV) Engine

For each TEST row:
- p = calibrated_model.predict_proba(X_test)[:,1]
- EV = p * PTS_TYPE

Outputs:
- EV distribution histogram
- EV grouped summaries:
  - EV by shot_dist_bin
  - EV by defender_tight
  - EV by is_late_clock

Save:
- outputs/ev_distribution.png
- outputs/ev_by_context.png (bar or line plots)

---

## 8) Strategy Simulation (5%, 10%, 15%)

### Objective
Quantify projected scoring gain if we reduce low-EV 2pt attempts and replace them with higher-EV attempts.

### Simulation definition (weekend-clean, defensible)
1) Identify all TEST shots with PTS_TYPE==2.
2) Compute EV for each (EV2).
3) For each scenario s in {0.05, 0.10, 0.15}:
   - Remove the bottom s fraction of 2pt shots by EV2 (lowest expected value).
   - Replacement EV assumption:
     - Use the average EV of existing TEST 3pt shots (EV3_mean), OR
     - Use the average EV of top-quartile 3pt shots by EV (choose one; default = EV3_mean for conservatism).
   - Projected delta points:
     delta = (num_replaced * replacement_EV) - (sum(EV of removed shots))
4) Report:
   - num_replaced
   - removed_EV_sum
   - replacement_EV
   - delta_points
   - delta_per_100_shots (normalize)

Save:
- outputs/simulation_results.csv
- outputs/simulation_plot.png (scenario vs delta)

---

## 9) Interpretability (SHAP)

Use the underlying XGBoost model (pre-calibration is fine for SHAP; document that calibration is monotonic).
Artifacts:
- Feature importance (gain)
- SHAP summary plot (top 15)
- SHAP dependence plots for:
  - CLOSE_DEF_DIST
  - SHOT_DIST
  - SHOT_CLOCK

Save in outputs/:
- feature_importance.png
- shap_summary.png
- shap_dependence_close_def_dist.png
- shap_dependence_shot_dist.png
- shap_dependence_shot_clock.png

---

## 10) Streamlit App (Internal Tool)

### Inputs (user controls)
- SHOT_DIST (slider)
- CLOSE_DEF_DIST (slider)
- SHOT_CLOCK (slider)
- DRIBBLES (slider)
- TOUCH_TIME (slider)
- PERIOD (selectbox)
- GAME_CLOCK (MM:SS input) OR time_remaining_sec slider
- PTS_TYPE (radio: 2 or 3)

### Player profile handling
Use a toggle:
- "Average player" (default): use global aggregate values
Optional later:
- dropdown to pick a player_id present in training aggregates

### Outputs
- Predicted P(make)
- Expected Value = P(make)*PTS_TYPE
- Recommendation text:
  - If PTS_TYPE==3 and EV3 > EV2_baseline (computed for same inputs but PTS_TYPE=2) -> "3pt favored"
  - Else -> "2pt favored"
- Show EV comparison panel:
  - EV2 vs EV3 for same context (compute both by setting PTS_TYPE accordingly)

### App requirements
- Must load:
  - models/xgb_calibrated.pkl
  - models/feature_list.json
  - global aggregate defaults (save as JSON during training)
- Must build a single-row DataFrame in correct column order
- Must apply same preprocessing (impute/feature engineering) as training

---

## 11) Implementation Sequence (Do in this order)
1) src/config.py: constants, bins, file paths
2) src/utils.py: clock parsing, plotting helpers, save/load helpers
3) src/split.py: time-based split by GAME_ID
4) src/features.py: engineered features + missing indicators + binning
5) src/prep.py: read raw, clean, split, compute aggregates, merge, save processed
6) src/train.py: train baseline/logreg/rf/xgb + calibrate, save models + feature list
7) src/evaluate.py: metrics + plots, write outputs/metrics.csv
8) src/simulate.py: EV computation + strategy simulation + plots
9) src/interpret.py: SHAP + feature importance plots
10) app/streamlit_app.py: UI + prediction + EV + recommendation
11) notebooks: only for quick sanity + final visuals (optional)

---

## 12) README Requirements (Recruiter-facing)
Include:
- 3–5 sentence summary (product framing)
- Data description
- Leakage prevention + split strategy
- Model comparison table (metrics)
- Key charts (embedded images)
- Optimization simulation results (table + 1 plot)
- How to run:
  - `python -m src.prep`
  - `python -m src.train`
  - `python -m src.evaluate`
  - `python -m src.simulate`
  - `python -m src.interpret`
  - `streamlit run app/streamlit_app.py`

---

## 13) Success Criteria (Definition of Done)
- Processed dataset saved with all features.
- Metrics table produced for baseline/logreg/rf/xgb + calibrated xgb.
- Calibration curve shows improvement post-calibration.
- Simulation outputs for 5/10/15% scenarios saved.
- SHAP plots generated.
- Streamlit app runs locally and produces P(make) + EV + recommendation.
