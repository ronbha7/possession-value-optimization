# Shot Value Optimization: Expected Value Modeling and Policy Optimization for NBA Shot Selection

A product-oriented data science project that turns shot prediction into a **policy optimization system**: we estimate expected value per shot, simulate counterfactual reallocation, and quantify how different decision thresholds affect team-level scoring.

---

## Problem Framing

NBA teams constantly face a single question: *which shots should we take?* Shot type, distance, and context all shape whether a possession ends in points. Small shifts in shot selection—pushing a few low-value attempts toward higher-value ones—can add up over a season.

This project reframes shot prediction as a **policy optimization problem**. We don’t stop at “this shot goes in 40% of the time.” We convert that into expected value (EV), then ask: *if we reallocate 2pt shots to 3pt only when the counterfactual 3pt EV beats the current 2pt EV by at least X, how much do we gain?* That’s the kind of lever a coach or front office can actually use.

---

## What We Built

**Data** — Shot-level logs (distance, defender, shot clock, dribbles, touch time, period, game clock, make/miss). We use a time-based 80/20 split by game so we never leak future information.

**Feature engineering** — We derive time remaining, late-clock and defender-tight flags, shot-distance bins, and missing-data indicators. Player-level aggregates (e.g. make rate, 3pt rate) are computed on the **train set only**; test and cold-start players get global defaults. No W or final margin—we avoid leakage.

**Modeling** — We train baseline, logistic regression, random forest, and XGBoost; we calibrate XGBoost (isotonic) so probabilities are usable for EV. Best test ROC-AUC in our run is ~0.64 (RF); we use the calibrated XGB for all EV and policy work.

**Expected value** — For every shot we compute EV = P(make) × point value (2 or 3). That’s the core output: expected points per attempt.

**Policy layer** — For each 2pt shot we estimate the *counterfactual* 3pt EV (same context, shot at 23.5 ft). Uplift = EV3_counterfactual − EV2. We then sweep thresholds: “Replace only when uplift > δ.” For each δ we get the share of shots reallocated and the gain per 100 shots. We run bootstrap (500 samples) at a default threshold (0.05) and report a 95% CI so we can say how stable the gain is.

---

## System Architecture

```
Raw shot data (CSV)
        ↓
Prep: time-based split, feature engineering, player aggregates (train only)
        ↓
Train: baseline, logreg, RF, XGB, calibrated XGB → models + feature list + defaults
        ↓
Evaluate: ROC, PR, calibration curves + metrics
        ↓
Simulate: EV on test → percentile-based scenarios + policy threshold sweep + bootstrap
        ↓
Interpret: feature importance, SHAP summary and dependence plots
        ↓
Streamlit: Shot Evaluation (P(make), EV, 2pt vs 3pt) + Policy Insights (efficiency curve, threshold slider, 95% CI)
```

---

## Key Results (From Our Run)

- **Model performance** — Calibrated XGB: ROC-AUC ~0.63, PR-AUC ~0.61, Brier ~0.23. Baseline (constant P(make)): ROC-AUC 0.5. The model adds clear separation over the baseline.

- **Policy at 0.05 uplift threshold** — We reallocate 2pt shots whose counterfactual 3pt EV exceeds their 2pt EV by at least 0.05. In our test set that policy yields **~23.0 expected points gained per 100 shots** (95% CI: **22.7–23.2**), from bootstrap over 500 samples. So we get a stable, quantified gain from a simple threshold rule.

- **Percentile-style simulation** — If we replace the bottom 5% of 2pt shots by EV with the average 3pt EV, we see ~458 projected points gained on the test set (~51 per 100 reallocated shots). At 10% and 15% the total gain grows while the per-100 rate stays in a similar band. The policy layer refines this by tying reallocation to *uplift* instead of a fixed percentile.

---

## Product and Analytics Thinking

This system demonstrates:

- **Counterfactual simulation** — “What would this possession be worth if it were a 3pt attempt?” We answer that per shot and use it to define uplift.
- **Policy evaluation** — We don’t pick one magic number. We sweep thresholds and plot the tradeoff: more reallocation vs. total gain. That’s how you compare strategies.
- **Decision optimization** — We turn model output into a rule: “Act when uplift > δ.” δ is tunable; the efficiency curve shows the cost/benefit of different choices.
- **Uncertainty** — We report a bootstrap 95% CI for the default policy so stakeholders see both the expected gain and the range of plausible outcomes.

This mirrors real-world product systems where we deploy an intervention only when expected marginal value exceeds a threshold—whether that’s marketing spend, recommendation targeting, or pricing. Here the “intervention” is reallocating a 2pt shot to a 3pt.

---

## Interactive Dashboard (Streamlit)

The app has two tabs:

**Shot Evaluation** — You set shot context (distance, defender distance, shot clock, dribbles, touch time, period, game clock, 2pt vs 3pt). We show predicted P(make) and expected value. We compare *this* 2pt shot to a *context-matched* 3pt (same defender/clock, typical arc distance) and recommend 2pt favored, 3pt favored, or roughly even, using a 5% EV threshold so we don’t overcall tiny differences.

**Policy Insights** — We load the policy sweep and bootstrap results (run `python -m src.simulate` first). You see the efficiency curve (percent of 2pt shots reallocated vs. gain per 100 shots). A slider lets you pick an uplift threshold; we show the corresponding percent replaced and EV gain per 100 shots. When the chosen threshold matches the bootstrap default (0.05), we display the 95% CI so you can see how precise the estimated gain is.

*(Add 1–2 screenshots here: e.g. Shot Evaluation with a recommendation, and Policy Insights with the curve and slider.)*

---

## Technical Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Streamlit · SHAP · matplotlib · seaborn · joblib · pyarrow

---

## What Makes This Different

Most shot-prediction projects stop at accuracy: ROC-AUC, log loss, maybe calibration. This one goes further:

- We **convert predictions into expected value** so every shot has a number in “points” that we can compare across types and contexts.
- We **simulate policy**, not just a single scenario. We sweep thresholds and plot the tradeoff, then quantify uncertainty with bootstrap. That’s how you support a decision, not just a model card.
- We **bridge modeling and strategy**: the same pipeline that trains the model also powers “what if we only reallocate when uplift > X?” That’s the difference between a one-off analysis and a reusable decision engine.

---

## How to Run

From the project root:

1. **Prep** — `python -m src.prep` (reads raw data, splits by game, builds features, saves processed data).
2. **Train** — `python -m src.train` (trains models, calibrates XGB, saves artifacts and global defaults).
3. **Evaluate** — `python -m src.evaluate` (metrics and ROC/PR/calibration plots).
4. **Simulate** — `python -m src.simulate` (EV, policy sweep, bootstrap; writes policy and bootstrap files).
5. **Interpret** — `python -m src.interpret` (feature importance and SHAP plots).
6. **App** — `streamlit run app/streamlit_app.py`. Open the URL (e.g. http://localhost:8501). The **Policy Insights** tab needs step 4 run first so the efficiency curve and bootstrap CI have data.

---

## Future Work

- Lineup- or unit-aware simulation (who is on the floor).
- Richer uncertainty (e.g. Bayesian or posterior over policy gain).
- Real-time or game-state conditioning for in-game tools.
- Optional: reinforcement learning or bandits for adaptive policy tuning.

---

## The Big Idea

Predictive modeling alone doesn’t change decisions. This project turns a make-probability model into an **expected-value and policy layer**: we quantify gain per shot, sweep decision thresholds, and report uncertainty. That’s how you go from “the model is accurate” to “here’s how much we gain if we deploy this rule.” For product or analytics DS roles, that’s the mindset that matters.

---

*Add your GitHub repo link, a short Loom walkthrough, or a live Streamlit link here when you have them.*
