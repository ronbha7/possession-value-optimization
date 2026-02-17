# Shot Value Optimization: Expected Value Modeling and Policy Optimization for NBA Shot Selection

## Why This Project

I love basketball! Not just watching games, but understanding *why* teams win. Shot selection, spacing, efficiency, and small tactical edges can quietly swing outcomes over a season. At the same time, I’m deeply interested in data science and how quantitative models can guide real decisions.

This project combines those two interests. I wanted to move beyond surface-level basketball stats and build something that treats shot selection as a decision system: estimate value, simulate alternatives, and quantify impact. The result is a project that blends my passion for the game with my interest in turning data into actionable strategy.

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

<img width="1654" height="932" alt="image" src="https://github.com/user-attachments/assets/e1fb8458-1b66-4263-bf57-d822f355ac2d" />


**Policy Insights** — We load the policy sweep and bootstrap results (run `python -m src.simulate` first). You see the efficiency curve (percent of 2pt shots reallocated vs. gain per 100 shots). A slider lets you pick an uplift threshold; we show the corresponding percent replaced and EV gain per 100 shots. When the chosen threshold matches the bootstrap default (0.05), we display the 95% CI so you can see how precise the estimated gain is.

<img width="927" height="619" alt="image" src="https://github.com/user-attachments/assets/11ddec78-d093-45a9-819f-53bf1f8dfa36" />

---

## Technical Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Streamlit · SHAP · matplotlib · seaborn · joblib · pyarrow

---

## What Makes This Different:

- **Predictions become expected value** — each shot gets a number in points, comparable across shot type and context.
- **Policy is simulated, not assumed** — thresholds are swept, the tradeoff is plotted, and bootstrap quantifies uncertainty. The output supports a concrete decision, not only a model card.
- **Modeling and strategy share one pipeline** — the same code that trains the model also answers “what if we reallocate only when uplift > X?” So it doubles as a reusable decision engine.
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

---

## The Bigger Picture

Basketball has always been more than entertainment to me. It’s a system of decisions under constraints. This project reflects how I think about both the game and data science, not just predicting outcomes, but improving choices. By turning probabilities into expected value and policy rules, I’m using the tools of data science to explore the same question I’ve always asked while watching games: what’s the smarter shot?

This project combined my love for basketball and my drive to build decision systems that create measurable impact, and I loved working on it!

---
