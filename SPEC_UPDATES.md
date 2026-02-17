🔥 PROJECT UPGRADE: Convert Simulation to Policy Optimization Framework

Load PROJECT_SPEC.md.

We are upgrading the project from a percentile-based shot replacement simulation to a learned threshold-based counterfactual policy optimization framework.

Follow instructions exactly.

Do NOT refactor unrelated files.
Do NOT retrain the model.
Do NOT modify preprocessing.
Only extend simulation and Streamlit.

============================================
PART 1 — Counterfactual Uplift Framework
============================================
Create new file:

src/policy.py

This file must include:

1️⃣ compute_counterfactual_ev3(df, model)

For each 2pt shot in test set:

Copy row

Modify:

PTS_TYPE = 3

SHOT_DIST = 23.5 (representative 3pt distance)

Keep all other context features identical

Predict probability using trained calibrated model

Compute:
EV3_counterfactual = P(make_counterfactual) * 3

Return EV3_counterfactual as numpy array

2️⃣ compute_uplift(df, ev2, ev3_cf)

uplift = ev3_cf − ev2

Return uplift

3️⃣ evaluate_threshold_policies(uplift, ev2, thresholds)

For each threshold δ in thresholds:

mask = uplift > δ

percent_replaced = mean(mask)

total_original_ev = sum(ev2)
total_new_ev = sum(
    where(mask, ev2 + uplift, ev2)
)

delta_total = total_new_ev − total_original_ev

delta_per_100 = (delta_total / len(ev2)) * 100


Store results in dataframe with columns:

threshold
percent_replaced
delta_total
delta_per_100


Return dataframe

============================================
PART 2 — Bootstrap Uncertainty
============================================

Add function in policy.py:

bootstrap_policy_gain(ev2, uplift, threshold, n_boot=500)

For each bootstrap iteration:

- Sample indices with replacement
- Compute policy gain for that sample
- Store delta_per_100


Return:
mean_gain
lower_95
upper_95

Use percentile method.

============================================
PART 3 — Update simulate.py
============================================

Modify simulate.py:

After computing ev2 on test set:

Filter test set to 2pt shots only.

Compute counterfactual EV3 using compute_counterfactual_ev3.

Compute uplift.

Define thresholds:

thresholds = np.linspace(-0.2, 0.5, 30)

Call evaluate_threshold_policies.

Save results to:

outputs/policy_threshold_results.csv

Generate efficiency curve plot:

X-axis: percent_replaced
Y-axis: delta_per_100

Save to:
outputs/policy_efficiency_curve.png

Choose a default deployment threshold:

threshold = 0.05

Run bootstrap_policy_gain for that threshold.

Save bootstrap summary to:

outputs/bootstrap_summary.json

Format:

{
"threshold": 0.05,
"mean_delta_per_100": value,
"ci_lower": value,
"ci_upper": value
}

============================================
PART 4 — Streamlit Upgrade
============================================

Modify app/app.py

Add second tab:

tabs = st.tabs(["Shot Evaluation", "Policy Insights"])

Keep existing functionality in first tab unchanged.

In "Policy Insights" tab:

1️⃣ Load:

policy_threshold_results.csv
bootstrap_summary.json


2️⃣ Display efficiency curve:

st.line_chart or matplotlib plot


3️⃣ Add slider:

threshold = st.slider(
    "Uplift Threshold (EV3 - EV2)",
    min_value=float(min_threshold),
    max_value=float(max_threshold),
    step=0.01,
    value=0.05
)


4️⃣ When slider moves:

- Filter dataframe to nearest threshold
- Display:

    st.metric("Percent Replaced", f"{percent*100:.2f}%")
    st.metric("EV Gain per 100 Shots", f"{delta:.3f}")


5️⃣ If selected threshold equals bootstrap threshold:

Display 95% CI:

    st.write(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

============================================
PART 5 — README Upgrade
============================================

Update README:

Add new section:

Counterfactual Policy Optimization

Explain:

We estimate context-specific counterfactual 3pt EV.

Compute uplift per shot.

Define deployment policies based on uplift thresholds.

Evaluate tradeoff between shot reallocation rate and expected value gain.

Quantify uncertainty via bootstrap confidence intervals.

Add product analogy:

"This mirrors real-world product decision systems where interventions are deployed when expected marginal ROI exceeds a tunable threshold (e.g., marketing spend allocation, recommendation targeting, pricing optimization)."

============================================
DESIGN RULES
============================================

No data leakage

No retraining

No deep learning

No reinforcement learning

Clean modular code

Keep functions pure

No hardcoded paths outside config

============================================
END STATE
============================================

After completion the project should:

Compute individualized counterfactual uplift

Evaluate threshold-based decision policies

Produce policy efficiency curve

Quantify uncertainty via bootstrap

Expose policy exploration dashboard in Streamlit

This upgrade reframes the project from simulation to decision policy optimization.

Implement cleanly and minimally.

Do not over-engineer.