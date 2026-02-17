"""
Streamlit internal tool: input shot context → P(make), EV, 2pt vs 3pt recommendation.
Run from project root: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# Ensure project root is on path when running streamlit run app/streamlit_app.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.features import get_feature_columns
from src.utils import load_json, load_pkl
from src import config
from src.utils import game_clock_to_seconds


def shot_dist_bin_value(shot_dist: float) -> int:
    """Map SHOT_DIST to ordinal bin [0,3),[3,10),[10,16),[16,22),[22,100) -> 0,1,2,3,4."""
    bins = config.SHOT_DIST_BINS  # [0, 3, 10, 16, 22, 100]
    for i in range(len(bins) - 1):
        if bins[i] <= shot_dist < bins[i + 1]:
            return i
    return len(bins) - 2  # 4 for >= 22


@st.cache_resource
def load_model_and_artifacts():
    """Load calibrated model, feature list, and global defaults once."""
    models_dir = config.MODELS_DIR
    model = load_pkl(models_dir / "xgb_calibrated.pkl")
    feature_list = load_json(models_dir / "feature_list.json")
    global_defaults = load_json(models_dir / "global_defaults.json")
    return model, feature_list, global_defaults


def build_feature_row(
    shot_dist: float,
    close_def_dist: float,
    shot_clock: float,
    dribbles: float,
    touch_time: float,
    period: int,
    time_remaining_sec: float,
    pts_type: int,
    global_defaults: dict,
) -> pd.DataFrame:
    """Build one-row feature DataFrame in model order (same preprocessing as training)."""
    is_late_clock = 1 if shot_clock < config.LATE_CLOCK_THRESHOLD_SEC else 0
    defender_tight = 1 if close_def_dist < config.DEFENDER_TIGHT_THRESHOLD_FT else 0
    bin_val = shot_dist_bin_value(shot_dist)

    row = dict(global_defaults)
    row.update({
        "SHOT_DIST": shot_dist,
        "CLOSE_DEF_DIST": close_def_dist,
        "SHOT_CLOCK": shot_clock,
        "DRIBBLES": dribbles,
        "TOUCH_TIME": touch_time,
        "PERIOD": period,
        "time_remaining_sec": time_remaining_sec,
        "is_late_clock": is_late_clock,
        "defender_tight": defender_tight,
        "shot_dist_bin": bin_val,
        "is_shot_clock_missing": 0,
        "is_dribbles_missing": 0,
        "is_touch_time_missing": 0,
        "PTS_TYPE": pts_type,
    })
    feature_cols = get_feature_columns()
    X = pd.DataFrame([row])[feature_cols].astype(np.float64)
    return X


# Relative EV threshold: only recommend 3pt/2pt when the better option is ahead by this much
EV_THRESHOLD = 0.05  # 5%


def main():
    st.set_page_config(page_title="Shot value", layout="centered")
    st.title("Possession value — shot probability & EV")

    tab_shot, tab_policy = st.tabs(["Shot Evaluation", "Policy Insights"])

    # ---------- Tab 1: Shot Evaluation (unchanged) ----------
    with tab_shot:
        try:
            model, feature_list, global_defaults = load_model_and_artifacts()
        except FileNotFoundError as e:
            st.error(f"Models not found. Run prep and train first. {e}")
            return

        typical_3pt_dist = global_defaults.get("typical_3pt_shot_dist", 23.5)
        defaults_for_row = {k: v for k, v in global_defaults.items() if k != "typical_3pt_shot_dist"}

        st.sidebar.header("Shot context")
        shot_dist = st.sidebar.slider("Shot distance (ft)", 0.0, 30.0, 15.0, 0.5)
        close_def_dist = st.sidebar.slider("Closest defender distance (ft)", 0.0, 15.0, 4.0, 0.5)
        shot_clock = st.sidebar.slider("Shot clock (sec)", 0.0, 24.0, 12.0, 0.5)
        dribbles = st.sidebar.slider("Dribbles", 0.0, 20.0, 2.0, 0.5)
        touch_time = st.sidebar.slider("Touch time (sec)", 0.0, 25.0, 2.0, 0.1)
        period = st.sidebar.selectbox("Period", [1, 2, 3, 4], index=0)
        game_clock_str = st.sidebar.text_input("Game clock (MM:SS)", "8:00")
        time_remaining_sec = game_clock_to_seconds(game_clock_str)
        if np.isnan(time_remaining_sec):
            time_remaining_sec = 480.0
        pts_type = st.sidebar.radio("Shot type", [2, 3], index=0, format_func=lambda x: f"{x} pt")

        X = build_feature_row(
            shot_dist, close_def_dist, shot_clock, dribbles, touch_time,
            period, float(time_remaining_sec), pts_type, defaults_for_row,
        )
        p = model.predict_proba(X)[0, 1]
        ev = p * pts_type

        st.subheader("Prediction")
        st.metric("P(make)", f"{p:.2%}")
        st.metric("Expected value", f"{ev:.3f} pts")

        X_2pt = build_feature_row(
            shot_dist, close_def_dist, shot_clock, dribbles, touch_time,
            period, float(time_remaining_sec), 2, defaults_for_row,
        )
        X_3pt_context_matched = build_feature_row(
            typical_3pt_dist, close_def_dist, shot_clock, dribbles, touch_time,
            period, float(time_remaining_sec), 3, defaults_for_row,
        )
        ev2 = model.predict_proba(X_2pt)[0, 1] * 2
        ev3 = model.predict_proba(X_3pt_context_matched)[0, 1] * 3

        st.subheader("EV comparison (context-matched)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("EV — this 2pt shot", f"{ev2:.3f} pts")
        with col2:
            st.metric(f"EV — 3pt at {typical_3pt_dist:.1f} ft (same D/clock)", f"{ev3:.3f} pts")

        if ev3 > ev2 * (1 + EV_THRESHOLD):
            st.success("**Recommendation: 3pt favored** — stepping back to the arc has higher expected value in this context.")
        elif ev2 > ev3 * (1 + EV_THRESHOLD):
            st.info("**Recommendation: 2pt favored** — this 2pt shot has higher expected value than a 3pt from the same situation.")
        else:
            st.warning("**Roughly even** — either shot is reasonable; difference in EV is small.")

        st.caption("In practice, we only reallocate a fraction of 2pt attempts; use this to spot low-EV 2s that could be replaced with 3s when possible.")

    # ---------- Tab 2: Policy Insights ----------
    with tab_policy:
        policy_path = config.OUTPUTS_DIR / "policy_threshold_results.csv"
        bootstrap_path = config.OUTPUTS_DIR / "bootstrap_summary.json"
        if not policy_path.exists() or not bootstrap_path.exists():
            st.info("Run `python -m src.simulate` to generate policy results, then refresh this page.")
            return

        st.markdown(
            "We use a **threshold-based policy**: for each 2pt shot we estimate how much *more* expected value we’d get "
            "if the same situation were a 3pt shot (same defender, clock, etc.). That difference is **uplift** (EV3 − EV2). "
            "We only “replace” a 2pt with a 3pt when uplift is above a chosen **threshold**. This tab shows the tradeoff "
            "between how many shots get reallocated and how much gain we expect."
        )

        policy_df = pd.read_csv(policy_path)
        bootstrap_summary = load_json(bootstrap_path)
        min_threshold = float(policy_df["threshold"].min())
        max_threshold = float(policy_df["threshold"].max())
        bootstrap_threshold = float(bootstrap_summary["threshold"])

        st.subheader("Efficiency curve")
        st.caption(
            "Each point is a different threshold. **X-axis:** share of 2pt shots we’d reallocate (replace with 3pt). "
            "**Y-axis:** expected points gained per 100 shots if we applied that policy. Moving right = more shots replaced; "
            "the curve shows how total gain changes."
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(policy_df["percent_replaced"], policy_df["delta_per_100"], "o-", color="steelblue", markersize=3)
        ax.set_xlabel("Percent replaced (uplift > threshold)")
        ax.set_ylabel("Delta per 100 shots")
        ax.set_title("Policy efficiency curve")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Choose a threshold")
        st.caption(
            "**Uplift threshold (EV3 − EV2):** Only reallocate a 2pt shot if switching to a 3pt would add *at least* this much "
            "expected value. Higher = we act on fewer shots (stricter policy); lower = we act on more shots."
        )
        default_threshold = 0.05 if min_threshold <= 0.05 <= max_threshold else min_threshold
        threshold = st.slider(
            "Uplift threshold (EV3 − EV2)",
            min_value=min_threshold,
            max_value=max_threshold,
            step=0.01,
            value=default_threshold,
        )

        idx = (policy_df["threshold"] - threshold).abs().idxmin()
        row = policy_df.loc[idx]
        st.metric("Percent replaced", f"{row['percent_replaced'] * 100:.2f}%")
        st.caption(
            "**Percent replaced:** Of all 2pt shots in the test set, the share we would reallocate to a 3pt under this threshold. "
            "E.g. 80% means we’d swap 4 out of 5 such shots when their uplift exceeds the chosen value."
        )
        st.metric("EV gain per 100 shots", f"{row['delta_per_100']:.3f}")
        st.caption(
            "**EV gain per 100 shots:** How many extra expected points we’d get per 100 2pt shots if we applied this policy. "
            "Computed as (total EV after reallocation − total EV before) ÷ number of 2pt shots × 100."
        )

        if abs(threshold - bootstrap_threshold) < 0.01:
            ci_lower = bootstrap_summary["ci_lower"]
            ci_upper = bootstrap_summary["ci_upper"]
            st.write(f"**95% confidence interval** for this threshold: [{ci_lower:.3f}, {ci_upper:.3f}] (EV gain per 100 shots). "
                     "Based on 500 bootstrap samples; the true gain is likely in this range.")


if __name__ == "__main__":
    main()
