"""
EV engine and strategy simulation: compute EV with calibrated model, plot distribution
and EV by context; simulate 5%/10%/15% reallocation; threshold-based policy evaluation.
Run: python -m src.simulate
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config
from .features import get_feature_columns
from .policy import (
    compute_counterfactual_ev3,
    compute_uplift,
    evaluate_threshold_policies,
    bootstrap_policy_gain,
)
from .utils import ensure_dir, load_processed, load_pkl, save_figure, save_json


def run() -> pd.DataFrame:
    """Compute EV on TEST, generate EV plots, run strategy simulation, save outputs."""
    ensure_dir(config.OUTPUTS_DIR)
    df = load_processed()
    test_df = df.loc[df["SPLIT"] == "test"].copy()

    feature_cols = get_feature_columns()
    X_test = test_df[feature_cols].astype(np.float64)
    xgb_cal = load_pkl(config.MODELS_DIR / "xgb_calibrated.pkl")

    # EV = p(make) * PTS_TYPE
    p = xgb_cal.predict_proba(X_test)[:, 1]
    test_df = test_df.assign(EV=p * test_df["PTS_TYPE"].values)

    # --- EV distribution histogram ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(test_df["EV"], bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Expected value (points)")
    ax.set_ylabel("Count")
    ax.set_title("EV distribution (TEST shots)")
    save_figure(fig, "ev_distribution.png")
    plt.close(fig)

    # --- EV by context: shot_dist_bin, defender_tight, is_late_clock ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ev_bin = test_df.groupby("shot_dist_bin", dropna=False)["EV"].mean()
    ev_bin.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_xlabel("shot_dist_bin")
    axes[0].set_ylabel("Mean EV")
    axes[0].set_title("EV by shot_dist_bin")
    axes[0].tick_params(axis="x", rotation=0)

    ev_def = test_df.groupby("defender_tight", dropna=False)["EV"].mean()
    ev_def.plot(kind="bar", ax=axes[1], color="steelblue", edgecolor="black")
    axes[1].set_xlabel("defender_tight")
    axes[1].set_ylabel("Mean EV")
    axes[1].set_title("EV by defender_tight")
    axes[1].tick_params(axis="x", rotation=0)

    ev_clock = test_df.groupby("is_late_clock", dropna=False)["EV"].mean()
    ev_clock.plot(kind="bar", ax=axes[2], color="steelblue", edgecolor="black")
    axes[2].set_xlabel("is_late_clock")
    axes[2].set_ylabel("Mean EV")
    axes[2].set_title("EV by is_late_clock")
    axes[2].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    save_figure(fig, "ev_by_context.png")
    plt.close(fig)

    # --- Strategy simulation: 5%, 10%, 15% of 2pt replaced by 3pt EV ---
    two_pt = test_df.loc[test_df["PTS_TYPE"] == 2].copy()
    three_pt = test_df.loc[test_df["PTS_TYPE"] == 3]
    replacement_EV = float(three_pt["EV"].mean()) if len(three_pt) > 0 else 0.0

    scenarios = [0.05, 0.10, 0.15]
    rows = []
    for s in scenarios:
        two_sorted = two_pt.sort_values("EV", ascending=True)
        n_replace = max(1, int(len(two_sorted) * s))
        removed = two_sorted.head(n_replace)
        removed_EV_sum = float(removed["EV"].sum())
        delta_points = (n_replace * replacement_EV) - removed_EV_sum
        delta_per_100 = (delta_points / n_replace * 100) if n_replace else 0.0
        rows.append({
            "scenario_pct": s,
            "num_replaced": n_replace,
            "removed_EV_sum": round(removed_EV_sum, 4),
            "replacement_EV": round(replacement_EV, 4),
            "delta_points": round(delta_points, 4),
            "delta_per_100_shots": round(delta_per_100, 4),
        })

    sim_df = pd.DataFrame(rows)
    sim_df.to_csv(config.OUTPUTS_DIR / "simulation_results.csv", index=False)
    print("Simulation results:")
    print(sim_df.to_string(index=False))

    # --- Simulation plot: scenario vs delta ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"{int(s*100)}%" for s in scenarios], sim_df["delta_points"], color="steelblue", edgecolor="black")
    ax.set_xlabel("Reallocation scenario")
    ax.set_ylabel("Projected delta (points)")
    ax.set_title("Projected scoring gain: replace bottom % of 2pt with avg 3pt EV")
    save_figure(fig, "simulation_plot.png")
    plt.close(fig)

    # --- Policy: counterfactual uplift + threshold sweep + bootstrap ---
    two_pt = test_df.loc[test_df["PTS_TYPE"] == 2].copy()
    ev2 = two_pt["EV"].values
    ev3_cf = compute_counterfactual_ev3(two_pt, xgb_cal)
    uplift = compute_uplift(two_pt, ev2, ev3_cf)

    thresholds = np.linspace(-0.2, 0.5, 30)
    policy_df = evaluate_threshold_policies(uplift, ev2, thresholds)
    policy_df.to_csv(config.OUTPUTS_DIR / "policy_threshold_results.csv", index=False)
    print("Policy threshold results saved to outputs/policy_threshold_results.csv")

    # Efficiency curve: percent_replaced vs delta_per_100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(policy_df["percent_replaced"], policy_df["delta_per_100"], "o-", color="steelblue", markersize=4)
    ax.set_xlabel("Percent replaced (uplift > threshold)")
    ax.set_ylabel("Delta per 100 shots")
    ax.set_title("Policy efficiency curve")
    save_figure(fig, "policy_efficiency_curve.png")
    plt.close(fig)

    threshold_default = 0.05
    mean_gain, ci_lower, ci_upper = bootstrap_policy_gain(ev2, uplift, threshold_default, n_boot=500)
    bootstrap_summary = {
        "threshold": threshold_default,
        "mean_delta_per_100": round(mean_gain, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }
    save_json(bootstrap_summary, config.OUTPUTS_DIR / "bootstrap_summary.json")
    print(f"Bootstrap (threshold={threshold_default}): mean delta_per_100={mean_gain:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")

    print("Saved ev_distribution.png, ev_by_context.png, simulation_results.csv, simulation_plot.png, "
          "policy_threshold_results.csv, policy_efficiency_curve.png, bootstrap_summary.json")
    return sim_df


if __name__ == "__main__":
    run()
