"""
Interpretability: feature importance (gain) and SHAP (summary + dependence plots).
Uses the underlying XGBoost model (pre-calibration); calibration is monotonic so
SHAP interpretation still holds. Run: python -m src.interpret
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from . import config
from .features import get_feature_columns
from .utils import ensure_dir, load_processed, load_pkl, save_figure


def run() -> None:
    """Load XGB and test data, compute feature importance and SHAP, save plots."""
    ensure_dir(config.OUTPUTS_DIR)
    df = load_processed()
    test_df = df.loc[df["SPLIT"] == "test"].copy()
    feature_cols = get_feature_columns()
    X_test = test_df[feature_cols].astype(np.float64)

    xgb = load_pkl(config.MODELS_DIR / "xgb.pkl")

    # --- Feature importance (gain) ---
    imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.tail(20).plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("XGBoost feature importance (gain)")
    save_figure(fig, "feature_importance.png")
    plt.close(fig)
    print("Saved feature_importance.png")

    # --- SHAP (use a sample for speed) ---
    n_sample = min(2000, len(X_test))
    X_sample = X_test.sample(n=n_sample, random_state=42) if len(X_test) > n_sample else X_test

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot (top 15) — creates its own figure
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, max_display=15, show=False)
    plt.tight_layout()
    save_figure(plt.gcf(), "shap_summary.png")
    plt.close()
    print("Saved shap_summary.png")

    # Dependence plots for CLOSE_DEF_DIST, SHOT_DIST, SHOT_CLOCK
    for feat, fname in [
        ("CLOSE_DEF_DIST", "shap_dependence_close_def_dist.png"),
        ("SHOT_DIST", "shap_dependence_shot_dist.png"),
        ("SHOT_CLOCK", "shap_dependence_shot_clock.png"),
    ]:
        if feat not in feature_cols:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.dependence_plot(
            feat,
            shap_values,
            X_sample,
            feature_names=feature_cols,
            ax=ax,
            show=False,
        )
        plt.tight_layout()
        save_figure(fig, fname)
        plt.close(fig)
        print(f"Saved {fname}")

    print("Interpret done.")


if __name__ == "__main__":
    run()
