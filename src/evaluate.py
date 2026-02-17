"""
Evaluation: compute metrics on TEST for each model, generate ROC/PR/calibration plots,
save outputs/metrics.csv and outputs/*.png. Run: python -m src.evaluate
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from . import config
from .features import get_feature_columns
from .utils import ensure_dir, load_processed, load_json, load_pkl, save_figure


def _get_test_data():
    """Load processed data and return X_test, y_test as numpy arrays."""
    df = load_processed()
    test_df = df.loc[df["SPLIT"] == "test"].copy()
    feature_cols = get_feature_columns()
    X_test = test_df[feature_cols].astype(np.float64)
    y_test = test_df["FGM"].values
    return X_test, y_test


def _predict_proba_baseline(y_test, p0):
    """Baseline: constant p0 for every row."""
    return np.full(len(y_test), p0)


def run() -> pd.DataFrame:
    """Load models and test data, compute metrics, generate plots, save outputs."""
    ensure_dir(config.OUTPUTS_DIR)
    X_test, y_test = _get_test_data()

    # Load models (baseline is JSON)
    baseline = load_json(config.MODELS_DIR / "baseline.json")
    p0 = baseline["p0"]
    logreg = load_pkl(config.MODELS_DIR / "logreg.pkl")
    rf = load_pkl(config.MODELS_DIR / "rf.pkl")
    xgb = load_pkl(config.MODELS_DIR / "xgb.pkl")
    xgb_cal = load_pkl(config.MODELS_DIR / "xgb_calibrated.pkl")

    # Predictions (probability of positive class)
    preds = {
        "baseline": _predict_proba_baseline(y_test, p0),
        "logreg": logreg.predict_proba(X_test)[:, 1],
        "rf": rf.predict_proba(X_test)[:, 1],
        "xgb": xgb.predict_proba(X_test)[:, 1],
        "xgb_calibrated": xgb_cal.predict_proba(X_test)[:, 1],
    }

    # Metrics
    rows = []
    for name, proba in preds.items():
        rows.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "log_loss": log_loss(y_test, proba),
            "brier": brier_score_loss(y_test, proba),
        })
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(config.OUTPUTS_DIR / "metrics.csv", index=False)
    print(f"Saved {config.OUTPUTS_DIR / 'metrics.csv'}")
    print(metrics_df.to_string(index=False))

    # --- ROC curve (best model + baseline) ---
    best_name = metrics_df.loc[metrics_df["roc_auc"].idxmax(), "model"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["baseline", best_name]:
        fpr, tpr, _ = roc_curve(y_test, preds[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, preds[name]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.set_aspect("equal")
    save_figure(fig, "roc_curve.png")
    plt.close(fig)

    # --- PR curve ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["baseline", best_name]:
        prec, rec, _ = precision_recall_curve(y_test, preds[name])
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, preds[name]):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    save_figure(fig, "pr_curve.png")
    plt.close(fig)

    # --- Calibration curve: uncalibrated vs calibrated XGB ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, label in [("xgb", "XGB (uncalibrated)"), ("xgb_calibrated", "XGB (calibrated)")]:
        proba = preds[name]
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    save_figure(fig, "calibration_curve.png")
    plt.close(fig)

    print("Saved roc_curve.png, pr_curve.png, calibration_curve.png")
    return metrics_df


if __name__ == "__main__":
    run()
