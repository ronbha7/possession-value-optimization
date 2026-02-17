"""
Counterfactual uplift and threshold-based policy evaluation.
No retraining; operates on processed test 2pt rows and calibrated model.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from . import config
from .features import get_feature_columns

# Representative 3pt distance for counterfactual (feet)
COUNTERFACTUAL_3PT_DIST = 23.5


def _shot_dist_bin(shot_dist: float) -> int:
    """Map SHOT_DIST to ordinal bin [0,3),[3,10),[10,16),[16,22),[22,100) -> 0..4."""
    bins = config.SHOT_DIST_BINS
    for i in range(len(bins) - 1):
        if bins[i] <= shot_dist < bins[i + 1]:
            return i
    return len(bins) - 2


def compute_counterfactual_ev3(df: pd.DataFrame, model) -> np.ndarray:
    """
    For each 2pt row: same context but PTS_TYPE=3, SHOT_DIST=23.5 (and shot_dist_bin updated).
    Returns EV3_counterfactual = P(make_counterfactual) * 3 per row.
    df must contain all feature columns (e.g. 2pt subset of processed test).
    """
    feature_cols = get_feature_columns()
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Counterfactual requires feature column: {c}")

    cf = df[feature_cols].copy()
    cf["PTS_TYPE"] = 3
    cf["SHOT_DIST"] = COUNTERFACTUAL_3PT_DIST
    cf["shot_dist_bin"] = _shot_dist_bin(COUNTERFACTUAL_3PT_DIST)

    X = cf[feature_cols].astype(np.float64)
    p = model.predict_proba(X)[:, 1]
    return (p * 3).astype(np.float64)


def compute_uplift(df: pd.DataFrame, ev2: np.ndarray, ev3_cf: np.ndarray) -> np.ndarray:
    """uplift = ev3_cf - ev2. df unused; kept for API consistency."""
    return (ev3_cf - ev2).astype(np.float64)


def evaluate_threshold_policies(
    uplift: np.ndarray,
    ev2: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """
    For each threshold δ: mask = (uplift > δ), percent_replaced, delta_total, delta_per_100.
    Returns DataFrame with columns: threshold, percent_replaced, delta_total, delta_per_100.
    """
    n = len(ev2)
    total_original_ev = float(np.sum(ev2))
    rows = []
    for delta in thresholds:
        mask = uplift > delta
        percent_replaced = float(np.mean(mask))
        total_new_ev = float(np.sum(np.where(mask, ev2 + uplift, ev2)))
        delta_total = total_new_ev - total_original_ev
        delta_per_100 = (delta_total / n) * 100 if n else 0.0
        rows.append({
            "threshold": float(delta),
            "percent_replaced": percent_replaced,
            "delta_total": delta_total,
            "delta_per_100": delta_per_100,
        })
    return pd.DataFrame(rows)


def bootstrap_policy_gain(
    ev2: np.ndarray,
    uplift: np.ndarray,
    threshold: float,
    n_boot: int = 500,
) -> Tuple[float, float, float]:
    """
    Bootstrap policy gain: sample indices with replacement, compute delta_per_100 per run.
    Returns (mean_gain, lower_95, upper_95) using percentile method.
    """
    n = len(ev2)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    gains = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ev2_b = ev2[idx]
        uplift_b = uplift[idx]
        mask = uplift_b > threshold
        total_original = np.sum(ev2_b)
        total_new = np.sum(np.where(mask, ev2_b + uplift_b, ev2_b))
        delta_per_100 = (total_new - total_original) / n * 100
        gains.append(delta_per_100)
    gains = np.array(gains)
    mean_gain = float(np.mean(gains))
    lower_95 = float(np.percentile(gains, 2.5))
    upper_95 = float(np.percentile(gains, 97.5))
    return mean_gain, lower_95, upper_95
