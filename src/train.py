"""
Train baseline, logistic regression, random forest, XGBoost; calibrate XGBoost.
Save models and feature list. Run: python -m src.train
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from . import config
from .features import get_feature_columns
from .utils import ensure_dir, load_processed, save_json, save_pkl


def run() -> None:
    """Load processed data, train all models, calibrate XGB, save artifacts."""
    ensure_dir(config.MODELS_DIR)

    # Load and split by SPLIT column
    df = load_processed()
    train_df = df.loc[df["SPLIT"] == "train"].copy()
    test_df = df.loc[df["SPLIT"] == "test"].copy()

    feature_cols = get_feature_columns()
    for c in feature_cols:
        if c not in train_df.columns:
            raise ValueError(f"Feature column missing in processed data: {c}")

    X_train = train_df[feature_cols].astype(np.float64)
    y_train = train_df["FGM"].values
    X_test = test_df[feature_cols].astype(np.float64)
    y_test = test_df["FGM"].values

    # --- 5.1 Baseline: p0 = mean(FGM) on TRAIN ---
    p0 = float(y_train.mean())
    save_json({"p0": p0}, config.MODELS_DIR / "baseline.json")
    print(f"Baseline: p0 = {p0:.4f}")

    # --- 5.2 Logistic Regression: scale -> LogisticRegression ---
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
    ])
    logreg.fit(X_train, y_train)
    save_pkl(logreg, config.MODELS_DIR / "logreg.pkl")
    print("LogReg fitted and saved.")

    # --- 5.3 Random Forest: RandomizedSearch ---
    rf = RandomForestClassifier(random_state=42)
    rf_search = RandomizedSearchCV(
        rf,
        param_distributions={
            "n_estimators": [200, 400],
            "max_depth": [6, 10, None],
            "min_samples_leaf": [50, 200],
            "max_features": ["sqrt", 0.5],
        },
        n_iter=8,
        cv=3,
        scoring="neg_log_loss",
        random_state=42,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    save_pkl(rf_search.best_estimator_, config.MODELS_DIR / "rf.pkl")
    print(f"RF best params: {rf_search.best_params_}")

    # --- 5.4 XGBoost: train/valid inside TRAIN, early stopping ---
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    xgb = XGBClassifier(
        objective="binary:logistic",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    save_pkl(xgb, config.MODELS_DIR / "xgb.pkl")
    print("XGB fitted and saved.")

    # --- 5.5 Calibration on TRAIN (isotonic, cv=3) ---
    xgb_for_cal = XGBClassifier(
        objective="binary:logistic",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_cal = CalibratedClassifierCV(xgb_for_cal, method="isotonic", cv=3)
    xgb_cal.fit(X_train, y_train)
    save_pkl(xgb_cal, config.MODELS_DIR / "xgb_calibrated.pkl")
    print("XGB calibrated and saved.")

    # --- Feature list and global defaults (for app) ---
    save_json(feature_cols, config.MODELS_DIR / "feature_list.json")
    global_defaults = {c: float(X_train[c].mean()) for c in feature_cols}
    # For context-matched 3pt comparison: median 3pt shot distance (same context, step back to arc)
    three_pt = train_df.loc[train_df["PTS_TYPE"] == 3, "SHOT_DIST"]
    typical_3pt_shot_dist = float(three_pt.median()) if len(three_pt) > 0 else 23.5
    global_defaults["typical_3pt_shot_dist"] = typical_3pt_shot_dist
    save_json(global_defaults, config.MODELS_DIR / "global_defaults.json")
    print(f"Saved feature_list.json ({len(feature_cols)} features), global_defaults.json, typical_3pt_shot_dist={typical_3pt_shot_dist:.1f}.")


if __name__ == "__main__":
    run()
