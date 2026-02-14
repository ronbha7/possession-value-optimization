"""
Configuration: paths, feature bins, and constants from PROJECT_SPEC.
"""
from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data paths ---
DATA_DIR = PROJECT_ROOT / "data"
# Raw data: prefer shot_logs.csv, fallback to raw.csv
RAW_DATA_PATHS = [
    DATA_DIR / "shot_logs.csv",
    DATA_DIR / "raw.csv",
]
PROCESSED_PATH = DATA_DIR / "processed.parquet"

# --- Model and output dirs ---
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# --- Feature engineering (from spec §4.2) ---
# shot_dist_bin edges: [0,3), [3,10), [10,16), [16,22), [22,100)
SHOT_DIST_BINS = [0, 3, 10, 16, 22, 100]

# is_late_clock = 1 if SHOT_CLOCK < 4 else 0
LATE_CLOCK_THRESHOLD_SEC = 4.0

# defender_tight = 1 if CLOSE_DEF_DIST < 3 else 0 (feet)
DEFENDER_TIGHT_THRESHOLD_FT = 3.0

# --- Split (from spec §3.2) ---
TRAIN_FRAC = 0.80  # first 80% of games = train, last 20% = test

# --- Modeling (reference; used in train.py) ---
# Baseline: mean(FGM) on TRAIN
# LogReg: C=1.0, penalty="l2", max_iter=2000
# RF/XGB: see train.py
