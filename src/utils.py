"""
Utilities: GAME_CLOCK parsing (MM:SS → seconds), plot/save helpers, load/save helpers for the pipeline.
"""
import re
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from . import config


# --- GAME_CLOCK parsing ---

def game_clock_to_seconds(clock: Union[str, float, int]) -> float:
    """
    Parse GAME_CLOCK (e.g. "MM:SS" or "M:SS") to total seconds.
    Formula: 60 * minutes + seconds.
    Invalid or missing values return NaN.
    """
    if pd.isna(clock):
        return float("nan")
    if isinstance(clock, (int, float)):
        if clock == clock:  # not NaN
            return float(clock)
        return float("nan")
    s = str(clock).strip()
    if not s:
        return float("nan")
    # Match M:SS or MM:SS (and optionally .ff for tenths)
    m = re.match(r"^(\d+):(\d{2})(?:\.(\d+))?$", s)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        frac = m.group(3)
        sec_frac = int(frac) / (10 ** len(frac)) if frac else 0.0
        return 60 * minutes + seconds + sec_frac
    # Try plain number
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_game_clock_series(series: pd.Series) -> pd.Series:
    """Convert a series of GAME_CLOCK strings to seconds. Returns float series (NaN where invalid)."""
    return series.map(game_clock_to_seconds)


# --- Plot / save helpers ---

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig, name: str, subdir: Optional[str] = None) -> Path:
    """
    Save a matplotlib figure to outputs/ (or outputs/subdir).
    name: filename without path (e.g. 'roc_curve.png').
    Returns path where saved.
    """
    out = config.OUTPUTS_DIR
    if subdir:
        out = out / subdir
    ensure_dir(out)
    filepath = out / name
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    return filepath


# --- Load / save helpers ---

def load_raw_data() -> pd.DataFrame:
    """
    Load raw shot data. Tries shot_logs.csv then raw.csv under data/.
    Raises FileNotFoundError if neither exists.
    """
    for p in config.RAW_DATA_PATHS:
        if p.exists():
            if p.suffix.lower() == ".csv":
                return pd.read_csv(p)
            return pd.read_parquet(p)
    raise FileNotFoundError(
        f"No raw data found. Tried: {[str(p) for p in config.RAW_DATA_PATHS]}"
    )


def load_processed() -> pd.DataFrame:
    """Load processed dataset (parquet)."""
    p = config.PROCESSED_PATH
    if not p.exists():
        raise FileNotFoundError(f"Processed data not found: {p}. Run prep first.")
    return pd.read_parquet(p)


def save_processed(df: pd.DataFrame) -> Path:
    """Save processed DataFrame to data/processed.parquet."""
    ensure_dir(config.DATA_DIR)
    config.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED_PATH, index=False)
    return config.PROCESSED_PATH


def save_json(data: Any, path: Union[str, Path]) -> Path:
    """Save JSON-serializable data to path (e.g. feature_list.json, defaults)."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON from path."""
    import json
    with open(path, "r") as f:
        return json.load(f)


def save_pkl(obj: Any, path: Union[str, Path]) -> Path:
    """Save object via joblib to path."""
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_pkl(path: Union[str, Path]) -> Any:
    """Load object from joblib pickle."""
    import joblib
    return joblib.load(path)
