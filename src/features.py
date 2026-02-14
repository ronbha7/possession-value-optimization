"""
Feature engineering: time_remaining_sec, is_late_clock, defender_tight, shot_dist_bin,
missing indicators, imputation with TRAIN medians, player aggregates from TRAIN only,
cold-start with global values, log1p(volume), PTS_TYPE as feature.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config
from .utils import parse_game_clock_series


# Columns that must not be missing (drop rows if missing)
REQUIRED_COLS = ["SHOT_DIST", "CLOSE_DEF_DIST"]
# Columns we impute with TRAIN median and add missing indicators
IMPUTE_COLS = ["SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME"]

# Player aggregate column names (suffix after player_)
PLAYER_AGG_NAMES = [
    "fgm_rate",
    "3pt_rate",
    "avg_shot_dist",
    "avg_close_def_dist",
    "avg_shot_clock",
    "shot_volume",
    "avg_dribbles",
    "avg_touch_time",
    "pct_3pt",
]


def _ensure_required(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing SHOT_DIST or CLOSE_DEF_DIST. Return copy."""
    out = df.copy()
    for c in REQUIRED_COLS:
        if c not in out.columns:
            raise ValueError(f"Required column missing: {c}")
        out = out.loc[out[c].notna()]
    return out


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_*_missing for SHOT_CLOCK, DRIBBLES, TOUCH_TIME (1 if missing, 0 else)."""
    out = df.copy()
    for c in IMPUTE_COLS:
        if c in out.columns:
            out[f"is_{c.lower()}_missing"] = out[c].isna().astype(np.int64)
        else:
            out[f"is_{c.lower()}_missing"] = 1
    return out


def add_base_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time_remaining_sec, is_late_clock, defender_tight, shot_dist_bin.
    Assumes REQUIRED_COLS present; CLOSE_DEF_DIST used for defender_tight.
    """
    out = df.copy()

    # time_remaining_sec from GAME_CLOCK
    if "GAME_CLOCK" in out.columns:
        out["time_remaining_sec"] = parse_game_clock_series(out["GAME_CLOCK"])
    else:
        out["time_remaining_sec"] = np.nan

    # is_late_clock = 1 if SHOT_CLOCK < 4 else 0; null -> 0
    if "SHOT_CLOCK" in out.columns:
        out["is_late_clock"] = (out["SHOT_CLOCK"] < config.LATE_CLOCK_THRESHOLD_SEC).fillna(False).astype(np.int64)
    else:
        out["is_late_clock"] = 0

    # defender_tight = 1 if CLOSE_DEF_DIST < 3 else 0
    out["defender_tight"] = (out["CLOSE_DEF_DIST"] < config.DEFENDER_TIGHT_THRESHOLD_FT).astype(np.int64)

    # shot_dist_bin ordinal [0, 3, 10, 16, 22, 100)
    bins = config.SHOT_DIST_BINS
    out["shot_dist_bin"] = pd.cut(
        out["SHOT_DIST"],
        bins=bins,
        labels=False,
        include_lowest=True,
        right=False,
    ).astype("Int64")  # nullable int; fillna with mode or 0 later if needed
    # If any NaN from out-of-range, fill with 0
    if out["shot_dist_bin"].isna().any():
        out["shot_dist_bin"] = out["shot_dist_bin"].fillna(0)
    out["shot_dist_bin"] = out["shot_dist_bin"].astype(np.int64)

    return out


def get_train_medians(train_df: pd.DataFrame) -> Dict[str, float]:
    """Compute TRAIN medians for SHOT_CLOCK, DRIBBLES, TOUCH_TIME (for imputation)."""
    medians = {}
    for c in IMPUTE_COLS:
        if c in train_df.columns:
            medians[c] = float(train_df[c].median())
        else:
            medians[c] = 0.0
    return medians


def impute_with_medians(
    df: pd.DataFrame,
    train_medians: Dict[str, float],
) -> pd.DataFrame:
    """Fill missing SHOT_CLOCK, DRIBBLES, TOUCH_TIME with train medians. Modifies copy."""
    out = df.copy()
    for c in IMPUTE_COLS:
        if c in out.columns and c in train_medians:
            out[c] = out[c].fillna(train_medians[c])
        elif c not in out.columns:
            out[c] = train_medians.get(c, 0.0)
    return out


def get_player_aggregates(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute player-level aggregates on TRAIN only.
    Returns DataFrame with columns: player_id, player_fgm_rate, player_3pt_rate, ...
    """
    if "player_id" not in train_df.columns:
        raise ValueError("train_df must have player_id")

    g = train_df.groupby("player_id", dropna=False)

    # mean(FGM)
    fgm_rate = g["FGM"].mean().rename("player_fgm_rate")

    # mean(FGM where PTS_TYPE==3): 3pt make rate per player (NaN if no 3pt attempts)
    if "PTS_TYPE" in train_df.columns:
        mask3 = train_df["PTS_TYPE"] == 3
        three_pt = train_df.loc[mask3].groupby("player_id", dropna=False)["FGM"].mean().rename("player_3pt_rate")
        pct_3pt = train_df.groupby("player_id", dropna=False)["PTS_TYPE"].apply(lambda x: (x == 3).mean()).rename("player_pct_3pt")
        aggs = pd.DataFrame({"player_fgm_rate": fgm_rate, "player_pct_3pt": pct_3pt})
        aggs = aggs.join(three_pt)  # NaNs for players with no 3pt attempts
    else:
        pct_3pt = pd.Series(dtype=float)
        aggs = pd.DataFrame({"player_fgm_rate": fgm_rate, "player_pct_3pt": pct_3pt})
        aggs["player_3pt_rate"] = np.nan

    for col, name in [
        ("SHOT_DIST", "player_avg_shot_dist"),
        ("CLOSE_DEF_DIST", "player_avg_close_def_dist"),
        ("SHOT_CLOCK", "player_avg_shot_clock"),
        ("DRIBBLES", "player_avg_dribbles"),
        ("TOUCH_TIME", "player_avg_touch_time"),
    ]:
        if col in train_df.columns:
            aggs[name] = g[col].mean()

    # shot_volume = count
    aggs["player_shot_volume"] = g.size().astype(np.int64)

    # Fill player_3pt_rate for players with no 3pt attempts (NaN) in merge step via global
    aggs = aggs.reset_index()
    return aggs


def get_global_aggregates(train_df: pd.DataFrame) -> Dict[str, float]:
    """
    Global TRAIN aggregates for cold-start: global_fgm_rate, global_3pt_rate,
    global_avg_*, global_pct_3pt, global_median_shot_volume.
    """
    globals_ = {}
    globals_["global_fgm_rate"] = float(train_df["FGM"].mean())
    if "PTS_TYPE" in train_df.columns:
        mask3 = train_df["PTS_TYPE"] == 3
        made3 = train_df.loc[mask3, "FGM"].sum()
        n3 = mask3.sum()
        globals_["global_3pt_rate"] = float(made3 / n3) if n3 > 0 else 0.0
        globals_["global_pct_3pt"] = float(mask3.mean())
    else:
        globals_["global_3pt_rate"] = 0.0
        globals_["global_pct_3pt"] = 0.0

    for col, key in [
        ("SHOT_DIST", "global_avg_shot_dist"),
        ("CLOSE_DEF_DIST", "global_avg_close_def_dist"),
        ("SHOT_CLOCK", "global_avg_shot_clock"),
        ("DRIBBLES", "global_avg_dribbles"),
        ("TOUCH_TIME", "global_avg_touch_time"),
    ]:
        if col in train_df.columns:
            globals_[key] = float(train_df[col].mean())

    # Volume: global median for cold-start and log1p
    if "player_id" in train_df.columns:
        vol = train_df.groupby("player_id", dropna=False).size()
        globals_["global_median_shot_volume"] = float(vol.median()) if len(vol) else 0.0
    else:
        globals_["global_median_shot_volume"] = 0.0

    return globals_


def merge_player_aggregates(
    df: pd.DataFrame,
    player_aggs: pd.DataFrame,
    global_aggs: Dict[str, float],
) -> pd.DataFrame:
    """
    Merge player aggregates into df; cold-start (players not in TRAIN) filled with global values.
    Adds player_* columns and log1p_player_shot_volume.
    """
    out = df.merge(player_aggs, on="player_id", how="left")

    # Cold-start: fill NaN with global
    for col in player_aggs.columns:
        if col == "player_id":
            continue
        if col == "player_3pt_rate":
            fill = global_aggs.get("global_3pt_rate", 0.0)
        elif col == "player_fgm_rate":
            fill = global_aggs.get("global_fgm_rate", 0.0)
        elif col == "player_pct_3pt":
            fill = global_aggs.get("global_pct_3pt", 0.0)
        elif col == "player_shot_volume":
            fill = global_aggs.get("global_median_shot_volume", 0.0)
        elif col == "player_avg_shot_dist":
            fill = global_aggs.get("global_avg_shot_dist", 0.0)
        elif col == "player_avg_close_def_dist":
            fill = global_aggs.get("global_avg_close_def_dist", 0.0)
        elif col == "player_avg_shot_clock":
            fill = global_aggs.get("global_avg_shot_clock", 0.0)
        elif col == "player_avg_dribbles":
            fill = global_aggs.get("global_avg_dribbles", 0.0)
        elif col == "player_avg_touch_time":
            fill = global_aggs.get("global_avg_touch_time", 0.0)
        else:
            fill = 0.0
        out[col] = out[col].fillna(fill)

    # log1p(player_shot_volume)
    out["log1p_player_shot_volume"] = np.log1p(out["player_shot_volume"].clip(lower=0))
    return out


def add_pts_type_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add PTS_TYPE as a model feature (per locked decision)."""
    out = df.copy()
    if "PTS_TYPE" not in out.columns:
        out["PTS_TYPE"] = 2  # default
    return out


def get_feature_columns() -> List[str]:
    """Ordered list of feature column names for modeling (X)."""
    return [
        "SHOT_DIST",
        "CLOSE_DEF_DIST",
        "SHOT_CLOCK",
        "DRIBBLES",
        "TOUCH_TIME",
        "PERIOD",
        "time_remaining_sec",
        "is_late_clock",
        "defender_tight",
        "shot_dist_bin",
        "is_shot_clock_missing",
        "is_dribbles_missing",
        "is_touch_time_missing",
        "player_fgm_rate",
        "player_3pt_rate",
        "player_avg_shot_dist",
        "player_avg_close_def_dist",
        "player_avg_shot_clock",
        "log1p_player_shot_volume",
        "player_avg_dribbles",
        "player_avg_touch_time",
        "player_pct_3pt",
        "PTS_TYPE",
    ]


def fit_transform_train(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full feature pipeline on TRAIN: drop missing core, indicators, base engineered,
    impute with train medians, player aggregates (on this train), merge, log1p, PTS_TYPE.
    Returns (train_df with features, fit_context) where fit_context has keys:
    train_medians, player_aggs, global_aggs (for transforming test or new data).
    """
    df = _ensure_required(train_df)
    df = add_missing_indicators(df)
    df = add_base_engineered(df)

    train_medians = get_train_medians(train_df)  # use original train for medians
    df = impute_with_medians(df, train_medians)

    player_aggs = get_player_aggregates(train_df)
    global_aggs = get_global_aggregates(train_df)
    df = merge_player_aggregates(df, player_aggs, global_aggs)
    df = add_pts_type_feature(df)

    # Ensure PERIOD exists for feature list
    if "PERIOD" not in df.columns:
        df["PERIOD"] = 1
    # time_remaining_sec fillna with 0 if any
    for c in ["time_remaining_sec", "PERIOD"]:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(0)

    fit_context = {
        "train_medians": train_medians,
        "player_aggs": player_aggs,
        "global_aggs": global_aggs,
    }
    return df, fit_context


def transform_test_or_inference(
    df: pd.DataFrame,
    fit_context: Dict[str, Any],
) -> pd.DataFrame:
    """
    Apply same feature pipeline to test or new data using fit_context from fit_transform_train.
    Drops rows missing SHOT_DIST/CLOSE_DEF_DIST, imputes with train medians, merges player
    aggregates with cold-start, adds log1p volume and PTS_TYPE.
    """
    df = _ensure_required(df)
    df = add_missing_indicators(df)
    df = add_base_engineered(df)
    df = impute_with_medians(df, fit_context["train_medians"])
    df = merge_player_aggregates(
        df,
        fit_context["player_aggs"],
        fit_context["global_aggs"],
    )
    df = add_pts_type_feature(df)
    if "PERIOD" not in df.columns:
        df["PERIOD"] = 1
    for c in ["time_remaining_sec", "PERIOD"]:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(0)
    return df
