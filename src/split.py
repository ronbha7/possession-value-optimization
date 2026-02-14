"""
Time-based split by GAME_ID: sort ascending, train = first 80%, test = last 20%.
Returns train and test DataFrames (or indices).
"""
from typing import List, Optional, Tuple, Union

import pandas as pd

from . import config


def get_train_test_game_ids(df: pd.DataFrame) -> Tuple[List, List]:
    """
    Get train and test game IDs using time-based split.
    - Sort unique GAME_ID ascending.
    - Train = first 80% of unique games, test = last 20%.
    Returns (train_game_ids, test_game_ids).
    """
    if "GAME_ID" not in df.columns:
        raise ValueError("DataFrame must have column GAME_ID")
    unique_games = df["GAME_ID"].dropna().unique()
    unique_games = sorted(unique_games)
    n = len(unique_games)
    n_train = int(n * config.TRAIN_FRAC)
    if n_train == 0:
        n_train = 1
    if n_train >= n:
        n_train = max(1, n - 1)
    train_game_ids = unique_games[:n_train]
    test_game_ids = unique_games[n_train:]
    return train_game_ids, test_game_ids


def split_by_game(
    df: pd.DataFrame,
    return_indices: bool = False,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.Series, pd.Series]]:
    """
    Time-based split by GAME_ID.
    - df: DataFrame with GAME_ID column.
    - return_indices: if True, return (train_idx, test_idx) as indexers; else (train_df, test_df).
    Returns (train, test).
    """
    train_game_ids, test_game_ids = get_train_test_game_ids(df)
    train_mask = df["GAME_ID"].isin(train_game_ids)
    test_mask = df["GAME_ID"].isin(test_game_ids)
    if return_indices:
        return df.index[train_mask], df.index[test_mask]
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()
