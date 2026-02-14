"""
Prep pipeline: read raw, split by GAME_ID, compute features (aggregates from TRAIN only),
merge, save processed.parquet.
Run: python -m src.prep
"""
import pandas as pd

from . import config
from .features import fit_transform_train, transform_test_or_inference
from .split import split_by_game
from .utils import ensure_dir, load_raw_data, save_processed


def run() -> pd.DataFrame:
    """
    Load raw data, time-based split, fit features on train and transform test,
    concatenate with SPLIT column, save to data/processed.parquet.
    Returns the processed DataFrame.
    """
    # 1) Read raw
    raw = load_raw_data()
    if raw.empty:
        raise ValueError("Raw data is empty")
    if "GAME_ID" not in raw.columns:
        raise ValueError("Raw data must have GAME_ID")
    if "player_id" not in raw.columns:
        raise ValueError("Raw data must have player_id for aggregates")

    # 2) Split by GAME_ID (train 80% / test 20%)
    train_df, test_df = split_by_game(raw)

    # 3) Fit feature pipeline on train; transform test with same context
    train_processed, fit_context = fit_transform_train(train_df)
    test_processed = transform_test_or_inference(test_df, fit_context)

    # 4) Mark split for downstream (train.py can filter SPLIT == "train" / "test")
    train_processed = train_processed.assign(SPLIT="train")
    test_processed = test_processed.assign(SPLIT="test")

    # 5) Concatenate and save
    processed = pd.concat([train_processed, test_processed], ignore_index=True)
    ensure_dir(config.DATA_DIR)
    save_processed(processed)

    print(
        f"Prep done: {len(raw)} raw rows -> {len(train_processed)} train, {len(test_processed)} test "
        f"-> saved {len(processed)} rows to {config.PROCESSED_PATH}"
    )
    return processed


if __name__ == "__main__":
    run()
