#!/usr/bin/env python3
"""
MVP Stage 1: Load and Prepare E. coli Protein Localization Data

This script loads the ecoli_data.txt dataset, validates it, performs train/test split
with stratification (with graceful fallback for rare classes), and saves the results.

Requirements:
    pandas
    scikit-learn

Usage:
    python load_and_prepare_data.py
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
INPUT_FILE = "ecoli.data"
NAMES_FILE = "ecoli.names"
TRAIN_OUTPUT = "train.csv"
TEST_OUTPUT = "test.csv"
COLUMN_NAMES = ["name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "target"]
EXPECTED_SHAPE = (336, 9)
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the E. coli dataset from a whitespace-delimited file.

    Args:
        filepath: Path to the input data file.

    Returns:
        DataFrame with loaded data.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If there are parsing errors during loading.
    """
    logger.info(f"Loading data from {filepath}")

    try:
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            names=COLUMN_NAMES,
            dtype={"name": str},
        )
        logger.info(f"Successfully loaded {len(df)} rows")
        return df

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        raise ValueError(f"Failed to parse data file: {e}")


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the dataset structure and types.

    Checks:
        - Shape matches expected (336, 9)
        - No NaN values
        - Feature columns are float64
        - Target column is string

    Args:
        df: Input DataFrame to validate.

    Returns:
        Validated DataFrame with correct types.

    Raises:
        ValueError: If validation checks fail.
    """
    logger.info("Validating data")

    # Check shape
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected shape {EXPECTED_SHAPE}, got {df.shape}"
        )
    logger.info(f"Shape validation passed: {df.shape}")

    # Check for NaN values
    if df.isnull().any().any():
        nan_count = df.isnull().sum().sum()
        raise ValueError(f"Found {nan_count} NaN values in the dataset")
    logger.info("No NaN values found")

    # Convert feature columns to float64
    feature_cols = [col for col in COLUMN_NAMES if col not in ("target", "name")]
    for col in feature_cols:
        try:
            df[col] = df[col].astype("float64")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert column '{col}' to float64: {e}")
    logger.info("Feature columns converted to float64")

    # Convert target to string
    df["target"] = df["target"].astype(str)
    logger.info("Target column converted to string")

    logger.info("Data validation completed successfully")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.

    Attempts stratified split first. If stratification fails due to classes
    with fewer than 2 samples (e.g., imL, imS), falls back to regular split
    with a warning.

    Args:
        df: Input DataFrame.
        test_size: Proportion of data for test set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    logger.info("Splitting data into train/test sets")

    X = df.drop(columns=["target"])
    y = df["target"]

    # Check class distribution for stratification feasibility
    class_counts = y.value_counts()
    problematic_classes = class_counts[class_counts < 2]

    if len(problematic_classes) > 0:
        logger.warning(
            f"Classes with < 2 samples detected: {problematic_classes.to_dict()}. "
            "Stratified split is not possible. Falling back to regular split."
        )
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        logger.info("Regular (non-stratified) split performed")
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        logger.info("Stratified split performed successfully")

    return train_df, test_df


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Save train and test DataFrames to CSV files.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
    """
    logger.info(f"Saving training data to {TRAIN_OUTPUT}")
    train_df.to_csv(TRAIN_OUTPUT, index=False)

    logger.info(f"Saving test data to {TEST_OUTPUT}")
    test_df.to_csv(TEST_OUTPUT, index=False)

    logger.info("Data saved successfully")


def log_class_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Log the class distribution in train and test sets.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
    """
    logger.info("=" * 50)
    logger.info("Class Distribution in Train Set:")
    train_dist = train_df["target"].value_counts().sort_index()
    for cls, count in train_dist.items():
        logger.info(f"  {cls}: {count}")

    logger.info("Class Distribution in Test Set:")
    test_dist = test_df["target"].value_counts().sort_index()
    for cls, count in test_dist.items():
        logger.info(f"  {cls}: {count}")
    logger.info("=" * 50)


def main() -> None:
    """
    Main pipeline function.

    Executes the full data preparation workflow:
    1. Load data
    2. Validate data
    3. Split into train/test
    4. Save results
    5. Log class distributions
    """
    logger.info("Starting E. coli data preparation pipeline")

    # Load data
    df = load_data(INPUT_FILE)

    # Validate data
    df = validate_data(df)

    # Split data
    train_df, test_df = split_data(df)

    # Save results
    save_data(train_df, test_df)

    # Log class distributions
    log_class_distribution(train_df, test_df)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
