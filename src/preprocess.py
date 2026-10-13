"""
Data ingestion and preprocessing for the MLOps pipeline.

Loads raw data, cleans it, splits it into train/test sets, and writes
processed artifacts to disk so that the training stage is fully
reproducible and decoupled from data acquisition.
"""
import os
import logging

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))


def load_raw_data() -> pd.DataFrame:
    """
    Load raw data.

    Swap this out for a real ingestion source (S3, a warehouse table,
    an internal API, etc.) — everything downstream only depends on
    the DataFrame contract returned here (features + a 'target' column).
    """
    dataset = load_wine(as_frame=True)
    df = dataset.frame
    df.rename(columns={"target": "target"}, inplace=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates and rows with nulls."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    logger.info("Cleaned data: %d -> %d rows", before, len(df))
    return df


def split_and_scale(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def run_preprocessing():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    logger.info("Loading raw data...")
    df = load_raw_data()

    logger.info("Cleaning data...")
    df = clean_data(df)

    logger.info("Splitting and scaling...")
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
        },
        os.path.join(PROCESSED_DATA_DIR, "dataset.joblib"),
    )
    joblib.dump(scaler, os.path.join(PROCESSED_DATA_DIR, "scaler.joblib"))

    logger.info("Preprocessing complete. Artifacts written to %s", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    run_preprocessing()
