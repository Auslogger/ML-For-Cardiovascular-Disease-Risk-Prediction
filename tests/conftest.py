"""Shared fixtures for the test suite.

The real preprocessing and feature-engineering steps take well under a second on
this dataset, so the tests run them on the actual CSVs rather than mocking them.
That way the regression tests check the data the models are trained on.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import pytest

from src.config import RAW_DATA_DIR, TRAIN_FILE


@pytest.fixture(autouse=True)
def figures_go_to_tmp(tmp_path, monkeypatch):
    """Stop tests from writing plots into reports/figures."""
    monkeypatch.setattr("src.utils.FIGURES_DIR", tmp_path / "figures")


@pytest.fixture(scope="session")
def raw_train() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / TRAIN_FILE)


@pytest.fixture(scope="session")
def preprocessed():
    """(preprocessor, train_df, test_df) after imputation, capping and binning."""
    from src.data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    preprocessor.load_datasets()
    preprocessor.analyze_data_quality()
    train_df, test_df = preprocessor.handle_missing_values(strategy="median")
    return preprocessor, train_df, test_df


@pytest.fixture(scope="session")
def engineered(preprocessed):
    """(train_engineered, test_engineered, engineer) on the real data."""
    from src.feature_engineering import create_feature_engineering_pipeline

    preprocessor, train_df, test_df = preprocessed
    train_engineered, engineer = create_feature_engineering_pipeline(
        train_df,
        preprocessor.categorical_features,
        preprocessor.numerical_features,
    )
    test_engineered = engineer.transform(test_df)
    return train_engineered, test_engineered, engineer
