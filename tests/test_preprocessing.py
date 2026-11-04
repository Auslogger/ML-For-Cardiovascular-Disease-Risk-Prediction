import numpy as np
import pandas as pd
import pytest

from src.config import PHASE7_EXPECTED_TEST_SIZE, TARGET_COLUMN
from src.data_preprocessing import SMOKING_INTENSITY_ORDER, make_train_val_test_split

BINARY_FLAGS = ["diabetes", "BPMeds", "prevalentHyp", "prevalentStroke"]


@pytest.mark.parametrize("col", BINARY_FLAGS)
def test_binary_flags_survive_preprocessing(raw_train, preprocessed, col):
    """Regression test: percentile capping used to clip rare flags to all zeros.

    Only ~3% of patients are diabetic, so the 95th percentile of `diabetes` is 0 and
    clipping to it erased every positive case before any model saw the data.
    """
    _, train_df, _ = preprocessed
    processed = train_df[col].astype(float)

    assert set(processed.unique()) == {0.0, 1.0}

    # Imputation may fill gaps, but every recorded value must come through unchanged
    observed = raw_train[col].notna()
    np.testing.assert_array_equal(
        processed[observed].to_numpy(),
        raw_train.loc[observed, col].astype(float).to_numpy(),
    )


def test_binary_flags_are_never_capped(preprocessed):
    preprocessor, _, _ = preprocessed
    assert not set(preprocessor.outlier_limits) & set(BINARY_FLAGS)


def test_capping_limits_come_from_training_data_and_apply_to_both_sets(preprocessed):
    preprocessor, train_df, test_df = preprocessed
    assert preprocessor.outlier_limits, "expected continuous features to be capped"

    for col, (low, high) in preprocessor.outlier_limits.items():
        assert train_df[col].between(low, high).all(), col
        assert test_df[col].between(low, high).all(), col


def test_capping_keeps_clinically_meaningful_ranges(preprocessed):
    """5th/95th capping squashed heart rate into 60-98 bpm and glucose to <= 110 mg/dL,
    which made the brady/tachy and diabetic-glucose categories impossible."""
    _, train_df, _ = preprocessed
    assert train_df["heartRate"].min() < 60
    assert train_df["heartRate"].max() > 100
    assert train_df["glucose"].max() >= 126


def test_features_stay_in_clinical_units(raw_train, preprocessed):
    """No log transform: feature engineering relies on thresholds in mmHg / mg/dL."""
    _, train_df, _ = preprocessed
    for col in ["sysBP", "glucose", "BMI"]:
        assert train_df[col].median() == pytest.approx(raw_train[col].median())


def test_no_missing_values_remain(preprocessed):
    _, train_df, test_df = preprocessed
    assert train_df.isna().sum().sum() == 0
    assert test_df.isna().sum().sum() == 0


def test_identifier_dropped_and_target_only_in_train(preprocessed):
    _, train_df, test_df = preprocessed
    assert "id" not in train_df.columns
    assert "id" not in test_df.columns
    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN not in test_df.columns


def test_cigs_per_day_is_binned_into_smoking_intensity(preprocessed):
    _, train_df, _ = preprocessed
    assert "cigsPerDay" not in train_df.columns
    assert set(train_df["smoking_intensity"].astype(str).unique()) <= set(SMOKING_INTENSITY_ORDER)


def test_feature_types_are_identified_sensibly(preprocessed):
    preprocessor, _, _ = preprocessed
    for col in ["age", "sysBP", "glucose", "diabetes", "BPMeds", "prevalentHyp"]:
        assert col in preprocessor.numerical_features
    for col in ["sex", "is_smoking", "smoking_intensity"]:
        assert col in preprocessor.categorical_features


# ---------------------------------------------------------------- train/val/test split

def test_split_is_60_20_20_stratified_and_disjoint(engineered):
    train_engineered, _, _ = engineered
    X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(train_engineered)

    n = len(train_engineered)
    assert len(X_train) + len(X_val) + len(X_test) == n
    assert len(X_val) == pytest.approx(0.2 * n, abs=1)
    assert len(X_test) == pytest.approx(0.2 * n, abs=1)

    train_idx, val_idx, test_idx = set(X_train.index), set(X_val.index), set(X_test.index)
    assert not train_idx & val_idx
    assert not train_idx & test_idx
    assert not val_idx & test_idx

    prevalence = train_engineered[TARGET_COLUMN].mean()
    for y in (y_train, y_val, y_test):
        assert y.mean() == pytest.approx(prevalence, abs=0.01)

    for X in (X_train, X_val, X_test):
        assert TARGET_COLUMN not in X.columns


def test_split_is_deterministic(engineered):
    train_engineered, _, _ = engineered
    first = make_train_val_test_split(train_engineered)
    second = make_train_val_test_split(train_engineered)
    for a, b in zip(first, second):
        pd.testing.assert_index_equal(a.index, b.index)


def test_test_split_size_matches_phase8_guardrail(engineered):
    train_engineered, _, _ = engineered
    *_, y_test = make_train_val_test_split(train_engineered)
    assert len(y_test) == PHASE7_EXPECTED_TEST_SIZE
