import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification

from src.class_imbalance import ImbalanceHandler


@pytest.fixture
def imbalanced_data():
    X, y = make_classification(
        n_samples=400, n_features=6, n_informative=4, weights=[0.85, 0.15], random_state=0
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), pd.Series(y, name="target")


def test_class_weights_are_balanced():
    y = pd.Series([0] * 80 + [1] * 20)
    weights = ImbalanceHandler().calculate_class_weights(y)
    # n_samples / (n_classes * count)
    assert weights[0] == pytest.approx(100 / (2 * 80))
    assert weights[1] == pytest.approx(100 / (2 * 20))


@pytest.mark.parametrize(
    "strategy, expected_type",
    [("smote", SMOTE), ("random_under", RandomUnderSampler)],
)
def test_get_sampler_returns_the_right_sampler(strategy, expected_type):
    assert isinstance(ImbalanceHandler().get_sampler(strategy), expected_type)


def test_get_sampler_none_means_no_resampling():
    assert ImbalanceHandler().get_sampler("none") is None
    assert ImbalanceHandler().get_sampler(None) is None


def test_smote_nc_needs_categorical_indices():
    handler = ImbalanceHandler()
    with pytest.raises(ValueError):
        handler.get_sampler("smote_nc")
    assert isinstance(handler.get_sampler("smote_nc", categorical_indices=[0]), SMOTENC)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        ImbalanceHandler().get_sampler("magic")


def test_selection_uses_cv_score_and_lets_no_resampling_win():
    """Phase 4 used to pick whichever method got closest to 1:1, which every SMOTE
    variant does by construction. Selection should follow model performance."""
    comparison = pd.DataFrame({
        "method": ["none", "smote", "random_under"],
        "total_samples": [1000, 1700, 300],
        "imbalance_ratio": [5.6, 1.0, 1.0],
        "cv_roc_auc_mean": [0.712, 0.705, 0.701],
    })
    assert ImbalanceHandler().select_best_sampling_method(comparison) == "none"

    comparison.loc[comparison["method"] == "smote", "cv_roc_auc_mean"] = 0.75
    assert ImbalanceHandler().select_best_sampling_method(comparison) == "smote"


def test_compare_sampling_methods_scores_every_method(imbalanced_data):
    X, y = imbalanced_data
    result = ImbalanceHandler().compare_sampling_methods(
        X, y, methods=["none", "smote", "random_under"], cv_folds=3
    )

    assert result["method"].tolist() == ["none", "smote", "random_under"]
    assert result["cv_roc_auc_mean"].between(0.5, 1.0).all()
    assert result["cv_ap_mean"].between(0.0, 1.0).all()

    by_method = result.set_index("method")
    # 'none' trains on the data as-is, the samplers balance it
    assert by_method.loc["none", "class_1_count"] == int(y.sum())
    assert by_method.loc["smote", "class_0_count"] == by_method.loc["smote", "class_1_count"]
    assert by_method.loc["random_under", "total_samples"] == 2 * int(y.sum())


def test_probe_corrects_imbalance_exactly_once():
    handler = ImbalanceHandler()

    weighted = handler._build_probe_pipeline("none")
    assert "sampler" not in weighted.named_steps
    assert weighted.named_steps["clf"].class_weight == "balanced"

    resampled = handler._build_probe_pipeline("smote")
    assert "sampler" in resampled.named_steps
    assert resampled.named_steps["clf"].class_weight is None


def test_resampling_does_not_change_number_of_predictions(imbalanced_data):
    """Samplers inside an imblearn Pipeline only run during fit, never at predict time,
    so validation folds are scored on real (unresampled) rows."""
    X, y = imbalanced_data
    pipe = ImbalanceHandler()._build_probe_pipeline("smote").fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
