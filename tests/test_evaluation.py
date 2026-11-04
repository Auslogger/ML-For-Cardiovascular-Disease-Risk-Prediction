import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation import (
    _net_benefit,
    bootstrap_cis,
    calibration_ece_mce,
    compute_threshold_metrics,
    delong_roc_test,
)


@pytest.fixture
def scored_cohort():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.15, size=600)
    informative = np.clip(0.15 + 0.5 * y + rng.normal(0, 0.15, size=600), 0.001, 0.999)
    random_scores = rng.uniform(size=600)
    return y, informative, random_scores


def test_threshold_metrics_on_a_hand_checked_example():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.9])  # at 0.5 -> one of each TP/FP/TN/FN
    m = compute_threshold_metrics(y_true, y_proba, threshold=0.5)

    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["recall_sensitivity"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["precision_ppv"] == pytest.approx(0.5)
    assert m["npv"] == pytest.approx(0.5)
    assert m["youden_j"] == pytest.approx(0.0)


def test_delong_matches_sklearn_auc_and_detects_a_real_difference(scored_cohort):
    y, informative, random_scores = scored_cohort
    result = delong_roc_test(y, informative, random_scores)

    assert result["auc_a"] == pytest.approx(roc_auc_score(y, informative))
    assert result["auc_b"] == pytest.approx(roc_auc_score(y, random_scores))
    assert result["delta"] > 0
    assert result["p_value"] < 0.001


def test_delong_identical_scores_are_not_different(scored_cohort):
    y, informative, _ = scored_cohort
    result = delong_roc_test(y, informative, informative.copy())
    assert result["delta"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    # 100 patients at 20% predicted risk with 20 events, 100 at 80% with 80 events
    y = np.array([1] * 20 + [0] * 80 + [1] * 80 + [0] * 20)
    p = np.array([0.2] * 100 + [0.8] * 100)
    assert calibration_ece_mce(y, p, n_bins=10)["ece"] == pytest.approx(0.0, abs=1e-9)


def test_ece_flags_overconfident_predictions():
    y = np.array([1] * 10 + [0] * 90)
    p = np.full(100, 0.9)  # says 90% risk, observed 10%
    assert calibration_ece_mce(y, p, n_bins=10)["ece"] == pytest.approx(0.8, abs=1e-6)


def test_net_benefit_of_flagging_everyone_matches_treat_all_formula():
    y = np.array([1] * 15 + [0] * 85)
    prevalence, t = 0.15, 0.10
    treat_all = prevalence - (1 - prevalence) * t / (1 - t)
    assert _net_benefit(y, np.ones_like(y, dtype=float), t) == pytest.approx(treat_all)


def test_net_benefit_of_flagging_no_one_is_zero():
    y = np.array([1] * 15 + [0] * 85)
    assert _net_benefit(y, np.zeros_like(y, dtype=float), 0.1) == pytest.approx(0.0)


def test_bootstrap_cis_bracket_the_estimate_and_are_reproducible(scored_cohort):
    y, informative, _ = scored_cohort
    kwargs = dict(y_true=y, y_proba=informative, threshold=0.5,
                  n_bootstraps=50, ci_alpha=0.05, seed=42)
    first = bootstrap_cis(**kwargs)
    second = bootstrap_cis(**kwargs)

    auc = first["roc_auc"]
    assert auc["lower"] <= roc_auc_score(y, informative) <= auc["upper"]
    assert first == second
