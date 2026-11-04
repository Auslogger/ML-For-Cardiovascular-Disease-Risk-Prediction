import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification

from src.config import PHASE6_MODELS
from src.model_training import ModelTrainer

CLASS_WEIGHTS = {0: 0.59, 1: 3.32}


@pytest.fixture
def X_small():
    X, _ = make_classification(n_samples=200, n_features=5, random_state=0)
    df = pd.DataFrame(X, columns=["age", "sysBP", "BMI", "glucose", "totChol"])
    df["sex_encoded"] = (df["age"] > 0).astype(int)  # one categorical column
    return df


@pytest.fixture
def y_small():
    _, y = make_classification(n_samples=200, n_features=5, weights=[0.8, 0.2], random_state=0)
    return pd.Series(y)


@pytest.fixture
def trainer():
    return ModelTrainer(random_state=0)


def test_resampling_and_class_weights_are_not_combined(trainer, X_small):
    """Applying SMOTE *and* class weights double-corrects the imbalance."""
    pipe, _ = trainer._build_phase6_pipeline(
        "random_forest", X_small, class_weight=CLASS_WEIGHTS, imbalance_strategy="smote"
    )
    assert "sampler" in pipe.named_steps
    assert pipe.named_steps["model"].class_weight is None


def test_class_weights_used_when_not_resampling(trainer, X_small):
    pipe, _ = trainer._build_phase6_pipeline(
        "random_forest", X_small, class_weight=CLASS_WEIGHTS, imbalance_strategy="none"
    )
    assert "sampler" not in pipe.named_steps
    assert pipe.named_steps["model"].class_weight == CLASS_WEIGHTS


def test_xgboost_gets_scale_pos_weight_when_not_resampling(trainer, X_small):
    pipe, _ = trainer._build_phase6_pipeline(
        "xgboost", X_small, class_weight=CLASS_WEIGHTS, imbalance_strategy="none"
    )
    assert pipe.named_steps["model"].get_params()["scale_pos_weight"] == pytest.approx(
        CLASS_WEIGHTS[1] / CLASS_WEIGHTS[0]
    )


def test_categorical_columns_are_detected_and_ordered_last(trainer, X_small):
    _, ctx = trainer._build_phase6_pipeline(
        "random_forest", X_small, class_weight=None, imbalance_strategy="smote_nc"
    )
    assert ctx["cat_cols"] == ["sex_encoded"]
    assert ctx["cat_indices_after"] == [len(ctx["num_cols"])]


def test_only_scale_sensitive_models_are_standardised(trainer, X_small):
    svm_pipe, _ = trainer._build_phase6_pipeline("svm", X_small, None, "none")
    rf_pipe, _ = trainer._build_phase6_pipeline("random_forest", X_small, None, "none")

    svm_num = dict((n, t) for n, t, _ in svm_pipe.named_steps["preprocess"].transformers)["num"]
    rf_num = dict((n, t) for n, t, _ in rf_pipe.named_steps["preprocess"].transformers)["num"]
    assert svm_num.__class__.__name__ == "StandardScaler"
    assert rf_num == "passthrough"


@pytest.mark.parametrize("strategy", ["none", "smote", "smote_nc"])
def test_phase6_pipeline_fits_and_predicts_every_row(trainer, X_small, y_small, strategy):
    pipe, _ = trainer._build_phase6_pipeline("random_forest", X_small, CLASS_WEIGHTS, strategy)
    pipe.set_params(model__n_estimators=20)
    pipe.fit(X_small, y_small)
    proba = pipe.predict_proba(X_small)[:, 1]
    assert proba.shape == (len(X_small),)
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.parametrize("model_name", PHASE6_MODELS)
@pytest.mark.parametrize("strategy", ["none", "smote"])
def test_every_phase6_pipeline_can_be_cloned(trainer, X_small, model_name, strategy):
    """Hyperparameter search clones the pipeline for every fit; CatBoost used to
    fail here because it rewrites a class_weights dict passed to its constructor."""
    pipe, _ = trainer._build_phase6_pipeline(model_name, X_small, CLASS_WEIGHTS, strategy)
    clone(pipe)


def test_catboost_does_not_write_training_files(trainer):
    assert trainer.get_model("catboost").get_params()["allow_writing_files"] is False


def test_unknown_model_raises(trainer):
    with pytest.raises(ValueError):
        trainer.get_model("not_a_model")
