import pandas as pd
import pytest

from src.config import TARGET_COLUMN
from src.data_preprocessing import SMOKING_INTENSITY_ORDER
from src.feature_engineering import FeatureEngineer

CATEGORICAL = ["education", "sex", "is_smoking", "prevalentStroke", "smoking_intensity"]
NUMERICAL = ["age", "BPMeds", "prevalentHyp", "diabetes", "totChol",
             "sysBP", "diaBP", "BMI", "heartRate", "glucose"]


def patient(**overrides) -> dict:
    """A healthy 50-year-old; override whatever a test cares about."""
    row = dict(
        age=50, education=2.0, sex="M", is_smoking="NO", BPMeds=0.0,
        prevalentStroke=0, prevalentHyp=0, diabetes=0, totChol=200.0,
        sysBP=120.0, diaBP=80.0, BMI=25.0, heartRate=70.0, glucose=80.0,
        smoking_intensity="non_smoker",
    )
    row.update(overrides)
    return row


def engineer_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # mimic the dtypes DataPreprocessor hands over
    for col in ["education", "prevalentStroke", "smoking_intensity"]:
        df[col] = df[col].astype("category")
    return FeatureEngineer(CATEGORICAL, NUMERICAL).fit_transform(df)


# ------------------------------------------------------------ hand-checked formulas

def test_blood_pressure_features():
    out = engineer_rows([patient(sysBP=120.0, diaBP=80.0)])
    assert out.loc[0, "pulse_pressure"] == pytest.approx(40.0)
    assert out.loc[0, "mean_arterial_pressure"] == pytest.approx(80 + 40 / 3)


@pytest.mark.parametrize(
    "prevalent_hyp, sys_bp, expected_stage",
    [
        (0, 200.0, 0),  # no hypertension diagnosis -> stage 0 whatever the reading
        (1, 125.0, 1),  # controlled
        (1, 135.0, 2),  # stage 1
        (1, 150.0, 3),  # stage 2
        (1, 185.0, 4),  # crisis
    ],
)
def test_hypertension_stage(prevalent_hyp, sys_bp, expected_stage):
    out = engineer_rows([patient(prevalentHyp=prevalent_hyp, sysBP=sys_bp)])
    assert out.loc[0, "hypertension_stage"] == expected_stage


@pytest.mark.parametrize("heart_rate, expected", [(55.0, 0), (70.0, 1), (100.0, 1), (110.0, 2)])
def test_heart_rate_category(heart_rate, expected):
    out = engineer_rows([patient(heartRate=heart_rate)])
    assert out.loc[0, "hr_category"] == expected


def test_interaction_and_risk_count_features():
    out = engineer_rows([
        patient(is_smoking="YES", diabetes=1, prevalentHyp=1, BMI=32.0,
                BPMeds=1.0, sysBP=150.0, age=60, totChol=250.0),
    ])
    row = out.loc[0]
    assert row["bmi_diabetes_interaction"] == pytest.approx(32.0)
    assert row["bp_meds_effectiveness"] == pytest.approx(0.0)  # on medication
    assert row["age_smoking_interaction"] == pytest.approx(60)
    assert row["chol_age_interaction"] == pytest.approx(250 * 60 / 100)
    assert row["risk_factor_count"] == 4  # smoker + diabetic + hypertensive + BMI > 30


def test_smoking_intensity_is_encoded_in_its_natural_order():
    """OrdinalEncoder used to sort alphabetically: heavy=0 ... non_smoker=3."""
    rows = [patient(smoking_intensity=level) for level in SMOKING_INTENSITY_ORDER]
    out = engineer_rows(rows)
    assert out["smoking_intensity_encoded"].tolist() == [0, 1, 2, 3]


# ------------------------------------------------------------------ on the real data

def test_real_data_smoking_mapping_is_ordinal(engineered):
    _, _, engineer = engineered
    assert engineer.categorical_mappings["smoking_intensity"] == {
        level: code for code, level in enumerate(SMOKING_INTENSITY_ORDER)
    }


@pytest.mark.parametrize(
    "col",
    ["diabetes", "BPMeds", "bmi_diabetes_interaction",
     "hr_category", "glucose_category", "hypertension_stage", "bmi_category"],
)
def test_engineered_features_are_not_constant(engineered, col):
    """Upstream capping bugs showed up here as all-zero or single-value columns."""
    train_engineered, _, _ = engineered
    assert train_engineered[col].nunique() > 1


def test_transform_matches_training_columns(engineered):
    train_engineered, test_engineered, _ = engineered
    expected = [c for c in train_engineered.columns if c != TARGET_COLUMN]
    assert test_engineered.columns.tolist() == expected


def test_new_features_are_listed_once(engineered):
    """transform() re-runs the same builders; the list used to double to 30."""
    _, _, engineer = engineered
    assert len(engineer.new_features) == len(set(engineer.new_features))
    assert len(engineer.get_new_feature_descriptions()) == len(engineer.new_features)


def test_all_model_inputs_are_numeric(engineered):
    train_engineered, test_engineered, _ = engineered
    assert train_engineered.select_dtypes(exclude="number").empty
    assert test_engineered.select_dtypes(exclude="number").empty
