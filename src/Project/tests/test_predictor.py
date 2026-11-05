import pytest

from ml.predictor import clean_patient, explain, predict_risk, what_if

POSITIVE = dict(gender="Nam", age=58, bmi=33.5, hypertension=1, heart_disease=0, smoking_history="former",
                HbA1c_level=7.5, blood_glucose_level=220)
NEGATIVE = dict(gender="Nữ", age=28, bmi=21.0, hypertension=0, heart_disease=0, smoking_history="never",
                HbA1c_level=5.0, blood_glucose_level=100)
NO_LABS = dict(gender="Female", age=62, bmi=34, hypertension=1, heart_disease=1)


def test_full_model_separates_clear_cases():
    pos, neg = predict_risk(POSITIVE), predict_risk(NEGATIVE)
    assert pos["ok"] and neg["ok"]
    assert pos["model_used"] == "full" and pos["label"] == 1 and neg["label"] == 0
    assert 0.01 <= neg["probability"] < pos["probability"] <= 0.99


def test_missing_labs_switch_to_screening():
    result = predict_risk(NO_LABS)
    assert result["model_used"] == "screening"
    assert result["missing_labs"] == ["HbA1c_level", "blood_glucose_level"]
    assert result["patient"]["smoking_history"] == "No Info"


@pytest.mark.parametrize("value", ["None", 0, "", None])
def test_llm_placeholders_count_as_missing(value):
    clean, errors = clean_patient({**POSITIVE, "HbA1c_level": value})
    assert clean["HbA1c_level"] is None and not errors


def test_missing_and_invalid_fields_are_reported():
    assert predict_risk(dict(age=40, bmi=25))["missing_fields"] == ["gender", "hypertension", "heart_disease"]
    bad = predict_risk({**POSITIVE, "bmi": 250})
    assert not bad["ok"] and "BMI" in bad["errors"][0]


def test_explain_ranks_lab_values_first_for_full_model():
    top = explain(POSITIVE, k=3)
    assert {top[0]["feature"], top[1]["feature"]} == {"HbA1c_level", "blood_glucose_level"}
    assert all(item["direction"] == "tăng nguy cơ" for item in top[:2])


def test_what_if_uses_same_model():
    result = what_if(NO_LABS, {"bmi": 24})
    assert result["ok"] and result["model_used"] == "screening"
    assert result["after"] <= result["before"]
