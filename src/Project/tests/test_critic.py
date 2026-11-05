from agents.critic import DOSAGE, allowed_numbers, diagnosis_sentences, rule_issues

STATE = {
    "question": "HbA1c 6.7%, đường huyết 150 mg/dL",
    "patient": {"age": 52.0, "bmi": 28.0, "HbA1c_level": 6.7, "blood_glucose_level": 150.0},
    "risk": {"probability": 0.109, "threshold": 0.4368, "what_if": []},
    "evidence": [{"n": 1, "fact": "Đái tháo đường được chẩn đoán bằng HbA1c (>= 6.5 %)", "quote": "HbA1c ≥ 6,5%",
                  "source_url": "u", "title": "t"}],
    "lab_checks": [{"feature": "HbA1c_level", "test": "HbA1c", "condition": "Đái tháo đường", "op": ">=", "value": 6.5,
                    "value_max": None, "unit": "%", "patient_value_converted": 6.7, "met": True}],
}


def test_dosage_regex_ignores_glucose_units():
    assert not DOSAGE.search("đường huyết 150 mg/dL")
    assert not DOSAGE.search("giảm BMI xuống 24 có thể giúp giảm nguy cơ")  # "xuống" chứa "uống"
    assert DOSAGE.search("uống metformin 500 mg mỗi ngày") and DOSAGE.search("ngày 2 viên")


def test_diagnosis_detection_respects_hedges():
    text = "Bạn bị tiểu đường type 2. Bạn có nguy cơ mắc tiểu đường. Bạn không nên khẳng định mình mắc bệnh."
    assert diagnosis_sentences(text) == ["Bạn bị tiểu đường type 2."]


def test_numbers_must_come_from_model_or_sources():
    assert {10.9, 43.7, 6.5, 6.7} <= allowed_numbers(STATE)
    good = "Xác suất 10,9% so với ngưỡng 43,7%. HbA1c 6,7% đạt ngưỡng ≥ 6,5% [1]."
    assert rule_issues(STATE, good) == []
    bad = rule_issues(STATE, "Xác suất 72% [1].")
    assert any("72" in issue for issue in bad)


def test_citations_and_threshold_mention_required():
    issues = rule_issues(STATE, "Nguy cơ thấp 10,9% [3].")
    assert any("[3]" in i for i in issues)
    assert any("HbA1c" in i for i in issues)


def test_llm_claims_backed_by_numbers_or_citations_are_dropped():
    from agents.critic import supported_by_rules

    allowed = allowed_numbers(STATE)
    assert supported_by_rules("HbA1c của bạn là 6,7%, đạt ngưỡng 6,5%.", STATE, allowed)
    assert supported_by_rules("Tăng huyết áp là yếu tố nguy cơ [1].", STATE, allowed)
    assert not supported_by_rules("Tiểu đường làm tăng nguy cơ ung thư tụy.", STATE, allowed)


def test_placeholder_citation_is_flagged():
    assert any("[n]" in issue for issue in rule_issues(STATE, "HbA1c 6,7% đạt ngưỡng 6,5% [1]. Giảm cân giúp ích [n]."))


def test_fasting_glucose_must_be_stated_as_assumption():
    state = {**STATE, "lab_checks": STATE["lab_checks"] + [
        {"feature": "blood_glucose_level", "test": "Đường huyết lúc đói", "condition": "Đái tháo đường", "op": ">=",
         "value": 126.0, "value_max": None, "unit": "mg/dL", "patient_value_converted": 150.0, "met": True}]}
    def flags_assumption(st, text):
        return any("giả định" in issue for issue in rule_issues(st, text))
    assumed = "HbA1c 6,7% đạt ngưỡng ≥ 6,5% [1]. Đường huyết lúc đói 150 mg/dL vượt ngưỡng 126 mg/dL [1]."
    assert flags_assumption(state, assumed)
    hedged = "HbA1c 6,7% đạt ngưỡng ≥ 6,5% [1]. Nếu đường huyết 150 mg/dL là kết quả đo lúc đói thì vượt ngưỡng 126 mg/dL [1]."
    assert not flags_assumption(state, hedged)
    assert not flags_assumption({**state, "question": "Đường huyết lúc đói của tôi là 150 mg/dL"}, assumed)


def test_source_titles_and_links_carry_no_dosage():
    from eval.metrics import visible_text
    from kb.textutil import source_title
    title = source_title("Thuốc Metformin Stella 1000mg điều trị đái tháo đường típ 2 (6 vỉ x 10 viên)")
    assert title == "Thuốc Metformin Stella điều trị đái tháo đường típ 2" and not DOSAGE.search(title)
    answer = f"Metformin có thể gây tiêu chảy [1].\n\n- [1] [{title}](https://x/thuoc/metformin-1000mg-stella.html)"
    assert not DOSAGE.search(visible_text(answer))


def test_lab_check_accepts_test_synonyms():
    state = {**STATE, "question": "đường huyết lúc đói 220 mg/dL", "lab_checks": [
        {"feature": "blood_glucose_level", "test": "Glucose", "condition": "Đái tháo đường", "op": ">=", "value": 200.0,
         "value_max": None, "unit": "mg/dL", "patient_value_converted": 220.0, "met": True}]}
    named_differently = "Đường huyết 220 mg/dL đạt ngưỡng ≥ 200 mg/dL [1]."
    assert not any("Phải nêu" in issue for issue in rule_issues(state, named_differently))
    assert any("Phải nêu" in issue for issue in rule_issues(state, "Bạn nên gặp bác sĩ [1]."))

