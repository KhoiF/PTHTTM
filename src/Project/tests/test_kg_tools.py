from tools import kg_tools as T


def test_link_entities_handles_synonyms(fixture_kg):
    top = T.link_entities("tiểu đường type II", k=3)["entities"][0]
    assert top["id"] == "Disease:dai thao duong tuyp 2"


def test_explain_feature_flags_hba1c_threshold(fixture_kg):
    result = T.explain_feature("HbA1c", 6.7)
    assert result["feature"] == "HbA1c_level"
    met = [t for t in result["thresholds"] if t.get("met")]
    assert any(t["condition"] == "Đái tháo đường" and t["value"] == 6.5 for t in met)
    assert all(f["source_url"].startswith("https://nhathuoclongchau.com.vn/") for f in result["facts"])


def test_explain_feature_converts_glucose_units(fixture_kg):
    result = T.explain_feature("blood_glucose_level", 150)
    fasting = next(t for t in result["thresholds"] if "đói" in t["test"])
    assert fasting["unit"] == "mmol/L" and fasting["patient_value_converted"] == 8.33 and fasting["met"]
    assert result["note"]  # phải nhắc giả định đo lúc đói hay bất kỳ


def test_neighbors_filters_relation(fixture_kg):
    facts = T.neighbors("đái tháo đường tuýp 2", ["HAS_COMPLICATION"])["facts"]
    assert facts and all("biến chứng" in f["fact"] for f in facts)


def test_drug_info_never_returns_dosage(fixture_kg):
    result = T.drug_info("metformin")
    assert result["drug"] == "Metformin" and "liều" in result["note"]


def test_search_chunks_returns_source_quotes(fixture_kg):
    facts = T.search_chunks("phòng ngừa bằng chế độ ăn", k=2)["facts"]
    assert facts and "phòng ngừa" in facts[0]["fact"].lower()


def test_met_respects_exclusive_upper_bound():
    assert T._met(125.9, "range", 100, 126, max_exclusive=True)
    assert not T._met(126, "range", 100, 126, max_exclusive=True)
    assert T._met(6.4, "range", 5.7, 6.4)          # "5,7 đến 6,4%": cận trên lấy dấu bằng

