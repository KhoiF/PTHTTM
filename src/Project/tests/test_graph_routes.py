"""Kiểm tra định tuyến của LangGraph bằng LLM giả lập (tất định, không cần Ollama/Neo4j)."""

from types import SimpleNamespace

import pytest

from agents.graph import build_graph, route_of, run_case
from tools import kg_tools

FACT = {"fact": "Đái tháo đường được chẩn đoán bằng HbA1c (>= 6.5 %)", "quote": "chẩn đoán khi HbA1c ≥ 6,5%",
        "source_url": "https://nhathuoclongchau.com.vn/benh/tieu-duong-tuyp-2.html", "title": "Tiểu đường type 2"}


class FakeLLM:
    def __init__(self, intent="assess", patient=None, drafts=None):
        self.intent, self.patient = intent, patient or {}
        self.drafts = list(drafts or ["Theo nguồn [1], chỉ số HbA1c là tiêu chí quan trọng. Bạn nên gặp bác sĩ."])

    def structured(self, schema, system, user, **kwargs):
        name = schema.__name__
        if name == "Intent":
            return schema(intent=self.intent, reason="giả lập")
        if name == "PatientExtraction":
            return schema(**self.patient)
        if name == "LabExtraction":
            return schema()
        return schema(recommends_doctor=True, unsupported_claims=[])

    def run_tools(self, system, user, tools, **kwargs):
        from agents.llm import ToolRun

        log = []
        for fn in tools:
            if fn.__name__ in {"predict_risk", "explain_risk"}:
                log.append({"tool": fn.__name__, "args": {}, "result": fn()})
            if fn.__name__ == "search_chunks":
                log.append({"tool": "search_chunks", "args": {"query": "q"}, "result": fn("q")})
        return ToolRun("tóm tắt", log)

    def chat(self, messages, **kwargs):
        draft = self.drafts.pop(0) if len(self.drafts) > 1 else self.drafts[0]
        return SimpleNamespace(message=SimpleNamespace(content=draft))


@pytest.fixture(autouse=True)
def fake_kg(monkeypatch):
    def search_chunks(query, k=4):
        return {"facts": [FACT, {**FACT, "quote": "đoạn khác"}, {**FACT, "quote": "đoạn thứ ba"}]}

    def explain_feature(feature, value=None):
        return {"facts": [FACT], "thresholds": []}

    monkeypatch.setattr(kg_tools, "search_chunks", search_chunks)
    monkeypatch.setattr(kg_tools, "explain_feature", explain_feature)


def run(llm, question="", form=None):
    return run_case(question, form, app=build_graph(llm))


def test_knowledge_question_skips_intake_and_risk():
    state = run(FakeLLM(intent="ask"), "Triệu chứng tiểu đường là gì?")
    assert route_of(state) == "Supervisor → Knowledge → Advisor → Critic → Finalize"


def test_missing_profile_asks_user_back():
    state = run(FakeLLM(patient={"age": 40, "bmi": 25}), "Tôi 40 tuổi, BMI 25")
    assert route_of(state) == "Supervisor → Intake → Finalize"
    assert "Giới tính" in state["answer"]


def test_grey_zone_without_labs_loops_back_to_intake():
    patient = dict(gender="Male", age=50, bmi=29, hypertension=False, heart_disease=False, smoking_history="never")
    state = run(FakeLLM(patient=patient), "Tôi nam 50 tuổi, BMI 29")
    assert "Risk → Intake → Risk" in route_of(state)
    assert state["risk"]["model_used"] == "screening" and state["follow_up"]


def test_critic_sends_unsafe_draft_back_to_advisor():
    drafts = ["Bạn bị tiểu đường type 2 [1].", "Bạn có nguy cơ, HbA1c là tiêu chí quan trọng [1]. Hãy gặp bác sĩ."]
    state = run(FakeLLM(intent="ask", drafts=drafts), "HbA1c là gì?")
    assert "Advisor → Critic → Advisor → Critic" in route_of(state)
    assert state["critique"]["passed"]


def test_ablation_without_critic():
    state = run_case("Triệu chứng?", app=build_graph(FakeLLM(intent="ask")), options={"use_critic": False})
    assert "Critic" not in route_of(state)


def test_strip_latex_turns_math_symbols_into_unicode():
    from agents.graph import strip_latex
    assert strip_latex(r"HbA1c $\ge$ 6,5% và glucose $\geq 126$") == "HbA1c ≥ 6,5% và glucose ≥ 126"
    assert strip_latex("giá 5$ và 10$") == "giá 5$ và 10$"


def test_finalize_lists_consulted_sources_when_draft_has_no_citation():
    from agents.graph import finalize_node
    evidence = [{"n": 1, "fact": "f", "quote": "q", "source_url": "https://a", "title": "Bài A"}]
    answer = finalize_node({"draft": "Bạn nên gặp bác sĩ.", "evidence": evidence})["answer"]
    assert "Nguồn đã tra cứu" in answer and "https://a" in answer
    cited = finalize_node({"draft": "Theo nguồn [1].", "evidence": evidence})["answer"]
    assert "Nguồn (Nhà thuốc Long Châu)" in cited and "Nguồn đã tra cứu" not in cited


def test_regex_profile_fills_what_the_llm_missed():
    from agents.intake import regex_profile
    assert regex_profile("Ông tôi 70 tuổi, BMI 35") == {"age": 70.0, "bmi": 35.0, "gender": "Male"}
    assert regex_profile("Mẹ tôi 58 tuổi") == {"age": 58.0, "gender": "Female"}
    assert "gender" not in regex_profile("Tôi sống ở Việt Nam, 40 tuổi")


def test_supervisor_upgrades_assess_with_knowledge_question_to_both():
    from agents.supervisor import Intent, supervisor_node

    class StubLLM:
        def structured(self, model, *args, **kwargs):
            return Intent(intent="assess", reason="có thông tin cá nhân")

    both = supervisor_node({"question": "Tôi 60 tuổi, hay khát nước. Đó có phải dấu hiệu tiểu đường không?"}, StubLLM())
    assert both["intent"] == "both"
    assess = supervisor_node({"question": "Tôi nam 50 tuổi, BMI 29. Nguy cơ tiểu đường của tôi thế nào?"}, StubLLM())
    assert assess["intent"] == "assess"

