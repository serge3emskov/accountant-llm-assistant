from __future__ import annotations

import json

from acctltm import AccountantAssistant, MemoryStore
from acctltm.llm import ScriptedLLM
from acctltm.schemas import QueryType
from acctltm.sections import Section


def make_llm(sections, extraction, answer="Ответ.", query_types=("posting",)):
    def responder(messages, meta):
        task = meta.get("task")
        if task == "classify_sections":
            return json.dumps({"sections": sections})
        if task in ("extract", "extract_repair"):
            return json.dumps(extraction.get(meta.get("section"), {"facts": [], "section_statuses": []}))
        if task == "classify_query":
            return json.dumps({"query_types": list(query_types), "subject": "self"})
        if task == "answer":
            return answer
        return "{}"
    return ScriptedLLM(responder)


REGIME = {"company_profile": {"facts": [
    {"section": "company_profile", "attribute": "regime", "value": "УСН", "evidence": "мы на усн"}],
    "section_statuses": []}}


def test_write_path_saves():
    a = AccountantAssistant(make_llm(["company_profile"], REGIME), store=MemoryStore())
    a.write_path.process("u", [{"role": "user", "content": "мы на усн"}], "d")
    assert a.store.get_fact("u", "self", Section.COMPANY_PROFILE, "regime")


def test_safety_net_adds_counterparty():
    extraction = {"counterparties": {"facts": [
        {"section": "counterparties", "name": "Ромашка", "status": "active", "evidence": "работаем с ромашкой"}],
        "section_statuses": []}}
    a = AccountantAssistant(make_llm([], extraction))
    res = a.write_path.process("u", [{"role": "user", "content": "Работаем с Ромашкой."}], "d")
    assert Section.COUNTERPARTIES in res.safety_net
    assert a.store.get_fact("u", "self", Section.COUNTERPARTIES, "ооо ромашка")


def test_off_section_dropped():
    extraction = {"counterparties": REGIME["company_profile"]}  # факт чужого раздела
    a = AccountantAssistant(make_llm(["counterparties"], extraction))
    res = a.write_path.process("u", [{"role": "user", "content": "усн"}], "d")
    assert res.stats.off_section >= 1
    assert a.store.get_fact("u", "self", Section.COMPANY_PROFILE, "regime") is None


def test_classifier_fallback_all_sections():
    def responder(messages, meta):
        if meta.get("task") == "classify_sections":
            return "сломанный json"
        if meta.get("task") in ("extract", "extract_repair"):
            return json.dumps({"facts": [], "section_statuses": []})
        return "{}"
    a = AccountantAssistant(ScriptedLLM(responder))
    res = a.write_path.process("u", [{"role": "user", "content": "тест"}], "d")
    assert res.classifier_failed
    assert len(res.sections) == len(list(Section))


def test_reply_compliance_appends():
    store = MemoryStore()
    a = AccountantAssistant(make_llm([], {}, answer="Можно принять расходы на аренду."), store=store)
    a.writer.write("u", {"facts": [
        {"section": "company_profile", "attribute": "regime", "value": "УСН", "evidence": "усн"},
        {"section": "company_profile", "attribute": "usn_object", "value": "доходы", "evidence": "доходы"},
    ], "section_statuses": []}, "d", dialog_text="усн доходы")
    reply = a.reply("u", "Можно принять расходы на аренду?")
    assert reply.compliance is not None
    assert reply.compliance.appended
    assert "Проверка по справочнику" in reply.text


def test_query_classifier_fallback_rules():
    a = AccountantAssistant(make_llm([], {}), use_llm_query_classifier=False)
    read = a.read_path("u", "когда сдавать декларацию")
    assert read.fallback_used
    assert QueryType.DEADLINE in read.classification.query_types
