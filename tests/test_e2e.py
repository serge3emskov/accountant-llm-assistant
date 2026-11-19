from __future__ import annotations

import json

from acctltm import AccountantAssistant, MemoryStore
from acctltm.llm import ScriptedLLM
from acctltm.schemas import QueryType
from acctltm.sections import Section


def test_session_close_persists():
    def responder(messages, meta):
        t = meta.get("task")
        if t == "classify_sections":
            return json.dumps({"sections": ["company_profile"]})
        if t in ("extract", "extract_repair"):
            return json.dumps({"facts": [
                {"section": "company_profile", "attribute": "regime", "value": "УСН", "evidence": "мы на усн"}],
                "section_statuses": []})
        if t == "classify_query":
            return json.dumps({"query_types": ["posting"], "subject": "self"})
        if t == "answer":
            return "Учту, что вы на УСН."
        return "{}"

    a = AccountantAssistant(ScriptedLLM(responder), store=MemoryStore())
    s = a.session("u", "d1")
    s.say("Мы на УСН.")
    s.close()
    assert a.store.get_fact("u", "self", Section.COMPANY_PROFILE, "regime")
    bundle = a.builder.build("u", [QueryType.POSTING])
    assert "УСН" in bundle.render()


def test_subject_isolation_e2e():
    a = AccountantAssistant(ScriptedLLM(lambda m, meta: "{}"))
    a.writer.write("u", {"facts": [
        {"section": "company_profile", "subject": "self", "attribute": "regime", "value": "УСН", "evidence": "усн"},
        {"section": "company_profile", "subject": "дочка", "attribute": "regime", "value": "ОСНО", "evidence": "осно"},
    ], "section_statuses": []}, "d", dialog_text="усн осно")
    b = a.builder.build("u", [QueryType.POSTING], subject="дочка")
    text = b.render()
    assert "ОСНО" in text and "УСН" not in text
