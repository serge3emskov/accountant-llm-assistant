"""Сквозное демо ядра ассистента бухгалтера без GPU (LLM-шаги замоканы ScriptedLLM).

Показывает: write-path (память), read-path (контекст), проверку соответствия
и ответ с предупреждениями, дописанными кодом. Запуск: python scripts/demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acctltm import AccountantAssistant, MemoryStore  # noqa: E402
from acctltm.llm import ScriptedLLM  # noqa: E402
from acctltm.synthetic import load_bank_statement  # noqa: E402

SECTIONS = ["company_profile", "counterparties", "contracts", "practices"]
EXTRACTION = {
    "company_profile": {"facts": [
        {"section": "company_profile", "attribute": "regime", "value": "УСН", "evidence": "Мы на УСН"},
        {"section": "company_profile", "attribute": "usn_object", "value": "доходы", "evidence": "УСН доходы"},
        {"section": "company_profile", "attribute": "vat_payer", "value": "нет", "evidence": "НДС не платим"},
    ], "section_statuses": []},
    "counterparties": {"facts": [
        {"section": "counterparties", "name": "Ромашка", "status": "active", "role": "поставщик",
         "evidence": "Работаем с Ромашкой как поставщик"},
    ], "section_statuses": []},
    "contracts": {"facts": [
        {"section": "contracts", "name": "договор поставки с Ромашкой", "status": "active",
         "amount": 150000, "payment": "аванс", "evidence": "договор поставки на 150 тысяч с авансом"},
    ], "section_statuses": []},
    "practices": {"facts": [
        {"section": "practices", "attribute": "cash_settlements", "value": "да", "evidence": "Расчёты наличными"},
    ], "section_statuses": []},
}


def responder(messages, meta):
    task = meta.get("task")
    if task == "classify_sections":
        return json.dumps({"sections": SECTIONS})
    if task in ("extract", "extract_repair"):
        return json.dumps(EXTRACTION.get(meta.get("section"), {"facts": [], "section_statuses": []}))
    if task == "classify_query":
        return json.dumps({"query_types": ["deduction", "posting"], "subject": "self"})
    if task == "answer":
        last = messages[-1]["content"]
        if "Можно оплатить" in last:
            return "Можно оплатить наличными и принять расходы на аренду."  # намеренно без учёта правил
        return "Понял, запомнил профиль компании. Чем помочь?"
    return "{}"


def main() -> None:
    store = MemoryStore()
    assistant = AccountantAssistant(ScriptedLLM(responder), store=store)
    docs = load_bank_statement()[:5]
    session = assistant.session("demo", "d1", documents=docs)

    print("=" * 80)
    turn1 = ("Мы на УСН доходы, НДС не платим. Работаем с Ромашкой как поставщик, "
             "договор поставки на 150 тысяч с авансом. Расчёты наличными.")
    print("Бухгалтер:", turn1)
    print("Ассистент:", session.say(turn1))
    result = session.close()
    print("\n-- write-path: изменения памяти --")
    print(json.dumps(result.summary()["changes"], ensure_ascii=False))
    print("\n-- факты в памяти --")
    for f in store.list_facts("demo"):
        print(f"  {f.section.value:18} {f.key:35} {f.status:12} {f.data}")

    print("\n" + "=" * 80)
    turn2 = "Можно оплатить Ромашке наличными 150 тысяч и принять расходы на аренду?"
    print("Бухгалтер:", turn2)
    session.say(turn2)
    reply = session.replies[-1]
    print("\n-- контекст (read-path), фрагмент --")
    ctx = reply.read.bundle.render()
    print(ctx[:1200])
    print("\n-- предупреждения проверки (compliance) --")
    if reply.compliance:
        for a in reply.compliance.alerts:
            print("  ", a.render())
    print("\n-- итоговый ответ (предупреждения дописаны кодом) --")
    print(reply.text)


if __name__ == "__main__":
    main()
