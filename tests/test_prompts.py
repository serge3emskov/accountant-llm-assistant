from __future__ import annotations

from acctltm.prompts import (
    CLASSIFIER_EXAMPLES, EXTRACTOR_EXAMPLES, QUERY_EXAMPLES, answer_messages,
    compact_dialog, extractor_messages, query_classifier_messages, section_classifier_messages,
)
from acctltm.schemas import QueryType, parse_extraction
from acctltm.sections import Section


def test_extractor_examples_valid():
    for sec, examples in EXTRACTOR_EXAMPLES.items():
        for turns, out in examples:
            facts, statuses, rej = parse_extraction(out)
            assert not rej, (sec, [r.errors for r in rej])
            for f in facts:
                assert f.section == sec.value


def test_classifier_examples_sections():
    for turns, secs in CLASSIFIER_EXAMPLES:
        for s in secs:
            Section(s)


def test_query_examples_types():
    for msg, out in QUERY_EXAMPLES:
        for t in out["query_types"]:
            QueryType(t)


def test_messages_build():
    turns = [{"role": "user", "content": "тест"}]
    assert extractor_messages(Section.COUNTERPARTIES, turns)
    assert section_classifier_messages(turns)
    assert query_classifier_messages("тест")
    assert answer_messages("контекст", [], "сообщение")


def test_compact_dialog_keeps_questions():
    turns = [{"role": "user", "content": "мы на усн"},
             {"role": "assistant", "content": "Длинный совет про налоги. Есть ли договоры?"}]
    out = compact_dialog(turns)
    assert out[-1]["content"] == "Есть ли договоры?"


def test_compact_dialog_drops_advice():
    turns = [{"role": "assistant", "content": "Очень длинный совет без вопроса про проводки и счета."}]
    assert compact_dialog(turns) == []
