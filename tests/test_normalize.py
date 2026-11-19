from __future__ import annotations

from acctltm.sections import Section


def test_counterparty_normalization(normalizer):
    assert normalizer.entity(Section.COUNTERPARTIES, "Ромашка").normalized == "ооо ромашка"
    assert normalizer.entity(Section.COUNTERPARTIES, "ООО Ромашка").normalized == "ооо ромашка"


def test_obligation_normalization(normalizer):
    assert normalizer.entity(Section.OBLIGATIONS, "декларация по НДС").normalized == "декларация ндс"
    assert normalizer.entity(Section.OBLIGATIONS, "РСВ").normalized == "рсв"


def test_contract_free_key(normalizer):
    nr = normalizer.entity(Section.CONTRACTS, "Договор поставки с Ромашкой")
    assert nr.normalized == "договор поставки с ромашкой"


def test_issue_free_key(normalizer):
    assert normalizer.entity(Section.ISSUES, "Не сходится касса").normalized == "не сходится касса"


def test_subject_mapping(normalizer):
    assert normalizer.subject("наша компания") == "self"
    assert normalizer.subject("дочка") == "дочка"
    assert normalizer.subject("ип директора") == "ип директора"
    assert normalizer.subject(None) == "self"


def test_unmatched_counterparty(normalizer):
    assert normalizer.entity(Section.COUNTERPARTIES, "НеизвестнаяФirma").method == "unmatched"


def test_amount_from_name(normalizer):
    nr = normalizer.entity(Section.CONTRACTS, "договор аренды на 200 тысяч")
    assert nr.amount == 200000
