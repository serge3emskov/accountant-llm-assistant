from __future__ import annotations

from acctltm.retrieval import DocumentRecord, select_sections, statuses_for
from acctltm.schemas import QueryType
from acctltm.sections import Section
from conftest import USER, ev, write


def test_select_sections_posting():
    secs = select_sections([QueryType.POSTING])
    assert Section.COMPANY_PROFILE in secs
    assert Section.COUNTERPARTIES in secs


def test_statuses_history_allows_terminated():
    allowed = statuses_for(Section.COUNTERPARTIES, [QueryType.HISTORY])
    assert "terminated" in allowed


def test_context_render(writer, builder):
    write(writer, ev("company_profile", attribute="regime", value="УСН", evidence="мы на усн"))
    write(writer, ev("counterparties", "Ромашка", status="active", evidence="работаем с ромашкой"))
    bundle = builder.build(USER, [QueryType.POSTING])
    text = bundle.render()
    assert "УСН" in text
    assert "ромашка" in text


def test_clarify_unknown_regime(builder):
    bundle = builder.build(USER, [QueryType.POSTING])
    assert any(c.section == Section.COMPANY_PROFILE and c.key == "regime" for c in bundle.clarify)


def test_stale_practices_flag(writer, builder, clock):
    write(writer, ev("practices", attribute="cash_settlements", value="да", evidence="расчёты наличными"))
    clock.advance(400)
    bundle = builder.build(USER, [QueryType.POSTING])
    assert any(c.section == Section.PRACTICES and c.reason == "stale" for c in bundle.clarify)


def test_documents_block(writer, builder):
    write(writer, ev("counterparties", "Ромашка", status="active", evidence="работаем с ромашкой"))
    docs = [DocumentRecord(date="2026-02-01", doc_type="платёжное поручение",
                           counterparty_raw="Ромашка", amount=150000)]
    bundle = builder.build(USER, [QueryType.DOCUMENT_CHECK], documents=docs)
    assert "ДОКУМЕНТИРОВАНО" in bundle.render()


def test_denied_render(writer, builder):
    write(writer, ev("counterparties", "Ромашка", polarity="negative", evidence="никогда не работали с ромашкой"))
    bundle = builder.build(USER, [QueryType.DOCUMENT_CHECK])
    assert "НЕТ" in bundle.render()


def test_general_no_context(builder):
    bundle = builder.build(USER, [QueryType.GENERAL])
    assert bundle.sections == []
    assert "персональный контекст не нужен" in bundle.render()
