from __future__ import annotations

from acctltm.memory import ChangeType, classify_transition
from acctltm.sections import POLICIES, Section
from conftest import USER, absent, ev, write


def test_classify_first_mention():
    assert classify_transition(POLICIES[Section.COUNTERPARTIES], None, "active", False) == ChangeType.FIRST_MENTION


def test_classify_negation():
    assert classify_transition(POLICIES[Section.COUNTERPARTIES], "active", "terminated", False) == ChangeType.DEACTIVATION


def test_negation_to_terminated(writer):
    write(writer, ev("counterparties", "Ромашка", status="active", evidence="работаем с ромашкой"))
    write(writer, ev("counterparties", "Ромашка", polarity="negative", evidence="не работаем с ромашкой"))
    f = writer.store.get_fact(USER, "self", Section.COUNTERPARTIES, "ооо ромашка")
    assert f.status == "terminated"


def test_reactivation(writer):
    write(writer, ev("counterparties", "Ромашка", status="active", evidence="работаем"))
    write(writer, ev("counterparties", "Ромашка", status="terminated", evidence="расторгли"))
    rep = write(writer, ev("counterparties", "Ромашка", status="active", evidence="снова работаем"))
    assert rep.changes(ChangeType.REACTIVATION)
    assert writer.store.get_fact(USER, "self", Section.COUNTERPARTIES, "ооо ромашка").status == "active"


def test_confirmation(writer):
    write(writer, ev("counterparties", "Ромашка", status="active", evidence="работаем"))
    rep = write(writer, ev("counterparties", "Ромашка", status="active", evidence="работаем"))
    assert rep.changes(ChangeType.CONFIRMATION)


def test_update_amount(writer):
    write(writer, ev("contracts", "договор аренды", status="active", amount=100000, evidence="договор"))
    rep = write(writer, ev("contracts", "договор аренды", status="active", amount=200000, evidence="договор"))
    assert rep.changes(ChangeType.UPDATE)
    assert writer.store.get_fact(USER, "self", Section.CONTRACTS, "договор аренды").data["amount"] == 200000


def test_section_absent(writer):
    write(writer, statuses=[absent("counterparties", evidence="контрагентов нет")])
    st = writer.store.get_section_state(USER, "self", Section.COUNTERPARTIES)
    assert st.state == "confirmed_absent"


def test_conflict(writer):
    write(writer, ev("counterparties", "Ромашка", status="active", evidence="работаем"))
    write(writer, statuses=[absent("counterparties", evidence="контрагентов нет")])
    st = writer.store.get_section_state(USER, "self", Section.COUNTERPARTIES)
    assert st.needs_clarification is True


def test_new_negative_document_missing(writer):
    write(writer, ev("documents", "накладная от Ромашки", polarity="negative", evidence="накладной нет"))
    f = writer.store.get_fact(USER, "self", Section.DOCUMENTS, "накладная от ромашки")
    assert f.status == "missing"


def test_practices_marker_reject(writer):
    rep = writer.write(USER, {"facts": [ev("practices", attribute="cash_settlements", value="да",
                                           evidence="работаем с ромашкой")], "section_statuses": []},
                       "d", dialog_text="работаем с ромашкой")
    assert rep.rejected
    assert writer.store.list_facts(USER, "self", [Section.PRACTICES]) == []


def test_evidence_grounding_reject(writer):
    rep = writer.write(USER, {"facts": [ev("counterparties", "Ромашка", status="active",
                                           evidence="оплатили наличными миллион")], "section_statuses": []},
                       "d", dialog_text="работаем с ромашкой")
    assert rep.rejected


def test_generic_counterparty_to_status(writer):
    write(writer, ev("counterparties", "контрагенты", polarity="negative", evidence="контрагентов нет"))
    st = writer.store.get_section_state(USER, "self", Section.COUNTERPARTIES)
    assert st.state == "confirmed_absent"
    assert writer.store.list_facts(USER, "self", [Section.COUNTERPARTIES]) == []
