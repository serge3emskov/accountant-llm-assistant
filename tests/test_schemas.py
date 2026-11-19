from __future__ import annotations

import pytest
from pydantic import ValidationError

from acctltm.schemas import (
    CompanyProfileEvent, ContractEvent, CounterpartyEvent, DocumentEvent,
    PracticesEvent, SectionStatusEvent, parse_extraction,
)


def test_regime_synonyms():
    assert CompanyProfileEvent(section="company_profile", attribute="regime", value="упрощёнка",
                               evidence="на упрощёнке").value == "УСН"
    assert CompanyProfileEvent(section="company_profile", attribute="regime", value="патент",
                               evidence="патент").value == "ПСН"
    assert CompanyProfileEvent(section="company_profile", attribute="regime", value="самозанятость",
                               evidence="самозанятость").value == "НПД"


def test_usn_object_synonyms():
    assert CompanyProfileEvent(section="company_profile", attribute="usn_object", value="15%",
                               evidence="15 процентов").value == "доходы-расходы"
    assert CompanyProfileEvent(section="company_profile", attribute="usn_object", value="6%",
                               evidence="6%").value == "доходы"


def test_revenue_parse():
    assert CompanyProfileEvent(section="company_profile", attribute="revenue_year", value="45 млн",
                               evidence="45 млн").value == 45000000


def test_employees_int():
    assert CompanyProfileEvent(section="company_profile", attribute="employees", value="5",
                               evidence="пятеро").value == 5


def test_bad_regime_rejected():
    with pytest.raises(ValidationError):
        CompanyProfileEvent(section="company_profile", attribute="regime", value="ЕНВД", evidence="енвд")


def test_bad_vat_payer_rejected():
    with pytest.raises(ValidationError):
        CompanyProfileEvent(section="company_profile", attribute="vat_payer", value="возможно", evidence="хз")


def test_contract_amount_from_text():
    assert ContractEvent(section="contracts", name="договор", amount="150 тысяч",
                         evidence="на 150 тысяч").amount == 150000


def test_inn_checksum():
    assert CounterpartyEvent(section="counterparties", name="Ромашка", inn="7701234560",
                             evidence="инн").inn == "7701234560"
    with pytest.raises(ValidationError):
        CounterpartyEvent(section="counterparties", name="Ромашка", inn="7701234561", evidence="инн")


def test_practices_value():
    assert PracticesEvent(section="practices", attribute="kkt", value="нет", evidence="кассы нет").value == "нет"
    with pytest.raises(ValidationError):
        PracticesEvent(section="practices", attribute="kkt", value="maybe", evidence="x")


def test_document_negative_ok():
    e = DocumentEvent(section="documents", name="накладная", polarity="negative", evidence="накладной нет")
    assert e.polarity == "negative"


def test_negation_drops_bad_status():
    e = CounterpartyEvent(section="counterparties", name="Ромашка", polarity="negative",
                          status="confirmed_absent", evidence="не работаем")
    assert e.status is None


def test_parse_extraction_partial():
    facts, statuses, rej = parse_extraction({"facts": [
        {"section": "company_profile", "attribute": "regime", "value": "УСН", "evidence": "усн"},
        {"section": "company_profile", "attribute": "regime", "value": "ЕНВД", "evidence": "енвд"},
    ], "section_statuses": []})
    assert len(facts) == 1
    assert len(rej) == 1


def test_parse_extraction_bad_json():
    facts, statuses, rej = parse_extraction("not json")
    assert facts == [] and rej and rej[0].kind == "output"


def test_section_status_sections():
    assert SectionStatusEvent(section="counterparties", status="confirmed_absent", evidence="нет").section == "counterparties"
    with pytest.raises(ValidationError):
        SectionStatusEvent(section="issues", status="confirmed_absent", evidence="нет")
