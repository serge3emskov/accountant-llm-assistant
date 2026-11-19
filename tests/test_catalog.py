from __future__ import annotations

from acctltm import AccountingCatalog


def test_resolve_operation():
    cat = AccountingCatalog()
    assert cat.resolve("вычет ндс").canonical == "ндс к вычету"
    assert cat.resolve("вычет ндс").kind == "operation"


def test_resolve_counterparty():
    cat = AccountingCatalog()
    info = cat.resolve("Ромашка")
    assert info.kind == "counterparty"
    assert info.inn == "7701234560"


def test_resolve_account():
    cat = AccountingCatalog()
    info = cat.resolve("расчётный счёт")
    assert info.kind == "account"
    assert info.canonical == "51"


def test_mentions():
    cat = AccountingCatalog()
    found = {e.canonical for e in cat.mentions("Оплатили Ромашке наличными за товары")}
    assert "ооо ромашка" in found
    assert "наличные расчёты" in found


def test_by_inn():
    cat = AccountingCatalog()
    assert cat.counterparty_by_inn("7701234560") == "ооо ромашка"


def test_registry_loaded():
    cat = AccountingCatalog()
    assert len(cat.counterparties) > 14  # стартовый набор расширен сгенерированным реестром


def test_seed_priority():
    custom = AccountingCatalog(
        counterparties=[{"name": "ООО «Ромашка»", "inn": "7700000000", "aliases": ["ромашка"],
                         "role": "поставщик", "risk": {}}],
        load_registry=False,
    )
    assert custom.resolve("Ромашка").inn == "7700000000"


def test_risk_flags():
    cat = AccountingCatalog()
    assert cat.resolve("Альфа-Трейд").risk.get("disqualified") is True
    assert cat.resolve("Маяк").risk.get("mass_address") is True
