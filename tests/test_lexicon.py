from __future__ import annotations

from acctltm.lexicon import (
    LexEntry, Lexicon, clean_name, evidence_coverage, find_inn, parse_amount, valid_inn,
)
from acctltm.reference_data import inn10, inn12


def test_clean_name_quotes():
    assert clean_name("ООО «Ромашка»") == "ооо ромашка"


def test_clean_name_strips_amount():
    assert clean_name("ООО «Ромашка» 150000 руб") == "ооо ромашка"


def test_parse_amount_variants():
    assert parse_amount("счёт на 150 000 руб") == 150000
    assert parse_amount("1,5 млн") == 1500000
    assert parse_amount("150 тыс.") == 150000


def test_parse_amount_require_unit():
    assert parse_amount("договор 12", require_unit=True) is None
    assert parse_amount("150 тысяч", require_unit=True) == 150000


def test_valid_inn():
    assert valid_inn(inn10("770123456"))
    assert valid_inn(inn12("7704123456"))
    assert not valid_inn("7701234561")
    assert not valid_inn("123")


def test_find_inn():
    assert find_inn("Наш ИНН 7701234560, спасибо") == "7701234560"
    assert find_inn("без инн") is None


def test_evidence_coverage():
    assert evidence_coverage("Ромашка поставщик", "Работаем с Ромашкой, они поставщик") >= 0.5
    assert evidence_coverage("оплатили наличными", "работаем с ромашкой") < 0.5


def test_lexicon_exact_and_alias():
    lex = Lexicon([LexEntry("ооо ромашка", ("ромашка",))])
    assert lex.lookup("Ромашка").normalized == "ооо ромашка"
    assert lex.lookup("ООО Ромашка").method == "exact"


def test_lexicon_unmatched():
    lex = Lexicon([LexEntry("ооо ромашка", ("ромашка",))])
    assert lex.lookup("совершенно другое").method == "unmatched"


def test_lexicon_mentions_case():
    lex = Lexicon([LexEntry("ооо ромашка", ("ромашка",))])
    found = lex.find_mentions("оплатили Ромашке за товары")
    assert any(f.normalized == "ооо ромашка" for f in found)
