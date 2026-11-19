from __future__ import annotations

from acctltm.query_rules import classify_query_rules
from acctltm.schemas import QueryType


def test_posting():
    assert QueryType.POSTING in classify_query_rules("какие проводки при покупке товаров").query_types


def test_deadline():
    assert QueryType.DEADLINE in classify_query_rules("когда сдавать декларацию по усн").query_types


def test_profile_review():
    assert QueryType.PROFILE_REVIEW in classify_query_rules("что ты знаешь о нашей компании").query_types


def test_general():
    assert classify_query_rules("добрый день").query_types == [QueryType.GENERAL]


def test_deduction():
    assert QueryType.DEDUCTION in classify_query_rules("можно ли принять ндс к вычету").query_types


def test_history():
    assert QueryType.HISTORY in classify_query_rules("с какими поставщиками мы работали раньше").query_types


def test_subject():
    assert classify_query_rules("какой режим у дочки").subject == "дочка"


def test_document_check():
    assert QueryType.DOCUMENT_CHECK in classify_query_rules("проверь контрагента ромашка").query_types
