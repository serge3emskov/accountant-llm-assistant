"""Baseline-классификатор запроса на регулярных выражениях.

Нужен, чтобы read-path работал без GPU, и как нижняя планка для
сравнения с LLM-классификатором на этапе оценки.
"""
from __future__ import annotations

import re

from .schemas import QueryClassification, QueryType

_RULES: list[tuple[QueryType, re.Pattern]] = [
    (QueryType.PROFILE_REVIEW, re.compile(
        r"что ты (\w+ ){0,2}(знаешь|помнишь|запомнил)|профиль (компании|фирмы)|что (у тебя )?записано о (нас|компании)")),
    (QueryType.HISTORY, re.compile(
        r"раньше|прежде|до этого|в прошлом|что (у нас )?было|сдавали ли|работали ли")),
    (QueryType.DEADLINE, re.compile(
        r"когда (сда|плат|уплат|перечисл)|срок|до какого (числа|дня)|дедлайн|когда отчит")),
    (QueryType.DEDUCTION, re.compile(
        r"можно ли (принять|учесть)|принять (расход|ндс)|к вычету|учесть расход|можно ли учесть|вычет|принимается ли")),
    (QueryType.DOCUMENT_CHECK, re.compile(
        r"провер|контрагент|фактур|накладн|первичк|надежн|однодневк|должная осмотрительн")),
    (QueryType.TAX, re.compile(
        r"ставк|налог|взнос|ндфл|ндс|усн|как рассчит|сколько (налог|платить|взнос)|рассчитать налог")),
    (QueryType.POSTING, re.compile(
        r"проводк|как (отраз|провести|учесть)|счет учета|счёт учета|дебет|кредит|корреспонденц|каким счетом")),
]

_SUBJECT_RE: list[tuple[str, re.Pattern]] = [
    ("дочка", re.compile(r"дочк|дочерн")),
    ("ип директора", re.compile(r"ип (директора|учредителя|руководителя|владельца)")),
    ("филиал", re.compile(r"филиал|обособлен")),
]


def classify_query_rules(text: str) -> QueryClassification:
    t = text.lower().replace("ё", "е")
    types = [qt for qt, rx in _RULES if rx.search(t)]
    subject = next((s for s, rx in _SUBJECT_RE if rx.search(t)), "self")
    return QueryClassification(query_types=types or [QueryType.GENERAL], subject=subject)
