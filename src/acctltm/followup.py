"""Уточняющие вопросы после ответа.

Что нужно выяснить перед рекомендацией, решает код (блок «НУЖНО УТОЧНИТЬ» — раздел памяти без
данных, противоречие, устаревшая запись). Задал ли модель эти вопросы, тоже проверяет код:
если в ответе нет вопроса о разделе, код дописывает его к ответу.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .retrieval import ClarifyItem
from .sections import Section
from .lexicon import fold

S = Section

_QUESTIONS: dict[tuple[Section, str | None], tuple[str, str, str]] = {
    (S.COMPANY_PROFILE, "regime"): ("Какой у компании режим налогообложения?",
                                    "Какой режим налогообложения у {subject}?", r"режим|усн|осно|псн|ндп|налогообложен"),
    (S.COMPANY_PROFILE, None): ("Какой у компании режим налогообложения?",
                                "Какой режим налогообложения у {subject}?", r"режим|усн|осно|псн|ндп|налогообложен"),
    (S.COUNTERPARTIES, None): ("С какими контрагентами работаете?", "С какими контрагентами работает {subject}?",
                               r"контрагент|поставщик|покупател"),
    (S.CONTRACTS, None): ("Есть ли действующие договоры?", "Есть ли действующие договоры у {subject}?",
                          r"договор"),
    (S.OBLIGATIONS, None): ("Какая отчётность предстоит?", "Какая отчётность предстоит {subject}?",
                            r"отчет|отчёт|деклараци|сдач"),
    (S.DOCUMENTS, None): ("Все ли первичные документы получены?", "Все ли первичные документы получены?",
                          r"документ|первичк|счет-фактур|счёт-фактур|накладн|акт"),
    (S.ISSUES, None): ("Есть ли нерешённые расхождения?", "Есть ли нерешённые расхождения?",
                       r"расхожден|проблем|не сходит|сверк"),
    (S.PRACTICES, None): ("Используете ли наличные расчёты, есть ли онлайн-касса?",
                          "Использует ли {subject} наличные расчёты, есть ли онлайн-касса?",
                          r"наличн|касс|ккт|эдо"),
}

_ASK_RE = re.compile(r"\?|уточн|подскаж|скажите|сообщите|напишите|расскажите")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*")


def _question(item: ClarifyItem) -> tuple[str, str, str]:
    return _QUESTIONS.get((item.section, item.key)) or _QUESTIONS[(item.section, None)]


@dataclass
class ClarifyCheck:
    answer: str
    text: str
    asked: list[ClarifyItem] = field(default_factory=list)
    appended: list[ClarifyItem] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        def label(i: ClarifyItem) -> str:
            return i.section.value + (f".{i.key}" if i.key else "")
        return {"asked": [label(i) for i in self.asked], "appended": [label(i) for i in self.appended],
                "questions": self.questions}


def clarify_check(answer: str, items: Sequence[ClarifyItem], subject: str = "self",
                  recommended: bool = True) -> ClarifyCheck:
    """Дописывает к ответу вопросы из «НУЖНО УТОЧНИТЬ», которых в ответе нет."""
    if not recommended:
        items = []
    asks = [fold(s) for s in _SENTENCE_RE.findall(answer) if _ASK_RE.search(fold(s))]
    asked, appended, questions = [], [], []
    for item in items:
        own, other, pattern = _question(item)
        if any(re.search(pattern, s) for s in asks):
            asked.append(item)
            continue
        appended.append(item)
        q = own if subject == "self" else other.format(subject=subject)
        if q not in questions:
            questions.append(q)
    text = answer
    if questions:
        who = "" if subject == "self" else f" (для: {subject})"
        lines = [f"❓ Чтобы ответить точно, уточните{who}:"] + [f"- {q}" for q in questions]
        text = answer.rstrip() + "\n\n" + "\n".join(lines)
    return ClarifyCheck(answer, text, asked, appended, questions)
