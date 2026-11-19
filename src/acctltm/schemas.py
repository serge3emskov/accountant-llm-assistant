"""Контракт между LLM и кодом.

Extractor (LLM) возвращает JSON. Каждое событие валидируется отдельно:
невалидное событие в память не попадает, но и не блокирует остальные.
В схемах нет ID, времени и типа изменения — их определяет код.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from .lexicon import parse_amount, valid_inn

ShortText = Annotated[str, Field(min_length=1, max_length=120)]
Evidence = Annotated[str, Field(min_length=1, max_length=500)]
Polarity = Literal["positive", "negative"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, coerce_numbers_to_str=True)

    @model_validator(mode="before")
    @classmethod
    def _null_means_absent(cls, data: Any) -> Any:
        """null и пустая строка от модели означают «поле не указано»: срабатывают значения по умолчанию."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None and not (isinstance(v, str) and not v.strip())}
        return data


# --------------------------------------------------------------------------- факты


def _literal_values(annotation: Any) -> set:
    values: set = set()
    for arg in get_args(annotation) or (annotation,):
        if get_origin(arg) is Literal:
            values |= set(get_args(arg))
    return values


class _EntityEvent(_Strict):
    subject: ShortText = "self"   # о какой компании: своя, дочка, ИП директора...
    name: ShortText               # как сказал пользователь, без нормализации
    polarity: Polarity = "positive"
    evidence: Evidence            # фрагмент реплики, из которого извлечён факт

    @model_validator(mode="before")
    @classmethod
    def _negation_drops_bad_status(cls, data: Any) -> Any:
        """«С Ромашкой не работаем» со статусом "confirmed_absent": смысл несёт отрицание,
        лишний статус отбрасываем. Для утверждения недопустимый статус по-прежнему ошибка."""
        if isinstance(data, dict) and data.get("polarity") == "negative" and "status" in cls.model_fields:
            if data.get("status") is not None and data["status"] not in _literal_values(
                    cls.model_fields["status"].annotation):
                data = {k: v for k, v in data.items() if k != "status"}
        return data


class CounterpartyEvent(_EntityEvent):
    section: Literal["counterparties"]
    status: Literal["active", "paused", "terminated"] | None = None
    role: Literal["покупатель", "поставщик", "подрядчик", "агент", "прочее"] | None = None
    inn: ShortText | None = None
    terms: ShortText | None = None

    @field_validator("inn")
    @classmethod
    def _check_inn(cls, v: str | None) -> str | None:
        if v is not None:
            digits = re.sub(r"\D", "", v)
            if not valid_inn(digits):
                raise ValueError(f"ИНН не прошёл проверку контрольных цифр: {v!r}")
            return digits
        return v


class ContractEvent(_EntityEvent):
    section: Literal["contracts"]
    status: Literal["planned", "active", "paused", "terminated"] | None = None
    amount: int | float | str | None = None
    subject: ShortText | None = None
    payment: Literal["аванс", "постоплата", "смешанная"] | None = None
    number: ShortText | None = None

    @field_validator("amount", mode="before")
    @classmethod
    def _parse_amount(cls, v: Any) -> Any:
        if v is None or isinstance(v, (int, float)):
            return v
        parsed = parse_amount(str(v))
        if parsed is None:
            raise ValueError(f"сумма не распознана: {v!r}")
        return parsed


class DocumentEvent(_EntityEvent):
    section: Literal["documents"]
    status: Literal["expected", "received", "posted"] | None = None
    amount: int | float | str | None = None
    number: ShortText | None = None
    counterparty: ShortText | None = None

    @field_validator("amount", mode="before")
    @classmethod
    def _parse_amount(cls, v: Any) -> Any:
        if v is None or isinstance(v, (int, float)):
            return v
        parsed = parse_amount(str(v))
        if parsed is None:
            raise ValueError(f"сумма не распознана: {v!r}")
        return parsed


class ObligationEvent(_EntityEvent):
    section: Literal["obligations"]
    status: Literal["upcoming", "overdue", "filed"] | None = None
    deadline: ShortText | None = None
    period: ShortText | None = None


class IssueEvent(_EntityEvent):
    section: Literal["issues"]
    status: Literal["active", "resolved"] | None = None
    details: ShortText | None = None


_REGIME_SYNONYMS = {
    "усн": "УСН", "упрощёнка": "УСН", "упрощенка": "УСН", "упрощённая": "УСН", "упрощенная": "УСН",
    "осно": "ОСНО", "общий": "ОСНО", "общая": "ОСНО", "осн": "ОСНО",
    "псн": "ПСН", "патент": "ПСН",
    "аусн": "АУСН", "автоматизированная упрощёнка": "АУСН",
    "ндп": "НПД", "нпд": "НПД", "самозанятость": "НПД", "самозанятый": "НПД",
    "есхн": "ЕСХН",
}
_USN_OBJECT_SYNONYMS = {
    "доходы": "доходы", "6%": "доходы", "6 процентов": "доходы", "доход": "доходы",
    "доходы-расходы": "доходы-расходы", "доходы минус расходы": "доходы-расходы",
    "15%": "доходы-расходы", "15 процентов": "доходы-расходы", "доход-расход": "доходы-расходы",
}
_YES_NO = {"да": "да", "нет": "нет", "есть": "да", "является": "да", "платим": "да",
           "не платим": "нет", "не является": "нет", "освобождены": "нет"}


class CompanyProfileEvent(_Strict):
    section: Literal["company_profile"]
    subject: ShortText = "self"
    attribute: Literal["regime", "vat_payer", "usn_object", "employees", "revenue_year", "vat_exempt"]
    value: int | float | str
    evidence: Evidence

    @model_validator(mode="after")
    def _check_value(self) -> "CompanyProfileEvent":
        raw = str(self.value).strip()
        key = raw.lower().replace("ё", "е")
        if self.attribute == "regime":
            canon = _REGIME_SYNONYMS.get(key) or _REGIME_SYNONYMS.get(key.replace(" ", ""))
            if canon is None:
                raise ValueError(f"недопустимый режим: {self.value!r}")
            self.value = canon
        elif self.attribute in ("vat_payer", "vat_exempt"):
            yn = _YES_NO.get(key)
            if yn is None:
                raise ValueError(f"{self.attribute}: ожидается да/нет, получено {self.value!r}")
            self.value = yn
        elif self.attribute == "usn_object":
            canon = _USN_OBJECT_SYNONYMS.get(key)
            if canon is None:
                raise ValueError(f"недопустимый объект УСН: {self.value!r}")
            self.value = canon
        elif self.attribute == "employees":
            num = self.value if isinstance(self.value, (int, float)) else None
            if num is None:
                m = re.search(r"\d+", raw)
                if not m:
                    raise ValueError(f"численность: ожидается число, получено {self.value!r}")
                num = float(m.group())
            if not 0 <= num <= 1_000_000:
                raise ValueError(f"численность вне диапазона: {num}")
            self.value = int(num)
        else:  # revenue_year
            parsed = self.value if isinstance(self.value, (int, float)) else parse_amount(raw)
            if parsed is None or parsed < 0:
                raise ValueError(f"выручка не распознана: {self.value!r}")
            self.value = int(parsed)
        return self


class PracticesEvent(_Strict):
    section: Literal["practices"]
    subject: ShortText = "self"
    attribute: Literal["cash_settlements", "kkt", "edo"]
    value: Literal["да", "нет", "иногда"]
    details: ShortText | None = None
    evidence: Evidence


FactEvent = Annotated[
    Union[
        CounterpartyEvent,
        ContractEvent,
        DocumentEvent,
        ObligationEvent,
        IssueEvent,
        CompanyProfileEvent,
        PracticesEvent,
    ],
    Field(discriminator="section"),
]

EntityEvent = Union[CounterpartyEvent, ContractEvent, DocumentEvent, ObligationEvent, IssueEvent]
AttributeEvent = Union[CompanyProfileEvent, PracticesEvent]

# Поля события, которые становятся атрибутами факта
FACT_DATA_FIELDS: dict[str, tuple[str, ...]] = {
    "counterparties": ("role", "inn", "terms"),
    "contracts": ("amount", "subject", "payment", "number"),
    "documents": ("amount", "number", "counterparty"),
    "obligations": ("deadline", "period"),
    "issues": ("details",),
    "company_profile": ("value",),
    "practices": ("value", "details"),
}


# ---------------------------------------------------------- статусы разделов


class SectionStatusEvent(_Strict):
    """«Контрагентов нет» / «не знаю» — сведения о разделе целиком."""

    section: Literal["counterparties", "contracts", "obligations"]
    subject: ShortText = "self"
    status: Literal["confirmed_absent", "dont_know"]
    evidence: Evidence


# ------------------------------------------------------------ разбор ответа LLM


class Rejection(BaseModel):
    kind: Literal["fact", "section_status", "output"]
    raw: Any
    errors: list[str]


_FACT_ADAPTER = TypeAdapter(FactEvent)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _format_errors(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        out.append(f"{loc}: {err['msg']}")
    return out


def parse_extraction(
    raw: str | dict,
) -> tuple[list[Any], list[SectionStatusEvent], list[Rejection]]:
    """Разбирает ответ extractor-а: `{"facts": [...], "section_statuses": [...]}`.

    Возвращает валидные факты, валидные статусы разделов и отклонённые элементы.
    Ошибка в одном элементе не отменяет остальные.
    """
    if isinstance(raw, str):
        text = _FENCE_RE.sub("", raw.strip())
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            return [], [], [Rejection(kind="output", raw=text[:2000], errors=[f"JSON: {exc}"])]
    if not isinstance(raw, dict):
        return [], [], [Rejection(kind="output", raw=raw, errors=["ожидается JSON-объект"])]

    unknown_keys = set(raw) - {"facts", "section_statuses"}
    rejections: list[Rejection] = []
    if unknown_keys:
        rejections.append(
            Rejection(kind="output", raw=sorted(unknown_keys), errors=["неизвестные ключи верхнего уровня"])
        )

    facts: list[Any] = []
    for item in raw.get("facts") or []:
        try:
            facts.append(_FACT_ADAPTER.validate_python(item))
        except ValidationError as exc:
            rejections.append(Rejection(kind="fact", raw=item, errors=_format_errors(exc)))

    statuses: list[SectionStatusEvent] = []
    for item in raw.get("section_statuses") or []:
        try:
            statuses.append(SectionStatusEvent.model_validate(item))
        except ValidationError as exc:
            rejections.append(Rejection(kind="section_status", raw=item, errors=_format_errors(exc)))

    return facts, statuses, rejections


# ------------------------------------------------------------ классификация запроса


class QueryType(str, Enum):
    POSTING = "posting"                    # как отразить операцию, проводки, счета
    TAX = "tax"                            # налоги, взносы, ставки, расчёт
    DOCUMENT_CHECK = "document_check"      # проверка документа, контрагента, счёта-фактуры
    DEDUCTION = "deduction"                # можно ли принять расход / вычет
    DEADLINE = "deadline"                  # сроки сдачи отчётности и уплаты
    HISTORY = "history"                    # что было раньше: договоры, контрагенты, отчёты
    PROFILE_REVIEW = "profile_review"      # что ассистент знает о компании
    GENERAL = "general"                    # без персонального контекста


class QueryClassification(_Strict):
    """Контракт classifier-а запроса."""

    query_types: list[QueryType] = Field(min_length=1)
    subject: ShortText = "self"
