"""Предметные разделы долгосрочной памяти ассистента бухгалтера и их политики.

Политика раздела — это всё, что код знает о разделе без LLM:
какие статусы допустимы, что считается «актуальным», различается ли
«нет данных» и «подтверждено отсутствие», сколько живёт факт и что
делать с устаревшим фактом.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Section(str, Enum):
    COMPANY_PROFILE = "company_profile"    # режим, НДС, объект УСН, численность, выручка
    COUNTERPARTIES = "counterparties"      # контрагенты: с кем работаем
    CONTRACTS = "contracts"                # договоры: суммы, условия оплаты
    DOCUMENTS = "documents"                # первичка и счета-фактуры: ожидаем/получен/проведён
    OBLIGATIONS = "obligations"            # отчётность и платежи: предстоит/сдано/просрочено
    ISSUES = "issues"                      # проблемы и расхождения: не сходится касса, нет закрывающих
    PRACTICES = "practices"                # наличные расчёты, ККТ, ЭДО


class FactKind(str, Enum):
    ENTITY = "entity"        # ключ факта — нормализованное название (контрагент, отчёт, документ)
    ATTRIBUTE = "attribute"  # ключ факта — имя атрибута (режим, численность, наличные)


class StaleAction(str, Enum):
    DROP = "drop"  # устаревший факт не возвращается в контекст (остаётся в истории)
    FLAG = "flag"  # возвращается с пометкой и попадает в список «уточнить»


# Подписи статусов для контекста основной модели
STATUS_LABELS: dict[str, str] = {
    "active": "актуально",
    "planned": "запланировано, ещё не начато",
    "paused": "приостановлено",
    "terminated": "прекращено",
    "expected": "ожидаем",
    "received": "получен, не проведён",
    "posted": "проведён",
    "missing": "отсутствует",
    "upcoming": "предстоит",
    "filed": "сдано / уплачено",
    "overdue": "просрочено",
    "resolved": "устранено",
    "denied": "подтверждено отсутствие",
}

# Уточнения подписей для отдельных разделов
SECTION_STATUS_LABELS: dict[str, dict[str, str]] = {
    "counterparties": {"active": "работаем", "denied": "не работаем"},
    "contracts": {"active": "действует", "planned": "подписан, не начат"},
    "obligations": {"upcoming": "предстоит сдать", "filed": "сдано"},
    "documents": {"expected": "ожидаем от контрагента"},
}

# Подписи атрибутов фактов
DATA_LABELS: dict[str, str] = {
    "role": "роль",
    "inn": "ИНН",
    "terms": "условия",
    "amount": "сумма, ₽",
    "subject": "предмет",
    "payment": "оплата",
    "number": "номер",
    "deadline": "срок",
    "period": "период",
    "counterparty": "контрагент",
    "details": "подробности",
    "value": "значение",
}

DATA_VALUE_LABELS: dict[str, str] = {}


def status_label(section: "Section", status: str) -> str:
    return SECTION_STATUS_LABELS.get(section.value, {}).get(status) or STATUS_LABELS.get(status, status)


@dataclass(frozen=True)
class SectionPolicy:
    section: Section
    title: str
    kind: FactKind
    statuses: tuple[str, ...]
    active_statuses: tuple[str, ...]
    default_status: str = "active"
    # Статус, в который переходит существующий факт при отрицании
    negation_status: str = "denied"
    # Различать «нет данных» и «подтверждено отсутствие» на уровне раздела
    absence_aware: bool = False
    # Показывать основной модели факты со статусом denied («с Ромашкой не работаем»)
    include_denied: bool = False
    # Срок актуальности в днях: общий (ключ "*") или по ключу факта
    ttl_days: dict[str, int] = field(default_factory=dict)
    on_stale: StaleAction = StaleAction.DROP
    # Атрибуты, допустимые для раздела (только для ATTRIBUTE)
    attributes: tuple[str, ...] = ()
    # Достаточно ли «нет» по одному пункту, чтобы раздел считался выясненным.
    # «Декларация УСН сдана» — да; «с Ромашкой не работаем» — нет, про других контрагентов неизвестно.
    denied_item_answers_section: bool = False

    @property
    def inactive_statuses(self) -> tuple[str, ...]:
        return tuple(s for s in self.statuses if s not in self.active_statuses)

    def ttl_for(self, key: str) -> int | None:
        return self.ttl_days.get(key, self.ttl_days.get("*"))


POLICIES: dict[Section, SectionPolicy] = {
    Section.COMPANY_PROFILE: SectionPolicy(
        section=Section.COMPANY_PROFILE,
        title="Профиль компании",
        kind=FactKind.ATTRIBUTE,
        statuses=("active",),
        active_statuses=("active",),
        # Выручка и численность меняются — давние значения надо переспросить
        ttl_days={"revenue_year": 365, "employees": 180},
        on_stale=StaleAction.FLAG,
        attributes=("regime", "vat_payer", "usn_object", "employees", "revenue_year", "vat_exempt"),
    ),
    Section.COUNTERPARTIES: SectionPolicy(
        section=Section.COUNTERPARTIES,
        title="Контрагенты",
        kind=FactKind.ENTITY,
        statuses=("active", "paused", "terminated", "denied"),
        active_statuses=("active", "paused"),
        negation_status="terminated",
        absence_aware=True,
        include_denied=True,
    ),
    Section.CONTRACTS: SectionPolicy(
        section=Section.CONTRACTS,
        title="Договоры",
        kind=FactKind.ENTITY,
        statuses=("planned", "active", "paused", "terminated", "denied"),
        active_statuses=("planned", "active", "paused"),
        negation_status="terminated",
        absence_aware=True,
    ),
    Section.DOCUMENTS: SectionPolicy(
        section=Section.DOCUMENTS,
        title="Первичные документы и счета-фактуры",
        kind=FactKind.ENTITY,
        statuses=("expected", "received", "posted", "missing"),
        active_statuses=("expected", "received"),
        default_status="expected",
        negation_status="missing",
    ),
    Section.OBLIGATIONS: SectionPolicy(
        section=Section.OBLIGATIONS,
        title="Отчётность и платежи",
        kind=FactKind.ENTITY,
        statuses=("upcoming", "filed", "overdue", "denied"),
        active_statuses=("upcoming", "overdue"),
        default_status="upcoming",
        negation_status="denied",
        absence_aware=True,
        include_denied=True,
        denied_item_answers_section=True,
    ),
    Section.ISSUES: SectionPolicy(
        section=Section.ISSUES,
        title="Проблемы и расхождения",
        kind=FactKind.ENTITY,
        statuses=("active", "resolved", "denied"),
        active_statuses=("active",),
        negation_status="resolved",
        include_denied=True,
        ttl_days={"*": 30},  # расхождение месячной давности уже не про текущую сверку
        on_stale=StaleAction.DROP,
    ),
    Section.PRACTICES: SectionPolicy(
        section=Section.PRACTICES,
        title="Учётные практики",
        kind=FactKind.ATTRIBUTE,
        statuses=("active",),
        active_statuses=("active",),
        ttl_days={"*": 365},
        on_stale=StaleAction.FLAG,
        attributes=("cash_settlements", "kkt", "edo"),
    ),
}

ABSENCE_AWARE_SECTIONS: tuple[Section, ...] = tuple(
    s for s, p in POLICIES.items() if p.absence_aware
)

ATTRIBUTE_LABELS: dict[str, str] = {
    "regime": "режим налогообложения",
    "vat_payer": "плательщик НДС",
    "usn_object": "объект УСН",
    "employees": "численность сотрудников",
    "revenue_year": "выручка за год, ₽",
    "vat_exempt": "освобождение от НДС (ст. 145 НК РФ)",
    "cash_settlements": "наличные расчёты",
    "kkt": "онлайн-касса (ККТ)",
    "edo": "электронный документооборот (ЭДО)",
}
