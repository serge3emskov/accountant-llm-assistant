"""Read-path: тип запроса → набор разделов → факты с подходящими статусами → контекст.

Без semantic top-K и без LLM-reranking: выборка предсказуема и тестируется
как обычная функция. Сюда же вынесены правила «забывания» (срок актуальности
по типу факта) и список того, что ассистенту нужно уточнить у бухгалтера.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Literal

from .catalog import EntityInfo
from .compliance import ComplianceChecker, ComplianceReport
from .lexicon import NormResult
from .normalize import Normalizer
from . import reference_data as ref
from .schemas import QueryType
from .sections import (ATTRIBUTE_LABELS, DATA_LABELS, POLICIES, FactKind, Section, StaleAction, status_label)
from .store import Fact, MemoryStore, SectionState

S = Section

QUERY_SECTIONS: dict[QueryType, tuple[Section, ...]] = {
    QueryType.POSTING: (S.COMPANY_PROFILE, S.COUNTERPARTIES, S.CONTRACTS, S.DOCUMENTS, S.PRACTICES),
    QueryType.TAX: (S.COMPANY_PROFILE, S.OBLIGATIONS, S.PRACTICES),
    QueryType.DOCUMENT_CHECK: (S.COUNTERPARTIES, S.CONTRACTS, S.DOCUMENTS, S.COMPANY_PROFILE),
    QueryType.DEDUCTION: (S.COMPANY_PROFILE, S.DOCUMENTS, S.CONTRACTS, S.PRACTICES),
    QueryType.DEADLINE: (S.OBLIGATIONS, S.COMPANY_PROFILE),
    QueryType.HISTORY: (S.COUNTERPARTIES, S.CONTRACTS, S.OBLIGATIONS),
    QueryType.PROFILE_REVIEW: tuple(Section),
    QueryType.GENERAL: (),
}

# Разделы, без которых рекомендацию давать рано: если о них ничего не известно — уточнить
CLARIFY_SECTIONS: dict[QueryType, tuple[Section, ...]] = {
    QueryType.POSTING: (S.COMPANY_PROFILE,),
    QueryType.DEDUCTION: (S.COMPANY_PROFILE,),
    QueryType.DOCUMENT_CHECK: (S.COUNTERPARTIES,),
    QueryType.TAX: (S.COMPANY_PROFILE,),
}

# Обязательные атрибуты профиля для запроса
REQUIRED_ATTRIBUTES: dict[QueryType, tuple[str, ...]] = {
    QueryType.POSTING: ("regime",),
    QueryType.DEDUCTION: ("regime",),
    QueryType.TAX: ("regime",),
}

# Запросы, после которых ассистент может предложить операцию: для них в контекст идут ограничения по профилю
RECOMMENDING_QUERIES: frozenset[QueryType] = frozenset({
    QueryType.POSTING, QueryType.TAX, QueryType.DOCUMENT_CHECK, QueryType.DEDUCTION,
})

# Запросы, для которых нужны и неактуальные записи
HISTORY_SECTIONS: dict[QueryType, tuple[Section, ...]] = {
    QueryType.HISTORY: (S.COUNTERPARTIES, S.CONTRACTS, S.OBLIGATIONS),
}

# Класс операции → тема налоговой справки для блока СПРАВОЧНИК
_CLASS_TOPIC: dict[str, tuple[str, ...]] = {
    "vat_deduction": ("ндс",), "vat_charge": ("ндс",), "vat_doc": ("ндс",),
    "usn_expense": ("усн", "лимиты усн"), "payroll": ("ндфл", "взносы"), "payroll_tax": ("ндфл", "взносы"),
    "dividends": ("ндфл",), "cash": ("наличные",), "retail": ("наличные",),
    "advance_in": ("ндс", "енп"), "advance_out": ("ндс",), "import": ("ндс",), "export": ("ндс",),
    "gph": ("ндфл", "взносы"), "selfemployed": ("ндп",), "penalties": ("усн",),
}

ClarifyReason = Literal["unknown", "conflict", "stale", "missing_attribute"]


@dataclass(frozen=True)
class ClarifyItem:
    section: Section
    reason: ClarifyReason
    key: str | None = None

    def render(self) -> str:
        title = POLICIES[self.section].title
        what = f"{title} → {ATTRIBUTE_LABELS.get(self.key, self.key)}" if self.key else title
        reason = {
            "unknown": "нет данных",
            "conflict": "противоречие: бухгалтер сказал «нет», но в памяти есть актуальные записи",
            "stale": "давно не подтверждалось",
            "missing_attribute": "нет данных",
        }[self.reason]
        return f"- {what}: {reason}"


@dataclass(frozen=True)
class ContextFact:
    fact: Fact
    stale: bool = False


@dataclass(frozen=True)
class DocumentRecord:
    """Документированный источник: запись синтетической выписки банка / реестра документов."""

    date: str
    doc_type: str
    counterparty_raw: str
    amount: int = 0


def select_sections(query_types: Iterable[QueryType]) -> list[Section]:
    out: list[Section] = []
    for qt in query_types:
        for s in QUERY_SECTIONS[qt]:
            if s not in out:
                out.append(s)
    return out


def statuses_for(section: Section, query_types: Iterable[QueryType]) -> set[str]:
    policy = POLICIES[section]
    qts = set(query_types)
    if QueryType.PROFILE_REVIEW in qts:
        return set(policy.statuses)
    allowed = set(policy.active_statuses)
    if policy.include_denied:
        allowed.add("denied")
    for qt in qts:
        if section in HISTORY_SECTIONS.get(qt, ()):
            allowed |= {s for s in policy.inactive_statuses if s != "denied"}
    return allowed


@dataclass
class ContextBundle:
    user_id: str
    subject: str
    query_types: list[QueryType]
    sections: list[Section]
    facts: dict[Section, list[ContextFact]] = field(default_factory=dict)
    section_states: dict[Section, SectionState] = field(default_factory=dict)
    clarify: list[ClarifyItem] = field(default_factory=list)
    documents: list[tuple[DocumentRecord, str]] = field(default_factory=list)
    reference: list[EntityInfo] = field(default_factory=list)
    tax_topics: list[str] = field(default_factory=list)
    now: datetime | None = None
    compliance: ComplianceReport | None = None

    def keys(self, section: Section | None = None) -> set[str]:
        secs = [section] if section else list(self.facts)
        return {cf.fact.key for s in secs for cf in self.facts.get(s, [])}

    def sections_with_facts(self) -> set[Section]:
        return {s for s, items in self.facts.items() if items}

    def render(self) -> str:
        if not self.sections:
            lines = ["### ПАМЯТЬ: для этого запроса персональный контекст не нужен"]
            lines += self._render_reference()
            lines += self.compliance.render() if self.compliance else []
            return "\n".join(lines)
        subj = "своя компания" if self.subject == "self" else self.subject
        lines = [
            "### ПАМЯТЬ АССИСТЕНТА — со слов бухгалтера (документами не подтверждено)",
            f"Субъект: {subj}",
        ]
        for section in self.sections:
            block = self._render_section(section)
            if block:
                lines += block
        if self.documents:
            lines += ["", "### ДОКУМЕНТИРОВАНО — синтетическая выписка банка / реестр документов"]
            for rec, normalized in self.documents:
                lines.append(f"- {rec.date}: {rec.doc_type} «{rec.counterparty_raw}» на {rec.amount} ₽ → {normalized}")
            lines.append("Если документы расходятся со словами бухгалтера, не выбирай одну версию сам — уточни.")
        lines += self._render_reference()
        lines += self.compliance.render() if self.compliance else []
        if self.clarify:
            note = "; проверка по справочнику учла только то, что есть в памяти, и эти вопросы не заменяет" \
                if self.compliance else ""
            lines += ["", "### НУЖНО УТОЧНИТЬ ПЕРЕД РЕКОМЕНДАЦИЕЙ (этого нет в памяти; "
                          f"если бухгалтер уже сказал об этом в текущем диалоге — не переспрашивай{note})"]
            lines += [c.render() for c in self.clarify]
        return "\n".join(lines)

    def _render_reference(self) -> list[str]:
        if not self.reference and not self.tax_topics:
            return []
        lines = ["", "### СПРАВОЧНИК — операции, контрагенты и ставки (проверить актуальность на дату операции)"]
        for e in self.reference:
            if e.kind == "counterparty":
                risk = ", ".join(e.risk) if e.risk else "без риск-флагов"
                lines.append(f"- «{e.raw}» → {e.canonical}, ИНН {e.inn}; риски: {risk}")
            elif e.kind == "account":
                lines.append(f"- «{e.raw}» → счёт {e.canonical} «{e.title}»")
            else:
                lines.append(f"- «{e.raw}» → {e.canonical}")
        for topic in self.tax_topics:
            if topic in ref.TAX_REFERENCE:
                lines.append(f"- {ref.TAX_REFERENCE[topic]}")
        return lines

    def _render_section(self, section: Section) -> list[str]:
        policy = POLICIES[section]
        items = self.facts.get(section, [])
        state = self.section_states.get(section)
        lines: list[str] = []
        if state and state.state == "confirmed_absent" and not items:
            lines.append("- подтверждено: нет")
        elif state and state.state == "dont_know":
            lines.append("- бухгалтер сказал, что не знает")
        for cf in items:
            lines.append(_render_fact(cf, policy.kind))
        if not lines:
            return []
        return ["", f"#### {policy.title}", *lines]


def _render_fact(cf: ContextFact, kind: FactKind) -> str:
    f = cf.fact
    stale = " ⚠ давно не подтверждалось" if cf.stale else ""
    confirmed = f.last_confirmed_at.date().isoformat()
    if kind == FactKind.ATTRIBUTE:
        label = ATTRIBUTE_LABELS.get(f.key, f.key)
        details = f" ({f.data['details']})" if f.data.get("details") else ""
        return f"- {label}: {f.data.get('value')}{details} [на {confirmed}]{stale}"
    if f.status == "denied":
        return f"- {f.key}: НЕТ (подтверждено отсутствие) [на {confirmed}]{stale}"
    parts = [f"- {f.key} [{status_label(f.section, f.status)}]"]
    attrs = {k: v for k, v in f.data.items() if v not in (None, "")}
    if attrs:
        parts.append("; ".join(f"{DATA_LABELS.get(k, k)}: {v}" for k, v in attrs.items()))
    if f.raw_names and [n.lower() for n in f.raw_names] != [f.key.lower()]:
        parts.append("со слов: " + ", ".join(f"«{n}»" for n in f.raw_names))
    if f.norm.get("method") == "unmatched":
        parts.append("название не найдено в справочнике")
    elif f.norm.get("method") == "fuzzy":
        parts.append("название распознано с опечаткой")
    parts.append(f"подтверждено {confirmed}")
    return " — ".join(parts) + stale


class ContextBuilder:
    def __init__(
        self,
        store: MemoryStore,
        normalizer: Normalizer | None = None,
        clock: Callable[[], datetime] | None = None,
        compliance: ComplianceChecker | None = None,
    ):
        self.store = store
        self.normalizer = normalizer or Normalizer()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.compliance = compliance

    def is_stale(self, fact: Fact, now: datetime) -> bool:
        ttl = POLICIES[fact.section].ttl_for(fact.key)
        return ttl is not None and now - fact.last_confirmed_at > timedelta(days=ttl)

    def build(
        self,
        user_id: str,
        query_types: Iterable[QueryType],
        subject: str = "self",
        documents: Iterable[DocumentRecord] | None = None,
        reference_text: str | None = None,
    ) -> ContextBundle:
        qts = list(dict.fromkeys(query_types)) or [QueryType.GENERAL]
        subject = self.normalizer.subject(subject)
        now = self.clock()
        sections = select_sections(qts)
        profile = QueryType.PROFILE_REVIEW in qts
        bundle = ContextBundle(user_id=user_id, subject=subject, query_types=qts, sections=sections, now=now)

        all_facts = self.store.list_facts(user_id, subject, sections)
        company = {f.key: f.data.get("value")
                   for f in self.store.list_facts(user_id, subject, [S.COMPANY_PROFILE])}

        for section in sections:
            policy = POLICIES[section]
            allowed = statuses_for(section, qts)
            selected: list[ContextFact] = []
            for f in all_facts:
                if f.section != section or f.status not in allowed:
                    continue
                stale = self.is_stale(f, now)
                if stale and policy.on_stale == StaleAction.DROP and not profile:
                    continue
                selected.append(ContextFact(f, stale))
                if stale and policy.on_stale == StaleAction.FLAG and not profile:
                    bundle.clarify.append(ClarifyItem(section, "stale", f.key))
            bundle.facts[section] = selected
            if policy.absence_aware:
                bundle.section_states[section] = self.store.get_section_state(user_id, subject, section)

        if not profile:
            self._add_clarifications(bundle, qts, all_facts, company)

        if documents and (S.COUNTERPARTIES in sections or S.CONTRACTS in sections):
            for rec in documents:
                nr = self.normalizer.entity(S.COUNTERPARTIES, rec.counterparty_raw)
                label = nr.normalized if nr.matched else f"{nr.normalized} (не найдено в справочнике)"
                bundle.documents.append((rec, label))
        if reference_text:
            bundle.reference = self.normalizer.catalog.mentions(reference_text)
            topics: list[str] = []
            for e in bundle.reference:
                for cls in e.classes:
                    for t in _CLASS_TOPIC.get(cls, ()):
                        if t not in topics:
                            topics.append(t)
            bundle.tax_topics = topics
        if self.compliance is not None:
            prof = self.compliance.profile(self.store, user_id, subject, now)
            bundle.compliance = self.compliance.report(
                prof, reference_text, with_restrictions=bool(RECOMMENDING_QUERIES & set(qts)))
        return bundle

    @staticmethod
    def _section_known(section: Section, state: SectionState | None, facts: list[Fact]) -> bool:
        if state is not None and state.state is not None:
            return True
        policy = POLICIES[section]
        known = set(policy.active_statuses)
        if policy.denied_item_answers_section:
            known.add("denied")
        return any(f.section == section and f.status in known for f in facts)

    def _add_clarifications(
        self, bundle: ContextBundle, qts: list[QueryType], all_facts: list[Fact], company: dict
    ) -> None:
        seen = {(c.section, c.key) for c in bundle.clarify}

        def add(item: ClarifyItem) -> None:
            if (item.section, item.key) not in seen:
                seen.add((item.section, item.key))
                bundle.clarify.append(item)

        for qt in qts:
            for section in CLARIFY_SECTIONS.get(qt, ()):
                state = bundle.section_states.get(section)
                if state and state.needs_clarification:
                    add(ClarifyItem(section, "conflict"))
                    continue
                if not self._section_known(section, state, all_facts):
                    add(ClarifyItem(section, "unknown"))
            for attr in REQUIRED_ATTRIBUTES.get(qt, ()):
                if attr not in company:
                    add(ClarifyItem(S.COMPANY_PROFILE, "missing_attribute", attr))

        for section, state in bundle.section_states.items():
            if state.needs_clarification:
                add(ClarifyItem(section, "conflict"))
