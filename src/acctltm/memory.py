"""Write-path: валидированное событие → изменение памяти.

После extraction зона ответственности LLM заканчивается. Всё, что ниже,
детерминировано: ключ факта (через нормализацию), ID, время, тип изменения,
итоговый статус и правила для разделов с подтверждённым отсутствием.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from .lexicon import clean_name, evidence_coverage, fold
from .normalize import Normalizer
from .schemas import FACT_DATA_FIELDS, Rejection, SectionStatusEvent, parse_extraction
from .sections import POLICIES, FactKind, Section, SectionPolicy
from .store import Fact, MemoryStore, new_id

CODE_RULE_PREFIX = "код: "


class ChangeType(str, Enum):
    FIRST_MENTION = "first_mention"
    DENIAL_RECORDED = "denial_recorded"
    CONFIRMATION = "confirmation"
    UPDATE = "update"
    DEACTIVATION = "deactivation"
    NEGATION = "negation"
    REACTIVATION = "reactivation"
    SECTION_CONFIRMED_ABSENT = "section_confirmed_absent"
    SECTION_DONT_KNOW = "section_dont_know"
    SECTION_STATE_CLEARED = "section_state_cleared"
    CONFLICT = "conflict"


def classify_transition(
    policy: SectionPolicy, old_status: str | None, new_status: str, data_changed: bool
) -> ChangeType:
    active = policy.active_statuses
    if old_status is None:
        return ChangeType.DENIAL_RECORDED if new_status == "denied" else ChangeType.FIRST_MENTION
    old_active, new_active = old_status in active, new_status in active
    if old_active and new_active:
        return ChangeType.UPDATE if (data_changed or old_status != new_status) else ChangeType.CONFIRMATION
    if old_active and not new_active:
        return ChangeType.NEGATION if new_status == "denied" else ChangeType.DEACTIVATION
    if not old_active and new_active:
        return ChangeType.REACTIVATION
    return ChangeType.UPDATE if (data_changed or old_status != new_status) else ChangeType.CONFIRMATION


@dataclass
class AppliedChange:
    change_type: ChangeType
    section: Section
    subject: str
    key: str | None
    fact_id: str | None
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    event_id: str


@dataclass
class WriteReport:
    applied: list[AppliedChange] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ch in self.applied:
            out[ch.change_type.value] = out.get(ch.change_type.value, 0) + 1
        if self.rejected:
            out["rejected"] = len(self.rejected)
        return out

    def changes(self, change_type: ChangeType) -> list[AppliedChange]:
        return [c for c in self.applied if c.change_type == change_type]

    def to_rows(self) -> list[dict[str, Any]]:
        rows = [
            {"change": c.change_type.value, "section": c.section.value, "subject": c.subject,
             "key": c.key, "before": c.before, "after": c.after}
            for c in self.applied
        ]
        rows += [{"change": "rejected", "section": None, "subject": None, "key": None,
                  "before": r.raw, "after": "; ".join(r.errors)} for r in self.rejected]
        return rows


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryWriter:
    def __init__(
        self,
        store: MemoryStore,
        normalizer: Normalizer | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.store = store
        self.normalizer = normalizer or Normalizer()
        self.clock = clock or _utcnow

    def write(self, user_id: str, extraction: str | dict, dialog_id: str | None = None,
              dialog_text: str | None = None) -> WriteReport:
        facts, statuses, rejections = parse_extraction(extraction)
        facts, statuses, by_rules = self._consistency_check(facts, statuses, dialog_text)
        rejections += by_rules
        now = self.clock()
        report = WriteReport(rejected=rejections)
        with self.store.transaction():
            for rej in rejections:
                self.store.add_rejection(user_id, dialog_id, now, rej.kind, rej.raw, rej.errors)
            for ev in facts:
                report.applied += self.apply_fact(user_id, ev, dialog_id, now)
            for st in statuses:
                report.applied += self.apply_section_status(user_id, st, dialog_id, now)
        return report

    # ------------------------------------------------------------------ проверки согласованности
    GENERIC_CP_TERMS = frozenset({
        "контрагенты", "контрагент", "поставщики", "поставщик", "покупатели", "покупатель",
        "никто", "все", "ни с кем", "никакие контрагенты",
    })
    MIN_EVIDENCE_COVERAGE = 0.5
    PRACTICES_MARKERS = {
        "cash_settlements": re.compile(r"наличн|нал\b|cash"),
        "kkt": re.compile(r"касс|ккт|54-фз|фз-54|онлайн"),
        "edo": re.compile(r"эдо|документооборот|диадек|сбис|электронн"),
    }
    _CP_GUARD_SECTIONS = (Section.OBLIGATIONS, Section.ISSUES)

    @staticmethod
    def _reject(ev: Any, reason: str) -> Rejection:
        raw = ev.model_dump() if hasattr(ev, "model_dump") else ev
        return Rejection(kind="fact", raw=raw, errors=[f"{CODE_RULE_PREFIX}{reason}"])

    def _consistency_check(
        self, facts: list[Any], statuses: list[SectionStatusEvent], dialog_text: str | None
    ) -> tuple[list[Any], list[SectionStatusEvent], list[Rejection]]:
        n = self.normalizer
        kept: list[Any] = []
        statuses = list(statuses)
        rejected: list[Rejection] = []

        if dialog_text:
            grounded: list[Any] = []
            for ev in facts:
                if evidence_coverage(ev.evidence, dialog_text) < self.MIN_EVIDENCE_COVERAGE:
                    rejected.append(self._reject(ev, f"цитата «{ev.evidence}» не найдена в диалоге"))
                else:
                    grounded.append(ev)
            facts = grounded
            ok_statuses = []
            for st in statuses:
                if evidence_coverage(st.evidence, dialog_text) < self.MIN_EVIDENCE_COVERAGE:
                    rejected.append(self._reject(st, f"цитата «{st.evidence}» не найдена в диалоге"))
                else:
                    ok_statuses.append(st)
            statuses = ok_statuses

        assistant_text = fold(dialog_text or "")

        # «С Ромашкой и Васильком не работаем» пришло как «контрагентов нет вообще»:
        # если в цитате статуса названы конкретные контрагенты, это отрицание этих пунктов
        positive_keys = {(n.subject(ev.subject), n.entity(Section(ev.section), ev.name).normalized)
                         for ev in facts if getattr(ev, "name", None) and ev.polarity == "positive"}
        converted_statuses = []
        for st in statuses:
            section = Section(st.section)
            lexicon = n.lexicon_for(section)
            mentions = lexicon.find_mentions(st.evidence) if (lexicon and st.status == "confirmed_absent") else []
            mentions = [m for m in mentions if (n.subject(st.subject), m.normalized) not in positive_keys]
            if not mentions:
                converted_statuses.append(st)
                continue
            for m in mentions:
                parsed, _, _ = parse_extraction({"facts": [{
                    "section": st.section, "subject": st.subject, "name": m.normalized, "polarity": "negative",
                    "evidence": st.evidence}]})
                facts.extend(parsed)
        statuses = converted_statuses

        cp_index = {clean_name(a) for a in n.counterparties._index}

        for ev in facts:
            section = Section(ev.section)
            name = getattr(ev, "name", None)
            folded = fold(name) if name else ""
            cleaned = clean_name(name) if name else ""

            # «Контрагентов нет» — это статус раздела, а не контрагент «контрагенты»
            if (section == Section.COUNTERPARTIES and ev.polarity == "negative"
                    and (cleaned in self.GENERIC_CP_TERMS or folded.strip(" .") in self.GENERIC_CP_TERMS)):
                statuses.append(SectionStatusEvent(section="counterparties", subject=ev.subject,
                                                   status="confirmed_absent", evidence=ev.evidence))
                continue

            # Название контрагента целиком в разделе отчётности/проблем — не туда
            if section in self._CP_GUARD_SECTIONS and cleaned and cleaned in cp_index:
                rejected.append(self._reject(
                    ev, f"«{name}» — контрагент из справочника, он относится к разделу counterparties"))
                continue

            # Атрибут практики без слова-маркера: «работаем с Ромашкой» не значит «наличные расчёты»
            if section == Section.PRACTICES:
                marker = self.PRACTICES_MARKERS[ev.attribute]
                if not marker.search(fold(ev.evidence)) and not marker.search(assistant_text):
                    rejected.append(self._reject(ev, f"в цитате «{ev.evidence}» нет ничего про {ev.attribute}"))
                    continue

            kept.append(ev)
        return kept, statuses, rejected

    # ------------------------------------------------------------------ факты
    def apply_fact(self, user_id: str, ev: Any, dialog_id: str | None, now: datetime) -> list[AppliedChange]:
        section = Section(ev.section)
        policy = POLICIES[section]
        subject = self.normalizer.subject(ev.subject)
        incoming = {f: getattr(ev, f) for f in FACT_DATA_FIELDS[section.value] if getattr(ev, f) is not None}

        if policy.kind == FactKind.ATTRIBUTE:
            key, raw_name, norm, polarity, explicit_status = ev.attribute, None, {"method": "attribute"}, "positive", None
        else:
            nr = self.normalizer.entity(section, ev.name)
            key, raw_name, norm, polarity = nr.normalized, ev.name, nr.as_dict(), ev.polarity
            explicit_status = getattr(ev, "status", None)
            if (polarity == "negative" and explicit_status in policy.statuses
                    and explicit_status not in policy.active_statuses and explicit_status != "denied"):
                polarity = "positive"
            if section == Section.CONTRACTS and "amount" not in incoming and nr.amount:
                incoming["amount"] = nr.amount  # сумму из «договор на 150 тысяч» достаёт код, а не LLM

        existing = self.store.get_fact(user_id, subject, section, key)
        if existing is None:
            if polarity == "negative":
                new_status = "denied" if "denied" in policy.statuses else policy.negation_status
            else:
                new_status = explicit_status or policy.default_status
            data = {} if polarity == "negative" else incoming
            fact = Fact(
                fact_id=new_id(), user_id=user_id, subject=subject, section=section, key=key,
                status=new_status, data=data, raw_names=[raw_name] if raw_name else [], norm=norm,
                first_seen_at=now, updated_at=now, last_confirmed_at=now, mentions=1,
            )
            before = None
            change = classify_transition(policy, None, new_status, bool(data))
        else:
            fact = existing
            before = existing.snapshot()
            if polarity == "negative":
                new_status = (
                    existing.status if existing.status not in policy.active_statuses else policy.negation_status
                )
                merged = existing.data
            else:
                if explicit_status:
                    new_status = explicit_status
                elif existing.status in policy.active_statuses:
                    new_status = existing.status
                else:
                    new_status = policy.default_status
                merged = {**existing.data, **incoming}
            data_changed = merged != existing.data
            change = classify_transition(policy, existing.status, new_status, data_changed)
            fact.status, fact.data = new_status, merged
            fact.mentions += 1
            fact.last_confirmed_at = now
            if change != ChangeType.CONFIRMATION:
                fact.updated_at = now
            if raw_name and raw_name not in fact.raw_names:
                fact.raw_names.append(raw_name)

        self.store.save_fact(fact)
        after = fact.snapshot()
        event_id = self.store.add_event(
            user_id=user_id, subject=subject, section=section, fact_id=fact.fact_id,
            change_type=change.value, dialog_id=dialog_id, created_at=now,
            payload=ev.model_dump(), before=before, after=after,
        )
        changes = [AppliedChange(change, section, subject, key, fact.fact_id, before, after, event_id)]
        if policy.absence_aware:
            changes += self._sync_section_state(user_id, subject, section, fact, dialog_id, now)
        return changes

    def _sync_section_state(
        self, user_id: str, subject: str, section: Section, fact: Fact, dialog_id: str | None, now: datetime
    ) -> list[AppliedChange]:
        policy = POLICIES[section]
        st = self.store.get_section_state(user_id, subject, section)
        before = {"state": st.state, "needs_clarification": st.needs_clarification}
        if fact.status in policy.active_statuses and st.state is not None:
            st.state = None
        st.needs_clarification = False
        after = {"state": st.state, "needs_clarification": st.needs_clarification}
        if after == before:
            return []
        self.store.save_section_state(st, now)
        event_id = self.store.add_event(
            user_id=user_id, subject=subject, section=section, fact_id=None,
            change_type=ChangeType.SECTION_STATE_CLEARED.value, dialog_id=dialog_id, created_at=now,
            payload={"trigger_fact": fact.key}, before=before, after=after,
        )
        return [AppliedChange(ChangeType.SECTION_STATE_CLEARED, section, subject, None, None, before, after, event_id)]

    # ------------------------------------------------------------------ статусы разделов
    def apply_section_status(
        self, user_id: str, ev: SectionStatusEvent, dialog_id: str | None, now: datetime
    ) -> list[AppliedChange]:
        section = Section(ev.section)
        policy = POLICIES[section]
        subject = self.normalizer.subject(ev.subject)
        st = self.store.get_section_state(user_id, subject, section)
        before = {"state": st.state, "needs_clarification": st.needs_clarification}
        active_keys = [f.key for f in self.store.list_facts(user_id, subject, [section], policy.active_statuses)]

        if ev.status == "confirmed_absent":
            if active_keys:
                st.needs_clarification = True
                change = ChangeType.CONFLICT
            else:
                change = (ChangeType.CONFIRMATION if st.state == "confirmed_absent"
                          else ChangeType.SECTION_CONFIRMED_ABSENT)
                st.state, st.needs_clarification = "confirmed_absent", False
        else:
            change = ChangeType.SECTION_DONT_KNOW
            if not active_keys:
                st.state = "dont_know"

        after = {"state": st.state, "needs_clarification": st.needs_clarification}
        if active_keys:
            after["active_keys"] = active_keys
        self.store.save_section_state(st, now)
        event_id = self.store.add_event(
            user_id=user_id, subject=subject, section=section, fact_id=None, change_type=change.value,
            dialog_id=dialog_id, created_at=now, payload=ev.model_dump(), before=before, after=after,
        )
        return [AppliedChange(change, section, subject, None, None, before, after, event_id)]
