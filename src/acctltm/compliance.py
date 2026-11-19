"""Проверка соответствия в коде: налоговые и учётные ограничения по справочнику.

Принцип тот же, что у памяти: LLM пишет ответ, а то, что можно проверить
детерминированно, проверяет код.

* Профиль — из памяти: режим, НДС, объект УСН, численность, выручка, практики,
  действующие договоры и контрагенты. Только сведения со слов бухгалтера.
* Операция/контрагент из вопроса → классы (compliance_data/classes.json) → правила.
* Правила (compliance_data/rules.json) сопоставляют операции с профилем.

Проверка работает в двух местах:
1. До ответа: операции из вопроса проверяются, ограничения по профилю попадают в контекст.
2. После ответа: операции, которые назвала модель, проверяются ещё раз. Неучтённое
   предупреждение дописывается к ответу кодом.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .catalog import AccountingCatalog, EntityInfo
from .lexicon import fold
from .sections import ATTRIBUTE_LABELS, POLICIES, Section
from .store import MemoryStore

Level = Literal["info", "caution", "avoid", "contraindicated"]
LEVELS: tuple[str, ...] = ("info", "caution", "avoid", "contraindicated")
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}
LEVEL_LABELS = {
    "contraindicated": "не допускается",
    "avoid": "не рекомендуется без главбуха",
    "caution": "с осторожностью",
    "info": "к сведению",
}
DATA_DIR = Path(__file__).resolve().parent / "compliance_data"
EXPENSE_CLASSES = frozenset({"usn_expense", "goods", "materials", "fixed_assets", "representative", "expense"})


# =========================================================================== правила


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComplianceClass(_Model):
    id: str
    label: str
    operations: list[str] = Field(default_factory=list)


class Selector(_Model):
    classes: list[str] = Field(default_factory=list)
    operations: list[str] = Field(default_factory=list)
    any: bool = False

    @model_validator(mode="after")
    def _not_empty(self) -> "Selector":
        if not (self.classes or self.operations or self.any):
            raise ValueError("пустой селектор: нужен classes, operations или any")
        return self

    def matches(self, e: EntityInfo) -> bool:
        return self.any or bool(set(self.classes) & set(e.classes)) or e.canonical in self.operations


class Rule(_Model):
    id: str
    kind: Literal["operation_operation", "regime", "attribute", "flag", "special", "info_op"]
    operation: Selector
    other: Selector | None = None
    regimes: list[str] = Field(default_factory=list)
    usn_object: list[str] = Field(default_factory=list)
    attribute: str | None = None
    attr_from: float | None = None
    attr_below: float | None = None
    flag: str | None = None
    flag_values: list[str] = Field(default_factory=lambda: ["да", "иногда"])
    contract_amount_above: int | None = None
    states: list[str] = Field(default_factory=list)
    level: Level
    message: str
    reason: str
    source: str = ""
    replaces: list[str] = Field(default_factory=list)
    hint: bool = True

    @model_validator(mode="after")
    def _kind_fields(self) -> "Rule":
        need = {
            "operation_operation": self.other is not None,
            "regime": bool(self.regimes),
            "attribute": self.attribute is not None and (self.attr_from is not None or self.attr_below is not None),
            "flag": self.flag is not None,
            "special": bool(self.states),
            "info_op": True,
        }[self.kind]
        if not need:
            raise ValueError(f"{self.id}: не заполнены поля для правила типа {self.kind}")
        re.compile(self.reason)
        return self


class BuiltinRule(_Model):
    level: Level
    message: str
    reason: str


class RuleSet(_Model):
    classes: list[ComplianceClass]
    rules: list[Rule]
    builtin: dict[Literal["cp_mass_address", "cp_disqualified", "cp_young", "no_primary_docs"], BuiltinRule]

    @model_validator(mode="after")
    def _references(self) -> "RuleSet":
        ids = [c.id for c in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("повторяющиеся id классов")
        rule_ids = [r.id for r in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("повторяющиеся id правил")
        known = set(ids)
        for r in self.rules:
            for sel in (r.operation, r.other):
                if sel is not None and set(sel.classes) - known:
                    raise ValueError(f"{r.id}: неизвестные классы {sorted(set(sel.classes) - known)}")
            if set(r.replaces) - set(rule_ids):
                raise ValueError(f"{r.id}: replaces ссылается на неизвестные правила")
        return self

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "RuleSet":
        d = Path(directory) if directory else DATA_DIR
        classes = json.loads((d / "classes.json").read_text(encoding="utf-8"))
        rules = json.loads((d / "rules.json").read_text(encoding="utf-8"))
        return cls(classes=classes["classes"], rules=rules["rules"], builtin=rules["builtin"])

    def class_label(self, class_id: str) -> str:
        return next((c.label for c in self.classes if c.id == class_id), class_id)


# =========================================================================== профиль


@dataclass
class Profile:
    subject: str = "self"
    regime: str | None = None
    vat_payer: str | None = None
    usn_object: str | None = None
    employees: int | None = None
    revenue_year: int | None = None
    vat_exempt: str | None = None
    practices: dict[str, str] = field(default_factory=dict)
    counterparties: list[EntityInfo] = field(default_factory=list)
    contracts: list[dict] = field(default_factory=list)
    issues: set[str] = field(default_factory=set)
    missing_docs: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject, "regime": self.regime, "vat_payer": self.vat_payer,
            "usn_object": self.usn_object, "employees": self.employees, "revenue_year": self.revenue_year,
            "vat_exempt": self.vat_exempt, "practices": self.practices,
            "counterparties": [c.label for c in self.counterparties],
            "contracts": self.contracts, "issues": sorted(self.issues), "missing_docs": self.missing_docs,
        }


# =========================================================================== результат


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


@dataclass(frozen=True)
class Alert:
    rule_id: str
    level: str
    entity: EntityInfo
    trigger_kind: str
    trigger: str
    message: str
    reason: str
    other: EntityInfo | None = None
    order: int = 0

    def render(self) -> str:
        level = LEVEL_LABELS[self.level]
        if self.other is not None:
            return _cap(f"{self.entity.label} + {self.other.label} — {level}: {self.message}.")
        return _cap(f"{self.entity.label} — {level} ({self.trigger}): {self.message}.")

    def as_dict(self) -> dict[str, Any]:
        return {"rule": self.rule_id, "level": self.level, "entity": self.entity.canonical,
                "entity_raw": self.entity.raw, "trigger": self.trigger, "trigger_kind": self.trigger_kind,
                "other": self.other.canonical if self.other else None, "text": self.render()}


@dataclass(frozen=True)
class Restriction:
    rule_id: str
    level: str
    target: str
    trigger: str
    message: str

    def render(self) -> str:
        return _cap(f"{self.target} — {LEVEL_LABELS[self.level]} ({self.trigger}): {self.message}")

    def as_dict(self) -> dict[str, Any]:
        return {"rule": self.rule_id, "level": self.level, "target": self.target,
                "trigger": self.trigger, "text": self.render()}


COMPLIANCE_HEADER = ("### ПРОВЕРКА ПО СПРАВОЧНИКУ (выполнена кодом по нормам НК РФ и учётным правилам). "
                     "Не предлагай операции с пометкой «не допускается» или «не рекомендуется без главбуха»; "
                     "если бухгалтер спрашивает о такой операции, прямо назови риск и причину")


@dataclass
class ComplianceReport:
    profile: Profile
    question_entities: list[EntityInfo] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    restrictions: list[Restriction] = field(default_factory=list)

    def render(self) -> list[str]:
        if not self.alerts and not self.restrictions:
            return []
        lines = ["", COMPLIANCE_HEADER]
        if self.alerts:
            lines.append("Операции и контрагенты из вопроса:")
            lines += [f"- {a.render()}" for a in self.alerts]
        if self.restrictions:
            lines.append("Ограничения по профилю компании:")
            lines += [f"- {r.render()}" for r in self.restrictions]
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {"profile": self.profile.as_dict(),
                "question_entities": [e.label for e in self.question_entities],
                "alerts": [a.as_dict() for a in self.alerts],
                "restrictions": [r.as_dict() for r in self.restrictions]}


@dataclass
class PostCheck:
    raw: str
    text: str
    answer_entities: list[EntityInfo]
    alerts: list[Alert]
    appended: list[Alert]
    covered: list[Alert]

    def as_dict(self) -> dict[str, Any]:
        return {"answer_entities": [e.label for e in self.answer_entities],
                "alerts": [a.as_dict() for a in self.alerts],
                "appended": [a.as_dict() for a in self.appended],
                "covered": [a.as_dict() for a in self.covered]}


# =========================================================================== проверка


class ComplianceChecker:
    def __init__(self, catalog: AccountingCatalog | None = None, rules: RuleSet | None = None):
        self.catalog = catalog or AccountingCatalog()
        self.rules = rules or RuleSet.load()
        self._classes = {c.id: c for c in self.rules.classes}
        self._replaces = {r.id: set(r.replaces) for r in self.rules.rules if r.replaces}

    def mentions(self, text: str) -> list[EntityInfo]:
        return self.catalog.mentions(text)

    # ------------------------------------------------------------------ профиль
    def profile(self, store: MemoryStore, user_id: str, subject: str = "self",
                now: datetime | None = None) -> Profile:
        now = now or datetime.now(timezone.utc)
        prof = Profile(subject=subject)
        for f in store.list_facts(user_id, subject):
            policy = POLICIES[f.section]
            if f.section == Section.COMPANY_PROFILE:
                v = f.data.get("value")
                if f.key == "regime":
                    prof.regime = str(v)
                elif f.key == "vat_payer":
                    prof.vat_payer = str(v)
                elif f.key == "usn_object":
                    prof.usn_object = str(v)
                elif f.key == "employees" and isinstance(v, (int, float)):
                    prof.employees = int(v)
                elif f.key == "revenue_year" and isinstance(v, (int, float)):
                    prof.revenue_year = int(v)
                elif f.key == "vat_exempt":
                    prof.vat_exempt = str(v)
            elif f.section == Section.PRACTICES and f.data.get("value"):
                prof.practices[f.key] = str(f.data["value"])
            elif f.section == Section.COUNTERPARTIES and f.status in policy.active_statuses:
                prof.counterparties.append(self.catalog.from_key(f.key, f.raw_names))
            elif f.section == Section.CONTRACTS and f.status in policy.active_statuses:
                prof.contracts.append({"key": f.key, "amount": f.data.get("amount"),
                                       "payment": f.data.get("payment")})
            elif f.section == Section.DOCUMENTS and f.status == "missing":
                prof.missing_docs = True
            elif f.section == Section.ISSUES and f.status == "active":
                prof.issues.add(f.key)
        return prof

    # ------------------------------------------------------------------ правила
    def _attr_value(self, prof: Profile, attr: str) -> float | None:
        v = getattr(prof, attr, None)
        return float(v) if isinstance(v, (int, float)) else None

    def _profile_trigger(self, r: Rule, prof: Profile) -> str | None:
        """Текст причины, если профиль сам по себе попадает под правило (для ограничений до ответа)."""
        if r.kind == "regime":
            if prof.regime in r.regimes and (not r.usn_object or prof.usn_object in r.usn_object):
                obj = f", объект «{prof.usn_object}»" if prof.usn_object else ""
                return f"режим {prof.regime}{obj}"
        elif r.kind == "attribute":
            if r.regimes and prof.regime not in r.regimes:
                return None
            val = self._attr_value(prof, r.attribute or "")
            if val is None:
                return None
            if (r.attr_from is not None and val >= r.attr_from) or (r.attr_below is not None and val < r.attr_below):
                return f"{ATTRIBUTE_LABELS.get(r.attribute, r.attribute)} = {int(val)}"
        elif r.kind == "flag":
            if prof.practices.get(r.flag or "") in r.flag_values:
                if r.contract_amount_above is not None and not any(
                        (c.get("amount") or 0) > r.contract_amount_above for c in prof.contracts):
                    return None
                return f"{ATTRIBUTE_LABELS.get(r.flag, r.flag)}: {prof.practices.get(r.flag)}"
        elif r.kind == "special":
            if "vat_exempt" in r.states and prof.vat_exempt == "да":
                return "освобождение от НДС (ст. 145 НК РФ)"
        return None

    def _entity_alerts(self, e: EntityInfo, prof: Profile, entities: Sequence[EntityInfo]) -> list[Alert]:
        out: list[Alert] = []
        for i, r in enumerate(self.rules.rules):
            if not r.operation.matches(e):
                continue
            if r.kind == "info_op":
                out.append(Alert(r.id, r.level, e, "operation", "операция", r.message, r.reason, order=i))
            elif r.kind == "operation_operation":
                assert r.other is not None
                for o in entities:
                    if o is not e and r.other.matches(o):
                        out.append(Alert(r.id, r.level, e, "operation_pair", o.label, r.message, r.reason,
                                         other=o, order=i))
            else:
                trig = self._profile_trigger(r, prof)
                if trig:
                    out.append(Alert(r.id, r.level, e, r.kind, trig, r.message, r.reason, order=i))
        # встроенные проверки контрагентов и первички
        b = self.rules.builtin
        if e.kind == "counterparty":
            for flag, key in (("mass_address", "cp_mass_address"), ("disqualified", "cp_disqualified"),
                              ("young", "cp_young")):
                if e.risk.get(flag):
                    br = b[key]
                    out.append(Alert(key, br.level, e, "counterparty_risk", flag, br.message, br.reason, order=-1))
        if EXPENSE_CLASSES & set(e.classes) and prof.missing_docs:
            br = b["no_primary_docs"]
            out.append(Alert("no_primary_docs", br.level, e, "documents", "нет первичных документов",
                             br.message, br.reason, order=-1))
        return self._dedup(out)

    def _dedup(self, alerts: list[Alert]) -> list[Alert]:
        replaced: set[str] = set()
        for a in alerts:
            replaced |= self._replaces.get(a.rule_id, set())
        seen: set[tuple] = set()
        out = []
        for a in sorted(alerts, key=lambda x: (x.order, -LEVEL_RANK[x.level])):
            if a.rule_id in replaced:
                continue
            k = (a.rule_id, a.entity.canonical, a.other.canonical if a.other else None)
            if k not in seen:
                seen.add(k)
                out.append(a)
        return out

    # ------------------------------------------------------------------ до ответа
    def report(self, prof: Profile, question_text: str | None, with_restrictions: bool = True) -> ComplianceReport:
        entities = self.mentions(question_text) if question_text else []
        alerts: list[Alert] = []
        for e in entities:
            alerts += self._entity_alerts(e, prof, entities)
        restrictions: list[Restriction] = []
        if with_restrictions:
            for r in self.rules.rules:
                if not r.hint or r.kind in ("operation_operation", "info_op"):
                    continue
                trig = self._profile_trigger(r, prof)
                if trig:
                    target = self.rules.class_label(r.operation.classes[0]) if r.operation.classes else \
                        (r.operation.operations[0] if r.operation.operations else "операции")
                    restrictions.append(Restriction(r.id, r.level, target, trig, r.message))
        return ComplianceReport(prof, entities, self._dedup(alerts), restrictions)

    # ------------------------------------------------------------------ после ответа
    def post_check(self, answer: str, prof: Profile, question_entities: Sequence[EntityInfo]) -> PostCheck:
        entities = self.mentions(answer)
        alerts: list[Alert] = []
        for e in entities:
            alerts += self._entity_alerts(e, prof, list(entities) + list(question_entities))
        alerts = self._dedup(alerts)
        folded = fold(answer)
        covered, appended = [], []
        for a in alerts:
            if re.search(a.reason, folded):
                covered.append(a)
            else:
                appended.append(a)
        text = answer.rstrip()
        if appended:
            lines = ["", "⚠ Проверка по справочнику:"] + [f"- {a.render()}" for a in appended]
            text = text + "\n" + "\n".join(lines)
        return PostCheck(answer, text, entities, alerts, appended, covered)
