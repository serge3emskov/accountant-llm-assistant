"""Бухгалтерский справочник: операции, план счетов, отчётность и реестр контрагентов.

Отвечает на вопросы:
* какой канонический ключ у операции/контрагента («Ромашка» → ООО «Ромашка», ИНН) — ключ факта в памяти;
* к каким классам относится операция — по ним правила соответствия находят ограничения;
* какие у контрагента ИНН и риск-флаги (синтетические) — для проверки должной осмотрительности.

Стартовый набор контрагентов (reference_data.COUNTERPARTIES_SEED) имеет приоритет над
сгенерированным реестром data/synthetic/counterparties.json, поэтому ключи памяти в тестах
не меняются при подключении полного реестра.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import reference_data as ref
from .lexicon import LexEntry, Lexicon, NormResult, clean_name, fold

_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "counterparties.json"


@dataclass(frozen=True)
class EntityInfo:
    """Сущность после сопоставления со справочником."""

    raw: str
    canonical: str
    kind: str                       # operation | account | counterparty | obligation
    classes: tuple[str, ...] = ()
    matched: bool = False
    method: str = "exact"
    inn: str | None = None
    risk: dict = field(default_factory=dict)
    title: str | None = None        # название счёта учёта

    @property
    def label(self) -> str:
        if self.kind == "account":
            return f"{self.canonical} «{self.title or ''}»"
        if self.kind == "counterparty" and self.inn:
            name = self.raw.strip() or self.canonical
            return f"{name} (ИНН {self.inn})"
        raw = self.raw.strip()
        if raw and clean_name(raw) != clean_name(self.canonical):
            return f"{raw[:1].upper()}{raw[1:]} ({self.canonical})"
        return self.canonical


def _cp_entry(rec: dict) -> LexEntry:
    aliases = tuple(rec.get("aliases", ()))
    inn = rec.get("inn")
    all_aliases = (rec["name"], *aliases, *([inn] if inn else ()))
    return LexEntry(clean_name(rec["name"]), all_aliases, classes=("counterparty",), kind="counterparty")


class AccountingCatalog:
    def __init__(
        self,
        operations: Iterable[LexEntry] | None = None,
        accounts: Iterable[LexEntry] | None = None,
        obligations: Iterable[LexEntry] | None = None,
        counterparties: Iterable[dict] | None = None,
        load_registry: bool = True,
    ):
        self.operations = Lexicon(list(operations) if operations is not None else ref.OPERATIONS)
        self.accounts = Lexicon(list(accounts) if accounts is not None else ref.ACCOUNTS)
        self.obligations = Lexicon(list(obligations) if obligations is not None else ref.OBLIGATIONS)

        # реестр контрагентов: стартовый набор + сгенерированный (стартовый важнее)
        records: dict[str, dict] = {}
        if load_registry and _REGISTRY_PATH.exists():
            for rec in json.loads(_REGISTRY_PATH.read_text(encoding="utf-8")):
                records[clean_name(rec["name"])] = rec
        for rec in (counterparties if counterparties is not None else ref.COUNTERPARTIES_SEED):
            records[clean_name(rec["name"])] = rec  # стартовый набор перекрывает реестр
        self._cp_records = records
        self._cp_by_inn = {rec["inn"]: key for key, rec in records.items() if rec.get("inn")}
        entries = [_cp_entry(rec) for rec in records.values()]
        mention_keys = {clean_name(a) for rec in records.values()
                        for a in (rec["name"], *rec.get("aliases", ()))
                        if clean_name(a) not in ref.CP_MENTION_STOPWORDS}
        self.counterparties = Lexicon(entries, mention_keys=mention_keys)

    # ------------------------------------------------------------------ поиск
    def _info(self, nr: NormResult, kind: str) -> EntityInfo:
        inn = risk = None
        title = None
        if kind == "counterparty":
            rec = self._cp_records.get(nr.normalized)
            if rec:
                inn, risk = rec.get("inn"), rec.get("risk", {})
        elif kind == "account":
            title = ref.ACCOUNT_TITLES.get(nr.normalized)
        return EntityInfo(raw=nr.raw, canonical=nr.normalized, kind=kind, classes=nr.classes,
                          matched=nr.matched, method=nr.method, inn=inn, risk=risk or {}, title=title)

    def resolve(self, name: str) -> EntityInfo:
        """Сопоставляет название с любой сущностью справочника (приоритет: операция → контрагент → счёт → отчёт)."""
        for lex, kind in ((self.operations, "operation"), (self.counterparties, "counterparty"),
                          (self.accounts, "account"), (self.obligations, "obligation")):
            nr = lex.lookup(name)
            if nr.method in ("exact", "prefix"):
                return self._info(nr, kind)
        # нечёткое — только если однозначное
        for lex, kind in ((self.operations, "operation"), (self.counterparties, "counterparty")):
            nr = lex.lookup(name)
            if nr.method == "fuzzy":
                return self._info(nr, kind)
        nr = self.operations.lookup(name)
        return self._info(nr, "operation")

    def from_key(self, key: str, raw_names: Iterable[str] = ()) -> EntityInfo:
        """Восстанавливает сущность по ключу факта из памяти (для профиля проверки)."""
        if key in self._cp_records:
            rec = self._cp_records[key]
            raw = next(iter(raw_names), rec["name"])
            return EntityInfo(raw=raw, canonical=key, kind="counterparty", classes=("counterparty",),
                              matched=True, inn=rec.get("inn"), risk=rec.get("risk", {}))
        for lex, kind in ((self.operations, "operation"), (self.accounts, "account"),
                          (self.obligations, "obligation")):
            nr = lex.lookup(key)
            if nr.matched:
                return self._info(nr, kind)
        return EntityInfo(raw=next(iter(raw_names), key), canonical=key, kind="operation", matched=False,
                          method="unmatched")

    def mentions(self, text: str) -> list[EntityInfo]:
        """Операции и контрагенты, упомянутые в свободном тексте (для блока СПРАВОЧНИК и проверки)."""
        out: dict[str, EntityInfo] = {}
        for nr in self.operations.find_mentions(text):
            out.setdefault(nr.normalized, self._info(nr, "operation"))
        for nr in self.counterparties.find_mentions(text):
            out.setdefault(nr.normalized, self._info(nr, "counterparty"))
        return list(out.values())

    def counterparty_by_inn(self, inn: str) -> str | None:
        return self._cp_by_inn.get(inn)

    def lexicon(self) -> Lexicon:
        return self.operations
