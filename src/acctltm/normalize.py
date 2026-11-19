"""Нормализация названий по разделам памяти.

LLM передаёт название так, как сказал пользователь («Ромашка», «ООО "Ромашка"»),
а канонический ключ факта определяет код по справочнику. Поэтому «Ромашка»
и «ООО Ромашка» в разных диалогах — один и тот же контрагент.

Договоры, документы и проблемы — свободные ключи (имя в начальной форме),
для них справочник не требуется.
"""
from __future__ import annotations

from .catalog import AccountingCatalog
from .lexicon import Lexicon, NormResult, clean_name, fold, parse_amount
from .sections import Section
from . import reference_data as ref


class Normalizer:
    def __init__(self, catalog: AccountingCatalog | None = None,
                 subjects: dict[str, tuple[str, ...]] | None = None):
        self.catalog = catalog or AccountingCatalog()
        self.counterparties = self.catalog.counterparties
        self.obligations = self.catalog.obligations
        self._subjects: dict[str, str] = {}
        for canonical, aliases in (subjects or ref.SUBJECTS).items():
            for alias in (canonical, *aliases):
                self._subjects[fold(alias)] = canonical

    def lexicon_for(self, section: Section) -> Lexicon | None:
        return {
            Section.COUNTERPARTIES: self.counterparties,
            Section.OBLIGATIONS: self.obligations,
        }.get(section)

    def entity(self, section: Section, name: str) -> NormResult:
        lexicon = self.lexicon_for(section)
        if lexicon is None:  # contracts / documents / issues: свободный ключ
            key = clean_name(name) or fold(name)
            return NormResult(raw=name, cleaned=key, normalized=key, method="exact",
                              amount=parse_amount(name, require_unit=True))
        return lexicon.lookup(name)

    @classmethod
    def from_catalog(cls, catalog: AccountingCatalog) -> "Normalizer":
        return cls(catalog=catalog)

    def operation_mentions(self, text: str) -> list[NormResult]:
        return self.catalog.operations.find_mentions(text)

    def counterparty_mentions(self, text: str) -> list[NormResult]:
        return self.catalog.counterparties.find_mentions(text)

    def subject(self, raw: str | None) -> str:
        if not raw:
            return "self"
        key = fold(raw)
        if key in self._subjects:
            return self._subjects[key]
        cleaned = clean_name(raw)
        if cleaned in self._subjects:
            return self._subjects[cleaned]
        return cleaned or key
