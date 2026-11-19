"""Сопоставление свободной формулировки со справочником.

Порядок: точное совпадение → самый длинный префикс по словам →
нечёткое совпадение (опечатки). Нечёткое совпадение принимается,
только если лучший кандидат заметно лучше второго: при неоднозначности
код честно возвращает «не распознано», а не угадывает контрагента.

Бухгалтерская специфика: из названия убираются суммы и кавычки
(«ООО "Ромашка"» → «ооо ромашка»), суммы распознаются в тексте
(«150 тыс. руб» → 150000), ИНН проверяется по контрольным цифрам.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Literal

from rapidfuzz import fuzz, process

_NUM = r"\d+(?:[\s\u00a0]\d{3})*(?:[.,]\d+)?"
AMOUNT_UNIT_RE = re.compile(rf"{_NUM}\s*(?:млрд|млн|тыс\w*|руб\w*|коп\w*|₽|р\.)", re.IGNORECASE)
AMOUNT_RE = re.compile(rf"({_NUM})\s*(млрд|млн|тыс\w*|руб\w*|коп\w*|₽|р\.)?", re.IGNORECASE)
_MULT = (("млрд", 1_000_000_000), ("млн", 1_000_000), ("тыс", 1_000))
_NON_WORD = re.compile(r"[^a-zа-я0-9+ ]+")
_SPACES = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-zа-я0-9]+")
_NO_NUM = re.compile(r"№\s*\d+")


def fold(text: str) -> str:
    return text.lower().replace("ё", "е").strip()


def evidence_coverage(evidence: str, text: str, min_token_len: int = 3, token_match: float = 85.0) -> float:
    """Доля слов цитаты, найденных в тексте диалога (с допуском на словоформу).

    Сравнение по словам, а не подстрокой: модель может сократить цитату
    («работаем с Ромашкой» из «работаем с Ромашкой и Васильком») — это не выдумка.
    А «оплатили наличными» в диалоге, где о наличных нет ни слова, — выдумка.
    """
    words = [w for w in _TOKEN.findall(fold(evidence)) if len(w) >= min_token_len]
    if not words:
        return 1.0
    vocab = set(_TOKEN.findall(fold(text)))
    found = 0
    for w in words:
        if w in vocab or any(abs(len(v) - len(w)) <= 3 and fuzz.ratio(w, v) >= token_match for v in vocab):
            found += 1
    return found / len(words)


def parse_amount(text: str, require_unit: bool = False) -> int | None:
    """«счёт на 150 000 руб» → 150000; «1,5 млн» → 1500000; «150 тыс.» → 150000.

    require_unit=True — для неявного извлечения из названия: «договор № 12» не сумма.
    """
    m = AMOUNT_RE.search(fold(text))
    if not m:
        return None
    unit = (m.group(2) or "").lower()
    if require_unit and not unit:
        return None
    num = float(m.group(1).replace(" ", "").replace("\u00a0", "").replace(",", "."))
    for prefix, mult in _MULT:
        if unit.startswith(prefix):
            return int(round(num * mult))
    return int(round(num))


def clean_name(text: str) -> str:
    """Убирает кавычки, суммы, номера и пунктуацию; организационную форму сохраняет."""
    s = fold(text)
    s = re.sub(r"[«»\"“”„']", " ", s)
    s = AMOUNT_UNIT_RE.sub(" ", s)
    s = _NO_NUM.sub(" ", s)
    s = s.replace("-", " ").replace("/", " ")
    s = _NON_WORD.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


# --------------------------------------------------------------------------- ИНН


def valid_inn(value: str) -> bool:
    """Контрольные цифры ИНН (10 знаков — организация, 12 — ИП/физлицо)."""
    s = value.strip()
    if not s.isdigit():
        return False
    if len(s) == 10:
        w = (2, 4, 10, 3, 5, 9, 4, 6, 8)
        return sum(int(a) * b for a, b in zip(s, w)) % 11 % 10 == int(s[9])
    if len(s) == 12:
        w1 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        w2 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        n1 = sum(int(a) * b for a, b in zip(s, w1)) % 11 % 10
        n2 = sum(int(a) * b for a, b in zip(s, w2)) % 11 % 10
        return n1 == int(s[10]) and n2 == int(s[11])
    return False


INN_RE = re.compile(r"\b\d{12}\b|\b\d{10}\b")


def find_inn(text: str) -> str | None:
    return next((m.group() for m in INN_RE.finditer(text) if valid_inn(m.group())), None)


# --------------------------------------------------------------------------- справочник

Kind = Literal["operation", "account", "counterparty", "obligation"]


@dataclass(frozen=True)
class LexEntry:
    canonical: str
    aliases: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()
    kind: Kind = "operation"


Method = Literal["exact", "prefix", "fuzzy", "unmatched"]


@dataclass(frozen=True)
class NormResult:
    raw: str
    cleaned: str
    normalized: str
    method: Method
    score: float = 100.0
    components: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()
    amount: int | None = None
    candidates: tuple[str, ...] = field(default_factory=tuple)
    kind: Kind = "operation"

    @property
    def matched(self) -> bool:
        return self.method != "unmatched"

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "score": round(self.score, 1),
            "components": list(self.components),
            "classes": list(self.classes),
            "candidates": list(self.candidates),
            "kind": self.kind,
        }


class Lexicon:
    def __init__(
        self,
        entries: Iterable[LexEntry],
        fuzzy_cutoff: float = 85.0,
        min_fuzzy_len: int = 5,
        ambiguity_margin: float = 3.0,
        mention_keys: Iterable[str] | None = None,
    ):
        """mention_keys — ключи, которые ищутся в свободном тексте (find_mentions).

        По умолчанию — все. Большому реестру контрагентов передаётся отфильтрованный набор:
        среди названий есть обычные слова («Маяк», «Прогресс»), и в речи они не всегда контрагент.
        """
        self.fuzzy_cutoff = fuzzy_cutoff
        self.min_fuzzy_len = min_fuzzy_len
        self.ambiguity_margin = ambiguity_margin
        self._index: dict[str, LexEntry] = {}
        for entry in entries:
            for alias in (entry.canonical, *entry.aliases):
                key = clean_name(alias)
                if key and key not in self._index:  # первая запись в списке имеет приоритет
                    self._index[key] = entry
        self._keys = list(self._index)
        self._mention_keys = None if mention_keys is None else frozenset(mention_keys) & self._index.keys()

    def __len__(self) -> int:
        return len(self._index)

    def _result(self, raw: str, cleaned: str, entry: LexEntry, method: Method, score: float) -> NormResult:
        components = entry.components or (entry.canonical,)
        return NormResult(
            raw=raw, cleaned=cleaned, normalized=entry.canonical, method=method, score=score,
            components=components, classes=entry.classes, amount=parse_amount(raw, require_unit=True),
            kind=entry.kind,
        )

    def _fuzzy(self, query: str) -> tuple[LexEntry | None, float, tuple[str, ...]]:
        if len(query) < self.min_fuzzy_len:
            return None, 0.0, ()
        hits = process.extract(query, self._keys, scorer=fuzz.ratio, limit=8, score_cutoff=self.fuzzy_cutoff)
        best: dict[str, tuple[float, LexEntry]] = {}
        for key, score, _ in hits:
            entry = self._index[key]
            if entry.canonical not in best or score > best[entry.canonical][0]:
                best[entry.canonical] = (score, entry)
        ranked = sorted(best.values(), key=lambda x: -x[0])
        if not ranked:
            return None, 0.0, ()
        names = tuple(e.canonical for _, e in ranked)
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.ambiguity_margin:
            return None, ranked[0][0], names  # неоднозначно — не угадываем
        return ranked[0][1], ranked[0][0], names

    def lookup(self, raw: str) -> NormResult:
        cleaned = clean_name(raw)
        if not cleaned:
            return NormResult(raw=raw, cleaned="", normalized=fold(raw), method="unmatched", score=0.0,
                              amount=parse_amount(raw, require_unit=True))
        if cleaned in self._index:
            return self._result(raw, cleaned, self._index[cleaned], "exact", 100.0)

        tokens = cleaned.split()
        for n in range(len(tokens) - 1, 0, -1):
            prefix = " ".join(tokens[:n])
            if prefix in self._index:
                return self._result(raw, cleaned, self._index[prefix], "prefix", 100.0)

        candidates: tuple[str, ...] = ()
        for query in dict.fromkeys((cleaned, tokens[0])):
            entry, score, names = self._fuzzy(query)
            candidates = candidates or names
            if entry is not None:
                return replace(self._result(raw, cleaned, entry, "fuzzy", score), candidates=names)

        return NormResult(raw=raw, cleaned=cleaned, normalized=cleaned, method="unmatched", score=0.0,
                          amount=parse_amount(raw, require_unit=True), candidates=candidates)

    # Окончания, которые отбрасываем при поиске упоминаний: «Ромашке» → «ромашк», «Васильком» → «васильк»
    _ENDINGS = ("ами", "ями", "ом", "ем", "ой", "ей", "ов", "ам", "ах", "а", "у", "е", "ы", "и", "ю")

    def find_mentions(self, text: str, min_len: int = 4) -> list[NormResult]:
        """Упоминания записей справочника в свободном тексте (только точные совпадения и падежные окончания).

        Нечёткий поиск здесь не используется: в обычной речи слишком много похожих слов.
        Записи короче min_len пропускаются (счета учёта «51» в свободном тексте не ищем).
        """
        words = _NON_WORD.sub(" ", fold(text).replace("-", " ")).split()
        found: dict[str, NormResult] = {}
        i = 0
        while i < len(words):
            hit, step = None, 1
            for n in (3, 2, 1):
                chunk = words[i:i + n]
                if len(chunk) < n:
                    continue
                last = chunk[-1]
                stems = [last[: -len(e)] for e in self._ENDINGS if last.endswith(e)]
                variants = [last, *stems, *(st + "а" for st in stems), *(st + "я" for st in stems)]
                for v in variants:
                    gram = " ".join(chunk[:-1] + [v])
                    if v and len(gram) >= min_len and gram in self._index and (
                            self._mention_keys is None or gram in self._mention_keys):
                        hit, step = gram, n
                        break
                if hit:
                    break
            if hit is not None:
                entry = self._index[hit]
                if entry.canonical not in found:
                    raw = " ".join(words[i:i + step])
                    found[entry.canonical] = self._result(raw, hit, entry, "exact", 100.0)
            i += step
        return list(found.values())
