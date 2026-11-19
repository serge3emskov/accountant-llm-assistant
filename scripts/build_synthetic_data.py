"""Генератор синтетических данных для демо ассистента бухгалтера.

Создаёт детерминированный (seed=42) набор:
* data/synthetic/counterparties.json — реестр контрагентов с валидными ИНН и риск-флагами;
* data/synthetic/documents.jsonl — реестр первичных документов и счетов-фактур;
* data/synthetic/bank_statement.csv — выписка банка (платёжные поручения).

Все данные вымышленные. ИНН генерируются с корректными контрольными цифрами.
Запуск: python scripts/build_synthetic_data.py
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acctltm.lexicon import valid_inn  # noqa: E402
from acctltm.reference_data import COUNTERPARTIES_SEED, inn10, inn12  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic"
SEED = 42
N_COUNTERPARTIES = 60

_PREFIX = ["Техно", "Строй", "Торг", "Альфа", "Бета", "Гамма", "Дельта", "Омега", "Вектор", "Гранд",
           "Союз", "Мастер", "Проф", "Регион", "Сфера", "Импульс", "Альянс", "Вершина", "Горизонт", "Компас",
           "Нева", "Волга", "Ока", "Заря", "Кристалл", "Орион", "Сатурн", "Меркурий", "Юпитер", "Нептун"]
_SUFFIX = ["Сервис", "Трейд", "Логистик", "Софт", "Маркет", "Плюс", "Групп", "Холдинг", "Снаб", "Комплект",
           "Системы", "Ресурс", "Опт", "Центр", "Альянс", "Проект", "Инвест", "Стандарт", "Формат", "Актив"]
_SURNAMES = ["Кузнецов", "Смирнов", "Иванов", "Петров", "Соколов", "Морозов", "Волков", "Зайцев", "Павлов",
             "Семёнов", "Голубев", "Виноградов", "Богданов", "Воробьёв", "Фёдоров", "Михайлов", "Беляев",
             "Тарасов", "Белов", "Комаров"]
_INITIALS = ["А.А.", "Б.В.", "В.Г.", "Г.Д.", "Д.Е.", "Е.Ж.", "Ж.З.", "З.И.", "И.К.", "К.Л."]
_ROLES = ["поставщик", "покупатель", "подрядчик", "агент"]
_DOC_TYPES = ["счёт на оплату", "акт выполненных работ", "товарная накладная", "счёт-фактура", "УПД"]


def _risk(rng: random.Random) -> dict:
    r = {}
    if rng.random() < 0.15:
        r["mass_address"] = True
    if rng.random() < 0.08:
        r["disqualified"] = True
    if rng.random() < 0.12:
        r["young"] = True
    return r


def build_counterparties(n: int = N_COUNTERPARTIES, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    records = [dict(rec) for rec in COUNTERPARTIES_SEED]
    seen_names = {rec["name"] for rec in records}
    seen_inn = {rec["inn"] for rec in records}
    used_bases: set[str] = set()
    while len(records) < n:
        is_ip = rng.random() < 0.25
        if is_ip:
            name = f"ИП {rng.choice(_SURNAMES)} {rng.choice(_INITIALS)}"
            base = "".join(str(rng.randint(0, 9)) for _ in range(10))
            inn = inn12(base)
            aliases = (name.split()[1],)
        else:
            name = f"ООО «{rng.choice(_PREFIX)}-{rng.choice(_SUFFIX)}»"
            base = "".join(str(rng.randint(0, 9)) for _ in range(9))
            inn = inn10(base)
            aliases = (name[4:-1].lower(),)  # без «ООО «...»»
        if name in seen_names or inn in seen_inn or base in used_bases:
            continue
        used_bases.add(base)
        seen_names.add(name)
        seen_inn.add(inn)
        assert valid_inn(inn)
        records.append({"name": name, "inn": inn, "aliases": list(aliases),
                        "role": rng.choice(_ROLES), "risk": _risk(rng)})
    return records


def build_documents(counterparties: list[dict], seed: int = SEED) -> list[dict]:
    rng = random.Random(seed + 1)
    docs = []
    names = [rec["name"] for rec in counterparties]
    for month in (1, 2, 3):
        for i in range(40):
            cp = rng.choice(names)
            day = rng.randint(1, 28)
            docs.append({
                "date": f"2026-{month:02d}-{day:02d}",
                "doc_type": rng.choice(_DOC_TYPES),
                "counterparty": cp,
                "number": f"{month}-{i + 1:03d}",
                "amount": rng.choice([15000, 48000, 120000, 250000, 90000, 340000, 76000]),
                "status": rng.choice(["received", "posted", "expected"]),
            })
    docs.sort(key=lambda d: d["date"])
    return docs


def build_bank_statement(counterparties: list[dict], seed: int = SEED) -> list[dict]:
    rng = random.Random(seed + 2)
    rows = []
    names = [rec["name"] for rec in counterparties]
    for month in (1, 2, 3):
        for i in range(28):
            cp = rng.choice(names)
            day = rng.randint(1, 28)
            direction = rng.choice(["incoming", "outgoing"])
            rows.append({
                "date": f"2026-{month:02d}-{day:02d}",
                "counterparty": cp,
                "direction": direction,
                "amount": rng.choice([15000, 48000, 120000, 250000, 90000, 340000, 76000, 100000, 150000]),
                "purpose": rng.choice(["оплата по договору", "аванс", "оплата за товары", "оплата услуг",
                                       "возврат займа", "пополнение счёта"]),
            })
    rows.sort(key=lambda r: r["date"])
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cps = build_counterparties()
    docs = build_documents(cps)
    bank = build_bank_statement(cps)
    (OUT / "counterparties.json").write_text(json.dumps(cps, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "documents.jsonl").open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    with (OUT / "bank_statement.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "counterparty", "direction", "amount", "purpose"])
        w.writeheader()
        w.writerows(bank)
    print(f"counterparties: {len(cps)} (valid INN: {sum(valid_inn(c['inn']) for c in cps)})")
    print(f"documents: {len(docs)}")
    print(f"bank_statement: {len(bank)}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
