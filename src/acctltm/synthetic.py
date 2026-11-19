"""Загрузчики синтетических данных для демо и тестов."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .retrieval import DocumentRecord

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"


def load_documents(path: str | Path | None = None) -> list[DocumentRecord]:
    p = Path(path) if path else DATA_DIR / "documents.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(DocumentRecord(date=d["date"], doc_type=d["doc_type"],
                                      counterparty_raw=d["counterparty"], amount=d.get("amount", 0)))
    return out


def load_bank_statement(path: str | Path | None = None) -> list[DocumentRecord]:
    p = Path(path) if path else DATA_DIR / "bank_statement.csv"
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(DocumentRecord(date=row["date"], doc_type="платёжное поручение",
                                      counterparty_raw=row["counterparty"], amount=int(row["amount"])))
    return out
