from __future__ import annotations

import importlib.util
from pathlib import Path

from acctltm.lexicon import valid_inn
from acctltm.synthetic import load_bank_statement, load_documents

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_synthetic_data.py"
spec = importlib.util.spec_from_file_location("build_synthetic_data", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_counterparties_valid():
    cps = mod.build_counterparties()
    assert len(cps) == 60
    assert all(valid_inn(c["inn"]) for c in cps)


def test_reproducible():
    assert mod.build_counterparties() == mod.build_counterparties()


def test_documents_count():
    assert len(mod.build_documents(mod.build_counterparties())) == 120


def test_bank_count():
    assert len(mod.build_bank_statement(mod.build_counterparties())) == 84


def test_loaders():
    assert len(load_documents()) == 120
    assert len(load_bank_statement()) == 84
