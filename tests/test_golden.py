from __future__ import annotations

from pathlib import Path

from acctltm.devcheck import load_jsonl
from acctltm.evaluation import eval_query_classifier, run_golden

DATA = Path(__file__).resolve().parents[1] / "data"


def test_golden_clean(normalizer):
    g = run_golden(load_jsonl(DATA / "golden" / "scenarios.jsonl"), normalizer)
    assert g["clean"] == g["total"], g["errors"]
    assert g["memory"]["f1"] == 1.0


def test_golden_queries_baseline():
    qr = eval_query_classifier(load_jsonl(DATA / "golden" / "queries.jsonl"))
    assert qr["accuracy"] >= 0.6


def test_golden_has_multisession():
    scen = load_jsonl(DATA / "golden" / "scenarios.jsonl")
    assert any(len(s["steps"]) > 1 for s in scen)
