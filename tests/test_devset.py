from __future__ import annotations

from pathlib import Path

from acctltm.devcheck import load_jsonl, run_devset

DATA = Path(__file__).resolve().parents[1] / "data"


def test_devset_100(normalizer):
    res = run_devset(load_jsonl(DATA / "dev_dialogs.jsonl"), normalizer)
    assert res["passed"] == res["total"], res["errors"]


def test_devset_nonempty(normalizer):
    dialogs = load_jsonl(DATA / "dev_dialogs.jsonl")
    assert len(dialogs) >= 10
