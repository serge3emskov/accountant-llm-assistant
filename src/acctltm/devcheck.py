"""Прогон dev-набора: эталонный ответ extractor-а через весь write-path.

Сравнивается итоговое состояние памяти (ключ после нормализации и статус после
state machine), а не дословный JSON. Так проверяются сами эталоны и код.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .memory import MemoryWriter
from .normalize import Normalizer
from .store import MemoryStore

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def dialog_text(turns: list[dict[str, str]]) -> str:
    return "\n".join(t["content"] for t in turns)


def fact_tuples(store: MemoryStore, user_id: str, sections: set[str] | None = None) -> set[tuple]:
    out = set()
    for f in store.list_facts(user_id):
        if sections is None or f.section.value in sections:
            out.add((f.section.value, f.subject, f.key, f.status))
    return out


def state_tuples(store: MemoryStore, user_id: str, sections: set[str] | None = None) -> set[tuple]:
    out = set()
    for s in store.list_section_states(user_id):
        if s.state and (sections is None or s.section.value in sections):
            out.add((s.section.value, s.subject, s.state))
    return out


def data_tuples(store: MemoryStore, user_id: str) -> set[tuple]:
    out = set()
    for f in store.list_facts(user_id):
        for k, v in f.data.items():
            out.add((f.section.value, f.subject, f.key, k, json.dumps(v, ensure_ascii=False, sort_keys=True)))
    return out


def check_state(store: MemoryStore, user_id: str, expect: dict) -> list[str]:
    """expect: {"facts": [[section, subject, key, status], ...], "states": [...], "data": [...]}."""
    exp_f = {tuple(x) for x in expect.get("facts", [])}
    exp_s = {tuple(x) for x in expect.get("states", [])}
    exp_d = {tuple(x) for x in expect.get("data", [])}
    sections = {x[0] for x in exp_f} | {x[0] for x in exp_s}
    act_f = fact_tuples(store, user_id, sections or None)
    act_s = state_tuples(store, user_id, sections or None)
    errors = []
    for m in sorted(exp_f - act_f):
        errors.append(f"нет факта: {m}")
    for e in sorted(act_f - exp_f):
        errors.append(f"лишний факт: {e}")
    for m in sorted(exp_s - act_s):
        errors.append(f"нет статуса раздела: {m}")
    for e in sorted(act_s - exp_s):
        errors.append(f"лишний статус раздела: {e}")
    if exp_d:
        act_d = data_tuples(store, user_id)
        for m in sorted(exp_d - act_d):
            errors.append(f"нет атрибута: {m}")
    return errors


def run_dialog(writer: MemoryWriter, user_id: str, dialog: dict) -> list[str]:
    if dialog.get("setup"):
        writer.write(user_id, dialog["setup"], dialog["id"] + "-setup",
                     dialog_text=dialog.get("setup_text", ""))
    writer.write(user_id, dialog["gold_extraction"], dialog["id"], dialog_text=dialog_text(dialog["turns"]))
    expect = {"facts": dialog.get("expected_facts", []), "states": dialog.get("expected_states", []),
              "data": dialog.get("expected_data", [])}
    return check_state(writer.store, user_id, expect)


def run_devset(dialogs: list[dict], normalizer: Normalizer | None = None, clock=None) -> dict[str, Any]:
    norm = normalizer or Normalizer()
    results = {}
    for d in dialogs:
        store = MemoryStore()
        writer = MemoryWriter(store, norm, clock)
        results[d["id"]] = run_dialog(writer, "u-dev", d)
    passed = sum(1 for e in results.values() if not e)
    return {"total": len(dialogs), "passed": passed, "errors": results}
