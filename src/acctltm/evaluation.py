"""Оценка по слоям на golden set (без GPU: эталонные ответы extractor-а).

Прогоняет эталонные ответы через write-path и read-path и сравнивает итоговое
состояние памяти и выборку контекста с разметкой. Метрики P/R/F1 считаются по
ключам фактов (раздел, субъект, ключ, статус). LLM-шаги оцениваются отдельно
при наличии модели (см. notebooks/02_llm_pipeline.ipynb).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .devcheck import check_state, dialog_text, fact_tuples, load_jsonl
from .memory import MemoryWriter
from .normalize import Normalizer
from .retrieval import ContextBuilder
from .schemas import QueryType
from .store import MemoryStore


class Clock:
    def __init__(self, start: datetime | None = None):
        self.now = start or datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, days: float = 0) -> None:
        self.now += timedelta(days=days)


def prf(expected: set, actual: set) -> dict[str, float]:
    tp = len(expected & actual)
    p = tp / len(actual) if actual else (1.0 if not expected else 0.0)
    r = tp / len(expected) if expected else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "expected": len(expected), "actual": len(actual), "tp": tp}


def run_scenario(scenario: dict, normalizer: Normalizer | None = None,
                 clock: Clock | None = None) -> dict[str, Any]:
    norm = normalizer or Normalizer()
    clock = clock or Clock()
    store = MemoryStore()
    writer = MemoryWriter(store, norm, clock)
    builder = ContextBuilder(store, norm, clock)
    user = "u-gold"
    errors: list[str] = []
    mem_exp: set = set()
    mem_act: set = set()
    for step in scenario["steps"]:
        clock.advance(step.get("after_days", 0))
        if step["type"] == "session":
            if step.get("setup"):
                writer.write(user, step["setup"], scenario["id"] + "-setup",
                             dialog_text=step.get("setup_text", ""))
            writer.write(user, step["gold_extraction"], scenario["id"], dialog_text=dialog_text(step["turns"]))
            errs = check_state(store, user, step.get("expect", {}))
            errors += [f"{scenario['id']}: {e}" for e in errs]
            exp = {tuple(x) for x in step.get("expect", {}).get("facts", [])}
            sections = {x[0] for x in exp}
            mem_exp |= exp
            mem_act |= fact_tuples(store, user, sections or None)
        elif step["type"] == "probe":
            qts = [QueryType(t) for t in step["query_types"]]
            bundle = builder.build(user, qts, step.get("subject", "self"))
            for section, keys in step.get("expect_keys", {}).items():
                actual = bundle.keys(section)
                for k in keys:
                    if k not in actual:
                        errors.append(f"{scenario['id']}: в контексте {section} нет ключа {k}")
                for k in actual - set(keys):
                    errors.append(f"{scenario['id']}: лишний ключ в контексте {section}: {k}")
    return {"id": scenario["id"], "errors": errors, "memory_expected": mem_exp, "memory_actual": mem_act}


def run_golden(scenarios: list[dict], normalizer: Normalizer | None = None) -> dict[str, Any]:
    norm = normalizer or Normalizer()
    results = [run_scenario(s, norm) for s in scenarios]
    exp: set = set()
    act: set = set()
    for r in results:
        exp |= r["memory_expected"]
        act |= r["memory_actual"]
    clean = sum(1 for r in results if not r["errors"])
    return {
        "total": len(scenarios),
        "clean": clean,
        "memory": prf(exp, act),
        "errors": [e for r in results for e in r["errors"]],
        "scenarios_without_errors": [r["id"] for r in results if not r["errors"]],
    }


def eval_query_classifier(queries: list[dict], classifier: Callable[[str], Any] | None = None) -> dict[str, Any]:
    from .query_rules import classify_query_rules
    clf = classifier or classify_query_rules
    correct = 0
    per_type: dict[str, list[int]] = {}
    misses = []
    for q in queries:
        pred = clf(q["text"])
        pred_types = {t.value if hasattr(t, "value") else t for t in pred.query_types}
        gold = set(q["query_types"])
        ok = pred_types == gold
        correct += ok
        for t in gold:
            per_type.setdefault(t, [0, 0])
            per_type[t][1] += 1
            per_type[t][0] += ok
        if not ok:
            misses.append({"id": q.get("id"), "text": q["text"], "gold": sorted(gold), "pred": sorted(pred_types)})
    return {"total": len(queries), "correct": correct,
            "accuracy": round(correct / len(queries), 3) if queries else 0.0,
            "per_type": {t: {"correct": c, "total": n} for t, (c, n) in per_type.items()},
            "misses": misses}
