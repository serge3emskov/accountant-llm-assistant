from __future__ import annotations

from acctltm.evaluation import Clock, eval_query_classifier, prf


def test_prf_partial():
    m = prf({"a", "b"}, {"b", "c"})
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5


def test_prf_empty():
    assert prf(set(), set())["f1"] == 1.0


def test_clock_advance():
    c = Clock()
    t0 = c()
    c.advance(5)
    assert (c() - t0).days == 5


def test_eval_classifier_perfect():
    queries = [{"text": "когда сдавать декларацию", "query_types": ["deadline"]}]
    assert eval_query_classifier(queries)["accuracy"] == 1.0
