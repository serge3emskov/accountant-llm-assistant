from __future__ import annotations

from pathlib import Path

from acctltm import ComplianceChecker, MemoryStore, MemoryWriter
from acctltm.devcheck import load_jsonl

CASES = Path(__file__).resolve().parents[1] / "data" / "compliance" / "cases.jsonl"


def _build(case, norm):
    store = MemoryStore()
    writer = MemoryWriter(store, norm)
    setup = case["setup"]
    dt = " ".join(f["evidence"] for f in setup["facts"]) + " " + \
         " ".join(s["evidence"] for s in setup["section_statuses"])
    writer.write("u", setup, case["id"], dialog_text=dt)
    return store


def test_compliance_cases(normalizer):
    chk = ComplianceChecker(normalizer.catalog)
    for c in load_jsonl(CASES):
        store = _build(c, normalizer)
        prof = chk.profile(store, "u", c["subject"])
        rep = chk.report(prof, c["question"])
        actual = {a.rule_id for a in rep.alerts}
        assert set(c["expect_rules"]) <= actual, (c["id"], sorted(actual))


def _usn_profile(norm):
    store = MemoryStore()
    w = MemoryWriter(store, norm)
    w.write("u", {"facts": [
        {"section": "company_profile", "attribute": "regime", "value": "УСН", "evidence": "усн"},
        {"section": "company_profile", "attribute": "usn_object", "value": "доходы", "evidence": "доходы"},
    ], "section_statuses": []}, "d", dialog_text="усн доходы")
    return ComplianceChecker(norm.catalog).profile(store, "u")


def test_post_check_appends(normalizer):
    chk = ComplianceChecker(normalizer.catalog)
    prof = _usn_profile(normalizer)
    post = chk.post_check("Можно принять расходы на аренду.", prof, [])
    assert post.appended
    assert "Проверка по справочнику" in post.text


def test_post_check_covered(normalizer):
    chk = ComplianceChecker(normalizer.catalog)
    prof = _usn_profile(normalizer)
    post = chk.post_check("При объекте доходы расходы не уменьшают налог.", prof, [])
    assert any(a.rule_id == "r-usn-income-exp" for a in post.covered)
    assert not any(a.rule_id == "r-usn-income-exp" for a in post.appended)


def test_restrictions(normalizer):
    chk = ComplianceChecker(normalizer.catalog)
    prof = _usn_profile(normalizer)
    rep = chk.report(prof, "что можно учесть?", with_restrictions=True)
    assert any(r.rule_id == "r-usn-income-exp" for r in rep.restrictions)


def test_ruleset_loads():
    from acctltm.compliance import RuleSet
    rs = RuleSet.load()
    assert len(rs.rules) >= 20
    assert len(rs.classes) >= 20
