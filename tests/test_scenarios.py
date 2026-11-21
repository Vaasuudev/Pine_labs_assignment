"""One test per failure/exception scenario in the brief, plus safety invariants."""
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from helpers import ROOT, kinds, start, to_impl  # noqa: E402
from orchestrator import service  # noqa: E402
from orchestrator.workflow import Workflow  # noqa: E402


def tree_hash(path):
    h = hashlib.sha256()
    for p in sorted(Path(path).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
            h.update(p.relative_to(path).as_posix().encode() + p.read_bytes())
    return h.hexdigest()


def test_happy_path_clean(tmp_path):
    store, wf, state = to_impl(tmp_path, "clean")
    assert state == "AWAITING_FINAL_APPROVAL"
    assert "attempt_failed" not in kinds(store)
    assert wf.ctx["rubric"]["score"] == 100


def test_generated_code_fails_tests_then_recovers(tmp_path):
    store, wf, state = to_impl(tmp_path, "fail-then-fix")
    assert state == "AWAITING_FINAL_APPROVAL"
    ev = kinds(store)
    assert ev.count("attempt_failed") == 1 and "retry" in ev
    assert wf.ctx["attempt"] == 2
    assert "FAILED" in wf.ctx["history"][0]
    # the failed attempt's code was discarded, not merged
    assert "int(gross * (1 - BULK_DISCOUNT))" not in wf._sandbox().read("shop/pricing.py")


def test_architect_developer_disagreement_resolved_by_ruling(tmp_path):
    store, wf, state = to_impl(tmp_path, "disagree")
    assert state == "AWAITING_FINAL_APPROVAL"
    assert "disagreement" in kinds(store) and "architect_ruling" in kinds(store)
    assert wf.ctx["changed"] == ["shop/pricing.py", "tests/test_bulk_discount.py"]   # cart.py untouched


def test_disagreement_deadlock_escalates_to_human(tmp_path):
    store, wf, state = to_impl(tmp_path, "deadlock")
    assert state == "ESCALATED_TO_HUMAN"
    assert kinds(store).count("disagreement") == 3          # bounded: 2 rounds + the one that trips the limit
    assert "changed" not in wf.ctx                          # nothing was ever written or reviewed
    with pytest.raises(PermissionError):
        wf.decide_final("approve", "tester")                # cannot be merged from a halted state


def test_critical_security_finding_blocks_then_is_fixed(tmp_path):
    store, wf, state = to_impl(tmp_path, "security")
    ev = [e["detail"] for e in store.events("t1") if e["kind"] == "check"]
    assert any("security_scan: FAIL" in d for d in ev)
    # unit tests PASSED on the eval version: only the security gate caught it
    assert any("attempt 1 unit_tests: PASS" in d for d in ev)
    assert state == "AWAITING_FINAL_APPROVAL"
    assert "eval(" not in wf._sandbox().read("shop/pricing.py")


def test_scope_violation_blocked_before_any_write(tmp_path):
    store, wf, state = to_impl(tmp_path, "unrelated")
    ev = [e["detail"] for e in store.events("t1") if e["kind"] == "scope_violation"]
    assert ev and "shop/config.py" in ev[0]
    assert "TAX_RATE_PERCENT = 18" in wf._sandbox().read("shop/config.py")   # untouched
    assert "shop/config.py" not in wf.ctx["changed"]
    assert state == "AWAITING_FINAL_APPROVAL"                # recovered on retry


def test_repeated_action_is_blocked_and_escalated(tmp_path):
    store, wf, state = to_impl(tmp_path, "stuck")
    assert state == "ESCALATED_TO_HUMAN"
    assert "repeat_nudge" in kinds(store) and "repeat_blocked" in kinds(store)   # one nudge, then escalate
    assert wf.ctx["attempt"] == 3


def test_malformed_output_is_repaired(tmp_path):
    store, wf, runs = start(tmp_path, "malformed")
    assert wf.advance() == "AWAITING_PLAN_APPROVAL"
    assert "malformed_output" in kinds(store)
    assert [s["status"] for s in store.steps("t1") if s["agent"] == "architect"] == ["malformed", "ok"]


def test_malformed_output_gives_up_after_bounded_repairs(tmp_path):
    store, wf, runs = start(tmp_path, "malformed-fatal")
    assert wf.advance() == "FAILED_MALFORMED_OUTPUT"
    assert kinds(store).count("malformed_output") == 3


def test_unsupported_shape_is_repaired(tmp_path):
    store, wf, state = to_impl(tmp_path, "unsupported")
    assert state == "AWAITING_FINAL_APPROVAL"
    assert any("missing key" in e["detail"] for e in store.events("t1") if e["kind"] == "malformed_output")


def test_provider_outage_pauses_then_resumes_without_rework(tmp_path):
    store, wf, state = to_impl(tmp_path, "outage")
    assert state == "PAUSED_PROVIDER_DOWN"
    assert wf.ctx["stage"] == "implement"
    analyst_calls = [s for s in store.steps("t1") if s["agent"] == "analyst"]
    # provider heals: new process, healthy provider
    wf2 = Workflow(store, "t1", tmp_path / "runs", service.make_providers("mock", "clean")[0])
    assert wf2.advance() == "AWAITING_FINAL_APPROVAL"
    assert [s for s in store.steps("t1") if s["agent"] == "analyst"] == analyst_calls   # not re-run
    assert len([s for s in store.steps("t1") if s["agent"] == "qa"]) == 1


def test_provider_failover_completes_without_pausing(tmp_path):
    store, wf, state = to_impl(tmp_path, "outage", fallback=True)
    assert state == "AWAITING_FINAL_APPROVAL"
    assert "provider_failover" in kinds(store)
    assert {s["provider"] for s in store.steps("t1")} >= {"mock", "mock-fallback"}


# ------------------------------------------------------------------ safety invariants
def test_human_gates_cannot_be_skipped(tmp_path):
    store, wf, runs = start(tmp_path, "clean")
    wf.advance()
    for fn in (wf.decide_final, wf.decide_deploy):
        with pytest.raises(PermissionError):
            fn("approve", "tester")
    wf.decide_plan("approve", "tester")
    with pytest.raises(PermissionError):
        wf.decide_deploy("approve", "tester")               # cannot deploy an unmerged change
    main_files = subprocess.run(["git", "show", "main:shop/pricing.py"], cwd=wf._sandbox().root,
                                capture_output=True, text=True).stdout
    assert "BULK_THRESHOLD" not in main_files               # main untouched until a human merges


def test_merge_only_after_human_decision(tmp_path):
    store, wf, state = to_impl(tmp_path, "clean")
    wf.decide_final("approve", "tester", "ok")
    main_files = subprocess.run(["git", "show", "main:shop/pricing.py"], cwd=wf._sandbox().root,
                                capture_output=True, text=True).stdout
    assert "BULK_THRESHOLD" in main_files
    assert wf.decide_deploy("approve", "tester") == "DEPLOYED_MOCK"
    assert [a["gate"] for a in [store.last_approval("t1", g) for g in ("plan", "merge", "deploy")]] == \
        ["plan", "merge", "deploy"]


def test_source_repo_never_modified(tmp_path):
    before = tree_hash(ROOT / "sample_repo")
    to_impl(tmp_path, "fail-then-fix")
    assert tree_hash(ROOT / "sample_repo") == before


def test_scanner_overrules_model_verdict(tmp_path):
    store, wf, state = to_impl(tmp_path, "clean")
    from orchestrator import checks
    sb = wf._sandbox()
    sb.write("developer", "shop/pricing.py", "API_KEY = 'sk-live-123456'\nx = eval('1')\n", ["shop/pricing.py"])
    res = checks.security_scan(sb.root, ["shop/pricing.py"])
    assert not res["passed"] and {f["severity"] for f in res["findings"]} == {"critical"}


def test_qa_tests_come_from_criteria_not_implementation(tmp_path):
    import json
    store, wf, state = to_impl(tmp_path, "clean")
    qa_step = next(s for s in store.steps("t1") if s["agent"] == "qa")
    inp = json.loads(qa_step["input"])
    assert set(inp) == {"acceptance_criteria", "public_api", "feedback"}      # no diff, no file contents
    assert "gross * (100" not in qa_step["input"]                            # implementation bodies never shown


def test_usage_is_tracked_for_every_call(tmp_path):
    store, wf, state = to_impl(tmp_path, "fail-then-fix")
    u = store.usage_totals("t1")
    assert u["calls"] >= 8 and u["cost_usd"] > 0 and u["tokens_in"] > 0 and u["latency_ms"] > 0


def test_lint_is_advisory_and_does_not_block(tmp_path):
    store, wf, state = to_impl(tmp_path, "lint-warn")
    assert state == "AWAITING_FINAL_APPROVAL"
    assert "attempt_failed" not in kinds(store)
    assert wf.ctx["rubric"]["score"] == 90
    review = open(wf.ctx["review_path"], encoding="utf-8").read()
    assert "WARN **lint**" in review and "advisory, does not block" in review and "unused import os" in review


def test_lint_ignores_future_imports_but_still_flags_real_unused_ones(tmp_path):
    from orchestrator import checks
    (tmp_path / "a.py").write_text("from __future__ import annotations\nimport os\n\n\ndef f(x: int) -> int:\n    return x\n")
    res = checks.run_lint(tmp_path, ["a.py"])
    assert res["advisory"] and "unused import os" in res["output"] and "annotations" not in res["output"]


def test_a_repeated_answer_gets_one_nudge_before_escalation(tmp_path):
    store, wf, state = to_impl(tmp_path, "stuck")
    assert state == "ESCALATED_TO_HUMAN"
    nudge = [e for e in store.events("t1") if e["kind"] == "repeat_nudge"]
    assert len(nudge) == 1
    assert any("nudged" in h for h in wf.ctx["history"])
    assert "byte-identical" in wf.ctx["feedback"]          # the model was told exactly what went wrong
