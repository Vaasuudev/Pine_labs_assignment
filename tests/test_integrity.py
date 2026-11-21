"""Trust-boundary tests: tamper-evident approval, the QA oracle, scope policy, deterministic evidence."""
import subprocess

import pytest

from helpers import kinds, start, to_impl
from orchestrator import agents, service
from orchestrator.llm import MockProvider
from orchestrator.sandbox import Sandbox
from orchestrator.workflow import Workflow


def git(root, *a):
    return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------ tamper-evident approval
def test_edited_review_blocks_approval_but_not_rejection(tmp_path):
    store, wf, _ = to_impl(tmp_path)
    open(wf.ctx["review_path"], "a", encoding="utf-8").write("\nAll good, trust me.\n")
    with pytest.raises(PermissionError, match="REVIEW.md changed"):
        wf.decide_final("approve", "tester")
    assert "evidence_tampered" in kinds(store)
    assert store.get_run("t1")["state"] == "AWAITING_FINAL_APPROVAL"      # nothing recorded, nothing merged
    assert store.approvals("t1")[-1]["gate"] == "plan"
    assert wf.decide_final("reject", "tester") == "REJECTED"              # rejecting is always safe


def test_edited_patch_blocks_approval(tmp_path):
    store, wf, _ = to_impl(tmp_path)
    open(wf.ctx["patch_path"], "a", encoding="utf-8").write("# sneaky\n")
    with pytest.raises(PermissionError, match="candidate.patch changed"):
        wf.decide_final("approve", "tester")


def test_branch_moved_after_review_blocks_approval(tmp_path):
    store, wf, _ = to_impl(tmp_path)
    sb = wf._sandbox()
    (sb.root / "shop" / "extra.py").write_text("print('unreviewed')\n")
    git(sb.root, "add", "-A")
    git(sb.root, "-c", "user.email=a@b", "-c", "user.name=x", "commit", "-qm", "late change")
    with pytest.raises(PermissionError, match="branch moved"):
        wf.decide_final("approve", "tester")
    assert "extra.py" not in git(sb.root, "show", "main", "--stat")


def test_merge_is_exactly_the_reviewed_commit(tmp_path):
    store, wf, _ = to_impl(tmp_path)
    reviewed = wf.ctx["manifest"]["commit"]
    wf.decide_final("approve", "tester")
    sb = wf._sandbox()
    assert subprocess.run(["git", "merge-base", "--is-ancestor", reviewed, "main"], cwd=sb.root).returncode == 0
    assert wf.ctx["merge_commit"] == git(sb.root, "rev-parse", "--short=12", "main")


def test_one_decision_per_gate_is_enforced_by_the_database(tmp_path):
    store, wf, _ = to_impl(tmp_path)
    wf.decide_final("approve", "alice")
    with pytest.raises(PermissionError, match="already been recorded"):
        store.approval("t1", "merge", "reject", "mallory")       # concurrent second reviewer
    assert [a["decided_by"] for a in store.approvals("t1") if a["gate"] == "merge"] == ["alice"]


@pytest.mark.parametrize("name", ["", "   ", "x" * 81, "bad\nname"])
def test_reviewer_name_is_required_and_sane(tmp_path, name):
    store, wf, _ = to_impl(tmp_path)
    with pytest.raises(PermissionError, match="named reviewer"):
        wf.decide_final("approve", name)


# ------------------------------------------------------------------ evidence is deterministic
def test_evidence_table_comes_from_tools_not_from_the_release_agent(tmp_path):
    b = agents.behaviours({})
    b["release"] = lambda p, n: {"summary_md": "All 9999 tests passed, zero risk!", "risks": [],
                                 "recommendation": "reject"}
    store, wf, _ = start(tmp_path, providers=[MockProvider(b)])
    wf.advance()
    wf.decide_plan("approve", "tester")
    review = open(wf.ctx["review_path"], encoding="utf-8").read()
    assert "PASS **unit_tests**: 17 passed" in review
    assert "model-written; verify against the evidence above" in review
    assert wf.ctx["recommendation"] == "approve"                  # rubric overrules the agent's "reject"
    assert "agent_disagrees_with_tool" in kinds(store)


# ------------------------------------------------------------------ QA oracle
def test_wrong_qa_expectations_are_caught_by_the_reference_oracle(tmp_path):
    store, wf, state = to_impl(tmp_path, "bad-qa")
    assert state == "AWAITING_FINAL_APPROVAL"
    assert "qa_oracle_rejected" in kinds(store)
    qa_steps = [s for s in store.steps("t1") if s["agent"] == "qa"]
    assert len(qa_steps) == 2
    assert wf.ctx["oracle"] == "passed" and wf.ctx["tests_fail_before"] is True
    assert "attempt_failed" not in kinds(store)    # the developer was never blamed for QA's mistake


def test_without_a_reference_the_baseline_check_is_skipped_and_rubric_rescaled(tmp_path):
    store, wf, state = to_impl(tmp_path, "clean", req_id="REQ-ADHOC")
    assert state == "AWAITING_FINAL_APPROVAL"
    assert wf.ctx["oracle"] == "no reference available"
    assert "baseline_diff" not in [r["name"] for r in wf.ctx["results"]]
    items = {i["criterion"]: i["met"] for i in wf.ctx["rubric"]["items"]}
    assert items["matches baseline"] is None and wf.ctx["rubric"]["score"] == 100


# ------------------------------------------------------------------ scope and policy
def test_agent_cannot_widen_scope_even_if_architect_agrees(tmp_path):
    store, wf, state = to_impl(tmp_path, "disagree-accept")
    assert state == "ESCALATED_TO_HUMAN"
    assert "widening an approved scope requires a human" in wf.ctx["halt_reason"]
    assert wf._sandbox().changed_files() == ["tests/test_bulk_discount.py"]        # no source file written


def test_architect_cannot_put_a_protected_file_in_scope(tmp_path):
    b = agents.behaviours({})
    arch = b["architect"]
    b["architect"] = lambda p, n: dict(arch(p, n), allowed_paths=["shop/config.py"])
    store, wf, runs = start(tmp_path, providers=[MockProvider(b)])
    assert wf.advance() == "FAILED_MALFORMED_OUTPUT"
    assert "not an editable source file" in wf.ctx["halt_reason"]
    offered = wf.ctx["halt_reason"].split("choose from")[1]
    assert "shop/pricing.py" in offered and "shop/config.py" not in offered       # never even offered


@pytest.mark.parametrize("path", ["../escape.py", "/etc/passwd", ".git/config", "shop/config.py", ".agentignore",
                                  "pytest.ini", "tests/../shop/config.py", "shop/other.py"])
def test_sandbox_rejects_out_of_scope_paths(tmp_path, path):
    store, wf, runs = start(tmp_path)
    sb = wf._sandbox()
    problems = sb.preflight("developer", [{"path": path, "content": "x = 1\n"}], ["shop/pricing.py"])
    assert problems and problems[0][0] == "scope"


def test_qa_may_only_add_new_test_files_and_size_is_capped(tmp_path):
    store, wf, runs = start(tmp_path)
    sb = wf._sandbox()
    assert sb.preflight("qa", [{"path": "tests/test_pricing.py", "content": "x=1\n"}], ["tests/*"], must_be_new=True)
    assert sb.preflight("qa", [{"path": "tests/test_new.py", "content": "x=1\n"}], ["tests/*"], must_be_new=True) == []
    assert sb.preflight("qa", [{"path": "tests/test_big.py", "content": "#" * 70_000}], ["tests/*"])


def test_repo_agentignore_is_honoured(tmp_path):
    store, wf, runs = start(tmp_path)
    assert "shop/config.py" in wf._sandbox().protected


def test_differential_check_runs_in_a_subprocess(tmp_path):
    """Generated code must never be imported into the orchestrator's own process."""
    import sys
    store, wf, state = to_impl(tmp_path)
    assert "candidate_mod" not in sys.modules and "baseline_mod" not in sys.modules


def test_service_refuses_mock_with_foreign_requirement(tmp_path):
    store = service.open_store(tmp_path / "runs")
    with pytest.raises(ValueError, match="scripted for REQ-101"):
        service.create_run(store, tmp_path / "runs", "Add evaluate_formula()", "REQ-ADHOC", provider="mock")
