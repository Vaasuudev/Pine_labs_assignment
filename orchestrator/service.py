"""Shared service layer used by both the CLI and the dashboard, so they cannot diverge."""
import re
import time
import uuid
from pathlib import Path

from . import agents
from .llm import GeminiProvider, MockProvider
from .sandbox import Sandbox
from .store import Store
from .workflow import Workflow

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = ROOT / "runs"
DEFAULT_REPO = ROOT / "sample_repo"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

# mock-model scenarios (scripted faults); the live provider ignores these
SCENARIOS = {
    "clean": {},
    "fail-then-fix": {"dev": "float_bug"},                       # DEFAULT demo: generated code fails tests
    "disagree": {"disagree": True, "disagree_rounds": 1},        # resolved by architect ruling
    "disagree-accept": {"disagree": True, "disagree_rounds": 1, "review": "accept"},   # agents may not widen scope
    "deadlock": {"disagree": True, "disagree_rounds": 99},       # escalates to a human
    "security": {"dev": "eval"},                                 # critical finding, then fixed
    "unrelated": {"dev": "unrelated"},                           # tries to edit shop/config.py
    "stuck": {"dev": "stuck"},                                   # repeats the same failed action
    "bad-qa": {"qa": "bad"},
    "lint-warn": {"dev": "lint"},                                # only lint complains: advisory, not blocking                                     # QA writes a wrong expected value
    "malformed": {"provider": {"malformed_on": "architect", "malformed_calls": 1}},
    "malformed-fatal": {"provider": {"malformed_on": "architect", "malformed_calls": 99}},
    "unsupported": {"provider": {"unsupported_on": "developer"}},
    "outage": {"provider": {"outage_on": "developer", "outage_calls": 99}},
}
PROVIDERS = ("mock", "gemini")


def open_store(runs):
    runs = Path(runs)
    runs.mkdir(parents=True, exist_ok=True)
    return Store(runs / "state.db")


def make_providers(provider, scenario="clean", fallback=False):
    """Returns (providers, backoff_seconds)."""
    if provider == "gemini":
        ps = [GeminiProvider()]
        import os
        if os.environ.get("GEMINI_FALLBACK_MODEL"):
            ps.append(GeminiProvider(model=os.environ["GEMINI_FALLBACK_MODEL"]))
        return ps, 2.0
    sc = SCENARIOS[scenario]
    dev_faults = {k: v for k, v in sc.items() if k != "provider"}
    ps = [MockProvider(agents.behaviours(dev_faults), sc.get("provider"))]
    if fallback:
        backup = MockProvider(agents.behaviours({}), None)
        backup.name, backup.model = "mock-fallback", "mock-small"
        ps.append(backup)
    return ps, 0.0


def workflow(store, run_id, runs, provider=None, scenario=None, fallback=False):
    """Workflow bound to a run. provider/scenario default to the ones the run started with."""
    ctx = store.get_run(run_id)["context"]
    provider = provider or ctx.get("provider", "mock")
    scenario = scenario or ctx.get("scenario", "clean")
    ps, backoff = make_providers(provider, scenario, fallback)
    return Workflow(store, run_id, runs, ps, backoff=backoff)


def new_run_id():
    return f"run-{time.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


MOCK_REQUIREMENT = "REQ-101"


def create_run(store, runs, requirement_text, requirement_id, repo=DEFAULT_REPO, provider="mock",
               scenario="fail-then-fix", run_id=None, strict_mock=True):
    """Create the run record and its isolated sandbox. Does not advance the workflow.

    The mock model is scripted for REQ-101 only and would silently ignore any other requirement
    text, so by default that combination is refused rather than producing a misleading run."""
    if provider == "mock" and strict_mock and requirement_id != MOCK_REQUIREMENT:
        raise ValueError("The mock model is scripted for REQ-101 only and ignores other requirement text. "
                         "Choose REQ-101.md, or switch the model to Gemini to use your own requirement.")
    run_id = run_id or new_run_id()
    if not RUN_ID_RE.match(run_id):
        raise ValueError("run id must be 1-40 chars of letters, digits, - or _")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    if not requirement_text.strip() or len(requirement_text) > 6000:
        raise ValueError("requirement must be 1-6000 characters")
    runs = Path(runs)
    (runs / run_id).mkdir(parents=True, exist_ok=False)
    store.create_run(run_id, requirement_text[:200])
    sb = Sandbox(Path(repo), runs / run_id / "repo", f"agent/{requirement_id.lower()}", store, run_id)
    sb.create()
    baseline = ROOT / "baselines" / f"{requirement_id}.py"
    wf = Workflow(store, run_id, runs, [])
    wf._save(stage="planning", requirement_id=requirement_id, requirement_text=requirement_text,
             source_repo=str(Path(repo).resolve()), sandbox=str(sb.root), branch=sb.branch,
             baseline=str(baseline) if baseline.exists() else None, scenario=scenario, provider=provider)
    (runs / run_id / "requirement.md").write_text(requirement_text, encoding="utf-8")
    return run_id


def requirement_files():
    d = ROOT / "requirements"
    return sorted(p.name for p in d.glob("*.md")) if d.exists() else []


def _clip(v, n):
    s = v if isinstance(v, str) else __import__("json").dumps(v, default=str)
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more chars]"


ACTIONS = {"AWAITING_PLAN_APPROVAL": "plan", "AWAITING_FINAL_APPROVAL": "merge",
           "MERGED_AWAITING_DEPLOY_APPROVAL": "deploy", "PAUSED_PROVIDER_DOWN": "resume"}
TERMINAL = {"DEPLOYED_MOCK", "REJECTED", "DEPLOY_REJECTED", "BLOCKED_NEEDS_HUMAN", "ESCALATED_TO_HUMAN",
            "FAILED_MALFORMED_OUTPUT", "FAILED_PROVIDER_ERROR", "MERGE_FAILED"}


def summary(store, run_id, runs):
    """Everything the dashboard needs about one run, as JSON-safe data."""
    run = store.get_run(run_id)
    c = run["context"]
    wf = Workflow(store, run_id, runs, [])
    steps = [{"seq": s["id"], "agent": s["agent"], "status": s["status"], "model": s["model"],
              "provider": s["provider"], "tokens_in": s["tokens_in"], "tokens_out": s["tokens_out"],
              "cost_usd": round(s["cost_usd"] or 0, 6), "latency_ms": round(s["latency_ms"] or 0, 1),
              "error": s["error"], "output": _clip(s["output"], 3500)} for s in store.steps(run_id)]
    review = diff = ""
    for key, name in (("review_path", "review"), ("patch_path", "diff")):
        p = c.get(key)
        if p and Path(p).exists():
            text = Path(p).read_text(encoding="utf-8")
            if name == "review":
                review = text
            else:
                diff = text
    evidence = None
    if c.get("manifest"):
        problems = wf.verify_evidence() if run["state"] in ("AWAITING_FINAL_APPROVAL",) else []
        evidence = {"intact": not problems, "problems": problems, "manifest": c["manifest"]}
    return {
        "id": run_id, "state": run["state"], "stage": c.get("stage"), "created": run["created"],
        "updated": run["updated"], "requirement": c.get("requirement_text", ""), "provider": c.get("provider"),
        "scenario": c.get("scenario"), "halt_reason": c.get("halt_reason"),
        "action": ACTIONS.get(run["state"]), "terminal": run["state"] in TERMINAL or run["state"] == "DEPLOYED_MOCK",
        "spec": c.get("spec"), "design": c.get("design"), "attempt": c.get("attempt"),
        "history": c.get("history", []), "changed": c.get("changed", []), "results": [
            {k: r.get(k) for k in ("name", "passed", "advisory", "summary", "output", "findings", "examples")}
            for r in c.get("results", [])], "rubric": c.get("rubric"),
        "tests_fail_before": c.get("tests_fail_before"), "oracle": c.get("oracle"),
        "recommendation": c.get("recommendation"), "events": [
            {"t": round(e["ts"] - run["created"], 2), "kind": e["kind"], "detail": e["detail"]}
            for e in store.events(run_id) if e["kind"] != "file_write"],
        "steps": steps, "usage": store.usage_totals(run_id),
        "approvals": [{k: a[k] for k in ("gate", "decision", "decided_by", "note", "ts")} for a in store.approvals(run_id)],
        "review": review, "diff": diff, "evidence": evidence, "merge_commit": c.get("merge_commit"),
    }
