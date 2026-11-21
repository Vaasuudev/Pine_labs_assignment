import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestrator import service  # noqa: E402
from orchestrator.workflow import Workflow  # noqa: E402


def start(tmp_path, scenario="clean", fallback=False, run_id="t1", providers=None, req_id="REQ-101", text=None):
    """Create a run (sandbox + context) and return (store, workflow, runs_dir)."""
    runs = tmp_path / "runs"
    store = service.open_store(runs)
    text = text or (ROOT / "requirements/REQ-101.md").read_text(encoding="utf-8")
    service.create_run(store, runs, text, req_id, ROOT / "sample_repo", "mock", scenario, run_id, strict_mock=False)
    if providers is None:
        providers, _ = service.make_providers("mock", scenario, fallback)
    return store, Workflow(store, run_id, runs, providers), runs


def to_impl(tmp_path, scenario="clean", **kw):
    store, wf, runs = start(tmp_path, scenario, **kw)
    assert wf.advance() == "AWAITING_PLAN_APPROVAL"
    return store, wf, wf.decide_plan("approve", "tester")


def kinds(store, run_id="t1"):
    return [e["kind"] for e in store.events(run_id)]
