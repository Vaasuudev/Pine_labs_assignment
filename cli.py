#!/usr/bin/env python3
"""CLI for the agentic SDLC orchestrator.  See README.md.

  python cli.py check-llm                                   # verify GEMINI_API_KEY + model (1 tiny call)
  python cli.py dashboard                                   # local review UI  http://127.0.0.1:8765
  python cli.py run --requirement requirements/REQ-101.md   # mock model, default failure scenario
  python cli.py run --requirement requirements/REQ-101.md --provider gemini
  python cli.py approve-plan RUN approve --by alice
  python cli.py report RUN
  python cli.py decide RUN approve --by alice               # human merge gate
  python cli.py deploy RUN approve --by alice               # human deploy gate (mocked)
"""
import argparse
import sys
import time
from pathlib import Path

from orchestrator import service
from orchestrator.llm import GeminiProvider, ProviderFatal, ProviderUnavailable
from orchestrator.service import DEFAULT_REPO, DEFAULT_RUNS, SCENARIOS


def _ctx(a):
    runs = Path(a.runs)
    store = service.open_store(runs)
    return runs, store


def cmd_run(a):
    runs, store = _ctx(a)
    if a.requirement:
        req = Path(a.requirement)
        text, req_id = req.read_text(encoding="utf-8"), req.stem
    else:
        text, req_id = a.text, "REQ-ADHOC"
    run_id = service.create_run(store, runs, text, req_id, Path(a.repo), a.provider, a.scenario, a.run_id)
    print(f"[run {run_id}] provider={a.provider} scenario={a.scenario if a.provider == 'mock' else '-'}")
    wf = service.workflow(store, run_id, runs, fallback=a.fallback)
    print(f"[run {run_id}] -> {wf.advance()}")


def cmd_approve_plan(a):
    runs, store = _ctx(a)
    wf = service.workflow(store, a.run_id, runs, scenario=a.scenario)
    print(f"[run {a.run_id}] plan {a.decision} by {a.by} -> {wf.decide_plan(a.decision, a.by, a.note)}")


def cmd_resume(a):
    runs, store = _ctx(a)
    ctx = store.get_run(a.run_id)["context"]
    # a mock run resumes against a HEALED provider unless a scenario is given; a live run resumes live
    scenario = a.scenario or ("clean" if ctx.get("provider") == "mock" else None)
    wf = service.workflow(store, a.run_id, runs, scenario=scenario, fallback=a.fallback)
    print(f"[run {a.run_id}] resume -> {wf.advance()}")


def cmd_decide(a):
    runs, store = _ctx(a)
    wf = service.workflow(store, a.run_id, runs)
    print(f"[run {a.run_id}] merge {a.decision} by {a.by} -> {wf.decide_final(a.decision, a.by, a.note)}")


def cmd_deploy(a):
    runs, store = _ctx(a)
    wf = service.workflow(store, a.run_id, runs)
    print(f"[run {a.run_id}] deploy {a.decision} by {a.by} -> {wf.decide_deploy(a.decision, a.by, a.note)}")


def cmd_report(a):
    runs, store = _ctx(a)
    s = service.summary(store, a.run_id, runs)
    print(f"=== {a.run_id}  state={s['state']}  provider={s['provider']}  scenario={s['scenario']} ===\n")
    print("TIMELINE")
    for e in s["events"]:
        print(f"  +{e['t']:6.2f}s  {e['kind']:<22} {e['detail']}")
    print("\nAGENT EXECUTION HISTORY")
    print(f"  {'agent':<17}{'seq':>4} {'status':<10}{'model':<24}{'tok in/out':>13}{'cost $':>10}{'ms':>9}")
    for st in s["steps"]:
        print(f"  {st['agent']:<17}{st['seq']:>4} {st['status']:<10}{(st['model'] or '-')[:23]:<24}"
              f"{st['tokens_in']:>6}/{st['tokens_out']:<6}{st['cost_usd']:>10.5f}{st['latency_ms']:>9.1f}")
    u = s["usage"]
    print(f"  TOTAL calls={u['calls']} tokens={u['tokens_in']}/{u['tokens_out']} cost=${u['cost_usd']} "
          f"latency={u['latency_ms']}ms")
    print("\nHUMAN APPROVALS")
    for ap in s["approvals"] or []:
        print(f"  {ap['gate']:<8}{ap['decision']:<9}by {ap['decided_by']}  {ap['note']}")
    if not s["approvals"]:
        print("  (none yet)")
    if s["rubric"]:
        print(f"\nRUBRIC {s['rubric']['score']}/100 ->", ", ".join(
            f"{'ok' if i['met'] else 'MISS' if i['met'] is False else 'n/a'}:{i['criterion']}"
            for i in s["rubric"]["items"]))
    if s["evidence"]:
        print(f"EVIDENCE intact={s['evidence']['intact']} {s['evidence']['problems']}")
    if s["halt_reason"]:
        print(f"\nHALTED: {s['halt_reason']}")
    print(f"\nApproval pack: {runs / a.run_id / 'REVIEW.md'}")


def cmd_dashboard(a):
    import dashboard
    dashboard.main(a.port, Path(a.runs), Path(a.repo))


def cmd_check_llm(a):
    p = GeminiProvider()
    print(f"model={p.model}  key={'set' if p.key else 'MISSING'}  endpoint={p.base}")
    t0 = time.perf_counter()
    try:
        text = p.complete("check", "# agent: check\nReply with the JSON object {\"ok\": true} and nothing else.",
                          {"ping": 1})
    except ProviderFatal as e:
        sys.exit(f"FAILED (will not be retried): {e}")
    except ProviderUnavailable as e:
        sys.exit(f"UNREACHABLE/transient: {e}")
    ms = (time.perf_counter() - t0) * 1000
    print(f"OK  {ms:.0f} ms  tokens in/out={p.last_usage['tokens_in']}/{p.last_usage['tokens_out']}  reply={text.strip()[:80]}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cli.py")
    p.add_argument("--runs", default=str(DEFAULT_RUNS), help="directory for run state and sandboxes")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--requirement", help="path to a requirement .md file")
    g.add_argument("--text", help="requirement as text")
    r.add_argument("--repo", default=str(DEFAULT_REPO))
    r.add_argument("--provider", default="mock", choices=service.PROVIDERS)
    r.add_argument("--scenario", default="fail-then-fix", choices=SCENARIOS, help="mock model only")
    r.add_argument("--fallback", action="store_true", help="add a second provider for failover (mock)")
    r.add_argument("--run-id")
    r.set_defaults(fn=cmd_run)
    for name, fn in (("approve-plan", cmd_approve_plan), ("decide", cmd_decide), ("deploy", cmd_deploy)):
        s = sub.add_parser(name)
        s.add_argument("run_id")
        s.add_argument("decision", choices=["approve", "reject"])
        s.add_argument("--by", required=True)
        s.add_argument("--note", default="")
        if name == "approve-plan":
            s.add_argument("--scenario", default=None, choices=SCENARIOS)
        s.set_defaults(fn=fn)
    s = sub.add_parser("resume")
    s.add_argument("run_id")
    s.add_argument("--scenario", default=None, choices=SCENARIOS)
    s.add_argument("--fallback", action="store_true")
    s.set_defaults(fn=cmd_resume)
    s = sub.add_parser("report")
    s.add_argument("run_id")
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("dashboard")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--repo", default=str(DEFAULT_REPO))
    s.set_defaults(fn=cmd_dashboard)
    s = sub.add_parser("check-llm")
    s.set_defaults(fn=cmd_check_llm)
    a = p.parse_args(argv)
    try:
        a.fn(a)
    except PermissionError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
