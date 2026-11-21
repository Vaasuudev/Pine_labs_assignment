"""Workflow engine: a resumable state machine. Stage pointer + persisted context make every
invocation of advance() safe to repeat. The ONLY code path to merge/deploy is a human decision.

Trust boundaries (what the orchestrator refuses to take on an agent's word):
  * file scope        -> enforced by the sandbox preflight, not by the developer's good manners
  * test results      -> produced by running the tools, never reported by a model
  * security verdict  -> the scanner overrules the security agent
  * test validity     -> QA tests must pass against a held-out reference, or they are rejected
  * scope widening    -> an architect "accept" never widens scope; only a human re-approval can
  * evidence           -> REVIEW.md + diff + commit are hashed; approval re-checks them"""
import fnmatch
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from . import agents, checks, evaluate, repo
from .agents import MalformedOutput, SPECS
from .llm import ProviderFatal, ResilientClient, WorkflowPaused
from .sandbox import Sandbox

MAX_ATTEMPTS = 3        # developer implement->check iterations
MAX_NEGOTIATION = 2     # architect/developer scope-disagreement rounds
MAX_REPAIRS = 2         # re-prompts after malformed output
MAX_QA_ROUNDS = 3       # QA test regenerations after the reference oracle rejects tests
CACHEABLE = {"analyst", "architect", "release"}   # deterministic, safe to reuse on resume


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_name(by):
    by = (by or "").strip()
    if not by or len(by) > 80 or any(ord(c) < 32 for c in by):
        raise PermissionError("a named reviewer (1-80 printable characters) is required")
    return by


class Workflow:
    def __init__(self, store, run_id, runs_dir, providers, backoff=0.0):
        self.store, self.run_id, self.runs_dir = store, run_id, Path(runs_dir)
        self.client = ResilientClient(providers, store, run_id, backoff=backoff)

    # ------------------------------------------------------------ plumbing
    @property
    def ctx(self):
        return self.store.get_run(self.run_id)["context"]

    def _save(self, **kw):
        c = self.ctx
        c.update(kw)
        self.store.save_context(self.run_id, c)

    def _sandbox(self):
        c = self.ctx
        sb = Sandbox(c["source_repo"], c["sandbox"], c["branch"], self.store, self.run_id)
        sb.load_policy()
        return sb

    def _halt(self, state, why):
        self.store.set_state(self.run_id, state)
        self.store.event(self.run_id, "halt", why)
        self._save(stage="halted", halt_reason=why)
        return state

    def invoke(self, agent, payload, cache=None, extra=None):
        """One agent call: validation, bounded repair loop, usage accounting."""
        cache = agent in CACHEABLE if cache is None else cache
        if cache:
            hit = self.store.completed(self.run_id, agent, 0)
            if hit is not None:
                return hit
        seq = len(self.store.steps(self.run_id)) + 1
        hint = None
        for repair in range(MAX_REPAIRS + 1):
            body = dict(payload, **({"repair_hint": hint} if hint else {}))
            text, usage = self.client.call(agent, SPECS[agent].system, body)   # may raise WorkflowPaused/Fatal
            try:
                data = agents.validate(agent, text)
                if extra:
                    extra(data)
                self.store.record_step(self.run_id, agent, 0 if cache else seq, "ok", payload, data, usage=usage)
                return data
            except MalformedOutput as e:
                hint = str(e)
                self.store.record_step(self.run_id, agent, seq, "malformed", payload, text[:500], str(e), usage)
                self.store.event(self.run_id, "malformed_output", f"{agent}: {e} (repair {repair+1}/{MAX_REPAIRS})")
        raise MalformedOutput(f"{agent} still malformed after {MAX_REPAIRS} repairs: {hint}")

    # ------------------------------------------------------------ driver
    def advance(self):
        """Run until a human gate, a halt, completion, or a provider pause. Idempotent."""
        try:
            while True:
                stage = self.ctx["stage"]
                if stage == "planning":
                    self._planning()
                elif stage == "tests":
                    if self._tests():
                        break
                elif stage == "implement":
                    if self._implement():
                        break
                elif stage == "release":
                    self._release()
                else:   # plan_gate, final_gate, merged, halted, done: waiting on a human or finished
                    break
        except WorkflowPaused as e:
            self.store.set_state(self.run_id, "PAUSED_PROVIDER_DOWN")
            self.store.event(self.run_id, "paused", f"{e}. State persisted; `resume` continues from stage "
                                                     f"'{self.ctx['stage']}'.")
        except ProviderFatal as e:
            self._halt("FAILED_PROVIDER_ERROR", str(e))
        except MalformedOutput as e:
            self._halt("FAILED_MALFORMED_OUTPUT", str(e))
        return self.store.get_run(self.run_id)["state"]

    # ------------------------------------------------------------ stages
    def _planning(self):
        self.store.set_state(self.run_id, "PLANNING")
        sb = self._sandbox()
        protected = lambda rel: any(fnmatch.fnmatchcase(rel, p) for p in sb.protected)   # noqa: E731
        index = [p for p in repo.python_sources(sb.root) if not protected(p)]
        spec = self.invoke("analyst", {"requirement": self.ctx["requirement_text"]})

        def check_design(d):
            for p in d["allowed_paths"]:
                if p not in index:
                    raise MalformedOutput(f"allowed path '{p}' is not an editable source file; choose from {index}")
            if len(set(d["allowed_paths"])) != len(d["allowed_paths"]):
                raise MalformedOutput("allowed_paths contains duplicates")

        design = self.invoke("architect", {"acceptance_criteria": spec["acceptance_criteria"],
                                           "repo_index": index, "sources": repo.read_sources(sb.root, index)},
                             extra=check_design)
        self._save(spec=spec, design=design, repo_index=index, stage="plan_gate")
        self.store.set_state(self.run_id, "AWAITING_PLAN_APPROVAL")

    def _oracle_check(self, sb, qa_file):
        """Run the QA tests against the held-out reference. A test failing here is a wrong test."""
        baseline = self.ctx.get("baseline")
        if not baseline:
            return None
        target = evaluate.baseline_meta(baseline)
        tmp = Path(tempfile.mkdtemp(prefix="oracle-"))
        try:
            dst = tmp / "repo"
            shutil.copytree(sb.root, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
            (dst / target).write_text(Path(baseline).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
            p = dst / qa_file["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(qa_file["content"], encoding="utf-8", newline="\n")
            return checks.run_pytest(dst)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _tests(self):
        """Returns True when the run halted."""
        c = self.ctx
        sb = self._sandbox()
        api = repo.api_summary(sb.root, repo.python_sources(sb.root))
        feedback = None
        for rnd in range(1, MAX_QA_ROUNDS + 1):
            qa = self.invoke("qa", {"acceptance_criteria": c["spec"]["acceptance_criteria"], "public_api": api,
                                    "feedback": feedback}, cache=False)
            problems = sb.preflight("qa", [qa], SPECS["qa"].write_scope, must_be_new=True)
            if problems:
                feedback = "; ".join(m for _, m in problems)
                self.store.event(self.run_id, "qa_rejected", f"round {rnd}/{MAX_QA_ROUNDS}: {feedback}")
                continue
            oracle = self._oracle_check(sb, qa)
            if oracle is not None and not oracle["passed"]:
                feedback = ("Your tests FAIL against the reference implementation, so your expected values are "
                            "wrong. Output:\n" + oracle["output"][-1200:])
                self.store.event(self.run_id, "qa_oracle_rejected",
                                 f"round {rnd}/{MAX_QA_ROUNDS}: tests fail on the reference ({oracle['summary']})")
                continue
            sb.write("qa", qa["path"], qa["content"], SPECS["qa"].write_scope)
            red = checks.run_pytest(sb.root)        # TDD 'red': tests must detect the missing behaviour
            sha_ = sb.commit("tests: acceptance tests from QA agent")
            self.store.event(self.run_id, "tests_red_check",
                             f"tests fail on unmodified code: {not red['passed']} ({red['summary']})")
            self._save(tests_commit=sha_, tests_path=qa["path"], stage="implement", attempt=1, feedback=None,
                       history=[], tests_fail_before=not red["passed"],
                       oracle=("passed" if oracle else "no reference available"))
            self.store.set_state(self.run_id, "IMPLEMENTING")
            return False
        return self._halt("BLOCKED_NEEDS_HUMAN", f"QA could not produce valid tests in {MAX_QA_ROUNDS} rounds: {feedback}")

    def _implement(self):
        """Returns True when the loop should stop (halt)."""
        c = self.ctx
        sb, design = self._sandbox(), c["design"]
        attempt, feedback, history = c["attempt"], c["feedback"], c["history"]
        allowed = design["allowed_paths"]
        files = {p: sb.read(p) for p in allowed if (sb.root / p).exists()}
        readonly = {}
        for item in design["impacted"]:
            pth = item["path"]
            if pth not in allowed and pth in c["repo_index"] and len(readonly) < 4:
                readonly[pth] = sb.read(pth)[:8000]

        # --- developer, with bounded architect/developer negotiation
        ruling, rounds = None, 0
        while True:
            out = self.invoke("developer", {"acceptance_criteria": c["spec"]["acceptance_criteria"],
                                            "design": design["design"], "allowed_paths": allowed,
                                            "files": files, "readonly_context": readonly,
                                            "feedback": feedback, "architect_ruling": ruling}, cache=False)
            if not out["objections"]:
                break
            rounds += 1
            obj = out["objections"][0]
            self.store.event(self.run_id, "disagreement",
                             f"developer wants {obj['path']} ({obj['reason']}); round {rounds}/{MAX_NEGOTIATION}")
            if rounds > MAX_NEGOTIATION:
                self._halt("ESCALATED_TO_HUMAN", f"architect and developer deadlocked on scope for {obj['path']} "
                           f"after {MAX_NEGOTIATION} rounds; human must rule")
                return True
            rev = self.invoke("architect_review", {"objection": obj, "design": design["design"],
                                                   "allowed_paths": allowed}, cache=False)
            self.store.event(self.run_id, "architect_ruling", f"{rev['decision']}: {rev['reason']}")
            if rev["decision"] == "accept":
                # Policy: agents can never widen an approved scope. A human re-approves the plan.
                self._halt("ESCALATED_TO_HUMAN", f"architect agrees {obj['path']} should be in scope; widening an "
                           "approved scope requires a human, so the run stops here")
                return True
            ruling = rev["reason"]
        if not out["files"]:
            return self._fail_attempt(attempt, history, "You returned no files and no unresolved objection.",
                                      "developer returned nothing")

        # --- preflight then write (no side effects if anything is out of scope or a repeat)
        problems = sb.preflight("developer", out["files"], allowed)
        if any(kind == "repeat" for kind, _ in problems):
            nudges = c.get("repeat_nudges", 0)
            if nudges >= 1:                       # second identical answer: stop and ask a human
                return self._no_progress(problems)
            self._save(repeat_nudges=nudges + 1)
            msg = "; ".join(m for _, m in problems)
            self.store.event(self.run_id, "repeat_nudge", f"{msg}; one corrective re-prompt before escalating")
            return self._fail_attempt(
                attempt, history,
                "Your last output was byte-identical to an earlier attempt that already failed. Submit DIFFERENT "
                "code that fixes the failure below, or return an objection.\n" + (feedback or ""),
                "repeated identical output (nudged)")
        if problems:
            msg = "; ".join(m for _, m in problems)
            self.store.event(self.run_id, "scope_violation", f"BLOCKED before write: {msg}")
            return self._fail_attempt(attempt, history, f"Write rejected: {msg}. Only modify {allowed}.",
                                      "scope violation (no files written)")
        for f in out["files"]:
            sb.write("developer", f["path"], f["content"], allowed)
        commit = sb.commit(f"developer attempt {attempt}: {out['rationale'][:100]}")
        self.store.event(self.run_id, "commit", f"{commit} on {sb.branch}")

        # --- checks (deterministic tools); agents interpret but cannot overrule
        changed = sb.changed_files()
        py = [f for f in changed if f.endswith(".py")]
        results = [checks.run_pytest(sb.root), checks.run_lint(sb.root, py)]
        scan = checks.security_scan(sb.root, py)
        results.append(scan)
        if c.get("baseline"):
            results.append(evaluate.differential(sb.root, c["baseline"]))
        diff = sb.diff()
        sec = self.invoke("security", {"findings": scan["findings"], "diff": diff[:20000]}, cache=False)
        if sec["verdict"] != ("pass" if scan["passed"] else "block"):
            self.store.event(self.run_id, "agent_disagrees_with_tool",
                             f"security agent said {sec['verdict']}, scanner passed={scan['passed']}; scanner wins")
        score = evaluate.rubric(changed, allowed, results, diff)
        for r in results:
            self.store.event(self.run_id, "check", f"attempt {attempt} {r['name']}: "
                                                    f"{'PASS' if r['passed'] else 'WARN' if r.get('advisory') else 'FAIL'} - {r['summary']}")
        failed = [r for r in results if not r["passed"] and not r.get("advisory")]
        if not failed and score["passed"]:
            history.append(f"attempt {attempt}: all checks passed, rubric {score['score']}/100")
            self._save(results=results, rubric=score, changed=changed, commit=commit, history=history,
                       security_notes=sec["notes"], stage="release")
            return False
        why = "; ".join(f"{r['name']}: {r['summary']}" for r in failed) or f"rubric {score['score']}/100"
        fb = "Checks failed. " + why + "\n" + "\n".join(
            r.get("output", "") or json.dumps(r.get("findings", r.get("examples", ""))) for r in failed)[:1800]
        self._save(results=results, rubric=score)
        sb.reset_hard(self.ctx["tests_commit"])    # discard failed attempt; keep QA tests
        return self._fail_attempt(attempt, history, fb, why)

    def _fail_attempt(self, attempt, history, feedback, why):
        history.append(f"attempt {attempt}: FAILED ({why})")
        self.store.event(self.run_id, "attempt_failed", f"attempt {attempt}/{MAX_ATTEMPTS}: {why}")
        if attempt >= MAX_ATTEMPTS:
            self._save(history=history)
            self._halt("BLOCKED_NEEDS_HUMAN", f"{MAX_ATTEMPTS} attempts exhausted; last failure: {why}")
            return True
        self._save(attempt=attempt + 1, feedback=feedback, history=history)
        self.store.event(self.run_id, "retry", f"handing feedback back to developer for attempt {attempt + 1}")
        return False

    def _no_progress(self, problems):
        msg = "; ".join(m for _, m in problems)
        self.store.event(self.run_id, "repeat_blocked", msg)
        self._halt("ESCALATED_TO_HUMAN", f"developer repeated an identical action (no progress): {msg}")
        return True

    # ------------------------------------------------------------ release + evidence
    def _evidence_markdown(self, c, rel, sb):
        u = self.store.usage_totals(self.run_id)
        models = sorted({f"{s['provider']}/{s['model']}" for s in self.store.steps(self.run_id) if s["model"]})
        L = [f"# Approval pack: {c['requirement_id']}", "",
             f"**Branch:** `{c['branch']}`  **Commit:** `{c['commit']}`  **Models:** {', '.join(models) or 'n/a'}",
             f"**Rubric score:** {c['rubric']['score']}/100 (pass mark {c['rubric']['pass_mark']})", "",
             "## Evidence (produced by tools, not by models)", ""]
        L += [f"- {'PASS' if r['passed'] else 'WARN' if r.get('advisory') else 'FAIL'} **{r['name']}**: {r['summary']}"
              + (" (advisory, does not block)" if r.get("advisory") and not r["passed"] else "") for r in c["results"]]
        L += [f"  - {x}" for r in c["results"] if r.get("advisory") and not r["passed"] for x in r.get("output", "").splitlines()[:5]]
        L += [f"- {'PASS' if c.get('tests_fail_before') else 'WARN'} **tests detect the missing behaviour**: "
              f"tests {'fail' if c.get('tests_fail_before') else 'do NOT fail'} on the unmodified code",
              f"- {'PASS' if c.get('oracle') == 'passed' else 'INFO'} **QA tests vs reference oracle**: {c.get('oracle')}",
              "", "## Iteration history", ""] + [f"- {h}" for h in c["history"]]
        L += ["", "## Files changed", ""] + [f"- `{f}`" for f in c["changed"]]
        L += ["", "## Usage", "", f"- {u['calls']} model calls, {u['tokens_in']} tokens in / {u['tokens_out']} out, "
              f"${u['cost_usd']} estimated, {u['latency_ms']} ms of model latency"]
        return L

    def _release(self):
        c = self.ctx
        sb = self._sandbox()
        ev = {"requirement_id": c["requirement_id"], "branch": c["branch"], "commit": c["commit"],
              "checks": [{k: r[k] for k in ("name", "passed", "summary")} for r in c["results"]],
              "rubric": c["rubric"], "changed": c["changed"], "history": c["history"],
              "security_notes": c["security_notes"]}
        rel = self.invoke("release", {"evidence": ev})
        recommendation = "approve" if c["rubric"]["passed"] else "reject"
        if rel["recommendation"] != recommendation:
            self.store.event(self.run_id, "agent_disagrees_with_tool",
                             f"release agent recommended {rel['recommendation']}; rubric says {recommendation}")
        lines = self._evidence_markdown(c, rel, sb)
        lines += ["", "## Release agent narrative (model-written; verify against the evidence above)", "",
                  rel["summary_md"], "", "## Known risks", ""]
        lines += [f"- {r}" for r in rel["risks"]]
        lines += [f"- Security note: {c['security_notes']}",
                  "- Sandbox isolates writes only; tests execute generated code on the host."]
        review = "\n".join(lines) + "\n"
        diff = sb.diff()
        rundir = self.runs_dir / self.run_id
        (rundir / "REVIEW.md").write_text(review, encoding="utf-8")
        (rundir / "candidate.patch").write_text(diff, encoding="utf-8")
        manifest = {"review_sha256": sha(review), "patch_sha256": sha(diff), "commit": sb.branch_head()}
        self._save(review_path=str(rundir / "REVIEW.md"), patch_path=str(rundir / "candidate.patch"),
                   recommendation=recommendation, manifest=manifest, stage="final_gate")
        self.store.set_state(self.run_id, "AWAITING_FINAL_APPROVAL")

    def verify_evidence(self):
        """Re-hash what the reviewer was shown. Returns a list of problems (empty = intact)."""
        c, problems = self.ctx, []
        m = c.get("manifest")
        if not m:
            return ["no evidence manifest"]
        try:
            if sha(Path(c["review_path"]).read_text(encoding="utf-8")) != m["review_sha256"]:
                problems.append("REVIEW.md changed after it was generated")
            if sha(Path(c["patch_path"]).read_text(encoding="utf-8")) != m["patch_sha256"]:
                problems.append("candidate.patch changed after it was generated")
            sb = self._sandbox()
            if sb.branch_head() != m["commit"]:
                problems.append(f"branch moved: reviewed {m['commit']}, now {sb.branch_head()}")
            elif sha(sb.diff()) != m["patch_sha256"]:
                problems.append("branch diff no longer matches the reviewed patch")
        except (OSError, KeyError, RuntimeError) as e:
            problems.append(f"cannot verify evidence: {e}")
        return problems

    # ------------------------------------------------------------ human decisions
    def _gate(self, stage, state):
        run = self.store.get_run(self.run_id)
        if run["context"]["stage"] != stage or run["state"] != state:
            raise PermissionError(f"run is {run['state']}; this decision is only valid in {state}")

    def decide_plan(self, decision, by, note=""):
        self._gate("plan_gate", "AWAITING_PLAN_APPROVAL")
        by = clean_name(by)
        self.store.approval(self.run_id, "plan", decision, by, note)
        if decision == "approve":
            self._save(stage="tests")
            return self.advance()
        return self._halt("REJECTED", f"plan rejected by {by}: {note}")

    def decide_final(self, decision, by, note=""):
        self._gate("final_gate", "AWAITING_FINAL_APPROVAL")
        by = clean_name(by)
        if decision == "approve":
            problems = self.verify_evidence()
            if problems:
                self.store.event(self.run_id, "evidence_tampered", "; ".join(problems))
                raise PermissionError("approval refused, evidence changed since review: " + "; ".join(problems))
        self.store.approval(self.run_id, "merge", decision, by, note)
        if decision == "approve":
            try:
                merge = self._sandbox().merge_to_main(self.ctx["manifest"]["commit"])
            except RuntimeError as e:
                return self._halt("MERGE_FAILED", str(e))
            self._save(stage="merged", merge_commit=merge)
            self.store.set_state(self.run_id, "MERGED_AWAITING_DEPLOY_APPROVAL")
            return "MERGED_AWAITING_DEPLOY_APPROVAL"
        return self._halt("REJECTED", f"merge rejected by {by}: {note}")

    def decide_deploy(self, decision, by, note=""):
        self._gate("merged", "MERGED_AWAITING_DEPLOY_APPROVAL")
        by = clean_name(by)
        self.store.approval(self.run_id, "deploy", decision, by, note)
        if decision == "approve":
            rec = self.runs_dir / self.run_id / "DEPLOY_REQUEST.json"
            rec.write_text(json.dumps({"mock": True, "commit": self.ctx["merge_commit"], "approved_by": by}),
                           encoding="utf-8")
            self._save(stage="done")
            self.store.set_state(self.run_id, "DEPLOYED_MOCK")
            return "DEPLOYED_MOCK"
        return self._halt("DEPLOY_REJECTED", f"deploy rejected by {by}: {note}")
