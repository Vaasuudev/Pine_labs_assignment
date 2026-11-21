"""Dashboard API flow and its security properties, against a real server on a loopback port."""
import http.client
import json
import threading
import time

import pytest

import dashboard
from helpers import ROOT


@pytest.fixture
def srv(tmp_path):
    server = dashboard.serve(0, tmp_path / "runs", ROOT / "sample_repo")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("GET", "/")
    html = c.getresponse().read().decode()
    token = html.split('const TOKEN = "')[1].split('"')[0]

    def call(method, path, body=None, token=token, host=None, raw=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        h = {"Host": host or f"127.0.0.1:{port}", "Content-Type": "application/json"}
        if token:
            h["X-Dashboard-Token"] = token
        h.update(headers or {})
        data = raw if raw is not None else (json.dumps(body) if body is not None else None)
        conn.request(method, path, body=data, headers=h)
        r = conn.getresponse()
        txt = r.read()
        try:
            return r.status, json.loads(txt)
        except ValueError:
            return r.status, txt

    server.call, server.html, server.token = call, html, token
    yield server
    server.shutdown()
    server.server_close()


def wait_for(srv, run_id, states, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        _, s = srv.call("GET", f"/api/runs/{run_id}")
        if s["state"] in states:
            return s
        time.sleep(0.1)
    raise AssertionError(f"timed out in {s['state']}")


def start_run(srv, **kw):
    code, r = srv.call("POST", "/api/runs", {"requirement_file": "REQ-101.md", "provider": "mock",
                                              "scenario": "fail-then-fix", **kw})
    assert code == 202, r
    return r["id"]


# ------------------------------------------------------------------ the happy path through the UI's API
def test_full_flow_with_failed_check_then_human_gates(srv):
    rid = start_run(srv)
    s = wait_for(srv, rid, {"AWAITING_PLAN_APPROVAL"})
    assert s["action"] == "plan" and s["design"]["allowed_paths"] == ["shop/pricing.py"]
    assert srv.call("POST", f"/api/runs/{rid}/plan", {"decision": "approve", "by": "vasu"})[0] == 200
    s = wait_for(srv, rid, {"AWAITING_FINAL_APPROVAL"})
    assert s["action"] == "merge" and s["evidence"]["intact"] is True and s["rubric"]["score"] == 100
    assert any(e["kind"] == "attempt_failed" for e in s["events"])             # the failed check is visible
    assert "+++ b/shop/pricing.py" in s["diff"] and "17 passed" in s["review"]
    assert s["usage"]["calls"] >= 8 and all("tokens_in" in st for st in s["steps"])
    code, r = srv.call("POST", f"/api/runs/{rid}/merge", {"decision": "approve", "by": "vasu", "note": "ok"})
    assert code == 200 and r["state"] == "MERGED_AWAITING_DEPLOY_APPROVAL"
    code, r = srv.call("POST", f"/api/runs/{rid}/deploy", {"decision": "approve", "by": "vasu"})
    assert r["state"] == "DEPLOYED_MOCK"
    _, s = srv.call("GET", f"/api/runs/{rid}")
    assert [a["gate"] for a in s["approvals"]] == ["plan", "merge", "deploy"] and s["terminal"]


def test_list_and_config(srv):
    rid = start_run(srv)
    _, rows = srv.call("GET", "/api/runs")
    assert rows[0]["id"] == rid
    _, cfg = srv.call("GET", "/api/config")
    assert "REQ-101.md" in cfg["requirements"] and "fail-then-fix" in cfg["scenarios"]
    assert "key" not in json.dumps(cfg).lower().replace("key_present", "")


def test_blocked_run_shows_reason_and_offers_no_action(srv):
    rid = start_run(srv, scenario="deadlock")
    wait_for(srv, rid, {"AWAITING_PLAN_APPROVAL"})
    srv.call("POST", f"/api/runs/{rid}/plan", {"decision": "approve", "by": "vasu"})
    s = wait_for(srv, rid, {"ESCALATED_TO_HUMAN"})
    assert s["action"] is None and "deadlocked" in s["halt_reason"] and s["terminal"]


def test_tampering_is_visible_in_the_api_and_refused(srv):
    rid = start_run(srv)
    wait_for(srv, rid, {"AWAITING_PLAN_APPROVAL"})
    srv.call("POST", f"/api/runs/{rid}/plan", {"decision": "approve", "by": "vasu"})
    s = wait_for(srv, rid, {"AWAITING_FINAL_APPROVAL"})
    with open(srv.dash.runs / rid / "REVIEW.md", "a", encoding="utf-8") as f:
        f.write("edited\n")
    _, s = srv.call("GET", f"/api/runs/{rid}")
    assert s["evidence"]["intact"] is False
    code, r = srv.call("POST", f"/api/runs/{rid}/merge", {"decision": "approve", "by": "vasu"})
    assert code == 409 and "evidence changed" in r["error"]


# ------------------------------------------------------------------ decisions are validated
def test_decisions_need_a_name_and_the_right_state(srv):
    rid = start_run(srv)
    wait_for(srv, rid, {"AWAITING_PLAN_APPROVAL"})
    assert srv.call("POST", f"/api/runs/{rid}/plan", {"decision": "approve", "by": ""})[0] == 409
    assert srv.call("POST", f"/api/runs/{rid}/plan", {"decision": "maybe", "by": "v"})[0] == 400
    assert srv.call("POST", f"/api/runs/{rid}/merge", {"decision": "approve", "by": "v"})[0] == 409   # too early
    assert srv.call("POST", f"/api/runs/{rid}/deploy", {"decision": "approve", "by": "v"})[0] == 409
    assert srv.call("POST", f"/api/runs/{rid}/resume", {})[0] == 409            # not paused


def test_double_approval_is_refused(srv):
    rid = start_run(srv)
    wait_for(srv, rid, {"AWAITING_PLAN_APPROVAL"})
    assert srv.call("POST", f"/api/runs/{rid}/plan", {"decision": "approve", "by": "a"})[0] == 200
    code, _ = srv.call("POST", f"/api/runs/{rid}/plan", {"decision": "approve", "by": "b"})
    assert code == 409                                  # busy, or already past the plan gate
    wait_for(srv, rid, {"AWAITING_FINAL_APPROVAL"})
    assert srv.call("POST", f"/api/runs/{rid}/plan", {"decision": "reject", "by": "b"})[0] == 409


def test_paused_run_can_be_resumed_from_the_ui(srv):
    rid = start_run(srv, scenario="outage")
    wait_for(srv, rid, {"AWAITING_PLAN_APPROVAL"})
    srv.call("POST", f"/api/runs/{rid}/plan", {"decision": "approve", "by": "v"})
    s = wait_for(srv, rid, {"PAUSED_PROVIDER_DOWN"})
    assert s["action"] == "resume"
    # resuming re-uses the run's own (still outage-scripted) provider here, so it pauses again:
    assert srv.call("POST", f"/api/runs/{rid}/resume", {})[0] == 200
    time.sleep(0.5)
    assert wait_for(srv, rid, {"PAUSED_PROVIDER_DOWN"})["action"] == "resume"


# ------------------------------------------------------------------ input validation
def test_start_run_validation(srv):
    c = srv.call
    assert c("POST", "/api/runs", {"requirement_file": "../../etc/passwd"})[0] == 400
    assert c("POST", "/api/runs", {"requirement_file": "REQ-101.md", "provider": "openai"})[0] == 400
    assert c("POST", "/api/runs", {"requirement_file": "REQ-101.md", "scenario": "nope"})[0] == 400
    assert c("POST", "/api/runs", {})[0] == 400
    assert c("POST", "/api/runs", {"requirement_text": "x" * 7000})[0] == 400
    assert c("POST", "/api/runs", raw="[1,2]")[0] == 400
    assert c("POST", "/api/runs", raw="{not json")[0] == 400


def test_gemini_requires_a_key_in_the_environment(srv, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    code, r = srv.call("POST", "/api/runs", {"requirement_file": "REQ-101.md", "provider": "gemini"})
    assert code == 400 and "GEMINI_API_KEY" in r["error"]


# ------------------------------------------------------------------ security properties
def test_post_needs_the_per_session_token(srv):
    assert srv.call("POST", "/api/runs", {"requirement_file": "REQ-101.md"}, token=None)[0] == 403
    assert srv.call("POST", "/api/runs", {"requirement_file": "REQ-101.md"}, token="wrong")[0] == 403


@pytest.mark.parametrize("host", ["evil.example.com", "127.0.0.1:1", "localhost", "attacker.test:80"])
def test_foreign_host_headers_are_rejected(srv, host):          # DNS-rebinding defence
    assert srv.call("GET", "/api/runs", host=host)[0] == 403
    assert srv.call("GET", "/", host=host)[0] == 403
    assert srv.call("POST", "/api/runs", {"requirement_file": "REQ-101.md"}, host=host)[0] == 403


def test_oversized_body_is_rejected(srv):
    assert srv.call("POST", "/api/runs", raw="x" * 20_000)[0] == 413


@pytest.mark.parametrize("rid", ["../state.db", "..%2f..%2fetc", "a%20b", "x" * 60, ".."])
def test_run_ids_cannot_traverse(srv, rid):
    assert srv.call("GET", f"/api/runs/{rid}")[0] == 404
    assert srv.call("POST", f"/api/runs/{rid}/merge", {"decision": "approve", "by": "v"})[0] == 404


def test_unknown_run_and_paths_404(srv):
    assert srv.call("GET", "/api/runs/run-nope")[0] == 404
    assert srv.call("GET", "/etc/passwd")[0] == 404
    assert srv.call("GET", "/dashboard.py")[0] == 404           # source files are not served


def test_page_has_csp_nonce_and_never_uses_innerhtml(srv):
    port = srv.server_address[1]
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("GET", "/")
    r = c.getresponse()
    csp = r.getheader("Content-Security-Policy")
    page = r.read().decode()
    assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp and "connect-src 'self'" in csp
    nonce = csp.split("'nonce-")[1].split("'")[0]
    assert f'nonce="{nonce}"' in page and "__NONCE__" not in page and "__TOKEN__" not in page
    assert "innerHTML" not in page and "outerHTML" not in page and "document.write" not in page
    assert "https://" not in page and "http://" not in page.replace("http://127", "")   # no external resources


def test_hostile_requirement_text_is_stored_as_data(srv):
    from orchestrator import service
    evil = "<img src=x onerror=alert(1)> Add a bulk discount of 10% for 10+ units"
    # created directly: the dashboard (rightly) refuses mock + foreign text, but the API must still
    # return any stored text purely as JSON data, which the UI renders via textContent
    rid = service.create_run(srv.dash.store(), srv.dash.runs, evil, "REQ-ADHOC", provider="mock",
                             run_id="hostile1", strict_mock=False)
    _, s = srv.call("GET", f"/api/runs/{rid}")
    assert s["requirement"] == evil


def test_a_run_cannot_be_started_twice_concurrently(srv):
    rid = start_run(srv)
    wait_for(srv, rid, {"AWAITING_PLAN_APPROVAL"})
    results = []

    def approve():
        results.append(srv.call("POST", f"/api/runs/{rid}/plan", {"decision": "approve", "by": "v"})[0])
    ts = [threading.Thread(target=approve) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results.count(200) == 1 and results.count(409) == 3
    wait_for(srv, rid, {"AWAITING_FINAL_APPROVAL"})
    _, s = srv.call("GET", f"/api/runs/{rid}")
    assert [a["gate"] for a in s["approvals"]].count("plan") == 1


def test_mock_refuses_requirements_it_cannot_actually_run(srv):
    code, r = srv.call("POST", "/api/runs", {"requirement_text": "Add evaluate_formula()", "provider": "mock"})
    assert code == 400 and "REQ-101" in r["error"]
    assert srv.call("GET", "/api/runs")[1] == []                 # no misleading run was registered
    assert srv.call("POST", "/api/runs", {"requirement_file": "REQ-101.md", "provider": "mock"})[0] == 202
