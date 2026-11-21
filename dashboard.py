"""Local review dashboard (standard library only).

Security model: binds to loopback; every request must carry a loopback Host header (blocks DNS
rebinding); every POST needs a per-process token that only the served page knows (blocks CSRF);
JSON bodies are size-capped; the UI never uses innerHTML; a CSP forbids everything but the
page's own nonce-tagged script. It is a single-reviewer tool: reviewer names are self-reported."""
import json
import os
import re
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from orchestrator import service
from orchestrator.workflow import clean_name

HERE = Path(__file__).resolve().parent
MAX_BODY = 16 * 1024
RUN_PATH = re.compile(r"^/api/runs/([A-Za-z0-9_-]{1,40})(?:/(plan|merge|deploy|resume))?$")


class Dashboard:
    def __init__(self, runs, repo):
        self.runs, self.repo = Path(runs), Path(repo)
        self.token = secrets.token_urlsafe(24)
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.active, self.lock = set(), threading.Lock()
        self.html = (HERE / "dashboard.html").read_text(encoding="utf-8")
        service.open_store(self.runs)        # create the DB up front

    # one Store per thread/request: sqlite connections are not shared across threads
    def store(self):
        return service.open_store(self.runs)

    def claim(self, run_id):
        with self.lock:
            if run_id in self.active:
                return False
            self.active.add(run_id)
            return True

    def release(self, run_id):
        with self.lock:
            self.active.discard(run_id)

    def submit(self, run_id, fn):
        """Run fn(workflow) in the background; at most one job per run at a time."""
        def job():
            store = self.store()
            try:
                fn(service.workflow(store, run_id, self.runs))
            except PermissionError as e:
                store.event(run_id, "refused", str(e))
            except Exception as e:  # noqa: BLE001 - never leave a run looking 'working' forever
                store.event(run_id, "internal_error", f"{type(e).__name__}: {e}")
                store.set_state(run_id, "FAILED_INTERNAL")
            finally:
                self.release(run_id)
        self.pool.submit(job)

    # ------------------------------------------------------------ API
    def config(self):
        key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        return {"gemini": {"key_present": key, "model": os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite"},
                "scenarios": list(service.SCENARIOS), "requirements": service.requirement_files(),
                "repo": self.repo.name}

    def start_run(self, body):
        provider = body.get("provider", "mock")
        if provider not in service.PROVIDERS:
            raise ValueError("unknown provider")
        if provider == "gemini" and not os.environ.get("GEMINI_API_KEY", "").strip():
            raise ValueError("GEMINI_API_KEY is not set in the environment that started the dashboard")
        name, text = body.get("requirement_file"), body.get("requirement_text")
        if name:
            if name not in service.requirement_files():
                raise ValueError("unknown requirement file")
            text, req_id = (service.ROOT / "requirements" / name).read_text(encoding="utf-8"), Path(name).stem
        elif isinstance(text, str):
            req_id = "REQ-ADHOC"
        else:
            raise ValueError("provide requirement_file or requirement_text")
        store = self.store()
        run_id = service.create_run(store, self.runs, text, req_id, self.repo, provider,
                                    body.get("scenario", "fail-then-fix"))
        self.claim(run_id)
        self.submit(run_id, lambda wf: wf.advance())
        return {"id": run_id}

    def act(self, run_id, action, body):
        store = self.store()
        store.get_run(run_id)                                   # KeyError -> 404
        decision = body.get("decision")
        if action != "resume" and decision not in ("approve", "reject"):
            raise ValueError("decision must be 'approve' or 'reject'")
        if not self.claim(run_id):
            raise PermissionError("this run is busy; wait for the current step to finish")
        try:
            wf = service.workflow(store, run_id, self.runs)
            by, note = body.get("by", ""), str(body.get("note", ""))[:500]
            if action == "plan" and decision == "approve":
                wf._gate("plan_gate", "AWAITING_PLAN_APPROVAL")  # validate now, run in the background
                clean_name(by)
                self.submit(run_id, lambda w: w.decide_plan("approve", by, note))
                return {"accepted": True}                        # job releases the claim
            if action == "resume":
                if store.get_run(run_id)["state"] != "PAUSED_PROVIDER_DOWN":
                    raise PermissionError("only a paused run can be resumed")
                self.submit(run_id, lambda w: w.advance())
                return {"accepted": True}
            fn = {"plan": wf.decide_plan, "merge": wf.decide_final, "deploy": wf.decide_deploy}[action]
            result = fn(decision, by, note)
            self.release(run_id)
            return {"state": result}
        except BaseException:
            self.release(run_id)
            raise


def make_handler(dash, port_holder):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AgenticConsole"

        def log_message(self, *a):
            pass

        def _send(self, code, data, mime="application/json", extra=None):
            raw = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _host_ok(self):
            port = port_holder["port"]
            return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, {"error": "forbidden host"})
            path = self.path.split("?")[0]
            try:
                if path == "/":
                    nonce = secrets.token_urlsafe(12)
                    page = dash.html.replace("__TOKEN__", dash.token).replace("__NONCE__", nonce)
                    csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; "
                           "connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; "
                           "form-action 'none'")
                    return self._send(200, page.encode(), "text/html; charset=utf-8",
                                      {"Content-Security-Policy": csp})
                if path == "/api/config":
                    return self._send(200, dash.config())
                if path == "/api/runs":
                    rows = [{"id": r["id"], "state": r["state"], "created": r["created"], "updated": r["updated"],
                             "requirement": r["requirement"][:90]} for r in dash.store().list_runs()]
                    return self._send(200, rows)
                m = RUN_PATH.match(path)
                if m and not m.group(2):
                    return self._send(200, service.summary(dash.store(), m.group(1), dash.runs))
                return self._send(404, {"error": "not found"})
            except KeyError:
                return self._send(404, {"error": "unknown run"})

        def do_POST(self):
            if not self._host_ok():
                return self._send(403, {"error": "forbidden host"})
            if not secrets.compare_digest(self.headers.get("X-Dashboard-Token", ""), dash.token):
                return self._send(403, {"error": "missing or invalid token"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._send(400, {"error": "bad content length"})
            if n > MAX_BODY:
                return self._send(413, {"error": "request too large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
                path = self.path.split("?")[0]
                if path == "/api/runs":
                    return self._send(202, dash.start_run(body))
                m = RUN_PATH.match(path)
                if m and m.group(2):
                    return self._send(200, dash.act(m.group(1), m.group(2), body))
                return self._send(404, {"error": "not found"})
            except KeyError:
                return self._send(404, {"error": "unknown run"})
            except PermissionError as e:
                return self._send(409, {"error": str(e)})
            except ValueError as e:
                return self._send(400, {"error": str(e)})

    return Handler


def serve(port, runs, repo):
    """Build (but do not start) the server. port=0 picks a free port (used by tests)."""
    dash = Dashboard(runs, repo)
    holder = {"port": port}
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(dash, holder))
    holder["port"] = server.server_address[1]
    server.dash = dash
    return server


def main(port, runs, repo):
    server = serve(port, runs, repo)
    print(f"Agentic SDLC console: http://127.0.0.1:{server.server_address[1]}   (Ctrl+C to stop)")
    print(f"Gemini key in environment: {'yes' if os.environ.get('GEMINI_API_KEY', '').strip() else 'no (mock only)'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
