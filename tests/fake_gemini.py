"""A local HTTP server that speaks Gemini's generateContent wire format, so the real
GeminiProvider code path (URL, headers, JSON mode, usage metadata, error codes) is exercised
without a network or API key. Replies come from the scripted mock behaviours; faults are queued."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from orchestrator import agents

PATH = re.compile(r"^/v1beta/models/([^/:]+):generateContent$")


class FakeGemini:
    def __init__(self, key="AIzaSyFAKEKEY-0123456789abcdefghijk", dev_faults=None):
        self.key = key
        self.behaviours = agents.behaviours(dev_faults or {})
        self.requests = []          # every request received: {"path","headers","body"}
        self.script = []            # queue of fault responses consumed before normal replies
        self.override = {}          # agent -> raw text to return instead of the scripted answer
        self.calls = {}
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, body, headers=None):
                raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "body": body})
                m = PATH.match(self.path)
                if not m:
                    return self._reply(404, {"error": {"status": "NOT_FOUND", "message": "no such model"}})
                if self.headers.get("x-goog-api-key") != outer.key:
                    return self._reply(400, {"error": {"status": "INVALID_ARGUMENT",
                                       "message": f"API key not valid: {self.headers.get('x-goog-api-key')}"}})
                if outer.script:
                    code, payload, hdrs = outer.script.pop(0)
                    return self._reply(code, payload, hdrs)
                system = body["systemInstruction"]["parts"][0]["text"]
                agent = re.search(r"# agent: (\w+)", system).group(1)
                payload = json.loads(body["contents"][0]["parts"][0]["text"])
                n = outer.calls[agent] = outer.calls.get(agent, 0) + 1
                text = outer.override.get(agent) or json.dumps(outer.behaviours[agent](payload, n))
                tin, tout = len(system + json.dumps(payload)) // 4, len(text) // 4
                self._reply(200, {"candidates": [{"content": {"parts": [{"text": text}], "role": "model"},
                                                  "finishReason": "STOP"}],
                                  "usageMetadata": {"promptTokenCount": tin, "candidatesTokenCount": tout,
                                                    "totalTokenCount": tin + tout}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def ok_body(text, **extra):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP", **extra}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}
