"""SQLite persistence: workflow state, per-agent execution history, approvals, audit events."""
import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY, requirement TEXT, state TEXT, created REAL, updated REAL, context TEXT);
CREATE TABLE IF NOT EXISTS steps(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, agent TEXT, attempt INTEGER,
  status TEXT, input TEXT, output TEXT, error TEXT, model TEXT, provider TEXT,
  tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, latency_ms REAL, ts REAL);
CREATE TABLE IF NOT EXISTS approvals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, gate TEXT, decision TEXT,
  decided_by TEXT, note TEXT, ts REAL, UNIQUE(run_id, gate));
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, kind TEXT, detail TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS actions(
  run_id TEXT, key TEXT, ts REAL, PRIMARY KEY(run_id, key));
"""


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path), timeout=15)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")      # dashboard reads while a run writes
        self.db.execute("PRAGMA busy_timeout=15000")
        self.db.executescript(SCHEMA)

    # runs
    def create_run(self, run_id, requirement):
        now = time.time()
        self.db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)",
                        (run_id, requirement, "CREATED", now, now, "{}"))
        self.db.commit()

    def set_state(self, run_id, state):
        self.db.execute("UPDATE runs SET state=?, updated=? WHERE id=?", (state, time.time(), run_id))
        self.db.commit()
        self.event(run_id, "state", state)

    def get_run(self, run_id):
        r = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not r:
            raise KeyError(run_id)
        d = dict(r)
        d["context"] = json.loads(d["context"])
        return d

    def save_context(self, run_id, ctx):
        self.db.execute("UPDATE runs SET context=?, updated=? WHERE id=?",
                        (json.dumps(ctx), time.time(), run_id))
        self.db.commit()

    # steps
    def record_step(self, run_id, agent, attempt, status, inp, out, error=None, usage=None):
        u = usage or {}
        self.db.execute(
            "INSERT INTO steps(run_id,agent,attempt,status,input,output,error,model,provider,"
            "tokens_in,tokens_out,cost_usd,latency_ms,ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, agent, attempt, status, json.dumps(inp, default=str)[:20000],
             json.dumps(out, default=str)[:20000], error, u.get("model"), u.get("provider"),
             u.get("tokens_in", 0), u.get("tokens_out", 0), u.get("cost_usd", 0.0),
             u.get("latency_ms", 0.0), time.time()))
        self.db.commit()

    def steps(self, run_id):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM steps WHERE run_id=? ORDER BY id", (run_id,))]

    def completed(self, run_id, agent, attempt):
        r = self.db.execute(
            "SELECT output FROM steps WHERE run_id=? AND agent=? AND attempt=? AND status='ok' "
            "ORDER BY id DESC LIMIT 1", (run_id, agent, attempt)).fetchone()
        return json.loads(r["output"]) if r else None

    # approvals / events / idempotency
    def approval(self, run_id, gate, decision, by, note=""):
        """One decision per gate, enforced by a UNIQUE constraint (also under concurrent reviewers)."""
        try:
            self.db.execute("INSERT INTO approvals(run_id,gate,decision,decided_by,note,ts) VALUES(?,?,?,?,?,?)",
                            (run_id, gate, decision, by, note, time.time()))
            self.db.commit()
        except sqlite3.IntegrityError:
            raise PermissionError(f"a '{gate}' decision has already been recorded for this run") from None
        self.event(run_id, "approval", f"{gate}:{decision} by {by}")

    def last_approval(self, run_id, gate):
        r = self.db.execute("SELECT * FROM approvals WHERE run_id=? AND gate=? ORDER BY id DESC LIMIT 1",
                            (run_id, gate)).fetchone()
        return dict(r) if r else None

    def event(self, run_id, kind, detail):
        self.db.execute("INSERT INTO events(run_id,kind,detail,ts) VALUES(?,?,?,?)",
                        (run_id, kind, str(detail)[:2000], time.time()))
        self.db.commit()

    def events(self, run_id):
        return [dict(r) for r in self.db.execute("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))]

    def claim_action(self, run_id, key):
        """Return True the first time an action key is seen for a run, False on any repeat."""
        try:
            self.db.execute("INSERT INTO actions VALUES(?,?,?)", (run_id, key, time.time()))
            self.db.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def approvals(self, run_id):
        return [dict(r) for r in self.db.execute("SELECT * FROM approvals WHERE run_id=? ORDER BY id", (run_id,))]

    def list_runs(self):
        return [dict(r) for r in self.db.execute(
            "SELECT id,requirement,state,created,updated FROM runs ORDER BY created DESC")]

    def has_action(self, run_id, key):
        return self.db.execute("SELECT 1 FROM actions WHERE run_id=? AND key=?", (run_id, key)).fetchone() is not None

    def usage_totals(self, run_id):
        r = self.db.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(tokens_in),0) ti, COALESCE(SUM(tokens_out),0) "
            "to_, COALESCE(SUM(cost_usd),0) c, COALESCE(SUM(latency_ms),0) l FROM steps WHERE run_id=?",
            (run_id,)).fetchone()
        return {"calls": r["n"], "tokens_in": r["ti"], "tokens_out": r["to_"],
                "cost_usd": round(r["c"], 6), "latency_ms": round(r["l"], 1)}
