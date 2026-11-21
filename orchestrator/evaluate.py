"""Compare agent output to a held-out baseline (differential test) and a scored rubric.

A baseline file is a drop-in replacement for one target file and declares:
    TARGET   = "shop/pricing.py"     file it replaces
    FUNCTION = "line_total"          function compared
    def cases(): yield argument tuples
It is written by a human and is never shown to any agent."""
import json
import re
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).with_name("_diff_runner.py")


def baseline_meta(baseline_path):
    """Read TARGET without importing the file."""
    m = re.search(r'^TARGET\s*=\s*["\']([^"\']+)["\']', Path(baseline_path).read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else None


def differential(sandbox_root, baseline_path):
    target = baseline_meta(baseline_path)
    if not target:
        return {"name": "baseline_diff", "passed": False, "summary": "baseline file declares no TARGET"}
    try:
        r = subprocess.run([sys.executable, "-I", str(RUNNER), str(Path(sandbox_root, target)), str(baseline_path)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        data = json.loads(r.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return {"name": "baseline_diff", "passed": False, "summary": "baseline runner failed or timed out"}
    if "error" in data:
        return {"name": "baseline_diff", "passed": False, "summary": data["error"], "mismatches": -1}
    return {"name": "baseline_diff", "passed": data["mismatches"] == 0, "cases": data["total"],
            "mismatches": data["mismatches"], "examples": data["examples"],
            "summary": f"{data['total'] - data['mismatches']}/{data['total']} cases match baseline"}


def rubric(changed, allowed, checks, diff_text):
    """Weighted rubric, 0-100, pass mark 85. Every item is computed from tool output, never
    self-reported by a model. Without a baseline, its weight is dropped and the rest rescaled."""
    by = {c["name"]: c for c in checks}
    added = sum(1 for l in diff_text.splitlines() if l.startswith("+") and not l.startswith("+++"))
    in_scope = all(any(Path(c).match(a) or c == a for a in allowed) for c in changed if not c.startswith("tests/"))
    items = [
        ("unit tests pass", 30, by.get("unit_tests", {}).get("passed", False)),
        ("matches baseline", 25, by["baseline_diff"]["passed"] if "baseline_diff" in by else None),
        ("no critical security findings", 20, by.get("security_scan", {}).get("passed", False)),
        ("lint clean", 10, by.get("lint", {}).get("passed", False)),
        ("change stays in approved scope", 10, in_scope),
        ("tests added with the change", 5, any(c.startswith("tests/") for c in changed)),
    ]
    live = [(n, w, ok) for n, w, ok in items if ok is not None]
    total = sum(w for _, w, _ in live)
    score = round(100 * sum(w for _, w, ok in live if ok) / total)
    return {"score": score, "pass_mark": 85, "passed": score >= 85, "lines_added": added,
            "items": [{"criterion": n, "weight": w, "met": bool(ok)} for n, w, ok in live]
                     + [{"criterion": n, "weight": w, "met": None} for n, w, ok in items if ok is None]}
