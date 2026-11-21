"""Deterministic quality gates. These are plain code, not model output: agents can *report* on
them but cannot overrule them."""
import ast
import re
import subprocess
import sys
from pathlib import Path

DANGEROUS_CALLS = {"eval": "critical", "exec": "critical", "compile": "high", "__import__": "high"}
DANGEROUS_ATTRS = {("os", "system"): "critical", ("subprocess", "Popen"): "high",
                   ("subprocess", "run"): "high", ("subprocess", "call"): "high",
                   ("pickle", "loads"): "critical", ("pickle", "load"): "critical"}
SECRET_RE = re.compile(r"(api[_-]?key|secret|passwd|password|token)\s*=\s*['\"][^'\"]{6,}['\"]", re.I)


def run_pytest(root, timeout=60):
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
                           cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"name": "unit_tests", "passed": False, "summary": "timeout", "output": ""}
    out = (r.stdout + r.stderr).strip()
    last = out.splitlines()[-1] if out else ""
    return {"name": "unit_tests", "passed": r.returncode == 0, "summary": last, "output": out[-3000:]}


def _py_files(root, rels):
    return [(rel, Path(root, rel)) for rel in rels if rel.endswith(".py") and Path(root, rel).exists()]


def run_lint(root, rels):
    """Syntax + unused imports + bare except. Stand-in for ruff/flake8 (not installed here)."""
    problems = []
    for rel, p in _py_files(root, rels):
        src = p.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append(f"{rel}:{e.lineno}: syntax error: {e.msg}")
            continue
        imported = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    imported[(a.asname or a.name).split(".")[0]] = n.lineno
            elif isinstance(n, ast.ImportFrom):
                if n.module == "__future__":      # compiler directives, never "unused"
                    continue
                for a in n.names:
                    imported[a.asname or a.name] = n.lineno
            elif isinstance(n, ast.ExceptHandler) and n.type is None:
                problems.append(f"{rel}:{n.lineno}: bare except")
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
               {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
        for name, line in imported.items():
            if name not in used and not rel.endswith("__init__.py"):
                problems.append(f"{rel}:{line}: unused import {name}")
    # advisory: a home-made linter must not be able to block a delivery on style alone
    return {"name": "lint", "passed": not problems, "advisory": True,
            "summary": f"{len(problems)} problem(s)", "output": "\n".join(problems)}


def security_scan(root, rels):
    findings = []
    for rel, p in _py_files(root, rels):
        src = p.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name) and f.id in DANGEROUS_CALLS:
                    findings.append({"severity": DANGEROUS_CALLS[f.id], "file": rel, "line": n.lineno,
                                     "issue": f"use of {f.id}() enables arbitrary code execution"})
                elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                        and (f.value.id, f.attr) in DANGEROUS_ATTRS:
                    findings.append({"severity": DANGEROUS_ATTRS[(f.value.id, f.attr)], "file": rel,
                                     "line": n.lineno, "issue": f"use of {f.value.id}.{f.attr}()"})
        for i, line in enumerate(src.splitlines(), 1):
            if SECRET_RE.search(line):
                findings.append({"severity": "critical", "file": rel, "line": i,
                                 "issue": "hard-coded credential"})
    crit = [f for f in findings if f["severity"] == "critical"]
    return {"name": "security_scan", "passed": not crit, "findings": findings,
            "summary": f"{len(findings)} finding(s), {len(crit)} critical"}
