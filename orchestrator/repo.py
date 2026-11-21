"""Read-only repository inspection used to build agent inputs."""
import ast
from pathlib import Path

MAX_SOURCE_BYTES = 40_000


def python_sources(root):
    """Production .py files (not tests, not hidden dirs, size-capped) as sorted posix paths."""
    out = []
    for p in Path(root).rglob("*.py"):
        rel = p.relative_to(root)
        parts = rel.parts
        if any(x.startswith(".") or x in ("__pycache__", "tests", "test") for x in parts):
            continue
        if p.name.startswith("test_") or p.name.endswith("_test.py") or p.stat().st_size > MAX_SOURCE_BYTES:
            continue
        out.append(rel.as_posix())
    return sorted(out)


def api_summary(root, rels):
    """Public interface only (signatures + first docstring line). No function bodies: QA writes
    tests from the acceptance criteria and this contract, never from the implementation."""
    summary = []
    for rel in rels:
        try:
            tree = ast.parse(Path(root, rel).read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        items = []

        def sig(fn):
            ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
            return f"def {fn.name}({ast.unparse(fn.args)}){ret}"

        def doc(n):
            d = ast.get_docstring(n)
            return d.splitlines()[0] if d else ""

        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_"):
                items.append({"signature": sig(n), "doc": doc(n)})
            elif isinstance(n, ast.ClassDef) and not n.name.startswith("_"):
                items.append({"signature": f"class {n.name}", "doc": doc(n)})
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_"):
                        items.append({"signature": "    " + sig(m), "doc": doc(m)})
        summary.append({"path": rel, "module": rel[:-3].replace("/", "."), "public_api": items})
    return summary


def read_sources(root, rels, budget=60_000):
    """File contents for the architect, bounded so prompts stay small."""
    out, used = {}, 0
    for rel in rels:
        text = Path(root, rel).read_text(encoding="utf-8")
        if used + len(text) > budget:
            break
        out[rel] = text
        used += len(text)
    return out
