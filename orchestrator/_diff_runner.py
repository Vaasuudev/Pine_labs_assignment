"""Subprocess helper: run candidate vs baseline over the baseline's declared cases.
Invoked as: python -I _diff_runner.py <candidate.py> <baseline.py>. Prints one JSON line."""
import importlib.util
import json
import sys


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def call(fn, args):
    try:
        return ["ok", fn(*args)]
    except Exception as e:  # noqa: BLE001 - candidate code is untrusted
        return ["raises", type(e).__name__]


def main(cand_path, base_path):
    try:
        cand, ref = load(cand_path, "candidate_mod"), load(base_path, "baseline_mod")
        fn_name = ref.FUNCTION
        cfn, rfn = getattr(cand, fn_name), getattr(ref, fn_name)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"cannot load candidate: {type(e).__name__}: {e}"}))
        return
    total, bad = 0, []
    for args in ref.cases():
        total += 1
        got, want = call(cfn, args), call(rfn, args)
        if got != want:
            bad.append([list(args), got, want])
    print(json.dumps({"total": total, "mismatches": len(bad), "examples": bad[:3]}))


main(sys.argv[1], sys.argv[2])
