"""Branch-isolated sandbox. The agent never touches the source repo: it works on a throwaway
copy with its own git history and a dedicated branch. Writes are gated by per-agent path
allowlists, a protected-path denylist (built-ins + the repo's own .agentignore), a size cap,
and an idempotency key per action.

NOTE: this is isolation of *writes*, not an OS sandbox. Generated code is executed by the
test runner on the host; production use needs a container with no network and resource limits."""
import fnmatch
import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path

BASE_PROTECTED = [".git/*", ".git", "pytest.ini", "*.env", ".env", ".agentignore", ".github/*",
                  "requirements/*", "requirements.txt"]
MAX_FILE_BYTES = 60_000


class ScopeViolation(Exception):
    pass


class DuplicateAction(Exception):
    pass


def git(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def _rmtree(path):
    def onerror(func, p, _exc):          # Windows: .git objects are read-only
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onerror=onerror)


class Sandbox:
    def __init__(self, source_repo, workdir, branch, store, run_id):
        self.source, self.root, self.branch = Path(source_repo), Path(workdir), branch
        self.store, self.run_id = store, run_id
        self.protected = list(BASE_PROTECTED)

    def load_policy(self):
        """Protected paths declared by the repo owners (one glob per line, like CODEOWNERS)."""
        f = self.root / ".agentignore"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    self.protected.append(line)

    def create(self):
        if self.root.exists():
            _rmtree(self.root)
        shutil.copytree(self.source, self.root, ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", ".git"))
        (self.root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")
        git(self.root, "init", "-q", "-b", "main")
        git(self.root, "config", "user.email", "agents@example.invalid")
        git(self.root, "config", "user.name", "agent-bot")
        git(self.root, "config", "core.autocrlf", "false")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "baseline")
        git(self.root, "checkout", "-q", "-b", self.branch)
        self.load_policy()

    # ------------------------------------------------------------ write policy
    def _check(self, rel, allowed):
        if not isinstance(rel, str) or not rel or "\x00" in rel:
            raise ScopeViolation(f"invalid path: {rel!r}")
        p = (self.root / rel).resolve()
        try:
            rel = p.relative_to(self.root.resolve()).as_posix()     # posix on every OS
        except ValueError:
            raise ScopeViolation(f"path escapes sandbox: {rel}") from None
        if any(fnmatch.fnmatchcase(rel, pat) for pat in self.protected):
            raise ScopeViolation(f"protected path: {rel}")
        if not any(fnmatch.fnmatchcase(rel, pat) for pat in allowed):
            raise ScopeViolation(f"{rel} is outside the agent's approved scope {allowed}")
        return p, rel

    @staticmethod
    def _key(agent, rel, content):
        return hashlib.sha256(f"{agent}|{rel}|{content}".encode()).hexdigest()[:16]

    def preflight(self, agent, files, allowed, must_be_new=False):
        """Validate every proposed write BEFORE touching disk: scope, protected paths, size,
        repeats. Claims nothing, so a rejected attempt leaves no side effects."""
        problems = []
        for f in files:
            try:
                p, rel = self._check(f["path"], allowed)
            except ScopeViolation as e:
                problems.append(("scope", str(e)))
                continue
            if len(f["content"].encode()) > MAX_FILE_BYTES:
                problems.append(("scope", f"{rel} exceeds {MAX_FILE_BYTES} bytes"))
            elif must_be_new and p.exists():
                problems.append(("scope", f"{rel} already exists; {agent} may only add new files"))
            elif self.store.has_action(self.run_id, self._key(agent, rel, f["content"])):
                problems.append(("repeat", f"{agent} already wrote identical content to {rel}"))
        return problems

    def write(self, agent, rel, content, allowed):
        p, rel = self._check(rel, allowed)
        key = self._key(agent, rel, content)
        if not self.store.claim_action(self.run_id, key):
            raise DuplicateAction(f"{agent} already wrote identical content to {rel}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="\n")
        self.store.event(self.run_id, "file_write", f"{agent} -> {rel}")
        return rel

    def read(self, rel):
        return (self.root / rel).read_text(encoding="utf-8")

    # ------------------------------------------------------------ git
    def commit(self, msg):
        git(self.root, "add", "-A")
        if git(self.root, "status", "--porcelain").strip():
            git(self.root, "commit", "-q", "-m", msg)
        return self.head()

    def reset_hard(self, commit):
        git(self.root, "reset", "-q", "--hard", commit)

    def head(self):
        return git(self.root, "rev-parse", "--short=12", "HEAD").strip()

    def branch_head(self):
        return git(self.root, "rev-parse", "--short=12", self.branch).strip()

    def diff(self):
        """Committed change on the branch relative to main (what a human would merge)."""
        return git(self.root, "diff", "main.." + self.branch)

    def changed_files(self):
        out = git(self.root, "diff", "--name-only", "main")
        untracked = git(self.root, "ls-files", "--others", "--exclude-standard")
        return sorted({*out.split(), *untracked.split()})

    def merge_to_main(self, expected_commit):
        """Only ever called from the human-approval path. Merges exactly the reviewed commit."""
        if self.branch_head() != expected_commit:
            raise RuntimeError("branch moved since review")
        git(self.root, "checkout", "-q", "main")
        git(self.root, "merge", "--no-ff", "-q", "-m",
            f"Merge {self.branch} @ {expected_commit} (human approved)", expected_commit)
        return git(self.root, "rev-parse", "--short=12", "HEAD").strip()
