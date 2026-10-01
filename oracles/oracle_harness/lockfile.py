"""The lock file: every pinned third-party repository and toolchain.

``oracles.lock.toml`` is the single source of truth. ``.gitmodules``, the
gitlinks in the index, the checked-out submodules, PROVENANCE.md and the golden
observations are all checked against it, so a pin cannot drift in one place
without the others noticing.
"""

from __future__ import annotations

import configparser
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import gitutil, paths
from .util import Finding, HarnessError, load_toml, sha256_bytes

SCHEMA = 1
ROLES = ("oracle", "corpus-source")
TIERS = ("commit", "nightly")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Pin:
    id: str
    role: str
    upstream: str
    url: str
    submodule: str
    path: str  # repository-relative, forward slashes
    commit: str
    tree: str
    ref: str
    committed: str
    licence: str
    licence_file: str
    licence_sha256: str
    summary: str
    used_as: str
    tier: str = ""  # oracles only: the cheapest CI tier that runs it
    notes: tuple[str, ...] = ()
    dependency_names: tuple[str, ...] = ()  # package-manager names the core must never depend on

    @property
    def checkout(self) -> Path:
        return paths.REPO_ROOT / self.path


@dataclass(frozen=True)
class Lock:
    pins: dict[str, Pin]
    toolchains: dict[str, dict] = field(default_factory=dict)

    def pin(self, pin_id: str) -> Pin:
        try:
            return self.pins[pin_id]
        except KeyError:
            raise HarnessError(f"no pinned repository named '{pin_id}' in {paths.LOCK_FILE.name}") from None

    def oracles(self) -> list[Pin]:
        return [pin for pin in self.pins.values() if pin.role == "oracle"]


def _require(table: dict, key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"{where}: missing or empty '{key}'")
    return value.strip()


def load(path: Path | None = None) -> Lock:
    path = path or paths.LOCK_FILE
    data = load_toml(path)
    if data.get("schema") != SCHEMA:
        raise HarnessError(f"{path.name}: expected schema = {SCHEMA}")
    pins: dict[str, Pin] = {}
    for pin_id, table in (data.get("repo") or {}).items():
        where = f"{path.name} [repo.{pin_id}]"
        role = _require(table, "role", where)
        if role not in ROLES:
            raise HarnessError(f"{where}: role must be one of {ROLES}")
        tier = table.get("tier", "")
        if role == "oracle" and tier not in TIERS:
            raise HarnessError(f"{where}: an oracle needs tier = one of {TIERS}")
        commit = _require(table, "commit", where)
        tree = _require(table, "tree", where)
        if not _SHA1.match(commit) or not _SHA1.match(tree):
            raise HarnessError(f"{where}: commit and tree must be full 40-hex object ids")
        licence_sha256 = _require(table, "licence_sha256", where)
        if not _SHA256.match(licence_sha256):
            raise HarnessError(f"{where}: licence_sha256 must be 64 hex characters")
        pins[pin_id] = Pin(
            id=pin_id,
            role=role,
            upstream=_require(table, "upstream", where),
            url=_require(table, "url", where),
            submodule=_require(table, "submodule", where),
            path=_require(table, "path", where).replace("\\", "/"),
            commit=commit,
            tree=tree,
            ref=_require(table, "ref", where),
            committed=_require(table, "committed", where),
            licence=_require(table, "licence", where),
            licence_file=_require(table, "licence_file", where),
            licence_sha256=licence_sha256,
            summary=_require(table, "summary", where),
            used_as=_require(table, "used_as", where),
            tier=tier,
            notes=tuple(table.get("notes", ())),
            dependency_names=tuple(table.get("dependency_names", ())),
        )
    if not pins:
        raise HarnessError(f"{path.name}: no [repo.*] entries")
    return Lock(pins=pins, toolchains=dict(data.get("toolchain") or {}))


def normalise_url(url: str) -> str:
    """Host and path of a repository URL, so https/ssh/.git spellings compare equal."""
    text = url.strip().lower()
    text = re.sub(r"^[a-z+]+://", "", text)
    text = re.sub(r"^git@([^:]+):", r"\1/", text)
    text = re.sub(r"^[^@/]+@", "", text)
    return re.sub(r"\.git$", "", text.rstrip("/"))


def read_gitmodules(repo_root: Path) -> dict[str, dict[str, str]]:
    path = repo_root / ".gitmodules"
    if not path.exists():
        return {}
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    # .gitmodules indents its keys with tabs; configparser would read that as a
    # continuation line, so the indentation is stripped first.
    text = "\n".join(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    parser.read_string(text)
    modules = {}
    for section in parser.sections():
        match = re.fullmatch(r'submodule "(.+)"', section)
        if match:
            modules[match.group(1)] = dict(parser[section])
    return modules


def verify(lock: Lock, repo_root: Path | None = None) -> list[Finding]:
    """Check that the repository really is pinned the way the lock file says."""
    repo_root = repo_root or paths.REPO_ROOT
    findings: list[Finding] = []
    modules = read_gitmodules(repo_root)
    by_name = {pin.submodule: pin for pin in lock.pins.values()}

    oracle_urls = {normalise_url(pin.url) for pin in lock.pins.values()}
    for name, module in modules.items():
        if name in by_name:
            continue
        # The lock file pins what the harness uses. A submodule of the engine's
        # own is the engine's business - unless it is one of the oracles again.
        path = module.get("path", "").replace("\\", "/")
        if path.startswith(paths.THIRD_PARTY_REL + "/") or normalise_url(module.get("url", "")) in oracle_urls:
            findings.append(Finding("unlocked-submodule", f"submodule '{name}' ({module.get('path')}) has no entry in the lock file"))
    for pin in lock.pins.values():
        module = modules.get(pin.submodule)
        if module is None:
            findings.append(Finding("missing-submodule", f"{pin.id}: .gitmodules has no submodule '{pin.submodule}'"))
            continue
        if module.get("path", "").replace("\\", "/") != pin.path:
            findings.append(Finding("submodule-path", f"{pin.id}: .gitmodules path '{module.get('path')}' != locked path '{pin.path}'"))
        if normalise_url(module.get("url", "")) != normalise_url(pin.url):
            findings.append(Finding("submodule-url", f"{pin.id}: .gitmodules url '{module.get('url')}' != locked url '{pin.url}'"))
        if not pin.path.startswith(paths.THIRD_PARTY_REL + "/"):
            findings.append(Finding("submodule-placement", f"{pin.id}: '{pin.path}' is outside {paths.THIRD_PARTY_REL}/"))

        gitlink = gitutil.index_gitlink(repo_root, pin.path)
        if gitlink is None:
            findings.append(Finding("no-gitlink", f"{pin.id}: '{pin.path}' is not a submodule gitlink in the index"))
        elif gitlink != pin.commit:
            findings.append(Finding("gitlink-mismatch", f"{pin.id}: index pins {gitlink[:12]}, lock file pins {pin.commit[:12]}"))

        checkout = repo_root / pin.path
        if not gitutil.is_checkout(checkout):
            findings.append(Finding(
                "not-initialised",
                f"{pin.id}: '{pin.path}' is not checked out (run: python {paths.HARNESS_REL}/harness.py sync)"))
            continue
        head = gitutil.head_commit(checkout)
        if head != pin.commit:
            findings.append(Finding("checkout-mismatch", f"{pin.id}: checked out at {head[:12]}, lock file pins {pin.commit[:12]}"))
        if not gitutil.has_commit(checkout, pin.commit):
            findings.append(Finding("missing-commit", f"{pin.id}: locked commit {pin.commit[:12]} is not in the submodule's object store"))
            continue
        changes = gitutil.worktree_changes(checkout)
        if changes:
            findings.append(Finding(
                "dirty-submodule",
                f"{pin.id}: '{pin.path}' has local changes ({len(changes)}; first: {changes[0].strip()}) - "
                "third_party is read-only"))
        tree = gitutil.tree_of(checkout, pin.commit)
        if tree != pin.tree:
            # In full: after a pin bump this is the value the lock file needs.
            findings.append(Finding("tree-mismatch", f"{pin.id}: the commit's tree is {tree}, the lock file says {pin.tree}"))
        if gitutil.blob_id(checkout, pin.commit, pin.licence_file) is None:
            findings.append(Finding("licence-missing", f"{pin.id}: no '{pin.licence_file}' at the pinned commit"))
        else:
            digest = sha256_bytes(gitutil.read_path(checkout, pin.commit, pin.licence_file))
            if digest != pin.licence_sha256:
                findings.append(Finding(
                    "licence-changed",
                    f"{pin.id}: '{pin.licence_file}' now hashes to {digest}, not the recorded {pin.licence_sha256} - "
                    "re-read the licence, then update the lock file and PROVENANCE.md"))
    return findings


def sync(lock: Lock, repo_root: Path | None = None) -> None:
    """Initialise every pinned submodule at its locked commit.

    Deliberately not recursive: the oracles' own submodules are not needed, and
    one of them is declared with an SSH URL that fails without GitHub keys.
    """
    repo_root = repo_root or paths.REPO_ROOT
    for pin in lock.pins.values():
        gitutil.git(repo_root, ["submodule", "update", "--init", "--", pin.path])
        checkout = repo_root / pin.path
        if gitutil.head_commit(checkout) != pin.commit:
            if not gitutil.has_commit(checkout, pin.commit):
                gitutil.git(checkout, ["fetch", "--quiet", "origin", pin.commit])
            gitutil.git(checkout, ["-c", "advice.detachedHead=false", "checkout", "--quiet", pin.commit])
