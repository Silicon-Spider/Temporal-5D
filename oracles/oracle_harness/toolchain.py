"""Locating (and, only when asked, fetching) the tools that build the oracles.

Nothing here touches the machine outside the harness cache: a bootstrapped tool
is a pip wheel or a checksum-verified archive unpacked under ``.cache``, with no
PATH or registry changes.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .lockfile import Lock
from .util import HarnessError, check_run, exe, is_windows, platform_key, run, sha256_file


@dataclass(frozen=True)
class Tool:
    name: str
    path: Path
    version: str

    def __str__(self) -> str:
        return f"{self.name} {self.version} ({self.path})"


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def _first_line(argv: list) -> str | None:
    try:
        done = run(argv, timeout=60)
    except OSError:
        return None
    if done.timed_out or done.returncode != 0:
        return None
    text = (done.out_text() or done.err_text()).strip()
    return text.splitlines()[0] if text else None


def _tools_venv() -> Path:
    return paths.cache_dir() / "tools" / "venv"


def _venv_bin(name: str) -> Path:
    return _tools_venv() / ("Scripts" if is_windows() else "bin") / exe(name)


def _bootstrap_pip_tools(lock: Lock) -> None:
    requirements = [
        spec for spec in (lock.toolchains.get(tool, {}).get("bootstrap_pip") for tool in ("cmake", "ninja")) if spec
    ]
    venv = _tools_venv()
    if not _venv_bin("python").exists():
        check_run([sys.executable, "-m", "venv", venv])
    # No pip cache: nothing is left behind outside the harness cache.
    check_run([_venv_bin("python"), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-cache-dir",
               *requirements], timeout=900)


def _find_pip_tool(name: str, lock: Lock, bootstrap: bool) -> Tool | None:
    candidates = [shutil.which(name), _venv_bin(name)]
    for attempt in range(2):
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                line = _first_line([candidate, "--version"])
                if line:
                    version = ".".join(str(part) for part in _version_tuple(line))
                    return Tool(name, Path(candidate), version)
        if attempt == 0 and bootstrap:
            _bootstrap_pip_tools(lock)
        else:
            break
    return None


def find_cmake(lock: Lock, bootstrap: bool = False) -> Tool:
    tool = _find_pip_tool("cmake", lock, bootstrap)
    if tool is None:
        raise HarnessError("cmake not found. Install CMake, or rerun with --bootstrap to fetch the pinned wheel into the harness cache.")
    minimum = lock.toolchains.get("cmake", {}).get("min_version", "3.20")
    if _version_tuple(tool.version) < _version_tuple(minimum):
        raise HarnessError(f"{tool} is older than the required {minimum}")
    return tool


def find_ninja(lock: Lock, bootstrap: bool = False) -> Tool | None:
    return _find_pip_tool("ninja", lock, bootstrap)


def find_cxx() -> Tool:
    """A C++23 compiler for the ftxi oracle (GCC >= 13, as in its own CI)."""
    names: list[str] = []
    if os.environ.get("CXX"):
        names.append(os.environ["CXX"])
    names += ["g++-14", "g++-13", "g++"]
    candidates = [shutil.which(name) or name for name in names]
    if is_windows():
        candidates += [r"C:\msys64\ucrt64\bin\g++.exe", r"C:\msys64\mingw64\bin\g++.exe"]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        line = _first_line([candidate, "-dumpfullversion", "-dumpversion"])
        if not line:
            continue
        if _version_tuple(line) >= (13,):
            return Tool("g++", Path(candidate), line.strip())
    raise HarnessError("no C++23 compiler found: need g++ 13 or newer on PATH (or set CXX). On Windows, MSYS2's ucrt64 g++ works.")


def _ghc_home(version: str) -> Path:
    return paths.cache_dir() / "toolchains" / f"ghc-{version}"


def _bootstrap_ghc(lock: Lock, version: str) -> None:
    dist = lock.toolchains.get("ghc", {}).get("dist", {}).get(platform_key())
    if not dist:
        raise HarnessError(
            f"no pinned GHC {version} archive for {platform_key()}; install GHC {version} yourself "
            f"(ghcup install ghc {version}, or haskell-actions/setup in CI) so that `ghc` is on PATH"
        )
    downloads = paths.cache_dir() / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / dist["url"].rsplit("/", 1)[-1]
    if not archive.exists() or sha256_file(archive) != dist["sha256"]:
        partial = archive.with_suffix(archive.suffix + ".part")
        with urllib.request.urlopen(dist["url"]) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
        partial.replace(archive)
    digest = sha256_file(archive)
    if digest != dist["sha256"]:
        archive.unlink()
        raise HarnessError(f"GHC archive checksum mismatch: got {digest}, lock file says {dist['sha256']}")
    home = _ghc_home(version)
    staging = home.with_name(home.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            parts = Path(member.name).parts[1:]  # drop the archive's top-level directory
            if not parts or parts[0] == "doc":  # the documentation is a third of the files and none of the compiler
                continue
            member.name = "/".join(parts)
            tar.extract(member, staging, filter="tar")
    shutil.rmtree(home, ignore_errors=True)
    staging.replace(home)


def find_ghc(lock: Lock, bootstrap: bool = False) -> Tool:
    version = lock.toolchains.get("ghc", {}).get("version")
    if not version:
        raise HarnessError("the lock file pins no GHC version ([toolchain.ghc] version)")
    cached = _ghc_home(version) / "bin" / exe("ghc")
    seen: list[str] = []
    for attempt in range(2):
        for candidate in (shutil.which("ghc"), cached):
            if candidate and Path(candidate).exists():
                found = _first_line([candidate, "--numeric-version"])
                if found == version:
                    return Tool("ghc", Path(candidate), found)
                if found:
                    seen.append(f"{candidate} is {found}")
        if attempt == 0 and bootstrap and not cached.exists():
            _bootstrap_ghc(lock, version)
        else:
            break
    detail = f" (found: {'; '.join(seen)})" if seen else ""
    raise HarnessError(
        f"GHC {version} not found{detail}. Put that exact version on PATH, or rerun with --bootstrap "
        f"to fetch the pinned archive into the harness cache (Windows only)."
    )


def find_node(lock: Lock) -> Tool:
    path = shutil.which("node")
    line = _first_line([path, "--version"]) if path else None
    if not line:
        raise HarnessError("node not found on PATH (needed for the Chess in 5D oracle)")
    minimum = str(lock.toolchains.get("node", {}).get("min_version", "18"))
    if _version_tuple(line) < _version_tuple(minimum):
        raise HarnessError(f"node {line} is older than the required {minimum}")
    return Tool("node", Path(path), line.lstrip("v"))


def find_npm() -> Tool:
    path = shutil.which("npm")
    line = _first_line([path, "--version"]) if path else None
    if not line:
        raise HarnessError("npm not found on PATH (needed to install the Chess in 5D oracle's pinned dependencies)")
    return Tool("npm", Path(path), line)


def build_env(*tools: Tool, ceiling: Path | None = None) -> dict[str, str]:
    """Environment for a build: the tools' directories first on PATH.

    ``ceiling`` stops git discovery at the build directory. Without it, an
    oracle whose build asks ``git describe`` for a version string would be
    answered by this repository, because its exported sources live inside it.
    """
    env = dict(os.environ)
    extra = [os.fspath(tool.path.parent) for tool in tools]
    env["PATH"] = os.pathsep.join(dict.fromkeys(extra + env.get("PATH", "").split(os.pathsep)))
    if ceiling is not None:
        env["GIT_CEILING_DIRECTORIES"] = os.fspath(ceiling)
    return env


def doctor(lock: Lock) -> list[tuple[str, str]]:
    """Which tools are present, without fetching anything."""
    rows: list[tuple[str, str]] = []
    probes = {
        "cmake": lambda: find_cmake(lock),
        "ninja": lambda: find_ninja(lock) or "not found (needed on Windows; optional elsewhere)",
        "c++": find_cxx,
        "ghc": lambda: find_ghc(lock),
        "node": lambda: find_node(lock),
        "npm": find_npm,
    }
    for name, probe in probes.items():
        try:
            rows.append((name, str(probe())))
        except HarnessError as error:
            rows.append((name, f"MISSING - {error}"))
    return rows
