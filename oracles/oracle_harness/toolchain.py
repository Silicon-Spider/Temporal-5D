"""Locating (and, only when asked, fetching) the tools that build the oracles.

Nothing here touches the machine outside the harness cache: a bootstrapped tool
is a pip wheel or a checksum-verified archive unpacked under ``.cache``, with no
PATH or registry changes. (The Rust toolchain is a set of such archives, merged
into one prefix as rustup's installer would - no rustup, no profile changes.)
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


# How long a bootstrap download may wait on the server (per connection and per read) before it fails.
DOWNLOAD_TIMEOUT_S = 120


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
        with urllib.request.urlopen(dist["url"], timeout=DOWNLOAD_TIMEOUT_S) as response, open(partial, "wb") as out:
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


def _rust_home(version: str) -> Path:
    return paths.cache_dir() / "toolchains" / f"rust-{version}"


def _fetch(url: str, sha256: str) -> Path:
    """A pinned archive in the harness cache's downloads, fetched if it is missing and checked against its SHA-256."""
    downloads = paths.cache_dir() / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / url.rsplit("/", 1)[-1]
    if not archive.exists() or sha256_file(archive) != sha256:
        partial = archive.with_suffix(archive.suffix + ".part")
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
        partial.replace(archive)
    digest = sha256_file(archive)
    if digest != sha256:
        archive.unlink()
        raise HarnessError(f"{archive.name}: checksum mismatch: got {digest}, the lock file says {sha256}")
    return archive


def merge_rust_component(archive: Path, prefix: Path) -> list[str]:
    """Unpack one Rust component archive into ``prefix`` the way its install.sh would; return the components it held.

    A dist archive (``rustc-1.99.0-<target>.tar.xz``) holds one top-level
    directory: the installer's own files beside one directory per component it
    ships (named in its ``components`` file), each the tree to copy into the
    prefix plus a ``manifest.in``. Only those trees are taken, and not their
    documentation (``share/doc``, ``share/man``), which no build reads.
    """
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        listing = next((member for member in members
                        if len(Path(member.name).parts) == 2 and Path(member.name).parts[1] == "components"), None)
        if listing is None:
            raise HarnessError(f"{archive.name}: no components file - not a Rust dist archive")
        components = tar.extractfile(listing).read().decode().split()
        if not components:
            raise HarnessError(f"{archive.name}: its components file names no component")
        for member in members:
            parts = Path(member.name).parts
            if len(parts) < 3 or parts[1] not in components or parts[2:] == ("manifest.in",):
                continue
            inner = parts[2:]
            if inner[0] == "share" and len(inner) > 1 and inner[1] in ("doc", "man"):
                continue
            member.name = "/".join(inner)
            tar.extract(member, prefix, filter="tar")
    return components


def _bootstrap_rust(lock: Lock, version: str) -> None:
    dist = lock.toolchains.get("rust", {}).get("dist", {}).get(platform_key())
    if not dist:
        raise HarnessError(
            f"no pinned Rust {version} archives for {platform_key()}; install Rust {version} yourself "
            f"(rustup toolchain install {version}) so that `cargo` and `rustc` of that version are on PATH"
        )
    home = _rust_home(version)
    staging = home.with_name(home.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for component in dist["components"]:
        merge_rust_component(_fetch(component["url"], component["sha256"]), staging)
    shutil.rmtree(home, ignore_errors=True)
    staging.replace(home)


def _rust_version(rustc: str | os.PathLike) -> str | None:
    """The release a rustc is ("rustc 1.99.0 (b940084d7 2026-09-28)" -> "1.99.0")."""
    line = _first_line([rustc, "--version"])
    match = re.match(r"rustc (\S+)", line or "")
    return match.group(1) if match else None


def rust_host(rustc: str | os.PathLike) -> str | None:
    """The target a rustc builds for by default ("host: x86_64-pc-windows-gnu" in `rustc -vV`)."""
    try:
        done = run([rustc, "-vV"], timeout=60)
    except OSError:
        return None
    if done.timed_out or done.returncode != 0:
        return None
    match = re.search(r"^host: (\S+)\s*$", done.out_text(), re.M)
    return match.group(1) if match else None


def find_rust(lock: Lock, bootstrap: bool = False) -> tuple[Tool, Tool]:
    """(cargo, rustc) of exactly the pinned Rust release, for the Kairos oracle.

    The harness's own toolchain in its cache comes first, then a cargo and rustc
    on PATH (a rustup proxy included) of exactly that release; with ``bootstrap``,
    a missing one is fetched into the cache from the pinned component archives.
    """
    version = lock.toolchains.get("rust", {}).get("version")
    if not version:
        raise HarnessError("the lock file pins no Rust version ([toolchain.rust] version)")
    cached = _rust_home(version) / "bin"
    # The target the lock file pins for this platform (Windows: the GNU one, whose linker ships with the
    # toolchain); a rustc of the right release built for another host - a rustup MSVC toolchain - is not it.
    target = lock.toolchains.get("rust", {}).get("dist", {}).get(platform_key(), {}).get("target")
    seen: list[str] = []
    for attempt in range(2):
        on_path = (shutil.which("cargo"), shutil.which("rustc"))
        for cargo, rustc in ((cached / exe("cargo"), cached / exe("rustc")), on_path):
            if not cargo or not rustc or not Path(cargo).exists() or not Path(rustc).exists():
                continue
            found = _rust_version(rustc)
            cargo_line = _first_line([cargo, "--version"]) or ""
            host = rust_host(rustc) if found == version else None
            if found == version and cargo_line.startswith(f"cargo {version}") and (target is None or host == target):
                return Tool("cargo", Path(cargo), version), Tool("rustc", Path(rustc), version)
            seen.append(f"{rustc} is {found or 'unreadable'}" + (f" for {host}, not {target}" if host and host != target else ""))
        if attempt == 0 and bootstrap and not (cached / exe("rustc")).exists():
            _bootstrap_rust(lock, version)
        else:
            break
    detail = f" (found: {'; '.join(seen)})" if seen else ""
    raise HarnessError(
        f"Rust {version}{' for ' + target if target else ''} not found{detail}. Put cargo and rustc of exactly that "
        f"release and target on PATH (rustup toolchain install {version}{'-' + target if target else ''}), "
        "or rerun with --bootstrap to fetch the pinned archives into the harness cache."
    )


def rust_env(cargo: Tool, rustc: Tool, *, ceiling: Path) -> dict[str, str]:
    """Environment for a cargo build: the pinned toolchain, nothing of the machine's Rust set-up.

    Cargo's own home (its registry and caches) moves into the harness cache, the
    compiler is named outright, and the variables that would add flags, a wrapper
    or another target directory to the build are dropped.
    """
    env = build_env(cargo, rustc, ceiling=ceiling)
    for name in list(env):
        upper = name.upper()
        if upper in ("RUSTFLAGS", "RUSTDOCFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTUP_TOOLCHAIN") \
                or upper.startswith("CARGO_"):
            del env[name]
    env["CARGO_HOME"] = os.fspath(paths.cache_dir() / "toolchains" / "cargo-home")
    # Always set, empty or not: it takes precedence over every other source of compiler flags - RUSTFLAGS, and
    # [build] and [target.*] rustflags in any cargo configuration file - so none of them reaches the build.
    env["CARGO_ENCODED_RUSTFLAGS"] = ""
    env["RUSTC"] = os.fspath(rustc.path)
    env["CARGO_INCREMENTAL"] = "0"
    env["CARGO_NET_OFFLINE"] = "true"
    return env


def cargo_configs(start: Path, cargo_home: Path) -> list[Path]:
    """The cargo configuration files a cargo run from ``start`` would read, beside the command line and environment.

    Cargo reads `.cargo/config.toml` (and the older `.cargo/config`) in the directory it runs in and in
    every parent, and in its home. Such a file can name another target, a compiler wrapper, a linker or
    flags - none of which the pinned build may take from the machine it runs on.
    """
    found: list[Path] = []
    for folder in [start, *start.parents]:
        found += [folder / ".cargo" / name for name in ("config.toml", "config") if (folder / ".cargo" / name).is_file()]
    found += [cargo_home / name for name in ("config.toml", "config") if (cargo_home / name).is_file()]
    return list(dict.fromkeys(found))


def refuse_cargo_configs(start: Path, cargo_home: Path) -> None:
    """Stop a build that a cargo configuration file on this machine would change (cargo_configs)."""
    found = cargo_configs(start, cargo_home)
    if found:
        raise HarnessError(
            "a cargo configuration file would change the pinned Kairos build: " + ", ".join(map(os.fspath, found))
            + " - cargo reads it from the build directory's parents and its home; move it aside (or set "
            "ORACLE_HARNESS_CACHE to a directory with no such file above it) and rebuild"
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
        "rust": lambda: ", ".join(str(tool) for tool in find_rust(lock)),
        "node": lambda: find_node(lock),
        "npm": find_npm,
    }
    for name, probe in probes.items():
        try:
            rows.append((name, str(probe())))
        except HarnessError as error:
            rows.append((name, f"MISSING - {error}"))
    return rows
