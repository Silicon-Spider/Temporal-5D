"""Building the oracles from their locked commits.

Every build starts from ``git archive`` of the locked commit, written into the
harness cache. The submodule work trees are never built in and never written to:
what runs is exactly the pinned tree, whatever state the checkout is in.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import gitutil, paths, toolchain
from .lockfile import Lock, Pin
from .util import HarnessError, check_run, exe, is_windows, platform_key, sha256_bytes

# Bump to invalidate every cached build after changing a recipe below.
RECIPE_VERSION = 1

Log = Callable[[str], None]


@dataclass(frozen=True)
class Build:
    oracle: str
    key: str
    root: Path
    artifacts: dict[str, Path]
    info: dict

    def artifact(self, name: str) -> Path:
        return self.artifacts[name]


def _driver_inputs(oracle: str) -> list[Path]:
    """Harness files compiled into a build (they change the key when edited)."""
    if oracle == "ftxi":
        return [paths.DRIVERS / "ftxi" / "CMakeLists.txt", paths.DRIVERS / "ftxi" / "ftxi_driver.cpp"]
    if oracle == "cwmtt":
        return [paths.DRIVERS / "cwmtt" / "Driver.hs"]
    if oracle == "kairos":
        return [paths.DRIVERS / "kairos" / name for name in KAIROS_DRIVER_FILES]
    return []


def build_key(pin: Pin) -> str:
    parts = [f"recipe={RECIPE_VERSION}", f"commit={pin.commit}", f"platform={platform_key()}"]
    for path in _driver_inputs(pin.id):
        parts.append(f"{path.name}={sha256_bytes(path.read_bytes().replace(b'\r\n', b'\n'))}")
    return sha256_bytes("\n".join(parts).encode())[:12]


def build_root(pin: Pin) -> Path:
    return paths.cache_dir() / "build" / pin.id / build_key(pin)


def load(pin: Pin) -> Build | None:
    """The finished build for the current pin and recipe, if there is one."""
    root = build_root(pin)
    stamp = root / "build.json"
    if not stamp.exists():
        return None
    data = json.loads(stamp.read_text(encoding="utf-8"))
    artifacts = {name: root / relative for name, relative in data["artifacts"].items()}
    if not all(path.exists() for path in artifacts.values()):
        return None
    return Build(pin.id, data["key"], root, artifacts, data.get("info", {}))


def _finish(pin: Pin, root: Path, artifacts: dict[str, str], info: dict) -> Build:
    stamp = {"oracle": pin.id, "key": build_key(pin), "commit": pin.commit, "recipe": RECIPE_VERSION,
             "platform": platform_key(), "artifacts": artifacts, "info": info}
    temporary = root / "build.json.tmp"
    temporary.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "build.json")
    return Build(pin.id, stamp["key"], root, {name: root / rel for name, rel in artifacts.items()}, info)


def _export(pin: Pin, root: Path, log: Log) -> Path:
    checkout = pin.checkout
    if not gitutil.is_checkout(checkout) or not gitutil.has_commit(checkout, pin.commit):
        raise HarnessError(f"{pin.id}: submodule not initialised at the locked commit (run: python {paths.HARNESS_REL}/harness.py sync)")
    source = root / "src"
    log(f"  exporting {pin.upstream} @ {pin.commit[:12]}")
    gitutil.export_commit(checkout, pin.commit, source)
    return source


def _cmake(cmake: toolchain.Tool, source: Path, binary: Path, target: str, env: dict[str, str],
           generator: list[str], defines: list[str]) -> None:
    check_run([cmake.path, "-S", source, "-B", binary, *generator, "-DCMAKE_BUILD_TYPE=Release", *defines],
              env=env, timeout=600)
    # A number, always: a bare --parallel lets the Makefile generator start an unbounded `make -j`.
    check_run([cmake.path, "--build", binary, "--target", target, "--config", "Release",
               "--parallel", str(os.cpu_count() or 2)], env=env, timeout=3600)


def _build_ftxi(pin: Pin, lock: Lock, root: Path, bootstrap: bool, log: Log) -> Build:
    cmake = toolchain.find_cmake(lock, bootstrap)
    ninja = toolchain.find_ninja(lock, bootstrap)
    cxx = toolchain.find_cxx()
    if ninja is None and is_windows():
        raise HarnessError("ninja not found (needed on Windows). Rerun with --bootstrap to fetch the pinned wheel into the harness cache.")
    source = _export(pin, root, log)
    tools = [tool for tool in (cxx, cmake, ninja) if tool is not None]
    env = toolchain.build_env(*tools, ceiling=root)
    generator = ["-G", "Ninja"] if ninja is not None else []
    defines = [f"-DCMAKE_CXX_COMPILER={cxx.path.as_posix()}"]
    if is_windows():
        # MinGW executables otherwise need the compiler's runtime DLLs on PATH at run time.
        defines.append("-DCMAKE_EXE_LINKER_FLAGS=-static")

    log("  building the stock 5dtools CLI with the oracle's own CMake project")
    _cmake(cmake, source, root / "stock", "5dtools", env, generator, [*defines, "-DTOOLS=ON"])
    log("  building the harness driver against the oracle's core library")
    _cmake(cmake, paths.DRIVERS / "ftxi", root / "driver", "ftxi_oracle_driver", env, generator,
           [*defines, f"-DFTXI_SOURCE_DIR={source.as_posix()}"])
    artifacts = {"tools": f"stock/{exe('5dtools')}", "driver": f"driver/{exe('ftxi_oracle_driver')}"}
    info = {"cmake": cmake.version, "cxx": f"{cxx.name} {cxx.version}", "ninja": ninja.version if ninja else None}
    return _finish(pin, root, artifacts, info)


def cwmtt_layouts(layouts_source: str) -> list[str]:
    """The Board names cwmtt knows, read from the `layouts` table of its Layouts.hs.

    cwmtt falls back to the Standard layout for any other name without saying
    so; the adapter needs this list to refuse those records.
    """
    match = re.search(r"^layouts\s*=\s*\n(.*?)^\s*\]", layouts_source, re.S | re.M)
    if not match:
        raise HarnessError("cwmtt: could not find the `layouts` table in Layouts.hs - the adapter needs updating for this pin")
    names = re.findall(r'\(\s*"([^"]+)"\s*,', match.group(1))
    if "Standard" not in names:
        raise HarnessError("cwmtt: the `layouts` table in Layouts.hs no longer lists Standard - the adapter needs updating for this pin")
    return names


def _build_cwmtt(pin: Pin, lock: Lock, root: Path, bootstrap: bool, log: Log) -> Build:
    ghc = toolchain.find_ghc(lock, bootstrap)
    source = _export(pin, root, log)
    output = root / exe("cwmtt")
    driver = root / exe("cwmtt_state_driver")
    env = toolchain.build_env(ghc, ceiling=root)
    # The flags of cwmtt.cabal (Haskell2010, -Wno-x-partial) at cabal's default
    # optimisation level. Its only dependencies are GHC boot packages.
    flags = ["--make", "-O1", "-XHaskell2010", "-Wno-x-partial", "-i."]
    log(f"  compiling the stock CLI with GHC {ghc.version}")
    check_run([ghc.path, *flags, "-outputdir", root / "obj", "-o", output, "Game/Chess/TimeTravel/Main.hs"],
              cwd=source, env=env, timeout=3600)
    # The harness's state driver (it reports what the CLI has no command for:
    # the boards cwmtt holds at each position), against the same modules with
    # the same flags. Both programs are module Main, so the driver is compiled
    # into an object directory of its own.
    log("  compiling the harness's state driver against the same modules")
    check_run([ghc.path, *flags, "-outputdir", root / "driver-obj", "-o", driver,
               paths.DRIVERS / "cwmtt" / "Driver.hs"],
              cwd=source, env=env, timeout=3600)
    layouts = cwmtt_layouts((source / "Game" / "Chess" / "TimeTravel" / "Layouts.hs").read_text(encoding="utf-8"))
    return _finish(pin, root, {"cli": output.name, "driver": driver.name}, {"ghc": ghc.version, "layouts": layouts})


def _build_chessin5d(pin: Pin, lock: Lock, root: Path, bootstrap: bool, log: Log) -> Build:
    node = toolchain.find_node(lock)
    npm = toolchain.find_npm()
    source = _export(pin, root, log)
    log("  installing the pinned runtime dependencies (npm ci, scripts disabled)")
    # npm's download cache is kept inside the harness cache, not in the user's profile.
    check_run([npm.path, "ci", "--omit=dev", "--ignore-scripts", "--no-audit", "--no-fund",
               "--cache", paths.cache_dir() / "npm"],
              cwd=source, env=toolchain.build_env(node, ceiling=root), timeout=900)
    dependencies = {}
    modules = source / "node_modules"
    for package in sorted(modules.glob("*/package.json")) + sorted(modules.glob("@*/*/package.json")):
        meta = json.loads(package.read_text(encoding="utf-8"))
        dependencies[meta["name"]] = {"version": meta.get("version"), "licence": meta.get("license")}
    return _finish(pin, root, {"root": "src"}, {"node": node.version, "npm": npm.version, "dependencies": dependencies})


# Prints the notation parser's table of named boards - the BOARDS object its
# game.js exports, which pgn.parse() looks a Board header up in - as JSON,
# {upper-case name: "WxH"}. Run by node on the export; it reads the table only.
_NOTATION_BOARDS_JS = (
    "const table = require(process.argv[1]).BOARDS;"
    "process.stdout.write(JSON.stringify(Object.fromEntries("
    "Object.entries(table).map((entry) => [entry[0], entry[1][2]]))));"
)


def notation_boards(output: str) -> dict[str, str]:
    """The notation parser's named boards, by the upper-case name it looks them up under, with their size.

    The parser sets up one empty board, without a word, for a Board it does not
    know, and sets a named layout up on boards of the Size header's size (8x8
    without one), never its own. The adapter needs this table to refuse both.
    """
    try:
        table = json.loads(output)
    except json.JSONDecodeError:
        table = None
    if (not isinstance(table, dict) or "STANDARD" not in table
            or not all(isinstance(name, str) and name == name.upper() and isinstance(size, str)
                       and re.fullmatch(r"[1-9]\d*x[1-9]\d*", size) for name, size in table.items())):
        raise HarnessError("notation: the parser's BOARDS table is not a table of upper-case names and \"WxH\" "
                           "sizes with Standard among them - the adapter needs updating for this pin")
    return dict(sorted(table.items()))


def _build_notation(pin: Pin, lock: Lock, root: Path, bootstrap: bool, log: Log) -> Build:
    node = toolchain.find_node(lock)
    source = _export(pin, root, log)
    # parsers/pgn.js and parsers/game.js require nothing but each other: no package is installed.
    log("  reading the parser's table of named boards")
    boards = notation_boards(check_run([node.path, "-e", _NOTATION_BOARDS_JS, source / "parsers" / "game.js"],
                                       env=toolchain.build_env(node, ceiling=root), timeout=120))
    return _finish(pin, root, {"root": "src", "parser": "src/parsers/pgn.js"}, {"node": node.version, "boards": boards})


# The harness's Kairos driver: a cargo package of its own, copied beside the export.
KAIROS_DRIVER_FILES = ("Cargo.toml", "Cargo.lock", "src/main.rs")


def _build_kairos(pin: Pin, lock: Lock, root: Path, bootstrap: bool, log: Log) -> Build:
    cargo, rustc = toolchain.find_rust(lock, bootstrap)
    _export(pin, root, log)  # root/src: the pinned crate, which the driver's Cargo.toml names as ../src
    driver = root / "driver"
    for name in KAIROS_DRIVER_FILES:
        target = driver / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths.DRIVERS / "kairos" / name, target)
    # Run from the build root, outside the export, so that the crate's .cargo/config.toml
    # (-C target-cpu=native) is not read: the same build on every x86-64 machine. Offline and
    # locked: the crate has no dependencies, and the driver's lock file says so.
    env = toolchain.rust_env(cargo, rustc, ceiling=root)
    env["CARGO_TARGET_DIR"] = os.fspath(root / "target")
    found = re.search(r"^host: (\S+)$", check_run([rustc.path, "-vV"], env=env, timeout=120), re.M)
    host = found.group(1) if found else None
    if host and host.endswith("-windows-gnu"):
        # Link with the MinGW linker the toolchain ships (rust-mingw), never a gcc that happens to be
        # on PATH: one built for another C runtime would link the driver against that runtime instead.
        env["RUSTFLAGS"] = "-C link-self-contained=yes"
    log(f"  building the harness driver against the pinned crate with Rust {rustc.version}")
    check_run([cargo.path, "build", "--release", "--locked", "--offline", "--manifest-path", driver / "Cargo.toml"],
              cwd=root, env=env, timeout=3600)
    artifacts = {"driver": f"target/release/{exe('kairos_oracle_driver')}"}
    return _finish(pin, root, artifacts, {"rustc": rustc.version, "cargo": cargo.version, "host": host})


_RECIPES = {"ftxi": _build_ftxi, "cwmtt": _build_cwmtt, "chessin5d": _build_chessin5d, "notation": _build_notation,
            "kairos": _build_kairos}


def recipe_tools(oracle: str) -> tuple[str, ...]:
    """The build tools a recipe uses, by the names `doctor` prints them under."""
    tools = {"ftxi": ("c++", "cmake"), "cwmtt": ("ghc",), "chessin5d": ("node", "npm"), "notation": ("node",),
             "kairos": ("rust",)}[oracle]
    return (*tools, "ninja") if oracle == "ftxi" and is_windows() else tools


def tools_needed(lock: Lock, oracles: list[str]) -> list[str]:
    """The build tools still needed to build ``oracles``: those of the ones with no up-to-date build.

    An up-to-date build is used as it is, without its toolchain - which is what
    lets CI skip installing a compiler when a cached build is still current.
    """
    return sorted({tool for oracle in oracles if load(lock.pin(oracle)) is None for tool in recipe_tools(oracle)})


def buildable(lock: Lock) -> list[str]:
    return [pin.id for pin in lock.oracles() if pin.id in _RECIPES]


def build(pin: Pin, lock: Lock, *, bootstrap: bool = False, force: bool = False, log: Log = print) -> Build:
    if pin.id not in _RECIPES:
        raise HarnessError(f"no build recipe for '{pin.id}'")
    if not force:
        existing = load(pin)
        if existing is not None:
            log(f"{pin.id}: build {existing.key} is up to date")
            return existing
    root = build_root(pin)
    log(f"{pin.id}: building into {os.path.relpath(root, paths.REPO_ROOT)}")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    built = _RECIPES[pin.id](pin, lock, root, bootstrap, log)
    prune_siblings(root)
    return built


def prune_siblings(root: Path) -> None:
    """Drop the builds of other pins, recipes and driver versions of the same oracle.

    Each is a full export plus a build tree; nothing reads them once the key has moved on.
    """
    for other in root.parent.iterdir():
        if other.is_dir() and other != root:
            shutil.rmtree(other, ignore_errors=True)
