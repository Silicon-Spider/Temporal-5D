#!/usr/bin/env python3
"""Build the 5D chess rules engine and, with --test, run its tests.

    py engine/build.py                 configure and build into build/engine
    py engine/build.py --test          ... then run t5d-tests
    py engine/build.py --test -- zobrist   ... only the tests named *zobrist*
    py engine/build.py --clean         start again from an empty build tree
    py engine/build.py --config Debug  another build type (default Release)
    py engine/build.py --werror        treat compiler warnings as errors

Tools: CMake and Ninja from the project's virtual environment (.venv) when it
has them, otherwise from PATH. Without Ninja, Makefiles are used (MinGW
Makefiles on Windows, Unix Makefiles elsewhere). On Windows the compiler is the
MinGW-w64 g++ on PATH (MSYS2 ucrt64) unless CXX names another; elsewhere it is
CMake's default, or CXX.

To get CMake and Ninja without installing anything system-wide:

    py -m venv --system-site-packages .venv
    .venv/Scripts/python -m pip install cmake ninja      (.venv/bin/python on Linux)

The engine's build never involves anything under oracles/ (AGENTS.md).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "engine"
BUILD = ROOT / "build" / "engine"
WINDOWS = os.name == "nt"
EXE = ".exe" if WINDOWS else ""


def venv_tool(name: str) -> str | None:
    """A tool installed in the project's virtual environment, if there is one."""
    for scripts in ("Scripts", "bin"):
        candidate = ROOT / ".venv" / scripts / (name + EXE)
        if candidate.is_file():
            return str(candidate)
    return None


def find_tool(name: str) -> str | None:
    return venv_tool(name) or shutil.which(name)


def show(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run(command: list[str]) -> int:
    print("+ " + show(command), flush=True)
    return subprocess.call(command)


def cached_generator() -> str | None:
    cache = BUILD / "CMakeCache.txt"
    if not cache.is_file():
        return None
    match = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", cache.read_text(errors="replace"), re.M)
    return match.group(1).strip() if match else None


def choose_generator() -> tuple[str, str | None]:
    """The CMake generator and the make program to give it."""
    ninja = find_tool("ninja")
    if ninja:
        return "Ninja", ninja
    if WINDOWS:
        make = shutil.which("mingw32-make")
        if make:
            return "MinGW Makefiles", make
        sys.exit(
            "build.py: neither ninja nor mingw32-make found; install ninja into .venv "
            "(see this script's docstring)"
        )
    return "Unix Makefiles", None


def compiler_arguments() -> list[str]:
    if os.environ.get("CXX"):
        return []  # CMake reads CXX itself
    if WINDOWS:
        gxx = shutil.which("g++")
        if not gxx:
            sys.exit(
                "build.py: no g++ on PATH (on Windows the engine builds with MSYS2's "
                "ucrt64 g++); or set CXX"
            )
        return [f"-DCMAKE_CXX_COMPILER={gxx}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the engine; with --test, run its tests.")
    parser.add_argument("--test", action="store_true", help="run t5d-tests after building")
    parser.add_argument("--clean", action="store_true", help="delete build/engine first")
    parser.add_argument("--config", default="Release", help="CMake build type (default: Release)")
    parser.add_argument("--werror", action="store_true", help="treat warnings as errors")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 2, help="parallel build jobs")
    parser.add_argument(
        "test_args", nargs="*", help="after --: arguments for t5d-tests (name fragments, --list)"
    )
    args = parser.parse_args()

    cmake = find_tool("cmake")
    if not cmake:
        sys.exit("build.py: cmake not found; install it into .venv (see this script's docstring)")
    generator, make_program = choose_generator()

    if args.clean and BUILD.exists():
        print(f"build.py: removing {BUILD.relative_to(ROOT)}", flush=True)
        shutil.rmtree(BUILD)
    previous = cached_generator()
    if previous is not None and previous != generator:
        print(
            f"build.py: the build tree was made with {previous!r}, not {generator!r}; "
            "starting again",
            flush=True,
        )
        shutil.rmtree(BUILD)

    configure = [
        cmake,
        "-S",
        str(SOURCE),
        "-B",
        str(BUILD),
        "-G",
        generator,
        f"-DCMAKE_BUILD_TYPE={args.config}",
        f"-DT5D_WERROR={'ON' if args.werror else 'OFF'}",
    ]
    if make_program:
        configure.append(f"-DCMAKE_MAKE_PROGRAM={make_program}")
    configure += compiler_arguments()
    if run(configure) != 0:
        return 1
    if (
        run([cmake, "--build", str(BUILD), "--config", args.config, "--parallel", str(args.jobs)])
        != 0
    ):
        return 1

    if args.test:
        tests = BUILD / ("t5d-tests" + EXE)
        if not tests.is_file():
            print(f"build.py: {tests} was not built", file=sys.stderr)
            return 1
        return 1 if run([str(tests), *args.test_args]) != 0 else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
