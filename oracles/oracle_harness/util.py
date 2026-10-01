"""Small shared helpers: findings, hashing, TOML loading, subprocesses."""

from __future__ import annotations

import contextlib
import hashlib
import os
import signal
import subprocess
import sys
import threading
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


class HarnessError(Exception):
    """A problem the user has to act on; reported without a traceback."""


@dataclass(frozen=True)
class Finding:
    """One thing a check found wrong. ``code`` is stable; ``message`` is for humans."""

    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise HarnessError(f"missing file: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise HarnessError(f"{path}: invalid TOML: {error}") from None


def is_windows() -> bool:
    return os.name == "nt"


def exe(name: str) -> str:
    return name + ".exe" if is_windows() else name


def platform_key() -> str:
    """The key toolchain distributions are listed under in the lock file."""
    import platform

    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(machine, machine)
    system = {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform, sys.platform)
    return f"{system}-{arch}"


@dataclass
class Completed:
    returncode: int | None  # None when the process was killed on timeout
    stdout: bytes
    stderr: bytes
    timed_out: bool

    def out_text(self) -> str:
        return self.stdout.decode("utf-8", "replace")

    def err_text(self) -> str:
        return self.stderr.decode("utf-8", "replace")


# How long a killed command gets to hand over what it had already printed.
_DRAIN_S = 15.0
_running: set[subprocess.Popen] = set()
_running_lock = threading.Lock()


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill a command and everything it started.

    Killing only the command itself is not enough: a wrapper (a shell script, a
    build tool, `cargo run`) leaves its child running with our pipes open, and
    the read that follows then waits for that child instead of returning.
    """
    if is_windows():
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False)
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)  # the command leads its own process group (see run)
    with contextlib.suppress(OSError):
        process.kill()


def kill_running() -> None:
    """Stop every command still running. For an interrupted run: nothing is left behind."""
    with _running_lock:
        live = list(_running)
    for process in live:
        _kill_tree(process)


def run(
    argv: Sequence[str | os.PathLike],
    *,
    cwd: Path | None = None,
    stdin: bytes | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> Completed:
    """Run a command, keeping whatever it printed even if it has to be killed.

    Oracle drivers flush one line per position precisely so that a slow record
    still yields the positions answered before the deadline.
    """
    process = subprocess.Popen(
        [os.fspath(part) for part in argv],
        cwd=os.fspath(cwd) if cwd else None,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        # Its own process group, so that a time-out can take its children with it.
        start_new_session=not is_windows(),
    )
    with _running_lock:
        _running.add(process)
    try:
        out, err = process.communicate(stdin, timeout=timeout)
        return Completed(process.returncode, out, err, False)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        try:
            out, err = process.communicate(timeout=_DRAIN_S)
        except subprocess.TimeoutExpired as stuck:
            # Something that outlived the kill still holds the pipes. What it
            # might yet print is given up rather than waited for. (The pipes are
            # left to the garbage collector: on Windows a reader thread is
            # blocked inside them, and closing them from here would block too.)
            out, err = stuck.output or b"", stuck.stderr or b""
        return Completed(None, out, err, True)
    except BaseException:
        _kill_tree(process)
        raise
    finally:
        with _running_lock:
            _running.discard(process)


def check_run(argv: Sequence[str | os.PathLike], *, cwd: Path | None = None, env: dict[str, str] | None = None,
              timeout: float | None = None) -> str:
    """Run a command that must succeed; return its stdout as text."""
    done = run(argv, cwd=cwd, env=env, timeout=timeout)
    if done.timed_out or done.returncode != 0:
        command = " ".join(os.fspath(part) for part in argv)
        tail = (done.err_text() or done.out_text()).strip()[-2000:]
        state = "timed out" if done.timed_out else f"exited with {done.returncode}"
        raise HarnessError(f"command {state}: {command}\n{tail}")
    return done.out_text()


def format_findings(title: str, findings: Iterable[Finding]) -> str:
    items = list(findings)
    if not items:
        return f"{title}: ok"
    lines = [f"{title}: {len(items)} problem(s)"]
    lines += [f"  {item}" for item in items]
    return "\n".join(lines)
