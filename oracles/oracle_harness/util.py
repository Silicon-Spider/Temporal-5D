"""Small shared helpers: findings, hashing, TOML loading, subprocesses."""

from __future__ import annotations

import bisect
import contextlib
import hashlib
import os
import signal
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
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


def sha256_text(data: bytes) -> str:
    """SHA-256 of a text file's bytes with CRLF folded to LF: a checkout on Windows hashes like one on Linux."""
    return sha256_bytes(data.replace(b"\r\n", b"\n"))


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
    returncode: int | None  # None when the process was killed: on timeout, or stopped (below)
    stdout: bytes
    stderr: bytes
    timed_out: bool
    # When the output arrived: after each read of stdout, (how many bytes of it had
    # arrived, seconds from the start of the command). Empty where nothing was
    # measured - a test's stand-in - and then nothing is known of when anything arrived.
    arrivals: tuple[tuple[int, float], ...] = ()
    # The caller's stop condition was met and the command was stopped before it
    # ended - never because its time ran out (that is timed_out).
    stopped: bool = False
    # Seconds from the start of the command until it ended (or was killed) and its output was read.
    elapsed_s: float = 0.0
    _ends: list[int] = field(default_factory=list, repr=False, compare=False)

    def out_text(self) -> str:
        return self.stdout.decode("utf-8", "replace")

    def err_text(self) -> str:
        return self.stderr.decode("utf-8", "replace")

    @property
    def killed(self) -> bool:
        """Did the harness end the command (out of time, or stopped), rather than the command itself?"""
        return self.timed_out or self.stopped

    def arrived(self, offset: int) -> float | None:
        """Seconds from the start of the command until stdout byte ``offset`` (counted from 0) had arrived.

        None when it never did, or when the arrival times were not measured.
        """
        if not self.arrivals or offset < 0:
            return None
        if len(self._ends) != len(self.arrivals):
            self._ends[:] = [end for end, _ in self.arrivals]
        slot = bisect.bisect_right(self._ends, offset)
        return self.arrivals[slot][1] if slot < len(self.arrivals) else None


# How long a killed command gets to hand over what it had already printed.
_DRAIN_S = 15.0
_running: set[subprocess.Popen] = set()
_running_lock = threading.Lock()
# Read from a command's stdout at most this much at a time. A read returns as soon
# as anything has arrived, so every line a driver flushes is timed as it lands.
_CHUNK = 1 << 16


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


Stop = Callable[[bytes], bool]


def run(
    argv: Sequence[str | os.PathLike],
    *,
    cwd: Path | None = None,
    stdin: bytes | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    stop: Stop | None = None,
) -> Completed:
    """Run a command, keeping whatever it printed even if it has to be killed.

    Oracle drivers flush one line per position precisely so that a slow record
    still yields the positions answered before the deadline. Its stdout is read
    as it arrives, and when each piece arrived is kept (Completed.arrivals): the
    cost of an answer is when it arrived, which is what a deadline cuts (H32).

    ``stop`` is shown every piece of stdout as it arrives, in order; once it
    returns True the command is killed - everything it needed has arrived - and
    the result says it was stopped, not that it timed out.
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
    started = time.monotonic()
    with _running_lock:
        _running.add(process)
    out, err = bytearray(), bytearray()
    arrivals: list[tuple[int, float]] = []
    halted = threading.Event()  # the stop condition was met

    def pump_out() -> None:
        watching = stop
        with contextlib.suppress(OSError, ValueError):
            while chunk := process.stdout.read1(_CHUNK):
                out.extend(chunk)
                arrivals.append((len(out), time.monotonic() - started))
                if watching is None:
                    continue
                try:
                    done = watching(chunk)
                except Exception:  # a broken stop condition never stops a run: it only stops watching
                    watching, done = None, False
                if done:
                    watching = None
                    halted.set()
                    _kill_tree(process)

    def pump_err() -> None:
        with contextlib.suppress(OSError, ValueError):
            while chunk := process.stderr.read1(_CHUNK):
                err.extend(chunk)

    def feed() -> None:
        # A command that exits without reading all of its input closes the pipe under us.
        with contextlib.suppress(OSError, ValueError):
            process.stdin.write(stdin)
        with contextlib.suppress(OSError, ValueError):
            process.stdin.close()

    threads = [threading.Thread(target=pump_out, daemon=True), threading.Thread(target=pump_err, daemon=True)]
    if stdin is not None:
        threads.append(threading.Thread(target=feed, daemon=True))
    for thread in threads:
        thread.start()
    deadline = None if timeout is None else started + timeout

    def left() -> float | None:
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    timed_out = False
    try:
        for thread in threads[:2]:
            thread.join(left())
        ended = not any(thread.is_alive() for thread in threads[:2])
        if ended:
            try:
                process.wait(left())
            except subprocess.TimeoutExpired:
                ended = False
        if not ended:
            timed_out = not halted.is_set()
            _kill_tree(process)
        if not ended or halted.is_set():
            # A killed command gets a moment to hand over what it had printed.
            # Something that outlived the kill may still hold the pipes: what it
            # might yet print is given up rather than waited for. (The pipes are
            # left to the garbage collector: on Windows a reader thread is
            # blocked inside them, and closing them from here would block too.)
            for thread in threads[:2]:
                thread.join(_DRAIN_S)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(_DRAIN_S)
    except BaseException:
        _kill_tree(process)
        raise
    finally:
        with _running_lock:
            _running.discard(process)
        # A pipe whose reader has finished is closed here; one a reader is still
        # blocked in (something outlived the kill) is left to the garbage collector.
        for thread, pipe in zip(threads, (process.stdout, process.stderr, process.stdin), strict=False):
            if not thread.is_alive() and pipe is not None:
                with contextlib.suppress(OSError, ValueError):
                    pipe.close()
    stopped = halted.is_set() and not timed_out
    # The arrivals first: a byte that arrives after this is not in `out` either.
    timeline = tuple(arrivals)
    return Completed(None if (timed_out or stopped) else process.returncode, bytes(out), bytes(err), timed_out,
                     arrivals=timeline, stopped=stopped, elapsed_s=time.monotonic() - started)


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
