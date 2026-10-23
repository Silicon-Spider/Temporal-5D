"""Exceptions of the engine binding: one per status of the C API (``t5d/capi.h``).

Every function of the C API that can fail returns a status and, where it takes a
``char** error``, a message. The binding raises the exception of that status with the
engine's own message, so a caller can catch a class of failure (an illegal move, a record
that cannot be read) without parsing text. Each also derives from the Python built-in
it resembles, so generic handlers keep working: an illegal move is a ``ValueError``.
"""

from __future__ import annotations

#: The C API's status codes (``t5d/capi.h``).
T5D_OK = 0
T5D_ERR_ARGUMENT = 1
T5D_ERR_NOTATION = 2
T5D_ERR_MEMORY = 3
T5D_ERR_INTERNAL = 4
T5D_ERR_UNSUPPORTED = 5
T5D_ERR_ILLEGAL = 6


class EngineError(Exception):
    """A call into the engine failed. ``status`` is the C API's status code, ``message``
    the engine's own message (empty if the engine gave none)."""

    status: int = T5D_ERR_INTERNAL

    def __init__(self, message: str = "", *, status: int | None = None, function: str = "") -> None:
        self.message = message
        self.function = function
        if status is not None:
            self.status = status
        where = f"{function}: " if function else ""
        super().__init__(f"{where}{message}" if message else f"{where}status {self.status}")


class EngineNotFoundError(EngineError, FileNotFoundError):
    """The engine's shared library could not be found or loaded."""

    status = -1


class EngineArgumentError(EngineError, ValueError):
    """T5D_ERR_ARGUMENT: a NULL pointer, an invalid size, an index out of range, a player to
    move that cannot be inferred, an expansion that is not of the current state."""

    status = T5D_ERR_ARGUMENT


class NotationError(EngineError, ValueError):
    """T5D_ERR_NOTATION: malformed 5DFEN, boards that form no multiverse, a 5dpgn record
    that cannot be read."""

    status = T5D_ERR_NOTATION


class EngineMemoryError(EngineError, MemoryError):
    """T5D_ERR_MEMORY: the engine ran out of memory."""

    status = T5D_ERR_MEMORY


class EngineInternalError(EngineError, RuntimeError):
    """T5D_ERR_INTERNAL: anything else - a bug in the engine."""

    status = T5D_ERR_INTERNAL


class UnsupportedError(EngineError, NotImplementedError):
    """T5D_ERR_UNSUPPORTED: a game or a feature the engine does not implement - a variant it
    does not set up, a relation it does not build (the reserved id 6)."""

    status = T5D_ERR_UNSUPPORTED


class IllegalError(EngineError, ValueError):
    """T5D_ERR_ILLEGAL: a move or a submission the rules refuse - a submission that leaves a
    royal piece of the mover's capturable included (C4, R1)."""

    status = T5D_ERR_ILLEGAL


_BY_STATUS: dict[int, type[EngineError]] = {
    T5D_ERR_ARGUMENT: EngineArgumentError,
    T5D_ERR_NOTATION: NotationError,
    T5D_ERR_MEMORY: EngineMemoryError,
    T5D_ERR_INTERNAL: EngineInternalError,
    T5D_ERR_UNSUPPORTED: UnsupportedError,
    T5D_ERR_ILLEGAL: IllegalError,
}


def error_for(status: int, message: str = "", function: str = "") -> EngineError:
    """The exception of a non-zero status (an unknown status is an internal error)."""
    cls = _BY_STATUS.get(int(status))
    if cls is None:
        return EngineInternalError(message or f"unknown status {status}", status=int(status), function=function)
    return cls(message, function=function)
