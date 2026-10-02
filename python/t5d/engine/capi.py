"""The ctypes layer over the engine's C API (``engine/include/t5d/capi.h`` and ``records.h``).

This module mirrors the C declarations exactly - structures, constants, signatures - and
finds and loads the shared library. Nothing here owns engine memory; ``game.py`` wraps
the handles, and ``expansion.py`` copies an expansion's arrays out before the next call.

**Finding the library** (``find_library``), first match wins:

1. an explicit path (``load_library(path)``), which must exist;
2. the environment variable ``T5D_ENGINE_LIB`` (a file, or a directory holding it),
   which must exist when set;
3. ``build/engine/`` under the repository root (where ``py engine/build.py`` puts it):
   ``t5d.dll`` on Windows, ``libt5d.so`` on Linux, ``libt5d.dylib`` on macOS.

A missing library raises ``EngineNotFoundError`` naming every place looked and how to build
it. On Windows the DLL links the MinGW runtime statically and imports only KERNEL32 and the
Universal CRT (engine README, "Building and testing"), so it loads without MSYS2 on PATH.

**Layouts.** Each ``Structure`` below is the C struct field for field, in order, so ctypes
lays it out as the C compiler does (natural alignment, no packing); ``_check_layouts``
asserts the sizes the headers fix (``t5d_move`` is 32 bytes, no padding). ``MOVE_DTYPE`` is
the same record as a numpy structured dtype, for copying a move array in one step.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from ctypes import POINTER, c_char, c_char_p, c_int8, c_int32, c_size_t, c_uint8, c_uint32, c_uint64, c_void_p
from pathlib import Path

import numpy as np

from .errors import T5D_OK, EngineNotFoundError, error_for

#: The environment variable that names the library (a file or its directory).
ENV_VAR = "T5D_ENGINE_LIB"

# ---- constants (t5d/capi.h, t5d/records.h) ------------------------------------------------

T5D_PARITY_INFER, T5D_PARITY_ODD, T5D_PARITY_EVEN = -1, 0, 1
T5D_TO_MOVE_INFER, T5D_TO_MOVE_WHITE, T5D_TO_MOVE_BLACK = -1, 0, 1
T5D_RULES_STANDARD = 0
T5D_RULES_MOVES_AFTER_SHIFT = 1 << 3
T5D_FEATURES_ALL, T5D_FEATURES_SCOPE, T5D_FEATURES_NONE = 0, 1, 2
T5D_PSEUDO_LEGAL, T5D_LEGAL = 0, 1
T5D_NO_WINDOW = 2**31 - 1  # INT32_MAX

# Move flags (t5d_move.flags).
T5D_MOVE_CAPTURE = 1
T5D_MOVE_BRANCHING = 2
T5D_MOVE_CASTLE = 4
T5D_MOVE_EN_PASSANT = 8
T5D_MOVE_DOUBLE_STEP = 16
T5D_MOVE_PROMOTION = 32
T5D_MOVE_TRAVEL = 64

# Relation ids: python/t5d/net/state.py RELATION_TYPES, in its order (records.h says so;
# tests/engine checks the two agree).
T5D_RELATION_MOVE = 0
T5D_RELATION_ATTACK = 1
T5D_RELATION_DEFENCE = 2
T5D_RELATION_PIN = 3  # reserved, not built
T5D_RELATION_XRAY = 4
T5D_RELATION_CHECK = 5  # reserved, not built
T5D_RELATION_TRAVEL_THREAT = 6  # reserved, not built

#: The relations the engine builds, by the network's names (``t5d_expand_options.relations``).
BUILT_RELATIONS: dict[str, int] = {"attack": T5D_RELATION_ATTACK, "defence": T5D_RELATION_DEFENCE, "xray": T5D_RELATION_XRAY}
#: Named by the network's contract but refused by the engine (T5D_ERR_UNSUPPORTED): pin
#: and check would read check detection; travel-threat has no edge defined by the spec.
RESERVED_RELATIONS: dict[str, int] = {"pin": T5D_RELATION_PIN, "check": T5D_RELATION_CHECK, "travel_threat": T5D_RELATION_TRAVEL_THREAT}


# ---- structures -----------------------------------------------------------------------------


class Fingerprint(ctypes.Structure):
    _fields_ = [("fingerprint", c_char * 33), ("placement", c_char * 33), ("boards", c_uint64), ("pieces", c_uint64)]


class StateInfo(ctypes.Structure):
    _fields_ = [
        ("boards", c_uint64),
        ("pieces", c_uint64),
        ("width", c_int32),
        ("height", c_int32),
        ("even", c_int32),
        ("to_move", c_int32),
        ("min_timeline", c_int32),
        ("max_timeline", c_int32),
        ("created_by_white", c_int32),
        ("created_by_black", c_int32),
        ("ruleset", c_uint32),
    ]


class MoveRecord(ctypes.Structure):
    """``t5d_move`` (records.h): 32 bytes, no padding."""

    _fields_ = [
        ("from_timeline", c_int32),
        ("from_half_turn", c_int32),
        ("to_timeline", c_int32),
        ("to_half_turn", c_int32),
        ("from_board", c_int32),
        ("to_board", c_int32),
        ("from_square", c_int8),
        ("to_square", c_int8),
        ("from_cell", c_int8),
        ("to_cell", c_int8),
        ("piece", c_uint8),
        ("captured", c_uint8),
        ("promotion", c_uint8),
        ("flags", c_uint8),
    ]


#: ``t5d_move`` as a numpy record (same field names, same offsets, 32 bytes).
MOVE_DTYPE = np.dtype(
    [
        ("from_timeline", "<i4"),
        ("from_half_turn", "<i4"),
        ("to_timeline", "<i4"),
        ("to_half_turn", "<i4"),
        ("from_board", "<i4"),
        ("to_board", "<i4"),
        ("from_square", "i1"),
        ("to_square", "i1"),
        ("from_cell", "i1"),
        ("to_cell", "i1"),
        ("piece", "u1"),
        ("captured", "u1"),
        ("promotion", "u1"),
        ("flags", "u1"),
    ]
)


class ExpandOptions(ctypes.Structure):
    _fields_ = [("window_k", c_int32), ("square_features", c_int32), ("relations", c_uint32)]


class GameSummary(ctypes.Structure):
    _fields_ = [
        ("boards", c_uint64),
        ("width", c_int32),
        ("height", c_int32),
        ("even", c_int32),
        ("to_move", c_int32),
        ("present", c_int32),
        ("timelines", c_int32),
        ("active_timelines", c_int32),
        ("obligations_met", c_int32),
        ("may_move", c_int32),
        ("may_submit", c_int32),
        ("ruleset", c_uint32),
        ("rules", c_uint64),
    ]


class CExpansion(ctypes.Structure):
    """``t5d_expansion``: counts, and pointers into the game handle's buffers - valid only
    until the handle is next changed (``expansion.py`` copies them out at once)."""

    _fields_ = [
        ("width", c_int32),
        ("height", c_int32),
        ("even", c_int32),
        ("to_move", c_int32),
        ("present", c_int32),
        ("timelines", c_int32),
        ("active_timelines", c_int32),
        ("window_k", c_int32),
        ("may_move", c_int32),
        ("origins_unknown", c_int32),
        ("board_count", c_size_t),
        ("move_count", c_size_t),
        ("edge_count", c_size_t),
        ("relation_count", c_size_t),
        ("window_count", c_size_t),
        ("e_count", c_size_t),
        ("src_extra_count", c_size_t),
        ("feature_board_count", c_size_t),
        ("origin_count", c_size_t),
        # per board
        ("timeline", POINTER(c_int32)),
        ("half_turn", POINTER(c_int32)),
        ("is_head", POINTER(c_uint8)),
        ("active", POINTER(c_uint8)),
        ("side_to_move", POINTER(c_uint8)),
        ("playable", POINTER(c_uint8)),
        ("window_depth", POINTER(c_int32)),
        ("band_depth", POINTER(c_int32)),
        ("is_origin", POINTER(c_uint8)),
        ("parent", POINTER(c_int32)),
        # per square
        ("feature_boards", POINTER(c_int32)),
        ("piece", POINTER(c_uint8)),
        ("colour", POINTER(c_uint8)),
        ("moved", POINTER(c_uint8)),
        # lists
        ("moves", POINTER(MoveRecord)),
        ("edges", POINTER(c_int32)),
        ("relations", POINTER(c_int32)),
        ("window", POINTER(c_int32)),
        ("e_boards", POINTER(c_int32)),
        ("src_extra", POINTER(c_int32)),
    ]


def _check_layouts() -> None:
    if ctypes.sizeof(MoveRecord) != 32 or MOVE_DTYPE.itemsize != 32:
        raise ImportError("t5d_move must be 32 bytes (t5d/records.h)")
    for name, _ in MoveRecord._fields_:
        if getattr(MoveRecord, name).offset != MOVE_DTYPE.fields[name][1]:
            raise ImportError(f"t5d_move.{name}: the ctypes and numpy layouts disagree")
    if ctypes.sizeof(ExpandOptions) != 12:
        raise ImportError("t5d_expand_options must be 12 bytes (three 32-bit fields)")


_check_layouts()

# ---- the library ------------------------------------------------------------------------------

_STATE = c_void_p  # t5d_state*
_GAME = c_void_p  # t5d_game*
_ERR = POINTER(c_void_p)  # char** error (a library string, freed with t5d_free)

#: name -> (restype, argtypes). Every function of capi.h.
_SIGNATURES: dict[str, tuple[object, list[object]]] = {
    "t5d_version": (c_char_p, []),
    "t5d_free": (None, [c_void_p]),
    "t5d_state_new_standard": (c_int32, [POINTER(_STATE)]),
    "t5d_state_new_standard_two_timelines": (c_int32, [POINTER(_STATE)]),
    "t5d_state_new_from_fen": (
        c_int32,
        [POINTER(c_char_p), c_size_t, c_int32, c_int32, c_int32, c_int32, c_uint32, POINTER(_STATE), _ERR],
    ),
    "t5d_state_copy": (c_int32, [_STATE, POINTER(_STATE)]),
    "t5d_state_destroy": (None, [_STATE]),
    "t5d_state_info": (c_int32, [_STATE, POINTER(StateInfo)]),
    "t5d_state_fingerprint": (c_int32, [_STATE, POINTER(Fingerprint)]),
    "t5d_state_canonical": (c_int32, [_STATE, POINTER(c_void_p)]),
    "t5d_state_zobrist": (c_int32, [_STATE, POINTER(c_uint64)]),
    "t5d_state_board_count": (c_int32, [_STATE, POINTER(c_size_t)]),
    "t5d_state_board": (c_int32, [_STATE, c_size_t, POINTER(c_int32), POINTER(c_int32), POINTER(c_void_p)]),
    "t5d_game_new_from_pgn": (c_int32, [c_char_p, c_size_t, c_uint64, POINTER(_GAME), _ERR]),
    "t5d_game_new_from_fen": (
        c_int32,
        [POINTER(c_char_p), c_size_t, c_int32, c_int32, c_int32, c_int32, c_uint64, POINTER(_GAME), _ERR],
    ),
    "t5d_game_new_variant": (c_int32, [c_char_p, c_uint64, POINTER(_GAME), _ERR]),
    "t5d_game_copy": (c_int32, [_GAME, POINTER(_GAME)]),
    "t5d_game_destroy": (None, [_GAME]),
    "t5d_game_info": (c_int32, [_GAME, POINTER(GameSummary)]),
    "t5d_game_record_actions": (c_int32, [_GAME, POINTER(c_size_t), POINTER(c_size_t), POINTER(c_int32)]),
    "t5d_game_replay_next": (c_int32, [_GAME, _ERR]),
    "t5d_game_apply": (c_int32, [_GAME, POINTER(MoveRecord), _ERR]),
    "t5d_game_apply_expanded": (c_int32, [_GAME, c_size_t, _ERR]),
    "t5d_game_submit": (c_int32, [_GAME, _ERR]),
    "t5d_game_set_legality": (c_int32, [_GAME, c_int32, _ERR]),
    "t5d_game_expand": (c_int32, [_GAME, POINTER(ExpandOptions), POINTER(CExpansion), _ERR]),
    "t5d_game_state": (c_int32, [_GAME, POINTER(_STATE)]),
}


def _library_names() -> tuple[str, ...]:
    if sys.platform == "win32" or sys.platform == "cygwin":
        return ("t5d.dll", "libt5d.dll")
    if sys.platform == "darwin":
        return ("libt5d.dylib", "libt5d.so")
    return ("libt5d.so",)


def repository_root() -> Path:
    """The repository root: this file is ``python/t5d/engine/capi.py``."""
    return Path(__file__).resolve().parents[3]


def default_build_dir() -> Path:
    """Where ``py engine/build.py`` puts the library: ``build/engine`` under the root."""
    return repository_root() / "build" / "engine"


def _in_directory(directory: Path) -> Path | None:
    for name in _library_names():
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def find_library(path: str | os.PathLike | None = None) -> Path:
    """The engine library's path (see the module docstring); ``EngineNotFoundError`` if none."""
    names = " or ".join(_library_names())
    if path is not None:
        p = Path(path)
        found = _in_directory(p) if p.is_dir() else (p if p.is_file() else None)
        if found is None:
            raise EngineNotFoundError(f"no engine library at {p} (expected a file, or a directory holding {names})")
        return found
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        p = Path(env)
        found = _in_directory(p) if p.is_dir() else (p if p.is_file() else None)
        if found is None:
            raise EngineNotFoundError(f"{ENV_VAR}={env} names no engine library (expected a file, or a directory holding {names})")
        return found
    build = default_build_dir()
    found = _in_directory(build)
    if found is None:
        raise EngineNotFoundError(
            f"the engine library ({names}) is not built: looked in {build} and {ENV_VAR} is unset. "
            "Build it with `py engine/build.py` (python3 elsewhere) from the repository root, "
            f"or set {ENV_VAR} to the library's path."
        )
    return found


class Library:
    """The loaded shared library with every C API function declared (``lib.fn.t5d_...``)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.fn = ctypes.CDLL(str(path))
        except OSError as exc:
            raise EngineNotFoundError(f"cannot load the engine library {path}: {exc}") from exc
        missing = []
        for name, (restype, argtypes) in _SIGNATURES.items():
            try:
                f = getattr(self.fn, name)
            except AttributeError:
                missing.append(name)
                continue
            f.restype = restype
            f.argtypes = argtypes
        if missing:
            raise EngineNotFoundError(f"{path} is not the engine's C API (or is out of date): it lacks {', '.join(missing)}")
        self.version = self.fn.t5d_version().decode("ascii", "replace")

    def __repr__(self) -> str:
        return f"Library({str(self.path)!r}, version={self.version!r})"

    # -- calling conventions --

    def take_string(self, pointer: c_void_p | int | None) -> str:
        """Copy a library-allocated string and free it."""
        address = pointer.value if isinstance(pointer, c_void_p) else pointer
        if not address:
            return ""
        try:
            return ctypes.string_at(address).decode("utf-8", "replace")
        finally:
            self.fn.t5d_free(address)

    def call(self, name: str, *args, error: bool = False) -> None:
        """Call ``name``; on a non-zero status raise its exception with the engine's message.
        ``error``: the function takes a trailing ``char** error``, supplied here."""
        f = getattr(self.fn, name)
        if error:
            message = c_void_p()
            status = f(*args, ctypes.byref(message))
            text = self.take_string(message)  # freed whatever the status (NULL on success)
        else:
            status = f(*args)
            text = ""
        if status != T5D_OK:
            raise error_for(status, text, name)


_LOCK = threading.Lock()
_LOADED: dict[Path, Library] = {}


def load_library(path: str | os.PathLike | None = None) -> Library:
    """Find (``find_library``) and load the engine library; one ``Library`` per file."""
    found = find_library(path).resolve()
    with _LOCK:
        lib = _LOADED.get(found)
        if lib is None:
            lib = _LOADED[found] = Library(found)
        return lib


def library_available(path: str | os.PathLike | None = None) -> bool:
    """Whether the engine library can be found and loaded (for skipping tests)."""
    try:
        load_library(path)
    except EngineNotFoundError:
        return False
    return True


def char_array(strings: list[str]) -> ctypes.Array:
    """A ``const char* const*`` of UTF-8 strings (the array keeps the bytes alive)."""
    encoded = [s.encode("utf-8") for s in strings]
    array = (c_char_p * max(1, len(encoded)))()
    for i, b in enumerate(encoded):
        array[i] = b
    return array


__all__ = [
    "BUILT_RELATIONS",
    "CExpansion",
    "ENV_VAR",
    "ExpandOptions",
    "Fingerprint",
    "GameSummary",
    "Library",
    "MOVE_DTYPE",
    "MoveRecord",
    "RESERVED_RELATIONS",
    "StateInfo",
    "char_array",
    "default_build_dir",
    "find_library",
    "library_available",
    "load_library",
    "repository_root",
]
