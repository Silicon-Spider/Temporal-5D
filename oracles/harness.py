#!/usr/bin/env python3
"""Entry point of the oracle harness: `python oracles/harness.py <command>`.

Needs Python 3.12+ and nothing else: the harness has no third-party Python
dependencies and is not installed into the engine's environment.

It runs nothing from the tree but source (decision H33: the comparison logic
lives in harness code the engine adapter cannot modify). Every harness file is
hashed when a gate run starts and when it ends - but Python would also run a
.pyc planted in a __pycache__ directory whose header copies its source file's
time and size, and would import a file in this directory in place of a
standard-library module of the same name, and an extension module (a .pyd or
.so) beside a source file in place of it. None of those changes a hashed file,
and each would run on every later run. So, before anything else is imported:
this directory is not searched ahead of the standard library; no bytecode is
written, and none is read from beside the source (Python's cache is pointed at a
fresh, empty directory of its own); a module in a directory of the harness is
imported from its source file and from nothing else; and the harness's package
is loaded from its source, by path.
"""

import os
import sys

if sys.version_info < (3, 12):  # noqa: UP036 - the point is a clear message on an older interpreter
    sys.exit("the oracle harness needs Python 3.12 or newer")

HERE = os.path.dirname(os.path.abspath(__file__))


def _same(path: str) -> bool:
    return os.path.normcase(os.path.abspath(path or os.curdir)) == os.path.normcase(HERE)


# Python put this script's directory first on the path, where a file in it would
# stand in for any module imported from here on.
sys.path[:] = [entry for entry in sys.path if not _same(entry)]
sys.dont_write_bytecode = True

import atexit  # noqa: E402 - only once the path is safe
import contextlib  # noqa: E402
import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import tempfile  # noqa: E402

sys.pycache_prefix = tempfile.mkdtemp(prefix="oracle-harness-pyc-")


@atexit.register
def _forget_the_cache(folder: str = sys.pycache_prefix) -> None:
    with contextlib.suppress(OSError):
        os.rmdir(folder)  # empty: nothing is ever written to it


def _source_only(path: str) -> importlib.machinery.FileFinder:
    """The finder for a directory of the harness: it imports a module from its source file, and from nothing else.

    Python's own finder would load an extension module, or a .pyc with no
    source, from the same directory - an extension module ahead of the source.
    """
    folder = os.path.normcase(os.path.abspath(path or os.curdir))
    root = os.path.normcase(HERE)
    if folder != root and not folder.startswith(root + os.sep):
        raise ImportError("not a directory of the oracle harness")  # the next hook's: Python's own finder
    return importlib.machinery.FileFinder(path, (importlib.machinery.SourceFileLoader,
                                                 importlib.machinery.SOURCE_SUFFIXES))


sys.path_hooks.insert(0, _source_only)
sys.path_importer_cache.clear()

_PACKAGE = os.path.join(HERE, "oracle_harness")
_spec = importlib.util.spec_from_file_location("oracle_harness", os.path.join(_PACKAGE, "__init__.py"),
                                               submodule_search_locations=[_PACKAGE])
_module = importlib.util.module_from_spec(_spec)
sys.modules["oracle_harness"] = _module
_spec.loader.exec_module(_module)

from oracle_harness.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
