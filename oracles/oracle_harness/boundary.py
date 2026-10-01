"""The boundary between the engine and the oracle harness.

An oracle is worth exactly as much as its independence from the thing it checks.
Every component the engine reuses from an oracle forfeits that oracle for what
the component covers, and the path of least resistance under pressure is to
import, link, fetch or paste. So the boundary is enforced, not requested:

  everything under oracles/ may use the engine;
  nothing outside oracles/ may use anything under oracles/.

Every file outside the harness is checked; what is looked for depends on what
the file is (gate.toml [boundary] sorts them):

  source         code: may not name a path into the harness, import the harness,
                 fetch or import an oracle, or contain code copied from one
  manifest       build files and scripts: the same, and may not list the harness
                 directory or name an oracle's package at all
  plumbing       CI and editor configuration, which has to name the harness (CI
                 runs it): may not fetch or name an oracle's repository or
                 package, or contain copied code. Git's own files (.gitmodules
                 pins the oracles) are not checked
  documentation  may cite anything; may not contain copied code

The rules:

  vendored-code         third-party files committed under third_party/ instead
                        of pinned as a submodule
  stray-submodule       a submodule under third_party/ that the lock file does not pin
  embedded-repository   a git repository inside the tree that is not a submodule
  references-harness    a path into the harness, or to the harness directory itself
  imports-harness       the harness package, by name or as a module
  harness-in-build      a build manifest lists the harness directory
  depends-on-oracle     an oracle's repository, anywhere; its package, in a
                        manifest or an import; an oracle as a submodule
  symlink-into-harness  a link from outside the harness into it
  copied-oracle-code    a run of lines identical to an oracle's source, after
                        white space and quote style are set aside - or one
                        long line (a data table, say) identical to one of its
  unscanned-core-file   a file that could not be read as text, so nothing above
                        was checked in it
  copy-check-skipped    a pinned repository is not checked out, so copies of its
                        code could not be looked for
  waiver-unused         a waiver in gate.toml that no longer excuses anything

Comments are taken out before a file is searched, by the comment syntax of its
type, so citing the harness or an oracle in a comment is fine - except for the
comment forms that are really instructions to a tool (TypeScript's ///, Go's
//go:, Haskell's {-# #-}, Python's "# /// script" blocks), which are searched.
Two references into the harness are allowed in code: its manual,
oracles/README.md, and - in a recipe file (Makefile, justfile, package.json,
tox.ini) - the gate's entry point oracles/harness.py, run with any command but
`observe` and `golden`, which print oracle answers. A command that changes into
the harness (cd oracles, make -C oracles) names it as a path does. A directory
list whose key says it excludes (exclude, ignore, ...) may name the harness
directory: that is how an honest build keeps it out. So may code beside a
package or directory of the engine's own that happens to share the name.

What this cannot see is a port: an oracle's logic retyped in another language,
or reshaped enough that no run of lines survives. That is the same offence, and
it is left to review (CONTRIBUTING.md).
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

from . import gitutil, paths
from .config import Boundary
from .lockfile import Lock, normalise_url, read_gitmodules
from .util import Finding

_MAX_BYTES = 2 * 1024 * 1024
_BINARY_PROBE = 8000
_REUSED = "an oracle reused by the engine is no longer an oracle"
_NEVER = "the harness is never a dependency of the engine"
_REACH = "the engine must not reach into the oracle harness"
_CODE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl", ".ipp",
    ".hs", ".lhs", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx",
    ".py", ".pyx", ".rs", ".go", ".java", ".kt", ".cs", ".swift", ".zig",
}
# Git's own files. They name the oracles because they pin them.
_GIT_FILES = {".gitmodules", ".gitignore", ".gitattributes", ".ignore", ".mailmap"}
# Files whose job is to run commands: the only place the gate's entry point may be named.
_RECIPE_NAMES = {"makefile", "gnumakefile", "justfile", "package.json", "tox.ini"}
_RECIPE_SUFFIXES = {".mk"}
# Commands of the entry point that print what an oracle says.
_ORACLE_OUTPUT_COMMANDS = {"observe", "golden"}
# Lines that carry no logic of their own: comments (licence headers included),
# preprocessor lines, and import lists. They never count towards a copied run.
_NOT_LOGIC = re.compile(
    r"^(//|/\*|\*|#|--|\{-|-\}|<!--|\"\"\"|'''"
    r"|import\b|from\b.*\bimport\b|using\b|use\b|package\b|module\b|export\s*\{"
    r"|(const|let|var)\s+[\w{}, ]+\s*=\s*require\()"
)
# A key whose list keeps things out: naming the harness there is the honest way to exclude it.
_EXCLUDING = re.compile(r"(?i)(exclude|ignore|norecursedirs|omit|skip|prune)")
# A value that is prose, inside a manifest: a description, a list of keywords.
_PROSE = re.compile(r"""(?i)(?:^|[\s{,])["']?(?:description|summary|synopsis|about|keywords|long_description|readme)["']?"""
                    r"""\s*[:=]\s*(?:(?P<quote>["'])(?P<quoted>(?:\\.|(?!(?P=quote)).)*)(?P=quote)|(?P<bare>[^\n]*))""")

# --- comment syntax, by file type ---------------------------------------------

_FAMILY_BY_SUFFIX = {
    **dict.fromkeys((".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl", ".ipp", ".m", ".mm",
                     ".cu", ".cuh"), "c"),
    ".rs": "rust",
    **dict.fromkeys((".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".mts", ".cts"), "js"),
    **dict.fromkeys((".java", ".kt", ".kts", ".scala", ".groovy", ".gradle", ".go", ".cs", ".swift", ".dart", ".php",
                     ".zig", ".proto", ".css", ".scss", ".less", ".jsonc", ".json5"), "slash"),
    **dict.fromkeys((".py", ".pyi", ".pyx"), "python"),
    **dict.fromkeys((".sh", ".bash", ".zsh", ".fish", ".ps1", ".psm1", ".rb", ".pl", ".pm", ".r", ".jl", ".tcl",
                     ".toml", ".yaml", ".yml", ".cfg", ".ini", ".conf", ".cmake", ".mk", ".nix", ".bzl", ".bazel",
                     ".txt", ".lock", ".properties", ".env", ".dockerfile", ".containerfile"), "hash"),
    **dict.fromkeys((".hs", ".elm", ".purs"), "haskell"),
    ".lhs": "literate-haskell",
    ".cabal": "cabal",
    ".lua": "lua",
    ".sql": "sql",
    **dict.fromkeys((".ads", ".adb"), "ada"),
    **dict.fromkeys((".bat", ".cmd"), "batch"),
    **dict.fromkeys((".html", ".htm", ".xml", ".xhtml", ".svg", ".xaml", ".csproj", ".vcxproj", ".fsproj", ".props",
                     ".targets", ".plist", ".xsd"), "markup"),
}
_FAMILY_BY_NAME = {
    "go.mod": "slash", "go.work": "slash",
    "makefile": "hash", "gnumakefile": "hash", "justfile": "hash", "cmakelists.txt": "hash", "build": "hash",
    "workspace": "hash", "meson.build": "hash", "pipfile": "hash", "gemfile": "hash", "rakefile": "hash",
    "sconstruct": "python", "sconscript": "python",
    "cabal.project": "cabal", "cabal.project.local": "cabal",
}


def comment_family(path: str) -> str:
    """Which comment syntax a file uses; "none" when this check does not know it."""
    name = posixpath.basename(path).lower()
    if name.endswith(".in"):
        name = name[:-3]  # a template: it has the syntax of what it becomes
    if name in _FAMILY_BY_NAME:
        return _FAMILY_BY_NAME[name]
    if name.startswith(("dockerfile", "containerfile")):
        return "hash"
    return _FAMILY_BY_SUFFIX.get(posixpath.splitext(name)[1], "none")


_DOUBLE = r'"(?:\\.|[^"\\\n])*"'
_SINGLE = r"'(?:\\.|[^'\\\n])*'"


def _tokens(*keep: str, drop: tuple[str, ...]) -> re.Pattern:
    """One pattern per family: what is kept as written (strings, tool directives), then the comments.

    At any one position the first alternative that matches wins, so a comment
    leader inside a string is never taken for a comment. A quote that opens no
    string on its line (an apostrophe, a lifetime, a prime) matches nothing and
    is passed over. A block comment that is never closed is not a comment.
    """
    return re.compile(f"(?P<keep>{'|'.join(keep)})|(?P<drop>{'|'.join(drop)})", re.S | re.M)


_BLOCK = r"/\*.*?\*/"
_TOKENS = {
    "c": _tokens(r'R"(?P<delimiter>[^()\\\s]{0,16})\(.*?\)(?P=delimiter)"', _DOUBLE, r"'(?:\\.|[^'\\\n])+'",
                 drop=(r"//[^\n]*", _BLOCK)),
    "rust": _tokens(r'b?r(?P<hashes>#*)".*?"(?P=hashes)', r'b?"(?:\\.|[^"\\])*"', r"b?'(?:\\.|[^'\\\n])'",
                    drop=(r"//[^\n]*", _BLOCK)),
    "js": _tokens(_DOUBLE, _SINGLE, r"`(?:\\.|[^`\\])*`",
                  # a regular expression literal, where one can start
                  r"(?:^|(?<=[(,=:\[!&|?{};+\-*%<>~^]))[ \t]*/(?![/*])(?:\\.|\[(?:\\.|[^\]\\\n])*\]|[^/\\\n\[])+/[A-Za-z]*",
                  r"///[^\n]*",  # a triple-slash directive: <reference path="..."/>
                  drop=(r"//[^\n]*", _BLOCK)),
    "slash": _tokens(r'"""(?:\\.|.)*?"""', _DOUBLE, _SINGLE, r"`[^`]*`",
                     r"//[a-z]+:[^\n]*",  # a directive: //go:embed, //go:generate
                     drop=(r"//[^\n]*", _BLOCK)),
    "python": _tokens(r"^\# /// [a-z]+\n(?:\#(?: [^\n]*)?\n)*?\# ///$",  # an inline script's metadata (PEP 723)
                      r'[rRbBuUfF]{0,2}"""(?:\\.|.)*?(?:"""|\Z)', r"[rRbBuUfF]{0,2}'''(?:\\.|.)*?(?:'''|\Z)",
                      _DOUBLE, _SINGLE,
                      # "#" opens a comment only where a word starts, and only when a
                      # space or another "#" follows: "#include", "#[path]" and "$#" are not.
                      drop=(r"(?<!\S)\#(?=[\s#!]|$)[^\n]*",)),
    "hash": _tokens(r'"""(?:\\.|.)*?(?:"""|\Z)', r"'''(?:\\.|.)*?(?:'''|\Z)", _DOUBLE, _SINGLE,
                    drop=(r"(?<!\S)\#(?=[\s#!]|$)[^\n]*",)),
    "haskell": _tokens(r"\{-\#.*?\#-\}", _DOUBLE, r"'(?:\\.|[^'\\\n])'",
                       drop=(r"(?<!\S)--++(?![!#$%&*+./<=>?@\\^|~:])[^\n]*", r"\{-.*?-\}")),
    "cabal": _tokens(r"(?!)", drop=(r"^[ \t]*--[^\n]*",)),
    "lua": _tokens(r"\[(?P<level>=*)\[.*?\](?P=level)\]", _DOUBLE, _SINGLE,
                   drop=(r"--\[(?P<comment>=*)\[.*?\](?P=comment)\]", r"--[^\n]*")),
    "sql": _tokens(r"'(?:''|[^'])*'", r'"(?:""|[^"])*"', drop=(r"--[^\n]*", _BLOCK)),
    "ada": _tokens(r'"(?:""|[^"\n])*"', drop=(r"--[^\n]*",)),
    "batch": _tokens(r"(?!)", drop=(r"(?i:^[ \t]*@?(?:rem(?:[ \t][^\n]*)?|::[^\n]*)$)",)),
    "markup": _tokens(r"(?!)", drop=(r"<!--.*?-->",)),
}


def _blanked(token: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in token)


def _docstring(text: str, start: int) -> bool:
    """Does the string starting at ``start`` stand where Python takes it for a docstring?"""
    line_start = text.rfind("\n", 0, start) + 1
    if not re.fullmatch(r"\s*[rRuU]?", text[line_start:start]):
        return False
    for line in reversed(text[:line_start].split("\n")):
        code = re.sub(r"(?<!\S)#.*$", "", line).rstrip()
        if code:
            return code.endswith(":")
    return True  # the module's own docstring


def strip_comments(text: str, family: str) -> str:
    """The text with its comments blanked out. Line numbers and string literals are untouched.

    A Python docstring counts as a comment: it is prose, and nothing runs it.
    """
    if family == "literate-haskell":
        # Only the lines marked with ">" are code; the rest is the commentary.
        lines = [" " + line[1:] if line.startswith(">") else _blanked(line) for line in text.split("\n")]
        return strip_comments("\n".join(lines), "haskell")
    pattern = _TOKENS.get(family)
    if pattern is None:
        return text

    def replace(match: re.Match) -> str:
        if match.group("drop") is not None:
            return _blanked(match.group(0))
        token = match.group(0)
        if family == "python" and token.lstrip("rRbBuUfF").startswith(('"""', "'''")) and _docstring(match.string, match.start()):
            return _blanked(token)
        return token

    return pattern.sub(replace, text)


_STRING = re.compile(r"""(["'`])(?:\\.|(?!\1)[^\\\n])*\1""")


def _without_strings(code: str) -> str:
    """The code with the contents of its one-line string literals blanked out."""
    return _STRING.sub(lambda match: match.group(1) + " " * (len(match.group(0)) - 2) + match.group(1), code)


# --- the files ----------------------------------------------------------------

@dataclass(frozen=True)
class CoreFile:
    path: str  # repository-relative, forward slashes
    kind: str  # source | manifest | plumbing | documentation
    text: str  # line endings normalised to \n


@dataclass(frozen=True)
class _Hit:
    rule: str
    path: str
    line: int  # 0 when the finding is about the file as a whole
    text: str  # the offending text (the path, for a whole-file finding): what a waiver names
    message: str

    def finding(self) -> Finding:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return Finding(self.rule, f"{where}: {self.message}")


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    # Case does not matter: on Windows and macOS it does not matter to the tools either.
    path = path.lower()
    name = posixpath.basename(path)
    return any(fnmatch.fnmatchcase(path, pattern.lower()) or fnmatch.fnmatchcase(name, pattern.lower()) for pattern in patterns)


def classify(path: str, config: Boundary, harness_rel: str) -> str:
    if path.lower() == harness_rel.lower() or path.lower().startswith(harness_rel.lower() + "/"):
        return "harness"
    if _matches(path, config.plumbing):
        return "plumbing"
    if _matches(path, config.documentation):
        return "documentation"
    if _matches(path, config.manifests):
        return "manifest"
    return "source"


def _read_text(path: Path) -> tuple[str | None, str]:
    """(text, "") or (None, why not). Line endings come back as \\n.

    Binary is recognised before size, so a large binary file (a network's
    weights, say) is passed over as binary rather than reported as unread.
    """
    utf16 = (b"\xff\xfe", b"\xfe\xff")
    try:
        with path.open("rb") as stream:
            head = stream.read(_BINARY_PROBE)
            if not head.startswith(utf16) and b"\0" in head:
                return None, "it looks binary"
            if os.fstat(stream.fileno()).st_size > _MAX_BYTES:
                return None, f"it is larger than {_MAX_BYTES // (1024 * 1024)} MiB"
            data = head + stream.read()
    except OSError as error:
        return None, f"it could not be read ({error.strerror or error})"
    text = data.decode("utf-16", "replace") if data.startswith(utf16) else data.decode("utf-8-sig", "replace").replace("\0", " ")
    return text.replace("\r\n", "\n").replace("\r", "\n"), ""


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _name_pattern(name: str) -> str:
    """A package name as package managers compare them: "-", "_" and "." are one."""
    return "".join("[-_.]" if char in "-_." else re.escape(char) for char in name)


# --- copied code --------------------------------------------------------------

def significant_lines(text: str, min_chars: int, literate: bool = False) -> list[tuple[int, str]]:
    """(line number, normalised line) for every line that is logic.

    White space is dropped and quotes are unified before lines are compared, so
    re-indenting, re-spacing or re-quoting a copy does not hide it. In literate
    Haskell only the lines marked ">" are code.
    """
    lines = []
    for number, written in enumerate(text.split("\n"), 1):
        if literate and not written.startswith(">"):
            continue
        collapsed = " ".join((written[1:] if literate else written).split())
        if len(collapsed) < min_chars or _NOT_LOGIC.match(collapsed) or not re.search(r"[A-Za-z0-9]", collapsed):
            continue
        lines.append((number, "".join(collapsed.split()).replace("'", '"')))
    return lines


def _windows(lines: list[tuple[int, str]], size: int):
    """(hash, first line number, last line number) of every run of ``size`` logic lines."""
    for start in range(len(lines) - size + 1):
        chunk = "\n".join(line for _, line in lines[start:start + size])
        yield hashlib.sha1(chunk.encode("utf-8", "replace")).digest(), lines[start][0], lines[start + size - 1][0]


def _line_digest(line: str) -> bytes:
    """The hash of one long logic line: a data table, say, copied whole."""
    return hashlib.sha1(("\0line\0" + line).encode("utf-8", "replace")).digest()


def oracle_fingerprints(lock: Lock, config: Boundary, repo_root: Path) -> tuple[dict[bytes, str], list[str]]:
    """Hash of every run of ``copy_window_lines`` logic lines in the pinned sources -> where it is.

    Every logic line of at least ``copy_long_line_chars`` characters (white space
    not counted) is fingerprinted on its own as well. Also returns the ids of the
    pins that could not be read: a repository that is not checked out cannot be
    searched for, and that must not pass silently.
    """
    index: dict[bytes, str] = {}
    unread: list[str] = []
    for pin in lock.pins.values():
        checkout = repo_root / pin.path
        if not gitutil.is_checkout(checkout) or not gitutil.has_commit(checkout, pin.commit):
            unread.append(pin.id)
            continue
        for mode, oid, path in gitutil.list_tree(checkout, pin.commit):
            suffix = posixpath.splitext(path)[1].lower()
            if not mode.startswith("100") or suffix not in _CODE_SUFFIXES:
                continue
            data = gitutil.read_blob(checkout, oid)
            if len(data) > _MAX_BYTES or b"\0" in data:
                continue
            text = data.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")
            lines = significant_lines(text, config.copy_min_line_chars, literate=suffix == ".lhs")
            for digest, line, _ in _windows(lines, config.copy_window_lines):
                index.setdefault(digest, f"{pin.upstream}:{path}:{line}")
            for line, key in lines:
                if len(key) >= config.copy_long_line_chars:
                    index.setdefault(_line_digest(key), f"{pin.upstream}:{path}:{line}")
    return index, unread


# --- what a core file says ----------------------------------------------------

_DELIMITERS = set(" \t\n\"'`<>|,;()[]{}=")


def _token_around(text: str, start: int, end: int) -> str:
    """The whole path-like token a match sits in."""
    while start > 0 and text[start - 1] not in _DELIMITERS:
        start -= 1
    while end < len(text) and text[end] not in _DELIMITERS:
        end += 1
    return text[start:end]


def _is_harness(token: str, core_path: str, repo_root: Path, harness_rel: str) -> bool:
    """Could this path be the harness, or something in it?

    Only a plain relative path that exists, next to the file that names it, and
    lies outside the harness is cleared - the engine may have a directory of its
    own with the same name. A path built from a variable could be anything, so it
    counts as the harness.
    """
    token = re.sub(r"^-[A-Za-z]+(?=[./\\\w])", "", token.strip())  # -I../x, -L../x
    token = re.sub(r"^file://", "", token)
    if re.match(r"(?i)^[a-z][a-z0-9+.-]*://", token):
        return False  # a URL of some other place
    if any(char in token for char in "$%{}@~*?"):
        return True
    candidate = Path(token) if os.path.isabs(token) else repo_root / posixpath.dirname(core_path) / token
    try:
        if not candidate.exists():
            return True
        resolved = os.path.normcase(str(candidate.resolve()))
        harness = os.path.normcase(str((repo_root / harness_rel).resolve()))
    except OSError:
        return True
    return resolved == harness or resolved.startswith(harness + os.sep)


def _excluding(code: str, offset: int) -> bool:
    """Is this in a list whose key keeps things out - on this line or the three before it?"""
    line_start = code.rfind("\n", 0, offset) + 1
    context_start = line_start
    for _ in range(3):
        if context_start == 0:
            break
        context_start = code.rfind("\n", 0, context_start - 1) + 1
    return _EXCLUDING.search(code, context_start, offset) is not None


def _line_text(code: str, offset: int) -> str:
    start = code.rfind("\n", 0, offset) + 1
    end = code.find("\n", offset)
    return code[start:len(code) if end < 0 else end]


def _in_prose(code: str, offset: int) -> bool:
    """Is this inside the value of a description, summary or keywords entry?"""
    start = code.rfind("\n", 0, offset) + 1
    line = _line_text(code, offset)
    for match in _PROSE.finditer(line):
        group = "quoted" if match.group("quote") else "bare"
        if match.start(group) <= offset - start < match.end(group):
            return True
    return False


def _in_table(code: str, offset: int, names: tuple[str, ...]) -> bool:
    """Is this under one of these [table] headers of a TOML file?"""
    headers = re.findall(r"(?m)^\s*\[+\s*([\w.\-]+)\s*\]+\s*$", code[:offset])
    return bool(headers) and headers[-1].lower() in names


def _recipe(core: CoreFile) -> bool:
    name = posixpath.basename(core.path).lower()
    return core.kind == "manifest" and (name in _RECIPE_NAMES or posixpath.splitext(name)[1] in _RECIPE_SUFFIXES)


def _runs_the_gate(core: CoreFile, code: str, start: int, end: int, harness_rel: str) -> bool:
    """Is ``oracles/harness.py`` here a recipe running a command that prints no oracle answer?"""
    if not _recipe(core):
        return False
    command = re.match(r"\s+([a-z-]+)", code[end:])
    return command is not None and command.group(1) not in _ORACLE_OUTPUT_COMMANDS


def _check_text(core: CoreFile, lock: Lock, harness_rel: str, repo_root: Path) -> list[_Hit]:
    code = strip_comments(core.text, comment_family(core.path))
    hits: dict[tuple[str, int, str], _Hit] = {}
    rel = re.escape(harness_rel)
    suffix = posixpath.splitext(core.path)[1].lower()
    manifest = core.kind == "manifest"

    def report(rule: str, start: int, end: int, message: str) -> None:
        line = _line_of(code, start)
        span = code[start:end]
        hits.setdefault((rule, line, span), _Hit(rule, core.path, line, span, message))

    gate_lines: set[int] = set()  # lines on which a recipe runs the gate
    if core.kind in ("source", "manifest"):
        # A path into the harness.
        inside: list[tuple[int, int]] = []
        for match in re.finditer(rf"(?<![A-Za-z0-9_.\-]){rel}[/\\][^\s\"'`<>|,;:(){{}}\[\]]*", code, re.I):
            inside.append(match.span())
            target = re.sub(r"/+", "/", match.group(0).replace("\\", "/")).rstrip("/.")
            if target.lower() == f"{harness_rel}/readme.md".lower():
                continue  # the harness's manual may be cited
            if target.lower() == f"{harness_rel}/harness.py" and _runs_the_gate(core, code, *match.span(), harness_rel):
                gate_lines.add(_line_of(code, match.start()))
                continue
            if _excluding(code, match.start()):
                continue
            token = _token_around(code, *match.span())
            if not _is_harness(token, core.path, repo_root, harness_rel):
                continue
            report("references-harness", *match.span(), f"names {match.group(0)}, inside {harness_rel}/ - {_REACH}")

        rule = "harness-in-build" if manifest else "references-harness"
        whole: list[tuple[int, int]] = []  # places the two rules below have judged, either way

        # The harness directory as a whole string: "oracles", "./oracles".
        for match in re.finditer(rf"""(["'`])((?:\.{{0,2}}[/\\])*{rel})\1""", code, re.I):
            whole.append(match.span())
            line = _line_text(code, match.start())
            key = re.match(r"\s*:", code[match.end():])  # {"oracles": ...} is a key, not a path
            # A Cargo feature, a cabal flag: a name, not a place.
            feature = re.search(r"\b(feature|cfg|flag)\b", line) or _in_table(code, match.start(), ("features",))
            if key or feature or _excluding(code, match.start()) or (manifest and _in_prose(code, match.start())):
                continue
            if not _is_harness(match.group(2), core.path, repo_root, harness_rel):
                continue
            report(rule, *match.span(), f"names the {harness_rel} directory - {_REACH}")

        # The harness directory as a bare word, in build files that do not quote
        # what they list: add_subdirectory(oracles), SUBDIRS = a oracles, - oracles, use ./oracles.
        if manifest and suffix not in (".toml", ".json", ".lock"):
            unquoted = _without_strings(code)
            for match in re.finditer(rf"(?<![\w.\-/\\])(?:\.{{1,2}}[/\\])*{rel}(?![\w.\-/\\])", unquoted, re.I):
                whole.append(match.span())
                line = _line_text(unquoted, match.start())
                target = re.match(rf"(?i)^\s*{rel}\s*:", line)  # a make or just target named after it
                flag = re.search(r"\bflag\b", line)  # a cabal flag
                if target or flag or _in_prose(unquoted, match.start()) or _excluding(unquoted, match.start()):
                    continue
                if not _is_harness(match.group(0), core.path, repo_root, harness_rel):
                    continue
                report("harness-in-build", *match.span(),
                       f"lists the {harness_rel} directory - the harness is a separate project, never part of the engine's build")

        # A path that ends in the harness directory: ../oracles, ${ROOT}/oracles, -I../oracles.
        for match in re.finditer(rf"(?<=[/\\]){rel}(?![\w.\-/\\])", code, re.I):
            if any(start <= match.start() < end for start, end in whole):
                continue
            whole.append(match.span())
            token = _token_around(code, *match.span())
            if _excluding(code, match.start()) or not _is_harness(token, core.path, repo_root, harness_rel):
                continue
            report(rule, *match.span(), f"names {token}, the {harness_rel} directory - {_REACH}")

        # A command that changes into the harness, inside a string or a script:
        # cd oracles && ..., pushd ./oracles, make -C oracles, npm --prefix oracles.
        changes_into = re.compile(
            rf"""(?:(?<![\w\-])(?:cd|pushd|chdir|set-location)|(?<![\w\-])-C|--directory|--cwd|--prefix)"""
            rf"""(?:[ \t]+|[ \t]*=[ \t]*)["'`]?(?P<dir>(?:\.{{1,2}}[/\\])*{rel})(?![\w.\-/\\])""", re.I)
        for match in changes_into.finditer(code):
            start, end = match.span("dir")
            if any(low <= start < high for low, high in whole):
                continue
            if _excluding(code, start) or not _is_harness(match.group("dir"), core.path, repo_root, harness_rel):
                continue
            report("references-harness", start, end, f"changes into the {harness_rel} directory - {_REACH}")

        # The harness package, by name or as a module.
        for match in re.finditer(r"(?<![A-Za-z0-9_])(?:oracle[_-]harness|ORACLE_HARNESS_CACHE)(?![A-Za-z0-9_])", code, re.I):
            if not any(start <= match.start() < end for start, end in inside):
                report("imports-harness", *match.span(), f"names {match.group(0)} - {_NEVER}")
        # A package of the engine's own with the same name, next to the file, is not the harness.
        own_package = not _is_harness(harness_rel, core.path, repo_root, harness_rel)
        for match in re.finditer(rf"(?<![\w.]){rel}\.harness\b", code, re.I):
            if not own_package:
                report("imports-harness", *match.span(), f"names the module {match.group(0)} - {_NEVER}")
        if comment_family(core.path) == "python" and not own_package:
            imported = re.compile(
                rf"^[ \t]*(?:from[ \t]+{rel}(?:\.[\w.]+)?[ \t]+import\b"
                rf"|import[ \t]+(?:[\w.]+(?:[ \t]+as[ \t]+\w+)?[ \t]*,[ \t]*)*{rel}(?![\w]))", re.M)
            for match in imported.finditer(code):
                report("imports-harness", *match.span(), f"imports {harness_rel} as a package - {_NEVER}")

    # The oracles: their repositories, anywhere but in prose; their packages.
    for pin in lock.pins.values() if core.kind != "documentation" else ():
        _, _, slug = normalise_url(pin.url).partition("/")
        owner, _, repository = slug.partition("/")
        repository_pattern = re.compile(
            rf"(?<![\w.\-]){re.escape(owner)}(?:/|%2f|:){re.escape(repository)}(?![\w\-])", re.I)
        for match in repository_pattern.finditer(code):
            if manifest and _in_prose(code, match.start()):
                continue
            report("depends-on-oracle", *match.span(), f"names {pin.upstream}'s repository - {_REUSED}")
        for name in pin.dependency_names:
            pattern = _name_pattern(name)
            word = re.compile(rf"(?<![A-Za-z0-9_.\-]){pattern}(?![A-Za-z0-9_]|[-.][A-Za-z])", re.I)
            if core.kind in ("manifest", "plumbing"):
                for match in word.finditer(code):
                    line_number = _line_of(code, match.start())
                    if line_number in gate_lines or _in_prose(code, match.start()):
                        continue  # the gate's own command line names oracles; prose may too
                    report("depends-on-oracle", *match.span(), f"names the package {name} ({pin.upstream}) - {_REUSED}")
            elif core.kind == "source":
                # An inline script's own dependency list (PEP 723) is a manifest.
                for block in re.finditer(r"(?m)^# /// [a-z]+\n(?:#(?: [^\n]*)?\n)*?# ///$", code):
                    for match in word.finditer(code, block.start(), block.end()):
                        report("depends-on-oracle", *match.span(), f"names the package {name} ({pin.upstream}) - {_REUSED}")
                imported = re.compile(
                    rf"""(?:\brequire(?:\.resolve)?\s*\(\s*|\bimport\s*\(\s*|\bfrom\s+|\bimport\s+)['"`](?:npm:|jsr:)?{pattern}['"`/@]"""
                    rf"""|^[ \t]*(?:import|from)[ \t]+(?:qualified[ \t]+)?{pattern}(?![\w\-])"""
                    rf"""|\bnpm:{pattern}\b|(?:esm\.sh|unpkg\.com|cdn\.jsdelivr\.net/npm|cdn\.skypack\.dev|jspm\.dev)/{pattern}\b"""
                    rf"""|\#\s*include\s*[<"]{pattern}/|\#\[link\s*\(\s*name\s*=\s*"{pattern}"|rustc-link-lib=(?:\w+=)?{pattern}\b"""
                    rf"""|(?<![\w\-])-l{pattern}\b""", re.M | re.I)
                for match in imported.finditer(code):
                    report("depends-on-oracle", *match.span(), f"uses the package {name} ({pin.upstream}) - {_REUSED}")
    return [hits[key] for key in sorted(hits, key=lambda key: (key[1], key[0], key[2]))]


def _link_target(absolute: Path) -> str:
    try:
        return os.readlink(absolute)
    except OSError:
        # Where links are checked out as plain files, the file holds the target.
        text, _ = _read_text(absolute)
        return (text or "").strip()


def _links_into_harness(path: str, absolute: Path, repo_root: Path, harness_rel: str) -> str | None:
    """Where the link at ``path`` leads, when that is inside the harness."""
    target = _link_target(absolute).replace("\\", "/")
    harness = harness_rel.lower()
    if not os.path.isabs(target):
        spelled = posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
        if spelled.lower() == harness or spelled.lower().startswith(harness + "/"):
            return spelled
    try:
        # Resolved on disk as well: a link can leave the repository and come back in by its name.
        resolved = os.path.normcase(str((absolute.parent / target).resolve()))
        root = os.path.normcase(str((repo_root / harness_rel).resolve()))
    except (OSError, ValueError):
        return None
    return target if resolved == root or resolved.startswith(root + os.sep) else None


def check(lock: Lock, config: Boundary, repo_root: Path | None = None, *, include_untracked: bool = True,
          harness_rel: str | None = None) -> list[Finding]:
    repo_root = repo_root or paths.REPO_ROOT
    harness_rel = harness_rel or paths.HARNESS_REL
    third_party_rel = f"{harness_rel}/third_party"
    locked_paths = {pin.path for pin in lock.pins.values()}
    oracle_urls = {normalise_url(pin.url): pin for pin in lock.pins.values()}
    submodule_urls = {module.get("path", "").replace("\\", "/"): module.get("url", "")
                      for module in read_gitmodules(repo_root).values()}
    hits: list[_Hit] = []
    core_files: list[CoreFile] = []

    for mode, path in gitutil.tracked_files(repo_root, include_untracked=include_untracked):
        if path.endswith("/"):
            # git lists a repository it finds inside the work tree as a directory.
            hits.append(_Hit("embedded-repository", path, 0, path,
                             "a git repository inside the tree that is not a submodule - an oracle is pinned under "
                             f"{third_party_rel}/, and the engine's own dependencies are added as submodules"))
            continue
        if mode == "160000":
            if path.startswith(third_party_rel + "/"):
                if path not in locked_paths:
                    hits.append(_Hit("stray-submodule", path, 0, path, "a submodule under third_party/ that the lock file does not pin"))
            else:
                pin = oracle_urls.get(normalise_url(submodule_urls.get(path, "")))
                if pin is not None:
                    hits.append(_Hit("depends-on-oracle", path, 0, path,
                                     f"checks out {pin.upstream} as a submodule outside the harness - {_REUSED}"))
            continue
        if path.startswith(third_party_rel + "/"):
            hits.append(_Hit("vendored-code", path, 0, path, "third-party files are pinned as submodules, never committed into the tree"))
            continue
        kind = classify(path, config, harness_rel)
        if kind == "harness" or (kind == "plumbing" and posixpath.basename(path) in _GIT_FILES):
            continue
        absolute = repo_root / path
        if mode == "120000" or absolute.is_symlink():
            leads = _links_into_harness(path, absolute, repo_root, harness_rel)
            if leads:
                hits.append(_Hit("symlink-into-harness", path, 0, path, f"links to {leads}"))
            continue
        if not absolute.is_file():
            continue  # tracked but deleted from the work tree
        text, problem = _read_text(absolute)
        if text is None:
            textual = posixpath.splitext(path)[1].lower() in _FAMILY_BY_SUFFIX or comment_family(path) != "none"
            if problem != "it looks binary" or textual:
                hits.append(_Hit("unscanned-core-file", path, 0, path,
                                 f"not checked, because {problem} - make it readable text or waive it"))
            continue
        core_files.append(CoreFile(path, kind, text))

    for core in core_files:
        hits += _check_text(core, lock, harness_rel, repo_root)

    findings: list[Finding] = []
    if core_files:
        index, unread = oracle_fingerprints(lock, config, repo_root)
        for pin_id in unread:
            findings.append(Finding(
                "copy-check-skipped",
                f"{pin_id} is not checked out at its locked commit, so code copied from it would go unnoticed "
                f"(run: python {harness_rel}/harness.py sync)"))
        for core in core_files:
            written = core.text.split("\n")
            lines = significant_lines(core.text, config.copy_min_line_chars,
                                      literate=posixpath.splitext(core.path)[1].lower() == ".lhs")
            covered = 0  # last line of the copied run already reported
            runs: list[tuple[int, int]] = []
            for digest, first, last in _windows(lines, config.copy_window_lines):
                origin = index.get(digest)
                if not origin:
                    continue
                if first > covered:
                    hits.append(_Hit(
                        "copied-oracle-code", core.path, first, written[first - 1].strip(),
                        f"{config.copy_window_lines} consecutive lines identical to {origin} - "
                        "reimplement from the rules, do not copy the oracle"))
                covered = max(covered, last)
                runs.append((first, last))
            for number, key in lines:
                origin = index.get(_line_digest(key)) if len(key) >= config.copy_long_line_chars else None
                if origin and not any(first <= number <= last for first, last in runs):
                    hits.append(_Hit(
                        "copied-oracle-code", core.path, number, written[number - 1].strip(),
                        f"a line of {len(key)} characters identical to {origin} - "
                        "reimplement from the rules, do not copy the oracle"))

    used: set[int] = set()
    for hit in hits:
        excuse = next((number for number, waiver in enumerate(config.waivers)
                       if (waiver.path, waiver.rule) == (hit.path, hit.rule) and waiver.text in hit.text), None)
        if excuse is None:
            findings.append(hit.finding())
        else:
            used.add(excuse)
    for number, waiver in enumerate(config.waivers):
        if number not in used:
            findings.append(Finding(
                "waiver-unused",
                f"gate.toml waives {waiver.rule} in {waiver.path} for the text '{waiver.text}', "
                "and nothing there needs it any more - remove the waiver"))
    return findings
