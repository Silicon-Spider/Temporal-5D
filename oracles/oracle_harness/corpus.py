"""The pinned game-record corpus.

Third-party records are never copied into this repository. The manifest names
each one by pinned source, path, git blob id and SHA-256, and the bytes are read
from that source's object store at its locked commit - so the corpus cannot
change without the manifest changing, and no record's licence is taken on by
redistributing it.

The harness treats a record as opaque text. It never parses 5dpgn: each oracle
reads the record with its own parser, the engine with its own (decision H28),
and positions are aligned by counting submitted actions.

With one exception, ruled at CHECKPOINT 3: a record of cwmtt's written in the
official client's export form, whose moves name their board twice, is converted
to the notation specification's 5dpgn before any source reads it
(export_form.py), and every source - the engine included - reads the converted
text; one with a move the specification cannot express is given to no source.
The manifest pins what the sources read as it pins the stored bytes: such a
record's entry says `converted`, the SHA-256 of the converted text (or
"unsupported"), and feed() refuses to hand over anything else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import export_form, gitutil, paths
from .headers import board_header, declared_board, has_fen_boards
from .lockfile import Lock
from .util import Finding, HarnessError, load_toml, sha256_bytes

__all__ = ["CONVERTED_SOURCE", "CUSTOM_POSITIONS", "EXPORT_FORM", "LOCAL", "TIERS", "UNSUPPORTED", "Excluded", "Feed",
           "Record", "board_header", "declared_board", "has_fen_boards"]

SCHEMA = 1
LOCAL = "local"
TIERS = ("commit", "nightly")
# The ruleset of every record that sets up its own position. Such a record
# exercises the rules, but it is evidence about no named variant.
CUSTOM_POSITIONS = "(custom positions)"
# The pinned source whose records are converted from the official client's
# export form before any source reads them (export_form.py; CHECKPOINT 3).
CONVERTED_SOURCE = "cwmtt"
# A manifest's `converted` for a record in the export form with a move the
# notation specification cannot express: no source is given it.
UNSUPPORTED = "unsupported"
# The reason every source gives for such a record (status "unsupported").
EXPORT_FORM = "export_form"
_ID = re.compile(r"^[a-z0-9]+/[A-Za-z0-9._-]+$")
_RECORD_SUFFIX = ".5dpgn"


@dataclass(frozen=True)
class Record:
    id: str
    source: str  # a pin id from the lock file, or "local"
    path: str  # inside the source repository (or repository-relative for local records)
    blob: str  # git blob id at the source's locked commit; empty for local records
    sha256: str  # of the exact stored bytes
    size: int
    board: str  # the variant it declares: its Board header, else its Variant header; empty when it has neither
    licence: str
    tier: str
    fen: bool = False  # the record sets up its own position with 5DFEN boards
    also_at: tuple[str, ...] = ()
    notes: str = ""
    # Only for a record in the official client's export form (CONVERTED_SOURCE's):
    # the SHA-256 of the 5dpgn text every source reads instead (export_form.py),
    # or UNSUPPORTED where a move of it has no form in the notation specification.
    converted: str = ""

    @property
    def fed_sha256(self) -> str:
        """The SHA-256 of what the sources' answers about this record are about.

        The stored record's, except for a record read as converted from the
        export form: then the converted text's, which is what every source read.
        """
        return self.converted if self.converted not in ("", UNSUPPORTED) else self.sha256

    @property
    def reading(self) -> str:
        """How the sources read this record, where it is not as stored: for reports and DISAGREEMENTS.md."""
        if self.converted == UNSUPPORTED:
            return ("in the official client's export form, with a move the notation specification cannot express: "
                    f"given to no source (every source: unsupported, {EXPORT_FORM})")
        if self.converted:
            return "read as converted from the official client's export form to the notation specification's 5dpgn"
        return ""

    @property
    def ruleset(self) -> str:
        """What the gate's verdict on this record counts towards.

        Exactness is inherited per variant, so the gate answers per variant. A
        record that names no variant is a Standard game (the notation's
        default). One that brings its own 5DFEN position, or names a custom
        set-up ("Custom...", "Even", "Odd" - the names ftxi takes as "the boards
        are in the record"), is not a game of any named variant, whatever else
        its headers say.
        """
        custom = self.board.lower().startswith("custom") or self.board.lower() in ("even", "odd")
        return CUSTOM_POSITIONS if self.fen or custom else (self.board or "Standard")


def load(path: Path | None = None) -> list[Record]:
    path = path or paths.CORPUS_MANIFEST
    data = load_toml(path)
    if data.get("schema") != SCHEMA:
        raise HarnessError(f"{path.name}: expected schema = {SCHEMA}")
    records = []
    for index, table in enumerate(data.get("record", [])):
        where = f"{path.name} record #{index + 1}"
        try:
            records.append(Record(
                id=table["id"], source=table["source"], path=table["path"], blob=table.get("blob", ""),
                sha256=table["sha256"], size=int(table["size"]), board=table.get("board", ""),
                licence=table["licence"], tier=table["tier"], fen=bool(table.get("fen", False)),
                also_at=tuple(table.get("also_at", ())), notes=table.get("notes", ""),
                converted=table.get("converted", ""),
            ))
        except KeyError as missing:
            raise HarnessError(f"{where}: missing key {missing}") from None
    return records


@dataclass(frozen=True)
class Excluded:
    """A record in a pinned source that is deliberately left out of the corpus."""

    source: str
    path: str
    reason: str


def load_excluded(path: Path | None = None) -> list[Excluded]:
    path = path or paths.CORPUS_MANIFEST
    entries = []
    for table in load_toml(path).get("excluded", []):
        try:
            entries.append(Excluded(table["source"], table["path"], table["reason"]))
        except KeyError as missing:
            raise HarnessError(f"{path.name}: an [[excluded]] entry is missing {missing}") from None
    return entries


def _stored_bytes(record: Record, lock: Lock) -> bytes:
    if record.source == LOCAL:
        return (paths.REPO_ROOT / record.path).read_bytes()
    return gitutil.read_blob(lock.pin(record.source).checkout, record.blob)


@dataclass(frozen=True)
class Feed:
    """What every source is given for one record."""

    data: bytes  # the bytes every source reads (for a refused record: the record, which no source is given)
    converted: int = 0  # export moves rewritten in the specification's form (export_form.py)
    unsupported: str = ""  # why no source is given the record; empty when every source is


def _conversion(record: Record, fed: bytes) -> tuple[export_form.Conversion | None, str]:
    """(the conversion of a record of CONVERTED_SOURCE's, or None; what its manifest entry's `converted` must say)."""
    if record.source != CONVERTED_SOURCE:
        return None, ""
    conversion = export_form.convert(fed)
    if not conversion.export_form:
        return conversion, ""
    return conversion, UNSUPPORTED if conversion.refused else sha256_bytes(conversion.text)


def feed(record: Record, lock: Lock) -> Feed:
    """The record as it is fed to every source: stored bytes with CRLF folded to LF - and,
    for a record of CONVERTED_SOURCE's in the official client's export form, converted
    to the notation specification's 5dpgn (export_form.py, CHECKPOINT 3).

    Some records are stored with CRLF line endings. Folding them keeps the input
    identical across platforms (a C++ oracle on Windows folds CRLF itself when it
    reads stdin in text mode; on Linux it would not).

    What the sources read is pinned like the stored bytes: a converted text, or a
    refusal, that is not what the manifest's `converted` says stops the run here
    (verify names the value).
    """
    data = _stored_bytes(record, lock)
    if sha256_bytes(data) != record.sha256:
        raise HarnessError(f"corpus record {record.id}: content does not match its pinned SHA-256")
    fed = data.replace(b"\r\n", b"\n")
    conversion, pinned = _conversion(record, fed)
    if pinned != record.converted:
        raise HarnessError(
            f"corpus record {record.id}: the manifest says converted = {record.converted!r}, and converting it from "
            f"the export form gives {pinned!r} (run: python {paths.HARNESS_REL}/harness.py verify)")
    if conversion is None or not conversion.export_form:
        return Feed(data=fed)
    if conversion.refused:
        return Feed(data=fed, unsupported=f"the record is in the official client's export form, and {conversion.refused}")
    return Feed(data=conversion.text, converted=conversion.moves)


def read(record: Record, lock: Lock) -> bytes:
    """The bytes every source reads for this record (feed()); an error for a record no source is given."""
    given = feed(record, lock)
    if given.unsupported:
        raise HarnessError(f"corpus record {record.id} is given to no source: {given.unsupported}")
    return given.data


def verify(records: list[Record], lock: Lock) -> list[Finding]:
    findings: list[Finding] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for record in records:
        tag = f"corpus record {record.id}"
        if not _ID.match(record.id):
            findings.append(Finding("record-id", f"{tag}: ids look like '<source>/<name>'"))
        if record.id in seen_ids:
            findings.append(Finding("record-duplicate-id", f"{tag}: id used twice"))
        seen_ids.add(record.id)
        if record.tier not in TIERS:
            findings.append(Finding("record-tier", f"{tag}: tier must be one of {TIERS}"))
        if not record.licence.strip():
            findings.append(Finding("record-licence", f"{tag}: every record states the licence it is used under"))

        if record.source == LOCAL:
            path = paths.REPO_ROOT / record.path
            if not path.is_file():
                findings.append(Finding("record-missing", f"{tag}: no file at {record.path}"))
                continue
            data = path.read_bytes()
        else:
            if record.source not in lock.pins:
                findings.append(Finding("record-source", f"{tag}: source '{record.source}' is not pinned in the lock file"))
                continue
            pin = lock.pin(record.source)
            if not gitutil.is_checkout(pin.checkout) or not gitutil.has_commit(pin.checkout, pin.commit):
                findings.append(Finding("record-unreadable", f"{tag}: source '{pin.id}' is not checked out at its locked commit"))
                continue
            actual = gitutil.blob_id(pin.checkout, pin.commit, record.path)
            if actual is None:
                findings.append(Finding("record-missing", f"{tag}: {record.path} does not exist at {pin.id}@{pin.commit[:12]}"))
                continue
            if actual != record.blob:
                findings.append(Finding(
                    "record-blob", f"{tag}: {pin.id}:{record.path} is blob {actual[:12]}, manifest pins {record.blob[:12]}"))
                continue
            data = gitutil.read_blob(pin.checkout, record.blob)
            for other in record.also_at:
                other_pin, _, other_path = other.partition(":")
                if other_pin not in lock.pins:
                    findings.append(Finding("record-also-at", f"{tag}: also_at '{other}' names an unpinned source"))
                    continue
                target = lock.pin(other_pin)
                if gitutil.blob_id(target.checkout, target.commit, other_path) != record.blob:
                    findings.append(Finding("record-also-at", f"{tag}: also_at '{other}' is not the same blob"))

        if sha256_bytes(data) != record.sha256:
            findings.append(Finding("record-sha256", f"{tag}: content hash differs from the manifest"))
        if len(data) != record.size:
            findings.append(Finding("record-size", f"{tag}: {len(data)} bytes, manifest says {record.size}"))
        if b"\r" in data.replace(b"\r\n", b"\n"):
            findings.append(Finding("record-line-endings", f"{tag}: contains a bare carriage return"))
        if record.sha256 in seen_hashes:
            findings.append(Finding(
                "record-duplicate",
                f"{tag}: same content as {seen_hashes[record.sha256]} (list the second location under also_at)"))
        seen_hashes[record.sha256] = record.id
        # Judged on the bytes the sources are fed (CRLF folded), not the stored ones.
        fed = data.replace(b"\r\n", b"\n")
        header = declared_board(fed) or ""
        if header != record.board:
            findings.append(Finding("record-board", f"{tag}: the record declares the variant '{header}', manifest says '{record.board}'"))
        if has_fen_boards(fed) != record.fen:
            findings.append(Finding(
                "record-fen",
                f"{tag}: the record {'carries' if not record.fen else 'carries no'} 5DFEN boards, "
                f"manifest says fen = {str(record.fen).lower()}"))
        conversion, pinned = _conversion(record, fed)
        if record.converted != pinned:
            if conversion is None:
                why = f"only {CONVERTED_SOURCE}'s records are converted from the export form: drop converted"
            elif not pinned:
                why = "the record is not in the official client's export form: drop converted"
            else:
                why = ("the record is in the official client's export form; converting it gives "
                       f'converted = "{pinned}"' + (f" ({conversion.refused})" if conversion.refused else ""))
            findings.append(Finding("record-converted", f"{tag}: manifest says converted = {record.converted!r}; {why}"))
    return findings


def verify_local(records: list[Record]) -> list[Finding]:
    """Every record kept in this repository is listed, and lives where such records live."""
    findings = []
    home = paths.CORPUS_DIR / "records"
    home_rel = home.relative_to(paths.REPO_ROOT).as_posix()
    manifest = paths.CORPUS_MANIFEST.relative_to(paths.REPO_ROOT).as_posix()
    listed = {record.path for record in records if record.source == LOCAL}
    for path in sorted(listed):
        if not path.startswith(home_rel + "/"):
            # Only that directory is exempt from line-ending conversion (.gitattributes).
            findings.append(Finding("record-placement", f"local corpus record {path}: records of our own live under {home_rel}/"))
    if home.is_dir():
        for file in sorted(home.rglob("*")):
            relative = file.relative_to(paths.REPO_ROOT).as_posix()
            if file.is_file() and relative not in listed:
                findings.append(Finding("record-unlisted", f"{relative}: not listed in {manifest} - list it, or remove it"))
    return findings


def verify_excluded(excluded: list[Excluded], lock: Lock) -> list[Finding]:
    findings = []
    for entry in excluded:
        tag = f"corpus exclusion {entry.source}:{entry.path}"
        if not entry.reason.strip():
            findings.append(Finding("excluded-reason", f"{tag}: say why it is left out"))
        if entry.source not in lock.pins:
            findings.append(Finding("excluded-source", f"{tag}: source is not pinned in the lock file"))
            continue
        pin = lock.pin(entry.source)
        if gitutil.is_checkout(pin.checkout) and gitutil.blob_id(pin.checkout, pin.commit, entry.path) is None:
            findings.append(Finding("excluded-missing", f"{tag}: no such file at the pinned commit - drop the exclusion"))
    return findings


def unlisted(records: list[Record], lock: Lock, excluded: list[Excluded] | None = None) -> list[tuple[str, str, str]]:
    """(pin id, path, blob) of .5dpgn files in pinned sources that the manifest neither lists nor excludes."""
    listed = {record.blob for record in records}
    skipped = {(entry.source, entry.path) for entry in excluded or ()}
    candidates = []
    for pin in lock.pins.values():
        if not gitutil.is_checkout(pin.checkout) or not gitutil.has_commit(pin.checkout, pin.commit):
            continue
        for mode, oid, path in gitutil.list_tree(pin.checkout, pin.commit):
            if not mode.startswith("100") or not path.lower().endswith(_RECORD_SUFFIX):
                continue
            if oid not in listed and (pin.id, path) not in skipped:
                candidates.append((pin.id, path, oid))
    return candidates


def describe_candidate(pin_id: str, path: str, blob: str, lock: Lock) -> dict:
    """A manifest entry for a record in a pinned source, ready for a human to review."""
    data = gitutil.read_blob(lock.pin(pin_id).checkout, blob)
    fed = data.replace(b"\r\n", b"\n")
    stub = {
        "id": f"{pin_id}/{Path(path).stem}",
        "source": pin_id,
        "path": path,
        "blob": blob,
        "sha256": sha256_bytes(data),
        "size": len(data),
        "board": declared_board(fed) or "",
    }
    if has_fen_boards(fed):
        stub["fen"] = True
    if pin_id == CONVERTED_SOURCE:
        conversion = export_form.convert(fed)
        if conversion.export_form:
            stub["converted"] = UNSUPPORTED if conversion.refused else sha256_bytes(conversion.text)
    return stub


def select(records: list[Record], tiers: tuple[str, ...] | list[str], only: list[str] | None = None) -> list[Record]:
    chosen = [record for record in records if record.tier in tiers]
    if only:
        wanted = set(only)
        unknown = wanted - {record.id for record in records}
        if unknown:
            raise HarnessError(f"unknown corpus record(s): {', '.join(sorted(unknown))}")
        chosen = [record for record in records if record.id in wanted]
    return chosen
