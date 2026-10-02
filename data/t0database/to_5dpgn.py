"""Write each game of t0database's Data.csv as a 5dpgn record of its own.

Usage: py data/t0database/to_5dpgn.py OUT_DIR

The conversion is the source's own and nothing more. A game's move text is
written unchanged (line endings as LF). A game with no headers gets the two that
the source's own server puts in front of such a game before reading it -
``[Board "Standard - Turn Zero"]`` and ``[Mode "5D"]`` (server.js, at the pinned
commit) - and a game with headers keeps its own. No move is read, checked or
rewritten here: a record that does not replay is refused by whatever reads it,
never edited. Rows with no game text are skipped and counted.

Standard library only; refuses a Data.csv whose bytes are not the pinned ones.
"""

from __future__ import annotations

import csv
import hashlib
import io
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "Data.csv"
SHA256 = "37fb7c1ce46efb27d42a8b9d0b230df13faff1b36bef24af8c15ba5c8a6c7a4e"
DEFAULT_HEADERS = '[Board "Standard - Turn Zero"]\n[Mode "5D"]\n'


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    raw = DATA.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        print(f"Data.csv is not the pinned file (sha256 {digest}, pinned {SHA256})", file=sys.stderr)
        return 1
    out = pathlib.Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    written = empty = own_headers = 0
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")):
        text = row["Game"].replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            empty += 1
            continue
        if "[" in text:
            own_headers += 1
        else:
            text = DEFAULT_HEADERS + text
        if not text.endswith("\n"):
            text += "\n"
        name = f"t0db-{int(row['GameNum']):04d}.5dpgn"
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        written += 1
    print(f"{written} records written to {out} ({own_headers} with their own headers); "
          f"{empty} rows with no game text skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
