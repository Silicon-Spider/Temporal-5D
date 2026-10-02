# t0database - real games, for measurement

761 games of *5D Chess With Multiverse Time Travel* played by people (2021-2025:
the World Championships of 2022 and 2024, leagues, tournaments), from
[MageOfficial/t0database](https://github.com/MageOfficial/t0database), MIT,
"Copyright (c) 2022 MageOChicken" (`LICENSE`, beside the data).

| | |
|---|---|
| Pinned commit | `d49da4054d02fec577fd3c53368993fe453c586e` |
| `Data.csv` | git blob `35eb1010d066ebe111428c45ca29bc9fd0ae47d1`, sha256 `37fb7c1ce46efb27d42a8b9d0b230df13faff1b36bef24af8c15ba5c8a6c7a4e`, 608,066 bytes |
| `LICENSE` | git blob `8a8b359a85399a2e943b62a1acc007be5c3241c7` |
| Downloaded | 2026-10-03, with the user's approval (CHECKPOINT 3) |

Both files are byte-for-byte copies (`.gitattributes` keeps git from changing
their line endings). Do not edit them: a new version of the database is a new
pin, recorded here, in `PROVENANCE.md` and in `CHANGELOG.md`.

**What it is for:** the engine's measurements - the window's |E| phase curve (M7)
and anything else measured over real play. **Not** an oracle, not a source for a
rule (the rulebook is `docs/rules/README.md`), and not part of the gate's corpus.
The games reach about 186 boards; nothing here says anything about 200-600
boards.

**Reading it:** `py data/t0database/to_5dpgn.py OUT_DIR` writes each game as a
5dpgn record of its own (`t0db-NNNN.5dpgn`, by the row's `GameNum`), refusing a
`Data.csv` that is not the pinned file. The conversion is the source's own and
nothing more: a game written without headers - all 761 are - gets the two
headers the source's own server puts in front of such a game before reading it,
`[Board "Standard - Turn Zero"]` and `[Mode "5D"]`; the move text is not touched.
A game that does not replay is refused by whatever reads it, never edited.
Columns: `GameNum`, `WhitePlayer`, `BlackPlayer`, `Game` (the move text), `White`
and `Black` (the score), `Date`, `Event`, `Link`.
