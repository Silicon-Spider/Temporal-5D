# Oracle harness

The differential-testing gate for the rules engine (design spec §11.0; decisions
H23–H34 and H36–H38), and everything it needs: three independent implementations of the rules
and an independent reader of the notation, pinned; a pinned corpus of game records;
the machinery that runs the one over the other; and the check that keeps all of it
out of the engine.

**The engine is configured.** `gate.toml`'s `[engine] command` is the engine's
gate adapter, `build/engine/t5d-gate` (built by `engine/build.py`), compared in
full mode by every tier: the commit tier as a smoke test that gates nothing - and
fails an engine that declines a record it read on the last green run, or reads
none - nightly and release gating the rulesets two or more oracles vouch for ("The
engine contract", below). That is not yet a passed gate: the proof channel stays
disabled until a complete run of the release tier says it may be enabled and the
channel is signed off (H23, H41).

## The rule

> Everything under `oracles/` may use the engine.
> Nothing outside `oracles/` may use anything under `oracles/`.

An oracle is worth what its independence is worth, so this directory is a separate
project: its own `pyproject.toml`, no third-party Python dependencies, never a
member of the engine's build. `harness.py boundary` checks every file outside it,
by what the file is (`gate.toml` `[boundary]` sorts them):

| Kind | Files | May not |
|---|---|---|
| source | code, and anything not sorted below | name a path into the harness, import the harness, fetch or import an oracle, contain code copied from an oracle |
| manifest | build files and scripts | do anything source may not, list the harness directory, or name an oracle's package at all |
| plumbing | CI and editor configuration, which has to run the harness | fetch or name an oracle's repository or package, contain copied code |
| documentation | Markdown, licences, citations | contain copied code - it may cite anything |

It also fails a submodule outside `third_party/` that checks out an oracle, a git
repository embedded in the tree, a link into the harness, third-party files
committed under `third_party/`, and a text file it cannot read (binary files are
passed over).

Comments are taken out before a file is searched, by the comment syntax of its
type, so citing the harness or an oracle in a comment is fine. In code, two
references into the harness are allowed: its manual, `oracles/README.md`, and - in
a recipe file (Makefile, justfile, `package.json`, `tox.ini`) - the gate's entry
point, `oracles/harness.py`, run with any command except `observe` and `golden`,
which print what the oracles answer. A command that changes into the harness
(`cd oracles && ...`, `make -C oracles`) names it as surely as a path does. A build
file's exclusion list (`exclude`, `ignore`, ...) may name the harness directory:
that is how an honest build keeps it out. Anything else needs a waiver in
`gate.toml` that quotes the offending text.

The check sees text. It finds a copy when six consecutive lines of logic match an
oracle's source once white space and quote style are set aside, or a single line
of 160 characters or more does (a data table, say) - not a copy with its
identifiers renamed or its lines rearranged, and not an oracle's logic retyped in
another language. Those are the same offence, and `CONTRIBUTING.md` and
`AGENTS.md` say so. `PROVENANCE.md` records what came from where.

## What is pinned

| Id | Project | Runs | Answers |
|---|---|---|---|
| `ftxi` | ftxi/5dchess_engine (C++) | every commit | check, action counts, mate / stalemate, legality |
| `cwmtt` | penteract/cwmtt (Haskell) | nightly, release | action counts, "no action exists", legality |
| `chessin5d` | 5d-chess-js, the rules library of Chess in 5D (JavaScript) | nightly, release | check, mate / stalemate, legality |
| `notation` | adri326/5dchess-notation: the notation spec, sample records, and the parser written beside it (JavaScript) | every commit | parse-only: the position it read at every ply, the side it reads as moving next, how many actions it read - no rules verdict |

Exact commits, licences and caveats are in `oracles.lock.toml` and `PROVENANCE.md`.
ftxi and cwmtt each have two action searches, a hypercuboid one and an exhaustive
one; the harness runs them as separate sources (`ftxi`, `ftxi:naive`, `cwmtt`,
`cwmtt:naive`), because two algorithms over one implementation are a check the
implementation cannot fake. ftxi's hypercuboid search descends from cwmtt's, so a
flaw in the method would not show between those two: that is what the naive
searches and Chess in 5D are for.

No oracle answers everything, and none reads every board variant. What that leaves
checked is tabulated, per variant, at the top of `DISAGREEMENTS.md`.

## Layout

```
oracles/
  harness.py              entry point: python oracles/harness.py <command>
  oracles.lock.toml       the pins: repositories, commits, licences, toolchains
  gate.toml               tiers, budgets, the stability policy, the rulebook pin, the engine command,
                          boundary policy
  known-divergences.toml  where an oracle is wrong, with the rule that says so
  DISAGREEMENTS.md        what the oracles cover, what each tier requires of them, and where they
                          contradict each other (generated)
  corpus/manifest.toml    the pinned game records, by source, blob and hash
  corpus/records/         the few records written for this project
  golden/                 what each oracle says about each record (generated, reviewed): the
                          answers (<source>.json), when they arrived (<source>.cost.json), the
                          calibration probe's time on the recording machine (calibration.json),
                          and the SHA-256 of each of those files (MANIFEST)
  drivers/                the programs of ours that load oracle code
  oracle_harness/         the harness (Python, standard library only)
  tests/                  its own tests
  third_party/            the pinned submodules - read-only
  .cache/                 toolchains, builds, reports - ignored by git
```

## Getting started

```
python oracles/harness.py sync               # check out the submodules at their pins
python oracles/harness.py doctor             # what is installed, what is built
python oracles/harness.py build --bootstrap  # build the oracles
python oracles/harness.py verify             # static checks
python oracles/harness.py selftest           # the harness's own tests
python engine/build.py                       # build the engine, whose adapter the gate runs
python oracles/harness.py gate --tier commit # the per-commit gate
```

The gate compares the engine's adapter as it finds it, `build/engine/t5d-gate`
(`gate.toml`, `[engine] command`): build it before a gate run, and again after any
change to the engine (`engine/README.md` says what that needs) - a gate run
compares whatever build is there, and its report names the file and its SHA-256
(`engine_run`). Unbuilt, every record fails with `engine-failed`.

The harness needs Python 3.12+ and git. Building the oracles needs:

| Oracle | Needs |
|---|---|
| `ftxi` | a C++23 compiler (g++ 13 or newer), CMake 3.20+, and Ninja on Windows |
| `cwmtt` | GHC 9.10.3 exactly - nothing else; no z3, no stack, no package download. It builds the stock CLI and the harness's state driver (below) |
| `chessin5d` | Node 18+ and npm (three small MIT packages, installed from its own lockfile with scripts disabled) |
| `notation` | Node 18+ - nothing is installed: the parser needs no package |
| `kairos` | Rust 1.99.0 exactly (cargo and rustc) - no crate is fetched: Kairos has no dependencies, and the build runs `--locked --offline`. It builds the harness's driver (`drivers/kairos/`) against the pinned crate, outside its tree, so the crate's `-C target-cpu=native` does not apply; on Windows the GNU target links with the MinGW linker the toolchain ships |

`--bootstrap` fetches what is missing into `oracles/.cache/` - CMake and Ninja as
pip wheels in a private virtualenv, GHC as a checksum-verified archive (Windows
only; elsewhere install GHC 9.10.3 yourself, e.g. with ghcup), and Rust as its
checksum-verified component archives (Windows and Linux, x86-64), merged into one
prefix as rustup would install them, with cargo's home inside the cache. The
harness's own Rust in the cache is used first, then a cargo and rustc of exactly
the pinned release on PATH. It changes nothing outside that directory. Never run `git submodule update --init --recursive`: the
oracles' own submodules are not needed and one of them is declared with an SSH URL.

Every build starts from `git archive` of the locked commit, so what runs is the
pinned tree whatever state the checkout is in; nothing is ever built inside, or
written to, `third_party/`.

It runs on Windows (where it was written) and on Linux (CI, ubuntu-24.04): both
workflows first went green on 2026-10-01, and the golden files recorded on Windows
reproduce exactly on Linux.

## What is compared

At every position of a record's main line - position *k* is the state after *k*
submitted actions - first what each source set up there, then three things about
it, plus legality itself:

| Observable | Meaning |
|---|---|
| state | What the source set up: every board of the multiverse. The harness hashes it (below); two sources that set up different boards are compared no further, and two whose unmoved markers alone differ are compared with that difference reported beside. |
| `in_check` | Check status (decision H36, the rulebook's R1): passing on all **active** boards **in the present**, could the opponent capture a royal from any of their playable boards? Optional boards ahead of the present and boards on inactive timelines are not passed; a capture counts even if it would branch into an inactive timeline. (Submission legality - whether an action may be submitted - is a separate test, on the actual turn: it shows as legality, below.) |
| `legal_actions` | Number of legal actions (whole turns), counted up to a cap of 1000. A count equal to the cap means "at least that many". An action is counted as the set of its moves, with its timeline-creating (branching) moves in the order they create their timelines: the order of its other moves does not count, and two orders of two branching moves are two actions - even where they reach the same state (rulebook J6: order matters within an action). That is what ftxi and cwmtt count, with both searches of each (below the table). It is the protocol's definition, chosen so that counts are comparable - not a rules ruling - and it is not the number of distinct states the mover can reach, which a search may count instead, separately. |
| `terminal` | `none` while a legal action exists; otherwise `checkmate` or `stalemate` (`no_action` from a source that cannot tell which). A verdict is a proof - an exhaustive search that terminated (H37): a search that ran out of its budget answers nothing (below). |
| legality | Whether each recorded action is accepted. One source refusing an action another went on to play is a disagreement; so is one source saying the record ends where another found a further action, whether it played that action or refused it. |

What the oracles count as an action was checked on their own listings
(`observe --actions K`, 2026-10-02). At eight positions that both read - among
them `cwmtt/ctp1` position 13 (195 actions), `cwmtt/NP` 60 (567) and
`notation/1.2` 10 (916) - ftxi and cwmtt listed exactly the same actions, ftxi's
naive search as well at `cwmtt/ctp1` 13, and the four searches' recorded counts
agree there. In every listing a set of moves with *b* branching moves appears
*b*! times, and every other set once. At `cwmtt/JustKings` position 3, which only
cwmtt reads, its 27 actions are 25 sets of moves, two of them in both orders of
their two branching moves; one of those pairs - two king jumps onto `(0T1)c2` -
reaches the same state either way, and is still two actions. Chess in 5D gives
the gate no count; its own enumerator, shown only by `observe --actions`, lists
each set of moves once (560 at `cwmtt/NP` 60, 904 at `notation/1.2` 10).

Positions are aligned by counting submitted actions. On its own that cannot tell a
rules disagreement from two parsers that read one move as two different legal
actions: after such a move every position is a different position under the same
number. So every source says, at every position and before it searches anything,
what it has set up - its *state event*, whose form is in "The engine contract"
below (decision H30) - and the harness, never the source, turns that into:

- a **fingerprint**: the notation specification's full-state hash (its README,
  "Hashing" and "5DFEN Strictness"), with three readings the specification leaves
  open fixed here. Every board is written `[placement:L:T:c]`, the boards are
  sorted by timeline (`-0` before `+0`), then by turn, then white's board before
  black's (the specification sorts by timeline and turn only), concatenated, and
  hashed with MD5; a positive timeline is always written with its `+` (the
  specification lets it be left out); and the unmoved marker `*` follows every
  pawn, rook, king and brawn that has not moved and no other piece - the
  specification's 5DFEN section asks for it on a piece "sensitive to having been
  moved" that "hasn't moved yet", and "5DFEN Strictness" names those four kinds
  and says to mark them "if needed", which is read as "when they have not moved",
  not "when a castling or a double step is still open to them" (so a rook keeps
  its marker after its king has moved). The specification's two published
  examples, the Standard start and the two-timeline Standard start, are
  conformance tests of `oracle_harness/fingerprint.py`; an oracle that computes
  the hash itself (5d-chess-js: `chess.hash`) hands it over with its state, and
  the harness's fingerprint must equal it at every position. A source that does
  not track which pieces have not moved has no fingerprint;
- a **placement** hash: the same, with every unmoved marker (`*`) removed - every
  board, its timeline, turn and colour as well as its pieces, without the markers;
- the number of boards and the number of pieces.

The golden files keep those four values, not the boards. Two reports are then
compared position by position, and what was set up comes first:

- **placement** - at the first position where the two placements differ, the two
  sources hold different boards - different pieces, or the same pieces on boards at
  other coordinates: they read the record's moves (or its set-up) as different
  positions. That is a *parse divergence*, reported once, and nothing either says
  from that position on is compared - no observable there or after it, no legality
  or length of a later action - because it would compare verdicts about two
  different positions. It is settled against the notation specification;
- **unmoved** - where the boards and pieces agree and the unmoved markers do not, a
  marker-only parse divergence: one for each run of such positions, reported at the
  first position of the run (a run ends where the markers agree again, and markers
  that part again after that are a new divergence). It suppresses no comparison, so
  triage sees it beside any rules disagreement it causes. It is settled against the
  notation specification's "5DFEN Strictness" or the rulebook (what having moved
  means).

Parse divergences are compared between any two sources, two searches of one
implementation included: those replay a record with the same code, so a parse
divergence between them is a bug in the harness or a driver. A source that cannot
say what it set up prints no state event, and never diverges - but every position
it reports is then aligned with the others by counting alone. One that can says it
at every position, from position 0, with the same `markers` throughout, or the run
is malformed. `DISAGREEMENTS.md` lists the parse divergences ahead of the rules
disagreements, and counts, for every pair of sources and observable, the positions
at which the two were known to hold the same state (markers included), those at
which they held the same boards and pieces with markers that differ or that one
does not give, and those whose alignment rests on counting alone.

A parse-only source (`notation`) applies any move it can place, legal or not. So
its having played an action contradicts another source's refusal of it only where
that refusal was a failure to read the move text (a soft one), never where the
action was held illegal; and it never stands behind the engine's replay of a
record, as a rules oracle that followed it would. What it set up is compared like
any other source's.

Each oracle answers with its own reading of these definitions - which boards its
check test passes, for one. Where the readings differ, the reports disagree, and
the rulebook (`docs/rules/README.md`, pinned - "Known divergences", below), not
the harness, says who is right (see "What the disagreements are" below).

Three principles keep the comparison honest:

- **An unknown never disagrees with anything.** A source that cannot answer a
  question, or ran out of time, says nothing. Nothing is inferred on its behalf.
  A mate verdict in particular is a proof, from an exhaustive search that
  terminated, and an exhausted budget is unknown, never mate (H37): a search cut at
  the run's deadline answers nothing at that position, and Chess in 5D's mate search,
  whose library answers "checkmate" when it runs out of time, leaves its verdict
  null and names the limit it ran into. A position that names a limit and gives a
  verdict anyway makes the run malformed.
- **A source is never asked about a game it would misread.** cwmtt and 5d-chess-js
  both replay a board variant they do not know on the Standard layout without a
  word; both do the same when their own reader skips the record's Board header,
  and neither reads a `Variant` or `Timeline` header; both ignore 5DFEN start
  positions in cases where ftxi honours them. 5d-chess-js also replays a
  parenthesised variation as part of the main line, does not split actions at a
  turn number written `12w.`, and can name the boards of a custom set-up at
  another turn than the record's (it guesses from the boards it holds whether the
  game has a turn zero); ftxi replays a variation that ends a record as its main
  line. The notation repository's parser sets up one empty board for a
  variant it does not know, and a named layout on boards of the size its `Size`
  header gives (8x8 without one), never the layout's own. Each adapter works out
  what its oracle would set up - the cwmtt one by following cwmtt's tag grammar,
  the 5d-chess-js one by asking the library after its own set-up, the notation one
  from the parser's own table of named boards, and again in its driver after the
  parser's set-up - and refuses the record (`unsupported`) when that is not the
  game the record declares, or when the record uses a notation its oracle
  misreads.
- **"Accepted" means the oracle holds the action legal.** ftxi's replay tools apply
  and submit a recorded action without testing whether it leaves a royal
  capturable; the driver applies ftxi's own legality test first. cwmtt's parser
  stops, without failing, at the first thing it does not read and replays what it
  had; when that leaves the last action half read, the adapter reports the record
  as `unsupported` from there, and does not pass the fragment's fate off as a
  verdict.

The harness never parses the moves of a record (it rewrites one textual pattern of
the export form, without reading a move: "How a record reaches the sources",
below). Each oracle reads a record with its own parser, the engine will read it
with its own (H28), positions are aligned by counting submitted actions, and the
alignment is checked by what each source says it set up: boards, which the
harness reads, never moves. What the harness reads
out of a record is what game it declares itself to be - its `Board` header (or its
`Variant` header when it has none), whether it carries 5DFEN boards, whether it has
a `Timeline` or `Promotions` header - and whether its move text holds a variation
or a turn number written `12w.`. One driver also drops a PGN result token (`1-0`,
`1/2-1/2`) from the end of the move text, because 5d-chess-js reads it as moves and
fails on it.

### How a record reaches the sources

Every source - each oracle search and the engine - is given the same bytes for a
record, on standard input: the stored record (read from its pinned source's object
store, `corpus/manifest.toml`), with CRLF folded to LF (`corpus.feed`; every run
goes through `runner.observe`). There is one exception, ruled at CHECKPOINT 3:
"the cwmtt driver converts the export form to the canonical 5dpgn form from the
pinned notation spec; if it expresses something the spec can't, mark the record
unsupported. The engine's parser is not extended." A record of cwmtt's written in
the official client's export form, whose moves name their board twice
(`(0T1)Pe2(0T1)e3` - the notation specification's "Export moves"), is converted
by `oracle_harness/export_form.py` before any source reads it, and the converted
text is what every source reads - the engine included. Converting it for cwmtt
alone would change nothing (cwmtt reads either form). Today that is one record,
`cwmtt/many2`.

The conversion is notation, never the position. A move whose two boards are the
same board loses its second board, `(0T1)Pe2(0T1)e3` becomes `(0T1)Pe2e3` (the
specification's physical move with its board and the piece's square in full; the
export form's castling, `(0T6)Ke1(0T6)g1`, becomes `(0T6)Ke1g1`, the form
"Castling" gives), and nothing is added - the export form marks no capture,
check or promotion, and none is guessed. Every other byte is copied: headers,
comments, turn numbers, separators, and the jumps, which the export already
writes as the specification does (`(0T2)Ng1>>(0T1)g3`). A record with a move the
specification cannot express - a move to another board without `>` or `>>`
(which one it takes is a question about the position), or other text that names
a board straight after a square - is not partly converted: no source is given it,
and every one reports it `unsupported`, reason `export_form`, with the move it
could not express. That keeps its ruleset ungated, as any record the engine did
not replay does.

What the sources read is pinned like the stored bytes: the record's manifest entry
says `converted`, the SHA-256 of the converted text (or `"unsupported"`); `verify`
checks it and prints the value to write, and a run refuses a record whose
conversion no longer gives it. The golden files and `known-divergences.toml` name
a converted record by that hash, so answers recorded about one text are never
taken for answers about another (`golden-stale`), and the gate's report
(`converted`), its log and `DISAGREEMENTS.md` ("Records not read as stored") say
which records were read converted.

A rejection is **soft** when the source could not match the move text to any move
it can generate (`no_matching_move`, `ambiguous_move`, `notation_mismatch`, or a
message the adapter does not know, `unrecognised`): that may be a rules
disagreement or a notation the source does not read, and the harness cannot tell
which. The engine's `unreadable` - its own parser could not read the move text,
or the record's set-up ("The engine contract") - is soft too. `illegal_move` and
`cannot_submit` are hard.

How a record is replayed - which actions are accepted, how many are read - belongs
to the implementation, not to each of its searches, so it is compared once per
implementation. The observables are compared for every search.

cwmtt's counts, and how far it replays a record, come from its stock CLI,
unmodified, which says nothing of the side to move and draws the boards only as a
picture of the last position (`cwmtt print`). The boards at every position and the
side to move come from a second program: the harness's state driver
(`drivers/cwmtt/Driver.hs`), compiled against the same pinned modules with the same
flags, which reads and replays the record beside the CLI and prints the state
events and the side to move. The adapter pairs a count with a state only where the
two set up the layout the record declares, left the same text unread, ended their
replays with the same failure and reported the same positions. Where they did not,
the run is an `error` (`driver_mismatch`, or `driver_error` when the driver itself
failed) and carries no state: a count is never paired with another position's
boards. The driver searches nothing, so it seldom runs out of time; when it does,
the CLI's answers and outcome stand, and the states it printed are paired with
them as far as it got, once its start event shows the layout the record declares
and the CLI's unread text. Where cwmtt's parser stopped inside an action, no
position from the one after that fragment on is reported, by either program,
however the CLI's run ended.

## Tiers

| Tier | When | Sources | Notes |
|---|---|---|---|
| `commit` | every push and pull request | `ftxi`, `notation` | three positions per record in the driver check (below); a **smoke test**: it gates no ruleset - with one rules oracle it could gate none, a ruleset needing two - and it fails an engine on a finding against it, on a record it read on the last green run and now declines, or on reading none (H47, "The engine contract") |
| `nightly` | every night | all six | the naive searches get shorter budgets; they time out on wide positions by design; it **fails an engine with no finding against it that comes through no ruleset** |
| `release` | before any generator release or proof-channel enablement | all six | the longest budgets - the golden files are recorded under them - and it **fails unless an engine was compared and came through at least one named variant** |

What a tier vouches for follows from what it runs, never from its name: only a
tier that runs every oracle search over the whole corpus gates a ruleset ("The
engine contract"). So only such a tier may require an engine
(`require_engine = true`): `gate.toml` is refused if any other tier does, since
it could gate nothing, and so no engine could come through a named variant in it.

Workflows: `.github/workflows/oracle-commit.yml` and `oracle-nightly.yml` (the
latter also runs the release tier, by hand or on a `generator-v*` tag). Each runs
`verify`, builds the oracles its tier needs, runs the harness's own tests
(`selftest`), and then the gate.

Each tier run does, in order: it reads everything it compares against once, and
verifies it (the golden, cost and calibration files against `golden/MANIFEST`,
the engine's baseline against its pin in `gate.toml`, and
`known-divergences.toml`), and hashes every harness file; the static checks;
the calibration probe; the oracle runs; a self-consistency check of every report;
a comparison of each fresh run with its tier's required set of recorded answers -
and of every other recorded answer it gave, or failed to give where no budget
explains it; a check of the ftxi driver against ftxi's stock CLI; when an engine
is configured, the engine differential and the comparison with the engine's
baseline; and last, a check that no harness file changed while it ran. The last
lines of the output say whether the engine was compared, whether the tier gates
anything (a smoke test says that it gates nothing), and whether the proof channel
may be enabled - and of a run restricted with `--source` or `--record`, only that
it is no tier result ("Other commands").

A tier has no coverage floor: it requires **100% of its required set** - every
recorded value its budgets can be shown to reach - and loses none of them without
failing ("Stable answers", below). A run of an oracle search that has everything
its tier requires of it, and of which nothing further is needed, is stopped there
(it shows as `stopped`): the values past that point are outside the required set
anyway. No run is stopped where an engine is compared - every oracle answer may
confirm or contradict it - nor in a tier that can enable the proof channel, which
runs every search to its budget. What stopping gives up is the comparison of
values past the required set, and the driver check of a record whose run was
stopped.

What the driver check covers: for each ftxi search, whether an action exists at
each position, the verdict where none does, the action count, the side to move,
the number and size of the boards its state event lists, and the boards themselves
are compared with `5dtools perftest`, `count`, `stats` and `print` - every position
of the main search in the nightly and release tiers, three per record otherwise.
`print` draws every board under ftxi's own name for it (`L0T1w  aka.(0T1)`); read
back as a state, its placement hash must be the driver's, which holds the driver's
timeline labels, its walk over the boards and the pieces it lists to ftxi's own
output (the drawing has no unmoved markers). The check verdict at a position that
still has a legal action has no stock counterpart: it comes from the same library
call the stock tool makes at a terminal position.

## Golden observations and `DISAGREEMENTS.md`

Pinned oracles over a pinned corpus have pinned answers. `golden/*.json` records
them, one position per line. That lets CI notice an oracle that answers differently
on another machine or compiler, turns a pin bump into a reviewable diff of
verdicts, and puts on record - before the engine exists - every point where the
oracles contradict each other. `DISAGREEMENTS.md` is rendered from the golden
files.

A golden value is a statement *by that oracle*, not a statement of truth. At every
entry in `DISAGREEMENTS.md` at least one oracle is wrong or reads the rules
differently. The rulebook has settled the check disagreements (R1) and two oracle
defects (R7), and where the engine meets one of them `known-divergences.toml`
records the losing oracle's answer, with its citation; the other entries are
listed, not adjudicated.

```
python oracles/harness.py golden --write --repeat 2   # run every oracle over the corpus twice, and record it
python oracles/harness.py golden --render             # only re-render DISAGREEMENTS.md
```

`DISAGREEMENTS.md` is rendered from the golden files, their cost files and the
tiers' budgets in `gate.toml`: besides where the oracles contradict each other, it
holds what each tier requires of each source ("What each tier requires", below).
`verify` renders it again and fails if the committed one differs, so a changed
budget or cost shows in review as a changed line of that table.

Each file opens with a signature: the pin, the search and its settings, the cap
where the oracle counts actions, the budget it was recorded under (the run's
`timeout_s`; for Chess in 5D also its mate search's `mate_budget_ms` and heap,
`mate_heap_mb`), and hashes of the adapter and driver code that stood between the
oracle and the file. If any of them changes, the file is stale and `verify` says
which. Editing a driver, an adapter, or the `base.py`, `fingerprint.py` and
`headers.py` the adapters share - even a comment in one - therefore means
`golden --write`, and a review of every verdict that moved. So does changing the
release tier's budgets, which the files are recorded under (`[stability]
recording_tier` in `gate.toml`). (`fingerprint.py` computes the hashes the files
record of what each oracle set up.) The files hold verdicts and, per position, the
hashes and board and piece counts of what the oracle set up (schema 3) - never the
boards themselves, and never an oracle's free-text messages, because they quote
the record. A crash is pinned like any other outcome.

What an oracle did not answer is recorded too, and why (schema 3, decision H32):

| Entry | Means |
|---|---|
| `"horizon": {"position": K, "budget": "timeout_s=300"}` | the run was cut at its deadline while it searched position K: what it set up there is recorded, its answers are not, and nothing after it - everything from there on is unknown at that budget |
| `"unknown": {"K": {"terminal": "mate_budget_ms=60000"}}` | a value missing inside the answered positions because a limit stopped it: Chess in 5D's mate search ran out of its time budget, or of its heap (`mate_heap_mb=1024`). A null with no limit beside it is a question the source never answers |
| `"unstable": {"K": {"legal_actions": [20, 21]}}` | a value that differed between independent recordings; the position's value is null, so nothing is compared with it |

`--repeat N` (default 2) records every source N times, each run a process of its
own. A value the same in every recording is recorded; one present in some
recordings and absent from others because a run hit its budget is unknown at that
budget (a horizon, or an `unknown` entry); one that differs, or is absent for no
reason a budget gives, is unstable. Recordings that end differently - not because
one ran out of time - replay the record nondeterministically: that is a finding,
and nothing is written. Each file says how many recordings it holds
(`"recordings"`), and `verify` refuses a file that holds fewer than two: one
recording cannot show a value unstable, so every value in it would be required.
The re-recording after a pin bump, a driver change or a budget change is
`golden --write --repeat 2`.

When it has written the files, `golden --write` compares what it recorded with
the files it replaced, and lists every terminal verdict and outcome they held that
the new recording does not, and every run now cut at an earlier position. Each is
a loss of checking - most often a budget lowered, or a recording made on a busy
machine - and is reviewed as such: re-record on a quiet machine, or raise the
budget; never commit a loss to make a gate run pass.

Beside each golden file is its **cost file**, `<source>.cost.json`, one record per
line: when the answers about each position arrived, in seconds from the start of
the run (the line that completed the position - its position event; for cwmtt the
later of the CLI's count, timed as it streams in, and the state driver's state),
when the outcome arrived (`end`), and the record's cost, its slowest answer; for
Chess in 5D also each position's mate search (`mate`, from its driver's own
timing). Costs are rounded up to two significant figures, are the most over the
recordings, and are cumulative: a position is reported only after every position
before it (cwmtt's only once its CLI has counted every earlier one, whenever its
state driver printed the boards), so its cost is at least theirs. They change
with every recording, so they are kept out of the golden files, whose diff stays
a diff of verdicts. They are the evidence for the required sets below - and what
those sets come to, per tier and source, is the table in `DISAGREEMENTS.md`, so
that a cost that moved far enough to change what a tier requires shows there.
`calibration.json` holds the time the calibration probe took on the recording
machine.

### What the disagreements are, at the current pins

Read this with `DISAGREEMENTS.md` open; revisit it when a pin or the corpus moves.

- **Check, with one visible cause.** ftxi and Chess in 5D both give a check verdict
  at about eight hundred positions and differ at about one in ten of them - always
  ftxi "check", Chess in 5D "no check", and always where the side to move has a
  timeline it *may* play on but need not. Where there is no such timeline the two
  never differ. (`selftest` re-checks both halves of that against ftxi's own
  timeline counts, `5dtools stats`, so it cannot go stale unnoticed when a pin or
  the corpus moves.) The two sources show why: before looking for a
  royal capture, ftxi advances every timeline on which the side to move may play
  (`state::phantom`), while 5d-chess-js advances only the timelines at the present
  (`blankAction`). This is a difference in what "check" means, and it decides
  checkmate versus stalemate when no action exists. The rulebook has settled it
  (R1, decision H36, CHECKPOINT 1): check status passes all active boards in the
  present, so at these positions ftxi's "check" is wrong and Chess in 5D's "no
  check" right. An engine that follows the rulebook disagrees with ftxi at each of
  them that it reaches; `known-divergences.toml` records ftxi's (and ftxi:naive's)
  answers there, with R1's citation.
- **Two mate verdicts.** Chess in 5D calls two final positions checkmate where ftxi
  counts legal actions. On one of them, the record cwmtt's author named
  `chessin5d-illegal-move-example`, cwmtt counts legal actions too; the other is a
  custom position cwmtt cannot read. At the first (position 66), `observe
  --actions 66` lists the same 25 actions from ftxi, from cwmtt and from Chess in
  5D's own enumerator - each of them an action the library's own `move()` plays
  and whose result passes the library's own check test and present test, the tests
  `submit()` applies to the action. `submit()` itself refuses every one of them,
  because the checkmate search it runs first calls the position checkmate: it is
  that mate search that misses them.
- **One count.** At one position ftxi counts two fewer legal actions than cwmtt,
  and each oracle's second search confirms its own number. At the other positions
  both counted, the counts are equal - but at nearly half of those, "equal" means
  both stopped at the cap of 1000.
- **No hard legality disagreement.** No oracle refuses, as illegal, an action
  another plays.
- **Soft legality entries.** The notation parser does not read the long-algebraic
  move text of one record (`cwmtt/100_timelines`, `a2a3`). (`cwmtt/many2`, written in
  the official client's export form, had three more: ftxi, Chess in 5D and the
  notation parser refused its first move, `(0T1)Pe2(0T1)e3`, which only cwmtt read.
  Since CHECKPOINT 3 every source reads it converted to 5dpgn - "How a record
  reaches the sources", above. Read so, ftxi and the notation parser replay all 62
  of its actions and hold, at all 63 positions, the boards cwmtt held reading the
  export form - ftxi with the same unmoved markers and the same 63 counts; `observe`,
  2026-10-03.) ftxi finds
  no move matching two capture-marked jumps that cwmtt and the notation parser play
  (Chess in 5D plays one of them too); for that one ftxi ships its own copy of the
  record with the capture marker removed, which it then replays. The other is
  notation as well: ftxi's move reader matches a timeline's sign exactly and names
  the positive timelines of a game that starts with one timeline without a `+`, so
  it finds no `(+1T10)` - with the `+` dropped, it replays that record to its end.
- **Agreement on the records written here.** Both oracles that read a custom
  position call the two stalemates stalemate and the checkmate checkmate, and every
  oracle refuses the action that ignores a check.

What this means for the engine: no engine can agree with all three oracles. One
that answered exactly as ftxi does would fail every tier on telltales - each of
ftxi's answers on record as wrong in `known-divergences.toml` is one (H34) - and,
in the nightly tier, on its disagreements with Chess in 5D at those positions. A
disagreement no entry records leaves the ruleset it sits in ungated until it is
adjudicated and the losing oracle's answer is recorded in `known-divergences.toml`.

## Stable answers: the required sets (H32)

An oracle that does not reproduce its own recorded answers is nondeterministic or
limited by its search budget (design spec §11.0). Budget-limited answers are
recorded as unknown at their budget, nondeterministic ones as unstable, and each
tier then requires 100% of the stable answers it can reach - a tolerance would
hide regressions inside itself.

A recorded value is **required** in a tier when it is recorded (not unknown, not
unstable) and its recorded cost times the margin (`[stability] margin` in
`gate.toml`, 4 to start) fits the tier's budget for that source - and, for a Chess
in 5D mate verdict, when its mate search's cost times the margin also fits the
tier's mate budget. The outcome of a run (its status, reason and actions
accepted) is required in the same way, by when it arrived. The margin is at least
1, so no tier can require an answer that arrived after its own deadline on the
recording machine: a required value a tier's budget cannot reach is impossible by
construction. `verify` prints, for every tier and source, how many values are
required and how many are not, and why: unknown at the tier's budget (recorded
too late for it), unknown at the recording budget (cut, or stopped by a limit),
or unstable.

A gate run then:

- **reproduces every required value, identically.** One it does not produce is a
  finding, `golden-unreproduced`, naming the source, record, position and field,
  the value's recorded cost and how long the run took - and a run cut at its
  deadline before its last required position is one finding with all of that.
  The summary counts a required value produced with another value, and a
  required outcome that changed, as not reproduced too;
- **agrees with every other recorded value it produces** - required or not, a
  value that differs from the recording is `golden-mismatch`, and an outcome that
  differs `golden-status`;
- **accounts for every recorded value it did not produce outside the required
  set.** A budget explains one only where the run never got to its position (it
  was cut, or stopped, before it), where it was cut while it searched that very
  position and the value comes from that search, or where the run names a limit
  it ran into there (Chess in 5D's mate search; cwmtt's state driver running out
  of time). Those are counted by the budget that put them outside the set
  ("unknown at budget timeout_s=120: 71 recorded values not produced") - a count,
  never a ratio. Any other is `golden-unreproduced`: a run that reached a position
  and no longer gives an answer it used to has lost it to no budget, whether or
  not the tier requires it.

Every gate report (`oracles/.cache/reports/gate-<tier>.json`) records each source's
budgets and signature, the margin, the number of jobs, the platform and CPU
count, and the **calibration probe**: a fixed workload (`[calibration]` in
`gate.toml`: ftxi's naive search over the first 38 positions of
`cwmtt/manyChecks`, about 12 s on the recording desktop) timed before the oracles
run, beside its time on the recording machine. When a required value goes
missing, the summary says how much slower the runner is ("this runner is 2.5x
slower than the recording machine; margin 4"), so that a slow machine can be told
from a regression.

The tier that can enable the proof channel is held to more, statically
(`proof-channel-vacuous`). Each source's required set there must hold:

- every terminal verdict (checkmate, stalemate, "no action exists") the source
  recorded;
- every terminal verdict another source recorded where this one's recording was
  cut at its budget - at or after its horizon, on a record the other source read
  as this one did (another search of the same implementation is named first);
- at least a reviewed share (`[stability] proof_share`, 90%) of what the source
  would have recorded but for its recording budget: its stable values, the values
  behind its horizons (the positions a complete replay of the record reaches -
  another search of the implementation that was not cut, or failing that any
  source that was not - at as many values a position as the source gives), the
  values a limit left unknown and the unstable ones. A source that recorded
  nothing on the tier's records fails it outright.

The recording budget is this tier's, so starving a source - a lower budget, a
recording made on a loaded machine - moves its answers behind a horizon, where
they still count against it: the check gets harder, not easier. Otherwise a
source could be listed in the tier and starved of time, and the tier would pass
on answers nobody asked for. This replaces a 90% ratio of reproduced answers
that used to be fixed in the code, and like it, it is about enabling the proof
channel: `verify` prints it, and a complete run of that tier fails on it; a run
restricted with `--source` or `--record`, and a run of any other tier, which can
enable nothing, do not. Its remedy is the tier's budget
for the source - which is also the recording budget, so changing it means
re-recording.

At the current pins and budgets the naive searches do not meet it: their
recordings are cut at 300 s on records whose last position another search calls
checkmate or "no action", and on a two-recording measurement on the recording
desktop (2026-10-01) the release tier required 84.9% of what `ftxi:naive` would
record and 77.3% of what `cwmtt:naive` would. The proof channel stays shut until
the release budgets, or the policy, change.

### What each tier requires

`DISAGREEMENTS.md` holds a table of every tier and source: its budget, how many
values and outcomes the tier requires of it and how many are recorded, and how
many of its recorded terminal verdicts are required. It is rendered from the
golden and cost files and `gate.toml`, and `verify` renders it again: a budget
lowered, a cost file edited, or a recording that moved answers behind a horizon
changes a line of it, and reviewing that line is reviewing what the gate checks.

## Recorded-answer integrity (H33)

Recorded answers are immutable, with committed content hashes verified before
every comparison; the comparison logic lives in harness code the engine adapter
cannot modify; harness changes require review (design spec §11.0).

- `golden/MANIFEST` holds the SHA-256 of every file in `golden/` - the golden
  files, their cost files and the calibration reference - in `sha256sum`'s format
  (`cd oracles/golden && sha256sum -c MANIFEST` checks it without the harness; the
  harness folds CRLF to LF before hashing). `golden --write` writes it, after the
  files it vouches for; a run that records only some sources keeps the other
  files' hashes only once they are shown to match. Every read of a recorded answer
  verifies its file against it first, and `verify` checks the whole directory: a
  file edited by hand, a file nobody recorded, a file that went missing
  (`golden-integrity`). It lives beside the files it covers rather than in the lock
  file, which holds the pins - a reviewed change with its own procedure that every
  re-recording would otherwise touch.
- What MANIFEST catches is an edit that does not rewrite MANIFEST as well: an
  accident, a partial edit, a bad merge, a file changed while a run lasts. A file
  edited together with its line of MANIFEST - one `sha256sum` away - passes it;
  only review catches that. What makes such an edit visible in review is that
  MANIFEST changes only with the files it covers, and that what each tier requires
  is the table in `DISAGREEMENTS.md`, which `verify` renders again from the golden
  and cost files: a cost file edited to shrink a required set changes a line of it.
- The engine's baseline, `engine-baseline.json` - what the engine read on the last
  green run of the commit tier (H47; "The engine's baseline", below) - is a recorded
  answer too, kept apart from `golden/`: `gate.toml` `[baseline]` pins it by its
  SHA-256, as it pins the rulebook, so that re-recording the oracles never re-pins
  it (every `golden --write` rewrites MANIFEST). It is written only by
  `harness.py baseline --write`, after a green run - never by a gate run, and never
  by the engine, which may not name anything in the harness - and every read
  verifies it against its pin first (`engine-baseline-integrity`): a gate run's,
  `verify`'s, and `baseline --write`'s own read of the baseline it replaces, which
  it lists the new one's losses against - so it runs and writes nothing while the
  file on disk is not the one pinned. The pin's limit is MANIFEST's: a file edited
  together with its pin passes it, and only review catches that. What shows such an
  edit in review is that the file and its pin change only together, one line per
  record, and that `verify` fails a baseline that holds no line for a record of its
  tier (`engine-baseline-stale`), so a line cannot simply be deleted.
- A gate run reads the golden, cost and calibration files, the engine's baseline
  and `known-divergences.toml` once, when it starts, verified, and hands that one
  snapshot to every consumer: nothing is read from them again while it runs, so the
  divergences that excuse the engine are the ones on record before it ran.
- Every file under `oracles/` is hashed when a run starts and again when it ends -
  the harness's code and tests, its entry point, the drivers, `golden/`,
  `known-divergences.toml`, `gate.toml`, the engine's baseline, the corpus,
  `DISAGREEMENTS.md`, the lock file - but for the caches (`.cache/` and other dot-directories, `__pycache__/`)
  and the pinned oracles under `third_party/`. Anything that changed, appeared or
  went while it ran fails it (`harness-modified`): a module planted beside the
  harness's code as surely as an edit to it.
- The entry point, `harness.py`, runs nothing from the tree but source. Python
  would otherwise run a `.pyc` planted in a `__pycache__` directory whose header
  copies its source file's time and size (the source, and so its hash, unchanged),
  would import a file beside `harness.py` in place of a standard-library module
  of the same name, and would load an extension module (`gate.cp314-win_amd64.pyd`,
  `gate.cpython-314-x86_64-linux-gnu.so`) ahead of the source file beside it - each
  planted once, and run by every later run. So before anything else is imported,
  `harness.py` takes its own directory off the module search path, writes no
  bytecode and reads none from beside the source (Python's cache points at a
  fresh, empty directory of its own), imports a module from any directory of the
  harness only from its source file, and loads the harness's package from its
  source by path. Run the harness through `harness.py`, never as
  `python -m oracle_harness`.
- What is not hashed is the oracle builds under `.cache/build`, which a run
  executes: what they answer is held to the golden files - every required value
  identically, and every other recorded one where it is produced.
- `.github/CODEOWNERS` puts `oracles/` and `.github/` - the workflows that decide
  which tier CI runs, and how, and CODEOWNERS itself - under its owner's review.
  It binds once branch protection on `main` requires a review from a code owner.

What remains is trust in review.

## Telltales (H34)

Every entry of `known-divergences.toml` records an oracle's error. An engine that
gives the same wrong answer at the same place - while it holds the position the
oracle held when it gave it - probably has that oracle's logic, copied, ported or
renamed past the boundary check. So in the engine differential, an engine answer
that agrees with an entry's recorded oracle answer at its record and position
(parse kinds and replays included) is a **telltale**: a finding that fails the
gate - "the engine reproduces chessin5d's recorded error at r/std position 1
(in_check = False): its logic may have been copied from that oracle". Agreeing is
judged by the comparison's own rule, not by the same words: an engine that
refuses an action the oracle wrongly refused reproduces the error whatever reason
each gives (a parser that fails where the oracle's could not match the move text,
say), one that calls checkmate or stalemate a position the oracle wrongly holds to
have no action reproduces that, and a count at the cap agrees with any as large.
It is checked for every entry whose record the engine replayed, whether or not
that oracle runs in the tier - against the golden file, while it still records the
entry's `oracle_value` (`verify` holds the entry to it) - and it is never reported
as `divergence-stale`, whose remedy - remove the entry - would be exactly the wrong
one. In parse mode the engine's having played an action is no verdict, so it
reproduces no oracle's acceptance of one; its refusals, and its reading of a
record's set-up, count as in full mode.

## The engine contract

The gate reaches the engine across a process boundary and nowhere else. Set
`[engine] command` in `gate.toml` to the argv of a program - kept with the engine,
never inside `oracles/`, where it would sit beside every answer the oracles have
given (`verify` refuses such a command) - that:

- is run from the repository root with `--cap N` appended - a program named by a
  relative path is the root's on every platform, wherever the harness was started
  (the harness resolves it there: Windows would otherwise look for it from the
  directory the harness was started in), so the file `verify` checks is the file
  that runs, and the gate's report names it, with its SHA-256 (`engine_run`);
- reads one 5dpgn record on stdin (UTF-8, LF line endings);
- writes one JSON object per line on stdout, flushing each line - for the
  Standard start and `1. e3`:

```
{"event":"state","index":0,"size":"8x8","markers":true,"boards":["0:1:w:r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*"]}
{"event":"position","index":0,"side":"white","in_check":false,"legal_actions":20,"terminal":"none"}
{"event":"state","index":1,"size":"8x8","markers":true,"boards":["0:1:w:r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*","0:1:b:r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/4P3/P*P*P*P*1P*P*P*/R*NBQK*BNR*"]}
{"event":"position","index":1,"side":"black","in_check":false,"legal_actions":20,"terminal":"none"}
{"event":"end","status":"ok","actions_accepted":1}
```

- reports position *k* exactly when it has accepted *k* actions, so a run that
  reports *n* positions ends with `actions_accepted` = *n* − 1;
- prints, before each position event and before it searches that position, a
  `state` event for the same index, with exactly the fields shown:
  - `size`: the boards' width and height, `"WxH"`;
  - `markers`: `true` - the engine tracks which pieces have not moved;
  - `boards`: every board of the multiverse - superseded and head boards, every
    timeline, both colours - in any order, each as `"<L>:<T>:<c>:<placement>"`.
    `L` is the timeline as the notation writes it: `0` for the central timeline
    of a game that starts with an odd number of timelines, `-0` and `+0` for the
    central pair of an even one, any other as a signed or unsigned integer (`+1`
    or `1`, `-1`). `T` is the turn as displayed (`0` only for a turn-zero board),
    `c` is `w` or `b` for the side to move on that board, and the placement is the
    5DFEN pieces field of the notation specification: ranks from the top down,
    separated by `/`, files from a; the letters `P N B R Q K U D S W C Y` (white
    upper case, black lower case); runs of empty squares as base-10 numbers; and a
    `*` directly after every pawn, rook, king or brawn that has not moved, and
    after no other piece - whether or not a castling or a double step is still
    open to it: a rook keeps its marker after its king has moved, and every
    pawn, rook, king and brawn of a named variant's set-up is marked.

  The engine prints boards, never a hash: the harness checks every board against
  the size, spells each label one way, and computes the fingerprint itself. Like
  any source that says what it set up, it says it at every position, from
  position 0, with `markers` the same throughout;
- answers **every** field of every position: `legal_actions` as `min(count, N)` -
  counting actions as "What is compared" defines them (a set of moves, the
  branching ones in creation order), not the distinct states they reach - and
  `terminal` as `none`, `checkmate` or `stalemate`. An oracle may leave a
  question unanswered, and may say nothing of its state; the engine is what is
  being certified, and a null from it would be a verdict nobody compared. Its
  `terminal` is a proof (H37): an engine whose search did not terminate says
  neither `checkmate` nor `stalemate` - it ends the record with `error`, or the
  gate's deadline cuts it, and either fails the gate;
- puts exactly the five fields shown in each state event and the six in each
  position event, and no others (the oracle drivers add timings of their own, so
  an oracle's output passed off as the engine's is refused);
- ends with exactly one `end` event, whose `status` is
  `ok` (every action replayed),
  `rejected` (the next action is not legal - `reason` is `illegal_move` or
  `cannot_submit` - or the engine's parser could not read its move text:
  `unreadable`),
  `unsupported` (a ruleset the engine does not implement, said before the first
  position - allowed, but that ruleset is then not gated), or `error`;
- may refuse the record's set-up - headers or board strings it cannot read, or
  that contradict each other - by ending `rejected`, with `reason` `unreadable`
  and `actions_accepted` 0, before its first position (and so with no state or
  position event at all). That is refusing "action 0", and it is compared like
  any refusal. Only as `unreadable`: a set-up is no action that can be illegal.

A refusal as `unreadable` is no rules verdict. Of an action an oracle played, or
of a set-up an oracle set up (action 0), it is a soft legality divergence: a
finding unless `known-divergences.toml` excuses it with an entry that cites the
notation specification (whether the text is valid notation is the
specification's to say, never the rulebook's). Excused or not, a record the
engine could not read is not one it replayed: its ruleset is not gated, and no
oracle - not even one that refuses the same action as illegal - stands behind it.

Breaking the contract, an `error` and a time-out each fail the gate. So does every
disagreement with an oracle - a parse divergence included - unless
`known-divergences.toml` records it. (The engine's time limit is its own:
`[engine] timeout_s` per record, or the tier's `timeout_s` where that is longer -
a tier's budgets decide what it requires of the oracles, H32, and the gate
compares the engine's answers, not its speed.) An oracle value that differed between the
recordings (unstable) is taken out of each fresh oracle report before it is
compared with the engine: it can neither fail the engine nor stand behind it. An entry for a placement divergence cites the
notation specification, never the rulebook; one for a marker-only divergence may
cite either. And an oracle that set up something other than the engine did checks
nothing there, excused or not:

- from the position where its boards and pieces part from the engine's (a
  placement divergence), it stands behind nothing: it no longer follows the record
  as the engine read it, and none of its answers there or after counts as checking
  the engine;
- at a position where only its unmoved markers differ from the engine's, it still
  follows the record - it read the same moves to the same pieces, and a marker
  that decides whether an action of the record is legal shows as a legality
  divergence - but its answers there are about a position with other rights to
  castle or to step twice, and do not count as checking the engine. Where the
  markers agree again, its answers count again.

The gate's verdict is per **ruleset**, because exactness is inherited per variant.
A record's ruleset is the variant it declares - its `Board` header, else its
`Variant` header, else Standard - unless it sets up its own position, with 5DFEN
boards or a custom board (`Custom ...`, `Even`, `Odd`): then it counts towards no
named variant and is filed under "(custom positions)". A ruleset is **gated** when,
over all of its records:

1. the engine replayed each one - to its end, or to an action it holds illegal
   (not one it could not read) - and none holds an unexplained divergence;
2. each one was followed as far as the engine went by at least **two** oracles
   (H47, CHECKPOINT 8: a single oracle cannot tell its own bug from the engine's) -
   two independent implementations, so two searches of one count once, and the
   notation parser, which judges nothing, never counts - and where the engine
   refuses an action, by oracles that refuse the same action as illegal (one that
   could not match the move's text confirms nothing). An oracle that replayed the
   record differently - accepting an action the engine refuses, or the other way
   round, reaching a different end, or setting up a different position - does not
   count, even where a known divergence excuses the difference;
3. for each of check, count and mate, the oracles give a verdict that settles it,
   and agrees with the engine, at no less than `min_checked` (75%) of the
   positions the engine reports. An answer the engine contradicts confirms
   nothing, even one on record as the oracle's error, and neither does an answer
   about another position than the engine's: after the oracle parted from the
   engine, or where its unmoved markers differ from the engine's.

The last two matter. Several variants in the corpus are read by a single oracle,
some by none, and the single oracle is sometimes cwmtt, which has no check verdict
to give: an engine must not "pass" a ruleset that nothing independent looked at -
nor one that a single oracle vouches for, since where that oracle and the engine
share a mistake nothing would show it. A ruleset with a record that only one
oracle follows is `NOT gated`, the reason naming the record and its oracle. For
each gated ruleset the gate prints how many positions had an independent
answer for each observable; at how many of the engine's positions an oracle held
the engine's boards and pieces, and at how many its exact state, unmoved markers
included; and the fewest oracles that stood behind any record - two at the least.
Where a ruleset falls short of the floor and answers were set aside because the
oracle's markers differed from the engine's, the reason says how many.

"Very Small - Open" is ungated in every tier first because the engine cannot read
two of its records, cwmtt/ctp1 and ftxi/verysmall: each writes a jump without the
moving piece's square, which the notation specification requires
(`known-divergences.toml`, KD-0044, KD-0045, KD-0049, KD-0050). And at the current
pins the last condition would keep it ungated even then: ftxi is its only oracle
with a check verdict, and ftxi's layout for it draws the king and rook of each side
without the unmoved marker, so it holds them as having moved where an engine that
follows the contract does not (`DISAGREEMENTS.md` shows the marker-only divergence
from cwmtt, which marks them). Castling is impossible in that layout, so the two
positions play alike - but the harness does not know the rules, and does not count
an answer about another position.

Only a tier that runs every oracle search over the whole corpus (`nightly`,
`release`) gates a ruleset, and there an engine with no finding against it that
comes through no ruleset has **failed** (`engine-nothing-gated`): agreeing with the
oracles about nothing is not a pass. Any other tier - `commit`, which runs one
rules oracle over the commit records, and so could gate no ruleset in any case -
is a **smoke test** (H47): what it says of the rulesets is never a gating verdict,
and it fails an engine only on a finding against it. Its last lines say
`engine differential: SMOKE TEST - PASSED` (or `FAILED`) and that the tier gates
nothing; every ruleset is listed as `NOT gated`, with its reason - in the commit
tier, with one rules oracle, never that it met the gating conditions, which no
ruleset can there; in a smoke test of a tier with two oracles (a tier cut to some
of nightly's records, say), one that met them with that tier's oracles alone says
so; and the report holds no ruleset under `gated`, and the reason under
`smoke_test`. The gate tells the two kinds of tier
apart by what a tier runs (`Tier.full_strength`), never by its name or by a count
of oracles. A smoke test still fails an engine that declines a record it read on
the last green run, or that reads none (H47, CHECKPOINT 8; below) - so an engine
that declines every record fails every tier.

In every tier the engine fails on an unexplained divergence - a disagreement with
an oracle that `known-divergences.toml` does not record - on an answer a known
divergence records as an oracle's error (a telltale, H34, above), on a breach
of the contract, an `error` or a time-out (above), on a record it read on the
last green run and now declines (`engine-declined`), and on reading none of the
tier's records (`engine-read-nothing`). A divergence on record fails no
tier: there the oracle's answer confirms nothing, and only another oracle's answer
can confirm the engine.

### The engine's baseline (H47)

"The commit tier still fails if the engine declines a record it read on the last
green run, or reads none" (H47, CHECKPOINT 8). The last green run is on record in
`engine-baseline.json`: for every record of the tier `gate.toml` `[baseline]`
names (`commit`), how the engine's replay of it ended - `ok`,
`rejected:<reason>`, `unsupported` - and how many positions it reported, on a
complete run of that tier, the engine in full mode, that passed. One record per
line, so that a re-recording is reviewed as a diff:

```
"cwmtt/ctp1": {"end": "rejected:unreadable", "positions": 10, "sha256": "0d7d5788..."},
```

- **Declines a record it read on the last green run.** The engine reads less of a
  record than then: it reports fewer positions, or as many but without the
  verdict it ended with then - other than `ok` where the baseline ended `ok`, or
  other than a refusal as illegal (`illegal_move`, `cannot_submit`) where the
  baseline ended in one (a refusal as unreadable is no verdict). A record the
  baseline gives no position - a ruleset the engine declined, a set-up it could not
  read - was not read, and cannot be declined; one the engine now reads further
  is no decline (the summary says how many, and that a new baseline would hold
  the engine to them); and a run that ended in an `error` or a time-out is
  `engine-failed` already, and is not reported twice.
- **Reads none.** In a complete run of a tier, the engine replays none of the
  tier's records: none ends `ok`, or in a refusal as illegal.
- **Where.** Every tier compares the engine with the baseline, over the records it
  holds that the run runs - nightly and release run every record the commit tier
  does, and a tier that runs every oracle checks at least everything the smoke test
  checks (H47 names the commit tier; spec issue 132 in `docs/BUILD_LOG.md` says why
  every tier). Not in parse mode: the baseline holds what the engine read in full
  mode. A run restricted with `--source` or `--record` reports each record declined
  (a finding about that record), but not reading none (a check of a tier as a
  whole). With an engine compared in full mode, `verify` fails while no baseline
  is pinned, or the file is not the one pinned (`engine-baseline-missing`,
  `-integrity`, `-invalid`), or it is about record text the corpus no longer holds,
  or holds no line for a record of its tier (`engine-baseline-stale`) - so a
  commit-tier record added to the corpus, or one whose text a pin bump changed,
  needs a new baseline (`CONTRIBUTING.md` says when, in its procedures for adding
  records and changing a pin).

```
python oracles/harness.py baseline           # is engine-baseline.json the baseline gate.toml pins, and current?
python oracles/harness.py baseline --write   # run the [baseline] tier in full; if it is green, record and pin it
```

`baseline --write` first reads the baseline it replaces, verified against its pin
like every read of it: where the file on disk is not the one pinned - edited by
hand, or gone - it runs and writes nothing (restore it: `git checkout --
oracles/engine-baseline.json`). It then runs the tier in full with the engine -
comparing it with no baseline, since one is being replaced - and records what the
engine read only if the run is green on everything else, reading none included:
never from a run that is not. Last, it lists what the new baseline holds otherwise
than the pinned one, losses first: a record the engine now reads less of is one
the commit tier no longer holds it to - a loss of checking, never churn, and never
recorded to make a gate run pass. The pin, like MANIFEST, guards against
accident, not against the file and its pin changed together - by hand, or by
recording a loss: that is review's to catch (H33; "Recorded-answer integrity",
above).

The proof channel may be enabled only by a complete run of a tier that requires an
engine and runs every oracle search over the whole corpus (the `release` tier),
with the engine compared in full mode, and only for the named variants it gated,
never on the strength of custom positions. Such a run passes only with 100% of
every source's required set reproduced, and `verify` holds those required sets
to every terminal verdict recorded and the reviewed share of the stable values
(`[stability] proof_share`), so that no tier can list an oracle and then starve it
of time.

### Parse mode

An engine that reads records but does not yet judge them - no check detection,
no counts, no mate - is compared in **parse mode**: set `[engine] mode = "parse"`
in `gate.toml`. The harness then appends `--parse-only` after `--cap N`, and the
engine:

- prints, for every position *k*, the `state` event exactly as above (the five
  fields, `markers` true) and a `position` event with exactly the six fields, in
  which `side` is set and `in_check`, `legal_actions` and `terminal` are `null` -
  never a guess: an answer there breaks the contract;
- ends as above, and may refuse an action, or the record's set-up, as above -
  `unreadable`, `illegal_move` or `cannot_submit`, the set-up only as
  `unreadable`; `unsupported` is still said only before the first position.

The gate then compares what the engine set up with what every oracle set up
(placement and marker-only divergences are findings unless excused, as above),
its side to move, and how it replayed each record - how many actions it read,
which it accepted and which it refused - with each implementation's replay. A
refusal as `unreadable` is a soft legality divergence, as in full mode. Its
playing an action an oracle holds illegal is no
verdict (it has no check detection, and plays what it reads, as a parse-only
source does): it is not a divergence, a known divergence about it is not looked
at (so it is never reported stale), and it reproduces no oracle's acceptance of
an action (no telltale). It never compares or gates check, counts or mate, never
marks a ruleset "gated", and the proof channel stays disabled ("parse mode");
telltales are otherwise checked as in full mode. The last lines of the output say
`engine differential: PARSE MODE - PASSED` (or `FAILED`) with the agreement
numbers: records read (and set-ups refused); positions at which an oracle said
what it set up - up to and including where it parted from the engine - and of
those, at which it held the engine's boards, and its exact state; sides compared
and agreed; and implementation replays compared, agreed, and agreed but for an
action the engine played that the oracle held illegal (counted apart: that is no
verdict either way). An engine in parse mode that replays no record of the tier,
or whose replays no oracle described and no implementation's replay was compared
with, has failed. The release tier, which requires an engine, fails in parse mode.

## Known divergences

When the engine and an oracle disagree, the rulebook decides. If the oracle is
wrong, the disagreement goes into `known-divergences.toml` with the citation that
settles it, and a link to the upstream report - or why it is unreported. Upstream,
a defect is reported as a bug and a disagreement with a documented rule as a
question citing the source (register H27, CHECKPOINT 8); the agent drafts them in
`docs/checkpoints/upstream/` and the project owner files them. None is filed yet,
and each entry says why: the two defects ruling R7 recorded are drafted as bug
reports, and ftxi's reading of the check test - the 147 entries R1 settles - as one
question, `ftxi-check-test-passed-boards.md`, which asks whether the game passes
the boards ftxi passes; if it does, the rulebook is wrong. Once a report is filed,
its URL takes the place of `unreported: ...` in each entry it answers.

The rulebook is `docs/rules/README.md` (decision H38), and it is pinned:
`gate.toml` `[rulebook]` holds the SHA-256 of its text (CRLF folded to LF), and
`verify` - and so every gate tier - fails with `rulebook-changed` once the file no
longer matches it ("the rulebook changed: re-run the gate and update the pin with
..."). A change to the rulebook is a reviewed change: read its diff, re-check every
entry that cites it, move the pin, and re-run the gate.

```
python oracles/harness.py rulebook         # is the rulebook still the text gate.toml pins?
python oracles/harness.py rulebook --pin   # pin its current text (after that review)
```

The pin lives in `gate.toml`, with the gate's policy and under the same review,
not in `golden/MANIFEST`: every `golden --write` rewrites MANIFEST, so
re-recording the oracles would re-pin the rulebook as a side effect - and the
recorded answers do not depend on the rulebook at all. An entry excuses exactly one
observation - this oracle, at this pin, on this record, at this position, saying
this, where the engine says that - and the gate fails if it stops matching for any
reason, including the disagreement going away. What an entry says the oracle
answers is checked against the golden files. An entry for a marker-only divergence
is at the first position of a run of positions whose markers differ, and excuses
that run; markers that part again after agreeing need an entry of their own.

Some entries cite the notation specification in place of the rulebook:

- a **placement** divergence must cite it - the oracle read the record's moves, or
  its set-up, as other boards than the engine's;
- a **marker-only** divergence may cite it ("5DFEN Strictness", which says which
  pieces carry the marker) or the rulebook (what having moved means);
- a **length** divergence, or a **legality** one where the oracle stopped because it
  could not match a move's text (a soft rejection) and the engine played on, may
  cite it where the cause is notation the oracle does not read and no rule at all -
  or, where the specification is silent on how such text is read, the rulebook's
  project decision that settles it (R11: a capture mark that disagrees with its
  move is read as the move's squares define it, register H51). If the oracle really
  does hold the move illegal, it is a rules matter after all;
- a **legality** one where the engine, in either mode, refused as unreadable
  text an oracle played (`engine_value = "rejected:unreadable"`) - an action's, or
  at action 0 the record's set-up: its headers or board strings - must cite it:
  whether the text is valid notation is the specification's to say, never the
  rulebook's.

Every other kind needs the rulebook. The file's header says how, and where its
entries came from: the engine's gate runs of 2026-10-02 and, with the engine
configured, of 2026-10-17 - between them every disagreement of the engine's with a
recorded oracle answer, each settled by its citation (`docs/BUILD_LOG.md`, step 2).
Capture marks are read leniently (CHECKPOINT 8: rulebook R11, register H51, a
project decision): a move is read as its squares define it, whether or not it is
written with `x`, and the engine logs a mark that disagrees as a warning (on its
standard error - never in the protocol). So the engine plays cwmtt/futures action
17, an `x` on a jump that takes nothing, as cwmtt, Chess in 5D and the notation
parser do, and ftxi's refusal of it is KD-0170, citing R11; KD-0167 to KD-0169,
which recorded the strict reading the engine followed before the ruling, were
withdrawn on its authority.

## Other commands

```
python oracles/harness.py observe --source ftxi --record cwmtt/standard   # one source, one record
python oracles/harness.py corpus             # list and verify the corpus
python oracles/harness.py corpus --unlisted  # records in pinned sources that the manifest omits
python oracles/harness.py boundary           # the boundary check alone
python oracles/harness.py provenance --write # refresh the inventory in PROVENANCE.md
python oracles/harness.py gate --tier nightly --source cwmtt --record cwmtt/standard   # a restricted run
```

A gate run restricted with `--source` or `--record` - any use of either, even one
that names every source of the tier - is a developer's convenience, and **never
produces a tier verdict** (H47, CHECKPOINT 8). Its summary and its report say
`restricted - not a tier result`; it prints no `oracle layer [tier]` line, states
no ruleset gated or not gated, and its proof-channel line is `DISABLED -
restricted - not a tier result`. It reports every finding it makes - a
divergence, a telltale, a breach of the engine contract, a record the engine read
on the last green run and now declines, a lost required value - and its exit code
reflects those findings only: 1 if it made any, 0 if not. The checks that judge a
tier as a whole are not made: that the engine came through some ruleset
(`engine-nothing-gated`), read some record (`engine-read-nothing`), or was compared
at all (parse mode's `engine-nothing-compared`); that a tier which requires an
engine has one; and that a tier which can enable the proof channel requires enough
of each source to enable it (`proof-channel-vacuous`, about enabling the proof
channel, which a restricted run never does - `verify` still prints it). Its report
is `gate-<tier>-restricted.json` in the harness cache, with `restricted` and
`restricted_to` set, the verdicts of a tier - `ok`, `oracle_layer_ok` - null, its
`findings` listed, and nothing under `gated` or `ungated`, so it never stands in
for a complete run's.

`observe` can also ask a source's driver for two diagnostics, which it prints after
the report as one JSON object per line. They are never compared and never
recorded, and a source whose driver does not print them says so:

```
python oracles/harness.py observe --source ftxi --record cwmtt/standard --explain     # what each check test looked at
python oracles/harness.py observe --source ftxi --record cwmtt/standard --actions 12  # the legal actions at position 12
```

`--explain` adds an `explain` event after each state event - for ftxi and
5d-chess-js, the timelines each advanced to phantom boards for its check test and,
in check, the captures it found; for cwmtt, the boards it holds playable and its
present; for the notation parser, the layout it set up and then the moves of each
action as it resolved them. `--actions K` adds an `actions` event listing the
source's legal actions at position *K* - ftxi, cwmtt and 5d-chess-js; the notation
parser has none - at most 5000, each a list of moves
`{"from":{"L":"0","T":5,"c":"w","square":"e2"},"to":{...},"promotion":null}`.
`truncated` says whether the list was cut short, and `count` how many actions the
source found: the number listed, or more where it went on counting past the list
(5d-chess-js) - when `truncated` is true, a lower bound. Fields a driver adds of its
own (5d-chess-js: `complete`, `sequences`, `accepted`, `limit`, `error`, `caveat`; cwmtt: `error`) say why a
list stopped, and are shown as printed. The harness spells the labels one way and
sorts the moves of each action and then the actions, so two sources' lists of one
position can be compared action by action. Sorting drops the order of an action's
moves: the same entry listed twice is one set of moves in two orders of its
branching moves, which count as two actions (`legal_actions`, above).

## What this is not, yet

- **Not a passed gate.** The engine is compared in every tier, and the nightly
  tier gates the rulesets its oracles can vouch for (`docs/BUILD_LOG.md`, step 2,
  names them). The proof channel opens only on a complete run of the release tier,
  whose naive searches' required sets are still too thin for it at their budgets
  (`verify`'s `proof-channel-vacuous` notes, H48), and on the user's sign-off (H41).
- **Not enforced as a required check.** The two workflows run on Linux - the
  commit tier on every push and pull request, the nightly tier every night - and
  first went green on 2026-10-01. They are not yet required status checks on the
  branch, so until they are, the rules bind whoever runs `verify`.
- **Not a large corpus.** The records are the `.5dpgn` test records the pinned
  projects ship, the notation repository's sample games, and four records written
  here to cover what those lack (two stalemates, a checkmate beside them, an
  action that ignores a check). cwmtt's tests in its own `.tn` format and the
  games embedded in 5d-chess-js's JavaScript tests are not used. Most records
  were written as test cases. Real-game corpora at scale are still to be found;
  the manifest format is ready for them.
- **Not an even cross-check.** The spec asks for move counts, check verdicts and
  mate/stalemate verdicts to be cross-checked. How far each is, per variant and per
  pair of oracles, is in the tables at the top of `DISAGREEMENTS.md`, and it is
  uneven: counts are capped at 1000, so on wide positions two oracles are compared
  only as "both at least 1000"; several variants have one oracle, or none; and
  stalemate is exercised only by the records written here.
- **Not adjudicated beyond the engine's disagreements.** The rulebook is pinned
  (H38) and has ruled on the check disagreements (R1) and two oracle defects (R7);
  `known-divergences.toml` records, each with its citation, where the engine and an
  oracle's recorded answer disagree. The oracles' disagreements among themselves
  that touch no answer of the engine's are listed in `DISAGREEMENTS.md`, not
  settled.
- **Not a repetition oracle.** None of the three observables touches the 5D
  repetition rules, the most load-bearing open item in the register.
- **Not proof against a port.** The boundary check finds pasted code. An oracle's
  logic rewritten in another language, or pasted with its names changed, would
  pass it.
- **Not proof against a dishonest adapter.** The gate trusts that the `[engine]
  command` asks the engine. A shim that printed what the oracles say would pass
  everything. The gate refuses a command inside the harness, the boundary check
  refuses engine code that names a path into it, an oracle driver's output
  passed through unchanged carries fields the engine's may not, a harness file
  changed while the gate ran fails it, and an engine that repeats a recorded
  oracle error is caught by its telltale - but a shim that took the answers some
  other way and rewrote them, where no oracle is on record as wrong, would go
  unseen. That is left to review.
- **Not a check that two parsers read the same move, where a source cannot say
  what it set up.** Two readings of one move are caught by the fingerprints only
  where both sources export a state event. A source that cannot is aligned with
  the others by counting accepted actions alone: if it and another read one move's
  text as two different legal actions, every position after it is a different
  position under the same number, and that shows only where their answers happen
  to differ. `DISAGREEMENTS.md` counts, per pair of sources, the positions whose
  alignment was verified and those whose alignment was not. And a fingerprint
  compares positions, not moves: two readings that reach the same position are
  not told apart - which leaves the verdicts compared there about the same
  position, as they should be.
- **Not a check of every marker inside a run.** The harness keeps hashes, not
  boards, so it can tell that two sources' unmoved markers differ at a position,
  not which markers. A marker-only divergence is reported where a run of such
  positions begins; a second difference that arises while the first lasts is not
  seen until the markers agree again.
