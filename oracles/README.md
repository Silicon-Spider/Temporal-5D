# Oracle harness

The differential-testing gate for the rules engine (design spec §11.0; decisions
H23–H30), and everything it needs: three independent implementations of the rules
and an independent reader of the notation, pinned; a pinned corpus of game records;
the machinery that runs the one over the other; and the check that keeps all of it
out of the engine.

**There is no engine yet.** What is here is the oracle side of the gate, built and
running, with the engine's slot empty.

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
  gate.toml               tiers, budgets, the engine command, boundary policy
  known-divergences.toml  where an oracle is wrong, with the rule that says so
  DISAGREEMENTS.md        what the oracles cover, and where they contradict each other (generated)
  corpus/manifest.toml    the pinned game records, by source, blob and hash
  corpus/records/         the few records written for this project
  golden/                 what each oracle says about each record (generated, reviewed)
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
python oracles/harness.py gate --tier commit # the per-commit gate
```

The harness needs Python 3.12+ and git. Building the oracles needs:

| Oracle | Needs |
|---|---|
| `ftxi` | a C++23 compiler (g++ 13 or newer), CMake 3.20+, and Ninja on Windows |
| `cwmtt` | GHC 9.10.3 exactly - nothing else; no z3, no stack, no package download. It builds the stock CLI and the harness's state driver (below) |
| `chessin5d` | Node 18+ and npm (three small MIT packages, installed from its own lockfile with scripts disabled) |
| `notation` | Node 18+ - nothing is installed: the parser needs no package |

`--bootstrap` fetches what is missing into `oracles/.cache/` - CMake and Ninja as
pip wheels in a private virtualenv, GHC as a checksum-verified archive (Windows
only; elsewhere install GHC 9.10.3 yourself, e.g. with ghcup). It changes nothing
outside that directory. Never run `git submodule update --init --recursive`: the
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
| `in_check` | The "phantom" check of spec §11.0: can the opponent capture a royal once every eligible timeline has been advanced to its phantom board? |
| `legal_actions` | Number of distinct legal actions (whole turns), counted up to a cap of 1000. A count equal to the cap means "at least that many". |
| `terminal` | `none` while a legal action exists; otherwise `checkmate` or `stalemate` (`no_action` from a source that cannot tell which). |
| legality | Whether each recorded action is accepted. One source refusing an action another went on to play is a disagreement; so is one source saying the record ends where another found a further action, whether it played that action or refused it. |

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

Each oracle answers with its own reading of these definitions - which timelines
count as "eligible", for one. Where the readings differ, the reports disagree, and
the rulebook, not the harness, says who is right (see "What the disagreements are"
below).

Three principles keep the comparison honest:

- **An unknown never disagrees with anything.** A source that cannot answer a
  question, or ran out of time, says nothing. Nothing is inferred on its behalf.
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

The harness never parses the moves of a record. Each oracle reads a record with its
own parser, the engine will read it with its own (H28), positions are aligned by
counting submitted actions, and the alignment is checked by what each source says
it set up: boards, which the harness reads, never moves. What the harness reads
out of a record is what game it declares itself to be - its `Board` header (or its
`Variant` header when it has none), whether it carries 5DFEN boards, whether it has
a `Timeline` or `Promotions` header - and whether its move text holds a variation
or a turn number written `12w.`. One driver also drops a PGN result token (`1-0`,
`1/2-1/2`) from the end of the move text, because 5d-chess-js reads it as moves and
fails on it.

A rejection is **soft** when the source could not match the move text to any move
it can generate (`no_matching_move`, `ambiguous_move`, `notation_mismatch`, or a
message the adapter does not know, `unrecognised`): that may be a rules
disagreement or a notation the source does not read, and the harness cannot tell
which. `illegal_move` and `cannot_submit` are hard.

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
| `commit` | every push and pull request | `ftxi`, `notation` | three positions per record in the driver check (below) |
| `nightly` | every night | all six | the naive searches get shorter budgets; they time out on wide positions by design |
| `release` | before any generator release or proof-channel enablement | all six | the longest budgets, and it **fails unless an engine was compared and came through at least one named variant** |

Workflows: `.github/workflows/oracle-commit.yml` and `oracle-nightly.yml` (the
latter also runs the release tier, by hand or on a `generator-v*` tag). Each runs
`verify`, builds the oracles its tier needs, runs the harness's own tests
(`selftest`), and then the gate.

Each tier run does, in order: the static checks; the oracle runs; a
self-consistency check of every report; a comparison of the fresh run with the
golden observations; a check of the ftxi driver against ftxi's stock CLI; and,
when an engine is configured, the engine differential. The last lines of the
output say whether the engine was compared and whether the proof channel may be
enabled.

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
differently, and none has been adjudicated.

```
python oracles/harness.py golden --write   # re-run every oracle over the corpus and record it
python oracles/harness.py golden --render  # only re-render DISAGREEMENTS.md
```

Each file opens with a signature: the pin, the search and its settings, the cap
where the oracle counts actions, and hashes of the adapter and driver code that
stood between the oracle and the file. If any of them changes, the file is stale
and `verify` says which. Editing a driver, an adapter, or the `base.py`,
`fingerprint.py` and `headers.py` the adapters share - even a comment in one -
therefore means `golden --write`, and a review of every verdict that moved.
(`fingerprint.py` computes the hashes the files record of what each oracle set
up.) The files hold verdicts
and, per position, the hashes and board and piece counts of what the oracle set up
(schema 2) - never the boards themselves, and never an oracle's free-text messages,
because they quote the record.

Time-outs are never pinned: a fresh run is compared with the golden values it
reproduced, and a separate coverage floor per tier fails a run in which too much
timed out to call it a check. A crash is pinned like any other outcome.

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
  checkmate versus stalemate when no action exists - so the rulebook has to settle
  it before the engine's check verdict can be gated against either.
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
- **Soft legality entries.** ftxi, Chess in 5D and the notation parser do not read
  the long-algebraic move text of one record (`cwmtt/many2`), and the notation
  parser does not read that of another (`cwmtt/100_timelines`, `a2a3`). ftxi finds
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
that answered exactly as ftxi does would pass the commit tier and then fail the
nightly tier at every entry of `DISAGREEMENTS.md` that has ftxi on one side - the
ruleset each entry sits in stays ungated until the entry is adjudicated and the
losing oracle's answer is recorded in `known-divergences.toml`.

## The engine contract

The gate reaches the engine across a process boundary and nowhere else. Set
`[engine] command` in `gate.toml` to the argv of a program - kept with the engine,
never inside `oracles/`, where it would sit beside every answer the oracles have
given (`verify` refuses such a command) - that:

- is run from the repository root with `--cap N` appended;
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
- answers **every** field of every position: `legal_actions` as `min(count, N)`,
  and `terminal` as `none`, `checkmate` or `stalemate`. An oracle may leave a
  question unanswered, and may say nothing of its state; the engine is what is
  being certified, and a null from it would be a verdict nobody compared;
- puts exactly the five fields shown in each state event and the six in each
  position event, and no others (the oracle drivers add timings of their own, so
  an oracle's output passed off as the engine's is refused);
- ends with exactly one `end` event, whose `status` is
  `ok` (every action replayed),
  `rejected` (the next action is not legal: `reason` is `illegal_move` or
  `cannot_submit`),
  `unsupported` (a ruleset the engine does not implement, said before the first
  position - allowed, but that ruleset is then not gated), or `error`.

Breaking the contract, an `error` and a time-out each fail the gate. So does every
disagreement with an oracle - a parse divergence included - unless
`known-divergences.toml` records it. An entry for a placement divergence cites the
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

1. the engine replayed each one, and none holds an unexplained divergence;
2. each one was followed as far as the engine went by at least one oracle - and
   where the engine refuses an action, by an oracle that refuses the same action as
   illegal (one that could not match the move's text confirms nothing). An oracle
   that replayed the record differently - accepting an action the engine refuses,
   or the other way round, reaching a different end, or setting up a different
   position - does not count, even where a known divergence excuses the
   difference;
3. for each of check, count and mate, the oracles give a verdict that settles it,
   and agrees with the engine, at no less than `min_checked` (75%) of the
   positions the engine reports. An answer the engine contradicts confirms
   nothing, even one on record as the oracle's error, and neither does an answer
   about another position than the engine's: after the oracle parted from the
   engine, or where its unmoved markers differ from the engine's.

The last two matter. Several variants in the corpus are read by a single oracle,
some by none, and the single oracle is sometimes cwmtt, which has no check verdict
to give: an engine must not "pass" a ruleset that nothing independent looked at.
For each gated ruleset the gate prints how many positions had an independent
answer for each observable; at how many of the engine's positions an oracle held
the engine's boards and pieces, and at how many its exact state, unmoved markers
included; and the fewest oracles that stood behind any record. Where a ruleset
falls short of the floor and answers were set aside because the oracle's markers
differed from the engine's, the reason says how many.

At the current pins this keeps "Very Small - Open" ungated: ftxi is its only
oracle with a check verdict, and ftxi's layout for it draws the king and rook of
each side without the unmoved marker, so it holds them as having moved where an
engine that follows the contract does not (`DISAGREEMENTS.md` shows the marker-only
divergence from cwmtt, which marks them). Castling is impossible in that layout,
so the two positions play alike - but the harness does not know the rules, and
does not count an answer about another position.

An engine that contradicts nothing but comes through no ruleset has **failed**:
agreeing with the oracles about nothing is not a pass.

The proof channel may be enabled only by a complete run of a tier that requires an
engine and runs every oracle search over the whole corpus (the `release` tier), in
which every oracle search reproduced at least 90% of its recorded answers - a
floor fixed in the code, not in `gate.toml`, so that no tier can list an oracle and
then starve it of time - and only for the named variants it gated, never on the
strength of custom positions.

## Known divergences

When the engine and an oracle disagree, the rulebook decides. If the oracle is
wrong, the disagreement goes into `known-divergences.toml` with the citation that
settles it and a link to the upstream report. An entry excuses exactly one
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
  cite it where the cause is notation the oracle does not read and no rule at all.
  If the oracle really does hold the move illegal, it is a rules matter after all.

Every other kind needs the rulebook. The file's header says how. The file is empty
until there is an engine to disagree with anything.

## Other commands

```
python oracles/harness.py observe --source ftxi --record cwmtt/standard   # one source, one record
python oracles/harness.py corpus             # list and verify the corpus
python oracles/harness.py corpus --unlisted  # records in pinned sources that the manifest omits
python oracles/harness.py boundary           # the boundary check alone
python oracles/harness.py provenance --write # refresh the inventory in PROVENANCE.md
python oracles/harness.py gate --tier nightly --source cwmtt --record cwmtt/standard   # a partial run
```

A gate run restricted with `--source` or `--record` is marked partial and can never
enable the proof channel.

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
position can be compared action by action.

## What this is not, yet

- **Not a passed gate.** No engine has been compared with anything.
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
- **Not adjudicated.** The oracle disagreements are listed, not settled. Settling
  them needs the rulebook pinned, which is part of writing the generator
  (spec §15.1, Phase 0).
- **Not a repetition oracle.** None of the three observables touches the 5D
  repetition rules, the most load-bearing open item in the register.
- **Not proof against a port.** The boundary check finds pasted code. An oracle's
  logic rewritten in another language, or pasted with its names changed, would
  pass it.
- **Not proof against a dishonest adapter.** The gate trusts that the `[engine]
  command` asks the engine. A shim that printed what the oracles say would pass
  everything. The gate refuses a command inside the harness, the boundary check
  refuses engine code that names a path into it, and an oracle driver's output
  passed through unchanged carries fields the engine's may not - but a shim that
  took the answers some other way and rewrote them would go unseen. That is left
  to review.
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
