# The rules engine (`t5d`)

The in-house rules engine of the 5D chess engine: the code in which **every rule
lives** (design spec §11.0, decision H22). Legality, check and game end are decided
here, exactly, never by the network - so this is the most correctness-critical code
in the project: a bug here does not misplay, it manufactures false proofs. It is
built in-house (H24) and checked against independent implementations by the
differential-testing gate (H23, H27); its notation parser is written from the
notation specification (H28).

It is C++20, with a C API (a shared library) for Python and other callers.

## Status: stages E3 and E4

Stage E1 built the parts of the engine that need no movement rules; stage E2a added
the rules mechanics and pseudo-legal move generation (§11.0 items 4-5 and the
movement rules); stage E2b adds the in-house 5dpgn reader (H28), the named variants,
the replay of a record, and the gate adapter in parse mode; stage E3 adds the
expansion - everything search and the network take from one position: moves, the
move graph, the window W(k), the travel destinations E(k), typed relations,
features - its C API, and the Phase 1 measurement tools (§15.1, §15.6: movegen
throughput, M9, and the |E| curve, M7); stage E4, after the CHECKPOINT 1 rulings,
adds check status and submission legality (H36), the legal actions and their count,
the naive and hypercuboid mate detectors (H37), and the gate adapter's full mode
(§11.0 items 1-3). Rule IDs (L*, T*, J*, 4.*, S*, Q*, V*) are those of the project's
rules reference, `docs/rules/README.md`, which the code cites throughout.

| Part | Where | Spec |
|---|---|---|
| Colours, piece kinds, pieces, squares, board sizes, half-turns | `include/t5d/types.hpp` | §11.0 |
| Timeline positions and the notation's labels, odd and even games | `include/t5d/timeline.hpp` | §12.6 (the ±0 mapping) |
| The immutable board, shared by reference | `include/t5d/board.hpp` | A4, §2.4 |
| The persistent multiverse, copy-on-write per timeline | `include/t5d/multiverse.hpp` | A4, §15.1 |
| 5DFEN board strings: reading, canonical writing; the standard starts | `include/t5d/fen.hpp` | H28 |
| The canonical fingerprint | `include/t5d/fingerprint.hpp` | H30 |
| MD5 (RFC 1321) | `include/t5d/md5.hpp` | H30 |
| The 5D Zobrist key | `include/t5d/zobrist.hpp` | H18, §11.3, I9 |
| **E2a:** the ruleset - every open rules point as a named parameter | `include/t5d/rules.hpp` | §11.3, I9 |
| **E2a:** 4D displacements and the pieces' direction tables | `include/t5d/geometry.hpp` | §11 |
| **E2a:** moves | `include/t5d/move.hpp` | §11.0 |
| **E2a:** the game state machine: activity, the present, the action, application, submission; the check ruling's hooks | `include/t5d/state.hpp` | §11.0 item 5, H22 |
| **E2a:** pseudo-legal move generation; castling's on-board attack test | `include/t5d/movegen.hpp` | §11.0 item 4 |
| **E2b:** the 5dpgn reader and the replay of a record | `include/t5d/pgn.hpp` | H28, §11.0 |
| **E2b:** the named variants (rules reference V1-V4) | `include/t5d/variants.hpp` | §12.6 |
| **E2b:** the gate adapter's protocol, parse mode | `include/t5d/gate.hpp`, `tools/t5d_gate.cpp` | H23, H30 |
| **E3:** expansion: moves as records, the move graph, W(k), E(k), src_extra(k), typed relations (provisional), board and square features | `include/t5d/expand.hpp`, `include/t5d/records.h` | §6, §10.4, B9, B11, M7, M9 |
| **E3:** geometric relations in the move generator's pass (defence, x-ray) | `include/t5d/movegen.hpp` | §6, B11 |
| The C API: states (E1); games and their expansion (E3) | `include/t5d/capi.h` | - |
| **E3:** the measurement tools, and SYNTHETIC positions by random play | `tools/t5d_measure.cpp`, `tools/t5d_bench.cpp`, `tools/synthetic.hpp` | §15.6, M7, M9 |
| **E4:** check status and submission legality; the engine's check ruling | `include/t5d/check.hpp` | §11.0 items 1-2, H36 |
| **E4:** legal actions, their count and listing, the legal moves; the naive and hypercuboid mate detectors | `include/t5d/search.hpp`, `src/search.cpp`, `src/hypercuboid.cpp` | §11.0 item 3, H37 |
| **E4:** the gate adapter's full mode and its diagnostics | `include/t5d/gate.hpp`, `tools/t5d_gate.cpp` | H23, H36, H37 |

Next: the gate run with the engine configured (parse mode, then full mode); the
expansion's pin and check relations; which moves the expansion offers (build-log
spec issue 28), then the M7/M9 re-measurement on them.

**Moves are pseudo-legal; turns are legal.** A move is pseudo-legal when it obeys
every movement rule and the action's bookkeeping; legality is judged per action, at
submission (C4): `GameState::submission_status` and `submit` apply the engine's check
ruling by default (rulebook R1, register H36; see "Stage E4") and say `Safe` or
`Unsafe`. `CheckRuling::none()` submits pseudo-legal play, untested (`Unchecked`). A
single move is legal when a legal action begins with it (`legal_moves`,
`t5d/search.hpp`); the expansion offers those behind a switch
(`ExpandOptions::legal_moves`, off by default).

## Building and testing

```
py engine/build.py --test          # Windows (python3 elsewhere)
```

configures `build/engine` (git-ignored), builds everything and runs `t5d-tests`.
Other options: `--clean`, `--config Debug`, `--werror`, and arguments for the test
runner after `--` (`py engine/build.py --test -- zobrist` runs the tests whose names
contain "zobrist"; `-- --list` lists them). The default run takes a few seconds in
Release; the long cross-checks of the two mate detectors (tests named `long_*`) run
only with `-- --long` (or a fragment that names them, `-- long_detectors`): about an
hour, most of it the brute-force reference's 4,000 positions and their children.
`tests/test_gate_exe.cpp` runs the `t5d-gate` executable itself (the build passes
its path), so building `t5d-tests` builds `t5d-gate` first.

Needs a C++20 compiler (g++ 13 or newer; on Windows the MSYS2 ucrt64 g++ on PATH),
CMake 3.20 or newer and, preferably, Ninja. `build.py` takes CMake and Ninja from
the project's virtual environment when it has them, so nothing need be installed
system-wide:

```
py -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install cmake ninja      # .venv/bin/python on Linux
```

Without Ninja it falls back to Makefiles. By hand:
`cmake -S engine -B build/engine -G Ninja && cmake --build build/engine`.

Targets (all in `build/engine`):

| Target | What |
|---|---|
| `t5d_core` | static library: the engine |
| `t5d` | shared library with the C API: `t5d.dll` on Windows, `libt5d.so` elsewhere |
| `t5d-tests` | the test runner (exit status non-zero on any failure) |
| `t5d-gate` | the gate's engine adapter: full mode and parse mode (see "The gate adapter") |
| `t5d_tools` | static library of the tools: synthetic positions, record input |
| `t5d-measure` | W(k), E(k), src_extra(k) at every position of records (see "t5d-measure") |
| `t5d-bench` | expansion throughput (see "t5d-bench") |

Release builds use -O2; every target builds with `-Wall -Wextra -Wpedantic -Wshadow
-Wconversion -Wsign-conversion`, warning-free (`--werror` makes that binding). On
Windows every binary is linked statically against the MinGW runtime, so `t5d.dll`
loads from Python without MSYS2 on PATH: it imports only KERNEL32 and the Universal
CRT.

The tests use a small in-house framework (`tests/test_framework.hpp`: `TEST`,
`CHECK`, `CHECK_EQ`, `REQUIRE`, `CHECK_THROWS_AS`, ...); there is no third-party test
library.

## Layout

```
engine/
  CMakeLists.txt     the build
  build.py           configure, build, test
  include/t5d/       the public headers (C++, and capi.h for C)
  src/               the implementation
  tests/             test_framework.hpp, test_main.cpp, one test file per part
  tools/t5d_gate.cpp the gate adapter's command line (the protocol is src/gate.cpp)
  tools/t5d_measure.cpp, t5d_bench.cpp   the measurement tools (stage E3)
  tools/synthetic.*  SYNTHETIC positions by seeded random play (tools and tests)
  tools/tool_support.*  what the tools share: reading records, ruleset options
```

## Conventions

- **Squares**: index = rank × 8 + file on every board size (a1 = 0, h8 = 63). A board
  W wide and H high uses files 0..W-1 and ranks 0..H-1, 1 ≤ W, H ≤ 8, so one 64-bit
  bitboard fits any board.
- **Time**: the half-turn index h = 2T + (0 if white is to move on the board, 1 if
  black), T the turn as displayed (0 on turn-zero boards). The standard start is at
  h = 2.
- **Timelines**: an integer position p; adjacent timelines differ by 1. Labels map to
  positions like this:

  | Label | odd game | even game |
  |---|---|---|
  | `-2` | -2 | -3 |
  | `-1` | -1 | -2 |
  | `-0` | - | -1 |
  | `0` | 0 | - |
  | `+0` | - | 0 |
  | `+1` | 1 | 1 |

  Positive labels mean the same position in both kinds of game; in an even game the
  non-positive ones shift down by one, so the ±0 pair are two adjacent timelines (the
  index mapping §12.6 asks for).
- **Pieces**: kind (P N B R Q K U D S W C Y, the notation's letters), colour, and an
  "unmoved" flag that only pawns, rooks, kings and brawns carry (the notation's
  "5DFEN Strictness" list). Royal pieces are the king and the royal queen.
- **Boards are immutable** (A4). A `BoardBuilder` assembles one; `publish()` freezes
  it behind a `std::shared_ptr<const Board>`, and from then on any number of
  timelines, states and threads share it. A board holds a mailbox, bitboards per
  colour and per kind, the unmoved bitboard, an en-passant field reserved for the move
  rules, and a coordinate-free content hash kept up to date as the board is built.
- **The multiverse is persistent**. Timelines occupy a contiguous range of positions;
  each holds its boards one per half-turn, contiguous, its creator and creation order
  (none for an initial timeline) and the board it emanated from, if known. Each
  timeline is an immutable object behind a shared pointer: copying a `Multiverse`
  copies the vector of pointers, and an edit replaces only the timeline it touches
  (copy-on-write per timeline). A timeline's boards are a persistent sequence
  (`BoardSequence`): all but the newest 1-16 sit in chunks of 16 that every version
  of the timeline shares, so the new timeline object an edit builds costs the same
  whatever the timeline's length - appending never copies its history. The board
  count, piece count, latest half-turn and Zobrist key are kept incrementally. The
  present, active timelines and turn submission are rules (`t5d/state.hpp`); the
  multiverse stores what they need (creators, creation order, the even/odd start,
  the ruleset ID, the player to move).
- **Moves** (`t5d/move.hpp`): source and destination as (timeline position,
  half-turn, square), the moving and the captured piece, flags (capture, branching,
  castle, en passant, double step, promotion) and, for castling and en passant, the
  rook's or the captured pawn's square. Displacements are (dx, dy, dL, dT) with dT in
  whole turns (two half-turns), so movement never visits a board of the other colour
  (L4). The text form, for messages and tests, is the 5dpgn export form with the piece
  letter: `(0T1)Pe2(0T1)e4`, `(-0T1)Nc3(+0T1)c3`, `(0T7)Ke1(0T7)g1`, `=Q` after a
  promotion.
- **The action in progress**: the player to move applies moves one at a time
  (`GameState::apply`). A board is the mover's to play iff it is the latest of its
  timeline and has the mover to move (L5), so a board once played is not offered
  again (T3), and nothing need record which boards were played - the boards do. The
  obligations are met when no mandatory board (active, in the present, the mover's;
  T6) is left, i.e. when the present has shifted to the opponent (T5); then
  `submit()`.

## Stage E2a: the rules mechanics

### Activity and the present (T4, T8-T13)

Initial timelines are always active (T10, T13). Under the default rules a created
timeline's activity is the count rule's: a player's n-th timeline is active iff the
opponent has created at least n-1 (T10); each creation activates the opponent's
earliest-created inactive timeline, exactly one (T11), which keeps the count rule
true at every moment - so it is also the activity of any position read from 5DFEN,
and the Zobrist key, which carries every timeline's creator, already identifies it.
The state keeps the set of inactive timelines (usually empty) and updates it at each
creation. The present is the half-turn column of the earliest playable board among
the active timelines (T4), brought up to date after every move: a branch that
creates an active timeline behind it rewinds it (T8), an inactive one never does,
and a creation that reactivates a timeline behind it rewinds it too (T12). It is
kept incrementally - the state counts the active heads in the present's column and
rescans the timelines only when the last of them is played. Whoever moves
first in a position read from 5DFEN is the owner of the earliest active playable
board (Q9): `game_from_fen` infers it when not given, which also settles the
positions E1's `multiverse_from_fen` refused.

### Move generation and application

`generate_moves` produces every pseudo-legal move from every playable board of the
mover - an empty list once the obligations are met (Q4). Sliders advance one unit
per moved axis per step over boards of the mover's colour, every intermediate
square empty, void blocking (J7, J8); knights leap over pieces and void; pawns and
brawns move forward on rank (+y White, -y Black) or timeline (-L White, +L Black),
one step or - with the right - two, and capture as 4.6 and 4.7 say; en passant only
within a board (S7); promotion on the last rank to the ruleset's set (S8); castling
on one board (S1-S5) with the developer's on-board attack test (S2,
`attacked_on_board`). A move branches iff it lands on a history board (J3; the
Quick Rules' literal reading, that any move to an earlier turn branches, is the
option `earlier_playable_jump_branches`).

`GameState::apply` makes the boards (T2): one for a move within a board, two for a
non-branching jump (J4), and for a branching jump the source's successor plus a new
timeline - White's at the next position above every timeline, Black's below (L7) -
whose only board is the destination's half-step successor, a copy with every piece,
duplicates included (J5). Several branches in one action take their indices in the
order they are made (J9). The moving piece loses its unmoved flag (both castling
pieces do); every new board has no en-passant square, except the one a double step
along the rank creates, and only when it can be taken: the piece that arrived is
still a pawn (or brawn) - not promoted - and an enemy pawn (or brawn) stands beside
it. Then activity and the present are brought up to date.

`apply` checks a move's structure against the state - the source on a playable
board of the mover's, the piece there, the destination board existing, of the
mover's colour and a destination under the ruleset (Q8, Q20), the branching flag,
the captured piece, the promotion, and everything the special moves' effects rest
on: castling's king, rook and squares, en passant's square and victim, and a double
step's right, direction and empty squares - so a malformed move throws
(`std::invalid_argument`) and changes nothing; it does not re-derive the geometry of
ordinary moves (`is_pseudo_legal` does, by generating). If memory runs out midway
through a move, the state is unspecified.

### The ruleset (`t5d/rules.hpp`)

Where the rules reference marks a point as open (Q*) or conflicting, the engine
implements the high-confidence or in-game Guide's (G) reading as the default and the
alternative as a parameter of `RuleOptions`. The parameters are the rules, so they
are folded into every exact-store key (§11.3, I9): `ruleset_id()` is 0 for the
defaults (the standard rules, shared by position and structure variants) and a
non-zero 32-bit hash of the canonical encoding otherwise; `GameState` stamps it on
its multiverse. Under the rules whose activity depends on the history of play (Q1
Consecutive, Q3 play activation, Q6 Deferred) the state's key also folds in the
inactive set, the deferred timelines and the last creator (four new key-schedule
values, appended at the end of the stream: no existing key changed).

| Parameter | Default | Alternative | Rules |
|---|---|---|---|
| `activation` | `Count`: the count rule - **the rule** since R2 closed Q1 (2026-10-01) | `Consecutive`: a player's creation right after their own previous one is inactive (a reading R2 rejected; kept for comparison) | T10, Q1, R2 |
| `initial_credit` | `None`: initial timelines count for neither side | `MinusZeroToBlack`: in an even start, Black counts as having created "-0" | T10, T13, Q2 |
| `play_activates` | false | a move by the non-creator on or onto an inactive timeline activates it | T11, Q3 |
| `moves_after_shift` | false: no move once the present has shifted (G) | moves until submission | T5, Q4 |
| `mandatory` | `InPresent`: active boards in the present (G) | `AllActive`: every active board of the mover's | T6, Q5 |
| `reactivation` | `Rewinds`: a reactivated board of the mover's behind the present must be played | `Deferred`: it obliges the mover only from their next turn (the engine's formalisation) | T12, Q6 |
| `turn_zero_reachable` | true: a T0 board is a history board, reachable by branching (L5) | a T0 board is no destination ("a board where a move has already occured") | L5, J3, Q8 |
| `sliders_cross_void` | false: void blocks (J8) | sliders pass void (never landing on it) | J8, Q18 |
| `past_self_blocks` | true: any piece blocks (J7) | a slide straight back in time on its own square passes pieces of its own kind and colour | J7, Q19 |
| `earlier_playable_jump_branches` | false: a move branches iff it lands on a history board (J3, high) | the Quick Rules' literal "moving a piece to the past causes a new timeline": a move landing at an earlier turn branches even onto another timeline's playable board, by J5's mechanism (that board is left unchanged and still to be played) | J3 (conflict), J5, Q20 |
| `inactive_history_is_destination` | true: J3 makes no exception (presumed) | a history board of an inactive timeline is no destination | J3, Q20 |
| `brawn_rank_timeline_capture` | true: the Guide's literal reading, 9 brawn captures | Perseus's 8 | 4.7, Q14 |
| `double_step_right` | `UnmovedFlag` (5DFEN's `*`) | `HomeRank`: the second rank | 4.6, Q15 |
| `timeline_double_step` | true (G: "one or two squares"), for both colours - R4 settled that the double step exists along either forward axis for both | no timeline double step (the reading R4 rejected) | 4.6, Q15, R4 |
| `double_step_over_void` | false: void blocks | a timeline double step may pass a missing board | 4.6, J8, Q15 |
| `branch_copies_keep_unmoved` | true: the copy keeps every piece (J5) | the copy's pieces lose their flags | J5, Q15, Q21 |
| `jumps_keep_unmoved` | false: any move clears the moving piece's flag | a pawn or brawn moving between boards keeps it | 4.6, Q15 |
| `en_passant_brawns` | true: brawns capture and are captured en passant (the only source) | pawns only | S7, Q16 |
| `brawns_promote` | true (medium) | brawns never promote | S8, 4.7, Q17 |
| `brawn_cross_board_promotion` | true: any arrival on the last rank promotes (S8's general statement) | a brawn arriving on its last rank of another board stays a brawn; only an arrival within its own board promotes | S8, 4.7, Q17 |
| `promotions` | the queen (the base game) | any set of pieces but pawn and brawn (a 5dpgn `Promotions` header, a client extension) | S8, Q17 |
| `castling_empty_between` | true (as in ordinary chess) | only the two destination squares must be free | S2, Q21 |
| `castling_destination_test` | false (S3) | the king's destination must not be attacked on the board either | S3, Q21 |
| `castling_rook_path_test` | false | the squares the rook crosses must not be attacked on the board | Q21 |
| `castling_any_rank` | true (S1; S5's note) | only on the castling player's back rank | S5 |
| `castling_rook_two_away` | true: a rook two squares away; the king lands on its square | the rook must be at least three squares away | S5, Q21 |
| `castling_on_optional_boards` | true (S1: "on a playable board") | only on mandatory boards | S1, Q21 |
| `family` (E2b) | `Standard`: the rules every position and structure variant and every custom position shares | `ReversedRoyalty`: read by no rule - royalty is carried by the pieces (Y royal, C common; 4.11, 4.12, S4) - it gives Standard - Reversed Royalty, a rule variant (§12.6), a ruleset ID of its own (§11.3, I9); bits 28-31 of the encoding | §12.6, §11.3 |

Not parameterised, and why:

- **Q1's third reading** (one active creation per turn): no mechanism is described
  (what of reactivation, of the second creation's count?).
- **Q3's deactivation** (an active timeline becoming inactive): no mechanism is
  described.
- **A common king castling** (S4's alternative): 5DFEN marks unmoved only pawns,
  rooks, kings and brawns ("5DFEN Strictness"), so an unmoved common king cannot be
  represented, and S4's default - only a royal king castles - is all that can hold.
- **En passant later than the board right after the double step** (Q16's first
  half): no reading describes when it would still be allowed.
- **Q20's remaining cases**, which have no described alternative: a branch forward in
  time onto a history board of a more advanced timeline branches (J3's criterion
  makes no exception for direction, and nothing says such a move would be refused);
  a slide back along a branched timeline stops at its first board, and a slide
  blocked by void passes once a board fills the gap (both consequences of J8). (Q20's
  other two cases are parameters: `earlier_playable_jump_branches` - the J3 conflict,
  whose mechanism is J5's, which does not require the destination to be a history
  board - and `inactive_history_is_destination`.)
- **Promotion of a piece already on its last rank** (custom positions only): a pawn
  or brawn standing on its last rank that moves along the timeline axis arrives on
  the last rank and promotes, like any arrival there (S8). No source raises the case.

### The check ruling's hooks (`t5d/state.hpp`)

Filled by stage E4 (see "Stage E4"); `standard_check_ruling()` is the engine's:

- `CheckRuling::royals_safe` - C4's test at submission. `submission_status()` and
  `submit()` use the engine's by default: `Safe` or `Unsafe` (refused). With
  `CheckRuling::none()` - no hook - it is `Unchecked` (the obligations are met;
  royal safety was not tested) and `submit()` proceeds: pseudo-legal play.
- `CheckRuling::in_check` and `CheckRuling::has_legal_action` - C6 and C5, read by
  `outcome()`, which answers `Unknown` without them and implements no rule itself.
- `append_moves_of(multiverse, rules, colour, timeline, out, inactive_timelines)`
  (`t5d/movegen.hpp`) - the moves of either colour from a playable board on which
  that colour is to move, whatever the turn, over a plain `Multiverse`, the real one
  or a hypothetical one. A `Multiverse` holds no activity, so the caller passes the
  inactive timelines (a state's `inactive_timelines()`) for the one option that
  reads them (`inactive_history_is_destination`). Setup is O(1), so calling it once
  per timeline is O(timelines) in all. The check code does not call it (its own scan
  stops at the first royal capture without building move lists); the tests hold the
  two equal.

### Performance notes

Direction tables are computed at compile time; generation writes into a caller's
reusable `MoveList`, iterates pieces by bitboard and looks boards up in O(1); a
slide stops at the multiverse's bounds, which the multiverse keeps (O(1)). The
functions that probe boards are compiled twice: for a ruleset that leaves the
options read per probe at their defaults (`turn_zero_reachable`,
`inactive_history_is_destination`, `earlier_playable_jump_branches`,
`brawn_cross_board_promotion`) - the standard rules - an instantiation that reads
none of them, so those options cost the standard rules nothing.

Copying a state copies one pointer per timeline: O(timelines), whatever the number
of boards. Applying a move builds its one or two boards, one new `Timeline` object
per touched timeline - O(1) in the timeline's length, thanks to the persistent board
sequence (once every 16 appends a timeline also rebuilds its list of chunk pointers,
O(length / 16)) - and brings the present up to date in O(1), with one pass over the
timelines when the last board in the present's column is played (`count_moves` and
`is_pseudo_legal` have overloads that reuse a caller's list).

## Stage E2b: the 5dpgn reader, the variants, the gate adapter

### The reader (`t5d/pgn.hpp`, H28)

Written from the notation specification's text alone - its README ("Coordinates",
"Moves", "Castling", "Jumps", "Promotions", "Inactive timeline reactivation",
"Complex scenarios", "Turns", "Headers", "Comments", "5DFEN and custom variants",
"Notes") and `fen.ebnf` - never from the parser its repository ships, which is a
parse-only oracle and was not read. Reading has two stages, as the notation intends
("there should be no need to hold the state of the game while initially parsing the
moves"):

1. `read_pgn(text)` - text to a `PgnRecord`: headers, 5DFEN board strings, and the
   actions, each a list of `PgnMove`s **as written**; no state.
2. `PgnReplay(record)` - sets up the starting position the headers declare and plays
   the actions one at a time (`play_next`, `play_all`): each written move is matched
   against the pseudo-legal moves of the board it names (`resolve`), applied, and the
   action submitted once its moves are played.

What is read:

| Construct | Forms | Notes |
|---|---|---|
| Headers | `[Name "Value"]`, any number on a line; between name and value spaces, tabs or line breaks (fen.ebnf's `space = ' ' \| '\n'`), or nothing; `\"` and `\\` in values | Names compare without regard to case (the README writes `Board`, the grammar `variant`). Read: `Board`, `Variant` (when there is no `Board`), `Size`, `Mode`, `Promotions`; every other header is kept and ignored. Two different values for a header the engine reads - compared as the engine reads them, so `7x7`/`7X7` or `Q,R`/`r q` are one value - are a fault. A fault in the headers or board strings is kept (`PgnRecord::headers_unreadable`, the first one) and reading goes on past the next `]` on its line, or from the next line; the set-up raises it only after deciding whether the game is one the engine plays |
| 5DFEN boards | `[pieces:timeline:turn:colour]` | Read by E1's 5DFEN reader (see "5DFEN") at set-up |
| Turns | `N.`, `Nw.`, `/`, `Nb.`; `1b.` first in black-first records | Colours alternate; with `check_turn_numbers` (default) the numbers count the turns. An action left empty at the end (`23. ... / 1-0`) was never played and is dropped; an empty one elsewhere is unreadable |
| Board coordinates | `(<a>T<b>)` and `(L<a> T<b>)` | Labels `0`, `-0`, `+0`, `+n`, `-n`, unsigned `n`; `-0`/`+0` only in even games, `0` only in odd ones |
| Physical moves | SAN: `[piece][file][rank][x]<square>` - `e4`, `exd5`, `Nbd7`, `R1a3`, `Qh4e1`, `Qg7g6` | Piece letters upper case for both colours: `P W K C Q Y`/`RQ` `S`/`PR` `N R B U D`; no letter: a pawn. The board prefix may be left out only while there is one timeline ("Moves") |
| Jumps | `(<a>T<b>)<piece><square>` `>` or `>>`, `x`, `(<a>T<b>)<square>` | The piece's square in full ("Omission"); the pawn letter may be left out (`(-1T19)c6>(0T19)c6`) |
| Promotions | `=X` before the marks | Required when the ruleset offers more than one piece |
| Castling | `O-O`, `O-O-O`; or the king's move, `Ke1g1`, `Kg1` | See `castling_shortcuts` |
| Marks | `+`, `#`, `*` (softmate), then `~`, then `!`/`?` (the examples' assessments, up to two) | `+`, `#`, `*` are read and never checked: a writer's mark is never a check verdict (the engine's is R1's, `t5d/check.hpp`) |
| Annotations | `(>Lx)`, `(~Tx)`, attached or after whitespace or a comment | Checked against the state (`verify_annotations`) |
| Comments | `{...}`, not nested | Anywhere between tokens |
| Results | `1-0`, `0-1`, `1/2-1/2`, `*` | Only comments may follow |

Not read (unreadable): parenthesised variations (the notation: "not yet supported"),
`N...`, `0-0` with zeros, `;` comments, NAGs, the export form `(0T1)e2(0T1)e4` (a
proposal; turns `w.`/`b.`), a header or board string after the move text. A UTF-8
byte-order mark is skipped; CR is whitespace; non-ASCII outside comments and header
values is unreadable. Messages give the line, the column and the token as written.

**The reader's readings** (`PgnOptions`; defaults first) - where the specification is
open, ambiguous or contradicts itself:

| Option | Default | Alternative | Grounds |
|---|---|---|---|
| `castling_shortcuts` | `EFileOnly`: `O-O`/`O-O-O` are `Ke1g1`/`Ke8g8` and `Ke1c1`/`Ke8c8` - the king's move from e1/e8 only - usable only when every king of the starting position stands on e1 (White) or e8 (Black) - "more than one king" read per board, so Two Timelines and Turn Zero use them, Half Reflected and Reversed Royalty do not; otherwise unreadable | `AnyKing`: castling towards file h / file a from wherever the king stands - the specification's own 2020 example game castles `O-O` from d1. `EFileOneKing`: the specification's letter, "more than one king for either player" read over the whole starting position - as `EFileOnly`, but not in Two Timelines or Turn Zero (two kings a side), where the king's move is written | S6 (high), Q26; notation "Castling" |
| `require_capture_marks` | false: a missing `x` is read (the squares determine the move, so it never makes one ambiguous); a written `x` must capture | true: SAN's letter - a capture without `x` is unreadable | "Moves", "Jumps" |
| `verify_annotations` | true: a written `~`, `(>Lx)`, `(~Tx)` must agree with the state, or the move is unreadable; one left out is never an error | false: ignored | T15 ("should match"), Q26 |
| `infer_omitted_boards` | false: a move without its board while there are several timelines is unreadable | true: matched on every playable board of the mover's, unreadable if ambiguous | "Moves" ("must be preceded") |
| `check_turn_numbers` | true | false: numbers ignored, colours still alternate | "Turns" |

`~` is checked one way: written, the move must branch, the new timeline must be
active (T10) and its first board no later than the column after the present - T8's
rewind, and Q26's open case (a branch in the column the present reaches anyway)
accepted. `(>Lx)` must name the timeline created (L7, J9); `(~Tx)` the present's turn
after the move (T4, T12).

**What fails, and how** (`RecordFault`, carried by `RecordError`):

- *Unsupported* - a game the engine does not play: `Mode` other than `5D`, a
  variant the engine does not set up, a custom position with boards over 8x8 (by
  its `Size` header, or by its board strings' own rows and squares, counted
  without the limit - with or without a `Size` header), a promotion set the rules
  refuse (a pawn or a brawn: `RuleOptions::validate`). Decided first, from the
  first value of each header that declares the game, so it holds **whatever else
  is wrong with the record** - its move text, its other headers, faults in its
  headers themselves (which the reader keeps rather than throws); only ever
  before the first position. A promotion set the rules refuse is unsupported,
  not unreadable, because its letters are read and name a ruleset the engine
  does not implement (the gate contract's "unsupported"); a `Promotions` value
  that is not piece letters cannot be read at all.
- *Unreadable* - text that cannot be read; headers or board strings that cannot
  be read or contradict each other (a named variant's `Size` other than its own -
  whatever the numbers - included); a move that fits **two or more** pseudo-legal
  moves (the reader never guesses: "Omission" lets a writer leave out a pinned
  piece, but legality is judged per action, so a pin is no per-move notion - see
  "Choices the spec leaves open"); a promotion left unwritten when
  several are offered; a `>`/`>>` or `x` that disagrees with the move (J3: `>>`
  must branch, `>` must not); an annotation that disagrees; an action by the
  player who is not to move - the first one included, refused by the replay
  after position 0, never by the set-up.
- *IllegalMove* - no pseudo-legal move fits: the board named is no playable board
  of the mover's (L5, T3: a board takes one move), no such timeline, no piece that
  can make the move (sections 3-5), or any move once the present has shifted (Q4).
- *CannotSubmit* - the action ends with the present on the mover's colour (T5).

A failed action leaves the replay where it was (`play_next` applies to a copy).

### The set-up

The game is the `Board` header's, else the `Variant` header's (spec issue 2g), else
`custom` when the record has 5DFEN boards, else Standard ("Headers"). A custom
position is `custom` ("5DFEN and custom variants") - or `Even`, `Odd` or a name
beginning `Custom` (`Custom - Even`, ...), which the gate's engine contract files
with it under "(custom positions)". It takes its board strings, whose labels say
whether the game is even or odd (`Even` and `Odd`, also as the last word of a
`Custom ...` name, must agree), and its `Size` (or the board strings' own size when
the header is missing); the player to move is the colour to move on its present
(T4, T5; Q9). A named variant takes no board strings, its `Size`, if given, must be
its own, and White moves first (V2). The set-up never reads the move text: a first
action by the other player is refused by the replay, as action 1 (Unreadable),
after position 0. A `Promotions` header (`"Q,R"` or `"Q R"`) sets the ruleset's
promotion set (S8: a client extension; the base game's `"Q"` is the standard
ruleset).

### The variants (`t5d/variants.hpp`)

Each from its row of the rules reference's V3 table, every pawn, rook, king and brawn
marked unmoved (none has moved; Q15 and Q21 decide what that allows), White to move
at T1 (V2). Names compare without regard to case or runs of spaces.

| Board header | Boards | Confidence (V3) | Notes |
|---|---|---|---|
| `Standard` | 8x8, `0` | high | |
| `Standard - Turn Zero` | 8x8, `0` at T0 (black, history) and T1 | high | L5, Q8 |
| `Standard - Two Timelines` | 8x8, `-0` and `+0` | high | L8 |
| `Standard - Half Reflected` | 8x8 | medium | Black's king d8, queen e8 |
| `Standard - Defended Pawn` | 8x8 | medium | |
| `Standard - Reversed Royalty` | 8x8 | high | royal queen Y, common king C; no castling (S4); `RuleFamily::ReversedRoyalty`, its own ruleset ID |
| `Very Small - Open` | 4x4 | medium | |
| `Very Small` | 4x4 | medium | Black's set-up since 2020-10-07 derived (Q22) |
| `Small` (also `Misc - Small`) | 5x5 | medium | the 2020 placement |
| `Focused - Just Kings` | 3x3 | medium | |
| `Focused - Just Pawns` | 5x5 | medium | back-rank double step: Q15 |
| `Focused - Just Brawns` | 5x5 | **low** | one oracle-adjacent source only |
| `Misc - Timeline Battleground` | 3 x 5x5, `-1`, `0`, `+1` | medium | timeline mapping inferred |
| `Misc - Timeline Tactician` | 2 x 4x4, `-0`, `+0` | medium | T13: White's first action must jump onto -0 |
| `Misc - Timeline Strategos` | 2 x 5x5 | medium | unicorns |
| `Misc - Timeline Formations` | 2 x 5x5 | medium | timeline mapping inferred; the guide's text says 4x4 |
| `custom` (also `Even`, `Odd`, `Custom ...`) | the record's 5DFEN boards | - | parity from the labels; a name that states one must agree |

Every other name is unsupported - among them the V3 rows not set up in this stage
(Standard - Princess, the Simple, Small - Flipped/Centered/Open, the other Focused and
the Checkmate Practice variants, Misc - Excessive, Global Warming), each one table row
away. Timeline Battleground's outer timelines are read as 5DFEN reads any non-central
timeline (created by the colour of their sign, order 1 each); the count rule treats
the two sides alike, which is Q2's default. Reversed Royalty's pawns promote to a
queen (S8); Q17's royal-queen reading is the promotion set `{Y}`.

### The gate adapter (`t5d-gate`, `t5d/gate.hpp`)

The protocol is the gate's engine contract, in the harness's manual
(`oracles/README.md`, "The engine contract"); as implemented:

```
t5d-gate [--cap N]                full mode: states, check, legal actions, terminal
t5d-gate --parse-only [--cap N]   replay pseudo-legally: states, no verdicts
```

It reads one 5dpgn record on standard input (UTF-8; read in binary, CR-LF or LF) and
writes one JSON object per line, flushed, with no other output on standard output.
`--cap N` (the gate appends it) caps the count of legal actions (1000 when not given,
the manual's cap). Human-readable notes - why a record was rejected, with the token,
line and column, or the capture that makes an action illegal - go to standard error.

**Full mode** (stage E4): for every position K (0 = the set-up, K = after K submitted
actions),

```
{"event":"state","index":K,"size":"WxH","markers":true,"boards":["<L>:<T>:<c>:<placement>",...]}
{"event":"position","index":K,"side":"white"|"black","in_check":true|false,"legal_actions":N,"terminal":"none"|"checkmate"|"stalemate"}
```

- The state event as in parse mode (below), printed before the position is searched.
- `in_check`: rules R1's check status (`t5d/check.hpp`; H36), as the harness's
  manual defines it.
- `legal_actions`: `min(count, N)`, the count being the legal actions as the manual
  defines them: an action is the set of its moves, with its timeline-creating moves
  in the order they create their timelines - two orders of two branching moves are
  two actions, even where they reach the same state (J6). See "Legal actions and
  their count" in "Stage E4".
- `terminal`: `none` if a legal action exists, else `checkmate` if in check, else
  `stalemate` (C5, C13) - never `unknown`: the count is a search run to completion,
  with no node or time budget (H37), so a mate verdict is always a proof; the
  harness's own timeout bounds the run. At least one action is sought even with
  `--cap 0`, so `terminal` is known.
- By default a **portfolio** counts: the hypercuboid detector, then the naive one,
  each on a node budget that doubles every round from 16,384; the first to complete
  answers. Each round starts afresh, so the cost is a small multiple of the faster
  detector's - on the corpus the hypercuboid answers every position in its first
  rounds. The other detector then cross-checks the count at every position (spec
  §11.0 item 3) under a node budget (50,000 nodes): where it finishes and its count
  differs, the run ends `error` ("the detectors disagree", with both counts). Where
  it does not finish it confirms nothing (H37).
- Every search has a **memory ceiling** (`SearchBudget::max_bytes`, 2 GiB by
  default; `--max-memory-mib N`). It is an estimate of what the search's stores
  hold, on the high side: the naive detector alone, stopped by it on the Half
  Reflected game below, held 0.7 GB at its peak. A search that reaches it stops; a
  count that stopped there - the portfolio's only once both detectors have - ends
  the run `error` ("the search reached its memory ceiling (2048 MiB) before it
  completed: no verdict (H37)"), never a guess.
- The replay is legal: an action is accepted when every move is pseudo-legal, the
  turn rules allow it, the present has shifted (T5) **and** no royal of the mover's is
  left capturable (C4, R1) - otherwise `rejected`, `cannot_submit`.

On this machine full mode replays the notation specification's own real game
(Shad vs PseudoAbstractMeta, Half Reflected, 45 actions, eight timelines) in 0.6 s:
every action legal, 46 positions answered. The naive detector alone
(`--detector naive`) reaches its memory ceiling at position 27 and ends `error`
there (with the ceiling lifted it once gave the same 46 answers in 25 s).

On the pinned corpus - every record of the gate's manifest, run as the gate runs
them, one process at a time - full mode takes 8.3 s in all on this machine, no
process above 50 MB at its peak; the slowest record, cwmtt's manyChecks (53
positions, up to 17 timelines, many of them in check), takes 4.1 s, its slowest
position 0.7 s. Three records need Q4's other reading (`--moves-after-shift`) to
read to the end: cwmtt's 100_timelines 5.7 s (132 MB), ftxi's futures 0.1 s, cwmtt's
niceAndrey 19.9 s for 122 positions (48 MB).

**Diagnostics** - for investigating a record by hand, never for the gate's
configuration (`gate.toml`): `--moves-after-shift` (rules Q4's other reading),
`--rule NAME=VALUE` (any `RuleOptions` field: `true`/`false`, or an enumerator such as
`activation=consecutive`, `mandatory=all_active`, `promotions=QRBN`),
`--lenient-annotations` (`~`, `(>Lx)`, `(~Tx)` not checked), `--pgn NAME=VALUE` (any
`PgnOptions` field, `castling_shortcuts=any_king` ...), `--actions K` (position K's
legal actions on standard error: their count, then one line per action, the
canonical action texts sorted - a second order of two branching moves is a line of
its own), `--detector naive|hypercuboid|portfolio` (which counts, the portfolio by
default; the other cross-checks), `--cross-check-nodes N`, `--no-cross-check`,
`--max-memory-mib N`.

**Parse mode** (`--parse-only`): for every position K,

```
{"event":"state","index":K,"size":"WxH","markers":true,"boards":["<L>:<T>:<c>:<placement>",...]}
{"event":"position","index":K,"side":"white"|"black","in_check":null,"legal_actions":null,"terminal":null}
```

- `boards`: every board, superseded and head, in the canonical order (ascending
  timeline, then half-turn); `L` the label - `0` in an odd game, `-0`/`+0` for an
  even game's central pair, signed otherwise (`+1`, `-1`); `T` the turn as displayed;
  `c` the colour to move on the board; the placement the canonical 5DFEN pieces field,
  `*` on exactly the pawns, rooks, kings and brawns that have not moved.
- `side`: the player to move. `in_check`, `legal_actions` and `terminal` are null.
- The replay is pseudo-legal: an action is accepted when every move is pseudo-legal
  and allowed by the turn rules and the present has shifted (T5); the royal-safety
  test at submission (C4) is not made (`CheckRuling::none()`), so parse mode follows a
  record past an illegal action to compare what was set up.

Then exactly one end event; every end event carries `actions_accepted`:

| Status | When | Fields |
|---|---|---|
| `ok` | every action read was replayed | `"actions_accepted":N` |
| `rejected` | `"reason":"illegal_move"`, `"cannot_submit"` or `"unreadable"` (text, an ambiguous move, a disagreeing annotation, an action by the player not to move; also a set-up that cannot be built - see below) | `"actions_accepted":N,"reason":...` |
| `unsupported` | before position 0: a game the engine does not play - mode, variant, custom boards over 8x8, a promotion set it does not implement - whatever else is wrong with the record | `"actions_accepted":0,"reason":"<what>"` |
| `error` | an internal failure (none is known), or - full mode - the two detectors disagreeing, or a count stopped by the memory ceiling (no verdict, H37) | `"actions_accepted":N,"reason":"<what>"` |

Position 0 is reported whenever the record's set-up can be built - the set-up never
reads the move text, so a first action by the wrong player is refused at action 1,
after position 0 - and then a run that reports n positions ends with
`actions_accepted` = n - 1. A run reports **no** position only when it ends
`unsupported` (the contract's case), `error`, or `rejected` with `"unreadable"` and
`"actions_accepted":0` because the set-up cannot be built: headers or board strings
that cannot be read or contradict each other (a named variant's `Size` other than its
own; a `Size` with no `Board` header and no board strings, as the specification's own
Rook Tactics I and Knight tactics III are written), a custom position without board
strings or with malformed ones. The engine contract allows it: a set-up the engine
cannot read is refused as `unreadable`, with `actions_accepted` 0, before any
position (the manual's "action 0").

Exit status: 0 for `ok`, `rejected` and `unsupported`, 1 for `error`, 2 for a bad
command line (with the usage on standard error and nothing on standard output).
`tests/test_gate_exe.cpp` runs the executable itself: `--cap` and the other options,
a bad command line, input with CR-LF line ends and a byte-order mark, and the exit
status of each end. The adapter answers from the engine alone; it never runs, loads
or reads the harness or an oracle.

### The specification's examples

`tests/test_pgn_examples.cpp` replays the four, with board counts and boards derived by
hand in its comments:

- **Rook Tactics I.** As written under "Full examples" it names no Board (so
  Standard) yet a 5x5 Size, and gives no position: unreadable at set-up, so the gate
  adapter reports no position (`rejected`, `unreadable`, see above). The "5DFEN"
  section's copy (its position, no board prefixes, no unmoved markers) replays; its
  moves plus the full record's `3. (0T3)Re5#` replay to six boards.
- **Knight tactics III.** Moves only (Q27: the puzzle's position is missing):
  unreadable at set-up as written, likewise. Replayed on a position made for the test
  that the moves fit - not the puzzle's - to nine boards and timeline +1.
- **"Actual game" (2020).** Simple - No Queens is not set up: unsupported as written.
  From V3's set-up as a custom position: under the default reading its `5. (0T5)O-O`
  from d1 is unreadable (S6, Q26); under `AnyKing` it replays 28 actions and stops at
  action 29 - `15. (1T13)Ba4+` is impossible (White's bishops on +1 stand on a3 and
  g2). A probe with a guessed correction (`Bxb4`) stops again at `16. ... (0T14)Bg2`
  (Black's only bishop left on timeline 0 is on f7): the record holds two errors, and
  it is not replayed past the first.
- **Shad vs PseudoAbstractMeta (2021, Half Reflected).** Replays to the end: 45
  actions, 85 boards, eight timelines; every `~` and `(>Lx)` agrees with the state;
  the empty action before `1-0` is dropped.

## Stage E3: expansion, and the measurement tools

**Provisional throughout.** Moves are pseudo-legal by default; stage E4's switch
`ExpandOptions::legal_moves` (off by default until build-log spec issue 28 is
settled) restricts them to the legal moves - those a legal action begins with - in
the one place the expansion's moves are chosen (`src/expand.cpp`,
`expansion_moves`). With pseudo-legal moves every |E| measured here is an upper bound;
every throughput figure is re-measured once the spec settles which moves the
expansion offers. Nothing below but that switch depends on what attacks a royal
piece (check, pins). Where E3 meets E4: a C API game submits under the engine's check
ruling (`t5d_game_set_legality` switches a game to pseudo-legal submission, as a
synthetic record's replay needs); the synthetic player submits with no ruling
(`CheckRuling::none()`: its games are pseudo-legal by design); and the tools replay
records as the gate adapter's parse mode does (`tools::parse_mode`), so a record's
positions do not depend on whether its actions were legal.

### Expansion (`t5d/expand.hpp`)

`expand(state, options, out)` computes, for the player to move, everything search and
the network take from one position (§6, §10.4, §15.6: "expansion - legal moves,
attention masks, typed edges and E"), into an `Expansion` that is reused from call to
call. Boards are indices in the canonical order - ascending timeline, then half-turn
(the fingerprint's order) - and `board_index(p, h)` maps back. Two square indices
are in use, and the names say which (`t5d/records.h`): a **square** is the engine's
rank × 8 + file; a **cell** is the network's rank × width + file
(`python/t5d/net/state.py`: "square index s = y*W + x"). They agree on 8x8 boards.
Every array made for the network (features, edges, relations) uses cells; a move
record carries both.

- **(a) moves**: `moves()` - the generator's own list (for `GameState::apply`) - and
  `move_records()`, the same moves as plain `t5d_move` records (`t5d/records.h`, 32
  bytes): source and destination (timeline, half-turn, square, cell, board index),
  the moving piece's type id, the captured piece's (en passant: the pawn taken), the
  promotion's (0 for none), and flags - capture, branching (J3), castle, en passant,
  double step, promotion (t5d::Move's bits) and **travel**: the destination board is
  not the source board. A branching move's destination is the history board it
  targets, never the board it creates.
- **(b) move_edges** (§6, B9: "which squares reach which, across all boards"): one
  edge (source board, source cell, destination board, destination cell) per distinct
  pair of (a) - promotions merge. In move order; promotions are generated one after
  another, so a pair repeats only right after itself.
- **(c) window(k)**, W(k): for every **active** timeline (T10), its head, the up-to-k
  boards of its own immediately before the head (one board per half-turn, so k
  counts half-turns) - its **band** - and its branch origin if it was created by
  branching (`Timeline::origin`; possibly in the set already, possibly on an inactive
  timeline). These are the first three tiers of §10.4.3's queue ("present band →
  heads' recent k → branch origins"); §10.4.3's budget mechanics call only the first
  two the "mandatory core" and leave the origins among the budgeted filler, so every
  count is also given against the bands alone (`Window::BandsOnly`; per board
  `band_depth()` and `board_is_origin()`; `origins()`). Inactive timelines get no
  band ("their boards remain travel destinations: they are E-material").
  `window_depth()` gives each board the least k with it in W(k) (an origin: 0;
  `kNoWindow` = INT32_MAX: none - a k that large is taken as INT32_MAX - 1, so it
  takes every band whole and still no inactive timeline), so every k is answered
  after one expansion. The spec's arithmetic holds: T active timelines at least
  k + 1 boards long, all but one branched, give |W(k)| = (k + 2)T - 1 - §15.6's "W
  at k = 8" column (59, 69, 79, 99, 119 at T = 6, 7, 8, 10, 12) and §10.4.3's scope
  column (40 / 72 / 104 / 136 / 168 at k = 0 / 4 / 8 / 12 / 16 with T = 8 and
  |E| = 25); the T - 1 term is the origins (the test
  `expand_window_takes_each_active_timelines_band_and_branch_origin` checks it). A
  position read from 5DFEN records no origins (`origins_unknown()` counts the active
  timelines created by branching whose origin is missing).
- **(d) e_boards(k)**, E(k) - §10.4.1's "out-of-window boards that are legal travel
  destinations": the boards outside W(k) on which a **travel** move of (a) lands (a
  move within a board lands on its own source board and is no travel).
  **src_extra(k)**: the boards outside W(k) that hold a move's source square (the
  head of an inactive timeline: playable, band-less). src_extra is never part of
  |E|; a board is in both only if some piece travels onto it, and **src_only(k)**
  counts the boards of src_extra(k) outside E(k) - what W(k) + E(k) misses of the
  policy pointer's source squares. `travel_destinations()` and `source_boards()`
  list them whatever the window.
- **(e) typed relations** (§6 "Typed edge biases", B11 - **provisional**, each off by
  default: `ExpandOptions::attack`, `defence`, `xray`). Rows of (source board, source
  cell, target board, target cell, relation id), the ids those of
  `python/t5d/net/state.py` (`RELATION_TYPES`: move 0, attack 1, defence 2, pin 3,
  x-ray 4, check 5, travel-threat 6). Only the subset that check detection cannot
  change, for the player to move's pieces on the boards (a)'s moves start from:
  - **attack**: source square → the square of the enemy piece a capture of (a)
    takes (en passant: the pawn taken, not the square moved to); once per pair;
  - **defence**: source square → a piece of the mover's that the source piece's
    movement geometry reaches with only that piece in the way - the capture it would
    make if that piece were an enemy (the same paths, J7/J8 and Q18/Q19, and landing
    rules, Q8/Q20; pawns and brawns by their capture geometry, Q14 included);
  - **x-ray** (sliders): the first piece along a ray, of either colour, is passed as
    if empty; the ray goes on under the same rules and stops at the next piece, of
    either colour, which is the target (recorded where a capture or defence of it
    would be, its board a destination under the ruleset);
  - **not built, reserved**: *pin* and *check*, which would read check detection
    (`t5d/check.hpp`, stage E4; R1, H36) - E4 built that, not these relations;
    *travel-threat* is named by §6 and register B11 and used by
    §11.1 as a class of forcing moves ("branching moves that create a timeline with
    an immediate threat"), but no edge - source and target squares - is defined
    anywhere. Asking for any of the three throws (the C API:
    `T5D_ERR_UNSUPPORTED`).

  Defence and x-ray come from the move generator's own geometry, in the same pass as
  the moves (`generate_moves(state, moves, relations, kinds)` in `t5d/movegen.hpp`),
  so a relation and a move never disagree about what a piece reaches.
- **(f) features**: per board (every board) - timeline position p (the engine's
  mapping, "-0" → -1), half-turn h = 2T + (black to move), head, active timeline,
  side to move (h & 1), **playable now** (the head of its timeline with the mover to
  move on it, L5, while a move may be made, Q4 - exactly the boards (a)'s moves
  start from), and the network contract's lineage, **parent** (`board_parent()`:
  the timeline's previous board, or for a timeline's first board its branch origin;
  -1 for an initial timeline or an origin not recorded). These arrays are computed
  on first access after an expansion (O(boards)); `board_info(i)` gives one board's
  entries without them. Per square, in cells: piece type id (0 empty, 1-12 = P N B R
  Q K U D S W C Y), colour (0 none, 1 white, 2 black), has-moved.
  `square_features` chooses the boards: `All` (the full-history input), `Scope`
  (W(k) + E(k) + src_extra(k): the windowed input) or `None`; `feature_boards()`
  lists them, and the feature arrays hold one block per board in that order. The
  Scope is a **board set, not a position**: it can hold a history board of a
  timeline without that timeline's head (an E board on an inactive timeline), which
  `python/t5d/net/state.py`'s `PositionState` refuses (one head per timeline) - what
  a windowed position holds is the window code's to define.

**Readings** (each a reading of the spec, for the build log):

- **"Some live piece"** (§10.4.1's E) is read as the player to move's pieces on their
  playable boards: E "comes from the move generator" at the node, and the opponent's
  destinations enter at the child ("Search re-partitions per node", §10.4.3).
  Reading it as both colours - the opponent's moves from the boards it cannot move on
  now, which `append_moves_of` can generate - is a choice of scope the spec does not
  make, and is not built.
- **E is travel destinations**: a move within a board adds nothing to E (§10.4.1:
  "travel destinations"); a band-less playable board whose pieces only move on it is
  src_extra, not E.
- **src_extra** is reported apart from |E| (the spec does not say whether scope must
  include source boards for the policy pointer); src_only says how many boards it
  would add.
- **The origins** are W's third tier, reported apart (above).
- **Has-moved** is 1 exactly on a pawn, rook, king or brawn without its unmoved
  marker (5DFEN's `*`), and 0 on every other piece and square: no rule reads the
  history of the other kinds, and 5DFEN does not record it.
- **W's present band** is the active heads (every active head is in W at every k).

**Cost.** Allocation-free once an `Expansion`'s buffers have grown (the test
`expand_allocates_nothing_in_steady_state` counts every `operator new`, the per-board
arrays included): records, edges and relations are written in place into buffers
that only grow. `expand()` does **no work per frozen board**: moves start only on
playable boards, so frozen boards are read only along a move's path, and the rest
is per timeline (bands, origins), per move (records, edges, relations, the boards'
roles - kept as bitsets, N/64 words), or per board of the scope (W, E, src_extra and
their features). Move generation's work grows with the moves - and the pseudo-legal
travel into a longer past grows with history, by the rules - not with the boards
(per move it falls). The per-board arrays (f) are O(boards) and computed only when
read; the square features of every board under `SquareFeatures::All` are
O(boards × squares) - the network's full-history input itself - decoded from each
board's mailbox (`Board::codes()`) sixteen cells at a time (the GCC/Clang vector
extension; scalar elsewhere). Under `Scope` the features are O(scope).

### The C API for games (`t5d/capi.h`)

A `t5d_game` is a state under the rules with its expansion buffers:
`t5d_game_new_from_pgn` (a 5dpgn record, set up at its first position; then
`t5d_game_replay_next`, `t5d_game_record_actions`), `t5d_game_new_from_fen`,
`t5d_game_new_variant`; `t5d_game_apply` (a move given as its source, destination
and promotion), `t5d_game_apply_expanded` (by index into the last expansion),
`t5d_game_submit`; `t5d_game_expand` (`t5d_expand_options`; NULL for k = 8, every
board's features, no relations), which fills a `t5d_expansion`: counts, and pointers
to the flat arrays above (per board: timeline, half-turn, head, active, side,
playable, window depth, band depth, origin, parent) - valid until the handle is next
changed; `t5d_game_info`, `t5d_game_state` (a `t5d_state` copy, for the fingerprint
and the boards), `t5d_game_copy`, `t5d_game_destroy`. Rulesets are passed as
`RuleOptions::encode()`'s 64-bit encoding (`RuleOptions::decode` reads it back; 0 is
standard, `T5D_RULES_MOVES_AFTER_SHIFT` is bit 3). New statuses:
`T5D_ERR_UNSUPPORTED` (a variant or a relation not built) and `T5D_ERR_ILLEGAL` (a
move or a submission the rules refuse - a submission that leaves a royal of the
mover's capturable included, C4, R1, unless `t5d_game_set_legality` set the game to
`T5D_PSEUDO_LEGAL`). The legal-moves switch is not in the C API yet
(`t5d_expand_options` is the Python binding's layout). The arrays are the
network contract's (`python/t5d/net/state.py`), so a Python binding copies them; a
square's token is board × (cells + registers) + cell.

### Synthetic positions (`tools/synthetic.hpp`)

No record in the repository reaches the 100-600 boards of §1 and §15.6, so the
benchmark plays its own: seeded random pseudo-legal play (xorshift64*, the same on
every platform), under the ruleset given. Each move is uniform among the pseudo-legal
moves - never one that takes a royal piece (4.13) - until the present has shifted
(T5); then the action is submitted. A game stuck with no move and no submission
stops. One knob steers the **active** timelines towards a target: below it, a move
branches with probability `branch_percent` when a branch would create an active
timeline; at it, no move branches unless every move does. So the classes can have
the spec's assumed T at each N, and one game can be taken at N = 100 ... 600 with T
held (the flatness series). Every record written carries a `Synthetic` header (with
its seed, settings and, if not standard, the ruleset) and replays in the engine's own
reader to the same state, branch origins included (5DFEN would lose them); every tool
output marks synthetic positions (t5d-bench files synthetic records given as input
in buckets of their own, and t5d-measure's summary leaves them out). **They are
random play: for throughput and stress only, never evidence for the |E| verdict
(M7), which needs real games.**

### t5d-measure

```
t5d-measure [--moves-after-shift] [--rules ENCODING] [--name NAME] [--summary] [RECORD ...]
```

Replays each 5dpgn record as the gate adapter's parse mode does (no RECORD, or `-`:
standard input), under the ruleset given (`--moves-after-shift` is Q4's other
reading; `--rules` takes any encoding, decimal or `0x` hex), and prints one JSON
object per line, deterministically:

- `{"event":"record","record":...,"game":...,"synthetic":...,"rules":...,"rules_encoding":"0x..","actions_in_record":N}`
- at every action start, position 0 included: `{"event":"position","record":...,
  "synthetic":...,"index":K,"side":...,"boards":N,"timelines":...,"active":...,
  "present":h,"present_turn":T,"moves":...,"move_edges":...,
  "travel_destinations":...,"sources":...,"origins":...,"origins_unknown":...,
  "w_k0":...,"e_k0":...,"src_extra_k0":...,"src_only_k0":...,"w_bands_k0":...,
  "e_bands_k0":...,"src_only_bands_k0":...}`, the same seven for k = 4, 8 and 16,
  and the position's `"fingerprint"` (H30);
- `{"event":"end","record":...,"status":"ok"|"rejected"|"unsupported"|"error",
  "positions":...,"actions_accepted":...,"actions_in_record":...,"truncated":...,
  "reason":...,"detail":...}` - how far it read, and why it stopped;
- with `--summary`, after the records, the |E| curve M7 reads, made so that a game
  counts once: positions are distinct by fingerprint and side to move (a position
  reached in several records - a standard start, a game recorded twice - is one
  position), and records are grouped into games (a record whose positions are a
  prefix of another's, or that shares its first 10 positions or more and at least
  half of the shorter record with another - two endings of one game - is that game;
  `{"event":"summary_game",...}` lists each group). Then per N bucket (1-49, 50-99,
  100-159, 160-239, 240-399, 400-599, 600+): `{"event":"summary","bucket":...,
  "positions":...,"games":...,"active_median":...,"w_k8_median":...,
  "w_bands_k8_median":...,"e_k0":{"median","p95","max","n"}` and the same for k = 4,
  8, 16 (quantiles interpolated linearly between order statistics),
  `"e_k8_game_medians"` and `"e_k4_game_medians"` (the spread of the games' own
  medians), `"positions_with_src_only_k8"`, and the synthetic positions left out.

Example, on the records of the pinned corpus (input data on the command line; the
engine itself never names anything under `oracles/`):

```
build/engine/t5d-measure --summary --moves-after-shift oracles/third_party/cwmtt/tests/5dpgn/NP.5dpgn ...
```

Several corpus records make moves after the present has shifted (Q4): under the
default reading they stop with `illegal_move`, under `--moves-after-shift` they read
to the end - the output says which ruleset was used.

### t5d-bench

```
t5d-bench [--classes N:T,...] [--no-synthetic] [--games G] [--children C]
          [--flatness T] [--flatness-boards N,...] [--flat-games G] [--flat-window P]
          [--flat-children C] [--flat-max M] [--no-flatness]
          [--seed S] [--branch-percent P] [--threads 1,2,4,...]
          [--samples S] [--sample-ms MS] [--flat-ms MS] [--no-loop] [--no-pin]
          [--legal] [--quick] [--emit DIR] [--json FILE]
          [--moves-after-shift] [--rules ENCODING] [RECORD ...]
```

Expansion throughput (§15.6, M9), and how it changes with the length of the history.
Moves are pseudo-legal but in the `--legal` configurations: every figure is
provisional.

*Positions.* Synthetic classes at §15.6's assumed (N, T) - 100:6, 160:7, 240:8,
400:10, 600:12 - `--games` games each (16 by default; a game's seed is a function of
the base seed, N, T and the game's number only, so one class can be re-measured
alone), played under the ruleset given; and every action start of any RECORD given,
grouped by N (synthetic records in buckets of their own, marked). *What is timed* is
a search-like **stream**: the one-move children of the class positions (the
positions a search expands next), `--children` per position per worker thread, each
worker's disjoint from the others' wherever a position has the moves, rebuilt on
boards of its own, and started at a different game (so that threads running at once -
SMT siblings included - expand different games). Cycling the class positions
themselves lets the branch predictor learn them; that **loop** is reported apart,
as an upper bound.

*Configurations.* `movegen` (generate_moves alone, the reference), `expand` (moves +
move edges + W(8) + E(8) + every board's features - the full-history input),
`expand+relations` (the same with attack, defence and x-ray), `expand-scope` (the
same as `expand` but the features of W(8) + E(8) + src_extra(8) only: the windowed
input), and `scope+relations` - §15.6's expansion, "legal moves, attention masks,
typed edges and E", with the windowed input. The full-history ones and
`scope+relations` also run on 1, 2, 4, 8, 12 and 24 threads. `--legal` adds
`expand-scope+legal` and `scope+relations+legal`: the same with the legal moves only
(`ExpandOptions::legal_moves`; see "Legal moves" in "Stage E4" for what it costs).

*Method.* A pool of worker threads, pinned one per physical core first, then on the
cores' second logical processors (`--no-pin`: unpinned), each with its own
`Expansion`. A measurement - (class, configuration, threads, stream or loop) - is
taken in samples of `--sample-ms`, and the samples are **interleaved**: round after
round over every measurement, after a warm-up round, so that a drift of the
machine's load spreads over all of them, and every ratio - the thread scaling, a
flatness growth - is taken between samples of the **same round**, then reported as
the median and range over the rounds. Each worker counts its expansions to a shared
steady_clock deadline and reads its own **thread CPU time** (QueryThreadCycleTime at
the measured TSC rate on Windows, `CLOCK_THREAD_CPUTIME_ID` elsewhere): the CPU
microseconds per expansion are the figure other load disturbs least (time spent
descheduled is not counted, though SMT siblings and shared caches still are). The
machine's busy fraction is read around every sample and reported. Every result feeds
a printed sink, so nothing is optimised away. The output names the compiler, the
flags, the build type, the CPU, its cores and logical processors, the thread counts,
the clock and the load. For the 240- and 600-board classes the 1-thread stream rates
are set against M9's targets (20,000/s and 10,000/s) as a range over the rounds -
never a verdict.

*Flatness* (§15.6: "flat in N at fixed |E| - ≤ 10% growth from 100 to 600
boards"). `--flat-games` games (24) with the active timelines held at `--flatness`
(8), taken at every action start whose N lies in a level's window [N, N +
`--flat-window`%) for each level of `--flatness-boards` (100, 160, 240, 400, 600);
the candidates are the one-move children of the positions along the action that
follows (so every level has positions of every depth into an action). Three series,
each timed on one thread, interleaved, the levels of a configuration back to back
(ascending in even rounds, descending in odd ones):

- **raw**: every candidate - moves, |E| and the material change with N along a game
  (random play loses material; the pseudo-legal travel into a longer past grows), so
  its growth is reported with them, and per generated move;
- **matched**: anchors at the first level whose move count and |E(8)| lie in the
  bands every level covers (the overlap of the levels' interquartile ranges, else
  inter-decile), each paired at every other level with its nearest position in
  (moves, |E(8)|, the mover's pieces) within a caliper - moves and the mover's pieces
  within 10%, |E(8)| within max(1, 20%) - or dropped (`--flat-children`, 48 per action
  start, sets how many anchors find partners); the levels' means and the pairs'
  mismatch are printed - this is the series M9's "at fixed |E|" asks about, and its growth is
  labelled "below/above +10% in every round" or "rounds straddle +10%: inconclusive";
- **no_moves**: the positions after the action's moves are played, where no move may
  be made (Q4): everything an expansion does but move generation - the history-length
  term measured directly, as nanoseconds per board (a least-squares slope over the
  levels) and as its share of a matched `scope+relations` expansion at the top level.

`--json` writes the machine-readable summary (positions with their seeds and
fingerprints, every measurement's statistics, the flatness series), `--emit` each
synthetic position's record; `--quick` is a smoke test, not a measurement.

## Stage E4: check, legality, legal actions, mate

Built after the user's CHECKPOINT 1 rulings (rulebook "Rulings" R1-R7; register
H36-H40). Rule IDs are the rulebook's (`docs/rules/README.md`).

### Check status and submission legality (`t5d/check.hpp`, H36)

Both rest on one question asked of a multiverse: can a piece of the attacker's,
standing on one of the attacker's **playable** boards (the latest board of its
timeline with the attacker to move, L5), capture a **royal** piece of the other
colour - a king or a royal queen, never a common king (4.11-4.13, C1) - by a
pseudo-legal move? A royal on any board counts, history boards included (C2): a
capture that would branch counts, even into a timeline that would be inactive (R1).
The attacker's pins do not matter: capturing a royal is all that is asked (C3).
`royal_capture_exists` answers it by the move generator's rules exactly - the same
paths (J7, J8, Q18, Q19), destinations (J2, L4, Q8, Q20) and pawn and brawn captures
(4.6, 4.7, Q14); castling and en passant never take a royal - but without building
move lists: it walks each piece's lines and stops at the first royal capture. A test
holds it equal to the generator's royal captures (`append_moves_of`) on 1500 random
multiverses under the default rules and every option the generator reads per probe.
`for_each_royal_capture` lists the captures with their lines (`capture_places`:
every place a capture rests on), which the hypercuboid detector uses.

- **Check status** (R1; C6): at the start of its turn the side to move S is in
  check iff, S passing - a null move: the board advanced unchanged to a copy with
  the opponent to move, unmoved flags kept, no en-passant square - on every
  **active** board **in the present** (`passed_timelines`: its mandatory boards under
  the default T6, whatever Q5's option says; optional boards ahead of the present and
  boards on inactive timelines are not passed), the opponent could capture one of S's
  royals from **any** of its playable boards: the passed copies and every other board
  on which it is to move, on active or inactive timelines. The passed boards are
  really there (`passed_multiverse`): a slide may cross them (J7, J8). A side with no
  royal is never in check (C13). `in_check`, `check_witness`.
- **Submission legality** (R1; C4): a separate test, on the actual turn - after the
  mover's moves, no piece on any of the opponent's playable boards (active or
  inactive timelines; a board on which the mover is still to move is not the
  opponent's) may be able to capture any of the mover's royals. No pass. A move into
  check during the turn is allowed: only the submitted position is tested.
  `royals_safe`, `submission_witness`.
- **The engine's check ruling** (`standard_check_ruling()`, `t5d/state.hpp`):
  `royals_safe` and `in_check` as above; `has_legal_action` (C5) an exhaustive search
  (below). `GameState::submission_status()` and `submit()` use it by default; a
  caller that plays pseudo-legally (the movement rules' own tests, the gate's parse
  mode) passes `CheckRuling::none()`.
- **The replay** (`PgnReplay`) submits under the engine's ruling by default, so an
  action that leaves a royal capturable is `CannotSubmit`, its message naming the
  capture ("the queen on (0T3)h5 can capture the king on (0T3)e8"); parse mode sets
  `CheckRuling::none()`. The reader still refuses a move written ambiguously: the
  notation's "Omission" lets a writer leave out a piece that a pin rules out, but a
  pin is no per-move notion here - legality is judged per action, at submission.
- **Castling** keeps its per-board attack test (S2; R6, H40): a king in check from
  another board may castle, and the turn is then legal only if every royal is safe.
  `CastlingAttacks::AnyBoard` is gone (bit 17 of the ruleset encoding is unused, so
  no ruleset ID changed).

What a scan costs: one pass over the attacker's playable boards, each piece's lines
until they meet a piece, void or the multiverse's bounds - no allocation. Check status
adds one copy of the multiverse (a pointer per timeline) and one board per passed
timeline.

### Legal actions and their count (`t5d/search.hpp`; C5)

A **legal action** is a sequence of moves by the side to move - at most one from each
board it starts on (T3), as the state machine allows them (the present and its shift;
Q4's default that no move is made once the present has shifted) - that ends in a
submittable state (T5) leaving every royal of the mover's safe (C4, R1).

**Counting.** As the harness's manual defines the gate's `legal_actions`: an action
is the **set of its moves, with its timeline-creating (branching) moves in the order
they create their timelines**. Two orders of the same moves that create no timeline
are one action; two orders of two branching moves are two actions, even where they
reach the same state (J6, J9) - say two rooks that each branch back onto a copy of an
empty board, while the creations are inactive (`search_two_creation_orders_are_two_actions_j6`:
two actions, one successor). `count_legal_actions(state, cap, detector)` counts them
and answers `min(count, cap)` (count == cap: "at least cap"). For example, two kings
a side on -0 and +0 (White's on a1, Black's on a5, 5x5) have 33 legal actions: 9
pairs of moves within the two boards, 6 jumps from one board onto the other (each
plays both, J4), and 18 in which one king moves and the other then jumps onto that
board's former head - history by then, so the jump branches (J6).

`count_legal_successors` counts something else, for search: the **distinct legal
successor states** - the positions after submission, compared in full (the same
boards in the same places, the timelines and their creators, the player to move,
activity - and, where activity is not a function of the boards, the last creator under
Q1's consecutive reading, a deferral under Q6's) through the Zobrist key, with an
exact comparison whenever two keys match (`same_position`). The board a timeline
branched from (`Timeline::origin`) is not compared: no rule reads it, and neither the
key nor the fingerprint holds it (`GameState`'s `==` does).

`list_legal_actions` lists the actions, one per action as counted, sorted by
`action_text`: the moves in the 5dpgn export form, first those that create no
timeline (by ascending source timeline), then the branching ones in the order they
were made (which numbers their timelines). That text names the action; it is not
always an order in which the moves can be played (Q4 puts the move that shifts the
present last), so each listed action also carries its moves in a playable order and
its successor.

Both counts, and the searches below, also run from a state **during** an action (its
moves so far made): they then count, or look for, the ways to finish it.

### Mate detection and H37

`detect_mate(state, detector, budget)` answers at the start of a turn (C5): **none**
(`Outcome::Ongoing`: a legal action exists, and one is returned), **checkmate** (none
exists and the side to move is in check, R1), **stalemate** (none exists and it is
not in check - C13: a side with no royal is never in check), or **unknown**.
`SearchBudget` limits nodes (one per move applied and one per finished action tested
for royal safety), wall-clock time and memory: the positions held, and a **memory
ceiling** every search has - `max_bytes`, 2 GiB by default even under
`SearchBudget::unlimited()`, by an estimate of a held position's cost (boards and
history are shared between positions, so the estimate does not grow with the
history; measured, it is about three times what is really held). **H37: a checkmate
or stalemate verdict is a proof** - it is given only by a search that examined every
action and terminated (`stats.exhaustive`); a search a budget stopped answers
`Unknown`, never checkmate or stalemate (`stats.stopped_by_budget`;
`stats.stopped_by_memory` when the ceiling stopped it). The searches are
deterministic, so a search that needs K nodes completes with a budget of K and answers
`Unknown` with any budget below K - the tests run every budget from 0 to K on mates,
stalemates and positions with legal actions, and check the property on random play.
`standard_check_ruling()`'s `has_legal_action` is a search with no node or time
budget; at the memory ceiling it answers `Unknown`, and so does `outcome()`.

**The naive detector** searches move sequences depth-first, as the state machine
generates them: every pseudo-legal move of the mover, applied to a copy, recursively;
a state whose obligations are met is a finished action, legal if its royals are safe.
Its pruning is a transposition table of every state reached (two orders of the same
moves reach one state, which is searched once). Counting actions, an entry is the
state together with the action so far - its moves as a set, its branching moves in
order - since two different sets of moves can reach one state (two identical pieces
exchanging destinations) and are two actions. It is fast where legal actions abound
and exponential in the number of boards where they are scarce: near mate it examines
every combination of moves; its table is what reaches the memory ceiling first.

### Legal moves (build-log spec issue 28)

R1 judges legality per action ("moving into check during a turn is allowed"), so a
single move is legal when a legal action begins with it. `legal_moves(state)` gives
them: the pseudo-legal moves after which the action can still be finished legally
(`can_finish_legally`) - at a turn's start, or during an action. One search from the
state first: if the action cannot be finished at all, no move is legal. Otherwise
each legal finish found is replayed with each of its moves first; a move that leads a
valid order needs no search of its own, and every other move gets one (to its first
legal finish, by the hypercuboid detector). A move whose search reaches the memory
ceiling is kept: it is not proven illegal. The test against the brute-force
reference compares them with every move's own enumeration.

The expansion offers them behind `ExpandOptions::legal_moves` (off by default, until
the spec settles what the expansion offers). What it costs, on t5d-bench's synthetic
classes (random play; each class's positions and four of their one-move children,
20 positions per class, one thread, this machine):

| Class (N boards, T active) | `expand-scope`, median | `+legal`, median | `+legal`, mean | `+legal`, worst | moves kept |
|---|---|---|---|---|---|
| 100, 6 | 0.012 ms | 11 ms | 19 ms | 94 ms | 56% |
| 160, 7 | 0.019 ms | 50 ms | 53 ms | 149 ms | 67% |
| 240, 8 | 0.028 ms | 157 ms | 175 ms | 0.8 s | 29% |
| 400, 10 | 0.036 ms | 218 ms | 15.7 s | 247 s | 27% |
| 600, 12 | 0.056 ms | 0.8 ms | 36.3 s | 522 s | 7% |

The filter is a search per move: tens to hundreds of milliseconds where
`expand-scope` takes hundredths of one - and a heavy tail: in the larger classes a few
positions (random play leaves kings in check on many boards at once, with a thousand
moves or more) need minutes, the searches of their moves proving one by one that the
action cannot be finished. Where no legal action exists at all, the first search
settles it for every move (the 600-board class's median). It is not fit to be on by
default as it stands; what the expansion should offer is the spec's to say first.

### The hypercuboid detector: design

The idea - a turn as one *semimove* per timeline, "play nothing here" among them,
searched as a product space - is the design spec's (§11.0 item 3), which credits it to
penteract's cwmtt and its adaptation in ftxi's 5dchess_engine. This design was worked
out from the spec's description and the rulebook alone; neither oracle's source or
documentation was read for it (AGENTS.md: the oracles are what it is compared with).

**The space.** At the turn's start S0 the *axes* are the mover's playable boards
A1..Ak (the timelines whose latest board has the mover to move). Every move of a
legal action starts on one of them: the boards a move creates are the opponent's (T2,
T7). Each axis has a list of *values* - its share of the turn:

- `Pass` - the board is not played;
- `Physical(m)` - a move within the board;
- `Depart(m)` and `Arrive(m)` - the two halves of a jump onto another axis's board,
  which plays both (J4); a point is *consistent* when every `Depart` is matched by the
  same move's `Arrive` on the other axis and the other way round;
- `Branch(m)` - a jump onto a history board (J3, J5), which plays only its source.
  Among them, J6's: a jump onto another axis's board *after* that board was played
  (its former latest board is history by then) - the same squares as the
  non-branching jump, branching, and allowed only if that axis is not `Pass` and its
  move comes first.

A *point* - one value per axis - with an order for its branching moves (which numbers
the timelines they create, J9) describes at most one action; every legal action is
such a point. The candidate moves come from generating every axis's moves at S0 with
the two activity-dependent restrictions lifted (castling only on mandatory boards,
Q21; no landing on an inactive timeline's history, Q20), plus J6's branching twins of
the jumps between axes: mid-action no other move can appear - the mover's moves visit
only boards of the mover's colour, and an action changes those only by playing them
(their contents unchanged). Whether a point's moves really are pseudo-legal, in an
order, is then decided by the state machine itself (below). S0 may also be a state
during an action (a search for the ways to finish it): the axes are then the boards
still to play.

**A point's verdict.** For a point x: (1) consistency; (2) J6's order; (3) its
*realisation*: the orders in which its moves can be played, each ending with the
obligations met (T5) - one final state per order of its branching moves, found by
replaying through `GameState` (each move checked by `is_pseudo_legal` in the state it
is played in); (4) royal safety (C4) of each final state. A safe final state is a
legal action - one per order of the branching moves (the count's definition).

**Realisation**, by the rules in force - three regimes:

- *Ordered*: Q4's default (no move once the present has shifted), activity changed
  by creations alone and deterministically in their number - the count rule or Q1's
  consecutive reading, `Rewinds`, no activation by play - including T11's
  reactivation (the j-th creation reactivates the opponent's j-th earliest-created
  inactive timeline) and T12's rewind by it; castling allowed on optional boards.
  With the creations' order fixed, whether the obligations are met depends only on
  which boards were played and how many creations were made, and with that number
  fixed it can only turn from unmet to met as boards are played. So a move that
  creates no timeline is best made as late as J6 allows: just before the first
  branch that needs its board, else after the last creation (one of them last), or
  all of them just before the last creation, which is then the last move. A
  depth-first search over the creation orders, stopping wherever a move ends the
  action with moves still to make, visits each realisable order once.
- *FreeOrder*: Q4's other reading (moves after the shift), otherwise as above: the
  moves that create no timeline first, then the branches in each creation order J6
  allows.
- *General* - Q3's activation by play, Q6's deferral, castling on mandatory boards
  only: a depth-first search over every order of the point's moves, with a table of
  the states reached (each with its creations so far, in order); between two
  creations the moves that create no timeline, castling apart, commute and the
  obligations only turn from unmet to met as boards are played, so they are made in
  ascending order but for the action's very last move. A castling may come at any
  point: under Q21's alternative it needs its board mandatory, and under Q3 with
  Q4's other reading another plain move can rewind the present off it. Exact for any
  rules, slower.

**Eliminating whole boxes.** The search keeps the unexplored part of the space as
disjoint *boxes* (a box: one subset of values per axis; the start: everything). It
takes a box, picks a point in it, judges the point, and removes from the box a
*region* - itself a box - that contains the point and, provably, no legal action; what
is left of the box is split into at most one box per constrained axis
(B minus R = the union over the constrained axes j of: R's subsets on the axes before
j, B's subset minus R's on j, B's subsets elsewhere). Regions:

- *inconsistency*: `Depart(m)` on axis i with anything but `Arrive(m)` on j (and the
  converse) - such a point is no action;
- *J6*: a J6 `Branch` onto axis j with `Pass` on j;
- *a J6 cycle*: J6 branches around a cycle of axes, each needing the next one's board
  played first - no order plays them; every combination of such branches around the
  same cycle fails alike;
- *a royal capture* (C4): the final state's capture C of a royal rests on a fixed set
  of places (`capture_places`: the attacker's square, every square it crosses, the
  royal's square). Places on boards that existed at S0 never change. A place on the
  board an axis's move creates (its successor, at the next half-turn) is constrained:
  that axis may take only values whose successor holds the same piece there (empty
  for a crossed square; under Q18 a crossed place may also stay void, `Pass`). Every
  point that agrees there has C too, in every order - so the box of those values is
  eliminated whole. A capture made *within* one board that a branching move created
  depends on that board alone - a copy of the move's destination with the arriving
  piece (J5) - whatever its timeline's number: it constrains the creating axis to the
  branching moves that leave the same pieces on the capture's squares. Any other
  capture that touches a created timeline is used only when the point creates exactly
  one timeline (its number and board are then fixed): the region keeps that branching
  move and allows no other. Under Q20's alternative a capture landing on a history
  board depends on activity and is not used. Of the captures found, the one that
  removes the largest share of the box is used;
- *royal captures, order by order* (outside the general regime, and not under Q20's
  alternative): where no capture's region holds for every creation order - a point
  that creates two timelines or more, its captures touching them - and every order
  in which the point is played leaves a royal capturable, each order's capture
  constrains the axes the point plays with moves that create no timeline (as above),
  and the region keeps the point's branches and passes exactly and every other axis
  on such moves (J6's targets on their kind of move): every point in it is played in
  exactly the same creation orders, with the same new boards under the same numbers,
  and has each order's capture;
- *no order realises the point* - the largest of: (i) its *skeleton class* (Ordered
  and FreeOrder; see below): the same axes on `Pass`, every other axis that plays no
  J6 target on any move that creates no timeline, every branch on the branches with
  the same J6 target and threshold class (and, under Q20's alternative, destination
  timeline); (ii) the other moves within a board (castling apart) on the axes where
  the point makes one; (iii) - the present never shifts (T5; outside the general
  regime) - a mandatory axis on `Pass` and, on every axis, no branching move that
  could rewind the present (T8, T12); under Q5's alternative, any active axis on
  `Pass`;
- otherwise the single point.

**The abstract check: whole boxes without a realisable point.** Outside the general
regime whether a point is realisable depends only on its *skeleton*: which axes it
passes; which it plays with a move that creates no timeline (and which of those J6
needs early); and of each branch, its J6 target and its *threshold class* - where its
new board lies among the axes' heads (the boards of the mover's that can count for
the present: the present's, and any a reactivation brings back). That is all of a
creation the state machine reads: the present is the earliest counting head (T4), on
the mover's colour while a counting head of the mover's precedes every counting one
of the opponent's (T5, T8); the creation's activity is its place in the order (T10),
and so is what it reactivates (T11, T12). So before a point is picked, `feasible()`
searches the skeletons the box allows, creation by creation, over the counting heads
alone - the same normal form as the realisation: the creations in each order (a
memo of the states already searched: the boards played, the number of creations, the
earliest active new board), and at each step the two ways to end - the plain moves
after the creations, one of them last (feasible iff, with every plain-capable board
played, the obligations are met, and they are still unmet without the forced ones,
or without one of them), or all before a last creation (feasible iff, of the largest
sets that keep the obligations unmet - one per unplayed counting board that keeps
them so - one leaves them met after the creation). It over-approximates: it ignores
Q20's activity test, and takes a jump's two boards as two moves within boards - which
is exact: at the end of an action, a jump that plays boards i and j can come last
exactly when one of the two separate moves can (were both of "all but i" and "all but
j" to meet the obligations while "all but i and j" does not, i's head would lie
before j's and j's before i's). So "no skeleton" proves the box holds no action, and
the box is removed whole (`stats.unrealisable_boxes`); a skeleton found picks the
point, which is then realisable unless the over-approximation spoke. The check takes
at most 16,384 steps per box; beyond that it answers "unknown" and the point is
picked as before.

Before a point is picked, the box is also *narrowed* value by value, without judging
anything: a jump's half whose other half has left the box, a J6 branch whose board can
no longer be played, and - outside the general regime - `Pass` on a mandatory board
when no value left in the box could rewind the present (under Q5's alternative,
`Pass` on any active board) are removed, until nothing changes; a box left with an
empty axis is dropped. Without a skeleton to follow, the point is picked consistent
where the box allows (a jump's halves together, a J6 branch's board played;
mandatory boards played, optional ones left).

As built, every region is checked to hold its own point (else the search throws: a
region that missed it would loop), and a final state's boards are checked against
the values that made them (likewise).

A legal point found while counting is removed alone. The search ends when no box is
left: every point was judged or lay in an eliminated region, so the search was
exhaustive - which is what a checkmate or stalemate verdict needs (H37). Each step
removes at least its point, so it terminates.

**Why it is faster near mate.** Where legal actions are scarce, most points fail on a
royal capture that rests on a few boards; one region then removes every combination of
the other boards' moves at once, where the naive search tries each. Where most points
cannot be played in any order - an opponent's inactive timelines behind the present,
brought back by the mover's creations (cwmtt's manyChecks) - the abstract check
removes them a box at a time and leads the search to the points that can. Where legal
actions abound, both find one at once.

**Checked against a brute-force reference.** `tests/test_reference.hpp` enumerates
every move sequence the state machine allows, one move at a time, with no table and
no pruning. On random tiny multiverses - 1x1 boards (every move travels), timelines
of both sides at staggered times, inactive ones among them, under random rulesets
(Q1-Q6, Q8, Q20, J3's other reading) - both detectors must give its count of actions,
its successors, its listing and its verdict, and after each first move the ways to
finish the action and the legal moves (`detectors_agree_with_the_brute_force_reference`:
400 positions and their children; `long_`: 4,000). Hand-built node bounds guard the
cases that once stalled the search (`search_the_hypercuboid_stays_small_when_every_branch_rewinds`,
`search_the_hypercuboid_finds_actions_past_reactivations`).

### The two detectors compared

Measured on this machine (Release, g++ 16.2; `build/` scratch benchmark, not a test):

| Position | naive | hypercuboid |
|---|---|---|
| A checkmate on timeline 0 with 2n further mandatory boards each holding a knight whose moves cannot help (`detectors_the_hypercuboid_stays_small_near_mate`), n = 1 (3 timelines) | 2,407 nodes | 86 nodes |
| the same, n = 2 (5 timelines) | 817,875 nodes, 0.6 s | 144 nodes, <1 ms |
| the same, n = 3 (7 timelines) | stopped by the memory ceiling after 1,007,435 nodes, 0.7 s: `unknown` | 208 nodes, <1 ms |
| 1x1 boards, timelines -3..+3, a queen on each at T4: every move a rewinding branch (`search_the_hypercuboid_stays_small_when_every_branch_rewinds`), 51 actions | 102 nodes | 102 nodes (34 million before the regimes) |
| cwmtt's manyChecks, position 42 (13 timelines, three of the opponent's inactive behind the present): 1,000 actions to the cap | 1 action, then the memory ceiling after 504,652 nodes, 1.9 s: `unknown` (with no ceiling, still 1 action after 30,000,000 nodes) | 8,541 nodes, 0.02 s |

The long cross-check (`t5d-tests --long`): random legal play from every named
variant (551 positions), from sparse starts that run into mates (806) and the same
under four alternative rulesets (Q1-Q6, Q18-Q20, J3; 927) - the two detectors gave
the same verdict wherever both proved one (2,150 positions, among them 56 checkmates
and 52 stalemates), the same count to 40 wherever both knew it (2,132) and the same
successor states wherever both were exhaustive (1,028); naive 20.3 s, hypercuboid
7.1 s in all. Against the brute-force reference: 4,000 tiny multiverses (2,313 with
legal actions, 1,931 with inactive timelines, 1,365 with an action creating two
timelines or more) and 17,207 states after a first move - every count, listing,
verdict and set of legal moves equal to the reference's; one naive search stopped by
the memory ceiling and was not compared. No disagreement. The default run plays fewer
(`detectors_agree_*`, a few seconds).
## 5DFEN

`parse_fen_board` reads one board string, `[pieces:timeline:turn:colour]`, for a
given board size; `multiverse_from_fen` builds a multiverse from a set of them;
`fen_board_string` writes the canonical form. Written from the notation
specification's text (its README, "5DFEN and custom variants", "Hashing", "5DFEN
Strictness", and its grammar `fen.ebnf`), never from another implementation's parser.

- **Accepted** variants: multi-digit blank counts (`10` is ten, one blank), `*` after a
  piece that does not track it (ignored, as the grammar allows), an unsigned timeline label
  (`3`, which the README's note on even-numbered starting boards allows).
- **Accepted by decision**: `+` before a piece letter, read as the same piece (`+K` is
  `K`). This is the project's ruling, not the specification's reading: the specification
  says the `+` prefix extends the piece alphabet, which makes `+K` a piece it never
  defines. The strict alternative - reject it as an unknown piece - is a change to
  `lex_pieces` alone. See "Choices the spec leaves open".
- **Rejected**, with a message that says what and where: missing brackets, whitespace
  anywhere (callers split documents into board strings), a field count other than
  four, rows or row widths that do not match the size, unknown letters, a misplaced
  `*` or `+`, a blank of 0, leading zeros, a colour other than `w`/`b`, a label that
  does not exist in the kind of game (`-0`/`+0` in an odd game, `0` in an even one).
  For a set of boards also: two boards in one place, a gap in a timeline, a gap
  between timelines.
- **The player to move**, which 5DFEN does not record, is the caller's when given.
  Otherwise `multiverse_from_fen` infers it only where the boards settle it whatever the
  timeline-activity rule (stage E2) says, and otherwise refuses (`std::invalid_argument`)
  rather than guess - see "Choices the spec leaves open".
- **Written** canonically: no `+`, `*` exactly on unmoved pawns, rooks, kings and
  brawns, maximal blank runs, rows from the top, signed labels except the odd game's
  `0`. Reading a canonical string and writing it back is the identity.

## The canonical fingerprint (H30)

The notation specification's full-state hash: the canonical board string of every
board - superseded and head - concatenated in ascending timeline, then turn, then
white before black, with no separators; `fingerprint` is its lower-case hex MD5,
`placement` the MD5 of the same string without `*`, plus the board and piece counts.
The Standard start gives `d574889fd9da3f2bc65249ff27249b00` and the two-timeline start
`3672761404ffcd15ae644c75401812be`, the specification's two published examples (both
are tests).

## The 5D Zobrist key (H18)

Board contents are hashed coordinate-free (XOR of per-piece-square keys, updated as a
board is built); each content hash is mixed with its (timeline, half-turn) key by
multiply-rotate rounds and an avalanche; the mixed values are combined by 64-bit
addition, which never cancels duplicated boards; and the ruleset ID, the colour to
move, the even/odd start and the board size are folded in by further multiply-rotate
rounds (§11.3, I9). Each timeline's creator is one more term of the sum.
`Multiverse::zobrist()` is incremental; `zobrist::key_from_scratch` recomputes it, and
the tests hold them equal on random sequences of edits full of duplicated boards. Every
key comes from one fixed-seed generator evaluated at compile time; the schedule is a
format, pinned by a test, and new ingredients draw from the end of the stream so that
the keys already drawn keep their values.

The board size is an ingredient because a content hash is placement only: the same
pieces on the same squares of a 5×5 and an 8×8 board have one content hash, and small
boards are structure variants under the standard rules (§12.6), so they share the
standard ruleset ID. Without the size, an exact store would serve a 5×5 puzzle's proof
for an 8×8 position.

## The C API

`include/t5d/capi.h`: opaque `t5d_state` handles; create one from the Standard start,
the two-timeline start or a list of 5DFEN board strings (with a size, an even/odd flag
and the player to move, each of which can be inferred - the player to move only where
the boards settle it); copy, destroy; read the fingerprint, the canonical string, the
Zobrist key, a summary, and every board's (timeline, half-turn, board string). Every
function that can fail returns a status (`t5d_version`, `t5d_free`,
`t5d_state_destroy` and `t5d_game_destroy` cannot fail and return none); no exception
crosses the boundary. Strings the library returns are freed with `t5d_free`. Stage E3
adds games - a state under the rules, moved, replayed, submitted and expanded (see
"The C API for games").

## Choices the spec leaves open

Each is recorded where it is made (the header comments) and listed here:

- **Colour order in the fingerprint** (notation "Hashing"; §11.0, H30): the
  specification sorts by timeline then turn and is silent on the two boards of one
  turn; white's board comes first (ascending half-turn).
- **Creators of 5DFEN timelines** (§11.0 item 5): 5DFEN records none. The central
  timelines (`0`, or `-0` and `+0`) are initial; a positive label was created by white
  and a negative one by black, the label's magnitude being the creation order - the
  notation's own numbering rule ("Coordinates"). No origins.
- **The player to move in a parsed position** (§11.0 item 5): 5DFEN does not record
  it. The caller's answer is taken when given (the 5dpgn movetext says whose turn it
  is). Otherwise `multiverse_from_fen` infers it only where activity cannot change it:
  the player to move is the colour to move on the present, the earliest head of the
  *active* timelines; initial timelines are always active, so the present is one of
  the heads no later than the earliest initial head, and if those all have the same
  colour to move, so does the present. Otherwise - for instance an inactive timeline
  whose head is earlier than the present and of the other colour - it refuses rather
  than guess. `game_from_fen` (`t5d/state.hpp`, stage E2a) settles every case: it
  computes activity and the present under the ruleset (Q9).
- **En passant and the fingerprint**: 5DFEN has no en-passant field, so the canonical
  string and the fingerprint cannot tell a board with an en-passant right from one
  without; the Zobrist key can (a board's content hash includes it). The key counts
  the square only while its board is its timeline's head
  (`zobrist::keyed_content_hash`): the right is used only by a move from the board
  the double step created (S7), and moves start only from playable boards (L5), so
  once that board is history the right has lapsed for good. A state in which a head
  board carries an en-passant square therefore reads back from its canonical string
  with an equal fingerprint and a different key; once that board is history, the keys
  agree again. The engine records the square only where it can be taken (an enemy
  pawn or brawn beside the arrival square, and no promotion), so that equal positions
  keep equal keys.
- **Unmoved flags** (Q15, Q21): cleared on any piece that moves (both castling pieces
  included), kept on every piece a branch copies (J5) - each with a parameter.
- **Ruleset IDs**: a non-default ruleset's ID is a 32-bit hash of the options'
  canonical 64-bit encoding (`RuleOptions::encode`), so two distinct option sets
  could in principle share one (probability about 2^-32 per pair).
- **What the Zobrist key covers** (§11.3, H18): the boards in their places, the ruleset
  ID and the player to move; also the even/odd start, each timeline's creator and the
  board size, because the rules read all three (the size bounds every move, and small
  boards share the standard ruleset). An en-passant square only on a head board (see
  "En passant and the fingerprint"). Not the creation order - under the notation's
  numbering, which the parser follows, it is the label's magnitude - and not the origin,
  which no rule reads (the labels and creators do not determine it: a timeline `+1`
  starting at T3 black may have branched from `0` or from `-1`).
- **`+` before a piece letter** (notation "5DFEN and custom variants"): read as the same
  piece, by instruction - the specification reserves `+` to extend the piece alphabet
  but defines no `+` piece, so its own reading would make `+K` an unknown piece. A
  ruling is needed if a game record ever uses it; until then the reading is pinned by
  `fen_accepts_the_specification_variants` and `fen_writing_normalises_the_variants`.
- **Coordinates off the board**: `Board::at` and `BoardBuilder::at` read the empty
  square for any square or (file, rank) off the board, so the move rules may step past
  an edge; `BoardBuilder::put` refuses such coordinates (`std::out_of_range`) - checked
  as coordinates, since `make_square(-1, 1)` is h1's index.
- **Timeline positions** are bounded by the labels they are written with (magnitude at
  most 2^20): `-2^20..2^20` in an odd game, `-2^20-1..2^20` in an even one, so every
  multiverse's canonical string reads back.
- **The ruleset ID** (§11.3, I9, §12.6): an opaque 32-bit number, 0 for the standard
  rules - which position and structure variants (small boards among them) share; the
  board size is a key ingredient of its own. Stage E2a assigns the IDs of rule
  variants: `RuleOptions::ruleset_id()` (above).
- **Strictness of the reader**: leading zeros and zero-square blanks are rejected
  (the grammar's `{number}` allows them; the prose's "written out in base 10" does
  not need them).
- **The 5dpgn reader's readings** (stage E2b): `O-O` outside e1/e8 games (S6 vs the
  specification's own example, Q26), and in starts with more than one king a side
  (per board by default; the specification's letter, over the whole starting
  position, is `EFileOneKing`), a missing `x`, checking `~`/`(>Lx)`/`(~Tx)` (T15,
  Q26), moves without their board, turn numbers - each a `PgnOptions` field, listed
  under "Stage E2b". Also read, with no option: `Variant` for a missing `Board`,
  header names in any case, a line break between a header's name and value
  (fen.ebnf), `"Q,R"` and `"Q R"` Promotions (build log spec issue 2), header values
  compared as read (`7x7` and `7X7` are one `Size`), a missing `Size` for `custom`
  taken from the board strings, `Even`, `Odd` and `Custom ...` as custom positions
  (the gate contract's grouping; the specification names only `custom`), assessments
  `!`/`?`, and an empty action before the result dropped.
- **Reversed Royalty's ruleset ID** (§12.6, §11.3): its royalty is carried by its
  pieces, so no rule differs, but §12.6 calls it a rule variant and §11.3 wants its
  exact-store facts kept apart: `RuleOptions::family` gives it an ID of its own. Other
  variants with fairy pieces (unicorns, brawns) share the standard ID: their positions
  name their pieces, whose rules are the same under every ruleset.

- **What the gate's `legal_actions` counts** (stage E4): the harness's manual
  defines it - an action is the set of its moves, with its branching moves in their
  order - and the engine counts exactly that (`count_legal_actions`); the distinct
  successor states, which a search may want instead, are a count of their own
  (`count_legal_successors`). See "Legal actions and their count".
- **A single move's legality** (stage E4; build-log spec issue 28): a move is legal
  when a legal action begins with it (`legal_moves`); the expansion offers those only
  behind `ExpandOptions::legal_moves`, off by default until the spec says which moves
  it offers.
- **Every search has a memory ceiling** (stage E4): reaching it is `unknown` (H37),
  and the gate's run ends `error` - never a verdict, and never growth without bound
  on a shared machine.
- **Check status passes the active boards in the present whatever Q5 says** (R1's
  words: "all active boards in the present"): under Q5's alternative, which makes
  boards ahead of the present mandatory too, they are still not passed.
- **Parse mode stays pseudo-legal** (stage E4): it was introduced to compare what was
  set up before the check ruling, and keeps following a record past an action that
  leaves a royal capturable; full mode refuses that action (`cannot_submit`).
- **An ambiguous move is still unreadable** (stage E4): the notation's "Omission" lets
  a writer leave out a piece that a pin rules out, but legality is judged per action,
  at submission, so a "pin" is no per-move notion; the reader does not try each
  candidate in the context of the whole action.

## The independence rule

This engine is compared, by the gate, with independent implementations of the
rules (ftxi's 5dchess_engine, penteract's cwmtt, the 5d-chess-js library of Chess in
5D). Everything it reuses from one of them makes that oracle worthless for what the
reused part covers. So (`AGENTS.md` and `CONTRIBUTING.md` state the rules in full):

- Nothing in the engine includes, links, loads, calls or names anything under the
  oracle harness - not at build time, not at run time, not by subprocess.
- No code from an oracle is copied into the engine or ported to another language -
  move generation, check, mate and stalemate detection, the turn state machine, the
  notation parser. Not even a paraphrase: the parts are written from the rulebook and
  the design spec.
- The notation parser is written from the notation specification's text (H28). Its
  repository ships parser code too; that code counts as a parse-only oracle and is
  not read.
- No code from Stockfish or Lc0 (H29): ideas from papers and the Chess Programming
  Wiki, written afresh.
- When the engine and an oracle disagree, the rulebook decides. If you are stuck on a
  rules bug, say so - do not reach for an oracle's implementation. The gate is run
  from the harness (its manual is `oracles/README.md`); the engine never runs it.
