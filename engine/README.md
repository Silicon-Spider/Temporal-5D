# The rules engine (`t5d`)

The in-house rules engine of the 5D chess engine: the code in which **every rule
lives** (design spec §11.0, decision H22). Legality, check and game end are decided
here, exactly, never by the network - so this is the most correctness-critical code
in the project: a bug here does not misplay, it manufactures false proofs. It is
built in-house (H24) and checked against independent implementations by the
differential-testing gate (H23, H27); its notation parser is written from the
notation specification (H28).

It is C++20, with a C API (a shared library) for Python and other callers.

## Status: stages E3 to E5c, E6a and E7

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
(§11.0 items 1-3); stage E5a decides a single move's legality in sound tiers - fast,
budgeted (legal / illegal / unknown) or exact - instead of a search per move; stage
E7, after the CHECKPOINT 2 rulings, makes moves after the present has shifted - until
submission - the default rules (Q4, ruled R8; register D8), the Guide-literal reading
a diagnostic, and gives the expansion and its C API the acting player and the submit
action, possible and legal (D8); stage E6a adds the main search - PUCT over a tree or,
behind a flag, a graph, both D2 arms, the in-tree solver behind the proof channel's
switch - off by rule (H41) - final selection, persistence, self-play
and match play, the step C API an external evaluator drives, and t5d-play - revised
after three adversarial reviews (MCGS identity, the solver's default, test strength,
call sequences, memory, the C ABI, validation). Rule IDs (L*, T*, J*, 4.*, S*, Q*, V*) are those of the project's
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
| **E5a:** the legal moves in tiers: exact, budgeted, fast | `include/t5d/legal.hpp`, `src/legal.cpp` | C4, R1; build-log spec issues 28, 46 |
| **E5b:** the tactical scanner: forcing actions, proofs and estimates, the solved store; experiment 6b's tool | `include/t5d/scanner.hpp`, `tools/t5d_scan6b.cpp` | §11.1-§11.3, §15.6, H1-H6, H15, H19, M8 |
| **E5c:** the scanner's in-check stage (M12's optimisation pass), Lazy SMP and the shared solved store; 6b per board-count range at the design thread budget | `include/t5d/scanner.hpp`, `tools/t5d_scan6b.cpp` | §11.3, §15.6, H17, H19, M12, M13 |
| **E7:** Q4 as ruled (R8) by default; the acting player and the submit action in the expansion and the C API | `include/t5d/rules.hpp`, `include/t5d/expand.hpp`, `include/t5d/capi.h` | §8.2, §11.0 item 2, D8 |
| **E6a:** the main search: PUCT, widening, both D2 arms, MCGS, the solver, final selection, persistence; built-in evaluators | `include/t5d/mcts.hpp`, `src/mcts.cpp`, `src/mcts_evaluators.cpp` | §8.1-8.7, §11.1, §15.4; D2, E1-E9, C11-C14, H2, H15 |
| **E6a:** self-play and match play; the turn cap | `include/t5d/selfplay.hpp`, `src/selfplay.cpp` | §8.7, §12.3; C12-C14, I11 |
| **E6a:** the search's C API: the step API for an external evaluator | `include/t5d/capi_search.h`, `src/capi_search.cpp` | §8.1, §8.3; D1, D3 |
| **E6a:** t5d-play, the engine against itself, and its 5dpgn record | `tools/t5d_play.cpp`, `tools/play_record.hpp` | §15.5 |
| **E8:** incremental royal-attack maps | `include/t5d/attacks.hpp`, `src/attacks.cpp` | §11.1 (CHECKPOINT 3), A4, M14 |
| **E8:** the proof channel: depth-limited df-pn over the semimove-decomposed turn tree; the disproof store | `include/t5d/prover.hpp`, `src/prover.cpp` | §11.1, §8.2, §11.0 item 3, H2, H15, H17-H19, H37, H39, M14 |
| **E8:** experiment 6b′'s tool | `tools/t5d_scan6bp.cpp` | §15.6 (CHECKPOINT 3), M15, M13 |
| **E9:** the prover for Guard and Hunt - prove-only, the literal class's single entry, the last-turn check filter, lines and certificates | `include/t5d/prover.hpp` | §11.4, H42, H43, H37 |
| **E9:** certificates checked against E4 alone | `include/t5d/certificate.hpp` | §11.1 (the solver paragraph), H37, H41 |
| **E9:** Guard - can the opponent force mate against the chosen action? | `include/t5d/guard.hpp` | §11.4, §8.6, H42 |
| **E9:** Hunt - do we have a forced mate? df-pn, prove-only | `include/t5d/hunt.hpp` | §11.4, §12.2, H43, H41 |
| **E9:** the superseded-board asset map | `include/t5d/assets.hpp` | §11.2 rung 4, §11.4, H10 |
| **E9:** Hunt's ground truth; experiment 6b″'s tool | `tools/t5d_hunt_truth.cpp`, `testdata/hunt-truth/`, `tools/t5d_scan6bpp.cpp` | §15.6 (CHECKPOINT 4), M17 |

Next: the gate run with the engine configured in full mode - Q4's ruling (R8) is
the default, and with it every one of the triage's 115 Q4 findings agrees with the
oracles; the proof channel's sign-off (H41: the in-tree solver's propagation stays off
until the gate passes, the search's soundness fixes land with tests, and the channel
is signed off, its proofs cross-checked against df-pn and oracle mate verdicts);
the network's stage N5 (the acting-player input and the submit logit the search already
takes); the expansion's pin and check relations; the spec issues the search's
placeholders raise (build-log spec issues 28, 46, and the E6a issues listed under
"Choices the spec leaves open"); and wiring verification v3 (E9) into the search - Guard's re-search from §8.6's
reserve, Hunt's gated proofs and (behind their flag) priors with §12.2's exclusions,
the asset map's boosts.

**Moves are pseudo-legal; turns are legal.** A move is pseudo-legal when it obeys
every movement rule and the action's bookkeeping; legality is judged per action, at
submission (C4): `GameState::submission_status` and `submit` apply the engine's check
ruling by default (rulebook R1, register H36; see "Stage E4") and say `Safe` or
`Unsafe`. `CheckRuling::none()` submits pseudo-legal play, untested (`Unchecked`). A
single move is legal when a legal action begins with it (`legal_moves`,
`t5d/search.hpp`; `classify_moves`, `t5d/legal.hpp`); the expansion offers those
behind a switch (`ExpandOptions::legal_moves`, off by default), exact or under a
budget.

## Building and testing

```
py engine/build.py --test          # Windows (python3 elsewhere)
```

configures `build/engine` (git-ignored), builds everything and runs `t5d-tests`.
Other options: `--clean`, `--config Debug`, `--werror`, and arguments for the test
runner after `--` (`py engine/build.py --test -- zobrist` runs the tests whose names
contain "zobrist"; `-- --list` lists them). The default run takes a few seconds in
Release; the long cross-checks of the two mate detectors (tests named `long_*`) run
only with `-- --long` (or a fragment that names them, `-- long_detectors`): about
four minutes, most of it the two detectors' cross-checks and the brute-force
reference's 4,000 positions and their children.
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
| `t5d-scan6b` | experiment 6b: the scanner's horizon per node budget (see "Stage E5b") |
| `t5d-play` | the engine plays itself and prints the 5dpgn record (see "Stage E6a") |

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
  `submit()` - or, by default (Q4, ruled R8), first any further moves on the mover's
  remaining playable boards: the turn ends only at submission. (Under Q4's
  Guide-literal reading, `moves_after_shift = false`, no move may follow the shift.)

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
mover - after the obligations are met too, until submission (Q4, ruled R8); under
Q4's Guide-literal reading, an empty list once they are met. Sliders advance one unit
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
| `initial_credit` | `None`: initial timelines count as created by neither player - always active, never counted (ruled R10, register H46, provisional) | `MinusZeroToBlack`: in an even start, Black counts as having created "-0" (diagnosis only) | T10, T13, Q2, R10 |
| `play_activates` | false | a move by the non-creator on or onto an inactive timeline activates it | T11, Q3 |
| `moves_after_shift` | true: moves until submission - **the rule** since R8 ruled Q4 (2026-10-02; register D8) | false: Q4's Guide-literal reading, no move once the present has shifted - a diagnostic (`--no-moves-after-shift`). Encoding bit 3 marks it: the bit meant the opposite before stage E7 | T5, Q4, R8 |
| `mandatory` | `InPresent`: active boards in the present (G) - settled by R9 (Q5) | `AllActive`: every active board of the mover's (the reading R9 ranks below the Guide; kept for comparison) | T6, Q5, R9 |
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
| Annotations | `(>Lx)`, `(~Tx)`, attached or after whitespace or a comment | Checked against the state (`verify_annotations`); a disagreement is a warning by default (`annotation_mismatch_is_error`) |
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
| `verify_annotations` | true: a written `~`, `(>Lx)`, `(~Tx)` is checked against the state; one left out is never an error | false: ignored | T15 ("should match"), Q26 |
| `annotation_mismatch_is_error` | false: a token that disagrees is a warning (`PgnReplay::warnings()`; `t5d-gate` prints it on stderr) and the move is played - the notation calls these tokens "purely for better human comprehension", so a wrong one is the writer's slip, and refusing would hide the record's later positions from the gate | true: the move is unreadable (`t5d-gate --strict-annotations`) | T15, Q26 |
| `infer_omitted_boards` | false: a move without its board while there are several timelines is unreadable | true: matched on every playable board of the mover's, unreadable if ambiguous | "Moves" ("must be preceded") |
| `check_turn_numbers` | true | false: numbers ignored, colours still alternate | "Turns" |
| `turn_number_mismatch_is_error` | false: a number that does not count the turns is a warning (`PgnRecord::warnings`, passed on to `PgnReplay::warnings()`; `t5d-gate` prints it) and reading goes on - a missing turn still shows, as the next action's moves are checked like any other | true: unreadable from there | "Turns" |

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
  can make the move (sections 3-5), or - under Q4's Guide-literal reading only - any
  move once the present has shifted (by default, R8, moves are read until submission).
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
them, one process at a time - full mode takes 36 s in all on this machine under the
default rules (stage E7, R8; the machine shared with other runs), every record
under 15 s and no position above 2 s, no process above 131 MB at its peak: cwmtt's
niceAndrey 14.3 s for 122 positions (slowest position 0.38 s), 100_timelines 4.9 s
(1.9 s; 131 MB), wide 4.6 s, manyChecks 4.4 s (1.2 s; 101 MB), every other record
under 2 s. The ten records that made moves after the present had shifted, and
stopped there under the Guide-literal reading, now read on: nine to their end, and
cwmtt's futures to action 17, whose `x` marks a capture the move does not make
(`unreadable`). The output is byte for byte what the stage-E4 engine printed under
`--moves-after-shift` on all 41 records, and `--no-moves-after-shift` reproduces its
default output (8.3 s in all at stage E4) on all 41.

**Diagnostics** - for investigating a record by hand, never for the gate's
configuration (`gate.toml`): `--no-moves-after-shift` (rules Q4's Guide-literal
reading, no move once the present has shifted - the same as `--rule
moves_after_shift=false`; `--moves-after-shift`, the default since R8, is accepted
and ignored), `--rule NAME=VALUE` (any `RuleOptions` field: `true`/`false`, or an enumerator such as
`activation=consecutive`, `mandatory=all_active`, `promotions=QRBN`),
`--lenient-annotations` (`~`, `(>Lx)`, `(~Tx)` not checked), `--strict-annotations` (a disagreeing one is unreadable), `--pgn NAME=VALUE` (any
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
expansion offers. Nothing below but that switch and (g)'s submit legality (stage
E7: the submission test, C4) depends on what attacks a royal piece (check, pins). Where E3 meets E4: a C API game submits under the engine's check
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
  move on it, L5, while a move may be made - always by default, R8; before the
  shift only under Q4's Guide-literal reading - exactly the boards (a)'s moves
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
- **(g) the action** (stage E7; register D8, rulebook R8: turns have variable
  length, so submitting is an action) - global, not per board, and changing none of
  (f): the **acting player** (`acting_player()`; the C API's `acting_player`) - the
  side whose action this is, the player to move, who moves until they submit; once
  the present has shifted inside the action it differs from the colour to move on
  the present (`present() & 1`), which is why the network needs it as an input of its
  own (BUILD_LOG issue 48): the boards' side-to-move flags cannot say whether the
  player has already submitted; and the **submit action** - `submit_possible()`, the
  action's obligations are met (T5), so under the ply split submitting is an action
  beside the moves of (a), with its own policy logit; and `submit_legal()`, possible
  and no royal of the acting player's capturable (C4, R1) - the submission test
  itself, exact, one scan of the opponent's playable boards and no allocation, run
  only once the obligations are met (never at a turn's start). Under whole-turn
  generation submitting ends every enumerated action, and nothing more is needed.

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
changed - and, appended at its end by stage E7 (every older field keeps its offset),
(g)'s three `int32_t`: `acting_player` (0 white, 1 black), `submit_possible` and
`submit_legal` (1 or 0); `t5d_game_info`, `t5d_game_state` (a `t5d_state` copy, for
the fingerprint and the boards), `t5d_game_copy`, `t5d_game_destroy`. Rulesets are
passed as `RuleOptions::encode()`'s 64-bit encoding (`RuleOptions::decode` reads it
back): 0 is the standard rules - moves after the shift until submission among them
(R8) - and `T5D_RULES_NO_MOVES_AFTER_SHIFT`, bit 3, Q4's Guide-literal reading. Until
stage E7 bit 3 meant the opposite and was named `T5D_RULES_MOVES_AFTER_SHIFT`; that
name is now 0 (the default), so a caller that asked for moves after the shift still
gets them, but a caller that wrote the value 8 itself now gets the Guide-literal
reading. New statuses:
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
t5d-measure [--no-moves-after-shift] [--rules ENCODING] [--name NAME] [--summary] [--legal] [RECORD ...]
```

Replays each 5dpgn record as the gate adapter's parse mode does (no RECORD, or `-`:
standard input), under the ruleset given (the engine's by default - moves after the
shift until submission, R8; `--no-moves-after-shift` is Q4's Guide-literal reading, a
diagnostic, and `--moves-after-shift` is accepted and ignored; `--rules` takes any
encoding, decimal or `0x` hex), and prints one JSON
object per line, deterministically:

- `{"event":"record","record":...,"game":...,"synthetic":...,"rules":...,"rules_encoding":"0x..","moves":"pseudo-legal"|"legal","actions_in_record":N}`
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
build/engine/t5d-measure --summary oracles/third_party/cwmtt/tests/5dpgn/NP.5dpgn ...
```

Several corpus records make moves after the present has shifted (Q4): by default
(R8) they read to the end; under `--no-moves-after-shift` they stop with
`illegal_move` - the output says which ruleset was used.

Moves are pseudo-legal by default, so every |E| is an upper bound. `--legal` expands
with the legal moves only (`ExpandOptions::legal_moves`, exact - no move left
unknown): |E(k)| then counts the destinations of moves a legal action can begin
with, M7's tighter bound. On the 41 corpus records it takes 10 s in all.

### t5d-bench

```
t5d-bench [--classes N:T,...] [--no-synthetic] [--games G] [--children C]
          [--flatness T] [--flatness-boards N,...] [--flat-games G] [--flat-window P]
          [--flat-children C] [--flat-max M] [--no-flatness]
          [--seed S] [--branch-percent P] [--threads 1,2,4,...]
          [--samples S] [--sample-ms MS] [--flat-ms MS] [--no-loop] [--no-pin]
          [--legal] [--legal-budget C,N] [--legal-profile] [--quick] [--emit DIR] [--json FILE]
          [--no-moves-after-shift] [--rules ENCODING] [RECORD ...]
```

Expansion throughput (§15.6, M9), and how it changes with the length of the history.
Moves are pseudo-legal but in the `--legal` configurations and `--legal-profile`:
every figure is provisional.

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
(`ExpandOptions::legal_moves`) under `--legal-budget C,N` - C checks and N search
nodes per expansion, 64,2000 by default (`LegalityOptions::budgeted`; a move left
unknown is kept). Exact classification has a tail of minutes, which would stall the
interleaved rounds; it is profiled by `--legal-profile` instead.

*Legal profile* (`--legal-profile`, in place of the throughput): every class position
and its worker-0 one-move children expanded one at a time on one thread with the
`scope+relations` options - pseudo-legal, then with the legal moves exact, under
`--legal-budget`, and fast (0,0) - each timed alone. Per class: the median, p95 and
worst microseconds per expansion, the moves kept and left unknown, and the share of
the moves each tier settled (see "Stage E5a").

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
  be made - played under Q4's Guide-literal reading, since by default (R8) the mover
  may go on after the shift: everything an expansion does but move generation - the history-length
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
by default, R8, moves after the shift until submission - under Q4's Guide-literal
reading none) - that ends in a submittable state (T5) leaving every royal of the
mover's safe (C4, R1). By default, then, an action that can be submitted and an
action that goes on from it to a further move are two actions, each counted.

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
always an order in which the moves can be played (under Q4's Guide-literal reading
the move that shifts the present comes last), so each listed action also carries its
moves in a playable order and its successor.

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
(`can_finish_legally`) - at a turn's start, or during an action. Stage E4 gave each
move a search of its own (to its first legal finish, by the hypercuboid detector):
tens to hundreds of milliseconds per expansion where pseudo-legal expansion takes
hundredths of one, and minutes in the tail. Stage E5a decides them in tiers, exactly
(see "Stage E5a: the legal moves in tiers"); `legal_moves` and `keep_legal_moves`
use it. A move whose search reaches the memory ceiling is kept: it is not proven
illegal.

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

- *Ordered*: Q4's Guide-literal reading (no move once the present has shifted - the
  default until stage E7, now a diagnostic), activity changed
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
- *FreeOrder*: Q4's ruled reading (R8: moves after the shift, until submission) -
  **the default rules' regime** since stage E7 - otherwise as above: the moves that
  create no timeline first, then the branches in each creation order J6 allows. It
  is no slower than Ordered: its realisation is one replay per creation order, and
  its abstract check ends with every plain move made after the creations. (The
  corpus records that both readings read in full take about the same time under
  either - manyChecks 4.4 s against 4.2 s, aivsai 1.6 s against 1.5 s, silly 0.8 s
  against 0.7 s; the default only has more actions to count.)
- *General* - Q3's activation by play, Q6's deferral, castling on mandatory boards
  only: a depth-first search over every order of the point's moves, with a table of
  the states reached (each with its creations so far, in order); between two
  creations the moves that create no timeline, castling apart, commute and the
  obligations only turn from unmet to met as boards are played, so they are made in
  ascending order but for the action's very last move. A castling may come at any
  point: under Q21's alternative it needs its board mandatory, and under Q3 with
  Q4's ruled reading (R8) another plain move can rewind the present off it. Exact for
  any rules, slower.

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

Measured on this machine (Release, g++ 16.2; `build/` scratch benchmark, not a test).
Since stage E5a a hypercuboid node also counts each step of a box's abstract check
(so that a node budget bounds its time), which raised its figures below:

| Position | naive | hypercuboid |
|---|---|---|
| A checkmate on timeline 0 with 2n further mandatory boards each holding a knight whose moves cannot help (`detectors_the_hypercuboid_stays_small_near_mate`), n = 1 (3 timelines) | 2,407 nodes | 126 nodes |
| the same, n = 2 (5 timelines) | 817,875 nodes, 0.6 s | 184 nodes, <1 ms |
| the same, n = 3 (7 timelines) | stopped by the memory ceiling after 1,007,435 nodes, 0.7 s: `unknown` | 248 nodes, <1 ms |
| 1x1 boards, timelines -3..+3, a queen on each at T4: every move a rewinding branch, under Q4's Guide-literal reading (`search_the_hypercuboid_stays_small_when_every_branch_rewinds`), 51 actions | 102 nodes | 510 nodes (34 million before the regimes) |
| cwmtt's manyChecks, position 42 (13 timelines, three of the opponent's inactive behind the present): 1,000 actions to the cap | 1 action, then the memory ceiling after 504,652 nodes, 1.9 s: `unknown` (with no ceiling, still 1 action after 30,000,000 nodes) | 12,445 nodes, 0.02 s |

The long cross-check (`t5d-tests --long`, at stage E5a): random legal play from every
named variant (551 positions), from sparse starts that run into mates (815) and the
same under four alternative rulesets (Q1-Q6, Q18-Q20, J3; 927) - the two detectors
gave the same verdict wherever both proved one (2,153 positions, among them 55
checkmates and 50 stalemates), the same count to 40 wherever both knew it (2,134) and
the same successor states wherever both were exhaustive (1,019); naive 27.8 s,
hypercuboid 8.4 s in all. Against the brute-force reference: 4,000 tiny multiverses
(2,323 with legal actions, 1,957 with inactive timelines, 1,361 with an action
creating two timelines or more; a third of them on 2x1 boards since E5a) and 23,321
states after a first move - every count, listing, verdict and set of legal moves
equal to the reference's; one naive search stopped by the memory ceiling and was not
compared. No disagreement. The default run plays fewer (`detectors_agree_*`, a few
seconds).

## Stage E5a: the legal moves in tiers (`t5d/legal.hpp`)

A move is **legal** when the action under way can still be finished legally after it
(C4, R1: legality is judged per action; build-log spec issue 28). A search per move
answers that exactly but slowly. `classify_moves(state, moves, out, budget, stats)`
answers it in tiers and calls each move `Legal`, `Illegal` or `Unknown`. Every tier is
sound. It calls a move legal only with a legal finish in hand or an argument that one
exists, and illegal only with a proof. Whatever no tier settles goes to a search. The
state may be a turn's start or a state during an action.

0. **A standing capture.** A royal capture that already stands - from a board on
   which the opponent is to move, onto a royal on a board of the opponent's colour -
   lasts to the end of the action:
   - the mover moves only on boards of its own colour, so it can neither take the
     attacker nor move the royal;
   - boards never change (a move makes a new one), so nothing the capture rests on
     changes either.

   No move is then legal. Under Q18's alternative (`sliders_cross_void`), a later
   move could put a board on a void the capture crosses, so only captures that cross
   no void count as standing.
1. **The stripped multiverse.** Give every playable board of the mover's a successor
   that keeps the board's opponent pieces and the mover's royals, and none of the
   mover's other pieces. Call a move **quiet** when it creates no timeline and moves
   no royal. A quiet action ends in a position whose royal captures are among the
   stripped multiverse's, for three reasons:
   - it has the same attackers or fewer, and the same royals;
   - it has more blockers;
   - where a board was not played, it has void where the stripped multiverse has a
     board, and a slide does not cross void (the tier is off under Q18's
     alternative, and when play activates timelines).

   So when the stripped multiverse holds no capture, every quiet action is safe, and
   a quiet move is legal iff the boards still to play can be played by quiet moves -
   decided per move in O(1). A capture found there is removed first, if it can be:
   - keep the mover's piece the capture crosses (a **kept blocker**); that piece's
     moves leave the tier;
   - or drop an optional board from the multiverse; its moves leave the tier.

   **Branches.** A branching move adds an outermost timeline, one board, a copy of
   its destination. One scan per destination board, on a stripped copy, decides
   every quiet branch onto it. The branch's effect on the present (T8, T10, T11,
   T12) - which boards still count, whether the new timeline is active - comes from
   a model of one creation followed by moves within boards (`PresentModel`). The
   model is used only where creations alone change activity
   (`Reactivation::Rewinds`, play not activating, inactive history a destination).

   Where a capture remains, two more arguments apply:
   - **A passed board refutes.** Take the state with board p played as it is (a
     successor of the opponent's colour, nothing moved). If that holds a capture, the
     capture stands by tier 0's lemma after any move on p that changes none of the
     squares it rests on (from, to, a castling rook's squares, an en passant victim).
     Such a move is illegal. One scan with every board passed comes first. Without
     Q18's alternative, adding boards takes no capture away, so only the boards
     that some capture there rests on need a scan of their own.
   - **Around a finish.** Take one legal finish A0, made greedily (a move within the
     board on each board to play, replayed and tested; a few tries) or found by one
     search. Each board that A0 plays within the board is stripped alone in A0's end
     position, the other boards fixed as A0 leaves them. If that holds no capture,
     every quiet move on that board, put in place of A0's move there, finishes
     legally too. The present is taken from one replay per board.
2. **Per move** (each a check of the budget). The move is illegal if a capture
   stands after it (tier 0's lemma). It is legal if any of these holds:
   - it ends a legal action;
   - the stripped multiverse is capture-free, and stays so with the move's own
     successor boards put in it (stripped the same way, its royals and kept pieces
     where they are), and the boards still to play can be played quietly. This is
     tier 1's argument for the moves tier 1 leaves out: a royal's, a kept piece's,
     a castling;
   - a finish is replayed through the state machine and tested (C4): A0's moves on
     the other boards after it, or quiet moves on the boards still to play.
3. **Search.**
   - First, one search from the state (unless a finish is already known). If the
     action has no legal finish at all, nothing is legal.
   - Then a search per remaining move, within the budget.
   - Every finish found also proves the moves that can lead it (replayed with each
     one first).

**Modes** (`LegalityOptions`):

| Mode | Options | `Unknown` | What it runs |
|---|---|---|---|
| exact | `exact()`, the default | never (except a search stopped by the memory ceiling, kept as not proven illegal) | every tier; the tail is that of mate detection: proving that no legal finish exists can take a search of the whole action |
| budgeted | `budgeted(checks, nodes)` | yes | `checks`: tier 2's per-move checks, the greedy tries, the branch scans; `nodes`: every search together (tier 3) |
| fast | `budgeted(0, 0)` | yes | tier 0, tier 1's quiet moves and the passed-board refutations: a few scans per playable board |

A node budget bounds a search's time because the hypercuboid charges each box's
abstract check by its steps (see "The two detectors compared"): a few microseconds
per node - 4.7 µs in a root search on cwmtt/manyChecks (13 timelines; 28,813 nodes
in 136 ms, and stopped at 2,000 in 4.2 ms), 3.6 µs in a 600-board synthetic class.
The cost of a check grows with the number of boards (a state copy, a replay, a
scan).

**The expansion.** `ExpandOptions::legal_moves` classifies under
`ExpandOptions::legality`:
- a move proved illegal is dropped;
- a move left unknown is kept;
- `Expansion::move_legality()` gives each kept move's status, and `legality_stats()`
  says how the moves were settled.

`legal_moves` and `keep_legal_moves` (`t5d/search.hpp`) use the exact classification.
`t5d-measure --legal` measures |E| from legal moves; `t5d-bench --legal-profile`
profiles the modes.

**Tests.**
- `detectors_agree_with_the_brute_force_reference`: every third tiny position is on
  2x1 boards (moves within a board: quiet moves, kept blockers, passed boards). Each
  state's moves are classified exact and under three budgets. Exact must equal the
  reference; a budget must never contradict it.
- `legal_moves_agree_with_a_search_per_move`: random games from the 5x5 to 8x8
  sparse starts and their one-move children; the exact classification against a
  search from every child.
- `long_` versions of both add the variants and the alternative rulesets.

| Test | Classifications | Disagreements |
|---|---|---|
| default reference run | 9,208 | 0 |
| long reference run | 93,284 | 0 |
| default per-move run | 3,103 | 0 |
| long per-move run | 19,984 (10 searches stopped by their budget, not compared) | 0 |

The tiers that settled the long per-move run's moves: standing at the start 2,304,
stripped quiet 2,837, stripped branches 582, refuted by a passed board 1,138,
finished 6,638, standing after 2,837, no finish at all 1,096, searched 2,405.

**What it costs.** Measured 2026-10-02 with `t5d-bench --legal-profile`:
- dev machine, Release, g++ 16.2 `-O2`, one thread;
- the machine was shared with other builds and tests, so these are not quiet-machine
  figures;
- each cell is median / p95 / worst per expansion of `scope+relations` (moves, move
  graph, W/E scope, relations), with the classification inside it;
- in brackets, the share of the moves left unknown.

The synthetic M9 classes: 4 games per class, each position and 4 one-move children,
20 expansions per class.

| Class | pseudo-legal | fast | budgeted 64,2000 | exact |
|---|---|---|---|---|
| N100 T6 | 21 µs / 43 µs / 43 µs | 86 µs / 114 µs / 114 µs (46%) | 2.0 ms / 2.9 ms / 2.9 ms (14%) | 1.5 ms / 17 ms / 17 ms |
| N160 T7 | 23 µs / 38 µs / 38 µs | 95 µs / 119 µs / 119 µs (64%) | 2.1 ms / 3.0 ms / 3.0 ms (44%) | 5.4 ms / 59 ms / 59 ms |
| N240 T8 | 26 µs / 46 µs / 46 µs | 102 µs / 162 µs / 162 µs (33%) | 2.5 ms / 4.1 ms / 4.1 ms (20%) | 7.3 ms / 148 ms / 148 ms |
| N400 T10 | 40 µs / 55 µs / 55 µs | 142 µs / 232 µs / 232 µs (47%) | 2.4 ms / 10 ms / 10 ms (25%) | 17 ms / 145 s / 145 s |
| N600 T12 | 54 µs / 84 µs / 84 µs | 162 µs / 273 µs / 273 µs (27%) | 0.75 ms / 11 ms / 11 ms (7%) | 0.72 ms / 200 s / 200 s |

The corpus records: every action start and its children, by N bucket, under Q4's
ruled reading (then `--moves-after-shift`; since stage E7 the default).

| Boards | expansions | pseudo-legal | fast | budgeted 64,2000 | exact |
|---|---|---|---|---|---|
| 1-49 | 858 | 2 µs / 8 µs / 25 µs | 6 µs / 27 µs / 96 µs (39%) | 22 µs / 454 µs / 2.1 ms (2%) | 25 µs / 422 µs / 22 ms |
| 50-99 | 319 | 4 µs / 21 µs / 60 µs | 16 µs / 106 µs / 218 µs (45%) | 0.41 ms / 2.0 ms / 3.4 ms (11%) | 0.26 ms / 4.0 ms / 33 ms |
| 100-159 | 261 | 11 µs / 42 µs / 109 µs | 55 µs / 186 µs / 418 µs (43%) | 1.1 ms / 3.2 ms / 8.1 ms (16%) | 0.82 ms / 14 ms / 3.5 s |
| 160-239 | 180 | 12 µs / 70 µs / 281 µs | 131 µs / 262 µs / 700 µs (47%) | 1.1 ms / 3.5 ms / 4.9 ms (24%) | 0.85 ms / 14 ms / 2.6 s |
| 240-399 | 138 | 67 µs / 327 µs / 359 µs | 288 µs / 1.6 ms / 2.1 ms (55%) | 3.9 ms / 11 ms / 18 ms (33%) | 7.1 ms / 144 ms / 250 ms |

The figures by mode:
- **Fast** costs 3-5 times the pseudo-legal expansion (11 times in the records'
  160-239 bucket), and its worst case stays within 2.1 ms.
- **Budgeted** is bounded by its budget: a node costs a few microseconds and a check
  more with more boards. At 64 checks and 2,000 nodes the worst case was 18 ms.
- **Exact** has a median of 7 ms at 240 boards; the E4 filter took 157 ms. Its tail is
  the searches': up to 145 s and 200 s in the synthetic 400- and 600-board classes
  (the E4 filter: up to 247 s and 522 s).

How the moves were settled depends on the position. Random play leaves kings
attacked on many boards, so in the synthetic classes the stripped multiverse rarely
holds no capture:
- standing captures and passed boards settle 36-73% of the moves, as illegal;
- the fast mode proves few moves legal: 14% at 100 boards, none above.

In the records, the stripped multiverse alone proves 28-46% of the moves legal in the
fast mode (35-49% in the others), and its branches another 9-17% when they may be
scanned. `t5d-bench
--legal-profile` prints every tier's share.

## Stage E5b: the tactical scanner and t5d-scan6b

The verification layer's brute-force auditor (spec §11.1-§11.3; register H1-H6, H15,
H19) and the tool for experiment 6b (§15.6, M8). `include/t5d/scanner.hpp` states every
definition in full; in brief:

- **Unit.** A node is a turn start, an edge one whole legal action; depth counts
  forcing half-turns (one side's action), board moves along a line are reported beside
  it. Check status (R1), terminal verdicts and stand-pat live at turn starts, so
  terminal detection runs only at turn boundaries, by E4's exhaustive detectors.
- **Forcing set** (§11.1 defines none; each class a parameter): captures (within a
  board, across timelines, through time); *checking moves* - a piece of the mover's on
  a board the move creates, that board advanced by a pass, can capture an enemy royal
  (`royal_capture_from`, a scan of given source boards added to `t5d/check.hpp`), which
  counts royals on history boards (C2) and is held equal to the definition made
  literal (the move made, its boards passed, `for_each_royal_capture` filtered) under
  every ruleset; *branch threats* - a branching move whose arriving piece, on the new
  timeline's passed board, can capture a piece worth at least 300 (a placeholder: the
  spec leaves "immediate threat" undefined); promotions optionally. A forcing action
  is a legal action that begins with a forcing move (its seed), finished by the
  completion rule - on each mandatory board left, the quietest move (a move within the
  board, capturing nothing, of the least valuable piece) - or, where that leaves a
  royal capturable or a board unplayable, by E4's first legal finish (within 2,000
  nodes; a seed it cannot finish in time is dropped - the estimate channel's loss).
- **Search.** Negamax alpha-beta, iterative deepening, node-capped (H3). In check: every
  legal action - first up to 64 actions built from seeds in an evasion order, each
  searched as it is built, so that a cutoff ends the node (stage E5c, below); then, if
  none cut off, E4's enumeration (`list_legal_actions`), up to `evasion_cap` (512);
  beyond it the node is *capped* - searched over the actions E4 listed, an estimate,
  and its iteration is not fully resolved. Otherwise stand-pat, then
  captures (MVV-LVA), then checks and branch threats (killers, history, continuation
  history: H4); a node with no forcing action is quiet (E4's detector tells a
  stalemate). No reductions, null move, futility, ProbCut or delta pruning (H5; the
  scanner's pruning waits on 6b, M5/M8).
- **Two channels** (H2). A proof - Win(d), Loss(d), Draw(0) - comes only from E4's
  exhaustive verdicts, "some action leads to a proven loss of the opponent's" and "every
  legal action, enumerated exhaustively, leads to a proven win of the opponent's"; so
  a node cap inside a proof line, stand-pat and a capped node give none. Proofs are
  exported to the solved store (`SolvedStore`, `MemorySolvedStore`: Zobrist key with the
  ruleset ID, full comparison on a match; contradictory proofs throw) the moment they
  are found (H19); the TT (fixed size, always replace) holds estimates only.
- **Material eval** (H6 v1): the side to move's material on every timeline's latest
  board minus the opponent's. P N B R Q are the chess convention; U 250, D 200, S 700,
  W 125, C 300 are placeholders (`PieceValues`); royals 0.
- **Determinism** (H15): with one thread (the default), a node budget and no clock,
  the same position gives the same search node for node. Lazy SMP (H17) behind
  `ScannerOptions::threads`: stage E5c, below.
- **Nodes**: a move applied, a royal-capture scan, or a node of an E4 search the
  scanner calls - one unit for all the work.
- **Stand-pat validity** (6b's fifth test; `stand_pat_validity`), operationalised
  before measuring: at a quiet position (not in check, no forcing move) stand-pat is
  invalid when every legal action of the side to move - all of them when there are at
  most 32, else a sample of 32 (each first move completed) - loses material to the
  opponent's best forcing reply within 2 forcing half-turns (4,000 nodes each).

`tests/test_scanner.cpp`: constructed mates - in 1 half-turn on one board, across
timelines, by stale-checkmate (C8), through time onto a royal on a history board (C2),
in 3 (mate in two by checks) and lost in 2 - each confirmed by E4's detector and by a
brute-force game tree (`tests/test_game_tree.hpp`: minimax over the brute-force
reference's successors); E4's terminal positions (C6, C8, C9, C13, R1, Reversed
Royalty); C9's rewind (no false loss); random play from sparse starts, at generous and
at random node budgets and under every alternative ruleset, every proof checked
against the game tree; proofs that survive a one-entry TT and are answered from the
store; determinism; the budget curve; stand-pat validity on a zugzwang. The `long_`
variants check every proof, of any distance, on larger samples.

### t5d-scan6b

```
t5d-scan6b [--every K] [--leaves N] [--leaf-depth D] [--max-nodes N] [--budgets B,...]
           [--scanner-threads T] [--threads T,...] [--eval-rates R,...] [--nps N]
           [--root-ms MS] [--no-root-wall]
           [--synthetic-variants V,... --synthetic-games G] [--synthetic-classes N:T,...]
           [--jobs N] [--json FILE] [--summary FILE] [--per-position]
           [scanner options] [--no-moves-after-shift] [--rules ENCODING] [RECORD ...]
```

**Positions.**
- Roots are the turn starts of the records given, replayed as parse mode does.
- SYNTHETIC positions are labelled and never in a verdict.
- Leaves are a few half-turns of random legal play below each root.
- A position whose side to move could capture a royal (its last action broke C4), or
  that has no legal action, is counted but not scanned.

**The deterministic curve.** Every position is scanned once, deterministically, by one
thread to `--max-nodes`. A smaller budget follows the same path, so that one curve
gives the depth any budget resolves: the deepest iteration completed, and not capped,
within it (a proof or an exhausted tree resolves every depth). Each job keeps its
solved store across its positions, so the curves depend on `--jobs`: use 1 for a
reproducible run.

**The measures**, all read per board-count range - under 50, 50-99, 100-199 and 200+
boards - and never pooled (register M12). Pooled lines are printed as information,
labelled "not a verdict (M12)". The scanner's design thread budget, `--scanner-threads`,
is 5 (M13).
- (i) The horizon per budget.
- (ii) Rung 1, at `--scanner-threads` x nodes/s per thread / evaluation rate. The
  verdict uses the size-matched rate: §15.6's design rates, 9,600/s below 420 boards
  and 4,600/s above. Other thread counts (`--threads`) and fixed rates are printed as
  information.
- (iii) The root scan. After the deterministic scans, each root is scanned alone for 50
  ms of wall clock by `--scanner-threads` threads (Lazy SMP), with a fresh scanner and
  store. The one-thread reading at 50 ms x nodes/s from the deterministic curve is
  printed beside it, as information.
- (iv) Quiet-leaf reach at the rung-1 budget.
- (v) Stand-pat validity, by board count and by mandatory boards.

**The verdicts.** Each is printed with the brief's STOP rules: a share within 45-55%,
or a median depth within ±10% of its threshold. The output also says what the passes
rest on: iterations to depth, exhausted trees or proofs. The summary applies M12's
outcome rules to (iii) mechanically:
- every range from 50 boards up passes: continue as designed;
- 50-199 pass and 200+ fails: phase-dependent verification;
- 50-199 fail: STOP.

Rung 1 and quiet-leaf reach are judged by their own rules, range by range.

**Speed.** The nodes/s per thread is measured over the deterministic scans (thread CPU
time, less the leaf classification's) unless `--nps` fixes it, so that a quiet-machine
figure can be applied to the same curves. The wall-clock phase reports its own nodes/s
per thread and how busy the machine was. The profile of the exhaustive paths is
printed by board count.

The records are input on the command line; the tool names nothing under the oracle
harness.

## Stage E5c: the bounded optimisation pass

CHECKPOINT 2's ruling on 6b (register M12) asked for one bounded optimisation pass on
the two paths where the first measurement spent 85% of the scanner's nodes - listing
every legal action in check, and the mate tests at the horizon - and then 6b re-run
unchanged. Nothing else in the scanner changed: its search, forcing set, eval and the
6b tests are as they were.

### The in-check stage (`ScannerOptions::evasion_probe`, 64)

The profile (below) said why the two paths cost so much. In-check interior nodes
failed high 50-85% of the time, yet each listed up to 513 legal actions first; and
84-96% of the horizon mate tests answered "no mate", each by a hypercuboid search for
one legal action. So, in check:

- **Interior nodes.** The pseudo-legal moves are ordered as evasions: the TT's move, a
  capture of the checking piece (R1's status gives one capture of a royal it rests
  on, `check_witness`), a royal's move, a move onto a square the check's line crosses,
  other captures (MVV-LVA), then the rest by killers and histories. Up to 64 of them,
  in that order, are each made into an action by the completion rule's own moves and
  searched as soon as it is built. Only if none cuts off does E4's enumeration run, as
  before, and its successors not yet searched are searched.
- **Horizon nodes.** The same seeds are tried until one makes a legal action: then the
  answer is "no mate" and the node is an in-check leaf, as before. Only if none of the
  64 does does E4's mate detector decide.

Why this is sound (H2, H37):
- A seed's action is built through the state machine and tested by C4's test
  (`royals_safe`, R1). A seed it leaves unsafe is passed over, with no fallback search.
  So every action of the stage is a legal action.
- A cutoff by a legal action is a lower bound on the node whatever the other actions
  are, so the node is resolved without the enumeration. Only a Win proof can come of
  it: some action leads to a proven loss of the opponent's (∃).
- A Loss proof still needs E4's enumeration, exhaustive, and every child of both
  stages a proven win of the opponent's (∀).
- At the horizon, one legal action in hand is an exact "not mate" (C5). A checkmate
  is still only E4's exhaustive verdict.
- A capped node, a TT hit and stand-pat give no proof, as before.

`evasion_probe = 0` is E5b's scanner exactly: on the corpus (805 roots and 780 leaves,
100,000-node scans) it searched the same trees as E5b, node for node.

**Tests.** `scanner_evasion_probe_keeps_the_proofs` and its `long_` version scan the
sparse starts' random play, under both readings of Q4 - the Guide-literal one, asked
for explicitly, and the ruled one (R8), the default since stage E7 - and the
constructed positions. Each is scanned with the stage off, at 64 and at 1, at a
generous budget and at random ones. Every proof of up to 1 half-turn (3 in the `long_`
version) is checked against the game tree, and where two settings both prove, they
prove the same. `scanner_evasion_probe_off_is_e5b` checks that 0 leaves the stage out.

**The profile**, from `t5d-scan6b` on the corpus records under Q4's ruled reading
(then `--moves-after-shift`; the default since stage E7), 100,000-node deterministic scans, one job. The machine was
shared with other builds and tests, so the times are not quiet-machine figures. Each
cell is the share of the scans' nodes, then of their wall time.

| Boards | E5b: listing in check | E5b: horizon mate tests | E5c: listing in check | E5c: horizon mate tests | E5c: the in-check stage | completion fallbacks (E5b → E5c) |
|---|---|---|---|---|---|---|
| < 50 | 62.7% / 59.7% | 17.7% / 8.2% | 25.6% / 22.2% | 5.9% / 3.3% | 19.4% / 15.6% | 4.7% → 12.6% of nodes |
| 50-99 | 44.2% / 48.8% | 23.4% / 11.6% | 25.2% / 25.8% | 7.7% / 3.6% | 16.0% / 13.4% | 17.6% → 23.3% |
| 100-199 | 32.6% / 38.5% | 31.1% / 24.5% | 20.1% / 21.3% | 9.4% / 7.3% | 15.0% / 14.7% | 26.1% → 38.1% |
| 200+ | 16.8% / 21.9% | 32.7% / 20.1% | 10.6% / 9.3% | 1.0% / 0.5% | 4.9% / 6.0% | 40.4% → 63.5% |

What changed:
- **The two paths together.** They fell from 64-80% of the nodes below 200 boards to
  45-51%, and at 200+ from 50% to 16%.
- **Nodes per tree position.** They fell from 84 to 37.
- **The probe's reach.** It resolved 87-96% of the in-check interior nodes by a cutoff.
  It answered 87-99% of the horizon tests.
- **Capped nodes.** In-check nodes capped fell from 8,196 to 3,475.

At 100 boards and more, the work is now led by the completion fallbacks (E4's first
legal finish when the quiet completion leaves a royal capturable): 38% of the nodes at
100-199 boards and 64% at 200+. That path is outside the pass M12 ruled.

### Lazy SMP (H17, `ScannerOptions::threads`)

The scanner's design budget is 5 threads (M13: 5 of the 8 cores per GPU). With
`threads` > 1 every thread searches the same root by iterative deepening (spec §11.3).
They share two things:
- **The TT.** It is keyed by the 5D Zobrist key with the ruleset ID (H18) and holds
  estimates only. It is lockless: a slot holds its entry packed in one word and, beside
  it, the key XORed with that word, so a slot another thread is writing reads as a
  miss, never as another position's entry. The best move's key, an ordering hint, sits
  in a third word, where a torn write only reorders moves.
- **The solved store.** Proofs go there the moment they are found (H19).
  `ConcurrentSolvedStore` (a scanner's own) spreads positions over 64 shards by key,
  each behind a reader-writer lock, and keeps `MemorySolvedStore`'s contract: it
  compares positions in full on a key match (`same_position`, §11.3's hashing trap) and
  refuses two kinds of proof for one position. A store that is not safe for threads
  has its calls serialised.

Everything else is per thread: killers, history and continuation history (§11.3).
They carry over from scan to scan until `clear()`, which is meant for between games.

**Diversity.** Two standard Lazy SMP means:
- **Staggered depths.** The main thread searches every depth. Helper `i` searches depth
  `d` only in alternate blocks of `1 + (i - 1) / 2` depths, shifted by `i`, so helpers
  are already at `d + 1` and `d + 2` while the main thread is at `d`.
- **Perturbed order.** A helper perturbs its order of quiet moves by a hash of the move
  and its index, below a killer's weight.

**Stopping and the result.**
- A thread that proves the root or exhausts its tree stops the others.
- The node budget is shared: each thread adds its nodes in batches of 256, so several
  threads may overrun it by a few hundred nodes each. The clock is shared too.
- The result is the deepest iteration any thread completed. `resolved_depth` is the
  deepest that completed uncapped.
- The root's proof is the store's. Every thread's root proof must agree with it, or
  the scan throws `std::logic_error`. An exception in any thread is rethrown by `scan`.

**Soundness.** It is unchanged: the TT never answers with a proof, a proof is built
only as in the single thread, and the store refuses a contradiction.

**Contention.** Threads that copy one state share its timelines and boards by reference
count, and contend on those counts at every copy and every move. So each helper
searches from its own copy of the root, rebuilt board by board (`isolated_copy`; where
the state cannot be rebuilt from its boards alone, because play decided its activity
under Q1's consecutive rule, Q3 or Q6, it shares the root).

`threads = 1` is the deterministic mode (H15). On the corpus (805 roots and 780
leaves, 100,000-node scans) it searched the same trees as the scanner before Lazy SMP,
node for node.

**Scaling on the dev machine.** This machine was shared with other work.
- **Long scans.** On 1-second scans of 68-105-board positions each thread ran at 94%
  of a lone thread's speed with 2 threads, and at 82% with 5.
- **The corpus's 50 ms root scans.** With 5 threads each ran at 61% of one thread's
  speed. Many of those scans are short, so starting the threads and copying the root
  weigh more.
- **Five processes.** Five separate processes each ran at 96%.

At 50 ms, 5 threads resolved at least one thread's depth in every board-count range,
under Q4's ruled reading. The shares of roots reaching 5 forcing half-turns were:

| Boards | 1 thread | 5 threads |
|---|---|---|
| < 50 | 75.7% | 76.5% |
| 50-99 | 28.6% | 30.7% |
| 100-199 | 22.7% | 24.7% |
| 200+ | 2.4% | 2.4% |

**Tests.**
- `scanner_lazy_smp_no_false_proofs` (and its `long_` version) scans with 2, 3, 4 and 8
  threads, by node budget and by the clock, over the sparse starts' random play under
  both readings of Q4 (the Guide-literal one asked for explicitly; R8's the default),
  and over the constructed positions. All
  scans of a thread count share one store. Every proof of up to 1 half-turn (3 in the
  `long_` version) is checked against the game tree, and every proof against one
  thread's.
- `scanner_lazy_smp_store_refuses_contradictions` checks the shared store's contract.
  Eight threads put proofs at once and then contradictions: every contradiction is
  refused, and the shortest distance is kept. A plain `MemorySolvedStore` shared by
  four threads gets every proof, each checked against the game tree.
- `scanner_lazy_smp_reaches_the_single_threads_depth` gives 5 threads and 1 thread the
  same 40 ms on a sample of positions from Standard and Two Timelines play. Summed
  over the sample, 5 threads must resolve at least one thread's depth (one retry, for
  timing noise).

## Stage E7: Q4 as ruled, and the submit action (D8)

Built after the user's CHECKPOINT 2 rulings (rulebook R8 and R9; register D8).

- **Q4, R8 - the default.** A player may keep moving on their remaining playable
  boards after the present has shifted; the turn ends only at submission.
  `RuleOptions::moves_after_shift` defaults to true, and every tool and the gate
  adapter follow it: `t5d-gate`, `t5d-measure`, `t5d-bench`, `t5d-scan6b` and the C
  API's ruleset 0. The Guide-literal reading - no move once the present has shifted -
  is a named alternative, for diagnosis only (never for `gate.toml`):
  `--no-moves-after-shift` (and `t5d-gate --rule moves_after_shift=false`);
  `--moves-after-shift` is accepted and ignored. The default ruleset's encoding and
  ruleset ID stay 0, so no default Zobrist key changed; **bit 3 of the encoding now
  marks the Guide-literal reading** - it meant the opposite before - and keeps the
  ruleset ID that bit always had (`T5D_RULES_NO_MOVES_AFTER_SHIFT`; the old name
  `T5D_RULES_MOVES_AFTER_SHIFT` is now 0). The legal actions the gate counts are then
  the oracles' (an action that may be submitted and one that goes on from it are two),
  the hypercuboid realises them in its FreeOrder regime, and on the corpus the
  adapter's output is byte for byte the stage-E4 engine's under
  `--moves-after-shift` (figures: "The gate adapter").
- **Q5, R9.** Only the active boards in the present are mandatory - the default
  already (`mandatory = InPresent`); `AllActive` stays as the documented alternative.
- **D8 in the expansion and the C API**: (g) of "Expansion" - the acting player and
  the submit action, possible (the obligations met) and legal (C4 too: the submission
  test, exact) - appended to `t5d_expansion` as `acting_player`, `submit_possible`,
  `submit_legal` (three `int32_t` after `src_extra`; every older field keeps its
  offset). No per-board or per-square feature changed. Under the ply split the submit
  action has its own policy logit (§8.2), read from a global register; under
  whole-turn generation submitting ends every enumerated action.
- **The Python binding** follows in the same change: `python/t5d/engine/capi.py`'s
  `CExpansion` mirrors the three fields, `T5D_RULES_NO_MOVES_AFTER_SHIFT` is bit 3 and
  `T5D_RULES_MOVES_AFTER_SHIFT` 0, `Expansion` copies `acting_player`,
  `submit_possible`, `submit_legal`, and the binding checks every mirrored struct
  against `t5d_layout` at load (see "The C API"). The network does not read the new
  fields yet: its acting-player input and submit logit are stage N5, and until then
  `to_position` refuses a state past the shift (`PresentMismatch`).

## Stage E6a: the main search (`t5d/mcts.hpp`)

PUCT over a tree - or, behind a flag, a graph - of game states, fed by a batched
evaluator (design spec §8.1-8.7; the consumer side of §11.1's proof channel; §11.2's
synchronous debug mode; §15.4's invariants). Written from the spec and the papers it
cites - AlphaZero's PUCT, progressive widening, MCTS-Solver (Winands et al. 2008) with
Cazenave and Saffidine's score bounds for draws, Monte-Carlo graph search with residual
backups (Czech, Korus and Kersting 2021), KataGo's variance-scaled cPUCT, uncertainty
weighting and LCB selection - and the Chess Programming Wiki; no engine's code (H29).
Single-threaded; no time management (E7, E8), scanner (rung 1) or Lazy SMP yet.

**Nodes and values.** A node holds a `GameState` and its acting player (register D8)
- the player whose action is in progress, until it submits; after the present has
shifted inside the action it is no longer the colour to move on the present. Values
are in [-1, 1] from that player's view; backing up across an edge flips the sign only
when the two nodes' acting players differ - under the ply split, only across a submit
edge. The evaluator's value is a 51-bin categorical
distribution (D3), bins centred on `(2k - 50) / 50`; its mean is the leaf's value and
its variance feeds E2, E3 and LCB.

**D2's two arms** (`SearchParams::split`; D2 is undecided, ablations 6g/6k):

- *Board-move as ply* (`BoardMove`, the placeholder default): a node is a turn's start
  or a state during an action; its children are the legal moves and - turns have
  variable length (rulebook R8, register D8, spec §8.2), so submitting is an action -
  a **submit edge** exactly where submitting is possible (the obligations met, T5:
  `Expansion::submit_possible`), legal exactly where the submission passes C4
  (`Expansion::submit_legal`, exact); an illegal one is kept, pruned, with no prior.
  Nothing is submitted automatically: after the move that shifts the present the node
  still belongs to the same acting player and offers submit beside any further move
  on that player's remaining playable boards (R8, the default rules; under Q4's
  Guide-literal reading, `moves_after_shift = false`, submit alone). Only a submit
  edge changes the acting player. Submit's prior is the evaluator's submit logit
  (D8: from a global register - the network's stage N5) when it gives one, else the
  placeholder `submit_share` (0.5) of the prior where moves are open too: with a flat
  move policy a submit logit of 0 among N moves would make every action play nearly
  every optional board (measured: 101 moves in 14 actions, 54 timelines). Mid-turn
  check is legal; terminal detection runs only at turn starts (§8.2).
- *The ends, from the rulebook.* At a turn's start, no legal action ends the game
  (C5): checkmate if the side to move is in check (R1's status), otherwise stalemate
  (C10, a draw; C13) - decided only by E4's exhaustive detector (H37). During an
  action there is no game end (C5 speaks of the start of a turn): a partial action
  whose remaining moves and submission are all illegal has no legal completion, so the
  move that reached it was not legal (R1: a move is legal when a legal action
  continues with it). It is a **dead end**, pruned and never scored (§8.2). The exact
  classification never reaches one; a budget (an unknown move) or the pseudo-legal
  test seam may, and it is found when the node is expanded. After the shift a legal
  submission finishes the action at once - but a further optional move (R8) can still
  expose a royal, and such a move may lead to a dead end like any other.
- *Whole turn* (`WholeTurn`): a node is always a turn's start; its children are the
  legal actions (`list_legal_actions`, E4), at most `max_actions` of them (a listing
  stopped there, or by `listing_nodes`, is incomplete and never proves a loss). Every
  listed action is stored with its node (its moves and an edge), so `max_actions`
  bounds a node's memory: at the old default of 4,096 a node of a 100-board position
  held about 1.5 MB (review B); the placeholder default is now 256. D8's submit is no
  decision of its own here (an action is listed submitted), so the evaluator's submit
  logit is not read and the C API reports `submit_possible` 0: this arm has no
  end-of-turn token (a spec issue; not built). The
  network's policy is over single moves (D1), so an action's prior comes from its
  moves' logits - their sum (a product of per-move factors; default) or their mean
  (`action_prior`) - each move matched to the turn start's pseudo-legal candidate with
  its squares and promotion (a J6 branch has the non-branching jump's coordinates).

**How legality enters** - one call site (`leaf_expansion` in `src/mcts.cpp`): a leaf's
expansion with `ExpandOptions::legal_moves`, classified by E5a's tiers under
`SearchParams::legality`. The policy for moves the budget leaves unknown (build-log
spec issue 46; each part a parameter):

| | What | Parameter (placeholder default) |
|---|---|---|
| (a) | proved-illegal moves dropped; legal and unknown kept, unknown flagged | `legality` (`budgeted(64, 2000)`; `exact()` leaves nothing unknown) |
| (b) | an unknown move gets its prior | `unknown_prior_scale` (1) |
| (c) | the first time selection picks an unknown edge it is verified: illegal - removed, siblings' priors renormalised, selection repeated, nothing backed up; legal - flag cleared; still unknown - kept, flagged | `verify_unknown` (on), `verify_legality` (`budgeted(4096, 200000)`) |
| (d) | a node is terminal only on an exact verdict of E4's detectors | - (H37) |
| (e) | a child through an unknown edge never raises its parent's lower bound (it still bounds it from above: a superset of the legal children is a sound upper bound), and is never a proven win for final selection | - |
| (f) | training's policy target is the root's visits (`root_children` lists the instantiated children; the rest have none); the network's move mask is a leaf's legal and unknown moves - every candidate its expansion kept - not the root's children | `RootChild::visits`; `Expansion::move_legality` (`t5d_leaf.legality`) |

Every pruning renormalises the siblings' priors. Before a move is played, an unknown
one is classified exactly (`Search::verify`; self-play does this, and plays only moves
known legal). `legal_moves_only = false` is a test seam: pseudo-legal children, whose
dead ends are pruned.

**Terminals.** At a turn start, E4's `detect_mate` under `mate_nodes` and
`mate_bytes`: checkmate (-1, proven), stalemate (0, proven, C10), or - the detector
stopped - unknown, **never terminal** (H37). Self-play's turn cap (I11, `turn_cap`: a
present past `turns` turns, or `boards` boards, at a turn start) scores 0 and proves
nothing; the rules' verdict comes first.

**Selection** (E1). PUCT, `U = cpuct * P * sqrt(N) / (1 + n)`, over the instantiated
children: the children sorted by prior, the first `ceil(k0 (N + 1)^alpha)` instantiated
(progressive widening), with `k0 = max(widening_min, widening_scale * exp(H))` from the
policy's entropy H - sharp policies narrow, flat ones widen. FPU: an unvisited child
scores `Q(parent) - fpu_reduction * sqrt(visited prior mass)`. Root Dirichlet noise for
self-play (`root_noise`, `dirichlet_alpha` per child, `dirichlet_epsilon`; seeded,
deterministic). Behind flags: E2 - cPUCT scaled by `clamp(1 + lambda (sigma / sigma_ref
- 1))`, sigma the node's value deviation (the categorical variance as a prior of
weight `variance_prior_weight`, blended with the backed-up values'), and backups
weighted `sigma_ref^2 / (sigma_ref^2 + var)`; E3 - U multiplied by `clamp(1 + beta
(sigma_child / sigma_ref - 1), 1, cap)`, never below 1, nothing for an unevaluated
child.

**Batches and virtual loss.** `gather` makes up to `batch_size` visits - the leaves it
sends for evaluation and the visits it completes without the evaluator (terminals,
solved nodes, cache hits, MCGS residuals) together, so that terminal visits cannot pile
up behind a waiting leaf's virtual loss; each waiting path
carries one virtual visit, counted as a loss, on every node and edge it crosses. A
second path reaching a waiting leaf is a collision (its virtual loss held until the
batch ends, so later selections diverge). One evaluation is asked per distinct state
in the batch. The virtual loss is always released: by `deliver`, by `abandon`, when the
evaluator throws, when an evaluation is malformed - and when anything fails inside
`gather` or `deliver`, an allocation included: `gather` abandons its batch before the
exception leaves, `deliver` keeps what it delivered and abandons the rest (the tests
inject a `bad_alloc` at every allocation of both). `run` resets every field of the
evaluation buffers it reuses, the submit logit included. The epoch cannot change while a
batch waits (its evaluations belong to the epoch they were asked under).

**MCGS** (C11, prototype, `graph`). Nodes keyed by `node_key(state, keying)`: the 5D
Zobrist key - the ruleset ID, every board in its place (so what has been played
inside the action: a played board has its successor) and the acting player (a state
after the shift and the same boards submitted differ in it) - mixed, by default, with
every timeline's branch origin (`NodeKey::StateAndOrigins`: C23 ruled a board's causal
past to be its ancestry, the lineage reading of spec issue 30, so states that differ
only in an origin are different positions for the network); `NodeKey::State` leaves
origins out. The key carries the acting player (H18 as amended: the 5D Zobrist key
itself does - `Multiverse::zobrist` mixes in the player to move, which is the acting
player until the submission - so a state whose present has shifted inside an unsubmitted
action and the opponent's turn start with the same boards have different keys, and the
exact comparison behind a key match tells them apart too). A node's identity in the
graph is its key **and whether an action is in progress**: a turn start and a state during an action can be one `GameState` (an
optional board played at the end of one action, or at the start of the next - same
boards, same player, same present), yet only a turn start can end the game (C5) or be
capped (I11), and only a partial action can be a dead end, so they are never one node
(a submit edge always leads to a turn start, a move to a state during an action;
`check_invariants` holds it). Whether a leaf is a turn start is a property of the path,
not of the state, so the evaluator is never told it: an evaluation is a function of
the state and the epoch, and the cache is shared between the two. An exact comparison
follows every key match (a collision gets an unshared node; `key_bits`, a test seam not
in the C API, cuts the keys short so that the tests meet collisions). Statistics
live on the edges (path statistics; in a tree they equal the child's). Reaching an
expanded node through an edge whose Q lags the node's by more than `residual_epsilon`
(or an edge never crossed) backs up the residual correction instead of descending.
5D has no repetition rule and every move adds a board, so the graph is acyclic.

**The solver** (H2, E9; behind `proof_channel`, **off by rule**). Register H41
(ruled at CHECKPOINT 4): "the search's own solver is part of the proof channel:
game-end scoring stays on; solved-status propagation and export are off until the gate
passes, the search's soundness fixes land with tests, and the proof channel is signed
off; its proofs are cross-checked against df-pn and oracle mate verdicts" (spec §11.1
item 2; H23). So `SearchParams::proof_channel` - named as the proof channel's switch,
`t5d-play --proof-channel` for experiments - is false: no node is marked solved, no
sibling pruned and no secure child chosen from proven bounds. A game-end position - C5
at a turn start, by E4's exhaustive detector, or I11's cap - is still scored as its
result, which is ordinary MCTS; nothing else is proven. Export is `proof_store`
(refused). With the switch on: bounds
`[lo, hi]` per node from the mover's view: a terminal's exact value; `lo` the best child's lower bound (children through known
edges), `hi` the best child's upper bound once every child exists and the list is
complete. Bounds only tighten; a contradiction throws. A solved node leaves selection
and consumes no evaluation (a visit backs up its exact value); a solved child is
selected only while it has fewer than `solved_visit_cap` visits (the "visit threshold
for solved-node selection", read so). **No persistent solved store**: H19's export
waits on the gate (H23); `proof_store = true` is refused.

**Final selection** (E5, E9; `choose`). With the solver on, a proven win is played;
children proven lost are ineligible unless all are; if an eligible child is solved, the secure child - the
largest lower bound, exact for a solved child and `Q - A / sqrt(n)` otherwise; else
`final_rule`: `Visits` (the baseline) or `Lcb` (default) - among children with at least
`lcb_floor` (15%) of the most visits, the largest `Q - z sqrt(var / n)`, var plus
`lcb_categorical_weight` times the child's categorical variance (the optional second
term). Under MCGS the visits are the edges'. No depth factor (E6, refused).
`select_final` is a pure function (`final_eligible` says who may be chosen).
Self-play samples the root's visits at a temperature instead (`choose_by_visits`).

**Persistence** (C12, C13). `advance(child)`, `advance_moves`, `advance_action` (the
opponent's reply) and `advance_to(state)` re-root: the new root's subtree is carried,
every node it cannot reach deleted (renumbered breadth-first); the evaluation cache -
state-keyed - survives; the root hook (`set_root_hook`) and the evaluator's
`on_new_root` are told, for an evaluator holding K/V of the old conditioning. Pondering
is `run` on the post-move root, then `advance_action` on the reply - atomic: the whole
action is checked (each move pseudo-legal where it is made, the submission legal) before
the first re-root, and a failure changes nothing; from a turn start an action has at
least one move. Evaluations are
cached per (state, refresh epoch) (`set_epoch`; 0 in the dense configuration).

**Memory.** The evaluation cache counts each entry for what it alone keeps alive once
its search is gone - the map's node, its logits, its state's list of timelines, and the
boards (with their timeline objects) the step into the state made
(`EvalCache::entry_bytes`; within 0.75-1.33 of the heap actually freed in the tests,
where the old estimate was 1.4-2.2 times low) - and is purged, deterministically, of
every state with fewer boards than the root, on a search's creation and on every
re-root: every move adds a board and none is ever removed, so those states cannot come
back (`purge_fewer_boards`; a cache shared across games would lose live entries). Along
a game a small shared cache so stays useful (review B: without it, no hit after about 60
decisions). The graph's memory is estimated the same way (`memory_bytes`: nodes, edges,
actions, states' own heap, the MCGS table; within 0.75-1.33 of the heap in the tests),
and `Limits` carries a node budget and a memory budget besides playouts and evaluations:
`run` stops between batches when one is reached - or when a wall-clock deadline
(`Limits::seconds`, for tools and match play, never training) passes - and returns why (`StopReason`, also
`stats().last_stop`).

**Determinism** (H15). One thread; with a deterministic evaluator a search is
bit-reproducible (`digest()` hashes every node's and edge's statistics in a canonical
order); `synchronous` is the debug mode - batches of one leaf.

**The evaluator** (`Evaluator`): a batch of `EvalRequest`s - the leaf's state, its
expansion (everything the network takes: features, candidate moves, move graph,
relations, W/E; `expand` sets k, the features and the relations), its key and epoch -
answered with one logit per candidate move and a 51-bin distribution. Built in:
`uniform` (logits 0, value 0), `random` (a function of seed, key and epoch only - so
still a function of the state), `material` (`tanh(balance / 10)` over every
timeline's head board; captures' logits 0.5 x the captured value; P N B R Q 1 3 3 5 9,
the fairy pieces at the scanner's placeholders - unicorn 2.5, dragon 2, princess 7,
brawn 1.25, common king 3 - build-log spec issue 56, one set for the engine).

**The C API** (`t5d/capi_search.h`, in `t5d.dll` / `libt5d.so`): the step API for an
external evaluator - `t5d_search_new(game, turn_start, params, &search)`,
`t5d_search_gather(search, max_leaves, max_playouts, &count)`, `t5d_search_leaf(search,
i, &leaf)` (a `t5d_leaf`: key, epoch, the acting player, D8's
`submit_possible` and `submit_legal`, the `t5d_expansion` - which carries them too -
and each move's legality), `t5d_search_deliver(search, logits, logit_count,
submit_logits, submit_count, values, value_count, value_kind)` - one float32 logit per
candidate move, logit i scoring `leaf.expansion.moves[i]`; one submit logit per leaf
(NaN: none) or none at all (NULL, 0: the placeholder share); 51 float32 values per leaf
as probabilities (`T5D_VALUE_PROBABILITIES`) or logits (`T5D_VALUE_LOGITS`), from the
leaf's acting player - and `t5d_search_abandon`; then `t5d_search_info`, `_child`,
`_choose`, `_choose_by_visits`, `_verify`, `_advance`, `_advance_action`,
`_root_game`, `_set_epoch`, `_run` (built-in evaluators) and `_run_limits` (within a
`t5d_search_limits`: playouts, evaluations, nodes, bytes; it reports the
`T5D_STOP_*` reason), `_new_sharing` (a new search sharing another's evaluation cache:
C14 through the binding), `_digest` (H15's digest - the
tests compare the step loop's graph with the C++ search's by it), and `t5d_search_layout`
(this header's struct layouts, as `t5d_layout` gives `t5d/capi.h`'s).
`t5d_search_params` mirrors `SearchParams` (fill it with `t5d_search_params_default`).
The header's structs are appended to, never reordered; the one struct of `t5d/capi.h`
each embeds (`t5d_leaf::expansion`, `t5d_search_params::expand`) - provisional, and
growing as that header does (E7 appended D8's fields to `t5d_expansion`) - is its last
member, so that its growth moves none of the others (a test holds it last).
A malformed batch is refused and abandoned. Errors: `T5D_ERR_ARGUMENT` for the caller's
(an argument, or a call the search's state does not allow - deliver with no batch
waiting, `max_leaves` 0, a new epoch under a waiting batch); `T5D_ERR_MEMORY`;
`T5D_ERR_INTERNAL` for the search's own failure (a broken invariant, a bounds check) -
the C++ API throws `std::invalid_argument` for the caller's errors and nothing else, so
the boundary can tell them apart. Solver bounds are search state, never to be persisted
as proofs (H19, H23; the header says so).

**Validation.** Every real parameter must be finite (NaN or infinity in, say,
`virtual_loss` or `cpuct_variance_scale` poisoned selection; each is refused by name),
as must `choose_by_visits`'s temperature and random number and `two_hot`'s value.

**Self-play** (`t5d/selfplay.hpp`, `play_game`). Self-play: a fresh search per decision
(C14), sharing only the evaluation cache, the move sampled from the visits at
`temperature` for `temperature_decisions` decisions, then the most visited, with root
Dirichlet noise (E1, §8.4: `PlayParams::search` has `root_noise` on by default; a bare
`SearchParams` has it off). Match play (`match_play`): one search carried across
decisions, the final rule, noise off.
Every turn start is judged first by E4's detector (checkmate, stalemate; unknown ends
the game unjudged), then by the turn cap (scored a draw, counted:
`GameRecord::cap_bound`, `search_turn_caps`). Every move is checked pseudo-legal and
every submission legal before it is made; a chosen child the legality budget left unknown
is classified exactly first (`GameRecord::verified_choices`), and the visits a decision's
root already held (`reused_visits`: C12's reuse in match play, 0 in self-play) are
counted. Under the board-move split an action is its
moves and then a submit decision (never automatic); actions are counted at
submissions, decisions as moves plus submissions.

**t5d-play** prints the game's 5dpgn record (the reader's long form; `*` and a
`Termination` header for a game the rules did not end, the turn cap's included), which
the reader and `t5d-gate` replay; a summary goes to standard error:

```
t5d-play [--variant NAME] [--evaluator uniform|random|material] [--playouts N]
         [--split ply|turn] [--graph] [--node-key state|origins] [--no-widening]
         [--e2] [--e3] [--proof-channel] [--batch N] [--synchronous] [--match]
         [--final visits|lcb]
         [--temperature T] [--temperature-decisions N] [--noise | --no-noise] [--cap-turns N]
         [--cap-boards N] [--max-actions N] [--listing-cap N] [--mate-nodes N]
         [--max-seconds S] [--seed S] [--no-moves-after-shift] [--rules ENCODING]
```

Defaults: Standard, uniform, 64 playouts per decision, the board-move split, a tree,
self-play with root noise (`--no-noise`: none), the proof channel off (H41;
`--proof-channel` for experiments, said in the Engine header), a cap of
20 turns or 300 boards (uniform play branches freely: by turn 12 a game can
hold 300 boards on 50 timelines, and with root noise time travel can hold the present
back for good - a Standard game reached 509 boards on 110 timelines in 39 actions
without passing turn 20). `--mate-nodes N` sets the search's terminal-check
budget; `--max-actions` (checked at turn starts) and `--max-seconds` (checked before
every decision, and each search's deadline: an action under way when it passes is not
recorded) are safety stops - the record ends `*`, its Termination header naming the
stop. Every number is range-checked (never wrapped into a narrower type; reals finite):
a bad one exits 2. The rules come through
the tools' shared option reader (`tools::ruleset_option`), so the default is the
engine's - Q4 as ruled, R8 - and `--no-moves-after-shift` (Q4's Guide-literal reading)
or `--rules ENCODING` names another ruleset, written into the record's `Rules` header
(the reader takes no rules from a record: replay it under the same option, as
`t5d-gate --no-moves-after-shift` does). The summary on standard error says where the
time went: the search's terminal checks, leaf expansions (the legality classification
included), action listings and verifications, and the game's own turn-start verdicts.

**Tests** (`tests/test_mcts.cpp`, `test_mcts_memory.cpp`, `test_selfplay.cpp`,
`test_capi_search.cpp`, `test_play_exe.cpp`): §15.4's invariants under uniform and
random evaluators, across six configurations (both splits, tree and graph, both node
keys, widening on and off, E2 and E3, the proof channel on in three and off in three) -
`check_invariants()` holds visit counts and backed-up values consistent
through backup (a node's visits = the visits that ended there + its edges'), no
virtual loss left, every node reachable, bounds never contradicting their children,
terminals never evaluated, the turn cap never a proof; the solver proving a mate in
one, a forced draw and a forced loss (each premise re-checked by E4's detectors);
final selection only among eligible children (constructed statistics, 3,000 random
sets, real searches); re-rooting keeping exactly the reachable nodes, the statistics
and the cached evaluations; bit-reproducibility; transpositions evaluated once and
duplicates bit-identical; the legality policy; self-play games under both splits from
five variants and two constructed endings, every action replayed legal, every ending
re-judged, every record replayed by the reader; `t5d-play`'s records replayed by
`t5d-gate`'s full mode; the C step loop building the same tree as the C++ search.
After D8: the submit edge offered exactly where submitting is possible, pruned exactly
where C4 fails, never taken automatically, and the only edge that changes the acting
player (both Q4 readings, with and without the pseudo-legal seam); the node key
telling apart the acting player and what was played; the solver against E7's
brute-force game tree (`test_game_tree.hpp`) under both Q4 readings - never opposite
verdicts, every mate in one proved; decisions counted as moves plus submissions;
`t5d-play`'s rules option. After the reviews: MCGS never merging a turn start with
the same state during an action (the reviewers' three repros, each line reached first
in turn; random graph searches with every dead end checked against the reference
enumeration); the proof channel off by rule (H41), game ends still scored; H18's acting
player in every key (a shifted, unsubmitted state and the opponent's turn start with
the same boards: different Zobrist and node keys, told apart by the cache under one
key, never one MCGS node with every key colliding); one test for each
mutation review C found surviving (the backed-up value's side across submit edges,
the residual's side, the step API against the C++ search by digest with finite submit
logits, value logits in their order, proofs through unknown edges, the visit
threshold, E2's weight, E3's floor, FPU, the submit share, the turn cap's boundary,
the cache's epoch, MCGS's exact comparison under colliding keys, self-play's
verification and fresh trees); the call sequences (atomic `advance_action`, the epoch
under a waiting batch, stale submit logits); a `bad_alloc` injected at every
allocation of `gather` and `deliver`; the cache's and the graph's byte counts against
the heap; the purge; the budgets and the deadline; validation; the C ABI's embedded
structs last; self-play's noise and time limit.

## Stage E8: the verification layer's redesign (M14, a prototype until 6b′)

6b failed from 50 boards up even after E5c (CHECKPOINT 3, spec §11.1 "The redesign
after 6b"): the brute-force scanner paid turn-level exactness - completing every
forcing line into a legal turn, listing every legal action in check, scanning for
royal captures - on every line, whether the line could end in a proof or only in an
estimate. Stage E8 builds the redesign the ruling (register M14) describes, as a
prototype for experiment 6b′ (M15) to decide. The E5b/E5c scanner is kept unchanged
beside it: below 50 boards M15 keeps whichever of brute force and df-pn decides more
roots, so both stay measurable.

This section is the design, written before the code (the brief asked for it in that
order); the parts' headers restate it next to the code.

### The channel split (H2, §11.1 part 1; architecture §13.2.7)

- **The proof channel** (`t5d/prover.hpp`) pays every exactness cost: legal actions
  only, every defence of the defender, terminal verdicts by E4's exhaustive
  detectors (H37). It answers one question at a root and nothing else (below).
- **The estimate channel** (`t5d/estimate.hpp`) pays none of them: a cheap, inexact
  local search over capture and threat sequences, *without turn completion* - a move
  is followed at once by the opponent's reply, as if the move were the whole turn -
  with the v1 material eval (H6). Its output is a score and a "tactically live"
  flag: soft priors and force-expansion hints only (H2). Its errors cost search
  efficiency, never correctness: it proves nothing, exports nothing to the solved
  store, and no move it makes reaches the tree - moves reach the tree only through
  the generator, which is exact.

### The proof channel: depth-limited df-pn over semimoves

**The question.** At a root - a turn start, its side to move the *attacker* A, the
other side the *defender* D - the channel asks: *does A force checkmate within d
forcing half-turns?* (d = 5 by default: A, D, A, D, A - mate-in-3, §15.6's
"dlshogi's leaf-solver level", M15). The answer is one of
- **proven** - A mates within d whatever D does: an exact game fact, exported at once
  to the solved store as `Win(k)`, k <= d the proof's own distance (H19);
- **disproven** - no such mate exists *within the attacker's forcing class* (below):
  recorded as such ("no mate within d", with the class) in the disproof store;
- **unknown** - the budget ran out first. Never "mate" on an exhausted budget (H37):
  a node is proven only from exact sub-results.

**The attacker's forcing class** (the spec defines none; each part a parameter,
`ProverOptions`, with a placeholder default and a spec issue):
- *seeds* - which first moves A's forcing actions may begin with: `Checks` (default,
  dlshogi's tsume rule: checking moves by the scanner's definition (b) - a piece of
  A's on a board the move creates, that board passed, can capture a royal);
  `Forcing` (the scanner's whole forcing set: captures, checks, branch threats);
  `All` (every pseudo-legal move - the literal "mate within d", used by the tests);
- *completion* - how a seed becomes a whole action: `Canonical` (default, 6b's unit
  - one action per seed, the first legal completion in a fixed preference order,
  below) or `Every` (every legal completion: the action space beginning with the
  seed, exactly).
Proofs are real-game facts whatever the class (A's options are a subset of its legal
actions; D's are all of them). Disproofs are facts about the class. With seeds `All`
and completion `Every` the question is exactly "mate within d" and the tests hold the
channel to the brute-force game tree both ways.

**The defender's turn: every legal action.** D gets no stand-pat and no forcing
restriction: an AND node is proven only when every legal action of D's leads to a
proven mate (or D has none and is in check: checkmate).

**OR and AND nodes when a turn is a sequence of semimoves.** A *semimove* is one
timeline's share of a turn (§8.2, §11.0 item 3): on one of the mover's playable
boards, "play nothing here" or one move from that board. Every node of the tree is
either a *turn node* - a turn start - or a *semimove node* - a partial action, some
of the mover's semimoves decided. A node is an **OR** node when A chooses there (A's
turn nodes and A's partial actions) and an **AND** node when D does. So one
half-turn is a chain of nodes of one kind: an AND turn node, its AND semimove nodes,
and at the end of each complete legal action a *submit* edge to an OR turn node -
the half-turn boundary, where the depth counter drops by one.

**How a turn decomposes - the canonical order, exactly.** At the turn start the
mover's playable boards are the *axes*, in an order fixed by the turn start (A's
turns: ascending timeline; D's turns: the boards a check rests on first, then the
mandatory ones, then the optional ones, ascending within each group - see "Evasions
per semimove"). A partial action is built in two phases:
1. **Plain phase.** Axis by axis in that order: if the axis's board has been played
   already (by a jump that landed on it), it is skipped; otherwise the mover chooses
   `Pass` or one of the board's *non-branching* moves (within the board, or a jump
   onto another axis's still-unplayed board, J4) - generated by the state machine at
   the partial state, so every move is pseudo-legal where it is made.
2. **Branch phase.** Then any sequence of *branching* moves (J3, J5; J6's twins
   included) from axes still unplayed, each generated at the partial state; and,
   whenever the obligations are met (T5), the choice to submit - legal iff every
   royal of the mover's is safe (C4).
This tree holds every legal action exactly once (as a set of moves with its creations
in order) **when the order of a turn's moves does not matter beyond the creations'
order**: under Q4 "moves after the shift allowed" (`moves_after_shift`, ruled R8,
the default since E7), activity changed by creations
alone (the count rule, `Rewinds`, no play activation) and castling allowed on
optional boards. The argument: the mover's moves visit only boards of its own colour,
and an action creates only boards of the opponent's colour, so a move's path and
landing squares never change during the action - only whether a jump onto another
axis's board lands on a playable board (J4) or, once that board is played, on
history (J6's branching twin); non-branching moves on different boards append boards
to different timelines and commute; a branch only adds an opponent-coloured board and
changes activity and the present, which no non-branching move reads under those
rules. So any legal action can be replayed with its non-branching moves first, in
axis order, and its branches after them in their creation order - every move still
pseudo-legal and the final state the same - and each such sequence is one path of the
tree. Under any other ruleset (Q4's Guide-literal reading among them, which puts the move that
shifts the present last; Q1's consecutive rule, Q3, Q6, Q21's alternative) the turn
is not decomposed: the node's children are E4's listing of the legal actions
(`list_legal_actions`, exhaustive, capped), and a capped listing makes the node
unprovable (AND) or undisprovable (OR) - never a false verdict.

**How a turn is completed incrementally, not re-solved per line.** A partial action
is a node of the tree, so its completions are searched once and shared by every line
through it - the scanner instead completed each forcing seed into a legal turn per
line ("the quietest move on each mandatory board; else E4's first legal finish"),
38-64% of its work at 100+ boards. Dead completions are cut where they arise, by
three sound tests:
- **A standing capture.** If, after some of the mover's moves, a piece of the
  opponent's on one of its playable boards can capture a royal of the mover's, that
  capture stands to the end of the action (E5a's tier 0: the mover moves only on its
  own colour's boards, boards never change, a new board appears only where there was
  void, which no existing capture crosses), so no completion is legal: the node is
  dead at once. This is C4 asked of a partial action, answered by the incremental
  royal-attack map (below) - so an AND node in check keeps only the moves that
  answer the check, the moment they are made. (Under Q18's alternative a new board
  could block a capture across void, so only a capture that crosses no void
  stands - E5a's rule.)
- **A pass that cannot be made good.** `Pass` on a mandatory board leaves it to be
  played later - by a jump onto it from an axis not yet decided (J4), or by a
  branch from it - or relieved by a creation that moves the present (T8, T12). It is
  offered only where one of those is possible: an axis later in the order has a
  jump onto it, it has a branching move or a jump (a possible J6 twin), or a relief
  is possible (the mover's next creation would be active and some jump lands
  before the present, or an inactive timeline exists - the opponent's, or since H46
  an initial one - that a creation might reactivate, T11). And a partial action dies the moment a board passed earlier is
  still unplayed, mandatory and beyond all three - no axis left to jump onto it.
  Each test over-approximates, from the moves at the turn's start: the candidate
  moves of an action are fixed there, since the mover's moves visit only boards of
  its own colour, which the action changes only by playing them.
- **E5a's tiers** (optional, `ProverOptions::legality_tiers`): the fast tiers classify
  a semimove node's moves; a move proved illegal (no legal finish at all) is not
  made.
A **canonical completion** (A's default) is the first legal leaf of the seed's
completion subtree in a fixed preference order - on an optional board `Pass` first,
then its non-branching moves from the quietest (a move within the board, capturing
nothing, by the least valuable piece; the scanner's order); on a mandatory board the
quietest move first and `Pass` last; then the branch phase, submitting as soon as the
obligations are met and C4 holds, else branches in generator order. A depth-first
search with the three tests, at most `completion_max_nodes` steps (2,000, the
scanner's fallback cap: a placeholder); a seed it cannot complete within them has no
action in the class - deterministic, so the class is well defined. It is computed
once per (turn start, seed) - when the turn node is expanded - and held by the tree.

**Evasions per semimove instead of listed whole.** An AND node never lists D's
actions: its children are one axis's semimoves. D's axes are ordered so that the
boards a check rests on come first (R1's captures in the passed multiverse: the
boards whose passed copies hold a place of one), then the mandatory boards, then the
optional ones. So the moves that do not answer a check die at the first axis - each
leaves a standing capture - and the free boards behind it are explored only below a
real evasion, lazily, in the order df-pn asks for them. E4 still decides the turn
node itself: before expanding an AND turn node, one quick escape search (the same
depth-first search as a completion, escape moves first, `escape_probe_nodes` steps)
looks for a legal action; if it finds none, E4's hypercuboid detector gives the
exhaustive verdict (`detect_mate`): checkmate - proven, distance 0, exported as
`Loss(0)` (H37: an exhaustive search that terminated); stalemate - disproven
(exported as `Draw`); a legal action - the node is expanded, that action's successor
first among its children (a duplicate of one of the semimove tree's leaves: harmless
for an AND node, and the first escape df-pn tries). At the last half-turn (no depth
left) an AND node is decided by that test alone.

**Proof and disproof numbers, thresholds.** Plain df-pn (Nagai's depth-first
proof-number search; the precedent is dlshogi's df-pn mate solver, cited, not
copied). Each node holds (pn, dn): proven (0, ∞), disproven (∞, 0), a fresh leaf
(1, 1). An OR node's pn is its children's least pn and its dn their sum; an AND
node's pn is the sum and its dn the least. `mid(n, thpn, thdn)` searches below n until
pn >= thpn or dn >= thdn: it picks the child with the least pn (OR) or dn (AND) and
gives it the thresholds `min(thpn, pn2 + 1)` and `thdn - dn(n) + dn(c)` (OR; AND
symmetrically, pn2/dn2 the second-best). The root is searched with infinite
thresholds until it is decided or the budget is spent. Sums saturate below ∞. A node
that can never be proven (an AND node whose legal actions E4 could not list within
its cap or memory ceiling) keeps pn just below ∞ and its dn: never proven, still
disprovable. A dead semimove node (no legal completion) is not an option of the
mover's: an AND-side one is proven, an OR-side one disproven. Distances: a proven
node keeps k, the half-turns its proof needs (an AND turn node checkmated now 0; a
submit edge adds 1; an OR node takes its proving child's, an AND node its longest
child's), so the root's proof is `Win(k)`, k <= d.

**The tree and the transposition table.** Each search thread keeps an explicit tree
(children made on expansion, the move or action leading to each; states recomputed
along the path as the search descends), because a node's children - one board's moves,
each applied and C4-tested, or a seed's completion - cost too much to regenerate at
every visit. Turn nodes also meet across paths and threads through a shared,
lockless **transposition table** of (pn, dn) *estimates*, keyed by the 5D Zobrist key
with the ruleset ID (H18), the node's role (A or D to move) and its remaining depth.
Solved facts never live only there (§11.3): a proof goes to the solved store and a
disproof to the disproof store the moment it is found, both keyed by position with a
full comparison on a key match, and a turn node consults both when it is created.
Read back: `Win(k)` for A with k within the depth left proves an OR node, `Loss(k)` for
D proves an AND node (proofs are real-game facts, so a proof from the scanner may
prove a df-pn node); `Win` or `Draw` for D, `Loss` for A, or a recorded disproof of the
same class at least as deep, disproves. GHI does not arise: there is no repetition rule
and every move creates a board, so the full state never recurs (H39) and the tree is
acyclic; proofs and disproofs depend only on the position and the depth left.

**Depth limiting.** A turn node carries the half-turns left. An OR turn node with r
left expands A's actions, each submit edge to an AND turn node with r - 1; an AND
turn node with 0 left is decided by the mate test alone (checkmate proven, otherwise
disproven). Proven at r is proven at any r' >= r; disproven at r is disproven at any
r' <= r; the stores and the TT key keep the depth.

**Parallelism (M13: 5 threads).** Lazy-SMP-style: every thread runs `mid` from the
root over a tree of its own; they share the TT (estimates), the solved store and the
disproof store, and a thread that decides the root stops the others. Diversity: a
helper thread breaks ties between equally good children by a per-thread hash, and
searches with thresholds raised by a per-thread factor (the "1 + ε" trick), so the
threads leave the main thread's path at different points. Soundness is unaffected: a
node is solved only from its own exact children or a store's exact fact, the TT is
never read as a fact, and the stores refuse contradictions (`std::logic_error`). One
thread is the deterministic mode (H15): with a node budget and no clock, the same
root, options and stores give the same search, node for node.

**Budget.** Work is counted in nodes, one unit for all of it: a node expanded, a move
applied, an attack-map query, a node of an E4 search the channel calls. A node
budget and a wall-clock limit (6b′: 50 ms) stop the search; what was decided before
the stop stands (it was exact), the rest is unknown.

### Incremental royal-attack maps (§11.1 part 3)

`RoyalAttackMap` (`t5d/attacks.hpp`) holds, for one colour and one multiverse, what
`royal_capture_exists` asks - can a piece of that colour, on one of its playable
boards, capture an enemy royal? - but kept as a record per *source* (a playable
board): the captures found from it, and every line from it that stopped at *void* (a
place with no board, J8), with where it stopped. A new board can only appear where
there was void, so moving to a multiverse that extends this one (`advanced`) needs:
- sources no longer playable (their board superseded) dropped;
- new playable boards scanned as new sources;
- for each new board's place, only the lines that stopped at void exactly there
  resumed from it - every other line is unchanged, because boards never change (A4)
  and a line that stopped at a piece or a board's edge cannot be reopened.
A source's record is computed once and shared, unchanged, by every map derived from
the one that made it: a board that is not played keeps its record from turn start to
turn start, and a superseded board is never scanned again - it is only crossed.
Under rules where a board's landability depends on activity or on its being history
(Q8's, Q20's alternatives) or a slide crosses void (Q18's), `advanced` recomputes from
scratch: always equal to a from-scratch scan, and the tests hold it so on random
positions and move sequences under every ruleset.

The proof channel derives each turn's maps from the previous turn's along the path:
C4 for the mover's partial actions asks the opponent's map, advanced by the boards the
action has made so far (each one an opponent source, plus the lines that stopped where
it now stands); R1's check status after an action asks the mover's map advanced by
the passed copies.

### The estimate channel (§11.1 part 1) - retired

**Retired at CHECKPOINT 4** (verification v3, §11.4: rung 1 fits its budget nowhere)
**and removed in stage E9** - `t5d/estimate.hpp`, its tests, the attack maps'
all-capture mode it used, and 6b′'s test (B). The asset map (stage E9) does not use
its code: it runs its own one-board quiescence on the move generator. What it was, as
the history 6b′'s figures below refer to:

A negamax alpha-beta over *single moves*: the side to move makes one capture (or, in
its first `threat_plies` plies, one local check: a move within a board after which a
piece on the board it creates attacks an enemy royal there), and the opponent replies
at once from the boards that move made and its other playable boards - no completion
of the turn, no obligations, no C4 test (a royal left capturable is simply captured,
for a large score). The boards a move creates must be answered on: those the reply
leaves are passed (advanced unchanged), so a threat there carries, and a royal
attacked on such a board is in local check (no stand-pat; every move of that board is
searched). Stand-pat (the static material eval) elsewhere, delta pruning (§11.1:
speculative pruning is estimate-channel only), MVV-LVA ordering, a ply limit and a
node cap. The captures came from all-capture maps advanced move by move. Its result
was the score after the sequence, the swing against
the static eval, and whether the position is tactically live (a swing of at least
`live_margin`). It is meant for rung 1 (§11.2): a per-leaf scan while the leaf waits
for its batched evaluation - which 6b′ tests.

### Experiment 6b′ (M15): `t5d-scan6bp`

The same positions as 6b (the records' turn starts and, until E9 removed test (B),
one leaf per root two half-turns of random legal play below it, made as `t5d-scan6b`
makes them), read per board-count range, never pooled:
- **the proof channel** at each root: 50 ms of wall clock, 5 threads; *decided* = proven
  or disproven. Pass: >= 50% of roots decided;
- (removed in E9 with the channel) **the estimate channel** at each leaf: its cost in
  nodes, against the rung-1 leaf budget. *Fits*, operationalised before measuring: the leaf budget is what M13's 5
  scanner threads can spend while one leaf waits for its batched evaluation -
  5 x (the channel's measured nodes per second of thread CPU time) / (§15.6's design
  evaluation rate, size-matched: 9,600/s below 420 boards, 4,600/s from 420) - and the
  channel fits a range when the 90th percentile of its per-leaf cost there is within
  that budget. The per-leaf wall time against 5 / rate is printed beside it;
- **brute force against df-pn below 50 boards**: the E5c scanner's 6b root scan (50 ms,
  5 threads; decided = 5 forcing half-turns resolved, or a proof, or an exhausted
  tree) beside df-pn's, roots decided by each.
The summary applies M15's outcome rules mechanically.

```
t5d-scan6bp [--record-list FILE --record-root DIR] [--every K] [--max-roots N]
            [--threads T] [--root-ms MS] [--brute-ms MS] [--no-brute-force]
            [--depth D] [--seeds checks|forcing|all] [--completion canonical|every]
            [--completion-max-nodes N] [--escape-probe-nodes N] [--legality-tiers]
            [--min-boards N] [--max-boards N]
            [--json FILE] [--summary FILE] [--per-position]
            [--no-moves-after-shift] [--rules ENCODING] [RECORD ...]
```

### As built: the parameters

Every one a placeholder (the spec gives the shapes, not these values), named and
with its alternatives:

| Parameter | Default | Alternatives, and why the default |
|---|---|---|
| `ProverOptions::depth` | 5 half-turns | M15's question (mate-in-3, §15.6) |
| `ProverOptions::seeds` | `Checks` - the scanner's checking moves (b) | `Forcing` (6b's whole forcing set), `All` (the literal question). dlshogi's precedent: a mate search by checks |
| `ProverOptions::completion` | `Canonical` - one action per seed | `Every` (every legal completion). 6b's unit: a forcing seed and one completion |
| `ProverOptions::completion_max_nodes` | 2,000 steps | the scanner's completion-fallback cap; beyond it a seed has no action in the class |
| `ProverOptions::escape_probe_nodes` | 256 steps | before E4's detector at an AND turn node (E5c's probe, 64 seeds, is the precedent) |
| `ProverOptions::legality_tiers` | off | E5a's fast tiers at semimove nodes; the standing-capture test already cuts what tier 0 would |
| `ProverOptions::listing_cap` | 4,096 actions | turns that do not decompose (E4's listing) |
| `ProverOptions::threads` | 1 (6b′: 5, M13) | 1 is the deterministic mode (H15) |
| `ProverOptions::tt_entries`, `max_tree_nodes` | 2^18, 2M per thread | memory bounds |

### As built: what the tests hold

- **Attack maps** (`tests/test_attacks.cpp`): every map - built, advanced, or asked
  without being built - gives check.hpp's from-scratch captures, the same list in
  the same order: on random multiverses (both colours, every ruleset), on chains of
  random extensions (boards appended, heads changed, timelines added; 11,520
  incremental advances in the default run, 31,236 lines resumed), along random
  legal play move by move with each turn's passed multiverse (R1's status equal to
  `in_check`; C4 of every partial action equal to `royals_safe`). A non-extension is
  recomputed and still equal. `long_` plays ten times as many moves.
- **The semimove tree** (`prover_semimove_tree_lists_exactly_the_legal_actions`):
  under every ruleset that decomposes, the prover's own trees - the defender's and the
  attacker's in the literal class, built by the production expansion
  (`Prover::tree_actions`: `expand_turn`, `expand_semi`, every node expanded; before
  stage E9's review the test walked a test-only copy of it) - hold exactly the
  brute-force reference's distinct successors, each replayed through the state machine
  and C4-tested, and E4's count of them; and every legal action (a move sequence,
  branches in the order made) is a path of the tree exactly once - the defender's
  witness besides (685 positions, 81,887 actions in the default run; `long_`: more and
  larger positions).
- **The canonical completion** against its definition walked independently (no
  attack maps, no pruning but the rulebook's standing-capture lemma tested from
  scratch), for every move at every position, under every ruleset, under one Q4
  reading or the other (24,549 seeds, 18,063 completed, none beyond the
  reference's cap).
- **Proofs and disproofs** on random play from the sparse starts, against exact mate
  references: the attacker's actions of the class - for the literal class (seeds
  `All`, completion `Every`) every legal action, for the default class the class's
  own actions computed independently (the scanner's check test made literal, the
  completion's definition walked) - against every legal defence, a mate at the
  horizon by E4's detector. A mate by the class's actions is a mate, so a proof is
  confirmed by its class's reference, else by the brute-force game tree within its
  distance, else by the literal reference; a disproof is the class reference's
  "no mate within d". The references list successors by the brute-force
  enumeration where it fits and by E4's exhaustive listing (another algorithm than
  the semimove tree) beyond it, within a work cap. Default run: literal class 112
  decided, 111 confirmed, 1 beyond the references; default class 74 decided, 74
  confirmed; every ruleset under both Q4 readings (listings where turns do not
  decompose; smaller caps) 2,797 decided, 2,790 confirmed, 7 beyond; no
  contradiction anywhere.
- **The `long_` runs** (`--long`, 2026-10-03, the merged tree): the semimove tree
  6,064 positions, 1,365,247 actions, all exact (18 min); the chained attack maps
  30,393 moves, 2,238 turn starts in check; random play to depth 5 under every
  ruleset (the default at full size, every alternative at a quarter; 40 min, a
  progress line per ruleset) - literal class 2,501 decided, 2,389 confirmed, 112
  beyond the references; default class 4,495 decided, 4,463 confirmed, 32 beyond;
  no contradiction.
- **H37**: for every root of a sample, budgets stepped from 0 to the full search's
  need: a decided answer only where the full search gives the same one (213 budgets in
  the default run, every one stopped unknown; `long_`: about 1,900).
- **H15**: one thread with a node budget gives the same search twice, node for node.
- **Threads**: 5 by node budget in the default run (21 decided, 20 confirmed, 1
  beyond the references); 2, 5 and 8, by node budget and by the clock (20 ms), in the
  `long_` run - with stores shared across the searches, every decided answer confirmed
  by the references and equal to one thread's where both decide.
- **Stores** (H19): a proof's root `Win(k)` and the defender's `Loss` entries are in
  the solved store; a second prover answers from it without expanding a node; a
  disproof is recorded with its depth and class, and only for its class.

### 6b′: preliminary figures (a shared machine - not the verdict)

Measured 2026-10-03 on the dev machine (AMD Ryzen 9 9900X, 24 logical processors)
while other agents built and tested in other worktrees - busy 37% during the proof
channel's phase, 36% during brute force's, 26% during the estimate channel's - so
these are not quiet-machine figures; the decisive run is a quiet machine's. Release,
g++ 16.2 `-O2`. Q4 as ruled (R8; measured before E7 reached this branch, with
`--moves-after-shift` - the default ruleset since E7); turns decompose. The 41 corpus
records: 805 roots, 780 leaves (17 roots and 12 leaves checkmated, 2 and 2
stalemated, 1 root breaking C4: not measured). Defaults throughout (the parameters
above).

| Boards | roots | (A) df-pn decided at 50 ms, 5 threads | of them: proven / disproven (with no action of the class at the root) | (C) brute force decided | leaves | (B) estimate p90 cost / budget |
|---|---|---|---|---|---|---|
| < 50 | 527 | **76.1% PASS** | 49 / 352 (256) | 74.4% | 495 | 3.63 - does not fit |
| 50-99 | 140 | **39.3% FAIL** | 22 / 33 (22) | 28.6% | 140 | 9.81 - does not fit |
| 100-199 | 97 | **23.7% FAIL** | 7 / 16 (12) | 23.7% | 100 | 23.5 - does not fit |
| 200+ | 41 | **0.0% FAIL** | 0 / 0 | 2.4% | 45 | 78.3 - does not fit |

The estimate channel ran at 154,498 nodes per second of thread CPU time, so a leaf's
budget was 5 x 154,498 / 9,600 = 80 nodes (521 us); its p90 wall time per leaf was
1.4 ms, 4.6 ms, 15 ms and 46 ms by range.

**M15's outcome, applied mechanically to these figures: 50-199 boards fail -> STOP
for a design review.** Rung 1: the estimate channel fits in no range - it stays
dropped. Below 50 boards df-pn decides more roots (401 of 527) than brute force
(392): keep df-pn there.

A second run after E7's merge, the default ruleset with no option (the same rules),
the machine busy 24% / 28% / 23%: (A) 76.7% PASS (51 / 353, 256), 40.7% FAIL
(23 / 34, 22), 23.7% FAIL (7 / 16, 12), 0.0% FAIL; (C) 74.8%, 30.7%, 24.7%, 2.4% -
df-pn 404 against brute force 394 below 50 boards; (B) at 228,153 nodes/s (a budget
of 119 nodes) p90 cost/budget 2.46, 6.64, 15.9, 53.1 and p90 wall 0.91, 3.6, 9.4,
31.6 ms - the same outcome on every count.

What the decisions rest on, and how they move (information, the same machine):
- **Most disproofs are a root with no checking action at all** (256 of 352 below 50
  boards; 22 of 33, 12 of 16 above): under `Checks` seeds "no mate by checks" is then
  immediate. Where the attacker has checks, df-pn rarely settles 5 half-turns above
  50 boards.
- **The class.** `Forcing` seeds: 36.6% / 22.9% / 7.2% / 0.0%. The literal question
  (`All`, `Every`): 9.3% / 2.9% / 1.0% / 0.0% - almost only proofs. `Checks` with
  `Every` completion: 72.3% / 26.4% / 16.5% / 0.0%.
- **Time.** At 500 ms (ten times the budget) and 5 threads: 50-99 45.0%, 100-199
  34.0%, 200+ 2.4%; at 1 s on one thread: 50-99 42.1%, 100-199 34.0%. The shortfall
  is the scale of the trees, not a constant factor.
- **The estimate channel's depth.** One ply (a capture or threat, no reply): p90
  cost/budget 0.20 / 0.53 / 0.99 / 2.99 - fits below 200 boards, but a capture with no
  reply is no exchange (76-100% of leaves "live"). Two plies: 1.23 / 2.97 / 7.0 /
  19.6 - fits nowhere. At every depth the root's own work (both sides' capture maps,
  the threats' move generation) grows with the boards: at 200+ boards even one ply
  costs 1.7 ms at p90 against 0.52 ms.
- **Where the work went** (nodes, every thread): attack-map queries 84-86% of the
  units - the standing-capture tests of partial actions (1.2 million partial actions
  cut below 50 boards, 1.0 million at 50-99) - E4's searches 0.0-0.3%; canonical
  completions outnumber node expansions two or three to one (six to one at 1 s).

The quiet-machine run (from the repository root, after building):

```
build/engine/t5d-scan6bp --per-position --record-list build/corpus.txt \
    --json build/6b/6bp-quiet.json --summary build/6b/6bp-quiet.txt
```

(`build/corpus.txt` lists the 41 records, one path per line, relative to the
repository root. The default ruleset is Q4 as ruled, R8; `--moves-after-shift` is
accepted and changes nothing.) It took 45 s on the shared machine: (A) at most
50 ms a root, (C) the same, (B) a few seconds.

## Stage E9: verification v3 - Guard, Hunt and the asset map (CHECKPOINT 4)

6b′ failed from 50 boards up (CHECKPOINT 4): exact settling - proving a forced result
or proving there is none - works only in small positions. The verification layer is
v3 (spec §11.4): exact rules always; the search's own solver for proofs (H41, gated);
**Guard** and **Hunt** at every size, each needing only the cheap half of settling for
its purpose; full settling scans only under 50 boards; the offline audit lane and the
league above. The user's rulings for this stage: carry E8's df-pn and incremental
attack maps into Hunt and Guard; drop the per-leaf estimate channel; build Guard (H42)
and Hunt (H43); keep the asset map (§11.2 rung 4) - "the scanner ladder" is the
reduced ladder of Guard, Hunt and the asset map; record Hunt's ground truth once; run
6b″ (M17). Not built: C25, the NNUE (held open, H6), the retired rungs 1, 3 and 5 (M5).
The components have clean interfaces; wiring them into the search is the next stage,
after the search core (E6a) merges.

| Part | Where |
|---|---|
| The prover's additions: prove-only, the literal class's single entry, the last-turn check filter, the proof's line and certificate, a depth override | `include/t5d/prover.hpp` |
| Certificates checked against E4 alone | `include/t5d/certificate.hpp` |
| Guard (H42) | `include/t5d/guard.hpp` |
| Hunt (H43) | `include/t5d/hunt.hpp` |
| The superseded-board asset map (§11.2 rung 4) | `include/t5d/assets.hpp` |
| Hunt's ground truth: the recorder, the file and its pin | `tools/t5d_hunt_truth.cpp`, `tools/hunt_truth.hpp`, `testdata/hunt-truth/` |
| Experiment 6b″ (M17) | `tools/t5d_scan6bpp.cpp` |

The per-leaf estimate channel is gone (rung 1 fits its budget nowhere): its code, its
tests, the attack maps' all-capture mode it used and 6b′'s test (B). No piece of it is
kept - the asset map runs its own one-board quiescence.

### The prover's additions

- **Prove-only** (`ProverOptions::prove_only`, Hunt): the search stops at its budget
  and claims nothing - a root it exhausts is reported Unknown (`exhausted`), never
  Disproven; it neither reads nor writes the caller's disproof store (no disproof is
  recorded or shared, none from outside prunes it); its own search's disproofs stay in
  a private memo, emptied at each search, so that a transposition it has exhausted is
  not searched again. Its thresholds stay df-pn's usual two. Measured against
  proof-number thresholds alone (a child's disproof threshold infinite - the reading
  "it never spends time disproving" suggests; kept as
  `prove_only_pn_thresholds`): on the tests' positions the usual thresholds found
  every mate the full search found at the same cost (26 of 26), proof numbers alone
  25 for 12% more nodes. Prove-only never spends time disproving in the sense that
  matters: it is never run on to settle a root.
- **The literal class enters the attacker's semimove tree once** (seeds All,
  completion Every): every legal action once, not once per its first move.
- **The last-turn check filter** (`check_filter`, OFF by default): mate needs check,
  so at the attacker's last half-turn a partial action that can no longer give check
  could be cut. A move is check-capable when, played alone from the turn start, it
  gives check in a passed multiverse that passes every opponent head (a superset of
  R1's sources) or makes a line stop at void where another of the attacker's moves
  could make a board (attack maps' new `VoidStop`s; a line through several new boards
  is opened by the first). Off where maps recompute (Q8, Q18, Q20) or a capture stands.
  **J6's twins:** a non-branching jump onto another of the attacker's boards becomes
  a branching move once that board has been played, landing on a new timeline the
  turn start's maps never saw - so while any such jump exists, a check-capable branch
  is assumed left to make (and a move the turn start does not offer is
  check-capable). E9's first filter missed this: the review found a mate in one by a
  twin that the filter cut - a false disproof, and in Guard a false "safe" (both
  positions are now tests). It was off by default already.
  On 6b's first 120 roots it cut almost nothing (none under 50 boards) and cost more
  than it saved - in positions of several timelines a new board's passed copy sends
  lines across the timelines beside it, so nearly every move is check-capable - hence
  off; its test switches it on. Whatever the filter, a defender's turn at the horizon
  is decided "not mated" from R1's status alone when it is not in check.
- **The proof's line** (the attacker's actions, the defender's longest defences, to
  the mate) and, on request (`record_certificate`), its **certificate** - every
  attacker position the prover proved by expanding it, in any thread and any of its
  searches, with its proving action (recorded as each proof is exported), so that a
  proof read back from the solved store is certified too. A proof read from a store
  another prover filled cannot be certified; the check names the missing position.
- **An attacker with no legal action** (`attacker_probe`, on): an attacker's turn is
  first asked E4's question - a quick search for one legal action, then E4's
  exhaustive detector where it finds none - so a mated or stalemated attacker is
  decided at once instead of by exhausting its semimove product (the review's 32-board
  position: E4 decides it in a few hundred nodes; the tree had not in 2M). Efficiency
  only: a test checks that it changes no answer.
- **A disproof's class** carries every option that shapes the attacker's class
  (`disproof_shape`: seeds, completion, the canonical completion's cap and values, the
  branch threat, promotions, E5a's tiers, the check filter): provers with different
  options never share a disproof.
- `prove(root, budget, depth)`: one prover, its TT and stores, at several depths.

### Certificates checked against E4 alone (`t5d/certificate.hpp`)

A mate is checked with none of the prover's machinery: the attacker's moves come from
the certificate, replayed through the state machine (pseudo-legal where made, T5, C4);
the defender's every legal action from E4's exhaustive listing (hypercuboid); every
checkmate by E4's hypercuboid detector and, where it finishes within its budget, the
naive one, which must agree. A stalemate, a defender not mated at the horizon, an
illegal move or a detector disagreement refutes; a missing position makes it
incomplete; a budget leaves it undecided - never confirmed on a budget stop (H37).

### Guard (H42; `t5d/guard.hpp`)

`guard(state, action, budget, options[, hook]) -> {completed_depth, refuted, line}`.
The chosen action - complete, legal, about to be submitted; under the ply split it is
checked at its submission, since the opponent moves only after it (a reading: spec
issue 97) - is submitted, and at the opponent's turn start Guard asks, depth by depth from
1 under one budget (nodes and/or wall clock): can the opponent force checkmate within
d of *their* actions (2d − 1 half-turns), whatever we answer? The prover runs with the
opponent as attacker and the literal class - every opponent action, once each - and
our turns as AND nodes decided by one refuting reply: a quick escape search, E4's
detector only where it finds none. Depth 1 is "for every opponent action that leaves
us in check, one legal escape exists": existence, never our full listing. Exact at
every depth that completes ("safe to depth d"); a refutation is a proof (H37) with the
opponent's forcing line, and rejects the move - the search re-searches from §8.6's
reserve (`GuardHook`, wired with the search). Where turns do not decompose, the
opponent's actions are E4's listing, up to 65,536. At most 50 depths (99 half-turns,
inside the prover's limit): every depth Guard reports is the depth searched, and a
refutation inside a depth already reported safe throws - never both in one result.

**What Guard's depth 1 costs.** "Existence, not the full listing" removes *our*
listing, not theirs: to say "safe" Guard must refute every opponent action, and the
opponent's actions are a product over their playable boards (one choice per board,
pass included where optional). The semimove tree enumerates that product once each,
cutting the opponent's illegal partial actions at once and deciding each of our turns
by R1's status (not in check: done) or one escape - but the product itself remains.
Guard completes depth 1 wherever the opponent has few boards to play; with several, no
exact method short of enumerating their actions is known to this design (spec issue
99).

### Hunt (H43; `t5d/hunt.hpp`)

df-pn in prove-only mode at the root: our forced mate within 5 half-turns (6b″) by our
forcing class (the prover's placeholder default: checking seeds, canonical
completions), against every legal defence. It stops at its budget and claims nothing.
A found mate is a proof, returned to the caller and never exported: no proof may enter
a permanent solved store until the generator passes the gate (H23), the search's
soundness fixes land and the proof channel is signed off (H41). `HuntOptions::export_proofs`
is that switch, enforced as the search's `proof_store` is - Hunt refuses to be built with
it on (`std::invalid_argument`); the export itself is not built.
Soft priors behind a flag (`suggest`, OFF): without a proof, the root's open actions,
least proof number first, weighted by 1/pn - §12.2 binds their use: a position where a
suggestion steered the search is excluded from the policy targets (or its steered
visits recorded and down-weighted - §12.2's open choice); that bookkeeping lands with
the search wiring.

### The superseded-board asset map (§11.2 rung 4; `t5d/assets.hpp`)

A superseded board is a travel destination, never a source, so its value as a branch
target is judged once, for the traveller - the colour to move on it: its material
balance (H6's v1 values) and a small fixed quiescence on the board alone (the traveller
to move; stand-pat, captures most-valuable-victim first, every move of a side in check
there, a royal capture worth `royal_value`; 4 plies, 128 nodes), on the engine's own
move generator. Flagged at `flag_margin` (300); boost `max_boost` (0.25) ×
min(1, value / `boost_scale` (900)). Once per board, ever: cached by the board's
content (content hash and a full comparison) and its colour to move, for the map's
ruleset; deterministic. Output for the search: a standing prior boost on travel moves
into flagged boards (`boost()`) - an estimate, never a bound (H2), under §12.2's
exclusions like any steering.

### Hunt's ground truth (ruling 5; `t5d-hunt-truth`, `testdata/hunt-truth/`)

One long offline search per 6b root - the same roots, `tools/sixb_roots.hpp`: the
records' turn starts, every one, distinct by position - recorded once. Per root, the
prover with node budgets, one thread and fresh stores (deterministic; roots run in
parallel), twice within 5 half-turns: the **literal question** (every first move,
every completion) and the **default class** (Hunt's own). The truth is the **union of
their proofs**: the literal question is the broadest there is, and the default class's
long search makes sure Hunt's own mates are in it - Hunt's 50 ms run is judged against
mates it did not have to find in its own class. Each class's status is kept: proven,
exhausted (no mate in the class) or unknown (budget). Every mate's certificate is
checked against E4 alone before the file is written (a refuted one stops the
recorder).

THE BUDGET, from the curve (`--curve 8`: every 8th measured root, 101 roots; mates
found against the node budget - a df-pn with a node budget is deterministic, so a mate
found at n nodes is found by any budget ≥ n):

| Nodes | 1,953 | 3,906 | 7,812 | 15,625 | 31,250 | 62,500 | 125,000 | 250,000 | 500,000 | 1M | 2M | 4M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Literal class | 5 | 5 | 6 | 6 | 6 | 6 | 6 | 7 | 8 | 8 | 8 | 8 |
| Default class | 5 | 6 | 7 | 7 | 9 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| Union | 5 | 6 | 8 | 8 | 9 | 11 | 11 | 11 | 12 | 12 | 12 | 12 |

(Mates found among the 101 roots; at 4M nodes the literal class had exhausted 6 roots
and the default class 57, and 87 and 33 were unknown; 608 s on 6 workers, a shared
machine.)

No class found a new mate past 500,000 nodes; the truth gives each class 4,000,000 -
eight times the last new mate's need. The literal class often stops earlier at its
tree cap (2,097,152 nodes: 42 of the curve's 202 searches) - also past the curve's last
new mate.

THE FILE, `testdata/hunt-truth/hunt-truth.tsv` (format: `tools/hunt_truth.hpp`): a
header naming the engine commit, the rules, the search with its classes and budgets,
the depth and the cross-check; per root its record - by content: "sha256:" and the
SHA-256 of its bytes with CRLF read as LF, since engine data may name no path under the
oracle harness (its boundary check refuses one) - position index, H30 fingerprint,
board count, verdict (mate with its distance, line and finder / none /
skipped), each class's status and nodes, and the cross-check. THE PIN,
`testdata/hunt-truth/MANIFEST`: the file's SHA-256. The 6b″ tool refuses a truth whose
bytes do not match it or whose roots are not the records' (record, index,
fingerprint); the engine's test refuses a changed byte and checks the committed file.

THE PROVENANCE, `testdata/hunt-truth/provenance.tsv` (stage E9's review): what the truth
was recorded from, beyond its header - the truth's SHA-256; the rules by **every option's
value** (`absolute_rules_text`: the header's `describe()` and encoding are relative to the
defaults of their day, which a later change of a default would silently reread); the
**code signature** of the build that recorded it (`cmake/code_signature.cmake`: the
SHA-256 over the rules engine, E4, the prover, the certificates, the record reader and
the recorder - `T5D_TRUTH_SOURCES` - each read with CRLF as LF); and the **records' content
ids in their order** (the order decides the roots' ids). The MANIFEST pins both files;
a reader refuses either changed, a provenance naming another truth, or roots out of its
records' order; the 6b″ tool also refuses a record list whose content ids, in order, are
not the provenance's, and rules that are not its rules. The engine's test pins the
committed truth's SHA-256 and the default rules' full text.

THE RECORD LIST, `build/corpus.txt` (untracked; one path per line, relative to the
repository root): every `[[record]]` of `oracles/corpus/manifest.toml`, in the file's
order, at `oracles/third_party/<source>/<path>` (a `local` record: at `<path>`) - 41
records. The provenance pins it: a list that differs in any record or in the order is
refused. (The engine names no path under the oracle harness; this rule is how a reader
rebuilds the list, and the content ids are how a tool checks it.)

Never edited by hand; re-recording is a reviewed change:

```
build/engine/t5d-hunt-truth --record-list build/corpus.txt --literal-nodes 4000000 \
    --default-nodes 4000000 --workers 6 --out engine/testdata/hunt-truth --engine <commit>
```

(`--provenance-only --record-list build/corpus.txt --out engine/testdata/hunt-truth`
writes the provenance of a truth already recorded: it rebuilds the roots from the records
given and refuses unless they are the truth's - record, index, fingerprint - and its
rules are the ones given; its code signature is then "unrecorded".)

THE TRUTH as committed (`testdata/hunt-truth/`, sha256
`ce62e5f8c52e0d53b1519ef145aae00760d84a4b271b9e9643e5fcab02cf1a73`; recorded at
`1d2d0d7` on 2026-10-03, a shared machine, 5,097 s on 6 workers): 825 roots, 805
measured (20 skipped as 6b skips them), **125 mates** - every one confirmed by its
certificate against E4 alone. The roots are 6b's exactly, the records read raw as 6b
and 6b′ read them: cwmtt/many2's export-form text gives an engine tool no action (the
gate converts it for every source; that conversion is the harness's), so many2 adds
only its starting position. A first recording (at `832aa30`, the records named by
path - which the boundary check refuses) gave every verdict, distance, finder, class
status and cross-check the same; 25 roots' node counts moved with H46's relief fix.

| Boards | measured roots | mates | of them: mate in 1 / 3 / 5 half-turns |
|---|---|---|---|
| < 50 | 527 | 71 | 52 / 10 / 9 |
| 50-99 | 140 | 33 | 26 / 6 / 1 |
| 100-199 | 97 | 20 | 20 / 0 / 0 |
| 200+ | 41 | 1 | 1 / 0 / 0 |
| all | 805 | 125 | 99 / 16 / 10 |

Found by both classes 88, by the default class alone 32, by the literal question alone
5. The literal question proved 93 roots, exhausted 35 and ran out of its budget (or tree)
at 677; the default class proved 120, exhausted 450, ran out at 235. A "none" is "no mate
found within the budget", never "no mate exists".

### Experiment 6b″ (M17): `t5d-scan6bpp`

```
t5d-scan6bpp --truth DIR [--record-list FILE --record-root DIR] [--threads T]
             [--guard-ms MS] [--hunt-ms MS] [--guard-depth D] [--hunt-depth D]
             [--no-crosscheck] [--check-positions N] [--no-truth]
             [--max-roots N] [--min-boards N] [--max-boards N]
             [--guard-nodes N] [--hunt-nodes N] [--check-filter] [--no-attacker-probe]
             [--json FILE] [--summary FILE] [--per-position] [--rules ENCODING] [RECORD ...]
```

6b's roots, per board-count range (< 50, 50-99, 100-199, 200+), never pooled, 5
threads, 50 ms each - pre-registered, unchanged:
- **Guard** at each root for the **chosen action = the action the record plays next**
  (a reading: the record's move is the move a search would be about to play; a root
  whose record plays nothing after it, or whose next action is illegal - C4 - is
  reported and not counted). Passes in a range where **depth 1 completes** - "safe to
  depth 1", or a refutation at depth 1 - in ≥ 95% of its roots; the depth reached per
  root and its distribution are reported.
- **Hunt** at each root. Passes in a range where it finds ≥ 50% of the ground truth's
  mates there; a range with no truth mate has no verdict; Hunt's mates the truth lacks
  are flagged.
- Per range: adopt Guard / adopt Hunt / not, with §13's note where each fails ("the
  chosen move goes unguarded in that range"; "proofs there come only from the search's
  solver"). No outcome stops anything.
- Every refutation and mate is cross-checked against E4 alone, outside the timed runs,
  and **only confirmed proofs count** (`tools/sixbpp.hpp`, where tests hold the counting
  and the thresholds - exactly 95% and exactly 50% pass): a refutation at depth 1 or a
  Hunt mate whose certificate is incomplete or undecided is reported apart and not
  counted; a proof E4 **refutes withholds every verdict** (a prover that claims one false
  mate is trusted nowhere until fixed); without the cross-check there is no verdict.
  A Hunt verdict resting on fewer than 10 truth mates is flagged **too few to read** (the
  pre-registered rule still gives it; 200+ has one).
- **The configuration is Guard's and Hunt's as they ship** - the prover's check filter
  off, the attacker probe on - and CONDITIONS and the JSON list every option of both
  (and the prover's under each). `--check-filter`, `--no-attacker-probe`, node budgets
  (`--guard-nodes`, `--hunt-nodes`: deterministic at `--threads 1`), other threads or
  budgets and subsets are **information runs**: the summary says no verdict is read from
  them. (Before the review the tool ran with the filter on - its default then - so the
  first preliminary figures below measured a configuration Guard does not ship.)
- A root Guard does not count says why, truthfully: its record ends there, or goes on
  with an action the engine cannot read (`PgnRecord::unreadable`) or play; the records'
  notes say "ok" only for a record read and played to its end. The roots stay 6b's.

### 6b″: preliminary figures (a shared machine - not the verdict)

2026-10-03, the merged branch (main at `621cb90`: E6a's search core in), AMD Ryzen 9
9900X, 24 logical processors, busy 26% during Guard and 27% during Hunt (other agents'
work and this tool's own threads); Release, g++ 16.2 `-O2`; the default ruleset (R8);
the committed truth. 805 measured roots (17 checkmated, 2 stalemated and 1 breaking C4
not measured); Guard counted 788 (16 roots whose record plays nothing after them, 1
whose next action breaks C4). Wall: Guard 38 s, Hunt 19 s.

| Boards | Guard: depth 1 completed (of roots) | completed depth 0 / 1 / 2 / 3 / 4 / 8 | refuted (at depth 1) | Hunt: truth mates found | Verdict |
|---|---|---|---|---|---|
| < 50 | **64.9%** of 513 - FAIL | 219 / 269 / 14 / 3 / 1 / 7 | 42 (39) | **70.4%** (50 of 71) - PASS | adopt Hunt; Guard not adopted |
| 50-99 | **15.1%** of 139 - FAIL | 122 / 15 / 0 / 0 / 0 / 2 | 4 (4) | **63.6%** (21 of 33) - PASS | adopt Hunt; Guard not adopted |
| 100-199 | **5.2%** of 96 - FAIL | 94 / 1 / 0 / 0 / 0 / 1 | 3 (3) | **35.0%** (7 of 20) - FAIL | neither |
| 200+ | **0.0%** of 40 - FAIL | 40 / 0 / 0 / 0 / 0 / 0 | 0 | **0.0%** (0 of 1) - FAIL | neither |

(Completed depth 8: the record's move left the opponent no legal action.) Hunt found no
mate the truth lacks. Every refutation and mate was cross-checked against E4 alone:
Guard's 48 confirmed, 1 incomplete (a node solved from another thread's tree); Hunt's 78
confirmed - none refuted. (A run before the merge, with the first recording: Guard
64.7 / 15.1 / 5.2 / 0.0%, Hunt 69.0 / 63.6 / 30.0 / 0.0% - the same verdicts.)

**M17's verdicts, applied mechanically: adopt Hunt below 100 boards; adopt Guard
nowhere.** §13's notes: the chosen move goes unguarded in every range; from 100 boards
up, proofs come only from the search's solver. No outcome stops anything.

What the verdicts rest on (information, the same run):
- **Guard's cost is the opponent's action product.** Depth 1 must refute every
  opponent action, and their actions multiply across their boards: the median root
  spent its 50 ms on 24,000 nodes under 50 boards; our turns were decided "not in
  check" 301,000 times and by an escape search 60,000 times there - and depth 1 still
  completed in only 65% of roots. Where the opponent plays one or two boards it
  completes at once; with more, not in 50 ms. The check filter does not change this
  (on: no better, its construction costs a map advance per move); Lazy SMP's helpers
  re-search the same product (an exhaustive refutation shares little through the
  stores).
- **The refutations are real:** 49 of the records' next moves (6%) allow a forced
  mate - these are test records, many played into checkmate - each confirmed by E4.
- **Hunt** finds the shallow mates below 100 boards. At 100-199 boards every one of the
  truth's 20 mates is a mate in one, and Hunt's own class found all 20 in the long
  search - but in 50 ms (a median 23,000 nodes at 5 threads) Hunt reaches 7: its root
  expansion completes every checking seed (a canonical completion each) and tests each
  defence before it descends, and at that size that does not finish. A root expanded
  lazily - a child tested as it is made - is the obvious remedy (not built: the
  pre-registered Hunt is what was measured).

The quiet-machine run (from the repository root, after building; the truth is
committed):

```
build/engine/t5d-scan6bpp --truth engine/testdata/hunt-truth --record-list build/corpus.txt \
    --per-position --json build/6b/6bpp-quiet.json --summary build/6b/6bpp-quiet.txt
```

(`build/corpus.txt`: the 41 records, one path per line, relative to the root - the
same list, in the same order, the truth was recorded from; the records are named by
content, CRLF read as LF, so the checkout's line endings do not matter; the tool
refuses a truth whose bytes or roots do not match. About a minute: Guard 38 s, Hunt 19
s, then the cross-checks.)

### As built: the parameters (placeholders, named, with their alternatives)

| Parameter | Default | Alternatives, and why the default |
|---|---|---|
| `ProverOptions::prove_only_pn_thresholds` | false: df-pn's usual thresholds | true: proof numbers alone - found fewer mates for more nodes |
| `ProverOptions::check_filter` | false (Guard, Hunt and the 6b″ tool alike) | true - cost more than it saved on 6b's roots; sound only since the review's J6-twin fix, after which it cuts almost nothing |
| `ProverOptions::attacker_probe` | true | false - stage E9's first prover (measurement only; the same answers) |
| `GuardOptions::max_depth` | 4 (the opponent's actions), clamped to 1..50 | any; the budget normally stops it first |
| `GuardOptions::threads` | 1 (6b″: 5, M13) | 1 is deterministic (H15) |
| Guard's listing cap (turns that do not decompose) | 65,536 | the prover's 4,096 (a capped listing never completes a depth) |
| `HuntOptions::prover` | depth 5, checking seeds, canonical completions | the literal class (9/3/1/0% decided at 50 ms in 6b′); `Forcing` seeds |
| `HuntOptions::export_proofs` | off - Hunt refuses it on | on only once the gate passes and the proof channel is signed off (H41, H23); the export is not built |
| `HuntOptions::suggest`, `max_suggestions` | off, 8 | on with §12.2's exclusions in place |
| `AssetOptions::qsearch_plies`, `qsearch_nodes` | 4, 128 | a small fixed budget (§11.2's table) |
| `AssetOptions::flag_margin`, `boost_scale`, `max_boost` | 300, 900, 0.25 | a minor piece; a queen; a prior weight |
| `AssetOptions::values`, `royal_value` | the scanner's `PieceValues`, 20,000 | H6's v1 (build-log issue 56) |
| The truth's classes | literal ∪ default | literal alone (cheaper to state, misses Hunt's own deep mates); add `Forcing` seeds |
| The truth's budgets | 4,000,000 nodes per class, tree cap 2,097,152 | from the curve: no new mate past 500,000 |
| 6b″'s chosen action | the record's next action | the search's move (needs the search: next stage) |

### As built: what the tests hold

Default run (2026-10-04, after stage E9's review; every count from it). E9's 47 default
tests take 84 s on a shared machine (one at a time; the review measured 1,038 s before
the heavy cases moved to `long_` tests - each default test has a `long_` twin at the
old size where it was cut):
- **The prover's additions** (`tests/test_certificate.cpp`): the check filter changes no
  verdict (34 positions decided both ways; `long_`: both Q4 readings, 130); it **keeps
  every legal action that leaves the defender in check** - its tree against the
  reference's actions and R1's status after each, on 100 positions: 2,296 checking
  actions kept, 2 of them only through a line stopped at a void another move fills (a
  position built for it), 1 action cut; J6's twins counted (the review's two positions,
  filter on and off, 1 and 5 threads, the certificate confirmed, the game tree
  agreeing). Prove-only never claims a disproof or records one and proves every mate the
  full search proves at its budget (13 of 13; proof numbers alone: 12). Every proof's
  line replays to E4's checkmate and every certificate is confirmed by E4 alone (30 of
  30); certificates at 5 threads, from a first and a second search of one prover (the
  root read from the store), all confirmed (58); a certificate without the attacker's
  answer to any one defence is refused (67 doctored, from 9 certificates), as is one
  asked of too short a distance, or an empty one. An attacker with no legal action is
  decided at once (the review's 32-board position, under 20,000 nodes); the attacker
  probe changes no answer (69 positions, 10,053 attacker turns probed).
- **Guard** (`tests/test_guard.cpp`): depth 1 against brute force - every opponent
  action (the brute-force reference, else E4's listing) and E4's verdict after it - 22
  safe and 6 refuted exactly where the reference says so, every refutation's line
  ending in our checkmate, at 1 thread and at 5 (`long_`, both Q4 readings: 134 and
  39). Depth 2 and every refutation against the restricted game and the game tree: 20
  depths and 6 refutations confirmed, at 1 and 5 threads. 112 budgets below the full
  search's need never claim more (H37); 78 results under small budgets, 65 of them
  stopped, every one reporting only depths it searched. The review's twin position: a
  refutation at depth 1 whatever the filter, at 1 and 5 threads. At most 50 depths.
  One thread deterministic; an illegal action refused; the hook called exactly on each
  refutation.
- **Hunt** (`tests/test_hunt.cpp`): the constructed mates proven at their distances,
  lines ending in E4's checkmate; the export switch refused (H23); 287 budgets below the
  need with a mate only where the full hunt has it (11 mates, each confirmed); at 5
  threads, by nodes and by the clock, 26 mates in 80 searches, each where one thread's
  full hunt has it, at its distance, its certificate confirmed by E4 alone; one thread
  deterministic; suggestions only behind the flag, ordered by proof number, weights
  summing to 1.
- **The asset map** (`tests/test_assets.cpp`): along random play from every sparse
  start, each superseded board assessed once (assessments equal the cache's size; a
  second refresh assesses nothing), heads never; two maps identical; a queen-up board
  flagged and its travel move boosted, an even one not; a royal the traveller can take
  worth `royal_value`.
- **The ground truth** (`tests/test_hunt_truth.cpp`): SHA-256 against FIPS 180-4's
  vectors; the file and its provenance round-trip; a valid file under a wrong MANIFEST,
  a changed byte that still parses, a provenance naming another truth or other records
  or the roots out of its order, and records given in another order are refused; the
  rules are recorded by every option's value (each option changed alone changes the
  text); the committed truth is the one pinned by its SHA-256, matches its MANIFEST and
  is well formed (825 roots, 125 mates, each within 5 half-turns and cross-checked; 41
  records).
- **6b″'s counting** (`tests/test_sixbpp.cpp`): the board-count ranges and the
  thresholds at their edges (exactly 95%, exactly 50% pass; one root fewer fails); only
  proofs E4 confirmed count; a refuted proof withholds every verdict; no cross-check,
  no verdict; "too few to read" below 10 truth mates; the option lists name every
  option, Guard's and Hunt's prover options as they run. 6b's roots from records
  written there: a record that ends, one that goes on with an action the engine cannot
  read, one with an illegal move - each last root labelled truthfully; content ids
  read CRLF as LF; duplicates dropped; a missing record list refused.
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
"The C API for games"). Stage E7 adds `t5d_layout(type, field, &out)`: the size of each
public struct (`t5d_fingerprint`, `t5d_info`, `t5d_move`, `t5d_expand_options`,
`t5d_game_summary`, `t5d_expansion`) and the offset of each of its fields, as the
library was compiled. The Python binding (`python/t5d/engine/capi.py`) checks every
struct it mirrors against it when it loads the library, and refuses a library whose
layout differs (or that lacks `t5d_layout`): a binding and a library from different
stages would otherwise let the library write past a Python buffer - as an expansion
did when stage E7 appended three fields to `t5d_expansion`.

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
- **A single move's legality** (stages E4, E5a; build-log spec issues 28, 46): a
  move is legal when a legal action begins with it (`legal_moves`,
  `classify_moves`); the expansion offers those only behind
  `ExpandOptions::legal_moves`, off by default until the spec says which moves it
  offers. Under a budget a move may be left unknown - kept, marked
  (`Expansion::move_legality`), never called illegal without proof. How a search
  treats it: stage E6a builds issue 46's proposal (proposed spec edit 43), parts
  (a)-(f) each a parameter (see "Stage E6a", "How legality enters"); the spec still has
  to adopt it.
- **Every search has a memory ceiling** (stage E4): reaching it is `unknown` (H37),
  and the gate's run ends `error` - never a verdict, and never growth without bound
  on a shared machine.
- **Check status passes the active boards in the present whatever Q5 says** (R1's
  words: "all active boards in the present"): under Q5's alternative, which makes
  boards ahead of the present mandatory too, they are still not passed.
- **Parse mode stays pseudo-legal** (stage E4): it was introduced to compare what was
  set up before the check ruling, and keeps following a record past an action that
  leaves a royal capturable; full mode refuses that action (`cannot_submit`).
- **The main search's placeholders and readings** (stage E6a; provisional build-log
  spec-issue numbers, in the order the report proposes them):
  75. submit's prior share when the evaluator gives no submit logit
      (`submit_share` 0.5; D8's network logit is stage N5);
  76. the whole-turn action prior from its moves' logits - their sum (default) or mean
      (`action_prior`; §8.2 does not say);
  77. the secure child's lower bound `Q - A / sqrt(n)`, A = 1 (`secure_child_a`);
  78. "the visit threshold for solved-node selection" read as a per-child visit cap,
      8 (`solved_visit_cap`);
  79. FPU applied to unvisited *instantiated* children (§8.4 says uninstantiated ones;
      with widening, uninstantiated children are never scored at all);
  80. LCB's z = 3, its floor 15% of the most visits, and the optional categorical
      term's weight 0 (`lcb_z`, `lcb_floor`, `lcb_categorical_weight`);
  81. E2's and E3's formulas and constants (the clamps, `variance_reference` 0.4,
      `variance_prior_weight` 2, the scale bounds, E3's cap 2);
  82. issue 46's policy as built, (a)-(f) (above);
  83. the evaluation cache shared across self-play's fresh searches (C14 says a fresh
      tree; the cache is state-keyed) and purged of states behind the root;
  84. the material evaluator's piece values, now the scanner's placeholders (issue 56);
  85. the whole-turn arm's missing end-of-turn token (D8; not built);
  86. the whole-turn listing's cap, now 256 for memory (`max_actions`);
  87. (settled: H41 rules the in-tree solver part of the proof channel - its
      propagation off by rule; no spec issue);
  88. MCGS identity: the state AND whether an action is in progress (§8.5's "values
      depend only on the state": proposed spec edit 61).
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
