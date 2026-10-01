# Checkpoint 1: check disagreements by cluster, and three traces

Harness-side analysis of the evidence in `raw.json`. Nothing tracked was changed, and nothing here is adjudicated. The seven clusters are triage groups; they pose four distinct rule questions, and Q1 bears on 18 of their positions. Rule IDs are those of `docs/rules/README.md`. Spec 11.0: "Triage by cluster; settle by rulebook, never by vote." The full data, with every capture classified, is in `clusters.json`; the scripts of this final pass are in `final/`, and the first pass is kept in `cluster_work/` and `final/previous/`. What changed from the first pass, finding by finding, is under "Review findings" at the end.

## Summary

- **78 in_check positions, 217 captures, all one kind of source.**
  - Every capture ftxi reports starts on a board where the **side to move** is to move. That is the playable (head) board of a timeline the side to move may play but need not.
  - ftxi advanced that board to its phantom; 5d-chess-js did not.
  - No capture comes from a board on which the attacker is already to move, and none from a history board.
  - An enumerator written here from `docs/rules` finds exactly ftxi's 217 captures with every mover head passed (reading 5) and none with only the mandatory boards passed (reading 2). That validates this analysis's code; it is not evidence about the rules.
- **Where the attacking board lies.** The positions split by the timeline of the capture that decides each one:
  - **69**: an **active** timeline whose board lies **ahead of** the present;
  - **9**: an **inactive** timeline whose board lies **behind** the present.
- **Seven clusters.** The geometry of the deciding capture splits those two groups further:

| Cluster | Attacker's board | Capture | Positions | Rule question | Representative |
|---|---|---|---|---|---|
| CHK-A1 | active timeline, ahead of the present | same board | 40 | RQ1 | `cwmtt/silly@28` |
| CHK-A2 | active timeline, ahead of the present | cross-board onto a royal on a history board; the branch would be active | 5 | RQ1 (settled by it) | `notation/Bg2@19` |
| CHK-A3 | active timeline, ahead of the present | cross-board onto a royal on a history board; the branch would be **inactive** | 17 | RQ2 | `cwmtt/niceAndrey@57` |
| CHK-A4 | active timeline, ahead of the present | cross-board, non-branching, onto a royal on a board where the attacker is to move | 2 | RQ1 (settled by it) | `cwmtt/silly@16` |
| CHK-A5 | active timeline, ahead of the present | cross-board onto a royal on **another optional board** (two boards passed) | 5 | RQ1 (settled by it) | `cwmtt/niceAndrey@41` |
| CHK-I1 | inactive timeline, behind the present | same board | 3 | RQ3 | `cwmtt/manyChecks@20` |
| CHK-I2 | inactive timeline, behind the present | cross-board onto a royal on a history board of a mandatory timeline; the branch would be **inactive** | 6 | RQ4 | `cwmtt/standard@13` |

- **Four rule questions** (details under "Rule questions"):
  - **RQ1**, active optional boards ahead of the present: CHK-A1, and through it CHK-A2, CHK-A4 and CHK-A5 (52 positions). Their extra elements are settled by high-confidence rules (C2, J4, C12 and C6).
  - **RQ2**, the same with a capture that would branch into an inactive timeline: CHK-A3 (17).
  - **RQ3**, inactive optional boards: CHK-I1 (3).
  - **RQ4**, the same with a capture that would branch into an inactive timeline: CHK-I2 (6).
- **What `docs/rules` says now.** C6 (high) computes check from a hypothetical pass, which excludes readings 6 and 7. C6 (medium at most) passes "every board in the present", and C7 (medium) says threats from boards ahead of the present or on inactive timelines are not check. Under T10's count rule both give "not in check" at all 78 positions, 5d-chess-js's answer. Reading 4 ([41]) would put the 69 CHK-A positions in check and reading 5 (wiki.gg only) all 78, ftxi's answer - the 17 of CHK-A3 and the 6 of CHK-I2 only if a capture that would branch into an inactive timeline counts.
- **Q1 (activation) moves 18 CHK-A positions** under its "consecutive" reading, and 3 under its per-turn reading; it moves no CHK-I position. Of the 18, four (cwmtt/wide@24, 28, 30, 32) would be in check under every pass reading, and at two (wide@38, 46) Black would be the side to move. cwmtt/wide is not a legal game under "consecutive" from action 23 on. Q1 is part of the ruling for those positions.
- **No terminal verdict in the corpus turns on these 78 positions.**
  - A legal action exists at each of them.
  - ftxi's and cwmtt's counts are equal wherever both answer: 10 below the cap and 67 at it.
  - The difference would decide checkmate against stalemate only at a position with no legal action (C5, C6).
- **The two mate disagreements** do not trace to a check cluster. Both trace to a defect in 5d-chess-js's checkmate search:
  - its "exhaustive" BFS passes its tree node where a board is expected, so it generates no moves and answers checkmate;
  - a legal turn exists at both positions by C5: one listed action at each was checked against T5 and C4 (all 25 and all 21 pass);
  - the library's own enumerator lists the same legal actions as ftxi and ftxi:naive at both positions (25 and 21), and as cwmtt and cwmtt:naive at chessin5d-illegal-move-example@66 (cwmtt cannot read ftxi/number1);
  - the defect is not specific to these positions. Every 5d-chess-js checkmate is reached without the BFS doing anything, so it means only that a single pass and a check-reducing DFS found no escape. The other 13 of its 15 golden checkmates are corroborated by another source, so these two are the defect's whole footprint in this corpus. A 5d-chess-js checkmate that no other source answers should be treated as unverified.
- **The count, 52 against 54,** traces to a colour asymmetry in ftxi's move generator: it never generates a black pawn's two-timeline step, but does generate the white one. Whether that step exists at all is a rule question in its own right (4.6 against [72], Q15).
- **Unexplained: nothing.**

## The two check tests

Both oracles look for a royal capture after a hypothetical pass (C6). They differ in which of the side to move's boards they pass:

- **ftxi** (`src/core/state.cpp:640-653`, `check_position.cpp:23-35`) appends a copy of **every** timeline's last board that is on the mover's colour. That is Q10 reading 5, "all boards".
  - It then scans every board on which the attacker is to move (`check_position.cpp:219-235`).
  - `checks()` lists every capture, same-board captures included (`check_position.cpp:249-258`).
- **5d-chess-js** `blankAction` (`src/mate.js:5-21`) copies only the last boards of its `present()` timelines (`board.js:307-347`). That is Q10 reading 2, "all active boards in the present".
  - At all 657 positions of these records, its `present()` timelines are exactly the mandatory boards.
  - `checks()` (`mate.js:23-53`) then tries every opponent move from every timeline whose last board is the opponent's.

So the boards ftxi passes and 5d-chess-js does not are the side to move's **optional** boards (T6):

- active ones ahead of the present;
- inactive ones wherever they lie.

An enemy piece standing on such a board is a source of attack for ftxi and not for 5d-chess-js.

## Rule questions

### RQ1: Active optional boards ahead of the present

Clusters: CHK-A1, CHK-A2, CHK-A4, CHK-A5 (52 positions).

Rule question 1 of 4. It decides CHK-A1 and, with it, CHK-A2, CHK-A4 and CHK-A5 (52 positions). When the side to move has a playable board on an ACTIVE timeline ahead of the present (T4), a board it may leave unplayed (T6), does an enemy piece on that board that could capture a royal once the board is passed put the side to move in check, for the verdict that separates checkmate from stalemate (C5, C6)? In spec 11.0 item 1's words: is that timeline "eligible", advanced to its phantom board before royal captures are sought? What docs/rules says now. C6 (high): check "is computed from a hypothetical pass, not from the client's per-timeline check indicator (C12)". That excludes Q10's readings 6 (per timeline, the indicator's test) and 7 (no pass), and leaves open only the passed set (readings 1/3, 2, 4, 5). C6 (medium at most): "The passed set is "every board in the present" (G-old)", reading 1, which passes no attacker's board at these positions (none stands in the present column). C7 (medium): of an enemy piece on a board where the mover is to move, "if that board lies ahead of the present or on an inactive timeline", "its threat is not check, does not bar submission and does not count for mate, until the present reaches that board or its timeline becomes active". Under T10's count rule both give "not in check" at every CHK-A position, and at all 78: 5d-chess-js's answer. The readings that give "in check" here are reading 4, "all active boards" ([41]: "For 5D-Chess I will assume that passing means passing on all active boards", with its Interpretation 3, "a piece can capture once we pass on all boards", which it found the client follows; the client is evidence, never authority, C14), and reading 5, "all boards" (wiki.gg [93][135], which docs/rules never takes as sole support). The ruling is one of: confirm C6/C7 (medium), so that no CHK-A or CHK-I position is in check; adopt reading 4, so that every CHK-A position is in check (CHK-A3 subject to rule question 2) and no CHK-I position; adopt reading 5, so that all 78 are (CHK-A3 and CHK-I2 subject to their branch question). Activity (T10, Q1). The count rule (high) makes the attacker's timeline active at every CHK-A1 position; Q1 is open, and its alternative readings change 14 of the 40 (positions[].q1). Under its "consecutive" reading (and, at niceAndrey@64, 66 and 68, its per-turn reading too) the attacker's timeline is inactive at niceAndrey@64, 66, 68, 90, 92, 94, 116 and 118, which are then visible under reading 5 only (rule question 3's kind, for a board ahead of the present). Under "consecutive" the present moves to T7w at wide@24, 28, 30 and 32, where the attacker's board (+2) becomes mandatory: in check under every pass reading, reading 2 (5d-chess-js's) included. At wide@38 and 46 it would make Black the side to move, so the question asked of these positions does not arise. Under "consecutive" cwmtt/wide is not a legal game from action 23 on: White's branch +4 would be inactive and would not move the present (T8). So its positions from 23 on stand under that reading only as set-ups (the boards and the creation order). Q1 is therefore part of the ruling for these positions.

**docs/rules now:** C6 (high): a hypothetical pass, not the per-timeline indicator; C6 (medium at most): the passed set is every board in the present (reading 1); C7 (medium): threats from boards ahead of the present or on inactive timelines are not check. Under T10 these give "not in check" here.

**Ruling options:**

- confirm C6/C7 (medium): not in check (5d-chess-js's answer)
- adopt reading 4 ([41], client observation; C14): in check
- adopt reading 5 (wiki.gg only): in check (ftxi's answer)

### RQ2: The same, with a capture that would branch into an inactive timeline

Clusters: CHK-A3 (17 positions).

Rule question 2 of 4 (17 positions): rule question 1, plus Q10's open item "whether a capture that would branch into a new inactive timeline counts". The attacker has created more timelines than the side to move, so its next timeline would be inactive (T10), and an inactive branch never moves the present (T8). C2 (high) counts attacks on past positions without distinguishing active from inactive branches, and docs/rules leaves the item open (C3, Q10). Under C6/C7 (medium) these positions are not in check whatever the item's answer; under readings 4 and 5 they are in check only if such a capture counts. Q1 changes none of the 17: its readings only make more timelines inactive, the branches are inactive already, and the attacker's timeline (+2) stays active.

**docs/rules now:** As RQ1; Q10 leaves open whether a capture branching into an inactive timeline counts; C2 (high) does not distinguish.

**Ruling options:**

- RQ1's ruling, and whether such a capture counts

### RQ3: Inactive optional boards (here behind the present)

Clusters: CHK-I1 (3 positions).

Rule question 3 of 4 (3 positions): does an enemy piece on a board of the side to move that lies on an INACTIVE timeline, here behind the present, count as an attacker for the check that separates checkmate from stalemate (C5, C6)? What docs/rules says now: C6 (high) leaves only the passed set open; C6 (medium at most) passes "every board in the present", which these boards are not; C7 (medium) covers boards "on an inactive timeline": not check "until the present reaches that board or its timeline becomes active". Of Q10's readings only 5, "all boards" (wiki.gg, never sole support), passes such a board: readings 1/3 pass inactive boards only in the present column, and 2 and 4 pass only active ones. The ruling: confirm C6/C7 (medium), so that these positions are not in check, or adopt reading 5, so that they are. Activity: T10's count rule (high) with T11 makes the attacker's timeline inactive. Q1's alternative readings only ever make timelines inactive, and they change none of the nine CHK-I positions. Q2 does not arise, since each record starts with one timeline. The attacker's timeline holds a single board at every CHK-I position, so Q3's reading that play activates a timeline cannot touch it.

**docs/rules now:** C6 (high) as RQ1; C6 (medium at most) passes no such board; C7 (medium): not check until the timeline becomes active.

**Ruling options:**

- confirm C6/C7 (medium): not in check
- adopt reading 5: in check

### RQ4: The same, with a capture that would branch into an inactive timeline

Clusters: CHK-I2 (6 positions).

Rule question 4 of 4 (6 positions): rule question 3, plus Q10's open item "whether a capture that would branch into a new inactive timeline counts". C2 (high) counts attacks on past positions without distinguishing; T8: an inactive branch never moves the present; T10: the attacker has created more timelines than the side to move. Under C6/C7 (medium) these positions are not in check whatever the item's answer; under reading 5 they are in check only if such a capture counts.

**docs/rules now:** As RQ3, plus the Q10 branch item.

**Ruling options:**

- RQ3's ruling, and whether such a capture counts

### Q1: Activation when the creator is behind on count (docs/rules Q1, against T10's count rule)

Positions moved under "consecutive" (18): `cwmtt/niceAndrey@64`, `cwmtt/niceAndrey@66`, `cwmtt/niceAndrey@68`, `cwmtt/niceAndrey@90`, `cwmtt/niceAndrey@92`, `cwmtt/niceAndrey@94`, `cwmtt/niceAndrey@116`, `cwmtt/niceAndrey@118`, `cwmtt/wide@15`, `cwmtt/wide@17`, `cwmtt/wide@19`, `cwmtt/wide@24`, `cwmtt/wide@28`, `cwmtt/wide@30`, `cwmtt/wide@32`, `cwmtt/wide@35`, `cwmtt/wide@38`, `cwmtt/wide@46`.

Positions moved under per-turn (3): `cwmtt/niceAndrey@64`, `cwmtt/niceAndrey@66`, `cwmtt/niceAndrey@68`.

Q1's alternative readings ("consecutive": a creation is inactive when the previous creation in the game was by the same player; per-turn: a second creation in one action is inactive; each keeping T10 as a limit and T11) only ever make timelines inactive. They change the filing of 18 CHK-A positions under "consecutive" and 3 under per-turn, and no CHK-I position. niceAndrey@64, 66, 68 (both readings), 90, 92, 94, 116, 118 and wide@35: the attacker's timeline is inactive, so the position is visible under reading 5 only. wide@15, 17, 19: the branch would be inactive, so the position is filed by its two-board capture, with the same readings. wide@24, 28, 30, 32: the present moves to T7w, so the attacker's board is mandatory and the position is in check under every pass reading. wide@38, 46: Black would be the side to move. cwmtt/wide is not a legal game under "consecutive" from action 23 on, so wide@24 and later stand under it only as set-ups.

**docs/rules now:** T10 (high): the count rule; T10 conflict and Q1: "consecutive" wordings and a per-turn wording disagree when the creator is behind on count.

**Ruling options:**

- confirm T10's count rule (the clusters stand as filed)
- adopt "consecutive" or per-turn (the positions listed move as stated)

## Method

Harness-side analysis over raw.json (the collector's evidence: golden rows, fresh fingerprints, every source's explain events, the boards each position holds) plus the pinned oracles' sources, read and never modified. No tracked file changed; outputs are under oracles/.cache/reports/checkpoint1/ (the final pass in final/, the first pass in cluster_work/ and final/previous/). For each of ftxi's 217 captures at the 78 in_check positions (final/cluster.py, final/paths.py): (a) whose turn it is on the attacking board in the real position, from the from-board kind (every from-board is ftxi's phantom of a head on the mover's colour that 5d-chess-js did not advance); (b) the attacker's timeline active or inactive by T10's count rule with T11 (raw.json derived, reproduced by final/common.py), and under Q1's two alternative readings, "consecutive" and per-turn, each keeping T10 as a limit and T11 (final/q1_full.py; both only ever make a timeline inactive that the count rule makes active, asserted at every position of the 14 records), with the present, side to move and roles recomputed under each; (c) the attacking board - always the timeline's playable head - and where it lies against the present (T4; ahead, behind or in its column); (d) the capture: same board or cross-board (dx, dy, dL, dT; through time, across timelines, or both), and the royal's board: the attacker's own board, the phantom of another of the mover's boards, a history board (a branching capture, with the activity of the timeline it would create by T10), or the head of a timeline where the attacker is to move. The boards a capture needs passed are the attacker's, the royal's when it is a phantom, and every board a slide crosses that exists only as a phantom (J7, J8). Each capture is then tested against Q10's readings of the passed set (1/3 every board in the present = mandatory + inactive heads in the present column; 2 all active boards in the present = mandatory = what 5d-chess-js passes; 4 all active boards = mandatory + active heads ahead; 5 all boards = what ftxi passes; 6 per timeline; 7 no pass). A position with several captures belongs to the cluster of the capture that makes the difference: one whose own set of readings equals the position's (so under every listed reading the position is in check exactly when that capture counts), then the fewest further open questions (a branch into an inactive timeline, more passes), then the simplest geometry (same board, non-branching jump, branching capture). That capture's readings contain every other capture's at all 78 positions. Clusters group positions by that capture's (a)-(d). An enumerator written here from docs/rules (final/c4_check.py) finds exactly ftxi's capture list at all 78 positions with every mover head passed (reading 5) and no capture with only the mandatory boards passed (reading 2); that validates this analysis's code, and is not evidence about the rules. Representatives were chosen by hand, not by a formula, for a person to verify: in each cluster a small position in the Standard variant (no Turn Zero T0 board, Q8, and no Reversed Royalty royal queens) with few captures and, where the cluster has one, no inactive optional board besides the attacker's own and a count below the cap. Each is its cluster's smallest position (fewest boards) or tied for it, except in CHK-A1 - cwmtt/100_timelines@15 (Turn Zero, 53 boards, five inactive optional boards, count at the cap) is smaller than cwmtt/silly@28 (Standard, 65 boards, no inactive timeline, count 37) - and in CHK-I1 - cwmtt/100_timelines@13 (Turn Zero, 28 boards; inactive optional boards +4 and +5 in the present column besides the attacker's +3) and cwmtt/wide@12 (47 boards, four captures) are smaller than cwmtt/manyChecks@20 (52 boards, one capture, the attacker's timeline the only inactive one). Each representative was re-drawn from the boards and checked square by square. Rule questions: the seven clusters are triage groups; they pose four distinct rule questions (rule_questions), with Q1 bearing on 18 CHK-A positions. Traces: the two mate disagreements by running the pinned 5d-chess-js library in memory at each position (cluster_work/mate_probe.js, re-run in review_traces/; review_traces/submit_probe.js, re-run here), by checking one listed action at each against T5 and C4 with final/c4_check.py (and all 25 and 21 the same way), and by setting every 5d-chess-js checkmate in the golden files against the other sources (final/mates_all.py, final/aivsai_check.py, one fresh ftxi run on a renumbered scratch copy of notation/aivsai); the count by reading ftxi's move generator, scanning the 14 records for open black and white pawn timeline double steps against the golden counts, and extra observe runs with --actions (notation/game-2 at 26; notation/1.2 at 17, fresh). Nothing is adjudicated; no oracle's side is taken. Review of the first pass: all 16 findings were checked against raw.json and fresh runs and applied (meta.review_dispositions); two were corrected in the applying - finding 2 (cwmtt's activity equals the count rule at all 611 positions with a cwmtt explain, not only 79) and finding 7 (Q1 moves 18 CHK-A positions, not 12: wide@24, 28, 30, 32 and wide@38, 46 added, and cwmtt/wide departs from "consecutive" at action 23); none was rejected.

Q10's readings, as the set of the side to move's boards each one passes:

- **R1/R3**: every board in the present / all present boards: mandatory + inactive heads in the present column
- **R2**: all active boards in the present: mandatory (what 5d-chess-js passes)
- **R4**: all active boards: mandatory + active heads ahead of the present
- **R5**: all boards: every head on the mover's colour (what ftxi passes)
- **R6**: per timeline: one active present timeline at a time (the indicator, C12) - excluded by C6 (high)
- **R7**: no pass - excluded by C6 (high)

Q1's readings of activation, as modelled:

- **count**: T10 count rule with T11 (high): the n-th timeline a player creates is active iff the opponent has created at least n-1
- **consec**: Q1 "consecutive" wordings: also inactive when the previous creation in the game was by the same player (T10 as a limit, T11 kept)
- **perturn**: Q1 per-turn reading: also inactive when the same player already created a timeline in the same action (T10 as a limit, T11 kept)

Under T10's count rule, each capture is visible under one of two sets of readings:

- **R4 and R5** when the attacker's board is active and ahead of the present: 202 captures.
- **R5 alone** when the attacker's board is inactive: 15 captures.

By position, 69 are in check under R4 and R5 and 9 under R5 alone. 185 captures need one board passed, 31 two and 1 three.

No attacker stands in the present column, so readings 1/3 never see a capture. Readings 2, 6 and 7 never do either.

## CHK-A1: Active timeline, ahead of the present; ordinary capture on the attacker's own board

**Positions (40; 170 captures at them):** `cwmtt/100_timelines@15`, `cwmtt/100_timelines@20`, `cwmtt/NP@77`, `cwmtt/NP@79`, `cwmtt/NP0@77`, `cwmtt/NP0@79`, `cwmtt/manyChecks@28`, `cwmtt/manyChecks@30`, `cwmtt/manyChecks@32`, `cwmtt/manyChecks@40`, `cwmtt/niceAndrey@56`, `cwmtt/niceAndrey@58`, `cwmtt/niceAndrey@60`, `cwmtt/niceAndrey@64`, `cwmtt/niceAndrey@66`, `cwmtt/niceAndrey@68`, `cwmtt/niceAndrey@90`, `cwmtt/niceAndrey@92`, `cwmtt/niceAndrey@94`, `cwmtt/niceAndrey@96`, `cwmtt/niceAndrey@111`, `cwmtt/niceAndrey@113`, `cwmtt/niceAndrey@116`, `cwmtt/niceAndrey@117`, `cwmtt/niceAndrey@118`, `cwmtt/silly@28`, `cwmtt/silly@30`, `cwmtt/silly@32`, `cwmtt/silly@34`, `cwmtt/wide@16`, `cwmtt/wide@18`, `cwmtt/wide@24`, `cwmtt/wide@28`, `cwmtt/wide@30`, `cwmtt/wide@32`, `cwmtt/wide@36`, `cwmtt/wide@38`, `cwmtt/wide@44`, `cwmtt/wide@46`, `notation/game-2@43`

Records: cwmtt/100_timelines 2, cwmtt/NP 2, cwmtt/NP0 2, cwmtt/manyChecks 4, cwmtt/niceAndrey 15, cwmtt/silly 4, cwmtt/wide 10, notation/game-2 1.

ftxi:naive in check: true at 24, no answer (timed out) at 16.

Positions Q1 would move: under "consecutive" `cwmtt/niceAndrey@64`, `cwmtt/niceAndrey@66`, `cwmtt/niceAndrey@68`, `cwmtt/niceAndrey@90`, `cwmtt/niceAndrey@92`, `cwmtt/niceAndrey@94`, `cwmtt/niceAndrey@116`, `cwmtt/niceAndrey@118`, `cwmtt/wide@24`, `cwmtt/wide@28`, `cwmtt/wide@30`, `cwmtt/wide@32`, `cwmtt/wide@38`, `cwmtt/wide@46`; under per-turn `cwmtt/niceAndrey@64`, `cwmtt/niceAndrey@66`, `cwmtt/niceAndrey@68`.

**Attacker's situation.** In the real position it is the side to move's turn on the attacking board: the attacking piece belongs to the other side and cannot move until that board is played or passed. The board is the playable (head) board of an ACTIVE timeline (T10) and lies AHEAD of the present (T4), so the side to move may leave it unplayed (T6). The capture is ordinary: on that same board, of a royal standing on it (after the pass, on the phantom copy of the head). ftxi advanced the attacker's timeline to its phantom (state::phantom advances every head on the mover's colour); 5d-chess-js did not (blankAction advances only its present timelines, which are the mandatory boards at every position checked).

**Representative: `cwmtt/silly@28`.**

Notation: a capture is written <piece>(<L>T<T><c>)<square> -> <royal>(<L>T<T><c>)<square>, c being the side to move on that board; upper case is White, lower case Black; K/k king; Y/y royal queen and C/c common king (Reversed Royalty). cwmtt/silly position 28 (Standard): White to move; present T2w (T4); created timelines White 3, Black 2; active -2, -1, 0, +1, +2, +3; inactive none; mandatory +3; ftxi advanced -2, +3, 5d-chess-js advanced +3. Deciding check: q(-2T7b)h4 -> K(-2T7b)e1 - attacker on (-2T7w) (L-2 active, ahead of the present), ordinary capture on the same board; dx -3 dy -3 dL +0 dT +0; passes needed: L-2 (optional_active_ahead); visible under Q10 readings R4, R5. Other captures ftxi reports here: q(-2T7b)h4 -> K(+1T7b)e1 - attacker on (-2T7w) (L-2 active, ahead of the present), cross-board capture onto a history board (branch would be active), across timelines, same turn; dx -3 dy -3 dL +3 dT +0; passes needed: L-2 (optional_active_ahead); visible under Q10 readings R4, R5; squares passed: (-1T7b)g3 empty, (0T7b)f2 empty. Boards (5DFEN placements, rank 8 first; a board marked phantom is not in the state - it is the copy of a head that exists only once that head is passed): (-2T7w) r*nbqk*2r*/p*p*p*p*p*p*1p*/6pb/8/2P1N2q/4P3/P*P*1P*P*1P*P*/R*1BQK*B1R*; (+1T7b) r*nbqk*2r*/p*1p*p*1p*p*p*/1p2p3/2b5/3N4/8/P*P*P*P*P*P*P*P*/R*NBQK*B1R* (target of q(-2T7b)h4 -> K(+1T7b)e1). 5d-chess-js advances only +3 and finds no capture.

(-2T7w): attacker's board in the real position: the playable (head) board of L-2, White to move on it; ftxi's phantom (-2T7b) is a copy of it with the other side to move. Marked: h4, e1.

```
8  r  n  b  q  k  .  .  r 
7  p  p  p  p  p  p  .  p 
6  .  .  .  .  .  .  p  b 
5  .  .  .  .  .  .  .  . 
4  .  .  P  .  N  .  . [q]
3  .  .  .  .  P  .  .  . 
2  P  P  .  P  P  .  P  P 
1  R  .  B  Q [K] B  .  R 
   a  b  c  d  e  f  g  h 
```

Other capture q(-2T7b)h4 -> K(+1T7b)e1:

(+1T7b): history board of L+1 (a capture onto it branches, J3/J5). Marked: e1.

```
8  r  n  b  q  k  .  .  r 
7  p  .  p  p  .  p  p  p 
6  .  p  .  .  p  .  .  . 
5  .  .  b  .  .  .  .  . 
4  .  .  .  N  .  .  .  . 
3  .  .  .  .  .  .  .  . 
2  P  P  P  P  P  P  P  P 
1  R  N  B  Q [K] B  .  R 
   a  b  c  d  e  f  g  h 
```

Squares it passes: (-1T7b)g3 empty; (0T7b)f2 empty.

**Verdicts at the representative.** in_check: ftxi true, ftxi:naive true, chessin5d false; cwmtt and cwmtt:naive have no check verdict (cwmtt's CLI answers counts and "no action" only, so its golden in_check is always null); legal_actions: ftxi 37, ftxi:naive 37, cwmtt 37, cwmtt:naive no answer: its golden run timed out at position 27; chessin5d gives no count by design; terminal: ftxi none, ftxi:naive none, cwmtt none, cwmtt:naive no answer, chessin5d none - a legal action exists, so the check verdict decides no terminal verdict here.

**Rule question (RQ1).** Rule question 1 of 4. It decides CHK-A1 and, with it, CHK-A2, CHK-A4 and CHK-A5 (52 positions). When the side to move has a playable board on an ACTIVE timeline ahead of the present (T4), a board it may leave unplayed (T6), does an enemy piece on that board that could capture a royal once the board is passed put the side to move in check, for the verdict that separates checkmate from stalemate (C5, C6)? In spec 11.0 item 1's words: is that timeline "eligible", advanced to its phantom board before royal captures are sought? What docs/rules says now. C6 (high): check "is computed from a hypothetical pass, not from the client's per-timeline check indicator (C12)". That excludes Q10's readings 6 (per timeline, the indicator's test) and 7 (no pass), and leaves open only the passed set (readings 1/3, 2, 4, 5). C6 (medium at most): "The passed set is "every board in the present" (G-old)", reading 1, which passes no attacker's board at these positions (none stands in the present column). C7 (medium): of an enemy piece on a board where the mover is to move, "if that board lies ahead of the present or on an inactive timeline", "its threat is not check, does not bar submission and does not count for mate, until the present reaches that board or its timeline becomes active". Under T10's count rule both give "not in check" at every CHK-A position, and at all 78: 5d-chess-js's answer. The readings that give "in check" here are reading 4, "all active boards" ([41]: "For 5D-Chess I will assume that passing means passing on all active boards", with its Interpretation 3, "a piece can capture once we pass on all boards", which it found the client follows; the client is evidence, never authority, C14), and reading 5, "all boards" (wiki.gg [93][135], which docs/rules never takes as sole support). The ruling is one of: confirm C6/C7 (medium), so that no CHK-A or CHK-I position is in check; adopt reading 4, so that every CHK-A position is in check (CHK-A3 subject to rule question 2) and no CHK-I position; adopt reading 5, so that all 78 are (CHK-A3 and CHK-I2 subject to their branch question). Activity (T10, Q1). The count rule (high) makes the attacker's timeline active at every CHK-A1 position; Q1 is open, and its alternative readings change 14 of the 40 (positions[].q1). Under its "consecutive" reading (and, at niceAndrey@64, 66 and 68, its per-turn reading too) the attacker's timeline is inactive at niceAndrey@64, 66, 68, 90, 92, 94, 116 and 118, which are then visible under reading 5 only (rule question 3's kind, for a board ahead of the present). Under "consecutive" the present moves to T7w at wide@24, 28, 30 and 32, where the attacker's board (+2) becomes mandatory: in check under every pass reading, reading 2 (5d-chess-js's) included. At wide@38 and 46 it would make Black the side to move, so the question asked of these positions does not arise. Under "consecutive" cwmtt/wide is not a legal game from action 23 on: White's branch +4 would be inactive and would not move the present (T8). So its positions from 23 on stand under that reading only as set-ups (the boards and the creation order). Q1 is therefore part of the ruling for these positions.

**Sources of attack (spec 11.0 item 1): yes.** Concerns which boards count as sources of attack (spec 11.0 item 1): the attacker stands on a board on which the side to move is to move, outside the present, and is a source only if that timeline is "eligible" - advanced to its phantom board.

## CHK-A2: Active timeline, ahead of the present; cross-board capture of a royal on a history board (the branch would be active)

**Positions (5; 8 captures at them):** `cwmtt/wide@15`, `cwmtt/wide@17`, `cwmtt/wide@19`, `notation/Bg2@19`, `notation/Bg2@21`

Records: cwmtt/wide 3, notation/Bg2 2.

ftxi:naive in check: no answer (timed out) at 3, true at 2.

Positions Q1 would move: under "consecutive" `cwmtt/wide@15`, `cwmtt/wide@17`, `cwmtt/wide@19`; under per-turn none.

**Attacker's situation.** Attacking board as in CHK-A1: the side to move's turn on it, ACTIVE timeline, its playable board, AHEAD of the present, optional. The capture is CROSS-BOARD - back in time on its own timeline, across timelines, or both - onto a royal standing on a HISTORY board (its past position), so it would branch (J3, J5); by the count rule (T10) the timeline it would create would be active. Only the attacker's board has to be passed. ftxi advanced it; 5d-chess-js did not.

**Representative: `notation/Bg2@19`.**

Notation: a capture is written <piece>(<L>T<T><c>)<square> -> <royal>(<L>T<T><c>)<square>, c being the side to move on that board; upper case is White, lower case Black; K/k king; Y/y royal queen and C/c common king (Reversed Royalty). notation/Bg2 position 19 (Standard): Black to move; present T3b (T4); created timelines White 1, Black 1; active -1, 0, +1; inactive none; mandatory -1; ftxi advanced -1, +1, 5d-chess-js advanced -1. Deciding check: Q(+1T6w)f3 -> k(+1T2w)f7 - attacker on (+1T5b) (L+1 active, ahead of the present), cross-board capture onto a history board (branch would be active), through time on its own timeline; dx +0 dy +4 dL +0 dT -4; passes needed: L+1 (optional_active_ahead); visible under Q10 readings R4, R5; squares passed: (+1T5w)f4 empty, (+1T4w)f5 empty, (+1T3w)f6 empty. ftxi reports no other capture here. Boards (5DFEN placements, rank 8 first; a board marked phantom is not in the state - it is the copy of a head that exists only once that head is passed): (+1T5b) r*nbqkb1r*/p*p*p*p*p*2p*/5np1/8/8/2N1PQ1N/P*P*P*P*1P*P*P*/R*1B1K*B1R*; (+1T2w) r*nbq1bnr*/p*p*p*p*p*kp*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*. 5d-chess-js advances only -1 and finds no capture.

(+1T5b): attacker's board in the real position: the playable (head) board of L+1, Black to move on it; ftxi's phantom (+1T6w) is a copy of it with the other side to move. Marked: f3.

```
8  r  n  b  q  k  b  .  r 
7  p  p  p  p  p  .  .  p 
6  .  .  .  .  .  n  p  . 
5  .  .  .  .  .  .  .  . 
4  .  .  .  .  .  .  .  . 
3  .  .  N  .  P [Q] .  N 
2  P  P  P  P  .  P  P  P 
1  R  .  B  .  K  B  .  R 
   a  b  c  d  e  f  g  h 
```

(+1T2w): history board of L+1 (a capture onto it branches, J3/J5). Marked: f7.

```
8  r  n  b  q  .  b  n  r 
7  p  p  p  p  p [k] p  p 
6  .  .  .  .  .  .  .  . 
5  .  .  .  .  .  .  .  . 
4  .  .  .  .  .  .  .  . 
3  .  .  .  .  .  .  .  . 
2  P  P  P  P  P  P  P  P 
1  R  N  B  Q  K  B  N  R 
   a  b  c  d  e  f  g  h 
```

Squares the slide passes: (+1T5w)f4 empty; (+1T4w)f5 empty; (+1T3w)f6 empty.

**Verdicts at the representative.** in_check: ftxi true, ftxi:naive true, chessin5d false; cwmtt and cwmtt:naive have no check verdict (cwmtt's CLI answers counts and "no action" only, so its golden in_check is always null); legal_actions: ftxi 62, ftxi:naive 62, cwmtt 62, cwmtt:naive 62; chessin5d gives no count by design; terminal: ftxi none, ftxi:naive none, cwmtt none, cwmtt:naive none, chessin5d none - a legal action exists, so the check verdict decides no terminal verdict here.

**Rule question (RQ1).** Settled by rule question 1 (CHK-A1); its extra element raises no open question. The capture lands on a royal's past position and would branch (J3, J5): C2 (high), "An attack on a king's past position, on any board and any timeline, is an ordinary check". The branch would be active (T10), so Q10's open item on branches into inactive timelines does not arise. The exception is Q1's "consecutive" reading at wide@15, 17 and 19, where White's next creation would follow its own +3 (White 3, Black 4) and be inactive. Those three would then be filed by their other capture, a non-branching jump onto another optional board, N(-3T7w)e6 -> k(-2T7w)e8 (CHK-A5's kind), visible under the same readings 4 and 5. Paths: the queen captures at Bg2@19 and 21 slide over boards (J7; three empty squares); the knight captures at wide@15, 17 and 19 jump (4.5, J8).

**Sources of attack (spec 11.0 item 1): yes.** Concerns which boards count as sources of attack (spec 11.0 item 1): the attacker stands on a board on which the side to move is to move, outside the present, and is a source only if that timeline is "eligible" - advanced to its phantom board.

## CHK-A3: Active timeline, ahead of the present; cross-board capture of a royal on a history board (the branch would be inactive)

**Positions (17; 17 captures at them):** `cwmtt/niceAndrey@57`, `cwmtt/niceAndrey@61`, `cwmtt/niceAndrey@63`, `cwmtt/niceAndrey@65`, `cwmtt/niceAndrey@69`, `cwmtt/niceAndrey@71`, `cwmtt/niceAndrey@75`, `cwmtt/niceAndrey@77`, `cwmtt/niceAndrey@79`, `cwmtt/niceAndrey@81`, `cwmtt/niceAndrey@87`, `cwmtt/niceAndrey@91`, `cwmtt/niceAndrey@95`, `cwmtt/niceAndrey@97`, `cwmtt/niceAndrey@101`, `cwmtt/niceAndrey@103`, `cwmtt/niceAndrey@105`

Records: cwmtt/niceAndrey 17.

ftxi:naive in check: true at 14, no answer (timed out) at 3.

Positions Q1 would move: none.

**Attacker's situation.** As CHK-A2 - the side to move's turn on the attacking board, ACTIVE timeline, its playable board, AHEAD of the present; a CROSS-BOARD capture onto a royal on a HISTORY board - except that the attacker has created more timelines than the side to move, so by the count rule (T10) the timeline the capture would create would be INACTIVE.

**Representative: `cwmtt/niceAndrey@57`.**

Notation: a capture is written <piece>(<L>T<T><c>)<square> -> <royal>(<L>T<T><c>)<square>, c being the side to move on that board; upper case is White, lower case Black; K/k king; Y/y royal queen and C/c common king (Reversed Royalty). cwmtt/niceAndrey position 57 (Standard - Reversed Royalty): Black to move; present T10b (T4); created timelines White 13, Black 3; active -3, -2, -1, 0, +1, +2, +3, +4; inactive +5, +6, +7, +8, +9, +10, +11, +12, +13; mandatory +4; ftxi advanced -1, 0, +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, 5d-chess-js advanced +4. Deciding check: B(+2T17w)d7 -> y(+2T16w)d8 - attacker on (+2T16b) (L+2 active, ahead of the present), cross-board capture onto a history board (branch would be inactive), through time on its own timeline; dx +0 dy +1 dL +0 dT -1; passes needed: L+2 (optional_active_ahead); visible under Q10 readings R4, R5. ftxi reports no other capture here. Boards (5DFEN placements, rank 8 first; a board marked phantom is not in the state - it is the copy of a head that exists only once that head is passed): (+2T16b) y2yc2r*/rp*p*Bb2p*/p1n2np1/8/3p1p1N/P1NP2P1/1P*P*CP*P*1P*/1R5R*; (+2T16w) y2yc2r*/rp*p*bb2p*/p1n2np1/5B2/3p1p1N/P1NP2P1/1P*P*CP*P*1P*/1R5R*. 5d-chess-js advances only +4 and finds no capture.

(+2T16b): attacker's board in the real position: the playable (head) board of L+2, Black to move on it; ftxi's phantom (+2T17w) is a copy of it with the other side to move. Marked: d7.

```
8  y  .  .  y  c  .  .  r 
7  r  p  p [B] b  .  .  p 
6  p  .  n  .  .  n  p  . 
5  .  .  .  .  .  .  .  . 
4  .  .  .  p  .  p  .  N 
3  P  .  N  P  .  .  P  . 
2  .  P  P  C  P  P  .  P 
1  .  R  .  .  .  .  .  R 
   a  b  c  d  e  f  g  h 
```

(+2T16w): history board of L+2 (a capture onto it branches, J3/J5). Marked: d8.

```
8  y  .  . [y] c  .  .  r 
7  r  p  p  b  b  .  .  p 
6  p  .  n  .  .  n  p  . 
5  .  .  .  .  .  B  .  . 
4  .  .  .  p  .  p  .  N 
3  P  .  N  P  .  .  P  . 
2  .  P  P  C  P  P  .  P 
1  .  R  .  .  .  .  .  R 
   a  b  c  d  e  f  g  h 
```

**Verdicts at the representative.** in_check: ftxi true, ftxi:naive true, chessin5d false; cwmtt and cwmtt:naive have no check verdict (cwmtt's CLI answers counts and "no action" only, so its golden in_check is always null); legal_actions: ftxi 1000 (the cap: at least 1000), ftxi:naive 1000 (the cap: at least 1000), cwmtt 1000 (the cap: at least 1000), cwmtt:naive no answer: its golden run timed out at position 51; chessin5d gives no count by design; terminal: ftxi none, ftxi:naive none, cwmtt none, cwmtt:naive no answer, chessin5d none - a legal action exists, so the check verdict decides no terminal verdict here.

**Rule question (RQ2).** Rule question 2 of 4 (17 positions): rule question 1, plus Q10's open item "whether a capture that would branch into a new inactive timeline counts". The attacker has created more timelines than the side to move, so its next timeline would be inactive (T10), and an inactive branch never moves the present (T8). C2 (high) counts attacks on past positions without distinguishing active from inactive branches, and docs/rules leaves the item open (C3, Q10). Under C6/C7 (medium) these positions are not in check whatever the item's answer; under readings 4 and 5 they are in check only if such a capture counts. Q1 changes none of the 17: its readings only make more timelines inactive, the branches are inactive already, and the attacker's timeline (+2) stays active.

**Sources of attack (spec 11.0 item 1): yes.** Concerns which boards count as sources of attack (spec 11.0 item 1): the attacker stands on a board on which the side to move is to move, outside the present, and is a source only if that timeline is "eligible" - advanced to its phantom board. It adds a question about which captures count (a branch into an inactive timeline), not about sources.

## CHK-A4: Active timeline, ahead of the present; non-branching jump onto a playable board on which the attacker is to move

**Positions (2; 5 captures at them):** `cwmtt/silly@16`, `cwmtt/wide@35`

Records: cwmtt/silly 1, cwmtt/wide 1.

ftxi:naive in check: true at 1, no answer (timed out) at 1.

Positions Q1 would move: under "consecutive" `cwmtt/wide@35`; under per-turn none.

**Attacker's situation.** Attacking board as in CHK-A1 (the side to move's turn on it, ACTIVE, playable board, AHEAD of the present, optional). The capture is CROSS-BOARD and non-branching (J4): onto the playable head of another timeline on which the attacker is already to move, where the royal stands - a board no reading has to pass. In both positions the same attacker also has a branching capture of a royal's past position, with an active branch and the same single pass: at silly@16 of the king of L0, the attacker's own timeline (not the -1 king the jump captures); at wide@35 of the same +1 king. So the head/history split does not change either position's verdict; they are kept apart because the triage splits on where the royal stands (a head or a history board).

**Representative: `cwmtt/silly@16`.**

Notation: a capture is written <piece>(<L>T<T><c>)<square> -> <royal>(<L>T<T><c>)<square>, c being the side to move on that board; upper case is White, lower case Black; K/k king; Y/y royal queen and C/c common king (Reversed Royalty). cwmtt/silly position 16 (Standard): White to move; present T5w (T4); created timelines White 2, Black 1; active -1, 0, +1, +2; inactive none; mandatory +2; ftxi advanced 0, +1, +2, 5d-chess-js advanced +2. Deciding check: n(0T6b)e3 -> K(-1T6b)e1 - attacker on (0T6w) (L0 active, ahead of the present), cross-board non-branching jump onto a board where the attacker is to move, across timelines, same turn; dx +0 dy -2 dL -1 dT +0; passes needed: L0 (optional_active_ahead); visible under Q10 readings R4, R5. Other captures ftxi reports here: n(0T6b)e3 -> K(+1T6b)e1 - attacker on (0T6w) (L0 active, ahead of the present), cross-board capture onto the phantom of another optional board, across timelines, same turn; dx +0 dy -2 dL +1 dT +0; passes needed: L0 (optional_active_ahead), L+1 (optional_active_ahead; the royal's board); visible under Q10 readings R4, R5 | n(0T6b)e3 -> K(0T5b)e1 - attacker on (0T6w) (L0 active, ahead of the present), cross-board capture onto a history board (branch would be active), through time on its own timeline; dx +0 dy -2 dL +0 dT -1; passes needed: L0 (optional_active_ahead); visible under Q10 readings R4, R5. Boards (5DFEN placements, rank 8 first; a board marked phantom is not in the state - it is the copy of a head that exists only once that head is passed): (0T6w) r*nbqk*2r*/p*p*p*p*p*p*1p*/6pb/8/2P1N3/4n3/P*P*1P*P*P*P*P*/R*1BQK*B1R*; (-1T6b) r*n1qk*b1r*/p*1p*p*p*p*p*p*/1p6/5N2/6n1/8/P*P*P*P*1P*P*P*/R*NB1K*BNR*; (+1T6b) [phantom: copy of head (+1T6w)] r*nbqk*b1r*/p*1p*p*1p*p*p*/1p2p3/4NN2/8/8/P*P*P*P*P*P*P*P*/R*NBQK*B1R* (target of n(0T6b)e3 -> K(+1T6b)e1); (0T5b) r*nbqk*2r*/p*p*p*p*p*p*1p*/6pb/3n4/2P1N3/8/P*P*1P*P*P*P*P*/R*1BQK*B1R* (target of n(0T6b)e3 -> K(0T5b)e1). 5d-chess-js advances only +2 and finds no capture.

(0T6w): attacker's board in the real position: the playable (head) board of L0, White to move on it; ftxi's phantom (0T6b) is a copy of it with the other side to move. Marked: e3.

```
8  r  n  b  q  k  .  .  r 
7  p  p  p  p  p  p  .  p 
6  .  .  .  .  .  .  p  b 
5  .  .  .  .  .  .  .  . 
4  .  .  P  .  N  .  .  . 
3  .  .  .  . [n] .  .  . 
2  P  P  .  P  P  P  P  P 
1  R  .  B  Q  K  B  .  R 
   a  b  c  d  e  f  g  h 
```

(-1T6b): playable (head) board of L-1, on which the attacker is to move (a non-branching jump, J4). Marked: e1.

```
8  r  n  .  q  k  b  .  r 
7  p  .  p  p  p  p  p  p 
6  .  p  .  .  .  .  .  . 
5  .  .  .  .  .  N  .  . 
4  .  .  .  .  .  .  n  . 
3  .  .  .  .  .  .  .  . 
2  P  P  P  P  .  P  P  P 
1  R  N  B  . [K] B  N  R 
   a  b  c  d  e  f  g  h 
```

Other capture n(0T6b)e3 -> K(+1T6b)e1:

(+1T6b): phantom, not in the state: a copy of the head (+1T6w) of L+1, which the side to move may leave unplayed; it exists only once that board is passed. Marked: e1.

```
8  r  n  b  q  k  b  .  r 
7  p  .  p  p  .  p  p  p 
6  .  p  .  .  p  .  .  . 
5  .  .  .  .  N  N  .  . 
4  .  .  .  .  .  .  .  . 
3  .  .  .  .  .  .  .  . 
2  P  P  P  P  P  P  P  P 
1  R  N  B  Q [K] B  .  R 
   a  b  c  d  e  f  g  h 
```

Other capture n(0T6b)e3 -> K(0T5b)e1:

(0T5b): history board of L0 (a capture onto it branches, J3/J5). Marked: e1.

```
8  r  n  b  q  k  .  .  r 
7  p  p  p  p  p  p  .  p 
6  .  .  .  .  .  .  p  b 
5  .  .  .  n  .  .  .  . 
4  .  .  P  .  N  .  .  . 
3  .  .  .  .  .  .  .  . 
2  P  P  .  P  P  P  P  P 
1  R  .  B  Q [K] B  .  R 
   a  b  c  d  e  f  g  h 
```

**Verdicts at the representative.** in_check: ftxi true, ftxi:naive true, chessin5d false; cwmtt and cwmtt:naive have no check verdict (cwmtt's CLI answers counts and "no action" only, so its golden in_check is always null); legal_actions: ftxi 1000 (the cap: at least 1000), ftxi:naive 1000 (the cap: at least 1000), cwmtt 1000 (the cap: at least 1000), cwmtt:naive 1000 (the cap: at least 1000); chessin5d gives no count by design; terminal: ftxi none, ftxi:naive none, cwmtt none, cwmtt:naive none, chessin5d none - a legal action exists, so the check verdict decides no terminal verdict here.

**Rule question (RQ1).** Settled by rule question 1 (CHK-A1); its extra element raises no open question. The capture is a non-branching jump (J4, high) onto another timeline's playable board on which the attacker is already to move: an ordinary capture (spec 11.0 item 1: captures "come from any board the attacker could move on and run through time and across timelines"). Q1: at wide@35 the "consecutive" reading makes the attacker's +5 inactive, leaving the position visible under reading 5 only (rule question 3's kind, for a board ahead of the present). cwmtt/wide is not a legal game under that reading from action 23 on, so this holds only for the position taken as a set-up.

**Sources of attack (spec 11.0 item 1): yes.** Concerns which boards count as sources of attack (spec 11.0 item 1): the attacker stands on a board on which the side to move is to move, outside the present, and is a source only if that timeline is "eligible" - advanced to its phantom board.

## CHK-A5: Active timeline, ahead of the present; cross-board capture of a royal on another optional board (two boards passed)

**Positions (5; 5 captures at them):** `cwmtt/niceAndrey@41`, `cwmtt/niceAndrey@43`, `cwmtt/niceAndrey@45`, `cwmtt/niceAndrey@47`, `cwmtt/niceAndrey@49`

Records: cwmtt/niceAndrey 5.

ftxi:naive in check: true at 5.

Positions Q1 would move: none.

**Attacker's situation.** Attacking board as in CHK-A1 (the side to move's turn on it, ACTIVE, playable board, AHEAD of the present, optional). The capture is CROSS-BOARD onto a royal standing on ANOTHER playable board of the side to move, also active and ahead of the present: the threat exists only when both boards are passed together. ftxi advanced both; 5d-chess-js neither.

**Representative: `cwmtt/niceAndrey@41`.**

Notation: a capture is written <piece>(<L>T<T><c>)<square> -> <royal>(<L>T<T><c>)<square>, c being the side to move on that board; upper case is White, lower case Black; K/k king; Y/y royal queen and C/c common king (Reversed Royalty). cwmtt/niceAndrey position 41 (Standard - Reversed Royalty): Black to move; present T9b (T4); created timelines White 9, Black 2; active -2, -1, 0, +1, +2, +3; inactive +4, +5, +6, +7, +8, +9; mandatory -2; ftxi advanced -2, 0, +1, +2, +3, +4, +5, +6, +7, +8, +9, 5d-chess-js advanced -2. Deciding check: N(0T15w)d7 -> y(+2T15w)d8 - attacker on (0T14b) (L0 active, ahead of the present), cross-board capture onto the phantom of another optional board, across timelines, same turn; dx +0 dy +1 dL +2 dT +0; passes needed: L0 (optional_active_ahead), L+2 (optional_active_ahead; the royal's board); visible under Q10 readings R4, R5. ftxi reports no other capture here. Boards (5DFEN placements, rank 8 first; a board marked phantom is not in the state - it is the copy of a head that exists only once that head is passed): (0T14b) y3c2r*/rp*p*Nb2p*/p2p1p2/4p3/3n1p2/P1NP1N2/1P*P*CP*P*1P*/1R3B1R*; (+2T15w) [phantom: copy of head (+2T14b)] y2yc2r*/rp*p*bb1p*p*/p1n2n2/3p1p2/5p1N/P1NP2PB/1P*P*CP*P*1P*/1R5R*. 5d-chess-js advances only -2 and finds no capture.

(0T14b): attacker's board in the real position: the playable (head) board of L0, Black to move on it; ftxi's phantom (0T15w) is a copy of it with the other side to move. Marked: d7.

```
8  y  .  .  .  c  .  .  r 
7  r  p  p [N] b  .  .  p 
6  p  .  .  p  .  p  .  . 
5  .  .  .  .  p  .  .  . 
4  .  .  .  n  .  p  .  . 
3  P  .  N  P  .  N  .  . 
2  .  P  P  C  P  P  .  P 
1  .  R  .  .  .  B  .  R 
   a  b  c  d  e  f  g  h 
```

(+2T15w): phantom, not in the state: a copy of the head (+2T14b) of L+2, which the side to move may leave unplayed; it exists only once that board is passed. Marked: d8.

```
8  y  .  . [y] c  .  .  r 
7  r  p  p  b  b  .  p  p 
6  p  .  n  .  .  n  .  . 
5  .  .  .  p  .  p  .  . 
4  .  .  .  .  .  p  .  N 
3  P  .  N  P  .  .  P  B 
2  .  P  P  C  P  P  .  P 
1  .  R  .  .  .  .  .  R 
   a  b  c  d  e  f  g  h 
```

**Verdicts at the representative.** in_check: ftxi true, ftxi:naive true, chessin5d false; cwmtt and cwmtt:naive have no check verdict (cwmtt's CLI answers counts and "no action" only, so its golden in_check is always null); legal_actions: ftxi 1000 (the cap: at least 1000), ftxi:naive 1000 (the cap: at least 1000), cwmtt 1000 (the cap: at least 1000), cwmtt:naive 1000 (the cap: at least 1000); chessin5d gives no count by design; terminal: ftxi none, ftxi:naive none, cwmtt none, cwmtt:naive none, chessin5d none - a legal action exists, so the check verdict decides no terminal verdict here.

**Rule question (RQ1).** Settled by rule question 1 (CHK-A1), applied to both boards. The threat needs two optional boards passed together: the attacker's and the royal's, both active and ahead of the present. docs/rules (high) already counts a threat that needs several boards passed. C12: "A threat that needs several boards advanced may show no indicator, and it still counts under C4–C6". C6: check "is computed from a hypothetical pass, not from the client's per-timeline check indicator (C12)". Under every tabled reading these positions get CHK-A1's verdict: in check under readings 4 and 5 only. A per-timeline test over optional boards (the Encyclopedia's wording of reading 6, without fandom's restriction to active present timelines) would separate them: it would see CHK-A1's single-pass captures and not these two-pass ones. C6 (high) excludes any per-timeline test, so it is not tabled. Q1 changes none of the five.

**Sources of attack (spec 11.0 item 1): yes.** Concerns which boards count as sources of attack (spec 11.0 item 1): the attacker stands on a board on which the side to move is to move, outside the present, and is a source only if that timeline is "eligible" - advanced to its phantom board. It also concerns whether the royal's own optional board is advanced (a second eligible timeline).

## CHK-I1: Inactive timeline, behind the present; ordinary capture on the attacker's own board

**Positions (3; 6 captures at them):** `cwmtt/100_timelines@13`, `cwmtt/manyChecks@20`, `cwmtt/wide@12`

Records: cwmtt/100_timelines 1, cwmtt/manyChecks 1, cwmtt/wide 1.

ftxi:naive in check: true at 2, no answer (timed out) at 1.

Positions Q1 would move: none.

**Attacker's situation.** In the real position it is the side to move's turn on the attacking board (the attacking piece cannot move until that board is played or passed). The board is the playable board of an INACTIVE timeline (T10: it is its creator's n-th timeline and the opponent has created fewer than n-1) and lies BEHIND the present (T4: inactive timelines never set the present; T6: they may be played wherever they lie). The capture is ordinary, on that same board. ftxi advanced it; 5d-chess-js did not.

**Representative: `cwmtt/manyChecks@20`.**

Notation: a capture is written <piece>(<L>T<T><c>)<square> -> <royal>(<L>T<T><c>)<square>, c being the side to move on that board; upper case is White, lower case Black; K/k king; Y/y royal queen and C/c common king (Reversed Royalty). cwmtt/manyChecks position 20 (Standard): White to move; present T9w (T4); created timelines White 1, Black 3; active -2, -1, 0, +1; inactive -3; mandatory -2, -1, 0, +1; ftxi advanced -3, -2, -1, 0, +1, 5d-chess-js advanced -2, -1, 0, +1. Deciding check: b(-3T8b)g3 -> K(-3T8b)e1 - attacker on (-3T8w) (L-3 inactive, behind the present), ordinary capture on the same board; dx -2 dy -2 dL +0 dT +0; passes needed: L-3 (optional_inactive_behind); visible under Q10 readings R5. ftxi reports no other capture here. Boards (5DFEN placements, rank 8 first; a board marked phantom is not in the state - it is the copy of a head that exists only once that head is passed): (-3T8w) r*nb2b1r*/p*p*1k1p*p*p*/2p1p3/8/8/2P2PbP/P*P*1P*P*1P*1/R*NB1K*B1R*. 5d-chess-js advances only -2, -1, 0, +1 and finds no capture.

(-3T8w): attacker's board in the real position: the playable (head) board of L-3, White to move on it; ftxi's phantom (-3T8b) is a copy of it with the other side to move. Marked: g3, e1.

```
8  r  n  b  .  .  b  .  r 
7  p  p  .  k  .  p  p  p 
6  .  .  p  .  p  .  .  . 
5  .  .  .  .  .  .  .  . 
4  .  .  .  .  .  .  .  . 
3  .  .  P  .  .  P [b] P 
2  P  P  .  P  P  .  P  . 
1  R  N  B  . [K] B  .  R 
   a  b  c  d  e  f  g  h 
```

**Verdicts at the representative.** in_check: ftxi true, ftxi:naive true, chessin5d false; cwmtt and cwmtt:naive have no check verdict (cwmtt's CLI answers counts and "no action" only, so its golden in_check is always null); legal_actions: ftxi 1000 (the cap: at least 1000), ftxi:naive 1000 (the cap: at least 1000), cwmtt 1000 (the cap: at least 1000), cwmtt:naive 1000 (the cap: at least 1000); chessin5d gives no count by design; terminal: ftxi none, ftxi:naive none, cwmtt none, cwmtt:naive none, chessin5d none - a legal action exists, so the check verdict decides no terminal verdict here.

**Rule question (RQ3).** Rule question 3 of 4 (3 positions): does an enemy piece on a board of the side to move that lies on an INACTIVE timeline, here behind the present, count as an attacker for the check that separates checkmate from stalemate (C5, C6)? What docs/rules says now: C6 (high) leaves only the passed set open; C6 (medium at most) passes "every board in the present", which these boards are not; C7 (medium) covers boards "on an inactive timeline": not check "until the present reaches that board or its timeline becomes active". Of Q10's readings only 5, "all boards" (wiki.gg, never sole support), passes such a board: readings 1/3 pass inactive boards only in the present column, and 2 and 4 pass only active ones. The ruling: confirm C6/C7 (medium), so that these positions are not in check, or adopt reading 5, so that they are. Activity: T10's count rule (high) with T11 makes the attacker's timeline inactive. Q1's alternative readings only ever make timelines inactive, and they change none of the nine CHK-I positions. Q2 does not arise, since each record starts with one timeline. The attacker's timeline holds a single board at every CHK-I position, so Q3's reading that play activates a timeline cannot touch it.

**Sources of attack (spec 11.0 item 1): yes.** Concerns which boards count as sources of attack (spec 11.0 item 1): the attacker stands on a board on which the side to move is to move, outside the present, and is a source only if that timeline is "eligible" - advanced to its phantom board.

## CHK-I2: Inactive timeline, behind the present; cross-board capture of a royal on a history board of a mandatory timeline (the branch would be inactive)

**Positions (6; 6 captures at them):** `cwmtt/exiledKings@12`, `cwmtt/exiledKings@14`, `cwmtt/exiledKings@16`, `cwmtt/exiledKings@18`, `cwmtt/standard@13`, `cwmtt/standard@15`

Records: cwmtt/exiledKings 4, cwmtt/standard 2.

ftxi:naive in check: true at 5, no answer (timed out) at 1.

Positions Q1 would move: none.

**Attacker's situation.** Attacking board as in CHK-I1 (the side to move's turn on it, INACTIVE timeline, its playable board, BEHIND the present). The capture is CROSS-BOARD onto a royal on a HISTORY board of a MANDATORY timeline (the royal's past position), so it would branch; the attacker has created more timelines than the side to move, so the new timeline would be inactive (T10).

**Representative: `cwmtt/standard@13`.**

Notation: a capture is written <piece>(<L>T<T><c>)<square> -> <royal>(<L>T<T><c>)<square>, c being the side to move on that board; upper case is White, lower case Black; K/k king; Y/y royal queen and C/c common king (Reversed Royalty). cwmtt/standard position 13 (Standard): Black to move; present T3b (T4); created timelines White 2, Black 0; active 0, +1; inactive +2; mandatory +1; ftxi advanced 0, +1, +2, 5d-chess-js advanced +1. Deciding check: N(+2T3w)f5 -> k(+1T3w)f7 - attacker on (+2T2b) (L+2 inactive, behind the present), cross-board capture onto a history board (branch would be inactive), across timelines, same turn; dx +0 dy +2 dL -1 dT +0; passes needed: L+2 (optional_inactive_behind); visible under Q10 readings R5. ftxi reports no other capture here. Boards (5DFEN placements, rank 8 first; a board marked phantom is not in the state - it is the copy of a head that exists only once that head is passed): (+2T2b) r*nbq1bnr*/p*p*p*p*p*kp*p*/8/5N2/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*; (+1T3w) r*nbq1bnr*/p*p*p*p*1kp*p*/4p3/8/8/5N2/P*P*P*P*P*P*P*P*/R*NBQK*B1R*. 5d-chess-js advances only +1 and finds no capture.

(+2T2b): attacker's board in the real position: the playable (head) board of L+2, Black to move on it; ftxi's phantom (+2T3w) is a copy of it with the other side to move. Marked: f5.

```
8  r  n  b  q  .  b  n  r 
7  p  p  p  p  p  k  p  p 
6  .  .  .  .  .  .  .  . 
5  .  .  .  .  . [N] .  . 
4  .  .  .  .  .  .  .  . 
3  .  .  .  .  .  .  .  . 
2  P  P  P  P  P  P  P  P 
1  R  N  B  Q  K  B  N  R 
   a  b  c  d  e  f  g  h 
```

(+1T3w): history board of L+1 (a capture onto it branches, J3/J5). Marked: f7.

```
8  r  n  b  q  .  b  n  r 
7  p  p  p  p  . [k] p  p 
6  .  .  .  .  p  .  .  . 
5  .  .  .  .  .  .  .  . 
4  .  .  .  .  .  .  .  . 
3  .  .  .  .  .  N  .  . 
2  P  P  P  P  P  P  P  P 
1  R  N  B  Q  K  B  .  R 
   a  b  c  d  e  f  g  h 
```

**Verdicts at the representative.** in_check: ftxi true, ftxi:naive true, chessin5d false; cwmtt and cwmtt:naive have no check verdict (cwmtt's CLI answers counts and "no action" only, so its golden in_check is always null); legal_actions: ftxi 1000 (the cap: at least 1000), ftxi:naive 1000 (the cap: at least 1000), cwmtt 1000 (the cap: at least 1000), cwmtt:naive 1000 (the cap: at least 1000); chessin5d gives no count by design; terminal: ftxi none, ftxi:naive none, cwmtt none, cwmtt:naive none, chessin5d none - a legal action exists, so the check verdict decides no terminal verdict here.

**Rule question (RQ4).** Rule question 4 of 4 (6 positions): rule question 3, plus Q10's open item "whether a capture that would branch into a new inactive timeline counts". C2 (high) counts attacks on past positions without distinguishing; T8: an inactive branch never moves the present; T10: the attacker has created more timelines than the side to move. Under C6/C7 (medium) these positions are not in check whatever the item's answer; under reading 5 they are in check only if such a capture counts.

**Sources of attack (spec 11.0 item 1): yes.** Concerns which boards count as sources of attack (spec 11.0 item 1): the attacker stands on a board on which the side to move is to move, outside the present, and is a source only if that timeline is "eligible" - advanced to its phantom board. It adds a question about which captures count (a branch into an inactive timeline), not about sources.

## Positions with several captures

31 positions hold more than one capture. Each is filed by the capture that makes the difference: one whose own set of Q10 readings equals the position's. That capture's readings contain every other capture's at every position, so under every listed reading the position is in check exactly when it counts. Ties are broken by the fewest further open questions, then the fewest passes, then the simplest geometry. A capture that needs more than one board passed lists them; a board a slide crosses that exists only as a phantom counts (J7, J8). The cases where the other captures differ in kind:

- `cwmtt/100_timelines@20` (CHK-A1), deciding: n(+27T10b)f3 -> K(+27T10b)d2 [active, ahead of the present; same board]. Others: n(-54T9b)c4 -> K(-54T9b)d2 [inactive, ahead of the present; same board]; n(-51T9b)e3 -> K(-50T9b)e1 [inactive, ahead of the present; cross-board onto another optional board (two passes: -51, -50)]; n(-51T9b)e3 -> K(-52T9b)e1 [inactive, ahead of the present; cross-board onto another optional board (two passes: -52, -51)]; n(+27T10b)f3 -> K(+27T10b)e1 [active, ahead of the present; same board]; n(+48T9b)e4 -> K(+48T9b)d2 [active, ahead of the present; same board].
- `cwmtt/manyChecks@28` (CHK-A1), deciding: q(-1T12b)b4 -> K(-1T12b)e1 [active, ahead of the present; same board]. Others: q(-1T12b)b4 -> K(-1T9b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)].
- `cwmtt/manyChecks@30` (CHK-A1), deciding: q(-1T12b)b4 -> K(-1T12b)e1 [active, ahead of the present; same board]. Others: q(-1T12b)b4 -> K(-1T9b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)].
- `cwmtt/manyChecks@32` (CHK-A1), deciding: q(-1T12b)b4 -> K(-1T12b)e1 [active, ahead of the present; same board]. Others: q(-1T12b)b4 -> K(-1T9b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)].
- `cwmtt/manyChecks@40` (CHK-A1), deciding: q(-1T12b)b4 -> K(-1T12b)e1 [active, ahead of the present; same board]. Others: q(-1T12b)b4 -> K(-4T12b)e1 [active, ahead of the present; cross-board onto another optional board (three passes: -4, -3, -1)]; q(-1T12b)b4 -> K(-1T9b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)].
- `cwmtt/silly@16` (CHK-A4), deciding: n(0T6b)e3 -> K(-1T6b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]. Others: n(0T6b)e3 -> K(+1T6b)e1 [active, ahead of the present; cross-board onto another optional board (two passes: 0, +1)]; n(0T6b)e3 -> K(0T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)].
- `cwmtt/silly@28` (CHK-A1), deciding: q(-2T7b)h4 -> K(-2T7b)e1 [active, ahead of the present; same board]. Others: q(-2T7b)h4 -> K(+1T7b)e1 [active, ahead of the present; cross-board onto a history board (active branch)].
- `cwmtt/silly@30` (CHK-A1), deciding: q(-2T7b)h4 -> K(-2T7b)e1 [active, ahead of the present; same board]. Others: q(-2T7b)h4 -> K(+1T7b)e1 [active, ahead of the present; cross-board onto a history board (active branch)].
- `cwmtt/silly@32` (CHK-A1), deciding: q(-2T7b)h4 -> K(-2T7b)e1 [active, ahead of the present; same board]. Others: q(-2T7b)h4 -> K(+1T7b)e1 [active, ahead of the present; cross-board onto a history board (active branch)].
- `cwmtt/silly@34` (CHK-A1), deciding: q(-2T7b)h4 -> K(-2T7b)e1 [active, ahead of the present; same board]. Others: q(-2T7b)h4 -> K(+1T7b)e1 [active, ahead of the present; cross-board onto a history board (active branch)].
- `cwmtt/wide@12` (CHK-I1), deciding: q(-4T3b)f2 -> K(-4T3b)e1 [inactive, behind the present; same board]. Others: q(-4T3b)f2 -> K(-3T3b)e1 [inactive, behind the present; cross-board onto a history board (inactive branch)]; q(-4T3b)f2 -> K(-3T4b)e1 [inactive, behind the present; cross-board onto a history board (inactive branch)]; q(-4T3b)f2 -> K(-3T2b)e1 [inactive, behind the present; cross-board onto a history board (inactive branch)].
- `cwmtt/wide@15` (CHK-A2), deciding: N(-3T7w)e6 -> k(-3T6w)e8 [active, ahead of the present; cross-board onto a history board (active branch)]. Others: N(-3T7w)e6 -> k(-2T7w)e8 [active, ahead of the present; cross-board onto another optional board (two passes: -3, -2)].
- `cwmtt/wide@17` (CHK-A2), deciding: N(-3T7w)e6 -> k(-3T6w)e8 [active, ahead of the present; cross-board onto a history board (active branch)]. Others: N(-3T7w)e6 -> k(-2T7w)e8 [active, ahead of the present; cross-board onto another optional board (two passes: -3, -2)].
- `cwmtt/wide@19` (CHK-A2), deciding: N(-3T7w)e6 -> k(-3T6w)e8 [active, ahead of the present; cross-board onto a history board (active branch)]. Others: N(-3T7w)e6 -> k(-2T7w)e8 [active, ahead of the present; cross-board onto another optional board (two passes: -3, -2)].
- `cwmtt/wide@24` (CHK-A1), deciding: q(+2T7b)f2 -> K(+2T7b)e1 [active, ahead of the present; same board]. Others: n(-4T7b)e3 -> K(-3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; n(-4T7b)e3 -> K(-4T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+1T7b)e3 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head (two passes: +1, +2)]; q(+1T7b)e3 -> K(-1T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+1T7b)e3 -> K(+3T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+1T7b)e3 -> K(-1T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+2T7b)f2 -> K(+1T7b)e1 [active, ahead of the present; cross-board onto another optional board (two passes: +1, +2)]; q(+2T7b)f2 -> K(+2T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+3T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+1T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)].
- `cwmtt/wide@28` (CHK-A1), deciding: q(+2T7b)f2 -> K(+2T7b)e1 [active, ahead of the present; same board]. Others: n(-4T7b)e3 -> K(-3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; n(-4T7b)e3 -> K(-4T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+1T7b)e3 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head (two passes: +1, +2)]; q(+1T7b)e3 -> K(-1T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+1T7b)e3 -> K(+3T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+1T7b)e3 -> K(-1T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+2T7b)f2 -> K(+1T7b)e1 [active, ahead of the present; cross-board onto another optional board (two passes: +1, +2)]; q(+2T7b)f2 -> K(+2T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+3T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+1T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)].
- `cwmtt/wide@30` (CHK-A1), deciding: q(+2T7b)f2 -> K(+2T7b)e1 [active, ahead of the present; same board]. Others: n(-4T7b)e3 -> K(-3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; n(-4T7b)e3 -> K(-4T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+1T7b)e3 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head (two passes: +1, +2)]; q(+1T7b)e3 -> K(-1T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+1T7b)e3 -> K(+3T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+1T7b)e3 -> K(-1T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+2T7b)f2 -> K(+1T7b)e1 [active, ahead of the present; cross-board onto another optional board (two passes: +1, +2)]; q(+2T7b)f2 -> K(+2T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+3T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+1T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)].
- `cwmtt/wide@32` (CHK-A1), deciding: q(+2T7b)f2 -> K(+2T7b)e1 [active, ahead of the present; same board]. Others: n(-4T7b)e3 -> K(-3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; n(-4T7b)e3 -> K(-4T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+1T7b)e3 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head (two passes: +1, +2)]; q(+1T7b)e3 -> K(-1T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+1T7b)e3 -> K(+3T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+1T7b)e3 -> K(-1T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+2T7b)f2 -> K(+1T7b)e1 [active, ahead of the present; cross-board onto another optional board (two passes: +1, +2)]; q(+2T7b)f2 -> K(+2T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+3T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+1T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)].
- `cwmtt/wide@35` (CHK-A4), deciding: Q(+5T7w)e4 -> k(+1T7w)e8 [active, ahead of the present; cross-board onto an attacker-to-move head]. Others: Q(+5T7w)e4 -> k(+1T3w)e8 [active, ahead of the present; cross-board onto a history board (active branch)].
- `cwmtt/wide@36` (CHK-A1), deciding: q(+2T7b)f2 -> K(+2T7b)e1 [active, ahead of the present; same board]. Others: n(-4T7b)e3 -> K(-3T7b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; n(-4T7b)e3 -> K(-4T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+1T7b)e3 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head (two passes: +1, +2)]; q(+1T7b)e3 -> K(-1T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+1T7b)e3 -> K(+3T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+1T7b)e3 -> K(-1T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+2T7b)f2 -> K(+1T7b)e1 [active, ahead of the present; cross-board onto another optional board (two passes: +1, +2)]; q(+2T7b)f2 -> K(+2T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+3T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+1T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)].
- `cwmtt/wide@38` (CHK-A1), deciding: q(+2T7b)f2 -> K(+2T7b)e1 [active, ahead of the present; same board]. Others: n(-4T7b)e3 -> K(-3T7b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)]; n(-4T7b)e3 -> K(-4T6b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)]; q(+1T7b)e3 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head (two passes: +1, +2)]; q(+1T7b)e3 -> K(-1T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+1T7b)e3 -> K(+3T5b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)]; q(+1T7b)e3 -> K(-1T5b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)]; q(+2T7b)f2 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+2T7b)f2 -> K(+1T7b)e1 [active, ahead of the present; cross-board onto another optional board (two passes: +1, +2)]; q(+2T7b)f2 -> K(+2T6b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)]; q(+2T7b)f2 -> K(+3T6b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)]; q(+2T7b)f2 -> K(+1T6b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)].
- `cwmtt/wide@44` (CHK-A1), deciding: q(+2T7b)f2 -> K(+2T7b)e1 [active, ahead of the present; same board]. Others: n(-4T7b)e3 -> K(-3T7b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; n(-4T7b)e3 -> K(-5T7b)e1 [active, ahead of the present; cross-board onto another optional board (two passes: -5, -4)]; n(-4T7b)e3 -> K(-4T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+1T7b)e3 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head (two passes: +1, +2)]; q(+1T7b)e3 -> K(-1T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+1T7b)e3 -> K(+3T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+1T7b)e3 -> K(-1T5b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+2T7b)f2 -> K(+1T7b)e1 [active, ahead of the present; cross-board onto another optional board (two passes: +1, +2)]; q(+2T7b)f2 -> K(+2T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+3T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)]; q(+2T7b)f2 -> K(+1T6b)e1 [active, ahead of the present; cross-board onto a history board (active branch)].
- `cwmtt/wide@46` (CHK-A1), deciding: q(+2T7b)f2 -> K(+2T7b)e1 [active, ahead of the present; same board]. Others: n(-4T7b)e3 -> K(-3T7b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)]; n(-4T7b)e3 -> K(-5T7b)e1 [active, ahead of the present; cross-board onto another optional board (two passes: -5, -4)]; n(-4T7b)e3 -> K(-4T6b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)]; q(+1T7b)e3 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head (two passes: +1, +2)]; q(+1T7b)e3 -> K(-1T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+1T7b)e3 -> K(+3T5b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)]; q(+1T7b)e3 -> K(-1T5b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)]; q(+2T7b)f2 -> K(+3T7b)e1 [active, ahead of the present; cross-board onto an attacker-to-move head]; q(+2T7b)f2 -> K(+1T7b)e1 [active, ahead of the present; cross-board onto another optional board (two passes: +1, +2)]; q(+2T7b)f2 -> K(+2T6b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)]; q(+2T7b)f2 -> K(+3T6b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)]; q(+2T7b)f2 -> K(+1T6b)e1 [active, ahead of the present; cross-board onto a history board (inactive branch)].
- `notation/game-2@43` (CHK-A1), deciding: Q(-1T16w)d6 -> k(-1T16w)e7 [active, ahead of the present; same board]. Others: Q(-1T16w)d6 -> k(-1T15w)e7 [active, ahead of the present; cross-board onto a history board (active branch)]; Q(-1T16w)d6 -> k(-2T15w)e7 [active, ahead of the present; cross-board onto a mandatory board's phantom (two passes: -2, -1)]; Q(0T16w)d7 -> k(-1T16w)e7 [active, ahead of the present; cross-board onto another optional board (two passes: -1, 0)]; Q(0T16w)d7 -> k(-1T15w)e7 [active, ahead of the present; cross-board onto a history board (active branch)].

The one mixed position is `cwmtt/100_timelines@20`.

- **Inactive attackers ahead of the present:** three captures, one from -54 and two from -51. They are visible only under reading 5.
- **Active attackers ahead of the present:** three same-board captures, two on +27 and one on +48. They are visible under readings 4 and 5.
- Every reading that passes the inactive boards also passes the active ones, so the active captures decide the position. It is filed in CHK-A1.

## Q1 at the 78 positions

Positions where Q1's "consecutive" or per-turn reading would change anything, against T10's count rule. "Moves" means the position's filing, readings or side to move change. "Record consistent" means the record's side to move agrees with the reading at every position up to this one. Where it does not, the position stands under that reading only as a set-up: the boards and the creation order.

| Position | Cluster | Reading | Moves | Record consistent | What changes |
|---|---|---|---|---|---|
| `cwmtt/100_timelines@20` | CHK-A1 | consec | no | yes | mandatory -41, -32, -27 -> -27; inactive: -50 ... -28 (23 timelines) |
| `cwmtt/100_timelines@20` | CHK-A1 | perturn | no | yes | mandatory -41, -32, -27 -> -27; inactive: -50 ... -28 (23 timelines) |
| `cwmtt/niceAndrey@56` | CHK-A1 | consec | no | yes | attacker's next branch inactive |
| `cwmtt/niceAndrey@58` | CHK-A1 | consec | no | yes | attacker's next branch inactive |
| `cwmtt/niceAndrey@60` | CHK-A1 | consec | no | yes | attacker's next branch inactive |
| `cwmtt/niceAndrey@64` | CHK-A1 | consec | yes | yes | inactive: -5; attacker's next branch inactive; readings R4, R5 -> R5 |
| `cwmtt/niceAndrey@64` | CHK-A1 | perturn | yes | yes | inactive: -5; readings R4, R5 -> R5 |
| `cwmtt/niceAndrey@65` | CHK-A3 | consec | no | yes | inactive: -5 |
| `cwmtt/niceAndrey@65` | CHK-A3 | perturn | no | yes | inactive: -5 |
| `cwmtt/niceAndrey@66` | CHK-A1 | consec | yes | yes | inactive: -5; attacker's next branch inactive; readings R4, R5 -> R5 |
| `cwmtt/niceAndrey@66` | CHK-A1 | perturn | yes | yes | inactive: -5; readings R4, R5 -> R5 |
| `cwmtt/niceAndrey@68` | CHK-A1 | consec | yes | yes | inactive: -5; attacker's next branch inactive; readings R4, R5 -> R5 |
| `cwmtt/niceAndrey@68` | CHK-A1 | perturn | yes | yes | inactive: -5; readings R4, R5 -> R5 |
| `cwmtt/niceAndrey@90` | CHK-A1 | consec | yes | yes | inactive: -9; attacker's next branch inactive; readings R4, R5 -> R5 |
| `cwmtt/niceAndrey@91` | CHK-A3 | consec | no | yes | inactive: -9 |
| `cwmtt/niceAndrey@92` | CHK-A1 | consec | yes | yes | inactive: -9; attacker's next branch inactive; readings R4, R5 -> R5 |
| `cwmtt/niceAndrey@94` | CHK-A1 | consec | yes | yes | inactive: -9; attacker's next branch inactive; readings R4, R5 -> R5 |
| `cwmtt/niceAndrey@116` | CHK-A1 | consec | yes | yes | inactive: -12; attacker's next branch inactive; readings R4, R5 -> R5 |
| `cwmtt/niceAndrey@117` | CHK-A1 | consec | no | yes | inactive: -12 |
| `cwmtt/niceAndrey@118` | CHK-A1 | consec | yes | yes | inactive: -12; attacker's next branch inactive; readings R4, R5 -> R5 |
| `cwmtt/wide@15` | CHK-A2 | consec | yes | yes | attacker's next branch inactive; deciding capture N(-3T7w)e6 -> k(-2T7w)e8; readings unchanged (R4, R5) |
| `cwmtt/wide@17` | CHK-A2 | consec | yes | yes | attacker's next branch inactive; deciding capture N(-3T7w)e6 -> k(-2T7w)e8; readings unchanged (R4, R5) |
| `cwmtt/wide@19` | CHK-A2 | consec | yes | yes | attacker's next branch inactive; deciding capture N(-3T7w)e6 -> k(-2T7w)e8; readings unchanged (R4, R5) |
| `cwmtt/wide@24` | CHK-A1 | consec | yes | no (departs at 23) | present T6w -> T7w; mandatory +4 -> -4, +1, +2; inactive: +4; readings R4, R5 -> R1/R3, R2, R4, R5, R6 |
| `cwmtt/wide@28` | CHK-A1 | consec | yes | no (departs at 23) | present T4w -> T7w; mandatory +5 -> -4, +1, +2; inactive: +4, +5; readings R4, R5 -> R1/R3, R2, R4, R5, R6 |
| `cwmtt/wide@30` | CHK-A1 | consec | yes | no (departs at 23) | present T5w -> T7w; mandatory +5 -> -4, +1, +2; inactive: +4, +5; readings R4, R5 -> R1/R3, R2, R4, R5, R6 |
| `cwmtt/wide@32` | CHK-A1 | consec | yes | no (departs at 23) | present T6w -> T7w; mandatory +5 -> -4, +1, +2; inactive: +4, +5; readings R4, R5 -> R1/R3, R2, R4, R5, R6 |
| `cwmtt/wide@35` | CHK-A4 | consec | yes | no (departs at 23) | inactive: +5; readings R4, R5 -> R5 |
| `cwmtt/wide@36` | CHK-A1 | consec | no | no (departs at 23) | inactive: +5; attacker's next branch inactive |
| `cwmtt/wide@38` | CHK-A1 | consec | yes | no (departs at 23) | side to move white -> black: the check question asked of this position does not arise; present T6w -> T6b; mandatory -6 -> -5, +5; inactive: -6 |
| `cwmtt/wide@44` | CHK-A1 | consec | no | no (departs at 23) | attacker's next branch inactive |
| `cwmtt/wide@46` | CHK-A1 | consec | yes | no (departs at 23) | side to move white -> black: the check question asked of this position does not arise; present T6w -> T6b; mandatory -8 -> +7; inactive: -8 |

## All 78 positions

| Position | To move | Present | Mandatory | Cluster | Deciding capture | Other captures | ftxi / ftxi:naive / chessin5d in_check | Count ftxi / cwmtt |
|---|---|---|---|---|---|---|---|---|
| `cwmtt/100_timelines@13` | black | T6b | -1, +1, +2 | CHK-I1 | Q(+3T6w)e3 -> k(+3T6w)e6 | - | true / true / false | 1000 / 1000 |
| `cwmtt/100_timelines@15` | black | T5b | -2 | CHK-A1 | B(+4T8w)f5 -> k(+4T8w)e6 | - | true / true / false | 1000 / 1000 |
| `cwmtt/100_timelines@20` | white | T7w | -41, -32, -27 | CHK-A1 | n(+27T10b)f3 -> K(+27T10b)d2 | 5 | true / no answer / false | 1000 / 1000 |
| `cwmtt/NP@77` | black | T31b | -4 | CHK-A1 | R(-3T34w)g2 -> k(-3T34w)b2 | 6 | true / true / false | 1000 / 1000 |
| `cwmtt/NP@79` | black | T32b | -4 | CHK-A1 | R(-1T34w)g5 -> k(-1T34w)g8 | 2 | true / true / false | 1000 / 1000 |
| `cwmtt/NP0@77` | black | T31b | -4 | CHK-A1 | R(-3T34w)g2 -> k(-3T34w)b2 | 6 | true / true / false | 1000 / 1000 |
| `cwmtt/NP0@79` | black | T32b | -4 | CHK-A1 | R(-1T34w)g5 -> k(-1T34w)g8 | 2 | true / true / false | 1000 / 1000 |
| `cwmtt/exiledKings@12` | white | T3w | -1 | CHK-I2 | n(-2T2b)f4 -> K(-1T2b)f2 | - | true / true / false | 1000 / 1000 |
| `cwmtt/exiledKings@14` | white | T4w | -1 | CHK-I2 | n(-2T2b)f4 -> K(-1T2b)f2 | - | true / true / false | 1000 / 1000 |
| `cwmtt/exiledKings@16` | white | T5w | -1, 0 | CHK-I2 | n(-2T2b)f4 -> K(-1T2b)f2 | - | true / true / false | 1000 / 1000 |
| `cwmtt/exiledKings@18` | white | T6w | -1, 0 | CHK-I2 | n(-2T2b)f4 -> K(-1T2b)f2 | - | true / no answer / false | 1000 / 1000 |
| `cwmtt/manyChecks@20` | white | T9w | -2, -1, 0, +1 | CHK-I1 | b(-3T8b)g3 -> K(-3T8b)e1 | - | true / true / false | 1000 / 1000 |
| `cwmtt/manyChecks@28` | white | T9w | -3, +2 | CHK-A1 | q(-1T12b)b4 -> K(-1T12b)e1 | 1 | true / true / false | 1000 / 1000 |
| `cwmtt/manyChecks@30` | white | T10w | -3, +2 | CHK-A1 | q(-1T12b)b4 -> K(-1T12b)e1 | 1 | true / true / false | 1000 / 1000 |
| `cwmtt/manyChecks@32` | white | T11w | -3, +2 | CHK-A1 | q(-1T12b)b4 -> K(-1T12b)e1 | 1 | true / true / false | 1000 / 1000 |
| `cwmtt/manyChecks@40` | white | T11w | -5 | CHK-A1 | q(-1T12b)b4 -> K(-1T12b)e1 | 2 | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@41` | black | T9b | -2 | CHK-A5 | N(0T15w)d7 -> y(+2T15w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@43` | black | T10b | -2 | CHK-A5 | N(0T15w)d7 -> y(+2T15w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@45` | black | T11b | -2 | CHK-A5 | N(0T15w)d7 -> y(+2T15w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@47` | black | T12b | -2, +3 | CHK-A5 | N(0T15w)d7 -> y(+2T15w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@49` | black | T13b | -2, +3 | CHK-A5 | N(0T15w)d7 -> y(+2T15w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@56` | white | T10w | +4 | CHK-A1 | b(-3T13b)c2 -> Y(-3T13b)d1 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@57` | black | T10b | +4 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@58` | white | T11w | +4 | CHK-A1 | b(-3T13b)c2 -> Y(-3T13b)d1 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@60` | white | T12w | +4 | CHK-A1 | b(-3T13b)c2 -> Y(-3T13b)d1 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@61` | black | T12b | +4 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@63` | black | T13b | -3, +4 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@64` | white | T10w | +5 | CHK-A1 | b(-5T13b)c2 -> Y(-5T13b)d1 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@65` | black | T10b | +5 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@66` | white | T11w | +5 | CHK-A1 | b(-5T13b)c2 -> Y(-5T13b)d1 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@68` | white | T12w | +5, +6 | CHK-A1 | b(-5T13b)c2 -> Y(-5T13b)d1 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@69` | black | T12b | +5, +6 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@71` | black | T13b | -5, +4, +5, +6 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@75` | black | T15b | -5, -3, +4, +5, +6 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@77` | black | T13b | +8 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@79` | black | T14b | -7, +8 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@81` | black | T15b | -7, -3, +4, +5, +6, +7, +8 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@87` | black | T15b | +9 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@90` | white | T12w | +10 | CHK-A1 | b(-9T16b)f2 -> Y(-9T16b)g1 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@91` | black | T12b | +10 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@92` | white | T13w | +10 | CHK-A1 | b(-9T16b)f2 -> Y(-9T16b)g1 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@94` | white | T14w | +10 | CHK-A1 | b(-9T16b)f2 -> Y(-9T16b)g1 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@95` | black | T14b | +10 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@96` | white | T15w | +10 | CHK-A1 | b(-9T16b)f2 -> Y(-9T16b)g1 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@97` | black | T15b | +10 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / true / false | 1000 / 1000 |
| `cwmtt/niceAndrey@101` | black | T13b | +11 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / no answer / false | 1000 / 1000 |
| `cwmtt/niceAndrey@103` | black | T14b | +11 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / no answer / false | 1000 / 1000 |
| `cwmtt/niceAndrey@105` | black | T15b | +11 | CHK-A3 | B(+2T17w)d7 -> y(+2T16w)d8 | - | true / no answer / false | 1000 / 1000 |
| `cwmtt/niceAndrey@111` | black | T15b | +12 | CHK-A1 | N(-7T18w)b6 -> y(-7T18w)a8 | 2 | true / no answer / false | 1000 / 1000 |
| `cwmtt/niceAndrey@113` | black | T16b | +12 | CHK-A1 | N(-7T18w)b6 -> y(-7T18w)a8 | 2 | true / no answer / false | 1000 / 1000 |
| `cwmtt/niceAndrey@116` | white | T16w | +13 | CHK-A1 | n(-12T18b)e2 -> Y(-12T18b)g1 | - | true / no answer / false | 1000 / 1000 |
| `cwmtt/niceAndrey@117` | black | T16b | +13 | CHK-A1 | N(-7T18w)b6 -> y(-7T18w)a8 | 2 | true / no answer / false | 1000 / 1000 |
| `cwmtt/niceAndrey@118` | white | T17w | +13 | CHK-A1 | n(-12T18b)e2 -> Y(-12T18b)g1 | - | true / no answer / false | 1000 / 1000 |
| `cwmtt/silly@16` | white | T5w | +2 | CHK-A4 | n(0T6b)e3 -> K(-1T6b)e1 | 2 | true / true / false | 1000 / 1000 |
| `cwmtt/silly@28` | white | T2w | +3 | CHK-A1 | q(-2T7b)h4 -> K(-2T7b)e1 | 1 | true / true / false | 37 / 37 |
| `cwmtt/silly@30` | white | T3w | +3 | CHK-A1 | q(-2T7b)h4 -> K(-2T7b)e1 | 1 | true / true / false | 57 / 57 |
| `cwmtt/silly@32` | white | T4w | +3 | CHK-A1 | q(-2T7b)h4 -> K(-2T7b)e1 | 1 | true / true / false | 86 / 86 |
| `cwmtt/silly@34` | white | T5w | +3 | CHK-A1 | q(-2T7b)h4 -> K(-2T7b)e1 | 1 | true / true / false | 110 / 110 |
| `cwmtt/standard@13` | black | T3b | +1 | CHK-I2 | N(+2T3w)f5 -> k(+1T3w)f7 | - | true / true / false | 1000 / 1000 |
| `cwmtt/standard@15` | black | T4b | +1 | CHK-I2 | N(+2T3w)f5 -> k(+1T3w)f7 | - | true / true / false | 1000 / 1000 |
| `cwmtt/wide@12` | white | T5w | -3, -2, -1, 0, +1, +2 | CHK-I1 | q(-4T3b)f2 -> K(-4T3b)e1 | 3 | true / no answer / false | 1000 / 1000 |
| `cwmtt/wide@15` | black | T3b | -4 | CHK-A2 | N(-3T7w)e6 -> k(-3T6w)e8 | 1 | true / no answer / false | 1000 / 1000 |
| `cwmtt/wide@16` | white | T4w | -4 | CHK-A1 | b(+2T6b)d2 -> K(+2T6b)e1 | - | true / no answer / false | 1000 / 1000 |
| `cwmtt/wide@17` | black | T4b | -4 | CHK-A2 | N(-3T7w)e6 -> k(-3T6w)e8 | 1 | true / no answer / false | 1000 / 1000 |
| `cwmtt/wide@18` | white | T5w | -4, +3 | CHK-A1 | b(+2T6b)d2 -> K(+2T6b)e1 | - | true / no answer / false | 1000 / 1000 |
| `cwmtt/wide@19` | black | T5b | -4, +3 | CHK-A2 | N(-3T7w)e6 -> k(-3T6w)e8 | 1 | true / no answer / false | 1000 / 1000 |
| `cwmtt/wide@24` | white | T6w | +4 | CHK-A1 | q(+2T7b)f2 -> K(+2T7b)e1 | 11 | true / no answer / false | 462 / 462 |
| `cwmtt/wide@28` | white | T4w | +5 | CHK-A1 | q(+2T7b)f2 -> K(+2T7b)e1 | 11 | true / no answer / false | 120 / 120 |
| `cwmtt/wide@30` | white | T5w | +5 | CHK-A1 | q(+2T7b)f2 -> K(+2T7b)e1 | 11 | true / no answer / false | 186 / 186 |
| `cwmtt/wide@32` | white | T6w | +5 | CHK-A1 | q(+2T7b)f2 -> K(+2T7b)e1 | 11 | true / no answer / false | 372 / 372 |
| `cwmtt/wide@35` | black | T5b | -5 | CHK-A4 | Q(+5T7w)e4 -> k(+1T7w)e8 | 1 | true / no answer / false | 1000 / 1000 |
| `cwmtt/wide@36` | white | T6w | -5 | CHK-A1 | q(+2T7b)f2 -> K(+2T7b)e1 | 11 | true / no answer / false | 1000 / 1000 |
| `cwmtt/wide@38` | white | T6w | -6 | CHK-A1 | q(+2T7b)f2 -> K(+2T7b)e1 | 11 | true / no answer / false | 1000 / 1000 |
| `cwmtt/wide@44` | white | T6w | -7 | CHK-A1 | q(+2T7b)f2 -> K(+2T7b)e1 | 12 | true / no answer / false | 1000 / 1000 |
| `cwmtt/wide@46` | white | T6w | -8 | CHK-A1 | q(+2T7b)f2 -> K(+2T7b)e1 | 12 | true / no answer / false | 1000 / 1000 |
| `notation/Bg2@19` | black | T3b | -1 | CHK-A2 | Q(+1T6w)f3 -> k(+1T2w)f7 | - | true / true / false | 62 / 62 |
| `notation/Bg2@21` | black | T4b | -1 | CHK-A2 | Q(+1T6w)f3 -> k(+1T2w)f7 | - | true / true / false | 79 / 79 |
| `notation/game-2@43` | black | T14b | -2 | CHK-A1 | Q(-1T16w)d6 -> k(-1T16w)e7 | 4 | true / true / false | 194 / - |

## Mate traces

### `cwmtt/chessin5d-illegal-move-example@66`

**Verdicts.** terminal: chessin5d checkmate; ftxi none (25 legal actions), ftxi:naive none (25), cwmtt none (25), cwmtt:naive none (25). in_check: ftxi true, ftxi:naive true, chessin5d true - the same single capture, r(0T33b)a1 x K(0T33b)b1; cwmtt has no check verdict.

**Trace.** Another cause: a defect in 5d-chess-js's checkmate search (mateFuncs.checkmate, src/mate.js) - not a check cluster and not a parse divergence. The rule that settles it: C5 (high), the game ends only when the player to move has no legal turn - "a sequence of moves ... that shifts the present to the opponent and leaves all the player's royal pieces safe"; checkmate is that, in check (C6). (1) Same state: every source's fingerprint is bee3cf6806ef94e95583a1da14ef80f4. (2) No check cluster: White's two playable timelines, 0 and +1, are both mandatory (present T33w; +2 is Black's to move), so ftxi and 5d-chess-js advance the same timelines {0, +1} and report the same capture; the in_check verdicts agree. (3) A legal turn exists, checked against the rules for one listed action. Its moves are played in the only order that is possible (L5, J6): K(+1T33w)f6>(+1T33w)e5, then K(0T33w)b1>(+1T33w)a1, capturing the rook. Both are king steps (4.4). The second lands on +1's T33w board after +1 has been moved on, so it branches (J6) and creates +3, White's third timeline, inactive by T10 because Black has created none. Both mandatory boards are moved on, so the present advances to T33b, Black's colour (T4, T5). Afterwards no board is left on which White is to move, so no pass enters the C4 test. No Black piece on a Black-to-move board - (0T33b) k d1, r a1, r h2; (+1T33b) k b3, r a1, r b8; (+2T24b) k e7, r a8, r b8; (+3T33b) k b3, r b8 - can capture any White royal on any Black-colour board, history boards included (C2, C3, C4; 47 royal placements). This was checked with final/c4_check.py, an enumerator written here from docs/rules that reproduces ftxi's capture lists at all 78 in_check positions with every mover head passed and finds none with only the mandatory boards passed. The enumeration counts every Black-to-move board, inactive ones included (no Black board lies ahead of the present), and every branching capture, so no reading of Q10's open items (which threats count) changes the C4 result; with no White board left to pass, no passed-set reading enters. All 25 listed actions pass the same T5 and C4 test. By C5 the position is not checkmate. (4) The implementations agree: ftxi, ftxi:naive, cwmtt, cwmtt:naive and 5d-chess-js's own enumerator list the same 25 actions (raw.json actions[0].comparison: all pairwise equal). The enumerator tested all 1109 sequences (complete) and accepted 25 under the library's per-action tests: validateFuncs.move for each move (move()), mateFuncs.checks (inCheck) and boardFuncs.present (submittable()'s present test). The stock submit() refuses all 25, because it first asks inCheckmate (index.js:501; submittable() and moves() answer false and [] when inCheckmate, index.js:520 and 448-450), and inCheckmate is the defective checkmate() of (5). Re-run here (review_traces/submit_probe.js, the pinned library, hash equal to the golden fingerprint; final/submit_illegal66.json): inCheckmate is true, chess.moves('raw') is empty, and after an accepted sequence submittable() is false and submit() throws "Cannot submit, currently in checkmate." The disagreement lies between the library's per-action legality tests and its mate search; it is not a rules reading. (5) Why the search misses them (cluster_work/mate_probe.js, the pinned library loaded in memory and unchanged; the position it reaches hashes to the golden fingerprint): there is 1 royal capture before any move, and none of the 56 single moves lowers that number (46 keep it, 10 raise it), so the single pass (mate.js:62-75) and the DFS, which follows only moves that strictly lower it (mate.js:77-101), find nothing. The "slow BFS exhaustive search" (mate.js:149-191) then hands its tree node {board, checkSig} - not node.board - to boardFuncs.moves (mate.js:157) and boardFuncs.copy (mate.js:161); with activeOnly and presentOnly false, moves() loops "for (l = 0; l < fullBoard.length; l++)" (board.js:370), and an object has no length, so it returns no move. The queue empties after one node and checkmate() returns [true, false] - checkmate, not a time-out - after 14 ms of its 60000 ms budget (8 ms on a re-run; stock run: 1 moves() call on a node, 0 moves returned). Re-run with moves()/copy() handed node.board, the same search returns [false, false] (not checkmate) after 2 nodes. (6) Reach of the defect: whenever the single pass and the DFS find no escape, checkmate() falls through to the BFS (mate.js:104-106 returns early only on [false, false] or a time-out), and the BFS never generates a move at any position, because its root is always a node {board, checkSig}. So every "checkmate" from 5d-chess-js means only that its single pass and its check-reducing DFS found no escape; none is the result of an exhaustive search. Corpus-wide check (final/mates_all.py, golden files): of 5d-chess-js's 15 golden checkmates, the other 13 are corroborated by another source - 10 by ftxi's checkmate (in check, no legal action) at the same position (cwmtt/NP0@83, cwmtt/silly@38, cwmtt/standard@16, cwmtt/wide@47, ftxi/futures@19, local/checkmate@1, notation/1.2@22, notation/Bg2@25, notation/e3@15, notation/game-2@45); cwmtt/futures@19 by cwmtt's no_action there and by ftxi's checkmate at ftxi/futures@19, which holds the same state (fingerprint e1bcc132dff9695dc02021bd11acd37c); cwmtt/test1@20 by cwmtt's and cwmtt:naive's no_action; and notation/aivsai@54, which only 5d-chess-js answers in the golden files (ftxi refuses the record, "non-consecutive turn number: 25b followed by 25w"; cwmtt stops at position 9), by ftxi run on a scratch copy with the repeated turn number renumbered (only the turn numbers "25." and "26." of the last two turns changed): it holds 5d-chess-js's state at all 55 positions and calls 54 checkmate, in check with no legal action (final/aivsai_check.py; final/aivsai_ftxi_fresh.jsonl reproduces review_traces/aivsai_ftxi_raw.jsonl). The two disagreements are therefore the defect's whole footprint in this corpus. A 5d-chess-js checkmate that no other source answers should be treated as unverified.

### `ftxi/number1@4`

**Verdicts.** terminal: chessin5d checkmate; ftxi none (21 legal actions), ftxi:naive none (21); cwmtt and cwmtt:naive no answer (the adapter refuses the record, unsupported: variant "Custom"). in_check: ftxi true, ftxi:naive true, chessin5d true - the same single capture, r(0T3b)e1 x K(+2T3b)e1.

**Trace.** Another cause: the same 5d-chess-js checkmate-search defect as at chessin5d-illegal-move-example@66 - not a check cluster and not a parse divergence. The rule that settles it: C5 (high), as at @66. (1) Same state: fingerprint 6c7ca4810883e4cf2285610963f1d904 from every source. (2) No check cluster: the four initial timelines -1, 0, +1 and +2 are all mandatory (present T3w), so both check tests advance the same boards and find the same capture. (3) A legal turn exists, checked against the rules for one listed action, played in this order: K(-1T3w)a5>(-1T2w)a5, then K(0T3w)a4>(+1T3w)a4, then K(+2T3w)e1>(+1T3w)e1. All three are king steps (4.4). The first goes one turn back on its own timeline, a branch (J3), and creates +3, White's first created timeline. It is active by T10 with the four initial timelines taken as created by nobody (Q2 is open), so the present moves back to its first board's column, T2b (T8). The second is a non-branching jump onto +1's playable board (J4). The third lands on +1's T3w board after +1 has been moved on, so it branches (J6) and creates +4, inactive (T10). Every mandatory board is moved on, so the present is T2b, Black's colour (T4, T5); it would be T3b, Black's colour too, if Q2 made +3 inactive. No White board is left to pass. No Black piece on a Black-to-move board - (-1T3b) k d5; (0T3b) k e5, r e1; (+1T3b) k e5, r d2, r d5; (+2T3b) k c3, r d2; (+3T2b) k e5; (+4T3b) k e5, r d2, r d5 - can capture any of the 13 White royal placements on Black-colour boards (C2, C3, C4; final/c4_check.py, as at @66). Black's boards at T3b lie ahead of the present: C3 (medium) counts pieces there, while TV Tropes' wording would not (Q10). The enumeration counts them, and every branching capture, so the C4 result holds under every reading. All 21 listed actions pass the same T5 and C4 test. By C5 the position is not checkmate. (4) ftxi, ftxi:naive and 5d-chess-js's own enumerator list the same 21 actions (the enumerator: 70060 sequences tested, complete, 42 accepted = 21 distinct actions, under the library's per-action tests). cwmtt and cwmtt:naive cannot read the record (unsupported: variant "Custom"), so they list nothing here. As at @66, the stock submit() refuses every one of them because it first asks inCheckmate (re-run: final/submit_number1_4.json - inCheckmate true, chess.moves('raw') empty, submittable() false, submit() throws "Cannot submit, currently in checkmate."). (5) Probe (cluster_work/mate_probe.js): 1 royal capture before any move; none of the 83 single moves lowers it (54 keep it, 29 raise it); the stock checkmate() expands one BFS node, gets 0 moves from boardFuncs.moves(node, ...) and returns [true, false] after 7 ms (5 ms on a re-run); with node.board passed it returns [false, false] after 312 nodes (294 ms). (6) The reach of the defect is stated at chessin5d-illegal-move-example@66, step (6).

## Count trace

### `notation/1.2@15`

**Extra actions.** (-1T3b)e7>(+1T3b)e7 and (-1T3b)g7>(+1T3b)g7: listed by cwmtt, cwmtt:naive and 5d-chess-js's enumerator (54 actions each), not by ftxi or ftxi:naive (52 each); the 52 are common to all five. Each is a one-move action by Black, whose only mandatory board is (-1T3b): an unmoved pawn (p* in every source's state) steps two timelines forward - Black's forward is towards increasing L (rules 4.6) - from (-1T3b) through (0T3b) to the same square of (+1T3b), a history board (+1's first board; its head is T5w), so the move branches (J3) and creates Black's second timeline (active by T10: White has created one). The square passed on (0T3b) and the landing square are empty.

**Trace.** Another cause: a colour asymmetry in ftxi's move generator - not a check cluster (in_check is false for every source; a count of whole legal actions) and not a parse divergence (same state, unmoved markers included, so ftxi itself holds both pawns unmoved). (1) In multiverse::gen_moves_impl (ftxi src/core/multiverse_base.cpp, the generator of moves off the board), the white pawn branch yields the timeline double step under "if constexpr(P == PAWN_UW || P == BRAWN_UW)" (line 865); the black pawn branch (from line 909) guards the same block with the same WHITE types (line 937), so for PAWN_UB and BRAWN_UB it is never compiled in: ftxi never generates a black pawn's or brawn's timeline double step. Its same-board generator, gen_physical_moves_impl, guards the black double step with PAWN_UB || BRAWN_UB (line 640), and ftxi's check scan records the pattern for brawns: "Match multiverse::gen_moves_impl: additional L-direction brawn captures are enabled only in its white pawn/brawn branch" (check_position.cpp:206-207). (2) ftxi's 52 actions include the same pawns' single timeline steps (-1T3b)e7>(0T3b)e7 and (-1T3b)g7>(0T3b)g7 and their same-board double steps e7-e5 and g7-g5. (3) ftxi does generate the white counterpart: at notation/game-2@26 it lists (+1T8w)e2>(-1T8w)e2, an unmoved white pawn's two-timeline step onto a history board, and its 98 actions equal 5d-chess-js's 98 (cluster_work/runs/game2_ftxi_a26.txt, game2_chessin5d_a26.txt). (4) In the 14 records, a black pawn timeline double step is open on the boards (unmoved pawn, both squares empty, both boards present) 119 times at 63 positions (cluster_work/find_black_ldouble.py, black_ldouble_counts.txt). At 60 of them ftxi and cwmtt both count the cap of 1000, and at one (cwmtt/100_timelines@21) both count 0. At one cwmtt has no count: notation/1.2@17, where ftxi's 53 is two short of 5d-chess-js's enumerator's 55 by the same two moves one turn later, (-1T4b)e7>(+1T4b)e7 and (-1T4b)g7>(+1T4b)g7 - outside the golden comparison, because cwmtt's run stops at position 15 and 5d-chess-js gives no count (fresh runs: final/notation_1.2_ftxi_a17.txt, final/notation_1.2_chessin5d_a17.txt). The only position where both count below the cap is notation/1.2@15, so no other golden count in these records can show it. (5) The ruling, kept apart from the code-level cause: two questions for the rulebook. (a) Does a pawn's timeline double step exist at all? 4.6 says an unmoved pawn may go "two steps forward along one axis, two ranks or two timelines", but the clause carries Q15's conflict: one guide [72] says pawns crossing timelines "can also only move one board at a time", against the Guide's "one or two squares" [80]. (b) Do pawns copied into a branch keep their unmoved state? Q15's lead is low; the -1 pawns are copies, and all three oracles hold them unmoved. J3: the step lands on a history board and branches. Under [72]'s reading the two actions are illegal and 52 is the count; under 4.6's main text, 54. Either way ftxi's two colours disagree with each other: it generates the white timeline double step (notation/game-2@26) and not the black one, so whichever way (a) is ruled, ftxi is wrong for one colour.

## Caveats

- Dominance for the one mixed position (cwmtt/100_timelines@20) rests on Q10's listed readings: every listed reading that passes inactive boards ahead of the present (only 5) also passes active ones. A reading that passed inactive but not active boards would move it; none is listed. Q1's alternative readings make 23 more of its timelines inactive and shrink its mandatory set from -41, -32, -27 to -27, but leave its filing and readings unchanged.
- In CHK-A4's two positions the non-branching jump and a branching capture of a royal's past position (at silly@16 the king of L0, the attacker's own timeline; at wide@35 the same +1 king), with an active branch, need the same single pass. The position is filed under the jump (no branching at all), and would sit in CHK-A2 under the other choice. Neither choice changes its verdict under any listed reading.
- Activity is T10's count rule (high) with T11. It is also what the oracles compute: 5d-chess-js's active list equals it at all 657 positions of these records, and cwmtt's active range (num_active, read as |L| <= n) and present equal it at the 611 positions cwmtt reached (final/cwmtt_active.py); ftxi reports no activity (its explain gives only phantom_timelines and checks). So the clustering describes the oracles' behaviour. That is not evidence for the count rule (spec 11.0: two oracles agreeing is not evidence on its own). Q1 is open: its "consecutive" and per-turn readings only ever make timelines inactive, and they change the filing of 18 CHK-A positions under "consecutive" and 3 under per-turn (rule_questions Q1; positions[].q1). Q2 does not arise at the 78 positions (every record starts with one timeline); Q3 is open and not modelled.
- The passes a capture needs include every board its slide crosses that exists only as a phantom (J7, J8). Nine captures need one more board passed than the first pass listed: q(+1T7b)e3 -> K(+3T7b)e1 crosses (+2T7b)e2 at the eight cwmtt/wide positions 24, 28, 30, 32, 36, 38, 44 and 46 (passes +1 and +2), and q(-1T12b)b4 -> K(-4T12b)e1 crosses (-3T12b)d2 at cwmtt/manyChecks@40 (passes -1, -3 and -4). Every added board is active and ahead of the present, so no capture's readings, no deciding capture and no filing change. Q18's dissent (checks through non-existing boards are unblockable) would drop the added pass; that changes nothing either.
- No terminal verdict in the corpus depends on these 78 positions: legal actions exist at all of them (ftxi and cwmtt counts equal wherever both answer: 10 below the cap, 67 at it). The check disagreement would decide checkmate against stalemate only at a position with no legal action (C5, C6).
- The C4 checks at the two mate positions and the validation of the readings use final/c4_check.py, a royal-capture enumerator written here from docs/rules. Its agreement with ftxi's capture lists validates the code, not the rules.

## Unexplained

Nothing. Every in_check position falls in a cluster. Both mate disagreements and the count disagreement trace to causes found in the oracles' own code and reproduced by runs.

## Review findings

How each finding of the review of the first pass was handled. Each was checked against `raw.json` and fresh runs before it was applied.

| # | Finding | Disposition | Note |
|---|---|---|---|
| 1 | passes needed ignored phantom boards crossed by a slide | applied | Confirmed exactly (final/paths.py): 9 captures, +2 at the 8 cwmtt/wide positions and -3 at manyChecks@40; no reading, deciding capture or filing changes (final/cluster.py compares with the first pass). |
| 2 | caveat cited ftxi's mandatory set | applied, count corrected | ftxi's explain has only phantom_timelines and checks (ftxi_driver.cpp emit_explain). cwmtt's num_active and present equal the count rule at all 611 positions with a cwmtt explain, not only the 79 in raw.json's positions; 5d-chess-js's list at 657. Kept as a data-consistency fact, not as evidence (finding 7). |
| 3 | CHK-A4: the branching capture at silly@16 hits a different royal | applied | -1 first board T3w; the other capture takes L0's king at (0T5b)e1. At wide@35 (+1T3w) is +1's history, the same king. |
| 4 | summary overstated cwmtt's agreement at ftxi/number1@4 | applied (with 15) | raw.json failures: cwmtt and cwmtt:naive unsupported, variant "Custom". |
| 5 | phantom boards listed unlabelled in the JSON details | applied | (+1T6b) at silly@16 and (+2T15w) at niceAndrey@41 are copies of heads (+1T6w) and (+2T14b); every other listed board is real. |
| 6 | representatives were not the smallest positions | applied by stating the rule used | Representatives kept; the method now states the hand-chosen criteria and names the smaller positions in CHK-A1 and CHK-I1. |
| 7 | Q1 bears on CHK-A filings | applied and extended | The 12 positions named are confirmed. The full model (final/q1_full.py, all 78 positions, present and side recomputed) finds 6 more under "consecutive": wide@24, 28, 30, 32 (present to T7w; attacker's board mandatory; in check under every pass reading) and wide@38, 46 (Black would be to move); and cwmtt/wide departs from "consecutive" at action 23. 18 positions in all; 3 under per-turn. |
| 8 | rule questions cited C6/C7 without their content | applied | Quoted with ratings; C6 (high) excludes readings 6 and 7; the ruling is put as confirm C6/C7, adopt reading 4 or adopt reading 5 - with Q1's exceptions. |
| 9 | CHK-A5's question mis-tied to C12 and reading 6 | applied | C12 and C6 quoted in full; CHK-A5 settled by RQ1 for both boards; the per-timeline test over optional boards named once as excluded by C6 (high), not tabled, so no visibility column is added. |
| 10 | CHK-A2 and CHK-A4 raise no rule question beyond CHK-A1's | applied | Four distinct rule questions (rule_questions) plus Q1; CHK-A2, A4 and A5 marked settled by RQ1 (C2, J4, C12/C6, spec 11.0 item 1); J7 cited only for the sliding captures (Bg2@19, 21). |
| 11 | count-trace rule pointer left out Q15's conflict | applied | Two questions: does a timeline double step exist ([72] against 4.6), do copied pawns keep it (Q15, low); the code-level cause kept apart. |
| 12 | mate traces named no rule | applied | C5 cited; one action at each position checked against T5 and C4 (final/c4_check.py, a rules-derived enumerator; all 25 and 21 pass); no Q10 reading affects the C4 test. |
| 13 | reach of the BFS defect | applied | Verified: 15 golden checkmates; 13 corroborated as stated; aivsai@54 re-run fresh on the renumbered copy (same states at all 55 positions; checkmate at 54). |
| 14 | submit() wording in step (3) | applied | index.js:501, 520, 448-450 read; submit_probe.js re-run at both positions: inCheckmate true, moves empty, submittable false, submit throws. |
| 15 | summary bullet on the enumerator's agreement | applied (with 4) | See finding 4. |
| 16 | find_black_ldouble.py header described the white scan | applied | Header corrected; notation/1.2@17 added to count-trace step (4) from fresh ftxi and 5d-chess-js runs (53 against 55, the same two moves; cwmtt stops at 15). |

## Files

- `oracles/.cache/reports/checkpoint1/raw.json`
- `oracles/.cache/reports/checkpoint1/final/run_all.sh`
- `oracles/.cache/reports/checkpoint1/final/common.py`
- `oracles/.cache/reports/checkpoint1/final/paths.py`
- `oracles/.cache/reports/checkpoint1/final/paths.json`
- `oracles/.cache/reports/checkpoint1/final/q1_full.py`
- `oracles/.cache/reports/checkpoint1/final/q1_full.json`
- `oracles/.cache/reports/checkpoint1/final/q1_sides.py`
- `oracles/.cache/reports/checkpoint1/final/cluster.py`
- `oracles/.cache/reports/checkpoint1/final/annotated.json`
- `oracles/.cache/reports/checkpoint1/final/build_clusters.py`
- `oracles/.cache/reports/checkpoint1/final/write_md.py`
- `oracles/.cache/reports/checkpoint1/final/c4_check.py`
- `oracles/.cache/reports/checkpoint1/final/c4_mates.py`
- `oracles/.cache/reports/checkpoint1/final/c4_movecheck.py`
- `oracles/.cache/reports/checkpoint1/final/cwmtt_active.py`
- `oracles/.cache/reports/checkpoint1/final/mates_all.py`
- `oracles/.cache/reports/checkpoint1/final/aivsai_check.py`
- `oracles/.cache/reports/checkpoint1/final/aivsai_ftxi_fresh.jsonl`
- `oracles/.cache/reports/checkpoint1/final/submit_illegal66.json`
- `oracles/.cache/reports/checkpoint1/final/submit_number1_4.json`
- `oracles/.cache/reports/checkpoint1/final/notation_1.2_ftxi_a17.txt`
- `oracles/.cache/reports/checkpoint1/final/notation_1.2_chessin5d_a17.txt`
- `oracles/.cache/reports/checkpoint1/cluster_work/mate_probe.js`
- `oracles/.cache/reports/checkpoint1/cluster_work/mate_probe_illegal66.json`
- `oracles/.cache/reports/checkpoint1/cluster_work/mate_probe_number1_4.json`
- `oracles/.cache/reports/checkpoint1/review_traces/mate_probe_illegal66_rerun.json`
- `oracles/.cache/reports/checkpoint1/review_traces/mate_probe_number1_rerun.json`
- `oracles/.cache/reports/checkpoint1/review_traces/submit_probe.js`
- `oracles/.cache/reports/checkpoint1/review_traces/aivsai_renumbered.5dpgn`
- `oracles/.cache/reports/checkpoint1/cluster_work/find_white_ldouble.py`
- `oracles/.cache/reports/checkpoint1/cluster_work/find_black_ldouble.py`
- `oracles/.cache/reports/checkpoint1/cluster_work/golden_counts.py`
- `oracles/.cache/reports/checkpoint1/cluster_work/black_ldouble_counts.txt`
- `oracles/.cache/reports/checkpoint1/cluster_work/white_ldouble.txt`
- `oracles/.cache/reports/checkpoint1/cluster_work/runs/game2_ftxi_a26.txt`
- `oracles/.cache/reports/checkpoint1/cluster_work/runs/game2_chessin5d_a26.txt`
- `oracles/.cache/reports/checkpoint1/final/previous/clusters.v1.json`
- `oracles/.cache/reports/checkpoint1/final/previous/clusters.v1.md`
