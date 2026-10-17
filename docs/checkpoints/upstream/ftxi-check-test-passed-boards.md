# Question: which boards should the check test pass? `state::phantom()` passes every board the player may play, not only those they must

- **Project:** ftxi/5dchess_engine, https://github.com/ftxi/5dchess_engine
- **Tested at:** commit `2ea9f1b82880f08faa9400d1ac20b20102604187` (tag v0.4.10): the stock `5dtools`, built with `-DTOOLS=ON -DCMAKE_BUILD_TYPE=Release`, and the library, both by g++ 16.2 on Windows 11; re-run 2026-10-17.
- **This is a question, not a bug report.** We read the rules one way and your engine reads them another. We would rather learn that our reading is wrong than report yours as a bug: if you know that the game behaves your engine's way, we want to change our reading.

## Summary

When a player has no legal turn, the check test decides between checkmate and stalemate: a player who would be in check "if they were to pass" is checkmated, otherwise stalemated. With several timelines the question is which of the player's boards that hypothetical pass covers.

Your engine passes **every timeline whose last board is the player's to play** (`state::phantom()`). Our rules reference passes only the **active boards in the present** - the boards the player must play this turn - following the sources below. In its words (ruling R1): "a royal is in check if, passing on all active boards in the present, the opponent could capture it from any of their playable boards. Optional boards ahead of the present and boards on inactive timelines are not passed. A royal capture counts even if it would branch into a timeline that would be inactive." The two readings differ when the only royal capture comes from an enemy piece standing on a board the player may play but need not:

1. a board of an **active** timeline that lies **ahead of the present**;
2. the same, where the capture would branch into a timeline that would be **inactive**;
3. a board of an **inactive** timeline (in the positions below, behind the present);
4. the same, where the capture would branch into an inactive timeline.

In all four your engine finds check and our reading does not. On public game records we found 92 such positions in 12 games, all four kinds among them (below). None of them decides a game result: a legal turn exists at each. But at a position with no legal turn, the two readings would call the same position checkmate (a loss) and stalemate (a draw).

**Our question:** does the game pass every board where the player is to move - as `phantom()` does - or only the active boards in the present? If you have evidence either way - a position played in the client, a developer statement, the source you followed - we would be glad to see it.

## The code

`state::phantom()` (`src/core/state.cpp`, lines 640-653) appends a copy of the last board of every timeline whose last board is the player's, whether or not that board must be played:

```cpp
        auto [t,c] = get_timeline_end(l);
        if(c == player)
```

Permalink: https://github.com/ftxi/5dchess_engine/blob/2ea9f1b82880f08faa9400d1ac20b20102604187/src/core/state.cpp#L640-L653

`5dtools checkmate` uses this test to tell the two endings apart (`s.phantom().find_checks(!player)`, `tools/position_tools.cpp` line 517), and `state::has_phantom_check()` is the same test without the copy. (`game::currently_check()` passes nothing; it is not what we are asking about.)

## Positions

Five positions from public game records, one of each kind (two of the first). Each is after the given number of actions, which is what `--at` selects. At each, `state::has_phantom_check()` is true and the capture shown is the one `phantom().find_checks(!player)` lists; your stock `5dtools stats --at <turn> --format json` agrees with the timeline counts given; and a legal turn exists. Captures are written `<piece>(<L>T<T><c>)<square> -> <royal>(<L>T<T><c>)<square>`, where `c` is the side to move on that board (in the attacker's case, on the copy `phantom()` makes - the real board has the other side to move), upper case is White, and `y` is a royal queen.

1. **An active board ahead of the present, same-board capture.** penteract/cwmtt `tests/5dpgn/silly.5dpgn` (https://github.com/penteract/cwmtt/blob/9632e9969dd455c6e3d6c27c57c1936c251d7e97/tests/5dpgn/silly.5dpgn), `--at 14b` (28 actions). White to move; the present is at T2, White's turn; timeline +3 must be played. `stats`: 6 timelines, all active, 1 mandatory, 1 optional. The optional one is -2, whose last board (-2T7) is White's to play, five turns ahead of the present.
   - Your capture: `q(-2T7b)h4 -> K(-2T7b)e1` - the black queen on (-2T7) takes White's king on the same board once White has passed that board. (It also lists `q(-2T7b)h4 -> K(+1T7b)e1`, the same queen across timelines.)
   - Our reading: White need not play (-2T7), and until White does, the queen there cannot move. The pass covers only +3, after which no capture exists: not in check.
2. **An active board ahead of the present, capture back in time.** adri326/5dchess-notation `test/Bg2.5dpgn` (https://github.com/adri326/5dchess-notation/blob/99f6163a9c95a6a9740b20fb5f3c7f6503a166a0/test/Bg2.5dpgn), `--at 10w` (19 actions). Black to move; the present is at T3, Black's turn; timeline -1 must be played. `stats`: 3 timelines, all active, 1 mandatory, 1 optional - timeline +1, whose last board (+1T5) is Black's to play.
   - Your capture: `Q(+1T6w)f3 -> k(+1T2w)f7` - the white queen on (+1T5) travels back four turns along its own timeline to the black king's past square.
   - Our reading: Black need not play (+1T5), so the queen there cannot move this turn: not in check.
3. **An active board ahead of the present, where the capture would create an inactive timeline.** penteract/cwmtt `tests/5dpgn/niceAndrey.5dpgn` (Standard - Reversed Royalty; https://github.com/penteract/cwmtt/blob/9632e9969dd455c6e3d6c27c57c1936c251d7e97/tests/5dpgn/niceAndrey.5dpgn), `--at 29w` (57 actions). Black to move; the present is at T10, Black's turn; timeline +4 must be played. `stats`: 17 timelines, 8 active and 9 inactive, 1 mandatory, 14 optional - among them +2, active, whose last board (+2T16) is Black's to play.
   - Your capture: `B(+2T17w)d7 -> y(+2T16w)d8` - the white bishop on (+2T16) takes Black's royal queen one turn back. White has created more timelines than Black, so the timeline this capture would create would be inactive.
   - Our reading: as in 2, the bishop's board need not be played, so it is not passed: not in check. (Whether a capture into a timeline that would be inactive counts does not arise; we count one where the attacker can move.)
4. **An inactive board behind the present, same-board capture.** penteract/cwmtt `tests/5dpgn/manyChecks.5dpgn` (https://github.com/penteract/cwmtt/blob/9632e9969dd455c6e3d6c27c57c1936c251d7e97/tests/5dpgn/manyChecks.5dpgn), `--at 10b` (20 actions). White to move; the present is at T9, White's turn; timelines -2, -1, 0 and +1 must be played. `stats`: 5 timelines, 4 active and 1 inactive, 4 mandatory, 1 optional. The inactive one is -3 - Black's third timeline, created when White had created one - whose last board (-3T8) is White's to play, a turn behind the present.
   - Your capture: `b(-3T8b)g3 -> K(-3T8b)e1` - the black bishop on (-3T8) takes White's king on the same board.
   - Our reading: an inactive timeline is never mandatory, and is not passed, so the bishop there cannot move this turn: not in check.
5. **An inactive board behind the present, capture into a timeline that would be inactive.** penteract/cwmtt `tests/5dpgn/standard.5dpgn` (https://github.com/penteract/cwmtt/blob/9632e9969dd455c6e3d6c27c57c1936c251d7e97/tests/5dpgn/standard.5dpgn), `--at 7w` (13 actions). Black to move; the present is at T3, Black's turn; timeline +1 must be played. `stats`: 3 timelines, 2 active and 1 inactive, 1 mandatory, 2 optional. The inactive one is +2 - White's second timeline, created when Black had created none - whose last board (+2T2) is Black's to play.
   - Your capture: `N(+2T3w)f5 -> k(+1T3w)f7` - the white knight on (+2T2) jumps to the black king on (+1T3), a past board of the timeline Black must play; it would branch into a timeline that would be inactive.
   - Our reading: as in 4: not in check.

5d-chess-js (`mateFuncs.checks` after `blankAction`, which advances the timelines of its `present()`) answers "not in check" at all five, as our reading does; penteract/cwmtt gives no check verdict at a position with a legal action.

## The sources

What we read, ranked as our rules reference ranks its sources (the in-game Guide first, then the developer's own statements, then established references, then forums):

- **In-game Guide, Checkmate tab** (the current text): "Stalemate can occur in these conditions if The Present cannot be shifted, but no king is immediately attacked." It does not say which boards are passed.
- **The older guide text that still ships in the client** ("Complete Rules and Justification"): "If a player has no way to end their turn, but they could pass on every board in the present without leaving themselves in check, they are stalemated rather than checkmated." That passes the boards *in the present* - none of the five attackers' boards is - but the client's 2020-10-07 update made "Big changes to the in-game guide", so we do not rely on it alone.
- **The developer**, quoted in "Check, Checkmate and Stalemate and why it can be so confusing" (https://docs.google.com/document/d/1NXRXd5Svvr4NBwQHQU4a83_KgjTZo-J9NdxDv7r_IMc), on why the check indicator ("!") covers one board: "The obvious alternative would be to pass on multiple boards at once, which would possibly be all the boards in the present."
- **Wikipedia, Rules** (https://en.wikipedia.org/wiki/5D_Chess_with_Multiverse_Time_Travel): "A player is in check in a situation where it is the player's turn and, if the player were to pass their move on all active boards in the present, then the opponent would be able to capture one of the player's kings." This is the sentence our reading follows: active boards, in the present.

And the evidence the other way, which is why we are asking rather than reporting:

- The same Google document assumes "passing means passing on all active boards", and reports of a client position that "the game was in accordance with Interpretation 3" - "a piece can capture once we pass on all boards".
- wiki.gg, Rules (https://5dchess.wiki.gg/wiki/Rules): "If forcibly skipping on all boards would leave one of the active player's royal pieces is in check, it is checkmate; otherwise, it is stalemate."

If the game passes every board where the player is to move, positions 1-5 are all in check. If it passes every *active* board, positions 1-3 are and 4-5 are not. If it passes only the active boards in the present, none is.

## How this was found

By differential testing: we compare check verdicts, legal-action counts and mate verdicts position by position across independent rules implementations, on public game records, and settle each difference from the sources rather than by vote. Your engine's check verdict differs from ours at 92 positions in 12 public games, and every one is a position where the only royal capture comes from an enemy piece on a board the player may leave unplayed, on one of the four kinds above.
