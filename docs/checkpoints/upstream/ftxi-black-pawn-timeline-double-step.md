# Black pawns and brawns never get the two-timeline double step (the Black branch of gen_moves_impl tests the White piece types)

- **Project:** ftxi/5dchess_engine, https://github.com/ftxi/5dchess_engine
- **Tested at:** commit `2ea9f1b82880f08faa9400d1ac20b20102604187` (tag v0.4.10), with the stock `5dtools` built with `-DTOOLS=ON -DCMAKE_BUILD_TYPE=Release` by g++ 16.2 on Windows 11. The code at fault is unchanged on `main` at `fd824c9` (2026-09-29).

## Summary

An unmoved White pawn or brawn can double-step along the timeline axis, two timelines forward, but the move generator never offers that step to a Black one. The Black branch of `multiverse::gen_moves_impl` guards its copy of the double-step block with the White piece types, so the block is never compiled in for Black. Both colours do get the same-board double step and the one-timeline step.

## Reproduction

The two records below are hand-built and play no actions. Here is `black-pawn.5dpgn`:

```
[Board "Custom"]
[Size "5x5"]
[4k/2p*2/5/5/K4:-1:1:b]
[4k/5/5/5/K4:0:1:b]
[3k1/5/5/5/K4:0:2:w]
[4k/5/5/5/K4:1:1:b]
[3k1/5/5/5/K4:1:2:w]
```

The position has three timelines:

- Black is to move only on (-1T1). Timelines 0 and +1 have already moved on to T2, so (0T1) and (+1T1) are past boards.
- The black pawn on (-1T1)c4 is unmoved.
- c4 is empty on both (0T1) and (+1T1).

So the pawn may step one timeline forward to (0T1)c4, or two to (+1T1)c4. Both are branching moves.

`white-pawn.5dpgn` is the exact mirror image, with colours swapped and ranks and timelines reflected:

```
[Board "Custom"]
[Size "5x5"]
[k4/5/5/2P*2/4K:1:1:w]
[k4/5/5/5/4K:0:1:w]
[k4/5/5/5/3K1:0:1:b]
[k4/5/5/5/4K:-1:1:w]
[k4/5/5/5/3K1:-1:1:b]
```

Steps: build `5dtools` as described in the README, save the two records, and run:

```sh
mkdir build && cd build && cmake .. -DTOOLS=on -DCMAKE_BUILD_TYPE=Release && cmake --build .
./5dtools all --at 0 < black-pawn.5dpgn
./5dtools all --at 0 < white-pawn.5dpgn
```

**Expected:** 10 actions for each colour, mirror images of each other. Black's list should include `(-1T1)c4>>(1T1)c4`.

**Actual:** Black gets 9 actions. The two-timeline step is missing, although the pawn's one-timeline step and its same-board double step `c4c2` are both listed:

```
(-1T1)Ke5>>(0T1)d5
(-1T1)Ke5>>(0T1)e4
(-1T1)Ke5>>(0T1)d4
(-1T1)c4>>(0T1)c4
(-1T1)Ke5d5
(-1T1)Ke5e4
(-1T1)Ke5d4
(-1T1)c4c3
(-1T1)c4c2
Summary: totally 9 options
```

White gets all 10, including `(1T1)c2>>(-1T1)c2`:

```
(1T1)c2>>(-1T1)c2
(1T1)c2>>(0T1)c2
(1T1)Ke1>>(0T1)e2
(1T1)Ke1>>(0T1)d2
(1T1)Ke1>>(0T1)d1
(1T1)c2c4
(1T1)c2c3
(1T1)Ke1e2
(1T1)Ke1d2
(1T1)Ke1d1
Summary: totally 10 options
```

`5dtools count --at 0` gives 9 and 10 with both the `balanced` and `naive` policies. If `p*` is replaced with `w*` (a brawn) in `black-pawn.5dpgn`, Black again gets 9 actions, and `(-1T1)Wc4>>(1T1)c4` is missing.

**In a public game record.** The same thing shows up in `test/1.2.5dpgn` of adri326/5dchess-notation at `99f6163a9c95a6a9740b20fb5f3c7f6503a166a0` (https://github.com/adri326/5dchess-notation/blob/99f6163a9c95a6a9740b20fb5f3c7f6503a166a0/test/1.2.5dpgn). Keep its headers and its first 15 actions - everything up to and including White's eighth move, `8. (-1T3)Kg3?` - and drop the rest; then:

- `./5dtools count --at 8w` prints `Summary: totally 52 options`, where 54 is expected.
- The two missing actions are `(-1T3)e7>>(1T3)e7` and `(-1T3)g7>>(1T3)g7`. They are Black's unmoved e- and g-pawns on (-1T3) stepping two timelines forward, through the empty e7 and g7 of (0T3), onto the empty e7 and g7 of the past board (+1T3), which branches.
- `5dtools all --at 8w` does list the same pawns' one-timeline steps `(-1T3)e7>>(0T3)e7` and `(-1T3)g7>>(0T3)g7`, and their same-board double steps `(-1T3)e7e5` and `(-1T3)g7g5`.

## Root cause

In `multiverse::gen_moves_impl` (`src/core/multiverse_base.cpp`), the White pawn/brawn branch (line 838) yields the two-timeline step inside this block:

```cpp
                if constexpr(P == PAWN_UW || P == BRAWN_UW)   // line 865
```

The Black branch (`P == PAWN_B || P == BRAWN_B || P == PAWN_UB || P == BRAWN_UB`, line 909) mirrors it, but its copy of the block keeps the White types:

```cpp
                if constexpr(P == PAWN_UW || P == BRAWN_UW)   // line 937
```

That condition can never hold inside the Black branch, so the block is compiled out for `PAWN_UB` and `BRAWN_UB`. The same-board double step is unaffected, because `gen_physical_moves_impl` guards its Black branch with the Black types (line 640). Permalink: https://github.com/ftxi/5dchess_engine/blob/2ea9f1b82880f08faa9400d1ac20b20102604187/src/core/multiverse_base.cpp#L937

**Suggested fix:** guard line 937 with `P == PAWN_UB || P == BRAWN_UB`.

**Related code we have not tested.** The two brawn blocks of the Black branch (lines 952 and 962) also test `BRAWN_W || BRAWN_UW`, so they are dead code for Black brawns too. `src/core/check_position.cpp` lines 206-207 deliberately mirror this ("additional L-direction brawn captures are enabled only in its white pawn/brawn branch"), so those places would need to change together.

**What it affects.** The missing step is not a capture, so check detection is unaffected. It does change Black's legal-action lists and counts, and with them the search. It can also change a checkmate or stalemate verdict wherever this step is Black's only legal turn.

## The rule

- Wikipedia, *5D Chess with Multiverse Time Travel*, Rules: "On the pawn's first move, it may move two spaces forward through two vacant squares along one axis." and "The brawn has the movement and promotion abilities of the pawn."
- In-game Guide, Pieces tab, pawn: "The pawn moves one or two squares in the "forward" direction. Here, "forward" includes the direction towards your opponent's back timeline."

Neither rule depends on colour. For Black, forward along the timeline axis is towards White's timelines (increasing L), which is the direction 5dtools already uses for Black's one-timeline step. Whatever reading is taken of the double step, the two colours should agree; at present White has it and Black does not.

## How this was found

This was found by differential testing. We compare legal-action counts position by position across independent rules implementations, on public game records. At the record position above, this engine counted 52 actions where two other implementations (penteract/cwmtt and 5d-chess-js) list 54. The two extra actions are exactly these Black two-timeline steps.
