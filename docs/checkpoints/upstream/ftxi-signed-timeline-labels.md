# `(+1T10)` in an odd-timeline game and `(1T1)` in an even one match no move: `match_absolute_board` compares a timeline's optional `+` sign as written

- **Project:** ftxi/5dchess_engine, https://github.com/ftxi/5dchess_engine
- **Tested at:** commit `2ea9f1b82880f08faa9400d1ac20b20102604187` (tag v0.4.10), with the stock `5dtools` built with `-DTOOLS=ON -DCMAKE_BUILD_TYPE=Release` by g++ 16.2 on Windows 11; run 2026-11-08. On `main` at `d765d3d` (2026-10-15), the newest commit we have, the code at fault is unchanged (read there, not run; see "Root cause").

## Summary

Shad's notation lets a timeline other than 0 be written with or without its `+` sign; only `-0` and `+0` must carry one. Your grammar says the same (`<line> ::= ['+' | '-'] <natural-number>`), and the parser reads the sign. But a move is resolved by comparing it with your own rendering of every candidate move, and `match_absolute_board` compares the sign exactly as written. The rendering writes `1` in an odd-timeline game and `+1` in an even one. So a move on or to `(+1T..)` in an odd game - or `(1T..)` in an even game - matches no candidate, and the record stops with `Invalid move`.

## Reproduction

Four hand-built records, each playing one action. Each pair differs only in how timeline 1 is written in the move text.

`odd-signed.5dpgn` - timelines 0 and 1, White to move on both:

```
[Board "custom"]
[Size "5x5"]
[4k/5/5/5/K4:0:1:w]
[4k/5/5/5/K4:1:1:w]

1. (0T1)Ka2 (+1T1)Ka2
```

`odd-unsigned.5dpgn` is the same with `(1T1)Ka2` as its last move.

`even-unsigned.5dpgn` - timelines -0, +0 and 1, White to move on all three:

```
[Board "custom"]
[Size "5x5"]
[4k/5/5/5/K4:-0:1:w]
[4k/5/5/5/K4:+0:1:w]
[4k/5/5/5/K4:+1:1:w]

1. (-0T1)Ka2 (+0T1)Ka2 (1T1)Ka2
```

`even-signed.5dpgn` is the same with `(+1T1)Ka2` as its last move.

Steps: build `5dtools` as described in the README, save the four records, and run:

```sh
for f in odd-signed odd-unsigned even-signed even-unsigned; do echo "$f:"; ./5dtools stats < $f.5dpgn 2>&1 | head -1; done
```

**Expected:** `actions played: 1` for all four. In each pair the two moves are one move, written two ways.

**Actual:**

```
odd-signed:
Runtime error: state(): Invalid move: mv:pm{b:ab{s:+,l:1,t:1},pn:K,ff:?,fr:?,cp:no,tf:a,tr:2,pt:?}
odd-unsigned:
actions played: 1
even-signed:
actions played: 1
even-unsigned:
Runtime error: state(): Invalid move: mv:pm{b:ab{s:0,l:1,t:1},pn:K,ff:?,fr:?,cp:no,tf:a,tr:2,pt:?}
```

**In a public game record.** penteract/cwmtt `tests/5dpgn/tesseractMageOChicken.5dpgn` at `9632e9969dd455c6e3d6c27c57c1936c251d7e97` (https://github.com/penteract/cwmtt/blob/9632e9969dd455c6e3d6c27c57c1936c251d7e97/tests/5dpgn/tesseractMageOChicken.5dpgn) is a 2021 league game in Standard - Turn Zero, 33 actions. It writes timeline 1 as `+1` from turn 14 on (lines 23-26).

- `./5dtools stats < tesseractMageOChicken.5dpgn` stops at action 28, Black's `(-1T10)Nf6>x(+1T10)g6` (line 23):

  ```
  Runtime error: state(): Invalid move: mv:sm{fb:ab{s:-,l:1,t:10},pn:N,ff:f,fr:6,ji:>,cp:yes,tb:ab{s:+,l:1,t:10},tf:g,tr:6,pt:?}
  ```

- With the sign taken out of the board coordinates only - `sed 's/(+\([0-9]\)T/(\1T/g'`, which leaves the record's `(>L+1)` and `(>L+2)` tokens alone (they are read) - `5dtools stats` prints `actions played: 33`.
- On that copy, `./5dtools count --at <turn> balanced 1000000` at `14b`, `15w`, `15b`, `16w`, `16b` and `17w` gives 179681, 34, 128714, 126, 74692 and 0 options. These are the numbers the record lists for those positions in its closing comment, and `5dtools checkmate --at 17w` prints `Checkmate`, as the record's last move `#` says.

## Root cause

`state::parse_move` (`src/core/state.cpp`) renders each candidate move with `full_move::pgn_impl` and parses the text back (lines 1042-1045 for physical moves, 1107-1111 for jumps). Then `match_absolute_board` (`src/core/pgnparser.cpp`, lines 1006-1011) compares the board as written with the candidate's:

```cpp
    return (!simple.line || std::pair{simple.sign, simple.line} == std::pair{full.sign, full.line})   // line 1009
```

`sign` is the token that was read: `POSITIVE`, `NEGATIVE`, or `NIL` when none is written (`parse_absolute_board`, lines 440 and 450-453). So `+1` and `1` are different boards to the matcher.

The candidate's sign comes from the rendering in `src/core/multiverse_variants.h`:

- `multiverse_odd::pretty_lt` writes `std::to_string(p0.l())`, with no sign for a positive line (line 28);
- `multiverse_even::pretty_lt` always writes one (lines 54-56).

So in an odd game only the unsigned form matches, and in an even game only the signed one. Permalink: https://github.com/ftxi/5dchess_engine/blob/2ea9f1b82880f08faa9400d1ac20b20102604187/src/core/pgnparser.cpp#L1006-L1011

Your 5DFEN reader already reads the two forms alike. In an odd game, `create_multiverse_from_variant_setup` takes a missing sign as `+` (`src/core/variants.cpp`, lines 182-186), so `[...:1:1:w]` and `[...:+1:1:w]` set up the same board.

On `main` at `d765d3d`:

- the rendering has moved to `timeline_layout::pretty_l` (`src/core/timeline_layout.h`), with the same output: no sign in odd games, always one in even games;
- `parse_move` still renders each candidate and parses it back;
- `match_absolute_board` is unchanged.

**Suggested fix:** in `match_absolute_board`, read a missing sign as `+` on every line except 0, on both sides, before comparing. Then `1` and `+1` are one timeline, while `-0`, `0` and `+0` stay three.

**What it affects.** Any record that writes a positive timeline with its sign in an odd game, or without it in an even game. Both of the library's readers resolve moves through `state::parse_move`, so both stop at such a move:

- the `state` constructor that `5dtools` uses (`src/core/state.cpp` line 46);
- `game::from_pgn` (`src/client/game.cpp` line 65).

## The notation

- Shad's notation (adri326/5dchess-notation, `README.md` at `99f6163a9c95a6a9740b20fb5f3c7f6503a166a0`), "Notes", "Even-numbered starting boards" (lines 548-549): "If `-0` and `+0` are used, the `+` sign is mandatory (just like how it's always displayed in such cases in the game). The `+` sign can be omitted for other integers." https://github.com/adri326/5dchess-notation/blob/99f6163a9c95a6a9740b20fb5f3c7f6503a166a0/README.md?plain=1#L548-L549
- The same README, "5DFEN and custom variants" (line 405), on a board string's timeline field: "may be `-1`, `-0`, `0`, `+0`, `+1`, etc."
- Your own grammar, `docs/pgn-bnf.txt` (line 32) and `docs/branched-5dpgn.md` (line 428): `<line> ::= ['+' | '-'] <natural-number>`. And `docs/branched-5dpgn.md` line 70: "If the game has odd timelines, `<L>` is an integer. If the game has even timelines, then `<L>` is one of: a non-zero integer, +0, or -0."

So `(+1T10)` in an odd game and `(1T1)` in an even game are valid. They name the same boards as `(1T10)` and `(+1T1)`.

## How this was found

This was found by differential testing. We replay public game records through several independent implementations and compare them position by position. On the record above, this engine stopped at action 28, where penteract/cwmtt and the parser bundled with adri326/5dchess-notation both read all 33 actions.
