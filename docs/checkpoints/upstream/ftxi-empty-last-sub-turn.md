# Question: may a game's last sub-turn be empty - `23. (-4T10)Bxd2 / 1-0`, or a record that ends in `/`? Shad's notation ends its example game that way, and the reader refuses it

- **Project:** ftxi/5dchess_engine, https://github.com/ftxi/5dchess_engine
- **Tested at:** commit `2ea9f1b82880f08faa9400d1ac20b20102604187` (tag v0.4.10), with the stock `5dtools` built with `-DTOOLS=ON -DCMAKE_BUILD_TYPE=Release` by g++ 16.2 on Windows 11; run 2026-11-08. `parse_actions`, `parse_gametree` and `parse_game` are unchanged on `main` at `d765d3d` (2026-10-15), the newest commit we have.
- **This is a question, not a bug report.** Your grammar requires a move after every turn serial, `/` included, and the reader does what the grammar says. Shad's notation, which Branched 5DPGN aims to stay compatible with, ends its own example game with a `/` that no move follows. We read Shad's notation as allowing that. If the restriction is deliberate, we would like to know why.

## Summary

A record whose last sub-turn is empty - a `/` followed by nothing, or only by a result - is refused as a whole, and no action is played. Here is `slash-result.5dpgn`:

```
[Mode "5D"]
[Board "Standard"]

1. e4 / e5
2. Nf3 / 1-0
```

```
$ ./5dtools stats < slash-result.5dpgn
Parse Error: parse_game(): Unexpected trailing input:
[Board "Standard"]

1. e4 / e5
2. Nf3 /
```

With the record ending in `2. Nf3 /` instead, the message is the same. With the record ending in `2. Nf3 1-0` or `2. Nf3`, it prints `actions played: 3`.

## Your grammar and the reader

- `docs/pgn-bnf.txt` (lines 8 and 10-11) and `docs/branched-5dpgn.md` (lines 407 and 409-410) define `<action> ::= <turn-serial> <space-comment>* <move> {<space-comment>+ <move>}* <space-comment>*`, with `<turn-serial> ::= ... | '/'`. So a move must follow `/`. A game tree may end in a result, but only where a subtree would begin, not after a `/`.
- `docs/branched-5dpgn.md` says the same in prose (line 197): "One player's sub-turn consists of an action to be submitted, which has one or more moves inside."
- The same document lists "compatibility with Shad's Notation" among Branched 5DPGN's goals (line 19).
- In the reader, `pgnparser::parse_actions` (`src/core/pgnparser.cpp`, lines 778-808) gives up when no move follows a turn token (line 792). `parse_gametree` then ends the tree before the `/`, and `parse_game` refuses what is left (lines 927-929).

## Shad's notation

adri326/5dchess-notation, `README.md` at `99f6163a9c95a6a9740b20fb5f3c7f6503a166a0`:

- "Turns" (lines 198-206): "Sub-turns are separated by a forward slash (`/`)." A turn is "`<x>.` or `<x>w.`", "White's moves", "`/` or `<x>b.`", "Black's moves". It does not say whether a sub-turn may hold no move.
- The example game between Shad Amethyst and PseudoAbstractMeta ends with an empty sub-turn (line 350): `23. (-4T10)Bxd2 / 1-0 {White wins by timeout; possible continuation would have been (-4T10)d4; 24. (-4T10)Qxd4#}`. https://github.com/adri326/5dchess-notation/blob/99f6163a9c95a6a9740b20fb5f3c7f6503a166a0/README.md?plain=1#L350
- A result is not required at the end of the move text: the puzzles "Rook Tactics I" (line 259) and "Knight tactics III" (line 271) end without one. A record that ends in `/` is the example's ending without its result.

## Our question

Is the move after `/` required on purpose? Should a reader take a last sub-turn with no moves - followed by a result, a comment or the end of the text - as a sub-turn that was not played, as Shad's example game uses it? If you have a reason to refuse it, for instance that a `/` with nothing after it is meant to be invalid, we would like to know it.

An aside: the "trailing input" in the message is not the input that could not be read. `parse_game` quotes `PARSED_MSG` (line 929; defined at line 339), which runs from where `parse_game` started - after the first token, which the constructor has already read (here the first header) - up to and including the current token, the `/`. The `1-0` that follows is not shown.

## How this was found

This was found by differential testing on public game records. penteract/cwmtt's `tests/brawnsPuzzles/another.5dpgn` at `9632e9969dd455c6e3d6c27c57c1936c251d7e97` (https://github.com/penteract/cwmtt/blob/9632e9969dd455c6e3d6c27c57c1936c251d7e97/tests/brawnsPuzzles/another.5dpgn) ends `4. We3 /` (line 7). `5dtools` refuses it with this message; penteract/cwmtt reads its seven actions.

That record's variant, Focused - Just Brawns, is not among the variants `5dtools` knows at this commit. So with the `/` removed, it stops at `derive_variant_setup(): Unknown variant: Focused - Just Brawns` instead. We give it only as the record where we met this.
