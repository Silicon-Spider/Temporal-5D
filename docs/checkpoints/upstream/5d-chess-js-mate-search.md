# inCheckmate is true in positions that have a legal turn: the exhaustive search in mateFuncs.checkmate never expands a node

- **Project:** 5d-chess-js, https://gitlab.com/5d-chess/5d-chess-js
- **Tested at:** commit `ee403c80e8c132fc4d168f5b99ddd9b8fe5cc17b` (2022-06-15, still the head of master; package.json version 1.2.1), with Node 24.19.0 on Windows 11.

## Summary

Sometimes a player in check can escape only with a turn whose first move removes no check: for example, a move on one board that turns it into a past board, so that a later jump onto it branches. In that case the library reports checkmate even though a legal turn exists: `inCheckmate` is `true`, `moves()` is empty, and `submit()` refuses the legal turn. The cause is in the "Slow BFS exhaustive search" of `mateFuncs.checkmate` (`src/mate.js`). It passes its search-tree node to `boardFuncs.moves()` instead of the node's board, so it never generates a move.

## Reproduction

This is a custom 5x5 position with two timelines and White to move on both. It plays no actions:

```
[Board "Custom"]
[Size "5x5"]
[5/5/1n3/RB2k/KB3:0:1:w]
[4k/5/5/PP3/K4:+1:1:w]
```

```
      L0, T1 (White to move)    L+1, T1 (White to move)
  5   . . . . .                 . . . . k
  4   . . . . .                 . . . . .
  3   . n . . .                 . . . . .
  2   R B . . k                 P P . . .
  1   K B . . .                 K . . . .
      a b c d e                 a b c d e
```

- **The check.** On L0 the black knight on b3 checks the white king on a1. The king's own rook and bishops box it in, and no white piece can take the knight.
- **The only king move fails on its own.** The king's one move is the jump to b1 on L+1. Played alone, it is unsafe: the knight reaches (+1T1)b1, one timeline and two ranks away.
- **The escape.** If White first moves a pawn on L+1, (+1T1) becomes a past board. The king's jump onto it then branches into a new timeline (+2), two timelines from the knight. Both of White's boards have moved, no white king is attacked, and the present passes to Black.
- **The legal turns.** `(1T1)a3` followed by `(0T1)Ka1>>(1T1)b1` is a legal turn. So is `(1T1)b3` followed by the same jump.

Steps:

```sh
git clone https://gitlab.com/5d-chess/5d-chess-js.git && cd 5d-chess-js
git checkout ee403c80e8c132fc4d168f5b99ddd9b8fe5cc17b
npm ci --omit=dev --ignore-scripts
node repro.js        # repro.js below, saved in the checkout's root
```

```js
// repro.js - run from the root of a 5d-chess-js checkout
const Chess = require('./src/index.js');

const pgn = `[Board "Custom"]
[Size "5x5"]
[5/5/1n3/RB2k/KB3:0:1:w]
[4k/5/5/PP3/K4:+1:1:w]
`;

const chess = new Chess(pgn);
console.log('inCheck:', chess.inCheck);
console.log('inCheckmate:', chess.inCheckmate);
console.log('moves():', chess.moves().length);

// A legal turn: a pawn move on +1 first, then the king's jump onto (+1T1)b1, which now branches.
chess.move('(1T1)a3');
chess.move('(0T1)Ka1>>(1T1)b1');
console.log('after the two moves, inCheck:', chess.inCheck);
console.log('boards still to move:', chess.raw.boardFuncs.present(chess.rawBoard, chess.rawAction).length);
try { chess.submit(); console.log('submit(): accepted'); }
catch (e) { console.log('submit():', e.message); }
```

**Expected:** `inCheckmate: false`, because White is in check but has a legal turn. `moves()` should be non-empty, and `submit()` should accept the turn.

**Actual:**

```
inCheck: true
inCheckmate: true
moves(): 0
after the two moves, inCheck: false
boards still to move: 0
submit(): Cannot submit, currently in checkmate.
```

The library's own `move()` validation accepts both moves, and its own check test finds every white king safe afterwards. Only the mate search blocks the turn.

The same happens at the end of two public game records, imported in full with `new Chess(text)`. In each, White is to move, `inCheckmate` is `true` and `moves()` is empty. Each legal turn shown below passes `move()`, leaves `inCheck` false and the present moved, and `submit()` still refuses it:

- `test/pgn/number1.5dpgn` of ftxi/5dchess_engine. All 4 actions, custom 5x5 variant. Legal turn: `(-1T3)Ka5>>(-1T2)a5 (0T3)Ka4>(1T3)a4 (2T3)Ke1>>(1T3)e1`. https://github.com/ftxi/5dchess_engine/blob/2ea9f1b82880f08faa9400d1ac20b20102604187/test/pgn/number1.5dpgn
- `tests/5dpgn/chessin5d-illegal-move-example.5dpgn` of penteract/cwmtt. All 66 actions, Standard variant. Legal turn: `(1T33)Kf6e5 (0T33)Kb1>>(1T33)a1`. https://github.com/penteract/cwmtt/blob/9632e9969dd455c6e3d6c27c57c1936c251d7e97/tests/5dpgn/chessin5d-illegal-move-example.5dpgn

## Root cause

`mateFuncs.checkmate` searches in three passes:

1. It tries every single move (`src/mate.js` lines 62-75).
2. It runs a depth-first search that follows only moves which reduce the number of checks (lines 77-106).
3. If neither finds an escape, it falls through to the "Slow BFS exhaustive search" (lines 149-191). Its queue holds nodes `{ board, checkSig }` (lines 144-147), but each node is passed on whole, not its board:

```js
      const moves = boardFuncs.moves(currNode, actionNum, false, false);   // line 157
        const tmpBoard = boardFuncs.copy(currNode);                       // line 161
```

With `activeOnly` and `presentOnly` false, `boardFuncs.moves()` loops `for (let l = 0; l < fullBoard.length; l++)` (`src/board.js` line 370). A node has no `length`, so the call returns no moves. The root node expands to nothing, the queue empties, and `checkmate` returns `[true, false]`: checkmate, not a time-out, a few milliseconds in.

As a result, the exhaustive search never runs. Every checkmate the library reports means only that no single move, and no chain of check-reducing moves, escapes. Permalink: https://gitlab.com/5d-chess/5d-chess-js/-/blob/ee403c80e8c132fc4d168f5b99ddd9b8fe5cc17b/src/mate.js#L157

## Suggested fix

Pass the node's board at both places: `boardFuncs.moves(currNode.board, ...)` and `boardFuncs.copy(currNode.board)`.

With that change:

- the script above prints `inCheckmate: false`, `moves(): 19` and `submit(): accepted`;
- both public records come out not checkmate, with `moves()` at 83 and 56, and their legal turns are accepted.

There is a knock-on effect: once the search really runs, it can use up `checkmateTimeout` on large positions. The `inCheckmate` getter returns `res[0]`, which is also `true` for a time-out (`[true, true]`, `src/index.js` lines 617-621). So a time-out would read as checkmate, and you may want to report it as "unknown" instead.

## The rule

A position is checkmate only if the player in check has no legal turn at all:

- In-game Guide, Checkmate tab, "1) What is checkmate?": "If a player has no way to move The Present without leaving their king attacked, then they are checkmated (or stalemated). ... Here, it means they cannot make any sequence of moves that shifts The Present without leaving one of their kings attacked."
- Wikipedia, *5D Chess with Multiverse Time Travel*, Rules: "If the player whose turn it is has no way to legally complete their turn, then the game ends in one of two ways:" ... "If the player is in check, then they are in checkmate, and they lose."

In the position above, White can shift the present without leaving a king attacked, so it is not checkmate.

## How this was found

This was found by differential testing. We replay public game records through several independent implementations of the rules and compare their verdicts position by position. The two record positions above are the only ones where this library reported checkmate and the other implementations listed legal turns. Every other checkmate it reported in those records was confirmed by at least one of the others. Chess in 5D (chessin5d.net) depends on this library and is probably affected too; we have not tested it there.
