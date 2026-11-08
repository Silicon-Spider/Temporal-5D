# A created-timeline token written with its sign, `(>L+1)`, is not removed before the moves are read: the import fails, or by default holds no action

- **Project:** 5d-chess-js, https://gitlab.com/5d-chess/5d-chess-js
- **Tested at:** commit `ee403c80e8c132fc4d168f5b99ddd9b8fe5cc17b` (2022-06-15, the head of master when we took it; package.json version 1.2.1), with Node 24.20.0 on Windows 11; run 2026-11-08.

## Summary

Your documentation says the library reads 5dpgn "as described by commit `92c68f0`" of adri326/5dchess-notation (`docs/README.md`, "PGN"). In that text:

- a move may be followed by the token `(>Lx)`, "with `x` being replaced by the new timeline's index";
- a timeline index other than 0 may be written with its `+` sign or without it;
- its own example game writes `(>L+3)`.

The move-text reader removes the token with the pattern `/\(>L\-?\d*\)/g`, which matches `(>L1)` and `(>L-1)` but not `(>L+1)`. The token is then read as a move of its own and fails, and the whole import fails with it. `import(text, variant, true)` throws, and `new Chess(text)` silently holds no action at all.

## Reproduction

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

// The game of the library's own test 'Simple moves' (src/tests/pgn.test.js),
// ended with the token the library's export writes, and with that token signed.
const game = `[Board "Standard"]

1. a4 / d6
2. (0T2)Nb1>>(0T1)b3~ `;

for (const token of ['(>L1)', '(>L+1)']) {
  const chess = new Chess();
  try {
    chess.import(game + token, undefined, true);
    console.log(`${token}: import() read ${chess.actionHistory.length} actions`);
  } catch (e) {
    console.log(`${token}: import() throws: ${e.message.split('\n')[0]} ...`);
  }
  console.log(`  new Chess(text) holds ${new Chess(game + token).actionHistory.length} actions`);
}

const chess = new Chess();
try {
  chess.raw.pgnFuncs.toActionHistory('1. a4 / d6\n2. (0T2)Nb1>>(0T1)b3~ (>L+1)', chess.rawBoard, 0);
} catch (e) {
  console.log(`toActionHistory() throws: ${e.message}`);
}
```

**Expected:** three actions both ways. `(>L+1)` names the same timeline as `(>L1)`: timeline 1, which the knight's jump creates.

**Actual:**

```
(>L1): import() read 3 actions
  new Chess(text) holds 3 actions
(>L+1): import() throws: Notation invalid and an error has occurred with string: 1. a4 / d6 ...
  new Chess(text) holds 0 actions
toActionHistory() throws: Source super-physical coordinates missing or incorrect!
```

**In a public game record.** penteract/cwmtt `tests/5dpgn/tesseractMageOChicken.5dpgn` at `9632e9969dd455c6e3d6c27c57c1936c251d7e97` (https://github.com/penteract/cwmtt/blob/9632e9969dd455c6e3d6c27c57c1936c251d7e97/tests/5dpgn/tesseractMageOChicken.5dpgn) is a 2021 league game in Standard - Turn Zero, 33 actions. White's 13th turn ends `(>L+1)` (line 22), and the last move `(>L+2)` (line 26).

- `new Chess(text)` holds no action.
- `import(text, undefined, true)` throws `Notation invalid and an error has occurred with string: 1. Nf3 / Nf6 ...`.
- `toActionHistory` throws `Source super-physical coordinates missing or incorrect!`.

The record's signed board coordinates, such as `(+1T10)`, are not the problem: with the fix below applied, the library imports all 33 actions.

## Root cause

`toAction` and `toMove` (`src/pgn.js`) remove the token with the same pattern:

```js
  tmpStr = tmpStr.replace(/\(>L\-?\d*\)/g, '');   // line 528, in toAction; line 256, in toMove, is the same
```

The pattern allows a minus sign but not a plus sign, so `(>L+1)` is left in place. Then:

1. `toAction` splits the action at spaces (lines 529-531), so the leftover `(>L+1)` becomes a move of its own.
2. `toMove` takes it for a jump, since it contains `>` (line 259).
3. It finds no source board at its start (line 266) and throws `Source super-physical coordinates missing or incorrect!` (line 305).
4. `convertFuncs.actions` turns that into `Notation invalid and an error has occurred with string: ...` (`src/convert.js`, line 67).
5. `import()` rethrows it only when `actionsRequired` is true (`src/index.js`, lines 228-238). Otherwise it drops the error before any action is played, so `new Chess(text)` holds the starting position.

Board coordinates are read with `/^\(L?\-?\+?\d+T\-?\+?\d+\)/` (lines 266, 327 and 367), which does allow the `+`. Permalink: https://gitlab.com/5d-chess/5d-chess-js/-/blob/ee403c80e8c132fc4d168f5b99ddd9b8fe5cc17b/src/pgn.js#L528

## Suggested fix

Allow the sign in both patterns: `/\(>L[+-]?\d*\)/g` at lines 256 and 528.

We applied that change to a copy of the pinned code:

- the script prints `(>L+1): import() read 3 actions` and `new Chess(text) holds 3 actions`, and `toActionHistory()` no longer throws;
- the public record imports all 33 actions.

## The notation

adri326/5dchess-notation at `92c68f0c19183b4ff4459b4a32107b0bf6e069a2`, the version your documentation names (`docs/README.md`, lines 92 and 96), `README.md`:

- "Complex scenarios" (lines 165-174): "To aid this, one may put the following token after a move: `(>Lx)`, with `x` being replaced by the new timeline's index." ... "Both present movement and created timeline tokens are purely for better human comprehension and should match with what can be induced from the described state of the game (if the piece of notation needs to be parsed)."
- "Notes", "Even-numbered starting boards" (lines 503-504): "If `-0` and `+0` are used, the `+` sign is mandatory (just like how it's always displayed in such cases in the game). The `+` sign can be omitted for other integers."
- Its example game, Shad Amethyst against PseudoAbstractMeta (line 302): `21. (2T13)Qg4>>x(2T10)g7~! (>L+3) / (3T10)Bd6>>x(1T10)d4~ (>L-3)`.

The current text, at `99f6163a9c95a6a9740b20fb5f3c7f6503a166a0`, keeps all three (lines 187-194, 548-549 and 348).

So `(>L+1)` names timeline 1, as `(>L1)` does, and a reader that drops the one should drop the other.

## How this was found

This was found by differential testing. We replay public game records through several independent implementations and compare them position by position. The library reads none of the record above; read one action at a time, it stops at White's 13th turn, where penteract/cwmtt and the parser bundled with adri326/5dchess-notation both read all 33 actions. Chess in 5D (chessin5d.net) depends on this library and is probably affected too; we have not tested it there.
