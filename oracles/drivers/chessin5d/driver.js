// Test-time driver for the pinned "Chess in 5D" rules library (5d-chess-js).
//
// This file belongs to the oracle harness, never to the engine. It loads the
// oracle from an export of the locked commit and reports, position by position,
// what the oracle itself says. Every verdict comes from the library's own
// functions. The driver does five things around them, none of which touches
// rules logic:
//
//   * it splits the library's import() into "set up" and "apply one action",
//     because import() offers no way to look at intermediate positions and its
//     checkmate search answers `true` when it runs out of time;
//   * it refuses records the library would replay as a different game. After
//     the library's own set-up it asks which variant the library chose and
//     compares that with the variant the record declares (the library falls
//     back to Standard, without saying so, for a name it does not know, for a
//     header its own reader does not match, and for a Variant header, which it
//     does not read); it refuses 5DFEN boards under any
//     variant but "custom" (the library ignores them), a custom position the
//     library set up only in part (it reads a board string only when the string
//     has a line to itself), and one whose boards the library names at other
//     coordinates than the record's (it guesses whether a game has a turn zero
//     from the boards it holds, and can move every board to another turn);
//   * it drops a PGN result token ("1-0", "1/2-1/2") from the end of the move
//     text, because the library reads it as moves and fails on it;
//   * it runs the library's mate search in a worker thread with its own heap
//     limit. On wide positions that search can exhaust memory long before its
//     time budget; in a worker that costs one unknown verdict, not the record.
//     (Any other failure of the search is an error of the run, not a silence.)
//   * it prints, at every position, the position the library holds (the state
//     event, decision H30), and on request two diagnostics (below).
//
// The adapter (oracle_harness/adapters/chessin5d.py) refuses, before this runs,
// two notations the library's move-text reader gets wrong without noticing:
// "12w." turn markers and parenthesised variations.
//
// usage : node driver.js <exported-root> [--board NAME] [--mate-budget-ms N] [--mate-heap-mb N]
//                        [--explain] [--actions K] [--actions-budget-ms N]
//         --board is the variant the record declares, as the caller reads it:
//         its Board header, else its Variant header, else "Standard". Without
//         it the driver reads the headers itself, leniently.
// stdin : one 5DPGN record.
// stdout: JSON Lines, flushed per line.
//   {"event":"start", ...}
//   then, for each position k:
//   {"event":"state","index":k,"size":"WxH","markers":true,
//    "boards":["<L>:<T>:<c>:<placement>", ...],"hash":"<md5>"}
//   {"event":"explain","index":k, ...}                          (--explain only)
//   {"event":"position","index":k,"side":"white|black","in_check":bool,
//    "legal_actions":null,"terminal":"none|checkmate|stalemate"|null,"ms":float
//    [,"limit":"mate_budget_ms|deadline|mate_heap_mb"]}
//   {"event":"actions","index":k,"count":n,"truncated":bool, ...,"actions":[...]}
//                                                               (--actions k only)
//   and last:
//   {"event":"end","status":"ok|rejected|unsupported|error", ...}
//
// legal_actions is always null: the library's actions() enumerates ordered move
// sequences under its own options, which is not the quantity the other oracles
// count, and as published it cannot list them all (see "actions" below) - a
// count assembled around it would be the driver's, not the library's. (The
// actions diagnostic does assemble one, for a person to read; it is never
// compared.) terminal is null when the library's mate search did not finish, and
// "limit" then names what stopped it: the library's own time-out flag
// ("mate_budget_ms"), the driver's hard deadline past that budget ("deadline"),
// or the worker's heap limit ("mate_heap_mb"). The harness does not compare it.
//
// The state event is the library's own export of every board it holds -
// superseded and head boards, every timeline, both colours: chess.fen(null,
// true), the string the library's .hash digests (the notation specification's
// full-state hash), one "[placement:L:T:c]" per board, sent as
// "L:T:c:placement". Its labels and its unmoved markers ("*" after a pawn,
// rook, king or brawn the library holds unmoved, and after nothing else) are
// the library's; the driver never turns an index of the library's arrays into a
// label. It is printed after the action is submitted and before the position's
// check test and mate search, so a search cut short cannot lose it, and only
// after the number of boards and pieces it lists has been held against the
// library's arrays: a board or a piece the export dropped would change the
// fingerprint without a trace. The size is that of the boards the library set up.
// The harness, not this driver, canonicalises and hashes it - and holds what it
// computes to "hash", the library's own MD5 of the same export (chess.hash), so
// that the harness's canonical form is checked at every position against an
// implementation of the notation's hashing it did not write.
//
// The explain and actions events are diagnostics, printed on request and never
// compared:
//   explain  what the library's check test (mateFuncs.checks, behind
//            chess.inCheck) looks at: "phantom_timelines", the timelines that
//            mateFuncs.blankAction() advanced by a copy of the mover's last board
//            before the test looks for a royal capture, and "phantom_boards", the
//            boards it added; "present_timelines" and "active_timelines", the
//            library's own lists (boardFuncs.present / .active) that
//            blankAction() works from; and, in check, "checks": every royal
//            capture the test finds, each as
//            {"from":{"L","T","c","square","piece"},"to":{...}} - the boards
//            named as in the state event, a phantom board included, and the
//            piece as its 5DFEN letter (upper case white).
//   actions  the legal actions of the position as the library enumerates them:
//            actionFuncs.actions(), the enumerator behind chess.actions(),
//            asked for moves from every timeline, not only the active or present
//            ones, and for actions that may create inactive timelines. Each
//            sequence it lists is kept when the library's move() and submit()
//            would take it: every move in the library's own move list for the
//            board it is played on (validateFuncs.move, behind moveable()), and
//            then no royal capturable (mateFuncs.checks, behind inCheck) and no
//            present timeline left unplayed (boardFuncs.present, behind
//            submittable()). Not asked: the checkmate and stalemate searches
//            submit() also runs, which judge the position before the action, are
//            the same for every candidate, and are what the position event
//            reports. The enumerator ends a sequence as soon as no present
//            timeline is left unplayed, and lists the moves before that in every
//            order it can play them; an action is listed once, in the first
//            order the library gave it. "count" is the number of distinct
//            actions, "truncated" whether "actions" lists fewer, "complete"
//            whether every sequence was tested; "sequences" and "accepted"
//            count the ordered sequences enumerated and taken. An enumeration
//            stopped by a limit says which ("limit": "actions_budget_ms" while
//            testing, else "deadline" or "mate_heap_mb", the worker's heap limit),
//            and one the library failed says why ("error"); the replay goes on.
//            Two things are worked around, not answered for the library: its
//            chess.actions() fails in its "object" and "json" formats (a
//            ReferenceError) and in "5dpgn" (a TypeError), and in every format
//            passes its promotion pieces in the place of the "spatial moves
//            only" switch of boardFuncs.moves() - so it would list no move
//            through time or across timelines. The driver calls
//            actionFuncs.actions() with no promotion pieces, which leaves that
//            switch off and lets the library offer its default promotions (every
//            non-royal piece kind on the boards); the move check above then
//            holds each promotion to the pieces the record allows. "caveat" says
//            when the record allows one the default omits.
'use strict';

const fs = require('fs');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

// 2: the state event (H30), the explain and actions diagnostics, and the limit
// a mate search that did not finish ran into.
const SCHEMA = 2;
// How long past its own budget a search in the worker may run before it is stopped.
const GRACE_MS = 3000;
// The most actions one actions event lists.
const ACTIONS_LISTED = 5000;

function loadLibrary(root) {
  return require(path.join(path.resolve(root), 'src', 'index.js'));
}

function message(err) {
  return String((err && err.message) || err);
}

// ------------------------------------------------------------- labels (both sides)

// Names for boards, timelines and squares, as the library's own export writes
// them: fenFuncs.toFen() - the function chess.fen(null, true) calls for each
// board - given the turn-zero and timeline-parity flags that export reads off
// the board it exports, and pgnFuncs.toSanCoord() for a square. Only the
// library turns an index of its arrays into a name.
function labeller(lib, current) {
  const turnZero = lib.boardFuncs.isTurnZero(current);
  const evenTimelines = lib.boardFuncs.isEvenTimeline(current);
  const board = (fullBoard, l, t) => {
    const fen = lib.fenFuncs.toFen(fullBoard[l][t], l, t, turnZero, evenTimelines);
    const match = /:([^:[\]]*):(\d+):([wb])\]$/.exec(fen);
    if (match === null) throw new Error('the library labelled a board "' + fen + '", which the driver cannot read');
    return { L: match[1], T: Number(match[2]), c: match[3] };
  };
  const square = (fullBoard, pos) =>
    Object.assign(board(fullBoard, pos[0], pos[1]), { square: lib.pgnFuncs.toSanCoord([pos[2], pos[3]]) });
  return {
    board,
    square,
    // A timeline, named by the label of its last board.
    timeline: (fullBoard, l) => board(fullBoard, l, fullBoard[l].length - 1).L,
    // A square and the piece on it, as its 5DFEN letter.
    piece: (fullBoard, pos) => {
      const code = Math.abs(fullBoard[pos[0]][pos[1]][pos[2]][pos[3]]);
      return Object.assign(square(fullBoard, pos), { piece: lib.fenFuncs.TO_FEN[code] || null });
    }
  };
}

// ---------------------------------------------------------------- worker side

// Would the library's move() and submit() take this sequence of raw moves?
// (See "actions" in the header for what is asked and what is not.)
function accepts(lib, fullBoard, actionNum, sequence, promotionPieces) {
  const board = lib.boardFuncs.copy(fullBoard);
  for (const move of sequence) {
    if (!lib.validateFuncs.move(board, actionNum, move, [], promotionPieces)) return false;
    lib.boardFuncs.move(board, move);
  }
  if (lib.mateFuncs.checks(board, actionNum, true)) return false;
  return lib.boardFuncs.present(board, actionNum).length <= 0;
}

// One action in the protocol's form: each move named on the boards as they
// stand when it is played, which is how the library itself names the moves of
// an action (parseFuncs.fromAction).
function describeAction(lib, labels, fullBoard, sequence) {
  const board = lib.boardFuncs.copy(fullBoard);
  return sequence.map((move) => {
    let promotion = null;
    if (move.length === 2 && move[1][4] !== undefined) {
      promotion = lib.pieceFuncs.toChar(move[1][4]);
      if (!promotion) throw new Error('the library promotes to a piece it cannot name: ' + JSON.stringify(move));
    }
    const described = { from: labels.square(board, move[0]), to: labels.square(board, move[1]), promotion };
    lib.boardFuncs.move(board, move);
    return described;
  });
}

function legalActions(lib, job) {
  const started = Date.now();
  // Every timeline, not only the active or present ones; inactive timelines
  // allowed; no promotion pieces (see the header).
  const sequences = lib.actionFuncs.actions(job.board, job.action, false, false, false, null);
  const labels = labeller(lib, job.board);
  const seen = new Set();
  const actions = [];
  let tested = 0;
  let accepted = 0;
  for (const sequence of sequences) {
    if (Date.now() - started > job.budgetMs) break;
    tested++;
    if (!accepts(lib, job.board, job.action, sequence, job.promotionPieces)) continue;
    accepted++;
    const key = sequence.map((move) => JSON.stringify(move)).sort().join(' ');
    if (seen.has(key)) continue;
    seen.add(key);
    if (actions.length < job.listed) actions.push(describeAction(lib, labels, job.board, sequence));
  }
  const complete = tested === sequences.length;
  return {
    count: seen.size, truncated: !complete || seen.size > actions.length, complete,
    sequences: sequences.length, accepted, actions
  };
}

function workerMain() {
  const Chess = loadLibrary(workerData.root);
  const lib = new Chess().raw;
  parentPort.on('message', (job) => {
    if (job.kind === 'actions') {
      parentPort.postMessage({ id: job.id, answer: legalActions(lib, job) });
      return;
    }
    // checkmate() and stalemate() both answer [verdict, timedOut].
    const answer = job.inCheck
      ? lib.mateFuncs.checkmate(job.board, job.action, job.budgetMs)
      : lib.mateFuncs.stalemate(job.board, job.action, job.budgetMs);
    parentPort.postMessage({ id: job.id, answer: [Boolean(answer[0]), Boolean(answer[1])] });
  });
}

// ------------------------------------------------------------------ main side

function emit(obj) {
  fs.writeSync(1, JSON.stringify(obj) + '\n');
}

function usage(code) {
  fs.writeSync(code === 0 ? 1 : 2,
    'usage: node driver.js <exported-root> [--board NAME] [--mate-budget-ms N] [--mate-heap-mb N]\n' +
    '                      [--explain] [--actions K] [--actions-budget-ms N]\n' +
    '  Reads a 5DPGN record from stdin; prints one JSON object per line.\n' +
    '  --explain    after each state event, what the library\'s check test looked at\n' +
    '  --actions K  at position K, list up to ' + ACTIONS_LISTED + ' legal actions, within --actions-budget-ms\n');
  process.exit(code);
}

function parseArgs(argv) {
  const opts = {
    root: null, board: null, mateBudgetMs: 5000, mateHeapMb: 1024,
    explain: false, actionsAt: null, actionsBudgetMs: 60000
  };
  const positive = (text) => {
    const value = Number(text);
    if (!(value > 0)) usage(2);
    return value;
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-h' || arg === '--help') usage(0);
    else if (arg === '--board' && i + 1 < argv.length) opts.board = argv[++i];
    else if (arg === '--mate-budget-ms' && i + 1 < argv.length) opts.mateBudgetMs = positive(argv[++i]);
    else if (arg === '--mate-heap-mb' && i + 1 < argv.length) opts.mateHeapMb = positive(argv[++i]);
    else if (arg === '--explain') opts.explain = true;
    else if (arg === '--actions' && i + 1 < argv.length) {
      const text = argv[++i];
      if (!/^\d+$/.test(text)) usage(2);
      opts.actionsAt = Number(text);
    } else if (arg === '--actions-budget-ms' && i + 1 < argv.length) opts.actionsBudgetMs = positive(argv[++i]);
    else if (opts.root === null) opts.root = arg;
    else usage(2);
  }
  if (opts.root === null) usage(2);
  return opts;
}

// A 5DFEN board string: [<pieces>:<timeline>:<turn>:<w|b>], wherever it stands.
// (The library itself reads one only when it has a line to itself.)
const FEN_BOARDS = /\[[A-Za-z0-9\/*+]+:[+-]?\d+:\d+:[wbWB]\]/g;
// A PGN game-result token closing the move text, with the comments that may
// follow it. It is not a move. (It has to go before the library splits the text
// into actions: "1/2-1/2" holds the very separator the library splits on.)
const RESULT_TOKEN = /(^|\s)(?:1-0|0-1|1\/2-1\/2|\*)((?:\s|\{[^{}]*\}|;[^\n]*)*)$/;

function withoutResult(record) {
  return record.replace(RESULT_TOKEN, '$1$2');
}

// The variant a record declares, read leniently: used only when the caller did
// not say (see --board).
function declaredBoard(record) {
  const header = record.match(/\[\s*Board\s+"([^"]*)"\s*\]/i) || record.match(/\[\s*Variant\s+"([^"]*)"\s*\]/i);
  return header === null ? 'Standard' : header[1].trim();
}

function countBoards(rawBoard) {
  let boards = 0;
  for (const timeline of rawBoard || []) {
    for (const board of timeline || []) {
      if (board) boards++;
    }
  }
  return boards;
}

// Would the library replay this record as the game it declares itself to be?
// Asked after the library's own set-up, of the library itself: it resolves a
// Board name it does not know - and a header its reader does not match - to
// "standard", it reads 5DFEN boards only under "custom", and even then only
// those that have a line to themselves. A record it would misread is refused,
// not answered.
function misread(chess, record, declared) {
  const selected = chess.metadata.board;
  const name = typeof selected === 'string' ? chess.raw.metadataFuncs.lookupVariantFull(selected) : null;
  if (name === null || name.toLowerCase() !== declared.trim().toLowerCase()) {
    return {
      reason: 'variant',
      detail: 'the record declares the variant "' + declared + '"; the library would set up "' +
        (name === null ? String(selected) : name) + '"'
    };
  }
  const strings = (record.match(FEN_BOARDS) || []).length;
  if (strings > 0 && selected !== 'custom') {
    return { reason: 'fen', detail: 'the record carries 5DFEN boards, which the library reads only when Board is "custom"' };
  }
  if (selected === 'custom') {
    const boards = countBoards(chess.rawBoard);
    if (boards === 0 || boards !== strings) {
      return {
        reason: 'fen',
        detail: 'the library set up ' + boards + ' board(s) from the record\'s ' + strings + ' 5DFEN board string(s)'
      };
    }
    // The library guesses from the boards it holds whether a game has a turn
    // zero (boardFuncs.isTurnZero: no board at its first time index), and from
    // the record's text whether it had one when it read the boards. A record
    // whose first board is not white's at turn 1 can be read one way and named
    // the other, which moves every board of the game to another turn. Its
    // boards must come out where the record put them.
    const declaredAt = labelsOf((record.match(FEN_BOARDS) || []).map((board) => board.slice(1, -1).split(':').slice(1)));
    const setUpAt = labelsOf(chess.fen(null, true).split('\n').filter((line) => line !== '')
      .map((line) => line.slice(1, -1).split(':').slice(1)));
    if (declaredAt.join(' ') !== setUpAt.join(' ')) {
      return {
        reason: 'fen',
        detail: 'the library set up the record\'s 5DFEN boards at ' + setUpAt.join(' ') + ', where the record puts them at ' +
          declaredAt.join(' ')
      };
    }
  }
  return null;
}

// The coordinates of a list of boards, each [L, T, c] as 5DFEN writes them, in
// one spelling ("+n" for a positive timeline, a lower-case colour) and sorted.
function labelsOf(boards) {
  return boards.map(([l, t, c]) => {
    const label = /^\d+$/.test(l) && Number(l) > 0 ? '+' + l : l;
    return label + ':' + Number(t) + ':' + String(c).toLowerCase();
  }).sort();
}

// The library's mate search and its action enumeration, behind a heap limit
// and a hard deadline. A question that exhausts either answers { limit }; the
// next question gets a fresh worker. A question that fails in any other way -
// the library throwing, or the worker exiting without an answer - answers
// { error }. A reply answers { answer }.
class Searches {
  constructor(opts) {
    this.opts = opts;
    this.worker = null;
    this.pending = null;
    this.nextId = 0;
  }

  start() {
    const worker = new Worker(__filename, {
      workerData: { root: this.opts.root },
      resourceLimits: { maxOldGenerationSizeMb: this.opts.mateHeapMb },
    });
    const mine = () => this.pending !== null && this.pending.worker === worker;
    worker.on('message', (reply) => {
      if (this.pending !== null && reply.id === this.pending.id) this.settle({ answer: reply.answer });
    });
    // Node reports a worker that ran out of heap with this error, before its exit.
    worker.on('error', (err) => {
      if (this.worker === worker) this.worker = null;
      if (!mine()) return;
      this.settle(err && err.code === 'ERR_WORKER_OUT_OF_MEMORY' ? { limit: 'mate_heap_mb' } : { error: message(err) });
    });
    worker.on('exit', (code) => {
      if (this.worker === worker) this.worker = null;
      if (mine()) this.settle({ error: 'the worker exited (code ' + code + ') without answering' });
    });
    this.worker = worker;
  }

  settle(result) {
    const pending = this.pending;
    this.pending = null;
    clearTimeout(pending.timer);
    pending.resolve(result);
  }

  ask(job, deadlineMs) {
    if (this.worker === null) this.start();
    const worker = this.worker;
    const id = this.nextId++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        if (this.worker === worker) this.worker = null;
        worker.terminate();
        if (this.pending !== null && this.pending.id === id) this.settle({ limit: 'deadline' });
      }, deadlineMs);
      this.pending = { id, worker, resolve, timer };
      worker.postMessage(Object.assign({ id }, job));
    });
  }

  async close() {
    if (this.worker !== null) {
      const worker = this.worker;
      this.worker = null;
      await worker.terminate();
    }
  }
}

// The width and height of the boards the library set up, from its own arrays
// (a board is a list of ranks, rank 1 first, each a list of squares, file a first).
function boardSize(setUp) {
  for (const timeline of setUp || []) {
    for (const board of timeline || []) {
      if (board) return [board[0].length, board.length];
    }
  }
  throw new Error('the library set up no board');
}

// The state event (see the header).
function stateEvent(chess, index) {
  const exported = chess.fen(null, true).split('\n').filter((line) => line !== '');
  const boards = [];
  let letters = 0;
  for (const line of exported) {
    const match = /^\[([^:[\]]*):([^:[\]]*):([^:[\]]*):([^:[\]]*)\]$/.exec(line);
    if (match === null) throw new Error('the library exported the board "' + line + '", which the driver cannot read');
    boards.push(match[2] + ':' + match[3] + ':' + match[4] + ':' + match[1]);
    letters += (match[1].match(/[A-Za-z]/g) || []).length;
  }
  const [width, height] = boardSize(chess.rawBoardHistory[0]);
  let held = 0;
  let pieces = 0;
  for (const timeline of chess.rawBoardHistory[chess.rawBoardHistory.length - 1] || []) {
    for (const board of timeline || []) {
      if (!board) continue;
      held++;
      if (board.length !== height || board.some((rank) => !rank || rank.length !== width)) {
        throw new Error('the library holds a board that is not ' + width + 'x' + height + ' at position ' + index);
      }
      for (const rank of board) pieces += rank.filter((square) => square !== 0).length;
    }
  }
  if (held !== boards.length || pieces !== letters) {
    throw new Error('the library\'s export of position ' + index + ' lists ' + boards.length + ' board(s) and ' +
      letters + ' piece(s); its arrays hold ' + held + ' and ' + pieces);
  }
  // The library's own full-state hash of the same export (chess.hash): the
  // harness holds its fingerprint to it.
  return { event: 'state', index: index, size: width + 'x' + height, markers: true, boards: boards, hash: chess.hash };
}

// The explain event (see the header).
function explainEvent(chess, index) {
  const lib = chess.raw;
  const current = chess.rawBoard;
  const mover = chess.rawAction;
  const labels = labeller(lib, current);
  // The multiverse the check test looks at, built as mateFuncs.checks() builds
  // it - and as chess.checks() builds it to name the captures it finds.
  const view = lib.boardFuncs.copy(current);
  lib.mateFuncs.blankAction(view, mover);
  const advanced = [];
  for (let l = 0; l < view.length; l++) {
    if (view[l] && current[l] && view[l].length > current[l].length) advanced.push(l);
  }
  const event = {
    event: 'explain',
    index: index,
    phantom_timelines: advanced.map((l) => labels.timeline(current, l)),
    phantom_boards: advanced.map((l) => {
      const board = labels.board(view, l, view[l].length - 1);
      return board.L + ':' + board.T + ':' + board.c;
    }),
    present_timelines: lib.boardFuncs.present(current, mover).map((l) => labels.timeline(current, l)),
    active_timelines: lib.boardFuncs.active(current).map((l) => labels.timeline(current, l))
  };
  const checks = lib.mateFuncs.checks(current, mover, false);
  if (checks.length > 0) {
    event.checks = checks.map((move) => ({ from: labels.piece(view, move[0]), to: labels.piece(view, move[1]) }));
  }
  return event;
}

// The actions event (see the header). A failure of the enumeration is reported
// in the event, not raised: it is a diagnostic, and the replay goes on.
async function actionsEvent(chess, index, search, opts) {
  const lib = chess.raw;
  const job = {
    kind: 'actions', board: chess.rawBoard, action: chess.rawAction,
    promotionPieces: chess.rawPromotionPieces, budgetMs: opts.actionsBudgetMs, listed: ACTIONS_LISTED
  };
  const result = await search.ask(job, opts.actionsBudgetMs + GRACE_MS);
  const found = result.answer !== undefined
    ? result.answer
    : { count: 0, truncated: true, complete: false, sequences: null, accepted: null, actions: [] };
  const event = {
    event: 'actions', index: index, count: found.count, truncated: found.truncated,
    complete: found.complete, sequences: found.sequences, accepted: found.accepted
  };
  if (result.answer !== undefined && !found.complete) event.limit = 'actions_budget_ms';
  if (result.limit !== undefined) event.limit = result.limit;
  if (result.error !== undefined) event.error = result.error;
  const offered = new Set(lib.pieceFuncs.availablePromotionPieces(chess.rawBoard));
  const omitted = (chess.rawPromotionPieces || []).filter((piece) => !offered.has(piece));
  if (omitted.length > 0) {
    event.caveat = 'the record allows promotion to ' +
      Array.from(new Set(omitted.map((piece) => lib.pieceFuncs.toChar(piece)))).join(', ') +
      ', which the library\'s default promotions omit: such actions are not listed';
  }
  event.actions = found.actions;
  return event;
}

async function observe(chess, index, search, opts) {
  emit(stateEvent(chess, index));
  if (opts.explain) emit(explainEvent(chess, index));
  const started = process.hrtime.bigint();
  const latest = chess.rawBoardHistory[chess.rawBoardHistory.length - 1];
  const inCheck = Boolean(chess.inCheck);
  // checkmate() answers "no" at once when not in check and stalemate() does the
  // same when in check, so only the relevant search is run.
  const result = await search.ask(
    { kind: 'mate', inCheck: inCheck, board: latest, action: chess.rawAction, budgetMs: opts.mateBudgetMs },
    opts.mateBudgetMs + GRACE_MS);
  if (result.error !== undefined) throw new Error('the mate search failed: ' + result.error);
  let terminal = null;
  let limit = result.limit === undefined ? null : result.limit;
  if (result.answer !== undefined) {
    if (result.answer[1]) limit = 'mate_budget_ms';
    else terminal = result.answer[0] ? (inCheck ? 'checkmate' : 'stalemate') : 'none';
  }
  const ms = Number(process.hrtime.bigint() - started) / 1e6;
  const event = {
    event: 'position',
    index: index,
    side: chess.player,
    in_check: inCheck,
    legal_actions: null,
    terminal: terminal,
    ms: Math.round(ms * 1000) / 1000
  };
  if (limit !== null) event.limit = limit;
  emit(event);
  if (opts.actionsAt === index) emit(await actionsEvent(chess, index, search, opts));
}

async function replay(opts, record, search) {
  let Chess;
  try {
    Chess = loadLibrary(opts.root);
  } catch (err) {
    emit({ event: 'end', status: 'error', actions_accepted: 0, reason: 'load', detail: message(err) });
    return;
  }

  const chess = new Chess();
  const convert = chess.raw.convertFuncs;
  const pgn = chess.raw.pgnFuncs;
  const convertActions = convert.actions;
  const toAction = pgn.toAction;

  // Step 1: let import() do its own set-up (variant, promotions, 5DFEN) with the
  // action list withheld.
  convert.actions = () => [];
  try {
    chess.import(record, undefined, true);
  } catch (err) {
    emit({ event: 'end', status: 'unsupported', actions_accepted: 0, reason: 'setup', detail: message(err) });
    return;
  } finally {
    convert.actions = convertActions;
  }
  const declared = opts.board === null ? declaredBoard(record) : opts.board;
  const unsupported = misread(chess, record, declared);
  if (unsupported !== null) {
    emit({ event: 'end', status: 'unsupported', actions_accepted: 0, reason: unsupported.reason, detail: unsupported.detail });
    return;
  }

  // Step 2: let the library convert the notation, keeping every action it
  // resolved before the first one it could not.
  const converted = [];
  let failure = null;
  pgn.toAction = (actionStr, board, actionNum, promotionPieces) => {
    try {
      const raw = toAction(actionStr, board, actionNum, promotionPieces);
      converted.push(raw);
      return raw;
    } catch (err) {
      if (failure === null) failure = { text: String(actionStr).trim(), detail: message(err) };
      throw err;
    }
  };
  let conversionError = null;
  try {
    convertActions(withoutResult(record), chess.rawBoardHistory[0], chess.rawStartingAction, chess.rawPromotionPieces);
  } catch (err) {
    conversionError = message(err);
  } finally {
    pgn.toAction = toAction;
  }
  if (conversionError !== null && failure === null) {
    // Nothing the library recognises as an action list.
    emit({ event: 'end', status: 'unsupported', actions_accepted: 0, reason: 'notation', detail: conversionError.slice(0, 300) });
    return;
  }

  const total = converted.length + (failure === null ? 0 : 1);
  emit({
    event: 'start', driver: 'chessin5d', schema: SCHEMA, board: chess.metadata.board,
    mate_budget_ms: opts.mateBudgetMs, mate_heap_mb: opts.mateHeapMb,
    actions_total: total, actions_total_is_lower_bound: failure !== null
  });

  // Step 3: replay with the library's own move() and submit(). Mate detection is
  // skipped inside submit() only: it is asked for explicitly in observe(), where
  // running out of time or memory can be told apart from a verdict.
  let accepted = 0;
  try {
    await observe(chess, 0, search, opts);
    for (let i = 0; i < converted.length; i++) {
      const last = failure === null && i + 1 === converted.length;
      try {
        for (const move of converted[i]) chess.move(move);
      } catch (err) {
        emit({ event: 'end', status: 'rejected', actions_accepted: accepted, reason: 'illegal_move', detail: message(err), last_action: last });
        return;
      }
      chess.skipDetection = true;
      try {
        chess.submit();
      } catch (err) {
        emit({ event: 'end', status: 'rejected', actions_accepted: accepted, reason: 'cannot_submit', detail: message(err), last_action: last });
        return;
      } finally {
        chess.skipDetection = false;
      }
      accepted++;
      await observe(chess, accepted, search, opts);
    }
  } catch (err) {
    emit({ event: 'end', status: 'error', actions_accepted: accepted, reason: 'exception', detail: message(err) });
    return;
  }

  if (failure !== null) {
    const noMove = failure.detail.includes('No valid move found');
    emit({
      event: 'end',
      status: noMove ? 'rejected' : 'unsupported',
      actions_accepted: accepted,
      reason: noMove ? 'no_matching_move' : 'notation',
      detail: (failure.detail + ' | action text: ' + failure.text).slice(0, 400)
    });
    return;
  }
  emit({ event: 'end', status: 'ok', actions_accepted: accepted });
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const record = fs.readFileSync(0, 'utf8');
  const search = new Searches(opts);
  try {
    await replay(opts, record, search);
  } finally {
    await search.close();
  }
}

if (isMainThread) {
  main().then(() => process.exit(0), (err) => {
    emit({ event: 'end', status: 'error', actions_accepted: 0, reason: 'exception', detail: message(err) });
    process.exit(1);
  });
} else {
  workerMain();
}
