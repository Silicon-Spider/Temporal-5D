// Test-time driver for the pinned "Chess in 5D" rules library (5d-chess-js).
//
// This file belongs to the oracle harness, never to the engine. It loads the
// oracle from an export of the locked commit and reports, position by position,
// what the oracle itself says. Every verdict comes from the library's own
// functions. The driver does four things around them, none of which touches
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
//     variant but "custom" (the library ignores them), and a custom position the
//     library set up only in part (it reads a board string only when the string
//     has a line to itself);
//   * it drops a PGN result token ("1-0", "1/2-1/2") from the end of the move
//     text, because the library reads it as moves and fails on it;
//   * it runs the library's mate search in a worker thread with its own heap
//     limit. On wide positions that search can exhaust memory long before its
//     time budget; in a worker that costs one unknown verdict, not the record.
//     (Any other failure of the search is an error of the run, not a silence.)
//
// The adapter (oracle_harness/adapters/chessin5d.py) refuses, before this runs,
// two notations the library's move-text reader gets wrong without noticing:
// "12w." turn markers and parenthesised variations.
//
// usage : node driver.js <exported-root> [--board NAME] [--mate-budget-ms N] [--mate-heap-mb N]
//         --board is the variant the record declares, as the caller reads it:
//         its Board header, else its Variant header, else "Standard". Without
//         it the driver reads the headers itself, leniently.
// stdin : one 5DPGN record.
// stdout: JSON Lines, flushed per line.
//   {"event":"start", ...}
//   {"event":"position","index":k,"side":"white|black","in_check":bool,
//    "legal_actions":null,"terminal":"none|checkmate|stalemate"|null,"ms":float}
//   {"event":"end","status":"ok|rejected|unsupported|error", ...}
//
// legal_actions is always null: the library's actions() enumerates ordered move
// sequences under its own options, which is not the quantity the other oracles
// count. terminal is null when the library's mate search ran out of time or
// memory.
'use strict';

const fs = require('fs');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

const SCHEMA = 1;
// How long past its own budget the mate search may run before it is stopped.
const GRACE_MS = 3000;

function loadLibrary(root) {
  return require(path.join(path.resolve(root), 'src', 'index.js'));
}

// ---------------------------------------------------------------- worker side

function workerMain() {
  const Chess = loadLibrary(workerData.root);
  const mate = new Chess().raw.mateFuncs;
  parentPort.on('message', (job) => {
    // checkmate() and stalemate() both answer [verdict, timedOut].
    const answer = job.inCheck
      ? mate.checkmate(job.board, job.action, job.budgetMs)
      : mate.stalemate(job.board, job.action, job.budgetMs);
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
    '  Reads a 5DPGN record from stdin; prints one JSON object per line.\n');
  process.exit(code);
}

function parseArgs(argv) {
  const opts = { root: null, board: null, mateBudgetMs: 5000, mateHeapMb: 1024 };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-h' || arg === '--help') usage(0);
    else if (arg === '--board' && i + 1 < argv.length) opts.board = argv[++i];
    else if (arg === '--mate-budget-ms' && i + 1 < argv.length) {
      opts.mateBudgetMs = Number(argv[++i]);
      if (!(opts.mateBudgetMs > 0)) usage(2);
    } else if (arg === '--mate-heap-mb' && i + 1 < argv.length) {
      opts.mateHeapMb = Number(argv[++i]);
      if (!(opts.mateHeapMb > 0)) usage(2);
    } else if (opts.root === null) opts.root = arg;
    else usage(2);
  }
  if (opts.root === null) usage(2);
  return opts;
}

function message(err) {
  return String((err && err.message) || err);
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
  }
  return null;
}

// The library's mate search, behind a heap limit and a hard deadline. A search
// that exhausts either answers null; the next question gets a fresh worker. A
// search that fails in any other way - the library throwing - answers
// { error }, and that is an error of the whole run, not a silence.
class MateSearch {
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
    const abandon = () => {
      if (this.worker === worker) this.worker = null;
      if (this.pending !== null && this.pending.worker === worker) this.settle(null);
    };
    worker.on('message', (reply) => {
      if (this.pending !== null && reply.id === this.pending.id) this.settle(reply.answer);
    });
    worker.on('error', (err) => {
      if (err && err.code === 'ERR_WORKER_OUT_OF_MEMORY') {
        abandon();
        return;
      }
      if (this.worker === worker) this.worker = null;
      if (this.pending !== null && this.pending.worker === worker) this.settle({ error: message(err) });
    });
    worker.on('exit', abandon);
    this.worker = worker;
  }

  settle(answer) {
    const pending = this.pending;
    this.pending = null;
    clearTimeout(pending.timer);
    pending.resolve(answer);
  }

  ask(inCheck, board, action) {
    if (this.worker === null) this.start();
    const worker = this.worker;
    const id = this.nextId++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        if (this.worker === worker) this.worker = null;
        worker.terminate();
        if (this.pending !== null && this.pending.id === id) this.settle(null);
      }, this.opts.mateBudgetMs + GRACE_MS);
      this.pending = { id, worker, resolve, timer };
      worker.postMessage({ id, inCheck, board, action, budgetMs: this.opts.mateBudgetMs });
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

async function observe(chess, index, search) {
  const started = process.hrtime.bigint();
  const latest = chess.rawBoardHistory[chess.rawBoardHistory.length - 1];
  const inCheck = Boolean(chess.inCheck);
  // checkmate() answers "no" at once when not in check and stalemate() does the
  // same when in check, so only the relevant search is run.
  const answer = await search.ask(inCheck, latest, chess.rawAction);
  if (answer !== null && answer.error !== undefined) throw new Error('the mate search failed: ' + answer.error);
  let terminal = null;
  if (answer !== null && !answer[1]) terminal = answer[0] ? (inCheck ? 'checkmate' : 'stalemate') : 'none';
  const ms = Number(process.hrtime.bigint() - started) / 1e6;
  emit({
    event: 'position',
    index: index,
    side: chess.player,
    in_check: inCheck,
    legal_actions: null,
    terminal: terminal,
    ms: Math.round(ms * 1000) / 1000
  });
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
    await observe(chess, 0, search);
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
      await observe(chess, accepted, search);
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
  const search = new MateSearch(opts);
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
