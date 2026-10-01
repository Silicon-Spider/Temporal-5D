// Test-time driver for the parser bundled with the notation specification
// (adri326/5dchess-notation: parsers/pgn.js and parsers/game.js), run as a
// parse-only oracle (decision H26).
//
// This file belongs to the oracle harness, never to the engine. It loads the
// parser from an export of the locked commit and reports what the parser read:
// at every position, every board it holds (the state event, decision H30), and
// how many actions it read before it stopped. It answers no rules question. The
// parser has no notion of check, legality or the end of a game: it applies each
// move as written, and finds the moving piece with a movement test of its own
// only where the move text leaves the origin square out. So a position event
// carries the side to move and nothing else, and a move the parser cannot apply
// is a failure to read the move text - a soft rejection - never a verdict on the
// move. Its acceptance of an action is not a verdict either.
//
// The parser has one entry point, pgn.parse(), which reads a whole record at
// once. The driver changes nothing in it; it watches it, through the module
// objects the parser reaches its helpers through:
//
//   * Game.prototype.play and .castle, the only calls that change a board. An
//     action is a run of moves of one colour, as the parser counts turns
//     (Game.play: a white move after a black one starts a new turn): when a move
//     of the other colour comes, the action before it is complete, and the
//     position after it is reported before that move is applied. A turn number
//     that does not change the colour ("1. e4 2. d4") therefore starts no
//     action, for the parser and for the driver alike.
//   * Game.prototype.active_player, which the parser sets at every turn number
//     and "/": the colour of a move the parser fails on before reaching play().
//   * the Game class, the board-set-up readers (parse_legacy_fen, parse_5dfen)
//     and the BOARDS table: which layout the parser looked up, what it read, and
//     when its set-up was done.
//
// Before position 0 is reported, what the parser set up is held against what
// the record declares (--board), and the record is refused - "unsupported",
// before any position - when the parser would replay another game: a board name
// it does not know (it would set up one empty board), a named layout set up on
// boards of another size (the parser takes the size from the Size header, or
// 8x8, never from its own layout), 5DFEN board strings under any Board but
// "custom" (it reads them only there) or not all read, and a set-up whose
// timeline labels mix "0" with "-0"/"+0" (reason "signed_zero", its own, so
// that the recorded outcome names it). That last one is a defect of the
// parser, left alone: parse_5dfen stores a "-0" board string as a signed zero,
// which the parser's own write_timeline() writes as "0", beside "+0". Where the
// set-up mixes nothing, the defect is reported as the parser holds it (a "-0"
// board labelled "0"), and the fingerprints show it.
//
// The parser reports an error by printing it and calling process.exit(1). The
// driver captures everything the parser prints - none of it reaches stdout - and
// turns process.exit into an exception it catches. Its own JSON lines are
// written with fs.writeSync, which nothing in the parser uses. The parser
// tokenises the whole record before it sets anything up, so a record it cannot
// tokenise yields no position at all ("unsupported", reason "notation"). A
// comment that is never closed sends its tokeniser into an endless loop; that
// record is refused before the parser sees it.
//
// usage : node driver.js <exported-root> [--board NAME] [--explain]
//         --board is the variant the record declares, as the caller reads it:
//         its Board header, else its Variant header, else "Standard". Without
//         it the driver reads the headers itself, leniently.
// stdin : one 5DPGN record.
// stdout: JSON Lines, one write per line.
//   {"event":"start","driver":"notation", ...}      once the set-up is accepted
//   then, for each position k:
//   {"event":"state","index":k,"size":"WxH","markers":false,"boards":["<L>:<T>:<c>:<placement>", ...]}
//   {"event":"explain","index":k, ...}               (--explain only)
//   {"event":"position","index":k,"side":"white"|"black"|null}
//   and last:
//   {"event":"end","status":"ok|rejected|unsupported|error","actions_accepted":n, ...}
//
// The state event lists every board the parser holds - every timeline, every
// board of it, superseded or not - named as the parser names them: the timeline
// by its own write_timeline(), the turn and colour by the formulas of its own
// export_5dfen(), and the pieces by its own PIECE_TO_FEN table, ranks from the
// top, files from a. markers is false: the parser drops the "*" of a 5DFEN
// board string, and keeps no record of which pieces have moved.
//
// side is the colour of the action that follows the position, as the parser
// read it: the colour it played that action's first move with (a turn number
// makes it white, a "/" black). After the last action it is the colour a
// turn number or "/" set after that action, if one did, and otherwise null.
//
// The explain event (--explain) is a diagnostic, never compared: at position 0
// the layout the parser set up; at position k the moves of action k as the
// parser resolved them, each {"from":{"L","T","c","square","piece"},
// "to":{"L","T","c","square","piece"},"promotion"} - the piece at "to" being the
// one taken, if any - with "castle" and "new_timeline" where they apply.
//
// A move the parser could not apply ends the run "rejected" with a soft reason:
// no_matching_move (no piece fits the move text, or the board it names does not
// exist), ambiguous_move, notation_mismatch (a pawn reaching the last rank with
// no promotion written) or unrecognised (anything else). A move it could not
// even place - no super-physical coordinates where there is more than one
// timeline - ends it "unsupported" (reason "notation"), with the positions read
// before it.
'use strict';

const fs = require('fs');
const path = require('path');
const util = require('util');

const SCHEMA = 1;
// How much of what the parser printed is kept for an end event's detail.
const PRINTED_KEPT = 2000;
// A 5DFEN board string, [<pieces>:<timeline>:<turn>:<w|b>], wherever it stands
// (the harness's headers.py looks for the same).
const FEN_BOARDS = /\[[A-Za-z0-9/*+]+:[+-]?\d+:\d+:[wbWB]\]/g;

const realExit = process.exit.bind(process);
const PAUSE = new Int32Array(new SharedArrayBuffer(4));

// One protocol line, written whole. stdout is a pipe the harness drains as it
// reads; should it ever be in non-blocking mode, a long line (a state event of a
// wide position) can be written in parts, or refused while the pipe is full.
function emit(obj) {
  const data = Buffer.from(JSON.stringify(obj) + '\n', 'utf8');
  let written = 0;
  while (written < data.length) {
    try {
      written += fs.writeSync(1, data, written, data.length - written);
    } catch (err) {
      if (err.code !== 'EAGAIN') throw err;
      Atomics.wait(PAUSE, 0, 0, 2); // the reader is behind: let it catch up
    }
  }
}

function message(err) {
  return String((err && err.message) || err);
}

function usage(code) {
  fs.writeSync(code === 0 ? 1 : 2,
    'usage: node driver.js <exported-root> [--board NAME] [--explain]\n' +
    '  Reads a 5DPGN record from stdin; prints one JSON object per line.\n' +
    '  --board NAME  the variant the record declares (Board header, else Variant header, else Standard)\n' +
    '  --explain     after each state event, the layout set up or the moves of the action, as the parser read them\n');
  realExit(code);
}

function parseArgs(argv) {
  const opts = { root: null, board: null, explain: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-h' || arg === '--help') usage(0);
    else if (arg === '--board' && i + 1 < argv.length) opts.board = argv[++i];
    else if (arg === '--explain') opts.explain = true;
    else if (opts.root === null && !arg.startsWith('-')) opts.root = arg;
    else usage(2);
  }
  if (opts.root === null) usage(2);
  return opts;
}

// The variant a record declares, read leniently: used only when the caller did
// not say (see --board). As the harness reads it: the Board header, else the
// Variant header, else Standard.
function declaredBoard(record) {
  for (const name of ['Board', 'Variant']) {
    const header = new RegExp('\\[\\s*' + name + '\\s+"([^"]*)"\\s*\\]', 'i').exec(record);
    if (header !== null && header[1].trim() !== '') return header[1].trim();
  }
  return 'Standard';
}

// --------------------------------------------------------------- what is printed

// Everything the parser prints, most recent last. It never reaches stdout.
let printed = '';

function capture(text) {
  printed = (printed + text).slice(-PRINTED_KEPT);
}

// The parser calls process.exit(1) after printing an error; this is thrown in
// its place, from inside the parser's own catch blocks, and caught by the driver.
class ParserExit extends Error {
  constructor(code) {
    super('the parser called process.exit(' + code + ')');
    this.code = code;
  }
}

// What a hook decided - a refusal of the record, or a failure of the driver's
// own - thrown to stop the parse. It is also remembered: the parser catches
// what play() throws, prints it and exits, and the exit is what reaches the driver.
class Stop extends Error {
  constructor(end) {
    super(end.detail);
    this.end = end;
  }
}

// The parser prints through process.stdout and console. Neither may reach the
// real stdout - and process.stdout is replaced before anything opens it: on
// Linux, opening it puts the pipe into non-blocking mode.
function silenceParser() {
  const sink = {
    isTTY: false,
    write(chunk, encoding, callback) {
      capture(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'));
      const done = typeof encoding === 'function' ? encoding : callback;
      if (typeof done === 'function') done();
      return true;
    },
    on() {
      return sink;
    },
    once() {
      return sink;
    },
    end() {}
  };
  Object.defineProperty(process, 'stdout', { configurable: true, enumerable: true, get: () => sink });
  for (const name of ['log', 'info', 'warn', 'error', 'debug', 'trace', 'dir']) {
    console[name] = (...args) => capture(util.format(...args) + '\n');
  }
  process.exit = (code) => {
    throw new ParserExit(code);
  };
}

// The last error the parser printed (console.error(err.toString()) before it exits).
function printedError() {
  const lines = printed.split('\n').map((line) => line.trim()).filter((line) => line !== '');
  const error = lines.filter((line) => /^[A-Za-z]*Error\b/.test(line)).pop();
  return error === undefined ? lines.slice(-3).join(' | ') : error;
}

function end(status, accepted, reason, detail) {
  const event = { event: 'end', status: status, actions_accepted: accepted };
  if (reason) event.reason = reason;
  if (detail) event.detail = String(detail).slice(0, 400);
  return event;
}

// ------------------------------------------------------------------ the boards

// A board as the 5DFEN pieces field writes it: ranks from the top (the parser
// stores rank 1 first), files from a, the parser's own letters, runs of empty
// squares as numbers.
function placement(game, lib, board, where) {
  const width = game.width;
  const height = game.height;
  if (!Array.isArray(board) || board.length !== width * height) {
    throw new Error('the parser holds a board at ' + where + ' that is not ' + width + 'x' + height + ' squares');
  }
  const ranks = [];
  for (let y = height - 1; y >= 0; y--) {
    let text = '';
    let blank = 0;
    for (let x = 0; x < width; x++) {
      const piece = board[x + y * width];
      if (piece === lib.PIECES.BLANK) {
        blank++;
        continue;
      }
      const letter = lib.PIECE_TO_FEN[piece];
      if (typeof letter !== 'string') {
        throw new Error('the parser holds a piece it has no 5DFEN letter for (' + String(piece) + ') at ' + where +
          ', ' + lib.index_to_letter(x) + (y + 1));
      }
      if (blank > 0) text += String(blank);
      blank = 0;
      text += letter;
    }
    if (blank > 0) text += String(blank);
    ranks.push(text);
  }
  return ranks.join('/');
}

// A board's turn and colour from the parser's time index (t = 2 * (T - 1) for
// white's board of turn T, one more for black's; turn zero is -2 and -1), by
// the formulas of the parser's own export_5dfen().
function turnOf(t) {
  return Math.floor((t + 2) / 2);
}

function colourOf(t) {
  return t % 2 ? 'b' : 'w';
}

function stateEvent(game, lib, index) {
  const boards = [];
  for (const timeline of game.timelines) {
    const label = lib.write_timeline(timeline.index);
    for (let dt = 0; dt < timeline.states.length; dt++) {
      const board = timeline.states[dt];
      if (board === undefined || board === null) continue; // a gap in a timeline set up from 5DFEN
      const t = timeline.begins_at + dt;
      const where = label + ':' + turnOf(t) + ':' + colourOf(t);
      boards.push(where + ':' + placement(game, lib, board, where));
    }
  }
  return { event: 'state', index: index, size: game.width + 'x' + game.height, markers: false, boards: boards };
}

// One move as the parser resolved it (its Move object), for --explain.
function describeMove(game, lib, move) {
  const colour = move.white ? 'w' : 'b';
  const letter = (code) => (code ? lib.PIECE_TO_FEN[code] || null : null);
  const at = (coordinates, piece) => ({
    L: lib.write_timeline(coordinates[0]),
    T: coordinates[1] + 1,
    c: colour,
    square: lib.index_to_letter(coordinates[2]) + String(coordinates[3] + 1),
    piece: letter(piece)
  });
  const described = { from: at(move.from, move.src_piece), to: at(move.to, move.dst_piece), promotion: letter(move.promotion) };
  if (move.castle) described.castle = move.castle_long ? 'long' : 'short';
  if (move.new_index !== null && move.new_index !== undefined) described.new_timeline = lib.write_timeline(move.new_index);
  return described;
}

// ----------------------------------------------------------------- the watcher

class Watcher {
  constructor(opts, record) {
    this.opts = opts;
    this.record = record;
    this.declared = opts.board === null ? declaredBoard(record) : opts.board.trim();
    this.lib = null; // game.js's exports
    this.layouts = null; // its BOARDS table, unwatched
    this.lookups = []; // the names the parser looked up in BOARDS
    this.games = []; // every Game the parser constructed, with the timelines it asked for
    this.reading = null; // 'layout' | '5dfen' while the parser reads a board set-up
    this.layoutsRead = 0;
    this.fenBoardsRead = 0;
    this.phase = 'tokens'; // 'tokens' -> 'setup' (a Game was constructed) -> 'moves'
    this.colour = true; // the colour the parse is in (white first, as the parser starts)
    this.tokenColour = null; // the colour a turn number or "/" set since the last move, if any
    this.actionColour = null; // the colour of the action being read
    this.accepted = 0; // actions complete
    this.reported = -1; // the last position reported
    this.actionMoves = []; // the action being read, as the parser resolved its moves
    this.lastAction = []; // the last action complete
    this.moveFailure = null; // { white, error, text } of a move play()/castle() failed on
    this.stop = null; // the end event a hook decided on
  }

  // Hooks on game.js's exports, installed before pgn.js reads them.
  hook(lib) {
    const watcher = this;
    this.lib = lib;
    this.layouts = lib.BOARDS;
    const Original = lib.Game;
    const constructed = new WeakSet();
    class WatchedGame extends Original {
      constructor(...args) {
        watcher.phase = 'setup';
        super(...args);
        constructed.add(this);
        watcher.games.push({ game: this, timelines: Array.isArray(args[2]) ? args[2].length : null });
      }
    }
    const ACTIVE = Symbol('active_player');
    let echo = false; // play() is followed by the parser setting active_player to the colour it played
    Object.defineProperty(Original.prototype, 'active_player', {
      configurable: true,
      get() {
        return this[ACTIVE];
      },
      set(value) {
        this[ACTIVE] = value;
        if (!constructed.has(this)) return; // the constructor's own default
        if (echo) {
          echo = false;
          return;
        }
        watcher.token(Boolean(value));
      }
    });
    for (const name of ['play', 'castle']) {
      const original = Original.prototype[name];
      Original.prototype[name] = function (...args) {
        const result = watcher.move(this, original, args);
        echo = name === 'play';
        return result;
      };
    }
    for (const [name, kind] of [['parse_legacy_fen', 'layout'], ['parse_5dfen', '5dfen']]) {
      const original = Original.prototype[name];
      Original.prototype[name] = function (...args) {
        watcher.reading = kind;
        const result = original.apply(this, args);
        watcher.reading = null; // left set if the parser throws: the set-up failed
        if (kind === 'layout') watcher.layoutsRead++;
        else watcher.fenBoardsRead++;
        return result;
      };
    }
    lib.Game = WatchedGame;
    lib.BOARDS = new Proxy(this.layouts, {
      get(target, key, receiver) {
        if (typeof key === 'string') watcher.lookups.push(key);
        return Reflect.get(target, key, receiver);
      }
    });
  }

  game() {
    return this.games.length > 0 ? this.games[this.games.length - 1].game : null;
  }

  // Stop the parse with this end event.
  halt(event) {
    this.stop = event;
    throw new Stop(event);
  }

  // Report a position: its state, the explanation if asked, and the position event.
  report(index, white) {
    const game = this.game();
    let state;
    let explain = null;
    try {
      state = stateEvent(game, this.lib, index);
      if (this.opts.explain) explain = this.explainEvent(game, index);
    } catch (err) {
      this.halt(end('error', this.accepted, 'state', message(err)));
    }
    emit(state);
    if (explain !== null) emit(explain);
    emit({ event: 'position', index: index, side: white === null ? null : white ? 'white' : 'black' });
    this.reported = index;
  }

  explainEvent(game, index) {
    if (index === 0) {
      return {
        event: 'explain', index: 0, board: this.lookups[0], layout: this.layout(),
        size: game.width + 'x' + game.height,
        timelines: game.timelines.map((timeline) => this.lib.write_timeline(timeline.index))
      };
    }
    return { event: 'explain', index: index, moves: this.lastAction.map((move) => describeMove(game, this.lib, move)) };
  }

  // The action being read is complete: report the position after it.
  complete(next) {
    this.accepted++;
    this.lastAction = this.actionMoves;
    this.actionMoves = [];
    this.report(this.accepted, next);
  }

  // The parser's set-up is done (it has moved on to the move text): check it,
  // and refuse the record if it is not the game the record declares.
  begin() {
    if (this.phase === 'moves') return;
    const refusal = this.checkSetup(true);
    if (refusal !== null) this.halt(refusal);
    this.phase = 'moves';
    const game = this.game();
    emit({
      event: 'start', driver: 'notation', schema: SCHEMA, board: this.lookups[0], layout: this.layout(),
      size: game.width + 'x' + game.height
    });
  }

  // Which branch of its set-up the parser took for the board name it looked up:
  // 'named' (a layout of its BOARDS table), 'custom' (a game with no timeline,
  // filled from the record's 5DFEN boards), or null (neither: one empty board).
  layout() {
    if (this.lookups.length === 0) return null;
    if (Object.prototype.hasOwnProperty.call(this.layouts, this.lookups[0])) return 'named';
    const last = this.games[this.games.length - 1];
    return this.games.length > 1 && last.timelines === 0 ? 'custom' : null;
  }

  // null when the parser set up the game the record declares; else the end
  // event refusing the record. ``complete`` says the set-up finished; when it did
  // not (the parser threw while setting up), only what can already be told is
  // checked, and null means "nothing to refuse it for yet".
  checkSetup(complete) {
    const unsupported = (reason, detail) => end('unsupported', 0, reason, detail);
    const game = this.game();
    if (game === null || this.lookups.length === 0) {
      if (!complete) return null;
      return end('error', 0, 'hooks', 'the driver did not see the parser set up a game (' + this.games.length +
        ' constructed, ' + this.lookups.length + ' board name(s) looked up): its hooks no longer fit the parser');
    }
    const key = this.lookups[0];
    const declared = this.declared;
    const layout = this.layout();
    if (layout === null) {
      return unsupported('variant', 'the record declares the variant "' + declared + '", which the parser does not know: ' +
        'it would set up a single empty board');
    }
    if (layout === 'named' ? key !== declared.toUpperCase() : declared.toLowerCase() !== 'custom') {
      return unsupported('variant', 'the record declares the variant "' + declared + '"; the parser set up "' + key + '"');
    }
    const strings = (this.record.match(FEN_BOARDS) || []).length;
    if (layout === 'named') {
      if (strings > 0) {
        return unsupported('fen', 'the record carries 5DFEN boards, which the parser reads only under Board "custom": ' +
          'it would replay its own layout of "' + declared + '"');
      }
      const size = this.layouts[key][2];
      if (game.width + 'x' + game.height !== size) {
        return unsupported('size', 'the parser set its "' + declared + '" layout (' + size + ') up on ' + game.width + 'x' +
          game.height + ' boards: it takes the size from the Size header, or 8x8 without one');
      }
    }
    if (!complete) return null;
    if (layout === 'custom' && (this.fenBoardsRead === 0 || this.fenBoardsRead !== strings)) {
      return unsupported('fen', 'the parser set up ' + this.fenBoardsRead + ' board(s) from the record\'s ' + strings +
        ' 5DFEN board string(s)');
    }
    const labels = game.timelines.map((timeline) => this.lib.write_timeline(timeline.index));
    if (labels.includes('0') && (labels.includes('-0') || labels.includes('+0'))) {
      // A reason of its own, so that the recorded outcome names the parser's
      // defect and is not taken for a set-up it merely could not do.
      return unsupported('signed_zero', 'the parser labels its timelines ' + labels.join(', ') + ': it stores a 5DFEN ' +
        '"-0" board as a signed zero, which it writes as "0", beside "+0" - an odd and an even game at once');
    }
    return null;
  }

  // A turn number (white) or "/" (black).
  token(white) {
    this.begin();
    this.colour = white;
    this.tokenColour = white;
  }

  // play() or castle(): its fourth argument is the colour the parser plays it with.
  move(game, original, args) {
    this.begin();
    const white = Boolean(args[3]);
    if (this.reported < 0) this.report(0, white);
    else if (white !== this.actionColour) this.complete(white);
    this.actionColour = white;
    this.colour = white;
    this.tokenColour = null;
    let result;
    try {
      result = original.apply(game, args);
    } catch (err) {
      const token = args[5];
      this.moveFailure = { white: white, error: err, text: token && typeof token.raw === 'string' ? token.raw : '' };
      throw err;
    }
    this.actionMoves.push(result);
    return result;
  }

  // The end event, once pgn.parse() returned (failure null) or threw.
  finish(failure) {
    if (this.stop !== null) return this.stop;
    try {
      return failure === null ? this.finishRead() : this.finishStopped(failure);
    } catch (err) {
      if (err instanceof Stop) return err.end;
      throw err;
    }
  }

  finishRead() {
    this.begin(); // a record with no move: its set-up has not been checked yet
    const next = this.tokenColour !== null && this.tokenColour !== this.actionColour ? this.tokenColour : null;
    if (this.reported < 0) this.report(0, next);
    else this.complete(next);
    return end('ok', this.accepted);
  }

  finishStopped(failure) {
    const said = failure instanceof ParserExit ? printedError() : message(failure);
    if (this.phase === 'tokens') {
      return end('unsupported', 0, 'notation', 'the parser could not read the record: ' + said);
    }
    if (this.phase === 'setup') {
      // The parser threw while setting up, or after it and before reaching a
      // move. A record it would set up as another game is refused as such.
      const refusal = this.checkSetup(false);
      if (refusal !== null) return refusal;
      // Its set-up done (nothing is being read, and the layout, if named, was
      // read), what it threw on is the first move: no super-physical coordinates
      // where there is more than one timeline.
      const layout = this.layout();
      const setUp = this.reading === null && this.game() !== null && layout !== null &&
        (layout === 'custom' || this.layoutsRead > 0);
      if (!setUp) return end('unsupported', 0, 'setup', 'the parser could not set the record\'s position up: ' + said);
      this.begin();
    }
    return this.finishMoves(failure, said);
  }

  // The parser stopped on a move.
  finishMoves(failure, said) {
    const inside = this.moveFailure !== null;
    const white = inside ? this.moveFailure.white : this.colour;
    if (this.reported < 0) this.report(0, white);
    else if (!inside && white !== this.actionColour) this.complete(white);
    if (!inside) {
      return end('unsupported', this.accepted, 'notation', 'the parser stopped reading the moves: ' + said);
    }
    const text = message(this.moveFailure.error);
    const detail = text + (this.moveFailure.text ? ' | move text: ' + this.moveFailure.text : '');
    return end('rejected', this.accepted, rejection(text), detail);
  }
}

// Why the parser could not apply a move, by its message (Game.play, Game.castle).
// Every reason is soft: the parser holds no move legal or illegal.
function rejection(text) {
  if (/^(No piece candidate found|Invalid source board|Invalid target board|No valid (king|rooks) found to castle)/.test(text)) {
    return 'no_matching_move';
  }
  if (/^Ambiguous (move|castling)/.test(text)) return 'ambiguous_move';
  if (/^Pawn reaches the end of the board without promoting/.test(text)) return 'notation_mismatch';
  return 'unrecognised';
}

// ----------------------------------------------------------------------- main

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const record = fs.readFileSync(0, 'utf8');
  if (record.lastIndexOf('{') > record.lastIndexOf('}')) {
    emit(end('unsupported', 0, 'notation',
      'the record holds a comment that is never closed, on which the parser\'s tokeniser loops forever'));
    return;
  }
  const watcher = new Watcher(opts, record);
  silenceParser();
  let pgn;
  try {
    const parsers = path.join(path.resolve(opts.root), 'parsers');
    watcher.hook(require(path.join(parsers, 'game.js')));
    pgn = require(path.join(parsers, 'pgn.js'));
  } catch (err) {
    emit(end('error', 0, 'load', message(err)));
    return;
  }
  let failure = null;
  try {
    pgn.parse(record);
  } catch (err) {
    failure = err;
  }
  emit(watcher.finish(failure));
}

try {
  main();
  realExit(0);
} catch (err) {
  emit(end('error', 0, 'exception', message(err)));
  realExit(1);
}
