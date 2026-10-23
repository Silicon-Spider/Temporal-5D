// Test-time driver for the pinned Kairos oracle (FirePlank/5d-chess-engine).
//
// This file belongs to the oracle harness, never to the engine. It is compiled against
// the oracle's exported crate (`../src` in the harness cache, Cargo.toml) and exists
// only so the gate can ask the oracle its questions about a 5dpgn record in one replay:
// none of its programs answers them for a record (PROVENANCE.md, "Kairos, the fourth
// oracle"). All rules logic is the oracle's own - its 5dpgn reader, its set-up, its
// replay, its check test, its mate test and its two action searches. The code below
// only walks the main line it read and prints what it says. Nothing here is copied
// from the oracle.
//
// usage: kairos_oracle_driver --cap N --search hypercuboid|naive [--actions K]
// stdin : one 5dpgn record, exact bytes (UTF-8).
// stdout: JSON Lines (oracles/oracle_harness/adapters/base.py):
//   {"event":"start", ...}
//   then, for each position k:
//   {"event":"state","index":k,"size":"WxH","markers":true,"boards":["<L>:<T>:<c>:<placement>", ...]}
//   {"event":"position","index":k,"side":"white|black","in_check":bool,
//    "legal_actions":n|null,"terminal":"none|checkmate|stalemate","ms":float}
//   {"event":"actions","index":k,"count":n,"truncated":bool,"actions":[...]}   (--actions k only)
//   and last:
//   {"event":"end","status":"ok|rejected|unsupported|error", ...}
//
// Position k is the state after k submitted actions of the main line. Every line is
// flushed as soon as it is known, so a caller that kills a slow run keeps the positions
// already answered; the state event of a position comes before anything is searched there.
//
// Reading the record. `web::pgn::parse_5dpgn` resolves every action of the record -
// variations included - against the legal moves before it returns, and fails the whole
// record on the first one it cannot match or submit. Then nothing is known about any
// position, and the run ends `unsupported` (reason `parse_failed`, the reader's message as
// the detail) before position 0: never a refusal, which would claim a position was read.
// The reader does not test royal safety. Kairos's game loader (`web::record::load_pgn`)
// replays the main line it read through `web::tree::play_action` - every move from a
// playable board, the present moved, and no royal of the mover's capturable after it -
// and fails the record on the first action that breaks it; this driver replays the same
// actions through the same function and stops there instead, refusing that action
// (`cannot_submit` for a royal left capturable or the present not moved, `illegal_move`
// for a move it cannot play), so the positions before it are reported.
//
// The state event (decision H30) is every board the oracle holds - superseded and head,
// every timeline, both colours - from `Multiverse::get_boards(true)`: the oracle's own
// turn and colour of each board, its timeline named by its layout's `pretty_l` (the
// notation's labels, `-0`/`+0` in an even game), and its own 5DFEN pieces field, which
// writes "*" after each king, rook, pawn and brawn it holds unmoved - the only pieces it
// tracks - hence "markers":true. Its own position hash is a Zobrist key, not the
// notation's MD5, so no "hash" is handed over.
//
// The answers, by search (`--search`):
//   in_check  both: `State::has_phantom_check()`.
//   hypercuboid
//     terminal       `State::get_mate_type()`: its hypercuboid search, uninterruptible -
//                    checkmate, stalemate, or none (a legal action exists: None or Softmate).
//     legal_actions  null. Its hypercuboid search leaves out the actions that waste a
//                    move, so what it yields is not the gate's count (PROVENANCE.md).
//   naive
//     legal_actions  `core::naive::naive_actions_budget(state, cap, unbounded)`: its
//                    exhaustive enumerator, stopped at the cap (a count equal to the cap
//                    means "at least that many").
//     terminal       none when it finds an action; when it finds none - an exhaustive
//                    search that terminated - checkmate if `has_phantom_check()`, else
//                    stalemate: the rule `get_mate_type()` applies to its own search.
//
// The actions event is a diagnostic, never compared: the legal actions of one position
// as the chosen search yields them, at most 5000, each move
// {"from":{"L","T","c","square"},"to":{...},"promotion"} - the promotion being the piece
// the oracle promotes to when it plays the move in that order (`normalize_promotion`),
// or null. "count" is how many are listed and "truncated" whether the search had more.
// For the hypercuboid search a "caveat" says what it leaves out.

use std::io::{self, Read, Write};
use std::time::Instant;

use kairos::core::hypercuboid::HcInfo;
use kairos::core::naive::naive_actions_budget;
use kairos::core::piece::NO_PIECE;
use kairos::core::action::{ExtMove, MoveSeq};
use kairos::core::state::{MateType, State};
use kairos::core::variants::state_from_setup;
use kairos::core::vec4::Vec4;
use kairos::web::pgn::parse_5dpgn;
use kairos::web::tree::play_action;

const SCHEMA: u32 = 1;
const ACTIONS_LIMIT: usize = 5000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Search {
    Hypercuboid,
    Naive,
}

struct Options {
    cap: usize,
    search: Search,
    actions_at: Option<usize>,
}

fn usage(message: &str) -> ! {
    eprintln!("kairos_oracle_driver: {message}");
    eprintln!("usage: kairos_oracle_driver --cap N --search hypercuboid|naive [--actions K]");
    std::process::exit(2);
}

fn parse_options() -> Options {
    let mut options = Options { cap: 1000, search: Search::Hypercuboid, actions_at: None };
    let mut search = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage(&format!("{arg} needs a value")));
        match arg.as_str() {
            "--cap" => {
                options.cap = value().parse().unwrap_or_else(|_| usage("--cap takes a whole number"));
                if options.cap == 0 {
                    usage("--cap must be at least 1");
                }
            }
            "--search" => {
                search = Some(match value().as_str() {
                    "hypercuboid" => Search::Hypercuboid,
                    "naive" => Search::Naive,
                    other => usage(&format!("unknown search '{other}'")),
                })
            }
            "--actions" => {
                options.actions_at = Some(value().parse().unwrap_or_else(|_| usage("--actions takes a position index")))
            }
            other => usage(&format!("unknown argument '{other}'")),
        }
    }
    options.search = search.unwrap_or_else(|| usage("--search is required"));
    options
}

/// A JSON string literal.
fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn emit(line: &str) {
    let mut out = io::stdout().lock();
    // A closed pipe (the harness stopped the run) ends the process quietly.
    if writeln!(out, "{line}").and_then(|_| out.flush()).is_err() {
        std::process::exit(0);
    }
}

fn emit_end(status: &str, accepted: usize, reason: &str, detail: &str, last_action: Option<bool>) {
    let mut line = format!("{{\"event\":\"end\",\"status\":{},\"actions_accepted\":{}", quote(status), accepted);
    if !reason.is_empty() {
        line += &format!(",\"reason\":{}", quote(reason));
    }
    if !detail.is_empty() {
        line += &format!(",\"detail\":{}", quote(detail));
    }
    if let Some(last) = last_action {
        line += &format!(",\"last_action\":{last}");
    }
    line.push('}');
    emit(&line);
}

fn colour(black: bool) -> &'static str {
    if black { "b" } else { "w" }
}

fn size_of(state: &State) -> String {
    let (x, y) = state.get_board_size();
    format!("{x}x{y}")
}

fn emit_state(state: &State, index: usize) {
    let multiverse = state.multiverse();
    let boards: Vec<String> = multiverse
        .get_boards(true)
        .into_iter()
        .map(|(l, t, black, placement)| quote(&format!("{}:{}:{}:{}", multiverse.pretty_l(l), t, colour(black), placement)))
        .collect();
    emit(&format!(
        "{{\"event\":\"state\",\"index\":{index},\"size\":{},\"markers\":true,\"boards\":[{}]}}",
        quote(&size_of(state)),
        boards.join(",")
    ));
}

/// The search's legal actions, up to `limit`, and whether it had more.
fn list_actions(state: &State, search: Search, limit: usize) -> (Vec<MoveSeq>, bool) {
    match search {
        Search::Hypercuboid => {
            let (info, space) = HcInfo::build_hc(state);
            let mut found: Vec<MoveSeq> = info.search(state, space).take(limit + 1).collect();
            let more = found.len() > limit;
            found.truncate(limit);
            (found, more)
        }
        Search::Naive => {
            let mut found = naive_actions_budget(state, limit + 1, u64::MAX).unwrap_or_default();
            let more = found.len() > limit;
            found.truncate(limit);
            (found, more)
        }
    }
}

fn coordinate(state: &State, v: Vec4, black: bool) -> String {
    let square = format!("{}{}", (b'a' + v.x() as u8) as char, v.y() + 1);
    format!(
        "{{\"L\":{},\"T\":{},\"c\":\"{}\",\"square\":\"{}\"}}",
        quote(&state.multiverse().pretty_l(v.l())),
        v.t(),
        colour(black),
        square
    )
}

/// One action as the protocol lists it: its moves in the order the search yields them,
/// each with the piece it promotes to when the oracle plays it in that order.
fn action_json(state: &State, action: &MoveSeq) -> String {
    let black = state.get_present().1;
    let mut played = state.clone();
    let mut moves = Vec::new();
    for (i, fm) in action.as_slice().iter().enumerate() {
        let wanted = ExtMove::new(*fm, action.promo(i));
        let normal = played.normalize_promotion(wanted).unwrap_or(wanted);
        let promotion = if normal.promote_to == NO_PIECE {
            "null".to_string()
        } else {
            quote(&(normal.promote_to as char).to_ascii_uppercase().to_string())
        };
        moves.push(format!(
            "{{\"from\":{},\"to\":{},\"promotion\":{}}}",
            coordinate(&played, fm.from, black),
            coordinate(&played, fm.to, black),
            promotion
        ));
        played.apply_ext::<true>(normal);
    }
    format!("[{}]", moves.join(","))
}

fn emit_actions(state: &State, index: usize, search: Search) {
    let (found, more) = list_actions(state, search, ACTIONS_LIMIT);
    let listed: Vec<String> = found.iter().map(|action| action_json(state, action)).collect();
    let caveat = match search {
        Search::Hypercuboid => ",\"caveat\":\"the hypercuboid search leaves out actions that waste a move\"",
        Search::Naive => "",
    };
    emit(&format!(
        "{{\"event\":\"actions\",\"index\":{index},\"count\":{},\"truncated\":{more}{caveat},\"actions\":[{}]}}",
        listed.len(),
        listed.join(",")
    ));
}

fn report(state: &State, index: usize, options: &Options) {
    emit_state(state, index);
    let started = Instant::now();
    let in_check = state.has_phantom_check();
    let (count, terminal) = match options.search {
        Search::Hypercuboid => {
            let terminal = match state.get_mate_type() {
                MateType::Checkmate => "checkmate",
                MateType::Stalemate => "stalemate",
                MateType::None | MateType::Softmate => "none",
            };
            ("null".to_string(), terminal)
        }
        Search::Naive => {
            let found = naive_actions_budget(state, options.cap, u64::MAX).unwrap_or_default().len();
            let terminal = match (found, in_check) {
                (0, true) => "checkmate",
                (0, false) => "stalemate",
                _ => "none",
            };
            (found.to_string(), terminal)
        }
    };
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    let side = if state.get_present().1 { "black" } else { "white" };
    emit(&format!(
        "{{\"event\":\"position\",\"index\":{index},\"side\":\"{side}\",\"in_check\":{in_check},\
         \"legal_actions\":{count},\"terminal\":\"{terminal}\",\"ms\":{ms:.3}}}"
    ));
    if options.actions_at == Some(index) {
        emit_actions(state, index, options.search);
    }
}

/// Why `play_action` refused an action, in the protocol's words: a move it cannot play is
/// `illegal_move`; an action whose moves it played but may not submit - the present not
/// moved, or a royal of the mover's left capturable (submission legality, as the other
/// oracles and the rulebook's R1 put it) - is `cannot_submit`.
fn refusal(message: &str) -> &'static str {
    if message.contains("does not move the present") || message.contains("leaves a royal piece in check") {
        "cannot_submit"
    } else if message.contains("is not from a playable board")
        || message.starts_with("illegal move")
        || message.starts_with("bad promotion")
    {
        "illegal_move"
    } else {
        "unrecognised"
    }
}

fn main() {
    let options = parse_options();
    let mut bytes = Vec::new();
    if let Err(error) = io::stdin().read_to_end(&mut bytes) {
        emit_end("error", 0, "stdin", &error.to_string(), None);
        return;
    }
    let text = match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(_) => {
            emit_end("unsupported", 0, "encoding", "the record is not UTF-8", None);
            return;
        }
    };
    let game = match parse_5dpgn(&text, None) {
        Ok(game) => game,
        Err(message) => {
            emit_end("unsupported", 0, "parse_failed", &message, None);
            return;
        }
    };
    let mut state = match state_from_setup(&game.setup) {
        Ok(state) => state,
        Err(message) => {
            emit_end("unsupported", 0, "setup", &message, None);
            return;
        }
    };
    let actions = game.root.mainline();
    let search = match options.search {
        Search::Hypercuboid => "hypercuboid",
        Search::Naive => "naive",
    };
    emit(&format!(
        "{{\"event\":\"start\",\"driver\":\"kairos\",\"schema\":{SCHEMA},\"cap\":{},\"search\":\"{search}\",\"actions_total\":{}}}",
        options.cap,
        actions.len()
    ));
    report(&state, 0, &options);
    for (accepted, action) in actions.iter().enumerate() {
        match play_action(&state, action.moves()) {
            Ok(next) => state = next,
            Err(message) => {
                emit_end("rejected", accepted, refusal(&message), &message, Some(accepted + 1 == actions.len()));
                return;
            }
        }
        report(&state, accepted + 1, &options);
    }
    emit_end("ok", actions.len(), "", "", None);
}
