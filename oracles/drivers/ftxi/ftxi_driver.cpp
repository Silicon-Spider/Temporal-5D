// Test-time driver for the pinned ftxi/5dchess_engine oracle.
//
// This file belongs to the oracle harness, never to the engine: it is compiled
// against the oracle's exported source tree and exists only so the gate can ask
// the oracle the questions its stock 5dtools CLI does not answer in one pass
// (notably the per-position check verdict). All rules logic is the oracle's own:
// its parser, its state machine, its searches. The glue below only walks the
// main line and prints what the oracle says.
//
// One question is asked here that the oracle's replay tools do not ask. They
// apply a recorded action and submit it without testing whether it leaves one
// of the mover's royals capturable, so they would "accept" an action that the
// oracle's own searches never generate. Before submitting, this driver applies
// the oracle's own test - find_checks(), the one its naive search prunes with -
// so that an accepted action means an action the oracle holds legal.
//
// stdin : one 5DPGN record, exact bytes.
// stdout: JSON Lines.
//   {"event":"start", ...}
//   then, for each position k:
//   {"event":"state","index":k,"size":"WxH","markers":true,
//    "boards":["<L>:<T>:<c>:<placement>", ...]}
//   {"event":"explain","index":k, ...}                       (--explain only)
//   {"event":"position","index":k,"side":"white|black","in_check":bool,
//    "legal_actions":n,"terminal":"none|checkmate|stalemate","size":"WxH",
//    "ms":float}
//   {"event":"actions","index":k,"count":n,"truncated":bool,"actions":[...]}
//                                                            (--actions k only)
//   and last:
//   {"event":"end","status":"ok|rejected|unsupported|error", ...}
//
// Position k is the state after k submitted actions of the main line. Every line
// is flushed as soon as it is known, so a caller that kills a slow run keeps the
// positions already answered.
//
// The state event (decision H30) is the position the oracle parsed: every board
// it holds - superseded and head boards, every timeline, both colours - as
// "<L>:<T>:<c>:<placement>". L is the timeline as the notation writes it, named
// from the oracle's own parity; T and c are the oracle's own turn and colour for
// the board (c is the side to move on it); the placement is the oracle's own
// 5DFEN pieces field (board::get_fen<true>), which writes "*" after each king,
// rook, pawn and brawn the oracle holds unmoved - the only pieces it tracks -
// hence "markers":true. It is printed before the position's check and count, and
// outside their timer, so a search cut short for time cannot lose it. The
// harness, not this driver, canonicalises and hashes it. The position event
// repeats the state event's board size (one function writes both) beside its
// timing: the harness keeps a driver's own fields on the observation and never
// compares them, and the driver check (fidelity.py) holds that size to the
// stock CLI's.
//
// The explain and actions events are diagnostics, never compared:
//   explain  "phantom_timelines": the timelines whose last board is the mover's,
//            which the oracle advances by a copy of that board before it looks
//            for a royal capture (check_position::for_phantom, the view behind
//            has_phantom_check); and, in check, "checks": every royal capture
//            the oracle finds in that view (check_position::checks), each as
//            {"from":{"L","T","c","square","piece"},"to":{...}}.
//   actions  the legal actions of the position, as the --policy search yields
//            them (the searches that count them), at most 5000; "count" is how
//            many are listed and "truncated" whether the search had more. Each
//            move is {"from":{"L","T","c","square"},"to":{...},"promotion"}, the
//            promotion being the piece the oracle promotes to when it plays the
//            move (its searches yield one action per promoting move, which it
//            plays with the record's default promotion piece), or null.
//
// -----------------------------------------------------------------------------
// apply_action(), mainline_action_count(), and in main() the set-up that parses
// the record and the main-line walk (both after read_position()) are adapted
// from tools/position_tools.cpp of ftxi/5dchess_engine, which is distributed
// under the following licence. The rest of this file is the project's own.
//
//   BSD 2-Clause License
//
//   Copyright (c) 2024-2026, ftxi
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions are met:
//
//   1. Redistributions of source code must retain the above copyright notice,
//      this list of conditions and the following disclaimer.
//
//   2. Redistributions in binary form must reproduce the above copyright notice,
//      this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//   POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------

#include <algorithm>
#include <bit>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "check_position.h"
#include "hypercuboid.h"
#include "pgnparser.h"
#include "search_tools.h"
#include "state.h"
#include "utils.h"

namespace
{
// 2: the state event (H30), and the explain and actions diagnostics.
constexpr int SCHEMA = 2;
// The most actions one actions event lists.
constexpr std::size_t ACTIONS_LISTED = 5000;

std::string json_escape(std::string_view text)
{
    std::string out;
    out.reserve(text.size() + 8);
    for(const char ch : text)
    {
        const auto byte = static_cast<unsigned char>(ch);
        switch(ch)
        {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if(byte < 0x20)
                {
                    static const char *digits = "0123456789abcdef";
                    out += "\\u00";
                    out += digits[byte >> 4];
                    out += digits[byte & 0x0f];
                }
                else
                {
                    out += ch;
                }
        }
    }
    return out;
}

struct replay_failure : std::runtime_error
{
    std::string reason;
    replay_failure(std::string reason, const std::string &detail)
        : std::runtime_error(detail), reason(std::move(reason)) {}
};

// Adapted from apply_action() in the oracle's tools/position_tools.cpp. Two
// changes: each failure carries a machine-readable reason, and an action that
// leaves one of the mover's royals capturable is refused (see the top of this
// file) instead of submitted.
void apply_action(state &current, const pgnparser_ast::actions &act)
{
    const bool mover = current.get_present().second;
    for(const auto &mv : act.moves)
    {
        auto [fm, promotion, candidates] = current.parse_move(mv);
        if(!fm)
        {
            std::ostringstream message;
            if(candidates.empty())
            {
                message << "Invalid move: " << mv;
                throw replay_failure("no_matching_move", message.str());
            }
            message << "Ambiguous move: " << mv << "; candidates: "
                    << range_to_string(candidates, "", "");
            throw replay_failure("ambiguous_move", message.str());
        }
        const bool applied = promotion
            ? current.apply_move(ext_move(*fm, to_white(*promotion)))
            : current.apply_move(*fm);
        if(!applied)
        {
            std::ostringstream message;
            message << "Illegal move: " << mv << " (parsed as: " << *fm << ')';
            throw replay_failure("illegal_move", message.str());
        }
    }
    // The oracle's own legality test for a whole action, as its naive search
    // applies it: no enemy capture of a royal on the boards just played.
    if(current.find_checks(!mover).first().has_value())
    {
        std::ostringstream message;
        message << "Leaves a royal capturable after these moves: " << act;
        throw replay_failure("cannot_submit", message.str());
    }
    if(!current.submit())
    {
        std::ostringstream message;
        message << "Cannot submit after parsing these moves: " << act;
        throw replay_failure("cannot_submit", message.str());
    }
}

std::size_t mainline_action_count(const pgnparser_ast::gametree &root)
{
    std::size_t actions = 0;
    const auto *node = &root;
    while(std::holds_alternative<pgnparser_ast::gametree::variations_t>(
        node->variations_or_outcome))
    {
        const auto &variations
            = std::get<pgnparser_ast::gametree::variations_t>(
                node->variations_or_outcome);
        if(variations.empty()) break;
        ++actions;
        node = variations.back().second.get();
    }
    return actions;
}

// The driver's own policy names, so a pin bump that adds or drops a search
// policy upstream does not change what the harness can ask for.
enum class policy_t { balanced, naive, stable, iterative };

std::optional<policy_t> parse_policy(std::string_view name)
{
    if(name == "balanced") return policy_t::balanced;
    if(name == "naive") return policy_t::naive;
    if(name == "stable") return policy_t::stable;
    if(name == "iterative") return policy_t::iterative;
    return std::nullopt;
}

std::size_t count_actions(const state &s, policy_t policy, int cap)
{
    switch(policy)
    {
        case policy_t::balanced: return count_balanced(s, cap);
        case policy_t::naive: return count_naive(s, cap);
        case policy_t::stable: return count_stable(s, cap);
        case policy_t::iterative: return count_iterative(s, cap);
    }
    throw std::logic_error("unknown search policy");
}

// Calls take(action) for each legal action the policy's search yields, in the
// order it yields them, until take returns false. These are the searches
// count_actions() runs, set up the way the oracle's counters set them up.
template<class Take>
void for_each_action(const state &s, policy_t policy, Take &&take)
{
    const auto walk = [&](generator<moveseq> found) {
        for(const moveseq &action : found)
        {
            if(!take(action)) return;
        }
    };
    if(policy == policy_t::naive)
    {
        walk(naive_search(s));
        return;
    }
    auto [info, space] = HC_info::build_HC(s);
    switch(policy)
    {
        case policy_t::balanced: walk(info.search(std::move(space))); return;
        case policy_t::stable: walk(info.stable_search(std::move(space))); return;
        case policy_t::iterative: walk(info.iterative_search(std::move(space))); return;
        case policy_t::naive: break;
    }
    throw std::logic_error("unknown search policy");
}

// --- The position the oracle holds --------------------------------------------

// "WxH": the size of the boards the oracle set up.
std::string board_size(const state &s)
{
    const auto [width, height] = s.get_board_size();
    return std::to_string(width) + "x" + std::to_string(height);
}

// A timeline as the notation writes it, named from the oracle's own parity. An
// even multiverse starts with the lines ftxi numbers -1 and 0, the notation's -0
// and +0: its negative lines are bitwise complements (-1 is -0, -2 is -1, ...).
// An odd one starts with line 0 alone.
std::string timeline_label(const state &s, int l)
{
    const auto initial = s.get_initial_lines_range();
    if(initial == std::make_pair(-1, 0))
        return (l >= 0 ? "+" : "-") + std::to_string(l >= 0 ? l : ~l);
    if(initial != std::make_pair(0, 0))
        throw std::logic_error("unknown timeline parity: initial lines "
                               + std::to_string(initial.first) + ".." + std::to_string(initial.second));
    if(l == 0) return "0";
    return (l > 0 ? "+" : "") + std::to_string(l);
}

// Calls visit(l, at, board) for every board the oracle holds - superseded and
// head boards alike, every timeline, both colours - walking each timeline from
// its first board to its last, as the oracle's own test_check_position.cpp
// does. Never outside that range: there the oracle's get_board() dumps the
// whole multiverse to stderr.
template<class Visit>
void for_each_board(const state &s, Visit &&visit)
{
    const auto [l_min, l_max] = s.get_lines_range();
    for(int l = l_min; l <= l_max; ++l)
    {
        const turn_t end = s.get_timeline_end(l);
        for(turn_t at = s.get_timeline_start(l); at <= end; at = next_turn(at))
        {
            const board *b = s.get_board_ptr(l, at.first, at.second);
            if(b == nullptr)
                throw std::logic_error("the oracle holds no board at L" + std::to_string(l)
                                       + " T" + std::to_string(at.first) + (at.second ? "b" : "w")
                                       + ", inside its timeline");
            visit(l, at, *b);
        }
    }
}

void emit_state(const state &s, std::size_t index)
{
    // Built whole before it is printed: a failure part-way leaves no half line.
    std::ostringstream line;
    line << "{\"event\":\"state\",\"index\":" << index
         << ",\"size\":\"" << board_size(s) << '"'
         << ",\"markers\":true,\"boards\":[";
    const char *separator = "";
    for_each_board(s, [&](int l, turn_t at, const board &b) {
        const std::string placement = b.get_fen<true>();
        // The pieces field is written square by square from the bitboards; one
        // dropped piece would change the fingerprint without a trace, so the
        // two are held to each other. A wall sets both colour bits and is no
        // piece, so the pieces are the squares with exactly one of them.
        const auto letters = std::count_if(placement.begin(), placement.end(), [](char ch) {
            return std::isalpha(static_cast<unsigned char>(ch)) != 0;
        });
        if(letters != std::popcount(b.white() ^ b.black()))
            throw std::logic_error("the pieces field '" + placement + "' of the board on line "
                                   + std::to_string(l) + " does not list the board's pieces");
        line << separator << '"' << timeline_label(s, l) << ':' << at.first << ':'
             << (at.second ? 'b' : 'w') << ':' << json_escape(placement) << '"';
        separator = ",";
    });
    line << "]}\n";
    std::cout << line.str() << std::flush;
}

// --- Diagnostics (--explain, --actions): printed on request, never compared ---

// One end of a move: the board, as in the state event, and the square on it.
void write_square(std::ostream &out, const state &s, vec4 p, bool c)
{
    out << "{\"L\":\"" << timeline_label(s, p.l()) << "\",\"T\":" << p.t()
        << ",\"c\":\"" << (c ? 'b' : 'w') << "\",\"square\":\""
        << static_cast<char>('a' + p.x()) << p.y() + 1 << '"';
}

void write_piece(std::ostream &out, const board *b, vec4 p)
{
    const piece_t piece = b ? b->get_piece(p.xy()) : NO_PIECE;
    out << ",\"piece\":";
    if(piece == NO_PIECE || piece == WALL_PIECE) out << "null";
    else out << '"' << static_cast<char>(piece_name(piece)) << '"';
}

void emit_explain(const state &s, std::size_t index)
{
    const bool attacker = !s.get_present().second;
    // The view has_phantom_check() tests, built the same way.
    const auto view = check_position::for_phantom(s);
    std::ostringstream line;
    line << "{\"event\":\"explain\",\"index\":" << index << ",\"phantom_timelines\":[";
    const auto [l_min, l_max] = s.get_lines_range();
    const char *separator = "";
    for(int l = l_min; l <= l_max; ++l)
    {
        // Read off the view itself: a timeline was advanced exactly when the
        // view holds a board past the timeline's last one.
        const turn_t after = next_turn(s.get_timeline_end(l));
        if(view.board_at(l, after.first, after.second) != nullptr)
        {
            line << separator << '"' << timeline_label(s, l) << '"';
            separator = ",";
        }
    }
    line << ']';
    const auto checks = view.checks(attacker);
    if(!checks.empty())
    {
        line << ",\"checks\":[";
        separator = "";
        for(const full_move &capture : checks)
        {
            line << separator << "{\"from\":";
            write_square(line, s, capture.from, attacker);
            write_piece(line, view.board_at(capture.from.l(), capture.from.t(), attacker), capture.from);
            line << "},\"to\":";
            write_square(line, s, capture.to, attacker);
            write_piece(line, view.board_at(capture.to.l(), capture.to.t(), attacker), capture.to);
            line << "}}";
            separator = ",";
        }
        line << ']';
    }
    line << "}\n";
    std::cout << line.str() << std::flush;
}

void emit_actions(const state &s, std::size_t index, policy_t policy)
{
    const bool mover = s.get_present().second;
    std::ostringstream listed;
    std::size_t count = 0;
    bool truncated = false;
    for_each_action(s, policy, [&](const moveseq &action) {
        if(count == ACTIONS_LISTED)
        {
            truncated = true;
            return false;
        }
        listed << (count ? ",[" : "[");
        const char *separator = "";
        for(const full_move &mv : action)
        {
            listed << separator << "{\"from\":";
            write_square(listed, s, mv.from, mover);
            listed << "},\"to\":";
            write_square(listed, s, mv.to, mover);
            // The piece the oracle promotes to when it plays this move: its
            // searches yield the move alone, and playing it resolves the
            // record's default promotion piece (state::normalize_promotion).
            const auto played = s.normalize_promotion(ext_move(mv, NO_PIECE));
            listed << "},\"promotion\":";
            if(played && played->promote_to != NO_PIECE)
                listed << '"' << static_cast<char>(to_white(piece_name(played->promote_to))) << '"';
            else
                listed << "null";
            listed << '}';
            separator = ",";
        }
        listed << ']';
        ++count;
        return true;
    });
    std::cout << "{\"event\":\"actions\",\"index\":" << index << ",\"count\":" << count
              << ",\"truncated\":" << (truncated ? "true" : "false")
              << ",\"actions\":[" << listed.str() << "]}\n" << std::flush;
}

void emit_position(const state &s, std::size_t index, policy_t policy, int cap)
{
    const auto started = std::chrono::steady_clock::now();
    const bool black = s.get_present().second;
    const bool in_check = s.has_phantom_check();
    const std::size_t legal = count_actions(s, policy, cap);
    const char *terminal = legal > 0 ? "none" : (in_check ? "checkmate" : "stalemate");
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "{\"event\":\"position\",\"index\":" << index
              << ",\"side\":\"" << (black ? "black" : "white") << '"'
              << ",\"in_check\":" << (in_check ? "true" : "false")
              << ",\"legal_actions\":" << legal
              << ",\"terminal\":\"" << terminal << '"'
              << ",\"size\":\"" << board_size(s) << '"'
              << ",\"ms\":" << ms << "}\n" << std::flush;
}

void emit_end(std::string_view status, std::size_t accepted,
              std::string_view reason = {}, std::string_view detail = {},
              std::optional<bool> last_action = std::nullopt)
{
    std::cout << "{\"event\":\"end\",\"status\":\"" << status << '"'
              << ",\"actions_accepted\":" << accepted;
    if(!reason.empty()) std::cout << ",\"reason\":\"" << json_escape(reason) << '"';
    if(!detail.empty()) std::cout << ",\"detail\":\"" << json_escape(detail) << '"';
    if(last_action) std::cout << ",\"last_action\":" << (*last_action ? "true" : "false");
    std::cout << "}\n" << std::flush;
}

int usage(std::ostream &out, int code)
{
    out << "usage: ftxi_oracle_driver [--cap N] [--policy balanced|naive|stable|iterative]\n"
        << "                          [--explain] [--actions K]\n"
        << "  Reads a 5DPGN record from stdin; prints one JSON object per line.\n"
        << "  --explain    after each state event, what the check test looked at\n"
        << "  --actions K  at position K, list up to " << ACTIONS_LISTED << " legal actions\n";
    return code;
}

struct options
{
    int cap = 1000;
    policy_t policy = policy_t::balanced;
    bool explain = false;
    std::optional<std::size_t> actions_at;
};

// Everything printed about one position, in protocol order. The state event
// (and the explanation) go out before any search, outside the position's timer.
void report(const state &s, std::size_t index, const options &opts)
{
    emit_state(s, index);
    if(opts.explain) emit_explain(s, index);
    emit_position(s, index, opts.policy, opts.cap);
    if(opts.actions_at == index) emit_actions(s, index, opts.policy);
}
}

int main(int argc, const char *argv[])
{
    options opts;
    std::string policy_name = "balanced";
    for(int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if(arg == "-h" || arg == "--help") return usage(std::cout, 0);
        if(arg == "--cap" && i + 1 < argc)
        {
            opts.cap = std::atoi(argv[++i]);
            if(opts.cap <= 0) return usage(std::cerr, 2);
        }
        else if(arg == "--policy" && i + 1 < argc)
        {
            policy_name = argv[++i];
            const auto parsed = parse_policy(policy_name);
            if(!parsed) return usage(std::cerr, 2);
            opts.policy = *parsed;
        }
        else if(arg == "--explain")
        {
            opts.explain = true;
        }
        else if(arg == "--actions" && i + 1 < argc)
        {
            const std::string_view value = argv[++i];
            std::size_t at = 0;
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), at);
            if(value.empty() || error != std::errc{} || end != value.data() + value.size())
                return usage(std::cerr, 2);
            opts.actions_at = at;
        }
        else return usage(std::cerr, 2);
    }

    std::ostringstream input;
    input << std::cin.rdbuf();
    const std::string pgn = input.str();

    std::unique_ptr<state> current;
    pgnparser_ast::gametree mainline;
    std::size_t total = 0;
    try
    {
        auto parsed = pgnparser(pgn).parse_game();
        if(!parsed) throw parse_error("no game found");
        pgnparser_ast::game game = std::move(*parsed);
        total = mainline_action_count(game.gt);
        mainline = std::move(game.gt);
        game.gt = pgnparser_ast::gametree{};
        current = std::make_unique<state>(game);
    }
    catch(const parse_error &error)
    {
        emit_end("unsupported", 0, "parse_error", error.what());
        return 0;
    }
    catch(const std::exception &error)
    {
        emit_end("unsupported", 0, "setup", error.what());
        return 0;
    }

    std::cout << "{\"event\":\"start\",\"driver\":\"ftxi\",\"schema\":" << SCHEMA
              << ",\"cap\":" << opts.cap << ",\"policy\":\"" << policy_name << '"'
              << ",\"actions_total\":" << total << "}\n" << std::flush;

    std::size_t accepted = 0;
    try
    {
        report(*current, 0, opts);
        const auto *node = &mainline;
        while(std::holds_alternative<pgnparser_ast::gametree::variations_t>(
            node->variations_or_outcome))
        {
            const auto &variations
                = std::get<pgnparser_ast::gametree::variations_t>(
                    node->variations_or_outcome);
            if(variations.empty()) break;
            const auto &[action, next] = variations.back();
            try
            {
                apply_action(*current, action);
            }
            catch(const replay_failure &failure)
            {
                emit_end("rejected", accepted, failure.reason, failure.what(),
                         accepted + 1 == total);
                return 0;
            }
            ++accepted;
            report(*current, accepted, opts);
            node = next.get();
        }
    }
    catch(const std::exception &error)
    {
        emit_end("error", accepted, "exception", error.what());
        return 0;
    }
    emit_end("ok", accepted);
    return 0;
}
