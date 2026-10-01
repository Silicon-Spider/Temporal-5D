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
//   {"event":"position","index":k,"side":"white|black","in_check":bool,
//    "legal_actions":n,"terminal":"none|checkmate|stalemate","ms":float}
//   {"event":"end","status":"ok|rejected|unsupported|error", ...}
//
// Position k is the state after k submitted actions of the main line. Every line
// is flushed as soon as it is known, so a caller that kills a slow run keeps the
// positions already answered.
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

#include "hypercuboid.h"
#include "pgnparser.h"
#include "search_tools.h"
#include "state.h"
#include "utils.h"

namespace
{
constexpr int SCHEMA = 1;

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
        << "  Reads a 5DPGN record from stdin; prints one JSON object per line.\n";
    return code;
}
}

int main(int argc, const char *argv[])
{
    int cap = 1000;
    policy_t policy = policy_t::balanced;
    std::string policy_name = "balanced";
    for(int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if(arg == "-h" || arg == "--help") return usage(std::cout, 0);
        if(arg == "--cap" && i + 1 < argc)
        {
            cap = std::atoi(argv[++i]);
            if(cap <= 0) return usage(std::cerr, 2);
        }
        else if(arg == "--policy" && i + 1 < argc)
        {
            policy_name = argv[++i];
            const auto parsed = parse_policy(policy_name);
            if(!parsed) return usage(std::cerr, 2);
            policy = *parsed;
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
              << ",\"cap\":" << cap << ",\"policy\":\"" << policy_name << '"'
              << ",\"actions_total\":" << total << "}\n" << std::flush;

    std::size_t accepted = 0;
    try
    {
        emit_position(*current, 0, policy, cap);
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
            emit_position(*current, accepted, policy, cap);
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
