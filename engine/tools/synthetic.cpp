// SYNTHETIC positions by seeded random pseudo-legal play (see synthetic.hpp).
// Throughput and stress only - never the |E| verdict.

#include "synthetic.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/timeline.hpp"
#include "t5d/types.hpp"
#include "t5d/variants.hpp"

namespace t5d::synthetic {

namespace {

// The play is PSEUDO-LEGAL: submission tests the present's shift (T5) and not
// royal safety (C4) - a ruling with no hook, which stage E4 names
// CheckRuling::none() (its submit() tests royal safety by default).
constexpr CheckRuling kPseudoLegal = CheckRuling::none();

int active_count(const GameState& s) {
  return s.multiverse().timeline_count() - static_cast<int>(s.inactive_timelines().size());
}

std::string board_text(int timeline, int half_turn, bool even) {
  return "(" + timeline_label_string(timeline, even) + "T" + std::to_string(turn_of(half_turn)) +
         ")";
}

const Variant& variant_of(const Spec& spec) {
  const Variant* v = find_variant(spec.variant);
  if (v == nullptr) {
    throw std::invalid_argument("synthetic: the engine sets up no variant named \"" +
                                spec.variant + "\"");
  }
  return *v;
}

// Plays one action from `s` (an action's start). Returns false if stuck: no
// move left while the action cannot be submitted.
bool play_action(GameState& s, const Spec& spec, Rng& rng, MoveList& moves,
                 std::vector<Move>& played) {
  played.clear();
  std::vector<std::size_t> quiet;
  std::vector<std::size_t> branching;
  while (!s.obligations_met()) {
    generate_moves(s, moves);
    quiet.clear();
    branching.clear();
    for (std::size_t i = 0; i < moves.size(); ++i) {
      const Move& m = moves[i];
      if (m.captured.royal()) continue;  // 4.13: no royal piece is ever taken
      (m.branching() ? branching : quiet).push_back(i);
    }
    if (quiet.empty() && branching.empty()) return false;  // stuck
    std::size_t pick = 0;
    bool chosen = false;
    if (!branching.empty() &&
        (!spec.active_branches_only || active_count(s) < spec.target_active) &&
        rng.chance(spec.branch_percent)) {
      // Branch - if the new timeline would be active. Activity depends on
      // who has created how many timelines, not on the move (T10, T11), so one
      // trial on a copy answers for every branching move of the mover's now.
      GameState trial = s;
      trial.apply(moves[branching.front()]);
      if (!spec.active_branches_only || active_count(trial) > active_count(s)) {
        pick = branching[rng.below(branching.size())];
        chosen = true;
      }
    }
    if (!chosen) {
      // At or above the target, or not branching this move: a non-branching
      // move, unless there is none.
      const std::vector<std::size_t>& from = quiet.empty() ? branching : quiet;
      pick = from[rng.below(from.size())];
    }
    const Move m = moves[pick];
    s.apply(m);
    played.push_back(m);
  }
  s.submit(kPseudoLegal);
  return true;
}

}  // namespace

const char* stop_name(Stop stop) noexcept {
  switch (stop) {
    case Stop::Target: return "target";
    case Stop::Stuck: return "stuck";
    case Stop::MaxActions: return "max_actions";
  }
  return "?";
}

Game play(const Spec& spec) {
  const GameState start = variant_start(variant_of(spec), spec.rules);
  Game g{spec, spec.seed, 1, start, {}, start, {}, Stop::Target};
  std::vector<int> wanted = spec.snapshots;
  std::sort(wanted.begin(), wanted.end());
  std::size_t next_snapshot = 0;
  Rng rng(spec.seed);
  MoveList moves;
  std::vector<Move> played;
  GameState& s = g.state;
  for (;;) {
    // An action's start: keep the snapshots it reaches; stop at the target.
    const auto boards = static_cast<int>(s.multiverse().board_count());
    while (next_snapshot < wanted.size() && boards >= wanted[next_snapshot]) {
      g.snapshots.push_back(Snapshot{wanted[next_snapshot], g.actions.size(), s});
      ++next_snapshot;
    }
    if (boards >= spec.target_boards) {
      g.stop = Stop::Target;
      return g;
    }
    if (static_cast<int>(g.actions.size()) >= spec.max_actions) {
      g.stop = Stop::MaxActions;
      return g;
    }
    const GameState before = s;
    if (!play_action(s, spec, rng, moves, played)) {
      s = before;  // the stuck action is not part of the game
      g.stop = Stop::Stuck;
      return g;
    }
    g.actions.push_back(played);
  }
}

Game play_until_reached(const Spec& spec, int attempts) {
  Spec next = spec;
  for (int a = 1;; ++a) {
    Game g = play(next);
    g.attempts = a;
    const bool complete = g.stop == Stop::Target && g.snapshots.size() == spec.snapshots.size();
    if (complete || a >= attempts) return g;
    ++next.seed;
  }
}

std::string pgn_move(const Move& m, bool even) {
  std::string out = board_text(m.from_timeline, m.from_half_turn, even);
  out += letter_of(m.piece.kind());
  out += square_name(m.from_square);
  if (m.physical()) {
    if (m.capture()) out += 'x';
    out += square_name(m.to_square);
  } else {
    out += m.branching() ? ">>" : ">";
    if (m.capture()) out += 'x';
    out += board_text(m.to_timeline, m.to_half_turn, even);
    out += square_name(m.to_square);
  }
  if (m.promotes()) {
    out += '=';
    out += letter_of(m.promotion);
  }
  return out;
}

std::string record(const Game& game, std::size_t actions) {
  const Spec& spec = game.spec;
  const bool even = game.start.multiverse().even();
  const std::size_t count = std::min(actions, game.actions.size());
  std::string out;
  out += "[Mode \"5D\"]\n";
  out += "[Board \"" + std::string(variant_of(spec).name) + "\"]\n";
  out += "[Synthetic \"random pseudo-legal play by t5d's generator: for throughput and stress "
         "tests only, never for the |E| verdict\"]\n";
  out += "[Seed \"" + std::to_string(game.seed) + "\"]\n";
  out += "[Generator \"target_active=" + std::to_string(spec.target_active) +
         " branch_percent=" + std::to_string(spec.branch_percent) +
         " target_boards=" + std::to_string(spec.target_boards) + "\"]\n";
  if (spec.rules.encode() != 0) {
    // Informational: the reader keeps it and reads no rules from it, so the
    // record replays to the game only under the same ruleset (the tools'
    // --rules ENCODING).
    char encoding[24];
    std::snprintf(encoding, sizeof encoding, "0x%llx",
                  static_cast<unsigned long long>(spec.rules.encode()));
    out += "[Rules \"" + std::string(encoding) + ": " + describe(spec.rules) + "\"]\n";
  }
  out += "\n";
  Colour colour = game.start.to_move();
  int turn = 1;
  for (std::size_t a = 0; a < count; ++a) {
    if (colour == Colour::White) {
      if (a != 0) out += '\n';
      out += std::to_string(turn) + ".";
    } else {
      out += a == 0 ? std::to_string(turn) + "b." : std::string(" /");
    }
    for (const Move& m : game.actions[a]) {
      out += ' ';
      out += pgn_move(m, even);
    }
    if (colour == Colour::Black) ++turn;
    colour = opposite(colour);
  }
  out += "\n*\n";
  return out;
}

GameState replay(const GameState& start, const std::vector<std::vector<Move>>& actions,
                 std::size_t count) {
  // Rebuilt from the start position's boards: the start is shared, every
  // board made by a move is new.
  GameState s = start;
  const std::size_t n = std::min(count, actions.size());
  for (std::size_t a = 0; a < n; ++a) {
    for (const Move& m : actions[a]) s.apply(m);
    s.submit(kPseudoLegal);
  }
  return s;
}

}  // namespace t5d::synthetic
