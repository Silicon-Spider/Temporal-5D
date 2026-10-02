// Games of the engine against itself (t5d/selfplay.hpp).

#include "t5d/selfplay.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "t5d/legal.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"

namespace t5d::mcts {

namespace {

std::uint64_t mix(std::uint64_t z) noexcept {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Plays `m`, checking that it is a move of the player to move now.
void play_checked(GameState& s, const Move& m, MoveList& scratch) {
  if (!s.may_move() || !is_pseudo_legal(s, m, scratch)) {
    throw std::logic_error("play: the search chose " + to_string(m, s.multiverse().even()) +
                           ", which is not a pseudo-legal move now");
  }
  s.apply(m);
}

// Submits the action, checking that the submission is legal (T5, C4/R1).
void submit_checked(GameState& s) {
  if (s.submission_status() != SubmitStatus::Safe) {
    throw std::logic_error("play: the search chose an action that cannot be submitted legally");
  }
  s.submit();
}

}  // namespace

const char* game_end_name(GameEnd end) noexcept {
  switch (end) {
    case GameEnd::Checkmate: return "checkmate";
    case GameEnd::Stalemate: return "stalemate";
    case GameEnd::TurnCap: return "turn_cap";
    case GameEnd::Unknown: return "unknown";
    case GameEnd::ActionLimit: return "action_limit";
  }
  return "?";
}

std::string GameRecord::result() const {
  if (end == GameEnd::Checkmate) return winner == 0 ? "1-0" : "0-1";
  if (end == GameEnd::Stalemate) return "1/2-1/2";
  return "*";
}

double GameRecord::white_score() const noexcept {
  switch (end) {
    case GameEnd::Checkmate: return winner == 0 ? 1.0 : 0.0;
    case GameEnd::Stalemate:
    case GameEnd::TurnCap: return 0.5;
    case GameEnd::Unknown:
    case GameEnd::ActionLimit: return -1.0;
  }
  return -1.0;
}

GameRecord play_game(const GameState& start, Evaluator& evaluator, const PlayParams& params) {
  const auto t0 = std::chrono::steady_clock::now();
  GameRecord g(start);
  SearchParams sp = params.search;
  sp.turn_cap = params.turn_cap;
  if (params.match_play) sp.root_noise = false;  // §12.3: noise off at match play
  std::shared_ptr<EvalCache> cache;
  if (sp.eval_cache_entries > 0) cache = std::make_shared<EvalCache>(sp.eval_cache_entries, sp.eval_cache_bytes);
  std::uint64_t rng = mix(params.seed ^ 0x73656C66706C6179ull);
  const auto uniform = [&rng] {
    rng += 0x9E3779B97F4A7C15ull;
    return static_cast<double>(mix(rng) >> 11) * 0x1.0p-53;
  };
  std::unique_ptr<Search> carried;
  GameState s = start;
  MoveList scratch;
  SearchBudget verdict_budget;  // no node or time limit: a verdict is a proof (H37)
  verdict_budget.max_bytes = sp.mate_bytes;
  for (;;) {
    // A turn start: the rules first, then the cap's convention.
    const auto v0 = std::chrono::steady_clock::now();
    const MateResult r = detect_mate(s, sp.detector, verdict_budget);
    g.seconds_verdicts += std::chrono::duration<double>(std::chrono::steady_clock::now() - v0).count();
    if (r.verdict == Outcome::Checkmate) {
      g.end = GameEnd::Checkmate;
      g.winner = index_of(opposite(s.to_move()));
      break;
    }
    if (r.verdict == Outcome::Stalemate) {
      g.end = GameEnd::Stalemate;
      break;
    }
    if (r.verdict == Outcome::Unknown) {
      g.end = GameEnd::Unknown;
      break;
    }
    if (params.turn_cap.reached(s)) {
      g.end = GameEnd::TurnCap;
      break;
    }
    if (g.actions.size() >= params.max_actions) {
      g.end = GameEnd::ActionLimit;
      break;
    }
    std::vector<Move> action;
    bool turn_start = true;
    for (bool submitted = false; !submitted;) {
      std::unique_ptr<Search> fresh;
      Search* search = nullptr;
      if (params.match_play) {
        if (!carried) carried = std::make_unique<Search>(s, sp, turn_start, cache);
        search = carried.get();
        if (!(search->root_state() == s)) {
          throw std::logic_error("play: the carried search's root is not the game's position");
        }
      } else {
        SearchParams dp = sp;
        dp.seed = mix(sp.seed + 0x9E3779B97F4A7C15ull * (g.decisions + 1));
        fresh = std::make_unique<Search>(s, dp, turn_start, cache);  // C14: fresh per move
        search = fresh.get();
      }
      const SearchStats before = search->stats();
      search->run(evaluator, Limits{params.playouts});
      const auto choose_now = [&]() -> std::size_t {
        bool visited = false;
        for (const RootChild& c : search->root_children()) visited = visited || c.visits > 0;
        if (!visited) search->run(evaluator, Limits{params.playouts});  // children were removed
        if (params.match_play) return search->choose().index;
        const double t = g.decisions < params.temperature_decisions ? params.temperature : 0.0;
        return search->choose_by_visits(t, uniform());
      };
      std::size_t pick = choose_now();
      // A move is played only once known legal: one the expansion's budget left
      // unknown is classified exactly first - illegal, it is removed and the
      // choice made again; still unknown (the memory ceiling), the most visited
      // child known legal is played instead.
      LegalityOptions exact = LegalityOptions::exact();
      exact.max_bytes = sp.mate_bytes;
      std::vector<RootChild> children = search->root_children();
      while (children.at(pick).legality != MoveLegality::Legal) {
        const MoveLegality v = search->verify(pick, exact);
        children = search->root_children();
        if (v == MoveLegality::Legal) break;
        if (v == MoveLegality::Illegal) {
          pick = choose_now();
          children = search->root_children();
          continue;
        }
        std::size_t best = children.size();
        for (std::size_t i = 0; i < children.size(); ++i) {
          if (children[i].legality == MoveLegality::Legal &&
              (best == children.size() || children[i].visits > children[best].visits)) {
            best = i;
          }
        }
        if (best == children.size()) {
          throw std::logic_error("play: no root child could be shown legal (the memory ceiling)");
        }
        pick = best;
      }
      const SearchStats& after = search->stats();
      g.playouts += after.playouts - before.playouts;
      g.evaluations += after.evaluations - before.evaluations;
      g.nodes += after.nodes_created - before.nodes_created;
      g.search_turn_caps += after.turn_caps - before.turn_caps;
      g.seconds_search_mate += after.seconds_mate - before.seconds_mate;
      g.seconds_search_expand += after.seconds_expand - before.seconds_expand;
      g.seconds_search_listing += after.seconds_listing - before.seconds_listing;
      g.seconds_search_verify += after.seconds_verify - before.seconds_verify;
      ++g.decisions;
      const RootChild& c = children.at(pick);
      if (sp.split == TurnSplit::WholeTurn) {
        for (const Move& m : c.moves) {
          play_checked(s, m, scratch);
          action.push_back(m);
        }
        submit_checked(s);
        submitted = true;
      } else if (c.submit) {
        submit_checked(s);
        submitted = true;
      } else {
        const Move& m = c.moves.at(0);
        play_checked(s, m, scratch);
        action.push_back(m);
        if (must_submit(s)) {
          submit_checked(s);
          submitted = true;
        }
      }
      if (params.match_play) search->advance(pick);
      turn_start = false;
    }
    g.actions.push_back(std::move(action));
  }
  g.final_state = s;
  g.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return g;
}

}  // namespace t5d::mcts
