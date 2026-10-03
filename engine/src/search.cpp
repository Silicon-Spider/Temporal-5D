// Legal actions and mate detection: the public entry points, what both
// detectors share, and the naive detector (see t5d/search.hpp). Rule IDs refer
// to the rulebook, docs/rules/README.md. Written from the rulebook and the
// design spec; no other implementation was consulted.

#include "t5d/search.hpp"

#include <algorithm>
#include <deque>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "search_internal.hpp"
#include "t5d/check.hpp"
#include "t5d/legal.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"

namespace t5d {

namespace detail {

// ---- positions ----------------------------------------------------------------------

std::size_t position_bytes(const GameState& start) {
  const Multiverse& mv = start.multiverse();
  std::size_t playable = 0;
  for (const TimelineRef& t : mv.timelines()) {
    if (to_move_at(t->last_half_turn()) == start.to_move()) ++playable;
  }
  // Each allocation counted with 32 bytes of overhead; a store entry (hash
  // node, index, tag) as 128 bytes.
  const std::size_t table = mv.timelines().size() * sizeof(TimelineRef) + 32;
  const std::size_t changed = sizeof(Timeline) + sizeof(Board) + 4 * 32 + 128;  // tail chunk too
  return sizeof(GameState) + table + (playable + 1) * changed + 128;
}

std::pair<std::size_t, bool> PositionSet::insert(const GameState& s) {
  const std::uint64_t key = s.zobrist();
  const auto [it, fresh] = first_.try_emplace(key, states_.size());
  if (fresh) {
    states_.push_back(s);
    return {states_.size() - 1, true};
  }
  if (same_position(states_[it->second], s)) return {it->second, false};
  // A key collision (two positions, one 64-bit key): compared in full.
  std::vector<std::size_t>& others = more_[key];
  for (const std::size_t i : others) {
    if (same_position(states_[i], s)) return {i, false};
  }
  others.push_back(states_.size());
  states_.push_back(s);
  return {states_.size() - 1, true};
}

std::pair<std::size_t, bool> TaggedPositionSet::insert(const GameState& s, const std::string& tag) {
  const std::uint64_t key = s.zobrist() ^ (std::hash<std::string>{}(tag) * 0x9E3779B97F4A7C15ull);
  std::vector<std::size_t>& slot = index_[key];
  for (const std::size_t i : slot) {
    if (tags_[i] == tag && same_position(states_[i], s)) return {i, false};
  }
  slot.push_back(states_.size());
  states_.push_back(s);
  tags_.push_back(tag);
  return {states_.size() - 1, true};
}

// ---- results --------------------------------------------------------------------------

// action_text's order: the moves that create no timeline by source timeline,
// then the branching ones as made.
std::vector<const Move*> canonical_order(const std::vector<Move>& moves) {
  std::vector<const Move*> plain;
  std::vector<const Move*> branching;
  for (const Move& m : moves) (m.branching() ? branching : plain).push_back(&m);
  // T3: no two moves start on one board, so the source timeline orders them.
  std::stable_sort(plain.begin(), plain.end(), [](const Move* a, const Move* b) {
    return a->from_timeline < b->from_timeline;
  });
  plain.insert(plain.end(), branching.begin(), branching.end());
  return plain;
}

std::string action_key(const std::vector<Move>& moves) {
  std::string out;
  for (const Move* m : canonical_order(moves)) {
    if (!out.empty()) out += ' ';
    out += to_string(*m, false);
    if (m->branching()) out += '*';
  }
  return out;
}

bool SearchRun::record(const GameState& after, const std::vector<Move>& moves) {
  GameState successor = after;
  successor.submit(CheckRuling::none());  // legal: the caller tested C4
  const bool even = after.multiverse().even();
  const auto [index, fresh] = successors.insert(successor);
  if (unit == Unit::Actions) {
    // An action is its set of moves with its creations in order: counted once,
    // whatever order its other moves were found in.
    std::string key = action_key(moves);
    const std::size_t key_bytes = key.size() + 64;  // with the set's node
    if (!action_keys.insert(std::move(key)).second) return true;
    kept_bytes += key_bytes;
    if (keep_actions || actions.empty()) {
      actions.push_back(moves);
      texts.push_back(keep_actions ? action_text(moves, even) : std::string());
      results.push_back(index);
      kept_bytes += moves.size() * sizeof(Move) + texts.back().size() + 96;
    }
    if (action_keys.size() >= cap) {
      reached_cap = true;
      return false;
    }
    return true;
  }
  if (fresh) {
    if (keep_actions || actions.empty()) {
      actions.push_back(moves);
      texts.push_back(keep_actions ? action_text(moves, even) : std::string());
      results.push_back(index);
      kept_bytes += moves.size() * sizeof(Move) + texts.back().size() + 96;
    }
    if (successors.size() >= cap) {
      reached_cap = true;
      return false;
    }
  } else if (keep_actions) {
    std::string text = action_text(moves, even);
    if (text < texts[index]) {
      texts[index] = std::move(text);
      actions[index] = moves;
    }
  }
  return true;
}

// ---- the naive detector -------------------------------------------------------------------
//
// Depth-first over move sequences, as the state machine generates them: from
// the turn's start, every pseudo-legal move of the mover (generate_moves -
// which, under Q4's Guide-literal reading, offers nothing once the present has
// shifted; by default, R8, the mover goes on until submitting), applied to
// a copy, recursively. A state whose obligations are met (T5) is a finished
// action: legal if royals_safe (C4, R1). The pruning is a transposition table
// of every state reached: the same moves in another order reach the same state,
// which is searched once - a state determines everything that follows it.
// Counting actions, the table holds each state with the action so far - its
// moves as a set, its branching moves in order (action_key): two orders of the
// same moves are one entry, but two different sets of moves that reach one
// state (two identical pieces exchanging destinations), or two creation
// orders, are two - they are different actions.

class Naive {
 public:
  explicit Naive(SearchRun& run)
      : run_(run), budget_(run.budget, run.stats, position_bytes(run.start)) {}

  void search() {
    const bool complete = visit(run_.start);
    run_.stats.exhaustive = complete && !budget_.stopped();
    run_.stats.seconds = budget_.seconds();
  }

 private:
  // False: stop the search (a legal action found in a search for one, the
  // cap reached, or the budget spent).
  bool visit(const GameState& s) {
    if (s.obligations_met()) {
      if (!budget_.spend()) return false;
      ++run_.stats.actions_tested;
      if (royals_safe(s) && !run_.record(s, path_)) return false;
      if (!budget_.hold(seen_.size() + run_.held(), run_.held_bytes())) return false;
      if (!s.may_move()) return true;  // Q4's Guide-literal reading: the action ends here
    }
    const std::size_t depth = path_.size();
    if (lists_.size() <= depth) lists_.emplace_back();
    MoveList& moves = lists_[depth];  // a deque: stable while deeper levels grow it
    generate_moves(s, moves);
    for (const Move& m : moves) {
      if (!budget_.spend()) return false;
      GameState child = s;
      child.apply(m);
      path_.push_back(m);
      // Counting actions, the action so far is part of the entry (see above).
      const std::string tag = run_.unit == Unit::Actions ? action_key(path_) : std::string();
      if (!seen_.insert(child, tag).second) {  // reached before, by other orders
        path_.pop_back();
        continue;
      }
      if (!budget_.hold(seen_.size() + run_.held(), run_.held_bytes())) return false;
      const bool go_on = visit(child);
      path_.pop_back();
      if (!go_on) return false;
    }
    return true;
  }

  SearchRun& run_;
  Budgeter budget_;
  TaggedPositionSet seen_;
  std::vector<Move> path_;
  std::deque<MoveList> lists_;
};

void run_naive(SearchRun& run) { Naive(run).search(); }

}  // namespace detail

// ---- public -------------------------------------------------------------------------------

const char* detector_name(Detector detector) noexcept {
  return detector == Detector::Naive ? "naive" : "hypercuboid";
}

namespace {

void run(detail::SearchRun& r, Detector detector) {
  if (detector == Detector::Naive) {
    detail::run_naive(r);
  } else {
    detail::run_hypercuboid(r);
  }
}

}  // namespace

MateResult detect_mate(const GameState& turn_start, Detector detector,
                       const SearchBudget& budget) {
  MateResult result;
  result.in_check = in_check(turn_start);
  detail::SearchRun r(turn_start, budget, 1, detail::Unit::Successors, false);
  run(r, detector);
  result.stats = r.stats;
  if (r.successors.size() > 0) {
    result.verdict = Outcome::Ongoing;
    result.witness = r.actions.front();
  } else if (r.stats.exhaustive && !r.stats.stopped_by_budget) {
    // H37: a proof - every action was examined and none is legal (C5).
    result.verdict = result.in_check ? Outcome::Checkmate : Outcome::Stalemate;
  } else {
    result.verdict = Outcome::Unknown;
  }
  return result;
}

namespace {

ActionCount count_units(const GameState& turn_start, std::size_t cap, Detector detector,
                        const SearchBudget& budget, detail::Unit unit) {
  ActionCount out;
  if (cap == 0) {
    out.reached_cap = true;
    return out;
  }
  detail::SearchRun r(turn_start, budget, cap, unit, false);
  run(r, detector);
  out.count = r.found();
  out.reached_cap = r.reached_cap;
  out.stats = r.stats;
  return out;
}

}  // namespace

ActionCount count_legal_actions(const GameState& turn_start, std::size_t cap, Detector detector,
                                const SearchBudget& budget) {
  return count_units(turn_start, cap, detector, budget, detail::Unit::Actions);
}

ActionCount count_legal_successors(const GameState& turn_start, std::size_t cap,
                                   Detector detector, const SearchBudget& budget) {
  return count_units(turn_start, cap, detector, budget, detail::Unit::Successors);
}

ActionList list_legal_actions(const GameState& turn_start, std::size_t cap, Detector detector,
                              const SearchBudget& budget) {
  ActionList out;
  if (cap == 0) {
    out.reached_cap = true;
    return out;
  }
  detail::SearchRun r(turn_start, budget, cap, detail::Unit::Actions, true);
  run(r, detector);
  std::vector<std::size_t> order(r.actions.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return r.texts[a] < r.texts[b]; });
  out.actions.reserve(order.size());
  for (const std::size_t i : order) {
    out.actions.push_back(LegalAction{r.actions[i], r.successors[r.results[i]]});
  }
  out.reached_cap = r.reached_cap;
  out.stats = r.stats;
  return out;
}

std::optional<bool> can_finish_legally(const GameState& state, Detector detector,
                                       const SearchBudget& budget) {
  if (state.obligations_met()) {
    if (royals_safe(state)) return true;  // submit now
    if (!state.may_move()) return false;  // Q4's Guide-literal reading: nothing more to play
  }
  detail::SearchRun r(state, budget, 1, detail::Unit::Successors, false);
  run(r, detector);
  if (r.found() > 0) return true;
  if (r.stats.exhaustive) return false;
  return std::nullopt;
}

void keep_legal_moves(const GameState& state, MoveList& moves, Detector detector) {
  // The tiers of t5d/legal.hpp, exact: a move is dropped only when proved
  // illegal (one left unknown - the memory ceiling - is kept).
  LegalityOptions options;
  options.detector = detector;
  std::vector<MoveLegality> status;
  classify_moves(state, moves, status, options);
  std::size_t kept = 0;
  for (std::size_t i = 0; i < moves.size(); ++i) {
    if (status[i] != MoveLegality::Illegal) moves[kept++] = moves[i];
  }
  moves.resize(kept);
}

void legal_moves(const GameState& state, MoveList& out, Detector detector) {
  generate_moves(state, out);
  keep_legal_moves(state, out, detector);
}

std::string action_text(const std::vector<Move>& moves, bool even) {
  std::string out;
  for (const Move* m : detail::canonical_order(moves)) {
    if (!out.empty()) out += ' ';
    out += to_string(*m, even);
  }
  return out;
}

bool same_position(const GameState& a, const GameState& b) noexcept {
  if (!(a.rules() == b.rules()) || a.inactive_timelines() != b.inactive_timelines() ||
      a.deferred_timelines() != b.deferred_timelines()) {
    return false;
  }
  // The last creator is read by Q1's Consecutive reading alone (t5d/state.hpp).
  if (a.rules().activation == ActivationRule::Consecutive && a.last_creator() != b.last_creator()) {
    return false;
  }
  const Multiverse& x = a.multiverse();
  const Multiverse& y = b.multiverse();
  if (!(x.board_size() == y.board_size()) || x.even() != y.even() || x.ruleset() != y.ruleset() ||
      x.to_move() != y.to_move() || x.timeline_count() != y.timeline_count()) {
    return false;
  }
  if (x.empty()) return true;
  if (x.min_timeline() != y.min_timeline()) return false;
  for (std::size_t i = 0; i < x.timelines().size(); ++i) {
    const Timeline& s = *x.timelines()[i];
    const Timeline& t = *y.timelines()[i];
    if (&s == &t) continue;  // shared: equal
    // Everything but origin(), which no rule reads.
    if (s.first_half_turn() != t.first_half_turn() || s.creator() != t.creator() ||
        s.creation_order() != t.creation_order() || s.length() != t.length()) {
      return false;
    }
    for (std::size_t k = 0; k < s.boards().size(); ++k) {
      const BoardRef& p = s.boards()[k];
      const BoardRef& q = t.boards()[k];
      if (p != q && !(*p == *q)) return false;
    }
  }
  return true;
}

}  // namespace t5d
