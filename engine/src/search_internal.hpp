// What the two detectors share (src/search.cpp, src/hypercuboid.cpp): the
// budget's accounting, a set of positions keyed by the Zobrist key, and the
// collection of legal successor states. Internal to the engine.

#ifndef T5D_SEARCH_INTERNAL_HPP
#define T5D_SEARCH_INTERNAL_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"

namespace t5d::detail {

/// Spends a SearchBudget. Every step goes through spend(); once a limit is
/// reached it says no, forever, and the search must stop - its verdict is
/// then Unknown (H37).
class Budgeter {
 public:
  /// `position_bytes`: what one held position costs, by estimate
  /// (position_bytes() below).
  Budgeter(const SearchBudget& budget, SearchStats& stats, std::size_t position_bytes)
      : budget_(budget),
        stats_(stats),
        start_(std::chrono::steady_clock::now()),
        most_held_(budget.max_bytes / (position_bytes == 0 ? 1 : position_bytes)) {}

  /// One step. False if the budget does not allow it (nothing is spent).
  bool spend() {
    if (stats_.stopped_by_budget) return false;
    if (stats_.nodes >= budget_.max_nodes) return stop();
    if (budget_.max_time && (stats_.nodes & 63u) == 0 &&
        std::chrono::steady_clock::now() - start_ > *budget_.max_time) {
      return stop();
    }
    ++stats_.nodes;
    return true;
  }
  /// The search now holds `held` positions. False if that is over a limit:
  /// max_stored, or the memory ceiling (max_bytes).
  bool hold(std::size_t held) {
    if (held > stats_.stored) stats_.stored = held;
    if (held > budget_.max_stored) return stop();
    if (held > most_held_) {
      stats_.stopped_by_memory = true;
      return stop();
    }
    return true;
  }
  bool stopped() const noexcept { return stats_.stopped_by_budget; }
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  }

 private:
  bool stop() {
    stats_.stopped_by_budget = true;
    return false;
  }
  const SearchBudget& budget_;
  SearchStats& stats_;
  std::chrono::steady_clock::time_point start_;
  std::size_t most_held_;  // the positions the memory ceiling allows
};

/// An estimate of the bytes one position held by a search of `start`'s
/// action costs: the state with its own timeline table and, for each board the
/// action may change (one per playable board of the mover's), a timeline, a
/// board and their bookkeeping; and the store's own entry. Boards and
/// history are shared between states (t5d/multiverse.hpp), so it does not
/// grow with the history. Generous: allocator overhead included.
std::size_t position_bytes(const GameState& start);

/// A set of positions: the Zobrist key, then same_position() on a key match.
class PositionSet {
 public:
  /// Adds `s`; returns its index and whether it was new.
  std::pair<std::size_t, bool> insert(const GameState& s);
  std::size_t size() const noexcept { return states_.size(); }
  const GameState& operator[](std::size_t i) const noexcept { return states_[i]; }
  void reserve(std::size_t n) { states_.reserve(n); }

 private:
  std::unordered_map<std::uint64_t, std::size_t> first_;
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> more_;  // key collisions
  std::vector<GameState> states_;
};

/// A set of (position, tag) pairs: a search table that tells apart one
/// position reached by different sequences of creations (the tag).
class TaggedPositionSet {
 public:
  std::pair<std::size_t, bool> insert(const GameState& s, const std::string& tag);
  std::size_t size() const noexcept { return states_.size(); }

 private:
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> index_;
  std::vector<GameState> states_;
  std::vector<std::string> tags_;
};

/// The moves of an action in action_text's order: those that create no
/// timeline by ascending source timeline, then the branching ones as made.
std::vector<const Move*> canonical_order(const std::vector<Move>& moves);

/// What a count counts (t5d/search.hpp): legal actions - a set of moves with
/// its timeline-creating moves in creation order - or distinct successor
/// states.
enum class Unit : std::uint8_t { Actions, Successors };

/// An action's identity: its moves in canonical order (action_text's order),
/// each with its branching flag - so a jump and its J6 branching twin differ.
std::string action_key(const std::vector<Move>& moves);

/// One search's task and its results, for either detector.
struct SearchRun {
  SearchRun(const GameState& start_state, const SearchBudget& search_budget, std::size_t max,
            Unit counted, bool keep)
      : start(start_state), budget(search_budget), cap(max), unit(counted), keep_actions(keep) {}

  const GameState& start;
  const SearchBudget budget;  // a copy: callers pass temporaries
  /// Stop once this many distinct units are found (1: mate detection).
  std::size_t cap;
  Unit unit;
  /// Keep the actions found: for Actions, every one; for Successors, for
  /// each state the action with the least canonical text.
  bool keep_actions;

  SearchStats stats;
  /// The successors found (after submission); the distinct actions found
  /// (counted under Unit::Actions only).
  PositionSet successors;
  std::unordered_set<std::string> action_keys;
  /// Kept actions (all of them if keep_actions; else the first), their
  /// canonical texts and their successors' indices into `successors`.
  std::vector<std::vector<Move>> actions;
  std::vector<std::string> texts;
  std::vector<std::size_t> results;
  bool reached_cap = false;

  /// How many units were found.
  std::size_t found() const noexcept {
    return unit == Unit::Actions ? action_keys.size() : successors.size();
  }
  /// Positions and keys held (for the memory budget).
  std::size_t held() const noexcept { return successors.size() + action_keys.size(); }

  /// Records the legal action `moves`, which reached `after` (before
  /// submission). Returns false once the cap is reached (the search stops).
  bool record(const GameState& after, const std::vector<Move>& moves);
};

/// The detectors: each fills run.stats (exhaustive, stopped_by_budget, ...),
/// run.successors and run.reached_cap.
void run_naive(SearchRun& run);
void run_hypercuboid(SearchRun& run);

}  // namespace t5d::detail

#endif  // T5D_SEARCH_INTERNAL_HPP
