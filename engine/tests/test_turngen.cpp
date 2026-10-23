// D2's flat form's turn generator (t5d/turngen.hpp), against spec §15.2's
// row: "Flat form's generator = an exhaustive listing of legal turns, in
// non-increasing summed-logit order, on small positions | Best-first
// generation is complete and ordered | D2".
//
// On small positions - random tiny multiverses under random rulesets (every
// regime: Ordered, FreeOrder, General; inactive timelines, reactivations,
// several creations, J6 twins), constructed multi-board presents (jumps
// between present boards, rewinding branches), and random games from every
// named variant's start (Very Small, Small and Two Timelines among them) -
// under several score assignments (continuous, and small integers that tie
// often), the generator's whole output must be:
//  (i)   the legal turns exactly, each once - the brute-force reference's
//        (test_reference.hpp) on the tiny positions and the engine's own
//        exhaustive listing (E4: list_legal_actions, the naive detector)
//        elsewhere - as actions and as distinct successor states;
//  (ii)  in non-increasing score;
//  (iii) each score the sum of its options' scores, recomputed here from its
//        moves alone, bit for bit;
//  (iv)  the sorted scores of every legal turn (so equal-score runs match as
//        multisets);
// and (v) the same on a second run; (vi) a generator its budget stopped is
// stopped(), never exhausted(), and has emitted a prefix of the full run.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "t5d/turngen.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"
#include "test_reference.hpp"
#include "test_support.hpp"
#include "test_tiny.hpp"

using namespace t5d;

namespace {

std::uint64_t mix(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Scores as a function of (seed, the option's coordinates, the board): a
// move by its source, destination and promotion (its branching flag ignored:
// a J6 twin scores as its jump), on the board it is scored on; leaving a
// board by its timeline. `levels` > 0: integers in [-levels, 0] (ties
// everywhere); 0: continuous, at a realistic logit scale.
struct HashScorer : TurnScorer {
  std::uint64_t seed;
  int levels;
  HashScorer(std::uint64_t s, int l) : seed(s), levels(l) {}
  double value(std::uint64_t key) const {
    const std::uint64_t h = mix(seed ^ mix(key));
    if (levels > 0) return -static_cast<double>(h % static_cast<std::uint64_t>(levels + 1));
    return (static_cast<double>(h >> 11) * 0x1.0p-53 - 0.5) * 6.0;
  }
  double leave(int timeline) const override {
    return value(0x6C65617665ull ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(timeline)) << 20));
  }
  double move(const Move& m, int timeline) const override {
    std::uint64_t k = static_cast<std::uint32_t>(m.from_timeline);
    for (const int v : {static_cast<int>(m.from_half_turn), static_cast<int>(m.from_square),
                        static_cast<int>(m.to_timeline), static_cast<int>(m.to_half_turn),
                        static_cast<int>(m.to_square), m.promotes() ? static_cast<int>(index_of(m.promotion)) + 1 : 0,
                        timeline}) {
      k = mix(k ^ static_cast<std::uint32_t>(v));
    }
    return value(k);
  }
};

// A turn's score recomputed from its moves alone: over the mover's playable
// boards in ascending timeline, the option it takes there - the move that
// leaves the board, or a non-branching jump arriving on it, or leaving it.
double recompute(const GameState& s, const std::vector<Move>& moves, const TurnScorer& scorer) {
  double sum = 0;
  for (const int t : s.playable_timelines()) {
    const Move* option = nullptr;
    bool arrival = false;
    for (const Move& m : moves) {
      if (m.from_timeline == t) option = &m;
    }
    if (option == nullptr) {
      for (const Move& m : moves) {
        if (m.jump() && !m.branching() && m.to_timeline == t) {
          option = &m;
          arrival = true;
        }
      }
    }
    (void)arrival;
    sum += option == nullptr ? scorer.leave(t) : scorer.move(*option, t);
  }
  return sum;
}

struct Drained {
  std::vector<GeneratedTurn> turns;
  bool exhausted = false;
  bool stopped = false;
  double bound = 0;  // the generator's bound() at the end
};

Drained drain(const GameState& s, const TurnScorer& scorer, TurnUnit unit,
              const SearchBudget& budget = SearchBudget::unlimited()) {
  TurnGenerator g(s, scorer, unit, budget);
  Drained out;
  double last_bound = std::numeric_limits<double>::infinity();
  while (std::optional<GeneratedTurn> t = g.next()) {
    // The bound never rises, and holds every turn still to come.
    CHECK(t->score <= last_bound);
    last_bound = g.bound();
    CHECK(last_bound <= t->score);
    out.turns.push_back(std::move(*t));
  }
  CHECK_EQ(g.generated(), out.turns.size());
  out.exhausted = g.exhausted();
  out.stopped = g.stopped();
  out.bound = g.bound();
  CHECK(out.exhausted != out.stopped);
  if (out.exhausted) CHECK(g.bound() == -std::numeric_limits<double>::infinity());
  return out;
}

struct Tally {
  int positions = 0;
  int turns = 0;
  int multi_board = 0;   // positions with two playable boards or more
  int ties = 0;          // turns whose score equals the one before it
  int branch_pairs = 0;  // positions with an action creating two timelines or more
  int stopped = 0;       // budget-stopped runs checked as prefixes
};

// Checks (i)-(vi) at `s` against `expected_keys` (the reference's action keys)
// and `expected_successors`, and `all`, every legal action with its moves.
void check_position(const GameState& s, const std::set<std::string>& expected_keys,
                    const std::vector<GameState>& expected_successors, const std::vector<std::vector<Move>>& all,
                    const TurnScorer& scorer, Tally& tally, const std::string& where) {
  const bool even = s.multiverse().even();
  ++tally.positions;
  if (s.playable_timelines().size() >= 2) ++tally.multi_board;
  // (iv)'s expectation: every legal action's score, sorted.
  std::vector<double> want;
  for (const std::vector<Move>& a : all) want.push_back(recompute(s, a, scorer));
  std::sort(want.begin(), want.end(), std::greater<>());
  for (const TurnUnit unit : {TurnUnit::Actions, TurnUnit::Successors}) {
    const Drained d = drain(s, scorer, unit);
    CHECK(d.exhausted);
    std::set<std::string> keys;
    std::vector<GameState> successors;
    bool ordered = true;
    bool exact = true;
    for (std::size_t i = 0; i < d.turns.size(); ++i) {
      const GeneratedTurn& t = d.turns[i];
      // (ii), (iii)
      if (i > 0 && t.score > d.turns[i - 1].score) ordered = false;
      if (i > 0 && t.score == d.turns[i - 1].score) ++tally.ties;
      if (t.score != recompute(s, t.moves, scorer)) exact = false;
      // The moves are a legal action, in a playable order, reaching `result`.
      GameState r = s;
      for (const Move& m : t.moves) r.apply(m);
      CHECK(r.obligations_met() && royals_safe(r));
      r.submit(CheckRuling::none());
      CHECK(same_position(r, t.result));
      keys.insert(t5d_test::Reference::key(t.moves, even));
      if (std::none_of(successors.begin(), successors.end(),
                       [&](const GameState& u) { return same_position(u, t.result); })) {
        successors.push_back(t.result);
      }
    }
    CHECK(ordered);
    CHECK(exact);
    // (i): each once - as actions, every key distinct; as successors, every
    // state distinct.
    if (unit == TurnUnit::Actions) {
      CHECK_EQ(keys.size(), d.turns.size());
      const bool same = keys == expected_keys;
      if (!same) {
        std::fprintf(stderr, "  actions: %zu generated, %zu expected at %s\n", keys.size(), expected_keys.size(),
                     where.c_str());
      }
      CHECK(same);
      // (iv)
      std::vector<double> got;
      for (const GeneratedTurn& t : d.turns) got.push_back(t.score);
      CHECK(got == want);
      tally.turns += static_cast<int>(d.turns.size());
      for (const std::string& k : keys) {
        if (std::count(k.begin(), k.end(), ';') >= 2) {
          ++tally.branch_pairs;
          break;
        }
      }
    } else {
      CHECK_EQ(successors.size(), d.turns.size());
      CHECK_EQ(successors.size(), expected_successors.size());
      for (const GameState& e : expected_successors) {
        CHECK(std::any_of(successors.begin(), successors.end(),
                          [&](const GameState& u) { return same_position(u, e); }));
      }
      // Each state's first action is its best: the highest score of the
      // actions reaching that state.
      std::map<std::uint64_t, double> best;
      for (const std::vector<Move>& a : all) {
        GameState r = s;
        for (const Move& m : a) r.apply(m);
        r.submit(CheckRuling::none());
        const double score = recompute(s, a, scorer);
        const auto [it, fresh] = best.try_emplace(r.zobrist(), score);
        if (!fresh) it->second = std::max(it->second, score);
      }
      // (The listing keeps one order of each action's moves: under Q3's play
      // activation another order can reach another state - one the
      // generator emits and the listing's actions do not reach.)
      std::size_t found = 0;
      for (const GeneratedTurn& t : d.turns) {
        const auto it = best.find(t.result.zobrist());
        if (it == best.end()) continue;
        ++found;
        CHECK(it->second == t.score);
      }
      CHECK_EQ(found, best.size());
    }
    // (v) the same again.
    const Drained again = drain(s, scorer, unit);
    CHECK_EQ(again.turns.size(), d.turns.size());
    for (std::size_t i = 0; i < std::min(again.turns.size(), d.turns.size()); ++i) {
      CHECK(again.turns[i].moves == d.turns[i].moves);
      CHECK(again.turns[i].score == d.turns[i].score);
    }
    // (vi) under a budget that stops it part-way: a prefix, stopped.
    const std::uint64_t used = [&] {
      TurnGenerator g(s, scorer, unit);
      while (g.next()) {
      }
      return g.stats().nodes;
    }();
    if (used >= 4) {
      const Drained cut = drain(s, scorer, unit, SearchBudget::nodes(used / 2));
      CHECK(cut.stopped);
      CHECK(!cut.exhausted);
      CHECK(cut.turns.size() <= d.turns.size());
      for (std::size_t i = 0; i < std::min(cut.turns.size(), d.turns.size()); ++i) {
        CHECK(cut.turns[i].moves == d.turns[i].moves);
      }
      // A stopped generator's bound still holds every turn it did not emit
      // (the box it was judging is queued again).
      if (cut.turns.size() < d.turns.size()) CHECK(cut.bound >= d.turns[cut.turns.size()].score);
      ++tally.stopped;
    }
  }
}

// Against the engine's own exhaustive listing (E4).
bool check_against_listing(const GameState& s, const TurnScorer& scorer, Tally& tally, std::size_t limit,
                           const std::string& where) {
  const ActionList list = list_legal_actions(s, limit, Detector::Naive, SearchBudget::nodes(3'000'000));
  if (!list.stats.exhaustive) return false;  // too large for an exhaustive comparison
  const bool even = s.multiverse().even();
  std::set<std::string> keys;
  std::vector<GameState> successors;
  std::vector<std::vector<Move>> all;
  for (const LegalAction& a : list.actions) {
    keys.insert(t5d_test::Reference::key(a.moves, even));
    all.push_back(a.moves);
    if (std::none_of(successors.begin(), successors.end(),
                     [&](const GameState& u) { return same_position(u, a.result); })) {
      successors.push_back(a.result);
    }
  }
  check_position(s, keys, successors, all, scorer, tally, where);
  return true;
}

void report(const char* what, const Tally& t) {
  std::printf("  [%s] %d positions (%d multi-board), %d turns, %d ties, %d with two creations, %d budget cuts\n",
              what, t.positions, t.multi_board, t.turns, t.ties, t.branch_pairs, t.stopped);
}

void tiny_positions(int count, int max_axes, std::uint64_t seed, Tally& tally) {
  t5d_test::Rng rng(seed);
  std::vector<std::string> boards;
  int done = 0;
  for (int attempt = 0; done < count && attempt < count * 20; ++attempt) {
    const int width = attempt % 3 == 1 ? 2 : 1;
    const std::uint64_t limit = width == 1 ? 200000 : 50000;
    const std::optional<GameState> s =
        t5d_test::tiny_position(rng, width == 1 ? max_axes : std::min(max_axes, 4), attempt % 3 == 2, width, boards);
    if (!s) continue;
    const t5d_test::Reference ref = t5d_test::reference_actions(*s, limit);
    if (ref.overflow) continue;
    // The actions' moves: the engine's exhaustive listing, its keys checked
    // against the reference's.
    const ActionList list = list_legal_actions(*s, 1000000, Detector::Naive);
    if (!list.stats.exhaustive) continue;
    std::vector<std::vector<Move>> all;
    std::set<std::string> listed;
    for (const LegalAction& a : list.actions) {
      all.push_back(a.moves);
      listed.insert(t5d_test::Reference::key(a.moves, s->multiverse().even()));
    }
    CHECK(listed == ref.actions);
    std::string where = describe(s->rules());
    for (const std::string& b : boards) where += " " + b;
    ++done;
    const std::uint64_t score_seed = mix(seed + static_cast<std::uint64_t>(attempt));
    const int levels = attempt % 4 == 0 ? 0 : attempt % 4;  // continuous, or 1-3 levels (ties)
    check_position(*s, ref.actions, ref.successors, all, HashScorer(score_seed, levels), tally, where);
  }
  CHECK_EQ(done, count);
}

// A random legal action (random sequences until one is safe, else a witness).
std::optional<std::vector<Move>> random_action(const GameState& s, t5d_test::Rng& rng) {
  MoveList moves;
  for (int attempt = 0; attempt < 24; ++attempt) {
    GameState t = s;
    std::vector<Move> seq;
    for (int step = 0; step < 64; ++step) {
      if (t.obligations_met() && rng.chance(60)) break;
      generate_moves(t, moves);
      if (moves.empty()) break;
      const Move m = moves[static_cast<std::size_t>(rng.below(static_cast<int>(moves.size())))];
      t.apply(m);
      seq.push_back(m);
    }
    if (t.obligations_met() && royals_safe(t)) return seq;
  }
  const MateResult r = detect_mate(s, Detector::Hypercuboid, SearchBudget::nodes(100000));
  if (r.verdict == Outcome::Ongoing) return r.witness;
  return std::nullopt;
}

void games_from(const GameState& start, int turns, std::uint64_t seed, Tally& tally, const std::string& name) {
  t5d_test::Rng rng(seed);
  GameState s = start;
  for (int turn = 0; turn < turns; ++turn) {
    for (const int levels : {0, 2}) {
      check_against_listing(s, HashScorer(seed * 31 + static_cast<std::uint64_t>(turn), levels), tally, 5000,
                            name + " turn " + std::to_string(turn));
    }
    const std::optional<std::vector<Move>> action = random_action(s, rng);
    if (!action) break;
    for (const Move& m : *action) s.apply(m);
    s.submit();
  }
}

// Constructed multi-board presents: rooks jumping between two present boards
// (a move between them resolves both), knights and a branch behind the
// present (a rewinding branch makes the rest optional), three boards.
std::vector<std::pair<std::string, GameState>> constructed() {
  std::vector<std::pair<std::string, GameState>> out;
  const auto add = [&](const std::string& name, const std::vector<std::string>& boards, BoardSize size) {
    out.emplace_back(name, t5d_test::game(boards, size));
  };
  add("two rooks across", {"[k4/5/5/5/K3R:-0:1:w]", "[k4/5/5/5/K3R:+0:1:w]"}, BoardSize{5, 5});
  add("three boards", {"[3k/4/4/KR2:-1:1:w]", "[4/1p2/2P1/4:0:1:w]", "[k3/4/4/2RK:+1:1:w]"}, BoardSize{4, 4});
  add("knights and bishops", {"[1k2/4/4/1KN1:-0:1:w]", "[2k1/4/4/B1K1:+0:1:w]"}, BoardSize{4, 4});
  add("pawns", {"[k4/p4/5/4P/K4:-0:1:w]", "[4k/5/2N2/5/4K:+0:1:w]"}, BoardSize{5, 5});
  // A history to branch into: white's queen can go back to turn 1 (behind the
  // present on one board, so the branch rewinds it).
  add("a rewinding branch",
      {"[k3/4/4/K1Q1:-0:1:w]", "[k3/4/4/K1Q1:-0:1:b]", "[k3/4/4/K1Q1:-0:2:w]", "[k3/4/4/KQ2:+0:1:w]",
       "[k3/4/4/KQ2:+0:1:b]", "[k3/4/4/KQ2:+0:2:w]"},
      BoardSize{4, 4});
  return out;
}

}  // namespace

TEST(turngen_equals_the_brute_force_reference_on_tiny_positions) {
  Tally t;
  tiny_positions(300, 5, 6101, t);
  report("tiny", t);
  CHECK(t.multi_board > 150);
  CHECK(t.ties > 500);
  CHECK(t.branch_pairs > 20);
  CHECK(t.stopped > 100);
}

TEST(long_turngen_equals_the_brute_force_reference_on_tiny_positions) {
  Tally t;
  tiny_positions(3000, 7, 6202, t);
  report("tiny, long", t);
}

TEST(turngen_equals_the_exhaustive_listing_on_constructed_presents) {
  Tally t;
  for (const auto& [name, s] : constructed()) {
    for (const int levels : {0, 1, 3}) {
      CHECK(check_against_listing(s, HashScorer(17 + static_cast<std::uint64_t>(levels), levels), t, 200000, name));
    }
  }
  report("constructed", t);
  CHECK_EQ(t.multi_board, t.positions);
  // The rewinding branch is there: a legal turn leaves a mandatory board
  // unplayed (T8: the branch puts the present behind it).
  const GameState rewind = constructed().back().second;
  const ActionList all = list_legal_actions(rewind, 100000, Detector::Naive);
  bool left = false;
  for (const LegalAction& a : all.actions) {
    for (const int p : rewind.mandatory_timelines()) {
      left = left || std::none_of(a.moves.begin(), a.moves.end(), [&](const Move& m) {
               return m.from_timeline == p || (m.jump() && !m.branching() && m.to_timeline == p);
             });
    }
  }
  CHECK(left);
}

TEST(turngen_equals_the_exhaustive_listing_on_every_variant) {
  Tally t;
  std::uint64_t seed = 401;
  for (const Variant& v : variants()) games_from(variant_start(v), 3, seed++, t, std::string(v.name));
  report("variants", t);
  CHECK(t.positions > 60);
}

TEST(long_turngen_equals_the_exhaustive_listing_on_every_variant) {
  Tally t;
  std::uint64_t seed = 901;
  for (const Variant& v : variants()) {
    for (int game = 0; game < 3; ++game) games_from(variant_start(v), 8, seed++, t, std::string(v.name));
  }
  report("variants, long", t);
}

TEST(turngen_refuses_a_state_during_an_action_and_a_non_finite_score) {
  const GameState s = variant_start(*find_variant("Very Small"));
  struct Bad : TurnScorer {
    double leave(int) const override { return 0; }
    double move(const Move&, int) const override { return std::numeric_limits<double>::quiet_NaN(); }
  };
  CHECK_THROWS_AS(TurnGenerator(s, Bad{}), std::invalid_argument);
  GameState mid = s;
  MoveList moves;
  generate_moves(mid, moves);
  mid.apply(moves[0]);  // one board: the obligations are met
  CHECK_THROWS_AS(TurnGenerator(mid, HashScorer(1, 0)), std::invalid_argument);
}

TEST(turngen_orders_a_single_board_by_its_scores) {
  // One board: every turn is one move, so the order is the moves' scores -
  // the scorer's, highest first; leaving the (mandatory) board is no turn.
  const GameState s = variant_start(*find_variant("Very Small"));
  const HashScorer scorer(5, 0);
  const Drained d = drain(s, scorer, TurnUnit::Actions);
  REQUIRE(d.exhausted);
  MoveList legal;
  legal_moves(s, legal);
  CHECK_EQ(d.turns.size(), legal.size());
  std::vector<double> want;
  for (const Move& m : legal) want.push_back(scorer.leave(0) * 0 + scorer.move(m, m.from_timeline));
  std::sort(want.begin(), want.end(), std::greater<>());
  for (std::size_t i = 0; i < d.turns.size(); ++i) {
    REQUIRE_EQ(d.turns[i].moves.size(), std::size_t{1});
    CHECK(d.turns[i].score == want[i]);
  }
}

TEST(turngen_a_checkmate_has_no_turn) {
  // The family of test_detectors' near-mate positions at n = 1: no legal turn.
  std::vector<std::string> boards;
  for (int p = -1; p <= 1; ++p) {
    const std::string label = p == 0 ? "0" : (p > 0 ? "+" : "") + std::to_string(p);
    boards.push_back("[" + std::string(p == 0 ? "k4/5/5/5/RR3" : "5/5/4n/5/RR3") + ":" + label + ":1:b]");
  }
  const GameState s = t5d_test::game(boards, BoardSize{5, 5});
  REQUIRE(detect_mate(s, Detector::Hypercuboid).verdict == Outcome::Checkmate);
  TurnGenerator g(s, HashScorer(3, 0));
  CHECK(!g.next());
  CHECK(g.exhausted());
  CHECK(!g.stopped());
  CHECK(g.stats().nodes < 2000);
}

// ---- the realisation cursor (review fixes, 2026-10-23) -------------------------------------

namespace {

// Sets how many realisations a step records (t5d/turngen.hpp's test seam) for
// a scope.
struct StepCap {
  explicit StepCap(std::size_t n) { detail::set_realisations_per_step(n); }
  ~StepCap() { detail::set_realisations_per_step(32); }
  StepCap(const StepCap&) = delete;
  StepCap& operator=(const StepCap&) = delete;
};

struct Emitted {
  std::vector<std::vector<Move>> moves;
  std::vector<double> scores;
  std::vector<std::uint64_t> results;  // the successors' keys
  std::uint64_t resumed = 0;
  bool exhausted = false;
};

Emitted emit_all(const GameState& s, const TurnScorer& scorer, TurnUnit unit) {
  TurnGenerator g(s, scorer, unit);
  Emitted out;
  while (std::optional<GeneratedTurn> t = g.next()) {
    out.moves.push_back(t->moves);
    out.scores.push_back(t->score);
    out.results.push_back(t->result.zobrist());
  }
  out.resumed = g.stats().resumed;
  out.exhausted = g.exhausted();
  return out;
}

// Each board's king can step back one turn on its own timeline (a branch):
// that move scores 0, every other option -10. The turn making all k of them is
// one point whose k! creation orders are k! legal turns (J9), the top score.
struct BackInTime : TurnScorer {
  double leave(int) const override { return -10; }
  double move(const Move& m, int timeline) const override {
    const bool back = timeline == m.from_timeline && m.to_timeline == m.from_timeline &&
                      m.to_half_turn < m.from_half_turn && m.to_square == m.from_square;
    return back ? 0 : -10;
  }
};

GameState kings_back(int half, const RuleOptions& rules) {
  static const char* const kKings[8] = {"k3/4", "1k2/4", "2k1/4", "3k/4", "4/k3", "4/1k2", "4/2k1", "4/3k"};
  std::vector<std::string> boards;
  for (int p = -half; p <= half; ++p) {
    const std::string label = p == 0 ? "0" : (p > 0 ? "+" : "") + std::to_string(p);
    const std::string black = kKings[(p + 12) % 8];
    boards.push_back("[" + black + "/4/K3:" + label + ":1:w]");
    boards.push_back("[" + black + "/4/K3:" + label + ":1:b]");
    boards.push_back("[" + black + "/4/KR2:" + label + ":2:w]");
  }
  return t5d_test::game(boards, BoardSize{4, 4}, rules);
}

}  // namespace

TEST(turngen_the_realisation_cursor_resumes_exactly) {
  // A point whose realisations reach a step's cap (32) is resumed where it
  // stopped (the realisation cursor), not judged again from its first. With
  // the cap lowered to 1 and 2, every point of two realisations or more is
  // stopped and resumed - under every regime: the default rules (FreeOrder),
  // Q4's Guide-literal reading (Ordered: both shapes) and Q3's play
  // activation (General: its tables kept across the resume) - and the turns
  // emitted must be the shipped cap's, in the same order, scores and
  // successors alike.
  t5d_test::Rng rng(7303);
  std::vector<std::string> boards;
  int positions = 0;
  int resumed = 0;
  int resumed_guide_literal = 0;
  int resumed_play_activates = 0;
  int resumed_default = 0;
  for (int attempt = 0; positions < 400 && attempt < 8000; ++attempt) {
    const int width = attempt % 3 == 1 ? 2 : 1;
    const std::optional<GameState> s =
        t5d_test::tiny_position(rng, width == 1 ? 6 : 4, attempt % 3 == 2, width, boards);
    if (!s) continue;
    const ActionCount count = count_legal_actions(*s, 4000, Detector::Hypercuboid, SearchBudget::nodes(400000));
    if (!count.stats.exhaustive) continue;
    ++positions;
    const HashScorer scorer(mix(0x5EED + static_cast<std::uint64_t>(attempt)), attempt % 2 == 0 ? 0 : 2);
    bool any = false;
    for (const TurnUnit unit : {TurnUnit::Actions, TurnUnit::Successors}) {
      const Emitted want = emit_all(*s, scorer, unit);
      CHECK(want.exhausted);
      for (const std::size_t cap : {std::size_t{1}, std::size_t{2}}) {
        const StepCap guard(cap);
        const Emitted got = emit_all(*s, scorer, unit);
        CHECK(got.exhausted);
        CHECK(got.scores == want.scores);
        CHECK(got.results == want.results);
        // As actions, the very moves. As successors the states and scores
        // are the same, but which of a state's actions (two realisations of one
        // point can reach one state: equal scores) stands for it depends on
        // when the step collects it - the cap - as before the cursor.
        if (unit == TurnUnit::Actions) CHECK(got.moves == want.moves);
        any = any || got.resumed > 0;
      }
    }
    if (any) {
      ++resumed;
      const RuleOptions& r = s->rules();
      if (r.play_activates) {
        ++resumed_play_activates;
      } else if (!r.moves_after_shift) {
        ++resumed_guide_literal;
      } else {
        ++resumed_default;
      }
    }
  }
  std::printf("  [cursor] %d positions, %d resumed (%d Q3, %d Q4 Guide-literal, %d other)\n", positions, resumed,
              resumed_play_activates, resumed_guide_literal, resumed_default);
  CHECK_EQ(positions, 400);
  CHECK(resumed_play_activates >= 5);
  CHECK(resumed_guide_literal >= 5);
  CHECK(resumed_default >= 20);
}

TEST(turngen_a_point_of_many_realisations_walks_each_once) {
  // A-M1 (review, 2026-10-23): k kings each stepping back on their own
  // timeline - one point of k! legal turns, the top score. Each realisation is
  // tested once (the cursor; judging the point again from its first
  // realisation tested 401,936 for 5,040 at k = 7 and spent 1,495,064 nodes),
  // at most 32 recorded per step (what a step holds: tested - emitted <= 31
  // throughout), and the nodes grow linearly - so the search arm's default
  // generator budget (200,000 nodes) reaches every top turn (1,824 of 5,040
  // before).
  struct Case {
    int half;
    bool play_activates;  // Q3: the general regime's walk
  };
  for (const Case c : {Case{3, false}, Case{2, true}}) {
    RuleOptions rules;
    rules.play_activates = c.play_activates;
    const GameState s = kings_back(c.half, rules);
    const std::size_t k = static_cast<std::size_t>(2 * c.half + 1);
    std::size_t n = 1;
    for (std::size_t i = 2; i <= k; ++i) n *= i;
    for (const TurnUnit unit : {TurnUnit::Actions, TurnUnit::Successors}) {
      TurnGenerator g(s, BackInTime{}, unit);
      std::size_t top = 0;
      std::uint64_t tested = 0;
      std::uint64_t nodes = 0;
      bool held = true;
      std::set<std::uint64_t> results;
      while (std::optional<GeneratedTurn> t = g.next()) {
        if (t->score != 0) break;
        ++top;
        results.insert(t->result.zobrist());
        tested = g.stats().actions_tested;
        nodes = g.stats().nodes;
        if (tested < g.generated() || tested > g.generated() + 31) held = false;
      }
      std::printf("  [k = %zu%s, %s] %zu top turns; tested %llu, nodes %llu, resumed %llu\n", k,
                  c.play_activates ? ", Q3" : "", unit == TurnUnit::Actions ? "actions" : "successors", top,
                  static_cast<unsigned long long>(tested), static_cast<unsigned long long>(nodes),
                  static_cast<unsigned long long>(g.stats().resumed));
      CHECK_EQ(top, n);
      CHECK_EQ(results.size(), n);
      CHECK(held);
      CHECK_EQ(tested, static_cast<std::uint64_t>(n));
      CHECK(nodes <= static_cast<std::uint64_t>(8 * n + 2000));
      CHECK(g.stats().resumed >= n / 32 - 1);
    }
    if (!c.play_activates) {
      SearchBudget budget;
      budget.max_nodes = 200'000;  // the search arm's generator_nodes
      TurnGenerator g(s, BackInTime{}, TurnUnit::Successors, budget);
      std::size_t top = 0;
      while (std::optional<GeneratedTurn> t = g.next()) {
        if (t->score != 0) break;
        ++top;
      }
      CHECK_EQ(top, n);
    }
  }
}

TEST(turngen_a_generator_stopped_in_its_set_up_bounds_nothing) {
  // A budget that stops the generator before its boxes are queued: nothing is
  // known about the turns, so bound() is +infinity (it read -infinity, an
  // exhausted generator's, before the review fixes).
  const GameState s = variant_start(*find_variant("Very Small"));
  TurnGenerator g(s, HashScorer(1, 0), TurnUnit::Successors, SearchBudget::nodes(1));
  CHECK(!g.next());
  CHECK(g.stopped());
  CHECK(!g.exhausted());
  CHECK(g.bound() == std::numeric_limits<double>::infinity());
}

TEST(turngen_equals_the_exhaustive_listing_with_castling_and_en_passant) {
  // A-L4 (review, 2026-10-23): castling and en passant through the generator.
  // Two present boards, the timeline +1 a turn ahead of timeline 0's present:
  // castling on both (on +1, an optional board: Q21), and an en passant
  // capture on timeline 0 (the pawn's double step - from its unmoved square,
  // marked - just played). Under the
  // default rules, Q21's alternative (no castling on an optional board) and
  // Q3's play activation, as actions and as successors, against E4's
  // exhaustive listing.
  const std::string base = "R*3K*2R*";
  // Black to move on timeline 0 (its present); 5DFEN cannot record an en
  // passant right, so Black's d7-d5 is played (S7: the board it creates
  // carries the right) and submitted.
  const std::vector<std::string> boards = {
      "[4k3/3p*4/8/8/4P3/8/8/" + base + ":0:1:w]", "[4k3/3p*4/8/4P3/8/8/8/" + base + ":0:1:b]",
      "[4k3/8/8/8/8/8/8/" + base + ":+1:2:w]", "[4k3/8/8/8/8/8/8/" + base + ":+1:2:b]",
      "[4k3/8/8/8/8/8/8/" + base + ":+1:3:w]"};
  struct Ruling {
    const char* name;
    RuleOptions rules;
  };
  RuleOptions q21;
  q21.castling_on_optional_boards = false;
  RuleOptions q3;
  q3.play_activates = true;
  Tally t;
  std::vector<std::size_t> castlings;
  for (const Ruling& r : {Ruling{"default", RuleOptions{}}, Ruling{"Q21 alternative", q21}, Ruling{"Q3", q3}}) {
    GameState s = t5d_test::game(boards, BoardSize{8, 8}, r.rules);
    const std::optional<Move> two = t5d_test::find_move(s, "(0T1)Pd7(0T1)d5");
    REQUIRE(two.has_value() && two->double_step());
    s.apply(*two);
    s.submit();
    REQUIRE(s.playable_timelines().size() == 2);
    const ActionList list = list_legal_actions(s, 100000, Detector::Naive);
    REQUIRE(list.stats.exhaustive);
    std::size_t castling = 0;
    std::size_t en_passant = 0;
    for (const LegalAction& a : list.actions) {
      castling += std::any_of(a.moves.begin(), a.moves.end(), [](const Move& m) { return m.castle(); }) ? 1u : 0u;
      en_passant += std::any_of(a.moves.begin(), a.moves.end(), [](const Move& m) { return m.en_passant(); }) ? 1u : 0u;
    }
    std::printf("  [%s] %zu actions, %zu castling, %zu en passant\n", r.name, list.actions.size(), castling,
                en_passant);
    CHECK(castling > 0);
    CHECK(en_passant > 0);
    castlings.push_back(castling);
    for (const int levels : {0, 2}) {
      CHECK(check_against_listing(s, HashScorer(41 + static_cast<std::uint64_t>(levels), levels), t, 100000, r.name));
    }
  }
  report("castling, en passant", t);
  // Q21's alternative: no castling on the optional board (+1), only on the
  // mandatory one.
  CHECK(castlings.size() == 3 && castlings[1] < castlings[0] && castlings[2] == castlings[0]);
}

TEST(turngen_ties_cost_little) {
  // B-M5 (review, 2026-10-23): with every option scoring alike (ties on every
  // board), the first turns cost a bounded number of nodes - ties searched
  // depth-first (the newest of equal boxes first; each axis's lowest equal
  // value first). Taken breadth-first, or the highest equal value first, the
  // cost grows several-fold with the boards (measured at the review fixes, 11
  // boards, first 16 turns: 1,568 nodes; oldest box first 9,828; highest value
  // first 15,717). The budgets below are the ceilings: a generator they stop
  // fails here at once, instead of a test tier timing out.
  struct Zero : TurnScorer {
    double leave(int) const override { return 0; }
    double move(const Move&, int) const override { return 0; }
  };
  struct Case {
    int half;
    std::uint64_t ceiling;  // nodes for the first 16 turns
  };
  for (const Case c : {Case{3, 600}, Case{4, 1000}, Case{5, 2000}}) {
    const GameState s = kings_back(c.half, RuleOptions{});
    for (const TurnUnit unit : {TurnUnit::Actions, TurnUnit::Successors}) {
      TurnGenerator g(s, Zero{}, unit, SearchBudget::nodes(c.ceiling));
      std::size_t n = 0;
      while (n < 16 && g.next()) ++n;
      std::printf("  [%zu boards, %s] first %zu turns: %llu nodes (ceiling %llu)\n", s.playable_timelines().size(),
                  unit == TurnUnit::Actions ? "actions" : "successors", n,
                  static_cast<unsigned long long>(g.stats().nodes), static_cast<unsigned long long>(c.ceiling));
      CHECK_EQ(n, std::size_t{16});
      CHECK(!g.stopped());
    }
  }
}
