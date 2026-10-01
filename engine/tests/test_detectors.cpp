// The two mate detectors cross-checked (design spec §11.0 item 3; register
// H37): on positions from random legal play - from every named variant, from
// sparse starts that run into mates within a few turns, and under the
// alternative rulesets - the naive and the hypercuboid searches must give the
// same verdict wherever both prove one, the same count (to a cap) wherever
// both know it, and the same successor states. Each search runs under a node
// budget, so either may answer Unknown; an Unknown is compared with nothing.
//
// The default run is kept short; the tests named long_* (t5d-tests --long, or
// a fragment beginning "long_") play many more games under larger budgets.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/fen.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"
#include "test_support.hpp"

using namespace t5d;

namespace {

struct Tally {
  int positions = 0;
  int proven_both = 0;
  int mates = 0;
  int stalemates = 0;
  int counts_compared = 0;
  int sets_compared = 0;
  int unknown = 0;
  double naive_seconds = 0;
  double hyper_seconds = 0;
};

std::string describe_position(const GameState& s) {
  std::string out = s.to_move() == Colour::White ? "white to move:" : "black to move:";
  const Multiverse& mv = s.multiverse();
  mv.for_each_board([&](int p, int h, const Board& b) {
    out += ' ';
    out += fen_board_string(b, p, h, mv.even());
  });
  out += " size " + to_string(mv.board_size()) + " rules " + describe(s.rules());
  return out;
}

const char* name_of(Outcome o) {
  switch (o) {
    case Outcome::Ongoing: return "none";
    case Outcome::Checkmate: return "checkmate";
    case Outcome::Stalemate: return "stalemate";
    case Outcome::Unknown: return "unknown";
  }
  return "?";
}

std::vector<std::uint64_t> keys_of(const ActionList& list) {
  std::vector<std::uint64_t> out;
  for (const LegalAction& a : list.actions) out.push_back(a.result.zobrist());
  std::sort(out.begin(), out.end());
  return out;
}

// Compares the detectors at `s`, a turn's start.
void compare(const GameState& s, Tally& t, std::uint64_t budget) {
  ++t.positions;
  const SearchBudget b = SearchBudget::nodes(budget);
  const MateResult n = detect_mate(s, Detector::Naive, b);
  const MateResult h = detect_mate(s, Detector::Hypercuboid, b);
  t.naive_seconds += n.stats.seconds;
  t.hyper_seconds += h.stats.seconds;
  CHECK_EQ(n.in_check, h.in_check);
  for (const MateResult* r : {&n, &h}) {
    // H37: a proof only from an exhaustive search.
    if (r->verdict == Outcome::Checkmate || r->verdict == Outcome::Stalemate) {
      CHECK(r->stats.exhaustive);
    }
    if (r->verdict == Outcome::Unknown) ++t.unknown;
  }
  if (n.verdict != Outcome::Unknown && h.verdict != Outcome::Unknown) {
    ++t.proven_both;
    if (n.verdict != h.verdict) {
      std::fprintf(stderr, "  detectors disagree: naive %s, hypercuboid %s at %s\n",
                   name_of(n.verdict), name_of(h.verdict), describe_position(s).c_str());
    }
    CHECK(n.verdict == h.verdict);
    if (n.verdict == Outcome::Checkmate) ++t.mates;
    if (n.verdict == Outcome::Stalemate) ++t.stalemates;
  }
  constexpr std::size_t kCap = 40;
  const ActionList ln = list_legal_actions(s, kCap, Detector::Naive, b);
  const ActionList lh = list_legal_actions(s, kCap, Detector::Hypercuboid, b);
  const bool known_n = ln.stats.exhaustive || ln.reached_cap;
  const bool known_h = lh.stats.exhaustive || lh.reached_cap;
  if (known_n && known_h) {
    ++t.counts_compared;
    if (ln.actions.size() != lh.actions.size()) {
      std::fprintf(stderr, "  counts disagree: naive %zu, hypercuboid %zu at %s\n",
                   ln.actions.size(), lh.actions.size(), describe_position(s).c_str());
    }
    CHECK_EQ(ln.actions.size(), lh.actions.size());
    if (ln.stats.exhaustive && lh.stats.exhaustive) {
      ++t.sets_compared;
      CHECK(keys_of(ln) == keys_of(lh));
    }
  }
  // A count's zero and a verdict's "none" agree.
  if (h.verdict != Outcome::Unknown && known_h) {
    CHECK_EQ(h.verdict == Outcome::Ongoing, !lh.actions.empty());
  }
}

// A random legal action: random pseudo-legal sequences until one is safe, else
// the hypercuboid's witness; nullopt if the turn has none (or none was found).
std::optional<std::vector<Move>> random_legal_action(const GameState& s, t5d_test::Rng& rng) {
  MoveList moves;
  for (int attempt = 0; attempt < 24; ++attempt) {
    GameState t = s;
    std::vector<Move> seq;
    for (int step = 0; step < 64; ++step) {
      const bool can_stop = t.obligations_met();
      if (can_stop && (!t.rules().moves_after_shift || rng.chance(60))) break;
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

void play_games(const GameState& start, int games, int turns, std::uint64_t seed, Tally& t,
                std::uint64_t budget) {
  t5d_test::Rng rng(seed);
  for (int g = 0; g < games; ++g) {
    GameState s = start;
    for (int turn = 0; turn < turns; ++turn) {
      compare(s, t, budget);
      const std::optional<std::vector<Move>> action = random_legal_action(s, rng);
      if (!action) break;
      for (const Move& m : *action) s.apply(m);
      s.submit();  // legal: the engine's ruling tests it again
    }
  }
}

// Sparse starts: few pieces, several timelines - random play runs into checks,
// mates and stalemates within a few turns.
struct Sparse {
  std::vector<std::string> boards;
  BoardSize size;
};
std::vector<Sparse> sparse_starts() {
  const BoardSize k5{5, 5};
  const BoardSize k4{4, 4};
  return {
      {{"[k4/5/5/5/K3R:-0:1:w]", "[k4/5/5/5/K3R:+0:1:w]"}, k5},
      {{"[2k2/5/5/5/QK3:0:1:w]"}, k5},
      {{"[k3/4/4/K1Q1:0:1:w]"}, k4},
      {{"[2k/3/K2:0:1:w]"}, BoardSize{3, 3}},
      {{"[3k/4/4/KR2:-1:1:w]", "[4/1p2/2P1/4:0:1:w]", "[k3/4/4/2RK:+1:1:w]"}, k4},
      {{"[k4/p4/5/4P/K4:-0:1:w]", "[4k/5/2N2/5/4K:+0:1:w]"}, k5},
      {{"[1k2/4/4/1KN1:-0:1:w]", "[2k1/4/4/B1K1:+0:1:w]"}, k4},
      {{"[k4/5/2q2/5/2K1R:0:1:w]"}, k5},
  };
}

void sparse_games(const RuleOptions& rules, int games, int turns, std::uint64_t seed, Tally& t,
                  std::uint64_t budget) {
  for (const Sparse& start : sparse_starts()) {
    play_games(t5d_test::game(start.boards, start.size, rules), games, turns, seed++, t, budget);
  }
}

std::vector<RuleOptions> alternative_rulesets() {
  std::vector<RuleOptions> out;
  {
    RuleOptions r;
    r.moves_after_shift = true;  // Q4
    out.push_back(r);
  }
  {
    RuleOptions r;
    r.mandatory = MandatoryBoards::AllActive;  // Q5
    r.earlier_playable_jump_branches = true;   // J3's other reading
    out.push_back(r);
  }
  {
    RuleOptions r;
    r.activation = ActivationRule::Consecutive;  // Q1
    r.play_activates = true;                     // Q3
    r.reactivation = Reactivation::Deferred;     // Q6
    r.inactive_history_is_destination = false;   // Q20
    out.push_back(r);
  }
  {
    RuleOptions r;
    r.sliders_cross_void = true;  // Q18
    r.past_self_blocks = false;   // Q19
    r.castling_on_optional_boards = false;
    r.initial_credit = InitialTimelineCredit::MinusZeroToBlack;  // Q2
    out.push_back(r);
  }
  return out;
}

void report(const char* what, const Tally& t) {
  std::printf("  [%s] %d positions: both proved %d (%d checkmates, %d stalemates), %d unknown; "
              "counts compared %d, successor sets %d; naive %.2fs, hypercuboid %.2fs\n",
              what, t.positions, t.proven_both, t.mates, t.stalemates, t.unknown,
              t.counts_compared, t.sets_compared, t.naive_seconds, t.hyper_seconds);
}

}  // namespace

TEST(detectors_agree_on_every_variant) {
  // Random legal games from every named variant (rules reference V3).
  Tally t;
  std::uint64_t seed = 1;
  for (const Variant& v : variants()) {
    play_games(variant_start(v), 1, 6, seed++, t, 20000);
  }
  report("variants", t);
  CHECK(t.positions > 60);
  CHECK(t.counts_compared > 50);
}

TEST(detectors_agree_near_mate) {
  // Sparse starts: the games reach checks, mates and stalemates.
  Tally t;
  sparse_games({}, 2, 14, 100, t, 20000);
  report("near mate", t);
  CHECK(t.positions > 100);
  CHECK(t.mates + t.stalemates > 3);  // proofs were compared
}

TEST(detectors_agree_under_alternative_rules) {
  // Q1-Q6, Q18-Q20, J3's other reading: the hypercuboid's general realisation.
  Tally t;
  std::uint64_t seed = 300;
  for (const RuleOptions& r : alternative_rulesets()) {
    sparse_games(r, 1, 10, seed, t, 20000);
    seed += 50;
  }
  report("alternative rules", t);
  CHECK(t.positions > 100);
}

TEST(detectors_the_hypercuboid_stays_small_near_mate) {
  // A family of mates (spec §11.0 item 3: "better worst-case performance,
  // especially near mate"): timelines -n..+n, all at T1 with Black to move and
  // active (each side has created n), so all mandatory. On 0 the black king a5
  // is checked by a rook a1, and a rook b1 covers b4 and b5: no square on its
  // board. Every other board holds white rooks a1 and b1 (so the king's jumps
  // onto a4, a5, b4, b5 there are taken) and a black knight e3, whose moves
  // never reach files a and b, take a rook, or block board 0's a-file.
  // Checkmate. The naive search tries the knights' moves in every combination
  // (2.4k nodes at n = 1, 0.8M at n = 2); the hypercuboid eliminates them
  // together, at a cost that does not grow with n.
  const auto position = [](int n) {
    std::vector<std::string> boards;
    for (int p = -n; p <= n; ++p) {
      const std::string label = p == 0 ? "0" : (p > 0 ? "+" : "") + std::to_string(p);
      boards.push_back("[" + std::string(p == 0 ? "k4/5/5/5/RR3" : "5/5/4n/5/RR3") + ":" + label +
                       ":1:b]");
    }
    return t5d_test::game(boards, BoardSize{5, 5});
  };
  for (int n = 0; n <= 1; ++n) {
    CHECK(detect_mate(position(n), Detector::Naive).verdict == Outcome::Checkmate);
  }
  for (int n = 0; n <= 5; ++n) {
    const MateResult r = detect_mate(position(n), Detector::Hypercuboid);
    CHECK(r.verdict == Outcome::Checkmate);
    CHECK(r.stats.exhaustive);
    CHECK(r.stats.nodes < 1000);
  }
  // And the naive search, on a budget the hypercuboid never needs, cannot
  // prove n = 2: Unknown, never a verdict (H37).
  const MateResult cut = detect_mate(position(2), Detector::Naive, SearchBudget::nodes(1000));
  CHECK(cut.verdict == Outcome::Unknown);
}

TEST(long_detectors_agree_on_many_games) {
  // The long run (t5d-tests --long): more games, more turns, larger budgets.
  Tally variants_tally;
  std::uint64_t seed = 1000;
  for (const Variant& v : variants()) {
    play_games(variant_start(v), 3, 12, seed++, variants_tally, 200000);
  }
  report("variants, long", variants_tally);
  Tally sparse_tally;
  sparse_games({}, 8, 24, 2000, sparse_tally, 200000);
  report("near mate, long", sparse_tally);
  Tally rules_tally;
  for (const RuleOptions& r : alternative_rulesets()) {
    sparse_games(r, 3, 16, seed, rules_tally, 200000);
    seed += 100;
  }
  report("alternative rules, long", rules_tally);
  CHECK(sparse_tally.mates + sparse_tally.stalemates > 20);
}
