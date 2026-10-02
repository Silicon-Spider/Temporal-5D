// The tactical scanner (t5d/scanner.hpp; design spec §11.1-§11.3, §15.4;
// register H1-H6, H15, H18, H19). Every position is written here from the
// rulebook (docs/rules/README.md) - never taken from a game corpus or another
// implementation's test data - and each mate is confirmed by E4's exhaustive
// detector (t5d/search.hpp) and by a brute-force game tree (test_game_tree.hpp)
// before the scanner's answer is compared with it.
//
// 5x5 boards unless said otherwise; 5DFEN rows run from rank 5 down to rank 1.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/fen.hpp"
#include "t5d/movegen.hpp"
#include "t5d/scanner.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_game_tree.hpp"
#include "test_positions.hpp"
#include "test_support.hpp"

using namespace t5d;
using t5d_test::game;
using t5d_test::GameTree;
using t5d_test::GameValue;

namespace {

constexpr BoardSize k5x5{5, 5};

ScanResult scan(const GameState& s, std::uint64_t nodes = 2'000'000,
                const ScannerOptions& o = {}) {
  Scanner scanner(o);
  return scanner.scan(s, ScanBudget::nodes(nodes));
}

[[maybe_unused]] std::string describe_position(const GameState& s) {
  std::string out = s.to_move() == Colour::White ? "white to move:" : "black to move:";
  const Multiverse& mv = s.multiverse();
  mv.for_each_board([&](int p, int h, const Board& b) {
    out += ' ';
    out += fen_board_string(b, p, h, mv.even());
  });
  out += " size " + to_string(mv.board_size()) + " rules " + describe(s.rules());
  return out;
}

// Replays an action given as move texts and submits it legally.
GameState after(const GameState& s, std::initializer_list<std::string_view> moves) {
  GameState t = s;
  t5d_test::act_legally(t, moves);
  return t;
}

// The scanner's proof against the brute-force game tree: a Win(d) or Loss(d)
// must be the same kind within d half-turns, a Draw a stalemate now. 1: it
// agrees; -1: it does not (a false proof); 0: the tree could not decide
// within its position budget (it answers Unknown) - never a false proof's
// excuse in the constructed positions, which are small enough to decide.
int verify(const Proof& p, const GameState& s, GameTree& tree) {
  if (!p.proven()) return 1;
  const GameValue v = tree.solve(s, p.distance);
  if (v.kind == GameValue::Kind::Unknown) return 0;
  switch (p.kind) {
    case Proof::Kind::Win:
      return v.kind == GameValue::Kind::Win && v.distance <= p.distance ? 1 : -1;
    case Proof::Kind::Loss:
      return v.kind == GameValue::Kind::Loss && v.distance <= p.distance ? 1 : -1;
    case Proof::Kind::Draw:
      return v.kind == GameValue::Kind::Draw ? 1 : -1;
    case Proof::Kind::None:
      break;
  }
  return 1;
}
bool agrees(const Proof& p, const GameState& s, GameTree& tree) { return verify(p, s, tree) == 1; }

// Proofs checked on random positions, each with a game tree of its own (the
// tree's memory is bounded per position).
struct ProofTally {
  /// Proofs longer than this are not checked (the game tree grows with the
  /// distance): the default run checks up to 3 half-turns, the long_ tests
  /// every one.
  int max_distance = 3;
  /// The game tree's work budget per proof (GameTree's max_work).
  std::uint64_t work = 300000;
  int proofs = 0;
  int verified = 0;
  int undecided = 0;
  int longer = 0;
  double tree_seconds = 0;
  double scan_seconds = 0;
  // Checks `r`'s proof at `s`; false for a false proof.
  bool check(const ScanResult& r, const GameState& s) {
    scan_seconds += r.stats.seconds;
    if (!r.proof.proven()) return true;
    ++proofs;
    if (r.proof.distance > max_distance) {
      ++longer;
      return true;
    }
    const auto t0 = std::chrono::steady_clock::now();
    GameTree tree(20000, 60000, work);
    const int v = verify(r.proof, s, tree);
    tree_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (v == 1) ++verified;
    if (v == 0) ++undecided;
    return v >= 0;
  }
  void print(const char* what) const {
    std::fprintf(stderr,
                 "  %s: %d proofs, %d confirmed by the game tree, %d beyond its budget, %d longer "
                 "than %d half-turns not checked (scans %.1f s, game trees %.1f s)\n",
                 what, proofs, verified, undecided, longer, max_distance, scan_seconds,
                 tree_seconds);
  }
};

// ---- positions ------------------------------------------------------------------------

// White to move: Qe4, Kb3 against Ka5. Qa4, Qb4 and Qe5 each mate (the king
// b3 covers a4 and b4; the queen the rest).
GameState mate_in_one() { return game({"[k4/4Q/1K3/5/5:0:1:w]"}, k5x5); }

// Across timelines: -0 holds White's queen a4 (and its king e1, a black pawn
// c3 to move); +0 Black's king a5 against White's king b3. The queen's jump
// one timeline up, (-0T1)Qa4 -> (+0T1)a4, plays both boards (J4) and checks;
// the king's every square on +0 is covered, and so is every square of -0 it
// could jump to (the queen reaches each along L, or L with x and y). Mate.
GameState cross_board_mate() {
  return game({"[5/Q4/2p2/5/4K:-0:1:w]", "[k4/5/1K3/5/5:+0:1:w]"}, k5x5);
}

// C8 one move earlier: White must play 0 and +1 (both at T1). Re1-a1 checks
// the king a5 on 0 - answerable alone by Kb4 or Kb5 - but +1 (White's rooks
// a1, b1 and a pawn e2) holds no black piece, and the king's only way to play
// it, a jump, lands under the rooks. White plays its pawn on +1. Stale-
// checkmate (C8).
GameState stale_checkmate_in_one() {
  return game({"[k3K/5/5/5/4R:0:1:w]", "[5/5/5/4P/RR3:+1:1:w]"}, k5x5);
}

// Mate in 2 by checks (three forcing half-turns): Black Ka4, White Re3, Rd2,
// Ke1. 1. Rd4+ (rank 4; a3 and b3 under Re3) Ka5 or Kb5 2. Re5#. The black
// king's branch back to (0T1) after 2. Re5 lands on a copy where Rd4 still
// checks the king a4: illegal (C4).
GameState mate_in_two() { return game({"[5/k4/4R/3R1/4K:0:1:w]"}, k5x5); }

// C2, a royal on a history board: one timeline, T1 to T3, White to move at
// T3. Black's king a5 is walled in by its own pawns a4, b4, b5 on every board
// (so it has no move, and no branch: every square one turn back is its own);
// a black pawn walks d5-d4-d3 and White's king e1-d1-e1. 1. Ne4-c5 checks
// nothing on the board - but advanced by a pass, the knight leaps two files
// and one turn back, onto (0T3)a5: the king's place in the past. History
// cannot be blocked; nothing of Black's reaches c5. Checkmate.
GameState history_royal_mate() {
  return game({"[kp1p1/pp2N/5/5/4K:0:1:w]", "[kp1p1/pp2N/5/5/3K1:0:1:b]",
               "[kp3/pp1pN/5/5/3K1:0:2:w]", "[kp3/pp1pN/5/5/4K:0:2:b]",
               "[kp3/pp2N/3p1/5/4K:0:3:w]"},
              k5x5);
}

// E4's C9 positions (test_search.cpp): Black in check on 0, +1 unplayable;
// with a board of Black's in the past, the king's branch rewinds the present
// (escape); with Black's third timeline inactive, it cannot (checkmate).
GameState rewind_escape() {
  return game({"[2k2/5/5/5/5:0:1:b]", "[k4/5/5/5/5:0:2:w]", "[k4/5/5/5/R4:0:2:b]",
               "[5/5/5/5/RR3:+1:2:b]"},
              k5x5);
}
GameState rewind_blocked() {
  return game({"[5/5/5/5/5:-2:3:w]", "[5/5/5/5/5:-1:3:w]", "[2k2/5/5/5/5:0:1:b]",
               "[k4/5/5/5/5:0:2:w]", "[k4/5/5/5/R4:0:2:b]", "[5/5/5/5/RR3:+1:2:b]"},
              k5x5);
}

// Sparse starts (test_detectors.cpp's kind): few pieces, several timelines -
// random play runs into checks, mates and stalemates within a few turns.
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
      {{"[k3/4/1R2/K3:0:1:w]"}, k4},
      {{"[1k3/5/1K3/5/R3R:0:1:w]"}, k5},
  };
}

// A random legal action: random pseudo-legal sequences until one is safe, else
// E4's witness; nullopt if the turn has none.
std::optional<std::vector<Move>> random_legal_action(const GameState& s, t5d_test::Rng& rng) {
  MoveList moves;
  for (int attempt = 0; attempt < 24; ++attempt) {
    GameState t = s;
    std::vector<Move> seq;
    for (int step = 0; step < 64; ++step) {
      if (t.obligations_met() && (!t.rules().moves_after_shift || rng.chance(60))) break;
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

// Positions from random legal play from the sparse starts, under `rules`: up
// to `turns` turn starts per game.
std::vector<GameState> sparse_positions(const RuleOptions& rules, int games, int turns,
                                        std::uint64_t seed) {
  std::vector<GameState> out;
  t5d_test::Rng rng(seed);
  for (const Sparse& start : sparse_starts()) {
    for (int g = 0; g < games; ++g) {
      GameState s = game(start.boards, start.size, rules);
      for (int turn = 0; turn < turns; ++turn) {
        out.push_back(s);
        const std::optional<std::vector<Move>> a = random_legal_action(s, rng);
        if (!a) break;
        for (const Move& m : *a) s.apply(m);
        s.submit();
      }
    }
  }
  return out;
}

// The reference for a move's check test (scanner (b)): make the move, append
// a pass of every board it created, and look for a royal capture from those
// copies among every capture check.hpp finds.
bool checks_reference(const GameState& s, const Move& m) {
  GameState t = s;
  t.apply(m);
  Multiverse mv = t.multiverse();
  std::vector<int> created{m.from_timeline};
  if (m.jump()) {
    created.push_back(!m.branching() ? m.to_timeline
                                     : (s.to_move() == Colour::White ? mv.max_timeline()
                                                                     : mv.min_timeline()));
  }
  std::vector<std::pair<int, int>> places;
  for (const int p : created) {
    BoardBuilder pass(*mv.head(p));
    pass.set_en_passant(kNoSquare);
    mv.append_board(p, pass.publish());
    places.emplace_back(p, mv.timeline(p)->last_half_turn());
  }
  return for_each_royal_capture(mv, s.rules(), s.to_move(), t.inactive_timelines(),
                                [&](const RoyalCapture& c) {
                                  return std::find(places.begin(), places.end(),
                                                   std::make_pair(c.from_timeline,
                                                                  c.from_half_turn)) !=
                                         places.end();
                                });
}

// The alternatives of the engine's ruleset, one parameter each.
std::vector<RuleOptions> alternative_rulesets() {
  std::vector<RuleOptions> out;
  const auto with = [&](auto edit) {
    RuleOptions r;
    edit(r);
    out.push_back(r);
  };
  with([](RuleOptions& r) { r.activation = ActivationRule::Consecutive; });
  with([](RuleOptions& r) { r.initial_credit = InitialTimelineCredit::MinusZeroToBlack; });
  with([](RuleOptions& r) { r.play_activates = true; });
  with([](RuleOptions& r) { r.moves_after_shift = true; });
  with([](RuleOptions& r) { r.mandatory = MandatoryBoards::AllActive; });
  with([](RuleOptions& r) { r.reactivation = Reactivation::Deferred; });
  with([](RuleOptions& r) { r.turn_zero_reachable = false; });
  with([](RuleOptions& r) { r.sliders_cross_void = true; });
  with([](RuleOptions& r) { r.past_self_blocks = false; });
  with([](RuleOptions& r) { r.earlier_playable_jump_branches = true; });
  with([](RuleOptions& r) { r.inactive_history_is_destination = false; });
  with([](RuleOptions& r) { r.brawn_rank_timeline_capture = false; });
  with([](RuleOptions& r) { r.double_step_right = DoubleStepRight::HomeRank; });
  with([](RuleOptions& r) { r.timeline_double_step = false; });
  with([](RuleOptions& r) { r.double_step_over_void = true; });
  with([](RuleOptions& r) { r.branch_copies_keep_unmoved = false; });
  with([](RuleOptions& r) { r.jumps_keep_unmoved = true; });
  with([](RuleOptions& r) { r.en_passant_brawns = false; });
  with([](RuleOptions& r) { r.brawns_promote = false; });
  with([](RuleOptions& r) { r.brawn_cross_board_promotion = false; });
  with([](RuleOptions& r) { r.castling_empty_between = false; });
  with([](RuleOptions& r) { r.castling_destination_test = true; });
  with([](RuleOptions& r) { r.castling_rook_path_test = true; });
  with([](RuleOptions& r) { r.castling_any_rank = false; });
  with([](RuleOptions& r) { r.castling_rook_two_away = false; });
  with([](RuleOptions& r) { r.castling_on_optional_boards = false; });
  return out;
}

}  // namespace

// ---- constructed mates -------------------------------------------------------------------

TEST(scanner_mate_in_one_on_one_board) {
  const GameState s = mate_in_one();
  // The position as described: Qa4 mates (E4's detector, exhaustive).
  const MateResult m = detect_mate(after(s, {"(0T1)Qe4(0T1)a4"}), Detector::Hypercuboid);
  REQUIRE(m.verdict == Outcome::Checkmate);
  const ScanResult r = scan(s);
  CHECK(r.proof == Proof::win(1));
  CHECK(r.score == kScanMate - 1);
  REQUIRE(!r.best_action.empty());
  // The action the scanner chose mates too.
  GameState t = s;
  for (const Move& mv : r.best_action) t.apply(mv);
  t.submit();
  CHECK(detect_mate(t, Detector::Naive).verdict == Outcome::Checkmate);
  GameTree tree;
  CHECK(agrees(r.proof, s, tree));
}

TEST(scanner_mate_in_one_across_timelines) {
  const GameState s = cross_board_mate();
  const MateResult m = detect_mate(after(s, {"(-0T1)Qa4(+0T1)a4"}), Detector::Hypercuboid);
  REQUIRE(m.verdict == Outcome::Checkmate);
  const ScanResult r = scan(s);
  CHECK(r.proof == Proof::win(1));
  GameTree tree;
  CHECK(agrees(r.proof, s, tree));
}

TEST(scanner_mate_in_one_by_stale_checkmate_c8) {
  const GameState s = stale_checkmate_in_one();
  const GameState t = after(s, {"(0T1)Re1(0T1)a1", "(+1T1)Pe2(+1T1)e3"});
  CHECK(in_check(t));
  REQUIRE(detect_mate(t, Detector::Hypercuboid).verdict == Outcome::Checkmate);
  const ScanResult r = scan(s);
  CHECK(r.proof == Proof::win(1));
  GameTree tree;
  CHECK(agrees(r.proof, s, tree));
}

TEST(scanner_mate_in_two_by_checks) {
  const GameState s = mate_in_two();
  GameTree tree;
  const GameValue v = tree.solve(s, 3);
  REQUIRE(v.kind == GameValue::Kind::Win);
  CHECK_EQ(v.distance, 3);
  const ScanResult r = scan(s);
  CHECK(r.proof == Proof::win(3));
  CHECK(agrees(r.proof, s, tree));
  // After 1. Rd4+, Black is mated in two half-turns whatever it does (∀ over
  // every legal action, enumerated by E4).
  const GameState b = after(s, {"(0T1)Rd2(0T1)d4"});
  CHECK(in_check(b));
  const ScanResult rb = scan(b);
  CHECK(rb.proof == Proof::loss(2));
  CHECK(agrees(rb.proof, b, tree));
}

TEST(scanner_mate_in_one_through_time_c2) {
  const GameState s = history_royal_mate();
  const GameState t = after(s, {"(0T3)Ne4(0T3)c5"});
  CHECK(in_check(t));  // the king's place on (0T3), history, is attacked
  REQUIRE(detect_mate(t, Detector::Naive).verdict == Outcome::Checkmate);
  Scanner scanner;
  // The knight's move is a checking move by (b) - nothing on its board does.
  const std::optional<Move> knight = t5d_test::find_move(s, "(0T3)Ne4(0T3)c5");
  REQUIRE(knight.has_value());
  CHECK(scanner.checks(s, *knight));
  const ScanResult r = scanner.scan(s, ScanBudget::nodes(2'000'000));
  CHECK(r.proof == Proof::win(1));
  GameTree tree;
  CHECK(agrees(r.proof, s, tree));
}

TEST(scanner_terminal_positions_are_e4s_verdicts) {
  // Checkmated now: E4's stale-checkmate (C8) and a king lost on the
  // opponent's board ahead of the present (R1); C9's blocked rewind.
  for (const GameState& s :
       {game({"[k4/5/5/5/R4:0:1:b]", "[5/5/5/5/RR3:+1:1:b]"}, k5x5),
        game({"[k4/5/5/5/5:0:1:b]", "[4k/5/5/5/4R:+1:2:w]"}, k5x5), rewind_blocked()}) {
    REQUIRE(detect_mate(s, Detector::Hypercuboid).verdict == Outcome::Checkmate);
    const ScanResult r = scan(s);
    CHECK(r.proof == Proof::loss(0));
    CHECK(r.score == -kScanMate);
  }
  // Stalemate: a threat across timelines only (C6, R1's pass), and a side
  // with no royal (C13).
  for (const GameState& s : {game({"[k4/5/5/5/5:0:1:b]", "[1R3/RR3/5/5/5:+1:2:w]"}, k5x5),
                             game({"[p4/R4/5/5/K4:0:1:b]"}, k5x5)}) {
    REQUIRE(detect_mate(s, Detector::Hypercuboid).verdict == Outcome::Stalemate);
    CHECK(scan(s).proof == Proof::draw());
  }
}

TEST(scanner_a_rewind_escapes_c9_no_false_mate) {
  // In check, the king's branch into the past rewinds the present: four legal
  // actions, every one an escape. No loss may be proven.
  const GameState s = rewind_escape();
  CHECK(in_check(s));
  const ScanResult r = scan(s);
  CHECK(r.proof.kind != Proof::Kind::Loss);
  GameTree tree;
  CHECK(agrees(r.proof, s, tree));
  const GameValue v = tree.solve(s, 1);
  CHECK(v.kind != GameValue::Kind::Loss);
}

namespace {

// Random play from the sparse starts with royal queens for kings, under
// Reversed Royalty (a ruleset ID of its own, H18): every proof is the game
// tree's.
void royal_queen_proofs(int games, int turns, std::uint64_t nodes, int max_distance) {
  RuleOptions rr;
  rr.family = RuleFamily::ReversedRoyalty;
  std::vector<GameState> positions;
  t5d_test::Rng rng(31);
  for (Sparse start : sparse_starts()) {
    for (std::string& b : start.boards) {
      std::replace(b.begin(), b.end(), 'K', 'Y');
      std::replace(b.begin(), b.end(), 'k', 'y');
    }
    for (int g = 0; g < games; ++g) {
      GameState s = game(start.boards, start.size, rr);
      for (int turn = 0; turn < turns; ++turn) {
        positions.push_back(s);
        const std::optional<std::vector<Move>> a = random_legal_action(s, rng);
        if (!a) break;
        for (const Move& m : *a) s.apply(m);
        s.submit();
      }
    }
  }
  ProofTally tally;
  tally.max_distance = max_distance;
  tally.work = max_distance > 3 ? 3000000 : 40000;
  for (const GameState& p : positions) {
    const ScanResult r = scan(p, nodes);
    if (!tally.check(r, p)) {
      std::fprintf(stderr, "  false proof %s at %s\n", to_string(r.proof).c_str(),
                   describe_position(p).c_str());
      CHECK(false);
    }
  }
  tally.print("royal queens");
  CHECK(tally.verified > 0);
}

}  // namespace

TEST(scanner_reversed_royalty_royal_queen) {
  // E4's position: Black's royal queen a5 checked down the a-file; mated.
  RuleOptions rr;
  rr.family = RuleFamily::ReversedRoyalty;
  const GameState s = game({"[yp3/1p3/5/5/RR3:0:1:b]"}, k5x5, rr);
  REQUIRE(detect_mate(s, Detector::Hypercuboid).verdict == Outcome::Checkmate);
  CHECK(scan(s).proof == Proof::loss(0));
  royal_queen_proofs(1, 4, 20'000, 2);
}

TEST(long_scanner_reversed_royalty_royal_queen) { royal_queen_proofs(3, 8, 100'000, 99); }

// ---- the forcing set -------------------------------------------------------------------------

namespace {

// (b) against its definition made literal: the move made, every board it
// created passed, check.hpp's captures filtered to those copies - on random
// legal play from the sparse starts and from three named variants, under the
// default rules and every alternative.
void check_test_equals_the_materialised_pass(int sparse_turns, int variant_turns) {
  int moves = 0;
  int checking = 0;
  std::vector<RuleOptions> rulesets = alternative_rulesets();
  rulesets.insert(rulesets.begin(), RuleOptions{});
  std::uint64_t seed = 11;
  for (const RuleOptions& rules : rulesets) {
    std::vector<GameState> positions = sparse_positions(rules, 1, sparse_turns, seed++);
    t5d_test::Rng rng(seed++);
    for (const char* variant : {"Standard", "Misc - Timeline Battleground", "Standard - Two Timelines"}) {
      const Variant* v = find_variant(variant);
      REQUIRE(v != nullptr);
      GameState s = variant_start(*v, rules);
      for (int turn = 0; turn < variant_turns; ++turn) {
        positions.push_back(s);
        const std::optional<std::vector<Move>> a = random_legal_action(s, rng);
        if (!a) break;
        for (const Move& m : *a) s.apply(m);
        s.submit();
      }
    }
    Scanner scanner;
    for (const GameState& s : positions) {
      MoveList all;
      generate_moves(s, all);
      for (const Move& m : all) {
        ++moves;
        const bool ref = checks_reference(s, m);
        if (ref) ++checking;
        if (scanner.checks(s, m) != ref) {
          std::fprintf(stderr, "  check test differs (reference %d) for %s at %s\n", ref ? 1 : 0,
                       to_string(m, s.multiverse().even()).c_str(), describe_position(s).c_str());
          CHECK(false);
        }
      }
    }
  }
  std::fprintf(stderr, "  %d moves, %d checking\n", moves, checking);
  CHECK(checking > 0);
}

}  // namespace

TEST(scanner_check_test_equals_the_materialised_pass) {
  check_test_equals_the_materialised_pass(3, 5);
}
TEST(long_scanner_check_test_equals_the_materialised_pass) {
  check_test_equals_the_materialised_pass(8, 24);
}

TEST(scanner_quiet_positions_have_no_forcing_move) {
  // The standard start: no capture, no check, no branch at all - quiet.
  Scanner scanner;
  const GameState s(standard_start());
  CHECK(!scanner.has_forcing_move(s));
  CHECK(scanner.forcing_moves(s).empty());
  CHECK_EQ(scanner.evaluate(s), 0);
  // Every completed forcing action is a legal action beginning with its seed.
  for (const GameState& p : sparse_positions(RuleOptions{}, 1, 5, 3)) {
    if (in_check(p)) continue;
    for (const Move& m : scanner.forcing_moves(p)) {
      const std::optional<BuiltAction> a = scanner.complete(p, m);
      if (!a) continue;
      CHECK(a->moves.front() == m);
      GameState t = p;
      for (const Move& x : a->moves) {
        CHECK(is_pseudo_legal(t, x));
        t.apply(x);
      }
      CHECK(t.submission_status() == SubmitStatus::Safe);
      t.submit();
      CHECK(same_position(t, a->successor));
    }
  }
}

// ---- proofs against the brute-force game tree ---------------------------------------------

namespace {

// Random legal play from the sparse starts: every proof the scanner gives is
// the game tree's, within its distance. And where the game tree finds a mate
// in one on a single timeline, so does the scanner (a mate needs a check, and
// on one timeline every check is a move's check, (b)).
void proofs_agree_with_the_game_tree(int games, int turns, std::uint64_t nodes,
                                     int max_distance) {
  ProofTally tally;
  tally.max_distance = max_distance;
  tally.work = max_distance > 3 ? 3000000 : 40000;
  int mates_in_one = 0;
  int found = 0;
  for (const GameState& s : sparse_positions(RuleOptions{}, games, turns, 101)) {
    const ScanResult r = scan(s, nodes);
    if (!tally.check(r, s)) {
      std::fprintf(stderr, "  false proof %s at %s\n", to_string(r.proof).c_str(),
                   describe_position(s).c_str());
      CHECK(false);
    }
    if (s.multiverse().timeline_count() == 1) {
      GameTree tree;
      const GameValue v = tree.solve(s, 1);
      if (v.kind == GameValue::Kind::Win && v.distance == 1) {
        ++mates_in_one;
        if (r.proof == Proof::win(1)) {
          ++found;
        } else {
          std::fprintf(stderr, "  mate in one missed (%s) at %s\n", to_string(r.proof).c_str(),
                       describe_position(s).c_str());
          CHECK(false);
        }
      }
    }
  }
  tally.print("random play");
  std::fprintf(stderr, "  %d of %d single-timeline mates in one found\n", found, mates_in_one);
  CHECK(tally.verified > 0);
  CHECK(mates_in_one > 0);
}

// H37's spirit: whatever the node budget, a proof is a proof. Random budgets
// on random play and the constructed mates; every proof checked against the
// game tree, and against the generous budget's answer (a proof never changes
// kind).
void no_false_proofs_under_random_budgets(int games, int turns, int runs_per_position,
                                          int most_nodes, int max_distance) {
  t5d_test::Rng rng(77);
  std::vector<GameState> positions = sparse_positions(RuleOptions{}, games, turns, 202);
  for (const GameState& s : {mate_in_one(), cross_board_mate(), stale_checkmate_in_one(),
                             mate_in_two(), history_royal_mate(), rewind_escape()}) {
    positions.push_back(s);
  }
  ProofTally tally;
  tally.max_distance = max_distance;
  tally.work = max_distance > 3 ? 3000000 : 40000;
  int runs = 0;
  for (const GameState& s : positions) {
    const ScanResult full = scan(s, 100'000);
    for (int i = 0; i < runs_per_position; ++i) {
      const auto budget = static_cast<std::uint64_t>(rng.below(most_nodes));
      const ScanResult r = scan(s, budget);
      ++runs;
      CHECK(r.stats.nodes <= budget);
      if (!tally.check(r, s)) {
        std::fprintf(stderr, "  false proof %s at budget %llu at %s\n", to_string(r.proof).c_str(),
                     static_cast<unsigned long long>(budget), describe_position(s).c_str());
        CHECK(false);
      }
      if (r.proof.proven() && full.proof.proven()) CHECK(full.proof.kind == r.proof.kind);
    }
  }
  tally.print("budget-limited scans");
  std::fprintf(stderr, "  %d scans\n", runs);
  CHECK(tally.verified > 0);
}

// A sample of positions under each alternative of the ruleset (the
// parameters of t5d/rules.hpp): every proof agrees with the game tree under
// the same rules.
void proofs_under_every_alternative_ruleset(int games, int turns, std::uint64_t nodes,
                                            int max_distance) {
  ProofTally tally;
  tally.max_distance = max_distance;
  tally.work = max_distance > 3 ? 3000000 : 40000;
  std::uint64_t seed = 500;
  for (const RuleOptions& rules : alternative_rulesets()) {
    for (const GameState& s : sparse_positions(rules, games, turns, seed++)) {
      const ScanResult r = scan(s, nodes);
      if (!tally.check(r, s)) {
        std::fprintf(stderr, "  false proof %s under %s at %s\n", to_string(r.proof).c_str(),
                     describe(rules).c_str(), describe_position(s).c_str());
        CHECK(false);
      }
    }
  }
  tally.print("alternative rulesets");
  CHECK(tally.verified > 0);
}

}  // namespace

namespace {

// The in-check stage (ScannerOptions::evasion_probe) against E4 first (0):
// random play from the sparse starts under the default rules and under Q4's
// ruled reading (moves after the shift, rulebook R8), and the constructed
// positions, at a generous budget and at random ones. Every proof of either
// is the game tree's; where both prove, they prove the same; and with ample
// nodes the stage loses no proof E4-first finds in a tree it exhausts.
void evasion_probe_keeps_the_proofs(int games, int turns, int runs_per_position, int most_nodes,
                                    int max_distance) {
  RuleOptions after_shift;
  after_shift.moves_after_shift = true;
  std::vector<GameState> positions = sparse_positions(RuleOptions{}, games, turns, 303);
  for (const GameState& s : sparse_positions(after_shift, games, turns, 304)) positions.push_back(s);
  for (const GameState& s : {mate_in_one(), cross_board_mate(), stale_checkmate_in_one(),
                             mate_in_two(), history_royal_mate(), rewind_escape(), rewind_blocked()}) {
    positions.push_back(s);
  }
  ProofTally tally;
  tally.max_distance = max_distance;
  tally.work = max_distance > 3 ? 3000000 : 40000;
  ScannerOptions first;
  first.evasion_probe = 0;
  ScannerOptions narrow;
  narrow.evasion_probe = 1;
  t5d_test::Rng rng(91);
  int both = 0;
  int one_way = 0;
  int in_check_positions = 0;
  std::uint64_t probe_cutoffs = 0;
  std::uint64_t probe_hits = 0;
  for (const GameState& s : positions) {
    if (in_check(s)) ++in_check_positions;
    const ScanResult a = scan(s, 100'000, first);
    const ScanResult b = scan(s, 100'000);
    const ScanResult c = scan(s, 100'000, narrow);
    probe_cutoffs += b.stats.probe_cutoffs;
    probe_hits += b.stats.horizon_probe_hits;
    for (const ScanResult* r : {&a, &b, &c}) {
      if (!tally.check(*r, s)) {
        std::fprintf(stderr, "  false proof %s at %s\n", to_string(r->proof).c_str(),
                     describe_position(s).c_str());
        CHECK(false);
      }
    }
    for (const ScanResult* r : {&b, &c}) {
      if (a.proof.proven() && r->proof.proven()) {
        ++both;
        CHECK(a.proof.kind == r->proof.kind);
      } else if (a.proof.proven() != r->proof.proven()) {
        ++one_way;  // alpha-beta's windows differ with the order: information
      }
    }
    for (int i = 0; i < runs_per_position; ++i) {
      const auto budget = static_cast<std::uint64_t>(rng.below(most_nodes));
      const ScanResult r = scan(s, budget);
      CHECK(r.stats.nodes <= budget);
      if (!tally.check(r, s)) {
        std::fprintf(stderr, "  false proof %s at budget %llu at %s\n", to_string(r.proof).c_str(),
                     static_cast<unsigned long long>(budget), describe_position(s).c_str());
        CHECK(false);
      }
      if (r.proof.proven() && b.proof.proven()) CHECK(r.proof.kind == b.proof.kind);
    }
  }
  tally.print("evasion probe");
  std::fprintf(stderr,
               "  %zu positions (%d in check), %d proofs found both ways, %d one way only, %llu "
               "probe cutoffs, %llu horizon tests answered by the probe\n",
               positions.size(), in_check_positions, both, one_way,
               static_cast<unsigned long long>(probe_cutoffs),
               static_cast<unsigned long long>(probe_hits));
  CHECK(tally.verified > 0);
  CHECK(both > 0);
  CHECK(probe_cutoffs > 0);
  CHECK(probe_hits > 0);
}

}  // namespace

TEST(scanner_evasion_probe_keeps_the_proofs) { evasion_probe_keeps_the_proofs(1, 4, 1, 3000, 1); }
TEST(long_scanner_evasion_probe_keeps_the_proofs) {
  evasion_probe_keeps_the_proofs(2, 8, 3, 20000, 5);
}

TEST(scanner_evasion_probe_off_is_e5b) {
  // evasion_probe 0 is E5b's scanner, the in-check stage left out: in check,
  // E4's enumeration first, every time, and E4's detector at the horizon.
  ScannerOptions first;
  first.evasion_probe = 0;
  for (const GameState& s : sparse_positions(RuleOptions{}, 1, 6, 404)) {
    const ScanResult r = scan(s, 20'000, first);
    CHECK_EQ(r.stats.probe_seeds, std::uint64_t{0});
    CHECK_EQ(r.stats.evasion_lists, r.stats.evasion_visits);
  }
}

TEST(scanner_proofs_agree_with_the_game_tree) { proofs_agree_with_the_game_tree(1, 6, 20'000, 3); }
TEST(long_scanner_proofs_agree_with_the_game_tree) {
  proofs_agree_with_the_game_tree(4, 12, 200'000, 99);
}
TEST(scanner_no_false_proofs_under_random_budgets) {
  no_false_proofs_under_random_budgets(1, 6, 3, 3000, 3);
}
TEST(long_scanner_no_false_proofs_under_random_budgets) {
  no_false_proofs_under_random_budgets(3, 10, 8, 20000, 99);
}
TEST(scanner_proofs_under_every_alternative_ruleset) {
  proofs_under_every_alternative_ruleset(1, 3, 10'000, 1);
}
TEST(long_scanner_proofs_under_every_alternative_ruleset) {
  proofs_under_every_alternative_ruleset(3, 8, 100'000, 99);
}

// ---- the solved store, the TT, determinism -----------------------------------------------

TEST(scanner_proofs_export_at_once_and_survive_tt_churn_h19) {
  // A one-entry TT - every store evicts the last - finds the same proofs as a
  // large one, and every proof is in the solved store when the scan returns.
  for (const GameState& s : {mate_in_one(), mate_in_two(), stale_checkmate_in_one(),
                             history_royal_mate()}) {
    ScannerOptions tiny;
    tiny.tt_entries = 1;
    MemorySolvedStore store;
    Scanner churned(tiny, &store);
    const ScanResult a = churned.scan(s, ScanBudget::nodes(2'000'000));
    const ScanResult b = scan(s);
    CHECK(a.proof.proven());
    CHECK(a.proof == b.proof);
    CHECK(store.find(s) == std::optional<Proof>(a.proof));
    CHECK(a.stats.proofs_exported >= 1);
    // A fresh scanner on the same store - an empty TT - answers from the
    // store at once.
    Scanner again(ScannerOptions{}, &store);
    const ScanResult c = again.scan(s, ScanBudget::nodes(2'000'000));
    CHECK(c.proof == a.proof);
    CHECK_EQ(c.stats.solved_hits, std::uint64_t{1});
    CHECK(c.stats.nodes == 0);
  }
  // Every stored proof is the game tree's.
  MemorySolvedStore store;
  Scanner scanner(ScannerOptions{}, &store);
  scanner.scan(mate_in_two(), ScanBudget::nodes(2'000'000));
  GameTree tree;
  for (const MemorySolvedStore::Entry& e : store.entries()) {
    CHECK(agrees(e.proof, e.position, tree));
  }
  CHECK(store.size() > 1);
}

TEST(scanner_the_solved_store_refuses_contradictions) {
  MemorySolvedStore store;
  const GameState s = mate_in_one();
  store.put(s, Proof::win(3));
  store.put(s, Proof::win(1));  // shorter: replaces
  CHECK(store.find(s) == std::optional<Proof>(Proof::win(1)));
  store.put(s, Proof::win(5));  // longer: kept as it was
  CHECK(store.find(s) == std::optional<Proof>(Proof::win(1)));
  CHECK_THROWS_AS(store.put(s, Proof::loss(2)), std::logic_error);
  // The ruleset is part of the key (H18): the same boards under another
  // ruleset are another position.
  RuleOptions other;
  other.moves_after_shift = true;
  const GameState t = game({"[k4/4Q/1K3/5/5:0:1:w]"}, k5x5, other);
  CHECK(!store.find(t).has_value());
}

TEST(scanner_is_deterministic_h15) {
  // The synchronous mode: the same position, options and budget give the
  // same search, node for node.
  for (const GameState& s : sparse_positions(RuleOptions{}, 1, 6, 9)) {
    const ScanResult a = scan(s, 20'000);
    const ScanResult b = scan(s, 20'000);
    CHECK_EQ(a.stats.nodes, b.stats.nodes);
    CHECK_EQ(a.stats.positions, b.stats.positions);
    CHECK_EQ(a.score, b.score);
    CHECK(a.proof == b.proof);
    CHECK(a.best_action == b.best_action);
    REQUIRE_EQ(a.iterations.size(), b.iterations.size());
    for (std::size_t i = 0; i < a.iterations.size(); ++i) {
      CHECK_EQ(a.iterations[i].nodes, b.iterations[i].nodes);
      CHECK_EQ(a.iterations[i].score, b.iterations[i].score);
    }
  }
}

TEST(scanner_budget_and_iterations) {
  // The node cap holds; iterations are cumulative; a smaller budget follows
  // the larger one's path and stops inside it (the 6b tool's reading of one
  // scan at several budgets rests on it).
  const GameState s = mate_in_two();
  const ScanResult big = scan(s, 2'000'000);
  REQUIRE(!big.iterations.empty());
  for (std::size_t i = 1; i < big.iterations.size(); ++i) {
    CHECK(big.iterations[i].nodes >= big.iterations[i - 1].nodes);
  }
  for (const IterationStats& it : big.iterations) {
    if (!it.completed) continue;
    const ScanResult small = scan(s, it.nodes);
    CHECK(small.completed_depth >= it.depth);
    if (it.nodes > 0) {
      const ScanResult less = scan(s, it.nodes - 1);
      CHECK(less.completed_depth < it.depth || less.proof.proven());
    }
  }
  // Zero nodes: nothing completes, no proof.
  const ScanResult none = scan(s, 0);
  CHECK_EQ(none.completed_depth, 0);
  CHECK(!none.proof.proven());
}

// ---- Lazy SMP (H17) -----------------------------------------------------------------------------

namespace {

// Positions of random legal play from named variants (larger trees than the
// sparse starts: the scans do not finish at once).
std::vector<GameState> variant_positions(const RuleOptions& rules, int actions_from, int actions_to,
                                         std::uint64_t seed) {
  std::vector<GameState> out;
  t5d_test::Rng rng(seed);
  for (const char* name : {"Standard", "Standard - Two Timelines"}) {
    const Variant* v = find_variant(name);
    REQUIRE(v != nullptr);
    GameState s = variant_start(*v, rules);
    for (int a = 0; a < actions_to; ++a) {
      if (a >= actions_from) out.push_back(s);
      const std::optional<std::vector<Move>> act = random_legal_action(s, rng);
      if (!act) break;
      for (const Move& m : *act) s.apply(m);
      s.submit();
    }
  }
  return out;
}

// Every proof a scan with 2-8 threads gives is the game tree's; where the
// single thread proves too, the same kind. One solved store per thread count
// is shared by all its scans, so a contradiction between any two would
// throw. Node budgets and a clock.
void lazy_smp_no_false_proofs(int games, int turns, int max_distance) {
  RuleOptions after_shift;
  after_shift.moves_after_shift = true;
  std::vector<GameState> positions = sparse_positions(RuleOptions{}, games, turns, 505);
  for (const GameState& s : sparse_positions(after_shift, games, turns, 506)) positions.push_back(s);
  for (const GameState& s : {mate_in_one(), cross_board_mate(), stale_checkmate_in_one(),
                             mate_in_two(), history_royal_mate(), rewind_escape(), rewind_blocked()}) {
    positions.push_back(s);
  }
  std::vector<ScanResult> single;
  for (const GameState& s : positions) single.push_back(scan(s, 30'000));
  ProofTally tally;
  tally.max_distance = max_distance;
  tally.work = max_distance > 3 ? 3000000 : 40000;
  int agreeing = 0;
  for (const int threads : {2, 3, 4, 8}) {
    ScannerOptions o;
    o.threads = threads;
    ConcurrentSolvedStore store;
    for (std::size_t i = 0; i < positions.size(); ++i) {
      const GameState& s = positions[i];
      for (const bool clock : {false, true}) {
        Scanner scanner(o, &store);
        ScanBudget b = ScanBudget::nodes(30'000);
        if (clock) b.max_time = std::chrono::milliseconds(5);
        const ScanResult r = scanner.scan(s, b);
        CHECK_EQ(r.threads, threads);
        if (!clock) {
          CHECK(r.stats.nodes <= static_cast<std::uint64_t>(threads) * (30'000 + 256));
        }
        if (!tally.check(r, s)) {
          std::fprintf(stderr, "  false proof %s with %d threads at %s\n", to_string(r.proof).c_str(),
                       threads, describe_position(s).c_str());
          CHECK(false);
        }
        if (r.proof.proven() && single[i].proof.proven()) {
          ++agreeing;
          CHECK(r.proof.kind == single[i].proof.kind);
        }
      }
    }
    for (std::size_t i = 0; i < positions.size(); ++i) {
      // Every proof in the shared store is a proof of the single thread's kind.
      const std::optional<Proof> p = store.find(positions[i]);
      if (p && single[i].proof.proven()) CHECK(p->kind == single[i].proof.kind);
    }
  }
  tally.print("Lazy SMP");
  std::fprintf(stderr, "  %zu positions x 4 thread counts x 2 budgets, %d proofs agreeing with one thread's\n",
               positions.size(), agreeing);
  CHECK(tally.verified > 0);
  CHECK(agreeing > 0);
}

}  // namespace

TEST(scanner_lazy_smp_no_false_proofs) { lazy_smp_no_false_proofs(1, 4, 1); }
TEST(long_scanner_lazy_smp_no_false_proofs) { lazy_smp_no_false_proofs(2, 8, 5); }

TEST(scanner_lazy_smp_store_refuses_contradictions) {
  // The shared store's contract, as MemorySolvedStore's.
  ConcurrentSolvedStore store;
  const GameState s = mate_in_one();
  CHECK(!store.find(s).has_value());
  store.put(s, Proof::win(3));
  store.put(s, Proof::win(1));
  CHECK(store.find(s) == std::optional<Proof>(Proof::win(1)));
  store.put(s, Proof::win(5));
  CHECK(store.find(s) == std::optional<Proof>(Proof::win(1)));
  CHECK_THROWS_AS(store.put(s, Proof::loss(2)), std::logic_error);
  CHECK_THROWS_AS(store.put(s, Proof::draw()), std::logic_error);
  RuleOptions other;
  other.moves_after_shift = true;
  CHECK(!store.find(game({"[k4/4Q/1K3/5/5:0:1:w]"}, k5x5, other)).has_value());  // H18
  // Eight threads at once: each puts a proof of one kind for every position,
  // then each tries the other kind for every position - every such put
  // throws, and nothing it did changes the store.
  std::vector<GameState> positions;
  for (const GameState& p : sparse_positions(RuleOptions{}, 1, 6, 606)) {
    if (std::none_of(positions.begin(), positions.end(),
                     [&](const GameState& t) { return same_position(p, t); })) {
      positions.push_back(p);
    }
  }
  ConcurrentSolvedStore shared;
  std::atomic<int> refused{0};
  std::atomic<int> errors{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < 8; ++t) {
    pool.emplace_back([&, t] {
      for (std::size_t i = 0; i < positions.size(); ++i) {
        const Proof p = i % 2 == 0 ? Proof::win(1 + static_cast<int>(i % 5) + t)
                                   : Proof::loss(static_cast<int>(i % 3) + t);
        try {
          shared.put(positions[i], p);
        } catch (const std::logic_error&) {
          errors.fetch_add(1);
        }
      }
      for (std::size_t i = 0; i < positions.size(); ++i) {
        const Proof q = i % 2 == 0 ? Proof::loss(1) : Proof::win(1);
        try {
          shared.put(positions[i], q);
        } catch (const std::logic_error&) {
          refused.fetch_add(1);
        }
      }
    });
  }
  for (std::thread& t : pool) t.join();
  CHECK_EQ(errors.load(), 0);
  CHECK_EQ(refused.load(), 8 * static_cast<int>(positions.size()));
  for (std::size_t i = 0; i < positions.size(); ++i) {
    const std::optional<Proof> p = shared.find(positions[i]);
    REQUIRE(p.has_value());
    CHECK(p->kind == (i % 2 == 0 ? Proof::Kind::Win : Proof::Kind::Loss));
    // The shortest distance any thread put.
    CHECK_EQ(p->distance, i % 2 == 0 ? 1 + static_cast<int>(i % 5) : static_cast<int>(i % 3));
  }
  CHECK_EQ(shared.size(), positions.size());
  // A store not safe for threads, shared by a scanner's threads: its calls
  // are serialised, and the proofs land in it.
  MemorySolvedStore plain;
  ScannerOptions four;
  four.threads = 4;
  Scanner scanner(four, &plain);
  const ScanResult r = scanner.scan(mate_in_two(), ScanBudget::nodes(2'000'000));
  CHECK(r.proof == Proof::win(3));
  CHECK(plain.find(mate_in_two()) == std::optional<Proof>(Proof::win(3)));
  GameTree tree;
  for (const MemorySolvedStore::Entry& e : plain.entries()) CHECK(agrees(e.proof, e.position, tree));
}

TEST(scanner_lazy_smp_reaches_the_single_threads_depth) {
  // The same wall time: 5 threads (M13's scanner budget) resolve at least the
  // depth one thread does, summed over a sample (a proof or an exhausted tree
  // counts as the deepest depth). Timing on a shared machine is noisy: one
  // retry.
  std::vector<GameState> sample = variant_positions(RuleOptions{}, 6, 14, 707);
  RuleOptions after_shift;
  after_shift.moves_after_shift = true;
  for (const GameState& s : variant_positions(after_shift, 6, 14, 708)) sample.push_back(s);
  const auto depth_of = [](const ScanResult& r) {
    return r.proof.proven() || r.exhausted ? 17 : r.resolved_depth;
  };
  std::uint64_t shared_roots = 0;
  const auto measure = [&](int threads) {
    int total = 0;
    ScannerOptions o;
    o.threads = threads;
    for (const GameState& s : sample) {
      Scanner scanner(o);
      ScanBudget b;
      b.max_time = std::chrono::milliseconds(40);
      const ScanResult r = scanner.scan(s, b);
      total += depth_of(r);
      shared_roots += r.stats.shared_roots;
    }
    return total;
  };
  int one = measure(1);
  int five = measure(5);
  if (five < one) {
    std::fprintf(stderr, "  retry: 5 threads %d < 1 thread %d\n", five, one);
    one = measure(1);
    five = measure(5);
  }
  std::fprintf(stderr, "  %zu positions at 40 ms: resolved depth summed, 1 thread %d, 5 threads %d\n",
               sample.size(), one, five);
  CHECK(five >= one);
  // Every helper searched its own copy of the root (count-rule activity: the
  // state is rebuilt from its boards).
  CHECK_EQ(shared_roots, std::uint64_t{0});
}

// ---- stand-pat validity (6b) -----------------------------------------------------------------

TEST(scanner_stand_pat_validity) {
  // The standard start: quiet, and an action that loses nothing exists.
  const StandPatVerdict v = stand_pat_validity(GameState(standard_start()), ScannerOptions{});
  CHECK(v.kind == StandPatVerdict::Kind::Valid);
  CHECK(v.holding > 0);
  // In check: not asked.
  CHECK(stand_pat_validity(rewind_escape(), ScannerOptions{}).kind ==
        StandPatVerdict::Kind::NotQuiet);
  // Zugzwang - the obligation to move: Black's only piece, a knight a5, on
  // the only board, must move, to b3 (under the pawn a2) or c4 (under d3).
  // Quiet - it captures nothing and threatens nothing - and every action
  // loses the knight: invalid, exhaustively.
  const GameState z = game({"[n4/5/3P1/P4/4K:0:1:b]"}, k5x5);
  const StandPatVerdict zv = stand_pat_validity(z, ScannerOptions{});
  CHECK(zv.kind == StandPatVerdict::Kind::Invalid);
  CHECK(zv.exhaustive);
  CHECK_EQ(zv.actions, 2);
  CHECK_EQ(zv.holding, 0);
  CHECK_EQ(zv.eval, 100);
}
