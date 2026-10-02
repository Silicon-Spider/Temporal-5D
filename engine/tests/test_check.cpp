// Check status and submission legality (rulebook ruling R1, register H36; C1-C7,
// C13; R6 for castling). Every position is written here from the rulebook -
// never taken from a game corpus or another implementation's test data - and
// what the rules say of it is derived in the comments.
//
// 5x5 boards unless said otherwise; 5DFEN rows run from rank 5 down to rank 1.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/fen.hpp"
#include "t5d/movegen.hpp"
#include "t5d/pgn.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"
#include "test_support.hpp"

using namespace t5d;
using t5d_test::game;
using t5d_test::play;

namespace {

constexpr BoardSize k5x5{5, 5};
constexpr BoardSize k8x8{8, 8};

// (from L, from h, from square, to L, to h, to square, branching)
using CaptureKey = std::tuple<int, int, int, int, int, int, bool>;

CaptureKey key_of(const RoyalCapture& c) {
  return {c.from_timeline, c.from_half_turn, c.from_square,
          c.to_timeline,   c.to_half_turn,   c.to_square, c.branching};
}

// The reference: the move generator's own moves (append_moves_of), those that
// capture a royal piece.
std::set<CaptureKey> generated_royal_captures(const Multiverse& mv, const RuleOptions& rules,
                                              Colour attacker, std::span<const int> inactive) {
  std::set<CaptureKey> out;
  MoveList moves;
  for (const TimelineRef& t : mv.timelines()) {
    moves.clear();
    append_moves_of(mv, rules, attacker, t->position(), moves, inactive);
    for (const Move& m : moves) {
      if (m.capture() && !m.captured.empty() && m.captured.royal()) {
        out.insert({m.from_timeline, m.from_half_turn, m.from_square, m.to_timeline,
                    m.to_half_turn, m.to_square, m.branching()});
      }
    }
  }
  return out;
}

std::set<CaptureKey> scanned_royal_captures(const Multiverse& mv, const RuleOptions& rules,
                                            Colour attacker, std::span<const int> inactive,
                                            std::vector<RoyalCapture>* all = nullptr) {
  std::set<CaptureKey> out;
  for_each_royal_capture(mv, rules, attacker, inactive, [&](const RoyalCapture& c) {
    out.insert(key_of(c));
    if (all != nullptr) all->push_back(c);
    return false;
  });
  return out;
}

// A random multiverse: 1-5 contiguous timelines, each 1-4 boards from a random
// half-turn, random pieces with royals plentiful.
Multiverse random_multiverse(t5d_test::Rng& rng) {
  const BoardSize size{rng.between(2, 6), rng.between(2, 6)};
  const bool even = rng.chance(50);
  Multiverse mv(size, even);
  const int count = rng.between(1, 5);
  const int first = -rng.below(count);
  for (int i = 0; i < count; ++i) {
    NewTimeline nt;
    nt.position = first + i;
    nt.first_half_turn = rng.between(0, 6);
    const int length = rng.between(1, 4);
    for (int k = 0; k < length; ++k) {
      BoardBuilder b(size);
      for (int sq = 0; sq < 64; ++sq) {
        if (!size.contains_square(sq) || !rng.chance(28)) continue;
        Piece piece = t5d_test::random_piece(rng);
        if (rng.chance(25)) {
          piece = Piece(rng.chance(70) ? PieceKind::King : PieceKind::RoyalQueen, piece.colour());
        }
        b.put(sq, piece);
      }
      nt.boards.push_back(b.publish());
    }
    if (nt.position != 0 && !(even && nt.position == -1)) {
      nt.creator = nt.position > 0 ? Colour::White : Colour::Black;
      nt.creation_order = 1;
    }
    mv.add_timeline(std::move(nt));
  }
  return mv;
}

// Passes, by the test's own reading of R1, every active board in the present
// of the side to move: an independent construction of the hypothetical.
Multiverse reference_passed(const GameState& s) {
  Multiverse mv = s.multiverse();
  for (const TimelineRef& t : s.multiverse().timelines()) {
    const int p = t->position();
    if (s.active(p) && t->last_half_turn() == s.present() &&
        to_move_at(t->last_half_turn()) == s.to_move()) {
      BoardBuilder b(*t->head());
      b.set_en_passant(kNoSquare);
      mv.append_board(p, b.publish());
    }
  }
  return mv;
}

}  // namespace

// ---- the scan equals the move generator ----------------------------------------------

TEST(check_royal_captures_equal_the_move_generators) {
  // On random multiverses, under the default rules and every option the
  // generator reads per probe or per slide (Q8, Q14, Q18, Q19, Q20, J3), the
  // scan finds exactly the royal captures among the generator's moves, with
  // the same branching flag - for both colours.
  std::vector<RuleOptions> rulesets(7);
  rulesets[1].sliders_cross_void = true;
  rulesets[2].past_self_blocks = false;
  rulesets[3].turn_zero_reachable = false;
  rulesets[4].inactive_history_is_destination = false;
  rulesets[5].brawn_rank_timeline_capture = false;
  rulesets[6].earlier_playable_jump_branches = true;
  t5d_test::Rng rng(2026);
  long long captures = 0;
  long long multiverses_with_one = 0;
  for (int round = 0; round < 1500; ++round) {
    const Multiverse mv = random_multiverse(rng);
    std::vector<int> inactive;
    for (const TimelineRef& t : mv.timelines()) {
      if (rng.chance(30)) inactive.push_back(t->position());
    }
    const RuleOptions& rules = rulesets[static_cast<std::size_t>(round) % rulesets.size()];
    for (const Colour c : {Colour::White, Colour::Black}) {
      std::vector<RoyalCapture> all;
      const std::set<CaptureKey> scanned = scanned_royal_captures(mv, rules, c, inactive, &all);
      const std::set<CaptureKey> generated = generated_royal_captures(mv, rules, c, inactive);
      CHECK(scanned == generated);
      CHECK_EQ(scanned.size(), all.size());  // each capture reported once
      CHECK_EQ(royal_capture_exists(mv, rules, c, inactive), !generated.empty());
      const std::optional<RoyalCapture> first = find_royal_capture(mv, rules, c, inactive);
      CHECK_EQ(first.has_value(), !generated.empty());
      if (first) CHECK(*first == all.front());
      captures += static_cast<long long>(all.size());
      if (!all.empty()) ++multiverses_with_one;
      // Every capture's places: the attacker first, the royal last, the
      // squares crossed empty (or void, or past selves under Q19's option).
      for (const RoyalCapture& rc : all) {
        const std::vector<CapturePlace> places = capture_places(rc);
        REQUIRE_EQ(places.size(), static_cast<std::size_t>(rc.steps) + 1);
        const Board* from = mv.board_at(places.front().timeline, places.front().half_turn);
        REQUIRE(from != nullptr);
        CHECK(from->at(places.front().square) == rc.attacker);
        const Board* to = mv.board_at(places.back().timeline, places.back().half_turn);
        REQUIRE(to != nullptr);
        CHECK(to->at(places.back().square) == rc.royal);
        CHECK(rc.royal.royal());
        CHECK(rc.royal.colour() != c);
        for (std::size_t i = 1; i + 1 < places.size(); ++i) {
          const Board* b = mv.board_at(places[i].timeline, places[i].half_turn);
          if (b == nullptr) {
            CHECK(rules.sliders_cross_void);
            continue;
          }
          const Piece q = b->at(places[i].square);
          CHECK(q.empty() || (!rules.past_self_blocks && q.colour() == c &&
                              q.kind() == rc.attacker.kind()));
        }
      }
    }
  }
  CHECK(captures > 2000);
  CHECK(multiverses_with_one > 500);
}

// ---- check status (R1) ----------------------------------------------------------------

TEST(check_status_on_one_board) {
  // Black to move on (0T1); a white rook on e1, the black king on e5. Passing
  // (0T1) gives White (0T2) with the same pieces: Re1xe5. In check.
  const GameState s = game({"[4k/5/5/5/4R:0:1:b]"}, k5x5);
  CHECK(s.to_move() == Colour::Black);
  CHECK(passed_timelines(s) == std::vector<int>{0});
  CHECK(in_check(s));
  const std::optional<RoyalCapture> w = check_witness(s);
  REQUIRE(w.has_value());
  CHECK(w->attacker == Piece(PieceKind::Rook, Colour::White));
  CHECK_EQ(w->from_half_turn, half_turn(2, Colour::White));  // from the passed copy
  CHECK(!w->branching);
  // A black pawn on e3 blocks the file: not in check.
  CHECK(!in_check(game({"[4k/5/4p/5/4R:0:1:b]"}, k5x5)));
  // The passed copy is the head's, unchanged, with White to move.
  const Multiverse passed = passed_multiverse(s);
  REQUIRE(passed.timeline(0)->last_half_turn() == half_turn(2, Colour::White));
  const Board* copy = passed.head(0);
  CHECK(copy->at(4, 4) == Piece(PieceKind::King, Colour::Black));
  CHECK(copy->at(4, 0) == Piece(PieceKind::Rook, Colour::White));
  CHECK_EQ(copy->piece_count(), 2);
}

TEST(check_from_a_royal_on_a_history_board_c2) {
  // C2: the black king stood on e5 at (0T2) (White's board there); it has
  // since gone to a1, and a white rook stands on e5 of the playable (0T3)
  // black board. Passed, (0T4) gives White the rook on e5: it slides back in
  // time along e5 - (0T3)e5 is empty - and takes the king's past self on
  // (0T2)e5, a capture that would branch. In check.
  const std::vector<std::string> boards = {"[4k/5/5/5/5:0:2:w]", "[4k/5/5/5/5:0:2:b]",
                                           "[5/5/5/5/k4:0:3:w]", "[4R/5/5/5/k4:0:3:b]"};
  const GameState s = game(boards, k5x5);
  CHECK(s.to_move() == Colour::Black);
  CHECK(in_check(s));
  const std::optional<RoyalCapture> w = check_witness(s);
  REQUIRE(w.has_value());
  CHECK(w->branching);
  CHECK_EQ(w->to_half_turn, half_turn(2, Colour::White));
  CHECK_EQ(w->steps, 2);
  // Without the pass the rook cannot move: its board is Black's (C3).
  CHECK(royals_safe(s));
  // Something on (0T3)e5 of the white board blocks the slide (J7).
  std::vector<std::string> blocked = boards;
  blocked[2] = "[4p/5/5/5/k4:0:3:w]";
  CHECK(!in_check(game(blocked, k5x5)));
}

namespace {

// Black to move at (0T1). Timeline +1, White's first (active, T10), has a
// board ahead of the present: (+1T2), Black to move - optional for Black (T6).
// On it a white rook on e1 and the black king on e5 (and a black pawn a4).
GameState optional_board_ahead() {
  return game({"[k4/5/5/5/5:0:1:b]", "[4k/p4/5/5/4R:+1:2:b]"}, k5x5);
}

// The same threat on an INACTIVE timeline: +2, White's second while Black has
// created none (T10). +1 (active) has a White board ahead, empty.
GameState inactive_board() {
  return game({"[k4/5/5/5/5:0:1:b]", "[5/5/5/5/5:+1:2:w]", "[4k/p4/5/5/4R:+2:1:b]"}, k5x5);
}

}  // namespace

TEST(check_threats_on_optional_boards_ahead_are_not_check_r1) {
  // R1: optional boards ahead of the present are not passed; the rook on
  // (+1T2) can move only if Black plays that board first (C7). Not check.
  GameState s = optional_board_ahead();
  REQUIRE(s.active(1));
  CHECK(s.mandatory_timelines() == std::vector<int>{0});
  CHECK(passed_timelines(s) == std::vector<int>{0});
  CHECK(!in_check(s));
  // Playing only the mandatory board: safe.
  GameState quiet = s;
  play(quiet, "(0T1)Ka5(0T1)a4");
  CHECK(quiet.submission_status() == SubmitStatus::Safe);
  // Playing the optional board too (first: Q4's default) without moving the
  // king hands White the board with the rook on it: the submission is illegal
  // (C4) - the threat counts once the opponent can move there.
  GameState loud = s;
  play(loud, "(+1T2)Pa4(+1T2)a3");
  play(loud, "(0T1)Ka5(0T1)a4");
  CHECK(loud.obligations_met());
  CHECK(loud.submission_status() == SubmitStatus::Unsafe);
  CHECK(!royals_safe(loud));
  const std::optional<RoyalCapture> w = submission_witness(loud);
  REQUIRE(w.has_value());
  CHECK_EQ(w->from_timeline, 1);
  const GameState before = loud;
  CHECK_THROWS_WITH(loud.submit(), std::logic_error, "C4");
  CHECK(loud == before);
  // Moving the king out of the rook's line on +1 instead: safe.
  GameState away = s;
  play(away, "(+1T2)Ke5(+1T2)d5");
  play(away, "(0T1)Ka5(0T1)a4");
  CHECK(away.submission_status() == SubmitStatus::Safe);
}

TEST(check_threats_on_inactive_timelines_are_not_check_r1) {
  // R1: boards on inactive timelines are not passed - but a capture from one
  // makes a submission illegal once the opponent can move there.
  GameState s = inactive_board();
  REQUIRE(!s.active(2));
  CHECK(passed_timelines(s) == std::vector<int>{0});
  CHECK(!in_check(s));
  GameState quiet = s;
  play(quiet, "(0T1)Ka5(0T1)a4");
  CHECK(quiet.submission_status() == SubmitStatus::Safe);
  GameState loud = s;
  play(loud, "(+2T1)Pa4(+2T1)a3");
  play(loud, "(0T1)Ka5(0T1)a4");
  CHECK(loud.submission_status() == SubmitStatus::Unsafe);
}

TEST(check_threats_from_the_opponents_boards_ahead_are_check_r1) {
  // R1, C7: "from any of their playable boards" - a White board ahead of the
  // present, (+1T2) with White to move, on which a white rook can take the
  // black king. Check, though no pass is needed for it.
  const GameState s = game({"[k4/5/5/5/5:0:1:b]", "[4k/5/5/5/4R:+1:2:w]"}, k5x5);
  REQUIRE(s.to_move() == Colour::Black);
  CHECK(in_check(s));
  CHECK(!royals_safe(s));
}

TEST(check_a_capture_that_would_branch_into_an_inactive_timeline_counts_r1) {
  // The C2 position above, with two White timelines added: +1 (active, in the
  // present, passed, empty) and +2 (inactive, empty). White has created two,
  // Black none, so White's next timeline would be inactive (T10) - the rook's
  // capture of the king's past self would branch into it. It counts (R1).
  const GameState s = game({"[4k/5/5/5/5:0:2:w]", "[4k/5/5/5/5:0:2:b]", "[5/5/5/5/k4:0:3:w]",
                            "[4R/5/5/5/k4:0:3:b]", "[5/5/5/5/5:+1:3:b]", "[5/5/5/5/5:+2:3:b]"},
                           k5x5);
  REQUIRE(s.to_move() == Colour::Black);
  REQUIRE(s.active(1));
  REQUIRE(!s.active(2));
  CHECK_EQ(s.multiverse().created_by(Colour::White), 2);
  CHECK_EQ(s.multiverse().created_by(Colour::Black), 0);
  CHECK(passed_timelines(s) == (std::vector<int>{0, 1}));
  CHECK(in_check(s));
  const std::optional<RoyalCapture> w = check_witness(s);
  REQUIRE(w.has_value());
  CHECK(w->branching);
}

TEST(check_slides_cross_the_passed_boards_and_inactive_timelines_attack_r1) {
  // Black to move at (0T1), its king on a1. +2 is White's second timeline
  // (inactive: Black has created none) with a White board (+2T2) holding a
  // rook on a1. The rook slides down the timelines at T2 (white): through +1,
  // then onto 0. The pieces of an inactive timeline's board count (R1), and
  // the slide needs a board at (+1T2) - the passed copy of +1's board (J8).
  const GameState crossing = game(
      {"[5/5/5/5/k4:0:1:b]", "[5/5/5/5/5:+1:1:b]", "[5/5/5/5/R4:+2:2:w]"}, k5x5);
  REQUIRE(!crossing.active(2));
  CHECK(passed_timelines(crossing) == (std::vector<int>{0, 1}));
  CHECK(in_check(crossing));
  const std::optional<RoyalCapture> w = check_witness(crossing);
  REQUIRE(w.has_value());
  CHECK_EQ(w->from_timeline, 2);
  CHECK_EQ(w->steps, 2);
  // When +1's board is ahead of the present (T3, black: not passed), (+1T2)
  // is void: the slide is blocked. Not in check.
  const GameState blocked = game(
      {"[5/5/5/5/k4:0:1:b]", "[5/5/5/5/5:+1:3:b]", "[5/5/5/5/R4:+2:2:w]"}, k5x5);
  CHECK(passed_timelines(blocked) == std::vector<int>{0});
  CHECK(!in_check(blocked));
}

TEST(check_royal_queens_are_royal_and_common_kings_are_not) {
  // 4.11-4.13 (Reversed Royalty's pieces): a royal queen (Y) on e5 under the
  // rook's file is in check; a common king (C) there is not royal (and Black
  // then has no royal piece: never in check, C13).
  RuleOptions rr;
  rr.family = RuleFamily::ReversedRoyalty;
  CHECK(in_check(game({"[4y/5/5/5/4R:0:1:b]"}, k5x5, rr)));
  CHECK(!in_check(game({"[4c/5/5/5/4R:0:1:b]"}, k5x5, rr)));
  CHECK(in_check(game({"[c3y/5/5/5/4R:0:1:b]"}, k5x5, rr)));
  CHECK(!in_check(game({"[y3c/5/5/5/4R:0:1:b]"}, k5x5, rr)));
}

TEST(check_a_side_with_no_royal_is_never_in_check_c13) {
  const GameState s = game({"[4n/5/5/5/4R:0:1:b]"}, k5x5);
  CHECK(!in_check(s));
  CHECK(!check_witness(s).has_value());
  GameState t = s;
  play(t, "(0T1)Ne5(0T1)c4");
  CHECK(t.submission_status() == SubmitStatus::Safe);
}

TEST(check_status_matches_a_reference_on_random_play) {
  // At the start of every turn of random pseudo-legal games from several
  // starts, the engine's check status equals the test's own construction:
  // pass the active boards in the present (R1), then ask the move generator.
  struct Start {
    std::vector<std::string> boards;
    BoardSize size;
  };
  const std::vector<Start> starts = {
      {{"[r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*:0:1:w]"}, k8x8},
      {{"[r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*:-0:1:w]",
        "[r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*:+0:1:w]"},
       k8x8},
      {{"[nbr*k*/p*p*p*p*/P*P*P*P*/K*R*BN:0:1:w]"}, BoardSize{4, 4}},
      {{"[r*r*k*r*r*/bbqbb/p*p*p*p*p*/5/P*P*P*P*P*:-1:1:w]",
        "[nnnnn/p*p*p*p*p*/5/P*P*P*P*P*/NNNNN:0:1:w]",
        "[p*p*p*p*p*/5/P*P*P*P*P*/BBQBB/R*R*K*R*R*:+1:1:w]"},
       k5x5},
      {{"[2k/3/K2:0:1:w]"}, BoardSize{3, 3}},
  };
  t5d_test::Rng rng(41);
  int compared = 0;
  int checks = 0;
  MoveList moves;
  for (const Start& start : starts) {
    for (int g = 0; g < 4; ++g) {
      GameState s = game(start.boards, start.size);
      for (int turn = 0; turn < 24; ++turn) {
        const bool expected =
            !generated_royal_captures(reference_passed(s), s.rules(), opposite(s.to_move()),
                                      s.inactive_timelines())
                 .empty();
        CHECK_EQ(in_check(s), expected);
        ++compared;
        if (expected) ++checks;
        // One random pseudo-legal action (retried from the turn's start at a
        // dead end); its submission's verdict against the reference too.
        const GameState turn_start = s;
        bool done = false;
        for (int attempt = 0; attempt < 6 && !done; ++attempt) {
          s = turn_start;
          while (!s.obligations_met()) {
            generate_moves(s, moves);
            if (moves.empty()) break;
            s.apply(moves[static_cast<std::size_t>(rng.below(static_cast<int>(moves.size())))]);
          }
          done = s.obligations_met();
        }
        if (!done) break;
        const bool unsafe = !generated_royal_captures(s.multiverse(), s.rules(),
                                                      opposite(s.to_move()),
                                                      s.inactive_timelines())
                                 .empty();
        CHECK(s.submission_status() == (unsafe ? SubmitStatus::Unsafe : SubmitStatus::Safe));
        s.submit(CheckRuling::none());
      }
    }
  }
  CHECK(compared > 300);
  CHECK(checks > 10);
}

// ---- submission legality (R1, C4) -------------------------------------------------------

TEST(check_moving_into_check_mid_turn_is_allowed_r1) {
  // Black must play -1, 0 and +1 (all active - -1 is Black's first timeline,
  // +1 White's - and in the present at T1). The black king stands on (0T1)b5;
  // on +1 a white rook a5 sits on Black's board beside a black knight b3; on
  // -1 a black knight c3. Passing all three, the rook (on White's (+1T2) then)
  // slides down the timelines to (0T2)a5: empty. Not in check.
  const GameState s =
      game({"[5/5/2n2/5/5:-1:1:b]", "[1k3/5/5/5/5:0:1:b]", "[R4/5/1n3/5/5:+1:1:b]"}, k5x5);
  REQUIRE(s.mandatory_timelines() == (std::vector<int>{-1, 0, 1}));
  CHECK(!in_check(s));
  // Black plays +1 without taking the rook, then walks the king into its line:
  // from that move on a royal is capturable (the rook on (+1T2) slides to
  // (0T2)a5). The engine refuses no move for it - only the submission (R1).
  GameState into = s;
  play(into, "(+1T1)Nb3(+1T1)d4");
  CHECK(royals_safe(into));
  play(into, "(0T1)Kb5(0T1)a5");
  CHECK(!royals_safe(into));
  CHECK(!into.obligations_met());
  CHECK_NOTHROW(play(into, "(-1T1)Nc3(-1T1)b1"));  // the turn goes on
  CHECK(into.obligations_met());
  CHECK(into.submission_status() == SubmitStatus::Unsafe);
  // The same king move, with the rook taken on +1: legal.
  GameState answered = s;
  play(answered, "(0T1)Kb5(0T1)a5");
  play(answered, "(+1T1)Nb3(+1T1)a5");
  play(answered, "(-1T1)Nc3(-1T1)b1");
  CHECK(answered.submission_status() == SubmitStatus::Safe);
}

TEST(check_submission_ignores_boards_the_mover_has_not_played) {
  // Boards on which the mover is still to move are not the opponent's: the
  // white rook on Black's optional board (+1T2) attacks nothing until Black
  // plays there (C3, R1).
  GameState s = optional_board_ahead();
  play(s, "(0T1)Ka5(0T1)b5");
  CHECK(royals_safe(s));
  CHECK(s.submission_status() == SubmitStatus::Safe);
  CHECK(s.submission_status(CheckRuling::none()) == SubmitStatus::Unchecked);
  s.submit();
  CHECK(s.to_move() == Colour::White);
}

TEST(check_castling_attack_test_is_per_board_r6) {
  // R6 (H40): castling is barred only by attacks on the castling board. A
  // black rook on +1's playable board at e1 checks the white king e1 across
  // timelines (passing (0T1) gives Black (0T1, black) - the rook slides down
  // from +1 onto e1). The king may still castle, and the turn is legal once
  // the king is safe: both castlings leave e1. A second black rook on +1 at g1
  // would take a king castled to g1: that turn cannot be submitted (C4).
  const GameState s = game({"[8/8/8/8/8/8/8/R*3K*2R*:0:1:w]", "[8/8/8/8/8/8/8/8:+1:1:w]",
                            "[8/8/8/8/8/8/8/4r3:+1:1:b]"},
                           k8x8);
  REQUIRE(s.to_move() == Colour::White);
  CHECK(in_check(s));
  GameState kingside = s;
  play(kingside, "(0T1)Ke1(0T1)g1");
  CHECK(kingside.submission_status() == SubmitStatus::Safe);
  GameState queenside = s;
  play(queenside, "(0T1)Ke1(0T1)c1");
  CHECK(queenside.submission_status() == SubmitStatus::Safe);
  const GameState two = game({"[8/8/8/8/8/8/8/R*3K*2R*:0:1:w]", "[8/8/8/8/8/8/8/8:+1:1:w]",
                              "[8/8/8/8/8/8/8/4r1r1:+1:1:b]"},
                             k8x8);
  GameState g1 = two;
  play(g1, "(0T1)Ke1(0T1)g1");  // generated: the per-board test passes
  CHECK(g1.submission_status() == SubmitStatus::Unsafe);
  GameState c1 = two;
  play(c1, "(0T1)Ke1(0T1)c1");
  CHECK(c1.submission_status() == SubmitStatus::Safe);
}

TEST(check_the_engines_ruling_is_the_default_r1) {
  // submission_status() and submit() apply the engine's ruling unless told
  // otherwise; CheckRuling::none() leaves royal safety untested.
  GameState s(standard_start());
  play(s, "(0T1)Pe2(0T1)e4");
  CHECK(s.submission_status() == SubmitStatus::Safe);
  CHECK(s.submission_status(CheckRuling::none()) == SubmitStatus::Unchecked);
  CHECK(s.submission_status(standard_check_ruling()) == SubmitStatus::Safe);
  const CheckRuling r = standard_check_ruling();
  CHECK(r.royals_safe != nullptr);
  CHECK(r.in_check != nullptr);
  CHECK(r.in_check(GameState(standard_start())) == Verdict::No);
}

TEST(check_the_replay_refuses_an_action_that_leaves_a_king_capturable_c4) {
  // 1. e4 f6 2. Qh5+: Black's 2... a6 leaves the king on e8 to the queen (h5,
  // g6, f7 empty). The replay refuses it at submission (CannotSubmit, C4); a
  // pseudo-legal replay (CheckRuling::none(), parse mode) plays it.
  const std::string text = "1. e4 / f6 2. Qh5 / a6\n";
  PgnReplay legal(read_pgn(text));
  legal.play_next();
  legal.play_next();
  legal.play_next();
  try {
    legal.play_next();
    CHECK(false);
  } catch (const RecordError& e) {
    CHECK(e.fault() == RecordFault::CannotSubmit);
    CHECK(std::string(e.what()).find("C4") != std::string::npos);
    CHECK(std::string(e.what()).find("queen") != std::string::npos);
  }
  CHECK_EQ(legal.actions_played(), std::size_t{3});
  PgnReplay pseudo(read_pgn(text));
  pseudo.set_check_ruling(CheckRuling::none());
  CHECK_NOTHROW(pseudo.play_all());
  CHECK_EQ(pseudo.actions_played(), std::size_t{4});
}
