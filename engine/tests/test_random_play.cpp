// Consistency under random pseudo-legal play: from several starting positions
// (the rules reference's V3 table) and under several rulesets, every generated
// move applies, and every state reached is well formed - boards immutable and
// shared, keys incremental == from scratch, fingerprints well formed and read
// back, counts right, activity and the present as the rules define them, the
// present on the mover's colour at every turn's start (T5).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "t5d/board.hpp"
#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/movegen.hpp"
#include "t5d/state.hpp"
#include "t5d/zobrist.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"
#include "test_support.hpp"

using namespace t5d;

namespace {

// The count rule, written again here from T10 (with Q2's credit), as an
// independent check of the engine's activity bookkeeping.
bool expected_active(const GameState& s, const Timeline& t) {
  if (!t.creator()) return true;
  const Multiverse& mv = s.multiverse();
  const auto credit = [&](Colour c) {
    return s.rules().initial_credit == InitialTimelineCredit::MinusZeroToBlack && mv.even() &&
                   c == Colour::Black
               ? 1
               : 0;
  };
  const Colour c = *t.creator();
  const Colour o = opposite(c);
  return mv.created_by(o) + credit(o) >= t.creation_order() + credit(c) - 1;
}

bool is_hex32(const std::string& s) {
  if (s.size() != 32) return false;
  for (const char c : s) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  }
  return true;
}

std::vector<std::string> split_boards(const std::string& canonical) {
  std::vector<std::string> out;
  for (std::size_t i = 0; i < canonical.size();) {
    const std::size_t end = canonical.find(']', i);
    out.push_back(canonical.substr(i, end - i + 1));
    i = end + 1;
  }
  return out;
}

// Light checks: keys, counts, content hashes, the present, activity.
void check_state(const GameState& s) {
  const Multiverse& mv = s.multiverse();
  CHECK_EQ(mv.zobrist(), zobrist::key_from_scratch(mv));
  std::size_t boards = 0;
  std::size_t pieces = 0;
  int present = 1 << 30;
  for (const TimelineRef& t : mv.timelines()) {
    for (const BoardRef& b : t->boards()) {
      ++boards;
      pieces += static_cast<std::size_t>(b->piece_count());
      CHECK_EQ(b->content_hash(), content_hash_from_scratch(*b));
      // An en-passant square is recorded only on a board created by a double
      // step: empty, with the stepping pawn (or brawn) just past it.
      if (b->en_passant() != kNoSquare) {
        CHECK(b->at(b->en_passant()).empty());
      }
    }
    // Every timeline but the initial ones lies on its creator's side (L7).
    if (t->creator() == Colour::White) CHECK(t->position() > 0);
    if (t->creator() == Colour::Black) CHECK(t->position() < (mv.even() ? -1 : 0));
    if (!s.rules().history_dependent_activity()) {
      CHECK_EQ(s.active(t->position()), expected_active(s, *t));
    }
    bool deferred = false;
    for (const GameState::Deferral& d : s.deferred_timelines()) {
      if (d.timeline == t->position()) deferred = true;
    }
    if (s.active(t->position()) && !deferred && t->last_half_turn() < present) {
      present = t->last_half_turn();
    }
  }
  CHECK_EQ(boards, mv.board_count());
  CHECK_EQ(pieces, mv.piece_count());
  CHECK_EQ(s.present(), present);
  if (!s.obligations_met() && s.rules().mandatory == MandatoryBoards::InPresent) {
    CHECK(s.present_colour() == s.to_move());
  }
  // The mandatory boards (T6; Q5), counted here from the definition, against
  // the state's incremental bookkeeping.
  int mandatory = 0;
  for (const TimelineRef& t : mv.timelines()) {
    bool deferred = false;
    for (const GameState::Deferral& d : s.deferred_timelines()) {
      if (d.timeline == t->position()) deferred = true;
    }
    const int h = t->last_half_turn();
    if (s.active(t->position()) && !deferred && to_move_at(h) == s.to_move() &&
        (s.rules().mandatory == MandatoryBoards::AllActive || h == present)) {
      ++mandatory;
    }
  }
  CHECK_EQ(s.mandatory_count(), mandatory);
  CHECK_EQ(static_cast<int>(s.mandatory_timelines().size()), mandatory);
}

// Deep checks: the fingerprint, and the canonical string read back. Returns
// whether a history board carried an en-passant square (a lapsed right).
bool check_deep(const GameState& s) {
  const Multiverse& mv = s.multiverse();
  const Fingerprint f = fingerprint(mv);
  CHECK(is_hex32(f.fingerprint));
  CHECK(is_hex32(f.placement));
  CHECK_EQ(f.boards, mv.board_count());
  CHECK_EQ(f.pieces, mv.piece_count());
  const std::string canonical = canonical_state_string(mv);
  const Multiverse back = multiverse_from_fen(split_boards(canonical), mv.board_size(),
                                              mv.even() ? Parity::Even : Parity::Odd,
                                              mv.ruleset(), mv.to_move());
  CHECK_EQ(canonical_state_string(back), canonical);
  // The labels carry the creators (L7), so the key reads back too - except an
  // en-passant square on a head board, which 5DFEN cannot record (the key
  // holds it; the fingerprint cannot). On a history board the right has
  // lapsed and the key leaves it out (zobrist::keyed_content_hash).
  bool head_en_passant = false;
  bool history_en_passant = false;
  for (const TimelineRef& t : mv.timelines()) {
    for (int h = t->first_half_turn(); h <= t->last_half_turn(); ++h) {
      if (t->board_at(h)->en_passant() == kNoSquare) continue;
      (h == t->last_half_turn() ? head_en_passant : history_en_passant) = true;
    }
  }
  if (head_en_passant) {
    CHECK_NE(back.zobrist(), mv.zobrist());
  } else {
    CHECK_EQ(back.zobrist(), mv.zobrist());
  }
  return history_en_passant;
}

struct Totals {
  long long moves_checked = 0;
  long long actions = 0;
  long long branches = 0;
  long long lapsed_en_passant = 0;  // states read back with an equal key despite one
};

// Plays up to `actions` random actions from `start`. At each step every
// generated move is applied to a copy and checked; then one is played. An
// action that reaches a dead end (obligations unmet, no move) is retried
// from the turn's start; a turn with no completed action ends the game.
void random_game(const GameState& start, std::uint64_t seed, int actions, Totals& totals) {
  t5d_test::Rng rng(seed);
  GameState s = start;
  MoveList moves;
  for (int a = 0; a < actions; ++a) {
    const GameState turn_start = s;
    CHECK(s.present_colour() == s.to_move());  // T5: every turn begins so
    bool done = false;
    for (int attempt = 0; attempt < 6 && !done; ++attempt) {
      s = turn_start;
      for (int step = 0; step < 48; ++step) {
        if (s.obligations_met() && (!s.rules().moves_after_shift || rng.chance(60))) {
          done = true;
          break;
        }
        generate_moves(s, moves);
        // generate_moves (which snapshots the timelines once) and the per-board
        // entry points (which look every board up through the multiverse)
        // agree, as does the check ruling's hook, given the activity.
        MoveList per_board;
        for (const TimelineRef& t : s.multiverse().timelines()) {
          append_board_moves(s, t->position(), per_board);
        }
        CHECK(per_board == moves);
        if (s.rules().castling_on_optional_boards) {
          MoveList hooked;
          for (const TimelineRef& t : s.multiverse().timelines()) {
            append_moves_of(s.multiverse(), s.rules(), s.to_move(), t->position(), hooked,
                            s.inactive_timelines());
          }
          CHECK(hooked == moves);
        }
        if (moves.empty()) {
          done = s.obligations_met();
          break;
        }
        // Every move applies to a copy; the parent never changes.
        const std::uint64_t parent_key = s.zobrist();
        const std::size_t parent_boards = s.multiverse().board_count();
        for (const Move& m : moves) {
          GameState child = s;
          child.apply(m);
          ++totals.moves_checked;
          if (m.branching()) ++totals.branches;
          CHECK_EQ(child.multiverse().board_count(),
                   parent_boards + (m.physical() ? 1u : 2u));
          // Boards the parent had are shared, not copied (A4).
          for (const TimelineRef& t : s.multiverse().timelines()) {
            const Timeline* u = child.multiverse().timeline(t->position());
            for (int h = t->first_half_turn(); h <= t->last_half_turn(); ++h) {
              CHECK(u->board_at(h) == t->board_at(h));
            }
          }
          CHECK(is_pseudo_legal(s, m));
          if (totals.moves_checked % 7 == 0) check_state(child);
        }
        CHECK_EQ(s.zobrist(), parent_key);
        check_state(s);
        s.apply(moves[static_cast<std::size_t>(rng.below(static_cast<int>(moves.size())))]);
      }
    }
    if (!done) return;
    if (check_deep(s)) ++totals.lapsed_en_passant;
    s.submit(CheckRuling::none());  // pseudo-legal play
    ++totals.actions;
    check_state(s);
  }
}

std::string standard_board_text() {
  return "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*";
}

struct Start {
  const char* name;
  GameState state;
};

std::vector<Start> starts(const RuleOptions& r) {
  std::vector<Start> out;
  const auto from = [&](const char* name, std::vector<std::string> boards, BoardSize size) {
    out.push_back(Start{name, t5d_test::game(boards, size, r)});
  };
  out.push_back(Start{"Standard", GameState(standard_start(), r)});
  out.push_back(Start{"Two Timelines", GameState(standard_two_timelines_start(), r)});
  from("Turn Zero", {"[" + standard_board_text() + ":0:0:b]",
                     "[" + standard_board_text() + ":0:1:w]"},
       BoardSize{8, 8});
  from("Very Small", {"[nbr*k*/p*p*p*p*/P*P*P*P*/K*R*BN:0:1:w]"}, BoardSize{4, 4});
  from("Timeline Battleground",
       {"[r*r*k*r*r*/bbqbb/p*p*p*p*p*/5/P*P*P*P*P*:-1:1:w]",
        "[nnnnn/p*p*p*p*p*/5/P*P*P*P*P*/NNNNN:0:1:w]",
        "[p*p*p*p*p*/5/P*P*P*P*P*/BBQBB/R*R*K*R*R*:+1:1:w]"},
       BoardSize{5, 5});
  from("Timeline Strategos", {"[nbk*ur*/p*p*p*p*p*/5/5/5:-0:1:w]",
                              "[5/5/5/P*P*P*P*P*/R*UK*BN:+0:1:w]"},
       BoardSize{5, 5});
  from("Just Brawns", {"[w*w*w*w*k*/5/5/5/K*W*W*W*W*:0:1:w]"}, BoardSize{5, 5});
  from("Mixed pieces", {"[r*sdk*yu/w*p*c2p*/6/6/P*2CW*P*/R*SDK*YU:0:1:w]"}, BoardSize{6, 6});
  return out;
}

}  // namespace

TEST(random_play_default_rules_keep_states_well_formed) {
  Totals totals;
  std::uint64_t seed = 1;
  for (const Start& start : starts({})) {
    check_state(start.state);
    check_deep(start.state);
    for (int game = 0; game < 2; ++game) random_game(start.state, seed++, 14, totals);
  }
  // The games exercised the machinery: many moves, many branches.
  CHECK(totals.moves_checked > 20000);
  CHECK(totals.branches > 1000);
  CHECK(totals.lapsed_en_passant > 0);  // the lapsed en-passant rights were exercised
  CHECK(totals.actions > 100);
}

TEST(random_play_alternative_rulesets_keep_states_well_formed) {
  std::vector<RuleOptions> rulesets;
  {
    RuleOptions r;
    r.activation = ActivationRule::Consecutive;
    r.play_activates = true;
    r.reactivation = Reactivation::Deferred;
    r.inactive_history_is_destination = false;
    r.brawn_cross_board_promotion = false;
    rulesets.push_back(r);
  }
  {
    RuleOptions r;
    r.moves_after_shift = true;
    r.mandatory = MandatoryBoards::AllActive;
    r.initial_credit = InitialTimelineCredit::MinusZeroToBlack;
    r.earlier_playable_jump_branches = true;
    rulesets.push_back(r);
  }
  {
    RuleOptions r;
    r.sliders_cross_void = true;
    r.past_self_blocks = false;
    r.turn_zero_reachable = false;
    r.double_step_right = DoubleStepRight::HomeRank;
    r.double_step_over_void = true;
    r.jumps_keep_unmoved = true;
    r.branch_copies_keep_unmoved = false;
    r.brawn_rank_timeline_capture = false;
    r.en_passant_brawns = false;
    r.brawns_promote = false;
    r.promotions = PromotionSet::of({PieceKind::Queen, PieceKind::Knight, PieceKind::Princess});
    r.castling_empty_between = false;
    r.castling_destination_test = true;
    r.castling_rook_path_test = true;
    r.castling_any_rank = false;
    r.castling_rook_two_away = false;
    r.castling_on_optional_boards = false;
    rulesets.push_back(r);
  }
  Totals totals;
  std::uint64_t seed = 100;
  for (const RuleOptions& r : rulesets) {
    for (const Start& start : starts(r)) {
      CHECK_EQ(start.state.multiverse().ruleset(), r.ruleset_id());
      random_game(start.state, seed++, 10, totals);
    }
  }
  CHECK(totals.moves_checked > 10000);
}

TEST(random_play_plain_and_general_generators_agree) {
  // The generator's probing functions are compiled twice (movegen.cpp,
  // kPlain): for rulesets that leave the options read per probe at their
  // defaults, and for the rest. turn_zero_reachable = false selects the
  // general instantiation yet changes nothing while no turn-zero board exists
  // - as in games from Standard and Two Timelines - so at every turn's start
  // both must generate the same moves in the same order.
  RuleOptions general;
  general.turn_zero_reachable = false;
  int compared = 0;
  for (const bool two : {false, true}) {
    t5d_test::Rng rng(two ? 31 : 30);
    GameState s(two ? standard_two_timelines_start() : standard_start());
    MoveList moves;
    MoveList other;
    for (int turn = 0; turn < 40; ++turn) {
      const GameState alt(s.multiverse(), general);
      generate_moves(s, moves);
      generate_moves(alt, other);
      CHECK(moves == other);
      ++compared;
      // One random action, retried from the turn's start at a dead end.
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
      s.submit(CheckRuling::none());  // pseudo-legal play
    }
  }
  CHECK(compared > 40);
}

TEST(random_play_is_deterministic) {
  // Same seed, same game: equal states and keys.
  const auto play_out = [](std::uint64_t seed) {
    t5d_test::Rng rng(seed);
    GameState s(standard_two_timelines_start());
    MoveList moves;
    for (int ply = 0; ply < 40; ++ply) {
      if (s.obligations_met()) {
        s.submit(CheckRuling::none());  // pseudo-legal play
        continue;
      }
      generate_moves(s, moves);
      if (moves.empty()) break;
      s.apply(moves[static_cast<std::size_t>(rng.below(static_cast<int>(moves.size())))]);
    }
    return s;
  };
  const GameState a = play_out(77);
  const GameState b = play_out(77);
  CHECK(a == b);
  CHECK_EQ(a.zobrist(), b.zobrist());
  CHECK_EQ(fingerprint(a.multiverse()).fingerprint, fingerprint(b.multiverse()).fingerprint);
}
