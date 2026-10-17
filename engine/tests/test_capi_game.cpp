// The C API's games (t5d/capi.h, stage E3): made from a record, from 5DFEN and
// from a variant; replayed, moved, submitted; expanded - every array the C API
// hands out equal to the C++ expansion of the same state; errors; ownership.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "synthetic.hpp"
#include "t5d/capi.h"
#include "t5d/expand.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"

using namespace t5d;

namespace {

struct Game {
  t5d_game* p = nullptr;
  Game() = default;
  Game(const Game&) = delete;
  Game& operator=(const Game&) = delete;
  ~Game() { t5d_game_destroy(p); }
};

struct State {
  t5d_state* p = nullptr;
  State() = default;
  State(const State&) = delete;
  State& operator=(const State&) = delete;
  ~State() { t5d_state_destroy(p); }
};

struct Text {
  char* p = nullptr;
  Text() = default;
  Text(const Text&) = delete;
  Text& operator=(const Text&) = delete;
  ~Text() { t5d_free(p); }
  bool has(std::string_view part) const {
    return p != nullptr && std::string_view(p).find(part) != std::string_view::npos;
  }
};

std::string fingerprint_of(const t5d_game* game) {
  State s;
  if (t5d_game_state(game, &s.p) != T5D_OK) return "<none>";
  t5d_fingerprint f{};
  if (t5d_state_fingerprint(s.p, &f) != T5D_OK) return "<none>";
  return f.fingerprint;
}

template <class T>
bool same(const T* a, std::span<const T> b, std::size_t n) {
  if (n != b.size()) return false;
  return n == 0 || std::memcmp(a, b.data(), n * sizeof(T)) == 0;
}

// Every count and array of a C expansion against the C++ one.
void check_view(const t5d_expansion& v, const Expansion& x) {
  CHECK_EQ(v.width, x.board_size().width);
  CHECK_EQ(v.height, x.board_size().height);
  CHECK_EQ(v.even, x.even() ? 1 : 0);
  CHECK_EQ(v.to_move, index_of(x.to_move()));
  CHECK_EQ(v.present, x.present());
  CHECK_EQ(v.timelines, x.timeline_count());
  CHECK_EQ(v.active_timelines, x.active_timeline_count());
  CHECK_EQ(v.window_k, x.window_k());
  CHECK_EQ(v.may_move, x.may_move() ? 1 : 0);
  CHECK_EQ(v.origins_unknown, x.origins_unknown());
  const std::size_t n = x.board_count();
  REQUIRE_EQ(v.board_count, n);
  CHECK(same(v.timeline, x.board_timelines(), n));
  CHECK(same(v.half_turn, x.board_half_turns(), n));
  CHECK(same(v.is_head, x.board_is_head(), n));
  CHECK(same(v.active, x.board_active(), n));
  CHECK(same(v.side_to_move, x.board_side_to_move(), n));
  CHECK(same(v.playable, x.board_playable(), n));
  CHECK(same(v.window_depth, x.window_depth(), n));
  CHECK(same(v.band_depth, x.band_depth(), n));
  CHECK(same(v.is_origin, x.board_is_origin(), n));
  CHECK(same(v.parent, x.board_parent(), n));
  CHECK_EQ(v.origin_count, x.origin_count());
  CHECK(same(v.feature_boards, x.feature_boards(), v.feature_board_count));
  const std::size_t cells = v.feature_board_count * static_cast<std::size_t>(v.width * v.height);
  CHECK(same(v.piece, x.square_pieces(), cells));
  CHECK(same(v.colour, x.square_colours(), cells));
  CHECK(same(v.moved, x.square_moved(), cells));
  CHECK(same(v.moves, x.move_records(), v.move_count));
  CHECK(same(v.edges, x.move_edges(), 4 * v.edge_count));
  CHECK(same(v.relations, x.relations(), 5 * v.relation_count));
  CHECK(same(v.window, x.window(), v.window_count));
  CHECK(same(v.e_boards, x.e_boards(), v.e_count));
  CHECK(same(v.src_extra, x.src_extra(), v.src_extra_count));
  CHECK_EQ(v.acting_player, index_of(x.acting_player()));
  CHECK_EQ(v.submit_possible, x.submit_possible() ? 1 : 0);
  CHECK_EQ(v.submit_legal, x.submit_legal() ? 1 : 0);
  CHECK_EQ(v.mandatory_remaining, x.mandatory_remaining());
  CHECK_EQ(v.optional_playable, x.optional_playable());
  CHECK_EQ(v.moves_made, x.moves_made());
}

synthetic::Game sample_game() {
  synthetic::Spec spec;
  spec.seed = 5;
  spec.target_boards = 60;
  spec.target_active = 4;
  spec.branch_percent = 50;
  return synthetic::play(spec);
}

}  // namespace

TEST(capi_game_from_a_record_replays_and_expands_as_the_engine) {
  const synthetic::Game g = sample_game();
  REQUIRE(g.stop == synthetic::Stop::Target);
  const std::string record = synthetic::record(g);
  Game game;
  Text error;
  REQUIRE_EQ(t5d_game_new_from_pgn(record.data(), record.size(), T5D_RULES_STANDARD, &game.p,
                                   &error.p),
             T5D_OK);
  // Synthetic play is pseudo-legal: replayed as such.
  REQUIRE_EQ(t5d_game_set_legality(game.p, T5D_PSEUDO_LEGAL, &error.p), T5D_OK);
  size_t total = 0;
  size_t played = 99;
  int32_t truncated = 9;
  CHECK_EQ(t5d_game_record_actions(game.p, &total, &played, &truncated), T5D_OK);
  CHECK_EQ(total, g.actions.size());
  CHECK_EQ(played, size_t{0});
  CHECK_EQ(truncated, 0);
  for (size_t a = 0; a < total; ++a) REQUIRE_EQ(t5d_game_replay_next(game.p, nullptr), T5D_OK);
  CHECK_EQ(t5d_game_replay_next(game.p, &error.p), T5D_ERR_ARGUMENT);  // nothing left
  CHECK(error.has("every action"));
  CHECK_EQ(fingerprint_of(game.p), fingerprint(g.state.multiverse()).fingerprint);

  // The expansion, every array, two option sets.
  for (const bool full : {false, true}) {
    t5d_expand_options o{};
    o.window_k = full ? 4 : 8;
    o.square_features = full ? T5D_FEATURES_ALL : T5D_FEATURES_SCOPE;
    o.relations = full ? (T5D_RELATIONS_ATTACK | T5D_RELATIONS_DEFENCE | T5D_RELATIONS_XRAY) : 0u;
    t5d_expansion v{};
    REQUIRE_EQ(t5d_game_expand(game.p, &o, &v, &error.p), T5D_OK);
    ExpandOptions eo;
    eo.window_k = o.window_k;
    eo.square_features = full ? SquareFeatures::All : SquareFeatures::Scope;
    eo.attack = eo.defence = eo.xray = full;
    Expansion x;
    expand(g.state, eo, x);
    check_view(v, x);
    CHECK(v.move_count > 0);
    if (full) CHECK(v.relation_count > 0);
  }
  // NULL options: k = 8, every board's features, no relations.
  t5d_expansion v{};
  REQUIRE_EQ(t5d_game_expand(game.p, nullptr, &v, nullptr), T5D_OK);
  CHECK_EQ(v.window_k, 8);
  CHECK_EQ(v.feature_board_count, v.board_count);
  CHECK_EQ(v.relation_count, size_t{0});

  t5d_game_summary info{};
  CHECK_EQ(t5d_game_info(game.p, &info), T5D_OK);
  CHECK_EQ(info.boards, static_cast<uint64_t>(g.state.multiverse().board_count()));
  CHECK_EQ(info.to_move, index_of(g.state.to_move()));
  CHECK_EQ(info.present, g.state.present());
  CHECK_EQ(info.may_move, 1);
  CHECK_EQ(info.may_submit, 0);  // an action's start: the present is on the mover's colour
  CHECK_EQ(info.rules, uint64_t{0});
}

TEST(capi_game_applies_moves_and_submits) {
  const synthetic::Game g = sample_game();
  const std::string record = synthetic::record(g, 10);
  Game game;
  REQUIRE_EQ(t5d_game_new_from_pgn(record.data(), record.size(), 0, &game.p, nullptr), T5D_OK);
  for (int a = 0; a < 10; ++a) REQUIRE_EQ(t5d_game_replay_next(game.p, nullptr), T5D_OK);
  GameState s = synthetic::replay(g.start, g.actions, 10);
  CHECK_EQ(fingerprint_of(game.p), fingerprint(s.multiverse()).fingerprint);

  // The game's next action, move by move: by record, then by index.
  const std::vector<Move>& action = g.actions[10];
  Text error;
  CHECK_EQ(t5d_game_apply_expanded(game.p, 0, &error.p), T5D_ERR_ARGUMENT);  // no expansion yet
  for (std::size_t i = 0; i < action.size(); ++i) {
    t5d_expansion v{};
    REQUIRE_EQ(t5d_game_expand(game.p, nullptr, &v, nullptr), T5D_OK);
    std::size_t index = v.move_count;
    MoveList moves;
    generate_moves(s, moves);
    for (std::size_t j = 0; j < moves.size(); ++j) {
      if (moves[j] == action[i]) index = j;
    }
    REQUIRE(index < v.move_count);
    if (i % 2 == 0) {
      CHECK_EQ(t5d_game_apply(game.p, &v.moves[index], &error.p), T5D_OK);
    } else {
      CHECK_EQ(t5d_game_apply_expanded(game.p, index, &error.p), T5D_OK);
    }
    s.apply(action[i]);
    CHECK_EQ(fingerprint_of(game.p), fingerprint(s.multiverse()).fingerprint);
    // The expansion is of the state before the move now.
    CHECK_EQ(t5d_game_apply_expanded(game.p, 0, nullptr), T5D_ERR_ARGUMENT);
  }
  CHECK_EQ(t5d_game_replay_next(game.p, &error.p), T5D_ERR_ARGUMENT);  // left the record
  CHECK(error.has("left its record"));
  REQUIRE_EQ(t5d_game_submit(game.p, &error.p), T5D_OK);
  s.submit(CheckRuling{});
  t5d_game_summary info{};
  CHECK_EQ(t5d_game_info(game.p, &info), T5D_OK);
  CHECK_EQ(info.to_move, index_of(s.to_move()));
  // At the next action's start, submitting is refused (T5) and a made-up move
  // is not a move.
  CHECK_EQ(t5d_game_submit(game.p, &error.p), T5D_ERR_ILLEGAL);
  CHECK(error.has("T5"));
  t5d_move bogus{};
  bogus.from_timeline = 0;
  bogus.from_half_turn = 0;
  bogus.from_square = 0;
  CHECK_EQ(t5d_game_apply(game.p, &bogus, &error.p), T5D_ERR_ILLEGAL);
  CHECK(error.has("no pseudo-legal move"));
}

TEST(capi_game_from_fen_and_variant) {
  const char* boards[] = {
      "[r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*:0:1:w]"};
  Game fen;
  REQUIRE_EQ(t5d_game_new_from_fen(boards, 1, 0, 0, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, 0, &fen.p,
                                   nullptr),
             T5D_OK);
  Game standard;
  REQUIRE_EQ(t5d_game_new_variant("Standard", 0, &standard.p, nullptr), T5D_OK);
  CHECK_EQ(fingerprint_of(fen.p), fingerprint_of(standard.p));
  t5d_expansion v{};
  REQUIRE_EQ(t5d_game_expand(standard.p, nullptr, &v, nullptr), T5D_OK);
  CHECK_EQ(v.move_count, size_t{20});
  CHECK_EQ(v.board_count, size_t{1});
  size_t total = 9;
  CHECK_EQ(t5d_game_record_actions(standard.p, &total, nullptr, nullptr), T5D_OK);
  CHECK_EQ(total, size_t{0});
  CHECK_EQ(t5d_game_replay_next(standard.p, nullptr), T5D_ERR_ARGUMENT);  // no record

  // A ruleset by its encoding: Q4's Guide-literal reading (bit 3 since R8
  // made moves after the shift the default; T5D_RULES_MOVES_AFTER_SHIFT is
  // now the default's 0).
  Game q4;
  REQUIRE_EQ(t5d_game_new_variant("standard - two timelines", T5D_RULES_NO_MOVES_AFTER_SHIFT,
                                  &q4.p, nullptr),
             T5D_OK);
  t5d_game_summary info{};
  CHECK_EQ(t5d_game_info(q4.p, &info), T5D_OK);
  CHECK_EQ(info.rules, T5D_RULES_NO_MOVES_AFTER_SHIFT);
  CHECK_EQ(info.rules, std::uint64_t{1} << 3);
  CHECK(info.ruleset != 0u);
  CHECK_EQ(info.even, 1);
  CHECK_EQ(info.timelines, 2);
  CHECK_EQ(T5D_RULES_MOVES_AFTER_SHIFT, T5D_RULES_STANDARD);
  Game ruled;
  REQUIRE_EQ(t5d_game_new_variant("standard - two timelines", T5D_RULES_MOVES_AFTER_SHIFT,
                                  &ruled.p, nullptr),
             T5D_OK);
  CHECK_EQ(t5d_game_info(ruled.p, &info), T5D_OK);
  CHECK_EQ(info.rules, std::uint64_t{0});
  CHECK_EQ(info.ruleset, 0u);

  // A copy is independent.
  Game copy;
  REQUIRE_EQ(t5d_game_copy(standard.p, &copy.p), T5D_OK);
  REQUIRE_EQ(t5d_game_expand(copy.p, nullptr, &v, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_game_apply_expanded(copy.p, 0, nullptr), T5D_OK);
  CHECK(fingerprint_of(copy.p) != fingerprint_of(standard.p));
}

TEST(capi_game_errors) {
  Game g;
  Text error;
  const std::string garbage = "1. (0T1)Qz9";
  CHECK_EQ(t5d_game_new_from_pgn(garbage.data(), garbage.size(), 0, &g.p, &error.p), T5D_OK);
  // The set-up reads; the move text stops at once (truncated), replay says why.
  int32_t truncated = 0;
  size_t total = 9;
  CHECK_EQ(t5d_game_record_actions(g.p, &total, nullptr, &truncated), T5D_OK);
  CHECK_EQ(total, size_t{0});
  CHECK_EQ(truncated, 1);
  CHECK_EQ(t5d_game_replay_next(g.p, &error.p), T5D_ERR_NOTATION);
  t5d_game_destroy(g.p);
  g.p = nullptr;

  const std::string invasion = "[Board \"Misc - Timeline Invasion\"]\n1. e4\n";
  CHECK_EQ(t5d_game_new_from_pgn(invasion.data(), invasion.size(), 0, &g.p, &error.p),
           T5D_ERR_UNSUPPORTED);
  CHECK(g.p == nullptr);
  CHECK_EQ(t5d_game_new_variant("No Such Variant", 0, &g.p, &error.p), T5D_ERR_UNSUPPORTED);
  CHECK_EQ(t5d_game_new_variant("Standard", uint64_t{1} << 21, &g.p, &error.p), T5D_ERR_ARGUMENT);
  CHECK(error.has("not the encoding"));
  CHECK_EQ(t5d_game_new_variant(nullptr, 0, &g.p, &error.p), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_game_new_variant("Standard", 0, nullptr, &error.p), T5D_ERR_ARGUMENT);

  REQUIRE_EQ(t5d_game_new_variant("Standard", 0, &g.p, nullptr), T5D_OK);
  t5d_expansion v{};
  t5d_expand_options o{};
  o.window_k = 8;
  o.relations = 1u << T5D_RELATION_PIN;
  CHECK_EQ(t5d_game_expand(g.p, &o, &v, &error.p), T5D_ERR_UNSUPPORTED);
  CHECK(error.has("check detection"));
  o.relations = 1u << T5D_RELATION_TRAVEL_THREAT;
  CHECK_EQ(t5d_game_expand(g.p, &o, &v, &error.p), T5D_ERR_UNSUPPORTED);
  o.relations = 1u << 9;
  CHECK_EQ(t5d_game_expand(g.p, &o, &v, &error.p), T5D_ERR_ARGUMENT);
  o.relations = 0;
  o.window_k = -1;
  CHECK_EQ(t5d_game_expand(g.p, &o, &v, &error.p), T5D_ERR_ARGUMENT);
  o.window_k = 8;
  o.square_features = 7;
  CHECK_EQ(t5d_game_expand(g.p, &o, &v, &error.p), T5D_ERR_ARGUMENT);
  // The largest k is a k like any other: every band whole (T5D_NO_WINDOW marks
  // "in no window", and no k reaches it).
  o.square_features = T5D_FEATURES_NONE;
  o.window_k = INT32_MAX;
  CHECK_EQ(t5d_game_expand(g.p, &o, &v, &error.p), T5D_OK);
  CHECK_EQ(v.window_count, size_t{1});
  CHECK_EQ(v.window_depth[0], 0);
  CHECK_EQ(t5d_game_expand(g.p, nullptr, nullptr, &error.p), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_game_info(nullptr, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_game_copy(nullptr, nullptr), T5D_ERR_ARGUMENT);
  t5d_game_destroy(nullptr);
}

namespace {

// The 5DFEN pieces field of the newest board of timeline 0.
std::string head_placement(const t5d_game* game) {
  State s;
  if (t5d_game_state(game, &s.p) != T5D_OK) return "<none>";
  size_t count = 0;
  if (t5d_state_board_count(s.p, &count) != T5D_OK || count == 0) return "<none>";
  int32_t timeline = 0;
  int32_t half_turn = 0;
  Text fen;
  if (t5d_state_board(s.p, count - 1, &timeline, &half_turn, &fen.p) != T5D_OK) return "<none>";
  const std::string text = fen.p;  // "[placement:L:T:c]"
  return text.substr(1, text.find(':') - 1);
}

}  // namespace

TEST(capi_game_applies_the_promotion_asked_for) {
  // A moved white pawn on b7, the promotion set Q R B N (S8, a Promotions
  // header's ruleset): b8 four ways. t5d_game_apply takes the promotion from
  // the move it is given; t5d_game_apply_expanded the move at its index.
  RuleOptions rules;
  rules.promotions = PromotionSet::of({PieceKind::Queen, PieceKind::Rook, PieceKind::Bishop,
                                       PieceKind::Knight});
  const char* boards[] = {"[4k3/1P6/8/8/8/8/8/4K3:0:1:w]"};
  const auto fresh = [&](Game& g) {
    REQUIRE_EQ(t5d_game_new_from_fen(boards, 1, 0, 0, T5D_PARITY_ODD, T5D_TO_MOVE_INFER,
                                     rules.encode(), &g.p, nullptr),
               T5D_OK);
  };
  // By record: each under-promotion, and the queen.
  const std::pair<uint8_t, const char*> wanted[] = {
      {2, "1N2k3"}, {3, "1B2k3"}, {4, "1R2k3"}, {5, "1Q2k3"}};
  for (const auto& [promotion, placement] : wanted) {
    Game g;
    fresh(g);
    t5d_move m{};
    m.from_timeline = 0;
    m.from_half_turn = 2;
    m.from_square = static_cast<int8_t>(make_square(1, 6));  // b7
    m.to_timeline = 0;
    m.to_half_turn = 2;
    m.to_square = static_cast<int8_t>(make_square(1, 7));  // b8
    m.promotion = promotion;
    Text error;
    REQUIRE_EQ(t5d_game_apply(g.p, &m, &error.p), T5D_OK);
    CHECK_EQ(head_placement(g.p).substr(0, 5), std::string(placement));
  }
  // A promotion the ruleset does not offer is no move.
  {
    Game g;
    fresh(g);
    t5d_move m{};
    m.from_half_turn = 2;
    m.from_square = static_cast<int8_t>(make_square(1, 6));
    m.to_half_turn = 2;
    m.to_square = static_cast<int8_t>(make_square(1, 7));
    m.promotion = 7;  // a unicorn
    CHECK_EQ(t5d_game_apply(g.p, &m, nullptr), T5D_ERR_ILLEGAL);
  }
  // By index into the expansion: the rook.
  Game g;
  fresh(g);
  t5d_expansion v{};
  REQUIRE_EQ(t5d_game_expand(g.p, nullptr, &v, nullptr), T5D_OK);
  size_t rook = v.move_count;
  for (size_t i = 0; i < v.move_count; ++i) {
    if (v.moves[i].promotion == 4) rook = i;
  }
  REQUIRE(rook < v.move_count);
  REQUIRE_EQ(t5d_game_apply_expanded(g.p, rook, nullptr), T5D_OK);
  CHECK_EQ(head_placement(g.p).substr(0, 5), std::string("1R2k3"));
}

TEST(capi_game_submissions_are_legal_by_default) {
  // Stage E4 (C4, R1): 1. e4 f6 2. Qh5+ a6 leaves Black's king to the queen.
  // A game's record replay refuses that action by default (T5D_ERR_ILLEGAL,
  // the game unchanged); T5D_PSEUDO_LEGAL replays it as parse mode does.
  const std::string record = "1. e4 / f6 2. Qh5 / a6\n";
  for (const int32_t legality : {T5D_LEGAL, T5D_PSEUDO_LEGAL}) {
    Game game;
    Text error;
    REQUIRE_EQ(t5d_game_new_from_pgn(record.data(), record.size(), T5D_RULES_STANDARD, &game.p,
                                     &error.p),
               T5D_OK);
    if (legality == T5D_PSEUDO_LEGAL) {
      REQUIRE_EQ(t5d_game_set_legality(game.p, legality, nullptr), T5D_OK);
    }
    for (int a = 0; a < 3; ++a) REQUIRE_EQ(t5d_game_replay_next(game.p, nullptr), T5D_OK);
    const t5d_status last = t5d_game_replay_next(game.p, &error.p);
    if (legality == T5D_LEGAL) {
      CHECK_EQ(last, T5D_ERR_ILLEGAL);
      CHECK(error.has("C4"));
      size_t played = 0;
      CHECK_EQ(t5d_game_record_actions(game.p, nullptr, &played, nullptr), T5D_OK);
      CHECK_EQ(played, size_t{3});
    } else {
      CHECK_EQ(last, T5D_OK);
    }
  }
  Game game;
  REQUIRE_EQ(t5d_game_new_variant("Standard", T5D_RULES_STANDARD, &game.p, nullptr), T5D_OK);
  CHECK_EQ(t5d_game_set_legality(game.p, 7, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_game_set_legality(nullptr, T5D_LEGAL, nullptr), T5D_ERR_ARGUMENT);
}

TEST(capi_expansion_says_who_acts_and_whether_submitting_is_legal_d8) {
  // Register D8 (rulebook R8): turns have variable length, so submitting is
  // an action. Appended to t5d_expansion at stage E7, after every older field
  // (python/t5d/engine/capi.py mirrors the layout): acting_player,
  // submit_possible, submit_legal - three int32, in that order.
  static_assert(offsetof(t5d_expansion, acting_player) >=
                offsetof(t5d_expansion, src_extra) + sizeof(const int32_t*));
  static_assert(offsetof(t5d_expansion, submit_possible) ==
                offsetof(t5d_expansion, acting_player) + sizeof(int32_t));
  static_assert(offsetof(t5d_expansion, submit_legal) ==
                offsetof(t5d_expansion, submit_possible) + sizeof(int32_t));
  static_assert(sizeof(t5d_expansion) >= offsetof(t5d_expansion, submit_legal) + sizeof(int32_t));

  // 1. e4 / f6 2. Qh5: Black to move, the queen on h5 eyes the king on e8.
  const std::string record = "1. e4 / f6 2. Qh5\n";
  for (const uint64_t rules : {T5D_RULES_STANDARD, T5D_RULES_NO_MOVES_AFTER_SHIFT}) {
    Game game;
    REQUIRE_EQ(t5d_game_new_from_pgn(record.data(), record.size(), rules, &game.p, nullptr), T5D_OK);
    for (int a = 0; a < 3; ++a) REQUIRE_EQ(t5d_game_replay_next(game.p, nullptr), T5D_OK);
    t5d_expansion v{};
    REQUIRE_EQ(t5d_game_expand(game.p, nullptr, &v, nullptr), T5D_OK);
    CHECK_EQ(v.acting_player, 1);
    CHECK_EQ(v.acting_player, v.to_move);
    CHECK_EQ(v.present & 1, 1);  // the turn's start: the present is Black's
    CHECK_EQ(v.submit_possible, 0);
    CHECK_EQ(v.submit_legal, 0);
    // ... a6 ignores the queen: the present shifts to White, so submitting is
    // possible - but not legal (C4, R1): the queen can take the king.
    t5d_move a6{};
    a6.from_timeline = a6.to_timeline = 0;
    a6.from_half_turn = a6.to_half_turn = 2 * 2 + 1;  // (0T2), Black to move
    a6.from_square = 6 * 8 + 0;                        // a7
    a6.to_square = 5 * 8 + 0;                          // a6
    REQUIRE_EQ(t5d_game_apply(game.p, &a6, nullptr), T5D_OK);
    REQUIRE_EQ(t5d_game_expand(game.p, nullptr, &v, nullptr), T5D_OK);
    CHECK_EQ(v.acting_player, 1);  // still Black's action
    CHECK_EQ(v.present & 1, 0);    // though the present is White's
    CHECK_EQ(v.submit_possible, 1);
    CHECK_EQ(v.submit_legal, 0);
    CHECK_EQ(v.may_move, rules == T5D_RULES_STANDARD ? 1 : 0);  // R8, or Q4's Guide-literal reading
    CHECK_EQ(v.move_count, size_t{0});  // either way: the one board has been played
    CHECK_EQ(t5d_game_submit(game.p, nullptr), T5D_ERR_ILLEGAL);
    t5d_game_summary info{};
    CHECK_EQ(t5d_game_info(game.p, &info), T5D_OK);
    CHECK_EQ(info.obligations_met, 1);
    CHECK_EQ(info.may_submit, 0);
  }
  // A legal submission: 1. e3 - possible and legal.
  Game game;
  REQUIRE_EQ(t5d_game_new_variant("Standard", T5D_RULES_STANDARD, &game.p, nullptr), T5D_OK);
  t5d_move e3{};
  e3.from_half_turn = e3.to_half_turn = 2;  // (0T1), White to move
  e3.from_square = 1 * 8 + 4;               // e2
  e3.to_square = 2 * 8 + 4;                 // e3
  REQUIRE_EQ(t5d_game_apply(game.p, &e3, nullptr), T5D_OK);
  t5d_expansion v{};
  REQUIRE_EQ(t5d_game_expand(game.p, nullptr, &v, nullptr), T5D_OK);
  CHECK_EQ(v.acting_player, 0);
  CHECK_EQ(v.present & 1, 1);
  CHECK_EQ(v.submit_possible, 1);
  CHECK_EQ(v.submit_legal, 1);
  CHECK_EQ(t5d_game_submit(game.p, nullptr), T5D_OK);
}

namespace {

t5d_move king_move(int32_t timeline, int32_t half_turn, int8_t from, int8_t to) {
  t5d_move m{};
  m.from_timeline = m.to_timeline = timeline;
  m.from_half_turn = m.to_half_turn = half_turn;
  m.from_square = from;
  m.to_square = to;
  return m;
}

int32_t moves_made(t5d_game* g) {
  t5d_game_summary info{};
  REQUIRE_EQ(t5d_game_info(g, &info), T5D_OK);
  return info.moves_made;
}

t5d_expansion expanded(t5d_game* g) {
  t5d_expansion v{};
  REQUIRE_EQ(t5d_game_expand(g, nullptr, &v, nullptr), T5D_OK);
  return v;
}

}  // namespace

TEST(capi_game_counts_the_moves_in_its_action_a6) {
  // Register A6: the moves made in the action in progress - no state records
  // them - are the handle's: 0 at every set-up (a turn start, T5), +1 per move
  // applied, 0 after a submission or a replayed action, carried by a copy, and
  // in every expansion beside the mandatory and optional boards. Appended to
  // t5d_expansion after D8's three fields (every older offset kept), and to
  // t5d_game_summary after `rules`.
  static_assert(offsetof(t5d_expansion, mandatory_remaining) ==
                offsetof(t5d_expansion, submit_legal) + sizeof(int32_t));
  static_assert(offsetof(t5d_expansion, optional_playable) ==
                offsetof(t5d_expansion, mandatory_remaining) + sizeof(int32_t));
  static_assert(offsetof(t5d_expansion, moves_made) == offsetof(t5d_expansion, optional_playable) + sizeof(int32_t));
  static_assert(sizeof(t5d_expansion) >= offsetof(t5d_expansion, moves_made) + sizeof(int32_t));
  static_assert(offsetof(t5d_game_summary, moves_made) >= offsetof(t5d_game_summary, rules) + sizeof(uint64_t));
  // White's timeline 0 at T1 - mandatory - and +1 at T2, ahead of the present -
  // optional.
  const char* boards[] = {"[3k/4/4/K3:0:1:w]", "[3k/4/4/K3:+1:2:w]"};
  const t5d_move on_zero = king_move(0, 2, 0, 8);  // (0T1)Ka1(0T1)a2
  const t5d_move on_one = king_move(1, 4, 0, 1);   // (+1T2)Ka1(+1T2)b1
  for (const uint64_t rules : {T5D_RULES_STANDARD, T5D_RULES_NO_MOVES_AFTER_SHIFT}) {
    const bool r8 = rules == T5D_RULES_STANDARD;
    Game game;
    REQUIRE_EQ(t5d_game_new_from_fen(boards, 2, 4, 4, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, rules, &game.p, nullptr),
               T5D_OK);
    CHECK_EQ(moves_made(game.p), 0);
    t5d_expansion v = expanded(game.p);
    CHECK_EQ(v.mandatory_remaining, 1);
    CHECK_EQ(v.optional_playable, 1);
    CHECK_EQ(v.moves_made, 0);
    // Refused moves and submissions count nothing.
    Text error;
    CHECK_EQ(t5d_game_submit(game.p, &error.p), T5D_ERR_ILLEGAL);
    t5d_move bogus = on_zero;
    bogus.to_square = 63;
    CHECK_EQ(t5d_game_apply(game.p, &bogus, nullptr), T5D_ERR_ILLEGAL);
    CHECK_EQ(moves_made(game.p), 0);
    // The mandatory board played: the present shifts (T5).
    REQUIRE_EQ(t5d_game_apply(game.p, &on_zero, nullptr), T5D_OK);
    CHECK_EQ(moves_made(game.p), 1);
    v = expanded(game.p);
    CHECK_EQ(v.mandatory_remaining, 0);
    CHECK_EQ(v.submit_possible, 1);
    CHECK_EQ(v.optional_playable, r8 ? 1 : 0);  // Q4: under the Guide-literal reading no move follows the shift
    CHECK_EQ(v.moves_made, 1);
    if (r8) {
      // A copy carries the count; each handle counts its own moves after it.
      Game copy;
      REQUIRE_EQ(t5d_game_copy(game.p, &copy.p), T5D_OK);
      CHECK_EQ(moves_made(copy.p), 1);
      v = expanded(copy.p);
      size_t index = v.move_count;
      for (size_t i = 0; i < v.move_count; ++i) {
        const t5d_move& m = v.moves[i];
        if (m.from_timeline == 1 && m.to_timeline == 1 && m.to_half_turn == 4 && m.to_square == 1) index = i;
      }
      REQUIRE(index < v.move_count);
      REQUIRE_EQ(t5d_game_apply_expanded(copy.p, index, nullptr), T5D_OK);  // the optional board, after the shift
      CHECK_EQ(moves_made(copy.p), 2);
      CHECK_EQ(moves_made(game.p), 1);
      v = expanded(copy.p);
      CHECK_EQ(v.mandatory_remaining, 0);
      CHECK_EQ(v.optional_playable, 0);
      CHECK_EQ(v.moves_made, 2);
      REQUIRE_EQ(t5d_game_apply(game.p, &on_one, nullptr), T5D_OK);
      CHECK_EQ(moves_made(game.p), 2);
    }
    // Submitted: Black's turn start.
    REQUIRE_EQ(t5d_game_submit(game.p, nullptr), T5D_OK);
    CHECK_EQ(moves_made(game.p), 0);
    v = expanded(game.p);
    CHECK_EQ(v.moves_made, 0);
    CHECK(v.mandatory_remaining > 0);
  }
  // A record's replay: every action whole, so 0 after each; a move after it
  // counts, and leaves the record.
  const std::string record = "1. e4 / e5 2. Nf3\n";
  Game game;
  REQUIRE_EQ(t5d_game_new_from_pgn(record.data(), record.size(), T5D_RULES_STANDARD, &game.p, nullptr), T5D_OK);
  for (int a = 0; a < 3; ++a) {
    REQUIRE_EQ(t5d_game_replay_next(game.p, nullptr), T5D_OK);
    CHECK_EQ(moves_made(game.p), 0);
  }
  const t5d_expansion v = expanded(game.p);
  REQUIRE(v.move_count > 0);
  REQUIRE_EQ(t5d_game_apply_expanded(game.p, 0, nullptr), T5D_OK);
  CHECK_EQ(moves_made(game.p), 1);
  CHECK_EQ(t5d_game_replay_next(game.p, nullptr), T5D_ERR_ARGUMENT);  // it left its record
  CHECK_EQ(moves_made(game.p), 1);
}

TEST(capi_layout_reports_the_structs_as_compiled) {
  // t5d_layout: what a binding checks its mirrors against at load (stage E7).
  size_t v = 0;
  REQUIRE_EQ(t5d_layout("t5d_expansion", nullptr, &v), T5D_OK);
  CHECK_EQ(v, sizeof(t5d_expansion));
  REQUIRE_EQ(t5d_layout("t5d_expansion", "acting_player", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_expansion, acting_player));
  REQUIRE_EQ(t5d_layout("t5d_expansion", "submit_possible", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_expansion, submit_possible));
  REQUIRE_EQ(t5d_layout("t5d_expansion", "submit_legal", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_expansion, submit_legal));
  REQUIRE_EQ(t5d_layout("t5d_expansion", "src_extra", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_expansion, src_extra));
  REQUIRE_EQ(t5d_layout("t5d_expansion", "mandatory_remaining", &v), T5D_OK);  // A6
  CHECK_EQ(v, offsetof(t5d_expansion, mandatory_remaining));
  REQUIRE_EQ(t5d_layout("t5d_expansion", "optional_playable", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_expansion, optional_playable));
  REQUIRE_EQ(t5d_layout("t5d_expansion", "moves_made", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_expansion, moves_made));
  REQUIRE_EQ(t5d_layout("t5d_game_summary", "moves_made", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_game_summary, moves_made));
  REQUIRE_EQ(t5d_layout("t5d_game_summary", nullptr, &v), T5D_OK);
  CHECK_EQ(v, sizeof(t5d_game_summary));
  REQUIRE_EQ(t5d_layout("t5d_move", nullptr, &v), T5D_OK);
  CHECK_EQ(v, size_t{32});
  REQUIRE_EQ(t5d_layout("t5d_move", "flags", &v), T5D_OK);
  CHECK_EQ(v, size_t{31});
  REQUIRE_EQ(t5d_layout("t5d_game_summary", "rules", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_game_summary, rules));
  for (const char* type : {"t5d_fingerprint", "t5d_info", "t5d_expand_options", "t5d_game_summary"}) {
    CHECK_EQ(t5d_layout(type, nullptr, &v), T5D_OK);
    CHECK(v > 0);
  }
  v = 7;
  CHECK_EQ(t5d_layout("t5d_state", nullptr, &v), T5D_ERR_ARGUMENT);  // opaque: no layout
  CHECK_EQ(t5d_layout("t5d_expansion", "no_such_field", &v), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_layout(nullptr, nullptr, &v), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_layout("t5d_move", nullptr, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(v, size_t{7});  // unchanged on failure
}

TEST(rule_options_decode_inverts_encode) {
  std::vector<RuleOptions> all(1);
  const auto add = [&](auto set) {
    RuleOptions r;
    set(r);
    all.push_back(r);
  };
  add([](RuleOptions& r) { r.activation = ActivationRule::Consecutive; });
  add([](RuleOptions& r) { r.initial_credit = InitialTimelineCredit::MinusZeroToBlack; });
  add([](RuleOptions& r) { r.play_activates = true; });
  add([](RuleOptions& r) { r.moves_after_shift = false; });  // Q4's Guide-literal reading
  add([](RuleOptions& r) { r.mandatory = MandatoryBoards::AllActive; });
  add([](RuleOptions& r) { r.reactivation = Reactivation::Deferred; });
  add([](RuleOptions& r) { r.turn_zero_reachable = false; });
  add([](RuleOptions& r) { r.sliders_cross_void = true; });
  add([](RuleOptions& r) { r.past_self_blocks = false; });
  add([](RuleOptions& r) { r.brawn_rank_timeline_capture = false; });
  add([](RuleOptions& r) { r.double_step_right = DoubleStepRight::HomeRank; });
  add([](RuleOptions& r) { r.timeline_double_step = false; });
  add([](RuleOptions& r) { r.double_step_over_void = true; });
  add([](RuleOptions& r) { r.branch_copies_keep_unmoved = false; });
  add([](RuleOptions& r) { r.jumps_keep_unmoved = true; });
  add([](RuleOptions& r) { r.en_passant_brawns = false; });
  add([](RuleOptions& r) { r.brawns_promote = false; });
  add([](RuleOptions& r) { r.castling_empty_between = false; });
  add([](RuleOptions& r) { r.castling_destination_test = true; });
  add([](RuleOptions& r) { r.castling_rook_path_test = true; });
  add([](RuleOptions& r) { r.castling_any_rank = false; });
  add([](RuleOptions& r) { r.castling_rook_two_away = false; });
  add([](RuleOptions& r) { r.castling_on_optional_boards = false; });
  add([](RuleOptions& r) { r.earlier_playable_jump_branches = true; });
  add([](RuleOptions& r) { r.inactive_history_is_destination = false; });
  add([](RuleOptions& r) { r.brawn_cross_board_promotion = false; });
  add([](RuleOptions& r) { r.family = RuleFamily::ReversedRoyalty; });
  add([](RuleOptions& r) {
    r.promotions = PromotionSet::of({PieceKind::Queen, PieceKind::Knight, PieceKind::Princess});
  });
  add([](RuleOptions& r) {
    r.moves_after_shift = false;
    r.past_self_blocks = false;
    r.family = RuleFamily::ReversedRoyalty;
    r.promotions = PromotionSet::of({PieceKind::RoyalQueen});
  });
  for (const RuleOptions& r : all) {
    CHECK(RuleOptions::decode(r.encode()) == r);
    CHECK_EQ(RuleOptions::decode(r.encode()).encode(), r.encode());
  }
  CHECK(RuleOptions::decode(0) == RuleOptions{});
  CHECK_THROWS_WITH(RuleOptions::decode(std::uint64_t{1} << 21), std::invalid_argument, "0x200000");
  // Bit 17 (castling_attacks = AnyBoard) was never a valid ruleset's.
  CHECK_THROWS_AS(RuleOptions::decode(std::uint64_t{1} << 17), std::invalid_argument);
  CHECK_THROWS_AS(RuleOptions::decode(std::uint64_t{1} << 50), std::invalid_argument);
  CHECK_THROWS_AS(RuleOptions::decode(std::uint64_t{5} << 28), std::invalid_argument);
}
