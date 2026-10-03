// The superseded-board asset map (stage E9; spec 11.2 rung 4, kept by 11.4;
// register H10): each board assessed once, ever, deterministically; a board
// where the traveller is far ahead flagged, its travel moves boosted.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/assets.hpp"
#include "t5d/check.hpp"
#include "t5d/movegen.hpp"
#include "t5d/prover.hpp"
#include "t5d/search.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_game_tree.hpp"
#include "test_mate_refs.hpp"
#include "test_play.hpp"
#include "test_positions.hpp"
#include "test_reference.hpp"
#include "test_rulesets.hpp"
#include "test_support.hpp"
#include "test_v3_support.hpp"

using namespace t5d;
using namespace t5d_test::v3;
using t5d_test::describe_position;
using t5d_test::game;
using t5d_test::GameTree;
using t5d_test::GameValue;
using t5d_test::playable;
using t5d_test::RestrictedGame;

// ---- the asset map ---------------------------------------------------------------------------

TEST(assets_each_board_is_assessed_once_ever) {
  t5d_test::Rng rng(31);
  for (const t5d_test::SparseStart& start : t5d_test::sparse_starts()) {
    GameState s = game(start.boards, start.size);
    AssetMap map;
    AssetMap twin;
    for (int turn = 0; turn < 6; ++turn) {
      const std::size_t before = map.boards();
      const std::size_t fresh = map.refresh(s);
      CHECK_EQ(map.boards(), before + fresh);
      CHECK_EQ(map.assessments(), static_cast<std::uint64_t>(map.boards()));
      CHECK_EQ(map.refresh(s), std::size_t{0});  // again: nothing new
      twin.refresh(s);
      // Deterministic: another map, the same assets.
      for (const TimelineRef& t : s.multiverse().timelines()) {
        for (int h = t->first_half_turn(); h < t->last_half_turn(); ++h) {
          const BoardAsset* a = map.asset(s, t->position(), h);
          const BoardAsset* b = twin.asset(s, t->position(), h);
          CHECK(a != nullptr && b != nullptr);
          if (a == nullptr || b == nullptr) continue;
          CHECK_EQ(a->value, b->value);
          CHECK_EQ(a->flagged, b->flagged);
          CHECK_EQ(a->nodes, b->nodes);
        }
        // A head is not superseded: no asset.
        CHECK(map.asset(s, t->position(), t->last_half_turn()) == nullptr);
      }
      const std::optional<std::vector<Move>> a = t5d_test::random_legal_action(s, rng);
      if (!a) break;
      for (const Move& m : *a) s.apply(m);
      s.submit();
    }
    CHECK_EQ(map.assessments(), static_cast<std::uint64_t>(map.boards()));
  }
}

TEST(assets_flag_a_board_where_the_traveller_is_far_ahead) {
  // White's superseded board (turn 1, White to move) holds White's queen
  // against a bare king; Black's board after it is even. White's travel move
  // there gets the boost; a move within a board none.
  const GameState s = game({"[k4/5/5/5/Q3K:0:1:w]", "[k4/5/5/5/4K:0:1:b]", "[k4/5/5/5/3K1:0:2:w]"}, k5x5);
  AssetMap map;
  CHECK_EQ(map.refresh(s), std::size_t{2});
  const BoardAsset* white_board = map.asset(s, 0, 2);
  const BoardAsset* black_board = map.asset(s, 0, 3);
  CHECK(white_board != nullptr && black_board != nullptr);
  if (white_board == nullptr || black_board == nullptr) return;
  CHECK(white_board->traveller == Colour::White);
  CHECK_EQ(white_board->material, 900);
  CHECK(white_board->flagged);
  CHECK(white_board->boost > 0);
  CHECK(black_board->traveller == Colour::Black);
  CHECK(!black_board->flagged);
  Move travel;
  travel.from_timeline = 0;
  travel.from_half_turn = 4;
  travel.to_timeline = 0;
  travel.to_half_turn = 2;
  travel.piece = Piece(PieceKind::King, Colour::White);
  CHECK(map.boost(s, travel) > 0);
  Move within = travel;
  within.to_half_turn = 4;
  CHECK(map.boost(s, within) == 0.0);
  // A royal the traveller can take there at once: worth royal_value.
  const GameState r = game({"[k4/5/5/5/R3K:0:1:w]", "[k4/5/5/5/R2K1:0:1:b]", "[k4/5/5/5/R2K1:0:2:w]"}, k5x5);
  AssetMap rmap;
  rmap.refresh(r);
  const BoardAsset* royal = rmap.asset(r, 0, 2);
  CHECK(royal != nullptr && royal->royal_capture && royal->value == rmap.options().royal_value);
}
