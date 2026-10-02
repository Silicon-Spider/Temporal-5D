// The search's C API (t5d/capi_search.h): the step API an external evaluator
// drives - gather leaves, read each leaf's expansion, deliver logits and value
// distributions - checked against the C++ search it wraps (the same
// evaluations give the same tree), its error handling (virtual loss released,
// nothing backed up), and the results, choice and persistence calls.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "t5d/capi.h"
#include "t5d/capi_search.h"
#include "t5d/mcts.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"

namespace {

struct GameDeleter {
  void operator()(t5d_game* g) const noexcept { t5d_game_destroy(g); }
};
struct SearchDeleter {
  void operator()(t5d_search* s) const noexcept { t5d_search_destroy(s); }
};
using Game = std::unique_ptr<t5d_game, GameDeleter>;
using SearchHandle = std::unique_ptr<t5d_search, SearchDeleter>;

Game variant_game(const char* name) {
  t5d_game* g = nullptr;
  REQUIRE_EQ(t5d_game_new_variant(name, T5D_RULES_STANDARD, &g, nullptr), T5D_OK);
  return Game(g);
}

t5d_search_params quick_params() {
  t5d_search_params p;
  t5d_search_params_default(&p);
  p.expand.square_features = T5D_FEATURES_NONE;
  return p;
}

SearchHandle new_search(const t5d_game* g, const t5d_search_params& p, int32_t turn_start = 1) {
  t5d_search* s = nullptr;
  char* error = nullptr;
  const t5d_status st = t5d_search_new(g, turn_start, &p, &s, &error);
  if (st != T5D_OK) {
    std::string message = error != nullptr ? error : "";
    t5d_free(error);
    throw std::runtime_error("t5d_search_new: " + message);
  }
  return SearchHandle(s);
}

// Drives a search through the step API with uniform logits and a value of
// exactly 0 - the uniform evaluator's answers - as probabilities or logits.
std::uint64_t drive(t5d_search* s, std::uint64_t playouts, int32_t value_kind, std::size_t batch) {
  std::uint64_t leaves = 0;
  for (int round = 0; round < 100000; ++round) {
    t5d_search_summary info;
    REQUIRE_EQ(t5d_search_info(s, &info), T5D_OK);
    if (info.playouts >= playouts || info.root_terminal || info.root_solved) break;
    std::size_t n = 0;
    REQUIRE_EQ(t5d_search_gather(s, batch, playouts - info.playouts, &n, nullptr), T5D_OK);
    if (n == 0) continue;
    std::vector<float> logits;
    std::vector<float> values;
    for (std::size_t i = 0; i < n; ++i) {
      t5d_leaf leaf;
      REQUIRE_EQ(t5d_search_leaf(s, i, &leaf, nullptr), T5D_OK);
      logits.insert(logits.end(), leaf.expansion.move_count, 0.0f);
      std::vector<float> v(T5D_VALUE_BINS, value_kind == T5D_VALUE_LOGITS ? -30.0f : 0.0f);
      v[25] = value_kind == T5D_VALUE_LOGITS ? 30.0f : 1.0f;  // all the mass at 0
      values.insert(values.end(), v.begin(), v.end());
    }
    REQUIRE_EQ(t5d_search_deliver(s, logits.data(), logits.size(), values.data(), values.size(),
                                  value_kind, nullptr),
               T5D_OK);
    leaves += n;
  }
  return leaves;
}

}  // namespace

TEST(capi_search_step_api_reproduces_the_cpp_search) {
  // The C step loop with the uniform evaluator's answers builds the tree the
  // C++ search builds with the uniform evaluator: the same root visits, child
  // by child.
  for (const int32_t split : {T5D_SPLIT_BOARD_MOVE, T5D_SPLIT_WHOLE_TURN}) {
    const Game g = variant_game("Standard - Two Timelines");
    t5d_search_params p = quick_params();
    p.split = split;
    p.max_actions = 96;
    p.batch_size = 4;
    const SearchHandle s = new_search(g.get(), p);
    drive(s.get(), 120, T5D_VALUE_PROBABILITIES, 4);

    t5d::mcts::SearchParams q;
    q.split = split == T5D_SPLIT_BOARD_MOVE ? t5d::mcts::TurnSplit::BoardMove : t5d::mcts::TurnSplit::WholeTurn;
    q.max_actions = 96;
    q.batch_size = 4;
    q.expand.square_features = t5d::SquareFeatures::None;
    t5d::mcts::Search reference(t5d::variant_start(*t5d::find_variant("Standard - Two Timelines")), q);
    const std::unique_ptr<t5d::mcts::Evaluator> ev = t5d::mcts::make_evaluator(t5d::mcts::BuiltinEvaluator::Uniform);
    reference.run(*ev, t5d::mcts::Limits{120});

    t5d_search_summary info;
    REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
    CHECK_EQ(info.playouts, reference.stats().playouts);
    CHECK_EQ(info.evaluations, reference.stats().evaluations);
    const std::vector<t5d::mcts::RootChild> children = reference.root_children();
    REQUIRE_EQ(info.root_children, static_cast<std::uint64_t>(children.size()));
    for (std::size_t i = 0; i < children.size(); ++i) {
      t5d_root_child c;
      std::vector<t5d_move> moves(8);
      REQUIRE_EQ(t5d_search_child(s.get(), i, &c, moves.data(), moves.size(), nullptr), T5D_OK);
      CHECK_EQ(c.visits, children[i].visits);
      CHECK_EQ(c.prior, children[i].prior);
      CHECK_EQ(c.move_count, children[i].moves.size());
      for (std::size_t m = 0; m < c.move_count && m < moves.size(); ++m) {
        CHECK_EQ(moves[m].from_timeline, children[i].moves[m].from_timeline);
        CHECK_EQ(moves[m].to_square, static_cast<int8_t>(children[i].moves[m].to_square));
        CHECK(moves[m].from_board >= 0);
        CHECK(moves[m].to_board >= 0);
      }
    }
  }
}

TEST(capi_search_takes_value_logits_or_probabilities) {
  const Game g = variant_game("Standard");
  const t5d_search_params p = quick_params();
  const SearchHandle a = new_search(g.get(), p);
  const SearchHandle b = new_search(g.get(), p);
  drive(a.get(), 50, T5D_VALUE_PROBABILITIES, 8);
  drive(b.get(), 50, T5D_VALUE_LOGITS, 8);
  t5d_search_summary ia;
  t5d_search_summary ib;
  REQUIRE_EQ(t5d_search_info(a.get(), &ia), T5D_OK);
  REQUIRE_EQ(t5d_search_info(b.get(), &ib), T5D_OK);
  CHECK_EQ(ia.playouts, ib.playouts);
  CHECK_EQ(ia.root_children, ib.root_children);
  CHECK(std::abs(ia.root_q - ib.root_q) < 1e-6);
}

TEST(capi_search_leaves_carry_the_network_input) {
  const Game g = variant_game("Standard");
  t5d_search_params p;
  t5d_search_params_default(&p);  // every board's features (the network contract)
  p.expand.relations = T5D_RELATIONS_ATTACK;
  const SearchHandle s = new_search(g.get(), p);
  std::size_t n = 0;
  REQUIRE_EQ(t5d_search_gather(s.get(), 8, T5D_NO_LIMIT, &n, nullptr), T5D_OK);
  REQUIRE_EQ(n, std::size_t{1});  // the root
  t5d_leaf leaf;
  REQUIRE_EQ(t5d_search_leaf(s.get(), 0, &leaf, nullptr), T5D_OK);
  CHECK_EQ(leaf.to_move, 0);
  CHECK_EQ(leaf.turn_start, 1);
  CHECK_EQ(leaf.expansion.move_count, std::size_t{20});
  CHECK_EQ(leaf.expansion.board_count, std::size_t{1});
  CHECK_EQ(leaf.expansion.feature_board_count, std::size_t{1});
  CHECK(leaf.expansion.piece != nullptr);
  CHECK_EQ(static_cast<int>(leaf.expansion.piece[0]), 4);  // a1: a rook (type id 4)
  CHECK(leaf.legality != nullptr);
  for (std::size_t i = 0; i < leaf.expansion.move_count; ++i) CHECK_EQ(leaf.legality[i], T5D_LEGALITY_LEGAL);
  t5d_state* state = nullptr;
  REQUIRE_EQ(t5d_search_leaf_state(s.get(), 0, &state, nullptr), T5D_OK);
  t5d_fingerprint fp;
  REQUIRE_EQ(t5d_state_fingerprint(state, &fp), T5D_OK);
  CHECK_EQ(std::string(fp.fingerprint), std::string("d574889fd9da3f2bc65249ff27249b00"));
  t5d_state_destroy(state);
  CHECK_EQ(t5d_search_leaf(s.get(), 1, &leaf, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_search_abandon(s.get()), T5D_OK);
}

TEST(capi_search_errors_release_the_batch) {
  const Game g = variant_game("Standard");
  const SearchHandle s = new_search(g.get(), quick_params());
  std::size_t n = 0;
  REQUIRE_EQ(t5d_search_gather(s.get(), 4, T5D_NO_LIMIT, &n, nullptr), T5D_OK);
  REQUIRE_EQ(n, std::size_t{1});
  // A second gather while a batch waits.
  std::size_t m = 0;
  char* error = nullptr;
  CHECK_EQ(t5d_search_gather(s.get(), 4, T5D_NO_LIMIT, &m, &error), T5D_ERR_ARGUMENT);
  CHECK(error != nullptr);
  t5d_free(error);
  // Too few logits: refused, the batch abandoned.
  std::vector<float> logits(3, 0.0f);
  std::vector<float> values(T5D_VALUE_BINS, 0.0f);
  values[25] = 1.0f;
  CHECK_EQ(t5d_search_deliver(s.get(), logits.data(), logits.size(), values.data(), values.size(),
                              T5D_VALUE_PROBABILITIES, nullptr),
           T5D_ERR_ARGUMENT);
  REQUIRE_EQ(t5d_search_gather(s.get(), 4, T5D_NO_LIMIT, &n, nullptr), T5D_OK);  // released
  REQUIRE_EQ(n, std::size_t{1});
  // A value kind that does not exist; a value that is no distribution.
  logits.assign(20, 0.0f);
  CHECK_EQ(t5d_search_deliver(s.get(), logits.data(), logits.size(), values.data(), values.size(), 7, nullptr),
           T5D_ERR_ARGUMENT);
  REQUIRE_EQ(t5d_search_gather(s.get(), 4, T5D_NO_LIMIT, &n, nullptr), T5D_OK);
  std::vector<float> zero(T5D_VALUE_BINS, 0.0f);
  CHECK_EQ(t5d_search_deliver(s.get(), logits.data(), logits.size(), zero.data(), zero.size(),
                              T5D_VALUE_PROBABILITIES, nullptr),
           T5D_ERR_ARGUMENT);
  REQUIRE_EQ(t5d_search_gather(s.get(), 4, T5D_NO_LIMIT, &n, nullptr), T5D_OK);
  CHECK_EQ(t5d_search_deliver(s.get(), logits.data(), logits.size(), values.data(), values.size(),
                              T5D_VALUE_PROBABILITIES, nullptr),
           T5D_OK);
  t5d_search_summary info;
  REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
  CHECK_EQ(info.playouts, std::uint64_t{1});
  CHECK_EQ(info.root_visits, std::uint32_t{1});
  // Bad parameters, bad handles.
  t5d_search_params p = quick_params();
  p.split = 9;
  t5d_search* bad = nullptr;
  CHECK_EQ(t5d_search_new(g.get(), 1, &p, &bad, nullptr), T5D_ERR_ARGUMENT);
  CHECK(bad == nullptr);
  p = quick_params();
  p.expand.relations = 1u << T5D_RELATION_PIN;
  CHECK_EQ(t5d_search_new(g.get(), 1, &p, &bad, nullptr), T5D_ERR_UNSUPPORTED);
  p = quick_params();
  p.split = T5D_SPLIT_WHOLE_TURN;
  CHECK_EQ(t5d_search_new(g.get(), 0, &p, &bad, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_search_new(nullptr, 1, nullptr, &bad, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_search_info(nullptr, &info), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_search_abandon(nullptr), T5D_ERR_ARGUMENT);
  size_t index = 0;
  CHECK_EQ(t5d_search_choose(nullptr, &index, nullptr, nullptr), T5D_ERR_ARGUMENT);
  t5d_search_destroy(nullptr);
}

TEST(capi_search_choose_advance_and_reply) {
  // Match play through the C API: run, choose, advance on our action, then on
  // the opponent's reply by its moves.
  const Game g = variant_game("Standard");
  t5d_search_params p = quick_params();
  p.graph = 1;
  const SearchHandle s = new_search(g.get(), p);
  REQUIRE_EQ(t5d_search_run(s.get(), T5D_EVALUATOR_RANDOM, 3, 60, nullptr), T5D_OK);
  size_t index = 0;
  int32_t reason = -1;
  REQUIRE_EQ(t5d_search_choose(s.get(), &index, &reason, nullptr), T5D_OK);
  CHECK(reason == T5D_CHOICE_LCB || reason == T5D_CHOICE_VISITS);
  int32_t legality = -1;
  REQUIRE_EQ(t5d_search_verify(s.get(), index, &legality, nullptr), T5D_OK);
  CHECK_EQ(legality, T5D_LEGALITY_LEGAL);
  size_t sampled = 0;
  REQUIRE_EQ(t5d_search_choose_by_visits(s.get(), 1.0, 0.5, &sampled, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_search_advance(s.get(), index, nullptr), T5D_OK);
  t5d_search_summary info;
  REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
  CHECK_EQ(info.root_turn_start, 1);  // one board: our move was the whole action
  // The reply: the first move of Black's first expansion, as a record.
  t5d_game* root = nullptr;
  REQUIRE_EQ(t5d_search_root_game(s.get(), &root, nullptr), T5D_OK);
  const Game root_game(root);
  t5d_expansion x;
  REQUIRE_EQ(t5d_game_expand(root, nullptr, &x, nullptr), T5D_OK);
  REQUIRE(x.move_count > 0);
  const t5d_move reply = x.moves[0];
  REQUIRE_EQ(t5d_search_run(s.get(), T5D_EVALUATOR_RANDOM, 3, 30, nullptr), T5D_OK);  // ponder
  REQUIRE_EQ(t5d_search_advance_action(s.get(), &reply, 1, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
  CHECK_EQ(info.root_turn_start, 1);
  REQUIRE_EQ(t5d_search_run(s.get(), T5D_EVALUATOR_MATERIAL, 0, 20, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_search_set_epoch(s.get(), 2, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
  CHECK_EQ(info.epoch, std::uint64_t{2});
  // A reply that is no move: refused.
  t5d_move bogus = reply;
  bogus.to_square = 63;
  bogus.to_timeline = 5;
  CHECK_EQ(t5d_search_advance_action(s.get(), &bogus, 1, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_search_advance(s.get(), 9999, nullptr), T5D_ERR_ARGUMENT);
}

TEST(capi_search_proves_a_mate_in_one) {
  const char* boards[] = {"[k4/4Q/1K3/5/5:0:1:w]"};
  t5d_game* g = nullptr;
  REQUIRE_EQ(t5d_game_new_from_fen(boards, 1, 5, 5, T5D_PARITY_INFER, T5D_TO_MOVE_INFER,
                                   T5D_RULES_STANDARD, &g, nullptr),
             T5D_OK);
  const Game game(g);
  const SearchHandle s = new_search(g, quick_params());
  REQUIRE_EQ(t5d_search_run(s.get(), T5D_EVALUATOR_UNIFORM, 0, 2000, nullptr), T5D_OK);
  t5d_search_summary info;
  REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
  CHECK_EQ(info.root_solved, 1);
  CHECK_EQ(info.root_lo, 1);
  CHECK(info.checkmates > 0);
  size_t index = 0;
  int32_t reason = -1;
  REQUIRE_EQ(t5d_search_choose(s.get(), &index, &reason, nullptr), T5D_OK);
  CHECK_EQ(reason, T5D_CHOICE_PROVEN_WIN);
  t5d_root_child c;
  REQUIRE_EQ(t5d_search_child(s.get(), index, &c, nullptr, 0, nullptr), T5D_OK);
  CHECK_EQ(c.lo, 1);
  CHECK_EQ(c.move_count, std::size_t{1});
}
