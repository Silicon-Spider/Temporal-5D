// The search's C API (t5d/capi_search.h): the step API an external evaluator
// drives - gather leaves, read each leaf's expansion, deliver logits and the
// value heads (D3) - checked against the C++ search it wraps (the same
// evaluations give the same tree), its error handling (virtual loss released,
// nothing backed up), and the results, choice and persistence calls.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
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

// One batch's evaluations through t5d_search_deliver_evaluations; an empty
// vector is "none given" (NULL, count 0) - the WDL included, which is refused.
t5d_status deliver(t5d_search* s, const std::vector<float>& logits, const std::vector<float>& submits,
                   const std::vector<float>& wdl, const std::vector<float>& values, int32_t kind,
                   const std::vector<float>& errors = {}, const std::vector<float>& moves_left = {},
                   char** error = nullptr) {
  const auto data = [](const std::vector<float>& v) { return v.empty() ? nullptr : v.data(); };
  t5d_evaluations b{};
  b.logits = data(logits);
  b.logit_count = logits.size();
  b.submit_logits = data(submits);
  b.submit_count = submits.size();
  b.wdl = data(wdl);
  b.wdl_count = wdl.size();
  b.wdl_kind = kind;
  b.value_kind = kind;
  b.values = data(values);
  b.value_count = values.size();
  b.errors = data(errors);
  b.error_count = errors.size();
  b.moves_left = data(moves_left);
  b.moves_left_count = moves_left.size();
  return t5d_search_deliver_evaluations(s, &b, error);
}

// Drives a search through the step API with uniform logits, a WDL of {1, 1, 1}
// and a value of exactly 0 - the uniform evaluator's answers - as
// probabilities or logits.
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
    std::vector<float> submits;
    std::vector<float> wdl;
    std::vector<float> values;
    for (std::size_t i = 0; i < n; ++i) {
      t5d_leaf leaf;
      REQUIRE_EQ(t5d_search_leaf(s, i, &leaf, nullptr), T5D_OK);
      CHECK_EQ(leaf.acting_player, leaf.expansion.acting_player);
      CHECK_EQ(leaf.submit_legal != 0, leaf.expansion.submit_legal != 0);
      logits.insert(logits.end(), leaf.expansion.move_count, 0.0f);
      // The submit logit where submitting is possible (D8); NaN elsewhere.
      submits.push_back(leaf.submit_possible ? 0.0f : std::nanf(""));
      wdl.insert(wdl.end(), T5D_WDL, value_kind == T5D_VALUE_LOGITS ? 0.0f : 1.0f);
      std::vector<float> v(T5D_VALUE_BINS, value_kind == T5D_VALUE_LOGITS ? -30.0f : 0.0f);
      v[25] = value_kind == T5D_VALUE_LOGITS ? 30.0f : 1.0f;  // all the mass at 0
      values.insert(values.end(), v.begin(), v.end());
    }
    REQUIRE_EQ(deliver(s, logits, submits, wdl, values, value_kind), T5D_OK);
    leaves += n;
  }
  return leaves;
}

// A deterministic, non-trivial answer for a leaf from its key alone: logits
// in quarter steps, a finite submit logit where submitting is possible, a WDL
// in eighths, a categorical of two bins off centre (0.75 and 0.25: exact in
// float32, summing to 1), an error and a moves-left in quarters.
std::uint64_t mix_key(std::uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return x;
}
float logit_of(std::uint64_t key, std::size_t i) {
  return static_cast<float>(static_cast<int>(mix_key(key + 0x9E3779B9ull * (i + 1)) % 17) - 8) * 0.25f;
}
float submit_of(std::uint64_t key) {
  return static_cast<float>(static_cast<int>(mix_key(key ^ 0x5B5Bull) % 9) - 4) * 0.5f;
}
void value_of(std::uint64_t key, float* out) {
  std::fill(out, out + T5D_VALUE_BINS, 0.0f);
  out[mix_key(key ^ 0xA1ull) % T5D_VALUE_BINS] += 0.75f;
  out[mix_key(key ^ 0xB2ull) % T5D_VALUE_BINS] += 0.25f;
}
void wdl_of_key(std::uint64_t key, float* out) {
  for (int k = 0; k < T5D_WDL; ++k) {
    out[k] = static_cast<float>(1 + mix_key(key ^ (0xC3ull + static_cast<std::uint64_t>(k))) % 8) * 0.125f;
  }
}
float error_of(std::uint64_t key) { return static_cast<float>(mix_key(key ^ 0xE5ull) % 5) * 0.25f; }
float moves_left_of(std::uint64_t key) { return static_cast<float>(mix_key(key ^ 0xF6ull) % 400) * 0.25f; }

// The same answers as a C++ evaluator.
class Hashed final : public t5d::mcts::Evaluator {
 public:
  void evaluate(std::span<const t5d::mcts::EvalRequest> batch, std::span<t5d::mcts::Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const std::size_t moves = batch[i].expansion->moves().size();
      out[i].logits.resize(moves);
      for (std::size_t m = 0; m < moves; ++m) out[i].logits[m] = logit_of(batch[i].key, m);
      out[i].submit_logit = batch[i].expansion->submit_possible() ? submit_of(batch[i].key) : std::nanf("");
      wdl_of_key(batch[i].key, out[i].wdl.data());
      std::array<float, T5D_VALUE_BINS> value{};
      value_of(batch[i].key, value.data());
      out[i].value = value;
      out[i].error = error_of(batch[i].key);
      out[i].moves_left = moves_left_of(batch[i].key);
    }
  }
};

// Drives a search through the step API with Hashed's answers.
void drive_hashed(t5d_search* s, std::uint64_t playouts, std::size_t batch) {
  for (int round = 0; round < 100000; ++round) {
    t5d_search_summary info;
    REQUIRE_EQ(t5d_search_info(s, &info), T5D_OK);
    if (info.playouts >= playouts || info.root_terminal || info.root_solved) break;
    std::size_t n = 0;
    REQUIRE_EQ(t5d_search_gather(s, batch, playouts - info.playouts, &n, nullptr), T5D_OK);
    if (n == 0) continue;
    std::vector<float> logits;
    std::vector<float> submits;
    std::vector<float> wdl(n * T5D_WDL);
    std::vector<float> values(n * T5D_VALUE_BINS);
    std::vector<float> errors;
    std::vector<float> moves_left;
    for (std::size_t i = 0; i < n; ++i) {
      t5d_leaf leaf;
      REQUIRE_EQ(t5d_search_leaf(s, i, &leaf, nullptr), T5D_OK);
      for (std::size_t m = 0; m < leaf.expansion.move_count; ++m) logits.push_back(logit_of(leaf.key, m));
      submits.push_back(leaf.submit_possible ? submit_of(leaf.key) : std::nanf(""));
      wdl_of_key(leaf.key, wdl.data() + i * T5D_WDL);
      value_of(leaf.key, values.data() + i * T5D_VALUE_BINS);
      errors.push_back(error_of(leaf.key));
      moves_left.push_back(moves_left_of(leaf.key));
    }
    REQUIRE_EQ(deliver(s, logits, submits, wdl, values, T5D_VALUE_PROBABILITIES, errors, moves_left), T5D_OK);
  }
}

}  // namespace

TEST(capi_search_step_api_reproduces_the_cpp_search) {
  // The C step loop with Hashed's answers - logits, finite submit logits (D8),
  // WDLs with draws, categoricals off centre, error and moves-left outputs
  // (D3) - builds the graph the C++ search builds with the same answers, bit
  // for bit (H15's digest: every node's and edge's statistics, priors and
  // backed-up draws included, so every submit prior below the root too), under
  // either uncertainty signal (6m).
  for (const int32_t split : {T5D_SPLIT_BOARD_MOVE, T5D_SPLIT_WHOLE_TURN}) {
    for (const int32_t graph : {0, 1}) {
      const int32_t source = graph != 0 ? T5D_UNCERTAINTY_ERROR_HEAD : T5D_UNCERTAINTY_CATEGORICAL;
      const Game g = variant_game("Standard - Two Timelines");
      t5d_search_params p = quick_params();
      p.split = split;
      p.graph = graph;
      p.max_actions = 96;
      p.batch_size = 4;
      p.uncertainty_source = source;
      p.variance_scaled_cpuct = 1;  // E2 on: the signal steers selection
      const SearchHandle s = new_search(g.get(), p);
      drive_hashed(s.get(), 200, 4);

      t5d::mcts::SearchParams q;
      q.split = split == T5D_SPLIT_BOARD_MOVE ? t5d::mcts::TurnSplit::BoardMove : t5d::mcts::TurnSplit::WholeTurn;
      q.graph = graph != 0;
      q.max_actions = 96;
      q.batch_size = 4;
      q.uncertainty_source = static_cast<t5d::mcts::UncertaintySource>(source);
      q.variance_scaled_cpuct = true;
      q.expand.square_features = t5d::SquareFeatures::None;
      t5d::mcts::Search reference(t5d::variant_start(*t5d::find_variant("Standard - Two Timelines")), q);
      Hashed ev;
      reference.run(ev, t5d::mcts::Limits{200});

      t5d_search_summary info;
      REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
      CHECK_EQ(info.playouts, reference.stats().playouts);
      CHECK_EQ(info.evaluations, reference.stats().evaluations);
      std::uint64_t digest = 0;
      REQUIRE_EQ(t5d_search_digest(s.get(), &digest), T5D_OK);
      CHECK_EQ(digest, reference.digest());
      const std::vector<t5d::mcts::RootChild> children = reference.root_children();
      REQUIRE_EQ(info.root_children, static_cast<std::uint64_t>(children.size()));
      for (std::size_t i = 0; i < children.size(); ++i) {
        t5d_root_child c;
        std::vector<t5d_move> moves(8);
        REQUIRE_EQ(t5d_search_child(s.get(), i, &c, moves.data(), moves.size(), nullptr), T5D_OK);
        CHECK_EQ(c.visits, children[i].visits);
        CHECK_EQ(c.prior, children[i].prior);
        CHECK_EQ(c.q, children[i].q);
        CHECK_EQ(c.d, children[i].d);
        CHECK_EQ(c.evaluated != 0, children[i].evaluated);
        if (c.evaluated != 0) {
          for (int k = 0; k < T5D_WDL; ++k) CHECK_EQ(c.network_wdl[k], children[i].network_wdl[static_cast<std::size_t>(k)]);
          CHECK_EQ(c.network_moves_left, children[i].network_moves_left);
          CHECK_EQ(c.network_error, children[i].network_error);
          CHECK(c.network_moves_left >= 0 && c.network_error >= 0);  // given at every leaf
        }
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
  // The digest's own errors.
  std::uint64_t d = 0;
  CHECK_EQ(t5d_search_digest(nullptr, &d), T5D_ERR_ARGUMENT);
}

TEST(capi_search_takes_value_logits_or_probabilities) {
  const Game g = variant_game("Standard");
  const t5d_search_params p = quick_params();
  // The root alone, its WDL (0.7, 0.2, 0.1) - value 0.6, draw 0.2 from its
  // player's view - as probabilities and as logits: win, draw, loss in that
  // order, not mirrored. The categorical (bin 10, -0.6) is not the value: D3.
  for (const int32_t kind : {T5D_VALUE_PROBABILITIES, T5D_VALUE_LOGITS}) {
    const SearchHandle s = new_search(g.get(), p);
    std::size_t n = 0;
    REQUIRE_EQ(t5d_search_gather(s.get(), 1, T5D_NO_LIMIT, &n, nullptr), T5D_OK);
    REQUIRE_EQ(n, std::size_t{1});
    std::vector<float> logits(20, 0.0f);
    std::vector<float> wdl{0.7f, 0.2f, 0.1f};
    if (kind == T5D_VALUE_LOGITS) {
      for (float& x : wdl) x = std::log(x);
    }
    std::vector<float> v(T5D_VALUE_BINS, kind == T5D_VALUE_LOGITS ? -30.0f : 0.0f);
    v[10] = kind == T5D_VALUE_LOGITS ? 30.0f : 1.0f;
    REQUIRE_EQ(deliver(s.get(), logits, {}, wdl, v, kind, {1.5f}, {12.0f}), T5D_OK);
    t5d_search_summary info;
    REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
    CHECK(std::abs(info.root_q - 0.6) < 1e-6);
    CHECK(std::abs(info.root_d - 0.2) < 1e-6);
    CHECK_EQ(info.root_evaluated, 1);
    CHECK(std::abs(info.root_wdl[0] - 0.7) < 1e-6 && std::abs(info.root_wdl[1] - 0.2) < 1e-6 &&
          std::abs(info.root_wdl[2] - 0.1) < 1e-6);
    CHECK_EQ(info.root_error, 1.5);
    CHECK_EQ(info.root_moves_left, 12.0);
  }
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
  CHECK_EQ(leaf.acting_player, 0);
  CHECK_EQ(leaf.submit_possible, 0);  // a turn start: the obligations are not met
  CHECK_EQ(leaf.submit_legal, 0);
  CHECK_EQ(leaf.expansion.acting_player, 0);
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
  // A new epoch while a batch waits (its evaluations belong to the old one).
  CHECK_EQ(t5d_search_set_epoch(s.get(), 3, nullptr), T5D_ERR_ARGUMENT);
  const std::vector<float> wdl{1.0f, 1.0f, 1.0f};
  std::vector<float> values(T5D_VALUE_BINS, 0.0f);
  values[25] = 1.0f;
  const std::vector<float> l(20, 0.0f);
  // Each malformed batch: refused (T5D_ERR_ARGUMENT, or _UNSUPPORTED), the batch abandoned.
  const auto refused = [&](t5d_status want, const auto& call) {
    CHECK_EQ(call(), want);
    REQUIRE_EQ(t5d_search_gather(s.get(), 4, T5D_NO_LIMIT, &n, nullptr), T5D_OK);  // released
    REQUIRE_EQ(n, std::size_t{1});
  };
  // Submit logits: one per leaf or none; never infinite.
  refused(T5D_ERR_ARGUMENT, [&] { return deliver(s.get(), l, {0.0f, 0.0f}, wdl, values, T5D_VALUE_PROBABILITIES); });
  refused(T5D_ERR_ARGUMENT, [&] {
    return deliver(s.get(), l, {std::numeric_limits<float>::infinity()}, wdl, values, T5D_VALUE_PROBABILITIES);
  });
  // Too few logits.
  refused(T5D_ERR_ARGUMENT, [&] { return deliver(s.get(), {0.0f, 0.0f, 0.0f}, {}, wdl, values, T5D_VALUE_PROBABILITIES); });
  // D3's WDL: required, 3 per leaf, a distribution (as probabilities: finite, non-negative, not all zero).
  refused(T5D_ERR_ARGUMENT, [&] { return deliver(s.get(), l, {}, {}, values, T5D_VALUE_PROBABILITIES); });
  refused(T5D_ERR_ARGUMENT, [&] { return deliver(s.get(), l, {}, {1.0f, 1.0f}, values, T5D_VALUE_PROBABILITIES); });
  refused(T5D_ERR_ARGUMENT, [&] { return deliver(s.get(), l, {}, {0.0f, 0.0f, 0.0f}, values, T5D_VALUE_PROBABILITIES); });
  refused(T5D_ERR_ARGUMENT, [&] { return deliver(s.get(), l, {}, {1.0f, -1.0f, 1.0f}, values, T5D_VALUE_PROBABILITIES); });
  refused(T5D_ERR_ARGUMENT, [&] {
    return deliver(s.get(), l, {}, {0.0f, std::numeric_limits<float>::infinity(), 0.0f}, {}, T5D_VALUE_LOGITS);
  });
  // A kind that does not exist; a categorical that is no distribution, or the wrong count.
  refused(T5D_ERR_ARGUMENT, [&] { return deliver(s.get(), l, {}, wdl, values, 7); });
  refused(T5D_ERR_ARGUMENT, [&] {
    return deliver(s.get(), l, {}, wdl, std::vector<float>(T5D_VALUE_BINS, 0.0f), T5D_VALUE_PROBABILITIES);
  });
  refused(T5D_ERR_ARGUMENT, [&] { return deliver(s.get(), l, {}, wdl, {1.0f}, T5D_VALUE_PROBABILITIES); });
  // The error head and moves-left: one per leaf or none; >= 0 or NaN.
  refused(T5D_ERR_ARGUMENT, [&] { return deliver(s.get(), l, {}, wdl, values, T5D_VALUE_PROBABILITIES, {-0.5f}); });
  refused(T5D_ERR_ARGUMENT, [&] { return deliver(s.get(), l, {}, wdl, values, T5D_VALUE_PROBABILITIES, {0.5f, 0.5f}); });
  refused(T5D_ERR_ARGUMENT, [&] {
    return deliver(s.get(), l, {}, wdl, values, T5D_VALUE_PROBABILITIES, {}, {std::numeric_limits<float>::infinity()});
  });
  // Per-board logits: reserved for D2's flat form, not built.
  refused(T5D_ERR_UNSUPPORTED, [&] {
    t5d_evaluations b{};
    b.logits = l.data();
    b.logit_count = l.size();
    b.wdl = wdl.data();
    b.wdl_count = wdl.size();
    const float board[] = {0.0f};
    b.board_logits = board;
    b.board_logit_count = 1;
    return t5d_search_deliver_evaluations(s.get(), &b, nullptr);
  });
  CHECK_EQ(t5d_search_deliver_evaluations(s.get(), nullptr, nullptr), T5D_ERR_ARGUMENT);  // no batch given
  REQUIRE_EQ(t5d_search_gather(s.get(), 4, T5D_NO_LIMIT, &n, nullptr), T5D_OK);
  // The optional heads absent (NULL, 0); NaN for "none given" at a leaf.
  CHECK_EQ(deliver(s.get(), l, {}, wdl, {}, T5D_VALUE_PROBABILITIES, {std::nanf("")}, {std::nanf("")}), T5D_OK);
  // No batch waiting: delivering is an error, and nothing changes.
  CHECK_EQ(deliver(s.get(), l, {}, wdl, values, T5D_VALUE_PROBABILITIES), T5D_ERR_ARGUMENT);
  // Zero leaves asked for: an error; any number at all: bounded, never overflowed.
  CHECK_EQ(t5d_search_gather(s.get(), 0, T5D_NO_LIMIT, &n, nullptr), T5D_ERR_ARGUMENT);
  REQUIRE_EQ(t5d_search_gather(s.get(), SIZE_MAX, 4, &n, nullptr), T5D_OK);
  CHECK(n <= 4);
  CHECK_EQ(t5d_search_abandon(s.get()), T5D_OK);
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
  p.uncertainty_source = 2;  // 6m: the categorical or the error head
  CHECK_EQ(t5d_search_new(g.get(), 1, &p, &bad, nullptr), T5D_ERR_ARGUMENT);
  p = quick_params();
  p.dirichlet_shaping = 2;  // a flag: 0 or 1
  CHECK_EQ(t5d_search_new(g.get(), 1, &p, &bad, nullptr), T5D_ERR_ARGUMENT);
  p = quick_params();
  p.root_policy_temperature = 0;
  CHECK_EQ(t5d_search_new(g.get(), 1, &p, &bad, nullptr), T5D_ERR_ARGUMENT);
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
  // One board: our move met the obligations, but White still acts (D8) - the
  // root's one child is submit.
  CHECK_EQ(info.root_turn_start, 0);
  REQUIRE_EQ(t5d_search_run(s.get(), T5D_EVALUATOR_RANDOM, 3, 10, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
  REQUIRE_EQ(info.root_children, std::uint64_t{1});
  t5d_root_child submit;
  REQUIRE_EQ(t5d_search_child(s.get(), 0, &submit, nullptr, 0, nullptr), T5D_OK);
  CHECK_EQ(submit.submit, 1);
  CHECK_EQ(submit.move_count, std::size_t{0});
  CHECK_EQ(submit.legality, T5D_LEGALITY_LEGAL);
  REQUIRE_EQ(t5d_search_advance(s.get(), 0, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
  CHECK_EQ(info.root_turn_start, 1);
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
  // A reply that is no move, or none at a turn start: refused, nothing changed.
  std::uint64_t before = 0;
  REQUIRE_EQ(t5d_search_digest(s.get(), &before), T5D_OK);
  t5d_move bogus = reply;
  bogus.to_square = 63;
  bogus.to_timeline = 5;
  CHECK_EQ(t5d_search_advance_action(s.get(), &bogus, 1, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_search_advance_action(s.get(), nullptr, 0, nullptr), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_search_advance(s.get(), 9999, nullptr), T5D_ERR_ARGUMENT);
  std::uint64_t after = 0;
  REQUIRE_EQ(t5d_search_digest(s.get(), &after), T5D_OK);
  CHECK_EQ(after, before);
}

TEST(capi_search_takes_the_moves_made_from_the_game_a6) {
  // Register A6: a search's root takes the game handle's count of moves made
  // in its action; turn_start must agree with it; every leaf's expansion
  // carries its own count, and a root game handle the root's.
  const auto summary = [](t5d_game* g) {
    t5d_game_summary info{};
    REQUIRE_EQ(t5d_game_info(g, &info), T5D_OK);
    return info;
  };
  const Game g = variant_game("Standard - Two Timelines");
  const t5d_search_params p = quick_params();
  t5d_search* bad = nullptr;
  CHECK_EQ(t5d_search_new(g.get(), 0, &p, &bad, nullptr), T5D_ERR_ARGUMENT);  // no move made: a turn start
  CHECK(bad == nullptr);
  {
    const SearchHandle s = new_search(g.get(), p, 1);
    std::size_t n = 0;
    REQUIRE_EQ(t5d_search_gather(s.get(), 1, T5D_NO_LIMIT, &n, nullptr), T5D_OK);
    REQUIRE_EQ(n, std::size_t{1});
    t5d_leaf leaf;
    REQUIRE_EQ(t5d_search_leaf(s.get(), 0, &leaf, nullptr), T5D_OK);
    CHECK_EQ(leaf.expansion.moves_made, 0);
    CHECK_EQ(leaf.expansion.mandatory_remaining, 2);  // -0 and +0 at T1
    CHECK_EQ(leaf.expansion.optional_playable, 0);
    CHECK_EQ(t5d_search_abandon(s.get()), T5D_OK);
  }
  // One move into White's action: the count is 1.
  t5d_expansion x;
  REQUIRE_EQ(t5d_game_expand(g.get(), nullptr, &x, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_game_apply_expanded(g.get(), 0, nullptr), T5D_OK);
  REQUIRE_EQ(summary(g.get()).moves_made, 1);
  CHECK_EQ(t5d_search_new(g.get(), 1, &p, &bad, nullptr), T5D_ERR_ARGUMENT);
  const SearchHandle s = new_search(g.get(), p, 0);
  t5d_search* sharing = nullptr;
  CHECK_EQ(t5d_search_new_sharing(s.get(), g.get(), 1, &p, &sharing, nullptr), T5D_ERR_ARGUMENT);
  REQUIRE_EQ(t5d_search_new_sharing(s.get(), g.get(), 0, &p, &sharing, nullptr), T5D_OK);
  t5d_search_destroy(sharing);
  t5d_game* root = nullptr;
  REQUIRE_EQ(t5d_search_root_game(s.get(), &root, nullptr), T5D_OK);
  CHECK_EQ(summary(root).moves_made, 1);
  t5d_game_destroy(root);
  // The leaves: each with its own count - and every turn start (count 0) owes a
  // board (T5).
  REQUIRE_EQ(t5d_search_run(s.get(), T5D_EVALUATOR_RANDOM, 3, 120, nullptr), T5D_OK);
  std::size_t n = 0;
  REQUIRE_EQ(t5d_search_gather(s.get(), 16, T5D_NO_LIMIT, &n, nullptr), T5D_OK);
  int deeper = 0;
  for (std::size_t i = 0; i < n; ++i) {
    t5d_leaf leaf;
    REQUIRE_EQ(t5d_search_leaf(s.get(), i, &leaf, nullptr), T5D_OK);
    CHECK(leaf.expansion.moves_made >= 0);
    if (leaf.expansion.moves_made == 0) CHECK(leaf.expansion.mandatory_remaining > 0);
    if (leaf.expansion.moves_made > 1) ++deeper;
    CHECK_EQ(leaf.expansion.mandatory_remaining == 0, leaf.expansion.submit_possible == 1);
  }
  CHECK_EQ(t5d_search_abandon(s.get()), T5D_OK);
  CHECK(n > 0);
  CHECK(deeper > 0);  // moves after the first: counts above the root's
  // Re-rooted on a move: the root's count goes on.
  t5d_search_summary info;
  REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
  std::size_t move_child = info.root_children;
  for (std::size_t i = 0; i < info.root_children && move_child == info.root_children; ++i) {
    t5d_root_child c;
    REQUIRE_EQ(t5d_search_child(s.get(), i, &c, nullptr, 0, nullptr), T5D_OK);
    if (c.submit == 0 && c.visits > 0) move_child = i;
  }
  REQUIRE(move_child < info.root_children);
  REQUIRE_EQ(t5d_search_advance(s.get(), move_child, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_search_root_game(s.get(), &root, nullptr), T5D_OK);
  CHECK_EQ(summary(root).moves_made, 2);
  t5d_game_destroy(root);

  // Two moves into an action: a count the turn-start flag could never give (0
  // or 1 - a search that read the handle's count as that flag passes every
  // check above, the count of 2 reached there by re-rooting, not from the
  // handle). Three White boards in the present, all mandatory (T5): two
  // played, one owed - the root's moves have a count of 3.
  const char* three[] = {"[3k/4/4/K3:-1:1:w]", "[3k/4/4/K3:0:1:w]", "[3k/4/4/K3:+1:1:w]"};
  t5d_game* raw = nullptr;
  REQUIRE_EQ(t5d_game_new_from_fen(three, 3, 4, 4, T5D_PARITY_INFER, T5D_TO_MOVE_INFER, T5D_RULES_STANDARD, &raw,
                                   nullptr),
             T5D_OK);
  const Game h(raw);
  for (const int32_t timeline : {-1, 0}) {
    REQUIRE_EQ(t5d_game_expand(h.get(), nullptr, &x, nullptr), T5D_OK);
    std::size_t index = x.move_count;
    for (std::size_t i = 0; i < x.move_count && index == x.move_count; ++i) {
      const t5d_move& m = x.moves[i];
      if (m.from_timeline == timeline && m.to_timeline == timeline && m.from_half_turn == m.to_half_turn) index = i;
    }
    REQUIRE(index < x.move_count);
    REQUIRE_EQ(t5d_game_apply_expanded(h.get(), index, nullptr), T5D_OK);
  }
  REQUIRE_EQ(summary(h.get()).moves_made, 2);
  CHECK_EQ(t5d_search_new(h.get(), 1, &p, &bad, nullptr), T5D_ERR_ARGUMENT);
  const SearchHandle two = new_search(h.get(), p, 0);
  sharing = nullptr;
  REQUIRE_EQ(t5d_search_new_sharing(two.get(), h.get(), 0, &p, &sharing, nullptr), T5D_OK);
  const SearchHandle shared(sharing);
  for (t5d_search* each : {two.get(), shared.get()}) {
    REQUIRE_EQ(t5d_search_root_game(each, &root, nullptr), T5D_OK);
    CHECK_EQ(summary(root).moves_made, 2);
    t5d_game_destroy(root);
    REQUIRE_EQ(t5d_search_gather(each, 1, T5D_NO_LIMIT, &n, nullptr), T5D_OK);
    REQUIRE_EQ(n, std::size_t{1});
    t5d_leaf leaf;
    REQUIRE_EQ(t5d_search_leaf(each, 0, &leaf, nullptr), T5D_OK);  // the root's own expansion
    CHECK_EQ(leaf.expansion.moves_made, 2);
    CHECK_EQ(leaf.expansion.mandatory_remaining, 1);
    CHECK_EQ(leaf.submit_possible, 0);
    CHECK_EQ(t5d_search_abandon(each), T5D_OK);
  }
  // One move below the root: 3.
  REQUIRE_EQ(t5d_search_run(two.get(), T5D_EVALUATOR_RANDOM, 3, 40, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_search_info(two.get(), &info), T5D_OK);
  move_child = info.root_children;
  for (std::size_t i = 0; i < info.root_children && move_child == info.root_children; ++i) {
    t5d_root_child c;
    REQUIRE_EQ(t5d_search_child(two.get(), i, &c, nullptr, 0, nullptr), T5D_OK);
    if (c.submit == 0 && c.visits > 0) move_child = i;
  }
  REQUIRE(move_child < info.root_children);
  REQUIRE_EQ(t5d_search_advance(two.get(), move_child, nullptr), T5D_OK);
  REQUIRE_EQ(t5d_search_root_game(two.get(), &root, nullptr), T5D_OK);
  CHECK_EQ(summary(root).moves_made, 3);
  t5d_game_destroy(root);
}

TEST(capi_search_game_expand_gives_a_leafs_legal_moves_i15) {
  // I15: a record's network input must be rebuildable - t5d_game_expand with
  // legal_moves and the search's legality budget gives a search leaf's moves,
  // in its order. At the Standard start every move is legal; with White's king
  // in check down the a-file one of its three moves is not.
  const char* checked[] = {"[r2k/4/4/K3:0:1:w]"};
  t5d_game* raw = nullptr;
  REQUIRE_EQ(t5d_game_new_from_fen(checked, 1, 4, 4, T5D_PARITY_INFER, T5D_TO_MOVE_WHITE, T5D_RULES_STANDARD, &raw,
                                   nullptr),
             T5D_OK);
  const Game check(raw);
  const Game standard = variant_game("Standard");
  for (t5d_game* g : {standard.get(), check.get()}) {
    const t5d_search_params p = quick_params();
    const SearchHandle s = new_search(g, p);
    std::size_t n = 0;
    REQUIRE_EQ(t5d_search_gather(s.get(), 1, T5D_NO_LIMIT, &n, nullptr), T5D_OK);
    REQUIRE_EQ(n, std::size_t{1});
    t5d_leaf leaf;
    REQUIRE_EQ(t5d_search_leaf(s.get(), 0, &leaf, nullptr), T5D_OK);
    const std::vector<t5d_move> want(leaf.expansion.moves, leaf.expansion.moves + leaf.expansion.move_count);
    const std::uint64_t root_leaf_key = leaf.key;
    t5d_expand_options o{};
    o.window_k = p.expand.window_k;
    o.square_features = p.expand.square_features;
    o.legal_moves = 1;
    o.legality_checks = p.legality_checks;
    o.legality_nodes = p.legality_nodes;
    t5d_expansion x;
    REQUIRE_EQ(t5d_game_expand(g, &o, &x, nullptr), T5D_OK);
    REQUIRE_EQ(x.move_count, want.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
      CHECK(std::memcmp(&x.moves[i], &want[i], sizeof(t5d_move)) == 0);
    }
    o.legal_moves = 0;  // the pseudo-legal moves: one more where the king is in check
    REQUIRE_EQ(t5d_game_expand(g, &o, &x, nullptr), T5D_OK);
    CHECK_EQ(x.move_count, want.size() + (g == check.get() ? 1u : 0u));
    o.legal_moves = 2;
    CHECK_EQ(t5d_game_expand(g, &o, &x, nullptr), T5D_ERR_ARGUMENT);
    // The root's key is its leaf's (the memo's identity, with the count and the epoch).
    REQUIRE_EQ(t5d_search_abandon(s.get()), T5D_OK);
    REQUIRE_EQ(t5d_search_run(s.get(), T5D_EVALUATOR_RANDOM, 2, 40, nullptr), T5D_OK);
    t5d_search_summary info;
    REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
    CHECK_EQ(info.root_key, root_leaf_key);
    CHECK_EQ(info.root_cpuct, p.cpuct);  // E2 off
    CHECK(info.root_best_since >= 1 && info.root_best_since <= info.root_visits);
    for (std::size_t i = 0; i < info.root_children; ++i) {
      t5d_root_child c;
      REQUIRE_EQ(t5d_search_child(s.get(), i, &c, nullptr, 0, nullptr), T5D_OK);
      CHECK_EQ(c.prior_noise_free, c.prior);  // no root noise
      CHECK(c.visits == 0 || c.child_key != 0);
      CHECK_EQ(c.exploration_scale, 1.0);
      CHECK(c.boost == 0.0 && c.forced_visits == 0 && c.steered_visits == 0);
      CHECK(c.value_variance >= 0.0);
    }
  }
  // A search's expand options never take the filter: its own parameters decide its leaves.
  t5d_search_params p = quick_params();
  p.expand.legal_moves = 1;
  t5d_search* bad = nullptr;
  CHECK_EQ(t5d_search_new(standard.get(), 1, &p, &bad, nullptr), T5D_ERR_ARGUMENT);
  CHECK(bad == nullptr);
}

TEST(capi_search_reports_its_layouts) {
  size_t v = 0;
  REQUIRE_EQ(t5d_search_layout("t5d_leaf", nullptr, &v), T5D_OK);
  CHECK_EQ(v, sizeof(t5d_leaf));
  REQUIRE_EQ(t5d_search_layout("t5d_leaf", "submit_legal", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_leaf, submit_legal));
  REQUIRE_EQ(t5d_search_layout("t5d_leaf", "expansion", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_leaf, expansion));
  // The embedded provisional structs come last (appending to t5d/capi.h's
  // structs moves none of this header's fields).
  CHECK(offsetof(t5d_leaf, expansion) > offsetof(t5d_leaf, legality));
  CHECK(sizeof(t5d_leaf) - offsetof(t5d_leaf, expansion) < sizeof(t5d_expansion) + alignof(t5d_leaf));
  CHECK(offsetof(t5d_search_params, expand) > offsetof(t5d_search_params, root_policy_temperature));
  CHECK(offsetof(t5d_search_params, dirichlet_concentration) > offsetof(t5d_search_params, uncertainty_source));
  CHECK(sizeof(t5d_search_params) - offsetof(t5d_search_params, expand) <
        sizeof(t5d_expand_options) + alignof(t5d_search_params));
  REQUIRE_EQ(t5d_search_layout("t5d_search_params", "expand", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_search_params, expand));
  REQUIRE_EQ(t5d_search_layout("t5d_search_params", nullptr, &v), T5D_OK);
  CHECK_EQ(v, sizeof(t5d_search_params));
  REQUIRE_EQ(t5d_search_layout("t5d_search_params", "eval_cache_bytes", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_search_params, eval_cache_bytes));
  REQUIRE_EQ(t5d_search_layout("t5d_search_summary", "epoch", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_search_summary, epoch));
  REQUIRE_EQ(t5d_search_layout("t5d_root_child", nullptr, &v), T5D_OK);
  CHECK_EQ(v, sizeof(t5d_root_child));
  // Engine ABI round 2 (D3): appended, or (t5d_search_params) in front of `expand`.
  REQUIRE_EQ(t5d_search_layout("t5d_search_params", "uncertainty_source", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_search_params, uncertainty_source));
  REQUIRE_EQ(t5d_search_layout("t5d_search_summary", "root_wdl", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_search_summary, root_wdl));
  CHECK(offsetof(t5d_search_summary, root_evaluated) > offsetof(t5d_search_summary, last_stop));
  REQUIRE_EQ(t5d_search_layout("t5d_root_child", "network_error", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_root_child, network_error));
  CHECK(offsetof(t5d_root_child, d) > offsetof(t5d_root_child, move_count));
  REQUIRE_EQ(t5d_search_layout("t5d_evaluations", nullptr, &v), T5D_OK);
  CHECK_EQ(v, sizeof(t5d_evaluations));
  REQUIRE_EQ(t5d_search_layout("t5d_evaluations", "board_logit_count", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_evaluations, board_logit_count));
  // Round 2c (I15, I2): appended.
  CHECK(offsetof(t5d_search_summary, root_key) > offsetof(t5d_search_summary, root_error));
  CHECK(offsetof(t5d_root_child, child_key) > offsetof(t5d_root_child, network_error));
  for (const char* field : {"root_key", "root_cpuct", "root_best_since"}) {
    CHECK_EQ(t5d_search_layout("t5d_search_summary", field, &v), T5D_OK);
  }
  for (const char* field : {"child_key", "prior_noise_free", "value_variance", "boost", "exploration_scale",
                            "forced_visits", "steered_visits"}) {
    CHECK_EQ(t5d_search_layout("t5d_root_child", field, &v), T5D_OK);
  }
  REQUIRE_EQ(t5d_layout("t5d_expand_options", "legality_nodes", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_expand_options, legality_nodes));
  t5d_search_params defaults;
  t5d_search_params_default(&defaults);
  CHECK_EQ(defaults.expand.legal_moves, 0);
  CHECK_EQ(defaults.uncertainty_source, T5D_UNCERTAINTY_CATEGORICAL);  // 6m: the register's default reading
  // E1, I19 (round 2b): the placeholders, through the C defaults.
  const t5d::mcts::SearchParams cpp;
  CHECK_EQ(defaults.dirichlet_concentration, 0.03 * 361);
  CHECK_EQ(defaults.dirichlet_concentration, cpp.dirichlet_concentration);
  CHECK_EQ(defaults.dirichlet_shaping, 0);
  CHECK_EQ(defaults.dirichlet_shaping_share, cpp.dirichlet_shaping_share);
  CHECK_EQ(defaults.dirichlet_shaping_cap, cpp.dirichlet_shaping_cap);
  CHECK_EQ(defaults.root_policy_temperature, 1.0);
  for (const char* field : {"dirichlet_concentration", "dirichlet_shaping", "dirichlet_shaping_share",
                            "dirichlet_shaping_cap", "root_policy_temperature"}) {
    CHECK_EQ(t5d_search_layout("t5d_search_params", field, &v), T5D_OK);
  }
  CHECK_EQ(t5d_search_layout("t5d_leaf", "to_move", &v), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_search_layout("t5d_no_such", nullptr, &v), T5D_ERR_ARGUMENT);
  CHECK_EQ(t5d_search_layout(nullptr, nullptr, &v), T5D_ERR_ARGUMENT);
}

TEST(capi_search_proves_a_mate_in_one) {
  const char* boards[] = {"[k4/4Q/1K3/5/5:0:1:w]"};
  t5d_game* g = nullptr;
  REQUIRE_EQ(t5d_game_new_from_fen(boards, 1, 5, 5, T5D_PARITY_INFER, T5D_TO_MOVE_INFER,
                                   T5D_RULES_STANDARD, &g, nullptr),
             T5D_OK);
  const Game game(g);
  t5d_search_params defaults;
  t5d_search_params_default(&defaults);
  CHECK_EQ(defaults.proof_channel, 0);  // off by rule (H41)
  for (const int32_t solver : {1, 0}) {
    t5d_search_params p = quick_params();
    p.proof_channel = solver;
    const SearchHandle s = new_search(g, p);
    REQUIRE_EQ(t5d_search_run(s.get(), T5D_EVALUATOR_UNIFORM, 0, 2000, nullptr), T5D_OK);
    t5d_search_summary info;
    REQUIRE_EQ(t5d_search_info(s.get(), &info), T5D_OK);
    CHECK(info.checkmates > 0);  // the terminal is valued as its result either way
    size_t index = 0;
    int32_t reason = -1;
    REQUIRE_EQ(t5d_search_choose(s.get(), &index, &reason, nullptr), T5D_OK);
    t5d_root_child c;
    REQUIRE_EQ(t5d_search_child(s.get(), index, &c, nullptr, 0, nullptr), T5D_OK);
    CHECK_EQ(c.move_count, std::size_t{1});
    if (solver == 1) {
      CHECK_EQ(info.root_solved, 1);
      CHECK_EQ(info.root_lo, 1);
      CHECK_EQ(reason, T5D_CHOICE_PROVEN_WIN);
      CHECK_EQ(c.lo, 1);
    } else {
      // Nothing proven: the mate is found by its value alone.
      CHECK_EQ(info.root_solved, 0);
      CHECK_EQ(info.root_lo, -1);
      CHECK(reason != T5D_CHOICE_PROVEN_WIN);
      CHECK_EQ(c.lo, -1);
      CHECK(c.q > 0.5);
    }
  }
}

TEST(capi_search_shares_its_cache_and_runs_within_budgets) {
  // C14 through the binding: a fresh search per move sharing the cache
  // (t5d_search_new_sharing) finds what the first one evaluated.
  const Game g = variant_game("Standard - Two Timelines");
  t5d_search_params p = quick_params();
  const SearchHandle first = new_search(g.get(), p);
  REQUIRE_EQ(t5d_search_run(first.get(), T5D_EVALUATOR_UNIFORM, 0, 200, nullptr), T5D_OK);
  t5d_search_summary a;
  REQUIRE_EQ(t5d_search_info(first.get(), &a), T5D_OK);
  CHECK(a.cache_entries > 0);
  CHECK(a.cache_bytes > 0);
  CHECK(a.memory_bytes > 0);
  CHECK_EQ(a.last_stop, T5D_STOP_PLAYOUTS);
  t5d_search* raw = nullptr;
  REQUIRE_EQ(t5d_search_new_sharing(first.get(), g.get(), 1, &p, &raw, nullptr), T5D_OK);
  const SearchHandle second(raw);
  REQUIRE_EQ(t5d_search_run(second.get(), T5D_EVALUATOR_UNIFORM, 0, 200, nullptr), T5D_OK);
  t5d_search_summary b;
  REQUIRE_EQ(t5d_search_info(second.get(), &b), T5D_OK);
  CHECK(b.cache_hits > 0);
  CHECK_EQ(b.evaluations, std::uint64_t{0});  // the same search again: everything cached
  CHECK_EQ(t5d_search_new_sharing(nullptr, g.get(), 1, &p, &raw, nullptr), T5D_ERR_ARGUMENT);
  // A run within a node budget stops there, and says so.
  const SearchHandle s = new_search(g.get(), p);
  t5d_search_limits limits{20000, T5D_NO_LIMIT, 300, T5D_NO_LIMIT, 0.0};
  int32_t stop = -1;
  REQUIRE_EQ(t5d_search_run_limits(s.get(), T5D_EVALUATOR_RANDOM, 3, &limits, &stop, nullptr), T5D_OK);
  CHECK_EQ(stop, T5D_STOP_NODES);
  t5d_search_summary c;
  REQUIRE_EQ(t5d_search_info(s.get(), &c), T5D_OK);
  CHECK(c.nodes >= 300);
  CHECK_EQ(c.last_stop, T5D_STOP_NODES);
  limits = t5d_search_limits{20000, T5D_NO_LIMIT, T5D_NO_LIMIT, c.memory_bytes, 0.0};
  REQUIRE_EQ(t5d_search_run_limits(s.get(), T5D_EVALUATOR_RANDOM, 3, &limits, &stop, nullptr), T5D_OK);
  CHECK_EQ(stop, T5D_STOP_MEMORY);
  CHECK_EQ(t5d_search_run_limits(s.get(), T5D_EVALUATOR_RANDOM, 3, nullptr, &stop, nullptr), T5D_ERR_ARGUMENT);
  limits = t5d_search_limits{T5D_NO_LIMIT, T5D_NO_LIMIT, T5D_NO_LIMIT, T5D_NO_LIMIT, 0.02};
  const SearchHandle timed = new_search(g.get(), p);
  REQUIRE_EQ(t5d_search_run_limits(timed.get(), T5D_EVALUATOR_RANDOM, 3, &limits, &stop, nullptr), T5D_OK);
  CHECK_EQ(stop, T5D_STOP_DEADLINE);
  limits.seconds = -1;
  CHECK_EQ(t5d_search_run_limits(timed.get(), T5D_EVALUATOR_RANDOM, 3, &limits, &stop, nullptr), T5D_ERR_ARGUMENT);
  size_t v = 0;
  REQUIRE_EQ(t5d_search_layout("t5d_search_limits", nullptr, &v), T5D_OK);
  CHECK_EQ(v, sizeof(t5d_search_limits));
  REQUIRE_EQ(t5d_search_layout("t5d_search_summary", "last_stop", &v), T5D_OK);
  CHECK_EQ(v, offsetof(t5d_search_summary, last_stop));
}
