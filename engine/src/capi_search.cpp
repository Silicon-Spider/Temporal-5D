// The main search's C API (t5d/capi_search.h): a thin layer over t5d/mcts.hpp
// that catches every exception at the boundary.

#include "t5d/capi_search.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "capi_internal.hpp"
#include "t5d/legal.hpp"
#include "t5d/mcts.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/types.hpp"

static_assert(static_cast<int>(t5d::MoveLegality::Legal) == T5D_LEGALITY_LEGAL);
static_assert(static_cast<int>(t5d::MoveLegality::Illegal) == T5D_LEGALITY_ILLEGAL);
static_assert(static_cast<int>(t5d::MoveLegality::Unknown) == T5D_LEGALITY_UNKNOWN);
static_assert(sizeof(t5d::MoveLegality) == sizeof(uint8_t));
static_assert(T5D_VALUE_BINS == t5d::mcts::kValueBins);
static_assert(static_cast<int>(t5d::mcts::ChoiceReason::ProvenWin) == T5D_CHOICE_PROVEN_WIN);
static_assert(static_cast<int>(t5d::mcts::ChoiceReason::SecureChild) == T5D_CHOICE_SECURE_CHILD);
static_assert(static_cast<int>(t5d::mcts::ChoiceReason::Lcb) == T5D_CHOICE_LCB);
static_assert(static_cast<int>(t5d::mcts::ChoiceReason::Visits) == T5D_CHOICE_VISITS);
static_assert(static_cast<int>(t5d::mcts::ChoiceReason::AllLost) == T5D_CHOICE_ALL_LOST);
static_assert(static_cast<int>(t5d::mcts::StopReason::Playouts) == T5D_STOP_PLAYOUTS);
static_assert(static_cast<int>(t5d::mcts::StopReason::Evaluations) == T5D_STOP_EVALUATIONS);
static_assert(static_cast<int>(t5d::mcts::StopReason::Finished) == T5D_STOP_FINISHED);
static_assert(static_cast<int>(t5d::mcts::StopReason::Exhausted) == T5D_STOP_EXHAUSTED);
static_assert(static_cast<int>(t5d::mcts::StopReason::Nodes) == T5D_STOP_NODES);
static_assert(static_cast<int>(t5d::mcts::StopReason::Memory) == T5D_STOP_MEMORY);
static_assert(static_cast<int>(t5d::mcts::StopReason::Deadline) == T5D_STOP_DEADLINE);

struct t5d_search {
  explicit t5d_search(t5d::mcts::Search s) : search(std::move(s)) {}
  t5d::mcts::Search search;
};

namespace {

using t5d::mcts::Search;
using t5d::mcts::SearchParams;

char* duplicate(std::string_view text) noexcept {
  auto* copy = static_cast<char*>(std::malloc(text.size() + 1));
  if (copy == nullptr) return nullptr;
  if (!text.empty()) std::memcpy(copy, text.data(), text.size());
  copy[text.size()] = '\0';
  return copy;
}

template <class Body>
t5d_status guarded(char** error, Body&& body) noexcept {
  if (error != nullptr) *error = nullptr;
  const auto fail = [&](t5d_status status, std::string_view message) {
    if (error != nullptr) *error = duplicate(message);
    return status;
  };
  try {
    return body();
  } catch (const t5d::capi_internal::Unsupported& e) {
    return fail(T5D_ERR_UNSUPPORTED, e.what());
  } catch (const std::bad_alloc&) {
    return fail(T5D_ERR_MEMORY, "out of memory");
  } catch (const std::invalid_argument& e) {  // the caller's error: an argument, or the call's order
    return fail(T5D_ERR_ARGUMENT, e.what());
  } catch (const std::exception& e) {  // any other logic_error (a bounds check, a broken invariant): ours
    return fail(T5D_ERR_INTERNAL, e.what());
  } catch (...) {
    return fail(T5D_ERR_INTERNAL, "unknown error");
  }
}

int32_t flag(bool b) noexcept { return b ? 1 : 0; }

bool boolean(int32_t v, const char* name) {
  if (v != 0 && v != 1) throw std::invalid_argument(std::string(name) + " must be 0 or 1");
  return v == 1;
}

t5d::LegalityOptions legality_of(uint64_t checks, uint64_t nodes) {
  t5d::LegalityOptions o = t5d::LegalityOptions::budgeted(checks, nodes);
  return o;
}

SearchParams params_from(const t5d_search_params* c) {
  SearchParams p;
  if (c == nullptr) return p;
  switch (c->split) {
    case T5D_SPLIT_BOARD_MOVE: p.split = t5d::mcts::TurnSplit::BoardMove; break;
    case T5D_SPLIT_WHOLE_TURN: p.split = t5d::mcts::TurnSplit::WholeTurn; break;
    default: throw std::invalid_argument("split must be T5D_SPLIT_BOARD_MOVE or _WHOLE_TURN");
  }
  switch (c->action_prior) {
    case T5D_ACTION_PRIOR_SUM: p.action_prior = t5d::mcts::ActionPrior::SumLogits; break;
    case T5D_ACTION_PRIOR_MEAN: p.action_prior = t5d::mcts::ActionPrior::MeanLogits; break;
    default: throw std::invalid_argument("action_prior must be T5D_ACTION_PRIOR_SUM or _MEAN");
  }
  p.max_actions = static_cast<std::size_t>(c->max_actions);
  p.listing_nodes = c->listing_nodes;
  p.submit_share = c->submit_share;
  p.legal_moves_only = boolean(c->legal_moves_only, "legal_moves_only");
  p.legality = legality_of(c->legality_checks, c->legality_nodes);
  p.unknown_prior_scale = c->unknown_prior_scale;
  p.verify_unknown = boolean(c->verify_unknown, "verify_unknown");
  p.verify_legality = legality_of(c->verify_checks, c->verify_nodes);
  p.cpuct = c->cpuct;
  p.widening = boolean(c->widening, "widening");
  p.widening_alpha = c->widening_alpha;
  p.widening_scale = c->widening_scale;
  p.widening_min = c->widening_min;
  p.fpu_reduction = c->fpu_reduction;
  p.root_noise = boolean(c->root_noise, "root_noise");
  p.dirichlet_alpha = c->dirichlet_alpha;
  p.dirichlet_epsilon = c->dirichlet_epsilon;
  p.policy_temperature = c->policy_temperature;
  p.variance_scaled_cpuct = boolean(c->variance_scaled_cpuct, "variance_scaled_cpuct");
  p.cpuct_variance_scale = c->cpuct_variance_scale;
  p.variance_reference = c->variance_reference;
  p.variance_prior_weight = c->variance_prior_weight;
  p.cpuct_scale_min = c->cpuct_scale_min;
  p.cpuct_scale_max = c->cpuct_scale_max;
  p.uncertainty_weighted_backup = boolean(c->uncertainty_weighted_backup, "uncertainty_weighted_backup");
  p.variance_bonus = boolean(c->variance_bonus, "variance_bonus");
  p.variance_bonus_scale = c->variance_bonus_scale;
  p.variance_bonus_cap = c->variance_bonus_cap;
  p.graph = boolean(c->graph, "graph");
  switch (c->node_key) {
    case T5D_NODE_KEY_STATE: p.node_key = t5d::mcts::NodeKey::State; break;
    case T5D_NODE_KEY_STATE_AND_ORIGINS: p.node_key = t5d::mcts::NodeKey::StateAndOrigins; break;
    default: throw std::invalid_argument("node_key must be T5D_NODE_KEY_STATE or _STATE_AND_ORIGINS");
  }
  p.residual_epsilon = c->residual_epsilon;
  p.solver = boolean(c->solver, "solver");
  p.solved_visit_cap = c->solved_visit_cap;
  switch (c->final_rule) {
    case T5D_FINAL_VISITS: p.final_rule = t5d::mcts::FinalRule::Visits; break;
    case T5D_FINAL_LCB: p.final_rule = t5d::mcts::FinalRule::Lcb; break;
    default: throw std::invalid_argument("final_rule must be T5D_FINAL_VISITS or _LCB");
  }
  p.lcb_floor = c->lcb_floor;
  p.lcb_z = c->lcb_z;
  p.lcb_categorical_weight = c->lcb_categorical_weight;
  p.secure_child_a = c->secure_child_a;
  p.batch_size = c->batch_size;
  p.virtual_loss = c->virtual_loss;
  p.synchronous = boolean(c->synchronous, "synchronous");
  p.seed = c->seed;
  p.expand = t5d::capi_internal::expand_options(&c->expand);
  p.mate_nodes = c->mate_nodes;
  p.mate_bytes = static_cast<std::size_t>(c->mate_bytes);
  if (c->turn_cap_turns < 0 || c->turn_cap_boards < 0) {
    throw std::invalid_argument("the turn cap's limits must be 0 (none) or more");
  }
  p.turn_cap.turns = c->turn_cap_turns;
  p.turn_cap.boards = c->turn_cap_boards;
  p.eval_cache_entries = static_cast<std::size_t>(c->eval_cache_entries);
  p.eval_cache_bytes = static_cast<std::size_t>(c->eval_cache_bytes);
  return p;
}

t5d::mcts::BuiltinEvaluator builtin_of(int32_t evaluator) {
  switch (evaluator) {
    case T5D_EVALUATOR_UNIFORM: return t5d::mcts::BuiltinEvaluator::Uniform;
    case T5D_EVALUATOR_RANDOM: return t5d::mcts::BuiltinEvaluator::Random;
    case T5D_EVALUATOR_MATERIAL: return t5d::mcts::BuiltinEvaluator::Material;
    default: throw std::invalid_argument("evaluator must be T5D_EVALUATOR_UNIFORM, _RANDOM or _MATERIAL");
  }
}

// A board's index in a multiverse's canonical order, or -1.
int32_t board_index(const t5d::Multiverse& mv, int p, int h) noexcept {
  int32_t index = 0;
  for (const t5d::TimelineRef& t : mv.timelines()) {
    if (t->position() == p) return t->contains(h) ? index + (h - t->first_half_turn()) : -1;
    index += t->length();
  }
  return -1;
}

uint8_t type_id(t5d::Piece p) noexcept {
  return p.empty() ? uint8_t{0} : static_cast<uint8_t>(t5d::index_of(p.kind()) + 1);
}

// A move as a record, its boards and cells those of `mv`'s canonical order.
t5d_move record_of(const t5d::Move& m, const t5d::Multiverse& mv) noexcept {
  const int width = mv.board_size().width;
  const auto cell = [width](int square) {
    return square < 0 ? int8_t{-1} : static_cast<int8_t>(t5d::rank_of(square) * width + t5d::file_of(square));
  };
  t5d_move r{};
  r.from_timeline = m.from_timeline;
  r.from_half_turn = m.from_half_turn;
  r.to_timeline = m.to_timeline;
  r.to_half_turn = m.to_half_turn;
  r.from_board = board_index(mv, m.from_timeline, m.from_half_turn);
  r.to_board = board_index(mv, m.to_timeline, m.to_half_turn);
  r.from_square = m.from_square;
  r.to_square = m.to_square;
  r.from_cell = cell(m.from_square);
  r.to_cell = cell(m.to_square);
  r.piece = type_id(m.piece);
  r.captured = type_id(m.captured);
  r.promotion = m.promotes() ? static_cast<uint8_t>(t5d::index_of(m.promotion) + 1) : uint8_t{0};
  r.flags = static_cast<uint8_t>(m.flags | (m.physical() ? 0u : T5D_MOVE_TRAVEL));
  return r;
}

// The generated move of `state` a record names (as t5d_game_apply matches it).
t5d::Move move_of(const t5d::GameState& state, const t5d_move& r) {
  t5d::MoveList moves;
  t5d::append_piece_moves(state, r.from_timeline, r.from_square, moves);
  for (const t5d::Move& m : moves) {
    const int promotion = m.promotes() ? t5d::index_of(m.promotion) + 1 : 0;
    if (m.from_half_turn == r.from_half_turn && m.to_timeline == r.to_timeline &&
        m.to_half_turn == r.to_half_turn && m.to_square == r.to_square && promotion == r.promotion) {
      return m;
    }
  }
  throw std::invalid_argument("no pseudo-legal move of the player to move goes from (" +
                              std::to_string(r.from_timeline) + ", " + std::to_string(r.from_half_turn) +
                              ", " + std::to_string(r.from_square) + ") to (" +
                              std::to_string(r.to_timeline) + ", " + std::to_string(r.to_half_turn) +
                              ", " + std::to_string(r.to_square) + ")");
}

std::array<float, t5d::mcts::kValueBins> value_from(const float* v, int32_t kind) {
  std::array<float, t5d::mcts::kValueBins> out{};
  if (kind == T5D_VALUE_PROBABILITIES) {
    for (std::size_t k = 0; k < out.size(); ++k) out[k] = v[k];
    return out;
  }
  if (kind != T5D_VALUE_LOGITS) {
    throw std::invalid_argument("value_kind must be T5D_VALUE_PROBABILITIES or T5D_VALUE_LOGITS");
  }
  double top = -std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k < out.size(); ++k) {
    if (!std::isfinite(v[k])) throw std::invalid_argument("a value logit is not finite");
    top = std::max(top, static_cast<double>(v[k]));
  }
  double sum = 0;
  std::array<double, t5d::mcts::kValueBins> e{};
  for (std::size_t k = 0; k < out.size(); ++k) {
    e[k] = std::exp(static_cast<double>(v[k]) - top);
    sum += e[k];
  }
  for (std::size_t k = 0; k < out.size(); ++k) out[k] = static_cast<float>(e[k] / sum);
  return out;
}

// t5d_search_layout's table: each struct's size (field nullptr) and every
// member's offset, as this library was compiled (t5d_layout's contract).
struct LayoutEntry {
  const char* type;
  const char* field;
  size_t value;
};

#define T5D_SEARCH_SIZE(T) LayoutEntry{#T, nullptr, sizeof(T)}
#define T5D_SEARCH_AT(T, F) LayoutEntry{#T, #F, offsetof(T, F)}

constexpr LayoutEntry kLayout[] = {
    T5D_SEARCH_SIZE(t5d_search_params),
    T5D_SEARCH_AT(t5d_search_params, split),
    T5D_SEARCH_AT(t5d_search_params, action_prior),
    T5D_SEARCH_AT(t5d_search_params, max_actions),
    T5D_SEARCH_AT(t5d_search_params, listing_nodes),
    T5D_SEARCH_AT(t5d_search_params, submit_share),
    T5D_SEARCH_AT(t5d_search_params, legal_moves_only),
    T5D_SEARCH_AT(t5d_search_params, legality_checks),
    T5D_SEARCH_AT(t5d_search_params, legality_nodes),
    T5D_SEARCH_AT(t5d_search_params, unknown_prior_scale),
    T5D_SEARCH_AT(t5d_search_params, verify_unknown),
    T5D_SEARCH_AT(t5d_search_params, verify_checks),
    T5D_SEARCH_AT(t5d_search_params, verify_nodes),
    T5D_SEARCH_AT(t5d_search_params, cpuct),
    T5D_SEARCH_AT(t5d_search_params, widening),
    T5D_SEARCH_AT(t5d_search_params, widening_alpha),
    T5D_SEARCH_AT(t5d_search_params, widening_scale),
    T5D_SEARCH_AT(t5d_search_params, widening_min),
    T5D_SEARCH_AT(t5d_search_params, fpu_reduction),
    T5D_SEARCH_AT(t5d_search_params, root_noise),
    T5D_SEARCH_AT(t5d_search_params, dirichlet_alpha),
    T5D_SEARCH_AT(t5d_search_params, dirichlet_epsilon),
    T5D_SEARCH_AT(t5d_search_params, policy_temperature),
    T5D_SEARCH_AT(t5d_search_params, variance_scaled_cpuct),
    T5D_SEARCH_AT(t5d_search_params, cpuct_variance_scale),
    T5D_SEARCH_AT(t5d_search_params, variance_reference),
    T5D_SEARCH_AT(t5d_search_params, variance_prior_weight),
    T5D_SEARCH_AT(t5d_search_params, cpuct_scale_min),
    T5D_SEARCH_AT(t5d_search_params, cpuct_scale_max),
    T5D_SEARCH_AT(t5d_search_params, uncertainty_weighted_backup),
    T5D_SEARCH_AT(t5d_search_params, variance_bonus),
    T5D_SEARCH_AT(t5d_search_params, variance_bonus_scale),
    T5D_SEARCH_AT(t5d_search_params, variance_bonus_cap),
    T5D_SEARCH_AT(t5d_search_params, graph),
    T5D_SEARCH_AT(t5d_search_params, node_key),
    T5D_SEARCH_AT(t5d_search_params, residual_epsilon),
    T5D_SEARCH_AT(t5d_search_params, solver),
    T5D_SEARCH_AT(t5d_search_params, solved_visit_cap),
    T5D_SEARCH_AT(t5d_search_params, final_rule),
    T5D_SEARCH_AT(t5d_search_params, lcb_floor),
    T5D_SEARCH_AT(t5d_search_params, lcb_z),
    T5D_SEARCH_AT(t5d_search_params, lcb_categorical_weight),
    T5D_SEARCH_AT(t5d_search_params, secure_child_a),
    T5D_SEARCH_AT(t5d_search_params, batch_size),
    T5D_SEARCH_AT(t5d_search_params, virtual_loss),
    T5D_SEARCH_AT(t5d_search_params, synchronous),
    T5D_SEARCH_AT(t5d_search_params, seed),
    T5D_SEARCH_AT(t5d_search_params, mate_nodes),
    T5D_SEARCH_AT(t5d_search_params, mate_bytes),
    T5D_SEARCH_AT(t5d_search_params, turn_cap_turns),
    T5D_SEARCH_AT(t5d_search_params, turn_cap_boards),
    T5D_SEARCH_AT(t5d_search_params, eval_cache_entries),
    T5D_SEARCH_AT(t5d_search_params, eval_cache_bytes),
    T5D_SEARCH_AT(t5d_search_params, expand),

    T5D_SEARCH_SIZE(t5d_leaf),
    T5D_SEARCH_AT(t5d_leaf, key),
    T5D_SEARCH_AT(t5d_leaf, epoch),
    T5D_SEARCH_AT(t5d_leaf, acting_player),
    T5D_SEARCH_AT(t5d_leaf, submit_possible),
    T5D_SEARCH_AT(t5d_leaf, submit_legal),
    T5D_SEARCH_AT(t5d_leaf, legality),
    T5D_SEARCH_AT(t5d_leaf, expansion),

    T5D_SEARCH_SIZE(t5d_search_summary),
    T5D_SEARCH_AT(t5d_search_summary, playouts),
    T5D_SEARCH_AT(t5d_search_summary, evaluations),
    T5D_SEARCH_AT(t5d_search_summary, cache_hits),
    T5D_SEARCH_AT(t5d_search_summary, nodes),
    T5D_SEARCH_AT(t5d_search_summary, nodes_created),
    T5D_SEARCH_AT(t5d_search_summary, transposition_hits),
    T5D_SEARCH_AT(t5d_search_summary, checkmates),
    T5D_SEARCH_AT(t5d_search_summary, stalemates),
    T5D_SEARCH_AT(t5d_search_summary, turn_caps),
    T5D_SEARCH_AT(t5d_search_summary, unknown_verdicts),
    T5D_SEARCH_AT(t5d_search_summary, unknown_edges),
    T5D_SEARCH_AT(t5d_search_summary, verified_illegal),
    T5D_SEARCH_AT(t5d_search_summary, root_turn_start),
    T5D_SEARCH_AT(t5d_search_summary, root_terminal),
    T5D_SEARCH_AT(t5d_search_summary, root_solved),
    T5D_SEARCH_AT(t5d_search_summary, root_lo),
    T5D_SEARCH_AT(t5d_search_summary, root_hi),
    T5D_SEARCH_AT(t5d_search_summary, root_visits),
    T5D_SEARCH_AT(t5d_search_summary, root_q),
    T5D_SEARCH_AT(t5d_search_summary, root_children),
    T5D_SEARCH_AT(t5d_search_summary, epoch),
    T5D_SEARCH_AT(t5d_search_summary, memory_bytes),
    T5D_SEARCH_AT(t5d_search_summary, cache_entries),
    T5D_SEARCH_AT(t5d_search_summary, cache_bytes),
    T5D_SEARCH_AT(t5d_search_summary, cache_purged),
    T5D_SEARCH_AT(t5d_search_summary, last_stop),

    T5D_SEARCH_SIZE(t5d_search_limits),
    T5D_SEARCH_AT(t5d_search_limits, playouts),
    T5D_SEARCH_AT(t5d_search_limits, evaluations),
    T5D_SEARCH_AT(t5d_search_limits, nodes),
    T5D_SEARCH_AT(t5d_search_limits, bytes),
    T5D_SEARCH_AT(t5d_search_limits, seconds),

    T5D_SEARCH_SIZE(t5d_root_child),
    T5D_SEARCH_AT(t5d_root_child, visits),
    T5D_SEARCH_AT(t5d_root_child, submit),
    T5D_SEARCH_AT(t5d_root_child, legality),
    T5D_SEARCH_AT(t5d_root_child, lo),
    T5D_SEARCH_AT(t5d_root_child, hi),
    T5D_SEARCH_AT(t5d_root_child, prior),
    T5D_SEARCH_AT(t5d_root_child, q),
    T5D_SEARCH_AT(t5d_root_child, move_count),
};

#undef T5D_SEARCH_SIZE
#undef T5D_SEARCH_AT

}  // namespace

extern "C" {

t5d_status t5d_search_layout(const char* type, const char* field, size_t* out) {
  if (type == nullptr || out == nullptr) return T5D_ERR_ARGUMENT;
  for (const LayoutEntry& e : kLayout) {
    if (std::strcmp(e.type, type) != 0) continue;
    const bool match = field == nullptr ? e.field == nullptr
                                        : e.field != nullptr && std::strcmp(e.field, field) == 0;
    if (match) {
      *out = e.value;
      return T5D_OK;
    }
  }
  return T5D_ERR_ARGUMENT;
}

void t5d_search_params_default(t5d_search_params* out) {
  if (out == nullptr) return;
  const SearchParams p;
  *out = t5d_search_params{};
  out->split = p.split == t5d::mcts::TurnSplit::BoardMove ? T5D_SPLIT_BOARD_MOVE : T5D_SPLIT_WHOLE_TURN;
  out->action_prior = p.action_prior == t5d::mcts::ActionPrior::SumLogits ? T5D_ACTION_PRIOR_SUM
                                                                           : T5D_ACTION_PRIOR_MEAN;
  out->max_actions = p.max_actions;
  out->listing_nodes = p.listing_nodes;
  out->submit_share = p.submit_share;
  out->legal_moves_only = flag(p.legal_moves_only);
  out->legality_checks = p.legality.checks;
  out->legality_nodes = p.legality.search_nodes;
  out->unknown_prior_scale = p.unknown_prior_scale;
  out->verify_unknown = flag(p.verify_unknown);
  out->verify_checks = p.verify_legality.checks;
  out->verify_nodes = p.verify_legality.search_nodes;
  out->cpuct = p.cpuct;
  out->widening = flag(p.widening);
  out->widening_alpha = p.widening_alpha;
  out->widening_scale = p.widening_scale;
  out->widening_min = p.widening_min;
  out->fpu_reduction = p.fpu_reduction;
  out->root_noise = flag(p.root_noise);
  out->dirichlet_alpha = p.dirichlet_alpha;
  out->dirichlet_epsilon = p.dirichlet_epsilon;
  out->policy_temperature = p.policy_temperature;
  out->variance_scaled_cpuct = flag(p.variance_scaled_cpuct);
  out->cpuct_variance_scale = p.cpuct_variance_scale;
  out->variance_reference = p.variance_reference;
  out->variance_prior_weight = p.variance_prior_weight;
  out->cpuct_scale_min = p.cpuct_scale_min;
  out->cpuct_scale_max = p.cpuct_scale_max;
  out->uncertainty_weighted_backup = flag(p.uncertainty_weighted_backup);
  out->variance_bonus = flag(p.variance_bonus);
  out->variance_bonus_scale = p.variance_bonus_scale;
  out->variance_bonus_cap = p.variance_bonus_cap;
  out->graph = flag(p.graph);
  out->node_key = p.node_key == t5d::mcts::NodeKey::State ? T5D_NODE_KEY_STATE : T5D_NODE_KEY_STATE_AND_ORIGINS;
  out->residual_epsilon = p.residual_epsilon;
  out->solver = flag(p.solver);
  out->solved_visit_cap = p.solved_visit_cap;
  out->final_rule = p.final_rule == t5d::mcts::FinalRule::Visits ? T5D_FINAL_VISITS : T5D_FINAL_LCB;
  out->lcb_floor = p.lcb_floor;
  out->lcb_z = p.lcb_z;
  out->lcb_categorical_weight = p.lcb_categorical_weight;
  out->secure_child_a = p.secure_child_a;
  out->batch_size = static_cast<uint32_t>(p.batch_size);
  out->virtual_loss = p.virtual_loss;
  out->synchronous = flag(p.synchronous);
  out->seed = p.seed;
  out->expand.window_k = p.expand.window_k;
  out->expand.square_features = T5D_FEATURES_ALL;
  out->expand.relations = 0;
  out->mate_nodes = p.mate_nodes;
  out->mate_bytes = p.mate_bytes;
  out->turn_cap_turns = p.turn_cap.turns;
  out->turn_cap_boards = p.turn_cap.boards;
  out->eval_cache_entries = p.eval_cache_entries;
  out->eval_cache_bytes = p.eval_cache_bytes;
}

t5d_status t5d_search_new(const t5d_game* game, int32_t turn_start, const t5d_search_params* params,
                          t5d_search** out, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (out == nullptr) throw std::invalid_argument("out is NULL");
    *out = nullptr;
    if (game == nullptr) throw std::invalid_argument("game is NULL");
    const bool start = boolean(turn_start, "turn_start");
    const SearchParams p = params_from(params);
    *out = new t5d_search(Search(t5d::capi_internal::game_state(*game), p, start));
    return T5D_OK;
  });
}

t5d_status t5d_search_new_sharing(const t5d_search* donor, const t5d_game* game, int32_t turn_start,
                                  const t5d_search_params* params, t5d_search** out, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (out == nullptr) throw std::invalid_argument("out is NULL");
    *out = nullptr;
    if (donor == nullptr || game == nullptr) throw std::invalid_argument("donor or game is NULL");
    const bool start = boolean(turn_start, "turn_start");
    const SearchParams p = params_from(params);
    *out = new t5d_search(Search(t5d::capi_internal::game_state(*game), p, start, donor->search.shared_cache()));
    return T5D_OK;
  });
}

void t5d_search_destroy(t5d_search* search) { delete search; }

t5d_status t5d_search_gather(t5d_search* search, size_t max_leaves, uint64_t max_playouts,
                             size_t* count, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr || count == nullptr) throw std::invalid_argument("search or count is NULL");
    *count = 0;
    *count = search->search.gather(max_leaves, max_playouts);
    return T5D_OK;
  });
}

t5d_status t5d_search_leaf(t5d_search* search, size_t index, t5d_leaf* out, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr || out == nullptr) throw std::invalid_argument("search or out is NULL");
    const std::span<const t5d::mcts::EvalRequest> requests = search->search.requests();
    if (index >= requests.size()) {
      throw std::invalid_argument("leaf " + std::to_string(index) + " of " + std::to_string(requests.size()));
    }
    const t5d::mcts::EvalRequest& r = requests[index];
    *out = t5d_leaf{};
    out->key = r.key;
    out->epoch = r.epoch;
    out->acting_player = t5d::index_of(r.expansion->acting_player());
    const bool ply = search->search.params().split == t5d::mcts::TurnSplit::BoardMove;
    out->submit_possible = flag(ply && r.expansion->submit_possible());
    out->submit_legal = flag(ply && r.expansion->submit_legal());
    t5d::capi_internal::describe_expansion(*r.expansion, out->expansion);
    const std::span<const t5d::MoveLegality> legality = r.expansion->move_legality();
    out->legality = legality.empty() ? nullptr : reinterpret_cast<const uint8_t*>(legality.data());
    return T5D_OK;
  });
}

t5d_status t5d_search_leaf_state(const t5d_search* search, size_t index, t5d_state** out, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr || out == nullptr) throw std::invalid_argument("search or out is NULL");
    *out = nullptr;
    const std::span<const t5d::mcts::EvalRequest> requests = search->search.requests();
    if (index >= requests.size()) {
      throw std::invalid_argument("leaf " + std::to_string(index) + " of " + std::to_string(requests.size()));
    }
    t5d_game* g = t5d::capi_internal::new_game(*requests[index].state);
    const t5d_status s = t5d_game_state(g, out);
    t5d_game_destroy(g);
    return s;
  });
}

t5d_status t5d_search_deliver(t5d_search* search, const float* logits, size_t logit_count,
                              const float* submit_logits, size_t submit_count,
                              const float* values, size_t value_count, int32_t value_kind,
                              char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr) throw std::invalid_argument("search is NULL");
    Search& s = search->search;
    const std::span<const t5d::mcts::EvalRequest> requests = s.requests();
    try {
      std::size_t want = 0;
      for (const t5d::mcts::EvalRequest& r : requests) want += r.expansion->moves().size();
      if (logit_count != want) {
        throw std::invalid_argument("deliver: " + std::to_string(logit_count) + " logits for " +
                                    std::to_string(want) + " candidate moves");
      }
      if (value_count != requests.size() * T5D_VALUE_BINS) {
        throw std::invalid_argument("deliver: " + std::to_string(value_count) + " values for " +
                                    std::to_string(requests.size()) + " leaves of 51 bins");
      }
      if (submit_count != 0 && submit_count != requests.size()) {
        throw std::invalid_argument("deliver: " + std::to_string(submit_count) +
                                    " submit logits for " + std::to_string(requests.size()) +
                                    " leaves (give one per leaf, or none)");
      }
      if ((logits == nullptr && logit_count > 0) || (values == nullptr && value_count > 0) ||
          (submit_logits == nullptr && submit_count > 0)) {
        throw std::invalid_argument("deliver: logits, submit_logits or values is NULL");
      }
      std::vector<t5d::mcts::Evaluation> evs(requests.size());
      std::size_t at = 0;
      for (std::size_t i = 0; i < requests.size(); ++i) {
        const std::size_t m = requests[i].expansion->moves().size();
        evs[i].logits.assign(logits + at, logits + at + m);
        at += m;
        if (submit_count > 0) evs[i].submit_logit = submit_logits[i];  // NaN: none
        evs[i].value = value_from(values + i * T5D_VALUE_BINS, value_kind);
      }
      s.deliver(evs);
    } catch (...) {
      s.abandon();
      throw;
    }
    return T5D_OK;
  });
}

t5d_status t5d_search_abandon(t5d_search* search) {
  if (search == nullptr) return T5D_ERR_ARGUMENT;
  search->search.abandon();
  return T5D_OK;
}

t5d_status t5d_search_run(t5d_search* search, int32_t evaluator, uint64_t evaluator_seed,
                          uint64_t playouts, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr) throw std::invalid_argument("search is NULL");
    const t5d::mcts::BuiltinEvaluator kind = builtin_of(evaluator);
    const std::unique_ptr<t5d::mcts::Evaluator> ev = t5d::mcts::make_evaluator(kind, evaluator_seed);
    search->search.run(*ev, t5d::mcts::Limits{playouts});
    return T5D_OK;
  });
}

t5d_status t5d_search_run_limits(t5d_search* search, int32_t evaluator, uint64_t evaluator_seed,
                                 const t5d_search_limits* limits, int32_t* stop, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr || limits == nullptr) throw std::invalid_argument("search or limits is NULL");
    const std::unique_ptr<t5d::mcts::Evaluator> ev = t5d::mcts::make_evaluator(builtin_of(evaluator), evaluator_seed);
    t5d::mcts::Limits l;
    l.playouts = limits->playouts;
    l.evaluations = limits->evaluations;
    l.nodes = limits->nodes;
    l.bytes = limits->bytes;
    l.seconds = limits->seconds;
    const t5d::mcts::StopReason r = search->search.run(*ev, l);
    if (stop != nullptr) *stop = static_cast<int32_t>(r);
    return T5D_OK;
  });
}

t5d_status t5d_search_info(const t5d_search* search, t5d_search_summary* out) {
  if (search == nullptr || out == nullptr) return T5D_ERR_ARGUMENT;
  return guarded(nullptr, [&]() -> t5d_status {
    const Search& s = search->search;
    const t5d::mcts::SearchStats& st = s.stats();
    const t5d::mcts::NodeView root = s.node(s.root());
    *out = t5d_search_summary{};
    out->playouts = st.playouts;
    out->evaluations = st.evaluations;
    out->cache_hits = st.cache_hits;
    out->nodes = s.node_count();
    out->nodes_created = st.nodes_created;
    out->transposition_hits = st.transposition_hits;
    out->checkmates = st.checkmates;
    out->stalemates = st.stalemates;
    out->turn_caps = st.turn_caps;
    out->unknown_verdicts = st.unknown_verdicts;
    out->unknown_edges = st.unknown_edges;
    out->verified_illegal = st.verified_illegal;
    out->root_turn_start = flag(s.root_turn_start());
    out->root_terminal = flag(s.root_terminal());
    out->root_solved = flag(s.root_solved());
    out->root_lo = root.lo;
    out->root_hi = root.hi;
    out->root_visits = root.visits;
    out->root_q = root.weight > 0 ? root.value / root.weight : 0.0;
    out->root_children = s.root_children().size();
    out->epoch = s.epoch();
    out->memory_bytes = s.memory_bytes();
    if (const t5d::mcts::EvalCache* cache = s.cache()) {
      out->cache_entries = cache->size();
      out->cache_bytes = cache->bytes();
    }
    out->cache_purged = st.cache_purged;
    out->last_stop = static_cast<int32_t>(st.last_stop);
    return T5D_OK;
  });
}

t5d_status t5d_search_child(const t5d_search* search, size_t index, t5d_root_child* out,
                            t5d_move* moves, size_t capacity, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr || out == nullptr) throw std::invalid_argument("search or out is NULL");
    if (moves == nullptr && capacity > 0) throw std::invalid_argument("moves is NULL");
    const Search& s = search->search;
    const std::vector<t5d::mcts::RootChild> children = s.root_children();
    if (index >= children.size()) {
      throw std::invalid_argument("root child " + std::to_string(index) + " of " + std::to_string(children.size()));
    }
    const t5d::mcts::RootChild& c = children[index];
    *out = t5d_root_child{};
    out->visits = c.visits;
    out->submit = flag(c.submit);
    out->legality = static_cast<int32_t>(c.legality);
    out->lo = c.lo;
    out->hi = c.hi;
    out->prior = c.prior;
    out->q = c.q;
    out->move_count = c.moves.size();
    const t5d::Multiverse& mv = s.root_state().multiverse();
    for (std::size_t i = 0; i < c.moves.size() && i < capacity; ++i) moves[i] = record_of(c.moves[i], mv);
    return T5D_OK;
  });
}

t5d_status t5d_search_choose(const t5d_search* search, size_t* index, int32_t* reason, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr || index == nullptr) throw std::invalid_argument("search or index is NULL");
    const t5d::mcts::Choice c = search->search.choose();
    *index = c.index;
    if (reason != nullptr) *reason = static_cast<int32_t>(c.reason);
    return T5D_OK;
  });
}

t5d_status t5d_search_choose_by_visits(const t5d_search* search, double temperature, double random,
                                       size_t* index, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr || index == nullptr) throw std::invalid_argument("search or index is NULL");
    *index = search->search.choose_by_visits(temperature, random);
    return T5D_OK;
  });
}

t5d_status t5d_search_verify(t5d_search* search, size_t index, int32_t* legality, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr || legality == nullptr) throw std::invalid_argument("search or legality is NULL");
    t5d::LegalityOptions exact = t5d::LegalityOptions::exact();
    exact.max_bytes = search->search.params().mate_bytes;
    *legality = static_cast<int32_t>(search->search.verify(index, exact));
    return T5D_OK;
  });
}

t5d_status t5d_search_advance(t5d_search* search, size_t index, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr) throw std::invalid_argument("search is NULL");
    search->search.advance(index);
    return T5D_OK;
  });
}

t5d_status t5d_search_advance_action(t5d_search* search, const t5d_move* moves, size_t count,
                                     char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr) throw std::invalid_argument("search is NULL");
    if (moves == nullptr && count > 0) throw std::invalid_argument("moves is NULL");
    // Each record matched in the state its predecessors reach.
    t5d::GameState s = search->search.root_state();
    std::vector<t5d::Move> action;
    for (std::size_t i = 0; i < count; ++i) {
      if (!s.may_move()) throw std::invalid_argument("advance_action: no move may be made now (T5, Q4)");
      const t5d::Move m = move_of(s, moves[i]);
      s.apply(m);
      action.push_back(m);
    }
    search->search.advance_action(action);
    return T5D_OK;
  });
}

t5d_status t5d_search_root_game(const t5d_search* search, t5d_game** out, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr || out == nullptr) throw std::invalid_argument("search or out is NULL");
    *out = nullptr;
    *out = t5d::capi_internal::new_game(search->search.root_state());
    return T5D_OK;
  });
}

t5d_status t5d_search_set_epoch(t5d_search* search, uint64_t epoch, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (search == nullptr) throw std::invalid_argument("search is NULL");
    search->search.set_epoch(epoch);
    return T5D_OK;
  });
}

t5d_status t5d_search_digest(const t5d_search* search, uint64_t* out) {
  if (search == nullptr || out == nullptr) return T5D_ERR_ARGUMENT;
  return guarded(nullptr, [&]() -> t5d_status {
    *out = search->search.digest();
    return T5D_OK;
  });
}

}  // extern "C"
