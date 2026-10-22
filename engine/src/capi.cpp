// The C API (t5d/capi.h): a thin layer over the C++ core that catches every
// exception at the boundary and reports it as a status and a message.

#include "t5d/capi.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "t5d/error.hpp"
#include "t5d/expand.hpp"
#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/pgn.hpp"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"
#include "t5d/variants.hpp"

#include "capi_internal.hpp"

#ifndef T5D_VERSION_STRING
#define T5D_VERSION_STRING "0.0.0"
#endif

struct t5d_state {
  t5d::Multiverse multiverse;
};

struct t5d_game {
  explicit t5d_game(t5d::GameState s, int32_t moves = 0) : state(std::move(s)), moves_in_action(moves) {}
  t5d_game(const t5d_game& other)
      : state(other.state),
        moves_in_action(other.moves_in_action),
        ruling(other.ruling),
        replay(other.replay),
        on_record(other.on_record) {}
  t5d_game& operator=(const t5d_game&) = delete;

  t5d::GameState state;
  // The moves made in the action in progress (register A6): no state records
  // it, so the handle counts - 0 at a turn start, +1 per move applied, 0 again
  // after a submission or a replayed action.
  int32_t moves_in_action = 0;
  // The ruling submissions are tested under: the engine's (C4, R1; stage
  // E4) - a submission that leaves a royal piece capturable is refused.
  t5d::CheckRuling ruling = t5d::standard_check_ruling();
  // A game made from a record: the record, replayed up to `state` while
  // on_record holds (t5d_game_apply and t5d_game_submit leave the record).
  std::optional<t5d::PgnReplay> replay;
  bool on_record = false;
  // The last expansion, and whether it is of the current state.
  t5d::Expansion expansion;
  bool expansion_current = false;
};

namespace {

// Raised inside the game functions for what the rules refuse, and for what
// the engine does not implement.
class IllegalAction : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
class NotImplemented : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

char* duplicate(std::string_view text) noexcept {
  auto* copy = static_cast<char*>(std::malloc(text.size() + 1));
  if (copy == nullptr) return nullptr;
  if (!text.empty()) std::memcpy(copy, text.data(), text.size());
  copy[text.size()] = '\0';
  return copy;
}

void set_error(char** error, std::string_view message) noexcept {
  if (error != nullptr) *error = duplicate(message);
}

// Runs `body`, turning any exception into a status (and a message).
template <class Body>
t5d_status guarded(char** error, Body&& body) noexcept {
  if (error != nullptr) *error = nullptr;
  try {
    return body();
  } catch (const t5d::NotationError& e) {
    set_error(error, e.what());
    return T5D_ERR_NOTATION;
  } catch (const t5d::RecordError& e) {
    set_error(error, e.what());
    switch (e.fault()) {
      case t5d::RecordFault::Unreadable: return T5D_ERR_NOTATION;
      case t5d::RecordFault::Unsupported: return T5D_ERR_UNSUPPORTED;
      case t5d::RecordFault::IllegalMove:
      case t5d::RecordFault::CannotSubmit: return T5D_ERR_ILLEGAL;
    }
    return T5D_ERR_INTERNAL;
  } catch (const IllegalAction& e) {
    set_error(error, e.what());
    return T5D_ERR_ILLEGAL;
  } catch (const NotImplemented& e) {
    set_error(error, e.what());
    return T5D_ERR_UNSUPPORTED;
  } catch (const std::bad_alloc&) {
    set_error(error, "out of memory");
    return T5D_ERR_MEMORY;
  } catch (const std::invalid_argument& e) {
    set_error(error, e.what());
    return T5D_ERR_ARGUMENT;
  } catch (const std::out_of_range& e) {
    set_error(error, e.what());
    return T5D_ERR_ARGUMENT;
  } catch (const std::exception& e) {
    set_error(error, e.what());
    return T5D_ERR_INTERNAL;
  } catch (...) {
    set_error(error, "unknown error");
    return T5D_ERR_INTERNAL;
  }
}

t5d_status new_state(t5d_state** out, t5d::Multiverse (*make)()) noexcept {
  if (out == nullptr) return T5D_ERR_ARGUMENT;
  *out = nullptr;
  return guarded(nullptr, [&] {
    *out = new t5d_state{make()};
    return T5D_OK;
  });
}

void copy_hex(char (&target)[33], const std::string& hex) noexcept {
  const std::size_t n = hex.size() < 32 ? hex.size() : 32;
  std::memcpy(target, hex.data(), n);
  target[n] = '\0';
}

// ---- games ------------------------------------------------------------------------

t5d::RuleOptions rules_from(std::uint64_t encoding) {
  const t5d::RuleOptions rules = t5d::RuleOptions::decode(encoding);
  rules.validate();
  return rules;
}

// A game from board strings, the player to move given or inferred. (No
// optional is copied while empty: g++ 13 and 14 warn, wrongly, that copying a
// disengaged optional may read its payload.)
t5d::GameState game_from_boards(std::span<const std::string_view> views, t5d::BoardSize size,
                                t5d::Parity parity, const t5d::RuleOptions& rules,
                                int32_t to_move) {
  switch (to_move) {
    case T5D_TO_MOVE_WHITE:
      return t5d::game_from_fen(views, size, parity, rules, t5d::Colour::White);
    case T5D_TO_MOVE_BLACK:
      return t5d::game_from_fen(views, size, parity, rules, t5d::Colour::Black);
    case T5D_TO_MOVE_INFER: return t5d::game_from_fen(views, size, parity, rules);
    default:
      throw std::invalid_argument("to_move must be T5D_TO_MOVE_WHITE, T5D_TO_MOVE_BLACK or "
                                  "T5D_TO_MOVE_INFER, not " + std::to_string(to_move));
  }
}

t5d::Parity parity_from(int32_t parity) {
  switch (parity) {
    case T5D_PARITY_ODD: return t5d::Parity::Odd;
    case T5D_PARITY_EVEN: return t5d::Parity::Even;
    case T5D_PARITY_INFER: return t5d::Parity::Infer;
    default:
      throw std::invalid_argument("parity must be T5D_PARITY_ODD, T5D_PARITY_EVEN or "
                                  "T5D_PARITY_INFER, not " + std::to_string(parity));
  }
}

// The board strings of a C array, the board size given or taken from the first.
std::vector<std::string_view> board_views(const char* const* boards, size_t count) {
  if (boards == nullptr && count > 0) throw std::invalid_argument("boards is NULL");
  std::vector<std::string_view> views;
  views.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (boards[i] == nullptr) {
      throw std::invalid_argument("board string #" + std::to_string(i + 1) + " is NULL");
    }
    views.emplace_back(boards[i]);
  }
  return views;
}

t5d::BoardSize size_from(int32_t width, int32_t height,
                         const std::vector<std::string_view>& views) {
  t5d::BoardSize size{width, height};
  if (width == 0 && height == 0) {
    if (views.empty()) throw t5d::NotationError("no board strings: a multiverse needs a board");
    return t5d::infer_fen_board_size(views.front());
  }
  if (!size.valid()) {
    throw std::invalid_argument("board size " + t5d::to_string(size) + " is outside 1x1..8x8");
  }
  return size;
}

t5d::ExpandOptions expand_options_from(const t5d_expand_options* o) {
  t5d::ExpandOptions options;
  if (o == nullptr) return options;
  if (o->window_k < 0) {
    throw std::invalid_argument("window_k must be 0 or more, not " + std::to_string(o->window_k));
  }
  options.window_k = o->window_k;
  switch (o->square_features) {
    case T5D_FEATURES_ALL: options.square_features = t5d::SquareFeatures::All; break;
    case T5D_FEATURES_SCOPE: options.square_features = t5d::SquareFeatures::Scope; break;
    case T5D_FEATURES_NONE: options.square_features = t5d::SquareFeatures::None; break;
    default:
      throw std::invalid_argument("square_features must be T5D_FEATURES_ALL, _SCOPE or _NONE, "
                                  "not " + std::to_string(o->square_features));
  }
  constexpr uint32_t kReserved = (1u << T5D_RELATION_PIN) | (1u << T5D_RELATION_CHECK) |
                                 (1u << T5D_RELATION_TRAVEL_THREAT);
  constexpr uint32_t kKnown = (1u << T5D_RELATION_MOVE) | T5D_RELATIONS_ATTACK |
                              T5D_RELATIONS_DEFENCE | T5D_RELATIONS_XRAY | kReserved;
  if ((o->relations & ~kKnown) != 0) {
    throw std::invalid_argument("relations: unknown bits " +
                                std::to_string(o->relations & ~kKnown));
  }
  if ((o->relations & kReserved) != 0) {
    throw NotImplemented(
        "relations: pin and check are not built (they would read check detection: rules "
        "reference R1, register H36), and travel-threat has no edge defined by "
        "the design spec (§6, B11; §11.1 uses travel threats only as a class of forcing "
        "moves) - none is built");
  }
  options.attack = (o->relations & T5D_RELATIONS_ATTACK) != 0;
  options.defence = (o->relations & T5D_RELATIONS_DEFENCE) != 0;
  options.xray = (o->relations & T5D_RELATIONS_XRAY) != 0;
  // Engine ABI round 2 (I15): the search leaves' legal-moves filter.
  if (o->legal_moves != 0 && o->legal_moves != 1) {
    throw std::invalid_argument("legal_moves must be 0 or 1, not " + std::to_string(o->legal_moves));
  }
  options.legal_moves = o->legal_moves == 1;
  if (options.legal_moves) options.legality = t5d::LegalityOptions::budgeted(o->legality_checks, o->legality_nodes);
  return options;
}

void describe(const t5d::Expansion& x, t5d_expansion& out) {
  out.width = x.board_size().width;
  out.height = x.board_size().height;
  out.even = x.even() ? 1 : 0;
  out.to_move = t5d::index_of(x.to_move());
  out.present = x.present();
  out.timelines = x.timeline_count();
  out.active_timelines = x.active_timeline_count();
  out.window_k = x.window_k();
  out.may_move = x.may_move() ? 1 : 0;
  out.origins_unknown = x.origins_unknown();
  out.board_count = x.board_count();
  out.move_count = x.move_records().size();
  out.edge_count = x.move_edge_count();
  out.relation_count = x.relation_count();
  out.window_count = x.window().size();
  out.e_count = x.e_boards().size();
  out.src_extra_count = x.src_extra().size();
  out.feature_board_count = x.feature_boards().size();
  out.origin_count = x.origin_count();
  out.timeline = x.board_timelines().data();
  out.half_turn = x.board_half_turns().data();
  out.is_head = x.board_is_head().data();
  out.active = x.board_active().data();
  out.side_to_move = x.board_side_to_move().data();
  out.playable = x.board_playable().data();
  out.window_depth = x.window_depth().data();
  out.band_depth = x.band_depth().data();
  out.is_origin = x.board_is_origin().data();
  out.parent = x.board_parent().data();
  out.feature_boards = x.feature_boards().data();
  out.piece = x.square_pieces().data();
  out.colour = x.square_colours().data();
  out.moved = x.square_moved().data();
  out.moves = x.move_records().data();
  out.edges = x.move_edges().data();
  out.relations = x.relations().data();
  out.window = x.window().data();
  out.e_boards = x.e_boards().data();
  out.src_extra = x.src_extra().data();
  out.acting_player = t5d::index_of(x.acting_player());
  out.submit_possible = x.submit_possible() ? 1 : 0;
  out.submit_legal = x.submit_legal() ? 1 : 0;
  out.mandatory_remaining = x.mandatory_remaining();
  out.optional_playable = x.optional_playable();
  out.moves_made = x.moves_made();
}

bool submittable(t5d::SubmitStatus s) noexcept {
  return s == t5d::SubmitStatus::Unchecked || s == t5d::SubmitStatus::Safe;
}

// Applies a move generated for the game's state: one more move in the action.
void apply_to(t5d_game& game, const t5d::Move& move) {
  game.state.apply(move);
  ++game.moves_in_action;
  game.on_record = false;
  game.expansion_current = false;
}

// t5d_layout's table: every public struct's size (field nullptr) and every
// member's offset, as this library was compiled.
struct LayoutEntry {
  const char* type;
  const char* field;
  size_t value;
};

#define T5D_SIZE(T) LayoutEntry{#T, nullptr, sizeof(T)}
#define T5D_AT(T, F) LayoutEntry{#T, #F, offsetof(T, F)}

constexpr LayoutEntry kLayout[] = {
    T5D_SIZE(t5d_fingerprint),
    T5D_AT(t5d_fingerprint, fingerprint),
    T5D_AT(t5d_fingerprint, placement),
    T5D_AT(t5d_fingerprint, boards),
    T5D_AT(t5d_fingerprint, pieces),

    T5D_SIZE(t5d_info),
    T5D_AT(t5d_info, boards),
    T5D_AT(t5d_info, pieces),
    T5D_AT(t5d_info, width),
    T5D_AT(t5d_info, height),
    T5D_AT(t5d_info, even),
    T5D_AT(t5d_info, to_move),
    T5D_AT(t5d_info, min_timeline),
    T5D_AT(t5d_info, max_timeline),
    T5D_AT(t5d_info, created_by_white),
    T5D_AT(t5d_info, created_by_black),
    T5D_AT(t5d_info, ruleset),

    T5D_SIZE(t5d_move),
    T5D_AT(t5d_move, from_timeline),
    T5D_AT(t5d_move, from_half_turn),
    T5D_AT(t5d_move, to_timeline),
    T5D_AT(t5d_move, to_half_turn),
    T5D_AT(t5d_move, from_board),
    T5D_AT(t5d_move, to_board),
    T5D_AT(t5d_move, from_square),
    T5D_AT(t5d_move, to_square),
    T5D_AT(t5d_move, from_cell),
    T5D_AT(t5d_move, to_cell),
    T5D_AT(t5d_move, piece),
    T5D_AT(t5d_move, captured),
    T5D_AT(t5d_move, promotion),
    T5D_AT(t5d_move, flags),

    T5D_SIZE(t5d_expand_options),
    T5D_AT(t5d_expand_options, window_k),
    T5D_AT(t5d_expand_options, square_features),
    T5D_AT(t5d_expand_options, relations),
    T5D_AT(t5d_expand_options, legal_moves),
    T5D_AT(t5d_expand_options, legality_checks),
    T5D_AT(t5d_expand_options, legality_nodes),

    T5D_SIZE(t5d_game_summary),
    T5D_AT(t5d_game_summary, boards),
    T5D_AT(t5d_game_summary, width),
    T5D_AT(t5d_game_summary, height),
    T5D_AT(t5d_game_summary, even),
    T5D_AT(t5d_game_summary, to_move),
    T5D_AT(t5d_game_summary, present),
    T5D_AT(t5d_game_summary, timelines),
    T5D_AT(t5d_game_summary, active_timelines),
    T5D_AT(t5d_game_summary, obligations_met),
    T5D_AT(t5d_game_summary, may_move),
    T5D_AT(t5d_game_summary, may_submit),
    T5D_AT(t5d_game_summary, ruleset),
    T5D_AT(t5d_game_summary, rules),
    T5D_AT(t5d_game_summary, moves_made),

    T5D_SIZE(t5d_expansion),
    T5D_AT(t5d_expansion, width),
    T5D_AT(t5d_expansion, height),
    T5D_AT(t5d_expansion, even),
    T5D_AT(t5d_expansion, to_move),
    T5D_AT(t5d_expansion, present),
    T5D_AT(t5d_expansion, timelines),
    T5D_AT(t5d_expansion, active_timelines),
    T5D_AT(t5d_expansion, window_k),
    T5D_AT(t5d_expansion, may_move),
    T5D_AT(t5d_expansion, origins_unknown),
    T5D_AT(t5d_expansion, board_count),
    T5D_AT(t5d_expansion, move_count),
    T5D_AT(t5d_expansion, edge_count),
    T5D_AT(t5d_expansion, relation_count),
    T5D_AT(t5d_expansion, window_count),
    T5D_AT(t5d_expansion, e_count),
    T5D_AT(t5d_expansion, src_extra_count),
    T5D_AT(t5d_expansion, feature_board_count),
    T5D_AT(t5d_expansion, origin_count),
    T5D_AT(t5d_expansion, timeline),
    T5D_AT(t5d_expansion, half_turn),
    T5D_AT(t5d_expansion, is_head),
    T5D_AT(t5d_expansion, active),
    T5D_AT(t5d_expansion, side_to_move),
    T5D_AT(t5d_expansion, playable),
    T5D_AT(t5d_expansion, window_depth),
    T5D_AT(t5d_expansion, band_depth),
    T5D_AT(t5d_expansion, is_origin),
    T5D_AT(t5d_expansion, parent),
    T5D_AT(t5d_expansion, feature_boards),
    T5D_AT(t5d_expansion, piece),
    T5D_AT(t5d_expansion, colour),
    T5D_AT(t5d_expansion, moved),
    T5D_AT(t5d_expansion, moves),
    T5D_AT(t5d_expansion, edges),
    T5D_AT(t5d_expansion, relations),
    T5D_AT(t5d_expansion, window),
    T5D_AT(t5d_expansion, e_boards),
    T5D_AT(t5d_expansion, src_extra),
    T5D_AT(t5d_expansion, acting_player),
    T5D_AT(t5d_expansion, submit_possible),
    T5D_AT(t5d_expansion, submit_legal),
    T5D_AT(t5d_expansion, mandatory_remaining),
    T5D_AT(t5d_expansion, optional_playable),
    T5D_AT(t5d_expansion, moves_made),
};

#undef T5D_AT
#undef T5D_SIZE

}  // namespace

extern "C" {

const char* t5d_version(void) { return T5D_VERSION_STRING; }

void t5d_free(void* pointer) { std::free(pointer); }

t5d_status t5d_layout(const char* type, const char* field, size_t* out) {
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

t5d_status t5d_state_new_standard(t5d_state** out) {
  return new_state(out, &t5d::standard_start);
}

t5d_status t5d_state_new_standard_two_timelines(t5d_state** out) {
  return new_state(out, &t5d::standard_two_timelines_start);
}

t5d_status t5d_state_new_from_fen(const char* const* boards, size_t count, int32_t width,
                                  int32_t height, int32_t parity, int32_t to_move,
                                  uint32_t ruleset, t5d_state** out, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (out == nullptr) throw std::invalid_argument("out is NULL");
    *out = nullptr;
    if (boards == nullptr && count > 0) throw std::invalid_argument("boards is NULL");
    std::vector<std::string_view> views;
    views.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (boards[i] == nullptr) {
        throw std::invalid_argument("board string #" + std::to_string(i + 1) + " is NULL");
      }
      views.emplace_back(boards[i]);
    }
    t5d::Parity p = t5d::Parity::Infer;
    switch (parity) {
      case T5D_PARITY_ODD: p = t5d::Parity::Odd; break;
      case T5D_PARITY_EVEN: p = t5d::Parity::Even; break;
      case T5D_PARITY_INFER: p = t5d::Parity::Infer; break;
      default:
        throw std::invalid_argument("parity must be T5D_PARITY_ODD, T5D_PARITY_EVEN or "
                                    "T5D_PARITY_INFER, not " + std::to_string(parity));
    }
    std::optional<t5d::Colour> player;
    switch (to_move) {
      case T5D_TO_MOVE_WHITE: player = t5d::Colour::White; break;
      case T5D_TO_MOVE_BLACK: player = t5d::Colour::Black; break;
      case T5D_TO_MOVE_INFER: break;
      default:
        throw std::invalid_argument("to_move must be T5D_TO_MOVE_WHITE, T5D_TO_MOVE_BLACK or "
                                    "T5D_TO_MOVE_INFER, not " + std::to_string(to_move));
    }
    t5d::BoardSize size{width, height};
    if (width == 0 && height == 0) {
      if (views.empty()) throw t5d::NotationError("no board strings: a multiverse needs a board");
      size = t5d::infer_fen_board_size(views.front());
    } else if (!size.valid()) {
      throw std::invalid_argument("board size " + t5d::to_string(size) +
                                  " is outside 1x1..8x8");
    }
    *out = new t5d_state{t5d::multiverse_from_fen(
        std::span<const std::string_view>(views), size, p, ruleset, player)};
    return T5D_OK;
  });
}

t5d_status t5d_state_copy(const t5d_state* state, t5d_state** out) {
  if (out == nullptr) return T5D_ERR_ARGUMENT;
  *out = nullptr;
  if (state == nullptr) return T5D_ERR_ARGUMENT;
  return guarded(nullptr, [&] {
    *out = new t5d_state{state->multiverse};
    return T5D_OK;
  });
}

void t5d_state_destroy(t5d_state* state) { delete state; }

t5d_status t5d_state_info(const t5d_state* state, t5d_info* out) {
  if (state == nullptr || out == nullptr) return T5D_ERR_ARGUMENT;
  const t5d::Multiverse& mv = state->multiverse;
  out->boards = mv.board_count();
  out->pieces = mv.piece_count();
  out->width = mv.board_size().width;
  out->height = mv.board_size().height;
  out->even = mv.even() ? 1 : 0;
  out->to_move = t5d::index_of(mv.to_move());
  out->min_timeline = mv.empty() ? 0 : mv.min_timeline();
  out->max_timeline = mv.empty() ? -1 : mv.max_timeline();
  out->created_by_white = mv.created_by(t5d::Colour::White);
  out->created_by_black = mv.created_by(t5d::Colour::Black);
  out->ruleset = mv.ruleset();
  return T5D_OK;
}

t5d_status t5d_state_fingerprint(const t5d_state* state, t5d_fingerprint* out) {
  if (state == nullptr || out == nullptr) return T5D_ERR_ARGUMENT;
  return guarded(nullptr, [&] {
    const t5d::Fingerprint f = t5d::fingerprint(state->multiverse);
    copy_hex(out->fingerprint, f.fingerprint);
    copy_hex(out->placement, f.placement);
    out->boards = f.boards;
    out->pieces = f.pieces;
    return T5D_OK;
  });
}

t5d_status t5d_state_canonical(const t5d_state* state, char** out) {
  if (out == nullptr) return T5D_ERR_ARGUMENT;
  *out = nullptr;
  if (state == nullptr) return T5D_ERR_ARGUMENT;
  return guarded(nullptr, [&] {
    *out = duplicate(t5d::canonical_state_string(state->multiverse));
    return *out != nullptr ? T5D_OK : T5D_ERR_MEMORY;
  });
}

t5d_status t5d_state_zobrist(const t5d_state* state, uint64_t* out) {
  if (state == nullptr || out == nullptr) return T5D_ERR_ARGUMENT;
  *out = state->multiverse.zobrist();
  return T5D_OK;
}

t5d_status t5d_state_board_count(const t5d_state* state, size_t* out) {
  if (state == nullptr || out == nullptr) return T5D_ERR_ARGUMENT;
  *out = state->multiverse.board_count();
  return T5D_OK;
}

t5d_status t5d_state_board(const t5d_state* state, size_t index, int32_t* timeline,
                           int32_t* half_turn, char** fen) {
  if (fen != nullptr) *fen = nullptr;
  if (state == nullptr) return T5D_ERR_ARGUMENT;
  const t5d::Multiverse& mv = state->multiverse;
  if (index >= mv.board_count()) return T5D_ERR_ARGUMENT;
  return guarded(nullptr, [&] {
    size_t remaining = index;
    for (const t5d::TimelineRef& t : mv.timelines()) {
      const auto length = static_cast<size_t>(t->length());
      if (remaining >= length) {
        remaining -= length;
        continue;
      }
      const int h = t->first_half_turn() + static_cast<int>(remaining);
      if (timeline != nullptr) *timeline = t->position();
      if (half_turn != nullptr) *half_turn = h;
      if (fen != nullptr) {
        *fen = duplicate(t5d::fen_board_string(*t->boards()[remaining], t->position(), h,
                                               mv.even()));
        if (*fen == nullptr) return T5D_ERR_MEMORY;
      }
      return T5D_OK;
    }
    return T5D_ERR_INTERNAL;  // unreachable: index < board_count()
  });
}

// ---- games ------------------------------------------------------------------------------

t5d_status t5d_game_new_from_pgn(const char* text, size_t length, uint64_t rules, t5d_game** out,
                                 char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (out == nullptr) throw std::invalid_argument("out is NULL");
    *out = nullptr;
    if (text == nullptr && length > 0) throw std::invalid_argument("text is NULL");
    const std::string_view record =
        length > 0 ? std::string_view(text, length) : std::string_view();
    t5d::PgnReplay replay(t5d::read_pgn(record), t5d::PgnOptions{}, rules_from(rules));
    auto game = std::make_unique<t5d_game>(replay.state());
    replay.set_check_ruling(game->ruling);  // a replay submits under the game's ruling
    game->replay.emplace(std::move(replay));
    game->on_record = true;
    *out = game.release();
    return T5D_OK;
  });
}

t5d_status t5d_game_new_from_fen(const char* const* boards, size_t count, int32_t width,
                                 int32_t height, int32_t parity, int32_t to_move, uint64_t rules,
                                 t5d_game** out, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (out == nullptr) throw std::invalid_argument("out is NULL");
    *out = nullptr;
    const std::vector<std::string_view> views = board_views(boards, count);
    const t5d::BoardSize size = size_from(width, height, views);
    *out = new t5d_game(game_from_boards(std::span<const std::string_view>(views), size,
                                         parity_from(parity), rules_from(rules), to_move));
    return T5D_OK;
  });
}

t5d_status t5d_game_new_variant(const char* name, uint64_t rules, t5d_game** out, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (out == nullptr) throw std::invalid_argument("out is NULL");
    *out = nullptr;
    if (name == nullptr) throw std::invalid_argument("name is NULL");
    const t5d::Variant* variant = t5d::find_variant(name);
    if (variant == nullptr) {
      throw NotImplemented(std::string("the variant \"") + name +
                           "\" is not one the engine sets up (engine README, \"The variants\")");
    }
    *out = new t5d_game(t5d::variant_start(*variant, rules_from(rules)));
    return T5D_OK;
  });
}

t5d_status t5d_game_copy(const t5d_game* game, t5d_game** out) {
  if (out == nullptr) return T5D_ERR_ARGUMENT;
  *out = nullptr;
  if (game == nullptr) return T5D_ERR_ARGUMENT;
  return guarded(nullptr, [&] {
    *out = new t5d_game(*game);
    return T5D_OK;
  });
}

void t5d_game_destroy(t5d_game* game) { delete game; }

t5d_status t5d_game_info(const t5d_game* game, t5d_game_summary* out) {
  if (game == nullptr || out == nullptr) return T5D_ERR_ARGUMENT;
  const t5d::GameState& s = game->state;
  const t5d::Multiverse& mv = s.multiverse();
  out->boards = mv.board_count();
  out->width = mv.board_size().width;
  out->height = mv.board_size().height;
  out->even = mv.even() ? 1 : 0;
  out->to_move = t5d::index_of(s.to_move());
  out->present = s.present();
  out->timelines = mv.timeline_count();
  out->active_timelines = mv.timeline_count() - static_cast<int32_t>(s.inactive_timelines().size());
  out->obligations_met = s.obligations_met() ? 1 : 0;
  out->may_move = s.may_move() ? 1 : 0;
  out->may_submit = submittable(s.submission_status(game->ruling)) ? 1 : 0;
  out->ruleset = mv.ruleset();
  out->rules = s.rules().encode();
  out->moves_made = game->moves_in_action;
  return T5D_OK;
}

t5d_status t5d_game_record_actions(const t5d_game* game, size_t* total, size_t* played,
                                   int32_t* truncated) {
  if (game == nullptr) return T5D_ERR_ARGUMENT;
  const bool record = game->replay.has_value();
  if (total != nullptr) *total = record ? game->replay->record().actions.size() : 0;
  if (played != nullptr) *played = record ? game->replay->actions_played() : 0;
  if (truncated != nullptr) {
    *truncated = record && game->replay->record().unreadable.has_value() ? 1 : 0;
  }
  return T5D_OK;
}

t5d_status t5d_game_replay_next(t5d_game* game, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (game == nullptr) throw std::invalid_argument("game is NULL");
    if (!game->replay) throw std::invalid_argument("the game was not made from a record");
    if (!game->on_record) {
      throw std::invalid_argument("the game has left its record (t5d_game_apply or "
                                  "t5d_game_submit): it cannot be replayed further");
    }
    if (game->replay->finished()) {
      if (game->replay->record().unreadable) {
        throw t5d::RecordError(t5d::RecordFault::Unreadable, *game->replay->record().unreadable);
      }
      throw std::invalid_argument("every action of the record has been replayed");
    }
    game->replay->play_next();  // the strong guarantee: unchanged on failure
    game->state = game->replay->state();
    game->moves_in_action = 0;  // a whole action, submitted: the next turn start
    game->expansion_current = false;
    return T5D_OK;
  });
}

t5d_status t5d_game_apply(t5d_game* game, const t5d_move* move, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (game == nullptr || move == nullptr) throw std::invalid_argument("game or move is NULL");
    if (!game->state.may_move()) {
      throw IllegalAction("no move may be made now: the action's obligations are met (the "
                          "present has shifted to the opponent), so it must be submitted (T5) - under "
                          "Q4's Guide-literal reading (moves_after_shift = false)");
    }
    t5d::MoveList moves;
    t5d::append_piece_moves(game->state, move->from_timeline, move->from_square, moves);
    for (const t5d::Move& m : moves) {
      const int promotion = m.promotes() ? t5d::index_of(m.promotion) + 1 : 0;
      if (m.from_half_turn == move->from_half_turn && m.to_timeline == move->to_timeline &&
          m.to_half_turn == move->to_half_turn && m.to_square == move->to_square &&
          promotion == move->promotion) {
        apply_to(*game, m);
        return T5D_OK;
      }
    }
    throw IllegalAction("no pseudo-legal move of the player to move goes from (" +
                        std::to_string(move->from_timeline) + ", " +
                        std::to_string(move->from_half_turn) + ", " +
                        std::to_string(move->from_square) + ") to (" +
                        std::to_string(move->to_timeline) + ", " +
                        std::to_string(move->to_half_turn) + ", " +
                        std::to_string(move->to_square) + ") with promotion " +
                        std::to_string(move->promotion));
  });
}

t5d_status t5d_game_apply_expanded(t5d_game* game, size_t index, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (game == nullptr) throw std::invalid_argument("game is NULL");
    if (!game->expansion_current) {
      throw std::invalid_argument("the handle's last expansion is not of its current state: "
                                  "call t5d_game_expand first");
    }
    const t5d::MoveList& moves = game->expansion.moves();
    if (index >= moves.size()) {
      throw std::invalid_argument("move index " + std::to_string(index) + " of " +
                                  std::to_string(moves.size()));
    }
    const t5d::Move m = moves[index];  // the expansion's buffers are reused by the next one
    apply_to(*game, m);
    return T5D_OK;
  });
}

t5d_status t5d_game_submit(t5d_game* game, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (game == nullptr) throw std::invalid_argument("game is NULL");
    const t5d::SubmitStatus status = game->state.submission_status(game->ruling);
    if (status == t5d::SubmitStatus::ObligationsUnmet) {
      throw IllegalAction("cannot submit: the present has not shifted to the opponent (T5)");
    }
    if (!submittable(status)) {
      throw IllegalAction("cannot submit: a royal piece of the mover's is left capturable (C4, R1)");
    }
    game->state.submit(game->ruling);
    game->moves_in_action = 0;  // the opponent's turn start
    game->on_record = false;
    game->expansion_current = false;
    return T5D_OK;
  });
}

t5d_status t5d_game_set_legality(t5d_game* game, int32_t legality, char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (game == nullptr) throw std::invalid_argument("game is NULL");
    if (legality != T5D_LEGAL && legality != T5D_PSEUDO_LEGAL) {
      throw std::invalid_argument("legality must be T5D_LEGAL or T5D_PSEUDO_LEGAL");
    }
    game->ruling =
        legality == T5D_LEGAL ? t5d::standard_check_ruling() : t5d::CheckRuling::none();
    if (game->replay) game->replay->set_check_ruling(game->ruling);
    return T5D_OK;
  });
}

t5d_status t5d_game_expand(t5d_game* game, const t5d_expand_options* options, t5d_expansion* out,
                           char** error) {
  return guarded(error, [&]() -> t5d_status {
    if (game == nullptr || out == nullptr) throw std::invalid_argument("game or out is NULL");
    const t5d::ExpandOptions o = expand_options_from(options);
    game->expansion_current = false;
    t5d::expand(game->state, o, game->expansion, game->moves_in_action);
    game->expansion_current = true;
    describe(game->expansion, *out);
    return T5D_OK;
  });
}

t5d_status t5d_game_state(const t5d_game* game, t5d_state** out) {
  if (out == nullptr) return T5D_ERR_ARGUMENT;
  *out = nullptr;
  if (game == nullptr) return T5D_ERR_ARGUMENT;
  return guarded(nullptr, [&] {
    *out = new t5d_state{game->state.multiverse()};
    return T5D_OK;
  });
}

}  // extern "C"

// ---- shared with src/capi_search.cpp (src/capi_internal.hpp) -----------------------

namespace t5d::capi_internal {

const GameState& game_state(const t5d_game& game) noexcept { return game.state; }
int game_moves_in_action(const t5d_game& game) noexcept { return game.moves_in_action; }
t5d_game* new_game(const GameState& state, int moves_in_action) {
  if (moves_in_action < 0) throw std::invalid_argument("a game's moves in its action must be 0 or more");
  return new t5d_game(state, moves_in_action);
}
void describe_expansion(const Expansion& expansion, t5d_expansion& out) { ::describe(expansion, out); }
ExpandOptions expand_options(const t5d_expand_options* options) {
  try {
    return ::expand_options_from(options);
  } catch (const ::NotImplemented& e) {
    throw Unsupported(e.what());
  }
}

}  // namespace t5d::capi_internal
