// The C API (t5d/capi.h): a thin layer over the C++ core that catches every
// exception at the boundary and reports it as a status and a message.

#include "t5d/capi.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "t5d/error.hpp"
#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/multiverse.hpp"

#ifndef T5D_VERSION_STRING
#define T5D_VERSION_STRING "0.0.0"
#endif

struct t5d_state {
  t5d::Multiverse multiverse;
};

namespace {

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

}  // namespace

extern "C" {

const char* t5d_version(void) { return T5D_VERSION_STRING; }

void t5d_free(void* pointer) { std::free(pointer); }

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

}  // extern "C"
