// Expansion (see t5d/expand.hpp). Rule IDs refer to the rules reference,
// docs/rules/README.md; sections (§) to the design spec,
// docs/design/5d-chess-transformer-design.md.

#include "t5d/expand.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

#include "t5d/board.hpp"
#include "t5d/check.hpp"
#include "t5d/multiverse.hpp"
#include "t5d/search.hpp"

namespace t5d {

namespace {

// ---- per-square features ---------------------------------------------------------------
//
// From a piece's code (t5d/types.hpp: 0 for the empty square, otherwise
// 1 + (kind << 2 | colour << 1 | unmoved)) by byte arithmetic the compiler can
// vectorise. The compile-time check below holds it to the Piece API for every
// code, so a change of encoding fails the build rather than the features.

constexpr int kPawn = index_of(PieceKind::Pawn);
constexpr int kRook = index_of(PieceKind::Rook);
constexpr int kKing = index_of(PieceKind::King);
constexpr int kBrawn = index_of(PieceKind::Brawn);

struct Features {
  std::uint8_t piece;
  std::uint8_t colour;
  std::uint8_t moved;
};

constexpr Features features_of_code(std::uint8_t code) noexcept {
  const auto v = static_cast<std::uint8_t>(code - 1u);  // wraps for the empty square
  const int kind = v >> 2;
  const bool occupied = code != 0;
  // Only these four kinds carry the unmoved flag (t5d/types.hpp, tracks_unmoved).
  const bool tracks = kind == kPawn || kind == kRook || kind == kKing || kind == kBrawn;
  return Features{
      static_cast<std::uint8_t>(occupied ? kind + 1 : 0),
      static_cast<std::uint8_t>(occupied ? ((v >> 1) & 1) + 1 : 0),
      static_cast<std::uint8_t>(occupied && tracks && (v & 1) == 0 ? 1 : 0)};
}

constexpr bool features_agree_with_pieces() noexcept {
  for (int code = 0; code <= Piece::kMaxCode; ++code) {
    const Piece p = Piece::from_code(code);
    const Features f = features_of_code(static_cast<std::uint8_t>(code));
    if (p.empty()) {
      if (f.piece != 0 || f.colour != 0 || f.moved != 0) return false;
      continue;
    }
    if (f.piece != index_of(p.kind()) + 1) return false;
    if (f.colour != index_of(p.colour()) + 1) return false;
    if (f.moved != (tracks_unmoved(p.kind()) && !p.unmoved() ? 1 : 0)) return false;
  }
  return true;
}
static_assert(features_agree_with_pieces(),
              "the feature decoding must agree with t5d::Piece for every code");

inline void decode_codes(const std::uint8_t* codes, int n, std::uint8_t* piece,
                         std::uint8_t* colour, std::uint8_t* moved) noexcept {
  for (int i = 0; i < n; ++i) {
    const Features f = features_of_code(codes[i]);
    piece[i] = f.piece;
    colour[i] = f.colour;
    moved[i] = f.moved;
  }
}

#if defined(__GNUC__)
// Sixteen cells at a time by the same arithmetic on byte vectors (the GCC and
// Clang vector extension: SSE2 or NEON code without intrinsics). A comparison
// gives 0 or all ones per byte.
typedef std::uint8_t U8x16 __attribute__((vector_size(16)));

inline U8x16 splat(int value) noexcept {
  U8x16 v;
  for (int i = 0; i < 16; ++i) v[i] = static_cast<std::uint8_t>(value);
  return v;
}

inline void decode16(const std::uint8_t* codes, std::uint8_t* piece, std::uint8_t* colour,
                     std::uint8_t* moved) noexcept {
  U8x16 c;
  std::memcpy(&c, codes, 16);
  const U8x16 one = splat(1);
  const U8x16 v = c - one;  // wraps for the empty square, masked below
  const U8x16 kind = v >> 2;
  const auto occupied = (U8x16)(c != splat(0));
  const auto tracks = (U8x16)((kind == splat(kPawn)) | (kind == splat(kRook)) |
                              (kind == splat(kKing)) | (kind == splat(kBrawn)));
  const auto unmoved = (U8x16)((v & one) != splat(0));
  const U8x16 p = (kind + one) & occupied;
  const U8x16 col = (((v >> 1) & one) + one) & occupied;
  const U8x16 m = tracks & occupied & ~unmoved & one;
  std::memcpy(piece, &p, 16);
  std::memcpy(colour, &col, 16);
  std::memcpy(moved, &m, 16);
}
#endif

// One board's cells, rank by rank (cell = rank * width + file).
void decode_board(const Board& board, BoardSize size, std::uint8_t* piece, std::uint8_t* colour,
                  std::uint8_t* moved) noexcept {
  const std::uint8_t* codes = board.codes();
  if (size.width == 8 && size.height == 8) {
#if defined(__GNUC__)
    for (int i = 0; i < kSquareCount; i += 16) {
      decode16(codes + i, piece + i, colour + i, moved + i);
    }
#else
    decode_codes(codes, kSquareCount, piece, colour, moved);
#endif
    return;
  }
  std::uint8_t cells[kSquareCount];
  int n = 0;
  for (int rank = 0; rank < size.height; ++rank) {
    for (int file = 0; file < size.width; ++file) cells[n++] = codes[make_square(file, rank)];
  }
  decode_codes(cells, n, piece, colour, moved);
}

inline std::int8_t cell_of(int square, int width) noexcept {
  return static_cast<std::int8_t>(rank_of(square) * width + file_of(square));
}

constexpr std::uint8_t type_id(Piece p) noexcept {
  return p.empty() ? std::uint8_t{0} : static_cast<std::uint8_t>(index_of(p.kind()) + 1);
}

// A k no window depth reaches: kNoWindow marks "in W(k) for no k", so a k that
// large (INT32_MAX) is taken as the largest k that leaves it out.
constexpr int window_k_of(int k) noexcept { return k < kNoWindow ? k : kNoWindow - 1; }

// THE MOVES AN EXPANSION HOLDS - the one place they are chosen: pseudo-legal
// (t5d/movegen.hpp), or with options.legal_moves those not proved illegal -
// the legal ones, and under a budget the undecided ones too (t5d/legal.hpp;
// C4, R1; build-log spec issue 28), with their status. Everything below - the
// move graph, attacks, E(k), src_extra(k) and the playable boards' roles - is
// derived from this list. The geometric relations (defence, x-ray) come from
// the same pass; they depend on no rule about royal safety, so the filter
// leaves them whole.
void expansion_moves(const GameState& state, const ExpandOptions& options, MoveList& moves,
                     RelationList& geometric, std::vector<MoveLegality>& legality,
                     LegalityStats& stats) {
  if (options.defence || options.xray) {
    generate_moves(state, moves, geometric, RelationKinds{options.defence, options.xray});
  } else {
    generate_moves(state, moves);
    geometric.clear();
  }
  legality.clear();
  stats = LegalityStats{};
  if (!options.legal_moves) return;
  std::vector<MoveLegality> status;
  classify_moves(state, moves, status, options.legality, &stats);
  std::size_t kept = 0;
  for (std::size_t i = 0; i < moves.size(); ++i) {
    if (status[i] == MoveLegality::Illegal) continue;
    moves[kept++] = moves[i];
    legality.push_back(status[i]);
  }
  moves.resize(kept);
}

}  // namespace

// ---- Expansion -----------------------------------------------------------------------------

int Expansion::board_index(int timeline, int half_turn) const noexcept {
  if (tl_position_.empty()) return -1;
  const long long i = static_cast<long long>(timeline) - tl_position_.front();
  if (i < 0 || i >= static_cast<long long>(tl_first_board_.size())) return -1;
  const auto t = static_cast<std::size_t>(i);
  const int offset = half_turn - tl_first_half_turn_[t];
  if (offset < 0 || offset >= tl_length_[t]) return -1;
  return tl_first_board_[t] + offset;
}

std::size_t Expansion::timeline_of_board(int board) const noexcept {
  // The last timeline whose first board is at or before `board`.
  const auto it = std::upper_bound(tl_first_board_.begin(), tl_first_board_.end(), board);
  return static_cast<std::size_t>(it - tl_first_board_.begin()) - 1;
}

std::int32_t Expansion::depth_of(int board, Window which) const noexcept {
  if (which == Window::WithOrigins && bit(origin_bits_, board)) return 0;
  const std::size_t t = timeline_of_board(board);
  if (tl_active_[t] == 0) return kNoWindow;
  return tl_first_board_[t] + tl_length_[t] - 1 - board;  // the distance from the head
}

BoardInfo Expansion::board_info(int board) const noexcept {
  const std::size_t t = timeline_of_board(board);
  const int j = board - tl_first_board_[t];
  const int length = tl_length_[t];
  BoardInfo b;
  b.timeline = tl_position_[t];
  b.half_turn = tl_first_half_turn_[t] + j;
  b.head = j == length - 1;
  b.active = tl_active_[t] != 0;
  b.side_to_move = static_cast<std::uint8_t>(b.half_turn & 1);
  // L5: the head is playable; the mover's iff the mover is to move on it.
  b.playable = b.head && may_move_ && to_move_at(b.half_turn) == to_move_;
  b.origin = bit(origin_bits_, board);
  b.band_depth = b.active ? length - 1 - j : kNoWindow;
  b.window_depth = b.origin ? 0 : b.band_depth;
  // The lineage: the timeline's previous board, or the first board's branch
  // origin.
  b.parent = j > 0 ? board - 1 : tl_origin_[t];
  return b;
}

void Expansion::materialize() const {
  if (boards_ready_) return;
  const std::size_t n = board_count_;
  timeline_of_.resize(n);
  half_turn_of_.resize(n);
  is_head_.resize(n);
  active_.resize(n);
  side_.resize(n);
  playable_.resize(n);
  depth_.resize(n);
  band_depth_.resize(n);
  origin_.resize(n);
  parent_.resize(n);
  for (std::size_t t = 0; t < tl_first_board_.size(); ++t) {
    const auto first = static_cast<std::size_t>(tl_first_board_[t]);
    const auto len = static_cast<std::size_t>(tl_length_[t]);
    const std::uint8_t is_active = tl_active_[t];
    std::fill_n(timeline_of_.begin() + static_cast<std::ptrdiff_t>(first), len, tl_position_[t]);
    std::fill_n(is_head_.begin() + static_cast<std::ptrdiff_t>(first), len, std::uint8_t{0});
    std::fill_n(active_.begin() + static_cast<std::ptrdiff_t>(first), len, is_active);
    std::fill_n(playable_.begin() + static_cast<std::ptrdiff_t>(first), len, std::uint8_t{0});
    for (std::size_t j = 0; j < len; ++j) {
      const std::size_t b = first + j;
      const int h = tl_first_half_turn_[t] + static_cast<int>(j);
      half_turn_of_[b] = h;
      side_[b] = static_cast<std::uint8_t>(h & 1);
      band_depth_[b] = is_active != 0 ? static_cast<std::int32_t>(len - 1 - j) : kNoWindow;
      const bool origin = bit(origin_bits_, static_cast<int>(b));
      origin_[b] = origin ? 1 : 0;
      depth_[b] = origin ? 0 : band_depth_[b];
      parent_[b] = static_cast<std::int32_t>(b) - 1;
    }
    parent_[first] = tl_origin_[t];
    const std::size_t head = first + len - 1;
    is_head_[head] = 1;
    if (may_move_ && to_move_at(half_turn_of_[head]) == to_move_) playable_[head] = 1;
  }
  boards_ready_ = true;
}

std::size_t Expansion::window_size(int k, Window which) const noexcept {
  k = window_k_of(k);
  std::size_t count = 0;
  for (std::size_t t = 0; t < tl_length_.size(); ++t) {
    if (tl_active_[t] != 0) count += static_cast<std::size_t>(std::min<long long>(tl_length_[t], 1LL + k));
  }
  if (which == Window::WithOrigins) {
    for (const std::int32_t o : origins_) count += depth_of(o, Window::BandsOnly) > k ? 1u : 0u;
  }
  return count;
}

std::size_t Expansion::e_count(int k, Window which) const noexcept {
  k = window_k_of(k);
  std::size_t count = 0;
  for (const std::int32_t b : destinations_) count += depth_of(b, which) > k ? 1u : 0u;
  return count;
}

std::size_t Expansion::src_extra_count(int k, Window which) const noexcept {
  k = window_k_of(k);
  std::size_t count = 0;
  for (const std::int32_t b : sources_) count += depth_of(b, which) > k ? 1u : 0u;
  return count;
}

std::size_t Expansion::src_only_count(int k, Window which) const noexcept {
  k = window_k_of(k);
  std::size_t count = 0;
  for (const std::int32_t b : sources_) {
    count += depth_of(b, which) > k && !bit(destination_bits_, b) ? 1u : 0u;
  }
  return count;
}

void Expansion::window(int k, std::vector<std::int32_t>& out, Window which) const {
  k = window_k_of(k);
  out.clear();
  // Timeline by timeline, ascending: the origins on it before its band, then
  // the band (an inactive timeline: its origins only).
  std::size_t next_origin = 0;
  for (std::size_t t = 0; t < tl_length_.size(); ++t) {
    const std::int32_t first = tl_first_board_[t];
    const std::int32_t end = first + tl_length_[t];
    const std::int32_t band_start =
        tl_active_[t] != 0
            ? static_cast<std::int32_t>(std::max<long long>(first, static_cast<long long>(end) - 1 - k))
            : end;
    if (which == Window::WithOrigins) {
      while (next_origin < origins_.size() && origins_[next_origin] < end) {
        if (origins_[next_origin] < band_start) out.push_back(origins_[next_origin]);
        ++next_origin;
      }
    }
    for (std::int32_t b = band_start; b < end; ++b) out.push_back(b);
  }
}

void Expansion::e_boards(int k, std::vector<std::int32_t>& out, Window which) const {
  k = window_k_of(k);
  out.clear();
  for (const std::int32_t b : destinations_) {
    if (depth_of(b, which) > k) out.push_back(b);
  }
}

void Expansion::src_extra(int k, std::vector<std::int32_t>& out, Window which) const {
  k = window_k_of(k);
  out.clear();
  for (const std::int32_t b : sources_) {
    if (depth_of(b, which) > k) out.push_back(b);
  }
}

// ---- expand ----------------------------------------------------------------------------------

namespace {

// The set bits of `bits`, ascending, into `out`.
void list_bits(const std::vector<std::uint64_t>& bits, std::vector<std::int32_t>& out) {
  out.clear();
  for (std::size_t w = 0; w < bits.size(); ++w) {
    for (std::uint64_t word = bits[w]; word != 0; word &= word - 1) {
      out.push_back(static_cast<std::int32_t>(64 * w + static_cast<std::size_t>(std::countr_zero(word))));
    }
  }
}

inline void set_bit(std::vector<std::uint64_t>& bits, std::int32_t board) noexcept {
  const auto b = static_cast<std::size_t>(board);
  bits[b >> 6] |= std::uint64_t{1} << (b & 63);
}

}  // namespace

void expand(const GameState& state, const ExpandOptions& options, Expansion& out) {
  if (options.window_k < 0) {
    throw std::invalid_argument("expand: window_k must be 0 or more, not " +
                                std::to_string(options.window_k));
  }
  if (options.pin || options.check) {
    throw std::invalid_argument(
        "expand: the pin and check relations are not built (they would read check "
        "detection, t5d/check.hpp: rules reference R1, register H36)");
  }
  if (options.travel_threat) {
    throw std::invalid_argument(
        "expand: the travel-threat relation is not built - the design spec names it (§6, "
        "B11) but defines no edge for it (§11.1 uses travel threats only as a class of "
        "forcing moves)");
  }

  const Multiverse& mv = state.multiverse();
  const BoardSize size = mv.board_size();
  const int width = size.width;
  const int k = window_k_of(options.window_k);
  const std::size_t n = mv.board_count();

  out.size_ = size;
  out.even_ = mv.even();
  out.to_move_ = state.to_move();
  out.present_ = state.present();
  out.may_move_ = state.may_move();
  // D8: the submit action - possible once the obligations are met (T5), legal
  // when C4 holds too (the submission test itself; it allocates nothing).
  out.submit_possible_ = state.obligations_met();
  out.submit_legal_ = out.submit_possible_ && royals_safe(state);
  out.window_k_ = options.window_k;
  out.timeline_count_ = mv.timeline_count();
  out.board_count_ = n;
  out.boards_ready_ = false;

  // -- per timeline (f): O(timelines), not O(boards)
  const auto timelines = static_cast<std::size_t>(mv.timeline_count());
  out.tl_position_.resize(timelines);
  out.tl_first_board_.resize(timelines);
  out.tl_first_half_turn_.resize(timelines);
  out.tl_length_.resize(timelines);
  out.tl_active_.resize(timelines);
  out.tl_origin_.resize(timelines);
  int active_count = 0;
  std::int32_t b = 0;
  for (std::size_t i = 0; i < timelines; ++i) {
    const Timeline& t = *mv.timelines()[i];
    const bool is_active = state.active(t.position());
    out.tl_position_[i] = t.position();
    out.tl_first_board_[i] = b;
    out.tl_first_half_turn_[i] = t.first_half_turn();
    out.tl_length_[i] = t.length();
    out.tl_active_[i] = is_active ? 1 : 0;
    active_count += is_active ? 1 : 0;
    b += t.length();
  }
  out.active_count_ = active_count;

  // -- the board bitsets: N/64 words each
  const std::size_t words = (n + 63) / 64;
  out.origin_bits_.assign(words, 0);
  out.destination_bits_.assign(words, 0);
  out.source_bits_.assign(words, 0);

  // -- branch origins: an active timeline's origin is in W(k) at every k
  //    (§10.4.3); every timeline's is its first board's parent
  int origins_unknown = 0;
  out.origins_.clear();
  for (std::size_t i = 0; i < timelines; ++i) {
    const Timeline& t = *mv.timelines()[i];
    out.tl_origin_[i] = -1;
    if (t.origin().has_value()) {
      const BoardCoord origin = *t.origin();
      const int o = out.board_index(origin.timeline, origin.half_turn);
      if (o < 0) continue;
      out.tl_origin_[i] = o;
      if (out.tl_active_[i] != 0 && !Expansion::bit(out.origin_bits_, o)) {
        set_bit(out.origin_bits_, o);
        out.origins_.push_back(o);
      }
    } else if (out.tl_active_[i] != 0 && t.creator().has_value()) {
      ++origins_unknown;  // created by branching, origin not recorded (5DFEN)
    }
  }
  std::sort(out.origins_.begin(), out.origins_.end());
  out.origins_unknown_ = origins_unknown;

  // -- the moves (a), and the geometric relations (e) in the same pass
  expansion_moves(state, options, out.moves_, out.geometric_, out.legality_, out.legality_stats_);

  // -- records (a), the move graph (b), attacks (e), the boards' roles (d),
  //    written in place into buffers that only ever grow
  const std::size_t moves = out.moves_.size();
  if (out.records_.size() < moves) out.records_.resize(moves);
  if (out.edges_.size() < 4 * moves) out.edges_.resize(4 * moves);
  const std::size_t relations_max =
      5 * ((options.attack ? moves : 0) + out.geometric_.size());
  if (out.relations_.size() < relations_max) out.relations_.resize(relations_max);
  MoveRecord* const records = out.records_.data();
  std::int32_t* const edges = out.edges_.data();
  std::int32_t* const rel = out.relations_.data();
  std::size_t edge_ints = 0;
  std::size_t rel_ints = 0;
  const std::int32_t* const first_board = out.tl_first_board_.data();
  const std::int32_t* const first_half = out.tl_first_half_turn_.data();
  const int min_p = timelines > 0 ? out.tl_position_.front() : 0;
  // Every move's boards exist: no range checks here (board_index has them).
  const auto index_of_board = [&](int p, int h) {
    const auto t = static_cast<std::size_t>(p - min_p);
    return first_board[t] + (h - first_half[t]);
  };
  for (std::size_t i = 0; i < moves; ++i) {
    const Move& m = out.moves_[i];
    MoveRecord& r = records[i];
    const std::int32_t from = index_of_board(m.from_timeline, m.from_half_turn);
    const std::int32_t to = index_of_board(m.to_timeline, m.to_half_turn);
    const std::int8_t from_cell = cell_of(m.from_square, width);
    const std::int8_t to_cell = cell_of(m.to_square, width);
    r.from_timeline = m.from_timeline;
    r.from_half_turn = m.from_half_turn;
    r.to_timeline = m.to_timeline;
    r.to_half_turn = m.to_half_turn;
    r.from_board = from;
    r.to_board = to;
    r.from_square = m.from_square;
    r.to_square = m.to_square;
    r.from_cell = from_cell;
    r.to_cell = to_cell;
    r.piece = type_id(m.piece);
    r.captured = type_id(m.captured);
    r.promotion = static_cast<std::uint8_t>(m.promotes() ? index_of(m.promotion) + 1 : 0);
    r.flags = static_cast<std::uint8_t>(m.flags | (m.physical() ? 0u : T5D_MOVE_TRAVEL));

    // (b): promotions are generated one after another, so a pair repeats only
    // right after itself.
    if (edge_ints == 0 || edges[edge_ints - 4] != from || edges[edge_ints - 3] != from_cell ||
        edges[edge_ints - 2] != to || edges[edge_ints - 1] != to_cell) {
      edges[edge_ints] = from;
      edges[edge_ints + 1] = from_cell;
      edges[edge_ints + 2] = to;
      edges[edge_ints + 3] = to_cell;
      edge_ints += 4;
    }

    // (e) attack: the square of the piece taken (en passant: the pawn's) -
    // once per pair, promotions merged as in (b).
    if (options.attack && m.capture()) {
      const std::int32_t target = m.en_passant() ? cell_of(m.aux_square, width) : to_cell;
      if (rel_ints == 0 || rel[rel_ints - 5] != from || rel[rel_ints - 4] != from_cell ||
          rel[rel_ints - 3] != to || rel[rel_ints - 2] != target) {
        rel[rel_ints] = from;
        rel[rel_ints + 1] = from_cell;
        rel[rel_ints + 2] = to;
        rel[rel_ints + 3] = target;
        rel[rel_ints + 4] = kRelationAttack;
        rel_ints += 5;
      }
    }

    // (d): every source; a destination only of travel (§10.4.1's "travel
    // destinations": a move within a board lands on its own source board).
    set_bit(out.source_bits_, from);
    if (to != from) set_bit(out.destination_bits_, to);
  }
  out.edge_ints_ = edge_ints;

  // (e) defence and x-ray, from the generator's geometry.
  for (const GeometricRelation& g : out.geometric_) {
    rel[rel_ints] = index_of_board(g.from_timeline, g.from_half_turn);
    rel[rel_ints + 1] = cell_of(g.from_square, width);
    rel[rel_ints + 2] = index_of_board(g.to_timeline, g.to_half_turn);
    rel[rel_ints + 3] = cell_of(g.to_square, width);
    rel[rel_ints + 4] =
        g.kind == GeometricRelation::Kind::Defence ? kRelationDefence : kRelationXRay;
    rel_ints += 5;
  }
  out.relation_ints_ = rel_ints;

  // -- W(k), E(k), src_extra(k) (c, d): per timeline, per origin, per listed
  //    board
  list_bits(out.destination_bits_, out.destinations_);
  list_bits(out.source_bits_, out.sources_);
  out.window(k, out.window_, Window::WithOrigins);
  out.e_boards(k, out.e_, Window::WithOrigins);
  out.src_extra(k, out.src_extra_, Window::WithOrigins);

  // -- per-square features (f)
  const auto cells = static_cast<std::size_t>(size.square_count());
  out.feature_boards_.clear();
  switch (options.square_features) {
    case SquareFeatures::All: {
      out.feature_boards_.resize(n);
      for (std::size_t i = 0; i < n; ++i) out.feature_boards_[i] = static_cast<std::int32_t>(i);
      out.piece_.resize(n * cells);
      out.colour_.resize(n * cells);
      out.moved_.resize(n * cells);
      std::uint8_t* piece = out.piece_.data();
      std::uint8_t* colour = out.colour_.data();
      std::uint8_t* moved = out.moved_.data();
      for (const TimelineRef& t : mv.timelines()) {
        t->boards().for_each([&](const BoardRef& board) {
          decode_board(*board, size, piece, colour, moved);
          piece += cells;
          colour += cells;
          moved += cells;
        });
      }
      break;
    }
    case SquareFeatures::Scope: {
      // The scope: W(k), E(k) and src_extra(k) - every square a move of (a)
      // starts from or lands on (a move within a board lands on its source).
      // W(k) is disjoint from the other two, which may share boards: a merge.
      const std::vector<std::int32_t>& w = out.window_;
      const std::vector<std::int32_t>& e = out.e_;
      const std::vector<std::int32_t>& s = out.src_extra_;
      std::size_t iw = 0;
      std::size_t ie = 0;
      std::size_t is = 0;
      constexpr std::int32_t kEnd = std::numeric_limits<std::int32_t>::max();
      for (;;) {
        const std::int32_t next = std::min({iw < w.size() ? w[iw] : kEnd, ie < e.size() ? e[ie] : kEnd,
                                            is < s.size() ? s[is] : kEnd});
        if (next == kEnd) break;
        out.feature_boards_.push_back(next);
        iw += iw < w.size() && w[iw] == next ? 1u : 0u;
        ie += ie < e.size() && e[ie] == next ? 1u : 0u;
        is += is < s.size() && s[is] == next ? 1u : 0u;
      }
      const std::size_t m = out.feature_boards_.size();
      out.piece_.resize(m * cells);
      out.colour_.resize(m * cells);
      out.moved_.resize(m * cells);
      for (std::size_t i = 0; i < m; ++i) {
        const std::int32_t board = out.feature_boards_[i];
        const std::size_t t = out.timeline_of_board(board);
        const Board* on = mv.timelines()[t]->board_at(out.tl_first_half_turn_[t] + (board - out.tl_first_board_[t]));
        decode_board(*on, size, out.piece_.data() + i * cells, out.colour_.data() + i * cells,
                     out.moved_.data() + i * cells);
      }
      break;
    }
    case SquareFeatures::None:
      out.piece_.clear();
      out.colour_.clear();
      out.moved_.clear();
      break;
  }
}

}  // namespace t5d
