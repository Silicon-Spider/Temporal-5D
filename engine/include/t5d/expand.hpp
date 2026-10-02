// t5d/expand.hpp - expansion: everything search and the network need from one
// position, in one call (design spec §6, §10.4, §11, §15.6; register B9, B11,
// M7, M9).
//
// "Search expands one node per network evaluation, so expansion - legal
// moves, attention masks, typed edges and E - must outrun evaluation with
// headroom" (§15.6, "Movegen throughput"). expand() computes, for the player to
// move:
//
//  a. moves - the pseudo-legal moves (t5d/movegen.hpp, generate_moves) - or,
//     with ExpandOptions::legal_moves, the legal moves only - each
//     also as a plain record (t5d/records.h): source and destination (timeline,
//     half-turn, square), their board indices and cells, the pieces, the
//     promotion, and the flags - capture, branching (J3), cross-board travel
//     (destination board != source board), castle, en passant, double step.
//  b. move_edges - the move graph (§6, B9: "which squares reach which, across
//     all boards"): one edge per distinct (source square, destination square)
//     pair of (a), so promotions merge.
//  c. window(k) - the board set W(k): for every ACTIVE timeline (T10) its
//     head, the up-to-k boards of its own immediately before the head (k
//     counts half-turns: one board per half-turn) - its BAND - and, if it was
//     created by branching, its branch origin (Timeline::origin), at every k.
//     These are the first three tiers of §10.4.3's queue ("present band ->
//     heads' recent k -> branch origins"); §10.4.3's budget mechanics call
//     only the first two the "mandatory core" and leave the origins among the
//     budgeted filler, so every count is also given for the bands alone
//     (Window::BandsOnly; board_is_origin(), band_depth()). Inactive timelines
//     get no band (§10.4.3: "their boards remain travel destinations: they
//     are E-material").
//  d. e_boards(k) - E(k) of §10.4.1, "out-of-window boards that are legal
//     travel destinations": the boards outside W(k) on which a TRAVEL move of
//     (a) lands - one whose destination board is not its source board (for a
//     branching move, the history board it targets, not the board it
//     creates). A move within a board is no travel: the playable head of an
//     inactive timeline (band-less) is in E only if a piece travels onto it.
//     src_extra(k) - the boards outside W(k) that hold the source square of a
//     move of (a) (such a head) - is never part of |E|; src_only(k) counts the
//     boards of src_extra(k) outside E(k): what W(k) + E(k) misses of the
//     policy pointer's source squares.
//  e. relations - the typed relations of §6 (B11, PROVISIONAL, each off by
//     default): attack, defence, x-ray. Reserved, not built: pin and check
//     (they would read check detection, t5d/check.hpp - stage E4 built that,
//     not these relations), and travel-threat (§6 and
//     B11 name the relation; §11.1 uses "travel threats" as a class of
//     forcing moves, "branching moves that create a timeline with an
//     immediate threat", but no edge - source and target squares - is
//     defined). Asking for any of the three throws.
//  f. boards - every board, in the canonical order (ascending timeline, then
//     half-turn), with the network contract's per-board features
//     (python/t5d/net/state.py): timeline position, half-turn, head,
//     active timeline, side to move, playable now, and the lineage's parent;
//     and per-square features - piece type id, colour, has-moved - for every
//     board or only for the scope W(k) + E(k) + src_extra(k)
//     (ExpandOptions::square_features).
//
// READINGS (engine README, "Expansion"):
//
//  * "some live piece" (§10.4.1's E) is read as the player to move's pieces on
//    their playable boards: E "comes from the move generator" at the node, and
//    the opponent's destinations enter at the child ("Search re-partitions per
//    node", §10.4.3). Reading it as both colours would add the opponent's
//    moves from boards it cannot move on now (append_moves_of, t5d/movegen.hpp,
//    can generate them) - a choice of scope the spec does not make.
//  * Moves are PSEUDO-LEGAL by default. ExpandOptions::legal_moves (off by
//    default, provisional) restricts them - and everything derived from them
//    - to the LEGAL moves: those that begin, or during an action continue, a
//    legal action (t5d/search.hpp, legal_moves). R1 judges legality per action
//    ("moving into check during a turn is allowed"), so that is a single
//    move's legality (build-log spec issue 28). One place chooses the moves
//    (expand.cpp, expansion_moves). With pseudo-legal moves every |E| is an
//    upper bound; the switch costs a search per move (the engine README gives
//    its cost on t5d-bench's classes).
//  * has-moved is 1 exactly on a pawn, rook, king or brawn that has lost its
//    unmoved marker (5DFEN's '*'); 0 on every other piece and empty square:
//    no rule reads the history of the other kinds, and 5DFEN does not record it.
//
// Cost. Allocation-free once an Expansion's buffers have grown (reuse one per
// thread). expand() does no work per frozen board: moves start only on
// playable boards, so frozen boards are read only along a move's path (move
// generation's work grows with the moves - and the pseudo-legal travel into a
// longer past grows with history, by the rules - not with the boards), and the
// rest is per timeline, per move, or per board of the scope - W(k), E(k),
// src_extra(k) - plus N/64 words of board bitsets. The per-board arrays (f)
// are O(boards): they are computed on first access after an expand() (one
// board alone: board_info()), so a caller that reads only the scope never pays
// for them. Square features are O(boards x squares) under SquareFeatures::All
// - the network's full-history input itself - and O(scope x squares) under
// SquareFeatures::Scope.

#ifndef T5D_EXPAND_HPP
#define T5D_EXPAND_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "t5d/move.hpp"
#include "t5d/movegen.hpp"
#include "t5d/records.h"
#include "t5d/state.hpp"
#include "t5d/types.hpp"

namespace t5d {

/// A move as the network and the C API see it (t5d/records.h).
using MoveRecord = ::t5d_move;

/// The relation ids of t5d/records.h (python/t5d/net/state.py RELATION_TYPES).
inline constexpr int kRelationMove = T5D_RELATION_MOVE;
inline constexpr int kRelationAttack = T5D_RELATION_ATTACK;
inline constexpr int kRelationDefence = T5D_RELATION_DEFENCE;
inline constexpr int kRelationPin = T5D_RELATION_PIN;
inline constexpr int kRelationXRay = T5D_RELATION_XRAY;
inline constexpr int kRelationCheck = T5D_RELATION_CHECK;
inline constexpr int kRelationTravelThreat = T5D_RELATION_TRAVEL_THREAT;

/// Which boards get per-square features.
enum class SquareFeatures : std::uint8_t {
  /// Every board: the full-history input.
  All = 0,
  /// The scope of the options' k: W(k), E(k) and src_extra(k) - the boards a
  /// windowed input is made of (§10.4.1), whose size does not grow with
  /// history at a fixed number of active timelines and a fixed |E|. A BOARD
  /// SET, not a position: it can hold a history board of a timeline without
  /// that timeline's head (an E board on an inactive timeline), which
  /// python/t5d/net/state.py's PositionState refuses (one head per timeline).
  /// What a windowed position holds is the window code's to define.
  Scope = 1,
  /// None.
  None = 2,
};

/// Which window a count or list is taken against: W(k) itself - the bands and
/// the branch origins - or the bands alone (§10.4.3 lists the origins as the
/// third tier of its queue, after the "mandatory core"; see the header comment).
enum class Window : std::uint8_t {
  WithOrigins = 0,
  BandsOnly = 1,
};

struct ExpandOptions {
  /// k of window(), e_boards() and src_extra() (and of the Scope features).
  /// Half-turns, >= 0; any k at least a timeline's length takes its whole
  /// band (INT32_MAX included). The tools report k in {0, 4, 8, 16}.
  int window_k = 8;
  SquareFeatures square_features = SquareFeatures::All;
  /// The legal moves only (t5d/search.hpp, legal_moves): the pseudo-legal
  /// moves after which the action can still be finished legally (C4, R1;
  /// build-log spec issue 28). PROVISIONAL, off by default: each move costs a
  /// search (to its first legal finish, by the hypercuboid detector), and the
  /// expansion then allocates.
  bool legal_moves = false;

  // The typed relations (§6, B11) - PROVISIONAL, each off by default. Only the
  // subset that check detection cannot change, for the player to move's
  // pieces on the boards its moves start from:

  /// attack: source square -> the square of the enemy piece a pseudo-legal
  /// capture of (a) takes (en passant: the pawn taken, not the square moved to).
  bool attack = false;
  /// defence: source square -> a piece of the mover's that the source piece's
  /// movement reaches with only that piece in the way (t5d/movegen.hpp).
  bool defence = false;
  /// x-ray: a slider's source square -> the second piece along one of its rays
  /// (t5d/movegen.hpp).
  bool xray = false;

  // Reserved: expand() throws std::invalid_argument if one is set.

  /// Would read check detection - what attacks a royal piece; pins follow
  /// from it (t5d/check.hpp, stage E4; R1, H36). Not built.
  bool pin = false;
  /// Would read check detection (t5d/check.hpp; R1, H36). Not built.
  bool check = false;
  /// Named by §6 and B11; §11.1's "travel threats" are a class of forcing
  /// moves, and no edge (source and target squares) is defined.
  bool travel_threat = false;
};

/// window_depth() and band_depth() of a board in W(k) for no k: a board of an
/// inactive timeline (that is no active timeline's branch origin, for
/// window_depth). Every board of an active timeline has a depth: its distance
/// from its head, in half-turns. No k reaches it: a k this large counts as
/// kNoWindow - 1.
inline constexpr std::int32_t kNoWindow = std::numeric_limits<std::int32_t>::max();

/// One board's features (f), as the per-board arrays hold them.
struct BoardInfo {
  std::int32_t timeline = 0;      // position p
  std::int32_t half_turn = 0;     // h = 2T + (1 if black is to move on the board)
  std::int32_t window_depth = 0;  // the least k with the board in W(k), or kNoWindow
  std::int32_t band_depth = 0;    // the same without the branch origins
  std::int32_t parent = -1;       // the lineage (board_parent())
  bool head = false;
  bool active = false;
  std::uint8_t side_to_move = 0;  // 0 white, 1 black
  bool playable = false;
  bool origin = false;            // an active timeline's branch origin
};

/// The result of expand(). Reuse one per thread: every call refills it without
/// allocating once its buffers have grown. Not thread-safe, not even for
/// reading: the per-board arrays are computed on first access. Views stay
/// valid until the next expand() into it.
class Expansion {
 public:
  // -- the position ---------------------------------------------------------------------

  BoardSize board_size() const noexcept { return size_; }
  bool even() const noexcept { return even_; }
  /// The player to move.
  Colour to_move() const noexcept { return to_move_; }
  /// The present (T4): a half-turn.
  int present() const noexcept { return present_; }
  /// Whether a move may be made now (Q4); false: (a) is empty.
  bool may_move() const noexcept { return may_move_; }
  int timeline_count() const noexcept { return timeline_count_; }
  int active_timeline_count() const noexcept { return active_count_; }
  /// Active timelines created by branching whose origin is unknown (a position
  /// read from 5DFEN, which records no origins): W(k) misses their origins.
  int origins_unknown() const noexcept { return origins_unknown_; }
  /// The options' k.
  int window_k() const noexcept { return window_k_; }

  // -- boards (f): the canonical order, ascending timeline then half-turn ---------------------
  //
  // The per-board arrays are computed on first access after an expand(),
  // O(boards); they may allocate then, until their buffers have grown.

  std::size_t board_count() const noexcept { return board_count_; }
  /// Timeline position p (t5d/timeline.hpp: "-0" -> -1, "+0" -> 0, ...).
  std::span<const std::int32_t> board_timelines() const {
    materialize();
    return timeline_of_;
  }
  /// Half-turn h = 2T + (1 if black is to move on the board).
  std::span<const std::int32_t> board_half_turns() const {
    materialize();
    return half_turn_of_;
  }
  /// 1: the head (latest board) of its timeline; 0: superseded.
  std::span<const std::uint8_t> board_is_head() const {
    materialize();
    return is_head_;
  }
  /// 1: its timeline is active (T10).
  std::span<const std::uint8_t> board_active() const {
    materialize();
    return active_;
  }
  /// The colour to move on the board: 0 white, 1 black (h & 1).
  std::span<const std::uint8_t> board_side_to_move() const {
    materialize();
    return side_;
  }
  /// 1: the player to move may move from the board now - its timeline's head,
  /// with the mover to move on it (L5), while a move may be made (Q4): exactly
  /// the boards (a)'s moves start from.
  std::span<const std::uint8_t> board_playable() const {
    materialize();
    return playable_;
  }
  /// The least k with the board in W(k) (0 for a branch origin), or kNoWindow.
  std::span<const std::int32_t> window_depth() const {
    materialize();
    return depth_;
  }
  /// The least k with the board in its timeline's band - W(k) without the
  /// branch origins: its distance from its active timeline's head, in
  /// half-turns, or kNoWindow (an inactive timeline).
  std::span<const std::int32_t> band_depth() const {
    materialize();
    return band_depth_;
  }
  /// 1: the board is the branch origin of an active timeline (in W(k) at every
  /// k, whatever its band depth).
  std::span<const std::uint8_t> board_is_origin() const {
    materialize();
    return origin_;
  }
  /// The network contract's lineage (python/t5d/net/state.py, `parent`): the
  /// board this board was made from - its timeline's previous board, or, for a
  /// timeline's first board, the board it branched from (Timeline::origin) -
  /// or -1: an initial timeline's first board, or an origin no one recorded (a
  /// position read from 5DFEN).
  std::span<const std::int32_t> board_parent() const {
    materialize();
    return parent_;
  }
  /// One board's entry of every per-board array, without computing them:
  /// O(log timelines). Precondition: 0 <= board < board_count().
  BoardInfo board_info(int board) const noexcept;
  /// The distinct boards that are an active timeline's branch origin,
  /// ascending.
  std::span<const std::int32_t> origins() const noexcept { return origins_; }
  std::size_t origin_count() const noexcept { return origins_.size(); }
  /// The index of the board at (timeline, half_turn), or -1.
  int board_index(int timeline, int half_turn) const noexcept;

  // -- per-square features (f) -------------------------------------------------------------

  /// The boards that have per-square features, ascending; the feature arrays
  /// hold one block of width x height cells per board, in this order.
  std::span<const std::int32_t> feature_boards() const noexcept { return feature_boards_; }
  /// Piece type id per cell: 0 empty, 1..12 = P N B R Q K U D S W C Y.
  std::span<const std::uint8_t> square_pieces() const noexcept { return piece_; }
  /// Colour per cell: 0 none, 1 white, 2 black.
  std::span<const std::uint8_t> square_colours() const noexcept { return colour_; }
  /// Has-moved per cell (see the header comment).
  std::span<const std::uint8_t> square_moved() const noexcept { return moved_; }

  // -- moves (a) ---------------------------------------------------------------------------

  /// The moves as t5d::Move, in generation order (for GameState::apply).
  const MoveList& moves() const noexcept { return moves_; }
  /// The same moves as records (t5d/records.h), in the same order.
  std::span<const MoveRecord> move_records() const noexcept {
    return {records_.data(), moves_.size()};
  }

  // -- the move graph (b) --------------------------------------------------------------------

  std::size_t move_edge_count() const noexcept { return edge_ints_ / 4; }
  /// Four int32 per edge: source board, source cell, destination board,
  /// destination cell. In move order, the first move of each pair kept.
  std::span<const std::int32_t> move_edges() const noexcept { return {edges_.data(), edge_ints_}; }

  // -- typed relations (e) ----------------------------------------------------------------------

  std::size_t relation_count() const noexcept { return relation_ints_ / 5; }
  /// Five int32 per relation: source board, source cell, target board, target
  /// cell, relation id (t5d/records.h). Attacks first (in move order, each pair
  /// once), then defences and x-rays (in generation order).
  std::span<const std::int32_t> relations() const noexcept {
    return {relations_.data(), relation_ints_};
  }

  // -- W, E and src_extra at the options' k (c, d): board indices, ascending ----------------

  std::span<const std::int32_t> window() const noexcept { return window_; }
  std::span<const std::int32_t> e_boards() const noexcept { return e_; }
  std::span<const std::int32_t> src_extra() const noexcept { return src_extra_; }

  /// The distinct boards that travel moves of (a) land on (destination board
  /// != source board), and that moves of (a) start from, ascending, whatever
  /// the window.
  std::span<const std::int32_t> travel_destinations() const noexcept { return destinations_; }
  std::span<const std::int32_t> source_boards() const noexcept { return sources_; }

  // -- the same at any k >= 0, against W(k) or the bands alone ------------------------------
  //
  // Per timeline, per origin and per listed board: none of these computes the
  // per-board arrays.

  std::size_t window_size(int k, Window which = Window::WithOrigins) const noexcept;
  std::size_t e_count(int k, Window which = Window::WithOrigins) const noexcept;
  std::size_t src_extra_count(int k, Window which = Window::WithOrigins) const noexcept;
  /// The boards of src_extra(k) that are not in E(k).
  std::size_t src_only_count(int k, Window which = Window::WithOrigins) const noexcept;
  void window(int k, std::vector<std::int32_t>& out, Window which = Window::WithOrigins) const;
  void e_boards(int k, std::vector<std::int32_t>& out, Window which = Window::WithOrigins) const;
  void src_extra(int k, std::vector<std::int32_t>& out, Window which = Window::WithOrigins) const;

 private:
  friend void expand(const GameState& state, const ExpandOptions& options, Expansion& out);

  void materialize() const;
  // The index of the timeline holding a board (0 <= board < board_count()).
  std::size_t timeline_of_board(int board) const noexcept;
  // A board's least k in W(k) or in its band.
  std::int32_t depth_of(int board, Window which) const noexcept;
  static bool bit(const std::vector<std::uint64_t>& bits, int board) noexcept {
    const auto b = static_cast<std::size_t>(board);
    return ((bits[b >> 6] >> (b & 63)) & 1u) != 0;
  }

  BoardSize size_{};
  bool even_ = false;
  Colour to_move_ = Colour::White;
  int present_ = 0;
  bool may_move_ = false;
  int timeline_count_ = 0;
  int active_count_ = 0;
  int origins_unknown_ = 0;
  int window_k_ = 0;
  std::size_t board_count_ = 0;

  // Per timeline, in the canonical order: its position, its first board's
  // index and half-turn, its length, whether it is active, and its branch
  // origin's board index (-1: none, or not recorded).
  std::vector<std::int32_t> tl_position_;
  std::vector<std::int32_t> tl_first_board_;
  std::vector<std::int32_t> tl_first_half_turn_;
  std::vector<std::int32_t> tl_length_;
  std::vector<std::uint8_t> tl_active_;
  std::vector<std::int32_t> tl_origin_;

  // Board sets as bitsets (one bit per board): the active timelines' branch
  // origins, the travel destinations, the sources.
  std::vector<std::uint64_t> origin_bits_;
  std::vector<std::uint64_t> destination_bits_;
  std::vector<std::uint64_t> source_bits_;

  // Per board, computed on first access (materialize()).
  mutable bool boards_ready_ = false;
  mutable std::vector<std::int32_t> timeline_of_;
  mutable std::vector<std::int32_t> half_turn_of_;
  mutable std::vector<std::uint8_t> is_head_;
  mutable std::vector<std::uint8_t> active_;
  mutable std::vector<std::uint8_t> side_;
  mutable std::vector<std::uint8_t> playable_;
  mutable std::vector<std::int32_t> depth_;
  mutable std::vector<std::int32_t> band_depth_;
  mutable std::vector<std::uint8_t> origin_;
  mutable std::vector<std::int32_t> parent_;

  // Per square.
  std::vector<std::int32_t> feature_boards_;
  std::vector<std::uint8_t> piece_;
  std::vector<std::uint8_t> colour_;
  std::vector<std::uint8_t> moved_;

  // Moves, the graph, relations. The last three grow and never shrink: their
  // used length is moves_.size(), edge_ints_ and relation_ints_.
  MoveList moves_;
  RelationList geometric_;
  std::vector<MoveRecord> records_;
  std::vector<std::int32_t> edges_;
  std::size_t edge_ints_ = 0;
  std::vector<std::int32_t> relations_;
  std::size_t relation_ints_ = 0;

  // Board sets, ascending.
  std::vector<std::int32_t> origins_;
  std::vector<std::int32_t> window_;
  std::vector<std::int32_t> e_;
  std::vector<std::int32_t> src_extra_;
  std::vector<std::int32_t> destinations_;
  std::vector<std::int32_t> sources_;
};

/// Expands `state` into `out` (see the header comment), reusing out's storage.
/// Throws std::invalid_argument for a negative window_k or a reserved relation.
/// (Any k >= kNoWindow - 1 is taken as kNoWindow - 1: every band whole.)
void expand(const GameState& state, const ExpandOptions& options, Expansion& out);

}  // namespace t5d

#endif  // T5D_EXPAND_HPP
