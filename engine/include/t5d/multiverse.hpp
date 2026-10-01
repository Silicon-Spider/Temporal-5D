// t5d/multiverse.hpp - a game state's boards: every timeline, every board.
//
// A Multiverse is a value that is cheap to copy and to branch, for search:
//
//  * Boards are immutable and shared (t5d/board.hpp, A4).
//  * Each timeline is an immutable Timeline object behind a shared pointer.
//    Copying a Multiverse copies only the vector of timeline pointers, so the
//    copy shares every timeline with the original. An edit (a new timeline, a
//    board appended, a head replaced) builds a new Timeline object for the one
//    timeline it touches and repoints that slot; every other copy keeps the
//    old object. That is copy-on-write per timeline.
//  * A timeline's boards are a persistent sequence (BoardSequence): its older
//    boards sit in fixed-size chunks that every version of the timeline
//    shares, and only the newest few are held by the Timeline object itself.
//    So the new Timeline object an edit builds costs the same however long
//    the timeline is - an append never copies the timeline's history.
//  * The 5D Zobrist key, the board count, the piece count and the latest
//    half-turn are maintained incrementally by every edit.
//
// Timelines occupy a contiguous range of positions p (t5d/timeline.hpp): new
// timelines are only ever created next to the outermost ones, so a timeline is
// found by offset. Each holds its boards one per half-turn, contiguous from its
// first half-turn; it records which colour created it and in which order (an
// initial timeline has no creator), and the board it emanated from, if any.
//
// What is NOT here, by design: the present, active and inactive timelines,
// turn submission, which boards a player must or may move on - those are rules
// (t5d/state.hpp). The multiverse stores the data they need: creators and
// creation order per timeline, the even/odd start, the ruleset ID and the
// player to move.
//
// Thread safety: Board and Timeline objects are immutable and may be read from
// any number of threads. A Multiverse value is like any value: concurrent reads
// are safe, a write needs exclusive access to that value (other copies are
// unaffected).

#ifndef T5D_MULTIVERSE_HPP
#define T5D_MULTIVERSE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "t5d/board.hpp"
#include "t5d/types.hpp"
#include "t5d/zobrist.hpp"

namespace t5d {

class Multiverse;

/// A timeline's boards, oldest first: a persistent sequence with structural
/// sharing. All but the newest 1..kChunkSize boards sit in full chunks of
/// kChunkSize boards; the chunks, and the list of them, are immutable once
/// built and shared by every copy. The newest boards (the "tail") are held
/// in the sequence itself. So copying a sequence costs O(kChunkSize) whatever
/// its length, push_back costs the same (plus, once every kChunkSize pushes,
/// a new list of chunk pointers: O(length / kChunkSize)), and indexing is
/// O(1). A sequence reached through a const Timeline is never changed.
class BoardSequence {
 public:
  static constexpr std::size_t kChunkSize = 16;

  BoardSequence() = default;
  /// Copies the chunk list's pointer and the tail only.
  BoardSequence(const BoardSequence& other) noexcept
      : chunks_(other.chunks_), tail_size_(other.tail_size_), size_(other.size_) {
    for (std::size_t i = 0; i < tail_size_; ++i) tail_[i] = other.tail_[i];
  }
  BoardSequence& operator=(const BoardSequence& other) noexcept {
    if (this != &other) {
      chunks_ = other.chunks_;
      for (std::size_t i = 0; i < kChunkSize; ++i) {
        tail_[i] = i < other.tail_size_ ? other.tail_[i] : BoardRef{};
      }
      tail_size_ = other.tail_size_;
      size_ = other.size_;
    }
    return *this;
  }
  BoardSequence(BoardSequence&&) noexcept = default;
  BoardSequence& operator=(BoardSequence&&) noexcept = default;
  ~BoardSequence() = default;

  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

  /// The i-th board, 0 <= i < size().
  const BoardRef& operator[](std::size_t i) const noexcept {
    const std::size_t shared = size_ - tail_size_;
    return i >= shared ? tail_[i - shared] : (*chunks_)[i / kChunkSize]->boards[i % kChunkSize];
  }
  /// The newest board. Precondition: !empty().
  const BoardRef& back() const noexcept { return tail_[tail_size_ - 1]; }

  /// Appends a board. When the tail is full it first becomes a new shared
  /// chunk. Strong guarantee: on bad_alloc nothing has changed.
  void push_back(BoardRef board);
  /// Replaces the newest board. Precondition: !empty().
  void set_back(BoardRef board) noexcept { tail_[tail_size_ - 1] = std::move(board); }

  class const_iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = BoardRef;
    using difference_type = std::ptrdiff_t;
    using pointer = const BoardRef*;
    using reference = const BoardRef&;

    const_iterator() noexcept = default;
    reference operator*() const noexcept { return (*seq_)[index_]; }
    pointer operator->() const noexcept { return &(*seq_)[index_]; }
    const_iterator& operator++() noexcept {
      ++index_;
      return *this;
    }
    const_iterator operator++(int) noexcept {
      const_iterator before = *this;
      ++index_;
      return before;
    }
    friend bool operator==(const const_iterator& a, const const_iterator& b) noexcept {
      return a.index_ == b.index_;
    }

   private:
    friend class BoardSequence;
    const_iterator(const BoardSequence* seq, std::size_t index) noexcept
        : seq_(seq), index_(index) {}
    const BoardSequence* seq_ = nullptr;
    std::size_t index_ = 0;
  };

  const_iterator begin() const noexcept { return const_iterator(this, 0); }
  const_iterator end() const noexcept { return const_iterator(this, size_); }

  /// Calls f(board) for every board, oldest first - chunk by chunk, without
  /// operator[]'s per-board lookup of its chunk (for walks over every board).
  template <class F>
  void for_each(F&& f) const {
    if (chunks_) {
      for (const std::shared_ptr<const Chunk>& chunk : *chunks_) {
        for (const BoardRef& board : chunk->boards) f(board);
      }
    }
    for (std::size_t i = 0; i < tail_size_; ++i) f(tail_[i]);
  }

 private:
  struct Chunk {
    std::array<BoardRef, kChunkSize> boards;
  };
  using ChunkList = std::vector<std::shared_ptr<const Chunk>>;

  std::shared_ptr<const ChunkList> chunks_;  // the full chunks; null while there are none
  std::array<BoardRef, kChunkSize> tail_{};  // the newest tail_size_ boards
  std::size_t tail_size_ = 0;                // 1..kChunkSize unless empty
  std::size_t size_ = 0;
};

/// One timeline: its boards, one per half-turn, contiguous from
/// first_half_turn(). Immutable once it belongs to a Multiverse.
class Timeline {
 public:
  int position() const noexcept { return position_; }
  int first_half_turn() const noexcept { return first_half_turn_; }
  /// The head board's half-turn.
  int last_half_turn() const noexcept {
    return first_half_turn_ + static_cast<int>(boards_.size()) - 1;
  }
  int length() const noexcept { return static_cast<int>(boards_.size()); }
  bool contains(int half_turn) const noexcept {
    return half_turn >= first_half_turn_ && half_turn <= last_half_turn();
  }

  /// The live board: the last one.
  const BoardRef& head() const noexcept { return boards_.back(); }
  /// The board at a half-turn, or nullptr outside the timeline.
  const Board* board_at(int half_turn) const noexcept {
    return contains(half_turn) ? boards_[static_cast<std::size_t>(half_turn - first_half_turn_)].get()
                               : nullptr;
  }
  const BoardRef* board_ref_at(int half_turn) const noexcept {
    return contains(half_turn) ? &boards_[static_cast<std::size_t>(half_turn - first_half_turn_)]
                               : nullptr;
  }
  /// All boards, oldest first (indexable and iterable like a vector).
  const BoardSequence& boards() const noexcept { return boards_; }

  /// The colour that created the timeline; nullopt for an initial timeline.
  std::optional<Colour> creator() const noexcept { return creator_; }
  /// 1 for its creator's first timeline, 2 for the second, ...; 0 if initial.
  int creation_order() const noexcept { return creation_order_; }
  /// The board the timeline emanated from, if known.
  std::optional<BoardCoord> origin() const noexcept { return origin_; }

 private:
  friend class Multiverse;

  int position_ = 0;
  int first_half_turn_ = 0;
  BoardSequence boards_;
  std::optional<Colour> creator_;
  int creation_order_ = 0;
  std::optional<BoardCoord> origin_;
};

using TimelineRef = std::shared_ptr<const Timeline>;

/// A timeline to add to a Multiverse.
struct NewTimeline {
  int position = 0;
  int first_half_turn = 0;
  /// One or more boards: first_half_turn, first_half_turn + 1, ...
  std::vector<BoardRef> boards;
  /// nullopt for an initial timeline; then creation_order must be 0.
  std::optional<Colour> creator;
  /// >= 1 when there is a creator.
  int creation_order = 0;
  /// The board it emanated from; must exist in the multiverse when given.
  std::optional<BoardCoord> origin;
};

class Multiverse {
 public:
  /// An empty multiverse (no timelines) for boards of `size`, in a game that
  /// starts with an even (true) or odd (false) number of timelines, under
  /// ruleset `ruleset`. Throws std::invalid_argument for an invalid size.
  Multiverse(BoardSize size, bool even, std::uint32_t ruleset = 0);

  /// Copies share every timeline (O(timelines) pointer copies). A copy keeps
  /// room for one more timeline, so that a branch made on it (what a search
  /// does with a copy) does not reallocate the vector it has just copied.
  Multiverse(const Multiverse& other);
  Multiverse& operator=(const Multiverse&) = default;
  /// A move leaves the source a valid, EMPTY multiverse of the same game (size,
  /// even/odd start, ruleset and player to move kept; no timelines, counts 0,
  /// the key of an empty state) - never a shell whose counts and key describe
  /// boards it no longer holds.
  Multiverse(Multiverse&& other) noexcept;
  Multiverse& operator=(Multiverse&& other) noexcept;
  ~Multiverse() = default;

  // -- the game ------------------------------------------------------------

  BoardSize board_size() const noexcept { return size_; }
  /// Whether the game started with an even number of timelines (labels -0/+0).
  bool even() const noexcept { return even_; }
  /// The ruleset (variant rules) ID. Opaque here; folded into the Zobrist key.
  /// The rules engine's states set it to their RuleOptions' ID (t5d/state.hpp).
  std::uint32_t ruleset() const noexcept { return ruleset_; }
  void set_ruleset(std::uint32_t ruleset) noexcept { ruleset_ = ruleset; }
  /// The player whose turn it is.
  Colour to_move() const noexcept { return to_move_; }
  void set_to_move(Colour c) noexcept { to_move_ = c; }

  // -- timelines -----------------------------------------------------------

  bool empty() const noexcept { return timelines_.empty(); }
  int timeline_count() const noexcept { return static_cast<int>(timelines_.size()); }
  /// Lowest and highest timeline positions. Precondition: !empty().
  int min_timeline() const noexcept { return min_position_; }
  int max_timeline() const noexcept { return min_position_ + timeline_count() - 1; }
  bool has_timeline(int p) const noexcept {
    return !timelines_.empty() && p >= min_position_ && p <= max_timeline();
  }
  /// The timeline at position p, or nullptr. Two copies of a multiverse return
  /// the same object for a timeline neither has edited.
  const Timeline* timeline(int p) const noexcept {
    return has_timeline(p) ? timelines_[static_cast<std::size_t>(p - min_position_)].get()
                           : nullptr;
  }
  /// Every timeline, in ascending position.
  const std::vector<TimelineRef>& timelines() const noexcept { return timelines_; }
  /// How many timelines a colour has created (initial timelines count for neither).
  int created_by(Colour c) const noexcept {
    return created_[static_cast<std::size_t>(index_of(c))];
  }

  // -- boards --------------------------------------------------------------

  /// The board at (p, h), or nullptr.
  const Board* board_at(int p, int h) const noexcept {
    const Timeline* t = timeline(p);
    return t != nullptr ? t->board_at(h) : nullptr;
  }
  const Board* board_at(BoardCoord c) const noexcept { return board_at(c.timeline, c.half_turn); }
  /// The board at (p, h) as a shared reference; empty if there is none.
  BoardRef board_ref(int p, int h) const;
  /// The head board of timeline p, or nullptr if there is no such timeline.
  const Board* head(int p) const noexcept {
    const Timeline* t = timeline(p);
    return t != nullptr ? t->head().get() : nullptr;
  }

  std::size_t board_count() const noexcept { return board_count_; }
  std::size_t piece_count() const noexcept { return piece_count_; }
  /// The latest half-turn of any board (the latest head); -1 when empty.
  /// Kept incrementally: O(1).
  int latest_half_turn() const noexcept { return latest_half_turn_; }

  /// Calls f(p, h, board) for every board, superseded and head alike, in the
  /// canonical order: ascending timeline, then ascending half-turn (so white's
  /// board before black's within a turn).
  template <class F>
  void for_each_board(F&& f) const {
    for (const TimelineRef& t : timelines_) {
      int h = t->first_half_turn();
      for (const BoardRef& b : t->boards()) {
        f(t->position(), h, *b);
        ++h;
      }
    }
  }

  // -- the 5D Zobrist key (H18) ---------------------------------------------

  /// The key of the whole state: every board in its place, the timelines'
  /// creators, the player to move, the even/odd start, the board size and the
  /// ruleset. A board's en-passant square counts only while the board is its
  /// timeline's head (zobrist::keyed_content_hash).
  std::uint64_t zobrist() const noexcept { return key_.key(to_move_, ruleset_, even_, size_); }
  /// The running sum behind the key (t5d/zobrist.hpp).
  const zobrist::Accumulator& zobrist_accumulator() const noexcept { return key_; }

  // -- edits ---------------------------------------------------------------
  //
  // Each edit touches one timeline and leaves every copy of this multiverse
  // unchanged. Each throws std::invalid_argument, leaving this multiverse
  // unchanged, if the edit would break an invariant.

  /// Adds a timeline. Its position must extend the range by one at either end
  /// (any position for the first). Its boards must be non-null, of this
  /// multiverse's size, and at half-turns >= 0.
  void add_timeline(NewTimeline timeline);

  /// Appends a board to timeline p, at the half-turn after its head.
  void append_board(int p, BoardRef board);

  /// Replaces the head board of timeline p. (The head is never superseded, so
  /// this does not touch an immutable past; it publishes a different head.)
  void replace_head(int p, BoardRef board);

  /// Value equality: same game data, same timelines, equal boards in the same
  /// places (compared by content, not by pointer).
  friend bool operator==(const Multiverse& a, const Multiverse& b) noexcept;

 private:
  void check_board(const BoardRef& board) const;
  TimelineRef& slot(int p);
  /// Drops every timeline and resets the counts and the key to an empty state's.
  void clear_contents() noexcept;

  BoardSize size_;
  bool even_ = false;
  std::uint32_t ruleset_ = 0;
  Colour to_move_ = Colour::White;
  int min_position_ = 0;
  std::vector<TimelineRef> timelines_;  // index: p - min_position_
  std::array<int, 2> created_{};
  std::size_t board_count_ = 0;
  std::size_t piece_count_ = 0;
  int latest_half_turn_ = -1;
  zobrist::Accumulator key_;
};

}  // namespace t5d

#endif  // T5D_MULTIVERSE_HPP
