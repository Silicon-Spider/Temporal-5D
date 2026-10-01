// Multiverse edits: copy-on-write per timeline, persistent board sequences,
// incremental counts and key.

#include "t5d/multiverse.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "t5d/timeline.hpp"

namespace t5d {

namespace {

// Every half-turn index stays within [0, kMaxHalfTurn].
constexpr int kMaxHalfTurn = 2 * kMaxTurn + 1;

}  // namespace

// ---- BoardSequence -------------------------------------------------------------------

void BoardSequence::push_back(BoardRef board) {
  if (tail_size_ == kChunkSize) {
    // The tail is full: it becomes a new chunk, shared from now on, and the
    // list of chunks is rebuilt with it. Allocate everything first, so that a
    // bad_alloc leaves the sequence unchanged.
    auto chunk = std::make_shared<Chunk>();
    auto list = std::make_shared<ChunkList>();
    list->reserve((chunks_ ? chunks_->size() : 0) + 1);
    if (chunks_) list->assign(chunks_->begin(), chunks_->end());
    for (std::size_t i = 0; i < kChunkSize; ++i) chunk->boards[i] = std::move(tail_[i]);
    list->push_back(std::move(chunk));  // within the reserved capacity: no allocation
    chunks_ = std::move(list);
    tail_size_ = 0;
  }
  tail_[tail_size_++] = std::move(board);
  ++size_;
}

// ---- Multiverse --------------------------------------------------------------------------

Multiverse::Multiverse(BoardSize size, bool even, std::uint32_t ruleset)
    : size_(size), even_(even), ruleset_(ruleset) {
  if (!size.valid()) {
    throw std::invalid_argument("board size " + to_string(size) + " is outside 1x1..8x8");
  }
}

Multiverse::Multiverse(const Multiverse& other)
    : size_(other.size_),
      even_(other.even_),
      ruleset_(other.ruleset_),
      to_move_(other.to_move_),
      min_position_(other.min_position_),
      created_(other.created_),
      board_count_(other.board_count_),
      piece_count_(other.piece_count_),
      latest_half_turn_(other.latest_half_turn_),
      key_(other.key_) {
  timelines_.reserve(other.timelines_.size() + 1);
  timelines_.assign(other.timelines_.begin(), other.timelines_.end());
}

Multiverse::Multiverse(Multiverse&& other) noexcept
    : size_(other.size_),
      even_(other.even_),
      ruleset_(other.ruleset_),
      to_move_(other.to_move_),
      min_position_(other.min_position_),
      timelines_(std::move(other.timelines_)),
      created_(other.created_),
      board_count_(other.board_count_),
      piece_count_(other.piece_count_),
      latest_half_turn_(other.latest_half_turn_),
      key_(other.key_) {
  other.clear_contents();
}

Multiverse& Multiverse::operator=(Multiverse&& other) noexcept {
  if (this != &other) {
    size_ = other.size_;
    even_ = other.even_;
    ruleset_ = other.ruleset_;
    to_move_ = other.to_move_;
    min_position_ = other.min_position_;
    timelines_ = std::move(other.timelines_);
    created_ = other.created_;
    board_count_ = other.board_count_;
    piece_count_ = other.piece_count_;
    latest_half_turn_ = other.latest_half_turn_;
    key_ = other.key_;
    other.clear_contents();
  }
  return *this;
}

void Multiverse::clear_contents() noexcept {
  timelines_.clear();
  min_position_ = 0;
  created_ = {};
  board_count_ = 0;
  piece_count_ = 0;
  latest_half_turn_ = -1;
  key_ = zobrist::Accumulator{};
}

BoardRef Multiverse::board_ref(int p, int h) const {
  const Timeline* t = timeline(p);
  if (t == nullptr) return BoardRef{};
  const BoardRef* ref = t->board_ref_at(h);
  return ref != nullptr ? *ref : BoardRef{};
}

void Multiverse::check_board(const BoardRef& board) const {
  if (!board) throw std::invalid_argument("a timeline cannot hold a null board");
  if (board->size() != size_) {
    throw std::invalid_argument("a " + to_string(board->size()) + " board in a game of " +
                                to_string(size_) + " boards");
  }
}

TimelineRef& Multiverse::slot(int p) {
  if (!has_timeline(p)) {
    throw std::invalid_argument("there is no timeline at position " + std::to_string(p));
  }
  return timelines_[static_cast<std::size_t>(p - min_position_)];
}

void Multiverse::add_timeline(NewTimeline nt) {
  const int p = nt.position;
  // Bounded by the label it would be written with, so that every multiverse's
  // canonical string reads back (t5d/fen.hpp).
  if (p < min_timeline_position(even_) || p > kMaxTimelinePosition) {
    throw std::invalid_argument("timeline position " + std::to_string(p) + " is out of range (" +
                                std::to_string(min_timeline_position(even_)) + ".." +
                                std::to_string(kMaxTimelinePosition) + " in " +
                                (even_ ? "an even" : "an odd") + " game)");
  }
  bool at_front = false;
  if (!timelines_.empty()) {
    if (p == max_timeline() + 1) {
      at_front = false;
    } else if (p == min_position_ - 1) {
      at_front = true;
    } else {
      throw std::invalid_argument(
          "a new timeline must be next to an outermost one: position " + std::to_string(p) +
          ", existing positions " + std::to_string(min_position_) + ".." +
          std::to_string(max_timeline()));
    }
  }
  if (nt.boards.empty()) throw std::invalid_argument("a timeline needs at least one board");
  for (const BoardRef& board : nt.boards) check_board(board);
  const long long last = static_cast<long long>(nt.first_half_turn) +
                         static_cast<long long>(nt.boards.size()) - 1;
  if (nt.first_half_turn < 0 || last > kMaxHalfTurn) {
    throw std::invalid_argument("half-turns " + std::to_string(nt.first_half_turn) + ".." +
                                std::to_string(last) + " are out of range");
  }
  if (nt.creation_order < 0 || nt.creator.has_value() != (nt.creation_order > 0)) {
    throw std::invalid_argument(
        "a timeline has a creator exactly when its creation order is 1 or more (initial "
        "timelines have neither)");
  }
  if (nt.origin.has_value() && board_at(*nt.origin) == nullptr) {
    throw std::invalid_argument("the origin board (" + std::to_string(nt.origin->timeline) +
                                ", " + std::to_string(nt.origin->half_turn) +
                                ") is not in the multiverse");
  }

  zobrist::Accumulator key = key_;
  key.add_timeline(p, nt.creator);
  std::size_t pieces = 0;
  const std::size_t boards = nt.boards.size();
  for (std::size_t k = 0; k < boards; ++k) {
    const Board& board = *nt.boards[k];
    key.add_board(zobrist::keyed_content_hash(board.content_hash(), board.en_passant(),
                                              k + 1 == boards),
                  p, nt.first_half_turn + static_cast<int>(k));
    pieces += static_cast<std::size_t>(board.piece_count());
  }

  auto t = std::make_shared<Timeline>();
  t->position_ = p;
  t->first_half_turn_ = nt.first_half_turn;
  for (BoardRef& board : nt.boards) t->boards_.push_back(std::move(board));
  t->creator_ = nt.creator;
  t->creation_order_ = nt.creation_order;
  t->origin_ = nt.origin;
  const std::optional<Colour> creator = t->creator_;
  const int latest = t->last_half_turn();

  // The only step that can still throw (bad_alloc); nothing has changed yet.
  if (timelines_.empty()) {
    timelines_.push_back(std::move(t));
    min_position_ = p;
  } else if (at_front) {
    timelines_.insert(timelines_.begin(), std::move(t));
    min_position_ = p;
  } else {
    timelines_.push_back(std::move(t));
  }

  key_ = key;
  board_count_ += boards;
  piece_count_ += pieces;
  latest_half_turn_ = std::max(latest_half_turn_, latest);
  if (creator) ++created_[static_cast<std::size_t>(index_of(*creator))];
}

void Multiverse::append_board(int p, BoardRef board) {
  check_board(board);
  TimelineRef& s = slot(p);
  const Timeline& old = *s;
  const int h = old.last_half_turn() + 1;
  if (h > kMaxHalfTurn) throw std::invalid_argument("half-turn out of range");
  const std::uint64_t hash = board->content_hash();
  const int pieces = board->piece_count();
  // Copy-on-write: the old Timeline object stays with every other copy. The
  // copy shares the old one's chunks of boards (BoardSequence).
  auto t = std::make_shared<Timeline>(old);
  t->boards_.push_back(std::move(board));
  // The old head becomes a history board: its en-passant square, if it has
  // one, leaves the key (zobrist::keyed_content_hash).
  const Board& former_head = *old.head();
  if (former_head.en_passant() != kNoSquare) {
    key_.replace_board(former_head.content_hash(),
                       zobrist::keyed_content_hash(former_head.content_hash(),
                                                   former_head.en_passant(), false),
                       p, h - 1);
  }
  key_.add_board(hash, p, h);
  board_count_ += 1;
  piece_count_ += static_cast<std::size_t>(pieces);
  latest_half_turn_ = std::max(latest_half_turn_, h);
  s = std::move(t);
}

void Multiverse::replace_head(int p, BoardRef board) {
  check_board(board);
  TimelineRef& s = slot(p);
  const Timeline& old = *s;
  const Board& former = *old.head();
  const int h = old.last_half_turn();
  const std::uint64_t old_hash = former.content_hash();
  const std::uint64_t new_hash = board->content_hash();
  const auto old_pieces = static_cast<std::size_t>(former.piece_count());
  const auto new_pieces = static_cast<std::size_t>(board->piece_count());
  auto t = std::make_shared<Timeline>(old);  // the only step that can throw
  t->boards_.set_back(std::move(board));
  // Head for head: both are keyed with their full content hash.
  key_.replace_board(old_hash, new_hash, p, h);
  piece_count_ = piece_count_ - old_pieces + new_pieces;
  s = std::move(t);
}

bool operator==(const Multiverse& a, const Multiverse& b) noexcept {
  if (a.size_ != b.size_ || a.even_ != b.even_ || a.ruleset_ != b.ruleset_ ||
      a.to_move_ != b.to_move_ || a.timelines_.size() != b.timelines_.size()) {
    return false;
  }
  if (!a.timelines_.empty() && a.min_position_ != b.min_position_) return false;
  for (std::size_t i = 0; i < a.timelines_.size(); ++i) {
    const Timeline& x = *a.timelines_[i];
    const Timeline& y = *b.timelines_[i];
    if (&x == &y) continue;  // shared: equal
    if (x.position() != y.position() || x.first_half_turn() != y.first_half_turn() ||
        x.creator() != y.creator() || x.creation_order() != y.creation_order() ||
        x.origin() != y.origin() || x.boards().size() != y.boards().size()) {
      return false;
    }
    for (std::size_t k = 0; k < x.boards().size(); ++k) {
      const BoardRef& bx = x.boards()[k];
      const BoardRef& by = y.boards()[k];
      if (bx != by && !(*bx == *by)) return false;
    }
  }
  return true;
}

}  // namespace t5d
