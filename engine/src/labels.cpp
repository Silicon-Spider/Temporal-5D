// The per-square labels (see t5d/labels.hpp). Rule IDs refer to the rules
// reference, docs/rules/README.md. Written from that reference and the design
// spec; no other implementation was consulted. Every rule is read through the engine's own predicates: the move
// generator's moves and relations (t5d/movegen.hpp), check status's pass and
// check (t5d/check.hpp), submission (t5d/state.hpp) and E4's mate detector
// (t5d/search.hpp).

#include "t5d/labels.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "t5d/board.hpp"
#include "t5d/check.hpp"
#include "t5d/geometry.hpp"
#include "t5d/move.hpp"
#include "t5d/movegen.hpp"

namespace t5d {

namespace {

// ---- where a board is in the canonical order --------------------------------------------

class BoardIndex {
 public:
  explicit BoardIndex(const Multiverse& mv) {
    int b = 0;
    min_ = mv.empty() ? 0 : mv.min_timeline();
    for (const TimelineRef& t : mv.timelines()) {
      first_board_.push_back(b);
      first_half_.push_back(t->first_half_turn());
      length_.push_back(t->length());
      b += t->length();
    }
  }
  // The board at (timeline, half_turn), or -1.
  int at(int timeline, int half_turn) const noexcept {
    const long long i = static_cast<long long>(timeline) - min_;
    if (i < 0 || i >= static_cast<long long>(first_board_.size())) return -1;
    const auto t = static_cast<std::size_t>(i);
    const int offset = half_turn - first_half_[t];
    if (offset < 0 || offset >= length_[t]) return -1;
    return first_board_[t] + offset;
  }

 private:
  int min_ = 0;
  std::vector<int> first_board_;
  std::vector<int> first_half_;
  std::vector<int> length_;
};

// One side's view: the multiverse its pieces are read on, and how its places
// are reported on the state's boards.
struct View {
  const Multiverse* mv = nullptr;
  Colour side = Colour::White;
  const PassView* pass = nullptr;  // the opponent's view; nullptr: the acting player's

  int half_turn(int timeline, int h) const noexcept {
    return pass != nullptr ? pass->state_half_turn(timeline, h) : h;
  }
};

// A square of the state: board * cells + cell; -1 where no board of the state
// holds the place (none expected: every place a view reads is a board of the
// state or a passed copy).
struct Squares {
  const BoardIndex* index = nullptr;
  int width = 0;
  int cells = 0;

  int of(const View& v, int timeline, int half_turn, int square) const noexcept {
    const int b = index->at(timeline, v.half_turn(timeline, half_turn));
    if (b < 0) return -1;
    return b * cells + rank_of(square) * width + file_of(square);
  }
};

// A capture or a defence of a square: who (a square of the state) and what
// the piece is worth - for a capture also the move (its view's first moves,
// `view`, index `move`), which the exchange's recapture reading replays.
// Ordered and equal by (target, value, source) alone: one contact per piece.
struct Contact {
  int target = 0;
  int source = 0;
  int value = 0;
  int view = 0;
  std::size_t move = 0;
  friend bool operator<(const Contact& a, const Contact& b) noexcept {
    return std::tie(a.target, a.value, a.source) < std::tie(b.target, b.value, b.source);
  }
  friend bool operator==(const Contact& a, const Contact& b) noexcept {
    return std::tie(a.target, a.value, a.source) == std::tie(b.target, b.value, b.source);
  }
};

int value_of(const SquareLabelOptions& o, Piece p) noexcept {
  return p.royal() ? o.royal_value : o.values.of(p.kind());
}

// The board a piece stands on after `m`, under the static reading
// (t5d/labels.hpp): the board it lands on as it stands - for a move within
// its board, its own board without it - with the piece on the destination, the
// captured piece (en passant: the pawn) gone, a castling rook moved, a
// promotion made; no en-passant square.
BoardRef landing_board(const Multiverse& mv, const Move& m) {
  const Board* base = mv.board_at(m.to_timeline, m.to_half_turn);
  if (base == nullptr) return {};
  BoardBuilder b(*base);
  if (m.physical()) b.remove(m.from_square);
  if (m.en_passant()) b.remove(m.aux_square);
  if (m.castle()) {
    const Piece rook = b.at(m.aux_square);
    b.remove(m.aux_square);
    b.put(make_square((file_of(m.from_square) + file_of(m.to_square)) / 2, rank_of(m.to_square)),
          Piece(rook.kind(), rook.colour(), false));
  }
  const PieceKind kind = m.promotes() ? m.promotion : m.piece.kind();
  b.put(m.to_square, Piece(kind, m.piece.colour(), false));
  b.set_en_passant(kNoSquare);
  return b.publish();
}

// The swap algorithm: what the first capture of a piece worth `victim` wins
// for the capturing side, each side capturing with its least valuable piece
// and free to stop (`attackers`, `defenders`: values, ascending).
int static_exchange(int victim, const std::vector<int>& attackers,
                    const std::vector<int>& defenders) {
  if (attackers.empty()) return 0;
  std::vector<int> gain;
  gain.reserve(attackers.size() + defenders.size() + 1);
  gain.push_back(victim);
  int on_square = attackers[0];
  std::size_t next[2] = {1, 0};  // the next attacker, the next defender
  const std::vector<int>* sides[2] = {&attackers, &defenders};
  int side = 1;  // the defenders recapture first
  for (;;) {
    const std::vector<int>& pieces = *sides[side];
    if (next[side] >= pieces.size()) break;
    gain.push_back(on_square - gain.back());
    on_square = pieces[next[side]++];
    side = 1 - side;
  }
  for (std::size_t d = gain.size() - 1; d > 0; --d) {
    gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
  }
  return gain[0];
}

// The exchange's recapturers where the occupant's side cannot answer on the
// board taken on (t5d/labels.hpp, item 0): `base` is the capturing side's
// state - the decision state itself, or the opponent's view as a turn start of
// its own - and `m` its first capture of the occupant. The capture is played
// (GameState::apply: the board it creates - a copy of the board taken on with
// the capturer placed, the occupant's side to move - and its source board
// advanced, J4, J5), the capturing side passes on the boards it has still to
// play (pass_view, as the opponent's view is read), and the occupant's side's
// pieces that can then capture the capturer on the board created are its
// recapturers: their values, one per piece, ascending. nullopt where this
// cannot be read: the pass undefined (Q6's alternative; t5d/views.hpp).
std::optional<std::vector<int>> recapturers(const GameState& base, const Move& m,
                                            const SquareLabelOptions& options) {
  GameState after = base;
  after.apply(m);
  const PassView pass = pass_view(after);
  if (!pass.pass_defined) return std::nullopt;
  const Colour capturer = base.to_move();
  const Colour side = opposite(capturer);
  // Where the capturer now stands: the successor of the board taken on - for a
  // branching capture the first board of the new timeline, the capturer's
  // outermost (J5, L7).
  const Multiverse& now = after.multiverse();
  const int timeline = !m.branching()             ? m.to_timeline
                       : capturer == Colour::White ? now.max_timeline()
                                                   : now.min_timeline();
  const int half_turn = m.to_half_turn + 1;
  MoveList moves;
  for (const TimelineRef& t : pass.multiverse.timelines()) {
    append_moves_of(pass.multiverse, base.rules(), side, t->position(), moves,
                    after.inactive_timelines());
  }
  std::vector<std::tuple<int, int, int, int>> pieces;  // (value, source place)
  for (const Move& r : moves) {
    if (!r.capture() || r.to_timeline != timeline || r.to_half_turn != half_turn ||
        r.to_square != m.to_square) {
      continue;
    }
    pieces.emplace_back(value_of(options, r.piece), r.from_timeline, r.from_half_turn,
                        r.from_square);
  }
  std::sort(pieces.begin(), pieces.end());
  pieces.erase(std::unique(pieces.begin(), pieces.end()), pieces.end());
  std::vector<int> values;
  values.reserve(pieces.size());
  for (const auto& piece : pieces) values.push_back(std::get<0>(piece));
  return values;
}

}  // namespace

const char* square_label_name(int label) noexcept {
  static constexpr std::array<const char*, kSquareLabelCount> kNames = {
      "exchange_loses_occupant", "reachable_in_two_ours",      "reachable_in_two_theirs",
      "arrival_gives_check_ours", "arrival_gives_check_theirs", "royal_zone_ours",
      "royal_zone_theirs",        "arrival_gives_mate"};
  return label >= 0 && label < kSquareLabelCount ? kNames[static_cast<std::size_t>(label)] : "";
}

void square_labels(const GameState& state, const SquareLabelOptions& options, SquareLabels& out) {
  const Multiverse& mv = state.multiverse();
  const RuleOptions& rules = state.rules();
  const BoardSize size = mv.board_size();
  const Colour acting = state.to_move();
  const Colour opponent = opposite(acting);
  const std::span<const int> inactive = state.inactive_timelines();
  const BoardIndex index(mv);
  const Squares squares{&index, size.width, size.width * size.height};

  out.size = size;
  out.board_count = mv.board_count();
  out.acting_player = acting;
  out.stats = SquareLabelStats{};
  out.labels.assign(out.board_count * static_cast<std::size_t>(squares.cells) * kSquareLabelCount, 0);
  const auto set = [&](int sq, int label, std::uint8_t value = 1) {
    if (sq < 0) return;
    out.labels[static_cast<std::size_t>(sq) * kSquareLabelCount + static_cast<std::size_t>(label)] = value;
  };
  const auto get = [&](int sq, int label) {
    return out.labels[static_cast<std::size_t>(sq) * kSquareLabelCount + static_cast<std::size_t>(label)];
  };

  // -- the two views' first moves and defences
  const PassView pass = pass_view(state);
  const View views[2] = {View{&mv, acting, nullptr}, View{&pass.multiverse, opponent, &pass}};
  MoveList first[2];
  RelationList defences[2];
  generate_moves(state, first[0], defences[0], RelationKinds{true, false, false});
  for (const TimelineRef& t : pass.multiverse.timelines()) {
    append_moves_of(pass.multiverse, rules, opponent, t->position(), first[1], defences[1],
                    RelationKinds{true, false, false}, inactive);
  }

  // -- royal zones (5, 6): every royal piece on every board
  mv.for_each_board([&](int p, int h, const Board& board) {
    for (const Colour c : {Colour::White, Colour::Black}) {
      const int label = c == acting ? kRoyalZoneOurs : kRoyalZoneTheirs;
      for (std::uint64_t bits = board.pieces(c); bits != 0; bits &= bits - 1) {
        const int square = std::countr_zero(bits);
        if (!board.at(square).royal()) continue;
        const int x = file_of(square);
        const int y = rank_of(square);
        set(squares.of(views[0], p, h, square), label);
        for (const Vec4 v : geometry::all_directions()) {
          if (!size.contains(x + v.x, y + v.y)) continue;
          if (mv.board_at(p + v.l, h + 2 * v.t) == nullptr) continue;
          set(squares.of(views[0], p + v.l, h + 2 * v.t, make_square(x + v.x, y + v.y)), label);
        }
      }
    }
  });

  // -- each view: captures and defences (0), two-move reach (1, 2), arrival
  //    checks (3, 4)
  std::vector<Contact> captures;   // by the side of each view, of a square's piece
  std::vector<Contact> defended;   // by the side of each view, of its own piece
  MoveList second;
  std::vector<int> one;
  std::vector<int> two;
  for (int vi = 0; vi < 2; ++vi) {
    const View& v = views[vi];
    const MoveList& moves = first[static_cast<std::size_t>(vi)];
    out.stats.first_moves[vi] = moves.size();
    const int reach_label = vi == 0 ? kReachableInTwoOurs : kReachableInTwoTheirs;
    const int check_label = vi == 0 ? kArrivalGivesCheckOurs : kArrivalGivesCheckTheirs;
    for (const GeometricRelation& g : defences[static_cast<std::size_t>(vi)]) {
      const Board* board = v.mv->board_at(g.from_timeline, g.from_half_turn);
      defended.push_back(Contact{squares.of(v, g.to_timeline, g.to_half_turn, g.to_square),
                                 squares.of(v, g.from_timeline, g.from_half_turn, g.from_square),
                                 value_of(options, board->at(g.from_square))});
    }
    // A piece's moves are generated one after another: one run per piece.
    for (std::size_t i = 0; i < moves.size();) {
      std::size_t j = i;
      while (j < moves.size() && moves[j].from_timeline == moves[i].from_timeline &&
             moves[j].from_half_turn == moves[i].from_half_turn &&
             moves[j].from_square == moves[i].from_square) {
        ++j;
      }
      const int own = squares.of(v, moves[i].from_timeline, moves[i].from_half_turn,
                                 moves[i].from_square);
      one.clear();
      two.clear();
      for (std::size_t k = i; k < j; ++k) {
        const Move& m = moves[k];
        one.push_back(squares.of(v, m.to_timeline, m.to_half_turn, m.to_square));
        if (m.capture()) {
          const int taken = m.en_passant() ? m.aux_square : m.to_square;
          captures.push_back(Contact{squares.of(v, m.to_timeline, m.to_half_turn, taken), own,
                                     value_of(options, m.piece), vi, k});
        }
        const BoardRef landed = landing_board(*v.mv, m);
        if (!landed) continue;
        second.clear();
        append_moves_from(*v.mv, rules, m.to_timeline, m.to_half_turn, *landed, m.to_square,
                          second, inactive);
        out.stats.second_moves[vi] += second.size();
        bool checks = false;
        for (const Move& s : second) {
          two.push_back(squares.of(v, s.to_timeline, s.to_half_turn, s.to_square));
          checks = checks || (s.capture() && s.captured.royal());
        }
        if (checks) set(squares.of(v, m.to_timeline, m.to_half_turn, m.to_square), check_label);
      }
      std::sort(one.begin(), one.end());
      std::sort(two.begin(), two.end());
      two.erase(std::unique(two.begin(), two.end()), two.end());
      for (const int sq : two) {
        if (sq != own && !std::binary_search(one.begin(), one.end(), sq)) set(sq, reach_label);
      }
      i = j;
    }
  }

  // -- the static exchange (0) on every occupied square. Stable: a piece's
  //    first capture of a square, in generation order, is the one kept.
  std::stable_sort(captures.begin(), captures.end());
  captures.erase(std::unique(captures.begin(), captures.end()), captures.end());
  std::sort(defended.begin(), defended.end());
  defended.erase(std::unique(defended.begin(), defended.end()), defended.end());
  // A capture's side is its view's: the enemy of the piece it takes; a
  // defence's side the piece's own. So a square's captures are its enemy's
  // and its defences its own side's - whatever the view.
  std::vector<const Board*> boards;
  std::vector<std::pair<int, int>> places;  // each board's (timeline, half-turn)
  boards.reserve(out.board_count);
  places.reserve(out.board_count);
  mv.for_each_board([&](int p, int h, const Board& board) {
    boards.push_back(&board);
    places.emplace_back(p, h);
  });
  // The occupant's side answers on the board taken on - its recapturers read
  // as defences of the square - only where that board is passed (an acting
  // player's board in the present that it has not played, read in the
  // opponent's view on its passed copy): elsewhere it can answer only on the
  // board the capture creates, and the capture is replayed (recapturers). The
  // opponent's captures are replayed from its view as a turn start of its own,
  // built when first needed; nullopt where that cannot be built (the pass
  // undefined, or a view that is no turn start).
  const auto answered_in_place = [&](int board) {
    const auto [p, h] = places[static_cast<std::size_t>(board)];
    return pass.pass_defined && h == state.present() &&
           std::binary_search(pass.passed.begin(), pass.passed.end(), p);
  };
  std::optional<GameState> opponent_base;
  bool opponent_base_built = false;
  const auto base_of = [&](int view) -> const GameState* {
    if (view == 0) return &state;
    if (!opponent_base_built) {
      opponent_base_built = true;
      if (pass.pass_defined) {
        Multiverse turn = pass.multiverse;
        turn.set_to_move(opponent);
        try {
          opponent_base.emplace(std::move(turn), rules);
        } catch (const std::invalid_argument&) {
          // No turn start of the opponent's (the present not on its colour).
        }
      }
    }
    return opponent_base.has_value() ? &*opponent_base : nullptr;
  };
  std::vector<int> attackers;
  std::vector<int> defenders;
  std::size_t di = 0;
  for (std::size_t ci = 0; ci < captures.size();) {
    const int target = captures[ci].target;
    const std::size_t group = ci;
    attackers.clear();
    defenders.clear();
    for (; ci < captures.size() && captures[ci].target == target; ++ci) {
      attackers.push_back(captures[ci].value);  // one per piece, least valuable first
    }
    while (di < defended.size() && defended[di].target < target) ++di;
    for (std::size_t k = di; k < defended.size() && defended[k].target == target; ++k) {
      defenders.push_back(defended[k].value);
    }
    if (target < 0) continue;
    const Board& board = *boards[static_cast<std::size_t>(target / squares.cells)];
    const int cell = target % squares.cells;
    const Piece occupant = board.at(make_square(cell % size.width, cell / size.width));
    if (occupant.empty()) continue;
    if (!answered_in_place(target / squares.cells)) {
      // Each of the enemy's least valuable capturers may begin (they differ
      // in the board the capture creates and the source board it advances):
      // the enemy picks the best. 1 if one wins material; else Unknown if one
      // could not be read; else 0.
      ++out.stats.exchange_replays;
      bool wins = false;
      bool unread = false;
      for (std::size_t k = group; k < ci && captures[k].value == attackers[0] && !wins; ++k) {
        const GameState* base = base_of(captures[k].view);
        const Move& capture = first[static_cast<std::size_t>(captures[k].view)][captures[k].move];
        std::optional<std::vector<int>> replayed;
        if (base == &state) {
          replayed = recapturers(state, capture, options);
        } else if (base != nullptr) {
          // A move of the opponent's view that its turn start refuses - under a
          // ruleset whose activity is history's (Q1, Q3, Q6) the state built
          // from the view reads activity by the count rule - is not read.
          try {
            replayed = recapturers(*base, capture, options);
          } catch (const std::invalid_argument&) {
            replayed.reset();
          }
        }
        if (!replayed.has_value()) {
          unread = true;
        } else {
          wins = static_exchange(value_of(options, occupant), attackers, *replayed) > 0;
        }
      }
      if (wins) {
        set(target, kExchangeLosesOccupant);
      } else if (unread) {
        ++out.stats.exchange_unknown;
        set(target, kExchangeLosesOccupant, kLabelUnknown);
      }
      continue;
    }
    if (static_exchange(value_of(options, occupant), attackers, defenders) > 0) {
      set(target, kExchangeLosesOccupant);
    }
  }

  // -- arrival_gives_mate (7): the acting player's moves after which the
  //    submission is legal and the opponent, to move, is checkmated
  for (const Move& m : first[0]) {
    const int sq = squares.of(views[0], m.to_timeline, m.to_half_turn, m.to_square);
    if (sq < 0 || get(sq, kArrivalGivesMate) == 1) continue;
    GameState after = state;
    after.apply(m);
    if (after.submission_status() != SubmitStatus::Safe) continue;
    after.submit(CheckRuling::none());
    if (!in_check(after)) continue;
    ++out.stats.mate_candidates;
    ++out.stats.mate_searches;
    const MateResult r = detect_mate(after, Detector::Hypercuboid, options.mate_budget);
    if (r.verdict == Outcome::Checkmate) {
      set(sq, kArrivalGivesMate, 1);
    } else if (r.verdict == Outcome::Unknown) {
      ++out.stats.mate_unknown;
      set(sq, kArrivalGivesMate, kLabelUnknown);
    }
  }
}

}  // namespace t5d
