// The proof channel (see t5d/prover.hpp and engine/README.md "Stage E8"):
// depth-limited df-pn over the semimove-decomposed turn tree. Rule IDs refer
// to the rulebook, docs/rules/README.md. Written from the design spec (§11.1,
// §8.2, §11.0 item 3), the rulebook and the README's design note. df-pn is
// Nagai's depth-first proof-number search, its threshold rule the textbook
// one and the "1 + epsilon" widening Pawlewicz and Lew's; dlshogi's df-pn mate
// solver is the precedent for using it as a mate solver beside a network
// search - cited, not read for code. No engine's or oracle's code was
// consulted.

#include "t5d/prover.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include "t5d/attacks.hpp"
#include "t5d/board.hpp"
#include "t5d/check.hpp"
#include "t5d/legal.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"

namespace t5d {

const char* seed_class_name(SeedClass c) noexcept {
  switch (c) {
    case SeedClass::Checks: return "checks";
    case SeedClass::Forcing: return "forcing";
    case SeedClass::All: return "all";
  }
  return "?";
}

const char* completion_class_name(CompletionClass c) noexcept {
  return c == CompletionClass::Canonical ? "canonical" : "every";
}

const char* proof_status_name(ProofStatus s) noexcept {
  switch (s) {
    case ProofStatus::Unknown: return "unknown";
    case ProofStatus::Proven: return "proven";
    case ProofStatus::Disproven: return "disproven";
  }
  return "?";
}

bool decomposes(const RuleOptions& r) noexcept {
  return r.moves_after_shift && r.activation == ActivationRule::Count &&
         r.reactivation == Reactivation::Rewinds && !r.play_activates &&
         r.castling_on_optional_boards;
}

// ---- the disproof store ------------------------------------------------------------------

struct DisproofStore::Shards {
  static constexpr std::size_t kCount = 64;
  struct Entry {
    GameState position;
    std::vector<std::pair<DisproofClass, int>> depths;
  };
  struct Shard {
    mutable std::shared_mutex mutex;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> index;
    std::vector<Entry> entries;
  };
  std::array<Shard, kCount> shard;
  Shard& of(const GameState& s) { return shard[static_cast<std::size_t>(s.zobrist() >> 58)]; }
  const Shard& of(const GameState& s) const {
    return shard[static_cast<std::size_t>(s.zobrist() >> 58)];
  }
};

DisproofStore::DisproofStore() : shards_(std::make_unique<Shards>()) {}
DisproofStore::~DisproofStore() = default;

int DisproofStore::find(const GameState& position, const DisproofClass& cls) const {
  const Shards::Shard& s = shards_->of(position);
  std::shared_lock<std::shared_mutex> lock(s.mutex);
  const auto it = s.index.find(position.zobrist());
  if (it == s.index.end()) return -1;
  for (const std::size_t i : it->second) {
    const Shards::Entry& e = s.entries[i];
    if (!same_position(e.position, position)) continue;
    for (const auto& [c, d] : e.depths) {
      if (c == cls) return d;
    }
    return -1;
  }
  return -1;
}

void DisproofStore::put(const GameState& position, const DisproofClass& cls, int depth) {
  Shards::Shard& s = shards_->of(position);
  std::unique_lock<std::shared_mutex> lock(s.mutex);
  std::vector<std::size_t>& slot = s.index[position.zobrist()];
  for (const std::size_t i : slot) {
    Shards::Entry& e = s.entries[i];
    if (!same_position(e.position, position)) continue;
    for (auto& [c, d] : e.depths) {
      if (c == cls) {
        d = std::max(d, depth);
        return;
      }
    }
    e.depths.emplace_back(cls, depth);
    return;
  }
  slot.push_back(s.entries.size());
  s.entries.push_back(Shards::Entry{position, {{cls, depth}}});
}

std::size_t DisproofStore::size() const {
  std::size_t n = 0;
  for (const Shards::Shard& s : shards_->shard) {
    std::shared_lock<std::shared_mutex> lock(s.mutex);
    n += s.entries.size();
  }
  return n;
}

void DisproofStore::clear() {
  for (Shards::Shard& s : shards_->shard) {
    std::unique_lock<std::shared_mutex> lock(s.mutex);
    s.index.clear();
    s.entries.clear();
  }
}

namespace {

// ---- proof numbers --------------------------------------------------------------------------

// Infinity, and "never": a value no search below can bring down - a node
// that cannot be proven (pn) or disproven (dn) because a listing of legal
// actions was incomplete or a detector ran out of memory. Every sum
// saturates at kNever.
constexpr std::uint64_t kInf = std::uint64_t{1} << 40;
constexpr std::uint64_t kNever = kInf - 1;

std::uint64_t sat_add(std::uint64_t a, std::uint64_t b) noexcept {
  return std::min(a + b, kNever);
}

// ---- the tree --------------------------------------------------------------------------------

enum class Kind : std::uint8_t { Turn, Semi };
// How a node is reached from its parent.
enum class Edge : std::uint8_t {
  Root,
  Action,  // a whole action from a turn start (moves, then the submission)
  Move,    // one move
  Pass,    // nothing played on the axis
  Enter,   // a turn's semimove tree, from its start
  Submit,  // the partial action submitted
};
enum class Phase : std::uint8_t { Plain, Branch };

// A turn's semimove order, fixed by its start: the axes (the mover's
// playable boards) in order; per axis, the last index of an axis with a jump
// that would land on it and play it (J4; -1 if none), and whether it has a
// branching move or a jump onto another axis (a later branch from it, J6's
// twin included) - what can make good a pass on it; how many leading axes a
// check rests on (D's turns); and whether a creation could relieve a
// mandatory board left unplayed (T8, T12).
struct TurnShape {
  std::vector<std::int32_t> axes;
  std::vector<int> arrival_last;
  std::vector<std::uint8_t> branchy;
  int hot = 0;
  bool relief = false;
};

struct Node {
  std::vector<Move> moves;              // Action: the action; Move: one move
  std::vector<std::uint32_t> children;
  TurnShape shape;                      // turn nodes, once expanded
  std::uint64_t pn = 1;
  std::uint64_t dn = 1;
  std::uint64_t key = 0;                // turn nodes: the TT key
  std::int16_t dist = -1;               // proven: half-turns to mate; -1 (AND side): dead
  std::int16_t axis = 0;                // semi, plain phase: the next axis
  std::int8_t remaining = 0;            // half-turns left at the (enclosing) turn's start
  Kind kind = Kind::Turn;
  Edge edge = Edge::Root;
  Phase phase = Phase::Plain;
  bool or_node = true;
  bool expanded = false;
  bool incomplete = false;              // a listing was capped: AND never proven, OR never disproven
  bool proven() const noexcept { return pn == 0; }
  bool disproven() const noexcept { return dn == 0; }
  bool solved() const noexcept { return pn == 0 || dn == 0; }
  bool stuck() const noexcept { return pn >= kNever && dn >= kNever; }
};

void set_proven(Node& n, int dist) {
  n.pn = 0;
  n.dn = kInf;
  n.dist = static_cast<std::int16_t>(dist);
}
void set_disproven(Node& n) {
  n.pn = kInf;
  n.dn = 0;
}

// ---- the transposition table (estimates only) -----------------------------------------------

// Turn nodes' (pn, dn), keyed by the 5D Zobrist key with the ruleset ID
// (H18), the role and the depth left. Lockless as the scanner's: a slot holds
// its entry packed in one word and the key XORed with it, so a slot being
// written reads as a miss. Never a fact: solved nodes live in the stores.
class EstimateTT {
 public:
  void resize(std::size_t entries) {
    std::size_t n = 1;
    while (n < entries) n <<= 1;
    slots_ = std::make_unique<Slot[]>(n);
    mask_ = n - 1;
  }
  void clear() {
    for (std::size_t i = 0; i <= mask_; ++i) {
      slots_[i].check.store(0, std::memory_order_relaxed);
      slots_[i].data.store(0, std::memory_order_relaxed);
    }
  }
  bool probe(std::uint64_t key, std::uint64_t& pn, std::uint64_t& dn) const {
    const Slot& s = slots_[static_cast<std::size_t>(key) & mask_];
    const std::uint64_t data = s.data.load(std::memory_order_relaxed);
    const std::uint64_t check = s.check.load(std::memory_order_relaxed);
    if (data == 0 || (check ^ data) != key) return false;
    pn = data & 0x7FFFFFFFu;
    dn = (data >> 31) & 0x7FFFFFFFu;
    return true;
  }
  void save(std::uint64_t key, std::uint64_t pn, std::uint64_t dn) {
    const std::uint64_t cap = 0x7FFFFFFFu;
    const std::uint64_t data =
        std::min(pn, cap) | (std::min(dn, cap) << 31) | (std::uint64_t{1} << 62);
    Slot& s = slots_[static_cast<std::size_t>(key) & mask_];
    s.data.store(data, std::memory_order_relaxed);
    s.check.store(key ^ data, std::memory_order_relaxed);
  }

 private:
  struct Slot {
    std::atomic<std::uint64_t> check{0};
    std::atomic<std::uint64_t> data{0};
  };
  std::unique_ptr<Slot[]> slots_;
  std::size_t mask_ = 0;
};

std::uint64_t tt_key(const GameState& s, bool or_node, int remaining) noexcept {
  std::uint64_t k = s.zobrist();
  k ^= (static_cast<std::uint64_t>(remaining) + 1) * 0x9E3779B97F4A7C15ull;
  k ^= or_node ? 0xD6E8FEB86659FD93ull : 0x2545F4914F6CDD1Dull;
  k ^= k >> 29;
  k *= 0xBF58476D1CE4E5B9ull;
  k ^= k >> 32;
  return k == 0 ? 1 : k;
}

// ---- helpers -----------------------------------------------------------------------------------

Piece arrived(const Move& m) {
  return m.promotes() ? Piece(m.promotion, m.piece.colour()) : m.piece.as_moved();
}

// A copy of `s` that shares no timeline and no board with it (the scanner's
// isolated_copy, for the same reason: threads that share a state's boards
// contend on their reference counts). Where the state cannot be rebuilt from
// its boards alone, `s` itself.
GameState isolated_copy(const GameState& s) {
  try {
    const Multiverse& mv = s.multiverse();
    Multiverse copy(mv.board_size(), mv.even(), mv.ruleset());
    copy.set_to_move(mv.to_move());
    for (const TimelineRef& t : mv.timelines()) {
      NewTimeline n;
      n.position = t->position();
      n.first_half_turn = t->first_half_turn();
      n.boards.reserve(static_cast<std::size_t>(t->length()));
      for (const BoardRef& b : t->boards()) n.boards.push_back(BoardBuilder(*b).publish());
      n.creator = t->creator();
      n.creation_order = t->creation_order();
      copy.add_timeline(std::move(n));
    }
    GameState out(std::move(copy), s.rules());
    if (out.zobrist() == s.zobrist() && same_position(out, s)) return out;
  } catch (const std::exception&) {
  }
  return s;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---- what the threads share --------------------------------------------------------------------

struct Shared {
  Shared(const ProverOptions& o, SolvedStore* p, DisproofStore* d)
      : opt(o), proofs(p != nullptr ? p : &own_proofs), disproofs(d != nullptr ? d : &own_disproofs) {
    opt.threads = std::max(1, opt.threads);
    opt.depth = std::max(1, std::min(opt.depth, 100));
    tt.resize(opt.tt_entries);
    lock_proofs = opt.threads > 1 && !proofs->concurrent();
  }
  ProverOptions opt;
  ConcurrentSolvedStore own_proofs;
  DisproofStore own_disproofs;
  SolvedStore* proofs;
  DisproofStore* disproofs;
  bool lock_proofs = false;
  std::mutex proofs_mutex;
  EstimateTT tt;

  // The search in progress.
  const ProveBudget* budget = nullptr;
  std::chrono::steady_clock::time_point start;
  std::size_t threads = 1;
  Colour attacker = Colour::White;
  std::atomic<std::uint64_t> nodes{0};
  std::atomic<bool> stop{false};

  std::optional<Proof> find_proof(const GameState& s) {
    if (!lock_proofs) return proofs->find(s);
    std::lock_guard<std::mutex> lock(proofs_mutex);
    return proofs->find(s);
  }
  void put_proof(const GameState& s, const Proof& p) {
    if (!lock_proofs) {
      proofs->put(s, p);
      return;
    }
    std::lock_guard<std::mutex> lock(proofs_mutex);
    proofs->put(s, p);
  }
  DisproofClass cls(bool or_node) const {
    return DisproofClass{or_node, opt.seeds, opt.completion};
  }
};

constexpr std::uint64_t kFlushNodes = 256;

// A turn's context during a search: the turn start, its node, and the two
// royal-attack maps of its start - the mover's (R1's status after the
// action) and the opponent's (C4 of the mover's partial actions).
struct TurnCtx {
  const GameState* start = nullptr;
  std::uint32_t node = 0;
  RoyalAttackMap mover_map;
  RoyalAttackMap opp_map;
};

// One action found by a depth-first search of a turn's semimove tree.
struct Found {
  std::vector<Move> moves;
  std::unique_ptr<GameState> after;  // before submission (a pointer: no optional to copy)
  RoyalAttackMap c4;      // the opponent's map there
};

enum class Prefer : std::uint8_t { Quiet, Escape };

// One search thread: its tree and its search.
struct Worker {
  Worker(Shared& shared, int index) : sh(shared), opt(shared.opt), id(index) {}

  Shared& sh;
  const ProverOptions& opt;
  int id = 0;
  double epsilon = 0;  // the helper's threshold widening

  std::deque<Node> tree;
  ProverStats st;
  std::uint64_t flushed = 0;
  bool stopped = false;
  std::vector<MoveList> lists;

  // -- budget -------------------------------------------------------------------------------

  std::uint64_t used() const {
    if (sh.threads == 1) return st.nodes;
    return sh.nodes.load(std::memory_order_relaxed) + (st.nodes - flushed);
  }
  void flush(bool all = false) {
    if (sh.threads == 1) return;
    const std::uint64_t mine = st.nodes - flushed;
    if (mine == 0 || (!all && mine < kFlushNodes)) return;
    sh.nodes.fetch_add(mine, std::memory_order_relaxed);
    flushed = st.nodes;
  }
  void halt() {
    stopped = true;
    if (sh.threads > 1) sh.stop.store(true, std::memory_order_relaxed);
  }
  bool spend(std::uint64_t n = 1) {
    if (stopped) return false;
    if (sh.budget == nullptr) {  // the tool functions: no budget
      st.nodes += n;
      return true;
    }
    if (sh.threads > 1 && sh.stop.load(std::memory_order_relaxed)) {
      stopped = true;
      return false;
    }
    if (used() + n > sh.budget->max_nodes) {
      st.stopped_by_nodes = true;
      halt();
      return false;
    }
    st.nodes += n;
    flush();
    if (sh.budget->max_time && (st.nodes & 63u) < n &&
        std::chrono::steady_clock::now() - sh.start > *sh.budget->max_time) {
      st.stopped_by_time = true;
      halt();
      return false;
    }
    return true;
  }
  SearchBudget detector_budget() const {
    SearchBudget b;
    b.max_bytes = opt.detector_max_bytes;
    if (sh.budget == nullptr) return b;
    const std::uint64_t u = used();
    b.max_nodes = sh.budget->max_nodes > u ? sh.budget->max_nodes - u : 0;
    if (sh.budget->max_time) {
      const auto left = *sh.budget->max_time - std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() - sh.start);
      b.max_time = left.count() > 0 ? left : std::chrono::nanoseconds(0);
    }
    return b;
  }
  // An E4 search's nodes are ours. False: it stopped on our budget.
  bool account(const SearchStats& s) {
    st.detector_nodes += s.nodes;
    st.nodes += s.nodes;
    flush();
    if (s.stopped_by_memory) {
      st.stopped_by_memory = true;
      return true;
    }
    if (s.stopped_by_budget && sh.budget != nullptr) {
      if (sh.budget->max_time &&
          std::chrono::steady_clock::now() - sh.start > *sh.budget->max_time) {
        st.stopped_by_time = true;
      } else {
        st.stopped_by_nodes = true;
      }
      halt();
      return false;
    }
    return true;
  }

  MoveList& scratch(std::size_t depth) {
    if (lists.size() <= depth) lists.resize(depth + 1);
    return lists[depth];
  }

  // -- the rules' pieces ------------------------------------------------------------------------

  void apply(GameState& s, const Move& m) {
    ++st.applies;
    s.apply(m);
  }

  RoyalAttackMap advance(const RoyalAttackMap& map, const GameState& t) {
    spend();
    ++st.attack_queries;
    return map.advanced(t.multiverse(), t.inactive_timelines());
  }
  // Whether a royal capture stands at the partial action `t` (`map`: the
  // opponent's map there): it lasts to the end of the action, so no
  // completion is legal (E5a's tier 0). Under Q18's alternative a new board
  // on void a capture crosses could block it, so only a capture that crosses
  // no void stands.
  static bool stands(const RoyalAttackMap& map, const GameState& t) {
    if (!map.any()) return false;
    if (!t.rules().sliders_cross_void) return true;
    for (const RoyalCapture& c : map.captures()) {
      bool over_void = false;
      for (const CapturePlace& p : capture_places(c)) {
        if (t.multiverse().board_at(p.timeline, p.half_turn) == nullptr) {
          over_void = true;
          break;
        }
      }
      if (!over_void) return true;
    }
    return false;
  }
  // The same, from the map before the move that made `t` (`c4`).
  bool standing(const RoyalAttackMap& c4, const GameState& t) {
    if (!spend()) return true;
    ++st.attack_queries;
    if (!t.rules().sliders_cross_void) return c4.any_capture_in(t.multiverse(), t.inactive_timelines());
    return stands(c4.advanced(t.multiverse(), t.inactive_timelines()), t);
  }

  int value(Piece p) const { return p.empty() ? 0 : opt.values.of(p.kind()); }

  // The scanner's (b): a piece of the mover's on a board `m` creates, that
  // board passed, can capture a royal of the opponent's.
  bool checks(const GameState& s, const Move& m) {
    if (!spend()) return false;
    ++st.attack_queries;
    const RuleOptions& r = s.rules();
    if (m.physical() && !m.castle() && !m.en_passant() && r.turn_zero_reachable &&
        r.inactive_history_is_destination) {
      // In place: the board the move creates, at its passed place (the
      // scanner's reasoning: the rules read the history board it leaves as
      // they read a playable one here).
      BoardBuilder b(*s.multiverse().board_at(m.from_timeline, m.from_half_turn));
      b.remove(m.from_square);
      b.put(m.to_square, arrived(m));
      const BoardRef created = b.publish();
      const ScanSource src{m.from_timeline, m.from_half_turn + 2, created.get()};
      return royal_capture_from(s.multiverse(), r, s.to_move(), s.inactive_timelines(),
                                std::span<const ScanSource>(&src, 1));
    }
    std::array<ScanSource, 2> sources{};
    Multiverse mv(s.multiverse().board_size(), s.multiverse().even());
    std::vector<int> inactive;
    const int n = materialise(s, m, mv, inactive, sources, nullptr);
    return royal_capture_from(mv, r, s.to_move(), inactive,
                              std::span<const ScanSource>(sources.data(), static_cast<std::size_t>(n)));
  }

  // The multiverse after `m` with every board `m` created advanced by a pass.
  int materialise(const GameState& s, const Move& m, Multiverse& mv, std::vector<int>& inactive,
                  std::array<ScanSource, 2>& sources, int* new_timeline) {
    GameState t = s;
    apply(t, m);
    mv = t.multiverse();
    inactive = t.inactive_timelines();
    std::array<int, 2> created{m.from_timeline, 0};
    int n = 1;
    if (!m.physical()) {
      const int nt = s.to_move() == Colour::White ? mv.max_timeline() : mv.min_timeline();
      if (new_timeline != nullptr) *new_timeline = nt;
      created[1] = m.branching() ? nt : m.to_timeline;
      n = 2;
    }
    for (int i = 0; i < n; ++i) {
      const int p = created[static_cast<std::size_t>(i)];
      BoardBuilder pass(*mv.head(p));
      pass.set_en_passant(kNoSquare);
      mv.append_board(p, pass.publish());
      sources[static_cast<std::size_t>(i)] = ScanSource{p, mv.timeline(p)->last_half_turn(), mv.head(p)};
    }
    return n;
  }

  // The scanner's (c): a branch whose new timeline, passed, holds an
  // immediate threat.
  bool threatens(const GameState& s, const Move& m) {
    if (opt.branch_threat == BranchThreat::None || !m.branching()) return false;
    if (!spend()) return false;
    std::array<ScanSource, 2> sources{};
    Multiverse mv(s.multiverse().board_size(), s.multiverse().even());
    std::vector<int> inactive;
    int nt = 0;
    materialise(s, m, mv, inactive, sources, &nt);
    MoveList out;
    append_moves_of(mv, s.rules(), s.to_move(), nt, out, inactive);
    for (const Move& r : out) {
      if (!r.capture() || r.captured.empty()) continue;
      if (opt.branch_threat == BranchThreat::ArrivalCapture && r.from_square != m.to_square) continue;
      if (value(r.captured) >= opt.branch_threat_value) return true;
    }
    return false;
  }

  bool is_seed(const GameState& s, const Move& m) {
    switch (opt.seeds) {
      case SeedClass::All: return true;
      case SeedClass::Checks: return checks(s, m);
      case SeedClass::Forcing:
        if (m.capture()) return true;
        if (checks(s, m)) return true;
        if (threatens(s, m)) return true;
        return opt.promotions_forcing && m.promotes();
    }
    return false;
  }

  // Whether a creation could relieve a mandatory board left unplayed: the
  // mover's next creation active (the count rule, T10) and some jump landing
  // before the present (T8), or an inactive timeline of the opponent's that
  // a creation would reactivate (T11, T12). Over-approximate: sound for
  // offering a pass, never for refusing one wrongly.
  bool relief_possible(const GameState& s, const MoveList& moves) const {
    if (s.rules().mandatory != MandatoryBoards::InPresent) return false;
    bool any_jump = false;
    bool early_jump = false;
    for (const Move& m : moves) {
      if (!m.jump()) continue;
      any_jump = true;
      if (m.to_half_turn < s.present()) early_jump = true;
    }
    if (!any_jump) return false;
    // Activity by other rules than the count rule's: any creation might.
    if (!decomposes(s.rules())) return true;
    const Multiverse& mv = s.multiverse();
    const Colour me = s.to_move();
    const Colour them = opposite(me);
    const auto credit = [&](Colour c) {
      return (s.rules().initial_credit == InitialTimelineCredit::MinusZeroToBlack && mv.even() &&
              c == Colour::Black)
                 ? 1
                 : 0;
    };
    const bool next_active =
        mv.created_by(them) + credit(them) >= mv.created_by(me) + 1 + credit(me) - 1;
    if (next_active && early_jump) return true;
    for (const int p : s.inactive_timelines()) {
      const Timeline* t = mv.timeline(p);
      if (t != nullptr && t->creator() == them) return true;
    }
    return false;
  }

  // -- move order ------------------------------------------------------------------------------

  // The scanner's quietness (its completion's choose()): a move within the
  // board capturing nothing, of the least valuable piece first; royals late;
  // jumps after; captures last.
  long quiet_rank(const Move& m) const {
    long r = m.piece.royal() ? 100000L : static_cast<long>(value(m.piece));
    if (!m.physical()) r += 1000000L;
    if (m.capture()) r += 2000000L;
    return r;
  }
  // An evasion's order on a board a check rests on: a royal's move, a
  // capture (the most valuable first), then the rest quietest first.
  long escape_rank(const Move& m) const {
    if (m.piece.royal()) return 0;
    if (m.capture()) return 1000L - value(m.captured);
    return 100000L + quiet_rank(m);
  }

  // A plain-phase node's options in preference order: indices into `moves`,
  // and -1 for the pass where offered.
  void order_options(const MoveList& moves, bool mandatory, bool pass_ok, bool hot, Prefer prefer,
                     std::vector<int>& out) const {
    out.clear();
    std::vector<int> idx(moves.size());
    for (std::size_t i = 0; i < moves.size(); ++i) idx[i] = static_cast<int>(i);
    const bool escape = prefer == Prefer::Escape && hot;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
      const Move& x = moves[static_cast<std::size_t>(a)];
      const Move& y = moves[static_cast<std::size_t>(b)];
      return escape ? escape_rank(x) < escape_rank(y) : quiet_rank(x) < quiet_rank(y);
    });
    if (pass_ok && !mandatory) out.push_back(-1);
    out.insert(out.end(), idx.begin(), idx.end());
    if (pass_ok && mandatory) out.push_back(-1);
  }

  // The plain-phase moves of axis `a` at `t`: the non-branching ones (into
  // `plain`) and whether the axis has a branching move or a jump onto
  // another board (a possible J6 twin) - what a pass on it may need.
  bool axis_moves(const GameState& t, int a, MoveList& plain) {
    plain.clear();
    if (!t.may_move()) return false;
    MoveList all;
    append_board_moves(t, a, all);
    bool branchy = false;
    for (const Move& m : all) {
      if (m.branching()) {
        branchy = true;
        continue;
      }
      if (m.jump()) branchy = true;
      plain.push_back(m);
    }
    if (opt.legality_tiers && !plain.empty()) {
      std::vector<MoveLegality> legal;
      classify_moves(t, plain, legal, LegalityOptions::budgeted(0, 0));
      std::size_t kept = 0;
      for (std::size_t i = 0; i < plain.size(); ++i) {
        if (legal[i] == MoveLegality::Illegal) {
          ++st.tier_cut;
          continue;
        }
        plain[kept++] = plain[i];
      }
      plain.resize(kept);
    }
    return branchy;
  }

  // -- the first action of a turn's semimove tree (completions, escapes) --------------------------

  // Depth-first over the semimove tree below the partial action `t` (its
  // opponent's map `c4`, no standing capture) in preference order; the first
  // legal action found, its moves appended to `path`. `capped`: the step cap
  // stopped it; otherwise "not found" means the subtree holds no legal action.
  // Whether a mandatory board at axis q, unplayed once the axes before `j`
  // are decided, can still be played or relieved: by a later branch from it
  // (its branching moves, or a jump that J6 may turn into one), by a jump
  // onto it from an axis not yet decided (J4), or by a creation's relief (T8,
  // T12). Over-approximate: never false where it could be.
  static bool can_be_made_good(const TurnShape& shape, std::size_t q, std::size_t j) {
    return shape.branchy[q] != 0 || shape.relief || shape.arrival_last[q] >= static_cast<int>(j);
  }
  // A pass on axis j: offered unless the board is mandatory and nothing could
  // make it good.
  static bool pass_viable(const GameState& t, const TurnShape& shape, std::size_t j) {
    return !t.mandatory(shape.axes[j]) || can_be_made_good(shape, j, j + 1);
  }
  // A partial action with an axis before `j` left unplayed, mandatory, and
  // beyond help: no completion is legal (T5).
  static bool pass_left_dead(const GameState& t, const TurnShape& shape, std::size_t j) {
    for (std::size_t q = 0; q < j && q < shape.axes.size(); ++q) {
      const int a = shape.axes[q];
      if (t.playable_by_mover(a) && t.mandatory(a) && !can_be_made_good(shape, q, j)) return true;
    }
    return false;
  }

  bool first_action(const GameState& t, const RoyalAttackMap& c4, const TurnShape& shape,
                    std::size_t j, Phase phase, Prefer prefer,
                    std::vector<Move>& path, std::uint64_t& steps, std::uint64_t cap, bool& capped,
                    Found& found) {
    const std::vector<std::int32_t>& axes = shape.axes;
    if (steps >= cap) {
      capped = true;
      return false;
    }
    ++steps;
    if (!spend()) return false;
    if (phase == Phase::Plain) {
      while (j < axes.size() && !t.playable_by_mover(axes[j])) ++j;
      if (j == axes.size()) phase = Phase::Branch;
    }
    if (pass_left_dead(t, shape, j)) {
      ++st.dead;
      return false;
    }
    if (phase == Phase::Plain) {
      const int a = axes[j];
      MoveList plain;
      axis_moves(t, a, plain);
      const bool mandatory = t.mandatory(a);
      const bool pass_ok = pass_viable(t, shape, j);
      if (!pass_ok) ++st.passes_cut;
      std::vector<int> order;
      order_options(plain, mandatory, pass_ok, static_cast<int>(j) < shape.hot, prefer, order);
      for (const int i : order) {
        if (stopped) return false;
        if (i < 0) {
          if (first_action(t, c4, shape, j + 1, Phase::Plain, prefer, path, steps, cap, capped,
                           found)) {
            return true;
          }
          continue;
        }
        const Move& m = plain[static_cast<std::size_t>(i)];
        GameState u = t;
        apply(u, m);
        const RoyalAttackMap next = advance(c4, u);
        if (stands(next, u)) {
          ++st.dead;
          continue;
        }
        path.push_back(m);
        if (first_action(u, next, shape, j + 1, Phase::Plain, prefer, path, steps, cap, capped,
                         found)) {
          return true;
        }
        path.pop_back();
        if (steps >= cap) {
          capped = true;
          return false;
        }
      }
      return false;
    }
    // The branch phase: submit, or branch on.
    if (t.obligations_met() && !c4.any()) {
      found.moves = path;
      found.after = std::make_unique<GameState>(t);
      found.c4 = c4;
      return true;
    }
    if (!t.may_move()) return false;
    MoveList all;
    generate_moves(t, all);
    for (const Move& b : all) {
      if (!b.branching()) continue;
      if (stopped) return false;
      GameState u = t;
      apply(u, b);
      const RoyalAttackMap next = advance(c4, u);
      if (stands(next, u)) {
        ++st.dead;
        continue;
      }
      path.push_back(b);
      if (first_action(u, next, shape, axes.size(), Phase::Branch, prefer, path, steps, cap, capped,
                       found)) {
        return true;
      }
      path.pop_back();
      if (steps >= cap) {
        capped = true;
        return false;
      }
    }
    return false;
  }

  // Every legal action of the semimove tree below `t` (tests): appended.
  void every_action(const GameState& t, const RoyalAttackMap& c4, const TurnShape& shape,
                    std::size_t j, Phase phase, std::vector<Move>& path, std::vector<Found>& out) {
    const std::vector<std::int32_t>& axes = shape.axes;
    if (phase == Phase::Plain) {
      while (j < axes.size() && !t.playable_by_mover(axes[j])) ++j;
      if (j == axes.size()) phase = Phase::Branch;
    }
    if (pass_left_dead(t, shape, j)) return;
    if (phase == Phase::Plain) {
      const int a = axes[j];
      MoveList plain;
      axis_moves(t, a, plain);
      if (pass_viable(t, shape, j)) every_action(t, c4, shape, j + 1, Phase::Plain, path, out);
      for (const Move& m : plain) {
        GameState u = t;
        apply(u, m);
        const RoyalAttackMap next = c4.advanced(u.multiverse(), u.inactive_timelines());
        if (stands(next, u)) continue;
        path.push_back(m);
        every_action(u, next, shape, j + 1, Phase::Plain, path, out);
        path.pop_back();
      }
      return;
    }
    if (t.obligations_met() && !c4.any()) out.push_back(Found{path, std::make_unique<GameState>(t), c4});
    if (!t.may_move()) return;
    MoveList all;
    generate_moves(t, all);
    for (const Move& b : all) {
      if (!b.branching()) continue;
      GameState u = t;
      apply(u, b);
      const RoyalAttackMap next = c4.advanced(u.multiverse(), u.inactive_timelines());
      if (stands(next, u)) continue;
      path.push_back(b);
      every_action(u, next, shape, axes.size(), Phase::Branch, path, out);
      path.pop_back();
    }
  }

  // The turn's axes: the mover's playable boards. A's turns (and the
  // canonical completion): ascending timeline. D's: the boards a check rests
  // on first (`hot` of them), then the mandatory ones, then the rest.
  std::vector<std::int32_t> attacker_axes(const GameState& s) const {
    std::vector<std::int32_t> out;
    for (const int p : s.playable_timelines()) out.push_back(p);
    return out;
  }
  std::vector<std::int32_t> defender_axes(const GameState& s, const RoyalAttackMap& opp_map, int& hot) {
    std::vector<std::int32_t> hot_axes;
    const std::vector<int> passed = passed_timelines(s);
    if (!passed.empty()) {
      const Multiverse pm = passed_multiverse(s);
      ++st.attack_queries;
      spend();
      const RoyalAttackMap m = opp_map.advanced(pm, s.inactive_timelines());
      for (const RoyalCapture& c : m.captures()) {
        for (const CapturePlace& place : capture_places(c)) {
          if (std::find(passed.begin(), passed.end(), place.timeline) == passed.end()) continue;
          const Timeline* t = s.multiverse().timeline(place.timeline);
          if (t == nullptr || place.half_turn != t->last_half_turn() + 1) continue;
          if (std::find(hot_axes.begin(), hot_axes.end(), place.timeline) == hot_axes.end()) {
            hot_axes.push_back(place.timeline);
          }
        }
      }
    }
    std::sort(hot_axes.begin(), hot_axes.end());
    std::vector<std::int32_t> out = hot_axes;
    std::vector<std::int32_t> mandatory;
    std::vector<std::int32_t> rest;
    for (const int p : s.playable_timelines()) {
      if (std::find(hot_axes.begin(), hot_axes.end(), p) != hot_axes.end()) continue;
      (s.mandatory(p) ? mandatory : rest).push_back(p);
    }
    out.insert(out.end(), mandatory.begin(), mandatory.end());
    out.insert(out.end(), rest.begin(), rest.end());
    hot = static_cast<int>(hot_axes.size());
    return out;
  }

  // The turn's shape from its start and its moves there (`moves`: every
  // pseudo-legal move at `s`). The candidate moves of an action are fixed at
  // its start (the mover's moves visit only boards of its own colour, which
  // the action changes only by playing them), so a later axis's jump onto an
  // axis exists mid-action only if it existed at the start.
  TurnShape shape_of(const GameState& s, std::vector<std::int32_t> axes, int hot,
                     const MoveList& moves) const {
    TurnShape shape;
    shape.axes = std::move(axes);
    shape.hot = hot;
    shape.relief = relief_possible(s, moves);
    shape.arrival_last.assign(shape.axes.size(), -1);
    shape.branchy.assign(shape.axes.size(), 0);
    const auto index = [&](int p) {
      const auto it = std::find(shape.axes.begin(), shape.axes.end(), p);
      return it == shape.axes.end() ? -1 : static_cast<int>(it - shape.axes.begin());
    };
    for (const Move& m : moves) {
      if (!m.jump()) continue;
      const int from = index(m.from_timeline);
      if (from >= 0) shape.branchy[static_cast<std::size_t>(from)] = 1;  // a branch, or a J6 twin
      if (m.branching()) continue;
      const int to = index(m.to_timeline);
      if (to >= 0 && from > to) {
        int& last = shape.arrival_last[static_cast<std::size_t>(to)];
        last = std::max(last, from);
      }
    }
    return shape;
  }

  // The canonical completion of `seed` at the turn start `s`: the first legal
  // leaf of its completion subtree in the quiet preference order.
  std::optional<Found> complete(const GameState& s, const RoyalAttackMap& opp_map,
                                const TurnShape& shape, const Move& seed) {
    GameState t = s;
    if (!spend()) return std::nullopt;
    apply(t, seed);
    const RoyalAttackMap c4 = advance(opp_map, t);
    if (stands(c4, t)) {
      ++st.dead;
      ++st.completion_failures;
      return std::nullopt;
    }
    std::vector<Move> path{seed};
    std::uint64_t steps = 0;
    bool capped = false;
    Found found;
    // A branching seed's completion has a plain phase too: the seed is the
    // action's first move (its creation first), its plain moves after it.
    const bool ok = first_action(t, c4, shape, 0, Phase::Plain, Prefer::Quiet, path, steps,
                                 opt.completion_max_nodes, capped, found);
    st.completion_steps += steps;
    if (!ok) {
      ++st.completion_failures;
      return std::nullopt;
    }
    ++st.completions;
    return found;
  }

  // -- nodes -----------------------------------------------------------------------------------

  std::uint32_t make_node() {
    if (tree.size() >= opt.max_tree_nodes) {
      st.stopped_by_memory = true;
      halt();
    }
    tree.emplace_back();
    ++st.tree_nodes;
    return static_cast<std::uint32_t>(tree.size() - 1);
  }

  // A turn node for `s` (a turn start): solved from the stores if they know,
  // else seeded from the TT's estimate.
  std::uint32_t make_turn(const GameState& s, int remaining, Edge edge, std::vector<Move> moves) {
    const std::uint32_t c = make_node();
    Node& n = tree[c];
    n.kind = Kind::Turn;
    n.edge = edge;
    n.moves = std::move(moves);
    n.or_node = s.to_move() == sh.attacker;
    n.remaining = static_cast<std::int8_t>(remaining);
    n.key = tt_key(s, n.or_node, remaining);
    ++st.turn_nodes;
    if (const std::optional<Proof> p = sh.find_proof(s)) {
      const Proof::Kind mine = n.or_node ? Proof::Kind::Win : Proof::Kind::Loss;
      if (p->kind == mine && p->distance <= remaining) {
        set_proven(n, p->distance);
        ++st.store_hits;
        return c;
      }
      if (p->kind != mine) {  // the other side wins, or a stalemate: no mate for A
        set_disproven(n);
        ++st.store_hits;
        return c;
      }
    }
    if (sh.disproofs->find(s, sh.cls(n.or_node)) >= remaining) {
      set_disproven(n);
      ++st.store_hits;
      return c;
    }
    if (n.or_node && remaining <= 0) {  // no half-turn left for A
      set_disproven(n);
      return c;
    }
    std::uint64_t pn = 0;
    std::uint64_t dn = 0;
    if (sh.tt.probe(n.key, pn, dn) && pn > 0 && dn > 0) {
      n.pn = pn;
      n.dn = dn;
      ++st.tt_hits;
    }
    return c;
  }

  std::uint32_t make_semi(bool or_node, int remaining, Edge edge, Phase phase, int axis,
                          std::optional<Move> move) {
    const std::uint32_t c = make_node();
    Node& n = tree[c];
    n.kind = Kind::Semi;
    n.edge = edge;
    n.or_node = or_node;
    n.remaining = static_cast<std::int8_t>(remaining);
    n.phase = phase;
    n.axis = static_cast<std::int16_t>(axis);
    if (move) n.moves.push_back(*move);
    return c;
  }

  void export_solved(const Node& n, const GameState& s) {
    if (n.kind != Kind::Turn) return;
    if (n.proven()) {
      if (n.dist < 0) throw std::logic_error("prover: a turn proven with no legal action under it");
      sh.put_proof(s, n.or_node ? Proof::win(n.dist) : Proof::loss(n.dist));
      ++st.proofs_exported;
    } else if (n.disproven() && n.remaining > 0 &&
               (opt.record_all_disproofs || n.edge == Edge::Root)) {
      sh.disproofs->put(s, sh.cls(n.or_node), n.remaining);
      ++st.disproofs_recorded;
    }
  }

  // -- expansion ---------------------------------------------------------------------------------

  // Successors already among a node's children (two actions may reach one).
  struct Seen {
    std::vector<GameState> states;
    bool add(const GameState& s) {
      const std::uint64_t k = s.zobrist();
      for (const GameState& x : states) {
        if (x.zobrist() == k && same_position(x, s)) return false;
      }
      states.push_back(s);
      return true;
    }
  };

  // E4's listing as a node's children (a turn that does not decompose).
  void list_children(std::uint32_t n, const GameState& s, bool attacker_turn) {
    ++st.listings;
    SearchBudget b = detector_budget();
    const ActionList l = list_legal_actions(s, opt.listing_cap + 1, Detector::Hypercuboid, b);
    if (!account(l.stats)) return;
    const bool complete = l.stats.exhaustive && !l.reached_cap;
    const int remaining = tree[n].remaining;
    Seen seen;
    for (const LegalAction& a : l.actions) {
      if (attacker_turn && opt.seeds != SeedClass::All) {
        bool seeded = false;
        for (const Move& m : a.moves) {
          if (is_seed(s, m)) {
            seeded = true;
            break;
          }
        }
        if (!seeded) continue;
      }
      if (!seen.add(a.result)) continue;
      const std::uint32_t c = make_turn(a.result, remaining - 1, Edge::Action, a.moves);
      tree[n].children.push_back(c);
    }
    if (!complete) tree[n].incomplete = true;
  }

  void expand_turn(std::uint32_t n, const GameState& s, TurnCtx& ctx) {
    ++st.expansions;
    tree[n].expanded = true;
    if (!spend()) return;
    MoveList& moves = scratch(0);
    generate_moves(s, moves);
    const MoveList all = moves;
    const int remaining = tree[n].remaining;
    if (tree[n].or_node) {
      expand_attacker(n, s, ctx, all, remaining);
    } else {
      expand_defender(n, s, ctx, all, remaining);
    }
  }

  void expand_attacker(std::uint32_t n, const GameState& s, TurnCtx& ctx, const MoveList& all,
                       int remaining) {
    const bool split = decomposes(s.rules());
    tree[n].shape = shape_of(s, attacker_axes(s), 0, all);
    if (opt.completion == CompletionClass::Every && !split) {
      list_children(n, s, true);
      return;
    }
    Seen seen;
    for (const Move& m : all) {
      if (stopped) return;
      if (!is_seed(s, m)) continue;
      ++st.seeds;
      if (opt.completion == CompletionClass::Canonical) {
        const std::optional<Found> f = complete(s, ctx.opp_map, tree[n].shape, m);
        if (!f) continue;
        GameState next = *f->after;
        next.submit(CheckRuling::none());  // legal: C4 held, the obligations met
        if (!seen.add(next)) continue;
        const std::uint32_t c = make_turn(next, remaining - 1, Edge::Action, f->moves);
        tree[n].children.push_back(c);
        continue;
      }
      GameState t = s;
      apply(t, m);
      if (standing(ctx.opp_map, t)) {
        ++st.dead;
        continue;
      }
      const std::uint32_t c = make_semi(true, remaining, Edge::Move, Phase::Plain, 0, m);
      tree[n].children.push_back(c);
    }
  }

  void expand_defender(std::uint32_t n, const GameState& s, TurnCtx& ctx, const MoveList& all,
                       int remaining) {
    const bool split = decomposes(s.rules());
    int hot = 0;
    std::vector<std::int32_t> axes = defender_axes(s, ctx.opp_map, hot);
    tree[n].shape = shape_of(s, std::move(axes), hot, all);
    // E4's question first: a quick escape search, then the hypercuboid
    // detector's exhaustive verdict.
    ++st.mate_tests;
    std::vector<Move> path;
    std::uint64_t steps = 0;
    bool capped = false;
    Found found;
    const bool escaped = first_action(s, ctx.opp_map, tree[n].shape, 0, Phase::Plain, Prefer::Escape,
                                      path, steps, opt.escape_probe_nodes, capped, found);
    if (stopped) return;
    std::vector<Move> witness;
    if (escaped) {
      ++st.escape_probe_hits;
      witness = found.moves;
    } else {
      const MateResult r = detect_mate(s, Detector::Hypercuboid, detector_budget());
      if (!account(r.stats)) return;
      switch (r.verdict) {
        case Outcome::Checkmate:
          ++st.checkmates;
          set_proven(tree[n], 0);
          return;
        case Outcome::Stalemate:
          ++st.stalemates;
          sh.put_proof(s, Proof::draw());
          ++st.proofs_exported;
          set_disproven(tree[n]);
          return;
        case Outcome::Ongoing:
          if (split && !capped) {
            throw std::logic_error(
                "prover: the semimove tree holds no legal action where E4 found one");
          }
          witness = r.witness;
          break;
        case Outcome::Unknown:
          // E4's memory ceiling: neither proven nor disproven - and never
          // proven by having no child.
          tree[n].incomplete = true;
          tree[n].pn = kNever;
          tree[n].dn = kNever;
          return;
      }
    }
    if (remaining == 0) {
      set_disproven(tree[n]);  // a legal action: not mated now, and no time left
      return;
    }
    GameState next = s;
    for (const Move& m : witness) apply(next, m);
    next.submit(CheckRuling::none());
    const std::uint32_t w = make_turn(next, remaining - 1, Edge::Action, witness);
    tree[n].children.push_back(w);
    if (split) {
      const std::uint32_t c = make_semi(false, remaining, Edge::Enter, Phase::Plain, 0, std::nullopt);
      tree[n].children.push_back(c);
    } else {
      list_children(n, s, false);
    }
  }

  void expand_semi(std::uint32_t n, const GameState& t, const TurnCtx& turn, const RoyalAttackMap& c4) {
    ++st.expansions;
    tree[n].expanded = true;
    if (!spend()) return;
    const TurnShape& shape = tree[turn.node].shape;
    const std::vector<std::int32_t>& axes = shape.axes;
    const bool or_node = tree[n].or_node;
    const int remaining = tree[n].remaining;
    std::size_t j = static_cast<std::size_t>(tree[n].axis);
    Phase phase = tree[n].phase;
    if (phase == Phase::Plain) {
      while (j < axes.size() && !t.playable_by_mover(axes[j])) ++j;
      if (j == axes.size()) phase = Phase::Branch;
    }
    if (pass_left_dead(t, shape, j)) {
      ++st.dead;  // no child: dead (an AND-side node proven, an OR-side one disproven)
      return;
    }
    if (phase == Phase::Plain) {
      const int a = axes[j];
      MoveList plain;
      axis_moves(t, a, plain);
      const bool mandatory = t.mandatory(a);
      const bool pass_ok = pass_viable(t, shape, j);
      if (!pass_ok) ++st.passes_cut;
      std::vector<int> order;
      order_options(plain, mandatory, pass_ok, !or_node && static_cast<int>(j) < shape.hot,
                    or_node ? Prefer::Quiet : Prefer::Escape, order);
      for (const int i : order) {
        if (stopped) return;
        if (i < 0) {
          const std::uint32_t c =
              make_semi(or_node, remaining, Edge::Pass, Phase::Plain, static_cast<int>(j) + 1, std::nullopt);
          tree[n].children.push_back(c);
          continue;
        }
        const Move& m = plain[static_cast<std::size_t>(i)];
        GameState u = t;
        apply(u, m);
        if (standing(c4, u)) {
          ++st.dead;
          continue;
        }
        const std::uint32_t c =
            make_semi(or_node, remaining, Edge::Move, Phase::Plain, static_cast<int>(j) + 1, m);
        tree[n].children.push_back(c);
      }
      return;
    }
    // The branch phase.
    tree[n].phase = Phase::Branch;
    if (t.obligations_met() && !c4.any()) {
      GameState next = t;
      next.submit(CheckRuling::none());  // legal: C4 holds, the obligations met
      const std::uint32_t c = make_turn(next, remaining - 1, Edge::Submit, {});
      tree[n].children.push_back(c);
    }
    if (!t.may_move()) return;
    MoveList all;
    generate_moves(t, all);
    for (const Move& b : all) {
      if (!b.branching()) continue;
      if (stopped) return;
      GameState u = t;
      apply(u, b);
      if (standing(c4, u)) {
        ++st.dead;
        continue;
      }
      const std::uint32_t c = make_semi(or_node, remaining, Edge::Move, Phase::Branch,
                                        static_cast<int>(axes.size()), b);
      tree[n].children.push_back(c);
    }
  }

  // -- proof and disproof numbers ----------------------------------------------------------------

  // A child's distance as its parent counts it: a submission ends a
  // half-turn; -1 (a dead AND-side node) adds nothing.
  static int contribution(const Node& c) {
    if (c.dist < 0) return -1;
    return c.dist + ((c.edge == Edge::Action || c.edge == Edge::Submit) ? 1 : 0);
  }

  void update(Node& n) {
    if (n.solved()) return;
    if (!n.expanded) return;
    std::uint64_t best = kInf;  // OR: least pn; AND: least dn
    std::uint64_t sum = 0;      // OR: dn; AND: pn
    bool any_open = false;
    int dist = n.or_node ? 1 << 14 : -1;
    bool all_solved_their_way = true;
    for (const std::uint32_t ci : n.children) {
      const Node& c = tree[ci];
      if (n.or_node) {
        if (c.proven()) {
          dist = std::min(dist, contribution(c));
          best = 0;
          continue;
        }
        best = std::min(best, c.pn);
        sum = sat_add(sum, c.dn);
        if (!c.disproven()) all_solved_their_way = false;
      } else {
        if (c.disproven()) {
          best = 0;
          continue;
        }
        best = std::min(best, c.dn);
        sum = sat_add(sum, c.pn);
        if (c.proven()) {
          dist = std::max(dist, contribution(c));
        } else {
          all_solved_their_way = false;
        }
      }
      if (!c.solved() && !c.stuck()) any_open = true;
    }
    if (n.or_node) {
      if (best == 0) {
        set_proven(n, dist);
        return;
      }
      if (all_solved_their_way && !n.incomplete) {
        set_disproven(n);
        return;
      }
      n.pn = best;
      n.dn = n.incomplete ? kNever : std::max<std::uint64_t>(sum, 1);
    } else {
      if (best == 0) {
        set_disproven(n);
        return;
      }
      if (all_solved_their_way && !n.incomplete) {
        if (n.kind == Kind::Turn) {
          for (const std::uint32_t ci : n.children) {
            if (tree[ci].edge == Edge::Enter && tree[ci].dist < 0) {
              throw std::logic_error(
                  "prover: a turn's semimove tree holds no legal action where one was found");
            }
          }
        }
        set_proven(n, dist);
        return;
      }
      n.dn = best;
      n.pn = n.incomplete ? kNever : std::max<std::uint64_t>(sum, 1);
    }
    if (!any_open) {
      n.pn = kNever;
      n.dn = kNever;
    }
  }

  // A helper's tie-break: a per-thread hash of the child's place.
  std::uint64_t jitter(std::uint32_t child) const {
    if (id == 0) return 0;
    std::uint64_t h = (static_cast<std::uint64_t>(child) + 1) * 0x9E3779B97F4A7C15ull ^
                      static_cast<std::uint64_t>(id) * 0xC2B2AE3D27D4EB4Full;
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 29;
    return h;
  }

  std::uint64_t widen(std::uint64_t second) const {
    if (second >= kNever) return kInf;
    std::uint64_t w = second + 1;
    if (epsilon > 0) {
      const auto e = static_cast<std::uint64_t>(std::ceil(static_cast<double>(second) * (1.0 + epsilon)));
      w = std::max(w, e);
    }
    return std::min(w, kInf);
  }

  // -- df-pn ---------------------------------------------------------------------------------

  void mid(std::uint32_t n, const GameState& s, TurnCtx& turn, const RoyalAttackMap& c4,
           std::uint64_t thpn, std::uint64_t thdn) {
    if (tree[n].solved()) return;
    if (!tree[n].expanded) {
      if (tree[n].kind == Kind::Turn) {
        expand_turn(n, s, turn);
      } else {
        expand_semi(n, s, turn, c4);
      }
      if (stopped) return;
      if (tree[n].solved()) {
        export_solved(tree[n], s);
        return;
      }
    }
    for (;;) {
      Node& node = tree[n];
      update(node);
      if (node.solved() || node.stuck() || node.pn >= thpn || node.dn >= thdn) break;
      if (stopped) return;
      // The child to search: the least pn (OR) or dn (AND) among the open
      // ones, and the second-best value.
      std::uint32_t pick = 0;
      bool picked = false;
      std::uint64_t first = kInf;
      std::uint64_t second = kInf;
      std::uint64_t first_tie = 0;
      for (const std::uint32_t ci : node.children) {
        const Node& c = tree[ci];
        if (c.solved() || c.stuck()) continue;
        const std::uint64_t v = node.or_node ? c.pn : c.dn;
        const std::uint64_t tie = jitter(ci);
        if (!picked || v < first || (v == first && tie < first_tie)) {
          if (picked) second = std::min(second, first);
          first = v;
          first_tie = tie;
          pick = ci;
          picked = true;
        } else {
          second = std::min(second, v);
        }
      }
      if (!picked) break;
      const Node& c = tree[pick];
      std::uint64_t cthpn = 0;
      std::uint64_t cthdn = 0;
      if (node.or_node) {
        cthpn = std::min(thpn, widen(second));
        const std::uint64_t others = node.dn >= c.dn ? node.dn - c.dn : 0;
        cthdn = thdn >= kInf ? kInf : (thdn > others ? thdn - others : 1);
      } else {
        cthdn = std::min(thdn, widen(second));
        const std::uint64_t others = node.pn >= c.pn ? node.pn - c.pn : 0;
        cthpn = thpn >= kInf ? kInf : (thpn > others ? thpn - others : 1);
      }
      descend(n, pick, s, turn, c4, cthpn, cthdn);
      if (stopped) return;
    }
    Node& node = tree[n];
    if (node.kind == Kind::Turn) {
      if (node.solved()) {
        export_solved(node, s);
      } else {
        sh.tt.save(node.key, node.pn, node.dn);
      }
    }
  }

  void descend(std::uint32_t n, std::uint32_t ci, const GameState& s, TurnCtx& turn,
               const RoyalAttackMap& c4, std::uint64_t thpn, std::uint64_t thdn) {
    const Edge edge = tree[ci].edge;
    switch (edge) {
      case Edge::Pass:
        mid(ci, s, turn, c4, thpn, thdn);
        return;
      case Edge::Enter:
        mid(ci, s, turn, turn.opp_map, thpn, thdn);
        return;
      case Edge::Move: {
        GameState t = s;
        if (!spend()) return;
        apply(t, tree[ci].moves.front());
        const RoyalAttackMap next =
            tree[n].kind == Kind::Turn ? advance(turn.opp_map, t) : advance(c4, t);
        mid(ci, t, turn, next, thpn, thdn);
        return;
      }
      case Edge::Submit: {
        GameState t = s;
        t.submit(CheckRuling::none());
        TurnCtx ctx;
        ctx.start = &t;
        ctx.node = ci;
        ctx.mover_map = c4;  // the opponent's map at the action's end is the new mover's
        ctx.opp_map = advance(turn.mover_map, t);
        mid(ci, t, ctx, ctx.opp_map, thpn, thdn);
        return;
      }
      case Edge::Action: {
        GameState t = s;
        if (!spend()) return;
        for (const Move& m : tree[ci].moves) apply(t, m);
        TurnCtx ctx;
        ctx.mover_map = advance(turn.opp_map, t);
        t.submit(CheckRuling::none());
        ctx.start = &t;
        ctx.node = ci;
        ctx.opp_map = advance(turn.mover_map, t);
        mid(ci, t, ctx, ctx.opp_map, thpn, thdn);
        return;
      }
      case Edge::Root:
        break;
    }
  }

  // The proof's first action at the root: the proven child's (an Every seed's
  // proven path, followed to its submission).
  std::vector<Move> first_action_of_proof(std::uint32_t root) const {
    std::vector<Move> out;
    std::uint32_t n = root;
    for (int guard = 0; guard < 4096; ++guard) {
      const Node& node = tree[n];
      std::uint32_t pick = 0;
      int best = 1 << 14;
      bool found = false;
      for (const std::uint32_t ci : node.children) {
        const Node& c = tree[ci];
        if (!c.proven()) continue;
        const int d = contribution(c);
        if (!found || d < best) {
          best = d;
          pick = ci;
          found = true;
        }
      }
      if (!found) return out;
      const Node& c = tree[pick];
      out.insert(out.end(), c.moves.begin(), c.moves.end());
      if (c.edge == Edge::Action || c.edge == Edge::Submit) return out;
      n = pick;
    }
    return out;
  }

  // This thread's search of `shared_root`.
  void run(const GameState& shared_root, std::uint32_t& root_out, bool& decided) {
    std::optional<GameState> own;
    if (id != 0) own.emplace(isolated_copy(shared_root));
    const GameState& root = own ? *own : shared_root;
    tree.clear();
    st = ProverStats{};
    flushed = 0;
    stopped = false;
    epsilon = id == 0 ? 0.0 : 0.25 * static_cast<double>(id);
    const std::uint32_t r = make_turn(root, opt.depth, Edge::Root, {});
    root_out = r;
    TurnCtx ctx;
    ctx.start = &root;
    ctx.node = r;
    spend(2);
    ctx.mover_map = RoyalAttackMap(root.multiverse(), root.rules(), root.to_move(),
                                   root.inactive_timelines());
    ctx.opp_map = RoyalAttackMap(root.multiverse(), root.rules(), opposite(root.to_move()),
                                 root.inactive_timelines());
    st.attack_queries += 2;
    if (!tree[r].solved()) mid(r, root, ctx, ctx.opp_map, kInf, kInf);
    decided = tree[r].solved();
    if (decided && sh.threads > 1) sh.stop.store(true, std::memory_order_relaxed);
    flush(true);
  }
};

void accumulate(ProverStats& into, const ProverStats& x) {
  into.nodes += x.nodes;
  into.tree_nodes += x.tree_nodes;
  into.turn_nodes += x.turn_nodes;
  into.expansions += x.expansions;
  into.applies += x.applies;
  into.attack_queries += x.attack_queries;
  into.dead += x.dead;
  into.passes_cut += x.passes_cut;
  into.tier_cut += x.tier_cut;
  into.seeds += x.seeds;
  into.completions += x.completions;
  into.completion_failures += x.completion_failures;
  into.completion_steps += x.completion_steps;
  into.mate_tests += x.mate_tests;
  into.escape_probe_hits += x.escape_probe_hits;
  into.checkmates += x.checkmates;
  into.stalemates += x.stalemates;
  into.detector_nodes += x.detector_nodes;
  into.listings += x.listings;
  into.tt_hits += x.tt_hits;
  into.store_hits += x.store_hits;
  into.proofs_exported += x.proofs_exported;
  into.disproofs_recorded += x.disproofs_recorded;
  into.stopped_by_nodes = into.stopped_by_nodes || x.stopped_by_nodes;
  into.stopped_by_time = into.stopped_by_time || x.stopped_by_time;
  into.stopped_by_memory = into.stopped_by_memory || x.stopped_by_memory;
}

}  // namespace

// ---- the prover ------------------------------------------------------------------------------

struct Prover::Impl {
  Impl(const ProverOptions& o, SolvedStore* p, DisproofStore* d) : shared(o, p, d) {
    for (int i = 0; i < shared.opt.threads; ++i) workers.push_back(std::make_unique<Worker>(shared, i));
  }
  Shared shared;
  std::vector<std::unique_ptr<Worker>> workers;

  ProveResult prove(const GameState& root, const ProveBudget& budget) {
    Shared& sh = shared;
    sh.budget = &budget;
    sh.start = std::chrono::steady_clock::now();
    sh.threads = workers.size();
    sh.attacker = root.to_move();
    sh.nodes.store(0, std::memory_order_relaxed);
    sh.stop.store(false, std::memory_order_relaxed);
    std::vector<std::uint32_t> roots(workers.size(), 0);
    std::vector<char> decided(workers.size(), 0);
    if (workers.size() == 1) {
      bool d = false;
      workers.front()->run(root, roots.front(), d);
      decided.front() = d ? 1 : 0;
    } else {
      std::vector<std::exception_ptr> errors(workers.size());
      const auto go = [&](std::size_t i) {
        try {
          bool d = false;
          workers[i]->run(root, roots[i], d);
          decided[i] = d ? 1 : 0;
        } catch (...) {
          errors[i] = std::current_exception();
          sh.stop.store(true, std::memory_order_relaxed);
        }
      };
      std::vector<std::thread> pool;
      pool.reserve(workers.size() - 1);
      for (std::size_t i = 1; i < workers.size(); ++i) pool.emplace_back(go, i);
      go(0);
      for (std::thread& t : pool) t.join();
      for (const std::exception_ptr& e : errors) {
        if (e) std::rethrow_exception(e);
      }
    }
    ProveResult result;
    result.depth = sh.opt.depth;
    result.threads = static_cast<int>(workers.size());
    {
      const Worker& main = *workers.front();
      if (roots.front() < main.tree.size() && main.tree[roots.front()].expanded) {
        result.root_actions = static_cast<int>(main.tree[roots.front()].children.size());
      }
    }
    for (std::size_t i = 0; i < workers.size(); ++i) accumulate(result.stats, workers[i]->st);
    // The root's verdict: any thread's (they agree, or the stores would have
    // refused a contradiction).
    bool proven = false;
    bool disproven = false;
    for (std::size_t i = 0; i < workers.size(); ++i) {
      if (!decided[i]) continue;
      const Node& r = workers[i]->tree[roots[i]];
      if (r.proven()) {
        if (!proven) {
          result.proof = Proof::win(r.dist);
          result.first_action = workers[i]->first_action_of_proof(roots[i]);
        } else if (r.dist < result.proof.distance) {
          result.proof = Proof::win(r.dist);
          result.first_action = workers[i]->first_action_of_proof(roots[i]);
        }
        proven = true;
      }
      if (r.disproven()) disproven = true;
    }
    if (proven && disproven) {
      throw std::logic_error("prover: the threads proved and disproved the root");
    }
    result.status = proven ? ProofStatus::Proven : disproven ? ProofStatus::Disproven : ProofStatus::Unknown;
    result.stats.seconds = seconds_since(sh.start);
    sh.budget = nullptr;
    return result;
  }
};

Prover::Prover(const ProverOptions& options, SolvedStore* proofs, DisproofStore* disproofs)
    : impl_(std::make_unique<Impl>(options, proofs, disproofs)) {}
Prover::~Prover() = default;

ProveResult Prover::prove(const GameState& root, const ProveBudget& budget) {
  return impl_->prove(root, budget);
}

void Prover::clear() { impl_->shared.tt.clear(); }

const ProverOptions& Prover::options() const noexcept { return impl_->shared.opt; }
SolvedStore& Prover::proofs() noexcept { return *impl_->shared.proofs; }
DisproofStore& Prover::disproofs() noexcept { return *impl_->shared.disproofs; }

std::vector<BuiltAction> Prover::attacker_actions(const GameState& turn_start) {
  Worker& w = *impl_->workers.front();
  impl_->shared.budget = nullptr;
  w.stopped = false;
  std::vector<BuiltAction> out;
  MoveList moves;
  generate_moves(turn_start, moves);
  const RoyalAttackMap opp(turn_start.multiverse(), turn_start.rules(), opposite(turn_start.to_move()),
                           turn_start.inactive_timelines());
  const TurnShape shape = w.shape_of(turn_start, w.attacker_axes(turn_start), 0, moves);
  Worker::Seen seen;
  for (const Move& m : moves) {
    if (!w.is_seed(turn_start, m)) continue;
    if (impl_->shared.opt.completion == CompletionClass::Canonical) {
      const std::optional<Found> f = w.complete(turn_start, opp, shape, m);
      if (!f) continue;
      GameState next = *f->after;
      next.submit(CheckRuling::none());
      if (seen.add(next)) out.push_back(BuiltAction{f->moves, next});
      continue;
    }
    GameState t = turn_start;
    t.apply(m);
    const RoyalAttackMap c4 = opp.advanced(t.multiverse(), t.inactive_timelines());
    if (Worker::stands(c4, t)) continue;
    std::vector<Move> path{m};
    std::vector<Found> found;
    w.every_action(t, c4, shape, 0, Phase::Plain, path, found);
    for (Found& f : found) {
      GameState next = *f.after;
      next.submit(CheckRuling::none());
      if (seen.add(next)) out.push_back(BuiltAction{f.moves, next});
    }
  }
  return out;
}

std::optional<BuiltAction> Prover::canonical_completion(const GameState& turn_start, const Move& seed) {
  Worker& w = *impl_->workers.front();
  impl_->shared.budget = nullptr;
  w.stopped = false;
  MoveList moves;
  generate_moves(turn_start, moves);
  const RoyalAttackMap opp(turn_start.multiverse(), turn_start.rules(), opposite(turn_start.to_move()),
                           turn_start.inactive_timelines());
  const std::optional<Found> f =
      w.complete(turn_start, opp, w.shape_of(turn_start, w.attacker_axes(turn_start), 0, moves), seed);
  if (!f) return std::nullopt;
  GameState next = *f->after;
  next.submit(CheckRuling::none());
  return BuiltAction{f->moves, next};
}

std::vector<BuiltAction> Prover::semimove_actions(const GameState& turn_start) {
  Worker& w = *impl_->workers.front();
  impl_->shared.budget = nullptr;
  w.stopped = false;
  std::vector<BuiltAction> out;
  MoveList moves;
  generate_moves(turn_start, moves);
  const RoyalAttackMap opp(turn_start.multiverse(), turn_start.rules(), opposite(turn_start.to_move()),
                           turn_start.inactive_timelines());
  int hot = 0;
  std::vector<std::int32_t> axes = w.defender_axes(turn_start, opp, hot);
  const TurnShape shape = w.shape_of(turn_start, std::move(axes), hot, moves);
  std::vector<Move> path;
  std::vector<Found> found;
  w.every_action(turn_start, opp, shape, 0, Phase::Plain, path, found);
  Worker::Seen seen;
  for (Found& f : found) {
    GameState next = *f.after;
    next.submit(CheckRuling::none());
    if (seen.add(next)) out.push_back(BuiltAction{f.moves, next});
  }
  return out;
}

}  // namespace t5d
