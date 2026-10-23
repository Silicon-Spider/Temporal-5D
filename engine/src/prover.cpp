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

std::uint64_t disproof_shape(const ProverOptions& o) noexcept {
  std::uint64_t h = std::uint64_t{0x9E3779B97F4A7C15};
  const auto mix = [&h](std::uint64_t v) {
    h ^= v + std::uint64_t{0x9E3779B97F4A7C15} + (h << 6) + (h >> 2);
    h *= std::uint64_t{0xBF58476D1CE4E5B9};
    h ^= h >> 31;
  };
  mix(static_cast<std::uint64_t>(o.seeds));
  mix(static_cast<std::uint64_t>(o.completion));
  mix(o.completion_max_nodes);
  for (const int v : o.values.value) mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(v)));
  mix(static_cast<std::uint64_t>(o.branch_threat));
  mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(o.branch_threat_value)));
  mix(o.promotions_forcing ? 1u : 0u);
  mix(o.legality_tiers ? 1u : 0u);
  mix(o.check_filter ? 1u : 0u);
  return h;
}

bool decomposes(const RuleOptions& r) noexcept {
  return r.moves_after_shift && r.activation == ActivationRule::Count &&
         r.reactivation == Reactivation::Rewinds && !r.play_activates &&
         r.castling_on_optional_boards;
}

// ---- the disproof store ------------------------------------------------------------------

// The solved store's sharding (scanner.cpp): 64 shards by the key's top bits,
// each behind its own ShardLock (exclusive: see t5d/scanner.hpp).
struct DisproofStore::Shards {
  static constexpr std::size_t kCount = 64;
  struct Entry {
    GameState position;
    std::vector<std::pair<DisproofClass, int>> depths;
  };
  struct Shard {
    mutable ShardLock mutex;
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
  const std::lock_guard<ShardLock> lock(s.mutex);
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
  const std::lock_guard<ShardLock> lock(s.mutex);
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
    const std::lock_guard<ShardLock> lock(s.mutex);
    n += s.entries.size();
  }
  return n;
}

// ---- proof certificates ----------------------------------------------------------------------

void ProofCertificate::add(const GameState& position, std::vector<Move> action, int distance) {
  const std::uint64_t key = position.zobrist();
  const auto range = index_.equal_range(key);
  for (auto it = range.first; it != range.second; ++it) {
    CertificateStep& s = steps_[it->second];
    if (!same_position(s.position, position)) continue;
    if (distance < s.distance) {
      s.action = std::move(action);
      s.distance = distance;
    }
    return;
  }
  steps_.push_back(CertificateStep{position, std::move(action), distance});
  index_.emplace(key, steps_.size() - 1);
}

const CertificateStep* ProofCertificate::find(const GameState& position) const {
  const auto range = index_.equal_range(position.zobrist());
  for (auto it = range.first; it != range.second; ++it) {
    if (same_position(steps_[it->second].position, position)) return &steps_[it->second];
  }
  return nullptr;
}

void DisproofStore::clear() {
  for (Shards::Shard& s : shards_->shard) {
    const std::lock_guard<ShardLock> lock(s.mutex);
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
// The last-turn check filter (t5d/prover.hpp): the attacker's moves at its
// last turn start, each check-capable or not; per axis index j, whether a
// check-capable non-branching move is left on axes j onward; whether a
// check-capable branching move exists.
struct CheckFilter {
  std::vector<Move> moves;
  std::vector<std::uint8_t> capable_move;
  std::vector<std::uint8_t> by_void;  // capable through a void stop alone
  std::unordered_multimap<std::uint64_t, std::size_t> index;
  std::vector<std::uint8_t> plain_from;  // axes.size() + 1 entries
  bool branch = false;

  static std::uint64_t key(const Move& m) noexcept {
    std::uint64_t k = static_cast<std::uint64_t>(static_cast<std::uint32_t>(m.from_timeline));
    k = k * 1000003u + static_cast<std::uint64_t>(static_cast<std::uint32_t>(m.from_half_turn));
    k = k * 1000003u + static_cast<std::uint64_t>(static_cast<std::uint32_t>(m.to_timeline));
    k = k * 1000003u + static_cast<std::uint64_t>(static_cast<std::uint32_t>(m.to_half_turn));
    k = k * 131u + static_cast<std::uint64_t>(static_cast<std::uint8_t>(m.from_square));
    k = k * 131u + static_cast<std::uint64_t>(static_cast<std::uint8_t>(m.to_square));
    k = k * 131u + static_cast<std::uint64_t>(m.flags);
    return k;
  }
  // A move the turn start did not offer in this form (a jump that became a
  // branch mid-action) is check-capable: the filter only ever cuts what it
  // knows.
  bool capable(const Move& m) const {
    const auto range = index.equal_range(key(m));
    for (auto it = range.first; it != range.second; ++it) {
      if (moves[it->second] == m) return capable_move[it->second] != 0;
    }
    return true;
  }
};

struct TurnShape {
  std::vector<std::int32_t> axes;
  std::vector<int> arrival_last;
  std::vector<std::uint8_t> branchy;
  int hot = 0;
  bool relief = false;
  std::shared_ptr<const CheckFilter> filter;  // the attacker's last turn
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
  bool check_made = false;              // semi, the attacker's last turn: a check-capable move is in it
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
  k ^= (static_cast<std::uint64_t>(remaining) + 1) * std::uint64_t{0x9E3779B97F4A7C15};
  k ^= or_node ? std::uint64_t{0xD6E8FEB86659FD93} : std::uint64_t{0x2545F4914F6CDD1D};
  k ^= k >> 29;
  k *= std::uint64_t{0xBF58476D1CE4E5B9};
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
    shape = disproof_shape(opt);
    lock_proofs = opt.threads > 1 && !proofs->concurrent();
  }
  ProverOptions opt;
  std::uint64_t shape = 0;  // disproof_shape(opt)
  ConcurrentSolvedStore own_proofs;
  DisproofStore own_disproofs;
  SolvedStore* proofs;
  DisproofStore* disproofs;
  bool lock_proofs = false;
  std::mutex proofs_mutex;
  EstimateTT tt;

  // The search in progress.
  const ProveBudget* budget = nullptr;
  int depth = 5;
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
    return DisproofClass{or_node, opt.seeds, opt.completion, shape};
  }
  // record_certificate: every attacker position proven by expansion, in
  // any thread and any search of this prover, with its proving action - so
  // a proof read back from the solved store (another thread's, an earlier
  // search's) is certified too.
  std::mutex cert_mutex;
  ProofCertificate cert;
  // Prove-only keeps its disproofs to itself - its search's working memory,
  // emptied at each search, never the caller's store (it claims none).
  DisproofStore private_disproofs;
  DisproofStore& disproof_store() { return opt.prove_only ? private_disproofs : *disproofs; }
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
    if (sh.budget->cancel != nullptr && sh.budget->cancel->load(std::memory_order_relaxed)) {
      st.stopped_by_cancel = true;  // the caller's stop (stage E10): a budget stop
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
  // before the present (T8), or any inactive timeline (the opponent's, or an
  // initial one, H46) a creation might reactivate (T11, T12). Over-approximate: sound for
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
    // Any inactive timeline, whoever made it - the opponent's, or an initial
    // one (H46: a variant's set-up reads every timeline as initial) - might be
    // reactivated by a creation (T11): over-approximate.
    if (!s.inactive_timelines().empty()) return true;
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

  // The turn's axes: the mover's playable boards. A's turns (and the
  // canonical completion): ascending timeline. D's: the boards a check rests
  // on first (`hot` of them), then the mandatory ones, then the rest.
  std::vector<std::int32_t> attacker_axes(const GameState& s) const {
    std::vector<std::int32_t> out;
    for (const int p : s.playable_timelines()) out.push_back(p);
    return out;
  }
  // `in_check`: R1's status at `s` (some capture in the passed multiverse).
  std::vector<std::int32_t> defender_axes(const GameState& s, const RoyalAttackMap& opp_map, int& hot,
                                          bool& in_check) {
    std::vector<std::int32_t> hot_axes;
    in_check = false;
    const std::vector<int> passed = passed_timelines(s);
    if (!passed.empty()) {
      const Multiverse pm = passed_multiverse(s);
      ++st.attack_queries;
      spend();
      const RoyalAttackMap m = opp_map.advanced(pm, s.inactive_timelines());
      in_check = m.any();
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

  // -- the last-turn check filter (t5d/prover.hpp) ---------------------------------------------

  // `mv` with every head of `colour`'s to move advanced by a pass (C6): every
  // board R1 could pass at the next turn start, and more (a superset of R1's
  // sources: more boards only extend lines, Q18's default).
  static Multiverse with_heads_passed(const Multiverse& mv, Colour colour) {
    Multiverse out = mv;
    for (const TimelineRef& t : mv.timelines()) {
      if (to_move_at(t->last_half_turn()) != colour) continue;
      BoardBuilder b(*t->head());
      b.set_en_passant(kNoSquare);
      out.append_board(t->position(), b.publish());
    }
    return out;
  }

  // The filter of the attacker's last turn at `s` (`all`: its moves there),
  // reordering `axes` so that the boards with a check-capable move come
  // first; null where it does not apply (attack maps recompute under the
  // rules, or a capture stands already: every action may give check).
  std::shared_ptr<const CheckFilter> make_check_filter(const GameState& s, const MoveList& all,
                                                       std::vector<std::int32_t>& axes) {
    if (!opt.check_filter || !attack_maps_incremental(s.rules())) return nullptr;
    const Colour me = s.to_move();
    const Colour them = opposite(me);
    if (!spend()) return nullptr;
    ++st.attack_queries;
    const Multiverse base = with_heads_passed(s.multiverse(), them);
    const RoyalAttackMap m0(base, s.rules(), me, s.inactive_timelines());
    if (m0.any()) return nullptr;
    // The places the attacker's action may fill: beyond the head of a
    // timeline it may play, or on a timeline it may create.
    const Multiverse& mv = s.multiverse();
    const auto may_fill = [&](int p, int h) {
      const Timeline* t = mv.timeline(p);
      if (t == nullptr) return true;
      return s.playable_by_mover(p) && h > t->last_half_turn();
    };
    std::vector<RoyalAttackMap::VoidStop> stops0;
    {
      std::vector<RoyalAttackMap::VoidStop> all_stops;
      m0.void_stops(all_stops);
      for (const auto& v : all_stops) {
        if (may_fill(v.timeline, v.half_turn)) stops0.push_back(v);
      }
      std::sort(stops0.begin(), stops0.end());
    }
    auto f = std::make_shared<CheckFilter>();
    f->moves.assign(all.begin(), all.end());
    f->capable_move.assign(all.size(), 0);
    f->by_void.assign(all.size(), 0);
    std::vector<RoyalAttackMap::VoidStop> stops;
    for (std::size_t i = 0; i < all.size(); ++i) {
      if (stopped) return nullptr;
      const Move& m = all[i];
      f->index.emplace(CheckFilter::key(m), i);
      GameState t = s;
      apply(t, m);
      if (!spend()) return nullptr;
      ++st.attack_queries;
      const RoyalAttackMap mm = m0.advanced(with_heads_passed(t.multiverse(), them), t.inactive_timelines());
      bool capable = mm.any();
      if (!capable) {
        stops.clear();
        mm.void_stops(stops);
        for (const auto& v : stops) {
          if (!may_fill(v.timeline, v.half_turn)) continue;
          if (std::binary_search(stops0.begin(), stops0.end(), v)) continue;
          capable = true;  // a new line (or a resumed one) stopped where a board may come
          f->by_void[i] = 1;
          break;
        }
      }
      f->capable_move[i] = capable ? 1 : 0;
    }
    // The boards with a check-capable non-branching move first.
    const auto axis_capable = [&](int p) {
      for (std::size_t i = 0; i < all.size(); ++i) {
        if (f->capable_move[i] != 0 && !all[i].branching() && all[i].from_timeline == p) return true;
      }
      return false;
    };
    std::stable_partition(axes.begin(), axes.end(), axis_capable);
    f->plain_from.assign(axes.size() + 1, 0);
    for (std::size_t j = axes.size(); j-- > 0;) {
      f->plain_from[j] = static_cast<std::uint8_t>(f->plain_from[j + 1] != 0 || axis_capable(axes[j]));
    }
    for (std::size_t i = 0; i < all.size(); ++i) {
      if (f->capable_move[i] != 0 && all[i].branching()) f->branch = true;
      // J6: a non-branching jump onto another of the attacker's playable
      // boards becomes a BRANCHING twin once that board has been played - a
      // move the turn start does not offer, landing on a new outermost
      // timeline whose lines were never measured. Its capability is unknown,
      // so a branch is always left to make while any such jump exists (the
      // review's twin position: cutting it was a false disproof).
      if (all[i].jump() && !all[i].branching()) f->branch = true;
    }
    return f;
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
    // Prove-only reads only its own search's (its working memory, never a
    // shared fact).
    if (sh.disproof_store().find(s, sh.cls(n.or_node)) >= remaining) {
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

  void export_solved(std::uint32_t index, const GameState& s) {
    const Node& n = tree[index];
    if (n.kind != Kind::Turn) return;
    if (n.proven()) {
      if (n.dist < 0) throw std::logic_error("prover: a turn proven with no legal action under it");
      sh.put_proof(s, n.or_node ? Proof::win(n.dist) : Proof::loss(n.dist));
      ++st.proofs_exported;
      if (opt.record_certificate && n.or_node && n.expanded) {
        GameState t = s;
        std::vector<Move> action;
        if (next_turn(index, t, action) >= 0) {
          std::lock_guard<std::mutex> lock(sh.cert_mutex);
          sh.cert.add(s, std::move(action), n.dist);
        }
      }
    } else if (n.disproven() && n.remaining > 0 && (opt.record_all_disproofs || n.edge == Edge::Root)) {
      sh.disproof_store().put(s, sh.cls(n.or_node), n.remaining);
      if (!opt.prove_only) ++st.disproofs_recorded;
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
    // E4's question first: a quick search for one legal action, then E4's
    // exhaustive detector where it finds none - an attacker that is mated or
    // stalemated has no move to mate with, decided at once instead of by
    // exhausting its semimove tree (the review's finding: on a 32-board
    // position E4 decided in 221 nodes what the tree had not in 2M).
    if (opt.attacker_probe) {
      ++st.attacker_tests;
      const TurnShape probe = shape_of(s, attacker_axes(s), 0, all);
      std::vector<Move> path;
      std::uint64_t steps = 0;
      bool capped = false;
      Found found;
      const bool any = first_action(s, ctx.opp_map, probe, 0, Phase::Plain, Prefer::Quiet, path, steps,
                                    opt.escape_probe_nodes, capped, found);
      if (stopped) return;
      if (!any) {
        const MateResult r = detect_mate(s, Detector::Hypercuboid, detector_budget());
        if (!account(r.stats)) return;
        if (r.verdict == Outcome::Checkmate || r.verdict == Outcome::Stalemate) {
          // Exhaustive (H37): the attacker, to move, has no legal action.
          r.verdict == Outcome::Checkmate ? ++st.checkmates : ++st.stalemates;
          sh.put_proof(s, r.verdict == Outcome::Checkmate ? Proof::loss(0) : Proof::draw());
          ++st.proofs_exported;
          set_disproven(tree[n]);
          return;
        }
        if (r.verdict == Outcome::Ongoing && split && !capped) {
          throw std::logic_error("prover: the attacker's semimove tree holds no legal action where E4 found one");
        }
        // A legal action exists, or E4 ran out of memory: expand as usual.
      }
    }
    std::vector<std::int32_t> axes = attacker_axes(s);
    std::shared_ptr<const CheckFilter> filter;
    if (remaining == 1 && opt.completion == CompletionClass::Every && split) {
      filter = make_check_filter(s, all, axes);
      if (stopped) return;
    }
    tree[n].shape = shape_of(s, std::move(axes), 0, all);
    tree[n].shape.filter = filter;
    if (opt.completion == CompletionClass::Every && !split) {
      list_children(n, s, true);
      return;
    }
    if (opt.completion == CompletionClass::Every && opt.seeds == SeedClass::All) {
      // The literal class: the semimove tree, entered once (every legal
      // action once, as the defender's).
      const std::uint32_t c = make_semi(true, remaining, Edge::Enter, Phase::Plain, 0, std::nullopt);
      tree[n].children.push_back(c);
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
      tree[c].check_made = filter != nullptr && filter->capable(m);
      tree[n].children.push_back(c);
    }
  }

  void expand_defender(std::uint32_t n, const GameState& s, TurnCtx& ctx, const MoveList& all,
                       int remaining) {
    const bool split = decomposes(s.rules());
    int hot = 0;
    bool in_check = false;
    std::vector<std::int32_t> axes = defender_axes(s, ctx.opp_map, hot, in_check);
    tree[n].shape = shape_of(s, std::move(axes), hot, all);
    if (remaining == 0 && !in_check) {
      // At the horizon only checkmate counts, and checkmate needs check
      // (R1): not mated, whatever the defender's actions.
      ++st.not_in_check;
      set_disproven(tree[n]);
      return;
    }
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
    // The attacker's last turn: no check made and none left to make - no
    // completion can mate (no child: an OR-side node disproven).
    const CheckFilter* filter = or_node ? shape.filter.get() : nullptr;
    const bool check_made = tree[n].check_made;
    if (filter != nullptr && !check_made) {
      const bool left = phase == Phase::Plain ? (filter->plain_from[j] != 0 || filter->branch) : filter->branch;
      if (!left) {
        ++st.check_cut;
        return;
      }
    }
    const auto made = [&](const Move& m) { return check_made || (filter != nullptr && filter->capable(m)); };
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
          tree[c].check_made = check_made;
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
        tree[c].check_made = made(m);
        tree[n].children.push_back(c);
      }
      return;
    }
    // The branch phase.
    tree[n].phase = Phase::Branch;
    if (t.obligations_met() && !c4.any()) {
      if (filter != nullptr && !check_made) {
        ++st.check_cut;  // submitting now leaves the defender out of check
      } else {
        GameState next = t;
        next.submit(CheckRuling::none());  // legal: C4 holds, the obligations met
        const std::uint32_t c = make_turn(next, remaining - 1, Edge::Submit, {});
        tree[n].children.push_back(c);
      }
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
      tree[c].check_made = made(b);
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
    std::uint64_t h = (static_cast<std::uint64_t>(child) + 1) * std::uint64_t{0x9E3779B97F4A7C15} ^
                      static_cast<std::uint64_t>(id) * std::uint64_t{0xC2B2AE3D27D4EB4F};
    h ^= h >> 31;
    h *= std::uint64_t{0xBF58476D1CE4E5B9};
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
        export_solved(n, s);
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
      // Prove-only: thresholds on proof numbers only - a subtree is left when
      // its proof number outgrows its share, never kept to finish a disproof.
      if (opt.prove_only && opt.prove_only_pn_thresholds) cthdn = kInf;
      descend(n, pick, s, turn, c4, cthpn, cthdn);
      if (stopped) return;
    }
    Node& node = tree[n];
    if (node.kind == Kind::Turn) {
      if (node.solved()) {
        export_solved(n, s);
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

  // -- the proof, read back from the tree --------------------------------------------------------

  // The state below an edge from `s` (the edge into `c`).
  static void cross(GameState& s, const Node& c) {
    switch (c.edge) {
      case Edge::Move:
        s.apply(c.moves.front());
        return;
      case Edge::Action:
        for (const Move& m : c.moves) s.apply(m);
        s.submit(CheckRuling::none());
        return;
      case Edge::Submit:
        s.submit(CheckRuling::none());
        return;
      case Edge::Pass:
      case Edge::Enter:
      case Edge::Root:
        return;
    }
  }

  // From the proven turn node `n` (state `s`) to the next turn node along the
  // proof: the attacker's (OR) proving child, or the defender's (AND) longest
  // defence. The action's moves into `action`, the state into `s`; the next
  // turn node, or -1 where the tree holds no continuation.
  long next_turn(std::uint32_t n, GameState& s, std::vector<Move>& action) const {
    action.clear();
    std::uint32_t m = n;
    for (int guard = 0; guard < 100000; ++guard) {
      const Node& node = tree[m];
      if (!node.expanded) return -1;
      std::uint32_t pick = 0;
      bool found = false;
      int best = 0;
      for (const std::uint32_t ci : node.children) {
        const Node& c = tree[ci];
        if (!c.proven() || c.dist < 0) continue;
        const int d = contribution(c);
        if (!found || (node.or_node ? d < best : d > best)) {
          best = d;
          pick = ci;
          found = true;
        }
      }
      if (!found) return -1;
      const Node& c = tree[pick];
      if (c.edge == Edge::Move) action.push_back(c.moves.front());
      if (c.edge == Edge::Action) action = c.moves;
      cross(s, c);
      if (c.kind == Kind::Turn) return static_cast<long>(pick);
      m = pick;
    }
    return -1;
  }

  // The proof's line from the proven root (`root`, state `s`): actions to the
  // mate. Where a node was solved from a store, an attacker's position is
  // looked up in `cert` (if any) and the line stops at a defender's.
  void proof_line(std::uint32_t root, GameState s, const ProofCertificate* cert,
                  std::vector<std::vector<Move>>& out, bool& complete) const {
    out.clear();
    complete = false;
    long n = static_cast<long>(root);
    bool attacker_turn = true;
    for (int guard = 0; guard < 256; ++guard) {
      const Node* node = n >= 0 ? &tree[static_cast<std::size_t>(n)] : nullptr;
      if (node != nullptr && !node->or_node && node->proven() && node->dist == 0) {
        complete = true;  // the defender is mated
        return;
      }
      std::vector<Move> action;
      if (node != nullptr && node->expanded) {
        n = next_turn(static_cast<std::uint32_t>(n), s, action);
        if (n < 0) return;  // the tree holds no continuation
        out.push_back(std::move(action));
        attacker_turn = !attacker_turn;
        continue;
      }
      // Solved from a store: an attacker's position may be in the certificate.
      if (!attacker_turn || cert == nullptr) return;
      const CertificateStep* step = cert->find(s);
      if (step == nullptr) return;
      for (const Move& m : step->action) s.apply(m);
      s.submit(CheckRuling::none());
      out.push_back(step->action);
      attacker_turn = false;
      n = -1;
      // The defender's turn after it is not in this tree: the line stops
      // unless that turn is a checkmate.
      const MateResult r = detect_mate(s, Detector::Hypercuboid, SearchBudget::nodes(200000));
      if (r.verdict == Outcome::Checkmate) complete = true;
      return;
    }
  }

  // Every action under node `n`, every node expanded by the prover's own
  // expansion (tests and tools, no budget): the moves from the turn start,
  // and the successor; `paths` counts them before any is dropped as a
  // duplicate.
  void enumerate(std::uint32_t n, const GameState& s, TurnCtx& turn, const RoyalAttackMap& c4,
                 std::vector<Move>& path, std::vector<BuiltAction>& out) {
    if (!tree[n].expanded) {
      if (tree[n].kind == Kind::Turn) {
        expand_turn(n, s, turn);
      } else {
        expand_semi(n, s, turn, c4);
      }
    }
    const std::vector<std::uint32_t> children = tree[n].children;  // the tree grows below
    const bool at_turn = tree[n].kind == Kind::Turn;
    for (const std::uint32_t ci : children) {
      const Edge edge = tree[ci].edge;
      const std::vector<Move> moves = tree[ci].moves;
      switch (edge) {
        case Edge::Pass:
          enumerate(ci, s, turn, c4, path, out);
          break;
        case Edge::Enter:
          enumerate(ci, s, turn, turn.opp_map, path, out);
          break;
        case Edge::Move: {
          GameState t = s;
          apply(t, moves.front());
          const RoyalAttackMap next = at_turn ? advance(turn.opp_map, t) : advance(c4, t);
          path.push_back(moves.front());
          enumerate(ci, t, turn, next, path, out);
          path.pop_back();
          break;
        }
        case Edge::Submit: {
          GameState t = s;
          t.submit(CheckRuling::none());
          out.push_back(BuiltAction{path, std::move(t)});
          break;
        }
        case Edge::Action: {
          GameState t = s;
          for (const Move& m : moves) t.apply(m);
          t.submit(CheckRuling::none());
          out.push_back(BuiltAction{moves, std::move(t)});
          break;
        }
        case Edge::Root:
          break;
      }
    }
  }

  // The tree's actions at `turn_start` for a side (tree_actions).
  std::vector<BuiltAction> actions_of(const GameState& turn_start) {
    tree.clear();
    st = ProverStats{};
    stopped = false;
    const std::uint32_t r = make_turn(turn_start, 1, Edge::Root, {});
    tree[r].pn = 1;  // never read from a store: expanded, whatever is stored
    tree[r].dn = 1;
    TurnCtx ctx;
    ctx.start = &turn_start;
    ctx.node = r;
    ctx.mover_map = RoyalAttackMap(turn_start.multiverse(), turn_start.rules(), turn_start.to_move(),
                                   turn_start.inactive_timelines());
    ctx.opp_map = RoyalAttackMap(turn_start.multiverse(), turn_start.rules(), opposite(turn_start.to_move()),
                                 turn_start.inactive_timelines());
    std::vector<Move> path;
    std::vector<BuiltAction> out;
    enumerate(r, turn_start, ctx, ctx.opp_map, path, out);
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
    const std::uint32_t r = make_turn(root, sh.depth, Edge::Root, {});
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
  into.check_cut += x.check_cut;
  into.not_in_check += x.not_in_check;
  into.attacker_tests += x.attacker_tests;
  into.stopped_by_nodes = into.stopped_by_nodes || x.stopped_by_nodes;
  into.stopped_by_time = into.stopped_by_time || x.stopped_by_time;
  into.stopped_by_cancel = into.stopped_by_cancel || x.stopped_by_cancel;
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

  // A proven root's first action and line, from a thread that proved it at the
  // least distance - one whose tree holds the whole line where any does. Under
  // Lazy SMP a thread can decide the root without searching it: the first
  // thread to prove the root exports the proof at once (H19), and a thread
  // that builds its root after that reads the proof from the store, with no
  // tree beneath it - thread 0 most often, which builds its root after
  // starting the helpers. Taking the lowest-numbered thread at the least
  // distance reported such a thread's empty line and no first action (Guard's
  // five-thread test, CI run 37146667584). So the threads that searched their
  // root come first, in thread order, the first complete line ends the
  // choice, and otherwise the longest is kept; of the threads that read the
  // root from the store, one is asked (none has a tree, so each line is the
  // certificate's, if any). The verdict and the distance are not touched.
  void report_proof(const std::vector<std::uint32_t>& roots, const std::vector<char>& decided,
                    const GameState& root, const ProofCertificate* cert, ProveResult& result) const {
    const auto at_the_least_distance = [&](std::size_t i) {
      if (decided[i] == 0) return false;
      const Node& r = workers[i]->tree[roots[i]];
      return r.proven() && r.dist == result.proof.distance;
    };
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < workers.size(); ++i) {
      if (at_the_least_distance(i) && workers[i]->tree[roots[i]].expanded) order.push_back(i);
    }
    for (std::size_t i = 0; i < workers.size(); ++i) {
      if (at_the_least_distance(i) && !workers[i]->tree[roots[i]].expanded) {
        order.push_back(i);
        break;
      }
    }
    bool have = false;
    std::size_t chosen = 0;
    for (const std::size_t i : order) {
      std::vector<std::vector<Move>> line;
      bool complete = false;
      workers[i]->proof_line(roots[i], root, cert, line, complete);
      if (!have || complete || line.size() > result.line.size()) {
        result.line = std::move(line);
        result.line_complete = complete;
        chosen = i;
        have = true;
      }
      if (complete) break;
    }
    result.first_action = workers[chosen]->first_action_of_proof(roots[chosen]);
    // A root read from the store has no tree: its first action is the line's
    // (the certificate's), if any.
    if (result.first_action.empty() && !result.line.empty()) result.first_action = result.line.front();
  }

  ProveResult prove(const GameState& root, const ProveBudget& budget, int depth) {
    Shared& sh = shared;
    sh.budget = &budget;
    sh.depth = std::max(1, std::min(depth, 100));
    sh.private_disproofs.clear();
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
    result.depth = sh.depth;
    result.threads = static_cast<int>(workers.size());
    {
      const Worker& main = *workers.front();
      if (roots.front() < main.tree.size() && main.tree[roots.front()].expanded) {
        const Node& r = main.tree[roots.front()];
        result.root_actions = static_cast<int>(r.children.size());
        for (const std::uint32_t ci : r.children) {
          const Node& c = main.tree[ci];
          result.root_children.push_back(ProveResult::RootChild{c.moves, c.pn, c.dn});
        }
      }
    }
    for (std::size_t i = 0; i < workers.size(); ++i) accumulate(result.stats, workers[i]->st);
    // The root's verdict: any thread's (they agree, or the stores would have
    // refused a contradiction). A proof's distance is the least any thread
    // proved.
    bool proven = false;
    bool disproven = false;
    for (std::size_t i = 0; i < workers.size(); ++i) {
      if (!decided[i]) continue;
      const Node& r = workers[i]->tree[roots[i]];
      if (r.proven()) {
        if (!proven || r.dist < result.proof.distance) result.proof = Proof::win(r.dist);
        proven = true;
      }
      if (r.disproven()) disproven = true;
    }
    if (proven && disproven) {
      throw std::logic_error("prover: the threads proved and disproved the root");
    }
    if (proven) {
      std::shared_ptr<ProofCertificate> cert;
      if (sh.opt.record_certificate) {
        std::lock_guard<std::mutex> lock(sh.cert_mutex);
        cert = std::make_shared<ProofCertificate>(sh.cert);
        result.certificate = cert;
      }
      report_proof(roots, decided, root, cert.get(), result);
    }
    result.status = proven ? ProofStatus::Proven : disproven ? ProofStatus::Disproven : ProofStatus::Unknown;
    if (sh.opt.prove_only && result.status == ProofStatus::Disproven) {
      // Prove-only claims nothing but a proof.
      result.status = ProofStatus::Unknown;
      result.exhausted = true;
    }
    result.stats.seconds = seconds_since(sh.start);
    sh.budget = nullptr;
    return result;
  }
};

Prover::Prover(const ProverOptions& options, SolvedStore* proofs, DisproofStore* disproofs)
    : impl_(std::make_unique<Impl>(options, proofs, disproofs)) {}
Prover::~Prover() = default;

ProveResult Prover::prove(const GameState& root, const ProveBudget& budget) {
  return impl_->prove(root, budget, impl_->shared.opt.depth);
}

ProveResult Prover::prove(const GameState& root, const ProveBudget& budget, int depth) {
  return impl_->prove(root, budget, depth);
}

void Prover::clear() { impl_->shared.tt.clear(); }

const ProverOptions& Prover::options() const noexcept { return impl_->shared.opt; }
SolvedStore& Prover::proofs() noexcept { return *impl_->shared.proofs; }
DisproofStore& Prover::disproofs() noexcept { return *impl_->shared.disproofs; }

std::vector<BuiltAction> Prover::attacker_actions(const GameState& turn_start) {
  return tree_actions(turn_start, true).actions;
}

TreeActions Prover::tree_actions(const GameState& turn_start, bool attacker) {
  Shared& sh = impl_->shared;
  Worker& w = *impl_->workers.front();
  sh.budget = nullptr;
  sh.depth = 1;
  sh.private_disproofs.clear();
  sh.attacker = attacker ? turn_start.to_move() : opposite(turn_start.to_move());
  TreeActions out;
  std::vector<BuiltAction> all = w.actions_of(turn_start);
  out.paths = all.size();
  for (BuiltAction& a : all) {
    bool fresh = true;
    for (std::size_t i = 0; i < out.actions.size(); ++i) {
      if (out.actions[i].successor.zobrist() == a.successor.zobrist() &&
          same_position(out.actions[i].successor, a.successor)) {
        out.repeated.emplace_back(out.actions[i].moves, a.moves);
        fresh = false;
        break;
      }
    }
    if (fresh) out.actions.push_back(std::move(a));
  }
  return out;
}

CheckFilterView Prover::check_filter_view(const GameState& turn_start) {
  Shared& sh = impl_->shared;
  Worker& w = *impl_->workers.front();
  sh.budget = nullptr;
  w.stopped = false;
  CheckFilterView v;
  generate_moves(turn_start, v.moves);
  std::vector<std::int32_t> axes = w.attacker_axes(turn_start);
  // Built whatever this prover's options say (the worker reads them).
  const bool was = sh.opt.check_filter;
  sh.opt.check_filter = true;
  const std::shared_ptr<const CheckFilter> f = w.make_check_filter(turn_start, v.moves, axes);
  sh.opt.check_filter = was;
  if (f == nullptr) return v;
  v.applies = true;
  v.capable.assign(f->capable_move.begin(), f->capable_move.end());
  v.by_void.assign(f->by_void.begin(), f->by_void.end());
  v.branch = f->branch;
  return v;
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

}  // namespace t5d
