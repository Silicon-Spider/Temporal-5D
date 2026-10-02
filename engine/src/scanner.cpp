// The tactical scanner (see t5d/scanner.hpp): node-capped alpha-beta over
// forcing actions, a material eval, proofs and estimates kept apart. Rule IDs
// refer to the rulebook, docs/rules/README.md. Written from the design spec
// (§11.1-§11.3) and the rulebook; alpha-beta, MVV-LVA, killers, history,
// continuation history, Lazy SMP and the lockless transposition table (the
// key XORed with the data) are the textbook techniques (Chess Programming
// Wiki), written afresh - no engine's code was consulted.

#include "t5d/scanner.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <exception>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "t5d/board.hpp"
#include "t5d/check.hpp"
#include "t5d/movegen.hpp"
#include "t5d/multiverse.hpp"

namespace t5d {

// ---- proofs and the solved store ---------------------------------------------------

std::string to_string(const Proof& proof) {
  switch (proof.kind) {
    case Proof::Kind::None: return "none";
    case Proof::Kind::Win: return "win " + std::to_string(proof.distance);
    case Proof::Kind::Loss: return "loss " + std::to_string(proof.distance);
    case Proof::Kind::Draw: return "draw";
  }
  return "?";
}

std::optional<Proof> MemorySolvedStore::find(const GameState& position) const {
  const auto it = index_.find(position.zobrist());
  if (it == index_.end()) return std::nullopt;
  for (const std::size_t i : it->second) {
    if (same_position(entries_[i].position, position)) return entries_[i].proof;
  }
  return std::nullopt;
}

void MemorySolvedStore::put(const GameState& position, const Proof& proof) {
  if (!proof.proven()) return;
  std::vector<std::size_t>& slot = index_[position.zobrist()];
  for (const std::size_t i : slot) {
    Entry& e = entries_[i];
    if (!same_position(e.position, position)) continue;
    if (e.proof.kind != proof.kind) {
      throw std::logic_error("solved store: contradictory proofs for one position (" +
                             to_string(e.proof) + " and " + to_string(proof) +
                             "): a false proof");
    }
    if (proof.distance < e.proof.distance) e.proof.distance = proof.distance;
    return;
  }
  slot.push_back(entries_.size());
  entries_.push_back(Entry{position, proof});
}

void MemorySolvedStore::clear() noexcept {
  index_.clear();
  entries_.clear();
}

// The shards: a power of two, chosen by the key's top bits (the low bits
// pick a TT slot - any bits would do, the key is mixed).
struct ConcurrentSolvedStore::Shards {
  static constexpr std::size_t kCount = 64;
  struct Shard {
    mutable std::shared_mutex mutex;
    MemorySolvedStore store;
  };
  std::array<Shard, kCount> shard;
  Shard& of(const GameState& s) { return shard[static_cast<std::size_t>(s.zobrist() >> 58)]; }
  const Shard& of(const GameState& s) const {
    return shard[static_cast<std::size_t>(s.zobrist() >> 58)];
  }
};

ConcurrentSolvedStore::ConcurrentSolvedStore() : shards_(std::make_unique<Shards>()) {}
ConcurrentSolvedStore::~ConcurrentSolvedStore() = default;

std::optional<Proof> ConcurrentSolvedStore::find(const GameState& position) const {
  const Shards::Shard& s = shards_->of(position);
  std::shared_lock<std::shared_mutex> lock(s.mutex);
  return s.store.find(position);
}

void ConcurrentSolvedStore::put(const GameState& position, const Proof& proof) {
  Shards::Shard& s = shards_->of(position);
  std::unique_lock<std::shared_mutex> lock(s.mutex);
  s.store.put(position, proof);  // a contradiction throws, the lock released
}

std::size_t ConcurrentSolvedStore::size() const {
  std::size_t n = 0;
  for (const Shards::Shard& s : shards_->shard) {
    std::shared_lock<std::shared_mutex> lock(s.mutex);
    n += s.store.size();
  }
  return n;
}

void ConcurrentSolvedStore::clear() {
  for (Shards::Shard& s : shards_->shard) {
    std::unique_lock<std::shared_mutex> lock(s.mutex);
    s.store.clear();
  }
}

// ---- the scanner -------------------------------------------------------------------------

namespace {

constexpr int kInfinity = kScanMate + 1;
// Scores beyond this are mate scores (a mate within 1000 half-turns).
constexpr int kMateBound = kScanMate - 1000;

enum class Bound : std::uint8_t { Upper = 0, Lower, Exact };

// The ordering key of a move: its piece, squares and displacement - the same
// move on another timeline or turn shares a key (history generalises along
// the 4D lattice).
std::uint32_t move_key(const Move& m) noexcept {
  std::uint64_t k = static_cast<std::uint64_t>(m.piece.as_moved().code());
  k = k * 131 + static_cast<std::uint64_t>(static_cast<std::uint8_t>(m.from_square));
  k = k * 131 + static_cast<std::uint64_t>(static_cast<std::uint8_t>(m.to_square));
  k = k * 131 + static_cast<std::uint64_t>(static_cast<std::uint32_t>(m.to_timeline - m.from_timeline) & 0xFFu);
  k = k * 131 + static_cast<std::uint64_t>(static_cast<std::uint32_t>(m.to_half_turn - m.from_half_turn) & 0xFFu);
  k = k * 131 + static_cast<std::uint64_t>(m.flags & (Move::kCapture | Move::kBranching));
  k ^= k >> 29;
  k *= 0xBF58476D1CE4E5B9ull;
  k ^= k >> 32;
  const std::uint32_t key = static_cast<std::uint32_t>(k);
  return key == 0 ? 1u : key;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int score_of(const Proof& p, int ply) noexcept {
  switch (p.kind) {
    case Proof::Kind::Win: return kScanMate - (ply + p.distance);
    case Proof::Kind::Loss: return -(kScanMate - (ply + p.distance));
    default: return 0;
  }
}

// Mate scores are stored relative to the node (the usual adjustment).
int to_tt(int score, int ply) noexcept {
  if (score > kMateBound) return score + ply;
  if (score < -kMateBound) return score - ply;
  return score;
}
int from_tt(int score, int ply) noexcept {
  if (score > kMateBound) return score - ply;
  if (score < -kMateBound) return score + ply;
  return score;
}

// ---- what the threads of a scan share (Lazy SMP, H17) -------------------------------------

// One TT entry: an estimate and an ordering hint, never a proof (H19).
struct TTEntry {
  std::int32_t score = 0;
  std::int16_t depth = -1;
  Bound bound = Bound::Upper;
  bool horizon = false;  // the subtree met the depth limit
  bool capped = false;   // the subtree held a capped node: not fully resolved
  std::uint32_t best = 0;
};

// The transposition table, keyed by GameState::zobrist() - the 5D key with
// the ruleset ID (H18, §11.3's non-cancelling mix). Lockless: a slot holds
// its entry packed in one word and, beside it, the key XORed with that word,
// so a slot another thread is writing reads as a miss - never as another
// position's entry. The best move's key, an ordering hint, sits in a third
// word: a torn hint only reorders moves. A fixed replacement policy (always
// replace); with one thread, a plain deterministic table (H15).
class SharedTT {
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
      slots_[i].best.store(0, std::memory_order_relaxed);
    }
  }
  bool probe(std::uint64_t key, TTEntry& out) const {
    const Slot& s = slots_[static_cast<std::size_t>(key) & mask_];
    const std::uint64_t data = s.data.load(std::memory_order_relaxed);
    const std::uint64_t check = s.check.load(std::memory_order_relaxed);
    if ((data & kValid) == 0 || (check ^ data) != key) return false;
    out.score = static_cast<std::int32_t>(static_cast<std::uint32_t>(data & 0xFFFFFFFFu));
    out.depth = static_cast<std::int16_t>(static_cast<std::uint16_t>((data >> 32) & 0xFFFFu));
    out.bound = static_cast<Bound>((data >> 48) & 3u);
    out.horizon = ((data >> 50) & 1u) != 0;
    out.capped = ((data >> 51) & 1u) != 0;
    out.best = s.best.load(std::memory_order_relaxed);
    return true;
  }
  void save(std::uint64_t key, const TTEntry& e) {
    std::uint64_t data = static_cast<std::uint32_t>(e.score);
    data |= static_cast<std::uint64_t>(static_cast<std::uint16_t>(e.depth)) << 32;
    data |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(e.bound)) << 48;
    if (e.horizon) data |= std::uint64_t{1} << 50;
    if (e.capped) data |= std::uint64_t{1} << 51;
    data |= kValid;
    Slot& s = slots_[static_cast<std::size_t>(key) & mask_];
    s.data.store(data, std::memory_order_relaxed);
    s.check.store(key ^ data, std::memory_order_relaxed);
    s.best.store(e.best, std::memory_order_relaxed);
  }

 private:
  static constexpr std::uint64_t kValid = std::uint64_t{1} << 52;
  struct Slot {
    std::atomic<std::uint64_t> check{0};
    std::atomic<std::uint64_t> data{0};
    std::atomic<std::uint32_t> best{0};
  };
  std::unique_ptr<Slot[]> slots_;
  std::size_t mask_ = 0;
};

// The state every thread of a scanner shares: the options, the solved store,
// the TT; during a scan, its budget, clock, node count and stop signal.
struct Shared {
  Shared(const ScannerOptions& o, SolvedStore* s) : opt(o), store(s != nullptr ? s : &own) {
    opt.threads = std::max(1, opt.threads);
    tt.resize(opt.tt_entries);
    lock_store = opt.threads > 1 && !store->concurrent();
  }

  ScannerOptions opt;
  ConcurrentSolvedStore own;
  SolvedStore* store;
  bool lock_store = false;  // a store not safe for threads, shared by several
  std::mutex store_mutex;
  SharedTT tt;

  // The scan in progress.
  const ScanBudget* budget = nullptr;  // the caller's
  std::chrono::steady_clock::time_point start;
  std::size_t threads = 1;
  std::atomic<std::uint64_t> nodes{0};  // flushed by the threads (more than one)
  std::atomic<bool> stop{false};        // a budget spent, or the root decided

  std::optional<Proof> find(const GameState& s) {
    if (!lock_store) return store->find(s);
    std::lock_guard<std::mutex> lock(store_mutex);
    return store->find(s);
  }
  void put(const GameState& s, const Proof& p) {
    if (!lock_store) {
      store->put(s, p);
      return;
    }
    std::lock_guard<std::mutex> lock(store_mutex);
    store->put(s, p);
  }
};

// What one thread's iterative deepening found.
struct WorkerRun {
  std::vector<IterationStats> iterations;
  ScanStats stats;
  int resolved_depth = 0;
  int completed_depth = 0;
  bool exhausted = false;
  int score = 0;
  std::vector<Move> best_action;
};

// Sums `x` into `into` (Lazy SMP's threads; seconds are set by the caller).
void accumulate(ScanStats& into, const ScanStats& x) {
  into.nodes += x.nodes;
  into.positions += x.positions;
  into.applies += x.applies;
  into.scans += x.scans;
  into.detector_calls += x.detector_calls;
  into.detector_nodes += x.detector_nodes;
  into.evasion_nodes += x.evasion_nodes;
  into.fallback_nodes += x.fallback_nodes;
  into.horizon_mate_nodes += x.horizon_mate_nodes;
  into.stalemate_test_nodes += x.stalemate_test_nodes;
  into.evasion_visits += x.evasion_visits;
  into.evasion_fail_highs += x.evasion_fail_highs;
  into.evasion_lists += x.evasion_lists;
  into.evasion_listed += x.evasion_listed;
  into.horizon_mate_tests += x.horizon_mate_tests;
  into.horizon_mates += x.horizon_mates;
  into.horizon_mate_nodes_on_mates += x.horizon_mate_nodes_on_mates;
  into.probe_seeds += x.probe_seeds;
  into.probe_nodes += x.probe_nodes;
  into.probe_seconds += x.probe_seconds;
  into.probe_actions += x.probe_actions;
  into.probe_cutoffs += x.probe_cutoffs;
  into.horizon_probe_hits += x.horizon_probe_hits;
  into.evasion_seconds += x.evasion_seconds;
  into.horizon_mate_seconds += x.horizon_mate_seconds;
  into.fallback_seconds += x.fallback_seconds;
  into.tt_hits += x.tt_hits;
  into.solved_hits += x.solved_hits;
  into.proofs_exported += x.proofs_exported;
  into.completion_fallbacks += x.completion_fallbacks;
  into.capped_nodes += x.capped_nodes;
  into.measurement_nodes += x.measurement_nodes;
  into.measurement_seconds += x.measurement_seconds;
  into.stopped_by_nodes = into.stopped_by_nodes || x.stopped_by_nodes;
  into.stopped_by_time = into.stopped_by_time || x.stopped_by_time;
  into.stopped_by_memory = into.stopped_by_memory || x.stopped_by_memory;
}

// A thread's nodes reach the shared count in batches of this many (so the
// budget of a scan with several threads holds to within threads x this).
constexpr std::uint64_t kFlushNodes = 256;

// A copy of `s` that shares no timeline and no board with it: the same
// position, rebuilt board by board. Threads that search copies of one state
// share its timelines and boards by reference count, and contend on those
// counts at every copy and every move; a helper thread searches from its own
// copy instead. Where the state cannot be rebuilt from its boards alone
// (activity decided by play - Q1's consecutive rule, Q3, Q6 - or a state the
// constructor refuses), the result is `s` itself, shared. The boards'
// origins (Timeline::origin) are left out: no rule reads them (and neither
// same_position nor the key holds them).
GameState isolated_copy(const GameState& s) {
  try {
    const Multiverse& mv = s.multiverse();
    Multiverse copy(mv.board_size(), mv.even(), mv.ruleset());
    copy.set_to_move(mv.to_move());
    for (const TimelineRef& t : mv.timelines()) {  // ascending: each extends the range
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
    // Not rebuildable: share.
  }
  return s;
}

// One thread of a scan: its search, and the tables Lazy SMP keeps per
// thread (§11.3) - killers, history, continuation history - which carry over
// from scan to scan until Scanner::clear() (between games).
struct Worker {
  Worker(Shared& shared, int index) : sh(shared), opt(shared.opt), id(index) {
    history.assign(std::size_t{1} << 14, 0);
    conthist.assign(std::size_t{1} << 16, 0);
    lists.resize(2);  // complete()'s and branch_threat()'s: never reallocated under them
  }

  // One action a node may search: how to build it, and its ordering score.
  struct Candidate {
    Move seed;
    /// The action, once built (at once for an evasion from E4's listing).
    std::optional<BuiltAction> action;
    std::int64_t order = 0;
    std::uint32_t key = 0;
  };

  struct Node {
    int score = 0;
    Proof proof;
  };

  Shared& sh;
  const ScannerOptions& opt;
  int id = 0;  // 0: the main thread

  std::vector<std::array<std::uint32_t, 2>> killers;
  std::vector<std::int32_t> history;
  std::vector<std::int32_t> conthist;

  // The scan in progress.
  ScanStats* stats = nullptr;  // this thread's, during a scan
  ScanStats scratch_stats;     // for the tool functions, outside a scan
  std::uint64_t flushed = 0;   // of stats->nodes, added to sh.nodes
  bool stopped = false;
  bool charging = true;  // false: classify_leaves' measurement (not charged)
  // The iteration in progress.
  bool capped = false;
  bool horizon_met = false;
  int max_board_moves = 0;
  LeafStats leaves;
  std::vector<Move> root_best;

  // -- budget -------------------------------------------------------------------------------

  ScanStats& st() { return stats != nullptr ? *stats : scratch_stats; }

  // The scan's nodes so far: this thread's alone, or (several threads) the
  // shared count and this thread's not yet flushed to it.
  std::uint64_t used() const {
    if (sh.threads == 1) return stats->nodes;
    return sh.nodes.load(std::memory_order_relaxed) + (stats->nodes - flushed);
  }
  void flush(bool all = false) {
    if (sh.threads == 1 || stats == nullptr) return;
    const std::uint64_t mine = stats->nodes - flushed;
    if (mine == 0 || (!all && mine < kFlushNodes)) return;
    sh.nodes.fetch_add(mine, std::memory_order_relaxed);
    flushed = stats->nodes;
  }
  // Stops this thread - and, a budget being spent, every thread.
  void halt() {
    stopped = true;
    if (sh.threads > 1) sh.stop.store(true, std::memory_order_relaxed);
  }

  bool spend(std::uint64_t n = 1) {
    if (!charging) {
      st().measurement_nodes += n;
      return true;
    }
    if (stopped) return false;
    ScanStats& s = st();
    if (stats != nullptr) {
      if (sh.threads > 1 && sh.stop.load(std::memory_order_relaxed)) {
        stopped = true;  // another thread spent the budget, or decided the root
        return false;
      }
      if (used() + n > sh.budget->max_nodes) {
        s.stopped_by_nodes = true;
        halt();
        return false;
      }
    }
    s.nodes += n;
    flush();
    if (stats != nullptr && sh.budget->max_time && (s.nodes & 255u) < n &&
        std::chrono::steady_clock::now() - sh.start > *sh.budget->max_time) {
      s.stopped_by_time = true;
      halt();
      return false;
    }
    return true;
  }

  std::uint64_t remaining() const {
    if (stats == nullptr || !charging) return std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t u = used();
    return sh.budget->max_nodes > u ? sh.budget->max_nodes - u : 0;
  }

  SearchBudget detector_budget() const {
    SearchBudget b;
    b.max_nodes = remaining();
    b.max_bytes = opt.detector_max_bytes;
    if (stats != nullptr && charging && sh.budget->max_time) {
      const auto used_time = std::chrono::steady_clock::now() - sh.start;
      const auto left =
          *sh.budget->max_time - std::chrono::duration_cast<std::chrono::nanoseconds>(used_time);
      b.max_time = left.count() > 0 ? left : std::chrono::nanoseconds(0);
    }
    return b;
  }

  // Accounts an E4 search: its nodes are the scanner's. Returns false if it
  // stopped on the scanner's budget (nodes or time): the scan must stop. A
  // search stopped by its memory ceiling answers nothing, and the scan goes on.
  bool account(const SearchStats& s) {
    ScanStats& x = st();
    ++x.detector_calls;
    x.detector_nodes += s.nodes;
    if (!charging) {
      x.measurement_nodes += s.nodes;
      return true;
    }
    x.nodes += s.nodes;
    flush();
    if (s.stopped_by_memory) {
      x.stopped_by_memory = true;
      return true;
    }
    if (s.stopped_by_budget && stats != nullptr) {
      if (sh.budget->max_time && std::chrono::steady_clock::now() - sh.start > *sh.budget->max_time) {
        x.stopped_by_time = true;
      } else {
        x.stopped_by_nodes = true;
      }
      halt();
      return false;
    }
    return true;
  }

  void applied(GameState& s, const Move& m) {
    ++st().applies;
    s.apply(m);
  }

  // -- eval ------------------------------------------------------------------------------------

  int material(const GameState& s, Colour c) const {
    int total = 0;
    for (const TimelineRef& t : s.multiverse().timelines()) {
      const Board& b = *t->head();
      for (int k = 0; k < kPieceKindCount; ++k) {
        const int v = opt.values.value[static_cast<std::size_t>(k)];
        if (v == 0) continue;
        total += v * std::popcount(b.pieces(c, static_cast<PieceKind>(k)));
      }
    }
    return total;
  }

  int evaluate(const GameState& s) const {
    return material(s, s.to_move()) - material(s, opposite(s.to_move()));
  }

  int value(Piece p) const { return p.empty() ? 0 : opt.values.of(p.kind()); }

  // -- the forcing set ---------------------------------------------------------------------

  // The piece a move leaves on its destination.
  static Piece arrived(const Move& m) {
    return m.promotes() ? Piece(m.promotion, m.piece.colour()) : m.piece.as_moved();
  }

  // Whether a plain move within a board can be judged in place, over the
  // multiverse before the move: it creates one board, and the rules read the
  // history board it leaves behind as they read a playable one (the history
  // flag only names a capture branching, and landability reads it only under
  // Q8's and Q20's alternatives). Exact where it applies.
  static bool in_place(const GameState& s, const Move& m) {
    return m.physical() && !m.castle() && !m.en_passant() && s.rules().turn_zero_reachable &&
           s.rules().inactive_history_is_destination;
  }

  // The multiverse after `m` with every board `m` created advanced by a pass
  // (an unchanged copy with the mover to move again, no en-passant square):
  // the place every forcing test below reads.
  struct Passed {
    Multiverse mv;
    std::vector<int> inactive;
    std::array<ScanSource, 2> sources{};
    int count = 0;
    int new_timeline = 0;  // a branch's
  };

  std::optional<Passed> materialise(const GameState& s, const Move& m) {
    const Colour mover = s.to_move();
    GameState t = s;
    if (!spend()) return std::nullopt;
    applied(t, m);
    Passed out{t.multiverse(), t.inactive_timelines(), {}, 0, 0};
    std::array<int, 2> created{m.from_timeline, 0};
    int n = 1;
    if (!m.physical()) {
      out.new_timeline = mover == Colour::White ? out.mv.max_timeline() : out.mv.min_timeline();
      created[1] = m.branching() ? out.new_timeline : m.to_timeline;
      n = 2;
    }
    for (int i = 0; i < n; ++i) {
      const int p = created[static_cast<std::size_t>(i)];
      BoardBuilder pass(*out.mv.head(p));
      pass.set_en_passant(kNoSquare);
      out.mv.append_board(p, pass.publish());
      out.sources[static_cast<std::size_t>(i)] =
          ScanSource{p, out.mv.timeline(p)->last_half_turn(), out.mv.head(p)};
    }
    out.count = n;
    return out;
  }

  // (b) in place (see in_place).
  bool checks_in_place(const GameState& s, const Move& m) {
    BoardBuilder b(*s.multiverse().board_at(m.from_timeline, m.from_half_turn));
    b.remove(m.from_square);
    b.put(m.to_square, arrived(m));
    const BoardRef created = b.publish();
    const ScanSource src{m.from_timeline, m.from_half_turn + 2, created.get()};
    if (!spend()) return false;
    ++st().scans;
    return royal_capture_from(s.multiverse(), s.rules(), s.to_move(), s.inactive_timelines(),
                              std::span<const ScanSource>(&src, 1));
  }

  // (b): a piece of the mover's on one of the passed boards can capture a
  // royal of the opponent's.
  bool checks_passed(const GameState& s, const Passed& p) {
    if (!spend()) return false;
    ++st().scans;
    return royal_capture_from(
        p.mv, s.rules(), s.to_move(), p.inactive,
        std::span<const ScanSource>(p.sources.data(), static_cast<std::size_t>(p.count)));
  }

  // (c): a branch whose new timeline, passed, holds an immediate threat.
  bool threat_passed(const GameState& s, const Move& m, const Passed& p) {
    if (opt.branch_threat == BranchThreat::None || !m.branching()) return false;
    if (!spend()) return false;
    ++st().scans;
    MoveList& out = scratch_list(1);
    out.clear();
    append_moves_of(p.mv, s.rules(), s.to_move(), p.new_timeline, out, p.inactive);
    for (const Move& r : out) {
      if (!r.capture() || r.captured.empty()) continue;
      if (opt.branch_threat == BranchThreat::ArrivalCapture && r.from_square != m.to_square) continue;
      if (value(r.captured) >= opt.branch_threat_value) return true;
    }
    return false;
  }

  bool checks(const GameState& s, const Move& m) {
    if (in_place(s, m)) return checks_in_place(s, m);
    const std::optional<Passed> p = materialise(s, m);
    return p && checks_passed(s, *p);
  }

  // Whether `m` is forcing, and of which class (0 not forcing, 1 capture,
  // 2 check, 3 branch threat, 4 promotion).
  int forcing_class(const GameState& s, const Move& m) {
    if (m.capture()) return 1;
    const bool threat_test = m.branching() && opt.branch_threat != BranchThreat::None;
    if (in_place(s, m)) {
      if (opt.checks && checks_in_place(s, m)) return 2;
    } else if (opt.checks || threat_test) {
      const std::optional<Passed> p = materialise(s, m);
      if (!p) return 0;
      if (opt.checks && checks_passed(s, *p)) return 2;
      if (threat_test && threat_passed(s, m, *p)) return 3;
    }
    if (stopped) return 0;
    if (opt.promotions_forcing && m.promotes()) return 4;
    return 0;
  }

  std::vector<MoveList> lists;
  MoveList& scratch_list(std::size_t i) {
    if (lists.size() <= i) lists.resize(i + 1);
    return lists[i];
  }

  // -- completion -------------------------------------------------------------------------------

  // The completion rule's choice among a board's moves.
  std::size_t choose(const MoveList& moves) const {
    std::size_t best = 0;
    if (opt.completion == Completion::Greedy) {
      int top = -1;
      for (std::size_t i = 0; i < moves.size(); ++i) {
        const Move& m = moves[i];
        if (!m.capture()) continue;
        const int mvv = value(m.captured) * 64 - (m.piece.royal() ? 0 : value(m.piece) / 16);
        if (mvv > top) {
          top = mvv;
          best = i;
        }
      }
      if (top >= 0) return best;
    }
    // The quietest: a move within the board capturing nothing, by the least
    // valuable piece (royals last), in generator order.
    long rank = -1;
    for (std::size_t i = 0; i < moves.size(); ++i) {
      const Move& m = moves[i];
      long r = (m.piece.royal() ? 100000L : static_cast<long>(value(m.piece)));
      if (!m.physical()) r += 1000000L;
      if (m.capture()) r += 2000000L;
      if (rank < 0 || r < rank) {
        rank = r;
        best = i;
      }
    }
    return best;
  }

  // The completion rule's own moves from `t` (an action under way, `moves` so
  // far): on each mandatory board left, choose()'s move; then C4 (R1). The
  // action, submitted, if that is legal - built through the state machine
  // and tested, so it is a legal action; nullopt if it is not, if a board
  // cannot be played from itself, or if the scan stopped.
  std::optional<BuiltAction> finish_by_rule(GameState t, std::vector<Move> moves) {
    MoveList& board_moves = scratch_list(0);
    while (!t.obligations_met()) {
      if (!t.may_move()) return std::nullopt;
      const std::vector<int> mandatory = t.mandatory_timelines();
      board_moves.clear();
      append_board_moves(t, mandatory.front(), board_moves);
      if (board_moves.empty()) return std::nullopt;  // not playable from itself
      const Move m = board_moves[choose(board_moves)];
      if (!spend()) return std::nullopt;
      applied(t, m);
      moves.push_back(m);
    }
    if (!spend()) return std::nullopt;
    ++st().scans;
    if (!royals_safe(t)) return std::nullopt;  // C4, R1
    t.submit(CheckRuling::none());             // legal: tested just above
    return BuiltAction{std::move(moves), std::move(t)};
  }

  // The legal action `seed` and the completion rule's moves make, with no
  // fallback (the in-check stage: a seed that is no evasion costs a few
  // applied moves and one scan).
  std::optional<BuiltAction> quick_action(const GameState& start_state, const Move& seed) {
    ScanStats& x = st();
    ++x.probe_seeds;
    const std::uint64_t n0 = x.nodes;
    const auto t0 = std::chrono::steady_clock::now();
    std::optional<BuiltAction> a;
    GameState t = start_state;
    if (spend()) {
      applied(t, seed);
      a = finish_by_rule(std::move(t), std::vector<Move>{seed});
    }
    x.probe_nodes += x.nodes - n0;
    x.probe_seconds += seconds_since(t0);
    return a;
  }

  std::optional<BuiltAction> complete(const GameState& start_state, const Move& seed) {
    GameState after_seed = start_state;
    if (!spend()) return std::nullopt;
    applied(after_seed, seed);
    if (opt.completion != Completion::Detector) {
      if (std::optional<BuiltAction> a = finish_by_rule(after_seed, std::vector<Move>{seed})) return a;
      if (stopped) return std::nullopt;
      ++st().completion_fallbacks;
    }
    // E4's first legal finish from the state after the seed, within the
    // fallback's own cap: a seed it cannot finish in time is dropped.
    SearchBudget b = detector_budget();
    const bool local_cap = b.max_nodes > opt.fallback_max_nodes;
    if (local_cap) b.max_nodes = opt.fallback_max_nodes;
    const auto t0 = std::chrono::steady_clock::now();
    const ActionList l = list_legal_actions(after_seed, 1, Detector::Hypercuboid, b);
    if (charging) {
      st().fallback_nodes += l.stats.nodes;
      st().fallback_seconds += seconds_since(t0);
    }
    if (local_cap && l.stats.stopped_by_budget && !l.stats.stopped_by_memory &&
        l.stats.nodes >= opt.fallback_max_nodes) {
      // Its own cap, not the scan's budget: charge the nodes, drop the seed.
      ScanStats& x = st();
      ++x.detector_calls;
      x.detector_nodes += l.stats.nodes;
      if (charging) {
        x.nodes += l.stats.nodes;
        flush();
      } else {
        x.measurement_nodes += l.stats.nodes;
      }
      return std::nullopt;
    }
    if (!account(l.stats) || l.actions.empty()) return std::nullopt;
    std::vector<Move> moves{seed};
    moves.insert(moves.end(), l.actions.front().moves.begin(), l.actions.front().moves.end());
    return BuiltAction{std::move(moves), l.actions.front().result};
  }

  // -- ordering ------------------------------------------------------------------------------------

  std::int64_t mvv_lva(const Move& m) const {
    return static_cast<std::int64_t>(value(m.captured)) * 64 -
           (m.piece.royal() ? 0 : static_cast<std::int64_t>(value(m.piece)) / 16);
  }

  // Lazy SMP's diversity (H17): a helper thread (id > 0) perturbs its order
  // of the quiet moves by a hash of the move and its id - deterministic per
  // thread, below a killer's weight and half the histories' range.
  std::int64_t jitter(std::uint32_t key) const {
    if (id == 0) return 0;
    std::uint64_t h = static_cast<std::uint64_t>(key) ^
                      (static_cast<std::uint64_t>(static_cast<unsigned>(id)) * 0x9E3779B97F4A7C15ull);
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 29;
    return static_cast<std::int64_t>(h & 0x7FFFFu);
  }

  std::int64_t quiet_order(std::uint32_t key, std::uint32_t prev, int ply) const {
    std::int64_t o = jitter(key);
    if (static_cast<std::size_t>(ply) < killers.size()) {
      if (killers[static_cast<std::size_t>(ply)][0] == key) o += 1'000'000;
      else if (killers[static_cast<std::size_t>(ply)][1] == key) o += 900'000;
    }
    o += history[key & (history.size() - 1)];
    o += conthist[(key ^ (prev * 0x9E3779B1u)) & (conthist.size() - 1)];
    return o;
  }

  void reward(std::uint32_t key, std::uint32_t prev, int ply, int depth, bool capture) {
    if (capture) return;  // MVV-LVA orders captures
    if (killers.size() <= static_cast<std::size_t>(ply)) killers.resize(static_cast<std::size_t>(ply) + 1, {0, 0});
    auto& k = killers[static_cast<std::size_t>(ply)];
    if (k[0] != key) {
      k[1] = k[0];
      k[0] = key;
    }
    const std::int32_t bonus = depth * depth;
    std::int32_t& h = history[key & (history.size() - 1)];
    h = std::min(h + bonus, 1 << 20);
    std::int32_t& c = conthist[(key ^ (prev * 0x9E3779B1u)) & (conthist.size() - 1)];
    c = std::min(c + bonus, 1 << 20);
  }

  // An action's representative move: its best capture (MVV-LVA), else its
  // first move.
  const Move& representative(const std::vector<Move>& moves) const {
    const Move* best = &moves.front();
    for (const Move& m : moves) {
      if (m.capture() && (!best->capture() || mvv_lva(m) > mvv_lva(*best))) best = &m;
    }
    return *best;
  }

  // -- TT ------------------------------------------------------------------------------------------

  void save(std::uint64_t key, int depth, int score, Bound bound, std::uint32_t best, bool horizon,
            bool was_capped, int ply) {
    TTEntry e;  // always replace: deterministic with one thread
    e.depth = static_cast<std::int16_t>(depth);
    e.score = to_tt(score, ply);
    e.bound = bound;
    e.best = best;
    e.horizon = horizon;
    e.capped = was_capped;
    sh.tt.save(key, e);
  }

  // -- proofs ------------------------------------------------------------------------------------

  void export_proof(const GameState& s, const Proof& p) {
    sh.put(s, p);  // H19: at once, never only in the TT
    ++st().proofs_exported;
  }

  // -- the search ----------------------------------------------------------------------------------

  // The triangular principal-line table: pv[ply] is the best line found from
  // the node at that ply, its actions in order.
  std::vector<std::vector<std::vector<Move>>> pv;

  std::vector<std::vector<Move>>& pv_at(int ply) {
    if (pv.size() <= static_cast<std::size_t>(ply) + 1) pv.resize(static_cast<std::size_t>(ply) + 2);
    return pv[static_cast<std::size_t>(ply)];
  }

  void note_pv(int ply, const std::vector<Move>& moves) {
    std::vector<std::vector<Move>>& here = pv_at(ply);
    const std::vector<std::vector<Move>>& below = pv[static_cast<std::size_t>(ply) + 1];
    here.assign(1, moves);
    here.insert(here.end(), below.begin(), below.end());
    if (ply == 0) root_best = moves;
  }

  bool quiet_now(const GameState& s) {
    // Measurement only: not charged, not counted in the nodes (nor in the
    // scan's tallies of applies and scans).
    const auto t0 = std::chrono::steady_clock::now();
    const bool was = charging;
    const std::uint64_t applies = st().applies;
    const std::uint64_t scans = st().scans;
    charging = false;
    bool quiet = !in_check(s);
    if (quiet) {
      MoveList moves;
      generate_moves(s, moves);
      for (const Move& m : moves) {
        if (forcing_class(s, m) != 0) {
          quiet = false;
          break;
        }
      }
    }
    charging = was;
    st().applies = applies;
    st().scans = scans;
    st().measurement_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return quiet;
  }

  Node search(const GameState& s, int depth, int alpha, int beta, int ply, std::uint32_t prev,
              int board_moves) {
    ++st().positions;
    max_board_moves = std::max(max_board_moves, board_moves);
    pv_at(ply).clear();
    if (const std::optional<Proof> p = sh.find(s)) {
      ++st().solved_hits;
      ++leaves.solved;
      return Node{score_of(*p, ply), *p};
    }
    const std::uint64_t key = s.zobrist();
    std::uint32_t tt_best = 0;
    TTEntry e;
    if (sh.tt.probe(key, e)) {
      tt_best = e.best;
      if (ply > 0 && e.depth >= depth) {
        const int sc = from_tt(e.score, ply);
        if (e.bound == Bound::Exact || (e.bound == Bound::Lower && sc >= beta) ||
            (e.bound == Bound::Upper && sc <= alpha)) {
          ++st().tt_hits;
          if (e.horizon) horizon_met = true;
          if (e.capped) capped = true;
          return Node{sc, Proof{}};  // an estimate: the TT never holds a proof
        }
      }
    }
    if (!spend()) return Node{};
    ++st().scans;
    // R1's check status (in_check), with one capture of a royal it rests on.
    const std::optional<RoyalCapture> witness = check_witness(s);
    return witness ? evasions(s, depth, alpha, beta, ply, prev, board_moves, key, tt_best, witness)
                   : forcing(s, depth, alpha, beta, ply, prev, board_moves, key, tt_best);
  }

  // A child's result, for this node.
  struct Tally {
    int best = -kInfinity;
    std::vector<Move> best_moves;
    std::uint32_t best_key = 0;
    Proof win;            // the shortest win found (some child is a proven loss)
    bool all_lost = true; // every child searched so far is a proven win for the opponent
    int longest_loss = 0;
    bool horizon = false;  // some child's subtree met the depth limit
    bool capped = false;   // ... or held a capped node, or this node is capped
  };

  // Searches one child; false: stop searching this node's children (the scan
  // stopped, a cutoff, or the fastest mate).
  bool child(const GameState& successor, const std::vector<Move>& moves, std::uint32_t key,
             int depth, int& alpha, int beta, int ply, int board_moves, Tally& t) {
    const bool horizon_before = horizon_met;
    const bool capped_before = capped;
    horizon_met = false;
    capped = false;
    const Node r = search(successor, depth - 1, -beta, -alpha, ply + 1, key,
                          board_moves + static_cast<int>(moves.size()));
    t.horizon = t.horizon || horizon_met;
    t.capped = t.capped || capped;
    horizon_met = horizon_met || horizon_before;
    capped = capped || capped_before;
    if (stopped) return false;
    const int score = -r.score;
    if (r.proof.kind == Proof::Kind::Loss) {
      if (!t.win.proven() || r.proof.distance + 1 < t.win.distance) {
        t.win = Proof::win(r.proof.distance + 1);
      }
    }
    if (r.proof.kind == Proof::Kind::Win) {
      t.longest_loss = std::max(t.longest_loss, r.proof.distance + 1);
    } else {
      t.all_lost = false;
    }
    if (score > t.best) {
      t.best = score;
      t.best_moves = moves;
      t.best_key = key;
      note_pv(ply, moves);
    }
    if (score > alpha) alpha = score;
    if (t.win.proven() && t.win.distance == 1) return false;  // the fastest mate there is
    return alpha < beta;
  }

  Node finish(const GameState& s, int depth, int alpha_orig, int beta, int ply, std::uint32_t prev,
              std::uint64_t key, Tally& t, bool all_searched, bool exhaustive) {
    Proof proof;
    if (t.win.proven()) {
      proof = t.win;  // ∃: one action leads to a proven loss of the opponent's
    } else if (exhaustive && all_searched && t.all_lost) {
      proof = Proof::loss(t.longest_loss);  // ∀: every legal action loses, proven
    }
    if (proof.proven()) {
      export_proof(s, proof);
      t.best = score_of(proof, ply);
    }
    const Bound bound = t.best <= alpha_orig ? Bound::Upper
                        : t.best >= beta      ? Bound::Lower
                                              : Bound::Exact;
    if (t.best >= beta && !t.best_moves.empty()) {
      reward(t.best_key, prev, ply, depth, representative(t.best_moves).capture());
    }
    save(key, depth, t.best, bound, t.best_key, t.horizon, t.capped, ply);
    if (t.horizon) horizon_met = true;
    if (t.capped) capped = true;
    return Node{t.best, proof};
  }

  // -- in check -------------------------------------------------------------------------------------

  // The order in which the in-check stage tries its seeds (net-free, H4):
  // the TT's move; a capture of the checking piece (`witness`, one capture of
  // a royal that R1's check status found); a move of a royal; a move onto a
  // square the check's line crosses (a block); any other capture (MVV-LVA);
  // the rest by killers and histories.
  std::vector<Candidate> evasion_seeds(const GameState& s, const std::optional<RoyalCapture>& witness,
                                       std::uint32_t tt_best, std::uint32_t prev, int ply) {
    MoveList moves;
    generate_moves(s, moves);
    std::vector<CapturePlace> places;
    if (witness) places = capture_places(*witness);
    std::vector<Candidate> out;
    out.reserve(moves.size());
    for (const Move& m : moves) {
      Candidate c;
      c.seed = m;
      c.key = move_key(m);
      // The board a move lands on, advanced by the pass R1's status reads.
      const auto lands_on = [&](const CapturePlace& p) {
        return m.to_timeline == p.timeline && m.to_half_turn + 1 == p.half_turn &&
               m.to_square == p.square;
      };
      std::int64_t cls = 0;
      if (m.capture() && !places.empty() && lands_on(places.front())) {
        cls = 5;  // takes the checking piece
      } else if (m.piece.royal()) {
        cls = 4;
      } else if (places.size() > 2 && std::any_of(places.begin() + 1, places.end() - 1, lands_on)) {
        cls = 3;  // blocks the line
      } else if (m.capture()) {
        cls = 2;
      } else {
        cls = 1;
      }
      c.order = (cls << 44) + (m.capture() ? mvv_lva(m) : quiet_order(c.key, prev, ply));
      if (c.key == tt_best) c.order += std::int64_t{1} << 50;
      out.push_back(c);
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const Candidate& a, const Candidate& b) { return a.order > b.order; });
    return out;
  }

  // In check (R1): every legal action (H2), in two stages.
  //  1. Up to evasion_probe seeds, in evasion_seeds' order, each made into a
  //     legal action by the completion rule (quick_action: the state machine
  //     and C4's test - a seed whose completion leaves a royal capturable is
  //     passed over), each searched as soon as it is built. A cutoff there
  //     ends the node: a legal action that scores at least beta is a lower
  //     bound whatever the other actions are, so the node is resolved and
  //     needs no enumeration (and an action leading to a proven loss of the
  //     opponent's is a Win proof, ∃).
  //  2. No cutoff: E4's enumeration, exhaustive or capped, as before; the
  //     successors not searched in stage 1 are searched. A Loss proof needs
  //     the enumeration exhaustive and every child - of both stages - a
  //     proven win of the opponent's (∀).
  // At the horizon (depth 0) the question is only "is this a mate?": one
  // legal action from stage 1's seeds answers no (an estimate leaf); only if
  // none of them completes legally does E4's exhaustive detector decide.
  Node evasions(const GameState& s, int depth, int alpha, int beta, int ply, std::uint32_t prev,
                int board_moves, std::uint64_t key, std::uint32_t tt_best,
                const std::optional<RoyalCapture>& witness) {
    std::vector<Candidate> seeds;
    if (opt.evasion_probe > 0) seeds = evasion_seeds(s, witness, tt_best, prev, ply);
    const std::size_t probe = std::min(seeds.size(), opt.evasion_probe);
    if (depth == 0) {
      horizon_met = true;
      ++st().horizon_mate_tests;
      for (std::size_t i = 0; i < probe; ++i) {
        const std::optional<BuiltAction> a = quick_action(s, seeds[i].seed);
        if (stopped) return Node{};
        if (a) {
          ++st().horizon_probe_hits;  // a legal action: no mate (C5)
          ++leaves.horizon_in_check;
          return Node{evaluate(s), Proof{}};
        }
      }
      const auto t0 = std::chrono::steady_clock::now();
      const MateResult r = detect_mate(s, Detector::Hypercuboid, detector_budget());
      st().horizon_mate_nodes += r.stats.nodes;
      st().horizon_mate_seconds += seconds_since(t0);
      if (!account(r.stats)) return Node{};
      if (r.verdict == Outcome::Checkmate) {
        ++st().horizon_mates;
        st().horizon_mate_nodes_on_mates += r.stats.nodes;
        ++leaves.terminal;
        const Proof p = Proof::loss(0);
        export_proof(s, p);
        return Node{score_of(p, ply), p};
      }
      if (r.verdict == Outcome::Unknown) capped = true;  // its memory ceiling: unresolved
      ++leaves.horizon_in_check;
      return Node{evaluate(s), Proof{}};
    }
    ++st().evasion_visits;
    const int alpha_orig = alpha;
    Tally t;
    // The successors searched so far, one child each (two actions may reach
    // one state).
    std::vector<GameState> searched;
    const auto seen = [&](const GameState& r) {
      const std::uint64_t k = r.zobrist();
      for (const GameState& x : searched) {
        if (x.zobrist() == k && same_position(x, r)) return true;
      }
      return false;
    };
    // Stage 1.
    for (std::size_t i = 0; i < probe; ++i) {
      const std::optional<BuiltAction> a = quick_action(s, seeds[i].seed);
      if (stopped) return Node{};
      if (!a || seen(a->successor)) continue;
      ++st().probe_actions;
      searched.push_back(a->successor);
      const std::uint32_t k = move_key(representative(a->moves));
      if (!child(a->successor, a->moves, k, depth, alpha, beta, ply, board_moves, t)) {
        if (stopped) return Node{};
        ++st().probe_cutoffs;
        ++st().evasion_fail_highs;
        return finish(s, depth, alpha_orig, beta, ply, prev, key, t, false, false);
      }
    }
    // Stage 2.
    const std::size_t cap = opt.evasion_cap;
    const auto t0 = std::chrono::steady_clock::now();
    const ActionList list = list_legal_actions(s, cap + 1, Detector::Hypercuboid, detector_budget());
    st().evasion_nodes += list.stats.nodes;
    ++st().evasion_lists;
    st().evasion_listed += list.actions.size();
    st().evasion_seconds += seconds_since(t0);
    if (!account(list.stats)) return Node{};
    const bool exhaustive = list.stats.exhaustive && !list.reached_cap;
    if (exhaustive && list.actions.empty()) {
      if (!searched.empty()) {
        // Stage 1 built a legal action the exhaustive enumeration does not
        // have: a contradiction in the rules code.
        throw std::logic_error("scanner: E4 lists no legal action where the completion built one");
      }
      ++leaves.terminal;  // checkmate (C5, R1): E4's exhaustive verdict
      const Proof p = Proof::loss(0);
      export_proof(s, p);
      return Node{score_of(p, ply), p};
    }
    if (!exhaustive) {
      // Capped (or E4's memory ceiling): the actions listed so far are
      // searched - an estimate, never a loss proof.
      capped = true;
      ++st().capped_nodes;
    }
    std::vector<Candidate> cands;
    for (const LegalAction& a : list.actions) {
      if (seen(a.result)) continue;
      searched.push_back(a.result);
      Candidate c;
      c.action = BuiltAction{a.moves, a.result};
      const Move& rep = representative(a.moves);
      c.seed = rep;
      c.key = move_key(rep);
      cands.push_back(std::move(c));
    }
    for (Candidate& c : cands) {
      c.order = c.seed.capture() ? (std::int64_t{1} << 40) + mvv_lva(c.seed)
                                 : quiet_order(c.key, prev, ply);
      if (c.key == tt_best) c.order += std::int64_t{1} << 50;
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Candidate& a, const Candidate& b) { return a.order > b.order; });
    t.capped = t.capped || !exhaustive;
    bool all = true;
    for (Candidate& c : cands) {
      if (!child(c.action->successor, c.action->moves, c.key, depth, alpha, beta, ply, board_moves,
                 t)) {
        if (stopped) return Node{};
        all = (&c == &cands.back());
        break;
      }
    }
    if (t.best == -kInfinity) {
      // E4 stopped at its memory ceiling before listing any action: an
      // estimate.
      t.best = evaluate(s);
      ++leaves.horizon_in_check;
    }
    if (t.best >= beta) ++st().evasion_fail_highs;
    return finish(s, depth, alpha_orig, beta, ply, prev, key, t, all, exhaustive);
  }

  bool stand_pat_on(const GameState& s) const {
    switch (opt.stand_pat) {
      case StandPat::On: return true;
      case StandPat::Off: return false;
      case StandPat::OffWhereObligationsBind: return s.mandatory_count() < opt.stand_pat_bind_boards;
    }
    return true;
  }

  // Not in check: stand-pat, then the forcing actions in stages.
  Node forcing(const GameState& s, int depth, int alpha, int beta, int ply, std::uint32_t prev,
               int board_moves, std::uint64_t key, std::uint32_t tt_best) {
    const int stand = evaluate(s);
    if (depth == 0) {
      horizon_met = true;
      ++leaves.horizon;
      if (opt.classify_leaves && quiet_now(s)) ++leaves.horizon_quiet;
      return Node{stand, Proof{}};
    }
    const bool pat = stand_pat_on(s);
    const int alpha_orig = alpha;
    if (pat) {
      if (stand >= beta) {
        ++leaves.stand_pat_cutoffs;
        if (opt.classify_leaves && quiet_now(s)) ++leaves.stand_pat_quiet;
        return Node{stand, Proof{}};  // an estimate: the opponent may decline
      }
      alpha = std::max(alpha, stand);
    }
    Tally t;
    if (pat) t.best = stand;
    t.all_lost = false;  // not every legal action is searched: no ∀ proof here
    MoveList all_moves;
    generate_moves(s, all_moves);
    bool any = false;
    std::vector<std::uint64_t> done;
    // Stage 1: captures, MVV-LVA (the TT's move first).
    std::vector<Candidate> stage;
    for (const Move& m : all_moves) {
      if (!m.capture()) continue;
      Candidate c;
      c.seed = m;
      c.key = move_key(m);
      c.order = mvv_lva(m) + (c.key == tt_best ? (std::int64_t{1} << 40) : 0);
      stage.push_back(c);
    }
    bool go_on = true;
    const auto run_stage = [&](std::vector<Candidate>& cands) {
      std::stable_sort(cands.begin(), cands.end(),
                       [](const Candidate& a, const Candidate& b) { return a.order > b.order; });
      for (Candidate& c : cands) {
        std::optional<BuiltAction> a = complete(s, c.seed);
        if (stopped) return;
        if (!a) continue;
        const std::uint64_t k = a->successor.zobrist();
        if (std::find(done.begin(), done.end(), k) != done.end()) continue;
        done.push_back(k);
        any = true;
        if (!child(a->successor, a->moves, c.key, depth, alpha, beta, ply, board_moves, t)) {
          go_on = false;
          return;
        }
      }
    };
    run_stage(stage);
    if (stopped) return Node{};
    // Stage 2: checks; stage 3: branch threats; stage 4: promotions - killers,
    // history and continuation history first.
    if (go_on) {
      stage.clear();
      for (const Move& m : all_moves) {
        if (m.capture()) continue;
        const int cls = forcing_class(s, m);
        if (stopped) return Node{};
        if (cls == 0) continue;
        Candidate c;
        c.seed = m;
        c.key = move_key(m);
        c.order = (static_cast<std::int64_t>(5 - cls) << 44) + quiet_order(c.key, prev, ply) +
                  (c.key == tt_best ? (std::int64_t{1} << 42) : 0);
        stage.push_back(c);
      }
      run_stage(stage);
      if (stopped) return Node{};
    }
    if (!any) {
      // No forcing action: a quiet node - unless no legal action at all.
      if (opt.quiet_stalemate_test) {
        const MateResult r = detect_mate(s, Detector::Hypercuboid, detector_budget());
        st().stalemate_test_nodes += r.stats.nodes;
        if (!account(r.stats)) return Node{};
        if (r.verdict == Outcome::Stalemate) {
          ++leaves.terminal;
          const Proof p = Proof::draw();  // C5, C13: E4's exhaustive verdict
          export_proof(s, p);
          return Node{0, p};
        }
        if (r.verdict == Outcome::Checkmate) {
          // Not in check, so impossible (C5): a contradiction in the rules code.
          throw std::logic_error("scanner: a checkmate verdict without check");
        }
        if (r.verdict == Outcome::Unknown) capped = true;
      }
      ++leaves.quiet;
      return Node{stand, Proof{}};
    }
    if (t.best == -kInfinity) t.best = stand;  // stand-pat off, every child failed? (never)
    return finish(s, depth, alpha_orig, beta, ply, prev, key, t, false, false);
  }

  // -- iterative deepening ------------------------------------------------------------------------

  // Lazy SMP's staggered depths: the main thread (id 0) searches every depth;
  // helper i searches depth d only in alternate blocks of b = 1 + (i - 1) / 2
  // depths, shifted by i - so that while the main thread searches d, helpers
  // are already at d + 1 or d + 2, filling the TT ahead of it.
  bool skips(int depth) const {
    if (id == 0) return false;
    const int block = 1 + (id - 1) / 2;
    return ((depth + id) / block) % 2 != 0;
  }

  // This thread's iterative deepening from `root`, into `run`. Any thread
  // that decides the root - a proof, or a tree exhausted - stops the others.
  void iterate(const GameState& shared_root, WorkerRun& run) {
    // A helper searches its own copy of the root (see isolated_copy).
    std::optional<GameState> own_root;
    if (id != 0) own_root.emplace(isolated_copy(shared_root));
    const GameState& root = own_root.has_value() ? *own_root : shared_root;
    WorkerRun& result = run;
    stats = &result.stats;
    flushed = 0;
    stopped = false;
    charging = true;
    for (int d = 1; d <= opt.max_depth; ++d) {
      if (sh.threads > 1 && sh.stop.load(std::memory_order_relaxed)) break;
      if (skips(d)) continue;
      capped = false;
      horizon_met = false;
      max_board_moves = 0;
      leaves = LeafStats{};
      pv.clear();
      root_best.clear();
      const Node r = search(root, d, -kInfinity, kInfinity, 0, 0, 0);
      IterationStats it;
      it.depth = d;
      it.completed = !stopped;
      it.capped = capped;
      it.exhausted = !stopped && !horizon_met;
      it.score = r.score;
      it.proof = stopped ? Proof{} : r.proof;
      it.nodes = used();
      it.positions = result.stats.positions;
      it.max_board_moves = max_board_moves;
      it.leaves = leaves;
      it.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - sh.start).count();
      if (!stopped && !pv.empty()) {
        int moves = 0;
        for (const std::vector<Move>& a : pv.front()) moves += static_cast<int>(a.size());
        it.pv_board_moves = moves;
        it.pv_half_turns = static_cast<int>(pv.front().size());
      }
      result.iterations.push_back(it);
      if (stopped) break;
      result.completed_depth = d;
      if (!capped) result.resolved_depth = d;
      result.score = r.score;
      result.best_action = root_best;
      if (r.proof.proven() || it.exhausted) {
        result.exhausted = !r.proof.proven();  // a proof ends the scan first
        if (sh.threads > 1) sh.stop.store(true, std::memory_order_relaxed);  // decided
        break;
      }
    }
    flush(true);
    stats = nullptr;
    stopped = false;
  }

  // Empties this thread's move-ordering tables (between games, §11.3).
  void clear_tables() {
    killers.clear();
    std::fill(history.begin(), history.end(), 0);
    std::fill(conthist.begin(), conthist.end(), 0);
  }
};

}  // namespace

// ---- the scanner: its threads -----------------------------------------------------------------

struct Scanner::Impl {
  Impl(const ScannerOptions& o, SolvedStore* s) : shared(o, s) {
    for (int i = 0; i < shared.opt.threads; ++i) workers.push_back(std::make_unique<Worker>(shared, i));
  }

  Shared shared;
  std::vector<std::unique_ptr<Worker>> workers;

  Worker& lead() { return *workers.front(); }

  ScanResult scan(const GameState& root, const ScanBudget& b) {
    Shared& sh = shared;
    sh.budget = &b;
    sh.start = std::chrono::steady_clock::now();
    sh.threads = workers.size();
    sh.nodes.store(0, std::memory_order_relaxed);
    sh.stop.store(false, std::memory_order_relaxed);
    std::vector<WorkerRun> runs(workers.size());
    if (workers.size() == 1) {
      workers.front()->iterate(root, runs.front());  // the deterministic mode (H15)
    } else {
      // Lazy SMP (H17): every thread searches the same root; they share the
      // TT and the solved store, and nothing else.
      std::vector<std::exception_ptr> errors(workers.size());
      const auto run = [&](std::size_t i) {
        try {
          workers[i]->iterate(root, runs[i]);
        } catch (...) {
          errors[i] = std::current_exception();
          sh.stop.store(true, std::memory_order_relaxed);
        }
      };
      std::vector<std::thread> pool;
      pool.reserve(workers.size() - 1);
      for (std::size_t i = 1; i < workers.size(); ++i) pool.emplace_back(run, i);
      run(0);
      for (std::thread& t : pool) t.join();
      for (const std::exception_ptr& e : errors) {
        if (e) std::rethrow_exception(e);
      }
    }
    ScanResult result;
    result.threads = static_cast<int>(workers.size());
    // The result: the deepest iteration any thread completed (the main thread
    // on a tie); every thread's work summed.
    std::size_t best = 0;
    for (std::size_t i = 0; i < runs.size(); ++i) {
      const WorkerRun& r = runs[i];
      if (i == 0) {
        result.stats = r.stats;
      } else {
        accumulate(result.stats, r.stats);
      }
      result.resolved_depth = std::max(result.resolved_depth, r.resolved_depth);
      result.exhausted = result.exhausted || r.exhausted;
      if (r.completed_depth > runs[best].completed_depth) best = i;
    }
    result.completed_depth = runs[best].completed_depth;
    result.score = runs[best].score;
    result.best_action = runs[best].best_action;
    // A proof found by an iteration the budget then stopped is still a proof
    // (its own subtree was complete): the store has it.
    if (const std::optional<Proof> p = sh.find(root)) result.proof = *p;
    // The threads' root proofs agree (the store refuses two kinds for one
    // position, so a disagreement here is a contradiction in the code).
    for (const WorkerRun& r : runs) {
      for (const IterationStats& it : r.iterations) {
        if (it.proof.proven() && (!result.proof.proven() || it.proof.kind != result.proof.kind)) {
          throw std::logic_error("scanner: the threads' proofs of the root disagree (" +
                                 to_string(it.proof) + ", " + to_string(result.proof) + ")");
        }
      }
    }
    result.iterations = std::move(runs.front().iterations);
    result.stats.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - sh.start).count();
    sh.budget = nullptr;
    return result;
  }

  void clear() {
    shared.tt.clear();
    for (const std::unique_ptr<Worker>& w : workers) w->clear_tables();
  }
};

Scanner::Scanner(const ScannerOptions& options, SolvedStore* store)
    : impl_(std::make_unique<Impl>(options, store)) {}
Scanner::~Scanner() = default;

ScanResult Scanner::scan(const GameState& root, const ScanBudget& budget) {
  return impl_->scan(root, budget);
}

void Scanner::clear() { impl_->clear(); }

const ScannerOptions& Scanner::options() const noexcept { return impl_->shared.opt; }
SolvedStore& Scanner::store() noexcept { return *impl_->shared.store; }

int Scanner::evaluate(const GameState& state) const { return impl_->workers.front()->evaluate(state); }

MoveList Scanner::forcing_moves(const GameState& turn_start) {
  MoveList moves;
  generate_moves(turn_start, moves);
  MoveList out;
  for (const Move& m : moves) {
    if (impl_->lead().forcing_class(turn_start, m) != 0) out.push_back(m);
  }
  return out;
}

bool Scanner::has_forcing_move(const GameState& turn_start) {
  MoveList moves;
  generate_moves(turn_start, moves);
  for (const Move& m : moves) {
    if (impl_->lead().forcing_class(turn_start, m) != 0) return true;
  }
  return false;
}

bool Scanner::checks(const GameState& turn_start, const Move& move) {
  return impl_->lead().checks(turn_start, move);
}

std::optional<BuiltAction> Scanner::complete(const GameState& turn_start, const Move& seed) {
  return impl_->lead().complete(turn_start, seed);
}

// ---- stand-pat validity ------------------------------------------------------------------------

StandPatVerdict stand_pat_validity(const GameState& q, const ScannerOptions& options,
                                   const StandPatCheck& check) {
  StandPatVerdict v;
  v.mandatory_boards = q.mandatory_count();
  ScannerOptions one = options;
  one.threads = 1;  // the check is deterministic (H15)
  Scanner probe(one);
  if (in_check(q) || probe.has_forcing_move(q)) return v;  // NotQuiet
  v.eval = probe.evaluate(q);
  const std::size_t most = static_cast<std::size_t>(std::max(check.max_actions, 1));
  // The actions: every one, when there are few enough; else a sample.
  std::vector<GameState> successors;
  const auto add = [&](const GameState& s) {
    for (const GameState& t : successors) {
      if (t.zobrist() == s.zobrist() && same_position(t, s)) return;
    }
    successors.push_back(s);
  };
  SearchBudget lb = SearchBudget::nodes(200000);
  lb.max_bytes = options.detector_max_bytes;
  const ActionList all = list_legal_actions(q, most + 1, Detector::Hypercuboid, lb);
  v.nodes += all.stats.nodes;
  if (all.stats.exhaustive && !all.reached_cap) {
    if (all.actions.empty()) {
      v.kind = StandPatVerdict::Kind::NoAction;
      v.exhaustive = true;
      return v;
    }
    for (const LegalAction& a : all.actions) add(a.result);
    v.exhaustive = true;
  } else {
    MoveList moves;
    generate_moves(q, moves);
    for (const Move& m : moves) {
      if (successors.size() >= most) break;
      if (const std::optional<BuiltAction> a = probe.complete(q, m)) add(a->successor);
    }
    for (const LegalAction& a : all.actions) {
      if (successors.size() >= most) break;
      add(a.result);
    }
  }
  ScannerOptions reply = one;
  reply.max_depth = check.reply_depth;
  reply.classify_leaves = false;
  for (const GameState& s : successors) {
    Scanner scanner(reply);
    const ScanResult r = scanner.scan(s, ScanBudget::nodes(check.reply_nodes));
    v.nodes += r.stats.nodes;
    ++v.actions;
    int for_mover = 0;
    bool known = true;
    if (r.proof.kind == Proof::Kind::Win) {
      for_mover = -kScanMate;
    } else if (r.proof.kind == Proof::Kind::Loss) {
      for_mover = kScanMate;
    } else if (r.proof.kind == Proof::Kind::Draw) {
      for_mover = 0;
    } else if (r.exhausted || r.completed_depth >= check.reply_depth) {
      for_mover = -r.score;
    } else {
      known = false;
    }
    if (!known) {
      ++v.unknown;
    } else if (for_mover >= v.eval) {
      ++v.holding;
    } else {
      ++v.losing;
    }
  }
  if (v.holding > 0) {
    v.kind = StandPatVerdict::Kind::Valid;
  } else if (v.unknown > 0 || v.actions == 0) {
    v.kind = StandPatVerdict::Kind::Undetermined;
  } else {
    v.kind = StandPatVerdict::Kind::Invalid;
  }
  return v;
}

}  // namespace t5d
