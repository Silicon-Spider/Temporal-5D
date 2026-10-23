// The reduced verification ladder, wired into the search (t5d/ladder.hpp;
// engine stage E10). Written from the design spec (11.2-11.4, 12.2) and the
// register; the provers are the engine's own (stage E8/E9).

#include "t5d/ladder.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace t5d::mcts {

const char* ladder_job_name(LadderJob job) noexcept {
  switch (job) {
    case LadderJob::Hunt: return "hunt";
    case LadderJob::RootSettling: return "root_settling";
    case LadderJob::PvSettling: return "pv_settling";
  }
  return "?";
}

const char* proof_use_name(ProofUse use) noexcept {
  switch (use) {
    case ProofUse::Injected: return "injected";
    case ProofUse::Withheld: return "withheld";
    case ProofUse::Deferred: return "deferred";
    case ProofUse::Unplaced: return "unplaced";
    case ProofUse::Unconfirmed: return "unconfirmed";
  }
  return "?";
}

LadderOptions ladder_options(const SearchParams& p) {
  LadderOptions o;
  o.assets = p.assets;
  o.hunt.prover.depth = p.hunt_depth;  // the class: the prover's defaults (H43's placeholder)
  o.settling.depth = p.settling_depth;  // 6b' (A): the default class
  o.synchronous = p.synchronous;
  o.threads = p.synchronous ? 1 : p.verification_threads;
  o.certify = p.ladder_certify;
  return o;
}

namespace {

bool same_assets(const AssetOptions& a, const AssetOptions& b) noexcept {
  return a.values.value == b.values.value && a.royal_value == b.royal_value && a.qsearch_plies == b.qsearch_plies &&
         a.qsearch_nodes == b.qsearch_nodes && a.flag_margin == b.flag_margin && a.boost_scale == b.boost_scale &&
         a.max_boost == b.max_boost;
}

}  // namespace

bool same_ladder_options(const LadderOptions& a, const LadderOptions& b) noexcept {
  return same_assets(a.assets, b.assets) && a.hunt.prover.depth == b.hunt.prover.depth &&
         a.settling.depth == b.settling.depth && a.threads == b.threads && a.synchronous == b.synchronous &&
         a.certify == b.certify && a.hunt.export_proofs == b.hunt.export_proofs;
}

bool ladder_wanted(const SearchParams& p) noexcept { return p.hunt || p.settling || p.asset_map || p.hunt_suggest; }

VerificationLadder::VerificationLadder(const LadderOptions& options, const RuleOptions& rules, std::uint32_t ruleset)
    : options_(options), rules_(rules), ruleset_(ruleset), assets_(options.assets, rules) {
  if (options_.hunt.export_proofs) {
    throw std::invalid_argument(
        "the verification ladder: Hunt's export_proofs - the proof channel is disabled, and no proof may enter a "
        "solved store until the generator has passed the gate (H23) and the proof channel is signed off (H41)");
  }
  if (options_.threads < 1) throw std::invalid_argument("the verification ladder: threads must be 1 or more");
  if (options_.synchronous) options_.threads = 1;
}

VerificationLadder::~VerificationLadder() {
  {
    std::unique_lock<std::mutex> lock(m_);
    quit_ = true;
    queue_.clear();
    cancel_.store(true, std::memory_order_relaxed);
  }
  work_cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

std::uint64_t VerificationLadder::begin_decision() {
  std::uint64_t current = 0;
  {
    std::lock_guard<std::mutex> lock(m_);
    current = decision_;
  }
  end_decision(current);
  std::lock_guard<std::mutex> lock(m_);
  return decision_;
}

void VerificationLadder::end_decision(std::uint64_t decision) {
  std::unique_lock<std::mutex> lock(m_);
  if (decision != decision_) return;
  stats_.cancelled += queue_.size();
  queue_.clear();
  if (running_) {
    cancel_.store(true, std::memory_order_relaxed);
    idle_cv_.wait(lock, [this] { return !running_; });
    cancel_.store(false, std::memory_order_relaxed);
  }
  done_.clear();
  ++decision_;
}

std::uint64_t VerificationLadder::decision() const {
  std::lock_guard<std::mutex> lock(m_);
  return decision_;
}

void VerificationLadder::ask(LadderJob kind, const GameState& position, std::vector<PathStep> path, int depth,
                             double ms, std::uint64_t nodes, bool suggest) {
  Job job;
  job.kind = kind;
  job.position = std::make_unique<GameState>(position);
  job.path = std::move(path);
  job.depth = depth;
  job.ms = ms;
  job.nodes = nodes;
  job.suggest = suggest;
  if (options_.synchronous) {
    {
      std::lock_guard<std::mutex> lock(m_);
      job.decision = decision_;
      ++stats_.asked;
    }
    LadderVerdict v = run(job);  // inline: the search's thread, no clock (H15)
    std::lock_guard<std::mutex> lock(m_);
    ++stats_.answered;
    stats_.seconds += v.seconds;
    done_.push_back(std::move(v));
    return;
  }
  {
    std::lock_guard<std::mutex> lock(m_);
    job.decision = decision_;
    ++stats_.asked;
    queue_.push_back(std::move(job));
    if (!thread_.joinable()) {
      thread_ = std::thread([this] { worker(); });
      ++stats_.threads_started;
    }
  }
  work_cv_.notify_one();
}

std::vector<LadderVerdict> VerificationLadder::take() {
  std::lock_guard<std::mutex> lock(m_);
  if (error_) {
    std::exception_ptr e = error_;
    error_ = nullptr;
    std::rethrow_exception(e);
  }
  std::vector<LadderVerdict> out = std::move(done_);
  done_.clear();
  return out;
}

void VerificationLadder::wait_idle() {
  std::unique_lock<std::mutex> lock(m_);
  idle_cv_.wait(lock, [this] { return queue_.empty() && !running_; });
}

std::size_t VerificationLadder::pending() const {
  std::lock_guard<std::mutex> lock(m_);
  return queue_.size() + (running_ ? 1u : 0u);
}

void VerificationLadder::carry(Carried c) { carried_.emplace(std::move(c)); }

VerificationLadder::Stats VerificationLadder::stats() const {
  std::lock_guard<std::mutex> lock(m_);
  return stats_;
}

void VerificationLadder::worker() {
  std::unique_lock<std::mutex> lock(m_);
  for (;;) {
    work_cv_.wait(lock, [this] { return quit_ || !queue_.empty(); });
    if (quit_) return;
    Job job = std::move(queue_.front());
    queue_.pop_front();
    running_ = true;
    lock.unlock();
    std::optional<LadderVerdict> v;
    std::exception_ptr error;
    try {
      v.emplace(run(job));
    } catch (...) {
      error = std::current_exception();
    }
    lock.lock();
    running_ = false;
    if (job.decision == decision_ && !quit_) {
      if (error) {
        error_ = error;
      } else if (v->stats.stopped_by_cancel) {
        ++stats_.cancelled;
      } else {
        ++stats_.answered;
        stats_.seconds += v->seconds;
        done_.push_back(std::move(*v));
      }
    } else {
      ++stats_.cancelled;
    }
    idle_cv_.notify_all();
  }
}

LadderVerdict VerificationLadder::run(const Job& job) {
  const auto t0 = std::chrono::steady_clock::now();
  ProveBudget budget;
  if (options_.synchronous) {
    budget.max_nodes = job.nodes;  // the caller's sync budget; no clock (H15)
  } else {
    if (job.nodes > 0) budget.max_nodes = job.nodes;
    if (job.ms > 0) {
      budget.max_time = std::chrono::nanoseconds(static_cast<std::int64_t>(job.ms * 1e6));
    }
    budget.cancel = &cancel_;
  }
  LadderVerdict v(*job.position);
  v.job = job.kind;
  v.path = job.path;
  v.boards = static_cast<int>(job.position->multiverse().board_count());
  v.depth = job.depth;
  if (job.kind == LadderJob::Hunt) {
    HuntOptions h = options_.hunt;
    h.threads = options_.threads;
    h.prover.depth = job.depth;
    h.prover.record_certificate = options_.certify;
    h.suggest = job.suggest;
    Hunt hunt(h);  // a fresh prover each question (t5d/hunt.hpp)
    HuntResult r = hunt.hunt(*job.position, budget);
    v.stats = r.stats;
    if (r.mate) {
      v.status = ProofStatus::Proven;
      v.proof = r.proof;
      v.first_action = std::move(r.first_action);
      v.certificate = std::move(r.certificate);
    }
    v.suggestions = std::move(r.suggestions);
  } else {
    if (!settling_) {
      ProverOptions po = options_.settling;
      po.threads = options_.threads;
      po.prove_only = false;
      po.record_certificate = options_.certify;
      // The game's prover: its own proof and disproof stores (never the
      // permanent store, H19, H23) and its TT of estimates, kept across the
      // game's scans (H17).
      settling_ = std::make_unique<Prover>(po);
    }
    const ProveResult r = settling_->prove(*job.position, budget, job.depth);
    v.stats = r.stats;
    v.status = r.status;
    if (r.status == ProofStatus::Proven) {
      v.proof = r.proof;
      v.first_action = r.first_action;
      v.certificate = r.certificate;
    }
  }
  v.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return v;
}

}  // namespace t5d::mcts
