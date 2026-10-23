// t5d/ladder.hpp - THE REDUCED VERIFICATION LADDER, WIRED INTO THE SEARCH
// (engine stage E10; design spec 11.2's scheduler as 11.4 reduced it, 11.3,
// 12.2; register H1, H10, H13, H14, H17, H19, H41-H43, M13; 6b''s adoption
// table).
//
// WHAT RUNS, WHERE (6b''s table; SearchParams' E10 block, each part ON by
// default in its range):
//  * HUNT (H43, t5d/hunt.hpp) at a decision's root - a turn start - under 100
//    boards: do WE have a forced mate within 5 half-turns? Prove-only: a stop
//    claims nothing. A fresh prover each question. Where the root's settling
//    scan is asked (under 50 boards) that scan asks the same question in the
//    same class and depth, and Hunt is not asked (SearchParams::
//    hunt_with_settling; the E10 review - build-log spec issue 181).
//  * THE FULL SETTLING SCANS (6b' (A): df-pn, the prover's default class,
//    depth 5) under 50 boards: at a turn-start root (our mate - proven,
//    disproven or unknown), and at each PV change, at the opponent's turn
//    start after the root's principal line (the refutation question: does
//    the opponent mate against our line? - build-log spec issue 181's reading).
//    One prover for the game (H17: its transposition table of estimates kept
//    across the game's scans; its proofs and its class-relative disproofs in
//    its own stores - never the permanent solved store, H19).
//  * THE ASSET MAP (H13, t5d/assets.hpp): each superseded board assessed once
//    per game, on the search's thread (deterministic), its boosts on travel
//    moves' priors (the search's set_priors; SearchParams::boost_rule). Off in
//    self-play by default until I2's steered visits are validated
//    (t5d/selfplay.hpp; issue 179).
//  * GUARD (H42) is adopted in no range: its hook is the driver's, OFF
//    (PlayParams::guard_final, t5d/selfplay.hpp).
// Not built: rungs 1, 3 and 5, the NNUE (M5); a lazily expanded Hunt root
// above 100 boards (H49, held open); settling above 50 boards (refused,
// 14.4); a whole-search background solver (H21, held open) - every job here
// is bounded and serves one decision.
//
// THE SCHEDULER (H14, M13: "its five cores serve Hunt, the settling scans and
// the audit lane"). One worker thread per ladder runs the jobs one at a time,
// in the order asked - the root's Hunt or its settling scan, then the PV
// changes' scans (a priority order the documents leave unstated: issue 182) -
// each prover on SearchParams::verification_threads threads (5) under its
// budget (50 ms: 6b''s measured configuration). The search never waits for a
// job: verdicts reach it between batches. A decision's end - the search
// destroyed or re-rooted - cancels its running job (ProveBudget::cancel) and
// drops its queued ones: NO JOB OUTLIVES ITS DECISION. Every verdict carries
// its decision's number; a search asks and takes only for its own decision
// (ask_for, take(decision)) - so a search whose decision another search sharing
// the ladder has ended (the game's next search began) gets nothing more from it
// (the review of E10 found a still-live donor taking its successor's verdicts).
// The asset map runs on
// the search's thread, on the movegen the search uses (its assessments are
// once per board, 128 nodes each - H13's "small, fixed").
//
// THE SYNCHRONOUS MODE (H15, SearchParams::synchronous): every job runs to
// its end where it is asked, on the search's thread, one prover thread, a
// node budget (sync_job_nodes) and no clock - the search stays bit-
// reproducible with the ladder on.
//
// PROOFS AND THE PROOF CHANNEL (H23, H41; AGENTS.md "The proof channel"). A
// proven verdict is a proof - exact sub-results only (H37) - and reaches the
// search's tree ONLY through Search::inject_proof, which with the proof
// channel off (the rule, until the gate passes and the channel is signed off)
// withholds it: nothing in the tree changes, the proof is counted and kept in
// the decision's verdicts (for E7's volatility signal and H44's audit lane).
// No proof is exported anywhere: the ladder has no store to export to, and
// it refuses Hunt's export switch (HuntOptions::export_proofs) as Hunt does.

#ifndef T5D_LADDER_HPP
#define T5D_LADDER_HPP

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "t5d/assets.hpp"
#include "t5d/certificate.hpp"
#include "t5d/hunt.hpp"
#include "t5d/mcts.hpp"
#include "t5d/prover.hpp"
#include "t5d/state.hpp"

namespace t5d::mcts {

/// The ladder's questions.
enum class LadderJob : std::uint8_t {
  Hunt = 0,          ///< our forced mate at a turn-start root (prove-only)
  RootSettling = 1,  ///< the settling scan at a turn-start root: our mate, proven or disproven
  PvSettling = 2,    ///< the settling scan after the principal line: the opponent's mate
};
const char* ladder_job_name(LadderJob job) noexcept;

/// One job's answer, as it reaches the search.
struct LadderVerdict {
  explicit LadderVerdict(GameState state) : position(std::move(state)) {}

  LadderJob job = LadderJob::Hunt;
  /// The decision it was asked for (VerificationLadder::begin_decision's
  /// number; a carried mate's: the decision it was offered in).
  std::uint64_t decision = 0;
  /// Not asked in this decision: the mate its action carries from the turn
  /// start (VerificationLadder::Carried) - `position` the root during the
  /// action, `first_action` the rest of the mating action.
  bool carried = false;
  /// The turn start asked about, and the path to it from the decision's root
  /// (empty: the root itself).
  GameState position;
  std::vector<PathStep> path;
  int boards = 0;  ///< the position's board count (the range it was judged in)
  int depth = 0;   ///< half-turns asked
  /// Hunt: Proven or Unknown (prove-only claims nothing else); the settling
  /// scans: Proven, Disproven (class-relative, H19 - never a bound) or Unknown.
  ProofStatus status = ProofStatus::Unknown;
  /// Proven: the position's MOVER mates within proof.distance half-turns.
  Proof proof;
  std::vector<Move> first_action;  ///< Proven: its first action, in a playable order
  std::shared_ptr<const ProofCertificate> certificate;  ///< SearchParams::ladder_certify
  std::vector<HuntSuggestion> suggestions;               ///< Hunt with hunt_suggest, no mate
  ProverStats stats;
  double seconds = 0;
  /// What became of the proof (Proven verdicts; Withheld while the proof
  /// channel is off; Expired with it on, where the decision ended before the
  /// verdict reached the tree), and the certificate's check where one was made.
  ProofUse use = ProofUse::Withheld;
  bool certificate_checked = false;
  CertificateVerdict certificate_verdict = CertificateVerdict::Undecided;
};

/// What a ladder holds for a whole game, fixed when it is made (a search
/// sharing it must ask for the same: ladder_options).
struct LadderOptions {
  AssetOptions assets;
  HuntOptions hunt;         ///< its prover's class (Hunt sets prove-only and the threads)
  ProverOptions settling;   ///< the settling scans' class (6b' (A): the default class)
  int threads = 5;          ///< each prover's threads (1 when synchronous)
  bool synchronous = false; ///< jobs inline, node budgets, no clock (H15)
  bool certify = true;      ///< the provers record certificates (SearchParams::ladder_certify)
};

/// The options a search's parameters ask of its ladder.
LadderOptions ladder_options(const SearchParams& params);
/// Whether two option sets make the same ladder.
bool same_ladder_options(const LadderOptions& a, const LadderOptions& b) noexcept;
/// Whether a search with these parameters needs a ladder at all.
bool ladder_wanted(const SearchParams& params) noexcept;

class VerificationLadder {
 public:
  /// A game's ladder, for positions of `ruleset` under `rules`. Throws
  /// std::invalid_argument if Hunt's export switch is set - no proof may enter
  /// a solved store until the generator has passed the gate (H23) and the
  /// proof channel is signed off (H41) - or for threads below 1.
  VerificationLadder(const LadderOptions& options, const RuleOptions& rules, std::uint32_t ruleset);
  ~VerificationLadder();
  VerificationLadder(const VerificationLadder&) = delete;
  VerificationLadder& operator=(const VerificationLadder&) = delete;

  const LadderOptions& options() const noexcept { return options_; }
  std::uint32_t ruleset() const noexcept { return ruleset_; }
  const RuleOptions& rules() const noexcept { return rules_; }
  /// The game's asset map - the searches' thread only.
  AssetMap& assets() noexcept { return assets_; }

  /// A new decision begins (a search's first batch, or a re-root): returns its
  /// number. The decision before it, if still open, ends (end_decision).
  std::uint64_t begin_decision();
  /// Decision `decision` ends - if it is the current one: its running job is
  /// cancelled (ProveBudget::cancel) and waited for, its queued jobs dropped,
  /// its verdicts not yet taken discarded. Another decision's number: nothing.
  void end_decision(std::uint64_t decision);
  std::uint64_t decision() const;

  /// Asks a question for the current decision: queued for the worker, or -
  /// synchronous - answered now, on this thread. `ms` (0: no clock) and
  /// `nodes` (0: none) bound it; synchronous, nodes only (0: the options'
  /// sync budget is the caller's to give). `suggest`: Hunt's soft priors.
  void ask(LadderJob job, const GameState& position, std::vector<PathStep> path, int depth, double ms,
           std::uint64_t nodes, bool suggest = false);
  /// The same for decision `decision` - asked only if it is the current one
  /// (returns whether it was): a search asks for its own decision only.
  bool ask_for(std::uint64_t decision, LadderJob job, const GameState& position, std::vector<PathStep> path,
               int depth, double ms, std::uint64_t nodes, bool suggest = false);
  /// The current decision's verdicts finished since the last take. Rethrows a
  /// job's exception (a contradiction in the rules code: std::logic_error).
  std::vector<LadderVerdict> take();
  /// The same for decision `decision`: nothing (and no exception) unless it is
  /// the current one - a search takes its own decision's verdicts only.
  std::vector<LadderVerdict> take(std::uint64_t decision);
  /// Blocks until no job is queued or running.
  void wait_idle();
  /// Jobs queued or running now.
  std::size_t pending() const;

  /// The mate carried through an action's decisions (the board-move split
  /// and the tree form, whose fresh searches each take one decision of it):
  /// the action proven - by Hunt or a root settling scan - to mate from
  /// `turn_start`.
  struct Carried {
    explicit Carried(GameState s) : turn_start(std::move(s)) {}
    GameState turn_start;
    std::vector<Move> action;
    Proof proof;
    std::shared_ptr<const ProofCertificate> certificate;
    ProofUse use = ProofUse::Withheld;
    bool certificate_checked = false;
    CertificateVerdict certificate_verdict = CertificateVerdict::Undecided;
  };
  void carry(Carried c);
  const std::optional<Carried>& carried() const noexcept { return carried_; }

  /// Over the ladder's life. Once nothing is pending, asked = answered +
  /// cancelled + discarded + contradictions + the verdicts not yet taken.
  struct Stats {
    std::uint64_t asked = 0;
    std::uint64_t answered = 0;   ///< verdicts taken by their decision
    std::uint64_t cancelled = 0;  ///< dropped from the queue, or stopped by the cancel flag
    /// Finished, but their decision ended before it took them (a job finishing
    /// while end_decision waits for it, or after the decision moved on).
    std::uint64_t discarded = 0;
    /// Jobs that threw - a contradiction in the rules code (a prover's
    /// std::logic_error included): reported on stderr whatever their decision,
    /// and rethrown by take() where their decision is the current one.
    std::uint64_t contradictions = 0;
    std::uint64_t threads_started = 0;  ///< worker threads made (one per ladder, lazily)
    double seconds = 0;           ///< in the provers
  };
  Stats stats() const;

  /// TESTS ONLY: a hook each verdict passes through as it is taken (before the
  /// search sees it) - the seam that feeds the search a certificate E4 rejects,
  /// so that the "only E4-confirmed proofs enter" gate is exercised. Null: none
  /// (the default; nothing outside the tests sets one).
  using VerdictHook = std::function<void(LadderVerdict&)>;
  void set_verdict_hook_for_tests(VerdictHook hook);

 private:
  struct Job {
    LadderJob kind = LadderJob::Hunt;
    std::unique_ptr<GameState> position;
    std::vector<PathStep> path;
    int depth = 5;
    double ms = 0;
    std::uint64_t nodes = 0;
    bool suggest = false;
    std::uint64_t decision = 0;
  };
  LadderVerdict run(const Job& job);
  void worker();

  LadderOptions options_;
  RuleOptions rules_;
  std::uint32_t ruleset_ = 0;
  AssetMap assets_;
  std::unique_ptr<Prover> settling_;  // the game's settling prover (the worker's, or inline)
  std::optional<Carried> carried_;

  mutable std::mutex m_;
  std::condition_variable work_cv_;
  std::condition_variable idle_cv_;
  std::deque<Job> queue_;
  std::vector<LadderVerdict> done_;
  std::exception_ptr error_;
  VerdictHook hook_;
  bool running_ = false;
  bool quit_ = false;
  std::atomic<bool> cancel_{false};
  std::uint64_t decision_ = 0;
  Stats stats_;
  std::thread thread_;
};

}  // namespace t5d::mcts

#endif  // T5D_LADDER_HPP
