// t5d/timeman.hpp - TIME MANAGEMENT (engine stage E11; design spec §8.6,
// §11.4's knock-ons, §12.3, §13.1's 6o; register E7, H41, H42, I4, M13, M22).
//
// THE FRAME (§8.6): "every stop/extend signal is a proxy for value-of-
// information versus opportunity cost ... Signals act on the soft bound; the
// hard bound is inviolable." A TimeManager serves ONE player's clock through a
// game (match play: one per side), decision by decision:
//
//  * begin() PLANS the decision: the soft bound (in evaluations, with its
//    wall-clock equivalent) and the hard bound (wall-clock seconds);
//  * check(), called BETWEEN BATCHES, says whether to stop - a stop signal, the
//    soft bound reached with no extension left to take, or the hard bound - or
//    extends the soft bound;
//  * end() records what the next decision's signals compare with.
// run_timed() drives a Search with them; the C API (t5d/capi_search.h) gives the
// same three calls to a binding that drives the step API itself.
//
// ALLOCATION - THE MOVES-LEFT INTEGRAL (§8.6, 6o's arms: Allocator). N grows
// with every board-move, so the cost of an evaluation over the rest of the
// game is known given the moves-left head: "allocate this move's share as its
// cost over the summed cost of all remaining moves, not 1/moves-left". The
// cost of an evaluation at N boards (seconds_per_evaluation) is the dense FLOP
// model's SHAPE - T (4T + (8 + 4r) d), T = tokens_per_board N + global_tokens,
// the model of python/t5d/net/cost.py written here in-house - times a SCALE,
// measured online: an exponential average of the seconds per evaluation
// observed between checks, the shape removed (the shape predicts, the scale is
// measured). Our remaining actions: half the turns the moves-left head
// predicts (D4: in turns, each one player's action - build-log spec issue 305),
// rounded up and floored (§8.6: "the noisiest input in the allocator - give it
// a conservative floor"); N advances between two of our actions by both
// players' boards per action (boards_per_action, per board-count range: M22's
// moves per turn on the human games, a placeholder until the pilot run
// measures the engine's own). The integral's evaluations per action are
// E = T_eff / sum_k s(N_k), T_eff = remaining + increment (A - 1) - overhead A.
// Under the ply split and the tree form an action is several decisions: its
// mandatory boards still to move (A6, exact), the submit decision and
// expected_optional_moves (a placeholder); the action's plan is split evenly
// over those left and re-planned at every decision. The flat form's decision
// is the whole action.
//
// (A share, cost_fixed_fraction, of an evaluation's cost at the reference
// board count is taken not to grow with N - a placeholder, 0 in §8.6's table.)
//
// THE STOP/EXTEND TABLE (§8.6; each a pure function below, every threshold a
// named placeholder - build-log spec issues 452-470):
//   stop    the visit gap between the top two children unbridgeable in the
//           remaining budget (Lc0's smart pruning);
//   stop    the root's visit distribution stops changing (the KLDGain
//           pattern: the KL divergence per root visit over kld_interval);
//   play    the root solved - ONLY with the proof channel on (H41); with it off
//           nothing is solved, and the stop is void;
//   extend  the top two children's categoricals overlap AND the KLD still
//           moves ("overlap alone can mean equivalent lines");
//   extend  volatility - the root's uncertainty signal (the categorical's
//           variance, or the error head's: SearchParams::uncertainty_source,
//           6m); the second source, a proof from Hunt or the settling scans
//           that contradicts the net, is a HOOK, OFF (proof_volatility) and
//           refused without the proof channel: whether those proofs may drive
//           time before the sign-off is open (build-log spec issue 180). With
//           both read, a calm net and an alarmed prover - DISAGREEMENT - extends
//           hardest;
//   extend  a falling eval: the root's value below the previous decision's by
//           fall_margin (asymmetric: a rising eval never extends);
//   extend  best-move instability: the root's most visited child changed
//           within the last instability_window of its visits.
// Stops are judged at every check (from stop_min_evaluations on); extensions
// when the soft bound is reached - each kind AT MOST ONCE PER DECISION, each
// by its share of the ORIGINAL plan, all of them together at most
// max_extension of it (§8.6's spiral guard: "the extension mechanism must
// never be able to flag itself"); none can cross the hard bound.
//
// THE HARD BOUND (§8.6: "the hard maximum is wall-clock and held tight at high
// N"): hard_multiple(N) times the soft bound - falling from hard_multiple_low
// at hard_low_boards to hard_multiple_high at hard_high_boards - and never more
// than hard_fraction of the clock. check() stops when the time spent plus
// batch_margin times the next batch's predicted time reaches it, so a batch
// never starts that would end past it; inside a batch, run_timed gives the
// search the bound as its deadline (Search::set_deadline), which the rules
// engine's calls inside the batch - the turn starts' mate checks, the legality
// classification and its verification, the flat form's fallback listing - stop
// at, answering Unknown, never a verdict (H37). The one exception: a decision
// whose root has no visited child yet runs one batch whatever the clock says
// (a move must be chosen, and choosing needs a visit).
//
// NO ROOT-SCAN RESERVE (§8.6's "Guards": "a reserve held for one re-search if
// that scan refutes"; register E7): its one named consumer, Guard's
// re-search (§11.4), runs in no range (H42), so none is held (build-log spec
// issue 306). Guard's hook, if an experiment switches it on, re-searches with
// what the hard bound leaves (t5d/selfplay.hpp). THE LADDER'S JOBS (stage E10,
// t5d/ladder.hpp) - Hunt under 100 boards, the settling scans under 50 - run on
// their own worker and threads (M13's five cores) beside the search: their
// wall time overlaps the decision's, and the decision's end cancels them, so
// they sit inside the hard bound by construction; the allocation deducts
// nothing for them (a reading: issue 466). §11.4's knock-on ("time management
// loses the scanner as a second volatility source above 50 boards") is moot
// while the proof source is off.
//
// CLOCKS AND DETERMINISM (H15). A Clock gives the time; WallClock is the
// steady clock. With no clock (nullptr) the MODEL CLOCK stands in: the time a
// decision has spent is its work times the cost model's seconds per
// evaluation, the scale never updated - so a timed search is a node-limited
// search, bit-reproducible with a deterministic evaluator (the synchronous
// mode's contract). Self-play does not use any of this: its budgets are
// PCR's (§12.5), and Limits::seconds is "never training".

#ifndef T5D_TIMEMAN_HPP
#define T5D_TIMEMAN_HPP

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "t5d/mcts.hpp"

namespace t5d::mcts {

/// 6o's three arms (§13.1: "integral allocator vs flat-evals vs
/// flat-wall-clock"; the arms are named, not defined - build-log spec issue
/// 453).
enum class Allocator : std::uint8_t {
  /// §8.6: equal evaluations for each of our remaining actions, the cost of
  /// each at its predicted N - this action's share is its cost over the sum.
  Integral = 0,
  /// The empirical engine's assumption (§8.6: "engines that estimate nps
  /// empirically must assume it persists"): equal evaluations per action at
  /// TODAY's cost - T_eff / (A s(N)) evaluations, executed by evaluations.
  FlatEvals = 1,
  /// Equal wall-clock time per action, T_eff / A seconds, executed by the
  /// clock (§8.6's "naive equal-time share").
  FlatWallClock = 2,
};
const char* allocator_name(Allocator a) noexcept;

/// Why a timed decision stopped (TimeVerdict::stop; None: go on).
enum class TimeStop : std::uint8_t {
  None = 0,
  SoftBudget = 1,    ///< the soft bound (with its extensions) reached, no extension left to take
  SmartPruning = 2,  ///< the visit gap unbridgeable in the remaining budget
  KldGain = 3,       ///< the root's visit distribution stopped changing
  Solved = 4,        ///< the root solved, the proof channel on (H41): play at once
  HardBound = 5,     ///< the wall-clock maximum (the next batch would cross it)
  Search = 6,        ///< the search stopped itself (a terminal or stuck root, exhausted, a node or memory limit)
};
const char* time_stop_name(TimeStop s) noexcept;
inline constexpr int kTimeStops = 7;

/// The extension kinds, bits of TimeVerdict::extensions.
inline constexpr std::uint32_t kExtendOverlap = 1;        ///< top-two overlap with churn
inline constexpr std::uint32_t kExtendVolatility = 2;     ///< the net's variance (or both sources alarmed)
inline constexpr std::uint32_t kExtendDisagreement = 4;   ///< a calm net, an alarmed prover (the hook, off)
inline constexpr std::uint32_t kExtendFall = 8;           ///< a falling eval
inline constexpr std::uint32_t kExtendInstability = 16;   ///< the best move changed recently

/// Every number is a named placeholder (build-log spec issues 452-470); the
/// engine README's "Time management" gives each with its standing.
struct TimeParams {
  // ---- 6o's arms -------------------------------------------------------------
  Allocator allocator = Allocator::Integral;
  /// The stop/extend signal set (6o: "on/off"). Off: the allocator's plan and
  /// the hard bound alone (a solved root still plays at once - with the
  /// proof channel on, the search itself stops there).
  bool signals = true;

  // ---- the cost model: s(N) = scale x shape(N) / shape(reference) -----------
  /// The dense FLOP model's tokens (§9: T = 66 N + 16) and width (d 256, FFN
  /// ratio 8: the 40 d of T (40 d + 4 T)).
  int cost_tokens_per_board = 66;
  int cost_global_tokens = 16;
  int cost_d_model = 256;
  double cost_ffn_ratio = 8;
  /// The board count the scale is quoted at.
  int cost_reference_boards = 100;
  /// The scale before any observation: seconds per evaluation at the reference
  /// board count. The first observation replaces it (placeholder; CPU figures
  /// are CPU-labelled, M19).
  double cost_seconds_per_evaluation = 0.01;
  /// The online average's weight on each later observation.
  double cost_ema = 0.2;
  /// The share of an evaluation's cost at the reference board count that does
  /// not grow with N (the driver, the conversion, the expansion's fixed part):
  /// s(N) = scale (f + (1 - f) shape(N) / shape(reference)). 0: the dense
  /// model alone (§8.6's table). Placeholder 0.05 - without it a scale
  /// measured at a few boards prices a 100-board evaluation hundreds of times
  /// too high (issue 452).
  double cost_fixed_fraction = 0.05;

  // ---- moves left and the board trajectory ------------------------------------
  /// The conservative floor on the moves-left prediction, in turns (both
  /// players'; §8.6).
  double moves_left_floor_turns = 20;
  /// Turns left assumed when no prediction is given (the built-in evaluators
  /// give none): about a whole human game (M22's data: 29,292 actions in 692
  /// games, 42 a game).
  double moves_left_default_turns = 40;
  /// Boards one action adds, by the board count at its start: M22's mean
  /// moves per turn on the human games under 50, 50-99 and 100+ boards (a
  /// travel move's second board left out; issue 457).
  std::array<double, 3> boards_per_action{1.05, 1.25, 1.77};
  std::array<int, 2> boards_per_action_ranges{50, 100};
  /// Ply split and tree form: the optional moves an action is expected to
  /// make beyond its mandatory boards - each a decision.
  double expected_optional_moves = 0.25;
  /// Per action: the clock's overhead (the move's transmission, the driver).
  double overhead_seconds = 0.05;
  /// No decision is planned below this many evaluations (inside the hard bound).
  double min_evaluations = 8;
  /// A simulation that needs no evaluation (a cache hit, a terminal) still
  /// costs: the decision's WORK is max(evaluations, playouts / this) - what the
  /// soft bound and the model clock count.
  double playouts_per_evaluation_cap = 4;

  // ---- the hard bound -----------------------------------------------------------
  double hard_multiple_low = 5;   ///< hard = this x soft at hard_low_boards or fewer ...
  double hard_multiple_high = 2;  ///< ... this at hard_high_boards or more, linear between
  int hard_low_boards = 50;
  int hard_high_boards = 400;
  /// The hard bound never exceeds this share of the clock left (after the overhead).
  double hard_fraction = 0.25;
  /// Stop when spent + batch_margin x the next batch's predicted time reaches the hard bound.
  double batch_margin = 1.5;

  // ---- the spiral guard -----------------------------------------------------------
  /// All extensions of a decision together, as a share of its original plan.
  double max_extension = 1.0;

  // ---- stops ------------------------------------------------------------------------
  /// No stop signal before this much work (evaluations).
  double stop_min_evaluations = 32;
  /// Smart pruning: stop when the second child cannot catch the first even if
  /// every remaining playout (remaining evaluations x the playouts per
  /// evaluation so far) went to it, scaled by this (1: exactly unbridgeable).
  double smart_pruning_factor = 1.0;
  /// KLDGain: the root's visit distribution compared every this many root visits ...
  std::uint32_t kld_interval = 100;
  /// ... the stop when the KL divergence per visit falls below this.
  double kld_threshold = 5e-5;

  // ---- extensions (each a share of the original plan) ----------------------------
  /// Overlap: the top two children's categoricals' histogram intersection
  /// (in [0, 1]) at least this ...
  double overlap_threshold = 0.5;
  /// ... and the last KLD per visit at least this (churn).
  double churn_kld_threshold = 2e-4;
  double extend_overlap = 0.5;
  /// Volatility: the root's uncertainty signal (a variance of values in
  /// [-1, 1]) at least this.
  double variance_threshold = 0.2;
  double extend_volatility = 0.5;
  /// THE SECOND VOLATILITY SOURCE - a HOOK, OFF: Hunt's and the settling scans'
  /// proofs read as an alarm where the proven value lies disagreement_margin or
  /// more from the network's. Refused unless the search's proof channel is on
  /// (issue 180: whether those proofs may drive time before the sign-off is
  /// open; until it is ruled, they may not).
  bool proof_volatility = false;
  double disagreement_margin = 0.5;
  double extend_disagreement = 1.0;  ///< "disagreement extends hardest"
  /// Falling eval: the root's backed-up value below the previous decision's by this.
  double fall_margin = 0.1;
  double extend_fall = 0.5;
  /// Instability: the most visited child changed within the last this share
  /// of the root's visits, once the root has instability_min_visits.
  double instability_window = 0.25;
  std::uint32_t instability_min_visits = 32;
  double extend_instability = 0.5;
};

/// Throws std::invalid_argument for a parameter out of its range.
void validate(const TimeParams& p);

/// One decision's clock (the caller's: match play's per-side clock).
struct TimeContext {
  double remaining_seconds = 0;  ///< our clock now (> 0)
  double increment_seconds = 0;  ///< added after each of our actions (a reading of the 5D clock: issue 456)
  /// D4's moves-left, in turns (both players'); NaN: the root's own network
  /// output if it has one, else moves_left_default_turns.
  double moves_left_turns = std::numeric_limits<double>::quiet_NaN();
};

/// A decision's plan.
struct TimePlan {
  int boards = 0;                    ///< N at the root
  double moves_left_turns = 0;       ///< after the floor
  bool moves_left_given = false;     ///< a prediction was read (else the default)
  double our_actions = 0;            ///< A: our remaining actions, this one included
  double decisions_left = 1;         ///< this action's decisions left, this one included
  double seconds_per_evaluation = 0; ///< s(N), the scale as it stands
  double usable_seconds = 0;         ///< T_eff
  double share = 0;                  ///< this action's share of T_eff
  double action_evaluations = 0;     ///< the action's planned evaluations
  double soft_evaluations = 0;       ///< this decision's soft bound ...
  double soft_seconds = 0;           ///< ... and its time (the arm FlatWallClock executes by it)
  double hard_seconds = 0;           ///< the hard bound
};

/// What check() decided.
struct TimeVerdict {
  TimeStop stop = TimeStop::None;
  double work = 0;                ///< the decision's work so far (max(evaluations, playouts / cap))
  double target_evaluations = 0;  ///< the soft bound now (the plan and its extensions)
  std::uint32_t extensions = 0;   ///< the kinds taken this decision (kExtend*)
  double elapsed_seconds = 0;     ///< by the clock given, or the model clock
  double predicted_batch_seconds = 0;
  double kld_gain = std::numeric_limits<double>::quiet_NaN();  ///< the last KLD per visit (NaN: none yet)
  /// The decision converged: it stopped on smart pruning, the KLD or a solved root.
  bool converged() const noexcept {
    return stop == TimeStop::SmartPruning || stop == TimeStop::KldGain || stop == TimeStop::Solved;
  }
};

/// The root as the signals read it, between batches (snapshot(); tests build
/// them by hand). Children in root_children() order, the root player's view.
struct RootSnapshot {
  std::uint64_t evaluations = 0;  ///< since the decision began
  std::uint64_t playouts = 0;     ///< since the decision began
  std::uint32_t root_visits = 0;
  std::vector<std::uint32_t> visits;  ///< path visits (MCGS: the edges')
  double root_q = 0;                  ///< the root's backed-up value
  double root_variance = 0;           ///< its uncertainty signal (0: none given)
  double root_net_value = 0;          ///< its own evaluation's W - L
  bool root_evaluated = false;
  std::uint32_t best_since = 0;       ///< Search::root_best_since
  /// The top two children's (by visits) categoricals, the root player's view
  /// (empty where none can be read).
  std::optional<std::array<float, kValueBins>> top_value[2];
  bool solved = false;
  bool proof_channel = false;
  std::size_t batch_size = 1;
  /// The proof hook (proof_volatility): the largest gap between a proof's
  /// value and the network's (0: no proof).
  double proof_gap = 0;
};

/// The root of `search` now; `evaluations` and `playouts` since the decision
/// began; `params` decides what is read (the categoricals with the signals on,
/// the proofs with proof_volatility).
RootSnapshot snapshot(const Search& search, std::uint64_t evaluations, std::uint64_t playouts, const TimeParams& params);

// ---- the signals as pure functions (§8.6's table) --------------------------------

/// The dense FLOP model's shape at `boards` boards: T (4 T + (8 + 4 r) d).
double cost_shape(const TimeParams& p, double boards) noexcept;
/// An evaluation's cost at `boards` boards relative to the reference board
/// count: cost_fixed_fraction + (1 - it) shape(N) / shape(reference).
double cost_relative(const TimeParams& p, double boards) noexcept;
/// Boards one action adds at `boards` boards (boards_per_action's range).
double boards_per_action(const TimeParams& p, double boards) noexcept;
/// hard_multiple(N): hard_multiple_low at hard_low_boards or fewer, falling
/// linearly to hard_multiple_high at hard_high_boards.
double hard_multiple(const TimeParams& p, double boards) noexcept;
/// The integral's sum in units of this action's cost - sum_k s(N_k) / s(N_0)
/// (its share is the inverse) - over `actions` actions, N_0 =
/// `boards`, N_{k+1} = N_k + `step` x boards_per_action(N_k) (step 2: both
/// players' actions between two of ours; 1: every action counted - §8.6's
/// table). Its evaluations per action follow as T / sum.
double integral_cost_sum(const TimeParams& p, double boards, double actions, double step) noexcept;
/// KL(prev || cur) of two visit distributions (each normalised over itself;
/// prev's zeros contribute nothing; cur longer than prev: the new children
/// held 0 before). Infinite if prev has mass where cur has none; NaN if prev
/// has none (nothing to compare with).
double visit_kld(std::span<const std::uint32_t> prev, std::span<const std::uint32_t> cur) noexcept;
/// Histogram intersection of two distributions (each normalised first): in [0, 1].
double value_overlap(const std::array<float, kValueBins>& a, const std::array<float, kValueBins>& b) noexcept;
/// Smart pruning: the most visited child's lead over the second exceeds what
/// `remaining_playouts` x factor could close. A single child: any visit.
bool smart_pruning_stop(std::span<const std::uint32_t> visits, double remaining_playouts, double factor) noexcept;
/// Instability: the best child changed within the last `window` of the root's
/// visits (root at least min_visits; best_since 0: never established).
bool unstable(std::uint32_t root_visits, std::uint32_t best_since, double window, std::uint32_t min_visits) noexcept;

/// The plan, as a pure function of the parameters, the cost scale and the
/// root: `boards`, its mandatory boards still to move (A6), whether the
/// split's decision is the whole action (the flat form), and the root's own
/// moves-left prediction (NaN: none; the context's comes first).
TimePlan plan_decision(const TimeParams& p, double scale, int boards, int mandatory_remaining, bool whole_action,
                       const TimeContext& ctx, double root_moves_left);

/// One player's time manager over a game (see the header comment).
class TimeManager {
 public:
  explicit TimeManager(TimeParams params = {});
  const TimeParams& params() const noexcept { return params_; }

  /// Plans a decision at `search`'s root and begins it. Throws
  /// std::invalid_argument for a context out of range (remaining not > 0, a
  /// negative increment or moves-left), or for proof_volatility without the
  /// search's proof channel (issue 180).
  TimePlan begin(const Search& search, const TimeContext& ctx);
  /// Begins a decision on a plan made elsewhere (plan_decision): the root's
  /// visits now (the KLD's first comparison) and its total. The tests' entry;
  /// begin(search, ...) is this after plan_decision.
  void begin(const TimePlan& plan, std::vector<std::uint32_t> visits = {}, std::uint32_t root_visits = 0);
  /// Between batches: stop, go on, or extend. `elapsed`: seconds since begin
  /// by the caller's clock (it also measures the cost scale); NaN: the model
  /// clock. Throws std::logic_error outside a decision.
  TimeVerdict check(const Search& search, double elapsed);
  /// The same on a snapshot (the tests' entry; check() builds one).
  TimeVerdict check(const RootSnapshot& root, double elapsed);
  /// Ends the decision: its root's backed-up value is the next one's falling-
  /// eval reference.
  void end(const Search& search);
  void end(double root_q);
  /// A new game: the falling-eval reference forgotten (the scale kept).
  void new_game() noexcept;

  bool in_decision() const noexcept { return active_; }
  const TimePlan& plan() const noexcept { return plan_; }
  const TimeVerdict& last_verdict() const noexcept { return verdict_; }
  /// The cost scale (seconds per evaluation at cost_reference_boards) and s(N).
  double scale() const noexcept { return scale_; }
  double seconds_per_evaluation(double boards) const noexcept;
  /// The model clock's time for the decision so far (work x s(N)).
  double model_seconds() const noexcept;

 private:
  TimeParams params_;
  double scale_ = 0;
  bool observed_ = false;
  bool active_ = false;
  TimePlan plan_{};
  TimeVerdict verdict_{};
  std::uint64_t e0_ = 0, p0_ = 0;     // the search's counts at begin
  double target_ = 0;                  // the soft bound, in evaluations
  double extended_ = 0;                // the extensions taken, in evaluations
  std::uint32_t taken_ = 0;            // their kinds
  double work_ = 0;                    // at the last check
  double last_elapsed_ = 0;
  double last_work_ = 0;
  std::vector<std::uint32_t> kld_prev_;
  std::uint32_t kld_prev_visits_ = 0;
  double kld_gain_ = std::numeric_limits<double>::quiet_NaN();
  std::optional<double> last_value_;   // the previous decision's root value
};

/// A clock for run_timed (seconds; any origin).
class Clock {
 public:
  virtual ~Clock() = default;
  virtual double now() = 0;
  /// The steady clock itself: run_timed then gives the search its deadline
  /// (the rules engine's calls inside a batch stop at the hard bound).
  virtual bool steady() const noexcept { return false; }
};
/// std::chrono::steady_clock.
class WallClock final : public Clock {
 public:
  double now() override;
  bool steady() const noexcept override { return true; }
};

/// One timed decision's result.
struct TimedRun {
  TimePlan plan;
  TimeVerdict verdict;                         ///< the last check's (its stop: why it ended)
  StopReason search_stop = StopReason::Playouts;  ///< the search's own stop, where verdict.stop is Search
  std::uint64_t evaluations = 0;
  std::uint64_t playouts = 0;
  std::uint64_t batches = 0;
  double seconds = 0;                ///< by the clock, or the model clock
  std::uint32_t best_since = 0;      ///< the effort label's source (I15): Search::root_best_since
};

/// Runs `search` with `evaluator` for one decision under `tm`: begin, then
/// slices - each Search::run for batch_size playouts (a step-API driver that
/// runs the same slices builds the same tree) - until check says stop, then
/// end. `clock` null: the model clock (deterministic). A
/// steady clock gives the search the hard bound as its deadline for the run.
TimedRun run_timed(Search& search, Evaluator& evaluator, TimeManager& tm, const TimeContext& ctx,
                   Clock* clock = nullptr);

}  // namespace t5d::mcts

#endif  // T5D_TIMEMAN_HPP
