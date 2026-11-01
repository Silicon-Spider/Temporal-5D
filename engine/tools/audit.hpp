// tools/audit.hpp - THE OFFLINE AUDIT LANE's tooling (engine stage E10, its
// second part; design spec 11.4's audit-lane row and "Whether shallow exact
// checks earn their time is measured, not assumed"; register H44, H42, H19,
// H23, I3, 6i).
//
//   "Samples self-play decisions - PV changes, demotions, final moves - and
//    runs exact proof search on them with large budgets, off the critical
//    path. Contradictions with the network feed the curriculum through
//    position selection (12.2's safe channel) and the adversarial suite."
//   "The offline audit lane reports how often the network's chosen move allows
//    a shallow forced mate, per board range."
//
// Built as the register says: tooling in the untrained build, the curriculum
// hooks as interfaces only. Three steps, each a function here and a
// subcommand of t5d-audit:
//
//  * THE DECISION LOG (the input): JSONL "t5d-decisions" version 1 - an
//    interim format until I15's records carry the events - written by t5d-play
//    --decision-log and Python's netplay --decision-log. A "game" line holds a
//    game's start (a named variant, or 5DFEN boards) and its actions (5dpgn's
//    move texts), so that every position is rebuilt by replaying - 5DFEN alone
//    loses the timelines' creators and branch origins. An "event" line names
//    its game, action and decision, the moves made in the action before it,
//    and its category's step:
//      final      - one per action: the action played, whole;
//      pv_change  - the root's most visited child changed: the new one, the
//                   old beside it;
//      demotion   - a child that was the most visited, or held at least
//                   `demotion_share` of the root's visits (spec 11.2 rung 2's
//                   "heavily visited", a placeholder), and was not chosen.
//  * SAMPLE: per category a rate, per (category, board range) a cap
//    (placeholders), seeded and deterministic: a job list ("job" lines, with
//    the games they need).
//  * RUN: per job, exact proof search with large budgets (placeholders):
//      Guard's question offline (H42: "kept as the audit lane's offline
//      question") on the job's action where it is complete - every opponent
//      action (the literal class), iterative deepening - a refutation's
//      certificate checked by E4 alone;
//      our forced mate at the action's turn start - the truth recorder's
//      method (t5d-hunt-truth: the literal class and the default class, the
//      shorter proof's certificate checked by E4) - once per turn start.
//    A CONTRADICTION goes to a CurriculumSink: "allowed_mate" (Guard refuted
//    the action - played, or promoted by a PV change - its certificate
//    confirmed), "missed_mate" (a confirmed mate at the turn start whose first
//    action is not the one played - inexact: the played action may mate too,
//    not checked: build-log spec issue 192), "abandoned_mate" (a demoted
//    child began the confirmed mate's first action). Nothing goes to any
//    solved store (H19, H23): there is no export code here.
//  * REPORT: per board range (6b's four, tools/sixbpp.hpp), the final moves
//    sampled, Guard's completed depths, the SHALLOW-MATE RATE - confirmed
//    refutations within `shallow_depth` among the finals Guard decided to
//    that depth - the undecided and the contradictions; counted as 6b''
//    counts: confirmed only, a refuted certificate withholds the verdict, too
//    few to read under sixbpp::kFewTruthMates finals; "weak at 200+" (11.4).
//
// The tools read engine-generated self-play logs only - never a path under
// oracles/ (AGENTS.md).

#ifndef T5D_TOOLS_AUDIT_HPP
#define T5D_TOOLS_AUDIT_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sixbpp.hpp"
#include "t5d/certificate.hpp"
#include "t5d/prover.hpp"
#include "t5d/selfplay.hpp"
#include "t5d/state.hpp"

namespace t5d::audit {

inline constexpr const char* kDecisionsFormat = "t5d-decisions";
inline constexpr const char* kResultsFormat = "t5d-audit-results";
inline constexpr const char* kContradictionsFormat = "t5d-contradictions";
inline constexpr int kFormatVersion = 1;

enum class Category : std::uint8_t { Final = 0, PvChange = 1, Demotion = 2 };
inline constexpr int kCategories = 3;
const char* category_name(Category c) noexcept;
std::optional<Category> category_of(std::string_view name) noexcept;

/// One decision of the search, as text: one move or a turn's moves (5dpgn's
/// export form, to_string(move, even)), or submitting.
struct Step {
  std::vector<std::string> moves;
  bool submit = false;
  friend bool operator==(const Step&, const Step&) = default;
};

/// A game of the log: its start and its actions.
struct GameLog {
  std::string record;   ///< the game's id (a content hash of its record)
  std::string variant;  ///< a named variant, or "custom" (then `start`)
  std::vector<std::string> start;  ///< custom: the start's 5DFEN boards
  int width = 0;        ///< custom: the board size
  int height = 0;
  std::uint64_t rules = 0;  ///< RuleOptions::encode()
  std::string split = "ply";  ///< "ply", "tree" or "flat"
  std::vector<std::vector<std::string>> actions;
};

/// One audited event (see the header comment).
struct Event {
  std::string record;
  std::uint32_t action = 0;
  std::uint32_t decision = 0;
  Category category = Category::Final;
  std::vector<std::string> prefix;  ///< the action's moves before the decision
  Step step;  ///< final: the whole action; pv_change: the new most visited; demotion: the demoted
  Step from;  ///< pv_change: the old (empty at a first establishment)
  int boards = 0;  ///< the decision root's board count
  std::uint32_t visits = 0;
  double q = 0;  ///< the step's backed-up value (the root player's view)
  std::uint32_t from_visits = 0;
  double from_q = 0;
  std::uint32_t root_visits = 0;
  double root_q = 0;
  double peak_share = 0;
};

/// A sampled event.
struct Job {
  std::uint64_t id = 0;  ///< its sampling key (seeded)
  int range = 0;         ///< sixbpp::range_of(event.boards)
  Event event;
};

/// A log: games, and events or jobs.
struct Log {
  std::vector<GameLog> games;
  std::vector<Event> events;
  std::vector<Job> jobs;
  const GameLog* game(std::string_view record) const;
};

std::string game_json(const GameLog& g);
std::string event_json(const Event& e);
std::string job_json(const Job& j);
/// A log's text: every line a game, an event or a job of format
/// "t5d-decisions" version 1 (blank lines skipped); std::invalid_argument for
/// a malformed line, another format or version, or an event or job whose game
/// is missing.
Log read_log(std::string_view text);
std::string log_text(const Log& log);

struct EventOptions {
  /// A child holding at least this share of the root's visits, not chosen, is
  /// demoted (placeholder; 11.2 rung 2's "heavily visited").
  double demotion_share = 0.3;
  /// The first establishment of a most visited child is a PV change too.
  bool first_pv = false;
};

/// A game played with PlayParams::decision_log as a log: its game line and
/// its events. `variant`: the start's variant name, or "custom" (the start's
/// boards written). Throws std::invalid_argument if the game has no decision
/// log.
Log log_of_game(const mcts::GameRecord& game, std::string_view variant, std::string_view record_id,
                const EventOptions& options = {});

/// A game's start; its turn start before `action` (the actions before it
/// replayed); and an event's decision state (its prefix played too).
GameState start_of(const GameLog& g);
GameState turn_start_of(const GameLog& g, std::uint32_t action);
/// The generated move whose export text is `text` (std::invalid_argument if
/// none).
Move move_of_text(const GameState& s, std::string_view text);

// ---- sampling ---------------------------------------------------------------------------

struct SampleOptions {
  std::uint64_t seed = 1;
  /// Per category (Final, PvChange, Demotion): the share sampled
  /// (placeholders).
  std::array<double, kCategories> rate{1.0, 0.25, 1.0};
  /// Per (category, board range): at most this many (placeholder).
  std::size_t cap = 1000;
};

/// Seeded and deterministic: an event is kept when its key - a hash of the
/// seed and the event's identity - falls under its category's rate, then the
/// lowest keys up to the cap per (category, range); the jobs in the events'
/// order. The log keeps the games the jobs need.
Log sample(const Log& events, const SampleOptions& options);

// ---- running ----------------------------------------------------------------------------

struct RunOptions {
  /// Guard offline: depths 1.. (the opponent's actions), one node budget for
  /// them all, the literal class (placeholders: "large budgets").
  int guard_max_depth = 2;
  std::uint64_t guard_nodes = 2'000'000;
  /// Our mate at the turn start: the truth recorder's method, per class
  /// (placeholders).
  int mate_depth = 5;
  std::uint64_t mate_nodes = 1'000'000;
  std::size_t tree_cap = std::size_t{2} << 20;
  int threads = 1;  ///< the provers' (1: deterministic)
  CertificateCheckOptions check;
  /// TESTS ONLY: applied to each certificate before E4 checks it (Guard's
  /// refutation, our mate's proof) - the seam that feeds the runner a
  /// certificate E4 rejects. Null (the default): none.
  std::function<void(std::shared_ptr<const ProofCertificate>&)> certificate_hook_for_tests;
};

struct GuardOutcome {
  bool asked = false;
  std::string why;  ///< not asked: why
  int completed_depth = 0;
  bool refuted = false;
  int refutation_depth = 0;
  tools::sixbpp::Check check = tools::sixbpp::Check::None;
  std::uint64_t nodes = 0;
  double seconds = 0;
};

struct MateOutcome {
  bool asked = false;
  ProofStatus status = ProofStatus::Unknown;  ///< the better class's
  int distance = 0;
  std::vector<std::string> first_action;
  tools::sixbpp::Check check = tools::sixbpp::Check::None;
  std::uint64_t nodes = 0;
  double seconds = 0;
};

/// A contradiction between the exact search and the network's search.
struct Contradiction {
  std::string kind;  ///< "allowed_mate", "missed_mate", "abandoned_mate"
  Event event;
  int distance = 0;          ///< half-turns of the proven mate
  std::string certificate;   ///< its check by E4 alone ("confirmed")
  double search_q = 0;       ///< the search's value of the step (the root player's view)
};

std::string contradiction_json(const Contradiction& c);

/// The curriculum's hook (H44: "contradictions feed the curriculum via
/// position selection and the adversarial suite") - AN INTERFACE ONLY: no
/// consumer is built (position selection and the adversarial suite are
/// training-time).
class CurriculumSink {
 public:
  virtual ~CurriculumSink() = default;
  virtual void emit(const Contradiction& c) = 0;
};

/// The one implementation: JSONL "t5d-contradictions" v1, a line each.
class JsonlCurriculumSink final : public CurriculumSink {
 public:
  explicit JsonlCurriculumSink(std::ostream& out) : out_(out) {}
  void emit(const Contradiction& c) override;
  std::size_t emitted() const noexcept { return emitted_; }

 private:
  std::ostream& out_;
  std::size_t emitted_ = 0;
};

struct JobResult {
  Job job;
  GuardOutcome guard;
  MateOutcome mate;
  std::vector<std::string> contradictions;  ///< their kinds
};

std::string result_json(const JobResult& r);
std::vector<JobResult> read_results(std::string_view text);

/// Runs a sampled log's jobs (see the header comment): each result, the
/// contradictions emitted to `sink` (may be null) as found. Our-mate answers
/// are shared by the jobs of one turn start.
std::vector<JobResult> run(const Log& jobs, const RunOptions& options, CurriculumSink* sink);

// ---- the report -------------------------------------------------------------------------

struct RangeReport {
  std::size_t finals = 0;       ///< final moves sampled
  std::size_t guard_asked = 0;
  std::size_t decided = 0;      ///< Guard completed shallow_depth, or a confirmed refutation within it
  std::size_t shallow = 0;      ///< ... of which a confirmed refutation within shallow_depth
  std::size_t undecided = 0;    ///< asked, neither
  std::size_t withheld = 0;     ///< a refutation E4 refuted: the verdict withheld
  std::size_t unconfirmed = 0;  ///< a refutation E4 left incomplete or undecided
  std::map<int, std::size_t> completed;  ///< Guard's completed depth -> finals
  std::size_t allowed_mates = 0;         ///< every category's confirmed refutations
  std::size_t missed_mates = 0;
  std::size_t abandoned_mates = 0;
  bool too_few = false;  ///< fewer than sixbpp::kFewTruthMates finals
  double shallow_rate() const noexcept {
    return decided == 0 ? 0.0 : static_cast<double>(shallow) / static_cast<double>(decided);
  }
};

struct Report {
  int shallow_depth = 1;
  std::array<RangeReport, tools::sixbpp::kRanges> ranges{};
  std::size_t jobs = 0;
};

/// `shallow_depth`: Guard's depth (the opponent's actions) counted shallow
/// (placeholder 1 - their mate in one; the completed depths are reported too).
Report make_report(const std::vector<JobResult>& results, int shallow_depth = 1);
std::string report_text(const Report& r);
std::string report_json(const Report& r);

}  // namespace t5d::audit

#endif  // T5D_TOOLS_AUDIT_HPP
