// Experiment 6b''s counting and verdicts (stage E9; design spec 15.6, the
// CHECKPOINT 4 block "6b'' - Guard and Hunt (pre-registered)"; register M17,
// H42, H43), apart from the tool (t5d_scan6bpp.cpp) so that tests hold them:
// from each root's outcome, the figures per board-count range and M17's
// verdicts.
//
// WHAT COUNTS. A proof counts only when E4 alone confirmed it
// (t5d/certificate.hpp): Guard's depth 1 completes at a root when Guard
// answered "safe to depth 1" (an exhaustive disproof - nothing to certify)
// or refuted the action at depth 1 with a CONFIRMED certificate; Hunt finds
// a truth mate only with a confirmed certificate. A proof the check left
// incomplete or undecided is reported apart and not counted. A proof E4
// REFUTES withholds every verdict: a prover that claims a false mate
// anywhere is trusted nowhere until it is fixed. Without the cross-check
// (--no-crosscheck) nothing is confirmed, and no verdict is given.
//
// THE THRESHOLDS (pre-registered): Guard passes a range where depth 1
// completes in >= 95% of its counted roots; Hunt where it finds >= 50% of
// the truth's mates there (a range with no truth mate has no verdict). Both
// in integer arithmetic: exactly 95%, exactly 50% pass. A Hunt verdict
// resting on fewer than kFewTruthMates truth mates is flagged "too few to
// read" - the pre-registered rule still gives it.

#ifndef T5D_TOOLS_SIXBPP_HPP
#define T5D_TOOLS_SIXBPP_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "t5d/certificate.hpp"
#include "t5d/guard.hpp"
#include "t5d/hunt.hpp"
#include "t5d/prover.hpp"

namespace t5d::tools::sixbpp {

/// The board-count ranges, never pooled (M12): < 50, 50-99, 100-199, 200+.
inline constexpr int kRanges = 4;
/// "< 50 boards", "50-99", "100-199", "200+".
const char* range_name(int range);
/// The range of a position of `boards` boards.
int range_of(int boards) noexcept;

/// Guard's threshold: depth 1 completes in >= 95% of `roots` (none: false).
bool guard_passes(std::size_t done, std::size_t roots) noexcept;
/// Hunt's threshold: >= 50% of the truth's mates found (none: false).
bool hunt_passes(std::size_t found, std::size_t truth_mates) noexcept;
/// Below this many truth mates a range's Hunt verdict is "too few to read"
/// (a reading, PLACEHOLDER: the pre-registration names no minimum).
inline constexpr std::size_t kFewTruthMates = 10;

/// A cross-check's outcome as the counting reads it.
enum class Check : std::uint8_t {
  None = 0,     // no proof, or not checked
  Confirmed,    // E4 alone confirmed it
  Unconfirmed,  // incomplete or undecided: not counted, reported apart
  Refuted,      // E4 refuted it: every verdict withheld
};
Check check_of(const CertificateCheck& c) noexcept;
const char* check_name(Check c) noexcept;

/// One root's outcome.
struct RootOutcome {
  int boards = 0;
  /// Guard ran at this root and is counted (its record's next action is
  /// legal and present).
  bool guard = false;
  int completed_depth = 0;
  bool refuted = false;
  int refutation_depth = 0;
  Check guard_check = Check::None;
  /// Hunt ran at this root.
  bool hunt = false;
  bool hunt_mate = false;
  Check hunt_check = Check::None;
  /// The ground truth has a mate here.
  bool truth_mate = false;
};

/// Guard's depth 1 completed, a refutation counting only when confirmed.
bool depth_one_done(const RootOutcome& r) noexcept;
/// Guard's depth 1 completed only by a refutation not confirmed.
bool depth_one_unconfirmed(const RootOutcome& r) noexcept;

struct GuardTally {
  std::size_t roots = 0;
  std::size_t depth_one = 0;    // depth_one_done
  std::size_t unconfirmed = 0;  // depth_one_unconfirmed (not in depth_one)
  std::size_t refuted = 0;      // refutations, any depth
  std::size_t refuted_one = 0;  // ... at depth 1
  std::map<int, std::size_t> completed;  // completed depth -> roots
};

struct HuntTally {
  std::size_t roots = 0;
  std::size_t mates = 0;              // Hunt's mates, confirmed or not
  std::size_t unconfirmed = 0;        // ... not confirmed (not counted)
  std::size_t truth = 0;              // the truth's mates
  std::size_t found = 0;              // truth mates Hunt found, confirmed
  std::size_t found_unconfirmed = 0;  // ... found but not confirmed (not counted)
  std::size_t extra = 0;              // confirmed mates the truth lacks
};

struct Tallies {
  /// Per range, and [kRanges]: pooled (information, never a verdict).
  GuardTally guard[kRanges + 1];
  HuntTally hunt[kRanges + 1];
  /// Proofs E4 refuted, Guard's and Hunt's: any withholds every verdict.
  std::size_t refuted_proofs = 0;
};
Tallies tally(const std::vector<RootOutcome>& roots);

enum class Verdict : std::uint8_t {
  Adopt = 0,
  NotAdopted,
  NoRoots,       // no root counted in the range
  NoTruthMates,  // Hunt: no truth mate to find
  NoTruth,       // Hunt: no ground truth given
  Withheld,      // a proof refuted by E4, or no cross-check
};
const char* verdict_name(Verdict v) noexcept;

Verdict guard_verdict(const Tallies& t, int range, bool crosschecked);
Verdict hunt_verdict(const Tallies& t, int range, bool have_truth, bool crosschecked);
/// Hunt's verdict in `range` rests on fewer than kFewTruthMates truth mates.
bool too_few_to_read(const Tallies& t, int range);

/// Every option, as name and value: the prover's, Guard's (its prover's
/// with Guard's overrides applied), Hunt's (likewise) - for the tool's
/// CONDITIONS and its JSON.
std::vector<std::pair<std::string, std::string>> prover_options(const ProverOptions& o);
std::vector<std::pair<std::string, std::string>> guard_options(const GuardOptions& o);
std::vector<std::pair<std::string, std::string>> hunt_options(const HuntOptions& o);

}  // namespace t5d::tools::sixbpp

#endif  // T5D_TOOLS_SIXBPP_HPP
