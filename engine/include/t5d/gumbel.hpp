// t5d/gumbel.hpp - register E12 (Gumbel root search), a prototype for
// self-play, off by default: the pure parts of the root's sampling and
// sequential halving (design spec §8.4; register E12; the search's hooks are
// t5d/mcts.hpp's SearchParams::gumbel_root).
//
// From the papers, as they describe it - no code from any engine or library
// (H29): Danihelka, Guez, Schrittwieser and Silver, "Policy improvement by
// planning with Gumbel" (ICLR 2022); Karnin, Koren and Somekh, "Almost
// optimal exploration in multi-armed bandits" (ICML 2013) - Sequential
// Halving; and the Gumbel-top-k trick (adding independent Gumbel(0, 1) noise
// to logits and keeping the k largest samples k actions without replacement
// from their softmax).
//
// At the root: add Gumbel noise g to every open edge's logit and keep the m
// largest - a sample without replacement - then spend n simulations on them by
// sequential halving: each phase visits its survivors equally, then keeps the
// better half by g + logit + sigma(q); play the survivor with the largest such
// score. sigma(q) = (c_visit + max_b N(b)) * c_scale * q, q the backed-up
// value mapped from W - L on [-1, 1] onto [0, 1] (spec §16, I21's form); an
// unvisited candidate's q is the mixed value of the root's evaluation and its
// visited children's (Danihelka et al.'s v_mix).
//
// PINNED FORMS (placeholders where the spec gives none; build-log spec
// issues for each - the engine block, 415 and up):
//  * The variate: g = -ln(-ln u), u = (floor(x / 2^12) + 0.5) * 2^-52 from a
//    64-bit x - 52 bits, so that u < 1 always (with 53, the largest u rounds
//    to 1.0 and g to +infinity): g lies in [-3.6038, 36.7368].
//  * Counter-based: an edge's variate depends on (stream, index) alone -
//    x = mix64(stream + 0x9E3779B97F4A7C15 * (index + 1)), mix64 SplitMix64's
//    finaliser - so a record's search seed, root key and candidate index give
//    it again, and closing one edge changes no other's.
//  * The stream: mix64(search_seed ^ mix64(root_key) ^ 0x47554D42454C3132)
//    (the salt is "GUMBEL12" in ASCII).
//  * The schedule: R = ceil(log2 m) phases for m >= 2 (one for m = 1); c_0 =
//    m survivors, c_{r+1} = ceil(c_r / 2); phase r gives each survivor
//    max(1, floor(n / (R * c_r))) more visits; the last phase has two;
//    simulations the rounding leaves over go on visiting the last survivors
//    equally (as more simulations than n do).
//  * The candidates: m_eff = min(m, n, open edges). An edge that closes (found
//    illegal at its first visit, a dead end) leaves its phase; in the first
//    phase the next open edge of the sample's order takes its place - still a
//    sample without replacement, over the open edges; if every candidate
//    closes, the open edges already shown are taken, else the next.

#ifndef T5D_GUMBEL_HPP
#define T5D_GUMBEL_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace t5d::gumbel {

/// Root::next's answers besides an edge: no candidate left, or wait - a
/// phase's last visits are still pending (their values decide the halving).
inline constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t kWait = std::numeric_limits<std::size_t>::max() - 1;

/// SplitMix64's finaliser (Steele, Lea and Flood, 2014).
std::uint64_t mix64(std::uint64_t x) noexcept;
/// The root's Gumbel stream: a function of the search's seed and the root's
/// key (see PINNED FORMS).
std::uint64_t stream_seed(std::uint64_t search_seed, std::uint64_t root_key) noexcept;
/// A Gumbel(0, 1) variate from 64 random bits (PINNED FORMS): finite.
double variate_of_bits(std::uint64_t bits) noexcept;
/// The variate of edge `index` in `stream`.
double variate(std::uint64_t stream, std::size_t index) noexcept;

/// The Gumbel-top-k order of a set of logits: every index, the open ones
/// (a finite logit) by g + logit descending - ties to the lower index - then
/// the closed ones (-infinity) in ascending order. Its first k are a sample
/// of k open indices without replacement from softmax(logits).
struct Sample {
  std::vector<std::size_t> order;
  std::vector<double> g;  ///< by index (not by order): g[i] = variate(stream, i)
};
/// Throws std::invalid_argument for a NaN or +infinity logit.
Sample sample_order(std::span<const double> logits, std::uint64_t stream);

/// One phase of sequential halving: its survivors and the visits each gets.
struct Phase {
  std::size_t considered = 0;
  std::uint64_t visits = 0;
};
/// The schedule for m candidates and n simulations (PINNED FORMS); m >= 1 and
/// n >= 1, else std::invalid_argument.
std::vector<Phase> halving_schedule(std::size_t m, std::uint64_t n);

/// A root edge as the halving reads it.
struct Action {
  double prior = 0;         ///< the root's prior (its logit: ln prior)
  std::uint32_t visits = 0;  ///< completed visits
  /// Visits in flight: the search passes the edge's virtual loss, which at
  /// a pick is its leaves awaiting evaluation alone - a collision under a
  /// Gumbel root ends the batch, so no collision path (whose virtual loss
  /// never becomes a visit) is counted.
  std::uint32_t pending = 0;
  double q = 0;             ///< backed-up W - L, the root player's view; read only where visits > 0
  bool open = false;        ///< may be taken: not pruned, not a dead end, prior > 0
};

/// sigma(q) = (c_visit + max_visits) * c_scale * (q + 1) / 2 - q's W - L on
/// [-1, 1] mapped onto [0, 1] (spec §16, I21's form; doubling c_scale is the
/// unmapped arm: the shift is common to every action).
double sigma(double q, std::uint32_t max_visits, double c_visit, double c_scale) noexcept;
/// Danihelka et al.'s v_mix over the open actions: v_net where none is visited
/// (or their prior mass is 0), else (v_net + N * sum_{visited} prior * q /
/// sum_{visited} prior) / (1 + N), N the visits in all.
double mixed_value(std::span<const Action> actions, double v_net) noexcept;

/// The largest c_visit and c_scale Root::make (and the search's validate)
/// accepts - PLACEHOLDERS, far above the paper's 50 and 1, chosen so that
/// sigma's (c_visit + max visits) x c_scale stays finite and far from overflow
/// for any visit count a search can reach (a child's visits are 32-bit:
/// (1e6 + 2^32) x 1e6 < 5e15). Unbounded, the product overflows to +infinity
/// and a lost candidate's sigma (q = -1: infinity x 0) is NaN.
inline constexpr double kMaxCVisit = 1e6;
inline constexpr double kMaxCScale = 1e6;

/// Root's parameters.
struct Config {
  std::size_t m = 16;
  std::uint64_t simulations = 0;
  double c_visit = 50.0;
  double c_scale = 1.0;
};

/// The halving at one root: its edges are the sample's order (edge i is the
/// order's i-th), g by edge. Movable; next() and best() allocate nothing.
class Root {
 public:
  Root() = default;
  /// open_count: the order's open prefix (the edges open at the draw; the
  /// rest never open). m_eff = min(m, simulations, open_count) candidates,
  /// edges 0..m_eff-1; m_eff 0: an inactive Root. Refills and the fallbacks
  /// look no further than the open prefix. Throws std::invalid_argument for a
  /// c_visit or c_scale that is not finite, is negative or is above its cap
  /// (kMaxCVisit, kMaxCScale).
  static Root make(std::vector<double> g_by_edge, std::size_t open_count, const Config& config);

  bool active() const noexcept { return active_; }
  /// The prefix of edges that are or were candidates (the root instantiates
  /// these, and lists them).
  std::size_t shown() const noexcept { return shown_; }
  std::size_t m() const noexcept { return m_; }
  std::size_t phase() const noexcept { return phase_; }
  std::uint64_t target() const noexcept { return target_; }
  const std::vector<Phase>& schedule() const noexcept { return schedule_; }
  const std::vector<std::size_t>& considered() const noexcept { return considered_; }
  double g(std::size_t edge) const noexcept { return edge < g_.size() ? g_[edge] : 0.0; }
  double c_visit() const noexcept { return c_visit_; }
  double c_scale() const noexcept { return c_scale_; }

  /// The edge to visit next, given every root edge's state (by edge) and the
  /// root's own evaluation v_net: the candidate with the fewest visits and
  /// pending ones (ties: the first) while it is below the phase's target -
  /// in the last phase, always - else kWait while a candidate's visits to
  /// the target are pending, else the phase ends (the better candidates by
  /// score kept) and the next one begins. kNone: no open edge at all.
  std::size_t next(std::span<const Action> actions, double v_net) noexcept;
  /// The move to play: the open candidate with the largest score (ties: the
  /// first); with none, the open shown edge with the largest; kNone. A NaN
  /// score (a NaN q or prior in `actions`) orders below every other, here and
  /// in the halving, so their orders stay strict weak orders.
  std::size_t best(std::span<const Action> actions, double v_net) const noexcept;

 private:
  /// g + ln prior + sigma(completed q), max visits and v_mix over `actions`;
  /// `edge` an open edge of `actions` (its callers, best and halve, pass only
  /// those).
  double score(std::span<const Action> actions, std::size_t edge, double v_mix, std::uint32_t max_visits) const noexcept;
  void refill(std::span<const Action> actions) noexcept;
  void halve(std::span<const Action> actions, double v_net) noexcept;

  bool active_ = false;
  std::vector<double> g_;
  std::vector<Phase> schedule_;
  std::vector<std::size_t> considered_;  ///< in edge order
  std::vector<std::pair<double, std::size_t>> scratch_;
  std::size_t m_ = 0;
  std::size_t open_count_ = 0;
  std::size_t shown_ = 0;
  std::size_t phase_ = 0;
  std::uint64_t target_ = 0;
  double c_visit_ = 50.0;
  double c_scale_ = 1.0;
};

/// A Gumbel root as tests and tools read it (Search::gumbel_view): rows
/// aligned with Search::root_children().
struct Row {
  double g = 0;
  double logit = 0;        ///< ln prior
  double completed_q = 0;  ///< q where visited, else v_mix
  double score = 0;        ///< g + logit + sigma(completed_q)
  bool considered = false;
};
struct RootView {
  bool drawn = false;
  std::size_t m = 0;
  std::size_t phase = 0;
  std::vector<Phase> schedule;
  double v_mix = 0;
  std::uint32_t max_visits = 0;
  std::vector<Row> rows;
};

}  // namespace t5d::gumbel

#endif  // T5D_GUMBEL_HPP
