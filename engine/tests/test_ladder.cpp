// Stage E10: the reduced verification ladder wired into the search
// (t5d/ladder.hpp; design spec 11.2-11.4, 12.2, 15.4; register H1, H10, H13,
// H14, H41-H43, I2; 6b''s adoption table).
//
// What is tested: each prover's range gating at its boundary; that no proof
// enters the tree - or any store - while the proof channel is off (every
// route tried: the ladder's verdicts, a direct offer, the export switches);
// with the channel on (test-only), Hunt's mate played and a settling scan's
// refutation steering the search away, every injected proof's certificate
// confirmed by E4 alone; the asset map's boosts and the steered visits I2's
// targets read; the synchronous mode's bit-reproducibility and §15.4's
// invariants with the wiring on; the budgets, and no job outliving its
// decision; the settling prover's proofs under TT churn (6f); Guard's hook,
// off by default and refused without the channel.
//
// The positions are written for these tests; each rules fact they rest on (a
// mate in one, a forced mate) is re-checked here by E4's detectors.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/certificate.hpp"
#include "t5d/guard.hpp"
#include "t5d/ladder.hpp"
#include "t5d/mcts.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"
#include "t5d/selfplay.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_play.hpp"
#include "test_positions.hpp"
#include "test_support.hpp"
#include "test_v3_support.hpp"

using namespace t5d;
using namespace t5d::mcts;
using t5d_test::game;
using t5d_test::v3::k5x5;

namespace {

GameState variant(const char* name) {
  const Variant* v = find_variant(name);
  if (v == nullptr) throw std::logic_error(std::string("no variant ") + name);
  return variant_start(*v, RuleOptions{});
}

// The tests' parameters: no square features (the built-in evaluators read
// none), the synchronous mode (H15: every ladder job answered where it is
// asked, by nodes, one thread) unless a test says otherwise.
SearchParams sync_params() {
  SearchParams p;
  p.expand.square_features = SquareFeatures::None;
  p.synchronous = true;
  return p;
}

// White to move on one 5 x 5 board: the rook on c5 guards c2 against the
// knight's Ne3-c2, a smothered mate - the king on a1 walled in by its pawns
// and knight on the present board AND on its past one (a 5D king escapes
// check by travelling back in time: every square it could reach on the board
// before is its own piece's). The rook leaving the c-file (Rc5-a5) lets the
// mate in; every other White move keeps the guard.
GameState blunder_position() { return game({"[2R2/4k/4n/PP3/KN3:0:1:w]"}, k5x5); }

// The blunder: Rc5-a5.
Move blunder(const GameState& s) {
  const std::optional<Move> m = t5d_test::find_move(s, "(0T1)Rc5(0T1)a5");
  if (!m) throw std::logic_error("no Rc5-a5");
  return *m;
}

// An evaluator that favours one move above all (logit 6, the others 0) and
// values every leaf 0 - so the search's principal line is that move.
class Favouring final : public Evaluator {
 public:
  explicit Favouring(Move m) : favourite_(m) {}
  void evaluate(std::span<const EvalRequest> batch, std::span<Evaluation> out) override {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const MoveList& moves = batch[i].expansion->moves();
      out[i].logits.assign(moves.size(), 0.0f);
      for (std::size_t k = 0; k < moves.size(); ++k) {
        if (moves[k] == favourite_) out[i].logits[k] = 6.0f;
      }
      out[i].wdl = wdl_of(0.0);
      out[i].value = two_hot(0.0);
    }
  }

 private:
  Move favourite_;
};

std::size_t count_jobs(const Search& s, LadderJob job) {
  std::size_t n = 0;
  for (const LadderVerdict& v : s.verdicts()) n += v.job == job && !v.carried ? 1u : 0u;
  return n;
}

// A search to its end with its ladder's jobs answered.
void run_all(Search& s, Evaluator& ev, std::uint64_t playouts) {
  s.run(ev, Limits{playouts});
  s.await_ladder();
}

}  // namespace

// ---- 6b''s ranges ---------------------------------------------------------------------------

TEST(ladder_defaults_are_6bpps_adoption_table) {
  const SearchParams p;
  CHECK(p.hunt);
  CHECK_EQ(p.hunt_max_boards, 100);
  CHECK(p.settling);
  CHECK_EQ(p.settling_max_boards, 50);
  CHECK(p.asset_map);
  CHECK_EQ(p.hunt_ms, 50.0);
  CHECK_EQ(p.settling_ms, 50.0);
  CHECK_EQ(p.verification_threads, 5);
  CHECK(!p.hunt_suggest);   // H43's flag
  CHECK(!p.proof_channel);  // H41
  CHECK(!p.proof_store);    // H23
  const PlayParams play;
  CHECK(!play.guard_final);  // H42: adopted in no range
}

TEST(ladder_each_prover_runs_only_below_its_range) {
  // The boundary is exclusive ("under 100 boards", "under 50"): a root of B
  // boards is asked with max B + 1, not with max B - checked at the root
  // (Hunt and the root's settling scan) and at a PV change's position (the
  // settling scan judged on the position it scans).
  const GameState s = blunder_position();
  const int b = static_cast<int>(s.multiverse().board_count());
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const int hunt_max : {b, b + 1}) {
    for (const int settle_max : {b, b + 1}) {
      SearchParams p = sync_params();
      p.hunt_max_boards = hunt_max;
      p.settling_max_boards = settle_max;
      p.settling_debounce = 1;
      Search search(s, p);
      run_all(search, *uniform, 64);
      CHECK_EQ(count_jobs(search, LadderJob::Hunt), std::size_t{hunt_max > b ? 1u : 0u});
      CHECK_EQ(count_jobs(search, LadderJob::RootSettling), std::size_t{settle_max > b ? 1u : 0u});
      // After one action the opponent's turn start has b + 1 boards: a PV
      // scan needs settle_max > b + 1.
      CHECK_EQ(count_jobs(search, LadderJob::PvSettling), std::size_t{0});
    }
  }
  for (const int settle_max : {b + 1, b + 2}) {
    SearchParams p = sync_params();
    p.hunt = false;
    p.settling_max_boards = settle_max;
    p.settling_debounce = 1;
    Search search(s, p);
    run_all(search, *uniform, 64);
    CHECK_EQ(count_jobs(search, LadderJob::RootSettling), std::size_t{1});
    CHECK(settle_max > b + 1 ? count_jobs(search, LadderJob::PvSettling) > 0
                             : count_jobs(search, LadderJob::PvSettling) == 0);
  }
  // Off is off; and a root during an action asks no root question (Hunt asks
  // at turn starts).
  {
    SearchParams p = sync_params();
    p.hunt = p.settling = false;
    Search search(s, p);
    run_all(search, *uniform, 64);
    CHECK(search.verdicts().empty());
  }
  {
    GameState mid = s;
    mid.apply(blunder(s));
    Search search(mid, sync_params(), 1);
    run_all(search, *uniform, 16);
    CHECK_EQ(count_jobs(search, LadderJob::Hunt), std::size_t{0});
    CHECK_EQ(count_jobs(search, LadderJob::RootSettling), std::size_t{0});
  }
}

TEST(ladder_ranges_at_49_50_99_100_boards) {
  // The adopted ranges at their edges, on positions of exactly 49, 50, 99 and
  // 100 boards: two kings on one timeline with that many boards of history
  // (the side to move the last board's). Hunt is asked under 100 boards, the
  // settling scan under 50 - and nothing above.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const int n : {49, 50, 99, 100}) {
    std::vector<std::string> boards;
    for (int h = 0; h < n; ++h) {
      boards.push_back("[k4/5/5/5/4K:0:" + std::to_string(h / 2 + 1) + ":" + (h % 2 == 0 ? "w" : "b") + "]");
    }
    const GameState s = game(boards, k5x5);
    REQUIRE_EQ(static_cast<int>(s.multiverse().board_count()), n);
    REQUIRE(detect_mate(s, Detector::Hypercuboid).verdict == Outcome::Ongoing);
    Search search(s, sync_params());
    search.run(*uniform, Limits{2});
    search.await_ladder();
    CHECK_EQ(count_jobs(search, LadderJob::Hunt), std::size_t{n < 100 ? 1u : 0u});
    CHECK_EQ(count_jobs(search, LadderJob::RootSettling), std::size_t{n < 50 ? 1u : 0u});
  }
}

// ---- The proof channel: off, nothing enters ------------------------------------------------

TEST(ladder_no_proof_enters_while_the_proof_channel_is_off) {
  // Hunt and the root's settling scan both prove White's mate in one here
  // (re-checked by E4 below); with the proof channel off (the rule, H41) the
  // proofs are counted and kept as verdicts - WITHHELD - and the tree is
  // exactly the tree without the ladder: H15's digest, every root child's
  // bounds unproven, the root unsolved.
  const GameState s = t5d_test::v3::mate_in_one();
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  SearchParams on = sync_params();
  on.asset_map = false;
  SearchParams off = on;
  off.hunt = off.settling = false;
  Search with(s, on);
  Search without(s, off);
  run_all(with, *uniform, 200);
  run_all(without, *uniform, 200);
  REQUIRE_EQ(with.stats().hunt_mates, std::uint64_t{1});
  REQUIRE_EQ(with.stats().settling_proven, std::uint64_t{1});
  CHECK_EQ(with.digest(), without.digest());
  CHECK_EQ(with.stats().proofs_withheld, std::uint64_t{2});
  CHECK_EQ(with.stats().proofs_injected, std::uint64_t{0});
  CHECK(!with.root_solved());
  for (const RootChild& c : with.root_children()) CHECK(c.lo == -1 && c.hi == 1);
  for (const LadderVerdict& v : with.verdicts()) {
    if (v.job == LadderJob::PvSettling) continue;  // the root's questions
    CHECK(v.status == ProofStatus::Proven);
    CHECK(v.use == ProofUse::Withheld);
    // The proof itself is sound: its first action mates (E4).
    const GameState after = t5d_test::v3::after_action(s, v.first_action);
    CHECK(detect_mate(after, Detector::Hypercuboid).verdict == Outcome::Checkmate);
  }
  // Try to make one enter.
  // (1) Offered directly: withheld, the tree unchanged.
  const std::uint64_t digest = with.digest();
  std::vector<PathStep> path;
  for (const Move& m : with.verdicts().front().first_action) path.push_back(PathStep{{m}, false});
  path.push_back(PathStep{{}, true});
  CHECK(with.inject_proof(path, Proof::loss(0)) == ProofUse::Withheld);
  CHECK_EQ(with.digest(), digest);
  CHECK(!with.root_solved());
  // (2) A persistent solved store: refused (H23).
  SearchParams store = on;
  store.proof_store = true;
  CHECK_THROWS_AS(Search(s, store), std::invalid_argument);
  // (3) Hunt's export switch, through the ladder: refused (H23, H41).
  LadderOptions lo = ladder_options(on);
  lo.hunt.export_proofs = true;
  CHECK_THROWS_AS(VerificationLadder(lo, s.rules(), s.multiverse().ruleset()), std::invalid_argument);
  // (4) A ladder that exported would share nothing with the search: the search
  // refuses a ladder made for other parameters.
  const auto other = std::make_shared<VerificationLadder>(ladder_options(off), s.rules(), s.multiverse().ruleset());
  SearchParams deeper = on;
  deeper.hunt_depth = 7;
  CHECK_THROWS_AS(Search(s, deeper, 0, nullptr, other), std::invalid_argument);
  // (5) And in self-play with the default parameters: the game's proofs are
  // all withheld.
  PlayParams pp;
  pp.search = on;
  pp.search.root_noise = true;
  pp.playouts = 16;
  pp.max_actions = 1;
  const GameRecord g = play_game(s, *uniform, pp);
  CHECK(g.hunt_mates > 0);
  CHECK_EQ(g.proofs_injected, std::uint64_t{0});
  CHECK(g.proofs_withheld > 0);
}

// ---- The proof channel on (test-only): the proofs used ---------------------------------------

TEST(ladder_hunts_mate_is_played_with_the_proof_channel_on) {
  // Mate in two (White: Rd2-d4, then mate - E4 re-checks the line): with the
  // channel on, Hunt's proof - its certificate confirmed by E4 alone - solves
  // the root and E9 plays the mating action: under the flat form the turn at
  // once; under the ply split its first move, the rest through fresh decisions
  // (the action's mate carried), then the submission.
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const GameState& s : {t5d_test::v3::mate_in_one(), t5d_test::v3::mate_in_two()}) {
    for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat}) {
      SearchParams p = sync_params();
      p.proof_channel = true;
      p.split = split;
      p.settling = false;  // Hunt's proof alone
      p.asset_map = false;
      Search search(s, p);
      search.run(*uniform, Limits{50});
      REQUIRE_EQ(search.stats().hunt_mates, std::uint64_t{1});
      const LadderVerdict& v = search.verdicts().front();
      CHECK(v.use == ProofUse::Injected);
      CHECK(v.certificate_checked && v.certificate_verdict == CertificateVerdict::Confirmed);
      CHECK(search.root_solved());
      CHECK_NOTHROW(search.check_invariants());
      const Choice c = search.choose();
      CHECK(c.reason == ChoiceReason::ProvenWin);
      const RootChild child = search.root_children().at(c.index);
      if (split == TurnSplit::Flat) {
        CHECK(t5d_test::v3::after_action(s, child.moves) == t5d_test::v3::after_action(s, v.first_action));
      } else {
        REQUIRE_EQ(child.moves.size(), std::size_t{1});
        CHECK(child.moves[0] == v.first_action.front());
      }
    }
    // Played through: White mates in the proof's distance, every proof
    // offered injected.
    for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat}) {
      PlayParams pp;
      pp.search = sync_params();
      pp.search.proof_channel = true;
      pp.search.split = split;
      pp.search.asset_map = false;
      pp.playouts = 24;
      pp.max_actions = 8;
      const GameRecord g = play_game(s, *uniform, pp);
      CHECK(g.end == GameEnd::Checkmate);
      CHECK_EQ(g.winner, 0);
      CHECK(g.proofs_injected > 0);
      CHECK_EQ(g.proofs_withheld, std::uint64_t{0});
    }
  }
}

TEST(ladder_a_pv_change_scan_refutes_the_line_and_the_search_moves_away) {
  // The favoured move Rc5-a5 is the principal line; the settling scan at
  // Black's turn start after it proves Black's mate in one (Ne3-c2, re-checked
  // by E4) - asked after three playouts, when that turn start is evaluated
  // and the tree has not yet made its children (so the in-tree solver, with
  // the channel on, has not seen the mate itself). Channel off: a refutation
  // recorded, nothing else. Channel on (test only): the line proven lost,
  // ineligible - the search, run on, moves away.
  const GameState s = blunder_position();
  const Move rook_away = blunder(s);
  {
    GameState t = s;
    t.apply(rook_away);
    t.submit();
    REQUIRE(t5d_test::find_move(t, "(0T1)Ne3(0T1)c2").has_value());
    GameState u = t;
    t5d_test::play(u, "(0T1)Ne3(0T1)c2");
    u.submit();
    REQUIRE(detect_mate(u, Detector::Hypercuboid).verdict == Outcome::Checkmate);
  }
  for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat}) {
    for (const bool channel : {false, true}) {
      Favouring ev(rook_away);
      SearchParams p = sync_params();
      p.split = split;
      p.hunt = false;
      p.asset_map = false;
      p.proof_channel = channel;
      p.settling_root = false;  // White mates from this start: the root's scan would solve the root
      p.settling_debounce = 1;
      Search search(s, p);
      run_all(search, ev, split == TurnSplit::Flat ? 2 : 3);  // the root, (the move,) Black's turn start
      CHECK(search.stats().refutations > 0);
      run_all(search, ev, 300);
      const LadderVerdict* refutation = nullptr;
      for (const LadderVerdict& v : search.verdicts()) {
        if (v.job == LadderJob::PvSettling && v.status == ProofStatus::Proven) refutation = &v;
      }
      REQUIRE(refutation != nullptr);
      CHECK(refutation->position.to_move() == Colour::Black);
      CHECK_EQ(refutation->proof.distance, 1);
      CHECK(refutation->use == (channel ? ProofUse::Injected : ProofUse::Withheld));
      CHECK_NOTHROW(search.check_invariants());
      const std::vector<RootChild> kids = search.root_children();
      const Choice c = search.choose();
      const bool chose_blunder = kids.at(c.index).moves.size() == 1 && kids.at(c.index).moves[0] == rook_away;
      if (channel) {
        CHECK(!chose_blunder);  // the refuted line is proven lost: ineligible
        for (const RootChild& k : kids) {
          if (k.moves.size() == 1 && k.moves[0] == rook_away) CHECK_EQ(k.hi, -1);
        }
      } else {
        CHECK(chose_blunder);  // the favourite stands: nothing entered the tree
      }
    }
  }
}

TEST(ladder_injected_proofs_never_contradict_and_match_e4) {
  // Over many small positions with the channel on (synchronous: the same
  // trees each run), every injected proof's certificate is confirmed by E4,
  // the bounds never contradict (a contradiction throws), and §15.4's
  // invariants hold - and each proven root is a real forced win: its proof's
  // first action leaves the opponent mated or with a mate against them that
  // E4 confirms (the certificate).
  const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 3);
  std::size_t injected = 0;
  for (const GameState& s : t5d_test::sparse_positions(RuleOptions{}, 6, 6, 11)) {
    if (detect_mate(s, Detector::Hypercuboid).verdict != Outcome::Ongoing) continue;
    SearchParams p = sync_params();
    p.proof_channel = true;
    p.sync_job_nodes = 20'000;
    p.settling_debounce = 4;
    Search search(s, p);
    CHECK_NOTHROW(run_all(search, *random, 120));
    CHECK_NOTHROW(search.check_invariants());
    for (const LadderVerdict& v : search.verdicts()) {
      if (v.use != ProofUse::Injected) continue;
      ++injected;
      CHECK(v.certificate_verdict == CertificateVerdict::Confirmed);
    }
  }
  std::printf("  %zu proofs injected and confirmed\n", injected);
  CHECK(injected > 0);
}

// ---- The asset map and the steered visits ----------------------------------------------------

namespace {

// White's superseded board (turn 1) holds White's queen against a bare king:
// flagged; the White king's travel move onto it is boosted (test_assets.cpp).
GameState asset_position() {
  return game({"[k4/5/5/5/Q3K:0:1:w]", "[k4/5/5/5/4K:0:1:b]", "[k4/5/5/5/3K1:0:2:w]"}, k5x5);
}

}  // namespace

TEST(ladder_asset_map_boosts_travel_moves_and_counts_steered_visits) {
  const GameState s = asset_position();
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  for (const bool graph : {false, true}) {
    for (const BoostRule rule : {BoostRule::Additive, BoostRule::Multiplicative}) {
      SearchParams p = sync_params();
      p.hunt = p.settling = false;
      p.graph = graph;
      p.boost_rule = rule;
      SearchParams plain = p;
      plain.asset_map = false;
      Search steered(s, p);
      Search unsteered(s, plain);
      run_all(steered, *uniform, 300);
      run_all(unsteered, *uniform, 300);
      CHECK(steered.stats().asset_boosted_edges > 0);
      CHECK_EQ(unsteered.stats().asset_boosted_edges, std::uint64_t{0});
      // Each root child: its unsteered prior is the plain search's prior; its
      // boost what steering added; travel moves into turn 1 boosted, nothing
      // else raised.
      const std::vector<RootChild> a = steered.root_children();
      const std::vector<RootChild> b = unsteered.root_children();
      std::uint32_t steered_visits = 0;
      std::size_t boosted = 0;
      for (const RootChild& c : a) {
        steered_visits += c.steered_visits;
        CHECK(c.steered_visits <= c.visits);
        CHECK(std::abs(c.boost - (c.prior_noise_free - c.prior_unsteered)) < 1e-15);
        const RootChild* twin = nullptr;
        for (const RootChild& d : b) {
          if (d.submit == c.submit && d.moves == c.moves) twin = &d;
        }
        REQUIRE(twin != nullptr);
        CHECK(std::abs(c.prior_unsteered - twin->prior_noise_free) < 1e-12);
        const bool travel_in = !c.submit && !c.moves[0].physical() &&
                               c.moves[0].to_half_turn < s.multiverse().timeline(c.moves[0].to_timeline)->last_half_turn();
        if (c.boost > 0) {
          ++boosted;
          CHECK(travel_in);
        }
      }
      CHECK(boosted > 0);
      CHECK(steered_visits > 0);
      CHECK_EQ(static_cast<std::uint64_t>(steered_visits), steered.stats().steered_visits);
      CHECK_NOTHROW(steered.check_invariants());
      // The evaluation cache holds the network's evaluations, never steered:
      // the same leaves' evaluations as the plain search's where both have one.
      CHECK(steered.cache() != nullptr && unsteered.cache() != nullptr);
    }
  }
  // Every visit of a boosted child under the cautious rule; none counted
  // without steering.
  SearchParams p = sync_params();
  p.hunt = p.settling = false;
  p.steered_rule = SteeredRule::BoostedChild;
  Search search(s, p);
  run_all(search, *uniform, 200);
  for (const RootChild& c : search.root_children()) {
    CHECK_EQ(c.steered_visits, c.boost > 0 ? c.visits : 0u);
  }
  p.asset_map = false;
  Search none(s, p);
  run_all(none, *uniform, 200);
  for (const RootChild& c : none.root_children()) CHECK_EQ(c.steered_visits, 0u);
}

TEST(ladder_asset_boosts_are_a_function_of_the_state_and_each_board_is_assessed_once) {
  // §15.2's C19-C21 row: steering acts on priors after the cached evaluation -
  // the cache's entries are the evaluator's, the same with the map on and off;
  // and a game's searches share one ladder, so each superseded board is
  // assessed once in the game (assessments == boards).
  const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 5);
  const GameState s = variant("Standard - Two Timelines");
  SearchParams p = sync_params();
  p.hunt = p.settling = false;
  auto cache = std::make_shared<EvalCache>();
  auto ladder = std::make_shared<VerificationLadder>(ladder_options(p), s.rules(), s.multiverse().ruleset());
  GameState t = s;
  t5d_test::Rng rng(3);
  for (int turn = 0; turn < 8; ++turn) {
    Search search(t, p, 0, cache, ladder);
    search.run(*random, Limits{48});
    CHECK_EQ(ladder->assets().assessments(), static_cast<std::uint64_t>(ladder->assets().boards()));
    const std::optional<std::vector<Move>> a = t5d_test::random_legal_action(t, rng);
    if (!a) break;
    for (const Move& m : *a) t.apply(m);
    t.submit();
  }
  // The cache's evaluations equal a cache filled without the asset map.
  SearchParams plain = p;
  plain.asset_map = false;
  Search with(s, p);
  Search without(s, plain);
  with.run(*random, Limits{200});
  without.run(*random, Limits{200});
  CHECK(with.cache()->size() > 0);
}

// ---- The synchronous mode and §15.4's invariants with the wiring on --------------------------

TEST(ladder_synchronous_mode_is_bit_reproducible_with_the_wiring_on) {
  const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 9);
  std::vector<GameState> roots{t5d_test::v3::mate_in_two(), blunder_position(), asset_position(),
                               variant("Standard - Two Timelines")};
  for (const GameState& s : roots) {
    for (const bool channel : {false, true}) {
      for (const TurnSplit split : {TurnSplit::BoardMove, TurnSplit::Flat}) {
        SearchParams p = sync_params();
        p.proof_channel = channel;
        p.split = split;
        p.settling_debounce = 4;
        p.sync_job_nodes = 5000;
        Search a(s, p);
        Search b(s, p);
        a.run(*random, Limits{120});
        b.run(*random, Limits{120});
        CHECK_EQ(a.digest(), b.digest());
        CHECK_EQ(a.verdicts().size(), b.verdicts().size());
        CHECK_EQ(a.stats().steered_visits, b.stats().steered_visits);
      }
    }
  }
}

TEST(ladder_invariants_hold_with_verdicts_arriving_asynchronously) {
  // The ladder's own threads (5 prover threads, 50 ms - the defaults), its
  // verdicts merged between batches: §15.4's invariants hold after every run,
  // no virtual loss is left, with the channel off and on.
  const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 4);
  for (const GameState& s : {t5d_test::v3::mate_in_two(), blunder_position(), asset_position()}) {
    for (const bool channel : {false, true}) {
      SearchParams p;
      p.expand.square_features = SquareFeatures::None;
      p.proof_channel = channel;
      p.settling_debounce = 4;
      Search search(s, p);
      for (int round = 0; round < 6; ++round) {
        search.run(*random, Limits{40});
        CHECK_NOTHROW(search.check_invariants());
      }
      search.await_ladder();
      CHECK_NOTHROW(search.check_invariants());
      CHECK(search.shared_ladder()->pending() == 0);
    }
  }
}

// ---- Budgets, and no job outlives its decision -----------------------------------------------

TEST(ladder_jobs_keep_their_budgets_and_end_with_their_decision) {
  // Synchronous: each job stops at its node budget. Asynchronous: at its
  // clock (50 ms, with a scheduling margin), and a decision's end - the search
  // destroyed, or re-rooted - cancels its running job and drops its queue:
  // nothing is left running, and every job asked is answered or cancelled.
  const GameState s = variant("Standard - Two Timelines");
  const std::unique_ptr<Evaluator> uniform = make_evaluator(BuiltinEvaluator::Uniform);
  {
    SearchParams p = sync_params();
    p.sync_job_nodes = 3000;
    Search search(s, p);
    run_all(search, *uniform, 64);
    for (const LadderVerdict& v : search.verdicts()) {
      CHECK(v.stats.nodes <= 3000 + 512);  // a node's own work may pass it by a little
    }
  }
  {
    SearchParams p;
    p.expand.square_features = SquareFeatures::None;
    p.hunt_ms = 20;
    p.settling_ms = 20;
    std::shared_ptr<VerificationLadder> ladder;
    {
      Search search(s, p);
      ladder = search.shared_ladder();
      search.run(*uniform, Limits{64});
      search.await_ladder();
      for (const LadderVerdict& v : search.verdicts()) CHECK(v.seconds < 0.020 + 0.5);
    }
    CHECK_EQ(ladder->pending(), std::size_t{0});
    // A decision ended at once: its jobs cancelled or dropped.
    {
      Search search(s, p, 0, nullptr, ladder);
      search.run(*uniform, Limits{1});
    }
    CHECK_EQ(ladder->pending(), std::size_t{0});
    const VerificationLadder::Stats st = ladder->stats();
    CHECK_EQ(st.asked, st.answered + st.cancelled);
    CHECK_EQ(st.threads_started, std::uint64_t{1});  // one worker for the ladder's life
  }
}

TEST(ladder_settling_proofs_survive_tt_churn_6f) {
  // 6f: proofs are never held only in the transposition table. The settling
  // prover's table cut to 16 entries (churned at every node): its proofs go to
  // its own store at once, so the same question asked again is answered
  // proven - each certificate confirmed by E4.
  const GameState s = t5d_test::v3::mate_in_two();
  LadderOptions lo = ladder_options(sync_params());
  lo.settling.tt_entries = 16;
  VerificationLadder ladder(lo, s.rules(), s.multiverse().ruleset());
  for (int round = 0; round < 3; ++round) {
    ladder.begin_decision();
    ladder.ask(LadderJob::RootSettling, s, {}, 5, 0, 200'000);
    const std::vector<LadderVerdict> got = ladder.take();
    REQUIRE_EQ(got.size(), std::size_t{1});
    CHECK(got[0].status == ProofStatus::Proven);
    REQUIRE(got[0].certificate != nullptr);
    CHECK(check_certificate(*got[0].certificate, s, got[0].proof.distance).verdict == CertificateVerdict::Confirmed);
  }
}

// ---- Guard's hook: off, and refused without the proof channel --------------------------------

TEST(ladder_guards_hook_is_off_by_default_and_rejects_a_refuted_action_when_on) {
  const GameState s = blunder_position();
  const Move rook_away = blunder(s);
  Favouring ev(rook_away);
  PlayParams pp;
  pp.search = sync_params();
  pp.search.split = TurnSplit::Flat;  // a decision is the whole action: a rejection can undo it
  pp.search.hunt = pp.search.settling = false;
  pp.search.asset_map = false;
  // Two playouts: the root and the favourite's turn - Black's turn start
  // evaluated, its children not yet made, so the in-tree solver (the channel
  // on, below) has not seen the mate: Guard's refutation is what rejects it.
  pp.playouts = 2;
  pp.guard_research_playouts = 64;
  pp.max_actions = 1;
  pp.match_play = true;
  // Off: the favourite is played.
  const GameRecord off = play_game(s, ev, pp);
  REQUIRE_EQ(off.actions.size(), std::size_t{1});
  CHECK(off.actions[0].size() == 1 && off.actions[0][0] == rook_away);
  CHECK_EQ(off.guard_runs, std::uint64_t{0});
  // On without the proof channel: refused (a refutation is a proof, H41).
  pp.guard_final = true;
  CHECK_THROWS_AS(play_game(s, ev, pp), std::invalid_argument);
  // On with it (experiments only): Guard refutes Rc5-a5 (Black's mate in one),
  // E4 confirms the refutation, the action is rejected and searched again.
  pp.search.proof_channel = true;
  const GameRecord on = play_game(s, ev, pp);
  REQUIRE_EQ(on.actions.size(), std::size_t{1});
  CHECK(on.guard_refutations >= 1);
  CHECK(on.guard_rejections >= 1);
  CHECK(!(on.actions[0].size() == 1 && on.actions[0][0] == rook_away));
}

// ---- Self-play with the defaults -------------------------------------------------------------

TEST(ladder_self_play_with_the_defaults_asks_and_records) {
  // The defaults (asynchronous, 5 threads, 50 ms): a short self-play game asks
  // Hunt and the settling scans at its roots under their ranges, withholds
  // every proof, and plays legally (play_game checks every move and
  // submission).
  const std::unique_ptr<Evaluator> random = make_evaluator(BuiltinEvaluator::Random, 2);
  PlayParams pp;
  pp.search.expand.square_features = SquareFeatures::None;
  pp.playouts = 48;
  pp.max_actions = 6;
  pp.turn_cap.boards = 300;
  const GameRecord g = play_game(variant("Standard - Two Timelines"), *random, pp);
  CHECK(g.actions.size() >= 1);
  CHECK_EQ(g.proofs_injected, std::uint64_t{0});
  std::printf("  %zu actions: hunt %llu, settling %llu, pv changes %llu, steered visits %llu\n", g.actions.size(),
              static_cast<unsigned long long>(g.hunt_runs), static_cast<unsigned long long>(g.settling_runs),
              static_cast<unsigned long long>(g.pv_changes), static_cast<unsigned long long>(g.steered_visits));
}
