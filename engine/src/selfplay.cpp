// Games of the engine against itself (t5d/selfplay.hpp).

#include "t5d/selfplay.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "t5d/certificate.hpp"
#include "t5d/ladder.hpp"
#include "t5d/legal.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"

namespace t5d::mcts {

namespace {

std::uint64_t mix(std::uint64_t z) noexcept {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Plays `m`, checking that it is a move of the player to move now.
void play_checked(GameState& s, const Move& m, MoveList& scratch) {
  if (!s.may_move() || !is_pseudo_legal(s, m, scratch)) {
    throw std::logic_error("play: the search chose " + to_string(m, s.multiverse().even()) +
                           ", which is not a pseudo-legal move now");
  }
  s.apply(m);
}

// Submits the action, checking that the submission is legal (T5, C4/R1).
void submit_checked(GameState& s) {
  if (s.submission_status() != SubmitStatus::Safe) {
    throw std::logic_error("play: the search chose an action that cannot be submitted legally");
  }
  s.submit();
}

}  // namespace

const char* game_end_name(GameEnd end) noexcept {
  switch (end) {
    case GameEnd::Checkmate: return "checkmate";
    case GameEnd::Stalemate: return "stalemate";
    case GameEnd::TurnCap: return "turn_cap";
    case GameEnd::Unknown: return "unknown";
    case GameEnd::ActionLimit: return "action_limit";
    case GameEnd::TimeLimit: return "time_limit";
  }
  return "?";
}

std::string GameRecord::result() const {
  if (end == GameEnd::Checkmate) return winner == 0 ? "1-0" : "0-1";
  if (end == GameEnd::Stalemate) return "1/2-1/2";
  return "*";
}

double GameRecord::white_score() const noexcept {
  switch (end) {
    case GameEnd::Checkmate: return winner == 0 ? 1.0 : 0.0;
    case GameEnd::Stalemate:
    case GameEnd::TurnCap: return 0.5;
    case GameEnd::Unknown:
    case GameEnd::ActionLimit:
    case GameEnd::TimeLimit: return -1.0;
  }
  return -1.0;
}

SearchParams self_play_search_defaults() {
  SearchParams p;
  p.root_noise = true;
  p.asset_map = false;  // until I2's steered visits are validated (issues 179, 189)
  return p;
}

GameRecord play_game(const GameState& start, Evaluator& evaluator, const PlayParams& params) {
  const auto t0 = std::chrono::steady_clock::now();
  GameRecord g(start);
  g.split = params.search.split;
  SearchParams sp = params.search;
  sp.turn_cap = params.turn_cap;
  if (params.match_play) sp.root_noise = false;  // §12.3: noise off at match play
  if (params.guard_final && !sp.proof_channel) {
    throw std::invalid_argument(
        "PlayParams::guard_final: Guard's refutation is a proof, and no proof may change play until the proof "
        "channel is signed off (H41) - Guard is adopted in no range (H42); experiments set search.proof_channel");
  }
  std::shared_ptr<EvalCache> cache;
  if (sp.eval_cache_entries > 0) cache = std::make_shared<EvalCache>(sp.eval_cache_entries, sp.eval_cache_bytes);
  // E10: the game's verification ladder (its asset map and settling prover live
  // as long as the game), shared by its searches.
  std::shared_ptr<VerificationLadder> ladder;
  if (ladder_wanted(sp)) {
    ladder = std::make_shared<VerificationLadder>(ladder_options(sp), start.rules(), start.multiverse().ruleset());
  }
  std::uint64_t rng = mix(params.seed ^ 0x73656C66706C6179ull);
  const auto uniform = [&rng] {
    rng += 0x9E3779B97F4A7C15ull;
    return static_cast<double>(mix(rng) >> 11) * 0x1.0p-53;
  };
  std::unique_ptr<Search> carried;
  GameState s = start;
  MoveList scratch;
  // The safety stop's deadline (max_seconds; THE DEADLINE in t5d/mcts.hpp:
  // none for 0 or from kNoDeadlineSeconds, NaN and negatives refused), handed
  // to every search as its Limits::seconds - the time left, recomputed before
  // each run - and to the rules engine's own calls, the turn-start verdicts
  // and the chosen moves' verifications, so that it reaches the detector
  // (build-log issue 406). A verdict or a verification it stops is Unknown,
  // never a verdict.
  const std::optional<std::chrono::steady_clock::time_point> deadline =
      deadline_after(t0, params.max_seconds, "PlayParams::max_seconds");
  const auto deadline_passed = [&] { return deadline && std::chrono::steady_clock::now() >= *deadline; };
  // A search's Limits::seconds: the time left before the game's deadline (at
  // least 1e-6, so that a run given it has one), or 0 - none.
  const auto seconds_left = [&]() -> double {
    if (!deadline) return 0;
    return std::max(std::chrono::duration<double>(time_left(*deadline)).count(), 1e-6);
  };
  SearchBudget verdict_budget;  // no node limit: a verdict is a proof (H37)
  verdict_budget.max_bytes = sp.mate_bytes;
  for (;;) {
    // A turn start: the rules first, then the cap's convention.
    const auto v0 = std::chrono::steady_clock::now();
    if (deadline) verdict_budget.max_time = time_left(*deadline);
    const MateResult r = detect_mate(s, sp.detector, verdict_budget);
    g.seconds_verdicts += std::chrono::duration<double>(std::chrono::steady_clock::now() - v0).count();
    if (r.verdict == Outcome::Unknown && !r.stats.stopped_by_memory && deadline_passed()) {
      g.end = GameEnd::TimeLimit;  // the safety stop, at this turn start
      break;
    }
    if (r.verdict == Outcome::Checkmate) {
      g.end = GameEnd::Checkmate;
      g.winner = index_of(opposite(s.to_move()));
      break;
    }
    if (r.verdict == Outcome::Stalemate) {
      g.end = GameEnd::Stalemate;
      break;
    }
    if (r.verdict == Outcome::Unknown) {
      g.end = GameEnd::Unknown;
      break;
    }
    if (params.turn_cap.reached(s)) {
      g.end = GameEnd::TurnCap;
      break;
    }
    if (g.actions.size() >= params.max_actions) {
      g.end = GameEnd::ActionLimit;
      break;
    }
    if (deadline_passed()) {
      g.end = GameEnd::TimeLimit;
      break;
    }
    std::vector<Move> action;
    // A6: the moves made in the action so far - the searches' roots carry the
    // count (the network sees it; equal states with other counts are other nodes).
    int moves_made = 0;
    const GameState action_start = s;
    bool timed_out = false;
    for (bool submitted = false; !submitted;) {
      // The safety stop before every decision, not only at turn starts; a
      // partial action is not recorded (the record ends at the turn start).
      if (moves_made > 0 && deadline_passed()) {
        timed_out = true;
        break;
      }
      std::unique_ptr<Search> fresh;
      Search* search = nullptr;
      bool built = false;  // the search was made for this decision
      if (params.match_play) {
        if (!carried) {
          carried = std::make_unique<Search>(s, sp, moves_made, cache, ladder);
          built = true;
        }
        search = carried.get();
        if (!(search->root_state() == s) || search->root_moves_made() != moves_made) {
          throw std::logic_error("play: the carried search's root is not the game's position (with its moves made, A6)");
        }
      } else {
        SearchParams dp = sp;
        dp.seed = mix(sp.seed + 0x9E3779B97F4A7C15ull * (g.decisions + 1));
        fresh = std::make_unique<Search>(s, dp, moves_made, cache, ladder);  // C14: fresh per move
        search = fresh.get();
        built = true;
      }
      // The counts before the search was built, for one built now: its
      // constructor's root check (the turn-start detector, under mate_nodes -
      // untimed; the review of 2026-11-09, L2) counts in the game's totals.
      const SearchStats before = built ? SearchStats{} : search->stats();
      g.reused_visits += search->node(search->root()).visits;
      // The decision's search work, added to the game's totals once - before
      // any break, a timed-out decision's included (L5).
      bool totalled = false;
      const auto add_totals = [&] {
        if (totalled) return;
        totalled = true;
        const SearchStats& after = search->stats();
        g.playouts += after.playouts - before.playouts;
        g.evaluations += after.evaluations - before.evaluations;
        g.nodes += after.nodes_created - before.nodes_created;
        g.search_turn_caps += after.turn_caps - before.turn_caps;
        g.seconds_search_mate += after.seconds_mate - before.seconds_mate;
        g.seconds_search_expand += after.seconds_expand - before.seconds_expand;
        g.seconds_search_generate += after.seconds_generate - before.seconds_generate;
        g.seconds_search_verify += after.seconds_verify - before.seconds_verify;
        g.hunt_runs += after.hunt_runs - before.hunt_runs;
        g.hunt_mates += after.hunt_mates - before.hunt_mates;
        g.settling_runs += after.settling_runs - before.settling_runs;
        g.settling_proven += after.settling_proven - before.settling_proven;
        g.settling_disproven += after.settling_disproven - before.settling_disproven;
        g.refutations += after.refutations - before.refutations;
        g.proofs_withheld += after.proofs_withheld - before.proofs_withheld;
        g.proofs_injected += after.proofs_injected - before.proofs_injected;
        g.pv_changes += after.pv_changes - before.pv_changes;
        g.asset_boosted_edges += after.asset_boosted_edges - before.asset_boosted_edges;
        g.steered_visits += after.steered_visits - before.steered_visits;
      };
      Limits limits{params.playouts};
      limits.seconds = seconds_left();
      search->run(evaluator, limits);
      const auto visited = [&] {
        for (const RootChild& c : search->root_children()) {
          if (c.visits > 0) return true;
        }
        return false;
      };
      // No child to choose and the deadline passed: TimeLimit (the run may
      // have stopped Deadline, or Playouts or Finished first - M1).
      if (!visited() && (search->stats().last_stop == StopReason::Deadline || deadline_passed())) {
        timed_out = true;
        add_totals();
        break;
      }
      // The choice, or none once the deadline has passed: never a run, and
      // never a choice, after it (M1). A run is made again first when no child
      // has a visit (children were removed), given the time left (L2).
      const auto choose_now = [&]() -> std::optional<std::size_t> {
        if (deadline_passed()) return std::nullopt;
        if (!visited()) {
          limits.seconds = seconds_left();
          search->run(evaluator, limits);
          if (!visited() && deadline_passed()) return std::nullopt;
        }
        if (params.match_play) return search->choose().index;
        const double t = g.decisions < params.temperature_decisions ? params.temperature : 0.0;
        return search->choose_by_visits(t, uniform());
      };
      const std::optional<std::size_t> chosen = choose_now();
      if (!chosen) {
        timed_out = true;
        add_totals();
        break;
      }
      std::size_t pick = *chosen;
      // A move is played only once known legal: one the expansion's budget left
      // unknown is classified exactly first - illegal, it is removed and the
      // choice made again; still unknown (the memory ceiling, or max_seconds'
      // deadline), the most visited child known legal is played instead - or,
      // with none and the deadline passed, the game stops there (TimeLimit).
      LegalityOptions exact = LegalityOptions::exact();
      exact.max_bytes = sp.mate_bytes;
      exact.deadline = deadline;
      std::vector<RootChild> children = search->root_children();
      // False: the deadline passed with no child known legal.
      const auto make_legal = [&]() -> bool {
        while (children.at(pick).legality != MoveLegality::Legal) {
          const MoveLegality v = search->verify(pick, exact);
          ++g.verified_choices;
          children = search->root_children();
          if (v == MoveLegality::Legal) break;
          if (v == MoveLegality::Illegal) {
            const std::optional<std::size_t> again = choose_now();
            if (!again) return false;  // the deadline passed: no choice after it
            pick = *again;
            children = search->root_children();
            continue;
          }
          std::size_t best = children.size();
          for (std::size_t i = 0; i < children.size(); ++i) {
            if (children[i].legality == MoveLegality::Legal &&
                (best == children.size() || children[i].visits > children[best].visits)) {
              best = i;
            }
          }
          if (best == children.size()) {
            if (deadline_passed()) return false;
            throw std::logic_error("play: no root child could be shown legal (the memory ceiling)");
          }
          pick = best;
        }
        return true;
      };
      if (!make_legal()) {
        timed_out = true;
        add_totals();
        break;
      }
      // E10: Guard at submission (H42 - adopted in no range: guard_final off).
      for (std::uint32_t tries = 0; params.guard_final && tries < params.guard_max_rejections; ++tries) {
        const RootChild& cc = children.at(pick);
        const bool flat = sp.split == TurnSplit::Flat;
        if (!flat && !cc.submit) break;  // the action is not complete yet
        std::vector<Move> full = action;
        if (flat) full.insert(full.end(), cc.moves.begin(), cc.moves.end());
        ProveBudget budget;
        GuardOptions go = params.guard;
        go.prover.record_certificate = true;
        if (sp.synchronous) {
          budget.max_nodes = params.guard_nodes > 0 ? params.guard_nodes : sp.sync_job_nodes;
          go.threads = 1;
        } else {
          if (params.guard_nodes > 0) budget.max_nodes = params.guard_nodes;
          budget.max_time = std::chrono::nanoseconds(static_cast<std::int64_t>(params.guard_ms * 1e6));
        }
        ++g.guard_runs;
        GuardResult gr = guard(action_start, full, budget, go);
        if (params.guard_certificate_hook_for_tests && gr.refuted) {
          params.guard_certificate_hook_for_tests(gr.certificate);
        }
        if (!gr.refuted) break;
        ++g.guard_refutations;
        // Only a refutation E4 confirms counts (6b'' rule).
        GameState refuted = action_start;
        for (const Move& m : full) refuted.apply(m);
        refuted.submit();
        const int distance = 2 * gr.refutation_depth - 1;
        if (!gr.certificate ||
            check_certificate(*gr.certificate, refuted, distance).verdict != CertificateVerdict::Confirmed) {
          break;
        }
        const PathStep step = flat ? PathStep{cc.moves, false} : PathStep{{}, true};
        if (search->inject_proof(std::span<const PathStep>(&step, 1), Proof::win(distance)) != ProofUse::Injected) {
          break;
        }
        ++g.guard_rejections;
        Limits again{params.guard_research_playouts > 0 ? params.guard_research_playouts : params.playouts};
        again.seconds = seconds_left();
        search->run(evaluator, again);
        const std::optional<std::size_t> rechosen = choose_now();
        if (!rechosen) {
          timed_out = true;
          break;
        }
        pick = *rechosen;
        children = search->root_children();
        if (!make_legal()) {
          timed_out = true;
          break;
        }
      }
      add_totals();
      if (timed_out) break;  // Guard's search again found no child known legal in time
      if (params.decision_log) {
        // H44: the decision as the audit lane reads it.
        DecisionLog d;
        d.action = static_cast<std::uint32_t>(g.actions.size());
        d.decision = static_cast<std::uint32_t>(g.decisions);
        d.prefix = action;
        d.boards = search->root_state().multiverse().board_count();
        const NodeView root = search->node(search->root());
        d.root_visits = root.visits;
        d.root_q = root.weight > 0 ? root.value / root.weight : 0.0;
        for (const RootChild& rc : children) {
          DecisionLog::Child c;
          c.step.moves = rc.moves;
          c.step.submit = rc.submit;
          c.visits = rc.visits;
          c.q = rc.q;
          c.was_pv = rc.was_pv;
          c.peak_share = rc.peak_share;
          d.children.push_back(std::move(c));
        }
        d.pv_changes = search->pv_changes();
        d.chosen = pick;
        g.decision_log.push_back(std::move(d));
      }
      ++g.decisions;
      const RootChild& c = children.at(pick);
      if (sp.split == TurnSplit::Flat) {
        for (const Move& m : c.moves) {
          play_checked(s, m, scratch);
          action.push_back(m);
        }
        submit_checked(s);
        submitted = true;
      } else if (c.submit) {
        submit_checked(s);
        submitted = true;
      } else {
        // A move; the action goes on until a submit is chosen (D8: submitting
        // is its own decision, never automatic).
        const Move& m = c.moves.at(0);
        play_checked(s, m, scratch);
        action.push_back(m);
      }
      if (params.match_play) search->advance(pick);
      if (!submitted) ++moves_made;
    }
    if (timed_out) {
      s = action_start;
      g.end = GameEnd::TimeLimit;
      break;
    }
    g.actions.push_back(std::move(action));
  }
  g.final_state = s;
  g.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return g;
}

}  // namespace t5d::mcts
