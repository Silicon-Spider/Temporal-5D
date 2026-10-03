// Experiment 6b''s counting and verdicts (see sixbpp.hpp).

#include "sixbpp.hpp"

namespace t5d::tools::sixbpp {

namespace {

std::string yes_no(bool b) { return b ? "true" : "false"; }

const char* branch_threat_name(BranchThreat b) noexcept {
  switch (b) {
    case BranchThreat::None: return "none";
    case BranchThreat::ArrivalCapture: return "arrival-capture";
    case BranchThreat::AnyCapture: return "any-capture";
  }
  return "?";
}

}  // namespace

const char* range_name(int range) {
  static const char* const names[kRanges] = {"< 50 boards", "50-99", "100-199", "200+"};
  return range >= 0 && range < kRanges ? names[range] : "pooled";
}

int range_of(int boards) noexcept { return boards < 50 ? 0 : boards < 100 ? 1 : boards < 200 ? 2 : 3; }

bool guard_passes(std::size_t done, std::size_t roots) noexcept { return roots > 0 && 100 * done >= 95 * roots; }

bool hunt_passes(std::size_t found, std::size_t truth_mates) noexcept {
  return truth_mates > 0 && 2 * found >= truth_mates;
}

Check check_of(const CertificateCheck& c) noexcept {
  switch (c.verdict) {
    case CertificateVerdict::Confirmed: return Check::Confirmed;
    case CertificateVerdict::Refuted: return Check::Refuted;
    case CertificateVerdict::Incomplete:
    case CertificateVerdict::Undecided: return Check::Unconfirmed;
  }
  return Check::Unconfirmed;
}

const char* check_name(Check c) noexcept {
  switch (c) {
    case Check::None: return "none";
    case Check::Confirmed: return "confirmed";
    case Check::Unconfirmed: return "unconfirmed";
    case Check::Refuted: return "REFUTED";
  }
  return "?";
}

bool depth_one_done(const RootOutcome& r) noexcept {
  if (!r.guard) return false;
  if (r.completed_depth >= 1) return true;
  return r.refuted && r.refutation_depth == 1 && r.guard_check == Check::Confirmed;
}

bool depth_one_unconfirmed(const RootOutcome& r) noexcept {
  return r.guard && r.completed_depth < 1 && r.refuted && r.refutation_depth == 1 &&
         r.guard_check != Check::Confirmed;
}

Tallies tally(const std::vector<RootOutcome>& roots) {
  Tallies t;
  for (const RootOutcome& r : roots) {
    if (r.guard && r.refuted && r.guard_check == Check::Refuted) ++t.refuted_proofs;
    if (r.hunt && r.hunt_mate && r.hunt_check == Check::Refuted) ++t.refuted_proofs;
    for (const int k : {range_of(r.boards), kRanges}) {
      if (r.guard) {
        GuardTally& g = t.guard[k];
        ++g.roots;
        if (depth_one_done(r)) ++g.depth_one;
        if (depth_one_unconfirmed(r)) ++g.unconfirmed;
        if (r.refuted) ++g.refuted;
        if (r.refuted && r.refutation_depth == 1) ++g.refuted_one;
        g.completed[r.completed_depth]++;
      }
      if (r.hunt) {
        HuntTally& h = t.hunt[k];
        ++h.roots;
        const bool confirmed = r.hunt_mate && r.hunt_check == Check::Confirmed;
        if (r.hunt_mate) ++h.mates;
        if (r.hunt_mate && !confirmed) ++h.unconfirmed;
        if (r.truth_mate) ++h.truth;
        if (r.truth_mate && confirmed) ++h.found;
        if (r.truth_mate && r.hunt_mate && !confirmed) ++h.found_unconfirmed;
        if (confirmed && !r.truth_mate) ++h.extra;
      }
    }
  }
  return t;
}

const char* verdict_name(Verdict v) noexcept {
  switch (v) {
    case Verdict::Adopt: return "adopt";
    case Verdict::NotAdopted: return "not adopted";
    case Verdict::NoRoots: return "no roots";
    case Verdict::NoTruthMates: return "no verdict (no mate in the truth to find)";
    case Verdict::NoTruth: return "no verdict (no ground truth)";
    case Verdict::Withheld: return "WITHHELD";
  }
  return "?";
}

Verdict guard_verdict(const Tallies& t, int range, bool crosschecked) {
  if (!crosschecked || t.refuted_proofs > 0) return Verdict::Withheld;
  const GuardTally& g = t.guard[range];
  if (g.roots == 0) return Verdict::NoRoots;
  return guard_passes(g.depth_one, g.roots) ? Verdict::Adopt : Verdict::NotAdopted;
}

Verdict hunt_verdict(const Tallies& t, int range, bool have_truth, bool crosschecked) {
  if (!have_truth) return Verdict::NoTruth;
  if (!crosschecked || t.refuted_proofs > 0) return Verdict::Withheld;
  const HuntTally& h = t.hunt[range];
  if (h.roots == 0) return Verdict::NoRoots;
  if (h.truth == 0) return Verdict::NoTruthMates;
  return hunt_passes(h.found, h.truth) ? Verdict::Adopt : Verdict::NotAdopted;
}

bool too_few_to_read(const Tallies& t, int range) {
  const std::size_t n = t.hunt[range].truth;
  return n > 0 && n < kFewTruthMates;
}

std::vector<std::pair<std::string, std::string>> prover_options(const ProverOptions& o) {
  std::string values;
  for (const int v : o.values.value) values += (values.empty() ? "" : "/") + std::to_string(v);
  return {
      {"depth", std::to_string(o.depth)},
      {"seeds", seed_class_name(o.seeds)},
      {"completion", completion_class_name(o.completion)},
      {"values", values},
      {"branch_threat", branch_threat_name(o.branch_threat)},
      {"branch_threat_value", std::to_string(o.branch_threat_value)},
      {"promotions_forcing", yes_no(o.promotions_forcing)},
      {"completion_max_nodes", std::to_string(o.completion_max_nodes)},
      {"escape_probe_nodes", std::to_string(o.escape_probe_nodes)},
      {"legality_tiers", yes_no(o.legality_tiers)},
      {"listing_cap", std::to_string(o.listing_cap)},
      {"tt_entries", std::to_string(o.tt_entries)},
      {"detector_max_bytes", std::to_string(o.detector_max_bytes)},
      {"max_tree_nodes", std::to_string(o.max_tree_nodes)},
      {"threads", std::to_string(o.threads)},
      {"record_all_disproofs", yes_no(o.record_all_disproofs)},
      {"prove_only", yes_no(o.prove_only)},
      {"prove_only_pn_thresholds", yes_no(o.prove_only_pn_thresholds)},
      {"check_filter", yes_no(o.check_filter)},
      {"record_certificate", yes_no(o.record_certificate)},
      {"attacker_probe", yes_no(o.attacker_probe)},
  };
}

std::vector<std::pair<std::string, std::string>> guard_options(const GuardOptions& o) {
  std::vector<std::pair<std::string, std::string>> out{
      {"max_depth", std::to_string(o.max_depth)},
      {"threads", std::to_string(o.threads)},
  };
  for (auto& [k, v] : prover_options(guard_prover_options(o))) out.emplace_back("prover." + k, v);
  return out;
}

std::vector<std::pair<std::string, std::string>> hunt_options(const HuntOptions& o) {
  std::vector<std::pair<std::string, std::string>> out{
      {"threads", std::to_string(o.threads)},
      {"export_proofs", yes_no(o.export_proofs)},
      {"suggest", yes_no(o.suggest)},
      {"max_suggestions", std::to_string(o.max_suggestions)},
  };
  for (auto& [k, v] : prover_options(hunt_prover_options(o))) out.emplace_back("prover." + k, v);
  return out;
}

}  // namespace t5d::tools::sixbpp
