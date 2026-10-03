// Checking a proof's certificate against E4 alone (see t5d/certificate.hpp).

#include "t5d/certificate.hpp"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "t5d/check.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/movegen.hpp"
#include "t5d/search.hpp"

namespace t5d {

const char* certificate_verdict_name(CertificateVerdict v) noexcept {
  switch (v) {
    case CertificateVerdict::Confirmed: return "confirmed";
    case CertificateVerdict::Refuted: return "refuted";
    case CertificateVerdict::Incomplete: return "incomplete";
    case CertificateVerdict::Undecided: return "undecided";
  }
  return "?";
}

namespace {

struct Checker {
  const ProofCertificate& cert;
  const CertificateCheckOptions& o;
  CertificateCheck& r;
  // Defender positions confirmed, with the half-turns they were confirmed
  // within (confirmed within d: within any d' >= d too).
  std::unordered_map<std::uint64_t, std::vector<std::pair<GameState, int>>> done;

  SearchBudget budget(std::uint64_t nodes) const {
    SearchBudget b = SearchBudget::nodes(nodes);
    b.max_bytes = o.max_bytes;
    return b;
  }

  bool seen(const GameState& t, int d) const {
    const auto it = done.find(t.zobrist());
    if (it == done.end()) return false;
    for (const auto& [s, within] : it->second) {
      if (within <= d && same_position(s, t)) return true;
    }
    return false;
  }
  void mark(const GameState& t, int d) { done[t.zobrist()].emplace_back(t, d); }

  CertificateVerdict fail(CertificateVerdict v, std::string why) {
    if (r.why.empty()) r.why = std::move(why);
    return v;
  }

  // `s`: the attacker to move, `d` half-turns left.
  CertificateVerdict attacker(const GameState& s, int d) {
    ++r.attacker_positions;
    if (d < 1) return fail(CertificateVerdict::Refuted, "no half-turn left at an attacker's position");
    const CertificateStep* step = cert.find(s);
    if (step == nullptr) {
      // Named, so that a proof that cannot be certified is known: a proof the
      // prover only read from a store another prover filled is not in its
      // certificate (t5d/prover.hpp).
      return fail(CertificateVerdict::Incomplete,
                  "an attacker's position the certificate does not hold (H30 fingerprint " +
                      fingerprint(s.multiverse()).fingerprint + ", " + std::to_string(d) +
                      " half-turns left) - a proof read from a store another prover filled cannot be certified");
    }
    GameState t = s;
    MoveList scratch;
    for (const Move& m : step->action) {
      if (!t.may_move() || !is_pseudo_legal(t, m, scratch)) {
        return fail(CertificateVerdict::Refuted, "a certificate move is not pseudo-legal where it is made");
      }
      t.apply(m);
    }
    if (!t.obligations_met()) return fail(CertificateVerdict::Refuted, "a certificate action leaves an obligation (T5)");
    if (!royals_safe(t)) return fail(CertificateVerdict::Refuted, "a certificate action leaves a royal capturable (C4)");
    t.submit(CheckRuling::none());
    return defender(t, d - 1);
  }

  // `t`: the defender to move, `d` half-turns left.
  CertificateVerdict defender(const GameState& t, int d) {
    if (seen(t, d)) return CertificateVerdict::Confirmed;
    if (++r.defender_positions > o.max_positions) {
      return fail(CertificateVerdict::Undecided, "more defender positions than the check's cap");
    }
    const MateResult m = detect_mate(t, Detector::Hypercuboid, budget(o.detector_nodes));
    switch (m.verdict) {
      case Outcome::Unknown:
        return fail(CertificateVerdict::Undecided, "E4's hypercuboid detector out of its budget");
      case Outcome::Stalemate:
        return fail(CertificateVerdict::Refuted, "a stalemate in the proof");
      case Outcome::Checkmate: {
        ++r.mates;
        if (o.naive) {
          const MateResult n = detect_mate(t, Detector::Naive, budget(o.naive_nodes));
          if (n.verdict == Outcome::Checkmate) {
            ++r.mates_naive;
          } else if (n.verdict == Outcome::Unknown) {
            ++r.naive_undecided;
          } else {
            return fail(CertificateVerdict::Refuted, "E4's two detectors disagree on a mate");
          }
        }
        mark(t, d);
        return CertificateVerdict::Confirmed;
      }
      case Outcome::Ongoing:
        break;
    }
    if (d < 1) return fail(CertificateVerdict::Refuted, "the defender is not mated at the horizon");
    const ActionList l = list_legal_actions(t, o.listing_cap, Detector::Hypercuboid, budget(o.listing_nodes));
    if (!l.stats.exhaustive || l.reached_cap) {
      return fail(CertificateVerdict::Undecided, "E4's listing of a defender's actions is incomplete");
    }
    if (l.actions.empty()) {
      // E4's detector said a legal action exists: its listing must hold one.
      return fail(CertificateVerdict::Refuted, "E4's listing and its detector disagree");
    }
    for (const LegalAction& a : l.actions) {
      ++r.defender_actions;
      const CertificateVerdict v = attacker(a.result, d - 1);
      if (v != CertificateVerdict::Confirmed) return v;
    }
    mark(t, d);
    return CertificateVerdict::Confirmed;
  }
};

}  // namespace

CertificateCheck check_certificate(const ProofCertificate& certificate, const GameState& root, int depth,
                                   const CertificateCheckOptions& options) {
  CertificateCheck r;
  Checker c{certificate, options, r, {}};
  r.verdict = c.attacker(root, depth);
  if (r.verdict == CertificateVerdict::Confirmed) r.why.clear();
  return r;
}

}  // namespace t5d
