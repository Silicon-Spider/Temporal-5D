// The named variants (see t5d/variants.hpp). Every set-up is its row of the
// rules reference's V3 table (docs/rules/README.md, section 8), with every
// pawn, rook, king and brawn marked unmoved.

#include "t5d/variants.hpp"

#include <array>
#include <cstddef>
#include <string_view>

namespace t5d {

namespace {

using namespace std::string_view_literals;

// The Standard placement: the notation specification's own example ("Hashing",
// "5DFEN and custom variants"), V3's Standard row.
#define T5D_STANDARD_PIECES "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*"

constexpr std::array kStandard = {"[" T5D_STANDARD_PIECES ":0:1:w]"sv};
constexpr std::array kTurnZero = {"[" T5D_STANDARD_PIECES ":0:0:b]"sv,
                                  "[" T5D_STANDARD_PIECES ":0:1:w]"sv};
constexpr std::array kTwoTimelines = {"[" T5D_STANDARD_PIECES ":-0:1:w]"sv,
                                      "[" T5D_STANDARD_PIECES ":+0:1:w]"sv};
#undef T5D_STANDARD_PIECES

constexpr std::array kHalfReflected = {
    "[r*nbk*qbnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*:0:1:w]"sv};
constexpr std::array kDefendedPawn = {
    "[r*qbnk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*QBNK*BNR*:0:1:w]"sv};
constexpr std::array kReversedRoyalty = {
    "[r*nbycbnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBYCBNR*:0:1:w]"sv};
constexpr std::array kVerySmallOpen = {"[nbr*k*/3p*/P*3/K*R*BN:0:1:w]"sv};
constexpr std::array kVerySmall = {"[nbr*k*/p*p*p*p*/P*P*P*P*/K*R*BN:0:1:w]"sv};
constexpr std::array kSmall = {"[k*qbnr*/p*p*p*p*p*/5/P*P*P*P*P*/K*QBNR*:0:1:w]"sv};
constexpr std::array kJustKings = {"[2k*/3/K*2:0:1:w]"sv};
constexpr std::array kJustPawns = {"[p*p*p*p*k*/5/5/5/K*P*P*P*P*:0:1:w]"sv};
constexpr std::array kJustBrawns = {"[w*w*w*w*k*/5/5/5/K*W*W*W*W*:0:1:w]"sv};
constexpr std::array kBattleground = {
    "[r*r*k*r*r*/bbqbb/p*p*p*p*p*/5/P*P*P*P*P*:-1:1:w]"sv,
    "[nnnnn/p*p*p*p*p*/5/P*P*P*P*P*/NNNNN:0:1:w]"sv,
    "[p*p*p*p*p*/5/P*P*P*P*P*/BBQBB/R*R*K*R*R*:+1:1:w]"sv};
constexpr std::array kTactician = {"[k*bnr*/p*p*p*p*/4/4:-0:1:w]"sv,
                                   "[4/4/P*P*P*P*/K*BNR*:+0:1:w]"sv};
constexpr std::array kStrategos = {"[nbk*ur*/p*p*p*p*p*/5/5/5:-0:1:w]"sv,
                                   "[5/5/5/P*P*P*P*P*/R*UK*BN:+0:1:w]"sv};
constexpr std::array kFormations = {"[p*p*p*p*p*/5/5/5/2K*2:-0:1:w]"sv,
                                    "[2k*2/5/5/5/P*P*P*P*P*:+0:1:w]"sv};

constexpr BoardSize k8x8{8, 8};
constexpr BoardSize k5x5{5, 5};
constexpr BoardSize k4x4{4, 4};
constexpr BoardSize k3x3{3, 3};

constexpr std::array kVariants = {
    Variant{"Standard", {}, k8x8, Parity::Odd, kStandard, RuleFamily::Standard,
            "rules reference V3 \"Standard\" (high; the notation specification's own example)",
            Confidence::High},
    Variant{"Standard - Turn Zero", {}, k8x8, Parity::Odd, kTurnZero, RuleFamily::Standard,
            "rules reference V3 \"Standard - Turn Zero\" (high): the Standard placement on (0T0) "
            "with Black to move, a history board from the start, and on (0T1) with White to move "
            "(L5, Q8)",
            Confidence::High},
    Variant{"Standard - Two Timelines", {}, k8x8, Parity::Even, kTwoTimelines,
            RuleFamily::Standard,
            "rules reference V3 \"Standard - Two Timelines\" (high): the Standard placement on -0 "
            "and +0 (L8)",
            Confidence::High},
    Variant{"Standard - Half Reflected", {}, k8x8, Parity::Odd, kHalfReflected,
            RuleFamily::Standard,
            "rules reference V3 \"Standard - Half Reflected\" (medium): Black's king d8, queen e8",
            Confidence::Medium},
    Variant{"Standard - Defended Pawn", {}, k8x8, Parity::Odd, kDefendedPawn,
            RuleFamily::Standard, "rules reference V3 \"Standard - Defended Pawn\" (medium)",
            Confidence::Medium},
    Variant{"Standard - Reversed Royalty", {}, k8x8, Parity::Odd, kReversedRoyalty,
            RuleFamily::ReversedRoyalty,
            "rules reference V3 \"Standard - Reversed Royalty\" (high): royal queen (Y) and "
            "common king (C), 4.11, 4.12; no castling, S4; a rule variant, design spec §12.6",
            Confidence::High},
    Variant{"Very Small - Open", {}, k4x4, Parity::Odd, kVerySmallOpen, RuleFamily::Standard,
            "rules reference V3 \"Very Small - Open\" (medium)", Confidence::Medium},
    Variant{"Very Small", {}, k4x4, Parity::Odd, kVerySmall, RuleFamily::Standard,
            "rules reference V3 \"Very Small\" (medium; Black's set-up since 2020-10-07 is "
            "derived, Q22)",
            Confidence::Medium},
    Variant{"Small", "Misc - Small", k5x5, Parity::Odd, kSmall, RuleFamily::Standard,
            "rules reference V3 \"Small\" (2020 \"Misc - Small\"; medium, the 2020 placement)",
            Confidence::Medium},
    Variant{"Focused - Just Kings", {}, k3x3, Parity::Odd, kJustKings, RuleFamily::Standard,
            "rules reference V3 \"Focused - Just Kings\" (medium)", Confidence::Medium},
    Variant{"Focused - Just Pawns", {}, k5x5, Parity::Odd, kJustPawns, RuleFamily::Standard,
            "rules reference V3 \"Focused - Just Pawns\" (medium placement; the back-rank double "
            "step is open, Q15)",
            Confidence::Medium},
    Variant{"Focused - Just Brawns", {}, k5x5, Parity::Odd, kJustBrawns, RuleFamily::Standard,
            "rules reference V3 \"Focused - Just Brawns\" (low: one oracle-adjacent source only; "
            "the back-rank double step is unverified, Q15)",
            Confidence::Low},
    Variant{"Misc - Timeline Battleground", {}, k5x5, Parity::Odd, kBattleground,
            RuleFamily::Standard,
            "rules reference V3 \"Misc - Timeline Battleground\" (medium; the timeline mapping "
            "is inferred from L7)",
            Confidence::Medium},
    Variant{"Misc - Timeline Tactician", {}, k4x4, Parity::Even, kTactician,
            RuleFamily::Standard,
            "rules reference V3 \"Misc - Timeline Tactician\" (medium; T13: White's first action "
            "must jump onto -0)",
            Confidence::Medium},
    Variant{"Misc - Timeline Strategos", {}, k5x5, Parity::Even, kStrategos,
            RuleFamily::Standard, "rules reference V3 \"Misc - Timeline Strategos\" (medium)",
            Confidence::Medium},
    Variant{"Misc - Timeline Formations", {}, k5x5, Parity::Even, kFormations,
            RuleFamily::Standard,
            "rules reference V3 \"Misc - Timeline Formations\" (medium; the timeline mapping is "
            "inferred, and the guide's text says 4x4 where its image shows 5x5, Q22)",
            Confidence::Medium},
};

constexpr char lower(char c) noexcept {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}
constexpr bool is_space(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

// Reads a name as find_variant compares it: lower case, trimmed, every run of
// whitespace one space. next() gives the next character, or kEnd at the end.
class NameReader {
 public:
  static constexpr int kEnd = -1;

  explicit constexpr NameReader(std::string_view text) noexcept : text_(text) {
    while (at_ < text_.size() && is_space(text_[at_])) ++at_;
  }
  constexpr int next() noexcept {
    if (at_ >= text_.size()) return kEnd;
    if (is_space(text_[at_])) {
      while (at_ < text_.size() && is_space(text_[at_])) ++at_;
      return at_ < text_.size() ? ' ' : kEnd;  // trailing whitespace is dropped
    }
    return static_cast<unsigned char>(lower(text_[at_++]));
  }

 private:
  std::string_view text_;
  std::size_t at_ = 0;
};

constexpr bool same_name(std::string_view a, std::string_view b) noexcept {
  NameReader x(a);
  NameReader y(b);
  for (;;) {
    const int c = x.next();
    if (c != y.next()) return false;
    if (c == NameReader::kEnd) return true;
  }
}

static_assert(same_name("Standard - Turn Zero", "  standard  -\tTURN zero "));
static_assert(!same_name("Standard", "Standard - Turn Zero"));
static_assert(!same_name("Small", "Very Small"));
static_assert(!same_name("Standard", "Standard\0"sv));

}  // namespace

std::span<const Variant> variants() noexcept { return kVariants; }

const Variant* find_variant(std::string_view name) noexcept {
  for (const Variant& v : kVariants) {
    if (same_name(v.name, name)) return &v;
    if (!v.alias.empty() && same_name(v.alias, name)) return &v;
  }
  return nullptr;
}

GameState variant_start(const Variant& variant, const RuleOptions& rules) {
  RuleOptions r = rules;
  r.family = variant.family;
  return game_from_fen(variant.boards, variant.size, variant.parity, r, Colour::White);
}

}  // namespace t5d
