// The named variants (t5d/variants.hpp) against the rules reference's V3
// table (docs/rules/README.md, section 8). V3 writes placements without the
// unmoved marker; the engine marks every pawn, rook, king and brawn of a set-up
// (none of them has moved). So each expected placement below is V3's row as
// the reference prints it, marked here by that rule - an independent
// derivation of the table's hand-typed strings.

#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "t5d/fen.hpp"
#include "t5d/fingerprint.hpp"
#include "t5d/movegen.hpp"
#include "t5d/state.hpp"
#include "t5d/timeline.hpp"
#include "t5d/variants.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"

using namespace t5d;

namespace {

// V3's placement with '*' after every P, R, K and W (either case).
std::string marked(std::string_view v3) {
  std::string out;
  for (const char c : v3) {
    out += c;
    switch (c) {
      case 'P': case 'p': case 'R': case 'r': case 'K': case 'k': case 'W': case 'w':
        out += '*';
        break;
      default:
        break;
    }
  }
  return out;
}

// Every board of a state as "<label>:<turn>:<colour>:<placement>", canonical order.
std::vector<std::string> boards_of(const GameState& s) {
  std::vector<std::string> out;
  const Multiverse& mv = s.multiverse();
  mv.for_each_board([&](int p, int h, const Board& b) {
    out.push_back(timeline_label_string(p, mv.even()) + ":" + std::to_string(turn_of(h)) + ":" +
                  colour_letter(to_move_at(h)) + ":" + fen_pieces(b));
  });
  return out;
}

struct Expected {
  std::string_view name;
  BoardSize size;
  std::vector<std::string> boards;  // label:turn:colour:placement
};

}  // namespace

TEST(variants_set_ups_follow_the_rules_reference_v3) {
  const std::string standard = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
  const std::vector<Expected> expected = {
      {"Standard", {8, 8}, {"0:1:w:" + marked(standard)}},
      // "the Standard placement on (0T0) with Black to move ... and on (0T1)
      // with White to move"
      {"Standard - Turn Zero", {8, 8}, {"0:0:b:" + marked(standard), "0:1:w:" + marked(standard)}},
      {"Standard - Two Timelines", {8, 8},
       {"-0:1:w:" + marked(standard), "+0:1:w:" + marked(standard)}},
      {"Standard - Half Reflected", {8, 8},
       {"0:1:w:" + marked("rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")}},
      {"Standard - Defended Pawn", {8, 8},
       {"0:1:w:" + marked("rqbnkbnr/pppppppp/8/8/8/8/PPPPPPPP/RQBNKBNR")}},
      {"Standard - Reversed Royalty", {8, 8},
       {"0:1:w:" + marked("rnbycbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBYCBNR")}},
      {"Very Small - Open", {4, 4}, {"0:1:w:" + marked("nbrk/3p/P3/KRBN")}},
      {"Very Small", {4, 4}, {"0:1:w:" + marked("nbrk/pppp/PPPP/KRBN")}},
      {"Small", {5, 5}, {"0:1:w:" + marked("kqbnr/ppppp/5/PPPPP/KQBNR")}},
      {"Focused - Just Kings", {3, 3}, {"0:1:w:" + marked("2k/3/K2")}},
      {"Focused - Just Pawns", {5, 5}, {"0:1:w:" + marked("ppppk/5/5/5/KPPPP")}},
      {"Focused - Just Brawns", {5, 5}, {"0:1:w:" + marked("wwwwk/5/5/5/KWWWW")}},
      {"Misc - Timeline Battleground", {5, 5},
       {"-1:1:w:" + marked("rrkrr/bbqbb/ppppp/5/PPPPP"),
        "0:1:w:" + marked("nnnnn/ppppp/5/PPPPP/NNNNN"),
        "+1:1:w:" + marked("ppppp/5/PPPPP/BBQBB/RRKRR")}},
      {"Misc - Timeline Tactician", {4, 4},
       {"-0:1:w:" + marked("kbnr/pppp/4/4"), "+0:1:w:" + marked("4/4/PPPP/KBNR")}},
      {"Misc - Timeline Strategos", {5, 5},
       {"-0:1:w:" + marked("nbkur/ppppp/5/5/5"), "+0:1:w:" + marked("5/5/5/PPPPP/RUKBN")}},
      {"Misc - Timeline Formations", {5, 5},
       {"-0:1:w:" + marked("ppppp/5/5/5/2K2"), "+0:1:w:" + marked("2k2/5/5/5/PPPPP")}},
  };
  CHECK_EQ(variants().size(), expected.size());
  for (const Expected& e : expected) {
    const Variant* v = find_variant(e.name);
    REQUIRE(v != nullptr);
    CHECK_EQ(v->name, e.name);
    CHECK(v->size == e.size);
    const GameState s = variant_start(*v);
    CHECK(s.multiverse().board_size() == e.size);
    CHECK_EQ(boards_of(s), e.boards);
    // V2: every variant starts with White to move.
    CHECK(s.to_move() == Colour::White);
    CHECK(!s.obligations_met());
    // The canonical state string reads back to the same multiverse.
    std::vector<std::string> strings;
    s.multiverse().for_each_board([&](int p, int h, const Board& b) {
      strings.push_back(fen_board_string(b, p, h, s.multiverse().even()));
    });
    const GameState again = game_from_fen(strings, e.size, Parity::Infer, s.rules());
    CHECK(again == s);
  }
}

TEST(variants_names_compare_without_case_or_spacing) {
  CHECK(find_variant("standard") == find_variant("Standard"));
  CHECK(find_variant("  Standard   -  Turn Zero ") == find_variant("Standard - Turn Zero"));
  CHECK(find_variant("STANDARD - TWO TIMELINES") != nullptr);
  // The 2020 name of Small (V3).
  REQUIRE(find_variant("Misc - Small") != nullptr);
  CHECK_EQ(find_variant("Misc - Small")->name, std::string_view("Small"));
  // Nothing else is forgiven.
  CHECK(find_variant("Standard-Turn Zero") == nullptr);
  CHECK(find_variant("Standard - Two Timeline") == nullptr);
  CHECK(find_variant("custom") == nullptr);
  CHECK(find_variant("") == nullptr);
  CHECK(find_variant("Very") == nullptr);
  // V3 rows the engine does not set up (not requested in this stage).
  CHECK(find_variant("Simple - No Queens") == nullptr);
  CHECK(find_variant("Standard - Princess") == nullptr);
  CHECK(find_variant("Misc - Excessive") == nullptr);
  CHECK(find_variant(std::string_view("Standard\0", 9)) == nullptr);
}

TEST(variants_every_set_up_is_a_valid_state_with_its_present_at_t1) {
  for (const Variant& v : variants()) {
    const GameState s = variant_start(v);
    // T4: the present is the earliest playable column of the active
    // timelines - (T1, white) everywhere (Turn Zero's T0 board is history).
    CHECK_EQ(s.present(), half_turn(1, Colour::White));
    CHECK(s.inactive_timelines().empty());  // T10, T13: initial timelines are active
    CHECK_EQ(s.multiverse().ruleset(), s.rules().ruleset_id());
    CHECK(!v.source.empty());
  }
}

TEST(variants_reversed_royalty_has_its_own_ruleset_id) {
  // Design spec §12.6 (a rule variant), §11.3 (I9: exact stores per ruleset).
  const Variant* rr = find_variant("Standard - Reversed Royalty");
  REQUIRE(rr != nullptr);
  CHECK(rr->family == RuleFamily::ReversedRoyalty);
  const GameState s = variant_start(*rr);
  CHECK(s.rules().family == RuleFamily::ReversedRoyalty);
  CHECK_NE(s.multiverse().ruleset(), 0u);
  // The same boards read as a custom position (the standard family) carry the
  // standard ID, and so another key.
  const GameState custom = t5d_test::game(
      {"[r*nbycbnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBYCBNR*:0:1:w]"}, {8, 8});
  CHECK_EQ(custom.multiverse().ruleset(), 0u);
  CHECK_NE(s.zobrist(), custom.zobrist());
  CHECK(fingerprint(s.multiverse()) == fingerprint(custom.multiverse()));
  // Every other variant is in the standard family (§12.6: position and
  // structure variants share the rules).
  for (const Variant& v : variants()) {
    if (&v == rr) continue;
    CHECK(v.family == RuleFamily::Standard);
    CHECK_EQ(variant_start(v).multiverse().ruleset(), 0u);
  }
  // S4: neither the royal queen nor the common king castles - no king of
  // White's has a castling move (there is no royal king at all).
  const Board* start = s.multiverse().board_at(0, half_turn(1, Colour::White));
  CHECK_EQ(start->pieces(Colour::White, PieceKind::King), std::uint64_t{0});
  CHECK(start->at(3, 0) == Piece(PieceKind::RoyalQueen, Colour::White));
  CHECK(start->at(4, 0) == Piece(PieceKind::CommonKing, Colour::White));
  CHECK(start->at(3, 7) == Piece(PieceKind::RoyalQueen, Colour::Black));
  CHECK(start->at(4, 7) == Piece(PieceKind::CommonKing, Colour::Black));
}

TEST(variants_reversed_royalty_promotion_is_the_queen_by_default_q17) {
  // S8 (high): promotion to a queen; Q17's alternative (a royal queen) is the
  // promotion set {Y}, a ruleset option.
  const Variant* rr = find_variant("Standard - Reversed Royalty");
  REQUIRE(rr != nullptr);
  CHECK(variant_start(*rr).rules().promotions == PromotionSet{});
  RuleOptions royal;
  royal.promotions = PromotionSet::of({PieceKind::RoyalQueen});
  const GameState alt = variant_start(*rr, royal);
  CHECK(alt.rules().promotions.contains(PieceKind::RoyalQueen));
  CHECK(alt.rules().family == RuleFamily::ReversedRoyalty);
  CHECK_NE(alt.multiverse().ruleset(), variant_start(*rr).multiverse().ruleset());
}

TEST(variants_turn_zero_has_a_history_board_from_the_start) {
  const GameState s = variant_start(*find_variant("Standard - Turn Zero"));
  const Timeline* t = s.multiverse().timeline(0);
  REQUIRE(t != nullptr);
  CHECK_EQ(t->length(), 2);
  CHECK_EQ(t->first_half_turn(), half_turn(0, Colour::Black));  // (0T0), Black to move
  CHECK(s.playable_by_mover(0));
  CHECK_EQ(s.mandatory_count(), 1);
}

TEST(variants_tactician_white_must_jump_onto_the_board_without_white_pieces_t13) {
  // T13: all initial timelines are active from the first move, so White must
  // move on -0 too - which holds no White piece: only a jump from +0 can.
  const GameState s = variant_start(*find_variant("Misc - Timeline Tactician"));
  CHECK_EQ(s.mandatory_count(), 2);
  const Board* minus0 = s.multiverse().board_at(-1, half_turn(1, Colour::White));
  REQUIRE(minus0 != nullptr);
  CHECK_EQ(minus0->pieces(Colour::White), std::uint64_t{0});
  MoveList moves;
  generate_moves(s, moves);
  int onto_minus0 = 0;
  for (const Move& m : moves) {
    CHECK_EQ(m.from_timeline, 0);  // every move starts on +0
    if (m.to_timeline == -1) {
      ++onto_minus0;
      CHECK(!m.branching());  // -0's T1 board is playable (J4)
    }
  }
  CHECK(onto_minus0 > 0);
}

TEST(variants_battleground_reads_its_outer_timelines_as_5dfen_does) {
  // Q2: the non-central initial timelines count as one creation each side;
  // the count rule treats the sides alike, so both stay active and White's
  // next timeline is +2 (L7).
  const GameState s = variant_start(*find_variant("Misc - Timeline Battleground"));
  CHECK_EQ(s.multiverse().timeline_count(), 3);
  CHECK(!s.multiverse().even());
  CHECK(s.active(-1) && s.active(0) && s.active(1));
  CHECK_EQ(s.mandatory_count(), 3);
  CHECK_EQ(s.multiverse().created_by(Colour::White), 1);
  CHECK_EQ(s.multiverse().created_by(Colour::Black), 1);
}

TEST(variants_just_pawns_back_rank_double_step_follows_the_ruleset_q15) {
  // Q15: under the default (the unmoved flag) the back-rank pawns double-step;
  // under HomeRank (the second rank) they do not.
  const Variant* v = find_variant("Focused - Just Pawns");
  REQUIRE(v != nullptr);
  const GameState flag = variant_start(*v);
  CHECK(t5d_test::has_move(flag, "(0T1)Pb1(0T1)b3"));
  RuleOptions home;
  home.double_step_right = DoubleStepRight::HomeRank;
  const GameState rank = variant_start(*v, home);
  CHECK(!t5d_test::has_move(rank, "(0T1)Pb1(0T1)b3"));
  CHECK(t5d_test::has_move(rank, "(0T1)Pb1(0T1)b2"));
}
