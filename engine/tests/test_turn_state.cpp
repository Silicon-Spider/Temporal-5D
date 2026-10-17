// Register A6 (spec §2.3; §15.2's row "Turn-state counts = a brute-force
// recount from the rules engine's obligations | The new inputs are exact"):
// the turn state the expansion and the C API export - the MANDATORY boards
// still to move, the OPTIONAL boards still playable, the MOVES made in the
// action - against a recount written here from the rulebook's definitions, at
// every state of random play under every ruleset (both Q4 readings among
// them), through t5d/expand.hpp and through a t5d_game handle playing the same
// moves. And the spec's arithmetic - "-1 per ordinary move, -2 for a move
// between two present boards, 0 after a rewinding branch" - wherever no
// activity changes, with what it does not say counted apart: T6's optional
// board moving onto a mandatory one, and T11/T12's reactivations, which can
// raise the count.

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "t5d/capi.h"
#include "t5d/expand.hpp"
#include "t5d/movegen.hpp"
#include "t5d/records.h"
#include "t5d/rules.hpp"
#include "t5d/state.hpp"
#include "test_framework.hpp"
#include "test_positions.hpp"
#include "test_rulesets.hpp"
#include "test_support.hpp"

using namespace t5d;

namespace {

// ---- the recount, from the definitions -------------------------------------------------------
//
// A timeline COUNTS for the present when it is active (T10) and not deferred
// (Q6); the PRESENT is the least head half-turn of the counting timelines
// (T4); a board is PLAYABLE by the mover when it is its timeline's head with
// the mover to move on it (L5); it is MANDATORY when it is a counting
// timeline's head with the mover to move in the present (T6) - anywhere, under
// Q5's AllActive reading; OPTIONAL when playable and not mandatory, while a
// move may be made (Q4: always, R8; under the Guide-literal reading only while
// a mandatory board is left).

struct Recount {
  int present = INT_MAX;
  std::set<int> counting;   // timelines
  std::set<int> playable;
  std::set<int> mandatory;
  std::set<int> optional;
};

bool deferred(const GameState& s, int p) {
  for (const GameState::Deferral& d : s.deferred_timelines()) {
    if (d.timeline == p) return true;
  }
  return false;
}

Recount recount(const GameState& s) {
  Recount r;
  const Multiverse& mv = s.multiverse();
  for (const TimelineRef& t : mv.timelines()) {
    if (s.active(t->position()) && !deferred(s, t->position())) {
      r.counting.insert(t->position());
      r.present = std::min(r.present, t->last_half_turn());
    }
  }
  for (const TimelineRef& t : mv.timelines()) {
    const int p = t->position();
    const int h = t->last_half_turn();
    if (to_move_at(h) != s.to_move()) continue;
    r.playable.insert(p);
    if (r.counting.count(p) != 0 && (s.rules().mandatory == MandatoryBoards::AllActive || h == r.present)) {
      r.mandatory.insert(p);
    }
  }
  const bool may_move = s.rules().moves_after_shift || !r.mandatory.empty();
  for (const int p : r.playable) {
    if (may_move && r.mandatory.count(p) == 0) r.optional.insert(p);
  }
  return r;
}

int size_of(const std::set<int>& s) { return static_cast<int>(s.size()); }

// ---- what the counts met -----------------------------------------------------------------------

struct Tally {
  long long states = 0;
  long long turn_starts = 0;
  long long mid_action = 0;
  long long after_shift = 0;          // states with the obligations met and a move still allowed (R8)
  long long no_move_after_shift = 0;  // ... and with none allowed (Q4's Guide-literal reading)
  long long with_optional = 0;        // states with an optional board
  long long arithmetic = 0;           // moves where no activity changed: the spec's arithmetic checked
  long long ordinary = 0;             // ... one mandatory board played: -1
  long long two_present = 0;          // ... a jump between two mandatory boards: -2
  long long optional_onto_mandatory = 0;  // ... T6: from an optional board onto a mandatory one: -1
  long long mandatory_onto_optional = 0;  // ... from a mandatory board onto an optional one: -1
  long long optional_only = 0;        // ... no mandatory board played: 0
  long long rewinding_branches = 0;   // a branch whose new timeline counts behind the present: 0 (T8)
  long long rewinds_released = 0;     // ... releasing a mandatory board it did not play
  long long activity_changes = 0;     // moves that changed what counts (T11, T12, Q1, Q3, Q6)
  long long raised = 0;               // ... after which the count is above the arithmetic's
  long long rose = 0;                 // ... above the count before the move (T12's rewind onto a board owed)
  long long capi_states = 0;

  void print(const char* name) const {
    std::fprintf(stderr,
                 "  %s: %lld states (%lld turn starts, %lld mid-action, %lld after the shift with moves, %lld "
                 "without; %lld with an optional board); arithmetic on %lld moves: %lld ordinary (-1), %lld "
                 "between two present boards (-2), %lld optional onto mandatory (T6), %lld mandatory onto "
                 "optional, %lld on optional boards only (0); %lld rewinding branches (0; %lld releasing an "
                 "unplayed board); %lld activity changes (%lld above the arithmetic, %lld above the count "
                 "before); %lld C API states\n",
                 name, states, turn_starts, mid_action, after_shift, no_move_after_shift, with_optional, arithmetic,
                 ordinary, two_present, optional_onto_mandatory, mandatory_onto_optional, optional_only,
                 rewinding_branches, rewinds_released, activity_changes, raised, rose, capi_states);
  }
};

// The expansion's counts (t5d/expand.hpp, h) at `s` with `moves_made` moves made.
void check_expansion(const GameState& s, int moves_made, Tally& tally) {
  ExpandOptions o;
  o.square_features = SquareFeatures::None;
  Expansion x;
  expand(s, o, x, moves_made);
  const Recount r = recount(s);
  CHECK_EQ(x.mandatory_remaining(), size_of(r.mandatory));
  CHECK_EQ(x.optional_playable(), size_of(r.optional));
  CHECK_EQ(x.moves_made(), moves_made);
  CHECK_EQ(x.mandatory_remaining(), s.mandatory_count());
  // T5: submitting is possible exactly when no mandatory board is left; a turn
  // start never has its obligations met.
  CHECK_EQ(x.submit_possible(), x.mandatory_remaining() == 0);
  if (moves_made == 0) CHECK(x.mandatory_remaining() > 0);
  // The boards (a)'s moves start from: the mandatory and the optional ones.
  int marked = 0;
  for (const std::uint8_t f : x.board_playable()) marked += f;
  CHECK_EQ(marked, x.mandatory_remaining() + x.optional_playable());
  ++tally.states;
  if (moves_made == 0) {
    ++tally.turn_starts;
  } else {
    ++tally.mid_action;
  }
  if (x.submit_possible()) ++(x.may_move() ? tally.after_shift : tally.no_move_after_shift);
  if (x.optional_playable() > 0) ++tally.with_optional;
}

// The spec's arithmetic for the move `m` from `s` to `t` (A6: "-1 per ordinary
// move, -2 for a move between two present boards, 0 after a rewinding
// branch"), wherever no timeline's counting changed; T6's arrivals counted.
void check_delta(const GameState& s, const Move& m, const GameState& t, Tally& tally) {
  const Recount a = recount(s);
  const Recount b = recount(t);
  // Did any timeline that existed start or stop counting (T11/T12's
  // reactivation, Q1, Q3's play activation, Q6's deferrals)?
  bool changed = false;
  for (const TimelineRef& tl : s.multiverse().timelines()) {
    const int p = tl->position();
    if ((a.counting.count(p) != 0) != (b.counting.count(p) != 0)) changed = true;
  }
  // A branch's new timeline: whether it counts, and behind the present.
  bool rewinds = false;
  if (m.branching()) {
    for (const TimelineRef& tl : t.multiverse().timelines()) {
      if (s.multiverse().has_timeline(tl->position())) continue;
      if (b.counting.count(tl->position()) != 0 && tl->last_half_turn() < a.present) rewinds = true;
    }
  }
  const bool src_mandatory = a.mandatory.count(m.from_timeline) != 0;
  // A jump that does not branch lands on its destination timeline's playable
  // board and plays it too (T3, T6, J4).
  const bool lands = m.jump() && !m.branching();
  const bool dst_mandatory = lands && a.mandatory.count(m.to_timeline) != 0;
  const int played = (src_mandatory ? 1 : 0) + (dst_mandatory ? 1 : 0);
  if (changed) {
    ++tally.activity_changes;
    if (size_of(b.mandatory) > size_of(a.mandatory) - played) ++tally.raised;
    if (size_of(b.mandatory) > size_of(a.mandatory)) ++tally.rose;
    return;
  }
  if (rewinds) {
    ++tally.rewinding_branches;
    if (s.rules().mandatory == MandatoryBoards::InPresent) {
      // T8: the present rewinds onto the new board - the opponent's colour.
      CHECK_EQ(size_of(b.mandatory), 0);
      if (size_of(a.mandatory) > played) ++tally.rewinds_released;
    } else {
      CHECK_EQ(size_of(b.mandatory), size_of(a.mandatory) - played);  // Q5: every active board still owed
    }
    return;
  }
  ++tally.arithmetic;
  CHECK_EQ(size_of(b.mandatory), size_of(a.mandatory) - played);
  if (played == 2) {
    ++tally.two_present;
  } else if (played == 1 && lands) {
    ++(src_mandatory ? tally.mandatory_onto_optional : tally.optional_onto_mandatory);
  } else if (played == 1) {
    ++tally.ordinary;
  } else {
    ++tally.optional_only;
  }
}

// ---- the C API's game handle, beside --------------------------------------------------------------

struct Handle {
  t5d_game* p = nullptr;
  Handle() = default;
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  ~Handle() { t5d_game_destroy(p); }
  void reset(t5d_game* q) {
    t5d_game_destroy(p);
    p = q;
  }
};

t5d_game* copy_of(const t5d_game* g) {
  t5d_game* out = nullptr;
  REQUIRE_EQ(t5d_game_copy(g, &out), T5D_OK);
  return out;
}

// The handle's counts at its state: its expansion's and its summary's.
void check_handle(t5d_game* g, const GameState& s, int moves_made, Tally& tally) {
  t5d_expand_options o{};
  o.window_k = 0;
  o.square_features = T5D_FEATURES_NONE;
  t5d_expansion v{};
  REQUIRE_EQ(t5d_game_expand(g, &o, &v, nullptr), T5D_OK);
  const Recount r = recount(s);
  CHECK_EQ(v.mandatory_remaining, size_of(r.mandatory));
  CHECK_EQ(v.optional_playable, size_of(r.optional));
  CHECK_EQ(v.moves_made, moves_made);
  t5d_game_summary info{};
  REQUIRE_EQ(t5d_game_info(g, &info), T5D_OK);
  CHECK_EQ(info.moves_made, moves_made);
  CHECK_EQ(info.obligations_met, r.mandatory.empty() ? 1 : 0);
  ++tally.capi_states;
}

t5d_move record_of(const Move& m) {
  t5d_move r{};
  r.from_timeline = m.from_timeline;
  r.from_half_turn = m.from_half_turn;
  r.to_timeline = m.to_timeline;
  r.to_half_turn = m.to_half_turn;
  r.from_square = m.from_square;
  r.to_square = m.to_square;
  r.promotion = static_cast<std::uint8_t>(m.promotes() ? index_of(m.promotion) + 1 : 0);
  return r;
}

// ---- random play -------------------------------------------------------------------------------------

struct Start {
  const char* name;
  std::vector<std::string> boards;
  BoardSize size;
};

std::string standard_board() { return "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*"; }

std::vector<Start> starts() {
  return {
      {"Standard", {"[" + standard_board() + ":0:1:w]"}, BoardSize{8, 8}},
      {"Two Timelines", {"[" + standard_board() + ":-0:1:w]", "[" + standard_board() + ":+0:1:w]"}, BoardSize{8, 8}},
      {"Very Small", {"[nbr*k*/p*p*p*p*/P*P*P*P*/K*R*BN:0:1:w]"}, BoardSize{4, 4}},
      {"Timeline Battleground",
       {"[r*r*k*r*r*/bbqbb/p*p*p*p*p*/5/P*P*P*P*P*:-1:1:w]", "[nnnnn/p*p*p*p*p*/5/P*P*P*P*P*/NNNNN:0:1:w]",
        "[p*p*p*p*p*/5/P*P*P*P*P*/BBQBB/R*R*K*R*R*:+1:1:w]"},
       BoardSize{5, 5}},
  };
}

// Plays `actions` random pseudo-legal actions from `start` under `rules`: at
// each state the expansion's and the handle's counts against the recount, at
// each move the arithmetic. An action is random moves - after the shift, while
// a move is still allowed, one more with probability 1/2 (R8) - then the
// submission; a dead end (no move, obligations unmet) retries the action from
// its turn start, and a turn with no action ends the game.
void random_game(const Start& start, const RuleOptions& rules, std::uint64_t seed, int actions, Tally& tally) {
  t5d_test::Rng rng(seed);
  GameState s = t5d_test::game(start.boards, start.size, rules);
  std::vector<const char*> strings;
  for (const std::string& b : start.boards) strings.push_back(b.c_str());
  Handle game;
  REQUIRE_EQ(t5d_game_new_from_fen(strings.data(), strings.size(), start.size.width, start.size.height,
                                   T5D_PARITY_INFER, T5D_TO_MOVE_INFER, rules.encode(), &game.p, nullptr),
             T5D_OK);
  REQUIRE_EQ(t5d_game_set_legality(game.p, T5D_PSEUDO_LEGAL, nullptr), T5D_OK);  // pseudo-legal play
  MoveList moves;
  for (int a = 0; a < actions; ++a) {
    const GameState turn_start = s;
    Handle at_start;
    at_start.reset(copy_of(game.p));
    bool done = false;
    for (int attempt = 0; attempt < 6 && !done; ++attempt) {
      s = turn_start;
      game.reset(copy_of(at_start.p));  // the copy keeps the count: 0
      int made = 0;
      for (int step = 0; step < 40; ++step) {
        check_expansion(s, made, tally);
        check_handle(game.p, s, made, tally);
        if (s.obligations_met() && (!s.may_move() || rng.chance(50))) {
          done = true;
          break;
        }
        generate_moves(s, moves);
        if (moves.empty()) {
          done = s.obligations_met();
          break;
        }
        const Move m = moves[static_cast<std::size_t>(rng.below(static_cast<int>(moves.size())))];
        GameState t = s;
        t.apply(m);
        check_delta(s, m, t, tally);
        const t5d_move r = record_of(m);
        REQUIRE_EQ(t5d_game_apply(game.p, &r, nullptr), T5D_OK);
        s = t;
        ++made;
      }
    }
    if (!done) return;
    s.submit(CheckRuling::none());
    REQUIRE_EQ(t5d_game_submit(game.p, nullptr), T5D_OK);
  }
  check_expansion(s, 0, tally);
  check_handle(game.p, s, 0, tally);
}

void every_ruleset_property(int games, int actions, std::uint64_t seed, Tally& tally) {
  for (const RuleOptions& rules : t5d_test::every_ruleset()) {
    for (const Start& start : starts()) {
      for (int g = 0; g < games; ++g) random_game(start, rules, seed++, actions, tally);
    }
  }
}

}  // namespace

TEST(turn_state_counts_equal_a_recount_under_every_ruleset) {
  Tally tally;
  every_ruleset_property(1, 10, 7100, tally);
  tally.print("every ruleset");
  // Every case the counts meet was met (the runs are deterministic; the
  // thresholds about a third of what they met when written).
  CHECK(tally.states > 3000);
  CHECK(tally.capi_states == tally.states);
  CHECK(tally.mid_action > 2000);
  CHECK(tally.after_shift > 400);
  CHECK(tally.no_move_after_shift > 10);  // Q4's Guide-literal reading is among the rulesets
  CHECK(tally.with_optional > 500);
  CHECK(tally.ordinary > 500);
  CHECK(tally.two_present > 60);
  CHECK(tally.optional_onto_mandatory > 5);
  CHECK(tally.mandatory_onto_optional > 10);
  CHECK(tally.optional_only > 200);
  CHECK(tally.rewinding_branches > 40);
  CHECK(tally.rewinds_released > 20);
  CHECK(tally.activity_changes > 40);
  CHECK(tally.raised > 9);
  CHECK(tally.rose > 3);  // T12: a reactivation rewinding onto a board of the mover's
}

TEST(long_turn_state_counts_equal_a_recount_under_every_ruleset) {
  Tally tally;
  every_ruleset_property(3, 24, 7200, tally);
  tally.print("every ruleset, longer games");
  CHECK(tally.states > 25000);
  CHECK(tally.two_present > 600);
  CHECK(tally.optional_onto_mandatory > 150);
  CHECK(tally.rewinding_branches > 200);
  CHECK(tally.rose > 200);
}

namespace {

struct Counts {
  int mandatory = 0;
  int optional = 0;
  int made = 0;
  friend bool operator==(const Counts&, const Counts&) = default;
};

Counts counts(const GameState& s, int made) {
  ExpandOptions o;
  o.square_features = SquareFeatures::None;
  Expansion x;
  expand(s, o, x, made);
  return Counts{x.mandatory_remaining(), x.optional_playable(), x.moves_made()};
}

}  // namespace

TEST(turn_state_the_specs_cases_by_construction) {
  using t5d_test::game;
  using t5d_test::play;
  for (const bool after_shift : {true, false}) {
    RuleOptions rules;
    rules.moves_after_shift = after_shift;  // R8, and Q4's Guide-literal reading
    // "-2 for a move between two present boards": -0 and +0 at T1, the rook from
    // -0's a1 to +0's a1 along the timeline axis - a jump onto +0's playable
    // board, which does not branch, so it plays both (T3, T6, J4). One move.
    {
      GameState s = game({"[3k/4/4/R2K:-0:1:w]", "[3k/4/4/3K:+0:1:w]"}, BoardSize{4, 4}, rules);
      CHECK((counts(s, 0) == Counts{2, 0, 0}));
      play(s, "(-0T1)Ra1(+0T1)a1");
      CHECK((counts(s, 1) == Counts{0, 0, 1}));
    }
    // T6, beside the spec's arithmetic: a piece from an optional board (+1, ahead
    // of the present) onto the mandatory one (0, in the present) - a jump that
    // does not branch: the mandatory board is played by the arrival, the
    // optional board by the departure.
    {
      GameState s = game({"[3k/4/4/3K:0:1:w]", "[3k/4/4/B2K:+1:2:w]"}, BoardSize{4, 4}, rules);
      CHECK((counts(s, 0) == Counts{1, 1, 0}));
      play(s, "(+1T2)Ba1(0T1)a1");
      CHECK((counts(s, 1) == Counts{0, 0, 1}));
    }
    // "0 once a branch rewinds the present": -0 and +0 at T2, both mandatory; the
    // knight jumps from -0's T2 back to -0's T1, a history board - a branch, whose
    // new timeline (White's first, active) starts at T1 with Black to move, behind
    // the present: the present rewinds onto it (T8), and +0's board - not
    // played - is no longer owed. It stays playable: optional under R8; under
    // Q4's Guide-literal reading no move may follow the shift.
    {
      GameState s = game({"[3k/4/4/K3:-0:1:w]", "[3k/4/4/K3:-0:1:b]", "[3k/4/4/KN2:-0:2:w]", "[3k/4/4/K3:+0:1:w]",
                          "[3k/4/4/K3:+0:1:b]", "[3k/4/4/K3:+0:2:w]"},
                         BoardSize{4, 4}, rules);
      CHECK((counts(s, 0) == Counts{2, 0, 0}));
      const Move m = play(s, "(-0T2)Nb1(-0T1)b3");
      CHECK(m.branching());
      CHECK((counts(s, 1) == Counts{0, after_shift ? 1 : 0, 1}));
      CHECK(s.obligations_met());
    }
    // Moves after the shift (R8): the optional boards stay playable until the
    // submission; the count of moves goes on.
    {
      GameState s = game({"[3k/4/4/K3:0:1:w]", "[3k/4/4/K3:+1:2:w]"}, BoardSize{4, 4}, rules);
      CHECK((counts(s, 0) == Counts{1, 1, 0}));
      play(s, "(0T1)Ka1(0T1)a2");
      CHECK((counts(s, 1) == Counts{0, after_shift ? 1 : 0, 1}));
      if (after_shift) {
        play(s, "(+1T2)Ka1(+1T2)b1");
        CHECK((counts(s, 2) == Counts{0, 0, 2}));
      }
    }
  }
  // The count is the caller's: a negative one is refused.
  Expansion x;
  CHECK_THROWS_AS(expand(t5d_test::game({"[3k/4/4/K3:0:1:w]"}, BoardSize{4, 4}), ExpandOptions{}, x, -1),
                  std::invalid_argument);
}
