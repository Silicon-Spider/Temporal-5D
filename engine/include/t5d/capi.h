/* t5d/capi.h - the rules engine's C API (the t5d shared library).
 *
 * For callers outside C++ - Python's ctypes first. Conventions:
 *
 *  - Handles are opaque: a t5d_state is created by a t5d_state_new_* or
 *    t5d_state_copy function and released with t5d_state_destroy.
 *  - Every function that can fail returns a t5d_status; T5D_OK is 0. No C++
 *    exception ever crosses this boundary.
 *  - Strings are UTF-8 (in practice ASCII), NUL-terminated. A string the
 *    library returns through a char** is allocated by the library and released
 *    by the caller with t5d_free. Where a function takes `char** error`, the
 *    argument may be NULL; otherwise it receives NULL on success and, on
 *    failure, a message to free with t5d_free (or NULL if even that could not
 *    be allocated).
 *  - A state is never modified by any function here, so one state may be read
 *    from several threads at once. A game (t5d_game, below: a state under the
 *    rules with its expansion buffers) is changed by the functions that move,
 *    submit, replay or expand it, and is used from one thread at a time.
 *  - The arrays an expansion hands out (t5d_expansion) belong to the game
 *    handle and are valid until that handle is next changed; copy what must
 *    outlive that.
 *
 * Timelines are integer positions p (adjacent timelines differ by 1; in a game
 * with an even number of starting timelines "-0" is -1 and "+0" is 0) and times
 * are half-turn indices h = 2*T + (0 white to move, 1 black), T as displayed.
 * The engine's README describes both.
 */
#ifndef T5D_CAPI_H
#define T5D_CAPI_H

#include <stddef.h>
#include <stdint.h>

#include "t5d/records.h"

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(T5D_BUILDING_DLL)
#    define T5D_API __declspec(dllexport)
#  elif defined(T5D_USING_DLL)
#    define T5D_API __declspec(dllimport)
#  else
#    define T5D_API
#  endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#  define T5D_API __attribute__((visibility("default")))
#else
#  define T5D_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- status codes ------------------------------------------------------- */

typedef int32_t t5d_status;

#define T5D_OK 0
#define T5D_ERR_ARGUMENT 1 /* a NULL pointer, an invalid size, an index out of range, a
                              player to move that cannot be inferred */
#define T5D_ERR_NOTATION 2 /* malformed 5DFEN, or boards that form no multiverse; a 5dpgn
                              record that cannot be read */
#define T5D_ERR_MEMORY 3   /* out of memory */
#define T5D_ERR_INTERNAL 4 /* anything else: a bug */
#define T5D_ERR_UNSUPPORTED 5 /* a game or a feature the engine does not implement: a variant
                                 it does not set up, a relation it does not build */
#define T5D_ERR_ILLEGAL 6     /* a move or a submission the rules refuse - a submission
                                 that leaves a royal piece capturable included (C4, R1) */

/* ---- parity: does the game start with an even number of timelines? ------- */

#define T5D_PARITY_INFER (-1) /* decide from the labels: "-0"/"+0" even, "0" odd */
#define T5D_PARITY_ODD 0
#define T5D_PARITY_EVEN 1

/* ---- the player to move ---------------------------------------------------- */

#define T5D_TO_MOVE_INFER (-1) /* only where the boards settle it; see t5d_state_new_from_fen */
#define T5D_TO_MOVE_WHITE 0
#define T5D_TO_MOVE_BLACK 1

/* ---- types --------------------------------------------------------------- */

typedef struct t5d_state t5d_state;

/* The canonical fingerprint (decision H30): see the engine README. */
typedef struct t5d_fingerprint {
  char fingerprint[33]; /* lower-case hex MD5 of the canonical state string */
  char placement[33];   /* the same with every '*' removed */
  uint64_t boards;      /* number of boards, superseded and head */
  uint64_t pieces;      /* number of pieces on all boards */
} t5d_fingerprint;

typedef struct t5d_info {
  uint64_t boards;
  uint64_t pieces;
  int32_t width;
  int32_t height;
  int32_t even;    /* 1: the game started with an even number of timelines */
  int32_t to_move; /* 0 white, 1 black: as given, or inferred (t5d_state_new_from_fen) */
  int32_t min_timeline;
  int32_t max_timeline;
  int32_t created_by_white; /* timelines created by each player */
  int32_t created_by_black;
  uint32_t ruleset;
} t5d_info;

/* ---- library -------------------------------------------------------------- */

/* The engine version, e.g. "0.1.0". A static string: do not free it. */
T5D_API const char* t5d_version(void);

/* Releases memory the library returned (strings). NULL is ignored. */
T5D_API void t5d_free(void* pointer);

/* The struct layouts this library was built with, for a binding that mirrors them
 * (python/t5d/engine/capi.py checks every one at load, so a binding and a library of
 * different stages refuse each other instead of writing past a buffer): with `field`
 * NULL, sizeof(`type`); otherwise the offset of `field` in `type`. `type` is one of
 * "t5d_fingerprint", "t5d_info", "t5d_move", "t5d_expand_options",
 * "t5d_game_summary", "t5d_expansion"; `field` is a member's name, as declared.
 * T5D_ERR_ARGUMENT for a NULL `type` or `out`, an unknown type or an unknown field.
 * (Stage E7.) */
T5D_API t5d_status t5d_layout(const char* type, const char* field, size_t* out);

/* ---- creating and releasing states ----------------------------------------- */

/* "Standard": the standard board on timeline 0 at T1, white to move. */
T5D_API t5d_status t5d_state_new_standard(t5d_state** out);

/* "Standard - Two Timelines": the standard board on -0 and +0 at T1. */
T5D_API t5d_status t5d_state_new_standard_two_timelines(t5d_state** out);

/* A state from `count` 5DFEN board strings ("[pieces:timeline:turn:colour]").
 * width, height: the board size (the Size header); 0 and 0 to take it from
 *   the first board string.
 * parity: T5D_PARITY_ODD, T5D_PARITY_EVEN or T5D_PARITY_INFER.
 * to_move: the player whose turn it is - T5D_TO_MOVE_WHITE or
 *   T5D_TO_MOVE_BLACK, taken as given - or T5D_TO_MOVE_INFER. 5DFEN does not
 *   record it, and inferring it in general needs the timeline-activity rule,
 *   which a later stage implements; until then INFER succeeds only where the
 *   boards settle it whatever that rule says (every head that could be the
 *   present has the same colour to move - every starting position does) and
 *   fails otherwise with T5D_ERR_ARGUMENT and a message naming two heads that
 *   could be the present. It feeds the Zobrist key and t5d_info.to_move; the
 *   fingerprint does not depend on it.
 * ruleset: the ruleset (variant rules) ID folded into the Zobrist key; 0 for
 *   the standard rules.
 * On failure *out is set to NULL. */
T5D_API t5d_status t5d_state_new_from_fen(const char* const* boards, size_t count,
                                          int32_t width, int32_t height, int32_t parity,
                                          int32_t to_move, uint32_t ruleset, t5d_state** out,
                                          char** error);

/* An independent copy (it shares the immutable boards). */
T5D_API t5d_status t5d_state_copy(const t5d_state* state, t5d_state** out);

/* Releases a state. NULL is ignored. */
T5D_API void t5d_state_destroy(t5d_state* state);

/* ---- reading a state ------------------------------------------------------- */

T5D_API t5d_status t5d_state_info(const t5d_state* state, t5d_info* out);

T5D_API t5d_status t5d_state_fingerprint(const t5d_state* state, t5d_fingerprint* out);

/* The canonical state string the fingerprint hashes. Free with t5d_free. */
T5D_API t5d_status t5d_state_canonical(const t5d_state* state, char** out);

/* The 5D Zobrist key (decision H18). */
T5D_API t5d_status t5d_state_zobrist(const t5d_state* state, uint64_t* out);

/* The number of boards; board indices run from 0 to count - 1 in the canonical
 * order (ascending timeline, then half-turn). */
T5D_API t5d_status t5d_state_board_count(const t5d_state* state, size_t* out);

/* The board at `index`: its timeline position, its half-turn and its canonical
 * 5DFEN board string (free with t5d_free). Any output pointer may be NULL. */
T5D_API t5d_status t5d_state_board(const t5d_state* state, size_t index, int32_t* timeline,
                                   int32_t* half_turn, char** fen);

/* =============================================================================
 * Games (stage E3): a game state under the rules - the player to move, the
 * present, the action in progress - with its expansion (t5d/expand.hpp, the
 * engine README's "Expansion") and the number of moves made in the action in
 * progress (register A6: no state records it - t5d_game_summary.moves_made,
 * t5d_expansion.moves_made). Every game is made at a turn start (count 0).
 * Moves are pseudo-legal; submissions are
 * LEGAL: a submission that leaves a royal piece of the mover's capturable is
 * refused (C4, R1; stage E4), and a game read from a record replays it so.
 *
 * A t5d_game handle is not thread-safe (it holds the expansion's buffers);
 * different handles may be used from different threads at once.
 *
 * Rulesets are passed as RuleOptions' canonical 64-bit encoding
 * (t5d/rules.hpp, RuleOptions::encode): 0 = the standard rules - with moves
 * after the present has shifted, until submission (Q4, ruled R8); bit 3 =
 * Q4's Guide-literal reading, no move once the present has shifted (a
 * diagnostic); the engine README lists every option. Each bit marks a choice
 * that differs from the default.
 * ============================================================================= */

typedef struct t5d_game t5d_game;

#define T5D_RULES_STANDARD UINT64_C(0)
/* Q4's Guide-literal reading: no move once the present has shifted. Bit 3 was
 * named T5D_RULES_MOVES_AFTER_SHIFT, and meant the opposite, until R8 made
 * moves after the shift the default (stage E7). */
#define T5D_RULES_NO_MOVES_AFTER_SHIFT (UINT64_C(1) << 3)
/* Moves after the shift: the default since R8, so no bit - kept so that a
 * caller that asked for moves after the shift still gets them. */
#define T5D_RULES_MOVES_AFTER_SHIFT UINT64_C(0)

/* Which boards get per-square features (t5d_expand_options.square_features). */
#define T5D_FEATURES_ALL 0   /* every board: the full-history input */
#define T5D_FEATURES_SCOPE 1 /* W(k), E(k) and src_extra(k): the windowed input */
#define T5D_FEATURES_NONE 2

/* Bits of t5d_expand_options.relations: 1 << relation id (t5d/records.h) -
 * register B11's five types (t5d/expand.hpp, (e)). The move graph is always
 * made; T5D_RELATION_RESERVED (6) is refused (T5D_ERR_UNSUPPORTED). Pin and
 * check (engine ABI round 3) read both sides, each from its own view - the
 * opponent's the multiverse check status passes (R1) - and make the expansion
 * allocate. */
#define T5D_RELATIONS_ATTACK (1u << T5D_RELATION_ATTACK)
#define T5D_RELATIONS_DEFENCE (1u << T5D_RELATION_DEFENCE)
#define T5D_RELATIONS_PIN (1u << T5D_RELATION_PIN)
#define T5D_RELATIONS_XRAY (1u << T5D_RELATION_XRAY)
#define T5D_RELATIONS_CHECK (1u << T5D_RELATION_CHECK)

/* window_depth and band_depth of a board in no window. */
#define T5D_NO_WINDOW INT32_MAX

typedef struct t5d_expand_options {
  int32_t window_k;        /* k of W(k), E(k), src_extra(k): half-turns, >= 0 (default 8);
                              any k at least a timeline's length takes its whole band */
  int32_t square_features; /* T5D_FEATURES_* (default ALL) */
  uint32_t relations;      /* T5D_RELATIONS_* bits: the typed relations - provisional, B11
                              (default 0: none) */
  /* appended in engine ABI round 2 (register I15): the legal-moves filter of the search's
     leaves, for t5d_game_expand - so that a record's network input (its candidate moves)
     can be rebuilt exactly. A search refuses it in t5d_search_params.expand (its own
     legal_moves_only and legality_checks/_nodes decide its leaves). */
  int32_t legal_moves;      /* 1: the moves after which the action can still be finished
                               legally (C4, R1), classified by E5a's tiers under the budget
                               below - a move proved illegal dropped, an unknown one kept;
                               0 (default): the pseudo-legal moves */
  uint64_t legality_checks; /* the classification's budget, as t5d_search_params':
                               checks ... */
  uint64_t legality_nodes;  /* ... and search nodes (T5D_NO_LIMIT both: exact; read only
                               with legal_moves) */
} t5d_expand_options;

typedef struct t5d_game_summary {
  uint64_t boards;
  int32_t width;
  int32_t height;
  int32_t even;             /* 1: the game started with an even number of timelines */
  int32_t to_move;          /* the player to move: 0 white, 1 black */
  int32_t present;          /* the present (T4): a half-turn */
  int32_t timelines;
  int32_t active_timelines; /* T10 */
  int32_t obligations_met;  /* 1: the present has shifted to the opponent (T5) */
  int32_t may_move;         /* 1: a move may be made now (by default always, R8; under Q4's
                               Guide-literal reading only before the shift) */
  int32_t may_submit;       /* 1: the action may be submitted: the present shifted (T5), every royal safe (C4) */
  uint32_t ruleset;         /* the ruleset ID (0: standard) */
  uint64_t rules;           /* the ruleset's canonical encoding */
  /* appended in engine ABI round 1 (register A6): */
  int32_t moves_made;       /* the moves made in the action in progress (the handle's
                               count - no state records it): 0 at a turn start, one more
                               per t5d_game_apply or _apply_expanded, 0 again after
                               t5d_game_submit and t5d_game_replay_next */
} t5d_game_summary;

/* The arrays of one expansion: every pointer stays valid until the next
 * t5d_game_expand, t5d_game_apply, t5d_game_apply_expanded, t5d_game_submit,
 * t5d_game_replay_next or t5d_game_destroy on the same handle. Boards are
 * indices in the canonical order (ascending timeline, then half-turn); cells
 * are rank * width + file (t5d/records.h). Board sets are ascending. W(k) is
 * every active timeline's band - its head and the k boards behind it - and
 * branch origin; E(k) the boards outside W(k) that a travel move (destination
 * board != source board) lands on. */
typedef struct t5d_expansion {
  int32_t width;
  int32_t height;
  int32_t even;
  int32_t to_move;
  int32_t present;
  int32_t timelines;
  int32_t active_timelines;
  int32_t window_k;
  int32_t may_move;
  int32_t origins_unknown; /* active branched timelines with no recorded origin (5DFEN) */

  size_t board_count;
  size_t move_count;
  size_t edge_count;
  size_t relation_count;
  size_t window_count;
  size_t e_count;
  size_t src_extra_count;
  size_t feature_board_count;
  size_t origin_count;     /* distinct boards that are an active timeline's branch origin */

  /* per board (board_count) */
  const int32_t* timeline;     /* timeline position p ("-0" -> -1, "+0" -> 0, ...) */
  const int32_t* half_turn;    /* 2T + (1 if black to move on the board) */
  const uint8_t* is_head;
  const uint8_t* active;       /* its timeline is active */
  const uint8_t* side_to_move; /* 0 white, 1 black */
  const uint8_t* playable;     /* the player to move may move from it now */
  const int32_t* window_depth; /* the least k with the board in W(k); T5D_NO_WINDOW: none */
  const int32_t* band_depth;   /* the same without the branch origins (its active
                                  timeline's band only); T5D_NO_WINDOW: none */
  const uint8_t* is_origin;    /* 1: an active timeline's branch origin (in W(k) at every k) */
  const int32_t* parent;       /* the lineage (python/t5d/net/state.py `parent`): the board it
                                  was made from - its timeline's previous board, or the
                                  branch origin of a timeline's first board; -1: none known */

  /* per square: feature_board_count blocks of width * height cells */
  const int32_t* feature_boards; /* the board of each block */
  const uint8_t* piece;          /* 0 empty, 1..12 = P N B R Q K U D S W C Y */
  const uint8_t* colour;         /* 0 none, 1 white, 2 black */
  const uint8_t* moved;          /* has-moved: a pawn, rook, king or brawn without '*' */

  const t5d_move* moves;    /* move_count pseudo-legal moves */
  const int32_t* edges;     /* edge_count x (src board, src cell, dst board, dst cell) */
  const int32_t* relations; /* relation_count x (src board, src cell, dst board, dst cell, id) */
  const int32_t* window;    /* W(k) */
  const int32_t* e_boards;  /* E(k): travel destinations outside W(k) */
  const int32_t* src_extra; /* src_extra(k): move sources outside W(k) */

  /* ---- appended at stage E7 (register D8; rulebook R8): every field above
   * keeps its offset. Turns have variable length, so submitting is an action;
   * these are global (the per-board and per-square features are unchanged). */
  int32_t acting_player;   /* 0 white, 1 black: the side whose action this is (= to_move).
                              Once the present has shifted inside the action it differs
                              from the colour to move on the present (present & 1) */
  int32_t submit_possible; /* 1: submitting is possible now - the action's obligations are
                              met (T5) - so under the ply split it is an action beside the
                              moves */
  int32_t submit_legal;    /* 1: submitting is legal now - possible, and no royal piece of
                              the acting player's left capturable (C4, R1; exact) */

  /* ---- appended in engine ABI round 1 (register A6, spec 2.3: the turn state, exact
   * counts from the rules engine): every field above keeps its offset. Global, like the
   * three above. */
  int32_t mandatory_remaining; /* the boards the acting player must still play before
                                  submitting is possible (T5, T6; Q5, Q6): 0 exactly when
                                  submit_possible */
  int32_t optional_playable;   /* the acting player's other playable boards (L5: playable,
                                  not mandatory); 0 when no move may be made (Q4) */
  int32_t moves_made;          /* the moves already made in the action in progress, since
                                  its turn start (t5d_game_summary.moves_made for a game's
                                  expansion; a search leaf's from its path): 0 at a turn
                                  start. No state records it: equal boards with different
                                  counts are different network inputs */
} t5d_expansion;

/* A game from a 5dpgn record (`length` bytes of UTF-8 text; the record's
 * headers name the variant or give 5DFEN boards), set up at its first position
 * and ready to replay. `rules`: the ruleset encoding (the record's Promotions
 * header and variant family are added to it). Errors: T5D_ERR_NOTATION (the
 * set-up cannot be read), T5D_ERR_UNSUPPORTED (a game the engine does not
 * play), T5D_ERR_ARGUMENT (an invalid ruleset). On failure *out is NULL. */
T5D_API t5d_status t5d_game_new_from_pgn(const char* text, size_t length, uint64_t rules,
                                         t5d_game** out, char** error);

/* A game from 5DFEN board strings, as t5d_state_new_from_fen (the player to
 * move INFER: the colour to move on the present, Q9). */
T5D_API t5d_status t5d_game_new_from_fen(const char* const* boards, size_t count, int32_t width,
                                         int32_t height, int32_t parity, int32_t to_move,
                                         uint64_t rules, t5d_game** out, char** error);

/* A named variant's start ("Standard", "Standard - Two Timelines", ...). */
T5D_API t5d_status t5d_game_new_variant(const char* name, uint64_t rules, t5d_game** out,
                                        char** error);

/* An independent copy (sharing the immutable boards; not the expansion), with
 * the count of moves made in the action. */
T5D_API t5d_status t5d_game_copy(const t5d_game* game, t5d_game** out);

/* Releases a game. NULL is ignored. */
T5D_API void t5d_game_destroy(t5d_game* game);

T5D_API t5d_status t5d_game_info(const t5d_game* game, t5d_game_summary* out);

/* For a game made from a record: the actions the record holds, how many have
 * been replayed, and whether its move text stopped early (1: unreadable from
 * there on). Any output pointer may be NULL. For other games: 0, 0, 0. */
T5D_API t5d_status t5d_game_record_actions(const t5d_game* game, size_t* total, size_t* played,
                                           int32_t* truncated);

/* Replays the record's next action - its moves, then the submission, under
 * the game's legality (t5d_game_set_legality: by default legally, as the
 * gate adapter's full mode does) - to the next turn start (moves made: 0).
 * On failure the game is unchanged:
 * T5D_ERR_ILLEGAL (an illegal move or a submission the rules refuse),
 * T5D_ERR_NOTATION (a move that cannot be read, or fits several moves),
 * T5D_ERR_ARGUMENT (no action left, or the game has left its record by
 * t5d_game_apply or t5d_game_submit). */
T5D_API t5d_status t5d_game_replay_next(t5d_game* game, char** error);

/* Applies one move of the player to move: the generated move with the same
 * source and destination (timeline, half-turn, square) and promotion (0 for
 * none) - the other fields of `move` are not read. T5D_ERR_ILLEGAL if there is
 * none, or no move may be made now (under Q4's Guide-literal reading, once the
 * present has shifted). The action has one more move (moves made + 1). */
T5D_API t5d_status t5d_game_apply(t5d_game* game, const t5d_move* move, char** error);

/* Applies the move at `index` of the handle's last expansion, which must be
 * of the current state (T5D_ERR_ARGUMENT otherwise); moves made + 1. */
T5D_API t5d_status t5d_game_apply_expanded(t5d_game* game, size_t index, char** error);

/* Submits the action (T5): the opponent's turn start (moves made: 0).
 * T5D_ERR_ILLEGAL while the present has not shifted to the opponent, or if a
 * royal piece of the mover's is left capturable (C4, R1). */
T5D_API t5d_status t5d_game_submit(t5d_game* game, char** error);

/* What a game's submissions - and its record's replay - are tested for:
 * T5D_LEGAL (the default: the present's shift, T5, and every royal piece of
 * the mover's safe, C4/R1) or T5D_PSEUDO_LEGAL (the present's shift only -
 * for replaying pseudo-legal play, such as synthetic records, as the gate
 * adapter's parse mode does). T5D_ERR_ARGUMENT for any other value. */
#define T5D_PSEUDO_LEGAL 0
#define T5D_LEGAL 1
T5D_API t5d_status t5d_game_set_legality(t5d_game* game, int32_t legality, char** error);

/* Expands the current state into the handle's buffers and describes them in
 * *out, with the handle's count of moves made in the action (A6). `options`
 * may be NULL: k = 8, every board's features, no relations, the pseudo-legal
 * moves. With options->legal_moves, the moves are a search leaf's under the
 * same budget (t5d_search_params.legality_checks and _nodes): the same moves in
 * the same order, the classification being deterministic - a leaf of a
 * board-move search with legal_moves_only (the defaults; T5D_NO_LIMIT both here
 * is exact, the search's default budget is not). A whole-turn search's leaves,
 * and those of the pseudo-legal test seam, carry the pseudo-legal moves
 * (legal_moves 0). */
T5D_API t5d_status t5d_game_expand(t5d_game* game, const t5d_expand_options* options,
                                   t5d_expansion* out, char** error);

/* The game's multiverse as a t5d_state (fingerprint, canonical string, boards). */
T5D_API t5d_status t5d_game_state(const t5d_game* game, t5d_state** out);

/* ---- the per-square labels (engine ABI round 3; register D4, M20, A7) ----------
 * The auxiliary targets of section 12.4's per-square row ("deep attack facts
 * beyond the section 6 input bias"), exact from the rules engine: for every
 * square of every board (the canonical order, cells rank * width + file), the
 * eight labels of python/t5d/net/heads.py SQUARE_LABELS, in its order, relative
 * to the ACTING player (A7: "ours" its, "theirs" the opponent's). Each is 0 or
 * 1, or T5D_LABEL_UNKNOWN where a mate search stopped on its budget or an
 * exchange's recaptures could not be read (t5d/labels.hpp, item 0). What each
 * means - each a placeholder reading - is t5d/labels.hpp's and the engine
 * README's ("Per-square labels"). */

#define T5D_SQUARE_LABEL_COUNT 8
#define T5D_LABEL_EXCHANGE_LOSES_OCCUPANT 0
#define T5D_LABEL_REACHABLE_IN_TWO_OURS 1
#define T5D_LABEL_REACHABLE_IN_TWO_THEIRS 2
#define T5D_LABEL_ARRIVAL_GIVES_CHECK_OURS 3
#define T5D_LABEL_ARRIVAL_GIVES_CHECK_THEIRS 4
#define T5D_LABEL_ROYAL_ZONE_OURS 5
#define T5D_LABEL_ROYAL_ZONE_THEIRS 6
#define T5D_LABEL_ARRIVAL_GIVES_MATE 7
#define T5D_LABEL_UNKNOWN 255

typedef struct t5d_label_options {
  uint64_t mate_nodes; /* each mate search's node budget (arrival_gives_mate); UINT64_MAX
                          (T5D_NO_LIMIT): no node limit (the search's memory limit stays).
                          NULL options: 200000 (a placeholder) */
} t5d_label_options;

typedef struct t5d_square_labels {
  size_t board_count;
  int32_t width;
  int32_t height;
  int32_t label_count;   /* T5D_SQUARE_LABEL_COUNT */
  int32_t acting_player; /* 0 white, 1 black: whose "ours" */
  const uint8_t* labels; /* board_count x (width * height) x label_count */
  uint64_t mate_candidates;     /* the moves whose mate was searched (submission legal, check) */
  uint64_t mate_unknown;        /* ... and those the budget left undecided */
  uint64_t first_moves_ours;    /* the moves each view read (see t5d/labels.hpp) */
  uint64_t first_moves_theirs;
  uint64_t second_moves_ours;
  uint64_t second_moves_theirs;
} t5d_square_labels;

/* The per-square labels of the game's current state into the handle's buffer:
 * `out->labels` stays valid until the next t5d_game_square_labels or
 * t5d_game_destroy on the same handle. `options` may be NULL. */
T5D_API t5d_status t5d_game_square_labels(t5d_game* game, const t5d_label_options* options,
                                          t5d_square_labels* out, char** error);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* T5D_CAPI_H */
