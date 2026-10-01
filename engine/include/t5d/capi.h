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
 *    from several threads at once.
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
#define T5D_ERR_NOTATION 2 /* malformed 5DFEN, or boards that form no multiverse */
#define T5D_ERR_MEMORY 3   /* out of memory */
#define T5D_ERR_INTERNAL 4 /* anything else: a bug */

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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* T5D_CAPI_H */
