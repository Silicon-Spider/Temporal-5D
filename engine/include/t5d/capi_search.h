/* t5d/capi_search.h - the main search's C API (engine stage E6a; t5d/mcts.hpp),
 * in the t5d shared library beside t5d/capi.h, whose conventions it keeps:
 * opaque handles, a t5d_status from every function that can fail, no C++
 * exception across the boundary, `char** error` messages freed with t5d_free.
 *
 * THE STEP API lets a caller outside C++ - Python serving the network - be the
 * search's evaluator:
 *
 *   t5d_search_new(game, ...)                      a search from a game's state
 *   loop:
 *     t5d_search_gather(s, max_leaves, ...)        leaves to evaluate: n
 *     for i < n: t5d_search_leaf(s, i, &leaf)       each leaf's network input:
 *                                                    leaf.expansion (t5d/capi.h)
 *     t5d_search_deliver(s, logits, values, ...)   their evaluations, in order
 *   t5d_search_choose / _child / _advance ...
 *
 * What a leaf carries is everything the network takes from it - the leaf's
 * expansion (t5d_expansion: per-board and per-square features, the candidate
 * moves, the move graph, typed relations, W(k)/E(k)), its node key and the
 * refresh epoch. Its evaluation is:
 *
 *  - POLICY: one float32 logit per candidate move, logit i scoring
 *    leaf.expansion.moves[i] (the expansion's own order). Under the whole-turn
 *    split (D2's other arm) the candidates are the turn start's pseudo-legal
 *    moves, and the search derives each legal action's prior from them itself.
 *  - VALUE: 51 float32 numbers (T5D_VALUE_BINS; bins centred on -1 + 2k/50,
 *    register D3), as PROBABILITIES (T5D_VALUE_PROBABILITIES: non-negative, the
 *    search normalises them) or as LOGITS (T5D_VALUE_LOGITS: the search takes
 *    their softmax), from the perspective of the leaf's player to move - the
 *    player whose action is in progress (leaf.to_move; under the board-move
 *    split a leaf is never a state whose present has shifted under Q4's
 *    default reading).
 *
 * The pointers a leaf hands out are valid until the next t5d_search_gather,
 * t5d_search_deliver, t5d_search_abandon or t5d_search_destroy on the handle:
 * copy what must outlive that. A search handle is used from one thread at a
 * time. Every probability-like output is a double; counts are unsigned.
 */
#ifndef T5D_CAPI_SEARCH_H
#define T5D_CAPI_SEARCH_H

#include <stddef.h>
#include <stdint.h>

#include "t5d/capi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct t5d_search t5d_search;

#define T5D_VALUE_BINS 51

/* D2 (undecided): the board-move split, or whole turns. */
#define T5D_SPLIT_BOARD_MOVE 0
#define T5D_SPLIT_WHOLE_TURN 1
/* The whole-turn split's action prior from its moves' logits: their sum or mean. */
#define T5D_ACTION_PRIOR_SUM 0
#define T5D_ACTION_PRIOR_MEAN 1
/* The MCGS node key (spec issue 30): the 5D Zobrist key, or with branch origins. */
#define T5D_NODE_KEY_STATE 0
#define T5D_NODE_KEY_STATE_AND_ORIGINS 1
/* The final-selection rule (E5). */
#define T5D_FINAL_VISITS 0
#define T5D_FINAL_LCB 1
/* t5d_search_deliver: what `values` holds. */
#define T5D_VALUE_PROBABILITIES 0
#define T5D_VALUE_LOGITS 1
/* The built-in evaluators (t5d_search_run). */
#define T5D_EVALUATOR_UNIFORM 0
#define T5D_EVALUATOR_RANDOM 1
#define T5D_EVALUATOR_MATERIAL 2
/* A move's legality (t5d/legal.hpp): illegal moves are never children. */
#define T5D_LEGALITY_LEGAL 0
#define T5D_LEGALITY_ILLEGAL 1
#define T5D_LEGALITY_UNKNOWN 2
/* Why final selection chose a child (t5d_search_choose). */
#define T5D_CHOICE_PROVEN_WIN 0
#define T5D_CHOICE_SECURE_CHILD 1
#define T5D_CHOICE_LCB 2
#define T5D_CHOICE_VISITS 3
#define T5D_CHOICE_ALL_LOST 4
/* No limit, for the budgets below. */
#define T5D_NO_LIMIT UINT64_MAX

/* The search's parameters (t5d/mcts.hpp, SearchParams, field for field; the
 * engine README gives each default and its standing - most are placeholders).
 * Fill with t5d_search_params_default, then change what you need. The
 * persistent solved store is not here: it stays disabled until the generator
 * has passed the gate (register H23). */
typedef struct t5d_search_params {
  /* D2 */
  int32_t split;               /* T5D_SPLIT_* */
  int32_t action_prior;        /* T5D_ACTION_PRIOR_* (whole-turn) */
  uint64_t max_actions;        /* whole-turn: actions listed per node */
  uint64_t listing_nodes;      /* whole-turn: the listing's node budget */
  float submit_logit;          /* board-move, Q4's other reading: "submit"'s logit */
  int32_t legal_moves_only;    /* board-move: 1 (0 is a test seam: pseudo-legal children) */
  /* the legality of the board-move split's children (t5d/legal.hpp; spec issue 46) */
  uint64_t legality_checks;    /* (a) the expansion's budget: checks ... */
  uint64_t legality_nodes;     /*     ... and search nodes (T5D_NO_LIMIT both: exact) */
  double unknown_prior_scale;  /* (b) */
  int32_t verify_unknown;      /* (c) */
  uint64_t verify_checks;      /* (c) the verification's budget */
  uint64_t verify_nodes;
  /* E1 */
  double cpuct;
  int32_t widening;
  double widening_alpha;
  double widening_scale;
  double widening_min;
  double fpu_reduction;
  int32_t root_noise;
  double dirichlet_alpha;
  double dirichlet_epsilon;
  double policy_temperature;
  /* E2 */
  int32_t variance_scaled_cpuct;
  double cpuct_variance_scale;
  double variance_reference;
  double variance_prior_weight;
  double cpuct_scale_min;
  double cpuct_scale_max;
  int32_t uncertainty_weighted_backup;
  /* E3 */
  int32_t variance_bonus;
  double variance_bonus_scale;
  double variance_bonus_cap;
  /* C11 */
  int32_t graph;
  int32_t node_key;            /* T5D_NODE_KEY_* */
  double residual_epsilon;
  /* H2, E9 */
  int32_t solver;
  uint32_t solved_visit_cap;
  /* E5, E9 */
  int32_t final_rule;          /* T5D_FINAL_* */
  double lcb_floor;
  double lcb_z;
  double lcb_categorical_weight;
  double secure_child_a;
  /* leaves */
  uint32_t batch_size;
  double virtual_loss;
  int32_t synchronous;         /* H15 */
  uint64_t seed;
  t5d_expand_options expand;   /* the leaves' network input (k, features, relations) */
  /* terminal detection (H37) and the turn cap (I11) */
  uint64_t mate_nodes;
  uint64_t mate_bytes;
  int32_t turn_cap_turns;      /* 0: none */
  int32_t turn_cap_boards;     /* 0: none */
  uint64_t eval_cache_entries; /* 0: no cache */
  uint64_t eval_cache_bytes;   /* its memory ceiling (estimated) */
} t5d_search_params;

/* One leaf of a gathered batch. */
typedef struct t5d_leaf {
  uint64_t key;          /* the node key (H18; t5d_search_params.node_key) */
  uint64_t epoch;        /* the refresh epoch (t5d_search_set_epoch) */
  int32_t to_move;       /* the player whose action is in progress: 0 white, 1 black */
  int32_t turn_start;    /* 1: no move made yet in the action */
  t5d_expansion expansion; /* the network's input; logit i scores expansion.moves[i] */
  const uint8_t* legality; /* per move, T5D_LEGALITY_LEGAL or _UNKNOWN (the board-move
                              split's classification); NULL when none was made */
} t5d_leaf;

/* What the search has done, and its root. */
typedef struct t5d_search_summary {
  uint64_t playouts;
  uint64_t evaluations;   /* leaves evaluated (the cache's hits not counted) */
  uint64_t cache_hits;
  uint64_t nodes;         /* nodes held now */
  uint64_t nodes_created;
  uint64_t transposition_hits;
  uint64_t checkmates;
  uint64_t stalemates;
  uint64_t turn_caps;
  uint64_t unknown_verdicts;
  uint64_t unknown_edges;
  uint64_t verified_illegal;
  int32_t root_turn_start;
  int32_t root_terminal;  /* the root's game is over (checkmate, stalemate, the cap) */
  int32_t root_solved;
  int32_t root_lo;        /* the root's solver bounds, its player's view: -1, 0, 1 */
  int32_t root_hi;
  uint32_t root_visits;
  double root_q;
  uint64_t root_children;
  uint64_t epoch;
} t5d_search_summary;

/* One child of the root (t5d_search_child), from the root player's view. */
typedef struct t5d_root_child {
  uint32_t visits;     /* the policy target (f) */
  int32_t submit;      /* 1: "submit" (board-move split, Q4's other reading) */
  int32_t legality;    /* T5D_LEGALITY_LEGAL or _UNKNOWN */
  int32_t lo;          /* solver bounds (-1 when unproven, and through an unknown move) */
  int32_t hi;
  double prior;
  double q;
  size_t move_count;   /* its moves: 1 (board-move), an action's (whole-turn), 0 (submit) */
} t5d_root_child;

/* The defaults (t5d/mcts.hpp). Cannot fail; NULL is ignored. */
T5D_API void t5d_search_params_default(t5d_search_params* out);

/* A search from the game's current state. turn_start: 1 if no move has been
 * made yet in the action under way (the whole-turn split requires it). `params`
 * NULL: the defaults. The game handle is not kept. On failure *out is NULL. */
T5D_API t5d_status t5d_search_new(const t5d_game* game, int32_t turn_start,
                                  const t5d_search_params* params, t5d_search** out,
                                  char** error);

/* Releases a search. NULL is ignored. */
T5D_API void t5d_search_destroy(t5d_search* search);

/* ---- the step API --------------------------------------------------------- */

/* Selects up to max_leaves leaves (virtual loss on their paths), completing on
 * the way the simulations that need no evaluation, while fewer than
 * max_playouts simulations are completed or waiting (T5D_NO_LIMIT: none).
 * *count: the evaluations wanted (one per distinct state among the leaves); 0
 * when the search needs none (its root is terminal or solved, or the
 * playouts are spent). T5D_ERR_ARGUMENT if a batch is already waiting. */
T5D_API t5d_status t5d_search_gather(t5d_search* search, size_t max_leaves,
                                     uint64_t max_playouts, size_t* count, char** error);

/* Leaf `index` (< count) of the batch waiting. */
T5D_API t5d_status t5d_search_leaf(t5d_search* search, size_t index, t5d_leaf* out,
                                   char** error);

/* Leaf `index`'s state, as a t5d_state (fingerprint, boards). */
T5D_API t5d_status t5d_search_leaf_state(const t5d_search* search, size_t index,
                                         t5d_state** out, char** error);

/* The batch's evaluations, leaf by leaf in order: `logits` concatenates each
 * leaf's move_count logits (logit_count = their sum); `values` holds
 * T5D_VALUE_BINS numbers per leaf (value_count = count x 51), probabilities or
 * logits by value_kind. Backs up and releases the virtual loss. On
 * T5D_ERR_ARGUMENT (counts that do not match, a non-finite number, a value
 * that is no distribution) the batch is abandoned: virtual loss released,
 * nothing backed up. */
T5D_API t5d_status t5d_search_deliver(t5d_search* search, const float* logits,
                                      size_t logit_count, const float* values,
                                      size_t value_count, int32_t value_kind, char** error);

/* Releases the waiting batch's virtual loss without backing anything up. */
T5D_API t5d_status t5d_search_abandon(t5d_search* search);

/* Runs `playouts` simulations with a built-in evaluator (T5D_EVALUATOR_*;
 * evaluator_seed for the random one), for tests and tools. */
T5D_API t5d_status t5d_search_run(t5d_search* search, int32_t evaluator, uint64_t evaluator_seed,
                                  uint64_t playouts, char** error);

/* ---- results, choice, persistence --------------------------------------- */

T5D_API t5d_status t5d_search_info(const t5d_search* search, t5d_search_summary* out);

/* Root child `index` (< info.root_children): its statistics, and up to
 * `capacity` of its moves into `moves` (may be NULL with capacity 0). A move's
 * boards and cells are those of the root state's canonical order. */
T5D_API t5d_status t5d_search_child(const t5d_search* search, size_t index,
                                    t5d_root_child* out, t5d_move* moves, size_t capacity,
                                    char** error);

/* Final selection (E5, E9): the child to play and why (T5D_CHOICE_*). */
T5D_API t5d_status t5d_search_choose(const t5d_search* search, size_t* index, int32_t* reason,
                                     char** error);

/* Self-play's choice: sampled from the root's visits at `temperature` (<= 0:
 * the most visits), `random` a uniform number in [0, 1). */
T5D_API t5d_status t5d_search_choose_by_visits(const t5d_search* search, double temperature,
                                               double random, size_t* index, char** error);

/* Classifies root child `index`'s move exactly (a move must be known legal
 * before it is played): *legality T5D_LEGALITY_*; an illegal child is removed
 * (the children's indices then change). */
T5D_API t5d_status t5d_search_verify(t5d_search* search, size_t index, int32_t* legality,
                                     char** error);

/* Re-roots on root child `index` (C12): its statistics carried, every node the
 * new root cannot reach deleted, cached evaluations kept. */
T5D_API t5d_status t5d_search_advance(t5d_search* search, size_t index, char** error);

/* Re-roots on the turn start an action reaches - its moves, in a playable
 * order (each matched as t5d_game_apply matches one): the opponent's reply. */
T5D_API t5d_status t5d_search_advance_action(t5d_search* search, const t5d_move* moves,
                                             size_t count, char** error);

/* The root's state, as a new game handle (legal submissions). */
T5D_API t5d_status t5d_search_root_game(const t5d_search* search, t5d_game** out, char** error);

/* The refresh epoch (§7.9): evaluations are cached per (state, epoch). */
T5D_API t5d_status t5d_search_set_epoch(t5d_search* search, uint64_t epoch, char** error);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* T5D_CAPI_SEARCH_H */
