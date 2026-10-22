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
 *     t5d_search_deliver_evaluations(s, &batch, ...)  their evaluations, in order
 *   t5d_search_choose / _child / _advance ...
 *
 * What a leaf carries is everything the network takes from it - the leaf's
 * expansion (t5d_expansion: per-board and per-square features, the candidate
 * moves, the move graph, typed relations, W(k)/E(k), and the turn state:
 * the mandatory and optional boards and the moves made in the leaf's action -
 * register A6), its node key and the refresh epoch. The search evaluates and
 * caches a leaf by its state AND its moves made in the action (no state
 * records the count; equal states with different counts are different
 * inputs). Its evaluation is:
 *
 *  - POLICY: one float32 logit per candidate move, logit i scoring
 *    leaf.expansion.moves[i] (the expansion's own order). Under the tree form a leaf
 *    is the root, a turn start or a mid-turn state where an interaction gate fired;
 *    the states between are expanded on inherited logits and never reach the caller.
 *    Under the whole-turn
 *    split (D2's other arm) the candidates are the turn start's pseudo-legal
 *    moves, and the search derives each legal action's prior from them itself.
 *  - SUBMIT (register D8; rulebook R8): turns have variable length, so under
 *    the board-move split submitting is an action of its own wherever it is
 *    possible (leaf.submit_possible: the obligations met, T5), legal only where
 *    the submission passes C4 (leaf.submit_legal, exact). Its logit - one
 *    float32 per leaf, read from the pooled global registers (D8) - is optional: NULL,
 *    or NaN for a leaf, gives the search's placeholder
 *    (t5d_search_params.submit_share). A leaf may be a state whose present has
 *    shifted inside the action (R8: moves may follow the shift).
 *  - THE VALUE HEADS (register D3, corrected at CHECKPOINT 7), each from the
 *    perspective of the leaf's ACTING PLAYER - the player whose action is in
 *    progress, until it submits (leaf.acting_player, = leaf.expansion.
 *    acting_player; after the shift it is not the colour to move on the
 *    present):
 *    - WDL (required): 3 float32 numbers (T5D_WDL: win, draw, loss) - the
 *      value search uses: the leaf's value is W - L, and its draw probability
 *      D is backed up beside it (t5d_root_child.d);
 *    - the CATEGORICAL over the root search value (optional): 51 float32
 *      numbers (T5D_VALUE_BINS; bins centred on -1 + 2k/50) - its variance is
 *      the default uncertainty signal (t5d_search_params.uncertainty_source),
 *      its mean is never backed up;
 *    - the ERROR head (optional): 1 float32 >= 0 - the predicted squared gap
 *      between the value and the short-term value; the other uncertainty signal.
 *      The signal uncertainty_source selects is REQUIRED of every leaf while E2,
 *      E3 or E5's term reads it (variance_scaled_cpuct, uncertainty_weighted_backup,
 *      variance_bonus, or final_rule T5D_FINAL_LCB with lcb_categorical_weight
 *      > 0): a delivery without it is refused (T5D_ERR_ARGUMENT, the batch
 *      abandoned). Read by nothing, a missing one is 0 and counted
 *      (t5d_search_summary.uncertainty_missing);
 *    - MOVES-LEFT (optional; register D4): 1 float32 >= 0, in the network's
 *      unit - recorded per node (t5d_root_child.network_moves_left), never
 *      backed up.
 *    The WDL and the categorical come as PROBABILITIES (T5D_VALUE_PROBABILITIES:
 *    non-negative, the search normalises them) or as LOGITS (T5D_VALUE_LOGITS:
 *    the search takes their softmax).
 *
 * t5d_search_layout reports this header's struct layouts, as t5d_layout does
 * t5d/capi.h's, for a binding that mirrors them.
 *
 * ERRORS: T5D_ERR_ARGUMENT for the caller's error - an argument, or a call the
 * search's state does not allow (gather while a batch waits, deliver with none
 * waiting, max_leaves 0, set_epoch under a waiting batch); T5D_ERR_MEMORY when
 * an allocation fails (the search stays consistent: no batch left half-made);
 * T5D_ERR_UNSUPPORTED for a feature not built; T5D_ERR_INTERNAL for the
 * search's own failure (a broken invariant, a contradiction) - a bug to report.
 *
 * SOLVER BOUNDS (lo, hi) are search state: they may guide play, but they are
 * never to be stored as proofs - no proof enters a persistent solved store
 * until the generator has passed the gate (register H19, H23). The in-tree
 * solver itself is part of the proof channel (H41) and off by rule
 * (t5d_search_params.proof_channel).
 *
 * The pointers a leaf hands out are valid until the next t5d_search_gather,
 * t5d_search_deliver_evaluations, t5d_search_abandon or t5d_search_destroy on the handle:
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
/* D3: the WDL head's outputs per leaf - win, draw, loss. */
#define T5D_WDL 3

/* D2 (undecided): the board-move split, whole turns, or the tree form (the board-move
 * split's tree with turn-start priors inherited inside a turn; re-evaluation only where an
 * interaction gate fires and at complete turns - t5d/mcts.hpp). */
#define T5D_SPLIT_BOARD_MOVE 0
#define T5D_SPLIT_WHOLE_TURN 1
#define T5D_SPLIT_TREE 2
/* The tree form's interaction gates (t5d_search_params.tree_gates, a mask; spec 8.2): a move
 * between boards still to move, a capture of a piece that could reach them, a change in
 * check, a shift of the present - and ALWAYS, a test seam (every mid-turn node evaluated:
 * spec 15.2's "every interaction gate firing", the board-move split's search exactly). */
#define T5D_TREE_GATE_BETWEEN 1
#define T5D_TREE_GATE_CAPTURE 2
#define T5D_TREE_GATE_CHECK 4
#define T5D_TREE_GATE_PRESENT 8
#define T5D_TREE_GATES_INTERACTION 15
#define T5D_TREE_GATE_ALWAYS 16
/* The tree form's readings (placeholders; t5d/mcts.hpp): "the boards still to move" - every
 * playable board of the mover's not yet played, or the mandatory ones left; "a piece that could
 * reach them" - one with a move onto one of them (passed, as R1 passes boards), or any capture;
 * "a change in check" - the acting player's royals capturable with the boards still to move
 * passed (R1's construction), or by C4's submission test as the state stands. */
#define T5D_STILL_TO_MOVE_PLAYABLE 0
#define T5D_STILL_TO_MOVE_MANDATORY 1
#define T5D_CAPTURE_REACH 0
#define T5D_CAPTURE_ANY 1
#define T5D_CHECK_PASSED 0
#define T5D_CHECK_SUBMISSION 1
/* ... and "a shift of the present" - any move of it (its advance as the obligations are met,
 * T5, on every action; a rewind) or a rewind only. */
#define T5D_PRESENT_ANY 0
#define T5D_PRESENT_REWIND 1
/* The whole-turn split's action prior from its moves' logits: their sum or mean. */
#define T5D_ACTION_PRIOR_SUM 0
#define T5D_ACTION_PRIOR_MEAN 1
/* The MCGS node key (spec issue 30): the 5D Zobrist key, or with branch origins. */
#define T5D_NODE_KEY_STATE 0
#define T5D_NODE_KEY_STATE_AND_ORIGINS 1
/* The final-selection rule (E5). */
#define T5D_FINAL_VISITS 0
#define T5D_FINAL_LCB 1
/* t5d_evaluations: what `wdl` and `values` hold. */
#define T5D_VALUE_PROBABILITIES 0
#define T5D_VALUE_LOGITS 1
/* The uncertainty signal search consumes (register I10; ablation 6m). */
#define T5D_UNCERTAINTY_CATEGORICAL 0 /* the categorical's variance (the default) */
#define T5D_UNCERTAINTY_ERROR_HEAD 1  /* the error head's output */
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

/* Why a run stopped (t5d/mcts.hpp, StopReason). */
#define T5D_STOP_PLAYOUTS 0
#define T5D_STOP_EVALUATIONS 1
#define T5D_STOP_FINISHED 2   /* the root is terminal or solved */
#define T5D_STOP_EXHAUSTED 3  /* no simulation could be made */
#define T5D_STOP_NODES 4      /* the node budget */
#define T5D_STOP_MEMORY 5     /* the memory budget */
#define T5D_STOP_DEADLINE 6   /* the deadline */
/* No limit, for the budgets below. */
#define T5D_NO_LIMIT UINT64_MAX

/* Layouts: each struct below is appended to, never reordered (t5d_layout's
 * contract; t5d_search_layout reports every offset). A struct of t5d/capi.h's
 * embedded in one of them - provisional, and growing as t5d/capi.h does - is
 * its LAST member, so that its growth moves none of the others: this file's
 * own new fields go in front of it (moving it; a binding reads its offset from
 * t5d_search_layout). */

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
  double submit_share;         /* board-move: submit's prior share when the evaluator gives no
                                  submit logit (D8; placeholder 0.5) */
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
  double dirichlet_alpha;       /* per edge - read only when dirichlet_concentration is 0
                                  (the arm before CHECKPOINT 7); unused by default */
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
  /* H2, E9, H41: the proof channel's switch - solved-status propagation in the
     tree; 0 by rule until the gate passes and the channel is signed off */
  int32_t proof_channel;
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
  /* terminal detection (H37) and the turn cap (I11) */
  uint64_t mate_nodes;
  uint64_t mate_bytes;
  int32_t turn_cap_turns;      /* 0: none */
  int32_t turn_cap_boards;     /* 0: none */
  uint64_t eval_cache_entries; /* 0: no cache */
  uint64_t eval_cache_bytes;   /* its memory ceiling (estimated) */
  /* engine ABI round 2 (register D3, I10) */
  int32_t uncertainty_source;  /* T5D_UNCERTAINTY_*: a node's variance for E2, E3 and E5's
                                  optional term (6m) */
  /* engine ABI round 2b (register E1, I19): the root noise's total concentration and its
     shaping, the root policy temperature - placeholders */
  double dirichlet_concentration; /* split across the root's open edges (default 0.03 x 361);
                                     0: dirichlet_alpha per edge */
  int32_t dirichlet_shaping;      /* KataGo-style shaping: a prototype, 0 (off) */
  double dirichlet_shaping_share; /* the shaped share of the total (0.5) */
  double dirichlet_shaping_cap;   /* the shaping's prior cap (0.01) */
  double root_policy_temperature; /* the root's softmax temperature, before the noise (1: off) */
  /* D2's tree form (split T5D_SPLIT_TREE; read by no other split; refused with graph - spec
     issue 301): its interaction gates and their readings, placeholders */
  uint32_t tree_gates;            /* T5D_TREE_GATE_* mask (T5D_TREE_GATES_INTERACTION); 0: the
                                     network only at turn starts and the root */
  int32_t tree_still_to_move;     /* T5D_STILL_TO_MOVE_* */
  int32_t tree_capture;           /* T5D_CAPTURE_* */
  int32_t tree_check;             /* T5D_CHECK_* */
  int32_t tree_inherit_value;     /* 1: an inherited node's value before any backup (its FPU
                                     base) and uncertainty signal are its parent's evaluation's */
  int32_t tree_present;           /* T5D_PRESENT_* */
  t5d_expand_options expand;   /* the leaves' network input (k, features, relations); last */
} t5d_search_params;

/* One leaf of a gathered batch. Its evaluation's identity - what the search
 * caches and de-duplicates it by - is (key, expansion.moves_made, epoch), the
 * state compared exactly behind the key: the key alone does not tell equal
 * states with different counts apart (A6), and neither does t5d_search_leaf_state. */
typedef struct t5d_leaf {
  uint64_t key;            /* the state's node key (H18; t5d_search_params.node_key) - not
                              the leaf's identity, which adds expansion.moves_made (A6) */
  uint64_t epoch;          /* the refresh epoch (t5d_search_set_epoch) */
  int32_t acting_player;   /* D8: whose action is in progress, until it submits: 0 white,
                              1 black - the value's perspective */
  int32_t submit_possible; /* D8: submitting is an action here (the obligations met, T5) -
                              board-move split and tree form; 0 under whole turns */
  int32_t submit_legal;    /* ... and legal (C4, exact) */
  const uint8_t* legality; /* per move, T5D_LEGALITY_LEGAL or _UNKNOWN (the board-move
                              split's classification); NULL when none was made */
  t5d_expansion expansion; /* the network's input; logit i scores expansion.moves[i]; last */
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
  int32_t root_lo;        /* the root's solver bounds, its player's view: -1, 0, 1 - search
                             state, never to be persisted as a proof (H19, H23) */
  int32_t root_hi;
  uint32_t root_visits;
  double root_q;          /* the root's backed-up mean value, its player's view (0 before a
                             visit) - a solved root's too, not its proven value: root_lo and
                             root_hi carry what is proven */
  uint64_t root_children;
  uint64_t epoch;
  uint64_t memory_bytes;  /* the graph's memory, estimated (the evaluation cache apart) */
  uint64_t cache_entries; /* the evaluation cache's entries (shared ones included) */
  uint64_t cache_bytes;   /* ... and their bytes, as it counts them */
  uint64_t cache_purged;  /* entries behind the root this search purged */
  int32_t last_stop;      /* T5D_STOP_*: why the last run stopped */
  /* engine ABI round 2 (register D3): the root's backed-up draw probability
     (root_q's companion) and its own evaluation - the network's raw outputs at
     the root (I15), its player's view; NaN: not given (or not evaluated) */
  int32_t root_evaluated;    /* 1: the root has an evaluation */
  double root_d;             /* backed-up average, as root_q (a solved root's too) */
  double root_wdl[T5D_WDL];  /* the root's network WDL, normalised */
  double root_moves_left;
  double root_error;
  /* engine ABI round 2c (register I15, I2, D4): the root's identity, its PUCT's effective
     cPUCT now (cpuct x E2's root scale; for a replica of the root's selection - I2's
     target-side pruning) and the effort label: the root's visits when its most visited
     child last changed (§8.6; counted from the re-root for a carried root) */
  uint64_t root_key;      /* the root's node key: with the moves made in its action and the
                             epoch, its evaluation's identity */
  double root_cpuct;
  uint32_t root_best_since;
  /* engine ABI round 2 review fixes (register I10, 6m): nodes expanded from an evaluation
     without the uncertainty signal uncertainty_source selects - only while nothing reads it
     (with E2, E3 or E5's term on such a delivery is refused) */
  uint64_t uncertainty_missing;
  /* D2's tree form: mid-turn nodes expanded on inherited priors (no evaluation), and the
     mid-turn nodes sent to the network (or the cache) instead, by the first gate found to fire
     (tested in the order always, present, between, capture, check) */
  uint64_t inherited_nodes;
  uint64_t gate_between;
  uint64_t gate_capture;
  uint64_t gate_check;
  uint64_t gate_present;
  uint64_t gate_always;
} t5d_search_summary;

/* A run's budgets (t5d/mcts.hpp, Limits): T5D_NO_LIMIT for none (seconds: 0). playouts and
 * evaluations count from the call; nodes and bytes are the graph's. A run
 * stops between batches once one is reached. */
typedef struct t5d_search_limits {
  uint64_t playouts;
  uint64_t evaluations;
  uint64_t nodes;
  uint64_t bytes;   /* t5d_search_summary.memory_bytes */
  double seconds;   /* a wall-clock deadline from the call; 0: none (not deterministic) */
} t5d_search_limits;

/* One child of the root (t5d_search_child), from the root player's view. */
typedef struct t5d_root_child {
  uint32_t visits;     /* the policy target (f) */
  int32_t submit;      /* 1: "submit" (board-move split, Q4's other reading) */
  int32_t legality;    /* T5D_LEGALITY_LEGAL or _UNKNOWN */
  int32_t lo;          /* solver bounds (-1 when unproven, and through an unknown move): search
                          state, never to be persisted as a proof (H19, H23) */
  int32_t hi;
  double prior;
  double q;            /* the backed-up value, W - L (D3) */
  size_t move_count;   /* its moves: 1 (board-move), an action's (whole-turn), 0 (submit) */
  /* engine ABI round 2 (register D3): the search's draw probability, and the
     child's own evaluation - the network's (I13's and I15's memo) */
  double d;            /* backed up beside q: the child's search WDL is
                          ((1 - d + q) / 2, d, (1 - d - q) / 2) */
  int32_t evaluated;   /* 1: the child node has an evaluation (the network_* fields) */
  double network_wdl[T5D_WDL];  /* its WDL, normalised, the root player's view */
  double network_moves_left;    /* NaN: none given (or not evaluated) */
  double network_error;         /* NaN: none given (or not evaluated) */
  /* engine ABI round 2c (register I15, I2): the root's side table - what the records and
     the target-side pruning need */
  uint64_t child_key;           /* the child node's key; 0: no child node yet */
  double prior_noise_free;      /* the prior before the root noise (after the temperatures,
                                   the unknown-move scale, the submit share) */
  double value_variance;        /* of the values backed up through it (0 before a visit) */
  double boost;                 /* E10's asset-map boost: 0 (not wired) */
  double exploration_scale;     /* E3's factor on its exploration term (1: off) */
  uint32_t forced_visits;       /* forced playouts: 0 (not built - pruning is target-side) */
  uint32_t steered_visits;      /* visits a steering source decided: 0 (none wired, E10) */
} t5d_root_child;

/* One batch's evaluations (t5d_search_deliver_evaluations), leaf by leaf in the
 * batch's order; a NULL array with count 0 is "none given" where the field is
 * optional. Appended to, never reordered (t5d_search_layout reports it). */
typedef struct t5d_evaluations {
  const float* logits;          /* each leaf's move_count logits, concatenated */
  size_t logit_count;           /* = the leaves' move counts summed */
  const float* submit_logits;   /* one per leaf (read where its submit is possible; NaN:
                                   none given), or NULL: the placeholder share everywhere */
  size_t submit_count;          /* the leaf count, or 0 */
  const float* wdl;             /* T5D_WDL per leaf (required) */
  size_t wdl_count;             /* = leaves x 3 */
  int32_t wdl_kind;             /* T5D_VALUE_PROBABILITIES or _LOGITS */
  int32_t value_kind;           /* ... for `values` - checked even when `values` is NULL */
  const float* values;          /* T5D_VALUE_BINS per leaf (the categorical), or NULL */
  size_t value_count;           /* = leaves x 51, or 0 */
  const float* errors;          /* one per leaf (>= 0; NaN: none given), or NULL */
  size_t error_count;           /* the leaf count, or 0 */
  const float* moves_left;      /* one per leaf (>= 0; NaN: none given), or NULL */
  size_t moves_left_count;      /* the leaf count, or 0 */
  /* Reserved for D2's flat form (per-board logits; not built): NULL and 0 -
     anything else is T5D_ERR_UNSUPPORTED. Its per-leaf layout is fixed when
     the flat form is built. */
  const float* board_logits;
  size_t board_logit_count;
} t5d_evaluations;

/* The defaults (t5d/mcts.hpp). Cannot fail; NULL is ignored. */
T5D_API void t5d_search_params_default(t5d_search_params* out);

/* A search from the game's current state, with the game handle's count of moves
 * made in the action under way (t5d_game_summary.moves_made; register A6).
 * turn_start: 1 if no move has been made yet in that action (the whole-turn
 * split requires it) - it must agree with the handle's count (1 exactly where
 * it is 0; T5D_ERR_ARGUMENT otherwise). `params` NULL: the defaults. The game
 * handle is not kept. On failure *out is NULL. */
T5D_API t5d_status t5d_search_new(const t5d_game* game, int32_t turn_start,
                                  const t5d_search_params* params, t5d_search** out,
                                  char** error);

/* A search like t5d_search_new's that shares `donor`'s evaluation cache (C14:
 * a fresh search per move, the cache carried - the step API's self-play), its
 * own parameters' cache sizes ignored. The donor may be destroyed afterwards;
 * the cache lives while a search holds it. The cache has no lock: the searches
 * sharing it must never be used from two threads at once (the caller's duty;
 * nothing checks it). */
T5D_API t5d_status t5d_search_new_sharing(const t5d_search* donor, const t5d_game* game,
                                          int32_t turn_start, const t5d_search_params* params,
                                          t5d_search** out, char** error);

/* Releases a search. NULL is ignored. */
T5D_API void t5d_search_destroy(t5d_search* search);

/* ---- the step API --------------------------------------------------------- */

/* Selects up to max_leaves leaves (virtual loss on their paths), completing on
 * the way the simulations that need no evaluation, while fewer than
 * max_playouts simulations are completed or waiting (T5D_NO_LIMIT: none).
 * *count: the evaluations wanted (one per distinct state and moves made in its
 * action among the leaves - one state with two counts is asked twice, A6); 0
 * when the search needs none (its root is terminal or solved, or the
 * playouts are spent). T5D_ERR_ARGUMENT if a batch is already waiting. */
T5D_API t5d_status t5d_search_gather(t5d_search* search, size_t max_leaves,
                                     uint64_t max_playouts, size_t* count, char** error);

/* Leaf `index` (< count) of the batch waiting. */
T5D_API t5d_status t5d_search_leaf(t5d_search* search, size_t index, t5d_leaf* out,
                                   char** error);

/* Leaf `index`'s state, as a t5d_state (fingerprint, boards). A state holds no
 * count: the leaf's moves made in its action are t5d_leaf.expansion.moves_made
 * (A6), so the fingerprint alone does not identify the leaf's input. */
T5D_API t5d_status t5d_search_leaf_state(const t5d_search* search, size_t index,
                                         t5d_state** out, char** error);

/* The batch's evaluations (t5d_evaluations, above): expands the leaves, backs up
 * W - L and D, releases the virtual loss. On T5D_ERR_ARGUMENT (counts that do
 * not match, a non-finite logit, an infinite submit logit, a WDL or categorical
 * that is no distribution, a negative or infinite error or moves-left, an
 * unknown kind) or T5D_ERR_UNSUPPORTED (per-board logits) the batch is
 * abandoned: virtual loss released, nothing backed up. (It replaced
 * t5d_search_deliver, which took the categorical alone and backed up its mean -
 * the reading D3 corrected at CHECKPOINT 7.) */
T5D_API t5d_status t5d_search_deliver_evaluations(t5d_search* search, const t5d_evaluations* batch,
                                                  char** error);

/* Releases the waiting batch's virtual loss without backing anything up. */
T5D_API t5d_status t5d_search_abandon(t5d_search* search);

/* Runs `playouts` simulations with a built-in evaluator (T5D_EVALUATOR_*;
 * evaluator_seed for the random one), for tests and tools. */
T5D_API t5d_status t5d_search_run(t5d_search* search, int32_t evaluator, uint64_t evaluator_seed,
                                  uint64_t playouts, char** error);

/* The same within `limits` (NULL: an error); *stop (may be NULL): T5D_STOP_*. */
T5D_API t5d_status t5d_search_run_limits(t5d_search* search, int32_t evaluator,
                                         uint64_t evaluator_seed, const t5d_search_limits* limits,
                                         int32_t* stop, char** error);

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
 * order (each matched as t5d_game_apply matches one): the opponent's reply.
 * Atomic: the whole action is checked first, and on any error nothing has
 * changed. From a turn start the action has at least one move (count 0 is an
 * error there); from a state during an action, count 0 submits it. */
T5D_API t5d_status t5d_search_advance_action(t5d_search* search, const t5d_move* moves,
                                             size_t count, char** error);

/* The root's state, as a new game handle (legal submissions), with the root's
 * count of moves made in its action (A6). */
T5D_API t5d_status t5d_search_root_game(const t5d_search* search, t5d_game** out, char** error);

/* The refresh epoch (§7.9): evaluations are cached per (state, moves made in its
 * action, epoch) (A6). An error while a batch is waiting (deliver or abandon it
 * first). */
T5D_API t5d_status t5d_search_set_epoch(t5d_search* search, uint64_t epoch, char** error);

/* H15: a digest of the whole search graph - every node's and edge's statistics
 * in a canonical order - equal for bit-identical searches (the synchronous
 * debug mode's check, and the step API's against the C++ search). */
T5D_API t5d_status t5d_search_digest(const t5d_search* search, uint64_t* out);

/* The layouts of this header's structs, as this library was built (t5d_layout's
 * contract, for "t5d_search_params", "t5d_leaf", "t5d_search_summary",
 * "t5d_search_limits", "t5d_root_child" and "t5d_evaluations"): with `field`
 * NULL, sizeof(`type`); otherwise the offset of `field`. T5D_ERR_ARGUMENT for a
 * NULL `type` or `out`, an unknown type or field. */
T5D_API t5d_status t5d_search_layout(const char* type, const char* field, size_t* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* T5D_CAPI_SEARCH_H */
