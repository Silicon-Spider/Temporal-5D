"""Training-side code for the untrained build (register M21).

The build brief allows training code to be written and smoke-tested with at most a few
optimiser steps on synthetic data, and no training runs (spec section 15: "build everything;
train nothing - first"). What exists:

* ``losses`` - every head's loss over a ``Targets`` interface; ``smoke`` - three optimiser steps
  on synthetic targets; ``reply_map`` - D4's reply prototype's move space.
* ``records`` - register I15's training records: one versioned file per self-play game.
* ``selfplay`` - the generator: self-play through the engine's search, written as records.
* ``selfplay_pool`` - register G12's generator run: the same games played concurrently behind one
  evaluator (``engine/pool.py``'s token-budget batches), their records unchanged, the run's batching
  conditions in its ``pool.json``; ``python -m t5d.train.selfplay_pool``.
* ``targets`` - every head's targets re-derived from a record: I2's pruned visits, D3's and D4's
  targets, I13's deblundering, I14's hook (refused until the sign-off), the batch's ``Targets``.
* ``completed_q`` - register I21's prototype policy target (Gumbel AlphaZero's completed-Q),
  re-derived from the same records; an arm for the shared-dataset ranking (M24) - the builder
  writes it beside the pruned-visit target behind ``TargetConfig.completed_q`` (off), and
  ``losses.completed_q_arm`` trains the main policy head on it (weight 0 by default).
* ``target_arms`` - M24's target arms for I21: the arms' configurations, the record-level
  comparison of the two policy targets, and the held-out yardstick (forward only); the ranking
  itself - per-arm trainings on a shared dataset - is not built: it waits for the pilot run's data.
* ``sampler`` - I12's surprise-weighted sampling and 6u's arms (I6's triggers); ``window`` - I16's
  sublinear replay window; ``swa`` - I17's weight average, the network published to self-play.
* ``recipe`` - register I23's recipe for the pilot run: batches assembled to a token budget within a
  board-count range, a learning-rate warm-up and schedule; the pilot's preset (``PILOT_RECIPE``,
  placeholders). Off by default: the trainer's default is the smoke test's recipe.
* ``seeded`` - register I22's stored positions for seeded starts (a 5dpgn record and the number of
  its actions replayed, set up by the engine's reader); the generator takes them behind
  ``selfplay.SeededStartConfig`` and the sampler holds I22's board-count floors behind
  ``SamplerConfig.range_floors`` - both provisional, both off.
* ``trainer`` - the loop: records -> sampler -> window -> targets -> losses -> at most three
  optimiser steps -> SWA; G8's QAT hook (``apply_qat``: G2's FP8 trunk, ``net/quant.py``) behind
  its flag, off.
"""
