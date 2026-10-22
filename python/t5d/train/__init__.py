"""Training-side code for the untrained build (register M21).

The build brief allows training code to be written and smoke-tested with at most a few
optimiser steps on synthetic data, and no training runs (spec section 15: "build everything;
train nothing - first"). What exists:

* ``losses`` - every head's loss over a ``Targets`` interface; ``smoke`` - three optimiser steps
  on synthetic targets; ``reply_map`` - D4's reply prototype's move space.
* ``records`` - register I15's training records: one versioned file per self-play game.
* ``selfplay`` - the generator: self-play through the engine's search, written as records.
* ``targets`` - every head's targets re-derived from a record: I2's pruned visits, D3's and D4's
  targets, I13's deblundering, I14's hook (refused until the sign-off), the batch's ``Targets``.
* ``sampler`` - I12's surprise-weighted sampling and 6u's arms (I6's triggers); ``window`` - I16's
  sublinear replay window; ``swa`` - I17's weight average, the network published to self-play.
"""
