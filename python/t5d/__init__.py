"""t5d - the network stack of the Temporal 5D chess engine.

Untrained build (design spec section 15): every network component exists with random
weights at realistic initialisation scale, so that correctness (section 15.2) and cost
(section 15.3) can be verified before any training compute is spent. Nothing in this
package trains a network beyond the few-step smoke test in ``t5d.train.smoke``.
"""

__version__ = "0.0.1"
