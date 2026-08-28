"""
Mechanisms that exist to make a specific EXPERIMENT valid, not to serve training.

Kept out of `models/` and `data/` on purpose: nothing here is on the training or
generation path, and an experiment's fairness machinery should not be able to
change what a production run computes.

Current contents support J24, the antigen-encoder comparison:

- :mod:`antigen_residues` -- guarantee both arms see the SAME antigen residues
  even though their tokenizers spend different numbers of special tokens.
- :mod:`init_parity` -- make the shared projection/fusion/head parameters
  bit-identical between arms at step zero, which construction order otherwise
  prevents.
- :mod:`antigen_cache` -- cache frozen ESM antigen encodings, keyed so a stale
  cache cannot survive a change that would invalidate it.
"""
