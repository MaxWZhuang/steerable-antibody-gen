"""Multiscale functional adaptation pilot: whole-mask draws, measured endpoint pairs, one runner.

This is *mixed adaptation* of an existing masked protein language model. It is not a new
diffusion sampler, it changes nothing about how the pretrained model decodes, and nothing here is
evidence that the adaptation improves biology: the pilot exists to find out whether it does.

Core (pure enough to test without a model checkpoint):

* :mod:`.masks` -- how many sites of a declared common region to hide (all of them with a stated
  probability, otherwise ``k`` uniform in ``1..n-1``) and which ones (the suffix, or a uniformly
  sampled subset of the same size). Labels never enter the choice.
* :mod:`.pairs` -- measured endpoint pairs: two training populations of one assay and one declared
  region, at exact Hamming distance 1, 2 or 3, each with an exact finite pKD. A bounded,
  order-independent hash reservoir keeps storage small; it does not equalise how often each
  population is an endpoint, and the pool reports the degree, coverage and exposure it has.
* :mod:`.objective` -- the pair loss: the model's sequentially revealed log probability of the
  changed sites, differenced between the endpoints, fitted to ``kappa`` times the measured pKD
  difference, optionally with the retained reconstruction term on the same endpoints.

Wiring (needs the services, so torch):

* :mod:`.draws` -- how an arm changes a base draw: keyed mask replacement on whole-window draws,
  measured pairs in place of pair-only HER2 local draws.
* :mod:`.evaluation` -- labelled before/after diagnostics, each block kept apart.
* :mod:`.runner` -- the independent fresh-run loop, the evaluation-only snapshot and the records.

Inputs (``config``, ``roots`` and ``membership`` import without torch; ``inputs`` does not):

* :mod:`.config` -- the closed pilot schema bound to the base routes config by digest.
* :mod:`.roots` -- the five roots, the context the services read through, the exclusive run root.
* :mod:`.membership` -- the frozen split restored and re-checked from its record.
* :mod:`.inputs` -- the narrow services subclass, source identities and post-load validation.

The command line is ``scripts/her2_multiscale_pilot.py``.
"""
