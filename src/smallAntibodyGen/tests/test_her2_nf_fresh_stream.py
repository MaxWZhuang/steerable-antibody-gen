"""The opt-in fresh-preservation-stream path.

The intervention it exists for: a tail/FKL arm normally replays a FIXED bank,
so one bank row is consumed several times over a long run. Setting
``replay_rows == batch_rows * horizon`` gives every preservation-row exposure
its own parent draw, consumed exactly once. Nothing else about the arm moves.

Default-off discipline (`change-control` Class K): with ``reuse_saved_arrays``
left at its default, the saved task stream and the saved replay bank are still
preferred exactly as before, so every existing run is byte-identical.
"""
from __future__ import annotations

import numpy as np

from smallAntibodyGen.experiments import her2_replay_streams as streams_lib


def test_a_bank_as_large_as_the_exposures_is_consumed_without_reuse():
    """The fresh-stream arm: one draw per exposure, each used exactly once."""
    exposures = 64 * 120
    order = streams_lib.replay_order(bank_rows=exposures, exposures=exposures,
                                     seed=777000301, parent_seed=20260918)
    assert order.size == exposures
    assert np.unique(order).size == exposures, "a fresh stream reuses no bank row"
    assert sorted(order.tolist()) == list(range(exposures))


def test_the_fixed_bank_arm_reuses_rows_and_that_is_the_contrast():
    """The control: a 100k bank over the same exposures repeats rows by design."""
    exposures = 64 * 3750
    order = streams_lib.replay_order(bank_rows=100_000, exposures=exposures,
                                     seed=777000301, parent_seed=20260918)
    assert order.size == exposures
    distinct = np.unique(order).size
    assert distinct == 100_000, "every row of the fixed bank is seen"
    # 240,000 exposures over 100,000 rows: 2.4 passes, so most rows repeat.
    assert exposures / distinct == 2.4


def test_the_two_arms_differ_only_in_reuse_not_in_exposure_count():
    """Matched total preservation-row exposures is what makes the pair a pair."""
    exposures = 64 * 3750
    fixed = streams_lib.replay_order(bank_rows=100_000, exposures=exposures,
                                     seed=777000301, parent_seed=20260918)
    fresh = streams_lib.replay_order(bank_rows=exposures, exposures=exposures,
                                     seed=777000301, parent_seed=20260918)
    assert fixed.size == fresh.size == exposures


def test_replay_order_is_unchanged_for_every_historical_call():
    """Pin: the fresh-stream work added no new argument and moved no default.

    ``replay_order`` is the function both arms go through. A change in its
    output for the inherited arguments would silently invalidate every reused
    historical trajectory, so the digest is pinned here rather than trusted.
    """
    order = streams_lib.replay_order(bank_rows=100_000, exposures=64 * 1000,
                                     seed=777000301, parent_seed=20260918)
    again = streams_lib.replay_order(bank_rows=100_000, exposures=64 * 1000,
                                     seed=777000301, parent_seed=20260918)
    assert np.array_equal(order, again), "the order must be a pure function of its arguments"
    # Independent reconstruction of the documented rule: cycle 0 is one shuffled
    # full bank, truncated to the exposures asked for.
    expected = np.random.default_rng([777000301, 20260918, 0]).permutation(100_000)[:64 * 1000]
    assert np.array_equal(order, expected)


def test_reuse_saved_arrays_defaults_to_preferring_the_saved_arrays():
    """The knob's default must leave the saved-first behaviour exactly in place."""
    import inspect

    from smallAntibodyGen.experiments import her2_nf_services as services

    for name in ("run_one_trajectory", "_streams_for", "_replay_bank",
                 "_resolve_replay_bank"):
        signature = inspect.signature(getattr(services.FlightServices, name))
        parameter = signature.parameters.get("reuse_saved_arrays")
        assert parameter is not None, f"{name} must accept reuse_saved_arrays"
        assert parameter.default is True, (
            f"{name}'s reuse_saved_arrays default must be True so every existing "
            "run keeps loading the saved task stream and the saved replay bank")
