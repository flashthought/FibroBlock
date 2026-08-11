"""Reproducibility tests: the same configuration must give the same numbers.

These support the report's central practical claim -- that an examiner cloning
the repository and running ``python scripts/make_all_figures.py`` obtains the
figures that are committed. Determinism is checked **bitwise**, not to a
tolerance: a model with no random input and a fixed sequence of floating-point
operations should reproduce exactly, and anything less would indicate hidden
state.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from fibroblock import config as cfg
from fibroblock import measure, simulate, utils


def short_config(**overrides) -> cfg.RunConfig:
    """Build a small, fast configuration for repeated runs."""
    base = cfg.default_config().replace(
        gap=cfg.GapParams(rho=0.3, gap_length_cm=0.1),
        solver=cfg.SolverParams(dt_ms=0.02, t_end_ms=60.0, record_every=25),
    )
    return base.replace(**overrides) if overrides else base


# ---------------------------------------------------------------------------
# Bitwise determinism
# ---------------------------------------------------------------------------


def test_same_config_gives_bitwise_identical_output() -> None:
    """Supports the claim that committed figures can be regenerated exactly.

    Bitwise, not approximate. Any difference at all would mean the result
    depends on something outside the configuration.
    """
    config = short_config()

    first = simulate.run_simulation(config)
    second = simulate.run_simulation(config)

    assert np.array_equal(first.V_snapshots, second.V_snapshots)
    assert np.array_equal(first.w_snapshots, second.w_snapshots)
    assert np.array_equal(first.snapshot_times, second.snapshot_times)
    assert np.array_equal(first.V_peak, second.V_peak)
    assert np.array_equal(first.charge_history, second.charge_history)
    # NaN != NaN, so activation maps need equal_nan.
    assert np.array_equal(
        first.activation_time_crossing,
        second.activation_time_crossing,
        equal_nan=True,
    )
    assert np.array_equal(
        first.activation_time_max_dvdt,
        second.activation_time_max_dvdt,
        equal_nan=True,
    )


def test_derived_measurements_are_bitwise_reproducible() -> None:
    """Supports every reported number, not just the raw arrays."""
    config = short_config()

    first = measure.measure_velocity(simulate.run_simulation(config))
    second = measure.measure_velocity(simulate.run_simulation(config))

    assert first.theta_cm_per_ms == second.theta_cm_per_ms
    assert first.r_squared == second.r_squared
    assert first.n_points == second.n_points


def test_block_verdict_is_reproducible() -> None:
    """Supports the phase diagram, which is built entirely from these verdicts."""
    config = short_config()

    first = measure.detect_block(simulate.run_simulation(config))
    second = measure.detect_block(simulate.run_simulation(config))

    assert first.blocked == second.blocked
    assert first.n_activated_beyond == second.n_activated_beyond
    assert first.first_arrival_ms == second.first_arrival_ms


def test_a_different_config_gives_a_different_answer() -> None:
    """A negative control for the determinism tests above.

    If every run returned the same numbers regardless of configuration, the
    determinism tests would pass for the wrong reason.
    """
    weak = simulate.run_simulation(short_config(gap=cfg.GapParams(rho=0.2)))
    strong = simulate.run_simulation(short_config(gap=cfg.GapParams(rho=0.9)))

    assert not np.array_equal(weak.V_snapshots, strong.V_snapshots)


def test_run_order_does_not_affect_results() -> None:
    """Supports running experiments individually or through the pipeline.

    No module may accumulate state between runs. Interleaving two different
    configurations must give the same answers as running each alone.
    """
    config_a = short_config(gap=cfg.GapParams(rho=0.25))
    config_b = short_config(gap=cfg.GapParams(rho=0.75))

    alone_a = simulate.run_simulation(config_a).V_snapshots
    alone_b = simulate.run_simulation(config_b).V_snapshots

    # Now interleave, with a third run in between to disturb any hidden state.
    interleaved_a = simulate.run_simulation(config_a).V_snapshots
    simulate.run_simulation(short_config(gap=cfg.GapParams(rho=0.5)))
    interleaved_b = simulate.run_simulation(config_b).V_snapshots

    assert np.array_equal(alone_a, interleaved_a)
    assert np.array_equal(alone_b, interleaved_b)


def test_frozen_config_cannot_be_mutated() -> None:
    """Supports the claim that a saved config is the config that ran.

    Both the top-level RunConfig and its nested sections must be frozen; a
    mutable inner section would let a sweep alter a configuration after it had
    been recorded.
    """
    config = cfg.default_config()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.fhn.a = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_record_is_json_serialisable() -> None:
    """Supports the requirement that every result carries a provenance stamp.

    If the record could not be serialised, the pipeline would fail at the very
    end, after all the expensive computation.
    """
    record = utils.provenance(cfg.default_config().to_dict())

    text = json.dumps(record)
    restored = json.loads(text)

    assert restored["config"]["fhn"]["a"] == 0.7
    assert "git_commit" in restored
    assert "generated_utc" in restored
    for library in ("numpy", "scipy", "matplotlib", "python"):
        assert library in restored["libraries"]


def test_git_commit_never_raises() -> None:
    """Supports running from a downloaded ZIP rather than a clone.

    A missing git installation must degrade the provenance, not break the run.
    """
    commit = utils.git_commit_hash()
    assert isinstance(commit, str)
    assert commit  # non-empty either way


def test_numpy_conversion_handles_the_types_measurements_actually_contain() -> None:
    """Supports saving measurement dictionaries without per-caller casting."""
    record = {
        "float64": np.float64(1.5),
        "int64": np.int64(3),
        "bool": np.bool_(True),
        "array": np.arange(3),
        "nested": {"values": [np.float64(0.5), np.float64(1.5)]},
        "nan": float("nan"),
    }

    converted = utils._jsonable(record)
    # NaN is preserved rather than dropped: a NaN activation time means the
    # wave never arrived, which is data.
    text = json.dumps(converted)

    assert isinstance(converted["float64"], float)
    assert isinstance(converted["int64"], int)
    assert isinstance(converted["bool"], bool)
    assert converted["array"] == [0, 1, 2]
    assert "NaN" in text


def test_seed_is_fixed_and_returns_a_generator() -> None:
    """Supports the brief's requirement that seeds are fixed and reported."""
    generator = utils.set_seed(cfg.default_config().seed, announce=False)
    first = generator.random(5)

    generator_again = utils.set_seed(cfg.default_config().seed, announce=False)
    second = generator_again.random(5)

    np.testing.assert_array_equal(first, second)


def test_config_to_dict_captures_every_section() -> None:
    """Supports tracing any figure back to the parameters that produced it."""
    as_dict = cfg.default_config().to_dict()

    for section in ("fhn", "grid", "gap", "stimulus", "solver", "measurement"):
        assert section in as_dict, f"missing section {section!r} in saved config"
    assert "seed" in as_dict
    assert "label" in as_dict
