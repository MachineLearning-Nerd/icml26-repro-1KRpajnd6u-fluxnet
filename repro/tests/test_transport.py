from __future__ import annotations

import numpy as np

from src.transport import d_head_1d, l_head_1d, periodic_transport_1d, periodic_transport_2d, u_head_1d


OFFSETS = (-2, -1, 1, 2)


def allocation() -> np.ndarray:
    return np.full((len(OFFSETS), 7), 1.0 / len(OFFSETS))


def test_periodic_signed_transport_conserves_mass() -> None:
    field = np.array([-1.0, 0.5, 2.0, -0.4, 1.1, 0.0, 0.7])
    flows = np.arange(len(OFFSETS) * len(field), dtype=float).reshape(len(OFFSETS), -1) / 17
    assert np.isclose(periodic_transport_1d(field, flows, OFFSETS).sum(), field.sum())


def test_l_head_preserves_lower_bound_and_mass() -> None:
    field = np.array([0.2, 1.0, 0.3, 2.1, 0.7, 0.25, 1.4])
    updated, _ = l_head_1d(field, np.full(7, 0.999), allocation(), OFFSETS, 0.2)
    assert updated.min() >= 0.2
    assert np.isclose(updated.sum(), field.sum())


def test_u_head_preserves_upper_bound_and_mass() -> None:
    field = np.array([0.1, 0.9, 0.2, 0.7, 0.4, 0.95, 0.3])
    updated, _ = u_head_1d(field, np.full(7, 0.999), allocation(), OFFSETS, 1.0)
    assert updated.max() <= 1.0
    assert np.isclose(updated.sum(), field.sum())


def test_periodic_two_dimensional_transport_conserves_mass() -> None:
    rng = np.random.default_rng(9)
    field = rng.normal(size=(5, 6))
    offsets = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    flows = rng.normal(size=(len(offsets), *field.shape))
    assert np.isclose(periodic_transport_2d(field, flows, offsets).sum(), field.sum())


def test_d_head_is_not_a_hard_dual_bound() -> None:
    rng = np.random.default_rng(20260717)
    saw_violation = False
    for _ in range(32):
        field = rng.uniform(size=5)
        updated = d_head_1d(
            field,
            rng.uniform(size=5),
            rng.dirichlet(np.ones(len(OFFSETS)), size=5).T,
            rng.uniform(size=5),
            rng.dirichlet(np.ones(len(OFFSETS)), size=5).T,
            OFFSETS,
            0.0,
            1.0,
        )
        saw_violation |= bool(updated.min() < 0.0 or updated.max() > 1.0)
    assert saw_violation
