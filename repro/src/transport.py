"""Periodic feasible-transport updates used by FluxNet's Proposition 1–3.

The functions are clean-room NumPy implementations of the mathematical
transport heads in arXiv:2602.01941. They separate hard one-sided guarantees
from the D-head's empirical dual-bound setup.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


Array = np.ndarray


def periodic_transport_1d(field: Array, flows: Array, offsets: Sequence[int]) -> Array:
    """Apply a periodic directed transport plan to a one-dimensional field."""
    field = np.asarray(field, dtype=float)
    flows = np.asarray(flows, dtype=float)
    _validate_1d(field, flows, offsets)
    updated = field - flows.sum(axis=0)
    for direction, offset in enumerate(offsets):
        updated += np.roll(flows[direction], -int(offset))
    return updated


def l_head_1d(
    field: Array,
    outflow_fraction: Array,
    allocation: Array,
    offsets: Sequence[int],
    lower_bound: float,
) -> tuple[Array, Array]:
    """L-head: capacity-limited outflow and Proposition 2's lower bound."""
    field = np.asarray(field, dtype=float)
    outflow_fraction = np.asarray(outflow_fraction, dtype=float)
    allocation = np.asarray(allocation, dtype=float)
    _validate_simplex_1d(field, outflow_fraction, allocation, offsets)
    if np.any(field < lower_bound):
        raise ValueError("L-head premise requires field >= lower_bound")
    flows = (field - lower_bound) * outflow_fraction * allocation
    return periodic_transport_1d(field, flows, offsets), flows


def u_head_1d(
    field: Array,
    inflow_fraction: Array,
    allocation: Array,
    offsets: Sequence[int],
    upper_bound: float,
) -> tuple[Array, Array]:
    """U-head: capacity-limited inflow and Proposition 3's upper bound."""
    field = np.asarray(field, dtype=float)
    inflow_fraction = np.asarray(inflow_fraction, dtype=float)
    allocation = np.asarray(allocation, dtype=float)
    _validate_simplex_1d(field, inflow_fraction, allocation, offsets)
    if np.any(field > upper_bound):
        raise ValueError("U-head premise requires field <= upper_bound")
    incoming_budget = (upper_bound - field) * inflow_fraction
    updated = field + incoming_budget
    for direction, offset in enumerate(offsets):
        updated -= np.roll(incoming_budget * allocation[direction], int(offset))
    return updated, incoming_budget * allocation


def d_head_1d(
    field: Array,
    outflow_fraction: Array,
    outflow_allocation: Array,
    inflow_fraction: Array,
    inflow_allocation: Array,
    offsets: Sequence[int],
    lower_bound: float,
    upper_bound: float,
) -> Array:
    """The D-head average: conservative but not universally dual-bounded."""
    lower_update, _ = l_head_1d(
        field, outflow_fraction, outflow_allocation, offsets, lower_bound
    )
    upper_update, _ = u_head_1d(
        field, inflow_fraction, inflow_allocation, offsets, upper_bound
    )
    return 0.5 * (lower_update + upper_update)


def periodic_transport_2d(
    field: Array, flows: Array, offsets: Sequence[tuple[int, int]]
) -> Array:
    """Apply the Proposition 1 transport update on a periodic 2D grid."""
    field = np.asarray(field, dtype=float)
    flows = np.asarray(flows, dtype=float)
    if field.ndim != 2 or flows.shape != (len(offsets), *field.shape):
        raise ValueError("flows must have shape [directions, height, width]")
    updated = field - flows.sum(axis=0)
    for direction, offset in enumerate(offsets):
        updated += np.roll(flows[direction], tuple(-x for x in offset), axis=(0, 1))
    return updated


def _validate_1d(field: Array, flows: Array, offsets: Sequence[int]) -> None:
    if field.ndim != 1 or flows.shape != (len(offsets), len(field)):
        raise ValueError("flows must have shape [directions, length]")
    if not offsets or any(offset == 0 for offset in offsets):
        raise ValueError("directions must be nonempty and exclude the center")


def _validate_simplex_1d(
    field: Array, fraction: Array, allocation: Array, offsets: Sequence[int]
) -> None:
    _validate_1d(field, allocation, offsets)
    if fraction.shape != field.shape:
        raise ValueError("fraction must have one value per cell")
    if np.any(fraction < 0.0) or np.any(fraction > 1.0):
        raise ValueError("fractions must lie in [0, 1]")
    if np.any(allocation < 0.0) or not np.allclose(allocation.sum(axis=0), 1.0):
        raise ValueError("allocation must be a nonnegative simplex at every cell")
