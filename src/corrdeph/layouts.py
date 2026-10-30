"""Chip layouts: jittered rectangular qubit grids and a hidden noise geometry."""
from __future__ import annotations

import numpy as np

GRID_SHAPES: dict[int, tuple[int, int]] = {4: (2, 2), 6: (2, 3), 8: (2, 4), 10: (2, 5), 12: (3, 4)}


def jittered_grid(n: int, rng: np.random.Generator, jitter: float = 0.25) -> np.ndarray:
    """``n`` qubit positions on a unit-pitch grid with Gaussian placement jitter."""
    rows, cols = GRID_SHAPES[n]
    xs, ys = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    pos = np.stack([xs.ravel(), ys.ravel()], axis=1)
    return pos + rng.normal(0.0, jitter, size=pos.shape)


def hidden_geometry(pos: np.ndarray, mismatch: float, rng: np.random.Generator) -> np.ndarray:
    """Positions that the noise *actually* sees (e.g. shared bias lines or flux paths).

    ``mismatch`` is the standard deviation, in units of the grid pitch, of the
    displacement between the chip layout and the noise geometry.  ``0`` means the
    noise follows the chip layout exactly.
    """
    return pos + rng.normal(0.0, mismatch, size=pos.shape) if mismatch > 0 else pos.copy()
