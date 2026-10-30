"""Spatial correlation kernels for quasi-static dephasing noise.

Every kernel is normalised so that its correlation at distance ``xi`` equals
``1/e``.  This makes "correlation length" comparable across kernel shapes: the
shapes differ only in how fast the correlation falls inside and beyond ``xi``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

KERNELS: tuple[str, ...] = ("exponential", "gaussian", "matern32", "powerlaw")

# Matern-3/2: (1 + u) exp(-u) = 1/e at u = _U_M32 (solved once).
_U_M32: float = brentq(lambda u: (1.0 + u) * np.exp(-u) - np.exp(-1.0), 0.5, 6.0)
# Power law 1 / (1 + c r / xi) = 1/e  ->  c = e - 1.
_C_POW: float = float(np.e - 1.0)


def kernel_value(kind: str, r: np.ndarray | float, xi: float) -> np.ndarray:
    """Correlation coefficient rho(r) in [0, 1] for distance ``r`` and length ``xi``."""
    r = np.asarray(r, dtype=float)
    if xi <= 0:
        raise ValueError("xi must be positive")
    x = r / xi
    if kind == "exponential":
        return np.exp(-x)
    if kind == "gaussian":
        return np.exp(-(x**2))
    if kind == "matern32":
        return (1.0 + _U_M32 * x) * np.exp(-_U_M32 * x)
    if kind == "powerlaw":
        return 1.0 / (1.0 + _C_POW * x)
    raise ValueError(f"unknown kernel {kind!r}; choose from {KERNELS}")


def pairwise_distances(pos: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix of an ``(n, 2)`` position array."""
    d = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((d**2).sum(-1))


def covariance(kind: str, pos: np.ndarray, xi: float, sigma2: float) -> np.ndarray:
    """Phase covariance ``sigma2 * rho(r_ij)`` of the quasi-static dephasing phases."""
    return sigma2 * kernel_value(kind, pairwise_distances(pos), xi)


def fit_length(kind: str, dist: np.ndarray, rho_target: np.ndarray) -> float:
    """Least-squares correlation length of kernel ``kind`` to a target correlation matrix.

    Only off-diagonal pairs enter the fit.  A log-spaced grid search is used: the
    objective is one-dimensional and cheap, and it avoids local-minimum surprises.
    """
    iu = np.triu_indices_from(dist, k=1)
    r, t = dist[iu], rho_target[iu]
    grid = np.geomspace(0.15, 20.0, 80)
    losses = [np.sum((kernel_value(kind, r, g) - t) ** 2) for g in grid]
    return float(grid[int(np.argmin(losses))])
