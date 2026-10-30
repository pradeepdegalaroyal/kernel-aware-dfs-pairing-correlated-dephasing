"""Compile-time encoding choices: which physical qubits form each DFS pair.

All candidates are enumerated exactly (105 pairings for 8 qubits, 10 395 for 12),
so "oracle" really is the optimum of the stated objective and every heuristic is
scored against it without search noise.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

from .channel import fidelity_all


@lru_cache(maxsize=None)
def all_pairings(n: int) -> np.ndarray:
    """All perfect matchings of ``n`` (even) qubits as an ``(P, n/2, 2)`` int array."""
    if n % 2:
        raise ValueError("number of physical qubits must be even")

    def rec(items: tuple[int, ...]) -> list[list[tuple[int, int]]]:
        if not items:
            return [[]]
        a, rest = items[0], items[1:]
        out = []
        for k, b in enumerate(rest):
            for tail in rec(rest[:k] + rest[k + 1 :]):
                out.append([(a, b)] + tail)
        return out

    return np.array(rec(tuple(range(n))), dtype=np.int64)


@lru_cache(maxsize=None)
def dfs_vectors(n: int) -> np.ndarray:
    """Logical-qubit vectors ``e_i - e_j`` of every pairing, shape ``(P, n/2, n)``."""
    pairs = all_pairings(n)
    p, l, _ = pairs.shape
    v = np.zeros((p, l, n))
    rows = np.arange(p)[:, None]
    cols = np.arange(l)[None, :]
    v[rows, cols, pairs[:, :, 0]] = 1.0
    v[rows, cols, pairs[:, :, 1]] = -1.0
    return v


@lru_cache(maxsize=None)
def bare_vectors(n: int, n_logical: int) -> np.ndarray:
    """Every choice of ``n_logical`` physical qubits for unencoded qubits, ``(C, L, n)``."""
    subsets = np.array(list(itertools.combinations(range(n), n_logical)), dtype=np.int64)
    c = len(subsets)
    v = np.zeros((c, n_logical, n))
    v[np.arange(c)[:, None], np.arange(n_logical)[None, :], subsets] = 1.0
    return v


def pairing_total_distance(dist: np.ndarray) -> np.ndarray:
    """Sum of pair distances of every pairing (geometry-only objective), shape ``(P,)``."""
    pairs = all_pairings(dist.shape[0])
    return dist[pairs[:, :, 0], pairs[:, :, 1]].sum(axis=1)


def score_strategies(
    sigma_true: np.ndarray,
    dist: np.ndarray,
    assumed: dict[str, np.ndarray],
) -> dict[str, float]:
    """Entanglement fidelity ``F_e`` (under the true covariance) of every strategy.

    ``assumed`` maps a strategy name to the covariance the compiler *believes*; the
    compiler picks the pairing maximising ``F_e`` under that belief and the pick is
    then scored under ``sigma_true``.  Fixed reference rows:

    * ``bare_best``   - best subset of unencoded qubits (oracle subset, true covariance)
    * ``bare_random`` - mean over all subsets
    * ``dfs_random``  - mean over all pairings (a kernel-blind compiler, exact average)
    * ``dfs_worst``   - worst pairing
    * ``dfs_geometry``- pairing with minimum total layout distance (no calibration data)
    * ``dfs_oracle``  - best pairing under the true covariance
    """
    n = sigma_true.shape[0]
    n_log = n // 2
    f_dfs = fidelity_all(dfs_vectors(n), sigma_true)
    f_bare = fidelity_all(bare_vectors(n, n_log), sigma_true)
    out = {
        "bare_best": float(f_bare.max()),
        "bare_random": float(f_bare.mean()),
        "dfs_random": float(f_dfs.mean()),
        "dfs_worst": float(f_dfs.min()),
        "dfs_geometry": float(f_dfs[int(np.argmin(pairing_total_distance(dist)))]),
        "dfs_oracle": float(f_dfs.max()),
    }
    for name, sig in assumed.items():
        believed = fidelity_all(dfs_vectors(n), sig)
        out[name] = float(f_dfs[int(np.argmax(believed))])
    return out
