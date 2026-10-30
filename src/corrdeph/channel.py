"""Exact logical idle-channel fidelity under correlated Gaussian dephasing.

Physical model: every physical qubit ``i`` picks up a random phase ``phi_i`` with
``phi ~ N(0, Sigma)`` (unitary ``exp(-i phi_i Z_i / 2)``).  A logical qubit is a
vector ``v_a`` in physical-qubit space:  a bare qubit has ``v = e_i``; a
decoherence-free-subspace (DFS) qubit encoded as ``|0_L> = |01>``,
``|1_L> = |10>`` on qubits ``(i, j)`` has ``v = e_i - e_j``.

For ``L`` logical qubits the entanglement (process) fidelity of the idle channel
with the identity is the exact average

    F_e = sum_t p(t) exp(-1/2 t^T M t),   M_ab = v_a^T Sigma v_b,

over ``t in {0, +1, -1}^L`` with ``p(t_a = 0) = 1/2`` and ``p(t_a = +-1) = 1/4``.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def ternary_table(n_logical: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(t, weights, outer)`` for all ``3**L`` ternary vectors ``t``."""
    t = np.array(list(itertools.product((0, 1, -1), repeat=n_logical)), dtype=float)
    w = np.prod(np.where(t == 0, 0.5, 0.25), axis=1)
    outer = (t[:, :, None] * t[:, None, :]).reshape(len(t), -1)
    return t, w, outer


def fidelity_all(vectors: np.ndarray, sigma: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Entanglement fidelity of every candidate encoding.

    ``vectors`` has shape ``(P, L, N)``: ``P`` candidate encodings of ``L``
    logical qubits on ``N`` physical qubits.  Returns an array of shape ``(P,)``.
    """
    n_cand, n_log, _ = vectors.shape
    _, w, outer = ternary_table(n_log)
    out = np.empty(n_cand)
    for lo in range(0, n_cand, chunk):
        v = vectors[lo : lo + chunk]
        m = np.einsum("pai,ij,pbj->pab", v, sigma, v, optimize=True)
        q = m.reshape(len(v), -1) @ outer.T
        out[lo : lo + chunk] = np.exp(-0.5 * q) @ w
    return out


def monte_carlo_fidelity(
    vectors: np.ndarray, sigma: np.ndarray, n_samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Sampled estimate (mean, standard error) of ``F_e`` for one encoding ``(L, N)``.

    Independent of the closed form: draws ``phi``, builds the logical phase of each
    computational basis string and evaluates ``|2^-L sum_x exp(i theta_x)|^2``.
    """
    n_log = vectors.shape[0]
    bits = np.array(list(itertools.product((0, 1), repeat=n_log)), dtype=float)  # (2^L, L)
    # logical string x contributes physical weight vector sum_a x_a v_a (relative phases)
    weights = bits @ vectors  # (2^L, N)
    phi = rng.multivariate_normal(np.zeros(sigma.shape[0]), sigma, size=n_samples, method="eigh")
    theta = -phi @ weights.T  # (S, 2^L); sign is irrelevant for |.|^2
    f = np.abs(np.exp(1j * theta).mean(axis=1)) ** 2
    return float(f.mean()), float(f.std(ddof=1) / np.sqrt(n_samples))
