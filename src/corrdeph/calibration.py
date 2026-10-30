"""Finite-shot calibration of the pairwise phase covariance.

A Bell-type Ramsey experiment on qubits ``(i, j)`` has visibility
``V_ij = exp(-Var(phi_i - phi_j) / 2) = exp(-(sigma2 - Sigma_ij))`` when the single-qubit
variance is ``sigma2``.  With ``M`` shots the visibility is estimated from a binomial
count, which gives a noisy, possibly non-positive-semidefinite covariance estimate.
"""
from __future__ import annotations

import numpy as np


def estimate_covariance(sigma_true: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Covariance estimate from ``shots`` Ramsey shots per pair.

    The single-qubit variance ``sigma2`` is taken as known (it is a 1-qubit Ramsey
    fit, much cheaper than the ``n(n-1)/2`` pair experiments); only the off-diagonal
    correlations are estimated.  ``shots <= 0`` returns the exact covariance.
    """
    if shots <= 0:
        return sigma_true.copy()
    n = sigma_true.shape[0]
    sigma2 = float(np.mean(np.diag(sigma_true)))
    est = np.diag(np.diag(sigma_true)).astype(float)
    iu = np.triu_indices(n, k=1)
    vis = np.exp(-(sigma2 - sigma_true[iu]))
    k = rng.binomial(shots, (1.0 + vis) / 2.0)
    vis_hat = np.clip(2.0 * k / shots - 1.0, 1e-3, 1.0)
    cov_hat = sigma2 + np.log(vis_hat)
    est[iu] = cov_hat
    est[(iu[1], iu[0])] = cov_hat
    return est
