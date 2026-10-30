"""Independent checks of the closed-form fidelity.

1. Sampled phases vs. the exact ``F_e`` formula (bare and DFS registers).
2. A Qiskit-built DFS encoder (CNOT + X) applied to sampled dephasing and decoded with its
   adjoint, so the algebra ``|0_L> = |01>, |1_L> = |10>`` is checked on real gates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from .channel import fidelity_all, monte_carlo_fidelity
from .compile import all_pairings, bare_vectors, dfs_vectors
from .kernels import covariance
from .layouts import jittered_grid


def _encoder_isometry(n_phys: int, pairs: np.ndarray) -> np.ndarray:
    """``2^N x 2^L`` isometry of the Qiskit encoder; pair ``(i, j)`` carries one logical qubit."""
    qc = QuantumCircuit(n_phys)
    for i, j in pairs:
        qc.cx(int(i), int(j))
        qc.x(int(j))
    u = Operator(qc).data
    n_log = len(pairs)
    cols = []
    for x in range(2**n_log):
        idx = 0
        for a, (i, _) in enumerate(pairs):
            if (x >> a) & 1:
                idx |= 1 << int(i)
        cols.append(u[:, idx])
    return np.stack(cols, axis=1)


def qiskit_dfs_fidelity(pairs: np.ndarray, sigma: np.ndarray, n_samples: int, rng: np.random.Generator) -> float:
    """Sampled ``F_e = E |Tr(E^dag D_phi E)|^2 / 4^L`` with the Qiskit encoder ``E``."""
    n = sigma.shape[0]
    n_log = len(pairs)
    iso = _encoder_isometry(n, pairs)
    states = np.arange(2**n)
    z = 1.0 - 2.0 * ((states[:, None] >> np.arange(n)[None, :]) & 1)  # +1 for bit 0
    phi = rng.multivariate_normal(np.zeros(n), sigma, size=n_samples, method="eigh")
    f = np.empty(n_samples)
    for s in range(n_samples):
        diag = np.exp(-0.5j * (z @ phi[s]))
        tr = np.trace(iso.conj().T @ (diag[:, None] * iso))
        f[s] = abs(tr) ** 2 / 4**n_log
    return float(f.mean())


def run_validation(n_samples: int = 20000, n_qiskit: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Compare closed form, direct sampling and the Qiskit encoder on random instances."""
    rng = np.random.default_rng(seed)
    rows = []
    for kernel, xi in (("exponential", 1.5), ("gaussian", 2.0), ("powerlaw", 1.0)):
        n = 6
        pos = jittered_grid(n, rng)
        sigma = covariance(kernel, pos, xi, 0.4)
        for name, vecs in (("dfs", dfs_vectors(n)[7]), ("bare", bare_vectors(n, 3)[3])):
            exact = float(fidelity_all(vecs[None], sigma)[0])
            mc, se = monte_carlo_fidelity(vecs, sigma, n_samples, rng)
            rows.append({"kernel": kernel, "encoding": name, "exact": exact, "monte_carlo": mc,
                         "std_error": se, "qiskit_encoder": float("nan")})
    n = 4
    pos = jittered_grid(n, rng)
    sigma = covariance("exponential", pos, 1.5, 0.4)
    idx = 1
    pairs = all_pairings(n)[idx]
    exact = float(fidelity_all(dfs_vectors(n)[idx][None], sigma)[0])
    mc, se = monte_carlo_fidelity(dfs_vectors(n)[idx], sigma, n_samples, rng)
    qk = qiskit_dfs_fidelity(pairs, sigma, n_qiskit, rng)
    rows.append({"kernel": "exponential", "encoding": "dfs_qiskit_n4", "exact": exact, "monte_carlo": mc,
                 "std_error": se, "qiskit_encoder": qk})
    return pd.DataFrame(rows)
