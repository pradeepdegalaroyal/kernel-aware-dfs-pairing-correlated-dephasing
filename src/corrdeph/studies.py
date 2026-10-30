"""The experiments: each function returns a tidy ``pandas.DataFrame`` of raw results."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibration import estimate_covariance
from .compile import score_strategies
from .kernels import KERNELS, covariance, fit_length, kernel_value, pairwise_distances
from .layouts import jittered_grid

SIGMA2_DEFAULT = 0.3  # single-qubit phase variance (rad^2) over the idle window


@dataclass(frozen=True)
class Instance:
    """One chip layout together with the true noise covariance on it."""

    dist: np.ndarray  # layout distances (what the compiler can see)
    sigma_true: np.ndarray
    rho_true: np.ndarray


def make_instance(
    n: int, seed: int, layout: int, kernel: str, xi: float, sigma2: float, mismatch: float = 0.0
) -> Instance:
    """Draw layout ``(seed, layout)``; the noise geometry is displaced by ``mismatch`` pitches.

    The displacement field is a fixed unit-normal draw scaled by ``mismatch`` so that
    sweeps over ``mismatch`` use nested, comparable instances (common random numbers).
    """
    rng = np.random.default_rng([seed, layout, n])
    pos = jittered_grid(n, rng)
    field = np.random.default_rng([seed, layout, n, 99]).normal(size=pos.shape)
    hidden = pos + mismatch * field
    sigma = covariance(kernel, hidden, xi, sigma2)
    return Instance(pairwise_distances(pos), sigma, sigma / sigma2)


def assumed_fits(inst: Instance, sigma2: float, kinds: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Parametric beliefs: each kernel family fitted to the (exact) pair correlations."""
    out = {}
    for kind in kinds:
        xi_hat = fit_length(kind, inst.dist, inst.rho_true)
        out[f"dfs_fit_{kind}"] = sigma2 * kernel_value(kind, inst.dist, xi_hat)
    return out


def retained_gain(eps_random: float, eps: float, eps_oracle: float) -> float:
    """Fraction of the oracle's infidelity reduction (vs a random pairing) that a strategy keeps."""
    denom = eps_random - eps_oracle
    return float("nan") if denom <= 1e-12 else (eps_random - eps) / denom


def _per_seed(rows: list[dict], keys: list[str]) -> pd.DataFrame:
    """Average layouts within a seed so that error bars are over seeds."""
    df = pd.DataFrame(rows)
    return df.groupby(keys + ["seed"], as_index=False).mean(numeric_only=True)


# ----------------------------------------------------------------------------- E1
def study_main(
    n: int, seeds: int, layouts: int, xis: list[float], mismatches: list[float],
    kernel: str = "exponential", sigma2: float = SIGMA2_DEFAULT, shots: int = 300,
) -> pd.DataFrame:
    """Strategy comparison vs. correlation length ``xi`` at fixed noise strength."""
    rows = []
    for seed in range(seeds):
        for lay in range(layouts):
            for mm in mismatches:
                for xi in xis:
                    inst = make_instance(n, seed, lay, kernel, xi, sigma2, mm)
                    rng = np.random.default_rng([seed, lay, n, 5])
                    assumed = assumed_fits(inst, sigma2, ("exponential",))
                    assumed["dfs_shots"] = estimate_covariance(inst.sigma_true, shots, rng)
                    sc = score_strategies(inst.sigma_true, inst.dist, assumed)
                    rows.append({"seed": seed, "mismatch": mm, "xi": xi, **{f"eps_{k}": 1 - v for k, v in sc.items()}})
    return _per_seed(rows, ["mismatch", "xi"])


# ----------------------------------------------------------------------------- E2
def study_shape(
    n: int, seeds: int, layouts: int, xi: float, mismatches: list[float], sigma2: float = SIGMA2_DEFAULT
) -> pd.DataFrame:
    """Retained gain for every (true kernel, assumed kernel) pair, calibrated on exact correlations."""
    rows = []
    for true_kind in KERNELS:
        for mismatch in mismatches:
            for seed in range(seeds):
                acc = []
                for lay in range(layouts):
                    inst = make_instance(n, seed, lay, true_kind, xi, sigma2, mismatch)
                    assumed = assumed_fits(inst, sigma2, KERNELS)
                    sc = score_strategies(inst.sigma_true, inst.dist, assumed)
                    acc.append({k: 1 - v for k, v in sc.items()})
                m = pd.DataFrame(acc).mean()
                for a in KERNELS:
                    rows.append({
                        "true_kernel": true_kind, "assumed_kernel": a, "seed": seed, "mismatch": mismatch,
                        "retained_gain": retained_gain(m["dfs_random"], m[f"dfs_fit_{a}"], m["dfs_oracle"]),
                        "retained_gain_geometry": retained_gain(m["dfs_random"], m["dfs_geometry"], m["dfs_oracle"]),
                        "eps_oracle": m["dfs_oracle"], "eps_random": m["dfs_random"],
                        "eps_assumed": m[f"dfs_fit_{a}"],
                    })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E3
def study_calibration(
    n: int, seeds: int, layouts: int, shots_list: list[int], mismatches: list[float],
    kernel: str = "exponential", xi: float = 2.0, sigma2: float = SIGMA2_DEFAULT,
) -> pd.DataFrame:
    """Retained gain of shot-calibrated, geometry-only and parametric-fit compilers."""
    rows = []
    for seed in range(seeds):
        for mm in mismatches:
            acc = []
            for lay in range(layouts):
                inst = make_instance(n, seed, lay, kernel, xi, sigma2, mm)
                fits = assumed_fits(inst, sigma2, ("exponential",))
                for si, m_shots in enumerate(shots_list):
                    rng = np.random.default_rng([seed, lay, n, si, 17])
                    assumed = dict(fits)
                    assumed["dfs_shots"] = estimate_covariance(inst.sigma_true, m_shots, rng)
                    sc = score_strategies(inst.sigma_true, inst.dist, assumed)
                    acc.append({"shots": m_shots, **{k: 1 - v for k, v in sc.items()}})
            df = pd.DataFrame(acc).groupby("shots").mean()
            for m_shots, r in df.iterrows():
                rows.append({
                    "seed": seed, "mismatch": mm, "shots": m_shots,
                    "retained_shots": retained_gain(r["dfs_random"], r["dfs_shots"], r["dfs_oracle"]),
                    "retained_geometry": retained_gain(r["dfs_random"], r["dfs_geometry"], r["dfs_oracle"]),
                    "retained_fit_exp": retained_gain(r["dfs_random"], r["dfs_fit_exponential"], r["dfs_oracle"]),
                    "eps_shots": r["dfs_shots"], "eps_oracle": r["dfs_oracle"],
                    "eps_random": r["dfs_random"], "eps_geometry": r["dfs_geometry"],
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E4
def study_phase(
    n: int, seeds: int, layouts: int, xis: list[float], sigma2s: list[float], kernel: str = "exponential",
    mismatch: float = 0.0, shots: int = 0,
) -> pd.DataFrame:
    """DFS vs. best bare qubits over (xi, sigma2); break-even gate error ``p*``.

    ``shots = 0`` uses the oracle pairing (an upper bound); ``shots > 0`` uses the pairing
    compiled from a ``shots``-per-pair Ramsey covariance estimate.

    Encoding and decoding each logical qubit costs one extra CNOT, so a DFS register
    carries ``2 L`` extra two-qubit gates.  To first order each costs ``p`` of
    entanglement fidelity, hence ``p* = (eps_bare - eps_dfs) / (2 L)``.
    """
    n_log = n // 2
    rows = []
    for seed in range(seeds):
        for xi in xis:
            insts = [make_instance(n, seed, lay, kernel, xi, 1.0, mismatch) for lay in range(layouts)]
            for s2 in sigma2s:
                eb, ed = [], []
                for lay, inst in enumerate(insts):
                    sig = inst.sigma_true * s2
                    assumed = {}
                    if shots > 0:
                        rng = np.random.default_rng([seed, lay, n, 41])
                        assumed["dfs_shots"] = estimate_covariance(sig, shots, rng)
                    sc = score_strategies(sig, inst.dist, assumed)
                    eb.append(1 - sc["bare_best"])
                    ed.append(1 - sc["dfs_shots" if shots > 0 else "dfs_oracle"])
                eps_b, eps_d = float(np.mean(eb)), float(np.mean(ed))
                rows.append({
                    "seed": seed, "xi": xi, "sigma2": s2, "eps_bare": eps_b, "eps_dfs": eps_d,
                    "p_break_even": (eps_b - eps_d) / (2 * n_log),
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E5
def study_scaling(
    ns: list[int], seeds: int, layouts: int, kernel: str = "exponential", xi: float = 2.0,
    sigma2: float = SIGMA2_DEFAULT, mismatch: float = 1.0, shots: int = 300,
) -> pd.DataFrame:
    """How the strategies scale with register size (n physical qubits, n/2 logical)."""
    rows = []
    for n in ns:
        for seed in range(seeds):
            acc = []
            for lay in range(layouts):
                inst = make_instance(n, seed, lay, kernel, xi, sigma2, mismatch)
                rng = np.random.default_rng([seed, lay, n, 23])
                assumed = {"dfs_shots": estimate_covariance(inst.sigma_true, shots, rng)}
                sc = score_strategies(inst.sigma_true, inst.dist, assumed)
                acc.append({k: 1 - v for k, v in sc.items()})
            m = pd.DataFrame(acc).mean()
            rows.append({
                "n_physical": n, "n_logical": n // 2, "seed": seed,
                **{f"eps_{k}": v for k, v in m.items()},
                "retained_shots": retained_gain(m["dfs_random"], m["dfs_shots"], m["dfs_oracle"]),
                "retained_geometry": retained_gain(m["dfs_random"], m["dfs_geometry"], m["dfs_oracle"]),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- E6
def study_ablation(
    n: int, seeds: int, layouts: int, pulses: int, kernel: str = "exponential", xi: float = 2.0,
    sigma2: float = SIGMA2_DEFAULT, mismatch: float = 1.0, shots: int = 300,
) -> pd.DataFrame:
    """Remove one ingredient at a time: dynamical decoupling, DFS, calibration, pairing choice."""
    from .control import suppression_factor

    g = suppression_factor(pulses)
    rows = []
    for seed in range(seeds):
        acc = []
        for lay in range(layouts):
            row = {}
            for label, scale in (("free", 1.0), ("dd", g)):
                inst = make_instance(n, seed, lay, kernel, xi, sigma2 * scale, mismatch)
                rng = np.random.default_rng([seed, lay, n, 31])
                assumed = {"dfs_shots": estimate_covariance(inst.sigma_true, shots, rng)}
                sc = score_strategies(inst.sigma_true, inst.dist, assumed)
                for k, v in sc.items():
                    row[f"{k}|{label}"] = 1 - v
            acc.append(row)
        m = pd.DataFrame(acc).mean()
        for key, val in m.items():
            strat, label = key.split("|")
            rows.append({"seed": seed, "strategy": strat, "dd": label == "dd", "eps": val, "dd_factor": g})
    return pd.DataFrame(rows)
