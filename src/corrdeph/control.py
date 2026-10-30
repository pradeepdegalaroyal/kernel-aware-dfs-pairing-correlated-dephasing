"""Dynamical-decoupling (Carr-Purcell-Meiboom-Gill, CPMG) suppression of quasi-static phase noise.

For a time-domain switching function ``y(t) = +-1`` the accumulated phase variance is
``(1/pi) * int_0^inf S(w) |F(w)|^2 dw`` with ``F(w) = int_0^T y(t) exp(i w t) dt``.
The temporal spectrum is a 1/f-type ``S(w) = 1/w`` between a low and a high cut-off.
Because the spatial and temporal parts are separable in this model, an identical
sequence on every qubit rescales the whole covariance by ``g(n) = Lambda(n)/Lambda(0)``.
"""
from __future__ import annotations

import numpy as np


def _filter_power(n_pulses: int, omega: np.ndarray, total_time: float = 1.0) -> np.ndarray:
    """``|F(w)|^2`` for CPMG with ``n_pulses`` pi pulses in time ``total_time``."""
    if n_pulses == 0:
        edges = np.array([0.0, total_time])
    else:
        centres = total_time * (np.arange(1, n_pulses + 1) - 0.5) / n_pulses
        edges = np.concatenate([[0.0], centres, [total_time]])
    f = np.zeros_like(omega, dtype=complex)
    for k in range(len(edges) - 1):
        f += (-1) ** k * (np.exp(1j * omega * edges[k + 1]) - np.exp(1j * omega * edges[k])) / (1j * omega)
    return np.abs(f) ** 2


def suppression_factor(n_pulses: int, w_lo: float = 2 * np.pi * 1e-3, w_hi: float = 2 * np.pi * 1e3) -> float:
    """``g(n) = Lambda(n) / Lambda(0)`` for 1/f noise (T = 1, frequencies in units of 1/T)."""
    omega = np.geomspace(w_lo, w_hi, 200_000)
    s = 1.0 / omega

    def lam(n: int) -> float:
        # integrate over ln(w): dw = w dlnw
        return float(np.trapezoid(s * _filter_power(n, omega) * omega, np.log(omega)))

    return lam(n_pulses) / lam(0)
