"""Fast tests: kernel normalisation, exact-vs-sampled fidelity, limiting cases, estimator."""
import numpy as np
import pytest

from corrdeph import (
    KERNELS, all_pairings, bare_vectors, covariance, dfs_vectors, estimate_covariance,
    fidelity_all, kernel_value, monte_carlo_fidelity, score_strategies,
)
from corrdeph.control import suppression_factor
from corrdeph.layouts import jittered_grid
from corrdeph.validation import qiskit_dfs_fidelity


@pytest.mark.parametrize("kind", KERNELS)
def test_kernel_is_one_over_e_at_xi(kind):
    assert kernel_value(kind, 2.5, 2.5) == pytest.approx(np.exp(-1), rel=1e-6)
    assert kernel_value(kind, 0.0, 2.5) == pytest.approx(1.0)


@pytest.mark.parametrize("kind", KERNELS)
def test_covariance_is_psd(kind):
    pos = jittered_grid(8, np.random.default_rng(0))
    ev = np.linalg.eigvalsh(covariance(kind, pos, 1.7, 0.3))
    assert ev.min() > -1e-9


def test_pairing_counts():
    assert [len(all_pairings(n)) for n in (4, 6, 8, 10)] == [3, 15, 105, 945]


def test_no_noise_gives_unit_fidelity():
    f = fidelity_all(dfs_vectors(6), np.zeros((6, 6)))
    assert np.allclose(f, 1.0)


def test_perfectly_correlated_noise_is_invisible_to_dfs():
    n = 6
    sigma = 0.5 * np.ones((n, n))  # one common phase
    assert np.allclose(fidelity_all(dfs_vectors(n), sigma), 1.0)
    assert fidelity_all(bare_vectors(n, 3), sigma).max() < 0.9


def test_independent_noise_dfs_is_worse_than_bare():
    n, s2 = 6, 0.3
    sigma = s2 * np.eye(n)
    assert fidelity_all(dfs_vectors(n), sigma).max() < fidelity_all(bare_vectors(n, 3), sigma).max()


def test_single_bare_qubit_closed_form():
    s2 = 0.4
    f = fidelity_all(np.array([[[1.0]]]), np.array([[s2]]))[0]
    assert f == pytest.approx(0.5 * (1 + np.exp(-s2 / 2)))


def test_closed_form_matches_sampling():
    rng = np.random.default_rng(1)
    sigma = covariance("exponential", jittered_grid(6, rng), 1.5, 0.4)
    v = dfs_vectors(6)[5]
    exact = fidelity_all(v[None], sigma)[0]
    mc, se = monte_carlo_fidelity(v, sigma, 20000, rng)
    assert abs(exact - mc) < 5 * se


def test_closed_form_matches_qiskit_encoder():
    rng = np.random.default_rng(2)
    sigma = covariance("gaussian", jittered_grid(4, rng), 1.5, 0.5)
    exact = fidelity_all(dfs_vectors(4)[2][None], sigma)[0]
    assert qiskit_dfs_fidelity(all_pairings(4)[2], sigma, 300, rng) == pytest.approx(exact, abs=0.02)


def test_oracle_beats_random_and_worst_is_lowest():
    rng = np.random.default_rng(3)
    pos = jittered_grid(8, rng)
    from corrdeph.kernels import pairwise_distances

    sigma = covariance("exponential", pos, 2.0, 0.3)
    sc = score_strategies(sigma, pairwise_distances(pos), {})
    assert sc["dfs_oracle"] >= sc["dfs_geometry"] >= sc["dfs_worst"]
    assert sc["dfs_oracle"] > sc["dfs_random"]


def test_shot_estimator_converges():
    rng = np.random.default_rng(4)
    sigma = covariance("exponential", jittered_grid(6, rng), 2.0, 0.3)
    err = lambda m: np.abs(estimate_covariance(sigma, m, np.random.default_rng(5)) - sigma).max()
    assert err(100000) < err(100)
    assert np.allclose(estimate_covariance(sigma, 0, rng), sigma)


def test_dynamical_decoupling_reduces_variance():
    assert suppression_factor(0) == pytest.approx(1.0)
    assert suppression_factor(4) < suppression_factor(2) < 1.0
