"""corrdeph: kernel-aware decoherence-free-subspace pairing under correlated dephasing."""
from .calibration import estimate_covariance
from .channel import fidelity_all, monte_carlo_fidelity
from .compile import all_pairings, bare_vectors, dfs_vectors, score_strategies
from .kernels import KERNELS, covariance, kernel_value

__all__ = [
    "KERNELS", "kernel_value", "covariance", "fidelity_all", "monte_carlo_fidelity",
    "all_pairings", "dfs_vectors", "bare_vectors", "score_strategies", "estimate_covariance",
]
__version__ = "0.1.0"
