# Method

## Noise model
Each physical qubit `i` receives a random phase `phi_i` (unitary `exp(-i phi_i Z_i / 2)`) with
`phi ~ N(0, Sigma)`, `Sigma_ij = sigma2 * rho(r_ij)`. This is quasi-static Gaussian dephasing over one idle window.
`rho` is one of four kernels (exponential, Gaussian, Matérn 3/2, power law `1/(1 + c r/xi)`), each scaled so that
`rho(xi) = 1/e`, which makes correlation length `xi` comparable across shapes. Distances `r_ij` are measured in qubit
pitches on a jittered grid (6-12 qubits, placement jitter 0.25 pitch).

**Geometry mismatch.** The noise sees positions displaced from the chip layout by a Gaussian field of standard
deviation `m` pitches (a synthetic stand-in for shared bias lines or flux paths). The compiler only knows the layout.

## Encodings and exact metric
* **Bare**: `L = N/2` unencoded qubits chosen from `N` physical qubits (vector `e_i`).
* **Decoherence-free subspace (DFS)**: a logical qubit on a qubit pair `(i, j)` with `|0_L> = |01>`, `|1_L> = |10>`
  (vector `e_i - e_j`); `N/2` pairs use all `N` qubits.

For logical vectors `v_a`, the entanglement fidelity of the idle channel with the identity is exactly
`F_e = sum_t p(t) exp(-1/2 t^T M t)`, `M_ab = v_a^T Sigma v_b`, over `t in {0, +1, -1}^L`
(`p = 1/2` for 0, `1/4` for each of +-1). This is the metric (no state dependence, no sampling noise). It is
checked against sampled phases and against a Qiskit-built CNOT+X encoder (`results/validation.csv`).

## Compile strategies (all pairings enumerated exactly)
| Name | What the compiler believes | Role |
|---|---|---|
| `bare_best` | true `Sigma`, best qubit subset | strong baseline: no encoding, oracle subset choice |
| `dfs_random` | nothing: exact mean over all pairings | kernel-blind compiler |
| `dfs_geometry` | layout only: minimum total pair distance | classical baseline, no calibration data |
| `dfs_fit_<kernel>` | layout + one kernel family, length fitted to exact correlations | kernel-misspecification test |
| `dfs_shots` | pairwise covariance estimated from `M` Ramsey shots per pair | realistic calibration |
| `dfs_oracle` | true `Sigma` | upper bound |

**Standard baselines.** The standard quantum-control baseline is dynamical decoupling (DD): a
Carr-Purcell-Meiboom-Gill (CPMG) sequence of `n` pi pulses against 1/f-type noise. Its filter function is integrated
numerically (`control.py`); for a separable spatial-times-temporal spectrum it rescales `Sigma` by a factor `g(n)`.
The standard encoding baseline is the plain DFS pairing above, with a nearest-neighbour choice.

**Shot calibration.** A Bell-type Ramsey experiment gives visibility `V_ij = exp(-(sigma2 - Sigma_ij))`; with `M`
shots, `V_hat` comes from a binomial count and `Sigma_hat_ij = sigma2 + ln V_hat` (single-qubit variance known).

## Metrics
`eps = 1 - F_e` (lower is better); retained gain `G` (see `RESEARCH_GAP.md`); break-even two-qubit error
`p* = (eps_bare - eps_dfs) / (2 L)`: encoding and decoding each logical qubit cost one extra controlled-NOT (CNOT),
and to first order each costs `p` of fidelity. This is a coarse budget, not a gate-level simulation.

## Experiments
* **E1** strategies vs `xi` (0.4-7 pitches), 10 physical qubits, mismatch 0 and 1, `sigma2 = 0.3` rad^2.
* **E2** kernel shape: 4 true x 4 assumed kernels, 8 qubits, `xi = 2`, mismatch 0 and 0.5.
* **E3** calibration: shots per pair 10-10000 x mismatch 0-2.
* **E4** noise/robustness sweep: `sigma2` 0.02-0.6 x `xi` 0.4-8, oracle vs 300-shot calibrated (mismatch 1).
* **E5** scaling: 6, 8, 10, 12 physical qubits.
* **E6** ablation: remove DD, DFS, calibration or pairing choice; CPMG with 1, 2, 4, 8 pulses.

## Ablation
Each row of E6 removes exactly one ingredient from the full pipeline (calibrated DFS + CPMG): without DD, without DFS
(`bare_best`), without calibration (`dfs_geometry`), without any pairing choice (`dfs_random`).

## Seeds
Five seeds; each seed is a set of 8 random chip layouts (E5 uses 3 seeds x 3 layouts). Layouts are averaged inside a
seed, error bars are the standard deviation over seeds. Layouts are shared across settings (common random numbers).
