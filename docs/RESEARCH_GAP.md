# Research gap: does the shape of a correlated-dephasing model matter for compile-time encoding?

Scale of everything below: classical simulation of 6-12 physical qubits (3-6 logical qubits) on a laptop.
Nothing here was run on hardware.

## State of the art in the provided literature

Quantum processors store information in relative phases, so spatially and temporally correlated dephasing is a
first-order error source. Abdel-Aty et al., 2027 (Digital Object Identifier, DOI, 10.1016/j.asej.2026.104502) formalise correlated dephasing as a
template with exponential, Gaussian, Matérn and power-law spatial kernels. They show that a per-layer exponential
approximation can overestimate loss in a Greenberger-Horne-Zeilinger (GHZ) preparation circuit. They keep
"mitigation and compilation claims quantitatively cautious", so no compile decision is driven by the kernel.
Tariq and Chatzinotas, 2026 (DOI 10.1038/s41534-026-01308-w) show, for quantum multiple-input multiple-output
(QuMIMO) channels, that an encoding tuned with channel state information (CSI) gains most when crosstalk dominates.
McClean et al., 2016 (DOI 10.1088/1367-2630/18/2/023023) argue that a noise-aware algorithm design can suppress
errors on a pre-threshold device. The adjacent communications papers treat the same statistical problem: Yahya et al.,
2027 (DOI 10.1016/j.aeue.2026.156647) model spatially correlated fluid-antenna ports with a Gaussian copula and test
aged CSI, and Todorovic et al., 2027 (DOI 10.1007/978-3-032-36445-6_10) stress comparison against existing channel models.

The standard control-side baseline (dynamical decoupling, DD) and the standard encoding-side baseline
(a decoherence-free subspace, DFS, built from a pair of qubits that see a common phase) are textbook techniques.
They are not part of the provided literature, so they are described in `docs/METHOD.md` without citation.

## Related work table

| Paper | Year | Role | What it does | What it does NOT cover (for this question) |
|---|---|---|---|---|
| Abdel-Aty et al., DOI 10.1016/j.asej.2026.104502 | 2027 | recent | Correlated-dephasing template with four spatial kernels, GHZ circuit test | Using the kernel to choose which qubits to pair; how wrong a kernel guess can be; calibration cost |
| Tariq and Chatzinotas, DOI 10.1038/s41534-026-01308-w | 2026 | recent | CSI-tuned adaptive diversity coding for QuMIMO channels | Quasi-static dephasing on a chip; shot cost of obtaining the CSI |
| Elkouss et al., DOI 10.1038/s42005-026-02583-x | 2026 | recent | Finite sufficient conditions for catalytic state transformations | Any noise model; it motivates asking for finite, checkable criteria rather than a numerical method |
| Ammendola et al., DOI 10.1038/s41377-026-02236-2 | 2026 | recent | Free-space photonic processor with few layers | Qubit dephasing and encodings |
| McClean et al., DOI 10.1088/1367-2630/18/2/023023 | 2016 | landmark | Variational algorithms, error suppression on pre-threshold devices | Correlated noise between qubits; encoding choice |
| Streltsov et al., DOI 10.1103/RevModPhys.89.041003 | 2017 | landmark | Coherence as a resource; defines the quantity dephasing destroys | Hardware-level mitigation; layout-dependent decay |
| Blais et al., DOI 10.1103/RevModPhys.93.025005 | 2021 | landmark | Circuit quantum electrodynamics, dispersive readout, decoherence sources | Statistics of noise correlations across a chip |
| Bogaerts et al., DOI 10.1038/s41586-020-2764-0 | 2020 | review | Programmable photonic circuits and their calibration/control strategies | Qubit-register noise, DFS encodings |
| Elshaari et al., DOI 10.1038/s41566-020-0609-x | 2020 | review | Hybrid integration; monolithic platforms struggle to meet stringent quantum demands | Compile-time use of noise structure |
| Wang et al., DOI 10.1038/s41566-019-0532-1 | 2020 | review | Integrated photonic quantum technologies | Correlated dephasing of a register |
| Yahya et al., DOI 10.1016/j.aeue.2026.156647 | 2027 | adjacent | Gaussian-copula spatial correlation of antenna ports, aged-CSI robustness | Quantum fidelity; calibration measured in shots |
| Todorovic et al., DOI 10.1007/978-3-032-36445-6_10 | 2027 | adjacent | Capacity under a new turbulence model, sensitivity to model parameters | Whether the model family changes a decision |

## The gap

Recent work supplies a kernel family for correlated dephasing, and the adjacent literature shows that design under
correlated channels depends on how well the correlation is known. What is missing is a measurement of whether a
compile-time decision (which physical qubits form a DFS pair) depends on the kernel *shape*, on the kernel
*geometry*, or on measured data, and how many calibration shots the data route costs. Without that, a lab cannot tell
whether fitting a smooth kernel is enough or whether it must spend pair-by-pair Ramsey time.

**Novelty in one sentence.** This project turns a correlated-dephasing kernel template into an exactly solvable compile
decision (optimal DFS pairing) and measures how much of the oracle gain survives kernel misspecification, geometry
mismatch and finite-shot calibration.

## Falsifiable hypotheses and success metric

Metric: logical idle entanglement infidelity `1 - F_e`, and the *retained gain*
`G = (eps_random - eps_strategy) / (eps_random - eps_oracle)`, the share of the oracle's improvement over a
kernel-blind pairing that a strategy keeps (1 = oracle, 0 = no better than random pairing).

- **H1 (shape is irrelevant).** When the noise geometry equals the chip layout, a one-parameter fit of *any* of the
  four kernels retains `G >= 0.90` for *every* true kernel. Falsified if any of the 16 combinations falls below 0.90.
- **H2 (calibration beats geometry once geometry is wrong).** When the noise geometry is displaced from the layout by
  one qubit pitch, geometry-based compilers (any kernel) retain `G < 0.5`, while a nonparametric estimate from
  300 Ramsey shots per pair retains `G >= 0.8`. Falsified if either bound fails.
- **H3 (when DFS pays).** The oracle DFS beats the best unencoded qubits only above a noise correlation length of
  about 1.5 qubit pitches; calibrated compilers need a larger length. Reported as a measured crossover, not asserted.

Design note: H1 and H2 were written before the first experiment run. After a first full run, which showed the
kernel-shape panel only at a 0.5-pitch mismatch (all entries between 0.66 and 0.73), a mismatch-0 panel was added so
that the shape question is isolated from the geometry question. Both panels are reported.
