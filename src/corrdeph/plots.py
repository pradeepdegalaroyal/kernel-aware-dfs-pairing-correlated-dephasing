"""Figures: shaded-band lines, an annotated heatmap, and a filled-contour phase diagram."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

TAB = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#9c755f", "#bab0ac"]
MUTED = LinearSegmentedColormap.from_list("muted_blue", ["#f4f6f9", "#a9bcd6", "#4e79a7", "#2b4468"])
DPI = 200


def _style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})


def _logx(ax, ticks) -> None:
    """Log x-axis with plain-number ticks (no overlapping scientific labels)."""
    ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()


def _band(ax, x, df: pd.DataFrame, col: str, label: str, color: str, ls: str, marker: str) -> None:
    g = df.groupby(x)[col].agg(["mean", "std"]).reset_index()
    ax.plot(g[x], g["mean"], ls=ls, marker=marker, color=color, label=label, lw=1.8, ms=5)
    ax.fill_between(g[x], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=color, alpha=0.18, lw=0)


def fig_strategies(df: pd.DataFrame, path: Path, mismatch: float, n: int) -> None:
    """Infidelity vs. correlation length for every compile strategy (line plot, mean +- std over seeds)."""
    _style()
    d = df[np.isclose(df["mismatch"], mismatch)]
    spec = [("eps_bare_best", "Bare qubits, best subset", TAB[8], "-", "s"),
            ("eps_dfs_random", "DFS, kernel-blind pairing (mean)", TAB[2], "--", "v"),
            ("eps_dfs_geometry", "DFS, nearest-neighbour geometry", TAB[1], "-.", "^"),
            ("eps_dfs_shots", "DFS, 300-shot calibrated", TAB[4], "-", "o"),
            ("eps_dfs_oracle", "DFS, oracle pairing", TAB[0], ":", "D")]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for col, lab, c, ls, mk in spec:
        _band(ax, "xi", d, col, lab, c, ls, mk)
    _logx(ax, [0.4, 0.7, 1, 1.5, 2, 3, 4.5, 7])
    ax.set_xlabel("Noise correlation length xi (qubit pitches)")
    ax.set_ylabel("Logical idle infidelity 1 - F_e")
    ax.set_title(f"Pairing strategies, {n} physical / {n // 2} logical qubits, mismatch = {mismatch:g} pitch")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_calibration(df: pd.DataFrame, path: Path, n: int) -> None:
    """Retained gain vs. calibration shots per pair, one line per geometry mismatch."""
    _style()
    d = df[df["shots"] > 0]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    mms = sorted(d["mismatch"].unique())
    for i, mm in enumerate(mms):
        sub = d[np.isclose(d["mismatch"], mm)]
        _band(ax, "shots", sub, "retained_shots", f"Shot-calibrated, mismatch {mm:g}", TAB[i], "-", "o")
        geo = df[np.isclose(df["mismatch"], mm)]["retained_geometry"].mean()
        ax.axhline(geo, color=TAB[i], ls=":", lw=1.2)
    ax.axhline(1.0, color="k", lw=0.8)
    _logx(ax, [10, 30, 100, 300, 1000, 10000])
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Ramsey shots per qubit pair")
    ax.set_ylabel("Share of oracle gain retained")
    ax.set_title(f"Cost of calibration ({n} qubits); dotted = geometry-only, no data")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _annotated_heatmap(ax, mat: np.ndarray, xt: list, yt: list, fmt: str = "{:.2f}", vmin=None, vmax=None):
    im = ax.imshow(mat, cmap=MUTED, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    ax.set_xticks(range(len(xt)), xt)
    ax.set_yticks(range(len(yt)), yt)
    ax.grid(False)
    lo, hi = np.nanmin(mat) if vmin is None else vmin, np.nanmax(mat) if vmax is None else vmax
    for (i, j), v in np.ndenumerate(mat):
        if np.isnan(v):
            continue
        dark = (v - lo) / max(hi - lo, 1e-9) > 0.55
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5, color="white" if dark else "black")
    return im


def fig_shape(df: pd.DataFrame, path: Path, kernels: tuple[str, ...]) -> None:
    """Heatmaps: retained gain for (true kernel, assumed kernel), one panel per geometry mismatch."""
    _style()
    pretty = {"exponential": "exponential", "gaussian": "Gaussian", "matern32": "Matern 3/2", "powerlaw": "power law"}
    names = [pretty[k] for k in kernels]
    mms = sorted(df["mismatch"].unique())
    fig, axes = plt.subplots(1, len(mms), figsize=(5.0 * len(mms) + 1.2, 4.6), squeeze=False,
                             sharey=True, layout="constrained")
    im = None
    for ax, mm in zip(axes[0], mms):
        sub = df[np.isclose(df["mismatch"], mm)]
        g = sub.groupby(["true_kernel", "assumed_kernel"])["retained_gain"].mean().unstack().loc[list(kernels), list(kernels)]
        im = _annotated_heatmap(ax, g.values, names, names, vmin=0.0, vmax=1.0)
        ax.set_xlabel("Kernel assumed by the compiler (length fitted)")
        ax.set_title(f"Geometry mismatch = {mm:g} pitch", fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    axes[0][0].set_ylabel("True noise kernel")
    fig.suptitle("Share of oracle DFS gain kept under kernel misspecification", fontsize=11)
    fig.colorbar(im, ax=axes[0].tolist(), label="Retained gain")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_shots_mismatch(df: pd.DataFrame, path: Path) -> None:
    """Heatmap: retained gain over (shots per pair, geometry mismatch); row 0 shots = geometry only."""
    _style()
    g = df.groupby(["mismatch", "shots"])["retained_shots"].mean().unstack()
    geo = df.groupby("mismatch")["retained_geometry"].mean()
    mat = np.column_stack([geo.values, g[[c for c in g.columns if c > 0]].values])
    xt = ["geometry\nonly"] + [str(c) for c in g.columns if c > 0]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    im = _annotated_heatmap(ax, mat, xt, [f"{m:g}" for m in g.index], vmin=0.0, vmax=1.0)
    ax.set_xlabel("Ramsey shots per qubit pair")
    ax.set_ylabel("Noise-geometry mismatch (pitches)")
    ax.set_title("Oracle gain retained: geometry-only vs shot-calibrated compiler")
    fig.colorbar(im, ax=ax, label="Retained gain")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_phase(panels: list[tuple[str, pd.DataFrame]], path: Path, n: int) -> None:
    """Filled contours: log10 break-even CNOT error where DFS stops paying off, one panel per compiler."""
    _style()
    levels = np.linspace(-4, -1.5, 11)
    marks = {np.log10(1e-3): "0.1 %", np.log10(3e-3): "0.3 %", np.log10(1e-2): "1 %"}
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.5), squeeze=False, sharey=True)
    cf = None
    for ax, (title, df) in zip(axes[0], panels):
        g = df.groupby(["sigma2", "xi"])["p_break_even"].mean().unstack()
        xi, s2, z = g.columns.values, g.index.values, g.values
        zl = np.log10(np.clip(z, 1e-9, None))
        cf = ax.contourf(xi, s2, zl, levels=levels, cmap=MUTED, extend="both")
        cs = ax.contour(xi, s2, zl, levels=list(marks), colors="k", linewidths=1.0)
        ax.clabel(cs, fmt=marks, fontsize=8)
        if z.min() < 0 < z.max():
            ax.contour(xi, s2, z, levels=[0.0], colors=TAB[2], linewidths=1.8, linestyles="--")
        _logx(ax, [0.5, 1, 2, 4, 8])
        ax.set_xlabel("Noise correlation length xi (qubit pitches)")
        ax.set_title(title, fontsize=10)
    axes[0][0].set_ylabel("Single-qubit phase variance sigma^2 (rad^2)")
    fig.suptitle(f"Break-even CNOT error of DFS vs bare qubits ({n} physical qubits); red dashed: p* = 0", fontsize=10.5)
    fig.colorbar(cf, ax=axes[0].tolist(), label="log10 of break-even two-qubit error p*")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_scaling(df: pd.DataFrame, path: Path) -> None:
    """Retained gain vs. register size for the two realistic compilers."""
    _style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    _band(ax, "n_physical", df, "retained_shots", "Shot-calibrated (300 shots/pair)", TAB[4], "-", "o")
    _band(ax, "n_physical", df, "retained_geometry", "Geometry-only", TAB[1], "--", "^")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("Physical qubits (logical = half)")
    ax.set_ylabel("Share of oracle gain retained")
    ax.set_title("Scaling with register size (mismatch 1 pitch)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
