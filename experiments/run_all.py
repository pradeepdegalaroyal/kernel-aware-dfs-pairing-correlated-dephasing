"""Reproduce every result and figure.  ``python experiments/run_all.py [--quick]``."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corrdeph import plots, studies  # noqa: E402
from corrdeph.control import suppression_factor  # noqa: E402
from corrdeph.kernels import KERNELS  # noqa: E402
from corrdeph.validation import run_validation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (< 1 minute)")
    ap.add_argument("--out", type=Path, default=ROOT, help="folder holding results/ and figures/")
    args = ap.parse_args()
    q = args.quick
    res, figs = args.out / "results", args.out / "figures"
    res.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    seeds, layouts = (2, 2) if q else (5, 8)
    n_main = 8 if q else 10
    t0 = time.time()
    cfg = {"quick": q, "seeds": seeds, "layouts_per_seed": layouts, "sigma2": studies.SIGMA2_DEFAULT}

    print("validation ...", flush=True)
    val = run_validation(3000 if q else 20000, 100 if q else 1500)
    val.to_csv(res / "validation.csv", index=False)

    print("E1 strategies vs xi ...", flush=True)
    xis = [0.5, 1.0, 2.0, 4.0] if q else [0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.5, 7.0]
    main_df = studies.study_main(n_main, seeds, layouts, xis, [0.0, 1.0])
    main_df.to_csv(res / "e1_strategies_vs_xi.csv", index=False)
    plots.fig_strategies(main_df, figs / "fig1_strategies_vs_xi.png", 0.0, n_main)
    plots.fig_strategies(main_df, figs / "fig1b_strategies_vs_xi_mismatch1.png", 1.0, n_main)

    print("E2 kernel shape ...", flush=True)
    shape_df = studies.study_shape(8, seeds, layouts, xi=2.0, mismatches=[0.0, 0.5])
    shape_df.to_csv(res / "e2_kernel_shape.csv", index=False)
    plots.fig_shape(shape_df, figs / "fig2_kernel_misspecification.png", KERNELS)

    print("E3 calibration shots x mismatch ...", flush=True)
    shots = [0, 30, 300] if q else [0, 10, 30, 100, 300, 1000, 10000]
    mms = [0.0, 1.0] if q else [0.0, 0.5, 1.0, 2.0]
    cal_df = studies.study_calibration(8, seeds, layouts, shots, mms)
    cal_df.to_csv(res / "e3_calibration.csv", index=False)
    plots.fig_shots_mismatch(cal_df, figs / "fig3_calibration_heatmap.png")
    plots.fig_calibration(cal_df, figs / "fig3b_calibration_lines.png", 8)

    print("E4 phase diagram ...", flush=True)
    pxi = np.geomspace(0.4, 8.0, 5 if q else 14).tolist()
    ps2 = np.linspace(0.02, 0.6, 5 if q else 14).tolist()
    ph_or = studies.study_phase(8, seeds, layouts, pxi, ps2)
    ph_sh = studies.study_phase(8, seeds, layouts, pxi, ps2, mismatch=1.0, shots=300)
    ph_or.assign(compiler="oracle_mismatch0").to_csv(res / "e4_phase_oracle.csv", index=False)
    ph_sh.assign(compiler="shots300_mismatch1").to_csv(res / "e4_phase_shots.csv", index=False)
    plots.fig_phase([("Oracle pairing, noise follows layout", ph_or),
                     ("300-shot calibrated, mismatch 1 pitch", ph_sh)], figs / "fig4_break_even_phase.png", 8)

    print("E5 scaling ...", flush=True)
    ns = [6, 8] if q else [6, 8, 10, 12]
    sc_df = studies.study_scaling(ns, min(seeds, 3) if not q else 2, 3 if not q else 2)
    sc_df.to_csv(res / "e5_scaling.csv", index=False)
    plots.fig_scaling(sc_df, figs / "fig5_scaling.png")

    print("E6 ablation + dynamical decoupling ...", flush=True)
    abl = []
    for p in ([2] if q else [1, 2, 4, 8]):
        d = studies.study_ablation(8, seeds, layouts, p)
        d["pulses"] = p
        abl.append(d)
    import pandas as pd

    pd.concat(abl).to_csv(res / "e6_ablation.csv", index=False)
    cfg["cpmg_suppression"] = {str(p): suppression_factor(p) for p in ([2] if q else [1, 2, 4, 8, 16])}

    cfg["wall_seconds"] = round(time.time() - t0, 1)
    (res / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"done in {cfg['wall_seconds']} s")


if __name__ == "__main__":
    main()
