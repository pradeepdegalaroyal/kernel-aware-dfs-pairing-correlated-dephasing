"""Print mean +- std (over seeds) tables from results/ for the README.  ``python experiments/summarise.py``."""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"


def ms(x: pd.Series, d: int = 3) -> str:
    return f"{x.mean():.{d}f} ± {x.std():.{d}f}"


def main() -> None:
    v = pd.read_csv(R / "validation.csv")
    print("VALIDATION\n", v.round(4).to_string(), "\n")

    e1 = pd.read_csv(R / "e1_strategies_vs_xi.csv")
    for mm in (0.0, 1.0):
        print(f"E1 mismatch={mm}: infidelity mean ± std")
        d = e1[e1.mismatch == mm]
        cols = ["eps_bare_best", "eps_dfs_random", "eps_dfs_geometry", "eps_dfs_shots", "eps_dfs_oracle"]
        print(d.groupby("xi")[cols].agg(ms).to_string(), "\n")

    e2 = pd.read_csv(R / "e2_kernel_shape.csv")
    print("E2 retained gain (min over assumed kernels, per true kernel & mismatch)")
    print(e2.groupby(["mismatch", "true_kernel", "assumed_kernel"]).retained_gain.mean().groupby(
        ["mismatch", "true_kernel"]).agg(["min", "max"]).round(3).to_string(), "\n")

    e3 = pd.read_csv(R / "e3_calibration.csv")
    print("E3 retained gain: shots / geometry / exp-fit")
    print(e3.groupby(["mismatch", "shots"])[["retained_shots", "retained_geometry", "retained_fit_exp"]].agg(ms).to_string(), "\n")

    for f in ("e4_phase_oracle.csv", "e4_phase_shots.csv"):
        p = pd.read_csv(R / f)
        g = p.groupby(["xi", "sigma2"]).p_break_even.mean().reset_index()
        pos = g[g.p_break_even > 0]
        print(f, "min xi with p*>0 per sigma2 (first/last):")
        mx = g.groupby("sigma2").apply(lambda t: t[t.p_break_even > 0].xi.min())
        print(mx.round(2).to_string())
        top = g[(g.sigma2 == g.sigma2.max())]
        print("p* at max sigma2:", top.assign(p=top.p_break_even.round(4))[["xi", "p"]].to_string(index=False), "\n")

    e5 = pd.read_csv(R / "e5_scaling.csv")
    cols = ["eps_bare_best", "eps_dfs_random", "eps_dfs_geometry", "eps_dfs_shots", "eps_dfs_oracle", "retained_shots", "retained_geometry"]
    print("E5 scaling\n", e5.groupby("n_physical")[cols].agg(ms).T.to_string(), "\n")

    e6 = pd.read_csv(R / "e6_ablation.csv")
    t = e6.groupby(["pulses", "dd", "strategy"]).eps.agg(ms).unstack()
    print("E6 ablation (infidelity)\n", t.to_string())
    print(e6.groupby("pulses").dd_factor.first().round(4).to_string())


if __name__ == "__main__":
    main()
