"""Plot a training run's survival curve for the docs Experiment log.

    python tools/plot_run.py runs_sim/fight_letty
    python tools/plot_run.py runs_sim/ppo_v29 --out docs/assets/curves

Reads <run_dir>/history.npy (schema depends on which trainer wrote it) and
writes <out>/<run_name>.png plus a copy of history.npy to runs_meta/ so the
plot is reproducible from the repo. .pt weights stay out of git.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# column layouts, keyed by width of history.npy
#   fight_*  (train_fight.py):  steps, med, mean, >60s
#   ppo v17  (train.py, 12):    wall, steps, mean, dec, ent, med, p90, f60, f120, f180, wallf, enemyf
#   ppo v28  (train.py, 16):    ...v17..., med1, p90_1, f60_1, mean1
SCHEMAS = {
    4:  {"x": 0, "series": [(1, "median"), (2, "mean")]},
    5:  {"x": 1, "series": [(2, "mean survival")]},
    12: {"x": 1, "series": [(5, "greedy median"), (2, "mean")]},
    16: {"x": 1, "series": [(12, "median (honest)"), (5, "greedy median"),
                            (2, "mean")]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/assets/curves")
    ap.add_argument("--no-meta", action="store_true",
                    help="don't copy history.npy into runs_meta/")
    args = ap.parse_args()

    run = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    name = run.name
    h = np.load(run / "history.npy")
    if h.ndim != 2:
        h = h.reshape(len(h), -1)
    sch = SCHEMAS.get(h.shape[1])
    if sch is None:
        raise SystemExit(f"unknown history width {h.shape[1]} for {name}; "
                         f"add it to SCHEMAS")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = h[:, sch["x"]] / 1e6
    rt_path = run / "realtransfer.npy"
    has_rt = rt_path.exists()

    # sim and real go in stacked panels - real transfer (~200-600s) dwarfs sim
    # survival (~20-40s) on a shared axis.
    nrows = 2 if has_rt else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(7.4, 2.4 * nrows + 0.4),
                             dpi=140, sharex=True,
                             gridspec_kw={"hspace": 0.12})
    axes = np.atleast_1d(axes)
    ax_sim = axes[0]

    colors = ["#ee6ea0", "#67c8e2", "#8b94a1"]
    for i, (col, lbl) in enumerate(sch["series"]):
        ax_sim.plot(x, h[:, col], color=colors[i % len(colors)], lw=1.7,
                    label=lbl, marker="o", ms=2.5)
    ax_sim.set_ylabel("sim survival (s)")
    ax_sim.set_title(name, loc="left", fontsize=11, fontweight="bold")
    ax_sim.legend(frameon=False, fontsize=8.5)

    if has_rt:
        rt = np.load(rt_path)                 # [wall, step, survival_s, score, flag]
        rs, rv = rt[:, 1] / 1e6, rt[:, 2]
        ax_real = axes[1]
        ax_real.scatter(rs, rv, s=10, color="#c9a227", alpha=.30, lw=0,
                        label="each eval", zorder=2)
        edges = np.linspace(rs.min(), rs.max(), 12)
        mids = 0.5 * (edges[:-1] + edges[1:])
        med = [np.median(rv[(rs >= a) & (rs < b)])
               if ((rs >= a) & (rs < b)).any() else np.nan
               for a, b in zip(edges[:-1], edges[1:])]
        ax_real.plot(mids, med, color="#c9a227", lw=2.3, marker="s", ms=3.5,
                     label="binned median", zorder=3)
        ax_real.set_ylabel("real survival (s)")
        ax_real.legend(frameon=False, fontsize=8.5)

    for ax in axes:
        ax.grid(True, alpha=.18, lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("training steps (millions)")
    fig.tight_layout()

    args.out.mkdir(parents=True, exist_ok=True)
    png = args.out / f"{name}.png"
    fig.savefig(png, bbox_inches="tight")
    print(f"wrote {png.relative_to(ROOT)}")

    if not args.no_meta:
        meta_dir = ROOT / "runs_meta"
        meta_dir.mkdir(exist_ok=True)
        shutil.copy(run / "history.npy", meta_dir / f"{name}.npy")
        print(f"wrote runs_meta/{name}.npy")


if __name__ == "__main__":
    main()
