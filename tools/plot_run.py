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
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=140)
    colors = ["#ee6ea0", "#67c8e2", "#8b94a1"]
    for i, (col, lbl) in enumerate(sch["series"]):
        ax.plot(x, h[:, col], color=colors[i % len(colors)], lw=1.8,
                label=lbl, marker="o", ms=2.5)
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel("survival (seconds)")
    ax.set_title(name, loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=.18, lw=.6)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
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
