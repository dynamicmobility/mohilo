"""Visualize per-trial parameter and cost traces in human_data/."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = [c for c in rows[0] if c != "trial_name"]
    return cols, {c: np.array([float(r[c]) for r in rows]) for c in cols}


def stats_table(data, columns, subjects):
    width = max(len(c) for c in columns) + 2
    for s in subjects:
        print(f"\n{s}  (n={len(next(iter(data[s].values())))})")
        print(f"  {'column'.ljust(width)}{'mean':>12}{'std':>12}")
        for c in columns:
            v = data[s][c]
            print(f"  {c.ljust(width)}{v.mean():>12.4f}{v.std(ddof=1):>12.4f}")
    print(f"\npooled across subjects")
    print(f"  {'column'.ljust(width)}{'mean':>12}{'std':>12}")
    for c in columns:
        v = np.concatenate([data[s][c] for s in subjects])
        print(f"  {c.ljust(width)}{v.mean():>12.4f}{v.std(ddof=1):>12.4f}")


def plot_grid(data, columns, subjects, args):
    ncols = min(args.ncols, len(columns))
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=args.figsize, dpi=args.dpi, squeeze=False, sharex=True
    )
    axes = axes.ravel()

    for ax, col in zip(axes, columns):
        for i, s in enumerate(subjects):
            y = data[s][col]
            x = np.arange(1, len(y) + 1)
            ax.plot(
                x,
                y,
                color=SERIES[i % len(SERIES)],
                lw=args.linewidth,
                marker="o" if args.markers else None,
                ms=3,
                label=s,
            )
        m = np.mean([data[s][col] for s in subjects], axis=0)
        ax.plot(np.arange(1, len(m) + 1), m, color="0.25", lw=args.linewidth + 0.6,
                ls="--", label="mean")
        ax.set_title(col, fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in axes[len(columns):]:
        ax.set_visible(False)
    for c in range(ncols):
        bottom = max(i for i in range(len(columns)) if i % ncols == c)
        axes[bottom].set_xlabel("trial", fontsize=8)
        axes[bottom].tick_params(labelbottom=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9)
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    return fig


def plot_box(data, columns, subjects, args):
    ncols = min(args.ncols, len(columns))
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=args.figsize, dpi=args.dpi,
                             squeeze=False)
    axes = axes.ravel()

    for ax, col in zip(axes, columns):
        vals = [data[s][col] for s in subjects]
        bp = ax.boxplot(vals, tick_labels=subjects, patch_artist=True, widths=0.55,
                        medianprops=dict(color="0.15", lw=1.6),
                        whiskerprops=dict(color="0.35", lw=1.0),
                        capprops=dict(color="0.35", lw=1.0),
                        flierprops=dict(marker="o", ms=4, mfc="none", mec="0.45"))
        for i, box in enumerate(bp["boxes"]):
            box.set(facecolor=SERIES[i % len(SERIES)], alpha=0.35,
                    edgecolor=SERIES[i % len(SERIES)], lw=1.4)
        for i, v in enumerate(vals):
            ax.plot(i + 1, v.mean(), marker="D", ms=5, color="0.15", zorder=3,
                    label="mean" if i == 0 else None)
            ax.annotate(f"{v.mean():.2f} ± {v.std(ddof=1):.2f}", (i + 1, v.mean()),
                        textcoords="offset points", xytext=(9, 0), fontsize=8,
                        color="0.3", va="center")
        ax.set_title(col, fontsize=10)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in axes[len(columns):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, fontsize=9)
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("human_data"))
    p.add_argument("--subjects", nargs="+", help="file stems, default all")
    p.add_argument("--columns", nargs="+", help="columns to plot, default all")
    p.add_argument("--kind", choices=["lines", "box"], default="lines")
    p.add_argument("--title", default="human_data")
    p.add_argument("--figsize", type=float, nargs=2, default=(12.0, 7.0))
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--ncols", type=int, default=3)
    p.add_argument("--linewidth", type=float, default=1.4)
    p.add_argument("--markers", action="store_true")
    p.add_argument("--save", type=Path)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    files = sorted(args.data_dir.glob("*.csv"))
    if args.subjects:
        files = [f for f in files if f.stem in args.subjects]
    if not files:
        raise SystemExit(f"no csv files matched in {args.data_dir}")

    data, all_cols = {}, None
    for f in files:
        cols, d = load(f)
        data[f.stem] = d
        all_cols = cols if all_cols is None else [c for c in all_cols if c in cols]

    columns = args.columns or all_cols
    missing = [c for c in columns if c not in all_cols]
    if missing:
        raise SystemExit(f"unknown columns {missing}; available: {all_cols}")

    subjects = list(data)
    stats_table(data, columns, subjects)
    fig = (plot_box if args.kind == "box" else plot_grid)(data, columns, subjects, args)

    if args.save:
        fig.savefig(args.save, bbox_inches="tight")
        print(f"\nsaved {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
