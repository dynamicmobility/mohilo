"""PCA the scale parameters of one CSV to 2D and surface-plot cost over them."""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = [c for c in rows[0] if c not in ("trial_name", "cost")]
    X = np.array([[float(r[c]) for c in cols] for r in rows])
    cost = np.array([float(r["cost"]) for r in rows])
    return cols, X, cost


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    args = p.parse_args()

    cols, X, cost = load(args.csv)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(Z[:, 0], Z[:, 1], cost, cmap="viridis", alpha=0.8,
                    edgecolor="0.3", linewidth=0.3)
    ax.scatter(Z[:, 0], Z[:, 1], cost, color="0.15", s=14, depthshade=False)
    ax.set_xlabel(f"PC1 ({var[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.0%} var)")
    ax.set_zlabel("cost")
    ax.set_title(f"{args.csv.stem}  cost over 2D PCA of scale params")

    print(f"{args.csv.stem}: n={len(cost)}, {len(cols)} scale params")
    print(f"explained variance: PC1 {var[0]:.1%}, PC2 {var[1]:.1%}, "
          f"total {var.sum():.1%}")
    print("PC1 loadings:")
    for c, w in sorted(zip(cols, pca.components_[0]), key=lambda t: -abs(t[1])):
        print(f"  {c:<22}{w:+.3f}")
    print("PC2 loadings:")
    for c, w in sorted(zip(cols, pca.components_[1]), key=lambda t: -abs(t[1])):
        print(f"  {c:<22}{w:+.3f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
