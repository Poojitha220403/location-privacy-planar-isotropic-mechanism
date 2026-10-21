"""Monte Carlo comparison of PIM and the Laplace baseline.

For each δ-location set and privacy budget ε, both mechanisms release the true
location N times; we report the mean and median Euclidean error. Writes
results/results.csv, results/error_vs_epsilon.png and results/noise_clouds.png.

Usage: python experiments/evaluate.py [--trials 20000] [--seed 7]
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pim import LaplaceMechanism, PlanarIsotropicMechanism  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DELTA_SETS = {
    # the 6-cell set from the original course notebook
    "notebook (6 cells)": np.array([[1, 1], [2, 2], [3, 1], [2, 3], [1, 2], [2, 1]], float),
    # the 3×3 grid used as the paper's Section 4 example
    "3x3 grid": np.array([[x, y] for x in (1, 2, 3) for y in (1, 2, 3)], float),
    # a user moving along a road: long and thin, where ℓ1 sensitivity over-perturbs
    "road (10 cells)": np.array([[x, 0.2 * x + 0.1 * (x % 2)] for x in range(10)], float),
    # a diagonal corridor: elongated and rotated 45°
    "diagonal (8 cells)": np.array([[i, i + (i % 2) * 0.5] for i in range(8)], float),
}
EPSILONS = [0.5, 1.0, 2.0]


def run(trials: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for name, dx in DELTA_SETS.items():
        true_loc = dx[len(dx) // 2]
        for eps in EPSILONS:
            for mech_name, cls in [("PIM", PlanarIsotropicMechanism), ("Laplace", LaplaceMechanism)]:
                mech = cls(dx, eps, rng)
                err = np.linalg.norm(mech.release(true_loc, n=trials) - true_loc, axis=1)
                rows.append({"delta_set": name, "epsilon": eps, "mechanism": mech_name,
                             "mean_error": round(float(err.mean()), 4),
                             "median_error": round(float(np.median(err)), 4),
                             "p90_error": round(float(np.quantile(err, 0.9)), 4)})
    return rows


def plot_errors(rows):
    fig, axes = plt.subplots(1, len(DELTA_SETS), figsize=(4 * len(DELTA_SETS), 3.6), sharey=False)
    for ax, name in zip(axes, DELTA_SETS):
        for mech, style in [("PIM", "o-"), ("Laplace", "s--")]:
            pts = [(r["epsilon"], r["mean_error"]) for r in rows
                   if r["delta_set"] == name and r["mechanism"] == mech]
            ax.plot(*zip(*pts), style, label=mech, linewidth=2)
        ax.set_title(name)
        ax.set_xlabel("ε")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean Euclidean error")
    axes[0].legend()
    fig.suptitle("Location error at equal privacy (lower is better)")
    fig.tight_layout()
    fig.savefig(RESULTS / "error_vs_epsilon.png", dpi=150)


def plot_clouds(seed: int):
    rng = np.random.default_rng(seed)
    dx = DELTA_SETS["road (10 cells)"]
    true_loc = dx[len(dx) // 2]
    pim = PlanarIsotropicMechanism(dx, 1.0, rng)
    lap = LaplaceMechanism(dx, 1.0, rng)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, mech, title in [(axes[0], pim, "PIM"), (axes[1], lap, "Laplace (ℓ1 sensitivity)")]:
        z = mech.release(true_loc, n=3000)
        ax.scatter(z[:, 0], z[:, 1], s=2, alpha=0.25)
        ax.scatter(dx[:, 0], dx[:, 1], c="black", s=18, label="δ-location set")
        ax.scatter(*true_loc, c="red", s=60, marker="*", label="true location")
        if mech is pim:
            k = np.vstack([pim.hull.vertices, pim.hull.vertices[:1]]) + true_loc
            ax.plot(k[:, 0], k[:, 1], "k-", linewidth=1, label="sensitivity hull K")
        ax.set_title(f"{title}, ε = 1")
        ax.set_xlim(-25, 35)
        ax.set_ylim(-25, 25)
        ax.set_aspect("equal")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("3,000 releases of the same location on a road-shaped δ-location set")
    fig.tight_layout()
    fig.savefig(RESULTS / "noise_clouds.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    rows = run(args.trials, args.seed)
    with open(RESULTS / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_errors(rows)
    plot_clouds(args.seed)

    print(f"{'δ-location set':<20} {'ε':>4} {'PIM':>8} {'Laplace':>8} {'PIM/Lap':>8}")
    for name in DELTA_SETS:
        for eps in EPSILONS:
            p, lap = (next(r["mean_error"] for r in rows if r["delta_set"] == name
                           and r["epsilon"] == eps and r["mechanism"] == m) for m in ("PIM", "Laplace"))
            print(f"{name:<20} {eps:>4} {p:>8.3f} {lap:>8.3f} {p / lap:>8.2f}")


if __name__ == "__main__":
    main()
