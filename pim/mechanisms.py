"""Planar Isotropic Mechanism (PIM) and the Laplace baseline.

Follows Algorithm 2 of Xiao & Xiong, "Protecting Locations with Differential
Privacy under Temporal Correlations", ACM CCS 2015.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class SensitivityHull:
    """Convex polygon K = Conv(ΔV), stored by its counter-clockwise vertices."""

    vertices: np.ndarray  # (h, 2)
    equations: np.ndarray  # (h, 3) half-planes a·x + c <= 0 from scipy

    @property
    def area(self) -> float:
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def norm(self, z: np.ndarray) -> np.ndarray:
        """Minkowski norm ||z||_K = min{t >= 0 : z ∈ tK}; K is centrally symmetric."""
        z = np.atleast_2d(z)
        a, c = self.equations[:, :2], self.equations[:, 2]
        # Each facet a·x <= -c (with -c > 0 because the origin is interior)
        return np.max((z @ a.T) / (-c), axis=1)

    def transformed(self, matrix: np.ndarray) -> "SensitivityHull":
        return _hull_from_points(self.vertices @ matrix.T)


def _hull_from_points(points: np.ndarray) -> SensitivityHull:
    hull = ConvexHull(points)
    return SensitivityHull(vertices=points[hull.vertices], equations=hull.equations)


def sensitivity_hull(delta_x: np.ndarray) -> SensitivityHull:
    """Lines 1–3: K' = Conv(ΔX); ΔV = {v_i − v_j} over vertices of K'; K = Conv(ΔV)."""
    delta_x = np.asarray(delta_x, dtype=float)
    k_prime = ConvexHull(delta_x)
    v = delta_x[k_prime.vertices]
    diffs = (v[:, None, :] - v[None, :, :]).reshape(-1, 2)
    return _hull_from_points(diffs)


def sample_uniform(hull: SensitivityHull, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform points in a convex polygon via a fan triangulation from its centroid."""
    v = hull.vertices
    center = v.mean(axis=0)
    a, b = v, np.roll(v, -1, axis=0)
    tri_areas = 0.5 * np.abs((a[:, 0] - center[0]) * (b[:, 1] - center[1])
                             - (a[:, 1] - center[1]) * (b[:, 0] - center[0]))
    idx = rng.choice(len(v), size=n, p=tri_areas / tri_areas.sum())
    u, w = rng.random(n), rng.random(n)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    return center + u[:, None] * (a[idx] - center) + w[:, None] * (b[idx] - center)


def isotropic_transform(hull: SensitivityHull, rng: np.random.Generator,
                        l: int = 20_000, tol: float = 1e-3, max_rounds: int = 8) -> np.ndarray:
    """Lines 4–6: T = ((1/l) Σ y yᵀ)^(-1/2), resampling with larger l until T is stable."""
    def estimate(size):
        y = sample_uniform(hull, size, rng)
        cov = y.T @ y / size
        evals, evecs = np.linalg.eigh(cov)
        return evecs @ np.diag(evals ** -0.5) @ evecs.T

    t = estimate(l)
    for _ in range(max_rounds):
        t_next = estimate(l)
        if np.linalg.norm(t_next - t, "fro") < tol * np.linalg.norm(t, "fro"):
            return t_next
        t, l = t_next, l * 2
    return t


class PlanarIsotropicMechanism:
    """ε-differentially private location release on a δ-location set (Algorithm 2)."""

    def __init__(self, delta_x: np.ndarray, epsilon: float, rng: np.random.Generator | None = None):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)
        self.rng = rng or np.random.default_rng()
        self.hull = sensitivity_hull(delta_x)                      # K
        self.T = isotropic_transform(self.hull, self.rng)           # line 6
        self.T_inv = np.linalg.inv(self.T)
        self.isotropic_hull = self.hull.transformed(self.T)         # line 7: K_I = TK

    def noise(self, n: int = 1) -> np.ndarray:
        z0 = sample_uniform(self.isotropic_hull, n, self.rng)       # line 8
        r = self.rng.gamma(shape=3.0, scale=1.0 / self.epsilon, size=n)  # line 9: Γ(3, 1/ε)
        return (r[:, None] * z0) @ self.T_inv.T                     # line 10: r·T⁻¹z'

    def release(self, true_location, n: int = 1) -> np.ndarray:
        return np.asarray(true_location, dtype=float) + self.noise(n)


class LaplaceMechanism:
    """Baseline from the paper: Laplace noise calibrated to the ℓ1 sensitivity of ΔX."""

    def __init__(self, delta_x: np.ndarray, epsilon: float, rng: np.random.Generator | None = None):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        delta_x = np.asarray(delta_x, dtype=float)
        self.epsilon = float(epsilon)
        self.rng = rng or np.random.default_rng()
        self.l1_sensitivity = float(np.max(np.abs(delta_x[:, None, :] - delta_x[None, :, :]).sum(axis=2)))
        self.scale = self.l1_sensitivity / self.epsilon

    def noise(self, n: int = 1) -> np.ndarray:
        return self.rng.laplace(0.0, self.scale, size=(n, 2))

    def release(self, true_location, n: int = 1) -> np.ndarray:
        return np.asarray(true_location, dtype=float) + self.noise(n)
