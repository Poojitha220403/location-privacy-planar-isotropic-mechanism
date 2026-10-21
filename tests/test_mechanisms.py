import numpy as np
import pytest
from scipy import stats

from pim import LaplaceMechanism, PlanarIsotropicMechanism, sample_uniform, sensitivity_hull

# δ-location sets used across the tests
NOTEBOOK_SET = np.array([[1, 1], [2, 2], [3, 1], [2, 3], [1, 2], [2, 1]], dtype=float)
GRID_3X3 = np.array([[x, y] for x in (1, 2, 3) for y in (1, 2, 3)], dtype=float)  # paper §4.1 example
ROAD = np.array([[x, 0.2 * x + 0.1 * (x % 2)] for x in range(10)], dtype=float)  # long, thin set
SETS = {"notebook": NOTEBOOK_SET, "grid": GRID_3X3, "road": ROAD}


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


# ---------- sensitivity hull ----------

@pytest.mark.parametrize("name", SETS)
def test_sensitivity_hull_is_centrally_symmetric(name):
    """Theorem 4.1: K = Conv(ΔV) is symmetric about the origin."""
    k = sensitivity_hull(SETS[name])
    mirrored = {tuple(np.round(-v, 9)) for v in k.vertices}
    assert mirrored == {tuple(np.round(v, 9)) for v in k.vertices}


@pytest.mark.parametrize("name", SETS)
def test_every_pairwise_difference_has_k_norm_at_most_one(name):
    """The privacy guarantee rests on ||x1 − x2||_K <= 1 for all x1, x2 in ΔX."""
    pts = SETS[name]
    k = sensitivity_hull(pts)
    diffs = (pts[:, None, :] - pts[None, :, :]).reshape(-1, 2)
    norms = k.norm(diffs)
    assert norms.max() == pytest.approx(1.0, abs=1e-9)  # tight: hull vertices reach 1


def test_hull_of_3x3_grid_matches_paper_example():
    k = sensitivity_hull(GRID_3X3)
    assert {tuple(v) for v in k.vertices} == {(2, 2), (-2, 2), (2, -2), (-2, -2)}
    assert k.area == pytest.approx(16.0)


# ---------- sampling and isotropic position ----------

def test_uniform_sampler_stays_inside_and_is_uniform(rng):
    k = sensitivity_hull(NOTEBOOK_SET)
    pts = sample_uniform(k, 100_000, rng)
    assert k.norm(pts).max() <= 1 + 1e-12
    # For a uniform point in a 2-D convex body, P(||u||_K <= t) = t²
    assert stats.kstest(k.norm(pts), lambda t: np.clip(t, 0, 1) ** 2).pvalue > 1e-3
    assert np.abs(pts.mean(axis=0)).max() < 0.02  # centered (symmetric hull)


@pytest.mark.parametrize("name", SETS)
def test_isotropic_transform_whitens_the_hull(name, rng):
    m = PlanarIsotropicMechanism(SETS[name], epsilon=1.0, rng=rng)
    y = sample_uniform(m.isotropic_hull, 200_000, rng)
    np.testing.assert_allclose(y.T @ y / len(y), np.eye(2), atol=0.03)


# ---------- the mechanisms ----------

@pytest.mark.parametrize("epsilon", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("name", SETS)
def test_pim_noise_follows_the_k_norm_distribution(name, epsilon, rng):
    """K-norm mechanism in 2-D: density ∝ exp(−ε||z||_K), so ||z||_K ~ Gamma(2, 1/ε)."""
    m = PlanarIsotropicMechanism(SETS[name], epsilon=epsilon, rng=rng)
    norms = m.hull.norm(m.noise(50_000))
    assert stats.kstest(norms, stats.gamma(a=2, scale=1 / epsilon).cdf).pvalue > 1e-3


def test_pim_privacy_ratio_bound(rng):
    """For every output z and pair x1, x2 in ΔX: p(z|x1)/p(z|x2) <= e^ε."""
    eps = 1.0
    m = PlanarIsotropicMechanism(NOTEBOOK_SET, epsilon=eps, rng=rng)
    z = m.release(NOTEBOOK_SET[0], n=2_000)
    for x1 in NOTEBOOK_SET:
        for x2 in NOTEBOOK_SET:
            log_ratio = eps * (m.hull.norm(z - x2) - m.hull.norm(z - x1))
            assert log_ratio.max() <= eps + 1e-9


def test_pim_is_unbiased(rng):
    m = PlanarIsotropicMechanism(NOTEBOOK_SET, epsilon=1.0, rng=rng)
    z = m.release([2, 2], n=200_000)
    np.testing.assert_allclose(z.mean(axis=0), [2, 2], atol=0.03)


def test_laplace_uses_l1_sensitivity(rng):
    lap = LaplaceMechanism(NOTEBOOK_SET, epsilon=2.0, rng=rng)
    assert lap.l1_sensitivity == 3.0  # e.g. (1,1) → (2,3) or (3,1) → (2,3)
    assert lap.scale == 1.5
    noise = lap.noise(200_000)
    assert stats.kstest(noise[:, 0], stats.laplace(scale=1.5).cdf).pvalue > 1e-3


@pytest.mark.parametrize("cls", [PlanarIsotropicMechanism, LaplaceMechanism])
def test_rejects_non_positive_epsilon(cls):
    with pytest.raises(ValueError):
        cls(NOTEBOOK_SET, epsilon=0)


def test_pim_beats_laplace_on_elongated_sets(rng):
    """The paper's motivation: ℓ1 sensitivity over-perturbs when ΔX is not 'round'."""
    pim = PlanarIsotropicMechanism(ROAD, epsilon=1.0, rng=rng)
    lap = LaplaceMechanism(ROAD, epsilon=1.0, rng=rng)
    pim_err = np.linalg.norm(pim.noise(100_000), axis=1).mean()
    lap_err = np.linalg.norm(lap.noise(100_000), axis=1).mean()
    assert pim_err < 0.8 * lap_err  # measured ≈ 0.69 with this seed
