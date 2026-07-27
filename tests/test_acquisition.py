import numpy as np
import pytest
from scipy.stats import norm

import pypolar as plr
from pypolar.sampler import _observation_noise


@pytest.fixture(params=[plr.ConjugateGP, plr.BoTorchGP],
                ids=["conjugate", "botorch"])
def backend(request):
    """Each regression backend, so every acquisition is exercised on both."""
    return request.param


@pytest.fixture
def gp(backend):
    """A GP fit to a handful of noisy samples of a 1D sinusoid."""
    rng = np.random.default_rng(0)
    actions = np.linspace(0, 4, 60).reshape(-1, 1)
    idx = np.array([3, 17, 28, 41, 55])
    y = np.sin(actions[idx].ravel()) + 0.05 * rng.standard_normal(idx.shape[0])

    model = backend(signal_variance=1.0, length_scale=0.8, rng=rng)
    model.set_data(actions, idx, y, precision=50.0)
    model.fit()
    return model


@pytest.fixture
def unfit_gp(backend):
    rng = np.random.default_rng(1)
    actions = np.linspace(0, 4, 60).reshape(-1, 1)
    return backend(signal_variance=1.0, length_scale=0.8, rng=rng), actions


class TestExpectedMaxOfLines:

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_monte_carlo(self, seed):
        rng = np.random.default_rng(seed)
        n = rng.integers(1, 10)
        a = 2 * rng.normal(size=n)
        b = rng.normal(size=n)

        z = rng.standard_normal(400_000)
        mc = np.max(a[:, None] + b[:, None] * z[None, :], axis=0).mean()

        assert plr.expected_max_of_lines(a, b) == pytest.approx(mc, abs=0.02)

    def test_zero_slopes(self):
        a = np.array([1.0, 3.0, 2.0])
        assert plr.expected_max_of_lines(a, np.zeros(3)) == pytest.approx(3.0)

    def test_single_line(self):
        got = plr.expected_max_of_lines(np.array([2.0]), np.array([5.0]))
        assert got == pytest.approx(2.0)

    def test_duplicate_slopes(self):
        a = np.array([1.0, 3.0, 0.0])
        b = np.array([2.0, 2.0, 2.0])
        assert plr.expected_max_of_lines(a, b) == pytest.approx(3.0)


class TestKnowledgeGradient:

    def test_non_negative(self, gp):
        idx = np.arange(gp.actions.shape[0])
        cov = gp.posterior_cov_cross(idx)
        denom = np.sqrt(gp.sigma2 + np.diag(cov))
        values = [
            plr.knowledge_gradient(gp.mu, cov[:, j] / denom[j])
            for j in range(idx.shape[0])
        ]
        assert np.all(np.array(values) >= 0)

    def test_zero_for_no_information(self, gp):
        assert plr.knowledge_gradient(gp.mu, np.zeros_like(gp.mu)) == 0.0


class TestPosteriorCovCross:

    def test_matches_full_posterior(self, gp):
        idx = np.array([0, 7, 30, 59])
        expected = gp.posterior_cov()[:, idx]
        np.testing.assert_allclose(
            gp.posterior_cov_cross(idx), expected, rtol=1e-8, atol=1e-10
        )

    def test_matches_full_posterior_without_feedback(self, unfit_gp):
        model, actions = unfit_gp
        model.set_data(actions, np.array([], dtype=int), np.array([]), precision=1.0)
        idx = np.array([2, 40])
        np.testing.assert_allclose(
            model.posterior_cov_cross(idx),
            model.posterior_cov()[:, idx],
            rtol=1e-8,
            atol=1e-10,
        )


class TestExpectedImprovementSampler:

    def test_matches_closed_form(self, gp):
        sampler = plr.ExpectedImprovementSampler(gp, np.random.default_rng(0), xi=0.1)
        mu, std = gp.mu, gp.std()
        z = (mu - mu.max() - 0.1) / std
        expected = std * (z * norm.cdf(z) + norm.pdf(z))

        got = sampler.acquisition(gp.actions)
        np.testing.assert_allclose(got, expected, rtol=1e-10)
        assert np.all(got >= 0)

    def test_zero_where_std_is_zero(self, gp):
        sampler = plr.ExpectedImprovementSampler(gp, np.random.default_rng(0))
        std = gp.std()
        std[:3] = 0.0
        gp.std = lambda r=None: std

        assert np.all(sampler.acquisition(gp.actions)[:3] == 0.0)


class TestMaxValueEntropySampler:

    def test_finite_and_non_negative(self, gp):
        sampler = plr.MaxValueEntropySampler(gp, np.random.default_rng(0), num_maxima=8)
        sampler.update_posterior()

        scores = sampler.acquisition(gp.actions)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)


@pytest.mark.parametrize(
    "make_sampler",
    [
        lambda gp, rng: plr.ExpectedImprovementSampler(gp, rng),
        lambda gp, rng: plr.KnowledgeGradientSampler(gp, rng),
        lambda gp, rng: plr.KnowledgeGradientSampler(gp, rng, num_candidates=5),
        lambda gp, rng: plr.MaxValueEntropySampler(gp, rng, num_maxima=4),
    ],
)
class TestSampling:

    def test_returns_an_action(self, gp, make_sampler):
        sampler = make_sampler(gp, np.random.default_rng(0))
        sampler.n_warmup = 0
        sampler.update_posterior()

        action = sampler.sample(gp.actions)
        assert action.shape == (gp.actions.shape[1],)
        assert np.any(np.all(gp.actions == action, axis=1))

    def test_falls_back_before_fit(self, unfit_gp, make_sampler):
        model, actions = unfit_gp
        sampler = make_sampler(model, np.random.default_rng(0))

        action = sampler.sample(actions)
        assert np.any(np.all(actions == action, axis=1))


class TestQNEHVISampler:

    @pytest.fixture
    def gps(self):
        """Two BoTorchGPs over a shared action space, fit to opposing objectives."""
        rng = np.random.default_rng(0)
        actions = np.linspace(0, 4, 40).reshape(-1, 1)
        idx = np.array([3, 11, 20, 29, 37])
        models = []
        for sign in (1.0, -1.0):
            model = plr.BoTorchGP(signal_variance=1.0, length_scale=0.8, rng=rng)
            model.set_data(actions, idx, sign * np.sin(actions[idx].ravel()),
                           precision=50.0)
            model.fit()
            models.append(model)
        return models, actions

    def test_returns_an_action(self, gps):
        models, actions = gps
        sampler = plr.QNEHVISampler(models, np.random.default_rng(0),
                                    num_samples=16, n_warmup=0)
        sampler.update_posterior()

        action = sampler.sample(actions)
        assert action.shape == (actions.shape[1],)
        assert np.any(np.all(actions == action, axis=1))

    def test_falls_back_before_fit(self, gps):
        _, actions = gps
        unfit = [plr.BoTorchGP(signal_variance=1.0, length_scale=0.8)
                 for _ in range(2)]
        for model in unfit:
            model.set_data(actions, np.array([], dtype=int), np.array([]),
                           precision=1.0)
        sampler = plr.QNEHVISampler(unfit, np.random.default_rng(0), num_samples=16)
        sampler.update_posterior()

        action = sampler.sample(actions)
        assert np.any(np.all(actions == action, axis=1))

    def test_explicit_reference_point_is_used(self, gps):
        models, actions = gps
        sampler = plr.QNEHVISampler(models, np.random.default_rng(0),
                                    ref_point=[-2.0, -2.0], num_samples=16)
        assert sampler._reference_point() == [-2.0, -2.0]


class TestObservationNoise:

    def test_defaults_to_the_gp(self, gp):
        assert _observation_noise(gp, None) == gp.sigma2

    def test_explicit_value_wins(self, gp):
        assert _observation_noise(gp, 0.25) == 0.25

    def test_raises_without_a_source(self):
        model = plr.LaplaceGP(signal_variance=1.0, length_scale=1.0)
        with pytest.raises(AttributeError, match="observation noise"):
            _observation_noise(model, None)
