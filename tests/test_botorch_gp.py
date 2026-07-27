"""Behavioural tests for the BoTorch-backed regression GP.

``ConjugateGP`` is the reference: both backends compute the same exact posterior
for Gaussian feedback, so they must agree up to the one thing that genuinely
differs between them, ``GPModel.JITTER``.
"""

import numpy as np
import pytest

import pypolar as plr
from pypolar.optimization.gp.base import GPModel


ACTIONS = np.linspace(0, 4, 60).reshape(-1, 1)
IDX = np.array([3, 17, 28, 41, 55])
PRECISION = 50.0


def _feedback():
    rng = np.random.default_rng(0)
    return np.sin(ACTIONS[IDX].ravel()) + 0.05 * rng.standard_normal(IDX.shape[0])


def _fitted(cls, **kwargs):
    model = cls(signal_variance=1.0, length_scale=0.8,
                rng=np.random.default_rng(0), **kwargs)
    model.set_data(ACTIONS, IDX, _feedback(), precision=PRECISION)
    model.fit()
    return model


@pytest.fixture
def gp():
    return _fitted(plr.BoTorchGP)


@pytest.fixture
def empty_gp():
    model = plr.BoTorchGP(signal_variance=1.0, length_scale=0.8,
                          rng=np.random.default_rng(1))
    model.set_data(ACTIONS, np.array([], dtype=int), np.array([]), precision=1.0)
    model.fit()
    return model


class TestParityWithConjugate:
    """The two backends are the same model, so they must agree numerically."""

    def test_posterior_mean(self, gp):
        np.testing.assert_allclose(
            gp.mu, _fitted(plr.ConjugateGP).mu, rtol=1e-4, atol=1e-6
        )

    def test_posterior_std(self, gp):
        # Looser than the mean: JITTER is added to the prior variance, which
        # shifts the standard deviation by ~JITTER/(2*std) -- a relative effect
        # of about 1e-3 where the posterior is confident.
        np.testing.assert_allclose(
            gp.std(), _fitted(plr.ConjugateGP).std(), rtol=2e-3, atol=1e-6
        )

    def test_jitter_is_the_only_difference(self, monkeypatch):
        """Without the jitter the two posteriors coincide to solver precision."""
        monkeypatch.setattr(GPModel, "JITTER", 0.0)
        botorch, conjugate = _fitted(plr.BoTorchGP), _fitted(plr.ConjugateGP)

        np.testing.assert_allclose(botorch.mu, conjugate.mu, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            botorch.std(), conjugate.std(), rtol=1e-6, atol=1e-8
        )


class TestAgainstTheModelPosterior:
    """``mu`` and ``std()`` are evaluated in data space; ``posterior_cov()`` is
    read straight off the model, so it is an independent check on that algebra.
    """

    def test_mean_matches(self, gp):
        import torch

        with torch.no_grad():
            mean = gp.model.posterior(gp._grid_tensor()).mean
        np.testing.assert_allclose(
            gp.mu, mean.reshape(-1).numpy(), rtol=1e-10, atol=1e-12
        )

    def test_std_matches_the_covariance_diagonal(self, gp):
        np.testing.assert_allclose(
            gp.std(), np.sqrt(np.diag(gp.posterior_cov())), rtol=1e-8, atol=1e-10
        )


class TestPosteriorCovCross:

    def test_matches_full_posterior(self, gp):
        idx = np.array([0, 7, 30, 59])
        np.testing.assert_allclose(
            gp.posterior_cov_cross(idx), gp.posterior_cov()[:, idx],
            rtol=1e-8, atol=1e-10
        )

    def test_matches_full_posterior_without_feedback(self, empty_gp):
        idx = np.array([2, 40])
        np.testing.assert_allclose(
            empty_gp.posterior_cov_cross(idx), empty_gp.posterior_cov()[:, idx],
            rtol=1e-8, atol=1e-10
        )


class TestWithoutFeedback:

    def test_mean_is_zero(self, empty_gp):
        np.testing.assert_array_equal(empty_gp.mu, np.zeros(ACTIONS.shape[0]))

    def test_std_is_the_prior(self, empty_gp):
        np.testing.assert_allclose(
            empty_gp.std(), np.sqrt(empty_gp.prior_var()), rtol=1e-12
        )

    def test_sample_has_the_right_shape(self, empty_gp):
        sample = empty_gp.sample_posterior(np.random.default_rng(0))
        assert sample.shape == (ACTIONS.shape[0],)


class TestObservationNoise:

    @pytest.mark.parametrize("precision", [50.0, np.array([50.0])])
    def test_derives_from_precision(self, precision):
        """``sigma2 = 1/(2*lambda)``, as a scalar float either way."""
        model = plr.BoTorchGP(signal_variance=1.0, length_scale=0.8)
        model.set_data(ACTIONS, IDX, _feedback(), precision=precision)

        assert isinstance(model.sigma2, float)
        assert model.sigma2 == pytest.approx(0.01)


class TestSampling:

    def test_is_reproducible(self, gp):
        first = gp.sample_posterior(np.random.default_rng(3))
        second = gp.sample_posterior(np.random.default_rng(3))
        np.testing.assert_array_equal(first, second)

    def test_leaves_the_torch_generator_alone(self, gp):
        import torch

        before = torch.random.get_rng_state()
        gp.sample_posterior(np.random.default_rng(3))
        assert torch.equal(before, torch.random.get_rng_state())

    def test_concentrates_on_the_posterior(self, gp):
        rng = np.random.default_rng(0)
        draws = np.array([gp.sample_posterior(rng) for _ in range(400)])

        # Standard error of the sample mean is std/sqrt(400); the tolerance is a
        # few of those, plus slack for the Fourier-feature approximation.
        assert np.abs(draws.mean(axis=0) - gp.mu).max() < 0.05
        assert np.abs(draws.std(axis=0) - gp.std()).max() < 0.05


class TestHyperparameterFitting:

    def test_lengthscale_moves_toward_the_truth(self):
        """Started an order of magnitude too small, the fit recovers the scale."""
        rng = np.random.default_rng(0)
        actions = np.linspace(0, 10, 200).reshape(-1, 1)

        prior = plr.ConjugateGP(signal_variance=1.0, length_scale=2.0, rng=rng)
        prior.set_data(actions, np.array([], dtype=int), np.array([]), precision=1.0)
        truth = prior.sample_posterior(rng)

        idx = rng.choice(actions.shape[0], size=40, replace=False)
        y = truth[idx] + 0.05 * rng.standard_normal(idx.shape[0])

        model = plr.BoTorchGP(signal_variance=1.0, length_scale=0.2, rng=rng,
                              fit_hypers=True)
        model.set_data(actions, idx, y, precision=200.0)
        model.fit()

        fitted = model.model.covar_module.base_kernel.lengthscale.detach().item()
        assert 0.8 < fitted < 5.0

    def test_ard_gives_one_lengthscale_per_dimension(self):
        rng = np.random.default_rng(0)
        grid = np.stack(np.meshgrid(np.linspace(0, 4, 12), np.linspace(0, 4, 12)),
                        axis=-1).reshape(-1, 2)
        idx = rng.choice(grid.shape[0], size=20, replace=False)
        y = np.sin(grid[idx, 0]) + 0.1 * grid[idx, 1]

        model = plr.BoTorchGP(signal_variance=1.0, length_scale=1.0, rng=rng,
                              ard=True, fit_hypers=True)
        model.set_data(grid, idx, y, precision=50.0)
        model.fit()

        lengthscale = model.model.covar_module.base_kernel.lengthscale
        assert lengthscale.numel() == grid.shape[1]
        assert model.mu.shape == (grid.shape[0],)


class TestMultiObjectiveBackend:

    def test_setup_keeps_the_botorch_backend(self):
        """``setup(regressions=...)`` must not fall back to ConjugateGP."""
        optimizer = plr.MultiObjectiveGP(
            num_objs=2, kernels=['squared_exp'] * 2, signal_variances=[1.0] * 2,
            length_scales=[0.8] * 2, x0_init_methods=['random'] * 2,
            backend='botorch'
        )
        gps = optimizer.gps
        y = _feedback()
        optimizer.setup(
            ACTIONS, regressions=[(IDX, y, PRECISION), (IDX, -y, PRECISION)]
        )
        optimizer.fit()

        assert all(isinstance(gp, plr.BoTorchGP) for gp in optimizer.gps)
        assert optimizer.gps is gps
        np.testing.assert_allclose(optimizer.gps[0].mu, -optimizer.gps[1].mu)

    def test_threads_fit_hypers_and_ard(self):
        optimizer = plr.MultiObjectiveGP(
            num_objs=2, kernels=['squared_exp'] * 2, signal_variances=[1.0] * 2,
            length_scales=[0.8] * 2, x0_init_methods=['random'] * 2,
            backend='botorch', fit_hypers=[True, False], ard=[False, True]
        )

        assert [gp.fit_hypers for gp in optimizer.gps] == [True, False]
        assert [gp.ard for gp in optimizer.gps] == [False, True]
