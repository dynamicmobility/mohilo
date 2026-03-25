import numpy as np
import jax
import jax.numpy as jnp
import pytest
from pypolar import PreferenceBasedLearning


class TestActionSpace:
    def test_1d_action_space(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([6]),
            action_dims=np.array([7]),
        )
        assert pbl.action_space.shape == (7, 1)
        np.testing.assert_allclose(pbl.action_space.flatten(), np.arange(7))

    def test_2d_action_space_shape(self):
        pbl = PreferenceBasedLearning(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            action_dims=np.array([3, 3]),
        )
        assert pbl.action_space.shape == (9, 2)

    def test_3d_action_space_shape(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0, 0, 0]),
            high=np.array([1, 1, 1]),
            action_dims=np.array([5, 5, 5]),
        )
        assert pbl.action_space.shape == (125, 3)


class TestGetIdx:
    def test_1d_get_idx_roundtrip(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([6]),
            action_dims=np.array([7]),
        )
        for i, action in enumerate(pbl.action_space):
            assert pbl.get_idx(action) == i

    def test_2d_get_idx_roundtrip(self):
        pbl = PreferenceBasedLearning(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            action_dims=np.array([3, 3]),
        )
        for i, action in enumerate(pbl.action_space):
            assert pbl.get_idx(action) == i


class TestSigmoid:
    def test_sigmoid_zero(self):
        assert PreferenceBasedLearning.sigmoid(0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        assert PreferenceBasedLearning.sigmoid(100) == pytest.approx(1.0)

    def test_sigmoid_large_negative(self):
        assert PreferenceBasedLearning.sigmoid(-100) == pytest.approx(0.0)

    def test_sigmoid_array(self):
        result = PreferenceBasedLearning.sigmoid(jnp.array([-100, 0, 100]))
        np.testing.assert_allclose(result, [0, 0.5, 1], atol=1e-10)

    def test_sigmoid_grad_at_zero(self):
        grad_fn = jax.grad(lambda x: PreferenceBasedLearning.sigmoid(x))
        assert float(grad_fn(0.0)) == pytest.approx(0.25)


class TestCompile:
    def test_compile_preference(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([4]),
            action_dims=np.array([5]),
        )
        pbl.add_feedback((np.array([2.0]), np.array([1.0]), True), None, None)
        pbl.compile()
        assert pbl._jax_pref is not None
        assert pbl._jax_pref.shape == (1, 2)
        assert pbl._jax_pref.dtype == jnp.int32

    def test_compile_empty(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([4]),
            action_dims=np.array([5]),
        )
        pbl.compile()
        assert pbl._jax_pref is None
        assert pbl._jax_coac is None
        assert pbl._jax_ordi_idx is None

    def test_compile_sets_flag(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([4]),
            action_dims=np.array([5]),
        )
        assert not pbl._compiled
        pbl.compile()
        assert pbl._compiled

    def test_add_feedback_clears_compiled(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([4]),
            action_dims=np.array([5]),
        )
        pbl.compile()
        assert pbl._compiled
        pbl.add_feedback((np.array([2.0]), np.array([1.0]), True), None, None)
        assert not pbl._compiled


class TestFeedback:
    @pytest.fixture
    def pbl(self):
        return PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([4]),
            action_dims=np.array([5]),
            preference_noise=0.01,
        )

    def test_add_preference_preferred(self, pbl):
        curr = np.array([2.0])
        prev = np.array([1.0])
        pbl.add_feedback((curr, prev, True), None, None)
        assert len(pbl.preference_fbk) == 1
        assert pbl.preference_fbk[0][0] == pbl.get_idx(curr)
        assert pbl.preference_fbk[0][1] == pbl.get_idx(prev)

    def test_add_preference_not_preferred(self, pbl):
        curr = np.array([2.0])
        prev = np.array([1.0])
        pbl.add_feedback((curr, prev, False), None, None)
        assert len(pbl.preference_fbk) == 1
        assert pbl.preference_fbk[0][0] == pbl.get_idx(prev)
        assert pbl.preference_fbk[0][1] == pbl.get_idx(curr)

    def test_add_none_feedback(self, pbl):
        pbl.add_feedback(None, None, None)
        assert len(pbl.preference_fbk) == 0
        assert len(pbl.coactive_fbk) == 0
        assert len(pbl.ordinal_fbk) == 0


class TestLikelihood:
    @pytest.fixture
    def pbl_with_feedback(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([4]),
            action_dims=np.array([5]),
            preference_noise=0.5,
        )
        pbl.add_feedback((np.array([3.0]), np.array([1.0]), True), None, None)
        pbl.compile()
        return pbl

    def test_overall_likelihood_returns_scalar(self, pbl_with_feedback):
        r = jnp.array(np.random.rand(5))
        result = pbl_with_feedback.overall_likelihood(r)
        assert result.shape == ()

    def test_higher_reward_gives_lower_nll(self, pbl_with_feedback):
        r_good = jnp.array([0.0, 0, 0, 1, 0])
        r_bad = jnp.array([0.0, 1, 0, 0, 0])
        assert pbl_with_feedback.overall_likelihood(r_good) < pbl_with_feedback.overall_likelihood(r_bad)

    def test_empty_likelihood_is_zero(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([4]),
            action_dims=np.array([5]),
        )
        pbl.compile()
        r = jnp.array(np.random.rand(5))
        assert pbl.overall_likelihood(r) == 0

    def test_likelihood_raises_without_compile(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([4]),
            action_dims=np.array([5]),
        )
        pbl.add_feedback((np.array([2.0]), np.array([1.0]), True), None, None)
        with pytest.raises(RuntimeError, match="compile"):
            pbl.overall_likelihood(jnp.ones(5))


class TestJaxCompatibility:
    @pytest.fixture
    def pbl_compiled(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([4]),
            action_dims=np.array([5]),
            preference_noise=0.5,
        )
        pbl.add_feedback((np.array([3.0]), np.array([1.0]), True), None, None)
        pbl.add_feedback((np.array([2.0]), np.array([0.0]), True), None, None)
        pbl.compile()
        return pbl

    def test_grad_returns_correct_shape(self, pbl_compiled):
        r = jnp.ones(5)
        grad = jax.grad(pbl_compiled.overall_likelihood)(r)
        assert grad.shape == (5,)

    def test_grad_is_finite(self, pbl_compiled):
        r = jnp.ones(5)
        grad = jax.grad(pbl_compiled.overall_likelihood)(r)
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_matches_eager(self, pbl_compiled):
        r = jnp.array([0.1, 0.5, 0.3, 0.9, 0.2])
        eager = pbl_compiled.overall_likelihood(r)
        jitted = jax.jit(pbl_compiled.overall_likelihood)(r)
        np.testing.assert_allclose(float(eager), float(jitted), rtol=1e-5)

    def test_hessian_shape(self, pbl_compiled):
        r = jnp.ones(5)
        hess = jax.hessian(pbl_compiled.overall_likelihood)(r)
        assert hess.shape == (5, 5)


class TestPredict:
    def test_predict_correct(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([4]),
            action_dims=np.array([5]),
        )
        r = np.array([0, 1, 2, 3, 4], dtype=float)
        assert pbl.predict(r, np.array([4.0]), np.array([0.0])) == True
        assert pbl.predict(r, np.array([0.0]), np.array([4.0])) == False


class TestOptimalAction:
    def test_optimal_action(self):
        pbl = PreferenceBasedLearning(
            low=np.array([0]),
            high=np.array([4]),
            action_dims=np.array([5]),
        )
        r = np.array([0, 1, 5, 3, 2], dtype=float)
        optimal = pbl.optimal_action(r)
        np.testing.assert_allclose(optimal, np.array([2.0]))
