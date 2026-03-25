import numpy as np
import pytest
from pypolar import SimulatedFeedback, SimulatedObjective


class TestSimulatedObjective:
    def test_1d_callable(self):
        obj = SimulatedObjective(function='1D')
        result = obj(np.array([1.0]))
        assert result.size == 1

    def test_2d_callable_single(self):
        obj = SimulatedObjective(function='2D')
        result = obj(np.array([0.5, 0.5]))
        assert np.isscalar(result) or result.shape == ()

    def test_2d_callable_batch(self):
        obj = SimulatedObjective(function='2D')
        result = obj(np.array([[0.5, 0.5], [0.1, 0.2]]))
        assert result.shape == (2,)

    def test_ordinal_labels(self):
        action_space = np.linspace(0, 6, 40).reshape(-1, 1)
        obj = SimulatedObjective(function='1D', ord_lbls=3, action_space=action_space)
        b0, b1 = obj.get_ordinal_label(np.array([1.0]))
        assert b0 < b1

    def test_invalid_function_raises(self):
        with pytest.raises(ValueError, match="Unknown objective function"):
            SimulatedObjective(function='invalid')


class TestSimulatedFeedback:
    @pytest.fixture
    def feedback(self):
        obj = SimulatedObjective(function='1D')
        action_space = np.linspace(0, 6, 10).reshape(-1, 1)
        return SimulatedFeedback(objective=obj, action_space=action_space)

    def test_pairwise_feedback(self, feedback):
        pref, coac, ordi = feedback.evaluate(
            curr=np.array([1.0]),
            prev=np.array([2.0]),
            get_pairwise=True,
        )
        assert pref is not None
        assert coac is None
        assert ordi is None
        curr, prev, label = pref
        assert bool(label) in (True, False)

    def test_coactive_feedback(self, feedback):
        pref, coac, ordi = feedback.evaluate(
            curr=np.array([1.0]),
            get_coactive=True,
        )
        assert pref is None
        assert coac is not None
        a_bar, curr, label = coac
        assert bool(label) in (True, False)

    def test_pairwise_requires_prev(self, feedback):
        with pytest.raises(ValueError, match="prev"):
            feedback.evaluate(curr=np.array([1.0]), get_pairwise=True)

    def test_coactive_requires_action_space(self):
        obj = SimulatedObjective(function='1D')
        fb = SimulatedFeedback(objective=obj, action_space=None)
        with pytest.raises(ValueError, match="action_space"):
            fb.evaluate(curr=np.array([1.0]), get_coactive=True)
