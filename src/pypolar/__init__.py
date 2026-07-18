import jax
jax.config.update("jax_enable_x64", True)

from pypolar.optimization.problem import (
    PreferenceBasedLearning, 
    Regression,
    MultiObjectiveRegression
)
from pypolar.optimization.gp import BasicGP, MultiObjectiveGP
from pypolar.sampler import RandomSampler, ThompsonSampler, DSTSampler
from pypolar.plotting import plot_gp_1d, plot_pareto_2d
from pypolar.feedback import (
    PerfectOracle,
    BradleyTerryOracle,
    MultiObjectiveOracle,
    NoisyRegressionOracle,
    IdealPoint,
    NonStationaryIdealPoint,
    MultiObjectiveIdealPoint,
    BoundedIdealPoint
)

__all__ = [
    # optimization
    "PreferenceBasedLearning",
    "Regression",
    "MultiObjectiveRegression",
    "BasicGP",
    "MultiObjectiveGP",
    # sampling
    "RandomSampler",
    "ThompsonSampler",
    "DSTSampler",
    # feedback oracles
    "PerfectOracle",
    "BradleyTerryOracle",
    "MultiObjectiveOracle",
    "NoisyRegressionOracle",
    # feedback rewards
    "IdealPoint",
    "NonStationaryIdealPoint",
    "MultiObjectiveIdealPoint",
    "BoundedIdealPoint",
    # plotting
    "plot_gp_1d",
    "plot_pareto_2d"
]
