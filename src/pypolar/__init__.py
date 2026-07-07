import jax
jax.config.update("jax_enable_x64", True)

from pypolar.optimization.likelihood import PreferenceBasedLearning, Regression
from pypolar.optimization.gp import BasicGP
from pypolar.sampler import RandomSampler, ThompsonSampler, DSTSampler
from pypolar.plotting import plot_gp_1d
from pypolar.feedback import (
    PerfectOracle,
    BradleyTerryOracle,
    MultiObjectiveOracle,
    NoisyRegressionOracle,
    IdealPoint,
    NonStationaryIdealPoint,
    MultiObjectiveIdealPoint,
)

__all__ = [
    # optimization
    "PreferenceBasedLearning",
    "Regression",
    "BasicGP",
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
    # plotting
    "plot_gp_1d",
]
