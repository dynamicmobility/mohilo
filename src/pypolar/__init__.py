import jax
jax.config.update("jax_enable_x64", True)

from pypolar.optimization.problem import (
    PreferenceBasedLearning, 
    Regression,
    MultiObjectiveRegression
)
from pypolar.optimization.gp import (
    GPModel,
    BoTorchGP,
    ConjugateGP,
    LaplaceGP,
    MultiObjectiveGP
)
from pypolar.sampler import (
    RandomSampler,
    UniformSampler,
    ThompsonSampler,
    DSTSampler,
    QNEHVISampler,
    AcquisitionSampler,
    ExpectedImprovementSampler,
    KnowledgeGradientSampler,
    MaxValueEntropySampler,
    expected_max_of_lines,
    knowledge_gradient
)
from pypolar.utils.pareto import get_pareto_statistics, get_nondominated, get_nondominated_tol, hypervolume_from_nondominated, sparsity_from_normalized_nondominated
from pypolar.utils.plotting import plot_gp_1d, plot_pareto_2d
from pypolar.utils.gp import derive_gp_hyperparams, derive_lengthscale, derive_precision, derive_prior_variance
from pypolar.feedback import (
    BradleyTerryOracle,
    MultiObjectiveOracle,
    NoisyRegressionOracle,
    IdealPoint,
    NonStationaryIdealPoint,
    MultiObjectiveIdealPoint,
    BoundedIdealPoint
)
from pypolar.performance.mo import pareto_overlay, groundtruth_hypervolume

__all__ = [
    # optimization
    "PreferenceBasedLearning",
    "Regression",
    "MultiObjectiveRegression",
    "GPModel",
    "BoTorchGP",
    "ConjugateGP",
    "LaplaceGP",
    "MultiObjectiveGP",
    # sampling
    "RandomSampler",
    "UniformSampler",
    "ThompsonSampler",
    "DSTSampler",
    "QNEHVISampler",
    "AcquisitionSampler",
    "ExpectedImprovementSampler",
    "KnowledgeGradientSampler",
    "MaxValueEntropySampler",
    "expected_max_of_lines",
    "knowledge_gradient",
    # feedback oracles
    "BradleyTerryOracle",
    "MultiObjectiveOracle",
    "NoisyRegressionOracle",
    # feedback rewards
    "IdealPoint",
    "NonStationaryIdealPoint",
    "MultiObjectiveIdealPoint",
    "BoundedIdealPoint",
    # pareto
    "get_pareto_statistics",
    "get_nondominated",
    "get_nondominated_tol",
    "hypervolume_from_nondominated",
    "sparsity_from_normalized_nondominated",
    # plotting
    "plot_gp_1d",
    "plot_pareto_2d",
    # performance
    "pareto_overlay",
    "groundtruth_hypervolume",
    # utils.gp
    "derive_gp_hyperparams", 
    "derive_lengthscale", 
    "derive_precision", 
    "derive_prior_variance"
]
