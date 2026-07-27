"""JSON round-trip checks for the experiment configs.

``experiment.py`` writes a config into every ``run.json``, so a config that does
not survive the round trip silently misreports what was run. The union-typed
``sampler`` field is the interesting case: it has to decode back to the sampler
that was saved, not to whichever union member happens to be declared first.
"""

import pytest

from config.aq_test import sim_1d
from config.base import (
    DSTS,
    HILO,
    MOHILO,
    ExpectedImprovement,
    KnowledgeGradient,
    MaxValueEntropy,
    QNEHVI,
    RandomSampling,
    ThompsonSampling,
)
from config.hipexo import hipexo_sim_idealized_2d


@pytest.mark.parametrize(
    "sampler",
    [RandomSampling(), DSTS(rho=0.01), QNEHVI(num_samples=64)],
    ids=lambda s: type(s).__name__,
)
def test_mo_sampler_survives_round_trip(sampler):
    config = hipexo_sim_idealized_2d.model_copy(deep=True)
    config.sampler = sampler

    restored = MOHILO.from_json_string(config.to_json_string())
    assert type(restored.sampler) is type(sampler)


@pytest.mark.parametrize(
    "sampler",
    [
        ThompsonSampling(),
        ExpectedImprovement(xi=0.2),
        KnowledgeGradient(num_candidates=7),
        MaxValueEntropy(num_maxima=9),
    ],
    ids=lambda s: type(s).__name__,
)
def test_single_objective_sampler_survives_round_trip(sampler):
    config = sim_1d.model_copy(deep=True)
    config.sampler = sampler

    restored = HILO.from_json_string(config.to_json_string())
    assert type(restored.sampler) is type(sampler)


def test_sampler_fields_survive_round_trip():
    config = hipexo_sim_idealized_2d.model_copy(deep=True)
    config.sampler = QNEHVI(num_samples=64, ref_point=[-1.0, -2.0])

    restored = MOHILO.from_json_string(config.to_json_string()).sampler
    assert restored.num_samples == 64
    assert list(restored.ref_point) == [-1.0, -2.0]


def test_gp_backend_fields_survive_round_trip():
    config = hipexo_sim_idealized_2d.model_copy(deep=True)
    config.optimizer.gptype = 'BoTorchGP'
    config.optimizer.fit_hypers = [True, False]

    restored = MOHILO.from_json_string(config.to_json_string()).optimizer
    assert restored.gptype == 'BoTorchGP'
    assert restored.fit_hypers == [True, False]
