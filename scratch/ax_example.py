"""Multi-objective HILO condition loop: metabolic cost vs. comfort."""

import numpy as np
import pandas as pd
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

SNAPSHOT = "p01_moo.json"
CONDITION_S = 120
STEADY_STATE_S = 60
N_CONDITIONS = 12
MASS_KG = 70.0
STANDING_W_PER_KG = 1.3
MIN_COMFORT_SEM = 0.25


def build_client(seed: int = 0) -> Client:
    client = Client(random_seed=seed)
    client.configure_experiment(
        name="p01_moo",
        parameters=[
            RangeParameterConfig(name="peak_torque", parameter_type="float", bounds=(10.0, 60.0)),
            RangeParameterConfig(name="peak_pct", parameter_type="float", bounds=(40.0, 60.0)),
            RangeParameterConfig(name="rise_pct", parameter_type="float", bounds=(10.0, 30.0)),
        ],
        parameter_constraints=["rise_pct <= peak_pct"],
    )
    # "-" marks minimization; comma separates objectives
    client.configure_optimization(
        objective="-metabolic_cost, comfort",
        outcome_constraints=["comfort >= 2.0"],
    )
    client.configure_generation_strategy(initialization_budget=8)
    return client


def metabolic_cost(breaths: pd.DataFrame) -> tuple[float, float]:
    """Net metabolic rate (W/kg) over the steady-state window, with SEM.

    breaths: columns t (s), vo2 (mL/s), vco2 (mL/s).
    """
    window = breaths[breaths["t"] >= breaths["t"].max() - STEADY_STATE_S]
    watts = (16.58 * window["vo2"] + 4.51 * window["vco2"]) / MASS_KG
    net = watts - STANDING_W_PER_KG
    # breath-by-breath samples are autocorrelated, so this SEM is optimistic
    return float(net.mean()), float(net.sem())


def comfort(ratings: list[float]) -> tuple[float, float]:
    """Mean and SEM of the 30 s tablet ratings collected during the condition."""
    r = np.asarray(ratings, dtype=float)
    sem = r.std(ddof=1) / np.sqrt(r.size) if r.size > 1 else np.inf
    return float(r.mean()), float(max(sem, MIN_COMFORT_SEM))


def run_condition(params: dict, exo, tablet) -> dict[str, tuple[float, float]]:
    # exo.apply(params)
    # tablet.start_prompts(interval_s=30)
    # breaths = exo.collect_metabolics(duration_s=CONDITION_S)
    # ratings = tablet.drain()
    # print(params)
    # quit()
    return {"metabolic_cost": -np.square(params['peak_pct'] - 50.0), "comfort": -np.square(params['rise_pct'] - 20.0)}


def main(exo, tablet) -> pd.DataFrame:
    client = build_client()
    print('here')

    for _ in range(N_CONDITIONS):
        trial_index, params = client.get_next_trials(max_trials=1).popitem()
        print(f"trial {trial_index}: {params}")

        try:
            raw_data = run_condition(params, exo, tablet)
        except KeyboardInterrupt:
            client.mark_trial_abandoned(trial_index)
            client.save_to_json_file(SNAPSHOT)
            raise

        client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        client.save_to_json_file(SNAPSHOT)

    rows = []
    for params, metrics, trial_index, arm_name in client.get_pareto_frontier():
        means = {k: (v[0] if isinstance(v, tuple) else v) for k, v in metrics.items()}
        rows.append({"trial_index": trial_index, "arm_name": arm_name, **params, **means})

    frontier = pd.DataFrame(rows)
    frontier.to_csv("p01_frontier.csv", index=False)
    client.compute_analyses(display=True)
    return frontier

if __name__ == '__main__':
    main(None, None)