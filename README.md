# pypolar

A Python implementation of POLAR (Preference Optimization and Learning Algorithm for Robotics) for preference-based learning. Learn a reward function from pairwise comparisons, coactive feedback, and ordinal labels using Gaussian processes with JAX autodiff.

## Installation

Requires Python 3.10+. Install in a conda environment:

```bash
conda create -n pypolar python=3.12
conda activate pypolar
pip install -e ".[dev]"
```

## Examples

```bash
conda activate pypolar
python examples/example_1d.py
python examples/example_2d.py
python examples/example_3d.py
```

- **`example_1d.py`** — Learns a 1D reward function using Thompson sampling with 20 pairwise comparisons. Refits the GP each iteration to actively select informative queries. Plots the learned reward vs. the true objective.
- **`example_2d.py`** — Learns a 2D reward function and visualizes it as a 3D surface alongside the ground truth.
- **`example_3d.py`** — Learns preferences over a 3D action space (e.g., RGB colors).

## Tests

```bash
python -m pytest tests/ -v
```

## How it works

1. Define a discretized action space with `PreferenceBasedLearning`
2. Collect pairwise preference feedback (simulated or real)
3. Call `pbl.compile()` to convert feedback into JAX arrays
4. Fit a GP to learn the latent reward — `ConjugateGP` (closed form, for regression feedback) or `LaplaceGP` (gradients and Hessians computed automatically via JAX)
5. Use `ThompsonSampler` to actively select the next query, or `pbl.predict()` / `pbl.optimal_action()` to use the learned reward


```
python -m hilo.fit_mogp --subject MB03 --no-emulate --connect
```

```
python -m hilo.fit_mogp --subject MB03 --no-emulate --connect --resume PATH
```

```
python -m hilo.compare_front --dataset ??? --connect --no-emulate
```

```
python -m hilo.explore_front --dataset ??? --connect --no-emulate
```