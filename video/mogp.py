"""A multi-objective GP learning both objectives, one measurement at a time.

It opens on what `ParetoScene` ends on -- the same axes, labels, true curves and
attainable landscape, in the same places -- so the two scenes cut together. That
is why the layout constants below are shared with that scene and why nothing here
is drawn with `Create` until the true front.

The left column is each objective against the action, the right panel is the two
against each other. Every controller measured gives a noisy reading of *both*
objectives at the same action; after each one the repo's own `DecoupledMOGP` is
refit and its posterior redrawn, so the band narrows and the front it predicts
settles onto the true one. The posterior over the action is drawn in `INK` and the
front it predicts in `LOSS_COLOR`, against the true front in `INK`; the objectives
themselves stay in their own colors.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from manim import (
    DOWN,
    UP,
    Create,
    DashedVMobject,
    Dot,
    FadeIn,
    Line,
    Polygon,
    ReplacementTransform,
    Scene,
    Tex,
    Transform,
    VGroup,
    VMobject,
    Write,
)

# manim loads this file by path rather than as a package, so the repo root is
# put on sys.path for the `video.style` import below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pypolar as plr  # noqa: E402
from video.objectives import (  # noqa: E402
    COMFORT_NAME,
    COST_NAME,
    CURVE_X_RANGE,
    TRUTH_PARAMS,
    attainable_actions,
    comfort,
    cost,
    front_actions,
    sobol_actions,
)
from video.style import (  # noqa: E402
    AXIS_Y_MAX,
    COMFORT_COLOR,
    COMFORT_Y,
    COST_COLOR,
    CURVE_STROKE,
    DROP_COLOR,
    GAIN_COLOR,
    INK,
    LANDSCAPE_STROKE,
    LOSS_COLOR,
    METABOLIC_Y,
    PLOT_SAMPLES,
    SAMPLE_COLOR,
    SCENE_TITLE_SIZE,
    TITLE_EDGE_BUFF,
    action_axes,
    action_plot,
    front_axes as front_axes_builder,
)

# The title, set over two lines because one line of it at this size is wider than
# the frame. Everything below is pushed down to clear it.
TITLE = ("Multi-Objective Human in the Loop", "Bayesian Optimization")
TITLE_LINE_BUFF = 0.15

# The design: a Sobol seed of N_SEED actions, chosen before anything is known, and
# then one queried action at a time up to N_POINTS. The seed exists because a GP
# needs a spread to standardize by, which one point does not have and two badly
# underestimate -- and because an acquisition scores against measurements.
N_POINTS = 10
N_SEED = 3
SAMPLE_SEED = 1

# The acquisition: qLogNParEGO draws a fresh random Chebyshev scalarization of the
# two objectives per call and runs noisy log-EI on it, so a different corner of the
# front is chased each step. Its seed is advanced per query for that reason; a
# fixed one would redraw the same weights every time.
ACQ_STRATEGY = 'qlognparego'
ACQ_SEED = 0

# `AcquisitionParams`' own seed reaches only the acquisition's QMC sampler. Four
# other draws read torch's global generator instead: qLogNParEGO's Chebyshev
# weights, `optimize_acqf`'s Sobol restarts, `prune_baseline`'s pruning, and the
# hyperparameters a failed fit is retried from. Seeding it once is what makes a
# render repeatable; without it the queried actions differ every time.
TORCH_SEED = 0

# Observation noise for this scene only, as a fraction of each objective's own
# spread: heavier than the package default, since what this scene shows is a GP
# recovering a curve the measurements only loosely trace. The oracle is rebuilt
# from the shared parameters rather than the shared oracle being changed, so the
# other scenes keep their own noise.
NOISE_STD = 0.5
NOISY_TRUTH = replace(TRUTH_PARAMS, rel_noise_std=NOISE_STD).build()

# The GP: a lengthscale floor, since below the design spacing a lengthscale is not
# identifiable, and the noise fitted under a prior centered on the truth's own.
MIN_LENGTH_SCALE = 0.15
GRID_POINTS = 201
BAND_POINTS = 61
BAND_STD = 1.0
BAND_OPACITY = 0.3

QUERY_STROKE = 4.0
TRUE_FRONT_STROKE = 16.0
TRUE_FRONT_OPACITY = 0.35
MEAN_STROKE = 3.5
MEAN_DASH = 0.1
DOT_RADIUS = 0.075

GRID = np.linspace(*CURVE_X_RANGE, GRID_POINTS)
COLORS = (COST_COLOR, COMFORT_COLOR)


def fit(actions, values):
    """A `DecoupledMOGP` over both objectives, fit to the measurements so far.

    Args:
        actions: `(n, 1)` the actions measured, in raw units.
        values: `(n, 2)` their noisy cost and comfort readings.

    The action bounds are pinned on both objectives on purpose: left unpinned, the
    normalized frame is the *measured* action range, so it would move every time a
    point is added and the posterior would shift for reasons that have nothing to
    do with the new measurement.
    """
    objectives = plr.DecoupledObjectives([
        plr.Objective.from_data(actions=actions, values=values[:, 0], maximize=False,
                                name=COST_NAME, action_bounds=list(CURVE_X_RANGE)),
        plr.Objective.from_data(actions=actions, values=values[:, 1], maximize=True,
                                name=COMFORT_NAME, action_bounds=list(CURVE_X_RANGE)),
    ])
    return plr.DecoupledMOGP(
        objectives          = objectives,
        fit_hyperparameters = True,
        noise               = plr.NoiseModel.prior(NOISE_STD),
        min_length_scale    = MIN_LENGTH_SCALE,
    )


def posterior(gp):
    """The fit read off on GRID: `(mu, std)` in raw units, and the predicted front.

    Returns `(mu, std, front)`, the first two `(GRID_POINTS, 2)` in each objective's
    own units and sign, and `front` the indices of the non-dominated grid points,
    ordered along the front. Both the dominance and the ordering are decided on the
    *maximization* posterior, since that is all the model has; the raw posterior is
    only what the action plots are drawn in.
    """
    grid = GRID[:, None]
    scan, _ = gp.posterior_at(grid)
    mu, std = gp.posterior_at(grid, raw=True)
    front = plr.get_nondominated(scan)
    return mu, std, front[np.argsort(-scan[front, 0])]


class MogpScene(Scene):
    def construct(self):
        torch.manual_seed(TORCH_SEED)
        title = VGroup(*(Tex(line, font_size=SCENE_TITLE_SIZE, color=INK)
                         for line in TITLE))
        title.arrange(DOWN, buff=TITLE_LINE_BUFF).to_edge(UP, buff=TITLE_EDGE_BUFF)
        metabolic_axes = action_axes(METABOLIC_Y)
        comfort_axes = action_axes(COMFORT_Y)
        metabolic = action_plot(
            metabolic_axes, "metabolic cost", cost, attainable_actions(), COST_COLOR
        )
        comfort_plot = action_plot(
            comfort_axes, "comfort", comfort, attainable_actions(), COMFORT_COLOR,
            x_label=True
        )
        front_axes, front_labels = front_axes_builder()
        landscape = front_axes.plot_parametric_curve(
            lambda a: (cost(a), comfort(a)),
            t_range=[*attainable_actions(), PLOT_SAMPLES],
            color=DROP_COLOR,
            stroke_width=LANDSCAPE_STROKE,
        )
        true_front = front_axes.plot_parametric_curve(
            lambda a: (cost(a), comfort(a)),
            t_range=[*front_actions(), PLOT_SAMPLES],
            color=INK,
            stroke_width=TRUE_FRONT_STROKE,
            stroke_opacity=TRUE_FRONT_OPACITY,
        )

        # Added rather than animated: this is exactly what `ParetoScene` ends on,
        # so the two cut together and the axes never redraw.
        self.add(metabolic_axes, comfort_axes, metabolic, comfort_plot,
                 front_axes, front_labels, landscape)

        self.play(Write(title, run_time=1.0))
        self.play(Create(true_front, run_time=1.2))
        self.wait(0.5)

        self._measure(metabolic_axes, comfort_axes, front_axes)
        self.wait(1.0)

    def _measure(self, metabolic_axes, comfort_axes, front_axes):
        """The seed design, then one acquisition-chosen action at a time.

        Both objectives are measured in one call per action, so a pair of dots is
        one visit to one controller. The fits happen here, while the scene is being
        built, and each one is a couple of exact GPs over at most N_POINTS points.
        """
        axes = (metabolic_axes, comfort_axes)
        actions = sobol_actions(N_SEED, SAMPLE_SEED)
        values = NOISY_TRUTH(actions)

        for n in range(N_SEED):
            self.play(FadeIn(self._dots(axes, actions[n, 0], values[n]),
                             scale=0.5, run_time=0.45))

        gp = fit(actions, values)
        drawn = self._draw_fit(axes, front_axes, gp, None)
        query = None

        for step in range(N_POINTS - N_SEED):
            action = self._query(gp, step)
            lines = VGroup(*(self._query_line(ax, action) for ax in axes))
            if query is None:
                self.play(Create(lines, run_time=0.6))
                query = lines
            else:
                # Transform moves the line already on screen, so `query` stays the
                # mobject being animated and `lines` is only its target.
                self.play(Transform(query, lines, run_time=0.6))
            self.wait(0.3)

            measured = NOISY_TRUTH(np.array([[action]]))
            actions = np.vstack([actions, [[action]]])
            values = np.vstack([values, measured])
            self.play(FadeIn(self._dots(axes, action, measured[0]),
                             scale=0.5, run_time=0.45))

            gp = fit(actions, values)
            drawn = self._draw_fit(axes, front_axes, gp, drawn)

    def _query(self, gp, step):
        """The action qLogNParEGO wants measured next, as a float in raw units."""
        acquisition = plr.AcquisitionParams(
            strategy       = ACQ_STRATEGY,
            seed           = ACQ_SEED + step,
            num_objectives = 2,
        ).build()
        return float(acquisition.query(gp, q=1)[0, 0])

    def _dots(self, axes, action, values):
        """One measurement of both objectives: a dot on each action plot."""
        return VGroup(*(
            Dot(ax.c2p(action, values[i]), radius=DOT_RADIUS, color=SAMPLE_COLOR)
            for i, ax in enumerate(axes)
        ))

    def _query_line(self, axes, action):
        """Where the next query lands, as a line up one action plot."""
        return Line(
            axes.c2p(action, 0.0),
            axes.c2p(action, AXIS_Y_MAX),
            color=GAIN_COLOR,
            stroke_width=QUERY_STROKE,
        )

    def _draw_fit(self, axes, front_axes, gp, drawn):
        """The bands, means and predicted front of one fit, morphed from the last.

        Returns what to pass back as `drawn` at the next fit; `None` means nothing
        is on screen yet, so everything is created rather than transformed.
        """
        mu, std, front = posterior(gp)
        bands = VGroup(*(self._band(ax, mu[:, i], std[:, i], color)
                         for i, (ax, color) in enumerate(zip(axes, COLORS))))
        means = VGroup(*(self._mean(ax, mu[:, i]) for i, ax in enumerate(axes)))
        predicted = self._front(front_axes, mu, front)

        if drawn is None:
            self.play(FadeIn(bands, run_time=0.8), Create(means, run_time=0.8),
                      Create(predicted, run_time=0.8))
        else:
            self.play(
                Transform(drawn[0], bands, run_time=0.8),
                ReplacementTransform(drawn[1], means, run_time=0.8),
                ReplacementTransform(drawn[2], predicted, run_time=0.8),
            )
            bands = drawn[0]     # Transform leaves the original on screen
        self.wait(0.25)
        return (bands, means, predicted)

    def _mean(self, axes, mu):
        """The posterior mean over the action box, dashed so the truth stays legible."""
        curve = axes.plot(
            lambda a: float(np.interp(a, GRID, mu)),
            x_range=attainable_actions(),
            color=INK,
            stroke_width=MEAN_STROKE,
        )
        return DashedVMobject(curve, dashed_ratio=0.55, num_dashes=42)

    def _band(self, axes, mu, std, color):
        """A BAND_STD band around the mean, as one polygon of fixed vertex count.

        Fixed, so that one step's band morphs into the next rather than being
        rebuilt from a different number of points. The standard deviation is the
        latent posterior's, which is the model's uncertainty about the noiseless
        curve -- what the truth curve should sit inside. An early band is wider than
        the plot, so it is clipped to the axes rather than spilling over them.
        """
        keep = np.linspace(0, GRID_POINTS - 1, BAND_POINTS).astype(int)
        edge = np.clip(
            [mu[keep] + BAND_STD * std[keep], mu[keep] - BAND_STD * std[keep]],
            0.0, AXIS_Y_MAX,
        )
        upper = [axes.c2p(GRID[i], y) for i, y in zip(keep, edge[0])]
        lower = [axes.c2p(GRID[i], y) for i, y in zip(keep[::-1], edge[1][::-1])]
        band = Polygon(*upper, *lower)
        band.set_fill(color, opacity=BAND_OPACITY).set_stroke(width=0)
        return band

    def _front(self, axes, mu, front):
        """The model's front: the true objectives at the actions the posterior calls
        non-dominated.

        The GP picks the actions; the curve is where those actions really land, so
        the gap to the true front is the cost of the model's mistakes rather than of
        its posterior mean being off. `mu` is unused here for that reason.
        """
        points = [axes.c2p(cost(GRID[i]), comfort(GRID[i])) for i in front]
        return VMobject(color=LOSS_COLOR, stroke_width=CURVE_STROKE).set_points_smoothly(points)

