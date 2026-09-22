"""A multi-objective GP learning both objectives, one measurement at a time.

It opens on what `ParetoScene` ends on -- the same axes, labels, true curves and
attainable landscape, in the same places -- so the two scenes cut together. That
is why the layout constants below are shared with that scene and why nothing here
is drawn with `Create` until the true front.

The left column is each objective against the action, the right panel is the two
against each other. Every controller measured gives a noisy reading of *both*
objectives at the same action, with one exception: at `MULTI_TRIAL`, comfort
alone gets `MULTI_N` readings of the same controller, stacked on top of each
other and boxed on its action plot, which is what the two objectives being
*decoupled* actually buys -- one need not carry the other's design. After each
measurement the repo's own `DecoupledMOGP` is
refit and its posterior redrawn, so the band narrows and the front it predicts
settles onto the true one. The posterior over the action is drawn in `INK` and the
front it predicts as dots graded from `COST_COLOR` to `COMFORT_COLOR` along the
predicted ordering, against the true front in `INK`; the objectives themselves
stay in their own colors.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Create,
    DashedLine,
    DashedVMobject,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Polygon,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Tex,
    Transform,
    VGroup,
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
    BOX_RADIUS,
    COMFORT_COLOR,
    COMFORT_Y,
    COST_COLOR,
    DROP_COLOR,
    GAIN_COLOR,
    INK,
    LANDSCAPE_STROKE,
    LOSS_COLOR,
    METABOLIC_Y,
    PLOT_SAMPLES,
    SAMPLE_COLOR,
    SCENE_TITLE_SIZE,
    STAR_COLOR,
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
SAMPLE_SEED = 3

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
FRONT_DOT_RADIUS = 0.06

# The one trial (1-indexed over every measurement, seed included) where comfort
# alone gets more than one reading: MULTI_N independent readings at the *same*
# queried action, stacked on top of each other on its action plot and boxed in
# STAR_COLOR the same way `method.py` boxes a region of a figure.
MULTI_TRIAL = 6
MULTI_N = 3
MULTI_BOX_COLOR = STAR_COLOR
MULTI_BOX_STROKE = 5.0
MULTI_BOX_PAD = 0.12

# The estimated Pareto set, marked on the action axes the same way `ParetoScene`
# marks the true one.
SET_COLOR = LOSS_COLOR
SET_LINE_WIDTH = 6.0
SET_LABEL_SIZE = 36

# The caption naming the estimated front, placed and pointed the same way
# `ParetoScene` names the true one -- static, since it captions the panel rather
# than tracking dots that move fit to fit.
POINTER_AT = (0.75, 0.5)
POINTER_LABEL_SIZE = 36

GRID = np.linspace(*CURVE_X_RANGE, GRID_POINTS)
COLORS = (COST_COLOR, COMFORT_COLOR)

# The legend: a key for the symbols not already captioned in the front panel
# (the Pareto set and front have their own labels there), two entries per row.
# It sits above the front axes, in the space between the title and the true
# front's own top -- taller than just the gap above the axes, since that top
# corner of the panel is blank canvas until the front curve rises into it. It
# is built at a comfortable size and then shrunk to whatever actually fits
# that space rather than a size guessed in advance.
LEGEND_FONT_SIZE = 34
LEGEND_SWATCH_LEN = 0.5
LEGEND_LABEL_BUFF = 0.14
LEGEND_COL_BUFF = 0.6
LEGEND_ROW_BUFF = 0.25
LEGEND_CLEARANCE = 0.15
LEGEND_MARGIN = 0.92


def fit(cost_actions, cost_values, comfort_actions, comfort_values):
    """A `DecoupledMOGP` over both objectives, fit to the measurements so far.

    Args:
        cost_actions: `(n, 1)` the actions cost was measured at, in raw units.
        cost_values: `(n,)` its noisy readings.
        comfort_actions: `(m, 1)` the actions comfort was measured at -- not
            necessarily the same set or count as cost's, since `MULTI_TRIAL`
            gives comfort `MULTI_N` readings against cost's one.
        comfort_values: `(m,)` its noisy readings.

    The action bounds are pinned on both objectives on purpose: left unpinned, the
    normalized frame is the *measured* action range, so it would move every time a
    point is added and the posterior would shift for reasons that have nothing to
    do with the new measurement.
    """
    objectives = plr.DecoupledObjectives([
        plr.Objective.from_data(actions=cost_actions, values=cost_values, maximize=False,
                                name=COST_NAME, action_bounds=list(CURVE_X_RANGE)),
        plr.Objective.from_data(actions=comfort_actions, values=comfort_values, maximize=True,
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

        legend = self._legend(title, front_axes, landscape)

        self.play(Write(title, run_time=1.0))
        self.play(Write(legend, run_time=0.8))
        self.play(Create(true_front, run_time=1.2))
        self.wait(0.5)

        self._measure(metabolic_axes, comfort_axes, front_axes)
        self.wait(1.0)

    def _measure(self, metabolic_axes, comfort_axes, front_axes):
        """The seed design, then one acquisition-chosen action at a time.

        Both objectives are measured in one call per action, so a pair of dots is
        one visit to one controller -- except at `MULTI_TRIAL`, where comfort gets
        `MULTI_N` readings against cost's one, so the two are tracked as separate
        per-objective arrays throughout rather than one shared `(n, 2)` table. The
        fits happen here, while the scene is being built, and each one is a couple
        of exact GPs over at most N_POINTS points of cost.
        """
        axes = (metabolic_axes, comfort_axes)
        seed_actions = sobol_actions(N_SEED, SAMPLE_SEED)
        seed_values = NOISY_TRUTH(seed_actions)
        cost_actions, cost_values = seed_actions, seed_values[:, 0]
        comfort_actions, comfort_values = seed_actions, seed_values[:, 1]

        for n in range(N_SEED):
            self.play(FadeIn(self._dots(axes, seed_actions[n, 0], seed_values[n]),
                             scale=0.5, run_time=0.45))

        gp = fit(cost_actions, cost_values, comfort_actions, comfort_values)
        drawn = self._draw_fit(axes, front_axes, gp, None)
        self.play(Write(self._labels(comfort_axes, front_axes), run_time=0.8))
        query = None
        box = None

        for step in range(N_POINTS - N_SEED):
            trial = N_SEED + step + 1
            action = self._query(gp, step)
            lines = VGroup(*(self._query_line(ax, action) for ax in axes))
            if query is None:
                self.play(Create(lines, run_time=0.6))
                query = lines
            else:
                # Transform moves the line already on screen, so `query` stays the
                # mobject being animated and `lines` is only its target. The
                # MULTI_TRIAL box, if the last step drew one, goes with it: it
                # marks that one trial's readings, not a permanent fixture.
                anims = [Transform(query, lines, run_time=0.6)]
                if box is not None:
                    anims.append(FadeOut(box, run_time=0.6))
                    box = None
                self.play(*anims)
            self.wait(0.3)

            if trial == MULTI_TRIAL:
                # MULTI_N repeats of the same action rather than MULTI_N different
                # ones, so the dots land on top of each other -- the same controller,
                # rated more than once.
                multi_actions = np.full((MULTI_N, 1), action)
                multi_values = NOISY_TRUTH(multi_actions)
                cost_measured = float(NOISY_TRUTH(np.array([[action]]))[0, 0])

                cost_actions = np.vstack([cost_actions, [[action]]])
                cost_values = np.append(cost_values, cost_measured)
                comfort_actions = np.vstack([comfort_actions, multi_actions])
                comfort_values = np.append(comfort_values, multi_values[:, 1])

                comfort_dots = VGroup(*(
                    self._dot(comfort_axes, a, v)
                    for a, v in zip(multi_actions[:, 0], multi_values[:, 1])
                ))
                box = self._multi_box(comfort_axes, multi_actions[:, 0], multi_values[:, 1])
                self.play(
                    FadeIn(self._dot(metabolic_axes, action, cost_measured),
                          scale=0.5, run_time=0.45),
                    FadeIn(comfort_dots, scale=0.5, run_time=0.45),
                )
                self.play(Create(box, run_time=0.5))
            else:
                measured = NOISY_TRUTH(np.array([[action]]))
                cost_actions = np.vstack([cost_actions, [[action]]])
                cost_values = np.append(cost_values, measured[0, 0])
                comfort_actions = np.vstack([comfort_actions, [[action]]])
                comfort_values = np.append(comfort_values, measured[0, 1])
                self.play(FadeIn(self._dots(axes, action, measured[0]),
                                 scale=0.5, run_time=0.45))

            gp = fit(cost_actions, cost_values, comfort_actions, comfort_values)
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
        return VGroup(*(self._dot(ax, action, values[i]) for i, ax in enumerate(axes)))

    def _dot(self, axes, action, value):
        """One measurement, as a dot on one action plot."""
        return Dot(axes.c2p(action, value), radius=DOT_RADIUS, color=SAMPLE_COLOR)

    def _multi_box(self, axes, actions, values):
        """A rounded box around several dots on one action plot, marking
        `MULTI_TRIAL`: the one trial where an objective gets more than one
        reading. `actions` and `values` are the same arrays the dots inside it
        were drawn from, so the box is sized off their own screen coordinates
        rather than a placed guess.
        """
        points = np.array([axes.c2p(a, v) for a, v in zip(actions, values)])
        low, high = points.min(axis=0), points.max(axis=0)
        box = RoundedRectangle(
            width=(high[0] - low[0]) + 2 * MULTI_BOX_PAD,
            height=(high[1] - low[1]) + 2 * MULTI_BOX_PAD,
            corner_radius=BOX_RADIUS,
            stroke_color=MULTI_BOX_COLOR,
            stroke_width=MULTI_BOX_STROKE,
            fill_opacity=0.0,
        )
        box.move_to((low + high) / 2)
        return box

    def _query_line(self, axes, action):
        """Where the next query lands, as a line up one action plot."""
        return Line(
            axes.c2p(action, 0.0),
            axes.c2p(action, AXIS_Y_MAX),
            color=GAIN_COLOR,
            stroke_width=QUERY_STROKE,
        )

    def _legend(self, title, front_axes, landscape):
        """The key, two entries per row, sized to whatever fits above
        `landscape`'s own top and below `title`, and centered in it.

        Built at `LEGEND_FONT_SIZE` and then uniformly shrunk, so the same code
        keeps working if the available space or the panel width changes rather
        than a size picked by hand for this one layout.
        """
        row1 = VGroup(
            self._legend_entry(Dot(radius=DOT_RADIUS, color=SAMPLE_COLOR), "measurement"),
            self._legend_entry(
                DashedLine(LEFT * LEGEND_SWATCH_LEN / 2, RIGHT * LEGEND_SWATCH_LEN / 2,
                           color=INK, stroke_width=MEAN_STROKE, dash_length=0.06),
                "posterior mean"),
        ).arrange(RIGHT, buff=LEGEND_COL_BUFF)
        row2 = VGroup(
            self._legend_entry(
                Line(LEFT * LEGEND_SWATCH_LEN / 2, RIGHT * LEGEND_SWATCH_LEN / 2,
                     color=GAIN_COLOR, stroke_width=QUERY_STROKE),
                "next query"),
            self._legend_entry(
                RoundedRectangle(width=LEGEND_SWATCH_LEN, height=LEGEND_SWATCH_LEN * 0.6,
                                  corner_radius=0.05, stroke_color=MULTI_BOX_COLOR,
                                  stroke_width=MULTI_BOX_STROKE * 0.5, fill_opacity=0.0),
                "repeated rating"),
        ).arrange(RIGHT, buff=LEGEND_COL_BUFF)
        entries = VGroup(row1, row2).arrange(DOWN, buff=LEGEND_ROW_BUFF)

        top = title.get_bottom()[1]
        bottom = landscape.get_top()[1] + LEGEND_CLEARANCE
        target_w = front_axes.width * LEGEND_MARGIN
        target_h = (top - bottom) * LEGEND_MARGIN
        entries.scale(min(target_w / entries.width, target_h / entries.height, 1.0))
        entries.move_to([front_axes.get_center()[0], (top + bottom) / 2, 0])
        return entries

    def _legend_entry(self, swatch, text):
        """One swatch and its label, the label to the swatch's right."""
        label = Tex(text, font_size=LEGEND_FONT_SIZE, color=INK)
        label.next_to(swatch, RIGHT, buff=LEGEND_LABEL_BUFF)
        return VGroup(swatch, label)

    def _draw_fit(self, axes, front_axes, gp, drawn):
        """The bands, means, predicted front and estimated Pareto set of one fit,
        morphed from the last.

        Returns what to pass back as `drawn` at the next fit; `None` means nothing
        is on screen yet, so everything is created rather than transformed.
        """
        mu, std, front = posterior(gp)
        bands = VGroup(*(self._band(ax, mu[:, i], std[:, i], color)
                         for i, (ax, color) in enumerate(zip(axes, COLORS))))
        means = VGroup(*(self._mean(ax, mu[:, i]) for i, ax in enumerate(axes)))
        predicted = self._front(front_axes, mu, front)
        sets = VGroup(*(self._pareto_set(ax, front) for ax in axes))

        if drawn is None:
            self.play(FadeIn(bands, run_time=0.8), Create(means, run_time=0.8),
                      Create(predicted, run_time=0.8), Create(sets, run_time=0.8))
        else:
            self.play(
                Transform(drawn[0], bands, run_time=0.8),
                ReplacementTransform(drawn[1], means, run_time=0.8),
                ReplacementTransform(drawn[2], predicted, run_time=0.8),
                ReplacementTransform(drawn[3], sets, run_time=0.8),
            )
            bands = drawn[0]     # Transform leaves the original on screen
        self.wait(0.25)
        return (bands, means, predicted, sets)

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
        non-dominated, one dot per action rather than a connecting line.

        The GP picks the actions; the dots sit where those actions really land, so
        the gap to the true front is the cost of the model's mistakes rather than of
        its posterior mean being off. `mu` is unused here for that reason. `front`
        is already ordered along the predicted front (`posterior`'s doc), so a dot's
        position in that order -- not its coordinates -- sets its color, from
        `COST_COLOR` to `COMFORT_COLOR`. A line through the true coordinates would
        zigzag wherever the predicted and true orderings disagree; dots show that
        disagreement instead of papering over it with an extra stroke.
        """
        n = len(front)
        weights = np.linspace(0.0, 1.0, n) if n > 1 else [0.0]
        return VGroup(*(
            Dot(axes.c2p(cost(GRID[i]), comfort(GRID[i])), radius=FRONT_DOT_RADIUS,
                color=COST_COLOR.interpolate(COMFORT_COLOR, w))
            for i, w in zip(front, weights)
        ))

    def _pareto_set(self, axes, front):
        """The estimated Pareto set on one action axis, in red: the same red line
        `ParetoScene` draws for the true set, over the action range the posterior
        -- not the truth -- calls non-dominated.

        `front` indexes `GRID`, but not in grid order (`posterior`'s doc orders it
        along the front instead), so it is re-sorted here to find where it is
        contiguous. A posterior front need not be a single interval, so each
        contiguous run of grid indices gets its own segment rather than one line
        spanning gaps the model itself does not call Pareto optimal.
        """
        order = np.sort(front)
        runs = np.split(order, np.where(np.diff(order) > 1)[0] + 1)
        return VGroup(*(
            Line(axes.c2p(GRID[run[0]], 0.0), axes.c2p(GRID[run[-1]], 0.0),
                 color=SET_COLOR, stroke_width=SET_LINE_WIDTH)
            for run in runs
        ))

    def _labels(self, comfort_axes, front_axes):
        """Captions naming the surrogate Pareto set and front, written once.

        Placed at the true front's own location (`front_actions`, `POINTER_AT`)
        rather than tracked to the estimate: the estimate is what moves fit to
        fit and settles onto the truth, so anchoring the captions to where it
        settles keeps them legible throughout instead of chasing every step.
        """
        start, end = front_actions()
        set_line = Line(comfort_axes.c2p(start, 0.0), comfort_axes.c2p(end, 0.0))
        set_label = Tex(
            r"Surrogate Pareto Set $\mathcal{P}_{\hat{\bm{f}}}$",
            font_size=SET_LABEL_SIZE, color=SET_COLOR,
        )
        set_label.next_to(set_line, DOWN, buff=0.15)

        front_label = Tex(
            r"Surrogate Pareto Front $\mathcal{F}_{\hat{\bm{f}}}$",
            font_size=POINTER_LABEL_SIZE, color=INK,
        )
        front_label.move_to(front_axes.c2p(*POINTER_AT))
        middle = sum(front_actions()) / 2.0
        arrow = Arrow(
            front_label.get_top(),
            front_axes.c2p(cost(middle), comfort(middle)),
            buff=0.12,
            color=INK,
            stroke_width=3.0,
            max_tip_length_to_length_ratio=0.15,
        )
        return VGroup(set_label, front_label, arrow)

