import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pypolar as plr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import CubicSpline, make_smoothing_spline
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
from dataclasses import dataclass
from hilo.create import create_pilot_regression
from config.hipexo import hipexo_pilot
import pandas as pd

NUM_QUERIES = 15
SEED        = 95
NOISE_FRAC  = 0.15   # noise_var as a fraction of the mean objective range

# roughness penalty for the smoothed Pareto path. At 1e-4 the grid zigzag is
# gone (mean |2nd difference| drops from ~1.2 grid steps to <0.15) while the
# curve still stays within about one grid step of the discrete front.
PATH_SMOOTHING = 1e-4

# divisor mapping the raw hip delay index into the configured action box
HIP_DELAY_SCALE = 160

# action space columns, in the order load_pilot_data returns them
ACTION_LABELS = [
    'hip flexion torque scale',
    'hip extension torque scale',
    f'hip delay idx / {HIP_DELAY_SCALE}',
]

# single-hue sequential ramp, truncated so the light end stays visible on white
TRADEOFF_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'blues_trunc', plt.get_cmap('Blues')(np.linspace(0.35, 1.0, 256))
)
OPTIMUM_COLOR   = '#c1443c'
MEASURED_COLOR  = '0.45'
OPTIMUM_MARKERS = ['*', 'P', 'D']


@dataclass
class Objective:
    """One metric column of the pilot data, and which direction is better."""
    name:     str    # label used when printing
    column:   str    # column in the metrics CSV
    maximize: bool   # True if larger is better

    @property
    def direction(self):
        """``'max'`` or ``'min'``, for labels."""
        return 'max' if self.maximize else 'min'

    @property
    def sign(self):
        """``+1`` if larger is better, ``-1`` otherwise."""
        return 1.0 if self.maximize else -1.0


@dataclass
class PilotFit:
    """A fitted MultiObjectiveGP over the pilot data, plus each objective's optimum."""
    objectives: list
    regression: object
    optimizer:  object
    best_idxs:  list    # action index of the optimum, per objective
    means:      list    # mean subtracted from each objective before fitting

    @property
    def label(self):
        """The objectives' names joined for titles."""
        return ' + '.join(obj.name for obj in self.objectives)

    @property
    def gps(self):
        """The per-objective fitted GPs."""
        return self.optimizer.gps

    @property
    def actions(self):
        """The ``(N, d)`` discretized action space."""
        return self.regression.action_space

    @property
    def best_actions(self):
        """The optimal action for each objective."""
        return self.actions[self.best_idxs]


# ---- data and fitting ------------------------------------------------------

def load_pilot_data(num_queries=NUM_QUERIES):
    """Metric values and the actions that produced them, first num_queries runs."""
    metrics_df = pd.read_csv('human_data/pilot_mohilo.csv', index_col='Run')
    actions_df = pd.read_csv('human_data/MH01_walk.csv', index_col='trial_name')

    actions = actions_df.to_numpy(dtype=float)[:num_queries, :-1]
    actions[:, 2] /= HIP_DELAY_SCALE
    return metrics_df.iloc[:num_queries], actions


def center_objectives(objectives, metrics_df):
    """Each objective column with its mean removed, and the means that were removed.
    Required because the GPs have a zero prior mean."""
    raw_values = [metrics_df[obj.column].to_numpy(dtype=float) for obj in objectives]
    means      = [float(v.mean()) for v in raw_values]
    return [v - m for v, m in zip(raw_values, means)], means


def pilot_config(objectives, values, noise_frac=NOISE_FRAC):
    """A copy of the pilot config with GP hyperparameters matched to the data's scale."""
    cfg = hipexo_pilot.model_copy(deep=True)   # each fit gets its own config
    objective_range = float(np.mean([v.max() - v.min() for v in values]))

    lengthscale, signal_var, precision = plr.derive_gp_hyperparams(
        domain_size    = float(np.max(cfg.problem.action_high
                                         - cfg.problem.action_low)),
        expected_range = objective_range,
        noise_var      = objective_range * noise_frac
    )
    cfg.num_objs                   = len(objectives)
    cfg.problem.precisions         = np.full(cfg.num_objs, precision)
    cfg.optimizer.signal_variances = [signal_var] * cfg.num_objs
    cfg.optimizer.length_scales    = [lengthscale] * cfg.num_objs
    return cfg


def fit_pilot(objectives, metrics_df, actions, seed=SEED, noise_frac=NOISE_FRAC):
    """Fit one MultiObjectiveGP over the given objectives and locate each optimum."""
    rng           = np.random.default_rng(seed)
    values, means = center_objectives(objectives, metrics_df)
    cfg           = pilot_config(objectives, values, noise_frac)

    regression, optimizer = create_pilot_regression(rng, cfg=cfg)
    for i in range(len(actions)):
        regression.add_feedback(actions[i], [v[i] for v in values])

    optimizer.setup(
        action_space = regression.action_space,
        regressions  = regression.get_regression_data()
    )
    optimizer.fit()

    best_idxs = [
        int(np.argmax(gp.mu) if obj.maximize else np.argmin(gp.mu))
        for obj, gp in zip(objectives, optimizer.gps)
    ]
    return PilotFit(objectives, regression, optimizer, best_idxs, means)


def report(fit):
    """Print the optimum for each objective and how far apart the optima are."""
    actions  = fit.best_actions
    width    = max(len(obj.name) for obj in fit.objectives)
    diagonal = float(np.linalg.norm(fit.actions.max(axis=0) - fit.actions.min(axis=0)))

    print(f'\n=== {fit.label} ===')
    print(f'action space: {fit.actions.shape}')

    # prior std is k(x, x) + jitter, the same for every action, so it is the
    # baseline std() decays from as data comes in
    for obj, gp in zip(fit.objectives, fit.gps):
        print(f'  prior std {obj.name:<{width}} = {np.sqrt(gp.prior_var()):.4f}')

    for obj, gp, idx, action, mean in zip(fit.objectives, fit.gps,
                                          fit.best_idxs, actions, fit.means):
        print(f'  best {obj.name:<{width}} ({obj.direction}): {action} '
              f'(mu = {gp.mu[idx] + mean:.4f}, std = {gp.std()[idx]:.4f})')

    # how far apart the optima are, relative to the box diagonal (the furthest
    # two points in the action space can possibly be)
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            separation = float(np.linalg.norm(actions[i] - actions[j]))
            pair = f'{fit.objectives[i].name} <-> {fit.objectives[j].name}'
            print(f'  separation {pair}: {separation:.4f} / {diagonal:.4f} '
                  f'= {100 * separation / diagonal:.2f}% of the diagonal')


# ---- Pareto front geometry -------------------------------------------------

def maximization_values(fit):
    """The fitted GP means as an ``(N, m)`` matrix in maximization space."""
    return np.column_stack([obj.sign * gp.mu
                            for obj, gp in zip(fit.objectives, fit.gps)])


def ordered_front(fit):
    """The two-objective Pareto front walked from one optimum to the other, as
    ``(idxs, actions, objs, position)`` with ``position`` in ``[0, 1]``."""
    if len(fit.objectives) != 2:
        raise ValueError('the Pareto front is only totally ordered for two '
                         f'objectives; got {len(fit.objectives)}')

    values_max = maximization_values(fit)
    front_idxs = plr.get_nondominated(values_max)

    # increasing f1, ties broken by decreasing f2
    order      = np.lexsort((-values_max[front_idxs, 1], values_max[front_idxs, 0]))
    front_idxs = front_idxs[order]

    objs = np.column_stack([gp.mu[front_idxs] + mean
                            for gp, mean in zip(fit.gps, fit.means)])
    position = (np.linspace(0, 1, len(front_idxs)) if len(front_idxs) > 1
                else np.zeros(len(front_idxs)))

    return front_idxs, fit.actions[front_idxs], objs, position


def smooth_path(path, position, smoothing, num=300, pin_ends=True):
    """A cubic smoothing spline through the ordered Pareto set, resampled at ``num``
    points. ``smoothing`` is the roughness penalty; 0 interpolates exactly."""
    position_dense = np.linspace(position[0], position[-1], num)

    if len(path) < 5:            # too few knots for a cubic fit either way
        return position_dense, np.column_stack(
            [np.interp(position_dense, position, path[:, d])
             for d in range(path.shape[1])])
    if smoothing <= 0:
        return position_dense, CubicSpline(position, path, axis=0)(position_dense)

    path_dense = np.column_stack([
        make_smoothing_spline(position, path[:, d], lam=smoothing)(position_dense)
        for d in range(path.shape[1])
    ])

    if pin_ends:
        # the correction is linear in position, so it costs nothing under the
        # second-derivative penalty
        progress = ((position_dense - position_dense[0])
                    / (position_dense[-1] - position_dense[0]))[:, None]
        path_dense += ((1 - progress) * (path[0] - path_dense[0])
                       + progress * (path[-1] - path_dense[-1]))

    return position_dense, path_dense


def nondominated_fraction(front_objs, dense_objs, objectives):
    """The fraction of the smoothed curve that the discrete Pareto front does not dominate."""
    sign      = np.array([obj.sign for obj in objectives])
    front_max, dense_max = front_objs * sign, dense_objs * sign
    dominated = (np.all(front_max[None] >= dense_max[:, None], axis=2)
                 & np.any(front_max[None] >  dense_max[:, None], axis=2))
    return 1.0 - float(dominated.any(axis=1).mean())


def zoom_limits(path, actions, pad=0.7, min_extent=0.25):
    """A ``(2, d)`` box around the Pareto set, floored so thin dimensions stay visible."""
    lo, hi = path.min(axis=0), path.max(axis=0)
    spread = float((hi - lo).max())
    if spread == 0:                       # a single-point front
        spread = float(np.ptp(actions, axis=0).max()) / 20
    extent = np.maximum(hi - lo, min_extent * spread)
    center = 0.5 * (lo + hi)
    return np.array([center - pad * extent, center + pad * extent]), spread


@dataclass
class ParetoPath:
    """The ordered Pareto front plus its smoothed curve, as every panel draws it."""
    actions:        np.ndarray   # (k, d) front actions, in order
    objs:           np.ndarray   # (k, m) their objective values, un-centered
    position:       np.ndarray   # (k,)   slider coordinate of each, in [0, 1]
    actions_dense:  np.ndarray   # (num, d) the smoothed curve
    position_dense: np.ndarray   # (num,)
    smoothing:      float

    @property
    def segment_colors(self):
        """One color per segment of the smoothed curve, at its midpoint position."""
        return TRADEOFF_CMAP(0.5 * (self.position_dense[:-1] + self.position_dense[1:]))

    @property
    def segments(self):
        """The smoothed curve as ``(num - 1, 2, d)`` line segments."""
        return np.stack([self.actions_dense[:-1], self.actions_dense[1:]], axis=1)


def pareto_path(fit, smoothing=PATH_SMOOTHING, num=300):
    """The ordered Pareto front of the fit, with a smoothing spline through it."""
    _, actions, objs, position = ordered_front(fit)
    position_dense, actions_dense = smooth_path(actions, position, smoothing, num)
    return ParetoPath(actions, objs, position, actions_dense, position_dense, smoothing)


# ---- plotting helpers ------------------------------------------------------

def save_figure(fig, save_path):
    """Write the figure to save_path, creating the directory, and close it."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure to {save_path.resolve()}')


def style_action_axes(ax, limits, max_ticks=None, box_aspect=(1, 1, 1)):
    """Label, window and de-emphasize a 3D action-space axes."""
    for k, setter in enumerate((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel)):
        setter(ACTION_LABELS[k] if k < len(ACTION_LABELS) else f'action dim {k}',
               fontsize=9)
    for k, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(limits[0][k], limits[1][k])

    # recessive panes and grid, so the marks carry the figure
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 0.0))
        pane._axinfo['grid'].update(color='0.9', linewidth=0.6)
        if max_ticks:   # a tight zoom box otherwise packs its short axes with labels
            pane.set_major_locator(MaxNLocator(nbins=max_ticks))
    ax.set_box_aspect(box_aspect)
    ax.view_init(elev=22, azim=-58)


def draw_optima(ax, points, objectives, size=220, annotate=False, show_direction=True):
    """Mark each objective's optimum; shape distinguishes them so color stays free.
    ``show_direction`` appends ``(min)``/``(max)`` to the legend label."""
    for obj, point, marker in zip(objectives, points, OPTIMUM_MARKERS):
        label = f'best {obj.name} ({obj.direction})' if show_direction else f'best {obj.name}'
        ax.scatter(*np.atleast_1d(point), marker=marker, s=size, color=OPTIMUM_COLOR,
                   edgecolors='white', linewidths=1.0, zorder=5, label=label,
                   **({'depthshade': False} if ax.name == '3d' else {}))
        if annotate:
            ax.text(*point, f'  {obj.name}', fontsize=8, color=OPTIMUM_COLOR)


# ---- figures ---------------------------------------------------------------

def plot_pareto_set(fit, save_path):
    """Scatter the Pareto-optimal actions in the 3D action space, colored by
    tradeoff: dark favors the first objective, light the second."""
    actions    = fit.actions
    values_max = maximization_values(fit)
    front_idxs = plr.get_nondominated(values_max)
    front      = actions[front_idxs]

    # tradeoff position, normalized over the front alone so the ramp spans it
    weight   = values_max[front_idxs, 0]
    span     = float(weight.max() - weight.min())
    tradeoff = (weight - weight.min()) / span if span > 0 else np.zeros(len(weight))

    fig = plt.figure(figsize=(8.5, 7))
    ax  = fig.add_subplot(111, projection='3d')

    front_points = ax.scatter(*front.T, c=tradeoff, cmap=TRADEOFF_CMAP, vmin=0,
                              vmax=1, s=26, depthshade=False, linewidths=0, alpha=0.9)

    measured = actions[fit.regression.get_feedback_idxs()]
    ax.scatter(*measured.T, marker='x', color='0.25', s=55, linewidths=1.6,
               depthshade=False, label=f'measured actions (n = {len(measured)})')
    draw_optima(ax, fit.best_actions, fit.objectives, annotate=True)

    limits = np.array([actions.min(axis=0), actions.max(axis=0)])
    style_action_axes(ax, limits)
    ax.set_title(f'Pareto set in action space: {fit.label}\n'
                 f'{len(front)} of {len(actions)} actions non-dominated', fontsize=11)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    cbar = fig.colorbar(front_points, ax=ax, shrink=0.55, pad=0.12)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([
        f'favors {fit.objectives[1].name}' if len(fit.objectives) == 2 else 'low',
        f'favors {fit.objectives[0].name}',
    ], fontsize=8)
    cbar.ax.set_ylabel('tradeoff along the front', fontsize=9)
    cbar.outline.set_visible(False)

    save_figure(fig, save_path)


def draw_action_panel(ax, fit, path, limits, point_size, title, legend):
    """Draw the Pareto path and its context marks into one 3D axes, windowed to limits."""
    ax.add_collection3d(Line3DCollection(
        path.segments, colors=path.segment_colors, linewidths=2, zorder=2,
        label=f'smoothed path (lam = {path.smoothing:g})'
    ))
    front_points = ax.scatter(*path.actions.T, c=path.position, cmap=TRADEOFF_CMAP,
                              vmin=0, vmax=1, s=point_size, zorder=3, depthshade=False,
                              edgecolors='white', linewidths=0.5,
                              label=f'Pareto set (n = {len(path.actions)})')

    # a 3D axes does not clip to its limits, so drop the marks outside the
    # window by hand rather than letting them draw over the zoomed box
    measured = fit.actions[fit.regression.get_feedback_idxs()]
    keep     = np.all((measured >= limits[0]) & (measured <= limits[1]), axis=1)
    if keep.any():
        ax.scatter(*measured[keep].T, marker='x', color=MEASURED_COLOR, s=45,
                   linewidths=1.4, depthshade=False,
                   label=f'measured actions (n = {int(keep.sum())})')

    draw_optima(ax, fit.best_actions, fit.objectives, size=230)

    style_action_axes(ax, limits, max_ticks=4,
                      box_aspect=tuple(limits[1] - limits[0]))   # true relative scale
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    return front_points


def draw_front_panel(ax, fit, path, dense_objs):
    """Draw the Pareto front in objective space, with the smoothed path over it."""
    kept = nondominated_fraction(path.objs, dense_objs, fit.objectives)
    ax.add_collection(LineCollection(
        np.stack([dense_objs[:-1], dense_objs[1:]], axis=1),
        colors=path.segment_colors, linewidths=2, zorder=2,
        label=f'smoothed path ({100 * kept:.0f}% non-dominated)'
    ))
    # small enough that the smoothed curve stays readable where they bunch up
    ax.scatter(path.objs[:, 0], path.objs[:, 1], c=path.position, cmap=TRADEOFF_CMAP,
               vmin=0, vmax=1, s=18, edgecolors='white', linewidths=0.4, zorder=3,
               label=f'Pareto set (n = {len(path.objs)})')

    ax.scatter(fit.regression.get_feedback_values(0) + fit.means[0],
               fit.regression.get_feedback_values(1) + fit.means[1],
               marker='x', color=MEASURED_COLOR, s=45, linewidths=1.4, zorder=1,
               label='measured objectives')

    # the walk's endpoints are the single-objective optima
    draw_optima(ax, [path.objs[-1], path.objs[0]], fit.objectives, size=230,
                show_direction=False)   # the axis labels already state it

    for obj, setter in zip(fit.objectives, (ax.set_xlabel, ax.set_ylabel)):
        setter(f'{obj.name} ({"higher" if obj.maximize else "lower"} is better)',
               fontsize=9)
    ax.set_title(f'Pareto front, ordered by {fit.objectives[0].name}', fontsize=10)
    ax.grid(color='0.92', linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)


def plot_pareto_path(fit, save_path, smoothing=PATH_SMOOTHING):
    """Three panels of the same ordered walk: the Pareto set in action space, a
    zoom on it, and the front in objective space, all colored by position."""
    path = pareto_path(fit, smoothing)

    fig      = plt.figure(figsize=(19, 6))
    space_ax = fig.add_subplot(131, projection='3d')
    zoom_ax  = fig.add_subplot(132, projection='3d')
    front_ax = fig.add_subplot(133)
    fig.subplots_adjust(wspace=0.45)   # 3D axis labels overhang their own cell

    full_limits  = np.array([fit.actions.min(axis=0), fit.actions.max(axis=0)])
    front_points = draw_action_panel(space_ax, fit, path, full_limits, 5,
                                     'Pareto set in action space', legend=True)

    zoom, spread = zoom_limits(path.actions, fit.actions)
    zoom_frac    = 100 * spread / (full_limits[1] - full_limits[0]).max()
    draw_action_panel(zoom_ax, fit, path, zoom, 45,
                      f'zoomed to the Pareto set ({zoom_frac:.0f}% of the action space)',
                      legend=False)

    # The line is the smoothed action path evaluated off grid through mu_at, so
    # it is the same curve the 3D panels draw. Connecting the discrete front
    # points instead would kink.
    dense_objs = np.column_stack([gp.mu_at(path.actions_dense) + mean
                                  for gp, mean in zip(fit.gps, fit.means)])
    draw_front_panel(front_ax, fit, path, dense_objs)

    cbar = fig.colorbar(front_points, ax=front_ax, shrink=0.85, pad=0.03)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([f'best {fit.objectives[1].name}',
                         f'best {fit.objectives[0].name}'], fontsize=8)
    cbar.ax.set_ylabel('position along the front (slider coordinate s)', fontsize=9)
    cbar.outline.set_visible(False)

    fig.suptitle(f'Pareto front as a path: {fit.label}', fontsize=12)
    save_figure(fig, save_path)


def main():
    metrics_df, actions = load_pilot_data()
    output_dir = Path(hipexo_pilot.save_dir)

    treadmill = [
        Objective('metabolic cost',   'Cost',              maximize=False),
        Objective('treadmill comfort', 'Comfort Treadmill', maximize=True),
    ]
    floor = [
        Objective('speed',         'Speed',         maximize=False),
        Objective('floor comfort', 'Comfort Floor', maximize=True),
    ]

    treadmill_fit = fit_pilot(treadmill, metrics_df, actions)
    floor_fit     = fit_pilot(floor, metrics_df, actions)

    for fit in (treadmill_fit, floor_fit):
        report(fit)

    plot_pareto_set(treadmill_fit, output_dir / 'treadmill_metabolic_comfort.svg')
    plot_pareto_set(floor_fit, output_dir / 'floor_speed_comfort.svg')

    plot_pareto_path(treadmill_fit, output_dir / 'treadmill_pareto_path.svg')
    plot_pareto_path(floor_fit, output_dir / 'floor_pareto_path.svg')


if __name__ == '__main__':
    main()
