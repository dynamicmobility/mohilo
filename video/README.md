# ICRA videos

Manim scenes for the paper. Rendered with Manim Community v0.21.0 in the
`pypolar` conda environment. The objectives and the GP fit to them come from
`pypolar` itself; the drawing does not.

```
video/
├── style.py        # palette, box geometry, manim config, box/arrow helpers
├── objectives.py   # the two objectives, as one pypolar groundtruth
├── clip.py         # VideoClip: a video file drawn as frames inside a scene
├── hilo.py         # HiloScene: device and optimization exchanging feedback
├── pareto.py       # ParetoScene: the same objectives, and their front
├── mogp.py         # MogpScene: a DecoupledMOGP learning them, point by point
├── subjects/       # subject footage
├── render.sh       # one scene at 1080p60
└── media/          # manim output (images/, videos/)
```

## Render

```bash
conda activate pypolar
video/render.sh                       # every scene at -qh (1080p60)
video/render.sh ParetoScene           # just that one
video/render.sh HiloScene -ql         # draft, 480p15
video/render.sh -ql                   # every scene as a draft
manim -ql -s video/hilo.py HiloScene  # last frame only, as a PNG
```

`render.sh` maps a scene name to its file, so it takes the scene rather than the
path, and refuses a name it does not know rather than guessing. Naming no scene
renders all of them; a first argument starting with `-` is read as a flag rather
than a scene, so flags alone apply to the whole set.

Output lands in `video/media/`, set by `style.py` so it does not depend on the
directory the command is run from.

## Notes

Every label is `Tex` or `MathTex`, so all type is Computer Modern and matches the
paper. This needs a LaTeX toolchain plus `dvisvgm`, which converts the DVI to the
SVG manim draws; BasicTeX does not ship `dvisvgm`, so it comes from
`brew install dvisvgm`. `style.TEX_TEMPLATE` adds `bm` to the preamble, which is
what `style.action_label()` typesets `\bm{a}` with.

The scene renders light on black: `style.BG` is the background and `style.INK`
every label. Each objective keeps one color across both videos — metabolic cost
`style.COST_COLOR` (blue), comfort `style.COMFORT_COLOR` (violet) — which leaves
`style.GAIN_COLOR` and `style.LOSS_COLOR` free to mean only gain and loss in the
Pareto scene. Comfort is violet rather than red for exactly that reason, and not
amber because `style.STAR_COLOR` already marks the optima in gold.

`objectives.py` holds the two objectives and no manim. They are not drawn curves
but a pypolar groundtruth: one `SyntheticOracleParams(func='IdealPoint', dim=1)`
builds a `MOSyntheticOracle` of two bowls sharing the action box `CURVE_X_RANGE`,
metabolic cost bottoming out at `OPTIMUM_A` and comfort peaking at
`COMFORT_OPTIMUM_A`. Comfort is a hump because its weight is *negative*, which is
how `IdealPoint` flips a bowl; direction is otherwise an `Objective`'s business,
not a truth's. `hilo/shared/simulation.py` builds the live experiment's
groundtruth the same way, in three action dimensions instead of one.

Both bowls are **bounded**: `low` and `high` squash them through a tanh into a
band, so every value lies in `[CURVE_FLOOR, CURVE_PEAK]` by construction. Three
things follow. Nothing needs clipping to the axes, so every curve is drawn over
the whole box and `attainable_actions()` returns just that. `CURVE_Y_MAX` is only
headroom for the axis tips, not a threshold. And the curves are no longer
quadratics, which is why `scalarized_argmin` is a bounded `scipy.optimize.
minimize_scalar` rather than a closed form — checked against a 4001-point grid
argmin across the whole simplex, where the two agree to the grid's resolution and
the path between the optima is monotone.

Using a truth rather than a formula is what lets a scene *measure*: `measure()`
returns a noisy reading of both objectives at the same actions, drawn from the
oracle at `NOISE_STD` of each objective's own spread. `HiloScene`'s search scatter
and `MogpScene`'s data both come from there, so one observation model feeds every
video and the GP is fit under the noise a viewer can see.

## HiloScene

`hilo.sample_measurements()` places the five actions the dot visits and
their costs: the offset from `OPTIMUM_A` shrinks by `SAMPLE_DECAY` each step and
falls on a random side of it, and the costs come from `objectives.measure()`, so
the points scatter off the curve by the oracle's own noise rather than by a
figure the scene invents. Both streams are seeded, so every render is identical. Each visited point stays on screen at
`TRAIL_OPACITY` once the dot moves on, until the comfort curve is drawn and
`_search`'s returned `samples` group fades out with it, leaving the two optima,
their dashed lines and the gap arrow. The dot finishes on the curve's optimum
rather than on a noisy value, and the star appears over it.

Both curves share one set of axes with a twin y axis: the left y axis is
metabolic cost in `style.COST_COLOR`, and `_objective(twin=True)` copies it to
`CURVE_X_RANGE[1]` for comfort in `style.COMFORT_COLOR`, each axis and its
rotated label taking its curve's color. The x axis runs on to `AXIS_X_MAX` so
its tip clears that second y axis. Both curves are drawn against the same
`y_range`, which is what lets one dashed line at the cost optimum be read
against either.

Each objective is one color throughout: the device sends metabolic cost up a
blue arrow at `COST_FEEDBACK_Y` from the moment the loop is drawn, and comfort
arrives later up a red one at `COMFORT_FEEDBACK_Y`, each arrow and label taking
its objective's color through the `color` argument on `style.flow_arrow` and
`style.arrow_label`. The controller arrow back to the device stays neutral.

Once the comfort arrow arrives, the comfort axis, label and curve are drawn. The
dashed scan line already spans the plot at the cost optimum, so it meets the
comfort curve away from comfort's own peak; a dot marks that crossing, a second
dashed line rises at `COMFORT_OPTIMUM_A`, and a double arrow at `GAP_ARROW_Y`
spans the two. `GAP_ARROW_Y` sits above both curves, which is the only band
between the two dashed lines that neither curve crosses.

Last, `_scalarize` writes the scalarized objective in the band below the boxes
and sweeps its weights. `w` is on the simplex, so one `ValueTracker` carries
`w_1` and the readout shows `1 - w_1` beside it, and the solid line is at
`scalarized_argmin(w_1)` — the argmin of the written objective itself, not an
interpolation between the two stars. It runs from the cost optimum at
`w = [1, 0]` to the comfort optimum at `w = [0, 1]`, and its speed along the way
is set by the two curvatures rather than by `w` alone. The solve is numeric, for
the reason the Notes give. A second `\bm{a}_{\bm{w}}` rides the x axis
under that line, which is why the axis's own `\bm{a}` label sits at the right end
of the axis rather than centered beneath it. The weights are `DecimalNumber`s dropped
into a single `MathTex`'s hidden slots: one LaTeX expression keeps the brackets
and comma on the numbers' baseline, and a `DecimalNumber` re-renders cached
digit glyphs each frame where a new `MathTex` would recompile LaTeX. `argmin` is
not a LaTeX primitive, so `style.TEX_TEMPLATE` declares it as an operator.

The sweep runs past the end of the footage, so `clip.play(loop=True)` keeps the
subject walking rather than freezing on the last frame.

manim renders mobjects, not video streams, so `clip.VideoClip` decodes a file to
PNG frames with ffmpeg and swaps them into an `ImageMobject` one at a time. The
frames are cached under `media/frames/`, keyed by crop, pixel height and fps, so
the decode happens once per render resolution. `CLIP_CROP` in `hilo.py` is
the ffmpeg `w:h:x:y` that strips the pillarbox bars from the portrait footage.
The scene runs at least as long as the clip does, and the clip loops to cover
the rest.

## ParetoScene

The same two objectives, read the other way round. The left column plots each
against the action, stacked on one shared x axis; the right panel plots them
against each other, which is the objective space the front lives in.

The attainable curve there is `(cost(a), comfort(a))` traced over
`objectives.attainable_actions()` — the whole action box, since a bounded bowl
never leaves its band. The whole landscape is drawn
dim first, then the bright front is drawn over the stretch of it spanned by
`front_actions()`, the interval between the two single-objective optima, and an
arrow from the middle of the panel names it `\mathcal{F}`. Its tip is the front's
own midpoint in action, computed rather than placed, so it follows the curve if
either optimum moves. That interval is the front because below `OPTIMUM_A` both
objectives are worse than they are at `OPTIMUM_A` — cost is higher and comfort
lower — so those points are dominated, and the same holds above
`COMFORT_OPTIMUM_A`.

`A1` and `A2` are two controllers on that front, and the order they arrive in is
the argument. `a_1` is drawn in all three plots at once: one dashed line
crossing both left plots at its action, a dot on each curve, and the point in
objective space whose coordinates those two dots are. Then the two dotted legs
leave it, green first, and `a_2` is drawn last where the red leg ends. The legs
are the sides of the right triangle spanning the two points, so each is the
change in one objective alone, and they carry no labels: the colors are the whole
statement. Going up first corners at `(cost(a_1), comfort(a_2))` — `a_2`'s
comfort at `a_1`'s cost — which is above the front and so attained by no
controller; the red leg is the walk back onto it. That is the point: the gain is
only there if the loss is paid, and a move that was all green would mean `a_1`
was dominated and not on the front at all.

## MogpScene

The same two objectives again, now being *learned*. No title and no annotation:
the two action plots on the left, and on the right the whole attainable curve in
one dim color, dominated arms and all, since this video is not about which part
of it is Pareto optimal.

Then the loop, `N_POINTS` actions from `objectives.sobol_actions()`. Each is one
visit to one controller: `measure()` is called once and returns a noisy reading of
*both* objectives, so the pair of dots that appears is one measurement, not two.
From `N_SEED` onward the scene refits `pypolar`'s own `DecoupledMOGP` after every
point and redraws the posterior. The first `N_SEED - 1` points arrive with no fit
behind them because a single measurement has no spread of its own to standardize
by and two badly underestimate it, so a band drawn from them would *widen* at the
third point rather than narrow.

`fit()` pins `action_bounds` on both objectives. Left unpinned, an `Objective`
normalizes actions by their measured range, so the frame — and with it every
lengthscale and the whole posterior — would shift each time a point is added, for
reasons having nothing to do with the new measurement. `posterior()` then reads
the fit twice: `posterior_at(grid)` in maximization space, where both objectives
are larger-is-better and `get_nondominated` can be asked which grid points are
non-dominated, and `posterior_at(grid, raw=True)` for the coordinates to draw. The
two calls are what `hilo/explore_front.py` makes for the same reason.

Everything the model believes is drawn in `style.INK`: a dashed posterior mean on
each left plot, and on the right the model's own front, its posterior mean at the
non-dominated grid points. The band is `BAND_STD` standard deviations of the
*latent* posterior — the model's uncertainty about the noiseless curve, which is
what a truth curve should sit inside — tinted in the objective's color, built from
a fixed `BAND_POINTS` vertices so one step's band morphs into the next, and
clipped to the axes since an early band is wider than the plot.

Expect the predicted front to sit slightly outside the true one even once the
band has collapsed. That is not a bug: taking the non-dominated subset of an
estimate selects the points where the estimate happened to be optimistic, so a
fitted front is biased outward.
