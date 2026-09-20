# ICRA videos

Manim scenes for the paper. Rendered with Manim Community v0.21.0 in the
`pypolar` conda environment; nothing here imports `pypolar`.

```
video/
├── style.py        # palette, box geometry, manim config, box/arrow helpers
├── objectives.py   # the two objective curves, shared by every scene
├── clip.py         # VideoClip: a video file drawn as frames inside a scene
├── hilo.py         # HiloScene: device and optimization exchanging feedback
├── pareto.py       # ParetoScene: the same objectives, and their front
├── subjects/       # subject footage
├── render.sh       # one scene at 1080p60
└── media/          # manim output (images/, videos/)
```

## Render

```bash
conda activate pypolar
video/render.sh                       # HiloScene at -qh (1080p60)
video/render.sh ParetoScene           # the other scene, same quality
video/render.sh HiloScene -ql         # draft, 480p15
manim -ql -s video/hilo.py HiloScene  # last frame only, as a PNG
```

`render.sh` maps a scene name to its file, so it takes the scene rather than the
path; anything else it does not know about it refuses rather than guessing.

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

`objectives.py` holds the two curves themselves and no manim: metabolic cost is
a bowl at `OPTIMUM_A` and comfort a hump at `COMFORT_OPTIMUM_A`, both from one
`parabola`. Both scenes import them, so the two videos cannot drift apart.

## HiloScene

`hilo.sample_measurements()` places the five actions the dot visits and
their costs: the offset from `OPTIMUM_A` shrinks by `SAMPLE_DECAY` each step and
falls on a random side of it, and each cost carries `SAMPLE_NOISE_STD` of
Gaussian noise, so the points scatter off the curve. Both come from one seeded
generator, so every render is identical. Each visited point stays on screen at
`TRAIL_OPACITY` once the dot moves on, until the comfort curve is drawn and
`_search`'s returned `samples` group fades out with it, leaving the two optima,
their dashed lines and the gap arrow. The dot finishes on the curve's optimum
rather than on a noisy value, and the star appears over it.

`hilo.CONVEX` picks the cost curve's direction: `True` draws a bowl with
its minimum starred, `False` a hump with its maximum starred. `hilo.comfort()`
is the same parabola flipped the other way and peaked at `COMFORT_OPTIMUM_A`,
so the two objectives disagree about which action is best.

Both curves share one set of axes with a twin y axis: the left y axis is
metabolic cost in `style.COST_COLOR`, and `_objective(twin=True)` copies it to
`CURVE_X_RANGE[1]` for comfort in `style.COMFORT_COLOR`, each axis and its
rotated label taking its curve's color. The x axis runs on to `AXIS_X_MAX` so
its tip clears that second y axis. Both curves are drawn against the same
`y_range`, which is what lets one dashed line at the cost optimum be read
against either.

`COST_WIDTH` and `COMFORT_WIDTH` narrow the parabolas: each is that fraction of
the width that would put `CURVE_PEAK` at the far end of the x range. Narrow
curves leave the axes before the ends of `CURVE_X_RANGE`, so `curve_domain()`
returns the interval where each one is still on screen and the curve is plotted
only there, rather than being clipped flat against the top. Comfort is the wider
of the two so that its value at the cost optimum lands mid-plot, clear of the
cost curve. `SAMPLE_SPREAD` is set to keep the sampled actions inside the
narrowed cost curve's own domain.

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
interpolation between the two stars. Both curves are parabolas, so subtracting
the comfort hump from the cost bowl leaves a sum of two upward parabolas whose
minimum is the curvature-weighted average `(w_1 k_c a_c + w_2 k_f a_f) / (w_1 k_c
+ w_2 k_f)`; that runs from the cost optimum at `w = [1, 0]` to the comfort
optimum at `w = [0, 1]`, and its speed along the way is set by the two
curvatures rather than by `w` alone. A second `\bm{a}_{\bm{w}}` rides the x axis
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
`objectives.attainable_actions()` — the actions where both curves are inside
their own axes, which by construction of `curve_domain` runs from where comfort
reaches zero to where cost reaches `CURVE_Y_MAX`. The whole landscape is drawn
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
