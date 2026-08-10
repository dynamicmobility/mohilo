# REMINDER

Ideas I thought about implementing but haven't had time to. This is a parking lot,
not a changelog — an entry stays here until the work is actually done, then it gets
deleted.

## How to add an entry

Every entry is a `###` heading with the short description, followed by four fields:

- **Time** — when the reminder was written, as `YYYY-MM-DD HH:MM TZ`. Absolute, never
  "yesterday" or "last week", so entries stay readable months later.
- **Priority** — `1`, `2`, or `3`, where **1 is the most important** and 3 the least.
  Always ask me for this; do not guess it.
  - **1 — most important.** Blocking or near-blocking. Something downstream is wrong
    or untrustworthy until this is done, so it comes before further work in that area.
  - **2 — medium.** Should happen soon, but current results are still usable without
    it. Typically a known misspecification whose effect I can reason around for now.
  - **3 — least important.** Nice-to-have: cleanup, deduplication, ergonomics.
    Revisit when there's spare time; nothing depends on it.
- **File** — the file (or files) where the change lands, or where the problem shows up.
  Use repo-relative paths, with line numbers if they help.
- **Why** — why this matters. Infer it from the description of the reminder if you can;
  ask if the reason isn't clear from what I said.

Newest entries go at the top.

---

### Standardize the objectives instead of only centering them

- **Time:** 2026-08-08 13:49 EDT
- **Priority:** 2
- **File:** `hilo/plot_pilot.py` (`Objective.__post_init__`, `Objectives.__post_init__`,
  `input_transform`/`output_transform`, and `pilot_config` at lines 212-227);
  `hilo/claude_plot_pilot.py:105` (`center_objectives`) does the same thing
- **Why:** The transform today is `y - mean` (plus a sign flip for the minimized
  objectives), which removes each objective's offset but leaves it on its own scale.
  Metabolic cost, 10m walk time, and the two comfort ratings have ranges that differ by
  orders of magnitude, and two things downstream assume they don't:
  1. **Shared GP hyperparameters.** `pilot_config` derives *one* `signal_var` and *one*
     `precision` from `objective_range`, the mean of the per-objective `max - min`, then
     broadcasts them across all objectives. A single amplitude prior is only correct if
     the objectives already share a scale; otherwise the wide objective is under-smoothed
     and the narrow one is over-smoothed by the same fit.
  2. **Hypervolume-based acquisition.** qNEHVI measures improvement as a volume in the
     raw objective space, so an objective with a larger spread contributes a
     proportionally larger share of every hypervolume box and dominates the acquisition.
     Reference-point inference from observed feedback inherits the same skew.
  Dividing by the per-objective standard deviation (or range) on top of the centering
  puts every objective in comparable units, which makes one set of shared
  hyperparameters and one reference point meaningful. `output_transform` has to invert
  the scaling as well as the shift, so the plotted values stay in physical units.
