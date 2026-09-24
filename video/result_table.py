"""ResultTableScene: the paper's validation table, with a bounding box that
slides from the hypervolume columns to the pairwise-ordering columns.

Draws `video/media/Tex/result_table.tex`'s own `tabular` -- not the surrounding
`table` float or its caption, since a caption belongs on the paper page, not on
screen -- as one manim `Tex`. The box is built from that `Tex`'s own isolated
cells (`substrings_to_isolate`), not hand-measured coordinates, so it tracks
whichever cells the table actually holds rather than a copy of today's numbers.
"""

import re
import sys
from copy import deepcopy
from pathlib import Path

from manim import (
    ORIGIN,
    UP,
    Create,
    FadeIn,
    FadeOut,
    Scene,
    SurroundingRectangle,
    Tex,
    Transform,
    VGroup,
    Write,
)

# manim loads this file by path rather than as a package, so the repo root is
# put on sys.path for the imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video.style import INK, SCENE_TITLE_SIZE, STAR_COLOR, TEX_TEMPLATE, TITLE_EDGE_BUFF  # noqa: E402

TITLE = "Validation Data Across All Subjects"

TABLE_TEX = Path(__file__).resolve().parent / "media" / "Tex" / "result_table.tex"

# `result_table.tex` needs `makecell` for its column headers, on top of the
# `bm`/`argmin` preamble every other scene shares.
TABLE_TEMPLATE = deepcopy(TEX_TEMPLATE)
TABLE_TEMPLATE.add_to_preamble(r"\usepackage{makecell}")

HV_HEADER = "Hypervolume"
PO_HEADER = "Pairwise Ordering"
DATA_ROW = re.compile(r"^(?:MB|\w)\S* & (.+?) & (.+?) & (.+?) & (.+?) \\\\\s*$", re.M)
BOLD_CELL = re.compile(r"\\textbf\{(.+)\}")
MAKECELL = re.compile(r"\\makecell\{([^}]+)\}")

TABLE_WIDTH = 11.0
BOX_BUFF = 0.14
HOLD = 1.0


def tabular_source(path=TABLE_TEX):
    """The bare `\\begin{tabular}...\\end{tabular}` inside `path`."""
    text = path.read_text()
    return re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, re.S).group(0)


def cell_text(raw):
    """A data cell's own tex, with a `\\textbf{...}` wrapper (the Pareto
    hypervolume column) stripped so the isolated substring is the bare value
    `Tex` actually renders it as."""
    match = BOLD_CELL.fullmatch(raw)
    return match.group(1) if match else raw


def column_groups(tabular):
    """Every subject row's own cells plus that column's own `\\makecell`
    label, split `(hypervolume, pairwise)` -- read off the table rather than
    hard-coded, so a changed table changes what gets isolated too.

    The labels matter as much as the data cells: "Anti-Pareto" is wider than
    any number under it, so a box built from the data cells alone falls short
    of it on the right.
    """
    pareto_label, antipareto_label, cost_label, comfort_label = MAKECELL.findall(tabular)
    hypervolume, pairwise = [pareto_label, antipareto_label], [cost_label, comfort_label]
    for pareto, antipareto, cost, comfort in DATA_ROW.findall(tabular):
        hypervolume += [cell_text(pareto), cell_text(antipareto)]
        pairwise += [cell_text(cost), cell_text(comfort)]
    return hypervolume, pairwise


def isolated_group(table, names):
    """Every submobject of `table` whose isolated tex exactly matches one of
    `names` -- there may be several per name, since e.g. "100\\% (3/3)"
    repeats across subjects and each occurrence is its own match."""
    names = set(names)
    return VGroup(*(table.id_to_vgroup_dict[id_]
                     for tex, id_ in table.matched_strings_and_ids if tex in names))


class ResultTableScene(Scene):
    """The validation table alone, then a box around its Hypervolume columns
    that slides over to its Pairwise Ordering columns."""

    def construct(self):
        title = Tex(TITLE, font_size=SCENE_TITLE_SIZE, color=INK)
        title.to_edge(UP, buff=TITLE_EDGE_BUFF)
        self.play(Write(title, run_time=1.0))

        tabular = tabular_source()
        hv_cells, po_cells = column_groups(tabular)
        isolate = list(dict.fromkeys([HV_HEADER, PO_HEADER, *hv_cells, *po_cells]))

        table = Tex(tabular, tex_template=TABLE_TEMPLATE, substrings_to_isolate=isolate, color=INK)
        table.scale_to_fit_width(TABLE_WIDTH)
        table.move_to(ORIGIN)

        self.play(FadeIn(table))
        self.wait(HOLD)

        hv_box = SurroundingRectangle(
            isolated_group(table, [HV_HEADER, *hv_cells]), color=STAR_COLOR, buff=BOX_BUFF)
        po_box = SurroundingRectangle(
            isolated_group(table, [PO_HEADER, *po_cells]), color=STAR_COLOR, buff=BOX_BUFF)

        self.play(Create(hv_box))
        self.wait(HOLD)
        self.play(Transform(hv_box, po_box))
        self.wait(HOLD)

        self.play(FadeOut(title), FadeOut(table), FadeOut(hv_box))
