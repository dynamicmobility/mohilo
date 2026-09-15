"""Compile a LaTeX table to a cropped SVG, with the text drawn as outlines.

The ``table`` float and its caption are dropped, since a standalone page cannot
hold a float; only the contents (the ``tabular``) are drawn.

    python scripts/tex_to_svg.py table.tex                 # writes table.svg
    python scripts/tex_to_svg.py table.tex -o out.svg --preamble macros.tex
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

WRAPPER = r"""\documentclass[border=2pt]{standalone}
\usepackage{amsmath,amssymb,array,booktabs,multirow,makecell,graphicx}
\usepackage[table]{xcolor}
\renewenvironment{table}[1][]{}{}
\renewenvironment{table*}[1][]{}{}
\RenewDocumentCommand{\caption}{s o m}{}
%(preamble)s
\begin{document}
\input{%(body)s}
\end{document}
"""


def tex_to_svg(tex, svg, preamble=None):
    """Compile ``tex`` and write its cropped page to ``svg``.

    tex: Path to the LaTeX snippet. Compiled from its own directory, so relative
        ``\\input`` and ``\\includegraphics`` paths inside it resolve.
    svg: Path the SVG is written to.
    preamble: optional Path to a file of extra ``\\usepackage``/``\\newcommand``
        lines, inserted before ``\\begin{document}``.
    """
    tex = tex.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        wrapper = Path(tmp) / "wrapper.tex"
        wrapper.write_text(WRAPPER % {
            "preamble": rf"\input{{{preamble.resolve()}}}" if preamble else "",
            "body": tex.name,
        })
        run = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={tmp}", str(wrapper)],
            cwd=tex.parent, capture_output=True, text=True,
        )
        if run.returncode != 0:
            log = "\n".join(run.stdout.splitlines()[-40:])
            raise SystemExit(f"{log}\n\npdflatex failed on {tex}")
        subprocess.run(["pdftocairo", "-svg", str(Path(tmp) / "wrapper.pdf"), str(svg)],
                       check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tex", type=Path, help="LaTeX file holding the table")
    parser.add_argument("-o", "--output", type=Path,
                        help="SVG path (default: TEX with an .svg suffix)")
    parser.add_argument("--preamble", type=Path,
                        help="file of extra \\usepackage/\\newcommand lines")
    args = parser.parse_args()

    svg = args.output or args.tex.with_suffix(".svg")
    tex_to_svg(args.tex, svg, args.preamble)
    print(f"wrote {svg}")
