"""IGD+ against query count for MO-HILBO and NSGA-II, at each objective count.

The same figure `final_dim.py` draws, over `objs<m>/` rather than `dim<n>/`:
same colors, same dashed NSGA-II, same query budget, no legend.
"""

from pathlib import Path

from scripts.icra.final_dim import figure

DATASET = Path('scripts/output/final_exp/obj-100')
OUTPUT  = DATASET / 'final_igd_plus.svg'
PREFIX  = 'objs'    # the condition subdirectory prefix
SYMBOL  = 'm'       # the label symbol the condition is named by


def main():
    figure(DATASET, PREFIX, SYMBOL, OUTPUT)


if __name__ == '__main__':
    main()
