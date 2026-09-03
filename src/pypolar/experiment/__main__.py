"""Prints a saved run: `plr-print <path>`, or `python -m pypolar.experiment <path>`."""

import argparse
from pathlib import Path

from pypolar.experiment.dataset import ExperimentDataset


def main():
    parser = argparse.ArgumentParser(description='Print a saved ExperimentDataset.')
    parser.add_argument('path', type=Path, help='the run json file')
    args = parser.parse_args()

    dataset = ExperimentDataset.load(args.path)
    print(dataset)
    print()
    print(dataset.get_objectives())


if __name__ == '__main__':
    main()
