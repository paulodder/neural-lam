# Standard library
import os
from argparse import ArgumentParser

# Third-party

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xa

# First-party
from neural_lam import vis
from bwdl.constants import DATASETS_DIR, GRAPHS_DIR


def main():
    """
    Pre-compute all static features related to the grid nodes
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="global_era5",
        help="Dataset to compute weights for (default: meps_example)",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=0,
        help="If fields should be plotted " "(default: 0 (false))",
    )
    args = parser.parse_args()

    fields_group_path = DATASETS_DIR / args.dataset / "fields.zarr"


if __name__ == "__main__":
    main()
