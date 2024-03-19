"""
This module provides functions to load the spatial data.
"""

from pathlib import Path
from typing import Iterator

import pandas as pd

__all__ = ["get_spatial_data", "iterate_datasets"]

file_path = Path(__file__)
project_root = file_path.parents[2]
spatial_data = project_root / "data"
dataset_dirs = {
    "intestine": "01_intestine",
    "embryo": "02_mouse_embryo",
    "hypothalamus": "03_hypothalamus",
    "brain": "04_mouse_brain",
}


def iterate_datasets() -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Returns an iterator that iterates over all available datasets.

    Yields:
        dataset_name: The name of the current dataset
        cells: cell positions in the tissue combined with the ground truth label
        genes: gene expression of each cell
    """
    for dataset in dataset_dirs:
        cells, genes = get_spatial_data(dataset)
        yield dataset, cells, genes


def get_spatial_data(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load spatial data. The coordinates will be normalized.

    Arguments:
        dataset: The dataset that should be loaded. Can be one of the following:
            intestine

    Returns:
        cells: cell positions in the tissue combined with the ground truth label
        genes: gene expression of each cell
    """
    assert isinstance(
        dataset, str
    ), f"Expected the dataset to be a string and not {type(dataset)}"
    dataset = dataset.lower().replace(" ", "_")

    assert (
        dataset in dataset_dirs
    ), f"The specified dataset '{dataset}' is not supported."
    dataset_dir = dataset_dirs[dataset]

    cell_coords = pd.read_parquet(spatial_data / dataset_dir / "cell_coords.parquet")
    cell_coords = (cell_coords - cell_coords.min()) / (
        cell_coords.max() - cell_coords.min()
    )
    cell_labels = pd.read_parquet(
        spatial_data / dataset_dir / "cluster_assignment.parquet"
    )
    genes = pd.read_parquet(spatial_data / dataset_dir / "cell_by_gene.parquet")

    cells = cell_coords.join(cell_labels)
    return cells, genes
