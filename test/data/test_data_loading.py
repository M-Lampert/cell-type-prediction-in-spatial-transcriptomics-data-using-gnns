"""
Test data loading.
"""
# pylint: disable=missing-function-docstring, protected-access, redefined-outer-name

import pandas as pd
import pytest

from ctgnn.data.data_loading import get_spatial_data, iterate_datasets


def test_get_spatial_data():
    with pytest.raises(AssertionError):
        get_spatial_data(7)

    with pytest.raises(AssertionError):
        get_spatial_data("foo")

    cells, genes = get_spatial_data("intestine")

    assert isinstance(cells, pd.DataFrame)
    assert isinstance(genes, pd.DataFrame)
    assert len(cells) > 10
    assert len(list(cells.columns)) >= 4
    assert len(cells) == len(genes)


def test_iterate_datasets():
    _i = -1
    for _i, (name, cells, genes) in enumerate(iterate_datasets()):
        assert isinstance(name, str)
        assert isinstance(cells, pd.DataFrame)
        assert isinstance(genes, pd.DataFrame)

        # Some data quality checks
        assert len(cells) == len(genes)
        assert not cells.isnull().any().any()
        assert not genes.isnull().any().any()
        assert len(genes) == len(genes.index.unique())
        assert len(cells) == len(cells.index.unique())
        assert len(cells) == len(cells.join(genes, how="inner"))

        assert "cluster_id" in cells.columns
        assert "cell_type" in cells.columns
        assert "x" in cells.columns
        assert "y" in cells.columns
        assert "cell_id" == cells.index.name
        assert "cell_id" == genes.index.name

    # Check if there are 4 datasets
    assert _i == 3
