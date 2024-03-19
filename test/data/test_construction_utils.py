"""
Test the utility functions for graph construction.
"""
# pylint: disable=missing-function-docstring, protected-access, redefined-outer-name

import numpy as np
import pytest
import torch_geometric as pyg

from ctgnn.data.construction_utils import (
    calc_deg,
    construct_graph_by_degree,
    get_constructed_graph,
)
from ctgnn.data.graph_construction import construct_graph


def test_get_constructed_graph(mocker, positions, features, labels):
    # Patching needs to be done where it is used aka
    # when it is imported and not where it is defined
    # Although we use a different library this applies here to:
    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch
    mocker.patch("ctgnn.data.construction_utils.construct_graph", return_value="foo")
    mocker.patch(
        "ctgnn.data.construction_utils.construct_graph_by_degree",
        return_value=("foo", 42, 3),
    )
    mocker.patch("ctgnn.data.construction_utils.calc_deg", return_value=(3, "foo"))

    graph, param = get_constructed_graph(
        "knn", 5, positions=positions, features=features, labels=labels
    )
    assert graph == "foo"
    assert param["k"] == 3
    assert param["degree"] == 3

    graph, param = get_constructed_graph(
        "radius", 5, positions=positions, features=features, labels=labels
    )
    assert graph == "foo"
    assert param["radius"] == 42

    graph, param = get_constructed_graph(
        "delaunay", 5, positions=positions, features=features, labels=labels
    )
    assert graph == "foo"
    del param["degree"]
    assert not param

    graph, param = get_constructed_graph(
        "radius_delaunay", 5, positions=positions, features=features, labels=labels
    )
    assert graph == "foo"
    assert param["radius"] == 42

    graph, param = get_constructed_graph(
        "radius_delaunay",
        positions=positions,
        features=features,
        labels=labels,
        radius_parameter={"radius": 0.1},
    )
    assert graph == "foo"
    assert param["radius"] == 0.1

    graph, param = get_constructed_graph(
        "empty", positions=positions, features=features, labels=labels
    )
    assert graph == "foo"
    del param["degree"]
    assert not param

    with pytest.raises(ValueError):
        graph, param = get_constructed_graph(
            "foo", 5, positions=positions, features=features, labels=labels
        )


def test_calc_deg_pyg(graph_pyg):
    deg, graph = calc_deg(3, lambda x: graph_pyg)
    assert deg - 2 == 0
    assert isinstance(graph, pyg.data.Data)


def test_construct_graph_by_degree(graph_pyg, positions, features, labels):
    same_params = {
        "positions": positions,
        "self_loops": True,
        "features": features,
        "labels": labels,
    }
    graph, _, _ = construct_graph_by_degree(
        graph_func=lambda x: graph_pyg, desired_avg_degree=2
    )
    assert isinstance(_, float)

    graph, radius, _ = construct_graph_by_degree(
        graph_func=lambda x: construct_graph(
            algorithm="radius", param=x, **same_params
        ),
        desired_avg_degree=7,
        tolerance=10,
        x0=10,
        verbose=False,
    )
    assert isinstance(radius, float)
    assert np.abs((graph.num_edges / graph.num_nodes) - 7) < 10

    graph, radius, _ = construct_graph_by_degree(
        graph_func=lambda x: construct_graph(
            algorithm="radius", param=x, **same_params
        ),
        desired_avg_degree=5,
        tolerance=0.001,
        x0=1e-10,
    )
    assert np.abs((graph.num_edges / graph.num_nodes) - 5) < 0.001
