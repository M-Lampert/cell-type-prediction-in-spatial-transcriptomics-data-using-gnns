"""
Module for constructing graphs from coordinates.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import (
    AddSelfLoops,
    Delaunay,
    FaceToEdge,
    KNNGraph,
    RadiusGraph,
    RemoveDuplicatedEdges,
    ToUndirected,
)
from torch_geometric.utils import coalesce

from ctgnn.utils.torch import to_tensor

__all__ = [
    "construct_graph",
]


def construct_graph(
    algorithm: str = "knn",
    param: int | float | None = None,
    positions: list | np.ndarray | None = None,
    features: list | np.ndarray | None = None,
    labels: list | np.ndarray | None = None,
    self_loops: bool = False,
) -> Data:
    """Constructs an undirected graph using the specified algorithm.

    Args:
        algorithm: The algorithm that should be used to construct the graph.
            Can be one of the following:
                knn: `k`-nearest neighbors graph
                radius: Radius graph
                delaunay: Delaunay graph
                radius_delaunay: Intersection of Delaunay graph and radius graph
        param: The parameter for the algorithm.
            Either `k` or the radius depending on the algorithm.
        positions: The coordinates of the nodes in space
            given as List or numpy array of shape (num_nodes, num_dims).
            Defaults to None.
        features: The node features given as List or numpy array of shape
            (num_nodes, num_features). Defaults to None.
        labels: The node labels given as List or numpy array of shape
            (num_nodes, 1). Defaults to None.
        self_loops: Whether to include self-loops in the graph. Defaults to True.

    Returns:
        The graph representation from PyG.
    """
    position_tensor = (
        to_tensor(positions, dtype=torch.float) if positions is not None else None
    )
    node_feature_tensor = (
        to_tensor(features, dtype=torch.float) if features is not None else None
    )
    label_tensor = to_tensor(labels, dtype=torch.long) if labels is not None else None

    empty_graph = Data(
        x=node_feature_tensor,
        y=label_tensor,
        pos=position_tensor,
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )

    match algorithm.lower():
        case "knn":
            assert (
                param is not None
            ), "Parameter `param` must be specified for `knn` algorithm."
            graph = KNNGraph(k=param, force_undirected=True, loop=self_loops)(
                empty_graph
            )

        case "radius":
            assert (
                param is not None
            ), "Parameter `param` must be specified for `radius` algorithm."
            graph = RadiusGraph(r=param, loop=self_loops)(empty_graph)

        case "delaunay":
            graph = get_Delaunay_graph(empty_graph, self_loops)

        case "radius_delaunay":
            graph_delaunay = get_Delaunay_graph(empty_graph, self_loops)
            assert (
                param is not None
            ), "Parameter param must be specified for radius_delaunay algorithm."
            graph_radius = RadiusGraph(r=param, loop=self_loops)(empty_graph)
            combined = torch.cat(
                [graph_delaunay.edge_index, graph_radius.edge_index], dim=1
            )
            uniques, counts = combined.unique(dim=1, return_counts=True)
            intersection = uniques[:, counts > 1]
            graph = Data(
                x=node_feature_tensor,
                y=label_tensor,
                pos=position_tensor,
                edge_index=intersection,
            )

        case "empty":
            assert self_loops, "Empty graph must have self-loops."
            graph = AddSelfLoops()(empty_graph)

        case _:
            raise ValueError(f"Unknown algorithm {algorithm}")

    # Make sure that there are no duplicates and the edge index is sorted
    # Makes that loading more efficient
    graph.edge_index = coalesce(
        edge_index=graph.edge_index, num_nodes=graph.num_nodes, sort_by_row=False
    )
    return graph


def get_Delaunay_graph(  # pylint: disable=invalid-name
    graph: Data, self_loops: bool
) -> Data:
    """
    Returns the Delaunay graph for 2D or 3D.

    Args:
        graph: The input `Data` object that does not yet contain any edges.
        self_loops: Whether the graph should contain self-loops.

    Returns:
        The `Data` object with edges.
    """
    graph = Delaunay()(graph)
    if graph.pos.size(1) == 2:
        graph = FaceToEdge()(graph)
    elif graph.pos.size(1) == 3:
        # https://github.com/pyg-team/pytorch_geometric/issues/1574
        tetra = graph.face
        graph.edge_index = torch.cat(
            [tetra[:2], tetra[1:3, :], tetra[-2:], tetra[::2], tetra[::3], tetra[1::2]],
            dim=1,
        )
        graph = RemoveDuplicatedEdges()(graph)
        graph = ToUndirected()(graph)
        graph.face = None
    else:
        raise ValueError(
            "Currently only 2D or 3D are supported for Delaunay graph construction"
        )

    if self_loops:
        graph = AddSelfLoops()(graph)

    return graph
