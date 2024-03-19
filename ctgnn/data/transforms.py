"""Data transforms for graph datasets."""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_undirected


class ShuffleEdges:
    """Shuffle the edges of the input graph."""

    def __call__(self, data: Data) -> Data:
        """Shuffle the edges of the input graph.

        Args:
            data: input graph
        """
        # remove reverse edges
        data.edge_index = data.edge_index[:, data.edge_index[0] <= data.edge_index[1]]
        # shuffle destination edges
        permutation = torch.randperm(data.edge_index.shape[1])
        data.edge_index[1] = data.edge_index[1, permutation]
        # make undirected again
        data.edge_index = to_undirected(data.edge_index)
        # Make sure that there are no duplicates and the edge index is sorted
        # Makes that loading more efficient
        data.edge_index = coalesce(
            edge_index=data.edge_index, num_nodes=data.num_nodes, sort_by_row=False
        )
        return data
