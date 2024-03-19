"""Data loading and preprocessing."""

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit

from ctgnn.data import (
    ShuffleEdges,
    construct_graph,
    get_constructed_graph,
    get_spatial_data,
)


def setup_data(
    config: DictConfig,
) -> tuple[NeighborLoader, NeighborLoader, NeighborLoader, tuple[int, int]]:
    """Download datasets and create dataloaders

    Args:
        config: needs to contain
            - `global_transform`
            - `batch_size`
            - `num_layers`
            - `num_neighbors`
            - `num_workers`
            - `construction_method`
            - `dataset`
            - `splits`
            - `self_loops`
            - `param`
    """
    graph = get_graph(config)

    # optionally preprocess features
    match config.features:
        case "raw":
            pass
        case "random":
            graph.x = torch.normal(mean=0, std=1, size=graph.x.shape, dtype=torch.float)
        case _:
            raise ValueError(f"Unknown feature preprocessing {config.features}")

    # optionally transform whole graph
    match config.global_transform:
        case "shuffle_edges":
            graph = ShuffleEdges()(graph)
        case None:
            pass
        case _:
            raise ValueError(f"Unknown global transform {config.global_transform}")

    # create dataloaders
    kwargs = {
        "data": graph,
        # Sample all neighbors per each GNN layer
        "num_neighbors": [config.num_neighbors] * config.num_layers,
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": config.num_workers,
        "is_sorted": True,
    }

    train_loader = NeighborLoader(
        input_nodes=graph.train_mask,
        **kwargs,
    )

    kwargs["shuffle"] = False
    eval_loader = NeighborLoader(
        input_nodes=graph.val_mask,
        **kwargs,
    )
    test_loader = NeighborLoader(
        input_nodes=graph.test_mask,
        **kwargs,
    )

    num_features = graph.x.shape[1]
    num_classes = len(set(graph.y.tolist()))

    return train_loader, eval_loader, test_loader, (num_features, num_classes)


def get_graph(config: DictConfig) -> Data:
    """
    Construct graph from spatial data according to the given config.

    Args:
        config: needs to contain
            - `construction_method`
            - `dataset`
            - `splits`
            - `self_loops`
            - `param`
    """
    # load data
    cells, genes = get_spatial_data(config.dataset)

    # convert pandas dataframes to numpy array
    if "z" in cells.columns:
        cell_coordinates = cells[["x", "y", "z"]].values
    else:
        cell_coordinates = cells[["x", "y"]].values
    features = genes.values
    true_labels = cells["cluster_id"].values

    # construct graph
    if "param" not in config:
        config.param = None

    if config.param is None and config.construction_method != "delaunay":
        graph, params = get_constructed_graph(
            constr_method=config.construction_method,
            desired_avg_degree=7 if config.self_loops else 6,
            positions=cell_coordinates,
            features=features,
            labels=true_labels,
            self_loops=config.self_loops,
            verbose=False,
        )
        if config.construction_method == "knn":
            config.param = params["k"]
        elif (
            config.construction_method == "radius"
            or config.construction_method == "radius_delaunay"
        ):
            config.param = float(params["radius"])
    else:
        graph = construct_graph(
            algorithm=config.construction_method,
            param=config.param,
            positions=cell_coordinates,
            features=features,
            labels=true_labels,
            self_loops=config.self_loops,
        )

    # split data into train, val, test
    graph = RandomNodeSplit(
        split="train_rest", num_val=config.splits[1], num_test=config.splits[2]
    )(graph)
    return graph
