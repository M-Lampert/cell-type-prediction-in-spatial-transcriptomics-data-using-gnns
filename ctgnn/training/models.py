"""Contains the model setup function."""

from omegaconf import DictConfig
from torch import nn

from ctgnn.nn import GAT, GCN, GIN, MLP, GraphSAGE

models_dict: dict[str, type[nn.Module]] = {
    "GAT": GAT,
    "GCN": GCN,
    "GIN": GIN,
    "GraphSAGE": GraphSAGE,
    "SAGE": GraphSAGE,
    "MLP": MLP,
}


def setup_model(config: DictConfig, dataset_sizes: tuple[int, int]) -> nn.Module:
    """Set up the GNN model that should be used.

    Args:
        config: needs to contain
        - `model_name`
        - `hidden_dim`
        - `num_layers`
        - `activation`
        - `dropout`
    """

    if config.model_name == "GAT":
        kwargs = {"heads": config.heads}
    else:
        kwargs = {}

    model = models_dict[config.model_name](
        in_channels=dataset_sizes[0],
        hidden_channels=config.hidden_dim,
        out_channels=dataset_sizes[1],
        num_layers=config.num_layers,
        activation=config.activation,
        dropout=config.dropout,
        **kwargs,
    )

    return model
