"""
Implementation of the SAGE model.
Uses the PyTorch Geometric library
"""

from typing import Any

from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from ctgnn.nn.utils import get_activation
from ctgnn.utils import to_tensor


class GraphSAGE(nn.Module):
    """
    Implementation of the GraphSAGE for graphs
    from the paper "Inductive Representation Learning on Large Graphs"
    by Hamilton et al. (https://arxiv.org/abs/1706.02216)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """Initialize GraphSAGE.

        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
            activation: Name of the activation function to use
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation

        layers = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            layers.append(
                SAGEConv_(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    normalize=True,
                    dropout=dropout,
                    activation=activation,
                )
            )
        self.layers = nn.Sequential(*layers)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            get_activation(activation),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, batch: Data) -> Tensor:
        """Forward pass of GraphSAGE.

        Args:
            batch: The batch of data to process

        Returns:
            The final node features
        """
        if hasattr(batch, "batch_size"):
            batch.batch_size = to_tensor(batch.batch_size)
            batch.num_sampled_nodes = to_tensor(batch.num_sampled_nodes)
            batch.num_sampled_edges = to_tensor(batch.num_sampled_edges)
        batch = self.layers(batch.clone())
        h_final: Tensor = self.final_mlp(batch.x)
        return h_final


class SAGEConv_(nn.Module):  # pylint: disable=invalid-name
    """Wrapper around the SAGEConv class from PyTorch Geometric.
    Adds the activation function and dropout to the forward pass.
    """

    def __init__(self, activation: str = "relu", dropout: float = 0.0, **kwargs: Any):
        super().__init__()
        self.pyg_conv = SAGEConv(aggr="max", root_weight=True, project=True, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, batch: Data) -> Data:  # pylint: disable=arguments-differ
        """Forward pass of the GATConv.

        Args:
            batch: The batch of data to process

        Returns:
            The updated batch
        """
        x, edge_index = batch.x, batch.edge_index
        x = self.pyg_conv.forward(x, edge_index)
        x = self.dropout(x)
        batch.x = self.activation(x)
        return batch
