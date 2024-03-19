"""
Implementation of the Graph Attention Network
Uses the PyTorch Geometric library
"""

from typing import Any

from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from ctgnn.nn.utils import get_activation
from ctgnn.utils import to_tensor


class GAT(nn.Module):
    """
    Implementation of the GAT for graphs
    from the paper "Graph Attention Networks"
    by Veličković et al. (https://arxiv.org/abs/1710.10903)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """Initialize GAT.

        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout rate
            activation: Name of the activation function to use
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.activation = activation

        layers = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else heads * hidden_channels
            layers.append(
                GATConv_(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                    activation=activation,
                )
            )
        self.layers = nn.Sequential(*layers)

        self.final_mlp = nn.Sequential(
            nn.Linear(heads * hidden_channels, hidden_channels),
            get_activation(activation),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, batch: Data) -> Tensor:
        """Forward pass of the GAT.

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


class GATConv_(nn.Module):  # pylint: disable=invalid-name
    """Wrapper around the GATConv that allows for
    a custom activation function.
    This simplifies the forward pass of GAT.
    """

    def __init__(
        self,
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs: Any,
    ):
        kwargs["dropout"] = dropout
        super().__init__()
        self.pyg_conv = GATConv(**kwargs)
        self.layer_dropout = nn.Dropout(dropout)
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
        x = self.layer_dropout(x)
        batch.x = self.activation(x)  # pyright: ignore[reportGeneralTypeIssues]
        return batch
