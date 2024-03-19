"""
Implementation of a simple MLP model for node classification.
This is used as baseline for the experiments.
"""

from torch import Tensor, nn
from torch_geometric.data import Data

from ctgnn.nn.utils import get_activation


class MLP(nn.Module):
    """Implementation of a simple MLP for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """Initialize MLP.

        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
            num_layers: Number of MLP layers
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

        layers: list[nn.Module] = []
        for i in range(num_layers - 1):
            in_channels = in_channels if i == 0 else hidden_channels
            layers.append(nn.Linear(in_channels, hidden_channels))
            layers.append(nn.Dropout(dropout))
            layers.append(get_activation(activation))
        layers.append(
            nn.Linear(hidden_channels if num_layers > 1 else in_channels, out_channels)
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, batch: Data) -> Tensor:
        """Forward pass of the MLP.

        Args:
            batch: Graph data

        Returns:
            Output of the MLP
        """
        x = self.layers(batch.x)
        return x  # type: ignore[no-any-return]
