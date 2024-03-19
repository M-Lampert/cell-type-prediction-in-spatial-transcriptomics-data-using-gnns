"""
This module contains the implementation of the neural networks used in the experiments.
All GNNs are implementations from PyTorch Geometric.
All GNNs are implemented so that the forward pass takes a batch of data
and returns the final result.
The last layer is always an MLP with `hidden_channels` hidden units
and `n_classes` output units.
"""

from .gat import GAT
from .gcn import GCN
from .gin import GIN
from .mlp import MLP
from .sage import GraphSAGE
