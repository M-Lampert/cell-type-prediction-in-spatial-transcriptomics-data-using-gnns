"""
Module that is used for data loading and preprocessing.
Preprocessing includes graph construction, feature construction and transformations.
"""

from .construction_utils import get_constructed_graph
from .data_loading import get_spatial_data, iterate_datasets
from .graph_construction import construct_graph
from .transforms import ShuffleEdges
