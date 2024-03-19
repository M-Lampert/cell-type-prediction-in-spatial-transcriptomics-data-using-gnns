"""
This module contains the training logic that is used in the `main.py` script.
It is based on the Ignite library (https://pytorch.org/ignite/).
"""

from .data import get_graph, setup_data
from .models import setup_model
from .trainers import setup_evaluator, setup_trainer
from .utils import save_config, setup_configs, setup_handlers, setup_output_dir
