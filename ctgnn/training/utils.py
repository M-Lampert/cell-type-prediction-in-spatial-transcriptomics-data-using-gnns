"""Utility functions for training script."""
import warnings
from argparse import ArgumentParser
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.terminate_on_nan import TerminateOnNan
from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = [
    "get_parser",
    "setup_configs",
    "validate_config",
    "expand_configs",
    "expand_dict",
    "remove_existing_configs",
    "get_results",
    "save_results",
    "setup_output_dir",
    "save_config",
    "setup_handlers",
]


def get_parser() -> ArgumentParser:
    """Get default parser for training script.

    Returns:
        The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Config file path")
    return parser


def setup_configs(config_path: str | None = None) -> list[dict]:
    """Setup configuration for training script.

    Args:
        config_path: The path to the config file. Defaults to None.
            If None, the method will try to infer the path
            from the command line arguments.

    Returns:
        The parsed config object.
    """
    if config_path is None:
        parser = get_parser()
        args = parser.parse_args()
        config_path = args.config
    config = OmegaConf.load(config_path)

    config_dict = dict(config)
    config_dict = validate_config(config_dict)
    for key, value in config_dict.items():
        if isinstance(value, ListConfig):
            config_dict[key] = list(value)

    # Convert all non-list values to list to make them iterable
    for key, value in config_dict.items():
        if not isinstance(value, list) or (
            key == "splits" and isinstance(value[0], float)
        ):
            config_dict[key] = [value]

    configs_list = expand_configs(config_dict)

    return configs_list


def validate_config(config_dict: dict) -> dict:
    """Validate the config.

    Args:
        config_dict: The config dictionary.

    Returns:
        The validated config dictionary.
    """
    mandatory_keys = [
        "dataset",
        "construction_method",
        "self_loops",
        "batch_size",
        "lr",
        "model_name",
        "num_layers",
        "hidden_dim",
        "activation",
    ]
    for key in mandatory_keys:
        if key not in config_dict:
            raise ValueError(f"Config is missing mandatory key `{key}`.")

    defaults = {
        "seed": 42,
        "experiment_name": "default",
        "runs": 1,
        "splits": [0.7, 0.1, 0.2],
        "global_transform": None,
        # -1: Sample all neighbors
        "num_neighbors": -1,
        # 0: Use the main thread.
        # Works fastest in my experience and does not have memory issues
        "num_workers": 0,
        "max_epochs": 1000,
        "patience": 1000,
        "heads": 1,
        "dropout": 0.0,
        "output_dir": "./data/checkpoints",
    }
    for key, value in defaults.items():
        if key not in config_dict:
            config_dict[key] = value
    return config_dict


def expand_configs(config_dict: dict[str, list]) -> list[dict]:
    """Expand the config dictionary to a list of configs and handle
    the special GAT parameter `heads`.

    GAT has the special parameter `heads` so this parameter
    needs to be handled separately. This is because otherwise
    the GNNs without this parameter would be trained multiple times
    on the same hyperparameters since the `heads` parameter is
    iterated over.

    Args:
        config_dict: The config dictionary where each hyperparameter
            is contained as key and the list of values as value.
            Note that this function assumes that all dictionary values are lists.

    Returns:
        The list of configs.
    """
    if len(config_dict["heads"]) > 1:
        if "GAT" in config_dict["model_name"]:
            gat_config_dict = config_dict.copy()
            gat_config_dict["model_name"] = ["GAT"]
            configs_list = expand_dict(gat_config_dict)

            config_dict["model_name"].remove("GAT")
            if len(config_dict["model_name"]) > 0:
                config_dict["heads"] = [1]
                other_configs_list = expand_dict(config_dict)
                configs_list.extend(other_configs_list)
        else:
            config_dict["heads"] = [1]
            configs_list = expand_dict(config_dict)

    else:
        configs_list = expand_dict(config_dict)
    return configs_list


def expand_dict(config_dict: dict) -> list[dict]:
    """Expand the provided dictionary to a list of dictionaries.

    Args:
        config_dict: The dictionary where each key has a list of values as value.

    Returns:
        The list of configs.
    """
    configs_list = [
        dict(zip(config_dict.keys(), param)) for param in product(*config_dict.values())
    ]
    return configs_list


def remove_existing_configs(
    configs: list[dict], res_df: pd.DataFrame
) -> list[DictConfig]:
    """Remove configs that have already been trained.

    Args:
        configs: The list of configs.
        res_df: The results dataframe.

    Returns:
        The filtered list of configs.
    """
    if res_df.empty:
        return [OmegaConf.create(config) for config in configs]
    configs_df = pd.DataFrame(configs)
    configs_cols = list(
        configs_df.drop(["splits", "output_dir"], axis="columns").columns
    )
    merged_df = pd.merge(
        configs_df,
        res_df.drop(["splits", "output_dir"], axis="columns"),
        on=configs_cols,
        how="outer",
        indicator=True,
    )
    filtered_df = merged_df.loc[merged_df["_merge"] == "left_only"].drop(
        "_merge", axis=1
    )
    n_removed = len(configs_df) - len(filtered_df)
    print(f"Removed {n_removed} configs that have already been trained.")
    filtered_df = filtered_df.loc[:, configs_df.columns]
    filtered_configs = filtered_df.to_dict(orient="records")
    return [OmegaConf.create(config) for config in filtered_configs]


def get_results(
    experiment_name: str, root: str | Path = "data/results/"
) -> pd.DataFrame:
    """Get results from parquet file.

    Args:
        experiment_name: The name of the experiment.
        root: The root directory. Defaults to "data/results/".

    Returns:
        The results dataframe.
    """
    if isinstance(root, str):
        root = Path(root)
    res_path = root / (experiment_name + ".parquet")
    if res_path.exists():
        res_df = pd.read_parquet(res_path)
    else:
        res_path.parent.mkdir(parents=True, exist_ok=True)
        res_df = pd.DataFrame()
    return res_df


def save_results(
    experiment_name: str, res_dict: dict, root: str | Path = "data/results/"
) -> None:
    """Save results to parquet file. If the file already exists,
    append the results to the existing file.

    Args:
        config: The config object.
        res_dict: The results dictionary.
        root: The root directory. Defaults to "data/results/".
    """
    if isinstance(root, str):
        root = Path(root)
    res_path = root / (experiment_name + ".parquet")
    res_df = get_results(experiment_name, root)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        res_df = pd.concat([res_df, pd.DataFrame(res_dict)], ignore_index=True)
    res_df.to_parquet(res_path, index=False)


def setup_output_dir(config: DictConfig) -> Path:
    """Create output folder.

    Args:
        config: The config object.
    """
    output_dir = config.output_dir + "/" + config.experiment_name
    now = datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-3]
    name = f"{now}"
    path = Path(output_dir, name)
    path.mkdir(parents=True, exist_ok=True)
    output_dir = path.as_posix()
    return Path(output_dir)


def save_config(config: DictConfig, output_dir: str) -> None:
    """Save configuration to config-lock.yaml for result reproducibility.

    Args:
        config: The config object.
        output_dir: The output directory.
    """
    with open(f"{output_dir}/config-lock.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(config, f)


def setup_handlers(
    trainer: Engine,
    evaluator: Engine,
    config: DictConfig,
    to_save_eval: dict,
) -> Checkpoint:
    """Setup Ignite handlers.

    Args:
        trainer: The trainer engine.
        evaluator: The evaluator engine.
        config: The config object.
            Config has to contain `output_dir` and `patience` attribute.
        to_save_eval: A dictionary with objects,
            e.g. {“trainer”: trainer, “model”: model, “optimizer”: optimizer, ...}
            that map to objects that are saved in a checkpoint handler for evaluation.

    Returns:
        ckpt_handler_eval: The checkpoint handler for evaluation.
    """
    ckpt_handler_eval = None
    # checkpointing
    saver = DiskSaver(config.output_dir, require_empty=False)
    ckpt_handler_eval = Checkpoint(
        to_save_eval,
        saver,
        filename_prefix="best",
        n_saved=1,
        score_name="eval_f1",
        score_function=Checkpoint.get_default_score_fn("eval_f1"),
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), ckpt_handler_eval)

    # early stopping
    def score_fn(engine: Engine) -> float:
        """Score function for early stopping."""
        score: float = -engine.state.metrics["eval_loss"]
        return score

    es = EarlyStopping(config.patience, score_fn, trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, es)
    # terminate on nan
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    return ckpt_handler_eval
