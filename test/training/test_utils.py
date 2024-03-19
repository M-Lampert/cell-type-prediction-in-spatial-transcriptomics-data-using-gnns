"""
Test the util functions for model training with PyTorch Ignite.
"""
# pylint: disable=missing-function-docstring, protected-access, redefined-outer-name

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint
from omegaconf import OmegaConf

from ctgnn.training.utils import (
    expand_configs,
    get_parser,
    get_results,
    remove_existing_configs,
    save_config,
    save_results,
    setup_configs,
    setup_handlers,
    setup_output_dir,
    validate_config,
)


@pytest.fixture
def config_path(example_config, tmp_path):
    path = tmp_path / "config.yaml"
    with path.open("w") as f:
        yaml.dump(example_config, f)
    return path


def test_get_parser():
    parser = get_parser()

    assert isinstance(parser, argparse.ArgumentParser)

    # Check that the parser has the correct number of arguments
    # (including the default help argument)
    assert len(parser._actions) == 2

    # Check that the first argument is of type Path
    assert isinstance(parser._actions[1], argparse._StoreAction)
    assert parser._actions[1].type == Path

    # Check that the first argument has the correct help message
    assert parser._actions[1].help == "Config file path"


def test_setup_configs_default_parser(mocker, config_path):
    mocker.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(config=config_path),
    )

    configs = setup_configs()

    assert isinstance(configs, list)
    assert len(configs) == 12
    assert configs[0]["seed"] == 43
    assert configs[0]["dataset"] == "intestine"
    assert configs[0]["experiment_name"] == "default"


def test_setup_configs_with_path(config_path):
    configs = setup_configs(config_path)

    assert isinstance(configs, list)
    assert len(configs) == 12


def test_validate_configs(example_config):
    conf = validate_config(example_config)
    assert conf["seed"] == 43
    assert conf["dataset"] == "intestine"

    del example_config["seed"]
    conf = validate_config(example_config)
    assert conf["seed"] == 42

    with pytest.raises(ValueError):
        del example_config["dataset"]
        validate_config(example_config)

    example_config["dataset"] = "embryo"
    example_config["construction_method"] = "delaunay"
    validate_config(example_config)


def test_expand_configs():
    config_dict = {
        "lr": [0.01, 0.001],
        "batch_size": [32, 64],
        "heads": [1],
    }
    configs_list = expand_configs(config_dict)

    assert isinstance(configs_list, list)
    assert len(configs_list) == 4
    assert configs_list[0]["lr"] == 0.01
    assert configs_list[0]["batch_size"] == 32
    assert configs_list[0]["heads"] == 1
    assert configs_list[1]["lr"] == 0.01
    assert configs_list[1]["batch_size"] == 64
    assert configs_list[1]["heads"] == 1
    assert configs_list[2]["lr"] == 0.001
    assert configs_list[2]["batch_size"] == 32
    assert configs_list[3]["lr"] == 0.001
    assert configs_list[3]["batch_size"] == 64


def test_expand_configs_with_gat_heads():
    config_dict = {
        "model_name": ["GAT"],
        "heads": [2, 3],
        "lr": [0.01, 0.001],
    }
    configs_list = expand_configs(config_dict)
    assert len(configs_list) == 4

    config_dict = {
        "model_name": ["GCN"],
        "heads": [2, 3],
        "lr": [0.01, 0.001],
    }
    configs_list = expand_configs(config_dict)
    assert len(configs_list) == 2

    config_dict = {
        "model_name": ["GAT", "GCN"],
        "heads": [2, 3],
        "lr": [0.01, 0.001],
    }
    configs_list = expand_configs(config_dict)
    assert len(configs_list) == 6


def test_expand_dict():
    config_dict = {
        "lr": [0.01, 0.001],
        "batch_size": [32, 64],
        "heads": [1],
    }
    configs_list = expand_configs(config_dict)

    assert isinstance(configs_list, list)
    assert len(configs_list) == 4
    assert configs_list[0]["lr"] == 0.01
    assert configs_list[0]["batch_size"] == 32
    assert configs_list[0]["heads"] == 1
    assert configs_list[1]["lr"] == 0.01
    assert configs_list[1]["batch_size"] == 64
    assert configs_list[1]["heads"] == 1
    assert configs_list[2]["lr"] == 0.001
    assert configs_list[2]["batch_size"] == 32
    assert configs_list[3]["lr"] == 0.001
    assert configs_list[3]["batch_size"] == 64


def test_remove_existing_configs(config_path):
    config_list = setup_configs(config_path)
    config_list_length = len(config_list)
    configs = remove_existing_configs(config_list, pd.DataFrame())
    assert len(configs) == config_list_length

    df = pd.DataFrame(dict(config_list[1]))
    configs = remove_existing_configs(config_list, df)
    assert len(configs) == config_list_length - 1


def test_get_results(tmp_path):
    res = get_results("foo", tmp_path)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0

    res_dict = {"foo": "bar"}
    res = pd.DataFrame(res_dict, index=[0])
    res.to_parquet(tmp_path / "foo.parquet")

    res = get_results("foo", tmp_path)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 1


def test_save_results(tmp_path):
    res = pd.DataFrame({"foo": "bar"}, index=[0])
    save_results("foo", res, tmp_path)

    res = pd.read_parquet(tmp_path / "foo.parquet")
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 1

    save_results("foo", res, tmp_path)

    res = pd.read_parquet(tmp_path / "foo.parquet")
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 2


def test_setup_output_dir(config_dict):
    output_dir = setup_output_dir(config_dict)

    assert isinstance(output_dir, Path)
    assert output_dir.exists()
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    expected_name = f"{now}"
    assert output_dir.name[:-3] == expected_name
    assert output_dir.is_dir()
    assert len(list(output_dir.iterdir())) == 0


def test_save_config(config_dict, tmp_path):
    # Call the save_config function
    save_config(config_dict, tmp_path)

    # Check that the config file exists
    config_file = Path(tmp_path) / "config-lock.yaml"
    assert config_file.exists()

    # Check that the config file contains the correct data
    with open(config_file, "r", encoding="utf-8") as f:
        saved_config = OmegaConf.load(f)
    assert saved_config == config_dict


@pytest.fixture
def trainer():
    return Engine(lambda engine, batch: None)


@pytest.fixture
def evaluator():
    return Engine(lambda engine, batch: None)


@pytest.fixture
def to_save(trainer):
    model = torch.nn.Linear(1, 1)
    return {
        "trainer": trainer,
        "model": model,
        "optimizer": torch.optim.Adam(model.parameters()),
    }


def test_setup_handlers(config_dict, trainer, evaluator, to_save):
    # Call the setup_handlers function
    ckpt_handler_eval = setup_handlers(
        trainer, evaluator, config_dict, to_save_eval=to_save
    )

    # Check that the checkpoint handlers are not None
    assert ckpt_handler_eval is not None

    # Check that the checkpoint handlers are instances of Checkpoint
    assert isinstance(ckpt_handler_eval, Checkpoint)

    # Check that the checkpoint handlers have the correct attributes
    assert ckpt_handler_eval.filename_prefix == "best"
    assert ckpt_handler_eval.n_saved == 1

    # Check for the number of handlers
    assert len(evaluator._event_handlers[Events.EPOCH_COMPLETED]) == 2
