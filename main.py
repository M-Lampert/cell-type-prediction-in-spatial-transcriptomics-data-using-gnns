# pylint: skip-file
# mypy: ignore-errors
# flake8: noqa
# pyright: reportPrivateImportUsage=false
import logging
import time
from pathlib import Path

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.handlers import Checkpoint
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed
from omegaconf import DictConfig
from torch import nn, optim
from tqdm import tqdm

from ctgnn.training.data import setup_data
from ctgnn.training.metrics import F1
from ctgnn.training.models import setup_model
from ctgnn.training.trainers import setup_evaluator, setup_trainer
from ctgnn.training.utils import *


def run(config: DictConfig):
    start_time = time.time()
    # make a certain seed
    manual_seed(config.seed + config.runs)

    # create output folder and copy config file to output dir
    output_dir = setup_output_dir(config)
    save_config(config, output_dir)
    config.output_dir = output_dir

    dataloader_train, dataloader_eval, dataloader_test, dataset_sizes = setup_data(
        config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_model(config, dataset_sizes).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss().to(device=device)

    trainer = setup_trainer(
        model=model, optimizer=optimizer, loss_fn=loss_fn, device=device
    )
    evaluator = setup_evaluator(model=model, device=device)

    accuracy = Accuracy(device=device)
    f1 = F1(average="weighted", device=device)
    metrics = {
        "loss": Loss(loss_fn, device=device),
        "accuracy": accuracy,
        "f1": f1,
    }
    for split, engine in zip(["train", "eval"], [trainer, evaluator]):
        for name, metric in metrics.items():
            metric.attach(engine, split + "_" + name)

    # setup ignite handlers
    to_save_eval = {"model": model}
    setup_handlers(trainer, evaluator, config, to_save_eval)

    # run evaluation at every training epoch end
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def _():
        evaluator.run(dataloader_eval)
        trainer.state.metrics.update(evaluator.state.metrics)

    ProgressBar(position=1).attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        closing_event_name=Events.COMPLETED,
        metric_names=[
            "train_accuracy",
            "train_loss",
            "train_f1",
            "eval_accuracy",
            "eval_loss",
            "eval_f1",
        ],
    )

    trainer.run(
        dataloader_train,
        max_epochs=config.max_epochs,
    )

    # Load from best checkpoint and evaluate on train, eval and test
    ckpt_paths = list(Path(output_dir).glob("best_model*"))
    if len(ckpt_paths) > 0:
        ckpt = torch.load(ckpt_paths[0])
        Checkpoint.load_objects(to_load=to_save_eval, checkpoint=ckpt)
        assert model == to_save_eval["model"]
        tester = setup_evaluator(model=model, device=device)
        for name, metric in metrics.items():
            metric.attach(tester, name)
        res_dict = dict(config)
        res_dict["runtime"] = time.time() - start_time
        res_dict["output_dir"] = str(
            output_dir
        )  # convert to string for saving to parquet
        res_dict["splits"] = [
            list(config.splits)
        ]  # convert to nested list for converting to dataframe
        for split, loader in zip(
            ["train", "eval", "test"],
            [dataloader_train, dataloader_eval, dataloader_test],
        ):
            tester.run(loader)
            res_dict[split + "_accuracy"] = tester.state.metrics["accuracy"]
            res_dict[split + "_f1"] = tester.state.metrics["f1"]
            res_dict[split + "_loss"] = tester.state.metrics["loss"]
    else:
        res_dict = dict(config)
        res_dict["runtime"] = time.time() - start_time
        res_dict["output_dir"] = str(
            output_dir
        )  # convert to string for saving to parquet
        res_dict["splits"] = [
            list(config.splits)
        ]  # convert to nested list for converting to dataframe
        for split, loader in zip(
            ["train", "eval", "test"],
            [dataloader_train, dataloader_eval, dataloader_test],
        ):
            res_dict[split + "_accuracy"] = torch.nan
            res_dict[split + "_f1"] = torch.nan
            res_dict[split + "_loss"] = torch.nan
    save_results(experiment_name=config.experiment_name, res_dict=res_dict)


# main entrypoint
def main():
    logging.disable(logging.INFO)
    configs = setup_configs()
    experiment_name = configs[0]["experiment_name"]
    res_df = get_results(experiment_name)
    filtered_configs = remove_existing_configs(configs, res_df)
    pbar = tqdm(filtered_configs, position=0)
    for i, config in enumerate(pbar):
        pbar.set_description(f"Run: [{i+1}/{len(filtered_configs)}]")
        pbar.set_postfix_str(
            f"Dataset: {config.dataset}, Model: {config.model_name}, Layers: {config.num_layers}, features: {config.features}"
        )
        run(config=config)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
