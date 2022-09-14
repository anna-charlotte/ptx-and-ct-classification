import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.nn import BCEWithLogitsLoss

from ptx_classification.data.chest_tube_data_frame import (
    ChestTubeDataFrame,
    read_in_csv_as_chest_tube_data_frame,
)
from ptx_classification.data.chexpert.chexpert import (
    CheXpertDataModule,
    CheXpertDataset,
    CheXpertLabels,
    CheXpertSplits,
)
from ptx_classification.evaluation.evaluate import read_in_prediction, youden_index
from ptx_classification.loss import FocalWithLogitsLoss
from ptx_classification.transforms import create_augmentation_transform, get_transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from pathlib import Path

import pytorch_lightning as pl
import ray
import torch.nn
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from ptx_classification.models import DenseNet121Model, ResNet50Model, get_lightning_module
from ptx_classification.utils import (
    RANDOM_SEED,
    RAY_RESULTS_DIR,
    REPO_ROOT_DIR,
    get_cache_dir,
    get_data_dir,
    load_json,
    save_commit_information,
    trial_dirname_creator,
)


def training_function(config: Dict[str, Any]) -> None:
    print(f"config = {config}")
    save_commit_information(filename=Path(os.getcwd()) / "commit_info.json")

    if "random_seed" in config.keys():
        random_seed = config["random_seed"]
        pl.seed_everything(random_seed, workers=True)

    data_dir = get_data_dir()
    cache_dir = get_cache_dir()
    trial_dir = Path(tune.get_trial_dir())

    data_aug_transforms = None
    if "data_aug_p" in config.keys() and config["data_aug_p"] > 0:
        data_aug_transforms = create_augmentation_transform(p=config["data_aug_p"])
    print(f"data_aug_transforms = {data_aug_transforms}")

    print(f"data_dir = {data_dir}")
    print(f"cache_dir = {cache_dir}")
    print(f"trial_dir = {trial_dir}")

    labels_to_use = config["labels_to_use"]
    ct_labels_path = config["chest_tube_labels"]
    if ct_labels_path is not None and "Chest Tube" not in labels_to_use:
        labels_to_use.append("Chest Tube")
        labels_to_use = sorted(labels_to_use)

    class_for_early_stopping = config["class_for_early_stopping"]
    if class_for_early_stopping not in labels_to_use:
        raise ValueError(
            f"Class that should be used for early stopping ({class_for_early_stopping}) is not part of the classes that are supposed to be used ({labels_to_use})"
        )

    # callbacks
    checkpoint_dir = Path(os.getcwd()) / "checkpoint_best_model"
    os.mkdir(checkpoint_dir)
    print(f"checkpoint_dir = {checkpoint_dir}")

    patience = 10
    early_stop_callback = EarlyStopping(
        monitor=f"AUC/{class_for_early_stopping}/CheXpert/VAL",
        min_delta=0.00,
        patience=patience,
        verbose=False,
        mode="max",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=f"AUC/{class_for_early_stopping}/CheXpert/VAL",
        mode="max",
        dirpath=checkpoint_dir,
        save_top_k=1,
        # filename="candid_ptx_{epoch:02d}_{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    metrics = metrics_to_monitor(dataset="CheXpert", labels_to_use=labels_to_use)
    print(f"metrics to monitor = {metrics}")
    metrics["memory_usage"] = "memory_usage"
    print(f"metrics to monitor = {metrics}")
    tune_report_callback = TuneReportCheckpointCallback(
        metrics=metrics,
        filename="checkpoint",
        on="validation_end",
    )

    trainer = pl.Trainer(
        default_root_dir=str(checkpoint_dir),
        devices=config["num_devices"],
        accelerator=config["accelerator"],
        max_epochs=config["model"]["max_epochs"],
        auto_lr_find=True if config["model"]["lr"] == "auto_lr_find" else False,
        enable_progress_bar=False,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        num_sanity_val_steps=0,
        # overfit_batches=10,
        callbacks=[early_stop_callback, checkpoint_callback, tune_report_callback, lr_monitor],
    )
    n_train = config["data_module"]["train_size"]
    n_val = config["data_module"]["val_size"]
    n_test = config["data_module"]["test_size"]
    batch_size = config["data_module"]["batch_size"]
    resize = None if "resize" not in config.keys() else config["resize"]
    print(f"train-val-test-split = {[n_train, n_val, n_test]}")
    print(f"resize = {resize}")

    ct_labels: Optional[ChestTubeDataFrame] = None
    cut_off = None
    if ct_labels_path is not None:
        ct_labels = read_in_csv_as_chest_tube_data_frame(
            path=Path(ct_labels_path)
            / "predictions"
            / "prediction_for_CheXpertDataset_FrontLat-Frontal_APPA-all.csv"
        )
        cut_off = load_json(Path(ct_labels_path) / "youden_index.json")["youden index"]
    print(f"cut_off = {cut_off}")

    datamodule = CheXpertDataModule(
        dataset=CheXpertDataset(
            root=data_dir,
            cache_dir=cache_dir,
            version="small" if resize[0] < 320 else "original",
            frontal_lateral="Frontal",
            ap_pa="all",
            resize=resize,
            class_labels=labels_to_use,
            ct_labels=ct_labels,
            ct_cut_off=cut_off,
        ),
        batch_size=batch_size,
        train_val_test_split=(n_train, n_val, n_test),
        train_class_weights=config["data_module"]["train_class_weights"],
        train_num_samples=5_000,
    )

    transform = get_transforms(config["model"]["transform"])
    model = get_lightning_module(
        model_class=config["model"]["model_class"],
        loss=config["model"]["loss"],
        lr=config["model"]["lr"],
        lr_scheduler_factor=config["lr_scheduler_factor"],
        lr_scheduler_monitor=config["lr_scheduler_monitor"],
        labels=labels_to_use,
        pretrained=config["pretrained"],
        # ten_crop_size=config["ten_crop_size"],
        transform=transform,
        trial_dir=trial_dir,
        weight_decay=0.0
        if "weight_decay" in config["model"].keys()
        else config["model"]["weight_decay"],
    )

    print("Start training ...")
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, dataloaders=datamodule.test_dataloader())

    # save youden index
    progress = pd.read_csv(trial_dir / "progress.csv")
    best_epoch = max(len(progress) - 1 - patience, 0)
    print(f"best_epoch = {best_epoch}")

    pred_dir = trial_dir / "predictions"
    y_pred_val = read_in_prediction(pred_dir / f"pred_ChestXray14_val_epoch={best_epoch}.csv")[
        class_for_early_stopping
    ].values
    y_target_val = read_in_prediction(pred_dir / f"pred_ChestXray14_val_target.csv")[
        class_for_early_stopping
    ].values
    cut_off_youden = youden_index(y_true=y_target_val, y_pred=y_pred_val)

    print(f"cut_off_youden = {cut_off_youden}")
    with open(trial_dir / "youden_index.json", "w") as file:
        json.dump({"youden index": cut_off_youden}, file)


def metrics_to_monitor(dataset: str, labels_to_use: List[str]) -> Dict[str, str]:
    metrics = {f"loss/{dataset}": "loss", f"loss_val/{dataset}": f"loss_val/{dataset}"}
    for label in labels_to_use:
        metrics[f"AUC {label}/{dataset}/TRAIN"] = f"AUC/{label}/{dataset}/TRAIN"
        metrics[f"AUC {label}/{dataset}/VAL"] = f"AUC/{label}/{dataset}/VAL"
    return metrics


def main() -> None:
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        accelerator = "gpu"
        num_devices = 1
    else:
        accelerator = "cpu"
        num_devices = 1
    resources_per_trial = {accelerator: num_devices}
    print(f"resources_per_trial = {resources_per_trial}")

    reporter = tune.CLIReporter(
        max_report_frequency=130,
        infer_limit=10,
    )
    local_dir = REPO_ROOT_DIR / "ray_results" / "pneumothorax_classification"
    print(f"local_dir = {local_dir}")

    ray.init()
    tune.run(
        training_function,
        local_dir=local_dir,
        trial_dirname_creator=trial_dirname_creator,
        resources_per_trial=resources_per_trial,
        num_samples=1,
        config={
            "accelerator": accelerator,
            "num_devices": num_devices,
            "random_seed": RANDOM_SEED,
            "data_module": {
                "train_size": CheXpertSplits.N_TRAIN,
                "val_size": CheXpertSplits.N_VAL,
                "test_size": CheXpertSplits.N_TEST,
                "train_class_weights": (1.0, 5.0),
                "batch_size": 16,
            },
            "data_aug_p": tune.grid_search([0.25, 0.5, 0.75]),
            "labels_to_use": [CheXpertLabels.PTX],
            "class_for_early_stopping": CheXpertLabels.PTX,
            "pretrained": True,
            "ten_crop_size": tune.grid_search([None]),
            "resize": tune.grid_search([(256, 256)]),
            "lr_scheduler_factor": tune.grid_search([0.1, 0.5]),
            "lr_scheduler_monitor": "loss_val/CheXpert",
            "chest_tube_labels": tune.grid_search(
                [
                    None,
                    RAY_RESULTS_DIR
                    / "training_function_2022-07-13_16-51-57_best_models_00008"
                    / "training_function_57e43_00008_8_data_aug_p=0.75_labels_to_use=ChestTube_model=model_class_ResNet18Model_max_epochs_100_lr_5e-05_transform_image_net_batch_size_32_loss_FocalWith",
                ]
            ),
            "model": tune.grid_search(
                [
                    {
                        "model_class": DenseNet121Model.__name__,
                        "max_epochs": 100,
                        "lr": 1e-03,
                        "transform": "image_net",
                        "loss": BCEWithLogitsLoss.__name__,
                        "weight_decay": 1e-04,
                    },
                    {
                        "model_class": DenseNet121Model.__name__,
                        "max_epochs": 100,
                        "lr": 1e-04,
                        "transform": "image_net",
                        "loss": BCEWithLogitsLoss.__name__,
                        "weight_decay": 1e-04,
                    },
                    {
                        "model_class": DenseNet121Model.__name__,
                        "max_epochs": 100,
                        "lr": 1e-05,
                        "transform": "image_net",
                        "loss": BCEWithLogitsLoss.__name__,
                        "weight_decay": 1e-04,
                    },
                    {
                        "model_class": DenseNet121Model.__name__,
                        "max_epochs": 100,
                        "lr": 1e-03,
                        "transform": "image_net",
                        "loss": FocalWithLogitsLoss.__name__,
                        "weight_decay": 1e-04,
                    },
                    {
                        "model_class": DenseNet121Model.__name__,
                        "max_epochs": 100,
                        "lr": 1e-04,
                        "transform": "image_net",
                        "loss": FocalWithLogitsLoss.__name__,
                        "weight_decay": 1e-04,
                    },
                    {
                        "model_class": DenseNet121Model.__name__,
                        "max_epochs": 100,
                        "lr": 1e-05,
                        "transform": "image_net",
                        "loss": FocalWithLogitsLoss.__name__,
                        "weight_decay": 1e-04,
                    },
                ]
            ),
        },
        progress_reporter=reporter,
    )


if __name__ == "__main__":
    main()
