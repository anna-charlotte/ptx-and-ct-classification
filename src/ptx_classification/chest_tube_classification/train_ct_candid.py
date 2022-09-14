import json
import os
from typing import Any, Dict, List

import pandas as pd
from torch.nn import BCEWithLogitsLoss

from ptx_classification.data.datamodules import DataModuleChestTubeClassification
from ptx_classification.loss import FocalWithLogitsLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from pathlib import Path

import pytorch_lightning as pl
import ray
import torch.nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from ptx_classification.data.candid_ptx.candid_ptx import (
    CandidPtxDataModule,
    CandidPtxDataset,
    CandidPtxLabels,
    CandidPtxSplits,
)
from ptx_classification.data.padchest.padchest import (
    PadChestDataModule,
    PadChestDataset,
    PadChestProjectionLabels,
)
from ptx_classification.evaluation.evaluate import (
    CandidPtxMetrics,
    PadChestMetrics,
    read_in_prediction,
    youden_index,
)
from ptx_classification.models import (
    EfficientNetB0Model,
    MobileNetV2Model,
    ResNet18Model,
    VGG11Model,
    get_lightning_module,
)
from ptx_classification.transforms import create_augmentation_transform, get_transforms
from ptx_classification.utils import (
    RANDOM_SEED,
    REPO_ROOT_DIR,
    get_cache_dir,
    get_data_dir,
    save_commit_information,
    trial_dirname_creator,
)


def training_function(config: Dict[str, Any]) -> None:
    print(f"config = {config}")
    save_commit_information(filename=Path(os.getcwd()) / "commit_info.json")

    labels_to_use = config["labels_to_use"]
    if CandidPtxLabels.CT not in labels_to_use:
        raise ValueError("Chest Tube is not listed in labels_to_use parameter. ")

    if "random_seed" in config.keys():
        random_seed = config["random_seed"]
        pl.seed_everything(random_seed, workers=True)

    data_dir = get_data_dir()
    cache_dir = get_cache_dir()
    trial_dir = Path(tune.get_trial_dir())

    print(f"data_dir = {data_dir}")
    print(f"cache_dir = {cache_dir}")
    print(f"trial_dir = {trial_dir}")

    data_aug_transforms = None
    if "data_aug_p" in config.keys() and config["data_aug_p"] > 0:
        data_aug_transforms = create_augmentation_transform(p=config["data_aug_p"])
    print(f"data_aug_transforms = {data_aug_transforms}")

    transform = get_transforms(config["model"]["transform"])
    model = get_lightning_module(
        model_class=config["model"]["model_class"],
        loss=config["model"]["loss"],
        lr=config["model"]["lr"],
        lr_scheduler_factor=config["lr_scheduler_factor"],
        lr_scheduler_monitor=config["lr_scheduler_monitor"],
        labels=config["labels_to_use"],
        pretrained=config["pretrained"],
        transform=transform,
        trial_dir=trial_dir,
        weight_decay=0.0
        if "weight_decay" in config["model"].keys()
        else config["model"]["weight_decay"],
    )

    # callbacks
    checkpoint_dir = Path(os.getcwd()) / "checkpoint_best_model"
    os.mkdir(checkpoint_dir)
    print(f"checkpoint_dir = {checkpoint_dir}")

    patience = 8
    early_stop_callback = EarlyStopping(
        monitor=PadChestMetrics.PADCHEST_AUC_VAL_CT,
        min_delta=0.00,
        patience=patience,
        verbose=False,
        mode="max",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=PadChestMetrics.PADCHEST_AUC_VAL_CT,
        mode="max",
        dirpath=checkpoint_dir,
        save_top_k=1,
        # filename="candid_ptx_{epoch:02d}_{val_loss:.2f}",
    )
    tune_report_callback = TuneReportCheckpointCallback(
        metrics=metrics_to_monitor(labels_to_use=labels_to_use),
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
        callbacks=[early_stop_callback, tune_report_callback, checkpoint_callback],
    )
    n_train = config["data_module"]["train_size"]
    n_val = config["data_module"]["val_size"]
    n_test = config["data_module"]["test_size"]
    resize = None if "resize" not in config.keys() else config["resize"]
    print(f"batch size = {config['model']['batch_size']}")
    print(f"train-val-test-split = {[n_train, n_val, n_test]}")
    print(f"resize = {resize}")

    candid_ptx_datamodule = CandidPtxDataModule(
        dataset=CandidPtxDataset(
            root=data_dir, cache_dir=cache_dir, labels_to_use=labels_to_use, resize=resize
        ),
        batch_size=config["model"]["batch_size"],
        train_val_test_split=(n_train, n_val, n_test),
        data_aug_transforms=data_aug_transforms,
    )
    padchest_datamodule = PadChestDataModule(
        dataset=PadChestDataset(
            root=data_dir,
            cache_dir=cache_dir,
            projection_labels=[
                PadChestProjectionLabels.AP,
                PadChestProjectionLabels.AP_h,
                PadChestProjectionLabels.PA,
            ],
            min_age=16,
            resize=resize,
        ),
        batch_size=1,
        train_val_test_split=(0, 0.25, 0.75),
        ratio_ct_to_no_ct=(1, 2),
    )
    datamodule = DataModuleChestTubeClassification(
        candid_ptx_datamodule=candid_ptx_datamodule,
        padchest_datamodule=padchest_datamodule,
    )
    if config["model"]["lr"] == "auto_lr_find":
        trainer.tune(model, datamodule=datamodule)

    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, dataloaders=datamodule.test_dataloader())

    # save youden index
    progress = pd.read_csv(trial_dir / "progress.csv")
    best_epoch = max(len(progress) - 1 - patience, 0)
    print(f"best_epoch = {best_epoch}")

    pred_dir = trial_dir / "predictions"
    y_pred_val = read_in_prediction(pred_dir / f"pred_PadChest_val_epoch={best_epoch}.csv").values
    y_target_val = read_in_prediction(pred_dir / f"pred_PadChest_val_target.csv").values
    cut_off_youden = youden_index(y_true=y_target_val, y_pred=y_pred_val)

    print(f"cut_off_youden = {cut_off_youden}")
    with open(trial_dir / "youden_index.json", "w") as file:
        json.dump({"youden index": cut_off_youden}, file)


def metrics_to_monitor(labels_to_use: List[CandidPtxLabels]) -> Dict[str, str]:
    metrics = {
        "loss/CandidPtx": CandidPtxMetrics.CANDID_PTX_LOSS_TRAIN,
        "loss_val/CandidPtx": CandidPtxMetrics.CANDID_PTX_LOSS_VAL,
        "loss_val/PadChest": PadChestMetrics.PADCHEST_LOSS_VAL,
        "AUC Chest Tube/PadChest/VAL": PadChestMetrics.PADCHEST_AUC_VAL_CT,
    }
    if CandidPtxLabels.CT in labels_to_use:
        metrics["AUC Chest Tube/CandidPtx/TRAIN"] = CandidPtxMetrics.CANDID_PTX_AUC_TRAIN_CT
        metrics["AUC Chest Tube/CandidPtx/VAL"] = CandidPtxMetrics.CANDID_PTX_AUC_VAL_CT
    if CandidPtxLabels.PTX in labels_to_use:
        metrics["AUC Pneumothorax/CandidPtx/TRAIN"] = CandidPtxMetrics.CANDID_PTX_AUC_TRAIN_PTX
        metrics["AUC Pneumothorax/CandidPtx/VAL"] = CandidPtxMetrics.CANDID_PTX_AUC_VAL_PTX
    if CandidPtxLabels.RF in labels_to_use:
        metrics["AUC Rib Fracture/CandidPtx/TRAIN"] = CandidPtxMetrics.CANDID_PTX_AUC_TRAIN_RF
        metrics["AUC Rib Fracture/CandidPtx/VAL"] = CandidPtxMetrics.CANDID_PTX_AUC_VAL_RF
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
    local_dir = REPO_ROOT_DIR / "ray_results" / "chest_tube_classification"
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
                "train_size": CandidPtxSplits.N_TRAIN,
                "val_size": CandidPtxSplits.N_VAL,
                "test_size": CandidPtxSplits.N_TEST,
            },
            "data_aug_p": tune.grid_search([0.0, 0.25, 0.5, 0.75, 1.0]),
            "labels_to_use": tune.grid_search(
                [
                    [CandidPtxDataset.CT],
                    # [CandidPtxDataset.CT, CandidPtxDataset.PTX],
                    # CandidPtxLabels.LABELS,
                ]
            ),
            "lr_scheduler_factor": tune.grid_search([0.5, 0.3, 0.1]),
            "lr_scheduler_monitor": "loss_val/CandidPtx",
            "pretrained": True,
            "resize": tune.grid_search([(512, 512), (256, 256)]),
            "model": tune.grid_search(
                [
                    {
                        "model_class": ResNet18Model.__name__,
                        "max_epochs": 100,
                        "lr": 5e-05,
                        "transform": "image_net",
                        "batch_size": 32,
                        "loss": BCEWithLogitsLoss.__name__,
                        "weight_decay": 1e-04,
                    },
                    {
                        "model_class": ResNet18Model.__name__,
                        "max_epochs": 100,
                        "lr": 5e-05,
                        "transform": "image_net",
                        "batch_size": 32,
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
