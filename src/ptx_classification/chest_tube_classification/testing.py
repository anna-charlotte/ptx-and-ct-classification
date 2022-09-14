import os
from pathlib import Path

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ptx_classification.data.candid_ptx.candid_ptx import CandidPtxDataModule, CandidPtxDataset
from ptx_classification.data.padchest.padchest import PadChestDataset, PadChestProjectionLabels
from ptx_classification.evaluation.evaluate import plot_roc_curves_for_ct
from ptx_classification.models import MultiLabelModel
from ptx_classification.utils import (
    RANDOM_SEED,
    REPO_ROOT_DIR,
    get_cache_dir,
    get_data_dir,
    load_json,
    set_random_seeds,
)


def main() -> None:

    random_seed = RANDOM_SEED
    set_random_seeds(seed=random_seed)
    if torch.cuda.is_available():
        accelerator = "gpu"
        num_devices = 1
    else:
        accelerator = "cpu"
        num_devices = 1

    root_dir = get_data_dir()
    cache_dir = get_cache_dir()

    # load model
    model_dir = (
        REPO_ROOT_DIR
        / "ray_results"
        / "training_function_2022-07-13_16-51-57_best_models_00008"
        / "training_function_57e43_00008_8_data_aug_p=0.75_labels_to_use=ChestTube_model=model_class_ResNet18Model_max_epochs_100_lr_5e-05_transform_image_net_batch_size_32_loss_FocalWith"
    )

    model_file = model_dir / "checkpoint_best_model" / "epoch=21-step=9262.ckpt"
    model = MultiLabelModel.load_from_checkpoint(checkpoint_path=str(model_file))
    labels = model.class_labels
    model.trial_dir = Path(os.getcwd())
    print(f"model.trial_dir = {model.trial_dir}")

    # read in config
    json_file = "/".join(str(model_file).split("/")[:-2]) + "/params.json"
    print(f"Reading in config from {json_file}")
    config = load_json(Path(json_file))
    batch_size = 1  # config["data_module"]["batch_size"]
    n_train = config["data_module"]["train_size"]
    n_val = config["data_module"]["val_size"]
    n_test = config["data_module"]["test_size"]

    y_targets = []
    y_preds = []
    subset_names = []

    projection_labels = [
        [
            PadChestProjectionLabels.AP,
            PadChestProjectionLabels.AP_h,
            PadChestProjectionLabels.PA,
        ],
    ]
    min_ages = [0]
    resize = config["resize"]
    print(f"resize = {resize}")

    for p_labels in projection_labels:
        for min_age in min_ages:
            print(f"min_age = {min_age}")
            print(f"p_labels = {p_labels}")
            print(f"resize = {resize}")
            # create test set
            padchest = PadChestDataset(
                root=root_dir,
                cache_dir=cache_dir,
                projection_labels=p_labels,
                min_age=min_age,
                resize=resize
            )
            padchest_test_set = padchest.get_subset_of_dataset(
                rng=np.random.RandomState(random_seed),
            )
            padchest_test_dataloader = DataLoader(padchest_test_set, batch_size=1, num_workers=32)
            # trainer
            trainer = pl.Trainer(
                devices=num_devices,
                accelerator=accelerator,
                max_epochs=100,
                enable_progress_bar=False,
                num_sanity_val_steps=-1,
            )

            print("Trainer.test on padchest")
            trainer.test(model=model, dataloaders=padchest_test_dataloader)
            results = model.results_test_end
            y_target = results[0]["y_true_test"]
            y_pred = results[0]["y_pred_test"]
            set_name = f"PadChest: Proj. labels: [{(', '.join(p_labels)).replace('horizontal', 'h')}], min age: {min_age}"

            y_targets.append(pd.DataFrame(data=y_target.cpu().detach().numpy(), columns=labels))
            y_preds.append(pd.DataFrame(data=y_pred.cpu().detach().numpy(), columns=labels))
            subset_names.append(set_name)

    candid_ptx = CandidPtxDataModule(
        dataset=CandidPtxDataset(
            root=root_dir,
            cache_dir=cache_dir,
            labels_to_use=labels,
            resize=resize,
        ),
        batch_size=batch_size,
        train_val_test_split=(n_train, n_val, n_test),
    )

    # trainer
    trainer = pl.Trainer(
        devices=num_devices,
        accelerator=accelerator,
        max_epochs=100,
        enable_progress_bar=False,
        num_sanity_val_steps=-1,
    )
    print("Trainer.test on candid")
    trainer.test(model=model, dataloaders=candid_ptx.test_dataloader())

    results = model.results_test_end
    y_target = results[0]["y_true_test"]
    y_pred = results[0]["y_pred_test"]
    set_name = f"CANDID PTX"

    y_targets.append(pd.DataFrame(data=y_target.cpu().detach().numpy(), columns=labels))
    y_preds.append(pd.DataFrame(data=y_pred.cpu().detach().numpy(), columns=labels))
    subset_names.append(set_name)

    # plotting
    selection = [i for i in range(len(subset_names))]
    plot_roc_curves_for_ct(
        y_targets=[y_targets[s] for s in selection],
        predictions=[y_preds[s] for s in selection],
        model_names=[subset_names[s] for s in selection],
        save_to_dir=REPO_ROOT_DIR / "plots",
        title_plot="ROC curves for ResNet18 Chest tube prediction",
    )


if __name__ == "__main__":
    main()
