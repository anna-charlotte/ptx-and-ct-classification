import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ptx_classification.data.chestxray14.chestxray14 import ChestXray14Dataset
from ptx_classification.evaluation.grad_cam import plot_grad_cam
from ptx_classification.models import get_model_name, load_multi_label_model_from_checkpoint
from ptx_classification.utils import get_date_and_time, load_json

"""
Usage:
python src/scripts/grad_cam_evaluation.py
--model_paths /Users/x/Desktop/bioinformatik/thesis/bachelor-thesis-charlotte-gerhaher/ray_results/training_function_57e43_00008_8_data_aug_p=0.75_labels_to_use=ChestTube_model=model_class_ResNet18Model_max_epochs_100_lr_5e-05_transform_image_net_batch_size_32_loss_FocalWith/checkpoint_best_model/epoch=21-step=9262.ckpt
--save_plot_to /Users/x/Desktop/bioinformatik/thesis/bachelor-thesis-charlotte-gerhaher/plots/grad_cam/grad_cam_tests/
--data_dir /Users/x/Desktop/bioinformatik/thesis/bachelor-thesis-charlotte-gerhaher/

python src/scripts/grad_cam_evaluation.py --model_paths /Users/x/Desktop/bioinformatik/thesis/bachelor-thesis-charlotte-gerhaher/ray_results/training_function_57e43_00008_8_data_aug_p=0.75_labels_to_use=ChestTube_model=model_class_ResNet18Model_max_epochs_100_lr_5e-05_transform_image_net_batch_size_32_loss_FocalWith/checkpoint_best_model/epoch=21-step=9262.ckpt --save_plot_to /Users/x/Desktop/bioinformatik/thesis/bachelor-thesis-charlotte-gerhaher/plots/grad_cam/grad_cam_tests/ --data_dir /Users/x/Desktop/bioinformatik/thesis/bachelor-thesis-charlotte-gerhaher/

-------

python 
src/scripts/grad_cam_evaluation.py
--model_paths
# candid models:
/u/home/gerh/thesis/ray_results/pneumothorax_classification/candid_ptx/training_function_2022-08-24_13-13-25/training_function_c5dc1_00011_11_data_aug_p=0.7500_labels_to_use=Pneumothorax_lr_scheduler_factor=None_model=model_class_ResNet18Model_ma
/u/home/gerh/thesis/ray_results/pneumothorax_classification/candid_ptx/training_function_2022-08-24_13-13-25/training_function_c5dc1_00008_8_data_aug_p=0.7500_labels_to_use=Chest_Tube_Pneumothorax_lr_scheduler_factor=None_model=model_class_ResNet
# chestxray14 models:
/u/home/gerh/thesis/ray_results/pneumothorax_classification/training_function_2022-08-31_21-34-26/training_function_ec769_00003_3_chest_tube_labels=None_data_aug_p=0.2500_train_class_weights=1_0_10_0_lr_scheduler_factor=0.1000_model=model_class_D
/u/home/gerh/thesis/ray_results/pneumothorax_classification/training_function_2022-08-20_09-06-05_chestxray14_with_ct/training_function_8f2ed_00011_11_chest_tube_labels=mnt_cephstorage_users_gerh_thesis_ray_results_training_function_2022-07-13_16-51-57_best_models_0
# chexpert models:
/u/home/gerh/thesis/ray_results/pneumothorax_classification/training_function_2022-08-28_17-47-55_chexpert_with_and_without_ct/training_function_c8294_00008_8_chest_tube_labels=None_data_aug_p=0.5000_lr_scheduler_factor=0.5000_model=model_class_DenseNet121Model_max_epochs_10
/u/home/gerh/thesis/ray_results/pneumothorax_classification/training_function_2022-08-28_17-47-55_chexpert_with_and_without_ct/training_function_c8294_00007_7_chest_tube_labels=mnt_cephstorage_users_gerh_thesis_ray_results_training_function_2022-07-13_16-51-57_best_models_00

--save_plot_to
plots/

--data_dir 
/u/home/gerh/share-all/

python src/scripts/grad_cam_evaluation.py --model_paths /u/home/gerh/thesis/ray_results/pneumothorax_classification/training_function_2022-08-28_17-47-55_chexpert_with_and_without_ct/training_function_c8294_00008_8_chest_tube_labels=None_data_aug_p=0.5000_lr_scheduler_factor=0.5000_model=model_class_DenseNet121Model_max_epochs_10 /u/home/gerh/thesis/ray_results/pneumothorax_classification/training_function_2022-08-28_17-47-55_chexpert_with_and_without_ct/training_function_c8294_00007_7_chest_tube_labels=mnt_cephstorage_users_gerh_thesis_ray_results_training_function_2022-07-13_16-51-57_best_models_00 --save_plot_to plots/ --data_dir /u/home/gerh/share-all/
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_paths",
        nargs="+",  # one or more
        type=Path,
        required=True,
        help="Path to model checkpoints that should be used.",
    )
    parser.add_argument(
        "--save_plot_to",
        type=Path,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        default=None,
    )
    parser.add_argument(
        "--num_imgs",
        type=int,
        required=False,
        default=1000,
    )

    args = parser.parse_args()

    model_paths = args.model_paths
    n_imgs = args.num_imgs
    data_dir = args.data_dir
    save_to = args.save_plot_to / f"grad_cam_{get_date_and_time(include_ms=False)}"
    Path.mkdir(save_to, parents=True, exist_ok=True)

    # extract checkpoint files, if not already given
    model_paths_remove = []
    for model_path in model_paths:
        if not str(model_path).endswith(".ckpt"):
            ckpt_path = ""
            for path in model_path.glob("*/*"):
                if ".ckpt" in str(path) and "checkpoint_best_model" in str(path):
                    ckpt_path = path
            model_paths_remove.append(model_path)
            model_paths.append(ckpt_path)
    for path in model_paths_remove:
        model_paths.remove(path)

    models = [load_multi_label_model_from_checkpoint(model_path=path) for path in model_paths]
    model_resizing = [
        tuple(load_json(path.parent.parent / "params.json")["resize"]) for path in model_paths
    ]
    print(f"model_resizing = {model_resizing}")

    model_names = [get_model_name(model) for model in models]
    cmd_params = {
        "model_paths": str(model_paths),
        "data_dir": str(data_dir),
        "save_to": str(save_to),
        "num_imgs": n_imgs,
    }
    json.dump(cmd_params, open(save_to / "cmp_params.json", "w"))
    chestxray14 = ChestXray14Dataset(root=data_dir)
    df = chestxray14.df_test_w_bbox
    img_paths = df[df["Finding Label"] == "Pneumothorax"]["Path"].tolist()

    for img_path in img_paths:
        imgs = [
            chestxray14.load_image(img_path=img_path, rgb=True, resize=resize)
            for resize in model_resizing
        ]
        img_with_bb = chestxray14.draw_bbox(img_path=img_path, class_label="Pneumothorax")
        for img in imgs:
            print(f"img.size() = {img.size()}")
        print(f"img_with_bb.size = {img_with_bb.size}\n")
        plot_grad_cam(
            models=models,
            model_names=model_names,
            imgs=imgs,
            img_id=chestxray14.get_image_id(img_path),
            save_to=save_to,
            ground_truth_image=img_with_bb,
            ground_truth_title="X-ray image with PTX",
        )
        if n_imgs != -1 and img_paths.index(img_path) + 1 >= n_imgs:
            break

    print(f"Saved all grad cam plots to: {save_to}")


if __name__ == "__main__":
    main()
