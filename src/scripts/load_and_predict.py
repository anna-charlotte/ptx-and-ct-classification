"""
Script to load given model and make predictions for ChestXray14Dataset or CheXpertDataset
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from ptx_classification.data.chestxray14.chestxray14 import ChestXray14Dataset
from ptx_classification.data.chexpert.chexpert import CheXpertDataset
from ptx_classification.data.datasets import XrayDataset
from ptx_classification.models import load_multi_label_model_from_checkpoint
from ptx_classification.utils import (
    RANDOM_SEED,
    get_cache_dir,
    get_data_dir,
    load_json,
    set_random_seeds,
)


def load_and_predict(
    model_path: Path, dataset: XrayDataset, save_to_dir: Optional[Path] = None
) -> pd.DataFrame:
    print("Start loading model ...")
    model = load_multi_label_model_from_checkpoint(model_path=model_path)
    print("Finished loading model ...")
    print(f"Start predicting for {dataset.__class__.__name__} ...")
    df = model.predict_for_xray_dataset(dataset=dataset, save_to_dir=save_to_dir)
    print(f"Finished predicting for {dataset.__class__.__name__} ...")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mp",
        "--model_path",
        type=Path,
        required=True,
        help="Path to model checkpoint that should be used.",
    )
    parser.add_argument(
        "--save_pred_to",
        type=str,
        required=False,
        default=None,
        help="Path to prediction csv file to.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        choices=["ChestXray14Dataset", "CheXpertDataset"],
        # default="ChestXray14Dataset",
        help="Dataset for which a prediction is to be made.",
    )
    args = parser.parse_args()
    model_path = args.model_path
    save_to = args.save_pred_to
    dataset = args.dataset
    print(f"save_to = {save_to}")
    set_random_seeds(seed=RANDOM_SEED)

    root_dir = get_data_dir()
    cache_dir = get_cache_dir()

    # read in model's params.json
    print("Reading in model's config")
    config = load_json(model_path.parent.parent / "params.json")
    resize = config["resize"]

    print("Loading dataset ...")
    if dataset == "ChestXray14Dataset":
        dataset = ChestXray14Dataset(root=root_dir, cache_dir=cache_dir, resize=resize)
    elif dataset == "CheXpertDataset":
        dataset = CheXpertDataset(
            root=root_dir,
            cache_dir=None if resize[0] > 320 else cache_dir,
            resize=resize,
            frontal_lateral="Frontal",
            ap_pa="all",
            version="original",
        )

    pred = load_and_predict(model_path=model_path, dataset=dataset, save_to_dir=Path(save_to))
    print(f"pred = {pred}")


if __name__ == "__main__":
    main()


"""
Usages:

python src/scripts/load_and_predict.py --model_path "../../ray_results/training_function_2022-05-31_21-54-38_data_aug/training_function_80ae5_00000_0_data_aug_p=1.0,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs': 1_2022-05-31_21-54-39/checkpoint_best_model/epoch=12-step=21892.ckpt" --save_pred_to "../../ray_results/training_function_2022-05-31_21-54-38_data_aug/training_function_80ae5_00000_0_data_aug_p=1.0,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs': 1_2022-05-31_21-54-39"
python src/scripts/load_and_predict.py --model_path "../../ray_results/training_function_2022-05-31_21-54-38_data_aug/training_function_80ae5_00000_0_data_aug_p=1.0,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs': 1_2022-05-31_21-54-39/checkpoint_best_model/epoch=12-step=21892.ckpt"

server:
nohup python -u train.py >> output.txt &
nohup python -u src/scripts/load_and_predict.py --model_path "/u/home/gerh/thesis/ray_results/training_function_2022-07-13_16-51-57_best_models_00008/training_function_57e43_00008_8_data_aug_p=0.75_labels_to_use=ChestTube_model=model_class_ResNet18Model_max_epochs_100_lr_5e-05_transform_image_net_batch_size_32_loss_FocalWith/checkpoint_best_model/epoch=21-step=9262.ckpt" --save_pred_to "/u/home/gerh/thesis/ray_results/training_function_2022-07-13_16-51-57_best_models_00008/training_function_57e43_00008_8_data_aug_p=0.75_labels_to_use=ChestTube_model=model_class_ResNet18Model_max_epochs_100_lr_5e-05_transform_image_net_batch_size_32_loss_FocalWith/predictions" --dataset CheXpertDataset >> outputs/output_load_and_predict_chexpert_2022-08-26_14-44.txt &

python src/scripts/load_and_predict.py --model_path "/u/home/gerh/thesis/ray_results/training_function_2022-07-05_20-21-36_best_models_00016/training_function_4e6f4_00016_16_data_aug_p=0.25,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs':_2022-07-06_07-35-08/checkpoint_best_model/epoch=15-step=6736.ckpt" --save_pred_to "/u/home/gerh/thesis/ray_results/training_function_2022-07-05_20-21-36_best_models_00016/training_function_4e6f4_00016_16_data_aug_p=0.25,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs':_2022-07-06_07-35-08/predictions"
nohup python -u src/scripts/load_and_predict.py --model_path "/u/home/gerh/thesis/ray_results/training_function_2022-07-05_20-21-36_best_models_00016/training_function_4e6f4_00016_16_data_aug_p=0.25,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs':_2022-07-06_07-35-08/checkpoint_best_model/epoch=15-step=6736.ckpt" --save_pred_to "/u/home/gerh/thesis/ray_results/training_function_2022-07-05_20-21-36_best_models_00016/training_function_4e6f4_00016_16_data_aug_p=0.25,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs':_2022-07-06_07-35-08/predictions" >> outputs/output_load_and_predict_2022-07-07_14-40.txt



nohup python -u train.py >> output.txt &
nohup python -u src/scripts/load_and_predict.py --model_path "/u/home/gerh/thesis/ray_results/training_function_2022-07-05_20-21-36_best_models_00016/training_function_4e6f4_00017_17_data_aug_p=0.5,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs': _2022-07-06_08-50-02/checkpoint_best_model/epoch=11-step=5052.ckpt" --save_pred_to "/u/home/gerh/thesis/ray_results/training_function_2022-07-05_20-21-36_best_models_00016/training_function_4e6f4_00017_17_data_aug_p=0.5,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs': _2022-07-06_08-50-02/" >> outputs/output_load_and_predict_202207-06_22-40.txt &
python src/scripts/load_and_predict.py --model_path "/u/home/gerh/thesis/ray_results/training_function_2022-07-05_20-21-36_best_models_00016/training_function_4e6f4_00017_17_data_aug_p=0.5,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs': _2022-07-06_08-50-02/checkpoint_best_model/epoch=11-step=5052.ckpt" --save_pred_to "/u/home/gerh/thesis/ray_results/training_function_2022-07-05_20-21-36_best_models_00016/training_function_4e6f4_00017_17_data_aug_p=0.5,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs': _2022-07-06_08-50-02/" 

"""
