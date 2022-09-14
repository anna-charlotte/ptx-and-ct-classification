import argparse
from pathlib import Path
from typing import List

from ptx_classification.evaluation.evaluation_on_ptx_cohorts import perform_ptx_cohorts_evaluation
from ptx_classification.models import MultiLabelModel, load_multi_label_model_from_checkpoint
from ptx_classification.utils import load_json

"""
python src/scripts/ptx_cohorts_evaluation.py
--model_paths
ray_results/pneumothorax_classification/training_function_2022-07-26_17-21-30_good_models_00004/training_function_9fea4_00004_4_chest_tube_labels=_mnt_cephstorage_users_gerh_thesis_ray_results_training_function_2022-07-13_16-51-57_best_models_00008_training_function_57e43/checkpoint_best_model/epoch=10-step=102476.ckpt
--data_dir
.

python src/scripts/ptx_cohorts_evaluation.py --model_paths ray_results/pneumothorax_classification/training_function_2022-07-26_17-21-30_good_models_00004/training_function_9fea4_00004_4_chest_tube_labels=_mnt_cephstorage_users_gerh_thesis_ray_results_training_function_2022-07-13_16-51-57_best_models_00008_training_function_57e43/checkpoint_best_model/epoch=10-step=102476.ckpt --data_dir .
python src/scripts/ptx_cohorts_evaluation.py --model_paths ray_results/training_function_2022-07-26_17-21-30_good_models/training_function_9fea4_00004_4_chest_tube_labels=_mnt_cephstorage_users_gerh_thesis_ray_results_training_function_2022-07-13_16-51-57_best_models_00008_training_function_57e43/checkpoint_best_model/epoch=10-step=102476.ckpt --data_dir /u/home/gerh/share-all --save_plot_to plots/

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
        "--model_names",
        nargs="+",  # one or more
        type=str,
        required=True,
        help="Model names.",
    )
    parser.add_argument(
        "--data_dir",
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
        "--show_plot",
        type=bool,
        required=False,
        default=False,
    )

    args = parser.parse_args()

    model_paths = args.model_paths
    model_names = args.model_names
    assert len(model_names) == len(model_paths)

    data_dir = args.data_dir
    show_plot = args.show_plot
    save_to = args.save_plot_to
    if save_to is not None:
        Path.mkdir(save_to, parents=True, exist_ok=True)

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

    models: List[MultiLabelModel] = [
        load_multi_label_model_from_checkpoint(model_path=path) for path in model_paths
    ]
    model_resizing = [
        tuple(load_json(path.parent.parent / "params.json")["resize"]) for path in model_paths
    ]
    print(f"model_resizing = {model_resizing}")

    perform_ptx_cohorts_evaluation(
        models, model_names, model_resizing, data_dir, save_to, show_plot,
    )


if __name__ == "__main__":
    main()
