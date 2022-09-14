from pathlib import Path

import pandas as pd
from torch.nn import BCEWithLogitsLoss

from ptx_classification.data.chest_tube_data_frame import (
    ChestTubeDataFrame,
    read_in_csv_as_chest_tube_data_frame,
)
from ptx_classification.data.chestxray14.chestxray14 import (
    ChestXray14Dataset,
    ChestXray14Labels,
    create_chestxray14_train_val_test_split_and_save_to_json,
)
from ptx_classification.models import MultiLabelModel, ResNet18Model
from ptx_classification.utils import get_data_dir

data_dir = get_data_dir()


class TestChestXray14Dataset:
    def test_load(self) -> None:
        chestxray14 = ChestXray14Dataset(root=data_dir)
        df_labels = chestxray14.get_labels()

        assert len(chestxray14.image_paths) == 24
        assert len(chestxray14.get_labels()) == 112120

    def test_load_with_chest_tube_labels(self) -> None:
        ct_labels = ChestTubeDataFrame(
            df=pd.DataFrame(
                data=[
                    ["00000001_000.png", 0.5],
                    ["00000001_001.png", 0.5],
                    ["00000001_002.png", 0.5],
                    ["00000002_000.png", 0.7],
                ],
                columns=["image_id", "Chest Tube"],
            )
        )
        ct_labels = read_in_csv_as_chest_tube_data_frame(
            Path(
                "/Users/x/Desktop/bioinformatik/thesis/bachelor-thesis-charlotte-gerhaher/ray_results/training_function_4e6f4_00016_16_data_aug_p=0.25,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs':_2022-07-06_07-35-08/predictions/prediction_for_ChestXray14Dataset.csv"
            )
        )
        chestxray14 = ChestXray14Dataset(root=data_dir, ct_labels=ct_labels)
        assert len(chestxray14.image_paths) == 24
        assert len(chestxray14.get_labels()) == 112120

    def test_training(self) -> None:
        ct_labels = read_in_csv_as_chest_tube_data_frame(
            path=Path(
                "/Users/x/Desktop/bioinformatik/thesis/bachelor-thesis-charlotte-gerhaher/ray_results/training_function_4e6f4_00016_16_data_aug_p=0.25,labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs':_2022-07-06_07-35-08/predictions/prediction_for_ChestXray14Dataset.csv"
            )
        )
        chestxray14 = ChestXray14Dataset(root=data_dir, ct_labels=ct_labels)
        labels = [ChestXray14Labels.PTX, "Chest Tube"]
        model = MultiLabelModel(
            model=ResNet18Model(pretrained=True, num_classes=len(labels)),
            lr=1e-04,
            loss=BCEWithLogitsLoss(),
            transform=None,
            labels=labels,
        )
        model.predict_for_xray_dataset(dataset=chestxray14)

    def test_draw_box(self) -> None:
        chestxray14 = ChestXray14Dataset(root=data_dir)
        df = chestxray14.df_test_w_bbox
        ptx_pos_paths = df[df["Finding Label"] == "Pneumothorax"]["Path"].tolist()
        for img_path in ptx_pos_paths:
            img_with_bb = chestxray14.draw_bbox(Path(img_path), class_label="Pneumothorax")
            img_with_bb.show()
            break


def test_create_train_val_test_split_and_save_to_json() -> None:
    chestxray14 = ChestXray14Dataset(root=data_dir)
    create_chestxray14_train_val_test_split_and_save_to_json(
        dataset=chestxray14, train_val_split=(0.5, 0.5)
    )
