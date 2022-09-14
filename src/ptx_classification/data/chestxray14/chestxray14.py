import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, WeightedRandomSampler

from ptx_classification.class_weights import compute_sample_weights_from_dataset
from ptx_classification.data import bbox
from ptx_classification.data.chest_tube_data_frame import ChestTubeDataFrame
from ptx_classification.data.datasets import AugmentationDataset, DatasetSubset, XrayDataset, \
    CiipDataset
from ptx_classification.utils import SPLITS_DIR, tensor_to_pil, unique


class ChestXray14Labels:
    PTX = "Pneumothorax"
    LABELS = [
        "Cardiomegaly",
        "Emphysema",
        "Effusion",
        "No Finding",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Atelectasis",
        PTX,
        "Pleural_Thickening",
        "Pneumonia",
        "Fibrosis",
        "Edema",
        "Consolidation",
    ]


class ChestXray14Splits:
    N_TRAIN = 74523
    N_VAL = 12000
    N_TEST = 25595
    SPLITS = (N_TRAIN, N_VAL, N_TEST)


class ChestXray14DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: "ChestXray14Dataset",
        batch_size: int,
        train_val_split: Tuple[int, int],
        data_aug_transforms: Optional[Callable] = None,
        train_class_weights: Tuple[float, float] = None,
        train_num_samples: int = 10_000,
    ) -> None:
        assert len(train_val_split) == 2, f"len(train_val_split) = {len(train_val_split)}"
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.data_aug_transforms = data_aug_transforms
        self.train_class_weights = train_class_weights
        self.train_num_samples = train_num_samples

        self.chestxray14_train, self.chestxray14_val, self.chestxray14_test = None, None, None
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        self.chestxray14_train, self.chestxray14_val, self.chestxray14_test = self.get_splits()

    def get_splits(
        self,
    ) -> Tuple["DatasetSubset", "DatasetSubset", "DatasetSubset"]:
        splits_file = (
            SPLITS_DIR
            / f"train_val_test_split_chestxray14_with_ratio_{str(self.train_val_split).replace(' ', '')}.json"
        )
        if not Path(splits_file).is_file():
            create_chestxray14_train_val_test_split_and_save_to_json(
                self.dataset, self.train_val_split
            )
        print(f"Loading train-val-test-splits from {splits_file}")
        set2file_names = json.load(open(splits_file, "r"))

        file_names = [img.split("/")[-1] for img in self.dataset.image_paths]
        train = DatasetSubset(
            dataset=AugmentationDataset(
                dataset=self.dataset, data_aug_transform=self.data_aug_transforms
            ),
            indices=[file_names.index(img_file) for img_file in set2file_names["train"]],
        )
        val = DatasetSubset(
            dataset=self.dataset,
            indices=[file_names.index(img_file) for img_file in set2file_names["val"]],
            ground_truth_cut_off=self.dataset.ct_cut_off,
        )
        test = DatasetSubset(
            dataset=self.dataset,
            indices=[file_names.index(img_file) for img_file in set2file_names["test"]],
            ground_truth_cut_off=self.dataset.ct_cut_off,
        )
        return train, val, test

    def train_dataloader(self) -> DataLoader:
        training_set = self.chestxray14_train
        sampler = None
        if self.train_class_weights is not None:
            sample_weights = compute_sample_weights_from_dataset(
                dataset=training_set,
                based_on_class_label=["Pneumothorax"],
                class_weights=self.train_class_weights,
            )
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=self.train_num_samples)
        print(f"Will train with {len(self.chestxray14_train)} imgs from {self.__class__.__name__}")
        return DataLoader(training_set, batch_size=self.batch_size, sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        print(f"Will validate with {len(self.chestxray14_val)} imgs from {self.__class__.__name__}")
        return DataLoader(self.chestxray14_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        print(f"Will test with {len(self.chestxray14_test)} imgs from {self.__class__.__name__}")
        return DataLoader(self.chestxray14_test, batch_size=self.batch_size)


class ChestXray14Dataset(CiipDataset, XrayDataset):
    """This is the ChestX-ray 14 data set.

    See: https://arxiv.org/abs/1705.02315
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        verbose: bool = False,
        resize: Optional[Tuple[int, int]] = None,
        class_labels: Optional[List[str]] = None,
        ct_labels: Optional[ChestTubeDataFrame] = None,
        ct_cut_off: Optional[float] = 0.5,
    ) -> None:
        super().__init__(root, cache_dir=cache_dir, verbose=verbose)
        self.resize = resize
        self.data_dir = self.root / "chestxray-data/chestxray14"
        self.cache_dir = cache_dir

        self.labels_path = self._cache(self.data_dir / "Data_Entry_2017_v2020.csv")
        self.image_dir = self.data_dir / "images"
        self.image_paths = [str(path) for path in list(self.image_dir.glob("*"))]

        self.split_train_val_images_path = self.data_dir / "train_val_list.txt"
        self.split_test_images_path = self.data_dir / "test_list.txt"

        with open(self.split_test_images_path, "r") as file:
            test_images = [str(self.image_dir / line.strip()) for line in file.readlines()]
        with open(self.split_train_val_images_path, "r") as file:
            train_val_images = [str(self.image_dir / line.strip()) for line in file.readlines()]

        self.image_paths = test_images + train_val_images

        if class_labels is None:
            self.class_labels = sorted(ChestXray14Labels.LABELS)
        else:
            self.class_labels = sorted(class_labels)

        self.ct_cut_off = ct_cut_off
        self.df_ct_labels = ct_labels

        if self.df_ct_labels is not None:
            self.class_labels = sorted(unique([*self.class_labels, "Chest Tube"]))
            if ct_cut_off is None:
                raise ValueError("The chest tube cut off (ct_cut_off) has not been set.")

        self.df_labels = self._read_in_labels()

        self.bbox_path = self.data_dir / "BBox_List_2017.csv"
        self.df_test_w_bbox = self._read_bbox_file()

    def _read_in_labels(self) -> pd.DataFrame:
        print("Reading in labels ...")
        df = pd.read_csv(self.labels_path)

        # Add disease labels as columns instead of string-based "Disease1|Disease2".
        for label in self.class_labels:
            df[label] = self._get_disease_label(df, label)
        df.drop("Finding Labels", axis=1, inplace=True)

        df["Path"] = df["Image Index"].apply(self._get_image_path)
        df["patient_id"] = df["Path"].apply(self.get_patient_id)
        df = df.rename({"Image Index": "image_id"}, axis=1)
        print(f"len(df) = {len(df)}")
        df_ct = self.df_ct_labels
        if df_ct is not None:
            ct_col = []
            for index, row in df.iterrows():
                img_id = row["image_id"]
                if img_id in df_ct.df["image_id"].tolist():
                    x = df_ct.df.loc[df_ct.df["image_id"] == img_id]["Chest Tube"].values[0]
                    ct_col.append(x)
                else:
                    raise ValueError(
                        f"Image with id {img_id} is missing in the given ChestTubeDataFrame"
                    )
            df["Chest Tube"] = ct_col
        print(f"df = {df}")

        self.df_train_val = df.loc[
            df["image_id"].isin(self._read_file(self.split_train_val_images_path))
        ]
        self.df_test = df.loc[df["image_id"].isin(self._read_file(self.split_test_images_path))]
        print("Finished reading in labels ...")
        return df

    def print_label_distributions(self) -> None:
        df = self.get_labels()
        for label in ChestXray14Labels.LABELS:
            train_ptx_positive = self.df_train_val[self.df_train_val[label] == 1.0]
            n_train_ptx_positive = len(train_ptx_positive)
            n_ids_train_ptx_positive = len(set(train_ptx_positive["Patient ID"].tolist()))

            train_ptx_negative = self.df_train_val[self.df_train_val[label] == 0.0]
            n_train_ptx_negative = len(train_ptx_negative)
            n_ids_train_ptx_negative = len(set(train_ptx_negative["Patient ID"].tolist()))

            self.df_test = df.loc[df["image_id"].isin(self._read_file(self.split_test_images_path))]
            test_ptx_positive = self.df_test[self.df_test[label] == 1.0]
            n_test_ptx_positive = len(test_ptx_positive)
            n_ids_test_ptx_positive = len(set(test_ptx_positive["Patient ID"].tolist()))

            test_ptx_negative = self.df_test[self.df_test[label] == 0.0]
            n_test_ptx_negative = len(test_ptx_negative)
            n_ids_test_ptx_negative = len(set(test_ptx_negative["Patient ID"].tolist()))

            n_train = n_train_ptx_positive + n_train_ptx_negative
            n_test = n_test_ptx_positive + n_test_ptx_negative

            n_ids_train = n_ids_train_ptx_positive + n_ids_train_ptx_negative
            n_ids_test = n_ids_test_ptx_positive + n_ids_test_ptx_negative

            pos_part_train_imgs = n_train_ptx_positive / n_train
            pos_part_train_ids = n_ids_train_ptx_positive / n_ids_train
            pos_part_test_imgs = n_test_ptx_positive / n_test
            pos_part_test_ids = n_ids_test_ptx_positive / n_ids_test

            print(f"\n\nlabel = {label}")
            print(f"\npos_part_train_imgs = {pos_part_train_imgs}")
            print(f"pos_part_test_imgs = {pos_part_test_imgs}")
            print(f"\npos_part_train_ids = {pos_part_train_ids}")
            print(f"pos_part_test_ids = {pos_part_test_ids}")

    def _read_file(self, path) -> List[str]:
        with open(path) as f:
            return f.read().split("\n")

    def _get_disease_label(self, df: pd.DataFrame, disease: str) -> pd.Series:
        return df["Finding Labels"].apply(lambda label: 1.0 if disease in label else 0.0)

    def _get_image_path(self, path: str) -> str:
        return str(self.image_dir / path)

    def _read_bbox_file(self) -> pd.DataFrame:
        """Test images with BBox annotation."""
        self.bbox_path = self._cache(self.data_dir / "BBox_List_2017.csv")
        df = pd.read_csv(self.bbox_path)

        # Remove empty columns.
        df.drop(columns=df.columns[6:], inplace=True)
        df.columns = ["image_id", "Finding Label", "x", "y", "width", "height"]
        # Some images have multiple BBoxes!
        df = df.merge(self.df_labels[["image_id", "Path"]])
        return df

    def draw_bbox(self, img_path: Path, class_label: str) -> Image:
        df = self.df_test_w_bbox
        sample = df[df["Path"] == str(img_path)][df["Finding Label"] == class_label].squeeze()
        return bbox.draw_bbox(
            tensor_to_pil(self.load_image(img_path, resize=(-1, -1))),
            x_min=sample.x,
            y_min=sample.y,
            x_max=sample.x + sample.width,
            y_max=sample.y + sample.height,
            outline="turquoise",
            width=4,
        )

    def load_image(
        self, img_path: Path, rgb: bool = True, resize: Tuple[int, int] = None
    ) -> torch.Tensor:
        """

        Args:
            img_path:
                Path to image file
            rgb:
                bool, set to True to return tensor, of shape (3, H, W), else (H, W)
            resize:
                Tuple of two integers: (H, W). Set to (-1, -1) to not do any resizing and to
                ignore self.resize.

        Returns:
            img:    torch.tensor of shape (3, H, W) of the given image.

        """
        pil_img = Image.open(self._cache(img_path)).convert("RGB")
        if resize != (-1, -1):
            if resize is not None:
                pil_img = pil_img.resize(resize, resample=Image.Resampling.BILINEAR)
            elif self.resize is not None:
                pil_img = pil_img.resize(self.resize, resample=Image.Resampling.BILINEAR)
        img = np.transpose(np.array(pil_img), axes=[2, 0, 1])
        if rgb:
            return torch.from_numpy(img)
        else:
            return torch.from_numpy(img[0])

    def get_label_tensor(self, img_path: Path) -> torch.Tensor:
        df_train = self.get_labels()
        row = df_train.loc[df_train["Path"] == str(img_path)][self.class_labels].values
        row = np.reshape(row, (len(self.class_labels)))
        return torch.from_numpy(row)

    def get_labels(self) -> pd.DataFrame:
        return self.df_labels

    @staticmethod
    def get_patient_id(img_path: Path) -> int:
        patient_id = int(str(img_path).split("/")[-1].split("_")[0])
        return patient_id

    def get_image_id(self, img_path: Path) -> str:
        image_id = str(img_path).split("/")[-1]
        return image_id

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], str]:
        img_path = self.image_paths[item]
        img = self.load_image(Path(img_path), rgb=True)
        label = self.get_label_tensor(img_path=Path(img_path))
        dataset_name = self.__class__.__name__.replace("Dataset", "")
        return img, label, self.class_labels, dataset_name

    def __len__(self) -> int:
        return len(self.image_paths)


def create_chestxray14_train_val_test_split_and_save_to_json(
    dataset: ChestXray14Dataset,
    train_val_split: Union[Tuple[int, int], Tuple[float, float]],
    save_to: str = "to_default",
) -> None:
    with open(dataset.split_test_images_path, "r") as file:
        test_images = [line.strip() for line in file.readlines()]
    with open(dataset.split_train_val_images_path, "r") as file:
        train_val_images = [line.strip() for line in file.readlines()]

    patient_ids_train_val = list(
        set([dataset.get_patient_id(Path(img_path)) for img_path in train_val_images])
    )
    indices = [i for i in range(len(patient_ids_train_val))]
    indices = shuffle(indices, random_state=np.random.RandomState(42))
    if sum(train_val_split) <= 1.0:
        lengths = tuple([int(len(dataset) * split) for split in train_val_split])
    else:
        lengths = train_val_split

    patient2img_path = defaultdict(lambda: [])
    for img in dataset.image_paths:
        patient_id = dataset.get_patient_id(Path(img))
        patient2img_path[patient_id].append(img.split("/")[-1])

    sets = ["train", "val"]
    set2files = {s: [] for s in sets}
    which_set = 0
    for i in indices:
        patient_id = patient_ids_train_val[i]
        if len(set2files[sets[which_set]]) >= lengths[which_set]:
            if which_set < len(sets) - 1:
                which_set += 1
            else:
                break
        if patient_id in patient2img_path.keys():
            for img_path in patient2img_path[patient_id]:
                set2files[sets[which_set]].append(img_path.split("/")[-1])

    set2files["test"] = test_images
    if save_to == "to_default":
        save_to = f"train_val_test_split_chestxray14_with_ratio_{str(train_val_split).replace(' ', '')}.json"
    json.dump(set2files, open(SPLITS_DIR / save_to, "w"))
