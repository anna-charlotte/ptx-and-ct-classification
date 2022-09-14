import json
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image, ImageFile
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, WeightedRandomSampler

from ptx_classification.class_weights import compute_sample_weights_from_dataset
from ptx_classification.data.chest_tube_data_frame import ChestTubeDataFrame
from ptx_classification.data.datasets import AugmentationDataset, DatasetSubset, XrayDataset, \
    CiipDataset
from ptx_classification.utils import SPLITS_DIR, get_date_and_time, unique

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CheXpertLabels:
    PTX = "Pneumothorax"
    LABELS = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        PTX,
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]


class CheXpertSplits:
    total = 191229
    N_TRAIN = 134_000
    N_VAL = 19_000
    N_TEST = 38_229
    SPLITS = (N_TRAIN, N_VAL, N_TEST)


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: "CheXpertDataset",
        batch_size: int,
        train_val_test_split: Union[Tuple[int, int, int], Tuple[float, float, float]],
        data_aug_transforms: Optional[Callable] = None,
        train_class_weights: Tuple[float, float] = None,
        train_num_samples: int = 10_000,
    ) -> None:
        assert (
            len(train_val_test_split) == 3
        ), f"len(train_val_test_split) = {len(train_val_test_split)}"
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.data_aug_transforms = data_aug_transforms
        self.train_class_weights = train_class_weights
        self.train_num_samples = train_num_samples

        self.chexpert_train, self.chexpert_val, self.chexpert_test = None, None, None
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        self.chexpert_train, self.chexpert_val, self.chexpert_test = self.get_splits()
        print(f"len(self.chexpert_train) = {len(self.chexpert_train)}")
        print(f"len(self.chexpert_val) = {len(self.chexpert_val)}")
        print(f"len(self.chexpert_test) = {len(self.chexpert_test)}")

    def get_splits(
        self,
    ) -> Tuple["DatasetSubset", "DatasetSubset", "DatasetSubset"]:
        splits_file = (
            SPLITS_DIR
            / f"train_val_test_split_chexpert_with_ratio_{str(self.train_val_test_split).replace(' ', '')}.json"
        )
        if not Path(splits_file).is_file():
            create_chexpert_train_val_test_split_and_save_to_json(
                self.dataset, self.train_val_test_split
            )
        print(f"Loading train-val-test-splits from {splits_file}")
        set2file_names = json.load(open(splits_file, "r"))
        file_names = [self.dataset.get_image_id(img_path) for img_path in self.dataset.image_paths]
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
        for subset in [train, val, test]:
            if subset.dataset.__class__.__name__ == AugmentationDataset.__name__:
                image_ids = [subset.dataset.dataset.image_ids[i] for i in subset.indices]
                subset_df = subset.dataset.dataset.df_labels[
                    subset.dataset.dataset.df_labels["image_id"].isin(image_ids)
                ]
            else:
                image_ids = [subset.dataset.image_ids[i] for i in subset.indices]
                subset_df = subset.dataset.df_labels[
                    subset.dataset.df_labels["image_id"].isin(image_ids)
                ]
            print(f"len(subset_df) = {len(subset_df)}")
            n_ptx_pos = len(subset_df[subset_df["Pneumothorax"] == 1.0])
            n_ptx_neg = len(subset_df[subset_df["Pneumothorax"] == 0.0])

            print(f"n_ptx_pos = {n_ptx_pos}")
            print(f"n_ptx_neg = {n_ptx_neg}")

        return train, val, test

    def train_dataloader(self) -> DataLoader:
        training_set = self.chexpert_train
        sampler = None
        if self.train_class_weights is not None:
            sample_weights = compute_sample_weights_from_dataset(
                dataset=training_set,
                based_on_class_label=["Pneumothorax"],
                class_weights=self.train_class_weights,
            )
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=self.train_num_samples)
        print(f"Will train with {len(self.chexpert_train)} images from {self.__class__.__name__}")
        return DataLoader(training_set, batch_size=self.batch_size, sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        print(f"Will validate with {len(self.chexpert_val)} images from {self.__class__.__name__}")
        return DataLoader(self.chexpert_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        print(f"Will test with {len(self.chexpert_test)} images from {self.__class__.__name__}")
        return DataLoader(self.chexpert_test, batch_size=self.batch_size)


def create_chexpert_train_val_test_split_and_save_to_json(
    dataset: "CheXpertDataset",
    train_val_test_split: Union[Tuple[int, int, int], Tuple[float, float, float]],
    save_to: str = "to_default",
) -> None:
    files = [dataset.get_image_id(img_path) for img_path in dataset.image_paths]

    patient_ids = list(set([dataset.get_patient_id(path) for path in dataset.image_paths]))
    indices = [i for i in range(len(patient_ids))]
    indices = shuffle(indices, random_state=np.random.RandomState(42))

    if sum(train_val_test_split) <= 1.0:
        lengths = tuple([int(len(dataset) * split) for split in train_val_test_split])
    else:
        assert sum(train_val_test_split) <= len(
            dataset.image_paths
        ), f"{sum(train_val_test_split)}, {len(dataset.image_paths)}"
        lengths = train_val_test_split

    patient2img_path = defaultdict(lambda: [])
    for file in files:
        patient_id = dataset.get_patient_id(file)
        patient2img_path[patient_id].append(file)

    sets = ["train", "val", "test"]
    set2files = {s: [] for s in sets}
    which_set = 0
    for i in indices:
        patient_id = patient_ids[i]
        if len(set2files[sets[which_set]]) >= lengths[which_set]:
            if which_set < len(sets) - 1:
                which_set += 1
            else:
                break
        if patient_id in patient2img_path.keys():
            for file in patient2img_path[patient_id]:
                set2files[sets[which_set]].append(file)

    if save_to == "to_default":
        save_to = f"train_val_test_split_chexpert_with_ratio_{str(train_val_test_split).replace(' ', '')}.json"
    json.dump(set2files, open(SPLITS_DIR / save_to, "w"))


class CheXpertDataset(CiipDataset, XrayDataset):
    """
    This is the CheXpert data set.
    See: https://stanfordmlgroup.github.io/competitions/chexpert/
    """

    def __init__(
        self,
        frontal_lateral: Literal["Frontal", "Lateral", "all"],
        ap_pa: Literal["AP", "PA", "all"],
        root: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        verbose: bool = False,
        version: str = "small",
        resize: Optional[Tuple[int, int]] = None,
        class_labels: Optional[List[str]] = None,
        ct_labels: Optional[ChestTubeDataFrame] = None,
        ct_cut_off: Optional[float] = 0.5,
    ) -> None:
        super().__init__(root=root, cache_dir=cache_dir, verbose=verbose)

        self.cache_dir = cache_dir
        self.versions = {"original": "CheXpert-v1.0", "small": "CheXpert-v1.0-small"}
        self.name = self.versions[version]
        self.data_dir = self.root / "chestxray-data/chexpert" / self.name
        self.frontal_lateral = frontal_lateral
        self.ap_pa = ap_pa
        self.resize = resize
        if class_labels is None:
            class_labels = CheXpertLabels.LABELS
        self.class_labels = sorted(class_labels)
        assert all(
            [cl in [*CheXpertLabels.LABELS, "Chest Tube"] for cl in class_labels]
        ), f"class_labels = {class_labels}"
        print(f"self.class_labels = self.class_labels")

        self.ct_cut_off = ct_cut_off
        self.df_ct_labels = ct_labels

        if self.df_ct_labels is not None:
            self.class_labels = sorted(unique([*self.class_labels, "Chest Tube"]))
            if ct_cut_off is None:
                raise ValueError("The chest tube cut off (ct_cut_off) has not been set.")

        self.train_labels_file = self._cache(self.data_dir / "train.csv")
        self.train_image_dir = self.data_dir / "train"
        self.train_image_paths = sorted([str(path) for path in self.train_image_dir.glob("*/*/*")])
        self.df_train = self._read_in_labels(self.train_labels_file)

        self.val_labels_file = self._cache(self.data_dir / "valid.csv")
        self.val_image_dir = self.data_dir / "valid"
        self.val_image_paths = sorted([str(path) for path in self.val_image_dir.glob("*/*/*")])
        self.df_val = self._read_in_labels(self.val_labels_file)

        # join train and val images to later apply own split later on
        self.image_paths = sorted(self.train_image_paths + self.val_image_paths)
        self.image_ids = [self.get_image_id(img_path) for img_path in self.image_paths]
        print(f"len(self.image_paths) = {len(self.image_paths)}")
        print(f"len(self.image_ids) = {len(self.image_ids)}")
        assert all(
            [
                train_col == val_col
                for train_col, val_col in zip(self.df_train.columns, self.df_val.columns)
            ]
        )
        self.df_labels = pd.concat([self.df_train, self.df_val])
        print(f"len(self.df_train) = {len(self.df_train)}")
        print(f"len(self.df_val) = {len(self.df_val)}")
        print(f"len(self.df_labels) = {len(self.df_labels)}")

        n_ptx_pos = len(self.df_labels[self.df_labels["Pneumothorax"] == 1.0])
        n_ptx_neg = len(self.df_labels[self.df_labels["Pneumothorax"] == 0.0])
        print(f"n_ptx_pos = {n_ptx_pos}")
        print(f"n_ptx_neg = {n_ptx_neg}")
        print(f"self.df_labels.columns = {self.df_labels.columns}")
        print(f"self.class_labels = {self.class_labels}")

        print(f"len(self.image_paths) = {len(self.image_paths)}")
        self.image_paths = [img_path for img_path in self.df_labels.Path.tolist()]
        self.image_ids = [self.get_image_id(img_path) for img_path in self.image_paths]
        print(f"len(self.image_paths) = {len(self.image_paths)}")

    def _get_image_path(self, path: str) -> str:
        return str(self.data_dir.parent / path)

    def _read_in_labels(
        self,
        path_csv_file,
    ):
        df = pd.read_csv(path_csv_file)
        if self.frontal_lateral != "all":
            df = df.loc[df["Frontal/Lateral"] == self.frontal_lateral]
        if self.ap_pa != "all":
            df = df.loc[df["AP/PA"] == self.ap_pa]

        df["patient_id"] = df.Path.apply(self.get_patient_id)
        df["Path"] = df.Path.apply(self._get_image_path)
        df["image_id"] = df.Path.apply(self.get_image_id)

        df = df.replace(to_replace=np.nan, value=0.0, inplace=False)
        df = df.replace(to_replace=-1.0, value=0.0, inplace=False)

        # add chest tube labels to df if given
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
        return df

    def get_labels(self):
        return self.df_labels

    def get_label_tensor(self, img_path: Path):
        df_train = self.get_labels()
        row = df_train.loc[df_train["Path"] == str(img_path)][self.class_labels].values
        row = np.reshape(row, (len(self.class_labels)))
        return torch.from_numpy(row)

    def load_image(self, img_path: Path, rgb: bool = True) -> torch.Tensor:
        pil_img = Image.open(self._cache(img_path)).convert("RGB")
        if self.resize is not None:
            pil_img = pil_img.resize(self.resize, resample=Image.Resampling.BILINEAR)
        img = np.transpose(np.array(pil_img), axes=[2, 0, 1])
        if rgb:
            return torch.from_numpy(img)
        else:
            return torch.from_numpy(img[0])

    def __getitem__(self, item):
        img_path = Path(self.image_paths[item])
        img = self.load_image(img_path, rgb=True)
        label = self.get_label_tensor(img_path=img_path)
        dataset_name = self.__class__.__name__.replace("Dataset", "")
        return img, label, self.class_labels, dataset_name

    def __len__(self) -> int:
        return len(self.df_labels)

    def get_patient_id(self, path: str) -> str:
        """
        Returns patient id for a given path or image_id.
        """
        return re.search("patient[0-9]+", path).group(0)

    def get_image_id(self, img_path: str) -> str:
        image_id = "/".join(str(img_path).split("/")[-4:])
        return image_id
