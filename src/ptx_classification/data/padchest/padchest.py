import json
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image, ImageFile
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset

from ptx_classification.data.datasets import (
    AugmentationDataset,
    DatasetSubset,
    ResizedDataset,
    XrayDataset, CiipDataset,
)
from ptx_classification.utils import REPO_ROOT_DIR, SPLITS_DIR, get_date_and_time

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PadChestDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: "PadChestDataset",
        batch_size: int,
        train_val_test_split: Union[Tuple[int, int, int], Tuple[float, float, float]],
        ratio_ct_to_no_ct: Tuple[int, int] = (1, 1),
        data_aug_transforms: Optional[Callable] = None,
    ) -> None:
        assert (
            len(train_val_test_split) == 3
        ), f"len(train_val_test_split) = {len(train_val_test_split)}"
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        # train-val-test-split:
        # absolute (list of integers, each >= 1) or relative (list of floats with sum() = 1.0)
        self.train_val_test_split = train_val_test_split
        self.ratio_ct_to_no_ct = ratio_ct_to_no_ct
        self.data_aug_transforms = data_aug_transforms

        self.padchest_train, self.padchest_val, self.padchest_test = self.get_splits(
            rng=np.random.RandomState(42),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        rng = np.random.RandomState(42)
        self.padchest_train, self.padchest_val, self.padchest_test = self.get_splits(
            rng=rng,
        )

    def get_splits(
        self,
        rng: np.random.RandomState,
    ) -> Tuple["DatasetSubset", "DatasetSubset", "DatasetSubset"]:
        age = self.dataset.min_age
        proj_labels = [label.replace("horizontal", "h") for label in self.dataset.projection_labels]
        ct_ratio = self.ratio_ct_to_no_ct

        splits_file = (
            SPLITS_DIR
            / f"train_val_test_split_padchest_with_ratio_{self.train_val_test_split}_ct-ratio_{ct_ratio}_min-age_{age}_proj-labels_{proj_labels}.json".replace(
                " ", ""
            )
        )
        if Path(splits_file).is_file():
            print(f"Loading train-val-test-splits from {splits_file}")
            set2img_ids = json.load(open(splits_file, "r"))
        else:
            ratio = (
                self.ratio_ct_to_no_ct[0] / sum(self.ratio_ct_to_no_ct),
                self.ratio_ct_to_no_ct[1] / sum(self.ratio_ct_to_no_ct),
            )
            print(f"ratio = {ratio}")
            category2image = self.dataset._get_images_for_subset(
                rng=rng, ratio_ct_to_no_ct=self.ratio_ct_to_no_ct
            )
            imgs_with_ct = category2image["with Chest Tube"]
            imgs_without_ct = category2image["without Chest Tube"]
            print(f"len(imgs_with_ct) = {len(imgs_with_ct)}")
            print(f"len(imgs_without_ct) = {len(imgs_without_ct)}")

            if sum(self.train_val_test_split) <= 1.0:
                num_imgs = len(imgs_with_ct) + ratio[1] / ratio[0] * len(imgs_with_ct)
                lengths = tuple([int(num_imgs * split) for split in self.train_val_test_split])
            else:
                lengths = self.train_val_test_split

            sets = ["train", "val", "test"]
            set2img_ids = {s: [] for s in sets}
            for images, percentage in [(imgs_with_ct, ratio[0]), (imgs_without_ct, 1)]:
                current_set = 0
                for img in images:
                    if len(set2img_ids[sets[current_set]]) >= lengths[current_set] * percentage:
                        if current_set < len(sets) - 1:
                            current_set += 1
                        else:
                            break

                    set2img_ids[sets[current_set]].append(img)

            if not Path(splits_file).is_file():
                print(f"Saving train-val-test-splits to {splits_file}")
                with open(splits_file, "w") as file:
                    json.dump(set2img_ids, file)

        train_indices = [self.dataset.image_ids.index(img) for img in set2img_ids["train"]]
        val_indices = [self.dataset.image_ids.index(img) for img in set2img_ids["val"]]
        test_indices = [self.dataset.image_ids.index(img) for img in set2img_ids["test"]]

        self.set2indices = {"train": train_indices, "val": val_indices, "test": test_indices}

        train = DatasetSubset(
            dataset=AugmentationDataset(
                dataset=self.dataset, data_aug_transform=self.data_aug_transforms
            ),
            indices=train_indices,
        )
        val = DatasetSubset(dataset=self.dataset, indices=val_indices)
        test = DatasetSubset(dataset=self.dataset, indices=test_indices)
        print("Length of PadChest DataModule Subsets")
        print(f"len(train) = {len(train)}")
        print(f"len(val) = {len(val)}")
        print(f"len(test) = {len(test)}")
        return train, val, test

    def train_dataloader(self) -> DataLoader:
        print(f"Will train with {len(self.padchest_train)} images from {self.__class__.__name__}")
        return DataLoader(self.padchest_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        print(f"Will validate with {len(self.padchest_val)} images from {self.__class__.__name__}")
        return DataLoader(self.padchest_val, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        print(f"Will test with {len(self.padchest_test)} images from {self.__class__.__name__}")
        return DataLoader(self.padchest_test, batch_size=1)


class PadChestProjectionLabels:
    AP = "AP"
    AP_h = "AP_horizontal"
    COSTAL = "COSTAL"
    EXCLUDE = "EXCLUDE"
    L = "L"
    PA = "PA"
    UNK = "UNK"
    LABELS = [AP, AP_h, COSTAL, EXCLUDE, L, PA, UNK]


class PadChestLabels:
    CT = "Chest Tube"
    LABELS = [CT]


class PadChestDataset(CiipDataset, XrayDataset):
    """
    This is the PadChest dataset with only including the label 'Chest Tube'.
    """

    def __init__(
        self,
        projection_labels: List[str],
        min_age: int = 0,
        root: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        verbose: bool = False,
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        print("Loading PadChestDataset ...")
        self.cache_dir = cache_dir
        super().__init__(root=root, cache_dir=cache_dir, verbose=verbose)

        self.data_dir = self.root / "chestxray-data" / "padchest" / "BIMCV-PadChest-FULL"
        self.image_dir = self.data_dir / "images"
        glob_result = list(self.image_dir.glob("*.png"))
        self.image_paths = [str(path) for path in glob_result]
        self.image_ids = [str(x).split("/")[-1] for x in self.image_dir.glob("*")]
        self.resize = resize

        self.copy_to = (
            str(
                REPO_ROOT_DIR.parent
                / "data_samples"
                / "padchest_subsets"
                / f"labels_{projection_labels}_min_age_{min_age}_{get_date_and_time()}"
            )
            .replace(" ", "")
            .replace("'", "")
        )
        if not os.path.exists(self.copy_to):
            os.mkdir(self.copy_to)
        label_file = self.data_dir / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
        self.class_labels = PadChestLabels.LABELS
        self.projection_labels = projection_labels
        self.min_age = min_age
        self.labels_df = self._read_in_labels(
            label_file,
            projection_labels=projection_labels,
            min_age=min_age,
        )

    @staticmethod
    def image_id(img_path: str) -> str:
        return str(img_path).split("/")[-1]

    def _read_in_labels(
        self,
        file,
        projection_labels: List[str],
        min_age: int = 0,
    ) -> pd.DataFrame:
        df = pd.read_csv(file, dtype=str)[
            [
                "ImageID",
                "Labels",
                "ViewPosition_DICOM",
                "Projection",
                "PatientBirth",
                "StudyDate_DICOM",
            ]
        ]
        df["PatientBirth"] = pd.to_numeric(df["PatientBirth"], downcast="integer")
        df["StudyDate_DICOM"] = df["StudyDate_DICOM"].astype(np.float32) / 10000
        df = df.loc[df["StudyDate_DICOM"] - df["PatientBirth"] >= min_age]

        print(f"min_age = {min_age}")
        print(f"projection_labels = {projection_labels}")
        projection = [str(p) for p in projection_labels]
        df = df.loc[df["Projection"].isin(projection)]
        df = pd.DataFrame(data=df.values, columns=df.columns)
        print(f"After extracting frontal images: len(df) = {len(df)}")

        labels = []
        rows_to_remove = []
        for i, row in enumerate(list(df["Labels"])):
            if type(row) is not str and math.isnan(row):
                rows_to_remove.append(i)
            else:
                l = row.replace("[", "").replace("]", "").replace("'", "").split(",")
                for x in [*l]:
                    labels.append(x.strip())
        print(f"Removing {len(rows_to_remove)} due their label being NaN")
        for rm in rows_to_remove:
            df = df.drop(index=rm)

        labels = sorted(list(set(labels)))
        data = np.zeros(shape=(len(df), len(labels)))

        for i, row in enumerate(list(df["Labels"])):
            for j, label in enumerate(labels):
                if label in row:
                    data[i, j] = 1.0

        labels_df = pd.DataFrame(data=data, columns=labels, index=list(df["ImageID"]))
        labels_df = (
            labels_df["chest drain tube"]
            .to_frame()
            .rename(columns={"chest drain tube": "Chest Tube"})
        )
        return labels_df

    def get_labels(self) -> pd.DataFrame:
        return self.labels_df

    def get_label_tensor(self, img_id: str) -> torch.Tensor:
        df = self.get_labels()
        return torch.from_numpy(df.loc[img_id].values)

    def load_image(self, img_path: Path, rgb: bool = True) -> torch.Tensor:
        img = Image.open(self._cache(img_path))
        if self.resize is not None:
            img = img.resize(self.resize, resample=Image.Resampling.BILINEAR)
        pixels = np.array(img)
        pixels = (255 * pixels / np.amax(pixels)).astype(np.uint8)
        # subprocess.call(["cp", img_path, self.copy_to])

        if rgb:
            return torch.from_numpy(np.array([pixels, pixels, pixels]))
        else:
            return torch.from_numpy(pixels)

    def get_image_id(self, img_path: Path) -> str:
        return str(img_path).split("/")[-1]

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], str]:
        """
        Args:
            item:
                Integer, index of image.
        Returns:
            img:    torch.tensor of shape (3, H, W) of the given image.
            label:  torch.tensor of shape (len(self.LABELS)) consisting of 0. and/or 1.

            Due to limitations of PyTorch Lightning we also want to return the labels that are
            represented within the dataset (self.LABELS) as well as the datasets name (dataset_name)
            to be able to access this information during evaluation.

        """
        img_path = self.image_paths[item]
        img = self.load_image(Path(img_path), rgb=True)
        label = self.get_label_tensor(img_id=self.image_id(img_path))
        dataset_name = self.__class__.__name__.replace("Dataset", "")
        return img, label, self.class_labels, dataset_name

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_subset_of_dataset(
        self,
        rng: np.random.RandomState,
        resize_images: Tuple[int, int] = None,
    ) -> Dataset:
        set2indices = self._get_images_for_subset(rng=rng)
        imgs_with_ct = set2indices["with Chest Tube"]
        imgs_without_ct = set2indices["without Chest Tube"]
        indices = [self.image_ids.index(img) for img in [*imgs_with_ct, *imgs_without_ct]]

        if resize_images is None:
            dataset_subset = DatasetSubset(dataset=self, indices=indices)
            return dataset_subset
        else:
            resized_dataset = ResizedDataset(
                dataset=self,
                indices=indices,
                new_size=resize_images,
            )
            return resized_dataset

    def _get_images_for_subset(
        self, rng: np.random.RandomState, ratio_ct_to_no_ct: Tuple[int, int] = (1, 1)
    ) -> Dict[str, List[str]]:
        labels = self.get_labels()
        imgs_with_ct = labels.loc[labels[PadChestLabels.CT] == 1.0].index.tolist()
        imgs_without_ct = labels.loc[labels[PadChestLabels.CT] == 0.0].index.tolist()
        imgs_without_ct = shuffle(imgs_without_ct, random_state=rng)

        imgs_with_ct = imgs_with_ct[:]
        imgs_without_ct = imgs_without_ct[
            : int(len(imgs_with_ct) * ratio_ct_to_no_ct[1] / ratio_ct_to_no_ct[0])
        ]
        leftover_imgs_without_ct = imgs_without_ct[
            int(len(imgs_with_ct) * ratio_ct_to_no_ct[1] / ratio_ct_to_no_ct[0]) :
        ]
        leftover_img = 0
        imgs_without_ct_add = []
        imgs_without_ct_rm = []

        for i, img in enumerate([*imgs_with_ct, *imgs_without_ct]):
            try:
                img_path = self.image_dir / img
                self.load_image(Path(img_path))
            except:
                if img in imgs_with_ct:
                    # in this case we dont have to remove any images from w/o
                    imgs_with_ct.remove(img)
                if img in imgs_without_ct:
                    # in this case we have to remove the flawed image and add a new on from
                    # leftover_imgs_without
                    for i in range(leftover_img, len(leftover_imgs_without_ct)):
                        try:
                            img_path = self.image_dir / leftover_imgs_without_ct[i]
                            self.load_image(Path(img_path))
                            imgs_without_ct_add.append(leftover_imgs_without_ct[i])
                            break
                        except:
                            pass

                    imgs_without_ct_rm.append(img)

        imgs_with_ct = imgs_with_ct[:]
        for x in imgs_without_ct_rm:
            if x in imgs_without_ct:
                imgs_without_ct.remove(x)
        imgs_without_ct += imgs_without_ct_add

        imgs_without_ct = imgs_without_ct[
            : int(len(imgs_with_ct) * ratio_ct_to_no_ct[1] / ratio_ct_to_no_ct[0])
        ]

        print(f"Images with Chest Tube: {len(imgs_with_ct)}")
        print(f"Images without Chest Tube: {len(imgs_without_ct)} out of {len(imgs_without_ct)}")
        return {
            "with Chest Tube": imgs_with_ct,
            "without Chest Tube": imgs_without_ct,
        }
