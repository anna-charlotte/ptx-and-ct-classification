import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image, ImageFile
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, WeightedRandomSampler

from ptx_classification.class_weights import compute_sample_weights_from_dataset
from ptx_classification.data.datasets import AugmentationDataset, DatasetSubset, XrayDataset, \
    CiipDataset
from ptx_classification.utils import SPLITS_DIR, dicom_to_array

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CandidPtxLabels:
    CT = "Chest Tube"
    PTX = "Pneumothorax"
    RF = "Rib Fracture"
    LABELS = [CT, PTX, RF]


class CandidPtxSplits:
    N_TRAIN = 13465
    N_VAL = 1925
    N_TEST = 3846
    SPLITS = (N_TRAIN, N_VAL, N_TEST)


class CandidPtxDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: "CandidPtxDataset",
        batch_size: int,
        train_val_test_split: Union[Tuple[int, int, int], Tuple[float, float, float]],
        train_class_weights: Tuple[float, float] = None,
        train_num_samples: int = 5_000,
        train_weights_based_on_labels: List[str] = None,
        data_aug_transforms: Optional[Callable] = None,
    ) -> None:
        assert (
            len(train_val_test_split) == 3
        ), f"len(train_val_test_split) = {len(train_val_test_split)}"
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.train_class_weights = train_class_weights
        self.train_num_samples = train_num_samples
        self.train_weights_based_on_labels = train_weights_based_on_labels
        self.data_aug_transforms = data_aug_transforms

        self.candid_ptx_train, self.candid_ptx_val, self.candid_ptx_test = self.get_splits()

    def setup(self, stage: Optional[str] = None) -> None:
        self.candid_ptx_train, self.candid_ptx_val, self.candid_ptx_test = self.get_splits()

    def get_splits(
        self,
    ) -> Tuple["DatasetSubset", "DatasetSubset", "DatasetSubset"]:

        splits_file = (
            SPLITS_DIR
            / f"train_val_test_split_candid_ptx_with_ratio_{str(self.train_val_test_split).replace(' ', '')}.json"
        )
        if not Path(splits_file).is_file():
            create_train_val_test_split_and_save_to_json(self.dataset, self.train_val_test_split)
        print(f"Loading train-val-test-splits from {splits_file}")
        set2img_ids = json.load(open(splits_file, "r"))

        all_img_ids = self.dataset.image_ids
        train = DatasetSubset(
            dataset=AugmentationDataset(
                dataset=self.dataset, data_aug_transform=self.data_aug_transforms
            ),
            indices=[all_img_ids.index(img) for img in set2img_ids["train"]],
        )
        val = DatasetSubset(
            dataset=self.dataset, indices=[all_img_ids.index(img) for img in set2img_ids["val"]]
        )
        test = DatasetSubset(
            dataset=self.dataset, indices=[all_img_ids.index(img) for img in set2img_ids["test"]]
        )
        return train, val, test

    def train_dataloader(self) -> DataLoader:
        training_set = self.candid_ptx_train
        sampler = None
        if self.train_class_weights is not None:
            sample_weights = compute_sample_weights_from_dataset(
                dataset=training_set,
                based_on_class_label=self.train_weights_based_on_labels,
                class_weights=self.train_class_weights,
            )
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=self.train_num_samples)
        print(f"Will train with {len(self.candid_ptx_train)} imgs from {self.__class__.__name__}")
        return DataLoader(training_set, batch_size=self.batch_size, sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        print(f"Will validate with {len(self.candid_ptx_val)} imgs from {self.__class__.__name__}")
        return DataLoader(self.candid_ptx_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        print(f"Will test with {len(self.candid_ptx_test)} imgs from {self.__class__.__name__}")
        return DataLoader(self.candid_ptx_test, batch_size=self.batch_size)


def create_train_val_test_split_and_save_to_json(
    dataset: "CandidPtxDataset",
    train_val_test_split: Union[Tuple[int, int, int], Tuple[float, float, float]],
    save_to="",
) -> None:
    print("Create train-val-test-splits for CandidPtxDataset ...")
    patient_ids = np.unique(dataset.get_labels().loc[:, "patient_id"].values)
    indices = [i for i in range(len(patient_ids))]
    indices = shuffle(indices, random_state=np.random.RandomState(42))

    if sum(train_val_test_split) <= 1.0:
        lengths = tuple([int(len(dataset) * split) for split in train_val_test_split])
    else:
        lengths = train_val_test_split

    sets = ["train", "val", "test"]
    split2img_ids = {s: [] for s in sets}
    which_set = 0
    for i in indices:
        patient_id = patient_ids[i]
        if len(split2img_ids[sets[which_set]]) >= lengths[which_set] and which_set < len(sets) - 1:
            which_set += 1
        if patient_id in dataset.patient2img_path.keys():
            for img_path in dataset.patient2img_path[patient_id]:
                split2img_ids[sets[which_set]].append(str(img_path).split("/")[-1])

    # self.set2indices = split2img_ids
    if save_to == "":
        save_to = f"train_val_test_split_candid_ptx_with_ratio_{str(train_val_test_split).replace(' ', '')}.json"
    json.dump(split2img_ids, open(SPLITS_DIR / save_to, "w"))


class CandidPtxDataset(CiipDataset, XrayDataset):
    """
    This is the CANDID-PTX dataset.
    """

    CT = "Chest Tube"
    PTX = "Pneumothorax"
    RF = "Rib Fracture"
    LABELS = [CT, PTX, RF]

    def __init__(
        self,
        labels_to_use: list,
        root: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        verbose: bool = False,
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        print("Loading CandidPtxDataset ...")
        self.cache_dir = cache_dir
        super().__init__(root=root, cache_dir=cache_dir, verbose=verbose)

        self.data_dir = self.root / "chestxray-data/candid-ptx"
        self.image_dir = self.data_dir / "dataset"
        glob_result = list(self.image_dir.glob("*"))
        self.image_paths = [str(path) for path in glob_result]
        self.image_ids = [str(x).split("/")[-1] for x in self.image_dir.glob("*")]

        label_dir = self.data_dir
        self.ct_labels_file = self._cache(label_dir / "chest_tube.csv")
        self.ptx_labels_file = self._cache(label_dir / "Pneumothorax_reports.csv")
        self.rf_labels_file = self._cache(label_dir / "acute_rib_fracture.csv")
        self.labels_df = self._read_in_labels(labels_to_use=labels_to_use)
        self.class_labels = labels_to_use
        self.resize = resize

        self._remove_flawed_images()

    def _remove_flawed_images(self) -> None:
        all_img_paths = list(itertools.chain(*self.patient2img_path.values()))
        count_img_paths = Counter(all_img_paths)

        for img_path, count in count_img_paths.items():
            if count > 1:
                img_id = self.image_id(img_path)
                self.labels_df = self.labels_df.drop(img_id, axis=0)
                self.image_ids.remove(img_id)
                self.image_paths.remove(str(img_path))

                rm = []
                for patient, paths in self.patient2img_path.items():
                    if img_path in paths:
                        self.patient2img_path[patient].remove(img_path)
                        if len(self.patient2img_path[patient]) == 0:
                            rm.append(patient)
                for patient in rm:
                    self.patient2img_path.pop(patient)
        print(
            f"Removed {len(count_img_paths) - len(self.image_ids)} flawed image(s) out of {len(count_img_paths)} images."
        )

    @staticmethod
    def image_id(img_path: str) -> str:
        return str(img_path).split("/")[-1]

    def _read_in_labels(self, labels_to_use: List[str]) -> pd.DataFrame:
        id2labels: Dict[str, Any] = {
            id_img: {**{label: 0.0 for label in self.LABELS}, **{"patient_id": None}}
            for i, id_img in enumerate(self.image_ids)
        }

        # read in chest tube labels
        df_chest_tubes = pd.read_csv(self.ct_labels_file, delimiter=",")
        for row in df_chest_tubes["anon_SOPUID"].tolist():
            if row in id2labels.keys():
                id2labels[row][self.CT] = 1.0

        # read in rib fracture labels
        df_rib_fracture = pd.read_csv(self.rf_labels_file, delimiter=",")
        for row in df_rib_fracture["anon_SOPUID"].tolist():
            if row in id2labels.keys():
                id2labels[row][self.RF] = 1.0

        # read in pneumothorax labels and check if images are assigned to only one patient each
        patient2img_path = defaultdict(lambda: [])
        df_pneumothorax = pd.read_csv(self.ptx_labels_file, delimiter=",")
        for _, row in df_pneumothorax.iterrows():
            patient_id = int(row["filename"].split("_")[0].replace("P", ""))
            img_id = row["SOPInstanceUID"]

            if img_id in id2labels.keys():
                img_path = self.image_dir / img_id
                id2labels[img_id]["patient_id"] = patient_id
                patient2img_path[patient_id].append(img_path)

                id2labels[img_id]["patient_id"] = patient_id
                if row["EncodedPixels"] != "-1":
                    id2labels[img_id][self.PTX] = 1.0

        self.patient2img_path = {k: list(set(v)) for k, v in patient2img_path.items()}
        y = [id2labels[img_id].values() for img_id in self.image_ids]
        column_names = self.LABELS + ["patient_id"]

        df = pd.DataFrame(data=y, columns=column_names, index=self.image_ids)
        for label in self.LABELS:
            if label not in labels_to_use:
                df = df.drop(label, axis=1)
        return df

    def get_labels(self) -> pd.DataFrame:
        return self.labels_df

    def get_label_tensor(self, img_id: str) -> torch.Tensor:
        df = self.get_labels()
        return torch.from_numpy(df.loc[img_id, df.columns != "patient_id"].values)

    def load_image(self, img_path: Path, rgb: bool = True) -> torch.Tensor:
        img = dicom_to_array(path=self._cache(img_path), rgb=False).astype(np.uint8)
        pil_img = Image.fromarray(img)
        if self.resize is not None:
            pil_img = pil_img.resize(self.resize, resample=Image.Resampling.BILINEAR)
        img = np.array(pil_img)

        if rgb:
            return torch.from_numpy(np.array([img, img, img]))
        else:
            return torch.from_numpy(img)

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
        img = self.load_image(Path(img_path))
        label = self.get_label_tensor(img_id=self.image_id(img_path))
        dataset_name = self.__class__.__name__.replace("Dataset", "")
        return img, label, self.class_labels, dataset_name

    def __len__(self) -> int:
        return len(self.image_paths)
