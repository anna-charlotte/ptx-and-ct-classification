import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
from torch.nn import BCEWithLogitsLoss

from ptx_classification.data.datasets import DatasetSubset, XrayDataset
from ptx_classification.utils import REPO_ROOT_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from ptx_classification.data.candid_ptx.candid_ptx import CandidPtxLabels
from ptx_classification.models import (
    EfficientNetB0Model,
    MobileNetV2Model,
    MultiLabelModel,
    ResNet18Model,
    VGG11Model,
)


class MockDataset(XrayDataset):
    def __init__(
        self, num_images: int = 32, img_h: int = 1024, img_w: int = 1024, num_labels: int = 3
    ) -> None:
        self.images: List[torch.Tensor] = [
            torch.randint(low=0, high=256, size=(3, img_h, img_w)).float()
            for _ in range(num_images)
        ]
        self.targets: List[torch.Tensor] = [
            torch.tensor(
                [1.0 if img[0, i_label, 0] > 100 else 0.0 for i_label in range(num_labels)]
            )
            for img in self.images
        ]
        self.class_labels = [f"label {i}" for i in range(num_labels)]
        self.image_paths = [str(i) for i in range(num_images)]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], str]:
        return self.images[index], self.targets[index], self.class_labels, self.__class__.__name__

    def __len__(self) -> int:
        return len(self.images)

    def get_labels(self) -> pd.DataFrame:
        raise NotImplemented

    def load_image(self, img_path: Path, rgb: bool = True) -> torch.Tensor:
        return self.images[self.image_paths.index(str(img_path))]

    def get_image_id(self, img_path: Path) -> str:
        return f"image id {self.image_paths.index(str(img_path))}"


class MockDataModule(pl.LightningDataModule):
    def __init__(self, dataset: "MockDataset", batch_size: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_splits()

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_splits()

    def get_splits(self) -> Tuple[DatasetSubset, DatasetSubset, DatasetSubset]:
        train = DatasetSubset(
            dataset=self.dataset,
            indices=[i for i in range(len(self.dataset.image_paths))],
        )
        val = DatasetSubset(
            dataset=self.dataset,
            indices=[i for i in range(len(self.dataset.image_paths))],
        )
        test = DatasetSubset(
            dataset=self.dataset,
            indices=[i for i in range(len(self.dataset.image_paths))],
        )
        return train, val, test

    def train_dataloader(self) -> DataLoader:
        print(f"self.batch_size = {self.batch_size}")
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def check_model(model: LightningModule) -> None:
    dataset = MockDataset(num_images=1)
    imgs_tensor = torch.stack(dataset.images)
    logits = model.forward(imgs_tensor)
    assert len(logits) == len(dataset.images)
    print("Done forward")

    trainer = Trainer(
        max_epochs=2,
        devices=1,
        accelerator="cpu",
    )
    print("Created trainer")

    dataloader = DataLoader(dataset=dataset, batch_size=2)
    print("Created dataloader")

    trainer.fit(model, train_dataloaders=dataloader)
    print("Done fitting")


class TestMultiLabelModel:
    def test_predict_from_dataset_without_saving_to_csv(self) -> None:
        dataset = MockDataset(num_images=2, img_h=256, img_w=256, num_labels=1)
        labels = [f"label {i}" for i in range(3)]
        model = MultiLabelModel(
            model=ResNet18Model(pretrained=True, num_classes=len(labels)),
            lr=1e-04,
            loss=BCEWithLogitsLoss(),
            transform=None,
            labels=labels,
            trial_dir=Path(os.getcwd()),
        )
        model.predict_for_xray_dataset(dataset=dataset)

    def test_predict_from_dataset_with_saving_to_csv(self) -> None:
        dataset = MockDataset(num_images=2, img_h=256, img_w=256, num_labels=3)
        labels = [f"label {i}" for i in range(3)]
        model = MultiLabelModel(
            model=ResNet18Model(pretrained=True, num_classes=len(labels)),
            lr=1e-04,
            loss=BCEWithLogitsLoss(),
            transform=None,
            labels=labels,
            trial_dir=Path(os.getcwd()),
        )
        model.predict_for_xray_dataset(dataset=dataset, save_to_dir=REPO_ROOT_DIR)

    def test_model_ResNet18Model(self) -> None:
        labels = [f"label {i}" for i in range(3)]
        model = MultiLabelModel(
            model=ResNet18Model(pretrained=True, num_classes=len(labels)),
            lr=1e-04,
            loss=BCEWithLogitsLoss(),
            transform=None,
            labels=labels,
            trial_dir=Path(os.getcwd()),
        )
        check_model(model)

    def test_model_VGG11(self) -> None:
        labels = [f"label {i}" for i in range(3)]
        model = MultiLabelModel(
            model=VGG11Model(pretrained=True, num_classes=len(labels)),
            lr=1e-04,
            loss=BCEWithLogitsLoss(),
            transform=None,
            labels=labels,
            trial_dir=Path(os.getcwd()),
        )
        check_model(model)

    def test_model_MobileNetV2Model(self) -> None:
        labels = [f"label {i}" for i in range(3)]
        model = MultiLabelModel(
            model=MobileNetV2Model(pretrained=True, num_classes=len(labels)),
            lr=1e-04,
            loss=BCEWithLogitsLoss(),
            transform=None,
            labels=labels,
            trial_dir=Path(os.getcwd()),
        )
        check_model(model)

    def test_model_EfficientNetB0Model(self) -> None:
        labels = [f"label {i}" for i in range(3)]
        model = MultiLabelModel(
            model=EfficientNetB0Model(pretrained=True, num_classes=len(labels)),
            lr=1e-04,
            loss=BCEWithLogitsLoss(),
            transform=None,
            labels=labels,
            trial_dir=Path(os.getcwd()),
        )
        check_model(model)
