from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ptx_classification.data.candid_ptx.candid_ptx import CandidPtxDataModule
from ptx_classification.data.chestxray14.chestxray14 import ChestXray14DataModule
from ptx_classification.data.chexpert.chexpert import CheXpertDataModule
from ptx_classification.data.padchest.padchest import PadChestDataModule


class DataModuleChestTubeClassification(pl.LightningDataModule):
    def __init__(
        self, candid_ptx_datamodule: CandidPtxDataModule, padchest_datamodule: PadChestDataModule
    ) -> None:
        super().__init__()
        self.candid_ptx_datamodule = candid_ptx_datamodule
        self.padchest_datamodule = padchest_datamodule

    def train_dataloader(self) -> DataLoader:
        return self.candid_ptx_datamodule.train_dataloader()

    def val_dataloader(self) -> List[DataLoader]:
        return [
            self.candid_ptx_datamodule.val_dataloader(),
            self.padchest_datamodule.val_dataloader(),
        ]

    def test_dataloader(self) -> List[DataLoader]:
        return [
            self.candid_ptx_datamodule.test_dataloader(),
            self.padchest_datamodule.test_dataloader(),
        ]


class DataModuleChestXray14CheXpert(pl.LightningDataModule):
    def __init__(
        self, chestxray14_datamodule: ChestXray14DataModule, chexpert_datamodule: CheXpertDataModule
    ):
        super().__init__()
        self.chestxray14_datamodule = chestxray14_datamodule
        self.chexpert_datamodule = chexpert_datamodule

    def train_dataloader(self) -> List[DataLoader]:
        return [
            self.chestxray14_datamodule.train_dataloader(),
            self.chexpert_datamodule.train_dataloader(),
        ]

    def val_dataloader(self) -> List[DataLoader]:
        return [
            self.chestxray14_datamodule.val_dataloader(),
            self.chexpert_datamodule.val_dataloader(),
        ]

    def test_dataloader(self) -> List[DataLoader]:
        return [
            self.chestxray14_datamodule.test_dataloader(),
            self.chexpert_datamodule.test_dataloader(),
        ]
