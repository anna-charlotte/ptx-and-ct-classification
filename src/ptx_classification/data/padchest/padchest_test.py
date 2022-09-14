import numpy as np
import pytorch_lightning as pl
import torch

from ptx_classification.data.padchest.padchest import (
    PadChestDataModule,
    PadChestDataset,
    PadChestProjectionLabels,
)
from ptx_classification.utils import get_cache_dir, get_data_dir

data_dir = get_data_dir()
cache_dir = get_cache_dir()


class TestPadChestDataModule:
    def test_set_up(self) -> None:
        size_train = 8
        size_val = 4
        size_test = 2
        batch_size = 1
        data_module = PadChestDataModule(
            dataset=PadChestDataset(root=data_dir, projection_labels=[PadChestProjectionLabels.AP]),
            batch_size=batch_size,
            train_val_test_split=(size_train, size_val, size_test),
        )
        train_dl = data_module.train_dataloader()
        val_dl = data_module.val_dataloader()
        test_dl = data_module.test_dataloader()
        val_features, val_labels, _, _ = next(iter(val_dl))
        assert val_labels[0].size() == (1,)

        for feat in val_features:
            assert torch.min(feat) == 0
            assert torch.max(feat) == 255

        assert train_dl.batch_size == batch_size

    def test_train_val_test_split_relative_vs_absolute(self) -> None:
        batch_size = 1
        data_module_relative = PadChestDataModule(
            dataset=PadChestDataset(
                root=data_dir, projection_labels=PadChestProjectionLabels.LABELS
            ),
            batch_size=batch_size,
            train_val_test_split=(8 / 14, 4 / 14, 2 / 14),
        )
        data_module_absolute = PadChestDataModule(
            dataset=PadChestDataset(
                root=data_dir, projection_labels=PadChestProjectionLabels.LABELS
            ),
            batch_size=batch_size,
            train_val_test_split=(33, 16, 8),
        )
        assert np.all(
            data_module_relative.set2indices["train"] == data_module_absolute.set2indices["train"]
        )
        assert np.all(
            data_module_relative.set2indices["val"] == data_module_absolute.set2indices["val"]
        )
        assert np.all(
            data_module_relative.set2indices["test"] == data_module_absolute.set2indices["test"]
        )

    def test_get_splits_deterministic(self) -> None:
        size_train = 8
        size_val = 4
        size_test = 2
        batch_size = 1

        pl.seed_everything(42, workers=True)
        data_module_1 = PadChestDataModule(
            dataset=PadChestDataset(
                root=data_dir, projection_labels=PadChestProjectionLabels.LABELS
            ),
            batch_size=batch_size,
            train_val_test_split=(size_train, size_val, size_test),
        )

        pl.seed_everything(42, workers=True)
        data_module_2 = PadChestDataModule(
            dataset=PadChestDataset(
                root=data_dir, projection_labels=PadChestProjectionLabels.LABELS
            ),
            batch_size=batch_size,
            train_val_test_split=(size_train, size_val, size_test),
        )
        assert np.all(data_module_1.set2indices["train"] == data_module_2.set2indices["train"])
        assert np.all(data_module_1.set2indices["val"] == data_module_2.set2indices["val"])
        assert np.all(data_module_1.set2indices["test"] == data_module_2.set2indices["test"])


class TestPadChestDataset:
    def test_load_without_resizing(self) -> None:
        padchest = PadChestDataset(
            root=data_dir, cache_dir=cache_dir, projection_labels=PadChestProjectionLabels.LABELS
        )

    def test_load_with_resizing(self) -> None:
        padchest = PadChestDataset(
            root=data_dir,
            cache_dir=cache_dir,
            projection_labels=PadChestProjectionLabels.LABELS,
            resize=(256, 256),
        )
        img = padchest[0][0]
        assert img.size() == (3, 256, 256)
