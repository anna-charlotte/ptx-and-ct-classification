import math

import torch

from ptx_classification.data.candid_ptx.candid_ptx import CandidPtxDataModule, CandidPtxDataset
from ptx_classification.utils import get_cache_dir, get_data_dir

data_dir = get_data_dir()
cache_dir = get_cache_dir()


class TestCandidPtxDataModule:
    def test_set_up(self) -> None:
        size_train = 33
        size_val = 11
        size_test = 20
        batch_size = 32

        data_module = CandidPtxDataModule(
            dataset=CandidPtxDataset(root=data_dir, labels_to_use=CandidPtxDataset.LABELS),
            batch_size=batch_size,
            train_val_test_split=(size_train, size_val, size_test),
        )
        train_dl = data_module.train_dataloader()
        val_dl = data_module.val_dataloader()
        val_features, val_labels = next(iter(val_dl))
        for feat in val_features:
            assert torch.min(feat) == 0
            assert torch.max(feat) == 255
            assert val_features[0].size() == (3, 1024, 1024)
            assert val_labels[0].size() == (3,)

        assert val_dl.batch_size == batch_size
        assert len(train_dl) == math.ceil(size_train / batch_size)
        assert len(val_dl) == math.ceil(size_val / batch_size)


n_images = 19236


class TestCandidPtx:
    def test_load_without_cache(self) -> None:
        candid_ptx = CandidPtxDataset(root=data_dir, labels_to_use=CandidPtxDataset.LABELS)
        assert candid_ptx.get_labels().shape == (n_images, 4)

    def test_load_with_cache(self) -> None:
        candid_ptx = CandidPtxDataset(
            root=data_dir, cache_dir=cache_dir, labels_to_use=CandidPtxDataset.LABELS
        )
        assert candid_ptx.get_labels().shape == (n_images, 4)

    def test_load_with_resize(self) -> None:
        candid_ptx = CandidPtxDataset(
            root=data_dir, labels_to_use=CandidPtxDataset.LABELS, resize=(256, 256)
        )
        assert candid_ptx.get_labels().shape == (n_images, 4)
        img = candid_ptx[0][0]
        assert img.size() == (3, 256, 256)

    def test_load_with_only_ct_labels(self) -> None:
        candid_ptx = CandidPtxDataset(root=data_dir, labels_to_use=[CandidPtxDataset.CT])
        assert candid_ptx.get_labels().shape == (n_images, 2)
        assert "patient_id" in candid_ptx.get_labels().columns
        assert CandidPtxDataset.CT in candid_ptx.get_labels().columns
