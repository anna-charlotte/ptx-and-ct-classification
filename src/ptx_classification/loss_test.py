import os
from pathlib import Path

import torch

from ptx_classification.data.candid_ptx.candid_ptx import CandidPtxLabels
from ptx_classification.loss import FocalWithLogitsLoss, binary_focal_loss
from ptx_classification.models import MultiLabelModel, ResNet18Model


class TestFocalLoss:
    def test_init(self) -> None:
        focal_loss = FocalWithLogitsLoss(reduction="mean")

    def test_binary_focal_loss(self) -> None:
        logits = torch.tensor([[0.2076], [0.2076]])
        loss = binary_focal_loss(logits, torch.tensor([[1.0], [0.0]]), reduction="none")
        assert loss.size() == (2,)

    def test_focal_loss_reduction_none(self) -> None:
        focal_loss_fn = FocalWithLogitsLoss(alpha=1.0, reduction="none")
        model = MultiLabelModel(
            model=ResNet18Model(pretrained=True, num_classes=len(CandidPtxLabels.LABELS)),
            lr=1e-04,
            loss=focal_loss_fn,
            transform=None,
            labels=CandidPtxLabels.LABELS,
            trial_dir=Path(os.getcwd()),
        )

        imgs = torch.full(size=(2, 3, 1024, 1024), fill_value=50)
        logits = model(imgs)
        loss = model.loss_fn(logits, torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
        print(f"loss = {loss}")
        assert loss.size() == (2, 3)

    def test_focal_loss_reduction_mean(self) -> None:
        focal_loss_fn = FocalWithLogitsLoss(alpha=1.0, reduction="mean")
        model = MultiLabelModel(
            model=ResNet18Model(pretrained=True, num_classes=len(CandidPtxLabels.LABELS)),
            lr=1e-04,
            loss=focal_loss_fn,
            transform=None,
            labels=CandidPtxLabels.LABELS,
            trial_dir=Path(os.getcwd()),
        )

        imgs = torch.full(size=(2, 3, 1024, 1024), fill_value=50)
        logits = model(imgs)
        loss = model.loss_fn(logits, torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
        print(f"loss = {loss}")
        assert loss.size() == (3,)
