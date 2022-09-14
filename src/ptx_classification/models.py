import gc
import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import densenet121, efficientnet_b0, mobilenet_v2, resnet18, resnet50, vgg11

from ptx_classification.data.chexpert.chexpert import CheXpertDataset
from ptx_classification.data.datasets import XrayDataset
from ptx_classification.evaluation.evaluate import evaluate, save_prediction
from ptx_classification.loss import FocalWithLogitsLoss
from ptx_classification.transforms import cast_tensor_to_float32, ten_crop
from ptx_classification.utils import (
    current_process_memory_usage_in_mb,
    get_date_and_time,
    intersection,
)


@dataclass
class BatchOutput:
    y: torch.Tensor
    logits: torch.Tensor
    labels_y_true: list
    dataset_name: list


class MultiLabelModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        transform: Optional[nn.Module],
        labels: List[str],
        lr_scheduler_factor: float = 0.1,
        lr_scheduler_monitor: str = "loss_val/ChestXray14",
        lr_scheduler_patience: int = 2,
        loss: nn.Module = BCEWithLogitsLoss(),
        trial_dir: Optional[Path] = None,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_monitor = lr_scheduler_monitor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.loss_fn = loss
        self.transform = transform
        self.class_labels = sorted(labels)
        self.weight_decay = weight_decay
        self.model_name = f"{self.__class__.__name__}_{model.__class__.__name__}_lr_{self.lr}_{get_date_and_time()}"
        print(f"self.model_name = {self.model_name}")

        self.trial_dir = Path(os.getcwd())
        if trial_dir is not None:
            Path(self.trial_dir / "images").mkdir(parents=True, exist_ok=True)
            Path(self.trial_dir / "predictions").mkdir(parents=True, exist_ok=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: images-tensor of shape (num_imgs, num:_channels, width, height) of integers
               between 0 and 255
        """
        assert torch.max(x) <= 255, f"torch.max(x) = {torch.max(x)}"
        assert torch.min(x) >= 0, f"torch.min(x) = {torch.min(x)}"
        if x.dtype is not torch.float32:
            x = cast_tensor_to_float32(x)
        if self.transform is not None:
            return self.model.forward(self.transform(x))
        else:
            return self.model.forward(x)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        if self.transform is not None:
            logits = self.model.forward(self.transform(images))
        else:
            logits = self.model.forward(images)
        y_pred = torch.sigmoid(input=logits)
        return y_pred

    def predict_for_xray_dataset(
        self, dataset: XrayDataset, save_to_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        predictions = []
        image_ids = []
        columns = [*self.class_labels, "image_id"]
        time = get_date_and_time()

        for img_path in dataset.image_paths:
            img_id = dataset.get_image_id(Path(img_path))
            image_ids.append(img_id)
            image = dataset.load_image(Path(img_path))[None, :].to(device)
            pred = self.predict(images=image).cpu().detach().tolist()[0]

            pred.append(img_id)
            predictions.append(pred)

        df = pd.DataFrame(data=predictions, columns=columns)
        if save_to_dir is not None:
            file_name = dataset.__class__.__name__
            if dataset.__class__.__name__ == CheXpertDataset.__name__:
                file_name = f"{file_name}_FrontLat-{dataset.frontal_lateral}_APPA-{dataset.ap_pa}"
            df.to_csv(save_to_dir / f"prediction_for_{file_name}_{time}.csv", header=True)
        return df

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler_factor is None:
            return optimizer
        else:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                verbose=True,
            )

            lr_schedulers = {
                "scheduler": lr_scheduler,
                "monitor": self.lr_scheduler_monitor,
            }  # "monitor": "AUC/Pneumothorax/ChestXray14/VAL"}
            return [optimizer], [lr_schedulers]

    def training_step(
        self, train_batch: Tuple[torch.Tensor, torch.Tensor, list, list], batch_idx: int
    ) -> dict:
        if len(train_batch) == 4:
            train_batch = [train_batch]

        losses = []
        dataset_names = []

        for i, batch in enumerate(train_batch):
            batch_output = self._batch_step(batch, batch_idx)
            loss = self.loss_fn(batch_output.logits, batch_output.y)
            losses.append(loss)
            y_pred = torch.sigmoid(input=batch_output.logits)
            if i == 0:
                y_trues = batch_output.y.cpu().detach().numpy()
                y_preds = y_pred.cpu().detach().numpy()
            else:
                y_t = batch_output.y.cpu().detach().numpy()
                y_trues = np.append(y_trues, y_t)
                y_preds = np.append(y_preds, y_pred.cpu().detach().numpy())
            dataset_names.append(batch_output.dataset_name[0])

        loss = sum(losses) / len(losses)
        self.log(f"loss", loss, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "y_true": y_trues,
            "y_pred": y_preds,
            "labels": batch_output.labels_y_true,
            "dataset_name": [", ".join(dataset_names)],
        }

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        if type(outputs[-1]) == dict:
            outputs = [outputs]
        for output in outputs:
            _, _, _ = self._on_end_of_epoch(output=output, log_for="TRAIN")

    def validation_step(
        self,
        val_batch: Tuple[torch.Tensor, torch.Tensor, list, list],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        batch_output = self._batch_step(val_batch, batch_idx)

        loss = self.loss_fn(batch_output.logits, batch_output.y)
        self.log(f"loss_val/{batch_output.dataset_name[-1]}", loss, on_step=False, on_epoch=True)
        y_pred = torch.sigmoid(input=batch_output.logits)

        return {
            "y_true": batch_output.y.cpu().detach().numpy(),
            "y_pred": y_pred.cpu().detach().numpy(),
            "labels": batch_output.labels_y_true,
            "dataset_name": batch_output.dataset_name,
        }

    def validation_epoch_end(
        self, outputs: Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]
    ) -> None:
        if type(outputs[-1]) == dict:
            outputs = [outputs]
        for output in outputs:
            y_true, y_pred, labels = self._on_end_of_epoch(output=output, log_for="VAL")

            if self.trial_dir is not None:
                dataset_name = output[-1]["dataset_name"][-1]
                save_prediction(
                    pred=pd.DataFrame(data=y_pred, columns=labels),
                    save_to=self.trial_dir
                    / "predictions"
                    / f"pred_{dataset_name}_val_epoch={self.current_epoch}.csv",
                )
                if self.current_epoch == 0:
                    save_prediction(
                        pred=pd.DataFrame(data=y_true, columns=labels),
                        save_to=self.trial_dir
                        / "predictions"
                        / f"pred_{dataset_name}_val_target.csv",
                    )

    def test_step(
        self,
        test_batch: Tuple[torch.Tensor, torch.Tensor, List[Any], List[Any]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        batch_output = self._batch_step(test_batch, batch_idx)
        y_pred = torch.sigmoid(input=batch_output.logits)

        return {
            "y_true": batch_output.y.cpu().detach().numpy(),
            "y_pred": y_pred.cpu().detach().numpy(),
            "labels": batch_output.labels_y_true,
            "dataset_name": batch_output.dataset_name,
        }

    def test_epoch_end(
        self, outputs: Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]
    ) -> None:
        self.results_test_end = {}

        if type(outputs[-1]) == dict:
            outputs = [outputs]
        for i, output in enumerate(outputs):
            y_true, y_pred, labels = self._on_end_of_epoch(output=output, log_for="TEST")
            if self.trial_dir is not None:
                pred_test = pd.DataFrame(data=y_pred, columns=labels)
                pred_test_target = pd.DataFrame(data=y_true, columns=labels)
                dataset_name = output[-1]["dataset_name"][-1]
                save_prediction(
                    pred=pred_test,
                    save_to=self.trial_dir
                    / "predictions"
                    / f"pred_{dataset_name}_test_epoch={self.current_epoch}.csv",
                )
                save_prediction(
                    pred=pred_test_target,
                    save_to=self.trial_dir / "predictions" / f"pred_{dataset_name}_test_target.csv",
                )
            results = {
                "y_pred_test": y_pred,
                "y_true_test": y_true,
            }
            self.results_test_end[i] = results

    def _batch_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, list, list], batch_idx: int
    ) -> BatchOutput:
        # log memory usage
        if batch_idx % 10 == 0:
            memory_usage = current_process_memory_usage_in_mb()
            self.log("memory_usage", memory_usage)

        # collect garbage
        if batch_idx % 50 == 0:
            gc.collect()

        x, y, labels, dataset_name = batch
        assert x.dim() == 4, f"x.dim() = {x.dim()} != 4, x.size() = {x.size()}"
        logits = self.forward(x)
        # Unfortunately, due to limitations of pytorch lightning the actual output from the datasets
        # __getitem__() get a little bit messed up and the types of items in 'batch' dont match the
        # output types of the __getitem__() function.
        # Therefore the following workarounds to extract the information for the dataset_name and
        # labels represented in the dataset.
        dataset_name = list({dataset_name[0]})
        labels_y_true = sorted(list(set([l[0] for l in labels])))
        labels_y_pred = self.class_labels
        labels_intersection = intersection(labels_y_pred, labels_y_true, sort=True)

        if y.size()[-1] > len(labels_intersection):
            indices = [labels_y_true.index(x) for x in labels_intersection]
            y = y[:, indices]
        if logits.size()[-1] > len(labels_intersection):
            indices = [self.class_labels.index(x) for x in labels_intersection]
            logits = logits[:, indices]

        return BatchOutput(
            y=y, logits=logits, labels_y_true=labels_y_true, dataset_name=dataset_name
        )

    def _on_end_of_epoch(
        self, output: List[Dict[str, Any]], log_for: str
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        y_true = np.concatenate([out["y_true"] for out in output], axis=0)
        y_pred = np.concatenate([out["y_pred"] for out in output], axis=0)

        labels = list(set(itertools.chain(*[out["labels"] for out in output])))
        dataset_name = list(set(itertools.chain(*[out["dataset_name"] for out in output])))

        if y_true.shape[-1] > len(self.class_labels):
            indices = [self.class_labels.index(x) for x in labels]
            y_true = y_true[:, indices]
        assert len(dataset_name) == 1, f"len(dataset_name) = {len(dataset_name)}"

        class2auc = evaluate(
            y_true=y_true,
            y_pred_probas=y_pred,
            labels=intersection(self.class_labels, labels, sort=True),
            dataset_name=dataset_name[-1],
            n_bootstrap=None,
        )
        for class_label, auc_score in class2auc.items():
            self.log(class_label + f"/{log_for}", auc_score)
        return y_true, y_pred, labels


class TenCropMultiLabelModel(MultiLabelModel):
    def __init__(
        self,
        crop_size: int,
        model: nn.Module,
        lr: float,
        transform: Optional[nn.Module],
        labels: List[str],
        loss: nn.Module = BCEWithLogitsLoss(),
        lr_scheduler_factor: float = 0.1,
        lr_scheduler_monitor: str = "loss_val/ChestXray14",
        lr_scheduler_patience: int = 2,
        trial_dir: Optional[Path] = None,
        weight_decay: float = 0.0,
    ):
        super(TenCropMultiLabelModel, self).__init__(
            model=model,
            lr=lr,
            transform=transform,
            labels=labels,
            loss=loss,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_monitor=lr_scheduler_monitor,
            lr_scheduler_patience=lr_scheduler_patience,
            trial_dir=trial_dir,
            weight_decay=weight_decay,
        )
        self.ten_crop_transform = ten_crop(crop_size=crop_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: images-tensor of shape (num_imgs, num:_channels, width, height) of integers
               between 0 and 255
        """
        assert torch.max(x) <= 255, f"torch.max(x) = {torch.max(x)}"
        assert torch.min(x) >= 0, f"torch.min(x) = {torch.min(x)}"
        if x.dtype is not torch.float32:
            x = cast_tensor_to_float32(x)

        logits = []
        for img in x:
            ten_cropped_img = self.ten_crop_transform(img)
            if self.transform is not None:
                ten_cropped_logits = self.model.forward(self.transform(ten_cropped_img))
                img_logit = self.model.forward(self.transform(img[None, :]))
            else:
                ten_cropped_logits = self.model.forward(ten_cropped_img)
                img_logit = self.model.forward(img[None, :])

            assert ten_cropped_logits.size() == (
                10,
                len(self.class_labels),
            ), f"ten_cropped_logits.size() = {ten_cropped_logits.size()}"

            logits.append(torch.mean(torch.cat((ten_cropped_logits, img_logit)), dim=0))

        logits = torch.stack(logits)
        return logits


class VGG11Model(nn.Module):
    def __init__(
        self,
        pretrained: bool,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model = vgg11(pretrained=pretrained)
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)


class MobileNetV2Model(nn.Module):
    def __init__(
        self,
        pretrained: bool,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model = mobilenet_v2(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)


class EfficientNetB0Model(nn.Module):
    def __init__(
        self,
        pretrained: bool,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model = efficientnet_b0(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)


class ResNet18Model(nn.Module):
    def __init__(
        self,
        pretrained: bool,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model = resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(512, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)


class ResNet50Model(nn.Module):
    def __init__(
        self,
        pretrained: bool,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.model = resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(2048, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)


class DenseNet121Model(nn.Module):
    def __init__(self, pretrained: bool, num_classes: int) -> None:
        super().__init__()
        self.model = densenet121(pretrained=pretrained)
        self.model.classifier = nn.Linear(1024, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)


def get_lightning_module(
    model_class: str,
    loss: Literal["BCEWithLogitsLoss", "FocalWithLogitsLoss"],
    lr: float,
    labels: List[str],
    pretrained: bool,
    trial_dir: Path,
    lr_scheduler_monitor: str,
    lr_scheduler_factor: float = 0.1,
    ten_crop_size: int = None,
    transform: Optional[torch.nn.Module] = None,
    weight_decay: float = 0.0,
) -> pl.LightningModule:
    num_classes = len(labels)

    sub_model: torch.nn.Module

    if model_class == VGG11Model.__name__:
        sub_model = VGG11Model(pretrained=pretrained, num_classes=num_classes)
    elif model_class == MobileNetV2Model.__name__:
        sub_model = MobileNetV2Model(pretrained=pretrained, num_classes=num_classes)
    elif model_class == EfficientNetB0Model.__name__:
        sub_model = EfficientNetB0Model(pretrained=pretrained, num_classes=num_classes)
    elif model_class == ResNet18Model.__name__:
        sub_model = ResNet18Model(pretrained=pretrained, num_classes=num_classes)
    elif model_class == ResNet50Model.__name__:
        sub_model = ResNet50Model(pretrained=pretrained, num_classes=num_classes)
    elif model_class == DenseNet121Model.__name__:
        sub_model = DenseNet121Model(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"The given model_class does not exist: {model_class}")
    loss_fn: nn.Module
    if loss == BCEWithLogitsLoss.__name__:
        loss_fn = BCEWithLogitsLoss()
    elif loss == FocalWithLogitsLoss.__name__:
        loss_fn = FocalWithLogitsLoss(reduction="mean")
    else:
        raise ValueError(f"The given loss does not exist: {loss}")

    model: MultiLabelModel
    if ten_crop_size is None:
        model = MultiLabelModel(
            model=sub_model,
            lr=lr if lr != "auto_lr_find" else 0.1,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_monitor=lr_scheduler_monitor,
            loss=loss_fn,
            labels=labels,
            transform=transform,
            trial_dir=trial_dir,
            weight_decay=weight_decay,
        )
    else:
        model = TenCropMultiLabelModel(
            model=sub_model,
            lr=lr if lr != "auto_lr_find" else 0.1,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_monitor=lr_scheduler_monitor,
            loss=loss_fn,
            labels=labels,
            crop_size=ten_crop_size,
            transform=transform,
            trial_dir=trial_dir,
            weight_decay=weight_decay,
        )

    return model


def get_model_name(model: MultiLabelModel) -> str:
    class_labels = ", ".join(model.class_labels)
    name = f"{model.model.__class__.__name__}\ntrained with {class_labels}"
    return name


def load_multi_label_model_from_checkpoint(model_path: Path) -> MultiLabelModel:
    """
    Loads a MultiLabelModel from a given checkpoint file.
    Args:
        model_path: Path to model checkpoint file: .ckpt

    Returns:
        MultiLabelModel from given path.
    """
    model = MultiLabelModel.load_from_checkpoint(checkpoint_path=str(model_path))
    return model


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(logits, labels)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _, predicted = torch.max(logits.data, dim=1)
    correct = (predicted == labels).sum().item()
    acc = torch.tensor(correct / len(labels))
    return acc
