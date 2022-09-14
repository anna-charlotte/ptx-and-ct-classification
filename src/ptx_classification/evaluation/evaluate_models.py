import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from ptx_classification.data.candid_ptx.candid_ptx import (
    CandidPtxDataModule,
    CandidPtxDataset,
    CandidPtxLabels,
    CandidPtxSplits,
)
from ptx_classification.evaluation.evaluate import (
    apply_cut_off_to_np_ndarray,
    evaluate,
    plot_confusion_matrix,
    plot_roc_curves_for_ct,
)
from ptx_classification.models import MultiLabelModel
from ptx_classification.utils import REPO_ROOT_DIR, get_cache_dir, get_data_dir


def save_y_target_test(save_to: Path) -> None:
    print("In save_y_true")
    data_dir = get_data_dir()
    cache_dir = get_cache_dir()
    print(f"data_dir = {data_dir}")
    print(f"cache_dir = {cache_dir}")

    n_train = CandidPtxSplits.N_TRAIN
    n_val = CandidPtxSplits.N_VAL
    n_test = CandidPtxSplits.N_TEST
    print(f"[n_train, n_val, n_test] = {[n_train, n_val, n_test]}")
    print(f"CandidPtxLabels.LABELS = {CandidPtxLabels.LABELS}")
    labels = CandidPtxLabels.LABELS
    datamodule = CandidPtxDataModule(
        dataset=CandidPtxDataset(
            root=data_dir, cache_dir=cache_dir, labels_to_use=CandidPtxLabels.LABELS
        ),
        batch_size=n_test,
        train_val_test_split=(n_train, n_val, n_test),
    )
    test_dl = datamodule.test_dataloader()
    ys = []
    for i in range(len(test_dl.dataset.indices)):
        x, y = test_dl.dataset[i]
        ys.append(y.tolist())
    print(f"len(ys) = {len(ys)}")
    df = pd.DataFrame(data=np.array(ys), columns=labels)
    df.to_csv(path_or_buf=save_to)


def eval_models(directory: Path, device: str) -> None:
    print(f"device = {device}")
    data_dir = get_data_dir()
    cache_dir = get_cache_dir()
    print(f"data_dir = {data_dir}")
    print(f"cache_dir = {cache_dir}")

    n_train = CandidPtxSplits.N_TRAIN
    n_val = CandidPtxSplits.N_VAL
    n_test = CandidPtxSplits.N_TEST
    print(f"[n_train, n_val, n_test] = {[n_train, n_val, n_test]}")
    print(f"CandidPtxLabels.LABELS = {CandidPtxLabels.LABELS}")
    # labels = CandidPtxLabels.LABELS
    datamodule = CandidPtxDataModule(
        dataset=CandidPtxDataset(
            root=data_dir, cache_dir=cache_dir, labels_to_use=CandidPtxLabels.LABELS
        ),
        batch_size=1,
        train_val_test_split=(n_train, n_val, n_test),
    )

    for path in sorted(directory.glob("*")):
        if "checkpoint_best_model" in str(path):
            print(f"path = {path}")
            params = json.load(open(path.parent / "params.json"))
            lr = params["model"]["lr"]
            transform = params["model"]["transform"]
            labels = params["labels_to_use"]

            model = MultiLabelModel.load_from_checkpoint(str(path))
            labels = model.class_labels
            labels_in_name = ""
            if CandidPtxLabels.CT in labels:
                labels_in_name += "CT"
            if CandidPtxLabels.PTX in labels:
                labels_in_name += ", PTX"
            if CandidPtxLabels.RF in labels:
                labels_in_name += ", RF"

            model_name = f"{model.__class__.__name__}, lr={model.lr}, labels={labels_in_name}"
            print(f"model_name = {model_name}")

            trainer = pl.Trainer()
            trainer.test(model, dataloaders=datamodule.test_dataloader())


def eval_and_plot() -> None:
    target = pd.read_csv(
        REPO_ROOT_DIR / "ray_results" / "pred_test_2" / "pred_test_target_2022-05-16_13-36-39.csv",
        index_col=0,
    )
    pred = pd.read_csv(
        REPO_ROOT_DIR / "ray_results" / "pred_test_2" / "pred_test_2022-05-16_13-36-39.csv",
        index_col=0,
    )
    all_pred = []
    model_names = [
        "ResNet18, lr=0.0001, labels=[CT, PTX, RF]",
        # "ResNet18, lr=0.0001, labels=[CT, PTX]",
        # "ResNet18, lr=0.0001, labels=[CT]",
    ]

    for i in range(1):
        all_pred.append(pred)
        labels = pred.columns
        y_target = target[labels]
        for row in y_target.values:
            if np.all((row == np.array([1.0, 1.0, 1.0]))):
                print(f"row = {row}")
        print(f"target.columns = {target.columns}")
        print(f"pred.columns = {pred.columns}")
        scores = evaluate(
            y_true=y_target.values, y_pred_probas=pred.values, labels=labels, dataset_name="test"
        )

        for cut in range(1, 2):
            cut_off = cut / 20
            print(f"cut = {cut}")
            y_pred_co = apply_cut_off_to_np_ndarray(pred.values, cut_off=cut_off)
            plot_confusion_matrix(
                y_target=y_target,
                y_pred_proba=y_pred_co,
                cut_off=cut_off,
                labels=labels,
            )


plot_roc_curves_for_ct(y_targets=target, predictions=all_pred, model_names=model_names)
