import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ptx_classification.data.candid_ptx.candid_ptx import CandidPtxDataset, CandidPtxLabels
from ptx_classification.evaluation.evaluate import (
    apply_cut_off_to_np_ndarray,
    apply_cut_off_to_tensor,
    evaluate,
    get_all_label_combinations,
    plot_confusion_matrix,
    plot_roc_curves_for_ct,
    read_in_prediction,
    save_prediction,
)


def test_get_all_label_combinations() -> None:
    label_combinations = get_all_label_combinations(CandidPtxLabels.LABELS)
    assert len(label_combinations) == 8


def test_apply_cut_off_to_tensor() -> None:
    cut_off = 0.5
    probabilities = torch.tensor(
        [
            [0.4, 0.2, 0.7, 0.4],
            [0.9, 0.2, 0.5, 0.6],
        ]
    )
    binary = apply_cut_off_to_tensor(probabilities, cut_off)
    assert torch.all(binary.eq(torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]])))


def test_apply_cut_off_to_np_ndarray() -> None:
    y_proba = np.array(
        [
            [0.0, 0.23, 0.4, 0.9, 0.5, 0.4, 0.3333, 0.7, 0.1],
            [0.4, 0.32, 0.8, 0.9, 0.5, 0.4, 0.3567, 0.6, 0.9],
        ]
    )
    y_cut05 = apply_cut_off_to_np_ndarray(y_proba=y_proba, cut_off=0.5)

    np.testing.assert_allclose(
        y_cut05,
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            ]
        ),
    )


def test_read_in_and_save_prediction() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 7, 4]])
    columns = ["Chest Tube", "Pneumothorax", "Rib Fracture"]
    pred_1 = pd.DataFrame(data=data, columns=columns)
    tmp_file = Path(os.getcwd()) / "tmp.csv"
    save_prediction(pred=pred_1, save_to=tmp_file)
    pred_2 = read_in_prediction(file=tmp_file)

    print(f"\npred_1 = {pred_1}")
    print(f"pred_2 = {pred_2}")
    print(f"pred_1.columns = {pred_1.columns}")
    print(f"pred_2.columns = {pred_2.columns}")

    assert (pred_1.columns == pred_2.columns).all()
    np.testing.assert_array_equal(pred_1.values, pred_2.values)
    os.remove(tmp_file)


def test_evaluate_without_bootstrap() -> None:
    y_true = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    y_pred = np.array([[0.8, 0.5, 0.0], [0.2, 0.9, 0.0], [0.8, 0.1, 1.0], [0.5, 0.7, 0.0]])
    evaluation = evaluate(y_true=y_true, y_pred_probas=y_pred, labels=CandidPtxLabels.LABELS)
    print(f"evaluation = {evaluation}")


def test_evaluate_with_bootstrap() -> None:
    y_true = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    y_pred = np.array(
        [
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
            [0.8, 0.5, 0.0],
            [0.2, 0.9, 0.0],
            [0.8, 0.1, 1.0],
            [0.5, 0.7, 0.0],
        ]
    )
    evaluation = evaluate(
        y_true=y_true, y_pred_probas=y_pred, labels=CandidPtxLabels.LABELS, n_bootstrap=100
    )
    print(f"evaluation = {evaluation}")


def test_plot_roc_curves_for_ct() -> None:
    target_data = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    pred_data = np.array(
        [
            [0.9, 0.3, 0.1],
            [0.4, 0.4, 0.1],
            [1.0, 0.2, 0.0],
            [0.0, 0.7, 0.0],
            [0.6, 0.9, 0.0],
            [0.1, 0.7, 0.0],
            [0.0, 0.4, 0.0],
            [0.4, 0.8, 0.0],
            [0.2, 0.4, 0.0],
            [0.0, 0.9, 0.0],
            [0.3, 0.2, 0.0],
            [0.0, 0.7, 0.0],
            [0.0, 0.0, 0.0],
            [0.6, 0.7, 0.0],
            [1.0, 0.6, 0.0],
            [1.0, 1.0, 0.0],
            [0.9, 0.4, 0.0],
            [0.7, 1.0, 0.0],
            [1.0, 0.9, 0.0],
            [0.4, 0.8, 0.0],
        ]
    )
    target = pd.DataFrame(data=target_data, columns=CandidPtxDataset.LABELS)
    pred = pd.DataFrame(data=pred_data, columns=CandidPtxDataset.LABELS)
    plot_roc_curves_for_ct([target, target], [pred, pred], model_names=["Model 1", "Model 2"])


def test_plot_confusion_matrix() -> None:
    y_true = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    y_pred = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    plot_confusion_matrix(
        y_target=y_true,
        y_pred_proba=y_pred,
        cut_off=0.5,
        labels=["a", "b", "c"],
        # save_to_dir=Path(os.getcwd())
    )
