import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import torch
from numpy import allclose
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from ptx_classification.data.candid_ptx.candid_ptx import CandidPtxDataset, CandidPtxLabels
from ptx_classification.utils import get_date_and_time

seaborn.set_style("white")


def save_prediction(pred: pd.DataFrame, save_to: Path) -> None:
    pred.to_csv(save_to)


def read_in_prediction(file: Path) -> pd.DataFrame:
    pred = pd.read_csv(file, index_col=0)
    return pred


class CandidPtxMetrics:
    CANDID_PTX_LOSS_TRAIN = "loss"
    CANDID_PTX_LOSS_VAL = "loss_val/CandidPtx"

    CANDID_PTX_AUC_TRAIN_CT = "AUC/Chest Tube/CandidPtx/TRAIN"
    CANDID_PTX_AUC_TRAIN_PTX = "AUC/Pneumothorax/CandidPtx/TRAIN"
    CANDID_PTX_AUC_TRAIN_RF = "AUC/Rib Fracture/CandidPtx/TRAIN"
    CANDID_PTX_AUC_VAL_CT = "AUC/Chest Tube/CandidPtx/VAL"
    CANDID_PTX_AUC_VAL_PTX = "AUC/Pneumothorax/CandidPtx/VAL"
    CANDID_PTX_AUC_VAL_RF = "AUC/Rib Fracture/CandidPtx/VAL"


class PadChestMetrics:
    PADCHEST_LOSS_VAL = "loss_val/PadChest"
    PADCHEST_AUC_VAL_CT = "AUC/Chest Tube/PadChest/VAL"


@dataclass
class ValueWithError:
    value: float
    std_error: float

    def confidence_interval_95_pct(self) -> Tuple[float, float]:
        interval = 1.96 * self.std_error
        return self.value - (interval / 2), self.value + (interval / 2)

    def confidence_interval_95_pct_deviation(self) -> float:
        interval = 1.96 * self.std_error
        return interval / 2


def evaluate(
    y_true: np.ndarray,
    y_pred_probas: np.ndarray,
    labels: list,
    dataset_name: str = "",
    n_bootstrap: int = None,
) -> Dict[str, float]:
    assert (
        y_true.shape == y_pred_probas.shape
    ), f"y_true.shape = {y_true.shape}, y_pred_probas.shape = {y_pred_probas.shape}"
    assert y_true.shape[1] == len(labels), f"y_true.shape = {y_true.shape}"
    scores = {}
    for i, label in enumerate(labels):
        y_true_col = y_true[:, i]
        y_pred_col = y_pred_probas[:, i]
        auc_with_error = compute_auc_with_std_error(y_true_col, y_pred_col, n_bootstrap)
        scores[f"AUC/{label}/{dataset_name}"] = auc_with_error.value
        if n_bootstrap is not None:
            scores[
                f"AUC with error/{label}/{dataset_name} "
            ] = f"{auc_with_error.value} ± {auc_with_error.confidence_interval_95_pct_deviation()}"
    print(f"scores = {scores}")
    return scores


def compute_auc_with_std_error(
    y_true: np.ndarray, y_pred_probas: np.ndarray, n_bootstrap: int
) -> ValueWithError:
    auc = auc_score(y_true, y_pred_probas)
    auc_std_err = None
    if n_bootstrap is not None:
        auc_std_err = auc_std_error(y_true, y_pred_probas, n_bootstrap)
    return ValueWithError(value=auc, std_error=auc_std_err)


def auc_score(y_true: np.ndarray, y_pred_probas: np.ndarray) -> float:
    assert len(y_true.shape) == 1, f"y_true.shape = {y_true.shape}"
    assert len(y_pred_probas.shape) == 1, f"y_pred_probas.shape = {y_pred_probas.shape}"
    try:
        auc = roc_auc_score(y_true=y_true, y_score=y_pred_probas)
    except ValueError as e:
        print(e)
        auc = -1.0
    return auc


def auc_std_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000, random_seed: int = 42
) -> float:
    """
    compute the standard error for the metrics currently in the evaluation dict using 1000 bootstrap intervals
    in each bootstrap interval, choose samples with replacement and compute the given metric
    Bootstrapping takes a lot of time, so I would not recommend to compute the standard error in every epoch and
    only calling the function at the end of the training
    - on the validation set with 219 sequences and only the accuracy metric it takes
    around 3 seconds for 10 bootstrap iterations -> approx. 300 seconds for all 1000 iterations

    Args:
        bootstrap_n: number of bootstrap intervals
        random_seed: set the random seed for reproducible results

    Returns:
        creates a new dict with the 95% confidence intervals for the available metrics
    """
    bootstrap_aucs = []
    indices = np.array([i for i in range(len(y_true))])
    for bootstrap in range(n_bootstrap):
        bootstrap_indices = np.random.choice(indices, size=len(y_true))
        y_true_sample = y_true[bootstrap_indices]
        y_pred_sample = y_pred[bootstrap_indices]
        bootstrap_aucs.append(auc_score(y_true=y_true_sample, y_pred_probas=y_pred_sample))

    # auc_mean = torch.mean(torch.tensor(bootstrap_aucs))
    auc_std_error = np.std(bootstrap_aucs)
    return auc_std_error


def youden_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Find data-driven cut-off for classification

    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).

    References
    ----------

    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.

    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.

    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]


def plot_roc_curve(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    model_names: List[str] = None,
    save_to_dir: Optional[Path] = None,
    show: bool = True,
    plot_title: str = "ROC curves",
    sub_plot_title: str = "",
    legend: bool = True,
    legend_title: str = "Datasets used for evaluation\n",
    legend_loc: str = "lower right",
    legend_bbox_to_anchor: Tuple[float, float] = None,
    plot_background_color: str = "white",
    figure: matplotlib.figure.Figure = None,
    figure_position: Tuple[int, int, int] = None,
    n_bootstrap: int = None,
    cohorts: bool = False,
) -> Optional[matplotlib.figure.Figure]:
    assert len(targets) == len(
        predictions
    ), f"len(targets) = {len(targets)}, len(predictions) = {len(predictions)}"

    n_preds = len(targets)
    if figure is not None:
        ax = figure.add_subplot(
            figure_position[0],
            figure_position[1],
            figure_position[2],
            facecolor=plot_background_color,
        )
        ax.set_facecolor(plot_background_color)
    else:
        plt.figure(figsize=(5, 8 + (0.3 * n_preds)), dpi=500)

    colors = ["darkorange", "blue", "green", "purple", "red"]
    linestyles = ["solid", "solid", "solid", "solid", "solid"]
    if cohorts:
        colors = ["darkorange", "darkorange", "blue", "blue", "green", "green"]
        linestyles = ["solid", "dotted", "solid", "dotted", "solid", "dotted"]

    for i, pred in enumerate(predictions):
        target = targets[i]
        pred = pred
        fpr, tpr, _ = metrics.roc_curve(target, pred)

        auc_with_error: ValueWithError = compute_auc_with_std_error(target, pred, n_bootstrap)
        value = f"{auc_with_error.value:0.2f}"
        if n_bootstrap is not None:
            value = f"{value} ± {auc_with_error.confidence_interval_95_pct_deviation():0.2f}"

        label = None
        if legend:
            if model_names is not None:
                label = f"{model_names[i]}, AUC={value}"
            else:
                label = f"AUC={value}"
        plt.plot(
            fpr,
            tpr,
            label=label,
            color=colors[i],
            linestyle=linestyles[i],
        )

    # add random model
    plt.plot(
        [0, 1],
        [0, 1],
        label="AUC=0.5" if model_names is None else "Random baseline, AUC=0.5",
        color="grey",
        linestyle="dashed",
        linewidth=1,
    )
    if legend:
        plt.legend(
            loc=legend_loc,
            title=legend_title,
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=8 if cohorts else 12,
            # bbox_to_anchor=(0.5, -0.3 - (0.05 * n_preds)),
        )

    plt.title(plot_title + "\n")
    if sub_plot_title is not None:
        props = dict(
            boxstyle="round,pad=0.3,rounding_size=0.1",
            facecolor="white",
            edgecolor="black",
            alpha=0.7,
            linestyle="-",
            linewidth=1.0,
        )

        # place a text box in upper left in axes coords
        plt.text(
            0.03,
            0.97,
            sub_plot_title,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )

    plt.xlabel("FPR (1 - Specificity)", fontsize=13)
    plt.ylabel("TPR (Sensitivity)", fontsize=13)
    plt.gca().set_aspect("equal")
    plt.grid()

    if save_to_dir is not None:
        save_to = f"{save_to_dir}/roc_curve/roc_curve_chest_tube_{get_date_and_time()}.png"
        print(f"Saving roc-curve-plot to: {save_to}")
        plt.savefig(save_to)
    if show:
        plt.show()

    if figure is not None:
        return figure


def plot_roc_curves_for_ct(
    y_targets: List[pd.DataFrame],
    predictions: List[pd.DataFrame],
    model_names: List[str],
    save_to_dir: Optional[Path] = None,
    show: bool = True,
    title_plot: str = "ROC curves",
) -> None:
    ct_preds = [pred["Chest Tube"] for pred in predictions]
    ct_targets = [target["Chest Tube"] for target in y_targets]
    plot_roc_curve(
        targets=ct_targets,
        predictions=ct_preds,
        model_names=model_names,
        save_to_dir=save_to_dir,
        show=show,
        plot_title=title_plot,
        legend_loc="lower center",
        legend_bbox_to_anchor=(0.5, -0.3 - (0.05 * len(ct_targets))),
    )


@dataclass
class LabelCombination:
    name: str
    labels: List[str]

    def label_indicator_matrix_row(self) -> List[int]:
        return [1 if label in self.labels else 0 for label in CandidPtxDataset.LABELS]


def get_all_label_combinations(labels: List[str]) -> List[LabelCombination]:
    label_combinations = [LabelCombination(name="none", labels=[])]
    for i in range(1, len(labels) + 1):
        for subset in itertools.combinations(labels, i):
            lc = LabelCombination(name=", \n".join(list(subset)), labels=list(subset))
            label_combinations.append(lc)
    return label_combinations


LABEL_COMBINATIONS = get_all_label_combinations(labels=CandidPtxLabels.LABELS)


def plot_confusion_matrix(
    y_target: np.ndarray,
    y_pred_proba: np.ndarray,
    cut_off: float,
    labels: list,
    save_to_dir: Optional[Path] = None,
    show: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    assert 0.0 <= cut_off <= 1.0, f"cut_off = {cut_off}"

    y_pred = apply_cut_off_to_np_ndarray(y_proba=y_pred_proba, cut_off=cut_off)
    y_pred_cm = np.argmax(multilabel_to_multiclass(y_pred), axis=1)
    y_target_cm = np.argmax(multilabel_to_multiclass(y_target), axis=1)

    # Generate the confusion matrix
    cf_matrix = confusion_matrix(y_target_cm, y_pred_cm)

    label_combinations = [lc.name for lc in get_all_label_combinations(labels)]

    plt.figure(figsize=(13, 9))
    heatmap = seaborn.heatmap(
        cf_matrix,
        linewidths=0.5,
        annot=True,
        cmap="Reds",
        xticklabels=label_combinations,
        yticklabels=label_combinations,
        fmt="g",
    )

    heatmap.set_title(title + "\n", fontsize=22)
    heatmap.set_xlabel("Predicted Values\n", fontsize=18)
    heatmap.set_ylabel("Actual Values", fontsize=18)

    if save_to_dir is not None:
        save_to = f"{save_to_dir}/confusion_matrix/confusion_matrix_{get_date_and_time()}.png"
        print(f"Saving confusion matrix to: {save_to}")
        figure = heatmap.get_figure()
        figure.savefig(save_to)
    if show:
        plt.show()


def apply_cut_off_to_tensor(probabilities: torch.Tensor, cut_off: float):
    return (probabilities > cut_off).float()


def apply_cut_off_to_np_ndarray(y_proba: np.ndarray, cut_off: float) -> np.ndarray:
    y_cut = np.copy(y_proba)
    y_cut[y_cut < cut_off] = 0.0
    y_cut[y_cut >= cut_off] = 1.0

    return y_cut


def multilabel_to_multiclass(label_indicator_matrix: np.ndarray) -> np.ndarray:
    y_multi_class_one_hot = []
    for row in label_indicator_matrix:
        y_multi_class_one_hot.append(
            [
                1 if allclose(row, np.asarray(combination.label_indicator_matrix_row())) else 0
                for combination in LABEL_COMBINATIONS
            ]
        )

    y_multi_class_one_hot = np.array(y_multi_class_one_hot)
    assert allclose(y_multi_class_one_hot.sum(axis=1), 1.0), y_multi_class_one_hot.sum(axis=1)
    return y_multi_class_one_hot
