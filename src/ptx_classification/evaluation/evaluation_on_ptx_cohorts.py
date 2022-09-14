from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.figure
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ptx_classification.data.candid_ptx.candid_ptx import (
    CandidPtxDataModule,
    CandidPtxDataset,
    CandidPtxSplits,
)
from ptx_classification.data.datasets import DatasetSubset
from ptx_classification.data.ptx_cohorts.ptx_cohorts import PtxCohortsCiipDataset
from ptx_classification.evaluation.evaluate import plot_roc_curve
from ptx_classification.models import MultiLabelModel, get_model_name
from ptx_classification.utils import get_date_and_time


@dataclass
class DataPoint:
    ptx_groundtruth: bool
    ct_groundtruth: bool
    ptx_size: str
    predicted_ptx_probabilities: Dict[str, float]  # model name to probability

    def __post_init__(self) -> None:
        for proba in self.predicted_ptx_probabilities.values():
            assert 0.0 <= proba <= 1.0, f"proba = {proba}"


class Filter:
    def __init__(
        self,
        ptx: Optional[bool] = None,
        ct: Optional[bool] = None,
        ptx_size: Optional[str] = None,
    ):
        def use(dp: DataPoint) -> bool:
            return (
                (ptx is None or ptx == dp.ptx_groundtruth)
                and (ct is None or ct == dp.ct_groundtruth)
                and (ptx_size is None or ptx_size == dp.ptx_size)
            )

        self.use = use

        ptx_split = "all"
        if ptx == False:
            ptx_split = "neg."
        elif ptx == True:
            ptx_split = "pos."

        ct_split = "w/wo"
        if ct == False:
            ct_split = "wo"
        elif ct == True:
            ct_split = "w"

        self.name = f"PTX-{ptx_split}: {ct_split} CT"

    def filter(self, data_points: List[DataPoint]) -> List[DataPoint]:
        return [dp for dp in data_points if self.use(dp)]


# filters down
PTX_NEGATIVE_FILTERS = [
    Filter(ptx=False, ct=True),
    Filter(ptx=False, ct=None),
    Filter(ptx=False, ct=False),
]

# filters right
PTX_POSITIVE_FILTERS = [
    Filter(ptx=True, ct=False),
    Filter(ptx=True, ct=None),
    Filter(ptx=True, ct=True),
]


def perform_analyses(
    data: List[DataPoint],
    filters_down: List[Filter],
    filters_right: List[Filter],
    show: bool = False,
    save_to: Path = None,
    candid_ptx: bool = False,
) -> None:
    width = len(filters_right) * 5
    height = len(filters_down) * 7
    fig = plt.figure(figsize=(width, height))

    outergs = gridspec.GridSpec(1, 1)
    outergs.update(bottom=0.27, left=0.07, right=0.95, top=0.9)
    outerax = fig.add_subplot(outergs[0])
    outerax.tick_params(axis="both", which="both", bottom=0, left=0, labelbottom=0, labelleft=0)

    pos = 1

    for row, filter_d in enumerate(filters_down):
        for col, filter_r in enumerate(filters_right):
            print(f"pos = {pos}")
            filtered_dataset: List[DataPoint] = filter_d.filter(data) + filter_r.filter(data)
            figure_position = (len(filters_down), len(filters_right), pos)
            print(f"figure_position = {figure_position}")
            background_color = "#FFFFFF"
            if pos == 1:
                background_color = "#FCE0E0"
            if pos == 5:
                background_color = "#F0F0F0"
            elif pos == 9:
                background_color = "#E0EFE0"

            add_roc_curve_subplot(
                filtered_dataset,
                figure=fig,
                figure_position=figure_position,
                background_color=background_color,
                plot_title=f"Dataset Subset:\n{filter_r.name}\n{filter_d.name}",
                legend=True,
            )
            pos += 1

    # plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.92, bottom=0.3, top=0.88, wspace=0.3, hspace=0.3)
    model_names = get_model_names_from_datapoints(data)
    plt.figlegend(
        model_names + ["Random Baseline"],
        loc="upper left",
        title="Model Legend\n",
        title_fontsize=20,
        fontsize=18,
        edgecolor="grey",
        borderaxespad=2.0,
        borderpad=1.0,
        bbox_to_anchor=(0.09, 0.23),
        frameon=True,
        ncol=1,
        # labelspacing=0.
    )
    # add title
    props = dict(fontsize=22, horizontalalignment="center", verticalalignment="center")
    title = (
        "ROC Curves for Pneumothorax Classification\non CANDID-PTX Dataset Cohorts"
        if candid_ptx
        else "ROC Curves for Pneumothorax Classification\non the LMU Dataset Cohorts"
    )
    plt.text(
        -0.94,
        4.3,
        title,
        fontweight="semibold",
        **props,
    )

    # add labels to x axis
    plt.text(-0.94, -0.55, "Subset of PTX-negative images", **props)
    plt.text(-2.44, -0.35, "with Chest Tubes", **props)
    plt.text(-0.94, -0.35, "with + without Chest Tubes", **props)
    plt.text(00.56, -0.35, "without Chest Tubes", **props)

    # add labels to y axis
    plt.text(-3.40, 1.9, "Subset of PTX-positive images", rotation="vertical", **props)
    plt.text(-3.25, 0.4, "with Chest Tubes", rotation="vertical", **props)
    plt.text(-3.25, 1.9, "with + without Chest Tubes", rotation="vertical", **props)
    plt.text(-3.25, 3.4, "without Chest Tubes", rotation="vertical", **props)

    if show:
        plt.show()
    if save_to is not None:
        file = save_to / f"ptx_cohorts_eval/ptx_cohorts_eval_{get_date_and_time()}.png"
        print(f"Saving plot to: {file}")
        plt.savefig(file)


def add_roc_curve_subplot(
    dps: List[DataPoint],
    figure: matplotlib.figure.Figure,
    figure_position: Tuple[int, int, int],
    background_color: str = "white",
    plot_title: str = "",
    legend: bool = True,
) -> None:
    model_name2pred = {}
    for dp in dps:
        for model_name, proba in dp.predicted_ptx_probabilities.items():
            if model_name not in model_name2pred:
                model_name2pred[model_name] = [proba]
            else:
                model_name2pred[model_name].append(proba)

    # model_names: List[str] = list(model_name2pred.keys())
    predictions: List[np.ndarray] = [np.array(pred) for pred in model_name2pred.values()]
    ground_truth: np.ndarray = np.array([dp.ptx_groundtruth for dp in dps]).astype(float)
    n_pred = len(predictions)

    plot_roc_curve(
        predictions=predictions,
        targets=[ground_truth for _ in range(n_pred)],
        # model_names=model_names,
        show=False,
        n_bootstrap=100,
        plot_title="",
        sub_plot_title=plot_title,
        plot_background_color=background_color,
        figure=figure,
        figure_position=figure_position,
        legend=legend,
        legend_title="",
        cohorts=True,
    )


def get_model_names_from_datapoints(dps: List[DataPoint]) -> List[str]:
    model_names = [str(name) for name, _ in dps[-1].predicted_ptx_probabilities.items()]
    return model_names


def list_of_datapoints(
    predictions: List[pd.DataFrame],
    ground_truth: pd.DataFrame,
    model_names: List[str],
    sort_by: str = "image_id",
) -> List[DataPoint]:
    ground_truth_sorted = ground_truth.sort_values(by=[sort_by])
    predictions_sorted = [p.sort_values(by=[sort_by]) for p in predictions]
    for p in predictions_sorted:
        assert (
            ground_truth_sorted[sort_by].tolist() == p[sort_by].tolist()
        ), f"ground_truth_sorted[sort_by].tolist() = {ground_truth_sorted[sort_by].tolist()}, p[sort_by].tolist() = {p[sort_by].tolist()}"

    n_datapoints = len(ground_truth_sorted)
    n_pred = len(predictions_sorted)
    datapoints = []
    for i in range(n_datapoints):
        datapoints.append(
            DataPoint(
                ptx_groundtruth=ground_truth_sorted.iloc[i]["Pneumothorax"],
                ct_groundtruth=ground_truth_sorted.iloc[i]["Chest Tube"],
                ptx_size="nan"
                if "Pneumothorax size" not in ground_truth_sorted.columns
                else ground_truth_sorted.iloc[i]["Pneumothorax size"],
                predicted_ptx_probabilities={
                    model_names[j]: predictions_sorted[j].iloc[i]["Pneumothorax"]
                    for j in range(n_pred)
                },
            )
        )

    return datapoints


def perform_ptx_cohorts_evaluation(
    models: List[MultiLabelModel],
    model_names: List[str],
    resizings: List[Tuple[int, int]],
    data_dir: Path,
    save_to: Path,
    show_plot: bool,
    candid_ptx: bool = False,
) -> None:
    assert len(models) == len(model_names)

    predictions: List[pd.DataFrame] = []
    for i, model in enumerate(models):
        if candid_ptx:
            dataset = CandidPtxDataset(
                labels_to_use=["Chest Tube", "Pneumothorax"], root=data_dir, resize=resizings[i]
            )
            datamodule = CandidPtxDataModule(
                dataset=dataset, batch_size=1, train_val_test_split=CandidPtxSplits.SPLITS
            )
            test_set: DatasetSubset = datamodule.test_dataloader().dataset
            dataset.image_paths = [dataset.image_paths[index] for index in test_set.indices]
        else:
            dataset = PtxCohortsCiipDataset(root=data_dir, resize=resizings[i])

        prediction = model.predict_for_xray_dataset(dataset, save_to_dir=model.trial_dir)
        predictions.append(prediction)

    if candid_ptx:
        labels_to_use = ["Chest Tube", "Pneumothorax"]
        dataset = CandidPtxDataset(labels_to_use=labels_to_use, root=data_dir)
        datamodule = CandidPtxDataModule(
            dataset=dataset, batch_size=1, train_val_test_split=CandidPtxSplits.SPLITS
        )
        test_set: DatasetSubset = datamodule.test_dataloader().dataset
        dataset.image_paths = [dataset.image_paths[index] for index in test_set.indices]

        columns = [*labels_to_use, "image_id"]
        print(f"columns = {columns}")

        gt = []
        for img_path in dataset.image_paths:
            img_id = dataset.get_image_id(Path(img_path))
            labels = dataset.get_label_tensor(img_id).tolist()

            labels.append(img_id)
            gt.append(labels)

        ground_truth = pd.DataFrame(data=gt, columns=columns)

    else:
        ground_truth = PtxCohortsCiipDataset(root=data_dir).get_labels()

    data_points = list_of_datapoints(
        predictions=predictions, ground_truth=ground_truth, model_names=model_names
    )

    perform_analyses(
        data=data_points,
        filters_down=PTX_NEGATIVE_FILTERS,
        filters_right=PTX_POSITIVE_FILTERS,
        save_to=save_to,
        show=show_plot,
        candid_ptx=candid_ptx,
    )
