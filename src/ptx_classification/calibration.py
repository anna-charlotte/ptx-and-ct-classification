import os
from pathlib import Path
from typing import Tuple

from sklearn.preprocessing import label_binarize
from sklearn.utils import check_consistent_length, column_or_1d

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pytorch_lightning as pl
import torch
from bokeh.layouts import gridplot
from bokeh.plotting import Figure, figure, output_file, save

from ptx_classification.data.candid_ptx.candid_ptx import CandidPtxDataModule, CandidPtxDataset
from ptx_classification.models import MultiLabelModel
from ptx_classification.utils import (
    RANDOM_SEED,
    REPO_ROOT_DIR,
    get_cache_dir,
    get_data_dir,
    get_date_and_time,
    load_json,
    set_random_seeds,
)


def apply_style(p: Figure) -> None:
    p.toolbar.logo = None
    p.background_fill_color = "#fafafa"


def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int,
    title: str = "Calibration",
    strategy: str = "uniform",
    add_perfect_calibration: bool = True,
) -> None:
    print("In plot_calibration() ...")
    print(f"strategy = {strategy}")
    prob_true, prob_pred, left_edges, right_edges = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy=strategy
    )
    print(f"prob_true = {prob_true}")
    print(f"prob_pred = {prob_pred}")

    height = 550
    width = 600

    p = figure(
        title=title,
        height=height,
        width=width,
    )
    apply_style(p)

    p.quad(
        top=prob_true,
        bottom=0,
        left=left_edges,
        right=right_edges,
        fill_color="navy",
        line_color="white",
        alpha=0.65,
        legend_label="ResNet18",
    )

    if add_perfect_calibration:
        bin_width = 1.0 / n_bins
        top = [l + (bin_width / 2) for l in left_edges]
        orange = "rgb(232, 97, 41)"
        orange = "rgb(232, 97, 41)"
        p.quad(
            top=top,
            bottom=0,
            left=left_edges,
            right=right_edges,
            fill_color=orange,
            fill_alpha=0.2,
            line_color=orange,
            line_width=2,
            legend_label="Perfectly Calibrated Model",
            hatch_pattern="right_diagonal_line",
            hatch_color=orange,
            hatch_alpha=0.6,
            hatch_scale=10,
            hatch_weight=1,
        )

    p.y_range.start = 0
    p.x_range.start = 0
    p.x_range.end = 1

    p.yaxis.axis_label = "Accuracy"
    p.xaxis.axis_label = "Confidence"
    p.title.text_font_size = "14pt"

    p.legend.location = "bottom_right"
    p.legend.title = "Models"

    p.legend.border_line_width = 3
    p.legend.border_line_alpha = 1.0
    p.legend.background_fill_alpha = 0.8

    if strategy == "uniform":
        p_counts = figure(
            x_range=p.x_range,
            height=300,
            width=width,
        )
        apply_style(p_counts)

        lengths = [0] * n_bins

        for i in range(n_bins):
            for pred in y_pred:
                lower_range = i / n_bins
                upper_range = (i + 1) / n_bins
                if pred >= lower_range and pred < upper_range:
                    lengths[i] = lengths[i] + 1
        print(f"lengths = {lengths}")

        p_counts.quad(
            top=lengths,
            bottom=0,
            left=left_edges,
            right=right_edges,
            fill_color="navy",
            line_color="white",
            line_width=1.5,
            alpha=0.65,
        )

        p_counts.xaxis.axis_label = "Confidence"
        p_counts.yaxis.axis_label = "Number of predictions"
        p_counts.ygrid.grid_line_color = "#fafafa"
        p_counts.toolbar.logo = None

        layout = gridplot([[p], [p_counts]], merge_tools=False)
    else:
        layout = gridplot([[p]], merge_tools=False)

    save_to = (
        REPO_ROOT_DIR
        / "plots"
        / "calibration_plots"
        / f"calibration_plot_{strategy}_n_bins_{n_bins}_{get_date_and_time()}.html"
    )
    print(f"Saving calibration plot to: {save_to}... ")
    output_file(save_to)
    save(layout)


def calibration_curve(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 5, strategy: str = "uniform"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute true and predicted probabilities for a calibration curve.
    Copied and modified from https://github.com/scikit-learn/scikit-learn/blob/80598905e/sklearn/calibration.py#L873
    Args:
        y_true:
            1D array of zeros and ones for the ground truth.
        y_pred:
            1D array of probability estimates for the positive class.
        n_bins:
            Number of bins to discretize the [0, 1] interval. A bigger number
            requires more data. Bins with no samples (i.e. without
            corresponding values in `y_prob`) will not be returned, thus the
            returned arrays may have less than `n_bins` values.
        strategy:
            {'uniform', 'quantile'}, default='uniform'
            Strategy used to define the widths of the bins.
            uniform: The bins have identical widths.
            quantile: The bins have the same number of samples and depend on `y_prob`.
    """

    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y_true, y_pred)

    assert y_true.shape == y_pred.shape
    assert set(y_true).issubset({0, 1}) or set(y_true).issubset({False, True})
    assert min(y_pred) >= 0.0, min(y_pred)
    assert max(y_pred) <= 1.0, max(y_pred)

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. Provided labels %s." % labels)
    y_true = label_binarize(y_true, classes=labels)[:, 0]

    print(f"strategy = {strategy}")
    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        print(f"quantiles = {quantiles}")
        bins = np.percentile(y_pred, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy " "must be either 'quantile' or 'uniform'."
        )
    bin_edges_left = bins[:-1]
    bin_edges_right = bins[1:]

    bin_ids = np.digitize(y_pred, bins) - 1
    bin_sums = np.bincount(bin_ids, weights=y_pred, minlength=n_bins)
    bin_true = np.bincount(bin_ids, weights=y_true, minlength=n_bins)
    bin_total = np.bincount(bin_ids, minlength=n_bins)

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    assert len(prob_true) == n_bins, len(prob_true)
    assert len(prob_pred) == n_bins

    return prob_true, prob_pred, bin_edges_left, bin_edges_right


def main() -> None:
    random_seed = RANDOM_SEED
    set_random_seeds(seed=random_seed)
    if torch.cuda.is_available():
        accelerator = "gpu"
        num_devices = 1
    else:
        accelerator = "cpu"
        num_devices = 1

    # load model
    model_file = (
        REPO_ROOT_DIR
        / "ray_results"
        / "training_function_2022-05-18_14-03-07"
        / "training_function_7aa10_00005_5_labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs': 100, 'lr': 5e-05_2022-05-19_00-36-10"
        / "checkpoint_best_model"
        / "epoch=21-step=18524.ckpt"
    )
    model = MultiLabelModel.load_from_checkpoint(checkpoint_path=str(model_file))
    transform = model.transform
    labels = model.class_labels

    # read in config
    json_file = "/".join(str(model_file).split("/")[:-2]) + "/params.json"
    print(f"Reading in config from {json_file}")
    config = load_json(Path(json_file))
    batch_size = config["data_module"]["batch_size"]
    n_train = config["data_module"]["train_size"]
    n_val = config["data_module"]["val_size"]
    n_test = config["data_module"]["test_size"]

    root_dir = get_data_dir()
    cache_dir = get_cache_dir()

    candid_ptx = CandidPtxDataModule(
        dataset=CandidPtxDataset(
            root=root_dir,
            cache_dir=cache_dir,
            # transform=transform,
            labels_to_use=labels,
        ),
        batch_size=batch_size,
        train_val_test_split=(n_train, n_val, n_test),
    )

    # trainer
    trainer = pl.Trainer(
        devices=num_devices,
        accelerator=accelerator,
        max_epochs=100,
        enable_progress_bar=False,
        num_sanity_val_steps=-1,
        # overfit_batches=10,
    )
    set = "test"
    if set == "test":
        dl = candid_ptx.test_dataloader()
    else:
        dl = candid_ptx.val_dataloader()
    print("Trainer.test on candid")
    trainer.test(model=model, dataloaders=dl)

    results = model.results_test_end
    y_target = results["y_true_test"]
    y_pred = results["y_pred_test"]

    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_target.cpu().detach().numpy()

    n_bins = 10
    strategy = "uniform"

    plot_calibration(
        y_true=y_true.flatten(),
        y_pred=y_pred.flatten(),
        n_bins=n_bins,
        strategy=strategy,
        title=f"Calibration (on {set} set, with {strategy} strategy)",
    )


if __name__ == "__main__":
    main()
