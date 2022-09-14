import numpy as np

from ptx_classification.calibration import plot_calibration


def test_plot_calibration_uniform() -> None:
    y_true = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    y_pred = np.array([0.0, 0.3, 0.3, 0.35, 0.5, 0.55, 0.7, 0.9])

    n_bins = 5
    plot_calibration(y_true, y_pred, n_bins=n_bins)


def test_plot_calibration_quantile() -> None:
    y_true = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    y_pred = np.array([0.0, 0.29, 0.3, 0.35, 0.5, 0.55, 0.7, 1.0])

    n_bins = len(y_true)
    plot_calibration(y_true, y_pred, n_bins=n_bins, strategy="quantile")
