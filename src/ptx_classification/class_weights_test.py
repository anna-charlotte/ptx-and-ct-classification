import numpy as np

from ptx_classification.class_weights import (
    compute_class_counts,
    compute_class_weights,
    compute_class_weights_from_dataset,
    compute_sample_weights,
)
from ptx_classification.data.datasets import DatasetSubset
from ptx_classification.models_test import MockDataModule, MockDataset


def test_compute_class_weights_from_data_loader() -> None:
    n_images = 6
    n_labels = 2
    dataset = MockDataset(num_images=n_images, img_h=2, img_w=2, num_labels=n_labels)
    datamodule = MockDataModule(dataset=dataset, batch_size=2)
    class_labels = dataset.class_labels
    train_dataset: DatasetSubset = datamodule.train_dataloader().dataset
    class_weights = compute_class_weights_from_dataset(
        dataset=train_dataset, mode="inverse", based_on_class_label=class_labels[0]
    )
    print(f"class_weights = {class_weights}")
    assert len(class_weights) == n_labels
    assert sum(class_weights) <= n_labels


def test_compute_sample_weights() -> None:
    y = np.array([0.0, 1.0, 2.0, 0.0])
    sample_weights = compute_sample_weights(targets=y)
    np.testing.assert_allclose(sample_weights, np.array([0.5, 1, 1, 0.5]))


def test_compute_class_weights() -> None:
    y = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    class_weights = compute_class_weights(targets=y)
    np.testing.assert_allclose(class_weights, np.array([0.25, 1.0]))


def test_compute_class_counts() -> None:
    y = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    class_counts = compute_class_counts(targets=y)
    np.testing.assert_allclose(class_counts, np.array([4, 1]))
