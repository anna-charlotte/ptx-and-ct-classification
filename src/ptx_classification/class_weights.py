from collections import Counter
from itertools import product
from typing import Callable, Dict, List, Literal, Tuple, Union

import numpy as np
import torch

from ptx_classification.data.datasets import DatasetSubset


def compute_class_weights_from_dataset(
    dataset: DatasetSubset,
    mode: Literal["none", "inverse", "sqrt_inverse"],
    based_on_class_label: str,
) -> np.ndarray:

    if mode == "none":
        return np.ones(2)
    elif mode == "inverse" or mode == "sqrt_inverse":
        all_targets = []

        for index in range(len(dataset.indices)):
            batch = dataset[index]
            class_labels: List[str] = batch[2]
            assert (
                based_on_class_label in class_labels
            ), f"Given class label is not in class_labels: {based_on_class_label}"
            index_cl = class_labels.index(based_on_class_label)
            target: torch.Tensor = batch[1][index_cl]
            all_targets.append(target.tolist())

        class_weights = compute_class_weights(targets=np.array(all_targets))

        if mode == "inverse":
            class_weights = np.array([cw * (1 / max(class_weights)) for cw in class_weights])
            return class_weights
        elif mode == "sqrt_inverse":
            class_weights = np.sqrt(class_weights)
            class_weights = np.array([cw * (1 / max(class_weights)) for cw in class_weights])
            return class_weights
    else:
        raise ValueError(f"Class weights do not exist for mode: {mode}")


def compute_sample_weights_from_dataset(
    dataset: DatasetSubset,
    based_on_class_label: List[str],
    class_weights: Tuple[float, float] = None,
) -> np.ndarray:
    if class_weights is None:
        return np.ones(shape=len(dataset))

    target2weight = dict()
    for i, item in enumerate(product([0, 1], repeat=len(based_on_class_label))):
        target2weight[item] = class_weights[i]

    print(f"class_weights = {class_weights}")
    sample_weights = []
    for index in range(len(dataset.indices)):
        batch = dataset[index]
        class_labels: List[str] = batch[2]
        for cl in based_on_class_label:
            assert (
                cl in class_labels
            ), f"Given class label is not in class_labels: {based_on_class_label}"
        cl_indices = [class_labels.index(cl) for cl in based_on_class_label]

        target = tuple(int(batch[1][index].item()) for index in cl_indices)
        sample_weights.append(target2weight[target])

    print(f"len(sample_weights) = {len(sample_weights)}")
    assert len(sample_weights) == len(
        dataset.indices
    ), f"{len(sample_weights)} != {len(dataset.indices)}"
    return np.array(sample_weights)


def compute_sample_weights(targets: np.ndarray) -> np.ndarray:
    """
    Computes class weights and returns an array of class weight for each label in labels.

     Example:
        Input:  [0.0, 1.0, 0.0, 0.0, 0.0]
        Output: np.array([0.25, 1, 0.25, 0.25, 0.25])

    """
    class_weights = compute_class_weights(targets=targets)
    class_labels: list = np.unique(targets).tolist()
    sample_weights = np.array([class_weights[class_labels.index(x)] for x in targets])

    assert len(targets) == len(
        sample_weights
    ), f"len(targets) = {len(targets)}, len(sample_weights) = {len(sample_weights)}"
    return sample_weights


def compute_class_weights(targets: np.ndarray) -> np.ndarray:
    """
    Computes class weights.

    Example:
        Input:  [0.0, 1.0, 0.0, 0.0, 0.0]
        Output: np.array([0.25, 1.0])

    """
    class_counts = compute_class_counts(targets)
    class_counts = np.array([1 / count for count in class_counts])
    return class_counts


def compute_class_counts(targets: np.ndarray) -> np.ndarray:
    """
    Computes the counts for each label and returns them in alphabetical/sorted order according to
    the targets elements.

    Example:
        Input: [0.0, 1.0, 0.0, 0.0, 0.0]
        Output: np.array([4, 1])

    """
    counts = Counter(targets)
    return np.array([counts[key] for key in sorted(counts.keys())])


ClassName = str


def make_sample_weight_function(
    class_weights_mode: Literal["none", "inverse", "sqrt_inverse"],
    class_name: str = "",
    dataset_subset: DatasetSubset = None,
) -> Callable[[Dict[ClassName, bool]], float]:
    """
    Creates a function that computes a weight for a single sample.
    """
    class_weights = compute_class_weights_from_dataset(
        dataset=dataset_subset, mode=class_weights_mode, based_on_class_label=class_name
    )
    print(f"class_weights = {class_weights}")

    def sample_weight(labels: Dict[ClassName, bool]) -> float:
        if class_weights_mode == "none":
            return 1.0
        else:
            return class_weights[labels[class_name]]

    return sample_weight
