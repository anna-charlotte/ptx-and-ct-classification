import numpy as np
import torch

from ptx_classification.utils import (
    REPO_ROOT_DIR,
    current_process_memory_usage_in_mb,
    intersection,
    save_image,
)


def test_save_image() -> None:
    img = torch.randint(low=0, high=256, size=(3, 100, 200))
    save_image(image=img, file=REPO_ROOT_DIR / "images" / "test.png")


def test_intersection() -> None:
    list_1 = [1, 2, 3, 4, 5, 6, 7, 10, 20, 30, 40]
    list_2 = [10, 8, 9, 11, 5, 1, 0, 12, 13]
    np.testing.assert_array_equal(intersection(list_1, list_2, sort=True), [1, 5, 10])


def test_current_process_memory_usage_in_mb() -> None:
    memory_usage = current_process_memory_usage_in_mb()
    print(f"memory_usage = {memory_usage}")
    assert type(memory_usage) == float
