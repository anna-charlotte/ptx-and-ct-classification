import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from ptx_classification.transforms import cast_tensor_to_uint8
from ptx_classification.utils import resize


class XrayDataset(ABC, Dataset):
    """Dataset Interface for all datasets used in this project."""

    image_paths: List[str]
    class_labels: List[str]

    @abstractmethod
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], str]:
        """
        Args:
            item:
                Integer, index of image.
        Returns:
            img:    torch.tensor of shape (3, H, W) of the given image.
            label:  torch.tensor of shape (len(self.LABELS)) consisting of 0. and/or 1.
            class labels:   List of the datasets class labels
            dataset name:   The datasets name of type string
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns:
            Integer, number of images included in the dataset.
        """
        pass

    @abstractmethod
    def load_image(self, img_path: Path, rgb: bool = True) -> torch.Tensor:
        """
        Returns:
            torch.tensor of image with the given path.
        """
        pass

    @abstractmethod
    def get_labels(self) -> pd.DataFrame:
        """
        Returns:
            Pandas DataFrame, that includes the labels for all images.
        """
        pass

    def get_image_id(self, img_path: Path) -> str:
        """

        Args:
            img_path:
                Path of a given image.
        Returns:
                The image identifier, for example the file name.
        """
        pass


# Dataset from ciip_dataset
class CiipDataset(ABC):
    data_path: Path

    def __init__(self, root=None, cache_dir=None, verbose=False):
        if root is None:
            root = "/mnt/cephstorage/share-all"

        self.root = Path(root)
        if not self.root.is_dir():
            print(f"self.root = {self.root}")
            raise FileNotFoundError

        if cache_dir is not None:
            cache_dir = Path(cache_dir).expanduser()
            if cache_dir.is_dir():
                self.cache_dir = cache_dir
            else:
                raise FileNotFoundError(f"{cache_dir} does not exist.")

        self.verbose = verbose

    def _cache(self, source):
        """Cache the source file if a cache directory is available.

        Parameters
        ----------
        source : str, Path
            Source of the file.
        """

        if self.cache_dir is not None:
            source = Path(source)
            destination = self.cache_dir / source.relative_to(self.root)
            if destination.exists():
                return destination

            command = f"mkdir -p {str(destination.parent)}"
            subprocess.run(command.split())
            command = f"rsync -avhW --progress {str(source)} {str(destination)}"

            if self.verbose:
                print(f"Caching {source} in {self.cache_dir}..")
                print(subprocess.check_output(command.split()))
                print(f"Copied {source} to {destination}.")
            else:
                subprocess.run(command.split(), stdout=subprocess.DEVNULL)
            return destination
        else:
            return source


class DatasetSubset(Dataset):
    def __init__(
        self, dataset: Dataset, indices: List[int], ground_truth_cut_off: Optional[float] = None
    ) -> None:
        """

        Args:
            dataset:
                Dataset.
            indices:
                List of Integer.
            ground_truth_cut_off:
                Float, in case the groundtruth labels are probabilities instead of
                0. or 1. we need cut off to evaluate performance with the AUC score.
        """
        self.dataset = dataset
        self.indices = indices
        self.gt_cut_off = ground_truth_cut_off

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], str]:
        index = self.indices[item]
        if self.gt_cut_off is None:
            return self.dataset[index]
        else:
            img, label, self.class_labels, dataset_name = self.dataset[index]
            label_with_cut_off = (label > self.gt_cut_off).float()
            return img, label_with_cut_off, self.class_labels, dataset_name

    def __len__(self) -> int:
        return len(self.indices)


class AugmentationDataset(Dataset):
    def __init__(self, dataset: Dataset, data_aug_transform: Optional[Callable]) -> None:
        self.dataset = dataset
        self.data_aug_transform = data_aug_transform

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], str]:
        x, y, labels, dataset_name = self.dataset[item]
        if self.data_aug_transform is not None:
            x = self.data_aug_transform(x)
        return x, y, labels, dataset_name


class ResizedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: List[int],
        new_size,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.new_size = new_size

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], str]:
        index = self.indices[item]
        x, y, labels, dataset_name = self.dataset[index]
        resized_imgs = resize(x, new_size=self.new_size)
        return cast_tensor_to_uint8(resized_imgs), y, labels, dataset_name

    def __len__(self) -> int:
        return len(self.indices)
