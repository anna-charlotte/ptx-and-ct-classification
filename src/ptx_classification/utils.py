import json
import os
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import psutil
import pydicom
import torch
from PIL import Image
from ray.tune import trial
from pydicom.pixel_data_handlers.util import apply_voi_lut

REPO_ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
RAY_RESULTS_DIR = REPO_ROOT_DIR / "ray_results"
SPLITS_DIR = REPO_ROOT_DIR / "train_val_test_splits"

RANDOM_SEED = 42


def set_random_seeds(seed: int = 42) -> None:
    np.random.seed(seed=seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def current_process_memory_usage_in_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1_000_000


def tensor_to_pil(t: torch.Tensor) -> Image:
    np_img = np.transpose(t.cpu().detach().numpy(), axes=([1, 2, 0])).astype("uint8")
    pil_img = Image.fromarray(np_img, "RGB")
    return pil_img


def pil_to_tensor(pil_img: Image) -> torch.Tensor:
    pixels = np.array(pil_img)
    pixels_rgb = np.array([pixels, pixels, pixels])
    img_tensor = torch.from_numpy(pixels_rgb)
    return img_tensor


def save_image(image: torch.Tensor, file: Path) -> None:
    pil_image = tensor_to_pil(image)
    pil_image.save(file)


def get_commit_id() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_changes_since_last_commit() -> str:
    return subprocess.check_output(["git", "diff"]).decode("ascii").strip()


def get_commit_information() -> Dict[str, str]:
    info = {"commit id": get_commit_id(), "git diff": get_changes_since_last_commit()}
    return info


def save_commit_information(filename: Path) -> None:
    commit_info = get_commit_information()
    with open(filename, "w") as file:
        json.dump(commit_info, file)
    print(f"Saved commit information to {filename}")


def get_date_and_time(include_ms: bool = True) -> str:
    if include_ms:
        return str(datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "_")
    else:
        return str(datetime.now()).replace(" ", "_").replace(":", "-").split(".")[0]


def get_data_dir() -> Path:
    if "BACHELOR_THESIS_DATA_DIR" in os.environ.keys():
        return Path(os.environ["BACHELOR_THESIS_DATA_DIR"])
    else:
        raise ValueError(
            "The data directory has not been set yet. Please add the path "
            "to your data directory to your .bash_profile file."
        )


def get_cache_dir() -> Path:
    if "BACHELOR_THESIS_CACHE_DIR" in os.environ.keys():
        return Path(os.environ["BACHELOR_THESIS_CACHE_DIR"])
    else:
        raise ValueError(
            "The cache directory has not been set yet. Please add the path"
            " to your cache directory to your .bash_profile file."
        )


def trial_dirname_creator(trial: trial.Trial) -> str:
    tag = trial.experiment_tag
    if "chest_tube_labels=_" in tag:
        tag = re.sub("chest_tube_labels=.*,", "chest_tube_labels=True,", tag)
    trial_dirname = (
        f"{trial.trainable_name}_{trial.trial_id}_{tag}".replace(": ", ",")
        .replace(" ", "")
        .replace("'", "")
        .replace("[", "")
        .replace("]", "")
        .replace("{", "")
        .replace("}", "")
        .replace(",", "_")
    )
    max_len_for_trial_dirname = 260 - len(trial.local_dir)
    return trial_dirname[:max_len_for_trial_dirname]


def load_json(json_file: Path) -> dict:
    file = open(json_file)
    json_dict = json.load(file)
    return json_dict


def intersection(list_1: list, list_2: list, sort: bool) -> list:
    list_3 = [value for value in list_1 if value in list_2]
    if sort:
        return sorted(list_3)
    else:
        return list_3


def unique(input: list) -> list:
    return list(set(input))


def resize(images: torch.Tensor, new_size: Tuple[int, int]) -> torch.Tensor:
    dims = len(images.size())
    if dims == 3:
        images = images[None, :]
    resized_images = []
    for img in images:
        np_img = np.transpose(img.cpu().detach().numpy(), axes=([1, 2, 0])).astype("uint8")
        pil_img = Image.fromarray(np_img, "RGB")
        resized_img = pil_img.resize(new_size, resample=Image.Resampling.BILINEAR)
        np_img = np.transpose(np.array(resized_img), axes=([2, 0, 1]))
        if len(resized_images) == 0:
            resized_images = np.array(np_img)
        else:
            resized_images = np.append(resized_images, np.expand_dims(np_img, axis=0), axis=0)

    resized_tensor = torch.from_numpy(resized_images).float()
    if dims == 3:
        return torch.squeeze(resized_tensor, 0)
    else:
        return resized_tensor


def dicom_to_array(path, voi_lut=True, fix_monochrome=True, rgb=False) -> np.ndarray:
    """
    Convert a DICOM to np.array.

    Source
    ------
    ciip_dataloader,
    Kaggle user: raddar
    https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    """
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    if rgb:
        data = np.stack([data, data, data]).transpose(1, 2, 0)
    return data