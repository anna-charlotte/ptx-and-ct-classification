from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from ptx_classification.evaluation.evaluate import evaluate
from ptx_classification.models import MultiLabelModel
from ptx_classification.utils import (
    RANDOM_SEED,
    REPO_ROOT_DIR,
    get_cache_dir,
    get_data_dir,
    set_random_seeds,
)

random_seed = RANDOM_SEED
set_random_seeds(seed=random_seed)
if torch.cuda.is_available():
    accelerator = "gpu"
    num_devices = 1
else:
    accelerator = "cpu"
    num_devices = 1

root_dir = get_data_dir()
cache_dir = get_cache_dir()

# load model
model_dir = (
    REPO_ROOT_DIR
    / "ray_results"
    / "training_function_2022-05-18_14-03-07"
    / "training_function_7aa10_00005_5_labels_to_use=['Chest Tube'],model={'model_class': 'ResNet18Model', 'max_epochs': 100, 'lr': 5e-05_2022-05-19_00-36-10"
)

model_file = model_dir / "checkpoint_best_model" / "epoch=21-step=18524.ckpt"
model = MultiLabelModel.load_from_checkpoint(checkpoint_path=str(model_file))
transform = model.transform
labels = model.class_labels


dir_ct_pos = Path(
    "/Users/x/Desktop/bioinformatik/thesis/data_samples/chexpert_sample/ct_positive_large"
)
dir_ct_neg = Path(
    "/Users/x/Desktop/bioinformatik/thesis/data_samples/chexpert_sample/ct_negative_large"
)

y_true = []
y_pred = []
for img_path in dir_ct_pos.glob("*"):
    if ".DS_Store" not in str(img_path):
        print(img_path)
        y_true.append([1.0])

        img = Image.open(img_path)
        pixels = np.array(img).astype(np.float32)
        print(f"pixels.shape = {pixels.shape}")

        img = torch.from_numpy(np.array([[pixels, pixels, pixels]]))
        print(f"img = {img}")
        y = model.predict(img)
        y_pred.append(y.tolist()[0])
        plt.imshow(np.transpose(img.numpy()[0], axes=[1, 2, 0]).astype(np.uint8))
        plt.xlabel(f"gt = {1.0}\npred = {y.tolist()[0]}")
        plt.show()

for img_path in dir_ct_neg.glob("*"):
    if ".DS_Store" not in str(img_path):
        print(img_path)
        y_true.append([0.0])

        img = Image.open(img_path)
        pixels = np.array(img).astype(np.float32)

        img = torch.from_numpy(np.array([[pixels, pixels, pixels]]))
        y = model.predict(img)
        y_pred.append(y.tolist()[0])
        plt.imshow(np.transpose(img.numpy()[0], axes=[1, 2, 0]).astype(np.uint8))
        plt.xlabel(f"gt = {0.0}\npred = {y.tolist()[0]}")
        plt.show()


print(f"y_true = {y_true}")
print(f"y_pred = {y_pred}")

assert len(y_true) == len(y_pred)

y_true = torch.tensor(y_true)
y_pred = torch.tensor(y_pred)

evaluate(np.array(y_true), np.array(y_pred), labels=["Chest Tube"], dataset_name="CheXpert")
