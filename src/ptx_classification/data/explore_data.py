import os
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from ptx_classification.data.candid_ptx.candid_ptx import (
    CandidPtxDataModule,
    CandidPtxDataset,
    CandidPtxLabels,
    CandidPtxSplits,
)
from ptx_classification.data.datasets import XrayDataset
from ptx_classification.utils import (
    RANDOM_SEED,
    REPO_ROOT_DIR,
    get_cache_dir,
    get_data_dir,
    get_date_and_time,
)


def plot_venn_diagram(y_target: pd.DataFrame) -> None:
    curr_time = get_date_and_time()
    csv_file = REPO_ROOT_DIR / "plots" / "venn_diagram" / f"{curr_time}.csv"
    out_file = REPO_ROOT_DIR / "plots" / "venn_diagram" / f"plot_venn_r_{curr_time}.svg"
    y_mod = y_target
    y_mod["img_id"] = y_mod.index

    pd.DataFrame.to_csv(y_mod, path_or_buf=csv_file, sep=";")
    r_script = REPO_ROOT_DIR / "scripts_r" / "plot_venn_diagram.R"
    subprocess.call(["Rscript", str(r_script), csv_file, out_file])
    os.remove(csv_file)
    print(f"Saving venn diagram plot to: {out_file}")


def plot_venn_diagram_for_candid_ptx_datamodule(datamodule: CandidPtxDataModule):
    dataset: XrayDataset = datamodule.dataset
    y_all = dataset.get_labels().loc[:, CandidPtxDataset.LABELS]
    plot_venn_diagram(y_target=y_all)

    dataloaders = [
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
        datamodule.train_dataloader(),
    ]

    for dataloader in dataloaders:
        y_pd = []
        dl = iter(dataloader)
        for i in range(len(dataloader)):
            x, y, _, _ = next(dl)
            y = y.numpy()
            if len(y_pd) == 0:
                y_pd = y
            else:
                y_pd = np.append(y_pd, y, axis=0)

        y_pd = pd.DataFrame(data=np.array(y_pd), columns=CandidPtxDataset.LABELS)
        plot_venn_diagram(y_target=y_pd)


def main() -> None:
    pl.seed_everything(RANDOM_SEED, workers=True)
    data_dir = get_data_dir()
    cache_dir = get_cache_dir()
    batch_size = 1

    datamodule = CandidPtxDataModule(
        dataset=CandidPtxDataset(
            root=data_dir, cache_dir=cache_dir, labels_to_use=CandidPtxLabels.LABELS
        ),
        batch_size=batch_size,
        train_val_test_split=CandidPtxSplits.SPLITS,
    )
    plot_venn_diagram_for_candid_ptx_datamodule(datamodule)


if __name__ == "__main__":
    main()
