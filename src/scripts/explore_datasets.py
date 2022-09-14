from typing import List

import numpy as np
import pandas as pd

from ptx_classification.data.candid_ptx.candid_ptx import (
    CandidPtxDataModule,
    CandidPtxDataset,
    CandidPtxLabels,
    CandidPtxSplits,
)
from ptx_classification.data.chestxray14.chestxray14 import (
    ChestXray14DataModule,
    ChestXray14Dataset,
    ChestXray14Labels,
    ChestXray14Splits,
)
from ptx_classification.data.explore_data import plot_venn_diagram_for_candid_ptx_datamodule
from ptx_classification.data.padchest.padchest import PadChestDataset, PadChestProjectionLabels
from ptx_classification.utils import get_data_dir


def print_candid_ptx_exploration() -> None:
    data_dir = get_data_dir()

    dataset = CandidPtxDataset(root=data_dir, labels_to_use=CandidPtxLabels.LABELS)
    datamodule = CandidPtxDataModule(
        dataset, batch_size=1, train_val_test_split=CandidPtxSplits.SPLITS
    )

    patient_ids = np.unique(dataset.get_labels().loc[:, "patient_id"].values)
    print(f"len(patient_ids) = {len(patient_ids)}\n")

    df = dataset.get_labels()
    ptx_pos_imgs = df[df["Pneumothorax"] == 1.0]
    patient_ids_ptx_pos = np.unique(ptx_pos_imgs.loc[:, "patient_id"].values)
    print(f"len(ptx_pos_imgs) = {len(ptx_pos_imgs)}")
    print(f"len(patient_ids_ptx_pos) = {len(patient_ids_ptx_pos)}\n")

    ct_pos_imgs = df[df["Chest Tube"] == 1.0]
    patient_ids_ct_pos = np.unique(ct_pos_imgs.loc[:, "patient_id"].values)
    print(f"len(ct_pos_imgs) = {len(ct_pos_imgs)}")
    print(f"len(patient_ids_ct_pos) = {len(patient_ids_ct_pos)}\n")

    rf_pos_imgs = df[df["Rib Fracture"] == 1.0]
    patient_ids_rf_pos = np.unique(rf_pos_imgs.loc[:, "patient_id"].values)
    print(f"len(rf_pos_imgs) = {len(rf_pos_imgs)}")
    print(f"len(patient_ids_rf_pos) = {len(patient_ids_rf_pos)}\n")

    plot_venn_diagram_for_candid_ptx_datamodule(datamodule)


def print_padchest_exploration() -> None:
    data_dir = get_data_dir()
    proj_labels = [
        PadChestProjectionLabels.LABELS,
        [
            PadChestProjectionLabels.AP,
            PadChestProjectionLabels.AP_h,
            PadChestProjectionLabels.PA,
        ],
    ]
    min_ages = [0, 16]
    for min_age in min_ages:
        for labels in proj_labels:
            print(f"\nmin_age = {min_age}")
            print(f"labels = {labels}")
            padchest = PadChestDataset(
                root=data_dir,
                projection_labels=labels,
                min_age=min_age,
            )
            df_labels = padchest.get_labels()
            print(f"len(df_labels) = {len(df_labels)}")
            ct_positive_imgs = df_labels[df_labels["Chest Tube"] == 1.0]
            print(f"len(ct_positive_imgs) = {len(ct_positive_imgs)}")


def print_chestxray14_exploration() -> None:
    data_dir = get_data_dir()
    labels = ChestXray14Labels.LABELS
    dataset = ChestXray14Dataset(root=data_dir, class_labels=labels)
    datamodule = ChestXray14DataModule(
        dataset, batch_size=1, train_val_split=(ChestXray14Splits.N_TRAIN, ChestXray14Splits.N_VAL)
    )
    df_labels = dataset.get_labels()
    patient_ids = np.unique(df_labels.loc[:, "patient_id"].values)
    print(f"len(patient_ids) = {len(patient_ids)}")

    for label in labels:
        pos = df_labels[df_labels[label] == 1.0]
        print(f"len({label}_pos) = {len(pos)}")

    print(f"df_labels.columns = {df_labels.columns}")
    df_labels = df_labels[labels]
    print(f"df_labels.columns = {df_labels.columns}")
    df_labels.drop(["No Finding"], axis=1, inplace=True)
    print(f"df_labels.columns = {df_labels.columns}")
    print(f"df_labels.shape = {df_labels.shape}")
    matrix = df_labels.corr()
    print(f"matrix.shape = {matrix.shape}")
    print(f"matrix = {matrix}")

    matrix[matrix < 0] = 0

    # matrix = matrix.multiply(100).astype(int)
    # Converting the DataFrame to a 2D List, as it is the required input format.
    matrix = matrix.values.tolist()

    from plotapi import Chord

    Chord.set_license(username="email", password="secret")

    matrix = [
        [10, 5, 6, 4, 7, 4],
        [5, 20, 5, 4, 6, 5],
        [6, 5, 30, 4, 5, 5],
        [4, 4, 4, 40, 5, 5],
        [7, 6, 5, 5, 50, 4],
        [4, 5, 5, 5, 4, 60],
    ]

    labels = ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Thriller"]
    print(f"len(matrix) = {len(matrix)}")
    print(f"len(labels) = {len(labels)}")
    # labels.remove("No Finding")
    Chord(matrix, labels, arc_numbers=True, padding=0.05).to_html()

    import holoviews as hv
    import pandas as pd
    from bokeh.sampledata.les_mis import data
    from holoviews import dim, opts

    hv.extension("bokeh")
    hv.output(size=200)

    links = pd.DataFrame(data["links"])
    print(f"links = {links}")
    print(f"links.shape = {links.shape}")

    nodes = hv.Dataset(pd.DataFrame(data["nodes"]), "index")
    print(f"data['nodes'] = {data['nodes']}")
    chord = hv.Chord((links, nodes)).select(value=(5, None))
    plot = chord.opts(
        opts.Chord(
            cmap="Category20",
            edge_cmap="Category20",
            edge_color=dim("source").str(),
            labels="name",
            node_color=dim("index").str(),
        )
    )
    hv.save(plot, "_out_hv.html", backend="bokeh")


def labels_to_matrix(data: pd.DataFrame) -> List[List[int]]:
    # TODO
    matrix = np.zeros(shape=(len(data.columns), len(data.columns)))

    data_values = data.values
    pos_indices = np.where(np.any(data_values == 1.0))
    print(f"data_values = {data_values}")
    print(f"pos_indices = {pos_indices}\n")
    print(f"data_values.nonzero() = {data_values.nonzero()}")
    print(f"np.transpose(data_values.nonzero()) = {np.transpose(data_values.nonzero())}")
    for i, row in enumerate(data_values):
        pos_indices = np.where(np.any(row == 1.0, axis=0))
        print(f"row = {row}")
        print(f"pos_indices = {pos_indices}")

    for i, row in data.iterrows():
        row = row.values
        # print(f"row = {row}")

        pos_indices = np.where(np.any(row == 1.0, axis=0))
        # print(f"pos_indices = {pos_indices}")


def main():
    print_candid_ptx_exploration()
    print_padchest_exploration()
    print_chestxray14_exploration()


if __name__ == "__main__":
    main()
