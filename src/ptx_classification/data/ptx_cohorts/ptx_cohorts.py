import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from PIL import Image

from ptx_classification.data.datasets import XrayDataset, CiipDataset
from ptx_classification.utils import dicom_to_array


class PtxCohortsCiipDataset(CiipDataset, XrayDataset):
    """
    This is the PneuStudien Kohorte dataset.

    """

    def __init__(
        self,
        root: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        verbose: bool = False,
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(root, cache_dir=cache_dir, verbose=verbose)
        self.resize = resize
        self.data_dir = self.root / "chestxray-data/PneuStudien_Kohorte/"
        self.cache_dir = cache_dir

        self.labels_path = self._cache(self.data_dir / "PTX_Kohorte_Labels_Rueckel.xlsx")
        self.image_dir = self.data_dir
        self.image_paths = [str(path) for path in list(self.image_dir.glob("*.dcm"))]

        self.df_labels = self._read_in_labels()

    def _read_in_labels(self) -> pd.DataFrame:
        print("Reading in labels ...")
        df = pd.read_excel(self.labels_path, engine="openpyxl", skipfooter=1)
        df = df.rename(
            columns={
                "Bild #": "image_id",
                "PNEU LINKS": "Pneumothorax left",
                "PNEU RECHTS": "Pneumothorax right",
            }
        )
        df["image_id"] = df.apply(lambda row: f"PneuStudie{row['image_id']}.dcm", axis=1)
        df["Path"] = df.apply(lambda row: self.image_dir / row["image_id"], axis=1)
        df["Chest Tube"] = df.apply(
            lambda row: 0.0 if row["TD_links"] == "Nein" and row["TD_rechts"] == "Nein" else 1.0,
            axis=1,
        )
        for side in ["left", "right"]:
            df[f"Pneumothorax {side}"] = df.apply(
                lambda row: f"{side}: < 1cm"
                if "< 1cm" in row[f"Pneumothorax {side}"]
                else (
                    f"{side}: 1-2 cm"
                    if "1-2cm" in row[f"Pneumothorax {side}"]
                    else (
                        f"{side}: > 2cm"
                        if "> 2cm" in row[f"Pneumothorax {side}"]
                        else f"{side}: None"
                    )
                ),
                axis=1,
            )
        df["Pneumothorax size"] = df.apply(
            lambda row: f"{row['Pneumothorax left']}, {row['Pneumothorax right']}", axis=1
        )
        df["Pneumothorax lateral"] = df.apply(
            lambda row: "None"
            if row["Pneumothorax size"] == "left: None, right: None"
            else (
                "unilateral"
                if len(re.findall("None", row["Pneumothorax size"])) == 1
                else "bilateral"
            ),
            axis=1,
        )
        df["Pneumothorax"] = df.apply(
            lambda row: 0.0 if row["Pneumothorax lateral"] == "None" else 1.0, axis=1
        )
        print(f"df = {df}")
        print(f"df.iloc[6404] = {df.iloc[6404]}")
        print(
            f"len(df[df['Pneumothorax lateral'] == None]) = {len(df[df['Pneumothorax lateral'] == 'None'])}"
        )
        print(
            f"len(df[df['Pneumothorax lateral'] == unilateral]) = {len(df[df['Pneumothorax lateral'] == 'unilateral'])}"
        )
        print(
            f"len(df[df['Pneumothorax lateral'] == bilateral]) = {len(df[df['Pneumothorax lateral'] == 'bilateral'])}"
        )

        print(f"len(df[df['Chest Tube'] == 0.0]) = {len(df[df['Chest Tube'] == 0.0])}")
        print(f"len(df[df['Chest Tube'] == 1.0]) = {len(df[df['Chest Tube'] == 1.0])}")

        df = df.drop(
            [
                "AI-Score (07_2021)",
                "Pneumothorax left",
                "Pneumothorax right",
                "TD_rechts",
                "TD_links",
            ],
            axis=1,
        )
        print("Finished reading in labels ...")
        return df

    def load_image(self, img_path: Path, rgb: bool = True) -> torch.Tensor:
        img = dicom_to_array(path=self._cache(img_path), rgb=False).astype(np.uint8)
        pil_img = Image.fromarray(img)
        if self.resize is not None:
            pil_img = pil_img.resize(self.resize, resample=Image.Resampling.BILINEAR)
        img = np.array(pil_img)

        if rgb:
            return torch.from_numpy(np.array([img, img, img]))
        else:
            return torch.from_numpy(img)

    def get_label_tensor(self, img_path) -> torch.Tensor:
        df_train = self.get_labels()
        row = df_train.loc[df_train["Path"] == img_path][self.class_labels].values
        row = np.reshape(row, (len(self.class_labels)))
        return torch.from_numpy(row)

    def get_labels(self) -> pd.DataFrame:
        return self.df_labels

    @staticmethod
    def get_patient_id(img_path: Path) -> int:
        patient_id = int(str(img_path).split("/")[-1].split("_")[0])
        return patient_id

    def get_image_id(self, img_path: Path) -> str:
        image_id = str(img_path).split("/")[-1]
        return image_id

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], str]:
        img_path = self.image_paths[item]
        img = self.load_image(Path(img_path), rgb=True)
        label = self.get_label_tensor(img_path=img_path)
        dataset_name = self.__class__.__name__.replace("Dataset", "")
        return img, label, self.class_labels, dataset_name

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_image_ids_of_bilateral_ptx_images(self):
        df = self.get_labels()
