from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ChestTubeDataFrame:
    CT = "Chest Tube"
    COL_NAMES = ["image_id", CT]
    df: pd.DataFrame

    def __post_init__(self) -> None:
        for col in self.df.columns:
            assert (
                col in self.COL_NAMES
            ), f"self.df.columns = {self.df.columns}, self.COL_NAMES = {self.COL_NAMES}"


def read_in_csv_as_chest_tube_data_frame(path: Path) -> ChestTubeDataFrame:
    df = pd.read_csv(path, index_col=[0])
    return ChestTubeDataFrame(df=df)
